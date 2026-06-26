from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:  # pragma: no cover - exercised in training env
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # pragma: no cover - keep light imports possible
    plt = None
    torch = None
    F = None
    nn = object
    DataLoader = Dataset = WeightedRandomSampler = object

from .descriptor_baselines import (
    DEFAULT_THRESHOLDS,
    PATH_COLUMN,
    apply_thresholds_to_df,
    build_neighbor_table,
    build_identity_holdout_split,
    dataframe_to_markdown_table,
    extract_embeddings,
    fuse_embedding_blocks,
    l2_normalize,
    load_descriptor_model,
    load_manifests,
    pick_best_thresholds,
    recall_at_k,
    run_threshold_sweep,
    summarize_cluster_metrics,
)
from .submission_baseline import _load_supervised_model_from_checkpoint
from .supervised_training import (
    ArcFaceHead,
    build_eval_transform,
    build_train_transform,
    collect_resource_snapshot,
    compute_feature_distillation_loss,
    compute_relation_distillation_loss,
    compute_view_pair_contrastive_loss,
    describe_transform,
    extract_student_embeddings,
    load_student_backbone,
    scale_learning_rate,
    seed_everything,
)


@dataclass(frozen=True)
class TeacherSourceSpec:
    kind: str
    value: str
    display_name: str


@dataclass(frozen=True)
class PseudoSeedBundle:
    target_df: pd.DataFrame
    anchor_threshold: float
    lower_threshold: float
    upper_threshold: float
    pseudo_seed_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    pair_table_df: pd.DataFrame
    teacher_anchor_pred_df: pd.DataFrame
    teacher_best_pred_df: pd.DataFrame
    teacher_anchor_metrics: dict[str, float]
    teacher_best_metrics: dict[str, float]


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("labeled_selftrain requires torch; run in the `wildfusion` environment.")


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for labeled_selftrain plots")


def parse_teacher_sources(descriptor_sources: list[str], checkpoint_sources: list[str]) -> list[TeacherSourceSpec]:
    specs: list[TeacherSourceSpec] = []
    for source in descriptor_sources:
        specs.append(TeacherSourceSpec(kind="descriptor", value=str(source), display_name=str(source)))
    for source in checkpoint_sources:
        path = Path(source).resolve()
        specs.append(TeacherSourceSpec(kind="checkpoint", value=str(path), display_name=path.stem))
    if not specs:
        raise ValueError("Need at least one teacher source")
    return specs


def attach_single_dataset_labels(fit_df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, dict[str, int]]:
    labeled = fit_df.copy().reset_index(drop=True)
    if labeled["dataset"].nunique() != 1 or str(labeled["dataset"].iloc[0]) != dataset:
        raise ValueError(f"Expected fit_df to contain only dataset={dataset}")
    identities = sorted(labeled["identity"].astype(str).unique().tolist())
    label_map = {identity: index for index, identity in enumerate(identities)}
    labeled["dataset_index"] = 0
    labeled["label_index"] = labeled["identity"].astype(str).map(label_map).astype(int)
    labeled["global_label_index"] = labeled["label_index"]
    labeled["identity_image_count_fit"] = labeled.groupby("identity")["image_id"].transform("size").astype(int)
    return labeled, label_map


def _teacher_component_name(spec: TeacherSourceSpec) -> str:
    if spec.kind == "descriptor":
        return str(spec.value)
    return Path(spec.value).stem


def _extract_checkpoint_embeddings(
    checkpoint_path: Path,
    df: pd.DataFrame,
    repo_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model, backbone_spec, config, _checkpoint = _load_supervised_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    embeddings = extract_student_embeddings(
        df=df,
        repo_root=repo_root,
        model=model,
        spec=backbone_spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return embeddings.astype(np.float32), config


def build_teacher_embeddings(
    *,
    repo_root: Path,
    fit_df: pd.DataFrame,
    target_df: pd.DataFrame,
    teacher_specs: list[TeacherSourceSpec],
    device: str,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    fit_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    component_rows: list[dict[str, object]] = []
    for spec in teacher_specs:
        if spec.kind == "descriptor":
            model, descriptor_spec = load_descriptor_model(spec.value, device=device)
            fit_embeddings = extract_embeddings(
                df=fit_df,
                repo_root=repo_root,
                model=model,
                spec=descriptor_spec,
                device=device,
                batch_size=descriptor_spec.default_batch_size,
                num_workers=num_workers,
            )
            target_embeddings = extract_embeddings(
                df=target_df,
                repo_root=repo_root,
                model=model,
                spec=descriptor_spec,
                device=device,
                batch_size=descriptor_spec.default_batch_size,
                num_workers=num_workers,
            )
            model_id = descriptor_spec.model_id
        elif spec.kind == "checkpoint":
            checkpoint_path = Path(spec.value)
            fit_embeddings, config = _extract_checkpoint_embeddings(
                checkpoint_path=checkpoint_path,
                df=fit_df,
                repo_root=repo_root,
                device=device,
                batch_size=32,
                num_workers=num_workers,
            )
            target_embeddings, _config = _extract_checkpoint_embeddings(
                checkpoint_path=checkpoint_path,
                df=target_df,
                repo_root=repo_root,
                device=device,
                batch_size=32,
                num_workers=num_workers,
            )
            model_id = str(config.get("student_model_id", checkpoint_path.name))
        else:  # pragma: no cover - guarded by parser
            raise ValueError(f"Unsupported teacher source kind: {spec.kind}")
        fit_blocks.append(fit_embeddings.astype(np.float32))
        target_blocks.append(target_embeddings.astype(np.float32))
        component_rows.append(
            {
                "teacher_source": _teacher_component_name(spec),
                "source_kind": spec.kind,
                "source_value": spec.value,
                "model_id": model_id,
                "fit_dim": int(fit_embeddings.shape[1]),
                "target_dim": int(target_embeddings.shape[1]),
            }
        )
        if device.startswith("cuda") and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    teacher_fit = fuse_embedding_blocks(fit_blocks, weights=[1.0] * len(fit_blocks)).astype(np.float32)
    teacher_target = fuse_embedding_blocks(target_blocks, weights=[1.0] * len(target_blocks)).astype(np.float32)
    return teacher_fit, teacher_target, pd.DataFrame(component_rows)


def _pair_label_matrix(labels: np.ndarray) -> np.ndarray:
    return labels[:, None] == labels[None, :]


def _cluster_members(labels: np.ndarray, index: int) -> set[int]:
    cluster_id = labels[index]
    return set(np.flatnonzero(labels == cluster_id).tolist())


def _mean_intra_cluster_similarity(score_matrix: np.ndarray, members: list[int]) -> float:
    if len(members) < 2:
        return 0.0
    block = score_matrix[np.ix_(members, members)].copy()
    upper = block[np.triu_indices_from(block, k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def build_stable_pseudo_seed_bundle(
    *,
    target_df: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    anchor_threshold: float,
    stability_delta: float,
    min_seed_cluster_size: int,
    max_seed_cluster_size: int,
    min_mean_similarity: float,
) -> PseudoSeedBundle:
    dataset = str(target_df["dataset"].iloc[0])
    lower_threshold = round(max(0.01, float(anchor_threshold - stability_delta)), 4)
    anchor_threshold = round(float(anchor_threshold), 4)
    upper_threshold = round(min(0.99, float(anchor_threshold + stability_delta)), 4)
    threshold_grid = sorted({lower_threshold, anchor_threshold, upper_threshold})
    sweep_df, prediction_df = run_threshold_sweep(df=target_df, embeddings=teacher_embeddings, thresholds=threshold_grid)
    pred_by_threshold = {
        round(float(threshold), 4): frame.reset_index(drop=True)
        for threshold, frame in prediction_df.groupby("threshold")
    }
    anchor_pred_df = pred_by_threshold[anchor_threshold].copy()
    lower_pred_df = pred_by_threshold[lower_threshold].copy()
    upper_pred_df = pred_by_threshold[upper_threshold].copy()
    score_matrix = np.clip(teacher_embeddings @ teacher_embeddings.T, -1.0, 1.0)

    anchor_labels = anchor_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    lower_labels = lower_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    upper_labels = upper_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    true_labels = target_df["identity"].astype(str).to_numpy()

    cluster_rows: list[dict[str, object]] = []
    pseudo_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    cluster_counter = 0
    accepted_indices: set[int] = set()

    for cluster_id in sorted(np.unique(anchor_labels).tolist()):
        members = np.flatnonzero(anchor_labels == cluster_id).tolist()
        member_set = set(members)
        lower_member_set = _cluster_members(lower_labels, members[0])
        upper_member_set = _cluster_members(upper_labels, members[0])
        mean_similarity = _mean_intra_cluster_similarity(score_matrix, members)
        is_stable = lower_member_set == member_set and upper_member_set == member_set
        size_ok = min_seed_cluster_size <= len(members) <= max_seed_cluster_size
        similarity_ok = mean_similarity >= min_mean_similarity
        accepted = bool(is_stable and size_ok and similarity_ok and len(members) >= 2)
        pseudo_identity = f"pseudo_{dataset}_{cluster_counter:04d}" if accepted else ""
        if accepted:
            cluster_counter += 1
            accepted_indices.update(member_set)
            for left_pos, left_index in enumerate(members):
                for right_index in members[left_pos + 1 :]:
                    pair_rows.append(
                        {
                            "image_id": str(target_df.iloc[left_index]["image_id"]),
                            "neighbor_image_id": str(target_df.iloc[right_index]["image_id"]),
                            "pair_kind": "cluster_internal",
                            "teacher_similarity": round(float(score_matrix[left_index, right_index]), 6),
                            "pseudo_identity": pseudo_identity,
                        }
                    )
        cluster_rows.append(
            {
                "anchor_cluster_id": int(cluster_id),
                "size": int(len(members)),
                "mean_similarity": round(float(mean_similarity), 6),
                "stable_lower": bool(lower_member_set == member_set),
                "stable_upper": bool(upper_member_set == member_set),
                "size_ok": bool(size_ok),
                "similarity_ok": bool(similarity_ok),
                "accepted_as_seed": bool(accepted),
                "pseudo_identity": pseudo_identity,
                "purity_vs_truth": round(
                    float(pd.Series(true_labels[members]).value_counts(normalize=True).iloc[0]),
                    6,
                ),
            }
        )
        for member_index in members:
            pseudo_rows.append(
                {
                    "image_id": str(target_df.iloc[member_index]["image_id"]),
                    "dataset": dataset,
                    "identity": str(target_df.iloc[member_index]["identity"]),
                    PATH_COLUMN: str(target_df.iloc[member_index][PATH_COLUMN]),
                    "anchor_cluster_id": int(cluster_id),
                    "seed_status": "seed" if accepted else "uncertain",
                    "pseudo_identity": pseudo_identity,
                }
            )

    pseudo_seed_df = pd.DataFrame(pseudo_rows)
    cluster_summary_df = pd.DataFrame(cluster_rows).sort_values(["accepted_as_seed", "size"], ascending=[False, False]).reset_index(drop=True)
    pair_table_df = pd.DataFrame(pair_rows)

    teacher_anchor_metrics = summarize_cluster_metrics(true_labels=true_labels, pred_labels=anchor_labels)
    best_row = pick_best_thresholds(sweep_df).iloc[0]
    teacher_best_threshold = round(float(best_row["threshold"]), 4)
    teacher_best_pred_df = pred_by_threshold[teacher_best_threshold].copy()
    teacher_best_metrics = summarize_cluster_metrics(
        true_labels=true_labels,
        pred_labels=teacher_best_pred_df["pred_cluster_id"].to_numpy(dtype=int),
    )

    return PseudoSeedBundle(
        target_df=target_df.reset_index(drop=True).copy(),
        anchor_threshold=float(anchor_threshold),
        lower_threshold=float(lower_threshold),
        upper_threshold=float(upper_threshold),
        pseudo_seed_df=pseudo_seed_df,
        cluster_summary_df=cluster_summary_df,
        threshold_summary_df=sweep_df,
        pair_table_df=pair_table_df,
        teacher_anchor_pred_df=anchor_pred_df,
        teacher_best_pred_df=teacher_best_pred_df,
        teacher_anchor_metrics=teacher_anchor_metrics,
        teacher_best_metrics=teacher_best_metrics,
    )


def build_mixed_training_frame(
    *,
    fit_df: pd.DataFrame,
    target_bundle: PseudoSeedBundle,
) -> tuple[pd.DataFrame, dict[str, int]]:
    labeled_fit = fit_df.copy().reset_index(drop=True)
    pseudo_df = target_bundle.pseudo_seed_df.copy().reset_index(drop=True)

    pseudo_identities = sorted([value for value in pseudo_df["pseudo_identity"].astype(str).unique().tolist() if value])
    pseudo_label_map = {pseudo_identity: index for index, pseudo_identity in enumerate(pseudo_identities)}

    labeled_fit["sample_role"] = "fit_labeled"
    labeled_fit["is_fit_labeled"] = True
    labeled_fit["is_pseudo_seed"] = False
    labeled_fit["pseudo_label_index"] = -1

    pseudo_df["sample_role"] = "target_seed"  # uncertain target samples are still included for distill only
    pseudo_df["is_fit_labeled"] = False
    pseudo_df["is_pseudo_seed"] = pseudo_df["seed_status"].eq("seed")
    pseudo_df["dataset_index"] = 0
    pseudo_df["label_index"] = -1
    pseudo_df["global_label_index"] = -1
    pseudo_df["identity_image_count_fit"] = 0
    pseudo_df["pseudo_label_index"] = -1
    pseudo_mask = pseudo_df["is_pseudo_seed"]
    if pseudo_label_map:
        pseudo_df.loc[pseudo_mask, "pseudo_label_index"] = (
            pseudo_df.loc[pseudo_mask, "pseudo_identity"].map(pseudo_label_map).astype(int)
        )
    desired_columns = labeled_fit.columns.tolist() + ["pseudo_identity", "seed_status"]
    for column in desired_columns:
        if column not in pseudo_df.columns:
            pseudo_df[column] = np.nan
    combined_df = pd.concat([labeled_fit, pseudo_df.reindex(columns=desired_columns)], ignore_index=True, sort=False)
    combined_df["pseudo_identity"] = combined_df["pseudo_identity"].fillna("").astype(str)
    combined_df["seed_status"] = combined_df["seed_status"].fillna("").astype(str)
    return combined_df.reset_index(drop=True), pseudo_label_map


class LabeledSelfTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        repo_root: Path,
        dataset_name: str,
        backbone_spec: Any,
        teacher_embeddings: np.ndarray,
    ) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.base_transform = build_eval_transform(backbone_spec, dataset=dataset_name)
        self.aug_transform = build_train_transform(backbone_spec, dataset=dataset_name)
        self.teacher_embeddings = teacher_embeddings.astype(np.float32, copy=False)
        if len(self.df) != len(self.teacher_embeddings):
            raise ValueError(f"Teacher embedding mismatch: df={len(self.df)} vs emb={len(self.teacher_embeddings)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            base_tensor = self.base_transform(image)
            aug_tensor = self.aug_transform(image)
        return {
            "base_image": base_tensor,
            "aug_image": aug_tensor,
            "teacher_embedding": torch.from_numpy(self.teacher_embeddings[index]),
            "is_fit_labeled": bool(row["is_fit_labeled"]),
            "is_pseudo_seed": bool(row["is_pseudo_seed"]),
            "label_index": int(row["label_index"]),
            "global_label_index": int(row["global_label_index"]),
            "identity_image_count_fit": int(row["identity_image_count_fit"]),
            "pseudo_label_index": int(row["pseudo_label_index"]),
        }


class LabeledSelfTrainModel(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        feature_dim: int,
        embedding_dim: int,
        fit_class_count: int,
        pseudo_class_count: int,
        teacher_dim: int,
        arcface_scale: float,
        arcface_margin: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.fit_head = ArcFaceHead(
            in_features=embedding_dim,
            out_features=fit_class_count,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        self.pseudo_head = (
            ArcFaceHead(
                in_features=embedding_dim,
                out_features=pseudo_class_count,
                scale=arcface_scale,
                margin=arcface_margin,
            )
            if pseudo_class_count > 0
            else None
        )
        self.teacher_projection = nn.Linear(embedding_dim, teacher_dim, bias=False) if teacher_dim > 0 else None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        from .descriptor_baselines import _coerce_model_output

        features = _coerce_model_output(self.backbone(images))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        embeddings = self.embedding_layer(features)
        embeddings = self.embedding_bn(embeddings)
        return F.normalize(embeddings, dim=1)

    def project_teacher_space(self, embeddings: torch.Tensor) -> torch.Tensor | None:
        if self.teacher_projection is None:
            return None
        projected = self.teacher_projection(embeddings)
        return F.normalize(projected, dim=1)


def compute_fit_arcface_loss(
    *,
    model: LabeledSelfTrainModel,
    embeddings: torch.Tensor,
    label_indices: torch.Tensor,
    is_fit_labeled: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    mask = is_fit_labeled & (label_indices >= 0)
    if not torch.any(mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    labels = label_indices[mask].long()
    logits = model.fit_head(embeddings[mask], labels)
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def compute_pseudo_arcface_loss(
    *,
    model: LabeledSelfTrainModel,
    embeddings: torch.Tensor,
    pseudo_label_indices: torch.Tensor,
    is_pseudo_seed: torch.Tensor,
    label_smoothing: float,
) -> torch.Tensor:
    if model.pseudo_head is None:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    mask = is_pseudo_seed & (pseudo_label_indices >= 0)
    if not torch.any(mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    labels = pseudo_label_indices[mask].long()
    logits = model.pseudo_head(embeddings[mask], labels)
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def maybe_limit_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True).copy()
    return df.head(limit).reset_index(drop=True).copy()


def build_optimizer(
    *,
    model: LabeledSelfTrainModel,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    head_modules: list[nn.Module] = [model.embedding_layer, model.embedding_bn, model.fit_head]
    if model.pseudo_head is not None:
        head_modules.append(model.pseudo_head)
    if model.teacher_projection is not None:
        head_modules.append(model.teacher_projection)
    head_params: list[torch.nn.Parameter] = []
    for module in head_modules:
        head_params.extend(list(module.parameters()))
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    warmup_ratio: float,
):
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(round(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def load_labeled_selftrain_init_checkpoint(
    *,
    model: LabeledSelfTrainModel,
    checkpoint_path: Path,
    dataset: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    for module_name in ["backbone", "embedding_layer", "embedding_bn", "teacher_projection"]:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        source_state = {
            key[len(module_name) + 1 :]: value
            for key, value in state_dict.items()
            if key.startswith(f"{module_name}.")
        }
        if not source_state:
            continue
        target_state = module.state_dict()
        compatible_state = {
            key: value
            for key, value in source_state.items()
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
        }
        incompatible = sorted(set(source_state) - set(compatible_state))
        module.load_state_dict(compatible_state, strict=False)
        loaded_keys.extend([f"{module_name}.{key}" for key in compatible_state])
        skipped_keys.extend([f"{module_name}.{key}" for key in incompatible])

    fit_head_key = f"arcface_heads.{dataset}.weight"
    if fit_head_key in state_dict and tuple(model.fit_head.weight.shape) == tuple(state_dict[fit_head_key].shape):
        with torch.no_grad():
            model.fit_head.weight.copy_(state_dict[fit_head_key])
        loaded_keys.append(fit_head_key)
    else:
        skipped_keys.append(fit_head_key)

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "loaded_keys_preview": loaded_keys[:20],
        "skipped_keys_preview": skipped_keys[:20],
    }


def load_labeled_selftrain_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: str,
) -> tuple[LabeledSelfTrainModel, Any, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    backbone, backbone_spec = load_student_backbone(str(config["student_backbone"]), device=device)
    model = LabeledSelfTrainModel(
        backbone=backbone,
        feature_dim=int(config["student_feature_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        fit_class_count=int(config["fit_class_count"]),
        pseudo_class_count=int(config.get("pseudo_class_count", 0)),
        teacher_dim=int(config["teacher_dim"]),
        arcface_scale=float(config["arcface_scale"]),
        arcface_margin=float(config["arcface_margin"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, backbone_spec, config, checkpoint


def build_mixed_sampler(df: pd.DataFrame, pseudo_seed_oversample_factor: float) -> WeightedRandomSampler | None:
    if pseudo_seed_oversample_factor <= 1.0:
        return None
    weights = np.ones(len(df), dtype=np.float32)
    if "is_pseudo_seed" in df.columns:
        weights[df["is_pseudo_seed"].to_numpy(dtype=bool)] = float(pseudo_seed_oversample_factor)
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(df), replacement=True)


def train_one_epoch(
    *,
    model: LabeledSelfTrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    scaler,
    supervised_loss_weight: float,
    pseudo_loss_weight: float,
    relation_distill_weight: float,
    feature_distill_weight: float,
    supcon_weight: float,
    supcon_temperature: float,
    label_smoothing: float,
    grad_clip_norm: float,
    max_train_batches: int | None,
) -> dict[str, float]:
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    totals = {
        "loss": 0.0,
        "fit_arcface_loss": 0.0,
        "pseudo_arcface_loss": 0.0,
        "relation_distill_loss": 0.0,
        "feature_distill_loss": 0.0,
        "supcon_loss": 0.0,
        "pseudo_seed_fraction": 0.0,
        "batches": 0,
    }
    use_amp = device.startswith("cuda")
    for batch_index, batch in enumerate(loader, start=1):
        if max_train_batches is not None and batch_index > max_train_batches:
            break
        base_images = batch["base_image"].to(device, non_blocking=True)
        aug_images = batch["aug_image"].to(device, non_blocking=True)
        teacher_embeddings = F.normalize(batch["teacher_embedding"].to(device, non_blocking=True), dim=1)
        is_fit_labeled = batch["is_fit_labeled"].to(device, non_blocking=True).bool()
        is_pseudo_seed = batch["is_pseudo_seed"].to(device, non_blocking=True).bool()
        label_indices = batch["label_index"].to(device, non_blocking=True).long()
        pseudo_label_indices = batch["pseudo_label_index"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model.encode(aug_images)
            fit_arcface_loss = compute_fit_arcface_loss(
                model=model,
                embeddings=embeddings,
                label_indices=label_indices,
                is_fit_labeled=is_fit_labeled,
                label_smoothing=label_smoothing,
            )
            pseudo_arcface_loss = compute_pseudo_arcface_loss(
                model=model,
                embeddings=embeddings,
                pseudo_label_indices=pseudo_label_indices,
                is_pseudo_seed=is_pseudo_seed,
                label_smoothing=label_smoothing,
            )
            projected_teacher = model.project_teacher_space(embeddings)
            feature_distill_loss = compute_feature_distillation_loss(projected_teacher, teacher_embeddings)
            relation_distill_loss = compute_relation_distillation_loss(
                student_embeddings=embeddings,
                teacher_embeddings=teacher_embeddings,
                dataset_indices=torch.zeros(len(embeddings), device=embeddings.device, dtype=torch.long),
            )
            if supcon_weight > 0:
                base_embeddings = model.encode(base_images)
                supcon_loss = compute_view_pair_contrastive_loss(
                    base_embeddings=base_embeddings,
                    augmented_embeddings=embeddings,
                    dataset_indices=torch.zeros(len(embeddings), device=embeddings.device, dtype=torch.long),
                    temperature=supcon_temperature,
                )
            else:
                supcon_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            loss = (supervised_loss_weight * fit_arcface_loss)
            loss = loss + (pseudo_loss_weight * pseudo_arcface_loss)
            loss = loss + (feature_distill_weight * feature_distill_loss)
            loss = loss + (relation_distill_weight * relation_distill_loss)
            loss = loss + (supcon_weight * supcon_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if (not use_amp) or (scaler.get_scale() >= scale_before_step):
            scheduler.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["fit_arcface_loss"] += float(fit_arcface_loss.detach().cpu())
        totals["pseudo_arcface_loss"] += float(pseudo_arcface_loss.detach().cpu())
        totals["relation_distill_loss"] += float(relation_distill_loss.detach().cpu())
        totals["feature_distill_loss"] += float(feature_distill_loss.detach().cpu())
        totals["supcon_loss"] += float(supcon_loss.detach().cpu())
        totals["pseudo_seed_fraction"] += float(is_pseudo_seed.float().mean().detach().cpu())
        totals["batches"] += 1

    batches = max(1, totals["batches"])
    return {
        "train_loss": round(totals["loss"] / batches, 6),
        "train_fit_arcface_loss": round(totals["fit_arcface_loss"] / batches, 6),
        "train_pseudo_arcface_loss": round(totals["pseudo_arcface_loss"] / batches, 6),
        "train_relation_distill_loss": round(totals["relation_distill_loss"] / batches, 6),
        "train_feature_distill_loss": round(totals["feature_distill_loss"] / batches, 6),
        "train_supcon_loss": round(totals["supcon_loss"] / batches, 6),
        "mean_pseudo_seed_fraction": round(totals["pseudo_seed_fraction"] / batches, 6),
        "peak_cuda_memory_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2)
        if device.startswith("cuda")
        else 0.0,
    }


def evaluate_target_embeddings(
    *,
    target_df: pd.DataFrame,
    target_embeddings: np.ndarray,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sweep_df, pred_df = run_threshold_sweep(df=target_df, embeddings=target_embeddings, thresholds=thresholds)
    best_df = pick_best_thresholds(sweep_df)
    recall_rows = [
        {
            "dataset": str(target_df["dataset"].iloc[0]),
            "recall_at_1": recall_at_k(target_embeddings, target_df["identity"].astype(str).to_numpy(), k=1),
            "recall_at_5": recall_at_k(target_embeddings, target_df["identity"].astype(str).to_numpy(), k=5),
        }
    ]
    return sweep_df, best_df, pd.DataFrame(recall_rows)


def write_plots(plots_dir: Path, training_log_df: pd.DataFrame) -> dict[str, Path]:
    _require_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    if training_log_df.empty:
        return {}
    epoch_values = training_log_df["epoch"].astype(float).to_numpy()
    paths: dict[str, Path] = {}

    loss_path = plots_dir / "training_loss_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epoch_values, training_log_df["train_loss"], marker="o", linewidth=2, color="#1f77b4", label="total")
    axes[0].set_title("Semi-SelfTrain Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    for column, label, color in [
        ("train_fit_arcface_loss", "Fit ArcFace", "#d62728"),
        ("train_pseudo_arcface_loss", "Pseudo ArcFace", "#9467bd"),
        ("train_relation_distill_loss", "Relation Distill", "#2ca02c"),
        ("train_feature_distill_loss", "Feature Distill", "#ff7f0e"),
        ("train_supcon_loss", "SupCon", "#8c564b"),
    ]:
        if column in training_log_df.columns:
            axes[1].plot(epoch_values, training_log_df[column], marker="o", linewidth=2, label=label, color=color)
    axes[1].set_title("Loss Breakdown")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(loss_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["loss"] = loss_path

    metric_path = plots_dir / "validation_metric_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epoch_values, training_log_df["target_ari"], marker="o", linewidth=2, color="#1f77b4", label="target ARI")
    axes[0].axhline(float(training_log_df["teacher_anchor_ari"].iloc[0]), color="#d62728", linestyle="--", label="teacher anchor ARI")
    axes[0].axhline(float(training_log_df["teacher_best_ari"].iloc[0]), color="#2ca02c", linestyle="--", label="teacher best ARI")
    axes[0].set_title("Target Validation ARI")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("ARI")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].plot(epoch_values, training_log_df["target_pairwise_f1"], marker="o", linewidth=2, color="#9467bd", label="target pairwise F1")
    axes[1].set_title("Target Validation Pairwise F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pairwise F1")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(metric_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["metric"] = metric_path
    return paths


def write_summary(
    *,
    output_dir: Path,
    config: dict[str, Any],
    pseudo_bundle: PseudoSeedBundle,
    teacher_component_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    best_eval_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> tuple[Path, Path]:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_json = reports_dir / "summary.json"
    summary_md = reports_dir / "summary.md"
    summary_payload = {
        **config,
        "teacher_anchor_metrics": pseudo_bundle.teacher_anchor_metrics,
        "teacher_best_metrics": pseudo_bundle.teacher_best_metrics,
        "best_eval": best_eval_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_row = best_eval_df.iloc[0]
    lines = [
        "# Labeled Semi-SelfTrain Summary",
        "",
        "## Experiment Card",
        "",
        f"- `experiment_id`: `{config['experiment_id']}`",
        f"- `dataset`: `{config['dataset']}`",
        f"- `goal`: `{config['goal']}`",
        f"- `student_backbone`: `{config['student_backbone']}`",
        f"- `student_model_id`: `{config['student_model_id']}`",
        f"- `student_init_checkpoint`: `{config.get('student_init_checkpoint', '') or 'none'}`",
        "",
        "## Split Protocol",
        "",
        f"- `fit_images`: `{config['fit_images']}`",
        f"- `target_images`: `{config['target_images']}`",
        f"- `fit_ids`: `{config['fit_ids']}`",
        f"- `target_ids`: `{config['target_ids']}`",
        f"- `val_identity_fraction`: `{config['val_identity_fraction']}`",
        f"- `split_seed`: `{config['split_seed']}`",
        "",
        "## Teacher Sources",
        "",
        dataframe_to_markdown_table(teacher_component_df),
        "",
        "## Pseudo Seed Summary",
        "",
        f"- `anchor_threshold`: `{pseudo_bundle.anchor_threshold}`",
        f"- `stability_thresholds`: `[{pseudo_bundle.lower_threshold}, {pseudo_bundle.anchor_threshold}, {pseudo_bundle.upper_threshold}]`",
        f"- `seed_images`: `{int(pseudo_bundle.pseudo_seed_df['seed_status'].eq('seed').sum())}`",
        f"- `seed_coverage_ratio`: `{round(float(pseudo_bundle.pseudo_seed_df['seed_status'].eq('seed').mean()), 6)}`",
        f"- `accepted_seed_clusters`: `{int(pseudo_bundle.cluster_summary_df['accepted_as_seed'].sum())}`",
        "",
        dataframe_to_markdown_table(
            pseudo_bundle.cluster_summary_df[
                [
                    "anchor_cluster_id",
                    "size",
                    "mean_similarity",
                    "stable_lower",
                    "stable_upper",
                    "accepted_as_seed",
                    "purity_vs_truth",
                ]
            ].head(20)
        ),
        "",
        "## Training Config",
        "",
        f"- `input_size`: `{config['input_size']}`",
        f"- `teacher_dim`: `{config['teacher_dim']}`",
        f"- `embedding_dim`: `{config['embedding_dim']}`",
        f"- `train_batch_size / eval_batch_size`: `{config['train_batch_size']} / {config['eval_batch_size']}`",
        f"- `resolved_backbone_lr / resolved_head_lr`: `{config['resolved_backbone_lr']} / {config['resolved_head_lr']}`",
        f"- `loss_weights`: `fit={config['supervised_loss_weight']}, pseudo={config['pseudo_loss_weight']}, relation={config['relation_distill_weight']}, feature={config['feature_distill_weight']}, supcon={config['supcon_weight']}`",
        "",
        "## Teacher Baseline On Target Split",
        "",
        f"- Anchor threshold metrics: `ARI={pseudo_bundle.teacher_anchor_metrics['ari']}`, `pairwise_f1={pseudo_bundle.teacher_anchor_metrics['pairwise_f1']}`, `clusters={pseudo_bundle.teacher_anchor_metrics['cluster_count']}`",
        f"- Best threshold metrics: `ARI={pseudo_bundle.teacher_best_metrics['ari']}`, `pairwise_f1={pseudo_bundle.teacher_best_metrics['pairwise_f1']}`, `clusters={pseudo_bundle.teacher_best_metrics['cluster_count']}`",
        "",
        "## Best Student Result",
        "",
        dataframe_to_markdown_table(best_eval_df),
        "",
        "## Target Recall",
        "",
        dataframe_to_markdown_table(recall_df),
        "",
        "## Epoch Log",
        "",
        dataframe_to_markdown_table(training_log_df),
    ]
    if plot_paths:
        lines.extend(
            [
                "",
                "## Monitoring Figures",
                "",
            ]
        )
        if "loss" in plot_paths:
            lines.extend(
                [
                    "![Training loss curves](plots/training_loss_curves.png)",
                    "",
                    "- 读图方式：先看总 `train_loss` 是否下降，再看 `fit / pseudo / distill / SupCon` 各分量谁在主导优化。",
                    "",
                ]
            )
        if "metric" in plot_paths:
            lines.extend(
                [
                    "![Validation metric curves](plots/validation_metric_curves.png)",
                    "",
                    "- 读图方式：先看 student `ARI` 是否超过 teacher anchor，再看相对 teacher best threshold 还有多大差距。",
                    "",
                ]
            )
    best_target_ari = float(best_row["ari"])
    teacher_anchor_ari = float(pseudo_bundle.teacher_anchor_metrics["ari"])
    if best_target_ari > teacher_anchor_ari:
        decision_line = (
            f"- Student `target ARI {best_target_ari:.6f}` exceeds teacher anchor `ARI {teacher_anchor_ari:.6f}`; "
            "this route is eligible for test-time adaptation and then a Kaggle submission variant."
        )
    else:
        decision_line = (
            f"- Student `target ARI {best_target_ari:.6f}` is still below teacher anchor `ARI {teacher_anchor_ari:.6f}`; "
            "do not submit yet. First improve pseudo coverage / seed purity or change the student training recipe."
        )
    lines.extend(
        [
            "## Next Decision",
            "",
            decision_line,
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_md, summary_json


def run_labeled_selftrain(
    *,
    repo_root: Path,
    output_dir: Path,
    experiment_id: str,
    dataset: str,
    student_backbone: str,
    teacher_descriptor_sources: list[str],
    teacher_checkpoint_sources: list[str],
    device: str = "cuda:0",
    anchor_threshold: float,
    stability_delta: float,
    epochs: int = 6,
    embedding_dim: int = 512,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    val_identity_fraction: float = 0.1,
    split_seed: int = 42,
    eval_thresholds: list[float] | None = None,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    lr_reference_batch_size: int = 4,
    lr_scale_mode: str = "linear",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.3,
    supervised_loss_weight: float = 1.0,
    pseudo_loss_weight: float = 0.5,
    relation_distill_weight: float = 0.2,
    feature_distill_weight: float = 0.05,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 1.0,
    pseudo_seed_oversample_factor: float = 2.0,
    min_seed_cluster_size: int = 2,
    max_seed_cluster_size: int = 12,
    min_mean_similarity: float = 0.0,
    max_fit_rows: int | None = None,
    max_target_rows: int | None = None,
    max_train_batches: int | None = None,
    goal: str | None = None,
    train_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
    student_init_checkpoint: Path | None = None,
) -> dict[str, Path]:
    _require_torch()
    if eval_thresholds is None:
        eval_thresholds = DEFAULT_THRESHOLDS
    seed_everything(split_seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    tables_dir = output_dir / "tables"
    embeddings_dir = output_dir / "embeddings"
    plots_dir = reports_dir / "plots"
    for path in [checkpoints_dir, reports_dir, tables_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)
    resource_snapshot = collect_resource_snapshot(device)

    train_df, _test_df = load_manifests(
        repo_root=repo_root,
        train_manifest_path=train_manifest_path,
        test_manifest_path=test_manifest_path,
    )
    split_df = build_identity_holdout_split(
        train_df=train_df,
        val_identity_fraction=val_identity_fraction,
        seed=split_seed,
        datasets=[dataset],
    )
    fit_df = split_df[split_df["split_role_v1"] == "fit"].copy().reset_index(drop=True)
    target_df = split_df[split_df["split_role_v1"] == "val"].copy().reset_index(drop=True)
    fit_df = maybe_limit_rows(fit_df, max_fit_rows)
    target_df = maybe_limit_rows(target_df, max_target_rows)
    if fit_df.empty or target_df.empty:
        raise ValueError(f"Split for dataset={dataset} produced empty fit/target slice")
    fit_df, fit_label_map = attach_single_dataset_labels(fit_df=fit_df, dataset=dataset)
    split_df.to_csv(tables_dir / "split_manifest_v1.csv", index=False)
    fit_df.to_csv(tables_dir / "fit_manifest_v1.csv", index=False)
    target_df.to_csv(tables_dir / "target_manifest_v1.csv", index=False)

    teacher_specs = parse_teacher_sources(teacher_descriptor_sources, teacher_checkpoint_sources)
    teacher_fit_embeddings, teacher_target_embeddings, teacher_component_df = build_teacher_embeddings(
        repo_root=repo_root,
        fit_df=fit_df,
        target_df=target_df,
        teacher_specs=teacher_specs,
        device=device,
        num_workers=num_workers,
    )
    teacher_component_df.to_csv(tables_dir / "teacher_components_v1.csv", index=False)
    np.save(embeddings_dir / "teacher_fit_embeddings_v1.npy", teacher_fit_embeddings.astype(np.float32))
    np.save(embeddings_dir / "teacher_target_embeddings_v1.npy", teacher_target_embeddings.astype(np.float32))

    pseudo_bundle = build_stable_pseudo_seed_bundle(
        target_df=target_df,
        teacher_embeddings=teacher_target_embeddings,
        anchor_threshold=anchor_threshold,
        stability_delta=stability_delta,
        min_seed_cluster_size=min_seed_cluster_size,
        max_seed_cluster_size=max_seed_cluster_size,
        min_mean_similarity=min_mean_similarity,
    )
    pseudo_bundle.pseudo_seed_df.to_csv(tables_dir / "pseudo_seed_assignments_v1.csv", index=False)
    pseudo_bundle.cluster_summary_df.to_csv(tables_dir / "pseudo_seed_clusters_v1.csv", index=False)
    pseudo_bundle.threshold_summary_df.to_csv(tables_dir / "teacher_target_threshold_sweep_v1.csv", index=False)
    pseudo_bundle.pair_table_df.to_csv(tables_dir / "pseudo_seed_pairs_v1.csv", index=False)
    pseudo_bundle.teacher_anchor_pred_df.to_csv(tables_dir / "teacher_anchor_predictions_v1.csv", index=False)
    pseudo_bundle.teacher_best_pred_df.to_csv(tables_dir / "teacher_best_predictions_v1.csv", index=False)

    mixed_df, pseudo_label_map = build_mixed_training_frame(
        fit_df=fit_df,
        target_bundle=pseudo_bundle,
    )
    target_teacher_index = {
        str(row["image_id"]): teacher_target_embeddings[index]
        for index, row in target_df.reset_index(drop=True).iterrows()
    }
    mixed_teacher_rows = [teacher_fit_embeddings.astype(np.float32)]
    pseudo_teacher_rows = np.stack(
        [target_teacher_index[str(image_id)] for image_id in mixed_df.loc[~mixed_df["is_fit_labeled"], "image_id"].astype(str).tolist()],
        axis=0,
    ).astype(np.float32)
    mixed_teacher_rows.append(pseudo_teacher_rows)
    mixed_teacher_embeddings = np.concatenate(mixed_teacher_rows, axis=0).astype(np.float32)
    if len(mixed_teacher_embeddings) != len(mixed_df):
        raise ValueError(f"Mixed teacher embedding mismatch: df={len(mixed_df)} vs emb={len(mixed_teacher_embeddings)}")
    mixed_df.to_csv(tables_dir / "mixed_training_manifest_v1.csv", index=False)
    np.save(embeddings_dir / "teacher_mixed_embeddings_v1.npy", mixed_teacher_embeddings)

    backbone, backbone_spec = load_student_backbone(student_backbone, device=device)
    if train_batch_size is None:
        train_batch_size = backbone_spec.default_train_batch_size
    if eval_batch_size is None:
        eval_batch_size = backbone_spec.default_eval_batch_size
    effective_batch_size = int(train_batch_size)
    resolved_backbone_lr = scale_learning_rate(
        base_lr=backbone_lr,
        effective_batch_size=effective_batch_size,
        reference_batch_size=lr_reference_batch_size,
        mode=lr_scale_mode,
    )
    resolved_head_lr = scale_learning_rate(
        base_lr=head_lr,
        effective_batch_size=effective_batch_size,
        reference_batch_size=lr_reference_batch_size,
        mode=lr_scale_mode,
    )

    model = LabeledSelfTrainModel(
        backbone=backbone,
        feature_dim=backbone_spec.feature_dim,
        embedding_dim=embedding_dim,
        fit_class_count=len(fit_label_map),
        pseudo_class_count=len(pseudo_label_map),
        teacher_dim=int(teacher_fit_embeddings.shape[1]),
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
    ).to(device)
    init_checkpoint_info: dict[str, object] | None = None
    if student_init_checkpoint is not None:
        init_checkpoint_info = load_labeled_selftrain_init_checkpoint(
            model=model,
            checkpoint_path=student_init_checkpoint,
            dataset=dataset,
        )
    train_transform = build_train_transform(backbone_spec, dataset=dataset)
    sampler = build_mixed_sampler(mixed_df, pseudo_seed_oversample_factor=pseudo_seed_oversample_factor)
    train_dataset = LabeledSelfTrainDataset(
        df=mixed_df,
        repo_root=repo_root,
        dataset_name=dataset,
        backbone_spec=backbone_spec,
        teacher_embeddings=mixed_teacher_embeddings,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    optimizer = build_optimizer(
        model=model,
        backbone_lr=resolved_backbone_lr,
        head_lr=resolved_head_lr,
        weight_decay=weight_decay,
    )
    steps_per_epoch = max(1, min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader))
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        warmup_ratio=warmup_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    config = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "goal": goal
        or (
            f"Run dataset-routed labeled semi-selftrain on `{dataset}`: labeled fit split supplies supervised ArcFace, "
            "target split is treated as unlabeled domain for stable pseudo seeds + teacher distillation."
        ),
        "student_backbone": student_backbone,
        "student_model_id": backbone_spec.model_id,
        "student_feature_dim": backbone_spec.feature_dim,
        "teacher_descriptor_sources": teacher_descriptor_sources,
        "teacher_checkpoint_sources": teacher_checkpoint_sources,
        "teacher_dim": int(teacher_fit_embeddings.shape[1]),
        "input_size": backbone_spec.input_size,
        "embedding_dim": embedding_dim,
        "device": device,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "val_identity_fraction": val_identity_fraction,
        "split_seed": split_seed,
        "eval_thresholds": eval_thresholds,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "lr_reference_batch_size": lr_reference_batch_size,
        "lr_scale_mode": lr_scale_mode,
        "resolved_backbone_lr": resolved_backbone_lr,
        "resolved_head_lr": resolved_head_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "arcface_scale": arcface_scale,
        "arcface_margin": arcface_margin,
        "supervised_loss_weight": supervised_loss_weight,
        "pseudo_loss_weight": pseudo_loss_weight,
        "relation_distill_weight": relation_distill_weight,
        "feature_distill_weight": feature_distill_weight,
        "supcon_weight": supcon_weight,
        "supcon_temperature": supcon_temperature,
        "label_smoothing": label_smoothing,
        "grad_clip_norm": grad_clip_norm,
        "pseudo_seed_oversample_factor": pseudo_seed_oversample_factor,
        "anchor_threshold": anchor_threshold,
        "stability_delta": stability_delta,
        "min_seed_cluster_size": min_seed_cluster_size,
        "max_seed_cluster_size": max_seed_cluster_size,
        "min_mean_similarity": min_mean_similarity,
        "fit_images": int(len(fit_df)),
        "target_images": int(len(target_df)),
        "fit_ids": int(fit_df["identity"].nunique()),
        "target_ids": int(target_df["identity"].nunique()),
        "fit_class_count": int(len(fit_label_map)),
        "pseudo_class_count": int(len(pseudo_label_map)),
        "effective_batch_size": effective_batch_size,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "amp_enabled": bool(device.startswith("cuda")),
        "train_augmentation": describe_transform(train_transform),
        "eval_preprocess": describe_transform(build_eval_transform(backbone_spec, dataset=dataset)),
        "resource_snapshot": resource_snapshot,
        "student_init_checkpoint": str(student_init_checkpoint) if student_init_checkpoint else "",
        "student_init_info": init_checkpoint_info or {},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    teacher_anchor_metrics = pseudo_bundle.teacher_anchor_metrics
    teacher_best_metrics = pseudo_bundle.teacher_best_metrics
    eval_history_rows: list[dict[str, object]] = []
    best_sort_key = (float("-inf"), float("-inf"))
    best_epoch = -1
    best_student_embeddings: np.ndarray | None = None

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            supervised_loss_weight=supervised_loss_weight,
            pseudo_loss_weight=pseudo_loss_weight,
            relation_distill_weight=relation_distill_weight,
            feature_distill_weight=feature_distill_weight,
            supcon_weight=supcon_weight,
            supcon_temperature=supcon_temperature,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
            max_train_batches=max_train_batches,
        )
        target_student_embeddings = extract_student_embeddings(
            df=target_df,
            repo_root=repo_root,
            model=model,
            spec=backbone_spec,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        sweep_df, best_eval_df, recall_df = evaluate_target_embeddings(
            target_df=target_df,
            target_embeddings=target_student_embeddings,
            thresholds=eval_thresholds,
        )
        best_row = best_eval_df.iloc[0]
        chosen_threshold = float(best_row["threshold"])
        chosen_pred_df = apply_thresholds_to_df(
            df=target_df,
            embeddings=target_student_embeddings,
            threshold_by_dataset={dataset: chosen_threshold},
        )
        chosen_pred_df.to_csv(tables_dir / f"target_predictions_epoch_{epoch:02d}.csv", index=False)
        sweep_df.to_csv(tables_dir / f"target_threshold_sweep_epoch_{epoch:02d}.csv", index=False)
        epoch_row = {
            "epoch": epoch,
            **train_metrics,
            "target_threshold": chosen_threshold,
            "target_ari": float(best_row["ari"]),
            "target_nmi": float(best_row["nmi"]),
            "target_pairwise_f1": float(best_row["pairwise_f1"]),
            "target_pairwise_precision": float(best_row["pairwise_precision"]),
            "target_pairwise_recall": float(best_row["pairwise_recall"]),
            "target_cluster_count": int(best_row["cluster_count"]),
            "target_singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
            "target_recall_at_1": float(recall_df.iloc[0]["recall_at_1"]),
            "target_recall_at_5": float(recall_df.iloc[0]["recall_at_5"]),
            "teacher_anchor_ari": float(teacher_anchor_metrics["ari"]),
            "teacher_best_ari": float(teacher_best_metrics["ari"]),
            "teacher_anchor_pairwise_f1": float(teacher_anchor_metrics["pairwise_f1"]),
            "teacher_best_pairwise_f1": float(teacher_best_metrics["pairwise_f1"]),
            "seed_images": int(pseudo_bundle.pseudo_seed_df["seed_status"].eq("seed").sum()),
            "seed_coverage_ratio": round(float(pseudo_bundle.pseudo_seed_df["seed_status"].eq("seed").mean()), 6),
        }
        eval_history_rows.append(epoch_row)
        training_log_df = pd.DataFrame(eval_history_rows)
        training_log_df.to_csv(tables_dir / "training_log_v1.csv", index=False)

        current_sort_key = (float(best_row["ari"]), float(best_row["pairwise_f1"]))
        if current_sort_key > best_sort_key:
            best_sort_key = current_sort_key
            best_epoch = epoch
            best_student_embeddings = target_student_embeddings.astype(np.float32)
            np.save(embeddings_dir / "best_target_embeddings_v1.npy", best_student_embeddings)
            chosen_pred_df.to_csv(tables_dir / "best_target_predictions_v1.csv", index=False)
            sweep_df.to_csv(tables_dir / "best_target_threshold_sweep_v1.csv", index=False)
            best_eval_df.to_csv(tables_dir / "best_eval_v1.csv", index=False)
            recall_df.to_csv(tables_dir / "target_recall_v1.csv", index=False)
            build_neighbor_table(
                df=target_df,
                embeddings=best_student_embeddings,
                top_k=5,
            ).to_csv(tables_dir / "best_target_neighbors_v1.csv", index=False)
            plot_paths = write_plots(plots_dir=plots_dir, training_log_df=training_log_df)
            write_summary(
                output_dir=output_dir,
                config=config,
                pseudo_bundle=pseudo_bundle,
                teacher_component_df=teacher_component_df,
                training_log_df=training_log_df,
                best_eval_df=best_eval_df,
                recall_df=recall_df,
                plot_paths=plot_paths,
            )
            torch.save(
                {
                    "epoch": epoch,
                    "best_target_ari": float(best_row["ari"]),
                    "config": config,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                },
                checkpoints_dir / "best.pt",
            )

        torch.save(
            {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_target_ari": best_sort_key[0],
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            },
            checkpoints_dir / "last.pt",
        )
        np.save(embeddings_dir / "last_target_embeddings_v1.npy", target_student_embeddings.astype(np.float32))
        print(
            f"[labeled_selftrain] epoch={epoch}/{epochs} dataset={dataset} "
            f"train_loss={train_metrics['train_loss']:.4f} target_ari={float(best_row['ari']):.4f} "
            f"target_pairwise_f1={float(best_row['pairwise_f1']):.4f} best_ari={best_sort_key[0]:.4f}",
            flush=True,
        )

    if best_student_embeddings is None:
        raise RuntimeError("No best student embeddings were produced")

    summary_payload = {
        "best_epoch": best_epoch,
        "best_target_ari": round(best_sort_key[0], 6),
        "best_target_pairwise_f1": round(best_sort_key[1], 6),
        "experiment_id": experiment_id,
        "dataset": dataset,
        "student_backbone": student_backbone,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "summary_path": reports_dir / "summary.md",
        "training_log_path": tables_dir / "training_log_v1.csv",
        "best_checkpoint_path": checkpoints_dir / "best.pt",
        "best_predictions_path": tables_dir / "best_target_predictions_v1.csv",
        "best_embeddings_path": embeddings_dir / "best_target_embeddings_v1.npy",
    }
