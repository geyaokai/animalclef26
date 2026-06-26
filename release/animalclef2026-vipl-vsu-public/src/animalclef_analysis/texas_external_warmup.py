from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:  # pragma: no cover - exercised in the training env
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.cuda.amp import GradScaler, autocast
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # pragma: no cover - keeps helper imports light
    plt = None
    torch = None
    F = None

    class _NNProxy:
        Module = object
        Linear = object
        BatchNorm1d = object

    nn = _NNProxy()
    GradScaler = autocast = DataLoader = Dataset = WeightedRandomSampler = object

from .descriptor_baselines import PATH_COLUMN, dataframe_to_markdown_table, l2_normalize, recall_at_k
from .texas_unsupervised import TEXAS_DATASET

if torch is not None:  # pragma: no cover - only imported in the training env
    from .supervised_training import (
        ArcFaceHead,
        build_eval_transform,
        build_train_transform,
        collect_resource_snapshot,
        describe_transform,
        load_student_backbone,
        scale_learning_rate,
        seed_everything,
    )
else:  # pragma: no cover - exercised in light unit-test envs
    ArcFaceHead = object


DEFAULT_WARMUP_DATASET_NAME = "TexasHornedLizards"


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("torch is required for Texas external warmup")


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for Texas external warmup plots")


def _string_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str)


def build_tcu_texas_warmup_manifest(
    *,
    repo_root: Path,
    chip_manifest_path: Path,
    output_path: Path,
    dataset_name: str = DEFAULT_WARMUP_DATASET_NAME,
) -> pd.DataFrame:
    chip_manifest_df = pd.read_csv(chip_manifest_path).copy()
    required_columns = {"chip_id", "chip_path_v1", "external_identity_v1", "external_identity_image_count_v1"}
    missing_columns = sorted(required_columns - set(chip_manifest_df.columns))
    if missing_columns:
        raise ValueError(f"TCU chip manifest is missing warmup columns: {missing_columns}")

    chip_manifest_df["chip_id"] = chip_manifest_df["chip_id"].astype(str)
    chip_manifest_df["chip_path_v1"] = chip_manifest_df["chip_path_v1"].fillna("").astype(str)
    chip_manifest_df["external_identity_v1"] = chip_manifest_df["external_identity_v1"].fillna("").astype(str)
    chip_manifest_df["external_identity_image_count_v1"] = (
        pd.to_numeric(chip_manifest_df["external_identity_image_count_v1"], errors="coerce").fillna(0).astype(int)
    )

    manifest_df = chip_manifest_df[
        chip_manifest_df["chip_path_v1"].ne("") & chip_manifest_df["external_identity_v1"].ne("")
    ].copy()
    if manifest_df.empty:
        raise ValueError(f"No parseable TCU Texas chips found in {chip_manifest_path}")

    manifest_df["path_exists_v1"] = manifest_df["chip_path_v1"].map(lambda value: (repo_root / value).exists())
    manifest_df = manifest_df[manifest_df["path_exists_v1"]].copy().reset_index(drop=True)
    if manifest_df.empty:
        raise ValueError(f"No existing chip paths remained after filtering {chip_manifest_path}")

    manifest_df["image_id"] = manifest_df["chip_id"].map(lambda value: f"tcu_chip_{int(value):04d}")
    manifest_df["dataset"] = str(dataset_name)
    manifest_df["split"] = "train"
    manifest_df["identity"] = manifest_df["external_identity_v1"]
    manifest_df[PATH_COLUMN] = manifest_df["chip_path_v1"]
    manifest_df["source_domain"] = "tcu_chips"
    manifest_df["label_source"] = "external_identity_v1"
    manifest_df["is_singleton_identity_v1"] = manifest_df["external_identity_image_count_v1"].eq(1)
    manifest_df["original_image_name_mapped_v1"] = _string_column(manifest_df, "original_image_name_mapped_v1")
    manifest_df["original_match_filename_v1"] = _string_column(manifest_df, "original_match_filename_v1")
    manifest_df["capture_year_v1"] = _string_column(manifest_df, "capture_year_v1")

    manifest_df = manifest_df.sort_values(["identity", "chip_id"], ascending=[True, True]).reset_index(drop=True)
    if manifest_df["image_id"].duplicated().any():
        duplicated = manifest_df.loc[manifest_df["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Duplicate warmup image_id values detected: {duplicated}")

    manifest_df = manifest_df[
        [
            "image_id",
            "dataset",
            "split",
            "identity",
            PATH_COLUMN,
            "chip_id",
            "external_identity_image_count_v1",
            "is_singleton_identity_v1",
            "source_domain",
            "label_source",
            "original_image_name_mapped_v1",
            "original_match_filename_v1",
            "capture_year_v1",
        ]
    ].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)
    return manifest_df


def summarize_warmup_classes(manifest_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        manifest_df.groupby("identity", dropna=False)
        .agg(
            images=("image_id", "count"),
            singleton=("is_singleton_identity_v1", "first"),
            capture_years=("capture_year_v1", lambda values: ",".join(sorted({str(value) for value in values if str(value).strip()}))),
        )
        .reset_index()
        .sort_values(["images", "identity"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df["identity_rank"] = np.arange(1, len(summary_df) + 1)
    return summary_df


def attach_warmup_labels(manifest_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    labeled_df = manifest_df.copy().reset_index(drop=True)
    identities = sorted(labeled_df["identity"].astype(str).unique().tolist())
    label_map = {identity: index for index, identity in enumerate(identities)}
    labeled_df["label_index"] = labeled_df["identity"].map(label_map).astype(int)
    labeled_df["identity_image_count_fit"] = labeled_df.groupby("identity")["image_id"].transform("size").astype(int)
    class_summary_df = summarize_warmup_classes(labeled_df)
    return labeled_df, label_map, class_summary_df


class TexasWarmupDataset(Dataset):
    def __init__(self, df: pd.DataFrame, repo_root: Path, transform: Any) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "label_index": int(row["label_index"]),
        }


class TexasWarmupModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        embedding_dim: int,
        class_count: int,
        arcface_scale: float,
        arcface_margin: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.pseudo_head = ArcFaceHead(
            in_features=embedding_dim,
            out_features=class_count,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        from .descriptor_baselines import _coerce_model_output

        features = _coerce_model_output(self.backbone(images))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        embeddings = self.embedding_layer(features)
        embeddings = self.embedding_bn(embeddings)
        return F.normalize(embeddings, dim=1)

    def classify(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.pseudo_head(embeddings, labels)


def build_warmup_sampler(df: pd.DataFrame) -> WeightedRandomSampler | None:
    class_counts = df["label_index"].value_counts().to_dict()
    if not class_counts:
        return None
    weights = df["label_index"].map(lambda value: 1.0 / float(class_counts[int(value)])).to_numpy(dtype=np.float32)
    return WeightedRandomSampler(weights=weights.tolist(), num_samples=len(df), replacement=True)


def build_warmup_optimizer(
    model: TexasWarmupModel,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    head_params: list[torch.nn.Parameter] = []
    for module in [model.embedding_layer, model.embedding_bn, model.pseudo_head]:
        head_params.extend(list(module.parameters()))
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
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


def train_warmup_one_epoch(
    *,
    model: TexasWarmupModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
    scaler: GradScaler,
    label_smoothing: float,
    grad_clip_norm: float,
    max_train_batches: int | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_examples = 0
    total_correct = 0
    peak_cuda_memory_mb = 0.0

    for step, batch in enumerate(loader, start=1):
        if max_train_batches is not None and step > max_train_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label_index"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.startswith("cuda")):
            embeddings = model.encode(images)
            logits = model.classify(embeddings, labels)
            loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_size = int(images.shape[0])
        total_examples += batch_size
        total_loss += float(loss.detach().cpu()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        if device.startswith("cuda"):
            peak_cuda_memory_mb = max(
                peak_cuda_memory_mb,
                float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)),
            )

    mean_loss = float(total_loss / max(total_examples, 1))
    train_accuracy = float(total_correct / max(total_examples, 1))
    return {
        "loss": round(mean_loss, 6),
        "train_accuracy": round(train_accuracy, 6),
        "peak_cuda_memory_mb": round(peak_cuda_memory_mb, 2),
    }


def extract_warmup_embeddings(
    *,
    df: pd.DataFrame,
    repo_root: Path,
    model: TexasWarmupModel,
    transform: Any,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    dataset = TexasWarmupDataset(df=df, repo_root=repo_root, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    embedding_blocks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            embeddings = model.encode(images)
            embedding_blocks.append(embeddings.detach().cpu().numpy().astype(np.float32))
    if not embedding_blocks:
        return np.empty((0, model.embedding_layer.out_features), dtype=np.float32)
    return np.vstack(embedding_blocks).astype(np.float32)


def evaluate_warmup_retrieval(manifest_df: pd.DataFrame, embeddings: np.ndarray) -> dict[str, float]:
    labels = manifest_df["identity"].astype(str).to_numpy()
    metrics = {
        "fit_recall_at_1": recall_at_k(embeddings, labels, k=1),
        "fit_recall_at_5": recall_at_k(embeddings, labels, k=5),
    }
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    positive_values: list[float] = []
    negative_values: list[float] = []
    for index, label in enumerate(labels):
        same_mask = labels == label
        same_mask[index] = False
        diff_mask = labels != label
        if np.any(same_mask):
            positive_values.append(float(np.max(similarity[index, same_mask])))
        if np.any(diff_mask):
            negative_values.append(float(np.max(similarity[index, diff_mask])))
    metrics["mean_best_positive_similarity"] = round(float(np.mean(positive_values)), 6) if positive_values else 0.0
    metrics["mean_best_negative_similarity"] = round(float(np.mean(negative_values)), 6) if negative_values else 0.0
    metrics["similarity_margin"] = round(
        float(metrics["mean_best_positive_similarity"] - metrics["mean_best_negative_similarity"]),
        6,
    )
    return metrics


def write_warmup_plots(plots_dir: Path, training_log_df: pd.DataFrame) -> dict[str, Path]:
    _require_matplotlib()
    plot_paths: dict[str, Path] = {}
    if training_log_df.empty:
        return plot_paths

    curves_path = plots_dir / "training_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(training_log_df["epoch"], training_log_df["loss"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].plot(training_log_df["epoch"], training_log_df["train_accuracy"], marker="o", linewidth=2, color="#ff7f0e")
    axes[0].set_title("TCU Warmup Train Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.3)
    axes[0].legend(["loss", "train_accuracy"], loc="best")

    axes[1].plot(training_log_df["epoch"], training_log_df["fit_recall_at_1"], marker="o", linewidth=2, color="#2ca02c")
    axes[1].plot(training_log_df["epoch"], training_log_df["fit_recall_at_5"], marker="o", linewidth=2, color="#9467bd")
    axes[1].set_title("Fit Retrieval Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall")
    axes[1].grid(alpha=0.3)
    axes[1].legend(["fit_recall_at_1", "fit_recall_at_5"], loc="best")

    fig.savefig(curves_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths["curves"] = curves_path
    return plot_paths


def write_warmup_report(
    output_path: Path,
    *,
    config: dict[str, object],
    class_summary_df: pd.DataFrame,
    training_log_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    best_row = training_log_df.sort_values(
        ["fit_recall_at_1", "fit_recall_at_5", "similarity_margin", "epoch"],
        ascending=[False, False, False, True],
    ).iloc[0]
    selected_gpu = config.get("resource_snapshot", {}).get("selected_gpu", {})
    tmux_sessions = config.get("resource_snapshot", {}).get("tmux_sessions", [])
    lines = [
        "# Texas External Warmup Summary",
        "",
        "## Experiment Card",
        "",
        f"- `experiment_id`: `{config['experiment_id']}`",
        f"- `goal`: `{config['goal']}`",
        f"- `dataset`: `{config['dataset']}`",
        f"- `manifest_path`: `{config['manifest_path']}`",
        f"- `student_backbone`: `{config['student_backbone']}`",
        f"- `student_model_id`: `{config['student_model_id']}`",
        f"- `input_size`: `{config['input_size']}`",
        f"- `train_augmentation`: `{config['train_augmentation']}`",
        f"- `eval_preprocess`: `{config['eval_preprocess']}`",
        "",
        "## Label Pool",
        "",
        f"- `images`: `{config['images']}`",
        f"- `classes`: `{config['classes']}`",
        f"- `singleton_classes`: `{config['singleton_classes']}`",
        f"- `multi_image_classes`: `{config['multi_image_classes']}`",
        "",
        dataframe_to_markdown_table(class_summary_df.head(25)),
        "",
        "## Training Config",
        "",
        f"- `embedding_dim`: `{config['embedding_dim']}`",
        f"- `arcface_scale`: `{config['arcface_scale']}`",
        f"- `arcface_margin`: `{config['arcface_margin']}`",
        f"- `seed`: `{config['seed']}`",
        f"- `epochs`: `{config['epochs']}`",
        f"- `per_device_train_batch`: `{config['train_batch_size']}`",
        f"- `per_device_eval_batch`: `{config['eval_batch_size']}`",
        f"- `effective_batch_size`: `{config['effective_batch_size']}`",
        f"- `reference_backbone_lr / reference_head_lr`: `{config['backbone_lr']} / {config['head_lr']}`",
        f"- `resolved_backbone_lr / resolved_head_lr`: `{config['resolved_backbone_lr']} / {config['resolved_head_lr']}`",
        f"- `weight_decay`: `{config['weight_decay']}`",
        f"- `warmup_ratio`: `{config['warmup_ratio']}`",
        f"- `label_smoothing`: `{config['label_smoothing']}`",
        f"- `grad_clip_norm`: `{config['grad_clip_norm']}`",
        f"- `class_balanced_sampler`: `{config['class_balanced_sampler']}`",
        "",
        "## Best Epoch",
        "",
        f"- `best_epoch`: `{int(best_row['epoch'])}`",
        f"- `fit_recall_at_1`: `{float(best_row['fit_recall_at_1']):.6f}`",
        f"- `fit_recall_at_5`: `{float(best_row['fit_recall_at_5']):.6f}`",
        f"- `similarity_margin`: `{float(best_row['similarity_margin']):.6f}`",
        f"- `loss`: `{float(best_row['loss']):.6f}`",
        f"- `peak_cuda_memory_mb`: `{float(best_row['peak_cuda_memory_mb']):.2f}`",
        "",
        "## Resource Snapshot",
        "",
        f"- `device`: `{config['device']}`",
        f"- `selected_gpu`: `{selected_gpu}`",
        f"- `resource_decision`: `{config['resource_decision']}`",
        f"- `probe_reuse_note`: `{config['probe_reuse_note']}`",
        f"- `tmux_sessions_at_launch`: `{tmux_sessions}`",
        "",
    ]
    if plot_paths:
        lines.extend(["## Monitoring Figures", ""])
        for name, path in plot_paths.items():
            rel = Path(os.path.relpath(path, start=output_path.parent))
            lines.append(f"- `{name}`: `{rel.as_posix()}`")
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_texas_external_warmup(
    *,
    repo_root: Path,
    output_dir: Path,
    experiment_id: str,
    warmup_manifest_path: Path,
    student_backbone: str,
    device: str,
    epochs: int,
    embedding_dim: int,
    arcface_scale: float,
    arcface_margin: float,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    lr_reference_batch_size: int = 4,
    lr_scale_mode: str = "linear",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
    max_train_batches: int | None = None,
    goal: str | None = None,
    resource_decision: str | None = None,
    probe_reuse_note: str | None = None,
) -> dict[str, Path]:
    _require_torch()
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    embeddings_dir = output_dir / "embeddings"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    plots_dir = reports_dir / "plots"
    for path in [checkpoints_dir, embeddings_dir, tables_dir, reports_dir, plots_dir]:
        path.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    resource_snapshot = collect_resource_snapshot(device)

    manifest_df = pd.read_csv(warmup_manifest_path).copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["identity"] = manifest_df["identity"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    manifest_df[PATH_COLUMN] = manifest_df[PATH_COLUMN].astype(str)
    train_df, label_map, class_summary_df = attach_warmup_labels(manifest_df)
    train_df.to_csv(tables_dir / "warmup_manifest_v1.csv", index=False)
    class_summary_df.to_csv(tables_dir / "class_summary_v1.csv", index=False)

    backbone, backbone_spec = load_student_backbone(student_backbone, device=device)
    if train_batch_size is None:
        train_batch_size = int(backbone_spec.default_train_batch_size)
    if eval_batch_size is None:
        eval_batch_size = int(backbone_spec.default_eval_batch_size)
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

    model = TexasWarmupModel(
        backbone=backbone,
        feature_dim=backbone_spec.feature_dim,
        embedding_dim=embedding_dim,
        class_count=len(label_map),
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
    ).to(device)
    train_transform = build_train_transform(backbone_spec, dataset=TEXAS_DATASET)
    eval_transform = build_eval_transform(backbone_spec, dataset=TEXAS_DATASET)

    train_dataset = TexasWarmupDataset(df=train_df, repo_root=repo_root, transform=train_transform)
    sampler = build_warmup_sampler(train_df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    optimizer = build_warmup_optimizer(
        model=model,
        backbone_lr=resolved_backbone_lr,
        head_lr=resolved_head_lr,
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=max(1, min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)),
        warmup_ratio=warmup_ratio,
    )
    scaler = GradScaler(enabled=device.startswith("cuda"))

    training_rows: list[dict[str, object]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_checkpoint_path = checkpoints_dir / "best.pt"
    last_checkpoint_path = checkpoints_dir / "last.pt"

    for epoch in range(1, epochs + 1):
        epoch_metrics = train_warmup_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
            max_train_batches=max_train_batches,
        )
        fit_embeddings = extract_warmup_embeddings(
            df=train_df,
            repo_root=repo_root,
            model=model,
            transform=eval_transform,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        fit_embeddings = l2_normalize(fit_embeddings.astype(np.float32, copy=False))
        retrieval_metrics = evaluate_warmup_retrieval(train_df, fit_embeddings)
        epoch_row = {"epoch": epoch, **epoch_metrics, **retrieval_metrics}
        training_rows.append(epoch_row)

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "config": {
                "experiment_id": experiment_id,
                "dataset": TEXAS_DATASET,
                "warmup_manifest_path": str(warmup_manifest_path),
                "student_backbone": student_backbone,
                "student_model_id": backbone_spec.model_id,
                "student_feature_dim": backbone_spec.feature_dim,
                "input_size": backbone_spec.input_size,
                "embedding_dim": embedding_dim,
                "fit_class_count": len(label_map),
                "teacher_dim": 0,
                "classification_head": "arcface",
                "arcface_scale": arcface_scale,
                "arcface_margin": arcface_margin,
            },
        }
        torch.save(checkpoint_payload, last_checkpoint_path)

        current_key = (
            float(epoch_row["fit_recall_at_1"]),
            float(epoch_row["fit_recall_at_5"]),
            float(epoch_row["similarity_margin"]),
            -float(epoch_row["loss"]),
        )
        if best_key is None or current_key > best_key:
            best_key = current_key
            torch.save(checkpoint_payload, best_checkpoint_path)
            np.save(embeddings_dir / "train_embeddings.npy", fit_embeddings.astype(np.float32))
            train_df.to_csv(embeddings_dir / "train_metadata.csv", index=False)

    training_log_df = pd.DataFrame(training_rows)
    training_log_path = tables_dir / "training_log_v1.csv"
    training_log_df.to_csv(training_log_path, index=False)
    plot_paths = write_warmup_plots(plots_dir=plots_dir, training_log_df=training_log_df)

    class_counts = train_df.groupby("identity")["image_id"].size()
    config = {
        "experiment_id": experiment_id,
        "goal": goal or "Warm up the Texas student on external TCU chips before trusted-seed Texas self-train.",
        "dataset": TEXAS_DATASET,
        "manifest_path": str(warmup_manifest_path),
        "student_backbone": student_backbone,
        "student_model_id": backbone_spec.model_id,
        "input_size": backbone_spec.input_size,
        "train_augmentation": describe_transform(train_transform),
        "eval_preprocess": describe_transform(eval_transform),
        "images": int(len(train_df)),
        "classes": int(len(label_map)),
        "singleton_classes": int((class_counts == 1).sum()),
        "multi_image_classes": int((class_counts > 1).sum()),
        "embedding_dim": embedding_dim,
        "arcface_scale": arcface_scale,
        "arcface_margin": arcface_margin,
        "seed": seed,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "effective_batch_size": effective_batch_size,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "resolved_backbone_lr": resolved_backbone_lr,
        "resolved_head_lr": resolved_head_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "label_smoothing": label_smoothing,
        "grad_clip_norm": grad_clip_norm,
        "class_balanced_sampler": sampler is not None,
        "device": device,
        "resource_snapshot": resource_snapshot,
        "resource_decision": resource_decision
        or "TCU warmup runs on one GPU; the label pool is small enough that full-fit retrieval evaluation is cheap.",
        "probe_reuse_note": probe_reuse_note or "Warmup uses a class-balanced sampler because TCU IDs range from singleton to 6 images.",
    }
    summary_path = reports_dir / "summary.md"
    write_warmup_report(
        output_path=summary_path,
        config=config,
        class_summary_df=class_summary_df,
        training_log_df=training_log_df,
        plot_paths=plot_paths,
    )
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "student_backbone": student_backbone,
                "best_checkpoint_path": str(best_checkpoint_path),
                "last_checkpoint_path": str(last_checkpoint_path),
                "fit_recall_at_1_best": float(training_log_df["fit_recall_at_1"].max()) if not training_log_df.empty else 0.0,
                "fit_recall_at_5_best": float(training_log_df["fit_recall_at_5"].max()) if not training_log_df.empty else 0.0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": summary_path,
        "summary_json_path": summary_json_path,
        "training_log_path": training_log_path,
        "best_checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": last_checkpoint_path,
        "train_embeddings_path": embeddings_dir / "train_embeddings.npy",
        "train_metadata_path": embeddings_dir / "train_metadata.csv",
        "warmup_manifest_path": tables_dir / "warmup_manifest_v1.csv",
        "class_summary_path": tables_dir / "class_summary_v1.csv",
    }
