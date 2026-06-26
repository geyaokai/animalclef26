from __future__ import annotations

import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:  # pragma: no cover - exercised in the training env
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - allows lighter imports in tests
    plt = None

try:  # pragma: no cover - exercised in the training env
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
except ModuleNotFoundError:  # pragma: no cover - allows lighter imports in tests
    torch = None
    F = None

    class _NNProxy:
        Module = object
        Linear = object
        BatchNorm1d = object
        ModuleDict = dict
        Parameter = object

    nn = _NNProxy()
    DataLoader = Dataset = object
    T = None

from .descriptor_baselines import (
    DESCRIPTOR_SPECS,
    LABELED_DATASETS,
    PATH_COLUMN,
    _coerce_model_output,
    apply_thresholds_to_df,
    build_identity_holdout_split,
    build_neighbor_table,
    create_baseline_qualitative_outputs,
    dataframe_to_markdown_table,
    extract_embeddings,
    fuse_embedding_blocks,
    l2_normalize,
    load_descriptor_model,
    load_manifests,
    pick_best_thresholds,
    recall_at_k,
    run_threshold_sweep,
)
from .qualitative_lynx_views import clahe_normalize_gray, histogram_normalize_gray


DEFAULT_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]
DATASET_TO_INDEX = {dataset: index for index, dataset in enumerate(LABELED_DATASETS)}
INDEX_TO_DATASET = {index: dataset for dataset, index in DATASET_TO_INDEX.items()}

DATASET_PREPROCESS_CONFIG: dict[str, dict[str, Any]] = {
    "LynxID2025": {"mode": "gray_percentile_rgb", "low_percentile": 1.0, "high_percentile": 99.0},
    "TexasHornedLizards": {"mode": "gray_percentile_rgb", "low_percentile": 2.0, "high_percentile": 98.0},
}
SUPPORTED_PREPROCESS_MODES = {"identity_rgb", "gray_percentile_rgb", "hist_norm_rgb", "clahe_rgb"}

DATASET_AUGMENT_CONFIG: dict[str, dict[str, Any]] = {
    "LynxID2025": {
        "crop_scale": (0.80, 1.0),
        "crop_ratio": (0.9, 1.1),
        "hflip_p": 0.5,
        "rotation_deg": 5.0,
        "color_jitter": None,
        "random_erasing_p": 0.20,
    },
    "SalamanderID2025": {
        "crop_scale": (0.82, 1.0),
        "crop_ratio": (0.85, 1.15),
        "hflip_p": 0.5,
        "rotation_deg": 7.0,
        "color_jitter": {"brightness": 0.08, "contrast": 0.08, "saturation": 0.08, "hue": 0.02},
        "random_erasing_p": 0.20,
    },
    "SeaTurtleID2022": {
        "crop_scale": (0.85, 1.0),
        "crop_ratio": (0.9, 1.1),
        "hflip_p": 0.5,
        "rotation_deg": 5.0,
        "color_jitter": {"brightness": 0.08, "contrast": 0.08, "saturation": 0.08, "hue": 0.02},
        "random_erasing_p": 0.20,
    },
    "TexasHornedLizards": {
        "crop_scale": (0.88, 1.0),
        "crop_ratio": (0.9, 1.1),
        "hflip_p": 0.5,
        "rotation_deg": 10.0,
        "color_jitter": None,
        "random_erasing_p": 0.10,
    },
}


@dataclass(frozen=True)
class StudentBackboneSpec:
    name: str
    family: str
    model_id: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    feature_dim: int
    default_train_batch_size: int
    default_eval_batch_size: int


STUDENT_BACKBONE_SPECS = {
    "mega": StudentBackboneSpec(
        name="mega",
        family="descriptor",
        model_id=DESCRIPTOR_SPECS["mega"].model_id,
        input_size=DESCRIPTOR_SPECS["mega"].input_size,
        mean=DESCRIPTOR_SPECS["mega"].mean,
        std=DESCRIPTOR_SPECS["mega"].std,
        feature_dim=1536,
        default_train_batch_size=8,
        default_eval_batch_size=16,
    ),
    "miew": StudentBackboneSpec(
        name="miew",
        family="descriptor",
        model_id=DESCRIPTOR_SPECS["miew"].model_id,
        input_size=DESCRIPTOR_SPECS["miew"].input_size,
        mean=DESCRIPTOR_SPECS["miew"].mean,
        std=DESCRIPTOR_SPECS["miew"].std,
        feature_dim=2152,
        default_train_batch_size=6,
        default_eval_batch_size=12,
    ),
    "convnext": StudentBackboneSpec(
        name="convnext",
        family="timm",
        model_id="convnext_large.fb_in22k_ft_in1k_384",
        input_size=384,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        feature_dim=1536,
        default_train_batch_size=8,
        default_eval_batch_size=16,
    ),
}


@dataclass(frozen=True)
class TeacherCacheBundle:
    fit_embeddings: np.ndarray
    val_embeddings: np.ndarray
    component_table: pd.DataFrame
    cache_dir: Path


def _require_torch() -> None:
    if torch is None or T is None:
        raise ModuleNotFoundError("torch and torchvision are required for supervised training")


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for supervised training plots")


def seed_everything(seed: int) -> None:
    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scale_learning_rate(
    base_lr: float,
    effective_batch_size: int,
    reference_batch_size: int,
    mode: str,
) -> float:
    if reference_batch_size <= 0:
        raise ValueError("reference_batch_size must be positive")
    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    if mode == "none":
        return float(base_lr)
    scale = float(effective_batch_size) / float(reference_batch_size)
    if mode == "linear":
        return float(base_lr) * scale
    if mode == "sqrt":
        return float(base_lr) * math.sqrt(scale)
    raise ValueError(f"Unsupported lr scaling mode: {mode}")


def _normalize_gray_percentile_rgb(
    image: Image.Image,
    *,
    low_percentile: float,
    high_percentile: float,
) -> Image.Image:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    low_value = float(np.percentile(gray, float(low_percentile)))
    high_value = float(np.percentile(gray, float(high_percentile)))
    if high_value <= low_value + 1e-6:
        normalized = gray
    else:
        scaled = np.clip((gray.astype(np.float32) - low_value) / (high_value - low_value), 0.0, 1.0)
        normalized = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    gray_image = Image.fromarray(normalized, mode="L")
    return Image.merge("RGB", (gray_image, gray_image, gray_image))


def _merge_gray_to_rgb(gray: np.ndarray) -> Image.Image:
    gray_image = Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L")
    return Image.merge("RGB", (gray_image, gray_image, gray_image))


def resolve_dataset_preprocess_config(
    dataset_preprocess_overrides: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    resolved = {dataset: dict(config) for dataset, config in DATASET_PREPROCESS_CONFIG.items()}
    if not dataset_preprocess_overrides:
        return resolved
    for dataset, raw_mode in dataset_preprocess_overrides.items():
        mode = str(raw_mode).strip()
        if mode not in SUPPORTED_PREPROCESS_MODES:
            raise ValueError(f"Unsupported preprocess mode for {dataset}: {raw_mode}")
        if mode == "identity_rgb":
            resolved[dataset] = {"mode": mode}
        elif mode == "gray_percentile_rgb":
            default = DATASET_PREPROCESS_CONFIG.get(dataset, {})
            resolved[dataset] = {
                "mode": mode,
                "low_percentile": float(default.get("low_percentile", 1.0)),
                "high_percentile": float(default.get("high_percentile", 99.0)),
            }
        elif mode == "hist_norm_rgb":
            resolved[dataset] = {
                "mode": mode,
                "low_percentile": 1.0,
                "high_percentile": 99.0,
            }
        elif mode == "clahe_rgb":
            resolved[dataset] = {
                "mode": mode,
                "clip_limit": 2.0,
                "grid_size": 8,
            }
    return resolved


class DatasetSpecificPreprocess:
    def __init__(
        self,
        dataset: str | None,
        stage: str,
        preprocess_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.dataset = str(dataset) if dataset else None
        self.stage = str(stage)
        self.preprocess_config = preprocess_config if preprocess_config is not None else DATASET_PREPROCESS_CONFIG

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.dataset is None:
            return image.convert("RGB")
        config = self.preprocess_config.get(self.dataset)
        if config is None:
            return image.convert("RGB")
        mode = str(config.get("mode", "identity_rgb"))
        if mode == "identity_rgb":
            return image.convert("RGB")
        if mode == "gray_percentile_rgb":
            return _normalize_gray_percentile_rgb(
                image.convert("RGB"),
                low_percentile=float(config["low_percentile"]),
                high_percentile=float(config["high_percentile"]),
            )
        if mode == "hist_norm_rgb":
            return _merge_gray_to_rgb(
                histogram_normalize_gray(
                    image.convert("RGB"),
                    low_percentile=float(config.get("low_percentile", 1.0)),
                    high_percentile=float(config.get("high_percentile", 99.0)),
                )
            )
        if mode == "clahe_rgb":
            return _merge_gray_to_rgb(
                clahe_normalize_gray(
                    image.convert("RGB"),
                    clip_limit=float(config.get("clip_limit", 2.0)),
                    grid_size=int(config.get("grid_size", 8)),
                )
            )
        return image.convert("RGB")

    def describe(self) -> str:
        if self.dataset is None:
            return "DatasetSpecificPreprocess(dataset=None)"
        config = self.preprocess_config.get(self.dataset)
        if config is None:
            return f"DatasetSpecificPreprocess(dataset={self.dataset}, mode=identity)"
        return (
            f"DatasetSpecificPreprocess(dataset={self.dataset}, stage={self.stage}, "
            f"mode={config.get('mode')}, low={config.get('low_percentile')}, high={config.get('high_percentile')})"
        )


def _resolve_dataset_augment_config(dataset: str | None) -> dict[str, Any]:
    base = {
        "crop_scale": (0.80, 1.0),
        "crop_ratio": (0.9, 1.1),
        "hflip_p": 0.5,
        "rotation_deg": 5.0,
        "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.02},
        "random_erasing_p": 0.25,
    }
    if dataset is None:
        return base
    return {**base, **DATASET_AUGMENT_CONFIG.get(str(dataset), {})}


def build_train_transform(
    spec: StudentBackboneSpec,
    dataset: str | None = None,
    preprocess_config: dict[str, dict[str, Any]] | None = None,
) -> T.Compose:
    _require_torch()
    config = _resolve_dataset_augment_config(dataset)
    transforms: list[Any] = [
        DatasetSpecificPreprocess(dataset=dataset, stage="train", preprocess_config=preprocess_config),
        T.RandomResizedCrop(spec.input_size, scale=tuple(config["crop_scale"]), ratio=tuple(config["crop_ratio"])),
    ]
    if float(config.get("hflip_p", 0.0)) > 0:
        transforms.append(T.RandomHorizontalFlip(p=float(config["hflip_p"])))
    if float(config.get("rotation_deg", 0.0)) > 0:
        transforms.append(T.RandomRotation(degrees=float(config["rotation_deg"])))
    color_jitter = config.get("color_jitter")
    if color_jitter:
        transforms.append(
            T.ColorJitter(
                brightness=float(color_jitter.get("brightness", 0.0)),
                contrast=float(color_jitter.get("contrast", 0.0)),
                saturation=float(color_jitter.get("saturation", 0.0)),
                hue=float(color_jitter.get("hue", 0.0)),
            )
        )
    transforms.extend(
        [
            T.ToTensor(),
            T.Normalize(mean=spec.mean, std=spec.std),
        ]
    )
    if float(config.get("random_erasing_p", 0.0)) > 0:
        transforms.append(T.RandomErasing(p=float(config["random_erasing_p"]), scale=(0.02, 0.1), value="random"))
    return T.Compose(transforms)


def build_eval_transform(
    spec: StudentBackboneSpec,
    dataset: str | None = None,
    preprocess_config: dict[str, dict[str, Any]] | None = None,
) -> T.Compose:
    _require_torch()
    return T.Compose(
        [
            DatasetSpecificPreprocess(dataset=dataset, stage="eval", preprocess_config=preprocess_config),
            T.Resize((spec.input_size, spec.input_size)),
            T.ToTensor(),
            T.Normalize(mean=spec.mean, std=spec.std),
        ]
    )


def attach_training_labels(fit_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]], pd.DataFrame]:
    labeled = fit_df.copy().reset_index(drop=True)
    label_maps: dict[str, dict[str, int]] = {}
    summary_rows: list[dict[str, object]] = []
    labeled["label_index"] = -1
    labeled["dataset_index"] = labeled["dataset"].map(DATASET_TO_INDEX).astype(int)
    labeled["identity_image_count_fit"] = (
        labeled.groupby(["dataset", "identity"], dropna=False)["image_id"].transform("size").astype(int)
    )
    global_identity_offset = 0
    labeled["global_label_index"] = -1

    datasets_in_fit = sorted(labeled["dataset"].dropna().astype(str).unique().tolist())
    for dataset in datasets_in_fit:
        dataset_mask = labeled["dataset"] == dataset
        dataset_identities = sorted(labeled.loc[dataset_mask, "identity"].unique().tolist())
        mapping = {identity: index for index, identity in enumerate(dataset_identities)}
        label_maps[dataset] = mapping
        labeled.loc[dataset_mask, "label_index"] = labeled.loc[dataset_mask, "identity"].map(mapping).astype(int)
        labeled.loc[dataset_mask, "global_label_index"] = labeled.loc[dataset_mask, "label_index"] + global_identity_offset
        summary_rows.append(
            {
                "dataset": dataset,
                "classes": len(mapping),
                "images": int(dataset_mask.sum()),
                "singleton_classes": int(
                    (
                        labeled.loc[dataset_mask]
                        .groupby("identity", dropna=False)
                        .size()
                        .eq(1)
                        .sum()
                    )
                ),
            }
        )
        global_identity_offset += len(mapping)

    class_summary_df = pd.DataFrame(summary_rows)
    return labeled, label_maps, class_summary_df


class SupervisedTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        repo_root: Path,
        spec: StudentBackboneSpec,
        teacher_embeddings: np.ndarray,
        preprocess_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        datasets = sorted(self.df["dataset"].astype(str).unique().tolist())
        self.base_transforms = {
            dataset: build_eval_transform(spec, dataset=dataset, preprocess_config=preprocess_config)
            for dataset in datasets
        }
        self.aug_transforms = {
            dataset: build_train_transform(spec, dataset=dataset, preprocess_config=preprocess_config)
            for dataset in datasets
        }
        self.teacher_embeddings = teacher_embeddings.astype(np.float32, copy=False)
        if len(self.df) != len(self.teacher_embeddings):
            raise ValueError(
                f"Teacher embeddings row mismatch: df={len(self.df)} vs embeddings={len(self.teacher_embeddings)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        dataset = str(row["dataset"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            base_tensor = self.base_transforms[dataset](image)
            aug_tensor = self.aug_transforms[dataset](image)
        return {
            "base_image": base_tensor,
            "aug_image": aug_tensor,
            "dataset_index": int(row["dataset_index"]),
            "label_index": int(row["label_index"]),
            "global_label_index": int(row["global_label_index"]),
            "identity_image_count_fit": int(row["identity_image_count_fit"]),
            "teacher_embedding": torch.from_numpy(self.teacher_embeddings[index]),
        }


class InferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        repo_root: Path,
        spec: StudentBackboneSpec,
        preprocess_config: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        datasets = sorted(self.df["dataset"].astype(str).unique().tolist())
        self.transforms = {
            dataset: build_eval_transform(spec, dataset=dataset, preprocess_config=preprocess_config)
            for dataset in datasets
        }

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        dataset = str(row["dataset"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transforms[dataset](image)


class ArcFaceHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.3,
        subcenters: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.subcenters = subcenters
        self.weight = nn.Parameter(torch.empty(out_features * subcenters, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def _compute_cosine(self, embeddings: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings, dim=1)
        normalized_weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(normalized_embeddings, normalized_weight)
        if self.subcenters == 1:
            return cosine
        cosine = cosine.view(-1, self.out_features, self.subcenters)
        cosine, _ = torch.max(cosine, dim=2)
        return cosine

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = self._compute_cosine(embeddings)
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels.long(), num_classes=self.out_features).to(dtype=cosine.dtype)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale


class SupervisedEmbeddingModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        embedding_dim: int,
        dataset_class_counts: dict[str, int],
        teacher_dim: int,
        arcface_scale: float,
        arcface_margin: float,
        salamander_subcenter_k: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.arcface_heads = nn.ModuleDict()
        for dataset, class_count in dataset_class_counts.items():
            subcenters = salamander_subcenter_k if dataset == "SalamanderID2025" and salamander_subcenter_k > 1 else 1
            self.arcface_heads[dataset] = ArcFaceHead(
                in_features=embedding_dim,
                out_features=class_count,
                scale=arcface_scale,
                margin=arcface_margin,
                subcenters=subcenters,
            )
        self.teacher_projection = nn.Linear(embedding_dim, teacher_dim, bias=False) if teacher_dim > 0 else None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
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


def load_student_backbone(name: str, device: str) -> tuple[nn.Module, StudentBackboneSpec]:
    _require_torch()
    spec = STUDENT_BACKBONE_SPECS[name]
    if spec.family == "descriptor":
        backbone, _descriptor_spec = load_descriptor_model(name, device=device)
        backbone.train()
        return backbone, spec
    if spec.family == "timm":
        import timm

        backbone = timm.create_model(spec.model_id, pretrained=True, num_classes=0)
        backbone = backbone.to(device)
        backbone.train()
        return backbone, spec
    raise ValueError(f"Unsupported backbone family: {spec.family}")


def load_matching_checkpoint_weights(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    scope: str = "encoder",
) -> dict[str, object]:
    _require_torch()
    if scope not in {"encoder", "all_matching"}:
        raise ValueError(f"Unsupported init checkpoint scope: {scope}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {checkpoint_path}")
    model_state = model.state_dict()
    allowed_prefixes = ("backbone.", "embedding_layer.", "embedding_bn.")
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    filtered_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if scope == "encoder" and not key.startswith(allowed_prefixes):
            skipped_keys.append(key)
            continue
        if key not in model_state:
            skipped_keys.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            skipped_keys.append(key)
            continue
        filtered_state[key] = value
        loaded_keys.append(key)
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    return {
        "checkpoint_path": str(checkpoint_path),
        "scope": scope,
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "missing_key_count": len(missing_keys),
        "unexpected_key_count": len(unexpected_keys),
        "loaded_key_examples": loaded_keys[:20],
        "skipped_key_examples": skipped_keys[:20],
    }


def replace_with_hardlink(source_path: Path, target_path: Path) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        os.link(source_path, target_path)
    except OSError:
        os.symlink(source_path.name, target_path)


def maybe_limit_rows(df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.iloc[:limit].reset_index(drop=True)


def compute_or_load_teacher_cache(
    repo_root: Path,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    teacher_sources: list[str],
    cache_dir: Path,
    device: str,
    num_workers: int,
) -> TeacherCacheBundle:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fit_metadata_path = cache_dir / "fit_metadata.csv"
    val_metadata_path = cache_dir / "val_metadata.csv"
    fit_embeddings_path = cache_dir / "fit_teacher_embeddings.npy"
    val_embeddings_path = cache_dir / "val_teacher_embeddings.npy"
    component_path = cache_dir / "teacher_components_v1.csv"

    if all(path.exists() for path in [fit_metadata_path, val_metadata_path, fit_embeddings_path, val_embeddings_path, component_path]):
        cached_fit_df = pd.read_csv(fit_metadata_path)
        cached_val_df = pd.read_csv(val_metadata_path)
        for frame in [cached_fit_df, cached_val_df]:
            frame["image_id"] = frame["image_id"].astype(str)
            frame["identity"] = frame["identity"].fillna("").astype(str)
            frame["dataset"] = frame["dataset"].astype(str)
            frame[PATH_COLUMN] = frame[PATH_COLUMN].astype(str)
        reference_fit = fit_df[["image_id", "dataset", "identity", PATH_COLUMN]].reset_index(drop=True)
        reference_val = val_df[["image_id", "dataset", "identity", PATH_COLUMN]].reset_index(drop=True)
        cached_fit = cached_fit_df[["image_id", "dataset", "identity", PATH_COLUMN]].reset_index(drop=True)
        cached_val = cached_val_df[["image_id", "dataset", "identity", PATH_COLUMN]].reset_index(drop=True)
        if reference_fit.equals(cached_fit) and reference_val.equals(cached_val):
            print(f"[supervised_training] reusing teacher cache: {cache_dir}", flush=True)
            return TeacherCacheBundle(
                fit_embeddings=np.load(fit_embeddings_path).astype(np.float32),
                val_embeddings=np.load(val_embeddings_path).astype(np.float32),
                component_table=pd.read_csv(component_path),
                cache_dir=cache_dir,
            )

    print(
        f"[supervised_training] building teacher cache | sources={teacher_sources} | cache_dir={cache_dir}",
        flush=True,
    )
    fit_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    component_rows: list[dict[str, object]] = []

    for source in teacher_sources:
        print(f"[supervised_training] teacher source start | source={source} | device={device}", flush=True)
        model, spec = load_descriptor_model(source, device=device)
        fit_embeddings = extract_embeddings(
            df=fit_df,
            repo_root=repo_root,
            model=model,
            spec=spec,
            device=device,
            batch_size=spec.default_batch_size,
            num_workers=num_workers,
        )
        val_embeddings = extract_embeddings(
            df=val_df,
            repo_root=repo_root,
            model=model,
            spec=spec,
            device=device,
            batch_size=spec.default_batch_size,
            num_workers=num_workers,
        )
        fit_blocks.append(fit_embeddings)
        val_blocks.append(val_embeddings)
        component_rows.append(
            {
                "teacher_source": source,
                "model_id": spec.model_id,
                "fit_dim": int(fit_embeddings.shape[1]),
                "val_dim": int(val_embeddings.shape[1]),
            }
        )
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[supervised_training] teacher source done | source={source} | fit_shape={fit_embeddings.shape} | val_shape={val_embeddings.shape}",
            flush=True,
        )

    fused_fit = fuse_embedding_blocks(fit_blocks, weights=[1.0] * len(fit_blocks))
    fused_val = fuse_embedding_blocks(val_blocks, weights=[1.0] * len(val_blocks))

    fit_df.to_csv(fit_metadata_path, index=False)
    val_df.to_csv(val_metadata_path, index=False)
    np.save(fit_embeddings_path, fused_fit.astype(np.float32))
    np.save(val_embeddings_path, fused_val.astype(np.float32))
    component_table = pd.DataFrame(component_rows)
    component_table.to_csv(component_path, index=False)
    print(
        f"[supervised_training] teacher cache ready | fit_shape={fused_fit.shape} | val_shape={fused_val.shape}",
        flush=True,
    )
    return TeacherCacheBundle(
        fit_embeddings=fused_fit.astype(np.float32),
        val_embeddings=fused_val.astype(np.float32),
        component_table=component_table,
        cache_dir=cache_dir,
    )


def compute_arcface_loss(
    model: SupervisedEmbeddingModel,
    embeddings: torch.Tensor,
    dataset_indices: torch.Tensor,
    label_indices: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    total_weight = 0
    total_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    for dataset in model.arcface_heads.keys():
        dataset_index = DATASET_TO_INDEX[dataset]
        mask = dataset_indices == dataset_index
        if not torch.any(mask):
            continue
        logits = model.arcface_heads[dataset](embeddings[mask], label_indices[mask])
        loss = F.cross_entropy(logits, label_indices[mask], label_smoothing=label_smoothing)
        weight = int(mask.sum().item())
        total_loss = total_loss + (loss * weight)
        total_weight += weight
    if total_weight == 0:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    return total_loss / total_weight


def compute_feature_distillation_loss(
    projected_embeddings: torch.Tensor | None,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if projected_embeddings is None:
        return torch.zeros((), device=teacher_embeddings.device, dtype=teacher_embeddings.dtype)
    cosine = F.cosine_similarity(projected_embeddings, teacher_embeddings, dim=1)
    return (1.0 - cosine).mean()


def compute_relation_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
    dataset_indices: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for dataset_index in dataset_indices.unique(sorted=True):
        mask = dataset_indices == dataset_index
        if int(mask.sum().item()) < 2:
            continue
        student_block = student_embeddings[mask]
        teacher_block = teacher_embeddings[mask]
        student_similarity = student_block @ student_block.T
        teacher_similarity = teacher_block @ teacher_block.T
        losses.append(F.mse_loss(student_similarity, teacher_similarity))
    if not losses:
        return torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)
    return torch.stack(losses).mean()


def compute_masked_supcon_loss(
    embeddings: torch.Tensor,
    dataset_indices: torch.Tensor,
    global_label_indices: torch.Tensor,
    identity_image_counts_fit: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    logits = (embeddings @ embeddings.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye_mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
    same_dataset = dataset_indices[:, None] == dataset_indices[None, :]
    same_identity = global_label_indices[:, None] == global_label_indices[None, :]
    eligible_identity = identity_image_counts_fit >= 2
    eligible_mask = eligible_identity[:, None] & eligible_identity[None, :]
    positive_mask = same_dataset & same_identity & eligible_mask & (~eye_mask)
    denominator_mask = same_dataset & (~eye_mask)

    if not torch.any(positive_mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    logits_exp = torch.exp(logits) * denominator_mask.to(dtype=embeddings.dtype)
    log_prob = logits - torch.log(torch.clamp(logits_exp.sum(dim=1, keepdim=True), min=1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid_anchor_mask = positive_count > 0
    if not torch.any(valid_anchor_mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    mean_log_prob_positive = (log_prob * positive_mask.to(dtype=embeddings.dtype)).sum(dim=1) / positive_count.clamp(min=1)
    return (-mean_log_prob_positive[valid_anchor_mask]).mean()


def compute_view_pair_contrastive_loss(
    *,
    base_embeddings: torch.Tensor,
    augmented_embeddings: torch.Tensor,
    dataset_indices: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if base_embeddings.shape != augmented_embeddings.shape:
        raise ValueError(
            f"base/aug embedding shape mismatch: {tuple(base_embeddings.shape)} vs {tuple(augmented_embeddings.shape)}"
        )
    if base_embeddings.shape[0] == 0:
        return torch.zeros((), device=base_embeddings.device, dtype=base_embeddings.dtype)
    losses: list[torch.Tensor] = []
    for dataset_index in dataset_indices.unique(sorted=True):
        mask = dataset_indices == dataset_index
        if not torch.any(mask):
            continue
        base_block = base_embeddings[mask]
        aug_block = augmented_embeddings[mask]
        logits = (base_block @ aug_block.T) / max(float(temperature), 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
    if not losses:
        return torch.zeros((), device=base_embeddings.device, dtype=base_embeddings.dtype)
    return torch.stack(losses).mean()


def extract_student_embeddings(
    df: pd.DataFrame,
    repo_root: Path,
    model: Any,
    spec: StudentBackboneSpec,
    device: str,
    batch_size: int,
    num_workers: int,
    horizontal_flip_tta: bool = False,
    preprocess_config: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    _require_torch()
    dataset = InferenceDataset(df=df, repo_root=repo_root, spec=spec, preprocess_config=preprocess_config)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            embeddings = model.encode(images)
            if horizontal_flip_tta:
                flipped_embeddings = model.encode(torch.flip(images, dims=[3]))
                embeddings = F.normalize((embeddings + flipped_embeddings) * 0.5, dim=1)
            rows.append(embeddings.detach().cpu().numpy().astype(np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return l2_normalize(np.concatenate(rows, axis=0))


def evaluate_validation_embeddings(
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sweep_df, _prediction_df = run_threshold_sweep(df=val_df, embeddings=val_embeddings, thresholds=thresholds)
    best_thresholds_df = pick_best_thresholds(sweep_df)
    recall_rows: list[dict[str, object]] = []
    for dataset in sorted(val_df["dataset"].unique()):
        dataset_mask = val_df["dataset"] == dataset
        dataset_embeddings = val_embeddings[dataset_mask]
        dataset_labels = val_df.loc[dataset_mask, "identity"].to_numpy()
        recall_rows.append(
            {
                "dataset": dataset,
                "recall_at_1": recall_at_k(dataset_embeddings, dataset_labels, k=1),
                "recall_at_5": recall_at_k(dataset_embeddings, dataset_labels, k=5),
            }
        )
    recall_df = pd.DataFrame(recall_rows)
    return sweep_df, best_thresholds_df, recall_df


def summarize_alignment(
    student_embeddings: np.ndarray,
    teacher_embeddings: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    def summarize_block(student_block: np.ndarray, teacher_block: np.ndarray) -> dict[str, float]:
        if len(student_block) < 2:
            return {
                "relation_mse": 0.0,
                "relation_mae": 0.0,
                "relation_corr": 0.0,
            }
        student_similarity = student_block @ student_block.T
        teacher_similarity = teacher_block @ teacher_block.T
        tri = np.triu_indices(len(student_block), k=1)
        student_values = student_similarity[tri]
        teacher_values = teacher_similarity[tri]
        delta = student_values - teacher_values
        if len(student_values) > 1 and np.std(student_values) > 0 and np.std(teacher_values) > 0:
            corr = float(np.corrcoef(student_values, teacher_values)[0, 1])
        else:
            corr = 0.0
        return {
            "relation_mse": round(float(np.mean(delta**2)), 6),
            "relation_mae": round(float(np.mean(np.abs(delta))), 6),
            "relation_corr": round(corr, 6),
        }

    rows: list[dict[str, object]] = []
    overall_summary = summarize_block(student_embeddings, teacher_embeddings)
    rows.append(
        {
            "dataset": "ALL_VAL",
            "samples": int(len(metadata_df)),
            **overall_summary,
        }
    )
    for dataset in sorted(metadata_df["dataset"].unique()):
        mask = metadata_df["dataset"] == dataset
        dataset_summary = summarize_block(student_embeddings[mask], teacher_embeddings[mask])
        rows.append(
            {
                "dataset": dataset,
                "samples": int(mask.sum()),
                **dataset_summary,
            }
        )
    return pd.DataFrame(rows)


def describe_transform(transform: T.Compose) -> list[str]:
    descriptions: list[str] = []
    for step in transform.transforms:
        name = type(step).__name__
        if hasattr(step, "describe"):
            descriptions.append(str(step.describe()))
        elif name == "RandomResizedCrop":
            descriptions.append(
                f"RandomResizedCrop(size={tuple(step.size) if isinstance(step.size, tuple) else step.size}, scale={tuple(step.scale)})"
            )
        elif name == "ColorJitter":
            descriptions.append(
                "ColorJitter("
                f"brightness={step.brightness}, contrast={step.contrast}, saturation={step.saturation}, hue={step.hue})"
            )
        elif name == "Normalize":
            descriptions.append(f"Normalize(mean={tuple(step.mean)}, std={tuple(step.std)})")
        elif name == "RandomErasing":
            descriptions.append(f"RandomErasing(p={step.p}, scale={tuple(step.scale)})")
        elif name == "RandomHorizontalFlip":
            descriptions.append(f"RandomHorizontalFlip(p={step.p})")
        elif name == "RandomRotation":
            descriptions.append(f"RandomRotation(degrees={step.degrees})")
        elif name == "Resize":
            descriptions.append(f"Resize(size={tuple(step.size) if isinstance(step.size, tuple) else step.size})")
        else:
            descriptions.append(name)
    return descriptions


def build_head_shape_summary(model: "SupervisedEmbeddingModel") -> dict[str, str]:
    summary: dict[str, str] = {}
    for dataset, head in model.arcface_heads.items():
        if head.subcenters > 1:
            summary[dataset] = f"{head.out_features} x {head.subcenters} x {head.in_features}"
        else:
            summary[dataset] = f"{head.out_features} x {head.in_features}"
    return summary


def collect_resource_snapshot(device: str) -> dict[str, object]:
    snapshot: dict[str, object] = {
        "device": device,
        "world_size": 1,
        "gradient_accumulation_steps": 1,
    }
    if device.startswith("cuda"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            gpu_rows: list[dict[str, object]] = []
            for line in result.stdout.strip().splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 6:
                    continue
                gpu_rows.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(parts[2]),
                        "memory_used_mb": int(parts[3]),
                        "memory_free_mb": int(parts[4]),
                        "utilization_gpu_pct": int(parts[5]),
                    }
                )
            snapshot["gpus"] = gpu_rows
            if ":" in device:
                selected_gpu_index = int(device.split(":", 1)[1])
                snapshot["selected_gpu_index"] = selected_gpu_index
                for row in gpu_rows:
                    if int(row["index"]) == selected_gpu_index:
                        snapshot["selected_gpu"] = row
                        break
        except (FileNotFoundError, ValueError):
            snapshot["gpus"] = []
    try:
        tmux_result = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True,
            check=False,
        )
        snapshot["tmux_sessions"] = [line.strip() for line in tmux_result.stdout.splitlines() if line.strip()]
    except FileNotFoundError:
        snapshot["tmux_sessions"] = []
    return snapshot


def resolve_training_datasets(
    requested_datasets: list[str] | None,
    available_datasets: list[str],
) -> list[str]:
    available = [dataset for dataset in available_datasets if dataset in LABELED_DATASETS]
    if requested_datasets is None:
        return sorted(available)
    normalized: list[str] = []
    seen: set[str] = set()
    for dataset in requested_datasets:
        dataset_name = str(dataset)
        if dataset_name in seen:
            continue
        seen.add(dataset_name)
        normalized.append(dataset_name)
    invalid = [dataset for dataset in normalized if dataset not in LABELED_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported supervised-training datasets: {invalid}; expected subset of {LABELED_DATASETS}")
    missing = [dataset for dataset in normalized if dataset not in available]
    if missing:
        raise ValueError(f"Requested datasets missing from labeled train manifest: {missing}")
    if not normalized:
        raise ValueError("datasets must be a non-empty subset of labeled datasets")
    return normalized


def build_best_metric_summary(training_log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric_specs: list[tuple[str, str]] = [("macro", "macro_ari")]
    metric_specs.extend((dataset, f"{dataset}_ari") for dataset in LABELED_DATASETS if f"{dataset}_ari" in training_log_df.columns)
    for selection_key, metric_column in metric_specs:
        if metric_column not in training_log_df.columns or training_log_df.empty:
            continue
        best_row = training_log_df.sort_values([metric_column, "epoch"], ascending=[False, True]).iloc[0]
        checkpoint_name = "best_macro.pt" if selection_key == "macro" else f"best_{selection_key}.pt"
        rows.append(
            {
                "selection_key": selection_key,
                "metric_column": metric_column,
                "best_epoch": int(best_row["epoch"]),
                "best_value": round(float(best_row[metric_column]), 6),
                "checkpoint_name": checkpoint_name,
            }
        )
    return pd.DataFrame(rows)


def _select_checkpoint_row(training_log_df: pd.DataFrame, checkpoint_key: str) -> pd.Series:
    if training_log_df.empty:
        raise ValueError("training_log_df must be non-empty")
    if checkpoint_key == "best_ari":
        sort_columns = ["macro_ari"]
        ascending = [False]
        if "macro_recall_at_1" in training_log_df.columns:
            sort_columns.append("macro_recall_at_1")
            ascending.append(False)
        sort_columns.append("epoch")
        ascending.append(True)
        return training_log_df.sort_values(sort_columns, ascending=ascending).iloc[0]
    if checkpoint_key == "best_recall1":
        if "macro_recall_at_1" not in training_log_df.columns:
            raise ValueError("macro_recall_at_1 is required to select best_recall1 checkpoint")
        sort_columns = ["macro_recall_at_1", "macro_ari", "epoch"]
        ascending = [False, False, True]
        return training_log_df.sort_values(sort_columns, ascending=ascending).iloc[0]
    if checkpoint_key == "last":
        return training_log_df.sort_values("epoch", ascending=True).iloc[-1]
    raise ValueError(f"Unsupported checkpoint selection key: {checkpoint_key}")


def build_checkpoint_comparison_summary(training_log_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for checkpoint_key, checkpoint_name, selection_metric in [
        ("best_ari", "best_ari.pt", "macro_ari"),
        ("best_recall1", "best_recall1.pt", "macro_recall_at_1"),
        ("last", "last.pt", "epoch"),
    ]:
        row = _select_checkpoint_row(training_log_df, checkpoint_key=checkpoint_key)
        rows.append(
            {
                "checkpoint_key": checkpoint_key,
                "checkpoint_name": checkpoint_name,
                "selection_metric": selection_metric,
                "selection_value": round(float(row[selection_metric]), 6),
                "epoch": int(row["epoch"]),
                "macro_ari": round(float(row["macro_ari"]), 6),
                "macro_recall_at_1": round(float(row.get("macro_recall_at_1", 0.0)), 6),
                "peak_cuda_memory_mb": round(float(row.get("peak_cuda_memory_mb", 0.0)), 2),
            }
        )
    return pd.DataFrame(rows)


def write_supervised_training_report(
    output_path: Path,
    config: dict[str, object],
    training_log_df: pd.DataFrame,
    best_metric_df: pd.DataFrame,
    checkpoint_comparison_df: pd.DataFrame,
    component_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    plot_paths: dict[str, Path] | None = None,
    status: str = "completed",
) -> None:
    best_row = training_log_df.sort_values(["macro_ari", "epoch"], ascending=[False, True]).iloc[0]
    resource_snapshot = config.get("resource_snapshot", {})
    selected_gpu = resource_snapshot.get("selected_gpu", {})
    tmux_sessions = resource_snapshot.get("tmux_sessions", [])
    qualitative_dir = output_path.parent.parent / "qualitative"
    lines = [
        "# Supervised Training Summary",
        "",
        "## Experiment Card",
        "",
        f"- `experiment_id`: `{config['experiment_id']}`",
        f"- `status`: `{status}`",
        f"- `goal`: `{config['goal']}`",
        f"- `student_backbone`: `{config['student_backbone']}`",
        f"- `student_model_id`: `{config['student_model_id']}`",
        f"- `teacher_sources`: `{config['teacher_sources']}`",
        "",
        "## Data Split",
        "",
        f"- `split_protocol`: `identity-level holdout`",
        f"- `datasets`: `{config['datasets']}`",
        f"- `val_identity_fraction`: `{config['val_identity_fraction']}`",
        f"- `seed`: `{config['split_seed']}`",
        f"- `fit_ids / fit_images`: `{config['fit_ids']} / {config['fit_images']}`",
        f"- `val_ids / val_images`: `{config['val_ids']} / {config['val_images']}`",
        f"- `validation_singleton_policy`: `{config['validation_singleton_policy']}`",
        f"- `fit_classes_by_dataset`: `{config['fit_classes']}`",
        "",
        "## Training Config",
        "",
        f"- `input_size`: `{config['input_size']}`",
        f"- `student_feature_shape`: `B x {config['student_feature_dim']}`",
        f"- `student_embedding_shape`: `B x {config['embedding_dim']}`",
        f"- `teacher_fused_shape`: `B x {config['teacher_dim']}`",
        f"- `train_augmentation`: `{config['train_augmentation']}`",
        f"- `eval_preprocess`: `{config['eval_preprocess']}`",
        f"- `head_shapes`: `{config['head_shapes']}`",
        f"- `per_device_train_batch`: `{config['train_batch_size']}`",
        f"- `per_device_eval_batch`: `{config['eval_batch_size']}`",
        f"- `gradient_accumulation_steps`: `{config['gradient_accumulation_steps']}`",
        f"- `world_size`: `{config['world_size']}`",
        f"- `effective_batch_size`: `{config['effective_batch_size']}`",
        f"- `optimizer`: `AdamW`",
        f"- `reference_batch_size`: `{config['lr_reference_batch_size']}`",
        f"- `reference_backbone_lr / reference_head_lr`: `{config['backbone_lr']} / {config['head_lr']}`",
        f"- `lr_scaling`: `{config['lr_scale_mode']}`",
        f"- `resolved_backbone_lr / resolved_head_lr`: `{config['resolved_backbone_lr']} / {config['resolved_head_lr']}`",
        f"- `weight_decay`: `{config['weight_decay']}`",
        f"- `scheduler`: `linear warmup + cosine decay`",
        f"- `warmup_ratio`: `{config['warmup_ratio']}`",
        f"- `epochs`: `{config['epochs']}`",
        f"- `amp_enabled`: `{config['amp_enabled']}`",
        f"- `grad_clip_norm`: `{config['grad_clip_norm']}`",
        f"- `losses`: `ArcFace + relation distill + feature distill + masked SupCon(optional)`",
        f"- `loss_weights`: `arcface=1.0, relation={config['relation_distill_weight']}, feature={config['feature_distill_weight']}, supcon={config['supcon_weight']}`",
        f"- `teacher_cache_dir`: `{config['teacher_cache_dir']}`",
        "",
        "## Best Result",
        "",
        f"- `best_epoch`: `{int(best_row['epoch'])}`",
        f"- `best_macro_ari`: `{float(best_row['macro_ari']):.4f}`",
        f"- `best_macro_recall_at_1`: `{float(best_row['macro_recall_at_1']):.4f}`",
        f"- `peak_cuda_memory_mb`: `{float(best_row['peak_cuda_memory_mb']):.2f}`",
        f"- `best_checkpoint_alias`: `checkpoints/best.pt -> checkpoints/best_macro.pt`",
        "",
        "## Checkpoint Policy",
        "",
        "- `best_ari.pt` 按 `macro_ari` 选最优 epoch，并同步写出兼容别名 `best.pt` / `best_macro.pt`。",
        "- `best_recall1.pt` 按 `macro_recall_at_1` 选最优 epoch，便于对比检索稳定性。",
        "- `last.pt` 始终保存最后一个 epoch，便于恢复训练或复盘收敛末态。",
        "- 另外为每个参与训练的 labeled dataset 各保存一份 `best_<dataset>.pt`，用于 dataset-wise route 复用单数据集最佳 epoch。",
        "",
        "## Checkpoint Comparison",
        "",
        dataframe_to_markdown_table(checkpoint_comparison_df),
        "",
        "## Best Checkpoints By Metric",
        "",
        dataframe_to_markdown_table(best_metric_df),
        "",
        "## Teacher Components",
        "",
        dataframe_to_markdown_table(component_df),
        "",
        "## Resource Decision",
        "",
        f"- `device`: `{config['device']}`",
        f"- `selected_gpu`: `{selected_gpu}`",
        f"- `resource_decision`: `{config['resource_decision']}`",
        f"- `probe_reuse_note`: `{config['probe_reuse_note']}`",
        f"- `tmux_sessions_at_launch`: `{tmux_sessions}`",
        "",
        "## Best Validation Thresholds",
        "",
        dataframe_to_markdown_table(best_thresholds_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]),
        "",
        "## Validation Recall",
        "",
        dataframe_to_markdown_table(recall_df),
        "",
        "## Teacher / Student Alignment",
        "",
        dataframe_to_markdown_table(alignment_df),
        "",
    ]
    if plot_paths:
        lines.extend(
            [
                "## Monitoring Figures",
                "",
            ]
        )
        if "loss" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["loss"], start=output_path.parent))
            lines.extend(
                [
                    f"![Training loss curves]({rel.as_posix()})",
                    "",
                    "- 读图方式：横轴是 epoch。上半图先看总 `train_loss` 是否稳定下降；下半图再看 `ArcFace / relation distill / feature distill / SupCon` 各分量谁在主导训练。",
                    "",
                ]
            )
        if "validation" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["validation"], start=output_path.parent))
            lines.extend(
                [
                    f"![Validation metric curves]({rel.as_posix()})",
                    "",
                    "- 读图方式：先看 `macro ARI` 和 `macro Recall@1` 的走势，再看每个 dataset 的 `ARI` 是否同步提升，避免被单一数据集掩盖。",
                    "",
                ]
            )
        if "alignment" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["alignment"], start=output_path.parent))
            lines.extend(
                [
                    f"![Alignment curves]({rel.as_posix()})",
                    "",
                    "- 读图方式：`relation_mse / relation_mae` 越低越好，`relation_corr` 越高越好。它反映 student 是否还在贴近 teacher 的局部相似度结构。",
                    "",
                ]
            )
    lines.extend(
        [
            "## Epoch Log",
            "",
            dataframe_to_markdown_table(training_log_df),
            "",
        ]
    )
    found_qualitative = False
    qualitative_lines = ["## Qualitative Review", ""]
    for dataset in LABELED_DATASETS:
        cluster_path = qualitative_dir / f"predicted_clusters_{dataset}.jpg"
        hard_path = qualitative_dir / f"hard_negatives_{dataset}.jpg"
        if cluster_path.exists():
            rel = Path(os.path.relpath(cluster_path, start=output_path.parent))
            qualitative_lines.extend(
                [
                    f"### {dataset} predicted clusters",
                    "",
                    f"![{dataset} predicted clusters]({rel.as_posix()})",
                    "",
                    "- 读图方式：随机看几行 cluster 内样本是否真的像同一只个体，优先观察是否存在明显背景主导或视角断裂。",
                    "",
                ]
            )
            found_qualitative = True
        if hard_path.exists():
            rel = Path(os.path.relpath(hard_path, start=output_path.parent))
            qualitative_lines.extend(
                [
                    f"### {dataset} hard negatives",
                    "",
                    f"![{dataset} hard negatives]({rel.as_posix()})",
                    "",
                    "- 读图方式：左边是 query，右边是高相似但不同 identity 的样本，优先看模型是否被局部纹理、遮挡或姿态误导。",
                    "",
                ]
            )
            found_qualitative = True
    if found_qualitative:
        lines.extend(qualitative_lines)
    lines.extend(
        [
            "## Conclusion And Next Decision",
            "",
            f"- `current_best_judgment`: `{config['next_step_judgment']}`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_training_monitor_plots(
    plots_dir: Path,
    training_log_df: pd.DataFrame,
    alignment_history_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    _require_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}
    if training_log_df.empty:
        return plot_paths

    epoch_values = training_log_df["epoch"].astype(float).to_numpy()

    loss_path = plots_dir / "training_loss_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epoch_values, training_log_df["train_loss"], marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    component_columns = [
        ("train_arcface_loss", "ArcFace", "#d62728"),
        ("train_relation_distill_loss", "Relation Distill", "#2ca02c"),
        ("train_feature_distill_loss", "Feature Distill", "#ff7f0e"),
        ("train_supcon_loss", "SupCon", "#9467bd"),
    ]
    for column, label, color in component_columns:
        if column in training_log_df.columns:
            axes[1].plot(epoch_values, training_log_df[column], marker="o", linewidth=2, label=label, color=color)
    axes[1].set_title("Loss Breakdown")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(loss_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths["loss"] = loss_path

    validation_path = plots_dir / "validation_metric_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epoch_values, training_log_df["macro_ari"], marker="o", linewidth=2, label="macro ARI", color="#1f77b4")
    if "macro_recall_at_1" in training_log_df.columns:
        axes[0].plot(
            epoch_values,
            training_log_df["macro_recall_at_1"],
            marker="o",
            linewidth=2,
            label="macro Recall@1",
            color="#ff7f0e",
        )
    axes[0].set_title("Validation Macro Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Metric")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    dataset_ari_columns = [column for column in training_log_df.columns if column.endswith("_ari") and not column.startswith("macro_")]
    color_cycle = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    for index, column in enumerate(sorted(dataset_ari_columns)):
        axes[1].plot(
            epoch_values,
            training_log_df[column],
            marker="o",
            linewidth=2,
            label=column.replace("_ari", ""),
            color=color_cycle[index % len(color_cycle)],
        )
    axes[1].set_title("Per-Dataset Validation ARI")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ARI")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(validation_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths["validation"] = validation_path

    if alignment_history_df is not None and not alignment_history_df.empty:
        alignment_path = plots_dir / "alignment_curves.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        overall_df = alignment_history_df[alignment_history_df["dataset"] == "ALL_VAL"].copy()
        if not overall_df.empty:
            overall_df = overall_df.sort_values("epoch")
            axes[0].plot(overall_df["epoch"], overall_df["relation_mse"], marker="o", linewidth=2, label="relation MSE", color="#1f77b4")
            axes[0].plot(overall_df["epoch"], overall_df["relation_mae"], marker="o", linewidth=2, label="relation MAE", color="#ff7f0e")
            axes[0].set_title("Teacher / Student Relation Error")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Error")
            axes[0].grid(alpha=0.3)
            axes[0].legend(loc="best")

            axes[1].plot(overall_df["epoch"], overall_df["relation_corr"], marker="o", linewidth=2, color="#2ca02c")
            axes[1].set_title("Teacher / Student Relation Correlation")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Correlation")
            axes[1].grid(alpha=0.3)
        fig.savefig(alignment_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        plot_paths["alignment"] = alignment_path

    return plot_paths


def build_optimizer(
    model: SupervisedEmbeddingModel,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    head_modules = [model.embedding_layer, model.embedding_bn, model.arcface_heads]
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


def train_one_epoch(
    model: SupervisedEmbeddingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    scaler,
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
        "arcface_loss": 0.0,
        "relation_distill_loss": 0.0,
        "feature_distill_loss": 0.0,
        "supcon_loss": 0.0,
        "batches": 0,
    }
    use_amp = device.startswith("cuda")
    use_feature_distill = float(feature_distill_weight) > 0.0
    use_relation_distill = float(relation_distill_weight) > 0.0

    for batch_index, batch in enumerate(loader, start=1):
        if max_train_batches is not None and batch_index > max_train_batches:
            break
        base_images = batch["base_image"].to(device, non_blocking=True)
        aug_images = batch["aug_image"].to(device, non_blocking=True)
        dataset_indices = batch["dataset_index"].to(device, non_blocking=True)
        label_indices = batch["label_index"].to(device, non_blocking=True)
        teacher_embeddings = None
        if use_feature_distill or use_relation_distill:
            teacher_embeddings = F.normalize(batch["teacher_embedding"].to(device, non_blocking=True), dim=1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model.encode(aug_images)
            arcface_loss = compute_arcface_loss(
                model=model,
                embeddings=embeddings,
                dataset_indices=dataset_indices,
                label_indices=label_indices,
                label_smoothing=label_smoothing,
            )
            if use_feature_distill:
                projected_teacher = model.project_teacher_space(embeddings)
                feature_distill_loss = compute_feature_distillation_loss(projected_teacher, teacher_embeddings)
            else:
                feature_distill_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            if use_relation_distill:
                relation_distill_loss = compute_relation_distillation_loss(
                    student_embeddings=embeddings,
                    teacher_embeddings=teacher_embeddings,
                    dataset_indices=dataset_indices,
                )
            else:
                relation_distill_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            if supcon_weight > 0:
                base_embeddings = model.encode(base_images)
                supcon_loss = compute_view_pair_contrastive_loss(
                    base_embeddings=base_embeddings,
                    augmented_embeddings=embeddings,
                    dataset_indices=dataset_indices,
                    temperature=supcon_temperature,
                )
            else:
                supcon_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            loss = arcface_loss
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
        totals["arcface_loss"] += float(arcface_loss.detach().cpu())
        totals["relation_distill_loss"] += float(relation_distill_loss.detach().cpu())
        totals["feature_distill_loss"] += float(feature_distill_loss.detach().cpu())
        totals["supcon_loss"] += float(supcon_loss.detach().cpu())
        totals["batches"] += 1

    batches = max(1, totals["batches"])
    return {
        "train_loss": round(totals["loss"] / batches, 6),
        "train_arcface_loss": round(totals["arcface_loss"] / batches, 6),
        "train_relation_distill_loss": round(totals["relation_distill_loss"] / batches, 6),
        "train_feature_distill_loss": round(totals["feature_distill_loss"] / batches, 6),
        "train_supcon_loss": round(totals["supcon_loss"] / batches, 6),
        "peak_cuda_memory_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2)
        if device.startswith("cuda")
        else 0.0,
    }


def save_best_eval_artifacts(
    output_dir: Path,
    repo_root: Path,
    split_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    best_thresholds_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    split_seed: int,
) -> pd.DataFrame:
    tables_dir = output_dir / "tables"
    embeddings_dir = output_dir / "embeddings"
    qualitative_dir = output_dir / "qualitative"
    tables_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    threshold_by_dataset = {
        row["dataset"]: float(row["threshold"])
        for _, row in best_thresholds_df.iterrows()
    }
    val_pred_df = apply_thresholds_to_df(df=val_df, embeddings=val_embeddings, threshold_by_dataset=threshold_by_dataset)
    neighbor_df = build_neighbor_table(df=val_df[["image_id", "dataset", "identity", PATH_COLUMN]], embeddings=val_embeddings, top_k=5)

    split_df.to_csv(tables_dir / "split_assignments_v1.csv", index=False)
    fit_df.to_csv(tables_dir / "fit_manifest_v1.csv", index=False)
    val_df.to_csv(tables_dir / "val_manifest_v1.csv", index=False)
    sweep_df.to_csv(tables_dir / "val_threshold_sweep_v1.csv", index=False)
    best_thresholds_df.to_csv(tables_dir / "best_thresholds_v1.csv", index=False)
    recall_df.to_csv(tables_dir / "val_recall_v1.csv", index=False)
    val_pred_df.to_csv(tables_dir / "val_predictions_v1.csv", index=False)
    neighbor_df.to_csv(tables_dir / "val_neighbors_v1.csv", index=False)
    np.save(embeddings_dir / "val_embeddings.npy", val_embeddings.astype(np.float32))
    val_df.to_csv(embeddings_dir / "val_metadata.csv", index=False)

    create_baseline_qualitative_outputs(
        val_pred_df=val_pred_df,
        neighbor_df=neighbor_df,
        repo_root=repo_root,
        qualitative_dir=qualitative_dir,
        seed=split_seed,
    )
    return val_pred_df


def run_supervised_training(
    repo_root: Path,
    output_dir: Path,
    experiment_id: str,
    student_backbone: str,
    teacher_sources: list[str],
    datasets: list[str] | None = None,
    device: str = "cuda:0",
    epochs: int = 8,
    embedding_dim: int = 512,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    val_identity_fraction: float = 0.1,
    split_seed: int = 42,
    thresholds: list[float] | None = None,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    lr_reference_batch_size: int = 4,
    lr_scale_mode: str = "linear",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.3,
    relation_distill_weight: float = 0.2,
    feature_distill_weight: float = 0.05,
    supcon_weight: float = 0.0,
    supcon_temperature: float = 0.1,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 1.0,
    salamander_subcenter_k: int = 1,
    teacher_cache_dir: Path | None = None,
    train_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
    init_checkpoint_path: Path | None = None,
    init_checkpoint_scope: str = "encoder",
    max_train_rows: int | None = None,
    max_val_rows: int | None = None,
    max_train_batches: int | None = None,
    dataset_preprocess_overrides: dict[str, str] | None = None,
    goal: str | None = None,
    resource_decision: str | None = None,
    probe_reuse_note: str | None = None,
) -> dict[str, Path]:
    _require_torch()
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    seed_everything(split_seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    reports_dir = output_dir / "reports"
    tables_dir = output_dir / "tables"
    for path in [checkpoints_dir, reports_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)
    resource_snapshot = collect_resource_snapshot(device)
    resolved_preprocess_config = resolve_dataset_preprocess_config(dataset_preprocess_overrides)

    train_df, _test_df = load_manifests(
        repo_root=repo_root,
        train_manifest_path=train_manifest_path,
        test_manifest_path=test_manifest_path,
    )
    available_datasets = sorted(train_df["dataset"].dropna().astype(str).unique().tolist())
    train_datasets = resolve_training_datasets(requested_datasets=datasets, available_datasets=available_datasets)
    train_df = train_df[train_df["dataset"].isin(train_datasets)].copy().reset_index(drop=True)
    split_df = build_identity_holdout_split(
        train_df=train_df,
        val_identity_fraction=val_identity_fraction,
        seed=split_seed,
        datasets=train_datasets,
    )
    fit_df = split_df[split_df["split_role_v1"] == "fit"].copy().reset_index(drop=True)
    val_df = split_df[split_df["split_role_v1"] == "val"].copy().reset_index(drop=True)
    fit_df = maybe_limit_rows(fit_df, max_train_rows)
    val_df = maybe_limit_rows(val_df, max_val_rows)
    fit_df, label_maps, class_summary_df = attach_training_labels(fit_df)
    dataset_class_counts = {dataset: len(mapping) for dataset, mapping in label_maps.items()}
    print(
        f"[supervised_training] split ready | fit_images={len(fit_df)} | val_images={len(val_df)} | class_counts={dataset_class_counts}",
        flush=True,
    )

    if teacher_cache_dir is None:
        teacher_cache_dir = output_dir / "teacher_cache"
    use_any_distill = float(relation_distill_weight) > 0.0 or float(feature_distill_weight) > 0.0
    if use_any_distill:
        if not teacher_sources:
            raise ValueError("teacher_sources must be non-empty when distillation weights are enabled")
        teacher_bundle = compute_or_load_teacher_cache(
            repo_root=repo_root,
            fit_df=fit_df,
            val_df=val_df,
            teacher_sources=teacher_sources,
            cache_dir=teacher_cache_dir,
            device=device,
            num_workers=num_workers,
        )
    else:
        teacher_cache_dir.mkdir(parents=True, exist_ok=True)
        empty_component_table = pd.DataFrame(columns=["teacher_source", "model_id", "fit_dim", "val_dim"])
        empty_component_table.to_csv(teacher_cache_dir / "teacher_components_v1.csv", index=False)
        teacher_bundle = TeacherCacheBundle(
            fit_embeddings=np.zeros((len(fit_df), 0), dtype=np.float32),
            val_embeddings=np.zeros((len(val_df), 0), dtype=np.float32),
            component_table=empty_component_table,
            cache_dir=teacher_cache_dir,
        )

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

    model = SupervisedEmbeddingModel(
        backbone=backbone,
        feature_dim=backbone_spec.feature_dim,
        embedding_dim=embedding_dim,
        dataset_class_counts=dataset_class_counts,
        teacher_dim=int(teacher_bundle.fit_embeddings.shape[1]),
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
        salamander_subcenter_k=salamander_subcenter_k,
    ).to(device)
    init_checkpoint_summary: dict[str, object] | None = None
    if init_checkpoint_path is not None:
        init_checkpoint_summary = load_matching_checkpoint_weights(
            model=model,
            checkpoint_path=init_checkpoint_path,
            scope=init_checkpoint_scope,
        )
        print(
            "[supervised_training] init checkpoint loaded | "
            f"path={init_checkpoint_path} | scope={init_checkpoint_scope} | "
            f"loaded_keys={init_checkpoint_summary['loaded_key_count']} | "
            f"skipped_keys={init_checkpoint_summary['skipped_key_count']}",
            flush=True,
        )
    print(
        f"[supervised_training] student ready | backbone={student_backbone} | input_size={backbone_spec.input_size} | "
        f"feature_dim={backbone_spec.feature_dim} | embedding_dim={embedding_dim}",
        flush=True,
    )

    optimizer = build_optimizer(
        model=model,
        backbone_lr=resolved_backbone_lr,
        head_lr=resolved_head_lr,
        weight_decay=weight_decay,
    )
    train_dataset = SupervisedTrainDataset(
        df=fit_df,
        repo_root=repo_root,
        spec=backbone_spec,
        teacher_embeddings=teacher_bundle.fit_embeddings,
        preprocess_config=resolved_preprocess_config,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    print(
        f"[supervised_training] loader ready | train_batch_size={train_batch_size} | eval_batch_size={eval_batch_size} | "
        f"steps_per_epoch={len(train_loader)} | effective_batch_size={effective_batch_size} | "
        f"resolved_backbone_lr={resolved_backbone_lr:.8f} | resolved_head_lr={resolved_head_lr:.8f}",
        flush=True,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=max(1, min(len(train_loader), max_train_batches) if max_train_batches else len(train_loader)),
        warmup_ratio=warmup_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    config = {
        "experiment_id": experiment_id,
        "goal": goal
        or f"Evaluate supervised fine-tuning with `{student_backbone}` initialization, per-dataset ArcFace heads, and teacher distillation.",
        "student_backbone": student_backbone,
        "student_model_id": backbone_spec.model_id,
        "datasets": train_datasets,
        "student_feature_dim": backbone_spec.feature_dim,
        "teacher_sources": teacher_sources,
        "teacher_dim": int(teacher_bundle.fit_embeddings.shape[1]),
        "input_size": backbone_spec.input_size,
        "embedding_dim": embedding_dim,
        "device": device,
        "epochs": epochs,
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "val_identity_fraction": val_identity_fraction,
        "split_seed": split_seed,
        "thresholds": thresholds,
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
        "relation_distill_weight": relation_distill_weight,
        "feature_distill_weight": feature_distill_weight,
        "supcon_weight": supcon_weight,
        "supcon_temperature": supcon_temperature,
        "label_smoothing": label_smoothing,
        "grad_clip_norm": grad_clip_norm,
        "salamander_subcenter_k": salamander_subcenter_k,
        "fit_images": int(len(fit_df)),
        "val_images": int(len(val_df)),
        "fit_classes": dataset_class_counts,
        "teacher_cache_dir": str(teacher_bundle.cache_dir),
        "init_checkpoint_path": str(init_checkpoint_path) if init_checkpoint_path is not None else "",
        "init_checkpoint_scope": init_checkpoint_scope,
        "init_checkpoint_summary": init_checkpoint_summary or {},
        "dataset_preprocess_overrides": dataset_preprocess_overrides or {},
        "resolved_preprocess_config": resolved_preprocess_config,
        "validation_singleton_policy": "Exclude singleton identities from validation holdout; keep them in fit only.",
        "fit_ids": int(fit_df["identity"].nunique()),
        "effective_batch_size": effective_batch_size,
        "val_ids": int(val_df["identity"].nunique()),
        "world_size": 1,
        "gradient_accumulation_steps": 1,
        "amp_enabled": bool(device.startswith("cuda")),
        "train_augmentation": {
            dataset: describe_transform(
                build_train_transform(backbone_spec, dataset=dataset, preprocess_config=resolved_preprocess_config)
            )
            for dataset in sorted(fit_df["dataset"].astype(str).unique().tolist())
        },
        "eval_preprocess": {
            dataset: describe_transform(
                build_eval_transform(backbone_spec, dataset=dataset, preprocess_config=resolved_preprocess_config)
            )
            for dataset in sorted(fit_df["dataset"].astype(str).unique().tolist())
        },
        "head_shapes": build_head_shape_summary(model),
        "resource_snapshot": resource_snapshot,
        "resource_decision": resource_decision
        or "single-GPU run; prefer independent single-GPU experiments on free devices before multi-GPU.",
        "probe_reuse_note": probe_reuse_note or "No explicit probe note recorded.",
        "next_step_judgment": "Inspect best epoch summary, compare against frozen fusion and rerank baselines, then decide whether to continue the current backbone or switch to the next matrix item.",
        "checkpoint_policy": {
            "canonical": {
                "best_ari": "best_ari.pt",
                "best_recall1": "best_recall1.pt",
                "last": "last.pt",
            },
            "legacy_aliases": {
                "best.pt": "best_ari.pt",
                "best_macro.pt": "best_ari.pt",
            },
            "dataset_specific_best": [f"best_{dataset}.pt" for dataset in train_datasets],
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    class_summary_df.to_csv(tables_dir / "class_summary_v1.csv", index=False)
    teacher_bundle.component_table.to_csv(tables_dir / "teacher_components_v1.csv", index=False)

    best_macro_ari = float("-inf")
    best_epoch = -1
    best_macro_recall_at_1 = float("-inf")
    best_recall_epoch = -1
    best_metric_scores: dict[str, float] = {
        "macro": float("-inf"),
        "best_ari": float("-inf"),
        "best_recall1": float("-inf"),
    }
    best_metric_scores.update({dataset: float("-inf") for dataset in train_datasets})
    best_metric_epochs: dict[str, int] = {
        "macro": -1,
        "best_ari": -1,
        "best_recall1": -1,
    }
    best_metric_epochs.update({dataset: -1 for dataset in train_datasets})
    training_rows: list[dict[str, object]] = []
    alignment_history_frames: list[pd.DataFrame] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            relation_distill_weight=relation_distill_weight,
            feature_distill_weight=feature_distill_weight,
            supcon_weight=supcon_weight,
            supcon_temperature=supcon_temperature,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
            max_train_batches=max_train_batches,
        )
        val_embeddings = extract_student_embeddings(
            df=val_df,
            repo_root=repo_root,
            model=model,
            spec=backbone_spec,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            preprocess_config=resolved_preprocess_config,
        )
        sweep_df, best_thresholds_df, recall_df = evaluate_validation_embeddings(
            val_df=val_df,
            val_embeddings=val_embeddings,
            thresholds=thresholds,
        )
        macro_ari = float(best_thresholds_df["ari"].mean())
        macro_recall_at_1 = float(recall_df["recall_at_1"].mean())
        val_alignment_df = summarize_alignment(
            student_embeddings=val_embeddings,
            teacher_embeddings=teacher_bundle.val_embeddings,
            metadata_df=val_df,
        )
        val_alignment_df["epoch"] = epoch
        alignment_history_frames.append(val_alignment_df)
        alignment_history_df = pd.concat(alignment_history_frames, ignore_index=True)
        overall_alignment_row = val_alignment_df[val_alignment_df["dataset"] == "ALL_VAL"].iloc[0]
        epoch_row: dict[str, object] = {
            "epoch": epoch,
            **train_metrics,
            "macro_ari": round(macro_ari, 6),
            "macro_recall_at_1": round(macro_recall_at_1, 6),
            "alignment_relation_mse": float(overall_alignment_row["relation_mse"]),
            "alignment_relation_mae": float(overall_alignment_row["relation_mae"]),
            "alignment_relation_corr": float(overall_alignment_row["relation_corr"]),
        }
        for _, row in best_thresholds_df.iterrows():
            dataset = str(row["dataset"])
            epoch_row[f"{dataset}_ari"] = float(row["ari"])
            epoch_row[f"{dataset}_threshold"] = float(row["threshold"])
        training_rows.append(epoch_row)
        training_log_df = pd.DataFrame(training_rows)
        training_log_df.to_csv(tables_dir / "training_log_v1.csv", index=False)
        alignment_history_df.to_csv(tables_dir / "alignment_history_v1.csv", index=False)
        best_metric_df = build_best_metric_summary(training_log_df)
        best_metric_df.to_csv(tables_dir / "best_checkpoints_v1.csv", index=False)
        checkpoint_comparison_df = build_checkpoint_comparison_summary(training_log_df)
        checkpoint_comparison_df.to_csv(tables_dir / "checkpoint_comparison_v1.csv", index=False)
        plot_paths = write_training_monitor_plots(
            plots_dir=reports_dir / "plots",
            training_log_df=training_log_df,
            alignment_history_df=alignment_history_df,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "best_macro_ari": max(best_macro_ari, macro_ari),
            "config": config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch_metrics": epoch_row,
        }

        current_metric_values: dict[str, float] = {
            "macro": macro_ari,
            "best_ari": macro_ari,
            "best_recall1": macro_recall_at_1,
        }
        for dataset in train_datasets:
            metric_key = f"{dataset}_ari"
            if metric_key in epoch_row:
                current_metric_values[dataset] = float(epoch_row[metric_key])
        for selection_key, metric_value in current_metric_values.items():
            if metric_value > best_metric_scores[selection_key]:
                best_metric_scores[selection_key] = metric_value
                best_metric_epochs[selection_key] = epoch
                if selection_key == "macro":
                    target_name = "best_macro.pt"
                elif selection_key == "best_ari":
                    target_name = "best_ari.pt"
                elif selection_key == "best_recall1":
                    target_name = "best_recall1.pt"
                else:
                    target_name = f"best_{selection_key}.pt"
                torch.save(
                    {
                        **checkpoint_payload,
                        "best_metric_name": selection_key,
                        "best_metric_value": metric_value,
                        "best_metric_epoch": epoch,
                    },
                    checkpoints_dir / target_name,
                )
                if selection_key == "best_ari":
                    replace_with_hardlink(checkpoints_dir / target_name, checkpoints_dir / "best_macro.pt")
                    replace_with_hardlink(checkpoints_dir / target_name, checkpoints_dir / "best.pt")

        if macro_ari > best_macro_ari:
            best_macro_ari = macro_ari
            best_epoch = epoch
            save_best_eval_artifacts(
                output_dir=output_dir,
                repo_root=repo_root,
                split_df=split_df,
                fit_df=fit_df,
                val_df=val_df,
                val_embeddings=val_embeddings,
                best_thresholds_df=best_thresholds_df,
                sweep_df=sweep_df,
                recall_df=recall_df,
                split_seed=split_seed,
            )
            val_alignment_df.to_csv(tables_dir / "val_teacher_alignment_v1.csv", index=False)
            write_supervised_training_report(
                output_path=reports_dir / "summary.md",
                config=config,
                training_log_df=training_log_df,
                best_metric_df=best_metric_df,
                checkpoint_comparison_df=checkpoint_comparison_df,
                component_df=teacher_bundle.component_table,
                best_thresholds_df=best_thresholds_df,
                recall_df=recall_df,
                alignment_df=val_alignment_df,
                plot_paths=plot_paths,
                status="running",
            )
        if macro_recall_at_1 > best_macro_recall_at_1:
            best_macro_recall_at_1 = macro_recall_at_1
            best_recall_epoch = epoch

        torch.save(
            {
                "epoch": epoch,
                "best_macro_ari": best_macro_ari,
                "best_macro_recall_at_1": best_macro_recall_at_1,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "epoch_metrics": epoch_row,
            },
            checkpoints_dir / "last.pt",
        )
        print(
            f"[supervised_training] epoch={epoch}/{epochs} "
            f"train_loss={train_metrics['train_loss']:.4f} macro_ari={macro_ari:.4f} "
            f"peak_cuda_memory_mb={train_metrics['peak_cuda_memory_mb']:.2f} best={best_macro_ari:.4f}",
            flush=True,
        )

    summary_payload = {
        "best_epoch": best_epoch,
        "best_macro_ari": round(best_macro_ari, 6),
        "best_recall_epoch": best_recall_epoch,
        "best_macro_recall_at_1": round(best_macro_recall_at_1, 6),
        "best_metric_epochs": best_metric_epochs,
        "best_metric_scores": {key: round(float(value), 6) for key, value in best_metric_scores.items()},
        "experiment_id": experiment_id,
        "student_backbone": student_backbone,
        "datasets": train_datasets,
        "teacher_sources": teacher_sources,
        "teacher_cache_dir": str(teacher_bundle.cache_dir),
        "validation_singleton_policy": config["validation_singleton_policy"],
        "checkpoint_comparison_path": str(tables_dir / "checkpoint_comparison_v1.csv"),
    }
    checkpoint_comparison_path = tables_dir / "checkpoint_comparison_v1.csv"
    if checkpoint_comparison_path.exists():
        checkpoint_summary_df = pd.read_csv(checkpoint_comparison_path)
        summary_payload["checkpoint_summary"] = {
            str(row["checkpoint_key"]): {
                "checkpoint_name": str(row["checkpoint_name"]),
                "epoch": int(row["epoch"]),
                "macro_ari": round(float(row["macro_ari"]), 6),
                "macro_recall_at_1": round(float(row["macro_recall_at_1"]), 6),
                "selection_metric": str(row["selection_metric"]),
                "selection_value": round(float(row["selection_value"]), 6),
            }
            for _, row in checkpoint_summary_df.iterrows()
        }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    best_thresholds_path = tables_dir / "best_thresholds_v1.csv"
    recall_path = tables_dir / "val_recall_v1.csv"
    alignment_path = tables_dir / "val_teacher_alignment_v1.csv"
    if best_thresholds_path.exists() and recall_path.exists() and alignment_path.exists() and checkpoint_comparison_path.exists():
        write_supervised_training_report(
            output_path=reports_dir / "summary.md",
            config=config,
            training_log_df=training_log_df,
            best_metric_df=best_metric_df,
            checkpoint_comparison_df=pd.read_csv(checkpoint_comparison_path),
            component_df=teacher_bundle.component_table,
            best_thresholds_df=pd.read_csv(best_thresholds_path),
            recall_df=pd.read_csv(recall_path),
            alignment_df=pd.read_csv(alignment_path),
            plot_paths=plot_paths,
            status="completed",
        )
    return {
        "summary_path": reports_dir / "summary.md",
        "training_log_path": tables_dir / "training_log_v1.csv",
        "best_checkpoint_path": checkpoints_dir / "best.pt",
        "best_checkpoint_paths": {
            "best_ari": checkpoints_dir / "best_ari.pt",
            "best_recall1": checkpoints_dir / "best_recall1.pt",
            "last": checkpoints_dir / "last.pt",
            "macro": checkpoints_dir / "best_macro.pt",
            **{dataset: checkpoints_dir / f"best_{dataset}.pt" for dataset in train_datasets},
        },
        "teacher_cache_dir": teacher_bundle.cache_dir,
    }
