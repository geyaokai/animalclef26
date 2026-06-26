from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

try:  # pragma: no cover - exercised in training env
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ModuleNotFoundError:  # pragma: no cover - keep import-light unit checks possible
    plt = None
    torch = None
    F = None
    nn = object
    DataLoader = Dataset = WeightedRandomSampler = object

from .descriptor_baselines import (
    PATH_COLUMN,
    apply_thresholds_to_df,
    build_identity_holdout_split,
    build_neighbor_table,
    build_submission,
    dataframe_to_markdown_table,
    pick_best_thresholds,
    recall_at_k,
    run_threshold_sweep,
)
from .qualitative_salamander_views import (
    DEFAULT_SCALE_TARGET_EXTENT_RATIO,
    build_salamander_crop_metadata,
    generate_heuristic_end_middle_crops,
    scale_normalize_aligned_foreground,
)
from .sam_orb_veto import infer_mask_from_masked_rgb
from .supervised_training import (
    ArcFaceHead,
    StudentBackboneSpec,
    build_scheduler,
    collect_resource_snapshot,
    load_student_backbone,
    scale_learning_rate,
    seed_everything,
    write_training_monitor_plots,
)


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
DEFAULT_BASE_PREDICTIONS = Path(
    "artifacts/submissions/kaggle_variant_salamander_yellow_combo_split52_gate2_singletonrescue_v1/tables/test_predictions_v1.csv"
)
DEFAULT_TRAIN_MANIFEST = Path("artifacts/manifests/sam_seg_trainprep_repaired_v1/tables/manifest_train_sam_trainprep_aligned_best_v1.csv")
DEFAULT_TEST_MANIFEST = Path("artifacts/manifests/sam_seg_trainprep_repaired_v1/tables/manifest_test_sam_trainprep_aligned_best_v1.csv")
DEFAULT_CACHE_DIR = Path("artifacts/training/cache/salamander_dualview_centertrunk_v1")


@dataclass(frozen=True)
class SalamanderDualViewCache:
    train_manifest_path: Path
    expanded_train_manifest_path: Path
    test_manifest_path: Path
    summary_path: Path


@dataclass(frozen=True)
class SalamanderSingletonPseudoRecipe:
    name: str
    description: str
    trunk_dirname: str
    trunk_extremity_trim_ratio: float
    trunk_minor_extent_ratio: float


DEFAULT_SINGLETON_PSEUDO_RECIPES: tuple[SalamanderSingletonPseudoRecipe, ...] = (
    SalamanderSingletonPseudoRecipe(
        name="singleton_center_tight_v1",
        description="A tighter center trunk crop that trims more head/tail and keeps a narrower body band.",
        trunk_dirname="singleton_center_tight",
        trunk_extremity_trim_ratio=0.24,
        trunk_minor_extent_ratio=0.60,
    ),
    SalamanderSingletonPseudoRecipe(
        name="singleton_center_loose_v1",
        description="A wider center trunk crop that keeps more mid-body context while still suppressing tail extremes.",
        trunk_dirname="singleton_center_loose",
        trunk_extremity_trim_ratio=0.10,
        trunk_minor_extent_ratio=0.84,
    ),
)


def _require_torch() -> None:
    if torch is None:
        raise ModuleNotFoundError("salamander_dualview_training requires torch; run in the `wildfusion` environment.")


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for Salamander dual-view plots.")


def _ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _dataset_rel_path(path: str) -> Path:
    parts = Path(str(path)).parts
    if SALAMANDER_DATASET in parts:
        start = parts.index(SALAMANDER_DATASET)
        return Path(*parts[start:])
    return Path(Path(str(path)).name)


def _resolve_existing_global_path(row: pd.Series, repo_root: Path) -> str:
    candidate_columns = [
        "sam_trainprep_aligned_resolved_path_v1",
        "sam_trainprep_aligned_path_v1",
        "path",
        PATH_COLUMN,
    ]
    for column in candidate_columns:
        if column not in row.index:
            continue
        value = str(row.get(column, "") or "").strip()
        if not value:
            continue
        if (repo_root / value).exists():
            return value
    raise FileNotFoundError(f"Could not resolve Salamander aligned path for image_id={row.get('image_id')}")


def _singleton_recipe_path_column(recipe: SalamanderSingletonPseudoRecipe) -> str:
    return f"pseudo_trunk_path__{recipe.name}"


def _singleton_recipe_metadata_column(recipe: SalamanderSingletonPseudoRecipe) -> str:
    return f"pseudo_trunk_metadata_json__{recipe.name}"


def _build_random_resized_crop(size: int, *, scale: tuple[float, float], ratio: tuple[float, float]) -> A.BasicTransform:
    try:
        return A.RandomResizedCrop(size=(size, size), scale=scale, ratio=ratio, p=1.0)
    except TypeError:
        return A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio, p=1.0)


def _build_resize(size: int) -> A.BasicTransform:
    try:
        return A.Resize(height=size, width=size)
    except TypeError:
        return A.Resize(size, size)


def _build_coarse_dropout(
    *,
    p: float,
    min_holes: int,
    max_holes: int,
    min_frac: float,
    max_frac: float,
) -> A.BasicTransform:
    try:
        return A.CoarseDropout(
            num_holes_range=(min_holes, max_holes),
            hole_height_range=(min_frac, max_frac),
            hole_width_range=(min_frac, max_frac),
            fill=0,
            p=p,
        )
    except TypeError:
        return A.CoarseDropout(
            max_holes=max_holes,
            min_holes=min_holes,
            max_height=int(round(max_frac * 440)),
            max_width=int(round(max_frac * 440)),
            min_height=max(1, int(round(min_frac * 440))),
            min_width=max(1, int(round(min_frac * 440))),
            fill_value=0,
            p=p,
        )


def _pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(_ensure_rgb(image), dtype=np.uint8)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _clean_optional_string(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _aug_to_tensor(image: np.ndarray) -> torch.Tensor:
    _require_torch()
    if image.ndim != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={tuple(image.shape)}")
    return torch.from_numpy(np.transpose(np.ascontiguousarray(image), (2, 0, 1))).float()


class AlbumentationsImageTransform:
    def __init__(self, compose: A.Compose) -> None:
        self.compose = compose

    def __call__(self, image: Image.Image) -> torch.Tensor:
        augmented = self.compose(image=_pil_to_numpy(image))
        return _aug_to_tensor(augmented["image"])


def build_global_train_transform(spec: StudentBackboneSpec) -> AlbumentationsImageTransform:
    return AlbumentationsImageTransform(
        A.Compose(
            [
                _build_random_resized_crop(spec.input_size, scale=(0.90, 1.0), ratio=(0.92, 1.08)),
                A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.35),
                A.GaussianBlur(blur_limit=(3, 5), p=0.10),
                A.GaussNoise(std_range=(0.01, 0.03), p=0.10),
                _build_coarse_dropout(p=0.05, min_holes=1, max_holes=2, min_frac=0.02, max_frac=0.06),
                A.Normalize(mean=spec.mean, std=spec.std),
            ]
        )
    )


def build_trunk_base_transform(spec: StudentBackboneSpec) -> AlbumentationsImageTransform:
    return AlbumentationsImageTransform(
        A.Compose(
            [
                _build_random_resized_crop(spec.input_size, scale=(0.95, 1.0), ratio=(0.90, 1.10)),
                A.Normalize(mean=spec.mean, std=spec.std),
            ]
        )
    )


def build_trunk_aug_transform(spec: StudentBackboneSpec) -> AlbumentationsImageTransform:
    return AlbumentationsImageTransform(
        A.Compose(
            [
                _build_random_resized_crop(spec.input_size, scale=(0.65, 0.95), ratio=(0.85, 1.20)),
                A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.45),
                A.GaussianBlur(blur_limit=(3, 5), p=0.20),
                A.GaussNoise(std_range=(0.02, 0.05), p=0.20),
                _build_coarse_dropout(p=0.20, min_holes=1, max_holes=4, min_frac=0.03, max_frac=0.12),
                A.Normalize(mean=spec.mean, std=spec.std),
            ]
        )
    )


def build_eval_transform(spec: StudentBackboneSpec) -> AlbumentationsImageTransform:
    return AlbumentationsImageTransform(
        A.Compose(
            [
                _build_resize(spec.input_size),
                A.Normalize(mean=spec.mean, std=spec.std),
            ]
        )
    )


def build_or_load_salamander_dualview_cache(
    *,
    repo_root: Path,
    output_dir: Path,
    train_manifest_path: Path,
    test_manifest_path: Path,
    target_extent_ratio: float = DEFAULT_SCALE_TARGET_EXTENT_RATIO,
) -> SalamanderDualViewCache:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    views_dir = output_dir / "views"
    global_dir = views_dir / "global_aligned"
    trunk_dir = views_dir / "center_trunk"
    singleton_recipe_dirs = [views_dir / recipe.trunk_dirname for recipe in DEFAULT_SINGLETON_PSEUDO_RECIPES]
    for path in [output_dir, tables_dir, global_dir, trunk_dir, *singleton_recipe_dirs]:
        path.mkdir(parents=True, exist_ok=True)

    cached_train_path = tables_dir / "manifest_train_salamander_dualview_v1.csv"
    cached_expanded_train_path = tables_dir / "manifest_train_salamander_dualview_expanded_v1.csv"
    cached_test_path = tables_dir / "manifest_test_salamander_dualview_v1.csv"
    summary_path = output_dir / "reports" / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if cached_train_path.exists() and cached_expanded_train_path.exists() and cached_test_path.exists() and summary_path.exists():
        return SalamanderDualViewCache(
            train_manifest_path=cached_train_path,
            expanded_train_manifest_path=cached_expanded_train_path,
            test_manifest_path=cached_test_path,
            summary_path=summary_path,
        )

    def build_one_split(source_manifest_path: Path, *, enable_singleton_expansion: bool) -> pd.DataFrame:
        source_df = pd.read_csv(source_manifest_path, low_memory=False)
        source_df = source_df[source_df["dataset"].astype(str).eq(SALAMANDER_DATASET)].copy().reset_index(drop=True)
        source_df["image_id"] = source_df["image_id"].astype(str)
        source_df["identity"] = source_df["identity"].fillna("").astype(str)
        source_df = source_df.sort_values(["identity", "image_id"]).reset_index(drop=True)
        identity_counts = source_df.groupby("identity")["image_id"].transform("size").astype(int)
        source_df["identity_image_count_base"] = identity_counts
        representative_source_ids = set(
            source_df.groupby("identity", sort=True, as_index=False)["image_id"].first()["image_id"].astype(str).tolist()
        )
        rows: list[dict[str, Any]] = []
        for row in source_df.itertuples(index=False):
            row_series = pd.Series(row._asdict())
            global_source_rel = _resolve_existing_global_path(row_series, repo_root=repo_root)
            global_source_abs = repo_root / global_source_rel
            rel_suffix = _dataset_rel_path(global_source_rel)
            global_export_rel = (Path("views") / "global_aligned" / rel_suffix).as_posix()
            trunk_export_rel = (Path("views") / "center_trunk" / rel_suffix).as_posix()
            global_export_abs = output_dir / global_export_rel
            trunk_export_abs = output_dir / trunk_export_rel
            global_export_abs.parent.mkdir(parents=True, exist_ok=True)
            trunk_export_abs.parent.mkdir(parents=True, exist_ok=True)
            trunk_metadata = build_salamander_crop_metadata(
                crop_name="middle",
                crop_ratio=(0.3, 0.7),
                extra_metadata={"fallback_reason": "uninitialized"},
            )
            singleton_recipe_payloads: dict[str, tuple[str, str]] = {}
            with Image.open(global_source_abs) as source_image:
                source_image = _ensure_rgb(source_image)
                inferred_mask = infer_mask_from_masked_rgb(source_image, nonzero_threshold=1)
                normalized_rgb, normalized_mask, scale_payload = scale_normalize_aligned_foreground(
                    source_image,
                    inferred_mask,
                    output_size=source_image.size,
                    target_extent_ratio=float(target_extent_ratio),
                    background=(0, 0, 0),
                )
                crops = generate_heuristic_end_middle_crops(
                    normalized_rgb,
                    normalized_mask,
                    scale_payload=scale_payload,
                )
                trunk_crop = crops["middle"]
                if not global_export_abs.exists():
                    normalized_rgb.save(global_export_abs, quality=95)
                if not trunk_export_abs.exists():
                    trunk_crop.rgb.save(trunk_export_abs, quality=95)
                trunk_metadata = dict(trunk_crop.metadata)
                trunk_metadata["scale_payload"] = scale_payload

                if enable_singleton_expansion and str(getattr(row, "image_id")) in representative_source_ids:
                    for recipe in DEFAULT_SINGLETON_PSEUDO_RECIPES:
                        recipe_rel = (Path("views") / recipe.trunk_dirname / rel_suffix).as_posix()
                        recipe_abs = output_dir / recipe_rel
                        recipe_abs.parent.mkdir(parents=True, exist_ok=True)
                        recipe_crops = generate_heuristic_end_middle_crops(
                            normalized_rgb,
                            normalized_mask,
                            scale_payload=scale_payload,
                            trunk_extremity_trim_ratio=float(recipe.trunk_extremity_trim_ratio),
                            trunk_minor_extent_ratio=float(recipe.trunk_minor_extent_ratio),
                        )
                        recipe_crop = recipe_crops["middle"]
                        recipe_metadata = dict(recipe_crop.metadata)
                        recipe_metadata["scale_payload"] = scale_payload
                        recipe_metadata["pseudo_recipe_name"] = recipe.name
                        recipe_metadata["pseudo_recipe_description"] = recipe.description
                        recipe_metadata["pseudo_recipe_extremity_trim_ratio"] = round(float(recipe.trunk_extremity_trim_ratio), 6)
                        recipe_metadata["pseudo_recipe_minor_extent_ratio"] = round(float(recipe.trunk_minor_extent_ratio), 6)
                        if not recipe_abs.exists():
                            recipe_crop.rgb.save(recipe_abs, quality=95)
                        singleton_recipe_payloads[recipe.name] = (
                            str((output_dir / recipe_rel).relative_to(repo_root).as_posix()),
                            json.dumps(recipe_metadata, ensure_ascii=False),
                        )
            row_payload = {
                "image_id": str(getattr(row, "image_id")),
                "source_image_id": str(getattr(row, "image_id")),
                "identity": "" if pd.isna(getattr(row, "identity", "")) else str(getattr(row, "identity", "")),
                "dataset": str(getattr(row, "dataset")),
                "split": str(getattr(row, "split")),
                "identity_image_count_base": int(getattr(row, "identity_image_count_base")),
                "global_path": str((output_dir / global_export_rel).relative_to(repo_root).as_posix()),
                "trunk_path": str((output_dir / trunk_export_rel).relative_to(repo_root).as_posix()),
                "source_global_path": str(global_source_rel),
                "global_variant": "sam_aligned_scale_normalized_v1",
                "trunk_variant": "center_trunk_rectangle_v1",
                "view_recipe_name": "base_center_trunk_v1",
                "is_pseudo_positive": False,
                "trunk_metadata_json": json.dumps(trunk_metadata, ensure_ascii=False),
            }
            for recipe in DEFAULT_SINGLETON_PSEUDO_RECIPES:
                recipe_path, recipe_metadata_json = singleton_recipe_payloads.get(recipe.name, ("", ""))
                row_payload[_singleton_recipe_path_column(recipe)] = recipe_path
                row_payload[_singleton_recipe_metadata_column(recipe)] = recipe_metadata_json
            rows.append(row_payload)
        return pd.DataFrame(rows)

    def build_expanded_train_manifest(base_train_df: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row in base_train_df.itertuples(index=False):
            base_row = {
                **dict(row._asdict()),
                "image_id": f"{row.source_image_id}::base_center_trunk_v1",
                "view_recipe_name": "base_center_trunk_v1",
                "is_pseudo_positive": False,
            }
            rows.append(base_row)
            for recipe in DEFAULT_SINGLETON_PSEUDO_RECIPES:
                recipe_path = _clean_optional_string(getattr(row, _singleton_recipe_path_column(recipe), ""))
                recipe_metadata_json = _clean_optional_string(getattr(row, _singleton_recipe_metadata_column(recipe), ""))
                if not recipe_path:
                    continue
                pseudo_row = {
                    **dict(row._asdict()),
                    "image_id": f"{row.source_image_id}::{recipe.name}",
                    "trunk_path": recipe_path,
                    "trunk_variant": recipe.name,
                    "view_recipe_name": recipe.name,
                    "is_pseudo_positive": True,
                    "trunk_metadata_json": recipe_metadata_json or str(getattr(row, "trunk_metadata_json", "")),
                }
                rows.append(pseudo_row)
        expanded_df = pd.DataFrame(rows)
        expanded_df["image_id"] = expanded_df["image_id"].astype(str)
        expanded_df["source_image_id"] = expanded_df["source_image_id"].astype(str)
        expanded_df["view_recipe_name"] = expanded_df["view_recipe_name"].astype(str)
        expanded_df["is_pseudo_positive"] = expanded_df["is_pseudo_positive"].astype(bool)
        return expanded_df.reset_index(drop=True)

    train_df = build_one_split(train_manifest_path.resolve(), enable_singleton_expansion=True)
    test_df = build_one_split(test_manifest_path.resolve(), enable_singleton_expansion=False)
    expanded_train_df = build_expanded_train_manifest(train_df)
    train_df.to_csv(cached_train_path, index=False)
    expanded_train_df.to_csv(cached_expanded_train_path, index=False)
    test_df.to_csv(cached_test_path, index=False)
    singleton_rows = int(train_df["identity_image_count_base"].eq(1).sum()) if not train_df.empty else 0
    representative_rows = int(train_df.sort_values(["identity", "source_image_id"] if "source_image_id" in train_df.columns else ["identity", "image_id"]).groupby("identity").head(1).shape[0]) if not train_df.empty else 0
    expanded_pseudo_rows = int(expanded_train_df["is_pseudo_positive"].sum()) if not expanded_train_df.empty else 0
    summary_lines = [
        "# Salamander Dual-View Cache",
        "",
        f"- Train manifest source: `{train_manifest_path.resolve()}`",
        f"- Test manifest source: `{test_manifest_path.resolve()}`",
        f"- Base train rows: `{len(train_df)}`",
        f"- Expanded train rows: `{len(expanded_train_df)}`",
        f"- Test rows: `{len(test_df)}`",
        f"- Singleton base rows: `{singleton_rows}`",
        f"- Representative images expanded across identities: `{representative_rows}`",
        f"- Materialized representative pseudo-positive rows: `{expanded_pseudo_rows}`",
        f"- Global variant: `sam_aligned_scale_normalized_v1`",
        f"- Trunk variant: `center_trunk_rectangle_v1`",
        f"- Singleton pseudo recipes: `{', '.join(recipe.name for recipe in DEFAULT_SINGLETON_PSEUDO_RECIPES)}`",
        f"- Target extent ratio: `{float(target_extent_ratio)}`",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return SalamanderDualViewCache(
        train_manifest_path=cached_train_path,
        expanded_train_manifest_path=cached_expanded_train_path,
        test_manifest_path=cached_test_path,
        summary_path=summary_path,
    )


class SalamanderDualViewTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        repo_root: Path,
        spec: StudentBackboneSpec,
    ) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.global_transform = build_global_train_transform(spec)
        self.trunk_transform = build_trunk_base_transform(spec)
        self.trunk_aug_transform = build_trunk_aug_transform(spec)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        with Image.open(self.repo_root / str(row["global_path"])) as global_image:
            global_image = _ensure_rgb(global_image)
            global_tensor = self.global_transform(global_image)
        with Image.open(self.repo_root / str(row["trunk_path"])) as trunk_image:
            trunk_image = _ensure_rgb(trunk_image)
            trunk_tensor = self.trunk_transform(trunk_image)
            trunk_aug_tensor = self.trunk_aug_transform(trunk_image)
        return {
            "global_image": global_tensor,
            "trunk_image": trunk_tensor,
            "trunk_aug_image": trunk_aug_tensor,
            "label_index": int(row["label_index"]),
            "identity_image_count_fit": int(row["identity_image_count_fit"]),
        }


class SalamanderDualViewInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, repo_root: Path, spec: StudentBackboneSpec) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.transform = build_eval_transform(spec)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with Image.open(self.repo_root / str(row["global_path"])) as global_image:
            global_image = _ensure_rgb(global_image)
            global_tensor = self.transform(global_image)
        with Image.open(self.repo_root / str(row["trunk_path"])) as trunk_image:
            trunk_image = _ensure_rgb(trunk_image)
            trunk_tensor = self.transform(trunk_image)
        return {
            "global_image": global_tensor,
            "trunk_image": trunk_tensor,
        }


class SalamanderDualViewModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        embedding_dim: int,
        class_count: int,
        arcface_scale: float,
        arcface_margin: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.global_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.global_bn = nn.BatchNorm1d(embedding_dim)
        self.trunk_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.trunk_bn = nn.BatchNorm1d(embedding_dim)
        self.arcface_global = ArcFaceHead(
            in_features=embedding_dim,
            out_features=class_count,
            scale=arcface_scale,
            margin=arcface_margin,
        )
        self.arcface_trunk = ArcFaceHead(
            in_features=embedding_dim,
            out_features=class_count,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def encode_features(self, images: torch.Tensor) -> torch.Tensor:
        from .descriptor_baselines import _coerce_model_output

        features = _coerce_model_output(self.backbone(images))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        return features

    def encode_global(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(images)
        return F.normalize(self.global_bn(self.global_layer(features)), dim=1)

    def encode_trunk(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(images)
        return F.normalize(self.trunk_bn(self.trunk_layer(features)), dim=1)

    def encode_fused(self, global_images: torch.Tensor, trunk_images: torch.Tensor, *, global_weight: float, trunk_weight: float) -> torch.Tensor:
        global_embedding = self.encode_global(global_images)
        trunk_embedding = self.encode_trunk(trunk_images)
        fused = (float(global_weight) * global_embedding) + (float(trunk_weight) * trunk_embedding)
        return F.normalize(fused, dim=1)


def _attach_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    labeled = df.copy().reset_index(drop=True)
    identities = sorted(labeled["identity"].astype(str).unique().tolist())
    label_map = {identity: index for index, identity in enumerate(identities)}
    labeled["label_index"] = labeled["identity"].astype(str).map(label_map).astype(int)
    labeled["identity_image_count_fit"] = labeled.groupby("identity")["image_id"].transform("size").astype(int)
    return labeled, label_map


def _build_train_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    _require_torch()
    class_counts = df.groupby("label_index")["image_id"].transform("size").astype(float)
    weights = 1.0 / np.clip(class_counts.to_numpy(dtype=np.float32), 1.0, None)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(df),
        replacement=True,
    )


def compute_multiview_supcon_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    if embeddings.shape[0] < 2:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    logits = (embeddings @ embeddings.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive_mask = (labels[:, None] == labels[None, :]) & (~eye_mask)
    if not torch.any(positive_mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    denominator_mask = ~eye_mask
    logits_exp = torch.exp(logits) * denominator_mask.to(dtype=embeddings.dtype)
    log_prob = logits - torch.log(torch.clamp(logits_exp.sum(dim=1, keepdim=True), min=1e-12))
    positive_count = positive_mask.sum(dim=1)
    valid_mask = positive_count > 0
    if not torch.any(valid_mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    mean_log_prob_positive = (
        (log_prob * positive_mask.to(dtype=embeddings.dtype)).sum(dim=1) / positive_count.clamp(min=1)
    )
    return (-mean_log_prob_positive[valid_mask]).mean()


def build_optimizer(model: SalamanderDualViewModel, *, backbone_lr: float, head_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    head_params: list[torch.nn.Parameter] = []
    for module in [model.global_layer, model.global_bn, model.trunk_layer, model.trunk_bn, model.arcface_global, model.arcface_trunk]:
        head_params.extend(list(module.parameters()))
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(backbone_lr)},
            {"params": head_params, "lr": float(head_lr)},
        ],
        weight_decay=float(weight_decay),
    )


def train_one_epoch(
    *,
    model: SalamanderDualViewModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: str,
    scaler: Any,
    pair_weight: float,
    supcon_weight: float,
    temperature: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    use_amp = device.startswith("cuda")
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    totals = {
        "loss": 0.0,
        "global_cls_loss": 0.0,
        "trunk_cls_loss": 0.0,
        "pair_loss": 0.0,
        "supcon_loss": 0.0,
        "batches": 0,
    }
    for batch in loader:
        global_images = batch["global_image"].to(device, non_blocking=True)
        trunk_images = batch["trunk_image"].to(device, non_blocking=True)
        trunk_aug_images = batch["trunk_aug_image"].to(device, non_blocking=True)
        labels = batch["label_index"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            global_embeddings = model.encode_global(global_images)
            trunk_embeddings = model.encode_trunk(trunk_images)
            trunk_aug_embeddings = model.encode_trunk(trunk_aug_images)
            global_cls_loss = F.cross_entropy(model.arcface_global(global_embeddings, labels), labels)
            trunk_cls_loss = F.cross_entropy(model.arcface_trunk(trunk_embeddings, labels), labels)
            pair_loss = 0.5 * (
                F.cross_entropy((global_embeddings @ trunk_embeddings.T) / max(float(temperature), 1e-6), torch.arange(len(labels), device=device))
                + F.cross_entropy((trunk_embeddings @ trunk_aug_embeddings.T) / max(float(temperature), 1e-6), torch.arange(len(labels), device=device))
            )
            all_embeddings = torch.cat([global_embeddings, trunk_embeddings, trunk_aug_embeddings], dim=0)
            all_labels = torch.cat([labels, labels, labels], dim=0)
            supcon_loss = compute_multiview_supcon_loss(all_embeddings, all_labels, temperature=float(temperature))
            loss = global_cls_loss + trunk_cls_loss + (float(pair_weight) * pair_loss) + (float(supcon_weight) * supcon_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if (not use_amp) or (scaler.get_scale() >= scale_before_step):
            scheduler.step()
        totals["loss"] += float(loss.detach().cpu())
        totals["global_cls_loss"] += float(global_cls_loss.detach().cpu())
        totals["trunk_cls_loss"] += float(trunk_cls_loss.detach().cpu())
        totals["pair_loss"] += float(pair_loss.detach().cpu())
        totals["supcon_loss"] += float(supcon_loss.detach().cpu())
        totals["batches"] += 1
    batches = max(1, totals["batches"])
    return {
        "train_loss": round(totals["loss"] / batches, 6),
        "train_global_cls_loss": round(totals["global_cls_loss"] / batches, 6),
        "train_trunk_cls_loss": round(totals["trunk_cls_loss"] / batches, 6),
        "train_pair_loss": round(totals["pair_loss"] / batches, 6),
        "train_supcon_loss": round(totals["supcon_loss"] / batches, 6),
        "peak_cuda_memory_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2) if use_amp else 0.0,
    }


def extract_dualview_embeddings(
    *,
    df: pd.DataFrame,
    repo_root: Path,
    model: SalamanderDualViewModel,
    spec: StudentBackboneSpec,
    device: str,
    batch_size: int,
    num_workers: int,
    global_weight: float,
    trunk_weight: float,
) -> np.ndarray:
    _require_torch()
    dataset = SalamanderDualViewInferenceDataset(df=df, repo_root=repo_root, spec=spec)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.startswith("cuda"),
    )
    rows: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            global_images = batch["global_image"].to(device, non_blocking=True)
            trunk_images = batch["trunk_image"].to(device, non_blocking=True)
            embeddings = model.encode_fused(
                global_images,
                trunk_images,
                global_weight=float(global_weight),
                trunk_weight=float(trunk_weight),
            )
            rows.append(embeddings.detach().cpu().numpy().astype(np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    from .descriptor_baselines import l2_normalize

    return l2_normalize(np.concatenate(rows, axis=0))


def _write_summary_markdown(
    *,
    output_dir: Path,
    config: dict[str, Any],
    training_log_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    train_cache: SalamanderDualViewCache,
) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "summary.md"
    plot_dir = reports_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Salamander Dual-View Training",
        "",
        "## Experiment Card",
        "",
        f"- `experiment_id`: `{config['experiment_id']}`",
        f"- `student_backbone`: `{config['student_backbone']}`",
        f"- `student_model_id`: `{config['student_model_id']}`",
        f"- `input_size`: `{config['input_size']}`",
        f"- `embedding_dim`: `{config['embedding_dim']}`",
        f"- `epochs`: `{config['epochs']}`",
        f"- `base_train_rows`: `{config['base_train_rows']}`",
        f"- `train_rows`: `{config['train_rows']}`",
        f"- `singleton_base_rows`: `{config['singleton_base_rows']}`",
        f"- `pseudo_positive_rows`: `{config['pseudo_positive_rows']}`",
        f"- `analysis_eval_rows`: `{config['analysis_eval_rows']}`",
        f"- `analysis_eval_identities`: `{config['analysis_eval_identities']}`",
        f"- `global_weight / trunk_weight`: `{config['global_weight']} / {config['trunk_weight']}`",
        f"- `pair_weight / supcon_weight`: `{config['pair_weight']} / {config['supcon_weight']}`",
        f"- `HorizontalFlip`: `0.0`",
        f"- `dualview_cache_train_manifest`: `{train_cache.train_manifest_path}`",
        f"- `dualview_cache_expanded_train_manifest`: `{train_cache.expanded_train_manifest_path}`",
        f"- `dualview_cache_test_manifest`: `{train_cache.test_manifest_path}`",
        "",
        "## Best Thresholds",
        "",
        dataframe_to_markdown_table(best_thresholds_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Monitoring",
        "",
        "![Training plots](plots/training_loss_curves.png)",
        "",
        "![Validation plots](plots/validation_metric_curves.png)",
        "",
        "## Epoch Log",
        "",
        dataframe_to_markdown_table(training_log_df),
        "",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_salamander_dualview_training(
    *,
    repo_root: Path,
    output_dir: Path,
    experiment_id: str,
    base_predictions_path: Path,
    train_manifest_path: Path = DEFAULT_TRAIN_MANIFEST,
    test_manifest_path: Path = DEFAULT_TEST_MANIFEST,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    student_backbone: str = "miew",
    device: str = "cuda:0",
    epochs: int = 8,
    embedding_dim: int = 512,
    train_batch_size: int = 12,
    eval_batch_size: int = 16,
    num_workers: int = 4,
    analysis_val_identity_fraction: float = 0.2,
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
    pair_weight: float = 0.5,
    supcon_weight: float = 0.75,
    temperature: float = 0.1,
    grad_clip_norm: float = 1.0,
    global_weight: float = 0.55,
    trunk_weight: float = 0.45,
) -> dict[str, Path]:
    _require_torch()
    _require_matplotlib()
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    base_predictions_path = base_predictions_path.resolve()
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)
    seed_everything(int(split_seed))
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    checkpoints_dir = output_dir / "checkpoints"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    embeddings_dir = output_dir / "embeddings"
    for path in [output_dir, checkpoints_dir, tables_dir, reports_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    cache_bundle = build_or_load_salamander_dualview_cache(
        repo_root=repo_root,
        output_dir=(repo_root / cache_dir).resolve(),
        train_manifest_path=(repo_root / train_manifest_path).resolve(),
        test_manifest_path=(repo_root / test_manifest_path).resolve(),
    )
    train_df = pd.read_csv(cache_bundle.train_manifest_path)
    expanded_train_df = pd.read_csv(cache_bundle.expanded_train_manifest_path)
    test_df = pd.read_csv(cache_bundle.test_manifest_path)
    for frame in [train_df, expanded_train_df, test_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        if "source_image_id" in frame.columns:
            frame["source_image_id"] = frame["source_image_id"].astype(str)
        frame["identity"] = frame["identity"].fillna("").astype(str)
        frame["dataset"] = frame["dataset"].astype(str)
        frame["split"] = frame["split"].astype(str)
    analysis_split_df = build_identity_holdout_split(
        train_df=train_df[["image_id", "identity", "dataset"]].merge(train_df, on=["image_id", "identity", "dataset"], how="left"),
        val_identity_fraction=float(analysis_val_identity_fraction),
        seed=int(split_seed),
        datasets=[SALAMANDER_DATASET],
    )
    analysis_eval_df = analysis_split_df[analysis_split_df["split_role_v1"] == "val"].copy().reset_index(drop=True)
    full_train_df, label_map = _attach_labels(expanded_train_df)

    backbone, spec = load_student_backbone(student_backbone, device=device)
    effective_batch_size = int(train_batch_size)
    resolved_backbone_lr = scale_learning_rate(
        base_lr=float(backbone_lr),
        effective_batch_size=effective_batch_size,
        reference_batch_size=int(lr_reference_batch_size),
        mode=str(lr_scale_mode),
    )
    resolved_head_lr = scale_learning_rate(
        base_lr=float(head_lr),
        effective_batch_size=effective_batch_size,
        reference_batch_size=int(lr_reference_batch_size),
        mode=str(lr_scale_mode),
    )
    model = SalamanderDualViewModel(
        backbone=backbone,
        feature_dim=int(spec.feature_dim),
        embedding_dim=int(embedding_dim),
        class_count=len(label_map),
        arcface_scale=float(arcface_scale),
        arcface_margin=float(arcface_margin),
    ).to(device)
    train_dataset = SalamanderDualViewTrainDataset(df=full_train_df, repo_root=repo_root, spec=spec)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_batch_size),
        sampler=_build_train_sampler(full_train_df),
        num_workers=int(num_workers),
        pin_memory=device.startswith("cuda"),
        drop_last=False,
    )
    optimizer = build_optimizer(
        model=model,
        backbone_lr=resolved_backbone_lr,
        head_lr=resolved_head_lr,
        weight_decay=float(weight_decay),
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        epochs=int(epochs),
        steps_per_epoch=max(1, len(train_loader)),
        warmup_ratio=float(warmup_ratio),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    config = {
        "experiment_id": experiment_id,
        "student_backbone": student_backbone,
        "student_model_id": spec.model_id,
        "input_size": int(spec.input_size),
        "embedding_dim": int(embedding_dim),
        "epochs": int(epochs),
        "base_train_rows": int(len(train_df)),
        "train_rows": int(len(full_train_df)),
        "analysis_eval_rows": int(len(analysis_eval_df)),
        "analysis_eval_identities": int(analysis_eval_df["identity"].nunique()),
        "singleton_base_rows": int(train_df["identity_image_count_base"].eq(1).sum()) if "identity_image_count_base" in train_df.columns else 0,
        "pseudo_positive_rows": int(full_train_df["is_pseudo_positive"].map(_as_bool).sum()) if "is_pseudo_positive" in full_train_df.columns else 0,
        "train_batch_size": int(train_batch_size),
        "eval_batch_size": int(eval_batch_size),
        "effective_batch_size": int(effective_batch_size),
        "resolved_backbone_lr": float(resolved_backbone_lr),
        "resolved_head_lr": float(resolved_head_lr),
        "analysis_val_identity_fraction": float(analysis_val_identity_fraction),
        "split_seed": int(split_seed),
        "pair_weight": float(pair_weight),
        "supcon_weight": float(supcon_weight),
        "temperature": float(temperature),
        "global_weight": float(global_weight),
        "trunk_weight": float(trunk_weight),
        "base_predictions_path": str(base_predictions_path),
        "resource_snapshot": collect_resource_snapshot(device),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    full_train_df.to_csv(tables_dir / "train_manifest_v1.csv", index=False)
    analysis_eval_df.to_csv(tables_dir / "analysis_eval_manifest_v1.csv", index=False)

    training_rows: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    best_thresholds_df = pd.DataFrame()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            pair_weight=float(pair_weight),
            supcon_weight=float(supcon_weight),
            temperature=float(temperature),
            grad_clip_norm=float(grad_clip_norm),
        )
        eval_embeddings = extract_dualview_embeddings(
            df=analysis_eval_df,
            repo_root=repo_root,
            model=model,
            spec=spec,
            device=device,
            batch_size=int(eval_batch_size),
            num_workers=int(num_workers),
            global_weight=float(global_weight),
            trunk_weight=float(trunk_weight),
        )
        sweep_df, _prediction_df = run_threshold_sweep(
            df=analysis_eval_df.rename(columns={"global_path": PATH_COLUMN}),
            embeddings=eval_embeddings,
            thresholds=thresholds,
        )
        current_best_thresholds_df = pick_best_thresholds(sweep_df)
        current_best_row = current_best_thresholds_df.iloc[0]
        epoch_row = {
            "epoch": epoch,
            **train_metrics,
            "analysis_eval_ari": float(current_best_row["ari"]),
            "analysis_eval_pairwise_f1": float(current_best_row["pairwise_f1"]),
            "analysis_eval_threshold": float(current_best_row["threshold"]),
            "analysis_eval_recall_at_1": float(recall_at_k(eval_embeddings, analysis_eval_df["identity"].to_numpy(), k=1)),
            "analysis_eval_recall_at_5": float(recall_at_k(eval_embeddings, analysis_eval_df["identity"].to_numpy(), k=5)),
        }
        training_rows.append(epoch_row)
        training_log_df = pd.DataFrame(training_rows)
        training_log_df["macro_ari"] = training_log_df["analysis_eval_ari"]
        training_log_df["macro_recall_at_1"] = training_log_df["analysis_eval_recall_at_1"]
        training_log_df["SalamanderID2025_ari"] = training_log_df["analysis_eval_ari"]
        training_log_df["SalamanderID2025_threshold"] = training_log_df["analysis_eval_threshold"]
        training_log_df.to_csv(tables_dir / "training_log_v1.csv", index=False)
        write_training_monitor_plots(
            plots_dir=reports_dir / "plots",
            training_log_df=training_log_df,
            alignment_history_df=None,
        )
        if float(current_best_row["ari"]) > best_metric:
            best_metric = float(current_best_row["ari"])
            best_epoch = epoch
            best_thresholds_df = current_best_thresholds_df.copy()
            checkpoint_payload = {
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint_payload, checkpoints_dir / "best.pt")
    if best_epoch < 0:
        raise RuntimeError("No epoch completed for Salamander dual-view training.")

    best_checkpoint = torch.load(checkpoints_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()

    val_embeddings = extract_dualview_embeddings(
        df=analysis_eval_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=device,
        batch_size=int(eval_batch_size),
        num_workers=int(num_workers),
        global_weight=float(global_weight),
        trunk_weight=float(trunk_weight),
    )
    test_embeddings = extract_dualview_embeddings(
        df=test_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=device,
        batch_size=int(eval_batch_size),
        num_workers=int(num_workers),
        global_weight=float(global_weight),
        trunk_weight=float(trunk_weight),
    )
    np.save(embeddings_dir / "salamander_val_embeddings.npy", val_embeddings.astype(np.float32))
    np.save(embeddings_dir / "salamander_test_embeddings.npy", test_embeddings.astype(np.float32))
    analysis_eval_df.to_csv(embeddings_dir / "salamander_val_metadata.csv", index=False)
    test_df.to_csv(embeddings_dir / "salamander_test_metadata.csv", index=False)

    chosen_threshold = float(best_thresholds_df.iloc[0]["threshold"])
    salamander_test_pred_df = apply_thresholds_to_df(
        df=test_df.rename(columns={"global_path": PATH_COLUMN}),
        embeddings=test_embeddings,
        threshold_by_dataset={SALAMANDER_DATASET: chosen_threshold},
    )
    salamander_test_pred_df["route_name"] = experiment_id
    salamander_test_pred_df["embedding_dim"] = int(test_embeddings.shape[1])
    salamander_test_pred_df["rerank_enabled"] = False
    salamander_test_pred_df["local_weight"] = 0.0
    salamander_test_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)

    base_pred_df = pd.read_csv(base_predictions_path)
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    kept_df = base_pred_df[base_pred_df["dataset"] != SALAMANDER_DATASET].copy()
    merged_pred_df = pd.concat([kept_df, salamander_test_pred_df], ignore_index=True)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=repo_root / "sample_submission.csv",
        output_path=output_dir / "submission.csv",
    )

    cluster_counts = salamander_test_pred_df["pred_cluster_id"].value_counts()
    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": SALAMANDER_DATASET,
                "samples": int(len(salamander_test_pred_df)),
                "clusters": int(cluster_counts.size),
                "singleton_clusters": int((cluster_counts == 1).sum()),
                "singleton_ratio": round(float((cluster_counts == 1).mean()) if len(cluster_counts) else 0.0, 6),
                "route_name": experiment_id,
                "embedding_dim": int(test_embeddings.shape[1]),
                "threshold": chosen_threshold,
            }
        ]
    )
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)
    route_df = pd.DataFrame(
        [
            {
                "dataset": SALAMANDER_DATASET,
                "route_name": experiment_id,
                "embedding_dim": int(test_embeddings.shape[1]),
                "threshold": chosen_threshold,
                "rerank_enabled": False,
                "local_weight": 0.0,
            }
        ]
    )
    route_df.to_csv(tables_dir / "route_config_v1.csv", index=False)
    val_neighbors_df = build_neighbor_table(
        df=analysis_eval_df.rename(columns={"global_path": PATH_COLUMN}),
        embeddings=val_embeddings,
        top_k=5,
    )
    val_neighbors_df.to_csv(tables_dir / "val_neighbors_v1.csv", index=False)

    training_log_df = pd.read_csv(tables_dir / "training_log_v1.csv")
    summary_path = _write_summary_markdown(
        output_dir=output_dir,
        config=config,
        training_log_df=training_log_df,
        best_thresholds_df=best_thresholds_df,
        cluster_summary_df=cluster_summary_df,
        train_cache=cache_bundle,
    )
    return {
        "summary_path": summary_path,
        "best_checkpoint_path": checkpoints_dir / "best.pt",
        "submission_path": output_dir / "submission.csv",
        "test_predictions_path": tables_dir / "test_predictions_v1.csv",
        "salamander_predictions_path": tables_dir / "salamander_test_predictions_v1.csv",
        "test_embeddings_path": embeddings_dir / "salamander_test_embeddings.npy",
        "route_config_path": tables_dir / "route_config_v1.csv",
    }
