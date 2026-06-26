from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

try:  # pragma: no cover - exercised in the training env
    import matplotlib.pyplot as plt
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from torchvision import transforms as T
except ModuleNotFoundError:  # pragma: no cover - keeps helper imports light
    plt = None
    torch = None
    F = None

    class _NNProxy:
        Module = object
        Linear = object
        BatchNorm1d = object
        ModuleDict = dict

    nn = _NNProxy()
    DataLoader = Dataset = WeightedRandomSampler = object
    T = None

from .descriptor_baselines import (
    PATH_COLUMN,
    apply_thresholds_to_df,
    dataframe_to_markdown_table,
    ensure_metadata_alignment,
    fuse_embedding_blocks,
    l2_normalize,
    load_cached_embedding_bundle,
    recall_at_k,
)
from .texas_unsupervised import (
    DEFAULT_TOP_K,
    TEXAS_DATASET,
    build_topk_indices,
    mean_topk_neighbor_overlap,
    pair_agreement_score,
    summarize_cluster_labels,
)

if torch is not None:  # pragma: no cover - only imported in the training env
    from .supervised_training import (
        ArcFaceHead,
        STUDENT_BACKBONE_SPECS,
        build_eval_transform,
        build_train_transform,
        collect_resource_snapshot,
        compute_view_pair_contrastive_loss,
        describe_transform,
        load_student_backbone,
        scale_learning_rate,
        seed_everything,
        summarize_alignment,
    )
else:  # pragma: no cover - exercised in light unit-test envs
    ArcFaceHead = object
    STUDENT_BACKBONE_SPECS = {}


DEFAULT_SELFTRAIN_THRESHOLDS = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
DEFAULT_ANCHOR_THRESHOLD = 0.38
DEFAULT_PSEUDO_LOSS_WEIGHT = 1.0
DEFAULT_RELATION_DISTILL_WEIGHT = 0.0
DEFAULT_FEATURE_DISTILL_WEIGHT = 0.0
DEFAULT_VIEW_PAIR_WEIGHT = 0.5
DEFAULT_VIEW_PAIR_TEMPERATURE = 0.07
DEFAULT_FALLBACK_PSEUDO_POSITIVE_RECIPES = [
    "crop_jitter_tight_v1",
    "rotate_mild_pos5_v1",
    "rotate_mild_neg5_v1",
    "scale_focus_in_v1",
]
DEFAULT_TOPK_OVERLAP = DEFAULT_TOP_K
DEFAULT_SEED_OVERSAMPLE_FACTOR = 2.0


@dataclass(frozen=True)
class TexasTeacherBundle:
    embeddings: np.ndarray
    component_table: pd.DataFrame
    source_dirs: list[Path]
    weights: list[float]


@dataclass(frozen=True)
class TexasPseudoBundle:
    all_df: pd.DataFrame
    seed_df: pd.DataFrame
    candidate_pair_df: pd.DataFrame
    seed_class_summary_df: pd.DataFrame
    pseudo_label_map: dict[str, int]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _require_torch() -> None:
    if torch is None or T is None:
        raise ModuleNotFoundError("torch and torchvision are required for Texas self-training")


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for Texas self-training plots")


def prepare_texas_pseudo_frame(assignments_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], pd.DataFrame]:
    frame = assignments_df.copy().reset_index(drop=True)
    if "dataset" not in frame.columns:
        raise ValueError("assignments_df must contain a dataset column")
    datasets = sorted(frame["dataset"].dropna().astype(str).unique().tolist())
    if datasets != [TEXAS_DATASET]:
        raise ValueError(f"Texas self-train expects only {TEXAS_DATASET}, got {datasets}")
    frame["image_id"] = frame["image_id"].astype(str)
    if "identity" in frame.columns:
        frame["identity"] = frame["identity"].fillna("").astype(str)
    else:
        frame["identity"] = ""
    frame[PATH_COLUMN] = frame[PATH_COLUMN].astype(str)
    if "seed_status" in frame.columns:
        frame["seed_status"] = frame["seed_status"].fillna("uncertain").astype(str)
    else:
        frame["seed_status"] = "uncertain"
    if "pseudo_identity" in frame.columns:
        frame["pseudo_identity"] = frame["pseudo_identity"].fillna("").astype(str)
    else:
        frame["pseudo_identity"] = ""
    frame["is_seed"] = (frame["seed_status"] == "seed") & frame["pseudo_identity"].ne("")
    if "pseudo_source_routes" in frame.columns:
        frame["pseudo_source_routes"] = frame["pseudo_source_routes"].fillna("").astype(str)
    else:
        frame["pseudo_source_routes"] = ""
    if "component_size" in frame.columns:
        frame["component_size"] = frame["component_size"].fillna(1).astype(int)
    else:
        frame["component_size"] = 1
    if "component_density" in frame.columns:
        frame["component_density"] = frame["component_density"].fillna(0.0).astype(float)
    else:
        frame["component_density"] = 0.0
    if "component_edge_count" in frame.columns:
        frame["component_edge_count"] = frame["component_edge_count"].fillna(0).astype(int)
    else:
        frame["component_edge_count"] = 0

    pseudo_identities = sorted(frame.loc[frame["is_seed"], "pseudo_identity"].unique().tolist())
    pseudo_label_map = {pseudo_identity: index for index, pseudo_identity in enumerate(pseudo_identities)}
    frame["pseudo_label_index"] = -1
    if pseudo_label_map:
        frame.loc[frame["is_seed"], "pseudo_label_index"] = (
            frame.loc[frame["is_seed"], "pseudo_identity"].map(pseudo_label_map).astype(int)
        )
    frame["pseudo_image_count"] = 0
    if pseudo_label_map:
        counts = frame.loc[frame["is_seed"]].groupby("pseudo_identity")["image_id"].transform("size").astype(int)
        frame.loc[frame["is_seed"], "pseudo_image_count"] = counts

    seed_df = frame[frame["is_seed"]].copy().reset_index(drop=True)
    seed_class_summary_df = (
        seed_df.groupby("pseudo_identity")
        .agg(
            size=("image_id", "count"),
            mean_component_density=("component_density", "mean"),
            mean_component_size=("component_size", "mean"),
        )
        .reset_index()
        .sort_values(["size", "pseudo_identity"], ascending=[False, True])
        .reset_index(drop=True)
        if not seed_df.empty
        else pd.DataFrame(columns=["pseudo_identity", "size", "mean_component_density", "mean_component_size"])
    )
    return frame, pseudo_label_map, seed_class_summary_df


def load_texas_pseudo_bundle(
    assignments_path: Path,
    candidate_pair_path: Path | None = None,
) -> TexasPseudoBundle:
    assignments_df = pd.read_csv(assignments_path)
    all_df, pseudo_label_map, seed_class_summary_df = prepare_texas_pseudo_frame(assignments_df)
    seed_df = all_df[all_df["is_seed"]].copy().reset_index(drop=True)

    if candidate_pair_path is not None and candidate_pair_path.exists():
        candidate_pair_df = pd.read_csv(candidate_pair_path)
        if not candidate_pair_df.empty:
            candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
            candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)
            if "mutual_topk_all_routes" in candidate_pair_df.columns:
                candidate_pair_df["mutual_topk_all_routes"] = candidate_pair_df["mutual_topk_all_routes"].map(
                    lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"}
                )
    else:
        candidate_pair_df = pd.DataFrame(columns=["image_id", "neighbor_image_id", "mutual_topk_all_routes"])

    return TexasPseudoBundle(
        all_df=all_df,
        seed_df=seed_df,
        candidate_pair_df=candidate_pair_df,
        seed_class_summary_df=seed_class_summary_df,
        pseudo_label_map=pseudo_label_map,
    )


def apply_trusted_membership_to_pseudo_bundle(
    pseudo_bundle: TexasPseudoBundle,
    *,
    trusted_membership_path: Path,
) -> TexasPseudoBundle:
    """Replace seed assignments with a human-trusted membership table.

    The trusted table is intentionally flexible because review exports may use
    different names for the same concept. At minimum, it needs `image_id` plus
    one component/identity column.
    """

    trusted_df = pd.read_csv(trusted_membership_path)
    if trusted_df.empty:
        raise ValueError(f"Trusted membership file is empty: {trusted_membership_path}")
    trusted_df["image_id"] = trusted_df["image_id"].astype(str)
    if "dataset" in trusted_df.columns:
        trusted_df["dataset"] = trusted_df["dataset"].astype(str)
        trusted_df = trusted_df[trusted_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    else:
        trusted_df["dataset"] = TEXAS_DATASET
    if trusted_df.empty:
        raise ValueError(f"No Texas rows found in trusted membership: {trusted_membership_path}")

    pseudo_identity_column = next(
        (
            column
            for column in [
                "pseudo_identity",
                "trusted_identity",
                "trusted_component_id",
                "component_id",
            ]
            if column in trusted_df.columns
        ),
        None,
    )
    if pseudo_identity_column is None:
        raise ValueError(
            "Trusted membership must contain one of: pseudo_identity, trusted_identity, trusted_component_id, component_id"
        )
    trusted_df["pseudo_identity"] = trusted_df[pseudo_identity_column].astype(str)
    trusted_df["seed_status"] = "seed"

    frame = pseudo_bundle.all_df.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    merged = frame.merge(
        trusted_df[["image_id", "dataset", "pseudo_identity", "seed_status"]].drop_duplicates(),
        on=["image_id", "dataset"],
        how="left",
        suffixes=("", "__trusted"),
    )
    merged["pseudo_identity"] = merged["pseudo_identity__trusted"].fillna("").astype(str)
    merged["seed_status"] = merged["seed_status__trusted"].fillna("uncertain").astype(str)
    merged = merged.drop(
        columns=[column for column in ["pseudo_identity__trusted", "seed_status__trusted"] if column in merged.columns]
    )
    prepared_df, pseudo_label_map, seed_class_summary_df = prepare_texas_pseudo_frame(merged)
    seed_df = prepared_df[prepared_df["is_seed"]].copy().reset_index(drop=True)
    return TexasPseudoBundle(
        all_df=prepared_df,
        seed_df=seed_df,
        candidate_pair_df=pseudo_bundle.candidate_pair_df.copy(),
        seed_class_summary_df=seed_class_summary_df,
        pseudo_label_map=pseudo_label_map,
    )


def remap_texas_paths_from_manifest(
    pseudo_bundle: TexasPseudoBundle,
    *,
    manifest_path: Path,
) -> TexasPseudoBundle:
    manifest_df = pd.read_csv(manifest_path)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    manifest_df = manifest_df[manifest_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if manifest_df.empty:
        raise ValueError(f"No Texas rows found in manifest: {manifest_path}")

    update_columns = [
        column
        for column in [
            PATH_COLUMN,
            "path",
            "preferred_path_v1",
            "preprocess_variant_v1",
            "manifest_view_name_v1",
            "manifest_view_requested_v1",
            "manifest_view_resolved_v1",
            "manifest_view_applied_v1",
        ]
        if column in manifest_df.columns
    ]
    if PATH_COLUMN not in update_columns:
        raise ValueError(f"Manifest is missing required path column `{PATH_COLUMN}`: {manifest_path}")

    manifest_view = manifest_df[["image_id", "dataset", *update_columns]].copy()

    def _apply(frame: pd.DataFrame) -> pd.DataFrame:
        merged = frame.merge(manifest_view, on=["image_id", "dataset"], how="left", suffixes=("", "__manifest"))
        if merged[f"{PATH_COLUMN}__manifest"].isna().any():
            missing = (
                merged.loc[merged[f"{PATH_COLUMN}__manifest"].isna(), ["image_id", "dataset"]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(f"Texas manifest remap is missing image_id/dataset pairs, examples: {missing}")
        for column in update_columns:
            manifest_column = f"{column}__manifest" if f"{column}__manifest" in merged.columns else column
            merged[column] = merged[manifest_column]
            if manifest_column != column:
                merged = merged.drop(columns=[manifest_column])
        return merged.reset_index(drop=True)

    all_df = _apply(pseudo_bundle.all_df)
    seed_df = _apply(pseudo_bundle.seed_df) if not pseudo_bundle.seed_df.empty else pseudo_bundle.seed_df.copy()
    return TexasPseudoBundle(
        all_df=all_df,
        seed_df=seed_df,
        candidate_pair_df=pseudo_bundle.candidate_pair_df.copy(),
        seed_class_summary_df=pseudo_bundle.seed_class_summary_df.copy(),
        pseudo_label_map=dict(pseudo_bundle.pseudo_label_map),
    )


def build_texas_training_frame(
    pseudo_bundle: TexasPseudoBundle,
    *,
    pseudo_positive_pairs_path: Path | None = None,
) -> pd.DataFrame:
    """Expand training rows from optional pseudo-positive metadata.

    If a view-pair table is provided, each row becomes one training sample:
    - `base_image` always points to the trusted/default Texas preprocessing path
    - `positive_recipe` tells the dataset which deterministic positive view to
      synthesize from that same source image

    This keeps the training rule aligned with the user's requirement:
    compare augmented views against the SAM-based base image, not aug-vs-aug.
    """

    def _expand_with_recipes(
        frame: pd.DataFrame,
        *,
        recipes: list[str],
        pair_source: str,
        view_supervision_group: str,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        for recipe in recipes:
            piece = frame.copy()
            piece["positive_recipe"] = str(recipe)
            piece["pair_source"] = str(pair_source)
            piece["view_supervision_group"] = str(view_supervision_group)
            rows.append(piece)
        if not rows:
            return frame.iloc[0:0].copy()
        return pd.concat(rows, ignore_index=True)

    base_df = pseudo_bundle.all_df.copy().reset_index(drop=True)
    base_df["image_id"] = base_df["image_id"].astype(str)
    base_df["dataset"] = base_df["dataset"].astype(str)
    if pseudo_positive_pairs_path is None or not pseudo_positive_pairs_path.exists():
        return _expand_with_recipes(
            base_df,
            recipes=list(DEFAULT_FALLBACK_PSEUDO_POSITIVE_RECIPES),
            pair_source="auto_single_view_fallback_no_pair_table",
            view_supervision_group="all_images_single_view",
        ).reset_index(drop=True)

    pair_df = pd.read_csv(pseudo_positive_pairs_path)
    if pair_df.empty:
        return _expand_with_recipes(
            base_df,
            recipes=list(DEFAULT_FALLBACK_PSEUDO_POSITIVE_RECIPES),
            pair_source="auto_single_view_fallback_empty_pair_table",
            view_supervision_group="all_images_single_view",
        ).reset_index(drop=True)
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    if "dataset" not in pair_df.columns:
        pair_df["dataset"] = TEXAS_DATASET
    pair_df["dataset"] = pair_df["dataset"].astype(str)
    pair_df = pair_df[pair_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if pair_df.empty:
        return _expand_with_recipes(
            base_df,
            recipes=list(DEFAULT_FALLBACK_PSEUDO_POSITIVE_RECIPES),
            pair_source="auto_single_view_fallback_non_texas_pair_table",
            view_supervision_group="all_images_single_view",
        ).reset_index(drop=True)
    if "positive_recipe" not in pair_df.columns:
        if "positive_recipe_name" in pair_df.columns:
            pair_df["positive_recipe"] = pair_df["positive_recipe_name"].astype(str)
        elif "view_recipe_name" in pair_df.columns:
            pair_df["positive_recipe"] = pair_df["view_recipe_name"].astype(str)
        else:
            pair_df["positive_recipe"] = "train_aug"
    if "pair_source" not in pair_df.columns:
        if "pair_kind" in pair_df.columns:
            pair_df["pair_source"] = pair_df["pair_kind"].astype(str)
        elif "source_type" in pair_df.columns:
            pair_df["pair_source"] = pair_df["source_type"].fillna("").astype(str)
        else:
            pair_df["pair_source"] = "pseudo_positive_pairs"

    keep_columns = ["image_id", "dataset", "positive_recipe", "pair_source"]
    matched = base_df.merge(pair_df[keep_columns].drop_duplicates(), on=["image_id", "dataset"], how="inner")
    if matched.empty:
        raise ValueError(
            f"Pseudo-positive pairs did not match any Texas images from the pseudo bundle: {pseudo_positive_pairs_path}"
        )
    matched["view_supervision_group"] = np.where(
        matched["is_seed"].to_numpy(dtype=bool),
        "trusted_view_pairs",
        "untrusted_view_pairs_from_table",
    )

    matched_image_ids = set(matched["image_id"].astype(str).tolist())
    unmatched = base_df[~base_df["image_id"].astype(str).isin(matched_image_ids)].copy().reset_index(drop=True)
    if unmatched.empty:
        return matched.reset_index(drop=True)

    fallback = _expand_with_recipes(
        unmatched,
        recipes=list(DEFAULT_FALLBACK_PSEUDO_POSITIVE_RECIPES),
        pair_source="auto_single_view_fallback_unmatched",
        view_supervision_group="fallback_view_pairs",
    )
    fallback["view_supervision_group"] = np.where(
        fallback["is_seed"].to_numpy(dtype=bool),
        "trusted_view_fallback",
        "untrusted_view_fallback",
    )
    training_df = pd.concat([matched, fallback], ignore_index=True)
    return training_df.reset_index(drop=True)


def expand_teacher_embeddings_for_training_rows(
    training_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    teacher_embeddings: np.ndarray | None,
) -> np.ndarray | None:
    if teacher_embeddings is None:
        return None
    if len(metadata_df) != len(teacher_embeddings):
        raise ValueError(
            f"Teacher embedding row mismatch before training expansion: metadata={len(metadata_df)} vs embeddings={len(teacher_embeddings)}"
        )
    mapping = {
        str(image_id): teacher_embeddings[index]
        for index, image_id in enumerate(metadata_df["image_id"].astype(str).tolist())
    }
    rows: list[np.ndarray] = []
    missing: list[str] = []
    for image_id in training_df["image_id"].astype(str).tolist():
        vector = mapping.get(str(image_id))
        if vector is None:
            missing.append(str(image_id))
            continue
        rows.append(vector)
    if missing:
        raise ValueError(f"Training rows reference image_ids missing from teacher cache, examples: {missing[:5]}")
    return np.stack(rows, axis=0).astype(np.float32, copy=False)


def apply_positive_recipe(
    image: Image.Image,
    *,
    recipe: str,
    train_transform: Any,
    eval_transform: Any,
) -> torch.Tensor:
    recipe_name = str(recipe).strip().lower()
    if recipe_name in {"", "train_aug", "default"}:
        return train_transform(image)
    if recipe_name == "identity_eval":
        return eval_transform(image)
    if recipe_name in {"hflip", "horizontal_flip_gated_v1"}:
        return eval_transform(ImageOps.mirror(image))
    if recipe_name in {"mild_rotate_p6", "rotate_mild_pos5_v1"}:
        degrees = 6.0 if recipe_name == "mild_rotate_p6" else 5.0
        return eval_transform(image.rotate(degrees, resample=Image.BILINEAR))
    if recipe_name in {"mild_rotate_n6", "rotate_mild_neg5_v1"}:
        degrees = -6.0 if recipe_name == "mild_rotate_n6" else -5.0
        return eval_transform(image.rotate(degrees, resample=Image.BILINEAR))
    if recipe_name in {"center_crop_92", "crop_jitter_tight_v1"}:
        width, height = image.size
        scale = 0.92 if recipe_name == "center_crop_92" else 0.96
        crop_w = max(4, int(round(width * scale)))
        crop_h = max(4, int(round(height * scale)))
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        return eval_transform(image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR))
    if recipe_name == "center_crop_88":
        width, height = image.size
        crop_w = max(4, int(round(width * 0.88)))
        crop_h = max(4, int(round(height * 0.88)))
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        return eval_transform(image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR))
    if recipe_name == "scale_focus_in_v1":
        width, height = image.size
        crop_w = max(4, int(round(width / 1.04)))
        crop_h = max(4, int(round(height / 1.04)))
        left = max(0, (width - crop_w) // 2)
        top = max(0, (height - crop_h) // 2)
        return eval_transform(image.crop((left, top, left + crop_w, top + crop_h)).resize((width, height), Image.BILINEAR))
    return train_transform(image)


def build_candidate_index_pairs(
    metadata_df: pd.DataFrame,
    candidate_pair_df: pd.DataFrame,
    *,
    mutual_topk_only: bool,
) -> list[tuple[int, int]]:
    if candidate_pair_df.empty:
        return []
    pair_df = candidate_pair_df.copy()
    if mutual_topk_only and "mutual_topk_all_routes" in pair_df.columns:
        pair_df = pair_df[pair_df["mutual_topk_all_routes"]].copy()
    if pair_df.empty:
        return []
    image_to_index = {str(image_id): index for index, image_id in enumerate(metadata_df["image_id"].astype(str).tolist())}
    pairs: list[tuple[int, int]] = []
    for row in pair_df.itertuples(index=False):
        left = image_to_index.get(str(row.image_id))
        right = image_to_index.get(str(row.neighbor_image_id))
        if left is None or right is None or left == right:
            continue
        pairs.append((min(left, right), max(left, right)))
    return sorted(set(pairs))


def compute_pair_keep_ratio(labels: np.ndarray, index_pairs: list[tuple[int, int]]) -> float:
    if not index_pairs:
        return 0.0
    hits = [bool(labels[left] == labels[right]) for left, right in index_pairs]
    return round(float(np.mean(hits)), 6)


def pick_best_texas_threshold(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        raise ValueError("summary_df must not be empty")
    sortable = summary_df.copy()
    sortable["proxy_score"] = sortable.apply(compute_threshold_proxy_score, axis=1)
    best = sortable.sort_values(
        [
            "proxy_score",
            "cluster_delta_vs_teacher_anchor",
            "largest_cluster_size",
            "threshold",
        ],
        ascending=[False, True, True, True],
    ).iloc[0]
    return pd.DataFrame([best]).reset_index(drop=True)


def evaluate_texas_thresholds(
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    thresholds: list[float],
    anchor_threshold: float,
    candidate_pair_df: pd.DataFrame,
    teacher_anchor_labels: np.ndarray | None = None,
    teacher_topk_indices: np.ndarray | None = None,
    top_k: int = DEFAULT_TOPK_OVERLAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_mask = metadata_df["is_seed"].to_numpy(dtype=bool)
    seed_labels = metadata_df.loc[seed_mask, "pseudo_label_index"].to_numpy(dtype=int) if seed_mask.any() else np.array([], dtype=int)
    all_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=False)
    mutual_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=True)
    student_topk_indices = build_topk_indices(embeddings, top_k=top_k)
    student_teacher_topk_overlap = (
        mean_topk_neighbor_overlap(student_topk_indices, teacher_topk_indices)
        if teacher_topk_indices is not None and student_topk_indices.shape == teacher_topk_indices.shape
        else np.nan
    )

    predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    anchor_labels: np.ndarray | None = None
    target_clusters = int(summarize_cluster_labels(teacher_anchor_labels)["clusters"]) if teacher_anchor_labels is not None else np.nan

    for threshold in thresholds:
        pred_df = apply_thresholds_to_df(
            df=metadata_df,
            embeddings=embeddings,
            threshold_by_dataset={TEXAS_DATASET: float(threshold)},
        )
        pred_df["threshold"] = float(threshold)
        labels = pred_df["pred_cluster_id"].to_numpy()
        stats = summarize_cluster_labels(labels)
        row: dict[str, object] = {
            "threshold": float(threshold),
            "samples": int(len(pred_df)),
            **stats,
            "seed_pair_agreement": pair_agreement_score(labels[seed_mask], seed_labels) if seed_mask.any() else np.nan,
            "seed_recall_at_1": recall_at_k(embeddings[seed_mask], seed_labels, k=1) if seed_mask.any() else np.nan,
            "candidate_pair_keep_ratio": compute_pair_keep_ratio(labels, all_candidate_pairs),
            "mutual_topk_pair_keep_ratio": compute_pair_keep_ratio(labels, mutual_candidate_pairs),
            "student_teacher_topk_overlap": student_teacher_topk_overlap,
            "teacher_anchor_clusters": target_clusters,
            "cluster_delta_vs_teacher_anchor": abs(int(stats["clusters"]) - int(target_clusters))
            if teacher_anchor_labels is not None
            else np.nan,
            "pair_agreement_vs_teacher_anchor": pair_agreement_score(labels, teacher_anchor_labels)
            if teacher_anchor_labels is not None
            else np.nan,
        }
        predictions.append(pred_df)
        summary_rows.append(row)
        if math.isclose(float(threshold), float(anchor_threshold), rel_tol=0.0, abs_tol=1e-9):
            anchor_labels = labels.copy()

    if anchor_labels is None:
        raise ValueError(f"anchor threshold {anchor_threshold} not found in threshold list")

    for row, pred_df in zip(summary_rows, predictions, strict=True):
        row["pair_agreement_vs_student_anchor"] = pair_agreement_score(pred_df["pred_cluster_id"].to_numpy(), anchor_labels)

    summary_df = pd.DataFrame(summary_rows).sort_values("threshold").reset_index(drop=True)
    if not summary_df.empty:
        summary_df["proxy_score"] = summary_df.apply(compute_threshold_proxy_score, axis=1)
    predictions_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return summary_df, predictions_df


def compute_threshold_proxy_score(row: pd.Series) -> float:
    seed_pair_agreement = _safe_float(row.get("seed_pair_agreement", 0.0), default=0.0)
    mutual_topk_pair_keep_ratio = _safe_float(row.get("mutual_topk_pair_keep_ratio", 0.0), default=0.0)
    seed_recall_at_1 = _safe_float(row.get("seed_recall_at_1", 0.0), default=0.0)
    cluster_delta = _safe_float(row.get("cluster_delta_vs_teacher_anchor", 0.0), default=0.0)
    largest_cluster_size = _safe_float(row.get("largest_cluster_size", 0.0), default=0.0)
    score = (0.45 * seed_pair_agreement) + (0.35 * mutual_topk_pair_keep_ratio) + (0.20 * seed_recall_at_1)
    score = score - (0.002 * cluster_delta) - (0.001 * largest_cluster_size)
    return round(score, 6)


def build_proxy_selection_tuple(row: pd.Series) -> tuple[float, float, float, float, float]:
    return (
        compute_threshold_proxy_score(row),
        float(row.get("seed_pair_agreement", -1.0)),
        float(row.get("mutual_topk_pair_keep_ratio", -1.0)),
        -float(row.get("cluster_delta_vs_teacher_anchor", np.inf)),
        -float(row.get("largest_cluster_size", np.inf)),
    )


def build_seed_sampling_weights(df: pd.DataFrame, *, seed_oversample_factor: float) -> np.ndarray:
    if seed_oversample_factor <= 1.0:
        return np.ones(len(df), dtype=np.float32)
    weights = np.ones(len(df), dtype=np.float32)
    if "is_seed" not in df.columns:
        return weights
    seed_mask = df["is_seed"].to_numpy(dtype=bool)
    weights[seed_mask] = float(seed_oversample_factor)
    return weights


def build_texas_teacher_bundle(
    metadata_df: pd.DataFrame,
    *,
    source_dirs: list[Path],
    weights: list[float] | None = None,
) -> TexasTeacherBundle:
    if not source_dirs:
        raise ValueError("Need at least one teacher source dir")
    if weights is None:
        weights = [1.0] * len(source_dirs)
    if len(weights) != len(source_dirs):
        raise ValueError("teacher weights length must match source_dirs")

    blocks: list[np.ndarray] = []
    component_rows: list[dict[str, object]] = []
    reference_df = metadata_df[["image_id", "dataset", PATH_COLUMN]].copy().reset_index(drop=True)
    for source_dir, weight in zip(source_dirs, weights, strict=True):
        bundle = load_cached_embedding_bundle(source_dir=source_dir.resolve(), name=source_dir.name, weight=float(weight))
        texas_df = bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
        texas_embeddings = bundle.test_embeddings[(bundle.test_df["dataset"] == TEXAS_DATASET).to_numpy()].astype(np.float32, copy=False)
        ref = reference_df[["image_id", "dataset"]].copy().reset_index(drop=True)
        cand = texas_df[["image_id", "dataset"]].copy().reset_index(drop=True)
        ref["image_id"] = ref["image_id"].astype(str)
        cand["image_id"] = cand["image_id"].astype(str)
        merged = ref.merge(
            cand.assign(_row=np.arange(len(cand), dtype=np.int32)),
            on=["image_id", "dataset"],
            how="left",
            validate="one_to_one",
        )
        if merged["_row"].isna().any():
            missing = merged.loc[merged["_row"].isna(), ["image_id", "dataset"]].head(5).to_dict(orient="records")
            raise ValueError(f"Texas teacher block is missing image_id/dataset pairs for {source_dir.name}, examples: {missing}")
        reorder_index = merged["_row"].astype(int).to_numpy()
        texas_df = texas_df.iloc[reorder_index].reset_index(drop=True).copy()
        texas_embeddings = texas_embeddings[reorder_index]
        ensure_metadata_alignment(
            reference_df=reference_df[["image_id", "dataset"]].copy(),
            candidate_df=texas_df[["image_id", "dataset"]].copy(),
            split_name="texas_selftrain_teacher",
            reference_name="pseudo_assignments",
            candidate_name=source_dir.name,
        )
        blocks.append(texas_embeddings)
        component_rows.append(
            {
                "teacher_source": source_dir.name,
                "source_dir": str(source_dir.resolve()),
                "weight": float(weight),
                "embedding_dim": int(texas_embeddings.shape[1]),
            }
        )
    fused = fuse_embedding_blocks(blocks, weights=weights).astype(np.float32, copy=False)
    return TexasTeacherBundle(
        embeddings=fused,
        component_table=pd.DataFrame(component_rows),
        source_dirs=[path.resolve() for path in source_dirs],
        weights=[float(weight) for weight in weights],
    )


class TexasSelfTrainDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        repo_root: Path,
        base_transform: Any,
        positive_transform: Any,
        teacher_embeddings: np.ndarray | None = None,
    ) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.base_transform = base_transform
        self.positive_transform = positive_transform
        self.teacher_embeddings = (
            teacher_embeddings.astype(np.float32, copy=False) if teacher_embeddings is not None else None
        )
        if self.teacher_embeddings is not None and len(self.df) != len(self.teacher_embeddings):
            raise ValueError(f"Teacher embedding row mismatch: df={len(self.df)} vs embeddings={len(self.teacher_embeddings)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            base_tensor = self.base_transform(image)
            positive_tensor = apply_positive_recipe(
                image,
                recipe=str(row.get("positive_recipe", "train_aug")),
                train_transform=self.positive_transform,
                eval_transform=self.base_transform,
            )
        payload = {
            "base_image": base_tensor,
            "positive_image": positive_tensor,
            "is_seed": bool(row["is_seed"]),
            "pseudo_label_index": int(row["pseudo_label_index"]),
        }
        if self.teacher_embeddings is not None:
            payload["teacher_embedding"] = torch.from_numpy(self.teacher_embeddings[index])
        return payload


class TexasInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, repo_root: Path, transform: Any) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


class TexasSelfTrainModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        *,
        feature_dim: int,
        embedding_dim: int,
        teacher_dim: int,
        pseudo_class_count: int,
        classification_head: str,
        arcface_scale: float,
        arcface_margin: float,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embedding_layer = nn.Linear(feature_dim, embedding_dim, bias=False)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.teacher_projection = nn.Linear(embedding_dim, teacher_dim, bias=False) if teacher_dim > 0 else None
        self.classification_head = classification_head
        if pseudo_class_count <= 0:
            self.pseudo_head = None
        elif classification_head == "arcface":
            self.pseudo_head = ArcFaceHead(
                in_features=embedding_dim,
                out_features=pseudo_class_count,
                scale=arcface_scale,
                margin=arcface_margin,
            )
        elif classification_head == "linear":
            self.pseudo_head = nn.Linear(embedding_dim, pseudo_class_count)
        else:
            raise ValueError(f"Unsupported classification head: {classification_head}")

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

    def classify(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.pseudo_head is None:
            raise RuntimeError("pseudo head is not initialized")
        if self.classification_head == "arcface":
            return self.pseudo_head(embeddings, labels)
        return self.pseudo_head(embeddings)


def compute_texas_pseudo_loss(
    model: TexasSelfTrainModel,
    embeddings: torch.Tensor,
    pseudo_label_indices: torch.Tensor,
    is_seed: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    mask = is_seed & (pseudo_label_indices >= 0)
    if not torch.any(mask):
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    labels = pseudo_label_indices[mask].long()
    logits = model.classify(embeddings[mask], labels)
    return F.cross_entropy(logits, labels, label_smoothing=label_smoothing)


def compute_feature_distillation_loss(
    projected_embeddings: torch.Tensor | None,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if projected_embeddings is None:
        return torch.zeros((), device=teacher_embeddings.device, dtype=teacher_embeddings.dtype)
    cosine = F.cosine_similarity(projected_embeddings, teacher_embeddings, dim=1)
    return (1.0 - cosine).mean()


def compute_texas_relation_distillation_loss(
    student_embeddings: torch.Tensor,
    teacher_embeddings: torch.Tensor,
) -> torch.Tensor:
    if student_embeddings.shape[0] < 2:
        return torch.zeros((), device=student_embeddings.device, dtype=student_embeddings.dtype)
    student_similarity = student_embeddings @ student_embeddings.T
    teacher_similarity = teacher_embeddings @ teacher_embeddings.T
    return F.mse_loss(student_similarity, teacher_similarity)


def build_texas_optimizer(
    model: TexasSelfTrainModel,
    *,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone_params = list(model.backbone.parameters())
    head_modules = [model.embedding_layer, model.embedding_bn]
    if model.teacher_projection is not None:
        head_modules.append(model.teacher_projection)
    if model.pseudo_head is not None:
        head_modules.append(model.pseudo_head)
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


def load_texas_selftrain_init_checkpoint(
    *,
    model: TexasSelfTrainModel,
    checkpoint_path: Path,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []

    # Only transplant the shared student trunk. Warmup classes and current
    # trusted-seed classes differ, so the pseudo head is normally incompatible.
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

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "loaded_key_count": len(loaded_keys),
        "skipped_key_count": len(skipped_keys),
        "loaded_keys_preview": loaded_keys[:20],
        "skipped_keys_preview": skipped_keys[:20],
    }


def train_texas_one_epoch(
    model: TexasSelfTrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    *,
    device: str,
    scaler,
    pseudo_loss_weight: float,
    relation_distill_weight: float,
    feature_distill_weight: float,
    view_pair_weight: float,
    view_pair_temperature: float,
    label_smoothing: float,
    grad_clip_norm: float,
    max_train_batches: int | None,
) -> dict[str, float]:
    model.train()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    totals = {
        "loss": 0.0,
        "pseudo_loss": 0.0,
        "relation_distill_loss": 0.0,
        "feature_distill_loss": 0.0,
        "view_pair_loss": 0.0,
        "seed_fraction": 0.0,
        "batches": 0,
    }
    use_amp = device.startswith("cuda")

    for batch_index, batch in enumerate(loader, start=1):
        if max_train_batches is not None and batch_index > max_train_batches:
            break
        base_images = batch["base_image"].to(device, non_blocking=True)
        positive_images = batch["positive_image"].to(device, non_blocking=True)
        teacher_embeddings = (
            F.normalize(batch["teacher_embedding"].to(device, non_blocking=True), dim=1)
            if "teacher_embedding" in batch
            else None
        )
        is_seed = batch["is_seed"].to(device, non_blocking=True).bool()
        pseudo_label_indices = batch["pseudo_label_index"].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            embeddings = model.encode(positive_images)
            pseudo_loss = compute_texas_pseudo_loss(
                model=model,
                embeddings=embeddings,
                pseudo_label_indices=pseudo_label_indices,
                is_seed=is_seed,
                label_smoothing=label_smoothing,
            )
            if teacher_embeddings is not None and feature_distill_weight > 0:
                projected_teacher = model.project_teacher_space(embeddings)
                feature_distill_loss = compute_feature_distillation_loss(projected_teacher, teacher_embeddings)
            else:
                feature_distill_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            if teacher_embeddings is not None and relation_distill_weight > 0:
                relation_distill_loss = compute_texas_relation_distillation_loss(
                    student_embeddings=embeddings,
                    teacher_embeddings=teacher_embeddings,
                )
            else:
                relation_distill_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            if view_pair_weight > 0:
                base_embeddings = model.encode(base_images)
                view_pair_loss = compute_view_pair_contrastive_loss(
                    base_embeddings=base_embeddings,
                    augmented_embeddings=embeddings,
                    dataset_indices=torch.zeros(len(embeddings), device=embeddings.device, dtype=torch.long),
                    temperature=view_pair_temperature,
                )
            else:
                view_pair_loss = torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
            loss = (pseudo_loss_weight * pseudo_loss)
            loss = loss + (feature_distill_weight * feature_distill_loss)
            loss = loss + (relation_distill_weight * relation_distill_loss)
            loss = loss + (view_pair_weight * view_pair_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if (not use_amp) or (scaler.get_scale() >= scale_before_step):
            scheduler.step()

        totals["loss"] += float(loss.detach().cpu())
        totals["pseudo_loss"] += float(pseudo_loss.detach().cpu())
        totals["relation_distill_loss"] += float(relation_distill_loss.detach().cpu())
        totals["feature_distill_loss"] += float(feature_distill_loss.detach().cpu())
        totals["view_pair_loss"] += float(view_pair_loss.detach().cpu())
        totals["seed_fraction"] += float(is_seed.float().mean().detach().cpu())
        totals["batches"] += 1

    batches = max(1, totals["batches"])
    return {
        "train_loss": round(totals["loss"] / batches, 6),
        "train_pseudo_loss": round(totals["pseudo_loss"] / batches, 6),
        "train_relation_distill_loss": round(totals["relation_distill_loss"] / batches, 6),
        "train_feature_distill_loss": round(totals["feature_distill_loss"] / batches, 6),
        "train_view_pair_loss": round(totals["view_pair_loss"] / batches, 6),
        "mean_seed_fraction": round(totals["seed_fraction"] / batches, 6),
        "peak_cuda_memory_mb": round(float(torch.cuda.max_memory_allocated() / (1024**2)), 2)
        if device.startswith("cuda")
        else 0.0,
    }


def extract_texas_student_embeddings(
    df: pd.DataFrame,
    repo_root: Path,
    model: TexasSelfTrainModel,
    *,
    transform: Any,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    _require_torch()
    dataset = TexasInferenceDataset(df=df, repo_root=repo_root, transform=transform)
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
            rows.append(embeddings.detach().cpu().numpy().astype(np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return l2_normalize(np.concatenate(rows, axis=0))


def write_texas_selftrain_plots(
    plots_dir: Path,
    training_log_df: pd.DataFrame,
    alignment_history_df: pd.DataFrame,
) -> dict[str, Path]:
    _require_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, Path] = {}
    if training_log_df.empty:
        return plot_paths

    epoch_values = training_log_df["epoch"].astype(float).to_numpy()

    loss_path = plots_dir / "training_loss_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epoch_values, training_log_df["train_loss"], marker="o", linewidth=2, color="#1f77b4", label="total")
    axes[0].set_title("Texas Self-Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    for column, label, color in [
        ("train_pseudo_loss", "Pseudo", "#d62728"),
        ("train_relation_distill_loss", "Relation Distill", "#2ca02c"),
        ("train_feature_distill_loss", "Feature Distill", "#ff7f0e"),
        ("train_view_pair_loss", "View Pair", "#9467bd"),
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
    plot_paths["loss"] = loss_path

    proxy_path = plots_dir / "proxy_metric_curves.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(
        epoch_values,
        training_log_df["best_seed_pair_agreement"],
        marker="o",
        linewidth=2,
        label="seed pair agreement",
        color="#1f77b4",
    )
    axes[0].plot(
        epoch_values,
        training_log_df["best_mutual_topk_pair_keep_ratio"],
        marker="o",
        linewidth=2,
        label="mutual-topk pair keep ratio",
        color="#ff7f0e",
    )
    axes[0].plot(
        epoch_values,
        training_log_df["best_seed_recall_at_1"],
        marker="o",
        linewidth=2,
        label="seed Recall@1",
        color="#2ca02c",
    )
    axes[0].set_title("Pseudo Proxy Metrics")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Metric")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(
        epoch_values,
        training_log_df["best_threshold"],
        marker="o",
        linewidth=2,
        label="chosen threshold",
        color="#9467bd",
    )
    axes[1].plot(
        epoch_values,
        training_log_df["best_cluster_count"],
        marker="o",
        linewidth=2,
        label="chosen clusters",
        color="#8c564b",
    )
    axes[1].plot(
        epoch_values,
        training_log_df["best_largest_cluster_size"],
        marker="o",
        linewidth=2,
        label="largest cluster size",
        color="#e377c2",
    )
    axes[1].set_title("Chosen Threshold Cluster Diagnostics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")
    fig.savefig(proxy_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    plot_paths["proxy"] = proxy_path

    if not alignment_history_df.empty:
        alignment_path = plots_dir / "alignment_curves.png"
        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        overall_df = alignment_history_df[alignment_history_df["dataset"] == "ALL_VAL"].copy()
        if overall_df.empty:
            overall_df = alignment_history_df.copy()
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


def write_texas_selftrain_report(
    output_path: Path,
    *,
    config: dict[str, object],
    training_log_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    best_threshold_df: pd.DataFrame,
    seed_class_summary_df: pd.DataFrame,
    teacher_component_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    plot_paths: dict[str, Path],
) -> None:
    best_epoch_row = training_log_df.sort_values(
        [
            "best_proxy_score",
            "best_seed_pair_agreement",
            "best_mutual_topk_pair_keep_ratio",
            "best_seed_recall_at_1",
            "epoch",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    selected_gpu = config.get("resource_snapshot", {}).get("selected_gpu", {})
    tmux_sessions = config.get("resource_snapshot", {}).get("tmux_sessions", [])
    lines = [
        "# Texas Self-Train Summary",
        "",
        "## Experiment Card",
        "",
        f"- `experiment_id`: `{config['experiment_id']}`",
        f"- `goal`: `{config['goal']}`",
        f"- `dataset`: `{TEXAS_DATASET}`",
        f"- `student_backbone`: `{config['student_backbone']}`",
        f"- `student_model_id`: `{config['student_model_id']}`",
        f"- `teacher_source_dirs`: `{config['teacher_source_dirs']}`",
        f"- `student_init_checkpoint`: `{config.get('student_init_checkpoint', '') or 'none'}`",
        f"- `student_init_info`: `{config.get('student_init_info', {})}`",
        f"- `anchor_threshold`: `{config['anchor_threshold']}`",
        f"- `trusted_membership_path`: `{config['trusted_membership_path']}`",
        f"- `pseudo_positive_pairs_path`: `{config['pseudo_positive_pairs_path']}`",
        "",
        "## Pseudo Labels",
        "",
        f"- `total_images`: `{config['total_images']}`",
        f"- `seed_images`: `{config['seed_images']}`",
        f"- `seed_coverage_ratio`: `{config['seed_coverage_ratio']}`",
        f"- `seed_clusters`: `{config['seed_clusters']}`",
        f"- `uncertain_images`: `{config['uncertain_images']}`",
        f"- `candidate_pairs`: `{config['candidate_pairs']}`",
        f"- `mutual_topk_pairs`: `{config['mutual_topk_pairs']}`",
        f"- `trusted_images_after_override`: `{config['trusted_images_after_override']}`",
        f"- `untrusted_images_after_override`: `{config['untrusted_images_after_override']}`",
        "",
        dataframe_to_markdown_table(seed_class_summary_df.head(20)),
        "",
        "## Training Config",
        "",
        f"- `input_size`: `{config['input_size']}`",
        f"- `student_feature_shape`: `B x {config['student_feature_dim']}`",
        f"- `student_embedding_shape`: `B x {config['embedding_dim']}`",
        f"- `teacher_fused_shape`: `B x {config['teacher_dim']}`",
        f"- `train_augmentation`: `{config['train_augmentation']}`",
        f"- `eval_preprocess`: `{config['eval_preprocess']}`",
        f"- `classification_head`: `{config['classification_head']}`",
        f"- `pseudo_head_shape`: `{config['pseudo_head_shape']}`",
        f"- `seed`: `{config['seed']}`",
        f"- `seed_oversample_factor`: `{config['seed_oversample_factor']}`",
        f"- `view_pair_weight`: `{config['view_pair_weight']}`",
        f"- `view_pair_temperature`: `{config['view_pair_temperature']}`",
        f"- `train_rows_after_view_expand`: `{config['train_rows_after_view_expand']}`",
        f"- `trusted_view_rows`: `{config['trusted_view_rows']}`",
        f"- `untrusted_view_rows`: `{config['untrusted_view_rows']}`",
        f"- `trusted_view_images`: `{config['trusted_view_images']}`",
        f"- `untrusted_view_images`: `{config['untrusted_view_images']}`",
        f"- `view_row_sources`: `{config['view_row_sources']}`",
        f"- `per_device_train_batch`: `{config['train_batch_size']}`",
        f"- `per_device_eval_batch`: `{config['eval_batch_size']}`",
        f"- `effective_batch_size`: `{config['effective_batch_size']}`",
        f"- `optimizer`: `AdamW`",
        f"- `reference_backbone_lr / reference_head_lr`: `{config['backbone_lr']} / {config['head_lr']}`",
        f"- `resolved_backbone_lr / resolved_head_lr`: `{config['resolved_backbone_lr']} / {config['resolved_head_lr']}`",
        f"- `weight_decay`: `{config['weight_decay']}`",
        f"- `scheduler`: `linear warmup + cosine decay`",
        f"- `warmup_ratio`: `{config['warmup_ratio']}`",
        f"- `epochs`: `{config['epochs']}`",
        f"- `amp_enabled`: `{config['amp_enabled']}`",
        f"- `grad_clip_norm`: `{config['grad_clip_norm']}`",
        f"- `loss_weights`: `pseudo={config['pseudo_loss_weight']}, relation={config['relation_distill_weight']}, feature={config['feature_distill_weight']}, view_pair={config['view_pair_weight']}`",
        "",
        "## Best Proxy Result",
        "",
        f"- `best_epoch`: `{int(best_epoch_row['epoch'])}`",
        f"- `best_threshold`: `{float(best_epoch_row['best_threshold']):.2f}`",
        f"- `best_proxy_score`: `{float(best_epoch_row['best_proxy_score']):.6f}`",
        f"- `best_seed_pair_agreement`: `{float(best_epoch_row['best_seed_pair_agreement']):.6f}`",
        f"- `best_mutual_topk_pair_keep_ratio`: `{float(best_epoch_row['best_mutual_topk_pair_keep_ratio']):.6f}`",
        f"- `best_seed_recall_at_1`: `{float(best_epoch_row['best_seed_recall_at_1']):.6f}`",
        f"- `best_cluster_count`: `{int(best_epoch_row['best_cluster_count'])}`",
        f"- `best_largest_cluster_size`: `{int(best_epoch_row['best_largest_cluster_size'])}`",
        f"- `peak_cuda_memory_mb`: `{float(best_epoch_row['peak_cuda_memory_mb']):.2f}`",
        "",
        "## Chosen Threshold Summary",
        "",
        dataframe_to_markdown_table(best_threshold_df),
        "",
        "## Full Threshold Sweep",
        "",
        dataframe_to_markdown_table(threshold_summary_df),
        "",
        "## Resource Snapshot",
        "",
        f"- `device`: `{config['device']}`",
        f"- `selected_gpu`: `{selected_gpu}`",
        f"- `resource_decision`: `{config['resource_decision']}`",
        f"- `probe_reuse_note`: `{config['probe_reuse_note']}`",
        f"- `tmux_sessions_at_launch`: `{tmux_sessions}`",
        "",
        "## Monitoring Figures",
        "",
    ]
    if not teacher_component_df.empty:
        lines.extend(
            [
                "## Teacher Components",
                "",
                dataframe_to_markdown_table(teacher_component_df),
                "",
            ]
        )
    if not alignment_df.empty:
        lines.extend(
            [
                "## Teacher / Student Alignment",
                "",
                dataframe_to_markdown_table(alignment_df),
                "",
            ]
        )
    if "loss" in plot_paths:
        rel = Path(os.path.relpath(plot_paths["loss"], start=output_path.parent))
        lines.extend(
            [
                f"![Training loss curves]({rel.as_posix()})",
                "",
                "- 读图方式：先看总 `train_loss` 是否下降，再看 `pseudo / relation / feature / view_pair` 哪个分量在主导优化。",
                "",
            ]
        )
    if "proxy" in plot_paths:
        rel = Path(os.path.relpath(plot_paths["proxy"], start=output_path.parent))
        lines.extend(
            [
                f"![Pseudo proxy curves]({rel.as_posix()})",
                "",
                "- 读图方式：上图越高越好；下图先看选中阈值是否稳定，再看 `largest_cluster_size` 是否出现塌缩。",
                "",
            ]
        )
    if "alignment" in plot_paths:
        rel = Path(os.path.relpath(plot_paths["alignment"], start=output_path.parent))
        lines.extend(
            [
                f"![Alignment curves]({rel.as_posix()})",
                "",
                "- 读图方式：`relation_mse / relation_mae` 越低越好，`relation_corr` 越高越好。它反映 student 是否还在贴近 teacher 的相似度结构。",
                "",
            ]
        )
    lines.extend(
        [
            "## Epoch Log",
            "",
            dataframe_to_markdown_table(training_log_df),
            "",
            "## Conclusion And Next Decision",
            "",
            f"- `current_best_judgment`: `{config['next_step_judgment']}`",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_texas_selftrain(
    repo_root: Path,
    output_dir: Path,
    *,
    experiment_id: str,
    assignments_path: Path,
    candidate_pair_path: Path,
    test_manifest_path: Path | None = None,
    trusted_membership_path: Path | None = None,
    pseudo_positive_pairs_path: Path | None = None,
    teacher_source_dirs: list[Path] | None = None,
    teacher_weights: list[float] | None = None,
    student_backbone: str = "miew",
    student_init_checkpoint: Path | None = None,
    device: str = "cuda:0",
    epochs: int = 8,
    embedding_dim: int = 512,
    train_batch_size: int | None = None,
    eval_batch_size: int | None = None,
    num_workers: int = 4,
    thresholds: list[float] | None = None,
    anchor_threshold: float = DEFAULT_ANCHOR_THRESHOLD,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    lr_reference_batch_size: int = 4,
    lr_scale_mode: str = "linear",
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    classification_head: str = "arcface",
    arcface_scale: float = 30.0,
    arcface_margin: float = 0.3,
    pseudo_loss_weight: float = DEFAULT_PSEUDO_LOSS_WEIGHT,
    relation_distill_weight: float = DEFAULT_RELATION_DISTILL_WEIGHT,
    feature_distill_weight: float = DEFAULT_FEATURE_DISTILL_WEIGHT,
    view_pair_weight: float = DEFAULT_VIEW_PAIR_WEIGHT,
    view_pair_temperature: float = DEFAULT_VIEW_PAIR_TEMPERATURE,
    label_smoothing: float = 0.0,
    grad_clip_norm: float = 1.0,
    max_train_batches: int | None = None,
    goal: str | None = None,
    resource_decision: str | None = None,
    probe_reuse_note: str | None = None,
    top_k: int = DEFAULT_TOPK_OVERLAP,
    seed: int = 42,
    seed_oversample_factor: float = DEFAULT_SEED_OVERSAMPLE_FACTOR,
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

    if thresholds is None:
        thresholds = DEFAULT_SELFTRAIN_THRESHOLDS
    thresholds = [float(value) for value in thresholds]
    if anchor_threshold not in thresholds:
        thresholds = sorted(set([*thresholds, float(anchor_threshold)]))

    seed_everything(seed)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    resource_snapshot = collect_resource_snapshot(device)

    pseudo_bundle = load_texas_pseudo_bundle(assignments_path=assignments_path, candidate_pair_path=candidate_pair_path)
    if test_manifest_path is not None:
        pseudo_bundle = remap_texas_paths_from_manifest(pseudo_bundle, manifest_path=test_manifest_path)
    if trusted_membership_path is not None:
        pseudo_bundle = apply_trusted_membership_to_pseudo_bundle(
            pseudo_bundle,
            trusted_membership_path=trusted_membership_path,
        )
    training_df = build_texas_training_frame(
        pseudo_bundle,
        pseudo_positive_pairs_path=pseudo_positive_pairs_path,
    )
    teacher_source_dirs = [path.resolve() for path in (teacher_source_dirs or [])]
    if teacher_source_dirs:
        teacher_bundle = build_texas_teacher_bundle(
            metadata_df=pseudo_bundle.all_df,
            source_dirs=teacher_source_dirs,
            weights=teacher_weights,
        )
        teacher_anchor_pred_df = apply_thresholds_to_df(
            df=pseudo_bundle.all_df,
            embeddings=teacher_bundle.embeddings,
            threshold_by_dataset={TEXAS_DATASET: float(anchor_threshold)},
        )
        teacher_anchor_labels = teacher_anchor_pred_df["pred_cluster_id"].to_numpy()
        teacher_topk_indices = build_topk_indices(teacher_bundle.embeddings, top_k=top_k)
        teacher_embeddings_for_training = teacher_bundle.embeddings
        teacher_dim = int(teacher_bundle.embeddings.shape[1])
        teacher_component_df = teacher_bundle.component_table.copy()
        teacher_source_dir_refs = [str(path) for path in teacher_bundle.source_dirs]
    else:
        teacher_bundle = None
        teacher_anchor_pred_df = pd.DataFrame()
        teacher_anchor_labels = None
        teacher_topk_indices = None
        teacher_embeddings_for_training = None
        teacher_dim = 0
        teacher_component_df = pd.DataFrame(columns=["teacher_source", "source_dir", "weight", "embedding_dim"])
        teacher_source_dir_refs = []
    teacher_embeddings_for_training = expand_teacher_embeddings_for_training_rows(
        training_df=training_df,
        metadata_df=pseudo_bundle.all_df,
        teacher_embeddings=teacher_embeddings_for_training,
    )

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

    model = TexasSelfTrainModel(
        backbone=backbone,
        feature_dim=backbone_spec.feature_dim,
        embedding_dim=embedding_dim,
        teacher_dim=teacher_dim,
        pseudo_class_count=max(1, len(pseudo_bundle.pseudo_label_map)),
        classification_head=classification_head,
        arcface_scale=arcface_scale,
        arcface_margin=arcface_margin,
    ).to(device)
    init_checkpoint_info: dict[str, object] | None = None
    if student_init_checkpoint is not None:
        init_checkpoint_info = load_texas_selftrain_init_checkpoint(
            model=model,
            checkpoint_path=student_init_checkpoint.resolve(),
        )
    train_transform = build_train_transform(backbone_spec, dataset=TEXAS_DATASET)
    eval_transform = build_eval_transform(backbone_spec, dataset=TEXAS_DATASET)

    train_dataset = TexasSelfTrainDataset(
        df=training_df,
        repo_root=repo_root,
        base_transform=eval_transform,
        positive_transform=train_transform,
        teacher_embeddings=teacher_embeddings_for_training,
    )
    sampler = None
    if seed_oversample_factor > 1.0 and training_df["is_seed"].any():
        sampler = WeightedRandomSampler(
            weights=build_seed_sampling_weights(training_df, seed_oversample_factor=seed_oversample_factor),
            num_samples=len(training_df),
            replacement=True,
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
    optimizer = build_texas_optimizer(
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
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))

    training_rows: list[dict[str, object]] = []
    alignment_rows: list[pd.DataFrame] = []
    best_key: tuple[float, float, float, float, float] | None = None
    best_paths: dict[str, Path] = {}
    best_payload: dict[str, Any] = {}

    for epoch in range(1, epochs + 1):
        epoch_metrics = train_texas_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            pseudo_loss_weight=pseudo_loss_weight,
            relation_distill_weight=relation_distill_weight,
            feature_distill_weight=feature_distill_weight,
            view_pair_weight=view_pair_weight,
            view_pair_temperature=view_pair_temperature,
            label_smoothing=label_smoothing,
            grad_clip_norm=grad_clip_norm,
            max_train_batches=max_train_batches,
        )
        student_embeddings = extract_texas_student_embeddings(
            df=pseudo_bundle.all_df,
            repo_root=repo_root,
            model=model,
            transform=eval_transform,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
        )
        threshold_summary_df, threshold_predictions_df = evaluate_texas_thresholds(
            metadata_df=pseudo_bundle.all_df,
            embeddings=student_embeddings,
            thresholds=thresholds,
            anchor_threshold=anchor_threshold,
            candidate_pair_df=pseudo_bundle.candidate_pair_df,
            teacher_anchor_labels=teacher_anchor_labels,
            teacher_topk_indices=teacher_topk_indices,
            top_k=top_k,
        )
        best_threshold_df = pick_best_texas_threshold(threshold_summary_df)
        best_threshold_row = best_threshold_df.iloc[0]
        if teacher_embeddings_for_training is not None:
            alignment_df = summarize_alignment(
                student_embeddings=student_embeddings,
                teacher_embeddings=teacher_bundle.embeddings,
                metadata_df=pseudo_bundle.all_df[["dataset"]],
            )
            alignment_df["epoch"] = epoch
            alignment_rows.append(alignment_df)
        else:
            alignment_df = pd.DataFrame(columns=["dataset", "samples", "relation_mse", "relation_mae", "relation_corr", "epoch"])

        epoch_row: dict[str, object] = {
            "epoch": epoch,
            **epoch_metrics,
            "best_threshold": float(best_threshold_row["threshold"]),
            "best_proxy_score": compute_threshold_proxy_score(best_threshold_row),
            "best_seed_pair_agreement": float(best_threshold_row["seed_pair_agreement"]),
            "best_mutual_topk_pair_keep_ratio": float(best_threshold_row["mutual_topk_pair_keep_ratio"]),
            "best_seed_recall_at_1": float(best_threshold_row["seed_recall_at_1"]),
            "best_cluster_count": int(best_threshold_row["clusters"]),
            "best_largest_cluster_size": int(best_threshold_row["largest_cluster_size"]),
            "best_cluster_delta_vs_teacher_anchor": float(best_threshold_row["cluster_delta_vs_teacher_anchor"]),
            "best_pair_agreement_vs_teacher_anchor": float(best_threshold_row["pair_agreement_vs_teacher_anchor"]),
            "best_student_teacher_topk_overlap": float(best_threshold_row["student_teacher_topk_overlap"]),
        }
        training_rows.append(epoch_row)

        current_key = build_proxy_selection_tuple(best_threshold_row)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_checkpoint_path = checkpoints_dir / "best_checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "config": {
                        "experiment_id": experiment_id,
                        "student_backbone": student_backbone,
                        "classification_head": classification_head,
                        "embedding_dim": embedding_dim,
                        "anchor_threshold": anchor_threshold,
                        "student_init_checkpoint": str(student_init_checkpoint) if student_init_checkpoint else "",
                        "student_init_info": init_checkpoint_info or {},
                    },
                    "best_threshold": float(best_threshold_row["threshold"]),
                },
                best_checkpoint_path,
            )

            seed_metadata_df = pseudo_bundle.seed_df.copy().reset_index(drop=True)
            if not seed_metadata_df.empty:
                seed_metadata_df["identity"] = seed_metadata_df["pseudo_identity"]
                seed_embeddings = student_embeddings[pseudo_bundle.all_df["is_seed"].to_numpy()]
                np.save(embeddings_dir / "val_embeddings.npy", seed_embeddings.astype(np.float32))
                seed_metadata_df.to_csv(embeddings_dir / "val_metadata.csv", index=False)
            else:
                np.save(embeddings_dir / "val_embeddings.npy", np.empty((0, embedding_dim), dtype=np.float32))
                pseudo_bundle.seed_df.to_csv(embeddings_dir / "val_metadata.csv", index=False)
            np.save(embeddings_dir / "test_embeddings.npy", student_embeddings.astype(np.float32))
            pseudo_bundle.all_df.to_csv(embeddings_dir / "test_metadata.csv", index=False)

            threshold_summary_df.to_csv(tables_dir / "threshold_sweep_v1.csv", index=False)
            threshold_predictions_df.to_csv(tables_dir / "threshold_predictions_v1.csv", index=False)
            best_threshold_df.to_csv(tables_dir / "best_threshold_v1.csv", index=False)
            chosen_threshold = float(best_threshold_row["threshold"])
            chosen_predictions_df = threshold_predictions_df[
                np.isclose(threshold_predictions_df["threshold"].to_numpy(dtype=float), chosen_threshold)
            ].copy()
            chosen_predictions_df.to_csv(tables_dir / "test_predictions_best_v1.csv", index=False)
            if not teacher_anchor_pred_df.empty:
                teacher_anchor_pred_df.to_csv(tables_dir / "teacher_anchor_predictions_v1.csv", index=False)
            pseudo_bundle.all_df.to_csv(tables_dir / "pseudo_assignments_v1.csv", index=False)
            pseudo_bundle.seed_df.to_csv(tables_dir / "pseudo_manifest_v1.csv", index=False)
            training_df.to_csv(tables_dir / "train_rows_v1.csv", index=False)
            pseudo_bundle.candidate_pair_df.to_csv(tables_dir / "candidate_pairs_v1.csv", index=False)
            pseudo_bundle.seed_class_summary_df.to_csv(tables_dir / "seed_class_summary_v1.csv", index=False)
            teacher_component_df.to_csv(tables_dir / "teacher_components_v1.csv", index=False)

            best_paths = {
                "best_checkpoint_path": best_checkpoint_path,
                "best_threshold_path": tables_dir / "best_threshold_v1.csv",
                "test_embeddings_path": embeddings_dir / "test_embeddings.npy",
                "test_metadata_path": embeddings_dir / "test_metadata.csv",
                "chosen_predictions_path": tables_dir / "test_predictions_best_v1.csv",
            }
            best_payload = {
                "threshold_summary_df": threshold_summary_df,
                "best_threshold_df": best_threshold_df,
                "alignment_df": alignment_df.drop(columns=["epoch"]),
            }

    training_log_df = pd.DataFrame(training_rows)
    training_log_path = tables_dir / "training_log_v1.csv"
    training_log_df.to_csv(training_log_path, index=False)
    alignment_history_df = pd.concat(alignment_rows, ignore_index=True) if alignment_rows else pd.DataFrame()
    alignment_history_path = tables_dir / "alignment_history_v1.csv"
    alignment_history_df.to_csv(alignment_history_path, index=False)

    plot_paths = write_texas_selftrain_plots(
        plots_dir=plots_dir,
        training_log_df=training_log_df,
        alignment_history_df=alignment_history_df,
    )
    best_alignment_df = (
        best_payload.get("alignment_df")
        if best_payload
        else pd.DataFrame(columns=["dataset", "samples", "relation_mse", "relation_mae", "relation_corr"])
    )
    config = {
        "experiment_id": experiment_id,
        "goal": goal
        or "Texas-only self-train with trusted seeds plus base-vs-positive contrastive pairs; teacher distillation is optional and disabled by default.",
        "student_backbone": student_backbone,
        "student_model_id": backbone_spec.model_id,
        "student_feature_dim": backbone_spec.feature_dim,
        "teacher_source_dirs": teacher_source_dir_refs,
        "student_init_checkpoint": str(student_init_checkpoint) if student_init_checkpoint else "",
        "student_init_info": init_checkpoint_info or {},
        "trusted_membership_path": str(trusted_membership_path) if trusted_membership_path else "",
        "pseudo_positive_pairs_path": str(pseudo_positive_pairs_path) if pseudo_positive_pairs_path else "",
        "test_manifest_path": str(test_manifest_path) if test_manifest_path else "",
        "teacher_dim": teacher_dim,
        "anchor_threshold": float(anchor_threshold),
        "input_size": backbone_spec.input_size,
        "embedding_dim": embedding_dim,
        "classification_head": classification_head,
        "pseudo_head_shape": f"{max(1, len(pseudo_bundle.pseudo_label_map))} x {embedding_dim}",
        "seed": seed,
        "seed_oversample_factor": seed_oversample_factor,
        "view_pair_weight": view_pair_weight,
        "view_pair_temperature": view_pair_temperature,
        "device": device,
        "epochs": epochs,
        "train_rows_after_view_expand": int(len(training_df)),
        "train_batch_size": train_batch_size,
        "eval_batch_size": eval_batch_size,
        "effective_batch_size": effective_batch_size,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "resolved_backbone_lr": resolved_backbone_lr,
        "resolved_head_lr": resolved_head_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "grad_clip_norm": grad_clip_norm,
        "pseudo_loss_weight": pseudo_loss_weight,
        "relation_distill_weight": relation_distill_weight,
        "feature_distill_weight": feature_distill_weight,
        "total_images": int(len(pseudo_bundle.all_df)),
        "seed_images": int(len(pseudo_bundle.seed_df)),
        "seed_coverage_ratio": round(float(len(pseudo_bundle.seed_df) / max(len(pseudo_bundle.all_df), 1)), 6),
        "seed_clusters": int(len(pseudo_bundle.pseudo_label_map)),
        "uncertain_images": int((~pseudo_bundle.all_df["is_seed"]).sum()),
        "trusted_images_after_override": int(pseudo_bundle.all_df["is_seed"].sum()),
        "untrusted_images_after_override": int((~pseudo_bundle.all_df["is_seed"]).sum()),
        "candidate_pairs": int(len(pseudo_bundle.candidate_pair_df)),
        "mutual_topk_pairs": int(
            pseudo_bundle.candidate_pair_df["mutual_topk_all_routes"].sum()
        )
        if not pseudo_bundle.candidate_pair_df.empty and "mutual_topk_all_routes" in pseudo_bundle.candidate_pair_df.columns
        else 0,
        "train_augmentation": describe_transform(train_transform),
        "eval_preprocess": describe_transform(eval_transform),
        "amp_enabled": device.startswith("cuda"),
        "resource_snapshot": resource_snapshot,
        "resource_decision": resource_decision
        or (
            "Texas self-train uses one student backbone on a single GPU with optional frozen teacher caches."
            if teacher_source_dirs
            else "Texas self-train runs teacher-free on a single GPU."
        ),
        "probe_reuse_note": probe_reuse_note or "This is a new Texas-only recipe; batch sizing should not be assumed comparable to labeled supervised runs.",
        "next_step_judgment": "Use this checkpoint only if a Texas-only submission variant beats the frozen `miew@0.38` public baseline `0.37277`.",
        "trusted_view_rows": int(training_df["is_seed"].sum()),
        "untrusted_view_rows": int((~training_df["is_seed"]).sum()),
        "trusted_view_images": int(training_df.loc[training_df["is_seed"], "image_id"].astype(str).nunique()),
        "untrusted_view_images": int(training_df.loc[~training_df["is_seed"], "image_id"].astype(str).nunique()),
        "view_row_sources": training_df["pair_source"].astype(str).value_counts().to_dict(),
    }
    summary_path = reports_dir / "summary.md"
    write_texas_selftrain_report(
        summary_path,
        config=config,
        training_log_df=training_log_df,
        threshold_summary_df=best_payload.get("threshold_summary_df", pd.DataFrame()),
        best_threshold_df=best_payload.get("best_threshold_df", pd.DataFrame()),
        seed_class_summary_df=pseudo_bundle.seed_class_summary_df,
        teacher_component_df=teacher_component_df,
        alignment_df=best_alignment_df,
        plot_paths=plot_paths,
    )
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "best_epoch": int(
                    training_log_df.sort_values(
                        [
                            "best_proxy_score",
                            "best_seed_pair_agreement",
                            "best_mutual_topk_pair_keep_ratio",
                            "best_seed_recall_at_1",
                            "epoch",
                        ],
                        ascending=[False, False, False, False, True],
                    ).iloc[0]["epoch"]
                ),
                "best_threshold": float(
                    training_log_df.sort_values(
                        [
                            "best_proxy_score",
                            "best_seed_pair_agreement",
                            "best_mutual_topk_pair_keep_ratio",
                            "best_seed_recall_at_1",
                            "epoch",
                        ],
                        ascending=[False, False, False, False, True],
                    ).iloc[0]["best_threshold"]
                ),
                "student_backbone": student_backbone,
                "teacher_source_dirs": teacher_source_dir_refs,
                "anchor_threshold": float(anchor_threshold),
                "best_checkpoint_path": str(best_paths.get("best_checkpoint_path", "")),
                "chosen_predictions_path": str(best_paths.get("chosen_predictions_path", "")),
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
        "alignment_history_path": alignment_history_path,
        **best_paths,
    }
