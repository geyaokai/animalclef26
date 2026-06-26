from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN, build_average_linkage, cluster_from_linkage, load_cached_embedding_bundle
from .orb_rerank_baseline import recall_at_k_from_score_matrix
from .pseudo_seed_features import append_pseudo_seed_pair_features
from .texas_black_patch_local import build_patch_pair_features
from .texas_black_pattern_orb_local import merge_texas_black_pattern_orb_local_scores
from .texas_selftrain import (
    build_candidate_index_pairs,
    compute_pair_keep_ratio,
    compute_threshold_proxy_score,
    pick_best_texas_threshold,
)
from .texas_unsupervised import TEXAS_DATASET, build_rank_matrix, build_topk_indices, mean_topk_neighbor_overlap, pair_agreement_score, summarize_cluster_labels


DEFAULT_TEXAS_EXPERIMENT_DIR = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1")
DEFAULT_TEXAS_TEACHER_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_miew_v1")
DEFAULT_TEXAS_BLACK_LOCAL_CSV = Path("artifacts/analysis/texas_black_pattern_orb_local_v1/tables/test_pair_local_scores_v1.csv")
DEFAULT_TEXAS_BLACK_IMAGE_FEATURE_CSV = Path("artifacts/analysis/texas_black_pattern_orb_local_v1/tables/image_feature_stats_v1.csv")
DEFAULT_TEXAS_THRESHOLDS = [0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
DEFAULT_TOP_K = 8

TEXAS_XGB_FEATURE_COLUMNS = [
    "route_global_score",
    "route_rank_forward",
    "route_rank_backward",
    "route_rank_pct_forward",
    "route_rank_pct_backward",
    "route_rank_pct_mean",
    "route_mutual_topk",
    "miew_similarity",
    "fusion_similarity",
    "miew_rank_forward",
    "miew_rank_backward",
    "fusion_rank_forward",
    "fusion_rank_backward",
    "miew_mutual_topk",
    "fusion_mutual_topk",
    "mutual_topk_all_routes",
    "same_cluster_all_routes",
    "same_teacher_anchor",
    "left_is_seeded",
    "right_is_seeded",
    "both_seeded",
    "one_seeded",
    "both_unseeded",
    "same_seed_cluster",
    "left_seed_cluster_size",
    "right_seed_cluster_size",
    "left_seed_mean_similarity",
    "right_seed_mean_similarity",
    "black_orb_local_score",
    "black_orb_local_raw_score",
    "black_orb_inliers",
    "black_orb_good_matches",
    "black_orb_left_keypoints",
    "black_orb_right_keypoints",
    "black_orb_keypoint_min",
    "black_orb_keypoint_max",
    "black_orb_inlier_ratio",
    "black_orb_match_density",
    "left_black_ratio",
    "right_black_ratio",
    "black_ratio_absdiff",
    "black_orb_support_flag",
    "black_orb_veto_flag",
    "black_patch_pair_valid_v1",
    "black_patch_overlap_pixels_v1",
    "black_patch_gray_corr_v1",
    "black_patch_gray_absdiff_v1",
    "black_patch_mask_iou_v1",
    "black_patch_mask_dice_v1",
    "black_patch_col_profile_corr_v1",
    "black_patch_col_profile_l1_v1",
    "black_patch_row_profile_corr_v1",
    "black_patch_row_profile_l1_v1",
    "black_patch_support_score_v1",
    "black_patch_veto_score_v1",
    "black_patch_support_flag_v1",
    "black_patch_veto_flag_v1",
]


@dataclass(frozen=True)
class TexasLocalArtifacts:
    metadata_df: pd.DataFrame
    route_df: pd.DataFrame
    pseudo_df: pd.DataFrame
    candidate_pair_df: pd.DataFrame
    teacher_anchor_labels: np.ndarray
    teacher_topk_indices: np.ndarray
    route_embeddings: np.ndarray
    route_score_matrix: np.ndarray
    route_topk_indices: np.ndarray
    route_rank_matrix: np.ndarray


def _resolve_input_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def _load_texas_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["image_id"] = frame["image_id"].astype(str)
    subset = frame[frame["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {path}")
    if PATH_COLUMN not in subset.columns and "path" in subset.columns:
        subset[PATH_COLUMN] = subset["path"].astype(str)
    return subset


def _reorder_frame_to_reference(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, name: str) -> pd.DataFrame:
    lookup = candidate_df.copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup = lookup.set_index("image_id", drop=False)
    image_ids = reference_df["image_id"].astype(str).tolist()
    missing = [image_id for image_id in image_ids if image_id not in lookup.index]
    if missing:
        raise ValueError(f"{name} missing Texas image_ids, examples: {missing[:5]}")
    return lookup.loc[image_ids].reset_index(drop=True).copy()


def _load_aligned_teacher_topk_indices(
    *,
    teacher_source_dir: Path,
    reference_df: pd.DataFrame,
    top_k: int,
) -> np.ndarray:
    bundle = load_cached_embedding_bundle(source_dir=teacher_source_dir.resolve(), name=teacher_source_dir.name)
    texas_df = bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    texas_embeddings = bundle.test_embeddings[(bundle.test_df["dataset"] == TEXAS_DATASET).to_numpy()]
    texas_df = _reorder_frame_to_reference(reference_df, texas_df, teacher_source_dir.name)
    image_to_index = {str(image_id): index for index, image_id in enumerate(bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET]["image_id"].astype(str).tolist())}
    reorder_index = np.asarray([image_to_index[str(image_id)] for image_id in texas_df["image_id"].astype(str).tolist()], dtype=np.int32)
    aligned_embeddings = texas_embeddings[reorder_index]
    return build_topk_indices(aligned_embeddings, top_k=top_k)


def build_similarity_score_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings.astype(np.float32, copy=False)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32, copy=False)


def build_topk_indices_from_score_matrix(score_matrix: np.ndarray, top_k: int) -> np.ndarray:
    sample_count = int(score_matrix.shape[0])
    width = min(int(top_k), sample_count - 1)
    if sample_count == 0 or width <= 0:
        return np.zeros((sample_count, 0), dtype=np.int32)
    masked = np.asarray(score_matrix, dtype=np.float32).copy()
    np.fill_diagonal(masked, -np.inf)
    topk = np.argpartition(-masked, kth=width - 1, axis=1)[:, :width]
    ordered = np.take_along_axis(
        topk,
        np.argsort(-np.take_along_axis(masked, topk, axis=1), axis=1),
        axis=1,
    )
    return ordered.astype(np.int32, copy=False)


def load_texas_local_artifacts(
    *,
    repo_root: Path,
    experiment_dir: Path = DEFAULT_TEXAS_EXPERIMENT_DIR,
    teacher_source_dir: Path = DEFAULT_TEXAS_TEACHER_SOURCE_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> TexasLocalArtifacts:
    resolved_experiment_dir = _resolve_input_path(repo_root, experiment_dir)
    resolved_teacher_source_dir = _resolve_input_path(repo_root, teacher_source_dir)

    route_df = _load_texas_frame(resolved_experiment_dir / "tables" / "test_predictions_best_v1.csv")
    pseudo_df = _reorder_frame_to_reference(
        route_df,
        pd.read_csv(resolved_experiment_dir / "tables" / "pseudo_assignments_v1.csv"),
        "pseudo_assignments",
    )
    pseudo_df["image_id"] = pseudo_df["image_id"].astype(str)
    pseudo_df["dataset"] = pseudo_df["dataset"].astype(str)
    if PATH_COLUMN not in pseudo_df.columns and "path" in pseudo_df.columns:
        pseudo_df[PATH_COLUMN] = pseudo_df["path"].astype(str)
    pseudo_df["is_seed"] = pseudo_df.get("is_seed", False)
    pseudo_df["is_seed"] = pseudo_df["is_seed"].fillna(False).astype(bool)
    pseudo_df["pseudo_label_index"] = pseudo_df.get("pseudo_label_index", -1)
    pseudo_df["pseudo_label_index"] = pseudo_df["pseudo_label_index"].fillna(-1).astype(int)

    candidate_pair_df = pd.read_csv(resolved_experiment_dir / "tables" / "candidate_pairs_v1.csv").copy()
    candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
    candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)
    teacher_anchor_df = _reorder_frame_to_reference(
        route_df,
        _load_texas_frame(resolved_experiment_dir / "tables" / "teacher_anchor_predictions_v1.csv"),
        "teacher_anchor_predictions",
    )
    teacher_anchor_labels = teacher_anchor_df["pred_cluster_id"].to_numpy(dtype=np.int32)

    route_embeddings = np.load(resolved_experiment_dir / "embeddings" / "test_embeddings.npy").astype(np.float32)
    if len(route_embeddings) != len(route_df):
        raise ValueError(f"Texas route embedding mismatch: embeddings={len(route_embeddings)} vs rows={len(route_df)}")
    route_score_matrix = build_similarity_score_matrix(route_embeddings)
    route_topk_indices = build_topk_indices(route_embeddings, top_k=top_k)
    route_rank_matrix = build_rank_matrix(route_topk_indices)
    teacher_topk_indices = _load_aligned_teacher_topk_indices(
        teacher_source_dir=resolved_teacher_source_dir,
        reference_df=route_df,
        top_k=top_k,
    )
    return TexasLocalArtifacts(
        metadata_df=pseudo_df.copy().reset_index(drop=True),
        route_df=route_df.copy().reset_index(drop=True),
        pseudo_df=pseudo_df.copy().reset_index(drop=True),
        candidate_pair_df=candidate_pair_df.reset_index(drop=True),
        teacher_anchor_labels=teacher_anchor_labels,
        teacher_topk_indices=teacher_topk_indices,
        route_embeddings=route_embeddings,
        route_score_matrix=route_score_matrix,
        route_topk_indices=route_topk_indices,
        route_rank_matrix=route_rank_matrix,
    )


def merge_texas_black_pattern_scores(
    *,
    repo_root: Path,
    pair_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    local_match_csv: Path = DEFAULT_TEXAS_BLACK_LOCAL_CSV,
) -> pd.DataFrame:
    resolved_local_match_csv = _resolve_input_path(repo_root, local_match_csv)
    frame = pair_df.copy().reset_index(drop=True)
    image_to_index = {str(image_id): index for index, image_id in enumerate(metadata_df["image_id"].astype(str).tolist())}
    if "left_index" not in frame.columns:
        frame["left_index"] = frame["image_id"].astype(str).map(image_to_index).astype(int)
    if "right_index" not in frame.columns:
        frame["right_index"] = frame["neighbor_image_id"].astype(str).map(image_to_index).astype(int)
    local_match_df = pd.read_csv(resolved_local_match_csv)
    return merge_texas_black_pattern_orb_local_scores(
        pair_df=frame,
        local_match_df=local_match_df,
        override_local_score=False,
    )


def merge_texas_black_patch_scores(
    *,
    repo_root: Path,
    pair_df: pd.DataFrame,
    image_feature_csv: Path = DEFAULT_TEXAS_BLACK_IMAGE_FEATURE_CSV,
) -> pd.DataFrame:
    resolved_image_feature_csv = _resolve_input_path(repo_root, image_feature_csv)
    image_feature_df = pd.read_csv(resolved_image_feature_csv)
    patch_pair_df = build_patch_pair_features(
        pair_df=pair_df,
        image_feature_df=image_feature_df,
        repo_root=repo_root,
        image_path_column="texas_black_pattern_aligned_rgb_path_v1",
        mask_path_column="texas_black_pattern_mask_path_v1",
    )
    return pair_df.merge(
        patch_pair_df,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )


def enrich_texas_pair_df(
    *,
    pair_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    route_score_matrix: np.ndarray,
    route_rank_matrix: np.ndarray,
    teacher_anchor_labels: np.ndarray,
    support_score_floor: float,
    support_inlier_floor: int,
    veto_score_ceiling: float,
    veto_inlier_ceiling: int,
    patch_support_gray_corr_floor: float,
    patch_support_absdiff_ceiling: float,
    patch_support_iou_floor: float,
    patch_veto_gray_corr_ceiling: float,
    patch_veto_absdiff_floor: float,
    patch_veto_iou_ceiling: float,
) -> pd.DataFrame:
    frame = pair_df.copy().reset_index(drop=True)
    image_to_index = {str(image_id): index for index, image_id in enumerate(metadata_df["image_id"].astype(str).tolist())}
    if "left_index" not in frame.columns:
        frame["left_index"] = frame["image_id"].map(image_to_index).astype(int)
    if "right_index" not in frame.columns:
        frame["right_index"] = frame["neighbor_image_id"].map(image_to_index).astype(int)
    left_index = frame["left_index"].to_numpy(dtype=np.int32)
    right_index = frame["right_index"].to_numpy(dtype=np.int32)

    route_global = route_score_matrix[left_index, right_index].astype(np.float32, copy=False)
    route_rank_forward = route_rank_matrix[left_index, right_index].astype(np.int32, copy=False)
    route_rank_backward = route_rank_matrix[right_index, left_index].astype(np.int32, copy=False)
    max_rank = max(int(route_rank_matrix.shape[1] - 1), 1)

    frame["route_global_score"] = np.round(route_global, 6)
    frame["route_rank_forward"] = route_rank_forward
    frame["route_rank_backward"] = route_rank_backward
    frame["route_rank_pct_forward"] = np.round(route_rank_forward.astype(np.float32) / float(max_rank), 6)
    frame["route_rank_pct_backward"] = np.round(route_rank_backward.astype(np.float32) / float(max_rank), 6)
    frame["route_rank_pct_mean"] = np.round(
        0.5 * (frame["route_rank_pct_forward"].to_numpy(dtype=np.float32) + frame["route_rank_pct_backward"].to_numpy(dtype=np.float32)),
        6,
    )
    frame["route_mutual_topk"] = ((route_rank_forward <= DEFAULT_TOP_K) & (route_rank_backward <= DEFAULT_TOP_K)).astype(int)
    frame["same_teacher_anchor"] = (teacher_anchor_labels[left_index] == teacher_anchor_labels[right_index]).astype(int)

    for column in [
        "same_cluster_all_routes",
        "miew_mutual_topk",
        "fusion_mutual_topk",
        "mutual_topk_all_routes",
    ]:
        if column in frame.columns:
            frame[column] = frame[column].fillna(False).astype(bool).astype(int)

    for column in [
        "miew_similarity",
        "fusion_similarity",
        "left_black_ratio",
        "right_black_ratio",
        "black_orb_local_score",
        "black_orb_local_raw_score",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    for column in [
        "miew_rank_forward",
        "miew_rank_backward",
        "fusion_rank_forward",
        "fusion_rank_backward",
        "black_orb_inliers",
        "black_orb_good_matches",
        "black_orb_left_keypoints",
        "black_orb_right_keypoints",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)

    left_black_ratio = (
        pd.to_numeric(frame["left_black_ratio"], errors="coerce").fillna(0.0)
        if "left_black_ratio" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=np.float32)
    )
    right_black_ratio = (
        pd.to_numeric(frame["right_black_ratio"], errors="coerce").fillna(0.0)
        if "right_black_ratio" in frame.columns
        else pd.Series(0.0, index=frame.index, dtype=np.float32)
    )
    frame["black_ratio_absdiff"] = np.round(np.abs(left_black_ratio.to_numpy(dtype=np.float32) - right_black_ratio.to_numpy(dtype=np.float32)), 6)
    left_keypoints = frame.get("black_orb_left_keypoints", pd.Series(0, index=frame.index)).astype(int).to_numpy(dtype=np.int32)
    right_keypoints = frame.get("black_orb_right_keypoints", pd.Series(0, index=frame.index)).astype(int).to_numpy(dtype=np.int32)
    good_matches = frame.get("black_orb_good_matches", pd.Series(0, index=frame.index)).astype(int).to_numpy(dtype=np.int32)
    inliers = frame.get("black_orb_inliers", pd.Series(0, index=frame.index)).astype(int).to_numpy(dtype=np.int32)
    keypoint_min = np.maximum(1, np.minimum(left_keypoints, right_keypoints))
    keypoint_max = np.maximum(left_keypoints, right_keypoints)
    frame["black_orb_keypoint_min"] = keypoint_min.astype(int)
    frame["black_orb_keypoint_max"] = keypoint_max.astype(int)
    frame["black_orb_inlier_ratio"] = np.round(inliers.astype(np.float32) / keypoint_min.astype(np.float32), 6)
    frame["black_orb_match_density"] = np.round(good_matches.astype(np.float32) / keypoint_min.astype(np.float32), 6)

    local_score = frame.get("black_orb_local_score", pd.Series(0.0, index=frame.index)).astype(float).to_numpy(dtype=np.float32)
    frame["black_orb_support_flag"] = (
        (local_score >= float(support_score_floor))
        & (inliers >= int(support_inlier_floor))
    ).astype(int)
    frame["black_orb_veto_flag"] = (
        (local_score <= float(veto_score_ceiling))
        & (inliers <= int(veto_inlier_ceiling))
    ).astype(int)

    patch_gray_corr = (
        pd.to_numeric(frame["black_patch_gray_corr_v1"], errors="coerce").fillna(0.0)
        if "black_patch_gray_corr_v1" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    patch_gray_absdiff = (
        pd.to_numeric(frame["black_patch_gray_absdiff_v1"], errors="coerce").fillna(1.0)
        if "black_patch_gray_absdiff_v1" in frame.columns
        else pd.Series(1.0, index=frame.index)
    )
    patch_mask_iou = (
        pd.to_numeric(frame["black_patch_mask_iou_v1"], errors="coerce").fillna(0.0)
        if "black_patch_mask_iou_v1" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    patch_col_profile_corr = (
        pd.to_numeric(frame["black_patch_col_profile_corr_v1"], errors="coerce").fillna(0.0)
        if "black_patch_col_profile_corr_v1" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    patch_row_profile_corr = (
        pd.to_numeric(frame["black_patch_row_profile_corr_v1"], errors="coerce").fillna(0.0)
        if "black_patch_row_profile_corr_v1" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    patch_valid = (
        frame["black_patch_pair_valid_v1"].fillna(False).astype(bool)
        if "black_patch_pair_valid_v1" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    patch_support_score = (
        0.35 * patch_gray_corr.to_numpy(dtype=np.float32)
        + 0.25 * patch_mask_iou.to_numpy(dtype=np.float32)
        + 0.20 * np.clip(1.0 - patch_gray_absdiff.to_numpy(dtype=np.float32), 0.0, 1.0)
        + 0.10 * np.clip((patch_col_profile_corr.to_numpy(dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)
        + 0.10 * np.clip((patch_row_profile_corr.to_numpy(dtype=np.float32) + 1.0) * 0.5, 0.0, 1.0)
    )
    patch_veto_score = (
        0.40 * np.clip(1.0 - np.maximum(patch_gray_corr.to_numpy(dtype=np.float32), 0.0), 0.0, 1.0)
        + 0.30 * np.clip(patch_gray_absdiff.to_numpy(dtype=np.float32), 0.0, 1.0)
        + 0.30 * np.clip(1.0 - patch_mask_iou.to_numpy(dtype=np.float32), 0.0, 1.0)
    )
    frame["black_patch_support_score_v1"] = np.round(patch_support_score, 6)
    frame["black_patch_veto_score_v1"] = np.round(patch_veto_score, 6)
    frame["black_patch_support_flag_v1"] = (
        patch_valid.to_numpy(dtype=bool)
        & (patch_gray_corr.to_numpy(dtype=np.float32) >= float(patch_support_gray_corr_floor))
        & (patch_gray_absdiff.to_numpy(dtype=np.float32) <= float(patch_support_absdiff_ceiling))
        & (patch_mask_iou.to_numpy(dtype=np.float32) >= float(patch_support_iou_floor))
    ).astype(int)
    frame["black_patch_veto_flag_v1"] = (
        patch_valid.to_numpy(dtype=bool)
        & (patch_gray_corr.to_numpy(dtype=np.float32) <= float(patch_veto_gray_corr_ceiling))
        & (patch_gray_absdiff.to_numpy(dtype=np.float32) >= float(patch_veto_absdiff_floor))
        & (patch_mask_iou.to_numpy(dtype=np.float32) <= float(patch_veto_iou_ceiling))
    ).astype(int)

    pseudo_assignment_df = metadata_df[["image_id", "is_seed", "pseudo_label_index", "pseudo_identity", "pseudo_image_count"]].copy()
    pseudo_assignment_df["is_seeded"] = pseudo_assignment_df["is_seed"].fillna(False).astype(int)
    pseudo_assignment_df["seed_cluster_id"] = np.where(
        pseudo_assignment_df["is_seeded"].eq(1),
        pseudo_assignment_df["pseudo_identity"].fillna("").astype(str),
        "",
    )
    pseudo_assignment_df["seed_cluster_size"] = np.where(
        pseudo_assignment_df["is_seeded"].eq(1),
        pseudo_assignment_df["pseudo_image_count"].fillna(0).astype(int),
        0,
    )
    pseudo_assignment_df["seed_mean_similarity"] = 0.0
    frame = append_pseudo_seed_pair_features(frame, pseudo_assignment_df)
    return frame.reset_index(drop=True)


def apply_texas_local_rerank(
    *,
    global_score_matrix: np.ndarray,
    pair_df: pd.DataFrame,
    support_weight: float,
    veto_weight: float,
) -> np.ndarray:
    reranked = np.asarray(global_score_matrix, dtype=np.float32).copy()
    if pair_df.empty:
        return reranked
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        base_score = float(reranked[left_index, right_index])
        orb_support_strength = float(getattr(row, "black_orb_local_score", 0.0)) if int(getattr(row, "black_orb_support_flag", 0)) == 1 else 0.0
        patch_support_strength = float(getattr(row, "black_patch_support_score_v1", 0.0)) if int(getattr(row, "black_patch_support_flag_v1", 0)) == 1 else 0.0
        support_strength = max(orb_support_strength, patch_support_strength)
        orb_veto_strength = float(1.0 - float(getattr(row, "black_orb_local_score", 0.0))) if int(getattr(row, "black_orb_veto_flag", 0)) == 1 else 0.0
        patch_veto_strength = float(getattr(row, "black_patch_veto_score_v1", 0.0)) if int(getattr(row, "black_patch_veto_flag_v1", 0)) == 1 else 0.0
        veto_strength = max(orb_veto_strength, patch_veto_strength)
        fused_score = float(np.clip(base_score + float(support_weight) * support_strength - float(veto_weight) * veto_strength, -1.0, 1.0))
        reranked[left_index, right_index] = fused_score
        reranked[right_index, left_index] = fused_score
    np.fill_diagonal(reranked, 1.0)
    return reranked


def apply_pair_probability_residual(
    *,
    base_score_matrix: np.ndarray,
    pair_df: pd.DataFrame,
    probability_col: str,
    blend_scale: float,
) -> np.ndarray:
    fused = np.asarray(base_score_matrix, dtype=np.float32).copy()
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        probability = float(getattr(row, probability_col))
        base_value = float(fused[left_index, right_index])
        score = float(np.clip(base_value + float(blend_scale) * (probability - 0.5), -1.0, 1.0))
        fused[left_index, right_index] = score
        fused[right_index, left_index] = score
    np.fill_diagonal(fused, 1.0)
    return fused


def evaluate_texas_thresholds_from_score_matrix(
    *,
    metadata_df: pd.DataFrame,
    score_matrix: np.ndarray,
    thresholds: list[float],
    candidate_pair_df: pd.DataFrame,
    teacher_anchor_labels: np.ndarray | None = None,
    teacher_topk_indices: np.ndarray | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_matrix = np.asarray(score_matrix, dtype=np.float32)
    seed_mask = metadata_df["is_seed"].fillna(False).astype(bool).to_numpy()
    seed_labels = metadata_df.loc[seed_mask, "pseudo_label_index"].to_numpy(dtype=int) if seed_mask.any() else np.asarray([], dtype=int)
    all_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=False)
    mutual_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=True)
    student_topk_indices = build_topk_indices_from_score_matrix(score_matrix, top_k=top_k)
    student_teacher_topk_overlap = (
        mean_topk_neighbor_overlap(student_topk_indices, teacher_topk_indices)
        if teacher_topk_indices is not None and student_topk_indices.shape == teacher_topk_indices.shape
        else np.nan
    )
    distance = 1.0 - np.clip(score_matrix, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = build_average_linkage(distance)
    target_clusters = int(summarize_cluster_labels(teacher_anchor_labels)["clusters"]) if teacher_anchor_labels is not None else np.nan

    predictions: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    anchor_labels: np.ndarray | None = None
    for threshold in [float(value) for value in thresholds]:
        pred_labels = cluster_from_linkage(linkage_matrix, len(metadata_df), threshold)
        frame = metadata_df.copy()
        frame["threshold"] = float(threshold)
        frame["pred_cluster_id"] = pred_labels
        frame["cluster_label"] = [f"cluster_{TEXAS_DATASET}_{int(label)}" for label in pred_labels]
        predictions.append(frame)
        stats = summarize_cluster_labels(pred_labels)
        row: dict[str, Any] = {
            "threshold": float(threshold),
            "samples": int(len(frame)),
            **stats,
            "seed_pair_agreement": pair_agreement_score(pred_labels[seed_mask], seed_labels) if seed_mask.any() else np.nan,
            "seed_recall_at_1": recall_at_k_from_score_matrix(score_matrix[np.ix_(seed_mask, seed_mask)], seed_labels, k=1) if seed_mask.any() else np.nan,
            "candidate_pair_keep_ratio": compute_pair_keep_ratio(pred_labels, all_candidate_pairs),
            "mutual_topk_pair_keep_ratio": compute_pair_keep_ratio(pred_labels, mutual_candidate_pairs),
            "student_teacher_topk_overlap": student_teacher_topk_overlap,
            "teacher_anchor_clusters": target_clusters,
            "cluster_delta_vs_teacher_anchor": abs(int(stats["clusters"]) - int(target_clusters)) if teacher_anchor_labels is not None else np.nan,
            "pair_agreement_vs_teacher_anchor": pair_agreement_score(pred_labels, teacher_anchor_labels) if teacher_anchor_labels is not None else np.nan,
        }
        summary_rows.append(row)
        if abs(float(threshold) - 0.38) < 1e-9:
            anchor_labels = pred_labels.copy()
    if anchor_labels is None and summary_rows:
        anchor_threshold = float(thresholds[0])
        anchor_labels = predictions[0]["pred_cluster_id"].to_numpy(dtype=np.int32)
        for row in summary_rows:
            row["student_anchor_threshold"] = anchor_threshold
    for row, pred_df in zip(summary_rows, predictions, strict=True):
        row["pair_agreement_vs_student_anchor"] = pair_agreement_score(pred_df["pred_cluster_id"].to_numpy(dtype=np.int32), anchor_labels)
        row["proxy_score"] = compute_threshold_proxy_score(pd.Series(row))
    summary_df = pd.DataFrame(summary_rows).sort_values("threshold").reset_index(drop=True)
    prediction_df = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return summary_df, prediction_df


def pick_best_proxy_row(summary_df: pd.DataFrame) -> pd.Series:
    return pick_best_texas_threshold(summary_df).iloc[0]
