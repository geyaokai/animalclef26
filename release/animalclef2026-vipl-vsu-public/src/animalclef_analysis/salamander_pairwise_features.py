from __future__ import annotations

import numpy as np
import pandas as pd


BASE_PAIR_FEATURE_COLUMNS = [
    "route_global_score",
    "fusion_global_score",
    "student_global_score",
    "distill_global_score",
    "route_minus_fusion",
    "route_minus_student",
    "student_minus_distill",
    "local_score",
    "local_raw_score",
    "inliers",
    "good_matches",
    "left_keypoints",
    "right_keypoints",
    "keypoint_min",
    "keypoint_max",
    "inlier_ratio",
    "match_density",
]

ORIENTATION_PAIR_BUCKETS = [
    "left__left",
    "left__right",
    "left__top",
    "right__right",
    "right__top",
    "top__top",
    "other_known",
    "unknown",
]

METADATA_PAIR_FEATURE_COLUMNS = [
    "orientation_known_pair",
    "same_orientation",
    "orientation_pair_is_left__left",
    "orientation_pair_is_left__right",
    "orientation_pair_is_left__top",
    "orientation_pair_is_right__right",
    "orientation_pair_is_right__top",
    "orientation_pair_is_top__top",
    "orientation_pair_is_other_known",
    "orientation_pair_is_unknown",
    "capture_date_known_pair",
    "same_capture_date",
    "capture_day_gap",
    "capture_day_gap_log1p",
    "capture_day_gap_le_7",
    "capture_day_gap_le_30",
    "capture_day_gap_le_180",
]

GRAPH_PAIR_FEATURE_COLUMNS = [
    "route_rank_pct_left_to_right",
    "route_rank_pct_right_to_left",
    "route_rank_pct_mean",
    "route_rank_pct_max",
    "route_rank_gap_pct",
    "route_reciprocal_rank_mean",
    "route_mutual_topk",
    "route_shared_neighbor_count",
    "route_shared_neighbor_ratio",
    "route_shared_neighbor_mean_score",
    "route_shared_neighbor_best_score",
    "route_shared_neighbor_support_score",
]

DUAL_VIEW_PAIR_FEATURE_COLUMNS = [
    "masked_student_global_score",
    "masked_distill_global_score",
    "student_masked_score_mean",
    "student_masked_score_gap",
    "student_cross_score_mean",
    "student_cross_score_max",
    "distill_masked_score_mean",
    "distill_masked_score_gap",
    "distill_cross_score_mean",
    "distill_cross_score_max",
    "dual_view_global_score_mean",
    "dual_view_cross_score_mean",
]

YELLOW_PAIR_FEATURE_COLUMNS = [
    "left_yellow_quality_flag_v1",
    "right_yellow_quality_flag_v1",
    "left_yellow_focus_available_v1",
    "right_yellow_focus_available_v1",
    "yellow_focus_pair_valid_v1",
    "yellow_orb_pair_valid_v1",
    "yellow_roi_left_keypoints",
    "yellow_roi_right_keypoints",
    "yellow_roi_good_matches",
    "yellow_roi_inliers",
    "yellow_roi_local_raw_score",
    "yellow_roi_local_score",
    "yellow_roi_keypoint_min",
    "yellow_patch_pair_valid_v1",
    "yellow_patch_overlap_pixels_v1",
    "yellow_patch_gray_corr_v1",
    "yellow_patch_gray_absdiff_v1",
    "yellow_patch_mask_iou_v1",
    "yellow_patch_mask_dice_v1",
    "yellow_patch_profile_corr_v1",
    "yellow_patch_profile_l1_v1",
    "yellow_orb_support_v1",
    "yellow_patch_support_v1",
    "yellow_pair_support_v1",
    "yellow_orb_fail_v1",
    "yellow_patch_hard_fail_v1",
    "yellow_patch_extreme_fail_v1",
    "yellow_patch_soft_fail_v1",
    "yellow_hard_veto_v1",
    "yellow_soft_veto_v1",
    "yellow_veto_applied_v1",
]

FEATURE_SET_BASIC = "basic"
FEATURE_SET_DUAL_VIEW_V1 = "dual_view_v1"
FEATURE_SET_META_GRAPH_V1 = "meta_graph_v1"
FEATURE_SET_META_GRAPH_DUAL_VIEW_V1 = "meta_graph_dual_view_v1"
FEATURE_SET_YELLOW_V1 = "yellow_v1"
FEATURE_SET_META_GRAPH_YELLOW_V1 = "meta_graph_yellow_v1"
VALID_FEATURE_SETS = [
    FEATURE_SET_BASIC,
    FEATURE_SET_DUAL_VIEW_V1,
    FEATURE_SET_META_GRAPH_V1,
    FEATURE_SET_META_GRAPH_DUAL_VIEW_V1,
    FEATURE_SET_YELLOW_V1,
    FEATURE_SET_META_GRAPH_YELLOW_V1,
]


def normalize_orientation_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "missing", "unknown"}:
        return ""
    return text


def canonical_orientation_pair(left_orientation: object, right_orientation: object) -> str:
    left_value = normalize_orientation_value(left_orientation)
    right_value = normalize_orientation_value(right_orientation)
    if not left_value or not right_value:
        return "unknown"
    pair_key = "__".join(sorted([left_value, right_value]))
    if pair_key in ORIENTATION_PAIR_BUCKETS:
        return pair_key
    return "other_known"


def resolve_pair_feature_columns(feature_set: str) -> list[str]:
    if feature_set == FEATURE_SET_BASIC:
        return list(BASE_PAIR_FEATURE_COLUMNS)
    if feature_set == FEATURE_SET_DUAL_VIEW_V1:
        return list(BASE_PAIR_FEATURE_COLUMNS) + DUAL_VIEW_PAIR_FEATURE_COLUMNS
    if feature_set == FEATURE_SET_META_GRAPH_V1:
        return list(BASE_PAIR_FEATURE_COLUMNS) + METADATA_PAIR_FEATURE_COLUMNS + GRAPH_PAIR_FEATURE_COLUMNS
    if feature_set == FEATURE_SET_META_GRAPH_DUAL_VIEW_V1:
        return (
            list(BASE_PAIR_FEATURE_COLUMNS)
            + DUAL_VIEW_PAIR_FEATURE_COLUMNS
            + METADATA_PAIR_FEATURE_COLUMNS
            + GRAPH_PAIR_FEATURE_COLUMNS
        )
    if feature_set == FEATURE_SET_YELLOW_V1:
        return list(BASE_PAIR_FEATURE_COLUMNS) + YELLOW_PAIR_FEATURE_COLUMNS
    if feature_set == FEATURE_SET_META_GRAPH_YELLOW_V1:
        return list(BASE_PAIR_FEATURE_COLUMNS) + YELLOW_PAIR_FEATURE_COLUMNS + METADATA_PAIR_FEATURE_COLUMNS + GRAPH_PAIR_FEATURE_COLUMNS
    raise ValueError(f"Unsupported Salamander pair feature set: {feature_set}")


def build_pair_feature_table(
    *,
    metadata_df: pd.DataFrame,
    local_pair_df: pd.DataFrame,
    route_score: np.ndarray,
    fusion_score: np.ndarray,
    student_score: np.ndarray,
    distill_score: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    identity_series = (
        metadata_df["identity"].fillna("").astype(str)
        if "identity" in metadata_df.columns
        else pd.Series([""] * len(metadata_df))
    )
    has_identity = identity_series.ne("").any()
    for row in local_pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        left_meta = metadata_df.iloc[left_index]
        right_meta = metadata_df.iloc[right_index]
        route_global_score = float(route_score[left_index, right_index])
        fusion_global_score = float(fusion_score[left_index, right_index])
        student_global_score = float(student_score[left_index, right_index])
        distill_global_score = float(distill_score[left_index, right_index])
        left_identity = "" if pd.isna(left_meta.get("identity", "")) else str(left_meta.get("identity", ""))
        right_identity = "" if pd.isna(right_meta.get("identity", "")) else str(right_meta.get("identity", ""))
        same_identity = int(left_identity == right_identity) if has_identity else -1
        left_keypoints = int(row.left_keypoints)
        right_keypoints = int(row.right_keypoints)
        inliers = int(row.inliers)
        good_matches = int(row.good_matches)
        keypoint_min = max(1, min(left_keypoints, right_keypoints))
        rows.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "image_id": str(left_meta["image_id"]),
                "neighbor_image_id": str(right_meta["image_id"]),
                "identity": left_identity,
                "neighbor_identity": right_identity,
                "same_identity": same_identity,
                "route_global_score": route_global_score,
                "fusion_global_score": fusion_global_score,
                "student_global_score": student_global_score,
                "distill_global_score": distill_global_score,
                "route_minus_fusion": route_global_score - fusion_global_score,
                "route_minus_student": route_global_score - student_global_score,
                "student_minus_distill": student_global_score - distill_global_score,
                "local_score": float(row.local_score),
                "local_raw_score": float(row.local_raw_score),
                "inliers": inliers,
                "good_matches": good_matches,
                "left_keypoints": left_keypoints,
                "right_keypoints": right_keypoints,
                "keypoint_min": keypoint_min,
                "keypoint_max": max(left_keypoints, right_keypoints),
                "inlier_ratio": float(inliers / keypoint_min),
                "match_density": float(good_matches / keypoint_min),
            }
        )
    return pd.DataFrame(rows)


def append_metadata_pair_features(pair_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    metadata = metadata_df.copy().reset_index(drop=True)
    metadata["image_id"] = metadata["image_id"].astype(str)
    orientation_series = metadata["orientation"] if "orientation" in metadata.columns else pd.Series([""] * len(metadata))
    metadata["orientation_norm"] = orientation_series.apply(normalize_orientation_value)
    date_series = metadata["date"] if "date" in metadata.columns else pd.Series([""] * len(metadata))
    parsed_dates = pd.to_datetime(date_series, errors="coerce")
    epoch = pd.Timestamp("1970-01-01")
    metadata["capture_day_ordinal"] = np.where(
        parsed_dates.notna(),
        (parsed_dates - epoch).dt.days.astype(np.float32),
        np.nan,
    )
    left_df = metadata[["image_id", "orientation_norm", "capture_day_ordinal"]].rename(
        columns={
            "orientation_norm": "left_orientation_norm",
            "capture_day_ordinal": "left_capture_day_ordinal",
        }
    )
    right_df = metadata[["image_id", "orientation_norm", "capture_day_ordinal"]].rename(
        columns={
            "image_id": "neighbor_image_id",
            "orientation_norm": "right_orientation_norm",
            "capture_day_ordinal": "right_capture_day_ordinal",
        }
    )
    merged = pair_df.merge(left_df, on="image_id", how="left").merge(right_df, on="neighbor_image_id", how="left")
    left_orientation = merged["left_orientation_norm"].fillna("").astype(str)
    right_orientation = merged["right_orientation_norm"].fillna("").astype(str)
    orientation_known = left_orientation.ne("") & right_orientation.ne("")
    merged["orientation_known_pair"] = orientation_known.astype(int)
    merged["same_orientation"] = (orientation_known & left_orientation.eq(right_orientation)).astype(int)
    orientation_pair_key = [
        canonical_orientation_pair(left_value, right_value)
        for left_value, right_value in zip(left_orientation.tolist(), right_orientation.tolist())
    ]
    for bucket in ORIENTATION_PAIR_BUCKETS:
        column_name = f"orientation_pair_is_{bucket}"
        merged[column_name] = np.asarray([int(key == bucket) for key in orientation_pair_key], dtype=np.int32)

    left_capture = merged["left_capture_day_ordinal"].to_numpy(dtype=np.float32, copy=True)
    right_capture = merged["right_capture_day_ordinal"].to_numpy(dtype=np.float32, copy=True)
    capture_known = np.isfinite(left_capture) & np.isfinite(right_capture)
    capture_gap = np.where(capture_known, np.abs(left_capture - right_capture), 0.0).astype(np.float32)
    merged["capture_date_known_pair"] = capture_known.astype(np.int32)
    merged["same_capture_date"] = (capture_known & np.isclose(capture_gap, 0.0)).astype(np.int32)
    merged["capture_day_gap"] = capture_gap
    merged["capture_day_gap_log1p"] = np.log1p(capture_gap).astype(np.float32)
    merged["capture_day_gap_le_7"] = (capture_known & (capture_gap <= 7.0)).astype(np.int32)
    merged["capture_day_gap_le_30"] = (capture_known & (capture_gap <= 30.0)).astype(np.int32)
    merged["capture_day_gap_le_180"] = (capture_known & (capture_gap <= 180.0)).astype(np.int32)
    return merged


def append_route_graph_pair_features(pair_df: pd.DataFrame, route_score: np.ndarray, top_k: int = 10) -> pd.DataFrame:
    score_matrix = np.asarray(route_score, dtype=np.float32)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("route_score must be a square similarity matrix.")
    sample_count = int(score_matrix.shape[0])
    if sample_count <= 1:
        merged = pair_df.copy()
        for column in GRAPH_PAIR_FEATURE_COLUMNS:
            merged[column] = 0.0
        return merged

    effective_top_k = max(1, min(int(top_k), sample_count - 1))
    sorted_index = np.argsort(-score_matrix, axis=1, kind="mergesort")
    rank_matrix = np.empty_like(sorted_index, dtype=np.int32)
    rank_matrix[np.arange(sample_count)[:, None], sorted_index] = np.arange(sample_count, dtype=np.int32)
    top_neighbor_arrays: list[np.ndarray] = []
    top_neighbor_sets: list[set[int]] = []
    for row_index in range(sample_count):
        neighbor_index = sorted_index[row_index]
        neighbor_index = neighbor_index[neighbor_index != row_index][:effective_top_k].astype(np.int32, copy=False)
        top_neighbor_arrays.append(neighbor_index)
        top_neighbor_sets.append(set(int(value) for value in neighbor_index.tolist()))

    denominator = float(max(1, sample_count - 1))
    rows: list[dict[str, float | int]] = []
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        left_rank = int(rank_matrix[left_index, right_index])
        right_rank = int(rank_matrix[right_index, left_index])
        left_rank_pct = float(left_rank / denominator)
        right_rank_pct = float(right_rank / denominator)
        shared_neighbors = sorted(top_neighbor_sets[left_index] & top_neighbor_sets[right_index])
        shared_count = int(len(shared_neighbors))
        union_count = int(len(top_neighbor_sets[left_index] | top_neighbor_sets[right_index]))
        if shared_neighbors:
            shared_index = np.asarray(shared_neighbors, dtype=np.int32)
            shared_scores = 0.5 * (
                score_matrix[left_index, shared_index] + score_matrix[right_index, shared_index]
            )
            shared_mean_score = float(np.mean(shared_scores))
            shared_best_score = float(np.max(shared_scores))
        else:
            shared_mean_score = 0.0
            shared_best_score = 0.0
        shared_ratio = float(shared_count / max(1, union_count))
        rows.append(
            {
                "route_rank_pct_left_to_right": left_rank_pct,
                "route_rank_pct_right_to_left": right_rank_pct,
                "route_rank_pct_mean": 0.5 * (left_rank_pct + right_rank_pct),
                "route_rank_pct_max": max(left_rank_pct, right_rank_pct),
                "route_rank_gap_pct": abs(left_rank_pct - right_rank_pct),
                "route_reciprocal_rank_mean": 0.5 * ((1.0 / max(left_rank, 1)) + (1.0 / max(right_rank, 1))),
                "route_mutual_topk": int(left_rank <= effective_top_k and right_rank <= effective_top_k),
                "route_shared_neighbor_count": shared_count,
                "route_shared_neighbor_ratio": shared_ratio,
                "route_shared_neighbor_mean_score": shared_mean_score,
                "route_shared_neighbor_best_score": shared_best_score,
                "route_shared_neighbor_support_score": float(shared_ratio * shared_mean_score),
            }
        )
    graph_df = pd.DataFrame(rows)
    return pd.concat([pair_df.reset_index(drop=True), graph_df], axis=1)


def append_dual_view_pair_features(
    pair_df: pd.DataFrame,
    *,
    masked_student_score: np.ndarray,
    masked_distill_score: np.ndarray,
    student_cross_score: np.ndarray,
    distill_cross_score: np.ndarray,
) -> pd.DataFrame:
    masked_student_score = np.asarray(masked_student_score, dtype=np.float32)
    masked_distill_score = np.asarray(masked_distill_score, dtype=np.float32)
    student_cross_score = np.asarray(student_cross_score, dtype=np.float32)
    distill_cross_score = np.asarray(distill_cross_score, dtype=np.float32)
    row_count = int(masked_student_score.shape[0])
    for matrix_name, matrix in [
        ("masked_student_score", masked_student_score),
        ("masked_distill_score", masked_distill_score),
        ("student_cross_score", student_cross_score),
        ("distill_cross_score", distill_cross_score),
    ]:
        if matrix.ndim != 2 or matrix.shape[0] != row_count or matrix.shape[1] != row_count:
            raise ValueError(f"{matrix_name} must be a square matrix aligned to pair_df indices.")

    rows: list[dict[str, float]] = []
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        student_orig_score = float(row.student_global_score)
        distill_orig_score = float(row.distill_global_score)
        masked_student_global_score = float(masked_student_score[left_index, right_index])
        masked_distill_global_score = float(masked_distill_score[left_index, right_index])
        student_cross_lr = float(student_cross_score[left_index, right_index])
        student_cross_rl = float(student_cross_score[right_index, left_index])
        distill_cross_lr = float(distill_cross_score[left_index, right_index])
        distill_cross_rl = float(distill_cross_score[right_index, left_index])
        student_cross_mean = 0.5 * (student_cross_lr + student_cross_rl)
        distill_cross_mean = 0.5 * (distill_cross_lr + distill_cross_rl)
        rows.append(
            {
                "masked_student_global_score": masked_student_global_score,
                "masked_distill_global_score": masked_distill_global_score,
                "student_masked_score_mean": 0.5 * (student_orig_score + masked_student_global_score),
                "student_masked_score_gap": student_orig_score - masked_student_global_score,
                "student_cross_score_mean": student_cross_mean,
                "student_cross_score_max": max(student_cross_lr, student_cross_rl),
                "distill_masked_score_mean": 0.5 * (distill_orig_score + masked_distill_global_score),
                "distill_masked_score_gap": distill_orig_score - masked_distill_global_score,
                "distill_cross_score_mean": distill_cross_mean,
                "distill_cross_score_max": max(distill_cross_lr, distill_cross_rl),
                "dual_view_global_score_mean": float(
                    np.mean(
                        [
                            student_orig_score,
                            masked_student_global_score,
                            distill_orig_score,
                            masked_distill_global_score,
                        ]
                    )
                ),
                "dual_view_cross_score_mean": float(np.mean([student_cross_mean, distill_cross_mean])),
            }
        )

    dual_df = pd.DataFrame(rows)
    return pd.concat([pair_df.reset_index(drop=True), dual_df], axis=1)


def append_feature_set(
    *,
    pair_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    route_score: np.ndarray,
    feature_set: str,
    graph_top_k: int = 10,
    masked_student_score: np.ndarray | None = None,
    masked_distill_score: np.ndarray | None = None,
    student_cross_score: np.ndarray | None = None,
    distill_cross_score: np.ndarray | None = None,
    yellow_pair_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if feature_set == FEATURE_SET_BASIC:
        return pair_df.copy()
    if feature_set == FEATURE_SET_YELLOW_V1:
        if yellow_pair_df is None:
            raise ValueError("yellow_v1 requires precomputed yellow_pair_df.")
        return yellow_pair_df.copy().reset_index(drop=True)
    if feature_set == FEATURE_SET_DUAL_VIEW_V1:
        if (
            masked_student_score is None
            or masked_distill_score is None
            or student_cross_score is None
            or distill_cross_score is None
        ):
            raise ValueError("dual_view_v1 requires masked and cross-view score matrices.")
        return append_dual_view_pair_features(
            pair_df=pair_df,
            masked_student_score=masked_student_score,
            masked_distill_score=masked_distill_score,
            student_cross_score=student_cross_score,
            distill_cross_score=distill_cross_score,
        )
    if feature_set == FEATURE_SET_META_GRAPH_V1:
        enriched = append_metadata_pair_features(pair_df=pair_df, metadata_df=metadata_df)
        return append_route_graph_pair_features(pair_df=enriched, route_score=route_score, top_k=graph_top_k)
    if feature_set == FEATURE_SET_META_GRAPH_YELLOW_V1:
        if yellow_pair_df is None:
            raise ValueError("meta_graph_yellow_v1 requires precomputed yellow_pair_df.")
        enriched = append_metadata_pair_features(pair_df=yellow_pair_df, metadata_df=metadata_df)
        return append_route_graph_pair_features(pair_df=enriched, route_score=route_score, top_k=graph_top_k)
    if feature_set == FEATURE_SET_META_GRAPH_DUAL_VIEW_V1:
        if (
            masked_student_score is None
            or masked_distill_score is None
            or student_cross_score is None
            or distill_cross_score is None
        ):
            raise ValueError("meta_graph_dual_view_v1 requires masked and cross-view score matrices.")
        enriched = append_dual_view_pair_features(
            pair_df=pair_df,
            masked_student_score=masked_student_score,
            masked_distill_score=masked_distill_score,
            student_cross_score=student_cross_score,
            distill_cross_score=distill_cross_score,
        )
        enriched = append_metadata_pair_features(pair_df=enriched, metadata_df=metadata_df)
        return append_route_graph_pair_features(pair_df=enriched, route_score=route_score, top_k=graph_top_k)
    raise ValueError(f"Unsupported Salamander pair feature set: {feature_set}")
