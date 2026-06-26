from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .body_orientation_probe import (
    DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    DEFAULT_ROTATION_CANVAS_FILL_MODE,
    compute_body_axis,
    extract_largest_component,
    rotate_and_crop,
    rotation_to_horizontal,
)
from .descriptor_baselines import PATH_COLUMN, dataframe_to_markdown_table
from .orb_rerank_baseline import (
    OrbFeature,
    build_local_match_table,
    extract_local_features,
    normalize_local_matcher_name,
)


SAM_ORB_VETO_ANALYSIS_NAME = "sam_orb_veto_v1"
MASKED_ALIGNED_VIEW_NAME = "sam_orb_veto_masked_aligned_v1"
MASKED_PATH_COLUMN = "sam_orb_veto_masked_path_v1"
ALIGNED_PATH_COLUMN = "sam_orb_veto_aligned_path_v1"

DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS = 512
DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE = 0.20
DEFAULT_LOCAL_MIN_KEYPOINTS = 24
DEFAULT_SUPPORT_LOCAL_SCORE = 0.18
DEFAULT_SUPPORT_INLIERS = 8
DEFAULT_FAIL_LOCAL_SCORE = 0.03
DEFAULT_FAIL_INLIERS = 4
DEFAULT_HARD_VETO_SCORE_MAX = 0.05
DEFAULT_SOFT_MASKED_SCORE_MAX = 0.08
DEFAULT_SOFT_ALIGNED_SCORE_MAX = 0.10
DEFAULT_HARD_VETO_SCORE_CAP = 0.02
DEFAULT_SOFT_VETO_SCORE_SCALE = 0.65

LOCAL_VALUE_COLUMNS = [
    "left_keypoints",
    "right_keypoints",
    "good_matches",
    "inliers",
    "local_raw_score",
    "local_score",
]


def infer_mask_from_masked_rgb(image: Image.Image, *, nonzero_threshold: int = 1) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.any(rgb > int(nonzero_threshold), axis=2).astype(np.uint8)
    return extract_largest_component(mask)


def _empty_feature(image_id: str, matcher_name: str) -> OrbFeature:
    return OrbFeature(
        image_id=str(image_id),
        matcher_name=str(matcher_name),
        point_count=0,
        points=np.empty((0, 2), dtype=np.float32),
        descriptors=None,
        width=0,
        height=0,
    )


def _resolve_relative_view_path(original_path: str, *, dataset: str = "") -> Path:
    path = Path(str(original_path))
    parts = list(path.parts)
    if "images" in parts:
        index = parts.index("images")
        return Path(*parts[index + 1 :])
    for dataset_name in ["SalamanderID2025", "LynxID2025", "SeaTurtleID2022", "TexasHornedLizards"]:
        if dataset_name in parts:
            index = parts.index(dataset_name)
            return Path(*parts[index:])
    if dataset:
        return Path(str(dataset)) / path.name
    return Path(path.name)


def _coerce_optional_path(row: object, candidates: list[str]) -> str:
    for column in candidates:
        value = getattr(row, column, "")
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null"}:
            return text
    return ""


def build_masked_aligned_roi_manifest(
    *,
    reference_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    alignment_min_foreground_pixels: int = DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS,
    alignment_min_axis_confidence: float = DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE,
    padding_ratio: float = DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    nonzero_threshold: int = 1,
) -> pd.DataFrame:
    if reference_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "dataset",
                "split",
                "identity",
                "path",
                MASKED_PATH_COLUMN,
                "sam_orb_veto_masked_available_v1",
                "sam_orb_veto_mask_area_ratio_v1",
                "sam_orb_veto_foreground_pixels_v1",
                "sam_orb_veto_axis_angle_deg_v1",
                "sam_orb_veto_axis_confidence_v1",
                "sam_orb_veto_rotation_applied_deg_v1",
                "sam_orb_veto_alignment_status_v1",
                "sam_orb_veto_alignment_reason_v1",
                "sam_orb_veto_alignment_applied_v1",
                ALIGNED_PATH_COLUMN,
                "sam_orb_veto_canvas_fill_mode_v1",
            ]
        )

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    views_dir = output_dir / "views" / MASKED_ALIGNED_VIEW_NAME
    views_dir.mkdir(parents=True, exist_ok=True)

    working_df = reference_df.copy().reset_index(drop=True)
    working_df["image_id"] = working_df["image_id"].astype(str)
    working_df["dataset"] = working_df["dataset"].astype(str)
    if "identity" in working_df.columns:
        working_df["identity"] = working_df["identity"].fillna("").astype(str)
    enriched_lookup = enriched_df.copy().reset_index(drop=True)
    enriched_lookup["image_id"] = enriched_lookup["image_id"].astype(str)
    enriched_lookup["dataset"] = enriched_lookup["dataset"].astype(str)
    available_columns = [
        "image_id",
        "dataset",
        "sam_masked_rgb_v1_export_path",
        "sam_masked_rgb_v1_applied",
        "original_rgb_path_v1",
    ]
    available_columns = [column for column in available_columns if column in enriched_lookup.columns]
    lookup_df = enriched_lookup[available_columns].drop_duplicates(subset=["image_id", "dataset"])
    merged = working_df.merge(lookup_df, on=["image_id", "dataset"], how="left", validate="one_to_one")

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        identity = "" if pd.isna(getattr(row, "identity", "")) else str(getattr(row, "identity", ""))
        original_path = _coerce_optional_path(
            row,
            ["path", "source_global_path", "global_path", "original_rgb_path_v1"],
        )
        masked_path = ""
        if bool(getattr(row, "sam_masked_rgb_v1_applied", False)):
            masked_path = str(getattr(row, "sam_masked_rgb_v1_export_path", "") or "")
        if not masked_path:
            rows.append(
                {
                    "image_id": str(row.image_id),
                    "dataset": str(row.dataset),
                    "split": str(getattr(row, "split", "")),
                    "identity": identity,
                    "path": original_path,
                    MASKED_PATH_COLUMN: "",
                    "sam_orb_veto_masked_available_v1": False,
                    "sam_orb_veto_mask_area_ratio_v1": 0.0,
                    "sam_orb_veto_foreground_pixels_v1": 0.0,
                    "sam_orb_veto_axis_angle_deg_v1": 0.0,
                    "sam_orb_veto_axis_confidence_v1": 0.0,
                    "sam_orb_veto_rotation_applied_deg_v1": 0.0,
                    "sam_orb_veto_alignment_status_v1": "skip",
                    "sam_orb_veto_alignment_reason_v1": "no_sam_mask",
                    "sam_orb_veto_alignment_applied_v1": False,
                    ALIGNED_PATH_COLUMN: "",
                    "sam_orb_veto_canvas_fill_mode_v1": "constant",
                }
            )
            continue

        masked_abs_path = repo_root / masked_path
        if not masked_abs_path.exists():
            rows.append(
                {
                    "image_id": str(row.image_id),
                    "dataset": str(row.dataset),
                    "split": str(getattr(row, "split", "")),
                    "identity": identity,
                    "path": original_path,
                    MASKED_PATH_COLUMN: masked_path,
                    "sam_orb_veto_masked_available_v1": False,
                    "sam_orb_veto_mask_area_ratio_v1": 0.0,
                    "sam_orb_veto_foreground_pixels_v1": 0.0,
                    "sam_orb_veto_axis_angle_deg_v1": 0.0,
                    "sam_orb_veto_axis_confidence_v1": 0.0,
                    "sam_orb_veto_rotation_applied_deg_v1": 0.0,
                    "sam_orb_veto_alignment_status_v1": "skip",
                    "sam_orb_veto_alignment_reason_v1": "masked_path_missing",
                    "sam_orb_veto_alignment_applied_v1": False,
                    ALIGNED_PATH_COLUMN: "",
                    "sam_orb_veto_canvas_fill_mode_v1": "constant",
                }
            )
            continue

        with Image.open(masked_abs_path) as masked_image:
            masked_image = masked_image.convert("RGB")
            inferred_mask = infer_mask_from_masked_rgb(masked_image, nonzero_threshold=nonzero_threshold)
            foreground_pixels = float(inferred_mask.sum())
            mask_area_ratio = round(float(foreground_pixels / max(inferred_mask.size, 1)), 6)
            axis_stats = compute_body_axis(inferred_mask)

            rotation_applied_deg = 0.0
            alignment_status = "skip"
            alignment_reason = "no_axis"
            alignment_applied = False
            aligned_rel_path = ""
            axis_confidence = 0.0
            axis_angle_deg = 0.0
            if axis_stats is not None:
                axis_confidence = float(axis_stats["axis_confidence"])
                axis_angle_deg = float(axis_stats["axis_angle_deg"])
                if foreground_pixels < float(alignment_min_foreground_pixels):
                    alignment_reason = "small_mask"
                elif axis_confidence < float(alignment_min_axis_confidence):
                    alignment_reason = "low_axis_confidence"
                else:
                    rotation_applied_deg = rotation_to_horizontal(axis_angle_deg)
                    aligned_image, _aligned_mask = rotate_and_crop(
                        masked_image,
                        inferred_mask,
                        rotation_applied_deg,
                        background=(0, 0, 0),
                        padding_ratio=float(padding_ratio),
                        keep_background=False,
                        canvas_fill_mode="constant",
                    )
                    relative_image_path = _resolve_relative_view_path(original_path, dataset=str(getattr(row, "dataset", "")))
                    aligned_export_rel = Path("views") / MASKED_ALIGNED_VIEW_NAME / relative_image_path
                    aligned_export_abs = output_dir / aligned_export_rel
                    aligned_export_abs.parent.mkdir(parents=True, exist_ok=True)
                    aligned_image.save(aligned_export_abs, quality=95)
                    aligned_rel_path = str((output_dir.relative_to(repo_root) / aligned_export_rel).as_posix())
                    alignment_status = "apply"
                    alignment_reason = "ok"
                    alignment_applied = True

        rows.append(
            {
                "image_id": str(row.image_id),
                "dataset": str(row.dataset),
                "split": str(getattr(row, "split", "")),
                "identity": identity,
                "path": original_path,
                MASKED_PATH_COLUMN: masked_path,
                "sam_orb_veto_masked_available_v1": True,
                "sam_orb_veto_mask_area_ratio_v1": mask_area_ratio,
                "sam_orb_veto_foreground_pixels_v1": foreground_pixels,
                "sam_orb_veto_axis_angle_deg_v1": round(axis_angle_deg, 6),
                "sam_orb_veto_axis_confidence_v1": round(axis_confidence, 6),
                "sam_orb_veto_rotation_applied_deg_v1": round(float(rotation_applied_deg), 6),
                "sam_orb_veto_alignment_status_v1": alignment_status,
                "sam_orb_veto_alignment_reason_v1": alignment_reason,
                "sam_orb_veto_alignment_applied_v1": bool(alignment_applied),
                ALIGNED_PATH_COLUMN: aligned_rel_path,
                "sam_orb_veto_canvas_fill_mode_v1": "constant",
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def summarize_roi_manifest(roi_manifest_df: pd.DataFrame) -> pd.DataFrame:
    if roi_manifest_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "masked_available",
                "masked_available_ratio",
                "aligned_applied",
                "aligned_applied_ratio",
                "mean_axis_confidence",
            ]
        )
    return (
        roi_manifest_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            masked_available=("sam_orb_veto_masked_available_v1", lambda s: int(np.sum(s))),
            masked_available_ratio=("sam_orb_veto_masked_available_v1", lambda s: round(float(np.mean(s)), 4)),
            aligned_applied=("sam_orb_veto_alignment_applied_v1", lambda s: int(np.sum(s))),
            aligned_applied_ratio=("sam_orb_veto_alignment_applied_v1", lambda s: round(float(np.mean(s)), 4)),
            mean_axis_confidence=("sam_orb_veto_axis_confidence_v1", lambda s: round(float(np.mean(s)), 4)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )


def _build_optional_feature_list(
    *,
    reference_df: pd.DataFrame,
    repo_root: Path,
    path_column: str,
    nfeatures: int,
    max_side: int,
    fast_threshold: int,
    clahe_clip_limit: float,
    local_matcher: str,
    hflip: bool = False,
) -> list[OrbFeature]:
    matcher_name = normalize_local_matcher_name(local_matcher)
    working_df = reference_df.copy().reset_index(drop=True)
    working_df["image_id"] = working_df["image_id"].astype(str)
    working_df["dataset"] = working_df["dataset"].astype(str)
    valid_mask = working_df[path_column].fillna("").astype(str).ne("")
    feature_map: dict[tuple[str, str], OrbFeature] = {}
    if valid_mask.any():
        valid_df = working_df.loc[valid_mask].copy()
        valid_df[PATH_COLUMN] = valid_df[path_column].astype(str)
        extracted = extract_local_features(
            df=valid_df,
            repo_root=repo_root,
            nfeatures=int(nfeatures),
            max_side=int(max_side),
            fast_threshold=int(fast_threshold),
            clahe_clip_limit=float(clahe_clip_limit),
            local_matcher=matcher_name,
            hflip=bool(hflip),
        )
        for row, feature in zip(valid_df.itertuples(index=False), extracted):
            feature_map[(str(row.image_id), str(row.dataset))] = feature
    features: list[OrbFeature] = []
    for row in working_df.itertuples(index=False):
        key = (str(row.image_id), str(row.dataset))
        features.append(feature_map.get(key, _empty_feature(str(row.image_id), matcher_name)))
    return features


def pair_df_to_local_pair_index(
    pair_df: pd.DataFrame,
    *,
    global_score_col: str = "route_global_score",
) -> list[tuple[int, int, float]]:
    if pair_df.empty:
        return []
    rows: list[tuple[int, int, float]] = []
    for row in pair_df.itertuples(index=False):
        left_index = int(getattr(row, "left_index"))
        right_index = int(getattr(row, "right_index"))
        global_score = float(getattr(row, global_score_col, 0.0))
        rows.append((left_index, right_index, global_score))
    return rows


def _prefix_local_match_table(local_df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    if local_df.empty:
        columns = ["left_index", "right_index", "image_id", "neighbor_image_id"]
        return pd.DataFrame(columns=columns)
    keep_columns = ["left_index", "right_index", "image_id", "neighbor_image_id", *LOCAL_VALUE_COLUMNS]
    optional_columns = ["flip_invariant_enabled", "right_flipped_match_selected"]
    keep_columns.extend([column for column in optional_columns if column in local_df.columns])
    result = local_df[keep_columns].copy()
    result[f"{prefix}_keypoint_min"] = result[["left_keypoints", "right_keypoints"]].min(axis=1).astype(int)
    rename_map = {
        "left_keypoints": f"{prefix}_left_keypoints",
        "right_keypoints": f"{prefix}_right_keypoints",
        "good_matches": f"{prefix}_good_matches",
        "inliers": f"{prefix}_inliers",
        "local_raw_score": f"{prefix}_local_raw_score",
        "local_score": f"{prefix}_local_score",
    }
    if "flip_invariant_enabled" in result.columns:
        rename_map["flip_invariant_enabled"] = f"{prefix}_flip_invariant_enabled"
    if "right_flipped_match_selected" in result.columns:
        rename_map["right_flipped_match_selected"] = f"{prefix}_right_flipped_match_selected"
    result = result.rename(columns=rename_map)
    return result


def build_view_local_match_table(
    *,
    reference_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    repo_root: Path,
    path_column: str,
    nfeatures: int = 1024,
    max_side: int = 768,
    fast_threshold: int = 7,
    clahe_clip_limit: float = 2.0,
    ratio_test: float = 0.75,
    ransac_threshold: float = 5.0,
    min_inliers: int = 8,
    local_matcher: str = "orb",
    global_score_col: str = "route_global_score",
    prefix: str = "view",
    flip_invariant: bool = True,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "image_id", "neighbor_image_id"])
    features = _build_optional_feature_list(
        reference_df=reference_df,
        repo_root=repo_root,
        path_column=path_column,
        nfeatures=nfeatures,
        max_side=max_side,
        fast_threshold=fast_threshold,
        clahe_clip_limit=clahe_clip_limit,
        local_matcher=local_matcher,
    )
    flipped_features = None
    if bool(flip_invariant):
        flipped_features = _build_optional_feature_list(
            reference_df=reference_df,
            repo_root=repo_root,
            path_column=path_column,
            nfeatures=nfeatures,
            max_side=max_side,
            fast_threshold=fast_threshold,
            clahe_clip_limit=clahe_clip_limit,
            local_matcher=local_matcher,
            hflip=True,
        )
    local_pair_index = pair_df_to_local_pair_index(pair_df=pair_df, global_score_col=global_score_col)
    local_df = build_local_match_table(
        df=reference_df,
        features=features,
        flipped_features=flipped_features,
        pair_index=local_pair_index,
        ratio_test=float(ratio_test),
        ransac_threshold=float(ransac_threshold),
        min_inliers=int(min_inliers),
        local_matcher=local_matcher,
    )
    return _prefix_local_match_table(local_df=local_df, prefix=prefix)


def merge_veto_pair_features(
    *,
    base_pair_df: pd.DataFrame,
    masked_local_df: pd.DataFrame,
    aligned_local_df: pd.DataFrame,
) -> pd.DataFrame:
    result = base_pair_df.copy().reset_index(drop=True)
    for column in LOCAL_VALUE_COLUMNS:
        result[f"raw_{column}"] = result[column]
    result["raw_keypoint_min"] = result[["left_keypoints", "right_keypoints"]].min(axis=1).astype(int)
    result = result.merge(
        masked_local_df,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )
    result = result.merge(
        aligned_local_df,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )
    fill_columns = [
        "masked_left_keypoints",
        "masked_right_keypoints",
        "masked_good_matches",
        "masked_inliers",
        "masked_local_raw_score",
        "masked_local_score",
        "masked_keypoint_min",
        "aligned_left_keypoints",
        "aligned_right_keypoints",
        "aligned_good_matches",
        "aligned_inliers",
        "aligned_local_raw_score",
        "aligned_local_score",
        "aligned_keypoint_min",
    ]
    for column in fill_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
    int_like_columns = [
        "masked_left_keypoints",
        "masked_right_keypoints",
        "masked_good_matches",
        "masked_inliers",
        "masked_keypoint_min",
        "aligned_left_keypoints",
        "aligned_right_keypoints",
        "aligned_good_matches",
        "aligned_inliers",
        "aligned_keypoint_min",
    ]
    for column in int_like_columns:
        if column in result.columns:
            result[column] = result[column].astype(int)
    return result


def compile_veto_decisions(
    *,
    pair_feature_df: pd.DataFrame,
    roi_manifest_df: pd.DataFrame,
    min_pair_keypoints: int = DEFAULT_LOCAL_MIN_KEYPOINTS,
    support_local_score: float = DEFAULT_SUPPORT_LOCAL_SCORE,
    support_inliers: int = DEFAULT_SUPPORT_INLIERS,
    fail_local_score: float = DEFAULT_FAIL_LOCAL_SCORE,
    fail_inliers: int = DEFAULT_FAIL_INLIERS,
    hard_veto_score_max: float = DEFAULT_HARD_VETO_SCORE_MAX,
    soft_masked_score_max: float = DEFAULT_SOFT_MASKED_SCORE_MAX,
    soft_aligned_score_max: float = DEFAULT_SOFT_ALIGNED_SCORE_MAX,
) -> pd.DataFrame:
    if pair_feature_df.empty:
        return pair_feature_df.copy()

    roi_cols = [
        "image_id",
        "dataset",
        "sam_orb_veto_masked_available_v1",
        "sam_orb_veto_alignment_applied_v1",
        "sam_orb_veto_alignment_status_v1",
        "sam_orb_veto_alignment_reason_v1",
        "sam_orb_veto_axis_confidence_v1",
    ]
    lookup = roi_manifest_df[roi_cols].drop_duplicates(subset=["image_id", "dataset"]).copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup["dataset"] = lookup["dataset"].astype(str)

    result = pair_feature_df.copy().reset_index(drop=True)
    result["image_id"] = result["image_id"].astype(str)
    result["neighbor_image_id"] = result["neighbor_image_id"].astype(str)
    result["dataset"] = result["dataset"].astype(str)

    left_lookup = lookup.rename(
        columns={
            "image_id": "image_id",
            "sam_orb_veto_masked_available_v1": "left_masked_available_v1",
            "sam_orb_veto_alignment_applied_v1": "left_alignment_applied_v1",
            "sam_orb_veto_alignment_status_v1": "left_alignment_status_v1",
            "sam_orb_veto_alignment_reason_v1": "left_alignment_reason_v1",
            "sam_orb_veto_axis_confidence_v1": "left_axis_confidence_v1",
        }
    )
    right_lookup = lookup.rename(
        columns={
            "image_id": "neighbor_image_id",
            "sam_orb_veto_masked_available_v1": "right_masked_available_v1",
            "sam_orb_veto_alignment_applied_v1": "right_alignment_applied_v1",
            "sam_orb_veto_alignment_status_v1": "right_alignment_status_v1",
            "sam_orb_veto_alignment_reason_v1": "right_alignment_reason_v1",
            "sam_orb_veto_axis_confidence_v1": "right_axis_confidence_v1",
        }
    )
    result = result.merge(left_lookup, on=["image_id", "dataset"], how="left")
    result = result.merge(right_lookup, on=["neighbor_image_id", "dataset"], how="left")
    bool_fill_columns = [
        "left_masked_available_v1",
        "right_masked_available_v1",
        "left_alignment_applied_v1",
        "right_alignment_applied_v1",
    ]
    for column in bool_fill_columns:
        result[column] = result[column].fillna(False).astype(bool)

    result["masked_pair_valid"] = (
        result["left_masked_available_v1"]
        & result["right_masked_available_v1"]
        & result["masked_keypoint_min"].ge(int(min_pair_keypoints))
    )
    result["aligned_pair_valid"] = (
        result["left_alignment_applied_v1"]
        & result["right_alignment_applied_v1"]
        & result["aligned_keypoint_min"].ge(int(min_pair_keypoints))
    )

    result["masked_support"] = (
        result["masked_pair_valid"]
        & result["masked_local_score"].ge(float(support_local_score))
        & result["masked_inliers"].ge(int(support_inliers))
    )
    result["aligned_support"] = (
        result["aligned_pair_valid"]
        & result["aligned_local_score"].ge(float(support_local_score))
        & result["aligned_inliers"].ge(int(support_inliers))
    )
    result["pair_support"] = result["masked_support"] | result["aligned_support"]

    result["masked_fail"] = (
        result["masked_pair_valid"]
        & result["masked_local_score"].le(float(fail_local_score))
        & result["masked_inliers"].lt(int(fail_inliers))
    )
    result["aligned_fail"] = (
        result["aligned_pair_valid"]
        & result["aligned_local_score"].le(float(fail_local_score))
        & result["aligned_inliers"].lt(int(fail_inliers))
    )

    result["hard_veto"] = (
        result["masked_pair_valid"]
        & result["aligned_pair_valid"]
        & result["masked_fail"]
        & result["aligned_fail"]
        & (~result["pair_support"])
        & result["masked_local_score"].le(float(hard_veto_score_max))
        & result["aligned_local_score"].le(float(hard_veto_score_max))
    )
    result["soft_veto"] = (
        (~result["hard_veto"])
        & (~result["pair_support"])
        & (
            (
                result["masked_fail"]
                & result["aligned_pair_valid"]
                & result["aligned_local_score"].le(float(soft_aligned_score_max))
            )
            | (
                result["aligned_fail"]
                & result["masked_pair_valid"]
                & result["masked_local_score"].le(float(soft_masked_score_max))
            )
        )
    )

    result["veto_decision"] = np.select(
        [
            result["hard_veto"],
            result["soft_veto"],
            result["pair_support"],
        ],
        [
            "hard_veto",
            "soft_veto",
            "support",
        ],
        default="unknown",
    )
    result["veto_applied"] = result["veto_decision"].isin(["hard_veto", "soft_veto"])
    return result


def summarize_veto_decisions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame(
            columns=[
                "veto_decision",
                "pairs",
                "pair_ratio",
                "same_identity_pairs",
                "same_identity_ratio",
            ]
        )
    total = max(int(len(decision_df)), 1)
    has_truth = "same_identity" in decision_df.columns and decision_df["same_identity"].isin([0, 1, True, False]).any()
    rows: list[dict[str, object]] = []
    for decision, group in decision_df.groupby("veto_decision"):
        row: dict[str, object] = {
            "veto_decision": str(decision),
            "pairs": int(len(group)),
            "pair_ratio": round(float(len(group) / total), 6),
            "same_identity_pairs": 0,
            "same_identity_ratio": np.nan,
        }
        if has_truth:
            truth = group["same_identity"].astype(int)
            row["same_identity_pairs"] = int(truth.sum())
            row["same_identity_ratio"] = round(float(truth.mean()), 6) if len(truth) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["pairs", "veto_decision"], ascending=[False, True]).reset_index(drop=True)


def apply_veto_penalty_as_score(
    *,
    base_score: np.ndarray,
    decision_df: pd.DataFrame,
    hard_veto_score_cap: float = DEFAULT_HARD_VETO_SCORE_CAP,
    soft_veto_score_scale: float = DEFAULT_SOFT_VETO_SCORE_SCALE,
) -> np.ndarray:
    fused = np.asarray(base_score, dtype=np.float32).copy()
    for row in decision_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        current_score = float(fused[left_index, right_index])
        decision = str(getattr(row, "veto_decision", "unknown"))
        if decision == "hard_veto":
            updated = min(current_score, float(hard_veto_score_cap))
        elif decision == "soft_veto":
            updated = current_score * float(soft_veto_score_scale)
        else:
            continue
        fused[left_index, right_index] = updated
        fused[right_index, left_index] = updated
    np.fill_diagonal(fused, 1.0)
    return fused


def build_threshold_delta_table(
    *,
    baseline_df: pd.DataFrame,
    veto_df: pd.DataFrame,
) -> pd.DataFrame:
    join_cols = [column for column in ["dataset", "threshold"] if column in baseline_df.columns and column in veto_df.columns]
    merged = baseline_df.merge(
        veto_df,
        on=join_cols,
        how="inner",
        suffixes=("_baseline", "_veto"),
    )
    for metric in ["ari", "pairwise_f1", "nmi", "cluster_count", "singleton_cluster_ratio"]:
        baseline_col = f"{metric}_baseline"
        veto_col = f"{metric}_veto"
        if baseline_col in merged.columns and veto_col in merged.columns:
            merged[f"delta_{metric}"] = pd.to_numeric(merged[veto_col], errors="coerce") - pd.to_numeric(
                merged[baseline_col], errors="coerce"
            )
    return merged.sort_values("threshold").reset_index(drop=True)


def build_markdown_report(
    *,
    output_path: Path,
    config: dict[str, Any],
    roi_summary_df: pd.DataFrame,
    val_veto_summary_df: pd.DataFrame,
    test_veto_summary_df: pd.DataFrame,
    val_threshold_delta_df: pd.DataFrame,
    val_best_rows_df: pd.DataFrame,
    test_shape_df: pd.DataFrame,
) -> None:
    lines = [
        "# Salamander SAM+ORB Veto Probe v1",
        "",
        "## Config",
        "",
        f"- Analysis id: `{config['analysis_id']}`",
        f"- Route dir: `{config['route_dir']}`",
        f"- XGBoost variant dir: `{config['xgb_variant_dir']}`",
        f"- Threshold candidates: `{', '.join(str(v) for v in config['threshold_candidates'])}`",
        f"- Chosen threshold: `{config['chosen_threshold']}`",
        f"- Hard veto score cap: `{config['hard_veto_score_cap']}`",
        f"- Soft veto score scale: `{config['soft_veto_score_scale']}`",
        "",
        "## ROI Summary",
        "",
        dataframe_to_markdown_table(roi_summary_df),
        "",
        "## Val Veto Decisions",
        "",
        dataframe_to_markdown_table(val_veto_summary_df),
        "",
        "## Test Veto Decisions",
        "",
        dataframe_to_markdown_table(test_veto_summary_df),
        "",
        "## Val Threshold Delta",
        "",
        dataframe_to_markdown_table(val_threshold_delta_df),
        "",
        "## Best Rows",
        "",
        dataframe_to_markdown_table(val_best_rows_df),
        "",
        "## Test Shape",
        "",
        dataframe_to_markdown_table(test_shape_df),
        "",
        "## Reading Notes",
        "",
        "- 这是一条 `v1a` 最小版 probe：只验证 `masked-first aligned ORB` 负证据是否有用，不替换当前主路。",
        "- 若 `delta_ari` 与 `delta_pairwise_f1` 稳定为正，并且 `hard_veto` 的 same-id 误伤率较低，才值得升级到 `3-strip` 或 official。",
        "- 若簇数和 singleton 明显暴涨，即便局部指标上涨，也应视为潜在的过碎风险。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
