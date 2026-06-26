from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .salamander_yellow_orb_local import (
    YELLOW_BAND_PATH_COLUMN,
    YELLOW_FOCUS_PATH_COLUMN,
    build_yellow_band_manifest,
    build_patch_pair_features,
    build_yellow_focus_manifest,
    compile_yellow_orb_local_decisions,
    merge_yellow_orb_local_pair_features,
    summarize_patch_pair_features,
    summarize_yellow_band_manifest,
    summarize_yellow_focus_manifest,
    summarize_yellow_orb_local_decisions,
)
from .sam_orb_veto import (
    build_masked_aligned_roi_manifest,
    build_view_local_match_table,
    summarize_roi_manifest,
)


@dataclass(frozen=True)
class YellowPairFeatureArtifacts:
    roi_manifest_df: pd.DataFrame
    roi_summary_df: pd.DataFrame
    focus_df: pd.DataFrame
    focus_summary_df: pd.DataFrame
    band_df: pd.DataFrame
    band_summary_df: pd.DataFrame
    yellow_roi_local_df: pd.DataFrame
    patch_pair_df: pd.DataFrame
    patch_summary_df: pd.DataFrame
    pair_feature_df: pd.DataFrame
    decision_summary_df: pd.DataFrame
    yellow_local_view: str


def build_yellow_pair_feature_artifacts(
    *,
    reference_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    alignment_min_foreground_pixels: int = 512,
    alignment_min_axis_confidence: float = 0.20,
    orb_features: int = 1024,
    orb_max_side: int = 512,
    fast_threshold: int = 7,
    clahe_clip_limit: float = 2.0,
    ratio_test: float = 0.75,
    ransac_threshold: float = 5.0,
    min_inliers: int = 8,
    local_matcher: str = "orb",
    yellow_local_view: str = "focus",
) -> YellowPairFeatureArtifacts:
    yellow_local_view = str(yellow_local_view).strip().lower()
    if yellow_local_view not in {"focus", "band"}:
        raise ValueError(f"Unsupported yellow_local_view: {yellow_local_view}")
    working_reference_df = reference_df.copy().reset_index(drop=True)
    working_reference_df["image_id"] = working_reference_df["image_id"].astype(str)
    working_reference_df["dataset"] = working_reference_df["dataset"].astype(str)
    if "identity" in working_reference_df.columns:
        working_reference_df["identity"] = working_reference_df["identity"].fillna("").astype(str)

    working_pair_df = pair_df.copy().reset_index(drop=True)
    if working_pair_df.empty:
        empty = working_pair_df.copy()
        return YellowPairFeatureArtifacts(
            roi_manifest_df=pd.DataFrame(),
            roi_summary_df=pd.DataFrame(),
            focus_df=pd.DataFrame(),
            focus_summary_df=pd.DataFrame(),
            band_df=pd.DataFrame(),
            band_summary_df=pd.DataFrame(),
            yellow_roi_local_df=pd.DataFrame(),
            patch_pair_df=pd.DataFrame(),
            patch_summary_df=pd.DataFrame(),
            pair_feature_df=empty,
            decision_summary_df=pd.DataFrame(),
            yellow_local_view=yellow_local_view,
        )
    if "dataset" not in working_pair_df.columns:
        default_dataset = (
            str(working_reference_df["dataset"].iloc[0])
            if not working_reference_df.empty and "dataset" in working_reference_df.columns
            else ""
        )
        working_pair_df["dataset"] = default_dataset
    working_pair_df["dataset"] = working_pair_df["dataset"].fillna("").astype(str)

    required_ids = set(working_pair_df["image_id"].astype(str).tolist()) | set(working_pair_df["neighbor_image_id"].astype(str).tolist())
    roi_reference_df = working_reference_df[working_reference_df["image_id"].isin(required_ids)].copy().reset_index(drop=True)

    roi_manifest_df = build_masked_aligned_roi_manifest(
        reference_df=roi_reference_df,
        enriched_df=enriched_df,
        repo_root=repo_root,
        output_dir=output_dir,
        alignment_min_foreground_pixels=int(alignment_min_foreground_pixels),
        alignment_min_axis_confidence=float(alignment_min_axis_confidence),
    )
    roi_summary_df = summarize_roi_manifest(roi_manifest_df=roi_manifest_df)

    focus_df = build_yellow_focus_manifest(
        roi_manifest_df=roi_manifest_df,
        repo_root=repo_root,
        output_dir=output_dir,
    )
    focus_summary_df = summarize_yellow_focus_manifest(focus_df=focus_df)
    band_df = pd.DataFrame()
    band_summary_df = pd.DataFrame()
    local_view_path_column = YELLOW_FOCUS_PATH_COLUMN
    local_view_reference_df = working_reference_df.merge(
        focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
    )
    if yellow_local_view == "band":
        band_df = build_yellow_band_manifest(
            focus_df=focus_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        band_summary_df = summarize_yellow_band_manifest(band_df=band_df)
        local_view_reference_df = local_view_reference_df.merge(
            band_df[["image_id", "dataset", YELLOW_BAND_PATH_COLUMN]],
            on=["image_id", "dataset"],
            how="left",
        )
        local_view_path_column = YELLOW_BAND_PATH_COLUMN
    yellow_roi_local_df = build_view_local_match_table(
        reference_df=local_view_reference_df,
        pair_df=working_pair_df,
        repo_root=repo_root,
        path_column=local_view_path_column,
        nfeatures=int(orb_features),
        max_side=int(orb_max_side),
        fast_threshold=int(fast_threshold),
        clahe_clip_limit=float(clahe_clip_limit),
        ratio_test=float(ratio_test),
        ransac_threshold=float(ransac_threshold),
        min_inliers=int(min_inliers),
        local_matcher=str(local_matcher),
        prefix="yellow_roi",
    )
    patch_pair_df = build_patch_pair_features(
        pair_df=working_pair_df,
        focus_df=focus_df,
        repo_root=repo_root,
    )
    patch_summary_df = summarize_patch_pair_features(pair_df=patch_pair_df)
    merged_pair_df = merge_yellow_orb_local_pair_features(
        base_pair_df=working_pair_df,
        yellow_roi_local_df=yellow_roi_local_df,
        patch_pair_df=patch_pair_df,
    )
    decision_pair_df = compile_yellow_orb_local_decisions(
        pair_feature_df=merged_pair_df,
        focus_df=focus_df,
    )
    decision_summary_df = summarize_yellow_orb_local_decisions(decision_df=decision_pair_df)
    return YellowPairFeatureArtifacts(
        roi_manifest_df=roi_manifest_df,
        roi_summary_df=roi_summary_df,
        focus_df=focus_df,
        focus_summary_df=focus_summary_df,
        band_df=band_df,
        band_summary_df=band_summary_df,
        yellow_roi_local_df=yellow_roi_local_df,
        patch_pair_df=patch_pair_df,
        patch_summary_df=patch_summary_df,
        pair_feature_df=decision_pair_df,
        decision_summary_df=decision_summary_df,
        yellow_local_view=yellow_local_view,
    )
