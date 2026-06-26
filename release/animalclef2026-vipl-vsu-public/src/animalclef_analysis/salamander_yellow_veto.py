from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .descriptor_baselines import dataframe_to_markdown_table
from .sam_orb_veto import (
    ALIGNED_PATH_COLUMN,
    MASKED_PATH_COLUMN,
    build_masked_aligned_roi_manifest,
    summarize_roi_manifest,
)

try:  # pragma: no cover - optional in light envs
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None


YELLOW_VETO_ANALYSIS_NAME = "yellow_veto_v1"
YELLOW_MASK_VIEW_NAME = "yellow_pattern_mask_v1"
YELLOW_MASK_PATH_COLUMN = "yellow_pattern_mask_path_v1"
DEFAULT_PROFILE_BINS = 16
DEFAULT_MIN_YELLOW_PIXELS = 48
DEFAULT_MIN_YELLOW_AREA_RATIO = 0.01
DEFAULT_MAX_YELLOW_AREA_RATIO = 0.60
DEFAULT_HARD_VETO_SCORE_CAP = 0.02
DEFAULT_SOFT_VETO_SCORE_SCALE = 0.70


def _component_masks(binary: np.ndarray) -> list[np.ndarray]:
    mask = (binary > 0).astype(np.uint8)
    if not mask.any():
        return []
    if ndimage is not None:
        labels, component_count = ndimage.label(mask)
        components: list[np.ndarray] = []
        for component_id in range(1, int(component_count) + 1):
            components.append((labels == component_id).astype(np.uint8))
        return components

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            component = np.zeros_like(mask, dtype=np.uint8)
            for cy, cx in coords:
                component[cy, cx] = 1
            components.append(component)
    return components


def remove_small_components(binary: np.ndarray, *, min_pixels: int) -> np.ndarray:
    kept = np.zeros_like(binary, dtype=np.uint8)
    for component in _component_masks(binary):
        if int(component.sum()) >= int(min_pixels):
            kept = np.maximum(kept, component.astype(np.uint8))
    return kept


def rgb_to_hsv_u8(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("HSV"), dtype=np.uint8)


def extract_yellow_mask(
    image: Image.Image,
    *,
    h_low: int = 18,
    h_high: int = 55,
    s_low: int = 45,
    v_low: int = 45,
    min_component_pixels: int = DEFAULT_MIN_YELLOW_PIXELS,
) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    foreground = np.any(rgb > 0, axis=2)
    hsv = rgb_to_hsv_u8(image)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    hsv_gate = (h >= int(h_low)) & (h <= int(h_high)) & (s >= int(s_low)) & (v >= int(v_low))
    rgb_gate = (
        (r >= 60)
        & (g >= 60)
        & (r >= b + 20)
        & (g >= b + 20)
        & (np.abs(r - g) <= 90)
    )
    yellow = foreground & hsv_gate & rgb_gate
    yellow = remove_small_components(yellow.astype(np.uint8), min_pixels=int(min_component_pixels))
    return yellow.astype(np.uint8)


def _profile_from_mask(mask: np.ndarray, *, bins: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(int(bins), dtype=np.float32)
    width = int(mask.shape[1])
    profile = mask.sum(axis=0).astype(np.float32)
    if float(profile.sum()) <= 0.0:
        return np.zeros(int(bins), dtype=np.float32)
    edges = np.linspace(0, width, int(bins) + 1, dtype=np.int32)
    binned = np.zeros(int(bins), dtype=np.float32)
    for index in range(int(bins)):
        left = int(edges[index])
        right = int(edges[index + 1])
        if right <= left:
            right = min(width, left + 1)
        binned[index] = float(profile[left:right].sum())
    total = float(binned.sum())
    if total <= 0.0:
        return np.zeros(int(bins), dtype=np.float32)
    return (binned / total).astype(np.float32, copy=False)


def compute_yellow_features(
    yellow_mask: np.ndarray,
    *,
    foreground_mask: np.ndarray | None = None,
    profile_bins: int = DEFAULT_PROFILE_BINS,
    min_yellow_pixels: int = DEFAULT_MIN_YELLOW_PIXELS,
    min_yellow_area_ratio: float = DEFAULT_MIN_YELLOW_AREA_RATIO,
    max_yellow_area_ratio: float = DEFAULT_MAX_YELLOW_AREA_RATIO,
) -> dict[str, Any]:
    yellow = (yellow_mask > 0).astype(np.uint8)
    if foreground_mask is None:
        foreground = np.ones_like(yellow, dtype=np.uint8)
    else:
        foreground = (foreground_mask > 0).astype(np.uint8)
    foreground_pixels = int(foreground.sum())
    yellow_pixels = int(yellow.sum())
    if foreground_pixels <= 0:
        area_ratio = 0.0
    else:
        area_ratio = float(yellow_pixels / foreground_pixels)

    components = _component_masks(yellow)
    component_sizes = sorted((int(component.sum()) for component in components), reverse=True)
    largest_component_pixels = int(component_sizes[0]) if component_sizes else 0
    component_count = int(len(component_sizes))
    centroid_x = 0.0
    centroid_y = 0.0
    if yellow_pixels > 0:
        ys, xs = np.where(yellow > 0)
        centroid_x = float(xs.mean() / max(yellow.shape[1] - 1, 1))
        centroid_y = float(ys.mean() / max(yellow.shape[0] - 1, 1))

    profile = _profile_from_mask(yellow, bins=int(profile_bins))
    quality = bool(
        (yellow_pixels >= int(min_yellow_pixels))
        and (area_ratio >= float(min_yellow_area_ratio))
        and (area_ratio <= float(max_yellow_area_ratio))
        and component_count >= 1
    )
    return {
        "yellow_pixels": yellow_pixels,
        "foreground_pixels": foreground_pixels,
        "yellow_area_ratio": round(float(area_ratio), 6),
        "yellow_component_count": component_count,
        "largest_yellow_component_ratio": round(float(largest_component_pixels / max(yellow_pixels, 1)), 6),
        "yellow_centroid_x": round(float(centroid_x), 6),
        "yellow_centroid_y": round(float(centroid_y), 6),
        "yellow_profile": profile.astype(np.float32),
        "yellow_quality_flag": quality,
        "yellow_presence_flag": bool(yellow_pixels > 0),
    }


def build_yellow_feature_manifest(
    *,
    roi_manifest_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    profile_bins: int = DEFAULT_PROFILE_BINS,
    min_yellow_pixels: int = DEFAULT_MIN_YELLOW_PIXELS,
    min_yellow_area_ratio: float = DEFAULT_MIN_YELLOW_AREA_RATIO,
    max_yellow_area_ratio: float = DEFAULT_MAX_YELLOW_AREA_RATIO,
) -> pd.DataFrame:
    if roi_manifest_df.empty:
        return pd.DataFrame()

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    views_dir = output_dir / "views" / YELLOW_MASK_VIEW_NAME
    views_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in roi_manifest_df.itertuples(index=False):
        image_id = str(row.image_id)
        dataset = str(row.dataset)
        split = str(getattr(row, "split", ""))
        identity = "" if pd.isna(getattr(row, "identity", "")) else str(getattr(row, "identity", ""))
        original_path = str(getattr(row, "path", ""))
        aligned_path = str(getattr(row, ALIGNED_PATH_COLUMN, "") or "")
        masked_path = str(getattr(row, MASKED_PATH_COLUMN, "") or "")
        source_path = aligned_path if aligned_path else masked_path
        source_kind = "aligned" if aligned_path else ("masked" if masked_path else "missing")
        yellow_mask_rel = ""
        if not source_path:
            feature_payload = compute_yellow_features(
                np.zeros((1, 1), dtype=np.uint8),
                foreground_mask=np.zeros((1, 1), dtype=np.uint8),
                profile_bins=int(profile_bins),
                min_yellow_pixels=int(min_yellow_pixels),
                min_yellow_area_ratio=float(min_yellow_area_ratio),
                max_yellow_area_ratio=float(max_yellow_area_ratio),
            )
            feature_payload["yellow_quality_flag"] = False
            feature_payload["yellow_presence_flag"] = False
        else:
            with Image.open(repo_root / source_path) as image:
                image = image.convert("RGB")
                rgb = np.asarray(image, dtype=np.uint8)
                foreground_mask = np.any(rgb > 0, axis=2).astype(np.uint8)
                yellow_mask = extract_yellow_mask(
                    image,
                    min_component_pixels=int(min_yellow_pixels),
                )
                feature_payload = compute_yellow_features(
                    yellow_mask,
                    foreground_mask=foreground_mask,
                    profile_bins=int(profile_bins),
                    min_yellow_pixels=int(min_yellow_pixels),
                    min_yellow_area_ratio=float(min_yellow_area_ratio),
                    max_yellow_area_ratio=float(max_yellow_area_ratio),
                )
                yellow_rgb = np.zeros_like(rgb, dtype=np.uint8)
                yellow_rgb[yellow_mask > 0] = np.array([255, 220, 0], dtype=np.uint8)
                relative_image_path = Path(original_path).relative_to("images")
                yellow_rel = Path("views") / YELLOW_MASK_VIEW_NAME / relative_image_path
                yellow_abs = output_dir / yellow_rel
                yellow_abs.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(yellow_rgb, mode="RGB").save(yellow_abs, quality=95)
                yellow_mask_rel = str((output_dir.relative_to(repo_root) / yellow_rel).as_posix())

        payload = {
            "image_id": image_id,
            "dataset": dataset,
            "split": split,
            "identity": identity,
            "path": original_path,
            "yellow_source_kind_v1": source_kind,
            YELLOW_MASK_PATH_COLUMN: yellow_mask_rel,
            "yellow_quality_flag_v1": bool(feature_payload["yellow_quality_flag"]),
            "yellow_presence_flag_v1": bool(feature_payload["yellow_presence_flag"]),
            "yellow_pixels_v1": int(feature_payload["yellow_pixels"]),
            "foreground_pixels_v1": int(feature_payload["foreground_pixels"]),
            "yellow_area_ratio_v1": float(feature_payload["yellow_area_ratio"]),
            "yellow_component_count_v1": int(feature_payload["yellow_component_count"]),
            "largest_yellow_component_ratio_v1": float(feature_payload["largest_yellow_component_ratio"]),
            "yellow_centroid_x_v1": float(feature_payload["yellow_centroid_x"]),
            "yellow_centroid_y_v1": float(feature_payload["yellow_centroid_y"]),
        }
        profile = np.asarray(feature_payload["yellow_profile"], dtype=np.float32)
        for index in range(int(profile_bins)):
            payload[f"yellow_profile_bin_{index:02d}_v1"] = round(float(profile[index]), 6)
        rows.append(payload)
    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def summarize_yellow_feature_manifest(yellow_df: pd.DataFrame) -> pd.DataFrame:
    if yellow_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "yellow_quality",
                "yellow_quality_ratio",
                "yellow_presence",
                "yellow_presence_ratio",
                "mean_yellow_area_ratio",
            ]
        )
    return (
        yellow_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            yellow_quality=("yellow_quality_flag_v1", lambda s: int(np.sum(s))),
            yellow_quality_ratio=("yellow_quality_flag_v1", lambda s: round(float(np.mean(s)), 4)),
            yellow_presence=("yellow_presence_flag_v1", lambda s: int(np.sum(s))),
            yellow_presence_ratio=("yellow_presence_flag_v1", lambda s: round(float(np.mean(s)), 4)),
            mean_yellow_area_ratio=("yellow_area_ratio_v1", lambda s: round(float(np.mean(s)), 4)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )


def _pair_profile_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in feature_df.columns if column.startswith("yellow_profile_bin_") and column.endswith("_v1")]


def _corr_or_zero(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    if np.allclose(left, 0.0) or np.allclose(right, 0.0):
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-8 or right_std <= 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def build_yellow_pair_features(
    *,
    pair_df: pd.DataFrame,
    yellow_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    profile_columns = _pair_profile_columns(yellow_feature_df)
    keep_columns = [
        "image_id",
        "dataset",
        "yellow_quality_flag_v1",
        "yellow_presence_flag_v1",
        "yellow_area_ratio_v1",
        "yellow_component_count_v1",
        "largest_yellow_component_ratio_v1",
        "yellow_centroid_x_v1",
        "yellow_centroid_y_v1",
        *profile_columns,
    ]
    lookup = yellow_feature_df[keep_columns].drop_duplicates(subset=["image_id", "dataset"]).copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup["dataset"] = lookup["dataset"].astype(str)

    result = pair_df.copy().reset_index(drop=True)
    result["image_id"] = result["image_id"].astype(str)
    result["neighbor_image_id"] = result["neighbor_image_id"].astype(str)
    result["dataset"] = result["dataset"].astype(str)

    left_lookup = lookup.rename(
        columns={column: f"left_{column}" for column in keep_columns if column not in {"image_id", "dataset"}}
    )
    right_lookup = lookup.rename(
        columns={
            "image_id": "neighbor_image_id",
            **{column: f"right_{column}" for column in keep_columns if column not in {"image_id", "dataset"}},
        }
    )
    result = result.merge(left_lookup, on=["image_id", "dataset"], how="left")
    result = result.merge(right_lookup, on=["neighbor_image_id", "dataset"], how="left")

    result["yellow_quality_both_v1"] = (
        result["left_yellow_quality_flag_v1"].fillna(False).astype(bool)
        & result["right_yellow_quality_flag_v1"].fillna(False).astype(bool)
    )
    result["yellow_presence_both_v1"] = (
        result["left_yellow_presence_flag_v1"].fillna(False).astype(bool)
        & result["right_yellow_presence_flag_v1"].fillna(False).astype(bool)
    )
    result["yellow_area_ratio_gap_v1"] = (
        pd.to_numeric(result["left_yellow_area_ratio_v1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(result["right_yellow_area_ratio_v1"], errors="coerce").fillna(0.0)
    ).abs()
    result["yellow_component_count_gap_v1"] = (
        pd.to_numeric(result["left_yellow_component_count_v1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(result["right_yellow_component_count_v1"], errors="coerce").fillna(0.0)
    ).abs()
    result["yellow_largest_component_gap_v1"] = (
        pd.to_numeric(result["left_largest_yellow_component_ratio_v1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(result["right_largest_yellow_component_ratio_v1"], errors="coerce").fillna(0.0)
    ).abs()
    result["yellow_centroid_shift_x_v1"] = (
        pd.to_numeric(result["left_yellow_centroid_x_v1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(result["right_yellow_centroid_x_v1"], errors="coerce").fillna(0.0)
    ).abs()
    result["yellow_centroid_shift_y_v1"] = (
        pd.to_numeric(result["left_yellow_centroid_y_v1"], errors="coerce").fillna(0.0)
        - pd.to_numeric(result["right_yellow_centroid_y_v1"], errors="coerce").fillna(0.0)
    ).abs()

    left_profile = result[[f"left_{column}" for column in profile_columns]].to_numpy(dtype=np.float32, copy=True)
    right_profile = result[[f"right_{column}" for column in profile_columns]].to_numpy(dtype=np.float32, copy=True)
    result["yellow_profile_l1_distance_v1"] = np.sum(np.abs(left_profile - right_profile), axis=1)
    result["yellow_profile_corr_v1"] = [
        _corr_or_zero(left_row, right_row)
        for left_row, right_row in zip(left_profile, right_profile)
    ]
    return result


def compile_yellow_veto_decisions(
    *,
    pair_feature_df: pd.DataFrame,
    hard_corr_max: float = 0.35,
    hard_l1_min: float = 1.00,
    hard_area_gap_min: float = 0.05,
    soft_corr_max: float = 0.55,
    soft_l1_min: float = 0.65,
    soft_area_gap_min: float = 0.03,
    support_corr_min: float = 0.88,
    support_l1_max: float = 0.35,
) -> pd.DataFrame:
    if pair_feature_df.empty:
        return pair_feature_df.copy()

    result = pair_feature_df.copy().reset_index(drop=True)
    quality = result["yellow_quality_both_v1"].fillna(False).astype(bool)
    corr = pd.to_numeric(result["yellow_profile_corr_v1"], errors="coerce").fillna(0.0)
    l1 = pd.to_numeric(result["yellow_profile_l1_distance_v1"], errors="coerce").fillna(0.0)
    area_gap = pd.to_numeric(result["yellow_area_ratio_gap_v1"], errors="coerce").fillna(0.0)
    component_gap = pd.to_numeric(result["yellow_component_count_gap_v1"], errors="coerce").fillna(0.0)

    result["yellow_support_v1"] = quality & (corr >= float(support_corr_min)) & (l1 <= float(support_l1_max))
    result["yellow_hard_veto_v1"] = (
        quality
        & (~result["yellow_support_v1"])
        & (corr <= float(hard_corr_max))
        & (l1 >= float(hard_l1_min))
        & ((area_gap >= float(hard_area_gap_min)) | (component_gap >= 2))
    )
    result["yellow_soft_veto_v1"] = (
        quality
        & (~result["yellow_support_v1"])
        & (~result["yellow_hard_veto_v1"])
        & (
            ((corr <= float(soft_corr_max)) & (l1 >= float(soft_l1_min)))
            | (area_gap >= float(soft_area_gap_min))
        )
    )
    result["yellow_veto_decision_v1"] = np.select(
        [
            result["yellow_hard_veto_v1"],
            result["yellow_soft_veto_v1"],
            result["yellow_support_v1"],
        ],
        [
            "hard_veto",
            "soft_veto",
            "support",
        ],
        default="unknown",
    )
    return result


def summarize_yellow_veto_decisions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame()
    total = max(int(len(decision_df)), 1)
    has_truth = "same_identity" in decision_df.columns and decision_df["same_identity"].isin([0, 1, True, False]).any()
    rows: list[dict[str, Any]] = []
    for decision, group in decision_df.groupby("yellow_veto_decision_v1"):
        row = {
            "yellow_veto_decision_v1": str(decision),
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
    return pd.DataFrame(rows).sort_values(["pairs", "yellow_veto_decision_v1"], ascending=[False, True]).reset_index(drop=True)


def apply_yellow_veto_penalty_as_score(
    *,
    base_score: np.ndarray,
    decision_df: pd.DataFrame,
    hard_veto_score_cap: float = DEFAULT_HARD_VETO_SCORE_CAP,
    soft_veto_score_scale: float = DEFAULT_SOFT_VETO_SCORE_SCALE,
) -> np.ndarray:
    fused = np.asarray(base_score, dtype=np.float32).copy()
    for row in decision_df.itertuples(index=False):
        decision = str(getattr(row, "yellow_veto_decision_v1", "unknown"))
        if decision not in {"hard_veto", "soft_veto"}:
            continue
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        current = float(fused[left_index, right_index])
        if decision == "hard_veto":
            updated = min(current, float(hard_veto_score_cap))
        else:
            updated = current * float(soft_veto_score_scale)
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
    merged = baseline_df.merge(veto_df, on=join_cols, how="inner", suffixes=("_baseline", "_veto"))
    for metric in ["ari", "pairwise_f1", "nmi", "cluster_count", "singleton_cluster_ratio"]:
        left = f"{metric}_baseline"
        right = f"{metric}_veto"
        if left in merged.columns and right in merged.columns:
            merged[f"delta_{metric}"] = pd.to_numeric(merged[right], errors="coerce") - pd.to_numeric(
                merged[left], errors="coerce"
            )
    return merged.sort_values("threshold").reset_index(drop=True)


def build_yellow_veto_report(
    *,
    output_path: Path,
    config: dict[str, Any],
    roi_summary_df: pd.DataFrame,
    yellow_summary_df: pd.DataFrame,
    val_veto_summary_df: pd.DataFrame,
    test_veto_summary_df: pd.DataFrame,
    threshold_delta_df: pd.DataFrame,
    val_best_rows_df: pd.DataFrame,
    test_shape_df: pd.DataFrame,
) -> None:
    lines = [
        "# Salamander Yellow Veto Probe v1",
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
        "## Yellow Single-Image Summary",
        "",
        dataframe_to_markdown_table(yellow_summary_df),
        "",
        "## Val Yellow Veto Summary",
        "",
        dataframe_to_markdown_table(val_veto_summary_df),
        "",
        "## Test Yellow Veto Summary",
        "",
        dataframe_to_markdown_table(test_veto_summary_df),
        "",
        "## Val Threshold Delta",
        "",
        dataframe_to_markdown_table(threshold_delta_df),
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
        "- 这条 probe 只验证一个先验：`Salamander` 的背部黄色图案分布是否能提供额外的错误 merge 否决证据。",
        "- `yellow_veto_v1` 仍然是叠加在当前 `Salamander` 强主线上的局部规则层，不是新的全局路由。",
        "- 若离线 delta 为正，下一步才值得考虑把黄色图案与 `ORB` 或更深的局部定位模型结合。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
