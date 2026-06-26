from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from .descriptor_baselines import dataframe_to_markdown_table
from .salamander_yellow_veto import (
    DEFAULT_MAX_YELLOW_AREA_RATIO,
    DEFAULT_MIN_YELLOW_AREA_RATIO,
    DEFAULT_MIN_YELLOW_PIXELS,
    compute_yellow_features,
    extract_yellow_mask,
)
from .sam_orb_veto import (
    ALIGNED_PATH_COLUMN,
    LOCAL_VALUE_COLUMNS,
    MASKED_PATH_COLUMN,
    _resolve_relative_view_path,
)


YELLOW_ORB_LOCAL_ANALYSIS_NAME = "yellow_orb_local_v1"
YELLOW_FOCUS_VIEW_NAME = "yellow_orb_local_focus_v1"
YELLOW_FOCUS_PATH_COLUMN = "yellow_orb_local_focus_rgb_path_v1"
YELLOW_FOCUS_MASK_PATH_COLUMN = "yellow_orb_local_focus_mask_path_v1"
YELLOW_BAND_VIEW_NAME = "yellow_band_rgb_v1"
YELLOW_BAND_PATH_COLUMN = "yellow_band_rgb_path_v1"

DEFAULT_FOCUS_CONTEXT_RATIO_X = 0.28
DEFAULT_FOCUS_CONTEXT_RATIO_Y = 0.36
DEFAULT_FOCUS_MIN_SIDE = 96
DEFAULT_PATCH_WIDTH = 160
DEFAULT_PATCH_HEIGHT = 96
DEFAULT_PATCH_MIN_FOREGROUND_PIXELS = 128
DEFAULT_PATCH_MIN_YELLOW_PIXELS = 32
DEFAULT_PATCH_MIN_OVERLAP_PIXELS = 96
DEFAULT_BAND_DILATE_RADIUS = 9
DEFAULT_BAND_ERODE_RADIUS = 5
DEFAULT_BAND_MIN_PIXELS = 96

DEFAULT_SUPPORT_ORB_LOCAL_SCORE = 0.18
DEFAULT_SUPPORT_ORB_INLIERS = 8
DEFAULT_FAIL_ORB_LOCAL_SCORE = 0.04
DEFAULT_FAIL_ORB_INLIERS = 4
DEFAULT_SUPPORT_PATCH_GRAY_CORR = 0.90
DEFAULT_SUPPORT_PATCH_GRAY_ABSDIFF = 0.12
DEFAULT_SUPPORT_PATCH_MASK_IOU = 0.45
DEFAULT_HARD_PATCH_GRAY_CORR_MAX = 0.58
DEFAULT_HARD_PATCH_GRAY_ABSDIFF_MIN = 0.20
DEFAULT_HARD_PATCH_MASK_IOU_MAX = 0.18
DEFAULT_SOFT_PATCH_GRAY_CORR_MAX = 0.74
DEFAULT_SOFT_PATCH_GRAY_ABSDIFF_MIN = 0.16
DEFAULT_SOFT_PATCH_MASK_IOU_MAX = 0.22
DEFAULT_HARD_VETO_SCORE_CAP = 0.02
DEFAULT_SOFT_VETO_SCORE_SCALE = 0.70


def _corr_or_zero(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0:
        return 0.0
    if np.allclose(left, 0.0) or np.allclose(right, 0.0):
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-8 or right_std <= 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    left = int(xs.min())
    right = int(xs.max()) + 1
    return left, top, right, bottom


def _odd_filter_size(radius: int) -> int:
    radius = max(0, int(radius))
    return max(3, radius * 2 + 1)


def dilate_binary_mask(binary: np.ndarray, *, radius: int) -> np.ndarray:
    mask = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(radius) <= 0:
        return (mask > 0).astype(np.uint8)
    image = Image.fromarray(mask, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(size=_odd_filter_size(int(radius))))
    return (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.uint8)


def erode_binary_mask(binary: np.ndarray, *, radius: int) -> np.ndarray:
    mask = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(radius) <= 0:
        return (mask > 0).astype(np.uint8)
    image = Image.fromarray(mask, mode="L")
    eroded = image.filter(ImageFilter.MinFilter(size=_odd_filter_size(int(radius))))
    return (np.asarray(eroded, dtype=np.uint8) > 0).astype(np.uint8)


def build_yellow_band_mask(
    yellow_mask: np.ndarray,
    *,
    dilate_radius: int = DEFAULT_BAND_DILATE_RADIUS,
    erode_radius: int = DEFAULT_BAND_ERODE_RADIUS,
    min_band_pixels: int = DEFAULT_BAND_MIN_PIXELS,
) -> np.ndarray:
    yellow = (np.asarray(yellow_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if not yellow.any():
        return np.zeros_like(yellow, dtype=np.uint8)
    dilated = dilate_binary_mask(yellow, radius=int(dilate_radius))
    eroded = erode_binary_mask(yellow, radius=int(erode_radius))
    band = ((dilated > 0) & (eroded == 0)).astype(np.uint8)
    if int(band.sum()) < int(min_band_pixels):
        fallback = ((dilate_binary_mask(yellow, radius=max(1, int(dilate_radius) // 2)) > 0) & (yellow > 0)).astype(np.uint8)
        if int(fallback.sum()) >= int(min_band_pixels):
            return fallback.astype(np.uint8)
    return band.astype(np.uint8)


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    *,
    width: int,
    height: int,
    context_ratio_x: float,
    context_ratio_y: float,
    min_side: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    box_width = max(1, int(right - left))
    box_height = max(1, int(bottom - top))
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (top + bottom)

    target_width = max(int(round(box_width * (1.0 + 2.0 * float(context_ratio_x)))), int(min_side))
    target_height = max(int(round(box_height * (1.0 + 2.0 * float(context_ratio_y)))), int(min_side))
    target_width = min(int(width), max(1, target_width))
    target_height = min(int(height), max(1, target_height))

    new_left = int(round(center_x - target_width / 2.0))
    new_top = int(round(center_y - target_height / 2.0))
    new_right = new_left + target_width
    new_bottom = new_top + target_height

    if new_left < 0:
        new_right -= new_left
        new_left = 0
    if new_top < 0:
        new_bottom -= new_top
        new_top = 0
    if new_right > int(width):
        shift = new_right - int(width)
        new_left = max(0, new_left - shift)
        new_right = int(width)
    if new_bottom > int(height):
        shift = new_bottom - int(height)
        new_top = max(0, new_top - shift)
        new_bottom = int(height)
    return int(new_left), int(new_top), int(new_right), int(new_bottom)


def extract_yellow_focus_crop(
    image: Image.Image,
    yellow_mask: np.ndarray,
    *,
    context_ratio_x: float = DEFAULT_FOCUS_CONTEXT_RATIO_X,
    context_ratio_y: float = DEFAULT_FOCUS_CONTEXT_RATIO_Y,
    min_side: int = DEFAULT_FOCUS_MIN_SIDE,
) -> tuple[Image.Image | None, np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    bbox = _bbox_from_mask(yellow_mask)
    if bbox is None:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return None, empty, {"focus_available": False}

    crop_bbox = _expand_bbox(
        bbox,
        width=int(width),
        height=int(height),
        context_ratio_x=float(context_ratio_x),
        context_ratio_y=float(context_ratio_y),
        min_side=int(min_side),
    )
    left, top, right, bottom = crop_bbox
    crop_rgb = rgb[top:bottom, left:right]
    crop_mask = yellow_mask[top:bottom, left:right].astype(np.uint8, copy=False)
    focus_image = Image.fromarray(crop_rgb, mode="RGB")
    payload = {
        "focus_available": True,
        "focus_bbox_left": int(left),
        "focus_bbox_top": int(top),
        "focus_bbox_right": int(right),
        "focus_bbox_bottom": int(bottom),
        "focus_width": int(max(1, right - left)),
        "focus_height": int(max(1, bottom - top)),
    }
    return focus_image, crop_mask, payload


def build_yellow_focus_manifest(
    *,
    roi_manifest_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    min_yellow_pixels: int = DEFAULT_MIN_YELLOW_PIXELS,
    min_yellow_area_ratio: float = DEFAULT_MIN_YELLOW_AREA_RATIO,
    max_yellow_area_ratio: float = DEFAULT_MAX_YELLOW_AREA_RATIO,
    focus_context_ratio_x: float = DEFAULT_FOCUS_CONTEXT_RATIO_X,
    focus_context_ratio_y: float = DEFAULT_FOCUS_CONTEXT_RATIO_Y,
    focus_min_side: int = DEFAULT_FOCUS_MIN_SIDE,
) -> pd.DataFrame:
    if roi_manifest_df.empty:
        return pd.DataFrame()

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    views_dir = output_dir / "views" / YELLOW_FOCUS_VIEW_NAME
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
        focus_rgb_rel = ""
        focus_mask_rel = ""
        feature_payload = compute_yellow_features(
            np.zeros((1, 1), dtype=np.uint8),
            foreground_mask=np.zeros((1, 1), dtype=np.uint8),
            min_yellow_pixels=int(min_yellow_pixels),
            min_yellow_area_ratio=float(min_yellow_area_ratio),
            max_yellow_area_ratio=float(max_yellow_area_ratio),
        )
        feature_payload["yellow_quality_flag"] = False
        feature_payload["yellow_presence_flag"] = False
        focus_payload: dict[str, Any] = {
            "focus_available": False,
            "focus_bbox_left": 0,
            "focus_bbox_top": 0,
            "focus_bbox_right": 0,
            "focus_bbox_bottom": 0,
            "focus_width": 0,
            "focus_height": 0,
        }

        if source_path:
            with Image.open(repo_root / source_path) as image:
                image = image.convert("RGB")
                rgb = np.asarray(image, dtype=np.uint8)
                foreground_mask = np.any(rgb > 0, axis=2).astype(np.uint8)
                yellow_mask = extract_yellow_mask(image, min_component_pixels=int(min_yellow_pixels))
                feature_payload = compute_yellow_features(
                    yellow_mask,
                    foreground_mask=foreground_mask,
                    min_yellow_pixels=int(min_yellow_pixels),
                    min_yellow_area_ratio=float(min_yellow_area_ratio),
                    max_yellow_area_ratio=float(max_yellow_area_ratio),
                )
                if bool(feature_payload["yellow_quality_flag"]):
                    focus_image, focus_mask, focus_payload = extract_yellow_focus_crop(
                        image,
                        yellow_mask,
                        context_ratio_x=float(focus_context_ratio_x),
                        context_ratio_y=float(focus_context_ratio_y),
                        min_side=int(focus_min_side),
                    )
                    if focus_image is not None and bool(focus_payload["focus_available"]):
                        relative_image_path = _resolve_relative_view_path(original_path, dataset=str(dataset))
                        focus_rgb_export_rel = Path("views") / YELLOW_FOCUS_VIEW_NAME / relative_image_path
                        focus_rgb_abs = output_dir / focus_rgb_export_rel
                        focus_rgb_abs.parent.mkdir(parents=True, exist_ok=True)
                        focus_image.save(focus_rgb_abs, quality=95)
                        focus_rgb_rel = str((output_dir.relative_to(repo_root) / focus_rgb_export_rel).as_posix())

                        focus_mask_export_rel = Path("views") / YELLOW_FOCUS_VIEW_NAME / relative_image_path.with_suffix(".png")
                        focus_mask_abs = output_dir / focus_mask_export_rel
                        focus_mask_abs.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray((focus_mask > 0).astype(np.uint8) * 255, mode="L").save(focus_mask_abs)
                        focus_mask_rel = str((output_dir.relative_to(repo_root) / focus_mask_export_rel).as_posix())

        row_payload = {
            "image_id": image_id,
            "dataset": dataset,
            "split": split,
            "identity": identity,
            "path": original_path,
            "yellow_focus_source_kind_v1": source_kind,
            "yellow_quality_flag_v1": bool(feature_payload["yellow_quality_flag"]),
            "yellow_presence_flag_v1": bool(feature_payload["yellow_presence_flag"]),
            "yellow_area_ratio_v1": float(feature_payload["yellow_area_ratio"]),
            "yellow_component_count_v1": int(feature_payload["yellow_component_count"]),
            "largest_yellow_component_ratio_v1": float(feature_payload["largest_yellow_component_ratio"]),
            "yellow_centroid_x_v1": float(feature_payload["yellow_centroid_x"]),
            "yellow_centroid_y_v1": float(feature_payload["yellow_centroid_y"]),
            "yellow_focus_available_v1": bool(focus_payload["focus_available"]) and bool(focus_rgb_rel) and bool(focus_mask_rel),
            YELLOW_FOCUS_PATH_COLUMN: focus_rgb_rel,
            YELLOW_FOCUS_MASK_PATH_COLUMN: focus_mask_rel,
            "yellow_focus_bbox_left_v1": int(focus_payload["focus_bbox_left"]),
            "yellow_focus_bbox_top_v1": int(focus_payload["focus_bbox_top"]),
            "yellow_focus_bbox_right_v1": int(focus_payload["focus_bbox_right"]),
            "yellow_focus_bbox_bottom_v1": int(focus_payload["focus_bbox_bottom"]),
            "yellow_focus_width_v1": int(focus_payload["focus_width"]),
            "yellow_focus_height_v1": int(focus_payload["focus_height"]),
        }
        rows.append(row_payload)

    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def summarize_yellow_focus_manifest(focus_df: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "yellow_quality",
                "yellow_quality_ratio",
                "focus_available",
                "focus_available_ratio",
                "mean_focus_width",
                "mean_focus_height",
            ]
        )
    return (
        focus_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            yellow_quality=("yellow_quality_flag_v1", lambda s: int(np.sum(s))),
            yellow_quality_ratio=("yellow_quality_flag_v1", lambda s: round(float(np.mean(s)), 4)),
            focus_available=("yellow_focus_available_v1", lambda s: int(np.sum(s))),
            focus_available_ratio=("yellow_focus_available_v1", lambda s: round(float(np.mean(s)), 4)),
            mean_focus_width=("yellow_focus_width_v1", lambda s: round(float(np.mean(s)), 2)),
            mean_focus_height=("yellow_focus_height_v1", lambda s: round(float(np.mean(s)), 2)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )


def build_yellow_band_manifest(
    *,
    focus_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    dilate_radius: int = DEFAULT_BAND_DILATE_RADIUS,
    erode_radius: int = DEFAULT_BAND_ERODE_RADIUS,
    min_band_pixels: int = DEFAULT_BAND_MIN_PIXELS,
) -> pd.DataFrame:
    if focus_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "dataset",
                "split",
                "identity",
                "path",
                "yellow_band_available_v1",
                YELLOW_BAND_PATH_COLUMN,
                "yellow_band_pixels_v1",
            ]
        )

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    views_dir = output_dir / "views" / YELLOW_BAND_VIEW_NAME
    views_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    working_df = focus_df.copy().reset_index(drop=True)
    working_df["image_id"] = working_df["image_id"].astype(str)
    working_df["dataset"] = working_df["dataset"].astype(str)
    if "identity" in working_df.columns:
        working_df["identity"] = working_df["identity"].fillna("").astype(str)

    for row in working_df.itertuples(index=False):
        band_path = ""
        band_available = False
        band_pixels = 0
        focus_rgb_path = str(getattr(row, YELLOW_FOCUS_PATH_COLUMN, "") or "")
        focus_mask_path = str(getattr(row, YELLOW_FOCUS_MASK_PATH_COLUMN, "") or "")
        original_path = str(getattr(row, "path", "") or "")
        if focus_rgb_path and focus_mask_path:
            rgb_abs = repo_root / focus_rgb_path
            mask_abs = repo_root / focus_mask_path
            if rgb_abs.exists() and mask_abs.exists():
                with Image.open(rgb_abs) as focus_image, Image.open(mask_abs) as focus_mask_image:
                    focus_rgb = np.asarray(focus_image.convert("RGB"), dtype=np.uint8)
                    focus_mask = (np.asarray(focus_mask_image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
                    band_mask = build_yellow_band_mask(
                        focus_mask,
                        dilate_radius=int(dilate_radius),
                        erode_radius=int(erode_radius),
                        min_band_pixels=int(min_band_pixels),
                    )
                    band_pixels = int(np.asarray(band_mask, dtype=np.uint8).sum())
                    if band_pixels > 0:
                        band_rgb = np.zeros_like(focus_rgb, dtype=np.uint8)
                        band_rgb[band_mask > 0] = focus_rgb[band_mask > 0]
                        relative_image_path = (
                            _resolve_relative_view_path(original_path, dataset=str(getattr(row, "dataset", "")))
                            if original_path
                            else Path(f"{str(row.image_id)}.jpg")
                        )
                        export_rel = Path("views") / YELLOW_BAND_VIEW_NAME / relative_image_path
                        export_abs = output_dir / export_rel
                        export_abs.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(band_rgb, mode="RGB").save(export_abs, quality=95)
                        band_path = str((output_dir.relative_to(repo_root) / export_rel).as_posix())
                        band_available = True

        rows.append(
            {
                "image_id": str(row.image_id),
                "dataset": str(row.dataset),
                "split": str(getattr(row, "split", "")),
                "identity": "" if pd.isna(getattr(row, "identity", "")) else str(getattr(row, "identity", "")),
                "path": original_path,
                "yellow_band_available_v1": bool(band_available),
                YELLOW_BAND_PATH_COLUMN: band_path,
                "yellow_band_pixels_v1": int(band_pixels),
            }
        )

    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def summarize_yellow_band_manifest(band_df: pd.DataFrame) -> pd.DataFrame:
    if band_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "band_available",
                "band_available_ratio",
                "mean_band_pixels",
            ]
        )
    return (
        band_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            band_available=("yellow_band_available_v1", lambda s: int(np.sum(s))),
            band_available_ratio=("yellow_band_available_v1", lambda s: round(float(np.mean(s)), 4)),
            mean_band_pixels=("yellow_band_pixels_v1", lambda s: round(float(np.mean(s)), 2)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )


def _resize_array_with_pad(
    array: np.ndarray,
    *,
    target_width: int,
    target_height: int,
    is_mask: bool,
) -> np.ndarray:
    if array.ndim == 2:
        pil = Image.fromarray(array.astype(np.uint8), mode="L")
        canvas = Image.new("L", (int(target_width), int(target_height)), 0)
    else:
        pil = Image.fromarray(array.astype(np.uint8), mode="RGB")
        canvas = Image.new("RGB", (int(target_width), int(target_height)), (0, 0, 0))

    width, height = pil.size
    scale = min(float(target_width) / max(width, 1), float(target_height) / max(height, 1))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    resized = pil.resize((resized_width, resized_height), resample=resample)
    offset_x = max(0, (int(target_width) - resized_width) // 2)
    offset_y = max(0, (int(target_height) - resized_height) // 2)
    canvas.paste(resized, (offset_x, offset_y))
    return np.asarray(canvas)


def compute_patch_descriptor(
    image: Image.Image,
    yellow_mask: np.ndarray,
    *,
    target_width: int = DEFAULT_PATCH_WIDTH,
    target_height: int = DEFAULT_PATCH_HEIGHT,
    min_foreground_pixels: int = DEFAULT_PATCH_MIN_FOREGROUND_PIXELS,
    min_yellow_pixels: int = DEFAULT_PATCH_MIN_YELLOW_PIXELS,
) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    foreground = np.any(rgb > 0, axis=2).astype(np.uint8)
    yellow = (yellow_mask > 0).astype(np.uint8)

    gray_patch = _resize_array_with_pad(
        gray,
        target_width=int(target_width),
        target_height=int(target_height),
        is_mask=False,
    ).astype(np.float32) / 255.0
    foreground_patch = (
        _resize_array_with_pad(
            foreground.astype(np.uint8) * 255,
            target_width=int(target_width),
            target_height=int(target_height),
            is_mask=True,
        )
        > 0
    )
    yellow_patch = (
        _resize_array_with_pad(
            yellow.astype(np.uint8) * 255,
            target_width=int(target_width),
            target_height=int(target_height),
            is_mask=True,
        )
        > 0
    )

    normalized_gray = np.zeros_like(gray_patch, dtype=np.float32)
    if np.any(foreground_patch):
        fg_values = gray_patch[foreground_patch]
        mean = float(fg_values.mean())
        std = float(fg_values.std())
        if std <= 1e-6:
            std = 1.0
        normalized_gray[foreground_patch] = (fg_values - mean) / std

    column_profile = yellow_patch.sum(axis=0).astype(np.float32)
    total = float(column_profile.sum())
    if total > 0.0:
        column_profile /= total

    return {
        "patch_valid": bool(
            int(foreground_patch.sum()) >= int(min_foreground_pixels)
            and int(yellow_patch.sum()) >= int(min_yellow_pixels)
        ),
        "gray_patch": gray_patch.astype(np.float32, copy=False),
        "normalized_gray_patch": normalized_gray.astype(np.float32, copy=False),
        "foreground_patch": foreground_patch.astype(bool, copy=False),
        "yellow_patch": yellow_patch.astype(bool, copy=False),
        "yellow_column_profile": column_profile.astype(np.float32, copy=False),
        "foreground_pixels": int(foreground_patch.sum()),
        "yellow_pixels": int(yellow_patch.sum()),
    }


def compute_patch_pair_metrics(
    left_descriptor: dict[str, Any],
    right_descriptor: dict[str, Any],
    *,
    min_overlap_pixels: int = DEFAULT_PATCH_MIN_OVERLAP_PIXELS,
) -> dict[str, Any]:
    left_valid = bool(left_descriptor.get("patch_valid", False))
    right_valid = bool(right_descriptor.get("patch_valid", False))
    if not left_valid or not right_valid:
        return {
            "yellow_patch_pair_valid_v1": False,
            "yellow_patch_overlap_pixels_v1": 0,
            "yellow_patch_gray_corr_v1": 0.0,
            "yellow_patch_gray_absdiff_v1": 1.0,
            "yellow_patch_mask_iou_v1": 0.0,
            "yellow_patch_mask_dice_v1": 0.0,
            "yellow_patch_profile_corr_v1": 0.0,
            "yellow_patch_profile_l1_v1": 1.0,
        }

    left_foreground = np.asarray(left_descriptor["foreground_patch"], dtype=bool)
    right_foreground = np.asarray(right_descriptor["foreground_patch"], dtype=bool)
    overlap = left_foreground & right_foreground
    overlap_pixels = int(overlap.sum())
    if overlap_pixels < int(min_overlap_pixels):
        return {
            "yellow_patch_pair_valid_v1": False,
            "yellow_patch_overlap_pixels_v1": overlap_pixels,
            "yellow_patch_gray_corr_v1": 0.0,
            "yellow_patch_gray_absdiff_v1": 1.0,
            "yellow_patch_mask_iou_v1": 0.0,
            "yellow_patch_mask_dice_v1": 0.0,
            "yellow_patch_profile_corr_v1": 0.0,
            "yellow_patch_profile_l1_v1": 1.0,
        }

    left_gray = np.asarray(left_descriptor["gray_patch"], dtype=np.float32)
    right_gray = np.asarray(right_descriptor["gray_patch"], dtype=np.float32)
    left_norm = np.asarray(left_descriptor["normalized_gray_patch"], dtype=np.float32)
    right_norm = np.asarray(right_descriptor["normalized_gray_patch"], dtype=np.float32)
    gray_corr = _corr_or_zero(left_norm[overlap], right_norm[overlap])
    gray_absdiff = float(np.mean(np.abs(left_gray[overlap] - right_gray[overlap]))) if overlap_pixels else 1.0

    left_yellow = np.asarray(left_descriptor["yellow_patch"], dtype=bool)
    right_yellow = np.asarray(right_descriptor["yellow_patch"], dtype=bool)
    yellow_intersection = int(np.logical_and(left_yellow, right_yellow).sum())
    yellow_union = int(np.logical_or(left_yellow, right_yellow).sum())
    left_yellow_pixels = int(left_yellow.sum())
    right_yellow_pixels = int(right_yellow.sum())
    mask_iou = float(yellow_intersection / yellow_union) if yellow_union > 0 else 0.0
    denom = max(1, left_yellow_pixels + right_yellow_pixels)
    mask_dice = float((2.0 * yellow_intersection) / denom)

    left_profile = np.asarray(left_descriptor["yellow_column_profile"], dtype=np.float32)
    right_profile = np.asarray(right_descriptor["yellow_column_profile"], dtype=np.float32)
    profile_corr = _corr_or_zero(left_profile, right_profile)
    profile_l1 = float(np.sum(np.abs(left_profile - right_profile)))
    return {
        "yellow_patch_pair_valid_v1": True,
        "yellow_patch_overlap_pixels_v1": overlap_pixels,
        "yellow_patch_gray_corr_v1": round(float(gray_corr), 6),
        "yellow_patch_gray_absdiff_v1": round(float(gray_absdiff), 6),
        "yellow_patch_mask_iou_v1": round(float(mask_iou), 6),
        "yellow_patch_mask_dice_v1": round(float(mask_dice), 6),
        "yellow_patch_profile_corr_v1": round(float(profile_corr), 6),
        "yellow_patch_profile_l1_v1": round(float(profile_l1), 6),
    }


def build_patch_pair_features(
    *,
    pair_df: pd.DataFrame,
    focus_df: pd.DataFrame,
    repo_root: Path,
    target_width: int = DEFAULT_PATCH_WIDTH,
    target_height: int = DEFAULT_PATCH_HEIGHT,
    min_foreground_pixels: int = DEFAULT_PATCH_MIN_FOREGROUND_PIXELS,
    min_yellow_pixels: int = DEFAULT_PATCH_MIN_YELLOW_PIXELS,
    min_overlap_pixels: int = DEFAULT_PATCH_MIN_OVERLAP_PIXELS,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    repo_root = repo_root.resolve()
    focus_lookup = focus_df.copy().reset_index(drop=True)
    focus_lookup["image_id"] = focus_lookup["image_id"].astype(str)
    focus_lookup["dataset"] = focus_lookup["dataset"].astype(str)

    descriptor_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in focus_lookup.itertuples(index=False):
        image_id = str(row.image_id)
        dataset = str(row.dataset)
        rgb_path = str(getattr(row, YELLOW_FOCUS_PATH_COLUMN, "") or "")
        mask_path = str(getattr(row, YELLOW_FOCUS_MASK_PATH_COLUMN, "") or "")
        if not rgb_path or not mask_path:
            descriptor_map[(image_id, dataset)] = {"patch_valid": False}
            continue
        rgb_abs = repo_root / rgb_path
        mask_abs = repo_root / mask_path
        if not rgb_abs.exists() or not mask_abs.exists():
            descriptor_map[(image_id, dataset)] = {"patch_valid": False}
            continue
        with Image.open(rgb_abs) as image, Image.open(mask_abs) as mask_image:
            descriptor_map[(image_id, dataset)] = compute_patch_descriptor(
                image.convert("RGB"),
                np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0,
                target_width=int(target_width),
                target_height=int(target_height),
                min_foreground_pixels=int(min_foreground_pixels),
                min_yellow_pixels=int(min_yellow_pixels),
            )

    rows: list[dict[str, Any]] = []
    for row in pair_df.itertuples(index=False):
        left_key = (str(row.image_id), str(row.dataset))
        right_key = (str(row.neighbor_image_id), str(row.dataset))
        left_descriptor = descriptor_map.get(left_key, {"patch_valid": False})
        right_descriptor = descriptor_map.get(right_key, {"patch_valid": False})
        metrics = compute_patch_pair_metrics(
            left_descriptor,
            right_descriptor,
            min_overlap_pixels=int(min_overlap_pixels),
        )
        rows.append(
            {
                "left_index": int(getattr(row, "left_index")),
                "right_index": int(getattr(row, "right_index")),
                "image_id": str(getattr(row, "image_id")),
                "neighbor_image_id": str(getattr(row, "neighbor_image_id")),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def merge_yellow_orb_local_pair_features(
    *,
    base_pair_df: pd.DataFrame,
    yellow_roi_local_df: pd.DataFrame,
    patch_pair_df: pd.DataFrame,
) -> pd.DataFrame:
    result = base_pair_df.copy().reset_index(drop=True)
    for column in LOCAL_VALUE_COLUMNS:
        if column in result.columns:
            result[f"raw_{column}"] = result[column]
    if {"left_keypoints", "right_keypoints"}.issubset(result.columns):
        result["raw_keypoint_min"] = result[["left_keypoints", "right_keypoints"]].min(axis=1).astype(int)

    result = result.merge(
        yellow_roi_local_df,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )
    result = result.merge(
        patch_pair_df,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )

    fill_zero_columns = [
        "yellow_roi_left_keypoints",
        "yellow_roi_right_keypoints",
        "yellow_roi_good_matches",
        "yellow_roi_inliers",
        "yellow_roi_local_raw_score",
        "yellow_roi_local_score",
        "yellow_roi_keypoint_min",
        "yellow_patch_overlap_pixels_v1",
        "yellow_patch_gray_corr_v1",
        "yellow_patch_gray_absdiff_v1",
        "yellow_patch_mask_iou_v1",
        "yellow_patch_mask_dice_v1",
        "yellow_patch_profile_corr_v1",
        "yellow_patch_profile_l1_v1",
    ]
    for column in fill_zero_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    int_like_columns = [
        "yellow_roi_left_keypoints",
        "yellow_roi_right_keypoints",
        "yellow_roi_good_matches",
        "yellow_roi_inliers",
        "yellow_roi_keypoint_min",
        "yellow_patch_overlap_pixels_v1",
    ]
    for column in int_like_columns:
        if column in result.columns:
            result[column] = result[column].astype(int)
    if "yellow_patch_pair_valid_v1" in result.columns:
        result["yellow_patch_pair_valid_v1"] = result["yellow_patch_pair_valid_v1"].fillna(False).astype(bool)
    return result


def compile_yellow_orb_local_decisions(
    *,
    pair_feature_df: pd.DataFrame,
    focus_df: pd.DataFrame,
    min_orb_keypoints: int = 20,
    support_orb_local_score: float = DEFAULT_SUPPORT_ORB_LOCAL_SCORE,
    support_orb_inliers: int = DEFAULT_SUPPORT_ORB_INLIERS,
    fail_orb_local_score: float = DEFAULT_FAIL_ORB_LOCAL_SCORE,
    fail_orb_inliers: int = DEFAULT_FAIL_ORB_INLIERS,
    support_patch_gray_corr: float = DEFAULT_SUPPORT_PATCH_GRAY_CORR,
    support_patch_gray_absdiff: float = DEFAULT_SUPPORT_PATCH_GRAY_ABSDIFF,
    support_patch_mask_iou: float = DEFAULT_SUPPORT_PATCH_MASK_IOU,
    hard_patch_gray_corr_max: float = DEFAULT_HARD_PATCH_GRAY_CORR_MAX,
    hard_patch_gray_absdiff_min: float = DEFAULT_HARD_PATCH_GRAY_ABSDIFF_MIN,
    hard_patch_mask_iou_max: float = DEFAULT_HARD_PATCH_MASK_IOU_MAX,
    soft_patch_gray_corr_max: float = DEFAULT_SOFT_PATCH_GRAY_CORR_MAX,
    soft_patch_gray_absdiff_min: float = DEFAULT_SOFT_PATCH_GRAY_ABSDIFF_MIN,
    soft_patch_mask_iou_max: float = DEFAULT_SOFT_PATCH_MASK_IOU_MAX,
) -> pd.DataFrame:
    if pair_feature_df.empty:
        return pair_feature_df.copy()

    lookup = focus_df[
        [
            "image_id",
            "dataset",
            "yellow_quality_flag_v1",
            "yellow_focus_available_v1",
            "yellow_focus_source_kind_v1",
        ]
    ].drop_duplicates(subset=["image_id", "dataset"])
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup["dataset"] = lookup["dataset"].astype(str)

    result = pair_feature_df.copy().reset_index(drop=True)
    result["image_id"] = result["image_id"].astype(str)
    result["neighbor_image_id"] = result["neighbor_image_id"].astype(str)
    result["dataset"] = result["dataset"].astype(str)

    left_lookup = lookup.rename(
        columns={
            "yellow_quality_flag_v1": "left_yellow_quality_flag_v1",
            "yellow_focus_available_v1": "left_yellow_focus_available_v1",
            "yellow_focus_source_kind_v1": "left_yellow_focus_source_kind_v1",
        }
    )
    right_lookup = lookup.rename(
        columns={
            "image_id": "neighbor_image_id",
            "yellow_quality_flag_v1": "right_yellow_quality_flag_v1",
            "yellow_focus_available_v1": "right_yellow_focus_available_v1",
            "yellow_focus_source_kind_v1": "right_yellow_focus_source_kind_v1",
        }
    )
    result = result.merge(left_lookup, on=["image_id", "dataset"], how="left")
    result = result.merge(right_lookup, on=["neighbor_image_id", "dataset"], how="left")

    for column in [
        "left_yellow_quality_flag_v1",
        "right_yellow_quality_flag_v1",
        "left_yellow_focus_available_v1",
        "right_yellow_focus_available_v1",
        "yellow_patch_pair_valid_v1",
    ]:
        if column in result.columns:
            result[column] = result[column].fillna(False).astype(bool)

    result["yellow_focus_pair_valid_v1"] = (
        result["left_yellow_quality_flag_v1"]
        & result["right_yellow_quality_flag_v1"]
        & result["left_yellow_focus_available_v1"]
        & result["right_yellow_focus_available_v1"]
    )
    result["yellow_orb_pair_valid_v1"] = (
        result["yellow_focus_pair_valid_v1"]
        & result["yellow_roi_keypoint_min"].ge(int(min_orb_keypoints))
    )
    result["yellow_orb_support_v1"] = (
        result["yellow_orb_pair_valid_v1"]
        & result["yellow_roi_local_score"].ge(float(support_orb_local_score))
        & result["yellow_roi_inliers"].ge(int(support_orb_inliers))
    )
    result["yellow_patch_support_v1"] = (
        result["yellow_patch_pair_valid_v1"]
        & result["yellow_patch_gray_corr_v1"].ge(float(support_patch_gray_corr))
        & result["yellow_patch_gray_absdiff_v1"].le(float(support_patch_gray_absdiff))
        & result["yellow_patch_mask_iou_v1"].ge(float(support_patch_mask_iou))
    )
    result["yellow_pair_support_v1"] = result["yellow_orb_support_v1"] | result["yellow_patch_support_v1"]

    result["yellow_orb_fail_v1"] = (
        result["yellow_orb_pair_valid_v1"]
        & result["yellow_roi_local_score"].le(float(fail_orb_local_score))
        & result["yellow_roi_inliers"].lt(int(fail_orb_inliers))
    )
    result["yellow_patch_hard_fail_v1"] = (
        result["yellow_patch_pair_valid_v1"]
        & result["yellow_patch_gray_corr_v1"].le(float(hard_patch_gray_corr_max))
        & result["yellow_patch_gray_absdiff_v1"].ge(float(hard_patch_gray_absdiff_min))
        & result["yellow_patch_mask_iou_v1"].le(float(hard_patch_mask_iou_max))
    )
    result["yellow_patch_extreme_fail_v1"] = (
        result["yellow_patch_pair_valid_v1"]
        & result["yellow_patch_gray_corr_v1"].le(0.45)
        & result["yellow_patch_gray_absdiff_v1"].ge(0.24)
        & result["yellow_patch_mask_iou_v1"].le(0.12)
    )
    result["yellow_patch_soft_fail_v1"] = (
        result["yellow_patch_pair_valid_v1"]
        & (
            (
                result["yellow_patch_gray_corr_v1"].le(float(soft_patch_gray_corr_max))
                & result["yellow_patch_gray_absdiff_v1"].ge(float(soft_patch_gray_absdiff_min))
            )
            | (
                result["yellow_patch_mask_iou_v1"].le(float(soft_patch_mask_iou_max))
                & result["yellow_patch_profile_corr_v1"].le(0.78)
            )
        )
    )

    result["yellow_hard_veto_v1"] = (
        (~result["yellow_pair_support_v1"])
        & (
            (
                result["yellow_orb_fail_v1"]
                & result["yellow_patch_hard_fail_v1"]
            )
            | result["yellow_patch_extreme_fail_v1"]
        )
    )
    result["yellow_soft_veto_v1"] = (
        (~result["yellow_hard_veto_v1"])
        & (~result["yellow_pair_support_v1"])
        & (
            (result["yellow_orb_fail_v1"] & result["yellow_patch_soft_fail_v1"])
            | result["yellow_patch_soft_fail_v1"]
        )
    )
    result["yellow_veto_decision_v1"] = np.select(
        [
            result["yellow_hard_veto_v1"],
            result["yellow_soft_veto_v1"],
            result["yellow_pair_support_v1"],
        ],
        [
            "hard_veto",
            "soft_veto",
            "support",
        ],
        default="unknown",
    )
    result["yellow_veto_applied_v1"] = result["yellow_veto_decision_v1"].isin(["hard_veto", "soft_veto"])
    return result


def summarize_yellow_orb_local_decisions(decision_df: pd.DataFrame) -> pd.DataFrame:
    if decision_df.empty:
        return pd.DataFrame(
            columns=[
                "yellow_veto_decision_v1",
                "pairs",
                "pair_ratio",
                "same_identity_pairs",
                "same_identity_ratio",
            ]
        )
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


def summarize_patch_pair_features(pair_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame(
            columns=[
                "pairs",
                "valid_pairs",
                "valid_ratio",
                "mean_gray_corr",
                "mean_gray_absdiff",
                "mean_mask_iou",
            ]
        )
    valid = pair_df["yellow_patch_pair_valid_v1"].fillna(False).astype(bool)
    valid_df = pair_df.loc[valid].copy()
    return pd.DataFrame(
        [
            {
                "pairs": int(len(pair_df)),
                "valid_pairs": int(len(valid_df)),
                "valid_ratio": round(float(len(valid_df) / max(len(pair_df), 1)), 6),
                "mean_gray_corr": round(float(valid_df["yellow_patch_gray_corr_v1"].mean()) if len(valid_df) else 0.0, 6),
                "mean_gray_absdiff": round(float(valid_df["yellow_patch_gray_absdiff_v1"].mean()) if len(valid_df) else 0.0, 6),
                "mean_mask_iou": round(float(valid_df["yellow_patch_mask_iou_v1"].mean()) if len(valid_df) else 0.0, 6),
            }
        ]
    )


def apply_yellow_orb_local_penalty_as_score(
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
        left_index = int(getattr(row, "left_index"))
        right_index = int(getattr(row, "right_index"))
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


def build_markdown_report(
    *,
    output_path: Path,
    config: dict[str, Any],
    roi_summary_df: pd.DataFrame,
    focus_summary_df: pd.DataFrame,
    val_patch_summary_df: pd.DataFrame,
    test_patch_summary_df: pd.DataFrame,
    val_decision_summary_df: pd.DataFrame,
    test_decision_summary_df: pd.DataFrame,
    threshold_delta_df: pd.DataFrame,
    best_rows_df: pd.DataFrame,
    test_shape_df: pd.DataFrame,
) -> None:
    lines = [
        "# Salamander Yellow ORB Local Probe v1",
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
        "## Yellow Focus Summary",
        "",
        dataframe_to_markdown_table(focus_summary_df),
        "",
        "## Val Patch Summary",
        "",
        dataframe_to_markdown_table(val_patch_summary_df),
        "",
        "## Test Patch Summary",
        "",
        dataframe_to_markdown_table(test_patch_summary_df),
        "",
        "## Val Decisions",
        "",
        dataframe_to_markdown_table(val_decision_summary_df),
        "",
        "## Test Decisions",
        "",
        dataframe_to_markdown_table(test_decision_summary_df),
        "",
        "## Val Threshold Delta",
        "",
        dataframe_to_markdown_table(threshold_delta_df),
        "",
        "## Best Rows",
        "",
        dataframe_to_markdown_table(best_rows_df),
        "",
        "## Test Shape",
        "",
        dataframe_to_markdown_table(test_shape_df),
        "",
        "## Reading Notes",
        "",
        "- 这条 probe 不再把黄色图案直接当作全局 rule veto，而是先抽出 `yellow focus ROI`，再在 ROI 上叠加 `ORB` 和 `patch match`。",
        "- `support` 只代表“局部黄纹 ROI 看起来相当一致”，主目标仍然是更稳地抓错并边，因此应优先观察 `hard_veto / soft_veto` 对 same-id 的误伤率。",
        "- 如果 `ARI` 或 `pairwise F1` 仍明显下跌，说明这版局部花纹证据依然太硬，下一步应继续降低它的 veto 权重，或者把它改成候选排序特征，而不是直接切边。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
