from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_PATCH_WIDTH = 192
DEFAULT_PATCH_HEIGHT = 96
DEFAULT_PATCH_MIN_FOREGROUND_PIXELS = 256
DEFAULT_PATCH_MIN_BLACK_PIXELS = 32
DEFAULT_PATCH_MIN_OVERLAP_PIXELS = 192


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


def _corr_or_zero(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return 0.0
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-6 or right_std <= 1e-6:
        return 0.0
    corr = np.corrcoef(left, right)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(np.clip(corr, -1.0, 1.0))


def compute_patch_descriptor(
    image: Image.Image,
    black_mask: np.ndarray,
    *,
    target_width: int = DEFAULT_PATCH_WIDTH,
    target_height: int = DEFAULT_PATCH_HEIGHT,
    min_foreground_pixels: int = DEFAULT_PATCH_MIN_FOREGROUND_PIXELS,
    min_black_pixels: int = DEFAULT_PATCH_MIN_BLACK_PIXELS,
) -> dict[str, Any]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    foreground = np.any(rgb > 0, axis=2).astype(np.uint8)
    black = (np.asarray(black_mask, dtype=np.uint8) > 0).astype(np.uint8)

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
    black_patch = (
        _resize_array_with_pad(
            black.astype(np.uint8) * 255,
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

    column_profile = black_patch.sum(axis=0).astype(np.float32)
    total = float(column_profile.sum())
    if total > 0.0:
        column_profile /= total

    row_profile = black_patch.sum(axis=1).astype(np.float32)
    row_total = float(row_profile.sum())
    if row_total > 0.0:
        row_profile /= row_total

    return {
        "patch_valid": bool(
            int(foreground_patch.sum()) >= int(min_foreground_pixels)
            and int(black_patch.sum()) >= int(min_black_pixels)
        ),
        "gray_patch": gray_patch.astype(np.float32, copy=False),
        "normalized_gray_patch": normalized_gray.astype(np.float32, copy=False),
        "foreground_patch": foreground_patch.astype(bool, copy=False),
        "black_patch": black_patch.astype(bool, copy=False),
        "black_column_profile": column_profile.astype(np.float32, copy=False),
        "black_row_profile": row_profile.astype(np.float32, copy=False),
        "foreground_pixels": int(foreground_patch.sum()),
        "black_pixels": int(black_patch.sum()),
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
            "black_patch_pair_valid_v1": False,
            "black_patch_overlap_pixels_v1": 0,
            "black_patch_gray_corr_v1": 0.0,
            "black_patch_gray_absdiff_v1": 1.0,
            "black_patch_mask_iou_v1": 0.0,
            "black_patch_mask_dice_v1": 0.0,
            "black_patch_col_profile_corr_v1": 0.0,
            "black_patch_col_profile_l1_v1": 1.0,
            "black_patch_row_profile_corr_v1": 0.0,
            "black_patch_row_profile_l1_v1": 1.0,
        }

    left_foreground = np.asarray(left_descriptor["foreground_patch"], dtype=bool)
    right_foreground = np.asarray(right_descriptor["foreground_patch"], dtype=bool)
    overlap = left_foreground & right_foreground
    overlap_pixels = int(overlap.sum())
    if overlap_pixels < int(min_overlap_pixels):
        return {
            "black_patch_pair_valid_v1": False,
            "black_patch_overlap_pixels_v1": overlap_pixels,
            "black_patch_gray_corr_v1": 0.0,
            "black_patch_gray_absdiff_v1": 1.0,
            "black_patch_mask_iou_v1": 0.0,
            "black_patch_mask_dice_v1": 0.0,
            "black_patch_col_profile_corr_v1": 0.0,
            "black_patch_col_profile_l1_v1": 1.0,
            "black_patch_row_profile_corr_v1": 0.0,
            "black_patch_row_profile_l1_v1": 1.0,
        }

    left_gray = np.asarray(left_descriptor["gray_patch"], dtype=np.float32)
    right_gray = np.asarray(right_descriptor["gray_patch"], dtype=np.float32)
    left_norm = np.asarray(left_descriptor["normalized_gray_patch"], dtype=np.float32)
    right_norm = np.asarray(right_descriptor["normalized_gray_patch"], dtype=np.float32)
    gray_corr = _corr_or_zero(left_norm[overlap], right_norm[overlap])
    gray_absdiff = float(np.mean(np.abs(left_gray[overlap] - right_gray[overlap]))) if overlap_pixels else 1.0

    left_black = np.asarray(left_descriptor["black_patch"], dtype=bool)
    right_black = np.asarray(right_descriptor["black_patch"], dtype=bool)
    black_intersection = int(np.logical_and(left_black, right_black).sum())
    black_union = int(np.logical_or(left_black, right_black).sum())
    left_black_pixels = int(left_black.sum())
    right_black_pixels = int(right_black.sum())
    mask_iou = float(black_intersection / black_union) if black_union > 0 else 0.0
    denom = max(1, left_black_pixels + right_black_pixels)
    mask_dice = float((2.0 * black_intersection) / denom)

    left_col_profile = np.asarray(left_descriptor["black_column_profile"], dtype=np.float32)
    right_col_profile = np.asarray(right_descriptor["black_column_profile"], dtype=np.float32)
    col_profile_corr = _corr_or_zero(left_col_profile, right_col_profile)
    col_profile_l1 = float(np.sum(np.abs(left_col_profile - right_col_profile)))

    left_row_profile = np.asarray(left_descriptor["black_row_profile"], dtype=np.float32)
    right_row_profile = np.asarray(right_descriptor["black_row_profile"], dtype=np.float32)
    row_profile_corr = _corr_or_zero(left_row_profile, right_row_profile)
    row_profile_l1 = float(np.sum(np.abs(left_row_profile - right_row_profile)))
    return {
        "black_patch_pair_valid_v1": True,
        "black_patch_overlap_pixels_v1": overlap_pixels,
        "black_patch_gray_corr_v1": round(float(gray_corr), 6),
        "black_patch_gray_absdiff_v1": round(float(gray_absdiff), 6),
        "black_patch_mask_iou_v1": round(float(mask_iou), 6),
        "black_patch_mask_dice_v1": round(float(mask_dice), 6),
        "black_patch_col_profile_corr_v1": round(float(col_profile_corr), 6),
        "black_patch_col_profile_l1_v1": round(float(col_profile_l1), 6),
        "black_patch_row_profile_corr_v1": round(float(row_profile_corr), 6),
        "black_patch_row_profile_l1_v1": round(float(row_profile_l1), 6),
    }


def build_patch_pair_features(
    *,
    pair_df: pd.DataFrame,
    image_feature_df: pd.DataFrame,
    repo_root: Path,
    image_path_column: str,
    mask_path_column: str,
    target_width: int = DEFAULT_PATCH_WIDTH,
    target_height: int = DEFAULT_PATCH_HEIGHT,
    min_foreground_pixels: int = DEFAULT_PATCH_MIN_FOREGROUND_PIXELS,
    min_black_pixels: int = DEFAULT_PATCH_MIN_BLACK_PIXELS,
    min_overlap_pixels: int = DEFAULT_PATCH_MIN_OVERLAP_PIXELS,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()

    repo_root = repo_root.resolve()
    feature_lookup = image_feature_df.copy().reset_index(drop=True)
    feature_lookup["image_id"] = feature_lookup["image_id"].astype(str)
    feature_lookup["dataset"] = feature_lookup["dataset"].astype(str)

    descriptor_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in feature_lookup.itertuples(index=False):
        image_id = str(row.image_id)
        dataset = str(row.dataset)
        rgb_rel = str(getattr(row, image_path_column, "") or "")
        mask_rel = str(getattr(row, mask_path_column, "") or "")
        if not rgb_rel or not mask_rel:
            descriptor_map[(image_id, dataset)] = {"patch_valid": False}
            continue
        rgb_abs = repo_root / rgb_rel
        mask_abs = repo_root / mask_rel
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
                min_black_pixels=int(min_black_pixels),
            )

    rows: list[dict[str, Any]] = []
    for row in pair_df.itertuples(index=False):
        dataset = str(getattr(row, "dataset", "TexasHornedLizards"))
        left_key = (str(row.image_id), dataset)
        right_key = (str(row.neighbor_image_id), dataset)
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
