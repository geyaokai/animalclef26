from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from .body_orientation_probe import (
    compute_body_axis,
    merge_masks,
    resolve_crop_padding_ratio,
    rotate_and_crop,
    rotation_to_horizontal,
)
from .descriptor_baselines import dataframe_to_markdown_table
from .orb_rerank_baseline import OrbFeature, build_local_match_table
from .sam_orb_veto import infer_mask_from_masked_rgb
from .sam3_probe import crop_to_union_mask, get_prompt_candidates_for_dataset, load_sam3, run_single_inference_with_prompt_backoff
from .texas_orb_local_probe import (
    DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    TEXAS_DATASET,
    align_reference_frame,
    build_texas_orb_pair_index,
    load_texas_reference_df,
    resolve_input_path,
    resolve_predictions_path,
)

try:  # pragma: no cover - exercised in wildfusion
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


DEFAULT_PAIR_CSV_PATH = Path("artifacts/analysis/texas_selftrain_review_orb_v1/tables/test_pair_disagreement_v1.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_black_pattern_orb_local_v1")
DEFAULT_ALIGNMENT_PADDING_RATIO = 0.12

BLACK_PATTERN_ALIGNED_PATH_COLUMN = "texas_black_pattern_aligned_rgb_path_v1"
BLACK_PATTERN_CORE_MASK_PATH_COLUMN = "texas_black_pattern_core_mask_path_v1"
BLACK_PATTERN_MASK_PATH_COLUMN = "texas_black_pattern_mask_path_v1"
BLACK_PATTERN_BAND_MASK_PATH_COLUMN = "texas_black_pattern_band_mask_path_v1"
BLACK_PATTERN_ORB_MASK_PATH_COLUMN = "texas_black_pattern_orb_mask_path_v1"
BLACK_PATTERN_ORB_REGION_KIND_COLUMN = "texas_black_pattern_orb_region_kind_v1"

DEFAULT_CORE_ERODE_RATIO = 0.08
DEFAULT_CORE_MIN_PIXELS = 256
DEFAULT_CORE_MIN_RADIUS = 3
DEFAULT_BLACK_QUANTILE = 0.24
DEFAULT_BLACK_MAX_QUANTILE = 0.32
DEFAULT_BLACK_MIN_PIXELS = 64
DEFAULT_BAND_DILATE_RADIUS = 7
DEFAULT_BAND_ERODE_RADIUS = 3
DEFAULT_BAND_MIN_PIXELS = 96
DEFAULT_ORB_MASK_MIN_PIXELS = 96
DEFAULT_ORB_BLACK_DILATE_RADIUS = 1

DEFAULT_NFEATURES = 2048
DEFAULT_MAX_SIDE = 768
DEFAULT_FAST_THRESHOLD = 12
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_RATIO_TEST = 0.85
DEFAULT_RANSAC_THRESHOLD = 5.0
DEFAULT_MIN_INLIERS = 4
DEFAULT_PROGRESS_EVERY = 10
DEFAULT_MASK_MORPH_MAX_SIDE = 1024
DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT = 3.0
DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE = 8
DEFAULT_BLACK_PRIOR_CONTRAST_LOW_PERCENTILE = 2.0
DEFAULT_BLACK_PRIOR_CONTRAST_HIGH_PERCENTILE = 98.0
DEFAULT_BLACK_RESPONSE_QUANTILE = 0.8
DEFAULT_BLACK_RESPONSE_RELAXED_QUANTILE = 0.7
DEFAULT_BLACK_RESPONSE_RADIUS_RATIO = 0.035
DEFAULT_BLACK_RESPONSE_MIN_RADIUS = 5
DEFAULT_ENABLE_SAM_FALLBACK = True
DEFAULT_SAM_FALLBACK_THRESHOLD = 0.35
DEFAULT_SAM_FALLBACK_MASK_THRESHOLD = 0.3
DEFAULT_SAM_FALLBACK_MIN_AREA_RATIO = 0.005
DEFAULT_SAM_FALLBACK_MAX_AREA_RATIO = 0.98
DEFAULT_SAM_FALLBACK_MIN_LARGEST_COMPONENT_RATIO = 0.4
DEFAULT_SAM_FALLBACK_DEVICE = "cuda:0"


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("Texas black-pattern ORB requires OpenCV in the active environment.")


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(np.asarray(mask, dtype=np.uint8) > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _odd_filter_size(radius: int) -> int:
    radius = max(0, int(radius))
    return max(3, radius * 2 + 1)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, mode="L").save(path)


def _apply_clahe_to_gray(
    gray: np.ndarray,
    *,
    clip_limit: float,
    grid_size: int,
) -> np.ndarray:
    gray_uint8 = np.asarray(gray, dtype=np.uint8)
    if float(clip_limit) <= 0:
        return gray_uint8
    if cv2 is not None:
        tile_size = max(2, int(grid_size))
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray_uint8)
    image = Image.fromarray(gray_uint8, mode="L")
    return np.asarray(ImageOps.autocontrast(image), dtype=np.uint8)


def _enhance_gray_inside_mask(
    gray: np.ndarray,
    focus_mask: np.ndarray,
    *,
    stats_mask: np.ndarray | None = None,
    clip_limit: float,
    grid_size: int,
    low_percentile: float = DEFAULT_BLACK_PRIOR_CONTRAST_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_BLACK_PRIOR_CONTRAST_HIGH_PERCENTILE,
) -> np.ndarray:
    gray_uint8 = np.asarray(gray, dtype=np.uint8)
    focus = (np.asarray(focus_mask, dtype=np.uint8) > 0)
    if not focus.any():
        return gray_uint8
    stats_binary = focus if stats_mask is None else (np.asarray(stats_mask, dtype=np.uint8) > 0)
    if not stats_binary.any():
        stats_binary = focus
    bbox = _bbox_from_mask(focus.astype(np.uint8))
    if bbox is None:
        return gray_uint8
    left, top, right, bottom = bbox
    crop = gray_uint8[top:bottom, left:right].copy()
    focus_crop = focus[top:bottom, left:right]
    stats_crop = stats_binary[top:bottom, left:right]
    if not stats_crop.any():
        stats_crop = focus_crop
    values = crop[stats_crop].astype(np.float32)
    if values.size <= 0:
        return gray_uint8

    neutral_value = int(np.clip(np.median(values), 0, 255))
    low = float(np.percentile(values, float(low_percentile)))
    high = float(np.percentile(values, float(high_percentile)))
    scaled_crop = np.full_like(crop, neutral_value, dtype=np.uint8)
    focus_values = crop[focus_crop].astype(np.float32)
    if high > low + 1.0:
        normalized = np.clip((focus_values - low) / max(high - low, 1e-6), 0.0, 1.0)
        scaled_crop[focus_crop] = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
    else:
        scaled_crop[focus_crop] = crop[focus_crop]
    enhanced_crop = _apply_clahe_to_gray(
        scaled_crop,
        clip_limit=float(clip_limit),
        grid_size=int(grid_size),
    )
    enhanced = gray_uint8.copy()
    enhanced_region = enhanced[top:bottom, left:right]
    enhanced_region[focus_crop] = enhanced_crop[focus_crop]
    enhanced[top:bottom, left:right] = enhanced_region
    return enhanced


def _iter_sam_fallback_attempts(
    *,
    threshold: float,
    mask_threshold: float,
) -> list[tuple[float, float]]:
    candidates = [
        (float(threshold), float(mask_threshold)),
        (min(float(threshold), 0.35), min(float(mask_threshold), 0.25)),
        (min(float(threshold), 0.25), min(float(mask_threshold), 0.20)),
    ]
    attempts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for det_threshold, seg_threshold in candidates:
        key = (round(float(det_threshold), 4), round(float(seg_threshold), 4))
        if key in seen:
            continue
        seen.add(key)
        attempts.append((float(det_threshold), float(seg_threshold)))
    return attempts


def _compute_local_darkness_response(
    gray: np.ndarray,
    focus_mask: np.ndarray,
) -> np.ndarray:
    gray_uint8 = np.asarray(gray, dtype=np.uint8)
    focus = (np.asarray(focus_mask, dtype=np.uint8) > 0).astype(np.uint8)
    bbox = _bbox_from_mask(focus)
    if bbox is None:
        return np.zeros_like(gray_uint8, dtype=np.uint8)
    left, top, right, bottom = bbox
    crop = gray_uint8[top:bottom, left:right]
    crop_height, crop_width = crop.shape[:2]
    crop_min_side = max(1, min(int(crop_width), int(crop_height)))
    radius = max(
        int(DEFAULT_BLACK_RESPONSE_MIN_RADIUS),
        int(round(float(crop_min_side) * float(DEFAULT_BLACK_RESPONSE_RADIUS_RATIO))),
    )
    if cv2 is not None:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_odd_filter_size(int(radius)), _odd_filter_size(int(radius))),
        )
        response_crop = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, kernel)
    else:
        blur_radius = max(1, int(radius // 2))
        smoothed = np.asarray(
            Image.fromarray(crop, mode="L").filter(ImageFilter.GaussianBlur(radius=blur_radius)),
            dtype=np.uint8,
        )
        response_crop = np.clip(smoothed.astype(np.int16) - crop.astype(np.int16), 0, 255).astype(np.uint8)
    response = np.zeros_like(gray_uint8, dtype=np.uint8)
    response[top:bottom, left:right] = response_crop
    return response


def _resize_rgb_image(
    image: Image.Image,
    *,
    max_side: int | None,
) -> tuple[Image.Image, float]:
    rgb = image.convert("RGB")
    if max_side is None or int(max_side) <= 0:
        return rgb, 1.0
    width, height = rgb.size
    longest = max(width, height)
    if longest <= int(max_side):
        return rgb, 1.0
    scale = float(max_side) / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_image = rgb.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    return resized_image, float(scale)


def _empty_feature(image_id: str) -> OrbFeature:
    return OrbFeature(
        image_id=str(image_id),
        matcher_name="orb",
        point_count=0,
        points=np.empty((0, 2), dtype=np.float32),
        descriptors=None,
        width=0,
        height=0,
    )


def dilate_binary_mask(binary: np.ndarray, *, radius: int) -> np.ndarray:
    mask = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(radius) <= 0:
        return (mask > 0).astype(np.uint8)
    if cv2 is not None:
        working_mask = mask
        restore_shape: tuple[int, int] | None = None
        height, width = working_mask.shape[:2]
        longest = max(height, width)
        scaled_radius = int(radius)
        if longest > int(DEFAULT_MASK_MORPH_MAX_SIDE):
            scale = float(DEFAULT_MASK_MORPH_MAX_SIDE) / float(longest)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            working_mask = cv2.resize(working_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            restore_shape = (width, height)
            scaled_radius = max(1, int(round(float(radius) * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_filter_size(int(scaled_radius)), _odd_filter_size(int(scaled_radius))))
        dilated = cv2.dilate(working_mask, kernel)
        if restore_shape is not None:
            dilated = cv2.resize(dilated, restore_shape, interpolation=cv2.INTER_NEAREST)
        return (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.uint8)
    image = Image.fromarray(mask, mode="L")
    dilated = image.filter(ImageFilter.MaxFilter(size=_odd_filter_size(int(radius))))
    return (np.asarray(dilated, dtype=np.uint8) > 0).astype(np.uint8)


def erode_binary_mask(binary: np.ndarray, *, radius: int) -> np.ndarray:
    mask = (np.asarray(binary, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if int(radius) <= 0:
        return (mask > 0).astype(np.uint8)
    if cv2 is not None:
        working_mask = mask
        restore_shape: tuple[int, int] | None = None
        height, width = working_mask.shape[:2]
        longest = max(height, width)
        scaled_radius = int(radius)
        if longest > int(DEFAULT_MASK_MORPH_MAX_SIDE):
            scale = float(DEFAULT_MASK_MORPH_MAX_SIDE) / float(longest)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            working_mask = cv2.resize(working_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
            restore_shape = (width, height)
            scaled_radius = max(1, int(round(float(radius) * scale)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd_filter_size(int(scaled_radius)), _odd_filter_size(int(scaled_radius))))
        eroded = cv2.erode(working_mask, kernel)
        if restore_shape is not None:
            eroded = cv2.resize(eroded, restore_shape, interpolation=cv2.INTER_NEAREST)
        return (np.asarray(eroded, dtype=np.uint8) > 0).astype(np.uint8)
    image = Image.fromarray(mask, mode="L")
    eroded = image.filter(ImageFilter.MinFilter(size=_odd_filter_size(int(radius))))
    return (np.asarray(eroded, dtype=np.uint8) > 0).astype(np.uint8)


def build_core_body_mask(
    foreground_mask: np.ndarray,
    *,
    erode_ratio: float = DEFAULT_CORE_ERODE_RATIO,
    min_pixels: int = DEFAULT_CORE_MIN_PIXELS,
    min_radius: int = DEFAULT_CORE_MIN_RADIUS,
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = (np.asarray(foreground_mask, dtype=np.uint8) > 0).astype(np.uint8)
    foreground_pixels = int(binary.sum())
    if foreground_pixels <= 0:
        empty = np.zeros_like(binary, dtype=np.uint8)
        return empty, {
            "core_pixels": 0,
            "core_ratio": 0.0,
            "core_erode_radius": 0,
            "core_mode": "empty",
        }
    bbox = _bbox_from_mask(binary)
    if bbox is None:
        return binary.copy(), {
            "core_pixels": foreground_pixels,
            "core_ratio": 1.0,
            "core_erode_radius": 0,
            "core_mode": "foreground",
        }
    left, top, right, bottom = bbox
    bbox_min_side = max(1, min(int(right - left), int(bottom - top)))
    base_radius = max(int(min_radius), int(round(float(bbox_min_side) * float(erode_ratio))))
    candidate_radii = [base_radius, max(1, base_radius // 2), 0]
    best_mask = binary.copy()
    chosen_radius = 0
    chosen_mode = "foreground"
    for radius in candidate_radii:
        candidate = erode_binary_mask(binary, radius=int(radius))
        if int(candidate.sum()) >= int(min_pixels):
            best_mask = candidate
            chosen_radius = int(radius)
            chosen_mode = "eroded" if int(radius) > 0 else "foreground"
            break
    core_pixels = int(best_mask.sum())
    return best_mask.astype(np.uint8), {
        "core_pixels": core_pixels,
        "core_ratio": round(float(core_pixels / max(foreground_pixels, 1)), 6),
        "core_erode_radius": int(chosen_radius),
        "core_mode": chosen_mode,
    }


def extract_black_pattern_mask(
    image: Image.Image,
    foreground_mask: np.ndarray,
    *,
    core_mask: np.ndarray | None = None,
    fallback_quantile: float = DEFAULT_BLACK_QUANTILE,
    max_quantile: float = DEFAULT_BLACK_MAX_QUANTILE,
    min_pixels: int = DEFAULT_BLACK_MIN_PIXELS,
    clahe_clip_limit: float = DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT,
    clahe_grid_size: int = DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    reference_mask = foreground_mask if core_mask is None else core_mask
    enhanced_gray = _enhance_gray_inside_mask(
        gray,
        foreground_mask,
        stats_mask=reference_mask,
        clip_limit=float(clahe_clip_limit),
        grid_size=int(clahe_grid_size),
    )
    foreground = (np.asarray(foreground_mask, dtype=np.uint8) > 0).astype(np.uint8)
    reference = foreground
    reference_name = "foreground"
    if core_mask is not None and np.asarray(core_mask, dtype=np.uint8).any():
        reference = (np.asarray(core_mask, dtype=np.uint8) > 0).astype(np.uint8)
        reference_name = "core"
    if int(reference.sum()) <= 0:
        empty = np.zeros_like(foreground, dtype=np.uint8)
        return empty, {
            "black_threshold": 0,
            "black_ratio_foreground": 0.0,
            "black_ratio_reference": 0.0,
            "black_reference_pixels": 0,
            "black_threshold_strategy": "empty",
            "black_reference_name": reference_name,
        }

    values = enhanced_gray[reference > 0]
    response_map = _compute_local_darkness_response(enhanced_gray, foreground_mask)
    response_values = response_map[reference > 0]
    response_threshold = int(np.quantile(response_values.astype(np.float32), float(DEFAULT_BLACK_RESPONSE_QUANTILE)))
    relaxed_response_threshold = int(
        np.quantile(response_values.astype(np.float32), float(DEFAULT_BLACK_RESPONSE_RELAXED_QUANTILE))
    )
    if cv2 is not None and response_values.size >= 32:
        response_otsu_threshold, _ = cv2.threshold(
            response_values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        response_threshold = int(max(float(response_threshold), float(response_otsu_threshold)))
    quantile_threshold = int(np.quantile(values.astype(np.float32), float(fallback_quantile)))
    cap_threshold = int(np.quantile(values.astype(np.float32), float(max_quantile)))
    threshold_strategy = "quantile"
    threshold = quantile_threshold
    if cv2 is not None and values.size >= 32:
        otsu_threshold, _ = cv2.threshold(
            values.reshape(-1, 1),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        threshold = int(min(float(otsu_threshold), float(cap_threshold)))
        threshold_strategy = "otsu_cap"
    black_mask = (
        (response_map >= int(response_threshold))
        & (enhanced_gray <= int(cap_threshold))
        & (reference > 0)
    ).astype(np.uint8)
    threshold_strategy = "blackhat_cap"
    if int(black_mask.sum()) < int(min_pixels):
        black_mask = (
            (response_map >= int(relaxed_response_threshold))
            & (enhanced_gray <= int(cap_threshold))
            & (reference > 0)
        ).astype(np.uint8)
        threshold = int(cap_threshold)
        threshold_strategy = "blackhat_relaxed_cap"
    if int(black_mask.sum()) < int(min_pixels):
        relaxed_threshold = int(cap_threshold)
        black_mask = ((enhanced_gray <= int(relaxed_threshold)) & (reference > 0)).astype(np.uint8)
        threshold = int(relaxed_threshold)
        threshold_strategy = "relaxed_cap"
    return black_mask.astype(np.uint8), {
        "black_threshold": int(threshold),
        "black_ratio_foreground": round(float(black_mask.sum() / max(foreground.sum(), 1)), 6),
        "black_ratio_reference": round(float(black_mask.sum() / max(reference.sum(), 1)), 6),
        "black_reference_pixels": int(reference.sum()),
        "black_threshold_strategy": threshold_strategy,
        "black_reference_name": reference_name,
        "black_prior_clahe_clip_limit": float(clahe_clip_limit),
        "black_prior_clahe_grid_size": int(clahe_grid_size),
    }


def build_black_pattern_band_mask(
    black_mask: np.ndarray,
    *,
    core_mask: np.ndarray,
    dilate_radius: int = DEFAULT_BAND_DILATE_RADIUS,
    erode_radius: int = DEFAULT_BAND_ERODE_RADIUS,
    min_pixels: int = DEFAULT_BAND_MIN_PIXELS,
) -> tuple[np.ndarray, dict[str, Any]]:
    black = (np.asarray(black_mask, dtype=np.uint8) > 0).astype(np.uint8)
    core = (np.asarray(core_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if int(black.sum()) <= 0 or int(core.sum()) <= 0:
        empty = np.zeros_like(core, dtype=np.uint8)
        return empty, {
            "band_pixels": 0,
            "band_ratio_core": 0.0,
            "band_mode": "empty",
        }
    dilated = dilate_binary_mask(black, radius=int(dilate_radius))
    eroded = erode_binary_mask(black, radius=int(erode_radius))
    band = ((dilated > 0) & (eroded == 0) & (core > 0)).astype(np.uint8)
    band_mode = "ring"
    if int(band.sum()) < int(min_pixels):
        fallback = ((dilated > 0) & (core > 0)).astype(np.uint8)
        if int(fallback.sum()) >= int(min_pixels):
            band = fallback.astype(np.uint8)
            band_mode = "dilate"
    if int(band.sum()) < int(min_pixels):
        band = ((black > 0) & (core > 0)).astype(np.uint8)
        band_mode = "black"
    return band.astype(np.uint8), {
        "band_pixels": int(band.sum()),
        "band_ratio_core": round(float(band.sum() / max(core.sum(), 1)), 6),
        "band_mode": band_mode,
    }


def _build_texas_sam_fallback_masked_rgb(
    *,
    image_path: Path,
    original_rel_path: str,
    repo_root: Path,
    output_dir: Path,
    sam_runtime: dict[str, Any],
    threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_largest_component_ratio: float,
    device: str,
) -> tuple[str, dict[str, Any]] | tuple[None, dict[str, Any]]:
    stats_payload: dict[str, Any] = {
        "fallback_status": "skip",
        "fallback_reason": "no_mask",
        "fallback_best_score": 0.0,
        "fallback_union_area_ratio": 0.0,
        "fallback_largest_component_ratio": 0.0,
        "fallback_mask_count": 0,
        "fallback_threshold_used": 0.0,
        "fallback_mask_threshold_used": 0.0,
    }
    if not image_path.exists():
        stats_payload["fallback_reason"] = "image_missing"
        return None, stats_payload
    resources = sam_runtime.get("resources")
    if resources is None:
        try:
            resources = load_sam3(device=device)
            sam_runtime["resources"] = resources
        except Exception as exc:
            message = str(exc).lower()
            if "out of memory" not in message or str(device).lower() == "cpu":
                raise
            print(
                f"[texas_black_pattern_orb_local] sam fallback device {device} OOM, retry on cpu",
                flush=True,
            )
            resources = sam_runtime.get("resources_cpu")
            if resources is None:
                resources = load_sam3(device="cpu")
                sam_runtime["resources_cpu"] = resources
            sam_runtime["resources"] = resources
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        final_reason = "no_mask"
        masked_crop: Image.Image | None = None
        for attempt_index, (attempt_threshold, attempt_mask_threshold) in enumerate(
            _iter_sam_fallback_attempts(threshold=float(threshold), mask_threshold=float(mask_threshold)),
            start=1,
        ):
            masks, stats = run_single_inference_with_prompt_backoff(
                image=image,
                prompts=get_prompt_candidates_for_dataset("TexasHornedLizards"),
                resources=resources,
                threshold=float(attempt_threshold),
                mask_threshold=float(attempt_mask_threshold),
            )
            stats_payload["fallback_mask_count"] = int(stats.get("mask_count", 0))
            stats_payload["fallback_best_score"] = float(stats.get("best_score", 0.0))
            stats_payload["fallback_threshold_used"] = float(attempt_threshold)
            stats_payload["fallback_mask_threshold_used"] = float(attempt_mask_threshold)
            if masks is None:
                final_reason = f"no_mask_try_{attempt_index}"
                continue
            component_mask, component_stats = merge_masks(masks)
            union_area_ratio = float(component_stats["union_area_ratio"])
            largest_component_ratio = float(component_stats["largest_component_ratio"])
            stats_payload["fallback_union_area_ratio"] = union_area_ratio
            stats_payload["fallback_largest_component_ratio"] = largest_component_ratio
            if union_area_ratio < float(min_area_ratio):
                final_reason = f"small_area_ratio_try_{attempt_index}"
                continue
            if union_area_ratio > float(max_area_ratio):
                final_reason = f"large_area_ratio_try_{attempt_index}"
                continue
            if largest_component_ratio < float(min_largest_component_ratio):
                final_reason = f"fragmented_mask_try_{attempt_index}"
                continue
            masked_crop = crop_to_union_mask(image, component_mask[None, ...])
            final_reason = "ok"
            break
        if masked_crop is None:
            stats_payload["fallback_reason"] = final_reason
            return None, stats_payload
    export_rel = (
        output_dir.relative_to(repo_root)
        / "views"
        / "texas_black_pattern_orb_local_v1"
        / "sam_fallback"
        / _resolve_relative_view_path(original_rel_path, suffix=".jpg")
    )
    export_abs = repo_root / export_rel
    export_abs.parent.mkdir(parents=True, exist_ok=True)
    masked_crop.save(export_abs, quality=95)
    stats_payload["fallback_status"] = "apply"
    stats_payload["fallback_reason"] = "ok"
    return export_rel.as_posix(), stats_payload


def _resolve_relative_view_path(original_path: str, *, suffix: str) -> Path:
    image_relative = Path(str(original_path)).relative_to("images")
    if str(suffix).startswith("."):
        return image_relative.with_suffix(str(suffix))
    return image_relative.with_name(f"{image_relative.stem}{suffix}")


def build_texas_black_pattern_view_manifest(
    *,
    reference_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
    view_manifest_path: Path,
    alignment_padding_ratio: float = DEFAULT_ALIGNMENT_PADDING_RATIO,
    core_erode_ratio: float = DEFAULT_CORE_ERODE_RATIO,
    core_min_pixels: int = DEFAULT_CORE_MIN_PIXELS,
    black_quantile: float = DEFAULT_BLACK_QUANTILE,
    black_max_quantile: float = DEFAULT_BLACK_MAX_QUANTILE,
    black_min_pixels: int = DEFAULT_BLACK_MIN_PIXELS,
    band_dilate_radius: int = DEFAULT_BAND_DILATE_RADIUS,
    band_erode_radius: int = DEFAULT_BAND_ERODE_RADIUS,
    band_min_pixels: int = DEFAULT_BAND_MIN_PIXELS,
    orb_mask_min_pixels: int = DEFAULT_ORB_MASK_MIN_PIXELS,
    orb_black_dilate_radius: int = DEFAULT_ORB_BLACK_DILATE_RADIUS,
    processing_max_side: int | None = None,
    black_prior_clahe_clip_limit: float = DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT,
    black_prior_clahe_grid_size: int = DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE,
    enable_sam_fallback: bool = DEFAULT_ENABLE_SAM_FALLBACK,
    sam_fallback_threshold: float = DEFAULT_SAM_FALLBACK_THRESHOLD,
    sam_fallback_mask_threshold: float = DEFAULT_SAM_FALLBACK_MASK_THRESHOLD,
    sam_fallback_min_area_ratio: float = DEFAULT_SAM_FALLBACK_MIN_AREA_RATIO,
    sam_fallback_max_area_ratio: float = DEFAULT_SAM_FALLBACK_MAX_AREA_RATIO,
    sam_fallback_min_largest_component_ratio: float = DEFAULT_SAM_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
    sam_fallback_device: str = DEFAULT_SAM_FALLBACK_DEVICE,
) -> pd.DataFrame:
    manifest_df = pd.read_csv(view_manifest_path).copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    subset = manifest_df[manifest_df["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    aligned_manifest = align_reference_frame(reference_df=reference_df, candidate_df=subset, name=view_manifest_path.name)

    views_root = output_dir / "views" / "texas_black_pattern_orb_local_v1"
    aligned_dir = views_root / "aligned"
    mask_dir = views_root / "masks"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    resolved_padding_ratio = resolve_crop_padding_ratio(
        TEXAS_DATASET,
        default_padding_ratio=float(alignment_padding_ratio),
    )
    total_images = int(len(aligned_manifest))
    print(
        f"[texas_black_pattern_orb_local] build_view_manifest start: images={total_images}, processing_max_side={0 if processing_max_side is None else int(processing_max_side)}",
        flush=True,
    )
    sam_runtime: dict[str, Any] = {}

    for index, row in enumerate(aligned_manifest.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        original_path = str(getattr(row, "path", ""))
        sam_path = ""
        if bool(getattr(row, "sam_masked_rgb_v1_applied", False)):
            sam_path = str(getattr(row, "sam_masked_rgb_v1_export_path", "") or "")
        payload: dict[str, Any] = {
            "image_id": image_id,
            "dataset": str(getattr(row, "dataset", "")),
            "split": str(getattr(row, "split", "")),
            "identity": "" if pd.isna(getattr(row, "identity", "")) else str(getattr(row, "identity", "")),
            "path": original_path,
            "sam_masked_rgb_path_v1": sam_path,
            "texas_black_pattern_sam_available_v1": False,
            "texas_black_pattern_sam_source_v1": "missing",
            "texas_black_pattern_sam_reason_v1": "missing_sam",
            "texas_black_pattern_axis_angle_deg_v1": 0.0,
            "texas_black_pattern_axis_confidence_v1": 0.0,
            "texas_black_pattern_rotation_applied_deg_v1": 0.0,
            "texas_black_pattern_alignment_status_v1": "missing_sam",
            "texas_black_pattern_foreground_pixels_v1": 0,
            "texas_black_pattern_foreground_ratio_v1": 0.0,
            "texas_black_pattern_processing_scale_v1": 1.0,
            BLACK_PATTERN_ALIGNED_PATH_COLUMN: "",
            BLACK_PATTERN_CORE_MASK_PATH_COLUMN: "",
            BLACK_PATTERN_MASK_PATH_COLUMN: "",
            BLACK_PATTERN_BAND_MASK_PATH_COLUMN: "",
            BLACK_PATTERN_ORB_MASK_PATH_COLUMN: "",
            BLACK_PATTERN_ORB_REGION_KIND_COLUMN: "none",
            "texas_black_pattern_core_pixels_v1": 0,
            "texas_black_pattern_core_ratio_v1": 0.0,
            "texas_black_pattern_core_erode_radius_v1": 0,
            "texas_black_pattern_core_mode_v1": "empty",
            "texas_black_pattern_black_pixels_v1": 0,
            "texas_black_pattern_black_ratio_foreground_v1": 0.0,
            "texas_black_pattern_black_ratio_reference_v1": 0.0,
            "texas_black_pattern_black_threshold_v1": 0,
            "texas_black_pattern_black_threshold_strategy_v1": "empty",
            "texas_black_pattern_black_prior_clahe_clip_limit_v1": float(black_prior_clahe_clip_limit),
            "texas_black_pattern_black_prior_clahe_grid_size_v1": int(black_prior_clahe_grid_size),
            "texas_black_pattern_band_pixels_v1": 0,
            "texas_black_pattern_band_ratio_core_v1": 0.0,
            "texas_black_pattern_band_mode_v1": "empty",
            "texas_black_pattern_orb_mask_pixels_v1": 0,
        }
        sam_abs_path: Path | None = None
        if sam_path:
            candidate_sam_abs_path = repo_root / sam_path
            if candidate_sam_abs_path.exists():
                sam_abs_path = candidate_sam_abs_path
                payload["texas_black_pattern_sam_source_v1"] = "manifest"
                payload["texas_black_pattern_sam_reason_v1"] = "ok"
            else:
                payload["texas_black_pattern_alignment_status_v1"] = "sam_path_missing"
                payload["texas_black_pattern_sam_reason_v1"] = "sam_path_missing"
        if sam_abs_path is None and bool(enable_sam_fallback):
            fallback_rel, fallback_payload = _build_texas_sam_fallback_masked_rgb(
                image_path=repo_root / original_path,
                original_rel_path=original_path,
                repo_root=repo_root,
                output_dir=output_dir,
                sam_runtime=sam_runtime,
                threshold=float(sam_fallback_threshold),
                mask_threshold=float(sam_fallback_mask_threshold),
                min_area_ratio=float(sam_fallback_min_area_ratio),
                max_area_ratio=float(sam_fallback_max_area_ratio),
                min_largest_component_ratio=float(sam_fallback_min_largest_component_ratio),
                device=str(sam_fallback_device),
            )
            payload["texas_black_pattern_sam_reason_v1"] = str(fallback_payload["fallback_reason"])
            if fallback_rel:
                sam_path = str(fallback_rel)
                sam_abs_path = repo_root / sam_path
                payload["sam_masked_rgb_path_v1"] = sam_path
                payload["texas_black_pattern_sam_source_v1"] = "fallback"
                payload["texas_black_pattern_alignment_status_v1"] = "apply"
        if sam_abs_path is None:
            rows.append(payload)
            continue

        with Image.open(sam_abs_path) as sam_image_handle:
            sam_image = sam_image_handle.convert("RGB")
        sam_image, processing_scale = _resize_rgb_image(
            sam_image,
            max_side=processing_max_side,
        )
        foreground_mask = infer_mask_from_masked_rgb(sam_image)
        payload["texas_black_pattern_sam_available_v1"] = True
        payload["texas_black_pattern_foreground_pixels_v1"] = int(foreground_mask.sum())
        payload["texas_black_pattern_foreground_ratio_v1"] = round(
            float(foreground_mask.sum() / max(foreground_mask.size, 1)),
            6,
        )
        payload["texas_black_pattern_processing_scale_v1"] = round(float(processing_scale), 6)

        axis_stats = compute_body_axis(foreground_mask)
        if axis_stats is None:
            rotation_deg = 0.0
            payload["texas_black_pattern_alignment_status_v1"] = "keep_masked_no_axis"
            aligned_rgb = sam_image.copy()
            aligned_mask = foreground_mask.astype(np.uint8, copy=False)
        else:
            rotation_deg = rotation_to_horizontal(float(axis_stats["axis_angle_deg"]))
            payload["texas_black_pattern_alignment_status_v1"] = "apply"
            payload["texas_black_pattern_axis_angle_deg_v1"] = round(float(axis_stats["axis_angle_deg"]), 6)
            payload["texas_black_pattern_axis_confidence_v1"] = round(float(axis_stats["axis_confidence"]), 6)
            aligned_rgb, aligned_mask = rotate_and_crop(
                sam_image,
                foreground_mask,
                rotation_deg,
                background=(0, 0, 0),
                padding_ratio=float(resolved_padding_ratio),
                keep_background=False,
                canvas_fill_mode="constant",
            )
        payload["texas_black_pattern_rotation_applied_deg_v1"] = round(float(rotation_deg), 6)

        core_mask, core_payload = build_core_body_mask(
            aligned_mask,
            erode_ratio=float(core_erode_ratio),
            min_pixels=int(core_min_pixels),
        )
        black_mask, black_payload = extract_black_pattern_mask(
            aligned_rgb,
            aligned_mask,
            core_mask=core_mask,
            fallback_quantile=float(black_quantile),
            max_quantile=float(black_max_quantile),
            min_pixels=int(black_min_pixels),
            clahe_clip_limit=float(black_prior_clahe_clip_limit),
            clahe_grid_size=int(black_prior_clahe_grid_size),
        )
        band_mask, band_payload = build_black_pattern_band_mask(
            black_mask,
            core_mask=core_mask,
            dilate_radius=int(band_dilate_radius),
            erode_radius=int(band_erode_radius),
            min_pixels=int(band_min_pixels),
        )
        orb_black_mask = dilate_binary_mask(black_mask, radius=int(orb_black_dilate_radius))
        orb_black_mask = ((orb_black_mask > 0) & (core_mask > 0)).astype(np.uint8)
        if int(orb_black_mask.sum()) >= int(orb_mask_min_pixels):
            orb_mask = orb_black_mask
            orb_region_kind = "black"
        elif int(black_mask.sum()) > 0:
            orb_mask = black_mask
            orb_region_kind = "black_sparse"
        else:
            orb_mask = np.zeros_like(core_mask, dtype=np.uint8)
            orb_region_kind = "none"

        aligned_rel = output_dir.relative_to(repo_root) / "views" / "texas_black_pattern_orb_local_v1" / "aligned" / _resolve_relative_view_path(original_path, suffix=".jpg")
        core_rel = output_dir.relative_to(repo_root) / "views" / "texas_black_pattern_orb_local_v1" / "masks" / _resolve_relative_view_path(original_path, suffix="_core.png")
        black_rel = output_dir.relative_to(repo_root) / "views" / "texas_black_pattern_orb_local_v1" / "masks" / _resolve_relative_view_path(original_path, suffix="_black.png")
        band_rel = output_dir.relative_to(repo_root) / "views" / "texas_black_pattern_orb_local_v1" / "masks" / _resolve_relative_view_path(original_path, suffix="_band.png")
        orb_rel = output_dir.relative_to(repo_root) / "views" / "texas_black_pattern_orb_local_v1" / "masks" / _resolve_relative_view_path(original_path, suffix="_orb.png")

        aligned_abs = repo_root / aligned_rel
        core_abs = repo_root / core_rel
        black_abs = repo_root / black_rel
        band_abs = repo_root / band_rel
        orb_abs = repo_root / orb_rel
        aligned_abs.parent.mkdir(parents=True, exist_ok=True)
        aligned_rgb.save(aligned_abs, quality=95)
        _save_mask(core_mask, core_abs)
        _save_mask(black_mask, black_abs)
        _save_mask(band_mask, band_abs)
        _save_mask(orb_mask, orb_abs)

        payload.update(
            {
                "texas_black_pattern_sam_available_v1": True,
                BLACK_PATTERN_ALIGNED_PATH_COLUMN: aligned_rel.as_posix(),
                BLACK_PATTERN_CORE_MASK_PATH_COLUMN: core_rel.as_posix(),
                BLACK_PATTERN_MASK_PATH_COLUMN: black_rel.as_posix(),
                BLACK_PATTERN_BAND_MASK_PATH_COLUMN: band_rel.as_posix(),
                BLACK_PATTERN_ORB_MASK_PATH_COLUMN: orb_rel.as_posix(),
                BLACK_PATTERN_ORB_REGION_KIND_COLUMN: orb_region_kind,
                "texas_black_pattern_core_pixels_v1": int(core_payload["core_pixels"]),
                "texas_black_pattern_core_ratio_v1": float(core_payload["core_ratio"]),
                "texas_black_pattern_core_erode_radius_v1": int(core_payload["core_erode_radius"]),
                "texas_black_pattern_core_mode_v1": str(core_payload["core_mode"]),
                "texas_black_pattern_black_pixels_v1": int(black_mask.sum()),
                "texas_black_pattern_black_ratio_foreground_v1": float(black_payload["black_ratio_foreground"]),
                "texas_black_pattern_black_ratio_reference_v1": float(black_payload["black_ratio_reference"]),
                "texas_black_pattern_black_threshold_v1": int(black_payload["black_threshold"]),
                "texas_black_pattern_black_threshold_strategy_v1": str(black_payload["black_threshold_strategy"]),
                "texas_black_pattern_black_prior_clahe_clip_limit_v1": float(black_payload["black_prior_clahe_clip_limit"]),
                "texas_black_pattern_black_prior_clahe_grid_size_v1": int(black_payload["black_prior_clahe_grid_size"]),
                "texas_black_pattern_band_pixels_v1": int(band_payload["band_pixels"]),
                "texas_black_pattern_band_ratio_core_v1": float(band_payload["band_ratio_core"]),
                "texas_black_pattern_band_mode_v1": str(band_payload["band_mode"]),
                "texas_black_pattern_orb_mask_pixels_v1": int(orb_mask.sum()),
            }
        )
        rows.append(payload)
        if index % DEFAULT_PROGRESS_EVERY == 0 or index == total_images:
            print(
                f"[texas_black_pattern_orb_local] build_view_manifest progress: {index}/{total_images} images",
                flush=True,
            )
    return pd.DataFrame(rows).reset_index(drop=True)


def _load_grayscale_and_mask(
    image_path: Path,
    mask_path: Path | None,
    *,
    max_side: int,
    clahe_clip_limit: float,
) -> tuple[np.ndarray, np.ndarray | None, int, int]:
    _require_cv2()
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read aligned Texas image: {image_path}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path is not None else None
    if mask is not None and mask.shape[:2] != gray.shape[:2]:
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    height, width = gray.shape[:2]
    if max(height, width) > int(max_side):
        scale = float(max_side) / float(max(height, width))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        if mask is not None:
            mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        width, height = resized_width, resized_height
    if mask is not None:
        mask = (mask > 0).astype(np.uint8) * 255
        gray = _enhance_gray_inside_mask(
            gray,
            mask,
            stats_mask=mask,
            clip_limit=float(clahe_clip_limit),
            grid_size=8,
        )
    elif float(clahe_clip_limit) > 0:
        gray = _apply_clahe_to_gray(gray, clip_limit=float(clahe_clip_limit), grid_size=8)
    return gray, mask, int(width), int(height)


def extract_texas_black_pattern_orb_features(
    *,
    view_df: pd.DataFrame,
    repo_root: Path,
    nfeatures: int = DEFAULT_NFEATURES,
    max_side: int = DEFAULT_MAX_SIDE,
    fast_threshold: int = DEFAULT_FAST_THRESHOLD,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
) -> list[OrbFeature]:
    _require_cv2()
    detector = cv2.ORB_create(nfeatures=int(nfeatures), fastThreshold=int(fast_threshold))
    features: list[OrbFeature] = []
    total_images = int(len(view_df))
    print(
        f"[texas_black_pattern_orb_local] extract_features start: images={total_images}, max_side={int(max_side)}, nfeatures={int(nfeatures)}",
        flush=True,
    )
    for index, row in enumerate(view_df.itertuples(index=False), start=1):
        image_id = str(row.image_id)
        aligned_rel = str(getattr(row, BLACK_PATTERN_ALIGNED_PATH_COLUMN, "") or "")
        orb_mask_rel = str(getattr(row, BLACK_PATTERN_ORB_MASK_PATH_COLUMN, "") or "")
        if not aligned_rel or not orb_mask_rel:
            features.append(_empty_feature(image_id))
            continue
        aligned_abs = repo_root / aligned_rel
        orb_mask_abs = repo_root / orb_mask_rel
        if not aligned_abs.exists() or not orb_mask_abs.exists():
            features.append(_empty_feature(image_id))
            continue
        gray, detect_mask, width, height = _load_grayscale_and_mask(
            aligned_abs,
            orb_mask_abs,
            max_side=int(max_side),
            clahe_clip_limit=float(clahe_clip_limit),
        )
        if detect_mask is None or int((detect_mask > 0).sum()) <= 0:
            features.append(_empty_feature(image_id))
            continue
        keypoints, descriptors = detector.detectAndCompute(gray, detect_mask)
        if keypoints:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.empty((0, 2), dtype=np.float32)
            descriptors = None
        features.append(
            OrbFeature(
                image_id=image_id,
                matcher_name="orb",
                point_count=int(len(points)),
                points=points,
                descriptors=descriptors,
                width=int(width),
                height=int(height),
            )
        )
        if index % DEFAULT_PROGRESS_EVERY == 0 or index == total_images:
            print(
                f"[texas_black_pattern_orb_local] extract_features progress: {index}/{total_images} images",
                flush=True,
            )
    return features


def _enrich_local_table(
    *,
    local_df: pd.DataFrame,
    view_df: pd.DataFrame,
    score_column: str,
) -> pd.DataFrame:
    if local_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "left_index",
                "right_index",
                "image_id",
                "neighbor_image_id",
                "pair_score",
                "pair_score_col",
                "black_orb_matcher_name",
                "black_orb_left_keypoints",
                "black_orb_right_keypoints",
                "black_orb_good_matches",
                "black_orb_inliers",
                "black_orb_local_raw_score",
                "black_orb_local_score",
            ]
        )
    renamed = local_df.rename(
        columns={
            "global_score": "pair_score",
            "matcher_name": "black_orb_matcher_name",
            "left_keypoints": "black_orb_left_keypoints",
            "right_keypoints": "black_orb_right_keypoints",
            "good_matches": "black_orb_good_matches",
            "inliers": "black_orb_inliers",
            "local_raw_score": "black_orb_local_raw_score",
            "local_score": "black_orb_local_score",
        }
    ).copy()
    renamed["pair_score_col"] = str(score_column)

    image_lookup = view_df.set_index("image_id", drop=False)
    renamed["left_black_ratio"] = [
        float(image_lookup.at[str(image_id), "texas_black_pattern_black_ratio_reference_v1"])
        for image_id in renamed["image_id"].astype(str)
    ]
    renamed["right_black_ratio"] = [
        float(image_lookup.at[str(image_id), "texas_black_pattern_black_ratio_reference_v1"])
        for image_id in renamed["neighbor_image_id"].astype(str)
    ]
    renamed["left_orb_region_kind"] = [
        str(image_lookup.at[str(image_id), BLACK_PATTERN_ORB_REGION_KIND_COLUMN])
        for image_id in renamed["image_id"].astype(str)
    ]
    renamed["right_orb_region_kind"] = [
        str(image_lookup.at[str(image_id), BLACK_PATTERN_ORB_REGION_KIND_COLUMN])
        for image_id in renamed["neighbor_image_id"].astype(str)
    ]
    return renamed


def merge_texas_black_pattern_orb_local_scores(
    pair_df: pd.DataFrame,
    local_match_df: pd.DataFrame,
    *,
    override_local_score: bool = False,
) -> pd.DataFrame:
    result = pair_df.copy().reset_index(drop=True)
    if local_match_df.empty:
        return result
    local = local_match_df.copy().reset_index(drop=True)
    for column in ["left_index", "right_index"]:
        if column in local.columns:
            local[column] = pd.to_numeric(local[column], errors="raise").astype(int)
    for column in ["image_id", "neighbor_image_id"]:
        if column in local.columns:
            local[column] = local[column].astype(str)
    keep_columns = [
        "left_index",
        "right_index",
        "image_id",
        "neighbor_image_id",
        "pair_score",
        "pair_score_col",
        "black_orb_matcher_name",
        "black_orb_left_keypoints",
        "black_orb_right_keypoints",
        "black_orb_good_matches",
        "black_orb_inliers",
        "black_orb_local_raw_score",
        "black_orb_local_score",
        "left_black_ratio",
        "right_black_ratio",
        "left_orb_region_kind",
        "right_orb_region_kind",
    ]
    keep_columns = [column for column in keep_columns if column in local.columns]
    local = (
        local.loc[:, keep_columns]
        .sort_values(["black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"], ascending=[False, False, False])
        .drop_duplicates(subset=["left_index", "right_index", "image_id", "neighbor_image_id"], keep="first")
        .reset_index(drop=True)
    )
    merged = result.merge(
        local,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )
    if override_local_score and "black_orb_local_score" in merged.columns:
        merged["local_score"] = pd.to_numeric(merged["black_orb_local_score"], errors="coerce").where(
            pd.to_numeric(merged["black_orb_local_score"], errors="coerce").notna(),
            pd.to_numeric(merged.get("local_score"), errors="coerce"),
        )
    return merged


def run_texas_black_pattern_orb_local_probe(
    *,
    repo_root: Path,
    predictions_path: Path,
    pair_csv_path: Path,
    output_dir: Path,
    view_manifest_path: Path = DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    score_column: str | None = None,
    nfeatures: int = DEFAULT_NFEATURES,
    max_side: int = DEFAULT_MAX_SIDE,
    fast_threshold: int = DEFAULT_FAST_THRESHOLD,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    ratio_test: float = DEFAULT_RATIO_TEST,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    core_erode_ratio: float = DEFAULT_CORE_ERODE_RATIO,
    core_min_pixels: int = DEFAULT_CORE_MIN_PIXELS,
    black_quantile: float = DEFAULT_BLACK_QUANTILE,
    black_max_quantile: float = DEFAULT_BLACK_MAX_QUANTILE,
    black_min_pixels: int = DEFAULT_BLACK_MIN_PIXELS,
    band_dilate_radius: int = DEFAULT_BAND_DILATE_RADIUS,
    band_erode_radius: int = DEFAULT_BAND_ERODE_RADIUS,
    band_min_pixels: int = DEFAULT_BAND_MIN_PIXELS,
    orb_mask_min_pixels: int = DEFAULT_ORB_MASK_MIN_PIXELS,
    orb_black_dilate_radius: int = DEFAULT_ORB_BLACK_DILATE_RADIUS,
    processing_max_side: int | None = None,
    black_prior_clahe_clip_limit: float = DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT,
    black_prior_clahe_grid_size: int = DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE,
    enable_sam_fallback: bool = DEFAULT_ENABLE_SAM_FALLBACK,
    sam_fallback_threshold: float = DEFAULT_SAM_FALLBACK_THRESHOLD,
    sam_fallback_mask_threshold: float = DEFAULT_SAM_FALLBACK_MASK_THRESHOLD,
    sam_fallback_min_area_ratio: float = DEFAULT_SAM_FALLBACK_MIN_AREA_RATIO,
    sam_fallback_max_area_ratio: float = DEFAULT_SAM_FALLBACK_MAX_AREA_RATIO,
    sam_fallback_min_largest_component_ratio: float = DEFAULT_SAM_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
    sam_fallback_device: str = DEFAULT_SAM_FALLBACK_DEVICE,
) -> dict[str, Path]:
    resolved_predictions_path = resolve_predictions_path(repo_root=repo_root, value=predictions_path)
    resolved_pair_csv_path = resolve_input_path(repo_root=repo_root, value=pair_csv_path)
    resolved_output_dir = resolve_input_path(repo_root=repo_root, value=output_dir)
    resolved_view_manifest_path = resolve_input_path(repo_root=repo_root, value=view_manifest_path)

    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    reference_df = load_texas_reference_df(resolved_predictions_path)
    raw_pair_df = pd.read_csv(resolved_pair_csv_path)
    pair_index, normalized_pair_df, resolved_score_column = build_texas_orb_pair_index(
        pair_df=raw_pair_df,
        reference_df=reference_df,
        score_column=score_column,
    )
    print(
        f"[texas_black_pattern_orb_local] pair_index ready: pairs={len(pair_index)}, images={len(reference_df)}",
        flush=True,
    )
    view_df = build_texas_black_pattern_view_manifest(
        reference_df=reference_df,
        repo_root=repo_root,
        output_dir=resolved_output_dir,
        view_manifest_path=resolved_view_manifest_path,
        core_erode_ratio=float(core_erode_ratio),
        core_min_pixels=int(core_min_pixels),
        black_quantile=float(black_quantile),
        black_max_quantile=float(black_max_quantile),
        black_min_pixels=int(black_min_pixels),
        band_dilate_radius=int(band_dilate_radius),
        band_erode_radius=int(band_erode_radius),
        band_min_pixels=int(band_min_pixels),
        orb_mask_min_pixels=int(orb_mask_min_pixels),
        orb_black_dilate_radius=int(orb_black_dilate_radius),
        processing_max_side=(None if processing_max_side is None else int(processing_max_side)),
        black_prior_clahe_clip_limit=float(black_prior_clahe_clip_limit),
        black_prior_clahe_grid_size=int(black_prior_clahe_grid_size),
        enable_sam_fallback=bool(enable_sam_fallback),
        sam_fallback_threshold=float(sam_fallback_threshold),
        sam_fallback_mask_threshold=float(sam_fallback_mask_threshold),
        sam_fallback_min_area_ratio=float(sam_fallback_min_area_ratio),
        sam_fallback_max_area_ratio=float(sam_fallback_max_area_ratio),
        sam_fallback_min_largest_component_ratio=float(sam_fallback_min_largest_component_ratio),
        sam_fallback_device=str(sam_fallback_device),
    )
    print(
        f"[texas_black_pattern_orb_local] view_manifest done: rows={len(view_df)}",
        flush=True,
    )
    features = extract_texas_black_pattern_orb_features(
        view_df=view_df,
        repo_root=repo_root,
        nfeatures=int(nfeatures),
        max_side=int(max_side),
        fast_threshold=int(fast_threshold),
        clahe_clip_limit=float(clahe_clip_limit),
    )
    print(
        f"[texas_black_pattern_orb_local] extract_features done: valid={sum(int(feature.point_count > 0) for feature in features)}",
        flush=True,
    )
    view_df["texas_black_pattern_orb_keypoints_v1"] = [int(feature.point_count) for feature in features]
    print(
        f"[texas_black_pattern_orb_local] build_local_match_table start: pairs={len(pair_index)}",
        flush=True,
    )
    local_df = build_local_match_table(
        df=view_df,
        features=features,
        pair_index=pair_index,
        ratio_test=float(ratio_test),
        ransac_threshold=float(ransac_threshold),
        min_inliers=int(min_inliers),
        local_matcher="orb",
    )
    print(
        f"[texas_black_pattern_orb_local] build_local_match_table done: rows={len(local_df)}",
        flush=True,
    )
    enriched_local_df = _enrich_local_table(local_df=local_df, view_df=view_df, score_column=resolved_score_column)

    local_table_path = tables_dir / "test_pair_local_scores_v1.csv"
    normalized_pair_path = tables_dir / "normalized_pairs_v1.csv"
    image_stats_path = tables_dir / "image_feature_stats_v1.csv"
    enriched_local_df.to_csv(local_table_path, index=False)
    normalized_pair_df.to_csv(normalized_pair_path, index=False)
    view_df.to_csv(image_stats_path, index=False)

    nonzero_pair_count = int(enriched_local_df["black_orb_local_score"].fillna(0.0).astype(float).gt(0.0).sum()) if not enriched_local_df.empty else 0
    orb_image_valid_count = int(view_df["texas_black_pattern_orb_keypoints_v1"].astype(int).gt(0).sum()) if not view_df.empty else 0
    sam_available_ratio = float(view_df["texas_black_pattern_sam_available_v1"].fillna(False).astype(bool).mean()) if not view_df.empty else 0.0
    sam_fallback_count = int(view_df["texas_black_pattern_sam_source_v1"].astype(str).eq("fallback").sum()) if not view_df.empty else 0
    mean_black_ratio = float(view_df["texas_black_pattern_black_ratio_reference_v1"].fillna(0.0).astype(float).mean()) if not view_df.empty else 0.0
    mean_keypoints = float(view_df["texas_black_pattern_orb_keypoints_v1"].fillna(0).astype(float).mean()) if not view_df.empty else 0.0
    summary = {
        "probe": resolved_output_dir.name,
        "dataset": TEXAS_DATASET,
        "predictions_path": _path_ref(repo_root, resolved_predictions_path),
        "pair_csv_path": _path_ref(repo_root, resolved_pair_csv_path),
        "view_manifest_path": _path_ref(repo_root, resolved_view_manifest_path),
        "pair_score_col": resolved_score_column,
        "image_count": int(len(view_df)),
        "pair_count": int(len(enriched_local_df)),
        "sam_available_ratio": round(float(sam_available_ratio), 6),
        "sam_fallback_count": int(sam_fallback_count),
        "orb_image_valid_count": int(orb_image_valid_count),
        "orb_image_valid_ratio": round(float(orb_image_valid_count / max(len(view_df), 1)), 6),
        "mean_black_ratio": round(float(mean_black_ratio), 6),
        "mean_keypoints": round(float(mean_keypoints), 6),
        "nonzero_local_pair_count": int(nonzero_pair_count),
        "mean_black_orb_local_score": round(float(enriched_local_df["black_orb_local_score"].astype(float).mean()) if not enriched_local_df.empty else 0.0, 6),
        "max_black_orb_local_score": round(float(enriched_local_df["black_orb_local_score"].astype(float).max()) if not enriched_local_df.empty else 0.0, 6),
        "mean_black_orb_inliers": round(float(enriched_local_df["black_orb_inliers"].astype(float).mean()) if not enriched_local_df.empty else 0.0, 6),
        "nfeatures": int(nfeatures),
        "max_side": int(max_side),
        "fast_threshold": int(fast_threshold),
        "clahe_clip_limit": float(clahe_clip_limit),
        "ratio_test": float(ratio_test),
        "ransac_threshold": float(ransac_threshold),
        "min_inliers": int(min_inliers),
        "core_erode_ratio": float(core_erode_ratio),
        "black_quantile": float(black_quantile),
        "black_max_quantile": float(black_max_quantile),
        "band_dilate_radius": int(band_dilate_radius),
        "band_erode_radius": int(band_erode_radius),
        "orb_black_dilate_radius": int(orb_black_dilate_radius),
        "processing_max_side": 0 if processing_max_side is None else int(processing_max_side),
        "black_prior_clahe_clip_limit": float(black_prior_clahe_clip_limit),
        "black_prior_clahe_grid_size": int(black_prior_clahe_grid_size),
        "black_prior_contrast_low_percentile": float(DEFAULT_BLACK_PRIOR_CONTRAST_LOW_PERCENTILE),
        "black_prior_contrast_high_percentile": float(DEFAULT_BLACK_PRIOR_CONTRAST_HIGH_PERCENTILE),
        "enable_sam_fallback": bool(enable_sam_fallback),
        "sam_fallback_threshold": float(sam_fallback_threshold),
        "sam_fallback_mask_threshold": float(sam_fallback_mask_threshold),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_pairs_df = (
        enriched_local_df.sort_values(["black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"], ascending=[False, False, False])
        .head(12)
        .reset_index(drop=True)
    )
    summary_md = "\n".join(
        [
            "# Texas Black Pattern ORB Local Probe v1",
            "",
            "## Goal",
            "",
            "- 用 `SAM masked -> mask-first aligned -> core body -> black pattern prior` 重做 Texas 的局部 ORB。",
            "- 这版 ORB 默认只在黑色纹路附近提点，不再把背景和整体边缘姿态当成主证据。",
            "- 黑纹提取前会先在前景内部做对比度增强，尽量避免 `SAM` 的黑背景把黑纹阈值拉偏。",
            "",
            "## How To Read",
            "",
            "- `image_feature_stats_v1.csv` 先看每张图的 `black_ratio`、`orb_region_kind` 和 `orb_keypoints`，判断黑纹提取是否落在你人工真正看的区域。",
            "- `test_pair_local_scores_v1.csv` 的 `black_orb_local_score` 越高，说明两张图在黑纹局部上越容易形成稳定 inlier。",
            "- 当前仍把它定位成 `negative evidence / human review aid`，不是直接替代全局同个体判断。",
            "",
            "## Inputs",
            "",
            f"- `predictions_path`: `{summary['predictions_path']}`",
            f"- `pair_csv_path`: `{summary['pair_csv_path']}`",
            f"- `view_manifest_path`: `{summary['view_manifest_path']}`",
            f"- `pair_score_col`: `{summary['pair_score_col']}`",
            "",
            "## Summary",
            "",
            f"- `image_count`: `{summary['image_count']}`",
            f"- `pair_count`: `{summary['pair_count']}`",
            f"- `sam_available_ratio`: `{summary['sam_available_ratio']}`",
            f"- `sam_fallback_count`: `{summary['sam_fallback_count']}`",
            f"- `orb_image_valid_ratio`: `{summary['orb_image_valid_ratio']}`",
            f"- `mean_black_ratio`: `{summary['mean_black_ratio']}`",
            f"- `mean_keypoints`: `{summary['mean_keypoints']}`",
            f"- `nonzero_local_pair_count`: `{summary['nonzero_local_pair_count']}`",
            f"- `mean_black_orb_local_score`: `{summary['mean_black_orb_local_score']}`",
            f"- `max_black_orb_local_score`: `{summary['max_black_orb_local_score']}`",
            f"- `mean_black_orb_inliers`: `{summary['mean_black_orb_inliers']}`",
            "",
            "## Top Pairs",
            "",
            dataframe_to_markdown_table(
                top_pairs_df[
                    [
                        column
                        for column in [
                            "image_id",
                            "neighbor_image_id",
                            "pair_score",
                            "black_orb_local_score",
                            "black_orb_inliers",
                            "black_orb_good_matches",
                            "left_black_ratio",
                            "right_black_ratio",
                            "left_orb_region_kind",
                            "right_orb_region_kind",
                        ]
                        if column in top_pairs_df.columns
                    ]
                ]
            ),
        ]
    )
    (reports_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    return {
        "local_table_path": local_table_path,
        "normalized_pair_path": normalized_pair_path,
        "image_stats_path": image_stats_path,
        "summary_path": reports_dir / "summary.md",
        "summary_json_path": reports_dir / "summary.json",
    }
