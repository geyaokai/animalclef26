from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

from .body_orientation_probe import compute_body_axis, mask_bbox

TEXAS_YOLO_WORLD_WHOLE_PROMPT_CANDIDATES: tuple[str, ...] = (
    "Texas horned lizard",
    "horned lizard",
    "lizard",
)
TEXAS_YOLO_WORLD_HEAD_PROMPT_CANDIDATES: tuple[str, ...] = (
    "Texas horned lizard head",
    "horned lizard head",
    "lizard head",
    "reptile head",
)
TEXAS_YOLO_WORLD_BODY_PROMPT_CANDIDATES: tuple[str, ...] = (
    "Texas horned lizard body",
    "horned lizard body",
    "lizard body",
    "reptile body",
)
TEXAS_YOLO_WORLD_PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "whole": TEXAS_YOLO_WORLD_WHOLE_PROMPT_CANDIDATES,
    "head": TEXAS_YOLO_WORLD_HEAD_PROMPT_CANDIDATES,
    "body": TEXAS_YOLO_WORLD_BODY_PROMPT_CANDIDATES,
}

DEFAULT_GRAY_LOW_PERCENTILE = 2.0
DEFAULT_GRAY_HIGH_PERCENTILE = 98.0
DEFAULT_SCALE_NORM_CANVAS_SIZE = (512, 512)
DEFAULT_TARGET_MAJOR_EXTENT_RATIO = 0.90
DEFAULT_MIDDLE_BAND_START_RATIO = 0.30
DEFAULT_MIDDLE_BAND_END_RATIO = 0.70
DEFAULT_MIDDLE_BAND_PADDING_RATIO = 0.08
DEFAULT_MIDDLE_BAND_MIN_FOREGROUND_RATIO = 0.10
DEFAULT_CENTER_SQUARE_PERCENTILE = 72.0
DEFAULT_CENTER_SQUARE_PADDING_RATIO = 0.10
DEFAULT_CENTER_SQUARE_MIN_SUBJECT_RATIO = 0.45
DEFAULT_CENTER_SQUARE_MIN_SIDE_RATIO = 0.36


def get_texas_yolo_world_prompt_candidates(part: str) -> list[str]:
    try:
        return list(TEXAS_YOLO_WORLD_PROMPT_CANDIDATES[str(part).lower()])
    except KeyError as exc:
        raise ValueError(f"Unsupported Texas detection part: {part}") from exc


def rgb_to_grayscale_rgb(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    return Image.merge("RGB", (gray, gray, gray))


def normalize_grayscale_array(
    gray: np.ndarray | Image.Image,
    *,
    focus_mask: np.ndarray | None = None,
    low_percentile: float = DEFAULT_GRAY_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_GRAY_HIGH_PERCENTILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray_array = _as_gray_array(gray)
    focus = _as_binary_mask(focus_mask, shape=gray_array.shape) if focus_mask is not None else np.ones_like(gray_array, dtype=bool)
    values = gray_array[focus] if focus.any() else gray_array.reshape(-1)
    if values.size == 0:
        return gray_array.copy(), {
            "mode": "gray_percentile",
            "low_percentile": float(low_percentile),
            "high_percentile": float(high_percentile),
            "low_value": 0.0,
            "high_value": 255.0,
            "fallback_reason": "empty_focus_mask",
        }

    low_value = float(np.percentile(values, float(low_percentile)))
    high_value = float(np.percentile(values, float(high_percentile)))
    normalized = gray_array.copy()
    fallback_reason: str | None = None
    if high_value <= low_value + 1e-6:
        fallback_reason = "flat_histogram"
    else:
        scaled = np.clip((gray_array[focus].astype(np.float32) - low_value) / (high_value - low_value), 0.0, 1.0)
        normalized[focus] = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    return normalized, {
        "mode": "gray_percentile",
        "low_percentile": float(low_percentile),
        "high_percentile": float(high_percentile),
        "low_value": round(low_value, 4),
        "high_value": round(high_value, 4),
        "focus_pixels": int(focus.sum()),
        "fallback_reason": fallback_reason,
    }


def grayscale_normalize_image(
    image: Image.Image,
    *,
    focus_mask: np.ndarray | None = None,
    low_percentile: float = DEFAULT_GRAY_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_GRAY_HIGH_PERCENTILE,
) -> tuple[Image.Image, dict[str, Any]]:
    normalized, payload = normalize_grayscale_array(
        image,
        focus_mask=focus_mask,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    gray_image = Image.fromarray(normalized, mode="L")
    return Image.merge("RGB", (gray_image, gray_image, gray_image)), payload


def scale_normalize_aligned_foreground(
    image: Image.Image,
    mask: np.ndarray,
    *,
    canvas_size: tuple[int, int] | None = None,
    target_major_extent_ratio: float = DEFAULT_TARGET_MAJOR_EXTENT_RATIO,
    background: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    """
    Enlarge the SAM-aligned Texas subject onto a square canvas.

    `target_major_extent_ratio` controls how much of the canvas the subject's
    major axis should occupy. This branch prefers enlargement over shrinking, so
    small subjects are scaled up, while already-large subjects keep their
    original size unless the canvas needs to expand.
    """
    rgb = image.convert("RGB")
    if canvas_size is None:
        square_edge = max(rgb.width, rgb.height, int(max(DEFAULT_SCALE_NORM_CANVAS_SIZE)))
        canvas_width = int(square_edge)
        canvas_height = int(square_edge)
    else:
        canvas_width, canvas_height = int(canvas_size[0]), int(canvas_size[1])
        if canvas_width <= 0 or canvas_height <= 0:
            raise ValueError("canvas_size must contain positive width and height")
    binary_mask = _as_binary_mask(mask, shape=(rgb.height, rgb.width)).astype(np.uint8)
    axis_stats = compute_body_axis(binary_mask)
    fallback_reason: str | None = None
    major_extent_source = "pca_major_extent"

    if axis_stats is not None and float(axis_stats.get("major_extent_px", 0.0)) > 1.0:
        source_major_extent = float(axis_stats["major_extent_px"])
        axis_angle_deg = float(axis_stats.get("axis_angle_deg", 0.0))
        axis_confidence = float(axis_stats.get("axis_confidence", 0.0))
    else:
        bbox = mask_bbox(binary_mask)
        if bbox is None:
            empty_canvas = Image.new("RGB", (canvas_width, canvas_height), color=background)
            return empty_canvas, np.zeros((canvas_height, canvas_width), dtype=np.uint8), {
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "target_major_extent_ratio": float(target_major_extent_ratio),
                "major_extent_before_px": 0.0,
                "major_extent_after_px": 0.0,
                "desired_scale_factor": 1.0,
                "fit_scale_factor": 1.0,
                "applied_scale_factor": 1.0,
                "major_extent_source": "empty_mask",
                "axis_angle_deg": 0.0,
                "axis_confidence": 0.0,
                "fallback_reason": "empty_mask",
            }
        x0, _y0, x1, _y1 = bbox
        source_major_extent = float(x1 - x0 + 1)
        axis_angle_deg = 0.0
        axis_confidence = 0.0
        major_extent_source = "bbox_width"
        fallback_reason = "axis_unavailable"

    target_major_extent_px = max(1.0, float(canvas_width) * float(target_major_extent_ratio))
    desired_scale_factor = max(target_major_extent_px / max(source_major_extent, 1.0), 1e-6)
    fit_scale_factor = min(canvas_width / max(rgb.width, 1), canvas_height / max(rgb.height, 1))
    applied_scale_factor = max(1.0, desired_scale_factor)
    if desired_scale_factor < 1.0 - 1e-6:
        fallback_reason = _append_reason(fallback_reason, "kept_original_scale")

    resized_width = max(1, int(round(rgb.width * applied_scale_factor)))
    resized_height = max(1, int(round(rgb.height * applied_scale_factor)))
    if resized_width > canvas_width or resized_height > canvas_height:
        canvas_width = max(canvas_width, resized_width)
        canvas_height = max(canvas_height, resized_height)
        fit_scale_factor = min(canvas_width / max(rgb.width, 1), canvas_height / max(rgb.height, 1))
        fallback_reason = _append_reason(fallback_reason, "expanded_canvas")

    resized_image = rgb.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    resized_mask = Image.fromarray(binary_mask * 255, mode="L").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )
    resized_mask_array = (np.asarray(resized_mask, dtype=np.uint8) > 0).astype(np.uint8)

    image_array = np.asarray(resized_image, dtype=np.uint8)
    canvas_image_array = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas_image_array[:] = np.asarray(background, dtype=np.uint8)
    canvas_mask_array = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    paste_x, paste_y = _compute_centered_foreground_offset(
        resized_mask_array,
        canvas_size=(canvas_width, canvas_height),
    )
    _paste_array(canvas_image_array, image_array, paste_x=paste_x, paste_y=paste_y)
    _paste_array(canvas_mask_array, resized_mask_array, paste_x=paste_x, paste_y=paste_y)

    normalized_axis_stats = compute_body_axis(canvas_mask_array)
    major_extent_after = float(normalized_axis_stats["major_extent_px"]) if normalized_axis_stats is not None else 0.0

    return Image.fromarray(canvas_image_array, mode="RGB"), canvas_mask_array, {
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "target_major_extent_ratio": float(target_major_extent_ratio),
        "target_major_extent_px": round(target_major_extent_px, 4),
        "major_extent_before_px": round(source_major_extent, 4),
        "major_extent_after_px": round(major_extent_after, 4),
        "desired_scale_factor": round(float(desired_scale_factor), 6),
        "fit_scale_factor": round(float(fit_scale_factor), 6),
        "applied_scale_factor": round(float(applied_scale_factor), 6),
        "major_extent_source": major_extent_source,
        "axis_angle_deg": round(axis_angle_deg, 4),
        "axis_confidence": round(axis_confidence, 6),
        "fallback_reason": fallback_reason,
    }


def crop_texas_middle_band(
    image: Image.Image,
    mask: np.ndarray,
    *,
    band_start_ratio: float = DEFAULT_MIDDLE_BAND_START_RATIO,
    band_end_ratio: float = DEFAULT_MIDDLE_BAND_END_RATIO,
    padding_ratio: float = DEFAULT_MIDDLE_BAND_PADDING_RATIO,
    min_foreground_ratio: float = DEFAULT_MIDDLE_BAND_MIN_FOREGROUND_RATIO,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    # Backward-compatible wrapper: keep the legacy entrypoint, but switch the
    # default Texas heuristic to a centered square body crop so downstream code
    # does not need to change immediately.
    del band_start_ratio, band_end_ratio
    return crop_texas_center_body_square(
        image,
        mask,
        padding_ratio=padding_ratio,
        min_subject_ratio=min_foreground_ratio,
    )


def crop_texas_center_body_square(
    image: Image.Image,
    mask: np.ndarray,
    *,
    center_percentile: float = DEFAULT_CENTER_SQUARE_PERCENTILE,
    padding_ratio: float = DEFAULT_CENTER_SQUARE_PADDING_RATIO,
    min_subject_ratio: float = DEFAULT_CENTER_SQUARE_MIN_SUBJECT_RATIO,
    min_side_ratio: float = DEFAULT_CENTER_SQUARE_MIN_SIDE_RATIO,
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    """
    Crop a center-focused square body view for Texas.

    The crop center comes from the foreground centroid, then expands with a
    percentile-based radius so the dorsal trunk stays dominant while limbs,
    head, and tail are weakened. `min_subject_ratio` is a safeguard: if the
    center crop keeps too little foreground, the function falls back to a
    slightly larger bbox-centered square and records the fallback in metadata.
    """
    rgb = image.convert("RGB")
    binary_mask = _as_binary_mask(mask, shape=(rgb.height, rgb.width)).astype(np.uint8)
    bbox = mask_bbox(binary_mask)
    if bbox is None:
        return rgb.copy(), binary_mask.copy(), {
            "crop_strategy": "center_body_square",
            "center_percentile": float(center_percentile),
            "padding_ratio": float(padding_ratio),
            "foreground_ratio_in_crop": 0.0,
            "foreground_ratio_of_subject": 0.0,
            "center_xy": (0.0, 0.0),
            "square_side_px": float(min(rgb.width, rgb.height)),
            "crop_box_xyxy": (0, 0, rgb.width, rgb.height),
            "fallback_reason": "empty_mask",
        }

    x0, y0, x1, y1 = bbox
    foreground_width = x1 - x0 + 1
    foreground_height = y1 - y0 + 1
    foreground_area = int(binary_mask.sum())
    ys, xs = np.nonzero(binary_mask)
    center_x = float(xs.mean())
    center_y = float(ys.mean())

    quantile = float(np.clip(center_percentile, 5.0, 99.0))
    radius_x = float(np.percentile(np.abs(xs.astype(np.float32) - center_x), quantile))
    radius_y = float(np.percentile(np.abs(ys.astype(np.float32) - center_y), quantile))
    base_half_side = max(radius_x, radius_y)
    min_half_side = 0.5 * float(max(foreground_width, foreground_height)) * float(np.clip(min_side_ratio, 0.05, 1.0))
    pad = max(2.0, float(max(foreground_width, foreground_height)) * float(padding_ratio))
    half_side = max(base_half_side + pad, min_half_side)

    crop_x0, crop_y0, crop_x1, crop_y1 = _centered_square_crop_box(
        center_x=center_x,
        center_y=center_y,
        half_side=half_side,
        image_width=rgb.width,
        image_height=rgb.height,
    )

    crop_mask = binary_mask[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_foreground_ratio = float(crop_mask.mean()) if crop_mask.size else 0.0
    subject_foreground_ratio = float(crop_mask.sum() / max(foreground_area, 1))
    fallback_reason: str | None = None

    if subject_foreground_ratio < float(min_subject_ratio):
        bbox_side = float(max(foreground_width, foreground_height))
        crop_x0, crop_y0, crop_x1, crop_y1 = _centered_square_crop_box(
            center_x=0.5 * (x0 + x1),
            center_y=0.5 * (y0 + y1),
            half_side=0.5 * bbox_side + pad,
            image_width=rgb.width,
            image_height=rgb.height,
        )
        crop_mask = binary_mask[crop_y0:crop_y1, crop_x0:crop_x1]
        crop_foreground_ratio = float(crop_mask.mean()) if crop_mask.size else 0.0
        subject_foreground_ratio = float(crop_mask.sum() / max(foreground_area, 1))
        fallback_reason = "sparse_center_square"

    crop_image = rgb.crop((crop_x0, crop_y0, crop_x1, crop_y1))
    return crop_image, crop_mask.copy(), {
        "crop_strategy": "center_body_square",
        "center_percentile": round(quantile, 4),
        "padding_ratio": round(float(padding_ratio), 4),
        "foreground_ratio_in_crop": round(crop_foreground_ratio, 6),
        "foreground_ratio_of_subject": round(subject_foreground_ratio, 6),
        "center_xy": (round(center_x, 4), round(center_y, 4)),
        "square_side_px": int(crop_x1 - crop_x0),
        "crop_box_xyxy": (int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)),
        "fallback_reason": fallback_reason,
    }


def build_texas_view_metadata(
    *,
    row: Mapping[str, Any] | None,
    view_name: str,
    yolo_part: str | None = None,
    prompt_payload: Mapping[str, Any] | None = None,
    grayscale_payload: Mapping[str, Any] | None = None,
    scale_payload: Mapping[str, Any] | None = None,
    crop_payload: Mapping[str, Any] | None = None,
    fallback_reason: str | None = None,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "dataset": _row_value(row, "dataset"),
        "split": _row_value(row, "split"),
        "image_id": _row_value(row, "image_id"),
        "view_name": str(view_name),
    }
    if yolo_part is not None:
        payload["yolo_part"] = str(yolo_part)

    if prompt_payload is not None:
        payload["selected_prompt"] = _mapping_value(prompt_payload, "selected_prompt")
        payload["prompt_rank"] = _mapping_value(prompt_payload, "prompt_rank")
        payload["attempted_prompt_count"] = _mapping_value(prompt_payload, "attempted_prompt_count")
        payload["detector_conf"] = _mapping_value(prompt_payload, "confidence", fallback_key="best_score")
        payload["detector_iou"] = _mapping_value(prompt_payload, "iou")

    if grayscale_payload is not None:
        payload["gray_low_percentile"] = _mapping_value(grayscale_payload, "low_percentile")
        payload["gray_high_percentile"] = _mapping_value(grayscale_payload, "high_percentile")
        payload["gray_low_value"] = _mapping_value(grayscale_payload, "low_value")
        payload["gray_high_value"] = _mapping_value(grayscale_payload, "high_value")

    if scale_payload is not None:
        payload["axis_angle_deg"] = _mapping_value(scale_payload, "axis_angle_deg")
        payload["axis_confidence"] = _mapping_value(scale_payload, "axis_confidence")
        payload["major_extent_before_px"] = _mapping_value(scale_payload, "major_extent_before_px")
        payload["major_extent_after_px"] = _mapping_value(scale_payload, "major_extent_after_px")
        payload["target_major_extent_ratio"] = _mapping_value(scale_payload, "target_major_extent_ratio")
        payload["scale_factor"] = _mapping_value(scale_payload, "applied_scale_factor")

    if crop_payload is not None:
        payload["crop_strategy"] = _mapping_value(crop_payload, "crop_strategy")
        payload["crop_center_xy"] = _mapping_value(crop_payload, "center_xy")
        payload["crop_square_side_px"] = _mapping_value(crop_payload, "square_side_px")
        payload["crop_center_percentile"] = _mapping_value(crop_payload, "center_percentile")
        payload["band_start_ratio"] = _mapping_value(crop_payload, "band_start_ratio")
        payload["band_end_ratio"] = _mapping_value(crop_payload, "band_end_ratio")
        payload["crop_foreground_ratio"] = _mapping_value(crop_payload, "foreground_ratio_in_crop")
        payload["subject_coverage_ratio"] = _mapping_value(crop_payload, "foreground_ratio_of_subject")
        payload["crop_box_xyxy"] = _mapping_value(crop_payload, "crop_box_xyxy")

    reasons = [
        str(value)
        for value in (
            fallback_reason,
            _mapping_value(grayscale_payload, "fallback_reason"),
            _mapping_value(scale_payload, "fallback_reason"),
            _mapping_value(crop_payload, "fallback_reason"),
            _mapping_value(prompt_payload, "fallback_reason"),
        )
        if value not in (None, "", "None")
    ]
    payload["fallback_reason"] = ";".join(dict.fromkeys(reasons)) or None

    if extra_payload is not None:
        for key, value in extra_payload.items():
            payload[str(key)] = value
    return payload


def _as_gray_array(gray: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(gray, Image.Image):
        return np.asarray(gray.convert("L"), dtype=np.uint8)
    gray_array = np.asarray(gray, dtype=np.uint8)
    if gray_array.ndim == 3:
        return np.asarray(Image.fromarray(gray_array, mode="RGB").convert("L"), dtype=np.uint8)
    if gray_array.ndim != 2:
        raise ValueError("gray input must be a 2D array or PIL image")
    return gray_array


def _as_binary_mask(mask: np.ndarray | None, *, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, dtype=bool)
    binary_mask = np.asarray(mask, dtype=np.uint8)
    if binary_mask.shape != shape:
        raise ValueError(f"mask shape mismatch: expected {shape}, received {binary_mask.shape}")
    return binary_mask > 0


def _compute_centered_foreground_offset(
    mask: np.ndarray,
    *,
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    canvas_width, canvas_height = int(canvas_size[0]), int(canvas_size[1])
    bbox = mask_bbox(mask.astype(np.uint8))
    if bbox is None:
        return max((canvas_width - mask.shape[1]) // 2, 0), max((canvas_height - mask.shape[0]) // 2, 0)
    x0, y0, x1, y1 = bbox
    foreground_cx = 0.5 * (x0 + x1)
    foreground_cy = 0.5 * (y0 + y1)
    paste_x = int(round((canvas_width - 1) * 0.5 - foreground_cx))
    paste_y = int(round((canvas_height - 1) * 0.5 - foreground_cy))
    return paste_x, paste_y


def _centered_square_crop_box(
    *,
    center_x: float,
    center_y: float,
    half_side: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    side = max(2, int(round(half_side * 2.0)))
    side = min(side, int(image_width), int(image_height))
    crop_x0 = int(round(center_x - 0.5 * side))
    crop_y0 = int(round(center_y - 0.5 * side))
    crop_x0 = max(0, min(crop_x0, int(image_width) - side))
    crop_y0 = max(0, min(crop_y0, int(image_height) - side))
    return crop_x0, crop_y0, crop_x0 + side, crop_y0 + side


def _paste_array(canvas: np.ndarray, source: np.ndarray, *, paste_x: int, paste_y: int) -> None:
    source_height, source_width = source.shape[:2]
    canvas_height, canvas_width = canvas.shape[:2]
    dst_x0 = max(0, int(paste_x))
    dst_y0 = max(0, int(paste_y))
    dst_x1 = min(canvas_width, int(paste_x) + source_width)
    dst_y1 = min(canvas_height, int(paste_y) + source_height)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return
    src_x0 = dst_x0 - int(paste_x)
    src_y0 = dst_y0 - int(paste_y)
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]


def _row_value(row: Mapping[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    return row.get(key)


def _mapping_value(mapping: Mapping[str, Any] | None, key: str, *, fallback_key: str | None = None) -> Any:
    if mapping is None:
        return None
    if key in mapping:
        return mapping.get(key)
    if fallback_key is not None and fallback_key in mapping:
        return mapping.get(fallback_key)
    return None


def _append_reason(existing: str | None, reason: str) -> str:
    if not existing:
        return str(reason)
    if reason in existing.split(";"):
        return existing
    return f"{existing};{reason}"
