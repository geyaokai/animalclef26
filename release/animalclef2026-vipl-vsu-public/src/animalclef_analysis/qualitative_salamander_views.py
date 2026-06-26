from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES: tuple[str, ...] = (
    "fire salamander",
    "salamander",
    "amphibian",
)
SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES: tuple[str, ...] = (
    "fire salamander head",
    "salamander head",
    "amphibian head",
)
SALAMANDER_YOLOWORLD_BODY_PROMPT_CANDIDATES: tuple[str, ...] = (
    "fire salamander body",
    "salamander body",
    "amphibian body",
)
SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "whole": SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES,
    "head": SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES,
    "body": SALAMANDER_YOLOWORLD_BODY_PROMPT_CANDIDATES,
}

DEFAULT_SCALE_TARGET_EXTENT_RATIO = 0.88
DEFAULT_END_A_RATIO = (0.0, 0.30)
DEFAULT_MIDDLE_RATIO = (0.30, 0.70)
DEFAULT_END_B_RATIO = (0.70, 1.0)
DEFAULT_VERTICAL_PADDING_RATIO = 0.06
DEFAULT_HORIZONTAL_PADDING_RATIO = 0.03
DEFAULT_TRUNK_EXTREMITY_TRIM_RATIO = 0.18
DEFAULT_TRUNK_MINOR_EXTENT_RATIO = 0.72


@dataclass(frozen=True)
class SalamanderCropResult:
    rgb: Image.Image
    mask: np.ndarray
    metadata: dict[str, Any]


def _compute_body_axis(mask: np.ndarray) -> dict[str, float] | None:
    from .body_orientation_probe import compute_body_axis

    return compute_body_axis(mask)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _clip_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left_i = max(0, min(int(round(left)), width - 1))
    top_i = max(0, min(int(round(top)), height - 1))
    right_i = max(left_i + 1, min(int(round(right)), width))
    bottom_i = max(top_i + 1, min(int(round(bottom)), height))
    return left_i, top_i, right_i, bottom_i


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return (np.asarray(Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").resize(size, Image.Resampling.NEAREST)) > 0).astype(np.uint8)


def _paste_centered(
    image: Image.Image,
    mask: np.ndarray,
    *,
    output_size: tuple[int, int],
    background: tuple[int, int, int],
) -> tuple[Image.Image, np.ndarray]:
    output_w, output_h = output_size
    canvas_rgb = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    canvas_rgb[:, :] = np.asarray(background, dtype=np.uint8)
    canvas_mask = np.zeros((output_h, output_w), dtype=np.uint8)

    source_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    source_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    source_bbox = _mask_bbox(source_mask)
    if source_bbox is None:
        return Image.fromarray(canvas_rgb, mode="RGB"), canvas_mask

    x0, y0, x1, y1 = source_bbox
    source_center_x = (x0 + x1) / 2.0
    source_center_y = (y0 + y1) / 2.0
    target_center_x = (output_w - 1) / 2.0
    target_center_y = (output_h - 1) / 2.0
    offset_x = int(round(target_center_x - source_center_x))
    offset_y = int(round(target_center_y - source_center_y))

    src_h, src_w = source_mask.shape
    dst_left = max(0, offset_x)
    dst_top = max(0, offset_y)
    dst_right = min(output_w, offset_x + src_w)
    dst_bottom = min(output_h, offset_y + src_h)
    if dst_left >= dst_right or dst_top >= dst_bottom:
        return Image.fromarray(canvas_rgb, mode="RGB"), canvas_mask

    src_left = max(0, -offset_x)
    src_top = max(0, -offset_y)
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    canvas_rgb[dst_top:dst_bottom, dst_left:dst_right] = source_rgb[src_top:src_bottom, src_left:src_right]
    canvas_mask[dst_top:dst_bottom, dst_left:dst_right] = source_mask[src_top:src_bottom, src_left:src_right]
    return Image.fromarray(canvas_rgb, mode="RGB"), canvas_mask


def _resolve_major_extent(mask: np.ndarray) -> tuple[float, dict[str, Any]]:
    axis_stats = _compute_body_axis(mask)
    bbox = _mask_bbox(mask)
    if axis_stats is not None and float(axis_stats.get("major_extent_px", 0.0)) > 1.0:
        return float(axis_stats["major_extent_px"]), {
            "axis_confidence": round(float(axis_stats.get("axis_confidence", 0.0)), 6),
            "axis_angle_deg": round(float(axis_stats.get("axis_angle_deg", 0.0)), 6),
            "major_extent_source": "body_axis",
        }
    if bbox is None:
        return 0.0, {
            "axis_confidence": 0.0,
            "axis_angle_deg": 0.0,
            "major_extent_source": "missing_mask",
        }
    x0, y0, x1, y1 = bbox
    bbox_width = float(x1 - x0 + 1)
    bbox_height = float(y1 - y0 + 1)
    return max(bbox_width, bbox_height), {
        "axis_confidence": round(float(axis_stats.get("axis_confidence", 0.0)), 6) if axis_stats is not None else 0.0,
        "axis_angle_deg": round(float(axis_stats.get("axis_angle_deg", 0.0)), 6) if axis_stats is not None else 0.0,
        "major_extent_source": "bbox",
    }


def _foreground_center(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    # Use the foreground median instead of the mean so long tails or curled ends
    # do not drag the crop center away from the main trunk.
    return float(np.median(xs)), float(np.median(ys))


def _resolve_major_axis(bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    bbox_width = float(x1 - x0 + 1)
    bbox_height = float(y1 - y0 + 1)
    if bbox_height >= bbox_width:
        return {
            "major_axis": "y",
            "minor_axis": "x",
            "major_min": float(y0),
            "major_max": float(y1 + 1),
            "minor_min": float(x0),
            "minor_max": float(x1 + 1),
            "major_extent": bbox_height,
            "minor_extent": bbox_width,
        }
    return {
        "major_axis": "x",
        "minor_axis": "y",
        "major_min": float(x0),
        "major_max": float(x1 + 1),
        "minor_min": float(y0),
        "minor_max": float(y1 + 1),
        "major_extent": bbox_width,
        "minor_extent": bbox_height,
    }


def _resolve_centered_trunk_interval(
    *,
    major_min: float,
    major_max: float,
    major_center: float,
    extremity_trim_ratio: float,
) -> tuple[float, float]:
    major_extent = max(1.0, float(major_max - major_min))
    trim_ratio = max(0.0, min(float(extremity_trim_ratio), 0.45))
    available_half_span = min(float(major_center - major_min), float(major_max - major_center))
    target_half_span = 0.5 * major_extent * (1.0 - 2.0 * trim_ratio)
    safe_half_span = max(1.0, min(available_half_span, target_half_span))
    trunk_min = float(major_center) - safe_half_span
    trunk_max = float(major_center) + safe_half_span
    if trunk_max - trunk_min < 2.0:
        return float(major_min), float(major_max)
    return trunk_min, trunk_max


def scale_normalize_aligned_foreground(
    image: Image.Image,
    mask: np.ndarray,
    *,
    output_size: tuple[int, int] | None = None,
    target_extent_ratio: float = DEFAULT_SCALE_TARGET_EXTENT_RATIO,
    background: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    """Scale the aligned foreground so its major extent fills a fixed canvas ratio."""
    binary_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if output_size is None:
        output_size = image.size
    output_w, output_h = int(output_size[0]), int(output_size[1])
    if output_w <= 0 or output_h <= 0:
        raise ValueError("output_size must contain positive integers.")

    major_extent_before, extent_payload = _resolve_major_extent(binary_mask)
    if major_extent_before <= 1.0:
        fallback_payload = {
            "scale_applied": False,
            "fallback_reason": "missing_major_extent",
            "scale_factor": 1.0,
            "major_extent_before_px": round(float(max(major_extent_before, 0.0)), 6),
            "major_extent_after_px": round(float(max(major_extent_before, 0.0)), 6),
            "target_extent_ratio": round(float(target_extent_ratio), 6),
            "target_major_extent_px": 0.0,
            "output_size": [output_w, output_h],
        }
        fallback_payload.update(extent_payload)
        return image.convert("RGB").copy(), binary_mask.copy(), fallback_payload

    target_major_extent_px = float(max(output_w, output_h)) * float(target_extent_ratio)
    desired_scale_factor = float(target_major_extent_px / major_extent_before)
    scale_factor = max(1.0, desired_scale_factor)
    fallback_reason = "kept_original_scale" if desired_scale_factor < 1.0 - 1e-6 else ""
    scaled_w = max(1, int(round(image.size[0] * scale_factor)))
    scaled_h = max(1, int(round(image.size[1] * scale_factor)))
    if scaled_w > output_w or scaled_h > output_h:
        output_w = max(output_w, scaled_w)
        output_h = max(output_h, scaled_h)
        fallback_reason = "|".join(value for value in [fallback_reason, "expanded_canvas"] if value)
    scaled_rgb = image.convert("RGB").resize((scaled_w, scaled_h), Image.Resampling.BILINEAR)
    scaled_mask = _resize_mask(binary_mask, (scaled_w, scaled_h))
    normalized_rgb, normalized_mask = _paste_centered(
        scaled_rgb,
        scaled_mask,
        output_size=(output_w, output_h),
        background=background,
    )
    major_extent_after, after_payload = _resolve_major_extent(normalized_mask)
    payload = {
        "scale_applied": True,
        "fallback_reason": fallback_reason,
        "scale_factor": round(scale_factor, 6),
        "desired_scale_factor": round(desired_scale_factor, 6),
        "major_extent_before_px": round(major_extent_before, 6),
        "major_extent_after_px": round(major_extent_after, 6),
        "target_extent_ratio": round(float(target_extent_ratio), 6),
        "target_major_extent_px": round(target_major_extent_px, 6),
        "output_size": [output_w, output_h],
    }
    payload.update(extent_payload)
    payload["axis_confidence_after"] = round(float(after_payload.get("axis_confidence", 0.0)), 6)
    payload["axis_angle_deg_after"] = round(float(after_payload.get("axis_angle_deg", 0.0)), 6)
    payload["major_extent_source_after"] = str(after_payload.get("major_extent_source", "unknown"))
    return normalized_rgb, normalized_mask, payload


def build_salamander_crop_metadata(
    *,
    crop_name: str,
    crop_ratio: tuple[float, float],
    scale_payload: Mapping[str, Any] | None = None,
    fallback_reasons: Sequence[str] | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "crop_name": str(crop_name),
        "crop_start_ratio": round(float(crop_ratio[0]), 6),
        "crop_end_ratio": round(float(crop_ratio[1]), 6),
        "fallback_reason": "|".join(str(reason) for reason in (fallback_reasons or []) if str(reason)),
    }
    if scale_payload is not None:
        payload["scale_factor"] = round(float(scale_payload.get("scale_factor", 1.0)), 6)
        payload["target_extent_ratio"] = round(float(scale_payload.get("target_extent_ratio", DEFAULT_SCALE_TARGET_EXTENT_RATIO)), 6)
    else:
        payload["scale_factor"] = 1.0
        payload["target_extent_ratio"] = round(float(DEFAULT_SCALE_TARGET_EXTENT_RATIO), 6)
    if crop_box is not None:
        payload["crop_box"] = [int(crop_box[0]), int(crop_box[1]), int(crop_box[2]), int(crop_box[3])]
        payload["crop_width_px"] = int(crop_box[2] - crop_box[0])
        payload["crop_height_px"] = int(crop_box[3] - crop_box[1])
    if extra_metadata is not None:
        payload.update(dict(extra_metadata))
    return payload


def generate_heuristic_end_middle_crops(
    image: Image.Image,
    mask: np.ndarray,
    *,
    scale_payload: Mapping[str, Any] | None = None,
    end_a_ratio: tuple[float, float] = DEFAULT_END_A_RATIO,
    middle_ratio: tuple[float, float] = DEFAULT_MIDDLE_RATIO,
    end_b_ratio: tuple[float, float] = DEFAULT_END_B_RATIO,
    vertical_padding_ratio: float = DEFAULT_VERTICAL_PADDING_RATIO,
    horizontal_padding_ratio: float = DEFAULT_HORIZONTAL_PADDING_RATIO,
    trunk_extremity_trim_ratio: float = DEFAULT_TRUNK_EXTREMITY_TRIM_RATIO,
    trunk_minor_extent_ratio: float = DEFAULT_TRUNK_MINOR_EXTENT_RATIO,
) -> dict[str, SalamanderCropResult]:
    """
    Build center-biased trunk crops from the aligned salamander foreground.

    The crop first finds a robust foreground center from the mask median, then expands
    symmetrically along the major axis while trimming both extremes. This reduces
    head/tail dominance and keeps the local crop focused on the trunk pattern.

    `end_a`, `middle`, and `end_b` are kept for interface compatibility, but they now
    partition the center-trimmed trunk interval instead of touching the full bbox ends.
    `trunk_minor_extent_ratio` controls how much of the minor axis is preserved around
    the foreground center, which suppresses background spill and thin tail tips.
    """
    binary_mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    bbox = _mask_bbox(binary_mask)
    crop_specs = {
        "end_a": end_a_ratio,
        "middle": middle_ratio,
        "end_b": end_b_ratio,
    }
    fallback_reasons: list[str] = []
    if bbox is None:
        fallback_reasons.append("missing_mask")
        full_rgb = image.convert("RGB").copy()
        full_mask = binary_mask.copy()
        return {
            name: SalamanderCropResult(
                rgb=full_rgb.copy(),
                mask=full_mask.copy(),
                metadata=build_salamander_crop_metadata(
                    crop_name=name,
                    crop_ratio=ratio,
                    scale_payload=scale_payload,
                    fallback_reasons=fallback_reasons,
                    crop_box=(0, 0, full_rgb.size[0], full_rgb.size[1]),
                ),
            )
            for name, ratio in crop_specs.items()
        }

    foreground_center = _foreground_center(binary_mask)
    if foreground_center is None:
        fallback_reasons.append("missing_foreground_center")
        foreground_center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    axis_layout = _resolve_major_axis(bbox)
    major_axis = str(axis_layout["major_axis"])
    minor_axis = str(axis_layout["minor_axis"])
    center_x, center_y = foreground_center
    major_center = center_y if major_axis == "y" else center_x
    minor_center = center_x if major_axis == "y" else center_y

    trunk_major_min, trunk_major_max = _resolve_centered_trunk_interval(
        major_min=float(axis_layout["major_min"]),
        major_max=float(axis_layout["major_max"]),
        major_center=float(major_center),
        extremity_trim_ratio=trunk_extremity_trim_ratio,
    )
    trunk_major_extent = max(1.0, trunk_major_max - trunk_major_min)
    minor_half_span = 0.5 * float(axis_layout["minor_extent"]) * max(0.2, min(float(trunk_minor_extent_ratio), 1.0))
    bbox_width = max(1, bbox[2] - bbox[0] + 1)
    bbox_height = max(1, bbox[3] - bbox[1] + 1)
    pad_x = float(bbox_width) * float(horizontal_padding_ratio)
    pad_y = float(bbox_height) * float(vertical_padding_ratio)
    image_rgb = image.convert("RGB")
    crops: dict[str, SalamanderCropResult] = {}

    for crop_name, crop_ratio in crop_specs.items():
        ratio_start = max(0.0, min(float(crop_ratio[0]), 1.0))
        ratio_end = max(ratio_start, min(float(crop_ratio[1]), 1.0))
        if ratio_end <= ratio_start:
            local_fallback = [*fallback_reasons, "invalid_crop_ratio"]
            crop_box = (0, 0, image_rgb.size[0], image_rgb.size[1])
        else:
            crop_major_start = trunk_major_min + trunk_major_extent * ratio_start
            crop_major_end = trunk_major_min + trunk_major_extent * ratio_end
            if major_axis == "x":
                crop_left = crop_major_start - pad_x
                crop_right = crop_major_end + pad_x
                crop_top = minor_center - minor_half_span - pad_y
                crop_bottom = minor_center + minor_half_span + pad_y
            else:
                crop_left = minor_center - minor_half_span - pad_x
                crop_right = minor_center + minor_half_span + pad_x
                crop_top = crop_major_start - pad_y
                crop_bottom = crop_major_end + pad_y
            crop_box = _clip_box(
                crop_left,
                crop_top,
                crop_right,
                crop_bottom,
                width=image_rgb.size[0],
                height=image_rgb.size[1],
            )
            local_fallback = fallback_reasons.copy()

        left, top, right, bottom = crop_box
        crop_rgb = image_rgb.crop((left, top, right, bottom))
        crop_mask = binary_mask[top:bottom, left:right].copy()
        crops[crop_name] = SalamanderCropResult(
            rgb=crop_rgb,
            mask=crop_mask,
            metadata=build_salamander_crop_metadata(
                crop_name=crop_name,
                crop_ratio=(ratio_start, ratio_end),
                scale_payload=scale_payload,
                fallback_reasons=local_fallback,
                crop_box=crop_box,
                extra_metadata={
                    "crop_strategy": "center_trunk_rectangle",
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "foreground_center_xy": [round(float(center_x), 3), round(float(center_y), 3)],
                    "foreground_center_source": "mask_median",
                    "trunk_extremity_trim_ratio": round(float(trunk_extremity_trim_ratio), 6),
                    "trunk_minor_extent_ratio": round(float(trunk_minor_extent_ratio), 6),
                    "trunk_major_range": [round(float(trunk_major_min), 3), round(float(trunk_major_max), 3)],
                },
            ),
        )
    return crops


__all__ = [
    "DEFAULT_END_A_RATIO",
    "DEFAULT_END_B_RATIO",
    "DEFAULT_HORIZONTAL_PADDING_RATIO",
    "DEFAULT_MIDDLE_RATIO",
    "DEFAULT_SCALE_TARGET_EXTENT_RATIO",
    "DEFAULT_TRUNK_EXTREMITY_TRIM_RATIO",
    "DEFAULT_TRUNK_MINOR_EXTENT_RATIO",
    "DEFAULT_VERTICAL_PADDING_RATIO",
    "SALAMANDER_YOLOWORLD_BODY_PROMPT_CANDIDATES",
    "SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES",
    "SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES",
    "SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES",
    "SalamanderCropResult",
    "build_salamander_crop_metadata",
    "generate_heuristic_end_middle_crops",
    "scale_normalize_aligned_foreground",
]
