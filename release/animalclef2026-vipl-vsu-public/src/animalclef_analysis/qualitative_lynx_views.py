from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .initial_audit import compute_sharpness


DEFAULT_HIST_LOW_PERCENTILE = 1.0
DEFAULT_HIST_HIGH_PERCENTILE = 99.0
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_GRID_SIZE = 8

LYNX_NORMALIZATION_MODES = {"gray", "hist_norm", "clahe"}


def _load_cv2() -> Any | None:
    try:  # pragma: no cover - depends on environment
        import cv2
    except ModuleNotFoundError:  # pragma: no cover
        return None
    return cv2


def _coerce_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.shape != shape:
        raise ValueError(f"mask shape {mask_array.shape} does not match grayscale image shape {shape}")
    return mask_array > 0


def _focus_values(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    binary = _coerce_mask(mask, gray.shape)
    if binary is None:
        return gray.reshape(-1)
    values = gray[binary]
    return values if values.size else gray.reshape(-1)


def _as_float(value: Any, digits: int = 6) -> float:
    return round(float(value), digits)


def _normalize_percentile_range(low_percentile: float, high_percentile: float) -> tuple[float, float]:
    low = float(low_percentile)
    high = float(high_percentile)
    if not (0.0 <= low <= 100.0 and 0.0 <= high <= 100.0):
        raise ValueError("percentiles must be in [0, 100]")
    if high <= low:
        raise ValueError("high_percentile must be greater than low_percentile")
    return low, high


def to_grayscale_uint8(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)

    array = np.asarray(image)
    if array.ndim == 2:
        return np.clip(array, 0, 255).astype(np.uint8, copy=False)
    if array.ndim != 3:
        raise ValueError("expected a PIL image or a 2D/3D numpy array")
    if array.shape[2] == 1:
        return np.clip(array[..., 0], 0, 255).astype(np.uint8, copy=False)
    if array.shape[2] < 3:
        raise ValueError("3D image arrays must have at least 3 channels")

    rgb = array[..., :3].astype(np.float32, copy=False)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def histogram_normalize_gray(
    image: Image.Image | np.ndarray,
    *,
    low_percentile: float = DEFAULT_HIST_LOW_PERCENTILE,
    high_percentile: float = DEFAULT_HIST_HIGH_PERCENTILE,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    low, high = _normalize_percentile_range(low_percentile, high_percentile)
    gray = to_grayscale_uint8(image)
    values = _focus_values(gray, mask).astype(np.float32)
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo + 1e-6:
        return gray.copy()

    normalized = (gray.astype(np.float32) - lo) / (hi - lo)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def clahe_normalize_gray(
    image: Image.Image | np.ndarray,
    *,
    clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    grid_size: int = DEFAULT_CLAHE_GRID_SIZE,
) -> np.ndarray:
    gray = to_grayscale_uint8(image)
    if float(clip_limit) <= 0:
        return gray.copy()

    cv2 = _load_cv2()
    if cv2 is not None:
        tile_size = max(2, int(grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(tile_size, tile_size),
        )
        return clahe.apply(gray)

    stretched = histogram_normalize_gray(gray)
    fallback = ImageOps.autocontrast(Image.fromarray(stretched, mode="L"))
    return np.asarray(fallback, dtype=np.uint8)


def compute_lightweight_contrast_stats(
    image: Image.Image | np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, float]:
    gray = to_grayscale_uint8(image)
    values = _focus_values(gray, mask).astype(np.float32)
    if values.size == 0:
        return {
            "gray_mean": 0.0,
            "gray_std": 0.0,
            "gray_p05": 0.0,
            "gray_p50": 0.0,
            "gray_p95": 0.0,
            "gray_contrast_p95_p05": 0.0,
            "gray_sharpness": 0.0,
        }

    p05 = float(np.percentile(values, 5.0))
    p50 = float(np.percentile(values, 50.0))
    p95 = float(np.percentile(values, 95.0))
    return {
        "gray_mean": _as_float(np.mean(values)),
        "gray_std": _as_float(np.std(values)),
        "gray_p05": _as_float(p05),
        "gray_p50": _as_float(p50),
        "gray_p95": _as_float(p95),
        "gray_contrast_p95_p05": _as_float(max(0.0, p95 - p05)),
        "gray_sharpness": _as_float(compute_sharpness(gray)),
    }


def build_lynx_normalization_metadata(
    *,
    mode: str,
    low_percentile: float | None = None,
    high_percentile: float | None = None,
    clip_limit: float | None = None,
    grid_size: int | None = None,
    backend: str | None = None,
    contrast_stats: dict[str, float] | None = None,
    image_shape: tuple[int, int] | None = None,
) -> dict[str, object]:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in LYNX_NORMALIZATION_MODES:
        raise ValueError(f"unsupported Lynx normalization mode: {mode}")

    payload: dict[str, object] = {
        "lynx_normalization_mode_v1": normalized_mode,
        "lynx_normalization_backend_v1": backend or "",
        "lynx_hist_low_percentile_v1": None,
        "lynx_hist_high_percentile_v1": None,
        "lynx_clahe_clip_limit_v1": None,
        "lynx_clahe_grid_size_v1": None,
    }
    if image_shape is not None:
        payload["lynx_image_height_v1"] = int(image_shape[0])
        payload["lynx_image_width_v1"] = int(image_shape[1])
    if low_percentile is not None:
        payload["lynx_hist_low_percentile_v1"] = _as_float(low_percentile, digits=4)
    if high_percentile is not None:
        payload["lynx_hist_high_percentile_v1"] = _as_float(high_percentile, digits=4)
    if clip_limit is not None:
        payload["lynx_clahe_clip_limit_v1"] = _as_float(clip_limit, digits=4)
    if grid_size is not None:
        payload["lynx_clahe_grid_size_v1"] = int(grid_size)
    if contrast_stats:
        payload.update(contrast_stats)
    return payload


def build_qualitative_lynx_view(
    image: Image.Image | np.ndarray,
    *,
    mode: str,
    hist_low_percentile: float = DEFAULT_HIST_LOW_PERCENTILE,
    hist_high_percentile: float = DEFAULT_HIST_HIGH_PERCENTILE,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    clahe_grid_size: int = DEFAULT_CLAHE_GRID_SIZE,
    include_contrast_stats: bool = True,
    mask: np.ndarray | None = None,
) -> tuple[Image.Image, dict[str, object]]:
    normalized_mode = str(mode).strip().lower()
    gray = to_grayscale_uint8(image)

    if normalized_mode == "gray":
        normalized = gray.copy()
        backend = "pil_luma"
    elif normalized_mode == "hist_norm":
        normalized = histogram_normalize_gray(
            gray,
            low_percentile=hist_low_percentile,
            high_percentile=hist_high_percentile,
            mask=mask,
        )
        backend = "percentile_histogram"
    elif normalized_mode == "clahe":
        normalized = clahe_normalize_gray(
            gray,
            clip_limit=clahe_clip_limit,
            grid_size=clahe_grid_size,
        )
        backend = "opencv_clahe" if _load_cv2() is not None else "pil_autocontrast_fallback"
    else:
        raise ValueError(f"unsupported Lynx normalization mode: {mode}")

    contrast_stats = compute_lightweight_contrast_stats(normalized, mask=mask) if include_contrast_stats else None
    metadata = build_lynx_normalization_metadata(
        mode=normalized_mode,
        low_percentile=hist_low_percentile if normalized_mode == "hist_norm" else None,
        high_percentile=hist_high_percentile if normalized_mode == "hist_norm" else None,
        clip_limit=clahe_clip_limit if normalized_mode == "clahe" else None,
        grid_size=clahe_grid_size if normalized_mode == "clahe" else None,
        backend=backend,
        contrast_stats=contrast_stats,
        image_shape=normalized.shape,
    )
    return Image.fromarray(normalized, mode="L"), metadata


__all__ = [
    "DEFAULT_CLAHE_CLIP_LIMIT",
    "DEFAULT_CLAHE_GRID_SIZE",
    "DEFAULT_HIST_HIGH_PERCENTILE",
    "DEFAULT_HIST_LOW_PERCENTILE",
    "LYNX_NORMALIZATION_MODES",
    "build_lynx_normalization_metadata",
    "build_qualitative_lynx_view",
    "clahe_normalize_gray",
    "compute_lightweight_contrast_stats",
    "histogram_normalize_gray",
    "to_grayscale_uint8",
]
