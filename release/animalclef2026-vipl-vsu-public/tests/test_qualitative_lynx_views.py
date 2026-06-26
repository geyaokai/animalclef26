from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image

from src.animalclef_analysis.qualitative_lynx_views import (
    build_lynx_normalization_metadata,
    build_qualitative_lynx_view,
    clahe_normalize_gray,
    compute_lightweight_contrast_stats,
    histogram_normalize_gray,
    to_grayscale_uint8,
)


class QualitativeLynxViewsTest(unittest.TestCase):
    def test_to_grayscale_uint8_converts_rgb_array(self) -> None:
        rgb = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 255]],
            ],
            dtype=np.uint8,
        )

        gray = to_grayscale_uint8(rgb)

        self.assertEqual(gray.shape, (2, 2))
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(int(gray[1, 1]), 255)

    def test_histogram_normalize_gray_stretches_percentile_window(self) -> None:
        gray = np.array(
            [
                [20, 25, 30],
                [35, 40, 45],
                [50, 55, 60],
            ],
            dtype=np.uint8,
        )

        normalized = histogram_normalize_gray(gray, low_percentile=10.0, high_percentile=90.0)

        self.assertEqual(int(normalized.min()), 0)
        self.assertEqual(int(normalized.max()), 255)
        self.assertGreater(int(normalized[1, 1]), int(normalized[0, 0]))

    def test_clahe_normalize_gray_has_safe_fallback_without_cv2(self) -> None:
        gray = np.array(
            [
                [100, 102, 104],
                [106, 108, 110],
                [112, 114, 116],
            ],
            dtype=np.uint8,
        )

        with mock.patch("src.animalclef_analysis.qualitative_lynx_views._load_cv2", return_value=None):
            normalized = clahe_normalize_gray(gray, clip_limit=2.0, grid_size=8)

        self.assertEqual(normalized.shape, gray.shape)
        self.assertEqual(normalized.dtype, np.uint8)
        self.assertGreaterEqual(int(normalized.max()), int(gray.max()))

    def test_compute_lightweight_contrast_stats_reports_expected_keys(self) -> None:
        gray = np.array(
            [
                [0, 10, 20],
                [30, 40, 50],
                [60, 70, 80],
            ],
            dtype=np.uint8,
        )

        stats = compute_lightweight_contrast_stats(gray)

        self.assertEqual(
            set(stats),
            {
                "gray_mean",
                "gray_std",
                "gray_p05",
                "gray_p50",
                "gray_p95",
                "gray_contrast_p95_p05",
                "gray_sharpness",
            },
        )
        self.assertGreater(stats["gray_contrast_p95_p05"], 0.0)

    def test_build_lynx_normalization_metadata_records_mode_and_params(self) -> None:
        metadata = build_lynx_normalization_metadata(
            mode="clahe",
            clip_limit=2.5,
            grid_size=10,
            backend="pil_autocontrast_fallback",
            image_shape=(48, 64),
        )

        self.assertEqual(metadata["lynx_normalization_mode_v1"], "clahe")
        self.assertEqual(metadata["lynx_normalization_backend_v1"], "pil_autocontrast_fallback")
        self.assertEqual(metadata["lynx_clahe_grid_size_v1"], 10)
        self.assertEqual(metadata["lynx_image_width_v1"], 64)

    def test_build_qualitative_lynx_view_returns_metadata_for_fallback_clahe(self) -> None:
        image = Image.fromarray(
            np.array(
                [
                    [[40, 30, 20], [80, 70, 60]],
                    [[120, 110, 100], [180, 170, 160]],
                ],
                dtype=np.uint8,
            ),
            mode="RGB",
        )

        with mock.patch("src.animalclef_analysis.qualitative_lynx_views._load_cv2", return_value=None):
            normalized, metadata = build_qualitative_lynx_view(image, mode="clahe")

        self.assertEqual(normalized.mode, "L")
        self.assertEqual(metadata["lynx_normalization_mode_v1"], "clahe")
        self.assertEqual(metadata["lynx_normalization_backend_v1"], "pil_autocontrast_fallback")
        self.assertIn("gray_contrast_p95_p05", metadata)


if __name__ == "__main__":
    unittest.main()
