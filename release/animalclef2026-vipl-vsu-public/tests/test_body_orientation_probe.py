from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.animalclef_analysis.body_orientation_probe import (
    PROMPTS_BY_DATASET,
    OrientationDecision,
    compute_body_axis,
    decide_orientation_application,
    ndimage,
    extract_largest_component,
    normalize_axis_angle_deg,
    resolve_crop_padding_ratio,
    rotate_and_crop,
)


class BodyOrientationProbeTest(unittest.TestCase):
    def test_prompts_cover_expected_datasets(self) -> None:
        self.assertEqual(
            set(PROMPTS_BY_DATASET),
            {"LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"},
        )

    def test_normalize_axis_angle_deg_wraps_to_unsigned_axis_range(self) -> None:
        self.assertAlmostEqual(normalize_axis_angle_deg(170.0), -10.0)
        self.assertAlmostEqual(normalize_axis_angle_deg(95.0), -85.0)
        self.assertAlmostEqual(normalize_axis_angle_deg(-100.0), 80.0)

    def test_extract_largest_component_keeps_biggest_blob(self) -> None:
        mask = np.zeros((12, 12), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        mask[6:11, 6:11] = 1
        largest = extract_largest_component(mask)
        self.assertEqual(int(largest.sum()), 25)
        self.assertEqual(int(largest[1:3, 1:3].sum()), 0)

    def test_compute_body_axis_horizontal_rectangle(self) -> None:
        mask = np.zeros((30, 50), dtype=np.uint8)
        mask[12:18, 6:40] = 1
        stats = compute_body_axis(mask)
        assert stats is not None
        self.assertLess(abs(stats["axis_angle_deg"]), 1.0)
        self.assertGreater(stats["axis_confidence"], 0.6)
        self.assertGreater(stats["major_extent_px"], stats["minor_extent_px"])

    def test_decide_orientation_application_gates_low_confidence(self) -> None:
        stats = {
            "foreground_pixels": 2500.0,
            "foreground_area_ratio": 0.12,
            "axis_confidence": 0.18,
        }
        decision = decide_orientation_application(
            stats,
            min_foreground_pixels=1024,
            min_area_ratio=0.015,
            max_area_ratio=0.85,
            min_axis_confidence=0.35,
            min_largest_component_ratio=0.8,
            largest_component_ratio=0.95,
        )
        self.assertIsInstance(decision, OrientationDecision)
        self.assertFalse(decision.should_apply)
        self.assertEqual(decision.reason, "low_axis_confidence")

    def test_resolve_crop_padding_ratio_uses_texas_override(self) -> None:
        self.assertAlmostEqual(
            resolve_crop_padding_ratio(
                "TexasHornedLizards",
                default_padding_ratio=0.06,
                padding_ratio_overrides={"TexasHornedLizards": 0.12},
            ),
            0.12,
        )
        self.assertAlmostEqual(
            resolve_crop_padding_ratio(
                "SalamanderID2025",
                default_padding_ratio=0.06,
                padding_ratio_overrides={"TexasHornedLizards": 0.12},
            ),
            0.06,
        )

    def test_rotate_and_crop_can_preserve_background(self) -> None:
        image_arr = np.zeros((6, 6, 3), dtype=np.uint8)
        image_arr[:, :] = np.array([10, 20, 30], dtype=np.uint8)
        image_arr[2:4, 2:4] = np.array([200, 180, 160], dtype=np.uint8)
        image = Image.fromarray(image_arr, mode="RGB")
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2:4, 2:4] = 1

        kept_crop, _ = rotate_and_crop(
            image,
            mask,
            0.0,
            padding_ratio=0.06,
            keep_background=True,
        )
        masked_crop, _ = rotate_and_crop(
            image,
            mask,
            0.0,
            padding_ratio=0.06,
            keep_background=False,
        )

        kept_arr = np.asarray(kept_crop)
        masked_arr = np.asarray(masked_crop)
        self.assertTrue(np.array_equal(kept_arr[0, 0], np.array([10, 20, 30], dtype=np.uint8)))
        self.assertTrue(np.array_equal(masked_arr[0, 0], np.array([0, 0, 0], dtype=np.uint8)))

    @unittest.skipIf(ndimage is None, "scipy.ndimage not available")
    def test_rotate_and_crop_edge_fill_avoids_black_rotation_corners(self) -> None:
        image_arr = np.zeros((24, 24, 3), dtype=np.uint8)
        image_arr[:, :] = np.array([40, 60, 80], dtype=np.uint8)
        image_arr[7:17, 7:17] = np.array([180, 160, 140], dtype=np.uint8)
        image = Image.fromarray(image_arr, mode="RGB")
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 7:17] = 1

        rotated_crop, _ = rotate_and_crop(
            image,
            mask,
            45.0,
            padding_ratio=0.4,
            keep_background=True,
            canvas_fill_mode="edge",
        )

        rotated_arr = np.asarray(rotated_crop)
        self.assertFalse(np.any(np.all(rotated_arr == np.array([0, 0, 0], dtype=np.uint8), axis=-1)))


if __name__ == "__main__":
    unittest.main()
