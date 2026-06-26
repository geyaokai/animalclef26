from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.animalclef_analysis.qualitative_texas_views import (
    TEXAS_YOLO_WORLD_BODY_PROMPT_CANDIDATES,
    TEXAS_YOLO_WORLD_HEAD_PROMPT_CANDIDATES,
    TEXAS_YOLO_WORLD_WHOLE_PROMPT_CANDIDATES,
    build_texas_view_metadata,
    crop_texas_center_body_square,
    crop_texas_middle_band,
    get_texas_yolo_world_prompt_candidates,
    grayscale_normalize_image,
    scale_normalize_aligned_foreground,
)


class QualitativeTexasViewsTest(unittest.TestCase):
    def test_prompt_constants_cover_whole_head_body(self) -> None:
        self.assertIn("Texas horned lizard", TEXAS_YOLO_WORLD_WHOLE_PROMPT_CANDIDATES)
        self.assertIn("horned lizard head", TEXAS_YOLO_WORLD_HEAD_PROMPT_CANDIDATES)
        self.assertIn("horned lizard body", TEXAS_YOLO_WORLD_BODY_PROMPT_CANDIDATES)
        self.assertEqual(
            get_texas_yolo_world_prompt_candidates("body"),
            list(TEXAS_YOLO_WORLD_BODY_PROMPT_CANDIDATES),
        )

    def test_grayscale_normalization_keeps_rgb_output(self) -> None:
        rgb = np.zeros((12, 12, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.linspace(30, 210, 12, dtype=np.uint8)
        rgb[:, :, 1] = 80
        rgb[:, :, 2] = 120
        focus_mask = np.zeros((12, 12), dtype=np.uint8)
        focus_mask[:, 3:9] = 1

        gray_image, payload = grayscale_normalize_image(
            Image.fromarray(rgb, mode="RGB"),
            focus_mask=focus_mask,
        )

        gray = np.asarray(gray_image, dtype=np.uint8)
        self.assertEqual(gray.shape, (12, 12, 3))
        self.assertTrue(np.array_equal(gray[:, :, 0], gray[:, :, 1]))
        self.assertTrue(np.array_equal(gray[:, :, 1], gray[:, :, 2]))
        self.assertEqual(int(payload["focus_pixels"]), int(focus_mask.sum()))

    def test_scale_normalize_aligned_foreground_targets_major_extent(self) -> None:
        rgb = np.zeros((80, 140, 3), dtype=np.uint8)
        rgb[24:56, 20:100] = np.array([160, 120, 80], dtype=np.uint8)
        mask = np.zeros((80, 140), dtype=np.uint8)
        mask[24:56, 20:100] = 1

        scaled_image, scaled_mask, payload = scale_normalize_aligned_foreground(
            Image.fromarray(rgb, mode="RGB"),
            mask,
            canvas_size=(160, 96),
            target_major_extent_ratio=0.50,
        )

        self.assertEqual(scaled_image.size, (160, 96))
        self.assertEqual(scaled_mask.shape, (96, 160))
        self.assertAlmostEqual(float(payload["major_extent_after_px"]), 80.0, delta=5.0)
        self.assertAlmostEqual(float(payload["target_major_extent_px"]), 80.0, delta=0.5)
        self.assertIsNone(payload["fallback_reason"])

    def test_center_body_square_crop_reduces_extremities(self) -> None:
        rgb = np.zeros((96, 160, 3), dtype=np.uint8)
        mask = np.zeros((96, 160), dtype=np.uint8)
        rgb[28:68, 46:114] = np.array([150, 100, 80], dtype=np.uint8)
        mask[28:68, 46:114] = 1

        # Simulate extremities that should be deemphasized by a center-focused crop.
        rgb[42:54, 16:46] = np.array([120, 80, 70], dtype=np.uint8)
        rgb[42:54, 114:144] = np.array([120, 80, 70], dtype=np.uint8)
        rgb[16:30, 68:92] = np.array([120, 80, 70], dtype=np.uint8)
        rgb[68:82, 68:92] = np.array([120, 80, 70], dtype=np.uint8)
        mask[42:54, 16:46] = 1
        mask[42:54, 114:144] = 1
        mask[16:30, 68:92] = 1
        mask[68:82, 68:92] = 1

        crop_image, crop_mask, crop_payload = crop_texas_center_body_square(
            Image.fromarray(rgb, mode="RGB"),
            mask,
            center_percentile=68.0,
            padding_ratio=0.06,
            min_subject_ratio=0.40,
        )

        crop_box = tuple(crop_payload["crop_box_xyxy"])
        self.assertEqual(crop_payload["crop_strategy"], "center_body_square")
        self.assertGreater(crop_box[0], 16)
        self.assertLess(crop_box[2], 144)
        self.assertGreater(crop_payload["foreground_ratio_of_subject"], 0.40)
        self.assertGreater(crop_payload["foreground_ratio_in_crop"], 0.50)
        self.assertEqual(crop_image.size[0], crop_image.size[1])
        self.assertGreater(int(crop_mask.sum()), 0)

    def test_legacy_middle_band_entrypoint_uses_center_square_metadata_builder(self) -> None:
        rgb = np.zeros((72, 160, 3), dtype=np.uint8)
        rgb[20:52, 28:132] = np.array([150, 100, 80], dtype=np.uint8)
        rgb[32:40, 12:28] = np.array([120, 80, 70], dtype=np.uint8)
        mask = np.zeros((72, 160), dtype=np.uint8)
        mask[20:52, 28:132] = 1
        mask[32:40, 12:28] = 1

        crop_image, crop_mask, crop_payload = crop_texas_middle_band(
            Image.fromarray(rgb, mode="RGB"),
            mask,
            band_start_ratio=0.25,
            band_end_ratio=0.75,
            padding_ratio=0.05,
        )

        metadata = build_texas_view_metadata(
            row={"dataset": "TexasHornedLizards", "split": "train", "image_id": "texas-1"},
            view_name="heuristic_middle_band",
            crop_payload=crop_payload,
            scale_payload={
                "axis_angle_deg": 0.2,
                "axis_confidence": 0.92,
                "major_extent_before_px": 100.0,
                "major_extent_after_px": 82.0,
                "target_major_extent_ratio": 0.82,
                "applied_scale_factor": 1.04,
            },
            fallback_reason="sam_missing",
        )

        self.assertGreater(crop_image.size[0], 0)
        self.assertGreater(int(crop_mask.sum()), 0)
        self.assertEqual(metadata["dataset"], "TexasHornedLizards")
        self.assertEqual(metadata["view_name"], "heuristic_middle_band")
        self.assertEqual(metadata["crop_strategy"], "center_body_square")
        self.assertIsNotNone(metadata["crop_center_xy"])
        self.assertGreater(float(metadata["crop_square_side_px"]), 0.0)
        self.assertIn("sam_missing", str(metadata["fallback_reason"]))
        self.assertAlmostEqual(float(metadata["scale_factor"]), 1.04, places=6)


if __name__ == "__main__":
    unittest.main()
