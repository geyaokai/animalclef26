from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.animalclef_analysis.qualitative_salamander_views import (
    SALAMANDER_YOLOWORLD_BODY_PROMPT_CANDIDATES,
    SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES,
    SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES,
    SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES,
    DEFAULT_TRUNK_EXTREMITY_TRIM_RATIO,
    DEFAULT_TRUNK_MINOR_EXTENT_RATIO,
    build_salamander_crop_metadata,
    generate_heuristic_end_middle_crops,
    scale_normalize_aligned_foreground,
)


class QualitativeSalamanderViewsTest(unittest.TestCase):
    def test_prompt_candidates_follow_backoff_order(self) -> None:
        self.assertEqual(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["whole"], SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES)
        self.assertEqual(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["head"], SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES)
        self.assertEqual(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["body"], SALAMANDER_YOLOWORLD_BODY_PROMPT_CANDIDATES)
        self.assertEqual(SALAMANDER_YOLOWORLD_WHOLE_PROMPT_CANDIDATES[0], "fire salamander")
        self.assertEqual(SALAMANDER_YOLOWORLD_HEAD_PROMPT_CANDIDATES[-1], "amphibian head")

    def test_scale_normalize_aligned_foreground_targets_major_extent(self) -> None:
        image_arr = np.zeros((40, 100, 3), dtype=np.uint8)
        image_arr[14:26, 20:60] = np.array([220, 180, 70], dtype=np.uint8)
        image = Image.fromarray(image_arr, mode="RGB")
        mask = np.zeros((40, 100), dtype=np.uint8)
        mask[14:26, 20:60] = 1

        scaled_rgb, scaled_mask, payload = scale_normalize_aligned_foreground(
            image,
            mask,
            output_size=(100, 40),
            target_extent_ratio=0.80,
        )

        self.assertEqual(scaled_rgb.size, (205, 82))
        self.assertEqual(scaled_mask.shape, (82, 205))
        self.assertTrue(payload["scale_applied"])
        self.assertAlmostEqual(payload["target_extent_ratio"], 0.8)
        self.assertGreater(payload["scale_factor"], 2.0)
        self.assertAlmostEqual(payload["desired_scale_factor"], payload["scale_factor"], places=6)
        self.assertIn("expanded_canvas", payload["fallback_reason"])
        self.assertGreater(payload["major_extent_after_px"], payload["major_extent_before_px"])
        self.assertLess(abs(payload["major_extent_after_px"] - 80.0), 4.5)

    def test_generate_heuristic_end_middle_crops_trims_extremities(self) -> None:
        image_arr = np.zeros((60, 160, 3), dtype=np.uint8)
        image_arr[20:40, 30:50] = np.array([30, 60, 220], dtype=np.uint8)
        image_arr[20:40, 50:110] = np.array([0, 220, 80], dtype=np.uint8)
        image_arr[20:40, 110:130] = np.array([220, 60, 30], dtype=np.uint8)
        image = Image.fromarray(image_arr, mode="RGB")
        mask = np.zeros((60, 160), dtype=np.uint8)
        mask[20:40, 30:130] = 1
        scale_payload = {"scale_factor": 1.25, "target_extent_ratio": 0.82}

        crops = generate_heuristic_end_middle_crops(
            image,
            mask,
            scale_payload=scale_payload,
            end_a_ratio=(0.0, 0.2),
            middle_ratio=(0.2, 0.8),
            end_b_ratio=(0.8, 1.0),
            vertical_padding_ratio=0.0,
            horizontal_padding_ratio=0.0,
        )

        self.assertEqual(set(crops), {"end_a", "middle", "end_b"})
        middle_box = crops["middle"].metadata["crop_box"]
        end_a_box = crops["end_a"].metadata["crop_box"]
        end_b_box = crops["end_b"].metadata["crop_box"]
        self.assertGreater(middle_box[0], 30)
        self.assertLess(middle_box[2], 130)
        self.assertGreater(end_a_box[0], 30)
        self.assertLess(end_b_box[2], 130)
        self.assertEqual(crops["middle"].metadata["scale_factor"], 1.25)
        self.assertEqual(crops["middle"].metadata["crop_strategy"], "center_trunk_rectangle")
        self.assertEqual(crops["middle"].metadata["major_axis"], "x")
        self.assertEqual(crops["middle"].metadata["trunk_extremity_trim_ratio"], DEFAULT_TRUNK_EXTREMITY_TRIM_RATIO)
        self.assertEqual(crops["middle"].metadata["trunk_minor_extent_ratio"], DEFAULT_TRUNK_MINOR_EXTENT_RATIO)

        middle_rgb = np.asarray(crops["middle"].rgb, dtype=np.uint8)
        self.assertGreater(float(middle_rgb[..., 1].mean()), float(middle_rgb[..., 0].mean()))
        self.assertGreater(float(middle_rgb[..., 1].mean()), float(middle_rgb[..., 2].mean()))

    def test_generate_heuristic_end_middle_crops_handles_vertical_body(self) -> None:
        image_arr = np.zeros((180, 80, 3), dtype=np.uint8)
        image_arr[20:50, 26:54] = np.array([220, 60, 30], dtype=np.uint8)
        image_arr[50:130, 26:54] = np.array([0, 220, 80], dtype=np.uint8)
        image_arr[130:160, 26:54] = np.array([30, 60, 220], dtype=np.uint8)
        image = Image.fromarray(image_arr, mode="RGB")
        mask = np.zeros((180, 80), dtype=np.uint8)
        mask[20:160, 26:54] = 1

        crops = generate_heuristic_end_middle_crops(
            image,
            mask,
            vertical_padding_ratio=0.0,
            horizontal_padding_ratio=0.0,
        )

        middle_box = crops["middle"].metadata["crop_box"]
        self.assertEqual(crops["middle"].metadata["major_axis"], "y")
        self.assertGreater(middle_box[1], 20)
        self.assertLess(middle_box[3], 160)
        self.assertLess(crops["middle"].rgb.size[0], image.size[0])
        self.assertLess(crops["middle"].rgb.size[1], image.size[1])

    def test_build_salamander_crop_metadata_keeps_fallback_reasons(self) -> None:
        payload = build_salamander_crop_metadata(
            crop_name="middle",
            crop_ratio=(0.3, 0.7),
            scale_payload={"scale_factor": 1.4, "target_extent_ratio": 0.82},
            fallback_reasons=["missing_head_box", "used_heuristic_crop"],
            crop_box=(10, 5, 42, 21),
            extra_metadata={"crop_strategy": "center_trunk_rectangle"},
        )
        self.assertEqual(payload["fallback_reason"], "missing_head_box|used_heuristic_crop")
        self.assertEqual(payload["crop_box"], [10, 5, 42, 21])
        self.assertEqual(payload["crop_width_px"], 32)
        self.assertEqual(payload["crop_height_px"], 16)
        self.assertEqual(payload["crop_strategy"], "center_trunk_rectangle")


if __name__ == "__main__":
    unittest.main()
