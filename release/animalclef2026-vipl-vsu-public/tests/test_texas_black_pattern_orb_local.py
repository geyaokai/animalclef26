from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.texas_black_pattern_orb_local import (
    build_black_pattern_band_mask,
    build_core_body_mask,
    extract_black_pattern_mask,
    merge_texas_black_pattern_orb_local_scores,
)


class TexasBlackPatternOrbLocalTest(unittest.TestCase):
    def test_build_core_body_mask_erodes_rectangle(self) -> None:
        mask = np.zeros((40, 60), dtype=np.uint8)
        mask[8:32, 12:48] = 1

        core_mask, payload = build_core_body_mask(mask, erode_ratio=0.15, min_pixels=64)

        self.assertGreater(int(mask.sum()), int(core_mask.sum()))
        self.assertTrue(bool(core_mask[20, 30]))
        self.assertGreater(int(payload["core_erode_radius"]), 0)

    def test_extract_black_pattern_mask_ignores_black_background(self) -> None:
        rgb = np.zeros((60, 80, 3), dtype=np.uint8)
        rgb[12:48, 18:62] = np.array([170, 120, 90], dtype=np.uint8)
        rgb[20:26, 28:34] = np.array([8, 8, 8], dtype=np.uint8)
        rgb[31:37, 44:50] = np.array([16, 16, 16], dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        foreground_mask = np.zeros((60, 80), dtype=np.uint8)
        foreground_mask[12:48, 18:62] = 1

        black_mask, payload = extract_black_pattern_mask(
            image,
            foreground_mask,
            core_mask=foreground_mask,
            fallback_quantile=0.2,
            max_quantile=0.3,
        )

        self.assertEqual(int(black_mask[:8, :].sum()), 0)
        self.assertGreater(int(black_mask[20:26, 28:34].sum()), 0)
        self.assertGreaterEqual(float(payload["black_ratio_foreground"]), 0.0)

    def test_build_black_pattern_band_mask_stays_inside_core(self) -> None:
        core = np.zeros((48, 64), dtype=np.uint8)
        core[10:38, 12:52] = 1
        black = np.zeros((48, 64), dtype=np.uint8)
        black[18:30, 24:40] = 1

        band_mask, payload = build_black_pattern_band_mask(black, core_mask=core, min_pixels=16)

        self.assertGreater(int(band_mask.sum()), 0)
        self.assertEqual(int((band_mask > core).sum()), 0)
        self.assertIn(str(payload["band_mode"]), {"ring", "dilate", "black"})

    def test_merge_texas_black_pattern_orb_local_scores_keeps_previous_local_score(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["15209"],
                "neighbor_image_id": ["15210"],
                "local_score": [0.61],
            }
        )
        local_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "image_id": ["15209"],
                "neighbor_image_id": ["15210"],
                "pair_score": [0.72],
                "pair_score_col": ["route_global_score"],
                "black_orb_matcher_name": ["orb"],
                "black_orb_left_keypoints": [32],
                "black_orb_right_keypoints": [31],
                "black_orb_good_matches": [9],
                "black_orb_inliers": [6],
                "black_orb_local_raw_score": [0.18],
                "black_orb_local_score": [0.37],
                "left_black_ratio": [0.12],
                "right_black_ratio": [0.13],
                "left_orb_region_kind": ["band"],
                "right_orb_region_kind": ["band"],
            }
        )

        merged = merge_texas_black_pattern_orb_local_scores(pair_df, local_df)

        self.assertAlmostEqual(float(merged.iloc[0]["local_score"]), 0.61, places=6)
        self.assertAlmostEqual(float(merged.iloc[0]["black_orb_local_score"]), 0.37, places=6)
        self.assertEqual(str(merged.iloc[0]["black_orb_matcher_name"]), "orb")


if __name__ == "__main__":
    unittest.main()
