from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from src.animalclef_analysis.texas_tail_prior_review import (
    extract_black_pattern_mask,
    infer_tail_side_from_mask,
)


class TexasTailPriorReviewTest(unittest.TestCase):
    def test_infer_tail_side_from_mask_picks_narrower_left_end(self) -> None:
        mask = np.zeros((40, 120), dtype=np.uint8)
        mask[15:25, 0:18] = 1
        mask[10:30, 18:96] = 1
        mask[7:33, 96:120] = 1

        payload = infer_tail_side_from_mask(mask, edge_fraction=0.2, min_columns=12, balance_margin=0.05)

        self.assertEqual(payload["tail_side"], "left")
        self.assertGreater(float(payload["right_span"]), float(payload["left_span"]))

    def test_extract_black_pattern_mask_keeps_dark_spots_inside_foreground(self) -> None:
        rgb = np.full((32, 32, 3), 220, dtype=np.uint8)
        rgb[8:12, 8:12] = 30
        rgb[18:22, 20:24] = 40
        image = Image.fromarray(rgb, mode="RGB")
        foreground = np.ones((32, 32), dtype=np.uint8)

        black_mask, payload = extract_black_pattern_mask(image, foreground)

        self.assertGreater(int(black_mask.sum()), 0)
        self.assertGreater(float(payload["black_ratio"]), 0.0)
        self.assertEqual(int(black_mask[9, 9]), 1)


if __name__ == "__main__":
    unittest.main()
