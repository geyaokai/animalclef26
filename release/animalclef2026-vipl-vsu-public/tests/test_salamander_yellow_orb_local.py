from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.salamander_yellow_orb_local import (
    compile_yellow_orb_local_decisions,
    compute_patch_descriptor,
    compute_patch_pair_metrics,
    extract_yellow_focus_crop,
    merge_yellow_orb_local_pair_features,
)


class SalamanderYellowOrbLocalTest(unittest.TestCase):
    def test_extract_yellow_focus_crop_expands_around_mask(self) -> None:
        rgb = np.zeros((40, 80, 3), dtype=np.uint8)
        rgb[10:24, 20:50] = np.array([220, 180, 60], dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")
        yellow = np.zeros((40, 80), dtype=np.uint8)
        yellow[12:20, 28:42] = 1

        focus_image, focus_mask, payload = extract_yellow_focus_crop(
            image,
            yellow,
            context_ratio_x=0.25,
            context_ratio_y=0.25,
            min_side=24,
        )

        self.assertIsNotNone(focus_image)
        self.assertTrue(bool(payload["focus_available"]))
        self.assertGreaterEqual(int(payload["focus_width"]), 24)
        self.assertGreaterEqual(int(payload["focus_height"]), 24)
        self.assertEqual(int(focus_mask.sum()), int(yellow[12:20, 28:42].sum()))

    def test_compute_patch_pair_metrics_prefers_similar_patterns(self) -> None:
        rgb_a = np.zeros((48, 96, 3), dtype=np.uint8)
        rgb_b = np.zeros((48, 96, 3), dtype=np.uint8)
        rgb_c = np.zeros((48, 96, 3), dtype=np.uint8)
        rgb_a[10:34, 18:74] = np.array([210, 180, 50], dtype=np.uint8)
        rgb_b[10:34, 18:74] = np.array([205, 176, 48], dtype=np.uint8)
        rgb_c[10:34, 18:74] = np.array([80, 80, 80], dtype=np.uint8)
        rgb_a[18:22, 32:36] = 0
        rgb_b[18:22, 33:37] = 0
        rgb_c[18:22, 52:56] = 0

        yellow_a = np.zeros((48, 96), dtype=np.uint8)
        yellow_b = np.zeros((48, 96), dtype=np.uint8)
        yellow_c = np.zeros((48, 96), dtype=np.uint8)
        yellow_a[10:34, 18:74] = 1
        yellow_b[10:34, 18:74] = 1
        yellow_c[10:34, 18:74] = 1

        desc_a = compute_patch_descriptor(Image.fromarray(rgb_a, mode="RGB"), yellow_a)
        desc_b = compute_patch_descriptor(Image.fromarray(rgb_b, mode="RGB"), yellow_b)
        desc_c = compute_patch_descriptor(Image.fromarray(rgb_c, mode="RGB"), yellow_c)

        similar = compute_patch_pair_metrics(desc_a, desc_b)
        mismatch = compute_patch_pair_metrics(desc_a, desc_c)

        self.assertTrue(bool(similar["yellow_patch_pair_valid_v1"]))
        self.assertGreater(float(similar["yellow_patch_gray_corr_v1"]), float(mismatch["yellow_patch_gray_corr_v1"]))
        self.assertLess(float(similar["yellow_patch_gray_absdiff_v1"]), float(mismatch["yellow_patch_gray_absdiff_v1"]))

    def test_compile_yellow_orb_local_decisions_marks_support_and_veto(self) -> None:
        base_pair_df = pd.DataFrame(
            {
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "left_index": [0, 1],
                "right_index": [2, 3],
                "image_id": ["a", "b"],
                "neighbor_image_id": ["c", "d"],
                "same_identity": [True, False],
                "local_score": [0.0, 0.0],
                "local_raw_score": [0.0, 0.0],
                "inliers": [0, 0],
                "good_matches": [0, 0],
                "left_keypoints": [0, 0],
                "right_keypoints": [0, 0],
            }
        )
        yellow_roi_local_df = pd.DataFrame(
            {
                "left_index": [0, 1],
                "right_index": [2, 3],
                "image_id": ["a", "b"],
                "neighbor_image_id": ["c", "d"],
                "yellow_roi_left_keypoints": [64, 48],
                "yellow_roi_right_keypoints": [63, 47],
                "yellow_roi_good_matches": [16, 2],
                "yellow_roi_inliers": [10, 0],
                "yellow_roi_local_raw_score": [0.25, 0.0],
                "yellow_roi_local_score": [0.42, 0.0],
                "yellow_roi_keypoint_min": [63, 47],
            }
        )
        patch_pair_df = pd.DataFrame(
            {
                "left_index": [0, 1],
                "right_index": [2, 3],
                "image_id": ["a", "b"],
                "neighbor_image_id": ["c", "d"],
                "yellow_patch_pair_valid_v1": [True, True],
                "yellow_patch_overlap_pixels_v1": [220, 220],
                "yellow_patch_gray_corr_v1": [0.96, 0.30],
                "yellow_patch_gray_absdiff_v1": [0.05, 0.31],
                "yellow_patch_mask_iou_v1": [0.72, 0.05],
                "yellow_patch_mask_dice_v1": [0.82, 0.09],
                "yellow_patch_profile_corr_v1": [0.92, 0.12],
                "yellow_patch_profile_l1_v1": [0.10, 1.30],
            }
        )
        focus_df = pd.DataFrame(
            {
                "image_id": ["a", "b", "c", "d"],
                "dataset": ["SalamanderID2025"] * 4,
                "yellow_quality_flag_v1": [True, True, True, True],
                "yellow_focus_available_v1": [True, True, True, True],
                "yellow_focus_source_kind_v1": ["aligned", "aligned", "aligned", "aligned"],
            }
        )

        merged = merge_yellow_orb_local_pair_features(
            base_pair_df=base_pair_df,
            yellow_roi_local_df=yellow_roi_local_df,
            patch_pair_df=patch_pair_df,
        )
        decisions = compile_yellow_orb_local_decisions(pair_feature_df=merged, focus_df=focus_df)

        self.assertEqual(decisions["yellow_veto_decision_v1"].tolist(), ["support", "hard_veto"])


if __name__ == "__main__":
    unittest.main()
