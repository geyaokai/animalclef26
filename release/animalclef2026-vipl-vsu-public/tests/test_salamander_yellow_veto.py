from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.salamander_yellow_veto import (
    apply_yellow_veto_penalty_as_score,
    build_threshold_delta_table,
    compile_yellow_veto_decisions,
    compute_yellow_features,
    extract_yellow_mask,
)


class SalamanderYellowVetoTest(unittest.TestCase):
    def test_extract_yellow_mask_finds_yellow_band(self) -> None:
        rgb = np.zeros((12, 18, 3), dtype=np.uint8)
        rgb[:, 4:12] = np.array([220, 200, 40], dtype=np.uint8)
        rgb[:, 12:14] = np.array([10, 100, 220], dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB")

        yellow = extract_yellow_mask(image, min_component_pixels=4)

        self.assertGreater(int(yellow.sum()), 0)
        self.assertEqual(int(yellow[:, 12:14].sum()), 0)

    def test_compute_yellow_features_builds_normalized_profile(self) -> None:
        yellow = np.zeros((10, 20), dtype=np.uint8)
        yellow[:, 5:10] = 1
        foreground = np.ones_like(yellow, dtype=np.uint8)

        features = compute_yellow_features(yellow, foreground_mask=foreground, profile_bins=4, min_yellow_pixels=4)

        profile = features["yellow_profile"]
        self.assertEqual(profile.shape[0], 4)
        self.assertAlmostEqual(float(profile.sum()), 1.0, places=6)
        self.assertTrue(bool(features["yellow_quality_flag"]))

    def test_compile_yellow_veto_decisions_marks_hard_soft_and_support(self) -> None:
        pair_df = pd.DataFrame(
            {
                "yellow_quality_both_v1": [True, True, True],
                "yellow_profile_corr_v1": [0.20, 0.50, 0.95],
                "yellow_profile_l1_distance_v1": [1.20, 0.80, 0.10],
                "yellow_area_ratio_gap_v1": [0.10, 0.04, 0.01],
                "yellow_component_count_gap_v1": [2, 1, 0],
            }
        )

        result = compile_yellow_veto_decisions(pair_feature_df=pair_df)

        self.assertEqual(result["yellow_veto_decision_v1"].tolist(), ["hard_veto", "soft_veto", "support"])

    def test_apply_yellow_veto_penalty_as_score_caps_and_scales_pairs(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.8, 0.7],
                [0.8, 1.0, 0.6],
                [0.7, 0.6, 1.0],
            ],
            dtype=np.float32,
        )
        decision_df = pd.DataFrame(
            {
                "left_index": [0, 1],
                "right_index": [1, 2],
                "yellow_veto_decision_v1": ["hard_veto", "soft_veto"],
            }
        )

        updated = apply_yellow_veto_penalty_as_score(
            base_score=score,
            decision_df=decision_df,
            hard_veto_score_cap=0.02,
            soft_veto_score_scale=0.5,
        )

        self.assertAlmostEqual(float(updated[0, 1]), 0.02, places=6)
        self.assertAlmostEqual(float(updated[1, 2]), 0.3, places=6)

    def test_build_threshold_delta_table_computes_metric_deltas(self) -> None:
        baseline_df = pd.DataFrame(
            {
                "dataset": ["SalamanderID2025"],
                "threshold": [0.25],
                "ari": [0.4],
                "pairwise_f1": [0.5],
                "cluster_count": [100],
            }
        )
        veto_df = pd.DataFrame(
            {
                "dataset": ["SalamanderID2025"],
                "threshold": [0.25],
                "ari": [0.42],
                "pairwise_f1": [0.53],
                "cluster_count": [104],
            }
        )

        delta_df = build_threshold_delta_table(baseline_df=baseline_df, veto_df=veto_df)

        self.assertAlmostEqual(float(delta_df.iloc[0]["delta_ari"]), 0.02, places=6)
        self.assertAlmostEqual(float(delta_df.iloc[0]["delta_pairwise_f1"]), 0.03, places=6)
        self.assertAlmostEqual(float(delta_df.iloc[0]["delta_cluster_count"]), 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
