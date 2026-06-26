from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.sam_orb_veto import (
    apply_veto_penalty_as_score,
    build_threshold_delta_table,
    compile_veto_decisions,
    infer_mask_from_masked_rgb,
)


class SamOrbVetoTest(unittest.TestCase):
    def test_infer_mask_from_masked_rgb_recovers_largest_component(self) -> None:
        array = np.zeros((8, 8, 3), dtype=np.uint8)
        array[1:6, 2:5] = np.array([120, 40, 20], dtype=np.uint8)
        array[0, 0] = np.array([5, 5, 5], dtype=np.uint8)
        image = Image.fromarray(array, mode="RGB")

        mask = infer_mask_from_masked_rgb(image)

        self.assertEqual(mask.shape, (8, 8))
        self.assertEqual(int(mask.sum()), 15)
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertEqual(int(mask[2, 3]), 1)

    def test_compile_veto_decisions_marks_hard_soft_and_support(self) -> None:
        pair_feature_df = pd.DataFrame(
            {
                "left_index": [0, 0, 1],
                "right_index": [1, 2, 2],
                "image_id": ["10", "10", "11"],
                "neighbor_image_id": ["11", "12", "12"],
                "dataset": ["SalamanderID2025"] * 3,
                "same_identity": [0, 0, 1],
                "masked_keypoint_min": [40, 40, 40],
                "aligned_keypoint_min": [40, 40, 40],
                "masked_local_score": [0.01, 0.02, 0.30],
                "aligned_local_score": [0.02, 0.08, 0.25],
                "masked_inliers": [0, 1, 10],
                "aligned_inliers": [0, 2, 12],
            }
        )
        roi_manifest_df = pd.DataFrame(
            {
                "image_id": ["10", "11", "12"],
                "dataset": ["SalamanderID2025"] * 3,
                "sam_orb_veto_masked_available_v1": [True, True, True],
                "sam_orb_veto_alignment_applied_v1": [True, True, True],
                "sam_orb_veto_alignment_status_v1": ["apply", "apply", "apply"],
                "sam_orb_veto_alignment_reason_v1": ["ok", "ok", "ok"],
                "sam_orb_veto_axis_confidence_v1": [0.8, 0.7, 0.9],
            }
        )

        decisions = compile_veto_decisions(pair_feature_df=pair_feature_df, roi_manifest_df=roi_manifest_df)

        self.assertEqual(decisions["veto_decision"].tolist(), ["hard_veto", "soft_veto", "support"])

    def test_apply_veto_penalty_as_score_caps_and_scales_pairs(self) -> None:
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
                "veto_decision": ["hard_veto", "soft_veto"],
            }
        )

        updated = apply_veto_penalty_as_score(base_score=score, decision_df=decision_df, hard_veto_score_cap=0.02, soft_veto_score_scale=0.5)

        self.assertAlmostEqual(float(updated[0, 1]), 0.02, places=6)
        self.assertAlmostEqual(float(updated[1, 0]), 0.02, places=6)
        self.assertAlmostEqual(float(updated[1, 2]), 0.3, places=6)
        self.assertAlmostEqual(float(updated[2, 1]), 0.3, places=6)

    def test_build_threshold_delta_table_computes_metric_deltas(self) -> None:
        baseline_df = pd.DataFrame(
            {
                "dataset": ["SalamanderID2025"],
                "threshold": [0.25],
                "ari": [0.40],
                "pairwise_f1": [0.50],
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
