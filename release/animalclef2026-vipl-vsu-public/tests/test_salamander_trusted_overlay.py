from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.salamander_trusted_overlay import apply_salamander_trusted_overlay


class SalamanderTrustedOverlayTest(unittest.TestCase):
    def test_clean_component_attach_merges_only_trusted_members(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "a", "dataset": "SalamanderID2025", "pred_cluster_id": 1, "cluster_label": "cluster_SalamanderID2025_1"},
                {"image_id": "b", "dataset": "SalamanderID2025", "pred_cluster_id": 2, "cluster_label": "cluster_SalamanderID2025_2"},
                {"image_id": "c", "dataset": "SalamanderID2025", "pred_cluster_id": 2, "cluster_label": "cluster_SalamanderID2025_2"},
            ]
        )
        clean_membership_df = pd.DataFrame(
            [
                {"component_id": "comp_001", "image_id": "a", "dataset": "SalamanderID2025", "manual_yes_degree": 1, "manual_no_degree": 0},
                {"component_id": "comp_001", "image_id": "b", "dataset": "SalamanderID2025", "manual_yes_degree": 1, "manual_no_degree": 0},
            ]
        )
        cannot_link_df = pd.DataFrame(columns=["pair_key", "image_id", "neighbor_image_id"])

        result = apply_salamander_trusted_overlay(
            pred_df=pred_df,
            clean_membership_df=clean_membership_df,
            cannot_link_df=cannot_link_df,
        )
        lookup = dict(zip(result.prediction_df["image_id"], result.prediction_df["pred_cluster_id"], strict=False))
        self.assertEqual(int(lookup["a"]), int(lookup["b"]))
        self.assertEqual(int(lookup["c"]), 2)
        self.assertEqual(len(result.operation_df), 1)
        self.assertEqual(len(result.changed_df), 1)

    def test_cannot_link_violations_are_audited_without_default_split(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "a", "dataset": "SalamanderID2025", "pred_cluster_id": 1, "cluster_label": "cluster_SalamanderID2025_1"},
                {"image_id": "b", "dataset": "SalamanderID2025", "pred_cluster_id": 1, "cluster_label": "cluster_SalamanderID2025_1"},
            ]
        )
        clean_membership_df = pd.DataFrame(columns=["component_id", "image_id", "dataset", "manual_yes_degree", "manual_no_degree"])
        cannot_link_df = pd.DataFrame([{"pair_key": "a|b", "image_id": "a", "neighbor_image_id": "b"}])

        result = apply_salamander_trusted_overlay(
            pred_df=pred_df,
            clean_membership_df=clean_membership_df,
            cannot_link_df=cannot_link_df,
        )
        self.assertEqual(len(result.operation_df), 0)
        self.assertEqual(len(result.cannot_link_violation_df), 1)
        self.assertEqual(int(result.summary_df["cannot_link_violations_after"].max()), 1)


if __name__ == "__main__":
    unittest.main()
