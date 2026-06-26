from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.singleton_rescue_review import build_singleton_rescue_review


class SingletonRescueReviewTest(unittest.TestCase):
    def test_build_singleton_rescue_review_extracts_strict_candidates(self) -> None:
        pred_df = pd.DataFrame(
            [
                {
                    "image_id": "a1",
                    "dataset": "SalamanderID2025",
                    "path": "images/a1.jpg",
                    "pred_cluster_id": 101,
                    "cluster_label": "cluster_SalamanderID2025_101",
                    "manual_overlay_enabled": True,
                    "manual_overlay_note": "compiled from pair judgments | cluster=10 | moved=a1",
                },
                {
                    "image_id": "a2",
                    "dataset": "SalamanderID2025",
                    "path": "images/a2.jpg",
                    "pred_cluster_id": 102,
                    "cluster_label": "cluster_SalamanderID2025_102",
                    "manual_overlay_enabled": True,
                    "manual_overlay_note": "compiled from pair judgments | cluster=10 | moved=a2",
                },
                {
                    "image_id": "b1",
                    "dataset": "SalamanderID2025",
                    "path": "images/b1.jpg",
                    "pred_cluster_id": 201,
                    "cluster_label": "cluster_SalamanderID2025_201",
                    "manual_overlay_enabled": True,
                    "manual_overlay_note": "compiled from pair judgments | cluster=20 | moved=b1",
                },
                {
                    "image_id": "c1",
                    "dataset": "SalamanderID2025",
                    "path": "images/c1.jpg",
                    "pred_cluster_id": 301,
                    "cluster_label": "cluster_SalamanderID2025_301",
                    "manual_overlay_enabled": False,
                    "manual_overlay_note": "",
                },
                {
                    "image_id": "c2",
                    "dataset": "SalamanderID2025",
                    "path": "images/c2.jpg",
                    "pred_cluster_id": 301,
                    "cluster_label": "cluster_SalamanderID2025_301",
                    "manual_overlay_enabled": False,
                    "manual_overlay_note": "",
                },
            ]
        )
        pair_feature_df = pd.DataFrame(
            [
                {
                    "image_id": "a1",
                    "neighbor_image_id": "a2",
                    "xgb_same_identity_prob": 0.99,
                    "local_score": 0.50,
                    "route_global_score": 0.80,
                },
                {
                    "image_id": "a1",
                    "neighbor_image_id": "c1",
                    "xgb_same_identity_prob": 0.70,
                    "local_score": 0.10,
                    "route_global_score": 0.20,
                },
                {
                    "image_id": "a2",
                    "neighbor_image_id": "c2",
                    "xgb_same_identity_prob": 0.60,
                    "local_score": 0.10,
                    "route_global_score": 0.20,
                },
                {
                    "image_id": "b1",
                    "neighbor_image_id": "c1",
                    "xgb_same_identity_prob": 0.98,
                    "local_score": 0.40,
                    "route_global_score": 0.70,
                },
                {
                    "image_id": "b1",
                    "neighbor_image_id": "c2",
                    "xgb_same_identity_prob": 0.96,
                    "local_score": 0.30,
                    "route_global_score": 0.60,
                },
            ]
        )

        result = build_singleton_rescue_review(
            pred_df,
            pair_feature_df,
            manual_no_pairs=set(),
            dataset="SalamanderID2025",
            singleton_singleton_min_prob=0.95,
            attach_member_min_prob=0.90,
            attach_min_support_count=2,
            attach_min_mean_prob=0.90,
            attach_min_max_prob=0.95,
        )

        self.assertEqual(len(result.merge_candidate_df), 2)
        self.assertEqual(
            sorted(result.merge_candidate_df["candidate_kind"].astype(str).tolist()),
            ["singleton_attach", "singleton_singleton"],
        )
        attach_row = result.merge_candidate_df[
            result.merge_candidate_df["candidate_kind"].astype(str).eq("singleton_attach")
        ].iloc[0]
        self.assertEqual(int(attach_row["support_pair_count"]), 2)
        self.assertEqual(str(attach_row["candidate_preview"]), "b1 -> c1|c2")
        self.assertEqual(int(result.stats_df[result.stats_df["metric"].eq("accepted_candidate_count")]["value"].iloc[0]), 2)

    def test_manual_no_blocks_candidates(self) -> None:
        pred_df = pd.DataFrame(
            [
                {
                    "image_id": "x1",
                    "dataset": "SalamanderID2025",
                    "path": "images/x1.jpg",
                    "pred_cluster_id": 11,
                    "cluster_label": "cluster_SalamanderID2025_11",
                    "manual_overlay_enabled": True,
                    "manual_overlay_note": "compiled from pair judgments | cluster=1 | moved=x1",
                },
                {
                    "image_id": "x2",
                    "dataset": "SalamanderID2025",
                    "path": "images/x2.jpg",
                    "pred_cluster_id": 12,
                    "cluster_label": "cluster_SalamanderID2025_12",
                    "manual_overlay_enabled": True,
                    "manual_overlay_note": "compiled from pair judgments | cluster=1 | moved=x2",
                },
            ]
        )
        pair_feature_df = pd.DataFrame(
            [
                {
                    "image_id": "x1",
                    "neighbor_image_id": "x2",
                    "xgb_same_identity_prob": 0.99,
                    "local_score": 0.40,
                    "route_global_score": 0.70,
                }
            ]
        )

        result = build_singleton_rescue_review(
            pred_df,
            pair_feature_df,
            manual_no_pairs={("x1", "x2")},
            dataset="SalamanderID2025",
        )

        self.assertTrue(result.merge_candidate_df.empty)
        self.assertEqual(len(result.rejected_candidate_df), 1)
        self.assertEqual(str(result.rejected_candidate_df.iloc[0]["reason"]), "manual_no_conflict")


if __name__ == "__main__":
    unittest.main()
