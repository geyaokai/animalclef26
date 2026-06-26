from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.singleton_rescue_compiler import compile_singleton_rescue_merge_judgments


class SingletonRescueCompilerTest(unittest.TestCase):
    def test_compile_singleton_rescue_merge_judgments(self) -> None:
        merge_candidate_df = pd.DataFrame(
            [
                {
                    "cluster_pair_key": "11|12",
                    "left_cluster_id": 11,
                    "right_cluster_id": 12,
                    "left_cluster_size": 1,
                    "right_cluster_size": 1,
                    "merged_total_size": 2,
                    "support_pair_count": 1,
                    "mean_pair_probability": 0.99,
                    "max_pair_probability": 0.99,
                    "candidate_kind": "singleton_singleton",
                    "candidate_preview": "x1|x2",
                    "support_image_ids": "x1|x2",
                    "origin_cluster_id": 5,
                },
                {
                    "cluster_pair_key": "21|31",
                    "left_cluster_id": 21,
                    "right_cluster_id": 31,
                    "left_cluster_size": 1,
                    "right_cluster_size": 2,
                    "merged_total_size": 3,
                    "support_pair_count": 2,
                    "mean_pair_probability": 0.95,
                    "max_pair_probability": 0.98,
                    "candidate_kind": "singleton_attach",
                    "candidate_preview": "y1 -> z1|z2",
                    "singleton_image_id": "y1",
                    "support_image_ids": "z1|z2",
                    "origin_cluster_id": 8,
                },
            ]
        )
        pair_df = pd.DataFrame(
            [
                {
                    "cluster_pair_key": "11|12",
                    "image_id": "x1",
                    "neighbor_image_id": "x2",
                    "xgb_same_identity_prob": 0.99,
                    "local_score": 0.3,
                    "route_global_score": 0.6,
                },
                {
                    "cluster_pair_key": "21|31",
                    "image_id": "y1",
                    "neighbor_image_id": "z1",
                    "xgb_same_identity_prob": 0.98,
                    "local_score": 0.4,
                    "route_global_score": 0.7,
                },
                {
                    "cluster_pair_key": "21|31",
                    "image_id": "y1",
                    "neighbor_image_id": "z2",
                    "xgb_same_identity_prob": 0.92,
                    "local_score": 0.2,
                    "route_global_score": 0.5,
                },
            ]
        )
        judgments = [
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "merge",
                "candidate_key": "11|12",
                "image_id": "x1",
                "neighbor_image_id": "x2",
                "label": "yes",
            },
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "merge",
                "candidate_key": "21|31",
                "image_id": "y1",
                "neighbor_image_id": "z1",
                "label": "yes",
            },
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "merge",
                "candidate_key": "21|31",
                "image_id": "y1",
                "neighbor_image_id": "z2",
                "label": "yes",
            },
        ]

        result = compile_singleton_rescue_merge_judgments(merge_candidate_df, pair_df, judgments)
        self.assertEqual(len(result.operations), 2)
        attach_ops = [item for item in result.operations if item["action"] == "attach_to_anchor"]
        self.assertEqual(len(attach_ops), 2)
        self.assertEqual(attach_ops[0]["anchor_image_id"], "x1")
        self.assertEqual(attach_ops[0]["member_image_ids"], ["x2"])
        self.assertEqual(attach_ops[1]["anchor_image_id"], "z1")
        self.assertEqual(attach_ops[1]["member_image_ids"], ["y1"])
        summary = result.candidate_summary_df.set_index("candidate_key")
        self.assertEqual(summary.loc["11|12", "compile_status"], "accepted_all_yes")
        self.assertEqual(summary.loc["21|31", "compile_status"], "accepted_all_yes")


if __name__ == "__main__":
    unittest.main()
