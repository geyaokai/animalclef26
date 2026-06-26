from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.graph_cluster_overlay import (
    aggregate_graph_merge_candidates,
    apply_graph_merge_overlay,
    extract_graph_merge_candidates,
    summarize_consensus_merge_candidates,
)


class GraphClusterOverlayTest(unittest.TestCase):
    def test_extract_candidates_only_keeps_cross_cluster_graph_groups(self) -> None:
        base_labels = np.asarray([0, 0, 1, 2, 3], dtype=np.int32)
        graph_labels = np.asarray([0, 0, 1, 1, 2], dtype=np.int32)
        score_matrix = np.asarray(
            [
                [1.0, 0.9, 0.1, 0.1, 0.1],
                [0.9, 1.0, 0.1, 0.1, 0.1],
                [0.1, 0.1, 1.0, 0.93, 0.2],
                [0.1, 0.1, 0.93, 1.0, 0.2],
                [0.1, 0.1, 0.2, 0.2, 1.0],
            ],
            dtype=np.float32,
        )

        candidate_df = extract_graph_merge_candidates(
            base_labels=base_labels,
            graph_labels=graph_labels,
            score_matrix=score_matrix,
            graph_threshold=0.88,
        )

        self.assertEqual(len(candidate_df), 1)
        row = candidate_df.iloc[0]
        self.assertEqual(str(row["member_indices"]), "2|3")
        self.assertEqual(str(row["base_cluster_ids"]), "1|2")
        self.assertAlmostEqual(float(row["mean_score"]), 0.93, places=6)

    def test_aggregate_candidates_counts_stability(self) -> None:
        candidate_df = pd.DataFrame(
            [
                {
                    "graph_threshold": 0.84,
                    "graph_cluster_id": 0,
                    "candidate_key": "2|3",
                    "member_indices": "2|3",
                    "candidate_size": 2,
                    "base_cluster_ids": "1|2",
                    "base_cluster_count": 2,
                    "mean_score": 0.93,
                    "min_score": 0.93,
                },
                {
                    "graph_threshold": 0.88,
                    "graph_cluster_id": 1,
                    "candidate_key": "2|3",
                    "member_indices": "2|3",
                    "candidate_size": 2,
                    "base_cluster_ids": "1|2",
                    "base_cluster_count": 2,
                    "mean_score": 0.93,
                    "min_score": 0.93,
                },
            ]
        )

        aggregated_df = aggregate_graph_merge_candidates(candidate_df)

        self.assertEqual(len(aggregated_df), 1)
        row = aggregated_df.iloc[0]
        self.assertEqual(int(row["stable_count"]), 2)
        self.assertEqual(str(row["stable_thresholds"]), "0.84|0.88")

    def test_apply_overlay_merges_base_clusters(self) -> None:
        base_labels = np.asarray([0, 0, 1, 2, 3], dtype=np.int32)
        candidate_df = pd.DataFrame(
            [
                {
                    "base_cluster_ids": "1|2",
                }
            ]
        )

        merged = apply_graph_merge_overlay(base_labels=base_labels, candidate_df=candidate_df)

        self.assertListEqual(merged.tolist(), [0, 0, 1, 1, 2])

    def test_summarize_consensus_merge_candidates_groups_cluster_pairs(self) -> None:
        pair_vote_df = pd.DataFrame(
            [
                {
                    "base_cluster_left": 0,
                    "base_cluster_right": 1,
                    "merge_votes": 3,
                    "vote_ratio": 1.0,
                    "pair_score": 0.92,
                    "xgb_same_identity_prob": 0.86,
                    "conflict_methods": "chinese_whispers|dbscan|finch_like",
                },
                {
                    "base_cluster_left": 1,
                    "base_cluster_right": 0,
                    "merge_votes": 2,
                    "vote_ratio": 0.666667,
                    "pair_score": 0.88,
                    "xgb_same_identity_prob": 0.81,
                    "conflict_methods": "chinese_whispers|dbscan",
                },
                {
                    "base_cluster_left": 2,
                    "base_cluster_right": 3,
                    "merge_votes": 1,
                    "vote_ratio": 0.333333,
                    "pair_score": 0.95,
                    "xgb_same_identity_prob": 0.90,
                    "conflict_methods": "chinese_whispers",
                },
            ]
        )

        candidate_df = summarize_consensus_merge_candidates(
            pair_vote_df,
            min_merge_votes=2,
            min_vote_ratio=0.66,
            min_support_pair_count=2,
            min_pair_score=0.85,
            min_pair_probability=0.8,
        )

        self.assertEqual(len(candidate_df), 1)
        row = candidate_df.iloc[0]
        self.assertEqual(str(row["base_cluster_ids"]), "0|1")
        self.assertEqual(int(row["support_pair_count"]), 2)
        self.assertAlmostEqual(float(row["max_vote_ratio"]), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
