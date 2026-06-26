from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.ambiguity_clustering import (
    AMBIGUITY_METHOD_AVERAGE,
    AMBIGUITY_METHOD_CHINESE_WHISPERS,
    AMBIGUITY_METHOD_DBSCAN,
    AMBIGUITY_METHOD_FINCH_LIKE,
    assign_ambiguity_components,
    build_pair_disagreement_table,
    build_pair_vote_summary_table,
    cluster_labels_from_dbscan_score_matrix,
    cluster_labels_from_finch_like_score_matrix,
    summarize_merge_candidates,
    summarize_split_candidates,
)


class AmbiguityClusteringTest(unittest.TestCase):
    def test_dbscan_noise_becomes_singletons(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.92, 0.10, 0.15, 0.10],
                [0.92, 1.0, 0.12, 0.10, 0.10],
                [0.10, 0.12, 1.0, 0.20, 0.18],
                [0.15, 0.10, 0.20, 1.0, 0.91],
                [0.10, 0.10, 0.18, 0.91, 1.0],
            ],
            dtype=np.float32,
        )

        labels = cluster_labels_from_dbscan_score_matrix(
            score_matrix=score,
            threshold=0.85,
            min_samples=2,
        )

        self.assertEqual(int(labels[0]), int(labels[1]))
        self.assertEqual(int(labels[3]), int(labels[4]))
        self.assertNotEqual(int(labels[0]), int(labels[2]))
        self.assertNotEqual(int(labels[2]), int(labels[3]))

    def test_finch_like_merges_first_neighbor_chain(self) -> None:
        score = np.asarray(
            [
                [1.0, 0.95, 0.93, 0.10],
                [0.95, 1.0, 0.90, 0.10],
                [0.93, 0.90, 1.0, 0.20],
                [0.10, 0.10, 0.20, 1.0],
            ],
            dtype=np.float32,
        )

        labels = cluster_labels_from_finch_like_score_matrix(
            score_matrix=score,
            min_link_score=0.85,
        )

        self.assertEqual(labels.tolist(), [0, 0, 0, 1])

    def test_pair_disagreement_and_candidate_summaries(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0, 2],
                "right_index": [1, 3],
                "image_id": ["100", "200"],
                "neighbor_image_id": ["101", "201"],
                "xgb_same_identity_prob": [0.24, 0.26],
            }
        )
        label_map = {
            AMBIGUITY_METHOD_AVERAGE: np.asarray([0, 1, 2, 2], dtype=np.int32),
            AMBIGUITY_METHOD_CHINESE_WHISPERS: np.asarray([0, 0, 2, 3], dtype=np.int32),
            AMBIGUITY_METHOD_DBSCAN: np.asarray([0, 0, 2, 3], dtype=np.int32),
            AMBIGUITY_METHOD_FINCH_LIKE: np.asarray([0, 0, 2, 3], dtype=np.int32),
        }

        disagreement_df = build_pair_disagreement_table(
            pair_df,
            label_map=label_map,
            base_method=AMBIGUITY_METHOD_AVERAGE,
            base_threshold=0.25,
        )

        merge_row = disagreement_df[disagreement_df["left_index"].astype(int).eq(0)].iloc[0]
        split_row = disagreement_df[disagreement_df["left_index"].astype(int).eq(2)].iloc[0]
        self.assertEqual(int(merge_row["merge_votes"]), 3)
        self.assertEqual(str(merge_row["vote_direction"]), "merge")
        self.assertEqual(int(split_row["split_votes"]), 3)
        self.assertEqual(str(split_row["vote_direction"]), "split")

        pair_vote_df = build_pair_vote_summary_table(
            pair_df,
            label_map=label_map,
            base_method=AMBIGUITY_METHOD_AVERAGE,
            base_threshold=0.25,
            score_matrix=np.asarray(
                [
                    [1.0, 0.87, 0.2, 0.2],
                    [0.87, 1.0, 0.2, 0.2],
                    [0.2, 0.2, 1.0, 0.28],
                    [0.2, 0.2, 0.28, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        merge_vote_row = pair_vote_df[pair_vote_df["left_index"].astype(int).eq(0)].iloc[0]
        self.assertEqual(int(merge_vote_row["total_votes"]), 3)
        self.assertAlmostEqual(float(merge_vote_row["vote_ratio"]), 1.0, places=6)
        self.assertAlmostEqual(float(merge_vote_row["pair_score"]), 0.87, places=6)

        disagreement_df["component_id"] = -1
        merge_candidate_df = summarize_merge_candidates(
            disagreement_df,
            base_labels=label_map[AMBIGUITY_METHOD_AVERAGE],
            min_merge_votes=2,
        )
        split_candidate_df = summarize_split_candidates(
            disagreement_df,
            base_labels=label_map[AMBIGUITY_METHOD_AVERAGE],
            min_split_votes=2,
        )

        self.assertEqual(len(merge_candidate_df), 1)
        self.assertEqual(str(merge_candidate_df.iloc[0]["cluster_pair_key"]), "0|1")
        self.assertEqual(len(split_candidate_df), 1)
        self.assertEqual(int(split_candidate_df.iloc[0]["base_cluster_id"]), 2)

    def test_assign_ambiguity_components_groups_connected_edges(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "left_index": 0,
                    "right_index": 1,
                    "image_id": "100",
                    "neighbor_image_id": "101",
                    "base_cluster_left": 0,
                    "base_cluster_right": 1,
                    "ambiguity_score": 0.90,
                    "base_conflict_ratio": 1.00,
                    "vote_direction": "merge",
                    "xgb_same_identity_prob": 0.82,
                },
                {
                    "left_index": 1,
                    "right_index": 2,
                    "image_id": "101",
                    "neighbor_image_id": "102",
                    "base_cluster_left": 1,
                    "base_cluster_right": 2,
                    "ambiguity_score": 0.85,
                    "base_conflict_ratio": 0.67,
                    "vote_direction": "merge",
                    "xgb_same_identity_prob": 0.78,
                },
                {
                    "left_index": 3,
                    "right_index": 4,
                    "image_id": "200",
                    "neighbor_image_id": "201",
                    "base_cluster_left": 4,
                    "base_cluster_right": 4,
                    "ambiguity_score": 0.88,
                    "base_conflict_ratio": 1.00,
                    "vote_direction": "split",
                    "xgb_same_identity_prob": 0.31,
                },
            ]
        )

        pair_with_components_df, component_df = assign_ambiguity_components(
            pair_df,
            min_ambiguity_score=0.8,
            min_conflict_ratio=0.5,
        )

        self.assertEqual(len(component_df), 2)
        merge_component = component_df[component_df["dominant_direction"].astype(str).eq("merge")].iloc[0]
        self.assertEqual(int(merge_component["image_count"]), 3)
        self.assertEqual(int((pair_with_components_df["component_id"].astype(int) >= 0).sum()), 3)


if __name__ == "__main__":
    unittest.main()
