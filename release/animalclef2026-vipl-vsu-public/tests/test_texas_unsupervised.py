from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.texas_unsupervised import (
    build_seed_clusters_from_candidate_pairs,
    mean_topk_neighbor_overlap,
    pair_agreement_score,
    summarize_cluster_labels,
)


class TexasUnsupervisedHelpersTest(unittest.TestCase):
    def test_pair_agreement_score_tracks_pairwise_partition_agreement(self) -> None:
        left = np.array([0, 0, 1, 2])
        right = np.array([0, 1, 1, 2])
        self.assertAlmostEqual(pair_agreement_score(left, right), 0.666667, places=6)

    def test_mean_topk_neighbor_overlap_uses_jaccard_per_row(self) -> None:
        topk_left = np.array([[1, 2], [0, 2]], dtype=np.int32)
        topk_right = np.array([[1, 3], [0, 3]], dtype=np.int32)
        self.assertAlmostEqual(mean_topk_neighbor_overlap(topk_left, topk_right), 0.333333, places=6)

    def test_summarize_cluster_labels_reports_size_distribution(self) -> None:
        labels = np.array([0, 0, 1, 2, 2, 2])
        summary = summarize_cluster_labels(labels)
        self.assertEqual(summary["clusters"], 3)
        self.assertEqual(summary["largest_cluster_size"], 3)
        self.assertEqual(summary["singleton_clusters"], 1)
        self.assertAlmostEqual(float(summary["non_singleton_image_ratio"]), 5 / 6, places=6)

    def test_build_seed_clusters_from_candidate_pairs_keeps_dense_components(self) -> None:
        pair_df = pd.DataFrame(
            {
                "image_id": ["1", "1", "2", "4"],
                "neighbor_image_id": ["2", "3", "3", "5"],
                "mutual_topk_all_routes": [True, True, True, True],
            }
        )
        assignments_df = build_seed_clusters_from_candidate_pairs(
            image_ids=["1", "2", "3", "4", "5", "6"],
            pair_df=pair_df,
            min_component_density=0.66,
            max_seed_cluster_size=8,
        )
        seed_df = assignments_df[assignments_df["seed_status"] == "seed"]
        uncertain_df = assignments_df[assignments_df["seed_status"] == "uncertain"]
        self.assertEqual(set(seed_df["pseudo_identity"].tolist()), {"texas_seed_0000", "texas_seed_0001"})
        self.assertEqual(
            seed_df.groupby("pseudo_identity")["image_id"].count().to_dict(),
            {"texas_seed_0000": 3, "texas_seed_0001": 2},
        )
        self.assertIn("6", set(uncertain_df["image_id"].tolist()))

    def test_build_seed_clusters_from_candidate_pairs_rejects_sparse_component(self) -> None:
        pair_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "neighbor_image_id": ["2", "3"],
                "mutual_topk_all_routes": [True, True],
            }
        )
        assignments_df = build_seed_clusters_from_candidate_pairs(
            image_ids=["1", "2", "3"],
            pair_df=pair_df,
            min_component_density=0.8,
            max_seed_cluster_size=8,
        )
        self.assertTrue((assignments_df["seed_status"] == "uncertain").all())


if __name__ == "__main__":
    unittest.main()
