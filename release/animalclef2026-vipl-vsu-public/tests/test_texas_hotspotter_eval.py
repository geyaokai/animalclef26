from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.texas_hotspotter_eval import _effective_recall_at_k, _first_hit_rank_df


class TexasHotspotterEvalTest(unittest.TestCase):
    def test_effective_recall_at_k_ignores_singletons(self) -> None:
        labels_df = pd.DataFrame(
            {
                "image_id": ["a1", "a2", "b1"],
                "identity": ["a", "a", "b"],
            }
        )
        ranking_df = pd.DataFrame(
            {
                "image_id": ["a1", "a2", "b1"],
                "neighbor_image_id": ["a2", "a1", "a1"],
                "rank": [1, 1, 1],
            }
        )
        self.assertAlmostEqual(_effective_recall_at_k(ranking_df=ranking_df, labels_df=labels_df, k=1), 1.0)

    def test_first_hit_rank_df_returns_first_matching_rank(self) -> None:
        labels_df = pd.DataFrame(
            {
                "image_id": ["a1", "a2", "a3", "b1"],
                "identity": ["a", "a", "a", "b"],
            }
        )
        ranking_df = pd.DataFrame(
            {
                "image_id": ["a1", "a1", "a2", "a3"],
                "neighbor_image_id": ["b1", "a2", "a1", "a1"],
                "rank": [1, 2, 1, 1],
                "local_score": [0.9, 0.8, 0.7, 0.6],
                "inliers": [10, 8, 9, 7],
            }
        )
        result = _first_hit_rank_df(ranking_df=ranking_df, labels_df=labels_df)
        a1 = result[result["image_id"] == "a1"].iloc[0]
        a2 = result[result["image_id"] == "a2"].iloc[0]
        self.assertEqual(int(a1["first_hit_rank"]), 2)
        self.assertEqual(int(a2["first_hit_rank"]), 1)
        self.assertNotIn("b1", result["image_id"].tolist())

    def test_effective_recall_at_k_counts_topk_hit(self) -> None:
        labels_df = pd.DataFrame(
            {
                "image_id": ["a1", "a2", "a3"],
                "identity": ["a", "a", "a"],
            }
        )
        ranking_df = pd.DataFrame(
            {
                "image_id": ["a1", "a1", "a2", "a3"],
                "neighbor_image_id": ["x", "a2", "a1", "a1"],
                "rank": [1, 2, 1, 1],
            }
        )
        self.assertAlmostEqual(_effective_recall_at_k(ranking_df=ranking_df, labels_df=labels_df, k=1), 2 / 3, places=6)
        self.assertAlmostEqual(_effective_recall_at_k(ranking_df=ranking_df, labels_df=labels_df, k=2), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
