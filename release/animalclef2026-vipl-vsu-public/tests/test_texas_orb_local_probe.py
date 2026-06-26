from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.texas_orb_local_probe import (
    TEXAS_DATASET,
    build_texas_orb_pair_index,
    merge_texas_orb_local_scores,
    normalize_texas_pair_df,
)


class TexasOrbLocalProbeHelpersTest(unittest.TestCase):
    def _reference_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_id": ["15209", "15210", "15211"],
                "dataset": [TEXAS_DATASET, TEXAS_DATASET, TEXAS_DATASET],
                "identity": ["", "", ""],
                "recommended_model_input_path_v1": [
                    "images/TexasHornedLizards/test/a.jpg",
                    "images/TexasHornedLizards/test/b.jpg",
                    "images/TexasHornedLizards/test/c.jpg",
                ],
            }
        )

    def test_normalize_texas_pair_df_maps_indices_and_orders_pair(self) -> None:
        pair_df = pd.DataFrame(
            {
                "image_id": ["15210", "15209"],
                "neighbor_image_id": ["15209", "15211"],
                "route_global_score": [0.42, 0.35],
            }
        )

        normalized_df, score_column = normalize_texas_pair_df(
            pair_df=pair_df,
            reference_df=self._reference_df(),
            score_column="route_global_score",
        )

        self.assertEqual(score_column, "route_global_score")
        self.assertEqual(normalized_df[["left_index", "right_index"]].values.tolist(), [[0, 1], [0, 2]])
        self.assertEqual(normalized_df["image_id"].tolist(), ["15209", "15209"])
        self.assertEqual(normalized_df["neighbor_image_id"].tolist(), ["15210", "15211"])

    def test_build_texas_orb_pair_index_uses_fallback_score_column(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [2],
                "right_index": [1],
                "image_id": ["15211"],
                "neighbor_image_id": ["15210"],
                "miew_similarity": [0.73],
            }
        )

        pair_index, normalized_df, score_column = build_texas_orb_pair_index(
            pair_df=pair_df,
            reference_df=self._reference_df(),
        )

        self.assertEqual(score_column, "miew_similarity")
        self.assertEqual(pair_index, [(1, 2, 0.73)])
        self.assertEqual(normalized_df.iloc[0]["image_id"], "15210")
        self.assertEqual(normalized_df.iloc[0]["neighbor_image_id"], "15211")

    def test_merge_texas_orb_local_scores_preserves_previous_local_score(self) -> None:
        pair_df = pd.DataFrame(
            {
                "left_index": [0, 0],
                "right_index": [1, 2],
                "image_id": ["15209", "15209"],
                "neighbor_image_id": ["15210", "15211"],
                "local_score": [0.55, 0.48],
            }
        )
        local_match_df = pd.DataFrame(
            {
                "left_index": [0, 0],
                "right_index": [1, 2],
                "image_id": ["15209", "15209"],
                "neighbor_image_id": ["15210", "15211"],
                "matcher_name": ["orb", "orb"],
                "global_score": [0.61, 0.58],
                "good_matches": [8, 2],
                "inliers": [6, 0],
                "local_raw_score": [0.14, 0.0],
                "local_score": [0.28, 0.0],
            }
        )

        merged_df = merge_texas_orb_local_scores(pair_df=pair_df, local_match_df=local_match_df)

        self.assertEqual(merged_df["miew_local_score"].tolist(), [0.55, 0.48])
        self.assertEqual(merged_df["local_score"].tolist(), [0.28, 0.0])
        self.assertEqual(merged_df["orb_matcher_name"].tolist(), ["orb", "orb"])
        self.assertEqual(merged_df["orb_inliers"].tolist(), [6, 0])
        self.assertEqual(merged_df["local_score_source"].tolist(), ["orb_local_probe", "orb_local_probe"])


if __name__ == "__main__":
    unittest.main()
