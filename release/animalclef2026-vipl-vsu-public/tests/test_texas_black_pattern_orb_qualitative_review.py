from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.texas_black_pattern_orb_qualitative_review import (
    CATEGORY_SPECS,
    select_category_rows,
    summarize_judged_pairs,
)


class TexasBlackPatternOrbQualitativeReviewTest(unittest.TestCase):
    def test_summarize_judged_pairs_tracks_black_and_old_orb(self) -> None:
        judged_pair_df = pd.DataFrame(
            {
                "label": ["yes", "yes", "no", "no"],
                "candidate_type": ["split", "merge", "merge", "split"],
                "black_orb_local_score": [0.92, 0.68, 0.21, 0.04],
                "old_orb_local_score": [0.70, 0.62, 0.77, 0.49],
                "miew_local_score": [0.63, 0.58, 0.61, 0.45],
                "black_orb_inliers": [14, 9, 3, 0],
                "black_minus_old_orb": [0.22, 0.06, -0.56, -0.45],
            }
        )

        label_summary_df, threshold_summary_df, delta_summary_df = summarize_judged_pairs(judged_pair_df)

        yes_row = label_summary_df[label_summary_df["label"] == "yes"].iloc[0]
        no_row = delta_summary_df[delta_summary_df["label"] == "no"].iloc[0]
        self.assertAlmostEqual(float(yes_row["mean_black_orb_local_score"]), 0.8, places=6)
        self.assertEqual(len(threshold_summary_df), 4)
        self.assertEqual(int(no_row["old_orb_ge_0p6"]), 1)

    def test_select_category_rows_prefers_fix_old_false_support(self) -> None:
        judged_pair_df = pd.DataFrame(
            {
                "image_id": ["a", "b", "c"],
                "neighbor_image_id": ["d", "e", "f"],
                "label": ["no", "no", "yes"],
                "black_orb_local_score": [0.10, 0.55, 0.80],
                "black_orb_inliers": [0, 8, 11],
                "black_orb_good_matches": [2, 18, 22],
                "old_orb_local_score": [0.82, 0.60, 0.50],
                "old_orb_minus_black": [0.72, 0.05, -0.30],
                "black_minus_old_orb": [-0.72, -0.05, 0.30],
                "candidate_type": ["merge", "merge", "split"],
                "candidate_key": ["1|2", "3|4", "5"],
            }
        )

        subset = select_category_rows(judged_pair_df=judged_pair_df, spec=CATEGORY_SPECS[4], top_k=1)

        self.assertEqual(subset.iloc[0]["image_id"], "a")


if __name__ == "__main__":
    unittest.main()
