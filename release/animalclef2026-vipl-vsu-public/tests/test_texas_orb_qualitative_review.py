from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.animalclef_analysis.texas_orb_qualitative_review import (
    CATEGORY_SPECS,
    load_texas_judged_pair_df,
    select_category_rows,
    summarize_judged_pairs,
)


class TexasOrbQualitativeReviewTest(unittest.TestCase):
    def test_select_category_rows_picks_expected_pairs(self) -> None:
        judged_pair_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4"],
                "neighbor_image_id": ["5", "6", "7", "8"],
                "label": ["yes", "yes", "no", "no"],
                "local_score": [0.9, 0.2, 0.8, 0.1],
                "miew_local_score": [0.5, 0.3, 0.4, 0.2],
                "orb_inliers": [12, 3, 11, 1],
                "orb_good_matches": [30, 10, 28, 5],
                "xgb_same_identity_prob": [0.7, 0.4, 0.6, 0.1],
                "candidate_type": ["split", "split", "merge", "merge"],
                "candidate_key": ["a", "a", "b", "c"],
                "orb_minus_miew": [0.4, -0.1, 0.4, -0.1],
            }
        )
        top_yes = select_category_rows(judged_pair_df, spec=CATEGORY_SPECS[0], top_k=1)
        top_no_false_support = select_category_rows(judged_pair_df, spec=CATEGORY_SPECS[2], top_k=1)

        self.assertEqual(top_yes.iloc[0]["image_id"], "1")
        self.assertEqual(top_no_false_support.iloc[0]["image_id"], "3")

    def test_summarize_judged_pairs_builds_label_tables(self) -> None:
        judged_pair_df = pd.DataFrame(
            {
                "label": ["yes", "yes", "no", "no"],
                "candidate_type": ["split", "merge", "merge", "merge"],
                "local_score": [0.9, 0.7, 0.3, 0.1],
                "miew_local_score": [0.8, 0.5, 0.4, 0.2],
                "orb_inliers": [10, 8, 3, 1],
                "orb_good_matches": [25, 20, 8, 4],
                "xgb_same_identity_prob": [0.65, 0.55, 0.2, 0.1],
            }
        )
        label_summary_df, threshold_summary_df = summarize_judged_pairs(judged_pair_df)

        self.assertEqual(sorted(label_summary_df["label"].tolist()), ["no", "yes"])
        self.assertEqual(len(threshold_summary_df), 4)
        yes_row = label_summary_df[label_summary_df["label"] == "yes"].iloc[0]
        self.assertAlmostEqual(float(yes_row["mean_orb_local_score"]), 0.8, places=6)

    def test_load_texas_judged_pair_df_matches_by_canonical_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            judgments_path = root / "judgments.json"
            judgments_path.write_text(
                """
{
  "session_name": "toy",
  "pair_judgments": [
    {
      "dataset": "TexasHornedLizards",
      "candidate_type": "merge",
      "candidate_key": "1|2",
      "image_id": "200",
      "neighbor_image_id": "100",
      "label": "no"
    }
  ]
}
                """.strip(),
                encoding="utf-8",
            )
            review_dir = root / "review"
            (review_dir / "tables").mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "image_id": ["100"],
                    "neighbor_image_id": ["200"],
                    "local_score": [0.8],
                    "miew_local_score": [0.5],
                    "orb_inliers": [9],
                    "orb_good_matches": [30],
                    "xgb_same_identity_prob": [0.4],
                }
            ).to_csv(review_dir / "tables" / "test_pair_disagreement_v1.csv", index=False)

            merged_df, session_name = load_texas_judged_pair_df(
                repo_root=root,
                pair_judgments_path=judgments_path,
                review_dir=review_dir,
            )

            self.assertEqual(session_name, "toy")
            self.assertEqual(len(merged_df), 1)
            self.assertAlmostEqual(float(merged_df.iloc[0]["local_score"]), 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
