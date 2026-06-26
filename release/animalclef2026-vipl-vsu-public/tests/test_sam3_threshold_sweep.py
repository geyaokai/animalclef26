from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.sam3_threshold_sweep import (
    SUMMARY_COLUMNS,
    summarize_probe_results,
    threshold_tag,
)


class Sam3ThresholdSweepTest(unittest.TestCase):
    def test_threshold_tag_formats_for_paths(self) -> None:
        self.assertEqual(threshold_tag(0.3), "0p3")
        self.assertEqual(threshold_tag(0.75), "0p75")
        self.assertEqual(threshold_tag(1.0), "1")

    def test_summarize_probe_results_aggregates_by_dataset_and_split(self) -> None:
        results_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3"],
                "dataset": ["SeaTurtleID2022", "SeaTurtleID2022", "SalamanderID2025"],
                "split": ["train", "train", "test"],
                "mask_count": [0, 2, 1],
                "mask_area_ratio": [0.0, 0.6, 0.4],
                "best_score": [0.0, 0.9, 0.8],
            }
        )

        summary_df = summarize_probe_results(results_df, threshold=0.3, mask_threshold=0.3)

        self.assertEqual(list(summary_df.columns), SUMMARY_COLUMNS)
        self.assertEqual(len(summary_df), 2)
        sea_row = summary_df[summary_df["dataset"] == "SeaTurtleID2022"].iloc[0]
        self.assertEqual(sea_row["samples"], 2)
        self.assertEqual(sea_row["positive_masks"], 1)
        self.assertAlmostEqual(sea_row["positive_ratio"], 0.5)
        self.assertAlmostEqual(sea_row["mean_area_ratio"], 0.3)
        self.assertAlmostEqual(sea_row["mean_best_score"], 0.45)


if __name__ == "__main__":
    unittest.main()
