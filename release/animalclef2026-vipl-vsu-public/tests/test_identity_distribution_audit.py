from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.animalclef_analysis.identity_distribution_audit import (
    build_distribution_table,
    build_identity_count_frame,
    build_summary_table,
    write_markdown_report,
)


class IdentityDistributionAuditTest(unittest.TestCase):
    def test_build_identity_count_frame(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["A", "A", "A", "B"],
                "identity": ["x", "x", "y", "z"],
                "split_role_v1": ["fit", "fit", "val", "fit"],
            }
        )
        counts = build_identity_count_frame(df, extra_group_columns=["split_role_v1"])
        expected = pd.DataFrame(
            {
                "dataset": ["A", "A", "B"],
                "split_role_v1": ["fit", "val", "fit"],
                "identity": ["x", "y", "z"],
                "images_per_identity": [2, 1, 1],
            }
        )
        self.assertTrue(counts.equals(expected))

    def test_distribution_and_summary_tables(self) -> None:
        counts_df = pd.DataFrame(
            {
                "dataset": ["ALL", "ALL", "ALL", "A", "A"],
                "images_per_identity": [1, 1, 3, 1, 3],
            }
        )
        distribution = build_distribution_table(counts_df, group_columns=["dataset"])
        summary = build_summary_table(counts_df, group_columns=["dataset"])

        all_dist = distribution[distribution["dataset"] == "ALL"].set_index("images_per_identity")["identity_count"].to_dict()
        self.assertEqual(all_dist, {1: 2, 3: 1})

        all_summary = summary[summary["dataset"] == "ALL"].iloc[0]
        self.assertEqual(int(all_summary["identities"]), 3)
        self.assertEqual(int(all_summary["singletons"]), 2)
        self.assertAlmostEqual(float(all_summary["singleton_ratio"]), 2 / 3, places=4)

    def test_write_markdown_report_includes_figures_and_reading_notes(self) -> None:
        overall_summary_df = pd.DataFrame(
            {
                "dataset": ["ALL_LABELED_TRAIN", "A"],
                "identities": [3, 2],
                "total_images": [5, 3],
                "min_images": [1, 1],
                "median_images": [1.0, 1.5],
                "mean_images": [1.667, 1.5],
                "max_images": [3, 2],
                "singletons": [2, 1],
                "singleton_ratio": [0.6667, 0.5],
            }
        )
        split_summary_df = pd.DataFrame(
            {
                "split_role_v1": ["fit", "val", "fit", "val"],
                "dataset": ["ALL_LABELED_TRAIN", "ALL_LABELED_TRAIN", "A", "A"],
                "identities": [2, 1, 1, 1],
                "total_images": [3, 2, 2, 1],
                "min_images": [1, 2, 2, 1],
                "median_images": [1.5, 2.0, 2.0, 1.0],
                "mean_images": [1.5, 2.0, 2.0, 1.0],
                "max_images": [2, 2, 2, 1],
                "singletons": [1, 0, 0, 1],
                "singleton_ratio": [0.5, 0.0, 0.0, 1.0],
            }
        )
        config = {"datasets": ["A"], "val_identity_fraction": 0.1, "split_seed": 42}

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            report_path = root / "reports" / "summary.md"
            plot_paths = {
                "overall": root / "plots" / "overall.png",
                "per_dataset": root / "plots" / "per_dataset.png",
                "split_overall": root / "plots" / "split_overall.png",
                "split_by_dataset": root / "plots" / "split_by_dataset.png",
            }
            write_markdown_report(
                output_path=report_path,
                overall_summary_df=overall_summary_df,
                split_summary_df=split_summary_df,
                config=config,
                plot_paths=plot_paths,
            )
            text = report_path.read_text(encoding="utf-8")

        self.assertIn("## 图怎么读", text)
        self.assertIn("![Overall labeled-train identity distribution](../plots/overall.png)", text)
        self.assertIn("图 1 说明：", text)
        self.assertIn("图 4 说明：", text)


if __name__ == "__main__":
    unittest.main()
