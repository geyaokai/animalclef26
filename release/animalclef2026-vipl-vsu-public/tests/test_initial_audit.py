from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.initial_audit import (
    build_count_table,
    compute_sharpness,
    ensure_required_columns,
    summarize_identity_distribution,
)


class InitialAuditTest(unittest.TestCase):
    def test_ensure_required_columns_accepts_expected_schema(self) -> None:
        df = pd.DataFrame(columns=[
            "image_id",
            "identity",
            "path",
            "date",
            "orientation",
            "species",
            "split",
            "dataset",
        ])
        ensure_required_columns(df)

    def test_compute_sharpness_prefers_edges_over_flat_image(self) -> None:
        flat = np.zeros((8, 8), dtype=np.uint8)
        edged = np.zeros((8, 8), dtype=np.uint8)
        edged[:, 4:] = 255
        self.assertLess(compute_sharpness(flat), compute_sharpness(edged))

    def test_build_count_table_counts_group_sizes(self) -> None:
        df = pd.DataFrame({"dataset": ["A", "A", "B"], "split": ["train", "test", "train"]})
        result = build_count_table(df, ["dataset", "split"])
        counts = {(row.dataset, row.split): row.count for row in result.itertuples(index=False)}
        self.assertEqual(counts[("A", "train")], 1)
        self.assertEqual(counts[("A", "test")], 1)
        self.assertEqual(counts[("B", "train")], 1)

    def test_summarize_identity_distribution_reports_singletons(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4", "5"],
                "identity": ["id1", "id1", "id2", "id3", ""],
                "path": ["a", "b", "c", "d", "e"],
                "date": ["", "", "", "", ""],
                "orientation": ["left", "left", "right", "top", "top"],
                "species": ["lynx", "lynx", "lynx", "lynx", ""],
                "split": ["train", "train", "train", "train", "test"],
                "dataset": ["LynxID2025"] * 5,
            }
        )
        summary = summarize_identity_distribution(df)
        row = summary.iloc[0]
        self.assertEqual(int(row["identities"]), 3)
        self.assertEqual(int(row["singletons"]), 2)


if __name__ == "__main__":
    unittest.main()
