from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.preprocessing_baselines import (
    build_duplicate_flags_from_metrics,
    create_enriched_metadata,
    rotation_angle_for_orientation,
)


class PreprocessingBaselinesTest(unittest.TestCase):
    def test_rotation_angle_for_orientation_matches_v1_rule(self) -> None:
        self.assertEqual(rotation_angle_for_orientation("top"), 0)
        self.assertEqual(rotation_angle_for_orientation("right"), 90)
        self.assertEqual(rotation_angle_for_orientation("left"), -90)
        self.assertEqual(rotation_angle_for_orientation("bottom"), 180)
        self.assertEqual(rotation_angle_for_orientation("unknown"), 0)

    def test_build_duplicate_flags_marks_exact_duplicates(self) -> None:
        metrics_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3"],
                "path": ["a.jpg", "b.jpg", "c.jpg"],
                "dataset": ["SeaTurtleID2022"] * 3,
                "split": ["train"] * 3,
                "identity": ["id1", "id1", "id2"],
                "sha1": ["hash_a", "hash_a", "hash_b"],
            }
        )
        result = build_duplicate_flags_from_metrics(metrics_df)
        row1 = result[result["image_id"] == "1"].iloc[0]
        row3 = result[result["image_id"] == "3"].iloc[0]
        self.assertTrue(bool(row1["is_exact_duplicate"]))
        self.assertEqual(int(row1["duplicate_group_size"]), 2)
        self.assertFalse(bool(row3["is_exact_duplicate"]))

    def test_create_enriched_metadata_prefers_normalized_salamander_path(self) -> None:
        metadata_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "identity": ["sal_1", "turtle_1"],
                "path": ["images/SalamanderID2025/train/x.jpg", "images/SeaTurtleID2022/train/y.jpg"],
                "date": ["", ""],
                "orientation": ["right", "left"],
                "species": ["salamander", "loggerhead turtle"],
                "split": ["train", "train"],
                "dataset": ["SalamanderID2025", "SeaTurtleID2022"],
            }
        )
        orientation_manifest_df = pd.DataFrame(
            {
                "image_id": ["1"],
                "rotation_degrees_v1": [90],
                "normalized_path_v1": ["artifacts/preprocessing_baselines/v1/salamander_orientation_v1/SalamanderID2025/train/x.jpg"],
                "normalization_applied_v1": [True],
            }
        )
        duplicate_flags_df = pd.DataFrame(
            {
                "image_id": ["2"],
                "exact_duplicate_sha1": ["hash_a"],
                "duplicate_group_size": [2],
                "duplicate_rank": [1],
                "is_exact_duplicate": [True],
                "is_duplicate_primary": [False],
            }
        )
        enriched = create_enriched_metadata(metadata_df, orientation_manifest_df, duplicate_flags_df)
        salamander = enriched[enriched["image_id"] == "1"].iloc[0]
        turtle = enriched[enriched["image_id"] == "2"].iloc[0]
        self.assertEqual(salamander["preferred_path_v1"], orientation_manifest_df.iloc[0]["normalized_path_v1"])
        self.assertTrue(bool(salamander["normalization_applied_v1"]))
        self.assertTrue(bool(turtle["is_exact_duplicate"]))


if __name__ == "__main__":
    unittest.main()
