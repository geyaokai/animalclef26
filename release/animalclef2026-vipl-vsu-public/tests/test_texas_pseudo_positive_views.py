from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.animalclef_analysis.texas_pseudo_positive_views import build_texas_pseudo_positive_views


class TexasPseudoPositiveViewsTest(unittest.TestCase):
    def test_builds_base_vs_positive_metadata_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trusted_path = root / "trusted.csv"
            manifest_path = root / "manifest.csv"
            output_dir = root / "out"

            pd.DataFrame(
                [
                    {
                        "image_id": "15212",
                        "dataset": "TexasHornedLizards",
                        "component_id": "trusted_comp_001",
                        "source_type": "human_yes_pair",
                        "trusted_level": "strong",
                    },
                    {
                        "image_id": "15214",
                        "dataset": "TexasHornedLizards",
                        "component_id": "trusted_comp_001",
                        "source_type": "trusted_class",
                        "trusted_level": "strong",
                    },
                ]
            ).to_csv(trusted_path, index=False)

            pd.DataFrame(
                [
                    {
                        "image_id": "15212",
                        "dataset": "TexasHornedLizards",
                        "path": "views/15212_gray.jpg",
                        "preferred_path_v1": "views/15212_gray.jpg",
                        "recommended_model_input_path_v1": "views/15212_gray.jpg",
                        "manifest_view_name_v1": "texas_center_body_square_gray_v1",
                        "preprocess_variant_v1": "texas_center_body_square_gray_v1",
                        "texas_center_body_foreground_ratio_in_crop_v1": 0.60,
                        "sam_trainprep_aligned_axis_confidence_v1": 0.44,
                    },
                    {
                        "image_id": "15214",
                        "dataset": "TexasHornedLizards",
                        "path": "views/15214_gray.jpg",
                        "preferred_path_v1": "views/15214_gray.jpg",
                        "recommended_model_input_path_v1": "views/15214_gray.jpg",
                        "manifest_view_name_v1": "texas_center_body_square_gray_v1",
                        "preprocess_variant_v1": "texas_center_body_square_gray_v1",
                        "texas_center_body_foreground_ratio_in_crop_v1": 0.22,
                        "sam_trainprep_aligned_axis_confidence_v1": 0.10,
                    },
                ]
            ).to_csv(manifest_path, index=False)

            outputs = build_texas_pseudo_positive_views(
                trusted_membership_path=trusted_path,
                manifest_path=manifest_path,
                output_dir=output_dir,
            )

            views_df = pd.read_csv(outputs["views_path"])
            pairs_df = pd.read_csv(outputs["pairs_path"])
            summary_text = outputs["summary_path"].read_text(encoding="utf-8")

            self.assertEqual(len(views_df[views_df["is_base_view"] == True]), 2)  # noqa: E712
            self.assertGreaterEqual(len(views_df[views_df["is_base_view"] == False]), 8)  # noqa: E712
            self.assertTrue((pairs_df["pair_kind"] == "intra_image_pseudo_positive").all())
            self.assertTrue((pairs_df["base_view_id"].str.startswith("base::")).all())
            self.assertTrue((pairs_df["positive_view_id"].str.startswith("pos::")).all())
            self.assertIn("Texas Pseudo-Positive Views", summary_text)
            self.assertIn("Recipe Coverage", summary_text)

            flip_row_good = views_df.loc[views_df["view_id"] == "pos::15212::horizontal_flip_gated_v1"].iloc[0]
            flip_row_bad = views_df.loc[views_df["view_id"] == "pos::15214::horizontal_flip_gated_v1"].iloc[0]
            self.assertTrue(bool(flip_row_good["gate_enabled_v1"]))
            self.assertFalse(bool(flip_row_bad["gate_enabled_v1"]))
            self.assertNotIn("pair::15214::horizontal_flip_gated_v1", set(pairs_df["pair_id"].tolist()))


if __name__ == "__main__":
    unittest.main()
