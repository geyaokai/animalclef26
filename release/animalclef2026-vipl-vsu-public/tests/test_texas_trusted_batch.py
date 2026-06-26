from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.animalclef_analysis.texas_trusted_batch import compile_texas_trusted_batch


class TexasTrustedBatchTest(unittest.TestCase):
    def test_compile_texas_trusted_batch_merges_manual_yes_and_approved_seed_classes(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manual_pairs_path = root / "manual_pairs.csv"
            metadata_path = root / "metadata.csv"
            review_package_dir = root / "review_package"
            tables_dir = review_package_dir / "tables"
            output_dir = root / "trusted_out"
            tables_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "dataset": "TexasHornedLizards",
                        "candidate_type": "yes",
                        "candidate_key": "agree:foo",
                        "pair_key": "100|101",
                        "image_id": "100",
                        "neighbor_image_id": "101",
                        "label": "yes",
                        "note": "",
                        "base_cluster_left": 1,
                        "base_cluster_right": 1,
                        "xgb_same_identity_prob": 0.9,
                        "ambiguity_score": 0.0,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "candidate_type": "yes",
                        "candidate_key": "agree:bar",
                        "pair_key": "101|102",
                        "image_id": "101",
                        "neighbor_image_id": "102",
                        "label": "no",
                        "note": "",
                        "base_cluster_left": 1,
                        "base_cluster_right": 2,
                        "xgb_same_identity_prob": 0.2,
                        "ambiguity_score": 0.7,
                    },
                ]
            ).to_csv(manual_pairs_path, index=False)

            pd.DataFrame(
                [
                    {"image_id": "100", "dataset": "TexasHornedLizards", "split": "test", "path": "images/TexasHornedLizards/test/100.jpg"},
                    {"image_id": "101", "dataset": "TexasHornedLizards", "split": "test", "path": "images/TexasHornedLizards/test/101.jpg"},
                    {"image_id": "200", "dataset": "TexasHornedLizards", "split": "test", "path": "images/TexasHornedLizards/test/200.jpg"},
                    {"image_id": "201", "dataset": "TexasHornedLizards", "split": "test", "path": "images/TexasHornedLizards/test/201.jpg"},
                    {"image_id": "15466", "dataset": "TexasHornedLizards", "split": "test", "path": "images/TexasHornedLizards/test/15466.jpg"},
                ]
            ).to_csv(metadata_path, index=False)

            pd.DataFrame(
                [
                    {"pseudo_identity": "seed_a", "size": 2, "contact_sheet_path": "dummy/01.jpg"},
                    {"pseudo_identity": "seed_b", "size": 3, "contact_sheet_path": "dummy/02.jpg"},
                ]
            ).to_csv(tables_dir / "seed_cluster_contact_sheets_v1.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "image_id": "200",
                        "dataset": "TexasHornedLizards",
                        "pseudo_identity": "seed_b",
                        "path": "views/200.jpg",
                        "preferred_path_v1": "views/200.jpg",
                        "recommended_model_input_path_v1": "views/200.jpg",
                        "original_rgb_path_v1": "images/TexasHornedLizards/test/200.jpg",
                        "texas_center_body_repaired_fallback_stage_v1": "none",
                    },
                    {
                        "image_id": "201",
                        "dataset": "TexasHornedLizards",
                        "pseudo_identity": "seed_b",
                        "path": "views/201.jpg",
                        "preferred_path_v1": "views/201.jpg",
                        "recommended_model_input_path_v1": "views/201.jpg",
                        "original_rgb_path_v1": "images/TexasHornedLizards/test/201.jpg",
                        "texas_center_body_repaired_fallback_stage_v1": "none",
                    },
                    {
                        "image_id": "15466",
                        "dataset": "TexasHornedLizards",
                        "pseudo_identity": "seed_b",
                        "path": "views/15466.jpg",
                        "preferred_path_v1": "views/15466.jpg",
                        "recommended_model_input_path_v1": "views/15466.jpg",
                        "original_rgb_path_v1": "images/TexasHornedLizards/test/15466.jpg",
                        "texas_center_body_repaired_fallback_stage_v1": "none",
                    },
                ]
            ).to_csv(tables_dir / "pseudo_manifest_v1.csv", index=False)

            artifacts = compile_texas_trusted_batch(
                repo_root=root,
                manual_pairs_path=manual_pairs_path,
                review_package_dir=review_package_dir,
                output_dir=output_dir,
                approved_class_indices=[2],
                class_exclusions={2: {"15466"}},
                metadata_path=metadata_path,
            )

            membership_df = pd.read_csv(artifacts.trusted_membership_path)
            trusted_pairs_df = pd.read_csv(artifacts.trusted_pairs_path)
            components_df = pd.read_csv(artifacts.trusted_components_path)

            self.assertEqual(set(membership_df["image_id"].astype(str)), {"100", "101", "200", "201"})
            self.assertNotIn("15466", set(membership_df["image_id"].astype(str)))
            self.assertEqual(len(trusted_pairs_df), 2)
            self.assertEqual(set(trusted_pairs_df["pair_key"].astype(str)), {"100|101", "200|201"})
            self.assertEqual(
                set(trusted_pairs_df["provenance"].astype(str)),
                {"manual_yes_pair", "approved_seed_class"},
            )
            self.assertEqual(len(components_df), 2)
            self.assertEqual(sorted(components_df["member_count"].astype(int).tolist()), [2, 2])


if __name__ == "__main__":
    unittest.main()
