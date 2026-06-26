from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.animalclef_analysis.tcu_texas_dataset import build_tcu_texas_dataset_artifacts


class TcuTexasDatasetTest(unittest.TestCase):
    def test_build_tcu_texas_dataset_artifacts_compiles_manifest_and_alignment_audit(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            chips_dir = root / "chips"
            original_dir = root / "original"
            output_dir = root / "out"
            mapping_path = root / "mapping.csv"
            hotspotter_output_path = root / "hotspotter.csv"
            chips_dir.mkdir(parents=True, exist_ok=True)
            original_dir.mkdir(parents=True, exist_ok=True)

            for name in ["cid1_CHIP(sz750).png", "cid2_CHIP(sz750).png", "cid3_CHIP(sz750).png"]:
                (chips_dir / name).write_bytes(b"chip")

            for name in ["2018-HL1.jpg", "2021-HL2 a.JPG", "2021-HL3a.HEIC"]:
                (original_dir / name).write_bytes(b"orig")

            pd.DataFrame(
                [
                    {"#ChipID": 1, "Image": "2018-HL1.jpg"},
                    {"#ChipID": 2, "Image": "2021-HL2a.jpg"},
                    {"#ChipID": 3, "Image": "2021-HL3a.jpg"},
                ]
            ).to_csv(mapping_path, index=False)

            pd.DataFrame(
                [
                    {
                        "Chip": "Chip 001",
                        "Image": "Test",
                        "Query Result": "Positive match",
                        "Rank 1 - Chip": 2,
                        "Rank 1 - Score": 123.0,
                        "Rank 2 - Chip": 3,
                        "Rank 2 - Score": 45.0,
                    },
                    {
                        "Chip": "Chip 002",
                        "Image": "Unique",
                        "Query Result": "",
                        "Rank 1 - Chip": 1,
                        "Rank 1 - Score": 99.0,
                        "Rank 2 - Chip": 3,
                        "Rank 2 - Score": 44.0,
                    },
                    {
                        "Chip": "Chip 003",
                        "Image": "Reference",
                        "Query Result": "",
                        "Rank 1 - Chip": 2,
                        "Rank 1 - Score": 88.0,
                        "Rank 2 - Chip": 1,
                        "Rank 2 - Score": 33.0,
                    },
                ]
            ).to_csv(hotspotter_output_path, index=False)

            artifacts = build_tcu_texas_dataset_artifacts(
                repo_root=root,
                chips_dir=chips_dir,
                original_dir=original_dir,
                mapping_path=mapping_path,
                hotspotter_output_path=hotspotter_output_path,
                output_dir=output_dir,
            )

            chip_manifest_df = pd.read_csv(artifacts.chip_manifest_path)
            original_manifest_df = pd.read_csv(artifacts.original_manifest_path)
            chip_audit_df = pd.read_csv(artifacts.chip_alignment_audit_path)

            self.assertEqual(len(chip_manifest_df), 3)
            self.assertEqual(len(original_manifest_df), 3)

            chip1 = chip_manifest_df.loc[chip_manifest_df["chip_id"] == 1].iloc[0]
            chip2 = chip_manifest_df.loc[chip_manifest_df["chip_id"] == 2].iloc[0]
            chip3 = chip_manifest_df.loc[chip_manifest_df["chip_id"] == 3].iloc[0]

            self.assertEqual(chip1["original_match_stage_v1"], "exact")
            self.assertEqual(chip2["original_match_stage_v1"], "normalized_name")
            self.assertEqual(chip3["original_match_stage_v1"], "normalized_stem")
            self.assertEqual(chip3["original_match_filename_v1"], "2021-HL3a.HEIC")
            self.assertEqual(chip2["external_identity_v1"], "HL2")
            self.assertEqual(int(chip2["external_identity_image_count_v1"]), 1)
            self.assertFalse(bool(chip2["supervised_warmup_candidate_v1"]))

            original3 = original_manifest_df.loc[original_manifest_df["original_filename_v1"] == "2021-HL3a.HEIC"].iloc[0]
            self.assertEqual(int(original3["mapped_chip_count_v1"]), 1)
            self.assertEqual(str(original3["mapped_chip_ids_v1"]), "3")

            self.assertEqual(set(chip_audit_df["original_match_stage_v1"].astype(str)), {"normalized_name", "normalized_stem"})


if __name__ == "__main__":
    unittest.main()
