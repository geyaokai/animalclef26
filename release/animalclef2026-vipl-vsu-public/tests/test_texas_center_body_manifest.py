from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.animalclef_analysis.texas_center_body_manifest import build_texas_center_body_manifest


class TexasCenterBodyManifestTest(unittest.TestCase):
    def test_build_texas_center_body_manifest_rewrites_primary_paths(self) -> None:
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            repo_root = tmp_path
            records_path = tmp_path / "records.csv"
            repaired_manifest_path = tmp_path / "repaired.csv"
            output_dir = tmp_path / "out"

            pd.DataFrame(
                [
                    {
                        "image_id": "1",
                        "dataset": "TexasHornedLizards",
                        "split": "test",
                        "original_path": str(tmp_path / "images" / "original.jpg"),
                        "aligned_path": str(tmp_path / "views" / "aligned.jpg"),
                        "source_aligned_path": str(tmp_path / "views" / "source.jpg"),
                        "scale_norm_path": str(tmp_path / "views" / "scale.jpg"),
                        "center_body_square_path": str(tmp_path / "views" / "square.jpg"),
                        "center_body_gray_path": str(tmp_path / "views" / "gray.jpg"),
                        "sam_stage": "none",
                        "sam_reason": "reuse_manifest",
                        "foreground_ratio_of_subject": 0.9,
                        "foreground_ratio_in_crop": 0.6,
                        "square_side_px": 440,
                        "crop_fallback_reason": "",
                        "gray_low_value": 4.0,
                        "gray_high_value": 200.0,
                        "scale_factor": 1.0,
                        "major_extent_after_px": 500.0,
                        "crop_payload_json": "{}",
                        "gray_payload_json": "{}",
                        "scale_payload_json": "{}",
                    }
                ]
            ).to_csv(records_path, index=False)

            pd.DataFrame(
                [
                    {
                        "image_id": "1",
                        "identity": "",
                        "path": "images/TexasHornedLizards/test/a.jpg",
                        "split": "test",
                        "dataset": "TexasHornedLizards",
                        "preferred_path_v1": "images/TexasHornedLizards/test/a.jpg",
                        "recommended_model_input_path_v1": "images/TexasHornedLizards/test/a.jpg",
                        "preprocess_variant_v1": "sam_trainprep_aligned_best_v1",
                        "original_rgb_path_v1": "images/TexasHornedLizards/test/a.jpg",
                        "sam_trainprep_masked_fallback_stage_v1": "sam_prompt_backoff",
                        "sam_trainprep_masked_prompt_used_v1": "animal",
                        "manifest_view_name_v1": "sam_trainprep_aligned_best_v1",
                        "manifest_view_requested_v1": "sam_trainprep_aligned_best_v1",
                        "manifest_view_resolved_v1": "sam_trainprep_aligned_best_v1",
                        "manifest_view_applied_v1": True,
                    }
                ]
            ).to_csv(repaired_manifest_path, index=False)

            outputs = build_texas_center_body_manifest(
                repo_root=repo_root,
                records_path=records_path,
                repaired_manifest_path=repaired_manifest_path,
                output_dir=output_dir,
            )

            manifest_df = pd.read_csv(outputs["manifest_path"])
            self.assertEqual(len(manifest_df), 1)
            self.assertEqual(manifest_df.loc[0, "path"], "views/gray.jpg")
            self.assertEqual(manifest_df.loc[0, "preferred_path_v1"], "views/gray.jpg")
            self.assertEqual(manifest_df.loc[0, "recommended_model_input_path_v1"], "views/gray.jpg")
            self.assertEqual(manifest_df.loc[0, "texas_center_body_repaired_fallback_stage_v1"], "sam_prompt_backoff")
            self.assertEqual(manifest_df.loc[0, "texas_center_body_repaired_prompt_v1"], "animal")
