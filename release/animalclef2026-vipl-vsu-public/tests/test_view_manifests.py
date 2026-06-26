from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.animalclef_analysis.frozen_view_gate import build_delta_table
from src.animalclef_analysis.view_manifests import (
    BODY_AXIS_VIEW_NAME,
    DEFAULT_VIEW_NAME,
    SAM_MASKED_VIEW_NAME,
    create_metadata_enriched,
    build_view_manifest,
    get_default_manifest_paths,
)


class ViewManifestTest(unittest.TestCase):
    def _metadata_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "image_id": ["1", "2", "3"],
                "identity": ["s1", "s2", ""],
                "path": [
                    "images/SalamanderID2025/train/s1/a.jpg",
                    "images/SalamanderID2025/train/s2/b.jpg",
                    "images/TexasHornedLizards/test/c.jpg",
                ],
                "date": ["", "", ""],
                "orientation": ["left", "right", "top"],
                "species": ["salamander", "salamander", "lizard"],
                "split": ["train", "train", "test"],
                "dataset": ["SalamanderID2025", "SalamanderID2025", "TexasHornedLizards"],
            }
        )

    def test_get_default_manifest_paths_points_to_new_manifest_root(self) -> None:
        train_path, test_path = get_default_manifest_paths(Path("/repo"))
        self.assertEqual(str(train_path), "/repo/artifacts/manifests/v1/tables/manifest_train_default_v1.csv")
        self.assertEqual(str(test_path), "/repo/artifacts/manifests/v1/tables/manifest_test_default_v1.csv")

    def test_build_view_manifest_uses_body_axis_when_applied(self) -> None:
        body_axis_export_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "split": ["train", "train"],
                "path": [
                    "images/SalamanderID2025/train/s1/a.jpg",
                    "images/SalamanderID2025/train/s2/b.jpg",
                ],
                "body_axis_unsigned_rgb_v1_prompt": ["salamander body", "salamander body"],
                "body_axis_unsigned_rgb_v1_mask_count": [1, 1],
                "body_axis_unsigned_rgb_v1_union_area_ratio": [0.1, 0.1],
                "body_axis_unsigned_rgb_v1_largest_component_ratio": [0.9, 0.9],
                "body_axis_unsigned_rgb_v1_foreground_pixels": [1000.0, 1000.0],
                "body_axis_unsigned_rgb_v1_foreground_area_ratio": [0.1, 0.1],
                "body_axis_unsigned_rgb_v1_bbox_fill_ratio": [0.8, 0.8],
                "body_axis_unsigned_rgb_v1_axis_angle_deg": [10.0, 5.0],
                "body_axis_unsigned_rgb_v1_axis_confidence": [0.8, 0.2],
                "body_axis_unsigned_rgb_v1_rotation_applied_deg": [-10.0, 0.0],
                "body_axis_unsigned_rgb_v1_aligned_foreground_ratio": [0.2, 0.1],
                "body_axis_unsigned_rgb_v1_padding_ratio": [0.06, 0.06],
                "body_axis_unsigned_rgb_v1_status": ["apply", "skip"],
                "body_axis_unsigned_rgb_v1_reason": ["ok", "low_axis_confidence"],
                "body_axis_unsigned_rgb_v1_applied": [True, False],
                "body_axis_unsigned_rgb_v1_export_path": [
                    "artifacts/manifests/v1/views/body_axis_unsigned_rgb_v1/SalamanderID2025/train/s1/a.jpg",
                    "",
                ],
                "body_axis_unsigned_rgb_v1_canvas_fill_mode": ["edge", "edge"],
            }
        )
        enriched_df = create_metadata_enriched(
            metadata_df=self._metadata_df(),
            body_axis_export_df=body_axis_export_df,
        )
        train_manifest_df = build_view_manifest(
            enriched_df=enriched_df,
            split="train",
            view_name=BODY_AXIS_VIEW_NAME,
        )
        self.assertEqual(train_manifest_df["image_id"].tolist(), ["1", "2"])
        self.assertEqual(
            train_manifest_df["recommended_model_input_path_v1"].tolist(),
            [
                "artifacts/manifests/v1/views/body_axis_unsigned_rgb_v1/SalamanderID2025/train/s1/a.jpg",
                "images/SalamanderID2025/train/s2/b.jpg",
            ],
        )
        self.assertEqual(
            train_manifest_df["manifest_view_resolved_v1"].tolist(),
            [BODY_AXIS_VIEW_NAME, DEFAULT_VIEW_NAME],
        )
        self.assertEqual(train_manifest_df["manifest_view_applied_v1"].tolist(), [True, False])

    def test_build_view_manifest_original_keeps_original_path(self) -> None:
        enriched_df = create_metadata_enriched(metadata_df=self._metadata_df())
        test_manifest_df = build_view_manifest(
            enriched_df=enriched_df,
            split="test",
            view_name=DEFAULT_VIEW_NAME,
        )
        self.assertEqual(test_manifest_df["image_id"].tolist(), ["3"])
        self.assertEqual(
            test_manifest_df.iloc[0]["recommended_model_input_path_v1"],
            "images/TexasHornedLizards/test/c.jpg",
        )
        self.assertEqual(test_manifest_df.iloc[0]["manifest_view_resolved_v1"], DEFAULT_VIEW_NAME)

    def test_build_view_manifest_uses_sam_masked_when_applied(self) -> None:
        sam_masked_export_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "split": ["train", "train"],
                "path": [
                    "images/SalamanderID2025/train/s1/a.jpg",
                    "images/SalamanderID2025/train/s2/b.jpg",
                ],
                "sam_masked_rgb_v1_prompt": ["salamander", "salamander"],
                "sam_masked_rgb_v1_mask_count": [1, 0],
                "sam_masked_rgb_v1_union_area_ratio": [0.12, 0.0],
                "sam_masked_rgb_v1_largest_component_ratio": [0.94, 0.0],
                "sam_masked_rgb_v1_foreground_pixels": [2048.0, 0.0],
                "sam_masked_rgb_v1_foreground_area_ratio": [0.12, 0.0],
                "sam_masked_rgb_v1_best_score": [0.91, 0.0],
                "sam_masked_rgb_v1_status": ["apply", "skip"],
                "sam_masked_rgb_v1_reason": ["ok", "no_mask"],
                "sam_masked_rgb_v1_applied": [True, False],
                "sam_masked_rgb_v1_export_path": [
                    "artifacts/manifests/v1/views/sam_masked_rgb_v1/SalamanderID2025/train/s1/a.jpg",
                    "",
                ],
            }
        )
        enriched_df = create_metadata_enriched(
            metadata_df=self._metadata_df(),
            sam_masked_export_df=sam_masked_export_df,
        )
        train_manifest_df = build_view_manifest(
            enriched_df=enriched_df,
            split="train",
            view_name=SAM_MASKED_VIEW_NAME,
        )
        self.assertEqual(train_manifest_df["image_id"].tolist(), ["1", "2"])
        self.assertEqual(
            train_manifest_df["recommended_model_input_path_v1"].tolist(),
            [
                "artifacts/manifests/v1/views/sam_masked_rgb_v1/SalamanderID2025/train/s1/a.jpg",
                "images/SalamanderID2025/train/s2/b.jpg",
            ],
        )
        self.assertEqual(
            train_manifest_df["manifest_view_resolved_v1"].tolist(),
            [SAM_MASKED_VIEW_NAME, DEFAULT_VIEW_NAME],
        )
        self.assertEqual(train_manifest_df["manifest_view_applied_v1"].tolist(), [True, False])


class FrozenViewGateSummaryTest(unittest.TestCase):
    def test_build_delta_table_compares_against_original_view(self) -> None:
        metrics_df = pd.DataFrame(
            {
                "view_name": [DEFAULT_VIEW_NAME, BODY_AXIS_VIEW_NAME],
                "descriptor": ["miew", "miew"],
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "ari": [0.20, 0.26],
                "recall_at_1": [0.51, 0.57],
            }
        )
        delta_df = build_delta_table(metrics_df=metrics_df, baseline_view=DEFAULT_VIEW_NAME)
        self.assertEqual(len(delta_df), 1)
        self.assertAlmostEqual(float(delta_df.iloc[0]["delta_ari"]), 0.06, places=6)
        self.assertAlmostEqual(float(delta_df.iloc[0]["delta_recall_at_1"]), 0.06, places=6)


if __name__ == "__main__":
    unittest.main()
