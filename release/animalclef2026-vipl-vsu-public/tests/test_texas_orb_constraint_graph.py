from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from src.animalclef_analysis.texas_orb_constraint_graph import (
    merge_manual_and_auto_texas_judgments,
    run_texas_orb_constraint_graph_probe,
    select_texas_orb_auto_no_pairs,
)


class TexasOrbConstraintGraphTest(unittest.TestCase):
    def test_select_texas_orb_auto_no_pairs_respects_mode_and_exclusion(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "image_id": "a",
                    "neighbor_image_id": "b",
                    "base_cluster_left": 10,
                    "base_cluster_right": 10,
                    "orb_local_score": 0.20,
                    "orb_inliers": 3,
                },
                {
                    "image_id": "a",
                    "neighbor_image_id": "c",
                    "base_cluster_left": 10,
                    "base_cluster_right": 10,
                    "orb_local_score": 0.20,
                    "orb_inliers": 8,
                },
                {
                    "image_id": "b",
                    "neighbor_image_id": "c",
                    "base_cluster_left": 10,
                    "base_cluster_right": 10,
                    "orb_local_score": 0.60,
                    "orb_inliers": 3,
                },
                {
                    "image_id": "x",
                    "neighbor_image_id": "y",
                    "base_cluster_left": 10,
                    "base_cluster_right": 11,
                    "orb_local_score": 0.10,
                    "orb_inliers": 1,
                },
            ]
        )

        both_df = select_texas_orb_auto_no_pairs(pair_df, mode="both")
        either_df = select_texas_orb_auto_no_pairs(pair_df, mode="either", exclude_pair_keys={"a|b"})

        self.assertEqual(both_df["pair_key_canonical"].tolist(), ["a|b"])
        self.assertEqual(sorted(either_df["pair_key_canonical"].tolist()), ["a|c", "b|c"])

    def test_merge_manual_and_auto_texas_judgments_keeps_manual_precedence(self) -> None:
        manual_judgments = [
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "split",
                "candidate_key": "10",
                "image_id": "a",
                "neighbor_image_id": "b",
                "label": "yes",
            },
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "merge",
                "candidate_key": "a|b",
                "image_id": "a",
                "neighbor_image_id": "b",
                "label": "no",
            },
        ]
        auto_judgments = [
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "split",
                "candidate_key": "10",
                "image_id": "b",
                "neighbor_image_id": "a",
                "label": "no",
            },
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "split",
                "candidate_key": "10",
                "image_id": "a",
                "neighbor_image_id": "c",
                "label": "no",
            },
        ]

        merged = merge_manual_and_auto_texas_judgments(
            manual_judgments=manual_judgments,
            auto_judgments=auto_judgments,
        )

        self.assertEqual(len(merged), 3)
        texas_split_pairs = [
            tuple(sorted((str(item["image_id"]), str(item["neighbor_image_id"]))))
            for item in merged
            if str(item["dataset"]) == "TexasHornedLizards" and str(item["candidate_type"]) == "split"
        ]
        self.assertEqual(sorted(texas_split_pairs), [("a", "b"), ("a", "c")])
        labels = {
            tuple(sorted((str(item["image_id"]), str(item["neighbor_image_id"])))): str(item["label"])
            for item in merged
            if str(item["dataset"]) == "TexasHornedLizards" and str(item["candidate_type"]) == "split"
        }
        self.assertEqual(labels[("a", "b")], "yes")

    def test_run_texas_orb_constraint_graph_probe_builds_variant_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            predictions_dir = repo_root / "predictions"
            (predictions_dir / "tables").mkdir(parents=True, exist_ok=True)
            review_dir = repo_root / "review"
            (review_dir / "tables").mkdir(parents=True, exist_ok=True)
            image_dir = repo_root / "images"
            image_dir.mkdir(parents=True, exist_ok=True)

            image_specs = {
                "a1": (255, 120, 120),
                "a2": (255, 180, 120),
                "b1": (120, 180, 255),
                "b2": (120, 220, 180),
            }
            path_by_image: dict[str, str] = {}
            for image_id, color in image_specs.items():
                image_path = image_dir / f"{image_id}.jpg"
                Image.new("RGB", (96, 64), color).save(image_path, quality=90)
                path_by_image[image_id] = str(image_path.relative_to(repo_root))

            pred_df = pd.DataFrame(
                [
                    {
                        "image_id": "a1",
                        "dataset": "TexasHornedLizards",
                        "path": path_by_image["a1"],
                        "pred_cluster_id": 10,
                        "cluster_label": "cluster_TexasHornedLizards_10",
                        "chosen_threshold": 0.44,
                    },
                    {
                        "image_id": "a2",
                        "dataset": "TexasHornedLizards",
                        "path": path_by_image["a2"],
                        "pred_cluster_id": 10,
                        "cluster_label": "cluster_TexasHornedLizards_10",
                        "chosen_threshold": 0.44,
                    },
                    {
                        "image_id": "b1",
                        "dataset": "TexasHornedLizards",
                        "path": path_by_image["b1"],
                        "pred_cluster_id": 10,
                        "cluster_label": "cluster_TexasHornedLizards_10",
                        "chosen_threshold": 0.44,
                    },
                    {
                        "image_id": "b2",
                        "dataset": "TexasHornedLizards",
                        "path": path_by_image["b2"],
                        "pred_cluster_id": 10,
                        "cluster_label": "cluster_TexasHornedLizards_10",
                        "chosen_threshold": 0.44,
                    },
                ]
            )
            pred_df.to_csv(predictions_dir / "tables" / "test_predictions_v1.csv", index=False)

            pair_df = pd.DataFrame(
                [
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "a1",
                        "neighbor_image_id": "a2",
                        "path": path_by_image["a1"],
                        "neighbor_path": path_by_image["a2"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.99,
                        "ambiguity_score": 0.10,
                        "orb_local_score": 0.95,
                        "orb_inliers": 16,
                        "orb_good_matches": 24,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "b1",
                        "neighbor_image_id": "b2",
                        "path": path_by_image["b1"],
                        "neighbor_path": path_by_image["b2"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.98,
                        "ambiguity_score": 0.10,
                        "orb_local_score": 0.92,
                        "orb_inliers": 15,
                        "orb_good_matches": 22,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "a1",
                        "neighbor_image_id": "b1",
                        "path": path_by_image["a1"],
                        "neighbor_path": path_by_image["b1"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.40,
                        "ambiguity_score": 0.90,
                        "orb_local_score": 0.10,
                        "orb_inliers": 2,
                        "orb_good_matches": 5,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "a1",
                        "neighbor_image_id": "b2",
                        "path": path_by_image["a1"],
                        "neighbor_path": path_by_image["b2"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.39,
                        "ambiguity_score": 0.90,
                        "orb_local_score": 0.11,
                        "orb_inliers": 2,
                        "orb_good_matches": 5,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "a2",
                        "neighbor_image_id": "b1",
                        "path": path_by_image["a2"],
                        "neighbor_path": path_by_image["b1"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.38,
                        "ambiguity_score": 0.90,
                        "orb_local_score": 0.12,
                        "orb_inliers": 3,
                        "orb_good_matches": 6,
                    },
                    {
                        "dataset": "TexasHornedLizards",
                        "image_id": "a2",
                        "neighbor_image_id": "b2",
                        "path": path_by_image["a2"],
                        "neighbor_path": path_by_image["b2"],
                        "base_cluster_left": 10,
                        "base_cluster_right": 10,
                        "xgb_same_identity_prob": 0.37,
                        "ambiguity_score": 0.90,
                        "orb_local_score": 0.13,
                        "orb_inliers": 3,
                        "orb_good_matches": 6,
                    },
                ]
            )
            pair_df.to_csv(review_dir / "tables" / "test_pair_disagreement_v1.csv", index=False)

            judgments_path = repo_root / "manual_pair_review_v1.json"
            judgments_path.write_text(
                json.dumps({"session_name": "toy", "pair_judgments": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            outputs = run_texas_orb_constraint_graph_probe(
                repo_root=repo_root,
                base_predictions_path=predictions_dir,
                review_dir=review_dir,
                pair_judgments_path=judgments_path,
                output_dir=repo_root / "outputs",
            )

            self.assertTrue(outputs["summary_path"].exists())
            self.assertTrue(outputs["variant_summary_path"].exists())
            self.assertTrue(outputs["review_index_path"].exists())
            self.assertTrue(outputs["review_pack_pair_path"].exists())
            variant_summary_df = pd.read_csv(outputs["variant_summary_path"])
            manual_row = variant_summary_df[variant_summary_df["variant"] == "manual_only"].iloc[0]
            orb_row = variant_summary_df[variant_summary_df["variant"] == "orb_auto_only"].iloc[0]
            combo_row = variant_summary_df[variant_summary_df["variant"] == "manual_plus_orb"].iloc[0]

            self.assertEqual(int(manual_row["operations"]), 0)
            self.assertEqual(int(orb_row["operations"]), 2)
            self.assertEqual(int(combo_row["operations"]), 2)
            self.assertEqual(int(orb_row["clusters"]), 2)
            self.assertEqual(int(combo_row["clusters"]), 2)

            auto_pair_df = pd.read_csv(outputs["auto_pair_path"])
            self.assertEqual(len(auto_pair_df), 4)
            review_index_df = pd.read_csv(outputs["review_index_path"])
            self.assertEqual(len(review_index_df), 4)
            self.assertTrue((repo_root / "outputs" / "figures" / "auto_split_review").exists())


if __name__ == "__main__":
    unittest.main()
