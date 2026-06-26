from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.animalclef_analysis.descriptor_baselines import PATH_COLUMN
from src.animalclef_analysis.texas_selftrain import (
    TEXAS_DATASET,
    TexasPseudoBundle,
    apply_trusted_membership_to_pseudo_bundle,
    build_candidate_index_pairs,
    build_texas_training_frame,
    compute_pair_keep_ratio,
    expand_teacher_embeddings_for_training_rows,
    pick_best_texas_threshold,
    prepare_texas_pseudo_frame,
    remap_texas_paths_from_manifest,
)


class TexasSelfTrainHelpersTest(unittest.TestCase):
    def test_prepare_texas_pseudo_frame_assigns_seed_labels(self) -> None:
        assignments_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4"],
                "dataset": [TEXAS_DATASET] * 4,
                PATH_COLUMN: [f"img_{i}.jpg" for i in range(4)],
                "seed_status": ["seed", "seed", "uncertain", "seed"],
                "pseudo_identity": ["texas_seed_0001", "texas_seed_0000", "", "texas_seed_0001"],
                "component_size": [2, 1, 1, 2],
                "component_density": [1.0, 1.0, 0.0, 1.0],
            }
        )
        frame, label_map, summary_df = prepare_texas_pseudo_frame(assignments_df)
        self.assertEqual(label_map, {"texas_seed_0000": 0, "texas_seed_0001": 1})
        self.assertEqual(frame["is_seed"].tolist(), [True, True, False, True])
        self.assertEqual(frame["pseudo_label_index"].tolist(), [1, 0, -1, 1])
        self.assertEqual(
            summary_df.set_index("pseudo_identity")["size"].to_dict(),
            {"texas_seed_0000": 1, "texas_seed_0001": 2},
        )

    def test_build_candidate_index_pairs_filters_on_mutual_flag(self) -> None:
        metadata_df = pd.DataFrame({"image_id": ["10", "11", "12"]})
        pair_df = pd.DataFrame(
            {
                "image_id": ["10", "10"],
                "neighbor_image_id": ["11", "12"],
                "mutual_topk_all_routes": [True, False],
            }
        )
        self.assertEqual(build_candidate_index_pairs(metadata_df, pair_df, mutual_topk_only=True), [(0, 1)])
        self.assertEqual(build_candidate_index_pairs(metadata_df, pair_df, mutual_topk_only=False), [(0, 1), (0, 2)])

    def test_compute_pair_keep_ratio_counts_matching_pairs(self) -> None:
        labels = np.array([0, 0, 1, 2])
        ratio = compute_pair_keep_ratio(labels, [(0, 1), (1, 2), (2, 3)])
        self.assertAlmostEqual(ratio, 1 / 3, places=6)

    def test_pick_best_texas_threshold_prefers_proxy_metrics_then_stability(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "threshold": 0.38,
                    "seed_pair_agreement": 0.9,
                    "mutual_topk_pair_keep_ratio": 0.8,
                    "seed_recall_at_1": 0.7,
                    "cluster_delta_vs_teacher_anchor": 0,
                    "largest_cluster_size": 20,
                },
                {
                    "threshold": 0.40,
                    "seed_pair_agreement": 0.9,
                    "mutual_topk_pair_keep_ratio": 0.8,
                    "seed_recall_at_1": 0.7,
                    "cluster_delta_vs_teacher_anchor": 0,
                    "largest_cluster_size": 18,
                },
                {
                    "threshold": 0.42,
                    "seed_pair_agreement": 0.88,
                    "mutual_topk_pair_keep_ratio": 0.95,
                    "seed_recall_at_1": 0.9,
                    "cluster_delta_vs_teacher_anchor": 0,
                    "largest_cluster_size": 10,
                },
            ]
        )
        best_df = pick_best_texas_threshold(summary_df)
        self.assertAlmostEqual(float(best_df.iloc[0]["threshold"]), 0.42)

    def test_remap_texas_paths_from_manifest_updates_recommended_paths(self) -> None:
        all_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": [TEXAS_DATASET, TEXAS_DATASET],
                PATH_COLUMN: ["images/TexasHornedLizards/test/1.jpg", "images/TexasHornedLizards/test/2.jpg"],
                "is_seed": [True, False],
            }
        )
        pseudo_bundle = TexasPseudoBundle(
            all_df=all_df,
            seed_df=all_df.iloc[[0]].copy().reset_index(drop=True),
            candidate_pair_df=pd.DataFrame(),
            seed_class_summary_df=pd.DataFrame(),
            pseudo_label_map={"texas_seed_0001": 0},
        )
        with TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "texas_manifest.csv"
            pd.DataFrame(
                {
                    "image_id": ["1", "2"],
                    "dataset": [TEXAS_DATASET, TEXAS_DATASET],
                    PATH_COLUMN: [
                        "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/1.jpg",
                        "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/2.jpg",
                    ],
                    "preferred_path_v1": [
                        "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/1.jpg",
                        "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/2.jpg",
                    ],
                }
            ).to_csv(manifest_path, index=False)
            remapped = remap_texas_paths_from_manifest(pseudo_bundle, manifest_path=manifest_path)
        self.assertEqual(
            remapped.all_df[PATH_COLUMN].tolist(),
            [
                "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/1.jpg",
                "artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/2.jpg",
            ],
        )
        self.assertEqual(
            remapped.seed_df[PATH_COLUMN].tolist(),
            ["artifacts/manifests/sam_seg_trainprep_v1/views/TexasHornedLizards/test/1.jpg"],
        )

    def test_apply_trusted_membership_overrides_seed_assignments(self) -> None:
        base_df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3"],
                "dataset": [TEXAS_DATASET] * 3,
                PATH_COLUMN: ["a.jpg", "b.jpg", "c.jpg"],
                "seed_status": ["uncertain", "seed", "uncertain"],
                "pseudo_identity": ["", "old_seed", ""],
            }
        )
        bundle = TexasPseudoBundle(
            all_df=prepare_texas_pseudo_frame(base_df)[0],
            seed_df=pd.DataFrame(),
            candidate_pair_df=pd.DataFrame(),
            seed_class_summary_df=pd.DataFrame(),
            pseudo_label_map={},
        )
        with TemporaryDirectory() as tmp_dir:
            trusted_path = Path(tmp_dir) / "trusted.csv"
            pd.DataFrame(
                {
                    "image_id": ["1", "3"],
                    "dataset": [TEXAS_DATASET, TEXAS_DATASET],
                    "trusted_component_id": ["trusted_0", "trusted_1"],
                }
            ).to_csv(trusted_path, index=False)
            updated = apply_trusted_membership_to_pseudo_bundle(bundle, trusted_membership_path=trusted_path)
        self.assertEqual(updated.all_df["is_seed"].tolist(), [True, False, True])
        self.assertEqual(updated.seed_df["pseudo_identity"].tolist(), ["trusted_0", "trusted_1"])

    def test_build_training_frame_and_expand_teacher_rows(self) -> None:
        all_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": [TEXAS_DATASET, TEXAS_DATASET],
                PATH_COLUMN: ["a.jpg", "b.jpg"],
                "is_seed": [True, False],
                "pseudo_label_index": [0, -1],
            }
        )
        bundle = TexasPseudoBundle(
            all_df=all_df,
            seed_df=all_df.iloc[[0]].copy(),
            candidate_pair_df=pd.DataFrame(),
            seed_class_summary_df=pd.DataFrame(),
            pseudo_label_map={"trusted_0": 0},
        )
        with TemporaryDirectory() as tmp_dir:
            pair_path = Path(tmp_dir) / "pairs.csv"
            pd.DataFrame(
                {
                    "image_id": ["1", "1", "2"],
                    "dataset": [TEXAS_DATASET] * 3,
                    "positive_recipe": ["hflip", "center_crop_92", "mild_rotate_p6"],
                    "pair_source": ["trusted", "trusted", "fallback"],
                }
            ).to_csv(pair_path, index=False)
            training_df = build_texas_training_frame(bundle, pseudo_positive_pairs_path=pair_path)
        self.assertEqual(training_df["positive_recipe"].tolist(), ["hflip", "center_crop_92", "mild_rotate_p6"])
        teacher = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        expanded = expand_teacher_embeddings_for_training_rows(training_df, all_df, teacher)
        self.assertEqual(tuple(expanded.shape), (3, 2))
        self.assertTrue(np.allclose(expanded[0], teacher[0]))
        self.assertTrue(np.allclose(expanded[1], teacher[0]))
        self.assertTrue(np.allclose(expanded[2], teacher[1]))

    def test_build_training_frame_accepts_positive_recipe_name_schema(self) -> None:
        all_df = pd.DataFrame(
            {
                "image_id": ["1"],
                "dataset": [TEXAS_DATASET],
                PATH_COLUMN: ["a.jpg"],
                "is_seed": [True],
                "pseudo_label_index": [0],
            }
        )
        bundle = TexasPseudoBundle(
            all_df=all_df,
            seed_df=all_df.copy(),
            candidate_pair_df=pd.DataFrame(),
            seed_class_summary_df=pd.DataFrame(),
            pseudo_label_map={"trusted_0": 0},
        )
        with TemporaryDirectory() as tmp_dir:
            pair_path = Path(tmp_dir) / "pairs.csv"
            pd.DataFrame(
                {
                    "image_id": ["1"],
                    "dataset": [TEXAS_DATASET],
                    "positive_recipe_name": ["rotate_mild_pos5_v1"],
                    "pair_kind": ["intra_image_pseudo_positive"],
                }
            ).to_csv(pair_path, index=False)
            training_df = build_texas_training_frame(bundle, pseudo_positive_pairs_path=pair_path)
        self.assertEqual(training_df["positive_recipe"].tolist(), ["rotate_mild_pos5_v1"])
        self.assertEqual(training_df["pair_source"].tolist(), ["intra_image_pseudo_positive"])

    def test_build_training_frame_adds_fallback_view_rows_for_unmatched_images(self) -> None:
        all_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": [TEXAS_DATASET, TEXAS_DATASET],
                PATH_COLUMN: ["a.jpg", "b.jpg"],
                "is_seed": [True, False],
                "pseudo_label_index": [0, -1],
            }
        )
        bundle = TexasPseudoBundle(
            all_df=all_df,
            seed_df=all_df.iloc[[0]].copy(),
            candidate_pair_df=pd.DataFrame(),
            seed_class_summary_df=pd.DataFrame(),
            pseudo_label_map={"trusted_0": 0},
        )
        with TemporaryDirectory() as tmp_dir:
            pair_path = Path(tmp_dir) / "pairs.csv"
            pd.DataFrame(
                {
                    "image_id": ["1"],
                    "dataset": [TEXAS_DATASET],
                    "positive_recipe_name": ["rotate_mild_pos5_v1"],
                    "pair_kind": ["trusted_pairs"],
                }
            ).to_csv(pair_path, index=False)
            training_df = build_texas_training_frame(bundle, pseudo_positive_pairs_path=pair_path)
        image1 = training_df[training_df["image_id"].eq("1")].copy()
        image2 = training_df[training_df["image_id"].eq("2")].copy()
        self.assertEqual(image1["positive_recipe"].tolist(), ["rotate_mild_pos5_v1"])
        self.assertEqual(sorted(image2["positive_recipe"].tolist()), ["crop_jitter_tight_v1", "rotate_mild_neg5_v1", "rotate_mild_pos5_v1", "scale_focus_in_v1"])
        self.assertTrue(image2["pair_source"].astype(str).eq("auto_single_view_fallback_unmatched").all())


if __name__ == "__main__":
    unittest.main()
