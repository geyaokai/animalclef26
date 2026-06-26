from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.local_matching.hotspotter_pipeline import (
    HotspotterConfig,
    HotspotterFeature,
    _trim_features_with_strategy,
    build_hotspotter_index,
    prescore_results_to_dataframe,
    query_hotspotter_all,
    query_neighbors,
    rank_results_to_dataframe,
    rootsift_descriptors,
    unique_pair_results_to_dataframe,
)
from src.animalclef_analysis.local_matching.hotspotter_spatial import homography_inliers
from src.animalclef_analysis.texas_hotspotter_probe import load_texas_view_df


class HotspotterLocalMatchingTest(unittest.TestCase):
    def test_rootsift_descriptors_returns_unit_norm_rows(self) -> None:
        desc = np.array([[1, 3, 0], [5, 5, 10]], dtype=np.uint8)
        normalized = rootsift_descriptors(desc)
        norms = np.linalg.norm(normalized, axis=1)
        self.assertTrue(np.allclose(norms, 1.0))
        self.assertEqual(normalized.dtype, np.float32)

    def test_query_neighbors_returns_self_first_for_exact_descriptors(self) -> None:
        feature_a = HotspotterFeature(
            image_id="a",
            rel_path="a.png",
            width=10,
            height=10,
            keypoints=np.zeros((2, 6), dtype=np.float32),
            descriptors=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        feature_b = HotspotterFeature(
            image_id="b",
            rel_path="b.png",
            width=10,
            height=10,
            keypoints=np.zeros((1, 6), dtype=np.float32),
            descriptors=np.array([[0.9, 0.1]], dtype=np.float32),
        )
        index = build_hotspotter_index([feature_a, feature_b])
        nn_idx, nn_dist = query_neighbors(index=index, query_desc=feature_a.descriptors, num_neighbors=2)
        self.assertEqual(nn_idx.shape, (2, 2))
        self.assertEqual(int(nn_idx[0, 0]), 0)
        self.assertAlmostEqual(float(nn_dist[0, 0]), 0.0, places=6)

    def test_trim_features_with_detection_order_keeps_original_prefix(self) -> None:
        keypoints = np.array(
            [
                [0, 0, 5, 0, 5, 0],
                [1, 1, 1, 0, 1, 0],
                [2, 2, 9, 0, 9, 0],
            ],
            dtype=np.float32,
        )
        descriptors = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        trimmed_kpts, trimmed_desc = _trim_features_with_strategy(
            keypoints=keypoints,
            descriptors=descriptors,
            limit=2,
            strategy="detection_order",
        )
        self.assertTrue(np.array_equal(trimmed_kpts[:, 0], np.array([0, 1], dtype=np.float32)))
        self.assertTrue(np.array_equal(trimmed_desc[:, 0], np.array([0, 4], dtype=np.uint8)))

    def test_homography_inliers_accepts_consistent_affine_matches(self) -> None:
        kpts1 = np.array(
            [
                [0, 0, 1, 0, 1, 0],
                [10, 0, 1, 0, 1, 0],
                [0, 10, 1, 0, 1, 0],
                [10, 10, 1, 0, 1, 0],
            ],
            dtype=np.float32,
        )
        kpts2 = kpts1.copy()
        kpts2[:, 0] += 5
        kpts2[:, 1] += 3
        fm = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32)
        sv_tup = homography_inliers(
            kpts1=kpts1,
            kpts2=kpts2,
            fm=fm,
            xy_thresh=0.05,
            max_scale=2.0,
            min_scale=0.5,
            min_num_inliers=4,
        )
        self.assertIsNotNone(sv_tup)
        _, inliers = sv_tup
        self.assertGreaterEqual(len(inliers), 2)

    def test_query_hotspotter_all_keeps_unique_best_pair(self) -> None:
        descriptors = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.7, 0.7, 0.0],
                [0.6, 0.2, 0.7],
                [0.2, 0.8, 0.5],
            ],
            dtype=np.float32,
        )
        feature_a = HotspotterFeature(
            image_id="a",
            rel_path="a.png",
            width=10,
            height=10,
            keypoints=np.array(
                [[0, 0, 1, 0, 1, 0], [10, 0, 1, 0, 1, 0], [0, 10, 1, 0, 1, 0], [10, 10, 1, 0, 1, 0]],
                dtype=np.float32,
            ),
            descriptors=descriptors[[0, 1, 4, 5]],
        )
        feature_b = HotspotterFeature(
            image_id="b",
            rel_path="b.png",
            width=10,
            height=10,
            keypoints=np.array(
                [[1, 1, 1, 0, 1, 0], [11, 1, 1, 0, 1, 0], [1, 11, 1, 0, 1, 0], [11, 11, 1, 0, 1, 0]],
                dtype=np.float32,
            ),
            descriptors=descriptors[[0, 1, 4, 5]],
        )
        feature_c = HotspotterFeature(
            image_id="c",
            rel_path="c.png",
            width=10,
            height=10,
            keypoints=np.array(
                [[0, 0, 1, 0, 1, 0], [10, 0, 1, 0, 1, 0], [0, 10, 1, 0, 1, 0], [10, 10, 1, 0, 1, 0]],
                dtype=np.float32,
            ),
            descriptors=descriptors[[2, 3, 2, 3]],
        )
        config = HotspotterConfig(k=3, knorm=1, n_shortlist=2, min_n_inliers=4, xy_thresh=0.05)
        results = query_hotspotter_all(features=[feature_a, feature_b, feature_c], config=config)
        prescore_df = prescore_results_to_dataframe(features=[feature_a, feature_b, feature_c], query_results=results, top_k=2)
        ranking_df = rank_results_to_dataframe(features=[feature_a, feature_b, feature_c], query_results=results, top_k=2)
        pair_df = unique_pair_results_to_dataframe(features=[feature_a, feature_b, feature_c], query_results=results, top_k=2)
        self.assertFalse(prescore_df.empty)
        self.assertIn("local_prescore", prescore_df.columns)
        self.assertFalse(ranking_df.empty)
        self.assertIn("local_score", ranking_df.columns)
        self.assertFalse(pair_df.empty)
        pair_ids = [set((row["image_id"], row["neighbor_image_id"])) for _, row in pair_df.iterrows()]
        self.assertIn({"a", "b"}, pair_ids)

    def test_load_texas_view_df_requires_model_input_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "manifest.csv"
            pd.DataFrame(
                {
                    "image_id": ["1"],
                    "dataset": ["TexasHornedLizards"],
                    "recommended_model_input_path_v1": ["images/TexasHornedLizards/test/1.png"],
                }
            ).to_csv(manifest_path, index=False)
            loaded = load_texas_view_df(manifest_path)
            self.assertEqual(loaded.iloc[0]["image_id"], "1")

    def test_extract_pipeline_uses_manifest_paths(self) -> None:
        try:
            import pyhesaff  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("pyhesaff unavailable in this environment")
        from src.animalclef_analysis.local_matching.hotspotter_pipeline import extract_hesaff_features

        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            image_rel = "images/TexasHornedLizards/test/sample.png"
            image_path = repo_root / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.pad(np.ones((24, 24), dtype=np.uint8) * 255, 12)).save(image_path)
            df = pd.DataFrame(
                {
                    "image_id": ["sample"],
                    "recommended_model_input_path_v1": [image_rel],
                    "path": [image_rel],
                }
            )
            features = extract_hesaff_features(df=df, repo_root=repo_root, config=HotspotterConfig())
            self.assertEqual(len(features), 1)
            self.assertEqual(features[0].image_id, "sample")


if __name__ == "__main__":
    unittest.main()
