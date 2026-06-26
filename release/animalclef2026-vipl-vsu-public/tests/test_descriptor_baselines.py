from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

try:
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    HAS_CLUSTER_DEPS = True
except ModuleNotFoundError:
    HAS_CLUSTER_DEPS = False

from src.animalclef_analysis.descriptor_baselines import (
    build_identity_holdout_split,
    cluster_from_linkage,
    cosine_distance_matrix,
    ensure_metadata_alignment,
    fuse_embedding_blocks,
    pick_best_thresholds,
    recall_at_k,
    run_threshold_sweep,
)


class DescriptorFusionHelpersTest(unittest.TestCase):
    def test_ensure_metadata_alignment_detects_order_mismatch(self) -> None:
        left = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["LynxID2025", "LynxID2025"],
                "identity": ["a", "b"],
                "recommended_model_input_path_v1": ["a.jpg", "b.jpg"],
            }
        )
        right = left.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "metadata order mismatch"):
            ensure_metadata_alignment(
                reference_df=left,
                candidate_df=right,
                split_name="val",
                reference_name="left",
                candidate_name="right",
            )

    def test_fuse_embedding_blocks_concatenates_and_normalizes(self) -> None:
        first = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        second = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
        fused = fuse_embedding_blocks([first, second], weights=[1.0, 0.5])
        self.assertEqual(fused.shape, (2, 4))
        self.assertTrue(np.allclose(np.linalg.norm(fused, axis=1), 1.0))
        self.assertGreater(float(fused[0, 0]), 0.0)
        self.assertGreater(float(fused[0, 2]), 0.0)


@unittest.skipUnless(HAS_CLUSTER_DEPS, "scipy and scikit-learn are required for descriptor baseline tests")
class DescriptorBaselinesTest(unittest.TestCase):
    def test_build_identity_holdout_split_keeps_identities_together(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": [str(i) for i in range(8)],
                "identity": ["a", "a", "b", "b", "c", "c", "d", "d"],
                "dataset": ["LynxID2025"] * 8,
                "recommended_model_input_path_v1": [f"img_{i}.jpg" for i in range(8)],
            }
        )
        split_df = build_identity_holdout_split(
            train_df=df,
            val_identity_fraction=0.25,
            seed=0,
            datasets=["LynxID2025"],
        )
        roles = split_df.groupby("identity")["split_role_v1"].nunique().to_dict()
        self.assertEqual(set(roles.values()), {1})
        self.assertIn("val", set(split_df["split_role_v1"]))
        self.assertIn("fit", set(split_df["split_role_v1"]))

    def test_recall_at_k_ignores_singletons(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        labels = np.array(["x", "x", "y"])
        self.assertAlmostEqual(recall_at_k(embeddings, labels, k=1), 1.0)

    def test_threshold_sweep_and_best_pick(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": ["1", "2", "3", "4"],
                "dataset": ["LynxID2025"] * 4,
                "identity": ["a", "a", "b", "b"],
                "recommended_model_input_path_v1": ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"],
            }
        )
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
                [0.01, 0.99],
            ],
            dtype=np.float32,
        )
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sweep_df, prediction_df = run_threshold_sweep(df=df, embeddings=embeddings, thresholds=[0.1, 1.0])
        self.assertEqual(len(prediction_df), 8)
        best_df = pick_best_thresholds(sweep_df)
        self.assertEqual(float(best_df.iloc[0]["threshold"]), 0.1)
        self.assertAlmostEqual(float(best_df.iloc[0]["ari"]), 1.0)

    def test_cluster_from_linkage_handles_single_sample(self) -> None:
        labels = cluster_from_linkage(linkage_matrix=None, sample_count=1, threshold=0.3)
        self.assertTrue(np.array_equal(labels, np.array([0])))

    def test_cosine_distance_matrix_has_zero_diagonal(self) -> None:
        embeddings = np.eye(3, dtype=np.float32)
        distance = cosine_distance_matrix(embeddings)
        self.assertTrue(np.allclose(np.diag(distance), 0.0))


if __name__ == "__main__":
    unittest.main()
