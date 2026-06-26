from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.animalclef_analysis.orb_rerank_baseline import (
    apply_local_rerank,
    build_smoke_summary,
    build_topk_pair_index,
    cosine_score_matrix,
    normalize_local_matcher_name,
    recall_at_k_from_score_matrix,
    resolve_existing_image_rel_path,
)


class OrbRerankHelpersTest(unittest.TestCase):
    def test_cosine_score_matrix_maps_unit_embeddings_to_zero_one_range(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        score = cosine_score_matrix(embeddings)
        self.assertEqual(score.shape, (2, 2))
        self.assertTrue(np.allclose(np.diag(score), 1.0))
        self.assertAlmostEqual(float(score[0, 1]), 0.5)

    def test_build_topk_pair_index_is_unique_and_query_scoped(self) -> None:
        score_matrix = np.array(
            [
                [1.0, 0.9, 0.2],
                [0.9, 1.0, 0.8],
                [0.2, 0.8, 1.0],
            ],
            dtype=np.float32,
        )
        pairs = build_topk_pair_index(score_matrix=score_matrix, top_k=2, query_indices=np.array([0, 1]))
        self.assertEqual(len(pairs), 3)
        self.assertEqual((pairs[0][0], pairs[0][1]), (0, 1))
        self.assertEqual((pairs[-1][0], pairs[-1][1]), (1, 2))

    def test_normalize_local_matcher_name_accepts_supported_matchers(self) -> None:
        self.assertEqual(normalize_local_matcher_name("ORB"), "orb")
        self.assertEqual(normalize_local_matcher_name("sift"), "sift")

    def test_normalize_local_matcher_name_rejects_unknown_matcher(self) -> None:
        with self.assertRaises(ValueError):
            normalize_local_matcher_name("surf")

    def test_apply_local_rerank_only_boosts_selected_pairs(self) -> None:
        global_score = np.array(
            [
                [1.0, 0.7, 0.6],
                [0.7, 1.0, 0.2],
                [0.6, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        pair_df = pd.DataFrame(
            {
                "left_index": [0],
                "right_index": [1],
                "global_score": [0.7],
                "local_score": [1.0],
            }
        )
        reranked = apply_local_rerank(global_score_matrix=global_score, pair_df=pair_df, local_weight=0.5)
        self.assertAlmostEqual(float(reranked[0, 1]), 0.85, places=6)
        self.assertAlmostEqual(float(reranked[1, 0]), 0.85, places=6)
        self.assertAlmostEqual(float(reranked[0, 2]), 0.6, places=6)

    def test_recall_at_k_from_score_matrix_ignores_singletons(self) -> None:
        score_matrix = np.array(
            [
                [1.0, 0.95, 0.1],
                [0.95, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.array(["a", "a", "b"])
        self.assertAlmostEqual(recall_at_k_from_score_matrix(score_matrix, labels, k=1), 1.0)

    def test_build_smoke_summary_enables_when_rerank_corrects_queries(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["LynxID2025"] * 4,
                "identity": ["a", "a", "b", "b"],
            }
        )
        global_score = np.array(
            [
                [1.0, 0.55, 0.7, 0.2],
                [0.55, 1.0, 0.4, 0.95],
                [0.7, 0.4, 1.0, 0.5],
                [0.2, 0.95, 0.5, 1.0],
            ],
            dtype=np.float32,
        )
        local_match_df = pd.DataFrame(
            {
                "dataset": ["LynxID2025", "LynxID2025", "LynxID2025", "LynxID2025"],
                "left_index": [0, 1, 0, 1],
                "right_index": [1, 3, 2, 2],
                "same_identity": [True, False, False, False],
                "global_score": [0.55, 0.95, 0.7, 0.4],
                "local_score": [1.0, 0.0, 0.0, 0.0],
            }
        )
        smoke_df, gate_df = build_smoke_summary(
            df=df,
            global_score_matrix=global_score,
            local_match_df=local_match_df,
            local_weights=[0.5],
            smoke_query_indices=np.array([0]),
        )
        self.assertEqual(len(smoke_df), 1)
        self.assertTrue(bool(gate_df.iloc[0]["enable_rerank"]))
        self.assertAlmostEqual(float(gate_df.iloc[0]["chosen_local_weight"]), 0.5)

    def test_resolve_existing_image_rel_path_falls_back_to_original_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir)
            image_rel = "images/SalamanderID2025/test/example.jpg"
            image_path = repo_root / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"test")
            row = pd.Series(
                {
                    "recommended_model_input_path_v1": "artifacts/missing/example.jpg",
                    "path": image_rel,
                    "preferred_path_v1": "artifacts/missing/example.jpg",
                }
            )
            self.assertEqual(resolve_existing_image_rel_path(row, repo_root=repo_root), image_rel)


if __name__ == "__main__":
    unittest.main()
