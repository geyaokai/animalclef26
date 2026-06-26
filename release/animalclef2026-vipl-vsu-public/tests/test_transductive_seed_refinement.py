from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from animalclef_analysis.transductive_seed_refinement import (  # noqa: E402
    apply_reverse_neighbor_penalty,
    compute_reverse_neighbor_counts,
)


class TransductiveSeedRefinementTest(unittest.TestCase):
    def setUp(self) -> None:
        self.score = np.array(
            [
                [1.0, 0.9, 0.8, 0.1],
                [0.9, 1.0, 0.7, 0.2],
                [0.8, 0.7, 1.0, 0.3],
                [0.1, 0.2, 0.3, 1.0],
            ],
            dtype=np.float32,
        )

    def test_compute_reverse_neighbor_counts_counts_topk_hits(self) -> None:
        reverse_counts, neighbor_index = compute_reverse_neighbor_counts(score_matrix=self.score, top_k=1)
        self.assertEqual(neighbor_index.shape, (4, 1))
        self.assertEqual(reverse_counts.tolist(), [2.0, 1.0, 1.0, 0.0])

    def test_apply_reverse_neighbor_penalty_only_penalizes_hubs(self) -> None:
        penalized, diagnostics = apply_reverse_neighbor_penalty(
            score_matrix=self.score,
            top_k=1,
            penalty_scale=0.1,
        )
        self.assertEqual(penalized.shape, self.score.shape)
        self.assertTrue(np.allclose(np.diag(penalized), 1.0))
        self.assertLess(float(penalized[0, 1]), float(self.score[0, 1]))
        self.assertLess(float(penalized[0, 3]), float(self.score[0, 3]))
        self.assertAlmostEqual(float(penalized[1, 2]), float(self.score[1, 2]), places=6)
        self.assertAlmostEqual(float(diagnostics["hub_image_ratio"]), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
