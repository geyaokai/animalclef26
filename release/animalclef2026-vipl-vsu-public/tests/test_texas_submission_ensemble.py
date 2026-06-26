from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.descriptor_baselines import PATH_COLUMN
from src.animalclef_analysis.texas_submission_ensemble import _reorder_texas_embeddings_to_reference


class TexasSubmissionEnsembleHelpersTest(unittest.TestCase):
    def test_reorder_texas_embeddings_allows_path_mismatch(self) -> None:
        reference_df = pd.DataFrame(
            {
                "image_id": ["1", "2"],
                "dataset": ["TexasHornedLizards", "TexasHornedLizards"],
                "identity": ["", ""],
                PATH_COLUMN: [
                    "images/TexasHornedLizards/test/1.jpg",
                    "images/TexasHornedLizards/test/2.jpg",
                ],
            }
        )
        candidate_df = pd.DataFrame(
            {
                "image_id": ["2", "1"],
                "dataset": ["TexasHornedLizards", "TexasHornedLizards"],
                "identity": ["", ""],
                PATH_COLUMN: [
                    "artifacts/alt/2.jpg",
                    "artifacts/alt/1.jpg",
                ],
            }
        )
        embeddings = np.asarray([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32)

        reordered_df, reordered_embeddings = _reorder_texas_embeddings_to_reference(
            reference_df=reference_df,
            candidate_df=candidate_df,
            embeddings=embeddings,
            candidate_name="toy",
        )

        self.assertEqual(reordered_df["image_id"].tolist(), ["1", "2"])
        np.testing.assert_allclose(
            reordered_embeddings,
            np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        )

    def test_reorder_texas_embeddings_requires_dataset_match(self) -> None:
        reference_df = pd.DataFrame(
            {
                "image_id": ["1"],
                "dataset": ["TexasHornedLizards"],
                "identity": [""],
                PATH_COLUMN: ["images/TexasHornedLizards/test/1.jpg"],
            }
        )
        candidate_df = pd.DataFrame(
            {
                "image_id": ["1"],
                "dataset": ["SeaTurtleID2022"],
                "identity": [""],
                PATH_COLUMN: ["images/SeaTurtleID2022/test/1.jpg"],
            }
        )
        embeddings = np.asarray([[1.0, 10.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Missing Texas image_id/dataset pairs"):
            _reorder_texas_embeddings_to_reference(
                reference_df=reference_df,
                candidate_df=candidate_df,
                embeddings=embeddings,
                candidate_name="toy",
            )


if __name__ == "__main__":
    unittest.main()
