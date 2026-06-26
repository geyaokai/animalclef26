from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.animalclef_analysis.descriptor_baselines import PATH_COLUMN
from src.animalclef_analysis.submission_baseline import _reorder_metadata_and_embeddings_to_reference


class SubmissionBaselineHelpersTest(unittest.TestCase):
    def test_reorder_metadata_allows_path_mismatch_when_ids_match(self) -> None:
        reference_df = pd.DataFrame(
            {
                "image_id": ["a", "b"],
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "identity": ["", ""],
                PATH_COLUMN: [
                    "images/SalamanderID2025/test/a.jpg",
                    "images/SalamanderID2025/test/b.jpg",
                ],
            }
        )
        candidate_df = pd.DataFrame(
            {
                "image_id": ["b", "a"],
                "dataset": ["SalamanderID2025", "SalamanderID2025"],
                "identity": ["", ""],
                PATH_COLUMN: [
                    "artifacts/preprocessing_baselines/v1/salamander_orientation_v1/b.jpg",
                    "artifacts/preprocessing_baselines/v1/salamander_orientation_v1/a.jpg",
                ],
            }
        )
        embeddings = np.asarray([[2.0, 20.0], [1.0, 10.0]], dtype=np.float32)

        reordered_df, reordered_embeddings = _reorder_metadata_and_embeddings_to_reference(
            reference_df=reference_df,
            candidate_df=candidate_df,
            embeddings=embeddings,
            split_name="test",
            candidate_name="fusion_cache",
        )

        self.assertEqual(reordered_df["image_id"].tolist(), ["a", "b"])
        self.assertEqual(
            reordered_df[PATH_COLUMN].tolist(),
            [
                "artifacts/preprocessing_baselines/v1/salamander_orientation_v1/a.jpg",
                "artifacts/preprocessing_baselines/v1/salamander_orientation_v1/b.jpg",
            ],
        )
        np.testing.assert_allclose(
            reordered_embeddings,
            np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        )

    def test_reorder_metadata_requires_image_id_dataset_pair(self) -> None:
        reference_df = pd.DataFrame(
            {
                "image_id": ["shared"],
                "dataset": ["TexasHornedLizards"],
                "identity": [""],
                PATH_COLUMN: ["images/TexasHornedLizards/test/shared.jpg"],
            }
        )
        candidate_df = pd.DataFrame(
            {
                "image_id": ["shared"],
                "dataset": ["SeaTurtleID2022"],
                "identity": [""],
                PATH_COLUMN: ["images/SeaTurtleID2022/test/shared.jpg"],
            }
        )
        embeddings = np.asarray([[1.0, 2.0]], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "missing image_id/dataset pairs"):
            _reorder_metadata_and_embeddings_to_reference(
                reference_df=reference_df,
                candidate_df=candidate_df,
                embeddings=embeddings,
                split_name="test",
                candidate_name="texas_cache",
            )


if __name__ == "__main__":
    unittest.main()
