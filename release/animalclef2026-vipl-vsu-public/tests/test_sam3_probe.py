from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from PIL import Image

from src.animalclef_analysis.sam3_probe import (
    PROMPT_CANDIDATES_BY_DATASET,
    PROMPTS_BY_DATASET,
    SKIP_DATASETS,
    crop_to_union_mask,
    mask_bbox,
    sample_rows_by_dataset,
)


class Sam3ProbeTest(unittest.TestCase):
    def test_prompts_cover_non_skipped_probe_datasets(self) -> None:
        expected = {"LynxID2025", "SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"}
        self.assertEqual(set(PROMPTS_BY_DATASET), expected)
        self.assertFalse(SKIP_DATASETS)
        self.assertEqual(PROMPTS_BY_DATASET["SeaTurtleID2022"], "sea turtle head")
        self.assertEqual(
            PROMPT_CANDIDATES_BY_DATASET["SalamanderID2025"],
            ["salamander", "salamander body", "animal"],
        )
        for candidates in PROMPT_CANDIDATES_BY_DATASET.values():
            self.assertEqual(candidates[-1], "animal")

    def test_sample_rows_by_dataset_respects_per_split_cap(self) -> None:
        df = pd.DataFrame(
            {
                "image_id": [str(i) for i in range(8)],
                "identity": ["", "", "", "", "", "", "", ""],
                "path": [f"img_{i}.jpg" for i in range(8)],
                "date": [""] * 8,
                "orientation": [""] * 8,
                "species": ["salamander"] * 8,
                "split": ["train"] * 5 + ["test"] * 3,
                "dataset": ["SalamanderID2025"] * 8,
            }
        )
        sampled = sample_rows_by_dataset(df, sample_seed=0, samples_per_split=2, datasets=["SalamanderID2025"])
        counts = sampled.groupby("split").size().to_dict()
        self.assertEqual(counts["train"], 2)
        self.assertEqual(counts["test"], 2)

    def test_mask_bbox_and_crop_to_union_mask(self) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 3:8] = 1
        bbox = mask_bbox(mask)
        self.assertEqual(bbox, (3, 2, 7, 5))
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[2:6, 3:8, :] = 255
        cropped = crop_to_union_mask(
            image=Image.fromarray(image),
            masks=np.expand_dims(mask, axis=0),
        )
        self.assertEqual(cropped.size, (5, 4))


if __name__ == "__main__":
    unittest.main()
