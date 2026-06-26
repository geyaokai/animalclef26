from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.animalclef_analysis.salamander_trusted_batch import compile_salamander_trusted_batch


class SalamanderTrustedBatchTest(unittest.TestCase):
    def test_compile_exports_clean_tables_and_excludes_conflicted_components(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pair_judgments_path = root / "pair_judgments.json"
            metadata_path = root / "metadata.csv"
            output_dir = root / "trusted_out"

            pair_judgments_path.write_text(
                json.dumps(
                    {
                        "session_name": "unit",
                        "pair_judgments": [
                            {
                                "dataset": "SalamanderID2025",
                                "candidate_type": "split",
                                "candidate_key": "1",
                                "image_id": "a",
                                "neighbor_image_id": "b",
                                "label": "yes",
                            },
                            {
                                "dataset": "SalamanderID2025",
                                "candidate_type": "split",
                                "candidate_key": "1",
                                "image_id": "b",
                                "neighbor_image_id": "c",
                                "label": "yes",
                            },
                            {
                                "dataset": "SalamanderID2025",
                                "candidate_type": "split",
                                "candidate_key": "1",
                                "image_id": "a",
                                "neighbor_image_id": "c",
                                "label": "no",
                            },
                            {
                                "dataset": "SalamanderID2025",
                                "candidate_type": "split",
                                "candidate_key": "2",
                                "image_id": "d",
                                "neighbor_image_id": "e",
                                "label": "yes",
                            },
                            {
                                "dataset": "SalamanderID2025",
                                "candidate_type": "split",
                                "candidate_key": "3",
                                "image_id": "d",
                                "neighbor_image_id": "x",
                                "label": "no",
                            },
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {"image_id": image_id, "dataset": "SalamanderID2025", "split": "test", "path": f"images/{image_id}.jpg"}
                    for image_id in ["a", "b", "c", "d", "e", "x"]
                ]
            ).to_csv(metadata_path, index=False)

            artifacts = compile_salamander_trusted_batch(
                repo_root=root,
                pair_judgments_path=pair_judgments_path,
                output_dir=output_dir,
                metadata_path=metadata_path,
            )

            membership_df = pd.read_csv(artifacts.trusted_membership_path)
            clean_membership_df = pd.read_csv(artifacts.clean_trusted_membership_path)
            clean_pairs_df = pd.read_csv(artifacts.clean_trusted_pairs_path)
            conflict_df = pd.read_csv(artifacts.conflict_pairs_path)
            summary = json.loads((output_dir / "reports" / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(set(membership_df["image_id"].astype(str)), {"a", "b", "c", "d", "e"})
            self.assertEqual(set(clean_membership_df["image_id"].astype(str)), {"d", "e"})
            self.assertEqual(set(clean_pairs_df["pair_key"].astype(str)), {"d|e"})
            self.assertEqual(set(conflict_df["pair_key"].astype(str)), {"a|c"})
            self.assertEqual(int(summary["clean_trusted_components"]), 1)
            self.assertEqual(int(summary["clean_trusted_members"]), 2)
            self.assertTrue(artifacts.review_html_path.exists())


if __name__ == "__main__":
    unittest.main()
