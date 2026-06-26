from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.manual_cluster_overlay import (
    ACTION_ATTACH_TO_ANCHOR,
    ACTION_SPLIT_TO_SINGLETONS,
    ManualOverlayOperation,
    ManualOverlaySpec,
    apply_manual_cluster_overlay,
)


class ManualClusterOverlayTest(unittest.TestCase):
    def _build_pred_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "image_id": "5401",
                    "dataset": "SalamanderID2025",
                    "path": "images/salamander_5401.jpg",
                    "pred_cluster_id": 52,
                    "cluster_label": "cluster_SalamanderID2025_52",
                    "route_name": "salamander_base",
                    "chosen_threshold": 0.25,
                    "embedding_dim": 512,
                    "rerank_enabled": False,
                    "local_weight": 0.0,
                },
                {
                    "image_id": "5903",
                    "dataset": "SalamanderID2025",
                    "path": "images/salamander_5903.jpg",
                    "pred_cluster_id": 52,
                    "cluster_label": "cluster_SalamanderID2025_52",
                    "route_name": "salamander_base",
                    "chosen_threshold": 0.25,
                    "embedding_dim": 512,
                    "rerank_enabled": False,
                    "local_weight": 0.0,
                },
                {
                    "image_id": "7001",
                    "dataset": "TexasHornedLizards",
                    "path": "images/texas_7001.jpg",
                    "pred_cluster_id": 10,
                    "cluster_label": "cluster_TexasHornedLizards_10",
                    "route_name": "texas_base",
                    "chosen_threshold": 0.44,
                    "embedding_dim": 512,
                    "rerank_enabled": False,
                    "local_weight": 0.0,
                },
                {
                    "image_id": "7002",
                    "dataset": "TexasHornedLizards",
                    "path": "images/texas_7002.jpg",
                    "pred_cluster_id": 11,
                    "cluster_label": "cluster_TexasHornedLizards_11",
                    "route_name": "texas_base",
                    "chosen_threshold": 0.44,
                    "embedding_dim": 512,
                    "rerank_enabled": False,
                    "local_weight": 0.0,
                },
                {
                    "image_id": "7003",
                    "dataset": "TexasHornedLizards",
                    "path": "images/texas_7003.jpg",
                    "pred_cluster_id": 11,
                    "cluster_label": "cluster_TexasHornedLizards_11",
                    "route_name": "texas_base",
                    "chosen_threshold": 0.44,
                    "embedding_dim": 512,
                    "rerank_enabled": False,
                    "local_weight": 0.0,
                },
            ]
        )

    def test_split_to_singletons_keeps_anchor_and_creates_new_cluster(self) -> None:
        pred_df = self._build_pred_df()
        spec = ManualOverlaySpec(
            rule_name="manual_cluster_overlay_v1",
            operations=(
                ManualOverlayOperation(
                    operation_id="split52",
                    dataset="SalamanderID2025",
                    action=ACTION_SPLIT_TO_SINGLETONS,
                    anchor_image_id="5401",
                    source_cluster_ids=(52,),
                    member_image_ids=tuple(),
                    exclude_image_ids=tuple(),
                    note="A-tier split",
                ),
            ),
            raw_payload={},
        )

        result_df, changed_df, operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)

        anchor_row = result_df[result_df["image_id"].astype(str).eq("5401")].iloc[0]
        moved_row = result_df[result_df["image_id"].astype(str).eq("5903")].iloc[0]
        self.assertEqual(int(anchor_row["pred_cluster_id"]), 52)
        self.assertEqual(int(moved_row["pred_cluster_id"]), 53)
        self.assertEqual(len(changed_df), 2)
        self.assertEqual(int(operation_df.iloc[0]["changed_count"]), 1)

    def test_attach_to_anchor_merges_other_cluster_members(self) -> None:
        pred_df = self._build_pred_df()
        spec = ManualOverlaySpec(
            rule_name="manual_cluster_overlay_v1",
            operations=(
                ManualOverlayOperation(
                    operation_id="merge11into10",
                    dataset="TexasHornedLizards",
                    action=ACTION_ATTACH_TO_ANCHOR,
                    anchor_image_id="7001",
                    source_cluster_ids=(11,),
                    member_image_ids=tuple(),
                    exclude_image_ids=tuple(),
                    note="manual merge",
                ),
            ),
            raw_payload={},
        )

        result_df, changed_df, operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)

        self.assertTrue((result_df[result_df["dataset"].astype(str).eq("TexasHornedLizards")]["pred_cluster_id"].astype(int) == [10, 10, 10]).all())
        self.assertEqual(len(changed_df), 2)
        self.assertEqual(int(operation_df.iloc[0]["changed_count"]), 2)

    def test_explicit_member_singletonize_only_moves_selected_images(self) -> None:
        pred_df = self._build_pred_df()
        pred_df = pd.concat(
            [
                pred_df,
                pd.DataFrame(
                    [
                        {
                            "image_id": "5904",
                            "dataset": "SalamanderID2025",
                            "path": "images/salamander_5904.jpg",
                            "pred_cluster_id": 52,
                            "cluster_label": "cluster_SalamanderID2025_52",
                            "route_name": "salamander_base",
                            "chosen_threshold": 0.25,
                            "embedding_dim": 512,
                            "rerank_enabled": False,
                            "local_weight": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        spec = ManualOverlaySpec(
            rule_name="manual_cluster_overlay_v1",
            operations=(
                ManualOverlayOperation(
                    operation_id="singletonize5904",
                    dataset="SalamanderID2025",
                    action=ACTION_SPLIT_TO_SINGLETONS,
                    anchor_image_id=None,
                    source_cluster_ids=tuple(),
                    member_image_ids=("5904",),
                    exclude_image_ids=tuple(),
                    note="explicit outlier",
                ),
            ),
            raw_payload={},
        )

        result_df, changed_df, operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)

        moved_row = result_df[result_df["image_id"].astype(str).eq("5904")].iloc[0]
        keep_rows = result_df[result_df["image_id"].astype(str).isin(["5401", "5903"])]
        self.assertEqual(int(moved_row["pred_cluster_id"]), 53)
        self.assertEqual(sorted(keep_rows["pred_cluster_id"].astype(int).tolist()), [52, 52])
        self.assertEqual(len(changed_df), 1)
        self.assertEqual(int(operation_df.iloc[0]["changed_count"]), 1)


if __name__ == "__main__":
    unittest.main()
