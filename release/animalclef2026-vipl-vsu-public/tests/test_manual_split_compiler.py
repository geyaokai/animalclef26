from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.manual_cluster_overlay import (
    ManualOverlayOperation,
    ManualOverlaySpec,
    apply_manual_cluster_overlay,
)
from src.animalclef_analysis.manual_split_compiler import compile_split_judgments_to_overlay


def _spec_from_operations(operations: list[dict]) -> ManualOverlaySpec:
    return ManualOverlaySpec(
        rule_name="manual_split_compiled_v1",
        operations=tuple(
            ManualOverlayOperation(
                operation_id=str(item["operation_id"]),
                dataset=str(item["dataset"]),
                action=str(item["action"]),
                anchor_image_id=str(item["anchor_image_id"]) if item.get("anchor_image_id") is not None else None,
                source_cluster_ids=tuple(int(value) for value in item.get("source_cluster_ids", [])),
                member_image_ids=tuple(str(value) for value in item.get("member_image_ids", [])),
                exclude_image_ids=tuple(str(value) for value in item.get("exclude_image_ids", [])),
                note=str(item.get("note", "")),
            )
            for item in operations
        ),
        raw_payload={"operations": operations},
    )


class ManualSplitCompilerTest(unittest.TestCase):
    def test_compile_split_judgments_partial_split_then_regroup(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "a1", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "a2", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "a3", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "b1", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "b2", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
            ]
        )
        judgments = [
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|a2", "image_id": "a1", "neighbor_image_id": "a2", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|a3", "image_id": "a1", "neighbor_image_id": "a3", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|a3", "image_id": "a2", "neighbor_image_id": "a3", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "b1|b2", "image_id": "b1", "neighbor_image_id": "b2", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|b1", "image_id": "a1", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|b1", "image_id": "a2", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a3|b1", "image_id": "a3", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|b2", "image_id": "a1", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|b2", "image_id": "a2", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a3|b2", "image_id": "a3", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.1, "ambiguity_score": 0.9},
        ]

        operations, candidate_df, image_df, component_df = compile_split_judgments_to_overlay(
            pred_df,
            judgments,
            min_no_degree=2,
            min_net_no_margin=1,
        )

        self.assertEqual(len(operations), 2)
        self.assertEqual(candidate_df.iloc[0]["compile_status"], "compiled_partial_split")
        self.assertEqual(int(candidate_df.iloc[0]["selected_split_images"]), 2)
        self.assertEqual(int(candidate_df.iloc[0]["attach_group_count"]), 1)
        self.assertEqual(sorted(image_df[image_df["selected_for_split"].astype(bool)]["image_id"].astype(str).tolist()), ["b1", "b2"])
        self.assertTrue(bool(component_df.iloc[0]["attach_after_split"]))

        spec = _spec_from_operations(operations)
        result_df, changed_df, _operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)

        a_cluster_ids = sorted(result_df[result_df["image_id"].astype(str).isin(["a1", "a2", "a3"])]["pred_cluster_id"].astype(int).unique().tolist())
        b_cluster_ids = sorted(result_df[result_df["image_id"].astype(str).isin(["b1", "b2"])]["pred_cluster_id"].astype(int).unique().tolist())
        self.assertEqual(a_cluster_ids, [52])
        self.assertEqual(len(b_cluster_ids), 1)
        self.assertNotEqual(b_cluster_ids[0], 52)
        self.assertEqual(len(changed_df), 3)

    def test_compile_split_judgments_skips_yes_only_candidate(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "x1", "dataset": "SalamanderID2025", "pred_cluster_id": 80, "cluster_label": "cluster_SalamanderID2025_80"},
                {"image_id": "x2", "dataset": "SalamanderID2025", "pred_cluster_id": 80, "cluster_label": "cluster_SalamanderID2025_80"},
            ]
        )
        judgments = [
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "split",
                "candidate_key": "80",
                "pair_key": "x1|x2",
                "image_id": "x1",
                "neighbor_image_id": "x2",
                "label": "yes",
                "xgb_same_identity_prob": 0.98,
                "ambiguity_score": 0.4,
            }
        ]

        operations, candidate_df, image_df, component_df = compile_split_judgments_to_overlay(
            pred_df,
            judgments,
            min_no_degree=2,
            min_net_no_margin=1,
        )

        self.assertEqual(operations, [])
        self.assertEqual(candidate_df.iloc[0]["compile_status"], "skip_no_selected_images")
        self.assertFalse(image_df["selected_for_split"].astype(bool).any())
        self.assertTrue(component_df.empty)


if __name__ == "__main__":
    unittest.main()
