from __future__ import annotations

import unittest

import pandas as pd

from src.animalclef_analysis.manual_cluster_overlay import (
    ManualOverlayOperation,
    ManualOverlaySpec,
    apply_manual_cluster_overlay,
)
from src.animalclef_analysis.manual_constraint_graph_compiler import compile_constraint_graph_to_overlay


def _spec_from_operations(operations: list[dict]) -> ManualOverlaySpec:
    return ManualOverlaySpec(
        rule_name="manual_constraint_graph_v1",
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


class ManualConstraintGraphCompilerTest(unittest.TestCase):
    def test_compile_constraint_graph_splits_cross_group_and_regroups_component(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "a1", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "a2", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "a3", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "b1", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
                {"image_id": "b2", "dataset": "SalamanderID2025", "pred_cluster_id": 52, "cluster_label": "cluster_SalamanderID2025_52"},
            ]
        )
        pair_df = pd.DataFrame(
            [
                {"image_id": "a1", "neighbor_image_id": "a2", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.1, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a1", "neighbor_image_id": "a3", "xgb_same_identity_prob": 0.98, "ambiguity_score": 0.1, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a2", "neighbor_image_id": "a3", "xgb_same_identity_prob": 0.97, "ambiguity_score": 0.1, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "b1", "neighbor_image_id": "b2", "xgb_same_identity_prob": 0.96, "ambiguity_score": 0.1, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a1", "neighbor_image_id": "b1", "xgb_same_identity_prob": 0.40, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a2", "neighbor_image_id": "b1", "xgb_same_identity_prob": 0.41, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a3", "neighbor_image_id": "b1", "xgb_same_identity_prob": 0.42, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a1", "neighbor_image_id": "b2", "xgb_same_identity_prob": 0.39, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a2", "neighbor_image_id": "b2", "xgb_same_identity_prob": 0.38, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
                {"image_id": "a3", "neighbor_image_id": "b2", "xgb_same_identity_prob": 0.37, "ambiguity_score": 0.9, "base_cluster_left": 52, "base_cluster_right": 52},
            ]
        )
        judgments = [
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|a2", "image_id": "a1", "neighbor_image_id": "a2", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|a3", "image_id": "a1", "neighbor_image_id": "a3", "label": "yes", "xgb_same_identity_prob": 0.98, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|a3", "image_id": "a2", "neighbor_image_id": "a3", "label": "yes", "xgb_same_identity_prob": 0.97, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "b1|b2", "image_id": "b1", "neighbor_image_id": "b2", "label": "yes", "xgb_same_identity_prob": 0.96, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|b1", "image_id": "a1", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.40, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|b1", "image_id": "a2", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.41, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a3|b1", "image_id": "a3", "neighbor_image_id": "b1", "label": "no", "xgb_same_identity_prob": 0.42, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a1|b2", "image_id": "a1", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.39, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a2|b2", "image_id": "a2", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.38, "ambiguity_score": 0.9},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "52", "pair_key": "a3|b2", "image_id": "a3", "neighbor_image_id": "b2", "label": "no", "xgb_same_identity_prob": 0.37, "ambiguity_score": 0.9},
        ]

        operations, candidate_df, component_df, edge_df = compile_constraint_graph_to_overlay(
            pred_df,
            pair_df,
            judgments,
            graph_threshold=0.25,
        )

        self.assertEqual(len(operations), 2)
        self.assertEqual(candidate_df.iloc[0]["compile_status"], "compiled_constrained_split")
        self.assertEqual(int(candidate_df.iloc[0]["component_count"]), 2)
        self.assertEqual(int(candidate_df.iloc[0]["moved_images"]), 2)
        self.assertEqual(int(candidate_df.iloc[0]["blocked_edges"]), 0)
        self.assertTrue(component_df["is_anchor_component"].astype(bool).any())
        self.assertIn("union", edge_df["decision"].astype(str).tolist())
        self.assertEqual(int(edge_df["decision"].astype(str).eq("skip_manual_no").sum()), 6)

        spec = _spec_from_operations(operations)
        result_df, changed_df, _operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)
        a_cluster_ids = sorted(result_df[result_df["image_id"].astype(str).isin(["a1", "a2", "a3"])]["pred_cluster_id"].astype(int).unique().tolist())
        b_cluster_ids = sorted(result_df[result_df["image_id"].astype(str).isin(["b1", "b2"])]["pred_cluster_id"].astype(int).unique().tolist())
        self.assertEqual(a_cluster_ids, [52])
        self.assertEqual(len(b_cluster_ids), 1)
        self.assertNotEqual(b_cluster_ids[0], 52)
        self.assertEqual(len(changed_df), 3)

    def test_compile_constraint_graph_blocks_conflicting_yes_edge(self) -> None:
        pred_df = pd.DataFrame(
            [
                {"image_id": "x1", "dataset": "SalamanderID2025", "pred_cluster_id": 80, "cluster_label": "cluster_SalamanderID2025_80"},
                {"image_id": "x2", "dataset": "SalamanderID2025", "pred_cluster_id": 80, "cluster_label": "cluster_SalamanderID2025_80"},
                {"image_id": "x3", "dataset": "SalamanderID2025", "pred_cluster_id": 80, "cluster_label": "cluster_SalamanderID2025_80"},
            ]
        )
        pair_df = pd.DataFrame(
            [
                {"image_id": "x1", "neighbor_image_id": "x2", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.1, "base_cluster_left": 80, "base_cluster_right": 80},
                {"image_id": "x1", "neighbor_image_id": "x3", "xgb_same_identity_prob": 0.98, "ambiguity_score": 0.1, "base_cluster_left": 80, "base_cluster_right": 80},
                {"image_id": "x2", "neighbor_image_id": "x3", "xgb_same_identity_prob": 0.97, "ambiguity_score": 0.1, "base_cluster_left": 80, "base_cluster_right": 80},
            ]
        )
        judgments = [
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "80", "pair_key": "x1|x2", "image_id": "x1", "neighbor_image_id": "x2", "label": "yes", "xgb_same_identity_prob": 0.99, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "80", "pair_key": "x1|x3", "image_id": "x1", "neighbor_image_id": "x3", "label": "yes", "xgb_same_identity_prob": 0.98, "ambiguity_score": 0.8},
            {"dataset": "SalamanderID2025", "candidate_type": "split", "candidate_key": "80", "pair_key": "x2|x3", "image_id": "x2", "neighbor_image_id": "x3", "label": "no", "xgb_same_identity_prob": 0.97, "ambiguity_score": 0.9},
        ]

        operations, candidate_df, component_df, edge_df = compile_constraint_graph_to_overlay(
            pred_df,
            pair_df,
            judgments,
            graph_threshold=0.25,
        )

        self.assertEqual(candidate_df.iloc[0]["compile_status"], "compiled_constrained_split")
        self.assertEqual(int(candidate_df.iloc[0]["blocked_yes_edges"]), 1)
        self.assertEqual(int(candidate_df.iloc[0]["component_count"]), 2)
        self.assertIn("block_even_yes_due_cannot_link", edge_df["decision"].astype(str).tolist())
        moved_component = component_df[~component_df["is_anchor_component"].astype(bool)].iloc[0]
        self.assertEqual(int(moved_component["component_size"]), 1)
        self.assertEqual(len(operations), 1)


if __name__ == "__main__":
    unittest.main()
