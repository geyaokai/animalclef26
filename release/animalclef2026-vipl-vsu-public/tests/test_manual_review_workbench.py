from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.animalclef_analysis.manual_review_workbench import (
    PAIR_LABEL_NO,
    PAIR_LABEL_YES,
    add_attach_operation,
    add_split_operation,
    build_judgment_preview_json,
    build_candidate_task_table,
    export_pair_judgments,
    export_operations_spec,
    find_next_unjudged_pair_index,
    get_pair_row,
    judgments_to_dataframe,
    list_candidate_choices,
    load_pair_judgments,
    load_review_bundle,
    remove_operation_at,
    render_candidate_view,
    summarize_pair_judgments,
    upsert_pair_judgment,
)


class ManualReviewWorkbenchTest(unittest.TestCase):
    def test_load_bundle_and_render_split_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts" / "submissions" / "base" / "tables").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "analysis" / "probe" / "tables").mkdir(parents=True, exist_ok=True)

            pred_df = pd.DataFrame(
                [
                    {
                        "image_id": "5401",
                        "dataset": "SalamanderID2025",
                        "path": "images/a.jpg",
                        "pred_cluster_id": 52,
                        "cluster_label": "cluster_SalamanderID2025_52",
                    },
                    {
                        "image_id": "5903",
                        "dataset": "SalamanderID2025",
                        "path": "images/b.jpg",
                        "pred_cluster_id": 52,
                        "cluster_label": "cluster_SalamanderID2025_52",
                    },
                ]
            )
            pred_df.to_csv(root / "artifacts" / "submissions" / "base" / "tables" / "test_predictions_v1.csv", index=False)

            split_df = pd.DataFrame(
                [
                    {
                        "base_cluster_id": 52,
                        "base_cluster_size": 2,
                        "ambiguous_image_count": 2,
                        "ambiguous_pair_count": 1,
                        "max_split_votes": 3,
                        "mean_pair_probability": 0.25,
                        "max_pair_probability": 0.25,
                        "mean_ambiguity_score": 0.89,
                        "max_ambiguity_score": 0.89,
                        "mean_border_score": 0.99,
                        "max_conflict_ratio": 1.0,
                        "conflict_methods": "dbscan|finch",
                        "component_ids": "37",
                        "image_indices": "1|2",
                        "image_ids": "5401|5903",
                    }
                ]
            )
            split_df.to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_split_candidates_v1.csv", index=False)
            pd.DataFrame(columns=["cluster_pair_key"]).to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_merge_candidates_v1.csv", index=False)
            pair_df = pd.DataFrame(
                [
                    {
                        "image_id": "5401",
                        "neighbor_image_id": "5903",
                        "xgb_same_identity_prob": 0.250827,
                        "merge_votes": 0,
                        "split_votes": 3,
                        "ambiguity_score": 0.896899,
                        "vote_direction": "split",
                        "base_cluster_left": 52,
                        "base_cluster_right": 52,
                        "conflict_methods": "dbscan|finch",
                        "component_id": 37,
                    }
                ]
            )
            pair_df.to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_pair_disagreement_v1.csv", index=False)

            bundle = load_review_bundle(
                repo_root=root,
                base_predictions_path="artifacts/submissions/base",
                probe_dir="artifacts/analysis/probe",
            )

            choices = list_candidate_choices(bundle, "split")
            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0][1], "52")

            payload = render_candidate_view(bundle, "split", "52")
            self.assertEqual(payload["dataset"], "SalamanderID2025")
            self.assertEqual(len(payload["pair_df"]), 1)
            self.assertIn("split cluster 52", payload["summary_markdown"])

    def test_load_bundle_and_render_yes_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts" / "submissions" / "base" / "tables").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "analysis" / "probe" / "tables").mkdir(parents=True, exist_ok=True)

            pred_df = pd.DataFrame(
                [
                    {
                        "image_id": "15306",
                        "dataset": "TexasHornedLizards",
                        "path": "images/a.jpg",
                        "pred_cluster_id": 141,
                        "cluster_label": "cluster_TexasHornedLizards_141",
                    },
                    {
                        "image_id": "15414",
                        "dataset": "TexasHornedLizards",
                        "path": "images/b.jpg",
                        "pred_cluster_id": 141,
                        "cluster_label": "cluster_TexasHornedLizards_141",
                    },
                    {
                        "image_id": "15437",
                        "dataset": "TexasHornedLizards",
                        "path": "images/c.jpg",
                        "pred_cluster_id": 141,
                        "cluster_label": "cluster_TexasHornedLizards_141",
                    },
                ]
            )
            pred_df.to_csv(root / "artifacts" / "submissions" / "base" / "tables" / "test_predictions_v1.csv", index=False)

            pd.DataFrame(columns=["base_cluster_id"]).to_csv(
                root / "artifacts" / "analysis" / "probe" / "tables" / "test_split_candidates_v1.csv",
                index=False,
            )
            pd.DataFrame(columns=["cluster_pair_key"]).to_csv(
                root / "artifacts" / "analysis" / "probe" / "tables" / "test_merge_candidates_v1.csv",
                index=False,
            )
            pd.DataFrame(columns=["image_id", "neighbor_image_id"]).to_csv(
                root / "artifacts" / "analysis" / "probe" / "tables" / "test_pair_disagreement_v1.csv",
                index=False,
            )
            yes_candidate_df = pd.DataFrame(
                [
                    {
                        "candidate_key": "split:141",
                        "dataset": "TexasHornedLizards",
                        "source_candidate_type": "split",
                        "source_candidate_key": "141",
                        "candidate_kind": "split_component_extension",
                        "candidate_preview": "split 141 | top=split_component_extension | imgs=3 | pairs=2",
                        "priority_score": 5.25,
                        "pair_count": 2,
                        "unique_image_count": 3,
                        "triangle_pair_count": 0,
                        "extension_pair_count": 1,
                        "local_supported_pair_count": 2,
                        "image_ids": "15306|15414|15437",
                        "top_pair_keys": "15306|15414|15306|15437",
                    }
                ]
            )
            yes_candidate_df.to_csv(
                root / "artifacts" / "analysis" / "probe" / "tables" / "test_yes_candidates_v1.csv",
                index=False,
            )
            yes_pair_df = pd.DataFrame(
                [
                    {
                        "candidate_type": "yes",
                        "candidate_key": "split:141",
                        "image_id": "15306",
                        "neighbor_image_id": "15414",
                        "xgb_same_identity_prob": 0.56,
                        "ambiguity_score": 0.88,
                        "vote_direction": "split",
                        "merge_votes": 0,
                        "split_votes": 3,
                        "base_cluster_left": 141,
                        "base_cluster_right": 141,
                        "yes_priority_score": 2.7,
                        "yes_candidate_reason": "split_component_extension",
                        "source_candidate_type": "split",
                        "source_candidate_key": "141",
                        "existing_yes_component_size": 2,
                    },
                    {
                        "candidate_type": "yes",
                        "candidate_key": "split:141",
                        "image_id": "15306",
                        "neighbor_image_id": "15437",
                        "xgb_same_identity_prob": 0.54,
                        "ambiguity_score": 0.85,
                        "vote_direction": "split",
                        "merge_votes": 0,
                        "split_votes": 3,
                        "base_cluster_left": 141,
                        "base_cluster_right": 141,
                        "yes_priority_score": 2.4,
                        "yes_candidate_reason": "split_local_hard_yes",
                        "source_candidate_type": "split",
                        "source_candidate_key": "141",
                        "existing_yes_component_size": 2,
                    },
                ]
            )
            yes_pair_df.to_csv(
                root / "artifacts" / "analysis" / "probe" / "tables" / "test_yes_pair_candidates_v1.csv",
                index=False,
            )

            bundle = load_review_bundle(
                repo_root=root,
                base_predictions_path="artifacts/submissions/base",
                probe_dir="artifacts/analysis/probe",
            )
            choices = list_candidate_choices(bundle, "yes")
            self.assertEqual(len(choices), 1)
            self.assertEqual(choices[0][1], "split:141")
            payload = render_candidate_view(bundle, "yes", "split:141")
            self.assertEqual(payload["dataset"], "TexasHornedLizards")
            self.assertEqual(len(payload["pair_df"]), 2)
            self.assertIn("source_candidate", payload["summary_markdown"])
            task_df = build_candidate_task_table(bundle, "yes", [])
            self.assertEqual(len(task_df), 1)
            self.assertEqual(task_df.iloc[0]["candidate_type"], "yes")

    def test_add_operations_and_export(self) -> None:
        operations: list[dict] = []
        operations = add_split_operation(
            operations,
            dataset="SalamanderID2025",
            cluster_id=52,
            anchor_image_id="5401",
            member_image_ids=["5401", "5903"],
            note="split review",
        )
        operations = add_attach_operation(
            operations,
            dataset="TexasHornedLizards",
            anchor_image_id="7001",
            member_image_ids=["7002"],
            source_cluster_ids=[],
            note="attach review",
        )
        self.assertEqual(len(operations), 2)
        self.assertEqual(operations[0]["action"], "split_to_singletons")
        self.assertEqual(operations[1]["action"], "attach_to_anchor")

        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = export_operations_spec(
                rule_name="manual_cluster_overlay_v1",
                operations=operations,
                output_path=Path(tmpdir) / "manual_overlay.json",
            )
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["rule_name"], "manual_cluster_overlay_v1")
            self.assertEqual(len(payload["operations"]), 2)

    def test_remove_operation_at_drops_selected_row(self) -> None:
        operations: list[dict] = []
        operations = add_split_operation(
            operations,
            dataset="SalamanderID2025",
            cluster_id=52,
            anchor_image_id="5401",
            member_image_ids=["5401", "5903"],
            note="split review",
        )
        operations = add_attach_operation(
            operations,
            dataset="TexasHornedLizards",
            anchor_image_id="7001",
            member_image_ids=["7002"],
            source_cluster_ids=[],
            note="attach review",
        )
        operations = remove_operation_at(operations, 0)
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["dataset"], "TexasHornedLizards")

    def test_upsert_pair_judgment_and_summary(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "image_id": "5401",
                    "neighbor_image_id": "5903",
                    "base_cluster_left": 52,
                    "base_cluster_right": 52,
                    "xgb_same_identity_prob": 0.250827,
                    "ambiguity_score": 0.896899,
                }
            ]
        )
        pair_row = get_pair_row(pair_df, 0)
        judgments: list[dict] = []
        judgments = upsert_pair_judgment(
            judgments,
            dataset="SalamanderID2025",
            candidate_type="split",
            candidate_key="52",
            pair_row=pair_row,
            label=PAIR_LABEL_NO,
            note="looks different",
        )
        judgments = upsert_pair_judgment(
            judgments,
            dataset="SalamanderID2025",
            candidate_type="split",
            candidate_key="52",
            pair_row=pair_row,
            label=PAIR_LABEL_YES,
            note="changed mind",
        )
        judgment_df = judgments_to_dataframe(judgments)
        self.assertEqual(len(judgment_df), 1)
        self.assertEqual(judgment_df.iloc[0]["label"], PAIR_LABEL_YES)
        summary_df = summarize_pair_judgments(judgments)
        self.assertEqual(len(summary_df), 1)
        self.assertEqual(int(summary_df.iloc[0]["yes_count"]), 1)

    def test_export_pair_judgments(self) -> None:
        judgments = [
            {
                "judgment_id": "j1",
                "dataset": "SalamanderID2025",
                "candidate_type": "split",
                "candidate_key": "52",
                "pair_key": "5401|5903",
                "image_id": "5401",
                "neighbor_image_id": "5903",
                "base_cluster_left": 52,
                "base_cluster_right": 52,
                "xgb_same_identity_prob": 0.25,
                "ambiguity_score": 0.89,
                "label": "no",
                "note": "looks different",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = export_pair_judgments(
                session_name="manual_pair_review_v1",
                judgments=judgments,
                output_path=Path(tmpdir) / "pair_judgments.json",
            )
            payload = json.loads(export_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["session_name"], "manual_pair_review_v1")
            self.assertEqual(len(payload["pair_judgments"]), 1)
            loaded_session_name, loaded_judgments = load_pair_judgments(export_path)
            self.assertEqual(loaded_session_name, "manual_pair_review_v1")
            self.assertEqual(len(loaded_judgments), 1)
            self.assertEqual(loaded_judgments[0]["label"], "no")

    def test_build_judgment_preview_json_truncates_large_payload(self) -> None:
        judgments = [
            {
                "judgment_id": f"j{i}",
                "dataset": "TexasHornedLizards",
                "candidate_type": "yes",
                "candidate_key": "split:141",
                "pair_key": f"{1000 + i}|{2000 + i}",
                "image_id": str(1000 + i),
                "neighbor_image_id": str(2000 + i),
                "base_cluster_left": 141,
                "base_cluster_right": 141,
                "xgb_same_identity_prob": 0.8,
                "ambiguity_score": 0.2,
                "label": "yes",
                "note": "",
            }
            for i in range(120)
        ]
        preview = json.loads(build_judgment_preview_json("manual_pair_review_v1", judgments, max_preview_items=25))
        self.assertTrue(preview["preview_truncated"])
        self.assertEqual(preview["total_judgments"], 120)
        self.assertEqual(preview["preview_count"], 25)
        self.assertEqual(preview["omitted_count"], 95)
        self.assertEqual(len(preview["pair_judgments_tail"]), 25)
        self.assertEqual(preview["pair_judgments_tail"][0]["judgment_id"], "j95")
        self.assertEqual(preview["pair_judgments_tail"][-1]["judgment_id"], "j119")

    def test_build_candidate_task_table_tracks_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "artifacts" / "submissions" / "base" / "tables").mkdir(parents=True, exist_ok=True)
            (root / "artifacts" / "analysis" / "probe" / "tables").mkdir(parents=True, exist_ok=True)

            pred_df = pd.DataFrame(
                [
                    {
                        "image_id": "5401",
                        "dataset": "SalamanderID2025",
                        "path": "images/a.jpg",
                        "pred_cluster_id": 52,
                        "cluster_label": "cluster_SalamanderID2025_52",
                    },
                    {
                        "image_id": "5903",
                        "dataset": "SalamanderID2025",
                        "path": "images/b.jpg",
                        "pred_cluster_id": 52,
                        "cluster_label": "cluster_SalamanderID2025_52",
                    },
                ]
            )
            pred_df.to_csv(root / "artifacts" / "submissions" / "base" / "tables" / "test_predictions_v1.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "base_cluster_id": 52,
                        "base_cluster_size": 2,
                        "ambiguous_image_count": 2,
                        "ambiguous_pair_count": 1,
                        "max_split_votes": 3,
                        "mean_pair_probability": 0.25,
                        "max_pair_probability": 0.25,
                        "mean_ambiguity_score": 0.89,
                        "max_ambiguity_score": 0.89,
                        "mean_border_score": 0.99,
                        "max_conflict_ratio": 1.0,
                        "conflict_methods": "dbscan|finch",
                        "component_ids": "37",
                        "image_indices": "1|2",
                        "image_ids": "5401|5903",
                    }
                ]
            ).to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_split_candidates_v1.csv", index=False)
            pd.DataFrame(columns=["cluster_pair_key"]).to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_merge_candidates_v1.csv", index=False)
            pair_df = pd.DataFrame(
                [
                    {
                        "image_id": "5401",
                        "neighbor_image_id": "5903",
                        "xgb_same_identity_prob": 0.250827,
                        "merge_votes": 0,
                        "split_votes": 3,
                        "ambiguity_score": 0.896899,
                        "vote_direction": "split",
                        "base_cluster_left": 52,
                        "base_cluster_right": 52,
                        "conflict_methods": "dbscan|finch",
                        "component_id": 37,
                    }
                ]
            )
            pair_df.to_csv(root / "artifacts" / "analysis" / "probe" / "tables" / "test_pair_disagreement_v1.csv", index=False)

            bundle = load_review_bundle(
                repo_root=root,
                base_predictions_path="artifacts/submissions/base",
                probe_dir="artifacts/analysis/probe",
            )
            task_df = build_candidate_task_table(bundle, "split", [])
            self.assertEqual(task_df.iloc[0]["status_code"], "pending")
            self.assertEqual(task_df.iloc[0]["progress"], "0/1")

            pair_row = get_pair_row(pair_df, 0)
            judgments = upsert_pair_judgment(
                [],
                dataset="SalamanderID2025",
                candidate_type="split",
                candidate_key="52",
                pair_row=pair_row,
                label=PAIR_LABEL_NO,
                note="looks different",
            )
            task_df = build_candidate_task_table(bundle, "split", judgments)
            self.assertEqual(task_df.iloc[0]["status_code"], "completed")
            self.assertEqual(task_df.iloc[0]["progress"], "1/1")
            self.assertEqual(int(task_df.iloc[0]["no_count"]), 1)

    def test_find_next_unjudged_pair_index_skips_reviewed_pairs(self) -> None:
        pair_df = pd.DataFrame(
            [
                {
                    "image_id": "1001",
                    "neighbor_image_id": "1002",
                },
                {
                    "image_id": "1002",
                    "neighbor_image_id": "1003",
                },
                {
                    "image_id": "1003",
                    "neighbor_image_id": "1004",
                },
            ]
        )
        judgments = [
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "split",
                "candidate_key": "52",
                "pair_key": "1001|1002",
            }
        ]
        next_idx = find_next_unjudged_pair_index(
            pair_df,
            judgments,
            dataset="SalamanderID2025",
            candidate_type="split",
            candidate_key="52",
            start_after_index=0,
        )
        self.assertEqual(next_idx, 1)

        judgments.append(
            {
                "dataset": "SalamanderID2025",
                "candidate_type": "split",
                "candidate_key": "52",
                "pair_key": "1002|1003",
            }
        )
        next_idx = find_next_unjudged_pair_index(
            pair_df,
            judgments,
            dataset="SalamanderID2025",
            candidate_type="split",
            candidate_key="52",
            start_after_index=1,
        )
        self.assertEqual(next_idx, 2)

    def test_find_next_unjudged_pair_index_yes_mode_only_counts_current_frame_pairs(self) -> None:
        pair_df = pd.DataFrame(
            [
                {"image_id": "15437", "neighbor_image_id": "15479"},
                {"image_id": "15451", "neighbor_image_id": "15479"},
                {"image_id": "15346", "neighbor_image_id": "15479"},
            ]
        )
        judgments = [
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "yes",
                "candidate_key": "seed_expand:comp_000:15322",
                "pair_key": "15322|15451",
            },
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "yes",
                "candidate_key": "seed_expand:comp_000:15322",
                "pair_key": "15322|15414",
            },
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "yes",
                "candidate_key": "seed_expand:comp_000:15322",
                "pair_key": "15322|15437",
            },
            {
                "dataset": "TexasHornedLizards",
                "candidate_type": "yes",
                "candidate_key": "seed_expand:comp_000:15479",
                "pair_key": "15437|15479",
            },
        ]
        next_idx = find_next_unjudged_pair_index(
            pair_df,
            judgments,
            dataset="TexasHornedLizards",
            candidate_type="yes",
            candidate_key="seed_expand:comp_000:15479",
            start_after_index=0,
            treat_dataset_pair_as_judged=True,
        )
        self.assertEqual(next_idx, 1)


if __name__ == "__main__":
    unittest.main()
