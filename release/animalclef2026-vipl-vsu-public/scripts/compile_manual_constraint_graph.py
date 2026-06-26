#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_SUBMISSION_DIR = Path("artifacts/submissions/manual_split_compiled_v1")
DEFAULT_PAIR_GRAPH_PATH = Path("artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1/tables/test_pair_disagreement_v1.csv")
DEFAULT_PAIR_JUDGMENTS = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/manual_constraint_graph_v1")
DEFAULT_RULE_NAME = "manual_constraint_graph_v1"


def _escape_markdown_cell(value: object) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_table_or_note(frame: pd.DataFrame, *, note: str, limit: int = 20) -> str:
    if frame.empty:
        return note
    preview = frame.head(int(limit)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_escape_markdown_cell(row[column]) for column in columns) + " |"
        for _, row in preview.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.manual_constraint_graph_compiler import compile_constraint_graph_file
    from animalclef_analysis.manual_review_workbench import build_overlay_command, export_operations_spec

    parser = argparse.ArgumentParser(description="Compile manual pair cannot-link/must-link judgments into constrained graph overlay operations.")
    parser.add_argument("--base-submission-dir", type=Path, default=DEFAULT_BASE_SUBMISSION_DIR)
    parser.add_argument("--pair-graph-path", type=Path, default=DEFAULT_PAIR_GRAPH_PATH)
    parser.add_argument("--pair-judgments", type=Path, default=DEFAULT_PAIR_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rule-name", type=str, default=DEFAULT_RULE_NAME)
    parser.add_argument("--submission-output-dir", type=Path, default=Path("artifacts/submissions/manual_constraint_graph_v1"))
    parser.add_argument("--submission-description", type=str, default="Manual constraint graph overlay v1")
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--candidate-keys", nargs="+", type=str, default=None)
    parser.add_argument("--graph-threshold", type=float, default=0.25)
    parser.add_argument("--min-judged-pairs", type=int, default=1)
    parser.add_argument("--min-no-pairs", type=int, default=1)
    args = parser.parse_args()

    base_submission_dir = (repo_root / args.base_submission_dir).resolve() if not args.base_submission_dir.is_absolute() else args.base_submission_dir.resolve()
    pair_graph_path = (repo_root / args.pair_graph_path).resolve() if not args.pair_graph_path.is_absolute() else args.pair_graph_path.resolve()
    pair_judgments_path = (repo_root / args.pair_judgments).resolve() if not args.pair_judgments.is_absolute() else args.pair_judgments.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    submission_output_dir = (repo_root / args.submission_output_dir).resolve() if not args.submission_output_dir.is_absolute() else args.submission_output_dir.resolve()

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    (
        session_name,
        judgments,
        operations,
        candidate_summary_df,
        component_summary_df,
        edge_summary_df,
    ) = compile_constraint_graph_file(
        base_predictions_path=base_submission_dir / "tables" / "test_predictions_v1.csv",
        pair_graph_path=pair_graph_path,
        pair_judgments_path=pair_judgments_path,
        datasets=args.datasets,
        candidate_keys=args.candidate_keys,
        graph_threshold=float(args.graph_threshold),
        min_judged_pairs=int(args.min_judged_pairs),
        min_no_pairs=int(args.min_no_pairs),
    )

    candidate_summary_path = tables_dir / "candidate_summary_v1.csv"
    component_summary_path = tables_dir / "component_summary_v1.csv"
    edge_summary_path = tables_dir / "edge_summary_v1.csv"
    operation_summary_path = tables_dir / "compiled_operations_v1.csv"
    spec_path = output_dir / "compiled_overlay_spec.json"

    candidate_summary_df.to_csv(candidate_summary_path, index=False)
    component_summary_df.to_csv(component_summary_path, index=False)
    edge_summary_df.to_csv(edge_summary_path, index=False)
    pd.DataFrame(operations).to_csv(operation_summary_path, index=False)
    export_operations_spec(
        rule_name=str(args.rule_name),
        operations=operations,
        output_path=spec_path,
    )

    build_command = build_overlay_command(
        repo_root=repo_root,
        base_submission_dir=base_submission_dir,
        spec_path=spec_path,
        output_dir=submission_output_dir,
        submission_description=str(args.submission_description),
    )

    summary_payload = {
        "session_name": str(session_name),
        "pair_judgments_path": str(pair_judgments_path),
        "pair_graph_path": str(pair_graph_path),
        "base_submission_dir": str(base_submission_dir),
        "rule_name": str(args.rule_name),
        "operations": operations,
        "candidate_summary_path": str(candidate_summary_path),
        "component_summary_path": str(component_summary_path),
        "edge_summary_path": str(edge_summary_path),
        "compiled_overlay_spec_path": str(spec_path),
        "suggested_build_command": build_command,
        "filters": {
            "datasets": args.datasets,
            "candidate_keys": args.candidate_keys,
            "graph_threshold": float(args.graph_threshold),
            "min_judged_pairs": int(args.min_judged_pairs),
            "min_no_pairs": int(args.min_no_pairs),
        },
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    total_split_judgments = sum(1 for item in judgments if str(item.get("candidate_type", "")) == "split")
    lines = [
        "# Manual Constraint Graph v1",
        "",
        f"- Session name: `{session_name}`",
        f"- Pair judgments path: `{pair_judgments_path}`",
        f"- Pair graph path: `{pair_graph_path}`",
        f"- Base submission dir: `{base_submission_dir}`",
        f"- Total loaded split judgments: `{total_split_judgments}`",
        f"- Candidate filter: `{args.candidate_keys or []}`",
        f"- Dataset filter: `{args.datasets or []}`",
        f"- Graph threshold: `{float(args.graph_threshold):.2f}`",
        f"- Review gate: `judged_pairs >= {int(args.min_judged_pairs)}` and `no_pairs >= {int(args.min_no_pairs)}`",
        f"- Compiled overlay spec: `{spec_path}`",
        "",
        "## What This Compiler Does",
        "",
        "- It treats manual `no` as pair-level `cannot-link`, not just as image-level split votes.",
        "- It rebuilds each reviewed split subgraph on `xgb_same_identity_prob` edges above threshold, while forbidding any merge that would violate a manual `cannot-link` pair.",
        "- It then converts the resulting partition back into the existing `split_to_singletons + attach_to_anchor` overlay operations.",
        "",
        "## Candidate Summary",
        "",
        _format_table_or_note(candidate_summary_df, note="_No candidate produced constrained graph operations under the current filters._", limit=60),
        "",
        "## Component Summary",
        "",
        _format_table_or_note(component_summary_df, note="_No constrained graph components were exported._", limit=80),
        "",
        "## Edge Decisions",
        "",
        _format_table_or_note(
            edge_summary_df[
                [
                    column
                    for column in [
                        "dataset",
                        "base_cluster_id",
                        "image_id",
                        "neighbor_image_id",
                        "score",
                        "manual_label",
                        "considered_for_merge",
                        "processed_order",
                        "decision",
                        "blocking_pair",
                    ]
                    if column in edge_summary_df.columns
                ]
            ]
            if not edge_summary_df.empty
            else edge_summary_df,
            note="_No edge-level decisions were recorded._",
            limit=120,
        ),
        "",
        "## Suggested Build Command",
        "",
        "```bash",
        build_command,
        "```",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[manual_constraint_graph] spec: {spec_path}")
    print(f"[manual_constraint_graph] candidate_summary: {candidate_summary_path}")
    print(f"[manual_constraint_graph] summary: {reports_dir / 'summary.md'}")
    print(f"[manual_constraint_graph] build_command: {build_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
