#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1")
DEFAULT_PAIR_JUDGMENTS = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/manual_split_compile_v1")
DEFAULT_RULE_NAME = "manual_split_compiled_v1"


def _format_table_or_note(frame: pd.DataFrame, *, note: str, limit: int = 20) -> str:
    def _escape_markdown_cell(value: object) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace("|", "\\|")
        text = text.replace("\n", "<br>")
        return text

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

    from animalclef_analysis.manual_review_workbench import build_overlay_command, export_operations_spec
    from animalclef_analysis.manual_split_compiler import compile_split_judgments_file

    parser = argparse.ArgumentParser(description="Compile manual split pair judgments into conservative overlay operations.")
    parser.add_argument("--base-submission-dir", type=Path, default=DEFAULT_BASE_SUBMISSION_DIR)
    parser.add_argument("--pair-judgments", type=Path, default=DEFAULT_PAIR_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rule-name", type=str, default=DEFAULT_RULE_NAME)
    parser.add_argument("--submission-output-dir", type=Path, default=Path("artifacts/submissions/manual_split_compiled_v1"))
    parser.add_argument("--submission-description", type=str, default="Manual split compiled overlay v1")
    parser.add_argument("--datasets", nargs="+", type=str, default=None)
    parser.add_argument("--candidate-keys", nargs="+", type=str, default=None)
    parser.add_argument("--min-no-degree", type=int, default=2)
    parser.add_argument("--min-net-no-margin", type=int, default=1)
    args = parser.parse_args()

    base_submission_dir = (repo_root / args.base_submission_dir).resolve() if not args.base_submission_dir.is_absolute() else args.base_submission_dir.resolve()
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
        image_summary_df,
        component_summary_df,
    ) = compile_split_judgments_file(
        base_predictions_path=base_submission_dir / "tables" / "test_predictions_v1.csv",
        pair_judgments_path=pair_judgments_path,
        datasets=args.datasets,
        candidate_keys=args.candidate_keys,
        min_no_degree=int(args.min_no_degree),
        min_net_no_margin=int(args.min_net_no_margin),
    )

    candidate_summary_path = tables_dir / "candidate_summary_v1.csv"
    image_summary_path = tables_dir / "image_summary_v1.csv"
    component_summary_path = tables_dir / "component_summary_v1.csv"
    operation_summary_path = tables_dir / "compiled_operations_v1.csv"
    spec_path = output_dir / "compiled_overlay_spec.json"

    candidate_summary_df.to_csv(candidate_summary_path, index=False)
    image_summary_df.to_csv(image_summary_path, index=False)
    component_summary_df.to_csv(component_summary_path, index=False)
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
        "base_submission_dir": str(base_submission_dir),
        "rule_name": str(args.rule_name),
        "operations": operations,
        "candidate_summary_path": str(candidate_summary_path),
        "image_summary_path": str(image_summary_path),
        "component_summary_path": str(component_summary_path),
        "compiled_overlay_spec_path": str(spec_path),
        "suggested_build_command": build_command,
        "filters": {
            "datasets": args.datasets,
            "candidate_keys": args.candidate_keys,
            "min_no_degree": int(args.min_no_degree),
            "min_net_no_margin": int(args.min_net_no_margin),
        },
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    total_split_judgments = sum(1 for item in judgments if str(item.get("candidate_type", "")) == "split")
    lines = [
        "# Manual Split Compiler v1",
        "",
        f"- Session name: `{session_name}`",
        f"- Pair judgments path: `{pair_judgments_path}`",
        f"- Base submission dir: `{base_submission_dir}`",
        f"- Total loaded split judgments: `{total_split_judgments}`",
        f"- Candidate filter: `{args.candidate_keys or []}`",
        f"- Dataset filter: `{args.datasets or []}`",
        f"- Conservative gate: `no_degree >= {int(args.min_no_degree)}` and `no_degree - yes_degree >= {int(args.min_net_no_margin)}`",
        f"- Compiled overlay spec: `{spec_path}`",
        "",
        "## What This Compiler Does",
        "",
        "- It does not explode the whole cluster by default.",
        "- It marks an image for split only when the manual split evidence is strong enough.",
        "- After singletonizing those moved images, it regroups any moved yes-component that has no internal no-conflict.",
        "",
        "## Candidate Summary",
        "",
        _format_table_or_note(candidate_summary_df, note="_No split candidate passed the current compile filters._", limit=40),
        "",
        "## Image Summary",
        "",
        _format_table_or_note(
            image_summary_df[
                [
                    column
                    for column in [
                        "dataset",
                        "base_cluster_id",
                        "image_id",
                        "judged_pair_count",
                        "yes_degree",
                        "no_degree",
                        "uncertain_degree",
                        "net_no_margin",
                        "selected_for_split",
                        "anchor_image_id",
                    ]
                    if column in image_summary_df.columns
                ]
            ]
            if not image_summary_df.empty
            else image_summary_df,
            note="_No image-level split score rows were produced._",
            limit=80,
        ),
        "",
        "## Regroup Components",
        "",
        _format_table_or_note(component_summary_df, note="_No moved yes-components were regrouped after split._", limit=40),
        "",
        "## Suggested Build Command",
        "",
        "```bash",
        build_command,
        "```",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[manual_split_compiler] spec: {spec_path}")
    print(f"[manual_split_compiler] candidate_summary: {candidate_summary_path}")
    print(f"[manual_split_compiler] summary: {reports_dir / 'summary.md'}")
    print(f"[manual_split_compiler] build_command: {build_command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
