#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_REVIEW_DIR = Path("artifacts/analysis/salamander_singleton_rescue_review_v1")
DEFAULT_PAIR_JUDGMENTS = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_singleton_rescue_compiled_v1")


def _resolve_path(repo_root: Path, input_path: Path) -> Path:
    return input_path.resolve() if input_path.is_absolute() else (repo_root / input_path).resolve()


def _table_markdown(frame: pd.DataFrame, *, top_k: int = 20) -> str:
    if frame.empty:
        return "_空表。_"
    preview = frame.head(int(top_k)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.manual_review_workbench import (
        build_overlay_command,
        export_operations_spec,
        load_pair_judgments,
    )
    from animalclef_analysis.singleton_rescue_compiler import compile_singleton_rescue_merge_judgments

    parser = argparse.ArgumentParser(description="Compile accepted singleton rescue merge judgments into a manual overlay spec.")
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--pair-judgments", type=Path, default=DEFAULT_PAIR_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rule-name", type=str, default="manual_singleton_rescue_v1")
    parser.add_argument(
        "--base-submission-dir",
        type=Path,
        default=Path("artifacts/submissions/kaggle_variant_seaturtle_origcrop_w0p3_on_gate2_v1"),
    )
    parser.add_argument(
        "--suggested-submission-dir",
        type=Path,
        default=Path("artifacts/submissions/kaggle_variant_salamander_singletonrescue_on_bestpublic_v1"),
    )
    args = parser.parse_args()

    review_dir = _resolve_path(repo_root, args.review_dir)
    pair_judgments_path = _resolve_path(repo_root, args.pair_judgments)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    merge_candidate_df = pd.read_csv(review_dir / "tables" / "test_merge_candidates_v1.csv")
    pair_df = pd.read_csv(review_dir / "tables" / "test_pair_disagreement_v1.csv")
    session_name, judgments = load_pair_judgments(pair_judgments_path)
    result = compile_singleton_rescue_merge_judgments(merge_candidate_df, pair_df, judgments)

    candidate_summary_df = result.candidate_summary_df
    candidate_summary_df.to_csv(tables_dir / "candidate_summary_v1.csv", index=False)
    operations_df = pd.DataFrame(result.operations)
    operations_df.to_csv(tables_dir / "compiled_operations_v1.csv", index=False)
    spec_path = export_operations_spec(
        rule_name=str(args.rule_name),
        operations=result.operations,
        output_path=output_dir / "compiled_overlay_spec.json",
    )

    build_cmd = build_overlay_command(
        repo_root=repo_root,
        base_submission_dir=args.base_submission_dir,
        spec_path=spec_path,
        output_dir=args.suggested_submission_dir,
        submission_description="Salamander singleton rescue overlay on best public v1",
    )
    summary_json = {
        "review_dir": str(review_dir),
        "pair_judgments_path": str(pair_judgments_path),
        "session_name": str(session_name),
        "compiled_overlay_spec_path": str(spec_path),
        "accepted_operation_count": int(len(result.operations)),
        "build_command": build_cmd,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Singleton Rescue Compile",
        "",
        f"- review dir: `{review_dir}`",
        f"- pair judgments: `{pair_judgments_path}`",
        f"- session name: `{session_name}`",
        f"- compiled spec: `{spec_path}`",
        f"- accepted operation count: `{len(result.operations)}`",
        "",
        "## Candidate Summary",
        "",
        _table_markdown(candidate_summary_df),
        "",
        "## Operations",
        "",
        _table_markdown(operations_df),
        "",
        "## Suggested Build Command",
        "",
        "```bash",
        build_cmd,
        "```",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[singleton_rescue_compile] summary: {reports_dir / 'summary.md'}")
    print(f"[singleton_rescue_compile] spec: {spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
