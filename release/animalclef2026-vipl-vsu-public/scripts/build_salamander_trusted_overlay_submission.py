#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_yellow_v1")
DEFAULT_TRUSTED_BATCH_DIR = Path("artifacts/analysis/salamander_trusted_batch_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/submissions/kaggle_variant_salamander_trusted_overlay_v1")


def _format_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([header, sep, *rows])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import build_submission
    from animalclef_analysis.salamander_trusted_overlay import apply_salamander_trusted_overlay, write_overlay_spec

    parser = argparse.ArgumentParser(description="Apply clean Salamander trusted labels as a low-risk overlay on a base submission.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--base-submission-dir", type=Path, default=DEFAULT_BASE_SUBMISSION_DIR)
    parser.add_argument("--trusted-batch-dir", type=Path, default=DEFAULT_TRUSTED_BATCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--rule-name", type=str, default="salamander_trusted_overlay_v1")
    parser.add_argument("--enable-cannot-link-singletons", action="store_true")
    parser.add_argument("--max-cannot-link-singletons", type=int, default=0)
    parser.add_argument("--submission-description", type=str, default="Salamander clean trusted label overlay v1")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    base_submission_dir = args.base_submission_dir if args.base_submission_dir.is_absolute() else repo_root / args.base_submission_dir
    trusted_batch_dir = args.trusted_batch_dir if args.trusted_batch_dir.is_absolute() else repo_root / args.trusted_batch_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    sample_submission_path = args.sample_submission_path if args.sample_submission_path.is_absolute() else repo_root / args.sample_submission_path

    base_predictions_path = base_submission_dir / "tables" / "test_predictions_v1.csv"
    clean_membership_path = trusted_batch_dir / "tables" / "trusted_membership_clean_v1.csv"
    cannot_link_path = trusted_batch_dir / "tables" / "cannot_link_pairs_v1.csv"
    if not base_predictions_path.exists():
        raise FileNotFoundError(base_predictions_path)
    if not clean_membership_path.exists():
        raise FileNotFoundError(clean_membership_path)
    if not cannot_link_path.exists():
        raise FileNotFoundError(cannot_link_path)

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(base_predictions_path)
    clean_membership_df = pd.read_csv(clean_membership_path)
    cannot_link_df = pd.read_csv(cannot_link_path)
    result = apply_salamander_trusted_overlay(
        pred_df=pred_df,
        clean_membership_df=clean_membership_df,
        cannot_link_df=cannot_link_df,
        enable_cannot_link_singletons=bool(args.enable_cannot_link_singletons),
        max_cannot_link_singletons=int(args.max_cannot_link_singletons),
        rule_name=str(args.rule_name),
    )

    result.prediction_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    result.changed_df.to_csv(tables_dir / "trusted_overlay_changed_rows_v1.csv", index=False)
    result.operation_df.to_csv(tables_dir / "trusted_overlay_operations_v1.csv", index=False)
    result.cannot_link_violation_df.to_csv(tables_dir / "cannot_link_violations_after_overlay_v1.csv", index=False)
    result.summary_df.to_csv(tables_dir / "trusted_overlay_summary_v1.csv", index=False)
    spec_path = output_dir / "trusted_overlay_spec.json"
    write_overlay_spec(spec_path, result.spec_payload)

    build_submission(
        test_pred_df=result.prediction_df,
        sample_submission_path=sample_submission_path.resolve(),
        output_path=output_dir / "submission.csv",
    )

    summary_payload = {
        "base_submission_dir": str(base_submission_dir.resolve()),
        "base_predictions_path": str(base_predictions_path.resolve()),
        "trusted_batch_dir": str(trusted_batch_dir.resolve()),
        "clean_membership_path": str(clean_membership_path.resolve()),
        "cannot_link_path": str(cannot_link_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "submission_path": str((output_dir / "submission.csv").resolve()),
        "rule_name": str(args.rule_name),
        "operation_count": int(len(result.operation_df)),
        "changed_rows": int(len(result.changed_df)),
        "enable_cannot_link_singletons": bool(args.enable_cannot_link_singletons),
        "max_cannot_link_singletons": int(args.max_cannot_link_singletons),
        "submission_description": str(args.submission_description),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Trusted Overlay v1",
        "",
        f"- Base submission: `{base_submission_dir}`",
        f"- Clean membership: `{clean_membership_path}`",
        f"- Cannot-link pairs: `{cannot_link_path}`",
        f"- Rule: `{args.rule_name}`",
        f"- Operations: `{len(result.operation_df)}`",
        f"- Changed rows: `{len(result.changed_df)}`",
        f"- Cannot-link singleton splits enabled: `{bool(args.enable_cannot_link_singletons)}`",
        f"- Max cannot-link singleton splits: `{int(args.max_cannot_link_singletons)}`",
        f"- Submission: `{output_dir / 'submission.csv'}`",
        "",
        "## Policy",
        "",
        "- Default behavior only enforces clean trusted components as must-link attach operations.",
        "- `cannot_link_pairs_v1.csv` is audited and exported as violations, but is not used for singleton splitting unless the explicit flag is enabled.",
        "- Conflicted trusted components are excluded upstream by `trusted_membership_clean_v1.csv`.",
        "",
        "## Summary",
        "",
        _format_table(result.summary_df),
        "",
        "## Operations Preview",
        "",
        _format_table(result.operation_df.head(40) if not result.operation_df.empty else result.operation_df),
        "",
        "## Cannot-Link Violations After Overlay Preview",
        "",
        _format_table(result.cannot_link_violation_df.head(80) if not result.cannot_link_violation_df.empty else result.cannot_link_violation_df),
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_trusted_overlay] submission: {output_dir / 'submission.csv'}")
    print(f"[salamander_trusted_overlay] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[salamander_trusted_overlay] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
