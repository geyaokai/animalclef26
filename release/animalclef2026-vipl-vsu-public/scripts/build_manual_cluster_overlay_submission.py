#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1")


def _first_non_null(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def _build_route_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    grouped = pred_df.groupby("dataset", sort=True)
    rows: list[dict[str, object]] = []
    for dataset, group in grouped:
        route_name = _first_non_null(group["route_name"]) if "route_name" in group.columns else None
        embedding_dim = _first_non_null(group["embedding_dim"]) if "embedding_dim" in group.columns else None
        threshold = _first_non_null(group["chosen_threshold"]) if "chosen_threshold" in group.columns else None
        rerank_enabled = _first_non_null(group["rerank_enabled"]) if "rerank_enabled" in group.columns else None
        local_weight = _first_non_null(group["local_weight"]) if "local_weight" in group.columns else None
        rows.append(
            {
                "dataset": str(dataset),
                "route_name": str(route_name) if route_name is not None else "",
                "embedding_dim": int(embedding_dim) if embedding_dim is not None else "",
                "threshold": float(threshold) if threshold is not None else "",
                "rerank_enabled": bool(rerank_enabled) if rerank_enabled is not None else False,
                "local_weight": float(local_weight) if local_weight is not None else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _build_cluster_delta_summary(base_df: pd.DataFrame, overlay_df: pd.DataFrame, changed_df: pd.DataFrame, operation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    datasets = sorted(set(base_df["dataset"].astype(str)).union(set(overlay_df["dataset"].astype(str))))
    for dataset in datasets:
        base_slice = base_df[base_df["dataset"].astype(str).eq(dataset)].copy()
        overlay_slice = overlay_df[overlay_df["dataset"].astype(str).eq(dataset)].copy()
        base_counts = base_slice["pred_cluster_id"].astype(int).value_counts()
        overlay_counts = overlay_slice["pred_cluster_id"].astype(int).value_counts()
        changed_slice = changed_df[changed_df["dataset"].astype(str).eq(dataset)].copy()
        op_slice = operation_df[operation_df["dataset"].astype(str).eq(dataset)].copy()
        rows.append(
            {
                "dataset": dataset,
                "base_clusters": int(base_counts.size),
                "overlay_clusters": int(overlay_counts.size),
                "base_singletons": int((base_counts == 1).sum()),
                "overlay_singletons": int((overlay_counts == 1).sum()),
                "changed_images": int(changed_slice["image_id"].astype(str).nunique()) if not changed_slice.empty else 0,
                "applied_operations": "|".join(op_slice["operation_id"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table
    from animalclef_analysis.manual_cluster_overlay import apply_manual_cluster_overlay, load_manual_overlay_spec

    parser = argparse.ArgumentParser(description="Build a dataset-agnostic manual cluster overlay submission variant.")
    parser.add_argument("--base-submission-dir", type=Path, default=DEFAULT_BASE_SUBMISSION_DIR)
    parser.add_argument("--overlay-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--baseline-public-score", type=float, default=0.48755)
    parser.add_argument("--current-best-public-score", type=float, default=0.48876)
    parser.add_argument("--current-best-description", type=str, default="SeaTurtle cropped fusion w=0.3 t=0.7 overlay")
    parser.add_argument("--submission-description", type=str, default="")
    args = parser.parse_args()

    base_submission_dir = (repo_root / args.base_submission_dir).resolve() if not args.base_submission_dir.is_absolute() else args.base_submission_dir.resolve()
    overlay_spec_path = (repo_root / args.overlay_spec).resolve() if not args.overlay_spec.is_absolute() else args.overlay_spec.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_pred_df = pd.read_csv(base_submission_dir / "tables" / "test_predictions_v1.csv")
    spec = load_manual_overlay_spec(overlay_spec_path)
    overlay_pred_df, changed_df, operation_df = apply_manual_cluster_overlay(base_pred_df, spec=spec)
    route_df = _build_route_summary(overlay_pred_df)
    cluster_summary_df = _build_cluster_delta_summary(base_pred_df, overlay_pred_df, changed_df, operation_df)

    overlay_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    changed_df.to_csv(tables_dir / "changed_images_v1.csv", index=False)
    operation_df.to_csv(tables_dir / "overlay_operations_v1.csv", index=False)
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)
    for dataset in operation_df["dataset"].astype(str).drop_duplicates().tolist():
        dataset_df = overlay_pred_df[overlay_pred_df["dataset"].astype(str).eq(dataset)].copy()
        dataset_df.to_csv(tables_dir / f"{dataset}_test_predictions_v1.csv", index=False)

    submission_path = output_dir / "submission.csv"
    build_submission(
        test_pred_df=overlay_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=submission_path,
    )

    summary_json = {
        "base_submission_dir": str(base_submission_dir),
        "overlay_spec_path": str(overlay_spec_path),
        "rule_name": str(spec.rule_name),
        "submission_description": str(args.submission_description),
        "submission_path": str(submission_path),
        "test_predictions_path": str(tables_dir / "test_predictions_v1.csv"),
        "changed_images_path": str(tables_dir / "changed_images_v1.csv"),
        "overlay_operations_path": str(tables_dir / "overlay_operations_v1.csv"),
        "cluster_summary_path": str(tables_dir / "cluster_summary_v1.csv"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "manual_overlay_spec.json").write_text(json.dumps(spec.raw_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    changed_preview_df = changed_df.head(40) if not changed_df.empty else pd.DataFrame(columns=["dataset", "image_id", "operation_id", "overlay_action"])
    lines = [
        "# Submission Variant",
        "",
        f"- Baseline submission: `{base_submission_dir.name}`",
        f"- Baseline public score when this artifact family was created: `{float(args.baseline_public_score):.5f}`",
        f"- Note on current leaderboard state: current best public in Kaggle list is `{float(args.current_best_public_score):.5f}` from `{args.current_best_description}`.",
        f"- Manual overlay spec: `{overlay_spec_path}`",
        f"- Overlay rule name: `{spec.rule_name}`",
        f"- Submission description: `{args.submission_description}`",
        "",
        "## Architecture",
        "",
        "- Overall system: `dataset-routed hybrid clustering pipeline + manual cluster overlay post-processor`.",
        "- Global flow: `image -> base route prediction -> manual split/attach overlay -> final submission cluster label`.",
        "- Per-dataset route summary:",
        "",
        dataframe_to_markdown_table(route_df),
        "",
        "## Overlay Operations",
        "",
        dataframe_to_markdown_table(operation_df if not operation_df.empty else pd.DataFrame(columns=["operation_id", "dataset", "action", "changed_count", "note"])),
        "",
        "## Changed Images",
        "",
        dataframe_to_markdown_table(changed_preview_df),
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Validation",
        "",
        "- `submission.csv` row count matches `sample_submission.csv`.",
        "- Columns are exactly `image_id,cluster`.",
        "- `image_id` order matches `sample_submission.csv`.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[manual_cluster_overlay] submission: {submission_path}")
    print(f"[manual_cluster_overlay] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[manual_cluster_overlay] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
