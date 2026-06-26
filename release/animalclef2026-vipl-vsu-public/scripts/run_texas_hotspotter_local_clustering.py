#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.local_matching import HotspotterConfig
    from animalclef_analysis.texas_hotspotter_local_clustering import (
        DEFAULT_BASE_PREDICTIONS_PATH,
        DEFAULT_BEST_PUBLIC_SCORE,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_SUBMISSION_DIR,
        DEFAULT_TEXAS_EXPERIMENT_DIR,
        build_hotspotter_affinity_matrices,
        build_texas_override_submission,
        count_kaggle_submissions_today,
        evaluate_local_clustering_candidates,
        kaggle_submit_and_poll,
        load_texas_proxy_bundle,
    )
    from animalclef_analysis.texas_hotspotter_probe import run_texas_hotspotter_probe

    parser = argparse.ArgumentParser(description="Run Texas HotSpotter local clustering and optionally submit a Texas-only override.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_TEXAS_EXPERIMENT_DIR)
    parser.add_argument("--base-predictions-path", type=Path, default=DEFAULT_BASE_PREDICTIONS_PATH)
    parser.add_argument("--manifest-path", type=Path, default=Path("artifacts/manifests/texas_center_body_square_repaired_v1/tables/manifest_test_texas_center_body_square_gray_v1.csv"))
    parser.add_argument("--analysis-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submission-output-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--probe-top-k", type=int, default=50)
    parser.add_argument("--n-shortlist", type=int, default=50)
    parser.add_argument("--max-features-per-image", type=int, default=128)
    parser.add_argument("--feature-selection-strategy", type=str, default="largest_scale")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--best-public-score", type=float, default=DEFAULT_BEST_PUBLIC_SCORE)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--poll-timeout-seconds", type=int, default=900)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    analysis_output_dir = (args.analysis_output_dir if args.analysis_output_dir.is_absolute() else (repo_root / args.analysis_output_dir)).resolve()
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = analysis_output_dir / "tables"
    reports_dir = analysis_output_dir / "reports"
    for path in [tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    proxy_bundle = load_texas_proxy_bundle(
        repo_root=repo_root,
        experiment_dir=args.experiment_dir,
        top_k=8,
    )

    probe_dir = analysis_output_dir / "hotspotter_probe"
    probe_outputs = run_texas_hotspotter_probe(
        repo_root=repo_root,
        manifest_path=args.manifest_path,
        output_dir=probe_dir,
        top_k=int(args.probe_top_k),
        config=HotspotterConfig(
            n_shortlist=int(args.n_shortlist),
            max_features_per_image=int(args.max_features_per_image),
            feature_selection_strategy=str(args.feature_selection_strategy),
        ),
    )
    ranking_df = pd.read_csv(probe_outputs["ranking_path"]).copy()
    pair_df = pd.read_csv(probe_outputs["pair_path"]).copy()
    ranking_df["image_id"] = ranking_df["image_id"].astype(str)
    ranking_df["neighbor_image_id"] = ranking_df["neighbor_image_id"].astype(str)
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    score_matrices, pair_summary_df = build_hotspotter_affinity_matrices(
        ranking_df=ranking_df,
        reference_df=proxy_bundle.metadata_df,
        top_k=int(args.probe_top_k),
    )
    summary_df, prediction_frames, candidates = evaluate_local_clustering_candidates(
        metadata_df=proxy_bundle.metadata_df,
        score_matrices=score_matrices,
        candidate_pair_df=proxy_bundle.candidate_pair_df,
        teacher_anchor_labels=proxy_bundle.teacher_anchor_labels,
        teacher_topk_indices=proxy_bundle.teacher_topk_indices,
        top_k_overlap=8,
    )
    if not candidates:
        raise ValueError("No Texas local clustering candidates were produced.")

    selected = candidates[0]
    second_best = candidates[1] if len(candidates) > 1 else None

    pair_summary_path = tables_dir / "hotspotter_pair_affinity_v1.csv"
    candidate_summary_path = tables_dir / "candidate_summary_v1.csv"
    top_candidates_path = tables_dir / "top_candidates_v1.csv"
    pair_summary_df.to_csv(pair_summary_path, index=False)
    summary_df.to_csv(candidate_summary_path, index=False)
    pd.DataFrame([candidate.summary_row for candidate in candidates[:10]]).to_csv(top_candidates_path, index=False)

    for name, frame in prediction_frames.items():
        safe_name = name.replace(":", "__")
        frame.to_csv(tables_dir / f"{safe_name}_predictions_v1.csv", index=False)

    route_name = f"hotspotter_local_{selected.score_matrix_name}_{selected.method}"
    submission_outputs = build_texas_override_submission(
        repo_root=repo_root,
        base_predictions_path=args.base_predictions_path,
        texas_pred_df=selected.prediction_df,
        output_dir=args.submission_output_dir,
        route_name=route_name,
        route_summary=selected.summary_row,
    )

    kaggle_status: dict[str, object] = {}
    if args.submit:
        used_before, submission_listing = count_kaggle_submissions_today(repo_root=repo_root)
        description = (
            args.description.strip()
            or f"Texas HotSpotter local {selected.method} {selected.score_matrix_name} ps={float(selected.summary_row.get('proxy_score', 0.0)):.4f}"
        )
        kaggle_status = {
            "used_slots_before_submit": int(used_before),
            "raw_submission_listing_before": submission_listing,
            **kaggle_submit_and_poll(
                repo_root=repo_root,
                submission_path=submission_outputs["submission_path"],
                description=description,
                poll_seconds=int(args.poll_seconds),
                timeout_seconds=int(args.poll_timeout_seconds),
            ),
        }
        kaggle_status["best_public_score_before"] = float(args.best_public_score)
        kaggle_status["improved"] = (
            kaggle_status.get("public_score") is not None
            and float(kaggle_status["public_score"]) > float(args.best_public_score)
        )
        kaggle_status["second_best_candidate"] = second_best.summary_row if second_best is not None else {}

    report = {
        "selected_candidate": selected.summary_row,
        "second_best_candidate": second_best.summary_row if second_best is not None else {},
        "probe_outputs": {key: str(value) for key, value in probe_outputs.items()},
        "pair_summary_path": str(pair_summary_path),
        "candidate_summary_path": str(candidate_summary_path),
        "submission_outputs": {key: str(value) for key, value in submission_outputs.items()},
        "kaggle_status": kaggle_status,
    }
    (reports_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    preview_columns = [
        "score_matrix_name",
        "method",
        "param_key",
        "proxy_score",
        "clusters",
        "largest_cluster_size",
        "singleton_ratio",
        "candidate_pair_keep_ratio",
        "mutual_topk_pair_keep_ratio",
        "cluster_delta_vs_teacher_anchor",
    ]
    summary_lines = [
        "# Texas HotSpotter Local Clustering",
        "",
        "## Selected Candidate",
        "",
        _markdown_table(pd.DataFrame([selected.summary_row])[preview_columns]),
        "",
        "## Top Candidates",
        "",
        _markdown_table(summary_df.loc[:, [column for column in preview_columns if column in summary_df.columns]].head(12)),
        "",
        "## Outputs",
        "",
        f"- `pair_summary_path`: `{pair_summary_path}`",
        f"- `candidate_summary_path`: `{candidate_summary_path}`",
        f"- `top_candidates_path`: `{top_candidates_path}`",
        f"- `submission_path`: `{submission_outputs['submission_path']}`",
        f"- `submission_summary_path`: `{submission_outputs['summary_path']}`",
        "",
    ]
    if kaggle_status:
        summary_lines.extend(
            [
                "## Kaggle",
                "",
                f"- `used_slots_before_submit`: `{kaggle_status.get('used_slots_before_submit')}`",
                f"- `submit_returncode`: `{kaggle_status.get('submit_returncode')}`",
                f"- `status`: `{kaggle_status.get('status', '')}`",
                f"- `public_score`: `{kaggle_status.get('public_score')}`",
                f"- `improved_vs_best_public`: `{kaggle_status.get('improved')}`",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"[texas_hotspotter_local] summary: {reports_dir / 'summary.md'}")
    print(f"[texas_hotspotter_local] submission: {submission_outputs['submission_path']}")
    if kaggle_status:
        print(f"[texas_hotspotter_local] kaggle_status: {kaggle_status.get('status', '')} score={kaggle_status.get('public_score')}")
    return 0


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    raise SystemExit(main())
