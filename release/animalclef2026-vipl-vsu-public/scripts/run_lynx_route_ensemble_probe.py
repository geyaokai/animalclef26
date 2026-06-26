#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/lynx_route_ensemble_probe_20260420")
LYNX_DATASET = "LynxID2025"


def _infer_reference_metadata(
    *,
    repo_root: Path,
    prediction_routes: list[list[str]],
    embedding_routes: list[list[str]],
    score_routes: list[list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from animalclef_analysis.descriptor_baselines import load_cached_embedding_bundle
    from animalclef_analysis.lynx_route_ensemble import load_reference_metadata

    for _name, val_prediction_path, test_prediction_path in prediction_routes:
        val_df = load_reference_metadata(Path(val_prediction_path))
        test_df = load_reference_metadata(Path(test_prediction_path))
        if not val_df.empty and not test_df.empty:
            return val_df, test_df
    for _name, source_dir in embedding_routes:
        bundle = load_cached_embedding_bundle(source_dir=(repo_root / source_dir).resolve(), name=source_dir)
        val_df = bundle.val_df[bundle.val_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
        test_df = bundle.test_df[bundle.test_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
        if not val_df.empty and not test_df.empty:
            val_df["image_id"] = val_df["image_id"].astype(str)
            test_df["image_id"] = test_df["image_id"].astype(str)
            if "identity" in val_df.columns:
                val_df["identity"] = val_df["identity"].fillna("").astype(str)
            if "identity" in test_df.columns:
                test_df["identity"] = test_df["identity"].fillna("").astype(str)
            return val_df, test_df
    for _name, _score_space, _val_score_path, val_metadata_path, _test_score_path, test_metadata_path in score_routes:
        val_df = load_reference_metadata(Path(val_metadata_path))
        test_df = load_reference_metadata(Path(test_metadata_path))
        if not val_df.empty and not test_df.empty:
            return val_df, test_df
    raise ValueError("Could not infer Lynx reference metadata from the provided route inputs.")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table
    from animalclef_analysis.lynx_route_ensemble import (
        LYNX_DATASET as MODULE_LYNX_DATASET,
        load_embedding_route_candidate,
        load_prediction_route_candidate,
        load_reference_metadata,
        load_score_route_candidate,
        run_route_ensemble_probe,
    )

    del MODULE_LYNX_DATASET

    parser = argparse.ArgumentParser(
        description="Probe a Lynx multi-route co-association ensemble from prediction tables, embedding dirs, or score matrices."
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction-route", nargs=3, action="append", default=[], metavar=("NAME", "VAL_PRED", "TEST_PRED"))
    parser.add_argument("--embedding-route", nargs=2, action="append", default=[], metavar=("NAME", "SOURCE_DIR"))
    parser.add_argument(
        "--score-route",
        nargs=6,
        action="append",
        default=[],
        metavar=("NAME", "SCORE_SPACE", "VAL_SCORE", "VAL_META", "TEST_SCORE", "TEST_META"),
    )
    parser.add_argument("--val-reference-metadata", type=Path)
    parser.add_argument("--test-reference-metadata", type=Path)
    parser.add_argument("--route-thresholds", nargs="+", type=float, default=[0.75, 0.8, 0.825, 0.85, 0.875, 0.9])
    parser.add_argument("--ensemble-thresholds", nargs="+", type=float, default=[0.5, 0.67, 0.75, 1.0])
    parser.add_argument("--min-route-count", type=int, default=2)
    parser.add_argument("--max-route-count", type=int, default=None)
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--coassociation-export-min-score", type=float, default=0.5)
    parser.add_argument("--route-name", type=str, default="lynx_route_ensemble_v1")
    parser.add_argument("--export-test-override", action="store_true")
    parser.add_argument("--base-predictions-path", type=Path)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    args = parser.parse_args()

    prediction_routes = [[str(value) for value in spec] for spec in args.prediction_route]
    embedding_routes = [[str(value) for value in spec] for spec in args.embedding_route]
    score_routes = [[str(value) for value in spec] for spec in args.score_route]
    if not prediction_routes and not embedding_routes and not score_routes:
        raise ValueError("Provide at least one route via --prediction-route, --embedding-route, or --score-route.")

    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if args.val_reference_metadata is not None and args.test_reference_metadata is not None:
        val_reference_df = load_reference_metadata(args.val_reference_metadata.resolve())
        test_reference_df = load_reference_metadata(args.test_reference_metadata.resolve())
    else:
        val_reference_df, test_reference_df = _infer_reference_metadata(
            repo_root=repo_root,
            prediction_routes=prediction_routes,
            embedding_routes=embedding_routes,
            score_routes=score_routes,
        )
    val_reference_df = val_reference_df[val_reference_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    test_reference_df = test_reference_df[test_reference_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    if val_reference_df.empty or test_reference_df.empty:
        raise ValueError("Reference metadata does not contain Lynx rows.")
    if "identity" not in val_reference_df.columns or not val_reference_df["identity"].fillna("").astype(str).ne("").any():
        raise ValueError("Validation reference metadata must contain identity labels for the Lynx val sweep.")

    candidates = []
    for route_name, val_prediction_path, test_prediction_path in prediction_routes:
        candidates.append(
            load_prediction_route_candidate(
                route_name=route_name,
                val_prediction_path=(repo_root / val_prediction_path).resolve(),
                test_prediction_path=(repo_root / test_prediction_path).resolve(),
                val_reference_df=val_reference_df,
                test_reference_df=test_reference_df,
            )
        )
    for route_name, source_dir in embedding_routes:
        candidates.append(
            load_embedding_route_candidate(
                route_name=route_name,
                source_dir=(repo_root / source_dir).resolve(),
                thresholds=[float(value) for value in args.route_thresholds],
                val_reference_df=val_reference_df,
                test_reference_df=test_reference_df,
            )
        )
    for route_name, score_space, val_score_path, val_metadata_path, test_score_path, test_metadata_path in score_routes:
        candidates.append(
            load_score_route_candidate(
                route_name=route_name,
                score_space=str(score_space),
                val_score_path=(repo_root / val_score_path).resolve(),
                val_metadata_path=(repo_root / val_metadata_path).resolve(),
                test_score_path=(repo_root / test_score_path).resolve(),
                test_metadata_path=(repo_root / test_metadata_path).resolve(),
                thresholds=[float(value) for value in args.route_thresholds],
                val_reference_df=val_reference_df,
                test_reference_df=test_reference_df,
            )
        )

    probe_result = run_route_ensemble_probe(
        candidates=candidates,
        val_reference_df=val_reference_df,
        test_reference_df=test_reference_df,
        ensemble_thresholds=[float(value) for value in args.ensemble_thresholds],
        min_route_count=int(args.min_route_count),
        max_route_count=int(args.max_route_count) if args.max_route_count is not None else None,
        max_route_candidates=int(args.max_route_candidates) if args.max_route_candidates is not None else None,
        coassociation_export_min_score=float(args.coassociation_export_min_score),
        export_test_override=bool(args.export_test_override),
        output_dir=output_dir,
        base_predictions_path=args.base_predictions_path.resolve() if args.base_predictions_path is not None else None,
        sample_submission_path=args.sample_submission_path.resolve() if args.sample_submission_path is not None else None,
        route_name=str(args.route_name),
    )

    probe_result.route_sweep_df.to_csv(tables_dir / "route_threshold_sweep_v1.csv", index=False)
    probe_result.route_candidates_df.to_csv(tables_dir / "route_candidates_v1.csv", index=False)
    probe_result.selected_routes_df.to_csv(tables_dir / "route_candidates_selected_v1.csv", index=False)
    probe_result.route_agreement_df.to_csv(tables_dir / "route_agreement_v1.csv", index=False)
    probe_result.ensemble_sweep_df.to_csv(tables_dir / "ensemble_val_sweep_v1.csv", index=False)
    probe_result.cluster_shape_df.to_csv(tables_dir / "cluster_shape_summary_v1.csv", index=False)
    probe_result.best_route_df.to_csv(tables_dir / "best_route_ensemble_v1.csv", index=False)
    probe_result.best_val_prediction_df.to_csv(tables_dir / "best_val_predictions_v1.csv", index=False)
    probe_result.best_test_prediction_df.to_csv(tables_dir / "lynx_test_predictions_v1.csv", index=False)
    probe_result.best_val_coassociation_df.to_csv(tables_dir / "best_val_coassociation_pairs_v1.csv", index=False)
    probe_result.best_test_coassociation_df.to_csv(tables_dir / "best_test_coassociation_pairs_v1.csv", index=False)

    summary_payload = {
        "input_route_count": int(len(candidates)),
        "selected_route_count": int(len(probe_result.selected_routes_df)),
        "best_single_route": probe_result.best_single_row,
        "best_route_ensemble": probe_result.best_row,
        "best_route_names": probe_result.best_route_names,
        "best_vote_threshold": float(probe_result.best_vote_threshold),
        "exported_paths": probe_result.exported_paths,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    ensemble_top_df = probe_result.ensemble_sweep_df.head(10).copy()
    route_top_df = probe_result.route_candidates_df.head(10).copy()
    agreement_top_df = probe_result.route_agreement_df.head(10).copy()
    best_shape_df = probe_result.cluster_shape_df[
        probe_result.cluster_shape_df["object_name"].astype(str).eq(str(args.route_name))
    ].copy().reset_index(drop=True)
    lines = [
        "# Lynx Route Ensemble Probe",
        "",
        "## Inputs",
        "",
        f"- 输入 route 数: `{len(candidates)}`，参与 ensemble sweep 的 route 数: `{len(probe_result.selected_routes_df)}`。",
        f"- 输入类型: `prediction={len(prediction_routes)}`, `embedding={len(embedding_routes)}`, `score={len(score_routes)}`。",
        f"- Ensemble route 名称: `{args.route_name}`。",
        "",
        "## Best Single Route",
        "",
        dataframe_to_markdown_table(pd.DataFrame([probe_result.best_single_row])),
        "",
        "## Best Ensemble",
        "",
        dataframe_to_markdown_table(pd.DataFrame([probe_result.best_row])),
        "",
        f"- Best subset: `{', '.join(probe_result.best_route_names)}`。",
        f"- Vote threshold: `{float(probe_result.best_vote_threshold):.6f}`。",
        "",
        "## Route Candidates Top-10",
        "",
        dataframe_to_markdown_table(route_top_df),
        "",
        "## Ensemble Sweep Top-10",
        "",
        dataframe_to_markdown_table(ensemble_top_df),
        "",
        "## Route Agreement Top-10",
        "",
        dataframe_to_markdown_table(agreement_top_df),
        "",
        "## Best Ensemble Cluster Shape",
        "",
        dataframe_to_markdown_table(best_shape_df),
        "",
        "## 输出",
        "",
        f"- Route val sweep: `{tables_dir / 'route_threshold_sweep_v1.csv'}`。",
        f"- Route agreement: `{tables_dir / 'route_agreement_v1.csv'}`。",
        f"- Ensemble val sweep: `{tables_dir / 'ensemble_val_sweep_v1.csv'}`。",
        f"- Val co-association 表: `{tables_dir / 'best_val_coassociation_pairs_v1.csv'}`。",
        f"- Test co-association 表: `{tables_dir / 'best_test_coassociation_pairs_v1.csv'}`。",
        f"- Lynx-only test override: `{tables_dir / 'lynx_test_predictions_v1.csv'}`。",
        "",
    ]
    if probe_result.exported_paths:
        lines.extend(
            [
                "## Test Override",
                "",
                f"- 合并后的 test predictions: `{tables_dir / 'test_predictions_v1.csv'}`。",
                f"- Submission: `{output_dir / 'submission.csv'}`。",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[lynx_route_ensemble_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[lynx_route_ensemble_probe] ensemble_sweep: {tables_dir / 'ensemble_val_sweep_v1.csv'}")
    if probe_result.exported_paths:
        print(f"[lynx_route_ensemble_probe] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
