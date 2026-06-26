#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_PROBE_DIRS = [
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_20260331"),
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_seed43_xgb"),
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_seed44_xgb"),
]
DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_XGB_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_transductive_seed_support_20260403")


def _apply_pair_probability_as_score(
    base_score: np.ndarray,
    pair_df: pd.DataFrame,
    probability_col: str,
    blend_scale: float,
) -> np.ndarray:
    fused = base_score.copy().astype(np.float32, copy=True)
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        base_value = float(base_score[left_index, right_index])
        probability = float(getattr(row, probability_col))
        score = min(1.0, base_value + float(blend_scale) * probability * (1.0 - base_value))
        fused[left_index, right_index] = score
        fused[right_index, left_index] = score
    np.fill_diagonal(fused, 1.0)
    return fused


def _align_embeddings_to_reference(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    candidate_df = candidate_df.copy().reset_index(drop=True)
    candidate_df["image_id"] = candidate_df["image_id"].astype(str)
    lookup = {str(row.image_id): index for index, row in enumerate(candidate_df.itertuples(index=False))}
    order = [lookup[str(image_id)] for image_id in reference_df["image_id"].astype(str).tolist()]
    return embeddings[order].astype(np.float32)


def _load_probe_reference_df(probe_dir: Path, path_column: str) -> pd.DataFrame:
    prediction_df = pd.read_csv(probe_dir / "tables" / "gbdt_predictions_v1.csv")
    prediction_df["image_id"] = prediction_df["image_id"].astype(str)
    first_threshold = sorted(prediction_df["threshold"].astype(float).unique().tolist())[0]
    reference_df = prediction_df[prediction_df["threshold"].astype(float).eq(first_threshold)].copy().reset_index(drop=True)
    keep_columns = [column for column in ["image_id", "dataset", "identity", path_column] if column in reference_df.columns]
    return reference_df.loc[:, keep_columns].copy().reset_index(drop=True)


def _load_probe_bundle(route_dir: Path, probe_dir: Path, path_column: str) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    reference_df = _load_probe_reference_df(probe_dir=probe_dir, path_column=path_column)
    route_val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    route_val_df["image_id"] = route_val_df["image_id"].astype(str)
    route_val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    eval_embeddings = _align_embeddings_to_reference(
        reference_df=reference_df,
        candidate_df=route_val_df,
        embeddings=route_val_embeddings,
    )
    eval_pair_df = pd.read_csv(probe_dir / "tables" / "eval_pair_features_v1.csv")
    return reference_df, eval_embeddings, eval_pair_df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import PATH_COLUMN, build_submission, dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.submission_baseline import _cluster_single_dataset_from_score_matrix
    from animalclef_analysis.transductive_seed_refinement import (
        apply_seed_center_support,
        build_stable_seed_bundle_from_score_matrix,
        pick_best_threshold_row,
        run_score_threshold_sweep,
    )

    parser = argparse.ArgumentParser(description="Probe Salamander transductive seed-center support on top of the current XGBoost route.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--probe-dirs", nargs="+", type=Path, default=[repo_root / path for path in DEFAULT_PROBE_DIRS])
    parser.add_argument("--route-dir", type=Path, default=repo_root / DEFAULT_ROUTE_DIR)
    parser.add_argument("--xgb-route-dir", type=Path, default=repo_root / DEFAULT_XGB_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchors", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.15, 0.20, 0.25, 0.30, 0.35])
    parser.add_argument("--support-scales", nargs="+", type=float, default=[0.50, 0.75, 1.00])
    parser.add_argument("--min-shared-affinities", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85])
    parser.add_argument("--min-seed-scores", nargs="+", type=float, default=[0.70, 0.75, 0.80])
    parser.add_argument("--support-modes", nargs="+", type=str, default=["same_best_seed", "any_seed"])
    parser.add_argument("--stability-delta", type=float, default=0.03)
    parser.add_argument("--min-seed-cluster-size", type=int, default=2)
    parser.add_argument("--max-seed-cluster-size", type=int, default=12)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--export-test-override", action="store_true")
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    probe_result_rows: list[dict[str, object]] = []
    best_by_probe: dict[str, dict[str, object]] = {}
    aggregate_inputs: list[pd.DataFrame] = []
    for probe_dir in [path.resolve() for path in args.probe_dirs]:
        probe_name = probe_dir.name
        eval_df, eval_embeddings, eval_pair_df = _load_probe_bundle(
            route_dir=args.route_dir.resolve(),
            probe_dir=probe_dir,
            path_column=PATH_COLUMN,
        )
        base_score = cosine_score_matrix(eval_embeddings)
        boosted_score = _apply_pair_probability_as_score(
            base_score=base_score,
            pair_df=eval_pair_df,
            probability_col="gbdt_same_identity_prob",
            blend_scale=float(args.blend_scale),
        )
        baseline_sweep_df, _baseline_prediction_df = run_score_threshold_sweep(
            df=eval_df,
            score_matrix=boosted_score,
            thresholds=[float(value) for value in args.thresholds],
        )
        baseline_best = pick_best_threshold_row(baseline_sweep_df)
        baseline_sweep_df.to_csv(tables_dir / f"{probe_name}_baseline_sweep_v1.csv", index=False)

        best_probe_sort_key: tuple[float, float, int] = (-1.0, -1.0, -1)
        best_probe_payload: dict[str, object] | None = None
        for anchor in [float(value) for value in args.anchors]:
            for min_seed_score in [float(value) for value in args.min_seed_scores]:
                seed_bundle = build_stable_seed_bundle_from_score_matrix(
                    target_df=eval_df,
                    score_matrix=boosted_score,
                    anchor_threshold=float(anchor),
                    stability_delta=float(args.stability_delta),
                    min_seed_cluster_size=int(args.min_seed_cluster_size),
                    max_seed_cluster_size=int(args.max_seed_cluster_size),
                    min_mean_score=float(min_seed_score),
                )
                for support_mode in args.support_modes:
                    for support_scale in [float(value) for value in args.support_scales]:
                        for min_shared_affinity in [float(value) for value in args.min_shared_affinities]:
                            refined_score, support_stats = apply_seed_center_support(
                                score_matrix=boosted_score,
                                seed_assignment_df=seed_bundle.seed_assignment_df,
                                support_scale=float(support_scale),
                                min_shared_affinity=float(min_shared_affinity),
                                mode=str(support_mode),
                            )
                            sweep_df, _prediction_df = run_score_threshold_sweep(
                                df=eval_df,
                                score_matrix=refined_score,
                                thresholds=[float(value) for value in args.thresholds],
                            )
                            best_row = pick_best_threshold_row(sweep_df)
                            row = {
                                "probe": probe_name,
                                "anchor_threshold": float(anchor),
                                "min_seed_score": float(min_seed_score),
                                "support_mode": str(support_mode),
                                "support_scale": float(support_scale),
                                "min_shared_affinity": float(min_shared_affinity),
                                "seed_clusters": int(support_stats["seed_clusters"]),
                                "boosted_pairs": int(support_stats["boosted_pairs"]),
                                "mean_delta": float(support_stats["mean_delta"]),
                                "max_delta": float(support_stats["max_delta"]),
                                "threshold": float(best_row["threshold"]),
                                "ari": float(best_row["ari"]),
                                "pairwise_f1": float(best_row["pairwise_f1"]),
                                "cluster_count": int(best_row["cluster_count"]),
                                "ari_delta_vs_xgb": round(float(best_row["ari"]) - float(baseline_best["ari"]), 6),
                                "pairwise_f1_delta_vs_xgb": round(float(best_row["pairwise_f1"]) - float(baseline_best["pairwise_f1"]), 6),
                            }
                            probe_result_rows.append(row)
                            sort_key = (
                                float(row["ari_delta_vs_xgb"]),
                                float(row["pairwise_f1_delta_vs_xgb"]),
                                int(row["seed_clusters"]),
                            )
                            if sort_key > best_probe_sort_key:
                                best_probe_sort_key = sort_key
                                best_probe_payload = row
        if best_probe_payload is None:
            raise RuntimeError(f"No transductive config evaluated for probe {probe_name}")
        best_by_probe[probe_name] = best_probe_payload

    probe_result_df = pd.DataFrame(probe_result_rows).sort_values(
        ["probe", "ari_delta_vs_xgb", "pairwise_f1_delta_vs_xgb", "seed_clusters"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    probe_result_df.to_csv(tables_dir / "probe_config_sweep_v1.csv", index=False)

    aggregate_df = (
        probe_result_df.groupby(
            ["anchor_threshold", "min_seed_score", "support_mode", "support_scale", "min_shared_affinity"],
            as_index=False,
        )
        .agg(
            probe_count=("probe", "nunique"),
            mean_ari=("ari", "mean"),
            mean_pairwise_f1=("pairwise_f1", "mean"),
            mean_ari_delta_vs_xgb=("ari_delta_vs_xgb", "mean"),
            min_ari_delta_vs_xgb=("ari_delta_vs_xgb", "min"),
            win_count=("ari_delta_vs_xgb", lambda values: int((pd.Series(values) > 0).sum())),
            mean_pairwise_f1_delta_vs_xgb=("pairwise_f1_delta_vs_xgb", "mean"),
            mean_seed_clusters=("seed_clusters", "mean"),
            mean_boosted_pairs=("boosted_pairs", "mean"),
        )
        .sort_values(
            ["mean_ari_delta_vs_xgb", "mean_pairwise_f1_delta_vs_xgb", "win_count", "mean_seed_clusters"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )
    aggregate_df.to_csv(tables_dir / "aggregate_summary_v1.csv", index=False)
    best_aggregate_row = aggregate_df.iloc[0].to_dict()
    matching_probe_rows = probe_result_df[
        probe_result_df["anchor_threshold"].astype(float).eq(float(best_aggregate_row["anchor_threshold"]))
        & probe_result_df["min_seed_score"].astype(float).eq(float(best_aggregate_row["min_seed_score"]))
        & probe_result_df["support_mode"].astype(str).eq(str(best_aggregate_row["support_mode"]))
        & probe_result_df["support_scale"].astype(float).eq(float(best_aggregate_row["support_scale"]))
        & probe_result_df["min_shared_affinity"].astype(float).eq(float(best_aggregate_row["min_shared_affinity"]))
    ].copy()
    if not matching_probe_rows.empty:
        chosen_threshold = float(
            matching_probe_rows["threshold"].value_counts().sort_values(ascending=False).index[0]
        )
    else:
        chosen_threshold = 0.25
    best_aggregate_row["chosen_threshold"] = chosen_threshold

    test_outputs: dict[str, str] = {}
    if args.export_test_override:
        xgb_route_dir = args.xgb_route_dir.resolve()
        test_df = pd.read_csv(xgb_route_dir / "embeddings" / "salamander_test_metadata.csv")
        test_df["image_id"] = test_df["image_id"].astype(str)
        test_embeddings = np.load(xgb_route_dir / "embeddings" / "salamander_route_test_embeddings.npy").astype(np.float32)
        test_pair_df = pd.read_csv(xgb_route_dir / "tables" / "test_pair_features_v1.csv")
        base_predictions_df = pd.read_csv(xgb_route_dir / "tables" / "test_predictions_v1.csv")
        base_predictions_df["image_id"] = base_predictions_df["image_id"].astype(str)
        base_predictions_df["dataset"] = base_predictions_df["dataset"].astype(str)

        test_base_score = cosine_score_matrix(test_embeddings)
        test_boosted_score = _apply_pair_probability_as_score(
            base_score=test_base_score,
            pair_df=test_pair_df,
            probability_col="xgb_same_identity_prob",
            blend_scale=float(args.blend_scale),
        )
        test_seed_bundle = build_stable_seed_bundle_from_score_matrix(
            target_df=test_df,
            score_matrix=test_boosted_score,
            anchor_threshold=float(best_aggregate_row["anchor_threshold"]),
            stability_delta=float(args.stability_delta),
            min_seed_cluster_size=int(args.min_seed_cluster_size),
            max_seed_cluster_size=int(args.max_seed_cluster_size),
            min_mean_score=float(best_aggregate_row["min_seed_score"]),
        )
        refined_test_score, support_stats = apply_seed_center_support(
            score_matrix=test_boosted_score,
            seed_assignment_df=test_seed_bundle.seed_assignment_df,
            support_scale=float(best_aggregate_row["support_scale"]),
            min_shared_affinity=float(best_aggregate_row["min_shared_affinity"]),
            mode=str(best_aggregate_row["support_mode"]),
        )
        test_sweep_df, _test_prediction_df = run_score_threshold_sweep(
            df=test_df,
            score_matrix=refined_test_score,
            thresholds=[float(value) for value in args.thresholds],
        )
        if chosen_threshold not in test_sweep_df["threshold"].astype(float).tolist():
            chosen_threshold = float(test_sweep_df.sort_values(["cluster_count", "threshold"], ascending=[True, True]).iloc[0]["threshold"])
        test_pred_df = _cluster_single_dataset_from_score_matrix(
            dataset_df=test_df,
            score_matrix=refined_test_score,
            threshold=chosen_threshold,
        )
        test_pred_df["route_name"] = "ft_miew_arcface_masked_supcon_v1_last_fusion_xgb_seed_support_v1"
        test_pred_df["embedding_dim"] = int(test_embeddings.shape[1])
        test_pred_df["rerank_enabled"] = True
        test_pred_df["local_weight"] = 0.0
        test_pred_df["chosen_threshold"] = chosen_threshold
        test_pred_df["pairwise_model"] = "xgboost_seed_support"
        test_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)
        test_sweep_df.to_csv(tables_dir / "salamander_test_threshold_sweep_v1.csv", index=False)
        test_seed_bundle.seed_assignment_df.to_csv(tables_dir / "salamander_test_seed_assignments_v1.csv", index=False)
        test_seed_bundle.cluster_summary_df.to_csv(tables_dir / "salamander_test_seed_clusters_v1.csv", index=False)

        merged_pred_df = pd.concat(
            [base_predictions_df[base_predictions_df["dataset"] != SALAMANDER_DATASET].copy(), test_pred_df],
            ignore_index=True,
        )
        merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
        build_submission(
            test_pred_df=merged_pred_df,
            sample_submission_path=args.sample_submission_path.resolve(),
            output_path=output_dir / "submission.csv",
        )
        test_outputs = {
            "submission_path": str((output_dir / "submission.csv").resolve()),
            "test_predictions_path": str((tables_dir / "test_predictions_v1.csv").resolve()),
            "salamander_override_path": str((tables_dir / "salamander_test_predictions_v1.csv").resolve()),
            "test_support_stats": json.dumps(support_stats, ensure_ascii=False),
        }

    summary_payload = {
        "probe_dirs": [str(path.resolve()) for path in args.probe_dirs],
        "best_by_probe": best_by_probe,
        "best_aggregate_row": best_aggregate_row,
        "test_outputs": test_outputs,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Seed-Center Transductive Probe",
        "",
        "## Aggregate Top-10",
        "",
        dataframe_to_markdown_table(aggregate_df.head(10)),
        "",
        "## Best By Probe",
        "",
        dataframe_to_markdown_table(pd.DataFrame(best_by_probe.values())),
        "",
        "## 方法解释",
        "",
        "- 先拿当前最强 `XGBoost pairwise fusion` 的 score matrix 作为 teacher graph。",
        "- 再在这个 graph 上抽稳定 seed cluster：只有在 `anchor ± delta` 都不变、而且簇大小和簇内平均分数都过线的 cluster，才会变成 seed。",
        "- 然后做 `seed-center support`：如果两张图都强烈指向同一个 seed center，就给它们的 pair score 一次额外支持，这相当于把 `close_to_same_seed_center` 真正落实到推理图上。",
        "- 这一步还是纯推理侧 transductive，不继续训练 backbone。",
        "",
        "## 当前最优聚合配置",
        "",
        dataframe_to_markdown_table(pd.DataFrame([best_aggregate_row])),
        "",
    ]
    if test_outputs:
        lines.extend(
            [
                "## Test Override",
                "",
                f"- Test override 已导出到 `{output_dir / 'submission.csv'}`。",
                f"- Salamander-only 覆盖表：`{tables_dir / 'salamander_test_predictions_v1.csv'}`。",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_transductive_seed_support_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_transductive_seed_support_probe] aggregate: {tables_dir / 'aggregate_summary_v1.csv'}")
    if test_outputs:
        print(f"[salamander_transductive_seed_support_probe] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
