#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LYNX_DATASET = "LynxID2025"
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/lynx_transductive_hubness_probe_20260404")
DEFAULT_VAL_EMBEDDINGS = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_embeddings.npy")
DEFAULT_VAL_METADATA = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_metadata.csv")
DEFAULT_TEST_EMBEDDINGS = Path("artifacts/submissions/kaggle_mixed_baseline_v2/embeddings/lynx_ft_mega_test_embeddings.npy")
DEFAULT_TEST_METADATA = Path("artifacts/submissions/kaggle_mixed_baseline_v2/embeddings/lynx_ft_mega_test_metadata.csv")
DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_variant_lynx_ftmega_seedsmooth_onxgb_v1/tables/test_predictions_v1.csv")


def _cluster_stats(prediction_df: pd.DataFrame) -> tuple[int, int, float]:
    counts = prediction_df["pred_cluster_id"].value_counts()
    cluster_count = int(counts.size)
    singleton_count = int((counts == 1).sum()) if cluster_count else 0
    singleton_ratio = float(singleton_count / cluster_count) if cluster_count else 0.0
    return cluster_count, singleton_count, singleton_ratio


def _sort_and_pick_best(result_df: pd.DataFrame, *, test_cluster_min: int | None, test_cluster_max: int | None, test_cluster_center: float | None) -> pd.DataFrame:
    ranked = result_df.copy()
    if test_cluster_min is not None and test_cluster_max is not None and "test_cluster_count" in ranked.columns:
        ranked["in_target_band"] = ranked["test_cluster_count"].between(int(test_cluster_min), int(test_cluster_max))
    else:
        ranked["in_target_band"] = True
    if test_cluster_center is not None and "test_cluster_count" in ranked.columns:
        ranked["test_cluster_center_abs_error"] = (ranked["test_cluster_count"] - float(test_cluster_center)).abs()
    else:
        ranked["test_cluster_center_abs_error"] = 0.0
    return ranked.sort_values(
        ["in_target_band", "ari", "pairwise_f1", "test_cluster_center_abs_error", "threshold", "hubness_top_k", "penalty_scale"],
        ascending=[False, False, False, True, True, True, True],
    ).reset_index(drop=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table
    from animalclef_analysis.transductive_seed_refinement import (
        apply_reverse_neighbor_penalty,
        cosine_similarity_matrix,
        pick_best_threshold_row,
        run_score_threshold_sweep,
    )

    parser = argparse.ArgumentParser(description="Run a Lynx transductive hubness penalty probe and optionally export a test override.")
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-embeddings-path", type=Path, default=repo_root / DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--val-metadata-path", type=Path, default=repo_root / DEFAULT_VAL_METADATA)
    parser.add_argument("--test-embeddings-path", type=Path, default=repo_root / DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--test-metadata-path", type=Path, default=repo_root / DEFAULT_TEST_METADATA)
    parser.add_argument("--base-predictions-path", type=Path, default=repo_root / DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--hubness-top-ks", nargs="+", type=int, default=[5, 8, 10, 15])
    parser.add_argument("--penalty-scales", nargs="+", type=float, default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05])
    parser.add_argument("--eval-thresholds", nargs="+", type=float, default=[0.775, 0.80, 0.825, 0.85, 0.875, 0.90])
    parser.add_argument("--test-cluster-min", type=int, default=22)
    parser.add_argument("--test-cluster-max", type=int, default=26)
    parser.add_argument("--test-cluster-center", type=float, default=24.0)
    parser.add_argument("--export-test-override", action="store_true")
    parser.add_argument("--route-name", type=str, default="ft_mega_arcface_distill_v1_hubness_penalty_v1")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_meta_all = pd.read_csv(args.val_metadata_path.resolve())
    val_meta_all["image_id"] = val_meta_all["image_id"].astype(str)
    val_df = val_meta_all[val_meta_all["dataset"] == LYNX_DATASET].copy().reset_index(drop=True)
    val_embeddings_all = np.load(args.val_embeddings_path.resolve()).astype(np.float32)
    val_embeddings = val_embeddings_all[(val_meta_all["dataset"] == LYNX_DATASET).to_numpy()]
    base_val_score = cosine_similarity_matrix(val_embeddings)
    base_sweep_df, base_prediction_df = run_score_threshold_sweep(
        df=val_df,
        score_matrix=base_val_score,
        thresholds=[float(value) for value in args.eval_thresholds],
        score_space="cosine_similarity",
    )
    base_best_row = pick_best_threshold_row(base_sweep_df)
    base_sweep_df.to_csv(tables_dir / "baseline_threshold_sweep_v1.csv", index=False)
    base_prediction_df.to_csv(tables_dir / "baseline_predictions_v1.csv", index=False)

    test_df = pd.read_csv(args.test_metadata_path.resolve())
    test_df["image_id"] = test_df["image_id"].astype(str)
    test_embeddings = np.load(args.test_embeddings_path.resolve()).astype(np.float32)
    base_test_score = cosine_similarity_matrix(test_embeddings)

    result_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    for top_k in [int(value) for value in args.hubness_top_ks]:
        for penalty_scale in [float(value) for value in args.penalty_scales]:
            penalized_val_score, hub_diag = apply_reverse_neighbor_penalty(
                base_val_score,
                top_k=top_k,
                penalty_scale=penalty_scale,
            )
            val_sweep_df, val_prediction_df = run_score_threshold_sweep(
                df=val_df,
                score_matrix=penalized_val_score,
                thresholds=[float(value) for value in args.eval_thresholds],
                score_space="cosine_similarity",
            )
            best_row = pick_best_threshold_row(val_sweep_df)
            chosen_threshold = float(best_row["threshold"])

            penalized_test_score, test_hub_diag = apply_reverse_neighbor_penalty(
                base_test_score,
                top_k=top_k,
                penalty_scale=penalty_scale,
            )
            test_sweep_df, test_prediction_df = run_score_threshold_sweep(
                df=test_df,
                score_matrix=penalized_test_score,
                thresholds=[chosen_threshold],
                score_space="cosine_similarity",
            )
            chosen_test_pred_df = test_prediction_df[test_prediction_df["threshold"] == chosen_threshold].copy().reset_index(drop=True)
            test_cluster_count, test_singletons, test_singleton_ratio = _cluster_stats(chosen_test_pred_df)
            row = {
                "hubness_top_k": int(top_k),
                "penalty_scale": float(penalty_scale),
                "threshold": chosen_threshold,
                "ari": float(best_row["ari"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "nmi": float(best_row["nmi"]),
                "val_cluster_count": int(best_row["cluster_count"]),
                "val_singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
                "ari_delta_vs_base": round(float(best_row["ari"]) - float(base_best_row["ari"]), 6),
                "pairwise_f1_delta_vs_base": round(float(best_row["pairwise_f1"]) - float(base_best_row["pairwise_f1"]), 6),
                "mean_reverse_neighbor_count": float(hub_diag["mean_reverse_neighbor_count"]),
                "p95_reverse_neighbor_count": float(hub_diag["p95_reverse_neighbor_count"]),
                "hub_image_ratio": float(hub_diag["hub_image_ratio"]),
                "test_mean_reverse_neighbor_count": float(test_hub_diag["mean_reverse_neighbor_count"]),
                "test_p95_reverse_neighbor_count": float(test_hub_diag["p95_reverse_neighbor_count"]),
                "test_hub_image_ratio": float(test_hub_diag["hub_image_ratio"]),
                "test_cluster_count": int(test_cluster_count),
                "test_singleton_clusters": int(test_singletons),
                "test_singleton_ratio": round(float(test_singleton_ratio), 6),
            }
            result_rows.append(row)
            payload = {
                "row": row,
                "val_sweep_df": val_sweep_df.copy(),
                "val_prediction_df": val_prediction_df.copy(),
                "test_prediction_df": chosen_test_pred_df.copy(),
                "test_sweep_df": test_sweep_df.copy(),
            }
            if best_payload is None:
                best_payload = payload
            else:
                ranked = _sort_and_pick_best(
                    pd.DataFrame([best_payload["row"], row]),
                    test_cluster_min=args.test_cluster_min,
                    test_cluster_max=args.test_cluster_max,
                    test_cluster_center=args.test_cluster_center,
                )
                if ranked.iloc[0]["hubness_top_k"] == int(top_k) and float(ranked.iloc[0]["penalty_scale"]) == float(penalty_scale):
                    best_payload = payload

    result_df = pd.DataFrame(result_rows)
    ranked_df = _sort_and_pick_best(
        result_df=result_df,
        test_cluster_min=args.test_cluster_min,
        test_cluster_max=args.test_cluster_max,
        test_cluster_center=args.test_cluster_center,
    )
    ranked_df.to_csv(tables_dir / "config_sweep_v1.csv", index=False)

    if best_payload is None:
        raise RuntimeError("No hubness configuration was evaluated.")

    best_row = dict(best_payload["row"])
    best_val_sweep_df = best_payload["val_sweep_df"]
    best_val_prediction_df = best_payload["val_prediction_df"]
    best_test_pred_df = best_payload["test_prediction_df"]
    best_test_sweep_df = best_payload["test_sweep_df"]
    chosen_threshold = float(best_row["threshold"])

    best_val_prediction_df.to_csv(tables_dir / "best_val_predictions_v1.csv", index=False)
    best_val_sweep_df.to_csv(tables_dir / "best_val_threshold_sweep_v1.csv", index=False)
    best_test_sweep_df.to_csv(tables_dir / "best_test_threshold_sweep_v1.csv", index=False)

    test_outputs: dict[str, str] = {}
    if args.export_test_override:
        export_df = best_test_pred_df.copy()
        export_df["route_name"] = str(args.route_name)
        export_df["embedding_dim"] = int(test_embeddings.shape[1])
        export_df["rerank_enabled"] = False
        export_df["local_weight"] = 0.0
        export_df["chosen_threshold"] = chosen_threshold
        export_df["dataset"] = LYNX_DATASET
        export_df["cluster_label"] = [f"cluster_{LYNX_DATASET}_{int(label)}" for label in export_df["pred_cluster_id"]]
        export_df.to_csv(tables_dir / "lynx_test_predictions_v1.csv", index=False)

        base_predictions_df = pd.read_csv(args.base_predictions_path.resolve())
        base_predictions_df["image_id"] = base_predictions_df["image_id"].astype(str)
        base_predictions_df["dataset"] = base_predictions_df["dataset"].astype(str)
        merged_pred_df = pd.concat(
            [
                base_predictions_df[base_predictions_df["dataset"] != LYNX_DATASET].copy(),
                export_df,
            ],
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
            "lynx_override_path": str((tables_dir / "lynx_test_predictions_v1.csv").resolve()),
        }

    summary_payload = {
        "base_best": {
            "threshold": float(base_best_row["threshold"]),
            "ari": float(base_best_row["ari"]),
            "pairwise_f1": float(base_best_row["pairwise_f1"]),
            "cluster_count": int(base_best_row["cluster_count"]),
        },
        "chosen_config": best_row,
        "selection_rule": {
            "test_cluster_min": int(args.test_cluster_min) if args.test_cluster_min is not None else None,
            "test_cluster_max": int(args.test_cluster_max) if args.test_cluster_max is not None else None,
            "test_cluster_center": float(args.test_cluster_center) if args.test_cluster_center is not None else None,
        },
        "test_outputs": test_outputs,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    top10_df = ranked_df.head(10).copy()
    report_columns = [
        "hubness_top_k",
        "penalty_scale",
        "threshold",
        "ari",
        "pairwise_f1",
        "ari_delta_vs_base",
        "pairwise_f1_delta_vs_base",
        "val_cluster_count",
        "test_cluster_count",
        "test_singleton_ratio",
        "in_target_band",
        "test_cluster_center_abs_error",
    ]
    lines = [
        "# Lynx Test-Aware Hubness Probe",
        "",
        "## Baseline",
        "",
        dataframe_to_markdown_table(pd.DataFrame([base_best_row])),
        "",
        "## Config Sweep Top-10",
        "",
        dataframe_to_markdown_table(top10_df[report_columns]),
        "",
        "## Chosen Config",
        "",
        dataframe_to_markdown_table(pd.DataFrame([best_row])),
        "",
        "## 解释",
        "",
        "- 这一步不改 backbone，也不引入训练，只在 target split 的相似度矩阵上做 `hubness` 校正。",
        "- 直觉是：如果一张图总是频繁出现在别人的 top-k 邻居里，它更像一个会把不同个体错误吸进来的 `hub`，应该在聚类前先把它相关边轻微压低。",
        "- `hubness_top_k` 定义 reverse-neighbor 统计窗口；`penalty_scale` 定义压边强度；最后仍然重新 sweep clustering threshold。",
        "- 选配置时不只看本地 `ARI / pairwise F1`，还要求 test cluster count 落在 `Lynx` 的高置信带宽附近。",
        "",
        "## 结果概览",
        "",
        f"- Baseline best: `ARI {float(base_best_row['ari']):.6f}`, `pairwise_f1 {float(base_best_row['pairwise_f1']):.6f}`, threshold `{float(base_best_row['threshold'])}`。",
        f"- Chosen hubness config: `top_k {int(best_row['hubness_top_k'])}`, `penalty_scale {float(best_row['penalty_scale'])}`, `threshold {float(best_row['threshold'])}`。",
        f"- Local delta vs baseline: `ARI {float(best_row['ari_delta_vs_base']):+.6f}`, `pairwise_f1 {float(best_row['pairwise_f1_delta_vs_base']):+.6f}`。",
        f"- Test shape under chosen config: `clusters {int(best_row['test_cluster_count'])}`, `singleton_ratio {float(best_row['test_singleton_ratio']):.6f}`.",
        "",
    ]
    if test_outputs:
        lines.extend(
            [
                "## Test Override",
                "",
                f"- 合并后的 submission: `{output_dir / 'submission.csv'}`。",
                f"- Lynx-only 覆盖表: `{tables_dir / 'lynx_test_predictions_v1.csv'}`。",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[lynx_transductive_hubness_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[lynx_transductive_hubness_probe] config_sweep: {tables_dir / 'config_sweep_v1.csv'}")
    if test_outputs:
        print(f"[lynx_transductive_hubness_probe] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
