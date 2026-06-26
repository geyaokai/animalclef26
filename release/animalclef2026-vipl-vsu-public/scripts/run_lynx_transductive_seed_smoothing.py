#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


LYNX_DATASET = "LynxID2025"
DEFAULT_AUDIT_DIR = Path("artifacts/analysis/lynx_anchor_seed_audit_20260403")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/lynx_transductive_seed_smoothing_20260403")
DEFAULT_VAL_EMBEDDINGS = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_embeddings.npy")
DEFAULT_VAL_METADATA = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_metadata.csv")
DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_mixed_baseline_v2/tables/test_predictions_v1.csv")
DEFAULT_TEST_EMBEDDINGS = Path("artifacts/submissions/kaggle_mixed_baseline_v2/embeddings/lynx_ft_mega_test_embeddings.npy")
DEFAULT_TEST_METADATA = Path("artifacts/submissions/kaggle_mixed_baseline_v2/embeddings/lynx_ft_mega_test_metadata.csv")


def _load_cached_seed_assignments(audit_dir: Path, anchor: float) -> pd.DataFrame | None:
    path = audit_dir / "tables" / f"seed_assignments_anchor_{_threshold_tag(anchor)}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_cached_cluster_summary(audit_dir: Path, anchor: float) -> pd.DataFrame | None:
    path = audit_dir / "tables" / f"seed_clusters_anchor_{_threshold_tag(anchor)}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _threshold_tag(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _merge_seed_stats(seed_assignment_df: pd.DataFrame, cluster_summary_df: pd.DataFrame) -> dict[str, float | int]:
    accepted = cluster_summary_df[cluster_summary_df["accepted_as_seed"]].copy().reset_index(drop=True)
    seed_images = int(seed_assignment_df["seed_status"].astype(str).eq("seed").sum())
    target_size = int(len(seed_assignment_df))
    coverage = float(seed_images / target_size) if target_size else 0.0
    weighted_purity = 0.0
    mean_score = 0.0
    if not accepted.empty:
        weights = accepted["size"].astype(float)
        purity_col = accepted["purity_vs_truth"].astype(float).fillna(0.0)
        weighted_purity = float(np.average(purity_col.to_numpy(), weights=weights.to_numpy()))
        score_column = "mean_score" if "mean_score" in accepted.columns else "mean_similarity"
        mean_score = float(np.average(accepted[score_column].astype(float).to_numpy(), weights=weights.to_numpy()))
    return {
        "seed_images": seed_images,
        "seed_coverage_ratio": round(coverage, 6),
        "accepted_seed_clusters": int(len(accepted)),
        "weighted_seed_purity": round(weighted_purity, 6),
        "mean_seed_score": round(mean_score, 6),
    }


def _align_embeddings_to_reference(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
    candidate_df = candidate_df.copy().reset_index(drop=True)
    candidate_df["image_id"] = candidate_df["image_id"].astype(str)
    lookup = {str(row.image_id): index for index, row in enumerate(candidate_df.itertuples(index=False))}
    order = [lookup[str(image_id)] for image_id in reference_df["image_id"].astype(str).tolist()]
    return embeddings[order].astype(np.float32)


def _resolve_target_df(audit_dir: Path, val_df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    target_manifest_path = audit_dir / "tables" / "target_manifest_v1.csv"
    val_df = val_df.copy().reset_index(drop=True)
    val_df["image_id"] = val_df["image_id"].astype(str)
    if not target_manifest_path.exists():
        return val_df, False
    target_df = pd.read_csv(target_manifest_path)
    target_df["image_id"] = target_df["image_id"].astype(str)
    val_image_ids = set(val_df["image_id"].tolist())
    if set(target_df["image_id"].tolist()).issubset(val_image_ids):
        return target_df, True
    return val_df, False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table
    from animalclef_analysis.transductive_seed_refinement import (
        apply_seed_centroid_smoothing,
        build_stable_seed_bundle_from_score_matrix,
        cosine_similarity_matrix,
        pick_best_threshold_row,
        run_score_threshold_sweep,
    )

    parser = argparse.ArgumentParser(description="Run Lynx transductive seed-centroid smoothing and optionally export a test override.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--audit-dir", type=Path, default=repo_root / DEFAULT_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-embeddings-path", type=Path, default=repo_root / DEFAULT_VAL_EMBEDDINGS)
    parser.add_argument("--val-metadata-path", type=Path, default=repo_root / DEFAULT_VAL_METADATA)
    parser.add_argument("--anchors", nargs="+", type=float, default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.10, 0.15, 0.20, 0.25, 0.30])
    parser.add_argument("--eval-thresholds", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--stability-delta", type=float, default=0.03)
    parser.add_argument("--min-seed-cluster-size", type=int, default=2)
    parser.add_argument("--max-seed-cluster-size", type=int, default=12)
    parser.add_argument("--min-mean-score", type=float, default=0.0)
    parser.add_argument("--export-test-override", action="store_true")
    parser.add_argument("--base-predictions-path", type=Path, default=repo_root / DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--test-embeddings-path", type=Path, default=repo_root / DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--test-metadata-path", type=Path, default=repo_root / DEFAULT_TEST_METADATA)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--route-name", type=str, default="ft_mega_arcface_distill_v1_seed_smoothing_v1")
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    predictions_dir = output_dir / "predictions"
    for path in [output_dir, tables_dir, reports_dir, predictions_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_metadata_path.resolve())
    val_df = val_df[val_df["dataset"] == LYNX_DATASET].copy().reset_index(drop=True)
    target_df, use_cached_seed_tables = _resolve_target_df(audit_dir=audit_dir, val_df=val_df)
    val_embeddings_all = np.load(args.val_embeddings_path.resolve()).astype(np.float32)
    val_metadata_all = pd.read_csv(args.val_metadata_path.resolve())
    val_embeddings = val_embeddings_all[(val_metadata_all["dataset"] == LYNX_DATASET).to_numpy()]
    target_embeddings = _align_embeddings_to_reference(reference_df=target_df, candidate_df=val_df, embeddings=val_embeddings)
    base_score_matrix = cosine_similarity_matrix(target_embeddings)
    base_sweep_df, base_prediction_df = run_score_threshold_sweep(
        df=target_df,
        score_matrix=base_score_matrix,
        thresholds=[float(value) for value in args.eval_thresholds],
        score_space="cosine_similarity",
    )
    base_best_row = pick_best_threshold_row(base_sweep_df)
    base_sweep_df.to_csv(tables_dir / "baseline_threshold_sweep_v1.csv", index=False)
    base_prediction_df.to_csv(tables_dir / "baseline_predictions_v1.csv", index=False)

    result_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    best_sort_key: tuple[float, float, float] = (-1.0, -1.0, 1.0)
    for anchor in [float(value) for value in args.anchors]:
        seed_assignment_df = _load_cached_seed_assignments(audit_dir=audit_dir, anchor=anchor) if use_cached_seed_tables else None
        cluster_summary_df = _load_cached_cluster_summary(audit_dir=audit_dir, anchor=anchor) if use_cached_seed_tables else None
        if seed_assignment_df is None or cluster_summary_df is None:
            seed_bundle = build_stable_seed_bundle_from_score_matrix(
                target_df=target_df,
                score_matrix=base_score_matrix,
                anchor_threshold=float(anchor),
                stability_delta=float(args.stability_delta),
                min_seed_cluster_size=int(args.min_seed_cluster_size),
                max_seed_cluster_size=int(args.max_seed_cluster_size),
                min_mean_score=float(args.min_mean_score),
                score_space="cosine_similarity",
            )
            seed_assignment_df = seed_bundle.seed_assignment_df
            cluster_summary_df = seed_bundle.cluster_summary_df
        seed_assignment_df["image_id"] = seed_assignment_df["image_id"].astype(str)
        seed_stats = _merge_seed_stats(seed_assignment_df=seed_assignment_df, cluster_summary_df=cluster_summary_df)
        for alpha in [float(value) for value in args.alphas]:
            smoothed_embeddings = apply_seed_centroid_smoothing(
                embeddings=target_embeddings,
                seed_assignment_df=seed_assignment_df,
                alpha=float(alpha),
            )
            score_matrix = cosine_similarity_matrix(smoothed_embeddings)
            sweep_df, prediction_df = run_score_threshold_sweep(
                df=target_df,
                score_matrix=score_matrix,
                thresholds=[float(value) for value in args.eval_thresholds],
                score_space="cosine_similarity",
            )
            best_row = pick_best_threshold_row(sweep_df)
            row = {
                "anchor_threshold": float(anchor),
                "alpha": float(alpha),
                **seed_stats,
                "threshold": float(best_row["threshold"]),
                "ari": float(best_row["ari"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "nmi": float(best_row["nmi"]),
                "cluster_count": int(best_row["cluster_count"]),
                "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
                "ari_delta_vs_base": round(float(best_row["ari"]) - float(base_best_row["ari"]), 6),
                "pairwise_f1_delta_vs_base": round(float(best_row["pairwise_f1"]) - float(base_best_row["pairwise_f1"]), 6),
            }
            result_rows.append(row)
            sort_key = (float(best_row["ari"]), float(best_row["pairwise_f1"]), -float(best_row["threshold"]))
            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_payload = {
                    "row": row,
                    "seed_assignment_df": seed_assignment_df.copy(),
                    "cluster_summary_df": cluster_summary_df.copy(),
                    "sweep_df": sweep_df.copy(),
                    "prediction_df": prediction_df.copy(),
                    "smoothed_embeddings": smoothed_embeddings.copy(),
                }

    result_df = pd.DataFrame(result_rows).sort_values(
        ["ari", "pairwise_f1", "seed_coverage_ratio", "anchor_threshold", "alpha"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    result_df.to_csv(tables_dir / "config_sweep_v1.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No Lynx transductive configuration was evaluated.")

    best_row = dict(best_payload["row"])
    best_prediction_df = best_payload["prediction_df"]
    chosen_threshold = float(best_row["threshold"])
    chosen_pred_df = best_prediction_df[best_prediction_df["threshold"] == chosen_threshold].copy().reset_index(drop=True)
    chosen_pred_df["route_name"] = str(args.route_name)
    chosen_pred_df["embedding_dim"] = int(target_embeddings.shape[1])
    chosen_pred_df["rerank_enabled"] = False
    chosen_pred_df["local_weight"] = 0.0
    chosen_pred_df["chosen_threshold"] = chosen_threshold
    chosen_pred_df["cluster_label"] = [f"cluster_{LYNX_DATASET}_{int(label)}" for label in chosen_pred_df["pred_cluster_id"]]
    chosen_pred_df.to_csv(tables_dir / "best_predictions_v1.csv", index=False)
    best_payload["seed_assignment_df"].to_csv(tables_dir / "best_seed_assignments_v1.csv", index=False)
    best_payload["cluster_summary_df"].to_csv(tables_dir / "best_seed_clusters_v1.csv", index=False)
    best_payload["sweep_df"].to_csv(tables_dir / "best_threshold_sweep_v1.csv", index=False)
    np.save(predictions_dir / "best_smoothed_embeddings_v1.npy", best_payload["smoothed_embeddings"].astype(np.float32))

    test_outputs: dict[str, str] = {}
    if args.export_test_override:
        test_df = pd.read_csv(args.test_metadata_path.resolve())
        test_df["image_id"] = test_df["image_id"].astype(str)
        test_embeddings = np.load(args.test_embeddings_path.resolve()).astype(np.float32)
        test_score_matrix = cosine_similarity_matrix(test_embeddings)
        test_seed_bundle = build_stable_seed_bundle_from_score_matrix(
            target_df=test_df,
            score_matrix=test_score_matrix,
            anchor_threshold=float(best_row["anchor_threshold"]),
            stability_delta=float(args.stability_delta),
            min_seed_cluster_size=int(args.min_seed_cluster_size),
            max_seed_cluster_size=int(args.max_seed_cluster_size),
            min_mean_score=float(args.min_mean_score),
            score_space="cosine_similarity",
        )
        smoothed_test_embeddings = apply_seed_centroid_smoothing(
            embeddings=test_embeddings,
            seed_assignment_df=test_seed_bundle.seed_assignment_df,
            alpha=float(best_row["alpha"]),
        )
        smoothed_test_score = cosine_similarity_matrix(smoothed_test_embeddings)
        test_thresholds = sorted({max(0.05, chosen_threshold - 0.05), chosen_threshold, min(0.99, chosen_threshold + 0.05)})
        test_sweep_df, test_prediction_df = run_score_threshold_sweep(
            df=test_df,
            score_matrix=smoothed_test_score,
            thresholds=[float(value) for value in test_thresholds],
            score_space="cosine_similarity",
        )
        test_pred_df = test_prediction_df[test_prediction_df["threshold"] == chosen_threshold].copy().reset_index(drop=True)
        test_pred_df["route_name"] = str(args.route_name)
        test_pred_df["embedding_dim"] = int(test_embeddings.shape[1])
        test_pred_df["rerank_enabled"] = False
        test_pred_df["local_weight"] = 0.0
        test_pred_df["chosen_threshold"] = chosen_threshold
        test_pred_df["dataset"] = LYNX_DATASET
        test_pred_df["cluster_label"] = [f"cluster_{LYNX_DATASET}_{int(label)}" for label in test_pred_df["pred_cluster_id"]]
        test_pred_df.to_csv(tables_dir / "lynx_test_predictions_v1.csv", index=False)
        test_sweep_df.to_csv(tables_dir / "lynx_test_threshold_sweep_v1.csv", index=False)
        test_seed_bundle.seed_assignment_df.to_csv(tables_dir / "lynx_test_seed_assignments_v1.csv", index=False)
        test_seed_bundle.cluster_summary_df.to_csv(tables_dir / "lynx_test_seed_clusters_v1.csv", index=False)
        np.save(predictions_dir / "lynx_test_smoothed_embeddings_v1.npy", smoothed_test_embeddings.astype(np.float32))

        base_predictions_df = pd.read_csv(args.base_predictions_path.resolve())
        base_predictions_df["image_id"] = base_predictions_df["image_id"].astype(str)
        base_predictions_df["dataset"] = base_predictions_df["dataset"].astype(str)
        merged_pred_df = pd.concat(
            [
                base_predictions_df[base_predictions_df["dataset"] != LYNX_DATASET].copy(),
                test_pred_df,
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
        "audit_dir": str(audit_dir),
        "base_best": {
            "threshold": float(base_best_row["threshold"]),
            "ari": float(base_best_row["ari"]),
            "pairwise_f1": float(base_best_row["pairwise_f1"]),
            "cluster_count": int(base_best_row["cluster_count"]),
        },
        "chosen_config": best_row,
        "test_outputs": test_outputs,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Lynx Test-Aware Seed Smoothing",
        "",
        "## Baseline",
        "",
        dataframe_to_markdown_table(pd.DataFrame([base_best_row])),
        "",
        "## Config Sweep Top-10",
        "",
        dataframe_to_markdown_table(result_df.head(10)),
        "",
        "## Chosen Config",
        "",
        dataframe_to_markdown_table(pd.DataFrame([best_row])),
        "",
        "## 解释",
        "",
        "- 这一步不继续训练 backbone，只在 target split 上先抽稳定 seed cluster，再把 seed 图的 embedding 轻微拉向各自 seed 质心。",
        "- `anchor_threshold` 控制 seed 提取时的聚类松紧；`alpha` 控制回写强度；最后还会重新 sweep clustering threshold。",
        "- 这个方法本质上是推理侧 transductive：它利用 test/target 的整体结构，但不改模型参数。",
        "",
        "## 结果概览",
        "",
        f"- Baseline best: `ARI {float(base_best_row['ari']):.6f}`, `pairwise_f1 {float(base_best_row['pairwise_f1']):.6f}`, threshold `{float(base_best_row['threshold'])}`。",
        f"- Chosen smoothing: `anchor {float(best_row['anchor_threshold'])}`, `alpha {float(best_row['alpha'])}`, `threshold {float(best_row['threshold'])}`, `ARI {float(best_row['ari']):.6f}`, `pairwise_f1 {float(best_row['pairwise_f1']):.6f}`。",
        f"- Delta vs baseline: `ARI {float(best_row['ari_delta_vs_base']):+.6f}`, `pairwise_f1 {float(best_row['pairwise_f1_delta_vs_base']):+.6f}`。",
        "",
    ]
    if test_outputs:
        lines.extend(
            [
                "## Test Override",
                "",
                f"- Test override 已导出，沿用本地最优配置：`anchor {float(best_row['anchor_threshold'])}`, `alpha {float(best_row['alpha'])}`, `threshold {float(best_row['threshold'])}`。",
                f"- 合并后的 submission: `{output_dir / 'submission.csv'}`。",
                f"- Lynx-only 覆盖表: `{tables_dir / 'lynx_test_predictions_v1.csv'}`。",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[lynx_transductive_seed_smoothing] summary: {reports_dir / 'summary.md'}")
    print(f"[lynx_transductive_seed_smoothing] config_sweep: {tables_dir / 'config_sweep_v1.csv'}")
    if test_outputs:
        print(f"[lynx_transductive_seed_smoothing] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
