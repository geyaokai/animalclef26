#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_XGB_VARIANT_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_graph_clustering_probe_v1")
DEFAULT_AVERAGE_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35]
DEFAULT_GRAPH_THRESHOLDS = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88]
DEFAULT_GRAPH_TOP_K_VALUES = [8, 10, 12]
SALAMANDER_DATASET = "SalamanderID2025"


def _load_route_bundle(route_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    test_df = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    for df in [val_df, test_df]:
        df["image_id"] = df["image_id"].astype(str)
        if "identity" in df.columns:
            df["identity"] = df["identity"].fillna("").astype(str)
    val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy").astype(np.float32)
    return val_df, val_embeddings, test_df, test_embeddings


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


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    ranking_columns = [column for column in ["ari", "pairwise_f1", "nmi", "threshold"] if column in df.columns]
    ascending = [False, False, False, True][: len(ranking_columns)]
    return df.sort_values(ranking_columns, ascending=ascending).iloc[0]


def _build_prediction_frame(df: pd.DataFrame, pred_labels: np.ndarray, *, method: str, threshold: float) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    result["clustering_method"] = str(method)
    result["chosen_threshold"] = float(threshold)
    result["pred_cluster_id"] = pred_labels.astype(int)
    result["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in pred_labels]
    return result


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table, summarize_cluster_metrics
    from animalclef_analysis.graph_clustering import GRAPH_METHODS, cluster_labels_from_score_graph, run_graph_threshold_sweep
    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.transductive_seed_refinement import run_score_threshold_sweep

    parser = argparse.ArgumentParser(description="Probe graph-based clustering on the current Salamander XGBoost score matrix.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--xgb-variant-dir", type=Path, default=DEFAULT_XGB_VARIANT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--average-thresholds", nargs="+", type=float, default=DEFAULT_AVERAGE_THRESHOLDS)
    parser.add_argument("--graph-thresholds", nargs="+", type=float, default=DEFAULT_GRAPH_THRESHOLDS)
    parser.add_argument("--graph-top-k-values", nargs="+", type=int, default=DEFAULT_GRAPH_TOP_K_VALUES)
    parser.add_argument("--graph-mutual-topk-options", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--methods", nargs="+", choices=GRAPH_METHODS, default=GRAPH_METHODS)
    parser.add_argument("--cw-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    route_dir = args.route_dir.resolve()
    xgb_variant_dir = args.xgb_variant_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_df, route_val_embeddings, test_df, route_test_embeddings = _load_route_bundle(route_dir=route_dir)
    route_val_score = cosine_score_matrix(route_val_embeddings)
    route_test_score = cosine_score_matrix(route_test_embeddings)

    val_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "val_pair_features_v1.csv")
    test_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "test_pair_features_v1.csv")
    boosted_val_score = _apply_pair_probability_as_score(
        base_score=route_val_score,
        pair_df=val_pair_df,
        probability_col="xgb_same_identity_prob",
        blend_scale=float(args.blend_scale),
    )
    boosted_test_score = _apply_pair_probability_as_score(
        base_score=route_test_score,
        pair_df=test_pair_df,
        probability_col="xgb_same_identity_prob",
        blend_scale=float(args.blend_scale),
    )

    average_sweep_df, _average_pred_df = run_score_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.average_thresholds],
        score_space="unit_interval",
    )
    average_sweep_df["clustering_method"] = "average_linkage"
    average_sweep_df["graph_top_k"] = -1
    average_sweep_df["mutual_top_k"] = False

    graph_rows: list[pd.DataFrame] = []
    graph_prediction_frames: list[pd.DataFrame] = []
    for method in [str(value) for value in args.methods]:
        for top_k in [int(value) for value in args.graph_top_k_values]:
            for mutual_option in [bool(int(value)) for value in args.graph_mutual_topk_options]:
                sweep_df, pred_df = run_graph_threshold_sweep(
                    df=val_df,
                    score_matrix=boosted_val_score,
                    thresholds=[float(value) for value in args.graph_thresholds],
                    method=method,
                    top_k=int(top_k),
                    mutual_top_k=bool(mutual_option),
                    iterations=int(args.cw_iterations),
                    seed=int(args.seed),
                )
                sweep_df = sweep_df.rename(columns={"method": "clustering_method", "top_k": "graph_top_k"})
                graph_rows.append(sweep_df)
                graph_prediction_frames.append(pred_df)
    graph_sweep_df = pd.concat(graph_rows, ignore_index=True) if graph_rows else pd.DataFrame()
    if graph_prediction_frames:
        pd.concat(graph_prediction_frames, ignore_index=True).to_csv(tables_dir / "val_graph_predictions_all_v1.csv", index=False)

    comparison_rows: list[dict[str, object]] = []
    average_best = _pick_best_row(average_sweep_df)
    comparison_rows.append(
        {
            "route": "xgb_average_linkage",
            "threshold": float(average_best["threshold"]),
            "clustering_method": "average_linkage",
            "graph_top_k": -1,
            "mutual_top_k": False,
            "ari": float(average_best["ari"]),
            "nmi": float(average_best["nmi"]),
            "pairwise_f1": float(average_best["pairwise_f1"]),
            "cluster_count": int(average_best["cluster_count"]),
        }
    )

    best_graph_row = None
    if not graph_sweep_df.empty:
        best_graph_row = _pick_best_row(graph_sweep_df)
        comparison_rows.append(
            {
                "route": "xgb_graph_clustering",
                "threshold": float(best_graph_row["threshold"]),
                "clustering_method": str(best_graph_row["clustering_method"]),
                "graph_top_k": int(best_graph_row["graph_top_k"]),
                "mutual_top_k": bool(best_graph_row["mutual_top_k"]),
                "ari": float(best_graph_row["ari"]),
                "nmi": float(best_graph_row["nmi"]),
                "pairwise_f1": float(best_graph_row["pairwise_f1"]),
                "cluster_count": int(best_graph_row["cluster_count"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    if len(comparison_df) >= 2:
        baseline_ari = float(comparison_df.iloc[0]["ari"])
        baseline_f1 = float(comparison_df.iloc[0]["pairwise_f1"])
        comparison_df["ari_delta_vs_average"] = np.round(comparison_df["ari"].astype(float) - baseline_ari, 6)
        comparison_df["pairwise_f1_delta_vs_average"] = np.round(comparison_df["pairwise_f1"].astype(float) - baseline_f1, 6)
    else:
        comparison_df["ari_delta_vs_average"] = 0.0
        comparison_df["pairwise_f1_delta_vs_average"] = 0.0
    comparison_df.to_csv(tables_dir / "comparison_summary_v1.csv", index=False)
    average_sweep_df.to_csv(tables_dir / "average_threshold_sweep_v1.csv", index=False)
    if not graph_sweep_df.empty:
        graph_sweep_df.to_csv(tables_dir / "graph_threshold_sweep_v1.csv", index=False)

    average_test_labels = None
    graph_test_labels = None
    if len(test_df):
        from animalclef_analysis.descriptor_baselines import build_average_linkage, cluster_from_linkage
        from animalclef_analysis.orb_rerank_baseline import score_matrix_to_distance

        avg_distance = score_matrix_to_distance(boosted_test_score)
        avg_linkage = build_average_linkage(avg_distance)
        average_test_labels = cluster_from_linkage(avg_linkage, len(test_df), float(average_best["threshold"]))
        average_test_pred_df = _build_prediction_frame(
            df=test_df,
            pred_labels=average_test_labels,
            method="average_linkage",
            threshold=float(average_best["threshold"]),
        )
        average_test_pred_df.to_csv(tables_dir / "test_average_predictions_v1.csv", index=False)
        if best_graph_row is not None:
            graph_test_labels = cluster_labels_from_score_graph(
                score_matrix=boosted_test_score,
                threshold=float(best_graph_row["threshold"]),
                method=str(best_graph_row["clustering_method"]),
                top_k=int(best_graph_row["graph_top_k"]),
                mutual_top_k=bool(best_graph_row["mutual_top_k"]),
                iterations=int(args.cw_iterations),
                seed=int(args.seed),
            )
            graph_test_pred_df = _build_prediction_frame(
                df=test_df,
                pred_labels=graph_test_labels,
                method=str(best_graph_row["clustering_method"]),
                threshold=float(best_graph_row["threshold"]),
            )
            graph_test_pred_df["graph_top_k"] = int(best_graph_row["graph_top_k"])
            graph_test_pred_df["mutual_top_k"] = bool(best_graph_row["mutual_top_k"])
            graph_test_pred_df.to_csv(tables_dir / "test_graph_predictions_v1.csv", index=False)

    test_summary_rows: list[dict[str, object]] = []
    if average_test_labels is not None:
        counts = pd.Series(average_test_labels).value_counts()
        test_summary_rows.append(
            {
                "route": "xgb_average_linkage",
                "clustering_method": "average_linkage",
                "threshold": float(average_best["threshold"]),
                "graph_top_k": -1,
                "mutual_top_k": False,
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
            }
        )
    if graph_test_labels is not None and best_graph_row is not None:
        counts = pd.Series(graph_test_labels).value_counts()
        test_summary_rows.append(
            {
                "route": "xgb_graph_clustering",
                "clustering_method": str(best_graph_row["clustering_method"]),
                "threshold": float(best_graph_row["threshold"]),
                "graph_top_k": int(best_graph_row["graph_top_k"]),
                "mutual_top_k": bool(best_graph_row["mutual_top_k"]),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
            }
        )
    test_summary_df = pd.DataFrame(test_summary_rows)
    if not test_summary_df.empty:
        test_summary_df.to_csv(tables_dir / "test_cluster_summary_v1.csv", index=False)

    summary = {
        "probe": "salamander_graph_clustering_probe",
        "route_dir": str(route_dir),
        "xgb_variant_dir": str(xgb_variant_dir),
        "blend_scale": float(args.blend_scale),
        "average_thresholds": [float(value) for value in args.average_thresholds],
        "graph_thresholds": [float(value) for value in args.graph_thresholds],
        "graph_top_k_values": [int(value) for value in args.graph_top_k_values],
        "graph_mutual_topk_options": [bool(int(value)) for value in args.graph_mutual_topk_options],
        "methods": [str(value) for value in args.methods],
        "comparison_rows": comparison_df.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Graph Clustering Probe",
        "",
        "- Goal: keep the current Salamander `XGBoost` pairwise score matrix fixed, and only replace the final clustering stage.",
        f"- Route dir: `{route_dir}`",
        f"- XGB score source: `{xgb_variant_dir}`",
        f"- Validation images: `{len(val_df)}`",
        f"- Test images: `{len(test_df)}`",
        f"- Blend scale used to rebuild the boosted score matrix: `{float(args.blend_scale)}`",
        "",
        "## Comparison",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
    ]
    if not test_summary_df.empty:
        lines.extend(
            [
                "## Test Cluster Summary",
                "",
                dataframe_to_markdown_table(test_summary_df),
                "",
            ]
        )
    lines.extend(
        [
            "## Reading Note",
            "",
            "- `xgb_average_linkage` reuses the current boosted score matrix and keeps the old average-linkage family.",
            "- `xgb_graph_clustering` keeps the same score matrix but switches the final cluster formation rule to a graph method.",
            "- If graph clustering wins here, it is a real inference-stage architecture change rather than another feature tweak.",
        ]
    )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[salamander_graph_clustering_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_graph_clustering_probe] comparison: {tables_dir / 'comparison_summary_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
