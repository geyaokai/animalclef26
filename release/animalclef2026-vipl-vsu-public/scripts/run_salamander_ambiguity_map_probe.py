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
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_ambiguity_map_probe_v1")
DEFAULT_AVERAGE_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35]
DEFAULT_GRAPH_THRESHOLDS = [0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88]
DEFAULT_GRAPH_TOP_K_VALUES = [8, 10, 12]
DEFAULT_DBSCAN_MIN_SAMPLES = [2, 3]
DEFAULT_REPORT_TOP_K = 10
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


def _ensure_prediction_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in [
        ("graph_top_k", -1),
        ("mutual_top_k", False),
        ("min_samples", -1),
        ("min_link_score", np.nan),
        ("eps", np.nan),
    ]:
        if column not in result.columns:
            result[column] = value
    return result


def _ensure_summary_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, value in [
        ("graph_top_k", -1),
        ("mutual_top_k", False),
        ("min_samples", -1),
        ("min_link_score", np.nan),
        ("eps", np.nan),
    ]:
        if column not in result.columns:
            result[column] = value
    return result


def _build_prediction_frame(
    df: pd.DataFrame,
    pred_labels: np.ndarray,
    *,
    method: str,
    threshold: float,
    config: dict[str, object],
) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    frame["method"] = str(method)
    frame["threshold"] = float(threshold)
    frame["pred_cluster_id"] = np.asarray(pred_labels, dtype=np.int32)
    frame["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in pred_labels]
    for key, value in config.items():
        frame[key] = value
    return frame


def _cluster_labels_from_best_row(
    score_matrix: np.ndarray,
    row: pd.Series,
    *,
    cw_iterations: int,
    seed: int,
) -> np.ndarray:
    from animalclef_analysis.ambiguity_clustering import (
        AMBIGUITY_METHOD_AVERAGE,
        AMBIGUITY_METHOD_CHINESE_WHISPERS,
        AMBIGUITY_METHOD_DBSCAN,
        AMBIGUITY_METHOD_FINCH_LIKE,
        cluster_labels_from_dbscan_score_matrix,
        cluster_labels_from_finch_like_score_matrix,
    )
    from animalclef_analysis.graph_clustering import cluster_labels_from_score_graph
    from animalclef_analysis.transductive_seed_refinement import cluster_labels_from_score_matrix

    method = str(row["method"])
    threshold = float(row["threshold"])
    if method == AMBIGUITY_METHOD_AVERAGE:
        return cluster_labels_from_score_matrix(score_matrix=score_matrix, threshold=threshold, score_space="unit_interval")
    if method == AMBIGUITY_METHOD_CHINESE_WHISPERS:
        top_k = int(row["graph_top_k"]) if "graph_top_k" in row and int(row["graph_top_k"]) >= 0 else None
        mutual_top_k = bool(row["mutual_top_k"]) if "mutual_top_k" in row else False
        return cluster_labels_from_score_graph(
            score_matrix=score_matrix,
            threshold=threshold,
            method=AMBIGUITY_METHOD_CHINESE_WHISPERS,
            top_k=top_k,
            mutual_top_k=mutual_top_k,
            iterations=int(cw_iterations),
            seed=int(seed),
        )
    if method == AMBIGUITY_METHOD_DBSCAN:
        min_samples = int(row["min_samples"]) if "min_samples" in row else 2
        return cluster_labels_from_dbscan_score_matrix(
            score_matrix=score_matrix,
            threshold=threshold,
            min_samples=min_samples,
        )
    if method == AMBIGUITY_METHOD_FINCH_LIKE:
        min_link_score = float(row["min_link_score"]) if "min_link_score" in row and pd.notna(row["min_link_score"]) else threshold
        return cluster_labels_from_finch_like_score_matrix(
            score_matrix=score_matrix,
            min_link_score=min_link_score,
        )
    raise ValueError(f"Unsupported method row: {method}")


def _format_table_or_note(frame: pd.DataFrame, *, note: str, limit: int) -> str:
    if frame.empty:
        return note
    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table

    return dataframe_to_markdown_table(frame.head(int(limit)))


def _build_pair_vote_summary(
    pair_df: pd.DataFrame,
    *,
    label_map: dict[str, np.ndarray],
    score_matrix: np.ndarray,
    probability_col: str = "xgb_same_identity_prob",
) -> pd.DataFrame:
    columns = [
        "left_index",
        "right_index",
        "image_id",
        "neighbor_image_id",
        "merge_votes",
        "split_votes",
        "total_votes",
        "vote_ratio",
        "split_ratio",
        "pair_score",
        "conflict_methods",
    ]
    if probability_col in pair_df.columns:
        columns.insert(-1, probability_col)
    if pair_df.empty:
        return pd.DataFrame(columns=columns)

    frame = pair_df[["left_index", "right_index", "image_id", "neighbor_image_id"]].copy().reset_index(drop=True)
    left_index = frame["left_index"].to_numpy(dtype=int)
    right_index = frame["right_index"].to_numpy(dtype=int)
    method_names = [str(name) for name in label_map.keys()]
    same_votes_by_method: list[np.ndarray] = []
    for method_name in method_names:
        labels = np.asarray(label_map[method_name], dtype=np.int32)
        if labels.ndim != 1:
            raise ValueError(f"Labels for method {method_name} must be 1D.")
        same_votes_by_method.append(labels[left_index] == labels[right_index])
    same_matrix = (
        np.column_stack(same_votes_by_method)
        if same_votes_by_method
        else np.zeros((len(frame), 0), dtype=bool)
    )
    total_votes = np.full(len(frame), len(method_names), dtype=np.int32)
    merge_votes = same_matrix.sum(axis=1).astype(np.int32) if len(method_names) else np.zeros(len(frame), dtype=np.int32)
    split_votes = (total_votes - merge_votes).astype(np.int32)

    frame["merge_votes"] = merge_votes
    frame["split_votes"] = split_votes
    frame["total_votes"] = total_votes
    frame["vote_ratio"] = np.round(
        merge_votes.astype(np.float64) / np.maximum(total_votes, 1),
        6,
    )
    frame["split_ratio"] = np.round(
        split_votes.astype(np.float64) / np.maximum(total_votes, 1),
        6,
    )
    frame["pair_score"] = np.round(score_matrix[left_index, right_index].astype(np.float64), 6)
    if probability_col in pair_df.columns:
        frame[probability_col] = np.round(pair_df[probability_col].astype(float).to_numpy(), 6)

    conflict_methods: list[str] = []
    for row_index in range(len(frame)):
        if merge_votes[row_index] > split_votes[row_index]:
            disagree_mask = ~same_matrix[row_index]
        elif split_votes[row_index] > merge_votes[row_index]:
            disagree_mask = same_matrix[row_index]
        else:
            disagree_mask = np.ones(len(method_names), dtype=bool)
        conflict_methods.append(
            "|".join(
                method_name
                for method_name, disagree in zip(method_names, disagree_mask, strict=True)
                if disagree
            )
        )
    frame["conflict_methods"] = conflict_methods
    return frame.sort_values(["left_index", "right_index"], ascending=[True, True]).reset_index(drop=True)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.ambiguity_clustering import (
        AMBIGUITY_METHOD_AVERAGE,
        AMBIGUITY_METHOD_CHINESE_WHISPERS,
        AMBIGUITY_METHOD_DBSCAN,
        AMBIGUITY_METHOD_FINCH_LIKE,
        assign_ambiguity_components,
        build_pair_disagreement_table,
        build_partition_agreement_table,
        run_dbscan_threshold_sweep,
        run_finch_like_threshold_sweep,
        summarize_merge_candidates,
        summarize_split_candidates,
    )
    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table
    from animalclef_analysis.graph_clustering import run_graph_threshold_sweep
    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.texas_unsupervised import summarize_cluster_labels
    from animalclef_analysis.transductive_seed_refinement import run_score_threshold_sweep

    parser = argparse.ArgumentParser(
        description="Probe Salamander ambiguity zones by comparing multiple clustering views on the current XGBoost score graph."
    )
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--xgb-variant-dir", type=Path, default=DEFAULT_XGB_VARIANT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--average-thresholds", nargs="+", type=float, default=DEFAULT_AVERAGE_THRESHOLDS)
    parser.add_argument("--graph-thresholds", nargs="+", type=float, default=DEFAULT_GRAPH_THRESHOLDS)
    parser.add_argument("--graph-top-k-values", nargs="+", type=int, default=DEFAULT_GRAPH_TOP_K_VALUES)
    parser.add_argument("--graph-mutual-topk-options", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--dbscan-thresholds", nargs="+", type=float, default=DEFAULT_GRAPH_THRESHOLDS)
    parser.add_argument("--dbscan-min-samples", nargs="+", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES)
    parser.add_argument("--finch-thresholds", nargs="+", type=float, default=DEFAULT_GRAPH_THRESHOLDS)
    parser.add_argument("--cw-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-ambiguity-score", type=float, default=0.58)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.34)
    parser.add_argument("--min-merge-votes", type=int, default=2)
    parser.add_argument("--min-split-votes", type=int, default=2)
    parser.add_argument("--report-top-k", type=int, default=DEFAULT_REPORT_TOP_K)
    args = parser.parse_args()

    route_dir = (repo_root / args.route_dir).resolve() if not args.route_dir.is_absolute() else args.route_dir.resolve()
    xgb_variant_dir = (
        (repo_root / args.xgb_variant_dir).resolve()
        if not args.xgb_variant_dir.is_absolute()
        else args.xgb_variant_dir.resolve()
    )
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_df, route_val_embeddings, test_df, route_test_embeddings = _load_route_bundle(route_dir=route_dir)
    route_val_score = cosine_score_matrix(route_val_embeddings)
    route_test_score = cosine_score_matrix(route_test_embeddings)

    val_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "val_pair_features_v1.csv")
    test_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "test_pair_features_v1.csv")
    val_pair_df["image_id"] = val_pair_df["image_id"].astype(str)
    val_pair_df["neighbor_image_id"] = val_pair_df["neighbor_image_id"].astype(str)
    test_pair_df["image_id"] = test_pair_df["image_id"].astype(str)
    test_pair_df["neighbor_image_id"] = test_pair_df["neighbor_image_id"].astype(str)

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

    average_sweep_df, average_prediction_df = run_score_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.average_thresholds],
        score_space="unit_interval",
    )
    average_sweep_df["method"] = AMBIGUITY_METHOD_AVERAGE
    average_sweep_df = _ensure_summary_columns(average_sweep_df)
    average_prediction_df["method"] = AMBIGUITY_METHOD_AVERAGE
    average_prediction_df = _ensure_prediction_columns(average_prediction_df)

    graph_rows: list[pd.DataFrame] = []
    graph_prediction_frames: list[pd.DataFrame] = []
    for top_k in [int(value) for value in args.graph_top_k_values]:
        for mutual_option in [bool(int(value)) for value in args.graph_mutual_topk_options]:
            sweep_df, prediction_df = run_graph_threshold_sweep(
                df=val_df,
                score_matrix=boosted_val_score,
                thresholds=[float(value) for value in args.graph_thresholds],
                method=AMBIGUITY_METHOD_CHINESE_WHISPERS,
                top_k=top_k,
                mutual_top_k=mutual_option,
                iterations=int(args.cw_iterations),
                seed=int(args.seed),
            )
            sweep_df = sweep_df.rename(columns={"top_k": "graph_top_k"})
            sweep_df["method"] = AMBIGUITY_METHOD_CHINESE_WHISPERS
            sweep_df = _ensure_summary_columns(sweep_df)
            graph_rows.append(sweep_df)

            prediction_df = prediction_df.rename(columns={"top_k": "graph_top_k"})
            prediction_df["method"] = AMBIGUITY_METHOD_CHINESE_WHISPERS
            prediction_df = _ensure_prediction_columns(prediction_df)
            graph_prediction_frames.append(prediction_df)
    graph_sweep_df = pd.concat(graph_rows, ignore_index=True) if graph_rows else pd.DataFrame()
    graph_prediction_df = pd.concat(graph_prediction_frames, ignore_index=True) if graph_prediction_frames else pd.DataFrame()

    dbscan_sweep_df, dbscan_prediction_df = run_dbscan_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.dbscan_thresholds],
        min_samples_values=[int(value) for value in args.dbscan_min_samples],
    )
    dbscan_sweep_df = _ensure_summary_columns(dbscan_sweep_df)
    dbscan_prediction_df = _ensure_prediction_columns(dbscan_prediction_df)

    finch_sweep_df, finch_prediction_df = run_finch_like_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        min_link_scores=[float(value) for value in args.finch_thresholds],
    )
    finch_sweep_df = _ensure_summary_columns(finch_sweep_df)
    finch_prediction_df = _ensure_prediction_columns(finch_prediction_df)

    val_sweep_df = pd.concat(
        [
            average_sweep_df,
            graph_sweep_df,
            dbscan_sweep_df,
            finch_sweep_df,
        ],
        ignore_index=True,
    )
    val_predictions_all_df = pd.concat(
        [
            average_prediction_df,
            graph_prediction_df,
            dbscan_prediction_df,
            finch_prediction_df,
        ],
        ignore_index=True,
    )
    val_sweep_df.to_csv(tables_dir / "val_method_sweep_v1.csv", index=False)
    val_predictions_all_df.to_csv(tables_dir / "val_predictions_all_v1.csv", index=False)

    best_rows: list[pd.Series] = []
    for method_name in [
        AMBIGUITY_METHOD_AVERAGE,
        AMBIGUITY_METHOD_CHINESE_WHISPERS,
        AMBIGUITY_METHOD_DBSCAN,
        AMBIGUITY_METHOD_FINCH_LIKE,
    ]:
        method_df = val_sweep_df[val_sweep_df["method"].astype(str).eq(method_name)].copy()
        if method_df.empty:
            continue
        best_rows.append(_pick_best_row(method_df))
    comparison_df = pd.DataFrame(best_rows).reset_index(drop=True)
    average_best_row = comparison_df[comparison_df["method"].astype(str).eq(AMBIGUITY_METHOD_AVERAGE)].iloc[0]
    baseline_ari = float(average_best_row["ari"])
    baseline_f1 = float(average_best_row["pairwise_f1"])
    comparison_df["ari_delta_vs_average"] = np.round(comparison_df["ari"].astype(float) - baseline_ari, 6)
    comparison_df["pairwise_f1_delta_vs_average"] = np.round(comparison_df["pairwise_f1"].astype(float) - baseline_f1, 6)
    comparison_df.to_csv(tables_dir / "comparison_summary_v1.csv", index=False)

    best_val_labels: dict[str, np.ndarray] = {}
    best_test_labels: dict[str, np.ndarray] = {}
    val_best_frames: list[pd.DataFrame] = []
    test_best_frames: list[pd.DataFrame] = []
    test_summary_rows: list[dict[str, object]] = []
    for row in comparison_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        method_name = str(row_series["method"])
        val_labels = _cluster_labels_from_best_row(
            score_matrix=boosted_val_score,
            row=row_series,
            cw_iterations=int(args.cw_iterations),
            seed=int(args.seed),
        )
        test_labels = _cluster_labels_from_best_row(
            score_matrix=boosted_test_score,
            row=row_series,
            cw_iterations=int(args.cw_iterations),
            seed=int(args.seed),
        )
        best_val_labels[method_name] = val_labels
        best_test_labels[method_name] = test_labels

        config = {
            "graph_top_k": int(row_series["graph_top_k"]) if pd.notna(row_series["graph_top_k"]) else -1,
            "mutual_top_k": bool(row_series["mutual_top_k"]) if pd.notna(row_series["mutual_top_k"]) else False,
            "min_samples": int(row_series["min_samples"]) if pd.notna(row_series["min_samples"]) else -1,
            "min_link_score": float(row_series["min_link_score"]) if pd.notna(row_series["min_link_score"]) else np.nan,
            "eps": float(row_series["eps"]) if pd.notna(row_series["eps"]) else np.nan,
        }
        val_best_frames.append(
            _build_prediction_frame(
                df=val_df,
                pred_labels=val_labels,
                method=method_name,
                threshold=float(row_series["threshold"]),
                config=config,
            )
        )
        test_best_frames.append(
            _build_prediction_frame(
                df=test_df,
                pred_labels=test_labels,
                method=method_name,
                threshold=float(row_series["threshold"]),
                config=config,
            )
        )
        summary_row = {
            "method": method_name,
            "threshold": float(row_series["threshold"]),
            **summarize_cluster_labels(test_labels),
        }
        if "graph_top_k" in row_series and pd.notna(row_series["graph_top_k"]):
            summary_row["graph_top_k"] = int(row_series["graph_top_k"])
        if "min_samples" in row_series and pd.notna(row_series["min_samples"]):
            summary_row["min_samples"] = int(row_series["min_samples"])
        if "min_link_score" in row_series and pd.notna(row_series["min_link_score"]):
            summary_row["min_link_score"] = float(row_series["min_link_score"])
        test_summary_rows.append(summary_row)

    val_best_predictions_df = pd.concat(val_best_frames, ignore_index=True)
    test_best_predictions_df = pd.concat(test_best_frames, ignore_index=True)
    val_best_predictions_df.to_csv(tables_dir / "val_best_predictions_v1.csv", index=False)
    test_best_predictions_df.to_csv(tables_dir / "test_best_predictions_v1.csv", index=False)

    val_partition_agreement_df = build_partition_agreement_table(best_val_labels)
    test_partition_agreement_df = build_partition_agreement_table(best_test_labels)
    val_partition_agreement_df.to_csv(tables_dir / "val_partition_agreement_v1.csv", index=False)
    test_partition_agreement_df.to_csv(tables_dir / "test_partition_agreement_v1.csv", index=False)

    test_summary_df = pd.DataFrame(test_summary_rows).sort_values(
        ["singleton_ratio", "clusters", "method"], ascending=[True, True, True]
    ).reset_index(drop=True)
    test_summary_df.to_csv(tables_dir / "test_cluster_summary_v1.csv", index=False)

    pair_disagreement_df = build_pair_disagreement_table(
        test_pair_df,
        label_map=best_test_labels,
        base_method=AMBIGUITY_METHOD_AVERAGE,
        base_threshold=float(average_best_row["threshold"]),
        probability_col="xgb_same_identity_prob",
    )
    pair_vote_summary_df = _build_pair_vote_summary(
        test_pair_df,
        label_map=best_test_labels,
        score_matrix=boosted_test_score,
        probability_col="xgb_same_identity_prob",
    )
    pair_disagreement_df, component_df = assign_ambiguity_components(
        pair_disagreement_df,
        min_ambiguity_score=float(args.min_ambiguity_score),
        min_conflict_ratio=float(args.min_conflict_ratio),
        probability_col="xgb_same_identity_prob",
    )
    merge_candidate_df = summarize_merge_candidates(
        pair_disagreement_df,
        base_labels=best_test_labels[AMBIGUITY_METHOD_AVERAGE],
        probability_col="xgb_same_identity_prob",
        min_merge_votes=int(args.min_merge_votes),
    )
    split_candidate_df = summarize_split_candidates(
        pair_disagreement_df,
        base_labels=best_test_labels[AMBIGUITY_METHOD_AVERAGE],
        probability_col="xgb_same_identity_prob",
        min_split_votes=int(args.min_split_votes),
    )

    pair_disagreement_df.to_csv(tables_dir / "test_pair_disagreement_v1.csv", index=False)
    pair_vote_summary_df.to_csv(tables_dir / "test_pair_vote_summary_v1.csv", index=False)
    component_df.to_csv(tables_dir / "test_ambiguity_components_v1.csv", index=False)
    merge_candidate_df.to_csv(tables_dir / "test_merge_candidates_v1.csv", index=False)
    split_candidate_df.to_csv(tables_dir / "test_split_candidates_v1.csv", index=False)

    summary = {
        "probe": "salamander_ambiguity_map_probe_v1",
        "route_dir": str(route_dir),
        "xgb_variant_dir": str(xgb_variant_dir),
        "blend_scale": float(args.blend_scale),
        "best_validation_rows": comparison_df.to_dict(orient="records"),
        "top_component_count": int(len(component_df)),
        "top_merge_candidate_count": int(len(merge_candidate_df)),
        "top_split_candidate_count": int(len(split_candidate_df)),
        "min_ambiguity_score": float(args.min_ambiguity_score),
        "min_conflict_ratio": float(args.min_conflict_ratio),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Ambiguity Map Probe",
        "",
        "- Goal: keep the current Salamander `XGBoost` score matrix fixed, and compare multiple clustering views only at the final partition stage.",
        f"- Route dir: `{route_dir}`",
        f"- XGB score source: `{xgb_variant_dir}`",
        f"- Validation images: `{len(val_df)}`",
        f"- Test images: `{len(test_df)}`",
        f"- Pair graph for ambiguity edges: `{len(test_pair_df)}` candidate pairs from `test_pair_features_v1.csv`.",
        "- Pair vote summary: `tables/test_pair_vote_summary_v1.csv` aggregates merge/split votes across the best partition from each clustering method on the boosted `XGBoost` score matrix.",
        "",
        "## Validation Best By Method",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
        "## Validation Partition Agreement",
        "",
        dataframe_to_markdown_table(val_partition_agreement_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(test_summary_df),
        "",
        "## Test Partition Agreement",
        "",
        dataframe_to_markdown_table(test_partition_agreement_df),
        "",
        "## Top Ambiguity Components",
        "",
        _format_table_or_note(
            component_df,
            note="_No ambiguity component passes the current score and conflict gates._",
            limit=int(args.report_top_k),
        ),
        "",
        "## Top Merge Candidates",
        "",
        _format_table_or_note(
            merge_candidate_df,
            note="_No merge candidate passes the current vote gate._",
            limit=int(args.report_top_k),
        ),
        "",
        "## Top Split Candidates",
        "",
        _format_table_or_note(
            split_candidate_df,
            note="_No split candidate passes the current vote gate._",
            limit=int(args.report_top_k),
        ),
        "",
        "## Reading Notes",
        "",
        "- `average_linkage` is the current official-aligned Salamander baseline and stays the base partition in the disagreement table.",
        "- `chinese_whispers` tests whether graph label propagation disagrees with average-linkage on the same score matrix.",
        "- `dbscan` tests density reachability without changing the upstream embedding or pairwise model.",
        "- `finch_like` is a first-neighbor connected-component approximation used only as an extra partition view, not as a claimed exact FINCH reproduction.",
        "- `test_pair_vote_summary_v1.csv` is the per-pair consensus artifact: `merge_votes` counts methods that keep a pair together, `split_votes` counts methods that separate it, and `conflict_methods` lists the methods off the majority side (or all methods on ties).",
        "- `test_pair_disagreement_v1.csv` is built on the existing pair-feature candidate graph, so it is a precise ambiguity map over plausible edges rather than an all-pairs exhaustive report.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[salamander_ambiguity_map_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_ambiguity_map_probe] pair votes: {tables_dir / 'test_pair_vote_summary_v1.csv'}")
    print(f"[salamander_ambiguity_map_probe] merge candidates: {tables_dir / 'test_merge_candidates_v1.csv'}")
    print(f"[salamander_ambiguity_map_probe] split candidates: {tables_dir / 'test_split_candidates_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
