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
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_graph_merge_overlay_probe_v1")
DEFAULT_BASE_THRESHOLDS = [0.2, 0.25, 0.3]
DEFAULT_GRAPH_THRESHOLDS = [0.82, 0.84, 0.86, 0.88]
DEFAULT_DBSCAN_MIN_SAMPLES = [2, 3]
DEFAULT_MIN_MERGE_VOTES = [2, 3]
DEFAULT_MIN_VOTE_RATIOS = [0.67, 1.0]
DEFAULT_MIN_PAIR_SCORES = [0.82, 0.84, 0.86]
DEFAULT_MIN_PAIR_PROBABILITIES = [0.75, 0.8]
DEFAULT_MIN_SUPPORT_PAIR_COUNTS = [1, 2]
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
    ranking_columns = [column for column in ["ari", "pairwise_f1", "accepted_merge_candidates", "threshold"] if column in df.columns]
    ascending = [False, False, True, True][: len(ranking_columns)]
    return df.sort_values(ranking_columns, ascending=ascending).iloc[0]


def _format_table_or_note(frame: pd.DataFrame, *, note: str, limit: int) -> str:
    if frame.empty:
        return note
    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table

    return dataframe_to_markdown_table(frame.head(int(limit)))


def _cluster_labels_from_row(
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
        return cluster_labels_from_score_graph(
            score_matrix=score_matrix,
            threshold=threshold,
            method=method,
            top_k=int(row["graph_top_k"]),
            mutual_top_k=bool(row["mutual_top_k"]),
            iterations=int(cw_iterations),
            seed=int(seed),
        )
    if method == AMBIGUITY_METHOD_DBSCAN:
        return cluster_labels_from_dbscan_score_matrix(
            score_matrix=score_matrix,
            threshold=threshold,
            min_samples=int(row["min_samples"]),
        )
    if method == AMBIGUITY_METHOD_FINCH_LIKE:
        return cluster_labels_from_finch_like_score_matrix(
            score_matrix=score_matrix,
            min_link_score=float(row["min_link_score"]),
        )
    raise ValueError(f"Unsupported method row: {method}")


def _build_overlay_prediction_frame(
    df: pd.DataFrame,
    pred_labels: np.ndarray,
    *,
    route_name: str,
    base_threshold: float,
    min_merge_votes: int,
    min_vote_ratio: float,
    min_pair_score: float,
    min_pair_probability: float,
    min_support_pair_count: int,
    accepted_merge_candidates: int,
) -> pd.DataFrame:
    frame = df.copy().reset_index(drop=True)
    frame["route_name"] = str(route_name)
    frame["chosen_threshold"] = float(base_threshold)
    frame["pred_cluster_id"] = np.asarray(pred_labels, dtype=np.int32)
    frame["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in pred_labels]
    frame["consensus_overlay_enabled"] = True
    frame["min_merge_votes"] = int(min_merge_votes)
    frame["min_vote_ratio"] = float(min_vote_ratio)
    frame["min_pair_score"] = float(min_pair_score)
    frame["min_pair_probability"] = float(min_pair_probability)
    frame["min_support_pair_count"] = int(min_support_pair_count)
    frame["accepted_merge_candidates"] = int(accepted_merge_candidates)
    return frame


def _build_summary_row(labels: np.ndarray, *, route: str, chosen_row: pd.Series, accepted_merge_candidates: int) -> dict[str, object]:
    counts = pd.Series(np.asarray(labels, dtype=np.int32)).value_counts()
    return {
        "route": str(route),
        "base_threshold": float(chosen_row["base_threshold"]),
        "min_merge_votes": int(chosen_row["min_merge_votes"]),
        "min_vote_ratio": float(chosen_row["min_vote_ratio"]),
        "min_pair_score": float(chosen_row["min_pair_score"]),
        "min_pair_probability": float(chosen_row["min_pair_probability"]),
        "min_support_pair_count": int(chosen_row["min_support_pair_count"]),
        "accepted_merge_candidates": int(accepted_merge_candidates),
        "clusters": int(counts.size),
        "singleton_clusters": int((counts == 1).sum()),
        "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.ambiguity_clustering import (
        AMBIGUITY_METHOD_AVERAGE,
        AMBIGUITY_METHOD_CHINESE_WHISPERS,
        AMBIGUITY_METHOD_DBSCAN,
        AMBIGUITY_METHOD_FINCH_LIKE,
        build_pair_vote_summary_table,
        run_dbscan_threshold_sweep,
        run_finch_like_threshold_sweep,
    )
    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table, summarize_cluster_metrics
    from animalclef_analysis.graph_cluster_overlay import apply_graph_merge_overlay, summarize_consensus_merge_candidates
    from animalclef_analysis.graph_clustering import run_graph_threshold_sweep
    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.texas_unsupervised import summarize_cluster_labels
    from animalclef_analysis.transductive_seed_refinement import cluster_labels_from_score_matrix, run_score_threshold_sweep

    parser = argparse.ArgumentParser(description="Probe Salamander multi-clustering consensus merge overlay on top of the current XGBoost score matrix.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--xgb-variant-dir", type=Path, default=DEFAULT_XGB_VARIANT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--base-thresholds", nargs="+", type=float, default=DEFAULT_BASE_THRESHOLDS)
    parser.add_argument("--graph-thresholds", nargs="+", type=float, default=DEFAULT_GRAPH_THRESHOLDS)
    parser.add_argument("--dbscan-min-samples", nargs="+", type=int, default=DEFAULT_DBSCAN_MIN_SAMPLES)
    parser.add_argument("--graph-top-k", type=int, default=8)
    parser.add_argument("--mutual-top-k", action="store_true")
    parser.add_argument("--cw-iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-merge-votes", nargs="+", type=int, default=DEFAULT_MIN_MERGE_VOTES)
    parser.add_argument("--min-vote-ratios", nargs="+", type=float, default=DEFAULT_MIN_VOTE_RATIOS)
    parser.add_argument("--min-pair-scores", nargs="+", type=float, default=DEFAULT_MIN_PAIR_SCORES)
    parser.add_argument("--min-pair-probabilities", nargs="+", type=float, default=DEFAULT_MIN_PAIR_PROBABILITIES)
    parser.add_argument("--min-support-pair-counts", nargs="+", type=int, default=DEFAULT_MIN_SUPPORT_PAIR_COUNTS)
    parser.add_argument("--official-base-threshold", type=float, default=0.25)
    parser.add_argument("--report-top-k", type=int, default=DEFAULT_REPORT_TOP_K)
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
    for pair_df in [val_pair_df, test_pair_df]:
        pair_df["image_id"] = pair_df["image_id"].astype(str)
        pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)

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

    base_sweep_df, _ = run_score_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.base_thresholds],
        score_space="unit_interval",
    )
    base_sweep_df["method"] = AMBIGUITY_METHOD_AVERAGE
    base_sweep_df.to_csv(tables_dir / "base_average_threshold_sweep_v1.csv", index=False)

    cw_sweep_df, _ = run_graph_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.graph_thresholds],
        method=AMBIGUITY_METHOD_CHINESE_WHISPERS,
        top_k=int(args.graph_top_k),
        mutual_top_k=bool(args.mutual_top_k),
        iterations=int(args.cw_iterations),
        seed=int(args.seed),
    )
    cw_sweep_df = cw_sweep_df.rename(columns={"top_k": "graph_top_k"})
    cw_sweep_df["method"] = AMBIGUITY_METHOD_CHINESE_WHISPERS
    cw_sweep_df["mutual_top_k"] = bool(args.mutual_top_k)

    dbscan_sweep_df, _ = run_dbscan_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        thresholds=[float(value) for value in args.graph_thresholds],
        min_samples_values=[int(value) for value in args.dbscan_min_samples],
    )

    finch_sweep_df, _ = run_finch_like_threshold_sweep(
        df=val_df,
        score_matrix=boosted_val_score,
        min_link_scores=[float(value) for value in args.graph_thresholds],
    )

    method_selection_rows = [
        _pick_best_row(cw_sweep_df.assign(threshold=cw_sweep_df["threshold"].astype(float))),
        _pick_best_row(dbscan_sweep_df.assign(threshold=dbscan_sweep_df["threshold"].astype(float))),
        _pick_best_row(finch_sweep_df.assign(threshold=finch_sweep_df["threshold"].astype(float))),
    ]
    method_selection_df = pd.DataFrame(method_selection_rows).reset_index(drop=True)
    method_selection_df.to_csv(tables_dir / "selected_alt_method_rows_v1.csv", index=False)

    selected_val_labels: dict[str, np.ndarray] = {}
    selected_test_labels: dict[str, np.ndarray] = {}
    for row in method_selection_df.to_dict(orient="records"):
        row_series = pd.Series(row)
        method_name = str(row_series["method"])
        selected_val_labels[method_name] = _cluster_labels_from_row(
            boosted_val_score,
            row_series,
            cw_iterations=int(args.cw_iterations),
            seed=int(args.seed),
        )
        selected_test_labels[method_name] = _cluster_labels_from_row(
            boosted_test_score,
            row_series,
            cw_iterations=int(args.cw_iterations),
            seed=int(args.seed),
        )

    consensus_rows: list[dict[str, object]] = []
    candidate_frames: list[pd.DataFrame] = []
    pair_vote_frames: list[pd.DataFrame] = []
    has_identity = "identity" in val_df.columns and val_df["identity"].fillna("").astype(str).ne("").any()
    true_labels = val_df["identity"].astype(str).to_numpy() if has_identity else None

    for base_threshold in [float(value) for value in args.base_thresholds]:
        base_val_labels = cluster_labels_from_score_matrix(
            score_matrix=boosted_val_score,
            threshold=float(base_threshold),
            score_space="unit_interval",
        )
        label_map = {
            AMBIGUITY_METHOD_AVERAGE: base_val_labels,
            **selected_val_labels,
        }
        pair_vote_df = build_pair_vote_summary_table(
            val_pair_df,
            label_map=label_map,
            base_method=AMBIGUITY_METHOD_AVERAGE,
            probability_col="xgb_same_identity_prob",
            base_threshold=float(base_threshold),
            score_matrix=boosted_val_score,
        )
        pair_vote_df["base_threshold"] = float(base_threshold)
        pair_vote_frames.append(pair_vote_df)

        for min_merge_votes in [int(value) for value in args.min_merge_votes]:
            for min_vote_ratio in [float(value) for value in args.min_vote_ratios]:
                for min_pair_score in [float(value) for value in args.min_pair_scores]:
                    for min_pair_probability in [float(value) for value in args.min_pair_probabilities]:
                        for min_support_pair_count in [int(value) for value in args.min_support_pair_counts]:
                            candidate_df = summarize_consensus_merge_candidates(
                                pair_vote_df,
                                min_merge_votes=int(min_merge_votes),
                                min_vote_ratio=float(min_vote_ratio),
                                min_support_pair_count=int(min_support_pair_count),
                                pair_score_col="pair_score",
                                min_pair_score=float(min_pair_score),
                                probability_col="xgb_same_identity_prob",
                                min_pair_probability=float(min_pair_probability),
                            )
                            if not candidate_df.empty:
                                candidate_df = candidate_df.copy()
                                candidate_df["base_threshold"] = float(base_threshold)
                                candidate_df["min_merge_votes"] = int(min_merge_votes)
                                candidate_df["min_vote_ratio"] = float(min_vote_ratio)
                                candidate_df["min_pair_score"] = float(min_pair_score)
                                candidate_df["min_pair_probability"] = float(min_pair_probability)
                                candidate_df["min_support_pair_count"] = int(min_support_pair_count)
                                candidate_frames.append(candidate_df)

                            overlay_labels = apply_graph_merge_overlay(base_labels=base_val_labels, candidate_df=candidate_df)
                            counts = pd.Series(overlay_labels).value_counts()
                            row: dict[str, object] = {
                                "dataset": SALAMANDER_DATASET,
                                "base_threshold": float(base_threshold),
                                "min_merge_votes": int(min_merge_votes),
                                "min_vote_ratio": float(min_vote_ratio),
                                "min_pair_score": float(min_pair_score),
                                "min_pair_probability": float(min_pair_probability),
                                "min_support_pair_count": int(min_support_pair_count),
                                "accepted_merge_candidates": int(len(candidate_df)),
                                "graph_top_k": int(args.graph_top_k),
                                "mutual_top_k": bool(args.mutual_top_k),
                                "samples": int(len(val_df)),
                                "cluster_count": int(counts.size),
                                "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                            }
                            if has_identity and true_labels is not None:
                                row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=overlay_labels))
                            consensus_rows.append(row)

    val_pair_vote_all_df = pd.concat(pair_vote_frames, ignore_index=True) if pair_vote_frames else pd.DataFrame()
    if not val_pair_vote_all_df.empty:
        val_pair_vote_all_df.to_csv(tables_dir / "val_pair_vote_summary_all_v1.csv", index=False)

    candidate_summary_df = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not candidate_summary_df.empty:
        candidate_summary_df.to_csv(tables_dir / "consensus_merge_candidates_all_v1.csv", index=False)

    consensus_sweep_df = pd.DataFrame(consensus_rows).sort_values(
        ["ari", "pairwise_f1", "accepted_merge_candidates", "min_vote_ratio", "min_pair_score", "base_threshold"],
        ascending=[False, False, True, False, False, True],
    ).reset_index(drop=True)
    consensus_sweep_df.to_csv(tables_dir / "consensus_overlay_sweep_v1.csv", index=False)

    local_best_base_row = _pick_best_row(base_sweep_df.assign(threshold=base_sweep_df["threshold"].astype(float)))
    official_base_row = base_sweep_df[np.isclose(base_sweep_df["threshold"].astype(float), float(args.official_base_threshold))].iloc[0]
    best_overlay_row = _pick_best_row(consensus_sweep_df)
    official_overlay_pool = consensus_sweep_df[
        np.isclose(consensus_sweep_df["base_threshold"].astype(float), float(args.official_base_threshold))
    ].copy()
    official_overlay_row = _pick_best_row(official_overlay_pool) if not official_overlay_pool.empty else None

    comparison_rows = [
        {
            "route": "xgb_average_linkage_local_best",
            "base_threshold": float(local_best_base_row["threshold"]),
            "accepted_merge_candidates": 0,
            "ari": float(local_best_base_row["ari"]),
            "pairwise_f1": float(local_best_base_row["pairwise_f1"]),
            "cluster_count": int(local_best_base_row["cluster_count"]),
        },
        {
            "route": "xgb_average_linkage_official_aligned",
            "base_threshold": float(official_base_row["threshold"]),
            "accepted_merge_candidates": 0,
            "ari": float(official_base_row["ari"]),
            "pairwise_f1": float(official_base_row["pairwise_f1"]),
            "cluster_count": int(official_base_row["cluster_count"]),
        },
        {
            "route": "xgb_consensus_merge_overlay_best_overall",
            "base_threshold": float(best_overlay_row["base_threshold"]),
            "accepted_merge_candidates": int(best_overlay_row["accepted_merge_candidates"]),
            "ari": float(best_overlay_row["ari"]),
            "pairwise_f1": float(best_overlay_row["pairwise_f1"]),
            "cluster_count": int(best_overlay_row["cluster_count"]),
        },
    ]
    if official_overlay_row is not None:
        comparison_rows.append(
            {
                "route": "xgb_consensus_merge_overlay_official_aligned",
                "base_threshold": float(official_overlay_row["base_threshold"]),
                "accepted_merge_candidates": int(official_overlay_row["accepted_merge_candidates"]),
                "ari": float(official_overlay_row["ari"]),
                "pairwise_f1": float(official_overlay_row["pairwise_f1"]),
                "cluster_count": int(official_overlay_row["cluster_count"]),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows)
    official_ari = float(official_base_row["ari"])
    official_f1 = float(official_base_row["pairwise_f1"])
    comparison_df["ari_delta_vs_official_aligned"] = np.round(comparison_df["ari"].astype(float) - official_ari, 6)
    comparison_df["pairwise_f1_delta_vs_official_aligned"] = np.round(comparison_df["pairwise_f1"].astype(float) - official_f1, 6)
    comparison_df.to_csv(tables_dir / "comparison_summary_v1.csv", index=False)

    test_summary_rows: list[dict[str, object]] = []
    for route_name, chosen_row in [
        ("xgb_consensus_merge_overlay_best_overall", best_overlay_row),
        ("xgb_consensus_merge_overlay_official_aligned", official_overlay_row),
    ]:
        if chosen_row is None:
            continue
        base_test_labels = cluster_labels_from_score_matrix(
            score_matrix=boosted_test_score,
            threshold=float(chosen_row["base_threshold"]),
            score_space="unit_interval",
        )
        label_map = {
            AMBIGUITY_METHOD_AVERAGE: base_test_labels,
            **selected_test_labels,
        }
        test_pair_vote_df = build_pair_vote_summary_table(
            test_pair_df,
            label_map=label_map,
            base_method=AMBIGUITY_METHOD_AVERAGE,
            probability_col="xgb_same_identity_prob",
            base_threshold=float(chosen_row["base_threshold"]),
            score_matrix=boosted_test_score,
        )
        candidate_df = summarize_consensus_merge_candidates(
            test_pair_vote_df,
            min_merge_votes=int(chosen_row["min_merge_votes"]),
            min_vote_ratio=float(chosen_row["min_vote_ratio"]),
            min_support_pair_count=int(chosen_row["min_support_pair_count"]),
            pair_score_col="pair_score",
            min_pair_score=float(chosen_row["min_pair_score"]),
            probability_col="xgb_same_identity_prob",
            min_pair_probability=float(chosen_row["min_pair_probability"]),
        )
        overlay_labels = apply_graph_merge_overlay(base_labels=base_test_labels, candidate_df=candidate_df)
        prediction_df = _build_overlay_prediction_frame(
            df=test_df,
            pred_labels=overlay_labels,
            route_name=route_name,
            base_threshold=float(chosen_row["base_threshold"]),
            min_merge_votes=int(chosen_row["min_merge_votes"]),
            min_vote_ratio=float(chosen_row["min_vote_ratio"]),
            min_pair_score=float(chosen_row["min_pair_score"]),
            min_pair_probability=float(chosen_row["min_pair_probability"]),
            min_support_pair_count=int(chosen_row["min_support_pair_count"]),
            accepted_merge_candidates=int(len(candidate_df)),
        )
        suffix = "best_overall" if "best_overall" in route_name else "official_aligned"
        prediction_df.to_csv(tables_dir / f"test_predictions_{suffix}_v1.csv", index=False)
        test_pair_vote_df.to_csv(tables_dir / f"test_pair_vote_summary_{suffix}_v1.csv", index=False)
        if not candidate_df.empty:
            candidate_df.to_csv(tables_dir / f"test_merge_candidates_{suffix}_v1.csv", index=False)
        test_summary_rows.append(
            _build_summary_row(
                overlay_labels,
                route=route_name,
                chosen_row=chosen_row,
                accepted_merge_candidates=int(len(candidate_df)),
            )
        )

    test_summary_df = pd.DataFrame(test_summary_rows)
    if not test_summary_df.empty:
        test_summary_df.to_csv(tables_dir / "test_cluster_summary_v1.csv", index=False)

    summary = {
        "probe": "salamander_graph_merge_overlay_probe_v1",
        "route_dir": str(route_dir),
        "xgb_variant_dir": str(xgb_variant_dir),
        "blend_scale": float(args.blend_scale),
        "selected_alt_method_rows": method_selection_df.to_dict(orient="records"),
        "comparison_rows": comparison_df.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Consensus Merge Overlay Probe",
        "",
        "- Goal: keep the current Salamander `XGBoost` score matrix fixed, and only add conservative merge overlays when multiple clustering methods agree on the same cross-cluster pair region.",
        f"- Route dir: `{route_dir}`",
        f"- XGB score source: `{xgb_variant_dir}`",
        f"- Validation images: `{len(val_df)}`",
        f"- Test images: `{len(test_df)}`",
        f"- Chinese whispers config: `top_k={int(args.graph_top_k)}`, `mutual_top_k={bool(args.mutual_top_k)}`",
        "",
        "## Selected Alt Methods",
        "",
        dataframe_to_markdown_table(method_selection_df),
        "",
        "## Comparison",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
        "## Top Consensus Overlay Rows",
        "",
        dataframe_to_markdown_table(consensus_sweep_df.head(10)),
        "",
        "## Top Consensus Merge Candidates",
        "",
        _format_table_or_note(
            candidate_summary_df,
            note="_No consensus merge candidate passes the current gates._",
            limit=int(args.report_top_k),
        ),
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
            "## Reading Notes",
            "",
            "- `xgb_average_linkage_official_aligned` is the operational baseline because the current Salamander official route uses `average-linkage @ 0.25`.",
            "- `consensus merge overlay` differs from old graph-stability overlay: it requires agreement across the best `average_linkage` / `chinese_whispers` / `dbscan` / `finch_like` partitions on the same boosted `XGBoost` score matrix.",
            "- Candidate merges are filtered by four gates together: `merge_votes`, `vote_ratio`, `pair_score`, and `xgb_same_identity_prob`; then they are aggregated to base-cluster pairs before applying union-find merges.",
            "- If validation gain comes from only a small number of accepted cluster-pair merges, the route is more likely to transfer than a broad repartitioning change.",
        ]
    )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[salamander_graph_merge_overlay_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_graph_merge_overlay_probe] comparison: {tables_dir / 'comparison_summary_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
