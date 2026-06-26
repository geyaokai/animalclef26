#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_local_matcher_sweep_20260330")
DEFAULT_MATCHERS = ["orb", "akaze", "sift", "brisk"]
DEFAULT_LOCAL_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
DEFAULT_TOP_K = 10
SALAMANDER_DATASET = "SalamanderID2025"


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["ari", "pairwise_f1", "nmi", "local_weight", "threshold"],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def _metric_key(row: pd.Series) -> tuple[float, float, float, float, float]:
    return (
        float(row["ari"]),
        float(row["pairwise_f1"]),
        float(row["nmi"]),
        -float(row["local_weight"]),
        -float(row["threshold"]),
    )


def _build_cluster_predictions(
    dataset_df: pd.DataFrame,
    score_matrix: np.ndarray,
    threshold: float,
    route_name: str,
    local_matcher: str,
    local_weight: float,
    embedding_dim: int,
):
    from animalclef_analysis.descriptor_baselines import build_average_linkage, cluster_from_linkage
    from animalclef_analysis.orb_rerank_baseline import score_matrix_to_distance

    distance = score_matrix_to_distance(score_matrix)
    linkage_matrix = build_average_linkage(distance)
    pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), threshold)

    result = dataset_df.copy().reset_index(drop=True)
    result["chosen_threshold"] = float(threshold)
    result["pred_cluster_id"] = pred_labels
    result["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in pred_labels]
    result["route_name"] = route_name
    result["embedding_dim"] = int(embedding_dim)
    result["rerank_enabled"] = True
    result["local_weight"] = float(local_weight)
    result["local_matcher"] = local_matcher
    return result


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import PATH_COLUMN, dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import (
        SUPPORTED_LOCAL_MATCHERS,
        apply_local_rerank,
        build_local_match_table,
        build_top1_transition_table,
        build_topk_pair_index,
        cosine_score_matrix,
        evaluate_threshold_sweep_from_score_matrix,
        extract_local_features,
        normalize_local_matcher_name,
        resolve_existing_image_rel_path,
    )

    parser = argparse.ArgumentParser(description="Compare Salamander local matchers with fixed global embeddings.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--matchers", nargs="+", default=None)
    parser.add_argument("--local-weights", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--nfeatures", type=int, default=1024)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--fast-threshold", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--ratio-test", type=float, default=0.8)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    args = parser.parse_args()

    route_dir = args.route_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    local_weights = args.local_weights or DEFAULT_LOCAL_WEIGHTS
    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    matchers = [normalize_local_matcher_name(name) for name in (args.matchers or DEFAULT_MATCHERS)]

    unsupported = sorted(set(matchers) - set(SUPPORTED_LOCAL_MATCHERS))
    if unsupported:
        raise ValueError(f"Unsupported matchers requested: {unsupported}")

    val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy")
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy")
    val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    test_df = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    val_df["image_id"] = val_df["image_id"].astype(str)
    test_df["image_id"] = test_df["image_id"].astype(str)
    if "identity" in val_df.columns:
        val_df["identity"] = val_df["identity"].fillna("").astype(str)
    if "identity" in test_df.columns:
        test_df["identity"] = test_df["identity"].fillna("").astype(str)
    val_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in val_df.iterrows()]
    test_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in test_df.iterrows()]

    base_pred_df = pd.read_csv(route_dir / "tables" / "test_predictions_v1.csv")
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)

    baseline_best_df = pd.read_csv(route_dir / "tables" / "best_config_v1.csv")
    baseline_best = baseline_best_df.iloc[0]
    baseline_cluster_df = pd.read_csv(route_dir / "tables" / "cluster_summary_v1.csv")
    baseline_salamander_cluster = baseline_cluster_df[baseline_cluster_df["dataset"] == SALAMANDER_DATASET].iloc[0]

    val_score = cosine_score_matrix(val_embeddings)
    test_score = cosine_score_matrix(test_embeddings)
    val_pair_index = build_topk_pair_index(score_matrix=val_score, top_k=int(args.top_k), query_indices=None)
    test_pair_index = build_topk_pair_index(score_matrix=test_score, top_k=int(args.top_k), query_indices=None)

    summary_rows: list[dict[str, object]] = []
    test_cluster_rows: list[dict[str, object]] = []
    global_best_meta: dict[str, object] | None = None

    for matcher_name in matchers:
        matcher_slug = matcher_name.replace(".", "p")
        val_features = extract_local_features(
            df=val_df,
            repo_root=repo_root,
            nfeatures=int(args.nfeatures),
            max_side=int(args.max_side),
            fast_threshold=int(args.fast_threshold),
            clahe_clip_limit=float(args.clahe_clip_limit),
            local_matcher=matcher_name,
        )
        test_features = extract_local_features(
            df=test_df,
            repo_root=repo_root,
            nfeatures=int(args.nfeatures),
            max_side=int(args.max_side),
            fast_threshold=int(args.fast_threshold),
            clahe_clip_limit=float(args.clahe_clip_limit),
            local_matcher=matcher_name,
        )

        pd.DataFrame(
            [{"image_id": feature.image_id, "keypoints": feature.point_count, "width": feature.width, "height": feature.height} for feature in val_features]
        ).to_csv(tables_dir / f"val_keypoints_{matcher_slug}_v1.csv", index=False)
        pd.DataFrame(
            [{"image_id": feature.image_id, "keypoints": feature.point_count, "width": feature.width, "height": feature.height} for feature in test_features]
        ).to_csv(tables_dir / f"test_keypoints_{matcher_slug}_v1.csv", index=False)

        val_pair_df = build_local_match_table(
            df=val_df,
            features=val_features,
            pair_index=val_pair_index,
            ratio_test=float(args.ratio_test),
            ransac_threshold=float(args.ransac_threshold),
            min_inliers=int(args.min_inliers),
            local_matcher=matcher_name,
        )
        val_pair_df.to_csv(tables_dir / f"val_local_match_scores_{matcher_slug}_v1.csv", index=False)

        test_pair_df = build_local_match_table(
            df=test_df,
            features=test_features,
            pair_index=test_pair_index,
            ratio_test=float(args.ratio_test),
            ransac_threshold=float(args.ransac_threshold),
            min_inliers=int(args.min_inliers),
            local_matcher=matcher_name,
        )
        test_pair_df.to_csv(tables_dir / f"test_local_match_scores_{matcher_slug}_v1.csv", index=False)

        matcher_best_row = None
        matcher_best_score = None
        matcher_best_prediction = None
        matcher_best_meta = None
        matcher_weight_rows = []

        for local_weight in local_weights:
            reranked_val_score = apply_local_rerank(
                global_score_matrix=val_score,
                pair_df=val_pair_df,
                local_weight=float(local_weight),
            )
            sweep_df, prediction_df = evaluate_threshold_sweep_from_score_matrix(
                df=val_df,
                score_matrix=reranked_val_score,
                thresholds=thresholds,
            )
            sweep_df["local_matcher"] = matcher_name
            sweep_df["local_weight"] = float(local_weight)
            prediction_df["local_matcher"] = matcher_name
            prediction_df["local_weight"] = float(local_weight)
            sweep_df.to_csv(
                tables_dir / f"val_threshold_sweep_{matcher_slug}_w{str(local_weight).replace('.', 'p')}_v1.csv",
                index=False,
            )

            best_row = _pick_best_row(sweep_df)
            matcher_weight_rows.append(
                {
                    "local_matcher": matcher_name,
                    "local_weight": float(local_weight),
                    "best_threshold": float(best_row["threshold"]),
                    "ari": float(best_row["ari"]),
                    "nmi": float(best_row["nmi"]),
                    "pairwise_f1": float(best_row["pairwise_f1"]),
                    "cluster_count": int(best_row["cluster_count"]),
                    "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
                }
            )
            if matcher_best_meta is None or _metric_key(best_row) > _metric_key(pd.Series(matcher_best_meta)):
                matcher_best_meta = best_row.to_dict()
                matcher_best_row = best_row
                matcher_best_score = reranked_val_score.copy()
                matcher_best_prediction = prediction_df[prediction_df["threshold"] == float(best_row["threshold"])].copy().reset_index(drop=True)

        if matcher_best_meta is None or matcher_best_score is None or matcher_best_prediction is None or matcher_best_row is None:
            raise RuntimeError(f"Failed to select best config for matcher={matcher_name}")

        pd.DataFrame(matcher_weight_rows).to_csv(tables_dir / f"val_weight_summary_{matcher_slug}_v1.csv", index=False)
        pd.DataFrame([matcher_best_meta]).to_csv(tables_dir / f"best_config_{matcher_slug}_v1.csv", index=False)
        matcher_best_prediction.to_csv(tables_dir / f"val_predictions_best_{matcher_slug}_v1.csv", index=False)

        val_transition_df = build_top1_transition_table(val_df, val_score, matcher_best_score)
        val_transition_df.to_csv(tables_dir / f"val_top1_transitions_{matcher_slug}_v1.csv", index=False)

        reranked_test_score = apply_local_rerank(
            global_score_matrix=test_score,
            pair_df=test_pair_df,
            local_weight=float(matcher_best_meta["local_weight"]),
        )
        route_name = f"ft_miew_arcface_masked_supcon_v1_last_fusion_{matcher_name}_v1"
        salamander_test_pred_df = _build_cluster_predictions(
            dataset_df=test_df,
            score_matrix=reranked_test_score,
            threshold=float(matcher_best_meta["threshold"]),
            route_name=route_name,
            local_matcher=matcher_name,
            local_weight=float(matcher_best_meta["local_weight"]),
            embedding_dim=int(test_embeddings.shape[1]),
        )
        salamander_test_pred_df.to_csv(tables_dir / f"salamander_test_predictions_{matcher_slug}_v1.csv", index=False)

        merged_test_pred_df = pd.concat(
            [base_pred_df[base_pred_df["dataset"] != SALAMANDER_DATASET].copy(), salamander_test_pred_df],
            ignore_index=True,
        )
        merged_test_pred_df.to_csv(tables_dir / f"test_predictions_{matcher_slug}_v1.csv", index=False)

        cluster_counts = salamander_test_pred_df["pred_cluster_id"].value_counts()
        test_cluster_rows.append(
            {
                "local_matcher": matcher_name,
                "threshold": float(matcher_best_meta["threshold"]),
                "local_weight": float(matcher_best_meta["local_weight"]),
                "clusters": int(cluster_counts.size),
                "singleton_clusters": int((cluster_counts == 1).sum()),
                "singleton_ratio": round(float((cluster_counts == 1).mean()) if len(cluster_counts) else 0.0, 6),
            }
        )

        summary_row = {
            "local_matcher": matcher_name,
            "threshold": float(matcher_best_meta["threshold"]),
            "local_weight": float(matcher_best_meta["local_weight"]),
            "ari": float(matcher_best_meta["ari"]),
            "nmi": float(matcher_best_meta["nmi"]),
            "pairwise_f1": float(matcher_best_meta["pairwise_f1"]),
            "cluster_count": int(matcher_best_meta["cluster_count"]),
            "singleton_cluster_ratio": float(matcher_best_meta["singleton_cluster_ratio"]),
            "ari_delta_vs_orb": round(float(matcher_best_meta["ari"]) - float(baseline_best["ari"]), 6),
            "pairwise_f1_delta_vs_orb": round(float(matcher_best_meta["pairwise_f1"]) - float(baseline_best["pairwise_f1"]), 6),
        }
        summary_rows.append(summary_row)

        if global_best_meta is None or _metric_key(pd.Series(summary_row)) > _metric_key(pd.Series(global_best_meta)):
            global_best_meta = summary_row

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "local_matcher"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    test_cluster_df = pd.DataFrame(test_cluster_rows).sort_values(
        ["local_matcher"],
        ascending=[True],
    ).reset_index(drop=True)
    summary_df.to_csv(tables_dir / "matcher_summary_v1.csv", index=False)
    test_cluster_df.to_csv(tables_dir / "matcher_test_cluster_summary_v1.csv", index=False)

    config = {
        "route_dir": str(route_dir),
        "matchers": matchers,
        "local_weights": local_weights,
        "thresholds": thresholds,
        "top_k": int(args.top_k),
        "nfeatures": int(args.nfeatures),
        "max_side": int(args.max_side),
        "fast_threshold": int(args.fast_threshold),
        "clahe_clip_limit": float(args.clahe_clip_limit),
        "ratio_test": float(args.ratio_test),
        "ransac_threshold": float(args.ransac_threshold),
        "min_inliers": int(args.min_inliers),
        "baseline_best_orb": {
            "threshold": float(baseline_best["threshold"]),
            "local_weight": float(baseline_best["local_weight"]),
            "ari": float(baseline_best["ari"]),
            "pairwise_f1": float(baseline_best["pairwise_f1"]),
            "cluster_count": int(baseline_best["cluster_count"]),
            "test_clusters": int(baseline_salamander_cluster["clusters"]),
        },
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Local Matcher Sweep",
        "",
        f"- Anchor route dir: `{route_dir}`",
        "- Fixed global branch: `ft_miew_arcface_masked_supcon_v1 last + frozen fusion`",
        "- Single-factor change: `only replace the local rerank matcher`",
        f"- Matchers: `{', '.join(matchers)}`",
        f"- Top-K candidate neighbors: `{int(args.top_k)}`",
        f"- Current ORB anchor local val: `ARI {float(baseline_best['ari']):.6f}`, `pairwise_f1 {float(baseline_best['pairwise_f1']):.6f}`, threshold `{float(baseline_best['threshold'])}`, local_weight `{float(baseline_best['local_weight'])}`",
        "",
        "## Matcher Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(test_cluster_df),
        "",
        "## Decision Hint",
        "",
    ]
    if global_best_meta is not None:
        best_matcher = str(global_best_meta["local_matcher"])
        best_delta = float(global_best_meta["ari_delta_vs_orb"])
        if best_delta > 0:
            lines.append(
                f"- Best offline matcher is `{best_matcher}` with `ARI +{best_delta:.6f}` vs current ORB anchor; this is the only matcher that currently deserves consideration for the reserved official slot."
            )
        else:
            lines.append(
                f"- No matcher beat the current ORB anchor offline. Best observed delta is `{best_delta:.6f}` by `{best_matcher}`; keep the reserved official slot unused for now."
            )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_local_matcher_sweep] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_local_matcher_sweep] matcher_summary: {tables_dir / 'matcher_summary_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
