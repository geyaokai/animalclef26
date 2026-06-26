#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover
    XGBClassifier = None

from sklearn.ensemble import GradientBoostingClassifier


def _build_pairwise_model(backend: str, split_seed: int):
    resolved_backend = str(backend)
    if resolved_backend == "auto":
        resolved_backend = "xgboost" if XGBClassifier is not None else "sklearn"
    if resolved_backend == "xgboost":
        if XGBClassifier is None:
            raise ModuleNotFoundError("Requested backend 'xgboost' but xgboost is not installed.")
        model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(split_seed),
            n_jobs=8,
            tree_method="hist",
        )
        return model, resolved_backend
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=int(split_seed),
    )
    return model, "sklearn"


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    weights = np.ones(len(labels), dtype=np.float32)
    if len(labels) == 0:
        return weights
    counts = np.bincount(labels.astype(int), minlength=2)
    for class_id in [0, 1]:
        if counts[class_id] == 0:
            continue
        weights[labels == class_id] = float(len(labels) / (2.0 * counts[class_id]))
    return weights


def _to_markdown_table(frame: pd.DataFrame, columns: list[str] | None = None, limit: int = 20) -> str:
    preview = frame.copy()
    if columns is not None:
        preview = preview.loc[:, [column for column in columns if column in preview.columns]].copy()
    preview = preview.head(int(limit)).copy()
    if preview.empty:
        return "_empty_"
    header = "| " + " | ".join(preview.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(preview.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in preview.columns.tolist()) + " |"
        for _, row in preview.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_local_pairwise import (
        DEFAULT_TEXAS_BLACK_LOCAL_CSV,
        DEFAULT_TEXAS_EXPERIMENT_DIR,
        DEFAULT_TEXAS_TEACHER_SOURCE_DIR,
        DEFAULT_TEXAS_THRESHOLDS,
        DEFAULT_TOP_K,
        TEXAS_XGB_FEATURE_COLUMNS,
        apply_pair_probability_residual,
        apply_texas_local_rerank,
        enrich_texas_pair_df,
        evaluate_texas_thresholds_from_score_matrix,
        load_texas_local_artifacts,
        merge_texas_black_patch_scores,
        merge_texas_black_pattern_scores,
        pick_best_proxy_row,
    )

    parser = argparse.ArgumentParser(description="Run Texas black-pattern local rerank -> XGBoost pairwise probe.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_TEXAS_EXPERIMENT_DIR)
    parser.add_argument("--teacher-source-dir", type=Path, default=DEFAULT_TEXAS_TEACHER_SOURCE_DIR)
    parser.add_argument("--local-match-csv", type=Path, default=DEFAULT_TEXAS_BLACK_LOCAL_CSV)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/texas_local_pairwise_probe_v1"))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_TEXAS_THRESHOLDS)
    parser.add_argument("--support-weights", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06])
    parser.add_argument("--veto-weights", nargs="+", type=float, default=[0.0, 0.02, 0.04, 0.06, 0.08])
    parser.add_argument("--support-score-floor", type=float, default=0.70)
    parser.add_argument("--support-inlier-floor", type=int, default=8)
    parser.add_argument("--veto-score-ceiling", type=float, default=0.40)
    parser.add_argument("--veto-inlier-ceiling", type=int, default=4)
    parser.add_argument("--patch-support-gray-corr-floor", type=float, default=0.45)
    parser.add_argument("--patch-support-absdiff-ceiling", type=float, default=0.18)
    parser.add_argument("--patch-support-iou-floor", type=float, default=0.10)
    parser.add_argument("--patch-veto-gray-corr-ceiling", type=float, default=0.10)
    parser.add_argument("--patch-veto-absdiff-floor", type=float, default=0.24)
    parser.add_argument("--patch-veto-iou-ceiling", type=float, default=0.04)
    parser.add_argument("--model-backend", choices=["auto", "xgboost", "sklearn"], default="auto")
    parser.add_argument("--blend-scales", nargs="+", type=float, default=[0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    artifacts = load_texas_local_artifacts(
        repo_root=repo_root,
        experiment_dir=args.experiment_dir,
        teacher_source_dir=args.teacher_source_dir,
        top_k=int(args.top_k),
    )
    pair_df = merge_texas_black_pattern_scores(
        repo_root=repo_root,
        pair_df=artifacts.candidate_pair_df,
        metadata_df=artifacts.metadata_df,
        local_match_csv=args.local_match_csv,
    )
    pair_df = merge_texas_black_patch_scores(
        repo_root=repo_root,
        pair_df=pair_df,
    )
    pair_df = enrich_texas_pair_df(
        pair_df=pair_df,
        metadata_df=artifacts.metadata_df,
        route_score_matrix=artifacts.route_score_matrix,
        route_rank_matrix=artifacts.route_rank_matrix,
        teacher_anchor_labels=artifacts.teacher_anchor_labels,
        support_score_floor=float(args.support_score_floor),
        support_inlier_floor=int(args.support_inlier_floor),
        veto_score_ceiling=float(args.veto_score_ceiling),
        veto_inlier_ceiling=int(args.veto_inlier_ceiling),
        patch_support_gray_corr_floor=float(args.patch_support_gray_corr_floor),
        patch_support_absdiff_ceiling=float(args.patch_support_absdiff_ceiling),
        patch_support_iou_floor=float(args.patch_support_iou_floor),
        patch_veto_gray_corr_ceiling=float(args.patch_veto_gray_corr_ceiling),
        patch_veto_absdiff_floor=float(args.patch_veto_absdiff_floor),
        patch_veto_iou_ceiling=float(args.patch_veto_iou_ceiling),
    )
    pair_df.to_csv(tables_dir / "pair_features_v1.csv", index=False)

    baseline_summary_df, baseline_prediction_df = evaluate_texas_thresholds_from_score_matrix(
        metadata_df=artifacts.metadata_df,
        score_matrix=artifacts.route_score_matrix,
        thresholds=[float(value) for value in args.thresholds],
        candidate_pair_df=artifacts.candidate_pair_df,
        teacher_anchor_labels=artifacts.teacher_anchor_labels,
        teacher_topk_indices=artifacts.teacher_topk_indices,
        top_k=int(args.top_k),
    )
    baseline_best = pick_best_proxy_row(baseline_summary_df)
    baseline_summary_df.to_csv(tables_dir / "baseline_threshold_sweep_v1.csv", index=False)
    baseline_prediction_df.to_csv(tables_dir / "baseline_threshold_predictions_v1.csv", index=False)

    rerank_rows: list[dict[str, object]] = []
    best_rerank_score_matrix = artifacts.route_score_matrix.copy()
    best_rerank_summary_df = baseline_summary_df.copy()
    best_rerank_prediction_df = baseline_prediction_df.copy()
    best_rerank_row = baseline_best.copy()
    best_rerank_support = 0.0
    best_rerank_veto = 0.0
    for support_weight in [float(value) for value in args.support_weights]:
        for veto_weight in [float(value) for value in args.veto_weights]:
            reranked_score = apply_texas_local_rerank(
                global_score_matrix=artifacts.route_score_matrix,
                pair_df=pair_df,
                support_weight=float(support_weight),
                veto_weight=float(veto_weight),
            )
            summary_df, prediction_df = evaluate_texas_thresholds_from_score_matrix(
                metadata_df=artifacts.metadata_df,
                score_matrix=reranked_score,
                thresholds=[float(value) for value in args.thresholds],
                candidate_pair_df=artifacts.candidate_pair_df,
                teacher_anchor_labels=artifacts.teacher_anchor_labels,
                teacher_topk_indices=artifacts.teacher_topk_indices,
                top_k=int(args.top_k),
            )
            best_row = pick_best_proxy_row(summary_df)
            rerank_rows.append(
                {
                    "support_weight": float(support_weight),
                    "veto_weight": float(veto_weight),
                    "best_threshold": float(best_row["threshold"]),
                    "proxy_score": float(best_row["proxy_score"]),
                    "seed_pair_agreement": float(best_row["seed_pair_agreement"]),
                    "mutual_topk_pair_keep_ratio": float(best_row["mutual_topk_pair_keep_ratio"]),
                    "seed_recall_at_1": float(best_row["seed_recall_at_1"]),
                    "clusters": int(best_row["clusters"]),
                    "largest_cluster_size": int(best_row["largest_cluster_size"]),
                }
            )
            current_key = (
                float(best_row["proxy_score"]),
                float(best_row["seed_pair_agreement"]),
                float(best_row["mutual_topk_pair_keep_ratio"]),
                -float(best_row["cluster_delta_vs_teacher_anchor"]),
                -float(best_row["largest_cluster_size"]),
            )
            best_key = (
                float(best_rerank_row["proxy_score"]),
                float(best_rerank_row["seed_pair_agreement"]),
                float(best_rerank_row["mutual_topk_pair_keep_ratio"]),
                -float(best_rerank_row["cluster_delta_vs_teacher_anchor"]),
                -float(best_rerank_row["largest_cluster_size"]),
            )
            if current_key > best_key:
                best_rerank_score_matrix = reranked_score.copy()
                best_rerank_summary_df = summary_df.copy()
                best_rerank_prediction_df = prediction_df.copy()
                best_rerank_row = best_row.copy()
                best_rerank_support = float(support_weight)
                best_rerank_veto = float(veto_weight)
    rerank_grid_df = pd.DataFrame(rerank_rows).sort_values(
        ["proxy_score", "seed_pair_agreement", "mutual_topk_pair_keep_ratio", "veto_weight", "support_weight"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    rerank_grid_df.to_csv(tables_dir / "rerank_grid_v1.csv", index=False)
    best_rerank_summary_df.to_csv(tables_dir / "rerank_threshold_sweep_v1.csv", index=False)
    best_rerank_prediction_df.to_csv(tables_dir / "rerank_threshold_predictions_v1.csv", index=False)

    rerank_positive = float(best_rerank_row["proxy_score"]) > float(baseline_best["proxy_score"])
    xgb_grid_df = pd.DataFrame()
    xgb_prediction_df = pd.DataFrame()
    xgb_summary_df = pd.DataFrame()
    xgb_best_row = None
    model_backend = ""
    train_pair_df = pair_df[pair_df["both_seeded"].astype(int).eq(1)].copy().reset_index(drop=True)
    if rerank_positive and not train_pair_df.empty:
        train_pair_df["pair_target"] = train_pair_df["same_seed_cluster"].astype(int)
        train_pair_df = train_pair_df[train_pair_df["pair_target"].isin([0, 1])].copy().reset_index(drop=True)
        positive_count = int(train_pair_df["pair_target"].eq(1).sum())
        negative_count = int(train_pair_df["pair_target"].eq(0).sum())
        if positive_count >= 8 and negative_count >= 8:
            feature_columns = [column for column in TEXAS_XGB_FEATURE_COLUMNS if column in pair_df.columns]
            model, model_backend = _build_pairwise_model(backend=str(args.model_backend), split_seed=int(args.split_seed))
            train_x = train_pair_df.loc[:, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
            train_y = train_pair_df["pair_target"].to_numpy(dtype=np.int32)
            train_w = _balanced_sample_weight(train_y)
            model.fit(train_x, train_y, sample_weight=train_w)

            full_x = pair_df.loc[:, feature_columns].fillna(0.0).to_numpy(dtype=np.float32)
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(full_x)[:, 1].astype(np.float32)
            else:
                probabilities = model.decision_function(full_x).astype(np.float32)
                probabilities = 1.0 / (1.0 + np.exp(-probabilities))
            pair_df["xgb_same_identity_prob"] = np.round(probabilities, 6)
            pair_df.to_csv(tables_dir / "pair_features_with_xgb_v1.csv", index=False)

            xgb_rows: list[dict[str, object]] = []
            best_xgb_score_matrix = best_rerank_score_matrix.copy()
            best_xgb_prediction_df = best_rerank_prediction_df.copy()
            best_xgb_summary_df = best_rerank_summary_df.copy()
            best_xgb_row_local = best_rerank_row.copy()
            best_blend_scale = 0.0
            for blend_scale in [float(value) for value in args.blend_scales]:
                fused_score = apply_pair_probability_residual(
                    base_score_matrix=best_rerank_score_matrix,
                    pair_df=pair_df,
                    probability_col="xgb_same_identity_prob",
                    blend_scale=float(blend_scale),
                )
                summary_df, prediction_df = evaluate_texas_thresholds_from_score_matrix(
                    metadata_df=artifacts.metadata_df,
                    score_matrix=fused_score,
                    thresholds=[float(value) for value in args.thresholds],
                    candidate_pair_df=artifacts.candidate_pair_df,
                    teacher_anchor_labels=artifacts.teacher_anchor_labels,
                    teacher_topk_indices=artifacts.teacher_topk_indices,
                    top_k=int(args.top_k),
                )
                best_row = pick_best_proxy_row(summary_df)
                xgb_rows.append(
                    {
                        "blend_scale": float(blend_scale),
                        "best_threshold": float(best_row["threshold"]),
                        "proxy_score": float(best_row["proxy_score"]),
                        "seed_pair_agreement": float(best_row["seed_pair_agreement"]),
                        "mutual_topk_pair_keep_ratio": float(best_row["mutual_topk_pair_keep_ratio"]),
                        "seed_recall_at_1": float(best_row["seed_recall_at_1"]),
                        "clusters": int(best_row["clusters"]),
                        "largest_cluster_size": int(best_row["largest_cluster_size"]),
                    }
                )
                current_key = (
                    float(best_row["proxy_score"]),
                    float(best_row["seed_pair_agreement"]),
                    float(best_row["mutual_topk_pair_keep_ratio"]),
                    -float(best_row["cluster_delta_vs_teacher_anchor"]),
                    -float(best_row["largest_cluster_size"]),
                )
                best_key = (
                    float(best_xgb_row_local["proxy_score"]),
                    float(best_xgb_row_local["seed_pair_agreement"]),
                    float(best_xgb_row_local["mutual_topk_pair_keep_ratio"]),
                    -float(best_xgb_row_local["cluster_delta_vs_teacher_anchor"]),
                    -float(best_xgb_row_local["largest_cluster_size"]),
                )
                if current_key > best_key:
                    best_xgb_score_matrix = fused_score.copy()
                    best_xgb_prediction_df = prediction_df.copy()
                    best_xgb_summary_df = summary_df.copy()
                    best_xgb_row_local = best_row.copy()
                    best_blend_scale = float(blend_scale)
            xgb_grid_df = pd.DataFrame(xgb_rows).sort_values(
                ["proxy_score", "seed_pair_agreement", "mutual_topk_pair_keep_ratio", "blend_scale"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)
            xgb_grid_df.to_csv(tables_dir / "xgb_grid_v1.csv", index=False)
            xgb_prediction_df = best_xgb_prediction_df.copy()
            xgb_summary_df = best_xgb_summary_df.copy()
            xgb_summary_df.to_csv(tables_dir / "xgb_threshold_sweep_v1.csv", index=False)
            xgb_prediction_df.to_csv(tables_dir / "xgb_threshold_predictions_v1.csv", index=False)
            xgb_best_row = dict(best_xgb_row_local)
            xgb_best_row["blend_scale"] = best_blend_scale
            xgb_best_row["backend"] = model_backend
            xgb_best_row["train_pairs"] = int(len(train_pair_df))
            xgb_best_row["train_positive_pairs"] = positive_count
            xgb_best_row["train_negative_pairs"] = negative_count

    summary_payload = {
        "baseline_best": baseline_best.to_dict(),
        "rerank_positive": bool(rerank_positive),
        "best_rerank_support_weight": float(best_rerank_support),
        "best_rerank_veto_weight": float(best_rerank_veto),
        "best_rerank": best_rerank_row.to_dict(),
        "xgb_best": xgb_best_row,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Texas Local Pairwise Probe v1",
        "",
        "- Goal: test whether `black-pattern ORB` should first enter Texas as a soft local rerank signal, and only then as a pairwise `XGBoost` feature branch.",
        f"- Experiment dir: `{args.experiment_dir}`",
        f"- Local match csv: `{args.local_match_csv}`",
        f"- Thresholds: `{[float(value) for value in args.thresholds]}`",
        f"- Local rule: support if `score >= {float(args.support_score_floor)}` and `inliers >= {int(args.support_inlier_floor)}`; veto if `score <= {float(args.veto_score_ceiling)}` and `inliers <= {int(args.veto_inlier_ceiling)}`.",
        "",
        "## Baseline Best",
        "",
        _to_markdown_table(pd.DataFrame([baseline_best.to_dict()])),
        "",
        "## Rerank Grid",
        "",
        _to_markdown_table(rerank_grid_df, limit=20),
        "",
        f"- Rerank positive vs baseline: `{bool(rerank_positive)}`",
        f"- Best rerank config: `support_weight={float(best_rerank_support):.4f}`, `veto_weight={float(best_rerank_veto):.4f}`",
        "",
    ]
    if xgb_best_row is not None:
        lines.extend(
            [
                "## XGBoost Grid",
                "",
                _to_markdown_table(xgb_grid_df, limit=20),
                "",
                "## XGBoost Best",
                "",
                _to_markdown_table(pd.DataFrame([xgb_best_row])),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## XGBoost Stage",
                "",
                "- Skipped because rerank did not beat the baseline proxy, or seeded pair supervision was insufficient.",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[texas_local_pairwise_probe] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
