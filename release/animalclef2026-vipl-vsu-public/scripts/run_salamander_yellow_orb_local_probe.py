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
DEFAULT_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_yellow_orb_local_v1")
DEFAULT_THRESHOLD_CANDIDATES = [0.20, 0.23, 0.25, 0.27, 0.30]
DEFAULT_CHOSEN_THRESHOLD = 0.25
SALAMANDER_DATASET = "SalamanderID2025"


def _load_route_bundle(route_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    test_df = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    for frame in [val_df, test_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        frame["dataset"] = frame["dataset"].astype(str)
        if "identity" in frame.columns:
            frame["identity"] = frame["identity"].fillna("").astype(str)
    val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy").astype(np.float32)
    return val_df, val_embeddings, test_df, test_embeddings


def _apply_pair_probability_as_score(
    *,
    base_score: np.ndarray,
    pair_df: pd.DataFrame,
    probability_col: str = "xgb_same_identity_prob",
    blend_scale: float = 1.0,
) -> np.ndarray:
    fused = np.asarray(base_score, dtype=np.float32).copy()
    for row in pair_df.itertuples(index=False):
        left_index = int(getattr(row, "left_index"))
        right_index = int(getattr(row, "right_index"))
        probability = float(getattr(row, probability_col))
        base_value = float(base_score[left_index, right_index])
        score = min(1.0, base_value + float(blend_scale) * probability * (1.0 - base_value))
        fused[left_index, right_index] = score
        fused[right_index, left_index] = score
    np.fill_diagonal(fused, 1.0)
    return fused


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    ranking_columns = [column for column in ["ari", "pairwise_f1", "nmi", "threshold"] if column in df.columns]
    ascending = [False, False, False, True][: len(ranking_columns)]
    return df.sort_values(ranking_columns, ascending=ascending).iloc[0]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.salamander_yellow_orb_local import (
        YELLOW_FOCUS_PATH_COLUMN,
        YELLOW_ORB_LOCAL_ANALYSIS_NAME,
        apply_yellow_orb_local_penalty_as_score,
        build_markdown_report,
        build_patch_pair_features,
        build_threshold_delta_table,
        build_yellow_focus_manifest,
        compile_yellow_orb_local_decisions,
        merge_yellow_orb_local_pair_features,
        summarize_patch_pair_features,
        summarize_yellow_focus_manifest,
        summarize_yellow_orb_local_decisions,
    )
    from animalclef_analysis.sam_orb_veto import (
        build_masked_aligned_roi_manifest,
        build_view_local_match_table,
        summarize_roi_manifest,
    )
    from animalclef_analysis.transductive_seed_refinement import run_score_threshold_sweep

    parser = argparse.ArgumentParser(description="Run the Salamander yellow-guided ORB local veto probe.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--xgb-variant-dir", type=Path, default=DEFAULT_XGB_VARIANT_DIR)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-candidates", nargs="+", type=float, default=DEFAULT_THRESHOLD_CANDIDATES)
    parser.add_argument("--chosen-threshold", type=float, default=DEFAULT_CHOSEN_THRESHOLD)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--orb-features", type=int, default=1024)
    parser.add_argument("--orb-max-side", type=int, default=512)
    parser.add_argument("--fast-threshold", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--local-matcher", type=str, default="orb")
    parser.add_argument("--alignment-min-foreground-pixels", type=int, default=512)
    parser.add_argument("--alignment-min-axis-confidence", type=float, default=0.20)
    parser.add_argument("--hard-veto-score-cap", type=float, default=0.02)
    parser.add_argument("--soft-veto-score-scale", type=float, default=0.70)
    parser.add_argument("--max-val-pairs", type=int)
    parser.add_argument("--max-test-pairs", type=int)
    args = parser.parse_args()

    route_dir = (repo_root / args.route_dir).resolve() if not args.route_dir.is_absolute() else args.route_dir.resolve()
    xgb_variant_dir = (
        (repo_root / args.xgb_variant_dir).resolve()
        if not args.xgb_variant_dir.is_absolute()
        else args.xgb_variant_dir.resolve()
    )
    manifest_root = (
        (repo_root / args.manifest_root).resolve()
        if not args.manifest_root.is_absolute()
        else args.manifest_root.resolve()
    )
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_df, route_val_embeddings, test_df, route_test_embeddings = _load_route_bundle(route_dir=route_dir)
    val_df = val_df[val_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    test_df = test_df[test_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    route_val_score = cosine_score_matrix(route_val_embeddings)
    route_test_score = cosine_score_matrix(route_test_embeddings)

    enriched_df = pd.read_csv(manifest_root / "tables" / "metadata_enriched_v1.csv")
    enriched_df["image_id"] = enriched_df["image_id"].astype(str)
    enriched_df["dataset"] = enriched_df["dataset"].astype(str)
    enriched_df = enriched_df[enriched_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)

    val_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "val_pair_features_v1.csv")
    test_pair_df = pd.read_csv(xgb_variant_dir / "tables" / "test_pair_features_v1.csv")
    for frame in [val_pair_df, test_pair_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)
        frame["dataset"] = SALAMANDER_DATASET
    if args.max_val_pairs is not None:
        val_pair_df = val_pair_df.head(int(args.max_val_pairs)).reset_index(drop=True)
    if args.max_test_pairs is not None:
        test_pair_df = test_pair_df.head(int(args.max_test_pairs)).reset_index(drop=True)

    required_val_ids = set(val_pair_df["image_id"].tolist()) | set(val_pair_df["neighbor_image_id"].tolist())
    required_test_ids = set(test_pair_df["image_id"].tolist()) | set(test_pair_df["neighbor_image_id"].tolist())
    roi_reference_df = pd.concat(
        [
            val_df[val_df["image_id"].isin(required_val_ids)].copy(),
            test_df[test_df["image_id"].isin(required_test_ids)].copy(),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["image_id", "dataset"])
    roi_manifest_df = build_masked_aligned_roi_manifest(
        reference_df=roi_reference_df,
        enriched_df=enriched_df,
        repo_root=repo_root,
        output_dir=output_dir,
        alignment_min_foreground_pixels=int(args.alignment_min_foreground_pixels),
        alignment_min_axis_confidence=float(args.alignment_min_axis_confidence),
    )
    roi_manifest_df.to_csv(tables_dir / "image_roi_manifest_v1.csv", index=False)
    roi_summary_df = summarize_roi_manifest(roi_manifest_df=roi_manifest_df)
    roi_summary_df.to_csv(tables_dir / "image_roi_summary_v1.csv", index=False)

    focus_df = build_yellow_focus_manifest(
        roi_manifest_df=roi_manifest_df,
        repo_root=repo_root,
        output_dir=output_dir,
    )
    focus_df.to_csv(tables_dir / "yellow_focus_manifest_v1.csv", index=False)
    focus_summary_df = summarize_yellow_focus_manifest(focus_df=focus_df)
    focus_summary_df.to_csv(tables_dir / "yellow_focus_summary_v1.csv", index=False)

    val_focus_reference_df = val_df.merge(
        focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
    )
    test_focus_reference_df = test_df.merge(
        focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
    )
    val_yellow_roi_local_df = build_view_local_match_table(
        reference_df=val_focus_reference_df,
        pair_df=val_pair_df,
        repo_root=repo_root,
        path_column=YELLOW_FOCUS_PATH_COLUMN,
        nfeatures=int(args.orb_features),
        max_side=int(args.orb_max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        ratio_test=float(args.ratio_test),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
        local_matcher=str(args.local_matcher),
        prefix="yellow_roi",
    )
    test_yellow_roi_local_df = build_view_local_match_table(
        reference_df=test_focus_reference_df,
        pair_df=test_pair_df,
        repo_root=repo_root,
        path_column=YELLOW_FOCUS_PATH_COLUMN,
        nfeatures=int(args.orb_features),
        max_side=int(args.orb_max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        ratio_test=float(args.ratio_test),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
        local_matcher=str(args.local_matcher),
        prefix="yellow_roi",
    )
    val_yellow_roi_local_df.to_csv(tables_dir / "val_yellow_roi_local_scores_v1.csv", index=False)
    test_yellow_roi_local_df.to_csv(tables_dir / "test_yellow_roi_local_scores_v1.csv", index=False)

    val_patch_pair_df = build_patch_pair_features(pair_df=val_pair_df, focus_df=focus_df, repo_root=repo_root)
    test_patch_pair_df = build_patch_pair_features(pair_df=test_pair_df, focus_df=focus_df, repo_root=repo_root)
    val_patch_pair_df.to_csv(tables_dir / "val_patch_pair_features_v1.csv", index=False)
    test_patch_pair_df.to_csv(tables_dir / "test_patch_pair_features_v1.csv", index=False)

    val_feature_df = merge_yellow_orb_local_pair_features(
        base_pair_df=val_pair_df,
        yellow_roi_local_df=val_yellow_roi_local_df,
        patch_pair_df=val_patch_pair_df,
    )
    test_feature_df = merge_yellow_orb_local_pair_features(
        base_pair_df=test_pair_df,
        yellow_roi_local_df=test_yellow_roi_local_df,
        patch_pair_df=test_patch_pair_df,
    )
    val_decision_df = compile_yellow_orb_local_decisions(pair_feature_df=val_feature_df, focus_df=focus_df)
    test_decision_df = compile_yellow_orb_local_decisions(pair_feature_df=test_feature_df, focus_df=focus_df)
    val_feature_df.to_csv(tables_dir / "val_yellow_orb_features_v1.csv", index=False)
    test_feature_df.to_csv(tables_dir / "test_yellow_orb_features_v1.csv", index=False)
    val_decision_df.to_csv(tables_dir / "val_yellow_orb_decisions_v1.csv", index=False)
    test_decision_df.to_csv(tables_dir / "test_yellow_orb_decisions_v1.csv", index=False)

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
    yellow_orb_val_score = apply_yellow_orb_local_penalty_as_score(
        base_score=boosted_val_score,
        decision_df=val_decision_df,
        hard_veto_score_cap=float(args.hard_veto_score_cap),
        soft_veto_score_scale=float(args.soft_veto_score_scale),
    )
    yellow_orb_test_score = apply_yellow_orb_local_penalty_as_score(
        base_score=boosted_test_score,
        decision_df=test_decision_df,
        hard_veto_score_cap=float(args.hard_veto_score_cap),
        soft_veto_score_scale=float(args.soft_veto_score_scale),
    )

    threshold_candidates = [float(value) for value in args.threshold_candidates]
    base_val_sweep_df, _ = run_score_threshold_sweep(df=val_df, score_matrix=boosted_val_score, thresholds=threshold_candidates)
    yellow_orb_val_sweep_df, _ = run_score_threshold_sweep(df=val_df, score_matrix=yellow_orb_val_score, thresholds=threshold_candidates)
    base_val_sweep_df["variant"] = "baseline_boosted"
    yellow_orb_val_sweep_df["variant"] = "yellow_orb_local"
    base_val_sweep_df.to_csv(tables_dir / "val_baseline_thresholds_v1.csv", index=False)
    yellow_orb_val_sweep_df.to_csv(tables_dir / "val_yellow_orb_thresholds_v1.csv", index=False)
    threshold_delta_df = build_threshold_delta_table(baseline_df=base_val_sweep_df, veto_df=yellow_orb_val_sweep_df)
    threshold_delta_df.to_csv(tables_dir / "val_threshold_delta_v1.csv", index=False)

    base_best_row = _pick_best_row(base_val_sweep_df)
    yellow_orb_best_row = _pick_best_row(yellow_orb_val_sweep_df)
    best_rows_df = pd.DataFrame([base_best_row, yellow_orb_best_row]).reset_index(drop=True)
    best_rows_df.to_csv(tables_dir / "val_best_rows_v1.csv", index=False)

    test_shape_rows = []
    for variant_name, score_matrix in [("baseline_boosted", boosted_test_score), ("yellow_orb_local", yellow_orb_test_score)]:
        sweep_df, prediction_df = run_score_threshold_sweep(df=test_df, score_matrix=score_matrix, thresholds=threshold_candidates)
        sweep_df["variant"] = variant_name
        sweep_df.to_csv(tables_dir / f"test_{variant_name}_thresholds_v1.csv", index=False)
        chosen_pred_df = prediction_df[prediction_df["threshold"] == float(args.chosen_threshold)].copy().reset_index(drop=True)
        if chosen_pred_df.empty:
            continue
        chosen_pred_df["variant"] = variant_name
        chosen_pred_df.to_csv(
            tables_dir / f"test_{variant_name}_predictions_t{str(args.chosen_threshold).replace('.', 'p')}_v1.csv",
            index=False,
        )
        counts = chosen_pred_df["pred_cluster_id"].value_counts()
        test_shape_rows.append(
            {
                "variant": variant_name,
                "threshold": float(args.chosen_threshold),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "largest_cluster": int(counts.max()) if len(counts) else 0,
            }
        )
    test_shape_df = pd.DataFrame(test_shape_rows).sort_values("variant").reset_index(drop=True)
    test_shape_df.to_csv(tables_dir / "test_shape_summary_v1.csv", index=False)

    val_patch_summary_df = summarize_patch_pair_features(pair_df=val_patch_pair_df)
    test_patch_summary_df = summarize_patch_pair_features(pair_df=test_patch_pair_df)
    val_patch_summary_df.to_csv(tables_dir / "val_patch_summary_v1.csv", index=False)
    test_patch_summary_df.to_csv(tables_dir / "test_patch_summary_v1.csv", index=False)
    val_decision_summary_df = summarize_yellow_orb_local_decisions(decision_df=val_decision_df)
    test_decision_summary_df = summarize_yellow_orb_local_decisions(decision_df=test_decision_df)
    val_decision_summary_df.to_csv(tables_dir / "val_decision_summary_v1.csv", index=False)
    test_decision_summary_df.to_csv(tables_dir / "test_decision_summary_v1.csv", index=False)

    config = {
        "analysis_id": YELLOW_ORB_LOCAL_ANALYSIS_NAME,
        "route_dir": str(route_dir),
        "xgb_variant_dir": str(xgb_variant_dir),
        "threshold_candidates": threshold_candidates,
        "chosen_threshold": float(args.chosen_threshold),
        "hard_veto_score_cap": float(args.hard_veto_score_cap),
        "soft_veto_score_scale": float(args.soft_veto_score_scale),
    }
    (tables_dir / "config_v1.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    build_markdown_report(
        output_path=reports_dir / "summary.md",
        config=config,
        roi_summary_df=roi_summary_df,
        focus_summary_df=focus_summary_df,
        val_patch_summary_df=val_patch_summary_df,
        test_patch_summary_df=test_patch_summary_df,
        val_decision_summary_df=val_decision_summary_df,
        test_decision_summary_df=test_decision_summary_df,
        threshold_delta_df=threshold_delta_df,
        best_rows_df=best_rows_df,
        test_shape_df=test_shape_df,
    )

    print(f"[yellow_orb_local] focus_manifest: {tables_dir / 'yellow_focus_manifest_v1.csv'}")
    print(f"[yellow_orb_local] val_decisions: {tables_dir / 'val_yellow_orb_decisions_v1.csv'}")
    print(f"[yellow_orb_local] test_decisions: {tables_dir / 'test_yellow_orb_decisions_v1.csv'}")
    print(f"[yellow_orb_local] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
