#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_mixed_baseline_v2/tables/test_predictions_v1.csv")
DEFAULT_FUSION_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_fusion_v1")
DEFAULT_ORB_SOURCE_DIR = Path("artifacts/descriptor_baselines/orb_rerank_v1")
DEFAULT_LOCAL_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["ari", "pairwise_f1", "nmi", "local_weight", "threshold"],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def _resolve_manifest_paths(df: pd.DataFrame, repo_root: Path, path_column: str, resolver) -> pd.DataFrame:
    resolved = df.copy().reset_index(drop=True)
    resolved[path_column] = [resolver(row, repo_root=repo_root) for _, row in resolved.iterrows()]
    return resolved


def _filter_salamander(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame["dataset"] == SALAMANDER_DATASET].copy().reset_index(drop=True)
    result["image_id"] = result["image_id"].astype(str)
    if "identity" in result.columns:
        result["identity"] = result["identity"].fillna("").astype(str)
    return result


def _reorder_to_reference(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, embeddings: np.ndarray, path_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    if len(candidate_df) != len(embeddings):
        raise ValueError(f"Metadata/embedding mismatch: df={len(candidate_df)} vs emb={len(embeddings)}")
    ref = reference_df[["image_id", "dataset", path_column]].copy().reset_index(drop=True)
    cand = candidate_df[["image_id", "dataset", path_column]].copy().reset_index(drop=True)
    ref["image_id"] = ref["image_id"].astype(str)
    cand["image_id"] = cand["image_id"].astype(str)
    merged = ref.merge(
        cand.assign(_row=np.arange(len(cand), dtype=np.int32)),
        on=["image_id", "dataset"],
        how="left",
        validate="one_to_one",
    )
    if merged["_row"].isna().any():
        missing = merged.loc[merged["_row"].isna(), ["image_id", "dataset"]].head(5).to_dict(orient="records")
        raise ValueError(f"Missing candidate rows when aligning embeddings: {missing}")
    index = merged["_row"].astype(int).to_numpy()
    return candidate_df.iloc[index].reset_index(drop=True).copy(), embeddings[index]


def _build_route_description(route_name: str, fused: bool, embedding_dim: int) -> str:
    if fused:
        return (
            f"supervised `Miew` student (`B x 3 x 440 x 440 -> B x 2152 -> B x 512`) "
            f"+ frozen fusion (`Mega 1536 + Miew 2152 -> 3688`), "
            f"weighted concat + L2 normalize -> `{embedding_dim}-d`, then ORB rerank"
        )
    return (
        f"supervised `Miew` student (`B x 3 x 440 x 440 -> B x 2152 -> B x 512`), "
        f"then ORB rerank, final embedding dim `{embedding_dim}`"
    )


def _metric_key(row: pd.Series) -> tuple[float, float, float, float, float]:
    return (
        float(row["ari"]),
        float(row["pairwise_f1"]),
        float(row["nmi"]),
        -float(row["local_weight"]),
        -float(row["threshold"]),
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import (
        PATH_COLUMN,
        apply_thresholds_to_df,
        build_submission,
        dataframe_to_markdown_table,
        ensure_metadata_alignment,
        fuse_embedding_blocks,
        load_cached_embedding_bundle,
        recall_at_k,
    )
    from animalclef_analysis.orb_rerank_baseline import (
        apply_local_rerank,
        build_local_match_table,
        build_top1_transition_table,
        build_topk_pair_index,
        cosine_score_matrix,
        create_qualitative_outputs,
        evaluate_threshold_sweep_from_score_matrix,
        extract_orb_features,
        resolve_existing_image_rel_path,
    )
    from animalclef_analysis.submission_baseline import _cluster_single_dataset_from_score_matrix, _load_supervised_model_from_checkpoint
    from animalclef_analysis.supervised_training import extract_student_embeddings
    from animalclef_analysis.view_manifests import get_default_manifest_paths

    parser = argparse.ArgumentParser(description="Evaluate Salamander supervised + ORB variants and build a submission candidate.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--route-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-weights", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--fusion-source-dir", type=Path)
    parser.add_argument("--student-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--orb-source-dir", type=Path, default=DEFAULT_ORB_SOURCE_DIR)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checkpoint_path = args.checkpoint_path.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    qualitative_dir = output_dir / "qualitative"
    embeddings_dir = output_dir / "embeddings"
    for path in [output_dir, tables_dir, reports_dir, qualitative_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    local_weights = args.local_weights or DEFAULT_LOCAL_WEIGHTS
    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    fused_mode = args.fusion_source_dir is not None
    sample_submission_path = args.sample_submission_path.resolve() if args.sample_submission_path else repo_root / "sample_submission.csv"
    if args.test_manifest_path is not None:
        test_manifest_path = args.test_manifest_path.resolve()
    else:
        _train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
    orb_source_dir = args.orb_source_dir.resolve() if args.orb_source_dir else (repo_root / DEFAULT_ORB_SOURCE_DIR).resolve()

    experiment_dir = checkpoint_path.parents[1]
    val_manifest_path = experiment_dir / "tables" / "val_manifest_v1.csv"
    val_df = pd.read_csv(val_manifest_path)
    val_df["image_id"] = val_df["image_id"].astype(str)
    val_df["identity"] = val_df["identity"].fillna("").astype(str)
    val_df = _resolve_manifest_paths(val_df, repo_root, PATH_COLUMN, resolve_existing_image_rel_path)
    salamander_val_df = _filter_salamander(val_df)

    test_df = pd.read_csv(test_manifest_path)
    test_df["image_id"] = test_df["image_id"].astype(str)
    if "identity" in test_df.columns:
        test_df["identity"] = test_df["identity"].fillna("").astype(str)
    test_df = _resolve_manifest_paths(test_df, repo_root, PATH_COLUMN, resolve_existing_image_rel_path)
    salamander_test_df = _filter_salamander(test_df)

    model, spec, checkpoint_config, checkpoint = _load_supervised_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    student_val_embeddings = extract_student_embeddings(
        df=salamander_val_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=args.device,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    student_test_embeddings = extract_student_embeddings(
        df=salamander_test_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=args.device,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )

    component_rows = [
        {
            "component": "student",
            "source": str(checkpoint_path),
            "weight": float(args.student_weight),
            "embedding_dim": int(student_val_embeddings.shape[1]),
        }
    ]

    val_embeddings = student_val_embeddings
    test_embeddings = student_test_embeddings
    if fused_mode:
        fusion_source_dir = args.fusion_source_dir.resolve()
        fusion_bundle = load_cached_embedding_bundle(source_dir=fusion_source_dir, name="fusion", weight=float(args.fusion_weight))
        fusion_val_df = _filter_salamander(fusion_bundle.val_df)
        fusion_test_df = _filter_salamander(fusion_bundle.test_df)
        fusion_val_emb = fusion_bundle.val_embeddings[(fusion_bundle.val_df["dataset"] == SALAMANDER_DATASET).to_numpy()]
        fusion_test_emb = fusion_bundle.test_embeddings[(fusion_bundle.test_df["dataset"] == SALAMANDER_DATASET).to_numpy()]
        fusion_val_df, fusion_val_emb = _reorder_to_reference(salamander_val_df, fusion_val_df, fusion_val_emb, PATH_COLUMN)
        fusion_test_df, fusion_test_emb = _reorder_to_reference(salamander_test_df, fusion_test_df, fusion_test_emb, PATH_COLUMN)
        fusion_val_df[PATH_COLUMN] = salamander_val_df[PATH_COLUMN].to_numpy()
        fusion_test_df[PATH_COLUMN] = salamander_test_df[PATH_COLUMN].to_numpy()
        ensure_metadata_alignment(
            reference_df=salamander_val_df[["image_id", "dataset", "identity", PATH_COLUMN]],
            candidate_df=fusion_val_df[["image_id", "dataset", "identity", PATH_COLUMN]],
            split_name="val",
            reference_name="student_val",
            candidate_name="fusion_val",
        )
        ensure_metadata_alignment(
            reference_df=salamander_test_df[["image_id", "dataset", PATH_COLUMN]],
            candidate_df=fusion_test_df[["image_id", "dataset", PATH_COLUMN]],
            split_name="test",
            reference_name="student_test",
            candidate_name="fusion_test",
        )
        val_embeddings = fuse_embedding_blocks(
            [student_val_embeddings, fusion_val_emb],
            weights=[float(args.student_weight), float(args.fusion_weight)],
        )
        test_embeddings = fuse_embedding_blocks(
            [student_test_embeddings, fusion_test_emb],
            weights=[float(args.student_weight), float(args.fusion_weight)],
        )
        component_rows.append(
            {
                "component": "fusion",
                "source": str(fusion_source_dir),
                "weight": float(args.fusion_weight),
                "embedding_dim": int(fusion_val_emb.shape[1]),
            }
        )

    np.save(embeddings_dir / "salamander_val_embeddings.npy", val_embeddings.astype(np.float32))
    np.save(embeddings_dir / "salamander_test_embeddings.npy", test_embeddings.astype(np.float32))
    salamander_val_df.to_csv(embeddings_dir / "salamander_val_metadata.csv", index=False)
    salamander_test_df.to_csv(embeddings_dir / "salamander_test_metadata.csv", index=False)
    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(tables_dir / "component_table_v1.csv", index=False)

    val_score = cosine_score_matrix(val_embeddings)
    val_pair_index = build_topk_pair_index(score_matrix=val_score, top_k=int(args.top_k), query_indices=None)
    val_features = extract_orb_features(
        df=salamander_val_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
    )
    val_pair_df = build_local_match_table(
        df=salamander_val_df,
        features=val_features,
        pair_index=val_pair_index,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    val_pair_df.to_csv(tables_dir / "val_local_match_scores_v1.csv", index=False)
    pd.DataFrame(
        [{"image_id": feature.image_id, "keypoints": feature.point_count, "width": feature.width, "height": feature.height} for feature in val_features]
    ).to_csv(tables_dir / "val_orb_keypoints_v1.csv", index=False)

    summary_rows = []
    best_prediction_df = None
    best_reranked_val_score = None
    for local_weight in local_weights:
        reranked_val_score = apply_local_rerank(
            global_score_matrix=val_score,
            pair_df=val_pair_df,
            local_weight=float(local_weight),
        )
        sweep_df, prediction_df = evaluate_threshold_sweep_from_score_matrix(
            df=salamander_val_df,
            score_matrix=reranked_val_score,
            thresholds=thresholds,
        )
        sweep_df["local_weight"] = float(local_weight)
        prediction_df["local_weight"] = float(local_weight)
        best_row = _pick_best_row(sweep_df)
        summary_rows.append(
            {
                "local_weight": float(local_weight),
                "best_threshold": float(best_row["threshold"]),
                "ari": float(best_row["ari"]),
                "nmi": float(best_row["nmi"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "cluster_count": int(best_row["cluster_count"]),
                "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
            }
        )
        sweep_df.to_csv(tables_dir / f"val_threshold_sweep_w{str(local_weight).replace('.', 'p')}_v1.csv", index=False)
        if best_prediction_df is None:
            best_prediction_df = prediction_df[prediction_df["threshold"] == float(best_row["threshold"])].copy().reset_index(drop=True)
            best_reranked_val_score = reranked_val_score.copy()
            best_meta = best_row.to_dict()
        else:
            current_best = pd.Series(best_meta)
            challenger = best_row
            challenger_key = _metric_key(challenger)
            current_key = _metric_key(current_best)
            if challenger_key > current_key:
                best_prediction_df = prediction_df[prediction_df["threshold"] == float(best_row["threshold"])].copy().reset_index(drop=True)
                best_reranked_val_score = reranked_val_score.copy()
                best_meta = best_row.to_dict()

    if best_prediction_df is None or best_reranked_val_score is None:
        raise RuntimeError("Failed to select a best Salamander rerank configuration")

    summary_df = pd.DataFrame(summary_rows).sort_values(["ari", "pairwise_f1", "nmi", "local_weight"], ascending=[False, False, False, True]).reset_index(drop=True)
    summary_df.to_csv(tables_dir / "val_weight_summary_v1.csv", index=False)
    best_df = pd.DataFrame([best_meta])
    best_df.to_csv(tables_dir / "best_config_v1.csv", index=False)
    best_prediction_df.to_csv(tables_dir / "val_predictions_best_v1.csv", index=False)

    val_transition_df = build_top1_transition_table(salamander_val_df, val_score, best_reranked_val_score)
    val_transition_df.to_csv(tables_dir / "val_top1_transitions_v1.csv", index=False)
    create_qualitative_outputs(
        df=salamander_val_df,
        transition_df=val_transition_df,
        repo_root=repo_root,
        qualitative_dir=qualitative_dir / "val",
    )

    current_baseline_df = pd.read_csv(orb_source_dir / "tables" / "rerank_best_thresholds_v1.csv")
    current_baseline_row = current_baseline_df[current_baseline_df["dataset"] == SALAMANDER_DATASET].iloc[0]
    comparison_df = pd.DataFrame(
        [
            {
                "route": "current_fusion_orb_rerank_v1",
                "threshold": float(current_baseline_row["threshold"]),
                "local_weight": 0.75,
                "ari": float(current_baseline_row["ari"]),
                "nmi": float(current_baseline_row["nmi"]),
                "pairwise_f1": float(current_baseline_row["pairwise_f1"]),
                "cluster_count": int(current_baseline_row["cluster_count"]),
                "singleton_cluster_ratio": float(current_baseline_row["singleton_cluster_ratio"]),
            },
            {
                "route": str(args.route_name),
                "threshold": float(best_meta["threshold"]),
                "local_weight": float(best_meta["local_weight"]),
                "ari": float(best_meta["ari"]),
                "nmi": float(best_meta["nmi"]),
                "pairwise_f1": float(best_meta["pairwise_f1"]),
                "cluster_count": int(best_meta["cluster_count"]),
                "singleton_cluster_ratio": float(best_meta["singleton_cluster_ratio"]),
            },
        ]
    )
    comparison_df["ari_delta_vs_current"] = np.round(comparison_df["ari"] - float(current_baseline_row["ari"]), 6)
    comparison_df["pairwise_f1_delta_vs_current"] = np.round(comparison_df["pairwise_f1"] - float(current_baseline_row["pairwise_f1"]), 6)
    comparison_df.to_csv(tables_dir / "comparison_vs_current_v1.csv", index=False)

    val_recall_df = pd.DataFrame(
        [
            {
                "route": str(args.route_name),
                "recall_at_1": float(recall_at_k(val_embeddings, salamander_val_df["identity"].to_numpy(), k=1)),
                "recall_at_5": float(recall_at_k(val_embeddings, salamander_val_df["identity"].to_numpy(), k=5)),
            }
        ]
    )
    val_recall_df.to_csv(tables_dir / "val_recall_v1.csv", index=False)

    test_score = cosine_score_matrix(test_embeddings)
    test_pair_index = build_topk_pair_index(score_matrix=test_score, top_k=int(args.top_k), query_indices=None)
    test_features = extract_orb_features(
        df=salamander_test_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
    )
    test_pair_df = build_local_match_table(
        df=salamander_test_df.assign(identity=""),
        features=test_features,
        pair_index=test_pair_index,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    test_pair_df.to_csv(tables_dir / "test_local_match_scores_v1.csv", index=False)
    pd.DataFrame(
        [{"image_id": feature.image_id, "keypoints": feature.point_count, "width": feature.width, "height": feature.height} for feature in test_features]
    ).to_csv(tables_dir / "test_orb_keypoints_v1.csv", index=False)
    reranked_test_score = apply_local_rerank(
        global_score_matrix=test_score,
        pair_df=test_pair_df,
        local_weight=float(best_meta["local_weight"]),
    )
    test_pred_df = _cluster_single_dataset_from_score_matrix(
        dataset_df=salamander_test_df,
        score_matrix=reranked_test_score,
        threshold=float(best_meta["threshold"]),
    )
    test_pred_df["route_name"] = str(args.route_name)
    test_pred_df["embedding_dim"] = int(test_embeddings.shape[1])
    test_pred_df["rerank_enabled"] = True
    test_pred_df["local_weight"] = float(best_meta["local_weight"])
    test_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)

    base_pred_df = pd.read_csv(args.base_predictions.resolve())
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    merged_pred_df = pd.concat(
        [base_pred_df[base_pred_df["dataset"] != SALAMANDER_DATASET].copy(), test_pred_df],
        ignore_index=True,
    )
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=sample_submission_path,
        output_path=output_dir / "submission.csv",
    )

    cluster_summary_rows = []
    for dataset, dataset_df in merged_pred_df.groupby("dataset"):
        counts = dataset_df["pred_cluster_id"].value_counts()
        cluster_summary_rows.append(
            {
                "dataset": str(dataset),
                "samples": int(len(dataset_df)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(dataset_df["route_name"].iloc[0]),
                "embedding_dim": int(dataset_df["embedding_dim"].iloc[0]),
                "threshold": float(dataset_df["chosen_threshold"].iloc[0]),
            }
        )
    cluster_summary_df = pd.DataFrame(cluster_summary_rows).sort_values("dataset").reset_index(drop=True)
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    route_summary_df = (
        merged_pred_df[["dataset", "route_name", "embedding_dim", "chosen_threshold", "rerank_enabled", "local_weight"]]
        .drop_duplicates(subset=["dataset"])
        .rename(columns={"chosen_threshold": "threshold"})
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    route_summary_df.to_csv(tables_dir / "route_config_v1.csv", index=False)

    config = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "route_name": str(args.route_name),
        "fused_mode": fused_mode,
        "student_weight": float(args.student_weight),
        "fusion_weight": float(args.fusion_weight) if fused_mode else None,
        "device": str(args.device),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "top_k": int(args.top_k),
        "local_weights": local_weights,
        "thresholds": thresholds,
        "base_predictions": str(args.base_predictions.resolve()),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander Supervised ORB Variant",
        "",
        f"- Route name: `{args.route_name}`",
        f"- Checkpoint path: `{checkpoint_path}`",
        f"- Checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- Fused with frozen fusion: `{fused_mode}`",
        f"- Device: `{args.device}`",
        f"- Eval batch size: `{int(args.eval_batch_size)}`",
        f"- Top-K candidate neighbors: `{int(args.top_k)}`",
        "",
        "## Components",
        "",
        dataframe_to_markdown_table(component_df),
        "",
        "## Best Local Validation Config",
        "",
        dataframe_to_markdown_table(best_df[["local_weight", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]),
        "",
        "## Comparison Vs Current Official Salamander Route",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
        "## Local Weight Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Validation Recall",
        "",
        dataframe_to_markdown_table(val_recall_df),
        "",
        "## Test Route Summary",
        "",
        dataframe_to_markdown_table(route_summary_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Architecture",
        "",
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow for Salamander: `image -> supervised embedding -> optional concat with frozen fusion -> ORB rerank on top-K neighbors -> average-linkage clustering`.",
        f"- Salamander branch: {_build_route_description(str(args.route_name), fused_mode, int(test_embeddings.shape[1]))}.",
        f"- Chosen local rerank weight: `{float(best_meta['local_weight'])}`.",
        f"- Chosen clustering threshold: `{float(best_meta['threshold'])}`.",
        "",
        "## Decision Rule",
        "",
        f"- Current official local target is `fusion + ORB = ARI {float(current_baseline_row['ari']):.4f}`.",
        "- If this route beats that number with a clean single-factor explanation, it is eligible for official submission.",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_supervised_orb] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_supervised_orb] best_config: {tables_dir / 'best_config_v1.csv'}")
    print(f"[salamander_supervised_orb] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
