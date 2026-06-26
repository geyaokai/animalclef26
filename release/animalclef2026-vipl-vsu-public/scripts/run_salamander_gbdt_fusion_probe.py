#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    XGBClassifier = None


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_FUSION_DIR = Path("artifacts/descriptor_baselines/embed_fusion_v1")
DEFAULT_MASKED_SUPCON_DIR = Path("artifacts/training/experiments/ft_miew_arcface_masked_supcon_v1")
DEFAULT_DISTILL_DIR = Path("artifacts/training/experiments/ft_miew_arcface_distill_v1")
DEFAULT_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_LOCAL_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_OUTPUT_DIR_BY_FEATURE_SET = {
    "basic": Path("artifacts/analysis/salamander_gbdt_fusion_probe_20260331"),
    "meta_graph_v1": Path("artifacts/analysis/salamander_gbdt_fusion_probe_metagraph_v1"),
    "yellow_v1": Path("artifacts/analysis/salamander_gbdt_fusion_probe_yellow_v1"),
    "meta_graph_yellow_v1": Path("artifacts/analysis/salamander_gbdt_fusion_probe_metagraph_yellow_v1"),
}


@dataclass(frozen=True)
class SplitBundle:
    calib_df: pd.DataFrame
    calib_indices: np.ndarray
    eval_df: pd.DataFrame
    eval_indices: np.ndarray


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["ari", "pairwise_f1", "nmi", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def _resolve_metadata_paths(metadata_df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path

    resolved = metadata_df.copy()
    resolved["image_id"] = resolved["image_id"].astype(str)
    resolved["identity"] = resolved["identity"].fillna("").astype(str)
    resolved["path"] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in resolved.iterrows()]
    return resolved


def _build_identity_split(metadata_df: pd.DataFrame, calib_identity_fraction: float, seed: int) -> SplitBundle:
    identity_counts = metadata_df.groupby("identity").size().sort_index()
    multi_ids = identity_counts[identity_counts > 1].index.tolist()
    single_ids = identity_counts[identity_counts == 1].index.tolist()
    rng = random.Random(seed)
    rng.shuffle(multi_ids)
    rng.shuffle(single_ids)

    calib_multi_count = min(len(multi_ids) - 1, max(5, int(round(len(multi_ids) * calib_identity_fraction))))
    calib_multi_count = max(1, calib_multi_count)
    calib_single_count = int(round(len(single_ids) * calib_identity_fraction))

    calib_ids = set(multi_ids[:calib_multi_count]) | set(single_ids[:calib_single_count])
    calib_mask = metadata_df["identity"].isin(calib_ids).to_numpy()
    eval_mask = ~calib_mask
    if calib_mask.sum() == 0 or eval_mask.sum() == 0:
        raise ValueError("Failed to build a non-empty calibration/eval split.")
    return SplitBundle(
        calib_df=metadata_df.loc[calib_mask].reset_index(drop=True),
        calib_indices=np.flatnonzero(calib_mask),
        eval_df=metadata_df.loc[eval_mask].reset_index(drop=True),
        eval_indices=np.flatnonzero(eval_mask),
    )


def _align_branch_embeddings(
    branch_df: pd.DataFrame,
    branch_embeddings: np.ndarray,
    reference_df: pd.DataFrame,
) -> np.ndarray:
    if len(branch_df) != len(branch_embeddings):
        raise ValueError("Branch embeddings do not match branch metadata.")
    branch_df = branch_df.copy().reset_index(drop=True)
    branch_df["image_id"] = branch_df["image_id"].astype(str)
    branch_df["dataset"] = branch_df["dataset"].astype(str)
    lookup = {
        (str(row.image_id), str(row.dataset)): branch_embeddings[index].astype(np.float32, copy=False)
        for index, row in enumerate(branch_df.itertuples(index=False))
    }
    rows: list[np.ndarray] = []
    for row in reference_df.itertuples(index=False):
        key = (str(row.image_id), str(row.dataset))
        if key not in lookup:
            raise KeyError(f"Missing aligned embedding for {key}")
        rows.append(lookup[key])
    return np.stack(rows).astype(np.float32)


def _load_route_bundle(route_dir: Path, repo_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    metadata_df = _resolve_metadata_paths(metadata_df=metadata_df, repo_root=repo_root)
    embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    if len(metadata_df) != len(embeddings):
        raise ValueError("Route embeddings do not match Salamander validation metadata rows.")
    return metadata_df, embeddings


def _load_experiment_branch(experiment_dir: Path, reference_df: pd.DataFrame) -> np.ndarray:
    manifest_df = pd.read_csv(experiment_dir / "tables" / "val_manifest_v1.csv")
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["identity"] = manifest_df["identity"].fillna("").astype(str)
    manifest_df = manifest_df[manifest_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    embeddings = np.load(experiment_dir / "embeddings" / "val_embeddings.npy").astype(np.float32)
    dataset_mask = pd.read_csv(experiment_dir / "tables" / "val_manifest_v1.csv")["dataset"].astype(str) == SALAMANDER_DATASET
    embeddings = embeddings[dataset_mask.to_numpy()]
    return _align_branch_embeddings(branch_df=manifest_df, branch_embeddings=embeddings, reference_df=reference_df)


def _load_fusion_branch(fusion_dir: Path, reference_df: pd.DataFrame) -> np.ndarray:
    from animalclef_analysis.descriptor_baselines import load_cached_embedding_bundle

    bundle = load_cached_embedding_bundle(source_dir=fusion_dir, name="fusion")
    branch_df = bundle.val_df.copy()
    branch_df["image_id"] = branch_df["image_id"].astype(str)
    branch_df["identity"] = branch_df["identity"].fillna("").astype(str)
    branch_df = branch_df[branch_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    branch_embeddings = bundle.val_embeddings[(bundle.val_df["dataset"] == SALAMANDER_DATASET).to_numpy()]
    return _align_branch_embeddings(branch_df=branch_df, branch_embeddings=branch_embeddings, reference_df=reference_df)


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2)
    weights = np.ones(len(labels), dtype=np.float32)
    for class_id in [0, 1]:
        if counts[class_id] == 0:
            continue
        weights[labels == class_id] = float(len(labels) / (2.0 * counts[class_id]))
    return weights


def _build_pairwise_model(
    backend: str,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
    subsample: float,
    split_seed: int,
):
    resolved_backend = str(backend)
    if resolved_backend == "auto":
        resolved_backend = "xgboost" if XGBClassifier is not None else "sklearn"
    if resolved_backend == "xgboost":
        if XGBClassifier is None:
            raise ModuleNotFoundError("Requested backend 'xgboost' but xgboost is not installed.")
        model = XGBClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=int(split_seed),
            n_jobs=8,
            tree_method="hist",
        )
        return model, resolved_backend
    if resolved_backend == "sklearn":
        model = GradientBoostingClassifier(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            subsample=float(subsample),
            random_state=int(split_seed),
        )
        return model, resolved_backend
    raise ValueError(f"Unsupported model backend: {backend}")


def _apply_pair_probability_as_score(
    base_score: np.ndarray,
    pair_df: pd.DataFrame,
    probability_col: str,
    blend_mode: str,
    blend_scale: float,
) -> np.ndarray:
    fused = base_score.copy().astype(np.float32, copy=True)
    for row in pair_df.itertuples(index=False):
        base_value = float(base_score[int(row.left_index), int(row.right_index)])
        probability = float(getattr(row, probability_col))
        if blend_mode == "replace":
            score = probability
        elif blend_mode == "max":
            score = max(base_value, probability)
        elif blend_mode == "avg":
            score = 0.5 * (base_value + probability)
        elif blend_mode == "boost":
            score = min(1.0, base_value + float(blend_scale) * probability * (1.0 - base_value))
        elif blend_mode == "residual":
            score = float(np.clip(base_value + float(blend_scale) * (probability - 0.5), 0.0, 1.0))
        else:
            raise ValueError(f"Unsupported blend_mode: {blend_mode}")
        fused[int(row.left_index), int(row.right_index)] = score
        fused[int(row.right_index), int(row.left_index)] = score
    np.fill_diagonal(fused, 1.0)
    return fused


def _pick_best_local_weight(
    calib_df: pd.DataFrame,
    route_score: np.ndarray,
    local_pair_df: pd.DataFrame,
    local_weights: list[float],
    thresholds: list[float],
) -> tuple[float, pd.DataFrame]:
    from animalclef_analysis.orb_rerank_baseline import apply_local_rerank, evaluate_threshold_sweep_from_score_matrix

    rows: list[dict[str, object]] = []
    for local_weight in local_weights:
        reranked = apply_local_rerank(
            global_score_matrix=route_score,
            pair_df=local_pair_df,
            local_weight=float(local_weight),
        )
        sweep_df, _pred_df = evaluate_threshold_sweep_from_score_matrix(
            df=calib_df,
            score_matrix=reranked,
            thresholds=thresholds,
        )
        best_row = _pick_best_row(sweep_df)
        rows.append(
            {
                "local_weight": float(local_weight),
                "threshold": float(best_row["threshold"]),
                "ari": float(best_row["ari"]),
                "nmi": float(best_row["nmi"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "cluster_count": int(best_row["cluster_count"]),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "local_weight"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return float(summary_df.iloc[0]["local_weight"]), summary_df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import (
        apply_local_rerank,
        build_local_match_table,
        build_topk_pair_index,
        cosine_score_matrix,
        evaluate_threshold_sweep_from_score_matrix,
        extract_orb_features,
        recall_at_k_from_score_matrix,
    )
    from animalclef_analysis.pseudo_seed_features import (
        PSEUDO_SEED_FEATURE_COLUMNS,
        append_pseudo_seed_pair_features,
        build_pseudo_seed_feature_bundle,
    )
    from animalclef_analysis.salamander_pairwise_features import (
        FEATURE_SET_META_GRAPH_YELLOW_V1,
        FEATURE_SET_YELLOW_V1,
        append_feature_set,
        build_pair_feature_table,
        resolve_pair_feature_columns,
    )
    from animalclef_analysis.salamander_yellow_xgb_features import build_yellow_pair_feature_artifacts

    parser = argparse.ArgumentParser(description="Run a Salamander xgboost-like GBDT fusion probe on top candidate pairs.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fusion-dir", type=Path, default=DEFAULT_FUSION_DIR)
    parser.add_argument("--masked-supcon-dir", type=Path, default=DEFAULT_MASKED_SUPCON_DIR)
    parser.add_argument("--distill-dir", type=Path, default=DEFAULT_DISTILL_DIR)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--feature-set", choices=["basic", "meta_graph_v1", "yellow_v1", "meta_graph_yellow_v1"], default="basic")
    parser.add_argument("--graph-top-k", type=int, default=10)
    parser.add_argument("--calib-identity-fraction", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--local-weights", nargs="+", type=float, default=None)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--model-backend", choices=["auto", "xgboost", "sklearn"], default="auto")
    parser.add_argument("--blend-mode", choices=["replace", "max", "avg", "boost", "residual"], default="boost")
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--pseudo-anchor-threshold", type=float, default=0.50)
    parser.add_argument("--pseudo-stability-delta", type=float, default=0.03)
    parser.add_argument("--pseudo-min-seed-cluster-size", type=int, default=2)
    parser.add_argument("--pseudo-max-seed-cluster-size", type=int, default=12)
    parser.add_argument("--pseudo-min-mean-similarity", type=float, default=0.0)
    args = parser.parse_args()

    route_dir = args.route_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / DEFAULT_OUTPUT_DIR_BY_FEATURE_SET[str(args.feature_set)]).resolve()
    )
    manifest_root = args.manifest_root.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    local_weights = args.local_weights or DEFAULT_LOCAL_WEIGHTS
    use_yellow_features = str(args.feature_set) in {FEATURE_SET_YELLOW_V1, FEATURE_SET_META_GRAPH_YELLOW_V1}

    metadata_df, route_embeddings = _load_route_bundle(route_dir=route_dir, repo_root=repo_root)
    split_bundle = _build_identity_split(
        metadata_df=metadata_df,
        calib_identity_fraction=float(args.calib_identity_fraction),
        seed=int(args.split_seed),
    )
    split_df = metadata_df[["image_id", "identity", "path"]].copy()
    split_df["probe_split"] = "eval"
    split_df.loc[split_bundle.calib_indices, "probe_split"] = "calib"
    split_df.to_csv(tables_dir / "split_assignments_v1.csv", index=False)

    fusion_embeddings = _load_fusion_branch(fusion_dir=args.fusion_dir.resolve(), reference_df=metadata_df)
    student_embeddings = _load_experiment_branch(experiment_dir=args.masked_supcon_dir.resolve(), reference_df=metadata_df)
    distill_embeddings = _load_experiment_branch(experiment_dir=args.distill_dir.resolve(), reference_df=metadata_df)

    route_calib_emb = route_embeddings[split_bundle.calib_indices]
    route_eval_emb = route_embeddings[split_bundle.eval_indices]
    fusion_calib_emb = fusion_embeddings[split_bundle.calib_indices]
    fusion_eval_emb = fusion_embeddings[split_bundle.eval_indices]
    student_calib_emb = student_embeddings[split_bundle.calib_indices]
    student_eval_emb = student_embeddings[split_bundle.eval_indices]
    distill_calib_emb = distill_embeddings[split_bundle.calib_indices]
    distill_eval_emb = distill_embeddings[split_bundle.eval_indices]
    yellow_enriched_df = None
    if use_yellow_features:
        yellow_enriched_df = pd.read_csv(manifest_root / "tables" / "metadata_enriched_v1.csv")
        yellow_enriched_df["image_id"] = yellow_enriched_df["image_id"].astype(str)
        yellow_enriched_df["dataset"] = yellow_enriched_df["dataset"].astype(str)
        yellow_enriched_df = yellow_enriched_df[yellow_enriched_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)

    route_calib_score = cosine_score_matrix(route_calib_emb)
    route_eval_score = cosine_score_matrix(route_eval_emb)
    fusion_calib_score = cosine_score_matrix(fusion_calib_emb)
    fusion_eval_score = cosine_score_matrix(fusion_eval_emb)
    student_calib_score = cosine_score_matrix(student_calib_emb)
    student_eval_score = cosine_score_matrix(student_eval_emb)
    distill_calib_score = cosine_score_matrix(distill_calib_emb)
    distill_eval_score = cosine_score_matrix(distill_eval_emb)

    calib_pair_index = build_topk_pair_index(score_matrix=route_calib_score, top_k=int(args.top_k), query_indices=None)
    eval_pair_index = build_topk_pair_index(score_matrix=route_eval_score, top_k=int(args.top_k), query_indices=None)

    calib_features = extract_orb_features(
        df=split_bundle.calib_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
    )
    eval_features = extract_orb_features(
        df=split_bundle.eval_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
    )
    calib_local_pair_df = build_local_match_table(
        df=split_bundle.calib_df,
        features=calib_features,
        pair_index=calib_pair_index,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    eval_local_pair_df = build_local_match_table(
        df=split_bundle.eval_df,
        features=eval_features,
        pair_index=eval_pair_index,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    calib_local_pair_df.to_csv(tables_dir / "calib_local_match_scores_v1.csv", index=False)
    eval_local_pair_df.to_csv(tables_dir / "eval_local_match_scores_v1.csv", index=False)

    calib_pair_df = build_pair_feature_table(
        metadata_df=split_bundle.calib_df,
        local_pair_df=calib_local_pair_df,
        route_score=route_calib_score,
        fusion_score=fusion_calib_score,
        student_score=student_calib_score,
        distill_score=distill_calib_score,
    )
    eval_pair_df = build_pair_feature_table(
        metadata_df=split_bundle.eval_df,
        local_pair_df=eval_local_pair_df,
        route_score=route_eval_score,
        fusion_score=fusion_eval_score,
        student_score=student_eval_score,
        distill_score=distill_eval_score,
    )
    yellow_calib_pair_df = None
    yellow_eval_pair_df = None
    if use_yellow_features:
        if yellow_enriched_df is None:
            raise ValueError("Yellow feature set requires enriched manifest metadata.")
        yellow_calib_artifacts = build_yellow_pair_feature_artifacts(
            reference_df=split_bundle.calib_df,
            pair_df=calib_pair_df,
            enriched_df=yellow_enriched_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        yellow_eval_artifacts = build_yellow_pair_feature_artifacts(
            reference_df=split_bundle.eval_df,
            pair_df=eval_pair_df,
            enriched_df=yellow_enriched_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        yellow_calib_pair_df = yellow_calib_artifacts.pair_feature_df
        yellow_eval_pair_df = yellow_eval_artifacts.pair_feature_df
        yellow_calib_artifacts.roi_manifest_df.to_csv(tables_dir / "calib_yellow_roi_manifest_v1.csv", index=False)
        yellow_eval_artifacts.roi_manifest_df.to_csv(tables_dir / "eval_yellow_roi_manifest_v1.csv", index=False)
        yellow_calib_artifacts.focus_df.to_csv(tables_dir / "calib_yellow_focus_manifest_v1.csv", index=False)
        yellow_eval_artifacts.focus_df.to_csv(tables_dir / "eval_yellow_focus_manifest_v1.csv", index=False)
        yellow_calib_artifacts.yellow_roi_local_df.to_csv(tables_dir / "calib_yellow_roi_local_scores_v1.csv", index=False)
        yellow_eval_artifacts.yellow_roi_local_df.to_csv(tables_dir / "eval_yellow_roi_local_scores_v1.csv", index=False)
        yellow_calib_artifacts.patch_pair_df.to_csv(tables_dir / "calib_yellow_patch_pair_features_v1.csv", index=False)
        yellow_eval_artifacts.patch_pair_df.to_csv(tables_dir / "eval_yellow_patch_pair_features_v1.csv", index=False)
        yellow_calib_artifacts.decision_summary_df.to_csv(tables_dir / "calib_yellow_decision_summary_v1.csv", index=False)
        yellow_eval_artifacts.decision_summary_df.to_csv(tables_dir / "eval_yellow_decision_summary_v1.csv", index=False)
    calib_pair_df = append_feature_set(
        pair_df=calib_pair_df,
        metadata_df=split_bundle.calib_df,
        route_score=route_calib_score,
        feature_set=str(args.feature_set),
        graph_top_k=int(args.graph_top_k),
        yellow_pair_df=yellow_calib_pair_df,
    )
    eval_pair_df = append_feature_set(
        pair_df=eval_pair_df,
        metadata_df=split_bundle.eval_df,
        route_score=route_eval_score,
        feature_set=str(args.feature_set),
        graph_top_k=int(args.graph_top_k),
        yellow_pair_df=yellow_eval_pair_df,
    )
    calib_pseudo_bundle = build_pseudo_seed_feature_bundle(
        metadata_df=split_bundle.calib_df,
        teacher_embeddings=route_calib_emb,
        anchor_threshold=float(args.pseudo_anchor_threshold),
        stability_delta=float(args.pseudo_stability_delta),
        min_seed_cluster_size=int(args.pseudo_min_seed_cluster_size),
        max_seed_cluster_size=int(args.pseudo_max_seed_cluster_size),
        min_mean_similarity=float(args.pseudo_min_mean_similarity),
    )
    eval_pseudo_bundle = build_pseudo_seed_feature_bundle(
        metadata_df=split_bundle.eval_df,
        teacher_embeddings=route_eval_emb,
        anchor_threshold=float(args.pseudo_anchor_threshold),
        stability_delta=float(args.pseudo_stability_delta),
        min_seed_cluster_size=int(args.pseudo_min_seed_cluster_size),
        max_seed_cluster_size=int(args.pseudo_max_seed_cluster_size),
        min_mean_similarity=float(args.pseudo_min_mean_similarity),
    )
    calib_pair_pseudo_df = append_pseudo_seed_pair_features(
        pair_df=calib_pair_df,
        assignment_df=calib_pseudo_bundle.assignment_df,
    )
    eval_pair_pseudo_df = append_pseudo_seed_pair_features(
        pair_df=eval_pair_df,
        assignment_df=eval_pseudo_bundle.assignment_df,
    )

    base_feature_columns = resolve_pair_feature_columns(str(args.feature_set))
    pseudo_feature_columns = base_feature_columns + PSEUDO_SEED_FEATURE_COLUMNS
    base_model, resolved_backend = _build_pairwise_model(
        backend=str(args.model_backend),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        split_seed=int(args.split_seed),
    )
    pseudo_model, _resolved_backend_2 = _build_pairwise_model(
        backend=str(args.model_backend),
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        split_seed=int(args.split_seed),
    )
    calib_y = calib_pair_df["same_identity"].to_numpy(dtype=int)
    calib_weight = _balanced_sample_weight(calib_y)
    base_model.fit(calib_pair_df[base_feature_columns].to_numpy(dtype=np.float32), calib_y, sample_weight=calib_weight)
    pseudo_model.fit(calib_pair_pseudo_df[pseudo_feature_columns].to_numpy(dtype=np.float32), calib_y, sample_weight=calib_weight)

    calib_pair_df["gbdt_same_identity_prob"] = base_model.predict_proba(calib_pair_df[base_feature_columns].to_numpy(dtype=np.float32))[:, 1]
    eval_pair_df["gbdt_same_identity_prob"] = base_model.predict_proba(eval_pair_df[base_feature_columns].to_numpy(dtype=np.float32))[:, 1]
    calib_pair_pseudo_df["gbdt_same_identity_prob"] = pseudo_model.predict_proba(
        calib_pair_pseudo_df[pseudo_feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    eval_pair_pseudo_df["gbdt_same_identity_prob"] = pseudo_model.predict_proba(
        eval_pair_pseudo_df[pseudo_feature_columns].to_numpy(dtype=np.float32)
    )[:, 1]
    base_feature_importance_df = pd.DataFrame(
        {
            "feature": base_feature_columns,
            "importance": base_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    pseudo_feature_importance_df = pd.DataFrame(
        {
            "feature": pseudo_feature_columns,
            "importance": pseudo_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    base_feature_importance_df.to_csv(tables_dir / "feature_importance_base_v1.csv", index=False)
    pseudo_feature_importance_df.to_csv(tables_dir / "feature_importance_pseudo_v1.csv", index=False)

    chosen_local_weight, local_weight_summary_df = _pick_best_local_weight(
        calib_df=split_bundle.calib_df,
        route_score=route_calib_score,
        local_pair_df=calib_local_pair_df,
        local_weights=[float(value) for value in local_weights],
        thresholds=thresholds,
    )
    local_weight_summary_df.to_csv(tables_dir / "local_weight_summary_v1.csv", index=False)

    baseline_sweep_df, baseline_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=route_eval_score,
        thresholds=thresholds,
    )
    orb_eval_score = apply_local_rerank(
        global_score_matrix=route_eval_score,
        pair_df=eval_local_pair_df,
        local_weight=float(chosen_local_weight),
    )
    orb_sweep_df, orb_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=orb_eval_score,
        thresholds=thresholds,
    )
    gbdt_eval_score = _apply_pair_probability_as_score(
        base_score=route_eval_score,
        pair_df=eval_pair_df,
        probability_col="gbdt_same_identity_prob",
        blend_mode=str(args.blend_mode),
        blend_scale=float(args.blend_scale),
    )
    gbdt_pseudo_eval_score = _apply_pair_probability_as_score(
        base_score=route_eval_score,
        pair_df=eval_pair_pseudo_df,
        probability_col="gbdt_same_identity_prob",
        blend_mode=str(args.blend_mode),
        blend_scale=float(args.blend_scale),
    )
    gbdt_sweep_df, gbdt_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=gbdt_eval_score,
        thresholds=thresholds,
    )
    gbdt_pseudo_sweep_df, gbdt_pseudo_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=gbdt_pseudo_eval_score,
        thresholds=thresholds,
    )

    baseline_best = _pick_best_row(baseline_sweep_df)
    orb_best = _pick_best_row(orb_sweep_df)
    gbdt_best = _pick_best_row(gbdt_sweep_df)
    gbdt_pseudo_best = _pick_best_row(gbdt_pseudo_sweep_df)

    baseline_sweep_df.to_csv(tables_dir / "baseline_threshold_sweep_v1.csv", index=False)
    orb_sweep_df.to_csv(tables_dir / "orb_threshold_sweep_v1.csv", index=False)
    gbdt_sweep_df.to_csv(tables_dir / "gbdt_threshold_sweep_v1.csv", index=False)
    gbdt_pseudo_sweep_df.to_csv(tables_dir / "gbdt_pseudo_threshold_sweep_v1.csv", index=False)
    baseline_pred_df.to_csv(tables_dir / "baseline_predictions_v1.csv", index=False)
    orb_pred_df.to_csv(tables_dir / "orb_predictions_v1.csv", index=False)
    gbdt_pred_df.to_csv(tables_dir / "gbdt_predictions_v1.csv", index=False)
    gbdt_pseudo_pred_df.to_csv(tables_dir / "gbdt_pseudo_predictions_v1.csv", index=False)
    calib_pair_df.to_csv(tables_dir / "calib_pair_features_base_v1.csv", index=False)
    eval_pair_df.to_csv(tables_dir / "eval_pair_features_base_v1.csv", index=False)
    calib_pair_pseudo_df.to_csv(tables_dir / "calib_pair_features_pseudo_v1.csv", index=False)
    eval_pair_pseudo_df.to_csv(tables_dir / "eval_pair_features_pseudo_v1.csv", index=False)
    calib_pseudo_bundle.assignment_df.to_csv(tables_dir / "calib_pseudo_seed_assignments_v1.csv", index=False)
    eval_pseudo_bundle.assignment_df.to_csv(tables_dir / "eval_pseudo_seed_assignments_v1.csv", index=False)
    calib_pseudo_bundle.cluster_summary_df.to_csv(tables_dir / "calib_pseudo_seed_clusters_v1.csv", index=False)
    eval_pseudo_bundle.cluster_summary_df.to_csv(tables_dir / "eval_pseudo_seed_clusters_v1.csv", index=False)
    calib_pseudo_bundle.threshold_summary_df.to_csv(tables_dir / "calib_pseudo_seed_thresholds_v1.csv", index=False)
    eval_pseudo_bundle.threshold_summary_df.to_csv(tables_dir / "eval_pseudo_seed_thresholds_v1.csv", index=False)
    pd.DataFrame([calib_pseudo_bundle.summary_row, eval_pseudo_bundle.summary_row], index=["calib", "eval"]).reset_index().rename(
        columns={"index": "split"}
    ).to_csv(tables_dir / "pseudo_seed_summary_v1.csv", index=False)

    eval_labels = split_bundle.eval_df["identity"].to_numpy()
    comparison_df = pd.DataFrame(
        [
            {
                "route": "baseline_global",
                "threshold": float(baseline_best["threshold"]),
                "ari": float(baseline_best["ari"]),
                "nmi": float(baseline_best["nmi"]),
                "pairwise_f1": float(baseline_best["pairwise_f1"]),
                "cluster_count": int(baseline_best["cluster_count"]),
                "recall_at_1": float(recall_at_k_from_score_matrix(route_eval_score, eval_labels, k=1)),
                "recall_at_5": float(recall_at_k_from_score_matrix(route_eval_score, eval_labels, k=5)),
            },
            {
                "route": "heuristic_orb_rerank",
                "threshold": float(orb_best["threshold"]),
                "ari": float(orb_best["ari"]),
                "nmi": float(orb_best["nmi"]),
                "pairwise_f1": float(orb_best["pairwise_f1"]),
                "cluster_count": int(orb_best["cluster_count"]),
                "recall_at_1": float(recall_at_k_from_score_matrix(orb_eval_score, eval_labels, k=1)),
                "recall_at_5": float(recall_at_k_from_score_matrix(orb_eval_score, eval_labels, k=5)),
            },
            {
                "route": "gbdt_pairwise_fusion",
                "threshold": float(gbdt_best["threshold"]),
                "ari": float(gbdt_best["ari"]),
                "nmi": float(gbdt_best["nmi"]),
                "pairwise_f1": float(gbdt_best["pairwise_f1"]),
                "cluster_count": int(gbdt_best["cluster_count"]),
                "recall_at_1": float(recall_at_k_from_score_matrix(gbdt_eval_score, eval_labels, k=1)),
                "recall_at_5": float(recall_at_k_from_score_matrix(gbdt_eval_score, eval_labels, k=5)),
            },
            {
                "route": "gbdt_pairwise_fusion_pseudo_seed",
                "threshold": float(gbdt_pseudo_best["threshold"]),
                "ari": float(gbdt_pseudo_best["ari"]),
                "nmi": float(gbdt_pseudo_best["nmi"]),
                "pairwise_f1": float(gbdt_pseudo_best["pairwise_f1"]),
                "cluster_count": int(gbdt_pseudo_best["cluster_count"]),
                "recall_at_1": float(recall_at_k_from_score_matrix(gbdt_pseudo_eval_score, eval_labels, k=1)),
                "recall_at_5": float(recall_at_k_from_score_matrix(gbdt_pseudo_eval_score, eval_labels, k=5)),
            },
        ]
    )
    comparison_df["ari_delta_vs_baseline"] = np.round(comparison_df["ari"] - float(baseline_best["ari"]), 6)
    comparison_df["pairwise_f1_delta_vs_baseline"] = np.round(
        comparison_df["pairwise_f1"] - float(baseline_best["pairwise_f1"]),
        6,
    )
    comparison_df.to_csv(tables_dir / "comparison_summary_v1.csv", index=False)

    summary = {
        "probe": "salamander_gbdt_fusion_probe",
        "date": "2026-03-31",
        "dataset": SALAMANDER_DATASET,
        "route_dir": str(route_dir),
        "fusion_dir": str(args.fusion_dir.resolve()),
        "masked_supcon_dir": str(args.masked_supcon_dir.resolve()),
        "distill_dir": str(args.distill_dir.resolve()),
        "val_image_count": int(len(metadata_df)),
        "calib_image_count": int(len(split_bundle.calib_df)),
        "eval_image_count": int(len(split_bundle.eval_df)),
        "calib_identity_count": int(split_bundle.calib_df["identity"].nunique()),
        "eval_identity_count": int(split_bundle.eval_df["identity"].nunique()),
        "top_k": int(args.top_k),
        "feature_set": str(args.feature_set),
        "graph_top_k": int(args.graph_top_k),
        "manifest_root": str(manifest_root) if use_yellow_features else "",
        "yellow_enabled": bool(use_yellow_features),
        "base_feature_columns": base_feature_columns,
        "thresholds": [float(value) for value in thresholds],
        "chosen_local_weight": float(chosen_local_weight),
        "gbdt_config": {
            "model_backend": resolved_backend,
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "blend_mode": str(args.blend_mode),
            "blend_scale": float(args.blend_scale),
        },
        "pseudo_seed_config": {
            "anchor_threshold": float(args.pseudo_anchor_threshold),
            "stability_delta": float(args.pseudo_stability_delta),
            "min_seed_cluster_size": int(args.pseudo_min_seed_cluster_size),
            "max_seed_cluster_size": int(args.pseudo_max_seed_cluster_size),
            "min_mean_similarity": float(args.pseudo_min_mean_similarity),
            "feature_columns": PSEUDO_SEED_FEATURE_COLUMNS,
        },
        "pseudo_seed_summary": {
            "calib": calib_pseudo_bundle.summary_row,
            "eval": eval_pseudo_bundle.summary_row,
        },
        "comparison_rows": comparison_df.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# Salamander GBDT Fusion Probe",
        "",
        "- Goal: test an `xgboost-like` pairwise fusion layer using local and multi-branch global scores on the current Salamander route.",
        f"- Model backend: `{resolved_backend}`.",
        f"- Route dir: `{route_dir}`",
        f"- Probe split: validation-only internal split with `calib_identity_fraction={float(args.calib_identity_fraction)}` and `seed={int(args.split_seed)}`.",
        f"- Calibration size: `{len(split_bundle.calib_df)}` images / `{split_bundle.calib_df['identity'].nunique()}` identities.",
        f"- Eval size: `{len(split_bundle.eval_df)}` images / `{split_bundle.eval_df['identity'].nunique()}` identities.",
        f"- Candidate pairs: top-`{int(args.top_k)}` from the current route global score.",
        f"- Pair feature set: `{args.feature_set}` with route-graph top-k `{int(args.graph_top_k)}`.",
        (
            f"- Yellow local features: `enabled` from `{manifest_root}` using yellow-focus ROI ORB + patch metrics."
            if use_yellow_features
            else "- Yellow local features: `disabled`."
        ),
        f"- ORB anchor local weight selected on calib split: `{float(chosen_local_weight)}`.",
        f"- Pseudo anchor: distance threshold `{float(args.pseudo_anchor_threshold)}` with stability delta `{float(args.pseudo_stability_delta)}`.",
        "",
        "## Comparison",
        "",
        dataframe_to_markdown_table(
            comparison_df[
                [
                    "route",
                    "threshold",
                    "ari",
                    "nmi",
                    "pairwise_f1",
                    "cluster_count",
                    "recall_at_1",
                    "recall_at_5",
                    "ari_delta_vs_baseline",
                    "pairwise_f1_delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Pseudo Seed Summary",
        "",
        dataframe_to_markdown_table(
            pd.DataFrame([calib_pseudo_bundle.summary_row, eval_pseudo_bundle.summary_row], index=["calib", "eval"])
            .reset_index()
            .rename(columns={"index": "split"})
        ),
        "",
        "## Feature Importance",
        "",
        "### Baseline",
        "",
        dataframe_to_markdown_table(base_feature_importance_df.head(10)),
        "",
        "### Pseudo Seed",
        "",
        dataframe_to_markdown_table(pseudo_feature_importance_df.head(12)),
        "",
        "## Reading Note",
        "",
        f"- The classifier is trained only on candidate pairs from the calibration split, then its same-identity probability is fused back with the baseline score using `{args.blend_mode}` (`scale={float(args.blend_scale)}`).",
        "- `left_is_seeded / right_is_seeded`: this image was accepted as a stable pseudo seed under the chosen anchor.",
        "- `both_seeded / one_seeded / both_unseeded`: tells the tree whether this pair sits inside the high-confidence seed world, half-known boundary, or fully unknown area.",
        "- `same_seed_cluster`: both images fall into the same accepted seed cluster; it is the strongest pseudo cue here.",
        "- `left_seed_cluster_size / right_seed_cluster_size`: larger seed clusters are usually less pure, but they also carry more coverage; the model learns the trade-off.",
        "- `left_seed_mean_similarity / right_seed_mean_similarity`: average teacher cosine similarity inside that seed cluster; it is a purity proxy, not a ground-truth label.",
        "- This is still a direction check for inference-time pseudo usage, not yet a submission-ready result.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[salamander_gbdt_fusion_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_gbdt_fusion_probe] comparison: {tables_dir / 'comparison_summary_v1.csv'}")
    print(f"[salamander_gbdt_fusion_probe] features(base): {tables_dir / 'feature_importance_base_v1.csv'}")
    print(f"[salamander_gbdt_fusion_probe] features(pseudo): {tables_dir / 'feature_importance_pseudo_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
