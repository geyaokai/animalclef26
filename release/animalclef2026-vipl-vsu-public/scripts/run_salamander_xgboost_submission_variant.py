#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import xgboost
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency in wildfusion env
    raise ModuleNotFoundError("This script requires `xgboost` in the active environment.") from exc


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_BASE_PREDICTIONS = DEFAULT_ROUTE_DIR / "tables" / "test_predictions_v1.csv"
DEFAULT_FUSION_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_fusion_v1")
DEFAULT_MASKED_SUPCON_CHECKPOINT = Path("artifacts/training/experiments/ft_miew_arcface_masked_supcon_v1/checkpoints/last.pt")
DEFAULT_DISTILL_CHECKPOINT = Path("artifacts/training/experiments/ft_miew_arcface_distill_v1/checkpoints/last.pt")
DEFAULT_REFERENCE_PROBE_DIRS = [
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_20260331"),
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_seed43_xgb"),
    Path("artifacts/analysis/salamander_gbdt_fusion_probe_seed44_xgb"),
]
DEFAULT_THRESHOLD_CANDIDATES = [0.15, 0.2, 0.25, 0.3, 0.35]
DEFAULT_CHOSEN_THRESHOLD = 0.25
DEFAULT_MASKED_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_OUTPUT_DIR_BY_FEATURE_SET = {
    "basic": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_v1"),
    "dual_view_v1": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_dualview_v1"),
    "meta_graph_v1": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_metagraph_v1"),
    "meta_graph_dual_view_v1": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_metagraph_dualview_v1"),
    "yellow_v1": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_yellow_v1"),
    "meta_graph_yellow_v1": Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_metagraph_yellow_v1"),
}
DEFAULT_ROUTE_NAME_BY_FEATURE_SET = {
    "basic": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_v1",
    "dual_view_v1": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_dualview_v1",
    "meta_graph_v1": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_metagraph_v1",
    "meta_graph_dual_view_v1": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_metagraph_dualview_v1",
    "yellow_v1": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_yellow_v1",
    "meta_graph_yellow_v1": "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_metagraph_yellow_v1",
}


def _default_output_dir_for_feature_set(feature_set: str, yellow_local_view: str) -> Path:
    base = DEFAULT_OUTPUT_DIR_BY_FEATURE_SET[str(feature_set)]
    if str(yellow_local_view) == "focus" or "yellow" not in str(feature_set):
        return base
    return base.with_name(base.name.replace("yellow_v1", "yellowband_v1"))


def _default_route_name_for_feature_set(feature_set: str, yellow_local_view: str) -> str:
    base = DEFAULT_ROUTE_NAME_BY_FEATURE_SET[str(feature_set)]
    if str(yellow_local_view) == "focus" or "yellow" not in str(feature_set):
        return base
    return base.replace("yellow_v1", "yellow_band_v1")


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["ari", "pairwise_f1", "nmi", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def _resolve_metadata_paths(metadata_df: pd.DataFrame, repo_root: Path, path_column: str, resolver) -> pd.DataFrame:
    resolved = metadata_df.copy().reset_index(drop=True)
    resolved["image_id"] = resolved["image_id"].astype(str)
    if "identity" in resolved.columns:
        resolved["identity"] = resolved["identity"].fillna("").astype(str)

    def _resolve_row_path(row: pd.Series) -> str:
        candidate_columns = [
            path_column,
            "path",
            "global_path",
            "source_global_path",
            "trunk_path",
            "preferred_path_v1",
            "normalized_path_v1",
        ]
        checked_paths: list[str] = []
        for column in candidate_columns:
            if column not in row.index:
                continue
            value = row.get(column, "")
            if pd.isna(value):
                continue
            rel_path = str(value).strip()
            if not rel_path or rel_path in checked_paths:
                continue
            checked_paths.append(rel_path)
            if (repo_root / rel_path).exists():
                return rel_path
        return resolver(row, repo_root=repo_root)

    resolved[path_column] = [_resolve_row_path(row) for _, row in resolved.iterrows()]
    resolved["path"] = resolved[path_column]
    return resolved


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


def _load_route_bundle(route_dir: Path, repo_root: Path, path_column: str, resolver) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    test_df = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    val_df = _resolve_metadata_paths(val_df, repo_root=repo_root, path_column=path_column, resolver=resolver)
    test_df = _resolve_metadata_paths(test_df, repo_root=repo_root, path_column=path_column, resolver=resolver)
    val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy").astype(np.float32)
    if len(val_df) != len(val_embeddings) or len(test_df) != len(test_embeddings):
        raise ValueError("Route embeddings do not match Salamander metadata rows.")
    return val_df, val_embeddings, test_df, test_embeddings


def _load_or_build_local_match_table(
    *,
    split_name: str,
    route_dir: Path,
    output_tables_dir: Path,
    metadata_df: pd.DataFrame,
    route_embeddings: np.ndarray,
    repo_root: Path,
    top_k: int,
    flip_invariant: bool,
) -> tuple[pd.DataFrame, str]:
    from animalclef_analysis.orb_rerank_baseline import (
        build_local_match_table,
        build_topk_pair_index,
        cosine_score_matrix,
        extract_local_features,
    )

    route_tables_dir = route_dir / "tables"
    route_local_path = route_tables_dir / f"{split_name}_local_match_scores_v1.csv"
    route_keypoint_path = route_tables_dir / f"{split_name}_orb_keypoints_v1.csv"
    output_local_path = output_tables_dir / f"{split_name}_local_match_scores_v1.csv"
    output_keypoint_path = output_tables_dir / f"{split_name}_orb_keypoints_v1.csv"

    if route_local_path.exists():
        local_pair_df = pd.read_csv(route_local_path)
        cache_supports_flip = {"flip_invariant_enabled", "right_flipped_match_selected"}.issubset(local_pair_df.columns)
        if (not bool(flip_invariant)) or cache_supports_flip:
            local_pair_df.to_csv(output_local_path, index=False)
            if route_keypoint_path.exists():
                shutil.copy2(route_keypoint_path, output_keypoint_path)
            return local_pair_df, "route_cache" if cache_supports_flip else "route_cache_no_flip"

    route_score = cosine_score_matrix(route_embeddings)
    pair_index = build_topk_pair_index(score_matrix=route_score, top_k=int(top_k), query_indices=None)
    feature_df = metadata_df.copy().reset_index(drop=True)
    if "identity" not in feature_df.columns:
        feature_df["identity"] = ""
    features = extract_local_features(
        df=feature_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
        local_matcher="orb",
        hflip=False,
    )
    flipped_features = None
    if bool(flip_invariant):
        flipped_features = extract_local_features(
            df=feature_df,
            repo_root=repo_root,
            nfeatures=1024,
            max_side=768,
            fast_threshold=7,
            clahe_clip_limit=2.0,
            local_matcher="orb",
            hflip=True,
        )
    local_pair_df = build_local_match_table(
        df=feature_df,
        features=features,
        flipped_features=flipped_features,
        pair_index=pair_index,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    local_pair_df.to_csv(output_local_path, index=False)
    pd.DataFrame(
        [
            {
                "image_id": feature.image_id,
                "keypoints": feature.point_count,
                "width": feature.width,
                "height": feature.height,
            }
            for feature in features
        ]
    ).to_csv(output_keypoint_path, index=False)
    return local_pair_df, "recomputed_from_route_embeddings_flip" if bool(flip_invariant) else "recomputed_from_route_embeddings"


def _load_fusion_branch(
    fusion_dir: Path,
    reference_df: pd.DataFrame,
    *,
    repo_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, str]:
    from animalclef_analysis.descriptor_baselines import (
        extract_embeddings,
        fuse_embedding_blocks,
        load_cached_embedding_bundle,
        load_descriptor_model,
    )

    bundle = load_cached_embedding_bundle(source_dir=fusion_dir, name="fusion")
    branch_df = bundle.test_df.copy() if reference_df["identity"].eq("").all() else bundle.val_df.copy()
    branch_df["image_id"] = branch_df["image_id"].astype(str)
    branch_df["identity"] = branch_df["identity"].fillna("").astype(str)
    branch_df = branch_df[branch_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    branch_embeddings = (
        bundle.test_embeddings[(bundle.test_df["dataset"] == SALAMANDER_DATASET).to_numpy()]
        if reference_df["identity"].eq("").all()
        else bundle.val_embeddings[(bundle.val_df["dataset"] == SALAMANDER_DATASET).to_numpy()]
    )
    try:
        return _align_branch_embeddings(branch_df=branch_df, branch_embeddings=branch_embeddings, reference_df=reference_df), "cache_aligned"
    except KeyError:
        mega_model, mega_spec = load_descriptor_model(descriptor="mega", device=device)
        mega_embeddings = extract_embeddings(
            df=reference_df,
            repo_root=repo_root,
            model=mega_model,
            spec=mega_spec,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        del mega_model
        miew_model, miew_spec = load_descriptor_model(descriptor="miew", device=device)
        miew_embeddings = extract_embeddings(
            df=reference_df,
            repo_root=repo_root,
            model=miew_model,
            spec=miew_spec,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        del miew_model
        fused_embeddings = fuse_embedding_blocks(
            [mega_embeddings, miew_embeddings],
            weights=[1.0, 1.0],
        )
        return fused_embeddings, "recomputed_for_route_split"


def _extract_checkpoint_embeddings(
    checkpoint_path: Path,
    eval_df: pd.DataFrame,
    repo_root: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    from animalclef_analysis.submission_baseline import _load_supervised_model_from_checkpoint
    from animalclef_analysis.supervised_training import extract_student_embeddings

    model, spec, _config, _checkpoint = _load_supervised_model_from_checkpoint(checkpoint_path=checkpoint_path, device=device)
    return extract_student_embeddings(
        df=eval_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def _balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=2)
    weights = np.ones(len(labels), dtype=np.float32)
    for class_id in [0, 1]:
        if counts[class_id] == 0:
            continue
        weights[labels == class_id] = float(len(labels) / (2.0 * counts[class_id]))
    return weights


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


def _load_probe_evidence(reference_probe_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for probe_dir in reference_probe_dirs:
        summary_path = probe_dir / "tables" / "comparison_summary_v1.csv"
        if not summary_path.exists():
            continue
        summary_df = pd.read_csv(summary_path)
        summary_df["probe_dir"] = str(probe_dir)
        rows.append(summary_df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return combined


def _cosine_cross_score_matrix(left_embeddings: np.ndarray, right_embeddings: np.ndarray) -> np.ndarray:
    if left_embeddings.shape != right_embeddings.shape:
        raise ValueError("Cross-view embeddings must have identical shapes.")
    similarity = np.clip(left_embeddings @ right_embeddings.T, -1.0, 1.0)
    return ((similarity + 1.0) / 2.0).astype(np.float32, copy=False)


def _resolve_masked_view_df(
    reference_df: pd.DataFrame,
    *,
    manifest_root: Path,
    resolved_path_column: str,
) -> tuple[pd.DataFrame, float]:
    metadata_path = manifest_root / "tables" / "metadata_enriched_v1.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Masked manifest metadata not found at {metadata_path}. Build manifests before using dual-view feature sets."
        )
    enriched_df = pd.read_csv(metadata_path)
    enriched_df["image_id"] = enriched_df["image_id"].astype(str)
    enriched_df["dataset"] = enriched_df["dataset"].astype(str)
    if resolved_path_column not in enriched_df.columns:
        raise KeyError(f"{resolved_path_column} not found in {metadata_path}")

    available_columns = ["image_id", "dataset", resolved_path_column]
    applied_column = resolved_path_column.replace("_resolved_path_v1", "_applied")
    if applied_column in enriched_df.columns:
        available_columns.append(applied_column)
    lookup_df = enriched_df[available_columns].drop_duplicates(subset=["image_id", "dataset"])
    merged = reference_df.merge(lookup_df, on=["image_id", "dataset"], how="left", validate="one_to_one")
    if merged[resolved_path_column].isna().any():
        missing_rows = (
            merged.loc[merged[resolved_path_column].isna(), ["image_id", "dataset"]]
            .head(5)
            .to_dict(orient="records")
        )
        raise KeyError(f"Missing masked-view paths for rows: {missing_rows}")
    resolved_df = reference_df.copy().reset_index(drop=True)
    resolved_df[resolved_path_column] = merged[resolved_path_column].astype(str)
    resolved_df["path"] = resolved_df[resolved_path_column]
    resolved_df["recommended_model_input_path_v1"] = resolved_df[resolved_path_column]
    applied_ratio = 0.0
    if applied_column in merged.columns:
        applied_ratio = float(pd.Series(merged[applied_column]).fillna(False).astype(bool).mean())
        resolved_df[applied_column] = pd.Series(merged[applied_column]).fillna(False).astype(bool)
    return resolved_df, applied_ratio


def _override_reference_paths_from_enriched(
    reference_df: pd.DataFrame,
    enriched_df: pd.DataFrame,
    *,
    path_column: str,
    enriched_path_column: str = "path",
) -> pd.DataFrame:
    if enriched_path_column not in enriched_df.columns:
        raise KeyError(f"{enriched_path_column} not found in enriched metadata.")
    override_column = f"__override_{enriched_path_column}"
    lookup_df = (
        enriched_df[["image_id", "dataset", enriched_path_column]]
        .rename(columns={enriched_path_column: override_column})
        .drop_duplicates(subset=["image_id", "dataset"])
        .copy()
    )
    lookup_df["image_id"] = lookup_df["image_id"].astype(str)
    lookup_df["dataset"] = lookup_df["dataset"].astype(str)
    merged = reference_df.merge(lookup_df, on=["image_id", "dataset"], how="left", validate="one_to_one")
    resolved_df = reference_df.copy().reset_index(drop=True)
    if merged[override_column].notna().any():
        override = merged[override_column].fillna(resolved_df[path_column]).astype(str)
        resolved_df[path_column] = override
        resolved_df["path"] = override
    return resolved_df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import PATH_COLUMN, build_submission, dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path
    from animalclef_analysis.salamander_pairwise_features import (
        FEATURE_SET_DUAL_VIEW_V1,
        FEATURE_SET_META_GRAPH_YELLOW_V1,
        FEATURE_SET_META_GRAPH_DUAL_VIEW_V1,
        FEATURE_SET_YELLOW_V1,
        append_feature_set,
        build_pair_feature_table,
        resolve_pair_feature_columns,
    )
    from animalclef_analysis.salamander_yellow_xgb_features import build_yellow_pair_feature_artifacts
    from animalclef_analysis.submission_baseline import _cluster_single_dataset_from_score_matrix
    from animalclef_analysis.view_manifests import SAM_MASKED_VIEW_NAME

    parser = argparse.ArgumentParser(description="Build a Salamander XGBoost pairwise-fusion submission variant.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--fusion-source-dir", type=Path, default=DEFAULT_FUSION_SOURCE_DIR)
    parser.add_argument("--student-checkpoint", type=Path, default=DEFAULT_MASKED_SUPCON_CHECKPOINT)
    parser.add_argument("--distill-checkpoint", type=Path, default=DEFAULT_DISTILL_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--route-name", type=str)
    parser.add_argument(
        "--feature-set",
        choices=["basic", "dual_view_v1", "meta_graph_v1", "meta_graph_dual_view_v1", "yellow_v1", "meta_graph_yellow_v1"],
        default="basic",
    )
    parser.add_argument("--graph-top-k", type=int, default=10)
    parser.add_argument("--local-flip-invariant", dest="local_flip_invariant", action="store_true")
    parser.add_argument("--no-local-flip-invariant", dest="local_flip_invariant", action="store_false")
    parser.add_argument("--masked-manifest-root", type=Path, default=DEFAULT_MASKED_MANIFEST_ROOT)
    parser.add_argument("--masked-view-name", type=str, default=SAM_MASKED_VIEW_NAME)
    parser.add_argument("--yellow-local-view", choices=["focus", "band"], default="focus")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--blend-scale", type=float, default=1.0)
    parser.add_argument("--chosen-threshold", type=float, default=DEFAULT_CHOSEN_THRESHOLD)
    parser.add_argument("--threshold-candidates", nargs="+", type=float, default=None)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--reference-probe-dirs", nargs="+", type=Path, default=DEFAULT_REFERENCE_PROBE_DIRS)
    parser.set_defaults(local_flip_invariant=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    route_dir = args.route_dir.resolve()
    base_predictions_path = args.base_predictions.resolve()
    fusion_source_dir = args.fusion_source_dir.resolve()
    student_checkpoint = args.student_checkpoint.resolve()
    distill_checkpoint = args.distill_checkpoint.resolve()
    masked_manifest_root = args.masked_manifest_root.resolve()
    sample_submission_path = args.sample_submission_path.resolve() if args.sample_submission_path else repo_root / "sample_submission.csv"
    threshold_candidates = args.threshold_candidates or DEFAULT_THRESHOLD_CANDIDATES
    use_dual_view = str(args.feature_set) in {FEATURE_SET_DUAL_VIEW_V1, FEATURE_SET_META_GRAPH_DUAL_VIEW_V1}
    use_yellow_features = str(args.feature_set) in {FEATURE_SET_YELLOW_V1, FEATURE_SET_META_GRAPH_YELLOW_V1}
    masked_view_name = str(args.masked_view_name)
    yellow_local_view = str(args.yellow_local_view)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / _default_output_dir_for_feature_set(str(args.feature_set), yellow_local_view)).resolve()
    )
    route_name = (
        str(args.route_name)
        if args.route_name is not None
        else _default_route_name_for_feature_set(str(args.feature_set), yellow_local_view)
    )

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    embeddings_dir = output_dir / "embeddings"
    for path in [output_dir, tables_dir, reports_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_df, route_val_embeddings, test_df, route_test_embeddings = _load_route_bundle(
        route_dir=route_dir,
        repo_root=repo_root,
        path_column=PATH_COLUMN,
        resolver=resolve_existing_image_rel_path,
    )
    student_val_embeddings = _extract_checkpoint_embeddings(
        checkpoint_path=student_checkpoint,
        eval_df=val_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    student_test_embeddings = _extract_checkpoint_embeddings(
        checkpoint_path=student_checkpoint,
        eval_df=test_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    distill_val_embeddings = _extract_checkpoint_embeddings(
        checkpoint_path=distill_checkpoint,
        eval_df=val_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    distill_test_embeddings = _extract_checkpoint_embeddings(
        checkpoint_path=distill_checkpoint,
        eval_df=test_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    masked_val_df = None
    masked_test_df = None
    masked_student_val_embeddings = None
    masked_student_test_embeddings = None
    masked_distill_val_embeddings = None
    masked_distill_test_embeddings = None
    masked_val_applied_ratio = 0.0
    masked_test_applied_ratio = 0.0
    yellow_enriched_df = None
    if use_dual_view:
        if masked_view_name != SAM_MASKED_VIEW_NAME:
            raise ValueError(f"Unsupported masked_view_name: {masked_view_name}")
        masked_val_df, masked_val_applied_ratio = _resolve_masked_view_df(
            val_df,
            manifest_root=masked_manifest_root,
            resolved_path_column="sam_masked_rgb_v1_resolved_path_v1",
        )
        masked_test_df, masked_test_applied_ratio = _resolve_masked_view_df(
            test_df,
            manifest_root=masked_manifest_root,
            resolved_path_column="sam_masked_rgb_v1_resolved_path_v1",
        )
        masked_student_val_embeddings = _extract_checkpoint_embeddings(
            checkpoint_path=student_checkpoint,
            eval_df=masked_val_df,
            repo_root=repo_root,
            device=str(args.device),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        masked_student_test_embeddings = _extract_checkpoint_embeddings(
            checkpoint_path=student_checkpoint,
            eval_df=masked_test_df,
            repo_root=repo_root,
            device=str(args.device),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        masked_distill_val_embeddings = _extract_checkpoint_embeddings(
            checkpoint_path=distill_checkpoint,
            eval_df=masked_val_df,
            repo_root=repo_root,
            device=str(args.device),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        masked_distill_test_embeddings = _extract_checkpoint_embeddings(
            checkpoint_path=distill_checkpoint,
            eval_df=masked_test_df,
            repo_root=repo_root,
            device=str(args.device),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
    if use_yellow_features:
        yellow_enriched_df = pd.read_csv(masked_manifest_root / "tables" / "metadata_enriched_v1.csv")
        yellow_enriched_df["image_id"] = yellow_enriched_df["image_id"].astype(str)
        yellow_enriched_df["dataset"] = yellow_enriched_df["dataset"].astype(str)
        yellow_enriched_df = yellow_enriched_df[yellow_enriched_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    fusion_val_embeddings, fusion_val_source = _load_fusion_branch(
        fusion_dir=fusion_source_dir,
        reference_df=val_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )
    fusion_test_embeddings, fusion_test_source = _load_fusion_branch(
        fusion_dir=fusion_source_dir,
        reference_df=test_df,
        repo_root=repo_root,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
    )

    val_local_pair_df, val_local_match_source = _load_or_build_local_match_table(
        split_name="val",
        route_dir=route_dir,
        output_tables_dir=tables_dir,
        metadata_df=val_df,
        route_embeddings=route_val_embeddings,
        repo_root=repo_root,
        top_k=int(args.graph_top_k),
        flip_invariant=bool(args.local_flip_invariant),
    )
    test_local_pair_df, test_local_match_source = _load_or_build_local_match_table(
        split_name="test",
        route_dir=route_dir,
        output_tables_dir=tables_dir,
        metadata_df=test_df,
        route_embeddings=route_test_embeddings,
        repo_root=repo_root,
        top_k=int(args.graph_top_k),
        flip_invariant=bool(args.local_flip_invariant),
    )

    route_val_score = cosine_score_matrix(route_val_embeddings)
    route_test_score = cosine_score_matrix(route_test_embeddings)
    fusion_val_score = cosine_score_matrix(fusion_val_embeddings)
    fusion_test_score = cosine_score_matrix(fusion_test_embeddings)
    student_val_score = cosine_score_matrix(student_val_embeddings)
    student_test_score = cosine_score_matrix(student_test_embeddings)
    distill_val_score = cosine_score_matrix(distill_val_embeddings)
    distill_test_score = cosine_score_matrix(distill_test_embeddings)
    masked_student_val_score = cosine_score_matrix(masked_student_val_embeddings) if use_dual_view else None
    masked_student_test_score = cosine_score_matrix(masked_student_test_embeddings) if use_dual_view else None
    masked_distill_val_score = cosine_score_matrix(masked_distill_val_embeddings) if use_dual_view else None
    masked_distill_test_score = cosine_score_matrix(masked_distill_test_embeddings) if use_dual_view else None
    student_val_cross_score = _cosine_cross_score_matrix(student_val_embeddings, masked_student_val_embeddings) if use_dual_view else None
    student_test_cross_score = _cosine_cross_score_matrix(student_test_embeddings, masked_student_test_embeddings) if use_dual_view else None
    distill_val_cross_score = _cosine_cross_score_matrix(distill_val_embeddings, masked_distill_val_embeddings) if use_dual_view else None
    distill_test_cross_score = _cosine_cross_score_matrix(distill_test_embeddings, masked_distill_test_embeddings) if use_dual_view else None

    val_pair_df = build_pair_feature_table(
        metadata_df=val_df,
        local_pair_df=val_local_pair_df,
        route_score=route_val_score,
        fusion_score=fusion_val_score,
        student_score=student_val_score,
        distill_score=distill_val_score,
    )
    test_pair_df = build_pair_feature_table(
        metadata_df=test_df,
        local_pair_df=test_local_pair_df,
        route_score=route_test_score,
        fusion_score=fusion_test_score,
        student_score=student_test_score,
        distill_score=distill_test_score,
    )
    yellow_val_pair_df = None
    yellow_test_pair_df = None
    if use_yellow_features:
        if yellow_enriched_df is None:
            raise ValueError("Yellow feature set requires enriched manifest metadata.")
        yellow_val_reference_df = _override_reference_paths_from_enriched(
            reference_df=val_df,
            enriched_df=yellow_enriched_df,
            path_column=PATH_COLUMN,
        )
        yellow_test_reference_df = _override_reference_paths_from_enriched(
            reference_df=test_df,
            enriched_df=yellow_enriched_df,
            path_column=PATH_COLUMN,
        )
        yellow_val_artifacts = build_yellow_pair_feature_artifacts(
            reference_df=yellow_val_reference_df,
            pair_df=val_pair_df,
            enriched_df=yellow_enriched_df,
            repo_root=repo_root,
            output_dir=output_dir,
            yellow_local_view=yellow_local_view,
        )
        yellow_test_artifacts = build_yellow_pair_feature_artifacts(
            reference_df=yellow_test_reference_df,
            pair_df=test_pair_df,
            enriched_df=yellow_enriched_df,
            repo_root=repo_root,
            output_dir=output_dir,
            yellow_local_view=yellow_local_view,
        )
        yellow_val_pair_df = yellow_val_artifacts.pair_feature_df
        yellow_test_pair_df = yellow_test_artifacts.pair_feature_df
        yellow_val_artifacts.roi_manifest_df.to_csv(tables_dir / "val_yellow_roi_manifest_v1.csv", index=False)
        yellow_test_artifacts.roi_manifest_df.to_csv(tables_dir / "test_yellow_roi_manifest_v1.csv", index=False)
        yellow_val_artifacts.roi_summary_df.to_csv(tables_dir / "val_yellow_roi_summary_v1.csv", index=False)
        yellow_test_artifacts.roi_summary_df.to_csv(tables_dir / "test_yellow_roi_summary_v1.csv", index=False)
        yellow_val_artifacts.focus_df.to_csv(tables_dir / "val_yellow_focus_manifest_v1.csv", index=False)
        yellow_test_artifacts.focus_df.to_csv(tables_dir / "test_yellow_focus_manifest_v1.csv", index=False)
        yellow_val_artifacts.focus_summary_df.to_csv(tables_dir / "val_yellow_focus_summary_v1.csv", index=False)
        yellow_test_artifacts.focus_summary_df.to_csv(tables_dir / "test_yellow_focus_summary_v1.csv", index=False)
        if not yellow_val_artifacts.band_df.empty or not yellow_test_artifacts.band_df.empty:
            yellow_val_artifacts.band_df.to_csv(tables_dir / "val_yellow_band_manifest_v1.csv", index=False)
            yellow_test_artifacts.band_df.to_csv(tables_dir / "test_yellow_band_manifest_v1.csv", index=False)
            yellow_val_artifacts.band_summary_df.to_csv(tables_dir / "val_yellow_band_summary_v1.csv", index=False)
            yellow_test_artifacts.band_summary_df.to_csv(tables_dir / "test_yellow_band_summary_v1.csv", index=False)
        yellow_val_artifacts.yellow_roi_local_df.to_csv(tables_dir / "val_yellow_roi_local_scores_v1.csv", index=False)
        yellow_test_artifacts.yellow_roi_local_df.to_csv(tables_dir / "test_yellow_roi_local_scores_v1.csv", index=False)
        yellow_val_artifacts.patch_pair_df.to_csv(tables_dir / "val_yellow_patch_pair_features_v1.csv", index=False)
        yellow_test_artifacts.patch_pair_df.to_csv(tables_dir / "test_yellow_patch_pair_features_v1.csv", index=False)
        yellow_val_artifacts.patch_summary_df.to_csv(tables_dir / "val_yellow_patch_summary_v1.csv", index=False)
        yellow_test_artifacts.patch_summary_df.to_csv(tables_dir / "test_yellow_patch_summary_v1.csv", index=False)
        yellow_val_artifacts.decision_summary_df.to_csv(tables_dir / "val_yellow_decision_summary_v1.csv", index=False)
        yellow_test_artifacts.decision_summary_df.to_csv(tables_dir / "test_yellow_decision_summary_v1.csv", index=False)
    val_pair_df = append_feature_set(
        pair_df=val_pair_df,
        metadata_df=val_df,
        route_score=route_val_score,
        feature_set=str(args.feature_set),
        graph_top_k=int(args.graph_top_k),
        masked_student_score=masked_student_val_score,
        masked_distill_score=masked_distill_val_score,
        student_cross_score=student_val_cross_score,
        distill_cross_score=distill_val_cross_score,
        yellow_pair_df=yellow_val_pair_df,
    )
    test_pair_df = append_feature_set(
        pair_df=test_pair_df,
        metadata_df=test_df,
        route_score=route_test_score,
        feature_set=str(args.feature_set),
        graph_top_k=int(args.graph_top_k),
        masked_student_score=masked_student_test_score,
        masked_distill_score=masked_distill_test_score,
        student_cross_score=student_test_cross_score,
        distill_cross_score=distill_test_cross_score,
        yellow_pair_df=yellow_test_pair_df,
    )
    feature_columns = resolve_pair_feature_columns(str(args.feature_set))
    model = XGBClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=8,
        tree_method="hist",
    )
    val_y = val_pair_df["same_identity"].to_numpy(dtype=int)
    val_weight = _balanced_sample_weight(val_y)
    model.fit(val_pair_df[feature_columns].to_numpy(dtype=np.float32), val_y, sample_weight=val_weight)
    val_pair_df["xgb_same_identity_prob"] = model.predict_proba(val_pair_df[feature_columns].to_numpy(dtype=np.float32))[:, 1]
    test_pair_df["xgb_same_identity_prob"] = model.predict_proba(test_pair_df[feature_columns].to_numpy(dtype=np.float32))[:, 1]

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)
    feature_importance_df.to_csv(tables_dir / "feature_importance_v1.csv", index=False)

    boosted_test_score = _apply_pair_probability_as_score(
        base_score=route_test_score,
        pair_df=test_pair_df,
        probability_col="xgb_same_identity_prob",
        blend_scale=float(args.blend_scale),
    )
    threshold_rows: list[dict[str, object]] = []
    threshold_prediction_frames: dict[float, pd.DataFrame] = {}
    for threshold in [float(value) for value in threshold_candidates]:
        pred_df = _cluster_single_dataset_from_score_matrix(
            dataset_df=test_df,
            score_matrix=boosted_test_score,
            threshold=float(threshold),
        )
        pred_df["route_name"] = route_name
        pred_df["embedding_dim"] = int(route_test_embeddings.shape[1])
        pred_df["rerank_enabled"] = True
        pred_df["local_weight"] = 0.0
        pred_df["pairwise_model"] = "xgboost"
        pred_df["blend_mode"] = "boost"
        pred_df["blend_scale"] = float(args.blend_scale)
        pred_df["pair_feature_set"] = str(args.feature_set)
        counts = pred_df["pred_cluster_id"].value_counts()
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "largest_cluster": int(counts.max()) if len(counts) else 0,
            }
        )
        threshold_prediction_frames[float(threshold)] = pred_df
    threshold_summary_df = pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)
    threshold_summary_df.to_csv(tables_dir / "test_threshold_candidates_v1.csv", index=False)

    chosen_threshold = float(args.chosen_threshold)
    if chosen_threshold not in threshold_prediction_frames:
        chosen_pred_df = _cluster_single_dataset_from_score_matrix(
            dataset_df=test_df,
            score_matrix=boosted_test_score,
            threshold=chosen_threshold,
        )
        chosen_pred_df["route_name"] = route_name
        chosen_pred_df["embedding_dim"] = int(route_test_embeddings.shape[1])
        chosen_pred_df["rerank_enabled"] = True
        chosen_pred_df["local_weight"] = 0.0
        chosen_pred_df["pairwise_model"] = "xgboost"
        chosen_pred_df["blend_mode"] = "boost"
        chosen_pred_df["blend_scale"] = float(args.blend_scale)
        chosen_pred_df["pair_feature_set"] = str(args.feature_set)
    else:
        chosen_pred_df = threshold_prediction_frames[chosen_threshold].copy()
    chosen_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)

    base_pred_df = pd.read_csv(base_predictions_path)
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    base_salamander_pred_df = base_pred_df[base_pred_df["dataset"] == SALAMANDER_DATASET].copy()
    merged_pred_df = pd.concat(
        [base_pred_df[base_pred_df["dataset"] != SALAMANDER_DATASET].copy(), chosen_pred_df],
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
        merged_pred_df[
            [
                "dataset",
                "route_name",
                "embedding_dim",
                "chosen_threshold",
                "rerank_enabled",
                "local_weight",
            ]
        ]
        .drop_duplicates(subset=["dataset"])
        .rename(columns={"chosen_threshold": "threshold"})
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    route_summary_df["pairwise_model"] = np.where(route_summary_df["dataset"] == SALAMANDER_DATASET, "xgboost", "")
    route_summary_df["pair_feature_set"] = np.where(
        route_summary_df["dataset"] == SALAMANDER_DATASET,
        str(args.feature_set),
        "",
    )
    route_summary_df.to_csv(tables_dir / "route_config_v1.csv", index=False)

    probe_evidence_df = _load_probe_evidence([path.resolve() for path in args.reference_probe_dirs])
    if not probe_evidence_df.empty:
        probe_evidence_df.to_csv(tables_dir / "internal_split_evidence_v1.csv", index=False)

    np.save(embeddings_dir / "salamander_route_test_embeddings.npy", route_test_embeddings.astype(np.float32))
    if use_dual_view:
        np.save(embeddings_dir / "salamander_masked_student_test_embeddings.npy", masked_student_test_embeddings.astype(np.float32))
        np.save(embeddings_dir / "salamander_masked_distill_test_embeddings.npy", masked_distill_test_embeddings.astype(np.float32))
    test_df.to_csv(embeddings_dir / "salamander_test_metadata.csv", index=False)
    if use_dual_view and masked_test_df is not None:
        masked_test_df.to_csv(embeddings_dir / "salamander_test_metadata_sam_masked_rgb_v1.csv", index=False)
    val_pair_df.to_csv(tables_dir / "val_pair_features_v1.csv", index=False)
    test_pair_df.to_csv(tables_dir / "test_pair_features_v1.csv", index=False)

    config = {
        "route_dir": str(route_dir),
        "base_predictions": str(base_predictions_path),
        "student_checkpoint": str(student_checkpoint),
        "distill_checkpoint": str(distill_checkpoint),
        "fusion_source_dir": str(fusion_source_dir),
        "fusion_val_source": str(fusion_val_source),
        "fusion_test_source": str(fusion_test_source),
        "route_name": route_name,
        "feature_set": str(args.feature_set),
        "val_local_match_source": str(val_local_match_source),
        "test_local_match_source": str(test_local_match_source),
        "local_flip_invariant": bool(args.local_flip_invariant),
        "graph_top_k": int(args.graph_top_k),
        "masked_manifest_root": str(masked_manifest_root) if use_dual_view else "",
        "masked_view_name": masked_view_name if use_dual_view else "",
        "masked_val_applied_ratio": round(masked_val_applied_ratio, 6) if use_dual_view else 0.0,
        "masked_test_applied_ratio": round(masked_test_applied_ratio, 6) if use_dual_view else 0.0,
        "yellow_enabled": bool(use_yellow_features),
        "yellow_local_view": yellow_local_view if use_yellow_features else "",
        "feature_columns": feature_columns,
        "device": str(args.device),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "xgboost_version": str(xgboost.__version__),
        "xgboost_config": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "max_depth": int(args.max_depth),
            "subsample": float(args.subsample),
            "blend_scale": float(args.blend_scale),
        },
        "chosen_threshold": chosen_threshold,
        "threshold_candidates": [float(value) for value in threshold_candidates],
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    if base_salamander_pred_df.empty:
        raise ValueError("Base predictions do not contain Salamander rows.")
    base_salamander_counts = base_salamander_pred_df["pred_cluster_id"].value_counts()
    base_salamander_summary = {
        "route": str(base_salamander_pred_df["route_name"].iloc[0]) if "route_name" in base_salamander_pred_df.columns else "base_predictions_salamander",
        "threshold": float(base_salamander_pred_df["chosen_threshold"].iloc[0]) if "chosen_threshold" in base_salamander_pred_df.columns else np.nan,
        "clusters": int(base_salamander_counts.size),
        "singleton_clusters": int((base_salamander_counts == 1).sum()),
        "singleton_ratio": float(round(float((base_salamander_counts == 1).mean()) if len(base_salamander_counts) else 0.0, 6)),
    }
    chosen_test_row = threshold_summary_df[threshold_summary_df["threshold"] == chosen_threshold].iloc[0]

    lines = [
        "# Salamander XGBoost Submission Variant",
        "",
        f"- Route name: `{route_name}`",
        f"- Base predictions: `{base_predictions_path}`",
        f"- Student checkpoint: `{student_checkpoint}`",
        f"- Distill checkpoint: `{distill_checkpoint}`",
        f"- Fusion source: `{fusion_source_dir}`",
        f"- Fusion val source: `{fusion_val_source}`",
        f"- Fusion test source: `{fusion_test_source}`",
        f"- Device: `{args.device}`",
        f"- XGBoost version: `{xgboost.__version__}`",
        f"- Pair feature set: `{args.feature_set}`",
        f"- Val local match source: `{val_local_match_source}`",
        f"- Test local match source: `{test_local_match_source}`",
        f"- Local flip invariant: `{bool(args.local_flip_invariant)}`",
        f"- Route graph top-k: `{int(args.graph_top_k)}`",
        f"- Chosen Salamander threshold: `{chosen_threshold}`",
        f"- Blend rule: `boost` with scale `{float(args.blend_scale)}`",
        (
            f"- Masked view: `{masked_view_name}` from `{masked_manifest_root}`; "
            f"val applied ratio `{masked_val_applied_ratio:.4f}`, test applied ratio `{masked_test_applied_ratio:.4f}`"
            if use_dual_view
            else "- Masked view: `disabled`"
        ),
        (
            f"- Yellow local features: `enabled` from `{masked_manifest_root}` using `yellow_{yellow_local_view}` local match input + patch metrics"
            if use_yellow_features
            else "- Yellow local features: `disabled`"
        ),
        "",
        "## Threshold Candidates",
        "",
        dataframe_to_markdown_table(threshold_summary_df),
        "",
        "## Feature Importance",
        "",
        dataframe_to_markdown_table(feature_importance_df.head(12)),
        "",
        "## Internal Split Evidence",
        "",
    ]
    if probe_evidence_df.empty:
        lines.extend(["- No reference probe tables were found."])
    else:
        lines.extend(
            [
                "- These rows come from prior internal `calib/eval` split probes and are the main reason this candidate deserves one official slot.",
                "",
                dataframe_to_markdown_table(
                    probe_evidence_df[
                        [
                            "probe_dir",
                            "route",
                            "threshold",
                            "ari",
                            "pairwise_f1",
                            "cluster_count",
                            "ari_delta_vs_baseline",
                            "pairwise_f1_delta_vs_baseline",
                        ]
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Test Cluster Comparison",
            "",
            dataframe_to_markdown_table(
                pd.DataFrame(
                    [
                        {
                            "route": str(base_salamander_summary["route"]),
                            "threshold": float(base_salamander_summary["threshold"]),
                            "clusters": int(base_salamander_summary["clusters"]),
                            "singleton_clusters": int(base_salamander_summary["singleton_clusters"]),
                            "singleton_ratio": float(base_salamander_summary["singleton_ratio"]),
                        },
                        {
                            "route": route_name,
                            "threshold": chosen_threshold,
                            "clusters": int(chosen_test_row["clusters"]),
                            "singleton_clusters": int(chosen_test_row["singleton_clusters"]),
                            "singleton_ratio": float(chosen_test_row["singleton_ratio"]),
                        },
                    ]
                )
            ),
            "",
            "## Architecture",
            "",
            "- Overall system: `dataset-routed hybrid clustering pipeline`.",
            (
                "- Salamander branch: `image -> original-view masked_supcon student(512) + original-view distill student(512) "
                "+ frozen fusion(3688) -> top-10 candidate pairs -> XGBoost pairwise fusion using multi-branch global + ORB local features "
                "+ optional SAM-masked dual-view scores -> average-linkage clustering`."
                if use_dual_view
                else "- Salamander branch: `image -> masked_supcon student(512) + frozen fusion(3688) -> 4200-d global embedding -> top-10 candidate pairs -> XGBoost pairwise fusion using multi-branch global + ORB local features -> average-linkage clustering`."
            ),
            (
                f"- Salamander pair feature set: `{args.feature_set}`; dual-view adds original-vs-masked student/distill scores, "
                "cross-view compatibility scores, and optional route-graph support features."
                if use_dual_view
                else (
                    f"- Salamander pair feature set: `{args.feature_set}`; yellow sets add `yellow_{yellow_local_view}` local scores, patch correlation, and learned support/fail flags."
                    if use_yellow_features
                    else f"- Salamander pair feature set: `{args.feature_set}`; `meta_graph_v1` adds orientation/date metadata and route-graph support features."
                )
            ),
            f"- Salamander threshold: `{chosen_threshold}`.",
            "- Lynx branch: `ft_mega_arcface_distill_v1`.",
            "- SeaTurtle branch: `fusion_v1`.",
            "- Texas branch: `ft_texas_miew_pseudo_v1`.",
            "",
            "## Single-Factor Change",
            "",
            "- Baseline being compared against: the current Salamander `XGBoost` route when `feature_set=meta_graph_v1`, otherwise the older `fusion + ORB` anchor.",
            (
                f"- Intended only change: keep non-Salamander datasets fixed and run Salamander `XGBoost` with pair feature set `{args.feature_set}` using additional `{masked_view_name}` dual-view evidence."
                if use_dual_view
                else (
                    f"- Intended only change: keep non-Salamander datasets fixed and run Salamander `XGBoost` with pair feature set `{args.feature_set}` using additional `yellow_{yellow_local_view}` local features."
                    if use_yellow_features
                    else f"- Intended only change: keep non-Salamander datasets fixed and run Salamander `XGBoost` with pair feature set `{args.feature_set}`."
                )
            ),
            (
                "- Main expected risk: SAM may over-crop or fragment a few animals, so dual-view could help true same-ID pairs while also creating masked-view false negatives."
                if use_dual_view
                else "- Main expected risk: richer pairwise features may fit local validation structure better than Kaggle test structure."
            ),
            "",
        ]
    )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_xgboost_submission_variant] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_xgboost_submission_variant] route_config: {tables_dir / 'route_config_v1.csv'}")
    print(f"[salamander_xgboost_submission_variant] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
