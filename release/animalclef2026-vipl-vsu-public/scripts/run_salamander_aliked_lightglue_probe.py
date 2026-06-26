#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as T


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_aliked_lightglue_probe_20260330")
DEFAULT_LOCAL_WEIGHTS = [0.25, 0.5, 0.75, 1.0]
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
SALAMANDER_DATASET = "SalamanderID2025"


@dataclass(frozen=True)
class LocalViewArtifacts:
    metadata_df: pd.DataFrame
    roi_manifest_df: pd.DataFrame
    roi_summary_df: pd.DataFrame
    focus_df: pd.DataFrame
    focus_summary_df: pd.DataFrame
    band_df: pd.DataFrame
    band_summary_df: pd.DataFrame
    resolved_ratio: float


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


def _default_route_name(local_view: str) -> str:
    base = "ft_miew_arcface_masked_supcon_v1_last_fusion_aliked_lightglue"
    if str(local_view) == "original":
        return f"{base}_v1"
    return f"{base}_{str(local_view)}_v1"


def _load_enriched_manifest(manifest_root: Path) -> pd.DataFrame:
    enriched_df = pd.read_csv(manifest_root / "tables" / "metadata_enriched_v1.csv")
    enriched_df["image_id"] = enriched_df["image_id"].astype(str)
    enriched_df["dataset"] = enriched_df["dataset"].astype(str)
    return enriched_df[enriched_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)


def _prepare_local_view_artifacts(
    *,
    metadata_df: pd.DataFrame,
    repo_root: Path,
    manifest_root: Path,
    local_view: str,
    output_dir: Path,
) -> LocalViewArtifacts:
    from animalclef_analysis.salamander_yellow_orb_local import (
        YELLOW_BAND_PATH_COLUMN,
        YELLOW_FOCUS_PATH_COLUMN,
        build_yellow_band_manifest,
        build_yellow_focus_manifest,
        summarize_yellow_band_manifest,
        summarize_yellow_focus_manifest,
    )
    from animalclef_analysis.sam_orb_veto import (
        MASKED_PATH_COLUMN,
        build_masked_aligned_roi_manifest,
        summarize_roi_manifest,
    )

    local_view = str(local_view)
    if local_view == "original":
        return LocalViewArtifacts(
            metadata_df=metadata_df.copy().reset_index(drop=True),
            roi_manifest_df=pd.DataFrame(),
            roi_summary_df=pd.DataFrame(),
            focus_df=pd.DataFrame(),
            focus_summary_df=pd.DataFrame(),
            band_df=pd.DataFrame(),
            band_summary_df=pd.DataFrame(),
            resolved_ratio=0.0,
        )

    working_df = metadata_df.copy().reset_index(drop=True)
    working_df["image_id"] = working_df["image_id"].astype(str)
    working_df["dataset"] = working_df["dataset"].astype(str)
    if "identity" in working_df.columns:
        working_df["identity"] = working_df["identity"].fillna("").astype(str)

    enriched_df = _load_enriched_manifest(manifest_root)
    roi_manifest_df = build_masked_aligned_roi_manifest(
        reference_df=working_df,
        enriched_df=enriched_df,
        repo_root=repo_root,
        output_dir=output_dir,
    )
    roi_summary_df = summarize_roi_manifest(roi_manifest_df=roi_manifest_df)
    focus_df = pd.DataFrame()
    focus_summary_df = pd.DataFrame()
    band_df = pd.DataFrame()
    band_summary_df = pd.DataFrame()

    resolved_df = working_df.copy()
    if local_view == "sam_masked":
        resolved_df = resolved_df.merge(
            roi_manifest_df[["image_id", "dataset", MASKED_PATH_COLUMN]],
            on=["image_id", "dataset"],
            how="left",
        )
        masked_path = resolved_df[MASKED_PATH_COLUMN].fillna("").astype(str)
        resolved_df["path"] = np.where(masked_path.ne(""), masked_path, resolved_df["path"].astype(str))
    else:
        focus_df = build_yellow_focus_manifest(
            roi_manifest_df=roi_manifest_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        focus_summary_df = summarize_yellow_focus_manifest(focus_df=focus_df)
        resolved_df = resolved_df.merge(
            focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
            on=["image_id", "dataset"],
            how="left",
        )
        focus_path = resolved_df[YELLOW_FOCUS_PATH_COLUMN].fillna("").astype(str)
        if local_view == "yellow_focus":
            resolved_df["path"] = np.where(focus_path.ne(""), focus_path, resolved_df["path"].astype(str))
        else:
            band_df = build_yellow_band_manifest(
                focus_df=focus_df,
                repo_root=repo_root,
                output_dir=output_dir,
            )
            band_summary_df = summarize_yellow_band_manifest(band_df=band_df)
            resolved_df = resolved_df.merge(
                band_df[["image_id", "dataset", YELLOW_BAND_PATH_COLUMN]],
                on=["image_id", "dataset"],
                how="left",
            )
            band_path = resolved_df[YELLOW_BAND_PATH_COLUMN].fillna("").astype(str)
            resolved_df["path"] = np.where(
                band_path.ne(""),
                band_path,
                np.where(focus_path.ne(""), focus_path, resolved_df["path"].astype(str)),
            )

    resolved_ratio = float((resolved_df["path"].astype(str) != working_df["path"].astype(str)).mean())
    return LocalViewArtifacts(
        metadata_df=resolved_df.reset_index(drop=True),
        roi_manifest_df=roi_manifest_df.reset_index(drop=True),
        roi_summary_df=roi_summary_df.reset_index(drop=True),
        focus_df=focus_df.reset_index(drop=True),
        focus_summary_df=focus_summary_df.reset_index(drop=True),
        band_df=band_df.reset_index(drop=True),
        band_summary_df=band_summary_df.reset_index(drop=True),
        resolved_ratio=resolved_ratio,
    )


def _build_image_dataset(metadata_df: pd.DataFrame, repo_root: Path):
    from wildlife_tools.data.dataset import ImageDataset

    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    return ImageDataset(
        metadata=metadata_df[["path", "identity"]].copy(),
        root=str(repo_root),
        transform=transform,
        col_path="path",
        col_label="identity",
        load_label=True,
    )


def _compute_lightglue_match_rows(
    metadata_df: pd.DataFrame,
    feature_dataset,
    pair_index: list[tuple[int, int, float]],
    device: str,
    init_threshold: float,
    batch_size: int,
    ransac_threshold: float,
    min_inliers: int,
) -> pd.DataFrame:
    from wildlife_tools.similarity.pairwise.collectors import CollectAll
    from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

    matcher = MatchLightGlue(
        features="aliked",
        init_threshold=float(init_threshold),
        device=device,
        batch_size=int(batch_size),
        num_workers=0,
        collector=CollectAll(),
        tqdm_silent=False,
    )
    pair_array = np.array([[left_index, right_index] for left_index, right_index, _ in pair_index], dtype=np.int32)
    global_score_map = {(int(left), int(right)): float(score) for left, right, score in pair_index}
    raw_results = matcher(feature_dataset, feature_dataset, pairs=pair_array)

    rows: list[dict[str, object]] = []
    for item in raw_results:
        left_index = int(item["idx0"])
        right_index = int(item["idx1"])
        left_row = metadata_df.iloc[left_index]
        right_row = metadata_df.iloc[right_index]
        match_scores = np.asarray(item["scores"], dtype=np.float32)
        kpts0 = np.asarray(item["kpts0"], dtype=np.float32)
        kpts1 = np.asarray(item["kpts1"], dtype=np.float32)
        good_matches = int(len(match_scores))
        mean_match_score = float(match_scores.mean()) if len(match_scores) else 0.0
        if len(kpts0) < 4 or len(kpts1) < 4:
            inliers = 0
        else:
            try:
                _homography, mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, float(ransac_threshold))
            except cv2.error:
                mask = None
            inliers = int(mask.sum()) if mask is not None else 0
        if inliers < int(min_inliers):
            local_raw_score = 0.0
        else:
            local_raw_score = float((inliers * max(mean_match_score, 1e-6)) / max(1, min(len(feature_dataset[left_index][0]["keypoints"]), len(feature_dataset[right_index][0]["keypoints"]))))
        rows.append(
            {
                "dataset": str(left_row["dataset"]),
                "matcher_name": "aliked_lightglue",
                "left_index": left_index,
                "right_index": right_index,
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "identity": str(left_row["identity"]),
                "neighbor_identity": str(right_row["identity"]),
                "same_identity": bool(str(left_row["identity"]) == str(right_row["identity"])),
                "global_score": round(global_score_map[(left_index, right_index)], 6),
                "left_keypoints": int(len(feature_dataset[left_index][0]["keypoints"])),
                "right_keypoints": int(len(feature_dataset[right_index][0]["keypoints"])),
                "good_matches": good_matches,
                "inliers": int(inliers),
                "mean_match_score": round(mean_match_score, 6),
                "local_raw_score": round(local_raw_score, 6),
            }
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        pair_df["local_score"] = pd.Series(dtype=float)
        return pair_df
    nonzero = pair_df.loc[pair_df["local_raw_score"] > 0.0, "local_raw_score"].to_numpy(dtype=float)
    if len(nonzero) == 0:
        pair_df["local_score"] = 0.0
        return pair_df
    upper = float(np.quantile(nonzero, 0.95))
    upper = max(upper, 1e-6)
    pair_df["local_score"] = np.round(np.clip(pair_df["local_raw_score"].to_numpy(dtype=float) / upper, 0.0, 1.0), 6)
    return pair_df


def _build_test_predictions(
    dataset_df: pd.DataFrame,
    score_matrix: np.ndarray,
    threshold: float,
    embedding_dim: int,
    local_weight: float,
    route_name: str,
    local_matcher: str,
) -> pd.DataFrame:
    from animalclef_analysis.descriptor_baselines import build_average_linkage, cluster_from_linkage
    from animalclef_analysis.orb_rerank_baseline import score_matrix_to_distance

    distance = score_matrix_to_distance(score_matrix)
    linkage_matrix = build_average_linkage(distance)
    pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), float(threshold))

    result = dataset_df.copy().reset_index(drop=True)
    result["chosen_threshold"] = float(threshold)
    result["pred_cluster_id"] = pred_labels
    result["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in pred_labels]
    result["route_name"] = str(route_name)
    result["embedding_dim"] = int(embedding_dim)
    result["rerank_enabled"] = True
    result["local_weight"] = float(local_weight)
    result["local_matcher"] = str(local_matcher)
    return result


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import PATH_COLUMN, build_submission, dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import (
        apply_local_rerank,
        build_topk_pair_index,
        cosine_score_matrix,
        evaluate_threshold_sweep_from_score_matrix,
        resolve_existing_image_rel_path,
    )
    from wildlife_tools.features.local import AlikedExtractor

    parser = argparse.ArgumentParser(description="Run Salamander ALIKED + LightGlue local rerank probe on top of fixed global embeddings.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--route-name", type=str)
    parser.add_argument("--local-view", choices=["original", "sam_masked", "yellow_focus", "yellow_band"], default="original")
    parser.add_argument("--masked-manifest-root", type=Path, default=Path("artifacts/manifests/v1"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--local-weights", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-num-keypoints", type=int, default=256)
    parser.add_argument("--lightglue-init-threshold", type=float, default=0.1)
    parser.add_argument("--matcher-batch-size", type=int, default=32)
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
    local_view = str(args.local_view)
    route_name = str(args.route_name) if args.route_name else _default_route_name(local_view)
    masked_manifest_root = args.masked_manifest_root.resolve()

    val_embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy")
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy")
    val_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    test_df = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    for frame in [val_df, test_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        if "identity" in frame.columns:
            frame["identity"] = frame["identity"].fillna("").astype(str)
        frame[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in frame.iterrows()]
        frame["path"] = frame[PATH_COLUMN]
        frame["dataset"] = frame["dataset"].astype(str)

    val_view_artifacts = _prepare_local_view_artifacts(
        metadata_df=val_df,
        repo_root=repo_root,
        manifest_root=masked_manifest_root,
        local_view=local_view,
        output_dir=output_dir / "val_local_view_cache",
    )
    test_view_artifacts = _prepare_local_view_artifacts(
        metadata_df=test_df,
        repo_root=repo_root,
        manifest_root=masked_manifest_root,
        local_view=local_view,
        output_dir=output_dir / "test_local_view_cache",
    )
    val_df = val_view_artifacts.metadata_df
    test_df = test_view_artifacts.metadata_df

    base_pred_df = pd.read_csv(route_dir / "tables" / "test_predictions_v1.csv")
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)

    orb_best_df = pd.read_csv(route_dir / "tables" / "best_config_v1.csv")
    orb_best = orb_best_df.iloc[0]
    orb_cluster_df = pd.read_csv(route_dir / "tables" / "cluster_summary_v1.csv")
    orb_salamander_cluster = orb_cluster_df[orb_cluster_df["dataset"] == SALAMANDER_DATASET].iloc[0]

    if not val_view_artifacts.roi_manifest_df.empty:
        val_view_artifacts.roi_manifest_df.to_csv(tables_dir / "val_roi_manifest_v1.csv", index=False)
        val_view_artifacts.roi_summary_df.to_csv(tables_dir / "val_roi_summary_v1.csv", index=False)
    if not test_view_artifacts.roi_manifest_df.empty:
        test_view_artifacts.roi_manifest_df.to_csv(tables_dir / "test_roi_manifest_v1.csv", index=False)
        test_view_artifacts.roi_summary_df.to_csv(tables_dir / "test_roi_summary_v1.csv", index=False)
    if not val_view_artifacts.focus_df.empty:
        val_view_artifacts.focus_df.to_csv(tables_dir / "val_yellow_focus_manifest_v1.csv", index=False)
        val_view_artifacts.focus_summary_df.to_csv(tables_dir / "val_yellow_focus_summary_v1.csv", index=False)
    if not test_view_artifacts.focus_df.empty:
        test_view_artifacts.focus_df.to_csv(tables_dir / "test_yellow_focus_manifest_v1.csv", index=False)
        test_view_artifacts.focus_summary_df.to_csv(tables_dir / "test_yellow_focus_summary_v1.csv", index=False)
    if not val_view_artifacts.band_df.empty:
        val_view_artifacts.band_df.to_csv(tables_dir / "val_yellow_band_manifest_v1.csv", index=False)
        val_view_artifacts.band_summary_df.to_csv(tables_dir / "val_yellow_band_summary_v1.csv", index=False)
    if not test_view_artifacts.band_df.empty:
        test_view_artifacts.band_df.to_csv(tables_dir / "test_yellow_band_manifest_v1.csv", index=False)
        test_view_artifacts.band_summary_df.to_csv(tables_dir / "test_yellow_band_summary_v1.csv", index=False)

    val_score = cosine_score_matrix(val_embeddings)
    test_score = cosine_score_matrix(test_embeddings)
    val_pair_index = build_topk_pair_index(score_matrix=val_score, top_k=int(args.top_k), query_indices=None)
    test_pair_index = build_topk_pair_index(score_matrix=test_score, top_k=int(args.top_k), query_indices=None)

    val_image_ds = _build_image_dataset(val_df, repo_root=repo_root)
    test_image_ds = _build_image_dataset(test_df, repo_root=repo_root)
    extractor = AlikedExtractor(device=str(args.device), max_num_keypoints=int(args.max_num_keypoints))
    val_feature_ds = extractor(val_image_ds)
    test_feature_ds = extractor(test_image_ds)

    val_pair_df = _compute_lightglue_match_rows(
        metadata_df=val_df,
        feature_dataset=val_feature_ds,
        pair_index=val_pair_index,
        device=str(args.device),
        init_threshold=float(args.lightglue_init_threshold),
        batch_size=int(args.matcher_batch_size),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
    )
    test_pair_df = _compute_lightglue_match_rows(
        metadata_df=test_df,
        feature_dataset=test_feature_ds,
        pair_index=test_pair_index,
        device=str(args.device),
        init_threshold=float(args.lightglue_init_threshold),
        batch_size=int(args.matcher_batch_size),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
    )
    val_pair_df.to_csv(tables_dir / "val_local_match_scores_v1.csv", index=False)
    test_pair_df.to_csv(tables_dir / "test_local_match_scores_v1.csv", index=False)

    pd.DataFrame(
        [{"image_id": str(val_df.iloc[idx]["image_id"]), "keypoints": int(len(val_feature_ds[idx][0]["keypoints"]))} for idx in range(len(val_feature_ds))]
    ).to_csv(tables_dir / "val_keypoints_v1.csv", index=False)
    pd.DataFrame(
        [{"image_id": str(test_df.iloc[idx]["image_id"]), "keypoints": int(len(test_feature_ds[idx][0]["keypoints"]))} for idx in range(len(test_feature_ds))]
    ).to_csv(tables_dir / "test_keypoints_v1.csv", index=False)

    summary_rows = []
    best_meta = None
    best_test_score = None
    for local_weight in local_weights:
        reranked_val_score = apply_local_rerank(val_score, val_pair_df, float(local_weight))
        sweep_df, _prediction_df = evaluate_threshold_sweep_from_score_matrix(
            df=val_df,
            score_matrix=reranked_val_score,
            thresholds=thresholds,
        )
        sweep_df["local_weight"] = float(local_weight)
        sweep_df.to_csv(tables_dir / f"val_threshold_sweep_w{str(local_weight).replace('.', 'p')}_v1.csv", index=False)
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
        if best_meta is None or _metric_key(best_row) > _metric_key(pd.Series(best_meta)):
            best_meta = best_row.to_dict()
            best_test_score = apply_local_rerank(test_score, test_pair_df, float(local_weight))

    if best_meta is None or best_test_score is None:
        raise RuntimeError("Failed to select best ALIKED + LightGlue config")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "local_weight"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    summary_df.to_csv(tables_dir / "val_weight_summary_v1.csv", index=False)
    pd.DataFrame([best_meta]).to_csv(tables_dir / "best_config_v1.csv", index=False)

    comparison_df = pd.DataFrame(
        [
            {
                "route": "current_orb_anchor",
                "threshold": float(orb_best["threshold"]),
                "local_weight": float(orb_best["local_weight"]),
                "ari": float(orb_best["ari"]),
                "nmi": float(orb_best["nmi"]),
                "pairwise_f1": float(orb_best["pairwise_f1"]),
                "cluster_count": int(orb_best["cluster_count"]),
                "singleton_cluster_ratio": float(orb_best["singleton_cluster_ratio"]),
            },
            {
                "route": f"aliked_lightglue[{local_view}]",
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
    comparison_df["ari_delta_vs_orb"] = np.round(comparison_df["ari"] - float(orb_best["ari"]), 6)
    comparison_df["pairwise_f1_delta_vs_orb"] = np.round(comparison_df["pairwise_f1"] - float(orb_best["pairwise_f1"]), 6)
    comparison_df.to_csv(tables_dir / "comparison_vs_orb_v1.csv", index=False)

    test_pred_df = _build_test_predictions(
        dataset_df=test_df,
        score_matrix=best_test_score,
        threshold=float(best_meta["threshold"]),
        embedding_dim=int(test_embeddings.shape[1]),
        local_weight=float(best_meta["local_weight"]),
        route_name=route_name,
        local_matcher=f"aliked_lightglue[{local_view}]",
    )
    test_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)
    merged_pred_df = pd.concat(
        [base_pred_df[base_pred_df["dataset"] != SALAMANDER_DATASET].copy(), test_pred_df],
        ignore_index=True,
    )
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=repo_root / "sample_submission.csv",
        output_path=output_dir / "submission.csv",
    )

    cluster_counts = test_pred_df["pred_cluster_id"].value_counts()
    test_cluster_df = pd.DataFrame(
        [
            {
                "local_matcher": f"aliked_lightglue[{local_view}]",
                "threshold": float(best_meta["threshold"]),
                "local_weight": float(best_meta["local_weight"]),
                "clusters": int(cluster_counts.size),
                "singleton_clusters": int((cluster_counts == 1).sum()),
                "singleton_ratio": round(float((cluster_counts == 1).mean()) if len(cluster_counts) else 0.0, 6),
                "clusters_delta_vs_orb": int(cluster_counts.size) - int(orb_salamander_cluster["clusters"]),
            }
        ]
    )
    test_cluster_df.to_csv(tables_dir / "test_cluster_summary_v1.csv", index=False)

    config = {
        "route_dir": str(route_dir),
        "route_name": route_name,
        "local_view": local_view,
        "masked_manifest_root": str(masked_manifest_root),
        "val_local_view_resolved_ratio": round(float(val_view_artifacts.resolved_ratio), 6),
        "test_local_view_resolved_ratio": round(float(test_view_artifacts.resolved_ratio), 6),
        "device": str(args.device),
        "top_k": int(args.top_k),
        "local_weights": local_weights,
        "thresholds": thresholds,
        "max_num_keypoints": int(args.max_num_keypoints),
        "lightglue_init_threshold": float(args.lightglue_init_threshold),
        "matcher_batch_size": int(args.matcher_batch_size),
        "ransac_threshold": float(args.ransac_threshold),
        "min_inliers": int(args.min_inliers),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Salamander ALIKED + LightGlue Probe",
        "",
        f"- Anchor route dir: `{route_dir}`",
        "- Fixed global branch: `ft_miew_arcface_masked_supcon_v1 last + frozen fusion`",
        "- Single-factor change: `replace ORB local matcher with ALIKED + LightGlue pretrained`",
        f"- Local view: `{local_view}`",
        f"- Route name: `{route_name}`",
        f"- Local-view manifest root: `{masked_manifest_root}`",
        f"- Local-view resolved ratio: val `{val_view_artifacts.resolved_ratio:.4f}` / test `{test_view_artifacts.resolved_ratio:.4f}`",
        f"- Device: `{args.device}`",
        f"- Top-K candidate neighbors: `{int(args.top_k)}`",
        f"- Max ALIKED keypoints: `{int(args.max_num_keypoints)}`",
        f"- LightGlue init threshold: `{float(args.lightglue_init_threshold)}`",
        "",
        "## Local View Summary",
        "",
    ]
    if not val_view_artifacts.roi_summary_df.empty:
        lines.extend(
            [
                "- Validation ROI summary",
                "",
                dataframe_to_markdown_table(val_view_artifacts.roi_summary_df),
                "",
            ]
        )
    if not test_view_artifacts.roi_summary_df.empty:
        lines.extend(
            [
                "- Test ROI summary",
                "",
                dataframe_to_markdown_table(test_view_artifacts.roi_summary_df),
                "",
            ]
        )
    if not val_view_artifacts.band_summary_df.empty:
        lines.extend(
            [
                "- Validation yellow-band summary",
                "",
                dataframe_to_markdown_table(val_view_artifacts.band_summary_df),
                "",
            ]
        )
    if not test_view_artifacts.band_summary_df.empty:
        lines.extend(
            [
                "- Test yellow-band summary",
                "",
                dataframe_to_markdown_table(test_view_artifacts.band_summary_df),
                "",
            ]
        )
    lines.extend(
        [
        "## Best Local Validation Config",
        "",
        dataframe_to_markdown_table(pd.DataFrame([best_meta])[["local_weight", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]),
        "",
        "## Comparison Vs ORB Anchor",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
        "## Local Weight Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(test_cluster_df),
        "",
    ]
    )
    best_ari_delta = float(comparison_df.iloc[1]["ari_delta_vs_orb"])
    if best_ari_delta > 0:
        lines.extend(
            [
                "## Decision Hint",
                "",
                f"- Offline `ARI` beats current ORB anchor by `+{best_ari_delta:.6f}`. This route is eligible for official consideration if the cluster structure is also acceptable.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Decision Hint",
                "",
                f"- Offline `ARI` does not beat current ORB anchor on this split (`{best_ari_delta:.6f}` vs ORB). Treat this as an exploratory official only if the local-view change is the main thing you want to test.",
                "",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[salamander_aliked_lightglue_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_aliked_lightglue_probe] comparison: {tables_dir / 'comparison_vs_orb_v1.csv'}")
    print(f"[salamander_aliked_lightglue_probe] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
