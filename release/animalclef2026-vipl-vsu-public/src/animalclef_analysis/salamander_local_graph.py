from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import build_submission, dataframe_to_markdown_table, summarize_cluster_metrics
from .orb_rerank_baseline import build_topk_pair_index, cosine_score_matrix
from .salamander_pairwise_features import append_metadata_pair_features, append_route_graph_pair_features
from .salamander_yellow_orb_local import (
    YELLOW_FOCUS_PATH_COLUMN,
    build_patch_pair_features,
    build_yellow_focus_manifest,
    compile_yellow_orb_local_decisions,
    merge_yellow_orb_local_pair_features,
    summarize_patch_pair_features,
    summarize_yellow_focus_manifest,
    summarize_yellow_orb_local_decisions,
)
from .sam_orb_veto import build_masked_aligned_roi_manifest, build_view_local_match_table, summarize_roi_manifest


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_BASE_PREDICTIONS = DEFAULT_ROUTE_DIR / "tables" / "test_predictions_v1.csv"
DEFAULT_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/submissions/kaggle_variant_salamander_local_graph_v1")
DEFAULT_ROUTE_NAME = "salamander_local_graph_v1"
DEFAULT_TOP_K = 10
DEFAULT_STRONG_THRESHOLDS = [0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]
DEFAULT_WEAK_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.04]
DEFAULT_WEAK_ATTACH_MIN_SUPPORT = 2
DEFAULT_WEAK_MIN_SHARED_NEIGHBORS = 2
DEFAULT_ORB_FEATURES = 1024
DEFAULT_ORB_MAX_SIDE = 512
DEFAULT_FAST_THRESHOLD = 7
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_RATIO_TEST = 0.75
DEFAULT_RANSAC_THRESHOLD = 5.0
DEFAULT_MIN_INLIERS = 8
DEFAULT_LOCAL_MATCHER = "orb"
DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS = 512
DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE = 0.20
DEFAULT_SOFT_VETO_SCORE_SCALE = 0.70
DEFAULT_HARD_VETO_SCORE_CAP = 0.02


def _load_cached_table(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    for column in ["image_id", "neighbor_image_id", "dataset", "identity", YELLOW_FOCUS_PATH_COLUMN]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str)
    return df


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


def _subset_dataset(
    metadata_df: pd.DataFrame,
    embeddings: np.ndarray,
    dataset: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    dataset_mask = metadata_df["dataset"].astype(str).eq(str(dataset)).to_numpy()
    return (
        metadata_df.loc[dataset_mask].reset_index(drop=True).copy(),
        np.asarray(embeddings, dtype=np.float32)[dataset_mask],
    )


def build_candidate_pair_df(
    *,
    metadata_df: pd.DataFrame,
    score_matrix: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    pair_index = build_topk_pair_index(score_matrix=np.asarray(score_matrix, dtype=np.float32), top_k=int(top_k), query_indices=None)
    rows: list[dict[str, Any]] = []
    has_identity = "identity" in metadata_df.columns and metadata_df["identity"].fillna("").astype(str).ne("").any()
    for left_index, right_index, global_score in pair_index:
        left_row = metadata_df.iloc[int(left_index)]
        right_row = metadata_df.iloc[int(right_index)]
        left_identity = str(left_row.get("identity", "")) if has_identity else ""
        right_identity = str(right_row.get("identity", "")) if has_identity else ""
        rows.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "dataset": str(left_row["dataset"]),
                "route_global_score": float(global_score),
                "same_identity": int(left_identity == right_identity) if has_identity else -1,
            }
        )
    return pd.DataFrame(rows).sort_values(["left_index", "right_index"]).reset_index(drop=True)


def summarize_candidate_recall(pair_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
    if pair_df.empty or "identity" not in metadata_df.columns:
        return pd.DataFrame([{"queries": 0, "candidate_recall_at_topk": 0.0, "avg_candidates": 0.0}])
    neighbors_by_index: dict[int, set[int]] = {}
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        neighbors_by_index.setdefault(left_index, set()).add(right_index)
        neighbors_by_index.setdefault(right_index, set()).add(left_index)

    hits: list[bool] = []
    candidate_counts: list[int] = []
    for index, row in metadata_df.reset_index(drop=True).iterrows():
        identity = str(row.get("identity", ""))
        if not identity:
            continue
        same_identity_indices = set(metadata_df.index[metadata_df["identity"].astype(str).eq(identity)].tolist())
        same_identity_indices.discard(int(index))
        if not same_identity_indices:
            continue
        candidate_neighbors = neighbors_by_index.get(int(index), set())
        candidate_counts.append(int(len(candidate_neighbors)))
        hits.append(bool(candidate_neighbors & same_identity_indices))
    return pd.DataFrame(
        [
            {
                "queries": int(len(hits)),
                "candidate_recall_at_topk": round(float(np.mean(hits)) if hits else 0.0, 6),
                "avg_candidates": round(float(np.mean(candidate_counts)) if candidate_counts else 0.0, 4),
            }
        ]
    )


def _series_or_default(df: pd.DataFrame, column: str, default: float | int | bool) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def build_local_only_score_table(
    decision_df: pd.DataFrame,
    *,
    hard_veto_score_cap: float,
    soft_veto_score_scale: float,
) -> pd.DataFrame:
    result = decision_df.copy().reset_index(drop=True)
    gray_corr = np.clip(pd.to_numeric(_series_or_default(result, "yellow_patch_gray_corr_v1", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    gray_absdiff = np.clip(pd.to_numeric(_series_or_default(result, "yellow_patch_gray_absdiff_v1", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    mask_dice = np.clip(pd.to_numeric(_series_or_default(result, "yellow_patch_mask_dice_v1", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    profile_corr = np.clip(pd.to_numeric(_series_or_default(result, "yellow_patch_profile_corr_v1", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    orb_score = np.clip(pd.to_numeric(_series_or_default(result, "yellow_roi_local_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    pair_support = np.clip(pd.to_numeric(_series_or_default(result, "yellow_pair_support_v1", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32), 0.0, 1.0)
    gray_support = np.clip(gray_corr * (1.0 - gray_absdiff), 0.0, 1.0)
    patch_support = np.maximum.reduce([gray_support, mask_dice, profile_corr]).astype(np.float32, copy=False)
    local_support = np.maximum(orb_score, pair_support).astype(np.float32, copy=False)
    final_score = local_support.copy()

    result["yellow_patch_support_score_v1"] = np.round(patch_support, 6)
    result["yellow_pair_support_score_v1"] = np.round(pair_support, 6)
    result["yellow_local_support_score_v1"] = np.round(local_support, 6)
    result["local_only_score_v1"] = np.round(final_score, 6)
    return result


def build_local_graph_edge_table(
    pair_df: pd.DataFrame,
    *,
    strong_threshold: float,
    weak_threshold: float,
    weak_min_shared_neighbors: int,
) -> pd.DataFrame:
    result = pair_df.copy().reset_index(drop=True)
    score = pd.to_numeric(result["local_only_score_v1"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    shared_neighbors = pd.to_numeric(result.get("route_shared_neighbor_count", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    mutual_topk = pd.to_numeric(result.get("route_mutual_topk", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    same_orientation = pd.to_numeric(result.get("same_orientation", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    yellow_support = pd.to_numeric(result.get("yellow_pair_support_v1", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    weak_gate = (
        (shared_neighbors >= int(weak_min_shared_neighbors))
        | (mutual_topk == 1)
        | (same_orientation == 1)
        | (yellow_support == 1)
    )
    strong_edge = score >= float(strong_threshold)
    weak_edge = (score >= float(weak_threshold)) & (~strong_edge) & weak_gate
    accepted_edge = strong_edge | weak_edge
    result["weak_gate_pass_v1"] = weak_gate.astype(bool)
    result["strong_edge_v1"] = strong_edge.astype(bool)
    result["weak_edge_v1"] = weak_edge.astype(bool)
    result["accepted_edge_v1"] = accepted_edge.astype(bool)
    result["edge_kind_v1"] = np.where(strong_edge, "strong", np.where(weak_edge, "weak", "reject"))
    return result


def _connected_components(sample_count: int, edge_df: pd.DataFrame, *, edge_mask_column: str) -> np.ndarray:
    adjacency: list[list[int]] = [[] for _ in range(int(sample_count))]
    for row in edge_df[edge_df[edge_mask_column].astype(bool)].itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        adjacency[left_index].append(right_index)
        adjacency[right_index].append(left_index)
    labels = -np.ones(int(sample_count), dtype=np.int32)
    next_label = 0
    for start in range(int(sample_count)):
        if labels[start] != -1:
            continue
        stack = [start]
        labels[start] = next_label
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if labels[neighbor] == -1:
                    labels[neighbor] = next_label
                    stack.append(neighbor)
        next_label += 1
    _, normalized = np.unique(labels, return_inverse=True)
    return normalized.astype(np.int32, copy=False)


def cluster_with_singleton_attach(
    *,
    metadata_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    weak_attach_min_support: int,
) -> np.ndarray:
    sample_count = int(len(metadata_df))
    labels = _connected_components(sample_count, edge_df, edge_mask_column="strong_edge_v1")
    component_sizes = pd.Series(labels).value_counts().to_dict()
    multi_node_labels = {int(label) for label, count in component_sizes.items() if int(count) >= 2}
    attach_candidates = edge_df[edge_df["weak_edge_v1"].astype(bool)].copy().reset_index(drop=True)
    if attach_candidates.empty or not multi_node_labels:
        _, normalized = np.unique(labels, return_inverse=True)
        return normalized.astype(np.int32, copy=False)

    for node_index in range(sample_count):
        current_label = int(labels[node_index])
        if int(component_sizes.get(current_label, 0)) != 1:
            continue
        related = attach_candidates[
            (attach_candidates["left_index"].astype(int).eq(int(node_index)))
            | (attach_candidates["right_index"].astype(int).eq(int(node_index)))
        ].copy()
        if related.empty:
            continue

        target_rows: list[dict[str, Any]] = []
        for row in related.itertuples(index=False):
            other_index = int(row.right_index) if int(row.left_index) == int(node_index) else int(row.left_index)
            target_label = int(labels[other_index])
            if target_label == current_label or target_label not in multi_node_labels:
                continue
            target_rows.append(
                {
                    "target_label": target_label,
                    "other_index": other_index,
                    "score": float(row.local_only_score_v1),
                }
            )
        if not target_rows:
            continue
        target_df = pd.DataFrame(target_rows)
        target_summary = (
            target_df.groupby("target_label")
            .agg(
                support_count=("other_index", lambda s: int(pd.Series(s).nunique())),
                best_score=("score", "max"),
                mean_score=("score", "mean"),
            )
            .reset_index()
            .sort_values(["support_count", "best_score", "mean_score", "target_label"], ascending=[False, False, False, True])
        )
        best = target_summary.iloc[0]
        if int(best["support_count"]) < int(weak_attach_min_support):
            continue
        labels[node_index] = int(best["target_label"])
        component_sizes[int(best["target_label"])] = int(component_sizes.get(int(best["target_label"]), 0)) + 1
        component_sizes[current_label] = 0
        multi_node_labels.add(int(best["target_label"]))
    _, normalized = np.unique(labels, return_inverse=True)
    return normalized.astype(np.int32, copy=False)


def build_prediction_frame(
    metadata_df: pd.DataFrame,
    labels: np.ndarray,
    *,
    route_name: str,
    strong_threshold: float,
    weak_threshold: float,
    top_k: int,
) -> pd.DataFrame:
    result = metadata_df.copy().reset_index(drop=True)
    result["pred_cluster_id"] = np.asarray(labels, dtype=np.int32)
    result["cluster_label"] = [f"cluster_{SALAMANDER_DATASET}_{int(label)}" for label in result["pred_cluster_id"]]
    result["route_name"] = str(route_name)
    result["chosen_threshold"] = float(strong_threshold)
    result["weak_threshold_v1"] = float(weak_threshold)
    result["candidate_top_k_v1"] = int(top_k)
    return result


def run_local_graph_threshold_sweep(
    *,
    val_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    strong_thresholds: list[float],
    weak_thresholds: list[float],
    weak_min_shared_neighbors: int,
    weak_attach_min_support: int,
) -> tuple[pd.DataFrame, dict[tuple[float, float], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    label_map: dict[tuple[float, float], np.ndarray] = {}
    true_labels = val_df["identity"].fillna("").astype(str).to_numpy()
    for strong_threshold in [float(value) for value in strong_thresholds]:
        for weak_threshold in [float(value) for value in weak_thresholds]:
            if float(weak_threshold) >= float(strong_threshold):
                continue
            edge_df = build_local_graph_edge_table(
                pair_df=pair_df,
                strong_threshold=float(strong_threshold),
                weak_threshold=float(weak_threshold),
                weak_min_shared_neighbors=int(weak_min_shared_neighbors),
            )
            labels = cluster_with_singleton_attach(
                metadata_df=val_df,
                edge_df=edge_df,
                weak_attach_min_support=int(weak_attach_min_support),
            )
            metrics = summarize_cluster_metrics(true_labels=true_labels, pred_labels=labels)
            counts = pd.Series(labels).value_counts()
            rows.append(
                {
                    "dataset": SALAMANDER_DATASET,
                    "strong_threshold": float(strong_threshold),
                    "weak_threshold": float(weak_threshold),
                    "accepted_edges": int(edge_df["accepted_edge_v1"].astype(bool).sum()),
                    "strong_edges": int(edge_df["strong_edge_v1"].astype(bool).sum()),
                    "weak_edges": int(edge_df["weak_edge_v1"].astype(bool).sum()),
                    "cluster_count": int(counts.size),
                    "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                    **metrics,
                }
            )
            label_map[(float(strong_threshold), float(weak_threshold))] = labels
    sweep_df = pd.DataFrame(rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "strong_threshold", "weak_threshold"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    return sweep_df, label_map


def _pick_best_threshold_row(sweep_df: pd.DataFrame) -> pd.Series:
    if sweep_df.empty:
        raise ValueError("Threshold sweep is empty.")
    return sweep_df.sort_values(
        ["ari", "pairwise_f1", "nmi", "strong_threshold", "weak_threshold"],
        ascending=[False, False, False, True, True],
    ).iloc[0]


def _merge_override_predictions(base_pred_df: pd.DataFrame, override_pred_df: pd.DataFrame) -> pd.DataFrame:
    base_pred_df = base_pred_df.copy()
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    if "dataset" in base_pred_df.columns:
        base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    override_pred_df = override_pred_df.copy()
    override_pred_df["image_id"] = override_pred_df["image_id"].astype(str)
    kept_df = (
        base_pred_df[base_pred_df["dataset"].astype(str) != SALAMANDER_DATASET].copy()
        if "dataset" in base_pred_df.columns
        else base_pred_df[~base_pred_df["image_id"].isin(override_pred_df["image_id"])].copy()
    )
    return pd.concat([kept_df, override_pred_df], ignore_index=True)


def _build_cluster_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, frame in pred_df.groupby("dataset"):
        counts = frame["pred_cluster_id"].value_counts()
        rows.append(
            {
                "dataset": str(dataset),
                "samples": int(len(frame)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(frame["route_name"].iloc[0]) if "route_name" in frame.columns else "",
                "chosen_threshold": float(frame["chosen_threshold"].iloc[0]) if "chosen_threshold" in frame.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _write_summary(
    *,
    output_path: Path,
    config: dict[str, Any],
    candidate_summary_df: pd.DataFrame,
    roi_summary_df: pd.DataFrame,
    focus_summary_df: pd.DataFrame,
    patch_summary_df: pd.DataFrame,
    decision_summary_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    best_row: pd.Series,
    test_shape_df: pd.DataFrame,
) -> None:
    lines = [
        "# Salamander Local Graph Submission",
        "",
        "## Architecture",
        "",
        "- Overall system: `two-stage local-graph clustering pipeline`.",
        "- Global flow: `image -> Salamander global embedding -> top-K candidate recall -> yellow ROI local scoring -> thresholded graph -> singleton attach -> submission cluster label`.",
        f"- Dataset branch: `{SALAMANDER_DATASET}` only; no XGBoost fusion score is used in the final pair decision.",
        f"- Candidate generator: global cosine top-K with `K={int(config['top_k'])}`.",
        f"- Final pair score: `max(yellow_roi_local_score, yellow_pair_support_v1)`; continuous patch statistics stay diagnostic-only.",
        f"- Core graph threshold: `{float(best_row['strong_threshold'])}`.",
        f"- Weak attach threshold: `{float(best_row['weak_threshold'])}`.",
        f"- Weak attach minimum support: `{int(config['weak_attach_min_support'])}`.",
        "",
        "## Candidate Recall",
        "",
        dataframe_to_markdown_table(candidate_summary_df),
        "",
        "## ROI Summary",
        "",
        dataframe_to_markdown_table(roi_summary_df),
        "",
        "## Yellow Focus Summary",
        "",
        dataframe_to_markdown_table(focus_summary_df),
        "",
        "## Patch Summary",
        "",
        dataframe_to_markdown_table(patch_summary_df),
        "",
        "## Decision Summary",
        "",
        dataframe_to_markdown_table(decision_summary_df),
        "",
        "## Validation Sweep Top Rows",
        "",
        dataframe_to_markdown_table(sweep_df.head(10)),
        "",
        "## Test Cluster Shape",
        "",
        dataframe_to_markdown_table(test_shape_df),
        "",
        "## Reading Note",
        "",
        "- `candidate_recall_at_topk` tells you whether the global branch is doing its only job: not missing the true mate.",
        "- `ari` is reported after graph clustering, so it reflects the full `召回 -> 局部打分 -> 阈值/拒识 -> 聚类` chain rather than retrieval alone.",
        "- If future experiments improve `candidate_recall_at_topk` but hurt `ari`, the local decision or graph thresholds should be retuned before changing the backbone again.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_salamander_local_graph_submission(
    *,
    repo_root: Path,
    route_dir: Path,
    base_predictions_path: Path,
    sample_submission_path: Path,
    manifest_root: Path,
    output_dir: Path,
    route_name: str = DEFAULT_ROUTE_NAME,
    top_k: int = DEFAULT_TOP_K,
    strong_thresholds: list[float] | None = None,
    weak_thresholds: list[float] | None = None,
    weak_attach_min_support: int = DEFAULT_WEAK_ATTACH_MIN_SUPPORT,
    weak_min_shared_neighbors: int = DEFAULT_WEAK_MIN_SHARED_NEIGHBORS,
    orb_features: int = DEFAULT_ORB_FEATURES,
    orb_max_side: int = DEFAULT_ORB_MAX_SIDE,
    fast_threshold: int = DEFAULT_FAST_THRESHOLD,
    clahe_clip_limit: float = DEFAULT_CLAHE_CLIP_LIMIT,
    ratio_test: float = DEFAULT_RATIO_TEST,
    ransac_threshold: float = DEFAULT_RANSAC_THRESHOLD,
    min_inliers: int = DEFAULT_MIN_INLIERS,
    local_matcher: str = DEFAULT_LOCAL_MATCHER,
    alignment_min_foreground_pixels: int = DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS,
    alignment_min_axis_confidence: float = DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE,
    soft_veto_score_scale: float = DEFAULT_SOFT_VETO_SCORE_SCALE,
    hard_veto_score_cap: float = DEFAULT_HARD_VETO_SCORE_CAP,
) -> dict[str, Path]:
    if strong_thresholds is None:
        strong_thresholds = DEFAULT_STRONG_THRESHOLDS
    if weak_thresholds is None:
        weak_thresholds = DEFAULT_WEAK_THRESHOLDS

    repo_root = repo_root.resolve()
    route_dir = route_dir.resolve()
    base_predictions_path = base_predictions_path.resolve()
    sample_submission_path = sample_submission_path.resolve()
    manifest_root = manifest_root.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    full_val_df, full_val_embeddings, full_test_df, full_test_embeddings = _load_route_bundle(route_dir=route_dir)
    val_df, val_embeddings = _subset_dataset(full_val_df, full_val_embeddings, dataset=SALAMANDER_DATASET)
    test_df, test_embeddings = _subset_dataset(full_test_df, full_test_embeddings, dataset=SALAMANDER_DATASET)
    val_score = cosine_score_matrix(val_embeddings)
    test_score = cosine_score_matrix(test_embeddings)

    val_pair_df = build_candidate_pair_df(metadata_df=val_df, score_matrix=val_score, top_k=int(top_k))
    test_pair_df = build_candidate_pair_df(metadata_df=test_df, score_matrix=test_score, top_k=int(top_k))
    val_pair_df = append_metadata_pair_features(pair_df=val_pair_df, metadata_df=val_df)
    val_pair_df = append_route_graph_pair_features(pair_df=val_pair_df, route_score=val_score, top_k=int(top_k))
    test_pair_df = append_metadata_pair_features(pair_df=test_pair_df, metadata_df=test_df)
    test_pair_df = append_route_graph_pair_features(pair_df=test_pair_df, route_score=test_score, top_k=int(top_k))
    val_pair_df.to_csv(tables_dir / "val_candidate_pairs_v1.csv", index=False)
    test_pair_df.to_csv(tables_dir / "test_candidate_pairs_v1.csv", index=False)

    candidate_summary_df = summarize_candidate_recall(pair_df=val_pair_df, metadata_df=val_df)
    candidate_summary_df.to_csv(tables_dir / "val_candidate_summary_v1.csv", index=False)

    enriched_df = pd.read_csv(manifest_root / "tables" / "metadata_enriched_v1.csv")
    enriched_df["image_id"] = enriched_df["image_id"].astype(str)
    enriched_df["dataset"] = enriched_df["dataset"].astype(str)
    enriched_df = enriched_df[enriched_df["dataset"].eq(SALAMANDER_DATASET)].reset_index(drop=True)

    required_ids = set(val_pair_df["image_id"].tolist()) | set(val_pair_df["neighbor_image_id"].tolist()) | set(test_pair_df["image_id"].tolist()) | set(test_pair_df["neighbor_image_id"].tolist())
    roi_reference_df = pd.concat(
        [
            val_df[val_df["image_id"].isin(required_ids)].copy(),
            test_df[test_df["image_id"].isin(required_ids)].copy(),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["image_id", "dataset"])
    roi_manifest_path = tables_dir / "image_roi_manifest_v1.csv"
    roi_summary_path = tables_dir / "image_roi_summary_v1.csv"
    roi_manifest_df = _load_cached_table(roi_manifest_path)
    if roi_manifest_df is None:
        roi_manifest_df = build_masked_aligned_roi_manifest(
            reference_df=roi_reference_df,
            enriched_df=enriched_df,
            repo_root=repo_root,
            output_dir=output_dir,
            alignment_min_foreground_pixels=int(alignment_min_foreground_pixels),
            alignment_min_axis_confidence=float(alignment_min_axis_confidence),
        )
        roi_manifest_df.to_csv(roi_manifest_path, index=False)
    roi_summary_df = _load_cached_table(roi_summary_path)
    if roi_summary_df is None:
        roi_summary_df = summarize_roi_manifest(roi_manifest_df=roi_manifest_df)
        roi_summary_df.to_csv(roi_summary_path, index=False)

    focus_manifest_path = tables_dir / "yellow_focus_manifest_v1.csv"
    focus_summary_path = tables_dir / "yellow_focus_summary_v1.csv"
    focus_df = _load_cached_table(focus_manifest_path)
    if focus_df is None:
        focus_df = build_yellow_focus_manifest(
            roi_manifest_df=roi_manifest_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        focus_df.to_csv(focus_manifest_path, index=False)
    focus_summary_df = _load_cached_table(focus_summary_path)
    if focus_summary_df is None:
        focus_summary_df = summarize_yellow_focus_manifest(focus_df=focus_df)
        focus_summary_df.to_csv(focus_summary_path, index=False)

    val_focus_df = val_df.merge(
        focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
    )
    test_focus_df = test_df.merge(
        focus_df[["image_id", "dataset", YELLOW_FOCUS_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
    )

    val_yellow_roi_local_path = tables_dir / "val_yellow_roi_local_scores_v1.csv"
    test_yellow_roi_local_path = tables_dir / "test_yellow_roi_local_scores_v1.csv"
    val_yellow_roi_local_df = _load_cached_table(val_yellow_roi_local_path)
    if val_yellow_roi_local_df is None:
        val_yellow_roi_local_df = build_view_local_match_table(
            reference_df=val_focus_df,
            pair_df=val_pair_df,
            repo_root=repo_root,
            path_column=YELLOW_FOCUS_PATH_COLUMN,
            nfeatures=int(orb_features),
            max_side=int(orb_max_side),
            fast_threshold=int(fast_threshold),
            clahe_clip_limit=float(clahe_clip_limit),
            ratio_test=float(ratio_test),
            ransac_threshold=float(ransac_threshold),
            min_inliers=int(min_inliers),
            local_matcher=str(local_matcher),
            prefix="yellow_roi",
        )
        val_yellow_roi_local_df.to_csv(val_yellow_roi_local_path, index=False)
    test_yellow_roi_local_df = _load_cached_table(test_yellow_roi_local_path)
    if test_yellow_roi_local_df is None:
        test_yellow_roi_local_df = build_view_local_match_table(
            reference_df=test_focus_df,
            pair_df=test_pair_df,
            repo_root=repo_root,
            path_column=YELLOW_FOCUS_PATH_COLUMN,
            nfeatures=int(orb_features),
            max_side=int(orb_max_side),
            fast_threshold=int(fast_threshold),
            clahe_clip_limit=float(clahe_clip_limit),
            ratio_test=float(ratio_test),
            ransac_threshold=float(ransac_threshold),
            min_inliers=int(min_inliers),
            local_matcher=str(local_matcher),
            prefix="yellow_roi",
        )
        test_yellow_roi_local_df.to_csv(test_yellow_roi_local_path, index=False)

    val_patch_pair_path = tables_dir / "val_patch_pair_features_v1.csv"
    test_patch_pair_path = tables_dir / "test_patch_pair_features_v1.csv"
    val_patch_pair_df = _load_cached_table(val_patch_pair_path)
    if val_patch_pair_df is None:
        val_patch_pair_df = build_patch_pair_features(pair_df=val_pair_df, focus_df=focus_df, repo_root=repo_root)
        val_patch_pair_df.to_csv(val_patch_pair_path, index=False)
    test_patch_pair_df = _load_cached_table(test_patch_pair_path)
    if test_patch_pair_df is None:
        test_patch_pair_df = build_patch_pair_features(pair_df=test_pair_df, focus_df=focus_df, repo_root=repo_root)
        test_patch_pair_df.to_csv(test_patch_pair_path, index=False)

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
    val_scored_pair_df = build_local_only_score_table(
        val_decision_df,
        hard_veto_score_cap=float(hard_veto_score_cap),
        soft_veto_score_scale=float(soft_veto_score_scale),
    )
    test_scored_pair_df = build_local_only_score_table(
        test_decision_df,
        hard_veto_score_cap=float(hard_veto_score_cap),
        soft_veto_score_scale=float(soft_veto_score_scale),
    )
    val_scored_pair_df.to_csv(tables_dir / "val_pair_local_scores_v1.csv", index=False)
    test_scored_pair_df.to_csv(tables_dir / "test_pair_local_scores_v1.csv", index=False)

    sweep_df, _label_map = run_local_graph_threshold_sweep(
        val_df=val_df,
        pair_df=val_scored_pair_df,
        strong_thresholds=[float(value) for value in strong_thresholds],
        weak_thresholds=[float(value) for value in weak_thresholds],
        weak_min_shared_neighbors=int(weak_min_shared_neighbors),
        weak_attach_min_support=int(weak_attach_min_support),
    )
    sweep_df.to_csv(tables_dir / "val_local_graph_threshold_sweep_v1.csv", index=False)
    best_row = _pick_best_threshold_row(sweep_df)
    best_df = pd.DataFrame([best_row])
    best_df.to_csv(tables_dir / "val_best_local_graph_row_v1.csv", index=False)

    best_strong_threshold = float(best_row["strong_threshold"])
    best_weak_threshold = float(best_row["weak_threshold"])
    val_best_edge_df = build_local_graph_edge_table(
        pair_df=val_scored_pair_df,
        strong_threshold=best_strong_threshold,
        weak_threshold=best_weak_threshold,
        weak_min_shared_neighbors=int(weak_min_shared_neighbors),
    )
    val_best_edge_df.to_csv(tables_dir / "val_graph_edges_v1.csv", index=False)
    val_labels = cluster_with_singleton_attach(
        metadata_df=val_df,
        edge_df=val_best_edge_df,
        weak_attach_min_support=int(weak_attach_min_support),
    )
    val_pred_df = build_prediction_frame(
        metadata_df=val_df,
        labels=val_labels,
        route_name=str(route_name),
        strong_threshold=best_strong_threshold,
        weak_threshold=best_weak_threshold,
        top_k=int(top_k),
    )
    val_pred_df.to_csv(tables_dir / "val_predictions_v1.csv", index=False)

    test_edge_df = build_local_graph_edge_table(
        pair_df=test_scored_pair_df,
        strong_threshold=best_strong_threshold,
        weak_threshold=best_weak_threshold,
        weak_min_shared_neighbors=int(weak_min_shared_neighbors),
    )
    test_edge_df.to_csv(tables_dir / "test_graph_edges_v1.csv", index=False)
    test_labels = cluster_with_singleton_attach(
        metadata_df=test_df,
        edge_df=test_edge_df,
        weak_attach_min_support=int(weak_attach_min_support),
    )
    salamander_test_pred_df = build_prediction_frame(
        metadata_df=test_df,
        labels=test_labels,
        route_name=str(route_name),
        strong_threshold=best_strong_threshold,
        weak_threshold=best_weak_threshold,
        top_k=int(top_k),
    )
    salamander_test_pred_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)

    base_pred_df = pd.read_csv(base_predictions_path)
    merged_pred_df = _merge_override_predictions(base_pred_df=base_pred_df, override_pred_df=salamander_test_pred_df)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=sample_submission_path,
        output_path=output_dir / "submission.csv",
    )

    patch_summary_df = summarize_patch_pair_features(pair_df=val_patch_pair_df)
    decision_summary_df = summarize_yellow_orb_local_decisions(decision_df=val_scored_pair_df)
    patch_summary_df.to_csv(tables_dir / "val_patch_summary_v1.csv", index=False)
    decision_summary_df.to_csv(tables_dir / "val_decision_summary_v1.csv", index=False)

    test_shape_df = _build_cluster_summary(salamander_test_pred_df)
    test_shape_df.to_csv(tables_dir / "salamander_test_cluster_summary_v1.csv", index=False)
    full_cluster_summary_df = _build_cluster_summary(merged_pred_df)
    full_cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    config = {
        "route_dir": str(route_dir),
        "base_predictions_path": str(base_predictions_path),
        "sample_submission_path": str(sample_submission_path),
        "manifest_root": str(manifest_root),
        "route_name": str(route_name),
        "top_k": int(top_k),
        "strong_thresholds": [float(value) for value in strong_thresholds],
        "weak_thresholds": [float(value) for value in weak_thresholds],
        "weak_attach_min_support": int(weak_attach_min_support),
        "weak_min_shared_neighbors": int(weak_min_shared_neighbors),
        "best_strong_threshold": best_strong_threshold,
        "best_weak_threshold": best_weak_threshold,
        "orb_features": int(orb_features),
        "orb_max_side": int(orb_max_side),
        "fast_threshold": int(fast_threshold),
        "clahe_clip_limit": float(clahe_clip_limit),
        "ratio_test": float(ratio_test),
        "ransac_threshold": float(ransac_threshold),
        "min_inliers": int(min_inliers),
        "local_matcher": str(local_matcher),
        "soft_veto_score_scale": float(soft_veto_score_scale),
        "hard_veto_score_cap": float(hard_veto_score_cap),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary(
        output_path=reports_dir / "summary.md",
        config=config,
        candidate_summary_df=candidate_summary_df,
        roi_summary_df=roi_summary_df,
        focus_summary_df=focus_summary_df,
        patch_summary_df=patch_summary_df,
        decision_summary_df=decision_summary_df,
        sweep_df=sweep_df,
        best_row=best_row,
        test_shape_df=test_shape_df,
    )

    return {
        "submission_path": output_dir / "submission.csv",
        "test_predictions_path": tables_dir / "test_predictions_v1.csv",
        "salamander_predictions_path": tables_dir / "salamander_test_predictions_v1.csv",
        "summary_path": reports_dir / "summary.md",
        "threshold_sweep_path": tables_dir / "val_local_graph_threshold_sweep_v1.csv",
    }
