from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import (
    PATH_COLUMN,
    build_submission,
    cluster_from_linkage,
    ensure_metadata_alignment,
    load_cached_embedding_bundle,
    summarize_cluster_metrics,
)
from .transductive_seed_refinement import (
    cluster_labels_from_score_matrix,
    cosine_similarity_matrix,
    pick_best_threshold_row,
    run_score_threshold_sweep,
)

try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
except ModuleNotFoundError:
    adjusted_rand_score = None
    normalized_mutual_info_score = None

try:
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
except ModuleNotFoundError:
    linkage = None
    squareform = None


LYNX_DATASET = "LynxID2025"


@dataclass(frozen=True)
class RouteCandidate:
    route_name: str
    source_kind: str
    source_ref: str
    chosen_threshold: float | None
    val_prediction_df: pd.DataFrame
    test_prediction_df: pd.DataFrame
    val_sweep_df: pd.DataFrame
    extra: dict[str, Any]

    @property
    def val_labels(self) -> np.ndarray:
        return self.val_prediction_df["pred_cluster_id"].to_numpy(dtype=int)

    @property
    def test_labels(self) -> np.ndarray:
        return self.test_prediction_df["pred_cluster_id"].to_numpy(dtype=int)

    def summary_row(self) -> dict[str, Any]:
        best_row = self.val_sweep_df.iloc[0].to_dict()
        row = {
            "route_name": self.route_name,
            "source_kind": self.source_kind,
            "source_ref": self.source_ref,
            "chosen_threshold": self.chosen_threshold,
        }
        row.update(best_row)
        row.update(self.extra)
        return row


@dataclass(frozen=True)
class EnsembleProbeResult:
    route_sweep_df: pd.DataFrame
    route_candidates_df: pd.DataFrame
    selected_routes_df: pd.DataFrame
    route_agreement_df: pd.DataFrame
    ensemble_sweep_df: pd.DataFrame
    cluster_shape_df: pd.DataFrame
    best_route_df: pd.DataFrame
    best_val_prediction_df: pd.DataFrame
    best_test_prediction_df: pd.DataFrame
    best_val_coassociation_df: pd.DataFrame
    best_test_coassociation_df: pd.DataFrame
    best_route_names: list[str]
    best_route_count: int
    best_vote_threshold: float
    best_row: dict[str, Any]
    best_single_row: dict[str, Any]
    exported_paths: dict[str, str]


def _require_sklearn() -> None:
    if adjusted_rand_score is None or normalized_mutual_info_score is None:
        raise ModuleNotFoundError("Lynx route ensemble requires scikit-learn in the active environment.")


def _normalize_metadata(frame: pd.DataFrame, *, dataset: str = LYNX_DATASET) -> pd.DataFrame:
    normalized = frame.copy().reset_index(drop=True)
    normalized["image_id"] = normalized["image_id"].astype(str)
    if "dataset" not in normalized.columns:
        normalized["dataset"] = dataset
    normalized["dataset"] = normalized["dataset"].astype(str)
    normalized = normalized[normalized["dataset"] == dataset].copy().reset_index(drop=True)
    if "identity" in normalized.columns:
        normalized["identity"] = normalized["identity"].fillna("").astype(str)
    return normalized


def _metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ["image_id", "dataset", "identity", PATH_COLUMN] if column in frame.columns]


def _prediction_keep_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ["image_id", "dataset", "identity", PATH_COLUMN, "threshold", "pred_cluster_id"] if column in frame.columns]


def _build_prediction_frame(reference_df: pd.DataFrame, labels: np.ndarray, threshold: float | None) -> pd.DataFrame:
    frame = reference_df.loc[:, _metadata_columns(reference_df)].copy().reset_index(drop=True)
    frame["threshold"] = np.nan if threshold is None else float(threshold)
    frame["pred_cluster_id"] = labels.astype(int)
    return frame


def _cluster_shape_row(
    *,
    split_name: str,
    object_name: str,
    object_kind: str,
    labels: np.ndarray,
    route_count: int,
) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts().sort_values()
    singleton_count = int((counts == 1).sum()) if not counts.empty else 0
    values = counts.to_numpy(dtype=float) if not counts.empty else np.array([], dtype=float)
    return {
        "split": split_name,
        "object_name": object_name,
        "object_kind": object_kind,
        "route_count": int(route_count),
        "samples": int(len(labels)),
        "cluster_count": int(len(counts)),
        "singleton_clusters": singleton_count,
        "singleton_cluster_ratio": round(float(singleton_count / len(counts)), 6) if len(counts) else 0.0,
        "mean_cluster_size": round(float(values.mean()), 6) if len(values) else 0.0,
        "median_cluster_size": round(float(np.median(values)), 6) if len(values) else 0.0,
        "p90_cluster_size": round(float(np.quantile(values, 0.9)), 6) if len(values) else 0.0,
        "max_cluster_size": int(values.max()) if len(values) else 0,
    }


def _reorder_prediction_frame(
    *,
    reference_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    route_name: str,
) -> pd.DataFrame:
    candidate_df = _normalize_metadata(prediction_df)
    if "pred_cluster_id" not in candidate_df.columns:
        raise ValueError(f"Prediction table for {route_name} is missing pred_cluster_id.")
    join_columns = ["image_id", "dataset"]
    merged = reference_df[join_columns].merge(
        candidate_df.assign(_row_index=np.arange(len(candidate_df), dtype=np.int32)),
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    if merged["_row_index"].isna().any():
        missing_rows = merged.loc[merged["_row_index"].isna(), join_columns].head(5).to_dict(orient="records")
        raise ValueError(f"Missing image_id/dataset rows for {route_name}, examples: {missing_rows}")
    order = merged["_row_index"].astype(int).to_numpy()
    ordered = candidate_df.iloc[order].reset_index(drop=True).copy()
    for column in ["identity", PATH_COLUMN]:
        if column not in ordered.columns and column in reference_df.columns:
            ordered[column] = reference_df[column].tolist()
    ensure_metadata_alignment(
        reference_df=reference_df.loc[:, _metadata_columns(reference_df)].reset_index(drop=True),
        candidate_df=ordered.loc[:, _metadata_columns(ordered)].reset_index(drop=True),
        split_name="lynx_route_ensemble",
        reference_name="reference_metadata",
        candidate_name=route_name,
    )
    return ordered.loc[:, _prediction_keep_columns(ordered)].copy().reset_index(drop=True)


def _pick_threshold_row(sweep_df: pd.DataFrame) -> pd.Series:
    if "ari" in sweep_df.columns:
        return pick_best_threshold_row(sweep_df)
    order_columns = [column for column in ["cluster_count", "threshold"] if column in sweep_df.columns]
    ascending = [True, True][: len(order_columns)]
    return sweep_df.sort_values(order_columns, ascending=ascending).iloc[0]


def _threshold_values(frame: pd.DataFrame) -> list[float | None]:
    if "threshold" not in frame.columns:
        return [None]
    values = frame["threshold"].tolist()
    unique_values: list[float | None] = []
    for value in values:
        normalized = None if pd.isna(value) else float(value)
        if not any((current is None and normalized is None) or (current is not None and normalized is not None and abs(current - normalized) <= 1e-9) for current in unique_values):
            unique_values.append(normalized)
    return unique_values or [None]


def _metrics_from_labels(reference_df: pd.DataFrame, labels: np.ndarray) -> dict[str, Any]:
    counts = pd.Series(labels).value_counts()
    row: dict[str, Any] = {
        "dataset": str(reference_df["dataset"].iloc[0]),
        "samples": int(len(reference_df)),
        "cluster_count": int(len(counts)),
        "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
    }
    if "identity" in reference_df.columns and reference_df["identity"].fillna("").astype(str).ne("").any():
        row.update(
            summarize_cluster_metrics(
                true_labels=reference_df["identity"].astype(str).to_numpy(),
                pred_labels=labels.astype(int),
            )
        )
    return row


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    _, normalized = np.unique(labels.astype(int), return_inverse=True)
    return normalized.astype(int)


def _linkage_labels_from_score_matrix(score_matrix: np.ndarray, threshold: float, method: str) -> np.ndarray:
    if linkage is None or squareform is None:
        raise RuntimeError("scipy is required for linkage routers.")
    if method not in {"average", "complete"}:
        raise ValueError(f"Unsupported linkage router: {method}")
    if len(score_matrix) < 2:
        return np.zeros(len(score_matrix), dtype=int)
    distance = 1.0 - np.clip(score_matrix, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    linkage_matrix = linkage(squareform(distance, checks=False), method=method)
    return cluster_from_linkage(linkage_matrix, sample_count=len(score_matrix), threshold=float(threshold))


def _mutual_knn_labels_from_score_matrix(score_matrix: np.ndarray, threshold: float, k: int) -> np.ndarray:
    sample_count = int(score_matrix.shape[0])
    if sample_count == 0:
        return np.array([], dtype=int)
    if sample_count == 1:
        return np.zeros(1, dtype=int)
    effective_k = max(1, min(int(k), sample_count - 1))
    score = np.array(score_matrix, dtype=np.float32, copy=True)
    np.fill_diagonal(score, -np.inf)
    topk = np.argpartition(-score, kth=effective_k - 1, axis=1)[:, :effective_k]
    neighbor_mask = np.zeros((sample_count, sample_count), dtype=bool)
    rows = np.arange(sample_count)[:, None]
    neighbor_mask[rows, topk] = True
    edge_mask = neighbor_mask & neighbor_mask.T & (score_matrix >= float(threshold))
    parent = np.arange(sample_count, dtype=np.int32)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    left_indices, right_indices = np.triu_indices(sample_count, k=1)
    for left, right in zip(left_indices[edge_mask[left_indices, right_indices]].tolist(), right_indices[edge_mask[left_indices, right_indices]].tolist(), strict=True):
        union(int(left), int(right))
    labels = np.array([find(index) for index in range(sample_count)], dtype=int)
    return _normalize_labels(labels)


def _router_threshold_sweep(
    *,
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    thresholds: list[float],
    router: str,
    mutual_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_identity = "identity" in df.columns and df["identity"].fillna("").astype(str).ne("").any()
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for threshold in [float(value) for value in thresholds]:
        if router in {"average", "complete"}:
            pred_labels = _linkage_labels_from_score_matrix(score_matrix=score_matrix, threshold=threshold, method=router)
        elif router == "mutual_knn":
            pred_labels = _mutual_knn_labels_from_score_matrix(score_matrix=score_matrix, threshold=threshold, k=mutual_k)
        else:
            raise ValueError(f"Unsupported router: {router}")
        row = _metrics_from_labels(reference_df=df, labels=pred_labels)
        row["threshold"] = threshold
        if has_identity and true_labels is not None:
            row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels))
        prediction_frames.append(_build_prediction_frame(reference_df=df, labels=pred_labels, threshold=threshold))
        rows.append(row)
    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def _prediction_sweep_from_table(
    *,
    route_name: str,
    reference_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[float | None, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    frame_by_threshold: dict[float | None, pd.DataFrame] = {}
    for threshold in _threshold_values(prediction_df):
        if threshold is None:
            current_df = prediction_df.copy().reset_index(drop=True)
        else:
            current_df = prediction_df[np.isclose(prediction_df["threshold"].astype(float), threshold)].copy().reset_index(drop=True)
        ordered_df = _reorder_prediction_frame(
            reference_df=reference_df,
            prediction_df=current_df,
            route_name=f"{route_name}@{threshold}",
        )
        labels = ordered_df["pred_cluster_id"].to_numpy(dtype=int)
        row = _metrics_from_labels(reference_df=reference_df, labels=labels)
        row["threshold"] = np.nan if threshold is None else float(threshold)
        rows.append(row)
        frame_by_threshold[threshold] = ordered_df
    sweep_df = pd.DataFrame(rows).sort_values("threshold", na_position="first").reset_index(drop=True)
    return sweep_df, frame_by_threshold


def load_prediction_route_candidate(
    *,
    route_name: str,
    val_prediction_path: Path,
    test_prediction_path: Path,
    val_reference_df: pd.DataFrame,
    test_reference_df: pd.DataFrame,
) -> RouteCandidate:
    val_prediction_df = pd.read_csv(val_prediction_path.resolve())
    test_prediction_df = pd.read_csv(test_prediction_path.resolve())
    val_sweep_df, val_by_threshold = _prediction_sweep_from_table(
        route_name=route_name,
        reference_df=val_reference_df,
        prediction_df=val_prediction_df,
    )
    best_row = _pick_threshold_row(val_sweep_df)
    chosen_threshold = None if pd.isna(best_row["threshold"]) else float(best_row["threshold"])
    chosen_val_df = val_by_threshold[chosen_threshold].copy().reset_index(drop=True)
    test_sweep_df, test_by_threshold = _prediction_sweep_from_table(
        route_name=route_name,
        reference_df=test_reference_df,
        prediction_df=test_prediction_df,
    )
    del test_sweep_df
    if chosen_threshold not in test_by_threshold:
        if len(test_by_threshold) == 1:
            chosen_test_df = next(iter(test_by_threshold.values())).copy().reset_index(drop=True)
        else:
            available = [value for value in test_by_threshold]
            raise ValueError(
                f"Test prediction table for {route_name} does not contain chosen threshold {chosen_threshold}. "
                f"Available: {available}"
            )
    else:
        chosen_test_df = test_by_threshold[chosen_threshold].copy().reset_index(drop=True)
    sweep_df = val_sweep_df.copy()
    sweep_df.insert(0, "route_name", route_name)
    sweep_df.insert(1, "source_kind", "prediction_table")
    return RouteCandidate(
        route_name=route_name,
        source_kind="prediction_table",
        source_ref=str(val_prediction_path.resolve()),
        chosen_threshold=chosen_threshold,
        val_prediction_df=chosen_val_df,
        test_prediction_df=chosen_test_df,
        val_sweep_df=sweep_df.sort_values(
            [column for column in ["ari", "pairwise_f1", "nmi", "threshold"] if column in sweep_df.columns],
            ascending=[False, False, False, True][: len([column for column in ["ari", "pairwise_f1", "nmi", "threshold"] if column in sweep_df.columns])],
        ).reset_index(drop=True),
        extra={
            "test_prediction_path": str(test_prediction_path.resolve()),
        },
    )


def _reorder_metadata_and_array(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    array: np.ndarray,
    route_name: str,
    is_pairwise_square: bool = False,
) -> tuple[pd.DataFrame, np.ndarray]:
    candidate_df = _normalize_metadata(candidate_df)
    if len(candidate_df) != len(array):
        raise ValueError(f"Row mismatch for {route_name}: metadata={len(candidate_df)} vs array={len(array)}")
    join_columns = ["image_id", "dataset"]
    merged = reference_df[join_columns].merge(
        candidate_df.assign(_row_index=np.arange(len(candidate_df), dtype=np.int32)),
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    if merged["_row_index"].isna().any():
        missing_rows = merged.loc[merged["_row_index"].isna(), join_columns].head(5).to_dict(orient="records")
        raise ValueError(f"Missing metadata rows for {route_name}, examples: {missing_rows}")
    order = merged["_row_index"].astype(int).to_numpy()
    ordered_df = candidate_df.iloc[order].reset_index(drop=True).copy()
    ensure_metadata_alignment(
        reference_df=reference_df.loc[:, _metadata_columns(reference_df)].reset_index(drop=True),
        candidate_df=ordered_df.loc[:, _metadata_columns(ordered_df)].reset_index(drop=True),
        split_name="lynx_route_ensemble",
        reference_name="reference_metadata",
        candidate_name=route_name,
    )
    reordered_array = array[order]
    if is_pairwise_square:
        reordered_array = reordered_array[:, order]
    return ordered_df, reordered_array


def load_embedding_route_candidate(
    *,
    route_name: str,
    source_dir: Path,
    thresholds: list[float],
    val_reference_df: pd.DataFrame,
    test_reference_df: pd.DataFrame,
) -> RouteCandidate:
    return load_embedding_router_route_candidate(
        route_name=route_name,
        source_dir=source_dir,
        thresholds=thresholds,
        val_reference_df=val_reference_df,
        test_reference_df=test_reference_df,
        router="average",
        mutual_k=5,
    )


def load_embedding_router_route_candidate(
    *,
    route_name: str,
    source_dir: Path,
    thresholds: list[float],
    val_reference_df: pd.DataFrame,
    test_reference_df: pd.DataFrame,
    router: str,
    mutual_k: int,
) -> RouteCandidate:
    bundle = load_cached_embedding_bundle(source_dir=source_dir.resolve(), name=route_name)
    val_df = bundle.val_df[bundle.val_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    test_df = bundle.test_df[bundle.test_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    val_embeddings = bundle.val_embeddings[(bundle.val_df["dataset"].astype(str) == LYNX_DATASET).to_numpy()]
    test_embeddings = bundle.test_embeddings[(bundle.test_df["dataset"].astype(str) == LYNX_DATASET).to_numpy()]
    ordered_val_df, ordered_val_embeddings = _reorder_metadata_and_array(
        reference_df=val_reference_df,
        candidate_df=val_df,
        array=val_embeddings,
        route_name=route_name,
        is_pairwise_square=False,
    )
    ordered_test_df, ordered_test_embeddings = _reorder_metadata_and_array(
        reference_df=test_reference_df,
        candidate_df=test_df,
        array=test_embeddings,
        route_name=route_name,
        is_pairwise_square=False,
    )
    val_score = cosine_similarity_matrix(ordered_val_embeddings.astype(np.float32, copy=False))
    val_sweep_df, val_prediction_df = _router_threshold_sweep(
        df=ordered_val_df,
        score_matrix=val_score,
        thresholds=[float(value) for value in thresholds],
        router=router,
        mutual_k=mutual_k,
    )
    best_row = pick_best_threshold_row(val_sweep_df)
    chosen_threshold = float(best_row["threshold"])
    chosen_val_df = val_prediction_df[np.isclose(val_prediction_df["threshold"].astype(float), chosen_threshold)].copy().reset_index(drop=True)
    test_score = cosine_similarity_matrix(ordered_test_embeddings.astype(np.float32, copy=False))
    if router in {"average", "complete"}:
        test_prediction_labels = _linkage_labels_from_score_matrix(score_matrix=test_score, threshold=chosen_threshold, method=router)
    elif router == "mutual_knn":
        test_prediction_labels = _mutual_knn_labels_from_score_matrix(score_matrix=test_score, threshold=chosen_threshold, k=mutual_k)
    else:
        raise ValueError(f"Unsupported router: {router}")
    chosen_test_df = _build_prediction_frame(
        reference_df=ordered_test_df,
        labels=test_prediction_labels,
        threshold=chosen_threshold,
    )
    sweep_df = val_sweep_df.copy()
    sweep_df.insert(0, "route_name", route_name)
    sweep_df.insert(1, "source_kind", "embedding_dir")
    return RouteCandidate(
        route_name=route_name,
        source_kind="embedding_dir",
        source_ref=str(source_dir.resolve()),
        chosen_threshold=chosen_threshold,
        val_prediction_df=chosen_val_df,
        test_prediction_df=chosen_test_df,
        val_sweep_df=sweep_df.sort_values(["ari", "pairwise_f1", "nmi", "threshold"], ascending=[False, False, False, True]).reset_index(drop=True),
        extra={
            "embedding_dim": int(ordered_val_embeddings.shape[1]),
            "router": router,
            "mutual_k": int(mutual_k) if router == "mutual_knn" else np.nan,
        },
    )


def load_score_route_candidate(
    *,
    route_name: str,
    score_space: str,
    val_score_path: Path,
    val_metadata_path: Path,
    test_score_path: Path,
    test_metadata_path: Path,
    thresholds: list[float],
    val_reference_df: pd.DataFrame,
    test_reference_df: pd.DataFrame,
) -> RouteCandidate:
    ordered_val_df, ordered_val_score = _reorder_metadata_and_array(
        reference_df=val_reference_df,
        candidate_df=pd.read_csv(val_metadata_path.resolve()),
        array=np.load(val_score_path.resolve()).astype(np.float32),
        route_name=route_name,
        is_pairwise_square=True,
    )
    ordered_test_df, ordered_test_score = _reorder_metadata_and_array(
        reference_df=test_reference_df,
        candidate_df=pd.read_csv(test_metadata_path.resolve()),
        array=np.load(test_score_path.resolve()).astype(np.float32),
        route_name=route_name,
        is_pairwise_square=True,
    )
    if ordered_val_score.ndim != 2 or ordered_val_score.shape[0] != ordered_val_score.shape[1]:
        raise ValueError(f"Validation score matrix for {route_name} must be square, got {ordered_val_score.shape}")
    if ordered_test_score.ndim != 2 or ordered_test_score.shape[0] != ordered_test_score.shape[1]:
        raise ValueError(f"Test score matrix for {route_name} must be square, got {ordered_test_score.shape}")
    val_sweep_df, val_prediction_df = run_score_threshold_sweep(
        df=ordered_val_df,
        score_matrix=ordered_val_score,
        thresholds=[float(value) for value in thresholds],
        score_space=score_space,
    )
    best_row = pick_best_threshold_row(val_sweep_df)
    chosen_threshold = float(best_row["threshold"])
    chosen_val_df = val_prediction_df[np.isclose(val_prediction_df["threshold"].astype(float), chosen_threshold)].copy().reset_index(drop=True)
    test_labels = cluster_labels_from_score_matrix(
        score_matrix=ordered_test_score,
        threshold=chosen_threshold,
        score_space=score_space,
    )
    chosen_test_df = _build_prediction_frame(
        reference_df=ordered_test_df,
        labels=test_labels,
        threshold=chosen_threshold,
    )
    sweep_df = val_sweep_df.copy()
    sweep_df.insert(0, "route_name", route_name)
    sweep_df.insert(1, "source_kind", "score_matrix")
    return RouteCandidate(
        route_name=route_name,
        source_kind="score_matrix",
        source_ref=str(val_score_path.resolve()),
        chosen_threshold=chosen_threshold,
        val_prediction_df=chosen_val_df,
        test_prediction_df=chosen_test_df,
        val_sweep_df=sweep_df.sort_values(["ari", "pairwise_f1", "nmi", "threshold"], ascending=[False, False, False, True]).reset_index(drop=True),
        extra={
            "score_space": score_space,
            "test_score_path": str(test_score_path.resolve()),
        },
    )


def build_route_agreement_table(candidates: list[RouteCandidate]) -> pd.DataFrame:
    _require_sklearn()
    if len(candidates) < 2:
        return pd.DataFrame(
            columns=[
                "split",
                "route_a",
                "route_b",
                "agreement_ratio",
                "same_cluster_jaccard",
                "ari_between_routes",
                "nmi_between_routes",
            ]
        )
    rows: list[dict[str, Any]] = []
    for split_name in ["val", "test"]:
        for left, right in itertools.combinations(candidates, 2):
            left_labels = left.val_labels if split_name == "val" else left.test_labels
            right_labels = right.val_labels if split_name == "val" else right.test_labels
            left_same = left_labels[:, None] == left_labels[None, :]
            right_same = right_labels[:, None] == right_labels[None, :]
            upper_mask = np.triu(np.ones_like(left_same, dtype=bool), k=1)
            left_upper = left_same[upper_mask]
            right_upper = right_same[upper_mask]
            intersection = int(np.logical_and(left_upper, right_upper).sum())
            union = int(np.logical_or(left_upper, right_upper).sum())
            rows.append(
                {
                    "split": split_name,
                    "route_a": left.route_name,
                    "route_b": right.route_name,
                    "agreement_ratio": round(float(np.mean(left_upper == right_upper)), 6),
                    "same_cluster_jaccard": round(float(intersection / union), 6) if union else 1.0,
                    "ari_between_routes": round(float(adjusted_rand_score(left_labels, right_labels)), 6),
                    "nmi_between_routes": round(float(normalized_mutual_info_score(left_labels, right_labels)), 6),
                    "route_a_clusters": int(pd.Series(left_labels).nunique()),
                    "route_b_clusters": int(pd.Series(right_labels).nunique()),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["split", "agreement_ratio", "same_cluster_jaccard", "route_a", "route_b"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)


def build_coassociation_score_matrix(label_list: list[np.ndarray]) -> np.ndarray:
    if not label_list:
        raise ValueError("Need at least one route to build co-association scores.")
    sample_count = len(label_list[0])
    score = np.zeros((sample_count, sample_count), dtype=np.float32)
    for labels in label_list:
        if len(labels) != sample_count:
            raise ValueError("Route label lengths do not match while building co-association scores.")
        same_cluster = (labels[:, None] == labels[None, :]).astype(np.float32)
        score += same_cluster
    score /= float(len(label_list))
    np.fill_diagonal(score, 1.0)
    return score.astype(np.float32, copy=False)


def build_coassociation_pair_table(
    *,
    reference_df: pd.DataFrame,
    route_names: list[str],
    label_list: list[np.ndarray],
    coassociation_score: np.ndarray,
    ensemble_labels: np.ndarray,
    min_score: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    upper_i, upper_j = np.triu_indices(len(reference_df), k=1)
    route_count = len(route_names)
    identities = reference_df["identity"].astype(str).tolist() if "identity" in reference_df.columns else None
    for left_index, right_index in zip(upper_i.tolist(), upper_j.tolist(), strict=True):
        score = float(coassociation_score[left_index, right_index])
        if score < float(min_score):
            continue
        vote_count = int(sum(int(labels[left_index] == labels[right_index]) for labels in label_list))
        row = {
            "image_id_a": str(reference_df.iloc[left_index]["image_id"]),
            "image_id_b": str(reference_df.iloc[right_index]["image_id"]),
            "coassociation_score": round(score, 6),
            "vote_count": vote_count,
            "route_count": route_count,
            "ensemble_same_cluster": bool(ensemble_labels[left_index] == ensemble_labels[right_index]),
        }
        if identities is not None:
            row["same_identity"] = bool(identities[left_index] == identities[right_index])
        rows.append(row)
    if not rows:
        columns = [
            "image_id_a",
            "image_id_b",
            "coassociation_score",
            "vote_count",
            "route_count",
            "ensemble_same_cluster",
        ]
        if identities is not None:
            columns.append("same_identity")
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["coassociation_score", "vote_count", "image_id_a", "image_id_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def _selected_candidates(candidates: list[RouteCandidate], max_route_candidates: int | None) -> list[RouteCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            float(candidate.val_sweep_df.iloc[0]["ari"]) if "ari" in candidate.val_sweep_df.columns else float("-inf"),
            float(candidate.val_sweep_df.iloc[0]["pairwise_f1"]) if "pairwise_f1" in candidate.val_sweep_df.columns else float("-inf"),
            float(candidate.val_sweep_df.iloc[0]["nmi"]) if "nmi" in candidate.val_sweep_df.columns else float("-inf"),
            -float(candidate.chosen_threshold) if candidate.chosen_threshold is not None else 0.0,
        ),
        reverse=True,
    )
    if max_route_candidates is None or max_route_candidates <= 0 or len(ranked) <= max_route_candidates:
        return ranked
    return ranked[: max_route_candidates]


def run_route_ensemble_probe(
    *,
    candidates: list[RouteCandidate],
    val_reference_df: pd.DataFrame,
    test_reference_df: pd.DataFrame,
    ensemble_thresholds: list[float],
    min_route_count: int,
    max_route_count: int | None,
    max_route_candidates: int | None,
    coassociation_export_min_score: float,
    export_test_override: bool,
    output_dir: Path,
    base_predictions_path: Path | None,
    sample_submission_path: Path | None,
    route_name: str,
) -> EnsembleProbeResult:
    if not candidates:
        raise ValueError("Need at least one route candidate for Lynx route ensemble.")
    route_sweep_df = (
        pd.concat([candidate.val_sweep_df for candidate in candidates], ignore_index=True)
        .sort_values(["route_name", "threshold"], na_position="first")
        .reset_index(drop=True)
    )
    route_candidates_df = pd.DataFrame([candidate.summary_row() for candidate in candidates]).sort_values(
        ["ari", "pairwise_f1", "nmi", "route_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = _selected_candidates(candidates=candidates, max_route_candidates=max_route_candidates)
    selected_routes_df = pd.DataFrame([candidate.summary_row() for candidate in selected]).sort_values(
        ["ari", "pairwise_f1", "nmi", "route_name"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    route_agreement_df = build_route_agreement_table(candidates=candidates)

    effective_max_route_count = len(selected) if max_route_count is None else min(int(max_route_count), len(selected))
    effective_min_route_count = min(max(1, int(min_route_count)), len(selected))
    if effective_min_route_count > effective_max_route_count:
        effective_min_route_count = effective_max_route_count

    best_single_row = route_candidates_df.iloc[0].to_dict()
    ensemble_rows: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_sort_key: tuple[float, float, float, int, float] = (-1.0, -1.0, -1.0, -1, 1.0)
    for route_count in range(effective_min_route_count, effective_max_route_count + 1):
        for subset in itertools.combinations(selected, route_count):
            subset_names = [candidate.route_name for candidate in subset]
            val_score = build_coassociation_score_matrix([candidate.val_labels for candidate in subset])
            val_sweep_df, val_prediction_df = run_score_threshold_sweep(
                df=val_reference_df,
                score_matrix=val_score,
                thresholds=[float(value) for value in ensemble_thresholds],
                score_space="unit_interval",
            )
            best_row = pick_best_threshold_row(val_sweep_df)
            chosen_threshold = float(best_row["threshold"])
            chosen_val_df = val_prediction_df[np.isclose(val_prediction_df["threshold"].astype(float), chosen_threshold)].copy().reset_index(drop=True)
            test_score = build_coassociation_score_matrix([candidate.test_labels for candidate in subset])
            test_labels = cluster_labels_from_score_matrix(
                score_matrix=test_score,
                threshold=chosen_threshold,
                score_space="unit_interval",
            )
            chosen_test_df = _build_prediction_frame(
                reference_df=test_reference_df,
                labels=test_labels,
                threshold=chosen_threshold,
            )
            row = {
                "route_names": "|".join(subset_names),
                "route_count": int(route_count),
                "vote_threshold": chosen_threshold,
                "ari": float(best_row["ari"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "nmi": float(best_row["nmi"]),
                "cluster_count": int(best_row["cluster_count"]),
                "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
                "mean_route_ari": round(float(np.mean([candidate.val_sweep_df.iloc[0]["ari"] for candidate in subset])), 6),
                "min_route_ari": round(float(np.min([candidate.val_sweep_df.iloc[0]["ari"] for candidate in subset])), 6),
                "ari_delta_vs_best_single": round(float(best_row["ari"]) - float(best_single_row["ari"]), 6),
                "pairwise_f1_delta_vs_best_single": round(float(best_row["pairwise_f1"]) - float(best_single_row["pairwise_f1"]), 6),
            }
            ensemble_rows.append(row)
            sort_key = (
                float(best_row["ari"]),
                float(best_row["pairwise_f1"]),
                float(best_row["nmi"]),
                int(route_count),
                -float(chosen_threshold),
            )
            if sort_key > best_sort_key:
                best_sort_key = sort_key
                best_payload = {
                    "row": row,
                    "route_names": subset_names,
                    "val_score": val_score,
                    "test_score": test_score,
                    "val_labels": chosen_val_df["pred_cluster_id"].to_numpy(dtype=int),
                    "test_labels": test_labels.astype(int),
                    "val_prediction_df": chosen_val_df.copy(),
                    "test_prediction_df": chosen_test_df.copy(),
                }
    if best_payload is None:
        raise RuntimeError("No Lynx route ensemble configuration was evaluated.")

    ensemble_sweep_df = pd.DataFrame(ensemble_rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "route_count", "vote_threshold"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    cluster_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cluster_rows.append(
            _cluster_shape_row(
                split_name="val",
                object_name=candidate.route_name,
                object_kind="single_route",
                labels=candidate.val_labels,
                route_count=1,
            )
        )
        cluster_rows.append(
            _cluster_shape_row(
                split_name="test",
                object_name=candidate.route_name,
                object_kind="single_route",
                labels=candidate.test_labels,
                route_count=1,
            )
        )
    cluster_rows.append(
        _cluster_shape_row(
            split_name="val",
            object_name=route_name,
            object_kind="best_ensemble",
            labels=best_payload["val_labels"],
            route_count=int(best_payload["row"]["route_count"]),
        )
    )
    cluster_rows.append(
        _cluster_shape_row(
            split_name="test",
            object_name=route_name,
            object_kind="best_ensemble",
            labels=best_payload["test_labels"],
            route_count=int(best_payload["row"]["route_count"]),
        )
    )
    cluster_shape_df = pd.DataFrame(cluster_rows).sort_values(
        ["object_kind", "split", "cluster_count", "object_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    best_val_prediction_df = best_payload["val_prediction_df"].copy().reset_index(drop=True)
    best_test_prediction_df = best_payload["test_prediction_df"].copy().reset_index(drop=True)
    best_val_prediction_df["route_name"] = route_name
    best_val_prediction_df["chosen_threshold"] = float(best_payload["row"]["vote_threshold"])
    best_val_prediction_df["ensemble_members"] = "|".join(best_payload["route_names"])
    best_val_prediction_df["vote_route_count"] = int(best_payload["row"]["route_count"])
    best_test_prediction_df["route_name"] = route_name
    best_test_prediction_df["chosen_threshold"] = float(best_payload["row"]["vote_threshold"])
    best_test_prediction_df["ensemble_members"] = "|".join(best_payload["route_names"])
    best_test_prediction_df["vote_route_count"] = int(best_payload["row"]["route_count"])
    best_test_prediction_df["dataset"] = LYNX_DATASET
    best_test_prediction_df["embedding_dim"] = 0
    best_test_prediction_df["rerank_enabled"] = False
    best_test_prediction_df["local_weight"] = 0.0
    best_test_prediction_df["cluster_label"] = [
        f"cluster_{LYNX_DATASET}_{int(label)}" for label in best_test_prediction_df["pred_cluster_id"].tolist()
    ]

    best_subset = [next(candidate for candidate in selected if candidate.route_name == name) for name in best_payload["route_names"]]
    best_val_coassociation_df = build_coassociation_pair_table(
        reference_df=val_reference_df,
        route_names=best_payload["route_names"],
        label_list=[candidate.val_labels for candidate in best_subset],
        coassociation_score=best_payload["val_score"],
        ensemble_labels=best_payload["val_labels"],
        min_score=coassociation_export_min_score,
    )
    best_test_coassociation_df = build_coassociation_pair_table(
        reference_df=test_reference_df,
        route_names=best_payload["route_names"],
        label_list=[candidate.test_labels for candidate in best_subset],
        coassociation_score=best_payload["test_score"],
        ensemble_labels=best_payload["test_labels"],
        min_score=coassociation_export_min_score,
    )

    exported_paths: dict[str, str] = {}
    if export_test_override:
        if base_predictions_path is None:
            raise ValueError("--export-test-override requires base_predictions_path")
        if sample_submission_path is None:
            raise ValueError("--export-test-override requires sample_submission_path")
        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)
        base_predictions_df = pd.read_csv(base_predictions_path.resolve())
        base_predictions_df["image_id"] = base_predictions_df["image_id"].astype(str)
        base_predictions_df["dataset"] = base_predictions_df["dataset"].astype(str)
        merged_pred_df = pd.concat(
            [
                base_predictions_df[base_predictions_df["dataset"] != LYNX_DATASET].copy(),
                best_test_prediction_df.copy(),
            ],
            ignore_index=True,
        )
        merged_path = tables_dir / "test_predictions_v1.csv"
        merged_pred_df.to_csv(merged_path, index=False)
        submission_path = output_dir / "submission.csv"
        build_submission(
            test_pred_df=merged_pred_df,
            sample_submission_path=sample_submission_path.resolve(),
            output_path=submission_path,
        )
        exported_paths = {
            "submission_path": str(submission_path.resolve()),
            "test_predictions_path": str(merged_path.resolve()),
            "lynx_override_path": str((tables_dir / "lynx_test_predictions_v1.csv").resolve()),
        }

    return EnsembleProbeResult(
        route_sweep_df=route_sweep_df,
        route_candidates_df=route_candidates_df,
        selected_routes_df=selected_routes_df,
        route_agreement_df=route_agreement_df,
        ensemble_sweep_df=ensemble_sweep_df,
        cluster_shape_df=cluster_shape_df,
        best_route_df=pd.DataFrame([best_payload["row"]]),
        best_val_prediction_df=best_val_prediction_df,
        best_test_prediction_df=best_test_prediction_df,
        best_val_coassociation_df=best_val_coassociation_df,
        best_test_coassociation_df=best_test_coassociation_df,
        best_route_names=list(best_payload["route_names"]),
        best_route_count=int(best_payload["row"]["route_count"]),
        best_vote_threshold=float(best_payload["row"]["vote_threshold"]),
        best_row=dict(best_payload["row"]),
        best_single_row=best_single_row,
        exported_paths=exported_paths,
    )


def load_reference_metadata(path: Path) -> pd.DataFrame:
    return _normalize_metadata(pd.read_csv(path.resolve()))


def write_ensemble_probe_result(result: EnsembleProbeResult, output_dir: Path) -> None:
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    table_map = {
        "route_sweep_v1.csv": result.route_sweep_df,
        "route_candidates_v1.csv": result.route_candidates_df,
        "selected_routes_v1.csv": result.selected_routes_df,
        "route_agreement_v1.csv": result.route_agreement_df,
        "ensemble_sweep_v1.csv": result.ensemble_sweep_df,
        "cluster_shape_v1.csv": result.cluster_shape_df,
        "best_route_v1.csv": result.best_route_df,
        "lynx_val_predictions_v1.csv": result.best_val_prediction_df,
        "lynx_test_predictions_v1.csv": result.best_test_prediction_df,
        "best_val_coassociation_pairs_v1.csv": result.best_val_coassociation_df,
        "best_test_coassociation_pairs_v1.csv": result.best_test_coassociation_df,
    }
    for filename, frame in table_map.items():
        frame.to_csv(tables_dir / filename, index=False)
    summary = {
        "route_name": str(result.best_test_prediction_df["route_name"].iloc[0]) if len(result.best_test_prediction_df) else "",
        "best_route_names": result.best_route_names,
        "best_route_count": result.best_route_count,
        "best_vote_threshold": result.best_vote_threshold,
        "best_row": result.best_row,
        "best_single_row": result.best_single_row,
        "exported_paths": result.exported_paths,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Lynx route co-association ensemble",
        "",
        f"- best routes: `{ '|'.join(result.best_route_names) }`",
        f"- route count: `{result.best_route_count}`",
        f"- vote threshold: `{result.best_vote_threshold}`",
        f"- val ARI: `{result.best_row.get('ari')}`",
        f"- val pairwise_f1: `{result.best_row.get('pairwise_f1')}`",
        f"- val cluster_count: `{result.best_row.get('cluster_count')}`",
        f"- submission: `{result.exported_paths.get('submission_path', '')}`",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _parse_embedding_route_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name, path
    name, path_text = text.split("=", 1)
    return name.strip(), Path(path_text.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LynxID2025 multi-router co-association ensemble.")
    parser.add_argument("--val-reference", required=True, type=Path)
    parser.add_argument("--test-reference", required=True, type=Path)
    parser.add_argument("--embedding-route", action="append", default=[], help="Route spec name=/path/to/experiment with embeddings/.")
    parser.add_argument("--routers", default="average,complete,mutual_knn")
    parser.add_argument("--thresholds", default="0.55,0.6,0.65,0.7,0.75,0.8,0.825,0.85,0.875,0.9,0.925")
    parser.add_argument("--mutual-k", type=int, default=5)
    parser.add_argument("--ensemble-thresholds", default="0.34,0.5,0.67,0.84,1.0")
    parser.add_argument("--min-route-count", type=int, default=2)
    parser.add_argument("--max-route-count", type=int, default=6)
    parser.add_argument("--max-route-candidates", type=int, default=6)
    parser.add_argument("--coassociation-export-min-score", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route-name", default="lynx_router_coassociation_v1")
    parser.add_argument("--base-predictions-path", type=Path)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--export-test-override", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    val_reference_df = load_reference_metadata(args.val_reference)
    test_reference_df = load_reference_metadata(args.test_reference)
    val_reference_df = val_reference_df[val_reference_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    test_reference_df = test_reference_df[test_reference_df["dataset"].astype(str) == LYNX_DATASET].copy().reset_index(drop=True)
    thresholds = _parse_float_list(args.thresholds)
    routers = [router.strip() for router in args.routers.split(",") if router.strip()]
    candidates: list[RouteCandidate] = []
    for spec in args.embedding_route:
        backbone_name, source_dir = _parse_embedding_route_spec(spec)
        for router in routers:
            route_name = f"{backbone_name}_{router}"
            candidates.append(
                load_embedding_router_route_candidate(
                    route_name=route_name,
                    source_dir=source_dir,
                    thresholds=thresholds,
                    val_reference_df=val_reference_df,
                    test_reference_df=test_reference_df,
                    router=router,
                    mutual_k=args.mutual_k,
                )
            )
    result = run_route_ensemble_probe(
        candidates=candidates,
        val_reference_df=val_reference_df,
        test_reference_df=test_reference_df,
        ensemble_thresholds=_parse_float_list(args.ensemble_thresholds),
        min_route_count=args.min_route_count,
        max_route_count=args.max_route_count,
        max_route_candidates=args.max_route_candidates,
        coassociation_export_min_score=args.coassociation_export_min_score,
        export_test_override=args.export_test_override,
        output_dir=args.output_dir,
        base_predictions_path=args.base_predictions_path,
        sample_submission_path=args.sample_submission_path,
        route_name=args.route_name,
    )
    write_ensemble_probe_result(result=result, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
