from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN, summarize_cluster_metrics


GRAPH_METHOD_CONNECTED_COMPONENTS = "connected_components"
GRAPH_METHOD_CHINESE_WHISPERS = "chinese_whispers"
GRAPH_METHODS = [GRAPH_METHOD_CONNECTED_COMPONENTS, GRAPH_METHOD_CHINESE_WHISPERS]


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.asarray([], dtype=int)
    _, normalized = np.unique(labels.astype(int), return_inverse=True)
    return normalized.astype(int)


def build_graph_adjacency(
    score_matrix: np.ndarray,
    *,
    threshold: float,
    top_k: int | None = None,
    mutual_top_k: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    score = np.asarray(score_matrix, dtype=np.float32)
    if score.ndim != 2 or score.shape[0] != score.shape[1]:
        raise ValueError("score_matrix must be square.")
    sample_count = int(score.shape[0])
    if sample_count == 0:
        return [], []

    effective_top_k = None if top_k is None else max(1, min(int(top_k), sample_count - 1))
    neighbor_indices: list[np.ndarray] = []
    neighbor_weights: list[np.ndarray] = []
    topk_sets: list[set[int]] = []
    for index in range(sample_count):
        row = score[index].copy()
        row[index] = -np.inf
        candidate_mask = row >= float(threshold)
        candidate_indices = np.flatnonzero(candidate_mask)
        if effective_top_k is not None and len(candidate_indices) > effective_top_k:
            candidate_scores = row[candidate_indices]
            order = np.argsort(-candidate_scores, kind="mergesort")[:effective_top_k]
            candidate_indices = candidate_indices[order]
        candidate_scores = row[candidate_indices]
        order = np.argsort(-candidate_scores, kind="mergesort")
        candidate_indices = candidate_indices[order].astype(np.int32, copy=False)
        candidate_scores = candidate_scores[order].astype(np.float32, copy=False)
        neighbor_indices.append(candidate_indices)
        neighbor_weights.append(candidate_scores)
        topk_sets.append(set(int(value) for value in candidate_indices.tolist()))

    if mutual_top_k:
        filtered_indices: list[np.ndarray] = []
        filtered_weights: list[np.ndarray] = []
        for index in range(sample_count):
            keep_mask = np.asarray(
                [index in topk_sets[int(neighbor)] for neighbor in neighbor_indices[index]],
                dtype=bool,
            )
            filtered_indices.append(neighbor_indices[index][keep_mask].astype(np.int32, copy=False))
            filtered_weights.append(neighbor_weights[index][keep_mask].astype(np.float32, copy=False))
        neighbor_indices = filtered_indices
        neighbor_weights = filtered_weights
    return neighbor_indices, neighbor_weights


def cluster_connected_components_from_score_matrix(
    score_matrix: np.ndarray,
    *,
    threshold: float,
    top_k: int | None = None,
    mutual_top_k: bool = False,
) -> np.ndarray:
    neighbor_indices, _neighbor_weights = build_graph_adjacency(
        score_matrix=score_matrix,
        threshold=threshold,
        top_k=top_k,
        mutual_top_k=mutual_top_k,
    )
    sample_count = len(neighbor_indices)
    labels = -np.ones(sample_count, dtype=np.int32)
    next_label = 0
    for start_index in range(sample_count):
        if labels[start_index] != -1:
            continue
        queue: deque[int] = deque([start_index])
        labels[start_index] = next_label
        while queue:
            node_index = queue.popleft()
            for neighbor_index in neighbor_indices[node_index]:
                neighbor_index_int = int(neighbor_index)
                if labels[neighbor_index_int] == -1:
                    labels[neighbor_index_int] = next_label
                    queue.append(neighbor_index_int)
        next_label += 1
    return _normalize_labels(labels)


def cluster_chinese_whispers_from_score_matrix(
    score_matrix: np.ndarray,
    *,
    threshold: float,
    top_k: int | None = None,
    mutual_top_k: bool = False,
    iterations: int = 20,
    seed: int = 42,
) -> np.ndarray:
    neighbor_indices, neighbor_weights = build_graph_adjacency(
        score_matrix=score_matrix,
        threshold=threshold,
        top_k=top_k,
        mutual_top_k=mutual_top_k,
    )
    sample_count = len(neighbor_indices)
    labels = np.arange(sample_count, dtype=np.int32)
    rng = random.Random(int(seed))
    update_order = list(range(sample_count))
    for _ in range(max(1, int(iterations))):
        rng.shuffle(update_order)
        changed = False
        for node_index in update_order:
            if len(neighbor_indices[node_index]) == 0:
                continue
            label_weight: dict[int, float] = {}
            for neighbor_index, neighbor_weight in zip(neighbor_indices[node_index], neighbor_weights[node_index]):
                label = int(labels[int(neighbor_index)])
                label_weight[label] = label_weight.get(label, 0.0) + float(neighbor_weight)
            best_label, _best_weight = sorted(label_weight.items(), key=lambda item: (-item[1], item[0]))[0]
            if int(labels[node_index]) != int(best_label):
                labels[node_index] = int(best_label)
                changed = True
        if not changed:
            break
    return _normalize_labels(labels)


def cluster_labels_from_score_graph(
    score_matrix: np.ndarray,
    *,
    threshold: float,
    method: str,
    top_k: int | None = None,
    mutual_top_k: bool = False,
    iterations: int = 20,
    seed: int = 42,
) -> np.ndarray:
    if method == GRAPH_METHOD_CONNECTED_COMPONENTS:
        return cluster_connected_components_from_score_matrix(
            score_matrix=score_matrix,
            threshold=threshold,
            top_k=top_k,
            mutual_top_k=mutual_top_k,
        )
    if method == GRAPH_METHOD_CHINESE_WHISPERS:
        return cluster_chinese_whispers_from_score_matrix(
            score_matrix=score_matrix,
            threshold=threshold,
            top_k=top_k,
            mutual_top_k=mutual_top_k,
            iterations=iterations,
            seed=seed,
        )
    raise ValueError(f"Unsupported graph clustering method: {method}")


def run_graph_threshold_sweep(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    thresholds: list[float],
    method: str,
    top_k: int | None = None,
    mutual_top_k: bool = False,
    iterations: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    has_identity = "identity" in df.columns and df["identity"].fillna("").astype(str).ne("").any()
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    keep_columns = [column for column in ["image_id", "dataset", "identity", PATH_COLUMN] if column in df.columns]
    dataset_name = str(df["dataset"].iloc[0])
    for threshold in [float(value) for value in thresholds]:
        pred_labels = cluster_labels_from_score_graph(
            score_matrix=score_matrix,
            threshold=threshold,
            method=method,
            top_k=top_k,
            mutual_top_k=mutual_top_k,
            iterations=iterations,
            seed=seed,
        )
        counts = pd.Series(pred_labels).value_counts()
        row: dict[str, object] = {
            "dataset": dataset_name,
            "method": str(method),
            "threshold": float(threshold),
            "top_k": int(top_k) if top_k is not None else -1,
            "mutual_top_k": bool(mutual_top_k),
            "samples": int(len(df)),
            "cluster_count": int(counts.size),
            "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
        }
        if has_identity and true_labels is not None:
            row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels))
        frame = df.loc[:, keep_columns].copy().reset_index(drop=True)
        frame["method"] = str(method)
        frame["threshold"] = float(threshold)
        frame["top_k"] = int(top_k) if top_k is not None else -1
        frame["mutual_top_k"] = bool(mutual_top_k)
        frame["pred_cluster_id"] = pred_labels.astype(int)
        prediction_frames.append(frame)
        rows.append(row)
    sweep_df = pd.DataFrame(rows).sort_values(["method", "threshold"]).reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df
