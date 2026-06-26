from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN, summarize_cluster_metrics
from .graph_clustering import cluster_labels_from_score_graph
from .transductive_seed_refinement import cluster_labels_from_score_matrix


@dataclass
class _UnionFind:
    parents: list[int]

    @classmethod
    def create(cls, size: int) -> "_UnionFind":
        return cls(parents=list(range(max(0, int(size)))))

    def find(self, value: int) -> int:
        parent = self.parents[value]
        while parent != self.parents[parent]:
            self.parents[parent] = self.parents[self.parents[parent]]
            parent = self.parents[parent]
        self.parents[value] = parent
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.asarray([], dtype=np.int32)
    _, normalized = np.unique(labels.astype(int), return_inverse=True)
    return normalized.astype(np.int32)


def _pair_score_stats(score_matrix: np.ndarray, members: list[int]) -> tuple[float, float]:
    if len(members) < 2:
        return 0.0, 0.0
    block = np.asarray(score_matrix[np.ix_(members, members)], dtype=np.float32)
    upper = block[np.triu_indices_from(block, k=1)]
    if len(upper) == 0:
        return 0.0, 0.0
    return float(np.mean(upper)), float(np.min(upper))


def extract_graph_merge_candidates(
    *,
    base_labels: np.ndarray,
    graph_labels: np.ndarray,
    score_matrix: np.ndarray,
    graph_threshold: float,
) -> pd.DataFrame:
    base = np.asarray(base_labels, dtype=np.int32)
    graph = np.asarray(graph_labels, dtype=np.int32)
    if len(base) != len(graph):
        raise ValueError("base_labels and graph_labels must have the same length.")
    rows: list[dict[str, object]] = []
    for graph_cluster_id in sorted(np.unique(graph).tolist()):
        members = np.flatnonzero(graph == int(graph_cluster_id)).tolist()
        base_cluster_ids = sorted(set(int(base[index]) for index in members))
        if len(base_cluster_ids) <= 1:
            continue
        mean_score, min_score = _pair_score_stats(score_matrix=score_matrix, members=members)
        rows.append(
            {
                "graph_threshold": float(graph_threshold),
                "graph_cluster_id": int(graph_cluster_id),
                "candidate_key": "|".join(str(index) for index in members),
                "member_indices": "|".join(str(index) for index in members),
                "candidate_size": int(len(members)),
                "base_cluster_ids": "|".join(str(value) for value in base_cluster_ids),
                "base_cluster_count": int(len(base_cluster_ids)),
                "mean_score": round(float(mean_score), 6),
                "min_score": round(float(min_score), 6),
            }
        )
    return pd.DataFrame(rows)


def aggregate_graph_merge_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
    if candidate_df.empty:
        return pd.DataFrame(
            columns=[
                "candidate_key",
                "member_indices",
                "candidate_size",
                "base_cluster_ids",
                "base_cluster_count",
                "mean_score",
                "min_score",
                "stable_count",
                "stable_thresholds",
            ]
        )
    grouped_rows: list[dict[str, object]] = []
    for candidate_key, group in candidate_df.groupby("candidate_key", sort=False):
        thresholds = sorted({round(float(value), 6) for value in group["graph_threshold"].tolist()})
        first_row = group.iloc[0]
        grouped_rows.append(
            {
                "candidate_key": str(candidate_key),
                "member_indices": str(first_row["member_indices"]),
                "candidate_size": int(first_row["candidate_size"]),
                "base_cluster_ids": str(first_row["base_cluster_ids"]),
                "base_cluster_count": int(first_row["base_cluster_count"]),
                "mean_score": float(first_row["mean_score"]),
                "min_score": float(first_row["min_score"]),
                "stable_count": int(len(thresholds)),
                "stable_thresholds": "|".join(f"{value:.2f}".rstrip("0").rstrip(".") for value in thresholds),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(
        ["stable_count", "mean_score", "candidate_size"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_consensus_merge_candidates(
    pair_vote_df: pd.DataFrame,
    *,
    min_merge_votes: int = 2,
    min_vote_ratio: float = 0.67,
    min_support_pair_count: int = 1,
    pair_score_col: str = "pair_score",
    min_pair_score: float = 0.0,
    probability_col: str = "xgb_same_identity_prob",
    min_pair_probability: float = 0.0,
) -> pd.DataFrame:
    if pair_vote_df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_pair_key",
                "base_cluster_ids",
                "left_cluster_id",
                "right_cluster_id",
                "support_pair_count",
                "max_merge_votes",
                "max_vote_ratio",
                "mean_vote_ratio",
                "max_pair_score",
                "mean_pair_score",
                "max_pair_probability",
                "mean_pair_probability",
                "conflict_methods",
                "member_pair_indices",
            ]
        )

    frame = pair_vote_df.copy().reset_index(drop=True)
    if pair_score_col not in frame.columns:
        frame[pair_score_col] = np.nan
    if probability_col not in frame.columns:
        frame[probability_col] = np.nan

    candidate_df = frame[
        frame["base_cluster_left"].astype(int).ne(frame["base_cluster_right"].astype(int))
        & frame["merge_votes"].astype(int).ge(int(min_merge_votes))
        & frame["vote_ratio"].astype(float).ge(float(min_vote_ratio))
        & frame[pair_score_col].fillna(-np.inf).astype(float).ge(float(min_pair_score))
        & frame[probability_col].fillna(-np.inf).astype(float).ge(float(min_pair_probability))
    ].copy()
    if candidate_df.empty:
        return pd.DataFrame()

    candidate_df["left_cluster_id"] = candidate_df[["base_cluster_left", "base_cluster_right"]].min(axis=1).astype(int)
    candidate_df["right_cluster_id"] = candidate_df[["base_cluster_left", "base_cluster_right"]].max(axis=1).astype(int)
    candidate_df["cluster_pair_key"] = (
        candidate_df["left_cluster_id"].astype(str) + "|" + candidate_df["right_cluster_id"].astype(str)
    )

    rows: list[dict[str, object]] = []
    for cluster_pair_key, group_df in candidate_df.groupby("cluster_pair_key", sort=False):
        support_pair_count = int(len(group_df))
        if support_pair_count < int(min_support_pair_count):
            continue
        left_cluster_id = int(group_df["left_cluster_id"].iloc[0])
        right_cluster_id = int(group_df["right_cluster_id"].iloc[0])
        rows.append(
            {
                "cluster_pair_key": str(cluster_pair_key),
                "base_cluster_ids": f"{left_cluster_id}|{right_cluster_id}",
                "left_cluster_id": left_cluster_id,
                "right_cluster_id": right_cluster_id,
                "support_pair_count": support_pair_count,
                "max_merge_votes": int(group_df["merge_votes"].astype(int).max()),
                "max_vote_ratio": round(float(group_df["vote_ratio"].astype(float).max()), 6),
                "mean_vote_ratio": round(float(group_df["vote_ratio"].astype(float).mean()), 6),
                "max_pair_score": round(float(group_df[pair_score_col].astype(float).max()), 6),
                "mean_pair_score": round(float(group_df[pair_score_col].astype(float).mean()), 6),
                "max_pair_probability": round(float(group_df[probability_col].astype(float).max()), 6),
                "mean_pair_probability": round(float(group_df[probability_col].astype(float).mean()), 6),
                "conflict_methods": "|".join(
                    sorted(
                        {
                            method
                            for value in group_df["conflict_methods"].astype(str).tolist()
                            for method in value.split("|")
                            if method
                        }
                    )
                ),
                "member_pair_indices": "|".join(str(int(index)) for index in group_df.index.tolist()),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["max_vote_ratio", "support_pair_count", "max_pair_score", "max_pair_probability"],
            ascending=[False, False, False, False],
        )
        .reset_index(drop=True)
    )


def apply_graph_merge_overlay(base_labels: np.ndarray, candidate_df: pd.DataFrame) -> np.ndarray:
    base = np.asarray(base_labels, dtype=np.int32)
    if base.size == 0:
        return np.asarray([], dtype=np.int32)
    if candidate_df.empty:
        return _normalize_labels(base)

    unique_base_ids = sorted(np.unique(base).tolist())
    base_to_index = {int(cluster_id): offset for offset, cluster_id in enumerate(unique_base_ids)}
    union_find = _UnionFind.create(len(unique_base_ids))

    for row in candidate_df.itertuples(index=False):
        cluster_ids = [int(value) for value in str(row.base_cluster_ids).split("|") if str(value) != ""]
        if len(cluster_ids) <= 1:
            continue
        leader = cluster_ids[0]
        for other in cluster_ids[1:]:
            union_find.union(base_to_index[leader], base_to_index[other])

    merged_labels = np.empty_like(base)
    root_to_label: dict[int, int] = {}
    next_label = 0
    for index, cluster_id in enumerate(base.tolist()):
        root = union_find.find(base_to_index[int(cluster_id)])
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        merged_labels[index] = root_to_label[root]
    return _normalize_labels(merged_labels)


def run_graph_merge_overlay_sweep(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    base_thresholds: list[float],
    graph_thresholds: list[float],
    method: str,
    top_k: int | None = None,
    mutual_top_k: bool = False,
    min_mean_scores: list[float],
    min_stable_counts: list[int],
    iterations: int = 20,
    seed: int = 42,
    score_space: str = "unit_interval",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    has_identity = "identity" in df.columns and df["identity"].fillna("").astype(str).ne("").any()
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    dataset_name = str(df["dataset"].iloc[0])
    keep_columns = [column for column in ["image_id", "dataset", "identity", PATH_COLUMN] if column in df.columns]

    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    candidate_summary_frames: list[pd.DataFrame] = []

    for base_threshold in [float(value) for value in base_thresholds]:
        base_labels = cluster_labels_from_score_matrix(
            score_matrix=score_matrix,
            threshold=float(base_threshold),
            score_space=score_space,
        )
        per_threshold_candidates: list[pd.DataFrame] = []
        for graph_threshold in [float(value) for value in graph_thresholds]:
            graph_labels = cluster_labels_from_score_graph(
                score_matrix=score_matrix,
                threshold=float(graph_threshold),
                method=method,
                top_k=top_k,
                mutual_top_k=mutual_top_k,
                iterations=iterations,
                seed=seed,
            )
            per_threshold_candidates.append(
                extract_graph_merge_candidates(
                    base_labels=base_labels,
                    graph_labels=graph_labels,
                    score_matrix=score_matrix,
                    graph_threshold=float(graph_threshold),
                )
            )
        candidate_df = (
            pd.concat(per_threshold_candidates, ignore_index=True)
            if per_threshold_candidates
            else pd.DataFrame()
        )
        aggregated_candidate_df = aggregate_graph_merge_candidates(candidate_df)
        if not aggregated_candidate_df.empty:
            aggregated_candidate_df = aggregated_candidate_df.copy()
            aggregated_candidate_df["base_threshold"] = float(base_threshold)
            candidate_summary_frames.append(aggregated_candidate_df)

        for min_stable_count in [int(value) for value in min_stable_counts]:
            for min_mean_score in [float(value) for value in min_mean_scores]:
                accepted_df = aggregated_candidate_df[
                    aggregated_candidate_df["stable_count"].astype(int) >= int(min_stable_count)
                ].copy()
                accepted_df = accepted_df[accepted_df["mean_score"].astype(float) >= float(min_mean_score)].copy()
                overlay_labels = apply_graph_merge_overlay(base_labels=base_labels, candidate_df=accepted_df)
                counts = pd.Series(overlay_labels).value_counts()
                row: dict[str, object] = {
                    "dataset": dataset_name,
                    "base_threshold": float(base_threshold),
                    "graph_method": str(method),
                    "graph_top_k": int(top_k) if top_k is not None else -1,
                    "mutual_top_k": bool(mutual_top_k),
                    "graph_threshold_min": min(float(value) for value in graph_thresholds),
                    "graph_threshold_max": max(float(value) for value in graph_thresholds),
                    "min_stable_count": int(min_stable_count),
                    "min_mean_score": float(min_mean_score),
                    "accepted_merge_candidates": int(len(accepted_df)),
                    "samples": int(len(df)),
                    "cluster_count": int(counts.size),
                    "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                }
                if has_identity and true_labels is not None:
                    row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=overlay_labels))
                rows.append(row)

                frame = df.loc[:, keep_columns].copy().reset_index(drop=True)
                frame["base_threshold"] = float(base_threshold)
                frame["graph_method"] = str(method)
                frame["graph_top_k"] = int(top_k) if top_k is not None else -1
                frame["mutual_top_k"] = bool(mutual_top_k)
                frame["min_stable_count"] = int(min_stable_count)
                frame["min_mean_score"] = float(min_mean_score)
                frame["accepted_merge_candidates"] = int(len(accepted_df))
                frame["pred_cluster_id"] = overlay_labels.astype(int)
                prediction_frames.append(frame)

    sweep_df = pd.DataFrame(rows)
    if not sweep_df.empty:
        sweep_df = sweep_df.sort_values(
            ["ari", "pairwise_f1", "accepted_merge_candidates", "min_mean_score", "base_threshold"],
            ascending=[False, False, True, False, True],
        ).reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    candidate_summary_df = (
        pd.concat(candidate_summary_frames, ignore_index=True) if candidate_summary_frames else pd.DataFrame()
    )
    return sweep_df, prediction_df, candidate_summary_df
