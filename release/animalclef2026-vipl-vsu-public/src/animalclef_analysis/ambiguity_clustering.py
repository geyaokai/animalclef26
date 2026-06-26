from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN, summarize_cluster_metrics
from .orb_rerank_baseline import score_matrix_to_distance
from .texas_unsupervised import pair_agreement_score


AMBIGUITY_METHOD_AVERAGE = "average_linkage"
AMBIGUITY_METHOD_CHINESE_WHISPERS = "chinese_whispers"
AMBIGUITY_METHOD_DBSCAN = "dbscan"
AMBIGUITY_METHOD_FINCH_LIKE = "finch_like"
AMBIGUITY_METHODS = [
    AMBIGUITY_METHOD_AVERAGE,
    AMBIGUITY_METHOD_CHINESE_WHISPERS,
    AMBIGUITY_METHOD_DBSCAN,
    AMBIGUITY_METHOD_FINCH_LIKE,
]


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.asarray([], dtype=np.int32)
    _, normalized = np.unique(labels.astype(np.int64), return_inverse=True)
    return normalized.astype(np.int32, copy=False)


def _validate_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return array


def _has_identity_labels(df: pd.DataFrame) -> bool:
    return "identity" in df.columns and df["identity"].fillna("").astype(str).ne("").any()


def _build_prediction_frame(
    df: pd.DataFrame,
    pred_labels: np.ndarray,
    *,
    method: str,
    threshold: float,
    extra_columns: dict[str, object] | None = None,
) -> pd.DataFrame:
    keep_columns = [column for column in ["image_id", "dataset", "identity", PATH_COLUMN] if column in df.columns]
    frame = df.loc[:, keep_columns].copy().reset_index(drop=True)
    frame["method"] = str(method)
    frame["threshold"] = float(threshold)
    frame["pred_cluster_id"] = np.asarray(pred_labels, dtype=np.int32)
    if extra_columns:
        for key, value in extra_columns.items():
            frame[key] = value
    return frame


def _score_to_distance_matrix(score_matrix: np.ndarray, *, score_space: str) -> np.ndarray:
    if score_space == "unit_interval":
        return score_matrix_to_distance(score_matrix)
    if score_space == "cosine_similarity":
        distance = 1.0 - np.clip(np.asarray(score_matrix, dtype=np.float32), -1.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        return distance.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported score_space: {score_space}")


def cluster_labels_from_dbscan_distance(
    distance_matrix: np.ndarray,
    *,
    eps: float,
    min_samples: int = 2,
) -> np.ndarray:
    distance = _validate_square_matrix(distance_matrix, name="distance_matrix")
    sample_count = int(distance.shape[0])
    if sample_count == 0:
        return np.asarray([], dtype=np.int32)
    if sample_count == 1:
        return np.asarray([0], dtype=np.int32)

    eps = float(eps)
    min_samples = max(1, int(min_samples))
    neighbor_indices = [np.flatnonzero(distance[index] <= eps).astype(np.int32) for index in range(sample_count)]
    is_core = np.asarray([len(indices) >= min_samples for indices in neighbor_indices], dtype=bool)

    visited = np.zeros(sample_count, dtype=bool)
    labels = -np.ones(sample_count, dtype=np.int32)
    next_cluster_id = 0
    for start_index in range(sample_count):
        if visited[start_index]:
            continue
        visited[start_index] = True
        if not is_core[start_index]:
            continue
        labels[start_index] = next_cluster_id
        queue: deque[int] = deque([int(start_index)])
        while queue:
            node_index = queue.popleft()
            if not is_core[node_index]:
                continue
            for neighbor_index in neighbor_indices[node_index]:
                neighbor_index_int = int(neighbor_index)
                if not visited[neighbor_index_int]:
                    visited[neighbor_index_int] = True
                    if is_core[neighbor_index_int]:
                        queue.append(neighbor_index_int)
                if labels[neighbor_index_int] == -1:
                    labels[neighbor_index_int] = next_cluster_id
        next_cluster_id += 1

    for index in range(sample_count):
        if labels[index] == -1:
            labels[index] = next_cluster_id
            next_cluster_id += 1
    return _normalize_labels(labels)


def cluster_labels_from_dbscan_score_matrix(
    score_matrix: np.ndarray,
    *,
    threshold: float,
    min_samples: int = 2,
    score_space: str = "unit_interval",
) -> np.ndarray:
    score = _validate_square_matrix(score_matrix, name="score_matrix")
    distance = _score_to_distance_matrix(score, score_space=score_space)
    eps = 1.0 - float(threshold)
    return cluster_labels_from_dbscan_distance(distance_matrix=distance, eps=eps, min_samples=min_samples)


def run_dbscan_threshold_sweep(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    thresholds: list[float],
    min_samples_values: list[int],
    score_space: str = "unit_interval",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    has_identity = _has_identity_labels(df)
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    dataset_name = str(df["dataset"].iloc[0])
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for min_samples in [max(1, int(value)) for value in min_samples_values]:
        for threshold in [float(value) for value in thresholds]:
            pred_labels = cluster_labels_from_dbscan_score_matrix(
                score_matrix=score_matrix,
                threshold=threshold,
                min_samples=min_samples,
                score_space=score_space,
            )
            counts = pd.Series(pred_labels).value_counts()
            row: dict[str, object] = {
                "dataset": dataset_name,
                "method": AMBIGUITY_METHOD_DBSCAN,
                "threshold": float(threshold),
                "eps": round(float(1.0 - float(threshold)), 6),
                "min_samples": int(min_samples),
                "samples": int(len(df)),
                "cluster_count": int(counts.size),
                "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
            }
            if has_identity and true_labels is not None:
                row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels))
            rows.append(row)
            prediction_frames.append(
                _build_prediction_frame(
                    df=df,
                    pred_labels=pred_labels,
                    method=AMBIGUITY_METHOD_DBSCAN,
                    threshold=threshold,
                    extra_columns={
                        "eps": round(float(1.0 - float(threshold)), 6),
                        "min_samples": int(min_samples),
                    },
                )
            )
    sweep_df = (
        pd.DataFrame(rows)
        .sort_values(["threshold", "min_samples"], ascending=[True, True])
        .reset_index(drop=True)
    )
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def cluster_labels_from_finch_like_score_matrix(
    score_matrix: np.ndarray,
    *,
    min_link_score: float = 0.0,
) -> np.ndarray:
    score = _validate_square_matrix(score_matrix, name="score_matrix")
    sample_count = int(score.shape[0])
    if sample_count == 0:
        return np.asarray([], dtype=np.int32)
    if sample_count == 1:
        return np.asarray([0], dtype=np.int32)

    masked = score.copy()
    np.fill_diagonal(masked, -np.inf)
    first_neighbor = np.argmax(masked, axis=1).astype(np.int32)
    first_neighbor_score = masked[np.arange(sample_count), first_neighbor].astype(np.float32, copy=False)

    parent = np.arange(sample_count, dtype=np.int32)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return int(index)

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    shared_neighbor_groups: dict[int, list[int]] = defaultdict(list)
    for index in range(sample_count):
        if float(first_neighbor_score[index]) < float(min_link_score):
            continue
        neighbor_index = int(first_neighbor[index])
        union(index, neighbor_index)
        shared_neighbor_groups[neighbor_index].append(index)

    for members in shared_neighbor_groups.values():
        if len(members) < 2:
            continue
        anchor = int(members[0])
        for member in members[1:]:
            union(anchor, int(member))

    labels = np.asarray([find(index) for index in range(sample_count)], dtype=np.int32)
    return _normalize_labels(labels)


def run_finch_like_threshold_sweep(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    *,
    min_link_scores: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    has_identity = _has_identity_labels(df)
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    dataset_name = str(df["dataset"].iloc[0])
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for min_link_score in [float(value) for value in min_link_scores]:
        pred_labels = cluster_labels_from_finch_like_score_matrix(
            score_matrix=score_matrix,
            min_link_score=min_link_score,
        )
        counts = pd.Series(pred_labels).value_counts()
        row: dict[str, object] = {
            "dataset": dataset_name,
            "method": AMBIGUITY_METHOD_FINCH_LIKE,
            "threshold": float(min_link_score),
            "min_link_score": float(min_link_score),
            "samples": int(len(df)),
            "cluster_count": int(counts.size),
            "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
        }
        if has_identity and true_labels is not None:
            row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels))
        rows.append(row)
        prediction_frames.append(
            _build_prediction_frame(
                df=df,
                pred_labels=pred_labels,
                method=AMBIGUITY_METHOD_FINCH_LIKE,
                threshold=min_link_score,
                extra_columns={"min_link_score": float(min_link_score)},
            )
        )
    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def build_partition_agreement_table(label_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    items = [(str(name), np.asarray(labels, dtype=np.int32)) for name, labels in label_map.items()]
    for method_left, labels_left in items:
        for method_right, labels_right in items:
            rows.append(
                {
                    "method_left": method_left,
                    "method_right": method_right,
                    "pair_agreement": pair_agreement_score(labels_left, labels_right),
                }
            )
    return pd.DataFrame(rows)


def build_pair_disagreement_table(
    pair_df: pd.DataFrame,
    *,
    label_map: dict[str, np.ndarray],
    base_method: str = AMBIGUITY_METHOD_AVERAGE,
    probability_col: str = "xgb_same_identity_prob",
    base_threshold: float = 0.25,
    border_width: float = 0.08,
) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()
    if base_method not in label_map:
        raise KeyError(f"Missing base method labels: {base_method}")

    frame = pair_df.copy().reset_index(drop=True)
    left_index = frame["left_index"].to_numpy(dtype=int)
    right_index = frame["right_index"].to_numpy(dtype=int)

    method_names = [str(name) for name in label_map.keys()]
    same_columns: list[str] = []
    for method_name in method_names:
        labels = np.asarray(label_map[method_name], dtype=np.int32)
        if labels.ndim != 1:
            raise ValueError(f"Labels for method {method_name} must be 1D.")
        max_left = int(np.max(left_index)) if len(left_index) else -1
        max_right = int(np.max(right_index)) if len(right_index) else -1
        if max_left >= len(labels) or max_right >= len(labels):
            raise ValueError(f"Pair indices exceed label length for method {method_name}.")
        same_column = f"same_{method_name}"
        frame[same_column] = labels[left_index] == labels[right_index]
        same_columns.append(same_column)

    base_same_column = f"same_{base_method}"
    base_labels = np.asarray(label_map[base_method], dtype=np.int32)
    frame["base_cluster_left"] = base_labels[left_index].astype(np.int32)
    frame["base_cluster_right"] = base_labels[right_index].astype(np.int32)
    frame["base_cluster_pair"] = [
        f"{min(int(left), int(right))}|{max(int(left), int(right))}"
        for left, right in zip(frame["base_cluster_left"], frame["base_cluster_right"], strict=True)
    ]

    alt_methods = [name for name in method_names if name != base_method]
    alt_same_columns = [f"same_{method_name}" for method_name in alt_methods]
    base_same = frame[base_same_column].to_numpy(dtype=bool)
    if alt_same_columns:
        alt_same = np.column_stack([frame[column].to_numpy(dtype=bool) for column in alt_same_columns])
    else:
        alt_same = np.zeros((len(frame), 0), dtype=bool)
    alt_count = int(len(alt_same_columns))

    merge_votes = np.where(~base_same, alt_same.sum(axis=1), 0).astype(np.int32)
    split_votes = np.where(base_same, (~alt_same).sum(axis=1), 0).astype(np.int32)
    conflict_ratio = np.zeros(len(frame), dtype=np.float32)
    if alt_count > 0:
        conflict_ratio = np.where(base_same, split_votes / alt_count, merge_votes / alt_count).astype(np.float32)

    same_vote_count = np.column_stack([frame[column].to_numpy(dtype=bool) for column in same_columns]).sum(axis=1)
    same_vote_ratio = same_vote_count.astype(np.float32) / max(1, len(same_columns))
    vote_balance_score = 1.0 - np.abs(same_vote_ratio - 0.5) / 0.5
    vote_balance_score = np.clip(vote_balance_score, 0.0, 1.0)

    if probability_col not in frame.columns:
        raise KeyError(f"Missing probability column: {probability_col}")
    probability = frame[probability_col].astype(float).to_numpy()
    border_score = np.clip(1.0 - np.abs(probability - float(base_threshold)) / max(float(border_width), 1e-6), 0.0, 1.0)

    ambiguity_score = 0.50 * conflict_ratio + 0.30 * border_score + 0.20 * vote_balance_score
    direction = np.where(
        merge_votes > split_votes,
        "merge",
        np.where(split_votes > merge_votes, "split", "mixed"),
    )
    direction = np.where((merge_votes == 0) & (split_votes == 0), "agree", direction)

    conflict_methods: list[str] = []
    for row_index in range(len(frame)):
        row_methods = []
        for method_name, same_column in zip(alt_methods, alt_same_columns, strict=True):
            row_same = bool(frame.at[row_index, same_column])
            if row_same != bool(base_same[row_index]):
                row_methods.append(str(method_name))
        conflict_methods.append("|".join(row_methods))

    frame["merge_votes"] = merge_votes
    frame["split_votes"] = split_votes
    frame["base_conflict_ratio"] = np.round(conflict_ratio, 6)
    frame["same_vote_ratio"] = np.round(same_vote_ratio, 6)
    frame["vote_balance_score"] = np.round(vote_balance_score, 6)
    frame["border_score"] = np.round(border_score, 6)
    frame["ambiguity_score"] = np.round(ambiguity_score, 6)
    frame["vote_direction"] = direction.astype(str)
    frame["conflict_methods"] = conflict_methods
    return frame.sort_values(
        ["ambiguity_score", probability_col, "merge_votes", "split_votes"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def build_pair_vote_summary_table(
    pair_df: pd.DataFrame,
    *,
    label_map: dict[str, np.ndarray],
    base_method: str = AMBIGUITY_METHOD_AVERAGE,
    probability_col: str = "xgb_same_identity_prob",
    base_threshold: float = 0.25,
    border_width: float = 0.08,
    score_matrix: np.ndarray | None = None,
    pair_score_col: str = "pair_score",
) -> pd.DataFrame:
    frame = build_pair_disagreement_table(
        pair_df=pair_df,
        label_map=label_map,
        base_method=base_method,
        probability_col=probability_col,
        base_threshold=base_threshold,
        border_width=border_width,
    )
    if frame.empty:
        return frame

    total_votes = max(0, len(label_map) - 1)
    frame["total_votes"] = int(total_votes)
    if total_votes > 0:
        frame["vote_ratio"] = np.round(frame["merge_votes"].astype(float) / float(total_votes), 6)
        frame["split_ratio"] = np.round(frame["split_votes"].astype(float) / float(total_votes), 6)
    else:
        frame["vote_ratio"] = 0.0
        frame["split_ratio"] = 0.0

    if score_matrix is not None:
        score = np.asarray(score_matrix, dtype=np.float32)
        left_index = frame["left_index"].to_numpy(dtype=int)
        right_index = frame["right_index"].to_numpy(dtype=int)
        frame[pair_score_col] = np.round(score[left_index, right_index].astype(np.float32), 6)
    elif pair_score_col not in frame.columns:
        frame[pair_score_col] = np.nan

    ordered_columns = [
        "left_index",
        "right_index",
        "image_id",
        "neighbor_image_id",
        "base_cluster_left",
        "base_cluster_right",
        "base_cluster_pair",
        "merge_votes",
        "split_votes",
        "total_votes",
        "vote_ratio",
        "split_ratio",
        "same_vote_ratio",
        "vote_balance_score",
        "border_score",
        "ambiguity_score",
        "vote_direction",
        "conflict_methods",
        pair_score_col,
        probability_col,
    ]
    tail_columns = [column for column in frame.columns if column not in ordered_columns]
    return frame.loc[:, ordered_columns + tail_columns].copy()


def assign_ambiguity_components(
    pair_df: pd.DataFrame,
    *,
    min_ambiguity_score: float = 0.6,
    min_conflict_ratio: float = 0.34,
    probability_col: str = "xgb_same_identity_prob",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pair_df.copy().reset_index(drop=True)
    frame["component_id"] = -1
    if frame.empty:
        return frame, pd.DataFrame()

    edge_mask = (
        frame["ambiguity_score"].astype(float).ge(float(min_ambiguity_score))
        & frame["base_conflict_ratio"].astype(float).ge(float(min_conflict_ratio))
    )
    candidate_df = frame[edge_mask].copy()
    if candidate_df.empty:
        return frame, pd.DataFrame()

    adjacency: dict[int, set[int]] = defaultdict(set)
    node_to_image_id: dict[int, str] = {}
    pair_indices_by_node: dict[int, list[int]] = defaultdict(list)
    for row_index, row in candidate_df.iterrows():
        left = int(row.left_index)
        right = int(row.right_index)
        adjacency[left].add(right)
        adjacency[right].add(left)
        node_to_image_id[left] = str(row.image_id)
        node_to_image_id[right] = str(row.neighbor_image_id)
        pair_indices_by_node[left].append(int(row_index))
        pair_indices_by_node[right].append(int(row_index))

    visited: set[int] = set()
    component_rows: list[dict[str, object]] = []
    next_component_id = 0
    for start_node in sorted(adjacency):
        if start_node in visited:
            continue
        queue: deque[int] = deque([int(start_node)])
        component_nodes: list[int] = []
        component_pair_indices: set[int] = set()
        while queue:
            node = int(queue.popleft())
            if node in visited:
                continue
            visited.add(node)
            component_nodes.append(node)
            component_pair_indices.update(pair_indices_by_node.get(node, []))
            for neighbor in sorted(adjacency.get(node, [])):
                if int(neighbor) not in visited:
                    queue.append(int(neighbor))

        component_pair_df = candidate_df.loc[sorted(component_pair_indices)].copy()
        if component_pair_df.empty:
            continue
        frame.loc[component_pair_df.index, "component_id"] = int(next_component_id)
        direction_counts = component_pair_df["vote_direction"].astype(str).value_counts()
        image_ids = [node_to_image_id[index] for index in sorted(component_nodes) if index in node_to_image_id]
        cluster_ids = sorted(
            {
                int(value)
                for value in np.concatenate(
                    [
                        component_pair_df["base_cluster_left"].to_numpy(dtype=int),
                        component_pair_df["base_cluster_right"].to_numpy(dtype=int),
                    ]
                )
            }
        )
        small_bonus = np.clip(1.0 - (len(component_nodes) - 2) / 10.0, 0.0, 1.0)
        best_edge_score = float(component_pair_df["ambiguity_score"].max())
        max_conflict_ratio = float(component_pair_df["base_conflict_ratio"].max())
        component_priority = 0.50 * best_edge_score + 0.35 * max_conflict_ratio + 0.15 * small_bonus
        component_rows.append(
            {
                "component_id": int(next_component_id),
                "image_count": int(len(component_nodes)),
                "pair_count": int(len(component_pair_df)),
                "merge_edge_count": int(direction_counts.get("merge", 0)),
                "split_edge_count": int(direction_counts.get("split", 0)),
                "base_cluster_count": int(len(cluster_ids)),
                "dominant_direction": str(direction_counts.index[0]),
                "mean_ambiguity_score": round(float(component_pair_df["ambiguity_score"].mean()), 6),
                "max_ambiguity_score": round(best_edge_score, 6),
                "max_conflict_ratio": round(max_conflict_ratio, 6),
                "component_priority": round(float(component_priority), 6),
                "mean_pair_probability": round(float(component_pair_df[probability_col].astype(float).mean()), 6),
                "image_indices": "|".join(str(int(index)) for index in sorted(component_nodes)),
                "image_ids": "|".join(str(image_id) for image_id in image_ids),
                "base_cluster_ids": "|".join(str(cluster_id) for cluster_id in cluster_ids),
            }
        )
        next_component_id += 1

    component_df = (
        pd.DataFrame(component_rows)
        .sort_values(["component_priority", "max_ambiguity_score", "pair_count"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    return frame, component_df


def summarize_merge_candidates(
    pair_df: pd.DataFrame,
    *,
    base_labels: np.ndarray,
    probability_col: str = "xgb_same_identity_prob",
    min_merge_votes: int = 2,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    labels = np.asarray(base_labels, dtype=np.int32)
    cluster_sizes = pd.Series(labels).value_counts().to_dict()
    candidate_df = pair_df[
        pair_df["vote_direction"].astype(str).eq("merge")
        & pair_df["merge_votes"].astype(int).ge(int(min_merge_votes))
        & pair_df["base_cluster_left"].astype(int).ne(pair_df["base_cluster_right"].astype(int))
    ].copy()
    if candidate_df.empty:
        return pd.DataFrame()

    candidate_df["left_cluster_id"] = candidate_df[["base_cluster_left", "base_cluster_right"]].min(axis=1).astype(int)
    candidate_df["right_cluster_id"] = candidate_df[["base_cluster_left", "base_cluster_right"]].max(axis=1).astype(int)
    candidate_df["cluster_pair_key"] = (
        candidate_df["left_cluster_id"].astype(str) + "|" + candidate_df["right_cluster_id"].astype(str)
    )

    rows: list[dict[str, object]] = []
    for cluster_pair_key, group_df in candidate_df.groupby("cluster_pair_key"):
        left_cluster_id = int(group_df["left_cluster_id"].iloc[0])
        right_cluster_id = int(group_df["right_cluster_id"].iloc[0])
        component_ids = sorted(set(int(value) for value in group_df["component_id"].tolist() if int(value) >= 0))
        rows.append(
            {
                "cluster_pair_key": str(cluster_pair_key),
                "left_cluster_id": left_cluster_id,
                "right_cluster_id": right_cluster_id,
                "left_cluster_size": int(cluster_sizes.get(left_cluster_id, 0)),
                "right_cluster_size": int(cluster_sizes.get(right_cluster_id, 0)),
                "merged_total_size": int(cluster_sizes.get(left_cluster_id, 0) + cluster_sizes.get(right_cluster_id, 0)),
                "support_pair_count": int(len(group_df)),
                "max_merge_votes": int(group_df["merge_votes"].max()),
                "mean_pair_probability": round(float(group_df[probability_col].astype(float).mean()), 6),
                "max_pair_probability": round(float(group_df[probability_col].astype(float).max()), 6),
                "mean_ambiguity_score": round(float(group_df["ambiguity_score"].mean()), 6),
                "max_ambiguity_score": round(float(group_df["ambiguity_score"].max()), 6),
                "mean_border_score": round(float(group_df["border_score"].mean()), 6),
                "max_conflict_ratio": round(float(group_df["base_conflict_ratio"].max()), 6),
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
                "component_ids": "|".join(str(component_id) for component_id in component_ids),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["max_ambiguity_score", "max_pair_probability", "support_pair_count"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )


def summarize_split_candidates(
    pair_df: pd.DataFrame,
    *,
    base_labels: np.ndarray,
    probability_col: str = "xgb_same_identity_prob",
    min_split_votes: int = 2,
) -> pd.DataFrame:
    if pair_df.empty:
        return pd.DataFrame()
    labels = np.asarray(base_labels, dtype=np.int32)
    cluster_sizes = pd.Series(labels).value_counts().to_dict()
    candidate_df = pair_df[
        pair_df["vote_direction"].astype(str).eq("split")
        & pair_df["split_votes"].astype(int).ge(int(min_split_votes))
        & pair_df["base_cluster_left"].astype(int).eq(pair_df["base_cluster_right"].astype(int))
    ].copy()
    if candidate_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for cluster_id, group_df in candidate_df.groupby("base_cluster_left"):
        image_indices = sorted(
            {
                int(value)
                for value in group_df["left_index"].astype(int).tolist() + group_df["right_index"].astype(int).tolist()
            }
        )
        image_ids = sorted(
            {
                str(value)
                for value in group_df["image_id"].astype(str).tolist() + group_df["neighbor_image_id"].astype(str).tolist()
            }
        )
        component_ids = sorted(set(int(value) for value in group_df["component_id"].tolist() if int(value) >= 0))
        rows.append(
            {
                "base_cluster_id": int(cluster_id),
                "base_cluster_size": int(cluster_sizes.get(int(cluster_id), 0)),
                "ambiguous_image_count": int(len(image_indices)),
                "ambiguous_pair_count": int(len(group_df)),
                "max_split_votes": int(group_df["split_votes"].max()),
                "mean_pair_probability": round(float(group_df[probability_col].astype(float).mean()), 6),
                "max_pair_probability": round(float(group_df[probability_col].astype(float).max()), 6),
                "mean_ambiguity_score": round(float(group_df["ambiguity_score"].mean()), 6),
                "max_ambiguity_score": round(float(group_df["ambiguity_score"].max()), 6),
                "mean_border_score": round(float(group_df["border_score"].mean()), 6),
                "max_conflict_ratio": round(float(group_df["base_conflict_ratio"].max()), 6),
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
                "component_ids": "|".join(str(component_id) for component_id in component_ids),
                "image_indices": "|".join(str(index) for index in image_indices),
                "image_ids": "|".join(image_ids),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["max_ambiguity_score", "base_cluster_size", "ambiguous_pair_count"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
