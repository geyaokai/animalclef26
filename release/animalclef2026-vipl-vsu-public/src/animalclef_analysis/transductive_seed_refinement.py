from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .descriptor_baselines import (
    PATH_COLUMN,
    build_average_linkage,
    cluster_from_linkage,
    summarize_cluster_metrics,
)
from .orb_rerank_baseline import score_matrix_to_distance


@dataclass(frozen=True)
class StableSeedBundle:
    target_df: pd.DataFrame
    anchor_threshold: float
    lower_threshold: float
    upper_threshold: float
    seed_assignment_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    anchor_prediction_df: pd.DataFrame
    anchor_metrics: dict[str, float]


def threshold_tag(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    score = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    np.fill_diagonal(score, 1.0)
    return score.astype(np.float32, copy=False)


def _score_to_distance(score_matrix: np.ndarray, score_space: str) -> np.ndarray:
    if score_space == "unit_interval":
        return score_matrix_to_distance(score_matrix)
    if score_space == "cosine_similarity":
        distance = 1.0 - np.clip(score_matrix, -1.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        return distance.astype(np.float32, copy=False)
    raise ValueError(f"Unsupported score_space: {score_space}")


def cluster_labels_from_score_matrix(score_matrix: np.ndarray, threshold: float, *, score_space: str = "unit_interval") -> np.ndarray:
    distance = _score_to_distance(score_matrix=score_matrix, score_space=score_space)
    linkage_matrix = build_average_linkage(distance)
    return cluster_from_linkage(linkage_matrix, sample_count=len(score_matrix), threshold=float(threshold))


def _cluster_members(labels: np.ndarray, index: int) -> set[int]:
    cluster_id = labels[index]
    return set(np.flatnonzero(labels == cluster_id).tolist())


def _mean_cluster_score(score_matrix: np.ndarray, members: list[int]) -> float:
    if len(members) < 2:
        return 0.0
    block = score_matrix[np.ix_(members, members)].copy()
    upper = block[np.triu_indices_from(block, k=1)]
    return float(np.mean(upper)) if len(upper) else 0.0


def _has_identity_labels(df: pd.DataFrame) -> bool:
    return "identity" in df.columns and df["identity"].fillna("").astype(str).ne("").any()


def _build_prediction_frame(df: pd.DataFrame, labels: np.ndarray, threshold: float) -> pd.DataFrame:
    keep_columns = [column for column in ["image_id", "dataset", "identity", PATH_COLUMN] if column in df.columns]
    frame = df.loc[:, keep_columns].copy().reset_index(drop=True)
    frame["threshold"] = float(threshold)
    frame["pred_cluster_id"] = labels.astype(int)
    return frame


def run_score_threshold_sweep(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    thresholds: list[float],
    *,
    score_space: str = "unit_interval",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_identity = _has_identity_labels(df)
    true_labels = df["identity"].astype(str).to_numpy() if has_identity else None
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for threshold in [float(value) for value in thresholds]:
        pred_labels = cluster_labels_from_score_matrix(score_matrix=score_matrix, threshold=threshold, score_space=score_space)
        counts = pd.Series(pred_labels).value_counts()
        row: dict[str, object] = {
            "dataset": str(df["dataset"].iloc[0]),
            "threshold": float(threshold),
            "samples": int(len(df)),
            "cluster_count": int(counts.size),
            "singleton_cluster_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
        }
        if has_identity and true_labels is not None:
            row.update(summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels))
        prediction_frames.append(_build_prediction_frame(df=df, labels=pred_labels, threshold=threshold))
        rows.append(row)
    sweep_df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def pick_best_threshold_row(sweep_df: pd.DataFrame) -> pd.Series:
    ranking_columns = [column for column in ["ari", "pairwise_f1", "nmi", "threshold"] if column in sweep_df.columns]
    ascending = [False, False, False, True][: len(ranking_columns)]
    return sweep_df.sort_values(ranking_columns, ascending=ascending).iloc[0]


def build_stable_seed_bundle_from_score_matrix(
    *,
    target_df: pd.DataFrame,
    score_matrix: np.ndarray,
    anchor_threshold: float,
    stability_delta: float,
    min_seed_cluster_size: int,
    max_seed_cluster_size: int,
    min_mean_score: float,
    pseudo_prefix: str | None = None,
    score_space: str = "unit_interval",
) -> StableSeedBundle:
    dataset = str(target_df["dataset"].iloc[0])
    prefix = pseudo_prefix or f"pseudo_{dataset}"
    lower_threshold = round(max(0.01, float(anchor_threshold - stability_delta)), 4)
    anchor_threshold = round(float(anchor_threshold), 4)
    upper_threshold = round(min(0.99, float(anchor_threshold + stability_delta)), 4)
    sweep_df, prediction_df = run_score_threshold_sweep(
        df=target_df,
        score_matrix=score_matrix,
        thresholds=sorted({lower_threshold, anchor_threshold, upper_threshold}),
        score_space=score_space,
    )
    pred_by_threshold = {
        round(float(threshold), 4): frame.reset_index(drop=True)
        for threshold, frame in prediction_df.groupby("threshold")
    }
    anchor_pred_df = pred_by_threshold[anchor_threshold].copy()
    lower_pred_df = pred_by_threshold[lower_threshold].copy()
    upper_pred_df = pred_by_threshold[upper_threshold].copy()
    anchor_labels = anchor_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    lower_labels = lower_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    upper_labels = upper_pred_df["pred_cluster_id"].to_numpy(dtype=int)
    has_identity = _has_identity_labels(target_df)
    true_labels = target_df["identity"].astype(str).to_numpy() if has_identity else None

    cluster_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []
    cluster_counter = 0
    for cluster_id in sorted(np.unique(anchor_labels).tolist()):
        members = np.flatnonzero(anchor_labels == cluster_id).tolist()
        member_set = set(members)
        lower_member_set = _cluster_members(lower_labels, members[0])
        upper_member_set = _cluster_members(upper_labels, members[0])
        mean_score = _mean_cluster_score(score_matrix=score_matrix, members=members)
        is_stable = lower_member_set == member_set and upper_member_set == member_set
        size_ok = min_seed_cluster_size <= len(members) <= max_seed_cluster_size
        score_ok = mean_score >= min_mean_score
        accepted = bool(is_stable and size_ok and score_ok and len(members) >= 2)
        pseudo_identity = f"{prefix}_{cluster_counter:04d}" if accepted else ""
        if accepted:
            cluster_counter += 1
        purity = np.nan
        if has_identity and true_labels is not None:
            purity = round(float(pd.Series(true_labels[members]).value_counts(normalize=True).iloc[0]), 6)
        cluster_rows.append(
            {
                "anchor_cluster_id": int(cluster_id),
                "size": int(len(members)),
                "mean_score": round(float(mean_score), 6),
                "stable_lower": bool(lower_member_set == member_set),
                "stable_upper": bool(upper_member_set == member_set),
                "size_ok": bool(size_ok),
                "score_ok": bool(score_ok),
                "accepted_as_seed": bool(accepted),
                "pseudo_identity": pseudo_identity,
                "purity_vs_truth": purity,
            }
        )
        for member_index in members:
            row = {
                "image_id": str(target_df.iloc[member_index]["image_id"]),
                "dataset": dataset,
                "anchor_cluster_id": int(cluster_id),
                "seed_status": "seed" if accepted else "uncertain",
                "pseudo_identity": pseudo_identity,
            }
            if "identity" in target_df.columns:
                row["identity"] = str(target_df.iloc[member_index]["identity"])
            if PATH_COLUMN in target_df.columns:
                row[PATH_COLUMN] = str(target_df.iloc[member_index][PATH_COLUMN])
            assignment_rows.append(row)

    anchor_metrics: dict[str, float] = {}
    if has_identity and true_labels is not None:
        anchor_metrics = summarize_cluster_metrics(true_labels=true_labels, pred_labels=anchor_labels)
    return StableSeedBundle(
        target_df=target_df.reset_index(drop=True).copy(),
        anchor_threshold=float(anchor_threshold),
        lower_threshold=float(lower_threshold),
        upper_threshold=float(upper_threshold),
        seed_assignment_df=pd.DataFrame(assignment_rows),
        cluster_summary_df=pd.DataFrame(cluster_rows).sort_values(
            ["accepted_as_seed", "size"], ascending=[False, False]
        ).reset_index(drop=True),
        threshold_summary_df=sweep_df,
        anchor_prediction_df=anchor_pred_df,
        anchor_metrics=anchor_metrics,
    )


def apply_seed_centroid_smoothing(
    embeddings: np.ndarray,
    seed_assignment_df: pd.DataFrame,
    alpha: float,
    *,
    normalize: bool = True,
) -> np.ndarray:
    smoothed = embeddings.astype(np.float32, copy=True)
    if seed_assignment_df.empty:
        return smoothed
    seed_df = seed_assignment_df[
        seed_assignment_df["seed_status"].astype(str).eq("seed") & seed_assignment_df["pseudo_identity"].astype(str).ne("")
    ].copy()
    if seed_df.empty:
        return smoothed
    image_to_index = {str(image_id): index for index, image_id in enumerate(seed_assignment_df["image_id"].astype(str).tolist())}
    for pseudo_identity, cluster_df in seed_df.groupby("pseudo_identity"):
        del pseudo_identity
        indices = [image_to_index[str(image_id)] for image_id in cluster_df["image_id"].astype(str).tolist()]
        if len(indices) < 2:
            continue
        centroid = np.mean(embeddings[indices], axis=0, dtype=np.float32)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm <= 1e-12:
            continue
        centroid = centroid / centroid_norm
        updated = (1.0 - float(alpha)) * embeddings[indices] + float(alpha) * centroid[None, :]
        if normalize:
            norms = np.linalg.norm(updated, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            updated = updated / norms
        smoothed[indices] = updated.astype(np.float32, copy=False)
    return smoothed


def build_seed_affinity_table(
    score_matrix: np.ndarray,
    seed_assignment_df: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    seed_df = seed_assignment_df[
        seed_assignment_df["seed_status"].astype(str).eq("seed") & seed_assignment_df["pseudo_identity"].astype(str).ne("")
    ].copy()
    if seed_df.empty:
        return np.zeros((len(seed_assignment_df), 0), dtype=np.float32), pd.DataFrame(columns=["pseudo_identity", "size"])
    image_to_index = {str(image_id): index for index, image_id in enumerate(seed_assignment_df["image_id"].astype(str).tolist())}
    cluster_rows: list[dict[str, object]] = []
    affinity_columns: list[np.ndarray] = []
    for pseudo_identity, cluster_df in seed_df.groupby("pseudo_identity"):
        indices = np.array([image_to_index[str(image_id)] for image_id in cluster_df["image_id"].astype(str).tolist()], dtype=int)
        if len(indices) < 2:
            continue
        affinity = score_matrix[:, indices].mean(axis=1).astype(np.float32, copy=False)
        if len(indices) > 1:
            intra = score_matrix[np.ix_(indices, indices)]
            sums = intra.sum(axis=1) - 1.0
            affinity[indices] = (sums / max(1, len(indices) - 1)).astype(np.float32, copy=False)
        affinity_columns.append(affinity)
        cluster_rows.append(
            {
                "pseudo_identity": str(pseudo_identity),
                "size": int(len(indices)),
                "mean_internal_score": round(float(_mean_cluster_score(score_matrix=score_matrix, members=indices.tolist())), 6),
            }
        )
    if not affinity_columns:
        return np.zeros((len(seed_assignment_df), 0), dtype=np.float32), pd.DataFrame(columns=["pseudo_identity", "size"])
    return np.stack(affinity_columns, axis=1).astype(np.float32), pd.DataFrame(cluster_rows)


def apply_seed_center_support(
    score_matrix: np.ndarray,
    seed_assignment_df: pd.DataFrame,
    *,
    support_scale: float,
    min_shared_affinity: float,
    mode: str = "any_seed",
) -> tuple[np.ndarray, dict[str, float]]:
    affinity_matrix, cluster_df = build_seed_affinity_table(score_matrix=score_matrix, seed_assignment_df=seed_assignment_df)
    updated = score_matrix.astype(np.float32, copy=True)
    if affinity_matrix.size == 0:
        return updated, {"seed_clusters": 0, "boosted_pairs": 0, "mean_delta": 0.0, "max_delta": 0.0}
    if mode not in {"any_seed", "same_best_seed"}:
        raise ValueError(f"Unsupported support mode: {mode}")
    if mode == "any_seed":
        support = np.zeros_like(updated)
        for column_index in range(affinity_matrix.shape[1]):
            affinity = affinity_matrix[:, column_index]
            support = np.maximum(support, np.minimum.outer(affinity, affinity))
    else:
        best_cluster = affinity_matrix.argmax(axis=1)
        best_affinity = affinity_matrix.max(axis=1)
        support = np.minimum.outer(best_affinity, best_affinity)
        support = np.where(best_cluster[:, None] == best_cluster[None, :], support, 0.0)
    support = support.astype(np.float32, copy=False)
    if float(min_shared_affinity) > 0:
        support[support < float(min_shared_affinity)] = 0.0
    delta = np.maximum(0.0, support - updated)
    updated = updated + float(support_scale) * delta
    updated = np.clip(updated, 0.0, 1.0)
    np.fill_diagonal(updated, 1.0)
    upper = np.triu(updated - score_matrix, k=1)
    pair_deltas = upper[upper > 0]
    return updated, {
        "seed_clusters": int(len(cluster_df)),
        "boosted_pairs": int(pair_deltas.size),
        "mean_delta": round(float(pair_deltas.mean()) if pair_deltas.size else 0.0, 6),
        "max_delta": round(float(pair_deltas.max()) if pair_deltas.size else 0.0, 6),
    }


def compute_reverse_neighbor_counts(score_matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    if score_matrix.ndim != 2 or score_matrix.shape[0] != score_matrix.shape[1]:
        raise ValueError("score_matrix must be square")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    sample_count = int(score_matrix.shape[0])
    if sample_count <= 1:
        return np.zeros(sample_count, dtype=np.float32), np.zeros((sample_count, 0), dtype=np.int32)
    effective_top_k = int(min(top_k, sample_count - 1))
    order = np.argsort(-score_matrix, axis=1)[:, 1 : effective_top_k + 1].astype(np.int32, copy=False)
    reverse_counts = np.zeros(sample_count, dtype=np.int32)
    for row_neighbors in order:
        reverse_counts[row_neighbors] += 1
    return reverse_counts.astype(np.float32), order


def apply_reverse_neighbor_penalty(
    score_matrix: np.ndarray,
    *,
    top_k: int,
    penalty_scale: float,
    positive_only: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    reverse_counts, neighbor_index = compute_reverse_neighbor_counts(score_matrix=score_matrix, top_k=top_k)
    if reverse_counts.size == 0:
        return score_matrix.astype(np.float32, copy=True), {
            "effective_top_k": 0,
            "mean_reverse_neighbor_count": 0.0,
            "std_reverse_neighbor_count": 0.0,
            "p95_reverse_neighbor_count": 0.0,
            "hub_image_ratio": 0.0,
        }
    mean_count = float(reverse_counts.mean())
    std_count = float(reverse_counts.std())
    if std_count <= 1e-12:
        normalized = np.zeros_like(reverse_counts, dtype=np.float32)
    else:
        normalized = (reverse_counts - mean_count) / std_count
    if positive_only:
        normalized = np.maximum(normalized, 0.0)
    penalty = float(penalty_scale) * (normalized[:, None] + normalized[None, :])
    updated = np.clip(score_matrix.astype(np.float32, copy=False) - penalty.astype(np.float32, copy=False), -1.0, 1.0)
    np.fill_diagonal(updated, 1.0)
    hub_threshold = float(min(max(1, top_k), score_matrix.shape[0] - 1))
    return updated, {
        "effective_top_k": int(neighbor_index.shape[1]),
        "mean_reverse_neighbor_count": round(mean_count, 6),
        "std_reverse_neighbor_count": round(std_count, 6),
        "p95_reverse_neighbor_count": round(float(np.quantile(reverse_counts, 0.95)), 6),
        "hub_image_ratio": round(float((reverse_counts >= hub_threshold).mean()), 6),
    }
