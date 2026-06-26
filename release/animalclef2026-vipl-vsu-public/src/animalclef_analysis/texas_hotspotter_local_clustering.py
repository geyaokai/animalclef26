from __future__ import annotations

import json
import math
import os
import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .descriptor_baselines import build_submission
from .graph_clustering import (
    GRAPH_METHOD_CHINESE_WHISPERS,
    GRAPH_METHOD_CONNECTED_COMPONENTS,
    cluster_labels_from_score_graph,
)
from .local_matching import HotspotterConfig
from .orb_rerank_baseline import recall_at_k_from_score_matrix
from .texas_hotspotter_probe import DEFAULT_TEXAS_VIEW_MANIFEST_PATH, run_texas_hotspotter_probe
from .texas_local_pairwise import build_topk_indices_from_score_matrix
from .texas_selftrain import (
    build_candidate_index_pairs,
    compute_pair_keep_ratio,
    compute_threshold_proxy_score,
)
from .texas_unsupervised import TEXAS_DATASET, mean_topk_neighbor_overlap, pair_agreement_score, summarize_cluster_labels

try:  # pragma: no cover
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
except ModuleNotFoundError:  # pragma: no cover
    AgglomerativeClustering = None
    DBSCAN = None


DEFAULT_TEXAS_EXPERIMENT_DIR = Path("artifacts/training/experiments/ft_texas_miew_tcuwarmup_trusted_views_v1")
DEFAULT_BASE_PREDICTIONS_PATH = Path(
    "artifacts/submissions/kaggle_variant_texas_tcuwarmup_trusted_views_v1/tables/test_predictions_v1.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_hotspotter_local_clustering_v1")
DEFAULT_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_texas_hotspotter_local_v1")
DEFAULT_BEST_PUBLIC_SCORE = 0.54053


@dataclass(frozen=True)
class TexasLocalCandidate:
    name: str
    method: str
    score_matrix_name: str
    param_key: str
    summary_row: dict[str, object]
    prediction_df: pd.DataFrame


@dataclass(frozen=True)
class TexasProxyBundle:
    metadata_df: pd.DataFrame
    candidate_pair_df: pd.DataFrame
    teacher_anchor_labels: np.ndarray
    teacher_topk_indices: np.ndarray
    teacher_score_matrix: np.ndarray


def _prediction_table_stem(score_matrix_name: str, method: str) -> str:
    return f"{score_matrix_name}__{method}_predictions_v1.csv"


def _candidate_partition_signature(prediction_df: pd.DataFrame) -> str:
    ordered = prediction_df.loc[:, ["image_id", "pred_cluster_id"]].copy()
    ordered["image_id"] = ordered["image_id"].astype(str)
    ordered["pred_cluster_id"] = ordered["pred_cluster_id"].astype(int)
    ordered = ordered.sort_values("image_id").reset_index(drop=True)
    payload = "|".join(
        f"{image_id}:{cluster_id}"
        for image_id, cluster_id in zip(ordered["image_id"].tolist(), ordered["pred_cluster_id"].tolist())
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _path_ref(repo_root: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=repo_root.resolve()).replace("\\", "/")


def _resolve_input_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _require_sklearn() -> None:
    if AgglomerativeClustering is None or DBSCAN is None:
        raise ModuleNotFoundError("scikit-learn is required for Texas local clustering.")


def load_texas_proxy_bundle(*, repo_root: Path, experiment_dir: Path, top_k: int = 8) -> TexasProxyBundle:
    resolved_experiment_dir = _resolve_input_path(repo_root, experiment_dir)
    route_df = pd.read_csv(resolved_experiment_dir / "tables" / "test_predictions_best_v1.csv").copy()
    route_df["image_id"] = route_df["image_id"].astype(str)
    route_df["dataset"] = route_df["dataset"].astype(str)
    route_df = route_df[route_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if route_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {resolved_experiment_dir / 'tables' / 'test_predictions_best_v1.csv'}")

    pseudo_df = pd.read_csv(resolved_experiment_dir / "tables" / "pseudo_assignments_v1.csv").copy()
    pseudo_df["image_id"] = pseudo_df["image_id"].astype(str)
    pseudo_df["dataset"] = pseudo_df["dataset"].astype(str)
    pseudo_df = pseudo_df[pseudo_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if pseudo_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in pseudo_assignments_v1.csv")
    pseudo_lookup = pseudo_df.set_index("image_id", drop=False)
    missing = [image_id for image_id in route_df["image_id"].tolist() if image_id not in pseudo_lookup.index]
    if missing:
        raise ValueError(f"pseudo_assignments missing image_ids, examples: {missing[:5]}")
    metadata_df = pseudo_lookup.loc[route_df["image_id"].tolist()].reset_index(drop=True).copy()
    metadata_df["is_seed"] = metadata_df.get("is_seed", False)
    metadata_df["is_seed"] = metadata_df["is_seed"].fillna(False).astype(bool)
    metadata_df["pseudo_label_index"] = metadata_df.get("pseudo_label_index", -1)
    metadata_df["pseudo_label_index"] = metadata_df["pseudo_label_index"].fillna(-1).astype(int)

    candidate_pair_df = pd.read_csv(resolved_experiment_dir / "tables" / "candidate_pairs_v1.csv").copy()
    candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
    candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)

    embedding_meta_df = pd.read_csv(resolved_experiment_dir / "embeddings" / "test_metadata.csv").copy()
    embedding_meta_df["image_id"] = embedding_meta_df["image_id"].astype(str)
    embedding_meta_df["dataset"] = embedding_meta_df["dataset"].astype(str)
    embedding_meta_df = embedding_meta_df[embedding_meta_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    embeddings = np.load(resolved_experiment_dir / "embeddings" / "test_embeddings.npy").astype(np.float32)
    texas_mask = (pd.read_csv(resolved_experiment_dir / "embeddings" / "test_metadata.csv")["dataset"].astype(str) == TEXAS_DATASET).to_numpy()
    texas_embeddings = embeddings[texas_mask]
    embed_lookup = {image_id: index for index, image_id in enumerate(embedding_meta_df["image_id"].tolist())}
    reorder_index = np.asarray([embed_lookup[image_id] for image_id in route_df["image_id"].tolist()], dtype=np.int32)
    texas_embeddings = texas_embeddings[reorder_index]
    teacher_score_matrix = np.clip(texas_embeddings @ texas_embeddings.T, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(teacher_score_matrix, 1.0)
    teacher_topk_indices = build_topk_indices_from_score_matrix(teacher_score_matrix, top_k=top_k)
    teacher_anchor_labels = route_df["pred_cluster_id"].to_numpy(dtype=np.int32)
    return TexasProxyBundle(
        metadata_df=metadata_df,
        candidate_pair_df=candidate_pair_df,
        teacher_anchor_labels=teacher_anchor_labels,
        teacher_topk_indices=teacher_topk_indices,
        teacher_score_matrix=teacher_score_matrix,
    )


def _load_probe_tables(probe_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_path = probe_dir / "tables" / "query_rankings_v1.csv"
    pair_path = probe_dir / "tables" / "test_pair_local_scores_v1.csv"
    if not ranking_path.exists() or not pair_path.exists():
        raise FileNotFoundError(f"Probe tables missing under {probe_dir}")
    ranking_df = pd.read_csv(ranking_path).copy()
    pair_df = pd.read_csv(pair_path).copy()
    ranking_df["image_id"] = ranking_df["image_id"].astype(str)
    ranking_df["neighbor_image_id"] = ranking_df["neighbor_image_id"].astype(str)
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    return ranking_df, pair_df


def _rank_confidence(rank: int, max_rank: int) -> float:
    if max_rank <= 1:
        return 1.0
    return max(0.0, 1.0 - (float(rank) - 1.0) / float(max_rank - 1))


def build_hotspotter_affinity_matrices(
    ranking_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    top_k: int,
    bidirectional_penalty: float = 0.55,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    image_to_index = {
        str(image_id): int(index)
        for index, image_id in enumerate(reference_df["image_id"].astype(str).tolist())
    }
    working = ranking_df.copy().reset_index(drop=True)
    working["left_index"] = working["image_id"].astype(str).map(image_to_index)
    working["right_index"] = working["neighbor_image_id"].astype(str).map(image_to_index)
    working = working.dropna(subset=["left_index", "right_index"]).copy()
    working["left_index"] = working["left_index"].astype(int)
    working["right_index"] = working["right_index"].astype(int)
    working = working[working["left_index"] != working["right_index"]].copy().reset_index(drop=True)
    working["local_score"] = pd.to_numeric(working["local_score"], errors="coerce").fillna(0.0)
    working["local_prescore"] = pd.to_numeric(working.get("local_prescore", 0.0), errors="coerce").fillna(0.0)
    working["inliers"] = pd.to_numeric(working.get("inliers", 0), errors="coerce").fillna(0).astype(int)
    working["good_matches"] = pd.to_numeric(working.get("good_matches", 0), errors="coerce").fillna(0).astype(int)
    working["rank"] = pd.to_numeric(working["rank"], errors="coerce").fillna(top_k).astype(int)
    positive_score = working.loc[working["local_score"] > 0, "local_score"]
    positive_prescore = working.loc[working["local_prescore"] > 0, "local_prescore"]
    positive_inliers = working.loc[working["inliers"] > 0, "inliers"]
    score_scale = float(positive_score.quantile(0.95)) if not positive_score.empty else 1.0
    prescore_scale = float(positive_prescore.quantile(0.95)) if not positive_prescore.empty else 1.0
    inlier_scale = float(positive_inliers.quantile(0.95)) if not positive_inliers.empty else 1.0
    score_scale = max(score_scale, 1e-6)
    prescore_scale = max(prescore_scale, 1e-6)
    inlier_scale = max(inlier_scale, 1.0)

    working["post_score_norm"] = np.clip(working["local_score"].to_numpy(dtype=np.float32) / score_scale, 0.0, 1.0)
    working["prescore_norm"] = np.clip(working["local_prescore"].to_numpy(dtype=np.float32) / prescore_scale, 0.0, 1.0)
    working["inlier_norm"] = np.clip(working["inliers"].to_numpy(dtype=np.float32) / inlier_scale, 0.0, 1.0)
    working["rank_norm"] = working["rank"].map(lambda value: _rank_confidence(int(value), int(top_k))).astype(np.float32)
    # Conservative directed confidence: actual verified score dominates, inliers and rank are tie-breakers.
    working["directed_post_conf"] = (
        0.60 * working["post_score_norm"].to_numpy(dtype=np.float32)
        + 0.25 * working["inlier_norm"].to_numpy(dtype=np.float32)
        + 0.15 * working["rank_norm"].to_numpy(dtype=np.float32)
    )
    working["directed_blend_conf"] = (
        0.45 * working["post_score_norm"].to_numpy(dtype=np.float32)
        + 0.25 * working["prescore_norm"].to_numpy(dtype=np.float32)
        + 0.20 * working["inlier_norm"].to_numpy(dtype=np.float32)
        + 0.10 * working["rank_norm"].to_numpy(dtype=np.float32)
    )
    working["pair_key"] = working.apply(
        lambda row: f"{min(int(row['left_index']), int(row['right_index']))}|{max(int(row['left_index']), int(row['right_index']))}",
        axis=1,
    )

    pair_rows: list[dict[str, object]] = []
    sample_count = int(len(reference_df))
    post_matrix = np.zeros((sample_count, sample_count), dtype=np.float32)
    blend_matrix = np.zeros((sample_count, sample_count), dtype=np.float32)
    rank_matrix = np.zeros((sample_count, sample_count), dtype=np.float32)
    np.fill_diagonal(post_matrix, 1.0)
    np.fill_diagonal(blend_matrix, 1.0)
    np.fill_diagonal(rank_matrix, 1.0)

    for _, group_df in working.groupby("pair_key", sort=False):
        group_df = group_df.sort_values("rank", ascending=True).reset_index(drop=True)
        left_index = int(min(group_df["left_index"].min(), group_df["right_index"].min()))
        right_index = int(max(group_df["left_index"].max(), group_df["right_index"].max()))
        support_count = int(len(group_df))
        if support_count >= 2:
            post_value = float(group_df["directed_post_conf"].min())
            blend_value = float(group_df["directed_blend_conf"].min())
            rank_value = float(group_df["rank_norm"].min())
        else:
            post_value = float(group_df["directed_post_conf"].iloc[0]) * float(bidirectional_penalty)
            blend_value = float(group_df["directed_blend_conf"].iloc[0]) * float(bidirectional_penalty)
            rank_value = float(group_df["rank_norm"].iloc[0]) * float(bidirectional_penalty)
        post_matrix[left_index, right_index] = post_matrix[right_index, left_index] = float(post_value)
        blend_matrix[left_index, right_index] = blend_matrix[right_index, left_index] = float(blend_value)
        rank_matrix[left_index, right_index] = rank_matrix[right_index, left_index] = float(rank_value)
        pair_rows.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "image_id": str(reference_df.iloc[left_index]["image_id"]),
                "neighbor_image_id": str(reference_df.iloc[right_index]["image_id"]),
                "support_count": int(support_count),
                "post_affinity": round(float(post_value), 6),
                "blend_affinity": round(float(blend_value), 6),
                "rank_affinity": round(float(rank_value), 6),
                "best_local_score": round(float(group_df["local_score"].max()), 6),
                "best_inliers": int(group_df["inliers"].max()),
                "best_rank": int(group_df["rank"].min()),
            }
        )
    pair_summary_df = pd.DataFrame(pair_rows).sort_values(
        ["blend_affinity", "post_affinity", "best_inliers"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return {
        "post_min": post_matrix,
        "blend_min": blend_matrix,
        "rank_only": rank_matrix,
    }, pair_summary_df


def _candidate_thresholds_from_matrix(score_matrix: np.ndarray) -> list[float]:
    upper = score_matrix[np.triu_indices(len(score_matrix), k=1)]
    positive = upper[upper > 0]
    if positive.size == 0:
        return [0.5]
    quantiles = [0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.96]
    values = sorted({round(float(np.quantile(positive, q)), 4) for q in quantiles if positive.size > 0})
    filtered = [value for value in values if 0.02 <= float(value) <= 0.98]
    return filtered if filtered else [round(float(np.median(positive)), 4)]


def _labels_from_agglomerative_complete(distance_matrix: np.ndarray, *, distance_threshold: float) -> np.ndarray:
    _require_sklearn()
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="complete",
        distance_threshold=float(distance_threshold),
        compute_full_tree=True,
    )
    labels = model.fit_predict(distance_matrix)
    _, normalized = np.unique(labels.astype(int), return_inverse=True)
    return normalized.astype(np.int32)


def _labels_from_dbscan(distance_matrix: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    _require_sklearn()
    model = DBSCAN(metric="precomputed", eps=float(eps), min_samples=int(min_samples), n_jobs=-1)
    labels = model.fit_predict(distance_matrix).astype(np.int32)
    if (labels == -1).any():
        next_label = int(labels[labels >= 0].max()) + 1 if (labels >= 0).any() else 0
        for index in np.flatnonzero(labels == -1):
            labels[index] = next_label
            next_label += 1
    _, normalized = np.unique(labels.astype(int), return_inverse=True)
    return normalized.astype(np.int32)


def _evaluate_labels_common(
    *,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
    candidate_pair_df: pd.DataFrame,
    teacher_anchor_labels: np.ndarray | None,
    teacher_topk_indices: np.ndarray | None,
    score_matrix: np.ndarray,
    top_k_overlap: int,
) -> dict[str, object]:
    seed_mask = metadata_df["is_seed"].fillna(False).astype(bool).to_numpy()
    seed_labels = metadata_df.loc[seed_mask, "pseudo_label_index"].to_numpy(dtype=int) if seed_mask.any() else np.asarray([], dtype=int)
    all_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=False)
    mutual_candidate_pairs = build_candidate_index_pairs(metadata_df, candidate_pair_df, mutual_topk_only=True)
    student_topk_indices = build_topk_indices_from_score_matrix(score_matrix, top_k=top_k_overlap)
    student_teacher_topk_overlap = (
        mean_topk_neighbor_overlap(student_topk_indices, teacher_topk_indices)
        if teacher_topk_indices is not None and student_topk_indices.shape == teacher_topk_indices.shape
        else np.nan
    )
    stats = summarize_cluster_labels(labels)
    target_clusters = int(summarize_cluster_labels(teacher_anchor_labels)["clusters"]) if teacher_anchor_labels is not None else np.nan
    row: dict[str, object] = {
        **stats,
        "samples": int(len(metadata_df)),
        "seed_pair_agreement": pair_agreement_score(labels[seed_mask], seed_labels) if seed_mask.any() else np.nan,
        "seed_recall_at_1": (
            recall_at_k_from_score_matrix(score_matrix[np.ix_(seed_mask, seed_mask)], seed_labels, k=1)
            if seed_mask.any()
            else np.nan
        ),
        "candidate_pair_keep_ratio": compute_pair_keep_ratio(labels, all_candidate_pairs),
        "mutual_topk_pair_keep_ratio": compute_pair_keep_ratio(labels, mutual_candidate_pairs),
        "student_teacher_topk_overlap": student_teacher_topk_overlap,
        "teacher_anchor_clusters": target_clusters,
        "cluster_delta_vs_teacher_anchor": abs(int(stats["clusters"]) - int(target_clusters)) if teacher_anchor_labels is not None else np.nan,
        "pair_agreement_vs_teacher_anchor": pair_agreement_score(labels, teacher_anchor_labels) if teacher_anchor_labels is not None else np.nan,
    }
    row["pair_agreement_vs_student_anchor"] = row["pair_agreement_vs_teacher_anchor"]
    row["proxy_score"] = compute_threshold_proxy_score(pd.Series(row))
    return row


def evaluate_local_clustering_candidates(
    *,
    metadata_df: pd.DataFrame,
    score_matrices: dict[str, np.ndarray],
    candidate_pair_df: pd.DataFrame,
    teacher_anchor_labels: np.ndarray | None,
    teacher_topk_indices: np.ndarray | None,
    top_k_overlap: int = 8,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[TexasLocalCandidate]]:
    summary_rows: list[dict[str, object]] = []
    prediction_frames: dict[str, list[pd.DataFrame]] = {}
    candidates: list[TexasLocalCandidate] = []

    for matrix_name, score_matrix in score_matrices.items():
        graph_thresholds = _candidate_thresholds_from_matrix(score_matrix)
        distance_thresholds = sorted({round(1.0 - float(value), 4) for value in graph_thresholds})
        distance_matrix = 1.0 - np.clip(score_matrix.astype(np.float32, copy=False), 0.0, 1.0)
        np.fill_diagonal(distance_matrix, 0.0)

        for method, graph_top_k, mutual_flag in [
            (GRAPH_METHOD_CONNECTED_COMPONENTS, 1, True),
            (GRAPH_METHOD_CONNECTED_COMPONENTS, 2, True),
            (GRAPH_METHOD_CONNECTED_COMPONENTS, 3, True),
            (GRAPH_METHOD_CONNECTED_COMPONENTS, 5, True),
            (GRAPH_METHOD_CHINESE_WHISPERS, 5, False),
            (GRAPH_METHOD_CHINESE_WHISPERS, 10, False),
        ]:
            for threshold in graph_thresholds:
                labels = cluster_labels_from_score_graph(
                    score_matrix=score_matrix,
                    threshold=float(threshold),
                    method=method,
                    top_k=int(graph_top_k),
                    mutual_top_k=bool(mutual_flag),
                    iterations=30,
                    seed=42,
                )
                row = _evaluate_labels_common(
                    labels=labels,
                    metadata_df=metadata_df,
                    candidate_pair_df=candidate_pair_df,
                    teacher_anchor_labels=teacher_anchor_labels,
                    teacher_topk_indices=teacher_topk_indices,
                    score_matrix=score_matrix,
                    top_k_overlap=top_k_overlap,
                )
                row.update(
                    {
                        "score_matrix_name": str(matrix_name),
                        "method": str(method),
                        "param_key": f"thr={float(threshold):.4f}|topk={int(graph_top_k)}|mutual={int(bool(mutual_flag))}",
                        "threshold": float(threshold),
                        "graph_top_k": int(graph_top_k),
                        "mutual_top_k": bool(mutual_flag),
                        "distance_threshold": np.nan,
                        "dbscan_min_samples": np.nan,
                    }
                )
                pred_df = metadata_df.copy().reset_index(drop=True)
                pred_df["pred_cluster_id"] = labels.astype(int)
                pred_df["cluster_label"] = [f"cluster_{TEXAS_DATASET}_{int(label)}" for label in labels]
                pred_df["route_name"] = f"hotspotter_local_{matrix_name}_{method}"
                pred_df["chosen_threshold"] = float(threshold)
                pred_df["embedding_dim"] = 0
                pred_df["rerank_enabled"] = True
                pred_df["local_weight"] = 1.0
                pred_df["clustering_method"] = str(method)
                pred_df["score_matrix_name"] = str(matrix_name)
                pred_df["graph_top_k"] = int(graph_top_k)
                pred_df["mutual_top_k"] = bool(mutual_flag)
                prediction_frames.setdefault(f"{matrix_name}:{method}", []).append(pred_df)
                candidate_name = f"{matrix_name}_{method}_thr{float(threshold):.4f}_topk{int(graph_top_k)}_m{int(bool(mutual_flag))}"
                pred_df["param_key"] = str(row["param_key"])
                pred_df["candidate_name"] = candidate_name
                candidates.append(
                    TexasLocalCandidate(
                        name=candidate_name,
                        method=str(method),
                        score_matrix_name=str(matrix_name),
                        param_key=str(row["param_key"]),
                        summary_row=row.copy(),
                        prediction_df=pred_df.copy(),
                    )
                )
                summary_rows.append(row)

        for distance_threshold in distance_thresholds:
            labels = _labels_from_agglomerative_complete(distance_matrix, distance_threshold=float(distance_threshold))
            row = _evaluate_labels_common(
                labels=labels,
                metadata_df=metadata_df,
                candidate_pair_df=candidate_pair_df,
                teacher_anchor_labels=teacher_anchor_labels,
                teacher_topk_indices=teacher_topk_indices,
                score_matrix=score_matrix,
                top_k_overlap=top_k_overlap,
            )
            row.update(
                {
                    "score_matrix_name": str(matrix_name),
                    "method": "agglomerative_complete",
                    "param_key": f"dist={float(distance_threshold):.4f}",
                    "threshold": np.nan,
                    "graph_top_k": -1,
                    "mutual_top_k": False,
                    "distance_threshold": float(distance_threshold),
                    "dbscan_min_samples": np.nan,
                }
            )
            pred_df = metadata_df.copy().reset_index(drop=True)
            pred_df["pred_cluster_id"] = labels.astype(int)
            pred_df["cluster_label"] = [f"cluster_{TEXAS_DATASET}_{int(label)}" for label in labels]
            pred_df["route_name"] = f"hotspotter_local_{matrix_name}_agglomerative_complete"
            pred_df["chosen_threshold"] = float(distance_threshold)
            pred_df["embedding_dim"] = 0
            pred_df["rerank_enabled"] = True
            pred_df["local_weight"] = 1.0
            pred_df["clustering_method"] = "agglomerative_complete"
            pred_df["score_matrix_name"] = str(matrix_name)
            pred_df["graph_top_k"] = -1
            pred_df["mutual_top_k"] = False
            pred_df["param_key"] = str(row["param_key"])
            pred_df["candidate_name"] = f"{matrix_name}_agglomerative_complete_dist{float(distance_threshold):.4f}"
            prediction_frames.setdefault(f"{matrix_name}:agglomerative_complete", []).append(pred_df)
            candidates.append(
                TexasLocalCandidate(
                    name=f"{matrix_name}_agglomerative_complete_dist{float(distance_threshold):.4f}",
                    method="agglomerative_complete",
                    score_matrix_name=str(matrix_name),
                    param_key=str(row["param_key"]),
                    summary_row=row.copy(),
                    prediction_df=pred_df.copy(),
                )
            )
            summary_rows.append(row)

        for distance_threshold in distance_thresholds[: min(4, len(distance_thresholds))]:
            for min_samples in [2, 3]:
                labels = _labels_from_dbscan(
                    distance_matrix,
                    eps=float(distance_threshold),
                    min_samples=int(min_samples),
                )
                row = _evaluate_labels_common(
                    labels=labels,
                    metadata_df=metadata_df,
                    candidate_pair_df=candidate_pair_df,
                    teacher_anchor_labels=teacher_anchor_labels,
                    teacher_topk_indices=teacher_topk_indices,
                    score_matrix=score_matrix,
                    top_k_overlap=top_k_overlap,
                )
                row.update(
                    {
                        "score_matrix_name": str(matrix_name),
                        "method": "dbscan",
                        "param_key": f"eps={float(distance_threshold):.4f}|min_samples={int(min_samples)}",
                        "threshold": np.nan,
                        "graph_top_k": -1,
                        "mutual_top_k": False,
                        "distance_threshold": float(distance_threshold),
                        "dbscan_min_samples": int(min_samples),
                    }
                )
                pred_df = metadata_df.copy().reset_index(drop=True)
                pred_df["pred_cluster_id"] = labels.astype(int)
                pred_df["cluster_label"] = [f"cluster_{TEXAS_DATASET}_{int(label)}" for label in labels]
                pred_df["route_name"] = f"hotspotter_local_{matrix_name}_dbscan"
                pred_df["chosen_threshold"] = float(distance_threshold)
                pred_df["embedding_dim"] = 0
                pred_df["rerank_enabled"] = True
                pred_df["local_weight"] = 1.0
                pred_df["clustering_method"] = "dbscan"
                pred_df["score_matrix_name"] = str(matrix_name)
                pred_df["graph_top_k"] = -1
                pred_df["mutual_top_k"] = False
                pred_df["param_key"] = str(row["param_key"])
                pred_df["candidate_name"] = f"{matrix_name}_dbscan_eps{float(distance_threshold):.4f}_ms{int(min_samples)}"
                prediction_frames.setdefault(f"{matrix_name}:dbscan", []).append(pred_df)
                candidates.append(
                    TexasLocalCandidate(
                        name=f"{matrix_name}_dbscan_eps{float(distance_threshold):.4f}_ms{int(min_samples)}",
                        method="dbscan",
                        score_matrix_name=str(matrix_name),
                        param_key=str(row["param_key"]),
                        summary_row=row.copy(),
                        prediction_df=pred_df.copy(),
                    )
                )
                summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["proxy_score", "cluster_delta_vs_teacher_anchor", "largest_cluster_size"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    prediction_df_by_name = {
        name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, frames in prediction_frames.items()
    }
    candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item.summary_row.get("proxy_score", -np.inf)),
            float(item.summary_row.get("cluster_delta_vs_teacher_anchor", np.inf)),
            float(item.summary_row.get("largest_cluster_size", np.inf)),
        ),
    )
    return summary_df, prediction_df_by_name, candidates


def load_saved_texas_local_candidate(
    *,
    analysis_dir: Path,
    candidate_row: pd.Series | dict[str, object],
) -> TexasLocalCandidate:
    candidate_series = pd.Series(candidate_row).copy()
    resolved_analysis_dir = analysis_dir.resolve()
    tables_dir = resolved_analysis_dir / "tables"
    score_matrix_name = str(candidate_series["score_matrix_name"])
    method = str(candidate_series["method"])
    prediction_path = tables_dir / _prediction_table_stem(score_matrix_name=score_matrix_name, method=method)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction table not found for candidate: {prediction_path}")

    prediction_df = pd.read_csv(prediction_path).copy()
    prediction_df["image_id"] = prediction_df["image_id"].astype(str)
    prediction_df["score_matrix_name"] = prediction_df["score_matrix_name"].astype(str)
    prediction_df["clustering_method"] = prediction_df["clustering_method"].astype(str)
    if "param_key" in prediction_df.columns and str(candidate_series.get("param_key", "")).strip():
        filtered_df = prediction_df[
            prediction_df["param_key"].astype(str) == str(candidate_series.get("param_key", ""))
        ].copy()
    else:
        filtered_df = prediction_df[
            (prediction_df["score_matrix_name"] == score_matrix_name)
            & (prediction_df["clustering_method"] == method)
        ].copy()

    if (
        "param_key" not in prediction_df.columns
        and pd.notna(candidate_series.get("threshold", np.nan))
    ):
        filtered_df = filtered_df[
            filtered_df["chosen_threshold"].astype(float).round(4) == round(float(candidate_series["threshold"]), 4)
        ].copy()
    elif (
        "param_key" not in prediction_df.columns
        and pd.notna(candidate_series.get("distance_threshold", np.nan))
    ):
        filtered_df = filtered_df[
            filtered_df["chosen_threshold"].astype(float).round(4)
            == round(float(candidate_series["distance_threshold"]), 4)
        ].copy()

    if (
        "param_key" not in prediction_df.columns
        and "graph_top_k" in filtered_df.columns
        and pd.notna(candidate_series.get("graph_top_k", np.nan))
    ):
        filtered_df = filtered_df[filtered_df["graph_top_k"].astype(int) == int(candidate_series["graph_top_k"])].copy()
    if (
        "param_key" not in prediction_df.columns
        and "mutual_top_k" in filtered_df.columns
        and pd.notna(candidate_series.get("mutual_top_k", np.nan))
    ):
        filtered_df = filtered_df[
            filtered_df["mutual_top_k"].astype(bool) == bool(candidate_series["mutual_top_k"])
        ].copy()
    if (
        "param_key" not in prediction_df.columns
        and "dbscan_min_samples" in filtered_df.columns
        and pd.notna(candidate_series.get("dbscan_min_samples", np.nan))
    ):
        filtered_df = filtered_df[
            filtered_df["dbscan_min_samples"].astype(float).astype(int) == int(candidate_series["dbscan_min_samples"])
        ].copy()

    if filtered_df.empty:
        raise ValueError(
            "Candidate filter produced no rows for "
            f"score_matrix_name={score_matrix_name} method={method} param_key={candidate_series.get('param_key', '')}"
        )

    duplicate_mask = filtered_df.duplicated(subset=["image_id"], keep=False)
    if duplicate_mask.any():
        duplicate_preview = (
            filtered_df.loc[duplicate_mask, ["image_id", "chosen_threshold", "graph_top_k", "mutual_top_k"]]
            .sort_values(["image_id", "chosen_threshold", "graph_top_k"])
            .head(12)
            .to_dict(orient="records")
        )
        raise ValueError(
            "Filtered candidate still has duplicate image_id rows. "
            f"Use the full candidate key from candidate_summary_v1.csv. Examples: {duplicate_preview}"
        )

    candidate_name = (
        f"{score_matrix_name}_{method}_{str(candidate_series.get('param_key', 'candidate')).replace('|', '_')}"
    )
    return TexasLocalCandidate(
        name=candidate_name,
        method=method,
        score_matrix_name=score_matrix_name,
        param_key=str(candidate_series.get("param_key", "")),
        summary_row=candidate_series.to_dict(),
        prediction_df=filtered_df.reset_index(drop=True),
    )


def load_distinct_saved_texas_local_candidates(
    *,
    analysis_dir: Path,
    top_n: int | None = None,
) -> list[TexasLocalCandidate]:
    resolved_analysis_dir = analysis_dir.resolve()
    summary_path = resolved_analysis_dir / "tables" / "candidate_summary_v1.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Candidate summary not found: {summary_path}")
    summary_df = pd.read_csv(summary_path).copy()

    distinct_candidates: list[TexasLocalCandidate] = []
    seen_signatures: set[str] = set()
    for _, row in summary_df.iterrows():
        candidate = load_saved_texas_local_candidate(
            analysis_dir=resolved_analysis_dir,
            candidate_row=row,
        )
        signature = _candidate_partition_signature(candidate.prediction_df)
        if signature in seen_signatures:
            continue
        candidate.summary_row["partition_signature"] = signature
        distinct_candidates.append(candidate)
        seen_signatures.add(signature)
        if top_n is not None and len(distinct_candidates) >= int(top_n):
            break
    return distinct_candidates


def build_texas_override_submission(
    *,
    repo_root: Path,
    base_predictions_path: Path,
    texas_pred_df: pd.DataFrame,
    output_dir: Path,
    route_name: str,
    route_summary: dict[str, object],
) -> dict[str, Path]:
    resolved_base_predictions_path = _resolve_input_path(repo_root, base_predictions_path)
    resolved_output_dir = _resolve_input_path(repo_root, output_dir)
    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    runtime_dir = resolved_output_dir / "runtime"
    for path in [resolved_output_dir, tables_dir, reports_dir, runtime_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(resolved_base_predictions_path).copy()
    base_df["image_id"] = base_df["image_id"].astype(str)
    base_df["dataset"] = base_df["dataset"].astype(str)
    texas_df = texas_pred_df.copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(str)
    texas_df["dataset"] = TEXAS_DATASET
    texas_df["route_name"] = str(route_name)
    if "embedding_dim" not in texas_df.columns:
        texas_df["embedding_dim"] = 0
    if "rerank_enabled" not in texas_df.columns:
        texas_df["rerank_enabled"] = True
    if "local_weight" not in texas_df.columns:
        texas_df["local_weight"] = 1.0
    if texas_df["image_id"].duplicated().any():
        duplicate_ids = texas_df.loc[texas_df["image_id"].duplicated(keep=False), "image_id"].astype(str).tolist()
        raise ValueError(
            "Texas override prediction frame must be one row per image_id before submission export. "
            f"Duplicate examples: {duplicate_ids[:10]}"
        )

    merged_df = pd.concat([base_df[base_df["dataset"] != TEXAS_DATASET].copy(), texas_df], ignore_index=True)
    predictions_path = tables_dir / "test_predictions_v1.csv"
    merged_df.to_csv(predictions_path, index=False)
    submission_path = resolved_output_dir / "submission.csv"
    build_submission(
        test_pred_df=merged_df,
        sample_submission_path=(repo_root / "sample_submission.csv"),
        output_path=submission_path,
    )

    cluster_counts = texas_df["pred_cluster_id"].value_counts()
    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": TEXAS_DATASET,
                "samples": int(len(texas_df)),
                "clusters": int(cluster_counts.size),
                "singleton_clusters": int((cluster_counts == 1).sum()),
                "singleton_ratio": round(float((cluster_counts == 1).mean()) if len(cluster_counts) else 0.0, 6),
                "route_name": str(route_name),
                "score_matrix_name": str(route_summary.get("score_matrix_name", "")),
                "method": str(route_summary.get("method", "")),
                "proxy_score": float(route_summary.get("proxy_score", 0.0)),
            }
        ]
    )
    cluster_summary_path = tables_dir / "cluster_summary_v1.csv"
    cluster_summary_df.to_csv(cluster_summary_path, index=False)

    summary_json_path = reports_dir / "summary.json"
    summary_json = {
        "base_predictions_path": _path_ref(repo_root, resolved_base_predictions_path),
        "route_name": str(route_name),
        "route_summary": route_summary,
        "submission_path": _path_ref(repo_root, submission_path),
    }
    summary_json_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# Texas HotSpotter Local Submission",
        "",
        f"- Base predictions: `{_path_ref(repo_root, resolved_base_predictions_path)}`",
        f"- Route name: `{route_name}`",
        f"- Score matrix: `{route_summary.get('score_matrix_name', '')}`",
        f"- Method: `{route_summary.get('method', '')}`",
        f"- Param key: `{route_summary.get('param_key', '')}`",
        f"- Proxy score: `{float(route_summary.get('proxy_score', 0.0)):.6f}`",
        f"- Candidate pair keep ratio: `{float(route_summary.get('candidate_pair_keep_ratio', 0.0)):.6f}`",
        f"- Mutual topk pair keep ratio: `{float(route_summary.get('mutual_topk_pair_keep_ratio', 0.0)):.6f}`",
        "",
        "## Texas Cluster Summary",
        "",
        _markdown_table(cluster_summary_df),
        "",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "predictions_path": predictions_path,
        "submission_path": submission_path,
        "summary_path": summary_path,
        "cluster_summary_path": cluster_summary_path,
        "runtime_dir": runtime_dir,
    }


def count_kaggle_submissions_today(*, repo_root: Path) -> tuple[int, str]:
    env = {**os.environ, "http_proxy": "http://127.0.0.1:9999", "https_proxy": "http://127.0.0.1:9999"}
    command = (
        "source /home/hechen/miniconda3/etc/profile.d/conda.sh && "
        "conda activate wildfusion && "
        "python - <<'PY'\n"
        "import os, subprocess, datetime\n"
        "cmd=['kaggle','competitions','submissions','-c','animal-clef-2026','-v']\n"
        "res=subprocess.run(cmd, capture_output=True, text=True, env=os.environ)\n"
        "print(res.stdout)\n"
        "PY"
    )
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return 0, stdout
    today_utc = time.strftime("%Y-%m-%d", time.gmtime())
    used = 0
    for line in stdout.splitlines()[1:]:
        if line.startswith("submission.csv,") and line.split(",", 2)[1].startswith(today_utc):
            used += 1
    return used, stdout


def kaggle_submit_and_poll(
    *,
    repo_root: Path,
    submission_path: Path,
    description: str,
    poll_seconds: int = 20,
    timeout_seconds: int = 900,
) -> dict[str, object]:
    env = {**os.environ, "http_proxy": "http://127.0.0.1:9999", "https_proxy": "http://127.0.0.1:9999"}
    submit_cmd = (
        "source /home/hechen/miniconda3/etc/profile.d/conda.sh && "
        "conda activate wildfusion && "
        f"kaggle competitions submit -c animal-clef-2026 -f {str(submission_path)!r} -m {description!r}"
    )
    submit_proc = subprocess.run(
        ["bash", "-lc", submit_cmd],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    result: dict[str, object] = {
        "submit_returncode": int(submit_proc.returncode),
        "submit_stdout": submit_proc.stdout,
        "submit_stderr": submit_proc.stderr,
        "description": str(description),
        "public_score": None,
        "status": "",
        "submission_row": "",
    }
    if submit_proc.returncode != 0:
        return result

    list_cmd = (
        "source /home/hechen/miniconda3/etc/profile.d/conda.sh && "
        "conda activate wildfusion && "
        "kaggle competitions submissions -c animal-clef-2026 -v"
    )
    start = time.time()
    while time.time() - start <= float(timeout_seconds):
        list_proc = subprocess.run(
            ["bash", "-lc", list_cmd],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if list_proc.returncode == 0:
            lines = list_proc.stdout.splitlines()
            for line in lines[1:]:
                if str(description) in line:
                    result["submission_row"] = line
                    if "SubmissionStatus.COMPLETE" in line:
                        result["status"] = "complete"
                        parts = line.split(",")
                        if len(parts) >= 5 and parts[4].strip():
                            try:
                                result["public_score"] = float(parts[4].strip())
                            except ValueError:
                                result["public_score"] = None
                        return result
                    if "SubmissionStatus.ERROR" in line:
                        result["status"] = "error"
                        return result
        time.sleep(max(5, int(poll_seconds)))
    result["status"] = "timeout"
    return result
