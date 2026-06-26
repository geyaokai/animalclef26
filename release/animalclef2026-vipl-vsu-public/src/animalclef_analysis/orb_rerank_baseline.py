from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .descriptor_baselines import (
    PATH_COLUMN,
    build_average_linkage,
    cluster_from_linkage,
    dataframe_to_markdown_table,
    load_cached_embedding_bundle,
    summarize_cluster_metrics,
)

try:  # pragma: no cover - exercised in the wildfusion env
    import cv2
except ModuleNotFoundError:  # pragma: no cover - keeps light test env working
    cv2 = None


DEFAULT_THRESHOLDS = [0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.35, 0.4, 0.5, 0.6]
DEFAULT_LOCAL_WEIGHTS = [0.25, 0.5, 0.75]
DEFAULT_LOCAL_MATCHER = "orb"
SUPPORTED_LOCAL_MATCHERS = ("orb", "akaze", "sift", "brisk", "kaze")
BINARY_LOCAL_MATCHERS = {"orb", "akaze", "brisk"}


@dataclass(frozen=True)
class OrbFeature:
    image_id: str
    matcher_name: str
    point_count: int
    points: np.ndarray
    descriptors: np.ndarray | None
    width: int
    height: int


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("ORB rerank requires OpenCV. Run in the 'wildfusion' environment.")


def normalize_local_matcher_name(local_matcher: str) -> str:
    normalized = str(local_matcher).strip().lower()
    if normalized not in SUPPORTED_LOCAL_MATCHERS:
        raise ValueError(f"Unsupported local matcher: {local_matcher}. Expected one of {SUPPORTED_LOCAL_MATCHERS}.")
    return normalized


def _create_feature_detector(local_matcher: str, nfeatures: int, fast_threshold: int):
    _require_cv2()
    matcher_name = normalize_local_matcher_name(local_matcher)
    if matcher_name == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=fast_threshold)
    if matcher_name == "akaze":
        return cv2.AKAZE_create()
    if matcher_name == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    if matcher_name == "brisk":
        return cv2.BRISK_create(thresh=max(5, int(fast_threshold)))
    if matcher_name == "kaze":
        return cv2.KAZE_create()
    raise ValueError(f"Unsupported local matcher: {local_matcher}")


def _feature_matcher_norm(local_matcher: str) -> int:
    _require_cv2()
    matcher_name = normalize_local_matcher_name(local_matcher)
    if matcher_name in BINARY_LOCAL_MATCHERS:
        return cv2.NORM_HAMMING
    return cv2.NORM_L2


def cosine_score_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    score = (similarity + 1.0) / 2.0
    np.fill_diagonal(score, 1.0)
    return score.astype(np.float32, copy=False)


def score_matrix_to_distance(score_matrix: np.ndarray) -> np.ndarray:
    distance = 1.0 - np.clip(score_matrix, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32, copy=False)


def recall_at_k_from_score_matrix(score_matrix: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(labels) < 2:
        return 0.0
    counts = pd.Series(labels).value_counts()
    valid_mask = np.array([counts[label] > 1 for label in labels], dtype=bool)
    if not valid_mask.any():
        return 0.0
    masked = score_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    width = min(k, len(labels) - 1)
    topk_indices = np.argpartition(-masked, kth=width - 1, axis=1)[:, :width]
    hits: list[bool] = []
    for index, neighbors in enumerate(topk_indices):
        if not valid_mask[index]:
            continue
        hits.append(bool(np.any(labels[neighbors] == labels[index])))
    return round(float(np.mean(hits)), 6) if hits else 0.0


def top1_neighbor_ids(score_matrix: np.ndarray) -> np.ndarray:
    if len(score_matrix) < 2:
        return np.array([], dtype=int)
    masked = score_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    return np.argmax(masked, axis=1).astype(int)


def sample_query_indices(labels: np.ndarray, max_queries: int, seed: int) -> np.ndarray:
    total = len(labels)
    if total < 2:
        return np.array([], dtype=int)
    if total <= max_queries:
        return np.arange(total, dtype=int)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return np.array(sorted(indices[:max_queries]), dtype=int)


def build_topk_pair_index(
    score_matrix: np.ndarray,
    top_k: int,
    query_indices: np.ndarray | None = None,
) -> list[tuple[int, int, float]]:
    if len(score_matrix) < 2:
        return []
    masked = score_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    width = min(top_k, len(score_matrix) - 1)
    topk_indices = np.argpartition(-masked, kth=width - 1, axis=1)[:, :width]
    selected_queries = (
        np.arange(len(score_matrix), dtype=int)
        if query_indices is None
        else np.array(sorted(set(int(index) for index in query_indices.tolist())), dtype=int)
    )
    pairs: dict[tuple[int, int], float] = {}
    for left_index in selected_queries:
        ranked = sorted(topk_indices[left_index].tolist(), key=lambda idx: masked[left_index, idx], reverse=True)
        for right_index in ranked:
            if left_index == right_index:
                continue
            key = (left_index, right_index) if left_index < right_index else (right_index, left_index)
            score = float(score_matrix[left_index, right_index])
            if key not in pairs or score > pairs[key]:
                pairs[key] = score
    return [(int(left), int(right), float(pairs[(left, right)])) for left, right in sorted(pairs)]


def _load_grayscale_image(
    image_path: Path,
    max_side: int,
    clahe_clip_limit: float,
    *,
    hflip: bool = False,
) -> tuple[np.ndarray, int, int]:
    _require_cv2()
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image for ORB matching: {image_path}")
    height, width = image.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        width, height = resized_width, resized_height
    if hflip:
        image = cv2.flip(image, 1)
    if clahe_clip_limit > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
        image = clahe.apply(image)
    return image, width, height


def _coerce_rel_path(record: object, column: str) -> str:
    if isinstance(record, pd.Series):
        value = record.get(column, "")
    elif isinstance(record, dict):
        value = record.get(column, "")
    else:
        value = getattr(record, column, "")
    if pd.isna(value):
        return ""
    return str(value).strip()


def resolve_existing_image_rel_path(record: object, repo_root: Path) -> str:
    checked_paths: list[str] = []
    for column in [PATH_COLUMN, "path", "preferred_path_v1", "normalized_path_v1"]:
        rel_path = _coerce_rel_path(record, column)
        if not rel_path or rel_path in checked_paths:
            continue
        checked_paths.append(rel_path)
        if (repo_root / rel_path).exists():
            return rel_path
    raise FileNotFoundError(
        "Could not resolve image path for ORB matching; checked paths: "
        + ", ".join(checked_paths[:4])
    )


def extract_local_features(
    df: pd.DataFrame,
    repo_root: Path,
    nfeatures: int,
    max_side: int,
    fast_threshold: int,
    clahe_clip_limit: float,
    local_matcher: str = DEFAULT_LOCAL_MATCHER,
    *,
    hflip: bool = False,
) -> list[OrbFeature]:
    _require_cv2()
    matcher_name = normalize_local_matcher_name(local_matcher)
    detector = _create_feature_detector(local_matcher=matcher_name, nfeatures=nfeatures, fast_threshold=fast_threshold)
    features: list[OrbFeature] = []
    for row in df.itertuples(index=False):
        image_path = repo_root / resolve_existing_image_rel_path(row, repo_root=repo_root)
        gray_image, width, height = _load_grayscale_image(
            image_path=image_path,
            max_side=max_side,
            clahe_clip_limit=clahe_clip_limit,
            hflip=bool(hflip),
        )
        keypoints, descriptors = detector.detectAndCompute(gray_image, None)
        if keypoints:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.empty((0, 2), dtype=np.float32)
            descriptors = None
        features.append(
            OrbFeature(
                image_id=str(getattr(row, "image_id")),
                matcher_name=matcher_name,
                point_count=int(len(points)),
                points=points,
                descriptors=descriptors,
                width=int(width),
                height=int(height),
            )
        )
    return features


def extract_orb_features(
    df: pd.DataFrame,
    repo_root: Path,
    nfeatures: int,
    max_side: int,
    fast_threshold: int,
    clahe_clip_limit: float,
) -> list[OrbFeature]:
    return extract_local_features(
        df=df,
        repo_root=repo_root,
        nfeatures=nfeatures,
        max_side=max_side,
        fast_threshold=fast_threshold,
        clahe_clip_limit=clahe_clip_limit,
        local_matcher="orb",
    )


def compute_local_match(
    left_feature: OrbFeature,
    right_feature: OrbFeature,
    ratio_test: float,
    ransac_threshold: float,
    min_inliers: int,
    local_matcher: str = DEFAULT_LOCAL_MATCHER,
) -> dict[str, object]:
    _require_cv2()
    matcher_name = normalize_local_matcher_name(local_matcher)
    if left_feature.descriptors is None or right_feature.descriptors is None:
        return {
            "good_matches": 0,
            "inliers": 0,
            "local_raw_score": 0.0,
        }
    if len(left_feature.descriptors) < 2 or len(right_feature.descriptors) < 2:
        return {
            "good_matches": 0,
            "inliers": 0,
            "local_raw_score": 0.0,
        }
    matcher = cv2.BFMatcher(_feature_matcher_norm(matcher_name), crossCheck=False)
    raw_matches = matcher.knnMatch(left_feature.descriptors, right_feature.descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_test * second.distance:
            good_matches.append(first)
    if len(good_matches) < 4:
        return {
            "good_matches": int(len(good_matches)),
            "inliers": 0,
            "local_raw_score": 0.0,
        }
    left_points = np.float32([left_feature.points[match.queryIdx] for match in good_matches])
    right_points = np.float32([right_feature.points[match.trainIdx] for match in good_matches])
    try:
        _homography, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, ransac_threshold)
    except cv2.error:
        mask = None
    inliers = int(mask.sum()) if mask is not None else 0
    if inliers < min_inliers:
        local_raw_score = 0.0
    else:
        local_raw_score = float(inliers / max(1, min(left_feature.point_count, right_feature.point_count)))
    return {
        "good_matches": int(len(good_matches)),
        "inliers": inliers,
        "local_raw_score": round(local_raw_score, 6),
    }


def compute_orb_match(
    left_feature: OrbFeature,
    right_feature: OrbFeature,
    ratio_test: float,
    ransac_threshold: float,
    min_inliers: int,
) -> dict[str, object]:
    return compute_local_match(
        left_feature=left_feature,
        right_feature=right_feature,
        ratio_test=ratio_test,
        ransac_threshold=ransac_threshold,
        min_inliers=min_inliers,
        local_matcher="orb",
    )


def build_local_match_table(
    df: pd.DataFrame,
    features: list[OrbFeature],
    pair_index: list[tuple[int, int, float]],
    ratio_test: float,
    ransac_threshold: float,
    min_inliers: int,
    local_matcher: str = DEFAULT_LOCAL_MATCHER,
    flipped_features: list[OrbFeature] | None = None,
) -> pd.DataFrame:
    matcher_name = normalize_local_matcher_name(local_matcher)
    rows: list[dict[str, object]] = []
    for left_index, right_index, global_score in pair_index:
        left_row = df.iloc[left_index]
        right_row = df.iloc[right_index]
        match = compute_local_match(
            left_feature=features[left_index],
            right_feature=features[right_index],
            ratio_test=ratio_test,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
            local_matcher=matcher_name,
        )
        flip_applied = False
        if flipped_features is not None:
            flipped_match = compute_local_match(
                left_feature=features[left_index],
                right_feature=flipped_features[right_index],
                ratio_test=ratio_test,
                ransac_threshold=ransac_threshold,
                min_inliers=min_inliers,
                local_matcher=matcher_name,
            )
            base_key = (
                float(match.get("local_raw_score", 0.0) or 0.0),
                int(match.get("inliers", 0) or 0),
                int(match.get("good_matches", 0) or 0),
            )
            flipped_key = (
                float(flipped_match.get("local_raw_score", 0.0) or 0.0),
                int(flipped_match.get("inliers", 0) or 0),
                int(flipped_match.get("good_matches", 0) or 0),
            )
            if flipped_key > base_key:
                match = flipped_match
                flip_applied = True
        rows.append(
            {
                "dataset": left_row["dataset"],
                "matcher_name": matcher_name,
                "left_index": int(left_index),
                "right_index": int(right_index),
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "identity": str(left_row["identity"]),
                "neighbor_identity": str(right_row["identity"]),
                "same_identity": bool(left_row["identity"] == right_row["identity"]),
                "global_score": round(float(global_score), 6),
                "left_keypoints": int(features[left_index].point_count),
                "right_keypoints": int(features[right_index].point_count),
                "flip_invariant_enabled": bool(flipped_features is not None),
                "right_flipped_match_selected": bool(flip_applied),
                **match,
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
    calibrated = np.clip(pair_df["local_raw_score"].to_numpy(dtype=float) / upper, 0.0, 1.0)
    pair_df["local_score"] = np.round(calibrated, 6)
    return pair_df


def apply_local_rerank(
    global_score_matrix: np.ndarray,
    pair_df: pd.DataFrame,
    local_weight: float,
) -> np.ndarray:
    reranked = global_score_matrix.copy().astype(np.float32, copy=True)
    if pair_df.empty or local_weight <= 0:
        return reranked
    for row in pair_df.itertuples(index=False):
        boost = float(local_weight * row.local_score * (1.0 - row.global_score))
        fused_score = min(1.0, float(row.global_score + boost))
        reranked[row.left_index, row.right_index] = fused_score
        reranked[row.right_index, row.left_index] = fused_score
    np.fill_diagonal(reranked, 1.0)
    return reranked


def evaluate_threshold_sweep_from_score_matrix(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].reset_index(drop=True)
        dataset_mask = (df["dataset"] == dataset).to_numpy()
        dataset_scores = score_matrix[np.ix_(dataset_mask, dataset_mask)]
        distance = score_matrix_to_distance(dataset_scores)
        linkage_matrix = build_average_linkage(distance)
        true_labels = dataset_df["identity"].to_numpy()
        for threshold in thresholds:
            pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), threshold)
            metrics = summarize_cluster_metrics(true_labels=true_labels, pred_labels=pred_labels)
            rows.append(
                {
                    "dataset": dataset,
                    "threshold": threshold,
                    "samples": int(len(dataset_df)),
                    **metrics,
                }
            )
            frame = dataset_df[["image_id", "dataset", "identity", PATH_COLUMN]].copy()
            frame["threshold"] = threshold
            frame["pred_cluster_id"] = pred_labels
            prediction_frames.append(frame)
    sweep_df = pd.DataFrame(rows).sort_values(["dataset", "threshold"]).reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def pick_best_rows_by_dataset(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for dataset, dataset_df in metrics_df.groupby("dataset"):
        best = dataset_df.sort_values(
            ["ari", "nmi", "pairwise_f1", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_smoke_summary(
    df: pd.DataFrame,
    global_score_matrix: np.ndarray,
    local_match_df: pd.DataFrame,
    local_weights: list[float],
    smoke_query_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = df["identity"].to_numpy()
    baseline_top1 = top1_neighbor_ids(global_score_matrix)
    baseline_hits = (labels[baseline_top1] == labels).astype(int) if len(baseline_top1) else np.array([], dtype=int)
    sampled_baseline = float(baseline_hits[smoke_query_indices].mean()) if len(smoke_query_indices) else 0.0

    rows: list[dict[str, object]] = []
    chosen_rows: list[pd.Series] = []
    positive_scores = local_match_df[local_match_df["same_identity"]]["local_score"].to_numpy(dtype=float)
    negative_scores = local_match_df[~local_match_df["same_identity"]]["local_score"].to_numpy(dtype=float)
    for local_weight in local_weights:
        reranked = apply_local_rerank(global_score_matrix=global_score_matrix, pair_df=local_match_df, local_weight=local_weight)
        reranked_top1 = top1_neighbor_ids(reranked)
        reranked_hits = (labels[reranked_top1] == labels).astype(int) if len(reranked_top1) else np.array([], dtype=int)
        corrected = int(
            np.sum(
                (baseline_hits[smoke_query_indices] == 0)
                & (reranked_hits[smoke_query_indices] == 1)
            )
        ) if len(smoke_query_indices) else 0
        harmed = int(
            np.sum(
                (baseline_hits[smoke_query_indices] == 1)
                & (reranked_hits[smoke_query_indices] == 0)
            )
        ) if len(smoke_query_indices) else 0
        rows.append(
            {
                "dataset": str(df["dataset"].iloc[0]),
                "local_weight": local_weight,
                "smoke_queries": int(len(smoke_query_indices)),
                "baseline_recall_at_1": round(sampled_baseline, 6),
                "rerank_recall_at_1": round(float(reranked_hits[smoke_query_indices].mean()) if len(smoke_query_indices) else 0.0, 6),
                "corrected_queries": corrected,
                "harmed_queries": harmed,
                "positive_local_score_mean": round(float(positive_scores.mean()) if len(positive_scores) else 0.0, 6),
                "negative_local_score_mean": round(float(negative_scores.mean()) if len(negative_scores) else 0.0, 6),
                "positive_pair_count": int(len(positive_scores)),
                "negative_pair_count": int(len(negative_scores)),
            }
        )
    smoke_df = pd.DataFrame(rows).sort_values(["rerank_recall_at_1", "corrected_queries", "local_weight"], ascending=[False, False, True]).reset_index(drop=True)
    best = smoke_df.iloc[0]
    enable_rerank = bool(
        (best["rerank_recall_at_1"] > best["baseline_recall_at_1"])
        or (
            best["rerank_recall_at_1"] == best["baseline_recall_at_1"]
            and best["corrected_queries"] >= best["harmed_queries"]
            and best["positive_local_score_mean"] > best["negative_local_score_mean"]
            and best["corrected_queries"] > 0
        )
    )
    gate_df = pd.DataFrame(
        [
            {
                "dataset": best["dataset"],
                "enable_rerank": enable_rerank,
                "chosen_local_weight": float(best["local_weight"]) if enable_rerank else 0.0,
                "baseline_recall_at_1": float(best["baseline_recall_at_1"]),
                "smoke_recall_at_1": float(best["rerank_recall_at_1"]),
                "corrected_queries": int(best["corrected_queries"]),
                "harmed_queries": int(best["harmed_queries"]),
                "positive_local_score_mean": float(best["positive_local_score_mean"]),
                "negative_local_score_mean": float(best["negative_local_score_mean"]),
            }
        ]
    )
    return smoke_df, gate_df


def build_neighbor_table_from_score_matrix(
    df: pd.DataFrame,
    score_matrix: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(
            columns=[
                "dataset",
                "image_id",
                "identity",
                "neighbor_rank",
                "neighbor_image_id",
                "neighbor_identity",
                "similarity",
                "same_identity",
            ]
        )
    masked = score_matrix.copy()
    np.fill_diagonal(masked, -np.inf)
    width = min(top_k, len(df) - 1)
    topk_indices = np.argpartition(-masked, kth=width - 1, axis=1)[:, :width]
    rows: list[dict[str, object]] = []
    for index, row in enumerate(df.itertuples(index=False)):
        ranked = sorted(topk_indices[index].tolist(), key=lambda idx: masked[index, idx], reverse=True)
        for rank, neighbor_index in enumerate(ranked, start=1):
            neighbor = df.iloc[neighbor_index]
            rows.append(
                {
                    "dataset": row.dataset,
                    "image_id": row.image_id,
                    "identity": row.identity,
                    "neighbor_rank": rank,
                    "neighbor_image_id": neighbor["image_id"],
                    "neighbor_identity": neighbor["identity"],
                    "similarity": round(float(score_matrix[index, neighbor_index]), 6),
                    "same_identity": bool(row.identity == neighbor["identity"]),
                }
            )
    return pd.DataFrame(rows)


def build_top1_transition_table(
    df: pd.DataFrame,
    global_score_matrix: np.ndarray,
    reranked_score_matrix: np.ndarray,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame()
    labels = df["identity"].astype(str).to_numpy()
    global_top1 = top1_neighbor_ids(global_score_matrix)
    reranked_top1 = top1_neighbor_ids(reranked_score_matrix)
    rows: list[dict[str, object]] = []
    for index in range(len(df)):
        rows.append(
            {
                "dataset": str(df.iloc[index]["dataset"]),
                "image_id": str(df.iloc[index]["image_id"]),
                "identity": labels[index],
                "global_top1_image_id": str(df.iloc[global_top1[index]]["image_id"]),
                "global_top1_identity": labels[global_top1[index]],
                "global_top1_same_identity": bool(labels[index] == labels[global_top1[index]]),
                "reranked_top1_image_id": str(df.iloc[reranked_top1[index]]["image_id"]),
                "reranked_top1_identity": labels[reranked_top1[index]],
                "reranked_top1_same_identity": bool(labels[index] == labels[reranked_top1[index]]),
                "global_score": round(float(global_score_matrix[index, global_top1[index]]), 6),
                "reranked_score": round(float(reranked_score_matrix[index, reranked_top1[index]]), 6),
            }
        )
    transition_df = pd.DataFrame(rows)
    transition_df["transition_type"] = np.select(
        [
            (~transition_df["global_top1_same_identity"]) & (transition_df["reranked_top1_same_identity"]),
            (transition_df["global_top1_same_identity"]) & (~transition_df["reranked_top1_same_identity"]),
            (transition_df["global_top1_same_identity"]) & (transition_df["reranked_top1_same_identity"]),
        ],
        [
            "corrected",
            "harmed",
            "stable_correct",
        ],
        default="stable_wrong",
    )
    return transition_df


def create_triplet_contact_sheet(
    rows_df: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    columns: int = 2,
    thumb_size: tuple[int, int] = (200, 200),
) -> None:
    if rows_df.empty:
        return
    margin = 12
    header_h = 34
    label_h = 72
    image_gap = 8
    panel_w, panel_h = thumb_size
    cell_w = panel_w * 3 + image_gap * 2
    cell_h = panel_h + label_h
    grid_rows = math.ceil(len(rows_df) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + grid_rows * cell_h + (grid_rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for index, row in enumerate(rows_df.itertuples(index=False)):
        gx = index % columns
        gy = index // columns
        x = margin + gx * (cell_w + margin)
        y = start_y + gy * (cell_h + margin)
        query_image = ImageOps.pad(Image.open(repo_root / getattr(row, "query_path")).convert("RGB"), thumb_size, color=(10, 10, 10))
        before_image = ImageOps.pad(Image.open(repo_root / getattr(row, "global_top1_path")).convert("RGB"), thumb_size, color=(10, 10, 10))
        after_image = ImageOps.pad(Image.open(repo_root / getattr(row, "reranked_top1_path")).convert("RGB"), thumb_size, color=(10, 10, 10))
        canvas.paste(query_image, (x, y))
        canvas.paste(before_image, (x + panel_w + image_gap, y))
        canvas.paste(after_image, (x + 2 * (panel_w + image_gap), y))
        caption = (
            f"q:{row.image_id} ({row.identity})\n"
            f"before:{row.global_top1_image_id} ({row.global_top1_identity}) s={row.global_score:.3f}\n"
            f"after:{row.reranked_top1_image_id} ({row.reranked_top1_identity}) s={row.reranked_score:.3f}"
        )
        draw.multiline_text((x, y + panel_h + 4), caption, fill=(30, 30, 30), font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def create_qualitative_outputs(
    df: pd.DataFrame,
    transition_df: pd.DataFrame,
    repo_root: Path,
    qualitative_dir: Path,
) -> None:
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    if transition_df.empty:
        return
    path_lookup = df.copy()
    path_lookup["query_path"] = path_lookup.apply(
        lambda row: resolve_existing_image_rel_path(row, repo_root=repo_root),
        axis=1,
    )
    path_lookup = path_lookup[["image_id", "query_path"]]
    for transition_type in ["corrected", "harmed"]:
        subset = transition_df[transition_df["transition_type"] == transition_type].copy()
        if subset.empty:
            continue
        subset = subset.head(6)
        merged = subset.merge(path_lookup, on="image_id", how="left")
        merged = merged.merge(
            path_lookup.rename(columns={"image_id": "global_top1_image_id", "query_path": "global_top1_path"}),
            on="global_top1_image_id",
            how="left",
        )
        merged = merged.merge(
            path_lookup.rename(columns={"image_id": "reranked_top1_image_id", "query_path": "reranked_top1_path"}),
            on="reranked_top1_image_id",
            how="left",
        )
        create_triplet_contact_sheet(
            rows_df=merged,
            repo_root=repo_root,
            output_path=qualitative_dir / f"top1_{transition_type}.jpg",
            title=f"Top-1 {transition_type.title()} | query vs global vs rerank",
        )


def write_markdown_report(
    output_path: Path,
    config: dict[str, object],
    smoke_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    baseline_best_df: pd.DataFrame,
    rerank_best_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    lines = [
        "# ORB Rerank Summary",
        "",
        f"- Source baseline: `{config['source_dir']}`",
        f"- Top-K candidate neighbors: `{config['top_k']}`",
        f"- Smoke max queries per dataset: `{config['smoke_max_queries']}`",
        f"- ORB features per image: `{config['orb_features']}`",
        f"- ORB resize max side: `{config['orb_max_side']}`",
        f"- Ratio test: `{config['ratio_test']}`",
        f"- Minimum RANSAC inliers: `{config['min_inliers']}`",
        f"- Local weight candidates: `{config['local_weights']}`",
        f"- Threshold sweep: `{config['thresholds']}`",
        "",
        "## Smoke Test Summary",
        "",
        dataframe_to_markdown_table(smoke_df),
        "",
        "## Dataset Gate Decisions",
        "",
        dataframe_to_markdown_table(gate_df),
        "",
        "## Baseline Best Validation Rows",
        "",
        dataframe_to_markdown_table(
            baseline_best_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]
        ),
        "",
        "## Rerank Best Validation Rows",
        "",
        dataframe_to_markdown_table(
            rerank_best_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]
        ),
        "",
        "## Before/After Comparison",
        "",
        dataframe_to_markdown_table(comparison_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_orb_rerank_baseline(
    repo_root: Path,
    source_dir: Path,
    output_dir: Path,
    top_k: int = 10,
    smoke_max_queries: int = 40,
    smoke_seed: int = 42,
    orb_features: int = 1024,
    orb_max_side: int = 768,
    fast_threshold: int = 7,
    clahe_clip_limit: float = 2.0,
    ratio_test: float = 0.8,
    ransac_threshold: float = 5.0,
    min_inliers: int = 8,
    local_weights: list[float] | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Path]:
    _require_cv2()
    if local_weights is None:
        local_weights = DEFAULT_LOCAL_WEIGHTS
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    bundle = load_cached_embedding_bundle(source_dir=source_dir, name="fusion")
    val_df = bundle.val_df.copy().reset_index(drop=True)
    val_embeddings = bundle.val_embeddings
    global_score = cosine_score_matrix(val_embeddings)

    print("[orb_rerank] running baseline sweep on cached fusion embeddings", flush=True)
    baseline_sweep_df, _baseline_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=val_df,
        score_matrix=global_score,
        thresholds=thresholds,
    )
    baseline_best_df = pick_best_rows_by_dataset(baseline_sweep_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    qualitative_dir = output_dir / "qualitative"
    for path in [tables_dir, reports_dir, qualitative_dir]:
        path.mkdir(parents=True, exist_ok=True)

    features_by_dataset: dict[str, list[OrbFeature]] = {}
    smoke_frames: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []
    full_pair_frames: list[pd.DataFrame] = []
    reranked_blocks: dict[str, np.ndarray] = {}
    transition_frames: list[pd.DataFrame] = []
    reranked_neighbor_frames: list[pd.DataFrame] = []
    global_neighbor_frames: list[pd.DataFrame] = []
    keypoint_rows: list[dict[str, object]] = []

    for dataset in sorted(val_df["dataset"].unique()):
        print(f"[orb_rerank] dataset={dataset} | extracting ORB features", flush=True)
        dataset_mask = (val_df["dataset"] == dataset).to_numpy()
        dataset_df = val_df.loc[dataset_mask].reset_index(drop=True)
        dataset_embeddings = val_embeddings[dataset_mask]
        dataset_score = cosine_score_matrix(dataset_embeddings)
        features = extract_orb_features(
            df=dataset_df,
            repo_root=repo_root,
            nfeatures=orb_features,
            max_side=orb_max_side,
            fast_threshold=fast_threshold,
            clahe_clip_limit=clahe_clip_limit,
        )
        features_by_dataset[dataset] = features
        keypoint_rows.extend(
            {
                "dataset": dataset,
                "image_id": feature.image_id,
                "keypoints": feature.point_count,
                "width": feature.width,
                "height": feature.height,
            }
            for feature in features
        )

        smoke_query_indices = sample_query_indices(
            labels=dataset_df["identity"].to_numpy(),
            max_queries=smoke_max_queries,
            seed=smoke_seed,
        )
        smoke_pairs = build_topk_pair_index(
            score_matrix=dataset_score,
            top_k=top_k,
            query_indices=smoke_query_indices,
        )
        print(
            f"[orb_rerank] dataset={dataset} | smoke queries={len(smoke_query_indices)} | candidate pairs={len(smoke_pairs)}",
            flush=True,
        )
        smoke_pair_df = build_local_match_table(
            df=dataset_df,
            features=features,
            pair_index=smoke_pairs,
            ratio_test=ratio_test,
            ransac_threshold=ransac_threshold,
            min_inliers=min_inliers,
        )
        smoke_summary_df, gate_df = build_smoke_summary(
            df=dataset_df,
            global_score_matrix=dataset_score,
            local_match_df=smoke_pair_df,
            local_weights=local_weights,
            smoke_query_indices=smoke_query_indices,
        )
        smoke_frames.append(smoke_summary_df)
        gate_rows.append(gate_df)

        enable_rerank = bool(gate_df.iloc[0]["enable_rerank"])
        chosen_weight = float(gate_df.iloc[0]["chosen_local_weight"])
        print(
            f"[orb_rerank] dataset={dataset} | enable_rerank={enable_rerank} | chosen_weight={chosen_weight}",
            flush=True,
        )
        if enable_rerank:
            full_pairs = build_topk_pair_index(
                score_matrix=dataset_score,
                top_k=top_k,
                query_indices=None,
            )
            print(f"[orb_rerank] dataset={dataset} | full rerank candidate pairs={len(full_pairs)}", flush=True)
            full_pair_df = build_local_match_table(
                df=dataset_df,
                features=features,
                pair_index=full_pairs,
                ratio_test=ratio_test,
                ransac_threshold=ransac_threshold,
                min_inliers=min_inliers,
            )
            reranked_score = apply_local_rerank(
                global_score_matrix=dataset_score,
                pair_df=full_pair_df,
                local_weight=chosen_weight,
            )
            full_pair_df["local_weight"] = chosen_weight
        else:
            full_pair_df = smoke_pair_df.copy()
            full_pair_df["local_weight"] = 0.0
            reranked_score = dataset_score

        full_pair_frames.append(full_pair_df)
        reranked_blocks[dataset] = reranked_score
        global_neighbor_frames.append(build_neighbor_table_from_score_matrix(dataset_df[["image_id", "dataset", "identity"]], dataset_score, top_k=5))
        reranked_neighbor_frames.append(build_neighbor_table_from_score_matrix(dataset_df[["image_id", "dataset", "identity"]], reranked_score, top_k=5))
        transition_frames.append(build_top1_transition_table(dataset_df, dataset_score, reranked_score))
        create_qualitative_outputs(
            df=dataset_df,
            transition_df=transition_frames[-1],
            repo_root=repo_root,
            qualitative_dir=qualitative_dir / dataset,
        )

    reranked_score = np.empty_like(global_score)
    for dataset in sorted(val_df["dataset"].unique()):
        dataset_mask = (val_df["dataset"] == dataset).to_numpy()
        dataset_indices = np.flatnonzero(dataset_mask)
        reranked_score[np.ix_(dataset_indices, dataset_indices)] = reranked_blocks[dataset]

    rerank_sweep_df, rerank_prediction_df = evaluate_threshold_sweep_from_score_matrix(
        df=val_df,
        score_matrix=reranked_score,
        thresholds=thresholds,
    )
    print("[orb_rerank] rerank sweep complete, writing outputs", flush=True)
    rerank_best_df = pick_best_rows_by_dataset(rerank_sweep_df)

    comparison_df = baseline_best_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count"]].merge(
        rerank_best_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count"]],
        on="dataset",
        suffixes=("_baseline", "_rerank"),
    )
    comparison_df["ari_delta"] = np.round(comparison_df["ari_rerank"] - comparison_df["ari_baseline"], 6)
    comparison_df["nmi_delta"] = np.round(comparison_df["nmi_rerank"] - comparison_df["nmi_baseline"], 6)
    comparison_df["pairwise_f1_delta"] = np.round(comparison_df["pairwise_f1_rerank"] - comparison_df["pairwise_f1_baseline"], 6)
    comparison_df["cluster_count_delta"] = comparison_df["cluster_count_rerank"] - comparison_df["cluster_count_baseline"]

    smoke_df = pd.concat(smoke_frames, ignore_index=True)
    gate_df = pd.concat(gate_rows, ignore_index=True)
    local_match_df = pd.concat(full_pair_frames, ignore_index=True) if full_pair_frames else pd.DataFrame()
    transition_df = pd.concat(transition_frames, ignore_index=True) if transition_frames else pd.DataFrame()
    global_neighbor_df = pd.concat(global_neighbor_frames, ignore_index=True) if global_neighbor_frames else pd.DataFrame()
    reranked_neighbor_df = pd.concat(reranked_neighbor_frames, ignore_index=True) if reranked_neighbor_frames else pd.DataFrame()
    keypoint_df = pd.DataFrame(keypoint_rows)

    smoke_df.to_csv(tables_dir / "smoke_summary_v1.csv", index=False)
    gate_df.to_csv(tables_dir / "dataset_gate_v1.csv", index=False)
    baseline_sweep_df.to_csv(tables_dir / "baseline_val_threshold_sweep_v1.csv", index=False)
    baseline_best_df.to_csv(tables_dir / "baseline_best_thresholds_v1.csv", index=False)
    rerank_sweep_df.to_csv(tables_dir / "rerank_val_threshold_sweep_v1.csv", index=False)
    rerank_best_df.to_csv(tables_dir / "rerank_best_thresholds_v1.csv", index=False)
    comparison_df.to_csv(tables_dir / "baseline_vs_rerank_v1.csv", index=False)
    local_match_df.to_csv(tables_dir / "local_match_scores_v1.csv", index=False)
    transition_df.to_csv(tables_dir / "top1_transitions_v1.csv", index=False)
    rerank_prediction_df.to_csv(tables_dir / "rerank_val_predictions_v1.csv", index=False)
    global_neighbor_df.to_csv(tables_dir / "baseline_val_neighbors_v1.csv", index=False)
    reranked_neighbor_df.to_csv(tables_dir / "rerank_val_neighbors_v1.csv", index=False)
    keypoint_df.to_csv(tables_dir / "orb_keypoints_v1.csv", index=False)

    config = {
        "source_dir": str(source_dir),
        "top_k": top_k,
        "smoke_max_queries": smoke_max_queries,
        "orb_features": orb_features,
        "orb_max_side": orb_max_side,
        "fast_threshold": fast_threshold,
        "clahe_clip_limit": clahe_clip_limit,
        "ratio_test": ratio_test,
        "ransac_threshold": ransac_threshold,
        "min_inliers": min_inliers,
        "local_weights": local_weights,
        "thresholds": thresholds,
    }
    write_markdown_report(
        output_path=reports_dir / "summary.md",
        config=config,
        smoke_df=smoke_df,
        gate_df=gate_df,
        baseline_best_df=baseline_best_df,
        rerank_best_df=rerank_best_df,
        comparison_df=comparison_df,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "summary_path": reports_dir / "summary.md",
        "comparison_path": tables_dir / "baseline_vs_rerank_v1.csv",
        "smoke_path": tables_dir / "smoke_summary_v1.csv",
        "qualitative_dir": qualitative_dir,
    }
