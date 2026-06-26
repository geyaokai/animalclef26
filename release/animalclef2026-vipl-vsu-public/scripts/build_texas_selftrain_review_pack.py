#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict, deque
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_EXPERIMENT_DIR = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1")
DEFAULT_FUSION_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_texas_fusion_t0p43")
DEFAULT_MIEW_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_miew_v1")
DEFAULT_FUSION_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_fusion_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_selftrain_review_v1")
DEFAULT_LOCAL_MATCH_CSV = Path("artifacts/analysis/texas_orb_local_probe_v1/tables/test_pair_local_scores_v1.csv")
DEFAULT_PAIRWISE_FEATURE_CSV = Path("artifacts/analysis/texas_local_pairwise_probe_v1/tables/pair_features_v1.csv")
DEFAULT_MANUAL_JUDGMENTS_PATH = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _resolve_predictions_path(repo_root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else (repo_root / value)
    path = path.resolve()
    if path.is_file():
        return path
    candidates = [
        path / "tables" / "test_predictions_v1.csv",
        path / "test_predictions_v1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve test predictions from {value}")


def _load_texas_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    subset = df[df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {path}")
    subset["pred_cluster_id"] = subset["pred_cluster_id"].astype(int)
    return subset


def _align_frame(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, name: str) -> pd.DataFrame:
    lookup = candidate_df.copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup = lookup.set_index("image_id", drop=False)
    missing = [image_id for image_id in reference_df["image_id"].astype(str).tolist() if image_id not in lookup.index]
    if missing:
        raise ValueError(f"{name} missing image_ids, examples: {missing[:5]}")
    return lookup.loc[reference_df["image_id"].astype(str).tolist()].reset_index(drop=True).copy()


def _load_source_embeddings(
    repo_root: Path,
    *,
    source_dir: Path,
    reference_df: pd.DataFrame,
) -> np.ndarray:
    from animalclef_analysis.descriptor_baselines import load_cached_embedding_bundle

    resolved = source_dir if source_dir.is_absolute() else (repo_root / source_dir)
    bundle = load_cached_embedding_bundle(source_dir=resolved.resolve(), name=resolved.name)
    candidate_df = bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    embeddings = bundle.test_embeddings[(bundle.test_df["dataset"] == TEXAS_DATASET).to_numpy()]
    aligned_df = _align_frame(reference_df, candidate_df, resolved.name)
    image_to_index = {str(image_id): index for index, image_id in enumerate(candidate_df["image_id"].astype(str).tolist())}
    indices = np.array([image_to_index[str(image_id)] for image_id in aligned_df["image_id"].astype(str).tolist()], dtype=np.int32)
    return embeddings[indices].astype(np.float32, copy=False)


def _pair_keys_from_labels(labels: np.ndarray) -> set[tuple[int, int]]:
    by_cluster: dict[int, list[int]] = {}
    for index, label in enumerate(np.asarray(labels, dtype=np.int32).tolist()):
        by_cluster.setdefault(int(label), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        for left, right in combinations(sorted(members), 2):
            pairs.add((int(left), int(right)))
    return pairs


def _build_partial_seed_labels(pseudo_df: pd.DataFrame) -> np.ndarray:
    pseudo_df = pseudo_df.copy().reset_index(drop=True)
    pseudo_df["image_id"] = pseudo_df["image_id"].astype(str)
    seed_mask = pseudo_df["is_seed"].fillna(False).astype(bool).to_numpy()
    seed_labels = np.arange(len(pseudo_df), dtype=np.int32) + 100000
    if "pseudo_label_index" in pseudo_df.columns:
        pseudo_indices = pseudo_df["pseudo_label_index"].fillna(-1).astype(int).to_numpy()
        valid_mask = seed_mask & (pseudo_indices >= 0)
        seed_labels[valid_mask] = pseudo_indices[valid_mask].astype(np.int32)
    return seed_labels


def _score_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings.astype(np.float32, copy=False)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32, copy=False)


def _candidate_lookup(candidate_pair_df: pd.DataFrame) -> dict[tuple[int, int], dict[str, Any]]:
    lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for row in candidate_pair_df.itertuples(index=False):
        left = int(row.left_index)
        right = int(row.right_index)
        key = (left, right) if left < right else (right, left)
        lookup[key] = {
            "same_cluster_all_routes": bool(getattr(row, "same_cluster_all_routes", False)),
            "mutual_topk_all_routes": bool(getattr(row, "mutual_topk_all_routes", False)),
            "miew_similarity_shortlist": float(getattr(row, "miew_similarity", np.nan)),
            "fusion_similarity_shortlist": float(getattr(row, "fusion_similarity", np.nan)),
            "miew_mutual_topk": bool(getattr(row, "miew_mutual_topk", False)),
            "fusion_mutual_topk": bool(getattr(row, "fusion_mutual_topk", False)),
        }
    return lookup


def _support_image_ids_for_merge(pair_df: pd.DataFrame, left_cluster_id: int, right_cluster_id: int) -> str:
    subset = pair_df[
        pair_df["vote_direction"].astype(str).eq("merge")
        & (
            (
                pair_df["base_cluster_left"].astype(int).eq(int(left_cluster_id))
                & pair_df["base_cluster_right"].astype(int).eq(int(right_cluster_id))
            )
            | (
                pair_df["base_cluster_left"].astype(int).eq(int(right_cluster_id))
                & pair_df["base_cluster_right"].astype(int).eq(int(left_cluster_id))
            )
        )
    ].copy()
    if subset.empty:
        return ""
    image_ids = []
    for value in subset["image_id"].astype(str).tolist() + subset["neighbor_image_id"].astype(str).tolist():
        if value not in image_ids:
            image_ids.append(value)
    return "|".join(image_ids[:8])


def _candidate_kind_for_merge(pair_df: pd.DataFrame, left_cluster_id: int, right_cluster_id: int) -> str:
    subset = pair_df[
        pair_df["vote_direction"].astype(str).eq("merge")
        & (
            (
                pair_df["base_cluster_left"].astype(int).eq(int(left_cluster_id))
                & pair_df["base_cluster_right"].astype(int).eq(int(right_cluster_id))
            )
            | (
                pair_df["base_cluster_left"].astype(int).eq(int(right_cluster_id))
                & pair_df["base_cluster_right"].astype(int).eq(int(left_cluster_id))
            )
        )
    ].copy()
    if subset.empty:
        return "merge"
    if "same_pseudo_seed" in subset.columns and bool(subset["same_pseudo_seed"].fillna(False).any()):
        return "seed_bridge"
    if "same_teacher_anchor" in subset.columns and bool(subset["same_teacher_anchor"].fillna(False).any()):
        return "teacher_anchor_merge"
    if "same_teacher_fusion" in subset.columns and bool(subset["same_teacher_fusion"].fillna(False).any()):
        return "fusion_merge"
    return "merge"


def _pair_key(image_id: object, neighbor_image_id: object) -> str:
    left, right = sorted([str(image_id), str(neighbor_image_id)])
    return f"{left}|{right}"


def _load_texas_manual_judgments(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "candidate_type",
                "candidate_key",
                "pair_key",
                "image_id",
                "neighbor_image_id",
                "label",
            ]
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rows = payload.get("pair_judgments", [])
    if not isinstance(raw_rows, list):
        return pd.DataFrame(columns=["dataset", "candidate_type", "candidate_key", "pair_key", "image_id", "neighbor_image_id", "label"])
    frame = pd.DataFrame(raw_rows)
    if frame.empty:
        return pd.DataFrame(columns=["dataset", "candidate_type", "candidate_key", "pair_key", "image_id", "neighbor_image_id", "label"])
    for column in ["dataset", "candidate_type", "candidate_key", "pair_key", "image_id", "neighbor_image_id", "label"]:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    subset = frame[frame["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if subset.empty:
        return subset
    subset["pair_key"] = subset.apply(lambda row: _pair_key(row["image_id"], row["neighbor_image_id"]), axis=1)
    return subset


def _build_yes_component_lookup(judgment_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, int], dict[str, list[str]], dict[str, int]]:
    yes_df = judgment_df[judgment_df["label"].astype(str).eq("yes")].copy()
    if yes_df.empty:
        return {}, {}, {}, {}
    adjacency: dict[str, set[str]] = defaultdict(set)
    for row in yes_df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        adjacency[left].add(right)
        adjacency[right].add(left)
    image_to_component: dict[str, str] = {}
    component_sizes: dict[str, int] = {}
    component_members: dict[str, list[str]] = {}
    image_yes_degree = {image_id: len(neighbors) for image_id, neighbors in adjacency.items()}
    component_index = 0
    for start_image in sorted(adjacency.keys()):
        if start_image in image_to_component:
            continue
        queue: deque[str] = deque([start_image])
        members: list[str] = []
        while queue:
            current = queue.popleft()
            if current in image_to_component:
                continue
            image_to_component[current] = f"comp_{component_index:03d}"
            members.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in image_to_component:
                    queue.append(neighbor)
        members = sorted(set(members))
        component_key = image_to_component[start_image]
        component_sizes[component_key] = len(members)
        component_members[component_key] = members
        component_index += 1
    return image_to_component, component_sizes, component_members, image_yes_degree


def _normalize_local_support(frame: pd.DataFrame) -> pd.Series:
    if "black_orb_local_score" in frame.columns:
        orb_score = pd.to_numeric(frame["black_orb_local_score"], errors="coerce").fillna(0.0)
    else:
        orb_score = pd.Series(0.0, index=frame.index, dtype=np.float32)
    if "black_patch_support_score_v1" in frame.columns:
        patch_score = pd.to_numeric(frame["black_patch_support_score_v1"], errors="coerce").fillna(0.0)
    else:
        patch_score = pd.Series(0.0, index=frame.index, dtype=np.float32)
    patch_norm = np.clip((patch_score.to_numpy(dtype=np.float32) - 0.28) / 0.30, 0.0, 1.0)
    orb_norm = np.clip(orb_score.to_numpy(dtype=np.float32), 0.0, 1.0)
    combined = np.maximum(orb_norm, patch_norm)
    return pd.Series(np.round(combined, 6), index=frame.index)


def _topk_neighbor_indices(score_matrix: np.ndarray, index: int, topk: int) -> list[int]:
    row = np.asarray(score_matrix[int(index)], dtype=np.float32)
    order = np.argsort(-row)
    neighbors: list[int] = []
    for candidate_index in order.tolist():
        if int(candidate_index) == int(index):
            continue
        neighbors.append(int(candidate_index))
        if len(neighbors) >= int(topk):
            break
    return neighbors


def _build_yes_seed_expansion_pairs(
    *,
    base_df: pd.DataFrame,
    student_score: np.ndarray,
    miew_score: np.ndarray,
    fusion_score: np.ndarray,
    teacher_anchor_labels: np.ndarray,
    fusion_labels: np.ndarray,
    pseudo_seed_labels: np.ndarray,
    manual_judgments_df: pd.DataFrame,
    topk_per_member: int,
    max_outsiders_per_component: int,
    max_pairs_per_outsider: int,
) -> pd.DataFrame:
    judged_pair_keys = (
        set(manual_judgments_df["pair_key"].astype(str).tolist())
        if not manual_judgments_df.empty and "pair_key" in manual_judgments_df.columns
        else set()
    )
    _, _, component_members, _ = _build_yes_component_lookup(manual_judgments_df)
    if not component_members:
        return pd.DataFrame()

    image_id_to_index = {
        str(image_id): int(index) for index, image_id in enumerate(base_df["image_id"].astype(str).tolist())
    }
    component_image_sets = {component_key: set(members) for component_key, members in component_members.items()}
    pair_rows: list[dict[str, Any]] = []

    for component_key, member_image_ids in component_members.items():
        member_indices = [
            image_id_to_index[str(image_id)]
            for image_id in member_image_ids
            if str(image_id) in image_id_to_index
        ]
        if not member_indices:
            continue

        outsider_scores: dict[int, dict[str, Any]] = {}
        anchor_clusters = {int(teacher_anchor_labels[index]) for index in member_indices}
        fusion_clusters = {int(fusion_labels[index]) for index in member_indices}
        pseudo_clusters = {
            int(pseudo_seed_labels[index])
            for index in member_indices
            if int(pseudo_seed_labels[index]) >= 0
        }

        def _bump_outsider(outsider_index: int, reason_tag: str, count_delta: float) -> None:
            outsider_image_id = str(base_df.iloc[int(outsider_index)]["image_id"])
            if outsider_image_id in component_image_sets[component_key]:
                return
            entry = outsider_scores.setdefault(
                int(outsider_index),
                {"reason_tags": set(), "neighbor_vote_score": 0.0},
            )
            entry["reason_tags"].add(str(reason_tag))
            entry["neighbor_vote_score"] = float(entry["neighbor_vote_score"]) + float(count_delta)

        for member_index in member_indices:
            for matrix_name, score_matrix in [
                ("student_topk", student_score),
                ("miew_topk", miew_score),
                ("fusion_topk", fusion_score),
            ]:
                for outsider_index in _topk_neighbor_indices(score_matrix, member_index, topk=int(topk_per_member)):
                    _bump_outsider(outsider_index, matrix_name, 1.0)

        for outsider_index, outsider_row in enumerate(base_df.itertuples(index=False)):
            outsider_image_id = str(outsider_row.image_id)
            if outsider_image_id in component_image_sets[component_key]:
                continue
            if int(teacher_anchor_labels[outsider_index]) in anchor_clusters:
                _bump_outsider(outsider_index, "teacher_anchor_overlap", 2.0)
            if int(fusion_labels[outsider_index]) in fusion_clusters:
                _bump_outsider(outsider_index, "teacher_fusion_overlap", 1.8)
            if pseudo_clusters and int(pseudo_seed_labels[outsider_index]) in pseudo_clusters:
                _bump_outsider(outsider_index, "pseudo_seed_overlap", 1.5)

        ranked_outsiders: list[dict[str, Any]] = []
        for outsider_index, outsider_meta in outsider_scores.items():
            pair_signal_rows: list[dict[str, Any]] = []
            max_student = -1.0
            max_miew = -1.0
            max_fusion = -1.0
            for member_index in member_indices:
                student_similarity = float(student_score[int(member_index), int(outsider_index)])
                miew_similarity = float(miew_score[int(member_index), int(outsider_index)])
                fusion_similarity = float(fusion_score[int(member_index), int(outsider_index)])
                max_student = max(max_student, student_similarity)
                max_miew = max(max_miew, miew_similarity)
                max_fusion = max(max_fusion, fusion_similarity)
                pair_signal_rows.append(
                    {
                        "member_index": int(member_index),
                        "outsider_index": int(outsider_index),
                        "student_similarity": student_similarity,
                        "miew_similarity": miew_similarity,
                        "fusion_similarity": fusion_similarity,
                        "pair_signal": float(
                            1.3 * student_similarity
                            + 0.9 * miew_similarity
                            + 0.7 * fusion_similarity
                            + 0.15 * float(int(teacher_anchor_labels[member_index] == teacher_anchor_labels[outsider_index]))
                            + 0.10 * float(int(fusion_labels[member_index] == fusion_labels[outsider_index]))
                        ),
                    }
                )
            reason_tags = set(str(tag) for tag in outsider_meta["reason_tags"])
            if "teacher_anchor_overlap" in reason_tags and "teacher_fusion_overlap" in reason_tags:
                candidate_reason = "seed_component_dual_route_neighbor"
            elif "teacher_anchor_overlap" in reason_tags:
                candidate_reason = "seed_component_anchor_neighbor"
            elif "teacher_fusion_overlap" in reason_tags:
                candidate_reason = "seed_component_fusion_neighbor"
            elif "pseudo_seed_overlap" in reason_tags:
                candidate_reason = "seed_component_pseudo_neighbor"
            else:
                candidate_reason = "seed_component_topk_neighbor"
            outsider_score = (
                1.8 * max_student
                + 1.0 * max_miew
                + 0.8 * max_fusion
                + 0.18 * float(outsider_meta["neighbor_vote_score"])
                + 0.30 * float(int("teacher_anchor_overlap" in reason_tags))
                + 0.25 * float(int("teacher_fusion_overlap" in reason_tags))
                + 0.20 * float(int("pseudo_seed_overlap" in reason_tags))
            )
            ranked_outsiders.append(
                {
                    "outsider_index": int(outsider_index),
                    "candidate_reason": candidate_reason,
                    "outsider_score": float(outsider_score),
                    "pair_signal_rows": sorted(
                        pair_signal_rows,
                        key=lambda row: (row["pair_signal"], row["student_similarity"], row["miew_similarity"]),
                        reverse=True,
                    ),
                }
            )

        ranked_outsiders = sorted(
            ranked_outsiders,
            key=lambda row: (row["outsider_score"], len(row["pair_signal_rows"])),
            reverse=True,
        )[: int(max_outsiders_per_component)]

        for outsider_rank, outsider_info in enumerate(ranked_outsiders, start=1):
            outsider_index = int(outsider_info["outsider_index"])
            outsider_row = base_df.iloc[outsider_index]
            outsider_image_id = str(outsider_row["image_id"])
            candidate_key = f"seed_expand:{component_key}:{outsider_image_id}"
            for pair_rank, pair_info in enumerate(
                outsider_info["pair_signal_rows"][: int(max_pairs_per_outsider)],
                start=1,
            ):
                member_index = int(pair_info["member_index"])
                member_row = base_df.iloc[member_index]
                pair_key = _pair_key(str(member_row["image_id"]), outsider_image_id)
                if pair_key in judged_pair_keys:
                    continue
                pair_rows.append(
                    {
                        "left_index": int(member_index),
                        "right_index": int(outsider_index),
                        "image_id": str(member_row["image_id"]),
                        "neighbor_image_id": outsider_image_id,
                        "path": str(member_row["path"]),
                        "neighbor_path": str(outsider_row["path"]),
                        "dataset": TEXAS_DATASET,
                        "xgb_same_identity_prob": round(float(pair_info["student_similarity"]), 6),
                        "local_score": round(float(pair_info["miew_similarity"]), 6),
                        "route_global_score": round(float(pair_info["fusion_similarity"]), 6),
                        "same_student_selftrain": bool(
                            int(member_row["pred_cluster_id"]) == int(outsider_row["pred_cluster_id"])
                        ),
                        "same_teacher_anchor": bool(
                            int(teacher_anchor_labels[member_index]) == int(teacher_anchor_labels[outsider_index])
                        ),
                        "same_teacher_fusion": bool(
                            int(fusion_labels[member_index]) == int(fusion_labels[outsider_index])
                        ),
                        "same_pseudo_seed": bool(
                            int(pseudo_seed_labels[member_index]) == int(pseudo_seed_labels[outsider_index])
                            and int(pseudo_seed_labels[member_index]) >= 0
                        ),
                        "merge_votes": int(
                            int(teacher_anchor_labels[member_index] == teacher_anchor_labels[outsider_index])
                            + int(fusion_labels[member_index] == fusion_labels[outsider_index])
                            + int(
                                int(pseudo_seed_labels[member_index]) == int(pseudo_seed_labels[outsider_index])
                                and int(pseudo_seed_labels[member_index]) >= 0
                            )
                        ),
                        "split_votes": 0,
                        "ambiguity_score": round(float(min(max(pair_info["student_similarity"], 0.0), 1.0)), 6),
                        "vote_direction": "seed_expand",
                        "base_cluster_left": int(member_row["pred_cluster_id"]),
                        "base_cluster_right": int(outsider_row["pred_cluster_id"]),
                        "candidate_type": "yes",
                        "pair_key": pair_key,
                        "source_candidate_type": "seed_expand",
                        "source_candidate_key": str(component_key),
                        "candidate_key": candidate_key,
                        "seed_component_key": str(component_key),
                        "seed_expand_reason": str(outsider_info["candidate_reason"]),
                        "seed_expand_score": round(float(outsider_info["outsider_score"]), 6),
                        "seed_candidate_rank": int(outsider_rank),
                        "seed_pair_rank": int(pair_rank),
                    }
                )

    if not pair_rows:
        return pd.DataFrame()
    return pd.DataFrame(pair_rows).drop_duplicates(
        subset=["candidate_key", "pair_key"],
        keep="first",
    ).reset_index(drop=True)


def _build_texas_yes_review_candidates(
    *,
    pair_disagreement_df: pd.DataFrame,
    pairwise_feature_csv: Path | None,
    manual_judgments_df: pd.DataFrame,
    seed_expansion_df: pd.DataFrame | None,
    top_pairs_per_candidate: int,
    top_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_pair_df = pair_disagreement_df.copy().reset_index(drop=True)
    base_pair_df["pair_key"] = base_pair_df.apply(lambda row: _pair_key(row["image_id"], row["neighbor_image_id"]), axis=1)

    judged_pair_keys = set(manual_judgments_df["pair_key"].astype(str).tolist()) if not manual_judgments_df.empty else set()
    image_to_component, component_sizes, component_members, image_yes_degree = _build_yes_component_lookup(manual_judgments_df)
    pair_feature_df = pd.DataFrame()
    if pairwise_feature_csv is not None and pairwise_feature_csv.exists():
        pair_feature_df = pd.read_csv(pairwise_feature_csv).copy()
        pair_feature_df["pair_key"] = pair_feature_df.apply(lambda row: _pair_key(row["image_id"], row["neighbor_image_id"]), axis=1)
        keep_columns = [
            "pair_key",
            "route_global_score",
            "mutual_topk_all_routes",
            "black_orb_local_score",
            "black_orb_inliers",
            "black_orb_support_flag",
            "black_orb_veto_flag",
            "black_patch_support_score_v1",
            "black_patch_veto_score_v1",
            "black_patch_support_flag_v1",
            "black_patch_veto_flag_v1",
            "same_teacher_anchor",
        ]
        pair_feature_df = pair_feature_df[[column for column in keep_columns if column in pair_feature_df.columns]].drop_duplicates("pair_key")
        base_pair_df = base_pair_df.merge(pair_feature_df, on="pair_key", how="left")

    if "source_candidate_type" not in base_pair_df.columns:
        base_pair_df["source_candidate_type"] = base_pair_df["vote_direction"].astype(str)
    if "source_candidate_key" not in base_pair_df.columns:
        base_pair_df["source_candidate_key"] = np.where(
            base_pair_df["vote_direction"].astype(str).eq("split"),
            base_pair_df["base_cluster_left"].astype(int).astype(str),
            base_pair_df.apply(
                lambda row: f"{min(int(row['base_cluster_left']), int(row['base_cluster_right']))}|{max(int(row['base_cluster_left']), int(row['base_cluster_right']))}",
                axis=1,
            ),
        )
    if "candidate_key" not in base_pair_df.columns:
        base_pair_df["candidate_key"] = (
            base_pair_df["source_candidate_type"].astype(str)
            + ":"
            + base_pair_df["source_candidate_key"].astype(str)
        )
    base_pair_df = base_pair_df[~base_pair_df["pair_key"].astype(str).isin(judged_pair_keys)].copy().reset_index(drop=True)
    if base_pair_df.empty:
        empty_candidate_df = pd.DataFrame(
            columns=[
                "candidate_key",
                "dataset",
                "source_candidate_type",
                "source_candidate_key",
                "candidate_kind",
                "candidate_preview",
                "priority_score",
                "pair_count",
                "unique_image_count",
                "triangle_pair_count",
                "extension_pair_count",
                "local_supported_pair_count",
                "image_ids",
                "top_pair_keys",
            ]
        )
        empty_pair_df = pd.DataFrame(columns=list(pair_disagreement_df.columns) + ["pair_key"])
        summary = {
            "yes_candidate_count": 0,
            "yes_pair_count": 0,
            "judged_yes_components": int(len(component_sizes)),
        }
        return empty_candidate_df, empty_pair_df, summary

    base_pair_df["existing_yes_component_left"] = base_pair_df["image_id"].map(image_to_component).fillna("")
    base_pair_df["existing_yes_component_right"] = base_pair_df["neighbor_image_id"].map(image_to_component).fillna("")
    base_pair_df["same_existing_yes_component"] = (
        base_pair_df["existing_yes_component_left"].astype(str).ne("")
        & base_pair_df["existing_yes_component_left"].astype(str).eq(base_pair_df["existing_yes_component_right"].astype(str))
    )
    base_pair_df["extends_existing_yes_component"] = (
        base_pair_df["existing_yes_component_left"].astype(str).ne(base_pair_df["existing_yes_component_right"].astype(str))
        & (
            base_pair_df["existing_yes_component_left"].astype(str).ne("")
            | base_pair_df["existing_yes_component_right"].astype(str).ne("")
        )
    )
    base_pair_df["bridge_existing_yes_components"] = (
        base_pair_df["existing_yes_component_left"].astype(str).ne("")
        & base_pair_df["existing_yes_component_right"].astype(str).ne("")
        & base_pair_df["existing_yes_component_left"].astype(str).ne(base_pair_df["existing_yes_component_right"].astype(str))
    )
    base_pair_df["existing_yes_component_size"] = base_pair_df.apply(
        lambda row: max(
            int(component_sizes.get(str(row["existing_yes_component_left"]), 0)),
            int(component_sizes.get(str(row["existing_yes_component_right"]), 0)),
        ),
        axis=1,
    )
    base_pair_df["existing_yes_degree_max"] = base_pair_df.apply(
        lambda row: max(
            int(image_yes_degree.get(str(row["image_id"]), 0)),
            int(image_yes_degree.get(str(row["neighbor_image_id"]), 0)),
        ),
        axis=1,
    )
    base_pair_df["local_support_score"] = _normalize_local_support(base_pair_df)
    probability = pd.to_numeric(base_pair_df["xgb_same_identity_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    mid_global_bonus = np.clip(1.0 - np.abs(probability - 0.56) / 0.22, 0.0, 1.0)
    ambiguity_bonus = np.clip(
        pd.to_numeric(base_pair_df["ambiguity_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) / 0.95,
        0.0,
        1.0,
    )
    mutual_bonus = (
        base_pair_df["mutual_topk_all_routes"].fillna(0).astype(int).to_numpy(dtype=np.float32)
        if "mutual_topk_all_routes" in base_pair_df.columns
        else np.zeros(len(base_pair_df), dtype=np.float32)
    )
    orb_flag = (
        base_pair_df["black_orb_support_flag"].fillna(0).astype(int).to_numpy(dtype=np.float32)
        if "black_orb_support_flag" in base_pair_df.columns
        else np.zeros(len(base_pair_df), dtype=np.float32)
    )
    patch_flag = (
        base_pair_df["black_patch_support_flag_v1"].fillna(0).astype(int).to_numpy(dtype=np.float32)
        if "black_patch_support_flag_v1" in base_pair_df.columns
        else np.zeros(len(base_pair_df), dtype=np.float32)
    )
    same_component_bonus = base_pair_df["same_existing_yes_component"].astype(int).to_numpy(dtype=np.float32)
    extend_component_bonus = base_pair_df["extends_existing_yes_component"].astype(int).to_numpy(dtype=np.float32)
    bridge_penalty = base_pair_df["bridge_existing_yes_components"].astype(int).to_numpy(dtype=np.float32)
    split_bonus = base_pair_df["vote_direction"].astype(str).eq("split").astype(np.float32).to_numpy()
    score = (
        3.0 * same_component_bonus
        + 2.1 * extend_component_bonus
        + 1.2 * split_bonus
        + 1.0 * base_pair_df["local_support_score"].to_numpy(dtype=np.float32)
        + 0.65 * mid_global_bonus
        + 0.45 * ambiguity_bonus
        + 0.30 * mutual_bonus
        + 0.20 * orb_flag
        + 0.15 * patch_flag
        - 0.75 * bridge_penalty
    )
    base_pair_df["yes_priority_score"] = np.round(score.astype(np.float32), 6)

    candidate_reason = np.full(len(base_pair_df), "local_supported_followup", dtype=object)
    split_mask = base_pair_df["vote_direction"].astype(str).eq("split").to_numpy()
    merge_mask = base_pair_df["vote_direction"].astype(str).eq("merge").to_numpy()
    local_support = base_pair_df["local_support_score"].to_numpy(dtype=np.float32)
    candidate_reason[same_component_bonus.astype(bool)] = "triangle_completion"
    candidate_reason[(~same_component_bonus.astype(bool)) & extend_component_bonus.astype(bool) & split_mask] = "split_component_extension"
    candidate_reason[(~same_component_bonus.astype(bool)) & extend_component_bonus.astype(bool) & (~split_mask)] = "component_extension"
    candidate_reason[
        (~same_component_bonus.astype(bool))
        & (~extend_component_bonus.astype(bool))
        & split_mask
        & (local_support >= 0.50)
        & (probability >= 0.30)
        & (probability <= 0.78)
    ] = "split_local_hard_yes"
    candidate_reason[
        (~same_component_bonus.astype(bool))
        & (~extend_component_bonus.astype(bool))
        & merge_mask
        & (local_support >= 0.58)
        & (probability >= 0.28)
        & (probability <= 0.72)
    ] = "merge_local_hard_yes"
    candidate_reason[
        (~same_component_bonus.astype(bool))
        & (~extend_component_bonus.astype(bool))
        & orb_flag.astype(bool)
        & (probability <= 0.70)
    ] = "orb_supported_hard_yes"
    candidate_reason[
        (~same_component_bonus.astype(bool))
        & (~extend_component_bonus.astype(bool))
        & patch_flag.astype(bool)
        & (probability <= 0.72)
    ] = "patch_supported_hard_yes"
    base_pair_df["yes_candidate_reason"] = candidate_reason

    keep_mask = (
        base_pair_df["same_existing_yes_component"]
        | base_pair_df["extends_existing_yes_component"]
        | (
            base_pair_df["vote_direction"].astype(str).eq("split")
            & (base_pair_df["local_support_score"].astype(float) >= 0.42)
            & pd.to_numeric(base_pair_df["xgb_same_identity_prob"], errors="coerce").fillna(0.0).between(0.25, 0.82)
        )
        | (
            base_pair_df["vote_direction"].astype(str).eq("merge")
            & (base_pair_df["local_support_score"].astype(float) >= 0.52)
            & pd.to_numeric(base_pair_df["xgb_same_identity_prob"], errors="coerce").fillna(0.0).between(0.25, 0.74)
        )
    )
    filtered_pair_df = base_pair_df[keep_mask].copy().reset_index(drop=True)
    if seed_expansion_df is not None and not seed_expansion_df.empty:
        expansion_df = seed_expansion_df.copy().reset_index(drop=True)
        if "pair_key" not in expansion_df.columns:
            expansion_df["pair_key"] = expansion_df.apply(
                lambda row: _pair_key(row["image_id"], row["neighbor_image_id"]),
                axis=1,
            )
        expansion_df = expansion_df[~expansion_df["pair_key"].astype(str).isin(judged_pair_keys)].copy().reset_index(drop=True)
        if not pair_feature_df.empty:
            overlapping_columns = [
                column for column in pair_feature_df.columns if column != "pair_key" and column in expansion_df.columns
            ]
            if overlapping_columns:
                expansion_df = expansion_df.drop(columns=overlapping_columns)
            expansion_df = expansion_df.merge(pair_feature_df, on="pair_key", how="left")
        expansion_df["existing_yes_component_left"] = expansion_df["image_id"].map(image_to_component).fillna("")
        expansion_df["existing_yes_component_right"] = expansion_df["neighbor_image_id"].map(image_to_component).fillna("")
        expansion_df["same_existing_yes_component"] = (
            expansion_df["existing_yes_component_left"].astype(str).ne("")
            & expansion_df["existing_yes_component_left"].astype(str).eq(expansion_df["existing_yes_component_right"].astype(str))
        )
        expansion_df["extends_existing_yes_component"] = (
            expansion_df["existing_yes_component_left"].astype(str).ne(expansion_df["existing_yes_component_right"].astype(str))
            & (
                expansion_df["existing_yes_component_left"].astype(str).ne("")
                | expansion_df["existing_yes_component_right"].astype(str).ne("")
            )
        )
        expansion_df["bridge_existing_yes_components"] = (
            expansion_df["existing_yes_component_left"].astype(str).ne("")
            & expansion_df["existing_yes_component_right"].astype(str).ne("")
            & expansion_df["existing_yes_component_left"].astype(str).ne(expansion_df["existing_yes_component_right"].astype(str))
        )
        expansion_df["local_support_score"] = _normalize_local_support(expansion_df)
        expansion_df["yes_priority_score"] = np.round(
            pd.to_numeric(expansion_df.get("seed_expand_score", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            + 1.10 * pd.to_numeric(expansion_df["xgb_same_identity_prob"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            + 0.45 * pd.to_numeric(expansion_df["local_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            + 0.35 * pd.to_numeric(expansion_df["route_global_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            + 0.30 * expansion_df["same_teacher_anchor"].fillna(False).astype(np.float32).to_numpy()
            + 0.20 * expansion_df["same_teacher_fusion"].fillna(False).astype(np.float32).to_numpy(),
            6,
        )
        expansion_df["yes_candidate_reason"] = expansion_df.get("seed_expand_reason", "seed_component_neighbor")
        filtered_pair_df = (
            pd.concat([filtered_pair_df, expansion_df], ignore_index=True)
            .drop_duplicates(subset=["candidate_key", "pair_key"], keep="first")
            .reset_index(drop=True)
        )
    if filtered_pair_df.empty:
        return _build_texas_yes_review_candidates(
            pair_disagreement_df=pair_disagreement_df.head(0),
            pairwise_feature_csv=None,
            manual_judgments_df=manual_judgments_df.head(0),
            seed_expansion_df=None,
            top_pairs_per_candidate=top_pairs_per_candidate,
            top_candidates=top_candidates,
        )

    filtered_pair_df = filtered_pair_df.sort_values(
        ["yes_priority_score", "same_existing_yes_component", "extends_existing_yes_component", "local_support_score", "ambiguity_score"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    candidate_rows: list[dict[str, Any]] = []
    selected_pair_rows: list[pd.DataFrame] = []
    for candidate_rank, (candidate_key, group) in enumerate(filtered_pair_df.groupby("candidate_key", sort=False), start=1):
        group = group.sort_values(
            ["yes_priority_score", "same_existing_yes_component", "extends_existing_yes_component", "local_support_score", "ambiguity_score"],
            ascending=[False, False, False, False, False],
        ).head(int(top_pairs_per_candidate)).copy().reset_index(drop=True)
        image_ids: list[str] = []
        for value in group["image_id"].astype(str).tolist() + group["neighbor_image_id"].astype(str).tolist():
            if value not in image_ids:
                image_ids.append(value)
        reason_counts = Counter(group["yes_candidate_reason"].astype(str).tolist())
        top_reason = reason_counts.most_common(1)[0][0] if reason_counts else "local_supported_followup"
        candidate_priority = float(
            group["yes_priority_score"].head(3).sum()
            + 0.35 * min(len(image_ids), 4)
            + 0.50 * int(group["same_existing_yes_component"].any())
            + 0.35 * int(group["extends_existing_yes_component"].any())
        )
        preview = (
            f"{str(group['source_candidate_type'].iloc[0])} {str(group['source_candidate_key'].iloc[0])}"
            f" | top={top_reason}"
            f" | imgs={len(image_ids)}"
            f" | pairs={len(group)}"
        )
        candidate_rows.append(
            {
                "candidate_key": str(candidate_key),
                "dataset": TEXAS_DATASET,
                "source_candidate_type": str(group["source_candidate_type"].iloc[0]),
                "source_candidate_key": str(group["source_candidate_key"].iloc[0]),
                "candidate_kind": str(top_reason),
                "candidate_preview": preview,
                "priority_score": round(candidate_priority, 6),
                "pair_count": int(len(group)),
                "unique_image_count": int(len(image_ids)),
                "triangle_pair_count": int(group["same_existing_yes_component"].astype(bool).sum()),
                "extension_pair_count": int(group["extends_existing_yes_component"].astype(bool).sum()),
                "local_supported_pair_count": int((group["local_support_score"].astype(float) >= 0.42).sum()),
                "image_ids": "|".join(image_ids),
                "top_pair_keys": "|".join(group["pair_key"].astype(str).head(4).tolist()),
            }
        )
        group["candidate_type"] = "yes"
        selected_pair_rows.append(group)

    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["priority_score", "pair_count", "candidate_key"],
        ascending=[False, False, True],
    ).head(int(top_candidates)).reset_index(drop=True)
    allowed_candidate_keys = set(candidate_df["candidate_key"].astype(str).tolist())
    yes_pair_df = (
        pd.concat(selected_pair_rows, ignore_index=True)
        if selected_pair_rows
        else pd.DataFrame(columns=list(filtered_pair_df.columns) + ["candidate_type"])
    )
    yes_pair_df = yes_pair_df[yes_pair_df["candidate_key"].astype(str).isin(allowed_candidate_keys)].copy().reset_index(drop=True)
    yes_pair_df = yes_pair_df.sort_values(
        ["candidate_key", "yes_priority_score", "ambiguity_score", "xgb_same_identity_prob"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    if "component_id" not in yes_pair_df.columns:
        yes_pair_df["component_id"] = -1

    summary = {
        "yes_candidate_count": int(len(candidate_df)),
        "yes_pair_count": int(len(yes_pair_df)),
        "judged_yes_component_count": int(len(component_sizes)),
        "judged_yes_pair_count": int(manual_judgments_df["label"].astype(str).eq("yes").sum()) if not manual_judgments_df.empty else 0,
        "unjudged_source_pair_count": int(len(base_pair_df)),
        "filtered_yes_pair_count": int(len(filtered_pair_df)),
        "reason_counts": Counter(yes_pair_df["yes_candidate_reason"].astype(str).tolist()),
        "source_type_counts": Counter(candidate_df["source_candidate_type"].astype(str).tolist()) if not candidate_df.empty else Counter(),
        "component_hint_candidate_count": int(candidate_df["candidate_kind"].astype(str).isin(["triangle_completion", "split_component_extension", "component_extension"]).sum()) if not candidate_df.empty else 0,
    }
    return candidate_df, yes_pair_df, summary


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.ambiguity_clustering import (
        assign_ambiguity_components,
        build_pair_disagreement_table,
        summarize_merge_candidates,
        summarize_split_candidates,
    )
    from animalclef_analysis.texas_orb_local_probe import merge_texas_orb_local_scores

    parser = argparse.ArgumentParser(
        description="Build a manual-review-compatible Texas self-train review pack from the current self-train outputs."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument("--fusion-route-dir", type=Path, default=DEFAULT_FUSION_ROUTE_DIR)
    parser.add_argument("--miew-source-dir", type=Path, default=DEFAULT_MIEW_SOURCE_DIR)
    parser.add_argument("--fusion-source-dir", type=Path, default=DEFAULT_FUSION_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--local-match-csv", type=Path, default=DEFAULT_LOCAL_MATCH_CSV)
    parser.add_argument("--pairwise-feature-csv", type=Path, default=DEFAULT_PAIRWISE_FEATURE_CSV)
    parser.add_argument("--manual-judgments-path", type=Path, default=DEFAULT_MANUAL_JUDGMENTS_PATH)
    parser.add_argument("--min-ambiguity-score", type=float, default=0.45)
    parser.add_argument("--min-conflict-ratio", type=float, default=0.34)
    parser.add_argument("--min-merge-votes", type=int, default=2)
    parser.add_argument("--min-split-votes", type=int, default=2)
    parser.add_argument("--border-width", type=float, default=0.08)
    parser.add_argument("--yes-top-pairs-per-candidate", type=int, default=6)
    parser.add_argument("--yes-top-candidates", type=int, default=40)
    parser.add_argument("--yes-seed-neighbor-topk", type=int, default=12)
    parser.add_argument("--yes-seed-max-outsiders-per-component", type=int, default=10)
    parser.add_argument("--yes-seed-max-pairs-per-outsider", type=int, default=3)
    parser.add_argument("--report-top-k", type=int, default=20)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir if args.experiment_dir.is_absolute() else (repo_root / args.experiment_dir)
    fusion_route_dir = args.fusion_route_dir if args.fusion_route_dir.is_absolute() else (repo_root / args.fusion_route_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_predictions_path = experiment_dir / "tables" / "test_predictions_best_v1.csv"
    teacher_anchor_path = experiment_dir / "tables" / "teacher_anchor_predictions_v1.csv"
    pseudo_assignments_path = experiment_dir / "tables" / "pseudo_assignments_v1.csv"
    candidate_pairs_path = experiment_dir / "tables" / "candidate_pairs_v1.csv"
    student_embeddings_path = experiment_dir / "embeddings" / "test_embeddings.npy"
    fusion_predictions_path = _resolve_predictions_path(repo_root, fusion_route_dir)

    base_df = _load_texas_predictions(base_predictions_path)
    teacher_anchor_df = _align_frame(base_df, _load_texas_predictions(teacher_anchor_path), "teacher_anchor_predictions")
    fusion_df = _align_frame(base_df, _load_texas_predictions(fusion_predictions_path), "fusion_predictions")
    pseudo_df = _align_frame(base_df, pd.read_csv(pseudo_assignments_path), "pseudo_assignments")
    pseudo_df["is_seed"] = pseudo_df["is_seed"].fillna(False).astype(bool)
    pseudo_df["pseudo_label_index"] = pseudo_df["pseudo_label_index"].fillna(-1).astype(int)

    student_embeddings = np.load(student_embeddings_path).astype(np.float32)
    if len(student_embeddings) != len(base_df):
        raise ValueError(
            f"Student embedding rows mismatch: embeddings={len(student_embeddings)} vs base_df={len(base_df)}"
        )
    miew_embeddings = _load_source_embeddings(
        repo_root,
        source_dir=args.miew_source_dir,
        reference_df=base_df,
    )
    fusion_embeddings = _load_source_embeddings(
        repo_root,
        source_dir=args.fusion_source_dir,
        reference_df=base_df,
    )

    image_to_index = {str(image_id): index for index, image_id in enumerate(base_df["image_id"].astype(str).tolist())}

    candidate_pair_df = pd.read_csv(candidate_pairs_path).copy()
    candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
    candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)
    candidate_pair_df["left_index"] = candidate_pair_df["image_id"].map(image_to_index).astype(int)
    candidate_pair_df["right_index"] = candidate_pair_df["neighbor_image_id"].map(image_to_index).astype(int)
    candidate_pair_lookup = _candidate_lookup(candidate_pair_df)

    student_labels = base_df["pred_cluster_id"].to_numpy(dtype=np.int32)
    teacher_anchor_labels = teacher_anchor_df["pred_cluster_id"].to_numpy(dtype=np.int32)
    fusion_labels = fusion_df["pred_cluster_id"].to_numpy(dtype=np.int32)
    pseudo_seed_labels = _build_partial_seed_labels(pseudo_df)

    pair_keys: set[tuple[int, int]] = set(candidate_pair_lookup.keys())
    for labels in [student_labels, teacher_anchor_labels, fusion_labels, pseudo_seed_labels]:
        pair_keys |= _pair_keys_from_labels(labels)
    pair_keys = {key for key in pair_keys if key[0] != key[1]}

    student_score = _score_matrix(student_embeddings)
    miew_score = _score_matrix(miew_embeddings)
    fusion_score = _score_matrix(fusion_embeddings)

    pair_rows: list[dict[str, Any]] = []
    for left_index, right_index in sorted(pair_keys):
        left_row = base_df.iloc[int(left_index)]
        right_row = base_df.iloc[int(right_index)]
        extra = candidate_pair_lookup.get((int(left_index), int(right_index)), {})
        pair_rows.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "path": str(left_row["path"]),
                "neighbor_path": str(right_row["path"]),
                "dataset": TEXAS_DATASET,
                "xgb_same_identity_prob": round(float(student_score[left_index, right_index]), 6),
                "local_score": round(float(miew_score[left_index, right_index]), 6),
                "route_global_score": round(float(fusion_score[left_index, right_index]), 6),
                "same_student_selftrain": bool(student_labels[left_index] == student_labels[right_index]),
                "same_teacher_anchor": bool(teacher_anchor_labels[left_index] == teacher_anchor_labels[right_index]),
                "same_teacher_fusion": bool(fusion_labels[left_index] == fusion_labels[right_index]),
                "same_pseudo_seed": bool(pseudo_seed_labels[left_index] == pseudo_seed_labels[right_index]),
                **extra,
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    resolved_local_match_path: Path | None = None
    resolved_pairwise_feature_path: Path | None = None
    if args.local_match_csv:
        candidate_local_match_path = (
            args.local_match_csv if args.local_match_csv.is_absolute() else (repo_root / args.local_match_csv)
        ).resolve()
        if candidate_local_match_path.exists():
            resolved_local_match_path = candidate_local_match_path
            local_match_df = pd.read_csv(candidate_local_match_path)
            pair_df = merge_texas_orb_local_scores(pair_df=pair_df, local_match_df=local_match_df)
        else:
            print(f"[texas_selftrain_review_pack] local match csv not found, skip merge: {candidate_local_match_path}")
    if args.pairwise_feature_csv:
        candidate_pairwise_feature_path = (
            args.pairwise_feature_csv if args.pairwise_feature_csv.is_absolute() else (repo_root / args.pairwise_feature_csv)
        ).resolve()
        if candidate_pairwise_feature_path.exists():
            resolved_pairwise_feature_path = candidate_pairwise_feature_path
        else:
            print(f"[texas_selftrain_review_pack] pairwise feature csv not found, skip merge: {candidate_pairwise_feature_path}")

    student_distance_threshold = float(base_df["chosen_threshold"].astype(float).iloc[0])
    student_similarity_threshold = 1.0 - student_distance_threshold
    label_map = {
        "student_selftrain": student_labels,
        "teacher_anchor": teacher_anchor_labels,
        "teacher_fusion": fusion_labels,
        "pseudo_seed": pseudo_seed_labels,
    }
    pair_disagreement_df = build_pair_disagreement_table(
        pair_df,
        label_map=label_map,
        base_method="student_selftrain",
        probability_col="xgb_same_identity_prob",
        base_threshold=student_similarity_threshold,
        border_width=float(args.border_width),
    )
    pair_disagreement_df["same_teacher_anchor"] = (
        teacher_anchor_labels[pair_disagreement_df["left_index"].to_numpy(dtype=int)]
        == teacher_anchor_labels[pair_disagreement_df["right_index"].to_numpy(dtype=int)]
    )
    pair_disagreement_df["same_teacher_fusion"] = (
        fusion_labels[pair_disagreement_df["left_index"].to_numpy(dtype=int)]
        == fusion_labels[pair_disagreement_df["right_index"].to_numpy(dtype=int)]
    )
    pair_disagreement_df["same_pseudo_seed"] = (
        pseudo_seed_labels[pair_disagreement_df["left_index"].to_numpy(dtype=int)]
        == pseudo_seed_labels[pair_disagreement_df["right_index"].to_numpy(dtype=int)]
    )

    pair_disagreement_df, component_df = assign_ambiguity_components(
        pair_disagreement_df,
        min_ambiguity_score=float(args.min_ambiguity_score),
        min_conflict_ratio=float(args.min_conflict_ratio),
        probability_col="xgb_same_identity_prob",
    )
    merge_candidate_df = summarize_merge_candidates(
        pair_disagreement_df,
        base_labels=student_labels,
        probability_col="xgb_same_identity_prob",
        min_merge_votes=int(args.min_merge_votes),
    )
    split_candidate_df = summarize_split_candidates(
        pair_disagreement_df,
        base_labels=student_labels,
        probability_col="xgb_same_identity_prob",
        min_split_votes=int(args.min_split_votes),
    )

    if not merge_candidate_df.empty:
        merge_candidate_df["candidate_kind"] = [
            _candidate_kind_for_merge(pair_disagreement_df, int(row.left_cluster_id), int(row.right_cluster_id))
            for row in merge_candidate_df.itertuples(index=False)
        ]
        merge_candidate_df["support_image_ids"] = [
            _support_image_ids_for_merge(pair_disagreement_df, int(row.left_cluster_id), int(row.right_cluster_id))
            for row in merge_candidate_df.itertuples(index=False)
        ]
        merge_candidate_df["candidate_preview"] = [
            f"{int(row.left_cluster_id)}|{int(row.right_cluster_id)} | kind={str(kind)}"
            for row, kind in zip(merge_candidate_df.itertuples(index=False), merge_candidate_df["candidate_kind"].tolist(), strict=True)
        ]

    resolved_manual_judgments_path = (
        args.manual_judgments_path if args.manual_judgments_path.is_absolute() else (repo_root / args.manual_judgments_path)
    ).resolve()
    manual_judgments_df = _load_texas_manual_judgments(resolved_manual_judgments_path)
    seed_expansion_df = _build_yes_seed_expansion_pairs(
        base_df=base_df,
        student_score=student_score,
        miew_score=miew_score,
        fusion_score=fusion_score,
        teacher_anchor_labels=teacher_anchor_labels,
        fusion_labels=fusion_labels,
        pseudo_seed_labels=pseudo_seed_labels,
        manual_judgments_df=manual_judgments_df,
        topk_per_member=int(args.yes_seed_neighbor_topk),
        max_outsiders_per_component=int(args.yes_seed_max_outsiders_per_component),
        max_pairs_per_outsider=int(args.yes_seed_max_pairs_per_outsider),
    )
    yes_candidate_df, yes_pair_df, yes_summary = _build_texas_yes_review_candidates(
        pair_disagreement_df=pair_disagreement_df,
        pairwise_feature_csv=resolved_pairwise_feature_path,
        manual_judgments_df=manual_judgments_df,
        seed_expansion_df=seed_expansion_df,
        top_pairs_per_candidate=int(args.yes_top_pairs_per_candidate),
        top_candidates=int(args.yes_top_candidates),
    )

    pair_disagreement_df.to_csv(tables_dir / "test_pair_disagreement_v1.csv", index=False)
    merge_candidate_df.to_csv(tables_dir / "test_merge_candidates_v1.csv", index=False)
    split_candidate_df.to_csv(tables_dir / "test_split_candidates_v1.csv", index=False)
    yes_candidate_df.to_csv(tables_dir / "test_yes_candidates_v1.csv", index=False)
    yes_pair_df.to_csv(tables_dir / "test_yes_pair_candidates_v1.csv", index=False)
    component_df.to_csv(tables_dir / "ambiguity_components_v1.csv", index=False)

    summary = {
        "probe": "texas_selftrain_review_v1",
        "dataset": TEXAS_DATASET,
        "base_predictions_path": _path_ref(repo_root, base_predictions_path),
        "teacher_anchor_path": _path_ref(repo_root, teacher_anchor_path),
        "fusion_predictions_path": _path_ref(repo_root, fusion_predictions_path),
        "pseudo_assignments_path": _path_ref(repo_root, pseudo_assignments_path),
        "candidate_pairs_path": _path_ref(repo_root, candidate_pairs_path),
        "student_embeddings_path": _path_ref(repo_root, student_embeddings_path),
        "student_threshold_distance": round(student_distance_threshold, 6),
        "student_threshold_similarity": round(student_similarity_threshold, 6),
        "pair_count": int(len(pair_disagreement_df)),
        "component_count": int(len(component_df)),
        "split_candidate_count": int(len(split_candidate_df)),
        "merge_candidate_count": int(len(merge_candidate_df)),
        "yes_candidate_count": int(yes_summary["yes_candidate_count"]),
        "yes_pair_count": int(yes_summary["yes_pair_count"]),
        "seed_image_count": int(pseudo_df["is_seed"].sum()),
        "seed_cluster_count": int(pseudo_df.loc[pseudo_df["is_seed"], "pseudo_label_index"].nunique()),
        "manual_judgments_path": _path_ref(repo_root, resolved_manual_judgments_path) if resolved_manual_judgments_path.exists() else "",
        "judged_yes_pair_count": int(yes_summary["judged_yes_pair_count"]),
        "judged_yes_component_count": int(yes_summary["judged_yes_component_count"]),
        "component_hint_candidate_count": int(yes_summary["component_hint_candidate_count"]),
    }
    if resolved_local_match_path is not None and "orb_local_score" in pair_disagreement_df.columns:
        summary["local_match_csv"] = _path_ref(repo_root, resolved_local_match_path)
        summary["orb_local_pair_count"] = int(pair_disagreement_df["orb_local_score"].notna().sum())
        summary["orb_local_nonzero_pair_count"] = int(
            pair_disagreement_df["orb_local_score"].fillna(0.0).astype(float).gt(0.0).sum()
        )
    if resolved_pairwise_feature_path is not None:
        summary["pairwise_feature_csv"] = _path_ref(repo_root, resolved_pairwise_feature_path)
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _frame_to_markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return ""
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in frame.iterrows()
        ]
        return "\n".join([header, separator, *rows])

    split_preview = (
        _frame_to_markdown(split_candidate_df.head(int(args.report_top_k)))
        if not split_candidate_df.empty
        else "_No split candidates passed the current filters._"
    )
    merge_preview = (
        _frame_to_markdown(merge_candidate_df.head(int(args.report_top_k)))
        if not merge_candidate_df.empty
        else "_No merge candidates passed the current filters._"
    )
    yes_preview = (
        _frame_to_markdown(yes_candidate_df.head(int(args.report_top_k)))
        if not yes_candidate_df.empty
        else "_No yes candidates passed the current filters._"
    )
    yes_reason_lines = []
    reason_counts = yes_summary.get("reason_counts", Counter())
    for reason, count in reason_counts.most_common(6):
        yes_reason_lines.append(f"- `{reason}`: `{int(count)}`")
    if not yes_reason_lines:
        yes_reason_lines.append("- `_none_`")
    summary_md = "\n".join(
        [
            "# Texas Self-Train Review Pack v1",
            "",
            "## Goal",
            "",
            "- 用当前 `ft_texas_miew_pseudo_v1` 的 test 聚类结果作为 `base_predictions`。",
            "- 把 `teacher_anchor(miew@0.38)`、`teacher_fusion@0.43`、`pseudo seed` 当作弱证据源，生成可直接喂给人工复核台的 `pair / split / merge` candidate。",
            "",
            "## Inputs",
            "",
            f"- `base_predictions`: `{_path_ref(repo_root, base_predictions_path)}`",
            f"- `teacher_anchor_predictions`: `{_path_ref(repo_root, teacher_anchor_path)}`",
            f"- `fusion_predictions`: `{_path_ref(repo_root, fusion_predictions_path)}`",
            f"- `pseudo_assignments`: `{_path_ref(repo_root, pseudo_assignments_path)}`",
            f"- `candidate_pairs`: `{_path_ref(repo_root, candidate_pairs_path)}`",
            (
                f"- `local_match_csv`: `{_path_ref(repo_root, resolved_local_match_path)}`"
                if resolved_local_match_path is not None
                else "- `local_match_csv`: `_not provided_`"
            ),
            (
                f"- `pairwise_feature_csv`: `{_path_ref(repo_root, resolved_pairwise_feature_path)}`"
                if resolved_pairwise_feature_path is not None
                else "- `pairwise_feature_csv`: `_not provided_`"
            ),
            (
                f"- `manual_judgments_path`: `{_path_ref(repo_root, resolved_manual_judgments_path)}`"
                if resolved_manual_judgments_path.exists()
                else "- `manual_judgments_path`: `_not provided_`"
            ),
            "",
            "## Summary",
            "",
            f"- `student_threshold(distance)`: `{summary['student_threshold_distance']}`",
            f"- `student_threshold(similarity)`: `{summary['student_threshold_similarity']}`",
            f"- `pair_count`: `{summary['pair_count']}`",
            f"- `component_count`: `{summary['component_count']}`",
            f"- `split_candidate_count`: `{summary['split_candidate_count']}`",
            f"- `merge_candidate_count`: `{summary['merge_candidate_count']}`",
            f"- `yes_candidate_count`: `{summary['yes_candidate_count']}`",
            f"- `yes_pair_count`: `{summary['yes_pair_count']}`",
            f"- `seed_image_count`: `{summary['seed_image_count']}`",
            f"- `seed_cluster_count`: `{summary['seed_cluster_count']}`",
            f"- `judged_yes_pair_count`: `{summary['judged_yes_pair_count']}`",
            f"- `judged_yes_component_count`: `{summary['judged_yes_component_count']}`",
            f"- `component_hint_candidate_count`: `{summary['component_hint_candidate_count']}`",
            (
                f"- `orb_local_pair_count`: `{summary['orb_local_pair_count']}`"
                if "orb_local_pair_count" in summary
                else "- `orb_local_pair_count`: `0`"
            ),
            (
                f"- `orb_local_nonzero_pair_count`: `{summary['orb_local_nonzero_pair_count']}`"
                if "orb_local_nonzero_pair_count" in summary
                else "- `orb_local_nonzero_pair_count`: `0`"
            ),
            "",
            "## Launch",
            "",
            "```bash",
            "python scripts/launch_manual_review_workbench.py \\",
            f"  --base-predictions { _path_ref(repo_root, base_predictions_path) } \\",
            f"  --probe-dir { _path_ref(repo_root, output_dir) } \\",
            "  --host 127.0.0.1 \\",
            "  --port 7861",
            "```",
            "",
            "## Texas Yes-Candidate Ranking",
            "",
            "- 目标：让人工复核台优先加载 `最有机会补出 high-value yes` 的 Texas pair，而不是继续只看 `split/no`。",
            "- 候选源：当前 `test_pair_disagreement_v1`，并尽量合并 `texas_local_pairwise_probe_v1` 的局部特征。",
            "- 排序优先级：",
            "  1. 能补成现有人工 `yes` 小组件的 pair（triangle / extension）。",
            "  2. `split` 方向里全局中等、局部更强的 hard-yes。",
            "  3. `merge` 方向里局部更强、值得确认的 hard-yes。",
            "  4. 已人工审过的 pair 会自动排除。",
            "",
            "### Top Yes Reasons",
            "",
            *yes_reason_lines,
            "",
            "## Top Yes Candidates",
            "",
            yes_preview,
            "",
            "## Top Split Candidates",
            "",
            split_preview,
            "",
            "## Top Merge Candidates",
            "",
            merge_preview,
        ]
    )
    (reports_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[texas_selftrain_review_pack] summary: {reports_dir / 'summary.md'}")
    print(f"[texas_selftrain_review_pack] split_candidates: {tables_dir / 'test_split_candidates_v1.csv'}")
    print(f"[texas_selftrain_review_pack] merge_candidates: {tables_dir / 'test_merge_candidates_v1.csv'}")
    print(f"[texas_selftrain_review_pack] yes_candidates: {tables_dir / 'test_yes_candidates_v1.csv'}")
    print(f"[texas_selftrain_review_pack] yes_pair_candidates: {tables_dir / 'test_yes_pair_candidates_v1.csv'}")
    print(f"[texas_selftrain_review_pack] pair_disagreement: {tables_dir / 'test_pair_disagreement_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
