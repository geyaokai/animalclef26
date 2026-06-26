from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .descriptor_baselines import l2_normalize
from .labeled_selftrain import build_stable_pseudo_seed_bundle


PSEUDO_SEED_FEATURE_COLUMNS = [
    "left_is_seeded",
    "right_is_seeded",
    "both_seeded",
    "one_seeded",
    "both_unseeded",
    "same_seed_cluster",
    "left_seed_cluster_size",
    "right_seed_cluster_size",
    "left_seed_mean_similarity",
    "right_seed_mean_similarity",
]


@dataclass(frozen=True)
class PseudoSeedFeatureBundle:
    assignment_df: pd.DataFrame
    cluster_summary_df: pd.DataFrame
    threshold_summary_df: pd.DataFrame
    pseudo_seed_df: pd.DataFrame
    summary_row: dict[str, float | int]


def build_pseudo_seed_feature_bundle(
    *,
    metadata_df: pd.DataFrame,
    teacher_embeddings: np.ndarray,
    anchor_threshold: float,
    stability_delta: float,
    min_seed_cluster_size: int,
    max_seed_cluster_size: int,
    min_mean_similarity: float,
) -> PseudoSeedFeatureBundle:
    normalized_embeddings = l2_normalize(teacher_embeddings.astype(np.float32, copy=False))
    bundle = build_stable_pseudo_seed_bundle(
        target_df=metadata_df.reset_index(drop=True).copy(),
        teacher_embeddings=normalized_embeddings,
        anchor_threshold=float(anchor_threshold),
        stability_delta=float(stability_delta),
        min_seed_cluster_size=int(min_seed_cluster_size),
        max_seed_cluster_size=int(max_seed_cluster_size),
        min_mean_similarity=float(min_mean_similarity),
    )

    cluster_df = bundle.cluster_summary_df.copy()
    cluster_df = cluster_df.rename(
        columns={
            "size": "seed_cluster_size",
            "mean_similarity": "seed_mean_similarity",
        }
    )
    assignment_df = bundle.pseudo_seed_df.copy()
    assignment_df["seed_cluster_id"] = assignment_df["pseudo_identity"].fillna("").astype(str)
    assignment_df["is_seeded"] = assignment_df["seed_status"].eq("seed").astype(int)
    assignment_df = assignment_df.merge(
        cluster_df[
            [
                "anchor_cluster_id",
                "seed_cluster_size",
                "seed_mean_similarity",
                "accepted_as_seed",
            ]
        ],
        on="anchor_cluster_id",
        how="left",
    )
    assignment_df["seed_cluster_size"] = np.where(
        assignment_df["is_seeded"].eq(1),
        assignment_df["seed_cluster_size"].fillna(0).astype(int),
        0,
    )
    assignment_df["seed_mean_similarity"] = np.where(
        assignment_df["is_seeded"].eq(1),
        assignment_df["seed_mean_similarity"].fillna(0.0).astype(np.float32),
        0.0,
    )
    assignment_df.loc[assignment_df["is_seeded"].eq(0), "seed_cluster_id"] = ""
    assignment_df["seed_cluster_id"] = assignment_df["seed_cluster_id"].astype(str)
    assignment_df["accepted_as_seed"] = assignment_df["accepted_as_seed"].fillna(False).astype(bool)

    seeded_df = assignment_df[assignment_df["is_seeded"].eq(1)].copy()
    summary_row = {
        "anchor_threshold": round(float(anchor_threshold), 4),
        "images": int(len(assignment_df)),
        "seed_images": int(len(seeded_df)),
        "seed_image_ratio": round(float(len(seeded_df) / len(assignment_df)) if len(assignment_df) else 0.0, 6),
        "seed_clusters": int(seeded_df["seed_cluster_id"].nunique()) if not seeded_df.empty else 0,
        "mean_seed_cluster_size": round(float(seeded_df["seed_cluster_size"].mean()) if not seeded_df.empty else 0.0, 6),
        "mean_seed_similarity": round(float(seeded_df["seed_mean_similarity"].mean()) if not seeded_df.empty else 0.0, 6),
        "teacher_anchor_ari": float(bundle.teacher_anchor_metrics["ari"]),
        "teacher_anchor_pairwise_f1": float(bundle.teacher_anchor_metrics["pairwise_f1"]),
    }
    return PseudoSeedFeatureBundle(
        assignment_df=assignment_df,
        cluster_summary_df=cluster_df,
        threshold_summary_df=bundle.threshold_summary_df.copy(),
        pseudo_seed_df=bundle.pseudo_seed_df.copy(),
        summary_row=summary_row,
    )


def append_pseudo_seed_pair_features(pair_df: pd.DataFrame, assignment_df: pd.DataFrame) -> pd.DataFrame:
    left_df = assignment_df[
        [
            "image_id",
            "is_seeded",
            "seed_cluster_id",
            "seed_cluster_size",
            "seed_mean_similarity",
        ]
    ].rename(
        columns={
            "is_seeded": "left_is_seeded",
            "seed_cluster_id": "left_seed_cluster_id",
            "seed_cluster_size": "left_seed_cluster_size",
            "seed_mean_similarity": "left_seed_mean_similarity",
        }
    )
    right_df = assignment_df[
        [
            "image_id",
            "is_seeded",
            "seed_cluster_id",
            "seed_cluster_size",
            "seed_mean_similarity",
        ]
    ].rename(
        columns={
            "image_id": "neighbor_image_id",
            "is_seeded": "right_is_seeded",
            "seed_cluster_id": "right_seed_cluster_id",
            "seed_cluster_size": "right_seed_cluster_size",
            "seed_mean_similarity": "right_seed_mean_similarity",
        }
    )
    merged = pair_df.merge(left_df, on="image_id", how="left").merge(right_df, on="neighbor_image_id", how="left")
    for column in [
        "left_is_seeded",
        "right_is_seeded",
        "left_seed_cluster_size",
        "right_seed_cluster_size",
        "left_seed_mean_similarity",
        "right_seed_mean_similarity",
    ]:
        merged[column] = merged[column].fillna(0)
    merged["left_is_seeded"] = merged["left_is_seeded"].astype(int)
    merged["right_is_seeded"] = merged["right_is_seeded"].astype(int)
    merged["left_seed_cluster_size"] = merged["left_seed_cluster_size"].astype(int)
    merged["right_seed_cluster_size"] = merged["right_seed_cluster_size"].astype(int)
    merged["left_seed_mean_similarity"] = merged["left_seed_mean_similarity"].astype(np.float32)
    merged["right_seed_mean_similarity"] = merged["right_seed_mean_similarity"].astype(np.float32)
    merged["left_seed_cluster_id"] = merged["left_seed_cluster_id"].fillna("").astype(str)
    merged["right_seed_cluster_id"] = merged["right_seed_cluster_id"].fillna("").astype(str)
    merged["both_seeded"] = (
        merged["left_is_seeded"].eq(1) & merged["right_is_seeded"].eq(1)
    ).astype(int)
    merged["one_seeded"] = (
        merged["left_is_seeded"].eq(1) ^ merged["right_is_seeded"].eq(1)
    ).astype(int)
    merged["both_unseeded"] = (
        merged["left_is_seeded"].eq(0) & merged["right_is_seeded"].eq(0)
    ).astype(int)
    merged["same_seed_cluster"] = (
        merged["both_seeded"].eq(1) & merged["left_seed_cluster_id"].eq(merged["right_seed_cluster_id"])
    ).astype(int)
    return merged
