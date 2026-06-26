from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import (
    PATH_COLUMN,
    apply_thresholds_to_df,
    ensure_metadata_alignment,
    load_cached_embedding_bundle,
)


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_TOP_K = 8
DEFAULT_MIN_COMPONENT_DENSITY = 0.66
DEFAULT_MAX_SEED_CLUSTER_SIZE = 8
DEFAULT_TEXAS_ROUTE_CONFIGS = {
    "miew": {
        "source_dir": Path("artifacts/descriptor_baselines/embed_miew_v1"),
        "thresholds": [0.36, 0.38, 0.40, 0.42, 0.44],
        "anchor_threshold": 0.38,
    },
    "fusion": {
        "source_dir": Path("artifacts/descriptor_baselines/embed_fusion_v1"),
        "thresholds": [0.41, 0.43, 0.45, 0.47, 0.49],
        "anchor_threshold": 0.43,
    },
}


@dataclass(frozen=True)
class TexasRouteConfig:
    name: str
    source_dir: Path
    thresholds: list[float]
    anchor_threshold: float


@dataclass(frozen=True)
class TexasRouteBundle:
    name: str
    source_dir: Path
    df: pd.DataFrame
    embeddings: np.ndarray
    topk_indices: np.ndarray
    rank_matrix: np.ndarray
    score_matrix: np.ndarray


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def build_topk_indices(embeddings: np.ndarray, top_k: int) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros((0, 0), dtype=np.int32)
    width = min(top_k, len(embeddings) - 1)
    if width <= 0:
        return np.zeros((len(embeddings), 0), dtype=np.int32)
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    np.fill_diagonal(similarity, -np.inf)
    topk = np.argpartition(-similarity, kth=width - 1, axis=1)[:, :width]
    ordered = np.take_along_axis(
        topk,
        np.argsort(
            -np.take_along_axis(similarity, topk, axis=1),
            axis=1,
        ),
        axis=1,
    )
    return ordered.astype(np.int32, copy=False)


def build_rank_matrix(topk_indices: np.ndarray) -> np.ndarray:
    sample_count = int(topk_indices.shape[0])
    default_rank = int(topk_indices.shape[1] + 1)
    rank_matrix = np.full((sample_count, sample_count), default_rank, dtype=np.int16)
    for index in range(sample_count):
        rank_matrix[index, index] = 0
        for rank, neighbor_idx in enumerate(topk_indices[index].tolist(), start=1):
            rank_matrix[index, int(neighbor_idx)] = rank
    return rank_matrix


def mean_topk_neighbor_overlap(topk_left: np.ndarray, topk_right: np.ndarray) -> float:
    if topk_left.shape != topk_right.shape:
        raise ValueError(f"top-k shape mismatch: {topk_left.shape} vs {topk_right.shape}")
    if topk_left.size == 0:
        return 0.0
    overlaps: list[float] = []
    for left_row, right_row in zip(topk_left.tolist(), topk_right.tolist(), strict=True):
        left = set(int(value) for value in left_row)
        right = set(int(value) for value in right_row)
        union = left | right
        if not union:
            overlaps.append(0.0)
            continue
        overlaps.append(len(left & right) / len(union))
    return round(float(np.mean(overlaps)), 6) if overlaps else 0.0


def pair_agreement_score(labels_left: np.ndarray, labels_right: np.ndarray) -> float:
    labels_left = np.asarray(labels_left)
    labels_right = np.asarray(labels_right)
    if labels_left.shape != labels_right.shape:
        raise ValueError(f"Label shape mismatch: {labels_left.shape} vs {labels_right.shape}")
    if len(labels_left) < 2:
        return 1.0
    same_left = labels_left[:, None] == labels_left[None, :]
    same_right = labels_right[:, None] == labels_right[None, :]
    upper = np.triu_indices(len(labels_left), k=1)
    return round(float(np.mean((same_left[upper] == same_right[upper]).astype(np.float32))), 6)


def summarize_cluster_labels(labels: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return {
            "clusters": 0,
            "largest_cluster_size": 0,
            "singleton_clusters": 0,
            "singleton_ratio": 0.0,
            "non_singleton_images": 0,
            "non_singleton_image_ratio": 0.0,
            "p90_cluster_size": 0.0,
        }
    counts = pd.Series(labels).value_counts()
    non_singleton_mask = counts > 1
    non_singleton_images = int(counts[non_singleton_mask].sum()) if non_singleton_mask.any() else 0
    return {
        "clusters": int(len(counts)),
        "largest_cluster_size": int(counts.max()),
        "singleton_clusters": int((counts == 1).sum()),
        "singleton_ratio": round(float((counts == 1).mean()), 6),
        "non_singleton_images": non_singleton_images,
        "non_singleton_image_ratio": round(float(non_singleton_images / len(labels)), 6),
        "p90_cluster_size": round(float(np.percentile(counts.to_numpy(), 90)), 3),
    }


def load_texas_route_bundle(
    repo_root: Path,
    *,
    route_name: str,
    source_dir: Path,
    top_k: int = DEFAULT_TOP_K,
) -> TexasRouteBundle:
    del repo_root
    bundle = load_cached_embedding_bundle(source_dir=source_dir.resolve(), name=route_name)
    texas_df = bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    texas_embeddings = bundle.test_embeddings[(bundle.test_df["dataset"] == TEXAS_DATASET).to_numpy()]
    topk_indices = build_topk_indices(texas_embeddings, top_k=top_k)
    rank_matrix = build_rank_matrix(topk_indices)
    score_matrix = np.clip(texas_embeddings @ texas_embeddings.T, -1.0, 1.0)
    return TexasRouteBundle(
        name=route_name,
        source_dir=source_dir.resolve(),
        df=texas_df,
        embeddings=texas_embeddings,
        topk_indices=topk_indices,
        rank_matrix=rank_matrix,
        score_matrix=score_matrix,
    )


def build_route_config(name: str, repo_root: Path) -> TexasRouteConfig:
    config = DEFAULT_TEXAS_ROUTE_CONFIGS[name]
    return TexasRouteConfig(
        name=name,
        source_dir=(repo_root / config["source_dir"]).resolve(),
        thresholds=[float(value) for value in config["thresholds"]],
        anchor_threshold=float(config["anchor_threshold"]),
    )


def summarize_texas_threshold_candidates(
    bundle: TexasRouteBundle,
    *,
    thresholds: list[float],
    anchor_threshold: float,
    route_neighbor_overlap: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    anchor_labels: np.ndarray | None = None

    for threshold in thresholds:
        pred_df = apply_thresholds_to_df(
            df=bundle.df,
            embeddings=bundle.embeddings,
            threshold_by_dataset={TEXAS_DATASET: float(threshold)},
        )
        pred_df["route_name"] = bundle.name
        pred_df["threshold"] = float(threshold)
        pred_frames.append(pred_df)
        labels = pred_df["pred_cluster_id"].to_numpy()
        stats = summarize_cluster_labels(labels)
        summary_rows.append(
            {
                "route_name": bundle.name,
                "source_dir": str(bundle.source_dir),
                "threshold": float(threshold),
                "samples": int(len(pred_df)),
                **stats,
            }
        )
        if math.isclose(float(threshold), float(anchor_threshold), rel_tol=0.0, abs_tol=1e-9):
            anchor_labels = labels.copy()

    if anchor_labels is None:
        raise ValueError(f"Anchor threshold {anchor_threshold} not found in thresholds for route {bundle.name}")

    for row, pred_df in zip(summary_rows, pred_frames, strict=True):
        row["anchor_threshold"] = float(anchor_threshold)
        row["pair_agreement_vs_anchor"] = pair_agreement_score(
            pred_df["pred_cluster_id"].to_numpy(),
            anchor_labels,
        )
        row["mean_topk_overlap_vs_other_route"] = route_neighbor_overlap if route_neighbor_overlap is not None else np.nan

    summary_df = pd.DataFrame(summary_rows).sort_values(["route_name", "threshold"]).reset_index(drop=True)
    predictions_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return summary_df, predictions_df


def build_route_neighbor_overlap_table(route_bundles: list[TexasRouteBundle]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_index, left_bundle in enumerate(route_bundles):
        for right_index, right_bundle in enumerate(route_bundles):
            if left_index >= right_index:
                continue
            ensure_metadata_alignment(
                reference_df=left_bundle.df,
                candidate_df=right_bundle.df,
                split_name="texas_test",
                reference_name=left_bundle.name,
                candidate_name=right_bundle.name,
            )
            rows.append(
                {
                    "route_left": left_bundle.name,
                    "route_right": right_bundle.name,
                    "top_k": int(left_bundle.topk_indices.shape[1]),
                    "mean_neighbor_overlap": mean_topk_neighbor_overlap(left_bundle.topk_indices, right_bundle.topk_indices),
                }
            )
    return pd.DataFrame(rows).sort_values(["route_left", "route_right"]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=["route_left", "route_right", "top_k", "mean_neighbor_overlap"]
    )


def write_texas_threshold_sweep_summary(
    output_path: Path,
    *,
    summary_df: pd.DataFrame,
    route_overlap_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Texas Threshold Sweep",
        "",
        f"- Dataset: `{TEXAS_DATASET}`",
        f"- Routes: `{config['routes']}`",
        f"- Top-k: `{config['top_k']}`",
        "",
        "## Route Neighbor Overlap",
        "",
        dataframe_to_markdown_table(route_overlap_df),
        "",
        "## Candidate Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Reading Notes",
        "",
        "- `largest_cluster_size` should not explode toward the full `274` images.",
        "- `pair_agreement_vs_anchor` close to `1.0` means this threshold behaves almost the same as the route anchor.",
        "- `mean_topk_overlap_vs_other_route` is route-level embedding agreement, not a threshold-specific metric.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_texas_threshold_sweep(
    repo_root: Path,
    output_dir: Path,
    *,
    route_configs: list[TexasRouteConfig] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if route_configs is None:
        route_configs = [build_route_config("miew", repo_root), build_route_config("fusion", repo_root)]

    route_bundles = [
        load_texas_route_bundle(
            repo_root=repo_root,
            route_name=config.name,
            source_dir=config.source_dir,
            top_k=top_k,
        )
        for config in route_configs
    ]
    route_overlap_df = build_route_neighbor_overlap_table(route_bundles)
    overlap_lookup: dict[str, float] = {}
    for bundle in route_bundles:
        overlaps = []
        for row in route_overlap_df.itertuples(index=False):
            if row.route_left == bundle.name or row.route_right == bundle.name:
                overlaps.append(float(row.mean_neighbor_overlap))
        overlap_lookup[bundle.name] = round(float(np.mean(overlaps)), 6) if overlaps else np.nan

    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    for config, bundle in zip(route_configs, route_bundles, strict=True):
        summary_df, predictions_df = summarize_texas_threshold_candidates(
            bundle,
            thresholds=config.thresholds,
            anchor_threshold=config.anchor_threshold,
            route_neighbor_overlap=overlap_lookup.get(bundle.name),
        )
        summary_frames.append(summary_df)
        prediction_frames.append(predictions_df)

    summary_df = pd.concat(summary_frames, ignore_index=True).sort_values(["route_name", "threshold"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True).sort_values(["route_name", "threshold", "image_id"]).reset_index(drop=True)
    summary_path = reports_dir / "summary.md"
    summary_table_path = tables_dir / "texas_threshold_candidates_v1.csv"
    predictions_path = tables_dir / "texas_threshold_predictions_v1.csv"
    route_overlap_path = tables_dir / "route_neighbor_overlap_v1.csv"
    summary_df.to_csv(summary_table_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    route_overlap_df.to_csv(route_overlap_path, index=False)

    config_payload = {
        "routes": [config.name for config in route_configs],
        "top_k": top_k,
        "route_configs": [
            {
                "name": config.name,
                "source_dir": str(config.source_dir),
                "thresholds": config.thresholds,
                "anchor_threshold": config.anchor_threshold,
            }
            for config in route_configs
        ],
    }
    write_texas_threshold_sweep_summary(
        summary_path,
        summary_df=summary_df,
        route_overlap_df=route_overlap_df,
        config=config_payload,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                **config_payload,
                "summary_table_path": str(summary_table_path),
                "predictions_path": str(predictions_path),
                "route_overlap_path": str(route_overlap_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": summary_path,
        "summary_table_path": summary_table_path,
        "predictions_path": predictions_path,
        "route_overlap_path": route_overlap_path,
    }


def build_consensus_pair_table(
    route_bundles: list[TexasRouteBundle],
    route_predictions: dict[str, pd.DataFrame],
    *,
    top_k: int,
) -> pd.DataFrame:
    if not route_bundles:
        return pd.DataFrame()
    image_ids = route_bundles[0].df["image_id"].astype(str).tolist()
    rows: list[dict[str, object]] = []
    route_by_name = {bundle.name: bundle for bundle in route_bundles}
    route_names = [bundle.name for bundle in route_bundles]
    label_lookup = {
        name: route_predictions[name].set_index("image_id")["pred_cluster_id"].astype(int).to_dict()
        for name in route_names
    }
    path_lookup = route_bundles[0].df.set_index("image_id")[PATH_COLUMN].astype(str).to_dict()

    for left_index in range(len(image_ids)):
        left_id = image_ids[left_index]
        for right_index in range(left_index + 1, len(image_ids)):
            right_id = image_ids[right_index]
            same_cluster_all_routes = all(
                int(label_lookup[name][left_id]) == int(label_lookup[name][right_id])
                for name in route_names
            )
            if not same_cluster_all_routes:
                continue
            row: dict[str, object] = {
                "image_id": left_id,
                "neighbor_image_id": right_id,
                "path": path_lookup[left_id],
                "neighbor_path": path_lookup[right_id],
                "same_cluster_all_routes": True,
            }
            mutual_topk_all_routes = True
            for name in route_names:
                bundle = route_by_name[name]
                rank_forward = int(bundle.rank_matrix[left_index, right_index])
                rank_backward = int(bundle.rank_matrix[right_index, left_index])
                similarity = float(bundle.score_matrix[left_index, right_index])
                row[f"{name}_cluster_id"] = int(label_lookup[name][left_id])
                row[f"{name}_similarity"] = round(similarity, 6)
                row[f"{name}_rank_forward"] = rank_forward
                row[f"{name}_rank_backward"] = rank_backward
                row[f"{name}_mutual_topk"] = bool(rank_forward <= top_k and rank_backward <= top_k)
                mutual_topk_all_routes = mutual_topk_all_routes and bool(rank_forward <= top_k and rank_backward <= top_k)
            row["mutual_topk_all_routes"] = mutual_topk_all_routes
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True) if rows else pd.DataFrame()


def build_seed_clusters_from_candidate_pairs(
    *,
    image_ids: list[str],
    pair_df: pd.DataFrame,
    min_component_density: float = DEFAULT_MIN_COMPONENT_DENSITY,
    max_seed_cluster_size: int = DEFAULT_MAX_SEED_CLUSTER_SIZE,
) -> pd.DataFrame:
    pair_subset = pair_df[pair_df["mutual_topk_all_routes"]].copy() if not pair_df.empty else pd.DataFrame(columns=["image_id", "neighbor_image_id"])
    adjacency: dict[str, set[str]] = {str(image_id): set() for image_id in image_ids}
    for row in pair_subset.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        adjacency[left].add(right)
        adjacency[right].add(left)

    visited: set[str] = set()
    rows: list[dict[str, object]] = []
    cluster_index = 0
    for image_id in image_ids:
        if image_id in visited:
            continue
        queue: deque[str] = deque([str(image_id)])
        visited.add(str(image_id))
        component: list[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component_size = len(component)
        edge_count = int(sum(len(adjacency[node] & set(component)) for node in component) // 2)
        possible_edges = component_size * (component_size - 1) // 2
        density = float(edge_count / possible_edges) if possible_edges else 0.0
        is_seed = (
            component_size >= 2
            and component_size <= max_seed_cluster_size
            and density >= min_component_density
        )
        pseudo_identity = f"texas_seed_{cluster_index:04d}" if is_seed else ""
        if is_seed:
            cluster_index += 1
        for node in sorted(component):
            rows.append(
                {
                    "image_id": node,
                    "seed_status": "seed" if is_seed else "uncertain",
                    "pseudo_identity": pseudo_identity,
                    "component_size": component_size,
                    "component_density": round(density, 6),
                    "component_edge_count": edge_count,
                }
            )
    return pd.DataFrame(rows).sort_values(["seed_status", "pseudo_identity", "image_id"]).reset_index(drop=True)


def write_texas_pseudo_seed_summary(
    output_path: Path,
    *,
    summary_df: pd.DataFrame,
    seed_cluster_df: pd.DataFrame,
    candidate_pair_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    lines = [
        "# Texas Pseudo Seed",
        "",
        f"- Dataset: `{TEXAS_DATASET}`",
        f"- Routes: `{config['routes']}`",
        f"- Thresholds: `{config['thresholds']}`",
        f"- Top-k: `{config['top_k']}`",
        f"- Min component density: `{config['min_component_density']}`",
        f"- Max seed cluster size: `{config['max_seed_cluster_size']}`",
        "",
        "## Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Seed Clusters",
        "",
        dataframe_to_markdown_table(seed_cluster_df),
        "",
        "## Candidate Pair Summary",
        "",
        dataframe_to_markdown_table(
            pd.DataFrame(
                [
                    {
                        "candidate_pairs": int(len(candidate_pair_df)),
                        "mutual_topk_pairs": int(candidate_pair_df["mutual_topk_all_routes"].sum()) if not candidate_pair_df.empty else 0,
                    }
                ]
            )
        ),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_texas_pseudo_seed(
    repo_root: Path,
    output_dir: Path,
    *,
    route_configs: list[TexasRouteConfig] | None = None,
    top_k: int = DEFAULT_TOP_K,
    min_component_density: float = DEFAULT_MIN_COMPONENT_DENSITY,
    max_seed_cluster_size: int = DEFAULT_MAX_SEED_CLUSTER_SIZE,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if route_configs is None:
        route_configs = [
            TexasRouteConfig(name="miew", source_dir=(repo_root / "artifacts/descriptor_baselines/embed_miew_v1").resolve(), thresholds=[0.38], anchor_threshold=0.38),
            TexasRouteConfig(name="fusion", source_dir=(repo_root / "artifacts/descriptor_baselines/embed_fusion_v1").resolve(), thresholds=[0.43], anchor_threshold=0.43),
        ]

    route_bundles = [
        load_texas_route_bundle(
            repo_root=repo_root,
            route_name=config.name,
            source_dir=config.source_dir,
            top_k=top_k,
        )
        for config in route_configs
    ]
    for left, right in zip(route_bundles, route_bundles[1:], strict=False):
        ensure_metadata_alignment(
            reference_df=left.df,
            candidate_df=right.df,
            split_name="texas_test",
            reference_name=left.name,
            candidate_name=right.name,
        )

    route_predictions: dict[str, pd.DataFrame] = {}
    for config, bundle in zip(route_configs, route_bundles, strict=True):
        route_predictions[config.name] = apply_thresholds_to_df(
            df=bundle.df,
            embeddings=bundle.embeddings,
            threshold_by_dataset={TEXAS_DATASET: float(config.anchor_threshold)},
        )

    candidate_pair_df = build_consensus_pair_table(
        route_bundles=route_bundles,
        route_predictions=route_predictions,
        top_k=top_k,
    )
    image_ids = route_bundles[0].df["image_id"].astype(str).tolist()
    assignments_df = build_seed_clusters_from_candidate_pairs(
        image_ids=image_ids,
        pair_df=candidate_pair_df,
        min_component_density=min_component_density,
        max_seed_cluster_size=max_seed_cluster_size,
    )

    reference_df = route_bundles[0].df.copy().reset_index(drop=True)
    assignment_df = reference_df.merge(assignments_df, on="image_id", how="left")
    assignment_df["seed_status"] = assignment_df["seed_status"].fillna("uncertain")
    assignment_df["pseudo_identity"] = assignment_df["pseudo_identity"].fillna("")
    assignment_df["component_size"] = assignment_df["component_size"].fillna(1).astype(int)
    assignment_df["component_density"] = assignment_df["component_density"].fillna(0.0)
    assignment_df["component_edge_count"] = assignment_df["component_edge_count"].fillna(0).astype(int)
    assignment_df["pseudo_source_routes"] = "+".join(config.name for config in route_configs)

    pseudo_manifest_df = assignment_df[assignment_df["seed_status"] == "seed"].copy().reset_index(drop=True)
    uncertain_pool_df = assignment_df[assignment_df["seed_status"] != "seed"].copy().reset_index(drop=True)
    seed_cluster_df = (
        pseudo_manifest_df.groupby("pseudo_identity")
        .agg(
            size=("image_id", "count"),
            mean_density=("component_density", "mean"),
        )
        .reset_index()
        .sort_values(["size", "pseudo_identity"], ascending=[False, True])
        .reset_index(drop=True)
    ) if not pseudo_manifest_df.empty else pd.DataFrame(columns=["pseudo_identity", "size", "mean_density"])
    summary_df = pd.DataFrame(
        [
            {
                "total_images": int(len(reference_df)),
                "seed_images": int(len(pseudo_manifest_df)),
                "seed_coverage_ratio": round(float(len(pseudo_manifest_df) / max(len(reference_df), 1)), 6),
                "seed_clusters": int(seed_cluster_df["pseudo_identity"].nunique()) if not seed_cluster_df.empty else 0,
                "uncertain_images": int(len(uncertain_pool_df)),
            }
        ]
    )

    assignments_path = tables_dir / "all_assignments_v1.csv"
    pseudo_manifest_path = tables_dir / "pseudo_manifest_v1.csv"
    uncertain_pool_path = tables_dir / "uncertain_pool_v1.csv"
    pair_table_path = tables_dir / "candidate_pairs_v1.csv"
    cluster_summary_path = tables_dir / "seed_cluster_summary_v1.csv"
    assignment_df.to_csv(assignments_path, index=False)
    pseudo_manifest_df.to_csv(pseudo_manifest_path, index=False)
    uncertain_pool_df.to_csv(uncertain_pool_path, index=False)
    candidate_pair_df.to_csv(pair_table_path, index=False)
    seed_cluster_df.to_csv(cluster_summary_path, index=False)

    config_payload = {
        "routes": [config.name for config in route_configs],
        "thresholds": {config.name: config.anchor_threshold for config in route_configs},
        "top_k": top_k,
        "min_component_density": min_component_density,
        "max_seed_cluster_size": max_seed_cluster_size,
    }
    summary_path = reports_dir / "summary.md"
    write_texas_pseudo_seed_summary(
        summary_path,
        summary_df=summary_df,
        seed_cluster_df=seed_cluster_df,
        candidate_pair_df=candidate_pair_df,
        config=config_payload,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                **config_payload,
                "assignments_path": str(assignments_path),
                "pseudo_manifest_path": str(pseudo_manifest_path),
                "uncertain_pool_path": str(uncertain_pool_path),
                "candidate_pair_path": str(pair_table_path),
                "cluster_summary_path": str(cluster_summary_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": summary_path,
        "assignments_path": assignments_path,
        "pseudo_manifest_path": pseudo_manifest_path,
        "uncertain_pool_path": uncertain_pool_path,
        "candidate_pair_path": pair_table_path,
        "cluster_summary_path": cluster_summary_path,
    }
