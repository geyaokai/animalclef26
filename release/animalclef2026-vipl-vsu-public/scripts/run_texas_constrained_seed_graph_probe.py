#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"


class PairUnionFind:
    def __init__(self, nodes: list[int], seed_clusters: set[int]) -> None:
        self.parent: dict[int, int] = {int(node): int(node) for node in nodes}
        self.seed_sets: dict[int, set[int]] = {
            int(node): ({int(node)} if int(node) in seed_clusters else set()) for node in nodes
        }

    def find(self, item: int) -> int:
        key = int(item)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: int, right: int) -> bool:
        root_left = self.find(int(left))
        root_right = self.find(int(right))
        if root_left == root_right:
            return False
        left_seed_set = self.seed_sets[root_left]
        right_seed_set = self.seed_sets[root_right]
        if left_seed_set and right_seed_set and left_seed_set != right_seed_set:
            return False
        self.parent[root_right] = root_left
        self.seed_sets[root_left] = left_seed_set.union(right_seed_set)
        return True


class StringUnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        key = str(item)
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Probe constrained seed-graph clustering on the current best Texas route.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_seeded_attach_on_059341_base_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--route-dir",
        type=Path,
        default=repo_root / "artifacts/training/experiments/ft_texas_miew_tcuwarmup_trusted_views_v1",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v2/texas_pair_registry_v2.csv",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=repo_root / "sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_constrained_seed_graph_probe_v1",
    )
    parser.add_argument(
        "--submission-output-dir",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_constrained_seed_graph_on_bestpublic_v1",
    )
    parser.add_argument("--route-name", type=str, default="texas_constrained_seed_graph_v1")
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--member-score-threshold", type=float, default=0.70)
    parser.add_argument("--mean-top-thresholds", nargs="+", type=float, default=[0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72])
    parser.add_argument("--min-support-values", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--allow-nonseed-nonseed-options", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--max-cluster-size-values", nargs="+", type=int, default=[1, 2])
    return parser.parse_args()


def _load_embeddings(route_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_all = pd.read_csv(route_dir / "embeddings" / "test_metadata.csv").copy()
    metadata_all["image_id"] = metadata_all["image_id"].astype(str)
    metadata_all["dataset"] = metadata_all["dataset"].astype(str)
    embeddings_all = np.load(route_dir / "embeddings" / "test_embeddings.npy").astype(np.float32)
    texas_mask = metadata_all["dataset"].eq(TEXAS_DATASET).to_numpy()
    metadata_df = metadata_all.loc[texas_mask].copy().reset_index(drop=True)
    embeddings = embeddings_all[texas_mask]
    return metadata_df, embeddings


def _build_must_link_components(registry_df: pd.DataFrame) -> list[list[str]]:
    must_link_df = registry_df[registry_df["constraint_type"].astype(str).eq("must-link")].copy()
    uf = StringUnionFind()
    for row in must_link_df.itertuples(index=False):
        uf.union(str(row.image_id_a), str(row.image_id_b))
    buckets: dict[str, list[str]] = defaultdict(list)
    nodes = sorted(
        set(must_link_df["image_id_a"].astype(str).tolist()).union(set(must_link_df["image_id_b"].astype(str).tolist()))
    )
    for node in nodes:
        buckets[uf.find(node)].append(node)
    return sorted([sorted(values) for values in buckets.values()], key=lambda values: (-len(values), values))


def _constraint_agreement(pred_df: pd.DataFrame, registry_df: pd.DataFrame) -> dict[str, float]:
    cluster_by_image = {
        str(image_id): int(cluster_id)
        for image_id, cluster_id in zip(
            pred_df["image_id"].astype(str).tolist(),
            pred_df["pred_cluster_id"].astype(int).tolist(),
        )
    }
    must_weight = 0
    must_hit = 0
    cannot_weight = 0
    cannot_hit = 0
    for row in registry_df.itertuples(index=False):
        support = int(getattr(row, "support_count", 1))
        left = str(row.image_id_a)
        right = str(row.image_id_b)
        same_cluster = cluster_by_image.get(left) == cluster_by_image.get(right)
        if str(row.constraint_type) == "must-link":
            must_weight += support
            if same_cluster:
                must_hit += support
        elif str(row.constraint_type) == "cannot-link":
            cannot_weight += support
            if not same_cluster:
                cannot_hit += support
    total_weight = must_weight + cannot_weight
    total_hit = must_hit + cannot_hit
    return {
        "must_link_agreement": float(must_hit / must_weight) if must_weight else np.nan,
        "cannot_link_agreement": float(cannot_hit / cannot_weight) if cannot_weight else np.nan,
        "constraint_agreement": float(total_hit / total_weight) if total_weight else np.nan,
    }


def _format_markdown_table(frame: pd.DataFrame, limit: int = 40) -> str:
    if frame.empty:
        return "_Empty_"
    preview = frame.head(int(limit)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    submission_output_dir = args.submission_output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    submission_tables_dir = submission_output_dir / "tables"
    submission_reports_dir = submission_output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir, submission_output_dir, submission_tables_dir, submission_reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root / "src"))
    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table
    from animalclef_analysis.texas_hotspotter_local_clustering import load_texas_proxy_bundle
    from animalclef_analysis.texas_selftrain import compute_pair_keep_ratio, compute_threshold_proxy_score
    from animalclef_analysis.texas_unsupervised import pair_agreement_score

    base_pred_df = pd.read_csv(args.base_predictions.resolve()).copy()
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    base_pred_df["pred_cluster_id"] = pd.to_numeric(base_pred_df["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    texas_df = base_pred_df[base_pred_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(str)

    registry_df = pd.read_csv(args.registry_path.resolve()).copy()
    registry_df["image_id_a"] = registry_df["image_id_a"].astype(str)
    registry_df["image_id_b"] = registry_df["image_id_b"].astype(str)
    registry_df["constraint_type"] = registry_df["constraint_type"].astype(str)
    registry_df["support_count"] = pd.to_numeric(registry_df["support_count"], errors="coerce").fillna(1).astype(int)

    embed_meta_df, texas_embeddings = _load_embeddings(route_dir=args.route_dir.resolve())
    embed_lookup = {str(image_id): index for index, image_id in enumerate(embed_meta_df["image_id"].tolist())}
    reorder_index = np.asarray([embed_lookup[image_id] for image_id in texas_df["image_id"].tolist()], dtype=np.int32)
    texas_embeddings = texas_embeddings[reorder_index]
    score_matrix = np.clip(texas_embeddings @ texas_embeddings.T, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(score_matrix, 1.0)

    cluster_to_members = {
        int(cluster_id): sorted(frame["image_id"].astype(str).tolist())
        for cluster_id, frame in texas_df.groupby("pred_cluster_id", sort=True)
    }
    cluster_ids = sorted(cluster_to_members)
    image_to_cluster = {image_id: int(cluster_id) for cluster_id, members in cluster_to_members.items() for image_id in members}
    image_to_index = {image_id: index for index, image_id in enumerate(texas_df["image_id"].tolist())}

    must_link_components = _build_must_link_components(registry_df=registry_df)
    seed_clusters = {
        int(image_to_cluster[component[0]])
        for component in must_link_components
        if component and component[0] in image_to_cluster
    }

    cannot_cluster_pairs: set[tuple[int, int]] = set()
    for row in registry_df[registry_df["constraint_type"].eq("cannot-link")].itertuples(index=False):
        left_cluster = image_to_cluster.get(str(row.image_id_a))
        right_cluster = image_to_cluster.get(str(row.image_id_b))
        if left_cluster is None or right_cluster is None or int(left_cluster) == int(right_cluster):
            continue
        cannot_cluster_pairs.add(tuple(sorted((int(left_cluster), int(right_cluster)))))

    top_m = max(1, int(args.top_m))
    member_score_threshold = float(args.member_score_threshold)
    edge_rows: list[dict[str, object]] = []
    for left_pos, left_cluster in enumerate(cluster_ids):
        left_members = cluster_to_members[int(left_cluster)]
        for right_cluster in cluster_ids[left_pos + 1 :]:
            right_members = cluster_to_members[int(right_cluster)]
            if tuple(sorted((int(left_cluster), int(right_cluster)))) in cannot_cluster_pairs:
                continue
            if int(left_cluster) in seed_clusters and int(right_cluster) in seed_clusters:
                continue
            cross_scores: list[float] = []
            for left_image_id in left_members:
                left_index = image_to_index[left_image_id]
                for right_image_id in right_members:
                    cross_scores.append(float(score_matrix[left_index, image_to_index[right_image_id]]))
            if not cross_scores:
                continue
            ordered_scores = np.sort(np.asarray(cross_scores, dtype=np.float32))[::-1]
            edge_rows.append(
                {
                    "left_cluster_id": int(left_cluster),
                    "right_cluster_id": int(right_cluster),
                    "left_cluster_size": int(len(left_members)),
                    "right_cluster_size": int(len(right_members)),
                    "left_is_seed": bool(int(left_cluster) in seed_clusters),
                    "right_is_seed": bool(int(right_cluster) in seed_clusters),
                    "left_members": "|".join(left_members),
                    "right_members": "|".join(right_members),
                    "mean_top_score": round(float(ordered_scores[: min(top_m, len(ordered_scores))].mean()), 6),
                    "max_score": round(float(ordered_scores[0]), 6),
                    "support_count": int((ordered_scores >= member_score_threshold).sum()),
                }
            )
    edge_df = pd.DataFrame(edge_rows).sort_values(
        ["mean_top_score", "max_score", "support_count", "left_cluster_size", "right_cluster_size"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    edge_df.to_csv(tables_dir / "cluster_edge_candidates_v1.csv", index=False)

    proxy_bundle = load_texas_proxy_bundle(repo_root=repo_root, experiment_dir=args.route_dir.resolve())
    proxy_meta_df = proxy_bundle.metadata_df.copy().reset_index(drop=True)
    seed_mask = proxy_meta_df["is_seed"].to_numpy(dtype=bool)
    seed_labels = proxy_meta_df.loc[seed_mask, "pseudo_label_index"].to_numpy(dtype=int)
    candidate_pair_df = proxy_bundle.candidate_pair_df.copy().reset_index(drop=True)
    candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
    candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)
    if "mutual_topk_all_routes" not in candidate_pair_df.columns:
        candidate_pair_df["mutual_topk_all_routes"] = False
    all_candidate_pairs = [
        (image_to_index[row.image_id], image_to_index[row.neighbor_image_id])
        for row in candidate_pair_df.itertuples(index=False)
        if str(row.image_id) in image_to_index and str(row.neighbor_image_id) in image_to_index
    ]
    mutual_candidate_pairs = [
        (image_to_index[row.image_id], image_to_index[row.neighbor_image_id])
        for row in candidate_pair_df.itertuples(index=False)
        if str(row.image_id) in image_to_index
        and str(row.neighbor_image_id) in image_to_index
        and bool(getattr(row, "mutual_topk_all_routes", False))
    ]

    base_labels = texas_df["pred_cluster_id"].astype(int).to_numpy()
    sweep_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    for mean_top_threshold in [float(value) for value in args.mean_top_thresholds]:
        for min_support in [int(value) for value in args.min_support_values]:
            for allow_nonseed_nonseed in [bool(int(value)) for value in args.allow_nonseed_nonseed_options]:
                for max_cluster_size in [int(value) for value in args.max_cluster_size_values]:
                    uf = PairUnionFind(nodes=cluster_ids, seed_clusters=seed_clusters)
                    selected_edges: list[dict[str, object]] = []
                    for row in edge_df.itertuples(index=False):
                        if float(row.mean_top_score) < float(mean_top_threshold):
                            continue
                        if int(row.support_count) < int(min_support):
                            continue
                        if max(int(row.left_cluster_size), int(row.right_cluster_size)) > int(max_cluster_size):
                            continue
                        if (not bool(row.left_is_seed) and not bool(row.right_is_seed)) and not bool(allow_nonseed_nonseed):
                            continue
                        if uf.union(int(row.left_cluster_id), int(row.right_cluster_id)):
                            selected_edges.append(
                                {
                                    "left_cluster_id": int(row.left_cluster_id),
                                    "right_cluster_id": int(row.right_cluster_id),
                                    "left_cluster_size": int(row.left_cluster_size),
                                    "right_cluster_size": int(row.right_cluster_size),
                                    "left_is_seed": bool(row.left_is_seed),
                                    "right_is_seed": bool(row.right_is_seed),
                                    "mean_top_score": float(row.mean_top_score),
                                    "max_score": float(row.max_score),
                                    "support_count": int(row.support_count),
                                    "left_members": str(row.left_members),
                                    "right_members": str(row.right_members),
                                }
                            )

                    grouped_clusters: dict[int, list[int]] = defaultdict(list)
                    for cluster_id in cluster_ids:
                        grouped_clusters[uf.find(int(cluster_id))].append(int(cluster_id))

                    variant_labels = base_labels.copy()
                    moved_images = 0
                    component_rows: list[dict[str, object]] = []
                    for component_root, component_cluster_ids in grouped_clusters.items():
                        del component_root
                        seed_component_clusters = [cluster_id for cluster_id in component_cluster_ids if int(cluster_id) in seed_clusters]
                        target_cluster_id = int(seed_component_clusters[0]) if seed_component_clusters else int(min(component_cluster_ids))
                        component_rows.append(
                            {
                                "component_root_cluster_id": int(target_cluster_id),
                                "component_cluster_ids": "|".join(str(value) for value in sorted(component_cluster_ids)),
                                "component_size": int(len(component_cluster_ids)),
                                "target_cluster_id": int(target_cluster_id),
                            }
                        )
                        for cluster_id in component_cluster_ids:
                            for image_id in cluster_to_members[int(cluster_id)]:
                                if int(cluster_id) != int(target_cluster_id):
                                    moved_images += 1
                                variant_labels[image_to_index[image_id]] = int(target_cluster_id)

                    variant_pred_df = texas_df.copy()
                    variant_pred_df["pred_cluster_id"] = variant_labels.astype(int)
                    variant_pred_df["cluster_label"] = [
                        f"cluster_{TEXAS_DATASET}_{int(cluster_id)}" for cluster_id in variant_labels.astype(int).tolist()
                    ]
                    variant_pred_df["route_name"] = str(args.route_name)

                    proxy_row = {
                        "seed_pair_agreement": pair_agreement_score(variant_labels[seed_mask], seed_labels) if seed_mask.any() else np.nan,
                        "seed_recall_at_1": 1.0,
                        "candidate_pair_keep_ratio": compute_pair_keep_ratio(variant_labels, all_candidate_pairs),
                        "mutual_topk_pair_keep_ratio": compute_pair_keep_ratio(variant_labels, mutual_candidate_pairs),
                    }
                    proxy_score = float(compute_threshold_proxy_score(pd.Series(proxy_row)))
                    constraint_stats = _constraint_agreement(pred_df=variant_pred_df, registry_df=registry_df)
                    variant_counts = pd.Series(variant_labels).value_counts()
                    sweep_row = {
                        "mean_top_threshold": float(mean_top_threshold),
                        "min_support": int(min_support),
                        "allow_nonseed_nonseed": bool(allow_nonseed_nonseed),
                        "max_cluster_size": int(max_cluster_size),
                        "merged_edge_count": int(len(selected_edges)),
                        "moved_images": int(moved_images),
                        "cluster_count": int(variant_counts.size),
                        "largest_cluster_size": int(variant_counts.max()),
                        "proxy_score": round(proxy_score, 6),
                        **{key: round(float(value), 6) if pd.notna(value) else np.nan for key, value in proxy_row.items()},
                        **{key: round(float(value), 6) if pd.notna(value) else np.nan for key, value in constraint_stats.items()},
                        "selection_score": round(proxy_score - (0.0002 * moved_images), 6),
                    }
                    sweep_rows.append(sweep_row)
                    payload = {
                        "selected_edges_df": pd.DataFrame(selected_edges),
                        "component_df": pd.DataFrame(component_rows),
                        "variant_pred_df": variant_pred_df.copy(),
                        "sweep_row": dict(sweep_row),
                    }
                    if best_payload is None:
                        best_payload = payload
                    else:
                        current = pd.Series(payload["sweep_row"])
                        best = pd.Series(best_payload["sweep_row"])
                        current_rank = (
                            float(current["selection_score"]),
                            float(current["proxy_score"]),
                            float(current.get("constraint_agreement", 0.0) or 0.0),
                            float(current.get("must_link_agreement", 0.0) or 0.0),
                            float(current.get("cannot_link_agreement", 0.0) or 0.0),
                            -float(current["moved_images"]),
                        )
                        best_rank = (
                            float(best["selection_score"]),
                            float(best["proxy_score"]),
                            float(best.get("constraint_agreement", 0.0) or 0.0),
                            float(best.get("must_link_agreement", 0.0) or 0.0),
                            float(best.get("cannot_link_agreement", 0.0) or 0.0),
                            -float(best["moved_images"]),
                        )
                        if current_rank > best_rank:
                            best_payload = payload

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["selection_score", "proxy_score", "constraint_agreement", "moved_images"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    sweep_df.to_csv(tables_dir / "constrained_seed_graph_sweep_v1.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No constrained seed graph payload was produced.")

    best_selected_edges_df = best_payload["selected_edges_df"].copy().reset_index(drop=True)
    best_component_df = best_payload["component_df"].copy().reset_index(drop=True)
    best_variant_texas_df = best_payload["variant_pred_df"].copy().reset_index(drop=True)
    best_selected_edges_df.to_csv(tables_dir / "best_selected_edges_v1.csv", index=False)
    best_component_df.to_csv(tables_dir / "best_cluster_components_v1.csv", index=False)
    best_variant_texas_df.to_csv(tables_dir / "best_texas_predictions_v1.csv", index=False)

    final_pred_df = pd.concat([base_pred_df[~base_pred_df["dataset"].eq(TEXAS_DATASET)].copy(), best_variant_texas_df], ignore_index=True)
    final_pred_df["image_id"] = final_pred_df["image_id"].astype(str)
    final_pred_df.to_csv(submission_tables_dir / "test_predictions_v1.csv", index=False)
    build_submission(
        test_pred_df=final_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=submission_output_dir / "submission.csv",
    )

    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": TEXAS_DATASET,
                "variant": "base_seeded_attach",
                "samples": int(len(texas_df)),
                "clusters": int(pd.Series(base_labels).nunique()),
                "singleton_clusters": int((pd.Series(base_labels).value_counts() == 1).sum()),
                "singleton_ratio": round(float((pd.Series(base_labels).value_counts() == 1).mean()), 6),
                "largest_cluster_size": int(pd.Series(base_labels).value_counts().max()),
                "route_name": str(texas_df["route_name"].iloc[0]),
                "chosen_threshold": float(texas_df["chosen_threshold"].iloc[0]),
            },
            {
                "dataset": TEXAS_DATASET,
                "variant": "constrained_seed_graph",
                "samples": int(len(best_variant_texas_df)),
                "clusters": int(best_variant_texas_df["pred_cluster_id"].astype(int).nunique()),
                "singleton_clusters": int((best_variant_texas_df["pred_cluster_id"].value_counts() == 1).sum()),
                "singleton_ratio": round(float((best_variant_texas_df["pred_cluster_id"].value_counts() == 1).mean()), 6),
                "largest_cluster_size": int(best_variant_texas_df["pred_cluster_id"].value_counts().max()),
                "route_name": str(args.route_name),
                "chosen_threshold": float(best_variant_texas_df["chosen_threshold"].iloc[0]),
            },
        ]
    )
    cluster_summary_df.to_csv(submission_tables_dir / "cluster_summary_v1.csv", index=False)

    config = {
        "base_predictions": str(args.base_predictions.resolve()),
        "route_dir": str(args.route_dir.resolve()),
        "registry_path": str(args.registry_path.resolve()),
        "best_sweep_row": best_payload["sweep_row"],
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (submission_reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    analysis_lines = [
        "# Texas Constrained Seed Graph Probe",
        "",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Route dir: `{args.route_dir.resolve()}`",
        f"- Registry path: `{args.registry_path.resolve()}`",
        f"- Edge candidate rows: `{len(edge_df)}`",
        f"- Best config: `{best_payload['sweep_row']}`",
        "",
        "## Top Edge Candidates",
        "",
        _format_markdown_table(edge_df.head(20), limit=20),
        "",
        "## Sweep Summary",
        "",
        dataframe_to_markdown_table(sweep_df.head(20)),
        "",
        "## Chosen Edges",
        "",
        _format_markdown_table(best_selected_edges_df, limit=40),
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")

    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> Texas embedding -> current seeded-attach clusters -> constrained cluster graph -> submission cluster label`.",
        "- Current route:",
        "  - `LynxID2025`: keep base route unchanged.",
        "  - `SalamanderID2025`: keep base route unchanged.",
        "  - `SeaTurtleID2022`: keep base route unchanged.",
        "  - `TexasHornedLizards`: start from the current best seeded-attach route, then greedily merge Texas clusters on a score graph while forbidding any merge that would combine two distinct seed clusters or violate a cannot-link cluster pair.",
    ]
    submission_lines = [
        "# Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Registry path: `{args.registry_path.resolve()}`",
        f"- Route name: `{args.route_name}`",
        f"- Best config: `{best_payload['sweep_row']}`",
        "",
        "## Architecture",
        "",
        *architecture_lines,
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Chosen Edges",
        "",
        _format_markdown_table(best_selected_edges_df, limit=40),
        "",
        "## Key Performance Tricks",
        "",
        "- `cluster-level graph`: operate on current Texas clusters instead of raw images, so only a few conservative merge decisions are considered.",
        "- `seed exclusivity`: never allow a merge that would place two different seed clusters into the same connected component.",
        "- `cannot-link veto`: remove cluster-cluster candidate edges whenever any reviewed cannot-link spans the two clusters.",
        "- `base-preserving fallback`: all Texas clusters that do not participate in a selected graph edge keep the current best route unchanged.",
        "",
    ]
    (submission_reports_dir / "summary.md").write_text("\n".join(submission_lines) + "\n", encoding="utf-8")

    print(f"[texas_constrained_seed_graph] analysis_summary: {reports_dir / 'summary.md'}")
    print(f"[texas_constrained_seed_graph] submission: {submission_output_dir / 'submission.csv'}")
    print(f"[texas_constrained_seed_graph] submission_summary: {submission_reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
