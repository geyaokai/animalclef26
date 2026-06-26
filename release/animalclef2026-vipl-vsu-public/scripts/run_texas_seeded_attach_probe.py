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


class UnionFind:
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
    parser = argparse.ArgumentParser(description="Probe seeded Texas attachment clustering on top of the current best Texas graph route.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_manual_graph_v2_on_059138_base_v1/tables/test_predictions_v1.csv",
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
        default=repo_root / "artifacts/analysis/texas_seeded_attach_probe_v1",
    )
    parser.add_argument(
        "--submission-output-dir",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_seeded_attach_on_059341_base_v1",
    )
    parser.add_argument("--route-name", type=str, default="texas_seeded_attach_v1")
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--member-score-threshold", type=float, default=0.70)
    parser.add_argument("--mean-top-thresholds", nargs="+", type=float, default=[0.70, 0.71, 0.72, 0.73, 0.74])
    parser.add_argument("--gap-thresholds", nargs="+", type=float, default=[0.35, 0.40, 0.45, 0.50])
    parser.add_argument("--min-support-values", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--max-base-cluster-sizes", nargs="+", type=int, default=[1, 2])
    return parser.parse_args()


def _load_embeddings(route_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    test_metadata_all = pd.read_csv(route_dir / "embeddings" / "test_metadata.csv").copy()
    test_metadata_all["image_id"] = test_metadata_all["image_id"].astype(str)
    test_metadata_all["dataset"] = test_metadata_all["dataset"].astype(str)
    test_embeddings_all = np.load(route_dir / "embeddings" / "test_embeddings.npy").astype(np.float32)
    texas_mask = test_metadata_all["dataset"].eq(TEXAS_DATASET).to_numpy()
    test_metadata = test_metadata_all.loc[texas_mask].copy().reset_index(drop=True)
    test_embeddings = test_embeddings_all[texas_mask]
    return test_metadata, test_embeddings


def _build_components(registry_df: pd.DataFrame) -> list[list[str]]:
    must_link_df = registry_df[registry_df["constraint_type"].astype(str).eq("must-link")].copy()
    uf = UnionFind()
    for row in must_link_df.itertuples(index=False):
        uf.union(str(row.image_id_a), str(row.image_id_b))
    buckets: dict[str, list[str]] = defaultdict(list)
    all_nodes = sorted(
        set(must_link_df["image_id_a"].astype(str).tolist()).union(set(must_link_df["image_id_b"].astype(str).tolist()))
    )
    for node in all_nodes:
        buckets[uf.find(node)].append(str(node))
    return sorted([sorted(nodes) for nodes in buckets.values()], key=lambda nodes: (-len(nodes), nodes))


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

    components = _build_components(registry_df=registry_df)
    cluster_by_image = {
        str(image_id): int(cluster_id)
        for image_id, cluster_id in zip(texas_df["image_id"].tolist(), texas_df["pred_cluster_id"].tolist())
    }
    cluster_size_by_id = texas_df["pred_cluster_id"].value_counts().astype(int).to_dict()
    cannot_link_adj: dict[str, set[str]] = defaultdict(set)
    for row in registry_df[registry_df["constraint_type"].eq("cannot-link")].itertuples(index=False):
        cannot_link_adj[str(row.image_id_a)].add(str(row.image_id_b))
        cannot_link_adj[str(row.image_id_b)].add(str(row.image_id_a))

    valid_components: list[dict[str, object]] = []
    for component_index, component_nodes in enumerate(components, start=1):
        cluster_ids = {cluster_by_image.get(node) for node in component_nodes if node in cluster_by_image}
        cluster_ids.discard(None)
        if len(cluster_ids) != 1:
            continue
        valid_components.append(
            {
                "component_index": int(component_index),
                "component_nodes": list(component_nodes),
                "target_cluster_id": int(next(iter(cluster_ids))),
                "component_size": int(len(component_nodes)),
            }
        )

    image_index_lookup = {image_id: index for index, image_id in enumerate(texas_df["image_id"].tolist())}
    seed_nodes = {node for component in valid_components for node in component["component_nodes"]}
    candidate_rows: list[dict[str, object]] = []
    top_m = max(1, int(args.top_m))
    member_score_threshold = float(args.member_score_threshold)
    for image_id in texas_df["image_id"].tolist():
        if image_id in seed_nodes:
            continue
        image_index = image_index_lookup[image_id]
        component_scores: list[dict[str, object]] = []
        for component in valid_components:
            component_nodes = [str(value) for value in component["component_nodes"]]
            if any(image_id in cannot_link_adj.get(node, set()) for node in component_nodes):
                continue
            member_indices = np.asarray([image_index_lookup[node] for node in component_nodes], dtype=np.int32)
            member_scores = score_matrix[image_index, member_indices].astype(np.float32, copy=False)
            sorted_scores = np.sort(member_scores)[::-1]
            mean_top = float(sorted_scores[: min(top_m, len(sorted_scores))].mean())
            component_scores.append(
                {
                    "component_index": int(component["component_index"]),
                    "target_cluster_id": int(component["target_cluster_id"]),
                    "component_size": int(component["component_size"]),
                    "component_nodes": "|".join(component_nodes),
                    "mean_top_score": round(float(mean_top), 6),
                    "max_score": round(float(sorted_scores[0]), 6),
                    "support_count": int((member_scores >= member_score_threshold).sum()),
                }
            )
        if not component_scores:
            continue
        component_scores = sorted(
            component_scores,
            key=lambda row: (
                float(row["mean_top_score"]),
                float(row["max_score"]),
                int(row["support_count"]),
                int(row["component_size"]),
            ),
            reverse=True,
        )
        best_row = component_scores[0]
        second_score = float(component_scores[1]["mean_top_score"]) if len(component_scores) > 1 else 0.0
        candidate_rows.append(
            {
                "image_id": str(image_id),
                "base_cluster_id": int(cluster_by_image[image_id]),
                "base_cluster_size": int(cluster_size_by_id[int(cluster_by_image[image_id])]),
                "best_component_index": int(best_row["component_index"]),
                "target_cluster_id": int(best_row["target_cluster_id"]),
                "target_component_size": int(best_row["component_size"]),
                "best_component_nodes": str(best_row["component_nodes"]),
                "mean_top_score": float(best_row["mean_top_score"]),
                "max_score": float(best_row["max_score"]),
                "support_count": int(best_row["support_count"]),
                "second_best_mean_top_score": round(float(second_score), 6),
                "margin_vs_second": round(float(best_row["mean_top_score"]) - float(second_score), 6),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows).sort_values(
        ["mean_top_score", "margin_vs_second", "support_count", "base_cluster_size"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    candidate_df.to_csv(tables_dir / "seed_attach_candidates_v1.csv", index=False)

    proxy_bundle = load_texas_proxy_bundle(repo_root=repo_root, experiment_dir=args.route_dir.resolve())
    proxy_meta_df = proxy_bundle.metadata_df.copy().reset_index(drop=True)
    seed_mask = proxy_meta_df["is_seed"].to_numpy(dtype=bool)
    seed_labels = proxy_meta_df.loc[seed_mask, "pseudo_label_index"].to_numpy(dtype=int)
    candidate_pair_df = proxy_bundle.candidate_pair_df.copy().reset_index(drop=True)
    candidate_pair_df["image_id"] = candidate_pair_df["image_id"].astype(str)
    candidate_pair_df["neighbor_image_id"] = candidate_pair_df["neighbor_image_id"].astype(str)
    if "mutual_topk_all_routes" not in candidate_pair_df.columns:
        candidate_pair_df["mutual_topk_all_routes"] = False
    image_to_index = image_index_lookup
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

    sweep_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    base_labels = texas_df["pred_cluster_id"].astype(int).to_numpy()
    for mean_top_threshold in [float(value) for value in args.mean_top_thresholds]:
        for gap_threshold in [float(value) for value in args.gap_thresholds]:
            for min_support in [int(value) for value in args.min_support_values]:
                for max_base_cluster_size in [int(value) for value in args.max_base_cluster_sizes]:
                    selected_df = candidate_df[
                        candidate_df["mean_top_score"].astype(float).ge(float(mean_top_threshold))
                        & candidate_df["margin_vs_second"].astype(float).ge(float(gap_threshold))
                        & candidate_df["support_count"].astype(int).ge(int(min_support))
                        & candidate_df["base_cluster_size"].astype(int).le(int(max_base_cluster_size))
                    ].copy()
                    variant_labels = base_labels.copy()
                    for row in selected_df.itertuples(index=False):
                        variant_labels[image_to_index[str(row.image_id)]] = int(row.target_cluster_id)
                    variant_pred_df = texas_df.copy()
                    variant_pred_df["pred_cluster_id"] = variant_labels.astype(int)
                    variant_pred_df["cluster_label"] = [
                        f"cluster_{TEXAS_DATASET}_{int(cluster_id)}" for cluster_id in variant_labels.astype(int).tolist()
                    ]
                    variant_pred_df["route_name"] = str(args.route_name)
                    variant_pred_df["chosen_threshold"] = variant_pred_df["chosen_threshold"].astype(float)

                    proxy_row = {
                        "seed_pair_agreement": pair_agreement_score(variant_labels[seed_mask], seed_labels) if seed_mask.any() else np.nan,
                        "seed_recall_at_1": 1.0,
                        "candidate_pair_keep_ratio": compute_pair_keep_ratio(variant_labels, all_candidate_pairs),
                        "mutual_topk_pair_keep_ratio": compute_pair_keep_ratio(variant_labels, mutual_candidate_pairs),
                    }
                    proxy_score = float(compute_threshold_proxy_score(pd.Series(proxy_row)))
                    constraint_stats = _constraint_agreement(pred_df=variant_pred_df, registry_df=registry_df)
                    variant_cluster_counts = pd.Series(variant_labels).value_counts()
                    moved_images = int(len(selected_df))
                    sweep_row = {
                        "mean_top_threshold": float(mean_top_threshold),
                        "gap_threshold": float(gap_threshold),
                        "min_support": int(min_support),
                        "max_base_cluster_size": int(max_base_cluster_size),
                        "moved_images": moved_images,
                        "cluster_count": int(variant_cluster_counts.size),
                        "largest_cluster_size": int(variant_cluster_counts.max()),
                        "proxy_score": round(proxy_score, 6),
                        **{key: round(float(value), 6) if pd.notna(value) else np.nan for key, value in proxy_row.items()},
                        **{key: round(float(value), 6) if pd.notna(value) else np.nan for key, value in constraint_stats.items()},
                        "selection_score": round(proxy_score - (0.0002 * moved_images), 6),
                    }
                    sweep_rows.append(sweep_row)
                    payload = {
                        "selected_df": selected_df.copy(),
                        "variant_pred_df": variant_pred_df.copy(),
                        "sweep_row": dict(sweep_row),
                    }
                    if best_payload is None:
                        best_payload = payload
                    else:
                        current = pd.Series(payload["sweep_row"])
                        best = pd.Series(best_payload["sweep_row"])
                        ranking_columns = [
                            "selection_score",
                            "proxy_score",
                            "constraint_agreement",
                            "must_link_agreement",
                            "cannot_link_agreement",
                            "moved_images",
                        ]
                        current_values = (
                            float(current["selection_score"]),
                            float(current["proxy_score"]),
                            float(current.get("constraint_agreement", 0.0) or 0.0),
                            float(current.get("must_link_agreement", 0.0) or 0.0),
                            float(current.get("cannot_link_agreement", 0.0) or 0.0),
                            -float(current["moved_images"]),
                        )
                        best_values = (
                            float(best["selection_score"]),
                            float(best["proxy_score"]),
                            float(best.get("constraint_agreement", 0.0) or 0.0),
                            float(best.get("must_link_agreement", 0.0) or 0.0),
                            float(best.get("cannot_link_agreement", 0.0) or 0.0),
                            -float(best["moved_images"]),
                        )
                        del ranking_columns
                        if current_values > best_values:
                            best_payload = payload

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        ["selection_score", "proxy_score", "constraint_agreement", "moved_images"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    sweep_df.to_csv(tables_dir / "seed_attach_sweep_v1.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No seeded attachment payload was produced.")

    best_selected_df = best_payload["selected_df"].copy().reset_index(drop=True)
    best_selected_df.to_csv(tables_dir / "best_selected_attachments_v1.csv", index=False)
    best_variant_texas_df = best_payload["variant_pred_df"].copy().reset_index(drop=True)
    best_variant_texas_df.to_csv(tables_dir / "best_texas_predictions_v1.csv", index=False)

    merged_pred_df = base_pred_df.copy()
    non_texas_df = merged_pred_df[~merged_pred_df["dataset"].eq(TEXAS_DATASET)].copy()
    final_pred_df = pd.concat([non_texas_df, best_variant_texas_df], ignore_index=True)
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
                "variant": "base_v2",
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
                "variant": "seeded_attach",
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
        "moved_images": int(len(best_selected_df)),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (submission_reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# Texas Seeded Attach Probe",
        "",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Route dir: `{args.route_dir.resolve()}`",
        f"- Registry path: `{args.registry_path.resolve()}`",
        f"- Candidate rows: `{len(candidate_df)}`",
        f"- Best config: `{best_payload['sweep_row']}`",
        "",
        "## Top Candidate Attachments",
        "",
        _format_markdown_table(candidate_df.head(20), limit=20),
        "",
        "## Sweep Summary",
        "",
        dataframe_to_markdown_table(sweep_df.head(20)),
        "",
        "## Chosen Attachments",
        "",
        _format_markdown_table(best_selected_df, limit=40),
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> Texas embedding -> current best graph-v2 clusters -> must-link seed component scoring -> conservative seeded attachment -> submission cluster label`.",
        "- Current route:",
        "  - `LynxID2025`: keep base route unchanged.",
        "  - `SalamanderID2025`: keep base route unchanged.",
        "  - `SeaTurtleID2022`: keep base route unchanged.",
        "  - `TexasHornedLizards`: start from the `0.59341` graph-v2 route, then only attach non-seed images to existing must-link components when component affinity, support count, and best-vs-second margin all exceed the chosen thresholds.",
    ]
    submission_summary_lines = [
        "# Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Registry path: `{args.registry_path.resolve()}`",
        f"- Route name: `{args.route_name}`",
        f"- Changed Texas images: `{int(len(best_selected_df))}`",
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
        "## Chosen Attachments",
        "",
        _format_markdown_table(best_selected_df, limit=40),
        "",
        "## Key Performance Tricks",
        "",
        "- `must-link as seeds`: use reviewed Texas must-link components as stable identity anchors rather than only as a final merge overlay.",
        "- `component-affinity attach`: for each non-seed image, score every seed component by mean top-member similarity and only attach the best component when the margin over the second-best component is large enough.",
        "- `small-cluster bias`: only rescue images that currently sit in singleton or tiny clusters, so the variant stays conservative.",
        "- `base-preserving fallback`: every Texas image that does not meet the seeded-attach gate keeps the current best `0.59341` route unchanged.",
        "",
    ]
    (submission_reports_dir / "summary.md").write_text("\n".join(submission_summary_lines) + "\n", encoding="utf-8")

    print(f"[texas_seeded_attach] analysis_summary: {reports_dir / 'summary.md'}")
    print(f"[texas_seeded_attach] submission: {submission_output_dir / 'submission.csv'}")
    print(f"[texas_seeded_attach] submission_summary: {submission_reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
