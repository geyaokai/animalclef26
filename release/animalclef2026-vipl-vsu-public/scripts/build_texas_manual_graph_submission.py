#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def components(self) -> list[list[int]]:
        buckets: dict[int, list[int]] = {}
        for item in list(self.parent):
            buckets.setdefault(self.find(item), []).append(item)
        return [sorted(nodes) for nodes in buckets.values()]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a Texas graph submission by overlaying manual positive components.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_tcuwarmup_trusted_views_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--positive-pairs",
        type=Path,
        default=repo_root / "artifacts/analysis/tcu_kaggle_texas_overlap_probe_v1/kaggle_texas_manual_positive_pairs_train_v1.csv",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=repo_root / "sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_manual_graph_on_tcuwarmup_bestpublic_v1",
    )
    parser.add_argument(
        "--route-name",
        type=str,
        default="texas_manual_graph_v1",
    )
    return parser.parse_args()


def build_components(pair_df: pd.DataFrame) -> list[list[int]]:
    uf = UnionFind()
    for row in pair_df.itertuples(index=False):
        uf.union(int(row.image_id_a), int(row.image_id_b))
    components = [nodes for nodes in uf.components() if len(nodes) >= 2]
    components.sort(key=lambda nodes: (-len(nodes), nodes))
    return components


def relabel_component(
    texas_df: pd.DataFrame,
    *,
    component_nodes: list[int],
    component_cluster_id: int,
    route_name: str,
    component_index: int,
    support_count: int,
) -> list[dict[str, object]]:
    changed_rows: list[dict[str, object]] = []
    node_set = {int(node) for node in component_nodes}
    for idx, row in texas_df.iterrows():
        image_id = int(row["image_id"])
        if image_id not in node_set:
            continue
        base_cluster_id = int(row["pred_cluster_id"])
        base_cluster_label = str(row["cluster_label"])
        texas_df.at[idx, "pred_cluster_id"] = int(component_cluster_id)
        texas_df.at[idx, "cluster_label"] = f"cluster_{TEXAS_DATASET}_{int(component_cluster_id)}"
        texas_df.at[idx, "route_name"] = str(route_name)
        texas_df.at[idx, "manual_overlay_enabled"] = True
        texas_df.at[idx, "manual_overlay_rule"] = f"{route_name}|positive_component|comp_{component_index:03d}"
        texas_df.at[idx, "manual_overlay_operation_id"] = f"texas_positive_component_{component_index:03d}"
        texas_df.at[idx, "manual_overlay_note"] = (
            f"manual positive component overlay | size={len(component_nodes)} | support_count={support_count}"
        )
        changed_rows.append(
            {
                "component_index": component_index,
                "component_cluster_id": int(component_cluster_id),
                "component_size": int(len(component_nodes)),
                "component_support_count": int(support_count),
                "image_id": image_id,
                "path": str(row["path"]),
                "base_cluster_id": base_cluster_id,
                "base_cluster_label": base_cluster_label,
                "new_cluster_id": int(component_cluster_id),
                "new_cluster_label": f"cluster_{TEXAS_DATASET}_{int(component_cluster_id)}",
                "cluster_changed": bool(base_cluster_id != component_cluster_id),
            }
        )
    return changed_rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root / "src"))
    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table

    pred_df = pd.read_csv(args.base_predictions.resolve())
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    if "manual_overlay_enabled" not in pred_df.columns:
        pred_df["manual_overlay_enabled"] = False
    else:
        pred_df["manual_overlay_enabled"] = pred_df["manual_overlay_enabled"].fillna(False).astype(bool)
    for column in ["manual_overlay_rule", "manual_overlay_operation_id", "manual_overlay_note"]:
        if column not in pred_df.columns:
            pred_df[column] = ""
        else:
            pred_df[column] = pred_df[column].fillna("").astype(str)

    texas_mask = pred_df["dataset"].eq(TEXAS_DATASET)
    texas_df = pred_df.loc[texas_mask].copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(int)
    base_texas_df = texas_df.copy()

    positive_pairs = pd.read_csv(args.positive_pairs.resolve())
    components = build_components(positive_pairs)
    support_by_pair = {
        (int(row.image_id_a), int(row.image_id_b)): int(row.support_count)
        for row in positive_pairs.itertuples(index=False)
    }
    support_by_component: dict[tuple[int, ...], int] = {}
    for nodes in components:
        node_set = set(nodes)
        support = 0
        for left, right in support_by_pair:
            if left in node_set and right in node_set:
                support += int(support_by_pair[(left, right)])
        support_by_component[tuple(nodes)] = support

    next_cluster_id = int(texas_df["pred_cluster_id"].astype(int).max()) + 1
    changed_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for component_index, nodes in enumerate(components, start=1):
        component_cluster_id = next_cluster_id
        next_cluster_id += 1
        support_count = int(support_by_component.get(tuple(nodes), 0))
        base_cluster_ids = sorted(
            {
                int(value)
                for value in texas_df.loc[texas_df["image_id"].isin(nodes), "pred_cluster_id"].astype(int).tolist()
            }
        )
        component_rows.append(
            {
                "component_index": component_index,
                "component_cluster_id": component_cluster_id,
                "component_size": len(nodes),
                "component_support_count": support_count,
                "base_cluster_ids": "|".join(str(value) for value in base_cluster_ids),
                "member_image_ids": "|".join(str(node) for node in nodes),
            }
        )
        changed_rows.extend(
            relabel_component(
                texas_df=texas_df,
                component_nodes=nodes,
                component_cluster_id=component_cluster_id,
                route_name=args.route_name,
                component_index=component_index,
                support_count=support_count,
            )
        )

    merged_pred_df = pd.concat([pred_df.loc[~texas_mask].copy(), texas_df.copy()], ignore_index=True)
    merged_pred_df["image_id"] = merged_pred_df["image_id"].astype(str)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=output_dir / "submission.csv",
    )

    component_df = pd.DataFrame(component_rows).sort_values(
        ["component_size", "component_support_count", "component_index"], ascending=[False, False, True]
    )
    component_df.to_csv(tables_dir / "texas_manual_graph_components_v1.csv", index=False)

    changed_df = pd.DataFrame(changed_rows)
    if not changed_df.empty:
        changed_df = changed_df.sort_values(["component_index", "image_id"]).reset_index(drop=True)
    changed_df.to_csv(tables_dir / "texas_manual_graph_delta_v1.csv", index=False)

    base_counts = base_texas_df["pred_cluster_id"].value_counts()
    new_counts = texas_df["pred_cluster_id"].value_counts()
    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": TEXAS_DATASET,
                "variant": "base",
                "samples": int(len(base_texas_df)),
                "clusters": int(base_counts.size),
                "singleton_clusters": int((base_counts == 1).sum()),
                "singleton_ratio": round(float((base_counts == 1).mean()), 6),
                "largest_cluster_size": int(base_counts.max()),
                "route_name": str(base_texas_df["route_name"].iloc[0]),
                "chosen_threshold": float(base_texas_df["chosen_threshold"].iloc[0]),
            },
            {
                "dataset": TEXAS_DATASET,
                "variant": "manual_graph",
                "samples": int(len(texas_df)),
                "clusters": int(new_counts.size),
                "singleton_clusters": int((new_counts == 1).sum()),
                "singleton_ratio": round(float((new_counts == 1).mean()), 6),
                "largest_cluster_size": int(new_counts.max()),
                "route_name": str(args.route_name),
                "chosen_threshold": float(texas_df["chosen_threshold"].iloc[0]),
            },
        ]
    )
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    changed_image_ids = set(changed_df["image_id"].astype(str).tolist()) if not changed_df.empty else set()
    delta_rows = []
    for row in base_texas_df.itertuples(index=False):
        image_id = str(row.image_id)
        new_row = texas_df.loc[texas_df["image_id"].eq(int(image_id))].iloc[0]
        delta_rows.append(
            {
                "image_id": image_id,
                "path": str(row.path),
                "pred_cluster_id_base": int(row.pred_cluster_id),
                "pred_cluster_id_new": int(new_row["pred_cluster_id"]),
                "route_name_base": str(row.route_name),
                "route_name_new": str(new_row["route_name"]),
                "cluster_changed": image_id in changed_image_ids,
            }
        )
    delta_df = pd.DataFrame(delta_rows)
    delta_df.to_csv(tables_dir / "texas_override_delta_v1.csv", index=False)

    config = {
        "base_predictions": str(args.base_predictions.resolve()),
        "positive_pairs": str(args.positive_pairs.resolve()),
        "route_name": str(args.route_name),
        "component_count": int(len(component_df)),
        "changed_images": int(len(changed_image_ids)),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> optional rerank/postprocess -> clustering -> submission cluster label`.",
        "- Current route:",
        "  - `LynxID2025`: keep base route unchanged.",
        "  - `SalamanderID2025`: keep base route unchanged.",
        "  - `SeaTurtleID2022`: keep base route unchanged.",
        f"  - `{TEXAS_DATASET}`: start from `ft_texas_miew_trusted_views_v1 @ 0.44`, then apply manual positive graph overlay by extracting reviewed positive components and reassigning each component to a dedicated cluster.",
    ]

    summary_lines = [
        "# Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Positive pairs: `{args.positive_pairs.resolve()}`",
        f"- Route name: `{args.route_name}`",
        f"- Manual positive components: `{len(component_df)}`",
        f"- Images moved by graph overlay: `{len(changed_image_ids)}`",
        "",
        "## Architecture",
        "",
        *architecture_lines,
        "",
        "## Texas Graph Delta",
        "",
        f"- Positive pair seeds used: `{len(positive_pairs)}`",
        f"- Connected components used: `{len(component_df)}`",
        f"- Cross-cluster positive pairs in base route: `{int((positive_pairs['image_id_a'].map(dict(zip(base_texas_df['image_id'], base_texas_df['pred_cluster_id']))) != positive_pairs['image_id_b'].map(dict(zip(base_texas_df['image_id'], base_texas_df['pred_cluster_id'])))).sum())}`",
        f"- Changed Texas images: `{len(changed_image_ids)} / {len(base_texas_df)}`",
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Component Summary",
        "",
        dataframe_to_markdown_table(component_df.head(20)) if not component_df.empty else "_No components._",
        "",
        "## Sample Delta",
        "",
        dataframe_to_markdown_table(delta_df[delta_df['cluster_changed']].head(20)) if not delta_df.empty else "_No delta rows._",
        "",
        "## Key Performance Tricks",
        "",
        "- `single-dataset override`: keep `Lynx / Salamander / SeaTurtle` fixed and modify only `TexasHornedLizards`.",
        "- `positive graph overlay`: do not retrain; directly convert reviewed positive pairs into connected components on the Texas test graph.",
        "- `component-only regroup`: move reviewed component members into dedicated clusters without dragging unrelated cluster mates along.",
        "- `base-preserving fallback`: all Texas images not covered by reviewed positive components keep the base trusted-views prediction.",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[texas_manual_graph] submission: {output_dir / 'submission.csv'}")
    print(f"[texas_manual_graph] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[texas_manual_graph] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
