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


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a conservative Texas graph v2 submission by applying cannot-link singleton splits on top of graph v1."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=repo_root
        / "artifacts/submissions/kaggle_variant_texas_manual_graph_on_tcuwarmup_bestpublic_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v1/texas_pair_registry_v1.csv",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=repo_root / "sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_manual_graph_v2_on_059138_base_v1",
    )
    parser.add_argument(
        "--route-name",
        type=str,
        default="texas_manual_graph_v2",
    )
    parser.add_argument(
        "--split-policy",
        choices=["singletonize_violating_nodes"],
        default="singletonize_violating_nodes",
    )
    parser.add_argument(
        "--skip-clusters-with-internal-must-link",
        action="store_true",
        default=True,
        help="If a current cluster already contains internal must-link support, keep it unchanged even when a cannot-link violation remains.",
    )
    return parser.parse_args()


def _load_must_link_components(registry_df: pd.DataFrame) -> dict[int, list[int]]:
    must_link_df = registry_df[registry_df["constraint_type"].astype(str).eq("must-link")].copy()
    if must_link_df.empty:
        return {}
    uf = UnionFind()
    for row in must_link_df.itertuples(index=False):
        uf.union(int(row.image_id_a), int(row.image_id_b))
    components: dict[int, list[int]] = {}
    for image_id in sorted(
        {
            int(value)
            for value in must_link_df["image_id_a"].astype(int).tolist() + must_link_df["image_id_b"].astype(int).tolist()
        }
    ):
        root = uf.find(int(image_id))
        components.setdefault(root, []).append(int(image_id))
    return {int(root): sorted(nodes) for root, nodes in components.items()}


def _format_markdown_table(frame: pd.DataFrame, limit: int = 40) -> str:
    if frame.empty:
        return "_Empty_"
    preview = frame.head(int(limit)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        values = []
        for column in columns:
            text = str(row[column]).replace("|", "\\|").replace("\n", "<br>")
            values.append(text)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


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
    pred_df["pred_cluster_id"] = pd.to_numeric(pred_df["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    if "manual_overlay_enabled" not in pred_df.columns:
        pred_df["manual_overlay_enabled"] = False
    else:
        pred_df["manual_overlay_enabled"] = pred_df["manual_overlay_enabled"].fillna(False).astype(bool)
    for column in ["manual_overlay_rule", "manual_overlay_operation_id", "manual_overlay_note"]:
        if column not in pred_df.columns:
            pred_df[column] = ""
        else:
            pred_df[column] = pred_df[column].fillna("").astype(str)

    registry_df = pd.read_csv(args.registry_path.resolve())
    registry_df["image_id_a"] = pd.to_numeric(registry_df["image_id_a"], errors="coerce").fillna(-1).astype(int)
    registry_df["image_id_b"] = pd.to_numeric(registry_df["image_id_b"], errors="coerce").fillna(-1).astype(int)
    registry_df["constraint_type"] = registry_df["constraint_type"].astype(str)
    registry_df["support_count"] = pd.to_numeric(registry_df["support_count"], errors="coerce").fillna(0).astype(int)

    texas_mask = pred_df["dataset"].eq(TEXAS_DATASET)
    texas_df = pred_df.loc[texas_mask].copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(int)
    base_texas_df = texas_df.copy()

    must_link_components = _load_must_link_components(registry_df)
    component_root_by_image: dict[int, int] = {}
    for root, nodes in must_link_components.items():
        for image_id in nodes:
            component_root_by_image[int(image_id)] = int(root)

    cluster_by_image = dict(zip(texas_df["image_id"].astype(int), texas_df["pred_cluster_id"].astype(int)))
    cluster_members: dict[int, list[int]] = (
        texas_df.groupby("pred_cluster_id")["image_id"].apply(lambda series: sorted(series.astype(int).tolist())).to_dict()
    )

    cannot_link_df = registry_df[registry_df["constraint_type"].eq("cannot-link")].copy()
    cannot_link_df["same_cluster"] = [
        cluster_by_image.get(int(left), -1) == cluster_by_image.get(int(right), -2)
        for left, right in zip(cannot_link_df["image_id_a"], cannot_link_df["image_id_b"])
    ]
    violation_df = cannot_link_df[cannot_link_df["same_cluster"]].copy().reset_index(drop=True)
    if not violation_df.empty:
        violation_df["cluster_id"] = [int(cluster_by_image[int(image_id)]) for image_id in violation_df["image_id_a"]]
        violation_df["cluster_size"] = violation_df["cluster_id"].map(
            {int(cluster_id): len(members) for cluster_id, members in cluster_members.items()}
        )
    else:
        violation_df["cluster_id"] = []
        violation_df["cluster_size"] = []
    violation_df.to_csv(tables_dir / "cannot_link_violations_v1.csv", index=False)

    next_cluster_id = int(texas_df["pred_cluster_id"].max()) + 1
    cluster_action_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    changed_image_ids: set[int] = set()

    for cluster_id, cluster_violation_df in violation_df.groupby("cluster_id", sort=True):
        member_image_ids = sorted(cluster_members.get(int(cluster_id), []))
        member_set = set(member_image_ids)
        internal_registry_df = registry_df[
            registry_df["image_id_a"].isin(member_set) & registry_df["image_id_b"].isin(member_set)
        ].copy()
        internal_must_link_df = internal_registry_df[internal_registry_df["constraint_type"].eq("must-link")].copy()
        violating_image_ids = sorted(
            {
                int(value)
                for value in cluster_violation_df["image_id_a"].astype(int).tolist()
                + cluster_violation_df["image_id_b"].astype(int).tolist()
            }
        )
        if args.skip_clusters_with_internal_must_link and not internal_must_link_df.empty:
            cluster_action_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(member_image_ids)),
                    "violation_pair_count": int(len(cluster_violation_df)),
                    "violating_image_ids": "|".join(str(value) for value in violating_image_ids),
                    "internal_must_link_pair_count": int(len(internal_must_link_df)),
                    "action": "skip_internal_must_link_conflict",
                    "new_cluster_ids": "",
                    "note": "leave cluster unchanged because current cluster contains internal must-link support",
                }
            )
            continue

        new_cluster_ids: list[int] = []
        for image_id in violating_image_ids:
            row_index = texas_df.index[texas_df["image_id"].eq(int(image_id))]
            if row_index.empty:
                continue
            idx = int(row_index[0])
            base_cluster_id = int(texas_df.at[idx, "pred_cluster_id"])
            base_cluster_label = str(texas_df.at[idx, "cluster_label"])
            assigned_cluster_id = int(next_cluster_id)
            next_cluster_id += 1

            texas_df.at[idx, "pred_cluster_id"] = assigned_cluster_id
            texas_df.at[idx, "cluster_label"] = f"cluster_{TEXAS_DATASET}_{assigned_cluster_id}"
            texas_df.at[idx, "route_name"] = str(args.route_name)
            texas_df.at[idx, "manual_overlay_enabled"] = True
            texas_df.at[idx, "manual_overlay_rule"] = f"{args.route_name}|cannot_link_singletonize|cluster_{int(cluster_id)}"
            texas_df.at[idx, "manual_overlay_operation_id"] = (
                f"texas_cannot_link_singletonize_cluster_{int(cluster_id)}_{int(image_id)}"
            )
            texas_df.at[idx, "manual_overlay_note"] = (
                "manual cannot-link singleton split"
                f" | base_cluster={int(cluster_id)}"
                f" | violating_images={'|'.join(str(value) for value in violating_image_ids)}"
            )
            changed_image_ids.add(int(image_id))
            new_cluster_ids.append(int(assigned_cluster_id))
            delta_rows.append(
                {
                    "image_id": int(image_id),
                    "path": str(texas_df.at[idx, "path"]),
                    "pred_cluster_id_base": int(base_cluster_id),
                    "pred_cluster_id_new": int(assigned_cluster_id),
                    "base_cluster_label": base_cluster_label,
                    "new_cluster_label": f"cluster_{TEXAS_DATASET}_{assigned_cluster_id}",
                    "route_name_new": str(args.route_name),
                    "cluster_changed": True,
                    "change_reason": "cannot_link_singletonize",
                }
            )

        cluster_action_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(member_image_ids)),
                "violation_pair_count": int(len(cluster_violation_df)),
                "violating_image_ids": "|".join(str(value) for value in violating_image_ids),
                "internal_must_link_pair_count": int(len(internal_must_link_df)),
                "action": "singletonize_violating_nodes",
                "new_cluster_ids": "|".join(str(value) for value in new_cluster_ids),
                "note": "split only images that participate in unresolved cannot-link violations",
            }
        )

    cluster_action_df = pd.DataFrame(cluster_action_rows).sort_values(["cluster_id"], ascending=[True]) if cluster_action_rows else pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_size",
            "violation_pair_count",
            "violating_image_ids",
            "internal_must_link_pair_count",
            "action",
            "new_cluster_ids",
            "note",
        ]
    )
    cluster_action_df.to_csv(tables_dir / "cluster_actions_v1.csv", index=False)

    delta_df = pd.DataFrame(delta_rows).sort_values(["pred_cluster_id_base", "image_id"], ascending=[True, True]) if delta_rows else pd.DataFrame(
        columns=[
            "image_id",
            "path",
            "pred_cluster_id_base",
            "pred_cluster_id_new",
            "base_cluster_label",
            "new_cluster_label",
            "route_name_new",
            "cluster_changed",
            "change_reason",
        ]
    )
    delta_df.to_csv(tables_dir / "texas_manual_graph_delta_v2.csv", index=False)

    merged_pred_df = pd.concat([pred_df.loc[~texas_mask].copy(), texas_df.copy()], ignore_index=True)
    merged_pred_df["image_id"] = merged_pred_df["image_id"].astype(str)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=output_dir / "submission.csv",
    )

    base_counts = base_texas_df["pred_cluster_id"].value_counts()
    new_counts = texas_df["pred_cluster_id"].value_counts()
    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": TEXAS_DATASET,
                "variant": "base_v1_graph",
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
                "variant": "graph_v2_cannot_link",
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

    updated_cluster_by_image = dict(zip(texas_df["image_id"].astype(int), texas_df["pred_cluster_id"].astype(int)))
    remaining_violation_df = cannot_link_df.copy()
    remaining_violation_df["same_cluster"] = [
        updated_cluster_by_image.get(int(left), -1) == updated_cluster_by_image.get(int(right), -2)
        for left, right in zip(remaining_violation_df["image_id_a"], remaining_violation_df["image_id_b"])
    ]
    remaining_violation_df = remaining_violation_df[remaining_violation_df["same_cluster"]].copy().reset_index(drop=True)
    if not remaining_violation_df.empty:
        remaining_violation_df["cluster_id"] = [
            int(updated_cluster_by_image[int(image_id)]) for image_id in remaining_violation_df["image_id_a"]
        ]
        remaining_violation_df["cluster_size"] = remaining_violation_df["cluster_id"].map(
            texas_df["pred_cluster_id"].value_counts().astype(int).to_dict()
        )
    else:
        remaining_violation_df["cluster_id"] = []
        remaining_violation_df["cluster_size"] = []
    remaining_violation_df.to_csv(tables_dir / "remaining_cannot_link_violations_v1.csv", index=False)

    config = {
        "base_predictions": str(args.base_predictions.resolve()),
        "registry_path": str(args.registry_path.resolve()),
        "route_name": str(args.route_name),
        "split_policy": str(args.split_policy),
        "skip_clusters_with_internal_must_link": bool(args.skip_clusters_with_internal_must_link),
        "violation_pair_count": int(len(violation_df)),
        "remaining_violation_pair_count": int(len(remaining_violation_df)),
        "clusters_with_violations": int(violation_df["cluster_id"].nunique()) if not violation_df.empty else 0,
        "changed_images": int(len(changed_image_ids)),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> graph-v1 manual positive overlay -> conservative cannot-link split -> submission cluster label`.",
        "- Current route:",
        "  - `LynxID2025`: keep base route unchanged.",
        "  - `SalamanderID2025`: keep base route unchanged.",
        "  - `SeaTurtleID2022`: keep base route unchanged.",
        "  - `TexasHornedLizards`: start from `0.59138` graph-v1 Texas route, then only split unresolved `cannot-link` violations that sit in pure-negative current clusters.",
    ]

    summary_lines = [
        "# Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Pair registry: `{args.registry_path.resolve()}`",
        f"- Route name: `{args.route_name}`",
        f"- Split policy: `{args.split_policy}`",
        f"- Skip mixed must-link clusters: `{bool(args.skip_clusters_with_internal_must_link)}`",
        f"- Current unresolved cannot-link pairs on base route: `{int(len(violation_df))}`",
        f"- Current unresolved cannot-link clusters on base route: `{int(violation_df['cluster_id'].nunique()) if not violation_df.empty else 0}`",
        f"- Changed Texas images: `{int(len(changed_image_ids))}`",
        "",
        "## Architecture",
        "",
        *architecture_lines,
        "",
        "## Texas Graph Delta",
        "",
        f"- `must-link` pairs in registry: `{int(registry_df['constraint_type'].eq('must-link').sum())}`",
        f"- `cannot-link` pairs in registry: `{int(registry_df['constraint_type'].eq('cannot-link').sum())}`",
        f"- Unresolved cannot-link pairs before v2: `{int(len(violation_df))}`",
        f"- Remaining cannot-link pairs after v2: `{int(len(remaining_violation_df))}`",
        f"- Pure-negative clusters split by v2: `{int(cluster_action_df['action'].eq('singletonize_violating_nodes').sum()) if not cluster_action_df.empty else 0}`",
        f"- Mixed clusters skipped by v2: `{int(cluster_action_df['action'].eq('skip_internal_must_link_conflict').sum()) if not cluster_action_df.empty else 0}`",
        "",
        "## Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Cluster Actions",
        "",
        _format_markdown_table(cluster_action_df, limit=40),
        "",
        "## Remaining Violation Pairs",
        "",
        _format_markdown_table(
            remaining_violation_df[
                [
                    column
                    for column in [
                        "pair_id",
                        "image_id_a",
                        "image_id_b",
                        "cluster_id",
                        "cluster_size",
                        "support_count",
                        "sources",
                    ]
                    if column in violation_df.columns
                ]
            ],
            limit=40,
        ),
        "",
        "## Sample Delta",
        "",
        _format_markdown_table(delta_df, limit=40),
        "",
        "## Key Performance Tricks",
        "",
        "- `baseline-preserving upgrade`: start from the already validated `0.59138` Texas graph-v1 route instead of reopening a wider retrain or recluster.",
        "- `pure-negative split only`: only touch clusters where remaining evidence is unresolved `cannot-link`, so this variant changes the smallest possible subset.",
        "- `must-link protection`: if a current cluster already contains internal `must-link` support, keep it unchanged to avoid breaking a previously validated positive component.",
        "- `single-factor submission design`: this variant changes only Texas postprocess, so any leaderboard movement stays attributable.",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[texas_graph_v2] submission: {output_dir / 'submission.csv'}")
    print(f"[texas_graph_v2] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[texas_graph_v2] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
