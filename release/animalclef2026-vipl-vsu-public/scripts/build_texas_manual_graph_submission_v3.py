#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, item: int) -> int:
        key = int(item)
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(int(left))
        root_right = self.find(int(right))
        if root_left != root_right:
            self.parent[root_right] = root_left


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build Texas graph v3 by protecting reviewed must-link components and splitting reviewed unsupported members."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_seeded_attach_on_059341_base_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v3/texas_pair_registry_v3.csv",
    )
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=repo_root / "sample_submission.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_manual_graph_v3_on_061885_base_v1",
    )
    parser.add_argument("--route-name", type=str, default="texas_manual_graph_v3")
    return parser.parse_args()


def _load_components(registry_df: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, int]]:
    must_link_df = registry_df[registry_df["constraint_type"].astype(str).eq("must-link")].copy()
    uf = UnionFind()
    for row in must_link_df.itertuples(index=False):
        uf.union(int(row.image_id_a), int(row.image_id_b))
    image_to_component: dict[int, int] = {}
    component_to_members: dict[int, list[int]] = defaultdict(list)
    for image_id in sorted(
        set(must_link_df["image_id_a"].astype(int).tolist()).union(set(must_link_df["image_id_b"].astype(int).tolist()))
    ):
        root = int(uf.find(int(image_id)))
        image_to_component[int(image_id)] = root
        component_to_members[root].append(int(image_id))
    component_to_members = {int(root): sorted(values) for root, values in component_to_members.items()}
    return component_to_members, image_to_component


def _format_markdown_table(frame: pd.DataFrame, limit: int = 40) -> str:
    if frame.empty:
        return "_Empty_"
    preview = frame.head(int(limit)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(str(row[column]).replace("|", "\\|").replace("\n", "<br>") for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root / "src"))
    from animalclef_analysis.descriptor_baselines import build_submission, dataframe_to_markdown_table

    pred_df = pd.read_csv(args.base_predictions.resolve()).copy()
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

    registry_df = pd.read_csv(args.registry_path.resolve()).copy()
    registry_df["image_id_a"] = pd.to_numeric(registry_df["image_id_a"], errors="coerce").fillna(-1).astype(int)
    registry_df["image_id_b"] = pd.to_numeric(registry_df["image_id_b"], errors="coerce").fillna(-1).astype(int)
    registry_df["constraint_type"] = registry_df["constraint_type"].astype(str)
    registry_df["support_count"] = pd.to_numeric(registry_df["support_count"], errors="coerce").fillna(0).astype(int)

    texas_mask = pred_df["dataset"].eq(TEXAS_DATASET)
    texas_df = pred_df.loc[texas_mask].copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(int)
    base_texas_df = texas_df.copy()

    component_to_members, image_to_component = _load_components(registry_df)
    cluster_by_image = dict(zip(texas_df["image_id"].astype(int), texas_df["pred_cluster_id"].astype(int)))
    cluster_members = {
        int(cluster_id): sorted(frame["image_id"].astype(int).tolist())
        for cluster_id, frame in texas_df.groupby("pred_cluster_id", sort=True)
    }

    cannot_df = registry_df[registry_df["constraint_type"].eq("cannot-link")].copy()
    cannot_df["same_cluster"] = [
        cluster_by_image.get(int(left), -1) == cluster_by_image.get(int(right), -2)
        for left, right in zip(cannot_df["image_id_a"], cannot_df["image_id_b"])
    ]
    violation_df = cannot_df[cannot_df["same_cluster"]].copy().reset_index(drop=True)
    if not violation_df.empty:
        violation_df["cluster_id"] = [int(cluster_by_image[int(image_id)]) for image_id in violation_df["image_id_a"]]
    else:
        violation_df["cluster_id"] = []
    violation_df.to_csv(tables_dir / "cannot_link_violations_v1.csv", index=False)

    next_cluster_id = int(texas_df["pred_cluster_id"].max()) + 1
    cluster_action_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    changed_image_ids: set[int] = set()

    for cluster_id, cluster_violation_df in violation_df.groupby("cluster_id", sort=True):
        member_image_ids = sorted(cluster_members.get(int(cluster_id), []))
        component_members: dict[int, list[int]] = {}
        component_sizes: dict[int, int] = {}
        for image_id in member_image_ids:
            component_id = int(image_to_component.get(int(image_id), int(image_id)))
            component_members.setdefault(component_id, []).append(int(image_id))
        component_members = {int(comp_id): sorted(values) for comp_id, values in component_members.items()}
        component_sizes = {int(comp_id): int(len(values)) for comp_id, values in component_members.items()}

        conflicting_components: set[int] = set()
        protected_components: set[int] = {int(comp_id) for comp_id, size in component_sizes.items() if int(size) > 1}
        for row in cluster_violation_df.itertuples(index=False):
            left_comp = int(image_to_component.get(int(row.image_id_a), int(row.image_id_a)))
            right_comp = int(image_to_component.get(int(row.image_id_b), int(row.image_id_b)))
            if left_comp == right_comp:
                continue
            conflicting_components.add(left_comp)
            conflicting_components.add(right_comp)

        components_to_split = sorted(int(comp_id) for comp_id in conflicting_components if int(comp_id) not in protected_components)
        protected_conflict_components = sorted(int(comp_id) for comp_id in conflicting_components if int(comp_id) in protected_components)

        if not components_to_split:
            cluster_action_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(member_image_ids)),
                    "violation_pair_count": int(len(cluster_violation_df)),
                    "conflicting_components": "|".join(str(value) for value in sorted(conflicting_components)),
                    "protected_components": "|".join(str(value) for value in protected_components),
                    "components_split": "",
                    "changed_image_ids": "",
                    "action": "skip_no_unprotected_component",
                    "note": "cluster has internal cannot-link but only protected must-link components participate",
                }
            )
            continue

        changed_images_for_cluster: list[int] = []
        assigned_cluster_ids: list[int] = []
        for component_id in components_to_split:
            members = component_members.get(int(component_id), [])
            # Unprotected components are singletons under the current review protocol.
            for image_id in members:
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
                texas_df.at[idx, "manual_overlay_rule"] = f"{args.route_name}|protected_component_split|cluster_{int(cluster_id)}"
                texas_df.at[idx, "manual_overlay_operation_id"] = (
                    f"texas_protected_component_split_cluster_{int(cluster_id)}_{int(image_id)}"
                )
                texas_df.at[idx, "manual_overlay_note"] = (
                    "manual protected-component split"
                    f" | base_cluster={int(cluster_id)}"
                    f" | protected_components={'|'.join(str(value) for value in protected_components) or '-'}"
                    f" | conflicting_components={'|'.join(str(value) for value in sorted(conflicting_components)) or '-'}"
                )
                changed_image_ids.add(int(image_id))
                changed_images_for_cluster.append(int(image_id))
                assigned_cluster_ids.append(int(assigned_cluster_id))
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
                        "change_reason": "protected_component_split",
                    }
                )

        cluster_action_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(member_image_ids)),
                "violation_pair_count": int(len(cluster_violation_df)),
                "conflicting_components": "|".join(str(value) for value in sorted(conflicting_components)),
                "protected_components": "|".join(str(value) for value in protected_components),
                "components_split": "|".join(str(value) for value in components_to_split),
                "changed_image_ids": "|".join(str(value) for value in changed_images_for_cluster),
                "new_cluster_ids": "|".join(str(value) for value in assigned_cluster_ids),
                "action": "split_unprotected_conflicting_components",
                "note": (
                    "keep reviewed must-link component(s) intact and split only unsupported members that conflict inside cluster"
                    if protected_conflict_components
                    else "split unsupported conflicting singleton components"
                ),
            }
        )

    cluster_action_df = pd.DataFrame(cluster_action_rows) if cluster_action_rows else pd.DataFrame(
        columns=[
            "cluster_id",
            "cluster_size",
            "violation_pair_count",
            "conflicting_components",
            "protected_components",
            "components_split",
            "changed_image_ids",
            "new_cluster_ids",
            "action",
            "note",
        ]
    )
    if not cluster_action_df.empty:
        cluster_action_df = cluster_action_df.sort_values(["cluster_id"], ascending=[True]).reset_index(drop=True)
    cluster_action_df.to_csv(tables_dir / "cluster_actions_v1.csv", index=False)

    delta_df = pd.DataFrame(delta_rows) if delta_rows else pd.DataFrame(
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
    if not delta_df.empty:
        delta_df = delta_df.sort_values(["pred_cluster_id_base", "image_id"], ascending=[True, True]).reset_index(drop=True)
    delta_df.to_csv(tables_dir / "texas_manual_graph_delta_v3.csv", index=False)

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
                "variant": "base_seeded_attach",
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
                "variant": "graph_v3_reviewed_split",
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
    remaining_violation_df = cannot_df.copy()
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

    summary_json = {
        "base_predictions": str(args.base_predictions.resolve()),
        "registry_path": str(args.registry_path.resolve()),
        "route_name": str(args.route_name),
        "violation_pair_count_before": int(len(violation_df)),
        "violation_pair_count_after": int(len(remaining_violation_df)),
        "clusters_with_internal_conflicts": int(violation_df["cluster_id"].nunique()) if not violation_df.empty else 0,
        "changed_images": int(len(changed_image_ids)),
        "changed_image_ids": sorted(int(value) for value in changed_image_ids),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> Texas seeded-attach base -> reviewed pair registry v3 -> protect must-link components -> split unsupported conflicting members -> submission cluster label`.",
        "- Current route:",
        "  - `LynxID2025`: keep base route unchanged.",
        "  - `SalamanderID2025`: keep base route unchanged.",
        "  - `SeaTurtleID2022`: keep base route unchanged.",
        "  - `TexasHornedLizards`: start from the current `0.61885` seeded-attach route, then apply user-reviewed negative constraints without breaking reviewed positive components.",
    ]

    summary_lines = [
        "# Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        f"- Pair registry: `{args.registry_path.resolve()}`",
        f"- Route name: `{args.route_name}`",
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
        f"- Internal cannot-link pairs on base route: `{int(len(violation_df))}`",
        f"- Remaining internal cannot-link pairs after graph v3: `{int(len(remaining_violation_df))}`",
        f"- Clusters touched by graph v3: `{int(cluster_action_df['action'].eq('split_unprotected_conflicting_components').sum()) if not cluster_action_df.empty else 0}`",
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
                    if column in remaining_violation_df.columns
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
        "- `review-first negatives`: only add splits for pairs you explicitly rejected during HTML review.",
        "- `must-link protection`: keep reviewed positive components intact even when a cluster contains a conflicting outsider.",
        "- `smallest-change policy`: split only unsupported conflicting members instead of reclustering whole Texas.",
        "- `base-preserving route`: all other datasets and all untouched Texas samples remain identical to the current best submission.",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"[texas_graph_v3] submission: {output_dir / 'submission.csv'}")
    print(f"[texas_graph_v3] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[texas_graph_v3] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
