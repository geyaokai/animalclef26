from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .manual_review_workbench import (
    PAIR_LABEL_NO,
    PAIR_LABEL_UNCERTAIN,
    PAIR_LABEL_YES,
    add_attach_operation,
    add_split_operation,
    judgments_to_dataframe,
    load_pair_judgments,
)


def _sorted_pair_key(left: str, right: str) -> tuple[str, str]:
    left_key = str(left)
    right_key = str(right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


def _normalize_split_judgment_frame(judgments: list[dict[str, Any]]) -> pd.DataFrame:
    judgment_df = judgments_to_dataframe(judgments)
    if judgment_df.empty:
        return judgment_df
    frame = judgment_df[
        judgment_df["candidate_type"].astype(str).eq("split")
        & judgment_df["label"].astype(str).isin([PAIR_LABEL_YES, PAIR_LABEL_NO, PAIR_LABEL_UNCERTAIN])
    ].copy()
    if frame.empty:
        return frame
    for column in ["candidate_key", "dataset", "image_id", "neighbor_image_id", "label"]:
        frame[column] = frame[column].astype(str)
    frame["xgb_same_identity_prob"] = pd.to_numeric(
        frame["xgb_same_identity_prob"], errors="coerce"
    ).fillna(0.0)
    frame["ambiguity_score"] = pd.to_numeric(frame["ambiguity_score"], errors="coerce").fillna(0.0)
    return frame


def _normalize_pair_df(pair_df: pd.DataFrame) -> pd.DataFrame:
    frame = pair_df.copy()
    if frame.empty:
        return frame
    for column in ["image_id", "neighbor_image_id"]:
        frame[column] = frame[column].astype(str)
    for column in ["base_cluster_left", "base_cluster_right"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(-1).astype(int)
    frame["xgb_same_identity_prob"] = pd.to_numeric(
        frame.get("xgb_same_identity_prob", 0.0), errors="coerce"
    ).fillna(0.0)
    frame["ambiguity_score"] = pd.to_numeric(frame.get("ambiguity_score", 0.0), errors="coerce").fillna(0.0)
    return frame


def _build_node_priority_table(
    node_ids: list[str],
    judgment_df: pd.DataFrame,
    *,
    dataset: str,
    base_cluster_id: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_id in sorted({str(value) for value in node_ids}):
        left_rows = judgment_df[judgment_df["image_id"].astype(str).eq(image_id)].copy()
        right_rows = judgment_df[judgment_df["neighbor_image_id"].astype(str).eq(image_id)].copy()
        related = pd.concat([left_rows, right_rows], ignore_index=True)
        yes_degree = int(related["label"].astype(str).eq(PAIR_LABEL_YES).sum()) if not related.empty else 0
        no_degree = int(related["label"].astype(str).eq(PAIR_LABEL_NO).sum()) if not related.empty else 0
        uncertain_degree = int(related["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum()) if not related.empty else 0
        rows.append(
            {
                "dataset": str(dataset),
                "base_cluster_id": int(base_cluster_id),
                "image_id": str(image_id),
                "yes_degree": yes_degree,
                "no_degree": no_degree,
                "uncertain_degree": uncertain_degree,
                "anchor_priority": int(yes_degree - no_degree),
            }
        )
    return pd.DataFrame(rows)


def _choose_anchor_image(component_image_ids: list[str], node_priority_df: pd.DataFrame) -> str:
    component_set = {str(value) for value in component_image_ids}
    sortable = node_priority_df[node_priority_df["image_id"].astype(str).isin(component_set)].copy()
    if sortable.empty:
        return sorted(component_set)[0]
    sortable = sortable.sort_values(
        ["anchor_priority", "yes_degree", "no_degree", "image_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return str(sortable.iloc[0]["image_id"])


class _UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        unique_nodes = [str(value) for value in nodes]
        self.parent: dict[str, str] = {node: node for node in unique_nodes}
        self.size: dict[str, int] = {node: 1 for node in unique_nodes}
        self.members: dict[str, set[str]] = {node: {node} for node in unique_nodes}

    def find(self, node: str) -> str:
        key = str(node)
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> str:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return left_root
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        self.members[left_root].update(self.members[right_root])
        del self.size[right_root]
        del self.members[right_root]
        return left_root

    def component_members(self, root: str) -> set[str]:
        return self.members[self.find(root)]

    def components(self) -> list[list[str]]:
        return sorted(
            [sorted(values) for values in self.members.values()],
            key=lambda values: (len(values), values),
            reverse=True,
        )


def _find_blocking_pair(
    left_members: set[str],
    right_members: set[str],
    cannot_link_adj: dict[str, set[str]],
) -> tuple[str, str] | None:
    if len(left_members) <= len(right_members):
        probe_members = sorted(left_members)
        other_members = right_members
    else:
        probe_members = sorted(right_members)
        other_members = left_members
    for image_id in probe_members:
        neighbors = cannot_link_adj.get(str(image_id), set())
        conflict = sorted(neighbors.intersection(other_members))
        if conflict:
            return _sorted_pair_key(str(image_id), str(conflict[0]))
    return None


def _component_internal_stats(
    component_image_ids: list[str],
    pair_info_df: pd.DataFrame,
) -> tuple[int, float, int, int]:
    component_set = {str(value) for value in component_image_ids}
    subset = pair_info_df[
        pair_info_df["image_id"].astype(str).isin(component_set)
        & pair_info_df["neighbor_image_id"].astype(str).isin(component_set)
    ].copy()
    if subset.empty:
        return 0, 0.0, 0, 0
    internal_mean_score = round(float(subset["score"].astype(float).mean()), 6)
    internal_yes_edges = int(subset["manual_label"].astype(str).eq(PAIR_LABEL_YES).sum())
    internal_no_edges = int(subset["manual_label"].astype(str).eq(PAIR_LABEL_NO).sum())
    return int(len(subset)), internal_mean_score, internal_yes_edges, internal_no_edges


def _build_local_pair_table(
    node_ids: list[str],
    cluster_pair_df: pd.DataFrame,
    cluster_judgment_df: pd.DataFrame,
) -> pd.DataFrame:
    pair_payload: dict[tuple[str, str], dict[str, Any]] = {}
    node_set = {str(value) for value in node_ids}

    for row in cluster_pair_df.itertuples(index=False):
        left_key, right_key = _sorted_pair_key(str(row.image_id), str(row.neighbor_image_id))
        if left_key not in node_set or right_key not in node_set or left_key == right_key:
            continue
        existing = pair_payload.get((left_key, right_key))
        score = round(float(getattr(row, "xgb_same_identity_prob", 0.0)), 6)
        ambiguity_score = round(float(getattr(row, "ambiguity_score", 0.0)), 6)
        if existing is None or score > float(existing["score"]):
            pair_payload[(left_key, right_key)] = {
                "image_id": left_key,
                "neighbor_image_id": right_key,
                "score": score,
                "ambiguity_score": ambiguity_score,
                "manual_label": "",
                "from_probe": True,
                "from_judgment": False,
            }

    for row in cluster_judgment_df.itertuples(index=False):
        left_key, right_key = _sorted_pair_key(str(row.image_id), str(row.neighbor_image_id))
        if left_key not in node_set or right_key not in node_set or left_key == right_key:
            continue
        payload = pair_payload.setdefault(
            (left_key, right_key),
            {
                "image_id": left_key,
                "neighbor_image_id": right_key,
                "score": round(float(getattr(row, "xgb_same_identity_prob", 0.0)), 6),
                "ambiguity_score": round(float(getattr(row, "ambiguity_score", 0.0)), 6),
                "manual_label": "",
                "from_probe": False,
                "from_judgment": True,
            },
        )
        payload["manual_label"] = str(getattr(row, "label", "")).strip().lower()
        payload["from_judgment"] = True
        payload["score"] = round(max(float(payload["score"]), float(getattr(row, "xgb_same_identity_prob", 0.0))), 6)
        payload["ambiguity_score"] = round(
            max(float(payload["ambiguity_score"]), float(getattr(row, "ambiguity_score", 0.0))),
            6,
        )

    pair_info_df = pd.DataFrame(pair_payload.values())
    if pair_info_df.empty:
        return pd.DataFrame(
            columns=[
                "image_id",
                "neighbor_image_id",
                "score",
                "ambiguity_score",
                "manual_label",
                "from_probe",
                "from_judgment",
            ]
        )
    return pair_info_df.sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)


def _run_constrained_graph_partition(
    *,
    node_ids: list[str],
    pair_info_df: pd.DataFrame,
    graph_threshold: float,
) -> tuple[list[list[str]], pd.DataFrame]:
    cannot_link_adj: dict[str, set[str]] = defaultdict(set)
    for row in pair_info_df.itertuples(index=False):
        if str(row.manual_label) != PAIR_LABEL_NO:
            continue
        cannot_link_adj[str(row.image_id)].add(str(row.neighbor_image_id))
        cannot_link_adj[str(row.neighbor_image_id)].add(str(row.image_id))

    edge_rows = pair_info_df.copy()
    if edge_rows.empty:
        edge_rows["considered_for_merge"] = []
        edge_rows["processed_order"] = []
        edge_rows["decision"] = []
        edge_rows["blocking_pair"] = []
        return [sorted({str(value) for value in node_ids})], edge_rows

    edge_rows["considered_for_merge"] = (
        edge_rows["score"].astype(float).ge(float(graph_threshold))
        | edge_rows["manual_label"].astype(str).eq(PAIR_LABEL_YES)
    )
    edge_rows["processed_order"] = -1
    edge_rows["decision"] = ""
    edge_rows["blocking_pair"] = ""

    sortable = edge_rows.copy()
    sortable["label_priority"] = sortable["manual_label"].map({PAIR_LABEL_YES: 0, PAIR_LABEL_UNCERTAIN: 2}).fillna(1)
    sortable = sortable.sort_values(
        ["label_priority", "score", "image_id", "neighbor_image_id"],
        ascending=[True, False, True, True],
    ).reset_index(drop=False)

    union_find = _UnionFind(sorted({str(value) for value in node_ids}))
    processed_order = 0
    for row in sortable.itertuples(index=False):
        frame_index = int(row.index)
        label = str(row.manual_label)
        if label == PAIR_LABEL_NO:
            edge_rows.at[frame_index, "decision"] = "skip_manual_no"
            continue
        if not bool(row.considered_for_merge):
            edge_rows.at[frame_index, "decision"] = "skip_below_threshold"
            continue
        processed_order += 1
        edge_rows.at[frame_index, "processed_order"] = int(processed_order)
        left_image_id = str(row.image_id)
        right_image_id = str(row.neighbor_image_id)
        left_root = union_find.find(left_image_id)
        right_root = union_find.find(right_image_id)
        if left_root == right_root:
            edge_rows.at[frame_index, "decision"] = "skip_same_component"
            continue
        blocking_pair = _find_blocking_pair(
            union_find.component_members(left_root),
            union_find.component_members(right_root),
            cannot_link_adj,
        )
        if blocking_pair is not None:
            edge_rows.at[frame_index, "decision"] = (
                "block_even_yes_due_cannot_link" if label == PAIR_LABEL_YES else "block_due_cannot_link"
            )
            edge_rows.at[frame_index, "blocking_pair"] = "|".join(blocking_pair)
            continue
        union_find.union(left_root, right_root)
        edge_rows.at[frame_index, "decision"] = "union"

    components = union_find.components()
    return components, edge_rows.sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)


def _choose_anchor_component(
    components: list[list[str]],
    pair_info_df: pd.DataFrame,
    node_priority_df: pd.DataFrame,
) -> tuple[int, str]:
    component_rows: list[dict[str, Any]] = []
    for component_index, component_image_ids in enumerate(components):
        internal_edge_count, internal_mean_score, internal_yes_edges, internal_no_edges = _component_internal_stats(
            component_image_ids,
            pair_info_df,
        )
        anchor_image_id = _choose_anchor_image(component_image_ids, node_priority_df)
        component_rows.append(
            {
                "component_index": int(component_index),
                "component_image_ids": list(component_image_ids),
                "component_size": int(len(component_image_ids)),
                "internal_edge_count": int(internal_edge_count),
                "internal_mean_score": float(internal_mean_score),
                "internal_yes_edges": int(internal_yes_edges),
                "internal_no_edges": int(internal_no_edges),
                "anchor_image_id": str(anchor_image_id),
            }
        )
    sortable = pd.DataFrame(component_rows).sort_values(
        ["component_size", "internal_mean_score", "internal_yes_edges", "anchor_image_id"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = sortable.iloc[0]
    return int(best["component_index"]), str(best["anchor_image_id"])


def compile_constraint_graph_to_overlay(
    pred_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    judgments: list[dict[str, Any]],
    *,
    datasets: list[str] | tuple[str, ...] | None = None,
    candidate_keys: list[str] | tuple[str, ...] | None = None,
    graph_threshold: float = 0.25,
    min_judged_pairs: int = 1,
    min_no_pairs: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_frame = pred_df.copy()
    pred_frame["dataset"] = pred_frame["dataset"].astype(str)
    pred_frame["image_id"] = pred_frame["image_id"].astype(str)
    pred_frame["pred_cluster_id"] = pd.to_numeric(pred_frame["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)

    pair_frame = _normalize_pair_df(pair_df)
    judgment_df = _normalize_split_judgment_frame(judgments)
    if datasets:
        dataset_values = [str(value) for value in datasets]
        judgment_df = judgment_df[judgment_df["dataset"].astype(str).isin(dataset_values)].copy()
    if candidate_keys:
        candidate_values = [str(value) for value in candidate_keys]
        judgment_df = judgment_df[judgment_df["candidate_key"].astype(str).isin(candidate_values)].copy()

    operations: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    if judgment_df.empty:
        return operations, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for (dataset, candidate_key), cluster_judgment_df in judgment_df.groupby(["dataset", "candidate_key"], sort=True):
        cluster_judgment_df = cluster_judgment_df.sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)
        judged_pairs = int(len(cluster_judgment_df))
        yes_pairs = int(cluster_judgment_df["label"].astype(str).eq(PAIR_LABEL_YES).sum())
        no_pairs = int(cluster_judgment_df["label"].astype(str).eq(PAIR_LABEL_NO).sum())
        uncertain_pairs = int(cluster_judgment_df["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum())
        try:
            base_cluster_id = int(str(candidate_key))
        except ValueError:
            candidate_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": -1,
                    "candidate_key": str(candidate_key),
                    "subgraph_size": 0,
                    "judged_pairs": int(judged_pairs),
                    "yes_pairs": int(yes_pairs),
                    "no_pairs": int(no_pairs),
                    "uncertain_pairs": int(uncertain_pairs),
                    "graph_pairs": 0,
                    "component_count": 0,
                    "moved_images": 0,
                    "moved_component_count": 0,
                    "anchor_component_size": 0,
                    "blocked_edges": 0,
                    "blocked_yes_edges": 0,
                    "anchor_image_id": "",
                    "compile_status": "skip_non_integer_candidate_key",
                    "generated_operation_count": 0,
                }
            )
            continue

        if judged_pairs < int(min_judged_pairs) or no_pairs < int(min_no_pairs):
            candidate_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": int(base_cluster_id),
                    "candidate_key": str(candidate_key),
                    "subgraph_size": 0,
                    "judged_pairs": int(judged_pairs),
                    "yes_pairs": int(yes_pairs),
                    "no_pairs": int(no_pairs),
                    "uncertain_pairs": int(uncertain_pairs),
                    "graph_pairs": 0,
                    "component_count": 0,
                    "moved_images": 0,
                    "moved_component_count": 0,
                    "anchor_component_size": 0,
                    "blocked_edges": 0,
                    "blocked_yes_edges": 0,
                    "anchor_image_id": "",
                    "compile_status": "skip_below_review_gate",
                    "generated_operation_count": 0,
                }
            )
            continue

        cluster_pair_df = pair_frame[
            pair_frame["base_cluster_left"].astype(int).eq(int(base_cluster_id))
            & pair_frame["base_cluster_right"].astype(int).eq(int(base_cluster_id))
        ].copy()
        node_ids = sorted(
            {
                str(value)
                for value in cluster_pair_df["image_id"].astype(str).tolist()
                + cluster_pair_df["neighbor_image_id"].astype(str).tolist()
                + cluster_judgment_df["image_id"].astype(str).tolist()
                + cluster_judgment_df["neighbor_image_id"].astype(str).tolist()
            }
        )
        if not node_ids:
            fallback_nodes = pred_frame[
                pred_frame["dataset"].astype(str).eq(str(dataset))
                & pred_frame["pred_cluster_id"].astype(int).eq(int(base_cluster_id))
            ]["image_id"].astype(str).tolist()
            node_ids = sorted({str(value) for value in fallback_nodes})

        if not node_ids:
            candidate_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": int(base_cluster_id),
                    "candidate_key": str(candidate_key),
                    "subgraph_size": 0,
                    "judged_pairs": int(judged_pairs),
                    "yes_pairs": int(yes_pairs),
                    "no_pairs": int(no_pairs),
                    "uncertain_pairs": int(uncertain_pairs),
                    "graph_pairs": 0,
                    "component_count": 0,
                    "moved_images": 0,
                    "moved_component_count": 0,
                    "anchor_component_size": 0,
                    "blocked_edges": 0,
                    "blocked_yes_edges": 0,
                    "anchor_image_id": "",
                    "compile_status": "skip_empty_subgraph",
                    "generated_operation_count": 0,
                }
            )
            continue

        pair_info_df = _build_local_pair_table(node_ids, cluster_pair_df, cluster_judgment_df)
        node_priority_df = _build_node_priority_table(node_ids, cluster_judgment_df, dataset=str(dataset), base_cluster_id=int(base_cluster_id))

        start_operation_count = len(operations)
        if pair_info_df.empty:
            components = [sorted(node_ids)]
            edge_info_df = pd.DataFrame(
                columns=[
                    "image_id",
                    "neighbor_image_id",
                    "score",
                    "ambiguity_score",
                    "manual_label",
                    "from_probe",
                    "from_judgment",
                    "considered_for_merge",
                    "processed_order",
                    "decision",
                    "blocking_pair",
                ]
            )
            compile_status = "skip_no_graph_pairs"
            anchor_component_index = 0
            anchor_image_id = _choose_anchor_image(components[0], node_priority_df)
        else:
            components, edge_info_df = _run_constrained_graph_partition(
                node_ids=node_ids,
                pair_info_df=pair_info_df,
                graph_threshold=float(graph_threshold),
            )
            anchor_component_index, anchor_image_id = _choose_anchor_component(components, pair_info_df, node_priority_df)
            compile_status = "compiled_constrained_split" if len(components) > 1 else "skip_single_component"

        for row in edge_info_df.itertuples(index=False):
            edge_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": int(base_cluster_id),
                    "image_id": str(row.image_id),
                    "neighbor_image_id": str(row.neighbor_image_id),
                    "score": round(float(row.score), 6),
                    "ambiguity_score": round(float(row.ambiguity_score), 6),
                    "manual_label": str(row.manual_label),
                    "from_probe": bool(row.from_probe),
                    "from_judgment": bool(row.from_judgment),
                    "considered_for_merge": bool(row.considered_for_merge),
                    "processed_order": int(row.processed_order),
                    "decision": str(row.decision),
                    "blocking_pair": str(row.blocking_pair),
                }
            )

        moved_images: list[str] = []
        moved_component_count = 0
        if len(components) > 1:
            anchor_component = sorted(components[int(anchor_component_index)])
            for component_index, component_image_ids in enumerate(components):
                if int(component_index) == int(anchor_component_index):
                    continue
                moved_images.extend(sorted(component_image_ids))
                moved_component_count += 1
            if moved_images:
                operations = add_split_operation(
                    operations,
                    dataset=str(dataset),
                    cluster_id=int(base_cluster_id),
                    anchor_image_id=str(anchor_image_id),
                    member_image_ids=sorted({str(value) for value in moved_images}),
                    note=(
                        "compiled from manual constraint graph"
                        f" | cluster={base_cluster_id}"
                        f" | threshold={float(graph_threshold):.2f}"
                        f" | moved={'|'.join(sorted({str(value) for value in moved_images}))}"
                    ),
                )
                for component_index, component_image_ids in enumerate(components):
                    if int(component_index) == int(anchor_component_index):
                        continue
                    component_image_ids = sorted(component_image_ids)
                    if len(component_image_ids) <= 1:
                        continue
                    component_anchor_image_id = _choose_anchor_image(component_image_ids, node_priority_df)
                    component_members = [
                        str(value)
                        for value in component_image_ids
                        if str(value) != str(component_anchor_image_id)
                    ]
                    if not component_members:
                        continue
                    operations = add_attach_operation(
                        operations,
                        dataset=str(dataset),
                        anchor_image_id=str(component_anchor_image_id),
                        member_image_ids=component_members,
                        source_cluster_ids=[],
                        note=(
                            "compiled regroup from manual constraint graph"
                            f" | cluster={base_cluster_id}"
                            f" | component={'|'.join(component_image_ids)}"
                        ),
                    )

        for component_index, component_image_ids in enumerate(components):
            component_image_ids = sorted(component_image_ids)
            internal_edge_count, internal_mean_score, internal_yes_edges, internal_no_edges = _component_internal_stats(
                component_image_ids,
                pair_info_df,
            )
            component_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": int(base_cluster_id),
                    "component_index": int(component_index),
                    "component_image_ids": "|".join(component_image_ids),
                    "component_size": int(len(component_image_ids)),
                    "is_anchor_component": bool(int(component_index) == int(anchor_component_index)),
                    "anchor_image_id": str(
                        anchor_image_id if int(component_index) == int(anchor_component_index) else _choose_anchor_image(component_image_ids, node_priority_df)
                    ),
                    "internal_edge_count": int(internal_edge_count),
                    "internal_mean_score": float(internal_mean_score),
                    "internal_yes_edges": int(internal_yes_edges),
                    "internal_no_edges": int(internal_no_edges),
                }
            )

        blocked_edges = int(edge_info_df["decision"].astype(str).isin(["block_due_cannot_link", "block_even_yes_due_cannot_link"]).sum()) if not edge_info_df.empty else 0
        blocked_yes_edges = int(edge_info_df["decision"].astype(str).eq("block_even_yes_due_cannot_link").sum()) if not edge_info_df.empty else 0
        anchor_component_size = int(len(components[int(anchor_component_index)])) if components else 0

        candidate_rows.append(
            {
                "dataset": str(dataset),
                "base_cluster_id": int(base_cluster_id),
                "candidate_key": str(candidate_key),
                "subgraph_size": int(len(node_ids)),
                "judged_pairs": int(judged_pairs),
                "yes_pairs": int(yes_pairs),
                "no_pairs": int(no_pairs),
                "uncertain_pairs": int(uncertain_pairs),
                "graph_pairs": int(len(pair_info_df)),
                "component_count": int(len(components)),
                "moved_images": int(len(sorted({str(value) for value in moved_images}))),
                "moved_component_count": int(moved_component_count),
                "anchor_component_size": int(anchor_component_size),
                "blocked_edges": int(blocked_edges),
                "blocked_yes_edges": int(blocked_yes_edges),
                "anchor_image_id": str(anchor_image_id),
                "compile_status": str(compile_status),
                "generated_operation_count": int(len(operations) - start_operation_count),
            }
        )

    candidate_summary_df = pd.DataFrame(candidate_rows)
    if not candidate_summary_df.empty:
        candidate_summary_df = candidate_summary_df.sort_values(
            ["compile_status", "dataset", "base_cluster_id"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    component_summary_df = pd.DataFrame(component_rows)
    if not component_summary_df.empty:
        component_summary_df = component_summary_df.sort_values(
            ["dataset", "base_cluster_id", "component_index"],
            ascending=[True, True, True],
        ).reset_index(drop=True)
    edge_summary_df = pd.DataFrame(edge_rows)
    if not edge_summary_df.empty:
        edge_summary_df = edge_summary_df.sort_values(
            ["dataset", "base_cluster_id", "image_id", "neighbor_image_id"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
    return operations, candidate_summary_df, component_summary_df, edge_summary_df


def compile_constraint_graph_file(
    *,
    base_predictions_path: str | Path,
    pair_graph_path: str | Path,
    pair_judgments_path: str | Path,
    datasets: list[str] | tuple[str, ...] | None = None,
    candidate_keys: list[str] | tuple[str, ...] | None = None,
    graph_threshold: float = 0.25,
    min_judged_pairs: int = 1,
    min_no_pairs: int = 1,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(Path(base_predictions_path).resolve())
    pair_df = pd.read_csv(Path(pair_graph_path).resolve())
    session_name, judgments = load_pair_judgments(Path(pair_judgments_path).resolve())
    operations, candidate_summary_df, component_summary_df, edge_summary_df = compile_constraint_graph_to_overlay(
        pred_df,
        pair_df,
        judgments,
        datasets=datasets,
        candidate_keys=candidate_keys,
        graph_threshold=float(graph_threshold),
        min_judged_pairs=int(min_judged_pairs),
        min_no_pairs=int(min_no_pairs),
    )
    return (
        session_name,
        judgments,
        operations,
        candidate_summary_df,
        component_summary_df,
        edge_summary_df,
    )
