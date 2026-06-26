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


def _sorted_unique_image_ids(values: list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _connected_components(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {str(node): set() for node in nodes}
    for left, right in edges:
        left_key = str(left)
        right_key = str(right)
        if left_key not in adjacency or right_key not in adjacency or left_key == right_key:
            continue
        adjacency[left_key].add(right_key)
        adjacency[right_key].add(left_key)

    components: list[list[str]] = []
    visited: set[str] = set()
    for node in sorted(adjacency.keys()):
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)
        components.append(sorted(component))
    return components


def _choose_anchor_image(image_score_df: pd.DataFrame) -> str:
    sortable = image_score_df.copy()
    sortable["anchor_priority"] = (
        sortable["yes_degree"].astype(int) - sortable["no_degree"].astype(int)
    )
    sortable = sortable.sort_values(
        ["anchor_priority", "yes_degree", "no_degree", "image_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return str(sortable.iloc[0]["image_id"])


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
    frame["candidate_key"] = frame["candidate_key"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["image_id"] = frame["image_id"].astype(str)
    frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)
    return frame


def _build_image_score_table(
    cluster_member_ids: list[str],
    candidate_df: pd.DataFrame,
    *,
    dataset: str,
    base_cluster_id: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_id in cluster_member_ids:
        left_rows = candidate_df[candidate_df["image_id"].astype(str).eq(str(image_id))].copy()
        right_rows = candidate_df[candidate_df["neighbor_image_id"].astype(str).eq(str(image_id))].copy()
        related = pd.concat([left_rows, right_rows], ignore_index=True)
        yes_mask = related["label"].astype(str).eq(PAIR_LABEL_YES)
        no_mask = related["label"].astype(str).eq(PAIR_LABEL_NO)
        uncertain_mask = related["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN)
        rows.append(
            {
                "dataset": str(dataset),
                "base_cluster_id": int(base_cluster_id),
                "image_id": str(image_id),
                "judged_pair_count": int(len(related)),
                "yes_degree": int(yes_mask.sum()),
                "no_degree": int(no_mask.sum()),
                "uncertain_degree": int(uncertain_mask.sum()),
                "mean_yes_probability": round(float(related.loc[yes_mask, "xgb_same_identity_prob"].astype(float).mean()), 6)
                if yes_mask.any()
                else 0.0,
                "mean_no_probability": round(float(related.loc[no_mask, "xgb_same_identity_prob"].astype(float).mean()), 6)
                if no_mask.any()
                else 0.0,
            }
        )
    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "base_cluster_id",
                "image_id",
                "judged_pair_count",
                "yes_degree",
                "no_degree",
                "uncertain_degree",
                "mean_yes_probability",
                "mean_no_probability",
                "net_no_margin",
            ]
        )
    score_df["net_no_margin"] = score_df["no_degree"].astype(int) - score_df["yes_degree"].astype(int)
    return score_df.sort_values("image_id").reset_index(drop=True)


def compile_split_judgments_to_overlay(
    pred_df: pd.DataFrame,
    judgments: list[dict[str, Any]],
    *,
    datasets: list[str] | tuple[str, ...] | None = None,
    candidate_keys: list[str] | tuple[str, ...] | None = None,
    min_no_degree: int = 2,
    min_net_no_margin: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_df = _normalize_split_judgment_frame(judgments)
    if datasets:
        split_df = split_df[split_df["dataset"].astype(str).isin([str(value) for value in datasets])].copy()
    if candidate_keys:
        split_df = split_df[split_df["candidate_key"].astype(str).isin([str(value) for value in candidate_keys])].copy()

    operations: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []

    if split_df.empty:
        return operations, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    pred_frame = pred_df.copy()
    pred_frame["dataset"] = pred_frame["dataset"].astype(str)
    pred_frame["image_id"] = pred_frame["image_id"].astype(str)
    pred_frame["pred_cluster_id"] = pred_frame["pred_cluster_id"].astype(int)

    for (dataset, candidate_key), candidate_df in split_df.groupby(["dataset", "candidate_key"], sort=True):
        candidate_df = candidate_df.sort_values(["image_id", "neighbor_image_id"]).reset_index(drop=True)
        try:
            base_cluster_id = int(str(candidate_key))
        except ValueError:
            candidate_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": -1,
                    "candidate_key": str(candidate_key),
                    "cluster_size": 0,
                    "judged_pairs": int(len(candidate_df)),
                    "yes_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_YES).sum()),
                    "no_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_NO).sum()),
                    "uncertain_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum()),
                    "selected_split_images": 0,
                    "attach_group_count": 0,
                    "anchor_image_id": "",
                    "compile_status": "skip_non_integer_candidate_key",
                    "generated_operation_count": 0,
                }
            )
            continue

        cluster_df = pred_frame[
            pred_frame["dataset"].astype(str).eq(str(dataset))
            & pred_frame["pred_cluster_id"].astype(int).eq(int(base_cluster_id))
        ].copy()
        cluster_member_ids = sorted(cluster_df["image_id"].astype(str).tolist())
        if cluster_df.empty:
            candidate_rows.append(
                {
                    "dataset": str(dataset),
                    "base_cluster_id": int(base_cluster_id),
                    "candidate_key": str(candidate_key),
                    "cluster_size": 0,
                    "judged_pairs": int(len(candidate_df)),
                    "yes_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_YES).sum()),
                    "no_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_NO).sum()),
                    "uncertain_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum()),
                    "selected_split_images": 0,
                    "attach_group_count": 0,
                    "anchor_image_id": "",
                    "compile_status": "skip_cluster_not_found",
                    "generated_operation_count": 0,
                }
            )
            continue

        image_score_df = _build_image_score_table(
            cluster_member_ids,
            candidate_df,
            dataset=str(dataset),
            base_cluster_id=int(base_cluster_id),
        )
        image_score_df["selected_for_split"] = (
            image_score_df["no_degree"].astype(int).ge(int(min_no_degree))
            & image_score_df["net_no_margin"].astype(int).ge(int(min_net_no_margin))
        )

        selected_image_ids = image_score_df.loc[
            image_score_df["selected_for_split"].astype(bool), "image_id"
        ].astype(str).tolist()
        anchor_candidates_df = image_score_df[
            ~image_score_df["image_id"].astype(str).isin([str(value) for value in selected_image_ids])
        ].copy()
        if anchor_candidates_df.empty:
            anchor_candidates_df = image_score_df.copy()
            if not anchor_candidates_df.empty:
                forced_anchor = _choose_anchor_image(anchor_candidates_df)
                selected_image_ids = [value for value in selected_image_ids if str(value) != forced_anchor]
                anchor_candidates_df = image_score_df[image_score_df["image_id"].astype(str).eq(str(forced_anchor))].copy()
        anchor_image_id = _choose_anchor_image(anchor_candidates_df) if not anchor_candidates_df.empty else ""

        image_score_df["selected_for_split"] = image_score_df["image_id"].astype(str).isin(
            [str(value) for value in selected_image_ids]
        )
        image_score_df["anchor_image_id"] = str(anchor_image_id)
        image_rows.extend(image_score_df.to_dict(orient="records"))

        start_operation_count = len(operations)
        attach_group_count = 0
        compile_status = "skip_no_selected_images"

        if selected_image_ids:
            operations = add_split_operation(
                operations,
                dataset=str(dataset),
                cluster_id=int(base_cluster_id),
                anchor_image_id=str(anchor_image_id) if anchor_image_id else None,
                member_image_ids=selected_image_ids,
                note=(
                    "compiled from pair judgments"
                    f" | cluster={base_cluster_id}"
                    f" | moved={'|'.join(selected_image_ids)}"
                    f" | no>={int(min_no_degree)}"
                    f" | no-yes>={int(min_net_no_margin)}"
                ),
            )
            compile_status = "compiled_partial_split"

            selected_set = {str(value) for value in selected_image_ids}
            yes_edges = [
                (str(row.image_id), str(row.neighbor_image_id))
                for row in candidate_df.itertuples(index=False)
                if str(row.label) == PAIR_LABEL_YES
                and str(row.image_id) in selected_set
                and str(row.neighbor_image_id) in selected_set
            ]
            no_pairs = {
                tuple(sorted((str(row.image_id), str(row.neighbor_image_id))))
                for row in candidate_df.itertuples(index=False)
                if str(row.label) == PAIR_LABEL_NO
                and str(row.image_id) in selected_set
                and str(row.neighbor_image_id) in selected_set
            }
            moved_components = _connected_components(sorted(selected_set), yes_edges)
            for component_index, component_image_ids in enumerate(moved_components, start=1):
                component_yes_edges = [
                    (left, right)
                    for left, right in yes_edges
                    if left in component_image_ids and right in component_image_ids
                ]
                component_no_edges = [
                    (left, right)
                    for left, right in no_pairs
                    if left in component_image_ids and right in component_image_ids
                ]
                attach_after_split = len(component_image_ids) > 1 and len(component_no_edges) == 0
                if attach_after_split:
                    operations = add_attach_operation(
                        operations,
                        dataset=str(dataset),
                        anchor_image_id=str(component_image_ids[0]),
                        member_image_ids=component_image_ids[1:],
                        source_cluster_ids=[],
                        note=(
                            "compiled regroup after split"
                            f" | cluster={base_cluster_id}"
                            f" | component={'|'.join(component_image_ids)}"
                        ),
                    )
                    attach_group_count += 1
                component_rows.append(
                    {
                        "dataset": str(dataset),
                        "base_cluster_id": int(base_cluster_id),
                        "component_index": int(component_index),
                        "component_image_ids": "|".join(component_image_ids),
                        "component_size": int(len(component_image_ids)),
                        "internal_yes_edges": int(len(component_yes_edges)),
                        "internal_no_edges": int(len(component_no_edges)),
                        "attach_after_split": bool(attach_after_split),
                    }
                )

        candidate_rows.append(
            {
                "dataset": str(dataset),
                "base_cluster_id": int(base_cluster_id),
                "candidate_key": str(candidate_key),
                "cluster_size": int(len(cluster_df)),
                "judged_pairs": int(len(candidate_df)),
                "yes_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_YES).sum()),
                "no_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_NO).sum()),
                "uncertain_pairs": int(candidate_df["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum()),
                "selected_split_images": int(len(selected_image_ids)),
                "attach_group_count": int(attach_group_count),
                "anchor_image_id": str(anchor_image_id),
                "compile_status": str(compile_status),
                "generated_operation_count": int(len(operations) - start_operation_count),
            }
        )

    candidate_summary_df = pd.DataFrame(candidate_rows).sort_values(
        ["compile_status", "dataset", "base_cluster_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True) if candidate_rows else pd.DataFrame()
    image_summary_df = pd.DataFrame(image_rows).sort_values(
        ["dataset", "base_cluster_id", "selected_for_split", "image_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True) if image_rows else pd.DataFrame()
    component_summary_df = pd.DataFrame(component_rows).sort_values(
        ["dataset", "base_cluster_id", "component_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True) if component_rows else pd.DataFrame()
    return operations, candidate_summary_df, image_summary_df, component_summary_df


def compile_split_judgments_file(
    *,
    base_predictions_path: str | Path,
    pair_judgments_path: str | Path,
    datasets: list[str] | tuple[str, ...] | None = None,
    candidate_keys: list[str] | tuple[str, ...] | None = None,
    min_no_degree: int = 2,
    min_net_no_margin: int = 1,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_csv(Path(base_predictions_path).resolve())
    session_name, judgments = load_pair_judgments(Path(pair_judgments_path).resolve())
    operations, candidate_summary_df, image_summary_df, component_summary_df = compile_split_judgments_to_overlay(
        pred_df,
        judgments,
        datasets=datasets,
        candidate_keys=candidate_keys,
        min_no_degree=min_no_degree,
        min_net_no_margin=min_net_no_margin,
    )
    return (
        session_name,
        judgments,
        operations,
        candidate_summary_df,
        image_summary_df,
        component_summary_df,
    )
