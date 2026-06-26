from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ACTION_SPLIT_TO_SINGLETONS = "split_to_singletons"
ACTION_ATTACH_TO_ANCHOR = "attach_to_anchor"
SUPPORTED_ACTIONS = {ACTION_SPLIT_TO_SINGLETONS, ACTION_ATTACH_TO_ANCHOR}


@dataclass(frozen=True)
class ManualOverlayOperation:
    operation_id: str
    dataset: str
    action: str
    anchor_image_id: str | None
    source_cluster_ids: tuple[int, ...]
    member_image_ids: tuple[str, ...]
    exclude_image_ids: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class ManualOverlaySpec:
    rule_name: str
    operations: tuple[ManualOverlayOperation, ...]
    raw_payload: dict[str, Any]


def _normalize_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        return (str(value),)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value)
    raise TypeError(f"Expected string or iterable of strings, got {type(value)!r}")


def _normalize_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (int(value),)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return tuple(int(item) for item in value)
    raise TypeError(f"Expected integer or iterable of integers, got {type(value)!r}")


def load_manual_overlay_spec(spec_path: Path) -> ManualOverlaySpec:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    rule_name = str(payload.get("rule_name") or "manual_cluster_overlay_v1")
    raw_operations = payload.get("operations")
    if not isinstance(raw_operations, list) or not raw_operations:
        raise ValueError("Manual overlay spec must contain a non-empty `operations` list")

    operations: list[ManualOverlayOperation] = []
    for idx, item in enumerate(raw_operations, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Operation #{idx} must be a JSON object")
        action = str(item.get("action") or "").strip()
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported manual overlay action: {action!r}")
        operation_id = str(item.get("operation_id") or f"op_{idx}")
        dataset = str(item.get("dataset") or "").strip()
        if not dataset:
            raise ValueError(f"Operation {operation_id!r} missing `dataset`")
        anchor_image_id = item.get("anchor_image_id")
        operations.append(
            ManualOverlayOperation(
                operation_id=operation_id,
                dataset=dataset,
                action=action,
                anchor_image_id=str(anchor_image_id) if anchor_image_id is not None else None,
                source_cluster_ids=_normalize_int_tuple(item.get("source_cluster_ids")),
                member_image_ids=_normalize_string_tuple(item.get("member_image_ids")),
                exclude_image_ids=_normalize_string_tuple(item.get("exclude_image_ids")),
                note=str(item.get("note") or ""),
            )
        )

    return ManualOverlaySpec(
        rule_name=rule_name,
        operations=tuple(operations),
        raw_payload=payload,
    )


def _append_trace(frame: pd.DataFrame, idx: int, column: str, token: str) -> None:
    current = ""
    if column in frame.columns and pd.notna(frame.at[idx, column]):
        current = str(frame.at[idx, column]).strip()
    frame.at[idx, column] = token if not current else f"{current}|{token}"


def _find_anchor_index(dataset_df: pd.DataFrame, anchor_image_id: str, *, operation_id: str) -> int:
    anchor_rows = dataset_df.index[dataset_df["image_id"].astype(str).eq(str(anchor_image_id))].tolist()
    if not anchor_rows:
        raise ValueError(f"Operation {operation_id!r} anchor image {anchor_image_id!r} not found in dataset slice")
    return int(anchor_rows[0])


def _resolve_cluster_member_indices(dataset_df: pd.DataFrame, cluster_ids: tuple[int, ...]) -> list[int]:
    if not cluster_ids:
        return []
    cluster_set = {int(value) for value in cluster_ids}
    return dataset_df.index[dataset_df["pred_cluster_id"].astype(int).isin(cluster_set)].tolist()


def _resolve_image_indices(dataset_df: pd.DataFrame, image_ids: tuple[str, ...], *, operation_id: str) -> list[int]:
    if not image_ids:
        return []
    image_to_index = {
        str(image_id): int(idx)
        for idx, image_id in zip(dataset_df.index.tolist(), dataset_df["image_id"].astype(str).tolist(), strict=True)
    }
    missing = [image_id for image_id in image_ids if str(image_id) not in image_to_index]
    if missing:
        raise ValueError(f"Operation {operation_id!r} missing image_ids in dataset slice: {missing[:5]}")
    return [image_to_index[str(image_id)] for image_id in image_ids]


def _allocate_new_cluster_id(next_cluster_id_by_dataset: dict[str, int], dataset: str) -> int:
    next_cluster_id = int(next_cluster_id_by_dataset[str(dataset)])
    next_cluster_id_by_dataset[str(dataset)] = next_cluster_id + 1
    return next_cluster_id


def _set_cluster_label(result_df: pd.DataFrame, idx: int, *, dataset: str, cluster_id: int) -> None:
    result_df.at[idx, "pred_cluster_id"] = int(cluster_id)
    result_df.at[idx, "cluster_label"] = f"cluster_{dataset}_{int(cluster_id)}"


def _record_change(
    changed_rows: list[dict[str, object]],
    *,
    row: pd.Series,
    dataset: str,
    operation: ManualOverlayOperation,
    overlay_action: str,
    base_cluster_id: int,
    base_cluster_label: str,
    final_cluster_id: int,
    final_cluster_label: str,
) -> None:
    changed_rows.append(
        {
            "dataset": dataset,
            "operation_id": operation.operation_id,
            "operation_action": operation.action,
            "overlay_action": overlay_action,
            "note": operation.note,
            "anchor_image_id": operation.anchor_image_id or "",
            "source_cluster_ids": "|".join(str(int(value)) for value in operation.source_cluster_ids),
            "member_image_ids": "|".join(str(value) for value in operation.member_image_ids),
            "exclude_image_ids": "|".join(str(value) for value in operation.exclude_image_ids),
            "image_id": str(row["image_id"]),
            "path": str(row["path"]) if "path" in row else "",
            "overlay_base_pred_cluster_id": int(base_cluster_id),
            "overlay_base_cluster_label": str(base_cluster_label),
            "final_pred_cluster_id": int(final_cluster_id),
            "final_cluster_label": str(final_cluster_label),
        }
    )


def _mark_row_trace(
    result_df: pd.DataFrame,
    idx: int,
    *,
    rule_name: str,
    operation_id: str,
    note: str,
    action_suffix: str,
) -> None:
    result_df.at[idx, "manual_overlay_enabled"] = True
    _append_trace(result_df, idx, "manual_overlay_rule", f"{rule_name}|{operation_id}|{action_suffix}")
    _append_trace(result_df, idx, "manual_overlay_operation_id", operation_id)
    if note:
        _append_trace(result_df, idx, "manual_overlay_note", note)


def apply_manual_cluster_overlay(
    pred_df: pd.DataFrame,
    *,
    spec: ManualOverlaySpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_columns = {"image_id", "dataset", "pred_cluster_id", "cluster_label"}
    missing_columns = required_columns - set(pred_df.columns)
    if missing_columns:
        raise ValueError(f"Prediction table missing required columns: {sorted(missing_columns)}")

    result_df = pred_df.copy().reset_index(drop=True)
    result_df["image_id"] = result_df["image_id"].astype(str)
    result_df["dataset"] = result_df["dataset"].astype(str)
    result_df["pred_cluster_id"] = result_df["pred_cluster_id"].astype(int)
    result_df["cluster_label"] = result_df["cluster_label"].astype(str)
    if "manual_overlay_enabled" not in result_df.columns:
        result_df["manual_overlay_enabled"] = False
    else:
        result_df["manual_overlay_enabled"] = result_df["manual_overlay_enabled"].fillna(False).astype(bool)
    for column in ["manual_overlay_rule", "manual_overlay_operation_id", "manual_overlay_note"]:
        if column not in result_df.columns:
            result_df[column] = ""
        else:
            result_df[column] = result_df[column].fillna("").astype(str)

    next_cluster_id_by_dataset = {
        str(dataset): int(group["pred_cluster_id"].astype(int).max()) + 1
        for dataset, group in result_df.groupby("dataset", sort=False)
    }

    changed_rows: list[dict[str, object]] = []
    operation_rows: list[dict[str, object]] = []

    for operation in spec.operations:
        dataset = str(operation.dataset)
        dataset_mask = result_df["dataset"].astype(str).eq(dataset)
        dataset_df = result_df.loc[dataset_mask].copy()
        if dataset_df.empty:
            raise ValueError(f"Operation {operation.operation_id!r} dataset {dataset!r} not found in prediction table")

        cluster_member_indices = _resolve_cluster_member_indices(dataset_df, operation.source_cluster_ids)
        explicit_member_indices = _resolve_image_indices(
            dataset_df,
            operation.member_image_ids,
            operation_id=operation.operation_id,
        )
        excluded_indices = set(
            _resolve_image_indices(dataset_df, operation.exclude_image_ids, operation_id=operation.operation_id)
        )

        anchor_idx: int | None = None
        if operation.anchor_image_id is not None:
            anchor_idx = _find_anchor_index(dataset_df, operation.anchor_image_id, operation_id=operation.operation_id)

        if operation.action == ACTION_SPLIT_TO_SINGLETONS:
            if explicit_member_indices:
                selected_indices = explicit_member_indices
            else:
                selected_indices = cluster_member_indices
            if not selected_indices:
                raise ValueError(f"Operation {operation.operation_id!r} resolves to zero rows for split")
            selected_indices = [idx for idx in selected_indices if idx not in excluded_indices]
            if not selected_indices:
                raise ValueError(f"Operation {operation.operation_id!r} has no rows after exclusions")
            if anchor_idx is None and cluster_member_indices:
                selected_rows = dataset_df.loc[selected_indices].copy()
                selected_rows = selected_rows.assign(_image_sort=selected_rows["image_id"].astype(str))
                anchor_idx = int(selected_rows.sort_values("_image_sort").index[0])
            singleton_indices = [idx for idx in selected_indices if idx != anchor_idx]
            if anchor_idx is not None and cluster_member_indices and anchor_idx not in cluster_member_indices:
                raise ValueError(
                    f"Operation {operation.operation_id!r} anchor image must belong to source cluster(s) for cluster split"
                )
            if anchor_idx is not None and cluster_member_indices:
                anchor_row = result_df.loc[anchor_idx]
                _mark_row_trace(
                    result_df,
                    anchor_idx,
                    rule_name=spec.rule_name,
                    operation_id=operation.operation_id,
                    note=operation.note,
                    action_suffix="keep_anchor",
                )
                _record_change(
                    changed_rows,
                    row=anchor_row,
                    dataset=dataset,
                    operation=operation,
                    overlay_action="keep_anchor",
                    base_cluster_id=int(anchor_row["pred_cluster_id"]),
                    base_cluster_label=str(anchor_row["cluster_label"]),
                    final_cluster_id=int(anchor_row["pred_cluster_id"]),
                    final_cluster_label=str(anchor_row["cluster_label"]),
                )
            changed_count = 0
            for idx in singleton_indices:
                base_cluster_id = int(result_df.at[idx, "pred_cluster_id"])
                base_cluster_label = str(result_df.at[idx, "cluster_label"])
                new_cluster_id = _allocate_new_cluster_id(next_cluster_id_by_dataset, dataset)
                _set_cluster_label(result_df, idx, dataset=dataset, cluster_id=new_cluster_id)
                _mark_row_trace(
                    result_df,
                    idx,
                    rule_name=spec.rule_name,
                    operation_id=operation.operation_id,
                    note=operation.note,
                    action_suffix="singleton",
                )
                row = result_df.loc[idx]
                _record_change(
                    changed_rows,
                    row=row,
                    dataset=dataset,
                    operation=operation,
                    overlay_action="singleton",
                    base_cluster_id=base_cluster_id,
                    base_cluster_label=base_cluster_label,
                    final_cluster_id=int(row["pred_cluster_id"]),
                    final_cluster_label=str(row["cluster_label"]),
                )
                changed_count += 1
            operation_rows.append(
                {
                    "operation_id": operation.operation_id,
                    "dataset": dataset,
                    "action": operation.action,
                    "anchor_image_id": operation.anchor_image_id or "",
                    "source_cluster_ids": "|".join(str(int(value)) for value in operation.source_cluster_ids),
                    "member_image_ids": "|".join(str(value) for value in operation.member_image_ids),
                    "exclude_image_ids": "|".join(str(value) for value in operation.exclude_image_ids),
                    "selected_count": int(len(selected_indices)),
                    "changed_count": int(changed_count),
                    "note": operation.note,
                }
            )
            continue

        if operation.action == ACTION_ATTACH_TO_ANCHOR:
            if anchor_idx is None:
                raise ValueError(f"Operation {operation.operation_id!r} requires `anchor_image_id` for attach_to_anchor")
            selected_index_set = set(cluster_member_indices) | set(explicit_member_indices)
            selected_index_set.difference_update(excluded_indices)
            selected_index_set.discard(anchor_idx)
            if not selected_index_set:
                raise ValueError(f"Operation {operation.operation_id!r} resolves to zero rows for attach")
            target_cluster_id = int(result_df.at[anchor_idx, "pred_cluster_id"])
            changed_count = 0
            for idx in sorted(selected_index_set):
                base_cluster_id = int(result_df.at[idx, "pred_cluster_id"])
                base_cluster_label = str(result_df.at[idx, "cluster_label"])
                if base_cluster_id == target_cluster_id:
                    continue
                _set_cluster_label(result_df, idx, dataset=dataset, cluster_id=target_cluster_id)
                _mark_row_trace(
                    result_df,
                    idx,
                    rule_name=spec.rule_name,
                    operation_id=operation.operation_id,
                    note=operation.note,
                    action_suffix="attach_to_anchor",
                )
                row = result_df.loc[idx]
                _record_change(
                    changed_rows,
                    row=row,
                    dataset=dataset,
                    operation=operation,
                    overlay_action="attach_to_anchor",
                    base_cluster_id=base_cluster_id,
                    base_cluster_label=base_cluster_label,
                    final_cluster_id=int(row["pred_cluster_id"]),
                    final_cluster_label=str(row["cluster_label"]),
                )
                changed_count += 1
            operation_rows.append(
                {
                    "operation_id": operation.operation_id,
                    "dataset": dataset,
                    "action": operation.action,
                    "anchor_image_id": operation.anchor_image_id or "",
                    "source_cluster_ids": "|".join(str(int(value)) for value in operation.source_cluster_ids),
                    "member_image_ids": "|".join(str(value) for value in operation.member_image_ids),
                    "exclude_image_ids": "|".join(str(value) for value in operation.exclude_image_ids),
                    "selected_count": int(len(selected_index_set)),
                    "changed_count": int(changed_count),
                    "note": operation.note,
                }
            )
            continue

        raise AssertionError(f"Unhandled action: {operation.action!r}")

    changed_df = pd.DataFrame(
        changed_rows,
        columns=[
            "dataset",
            "operation_id",
            "operation_action",
            "overlay_action",
            "note",
            "anchor_image_id",
            "source_cluster_ids",
            "member_image_ids",
            "exclude_image_ids",
            "image_id",
            "path",
            "overlay_base_pred_cluster_id",
            "overlay_base_cluster_label",
            "final_pred_cluster_id",
            "final_cluster_label",
        ],
    )
    operation_df = pd.DataFrame(
        operation_rows,
        columns=[
            "operation_id",
            "dataset",
            "action",
            "anchor_image_id",
            "source_cluster_ids",
            "member_image_ids",
            "exclude_image_ids",
            "selected_count",
            "changed_count",
            "note",
        ],
    )
    return result_df, changed_df, operation_df


def summarize_cluster_counts(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, group in pred_df.groupby("dataset", sort=True):
        counts = group["pred_cluster_id"].astype(int).value_counts()
        rows.append(
            {
                "dataset": str(dataset),
                "clusters": int(counts.size),
                "singletons": int((counts == 1).sum()),
            }
        )
    return pd.DataFrame(rows)
