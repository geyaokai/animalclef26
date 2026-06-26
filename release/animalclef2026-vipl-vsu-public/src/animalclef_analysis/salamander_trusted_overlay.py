from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .manual_cluster_overlay import (
    ACTION_ATTACH_TO_ANCHOR,
    ACTION_SPLIT_TO_SINGLETONS,
    ManualOverlayOperation,
    ManualOverlaySpec,
    apply_manual_cluster_overlay,
    summarize_cluster_counts,
)


SALAMANDER_DATASET = "SalamanderID2025"


@dataclass(frozen=True)
class SalamanderTrustedOverlayResult:
    prediction_df: pd.DataFrame
    changed_df: pd.DataFrame
    operation_df: pd.DataFrame
    cannot_link_violation_df: pd.DataFrame
    summary_df: pd.DataFrame
    spec_payload: dict[str, Any]


def _normalize_predictions(pred_df: pd.DataFrame) -> pd.DataFrame:
    frame = pred_df.copy().reset_index(drop=True)
    frame["image_id"] = frame["image_id"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["pred_cluster_id"] = pd.to_numeric(frame["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    if "cluster_label" not in frame.columns:
        frame["cluster_label"] = frame.apply(
            lambda row: f"cluster_{row['dataset']}_{int(row['pred_cluster_id'])}",
            axis=1,
        )
    else:
        frame["cluster_label"] = frame["cluster_label"].astype(str)
    return frame


def _normalize_text_id_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].fillna("").astype(str)
    return result


def _choose_component_anchor(component_df: pd.DataFrame, pred_df: pd.DataFrame) -> str:
    pred_lookup = pred_df[["image_id", "pred_cluster_id"]].drop_duplicates(subset=["image_id"])
    merged = component_df[["image_id", "manual_yes_degree", "manual_no_degree"]].merge(
        pred_lookup,
        on="image_id",
        how="left",
    )
    merged["pred_cluster_id"] = pd.to_numeric(merged["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    cluster_size = merged["pred_cluster_id"].map(merged["pred_cluster_id"].value_counts()).astype(int)
    merged = merged.assign(_cluster_size=cluster_size)
    merged["manual_yes_degree"] = pd.to_numeric(merged["manual_yes_degree"], errors="coerce").fillna(0).astype(int)
    merged["manual_no_degree"] = pd.to_numeric(merged["manual_no_degree"], errors="coerce").fillna(0).astype(int)
    merged = merged.sort_values(
        ["_cluster_size", "manual_yes_degree", "manual_no_degree", "image_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    return str(merged.iloc[0]["image_id"])


def build_clean_trusted_attach_operations(
    *,
    pred_df: pd.DataFrame,
    clean_membership_df: pd.DataFrame,
    dataset: str = SALAMANDER_DATASET,
) -> list[dict[str, Any]]:
    if clean_membership_df.empty:
        return []
    clean_membership_df = _normalize_text_id_columns(clean_membership_df, ["component_id", "image_id", "dataset"])
    pred_df = _normalize_predictions(pred_df)
    salamander_pred = pred_df[pred_df["dataset"].eq(str(dataset))].copy()
    pred_image_ids = set(salamander_pred["image_id"].astype(str))

    operations: list[dict[str, Any]] = []
    for component_id, group in clean_membership_df.groupby("component_id", sort=True):
        group = group[group["image_id"].astype(str).isin(pred_image_ids)].copy()
        if len(group) < 2:
            continue
        pred_clusters = (
            group[["image_id"]]
            .merge(salamander_pred[["image_id", "pred_cluster_id"]], on="image_id", how="left")
            ["pred_cluster_id"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        if len(pred_clusters) <= 1:
            continue
        anchor_image_id = _choose_component_anchor(group, salamander_pred)
        operations.append(
            {
                "operation_id": f"salamander_clean_trusted_attach_{len(operations) + 1:03d}",
                "dataset": str(dataset),
                "action": ACTION_ATTACH_TO_ANCHOR,
                "anchor_image_id": str(anchor_image_id),
                "source_cluster_ids": [],
                "member_image_ids": sorted(group["image_id"].astype(str).tolist()),
                "exclude_image_ids": [],
                "note": f"clean_trusted_component={component_id};source_clusters={'|'.join(str(value) for value in sorted(pred_clusters))}",
            }
        )
    return operations


def find_cannot_link_violations(
    *,
    pred_df: pd.DataFrame,
    cannot_link_df: pd.DataFrame,
    dataset: str = SALAMANDER_DATASET,
) -> pd.DataFrame:
    pred_df = _normalize_predictions(pred_df)
    cannot_link_df = _normalize_text_id_columns(cannot_link_df, ["pair_key", "image_id", "neighbor_image_id"])
    salamander_pred = pred_df[pred_df["dataset"].eq(str(dataset))].copy()
    lookup = dict(zip(salamander_pred["image_id"].astype(str), salamander_pred["pred_cluster_id"].astype(int), strict=False))
    rows: list[dict[str, Any]] = []
    for row in cannot_link_df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        left_cluster = lookup.get(left)
        right_cluster = lookup.get(right)
        if left_cluster is None or right_cluster is None or int(left_cluster) != int(right_cluster):
            continue
        rows.append(
            {
                "pair_key": str(getattr(row, "pair_key", f"{left}|{right}")),
                "image_id": left,
                "neighbor_image_id": right,
                "pred_cluster_id": int(left_cluster),
                "candidate_types": str(getattr(row, "candidate_types", "")),
                "candidate_keys": str(getattr(row, "candidate_keys", "")),
                "manual_pair_count": int(getattr(row, "manual_pair_count", 1)),
                "max_xgb_same_identity_prob": float(getattr(row, "max_xgb_same_identity_prob", 0.0)),
                "max_ambiguity_score": float(getattr(row, "max_ambiguity_score", 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(["pred_cluster_id", "image_id", "neighbor_image_id"]).reset_index(drop=True) if rows else pd.DataFrame(
        columns=[
            "pair_key",
            "image_id",
            "neighbor_image_id",
            "pred_cluster_id",
            "candidate_types",
            "candidate_keys",
            "manual_pair_count",
            "max_xgb_same_identity_prob",
            "max_ambiguity_score",
        ]
    )


def build_cannot_link_singleton_operations(
    *,
    violation_df: pd.DataFrame,
    max_splits: int,
    dataset: str = SALAMANDER_DATASET,
) -> list[dict[str, Any]]:
    if violation_df.empty or int(max_splits) <= 0:
        return []
    operations: list[dict[str, Any]] = []
    used_split_images: set[str] = set()
    ranked = violation_df.sort_values(
        ["max_xgb_same_identity_prob", "max_ambiguity_score", "pair_key"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    for row in ranked.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        split_image = right if right not in used_split_images else left
        if split_image in used_split_images:
            continue
        anchor_image = left if split_image == right else right
        operations.append(
            {
                "operation_id": f"salamander_cannot_link_singleton_{len(operations) + 1:03d}",
                "dataset": str(dataset),
                "action": ACTION_SPLIT_TO_SINGLETONS,
                "anchor_image_id": anchor_image,
                "source_cluster_ids": [],
                "member_image_ids": [anchor_image, split_image],
                "exclude_image_ids": [],
                "note": f"manual_cannot_link_pair={row.pair_key};base_cluster={int(row.pred_cluster_id)}",
            }
        )
        used_split_images.add(split_image)
        if len(operations) >= int(max_splits):
            break
    return operations


def apply_salamander_trusted_overlay(
    *,
    pred_df: pd.DataFrame,
    clean_membership_df: pd.DataFrame,
    cannot_link_df: pd.DataFrame,
    enable_cannot_link_singletons: bool = False,
    max_cannot_link_singletons: int = 0,
    rule_name: str = "salamander_trusted_overlay_v1",
    dataset: str = SALAMANDER_DATASET,
) -> SalamanderTrustedOverlayResult:
    pred_df = _normalize_predictions(pred_df)
    before_violations = find_cannot_link_violations(pred_df=pred_df, cannot_link_df=cannot_link_df, dataset=dataset)
    operations = build_clean_trusted_attach_operations(
        pred_df=pred_df,
        clean_membership_df=clean_membership_df,
        dataset=dataset,
    )
    if bool(enable_cannot_link_singletons):
        operations.extend(
            build_cannot_link_singleton_operations(
                violation_df=before_violations,
                max_splits=int(max_cannot_link_singletons),
                dataset=dataset,
            )
        )

    spec_payload = {"rule_name": str(rule_name), "operations": operations}
    if operations:
        spec = ManualOverlaySpec(
            rule_name=str(rule_name),
            operations=tuple(
                ManualOverlayOperation(
                    operation_id=str(item["operation_id"]),
                    dataset=str(item["dataset"]),
                    action=str(item["action"]),
                    anchor_image_id=str(item.get("anchor_image_id")) if item.get("anchor_image_id") is not None else None,
                    source_cluster_ids=tuple(int(value) for value in item.get("source_cluster_ids", [])),
                    member_image_ids=tuple(str(value) for value in item.get("member_image_ids", [])),
                    exclude_image_ids=tuple(str(value) for value in item.get("exclude_image_ids", [])),
                    note=str(item.get("note", "")),
                )
                for item in operations
            ),
            raw_payload=spec_payload,
        )
        result_df, changed_df, operation_df = apply_manual_cluster_overlay(pred_df, spec=spec)
    else:
        result_df = pred_df.copy()
        changed_df = pd.DataFrame()
        operation_df = pd.DataFrame()

    after_violations = find_cannot_link_violations(pred_df=result_df, cannot_link_df=cannot_link_df, dataset=dataset)
    before_counts = summarize_cluster_counts(pred_df)
    after_counts = summarize_cluster_counts(result_df)
    summary_df = before_counts.merge(
        after_counts,
        on="dataset",
        how="outer",
        suffixes=("_before", "_after"),
    ).fillna(0)
    summary_df["operation_count"] = 0
    summary_df["changed_rows"] = 0
    summary_df["cannot_link_violations_before"] = 0
    summary_df["cannot_link_violations_after"] = 0
    dataset_mask = summary_df["dataset"].astype(str).eq(str(dataset))
    summary_df.loc[dataset_mask, "operation_count"] = int(len(operations))
    summary_df.loc[dataset_mask, "changed_rows"] = int(len(changed_df))
    summary_df.loc[dataset_mask, "cannot_link_violations_before"] = int(len(before_violations))
    summary_df.loc[dataset_mask, "cannot_link_violations_after"] = int(len(after_violations))

    return SalamanderTrustedOverlayResult(
        prediction_df=result_df,
        changed_df=changed_df,
        operation_df=operation_df,
        cannot_link_violation_df=after_violations,
        summary_df=summary_df,
        spec_payload=spec_payload,
    )


def write_overlay_spec(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
