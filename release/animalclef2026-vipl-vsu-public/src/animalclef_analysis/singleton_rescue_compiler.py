from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .manual_review_workbench import (
    PAIR_LABEL_NO,
    PAIR_LABEL_UNCERTAIN,
    PAIR_LABEL_YES,
    add_attach_operation,
)
from .singleton_rescue_review import SALAMANDER_DATASET


def _split_pipe_values(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [chunk.strip() for chunk in text.split("|") if chunk.strip()]


def _normalize_image_token(raw: object) -> str:
    text = "" if raw is None else str(raw).strip()
    if not text or text.lower() == "nan":
        return ""
    if text.endswith(".0"):
        head = text[:-2]
        if head.replace("-", "", 1).isdigit():
            return head
    return text


def _sorted_pair_key(left: str, right: str) -> str:
    left_key = str(left)
    right_key = str(right)
    return f"{left_key}|{right_key}" if left_key <= right_key else f"{right_key}|{left_key}"


@dataclass(frozen=True)
class SingletonRescueCompileResult:
    operations: list[dict[str, Any]]
    candidate_summary_df: pd.DataFrame


def compile_singleton_rescue_merge_judgments(
    merge_candidate_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    judgments: list[dict[str, Any]],
    *,
    dataset: str = SALAMANDER_DATASET,
) -> SingletonRescueCompileResult:
    candidate_frame = merge_candidate_df.copy()
    pair_frame = pair_df.copy()
    judgment_df = pd.DataFrame(judgments)

    if candidate_frame.empty:
        return SingletonRescueCompileResult(operations=[], candidate_summary_df=pd.DataFrame())

    candidate_frame["cluster_pair_key"] = candidate_frame["cluster_pair_key"].astype(str)
    candidate_frame["candidate_kind"] = candidate_frame.get("candidate_kind", "singleton_singleton").astype(str)
    if "singleton_image_id" in candidate_frame.columns:
        candidate_frame["singleton_image_id"] = candidate_frame["singleton_image_id"].apply(_normalize_image_token)
    else:
        candidate_frame["singleton_image_id"] = ""
    if "support_image_ids" not in candidate_frame.columns:
        candidate_frame["support_image_ids"] = ""
    candidate_frame["support_image_ids"] = candidate_frame["support_image_ids"].fillna("").astype(str)
    if "candidate_preview" not in candidate_frame.columns:
        candidate_frame["candidate_preview"] = candidate_frame["cluster_pair_key"]
    candidate_frame["candidate_preview"] = candidate_frame["candidate_preview"].fillna("").astype(str)
    if "origin_cluster_id" not in candidate_frame.columns:
        candidate_frame["origin_cluster_id"] = ""

    if not pair_frame.empty:
        pair_frame["cluster_pair_key"] = pair_frame["cluster_pair_key"].astype(str)
        pair_frame["image_id"] = pair_frame["image_id"].astype(str)
        pair_frame["neighbor_image_id"] = pair_frame["neighbor_image_id"].astype(str)
        for column in ["xgb_same_identity_prob", "local_score", "route_global_score"]:
            if column not in pair_frame.columns:
                pair_frame[column] = 0.0
            pair_frame[column] = pd.to_numeric(pair_frame[column], errors="coerce").fillna(0.0)
        pair_frame["pair_key"] = pair_frame.apply(
            lambda row: _sorted_pair_key(str(row["image_id"]), str(row["neighbor_image_id"])),
            axis=1,
        )

    if judgment_df.empty:
        judgment_df = pd.DataFrame(
            columns=["dataset", "candidate_type", "candidate_key", "image_id", "neighbor_image_id", "label"]
        )
    else:
        for column in ["dataset", "candidate_type", "candidate_key", "image_id", "neighbor_image_id", "label"]:
            if column not in judgment_df.columns:
                judgment_df[column] = ""
            judgment_df[column] = judgment_df[column].astype(str)
        judgment_df["pair_key"] = judgment_df.apply(
            lambda row: _sorted_pair_key(str(row["image_id"]), str(row["neighbor_image_id"])),
            axis=1,
        )
        judgment_df = judgment_df[
            judgment_df["dataset"].eq(str(dataset)) & judgment_df["candidate_type"].eq("merge")
        ].copy()

    operations: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for row in candidate_frame.sort_values(
        ["max_pair_probability", "support_pair_count", "cluster_pair_key"],
        ascending=[False, False, True],
    ).itertuples(index=False):
        candidate_key = str(row.cluster_pair_key)
        candidate_kind = str(row.candidate_kind)
        support_rows = pair_frame[pair_frame["cluster_pair_key"].astype(str).eq(candidate_key)].copy()
        support_rows = support_rows.sort_values(
            ["xgb_same_identity_prob", "local_score", "route_global_score", "pair_key"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        pair_total = int(len(support_rows))
        candidate_judgments = judgment_df[judgment_df["candidate_key"].eq(candidate_key)].copy()
        if not candidate_judgments.empty:
            candidate_judgments = candidate_judgments.drop_duplicates("pair_key", keep="last").reset_index(drop=True)
        judged_pairs = int(len(candidate_judgments))
        yes_count = int(candidate_judgments["label"].eq(PAIR_LABEL_YES).sum()) if not candidate_judgments.empty else 0
        no_count = int(candidate_judgments["label"].eq(PAIR_LABEL_NO).sum()) if not candidate_judgments.empty else 0
        uncertain_count = int(candidate_judgments["label"].eq(PAIR_LABEL_UNCERTAIN).sum()) if not candidate_judgments.empty else 0

        compile_status = "pending"
        if pair_total > 0 and judged_pairs == pair_total and yes_count == pair_total:
            compile_status = "accepted_all_yes"
        elif no_count > 0:
            compile_status = "rejected_has_no"
        elif uncertain_count > 0:
            compile_status = "pending_has_uncertain"
        elif judged_pairs >= pair_total and pair_total > 0:
            compile_status = "pending_mixed"

        generated_operation_id = ""
        if compile_status == "accepted_all_yes":
            if candidate_kind == "singleton_attach":
                singleton_image_id = str(getattr(row, "singleton_image_id", "")).strip()
                best_support = support_rows.iloc[0]
                anchor_image_id = (
                    str(best_support["neighbor_image_id"])
                    if str(best_support["image_id"]) == singleton_image_id
                    else str(best_support["image_id"])
                )
                operations = add_attach_operation(
                    operations,
                    dataset=str(dataset),
                    anchor_image_id=anchor_image_id,
                    member_image_ids=[singleton_image_id],
                    source_cluster_ids=[],
                    note=(
                        "compiled from singleton rescue manual review"
                        f" | candidate={candidate_key}"
                        f" | kind={candidate_kind}"
                        f" | preview={str(getattr(row, 'candidate_preview', ''))}"
                        f" | judged_pairs={judged_pairs}/{pair_total}"
                    ),
                )
                generated_operation_id = str(operations[-1]["operation_id"])
            else:
                support_image_ids = sorted(
                    set(_normalize_image_token(value) for value in _split_pipe_values(getattr(row, "support_image_ids", "")))
                    or set(
                        support_rows["image_id"].astype(str).tolist()
                        + support_rows["neighbor_image_id"].astype(str).tolist()
                    )
                )
                support_image_ids = [value for value in support_image_ids if value]
                if len(support_image_ids) < 2:
                    raise ValueError(f"singleton_singleton candidate {candidate_key!r} has fewer than 2 images")
                anchor_image_id = str(support_image_ids[0])
                member_image_ids = [str(value) for value in support_image_ids[1:]]
                operations = add_attach_operation(
                    operations,
                    dataset=str(dataset),
                    anchor_image_id=anchor_image_id,
                    member_image_ids=member_image_ids,
                    source_cluster_ids=[],
                    note=(
                        "compiled from singleton rescue manual review"
                        f" | candidate={candidate_key}"
                        f" | kind={candidate_kind}"
                        f" | preview={str(getattr(row, 'candidate_preview', ''))}"
                        f" | judged_pairs={judged_pairs}/{pair_total}"
                    ),
                )
                generated_operation_id = str(operations[-1]["operation_id"])

        candidate_rows.append(
            {
                "dataset": str(dataset),
                "candidate_key": candidate_key,
                "candidate_kind": candidate_kind,
                "candidate_preview": str(getattr(row, "candidate_preview", "")),
                "support_pair_count": pair_total,
                "judged_pairs": judged_pairs,
                "yes_count": yes_count,
                "no_count": no_count,
                "uncertain_count": uncertain_count,
                "compile_status": compile_status,
                "generated_operation_id": generated_operation_id,
                "mean_pair_probability": round(float(getattr(row, "mean_pair_probability", 0.0)), 6),
                "max_pair_probability": round(float(getattr(row, "max_pair_probability", 0.0)), 6),
                "origin_cluster_id": getattr(row, "origin_cluster_id", ""),
            }
        )

    candidate_summary_df = pd.DataFrame(candidate_rows)
    if not candidate_summary_df.empty:
        candidate_summary_df = candidate_summary_df.sort_values(
            ["compile_status", "max_pair_probability", "candidate_key"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return SingletonRescueCompileResult(
        operations=operations,
        candidate_summary_df=candidate_summary_df,
    )
