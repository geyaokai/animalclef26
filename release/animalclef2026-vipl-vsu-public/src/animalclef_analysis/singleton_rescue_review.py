from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"


def empty_merge_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "cluster_pair_key",
            "left_cluster_id",
            "right_cluster_id",
            "left_cluster_size",
            "right_cluster_size",
            "merged_total_size",
            "support_pair_count",
            "max_merge_votes",
            "mean_pair_probability",
            "max_pair_probability",
            "mean_ambiguity_score",
            "max_ambiguity_score",
            "mean_border_score",
            "max_conflict_ratio",
            "conflict_methods",
            "component_ids",
            "candidate_kind",
            "candidate_preview",
            "singleton_image_id",
            "support_image_ids",
            "origin_cluster_id",
        ]
    )


def empty_pair_disagreement() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_id",
            "neighbor_image_id",
            "xgb_same_identity_prob",
            "merge_votes",
            "split_votes",
            "ambiguity_score",
            "vote_direction",
            "base_cluster_left",
            "base_cluster_right",
            "conflict_methods",
            "component_id",
            "local_score",
            "route_global_score",
            "candidate_kind",
            "cluster_pair_key",
        ]
    )


def empty_stats() -> pd.DataFrame:
    return pd.DataFrame(columns=["metric", "value"])


def empty_rejected_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_kind",
            "image_id",
            "neighbor_image_id",
            "target_cluster_id",
            "reason",
            "xgb_same_identity_prob",
            "local_score",
            "route_global_score",
            "support_pair_count",
            "mean_pair_probability",
            "max_pair_probability",
            "support_image_ids",
            "cluster_pair_key",
        ]
    )


@dataclass(frozen=True)
class SingletonRescueReviewResult:
    merge_candidate_df: pd.DataFrame
    pair_df: pd.DataFrame
    stats_df: pd.DataFrame
    rejected_candidate_df: pd.DataFrame


def _sorted_pair_key(left: str, right: str) -> tuple[str, str]:
    left_key = str(left)
    right_key = str(right)
    return (left_key, right_key) if left_key <= right_key else (right_key, left_key)


def _extract_origin_cluster_id(note: object) -> int | None:
    text = "" if note is None else str(note)
    match = re.search(r"cluster=(\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def load_manual_no_pairs(
    judgments_path: str | Path,
    *,
    dataset: str = SALAMANDER_DATASET,
) -> set[tuple[str, str]]:
    path = Path(judgments_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    judgments = payload.get("pair_judgments", [])
    if not isinstance(judgments, list):
        raise ValueError("pair_judgments must be a list")
    pair_keys: set[tuple[str, str]] = set()
    for item in judgments:
        if not isinstance(item, dict):
            continue
        if str(item.get("dataset", "")) != str(dataset):
            continue
        if str(item.get("label", "")).strip().lower() != "no":
            continue
        image_id = str(item.get("image_id", "")).strip()
        neighbor_image_id = str(item.get("neighbor_image_id", "")).strip()
        if not image_id or not neighbor_image_id or image_id == neighbor_image_id:
            continue
        pair_keys.add(_sorted_pair_key(image_id, neighbor_image_id))
    return pair_keys


def _normalize_prediction_frame(
    pred_df: pd.DataFrame,
    *,
    dataset: str,
) -> pd.DataFrame:
    frame = pred_df.copy()
    required_columns = {"image_id", "dataset", "pred_cluster_id", "cluster_label"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table missing required columns: {sorted(missing)}")
    frame["image_id"] = frame["image_id"].astype(str)
    frame["dataset"] = frame["dataset"].astype(str)
    frame["pred_cluster_id"] = pd.to_numeric(frame["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    frame["cluster_label"] = frame["cluster_label"].astype(str)
    if "path" in frame.columns:
        frame["path"] = frame["path"].astype(str)
    if "manual_overlay_enabled" not in frame.columns:
        frame["manual_overlay_enabled"] = False
    else:
        frame["manual_overlay_enabled"] = frame["manual_overlay_enabled"].fillna(False).astype(bool)
    if "manual_overlay_note" not in frame.columns:
        frame["manual_overlay_note"] = ""
    else:
        frame["manual_overlay_note"] = frame["manual_overlay_note"].fillna("").astype(str)
    frame = frame[frame["dataset"].eq(str(dataset))].copy().reset_index(drop=True)
    frame["origin_cluster_id"] = frame["manual_overlay_note"].apply(_extract_origin_cluster_id)
    return frame


def _normalize_pair_feature_frame(
    pair_feature_df: pd.DataFrame,
    *,
    cluster_by_image: dict[str, int],
    allowed_image_ids: set[str],
) -> pd.DataFrame:
    frame = pair_feature_df.copy()
    required_columns = {"image_id", "neighbor_image_id", "xgb_same_identity_prob"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Pair feature table missing required columns: {sorted(missing)}")
    frame["image_id"] = frame["image_id"].astype(str)
    frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)
    frame = frame[
        frame["image_id"].isin(sorted(allowed_image_ids))
        & frame["neighbor_image_id"].isin(sorted(allowed_image_ids))
        & frame["image_id"].ne(frame["neighbor_image_id"])
    ].copy()
    if frame.empty:
        return frame
    frame["xgb_same_identity_prob"] = pd.to_numeric(frame["xgb_same_identity_prob"], errors="coerce").fillna(0.0)
    for column in ["local_score", "route_global_score"]:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["left_image_id"] = frame.apply(
        lambda row: _sorted_pair_key(str(row["image_id"]), str(row["neighbor_image_id"]))[0],
        axis=1,
    )
    frame["right_image_id"] = frame.apply(
        lambda row: _sorted_pair_key(str(row["image_id"]), str(row["neighbor_image_id"]))[1],
        axis=1,
    )
    frame["pair_key"] = frame["left_image_id"] + "|" + frame["right_image_id"]
    frame = frame.sort_values(
        ["pair_key", "xgb_same_identity_prob", "local_score", "route_global_score"],
        ascending=[True, False, False, False],
    ).drop_duplicates("pair_key", keep="first")
    frame["base_cluster_left"] = frame["left_image_id"].map(cluster_by_image).astype(int)
    frame["base_cluster_right"] = frame["right_image_id"].map(cluster_by_image).astype(int)
    frame = frame[frame["base_cluster_left"].ne(frame["base_cluster_right"])].copy().reset_index(drop=True)
    return frame


def _best_partner_rows(
    pair_df: pd.DataFrame,
    *,
    anchor_image_ids: set[str],
) -> dict[str, pd.Series]:
    best_rows: dict[str, pd.Series] = {}
    for image_id in sorted(anchor_image_ids):
        subset = pair_df[
            pair_df["left_image_id"].eq(str(image_id)) | pair_df["right_image_id"].eq(str(image_id))
        ].copy()
        if subset.empty:
            continue
        subset["other_image_id"] = subset.apply(
            lambda row: str(row["right_image_id"]) if str(row["left_image_id"]) == str(image_id) else str(row["left_image_id"]),
            axis=1,
        )
        subset = subset.sort_values(
            ["xgb_same_identity_prob", "local_score", "route_global_score", "other_image_id"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        best_rows[str(image_id)] = subset.iloc[0]
    return best_rows


def build_singleton_rescue_review(
    pred_df: pd.DataFrame,
    pair_feature_df: pd.DataFrame,
    *,
    manual_no_pairs: set[tuple[str, str]],
    dataset: str = SALAMANDER_DATASET,
    singleton_singleton_min_prob: float = 0.95,
    attach_member_min_prob: float = 0.90,
    attach_min_support_count: int = 2,
    attach_min_mean_prob: float = 0.90,
    attach_min_max_prob: float = 0.95,
) -> SingletonRescueReviewResult:
    pred_frame = _normalize_prediction_frame(pred_df, dataset=dataset)
    if pred_frame.empty:
        return SingletonRescueReviewResult(
            merge_candidate_df=empty_merge_candidates(),
            pair_df=empty_pair_disagreement(),
            stats_df=empty_stats(),
            rejected_candidate_df=empty_rejected_candidates(),
        )

    cluster_sizes = pred_frame["pred_cluster_id"].value_counts()
    singleton_cluster_ids = set(cluster_sizes[cluster_sizes.eq(1)].index.astype(int).tolist())
    manual_singleton_df = pred_frame[
        pred_frame["pred_cluster_id"].isin(singleton_cluster_ids) & pred_frame["manual_overlay_enabled"]
    ].copy().reset_index(drop=True)
    manual_singleton_ids = set(manual_singleton_df["image_id"].astype(str).tolist())
    allowed_image_ids = set(pred_frame["image_id"].astype(str).tolist())
    cluster_by_image = dict(zip(pred_frame["image_id"].astype(str), pred_frame["pred_cluster_id"].astype(int), strict=True))
    origin_cluster_by_image = dict(zip(pred_frame["image_id"].astype(str), pred_frame["origin_cluster_id"].tolist(), strict=True))
    cluster_members_by_cluster = {
        int(cluster_id): sorted(group["image_id"].astype(str).tolist())
        for cluster_id, group in pred_frame.groupby("pred_cluster_id", sort=True)
    }

    pair_frame = _normalize_pair_feature_frame(
        pair_feature_df,
        cluster_by_image=cluster_by_image,
        allowed_image_ids=allowed_image_ids,
    )
    if pair_frame.empty or not manual_singleton_ids:
        stats_df = pd.DataFrame(
            [
                {"metric": "dataset", "value": str(dataset)},
                {"metric": "dataset_image_count", "value": int(len(pred_frame))},
                {"metric": "cluster_count", "value": int(cluster_sizes.size)},
                {"metric": "singleton_count", "value": int(len(singleton_cluster_ids))},
                {"metric": "manual_singleton_count", "value": int(len(manual_singleton_ids))},
                {"metric": "accepted_candidate_count", "value": 0},
            ]
        )
        return SingletonRescueReviewResult(
            merge_candidate_df=empty_merge_candidates(),
            pair_df=empty_pair_disagreement(),
            stats_df=stats_df,
            rejected_candidate_df=empty_rejected_candidates(),
        )

    candidate_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    candidate_index = 0
    accepted_singleton_ids: set[str] = set()

    singleton_pair_frame = pair_frame[
        pair_frame["left_image_id"].isin(sorted(manual_singleton_ids))
        & pair_frame["right_image_id"].isin(sorted(manual_singleton_ids))
    ].copy()
    best_partner = _best_partner_rows(singleton_pair_frame, anchor_image_ids=manual_singleton_ids)
    seen_pair_keys: set[tuple[str, str]] = set()

    for image_id, row in sorted(
        best_partner.items(),
        key=lambda item: (
            -float(item[1]["xgb_same_identity_prob"]),
            -float(item[1]["local_score"]),
            -float(item[1]["route_global_score"]),
            item[0],
        ),
    ):
        other_image_id = str(row["other_image_id"])
        image_pair_key = _sorted_pair_key(str(image_id), other_image_id)
        if image_pair_key in seen_pair_keys:
            continue
        reverse_row = best_partner.get(other_image_id)
        if reverse_row is None or str(reverse_row["other_image_id"]) != str(image_id):
            continue
        xgb_score = float(row["xgb_same_identity_prob"])
        local_score = float(row["local_score"])
        route_score = float(row["route_global_score"])
        if xgb_score < float(singleton_singleton_min_prob):
            continue
        if image_pair_key in manual_no_pairs:
            rejected_rows.append(
                {
                    "candidate_kind": "singleton_singleton",
                    "image_id": str(image_id),
                    "neighbor_image_id": other_image_id,
                    "target_cluster_id": "",
                    "reason": "manual_no_conflict",
                    "xgb_same_identity_prob": round(xgb_score, 6),
                    "local_score": round(local_score, 6),
                    "route_global_score": round(route_score, 6),
                    "support_pair_count": 1,
                    "mean_pair_probability": round(xgb_score, 6),
                    "max_pair_probability": round(xgb_score, 6),
                    "support_image_ids": other_image_id,
                    "cluster_pair_key": "",
                }
            )
            seen_pair_keys.add(image_pair_key)
            continue

        candidate_index += 1
        left_cluster_id, right_cluster_id = sorted(
            [int(cluster_by_image[str(image_id)]), int(cluster_by_image[other_image_id])]
        )
        cluster_pair_key = f"{left_cluster_id}|{right_cluster_id}"
        origin_cluster_values = {
            origin_cluster_by_image.get(str(image_id)),
            origin_cluster_by_image.get(str(other_image_id)),
        }
        origin_cluster_values.discard(None)
        origin_cluster_id = int(next(iter(origin_cluster_values))) if len(origin_cluster_values) == 1 else ""
        preview = f"{image_pair_key[0]}|{image_pair_key[1]}"
        candidate_rows.append(
            {
                "cluster_pair_key": cluster_pair_key,
                "left_cluster_id": int(left_cluster_id),
                "right_cluster_id": int(right_cluster_id),
                "left_cluster_size": 1,
                "right_cluster_size": 1,
                "merged_total_size": 2,
                "support_pair_count": 1,
                "max_merge_votes": 1,
                "mean_pair_probability": round(xgb_score, 6),
                "max_pair_probability": round(xgb_score, 6),
                "mean_ambiguity_score": round(xgb_score, 6),
                "max_ambiguity_score": round(xgb_score, 6),
                "mean_border_score": round(local_score, 6),
                "max_conflict_ratio": 0.0,
                "conflict_methods": "singleton_rescue_v1_mutual_top1",
                "component_ids": preview,
                "candidate_kind": "singleton_singleton",
                "candidate_preview": preview,
                "singleton_image_id": "",
                "support_image_ids": preview,
                "origin_cluster_id": origin_cluster_id,
            }
        )
        pair_rows.append(
            {
                "image_id": str(image_pair_key[0]),
                "neighbor_image_id": str(image_pair_key[1]),
                "xgb_same_identity_prob": round(xgb_score, 6),
                "merge_votes": 1,
                "split_votes": 0,
                "ambiguity_score": round(xgb_score, 6),
                "vote_direction": "merge",
                "base_cluster_left": int(left_cluster_id),
                "base_cluster_right": int(right_cluster_id),
                "conflict_methods": "singleton_rescue_v1_mutual_top1",
                "component_id": int(candidate_index),
                "local_score": round(local_score, 6),
                "route_global_score": round(route_score, 6),
                "candidate_kind": "singleton_singleton",
                "cluster_pair_key": cluster_pair_key,
            }
        )
        seen_pair_keys.add(image_pair_key)
        accepted_singleton_ids.update([str(image_pair_key[0]), str(image_pair_key[1])])

    for singleton_image_id in sorted(manual_singleton_ids):
        if str(singleton_image_id) in accepted_singleton_ids:
            continue
        singleton_cluster_id = int(cluster_by_image[str(singleton_image_id)])
        support_pool = pair_frame[
            (
                pair_frame["left_image_id"].eq(str(singleton_image_id))
                | pair_frame["right_image_id"].eq(str(singleton_image_id))
            )
        ].copy()
        if support_pool.empty:
            continue
        support_pool["other_image_id"] = support_pool.apply(
            lambda row: str(row["right_image_id"])
            if str(row["left_image_id"]) == str(singleton_image_id)
            else str(row["left_image_id"]),
            axis=1,
        )
        support_pool["other_cluster_id"] = support_pool["other_image_id"].map(cluster_by_image).astype(int)
        support_pool = support_pool[
            ~support_pool["other_cluster_id"].isin(sorted(singleton_cluster_ids))
            & support_pool["xgb_same_identity_prob"].ge(float(attach_member_min_prob))
        ].copy()
        if support_pool.empty:
            continue

        attach_candidates: list[dict[str, Any]] = []
        for target_cluster_id, group in support_pool.groupby("other_cluster_id", sort=True):
            target_cluster_members = cluster_members_by_cluster.get(int(target_cluster_id), [])
            manual_no_conflict = not any(
                _sorted_pair_key(str(singleton_image_id), str(member_image_id)) in manual_no_pairs
                for member_image_id in target_cluster_members
            )
            support_image_ids = sorted(group["other_image_id"].astype(str).unique().tolist())
            mean_xgb = float(group["xgb_same_identity_prob"].mean())
            max_xgb = float(group["xgb_same_identity_prob"].max())
            mean_local = float(group["local_score"].mean())
            attach_candidates.append(
                {
                    "target_cluster_id": int(target_cluster_id),
                    "support_pair_count": int(len(support_image_ids)),
                    "mean_pair_probability": mean_xgb,
                    "max_pair_probability": max_xgb,
                    "mean_border_score": mean_local,
                    "manual_no_conflict": bool(manual_no_conflict),
                    "support_image_ids": support_image_ids,
                }
            )

        attach_candidate_df = pd.DataFrame(attach_candidates)
        if attach_candidate_df.empty:
            continue
        qualifying_df = attach_candidate_df[
            attach_candidate_df["support_pair_count"].astype(int).ge(int(attach_min_support_count))
            & attach_candidate_df["mean_pair_probability"].astype(float).ge(float(attach_min_mean_prob))
            & attach_candidate_df["max_pair_probability"].astype(float).ge(float(attach_min_max_prob))
            & attach_candidate_df["manual_no_conflict"].astype(bool)
        ].copy()
        if qualifying_df.empty:
            best_reject = attach_candidate_df.sort_values(
                ["support_pair_count", "mean_pair_probability", "max_pair_probability", "target_cluster_id"],
                ascending=[False, False, False, True],
            ).iloc[0]
            rejected_rows.append(
                {
                    "candidate_kind": "singleton_attach",
                    "image_id": str(singleton_image_id),
                    "neighbor_image_id": "",
                    "target_cluster_id": int(best_reject["target_cluster_id"]),
                    "reason": "below_attach_gate" if bool(best_reject["manual_no_conflict"]) else "manual_no_conflict",
                    "xgb_same_identity_prob": "",
                    "local_score": "",
                    "route_global_score": "",
                    "support_pair_count": int(best_reject["support_pair_count"]),
                    "mean_pair_probability": round(float(best_reject["mean_pair_probability"]), 6),
                    "max_pair_probability": round(float(best_reject["max_pair_probability"]), 6),
                    "support_image_ids": "|".join(best_reject["support_image_ids"]),
                    "cluster_pair_key": "",
                }
            )
            continue

        best_attach = qualifying_df.sort_values(
            ["support_pair_count", "mean_pair_probability", "max_pair_probability", "mean_border_score", "target_cluster_id"],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        target_cluster_id = int(best_attach["target_cluster_id"])
        support_member_ids = list(best_attach["support_image_ids"])
        target_pairs = support_pool[
            support_pool["other_cluster_id"].astype(int).eq(int(target_cluster_id))
            & support_pool["other_image_id"].astype(str).isin(support_member_ids)
        ].copy().sort_values(
            ["xgb_same_identity_prob", "local_score", "route_global_score", "other_image_id"],
            ascending=[False, False, False, True],
        )
        candidate_index += 1
        left_cluster_id, right_cluster_id = sorted([int(singleton_cluster_id), int(target_cluster_id)])
        cluster_pair_key = f"{left_cluster_id}|{right_cluster_id}"
        origin_cluster_id = origin_cluster_by_image.get(str(singleton_image_id))
        candidate_rows.append(
            {
                "cluster_pair_key": cluster_pair_key,
                "left_cluster_id": int(left_cluster_id),
                "right_cluster_id": int(right_cluster_id),
                "left_cluster_size": int(cluster_sizes[int(left_cluster_id)]),
                "right_cluster_size": int(cluster_sizes[int(right_cluster_id)]),
                "merged_total_size": int(cluster_sizes[int(left_cluster_id)] + cluster_sizes[int(right_cluster_id)]),
                "support_pair_count": int(best_attach["support_pair_count"]),
                "max_merge_votes": int(best_attach["support_pair_count"]),
                "mean_pair_probability": round(float(best_attach["mean_pair_probability"]), 6),
                "max_pair_probability": round(float(best_attach["max_pair_probability"]), 6),
                "mean_ambiguity_score": round(float(best_attach["mean_pair_probability"]), 6),
                "max_ambiguity_score": round(float(best_attach["max_pair_probability"]), 6),
                "mean_border_score": round(float(best_attach["mean_border_score"]), 6),
                "max_conflict_ratio": 0.0,
                "conflict_methods": "singleton_rescue_v1_attach_support",
                "component_ids": "|".join(sorted(support_member_ids)),
                "candidate_kind": "singleton_attach",
                "candidate_preview": f"{singleton_image_id} -> {'|'.join(sorted(support_member_ids))}",
                "singleton_image_id": str(singleton_image_id),
                "support_image_ids": "|".join(sorted(support_member_ids)),
                "origin_cluster_id": "" if origin_cluster_id is None else int(origin_cluster_id),
            }
        )
        merge_votes = int(best_attach["support_pair_count"])
        for row in target_pairs.itertuples(index=False):
            pair_rows.append(
                {
                    "image_id": str(row.left_image_id),
                    "neighbor_image_id": str(row.right_image_id),
                    "xgb_same_identity_prob": round(float(row.xgb_same_identity_prob), 6),
                    "merge_votes": merge_votes,
                    "split_votes": 0,
                    "ambiguity_score": round(float(row.xgb_same_identity_prob), 6),
                    "vote_direction": "merge",
                    "base_cluster_left": int(row.base_cluster_left),
                    "base_cluster_right": int(row.base_cluster_right),
                    "conflict_methods": "singleton_rescue_v1_attach_support",
                    "component_id": int(candidate_index),
                    "local_score": round(float(row.local_score), 6),
                    "route_global_score": round(float(row.route_global_score), 6),
                    "candidate_kind": "singleton_attach",
                    "cluster_pair_key": cluster_pair_key,
                }
            )

    merge_candidate_df = empty_merge_candidates() if not candidate_rows else pd.DataFrame(candidate_rows)
    if not merge_candidate_df.empty:
        merge_candidate_df = merge_candidate_df.sort_values(
            ["max_pair_probability", "support_pair_count", "candidate_kind", "cluster_pair_key"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    pair_disagreement_df = empty_pair_disagreement() if not pair_rows else pd.DataFrame(pair_rows)
    if not pair_disagreement_df.empty:
        pair_disagreement_df = pair_disagreement_df.sort_values(
            ["cluster_pair_key", "xgb_same_identity_prob", "local_score", "route_global_score"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
    rejected_candidate_df = empty_rejected_candidates() if not rejected_rows else pd.DataFrame(rejected_rows)
    if not rejected_candidate_df.empty:
        rejected_candidate_df = rejected_candidate_df.sort_values(
            ["candidate_kind", "reason", "mean_pair_probability", "max_pair_probability", "image_id"],
            ascending=[True, True, False, False, True],
        ).reset_index(drop=True)

    stats_df = pd.DataFrame(
        [
            {"metric": "dataset", "value": str(dataset)},
            {"metric": "dataset_image_count", "value": int(len(pred_frame))},
            {"metric": "cluster_count", "value": int(cluster_sizes.size)},
            {"metric": "singleton_count", "value": int(len(singleton_cluster_ids))},
            {"metric": "manual_singleton_count", "value": int(len(manual_singleton_ids))},
            {
                "metric": "singleton_singleton_candidate_count",
                "value": int(
                    merge_candidate_df["candidate_kind"].astype(str).eq("singleton_singleton").sum()
                    if not merge_candidate_df.empty
                    else 0
                ),
            },
            {
                "metric": "singleton_attach_candidate_count",
                "value": int(
                    merge_candidate_df["candidate_kind"].astype(str).eq("singleton_attach").sum()
                    if not merge_candidate_df.empty
                    else 0
                ),
            },
            {"metric": "accepted_candidate_count", "value": int(len(merge_candidate_df))},
            {"metric": "support_pair_row_count", "value": int(len(pair_disagreement_df))},
            {"metric": "rejected_candidate_count", "value": int(len(rejected_candidate_df))},
        ]
    )
    return SingletonRescueReviewResult(
        merge_candidate_df=merge_candidate_df,
        pair_df=pair_disagreement_df,
        stats_df=stats_df,
        rejected_candidate_df=rejected_candidate_df,
    )
