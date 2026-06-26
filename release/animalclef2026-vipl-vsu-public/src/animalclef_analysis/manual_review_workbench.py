from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .manual_cluster_overlay import ACTION_ATTACH_TO_ANCHOR, ACTION_SPLIT_TO_SINGLETONS


PAIR_LABEL_YES = "yes"
PAIR_LABEL_NO = "no"
PAIR_LABEL_UNCERTAIN = "uncertain"
SUPPORTED_PAIR_LABELS = {PAIR_LABEL_YES, PAIR_LABEL_NO, PAIR_LABEL_UNCERTAIN}


def _empty_split_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "base_cluster_id",
            "base_cluster_size",
            "ambiguous_image_count",
            "ambiguous_pair_count",
            "max_split_votes",
            "mean_pair_probability",
            "max_pair_probability",
            "mean_ambiguity_score",
            "max_ambiguity_score",
            "mean_border_score",
            "max_conflict_ratio",
            "conflict_methods",
            "component_ids",
            "image_indices",
            "image_ids",
        ]
    )


def _empty_merge_candidates() -> pd.DataFrame:
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


def _empty_pair_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_type",
            "candidate_key",
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
        ]
    )


def _empty_yes_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "candidate_key",
            "dataset",
            "source_candidate_type",
            "source_candidate_key",
            "candidate_kind",
            "candidate_preview",
            "priority_score",
            "pair_count",
            "unique_image_count",
            "triangle_pair_count",
            "extension_pair_count",
            "local_supported_pair_count",
            "image_ids",
            "top_pair_keys",
        ]
    )


@dataclass
class ReviewBundle:
    repo_root: Path
    predictions_path: Path
    probe_dir: Path | None
    pred_df: pd.DataFrame
    split_candidate_df: pd.DataFrame
    merge_candidate_df: pd.DataFrame
    pair_df: pd.DataFrame
    yes_candidate_df: pd.DataFrame
    yes_pair_df: pd.DataFrame
    cluster_sizes_df: pd.DataFrame


def _empty_judgment_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "judgment_id",
            "dataset",
            "candidate_type",
            "candidate_key",
            "pair_key",
            "image_id",
            "neighbor_image_id",
            "base_cluster_left",
            "base_cluster_right",
            "xgb_same_identity_prob",
            "ambiguity_score",
            "label",
            "note",
        ]
    )


def resolve_predictions_path(repo_root: Path, input_path: str | Path) -> Path:
    raw = Path(input_path)
    path = raw if raw.is_absolute() else (repo_root / raw)
    path = path.resolve()
    if path.is_file():
        return path
    candidates = [
        path / "tables" / "test_predictions_v1.csv",
        path / "test_predictions_v1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve test_predictions_v1.csv from {input_path!r}")


def _load_optional_csv(path: Path, empty_factory: Any) -> pd.DataFrame:
    if not path.exists():
        return empty_factory()
    return pd.read_csv(path)


def load_review_bundle(
    *,
    repo_root: Path,
    base_predictions_path: str | Path,
    probe_dir: str | Path | None = None,
) -> ReviewBundle:
    predictions_path = resolve_predictions_path(repo_root, base_predictions_path)
    pred_df = pd.read_csv(predictions_path).copy()
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    pred_df["pred_cluster_id"] = pred_df["pred_cluster_id"].astype(int)
    pred_df["cluster_label"] = pred_df["cluster_label"].astype(str)
    if "path" in pred_df.columns:
        pred_df["path"] = pred_df["path"].astype(str)

    resolved_probe_dir: Path | None = None
    if probe_dir:
        raw_probe = Path(probe_dir)
        resolved_probe_dir = raw_probe if raw_probe.is_absolute() else (repo_root / raw_probe)
        resolved_probe_dir = resolved_probe_dir.resolve()

    split_candidate_df = (
        _load_optional_csv(resolved_probe_dir / "tables" / "test_split_candidates_v1.csv", _empty_split_candidates)
        if resolved_probe_dir is not None
        else _empty_split_candidates()
    )
    merge_candidate_df = (
        _load_optional_csv(resolved_probe_dir / "tables" / "test_merge_candidates_v1.csv", _empty_merge_candidates)
        if resolved_probe_dir is not None
        else _empty_merge_candidates()
    )
    pair_df = (
        _load_optional_csv(resolved_probe_dir / "tables" / "test_pair_disagreement_v1.csv", _empty_pair_df)
        if resolved_probe_dir is not None
        else _empty_pair_df()
    )
    yes_candidate_df = (
        _load_optional_csv(resolved_probe_dir / "tables" / "test_yes_candidates_v1.csv", _empty_yes_candidates)
        if resolved_probe_dir is not None
        else _empty_yes_candidates()
    )
    yes_pair_df = (
        _load_optional_csv(resolved_probe_dir / "tables" / "test_yes_pair_candidates_v1.csv", _empty_pair_df)
        if resolved_probe_dir is not None
        else _empty_pair_df()
    )
    if not split_candidate_df.empty:
        split_candidate_df["base_cluster_id"] = split_candidate_df["base_cluster_id"].astype(int)
    if not merge_candidate_df.empty:
        merge_candidate_df["cluster_pair_key"] = merge_candidate_df["cluster_pair_key"].astype(str)
        merge_candidate_df["left_cluster_id"] = merge_candidate_df["left_cluster_id"].astype(int)
        merge_candidate_df["right_cluster_id"] = merge_candidate_df["right_cluster_id"].astype(int)
    if not pair_df.empty:
        pair_df["image_id"] = pair_df["image_id"].astype(str)
        pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
        for column in ["base_cluster_left", "base_cluster_right", "component_id"]:
            if column in pair_df.columns:
                pair_df[column] = pd.to_numeric(pair_df[column], errors="coerce").fillna(-1).astype(int)
    if not yes_candidate_df.empty:
        for column in [
            "candidate_key",
            "dataset",
            "source_candidate_type",
            "source_candidate_key",
            "candidate_kind",
            "candidate_preview",
            "image_ids",
            "top_pair_keys",
        ]:
            if column in yes_candidate_df.columns:
                yes_candidate_df[column] = yes_candidate_df[column].astype(str)
    if not yes_pair_df.empty:
        for column in ["candidate_key", "candidate_type", "image_id", "neighbor_image_id"]:
            if column in yes_pair_df.columns:
                yes_pair_df[column] = yes_pair_df[column].astype(str)
        for column in ["base_cluster_left", "base_cluster_right", "component_id"]:
            if column in yes_pair_df.columns:
                yes_pair_df[column] = pd.to_numeric(yes_pair_df[column], errors="coerce").fillna(-1).astype(int)

    cluster_sizes_df = (
        pred_df.groupby(["dataset", "pred_cluster_id"], sort=True)
        .size()
        .reset_index(name="cluster_size")
        .sort_values(["dataset", "cluster_size", "pred_cluster_id"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return ReviewBundle(
        repo_root=repo_root,
        predictions_path=predictions_path,
        probe_dir=resolved_probe_dir,
        pred_df=pred_df,
        split_candidate_df=split_candidate_df,
        merge_candidate_df=merge_candidate_df,
        pair_df=pair_df,
        yes_candidate_df=yes_candidate_df,
        yes_pair_df=yes_pair_df,
        cluster_sizes_df=cluster_sizes_df,
    )


def list_dataset_choices(bundle: ReviewBundle) -> list[tuple[str, str]]:
    datasets = sorted(bundle.pred_df["dataset"].astype(str).drop_duplicates().tolist())
    return [(dataset, dataset) for dataset in datasets]


def list_cluster_choices(bundle: ReviewBundle, dataset: str) -> list[tuple[str, str]]:
    subset = bundle.cluster_sizes_df[bundle.cluster_sizes_df["dataset"].astype(str).eq(str(dataset))].copy()
    choices: list[tuple[str, str]] = []
    for row in subset.itertuples(index=False):
        value = str(int(row.pred_cluster_id))
        label = f"{value} | size={int(row.cluster_size)}"
        choices.append((label, value))
    return choices


def list_candidate_choices(bundle: ReviewBundle, direction: str) -> list[tuple[str, str]]:
    direction = str(direction)
    if direction == "split":
        subset = bundle.split_candidate_df.sort_values(
            ["max_ambiguity_score", "base_cluster_size", "base_cluster_id"],
            ascending=[False, True, True],
        )
        return [
            (
                f"{int(row.base_cluster_id)} | size={int(row.base_cluster_size)} | score={float(row.max_ambiguity_score):.3f} | images={str(row.image_ids)}",
                str(int(row.base_cluster_id)),
            )
            for row in subset.itertuples(index=False)
        ]
    if direction == "yes":
        subset = bundle.yes_candidate_df.sort_values(
            ["priority_score", "pair_count", "candidate_key"],
            ascending=[False, False, True],
        )
        return [
            (
                (
                    f"{str(getattr(row, 'candidate_preview', '')).strip() or str(row.candidate_key)}"
                    f" | score={float(getattr(row, 'priority_score', 0.0)):.3f}"
                    f" | pairs={int(getattr(row, 'pair_count', 0))}"
                ),
                str(row.candidate_key),
            )
            for row in subset.itertuples(index=False)
        ]
    subset = bundle.merge_candidate_df.sort_values(
        ["max_ambiguity_score", "merged_total_size", "cluster_pair_key"],
        ascending=[False, True, True],
    )
    return [
        (
            (
                f"{str(getattr(row, 'candidate_preview', '')).strip() or str(row.cluster_pair_key)}"
                f" | total={int(row.merged_total_size)}"
                f" | score={float(row.max_ambiguity_score):.3f}"
                f" | support={int(row.support_pair_count)}"
            ),
            str(row.cluster_pair_key),
        )
        for row in subset.itertuples(index=False)
    ]


def _resolve_abs_path(repo_root: Path, rel_path: str) -> str | None:
    if not rel_path:
        return None
    path = Path(rel_path)
    resolved = path if path.is_absolute() else (repo_root / path)
    resolved = resolved.resolve()
    return str(resolved) if resolved.exists() else None


def _gallery_rows_to_items(bundle: ReviewBundle, frame: pd.DataFrame) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if frame.empty or "path" not in frame.columns:
        return items
    for row in frame.itertuples(index=False):
        abs_path = _resolve_abs_path(bundle.repo_root, str(row.path))
        if abs_path is None:
            continue
        caption_bits = [f"image={row.image_id}"]
        if hasattr(row, "pred_cluster_id"):
            caption_bits.append(f"cluster={int(row.pred_cluster_id)}")
        items.append((abs_path, " | ".join(caption_bits)))
    return items


def _support_pair_choices(pair_df: pd.DataFrame) -> list[tuple[str, str]]:
    rows = pair_df.reset_index(drop=True)
    choices: list[tuple[str, str]] = []
    for idx, row in rows.iterrows():
        local_suffix = ""
        if "local_score" in row and pd.notna(row["local_score"]):
            local_suffix = f" | local={float(row.get('local_score', 0.0)):.3f}"
        label = (
            f"{idx} | {row['image_id']} vs {row['neighbor_image_id']} | "
            f"sim={float(row.get('xgb_same_identity_prob', 0.0)):.3f} | "
            f"amb={float(row.get('ambiguity_score', 0.0)):.3f}"
            f"{local_suffix}"
        )
        choices.append((label, str(idx)))
    return choices


def _pair_markdown(row: pd.Series) -> str:
    bits = [
        f"- `image_id`: `{row['image_id']}` vs `{row['neighbor_image_id']}`",
        f"- `xgb_same_identity_prob`: `{float(row.get('xgb_same_identity_prob', 0.0)):.6f}`",
        f"- `ambiguity_score`: `{float(row.get('ambiguity_score', 0.0)):.6f}`",
        f"- `vote_direction`: `{row.get('vote_direction', '')}`",
        f"- `merge_votes/split_votes`: `{int(row.get('merge_votes', 0))}/{int(row.get('split_votes', 0))}`",
        f"- `conflict_methods`: `{row.get('conflict_methods', '')}`",
    ]
    if "local_score" in row.index and pd.notna(row.get("local_score")):
        bits.append(f"- `local_score`: `{float(row.get('local_score', 0.0)):.6f}`")
    if "route_global_score" in row.index and pd.notna(row.get("route_global_score")):
        bits.append(f"- `route_global_score`: `{float(row.get('route_global_score', 0.0)):.6f}`")
    if "candidate_kind" in row.index and str(row.get("candidate_kind", "")).strip():
        bits.append(f"- `candidate_kind`: `{row.get('candidate_kind', '')}`")
    if "yes_priority_score" in row.index and pd.notna(row.get("yes_priority_score")):
        bits.append(f"- `yes_priority_score`: `{float(row.get('yes_priority_score', 0.0)):.6f}`")
    if "yes_candidate_reason" in row.index and str(row.get("yes_candidate_reason", "")).strip():
        bits.append(f"- `yes_candidate_reason`: `{row.get('yes_candidate_reason', '')}`")
    if "source_candidate_type" in row.index and str(row.get("source_candidate_type", "")).strip():
        bits.append(
            f"- `source_candidate`: `{row.get('source_candidate_type', '')}` `{row.get('source_candidate_key', '')}`"
        )
    if "existing_yes_component_size" in row.index and pd.notna(row.get("existing_yes_component_size")):
        bits.append(f"- `existing_yes_component_size`: `{int(row.get('existing_yes_component_size', 0))}`")
    return "\n".join(bits)


def _rows_for_cluster(bundle: ReviewBundle, dataset: str, cluster_id: int) -> pd.DataFrame:
    subset = bundle.pred_df[
        bundle.pred_df["dataset"].astype(str).eq(str(dataset))
        & bundle.pred_df["pred_cluster_id"].astype(int).eq(int(cluster_id))
    ].copy()
    return subset.sort_values("image_id").reset_index(drop=True)


def _split_pipe_values(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    text = str(raw).strip()
    if not text:
        return []
    return [chunk.strip() for chunk in text.split("|") if chunk.strip()]


def _infer_dataset_from_image_ids(bundle: ReviewBundle, image_ids: list[str]) -> str:
    if not image_ids:
        return ""
    subset = bundle.pred_df[bundle.pred_df["image_id"].astype(str).isin([str(value) for value in image_ids])].copy()
    if subset.empty:
        return ""
    datasets = subset["dataset"].astype(str).drop_duplicates().tolist()
    return str(datasets[0]) if datasets else ""


def render_cluster_view(bundle: ReviewBundle, dataset: str, cluster_id: str) -> dict[str, Any]:
    cluster_int = int(cluster_id)
    members_df = _rows_for_cluster(bundle, dataset=dataset, cluster_id=cluster_int)
    gallery_items = _gallery_rows_to_items(bundle, members_df)
    summary = "\n".join(
        [
            f"- `dataset`: `{dataset}`",
            f"- `cluster_id`: `{cluster_int}`",
            f"- `cluster_size`: `{len(members_df)}`",
            f"- `image_ids`: `{ '|'.join(members_df['image_id'].astype(str).tolist()) }`",
        ]
    )
    member_choices = [(f"{row.image_id}", str(row.image_id)) for row in members_df.itertuples(index=False)]
    return {
        "summary_markdown": summary,
        "gallery_items": gallery_items,
        "anchor_choices": member_choices,
        "member_choices": member_choices,
        "default_selected_members": [value for _, value in member_choices],
    }


def render_candidate_view(bundle: ReviewBundle, direction: str, candidate_value: str) -> dict[str, Any]:
    direction = str(direction)
    if direction == "split":
        cluster_id = int(candidate_value)
        candidate_row = bundle.split_candidate_df[
            bundle.split_candidate_df["base_cluster_id"].astype(int).eq(cluster_id)
        ].iloc[0]
        candidate_image_ids = _split_pipe_values(candidate_row.get("image_ids", ""))
        dataset = _infer_dataset_from_image_ids(bundle, candidate_image_ids)
        dataset_rows = _rows_for_cluster(bundle, dataset=dataset, cluster_id=cluster_id) if dataset else pd.DataFrame(columns=bundle.pred_df.columns)
        support_df = bundle.pair_df[
            bundle.pair_df["vote_direction"].astype(str).eq("split")
            & bundle.pair_df["base_cluster_left"].astype(int).eq(cluster_id)
            & bundle.pair_df["base_cluster_right"].astype(int).eq(cluster_id)
        ].copy()
        if dataset:
            support_df = support_df[
                support_df["image_id"].astype(str).isin(dataset_rows["image_id"].astype(str))
                | support_df["neighbor_image_id"].astype(str).isin(dataset_rows["image_id"].astype(str))
            ].copy()
        support_df = support_df.sort_values(
            ["split_votes", "ambiguity_score", "xgb_same_identity_prob"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        member_choices = [(f"{row.image_id}", str(row.image_id)) for row in dataset_rows.sort_values("image_id").itertuples(index=False)]
        summary = "\n".join(
            [
                f"- `candidate`: `split cluster {cluster_id}`",
                f"- `dataset`: `{dataset}`",
                f"- `base_cluster_size`: `{int(candidate_row['base_cluster_size'])}`",
                f"- `ambiguous_pair_count`: `{int(candidate_row['ambiguous_pair_count'])}`",
                f"- `max_split_votes`: `{int(candidate_row['max_split_votes'])}`",
                f"- `max_ambiguity_score`: `{float(candidate_row['max_ambiguity_score']):.6f}`",
                f"- `image_ids`: `{str(candidate_row['image_ids'])}`",
                f"- `conflict_methods`: `{str(candidate_row['conflict_methods'])}`",
            ]
        )
        return {
            "dataset": dataset,
            "candidate_key": str(cluster_id),
            "summary_markdown": summary,
            "gallery_items": _gallery_rows_to_items(bundle, dataset_rows.sort_values("image_id")),
            "pair_df": support_df,
            "pair_choices": _support_pair_choices(support_df),
            "anchor_choices": member_choices,
            "member_choices": member_choices,
            "default_selected_members": [value for _, value in member_choices],
            "default_note": f"split cluster {cluster_id}",
        }
    if direction == "yes":
        candidate_key = str(candidate_value)
        candidate_row = bundle.yes_candidate_df[
            bundle.yes_candidate_df["candidate_key"].astype(str).eq(candidate_key)
        ].iloc[0]
        support_df = bundle.yes_pair_df[
            bundle.yes_pair_df["candidate_key"].astype(str).eq(candidate_key)
        ].copy()
        dataset = str(candidate_row.get("dataset", "")).strip()
        image_ids = _split_pipe_values(candidate_row.get("image_ids", ""))
        dataset_rows = (
            bundle.pred_df[
                bundle.pred_df["dataset"].astype(str).eq(dataset)
                & bundle.pred_df["image_id"].astype(str).isin(image_ids)
            ].copy()
            if dataset and image_ids
            else pd.DataFrame(columns=bundle.pred_df.columns)
        )
        support_df = support_df.sort_values(
            ["yes_priority_score", "ambiguity_score", "xgb_same_identity_prob"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        member_choices = [
            (f"{row.image_id} | cluster={int(row.pred_cluster_id)}", str(row.image_id))
            for row in dataset_rows.sort_values(["pred_cluster_id", "image_id"]).itertuples(index=False)
        ]
        summary_bits = [
            f"- `candidate`: `yes {candidate_key}`",
            f"- `dataset`: `{dataset}`",
            f"- `source_candidate`: `{str(candidate_row.get('source_candidate_type', ''))}` `{str(candidate_row.get('source_candidate_key', ''))}`",
            f"- `candidate_kind`: `{str(candidate_row.get('candidate_kind', ''))}`",
            f"- `priority_score`: `{float(candidate_row.get('priority_score', 0.0)):.6f}`",
            f"- `pair_count`: `{int(candidate_row.get('pair_count', 0))}`",
            f"- `unique_image_count`: `{int(candidate_row.get('unique_image_count', 0))}`",
            f"- `triangle_pair_count`: `{int(candidate_row.get('triangle_pair_count', 0))}`",
            f"- `extension_pair_count`: `{int(candidate_row.get('extension_pair_count', 0))}`",
            f"- `local_supported_pair_count`: `{int(candidate_row.get('local_supported_pair_count', 0))}`",
            f"- `image_ids`: `{str(candidate_row.get('image_ids', ''))}`",
        ]
        preview = str(candidate_row.get("candidate_preview", "")).strip()
        if preview:
            summary_bits.append(f"- `preview`: `{preview}`")
        return {
            "dataset": dataset,
            "candidate_key": candidate_key,
            "summary_markdown": "\n".join(summary_bits),
            "gallery_items": _gallery_rows_to_items(bundle, dataset_rows.sort_values(["pred_cluster_id", "image_id"])),
            "pair_df": support_df,
            "pair_choices": _support_pair_choices(support_df),
            "anchor_choices": member_choices,
            "member_choices": member_choices,
            "default_selected_members": [value for _, value in member_choices],
            "default_note": f"yes candidate {candidate_key}",
        }

    merge_key = str(candidate_value)
    candidate_row = bundle.merge_candidate_df[
        bundle.merge_candidate_df["cluster_pair_key"].astype(str).eq(merge_key)
    ].iloc[0]
    left_cluster_id = int(candidate_row["left_cluster_id"])
    right_cluster_id = int(candidate_row["right_cluster_id"])
    support_df = bundle.pair_df[
        bundle.pair_df["vote_direction"].astype(str).eq("merge")
        & (
            (
                bundle.pair_df["base_cluster_left"].astype(int).eq(left_cluster_id)
                & bundle.pair_df["base_cluster_right"].astype(int).eq(right_cluster_id)
            )
            | (
                bundle.pair_df["base_cluster_left"].astype(int).eq(right_cluster_id)
                & bundle.pair_df["base_cluster_right"].astype(int).eq(left_cluster_id)
            )
        )
    ].copy()
    dataset = _infer_dataset_from_image_ids(
        bundle,
        support_df["image_id"].astype(str).tolist() + support_df["neighbor_image_id"].astype(str).tolist(),
    )
    dataset_rows = (
        bundle.pred_df[
            bundle.pred_df["dataset"].astype(str).eq(dataset)
            & bundle.pred_df["pred_cluster_id"].astype(int).isin([left_cluster_id, right_cluster_id])
        ].copy()
        if dataset
        else pd.DataFrame(columns=bundle.pred_df.columns)
    )
    if dataset:
        support_df = support_df[
            support_df["image_id"].astype(str).isin(dataset_rows["image_id"].astype(str))
            | support_df["neighbor_image_id"].astype(str).isin(dataset_rows["image_id"].astype(str))
        ].copy()
    support_df = support_df.sort_values(
        ["merge_votes", "ambiguity_score", "xgb_same_identity_prob"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    member_choices = [(f"{row.image_id} | cluster={int(row.pred_cluster_id)}", str(row.image_id)) for row in dataset_rows.sort_values(["pred_cluster_id", "image_id"]).itertuples(index=False)]
    summary = "\n".join(
        [
            f"- `candidate`: `merge {merge_key}`",
            f"- `candidate_kind`: `{str(candidate_row.get('candidate_kind', 'merge'))}`",
            f"- `candidate_preview`: `{str(candidate_row.get('candidate_preview', merge_key))}`",
            f"- `dataset`: `{dataset}`",
            f"- `left/right_size`: `{left_cluster_id}:{int(candidate_row['left_cluster_size'])}` / `{right_cluster_id}:{int(candidate_row['right_cluster_size'])}`",
            f"- `support_pair_count`: `{int(candidate_row['support_pair_count'])}`",
            f"- `max_merge_votes`: `{int(candidate_row['max_merge_votes'])}`",
            f"- `max_ambiguity_score`: `{float(candidate_row['max_ambiguity_score']):.6f}`",
            f"- `conflict_methods`: `{str(candidate_row['conflict_methods'])}`",
        ]
    )
    optional_lines = []
    if "support_image_ids" in candidate_row.index and str(candidate_row.get("support_image_ids", "")).strip():
        optional_lines.append(f"- `support_image_ids`: `{str(candidate_row.get('support_image_ids', ''))}`")
    if "origin_cluster_id" in candidate_row.index and str(candidate_row.get("origin_cluster_id", "")).strip():
        optional_lines.append(f"- `origin_cluster_id`: `{str(candidate_row.get('origin_cluster_id', ''))}`")
    if optional_lines:
        summary = summary + "\n" + "\n".join(optional_lines)
    return {
        "dataset": dataset,
        "candidate_key": merge_key,
        "summary_markdown": summary,
        "gallery_items": _gallery_rows_to_items(bundle, dataset_rows.sort_values(["pred_cluster_id", "image_id"])),
        "pair_df": support_df,
        "pair_choices": _support_pair_choices(support_df),
        "anchor_choices": member_choices,
        "member_choices": member_choices,
        "default_selected_members": [value for _, value in member_choices],
        "default_note": f"merge candidate {merge_key}",
    }


def render_pair_detail(bundle: ReviewBundle, pair_df: pd.DataFrame, pair_index: str | int | None) -> dict[str, Any]:
    if pair_df.empty:
        return {
            "left_image": None,
            "right_image": None,
            "pair_markdown": "_No support pair selected._",
        }
    pair_idx = 0 if pair_index in (None, "") else int(pair_index)
    pair_idx = max(0, min(pair_idx, len(pair_df) - 1))
    row = pair_df.iloc[pair_idx]
    left_row = bundle.pred_df[bundle.pred_df["image_id"].astype(str).eq(str(row["image_id"]))].head(1)
    right_row = bundle.pred_df[bundle.pred_df["image_id"].astype(str).eq(str(row["neighbor_image_id"]))].head(1)
    left_image = None if left_row.empty else _resolve_abs_path(bundle.repo_root, str(left_row.iloc[0].get("path", "")))
    right_image = None if right_row.empty else _resolve_abs_path(bundle.repo_root, str(right_row.iloc[0].get("path", "")))
    return {
        "left_image": left_image,
        "right_image": right_image,
        "pair_markdown": _pair_markdown(row),
    }


def get_pair_row(pair_df: pd.DataFrame, pair_index: str | int | None) -> pd.Series:
    if pair_df.empty:
        raise ValueError("No pair rows available")
    pair_idx = 0 if pair_index in (None, "") else int(pair_index)
    pair_idx = max(0, min(pair_idx, len(pair_df) - 1))
    return pair_df.iloc[pair_idx]


def _pair_key(image_id: str, neighbor_image_id: str) -> str:
    left, right = sorted([str(image_id), str(neighbor_image_id)])
    return f"{left}|{right}"


def _judgment_pair_key(item: dict[str, Any]) -> str:
    pair_key = str(item.get("pair_key", "")).strip()
    if pair_key:
        return pair_key
    image_id = str(item.get("image_id", "")).strip()
    neighbor_image_id = str(item.get("neighbor_image_id", "")).strip()
    if not image_id or not neighbor_image_id:
        return ""
    return _pair_key(image_id, neighbor_image_id)


def _pair_keys_from_frame(pair_df: pd.DataFrame) -> list[str]:
    if pair_df is None or pair_df.empty:
        return []
    ordered_keys: list[str] = []
    seen: set[str] = set()
    for row in pair_df.itertuples(index=False):
        pair_key = _pair_key(str(getattr(row, "image_id", "")), str(getattr(row, "neighbor_image_id", "")))
        if not pair_key or pair_key in seen:
            continue
        seen.add(pair_key)
        ordered_keys.append(pair_key)
    return ordered_keys


def _judgment_map_for_scope(
    judgments: list[dict[str, Any]],
    *,
    dataset: str,
    candidate_type: str,
    candidate_key: str,
    treat_dataset_pair_as_judged: bool = False,
) -> dict[str, dict[str, Any]]:
    scoped_items: dict[str, dict[str, Any]] = {}
    for item in judgments:
        if str(item.get("dataset", "")) != str(dataset):
            continue
        if not treat_dataset_pair_as_judged:
            if str(item.get("candidate_type", "")) != str(candidate_type):
                continue
            if str(item.get("candidate_key", "")) != str(candidate_key):
                continue
        pair_key = _judgment_pair_key(item)
        if not pair_key:
            continue
        normalized_item = dict(item)
        normalized_item["pair_key"] = pair_key
        scoped_items[pair_key] = normalized_item
    return scoped_items


def _aggregate_pair_frame_judgment_counts(
    judgments: list[dict[str, Any]],
    pair_df: pd.DataFrame,
    *,
    dataset: str,
    candidate_type: str,
    candidate_key: str,
    treat_dataset_pair_as_judged: bool = False,
) -> dict[str, int]:
    pair_keys = _pair_keys_from_frame(pair_df)
    if not pair_keys:
        return {
            "judged_pairs": 0,
            "yes_count": 0,
            "no_count": 0,
            "uncertain_count": 0,
        }
    judgment_map = _judgment_map_for_scope(
        judgments,
        dataset=str(dataset),
        candidate_type=str(candidate_type),
        candidate_key=str(candidate_key),
        treat_dataset_pair_as_judged=bool(treat_dataset_pair_as_judged),
    )
    labels = [str(judgment_map[pair_key].get("label", "")).strip().lower() for pair_key in pair_keys if pair_key in judgment_map]
    return {
        "judged_pairs": int(len(labels)),
        "yes_count": int(sum(label == PAIR_LABEL_YES for label in labels)),
        "no_count": int(sum(label == PAIR_LABEL_NO for label in labels)),
        "uncertain_count": int(sum(label == PAIR_LABEL_UNCERTAIN for label in labels)),
    }


def find_next_unjudged_pair_index(
    pair_df: pd.DataFrame,
    judgments: list[dict[str, Any]],
    *,
    dataset: str,
    candidate_type: str,
    candidate_key: str,
    start_after_index: int | None = None,
    treat_dataset_pair_as_judged: bool = False,
) -> int | None:
    if pair_df.empty:
        return None

    pair_keys = _pair_keys_from_frame(pair_df)
    if not pair_keys:
        return None

    judged_pair_keys = set(
        _judgment_map_for_scope(
            judgments,
            dataset=str(dataset),
            candidate_type=str(candidate_type),
            candidate_key=str(candidate_key),
            treat_dataset_pair_as_judged=bool(treat_dataset_pair_as_judged),
        ).keys()
    )
    judged_pair_keys = judged_pair_keys.intersection(pair_keys)
    if len(judged_pair_keys) >= len(pair_keys):
        return None

    row_count = int(len(pair_df))
    start_index = -1 if start_after_index is None else int(start_after_index)
    scan_order = [((start_index + 1 + offset) % row_count) for offset in range(row_count)]
    for pair_idx in scan_order:
        row = pair_df.iloc[pair_idx]
        pair_key = _pair_key(str(row["image_id"]), str(row["neighbor_image_id"]))
        if pair_key not in judged_pair_keys:
            return int(pair_idx)
    return None


def judgments_to_dataframe(judgments: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        judgments,
        columns=[
            "judgment_id",
            "dataset",
            "candidate_type",
            "candidate_key",
            "pair_key",
            "image_id",
            "neighbor_image_id",
            "base_cluster_left",
            "base_cluster_right",
            "xgb_same_identity_prob",
            "ambiguity_score",
            "label",
            "note",
        ],
    )


def build_judgment_preview_json(
    session_name: str,
    judgments: list[dict[str, Any]],
    *,
    max_preview_items: int = 80,
) -> str:
    if max_preview_items <= 0 or len(judgments) <= max_preview_items:
        payload = {"session_name": str(session_name), "pair_judgments": judgments}
        return json.dumps(payload, indent=2, ensure_ascii=False)

    preview_items = list(judgments[-max_preview_items:])
    payload = {
        "session_name": str(session_name),
        "preview_truncated": True,
        "total_judgments": int(len(judgments)),
        "preview_count": int(len(preview_items)),
        "omitted_count": int(len(judgments) - len(preview_items)),
        "preview_note": "UI preview only shows recent judgments to keep the workbench responsive.",
        "pair_judgments_tail": preview_items,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _candidate_status_from_progress(pair_total: int, judged_pairs: int) -> tuple[str, str]:
    if int(pair_total) <= 0:
        return "empty", "∅ 无 pair"
    if int(judged_pairs) <= 0:
        return "pending", "☐ 待处理"
    if int(judged_pairs) >= int(pair_total):
        return "completed", "☑ 已完成"
    return "in_progress", "◐ 进行中"


def _split_support_pair_frame(bundle: ReviewBundle, cluster_id: int) -> pd.DataFrame:
    return bundle.pair_df[
        bundle.pair_df["vote_direction"].astype(str).eq("split")
        & bundle.pair_df["base_cluster_left"].astype(int).eq(int(cluster_id))
        & bundle.pair_df["base_cluster_right"].astype(int).eq(int(cluster_id))
    ].copy()


def _merge_support_pair_frame(bundle: ReviewBundle, left_cluster_id: int, right_cluster_id: int) -> pd.DataFrame:
    return bundle.pair_df[
        bundle.pair_df["vote_direction"].astype(str).eq("merge")
        & (
            (
                bundle.pair_df["base_cluster_left"].astype(int).eq(int(left_cluster_id))
                & bundle.pair_df["base_cluster_right"].astype(int).eq(int(right_cluster_id))
            )
            | (
                bundle.pair_df["base_cluster_left"].astype(int).eq(int(right_cluster_id))
                & bundle.pair_df["base_cluster_right"].astype(int).eq(int(left_cluster_id))
            )
        )
    ].copy()


def _aggregate_candidate_judgment_counts(
    summary_df: pd.DataFrame,
    *,
    candidate_type: str,
    candidate_key: str,
    dataset: str,
) -> dict[str, int]:
    if summary_df.empty:
        return {
            "judged_pairs": 0,
            "yes_count": 0,
            "no_count": 0,
            "uncertain_count": 0,
        }
    subset = summary_df[
        summary_df["candidate_type"].astype(str).eq(str(candidate_type))
        & summary_df["candidate_key"].astype(str).eq(str(candidate_key))
    ].copy()
    if dataset:
        exact_subset = subset[subset["dataset"].astype(str).eq(str(dataset))].copy()
        if not exact_subset.empty:
            subset = exact_subset
    if subset.empty:
        return {
            "judged_pairs": 0,
            "yes_count": 0,
            "no_count": 0,
            "uncertain_count": 0,
        }
    return {
        "judged_pairs": int(subset["judged_pairs"].fillna(0).sum()),
        "yes_count": int(subset["yes_count"].fillna(0).sum()),
        "no_count": int(subset["no_count"].fillna(0).sum()),
        "uncertain_count": int(subset["uncertain_count"].fillna(0).sum()),
    }


def build_candidate_task_table(
    bundle: ReviewBundle,
    direction: str,
    judgments: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    direction = str(direction)

    if direction == "split":
        candidate_df = bundle.split_candidate_df.sort_values(
            ["max_ambiguity_score", "base_cluster_size", "base_cluster_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        for priority_rank, row in enumerate(candidate_df.itertuples(index=False), start=1):
            cluster_id = int(row.base_cluster_id)
            candidate_key = str(cluster_id)
            image_ids = _split_pipe_values(getattr(row, "image_ids", ""))
            dataset = _infer_dataset_from_image_ids(bundle, image_ids)
            support_df = _split_support_pair_frame(bundle, cluster_id)
            pair_total = int(len(_pair_keys_from_frame(support_df)))
            judgment_counts = _aggregate_pair_frame_judgment_counts(
                judgments or [],
                support_df,
                candidate_type="split",
                candidate_key=candidate_key,
                dataset=dataset,
            )
            status_code, status = _candidate_status_from_progress(
                pair_total=pair_total,
                judged_pairs=judgment_counts["judged_pairs"],
            )
            rows.append(
                {
                    "status_code": status_code,
                    "status": status,
                    "dataset": dataset,
                    "candidate_type": "split",
                    "candidate_key": candidate_key,
                    "progress": f"{judgment_counts['judged_pairs']}/{pair_total}",
                    "judged_pairs": int(judgment_counts["judged_pairs"]),
                    "pair_total": pair_total,
                    "yes_count": int(judgment_counts["yes_count"]),
                    "no_count": int(judgment_counts["no_count"]),
                    "uncertain_count": int(judgment_counts["uncertain_count"]),
                    "priority_score": round(float(getattr(row, "max_ambiguity_score", 0.0)), 6),
                    "size_hint": int(getattr(row, "base_cluster_size", 0)),
                    "preview": str(getattr(row, "image_ids", "")),
                    "priority_rank": priority_rank,
                }
            )
    elif direction == "yes":
        candidate_df = bundle.yes_candidate_df.sort_values(
            ["priority_score", "pair_count", "candidate_key"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        for priority_rank, row in enumerate(candidate_df.itertuples(index=False), start=1):
            candidate_key = str(row.candidate_key)
            dataset = str(getattr(row, "dataset", "")).strip()
            support_df = bundle.yes_pair_df[bundle.yes_pair_df["candidate_key"].astype(str).eq(candidate_key)].copy()
            pair_total = int(len(_pair_keys_from_frame(support_df)))
            judgment_counts = _aggregate_pair_frame_judgment_counts(
                judgments or [],
                support_df,
                candidate_type="yes",
                candidate_key=candidate_key,
                dataset=dataset,
                treat_dataset_pair_as_judged=True,
            )
            status_code, status = _candidate_status_from_progress(
                pair_total=pair_total,
                judged_pairs=judgment_counts["judged_pairs"],
            )
            rows.append(
                {
                    "status_code": status_code,
                    "status": status,
                    "dataset": dataset,
                    "candidate_type": "yes",
                    "candidate_key": candidate_key,
                    "progress": f"{judgment_counts['judged_pairs']}/{pair_total}",
                    "judged_pairs": int(judgment_counts["judged_pairs"]),
                    "pair_total": pair_total,
                    "yes_count": int(judgment_counts["yes_count"]),
                    "no_count": int(judgment_counts["no_count"]),
                    "uncertain_count": int(judgment_counts["uncertain_count"]),
                    "priority_score": round(float(getattr(row, "priority_score", 0.0)), 6),
                    "size_hint": int(getattr(row, "unique_image_count", 0)),
                    "preview": str(getattr(row, "candidate_preview", "")).strip() or str(candidate_key),
                    "priority_rank": priority_rank,
                }
            )
    else:
        candidate_df = bundle.merge_candidate_df.sort_values(
            ["max_ambiguity_score", "merged_total_size", "cluster_pair_key"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        for priority_rank, row in enumerate(candidate_df.itertuples(index=False), start=1):
            candidate_key = str(row.cluster_pair_key)
            left_cluster_id = int(row.left_cluster_id)
            right_cluster_id = int(row.right_cluster_id)
            support_df = _merge_support_pair_frame(bundle, left_cluster_id, right_cluster_id)
            dataset = _infer_dataset_from_image_ids(
                bundle,
                support_df["image_id"].astype(str).tolist() + support_df["neighbor_image_id"].astype(str).tolist(),
            )
            pair_total = int(len(_pair_keys_from_frame(support_df)))
            judgment_counts = _aggregate_pair_frame_judgment_counts(
                judgments or [],
                support_df,
                candidate_type="merge",
                candidate_key=candidate_key,
                dataset=dataset,
            )
            status_code, status = _candidate_status_from_progress(
                pair_total=pair_total,
                judged_pairs=judgment_counts["judged_pairs"],
            )
            rows.append(
                {
                    "status_code": status_code,
                    "status": status,
                    "dataset": dataset,
                    "candidate_type": "merge",
                    "candidate_key": candidate_key,
                    "progress": f"{judgment_counts['judged_pairs']}/{pair_total}",
                    "judged_pairs": int(judgment_counts["judged_pairs"]),
                    "pair_total": pair_total,
                    "yes_count": int(judgment_counts["yes_count"]),
                    "no_count": int(judgment_counts["no_count"]),
                    "uncertain_count": int(judgment_counts["uncertain_count"]),
                    "priority_score": round(float(getattr(row, "max_ambiguity_score", 0.0)), 6),
                    "size_hint": int(getattr(row, "merged_total_size", 0)),
                    "preview": str(getattr(row, "candidate_preview", "")).strip()
                    or (
                        f"{int(getattr(row, 'left_cluster_id', -1))}|"
                        f"{int(getattr(row, 'right_cluster_id', -1))}"
                    ),
                    "priority_rank": priority_rank,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "status_code",
                "status",
                "dataset",
                "candidate_type",
                "candidate_key",
                "progress",
                "judged_pairs",
                "pair_total",
                "yes_count",
                "no_count",
                "uncertain_count",
                "priority_score",
                "size_hint",
                "preview",
                "priority_rank",
            ]
        )

    task_df = pd.DataFrame(rows)
    status_rank_map = {"in_progress": 0, "pending": 1, "completed": 2, "empty": 3}
    task_df["status_rank"] = task_df["status_code"].map(status_rank_map).fillna(9).astype(int)
    task_df = task_df.sort_values(
        ["status_rank", "priority_rank", "dataset", "candidate_key"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return task_df


def suggest_candidate_value(task_df: pd.DataFrame) -> str | None:
    if task_df.empty:
        return None
    return str(task_df.iloc[0]["candidate_key"])


def upsert_pair_judgment(
    judgments: list[dict[str, Any]],
    *,
    dataset: str,
    candidate_type: str,
    candidate_key: str,
    pair_row: pd.Series,
    label: str,
    note: str,
) -> list[dict[str, Any]]:
    label = str(label).strip().lower()
    if label not in SUPPORTED_PAIR_LABELS:
        raise ValueError(f"Unsupported pair label: {label!r}")

    image_id = str(pair_row["image_id"])
    neighbor_image_id = str(pair_row["neighbor_image_id"])
    pair_key = _pair_key(image_id, neighbor_image_id)
    next_judgments: list[dict[str, Any]] = []
    for item in judgments:
        if str(item.get("pair_key", "")) == pair_key and str(item.get("dataset", "")) == str(dataset):
            continue
        next_judgments.append(dict(item))
    next_judgments.append(
        {
            "judgment_id": _slug(f"{dataset}_{candidate_type}_{candidate_key}_{pair_key}"),
            "dataset": str(dataset),
            "candidate_type": str(candidate_type),
            "candidate_key": str(candidate_key),
            "pair_key": pair_key,
            "image_id": image_id,
            "neighbor_image_id": neighbor_image_id,
            "base_cluster_left": int(pair_row.get("base_cluster_left", -1)),
            "base_cluster_right": int(pair_row.get("base_cluster_right", -1)),
            "xgb_same_identity_prob": round(float(pair_row.get("xgb_same_identity_prob", 0.0)), 6),
            "ambiguity_score": round(float(pair_row.get("ambiguity_score", 0.0)), 6),
            "label": label,
            "note": str(note).strip(),
        }
    )
    next_judgments = sorted(
        next_judgments,
        key=lambda item: (
            str(item.get("candidate_type", "")),
            str(item.get("candidate_key", "")),
            str(item.get("pair_key", "")),
        ),
    )
    return next_judgments


def remove_pair_judgment_at(judgments: list[dict[str, Any]], index: int | None) -> list[dict[str, Any]]:
    if not judgments:
        return []
    if index is None:
        return list(judgments)
    row_index = int(index)
    if row_index < 0 or row_index >= len(judgments):
        return list(judgments)
    return [item for idx, item in enumerate(judgments) if idx != row_index]


def clear_pair_judgments() -> list[dict[str, Any]]:
    return []


def summarize_pair_judgments(judgments: list[dict[str, Any]]) -> pd.DataFrame:
    judgment_df = judgments_to_dataframe(judgments)
    if judgment_df.empty:
        return pd.DataFrame(
            columns=[
                "candidate_type",
                "candidate_key",
                "dataset",
                "judged_pairs",
                "yes_count",
                "no_count",
                "uncertain_count",
                "hint",
            ]
        )
    rows: list[dict[str, Any]] = []
    group_columns = ["candidate_type", "candidate_key", "dataset"]
    for (candidate_type, candidate_key, dataset), group in judgment_df.groupby(group_columns, sort=True):
        yes_count = int(group["label"].astype(str).eq(PAIR_LABEL_YES).sum())
        no_count = int(group["label"].astype(str).eq(PAIR_LABEL_NO).sum())
        uncertain_count = int(group["label"].astype(str).eq(PAIR_LABEL_UNCERTAIN).sum())
        hint = "needs_more_review"
        if str(candidate_type) == "split":
            if no_count > 0:
                hint = "has_split_evidence"
            elif yes_count > 0 and uncertain_count == 0:
                hint = "split_not_supported"
        elif str(candidate_type) == "yes":
            if yes_count > 0 and no_count == 0:
                hint = "has_yes_support"
            elif no_count > 0 and yes_count == 0:
                hint = "candidate_rejected"
            elif yes_count > 0 and no_count > 0:
                hint = "mixed_yes_signal"
        elif str(candidate_type) == "merge":
            if yes_count > 0 and no_count == 0:
                hint = "has_merge_evidence"
            elif no_count > 0:
                hint = "merge_conflicted"
        rows.append(
            {
                "candidate_type": str(candidate_type),
                "candidate_key": str(candidate_key),
                "dataset": str(dataset),
                "judged_pairs": int(len(group)),
                "yes_count": yes_count,
                "no_count": no_count,
                "uncertain_count": uncertain_count,
                "hint": hint,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["candidate_type", "dataset", "candidate_key"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def export_pair_judgments(
    *,
    session_name: str,
    judgments: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"session_name": str(session_name), "pair_judgments": judgments}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_pair_judgments(input_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    path = Path(input_path).resolve()
    if not path.exists():
        return "", []
    payload = json.loads(path.read_text(encoding="utf-8"))
    session_name = str(payload.get("session_name", "")).strip()
    raw_judgments = payload.get("pair_judgments", [])
    if not isinstance(raw_judgments, list):
        raise ValueError("pair_judgments must be a list")

    normalized: list[dict[str, Any]] = []
    for item in raw_judgments:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip().lower()
        if label and label not in SUPPORTED_PAIR_LABELS:
            continue
        normalized.append(
            {
                "judgment_id": str(item.get("judgment_id", "")).strip(),
                "dataset": str(item.get("dataset", "")).strip(),
                "candidate_type": str(item.get("candidate_type", "")).strip(),
                "candidate_key": str(item.get("candidate_key", "")).strip(),
                "pair_key": _judgment_pair_key(item),
                "image_id": str(item.get("image_id", "")).strip(),
                "neighbor_image_id": str(item.get("neighbor_image_id", "")).strip(),
                "base_cluster_left": int(item.get("base_cluster_left", -1)),
                "base_cluster_right": int(item.get("base_cluster_right", -1)),
                "xgb_same_identity_prob": round(float(item.get("xgb_same_identity_prob", 0.0)), 6),
                "ambiguity_score": round(float(item.get("ambiguity_score", 0.0)), 6),
                "label": label,
                "note": str(item.get("note", "")).strip(),
            }
        )
    normalized = sorted(
        normalized,
        key=lambda item: (
            str(item.get("candidate_type", "")),
            str(item.get("candidate_key", "")),
            str(item.get("pair_key", "")),
        ),
    )
    return session_name, normalized


def operations_to_dataframe(operations: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        operations,
        columns=[
            "operation_id",
            "dataset",
            "action",
            "anchor_image_id",
            "source_cluster_ids",
            "member_image_ids",
            "exclude_image_ids",
            "note",
        ],
    )


def build_operations_preview_json(rule_name: str, operations: list[dict[str, Any]]) -> str:
    payload = {"rule_name": rule_name, "operations": operations}
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _slug(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return value or "op"


def _next_operation_id(existing_operations: list[dict[str, Any]], prefix: str) -> str:
    prefix = _slug(prefix)
    existing_ids = {str(item.get("operation_id", "")) for item in existing_operations}
    if prefix not in existing_ids:
        return prefix
    suffix = 2
    while f"{prefix}_{suffix}" in existing_ids:
        suffix += 1
    return f"{prefix}_{suffix}"


def add_split_operation(
    operations: list[dict[str, Any]],
    *,
    dataset: str,
    cluster_id: str | int | None,
    anchor_image_id: str | None,
    member_image_ids: list[str] | tuple[str, ...] | None,
    note: str,
) -> list[dict[str, Any]]:
    next_ops = list(operations)
    payload: dict[str, Any] = {
        "operation_id": _next_operation_id(next_ops, f"split_{dataset}_{cluster_id or anchor_image_id or 'manual'}"),
        "dataset": str(dataset),
        "action": ACTION_SPLIT_TO_SINGLETONS,
        "anchor_image_id": str(anchor_image_id) if anchor_image_id else None,
        "note": str(note).strip(),
    }
    selected_members = [str(value) for value in (member_image_ids or []) if str(value)]
    if selected_members:
        payload["member_image_ids"] = selected_members
    elif cluster_id not in (None, ""):
        payload["source_cluster_ids"] = [int(cluster_id)]
    else:
        raise ValueError("Split operation needs either member_image_ids or source cluster id")
    next_ops.append(payload)
    return next_ops


def add_attach_operation(
    operations: list[dict[str, Any]],
    *,
    dataset: str,
    anchor_image_id: str,
    member_image_ids: list[str] | tuple[str, ...] | None,
    source_cluster_ids: list[int] | tuple[int, ...] | None,
    note: str,
) -> list[dict[str, Any]]:
    if not anchor_image_id:
        raise ValueError("Attach operation requires anchor_image_id")
    payload: dict[str, Any] = {
        "operation_id": _next_operation_id(next_ops := list(operations), f"attach_{dataset}_{anchor_image_id}"),
        "dataset": str(dataset),
        "action": ACTION_ATTACH_TO_ANCHOR,
        "anchor_image_id": str(anchor_image_id),
        "note": str(note).strip(),
    }
    selected_members = [str(value) for value in (member_image_ids or []) if str(value)]
    selected_clusters = [int(value) for value in (source_cluster_ids or [])]
    if selected_members:
        payload["member_image_ids"] = selected_members
    if selected_clusters:
        payload["source_cluster_ids"] = selected_clusters
    if not selected_members and not selected_clusters:
        raise ValueError("Attach operation needs member_image_ids or source_cluster_ids")
    next_ops.append(payload)
    return next_ops


def remove_last_operation(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not operations:
        return []
    return list(operations[:-1])


def remove_operation_at(operations: list[dict[str, Any]], index: int | None) -> list[dict[str, Any]]:
    if not operations:
        return []
    if index is None:
        return list(operations)
    row_index = int(index)
    if row_index < 0 or row_index >= len(operations):
        return list(operations)
    return [item for idx, item in enumerate(operations) if idx != row_index]


def clear_operations() -> list[dict[str, Any]]:
    return []


def export_operations_spec(
    *,
    rule_name: str,
    operations: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"rule_name": str(rule_name), "operations": operations}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_overlay_command(
    *,
    repo_root: Path,
    base_submission_dir: str | Path,
    spec_path: str | Path,
    output_dir: str | Path,
    submission_description: str,
) -> str:
    return (
        "python scripts/build_manual_cluster_overlay_submission.py "
        f"--base-submission-dir {Path(base_submission_dir)} "
        f"--overlay-spec {Path(spec_path)} "
        f"--output-dir {Path(output_dir)} "
        f"--submission-description \"{submission_description}\""
    )
