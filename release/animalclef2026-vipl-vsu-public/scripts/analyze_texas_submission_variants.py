#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"


@dataclass(frozen=True)
class ClusterPairStats:
    pair_precision_vs_anchor: float
    pair_recall_vs_anchor: float
    pair_f1_vs_anchor: float
    changed_pair_fraction: float
    anchor_same_pair_count: int
    variant_same_pair_count: int
    shared_same_pair_count: int


def _load_texas_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    texas_df = df[df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if texas_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows found in {path}")
    required_columns = {"image_id", "path", "pred_cluster_id"}
    missing = required_columns - set(texas_df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    texas_df["pred_cluster_id"] = texas_df["pred_cluster_id"].astype(int)
    return texas_df


def _load_pseudo_assignments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    df = df[df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if "is_seed" not in df.columns or "pseudo_label_index" not in df.columns:
        raise ValueError("pseudo_assignments must contain is_seed and pseudo_label_index")
    df["is_seed"] = df["is_seed"].fillna(False).astype(bool)
    df["pseudo_label_index"] = df["pseudo_label_index"].fillna(-1).astype(int)
    return df


def _load_candidate_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    df["neighbor_image_id"] = df["neighbor_image_id"].astype(str)
    return df


def _align_frame(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, name: str) -> pd.DataFrame:
    lookup = candidate_df.set_index("image_id", drop=False)
    missing = [image_id for image_id in reference_df["image_id"] if image_id not in lookup.index]
    if missing:
        raise ValueError(f"{name} is missing image_ids, examples: {missing[:5]}")
    aligned = lookup.loc[reference_df["image_id"].tolist()].reset_index(drop=True).copy()
    return aligned


def _pair_keys_from_clusters(labels: np.ndarray) -> set[tuple[int, int]]:
    by_cluster: dict[int, list[int]] = {}
    for index, label in enumerate(labels.tolist()):
        by_cluster.setdefault(int(label), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        for left, right in combinations(members, 2):
            pairs.add((left, right))
    return pairs


def _cluster_pair_stats(anchor_labels: np.ndarray, variant_labels: np.ndarray) -> ClusterPairStats:
    anchor_pairs = _pair_keys_from_clusters(anchor_labels)
    variant_pairs = _pair_keys_from_clusters(variant_labels)
    shared_pairs = anchor_pairs & variant_pairs
    precision = len(shared_pairs) / len(variant_pairs) if variant_pairs else 0.0
    recall = len(shared_pairs) / len(anchor_pairs) if anchor_pairs else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    n = len(anchor_labels)
    total_pairs = n * (n - 1) // 2
    changed = 0
    for left in range(n):
        left_anchor = anchor_labels[left] == anchor_labels[left + 1 :]
        left_variant = variant_labels[left] == variant_labels[left + 1 :]
        changed += int(np.count_nonzero(left_anchor != left_variant))
    changed_pair_fraction = changed / total_pairs if total_pairs else 0.0
    return ClusterPairStats(
        pair_precision_vs_anchor=round(float(precision), 6),
        pair_recall_vs_anchor=round(float(recall), 6),
        pair_f1_vs_anchor=round(float(f1), 6),
        changed_pair_fraction=round(float(changed_pair_fraction), 6),
        anchor_same_pair_count=int(len(anchor_pairs)),
        variant_same_pair_count=int(len(variant_pairs)),
        shared_same_pair_count=int(len(shared_pairs)),
    )


def _pair_set_from_image_ids(reference_index: dict[str, int], rows: Iterable[tuple[str, str]]) -> set[tuple[int, int]]:
    pair_set: set[tuple[int, int]] = set()
    for left_id, right_id in rows:
        if left_id not in reference_index or right_id not in reference_index:
            continue
        left = reference_index[left_id]
        right = reference_index[right_id]
        if left == right:
            continue
        pair_set.add((left, right) if left < right else (right, left))
    return pair_set


def _pair_keep_ratio(labels: np.ndarray, pair_set: set[tuple[int, int]]) -> float:
    if not pair_set:
        return 0.0
    keep = sum(1 for left, right in pair_set if labels[left] == labels[right])
    return round(float(keep / len(pair_set)), 6)


def _build_seed_pair_set(pseudo_df: pd.DataFrame, image_index: dict[str, int]) -> set[tuple[int, int]]:
    pair_set: set[tuple[int, int]] = set()
    seed_df = pseudo_df[pseudo_df["is_seed"] & (pseudo_df["pseudo_label_index"] >= 0)].copy()
    for _, frame in seed_df.groupby("pseudo_label_index"):
        indices = [image_index[str(image_id)] for image_id in frame["image_id"].tolist() if str(image_id) in image_index]
        for left, right in combinations(sorted(indices), 2):
            pair_set.add((left, right))
    return pair_set


def _cluster_transition_table(
    reference_df: pd.DataFrame,
    anchor_labels: np.ndarray,
    variant_labels: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    anchor_df = reference_df[["image_id", "path", "is_seed", "pseudo_label_index"]].copy()
    anchor_df["anchor_cluster_id"] = anchor_labels
    anchor_df["variant_cluster_id"] = variant_labels
    anchor_sizes = anchor_df["anchor_cluster_id"].value_counts().to_dict()
    variant_sizes = anchor_df["variant_cluster_id"].value_counts().to_dict()
    for anchor_cluster_id, frame in anchor_df.groupby("anchor_cluster_id"):
        variant_counts = frame["variant_cluster_id"].value_counts().sort_values(ascending=False)
        split_count = int(variant_counts.size)
        main_variant_cluster_id = int(variant_counts.index[0])
        main_overlap = int(variant_counts.iloc[0])
        main_variant_size = int(variant_sizes[main_variant_cluster_id])
        merged_extra = max(0, main_variant_size - main_overlap)
        if split_count == 1 and merged_extra == 0:
            transition = "exact"
        elif split_count > 1 and merged_extra == 0:
            transition = "split_only"
        elif split_count == 1 and merged_extra > 0:
            transition = "merge_only"
        else:
            transition = "split_and_merge"
        seed_members = int(frame["is_seed"].sum())
        rows.append(
            {
                "anchor_cluster_id": int(anchor_cluster_id),
                "anchor_cluster_size": int(anchor_sizes[int(anchor_cluster_id)]),
                "variant_cluster_count": split_count,
                "main_variant_cluster_id": main_variant_cluster_id,
                "main_overlap_size": main_overlap,
                "main_variant_cluster_size": main_variant_size,
                "merged_extra_members": merged_extra,
                "seed_members": seed_members,
                "transition_type": transition,
                "variant_cluster_ids": ",".join(str(int(value)) for value in variant_counts.index.tolist()),
                "top_member_image_ids": ",".join(frame["image_id"].astype(str).head(5).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["transition_type", "anchor_cluster_size", "merged_extra_members", "variant_cluster_count"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)


def _image_change_table(
    reference_df: pd.DataFrame,
    anchor_labels: np.ndarray,
    variant_labels: np.ndarray,
    candidate_pair_set: set[tuple[int, int]],
    mutual_pair_set: set[tuple[int, int]],
) -> pd.DataFrame:
    n = len(reference_df)
    anchor_members = [set(np.where(anchor_labels == anchor_labels[index])[0].tolist()) - {index} for index in range(n)]
    variant_members = [set(np.where(variant_labels == variant_labels[index])[0].tolist()) - {index} for index in range(n)]
    rows: list[dict[str, object]] = []
    for index, row in enumerate(reference_df.itertuples(index=False)):
        removed = sorted(anchor_members[index] - variant_members[index])
        added = sorted(variant_members[index] - anchor_members[index])
        candidate_partners = {
            right if left == index else left
            for left, right in candidate_pair_set
            if left == index or right == index
        }
        mutual_partners = {
            right if left == index else left
            for left, right in mutual_pair_set
            if left == index or right == index
        }
        broken_candidate = sorted(partner for partner in candidate_partners if partner in removed)
        added_candidate = sorted(partner for partner in candidate_partners if partner in added)
        broken_mutual = sorted(partner for partner in mutual_partners if partner in removed)
        rows.append(
            {
                "image_id": str(row.image_id),
                "path": str(row.path),
                "is_seed": bool(row.is_seed),
                "pseudo_label_index": int(row.pseudo_label_index),
                "anchor_cluster_id": int(anchor_labels[index]),
                "variant_cluster_id": int(variant_labels[index]),
                "anchor_cluster_size": int(len(anchor_members[index]) + 1),
                "variant_cluster_size": int(len(variant_members[index]) + 1),
                "removed_partner_count": int(len(removed)),
                "added_partner_count": int(len(added)),
                "changed_partner_count": int(len(removed) + len(added)),
                "broken_candidate_pair_count": int(len(broken_candidate)),
                "added_candidate_pair_count": int(len(added_candidate)),
                "broken_mutual_pair_count": int(len(broken_mutual)),
                "removed_partner_image_ids": ",".join(reference_df.iloc[removed]["image_id"].astype(str).tolist()[:12]),
                "added_partner_image_ids": ",".join(reference_df.iloc[added]["image_id"].astype(str).tolist()[:12]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["broken_candidate_pair_count", "broken_mutual_pair_count", "changed_partner_count", "anchor_cluster_size"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _merge_events_table(reference_df: pd.DataFrame, anchor_labels: np.ndarray, variant_labels: np.ndarray) -> pd.DataFrame:
    frame = reference_df[["image_id", "path", "is_seed"]].copy()
    frame["anchor_cluster_id"] = anchor_labels
    frame["variant_cluster_id"] = variant_labels
    anchor_sizes = frame["anchor_cluster_id"].value_counts().to_dict()
    rows: list[dict[str, object]] = []
    for variant_cluster_id, variant_frame in frame.groupby("variant_cluster_id"):
        anchor_counts = variant_frame["anchor_cluster_id"].value_counts().sort_values(ascending=False)
        if anchor_counts.size <= 1:
            continue
        contributing = [
            f"{int(anchor_id)}:{int(size)}"
            for anchor_id, size in anchor_counts.items()
        ]
        rows.append(
            {
                "variant_cluster_id": int(variant_cluster_id),
                "variant_cluster_size": int(len(variant_frame)),
                "source_anchor_cluster_count": int(anchor_counts.size),
                "source_anchor_clusters": ",".join(contributing),
                "source_anchor_sizes": ",".join(str(int(anchor_sizes[int(anchor_id)])) for anchor_id in anchor_counts.index.tolist()),
                "seed_members": int(variant_frame["is_seed"].sum()),
                "image_ids": ",".join(variant_frame["image_id"].astype(str).head(8).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["source_anchor_cluster_count", "variant_cluster_size", "seed_members"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _build_summary_row(
    *,
    variant_name: str,
    variant_public_score: float | None,
    anchor_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    pair_stats: ClusterPairStats,
    seed_pair_set: set[tuple[int, int]],
    candidate_pair_set: set[tuple[int, int]],
    mutual_pair_set: set[tuple[int, int]],
    transition_df: pd.DataFrame,
) -> dict[str, object]:
    anchor_labels = anchor_df["pred_cluster_id"].to_numpy(dtype=int)
    variant_labels = variant_df["pred_cluster_id"].to_numpy(dtype=int)
    anchor_cluster_sizes = anchor_df["pred_cluster_id"].value_counts()
    variant_cluster_sizes = variant_df["pred_cluster_id"].value_counts()
    exact_count = int((transition_df["transition_type"] == "exact").sum())
    split_only_count = int((transition_df["transition_type"] == "split_only").sum())
    merge_only_count = int((transition_df["transition_type"] == "merge_only").sum())
    split_merge_count = int((transition_df["transition_type"] == "split_and_merge").sum())
    return {
        "variant_name": variant_name,
        "public_score": variant_public_score,
        "anchor_clusters": int(anchor_cluster_sizes.size),
        "variant_clusters": int(variant_cluster_sizes.size),
        "cluster_delta_vs_anchor": int(variant_cluster_sizes.size - anchor_cluster_sizes.size),
        "anchor_largest_cluster_size": int(anchor_cluster_sizes.max()),
        "variant_largest_cluster_size": int(variant_cluster_sizes.max()),
        "anchor_singleton_ratio": round(float((anchor_cluster_sizes == 1).mean()), 6),
        "variant_singleton_ratio": round(float((variant_cluster_sizes == 1).mean()), 6),
        "pair_precision_vs_anchor": pair_stats.pair_precision_vs_anchor,
        "pair_recall_vs_anchor": pair_stats.pair_recall_vs_anchor,
        "pair_f1_vs_anchor": pair_stats.pair_f1_vs_anchor,
        "changed_pair_fraction": pair_stats.changed_pair_fraction,
        "seed_pair_keep_ratio": _pair_keep_ratio(variant_labels, seed_pair_set),
        "candidate_pair_keep_ratio": _pair_keep_ratio(variant_labels, candidate_pair_set),
        "mutual_topk_pair_keep_ratio": _pair_keep_ratio(variant_labels, mutual_pair_set),
        "exact_anchor_clusters": exact_count,
        "split_only_anchor_clusters": split_only_count,
        "merge_only_anchor_clusters": merge_only_count,
        "split_and_merge_anchor_clusters": split_merge_count,
    }


def _write_report(
    output_path: Path,
    *,
    anchor_name: str,
    anchor_public_score: float | None,
    summary_df: pd.DataFrame,
    config: dict[str, object],
) -> None:
    table_df = summary_df.copy().fillna("")
    headers = [str(column) for column in table_df.columns.tolist()]
    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_df.itertuples(index=False):
        table_lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines = [
        "# Texas Submission Variant Audit",
        "",
        "## Anchor",
        "",
        f"- Anchor route: `{anchor_name}`",
        f"- Anchor public score: `{anchor_public_score}`" if anchor_public_score is not None else "- Anchor public score: `N/A`",
        "",
        "## Variant Summary",
        "",
        *table_lines,
        "",
        "## Reading Notes",
        "",
        "- `pair_precision_vs_anchor`: 变体里被并到一起的图片对，有多少原本就在 anchor 同簇；越低说明新增误并越多。",
        "- `pair_recall_vs_anchor`: anchor 原本同簇的图片对，有多少还被变体保住；越低说明切碎更严重。",
        "- `changed_pair_fraction`: 所有图片对里，anchor 与变体对“是否同簇”的判断不同的比例；小比例也可能对应高价值样本被改坏。",
        "- `exact/split/merge`: 以 anchor 的簇为基准，判断变体是原样保留、切碎、并错，还是同时切碎又并错。",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Texas-only submission variants against an anchor route.")
    parser.add_argument("--anchor-name", type=str, required=True)
    parser.add_argument("--anchor-predictions", type=Path, required=True)
    parser.add_argument("--anchor-public-score", type=float)
    parser.add_argument("--variant-name", type=str, action="append", required=True)
    parser.add_argument("--variant-predictions", type=Path, action="append", required=True)
    parser.add_argument("--variant-public-score", type=float, action="append")
    parser.add_argument("--pseudo-assignments", type=Path, required=True)
    parser.add_argument("--candidate-pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if len(args.variant_name) != len(args.variant_predictions):
        raise ValueError("variant-name and variant-predictions must have the same length")
    if args.variant_public_score is not None and len(args.variant_public_score) > len(args.variant_predictions):
        raise ValueError("variant-public-score cannot be longer than the number of variants")

    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    anchor_df = _load_texas_predictions(args.anchor_predictions.resolve())
    pseudo_df = _align_frame(anchor_df, _load_pseudo_assignments(args.pseudo_assignments.resolve()), "pseudo_assignments")
    anchor_df = anchor_df.drop(columns=["is_seed", "pseudo_label_index"], errors="ignore").merge(
        pseudo_df[["image_id", "is_seed", "pseudo_label_index"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    anchor_df["is_seed"] = anchor_df["is_seed"].fillna(False).astype(bool)
    anchor_df["pseudo_label_index"] = anchor_df["pseudo_label_index"].fillna(-1).astype(int)
    image_index = {image_id: index for index, image_id in enumerate(anchor_df["image_id"].tolist())}
    candidate_df = _load_candidate_pairs(args.candidate_pairs.resolve())
    candidate_pair_set = _pair_set_from_image_ids(
        image_index,
        candidate_df[["image_id", "neighbor_image_id"]].itertuples(index=False, name=None),
    )
    mutual_pair_set = _pair_set_from_image_ids(
        image_index,
        candidate_df.loc[candidate_df["mutual_topk_all_routes"].fillna(False).astype(bool), ["image_id", "neighbor_image_id"]].itertuples(index=False, name=None),
    )
    seed_pair_set = _build_seed_pair_set(anchor_df, image_index)

    anchor_labels = anchor_df["pred_cluster_id"].to_numpy(dtype=int)
    summary_rows: list[dict[str, object]] = []
    public_scores = list(args.variant_public_score or [])
    if len(public_scores) < len(args.variant_predictions):
        public_scores.extend([None] * (len(args.variant_predictions) - len(public_scores)))
    for variant_name, variant_path, variant_public_score in zip(
        args.variant_name,
        args.variant_predictions,
        public_scores,
        strict=True,
    ):
        variant_df = _align_frame(anchor_df, _load_texas_predictions(variant_path.resolve()), variant_name)
        variant_labels = variant_df["pred_cluster_id"].to_numpy(dtype=int)
        pair_stats = _cluster_pair_stats(anchor_labels, variant_labels)
        transition_df = _cluster_transition_table(anchor_df, anchor_labels, variant_labels)
        image_change_df = _image_change_table(anchor_df, anchor_labels, variant_labels, candidate_pair_set, mutual_pair_set)
        merge_df = _merge_events_table(anchor_df, anchor_labels, variant_labels)
        transition_df.to_csv(tables_dir / f"{variant_name}_cluster_transitions_v1.csv", index=False)
        image_change_df.to_csv(tables_dir / f"{variant_name}_image_changes_v1.csv", index=False)
        merge_df.to_csv(tables_dir / f"{variant_name}_merge_events_v1.csv", index=False)
        summary_rows.append(
            _build_summary_row(
                variant_name=variant_name,
                variant_public_score=variant_public_score,
                anchor_df=anchor_df,
                variant_df=variant_df,
                pair_stats=pair_stats,
                seed_pair_set=seed_pair_set,
                candidate_pair_set=candidate_pair_set,
                mutual_pair_set=mutual_pair_set,
                transition_df=transition_df,
            )
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("variant_name").reset_index(drop=True)
    summary_df.to_csv(tables_dir / "variant_summary_v1.csv", index=False)
    config = {
        "anchor_name": args.anchor_name,
        "anchor_predictions": str(args.anchor_predictions.resolve()),
        "anchor_public_score": args.anchor_public_score,
        "variant_names": args.variant_name,
        "variant_predictions": [str(path.resolve()) for path in args.variant_predictions],
        "variant_public_scores": public_scores,
        "pseudo_assignments": str(args.pseudo_assignments.resolve()),
        "candidate_pairs": str(args.candidate_pairs.resolve()),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(
        reports_dir / "summary.md",
        anchor_name=args.anchor_name,
        anchor_public_score=args.anchor_public_score,
        summary_df=summary_df,
        config=config,
    )
    print(f"[texas_audit] summary: {reports_dir / 'summary.md'}")
    print(f"[texas_audit] variant_summary: {tables_dir / 'variant_summary_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
