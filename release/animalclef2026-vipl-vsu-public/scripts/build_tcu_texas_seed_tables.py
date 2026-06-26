from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCORE_BINS = [-np.inf, 4000, 8000, 12000, 20000, np.inf]
SCORE_BIN_LABELS = ["<=4k", "4k-8k", "8k-12k", "12k-20k", ">20k"]

GAP_BINS = [-np.inf, 500, 1000, 2000, 5000, np.inf]
GAP_BIN_LABELS = ["<=500", "500-1k", "1k-2k", "2k-5k", ">5k"]

UNIQUE_SCORE_BINS = [-np.inf, 4000, 6000, 8000, 10000, 12000, 16000, 20000, np.inf]
UNIQUE_SCORE_BIN_LABELS = ["<=4k", "4k-6k", "6k-8k", "8k-10k", "10k-12k", "12k-16k", "16k-20k", ">20k"]

UNIQUE_GAP_BINS = [-np.inf, 500, 1000, 2000, 4000, 8000, np.inf]
UNIQUE_GAP_BIN_LABELS = ["<=500", "500-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trusted pairs and pseudo-seed candidate tables from manual TCU Texas annotations.")
    parser.add_argument(
        "--annotation-csv",
        type=Path,
        default=Path("tcu_texas_manual_annotations.csv"),
        help="Manual annotation export from the HTML review board.",
    )
    parser.add_argument(
        "--origin-view-csv",
        type=Path,
        default=Path("artifacts/analysis/tcu_texas_hotspotter_origin_view_v1/6._Output_from_HotSpotter_origin_view.csv"),
        help="Expanded origin-view HotSpotter table with rank-10 candidates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/analysis/tcu_texas_manual_annotation_analysis_v1"),
        help="Directory for trusted pairs and pseudo-seed outputs.",
    )
    return parser.parse_args()


def parse_pipe_ints(value: object) -> list[int]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    items: list[int] = []
    for token in text.split("|"):
        token = token.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except ValueError:
            continue
    return items


def parse_pipe_strings(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip() for token in text.split("|") if token.strip()]


def parse_score(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        if "+" in text and "e" not in text.lower():
            left, right = text.split("+", 1)
            try:
                return float(left) * (10 ** int(right))
            except ValueError:
                return np.nan
        return np.nan


def load_inputs(annotation_csv: Path, origin_view_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ann = pd.read_csv(annotation_csv)
    orig = pd.read_csv(origin_view_csv)
    ann["selected_rank_list"] = ann["selected_ranks"].fillna("").map(parse_pipe_ints)
    ann["selected_candidate_chip_list"] = ann["selected_candidate_chips"].fillna("").map(parse_pipe_strings)
    ann["selected_candidate_name_list"] = ann["selected_candidate_names"].fillna("").map(parse_pipe_strings)
    ann["manual_match"] = ann["selected_rank_list"].map(lambda items: len(items) > 0)
    ann["selected_count"] = ann["selected_rank_list"].map(len)
    ann["best_selected_rank"] = ann["selected_rank_list"].map(lambda items: min(items) if items else np.nan)
    ann["reviewed"] = ann["reviewed"].astype(bool)
    ann["no_match"] = ann["no_match"].astype(bool)
    ann["role_clean"] = ann["role"].astype(str).str.strip()
    ann["result_clean"] = ann["result"].astype(str).str.strip()

    for rank_idx in range(1, 11):
        score_col = f"Rank {rank_idx} - Score"
        if score_col in orig.columns:
            orig[f"rank_{rank_idx}_score"] = orig[score_col].map(parse_score)
        elif f"Rank {rank_idx}" in orig.columns:
            orig[f"rank_{rank_idx}_score"] = orig[f"Rank {rank_idx}"].map(parse_score)

    return ann, orig


def build_analysis_table(ann: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    keep_columns = [
        "query_chip_id",
        "query_origin_image_name",
        "query_origin_label_hl",
        "role_image",
        "Query Result",
    ]
    for rank_idx in range(1, 11):
        keep_columns.extend(
            [
                f"rank_{rank_idx}_id",
                f"rank_{rank_idx}_origin_image_name",
                f"rank_{rank_idx}_origin_label_hl",
                f"rank_{rank_idx}_score",
            ]
        )
    merged = ann.merge(orig[keep_columns], on="query_chip_id", how="left")
    merged["top1_score"] = merged["rank_1_score"]
    merged["top2_score"] = merged["rank_2_score"]
    merged["top1_gap"] = merged["rank_1_score"] - merged["rank_2_score"]
    merged["top1_ratio"] = merged["rank_1_score"] / merged["rank_2_score"]
    merged["top1_is_selected"] = merged["selected_rank_list"].map(lambda items: 1 in items)

    best_selected_scores: list[float] = []
    for _, row in merged.iterrows():
        values = []
        for rank_idx in row["selected_rank_list"]:
            values.append(row.get(f"rank_{rank_idx}_score", np.nan))
        values = [value for value in values if pd.notna(value)]
        best_selected_scores.append(max(values) if values else np.nan)
    merged["best_selected_score"] = best_selected_scores
    return merged


def build_bin_match_rates(
    df: pd.DataFrame,
    value_column: str,
    label_column: str,
    bins: list[float],
    labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out[label_column] = pd.cut(out[value_column], bins=bins, labels=labels)
    reviewed = out[out["reviewed"]].copy()
    rates = (
        reviewed.groupby(label_column, observed=False)["manual_match"]
        .agg(["count", "mean"])
        .rename(columns={"count": "reviewed_count", "mean": "manual_match_rate"})
        .reset_index()
    )
    return out, rates


def attach_match_rates(
    base_df: pd.DataFrame,
    value_column: str,
    label_column: str,
    bins: list[float],
    labels: list[str],
    prefix: str,
) -> pd.DataFrame:
    framed, rates = build_bin_match_rates(
        df=base_df,
        value_column=value_column,
        label_column=label_column,
        bins=bins,
        labels=labels,
    )
    rates = rates.rename(
        columns={
            "manual_match_rate": f"{prefix}_manual_match_rate",
            "reviewed_count": f"{prefix}_reviewed_count",
        }
    )
    return framed.merge(rates, on=label_column, how="left")


def build_trusted_pairs(analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in analysis_df.iterrows():
        query_chip = int(row["query_chip_id"])
        query_image = row["query_origin_image_name"]
        query_label_hl = row["query_origin_label_hl"]
        for rank_idx in row["selected_rank_list"]:
            candidate_chip = row.get(f"rank_{rank_idx}_id", np.nan)
            candidate_name = row.get(f"rank_{rank_idx}_origin_image_name", "")
            candidate_label_hl = row.get(f"rank_{rank_idx}_origin_label_hl", "")
            selected_score = row.get(f"rank_{rank_idx}_score", np.nan)
            if pd.isna(candidate_chip):
                continue
            candidate_chip_int = int(candidate_chip)
            pair_min, pair_max = sorted([query_chip, candidate_chip_int])
            rows.append(
                {
                    "pair_id": f"{pair_min:03d}_{pair_max:03d}",
                    "chip_a": pair_min,
                    "chip_b": pair_max,
                    "query_chip_id": query_chip,
                    "candidate_chip_id": candidate_chip_int,
                    "query_origin_image_name": query_image,
                    "candidate_origin_image_name": candidate_name,
                    "query_origin_label_hl": query_label_hl,
                    "candidate_origin_label_hl": candidate_label_hl,
                    "selected_rank": rank_idx,
                    "selected_score": selected_score,
                    "query_top1_score": row["top1_score"],
                    "query_top1_gap": row["top1_gap"],
                    "role": row["role_clean"],
                    "result": row["result_clean"],
                    "group": row["group"],
                }
            )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs

    def join_unique(values: Iterable[object]) -> str:
        cleaned = []
        for value in values:
            text = str(value).strip()
            if not text or text == "nan":
                continue
            cleaned.append(text)
        return "|".join(sorted(set(cleaned)))

    aggregated = (
        pairs.groupby(["pair_id", "chip_a", "chip_b"], as_index=False)
        .agg(
            support_count=("pair_id", "size"),
            query_chip_ids=("query_chip_id", join_unique),
            candidate_chip_ids=("candidate_chip_id", join_unique),
            query_origin_image_names=("query_origin_image_name", join_unique),
            candidate_origin_image_names=("candidate_origin_image_name", join_unique),
            query_origin_label_hls=("query_origin_label_hl", join_unique),
            candidate_origin_label_hls=("candidate_origin_label_hl", join_unique),
            groups=("group", join_unique),
            roles=("role", join_unique),
            results=("result", join_unique),
            selected_ranks=("selected_rank", lambda values: "|".join(str(int(v)) for v in sorted(values))),
            best_selected_rank=("selected_rank", "min"),
            best_selected_score=("selected_score", "max"),
            median_selected_score=("selected_score", "median"),
            max_query_top1_score=("query_top1_score", "max"),
            median_query_top1_gap=("query_top1_gap", "median"),
        )
        .sort_values(["support_count", "best_selected_score", "best_selected_rank"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    return aggregated


def build_pseudo_seed_candidates(analysis_df: pd.DataFrame) -> pd.DataFrame:
    candidates = analysis_df.copy()
    candidates = attach_match_rates(
        base_df=candidates,
        value_column="top1_score",
        label_column="score_bin_all",
        bins=SCORE_BINS,
        labels=SCORE_BIN_LABELS,
        prefix="score_bin_all",
    )
    candidates = attach_match_rates(
        base_df=candidates,
        value_column="top1_gap",
        label_column="gap_bin_all",
        bins=GAP_BINS,
        labels=GAP_BIN_LABELS,
        prefix="gap_bin_all",
    )

    unique_df = candidates[candidates["group"] == "unique"].copy()
    unique_df = attach_match_rates(
        base_df=unique_df,
        value_column="top1_score",
        label_column="score_bin_unique",
        bins=UNIQUE_SCORE_BINS,
        labels=UNIQUE_SCORE_BIN_LABELS,
        prefix="score_bin_unique",
    )
    unique_df = attach_match_rates(
        base_df=unique_df,
        value_column="top1_gap",
        label_column="gap_bin_unique",
        bins=UNIQUE_GAP_BINS,
        labels=UNIQUE_GAP_BIN_LABELS,
        prefix="gap_bin_unique",
    )
    unique_lookup = unique_df[
        [
            "query_chip_id",
            "score_bin_unique",
            "score_bin_unique_manual_match_rate",
            "score_bin_unique_reviewed_count",
            "gap_bin_unique",
            "gap_bin_unique_manual_match_rate",
            "gap_bin_unique_reviewed_count",
        ]
    ]
    candidates = candidates.merge(unique_lookup, on="query_chip_id", how="left")

    candidates["candidate_rank"] = 1
    candidates["candidate_chip_id"] = candidates["rank_1_id"]
    candidates["candidate_origin_image_name"] = candidates["rank_1_origin_image_name"]
    candidates["candidate_origin_label_hl"] = candidates["rank_1_origin_label_hl"]
    candidates["candidate_score"] = candidates["rank_1_score"]
    candidates["manual_label_top1"] = np.where(
        candidates["reviewed"],
        candidates["top1_is_selected"].astype(int),
        np.nan,
    )
    candidates["manual_label_any"] = np.where(
        candidates["reviewed"],
        candidates["manual_match"].astype(int),
        np.nan,
    )
    candidates["pseudo_seed_status"] = np.select(
        [
            candidates["manual_label_top1"] == 1,
            (candidates["reviewed"]) & (candidates["manual_label_top1"] == 0),
        ],
        [
            "trusted_top1_pair",
            "reviewed_not_top1_pair",
        ],
        default="unreviewed_or_unlabeled",
    )
    candidates["very_high_precision_rule_v1"] = candidates["top1_score"] > 20000
    candidates["high_gap_rule_v1"] = candidates["top1_gap"] > 5000

    candidate_columns = [
        "query_chip_id",
        "query_origin_image_name",
        "query_origin_label_hl",
        "role_clean",
        "result_clean",
        "group",
        "reviewed",
        "no_match",
        "manual_match",
        "manual_label_any",
        "top1_is_selected",
        "manual_label_top1",
        "selected_count",
        "best_selected_rank",
        "selected_ranks",
        "selected_candidate_chips",
        "selected_candidate_names",
        "candidate_rank",
        "candidate_chip_id",
        "candidate_origin_image_name",
        "candidate_origin_label_hl",
        "candidate_score",
        "top1_score",
        "top2_score",
        "top1_gap",
        "top1_ratio",
        "score_bin_all",
        "score_bin_all_manual_match_rate",
        "score_bin_all_reviewed_count",
        "gap_bin_all",
        "gap_bin_all_manual_match_rate",
        "gap_bin_all_reviewed_count",
        "score_bin_unique",
        "score_bin_unique_manual_match_rate",
        "score_bin_unique_reviewed_count",
        "gap_bin_unique",
        "gap_bin_unique_manual_match_rate",
        "gap_bin_unique_reviewed_count",
        "very_high_precision_rule_v1",
        "high_gap_rule_v1",
        "pseudo_seed_status",
    ]
    available_columns = [column for column in candidate_columns if column in candidates.columns]
    candidates = candidates[available_columns].copy()
    candidates = candidates.sort_values(
        by=["very_high_precision_rule_v1", "high_gap_rule_v1", "top1_score", "top1_gap"],
        ascending=[False, False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    return candidates


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ann, orig = load_inputs(annotation_csv=args.annotation_csv, origin_view_csv=args.origin_view_csv)
    analysis_df = build_analysis_table(ann=ann, orig=orig)

    trusted_pairs = build_trusted_pairs(analysis_df)
    trusted_pairs.to_csv(args.output_dir / "trusted_pairs_v1.csv", index=False)

    pseudo_seed_candidates = build_pseudo_seed_candidates(analysis_df)
    pseudo_seed_candidates.to_csv(args.output_dir / "pseudo_seed_candidates_v1.csv", index=False)

    analysis_df.to_csv(args.output_dir / "manual_annotation_analysis_table.csv", index=False)
    print(args.output_dir / "trusted_pairs_v1.csv")
    print(args.output_dir / "pseudo_seed_candidates_v1.csv")


if __name__ == "__main__":
    main()
