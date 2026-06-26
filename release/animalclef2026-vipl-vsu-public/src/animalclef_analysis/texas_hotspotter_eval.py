from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN
from .local_matching import (
    HotspotterConfig,
    create_match_board,
    extract_hesaff_features_with_report,
    prescore_results_to_dataframe,
    query_hotspotter_all,
    rank_results_to_dataframe,
)


DEFAULT_TCU_WARMUP_MANIFEST_PATH = Path(
    "artifacts/training/cache/tcu_texas_warmup_manifest_v1/tables/tcu_texas_warmup_manifest_v1.csv"
)
DEFAULT_TEXAS_HOTSPOTTER_EVAL_OUTPUT_DIR = Path("artifacts/analysis/tcu_texas_hotspotter_eval_v1")
DEFAULT_TEXAS_HOTSPOTTER_CACHE_DIR = Path("artifacts/cache/hotspotter_features")


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def resolve_input_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def load_labeled_manifest(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path).copy()
    required_columns = {"image_id", "identity", PATH_COLUMN}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"{manifest_path} is missing required columns: {missing}")
    df["image_id"] = df["image_id"].astype(str)
    df["identity"] = df["identity"].fillna("").astype(str)
    df[PATH_COLUMN] = df[PATH_COLUMN].fillna("").astype(str)
    df = df[df["identity"].ne("") & df[PATH_COLUMN].ne("")].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No labeled rows with valid paths in {manifest_path}")
    return df


def _effective_recall_at_k(ranking_df: pd.DataFrame, labels_df: pd.DataFrame, k: int) -> float:
    label_counts = labels_df["identity"].value_counts()
    valid_ids = set(label_counts[label_counts > 1].index.tolist())
    if not valid_ids:
        return 0.0
    valid_queries = labels_df[labels_df["identity"].isin(valid_ids)][["image_id", "identity"]].copy()
    merged = valid_queries.merge(
        ranking_df[ranking_df["rank"].astype(int) <= int(k)][["image_id", "neighbor_image_id", "rank"]],
        on="image_id",
        how="left",
    )
    neighbor_lookup = labels_df[["image_id", "identity"]].rename(
        columns={"image_id": "neighbor_image_id", "identity": "neighbor_identity"}
    )
    merged = merged.merge(neighbor_lookup, on="neighbor_image_id", how="left")
    hit_df = (
        merged.assign(is_hit=merged["identity"].eq(merged["neighbor_identity"]))
        .groupby("image_id", as_index=False)["is_hit"]
        .max()
    )
    return round(float(hit_df["is_hit"].mean()), 6) if not hit_df.empty else 0.0


def _first_hit_rank_df(ranking_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    neighbor_lookup = labels_df[["image_id", "identity"]].rename(
        columns={"image_id": "neighbor_image_id", "identity": "neighbor_identity"}
    )
    merged = ranking_df.merge(
        labels_df[["image_id", "identity"]],
        on="image_id",
        how="left",
    ).merge(
        neighbor_lookup,
        on="neighbor_image_id",
        how="left",
    )
    merged["same_identity"] = merged["identity"].eq(merged["neighbor_identity"])
    value_columns = [column for column in ["local_score", "inliers", "local_prescore", "good_matches"] if column in merged.columns]
    hit_rows = (
        merged[merged["same_identity"]]
        .sort_values(["image_id", "rank"], ascending=[True, True])
        .drop_duplicates(subset=["image_id"], keep="first")
        .loc[:, ["image_id", "rank", "neighbor_image_id", *value_columns]]
        .rename(columns={"rank": "first_hit_rank"})
    )
    counts = labels_df["identity"].value_counts()
    valid_queries = labels_df[labels_df["identity"].isin(counts[counts > 1].index)].copy()
    return valid_queries[["image_id", "identity"]].merge(hit_rows, on="image_id", how="left")


def _query_rank_lookup(ranking_df: pd.DataFrame, rank_column: str) -> pd.DataFrame:
    frame = ranking_df.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)
    return frame[["image_id", "neighbor_image_id", "rank"]].rename(columns={"rank": rank_column})


def _build_candidate_lookup(query_results) -> dict[tuple[int, int], object]:
    lookup: dict[tuple[int, int], object] = {}
    for query_result in query_results:
        for candidate in query_result.candidates:
            lookup[(int(query_result.query_index), int(candidate.candidate_index))] = candidate
    return lookup


def _build_qualitative_rows(
    *,
    labeled_df: pd.DataFrame,
    prescore_ranking_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    query_results,
    limit: int = 8,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    label_lookup = labeled_df[["image_id", "identity"]].rename(columns={"identity": "neighbor_identity"})
    pair_df = ranking_df.merge(
        labeled_df[["image_id", "identity"]],
        on="image_id",
        how="left",
    ).merge(
        label_lookup.rename(columns={"image_id": "neighbor_image_id"}),
        on="neighbor_image_id",
        how="left",
    )
    pair_df["same_identity"] = pair_df["identity"].eq(pair_df["neighbor_identity"])
    pre_rank_df = _query_rank_lookup(prescore_ranking_df, "pre_rank")
    post_rank_df = _query_rank_lookup(ranking_df, "post_rank")
    pair_df = pair_df.merge(pre_rank_df, on=["image_id", "neighbor_image_id"], how="left")
    pair_df = pair_df.merge(post_rank_df, on=["image_id", "neighbor_image_id"], how="left")
    candidate_lookup = _build_candidate_lookup(query_results)

    def to_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for row in frame.itertuples(index=False):
            key = (int(row.left_index), int(row.right_index))
            candidate = candidate_lookup.get(key)
            if candidate is None or len(candidate.fm) == 0:
                continue
            rows.append(
                {
                    "image_id": str(row.image_id),
                    "neighbor_image_id": str(row.neighbor_image_id),
                    "identity": str(row.identity),
                    "neighbor_identity": str(row.neighbor_identity),
                    "left_index": int(row.left_index),
                    "right_index": int(row.right_index),
                    "pre_rank": "" if pd.isna(row.pre_rank) else int(row.pre_rank),
                    "post_rank": "" if pd.isna(row.post_rank) else int(row.post_rank),
                    "local_prescore": float(row.local_prescore),
                    "local_score": float(row.local_score),
                    "inliers": int(row.inliers),
                    "fm": candidate.fm.copy(),
                    "fs": candidate.fs.copy(),
                }
            )
        return rows

    true_positive_rows = to_rows(
        pair_df[pair_df["same_identity"]]
        .sort_values(["inliers", "local_score"], ascending=[False, False])
        .head(int(limit))
        .reset_index(drop=True)
    )
    false_positive_rows = to_rows(
        pair_df[~pair_df["same_identity"]]
        .sort_values(["inliers", "local_score"], ascending=[False, False])
        .head(int(limit))
        .reset_index(drop=True)
    )
    return true_positive_rows, false_positive_rows


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def run_texas_hotspotter_eval(
    *,
    repo_root: Path,
    manifest_path: Path = DEFAULT_TCU_WARMUP_MANIFEST_PATH,
    output_dir: Path = DEFAULT_TEXAS_HOTSPOTTER_EVAL_OUTPUT_DIR,
    cache_dir: Path = DEFAULT_TEXAS_HOTSPOTTER_CACHE_DIR,
    top_k: int = 10,
    config: HotspotterConfig | None = None,
) -> dict[str, Path]:
    resolved_manifest_path = resolve_input_path(repo_root=repo_root, value=manifest_path)
    resolved_output_dir = resolve_input_path(repo_root=repo_root, value=output_dir)
    resolved_cache_dir = resolve_input_path(repo_root=repo_root, value=cache_dir)
    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, reports_dir, resolved_cache_dir]:
        path.mkdir(parents=True, exist_ok=True)

    resolved_config = HotspotterConfig() if config is None else config
    labeled_df = load_labeled_manifest(resolved_manifest_path)
    feature_start = time.perf_counter()
    features, extraction_report = extract_hesaff_features_with_report(
        df=labeled_df,
        repo_root=repo_root,
        config=resolved_config,
        path_column=PATH_COLUMN,
        cache_dir=resolved_cache_dir,
    )
    feature_seconds = time.perf_counter() - feature_start
    query_start = time.perf_counter()
    query_results = query_hotspotter_all(features=features, config=resolved_config)
    query_seconds = time.perf_counter() - query_start
    prescore_ranking_df = prescore_results_to_dataframe(features=features, query_results=query_results, top_k=top_k)
    ranking_df = rank_results_to_dataframe(features=features, query_results=query_results, top_k=top_k)
    first_hit_df = _first_hit_rank_df(ranking_df=ranking_df, labels_df=labeled_df)
    prescore_first_hit_df = _first_hit_rank_df(ranking_df=prescore_ranking_df, labels_df=labeled_df)

    recall_df = pd.DataFrame(
        [
            {"stage": "pre_sv", "metric": "recall_at_1", "value": _effective_recall_at_k(ranking_df=prescore_ranking_df, labels_df=labeled_df, k=1)},
            {"stage": "pre_sv", "metric": "recall_at_5", "value": _effective_recall_at_k(ranking_df=prescore_ranking_df, labels_df=labeled_df, k=5)},
            {"stage": "pre_sv", "metric": "recall_at_10", "value": _effective_recall_at_k(ranking_df=prescore_ranking_df, labels_df=labeled_df, k=10)},
            {"stage": "post_sv", "metric": "recall_at_1", "value": _effective_recall_at_k(ranking_df=ranking_df, labels_df=labeled_df, k=1)},
            {"stage": "post_sv", "metric": "recall_at_5", "value": _effective_recall_at_k(ranking_df=ranking_df, labels_df=labeled_df, k=5)},
            {"stage": "post_sv", "metric": "recall_at_10", "value": _effective_recall_at_k(ranking_df=ranking_df, labels_df=labeled_df, k=10)},
        ]
    )

    prescore_ranking_path = tables_dir / "query_rankings_presv_v1.csv"
    ranking_path = tables_dir / "query_rankings_v1.csv"
    recall_path = tables_dir / "recall_summary_v1.csv"
    first_hit_path = tables_dir / "first_hit_rank_v1.csv"
    prescore_first_hit_path = tables_dir / "first_hit_rank_presv_v1.csv"
    prescore_ranking_df.to_csv(prescore_ranking_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    recall_df.to_csv(recall_path, index=False)
    first_hit_df.to_csv(first_hit_path, index=False)
    prescore_first_hit_df.to_csv(prescore_first_hit_path, index=False)

    qualitative_dir = resolved_output_dir / "qualitative"
    true_positive_rows, false_positive_rows = _build_qualitative_rows(
        labeled_df=labeled_df,
        prescore_ranking_df=prescore_ranking_df,
        ranking_df=ranking_df,
        query_results=query_results,
        limit=8,
    )
    true_positive_board_path = qualitative_dir / "top_true_positive_inliers_v1.jpg"
    false_positive_board_path = qualitative_dir / "top_false_positive_inliers_v1.jpg"
    create_match_board(
        repo_root=repo_root,
        features=features,
        rows=true_positive_rows,
        output_path=true_positive_board_path,
        title="Top True Positive Inlier Matches",
    )
    create_match_board(
        repo_root=repo_root,
        features=features,
        rows=false_positive_rows,
        output_path=false_positive_board_path,
        title="Top False Positive Inlier Matches",
    )

    label_counts = labeled_df["identity"].value_counts()
    multi_query_count = int((labeled_df["identity"].map(label_counts) > 1).sum())
    pre_sv_recall_at_1 = float(recall_df.loc[(recall_df["stage"].eq("pre_sv")) & (recall_df["metric"].eq("recall_at_1")), "value"].iloc[0])
    pre_sv_recall_at_5 = float(recall_df.loc[(recall_df["stage"].eq("pre_sv")) & (recall_df["metric"].eq("recall_at_5")), "value"].iloc[0])
    pre_sv_recall_at_10 = float(recall_df.loc[(recall_df["stage"].eq("pre_sv")) & (recall_df["metric"].eq("recall_at_10")), "value"].iloc[0])
    post_sv_recall_at_1 = float(recall_df.loc[(recall_df["stage"].eq("post_sv")) & (recall_df["metric"].eq("recall_at_1")), "value"].iloc[0])
    post_sv_recall_at_5 = float(recall_df.loc[(recall_df["stage"].eq("post_sv")) & (recall_df["metric"].eq("recall_at_5")), "value"].iloc[0])
    post_sv_recall_at_10 = float(recall_df.loc[(recall_df["stage"].eq("post_sv")) & (recall_df["metric"].eq("recall_at_10")), "value"].iloc[0])
    summary = {
        "manifest_path": _path_ref(repo_root, resolved_manifest_path),
        "image_count": int(len(labeled_df)),
        "identity_count": int(labeled_df["identity"].nunique()),
        "singleton_identity_count": int((label_counts == 1).sum()),
        "multi_identity_count": int((label_counts >= 2).sum()),
        "multi_query_count": multi_query_count,
        "pre_sv_recall_at_1": pre_sv_recall_at_1,
        "pre_sv_recall_at_5": pre_sv_recall_at_5,
        "pre_sv_recall_at_10": pre_sv_recall_at_10,
        "post_sv_recall_at_1": post_sv_recall_at_1,
        "post_sv_recall_at_5": post_sv_recall_at_5,
        "post_sv_recall_at_10": post_sv_recall_at_10,
        "feature_seconds": round(float(feature_seconds), 6),
        "query_seconds": round(float(query_seconds), 6),
        "feature_cache_dir": _path_ref(repo_root, resolved_cache_dir),
        "feature_cache_hits": int(extraction_report.cache_hit_count),
        "feature_cache_misses": int(extraction_report.cache_miss_count),
        "feature_cache_writes": int(extraction_report.cache_write_count),
        "config": {
            "k": int(resolved_config.k),
            "knorm": int(resolved_config.knorm),
            "n_shortlist": int(resolved_config.n_shortlist),
            "xy_thresh": float(resolved_config.xy_thresh),
            "scale_thresh_low": float(resolved_config.scale_thresh_low),
            "scale_thresh_high": float(resolved_config.scale_thresh_high),
            "min_n_inliers": int(resolved_config.min_n_inliers),
            "rootsift": bool(resolved_config.rootsift),
            "max_side": None if resolved_config.max_side is None else int(resolved_config.max_side),
            "max_features_per_image": (
                None if resolved_config.max_features_per_image is None else int(resolved_config.max_features_per_image)
            ),
            "feature_selection_strategy": str(resolved_config.feature_selection_strategy),
        },
        "true_positive_board_path": _path_ref(repo_root, true_positive_board_path),
        "false_positive_board_path": _path_ref(repo_root, false_positive_board_path),
    }
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    hardest_queries = (
        first_hit_df[first_hit_df["first_hit_rank"].isna() | first_hit_df["first_hit_rank"].gt(5)]
        .head(15)
        .reset_index(drop=True)
    )
    easiest_queries = first_hit_df[first_hit_df["first_hit_rank"].notna()].sort_values("first_hit_rank").head(15).reset_index(drop=True)
    summary_md = "\n".join(
        [
            "# TCU Texas HotSpotter Eval v1",
            "",
            "## Summary",
            "",
            f"- `manifest_path`: `{summary['manifest_path']}`",
            f"- `image_count`: `{summary['image_count']}`",
            f"- `identity_count`: `{summary['identity_count']}`",
            f"- `singleton_identity_count`: `{summary['singleton_identity_count']}`",
            f"- `multi_identity_count`: `{summary['multi_identity_count']}`",
            f"- `multi_query_count`: `{summary['multi_query_count']}`",
            f"- `pre-SV Recall@1`: `{summary['pre_sv_recall_at_1']}`",
            f"- `pre-SV Recall@5`: `{summary['pre_sv_recall_at_5']}`",
            f"- `pre-SV Recall@10`: `{summary['pre_sv_recall_at_10']}`",
            f"- `post-SV Recall@1`: `{summary['post_sv_recall_at_1']}`",
            f"- `post-SV Recall@5`: `{summary['post_sv_recall_at_5']}`",
            f"- `post-SV Recall@10`: `{summary['post_sv_recall_at_10']}`",
            f"- `feature_seconds`: `{summary['feature_seconds']}`",
            f"- `query_seconds`: `{summary['query_seconds']}`",
            f"- `feature_cache_dir`: `{summary['feature_cache_dir']}`",
            f"- `feature_cache_hits`: `{summary['feature_cache_hits']}`",
            f"- `feature_cache_misses`: `{summary['feature_cache_misses']}`",
            f"- `feature_cache_writes`: `{summary['feature_cache_writes']}`",
            f"- `feature_selection_strategy`: `{summary['config']['feature_selection_strategy']}`",
            f"- `true_positive_board_path`: `{summary['true_positive_board_path']}`",
            f"- `false_positive_board_path`: `{summary['false_positive_board_path']}`",
            "",
            "## Easy Queries",
            "",
            _markdown_table(easiest_queries),
            "",
            "## Hard Queries",
            "",
            _markdown_table(hardest_queries),
        ]
    )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    return {
        "prescore_ranking_path": prescore_ranking_path,
        "ranking_path": ranking_path,
        "recall_path": recall_path,
        "prescore_first_hit_path": prescore_first_hit_path,
        "first_hit_path": first_hit_path,
        "true_positive_board_path": true_positive_board_path,
        "false_positive_board_path": false_positive_board_path,
        "summary_path": summary_path,
        "summary_json_path": summary_json_path,
    }
