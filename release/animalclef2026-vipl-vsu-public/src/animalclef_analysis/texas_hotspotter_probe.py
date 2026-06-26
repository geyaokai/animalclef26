from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from .descriptor_baselines import PATH_COLUMN
from .local_matching import (
    HotspotterConfig,
    extract_hesaff_features_with_report,
    query_hotspotter_all,
    rank_results_to_dataframe,
    unique_pair_results_to_dataframe,
)
from .local_matching.hotspotter_pipeline import summarize_feature_counts


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_TEXAS_VIEW_MANIFEST_PATH = Path(
    "artifacts/manifests/texas_center_body_square_repaired_v1/tables/manifest_test_texas_center_body_square_gray_v1.csv"
)
DEFAULT_TEXAS_HOTSPOTTER_OUTPUT_DIR = Path("artifacts/analysis/texas_hotspotter_probe_v1")
DEFAULT_TEXAS_HOTSPOTTER_CACHE_DIR = Path("artifacts/cache/hotspotter_features")


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def resolve_input_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def load_texas_view_df(manifest_path: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path).copy()
    df["image_id"] = df["image_id"].astype(str)
    subset = df[df["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {manifest_path}")
    if PATH_COLUMN not in subset.columns:
        raise ValueError(f"{manifest_path} is missing {PATH_COLUMN}")
    subset[PATH_COLUMN] = subset[PATH_COLUMN].fillna("").astype(str)
    if subset[PATH_COLUMN].eq("").any():
        examples = subset.loc[subset[PATH_COLUMN].eq(""), "image_id"].head(5).tolist()
        raise ValueError(f"Missing Texas view paths in {manifest_path}; examples: {examples}")
    return subset


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def run_texas_hotspotter_probe(
    *,
    repo_root: Path,
    manifest_path: Path = DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    output_dir: Path = DEFAULT_TEXAS_HOTSPOTTER_OUTPUT_DIR,
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
    texas_df = load_texas_view_df(resolved_manifest_path)
    feature_start = time.perf_counter()
    features, extraction_report = extract_hesaff_features_with_report(
        df=texas_df,
        repo_root=repo_root,
        config=resolved_config,
        path_column=PATH_COLUMN,
        cache_dir=resolved_cache_dir,
    )
    feature_seconds = time.perf_counter() - feature_start
    query_start = time.perf_counter()
    query_results = query_hotspotter_all(features=features, config=resolved_config)
    query_seconds = time.perf_counter() - query_start
    ranking_df = rank_results_to_dataframe(features=features, query_results=query_results, top_k=top_k)
    pair_df = unique_pair_results_to_dataframe(features=features, query_results=query_results, top_k=top_k)
    feature_df = summarize_feature_counts(features)

    ranking_path = tables_dir / "query_rankings_v1.csv"
    pair_path = tables_dir / "test_pair_local_scores_v1.csv"
    feature_path = tables_dir / "image_feature_stats_v1.csv"
    ranking_df.to_csv(ranking_path, index=False)
    pair_df.to_csv(pair_path, index=False)
    feature_df.to_csv(feature_path, index=False)

    summary = {
        "probe": resolved_output_dir.name,
        "dataset": TEXAS_DATASET,
        "manifest_path": _path_ref(repo_root, resolved_manifest_path),
        "image_count": int(len(texas_df)),
        "query_count": int(len(query_results)),
        "ranking_rows": int(len(ranking_df)),
        "unique_pair_rows": int(len(pair_df)),
        "mean_keypoints": round(float(feature_df["keypoints"].mean()) if not feature_df.empty else 0.0, 6),
        "median_keypoints": round(float(feature_df["keypoints"].median()) if not feature_df.empty else 0.0, 6),
        "mean_local_score": round(float(pair_df["local_score"].mean()) if not pair_df.empty else 0.0, 6),
        "mean_inliers": round(float(pair_df["inliers"].mean()) if not pair_df.empty else 0.0, 6),
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
        },
    }
    summary_json_path = reports_dir / "summary.json"
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_pairs = pair_df.head(15).loc[
        :,
        [
            "image_id",
            "neighbor_image_id",
            "local_score",
            "local_prescore",
            "good_matches",
            "inliers",
            "rank",
        ],
    ]
    summary_md = "\n".join(
        [
            "# Texas HotSpotter Probe v1",
            "",
            "## Goal",
            "",
            "- 使用 HotSpotter 风格的 `HesAff + RootSIFT + LNBNN + spatial verification` 做 Texas 局部召回。",
            "- 当前查询库与数据库都来自 `texas_center_body_square_gray_v1` 视角，不再依赖全局 embedding 粗排。",
            "",
            "## Inputs",
            "",
            f"- `manifest_path`: `{_path_ref(repo_root, resolved_manifest_path)}`",
            f"- `top_k`: `{int(top_k)}`",
            "",
            "## Summary",
            "",
            f"- `image_count`: `{summary['image_count']}`",
            f"- `ranking_rows`: `{summary['ranking_rows']}`",
            f"- `unique_pair_rows`: `{summary['unique_pair_rows']}`",
            f"- `mean_keypoints`: `{summary['mean_keypoints']}`",
            f"- `median_keypoints`: `{summary['median_keypoints']}`",
            f"- `mean_local_score`: `{summary['mean_local_score']}`",
            f"- `mean_inliers`: `{summary['mean_inliers']}`",
            f"- `feature_seconds`: `{summary['feature_seconds']}`",
            f"- `query_seconds`: `{summary['query_seconds']}`",
            f"- `feature_cache_hits`: `{summary['feature_cache_hits']}`",
            f"- `feature_cache_misses`: `{summary['feature_cache_misses']}`",
            f"- `feature_cache_writes`: `{summary['feature_cache_writes']}`",
            "",
            "## Top Pairs",
            "",
            _markdown_table(top_pairs),
        ]
    )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    return {
        "ranking_path": ranking_path,
        "pair_path": pair_path,
        "feature_path": feature_path,
        "summary_path": summary_path,
        "summary_json_path": summary_json_path,
    }
