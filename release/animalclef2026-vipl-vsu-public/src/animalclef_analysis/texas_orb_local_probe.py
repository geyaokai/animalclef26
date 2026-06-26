from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .descriptor_baselines import PATH_COLUMN
from .orb_rerank_baseline import build_local_match_table, extract_local_features


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_TEXAS_VIEW_MANIFEST_PATH = Path("artifacts/manifests/v1/tables/manifest_test_body_axis_unsigned_rgb_v1.csv")
DEFAULT_TEXAS_ORB_OUTPUT_DIR = Path("artifacts/analysis/texas_orb_local_probe_v1")
DEFAULT_PAIR_SCORE_COLUMNS = (
    "xgb_same_identity_prob",
    "route_global_score",
    "miew_similarity",
    "fusion_similarity",
    "miew_similarity_shortlist",
    "fusion_similarity_shortlist",
)


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def resolve_input_path(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else (repo_root / value)).resolve()


def resolve_predictions_path(repo_root: Path, value: Path) -> Path:
    path = resolve_input_path(repo_root=repo_root, value=value)
    if path.is_file():
        return path
    candidates = [
        path / "tables" / "test_predictions_best_v1.csv",
        path / "tables" / "test_predictions_v1.csv",
        path / "test_predictions_best_v1.csv",
        path / "test_predictions_v1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve test predictions from {value}")


def load_texas_reference_df(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path).copy()
    df["image_id"] = df["image_id"].astype(str)
    subset = df[df["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {predictions_path}")
    if PATH_COLUMN not in subset.columns and "path" in subset.columns:
        subset[PATH_COLUMN] = subset["path"]
    return subset


def align_reference_frame(reference_df: pd.DataFrame, candidate_df: pd.DataFrame, name: str) -> pd.DataFrame:
    lookup = candidate_df.copy()
    lookup["image_id"] = lookup["image_id"].astype(str)
    lookup = lookup.set_index("image_id", drop=False)
    missing = [image_id for image_id in reference_df["image_id"].astype(str).tolist() if image_id not in lookup.index]
    if missing:
        raise ValueError(f"{name} missing image_ids, examples: {missing[:5]}")
    return lookup.loc[reference_df["image_id"].astype(str).tolist()].reset_index(drop=True).copy()


def load_aligned_texas_view_df(repo_root: Path, reference_df: pd.DataFrame, manifest_path: Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path).copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    subset = manifest_df[manifest_df["dataset"].astype(str).eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows in {manifest_path}")
    aligned = align_reference_frame(reference_df=reference_df, candidate_df=subset, name=manifest_path.name)
    if PATH_COLUMN not in aligned.columns:
        raise ValueError(f"{manifest_path} is missing {PATH_COLUMN}")
    aligned[PATH_COLUMN] = aligned[PATH_COLUMN].fillna("").astype(str)
    if aligned[PATH_COLUMN].eq("").any():
        examples = aligned.loc[aligned[PATH_COLUMN].eq(""), "image_id"].astype(str).head(5).tolist()
        raise ValueError(f"{manifest_path} has empty {PATH_COLUMN} for image_ids: {examples}")
    return aligned


def resolve_pair_score_column(pair_df: pd.DataFrame, requested: str | None = None) -> str:
    if requested is not None and str(requested) in pair_df.columns:
        return str(requested)
    for column in DEFAULT_PAIR_SCORE_COLUMNS:
        if column in pair_df.columns:
            return column
    raise KeyError(
        "Could not resolve pair score column. Expected one of "
        + ", ".join(DEFAULT_PAIR_SCORE_COLUMNS)
        + (f", or explicit column {requested!r}" if requested else "")
    )


def _build_index_lookup(reference_df: pd.DataFrame) -> dict[str, int]:
    return {
        str(image_id): int(index)
        for index, image_id in enumerate(reference_df["image_id"].astype(str).tolist())
    }


def normalize_texas_pair_df(
    pair_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    score_column: str | None = None,
) -> tuple[pd.DataFrame, str]:
    frame = pair_df.copy().reset_index(drop=True)
    image_to_index = _build_index_lookup(reference_df=reference_df)

    if "image_id" in frame.columns:
        frame["image_id"] = frame["image_id"].astype(str)
    if "neighbor_image_id" in frame.columns:
        frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)

    if "left_index" not in frame.columns or "right_index" not in frame.columns:
        if "image_id" not in frame.columns or "neighbor_image_id" not in frame.columns:
            raise KeyError("Pair table needs either (left_index, right_index) or (image_id, neighbor_image_id).")
        frame["left_index"] = frame["image_id"].map(image_to_index)
        frame["right_index"] = frame["neighbor_image_id"].map(image_to_index)
        if frame["left_index"].isna().any() or frame["right_index"].isna().any():
            missing_rows = frame[frame["left_index"].isna() | frame["right_index"].isna()].head(5)
            raise ValueError(
                "Could not map image ids to Texas reference indices, examples: "
                + missing_rows[["image_id", "neighbor_image_id"]].to_dict(orient="records").__repr__()
            )
    frame["left_index"] = frame["left_index"].astype(int)
    frame["right_index"] = frame["right_index"].astype(int)

    if "image_id" not in frame.columns:
        frame["image_id"] = reference_df.iloc[frame["left_index"].to_numpy(dtype=int)]["image_id"].astype(str).to_numpy()
    if "neighbor_image_id" not in frame.columns:
        frame["neighbor_image_id"] = reference_df.iloc[frame["right_index"].to_numpy(dtype=int)]["image_id"].astype(str).to_numpy()

    if (frame["left_index"] == frame["right_index"]).any():
        frame = frame[frame["left_index"] != frame["right_index"]].copy().reset_index(drop=True)

    swap_mask = frame["left_index"].to_numpy(dtype=int) > frame["right_index"].to_numpy(dtype=int)
    if swap_mask.any():
        left_index = frame.loc[swap_mask, "left_index"].copy()
        frame.loc[swap_mask, "left_index"] = frame.loc[swap_mask, "right_index"].to_numpy(dtype=int)
        frame.loc[swap_mask, "right_index"] = left_index.to_numpy(dtype=int)
        left_ids = frame.loc[swap_mask, "image_id"].copy()
        frame.loc[swap_mask, "image_id"] = frame.loc[swap_mask, "neighbor_image_id"].astype(str).to_numpy()
        frame.loc[swap_mask, "neighbor_image_id"] = left_ids.astype(str).to_numpy()

    resolved_score_column = resolve_pair_score_column(frame, requested=score_column)
    frame[resolved_score_column] = pd.to_numeric(frame[resolved_score_column], errors="coerce").fillna(0.0)

    dedupe_columns = ["left_index", "right_index", "image_id", "neighbor_image_id"]
    frame = frame.sort_values([resolved_score_column], ascending=[False]).drop_duplicates(
        subset=dedupe_columns,
        keep="first",
    )
    return frame.reset_index(drop=True), resolved_score_column


def build_texas_orb_pair_index(
    pair_df: pd.DataFrame,
    *,
    reference_df: pd.DataFrame,
    score_column: str | None = None,
) -> tuple[list[tuple[int, int, float]], pd.DataFrame, str]:
    normalized_pair_df, resolved_score_column = normalize_texas_pair_df(
        pair_df=pair_df,
        reference_df=reference_df,
        score_column=score_column,
    )
    pair_index = [
        (int(row.left_index), int(row.right_index), float(getattr(row, resolved_score_column)))
        for row in normalized_pair_df.itertuples(index=False)
    ]
    return pair_index, normalized_pair_df, resolved_score_column


def _markdown_table(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    preview = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    if preview.empty:
        return "_empty_"
    header = "| " + " | ".join(preview.columns.tolist()) + " |"
    separator = "| " + " | ".join(["---"] * len(preview.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in preview.columns.tolist()) + " |"
        for _, row in preview.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def run_texas_orb_local_probe(
    *,
    repo_root: Path,
    predictions_path: Path,
    pair_csv_path: Path,
    output_dir: Path,
    view_manifest_path: Path = DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    score_column: str | None = None,
    nfeatures: int = 2048,
    max_side: int = 768,
    fast_threshold: int = 12,
    clahe_clip_limit: float = 2.0,
    ratio_test: float = 0.85,
    ransac_threshold: float = 5.0,
    min_inliers: int = 4,
) -> dict[str, Path]:
    resolved_predictions_path = resolve_predictions_path(repo_root=repo_root, value=predictions_path)
    resolved_pair_csv_path = resolve_input_path(repo_root=repo_root, value=pair_csv_path)
    resolved_output_dir = resolve_input_path(repo_root=repo_root, value=output_dir)
    resolved_view_manifest_path = resolve_input_path(repo_root=repo_root, value=view_manifest_path)

    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    reference_df = load_texas_reference_df(resolved_predictions_path)
    raw_pair_df = pd.read_csv(resolved_pair_csv_path)
    pair_index, normalized_pair_df, resolved_score_column = build_texas_orb_pair_index(
        pair_df=raw_pair_df,
        reference_df=reference_df,
        score_column=score_column,
    )
    view_df = load_aligned_texas_view_df(
        repo_root=repo_root,
        reference_df=reference_df,
        manifest_path=resolved_view_manifest_path,
    )
    features = extract_local_features(
        df=view_df,
        repo_root=repo_root,
        nfeatures=int(nfeatures),
        max_side=int(max_side),
        fast_threshold=int(fast_threshold),
        clahe_clip_limit=float(clahe_clip_limit),
        local_matcher="orb",
    )
    local_df = build_local_match_table(
        df=view_df,
        features=features,
        pair_index=pair_index,
        ratio_test=float(ratio_test),
        ransac_threshold=float(ransac_threshold),
        min_inliers=int(min_inliers),
        local_matcher="orb",
    )
    local_df["pair_score_col"] = resolved_score_column
    image_lookup = view_df.set_index("image_id", drop=False)
    local_df["left_view_path"] = [str(image_lookup.at[str(image_id), PATH_COLUMN]) for image_id in local_df["image_id"]]
    local_df["right_view_path"] = [
        str(image_lookup.at[str(image_id), PATH_COLUMN]) for image_id in local_df["neighbor_image_id"]
    ]
    if "manifest_view_resolved_v1" in view_df.columns:
        local_df["left_view_resolved"] = [
            str(image_lookup.at[str(image_id), "manifest_view_resolved_v1"]) for image_id in local_df["image_id"]
        ]
        local_df["right_view_resolved"] = [
            str(image_lookup.at[str(image_id), "manifest_view_resolved_v1"]) for image_id in local_df["neighbor_image_id"]
        ]
    if "manifest_view_applied_v1" in view_df.columns:
        local_df["left_view_applied"] = [
            bool(image_lookup.at[str(image_id), "manifest_view_applied_v1"]) for image_id in local_df["image_id"]
        ]
        local_df["right_view_applied"] = [
            bool(image_lookup.at[str(image_id), "manifest_view_applied_v1"]) for image_id in local_df["neighbor_image_id"]
        ]

    local_table_path = tables_dir / "test_pair_local_scores_v1.csv"
    normalized_pair_path = tables_dir / "normalized_pairs_v1.csv"
    local_df.to_csv(local_table_path, index=False)
    normalized_pair_df.to_csv(normalized_pair_path, index=False)

    nonzero_mask = local_df["local_score"].astype(float).gt(0.0) if not local_df.empty else pd.Series(dtype=bool)
    summary = {
        "probe": resolved_output_dir.name,
        "dataset": TEXAS_DATASET,
        "predictions_path": _path_ref(repo_root, resolved_predictions_path),
        "pair_csv_path": _path_ref(repo_root, resolved_pair_csv_path),
        "view_manifest_path": _path_ref(repo_root, resolved_view_manifest_path),
        "pair_score_col": resolved_score_column,
        "image_count": int(len(reference_df)),
        "pair_count": int(len(local_df)),
        "nonzero_local_pair_count": int(nonzero_mask.sum()) if len(nonzero_mask) else 0,
        "mean_local_score": round(float(local_df["local_score"].astype(float).mean()) if not local_df.empty else 0.0, 6),
        "max_local_score": round(float(local_df["local_score"].astype(float).max()) if not local_df.empty else 0.0, 6),
        "mean_inliers": round(float(local_df["inliers"].astype(float).mean()) if not local_df.empty else 0.0, 6),
        "mean_good_matches": round(float(local_df["good_matches"].astype(float).mean()) if not local_df.empty else 0.0, 6),
        "view_applied_ratio": round(
            float(view_df["manifest_view_applied_v1"].fillna(False).astype(bool).mean())
            if "manifest_view_applied_v1" in view_df.columns
            else 0.0,
            6,
        ),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    top_preview = (
        local_df.sort_values(["local_score", "inliers", "good_matches"], ascending=[False, False, False])
        .head(12)
        .reset_index(drop=True)
    )
    summary_md = "\n".join(
        [
            "# Texas ORB Local Probe v1",
            "",
            "## Goal",
            "",
            "- 对 Texas self-train 当前待审 pair 计算 ORB 局部匹配证据。",
            "- 默认使用 `body_axis_unsigned_rgb_v1` 视角，优先让黑色内部花纹在相近身体坐标系下对齐。",
            "",
            "## Inputs",
            "",
            f"- `predictions_path`: `{_path_ref(repo_root, resolved_predictions_path)}`",
            f"- `pair_csv_path`: `{_path_ref(repo_root, resolved_pair_csv_path)}`",
            f"- `view_manifest_path`: `{_path_ref(repo_root, resolved_view_manifest_path)}`",
            f"- `pair_score_col`: `{resolved_score_column}`",
            "",
            "## Summary",
            "",
            f"- `image_count`: `{summary['image_count']}`",
            f"- `pair_count`: `{summary['pair_count']}`",
            f"- `nonzero_local_pair_count`: `{summary['nonzero_local_pair_count']}`",
            f"- `mean_local_score`: `{summary['mean_local_score']}`",
            f"- `max_local_score`: `{summary['max_local_score']}`",
            f"- `mean_inliers`: `{summary['mean_inliers']}`",
            f"- `mean_good_matches`: `{summary['mean_good_matches']}`",
            f"- `view_applied_ratio`: `{summary['view_applied_ratio']}`",
            "",
            "## Top Pairs",
            "",
            _markdown_table(
                top_preview,
                columns=[
                    "image_id",
                    "neighbor_image_id",
                    "global_score",
                    "local_score",
                    "inliers",
                    "good_matches",
                    "left_keypoints",
                    "right_keypoints",
                ],
            ),
        ]
    )
    (reports_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    return {
        "local_table_path": local_table_path,
        "normalized_pair_path": normalized_pair_path,
        "summary_path": reports_dir / "summary.md",
        "summary_json_path": reports_dir / "summary.json",
    }


def merge_texas_orb_local_scores(
    pair_df: pd.DataFrame,
    local_match_df: pd.DataFrame,
    *,
    preserve_existing_local_col: str = "miew_local_score",
    override_local_score: bool = True,
) -> pd.DataFrame:
    result = pair_df.copy().reset_index(drop=True)
    if local_match_df.empty:
        if preserve_existing_local_col and "local_score" in result.columns and preserve_existing_local_col not in result.columns:
            result[preserve_existing_local_col] = pd.to_numeric(result["local_score"], errors="coerce")
        return result

    local = local_match_df.copy().reset_index(drop=True)
    for column in ["left_index", "right_index"]:
        local[column] = pd.to_numeric(local[column], errors="raise").astype(int)
    for column in ["image_id", "neighbor_image_id"]:
        if column in local.columns:
            local[column] = local[column].astype(str)

    rename_map = {
        "matcher_name": "orb_matcher_name",
        "global_score": "orb_probe_global_score",
        "left_keypoints": "orb_left_keypoints",
        "right_keypoints": "orb_right_keypoints",
        "good_matches": "orb_good_matches",
        "inliers": "orb_inliers",
        "local_raw_score": "orb_local_raw_score",
        "local_score": "orb_local_score",
    }
    keep_columns = ["left_index", "right_index", "image_id", "neighbor_image_id"] + [
        column
        for column in (
            "matcher_name",
            "left_keypoints",
            "right_keypoints",
            "good_matches",
            "inliers",
            "local_raw_score",
            "local_score",
        )
        if column in local.columns
    ]
    local = (
        local.loc[:, keep_columns]
        .sort_values(["local_score", "inliers", "good_matches"], ascending=[False, False, False])
        .drop_duplicates(subset=["left_index", "right_index", "image_id", "neighbor_image_id"], keep="first")
        .rename(columns=rename_map)
        .reset_index(drop=True)
    )

    if preserve_existing_local_col and "local_score" in result.columns and preserve_existing_local_col not in result.columns:
        result[preserve_existing_local_col] = pd.to_numeric(result["local_score"], errors="coerce")

    merged = result.merge(
        local,
        on=["left_index", "right_index", "image_id", "neighbor_image_id"],
        how="left",
    )
    if override_local_score and "orb_local_score" in merged.columns:
        orb_local = pd.to_numeric(merged["orb_local_score"], errors="coerce")
        existing_local = (
            pd.to_numeric(merged["local_score"], errors="coerce")
            if "local_score" in merged.columns
            else pd.Series(np.nan, index=merged.index, dtype=float)
        )
        merged["local_score"] = orb_local.where(orb_local.notna(), existing_local)
        merged["local_score_source"] = np.where(
            orb_local.notna(),
            "orb_local_probe",
            np.where(existing_local.notna(), "precomputed_local_score", ""),
        )
    return merged
