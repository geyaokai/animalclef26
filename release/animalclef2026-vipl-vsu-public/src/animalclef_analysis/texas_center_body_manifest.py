from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .descriptor_baselines import PATH_COLUMN
from .texas_unsupervised import TEXAS_DATASET, dataframe_to_markdown_table


MANIFEST_NAME = "manifest_test_texas_center_body_square_gray_v1.csv"
VIEW_NAME = "texas_center_body_square_gray_v1"


def _to_repo_relative(repo_root: Path, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    return os.path.relpath(path.resolve(), start=repo_root.resolve()).replace("\\", "/")


def _normalize_optional_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _load_texas_records(records_path: Path) -> pd.DataFrame:
    records_df = pd.read_csv(records_path)
    records_df["image_id"] = records_df["image_id"].astype(str)
    records_df["dataset"] = records_df["dataset"].astype(str)
    records_df["split"] = records_df["split"].astype(str)
    texas_df = records_df[records_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if texas_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows found in qualitative records: {records_path}")
    if texas_df["image_id"].duplicated().any():
        duplicated = texas_df.loc[texas_df["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Duplicate Texas image_ids in qualitative records: {duplicated}")
    return texas_df


def _load_texas_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    if "split" in manifest_df.columns:
        manifest_df["split"] = manifest_df["split"].astype(str)
    texas_df = manifest_df[(manifest_df["dataset"] == TEXAS_DATASET) & (manifest_df["split"] == "test")].copy().reset_index(drop=True)
    if texas_df.empty:
        raise ValueError(f"No Texas test rows found in repaired manifest: {manifest_path}")
    if texas_df["image_id"].duplicated().any():
        duplicated = texas_df.loc[texas_df["image_id"].duplicated(), "image_id"].tolist()[:5]
        raise ValueError(f"Duplicate Texas image_ids in repaired manifest: {duplicated}")
    return texas_df


def _build_summary_lines(
    *,
    summary_df: pd.DataFrame,
    fallback_stage_df: pd.DataFrame,
    manifest_rel_path: str,
    records_rel_path: str,
    repaired_manifest_rel_path: str,
) -> list[str]:
    lines = [
        "# Texas Center Body Manifest",
        "",
        "## Inputs",
        "",
        f"- Qualitative records: `{records_rel_path}`",
        f"- Repaired SAM manifest: `{repaired_manifest_rel_path}`",
        f"- Output manifest: `{manifest_rel_path}`",
        "",
        "## Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Repaired Fallback Stage",
        "",
        dataframe_to_markdown_table(fallback_stage_df),
        "",
    ]
    return lines


def build_texas_center_body_manifest(
    *,
    repo_root: Path,
    records_path: Path,
    repaired_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    records_path = records_path.resolve()
    repaired_manifest_path = repaired_manifest_path.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    records_df = _load_texas_records(records_path)
    manifest_df = _load_texas_manifest(repaired_manifest_path)
    merged_df = manifest_df.merge(
        records_df,
        on=["image_id", "dataset", "split"],
        how="inner",
        suffixes=("", "__record"),
    )
    if len(merged_df) != len(manifest_df):
        missing = sorted(set(manifest_df["image_id"]) - set(merged_df["image_id"]))[:5]
        raise ValueError(f"Failed to align Texas manifest with qualitative records; missing examples: {missing}")

    new_df = merged_df.copy()
    new_df["texas_center_body_view_name_v1"] = VIEW_NAME
    new_df["texas_center_body_original_path_v1"] = new_df["original_path"].map(lambda value: _to_repo_relative(repo_root, value))
    new_df["texas_center_body_source_aligned_path_v1"] = new_df["source_aligned_path"].map(
        lambda value: _to_repo_relative(repo_root, value)
    )
    new_df["texas_center_body_aligned_path_v1"] = new_df["aligned_path"].map(lambda value: _to_repo_relative(repo_root, value))
    new_df["texas_center_body_scale_norm_path_v1"] = new_df["scale_norm_path"].map(lambda value: _to_repo_relative(repo_root, value))
    new_df["texas_center_body_square_path_v1"] = new_df["center_body_square_path"].map(
        lambda value: _to_repo_relative(repo_root, value)
    )
    new_df["texas_center_body_gray_path_v1"] = new_df["center_body_gray_path"].map(lambda value: _to_repo_relative(repo_root, value))
    new_df["texas_center_body_sam_stage_v1"] = _normalize_optional_text(new_df["sam_stage"])
    new_df["texas_center_body_sam_reason_v1"] = _normalize_optional_text(new_df["sam_reason"])
    new_df["texas_center_body_crop_fallback_reason_v1"] = _normalize_optional_text(new_df["crop_fallback_reason"])
    new_df["texas_center_body_crop_payload_json_v1"] = _normalize_optional_text(new_df["crop_payload_json"])
    new_df["texas_center_body_gray_payload_json_v1"] = _normalize_optional_text(new_df["gray_payload_json"])
    new_df["texas_center_body_scale_payload_json_v1"] = _normalize_optional_text(new_df["scale_payload_json"])
    new_df["texas_center_body_foreground_ratio_of_subject_v1"] = new_df["foreground_ratio_of_subject"].astype(float)
    new_df["texas_center_body_foreground_ratio_in_crop_v1"] = new_df["foreground_ratio_in_crop"].astype(float)
    new_df["texas_center_body_square_side_px_v1"] = new_df["square_side_px"].astype(int)
    new_df["texas_center_body_gray_low_value_v1"] = new_df["gray_low_value"].astype(float)
    new_df["texas_center_body_gray_high_value_v1"] = new_df["gray_high_value"].astype(float)
    new_df["texas_center_body_scale_factor_v1"] = pd.to_numeric(new_df["scale_factor"], errors="coerce")
    new_df["texas_center_body_major_extent_after_px_v1"] = pd.to_numeric(new_df["major_extent_after_px"], errors="coerce")
    new_df["texas_center_body_repaired_fallback_stage_v1"] = _normalize_optional_text(
        new_df["sam_trainprep_masked_fallback_stage_v1"]
    )
    new_df["texas_center_body_repaired_prompt_v1"] = _normalize_optional_text(new_df["sam_trainprep_masked_prompt_used_v1"])

    center_body_rel_path = new_df["texas_center_body_gray_path_v1"]
    new_df["path"] = center_body_rel_path
    new_df[PATH_COLUMN] = center_body_rel_path
    new_df["preferred_path_v1"] = center_body_rel_path
    new_df["recommended_model_input_path_v1"] = center_body_rel_path
    new_df["preprocess_variant_v1"] = VIEW_NAME
    new_df["manifest_view_name_v1"] = VIEW_NAME
    new_df["manifest_view_requested_v1"] = VIEW_NAME
    new_df["manifest_view_resolved_v1"] = VIEW_NAME
    new_df["manifest_view_applied_v1"] = True

    drop_columns = [
        "original_path",
        "aligned_path",
        "source_aligned_path",
        "scale_norm_path",
        "center_body_square_path",
        "center_body_gray_path",
        "sam_stage",
        "sam_reason",
        "foreground_ratio_of_subject",
        "foreground_ratio_in_crop",
        "square_side_px",
        "crop_fallback_reason",
        "gray_low_value",
        "gray_high_value",
        "scale_factor",
        "major_extent_after_px",
        "crop_payload_json",
        "gray_payload_json",
        "scale_payload_json",
    ]
    new_df = new_df.drop(columns=[column for column in drop_columns if column in new_df.columns])

    manifest_path = tables_dir / MANIFEST_NAME
    new_df.to_csv(manifest_path, index=False)

    summary_df = pd.DataFrame(
        [
            {
                "total_images": int(len(new_df)),
                "seed_ready_images": int(len(new_df)),
                "repaired_fallback_images": int(new_df["texas_center_body_repaired_fallback_stage_v1"].ne("none").sum()),
                "raw_original_paths_remaining": int(new_df[PATH_COLUMN].eq(_normalize_optional_text(new_df["original_rgb_path_v1"])).sum()),
            }
        ]
    )
    fallback_stage_df = (
        new_df["texas_center_body_repaired_fallback_stage_v1"]
        .value_counts(dropna=False)
        .rename_axis("fallback_stage")
        .reset_index(name="images")
    )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            _build_summary_lines(
                summary_df=summary_df,
                fallback_stage_df=fallback_stage_df,
                manifest_rel_path=_to_repo_relative(repo_root, manifest_path),
                records_rel_path=_to_repo_relative(repo_root, records_path),
                repaired_manifest_rel_path=_to_repo_relative(repo_root, repaired_manifest_path),
            )
        ),
        encoding="utf-8",
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": TEXAS_DATASET,
                "view_name": VIEW_NAME,
                "manifest_path": _to_repo_relative(repo_root, manifest_path),
                "records_path": _to_repo_relative(repo_root, records_path),
                "repaired_manifest_path": _to_repo_relative(repo_root, repaired_manifest_path),
                "total_images": int(len(new_df)),
                "repaired_fallback_images": int(new_df["texas_center_body_repaired_fallback_stage_v1"].ne("none").sum()),
                "fallback_stage_counts": fallback_stage_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path,
        "summary_path": summary_path,
    }
