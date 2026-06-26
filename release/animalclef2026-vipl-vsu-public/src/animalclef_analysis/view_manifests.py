from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .body_orientation_probe import (
    ALIGNED_CROP_PADDING_RATIO_OVERRIDES,
    DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    DEFAULT_ROTATION_CANVAS_FILL_MODE,
    PROMPTS_BY_DATASET as BODY_AXIS_PROMPTS_BY_DATASET,
    compute_body_axis,
    decide_orientation_application,
    merge_masks,
    resolve_crop_padding_ratio,
    rotate_and_crop,
    rotation_to_horizontal,
)
from .initial_audit import load_metadata
from .preprocessing_baselines import build_duplicate_flags_from_metrics
from .sam3_probe import (
    PROMPTS_BY_DATASET as SAM_MASKED_PROMPTS_BY_DATASET,
    Sam3Resources,
    crop_to_union_mask,
    get_prompt_candidates_for_dataset,
    load_sam3,
    run_single_inference,
    run_single_inference_with_prompt_backoff,
)


PATH_COLUMN = "recommended_model_input_path_v1"
DEFAULT_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_VIEW_NAME = "original_only"
BODY_AXIS_VIEW_NAME = "body_axis_unsigned_rgb_v1"
SAM_MASKED_VIEW_NAME = "sam_masked_rgb_v1"
DEFAULT_BODY_AXIS_DATASETS = ("SalamanderID2025", "TexasHornedLizards")
DEFAULT_SAM_MASKED_DATASETS = ("SalamanderID2025", "TexasHornedLizards")

MANIFEST_FILENAMES = {
    DEFAULT_VIEW_NAME: {
        "train": "manifest_train_original_only_v1.csv",
        "test": "manifest_test_original_only_v1.csv",
    },
    BODY_AXIS_VIEW_NAME: {
        "train": "manifest_train_body_axis_unsigned_rgb_v1.csv",
        "test": "manifest_test_body_axis_unsigned_rgb_v1.csv",
    },
    SAM_MASKED_VIEW_NAME: {
        "train": "manifest_train_sam_masked_rgb_v1.csv",
        "test": "manifest_test_sam_masked_rgb_v1.csv",
    },
}
DEFAULT_MANIFEST_FILENAMES = {
    "train": "manifest_train_default_v1.csv",
    "test": "manifest_test_default_v1.csv",
}


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def get_default_manifest_root(repo_root: Path) -> Path:
    return repo_root / DEFAULT_MANIFEST_ROOT


def get_default_manifest_paths(repo_root: Path) -> tuple[Path, Path]:
    manifest_root = get_default_manifest_root(repo_root)
    return (
        manifest_root / "tables" / DEFAULT_MANIFEST_FILENAMES["train"],
        manifest_root / "tables" / DEFAULT_MANIFEST_FILENAMES["test"],
    )


def get_manifest_paths_for_view(manifest_root: Path, view_name: str) -> tuple[Path, Path]:
    if view_name not in MANIFEST_FILENAMES:
        raise ValueError(f"Unsupported view_name: {view_name}")
    filenames = MANIFEST_FILENAMES[view_name]
    return (
        manifest_root / "tables" / filenames["train"],
        manifest_root / "tables" / filenames["test"],
    )


def _export_join_keys() -> list[str]:
    return ["image_id", "dataset", "split", "path"]


def _load_existing_export_frame(path: Path, empty_factory) -> pd.DataFrame:
    if not path.exists():
        return empty_factory()
    export_df = pd.read_csv(path)
    for column in _export_join_keys():
        if column in export_df.columns:
            export_df[column] = export_df[column].astype(str)
    return export_df


def _pending_export_df(
    metadata_df: pd.DataFrame,
    selected_datasets: list[str],
    existing_export_df: pd.DataFrame | None,
) -> pd.DataFrame:
    export_df = metadata_df[metadata_df["dataset"].isin(selected_datasets)].copy()
    for column in _export_join_keys():
        if column in export_df.columns:
            export_df[column] = export_df[column].astype(str)
    if export_df.empty:
        return export_df
    if existing_export_df is None or existing_export_df.empty:
        return export_df

    pending = export_df.merge(
        existing_export_df[_export_join_keys()].drop_duplicates(),
        on=_export_join_keys(),
        how="left",
        indicator=True,
    )
    return pending[pending["_merge"] == "left_only"][metadata_df.columns].reset_index(drop=True)


def _merge_export_frames(existing_export_df: pd.DataFrame, new_export_df: pd.DataFrame) -> pd.DataFrame:
    if existing_export_df.empty and new_export_df.empty:
        return new_export_df.copy()
    if existing_export_df.empty:
        for column in _export_join_keys():
            if column in new_export_df.columns:
                new_export_df[column] = new_export_df[column].astype(str)
        return new_export_df.sort_values(_export_join_keys()).reset_index(drop=True)
    if new_export_df.empty:
        for column in _export_join_keys():
            if column in existing_export_df.columns:
                existing_export_df[column] = existing_export_df[column].astype(str)
        return existing_export_df.sort_values(_export_join_keys()).reset_index(drop=True)

    combined = pd.concat([existing_export_df, new_export_df], ignore_index=True)
    for column in _export_join_keys():
        if column in combined.columns:
            combined[column] = combined[column].astype(str)
    combined = combined.drop_duplicates(subset=_export_join_keys(), keep="last")
    return combined.sort_values(_export_join_keys()).reset_index(drop=True)


def _coerce_duplicate_flags(metadata_df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    del metadata_df
    metrics_path = repo_root / "artifacts" / "initial_audit" / "tables" / "image_metrics.csv"
    if not metrics_path.exists():
        columns = [
            "image_id",
            "exact_duplicate_sha1",
            "duplicate_group_size",
            "duplicate_rank",
            "is_exact_duplicate",
            "is_duplicate_primary",
        ]
        return pd.DataFrame(columns=columns)
    metrics_df = pd.read_csv(metrics_path)
    if "image_id" in metrics_df.columns:
        metrics_df["image_id"] = metrics_df["image_id"].astype(str)
    return build_duplicate_flags_from_metrics(metrics_df)


def _empty_body_axis_export_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_id",
            "dataset",
            "split",
            "path",
            "body_axis_unsigned_rgb_v1_prompt",
            "body_axis_unsigned_rgb_v1_mask_count",
            "body_axis_unsigned_rgb_v1_union_area_ratio",
            "body_axis_unsigned_rgb_v1_largest_component_ratio",
            "body_axis_unsigned_rgb_v1_foreground_pixels",
            "body_axis_unsigned_rgb_v1_foreground_area_ratio",
            "body_axis_unsigned_rgb_v1_bbox_fill_ratio",
            "body_axis_unsigned_rgb_v1_axis_angle_deg",
            "body_axis_unsigned_rgb_v1_axis_confidence",
            "body_axis_unsigned_rgb_v1_rotation_applied_deg",
            "body_axis_unsigned_rgb_v1_aligned_foreground_ratio",
            "body_axis_unsigned_rgb_v1_padding_ratio",
            "body_axis_unsigned_rgb_v1_status",
            "body_axis_unsigned_rgb_v1_reason",
            "body_axis_unsigned_rgb_v1_applied",
            "body_axis_unsigned_rgb_v1_export_path",
            "body_axis_unsigned_rgb_v1_canvas_fill_mode",
        ]
    )


def export_body_axis_view(
    repo_root: Path,
    output_dir: Path,
    metadata_df: pd.DataFrame,
    *,
    datasets: list[str] | None = None,
    existing_export_df: pd.DataFrame | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda:0",
    min_foreground_pixels: int = 1024,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.85,
    min_axis_confidence: float = 0.35,
    min_largest_component_ratio: float = 0.8,
    aligned_crop_padding_ratio: float = DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    aligned_crop_padding_ratio_overrides: dict[str, float] | None = None,
    keep_background: bool = True,
    rotation_canvas_fill_mode: str = DEFAULT_ROTATION_CANVAS_FILL_MODE,
) -> pd.DataFrame:
    selected_datasets = list(DEFAULT_BODY_AXIS_DATASETS if datasets is None else datasets)
    export_df = _pending_export_df(
        metadata_df=metadata_df,
        selected_datasets=selected_datasets,
        existing_export_df=existing_export_df,
    )
    if export_df.empty:
        return _empty_body_axis_export_frame()

    if aligned_crop_padding_ratio_overrides is None:
        aligned_crop_padding_ratio_overrides = ALIGNED_CROP_PADDING_RATIO_OVERRIDES

    views_dir = output_dir / "views" / BODY_AXIS_VIEW_NAME
    views_dir.mkdir(parents=True, exist_ok=True)
    resources: Sam3Resources = load_sam3(device=device)
    rows: list[dict[str, Any]] = []
    ordered_df = export_df.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    total = len(ordered_df)

    for index, row in enumerate(ordered_df.itertuples(index=False), start=1):
        image_path = repo_root / row.path
        prompt_candidates = (
            ["horned lizard body", "Texas horned lizard body", "lizard body", "lizard", "animal body"]
            if str(row.dataset) == "TexasHornedLizards"
            else [BODY_AXIS_PROMPTS_BY_DATASET[row.dataset]]
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            masks, stats = run_single_inference_with_prompt_backoff(
                image=image,
                prompts=prompt_candidates,
                resources=resources,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            if masks is None:
                component_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                stats["largest_component_ratio"] = 0.0
                stats["foreground_pixels"] = 0
                stats["union_area_ratio"] = 0.0
            else:
                component_mask, component_stats = merge_masks(masks)
                stats["largest_component_ratio"] = component_stats["largest_component_ratio"]
                stats["foreground_pixels"] = int(component_stats["foreground_pixels"])
                stats["union_area_ratio"] = component_stats["union_area_ratio"]

            axis_stats = compute_body_axis(component_mask)
            decision = decide_orientation_application(
                axis_stats,
                min_foreground_pixels=min_foreground_pixels,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                min_axis_confidence=min_axis_confidence,
                min_largest_component_ratio=min_largest_component_ratio,
                largest_component_ratio=float(stats["largest_component_ratio"]),
            )
            rotation_applied_deg = rotation_to_horizontal(axis_stats["axis_angle_deg"]) if axis_stats is not None else 0.0
            if not decision.should_apply:
                rotation_applied_deg = 0.0

            crop_padding_ratio = resolve_crop_padding_ratio(
                row.dataset,
                default_padding_ratio=aligned_crop_padding_ratio,
                padding_ratio_overrides=aligned_crop_padding_ratio_overrides,
            )
            export_rel = ""
            aligned_foreground_ratio = round(float(component_mask.mean()), 6)
            if axis_stats is not None and decision.should_apply:
                aligned_crop, rotated_mask = rotate_and_crop(
                    image,
                    component_mask,
                    rotation_applied_deg,
                    padding_ratio=crop_padding_ratio,
                    keep_background=keep_background,
                    canvas_fill_mode=rotation_canvas_fill_mode,
                )
                aligned_foreground_ratio = round(float(rotated_mask.mean()), 6)
                relative_image_path = Path(row.path).relative_to("images")
                export_rel_path = Path("views") / BODY_AXIS_VIEW_NAME / relative_image_path
                export_abs_path = output_dir / export_rel_path
                export_abs_path.parent.mkdir(parents=True, exist_ok=True)
                aligned_crop.save(export_abs_path, quality=95)
                export_rel = str((output_dir.relative_to(repo_root) / export_rel_path).as_posix())

        rows.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "split": row.split,
                "path": row.path,
                "body_axis_unsigned_rgb_v1_prompt": str(stats.get("selected_prompt", prompt_candidates[0])),
                "body_axis_unsigned_rgb_v1_mask_count": int(stats.get("mask_count", 0)),
                "body_axis_unsigned_rgb_v1_union_area_ratio": float(stats.get("union_area_ratio", 0.0)),
                "body_axis_unsigned_rgb_v1_largest_component_ratio": float(stats.get("largest_component_ratio", 0.0)),
                "body_axis_unsigned_rgb_v1_foreground_pixels": float(axis_stats["foreground_pixels"]) if axis_stats is not None else 0.0,
                "body_axis_unsigned_rgb_v1_foreground_area_ratio": float(axis_stats["foreground_area_ratio"]) if axis_stats is not None else 0.0,
                "body_axis_unsigned_rgb_v1_bbox_fill_ratio": float(axis_stats["bbox_fill_ratio"]) if axis_stats is not None else 0.0,
                "body_axis_unsigned_rgb_v1_axis_angle_deg": float(axis_stats["axis_angle_deg"]) if axis_stats is not None else 0.0,
                "body_axis_unsigned_rgb_v1_axis_confidence": float(axis_stats["axis_confidence"]) if axis_stats is not None else 0.0,
                "body_axis_unsigned_rgb_v1_rotation_applied_deg": float(rotation_applied_deg),
                "body_axis_unsigned_rgb_v1_aligned_foreground_ratio": aligned_foreground_ratio,
                "body_axis_unsigned_rgb_v1_padding_ratio": float(crop_padding_ratio),
                "body_axis_unsigned_rgb_v1_status": decision.status,
                "body_axis_unsigned_rgb_v1_reason": decision.reason,
                "body_axis_unsigned_rgb_v1_applied": bool(decision.should_apply),
                "body_axis_unsigned_rgb_v1_export_path": export_rel,
                "body_axis_unsigned_rgb_v1_canvas_fill_mode": rotation_canvas_fill_mode,
            }
        )
        print(
            (
                f"[build_view_manifests] body-axis {index}/{total} done | {row.dataset} | {row.image_id} | "
                f"masks={int(stats.get('mask_count', 0))} | status={decision.status} | reason={decision.reason}"
            ),
            flush=True,
        )

    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def _empty_sam_masked_export_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_id",
            "dataset",
            "split",
            "path",
            "sam_masked_rgb_v1_prompt",
            "sam_masked_rgb_v1_mask_count",
            "sam_masked_rgb_v1_union_area_ratio",
            "sam_masked_rgb_v1_largest_component_ratio",
            "sam_masked_rgb_v1_foreground_pixels",
            "sam_masked_rgb_v1_foreground_area_ratio",
            "sam_masked_rgb_v1_best_score",
            "sam_masked_rgb_v1_status",
            "sam_masked_rgb_v1_reason",
            "sam_masked_rgb_v1_applied",
            "sam_masked_rgb_v1_export_path",
        ]
    )


def export_sam_masked_view(
    repo_root: Path,
    output_dir: Path,
    metadata_df: pd.DataFrame,
    *,
    datasets: list[str] | None = None,
    existing_export_df: pd.DataFrame | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda:0",
    min_area_ratio: float = 0.01,
    max_area_ratio: float = 0.95,
    min_largest_component_ratio: float = 0.7,
) -> pd.DataFrame:
    selected_datasets = list(DEFAULT_SAM_MASKED_DATASETS if datasets is None else datasets)
    export_df = _pending_export_df(
        metadata_df=metadata_df,
        selected_datasets=selected_datasets,
        existing_export_df=existing_export_df,
    )
    if export_df.empty:
        return _empty_sam_masked_export_frame()

    views_dir = output_dir / "views" / SAM_MASKED_VIEW_NAME
    views_dir.mkdir(parents=True, exist_ok=True)
    resources: Sam3Resources = load_sam3(device=device)
    rows: list[dict[str, Any]] = []
    ordered_df = export_df.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    total = len(ordered_df)

    for index, row in enumerate(ordered_df.itertuples(index=False), start=1):
        image_path = repo_root / row.path
        prompt_candidates = get_prompt_candidates_for_dataset(str(row.dataset))
        export_rel = ""
        status = "skip"
        reason = "no_mask"
        applied = False
        union_area_ratio = 0.0
        largest_component_ratio = 0.0
        foreground_pixels = 0.0
        foreground_area_ratio = 0.0
        best_score = 0.0
        mask_count = 0

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            masks, stats = run_single_inference_with_prompt_backoff(
                image=image,
                prompts=prompt_candidates,
                resources=resources,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            best_score = float(stats.get("best_score", 0.0))
            mask_count = int(stats.get("mask_count", 0))
            if masks is not None:
                component_mask, component_stats = merge_masks(masks)
                union_area_ratio = float(component_stats["union_area_ratio"])
                largest_component_ratio = float(component_stats["largest_component_ratio"])
                foreground_pixels = float(component_stats["foreground_pixels"])
                foreground_area_ratio = float(component_mask.mean()) if component_mask.size else 0.0
                if union_area_ratio < min_area_ratio:
                    reason = "small_area_ratio"
                elif union_area_ratio > max_area_ratio:
                    reason = "large_area_ratio"
                elif largest_component_ratio < min_largest_component_ratio:
                    reason = "fragmented_mask"
                else:
                    masked_crop = crop_to_union_mask(image, component_mask[None, ...])
                    relative_image_path = Path(row.path).relative_to("images")
                    export_rel_path = Path("views") / SAM_MASKED_VIEW_NAME / relative_image_path
                    export_abs_path = output_dir / export_rel_path
                    export_abs_path.parent.mkdir(parents=True, exist_ok=True)
                    masked_crop.save(export_abs_path, quality=95)
                    export_rel = str((output_dir.relative_to(repo_root) / export_rel_path).as_posix())
                    status = "apply"
                    reason = "ok"
                    applied = True

        rows.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "split": row.split,
                "path": row.path,
                "sam_masked_rgb_v1_prompt": str(stats.get("selected_prompt", prompt_candidates[0])),
                "sam_masked_rgb_v1_mask_count": mask_count,
                "sam_masked_rgb_v1_union_area_ratio": union_area_ratio,
                "sam_masked_rgb_v1_largest_component_ratio": largest_component_ratio,
                "sam_masked_rgb_v1_foreground_pixels": foreground_pixels,
                "sam_masked_rgb_v1_foreground_area_ratio": foreground_area_ratio,
                "sam_masked_rgb_v1_best_score": best_score,
                "sam_masked_rgb_v1_status": status,
                "sam_masked_rgb_v1_reason": reason,
                "sam_masked_rgb_v1_applied": applied,
                "sam_masked_rgb_v1_export_path": export_rel,
            }
        )
        print(
            (
                f"[build_view_manifests] sam-masked {index}/{total} done | {row.dataset} | {row.image_id} | "
                f"masks={mask_count} | status={status} | reason={reason}"
            ),
            flush=True,
        )

    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def create_metadata_enriched(
    metadata_df: pd.DataFrame,
    *,
    duplicate_flags_df: pd.DataFrame | None = None,
    body_axis_export_df: pd.DataFrame | None = None,
    sam_masked_export_df: pd.DataFrame | None = None,
    default_view_name: str = DEFAULT_VIEW_NAME,
) -> pd.DataFrame:
    enriched = metadata_df.copy()
    enriched["image_id"] = enriched["image_id"].astype(str)
    enriched["identity"] = enriched["identity"].fillna("").astype(str)
    enriched["original_rgb_path_v1"] = enriched["path"].astype(str)
    enriched["preferred_path_v1"] = enriched["original_rgb_path_v1"]
    enriched["preprocess_variant_v1"] = default_view_name
    enriched["rotation_degrees_v1"] = 0.0
    enriched["normalized_path_v1"] = ""
    enriched["normalization_applied_v1"] = False
    enriched["manifest_default_view_v1"] = default_view_name

    if duplicate_flags_df is None:
        duplicate_flags_df = pd.DataFrame(columns=["image_id"])
    duplicate_cols = [
        "image_id",
        "exact_duplicate_sha1",
        "duplicate_group_size",
        "duplicate_rank",
        "is_exact_duplicate",
        "is_duplicate_primary",
    ]
    available_duplicate_cols = [column for column in duplicate_cols if column in duplicate_flags_df.columns]
    enriched = enriched.merge(duplicate_flags_df[available_duplicate_cols], on="image_id", how="left")
    if "exact_duplicate_sha1" not in enriched.columns:
        enriched["exact_duplicate_sha1"] = ""
    if "duplicate_group_size" not in enriched.columns:
        enriched["duplicate_group_size"] = 1
    if "duplicate_rank" not in enriched.columns:
        enriched["duplicate_rank"] = 0
    if "is_exact_duplicate" not in enriched.columns:
        enriched["is_exact_duplicate"] = False
    if "is_duplicate_primary" not in enriched.columns:
        enriched["is_duplicate_primary"] = True
    enriched["exact_duplicate_sha1"] = enriched["exact_duplicate_sha1"].fillna("")
    enriched["duplicate_group_size"] = enriched["duplicate_group_size"].fillna(1).astype(int)
    enriched["duplicate_rank"] = enriched["duplicate_rank"].fillna(0).astype(int)
    enriched["is_exact_duplicate"] = enriched["is_exact_duplicate"].fillna(False).astype(bool)
    enriched["is_duplicate_primary"] = enriched["is_duplicate_primary"].fillna(True).astype(bool)

    enriched["local_label_available_v1"] = enriched["identity"] != ""
    enriched["sea_turtle_duplicate_nonprimary_v1"] = (
        (enriched["dataset"] == "SeaTurtleID2022")
        & enriched["is_exact_duplicate"]
        & (~enriched["is_duplicate_primary"])
    )
    enriched["recommended_train_keep_all_v1"] = (
        (enriched["split"] == "train") & enriched["local_label_available_v1"]
    )
    enriched["recommended_train_keep_dedup_v1"] = (
        enriched["recommended_train_keep_all_v1"] & (~enriched["sea_turtle_duplicate_nonprimary_v1"])
    )

    if body_axis_export_df is None or body_axis_export_df.empty:
        body_axis_export_df = _empty_body_axis_export_frame()
    body_axis_export_df = body_axis_export_df.copy()
    if "image_id" in body_axis_export_df.columns:
        body_axis_export_df["image_id"] = body_axis_export_df["image_id"].astype(str)
    enriched = enriched.merge(body_axis_export_df, on=["image_id", "dataset", "split", "path"], how="left")

    if sam_masked_export_df is None or sam_masked_export_df.empty:
        sam_masked_export_df = _empty_sam_masked_export_frame()
    sam_masked_export_df = sam_masked_export_df.copy()
    if "image_id" in sam_masked_export_df.columns:
        sam_masked_export_df["image_id"] = sam_masked_export_df["image_id"].astype(str)
    enriched = enriched.merge(sam_masked_export_df, on=["image_id", "dataset", "split", "path"], how="left")

    default_body_axis_values: dict[str, Any] = {
        "body_axis_unsigned_rgb_v1_prompt": "",
        "body_axis_unsigned_rgb_v1_mask_count": 0,
        "body_axis_unsigned_rgb_v1_union_area_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_largest_component_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_foreground_pixels": 0.0,
        "body_axis_unsigned_rgb_v1_foreground_area_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_bbox_fill_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_axis_angle_deg": 0.0,
        "body_axis_unsigned_rgb_v1_axis_confidence": 0.0,
        "body_axis_unsigned_rgb_v1_rotation_applied_deg": 0.0,
        "body_axis_unsigned_rgb_v1_aligned_foreground_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_padding_ratio": 0.0,
        "body_axis_unsigned_rgb_v1_status": "skip",
        "body_axis_unsigned_rgb_v1_reason": "dataset_not_selected",
        "body_axis_unsigned_rgb_v1_applied": False,
        "body_axis_unsigned_rgb_v1_export_path": "",
        "body_axis_unsigned_rgb_v1_canvas_fill_mode": DEFAULT_ROTATION_CANVAS_FILL_MODE,
    }
    for column, default_value in default_body_axis_values.items():
        if column not in enriched.columns:
            enriched[column] = default_value
        elif isinstance(default_value, str):
            enriched[column] = enriched[column].fillna(default_value)
        else:
            enriched[column] = enriched[column].fillna(default_value)
    enriched["body_axis_unsigned_rgb_v1_applied"] = enriched["body_axis_unsigned_rgb_v1_applied"].fillna(False).astype(bool)
    enriched["body_axis_unsigned_rgb_v1_resolved_variant_v1"] = np.where(
        enriched["body_axis_unsigned_rgb_v1_applied"]
        & (enriched["body_axis_unsigned_rgb_v1_export_path"].astype(str) != ""),
        BODY_AXIS_VIEW_NAME,
        DEFAULT_VIEW_NAME,
    )
    enriched["body_axis_unsigned_rgb_v1_resolved_path_v1"] = np.where(
        enriched["body_axis_unsigned_rgb_v1_applied"]
        & (enriched["body_axis_unsigned_rgb_v1_export_path"].astype(str) != ""),
        enriched["body_axis_unsigned_rgb_v1_export_path"].astype(str),
        enriched["original_rgb_path_v1"].astype(str),
    )

    default_sam_masked_values: dict[str, Any] = {
        "sam_masked_rgb_v1_prompt": "",
        "sam_masked_rgb_v1_mask_count": 0,
        "sam_masked_rgb_v1_union_area_ratio": 0.0,
        "sam_masked_rgb_v1_largest_component_ratio": 0.0,
        "sam_masked_rgb_v1_foreground_pixels": 0.0,
        "sam_masked_rgb_v1_foreground_area_ratio": 0.0,
        "sam_masked_rgb_v1_best_score": 0.0,
        "sam_masked_rgb_v1_status": "skip",
        "sam_masked_rgb_v1_reason": "dataset_not_selected",
        "sam_masked_rgb_v1_applied": False,
        "sam_masked_rgb_v1_export_path": "",
    }
    for column, default_value in default_sam_masked_values.items():
        if column not in enriched.columns:
            enriched[column] = default_value
        elif isinstance(default_value, str):
            enriched[column] = enriched[column].fillna(default_value)
        else:
            enriched[column] = enriched[column].fillna(default_value)
    enriched["sam_masked_rgb_v1_applied"] = enriched["sam_masked_rgb_v1_applied"].fillna(False).astype(bool)
    enriched["sam_masked_rgb_v1_resolved_variant_v1"] = np.where(
        enriched["sam_masked_rgb_v1_applied"]
        & (enriched["sam_masked_rgb_v1_export_path"].astype(str) != ""),
        SAM_MASKED_VIEW_NAME,
        DEFAULT_VIEW_NAME,
    )
    enriched["sam_masked_rgb_v1_resolved_path_v1"] = np.where(
        enriched["sam_masked_rgb_v1_applied"]
        & (enriched["sam_masked_rgb_v1_export_path"].astype(str) != ""),
        enriched["sam_masked_rgb_v1_export_path"].astype(str),
        enriched["original_rgb_path_v1"].astype(str),
    )

    enriched[PATH_COLUMN] = enriched["original_rgb_path_v1"]
    return enriched


def build_view_manifest(
    enriched_df: pd.DataFrame,
    *,
    split: str,
    view_name: str,
) -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    if view_name not in MANIFEST_FILENAMES:
        raise ValueError(f"Unsupported view_name: {view_name}")

    if split == "train":
        manifest_df = enriched_df[enriched_df["recommended_train_keep_all_v1"]].copy()
    else:
        manifest_df = enriched_df[enriched_df["split"] == "test"].copy()

    if view_name == DEFAULT_VIEW_NAME:
        manifest_df["manifest_view_name_v1"] = DEFAULT_VIEW_NAME
        manifest_df["manifest_view_requested_v1"] = DEFAULT_VIEW_NAME
        manifest_df["manifest_view_resolved_v1"] = DEFAULT_VIEW_NAME
        manifest_df["manifest_view_applied_v1"] = False
        manifest_df[PATH_COLUMN] = manifest_df["original_rgb_path_v1"].astype(str)
    elif view_name == BODY_AXIS_VIEW_NAME:
        manifest_df["manifest_view_name_v1"] = BODY_AXIS_VIEW_NAME
        manifest_df["manifest_view_requested_v1"] = BODY_AXIS_VIEW_NAME
        manifest_df["manifest_view_resolved_v1"] = manifest_df["body_axis_unsigned_rgb_v1_resolved_variant_v1"].astype(str)
        manifest_df["manifest_view_applied_v1"] = manifest_df["body_axis_unsigned_rgb_v1_applied"].astype(bool)
        manifest_df[PATH_COLUMN] = manifest_df["body_axis_unsigned_rgb_v1_resolved_path_v1"].astype(str)
    else:
        manifest_df["manifest_view_name_v1"] = SAM_MASKED_VIEW_NAME
        manifest_df["manifest_view_requested_v1"] = SAM_MASKED_VIEW_NAME
        manifest_df["manifest_view_resolved_v1"] = manifest_df["sam_masked_rgb_v1_resolved_variant_v1"].astype(str)
        manifest_df["manifest_view_applied_v1"] = manifest_df["sam_masked_rgb_v1_applied"].astype(bool)
        manifest_df[PATH_COLUMN] = manifest_df["sam_masked_rgb_v1_resolved_path_v1"].astype(str)

    manifest_df["preferred_path_v1"] = manifest_df[PATH_COLUMN]
    manifest_df["preprocess_variant_v1"] = manifest_df["manifest_view_resolved_v1"]
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["identity"] = manifest_df["identity"].fillna("").astype(str)
    return manifest_df.sort_values(["dataset", "identity", "image_id"]).reset_index(drop=True)


def _summarize_body_axis_export(body_axis_export_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if body_axis_export_df.empty:
        aggregate_df = pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "applied",
                "applied_ratio",
                "mean_axis_confidence",
                "mean_foreground_area_ratio",
            ]
        )
        reason_df = pd.DataFrame(columns=["dataset", "reason", "count"])
        return aggregate_df, reason_df

    aggregate_df = (
        body_axis_export_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            applied=("body_axis_unsigned_rgb_v1_applied", lambda s: int(np.sum(s))),
            applied_ratio=("body_axis_unsigned_rgb_v1_applied", lambda s: round(float(np.mean(s)), 4)),
            mean_axis_confidence=(
                "body_axis_unsigned_rgb_v1_axis_confidence",
                lambda s: round(float(np.mean(s)), 4),
            ),
            mean_foreground_area_ratio=(
                "body_axis_unsigned_rgb_v1_foreground_area_ratio",
                lambda s: round(float(np.mean(s)), 4),
            ),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
    )
    reason_df = (
        body_axis_export_df.groupby(["dataset", "body_axis_unsigned_rgb_v1_reason"])
        .size()
        .reset_index(name="count")
        .rename(columns={"body_axis_unsigned_rgb_v1_reason": "reason"})
        .sort_values(["dataset", "count", "reason"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return aggregate_df, reason_df


def _summarize_sam_masked_export(sam_masked_export_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sam_masked_export_df.empty:
        aggregate_df = pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "images",
                "applied",
                "applied_ratio",
                "mean_mask_area_ratio",
                "mean_best_score",
            ]
        )
        reason_df = pd.DataFrame(columns=["dataset", "reason", "count"])
        return aggregate_df, reason_df

    aggregate_df = (
        sam_masked_export_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            applied=("sam_masked_rgb_v1_applied", lambda s: int(np.sum(s))),
            applied_ratio=("sam_masked_rgb_v1_applied", lambda s: round(float(np.mean(s)), 4)),
            mean_mask_area_ratio=(
                "sam_masked_rgb_v1_union_area_ratio",
                lambda s: round(float(np.mean(s)), 4),
            ),
            mean_best_score=(
                "sam_masked_rgb_v1_best_score",
                lambda s: round(float(np.mean(s)), 4),
            ),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
    )
    reason_df = (
        sam_masked_export_df.groupby(["dataset", "sam_masked_rgb_v1_reason"])
        .size()
        .reset_index(name="count")
        .rename(columns={"sam_masked_rgb_v1_reason": "reason"})
        .sort_values(["dataset", "count", "reason"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return aggregate_df, reason_df


def write_manifest_summary(
    output_path: Path,
    *,
    config: dict[str, Any],
    manifest_table_df: pd.DataFrame,
    body_axis_aggregate_df: pd.DataFrame,
    body_axis_reason_df: pd.DataFrame,
    sam_masked_aggregate_df: pd.DataFrame,
    sam_masked_reason_df: pd.DataFrame,
) -> None:
    lines = [
        "# View Manifests v1",
        "",
        "## Config",
        "",
        f"- Default view: `{config['default_view_name']}`",
        f"- Body-axis datasets: `{', '.join(config['body_axis_datasets']) if config['body_axis_datasets'] else 'none'}`",
        f"- SAM-masked datasets: `{', '.join(config['sam_masked_datasets']) if config['sam_masked_datasets'] else 'none'}`",
        f"- Body-axis export: `rotate RGB + keep background`",
        f"- SAM-masked export: `largest-component masked crop with black background`",
        f"- Rotation canvas fill: `{config['rotation_canvas_fill_mode']}`",
        (
            "- Body-axis gate: "
            f"`min_foreground_pixels={config['min_foreground_pixels']}`, "
            f"`min_area_ratio={config['min_area_ratio']}`, "
            f"`max_area_ratio={config['max_area_ratio']}`, "
            f"`min_axis_confidence={config['min_axis_confidence']}`, "
            f"`min_largest_component_ratio={config['min_largest_component_ratio']}`"
        ),
        (
            "- SAM-masked gate: "
            f"`min_area_ratio={config['sam_min_area_ratio']}`, "
            f"`max_area_ratio={config['sam_max_area_ratio']}`, "
            f"`min_largest_component_ratio={config['sam_min_largest_component_ratio']}`"
        ),
        f"- Duplicate flags source: `{config['duplicate_flags_source']}`",
        "",
        "## Manifest Files",
        "",
        dataframe_to_markdown_table(manifest_table_df),
        "",
        "## Body-Axis Aggregate",
        "",
        dataframe_to_markdown_table(body_axis_aggregate_df),
        "",
        "## Body-Axis Reasons",
        "",
        dataframe_to_markdown_table(body_axis_reason_df),
        "",
        "## SAM-Masked Aggregate",
        "",
        dataframe_to_markdown_table(sam_masked_aggregate_df),
        "",
        "## SAM-Masked Reasons",
        "",
        dataframe_to_markdown_table(sam_masked_reason_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_view_manifests(
    repo_root: Path,
    output_dir: Path,
    *,
    default_view_name: str = DEFAULT_VIEW_NAME,
    body_axis_datasets: list[str] | None = None,
    sam_masked_datasets: list[str] | None = None,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda:0",
    min_foreground_pixels: int = 1024,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.85,
    min_axis_confidence: float = 0.35,
    min_largest_component_ratio: float = 0.8,
    aligned_crop_padding_ratio: float = DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    aligned_crop_padding_ratio_overrides: dict[str, float] | None = None,
    keep_background: bool = True,
    rotation_canvas_fill_mode: str = DEFAULT_ROTATION_CANVAS_FILL_MODE,
    sam_min_area_ratio: float = 0.01,
    sam_max_area_ratio: float = 0.95,
    sam_min_largest_component_ratio: float = 0.7,
) -> dict[str, Path]:
    if default_view_name != DEFAULT_VIEW_NAME:
        raise ValueError(f"Unsupported default_view_name: {default_view_name}")

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = load_metadata(repo_root / "metadata.csv")
    duplicate_flags_df = _coerce_duplicate_flags(metadata_df=metadata_df, repo_root=repo_root)
    body_axis_export_path = tables_dir / "body_axis_unsigned_rgb_v1_exports.csv"
    sam_masked_export_path = tables_dir / "sam_masked_rgb_v1_exports.csv"
    existing_body_axis_export_df = _load_existing_export_frame(body_axis_export_path, _empty_body_axis_export_frame)
    existing_sam_masked_export_df = _load_existing_export_frame(sam_masked_export_path, _empty_sam_masked_export_frame)
    body_axis_export_new_df = export_body_axis_view(
        repo_root=repo_root,
        output_dir=output_dir,
        metadata_df=metadata_df,
        datasets=body_axis_datasets,
        existing_export_df=existing_body_axis_export_df,
        threshold=threshold,
        mask_threshold=mask_threshold,
        device=device,
        min_foreground_pixels=min_foreground_pixels,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        min_axis_confidence=min_axis_confidence,
        min_largest_component_ratio=min_largest_component_ratio,
        aligned_crop_padding_ratio=aligned_crop_padding_ratio,
        aligned_crop_padding_ratio_overrides=aligned_crop_padding_ratio_overrides,
        keep_background=keep_background,
        rotation_canvas_fill_mode=rotation_canvas_fill_mode,
    )
    body_axis_export_df = _merge_export_frames(existing_body_axis_export_df, body_axis_export_new_df)
    sam_masked_export_new_df = export_sam_masked_view(
        repo_root=repo_root,
        output_dir=output_dir,
        metadata_df=metadata_df,
        datasets=sam_masked_datasets,
        existing_export_df=existing_sam_masked_export_df,
        threshold=threshold,
        mask_threshold=mask_threshold,
        device=device,
        min_area_ratio=sam_min_area_ratio,
        max_area_ratio=sam_max_area_ratio,
        min_largest_component_ratio=sam_min_largest_component_ratio,
    )
    sam_masked_export_df = _merge_export_frames(existing_sam_masked_export_df, sam_masked_export_new_df)
    enriched_df = create_metadata_enriched(
        metadata_df=metadata_df,
        duplicate_flags_df=duplicate_flags_df,
        body_axis_export_df=body_axis_export_df,
        sam_masked_export_df=sam_masked_export_df,
        default_view_name=default_view_name,
    )

    metadata_path = tables_dir / "metadata_enriched_v1.csv"
    enriched_df.to_csv(metadata_path, index=False)
    body_axis_export_df.to_csv(body_axis_export_path, index=False)
    sam_masked_export_df.to_csv(sam_masked_export_path, index=False)

    manifest_rows: list[dict[str, object]] = []
    written_paths: dict[str, Path] = {
        "metadata_enriched_path": metadata_path,
        "body_axis_export_path": body_axis_export_path,
        "sam_masked_export_path": sam_masked_export_path,
    }
    for view_name in [DEFAULT_VIEW_NAME, BODY_AXIS_VIEW_NAME, SAM_MASKED_VIEW_NAME]:
        for split in ["train", "test"]:
            manifest_df = build_view_manifest(
                enriched_df=enriched_df,
                split=split,
                view_name=view_name,
            )
            manifest_path = output_dir / "tables" / MANIFEST_FILENAMES[view_name][split]
            manifest_df.to_csv(manifest_path, index=False)
            written_paths[f"{view_name}_{split}_manifest_path"] = manifest_path
            manifest_rows.append(
                {
                    "manifest": manifest_path.name,
                    "split": split,
                    "view_name": view_name,
                    "rows": int(len(manifest_df)),
                    "path": str(manifest_path),
                }
            )

    default_train_path = output_dir / "tables" / DEFAULT_MANIFEST_FILENAMES["train"]
    default_test_path = output_dir / "tables" / DEFAULT_MANIFEST_FILENAMES["test"]
    default_train_df = build_view_manifest(enriched_df=enriched_df, split="train", view_name=default_view_name)
    default_test_df = build_view_manifest(enriched_df=enriched_df, split="test", view_name=default_view_name)
    default_train_df.to_csv(default_train_path, index=False)
    default_test_df.to_csv(default_test_path, index=False)
    written_paths["default_train_manifest_path"] = default_train_path
    written_paths["default_test_manifest_path"] = default_test_path
    manifest_rows.extend(
        [
            {
                "manifest": default_train_path.name,
                "split": "train",
                "view_name": default_view_name,
                "rows": int(len(default_train_df)),
                "path": str(default_train_path),
            },
            {
                "manifest": default_test_path.name,
                "split": "test",
                "view_name": default_view_name,
                "rows": int(len(default_test_df)),
                "path": str(default_test_path),
            },
        ]
    )

    manifest_table_df = pd.DataFrame(manifest_rows).sort_values(["split", "manifest"]).reset_index(drop=True)
    body_axis_aggregate_df, body_axis_reason_df = _summarize_body_axis_export(body_axis_export_df)
    sam_masked_aggregate_df, sam_masked_reason_df = _summarize_sam_masked_export(sam_masked_export_df)
    summary_path = reports_dir / "summary.md"
    config = {
        "default_view_name": default_view_name,
        "body_axis_datasets": list(DEFAULT_BODY_AXIS_DATASETS if body_axis_datasets is None else body_axis_datasets),
        "sam_masked_datasets": list(DEFAULT_SAM_MASKED_DATASETS if sam_masked_datasets is None else sam_masked_datasets),
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "device": device,
        "min_foreground_pixels": min_foreground_pixels,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "min_axis_confidence": min_axis_confidence,
        "min_largest_component_ratio": min_largest_component_ratio,
        "aligned_crop_padding_ratio": aligned_crop_padding_ratio,
        "aligned_crop_padding_ratio_overrides": (
            aligned_crop_padding_ratio_overrides
            if aligned_crop_padding_ratio_overrides is not None
            else ALIGNED_CROP_PADDING_RATIO_OVERRIDES
        ),
        "keep_background": keep_background,
        "rotation_canvas_fill_mode": rotation_canvas_fill_mode,
        "sam_min_area_ratio": sam_min_area_ratio,
        "sam_max_area_ratio": sam_max_area_ratio,
        "sam_min_largest_component_ratio": sam_min_largest_component_ratio,
        "duplicate_flags_source": str((repo_root / "artifacts" / "initial_audit" / "tables" / "image_metrics.csv")),
    }
    write_manifest_summary(
        summary_path,
        config=config,
        manifest_table_df=manifest_table_df,
        body_axis_aggregate_df=body_axis_aggregate_df,
        body_axis_reason_df=body_axis_reason_df,
        sam_masked_aggregate_df=sam_masked_aggregate_df,
        sam_masked_reason_df=sam_masked_reason_df,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                **config,
                "default_train_manifest_path": str(default_train_path),
                "default_test_manifest_path": str(default_test_path),
                "metadata_enriched_path": str(metadata_path),
                "body_axis_export_path": str(body_axis_export_path),
                "sam_masked_export_path": str(sam_masked_export_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    written_paths["summary_path"] = summary_path
    return written_paths
