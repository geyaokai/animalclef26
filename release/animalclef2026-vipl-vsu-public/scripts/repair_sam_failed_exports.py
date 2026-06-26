#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.sam_augmented_manifests import (
        DEFAULT_SOURCE_MANIFEST_ROOT,
        TRAIN_MULTIVIEW_NAME,
        TRAIN_SINGLE_ALIGNED_VIEW_NAME,
        TRAIN_SINGLE_MASKED_VIEW_NAME,
        _load_base_metadata,
        _merge_export_frames,
        _run_dataset_fallback_mask,
        build_trainprep_aligned_exports,
        build_trainprep_multiview_manifest,
        build_trainprep_single_view_manifest,
        create_trainprep_enriched_metadata,
        write_trainprep_summary,
        _summarize_aligned_exports,
        _summarize_masked_exports,
    )

    parser = argparse.ArgumentParser(description="Repair only failed SAM train-prep exports.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--source-manifest-root", type=Path, default=repo_root / DEFAULT_SOURCE_MANIFEST_ROOT)
    parser.add_argument("--base-sam-root", type=Path, default=repo_root / "artifacts/manifests/sam_seg_trainprep_v1")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts/manifests/sam_seg_trainprep_repaired_v1")
    parser.add_argument("--target-datasets", nargs="*", default=["SalamanderID2025", "TexasHornedLizards"])
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--mask-threshold", type=float, default=0.30)
    parser.add_argument("--min-area-ratio", type=float, default=0.005)
    parser.add_argument("--max-area-ratio", type=float, default=0.98)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.40)
    parser.add_argument("--yolo-model", type=str, default="/home/hechen/gyk/yolov8s-worldv2.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.05)
    parser.add_argument("--yolo-iou", type=float, default=0.50)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-max-det", type=int, default=8)
    parser.add_argument("--alignment-min-foreground-pixels", type=int, default=256)
    parser.add_argument("--alignment-min-area-ratio", type=float, default=0.005)
    parser.add_argument("--alignment-max-area-ratio", type=float, default=0.98)
    parser.add_argument("--alignment-padding-ratio", type=float, default=0.06)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_manifest_root = args.source_manifest_root.resolve()
    base_sam_root = args.base_sam_root.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    base_masked_path = base_sam_root / "tables" / "sam_trainprep_masked_exports_v1.csv"
    base_aligned_path = base_sam_root / "tables" / "sam_trainprep_aligned_exports_v1.csv"
    if not base_masked_path.exists() or not base_aligned_path.exists():
        raise FileNotFoundError("Base SAM export tables are required.")

    base_df = _load_base_metadata(source_manifest_root)
    target_datasets = [str(value) for value in args.target_datasets]
    masked_df = pd.read_csv(base_masked_path)
    aligned_df = pd.read_csv(base_aligned_path)
    masked_df["image_id"] = masked_df["image_id"].astype(str)
    aligned_df["image_id"] = aligned_df["image_id"].astype(str)

    failure_df = masked_df[
        masked_df["dataset"].astype(str).isin(target_datasets)
        & ~masked_df["sam_trainprep_masked_applied_v1"].fillna(False).astype(bool)
    ].copy()

    runtime: dict[str, object] = {}
    repair_rows: list[dict[str, object]] = []
    total = len(failure_df)
    for index, row in enumerate(failure_df.sort_values(["dataset", "split", "image_id"]).itertuples(index=False), start=1):
        fallback_rel, payload = _run_dataset_fallback_mask(
            repo_root=repo_root,
            output_dir=output_dir,
            runtime=runtime,
            dataset=str(row.dataset),
            original_path=str(row.original_rgb_path_v1),
            threshold=float(args.threshold),
            mask_threshold=float(args.mask_threshold),
            min_area_ratio=float(args.min_area_ratio),
            max_area_ratio=float(args.max_area_ratio),
            min_largest_component_ratio=float(args.min_largest_component_ratio),
            device=str(args.device),
            yolo_fallback_enabled=True,
            yolo_model_name=str(args.yolo_model),
            yolo_conf=float(args.yolo_conf),
            yolo_iou=float(args.yolo_iou),
            yolo_imgsz=int(args.yolo_imgsz),
            yolo_max_det=int(args.yolo_max_det),
            geometric_fallback_enabled=True,
        )
        repaired = row._asdict()
        repaired.update(payload)
        if fallback_rel is not None:
            repaired["sam_trainprep_masked_applied_v1"] = True
            repaired["sam_trainprep_masked_path_v1"] = str(fallback_rel)
        repair_rows.append(repaired)
        print(
            f"[repair_sam_failed_exports] repaired {index}/{total} | {row.dataset} | {row.split} | {row.image_id} | "
            f"applied={int(bool(repaired['sam_trainprep_masked_applied_v1']))} | "
            f"stage={repaired.get('sam_trainprep_masked_fallback_stage_v1', '')} | "
            f"reason={repaired.get('sam_trainprep_masked_reason_v1', '')}",
            flush=True,
        )
        pd.DataFrame(repair_rows).to_csv(tables_dir / "sam_trainprep_masked_repairs_v1.csv", index=False)

    repaired_masked_df = _merge_export_frames(masked_df, pd.DataFrame(repair_rows))

    applied_aligned_df = aligned_df[aligned_df["sam_trainprep_aligned_applied_v1"].fillna(False).astype(bool)].copy()
    new_aligned_df = build_trainprep_aligned_exports(
        repo_root=repo_root,
        output_dir=output_dir,
        masked_export_df=repaired_masked_df,
        existing_export_df=applied_aligned_df,
        min_foreground_pixels=int(args.alignment_min_foreground_pixels),
        min_area_ratio=float(args.alignment_min_area_ratio),
        max_area_ratio=float(args.alignment_max_area_ratio),
        padding_ratio=float(args.alignment_padding_ratio),
    )
    repaired_aligned_df = _merge_export_frames(applied_aligned_df, new_aligned_df)

    masked_lookup = repaired_masked_df.set_index(["image_id", "dataset", "split", "original_rgb_path_v1"])
    aligned_failure_df = pd.DataFrame()
    if not repaired_aligned_df.empty:
        aligned_failure_df = repaired_aligned_df[
            ~repaired_aligned_df["sam_trainprep_aligned_applied_v1"].fillna(False).astype(bool)
        ].copy()
        repaired_aligned_df = repaired_aligned_df[
            repaired_aligned_df["sam_trainprep_aligned_applied_v1"].fillna(False).astype(bool)
        ].copy()
    missing_aligned_keys = repaired_masked_df[
        repaired_masked_df["sam_trainprep_masked_applied_v1"].fillna(False).astype(bool)
    ][["image_id", "dataset", "split", "identity", "original_rgb_path_v1", "sam_trainprep_masked_path_v1"]].copy()
    if not repaired_aligned_df.empty:
        aligned_keys = repaired_aligned_df[["image_id", "dataset", "split", "original_rgb_path_v1"]].drop_duplicates()
        missing_aligned_keys = missing_aligned_keys.merge(
            aligned_keys,
            on=["image_id", "dataset", "split", "original_rgb_path_v1"],
            how="left",
            indicator=True,
        )
        missing_aligned_keys = missing_aligned_keys[missing_aligned_keys["_merge"] == "left_only"].drop(columns=["_merge"])
    if not aligned_failure_df.empty:
        missing_aligned_keys = pd.concat(
            [
                missing_aligned_keys,
                aligned_failure_df[["image_id", "dataset", "split", "identity", "original_rgb_path_v1"]].merge(
                    repaired_masked_df[["image_id", "dataset", "split", "original_rgb_path_v1", "sam_trainprep_masked_path_v1"]],
                    on=["image_id", "dataset", "split", "original_rgb_path_v1"],
                    how="left",
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(subset=["image_id", "dataset", "split", "original_rgb_path_v1"], keep="last")
    fallback_aligned_rows = []
    for row in missing_aligned_keys.itertuples(index=False):
        key = (str(row.image_id), str(row.dataset), str(row.split), str(row.original_rgb_path_v1))
        masked_row = masked_lookup.loc[key]
        fallback_aligned_rows.append(
            {
                "image_id": str(row.image_id),
                "dataset": str(row.dataset),
                "split": str(row.split),
                "identity": str(row.identity),
                "original_rgb_path_v1": str(row.original_rgb_path_v1),
                "sam_trainprep_aligned_applied_v1": True,
                "sam_trainprep_aligned_reason_v1": "fallback_to_masked_foreground",
                "sam_trainprep_aligned_path_v1": str(masked_row["sam_trainprep_masked_path_v1"]),
                "sam_trainprep_aligned_foreground_pixels_v1": 0.0,
                "sam_trainprep_aligned_foreground_area_ratio_v1": float(masked_row.get("sam_trainprep_masked_union_area_ratio_v1", 0.0) or 0.0),
                "sam_trainprep_aligned_axis_angle_deg_v1": 0.0,
                "sam_trainprep_aligned_axis_confidence_v1": 0.0,
                "sam_trainprep_aligned_rotation_applied_deg_v1": 0.0,
                "sam_trainprep_aligned_padding_ratio_v1": float(args.alignment_padding_ratio),
            }
        )
    repaired_aligned_df = _merge_export_frames(repaired_aligned_df, pd.DataFrame(fallback_aligned_rows))

    enriched_df = create_trainprep_enriched_metadata(
        base_df=base_df,
        masked_export_df=repaired_masked_df,
        aligned_export_df=repaired_aligned_df,
    )

    repaired_masked_df.to_csv(tables_dir / "sam_trainprep_masked_exports_v1.csv", index=False)
    repaired_aligned_df.to_csv(tables_dir / "sam_trainprep_aligned_exports_v1.csv", index=False)
    enriched_df.to_csv(tables_dir / "metadata_enriched_trainprep_v1.csv", index=False)
    if base_sam_root != output_dir:
        for relative in [
            "tables/sam_trainprep_masked_exports_v1.csv",
            "tables/sam_trainprep_aligned_exports_v1.csv",
        ]:
            source_path = base_sam_root / relative
            dest_path = output_dir / relative
            if not dest_path.exists() and source_path.exists():
                shutil.copy2(source_path, dest_path)

    manifest_rows: list[dict[str, object]] = []
    outputs: dict[str, Path] = {
        "metadata_path": tables_dir / "metadata_enriched_trainprep_v1.csv",
        "masked_export_path": tables_dir / "sam_trainprep_masked_exports_v1.csv",
        "aligned_export_path": tables_dir / "sam_trainprep_aligned_exports_v1.csv",
    }
    for split, view_name in [
        ("train", TRAIN_SINGLE_MASKED_VIEW_NAME),
        ("test", TRAIN_SINGLE_MASKED_VIEW_NAME),
        ("train", TRAIN_SINGLE_ALIGNED_VIEW_NAME),
        ("test", TRAIN_SINGLE_ALIGNED_VIEW_NAME),
    ]:
        manifest_df = build_trainprep_single_view_manifest(
            enriched_df=enriched_df,
            split=split,
            view_name=view_name,
            target_datasets=target_datasets,
        )
        manifest_path = tables_dir / f"manifest_{split}_{view_name}.csv"
        manifest_df.to_csv(manifest_path, index=False)
        manifest_rows.append(
            {"manifest": manifest_path.name, "split": split, "view_name": view_name, "rows": int(len(manifest_df)), "path": str(manifest_path)}
        )
    multiview_df = build_trainprep_multiview_manifest(enriched_df=enriched_df, target_datasets=target_datasets)
    multiview_path = tables_dir / f"manifest_train_{TRAIN_MULTIVIEW_NAME}.csv"
    multiview_df.to_csv(multiview_path, index=False)
    manifest_rows.append({"manifest": multiview_path.name, "split": "train", "view_name": TRAIN_MULTIVIEW_NAME, "rows": int(len(multiview_df)), "path": str(multiview_path)})

    masked_summary_df = _summarize_masked_exports(repaired_masked_df)
    aligned_summary_df = _summarize_aligned_exports(repaired_aligned_df)
    manifest_table_df = pd.DataFrame(manifest_rows).sort_values(["split", "manifest"]).reset_index(drop=True)
    config = {
        "source_manifest_root": str(source_manifest_root),
        "output_root": str(output_dir),
        "target_datasets": target_datasets,
        "enable_texas_fallback": True,
        "texas_fallback_threshold": float(args.threshold),
        "texas_fallback_mask_threshold": float(args.mask_threshold),
        "texas_fallback_min_area_ratio": float(args.min_area_ratio),
        "texas_fallback_max_area_ratio": float(args.max_area_ratio),
        "texas_fallback_min_largest_component_ratio": float(args.min_largest_component_ratio),
        "texas_fallback_device": str(args.device),
        "yolo_fallback_enabled": True,
        "yolo_fallback_model": str(args.yolo_model),
        "yolo_fallback_conf": float(args.yolo_conf),
        "yolo_fallback_iou": float(args.yolo_iou),
        "yolo_fallback_imgsz": int(args.yolo_imgsz),
        "yolo_fallback_max_det": int(args.yolo_max_det),
        "geometric_fallback_enabled": True,
        "alignment_min_foreground_pixels": int(args.alignment_min_foreground_pixels),
        "alignment_min_area_ratio": float(args.alignment_min_area_ratio),
        "alignment_max_area_ratio": float(args.alignment_max_area_ratio),
        "alignment_min_axis_confidence": 0.0,
        "alignment_min_axis_confidence_overrides": {},
        "alignment_padding_ratio": float(args.alignment_padding_ratio),
    }
    write_trainprep_summary(
        output_path=reports_dir / "summary.md",
        config=config,
        manifest_table_df=manifest_table_df,
        masked_summary_df=masked_summary_df,
        aligned_summary_df=aligned_summary_df,
    )
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[repair_sam_failed_exports] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
