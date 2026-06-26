#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.sam_augmented_manifests import (
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_SOURCE_MANIFEST_ROOT,
        DEFAULT_TARGET_DATASETS,
        build_sam_augmented_manifests,
    )

    parser = argparse.ArgumentParser(description="Build SAM segmentation train-prep manifests with foreground fallback views.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--source-manifest-root", type=Path, default=repo_root / DEFAULT_SOURCE_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--target-datasets", nargs="*", default=list(DEFAULT_TARGET_DATASETS))
    parser.add_argument("--disable-texas-fallback", action="store_true")
    parser.add_argument("--texas-fallback-threshold", type=float, default=0.35)
    parser.add_argument("--texas-fallback-mask-threshold", type=float, default=0.30)
    parser.add_argument("--texas-fallback-min-area-ratio", type=float, default=0.005)
    parser.add_argument("--texas-fallback-max-area-ratio", type=float, default=0.98)
    parser.add_argument("--texas-fallback-min-largest-component-ratio", type=float, default=0.40)
    parser.add_argument("--texas-fallback-device", type=str, default="cuda:0")
    parser.add_argument("--disable-yolo-fallback", action="store_true")
    parser.add_argument("--yolo-fallback-model", type=str, default="yolov8s-worldv2.pt")
    parser.add_argument("--yolo-fallback-conf", type=float, default=0.05)
    parser.add_argument("--yolo-fallback-iou", type=float, default=0.50)
    parser.add_argument("--yolo-fallback-imgsz", type=int, default=640)
    parser.add_argument("--yolo-fallback-max-det", type=int, default=8)
    parser.add_argument("--disable-geometric-fallback", action="store_true")
    parser.add_argument("--alignment-min-foreground-pixels", type=int, default=256)
    parser.add_argument("--alignment-min-area-ratio", type=float, default=0.01)
    parser.add_argument("--alignment-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--alignment-padding-ratio", type=float, default=0.06)
    args = parser.parse_args()

    outputs = build_sam_augmented_manifests(
        repo_root=args.repo_root.resolve(),
        source_manifest_root=args.source_manifest_root.resolve(),
        output_dir=args.output_dir.resolve(),
        target_datasets=list(args.target_datasets),
        enable_texas_fallback=not bool(args.disable_texas_fallback),
        texas_fallback_threshold=float(args.texas_fallback_threshold),
        texas_fallback_mask_threshold=float(args.texas_fallback_mask_threshold),
        texas_fallback_min_area_ratio=float(args.texas_fallback_min_area_ratio),
        texas_fallback_max_area_ratio=float(args.texas_fallback_max_area_ratio),
        texas_fallback_min_largest_component_ratio=float(args.texas_fallback_min_largest_component_ratio),
        texas_fallback_device=str(args.texas_fallback_device),
        yolo_fallback_enabled=not bool(args.disable_yolo_fallback),
        yolo_fallback_model=str(args.yolo_fallback_model),
        yolo_fallback_conf=float(args.yolo_fallback_conf),
        yolo_fallback_iou=float(args.yolo_fallback_iou),
        yolo_fallback_imgsz=int(args.yolo_fallback_imgsz),
        yolo_fallback_max_det=int(args.yolo_fallback_max_det),
        geometric_fallback_enabled=not bool(args.disable_geometric_fallback),
        alignment_min_foreground_pixels=int(args.alignment_min_foreground_pixels),
        alignment_min_area_ratio=float(args.alignment_min_area_ratio),
        alignment_max_area_ratio=float(args.alignment_max_area_ratio),
        alignment_padding_ratio=float(args.alignment_padding_ratio),
    )
    print(f"[sam_augmented_manifests] summary: {outputs['summary_path']}")
    print(f"[sam_augmented_manifests] metadata: {outputs['metadata_path']}")
    print(f"[sam_augmented_manifests] multiview_train: {outputs['train_sam_trainprep_multiview_v1_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
