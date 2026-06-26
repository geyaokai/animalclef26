#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.view_manifests import (
        BODY_AXIS_VIEW_NAME,
        DEFAULT_BODY_AXIS_DATASETS,
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_SAM_MASKED_DATASETS,
        SAM_MASKED_VIEW_NAME,
        build_view_manifests,
    )

    parser = argparse.ArgumentParser(description="Build unified train/test manifests with optional body-axis view exports.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / DEFAULT_MANIFEST_ROOT,
        help="Directory where unified manifests and body-axis exports will be written.",
    )
    parser.add_argument(
        "--body-axis-datasets",
        nargs="*",
        default=list(DEFAULT_BODY_AXIS_DATASETS),
        help=f"Datasets that should export `{BODY_AXIS_VIEW_NAME}`. Pass no values to disable export.",
    )
    parser.add_argument(
        "--sam-masked-datasets",
        nargs="*",
        default=list(DEFAULT_SAM_MASKED_DATASETS),
        help=f"Datasets that should export `{SAM_MASKED_VIEW_NAME}`. Pass no values to disable export.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-foreground-pixels", type=int, default=1024)
    parser.add_argument("--min-area-ratio", type=float, default=0.015)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--min-axis-confidence", type=float, default=0.35)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.8)
    parser.add_argument("--sam-min-area-ratio", type=float, default=0.01)
    parser.add_argument("--sam-max-area-ratio", type=float, default=0.95)
    parser.add_argument("--sam-min-largest-component-ratio", type=float, default=0.7)
    args = parser.parse_args()

    outputs = build_view_manifests(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        body_axis_datasets=args.body_axis_datasets,
        sam_masked_datasets=args.sam_masked_datasets,
        device=args.device,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        min_foreground_pixels=args.min_foreground_pixels,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_axis_confidence=args.min_axis_confidence,
        min_largest_component_ratio=args.min_largest_component_ratio,
        sam_min_area_ratio=args.sam_min_area_ratio,
        sam_max_area_ratio=args.sam_max_area_ratio,
        sam_min_largest_component_ratio=args.sam_min_largest_component_ratio,
    )
    print(f"[build_view_manifests] summary: {outputs['summary_path']}")
    print(f"[build_view_manifests] metadata: {outputs['metadata_enriched_path']}")
    print(f"[build_view_manifests] default_train: {outputs['default_train_manifest_path']}")
    print(f"[build_view_manifests] default_test: {outputs['default_test_manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
