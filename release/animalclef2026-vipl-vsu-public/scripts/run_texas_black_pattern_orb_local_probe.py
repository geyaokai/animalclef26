#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_black_pattern_orb_local import (
        DEFAULT_BAND_DILATE_RADIUS,
        DEFAULT_BAND_ERODE_RADIUS,
        DEFAULT_BAND_MIN_PIXELS,
        DEFAULT_BLACK_MAX_QUANTILE,
        DEFAULT_BLACK_MIN_PIXELS,
        DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT,
        DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE,
        DEFAULT_BLACK_QUANTILE,
        DEFAULT_CORE_ERODE_RATIO,
        DEFAULT_CORE_MIN_PIXELS,
        DEFAULT_ENABLE_SAM_FALLBACK,
        DEFAULT_FAST_THRESHOLD,
        DEFAULT_MAX_SIDE,
        DEFAULT_MIN_INLIERS,
        DEFAULT_NFEATURES,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_ORB_BLACK_DILATE_RADIUS,
        DEFAULT_PAIR_CSV_PATH,
        DEFAULT_RATIO_TEST,
        DEFAULT_RANSAC_THRESHOLD,
        DEFAULT_SAM_FALLBACK_DEVICE,
        DEFAULT_SAM_FALLBACK_MASK_THRESHOLD,
        DEFAULT_SAM_FALLBACK_MAX_AREA_RATIO,
        DEFAULT_SAM_FALLBACK_MIN_AREA_RATIO,
        DEFAULT_SAM_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
        DEFAULT_SAM_FALLBACK_THRESHOLD,
        DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
        DEFAULT_CLAHE_CLIP_LIMIT,
        DEFAULT_ORB_MASK_MIN_PIXELS,
        run_texas_black_pattern_orb_local_probe,
    )

    parser = argparse.ArgumentParser(description="Run Texas black-pattern ORB local probe.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_miew_pseudo_v1",
    )
    parser.add_argument("--pair-csv", type=Path, default=repo_root / DEFAULT_PAIR_CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--view-manifest-path", type=Path, default=repo_root / DEFAULT_TEXAS_VIEW_MANIFEST_PATH)
    parser.add_argument("--score-column", type=str)
    parser.add_argument("--nfeatures", type=int, default=DEFAULT_NFEATURES)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument("--fast-threshold", type=int, default=DEFAULT_FAST_THRESHOLD)
    parser.add_argument("--clahe-clip-limit", type=float, default=DEFAULT_CLAHE_CLIP_LIMIT)
    parser.add_argument("--ratio-test", type=float, default=DEFAULT_RATIO_TEST)
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--core-erode-ratio", type=float, default=DEFAULT_CORE_ERODE_RATIO)
    parser.add_argument("--core-min-pixels", type=int, default=DEFAULT_CORE_MIN_PIXELS)
    parser.add_argument("--black-quantile", type=float, default=DEFAULT_BLACK_QUANTILE)
    parser.add_argument("--black-max-quantile", type=float, default=DEFAULT_BLACK_MAX_QUANTILE)
    parser.add_argument("--black-min-pixels", type=int, default=DEFAULT_BLACK_MIN_PIXELS)
    parser.add_argument("--black-prior-clahe-clip-limit", type=float, default=DEFAULT_BLACK_PRIOR_CLAHE_CLIP_LIMIT)
    parser.add_argument("--black-prior-clahe-grid-size", type=int, default=DEFAULT_BLACK_PRIOR_CLAHE_GRID_SIZE)
    parser.add_argument("--band-dilate-radius", type=int, default=DEFAULT_BAND_DILATE_RADIUS)
    parser.add_argument("--band-erode-radius", type=int, default=DEFAULT_BAND_ERODE_RADIUS)
    parser.add_argument("--band-min-pixels", type=int, default=DEFAULT_BAND_MIN_PIXELS)
    parser.add_argument("--orb-mask-min-pixels", type=int, default=DEFAULT_ORB_MASK_MIN_PIXELS)
    parser.add_argument("--orb-black-dilate-radius", type=int, default=DEFAULT_ORB_BLACK_DILATE_RADIUS)
    parser.add_argument("--disable-sam-fallback", action="store_true")
    parser.add_argument("--sam-fallback-threshold", type=float, default=DEFAULT_SAM_FALLBACK_THRESHOLD)
    parser.add_argument("--sam-fallback-mask-threshold", type=float, default=DEFAULT_SAM_FALLBACK_MASK_THRESHOLD)
    parser.add_argument("--sam-fallback-min-area-ratio", type=float, default=DEFAULT_SAM_FALLBACK_MIN_AREA_RATIO)
    parser.add_argument("--sam-fallback-max-area-ratio", type=float, default=DEFAULT_SAM_FALLBACK_MAX_AREA_RATIO)
    parser.add_argument(
        "--sam-fallback-min-largest-component-ratio",
        type=float,
        default=DEFAULT_SAM_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
    )
    parser.add_argument("--sam-fallback-device", type=str, default=DEFAULT_SAM_FALLBACK_DEVICE)
    parser.add_argument(
        "--processing-max-side",
        type=int,
        default=0,
        help="预对齐阶段是否先缩放 masked RGB；0 表示关闭，保留原始分辨率。",
    )
    args = parser.parse_args()

    outputs = run_texas_black_pattern_orb_local_probe(
        repo_root=repo_root,
        predictions_path=args.predictions_path,
        pair_csv_path=args.pair_csv,
        output_dir=args.output_dir,
        view_manifest_path=args.view_manifest_path,
        score_column=args.score_column,
        nfeatures=int(args.nfeatures),
        max_side=int(args.max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        ratio_test=float(args.ratio_test),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
        core_erode_ratio=float(args.core_erode_ratio),
        core_min_pixels=int(args.core_min_pixels),
        black_quantile=float(args.black_quantile),
        black_max_quantile=float(args.black_max_quantile),
        black_min_pixels=int(args.black_min_pixels),
        black_prior_clahe_clip_limit=float(args.black_prior_clahe_clip_limit),
        black_prior_clahe_grid_size=int(args.black_prior_clahe_grid_size),
        band_dilate_radius=int(args.band_dilate_radius),
        band_erode_radius=int(args.band_erode_radius),
        band_min_pixels=int(args.band_min_pixels),
        orb_mask_min_pixels=int(args.orb_mask_min_pixels),
        orb_black_dilate_radius=int(args.orb_black_dilate_radius),
        enable_sam_fallback=(False if args.disable_sam_fallback else bool(DEFAULT_ENABLE_SAM_FALLBACK)),
        sam_fallback_threshold=float(args.sam_fallback_threshold),
        sam_fallback_mask_threshold=float(args.sam_fallback_mask_threshold),
        sam_fallback_min_area_ratio=float(args.sam_fallback_min_area_ratio),
        sam_fallback_max_area_ratio=float(args.sam_fallback_max_area_ratio),
        sam_fallback_min_largest_component_ratio=float(args.sam_fallback_min_largest_component_ratio),
        sam_fallback_device=str(args.sam_fallback_device),
        processing_max_side=(None if int(args.processing_max_side) <= 0 else int(args.processing_max_side)),
    )
    print(f"[texas_black_pattern_orb_local] local_table: {outputs['local_table_path']}")
    print(f"[texas_black_pattern_orb_local] image_stats: {outputs['image_stats_path']}")
    print(f"[texas_black_pattern_orb_local] summary: {outputs['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
