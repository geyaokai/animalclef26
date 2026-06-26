#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.salamander_local_graph import (
        DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE,
        DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS,
        DEFAULT_BASE_PREDICTIONS,
        DEFAULT_CLAHE_CLIP_LIMIT,
        DEFAULT_FAST_THRESHOLD,
        DEFAULT_HARD_VETO_SCORE_CAP,
        DEFAULT_LOCAL_MATCHER,
        DEFAULT_MANIFEST_ROOT,
        DEFAULT_MIN_INLIERS,
        DEFAULT_ORB_FEATURES,
        DEFAULT_ORB_MAX_SIDE,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_RANSAC_THRESHOLD,
        DEFAULT_RATIO_TEST,
        DEFAULT_ROUTE_DIR,
        DEFAULT_ROUTE_NAME,
        DEFAULT_SOFT_VETO_SCORE_SCALE,
        DEFAULT_STRONG_THRESHOLDS,
        DEFAULT_TOP_K,
        DEFAULT_WEAK_ATTACH_MIN_SUPPORT,
        DEFAULT_WEAK_MIN_SHARED_NEIGHBORS,
        DEFAULT_WEAK_THRESHOLDS,
        run_salamander_local_graph_submission,
    )

    parser = argparse.ArgumentParser(description="Run the Salamander top-K recall + local graph submission chain.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--route-dir", type=Path, default=repo_root / DEFAULT_ROUTE_DIR)
    parser.add_argument("--base-predictions-path", type=Path, default=repo_root / DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--manifest-root", type=Path, default=repo_root / DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--route-name", type=str, default=DEFAULT_ROUTE_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--strong-thresholds", nargs="+", type=float, default=DEFAULT_STRONG_THRESHOLDS)
    parser.add_argument("--weak-thresholds", nargs="+", type=float, default=DEFAULT_WEAK_THRESHOLDS)
    parser.add_argument("--weak-attach-min-support", type=int, default=DEFAULT_WEAK_ATTACH_MIN_SUPPORT)
    parser.add_argument("--weak-min-shared-neighbors", type=int, default=DEFAULT_WEAK_MIN_SHARED_NEIGHBORS)
    parser.add_argument("--orb-features", type=int, default=DEFAULT_ORB_FEATURES)
    parser.add_argument("--orb-max-side", type=int, default=DEFAULT_ORB_MAX_SIDE)
    parser.add_argument("--fast-threshold", type=int, default=DEFAULT_FAST_THRESHOLD)
    parser.add_argument("--clahe-clip-limit", type=float, default=DEFAULT_CLAHE_CLIP_LIMIT)
    parser.add_argument("--ratio-test", type=float, default=DEFAULT_RATIO_TEST)
    parser.add_argument("--ransac-threshold", type=float, default=DEFAULT_RANSAC_THRESHOLD)
    parser.add_argument("--min-inliers", type=int, default=DEFAULT_MIN_INLIERS)
    parser.add_argument("--local-matcher", type=str, default=DEFAULT_LOCAL_MATCHER)
    parser.add_argument("--alignment-min-foreground-pixels", type=int, default=DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS)
    parser.add_argument("--alignment-min-axis-confidence", type=float, default=DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE)
    parser.add_argument("--soft-veto-score-scale", type=float, default=DEFAULT_SOFT_VETO_SCORE_SCALE)
    parser.add_argument("--hard-veto-score-cap", type=float, default=DEFAULT_HARD_VETO_SCORE_CAP)
    args = parser.parse_args()

    outputs = run_salamander_local_graph_submission(
        repo_root=args.repo_root.resolve(),
        route_dir=args.route_dir.resolve(),
        base_predictions_path=args.base_predictions_path.resolve(),
        sample_submission_path=args.sample_submission_path.resolve(),
        manifest_root=args.manifest_root.resolve(),
        output_dir=args.output_dir.resolve(),
        route_name=args.route_name,
        top_k=args.top_k,
        strong_thresholds=[float(value) for value in args.strong_thresholds],
        weak_thresholds=[float(value) for value in args.weak_thresholds],
        weak_attach_min_support=args.weak_attach_min_support,
        weak_min_shared_neighbors=args.weak_min_shared_neighbors,
        orb_features=args.orb_features,
        orb_max_side=args.orb_max_side,
        fast_threshold=args.fast_threshold,
        clahe_clip_limit=args.clahe_clip_limit,
        ratio_test=args.ratio_test,
        ransac_threshold=args.ransac_threshold,
        min_inliers=args.min_inliers,
        local_matcher=args.local_matcher,
        alignment_min_foreground_pixels=args.alignment_min_foreground_pixels,
        alignment_min_axis_confidence=args.alignment_min_axis_confidence,
        soft_veto_score_scale=args.soft_veto_score_scale,
        hard_veto_score_cap=args.hard_veto_score_cap,
    )
    print(f"[salamander_local_graph] submission: {outputs['submission_path']}")
    print(f"[salamander_local_graph] predictions: {outputs['test_predictions_path']}")
    print(f"[salamander_local_graph] salamander_override: {outputs['salamander_predictions_path']}")
    print(f"[salamander_local_graph] summary: {outputs['summary_path']}")
    print(f"[salamander_local_graph] threshold_sweep: {outputs['threshold_sweep_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
