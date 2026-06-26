#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_orb_local_probe import (
        DEFAULT_TEXAS_ORB_OUTPUT_DIR,
        DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
        run_texas_orb_local_probe,
    )

    parser = argparse.ArgumentParser(description="Run Texas ORB local probe on current review/self-train pairs.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_miew_pseudo_v1",
    )
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "texas_selftrain_review_v1" / "tables" / "test_pair_disagreement_v1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_TEXAS_ORB_OUTPUT_DIR)
    parser.add_argument("--view-manifest-path", type=Path, default=repo_root / DEFAULT_TEXAS_VIEW_MANIFEST_PATH)
    parser.add_argument("--score-column", type=str)
    parser.add_argument("--nfeatures", type=int, default=2048)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--fast-threshold", type=int, default=12)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--ratio-test", type=float, default=0.85)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=4)
    args = parser.parse_args()

    outputs = run_texas_orb_local_probe(
        repo_root=repo_root,
        predictions_path=args.predictions_path.resolve() if args.predictions_path.is_absolute() else args.predictions_path,
        pair_csv_path=args.pair_csv.resolve() if args.pair_csv.is_absolute() else args.pair_csv,
        output_dir=args.output_dir.resolve() if args.output_dir.is_absolute() else args.output_dir,
        view_manifest_path=(
            args.view_manifest_path.resolve() if args.view_manifest_path.is_absolute() else args.view_manifest_path
        ),
        score_column=args.score_column,
        nfeatures=int(args.nfeatures),
        max_side=int(args.max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        ratio_test=float(args.ratio_test),
        ransac_threshold=float(args.ransac_threshold),
        min_inliers=int(args.min_inliers),
    )
    print(f"[texas_orb_local_probe] local_table: {outputs['local_table_path']}")
    print(f"[texas_orb_local_probe] normalized_pairs: {outputs['normalized_pair_path']}")
    print(f"[texas_orb_local_probe] summary: {outputs['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
