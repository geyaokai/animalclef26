#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_orb_qualitative_review import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TEXAS_JUDGMENTS_PATH,
        DEFAULT_TEXAS_REVIEW_DIR,
        run_texas_orb_qualitative_review,
    )

    parser = argparse.ArgumentParser(description="Build a qualitative Texas ORB review pack from judged pairs.")
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_miew_pseudo_v1",
    )
    parser.add_argument("--review-dir", type=Path, default=repo_root / DEFAULT_TEXAS_REVIEW_DIR)
    parser.add_argument("--pair-judgments-path", type=Path, default=repo_root / DEFAULT_TEXAS_JUDGMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-per-category", type=int, default=12)
    parser.add_argument("--nfeatures", type=int, default=2048)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--fast-threshold", type=int, default=12)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    args = parser.parse_args()

    outputs = run_texas_orb_qualitative_review(
        repo_root=repo_root,
        predictions_path=args.predictions_path,
        review_dir=args.review_dir,
        pair_judgments_path=args.pair_judgments_path,
        output_dir=args.output_dir,
        top_k_per_category=int(args.top_k_per_category),
        nfeatures=int(args.nfeatures),
        max_side=int(args.max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
    )
    print(f"[texas_orb_qualitative_review] summary: {outputs['summary_path']}")
    print(f"[texas_orb_qualitative_review] judged_pairs: {outputs['judged_pairs_path']}")
    print(f"[texas_orb_qualitative_review] label_summary: {outputs['label_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
