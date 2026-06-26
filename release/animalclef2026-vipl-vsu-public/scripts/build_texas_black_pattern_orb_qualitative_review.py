#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_black_pattern_orb_qualitative_review import (
        DEFAULT_LOCAL_PROBE_DIR,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAIR_JUDGMENTS_PATH,
        DEFAULT_REVIEW_SOURCE_DIR,
        DEFAULT_TOP_K_PER_CATEGORY,
        run_texas_black_pattern_orb_qualitative_review,
    )

    parser = argparse.ArgumentParser(description="Build Texas black-pattern ORB qualitative review.")
    parser.add_argument("--local-probe-dir", type=Path, default=repo_root / DEFAULT_LOCAL_PROBE_DIR)
    parser.add_argument("--review-source-dir", type=Path, default=repo_root / DEFAULT_REVIEW_SOURCE_DIR)
    parser.add_argument("--pair-judgments-path", type=Path, default=repo_root / DEFAULT_PAIR_JUDGMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k-per-category", type=int, default=DEFAULT_TOP_K_PER_CATEGORY)
    args = parser.parse_args()

    outputs = run_texas_black_pattern_orb_qualitative_review(
        repo_root=repo_root,
        local_probe_dir=args.local_probe_dir,
        review_source_dir=args.review_source_dir,
        pair_judgments_path=args.pair_judgments_path,
        output_dir=args.output_dir,
        top_k_per_category=int(args.top_k_per_category),
    )
    print(f"[texas_black_pattern_orb_review] summary: {outputs['summary_path']}")
    print(f"[texas_black_pattern_orb_review] judged_pairs: {outputs['judged_pairs_path']}")
    print(f"[texas_black_pattern_orb_review] label_summary: {outputs['label_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
