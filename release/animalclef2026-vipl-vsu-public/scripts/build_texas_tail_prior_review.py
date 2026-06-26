#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_tail_prior_review import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_REVIEW_INDEX_PATH,
        run_texas_tail_prior_review,
    )

    parser = argparse.ArgumentParser(
        description="Build a Texas qualitative review pack for SAM mask, body alignment, tail-side prior, and black-pattern extraction."
    )
    parser.add_argument("--manifest-path", type=Path, default=repo_root / DEFAULT_MANIFEST_PATH)
    parser.add_argument("--review-index-path", type=Path, default=repo_root / DEFAULT_REVIEW_INDEX_PATH)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = run_texas_tail_prior_review(
        repo_root=repo_root,
        manifest_path=args.manifest_path,
        review_index_path=args.review_index_path,
        output_dir=args.output_dir,
    )
    print(f"[texas_tail_prior_review] summary: {outputs['summary_path']}")
    print(f"[texas_tail_prior_review] figures: {outputs['figures_dir']}")
    print(f"[texas_tail_prior_review] stats: {outputs['stat_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
