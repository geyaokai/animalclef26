#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.salamander_aligned_crop_review import (
        DEFAULT_MANIFEST_PATH,
        DEFAULT_OUTPUT_DIR,
        run_salamander_aligned_crop_review,
    )

    parser = argparse.ArgumentParser(description="Build a qualitative review pack for Salamander aligned local crop candidates.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--dataset", type=str, default="SalamanderID2025")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    outputs = run_salamander_aligned_crop_review(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        dataset=args.dataset,
        split=args.split,
        sample_count=args.sample_count,
        sample_seed=args.sample_seed,
    )
    print(f"[salamander_aligned_crop_review] summary: {outputs['summary_path']}")
    print(f"[salamander_aligned_crop_review] sampled_rows: {outputs['sampled_rows_path']}")
    print(f"[salamander_aligned_crop_review] qualitative_dir: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
