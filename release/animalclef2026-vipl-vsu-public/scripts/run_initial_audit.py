#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.initial_audit import run_initial_audit

    parser = argparse.ArgumentParser(description="Run the initial AnimalCLEF data audit.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root containing metadata.csv and images/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "initial_audit",
        help="Directory where audit artifacts will be written.",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Skip exact duplicate hashing to speed up the audit.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for qualitative samples.",
    )
    args = parser.parse_args()

    outputs = run_initial_audit(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        compute_hashes=not args.skip_hashes,
        sample_seed=args.sample_seed,
    )

    print(f"[audit] report: {outputs['report_path']}")
    print(f"[audit] tables: {outputs['tables_dir']}")
    print(f"[audit] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

