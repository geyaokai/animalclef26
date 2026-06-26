#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.preprocessing_baselines import build_preprocessing_baselines

    parser = argparse.ArgumentParser(description="Build reversible preprocessing baselines.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root containing metadata.csv.",
    )
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=repo_root / "artifacts" / "initial_audit",
        help="Directory containing initial audit outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "preprocessing_baselines" / "v1",
        help="Directory where baseline preprocessing artifacts will be written.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for qualitative preview sampling.",
    )
    args = parser.parse_args()

    outputs = build_preprocessing_baselines(
        repo_root=args.repo_root.resolve(),
        audit_dir=args.audit_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        sample_seed=args.sample_seed,
    )
    print(f"[prep] summary: {outputs['summary_path']}")
    print(f"[prep] enriched metadata: {outputs['enriched_metadata_path']}")
    print(f"[prep] preview: {outputs['preview_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

