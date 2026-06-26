#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_seed_review import build_texas_seed_review_package

    parser = argparse.ArgumentParser(description="Build an audit-friendly Texas pseudo seed review package.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--pseudo-cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    pseudo_cache_dir = args.pseudo_cache_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else pseudo_cache_dir / "review_package"
    outputs = build_texas_seed_review_package(
        repo_root=args.repo_root.resolve(),
        assignments_path=pseudo_cache_dir / "tables" / "all_assignments_v1.csv",
        candidate_pairs_path=pseudo_cache_dir / "tables" / "candidate_pairs_v1.csv",
        manifest_path=args.manifest_path.resolve(),
        output_dir=output_dir,
    )
    print(f"[texas_seed_review] summary: {outputs['summary_path']}")
    print(f"[texas_seed_review] pseudo_assignments: {outputs['pseudo_assignments_path']}")
    print(f"[texas_seed_review] pseudo_manifest: {outputs['pseudo_manifest_path']}")
    print(f"[texas_seed_review] seed_class_summary: {outputs['seed_class_summary_path']}")
    print(f"[texas_seed_review] contact_sheet_index: {outputs['contact_sheet_index_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
