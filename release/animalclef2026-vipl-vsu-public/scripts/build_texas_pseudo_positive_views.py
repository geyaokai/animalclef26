#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_pseudo_positive_views import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TEXAS_CENTER_BODY_MANIFEST,
        build_texas_pseudo_positive_views,
    )

    parser = argparse.ArgumentParser(description="Build Texas pseudo-positive base-vs-positive view metadata.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--trusted-membership-path", type=Path, required=True)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=repo_root / DEFAULT_TEXAS_CENTER_BODY_MANIFEST,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args()

    outputs = build_texas_pseudo_positive_views(
        trusted_membership_path=args.trusted_membership_path.resolve(),
        manifest_path=args.manifest_path.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"[texas_pseudo_positive_views] views: {outputs['views_path']}")
    print(f"[texas_pseudo_positive_views] pairs: {outputs['pairs_path']}")
    print(f"[texas_pseudo_positive_views] summary: {outputs['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

