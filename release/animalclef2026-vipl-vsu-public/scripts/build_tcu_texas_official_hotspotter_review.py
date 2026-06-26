#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.tcu_texas_official_hotspotter_review import (
        DEFAULT_OUTPUT_DIR,
        DEFAULT_TCU_TEXAS_CHIP_MANIFEST_PATH,
        build_tcu_texas_official_hotspotter_review,
    )

    parser = argparse.ArgumentParser(
        description="Build a manual review package for the official TCU Texas HotSpotter top1 results."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--chip-manifest-path", type=Path, default=DEFAULT_TCU_TEXAS_CHIP_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    outputs = build_tcu_texas_official_hotspotter_review(
        repo_root=args.repo_root.resolve(),
        chip_manifest_path=args.chip_manifest_path,
        output_dir=args.output_dir,
    )
    print(f"[tcu_texas_official_review] summary: {outputs['summary_path']}")
    print(f"[tcu_texas_official_review] review_pairs: {outputs['review_pairs_path']}")
    print(f"[tcu_texas_official_review] judgment_template: {outputs['judgment_template_path']}")
    print(f"[tcu_texas_official_review] overview: {outputs['overview_path']}")
    print(f"[tcu_texas_official_review] pair_board_dir: {outputs['pair_board_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
