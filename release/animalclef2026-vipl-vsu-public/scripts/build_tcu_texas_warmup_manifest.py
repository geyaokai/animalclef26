#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_external_warmup import build_tcu_texas_warmup_manifest

    parser = argparse.ArgumentParser(description="Build a Texas warmup manifest from the audited TCU chip table.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--chip-manifest-path",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "tcu_texas_dataset_v1" / "tables" / "tcu_texas_chip_manifest_v1.csv",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=repo_root
        / "artifacts"
        / "training"
        / "cache"
        / "tcu_texas_warmup_manifest_v1"
        / "tables"
        / "tcu_texas_warmup_manifest_v1.csv",
    )
    args = parser.parse_args()

    manifest_df = build_tcu_texas_warmup_manifest(
        repo_root=args.repo_root.resolve(),
        chip_manifest_path=args.chip_manifest_path.resolve(),
        output_path=args.output_path.resolve(),
    )
    print(f"[tcu_texas_warmup_manifest] output: {args.output_path.resolve()}")
    print(f"[tcu_texas_warmup_manifest] rows: {len(manifest_df)}")
    print(f"[tcu_texas_warmup_manifest] classes: {manifest_df['identity'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
