#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_center_body_manifest import build_texas_center_body_manifest

    parser = argparse.ArgumentParser(description="Build the formal Texas center-body gray test manifest.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--records-path",
        type=Path,
        default=repo_root
        / "artifacts"
        / "preprocessing_qualitative"
        / "texas_center_body_square_repaired_v1"
        / "tables"
        / "texas_center_body_square_records_v1.csv",
    )
    parser.add_argument(
        "--repaired-manifest-path",
        type=Path,
        default=repo_root
        / "artifacts"
        / "manifests"
        / "sam_seg_trainprep_repaired_v1"
        / "tables"
        / "manifest_test_sam_trainprep_aligned_best_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "manifests" / "texas_center_body_square_repaired_v1",
    )
    args = parser.parse_args()

    outputs = build_texas_center_body_manifest(
        repo_root=args.repo_root.resolve(),
        records_path=args.records_path.resolve(),
        repaired_manifest_path=args.repaired_manifest_path.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"[texas_center_body_manifest] manifest: {outputs['manifest_path']}")
    print(f"[texas_center_body_manifest] summary: {outputs['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
