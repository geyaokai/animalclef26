#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    workspace_root = repo_root.parent
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.tcu_texas_dataset import build_tcu_texas_dataset_artifacts

    parser = argparse.ArgumentParser(description="Build TCU Texas chip/original manifests plus alignment audits.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--chips-dir",
        type=Path,
        default=workspace_root / "external" / "datasets" / "tcu_texas_horned_lizard_v1" / "extracted" / "8. THL images - Chips",
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=workspace_root / "external" / "datasets" / "tcu_texas_horned_lizard_v1" / "extracted" / "7. THL images - Original",
    )
    parser.add_argument(
        "--mapping-path",
        type=Path,
        default=workspace_root / "external" / "datasets" / "tcu_texas_horned_lizard_v1" / "raw" / "5._HotSpotter_Chip_ID_and_Corresponding_Image_ID.csv",
    )
    parser.add_argument(
        "--hotspotter-output-path",
        type=Path,
        default=workspace_root / "external" / "datasets" / "tcu_texas_horned_lizard_v1" / "raw" / "6._Output_from_HotSpotter.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "tcu_texas_dataset_v1",
    )
    args = parser.parse_args()

    artifacts = build_tcu_texas_dataset_artifacts(
        repo_root=args.repo_root.resolve(),
        chips_dir=args.chips_dir.resolve(),
        original_dir=args.original_dir.resolve(),
        mapping_path=args.mapping_path.resolve(),
        hotspotter_output_path=args.hotspotter_output_path.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"[tcu_texas_dataset] chip_manifest: {artifacts.chip_manifest_path}")
    print(f"[tcu_texas_dataset] original_manifest: {artifacts.original_manifest_path}")
    print(f"[tcu_texas_dataset] chip_alignment_audit: {artifacts.chip_alignment_audit_path}")
    print(f"[tcu_texas_dataset] original_coverage_audit: {artifacts.original_coverage_audit_path}")
    print(f"[tcu_texas_dataset] summary: {artifacts.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
