#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.sam3_probe import run_sam3_probe

    parser = argparse.ArgumentParser(description="Run a sampled SAM3 segmentation probe.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "sam3_probe" / "v1",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"],
    )
    parser.add_argument("--samples-per-split", type=int, default=6)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    outputs = run_sam3_probe(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        datasets=args.datasets,
        samples_per_split=args.samples_per_split,
        sample_seed=args.sample_seed,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
    )
    print(f"[sam3_probe] summary: {outputs['summary_path']}")
    print(f"[sam3_probe] results: {outputs['results_path']}")
    print(f"[sam3_probe] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

