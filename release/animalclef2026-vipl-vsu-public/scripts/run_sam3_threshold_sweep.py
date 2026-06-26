#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.sam3_threshold_sweep import run_sam3_threshold_sweep

    parser = argparse.ArgumentParser(description="Run a SAM3 threshold sweep on sampled images.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "sam3_threshold_sweep" / "v1",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SalamanderID2025", "SeaTurtleID2022", "TexasHornedLizards"],
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--mask-thresholds", nargs="+", type=float)
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    outputs = run_sam3_threshold_sweep(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        thresholds=args.thresholds,
        mask_thresholds=args.mask_thresholds,
        datasets=args.datasets,
        samples_per_split=args.samples_per_split,
        sample_seed=args.sample_seed,
        device=args.device,
    )
    print(f"[sam3_threshold_sweep] summary: {outputs['summary_md_path']}")
    print(f"[sam3_threshold_sweep] table: {outputs['summary_csv_path']}")
    print(f"[sam3_threshold_sweep] output_dir: {outputs['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
