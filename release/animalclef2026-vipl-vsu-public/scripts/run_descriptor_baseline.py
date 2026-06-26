#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import DESCRIPTOR_SPECS, run_descriptor_baseline

    parser = argparse.ArgumentParser(description="Run a frozen descriptor clustering baseline.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where artifacts for this descriptor baseline will be written.",
    )
    parser.add_argument(
        "--descriptor",
        choices=sorted(DESCRIPTOR_SPECS),
        required=True,
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--train-manifest-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    args = parser.parse_args()

    outputs = run_descriptor_baseline(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        descriptor=args.descriptor,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_identity_fraction=args.val_identity_fraction,
        thresholds=args.thresholds,
        split_seed=args.split_seed,
        train_manifest_path=args.train_manifest_path.resolve() if args.train_manifest_path else None,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
    )
    print(f"[descriptor_baseline] summary: {outputs['summary_path']}")
    print(f"[descriptor_baseline] submission: {outputs['submission_path']}")
    print(f"[descriptor_baseline] threshold_sweep: {outputs['threshold_sweep_path']}")
    print(f"[descriptor_baseline] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
