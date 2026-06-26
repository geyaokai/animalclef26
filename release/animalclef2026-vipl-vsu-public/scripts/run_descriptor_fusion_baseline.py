#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import run_descriptor_fusion_baseline

    parser = argparse.ArgumentParser(description="Fuse cached descriptor embeddings and run the clustering baseline.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where fused artifacts will be written.",
    )
    parser.add_argument(
        "--source-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Baseline artifact directories that contain embeddings/val_embeddings.npy and embeddings/test_embeddings.npy",
    )
    parser.add_argument(
        "--component-names",
        nargs="+",
        default=None,
        help="Optional readable names for the fused components, aligned with --source-dirs",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional per-component weights, aligned with --source-dirs",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--fusion-mode", choices=["concat_l2"], default="concat_l2")
    args = parser.parse_args()

    outputs = run_descriptor_fusion_baseline(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        source_dirs=[path.resolve() for path in args.source_dirs],
        component_names=args.component_names,
        weights=args.weights,
        thresholds=args.thresholds,
        split_seed=args.split_seed,
        fusion_mode=args.fusion_mode,
    )
    print(f"[descriptor_fusion] summary: {outputs['summary_path']}")
    print(f"[descriptor_fusion] submission: {outputs['submission_path']}")
    print(f"[descriptor_fusion] threshold_sweep: {outputs['threshold_sweep_path']}")
    print(f"[descriptor_fusion] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
