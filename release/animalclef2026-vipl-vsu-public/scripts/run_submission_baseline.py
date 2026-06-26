#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.submission_baseline import run_submission_baseline

    parser = argparse.ArgumentParser(description="Build the current Kaggle submission baseline.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--fusion-source-dir", type=Path)
    parser.add_argument("--texas-source-dir", type=Path)
    parser.add_argument("--texas-threshold", type=float, default=0.44)
    parser.add_argument("--orb-source-dir", type=Path)
    parser.add_argument("--lynx-checkpoint-path", type=Path)
    parser.add_argument("--lynx-threshold-table-path", type=Path)
    args = parser.parse_args()

    outputs = run_submission_baseline(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        device=args.device,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        sample_submission_path=args.sample_submission_path.resolve() if args.sample_submission_path else None,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
        fusion_source_dir=args.fusion_source_dir.resolve() if args.fusion_source_dir else None,
        texas_source_dir=args.texas_source_dir.resolve() if args.texas_source_dir else None,
        texas_threshold=float(args.texas_threshold),
        orb_source_dir=args.orb_source_dir.resolve() if args.orb_source_dir else None,
        lynx_checkpoint_path=args.lynx_checkpoint_path.resolve() if args.lynx_checkpoint_path else None,
        lynx_threshold_table_path=args.lynx_threshold_table_path.resolve() if args.lynx_threshold_table_path else None,
    )
    print(f"[submission_baseline] submission: {outputs['submission_path']}")
    print(f"[submission_baseline] summary: {outputs['summary_path']}")
    print(f"[submission_baseline] predictions: {outputs['prediction_path']}")
    print(f"[submission_baseline] route_config: {outputs['route_config_path']}")
    print(f"[submission_baseline] cluster_summary: {outputs['cluster_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
