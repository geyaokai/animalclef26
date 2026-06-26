#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.orb_rerank_baseline import run_orb_rerank_baseline

    parser = argparse.ArgumentParser(description="Run ORB-based local rerank on top of cached fusion embeddings.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "artifacts" / "descriptor_baselines" / "embed_fusion_v1",
        help="Baseline artifact directory that contains embeddings/val_embeddings.npy and embeddings/val_metadata.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where rerank artifacts will be written.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--smoke-max-queries", type=int, default=40)
    parser.add_argument("--smoke-seed", type=int, default=42)
    parser.add_argument("--orb-features", type=int, default=1024)
    parser.add_argument("--orb-max-side", type=int, default=768)
    parser.add_argument("--fast-threshold", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--ratio-test", type=float, default=0.8)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=8)
    parser.add_argument("--local-weights", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    args = parser.parse_args()

    outputs = run_orb_rerank_baseline(
        repo_root=args.repo_root.resolve(),
        source_dir=args.source_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        top_k=args.top_k,
        smoke_max_queries=args.smoke_max_queries,
        smoke_seed=args.smoke_seed,
        orb_features=args.orb_features,
        orb_max_side=args.orb_max_side,
        fast_threshold=args.fast_threshold,
        clahe_clip_limit=args.clahe_clip_limit,
        ratio_test=args.ratio_test,
        ransac_threshold=args.ransac_threshold,
        min_inliers=args.min_inliers,
        local_weights=args.local_weights,
        thresholds=args.thresholds,
    )
    print(f"[orb_rerank] summary: {outputs['summary_path']}")
    print(f"[orb_rerank] comparison: {outputs['comparison_path']}")
    print(f"[orb_rerank] smoke: {outputs['smoke_path']}")
    print(f"[orb_rerank] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
