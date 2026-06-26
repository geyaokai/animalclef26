#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_unsupervised import (
        DEFAULT_MAX_SEED_CLUSTER_SIZE,
        DEFAULT_MIN_COMPONENT_DENSITY,
        DEFAULT_TOP_K,
        TexasRouteConfig,
        build_texas_pseudo_seed,
    )

    parser = argparse.ArgumentParser(description="Build high-precision Texas pseudo seeds from cached route consensus.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "training" / "cache" / "texas_pseudo_seed_v1",
    )
    parser.add_argument(
        "--miew-source-dir",
        type=Path,
        default=repo_root / "artifacts" / "descriptor_baselines" / "embed_miew_v1",
    )
    parser.add_argument("--miew-threshold", type=float, default=0.38)
    parser.add_argument(
        "--fusion-source-dir",
        type=Path,
        default=repo_root / "artifacts" / "descriptor_baselines" / "embed_fusion_v1",
    )
    parser.add_argument("--fusion-threshold", type=float, default=0.43)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--min-component-density", type=float, default=DEFAULT_MIN_COMPONENT_DENSITY)
    parser.add_argument("--max-seed-cluster-size", type=int, default=DEFAULT_MAX_SEED_CLUSTER_SIZE)
    args = parser.parse_args()

    outputs = build_texas_pseudo_seed(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        route_configs=[
            TexasRouteConfig(
                name="miew",
                source_dir=args.miew_source_dir.resolve(),
                thresholds=[float(args.miew_threshold)],
                anchor_threshold=float(args.miew_threshold),
            ),
            TexasRouteConfig(
                name="fusion",
                source_dir=args.fusion_source_dir.resolve(),
                thresholds=[float(args.fusion_threshold)],
                anchor_threshold=float(args.fusion_threshold),
            ),
        ],
        top_k=args.top_k,
        min_component_density=args.min_component_density,
        max_seed_cluster_size=args.max_seed_cluster_size,
    )
    print(f"[texas_pseudo_seed] summary: {outputs['summary_path']}")
    print(f"[texas_pseudo_seed] pseudo_manifest: {outputs['pseudo_manifest_path']}")
    print(f"[texas_pseudo_seed] candidate_pairs: {outputs['candidate_pair_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
