#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_unsupervised import (
        DEFAULT_TOP_K,
        build_route_config,
        run_texas_threshold_sweep,
    )

    parser = argparse.ArgumentParser(description="Run a Texas-only cached-embedding threshold sweep.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "descriptor_baselines" / "texas_threshold_sweep_v1",
    )
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=["miew", "fusion"],
        default=["miew", "fusion"],
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    outputs = run_texas_threshold_sweep(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        route_configs=[build_route_config(name, args.repo_root.resolve()) for name in args.routes],
        top_k=args.top_k,
    )
    print(f"[texas_threshold_sweep] summary: {outputs['summary_path']}")
    print(f"[texas_threshold_sweep] candidates: {outputs['summary_table_path']}")
    print(f"[texas_threshold_sweep] predictions: {outputs['predictions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
