#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.frozen_view_gate import SUPPORTED_DESCRIPTORS, run_frozen_view_gate
    from animalclef_analysis.view_manifests import BODY_AXIS_VIEW_NAME, DEFAULT_MANIFEST_ROOT, DEFAULT_VIEW_NAME

    parser = argparse.ArgumentParser(description="Run frozen descriptor view gating across unified manifests.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--manifest-root", type=Path, default=repo_root / DEFAULT_MANIFEST_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "descriptor_baselines" / "frozen_view_gate_v1",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=[DEFAULT_VIEW_NAME, BODY_AXIS_VIEW_NAME],
    )
    parser.add_argument(
        "--descriptors",
        nargs="+",
        choices=list(SUPPORTED_DESCRIPTORS),
        default=["miew"],
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    args = parser.parse_args()

    outputs = run_frozen_view_gate(
        repo_root=args.repo_root.resolve(),
        manifest_root=args.manifest_root.resolve(),
        output_dir=args.output_dir.resolve(),
        views=args.views,
        descriptors=args.descriptors,
        device=args.device,
        num_workers=args.num_workers,
        val_identity_fraction=args.val_identity_fraction,
        split_seed=args.split_seed,
        thresholds=args.thresholds,
    )
    print(f"[frozen_view_gate] summary: {outputs['summary_path']}")
    print(f"[frozen_view_gate] metrics: {outputs['metrics_path']}")
    print(f"[frozen_view_gate] deltas: {outputs['delta_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
