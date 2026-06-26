#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.identity_distribution_audit import run_identity_distribution_audit

    parser = argparse.ArgumentParser(description="Audit identity-count distributions for labeled train data and baseline splits.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "identity_distribution_audit" / "v1",
    )
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    outputs = run_identity_distribution_audit(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        val_identity_fraction=args.val_identity_fraction,
        split_seed=args.split_seed,
    )
    print(f"[identity_distribution] report: {outputs['report_path']}")
    print(f"[identity_distribution] plots: {outputs['plots_dir']}")
    print(f"[identity_distribution] tables: {outputs['tables_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
