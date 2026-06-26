#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_orb_constraint_graph import (
        DEFAULT_MAX_ORB_INLIERS,
        DEFAULT_MAX_ORB_LOCAL_SCORE,
        DEFAULT_ORB_NEGATIVE_MODE,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_PAIR_JUDGMENTS_PATH,
        DEFAULT_REVIEW_DIR,
        run_texas_orb_constraint_graph_probe,
    )

    parser = argparse.ArgumentParser(
        description="Build Texas ORB auto cannot-link judgments and compare constraint-graph overlay variants."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_miew_pseudo_v1",
    )
    parser.add_argument("--review-dir", type=Path, default=repo_root / DEFAULT_REVIEW_DIR)
    parser.add_argument("--pair-judgments-path", type=Path, default=repo_root / DEFAULT_PAIR_JUDGMENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-orb-local-score", type=float, default=DEFAULT_MAX_ORB_LOCAL_SCORE)
    parser.add_argument("--max-orb-inliers", type=int, default=DEFAULT_MAX_ORB_INLIERS)
    parser.add_argument("--orb-negative-mode", type=str, default=DEFAULT_ORB_NEGATIVE_MODE, choices=["both", "either"])
    parser.add_argument("--min-auto-pairs-per-cluster", type=int, default=1)
    parser.add_argument("--graph-threshold", type=float)
    args = parser.parse_args()

    outputs = run_texas_orb_constraint_graph_probe(
        repo_root=repo_root,
        base_predictions_path=args.predictions_path,
        review_dir=args.review_dir,
        pair_judgments_path=args.pair_judgments_path,
        output_dir=args.output_dir,
        max_orb_local_score=float(args.max_orb_local_score),
        max_orb_inliers=int(args.max_orb_inliers),
        orb_negative_mode=str(args.orb_negative_mode),
        min_auto_pairs_per_cluster=int(args.min_auto_pairs_per_cluster),
        graph_threshold=float(args.graph_threshold) if args.graph_threshold is not None else None,
    )
    print(f"[texas_orb_constraint_graph] auto_pairs: {outputs['auto_pair_path']}")
    print(f"[texas_orb_constraint_graph] combined_judgments: {outputs['combined_judgments_path']}")
    print(f"[texas_orb_constraint_graph] variant_summary: {outputs['variant_summary_path']}")
    print(f"[texas_orb_constraint_graph] review_index: {outputs['review_index_path']}")
    print(f"[texas_orb_constraint_graph] review_pack: {outputs['review_pack_dir']}")
    print(f"[texas_orb_constraint_graph] summary: {outputs['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
