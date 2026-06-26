#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_submission_ensemble import run_texas_ensemble_submission_variant

    parser = argparse.ArgumentParser(description="Build a Texas-only ensemble submission variant on top of the mixed baseline.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-name", type=str, required=True)
    parser.add_argument("--source-dir", type=Path, action="append")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-component-name", type=str)
    parser.add_argument("--component-name", type=str, action="append")
    parser.add_argument("--weight", type=float, action="append")
    parser.add_argument("--threshold", type=float, action="append", dest="thresholds")
    parser.add_argument("--anchor-threshold", type=float, default=0.44)
    parser.add_argument("--base-predictions", type=Path)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pseudo-assignments-path", type=Path)
    parser.add_argument("--candidate-pairs-path", type=Path)
    parser.add_argument("--teacher-anchor-predictions-path", type=Path)
    parser.add_argument("--teacher-topk-source-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    if bool(args.source_dir) == bool(args.checkpoint_path):
        parser.error("Pass exactly one input mode: either --source-dir ... or --checkpoint-path ...")
    if args.checkpoint_path and args.weight:
        parser.error("--weight is only supported with --source-dir")
    if args.checkpoint_path and args.component_name:
        parser.error("--component-name is only supported with --source-dir; use --checkpoint-component-name")

    outputs = run_texas_ensemble_submission_variant(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        route_name=str(args.route_name),
        source_dirs=[path.resolve() for path in (args.source_dir or [])],
        checkpoint_path=args.checkpoint_path.resolve() if args.checkpoint_path else None,
        checkpoint_component_name=str(args.checkpoint_component_name) if args.checkpoint_component_name else None,
        component_names=args.component_name,
        weights=args.weight,
        thresholds=args.thresholds,
        anchor_threshold=float(args.anchor_threshold),
        base_predictions=args.base_predictions.resolve() if args.base_predictions else None,
        sample_submission_path=args.sample_submission_path.resolve() if args.sample_submission_path else None,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
        device=args.device,
        eval_batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        pseudo_assignments_path=args.pseudo_assignments_path.resolve() if args.pseudo_assignments_path else None,
        candidate_pairs_path=args.candidate_pairs_path.resolve() if args.candidate_pairs_path else None,
        teacher_anchor_predictions_path=args.teacher_anchor_predictions_path.resolve() if args.teacher_anchor_predictions_path else None,
        teacher_topk_source_dir=args.teacher_topk_source_dir.resolve() if args.teacher_topk_source_dir else None,
        top_k=int(args.top_k),
    )
    print(f"[texas_submission_ensemble] submission: {outputs['submission_path']}")
    print(f"[texas_submission_ensemble] summary: {outputs['summary_path']}")
    print(f"[texas_submission_ensemble] predictions: {outputs['prediction_path']}")
    print(f"[texas_submission_ensemble] best_threshold: {outputs['best_threshold_path']}")
    print(f"[texas_submission_ensemble] threshold_sweep: {outputs['threshold_sweep_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
