#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.salamander_dualview_training import (
        DEFAULT_BASE_PREDICTIONS,
        DEFAULT_CACHE_DIR,
        DEFAULT_TEST_MANIFEST,
        DEFAULT_THRESHOLDS,
        DEFAULT_TRAIN_MANIFEST,
        run_salamander_dualview_training,
    )

    parser = argparse.ArgumentParser(description="Train Salamander dual-view backbone and build a submission override.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-id", type=str, default="ft_salamander_dualview_centertrunk_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_salamander_dualview_centertrunk_v1",
    )
    parser.add_argument("--base-predictions-path", type=Path, default=repo_root / DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--train-manifest-path", type=Path, default=repo_root / DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--test-manifest-path", type=Path, default=repo_root / DEFAULT_TEST_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--student-backbone", choices=["mega", "miew", "convnext"], default="miew")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--analysis-val-identity-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lr-reference-batch-size", type=int, default=4)
    parser.add_argument("--lr-scale-mode", choices=["none", "linear", "sqrt"], default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--supcon-weight", type=float, default=0.75)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=0.55)
    parser.add_argument("--trunk-weight", type=float, default=0.45)
    args = parser.parse_args()

    outputs = run_salamander_dualview_training(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        experiment_id=args.experiment_id,
        base_predictions_path=args.base_predictions_path.resolve(),
        train_manifest_path=args.train_manifest_path.resolve(),
        test_manifest_path=args.test_manifest_path.resolve(),
        cache_dir=args.cache_dir,
        student_backbone=args.student_backbone,
        device=args.device,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        analysis_val_identity_fraction=args.analysis_val_identity_fraction,
        split_seed=args.split_seed,
        thresholds=args.thresholds,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        lr_reference_batch_size=args.lr_reference_batch_size,
        lr_scale_mode=args.lr_scale_mode,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        pair_weight=args.pair_weight,
        supcon_weight=args.supcon_weight,
        temperature=args.temperature,
        grad_clip_norm=args.grad_clip_norm,
        global_weight=args.global_weight,
        trunk_weight=args.trunk_weight,
    )
    print(f"[salamander_dualview_training] summary: {outputs['summary_path']}")
    print(f"[salamander_dualview_training] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[salamander_dualview_training] submission: {outputs['submission_path']}")
    print(f"[salamander_dualview_training] predictions: {outputs['test_predictions_path']}")
    print(f"[salamander_dualview_training] embeddings: {outputs['test_embeddings_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
