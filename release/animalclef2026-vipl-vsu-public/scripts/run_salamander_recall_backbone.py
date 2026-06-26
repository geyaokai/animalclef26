#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_EXPERIMENT_ID = "ft_salamander_recall_backbone_v1"
DEFAULT_EPOCHS = 30
DEFAULT_WARMUP_EPOCHS = 2
DEFAULT_WARMUP_RATIO = DEFAULT_WARMUP_EPOCHS / DEFAULT_EPOCHS


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.supervised_training import STUDENT_BACKBONE_SPECS, run_supervised_training

    parser = argparse.ArgumentParser(
        description="Run Salamander single-dataset recall backbone training with a fixed supervised recipe."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-id", type=str, default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--student-backbone", choices=sorted(STUDENT_BACKBONE_SPECS), default="miew")
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--val-identity-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    args = parser.parse_args()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else args.repo_root.resolve() / "artifacts" / "training" / "experiments" / args.experiment_id
    )

    outputs = run_supervised_training(
        repo_root=args.repo_root.resolve(),
        output_dir=output_dir,
        experiment_id=args.experiment_id,
        student_backbone=args.student_backbone,
        teacher_sources=[],
        datasets=[SALAMANDER_DATASET],
        device=args.device,
        epochs=DEFAULT_EPOCHS,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        val_identity_fraction=args.val_identity_fraction,
        split_seed=args.split_seed,
        weight_decay=0.01,
        warmup_ratio=DEFAULT_WARMUP_RATIO,
        relation_distill_weight=0.0,
        feature_distill_weight=0.0,
        supcon_weight=0.0,
        max_train_batches=args.max_train_batches,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        goal=(
            "Train a stable SalamanderID2025 single-dataset recall backbone with per-dataset ArcFace, "
            "30 epochs, 2 warmup epochs, and no distillation."
        ),
    )
    print(f"[salamander_recall_backbone] summary: {outputs['summary_path']}")
    print(f"[salamander_recall_backbone] training_log: {outputs['training_log_path']}")
    print(f"[salamander_recall_backbone] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[salamander_recall_backbone] best_checkpoints: {outputs['best_checkpoint_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
