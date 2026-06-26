#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_dataset_preprocess_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected dataset preprocess override in DATASET=MODE format, got: {item}")
        dataset, mode = item.split("=", 1)
        dataset = dataset.strip()
        mode = mode.strip()
        if not dataset or not mode:
            raise ValueError(f"Invalid dataset preprocess override: {item}")
        overrides[dataset] = mode
    return overrides


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.supervised_training import (
        DEFAULT_THRESHOLDS,
        LABELED_DATASETS,
        STUDENT_BACKBONE_SPECS,
        run_supervised_training,
    )

    parser = argparse.ArgumentParser(description="Run supervised ArcFace + distillation training.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-backbone", choices=sorted(STUDENT_BACKBONE_SPECS), required=True)
    parser.add_argument("--teacher-sources", nargs="*", choices=["mega", "miew"], default=["mega", "miew"])
    parser.add_argument("--datasets", nargs="+", choices=LABELED_DATASETS)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
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
    parser.add_argument("--relation-distill-weight", type=float, default=0.2)
    parser.add_argument("--feature-distill-weight", type=float, default=0.05)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--salamander-subcenter-k", type=int, default=1)
    parser.add_argument("--teacher-cache-dir", type=Path)
    parser.add_argument("--train-manifest-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--init-checkpoint-path", type=Path)
    parser.add_argument("--init-checkpoint-scope", choices=["encoder", "all_matching"], default="encoder")
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument(
        "--dataset-preprocess-override",
        action="append",
        default=[],
        help="Override preprocess mode as DATASET=MODE, e.g. LynxID2025=hist_norm_rgb",
    )
    parser.add_argument("--goal", type=str)
    parser.add_argument("--resource-decision", type=str)
    parser.add_argument("--probe-reuse-note", type=str)
    args = parser.parse_args()

    outputs = run_supervised_training(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        experiment_id=args.experiment_id,
        student_backbone=args.student_backbone,
        teacher_sources=args.teacher_sources,
        datasets=args.datasets,
        device=args.device,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_identity_fraction=args.val_identity_fraction,
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
        relation_distill_weight=args.relation_distill_weight,
        feature_distill_weight=args.feature_distill_weight,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        salamander_subcenter_k=args.salamander_subcenter_k,
        teacher_cache_dir=args.teacher_cache_dir.resolve() if args.teacher_cache_dir else None,
        train_manifest_path=args.train_manifest_path.resolve() if args.train_manifest_path else None,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
        init_checkpoint_path=args.init_checkpoint_path.resolve() if args.init_checkpoint_path else None,
        init_checkpoint_scope=args.init_checkpoint_scope,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_train_batches=args.max_train_batches,
        dataset_preprocess_overrides=_parse_dataset_preprocess_overrides(args.dataset_preprocess_override),
        goal=args.goal,
        resource_decision=args.resource_decision,
        probe_reuse_note=args.probe_reuse_note,
    )
    print(f"[supervised_training] summary: {outputs['summary_path']}")
    print(f"[supervised_training] training_log: {outputs['training_log_path']}")
    print(f"[supervised_training] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[supervised_training] best_checkpoints: {outputs['best_checkpoint_paths']}")
    print(f"[supervised_training] teacher_cache: {outputs['teacher_cache_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
