#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def resolve_default_warmup_manifest_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "artifacts"
        / "training"
        / "cache"
        / "tcu_texas_warmup_manifest_v1"
        / "tables"
        / "tcu_texas_warmup_manifest_v1.csv"
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_external_warmup import build_tcu_texas_warmup_manifest, run_texas_external_warmup

    parser = argparse.ArgumentParser(description="Run Texas external supervised warmup on TCU chips.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-id", type=str, default="ft_texas_external_warmup_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_external_warmup_v1",
    )
    parser.add_argument("--warmup-manifest-path", type=Path, default=resolve_default_warmup_manifest_path(repo_root))
    parser.add_argument(
        "--chip-manifest-path",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "tcu_texas_dataset_v1" / "tables" / "tcu_texas_chip_manifest_v1.csv",
    )
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--student-backbone", type=str, default="miew")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lr-reference-batch-size", type=int, default=4)
    parser.add_argument("--lr-scale-mode", choices=["none", "linear", "sqrt"], default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--goal", type=str)
    parser.add_argument("--resource-decision", type=str)
    parser.add_argument("--probe-reuse-note", type=str)
    args = parser.parse_args()

    warmup_manifest_path = args.warmup_manifest_path.resolve()
    if args.rebuild_manifest or not warmup_manifest_path.exists():
        build_tcu_texas_warmup_manifest(
            repo_root=args.repo_root.resolve(),
            chip_manifest_path=args.chip_manifest_path.resolve(),
            output_path=warmup_manifest_path,
        )

    outputs = run_texas_external_warmup(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        experiment_id=args.experiment_id,
        warmup_manifest_path=warmup_manifest_path,
        student_backbone=args.student_backbone,
        device=args.device,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        lr_reference_batch_size=args.lr_reference_batch_size,
        lr_scale_mode=args.lr_scale_mode,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        max_train_batches=args.max_train_batches,
        goal=args.goal,
        resource_decision=args.resource_decision,
        probe_reuse_note=args.probe_reuse_note,
    )
    print(f"[texas_external_warmup] summary: {outputs['summary_path']}")
    print(f"[texas_external_warmup] training_log: {outputs['training_log_path']}")
    print(f"[texas_external_warmup] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[texas_external_warmup] last_checkpoint: {outputs['last_checkpoint_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
