#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


LABELED_DATASETS = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]


def resolve_default_teacher_config(dataset: str, repo_root: Path) -> tuple[str, list[str], list[Path]]:
    if dataset == "SalamanderID2025":
        return (
            "miew",
            ["mega", "miew"],
            [repo_root / "artifacts" / "training" / "experiments" / "ft_miew_arcface_masked_supcon_v1" / "checkpoints" / "last.pt"],
        )
    if dataset == "LynxID2025":
        return (
            "mega",
            ["mega"],
            [repo_root / "artifacts" / "training" / "experiments" / "ft_mega_arcface_distill_v1" / "checkpoints" / "best.pt"],
        )
    if dataset == "SeaTurtleID2022":
        return (
            "miew",
            ["mega", "miew"],
            [],
        )
    raise ValueError(f"Unsupported labeled dataset: {dataset}")


def resolve_default_student_init_checkpoint(dataset: str, repo_root: Path) -> Path | None:
    if dataset == "SalamanderID2025":
        return repo_root / "artifacts" / "training" / "experiments" / "ft_miew_arcface_masked_supcon_v1" / "checkpoints" / "last.pt"
    if dataset == "LynxID2025":
        return repo_root / "artifacts" / "training" / "experiments" / "ft_mega_arcface_distill_v1" / "checkpoints" / "best.pt"
    if dataset == "SeaTurtleID2022":
        return None
    raise ValueError(f"Unsupported labeled dataset: {dataset}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.labeled_selftrain import DEFAULT_THRESHOLDS, run_labeled_selftrain

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--dataset", choices=LABELED_DATASETS, default="SalamanderID2025")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    default_student_backbone, default_descriptor_sources, default_checkpoint_sources = resolve_default_teacher_config(
        bootstrap_args.dataset,
        repo_root,
    )
    default_student_init_checkpoint = resolve_default_student_init_checkpoint(bootstrap_args.dataset, repo_root)
    default_experiment_id = f"labeled_selftrain_{bootstrap_args.dataset.lower()}_v1"
    default_output_dir = repo_root / "artifacts" / "training" / "experiments" / default_experiment_id

    parser = argparse.ArgumentParser(description="Run labeled semi-selftrain on one AnimalCLEF labeled dataset.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset", choices=LABELED_DATASETS, default=bootstrap_args.dataset)
    parser.add_argument("--experiment-id", type=str, default=default_experiment_id)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--student-backbone", choices=["mega", "miew", "convnext"], default=default_student_backbone)
    parser.add_argument("--teacher-descriptor-sources", nargs="+", default=default_descriptor_sources)
    parser.add_argument("--teacher-checkpoint-sources", nargs="*", type=Path, default=default_checkpoint_sources)
    parser.add_argument("--student-init-checkpoint", type=Path, default=default_student_init_checkpoint)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--anchor-threshold", type=float, default=0.30)
    parser.add_argument("--stability-delta", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--eval-thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lr-reference-batch-size", type=int, default=4)
    parser.add_argument("--lr-scale-mode", choices=["none", "linear", "sqrt"], default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--supervised-loss-weight", type=float, default=1.0)
    parser.add_argument("--pseudo-loss-weight", type=float, default=0.5)
    parser.add_argument("--relation-distill-weight", type=float, default=0.2)
    parser.add_argument("--feature-distill-weight", type=float, default=0.05)
    parser.add_argument("--supcon-weight", type=float, default=0.0)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--pseudo-seed-oversample-factor", type=float, default=2.0)
    parser.add_argument("--min-seed-cluster-size", type=int, default=2)
    parser.add_argument("--max-seed-cluster-size", type=int, default=12)
    parser.add_argument("--min-mean-similarity", type=float, default=0.0)
    parser.add_argument("--max-fit-rows", type=int)
    parser.add_argument("--max-target-rows", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--goal", type=str)
    parser.add_argument("--train-manifest-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    args = parser.parse_args()

    outputs = run_labeled_selftrain(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        experiment_id=args.experiment_id,
        dataset=args.dataset,
        student_backbone=args.student_backbone,
        teacher_descriptor_sources=[str(value) for value in args.teacher_descriptor_sources],
        teacher_checkpoint_sources=[str(path.resolve()) for path in args.teacher_checkpoint_sources],
        device=args.device,
        anchor_threshold=args.anchor_threshold,
        stability_delta=args.stability_delta,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        val_identity_fraction=args.val_identity_fraction,
        split_seed=args.split_seed,
        eval_thresholds=args.eval_thresholds,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        lr_reference_batch_size=args.lr_reference_batch_size,
        lr_scale_mode=args.lr_scale_mode,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        supervised_loss_weight=args.supervised_loss_weight,
        pseudo_loss_weight=args.pseudo_loss_weight,
        relation_distill_weight=args.relation_distill_weight,
        feature_distill_weight=args.feature_distill_weight,
        supcon_weight=args.supcon_weight,
        supcon_temperature=args.supcon_temperature,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        pseudo_seed_oversample_factor=args.pseudo_seed_oversample_factor,
        min_seed_cluster_size=args.min_seed_cluster_size,
        max_seed_cluster_size=args.max_seed_cluster_size,
        min_mean_similarity=args.min_mean_similarity,
        max_fit_rows=args.max_fit_rows,
        max_target_rows=args.max_target_rows,
        max_train_batches=args.max_train_batches,
        goal=args.goal,
        train_manifest_path=args.train_manifest_path.resolve() if args.train_manifest_path else None,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
        student_init_checkpoint=args.student_init_checkpoint.resolve() if args.student_init_checkpoint else None,
    )
    print(f"[labeled_selftrain] summary: {outputs['summary_path']}")
    print(f"[labeled_selftrain] training_log: {outputs['training_log_path']}")
    print(f"[labeled_selftrain] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[labeled_selftrain] best_predictions: {outputs['best_predictions_path']}")
    print(f"[labeled_selftrain] best_embeddings: {outputs['best_embeddings_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
