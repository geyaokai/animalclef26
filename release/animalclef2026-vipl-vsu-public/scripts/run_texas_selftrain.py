#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def resolve_default_pseudo_cache_dir(repo_root: Path) -> Path:
    candidates = [
        repo_root / "artifacts" / "training" / "cache" / "texas_pseudo_seed_v2",
        repo_root / "artifacts" / "training" / "cache" / "texas_pseudo_seed_v1",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def resolve_default_test_manifest_path(repo_root: Path) -> Path | None:
    candidates = [
        repo_root / "artifacts" / "manifests" / "sam_seg_trainprep_v1" / "tables" / "manifest_test_sam_trainprep_aligned_best_v1.csv",
        repo_root / "artifacts" / "manifests" / "v1" / "tables" / "manifest_test_default_v1.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_selftrain import (
        DEFAULT_ANCHOR_THRESHOLD,
        DEFAULT_FEATURE_DISTILL_WEIGHT,
        DEFAULT_PSEUDO_LOSS_WEIGHT,
        DEFAULT_RELATION_DISTILL_WEIGHT,
        DEFAULT_SEED_OVERSAMPLE_FACTOR,
        DEFAULT_SELFTRAIN_THRESHOLDS,
        DEFAULT_TOPK_OVERLAP,
        DEFAULT_VIEW_PAIR_TEMPERATURE,
        DEFAULT_VIEW_PAIR_WEIGHT,
        run_texas_selftrain,
    )
    from animalclef_analysis.texas_unsupervised import DEFAULT_TOP_K

    parser = argparse.ArgumentParser(description="Run Texas-only pseudo-label self-training.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--experiment-id",
        type=str,
        default="ft_texas_miew_pseudo_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "training" / "experiments" / "ft_texas_miew_pseudo_v1",
    )
    parser.add_argument(
        "--pseudo-cache-dir",
        type=Path,
        default=resolve_default_pseudo_cache_dir(repo_root),
    )
    parser.add_argument("--assignments-path", type=Path)
    parser.add_argument("--candidate-pairs-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path, default=resolve_default_test_manifest_path(repo_root))
    parser.add_argument("--trusted-membership-path", type=Path)
    parser.add_argument("--pseudo-positive-pairs-path", type=Path)
    parser.add_argument(
        "--teacher-source-dirs",
        nargs="+",
        type=Path,
        default=[],
    )
    parser.add_argument("--teacher-weights", nargs="+", type=float)
    parser.add_argument("--student-backbone", type=str, default="miew")
    parser.add_argument("--student-init-checkpoint", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_SELFTRAIN_THRESHOLDS)
    parser.add_argument("--anchor-threshold", type=float, default=DEFAULT_ANCHOR_THRESHOLD)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lr-reference-batch-size", type=int, default=4)
    parser.add_argument("--lr-scale-mode", choices=["none", "linear", "sqrt"], default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--classification-head", choices=["arcface", "linear"], default="arcface")
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--pseudo-loss-weight", type=float, default=DEFAULT_PSEUDO_LOSS_WEIGHT)
    parser.add_argument("--relation-distill-weight", type=float, default=DEFAULT_RELATION_DISTILL_WEIGHT)
    parser.add_argument("--feature-distill-weight", type=float, default=DEFAULT_FEATURE_DISTILL_WEIGHT)
    parser.add_argument("--view-pair-weight", type=float, default=DEFAULT_VIEW_PAIR_WEIGHT)
    parser.add_argument("--view-pair-temperature", type=float, default=DEFAULT_VIEW_PAIR_TEMPERATURE)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-oversample-factor", type=float, default=DEFAULT_SEED_OVERSAMPLE_FACTOR)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--goal", type=str)
    parser.add_argument("--resource-decision", type=str)
    parser.add_argument("--probe-reuse-note", type=str)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K if DEFAULT_TOPK_OVERLAP == DEFAULT_TOP_K else DEFAULT_TOPK_OVERLAP)
    args = parser.parse_args()

    pseudo_cache_dir = args.pseudo_cache_dir.resolve()
    assignments_path = args.assignments_path.resolve() if args.assignments_path else pseudo_cache_dir / "tables" / "all_assignments_v1.csv"
    candidate_pairs_path = (
        args.candidate_pairs_path.resolve()
        if args.candidate_pairs_path
        else pseudo_cache_dir / "tables" / "candidate_pairs_v1.csv"
    )
    outputs = run_texas_selftrain(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        experiment_id=args.experiment_id,
        assignments_path=assignments_path,
        candidate_pair_path=candidate_pairs_path,
        test_manifest_path=args.test_manifest_path.resolve() if args.test_manifest_path else None,
        trusted_membership_path=args.trusted_membership_path.resolve() if args.trusted_membership_path else None,
        pseudo_positive_pairs_path=args.pseudo_positive_pairs_path.resolve() if args.pseudo_positive_pairs_path else None,
        teacher_source_dirs=[path.resolve() for path in args.teacher_source_dirs],
        teacher_weights=args.teacher_weights,
        student_backbone=args.student_backbone,
        student_init_checkpoint=args.student_init_checkpoint.resolve() if args.student_init_checkpoint else None,
        device=args.device,
        epochs=args.epochs,
        embedding_dim=args.embedding_dim,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        thresholds=args.thresholds,
        anchor_threshold=args.anchor_threshold,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        lr_reference_batch_size=args.lr_reference_batch_size,
        lr_scale_mode=args.lr_scale_mode,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        classification_head=args.classification_head,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        pseudo_loss_weight=args.pseudo_loss_weight,
        relation_distill_weight=args.relation_distill_weight,
        feature_distill_weight=args.feature_distill_weight,
        view_pair_weight=args.view_pair_weight,
        view_pair_temperature=args.view_pair_temperature,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        seed_oversample_factor=args.seed_oversample_factor,
        max_train_batches=args.max_train_batches,
        goal=args.goal,
        resource_decision=args.resource_decision,
        probe_reuse_note=args.probe_reuse_note,
        top_k=args.top_k,
    )
    print(f"[texas_selftrain] summary: {outputs['summary_path']}")
    print(f"[texas_selftrain] training_log: {outputs['training_log_path']}")
    print(f"[texas_selftrain] best_checkpoint: {outputs['best_checkpoint_path']}")
    print(f"[texas_selftrain] test_embeddings: {outputs['test_embeddings_path']}")
    print(f"[texas_selftrain] chosen_predictions: {outputs['chosen_predictions_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
