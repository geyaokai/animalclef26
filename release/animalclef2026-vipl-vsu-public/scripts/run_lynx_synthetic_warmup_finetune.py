#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(command: list[str], *, dry_run: bool) -> None:
    print("[lynx_synthetic_warmup] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run CzechLynx synthetic warmup followed by AnimalCLEF Lynx fine-tune.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-root", type=Path, default=repo_root / "artifacts" / "training" / "experiments")
    parser.add_argument("--student-backbone", choices=["mega", "miew", "convnext"], default="mega")
    parser.add_argument("--experiment-prefix", type=str, default="lynx_synthetic_warmup")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--warmup-train-batch-size", type=int)
    parser.add_argument("--finetune-train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-warmup-rows", type=int)
    parser.add_argument("--max-finetune-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--warmup-checkpoint", choices=["best_ari", "best_recall1", "last"], default="last")
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--lr-reference-batch-size", type=int, default=4)
    parser.add_argument("--lr-scale-mode", choices=["none", "linear", "sqrt"], default="linear")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--arcface-scale", type=float, default=30.0)
    parser.add_argument("--arcface-margin", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-finetune", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    manifest_dir = repo_root / "artifacts" / "external_data" / "czechlynx" / "manifests"
    synthetic_manifest = manifest_dir / "tables" / "manifest_train_czechlynx_synthetic_as_lynx_v1.csv"
    synthetic_test_manifest = repo_root / "artifacts" / "manifests" / "v1" / "tables" / "manifest_test_default_v1.csv"

    build_manifest_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "build_czechlynx_synthetic_manifest.py"),
        "--repo-root",
        str(repo_root),
        "--output-dir",
        str(manifest_dir),
    ]
    if args.max_warmup_rows is not None:
        build_manifest_cmd.extend(["--max-rows", str(args.max_warmup_rows)])
    _run(build_manifest_cmd, dry_run=args.dry_run)

    warmup_experiment_id = f"{args.experiment_prefix}_{args.student_backbone}_synthetic_seed{args.split_seed}"
    finetune_experiment_id = f"{args.experiment_prefix}_{args.student_backbone}_finetune_seed{args.split_seed}"
    warmup_dir = output_root / warmup_experiment_id
    finetune_dir = output_root / finetune_experiment_id
    checkpoint_name = "last.pt" if args.warmup_checkpoint == "last" else f"{args.warmup_checkpoint}.pt"
    warmup_checkpoint_path = warmup_dir / "checkpoints" / checkpoint_name

    common_train_args = [
        sys.executable,
        str(repo_root / "scripts" / "run_supervised_training.py"),
        "--repo-root",
        str(repo_root),
        "--student-backbone",
        args.student_backbone,
        "--teacher-sources",
        "--datasets",
        "LynxID2025",
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--val-identity-fraction",
        str(args.val_identity_fraction),
        "--split-seed",
        str(args.split_seed),
        "--backbone-lr",
        str(args.backbone_lr),
        "--head-lr",
        str(args.head_lr),
        "--lr-reference-batch-size",
        str(args.lr_reference_batch_size),
        "--lr-scale-mode",
        args.lr_scale_mode,
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--arcface-scale",
        str(args.arcface_scale),
        "--arcface-margin",
        str(args.arcface_margin),
        "--relation-distill-weight",
        "0.0",
        "--feature-distill-weight",
        "0.0",
        "--label-smoothing",
        str(args.label_smoothing),
        "--grad-clip-norm",
        str(args.grad_clip_norm),
        "--dataset-preprocess-override",
        "LynxID2025=hist_norm_rgb",
    ]
    if args.eval_batch_size is not None:
        common_train_args.extend(["--eval-batch-size", str(args.eval_batch_size)])
    if args.max_val_rows is not None:
        common_train_args.extend(["--max-val-rows", str(args.max_val_rows)])
    if args.max_train_batches is not None:
        common_train_args.extend(["--max-train-batches", str(args.max_train_batches)])

    if not args.skip_warmup:
        warmup_cmd = [
            *common_train_args,
            "--experiment-id",
            warmup_experiment_id,
            "--output-dir",
            str(warmup_dir),
            "--epochs",
            str(args.warmup_epochs),
            "--train-manifest-path",
            str(synthetic_manifest),
            "--test-manifest-path",
            str(synthetic_test_manifest),
            "--goal",
            "Warm up the Lynx backbone on CzechLynx synthetic identities only, without distillation or real CzechLynx labels.",
            "--resource-decision",
            f"Single GPU warmup on {args.device}; synthetic labels are independent from AnimalCLEF identities.",
            "--probe-reuse-note",
            "Warmup uses the normal supervised loop with dataset=LynxID2025 only to reuse Lynx preprocessing.",
        ]
        if args.warmup_train_batch_size is not None:
            warmup_cmd.extend(["--train-batch-size", str(args.warmup_train_batch_size)])
        _run(warmup_cmd, dry_run=args.dry_run)

    if not args.skip_finetune:
        if not args.dry_run and not warmup_checkpoint_path.exists():
            raise FileNotFoundError(f"Warmup checkpoint is missing: {warmup_checkpoint_path}")
        finetune_cmd = [
            *common_train_args,
            "--experiment-id",
            finetune_experiment_id,
            "--output-dir",
            str(finetune_dir),
            "--epochs",
            str(args.finetune_epochs),
            "--init-checkpoint-path",
            str(warmup_checkpoint_path),
            "--init-checkpoint-scope",
            "encoder",
            "--goal",
            "Fine-tune AnimalCLEF Lynx after CzechLynx synthetic warmup, loading encoder weights only and reinitializing the Lynx ArcFace head.",
            "--resource-decision",
            f"Single GPU fine-tune on {args.device}; CzechLynx real is excluded to avoid leakage.",
            "--probe-reuse-note",
            "This run tests whether synthetic warmup improves Lynx-only supervised representation quality.",
        ]
        if args.finetune_train_batch_size is not None:
            finetune_cmd.extend(["--train-batch-size", str(args.finetune_train_batch_size)])
        if args.max_finetune_rows is not None:
            finetune_cmd.extend(["--max-train-rows", str(args.max_finetune_rows)])
        _run(finetune_cmd, dry_run=args.dry_run)

    plan = {
        "synthetic_manifest": str(synthetic_manifest),
        "warmup_experiment_id": warmup_experiment_id,
        "warmup_dir": str(warmup_dir),
        "warmup_checkpoint_path": str(warmup_checkpoint_path),
        "finetune_experiment_id": finetune_experiment_id,
        "finetune_dir": str(finetune_dir),
        "student_backbone": args.student_backbone,
        "device": args.device,
        "synthetic_label_policy": "CzechLynx_Synthetic identities are independent and used only for warmup.",
        "real_czechlynx_policy": "CzechLynx real is excluded from both stages.",
    }
    plan_path = output_root / f"{args.experiment_prefix}_{args.student_backbone}_seed{args.split_seed}_plan.json"
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[lynx_synthetic_warmup] plan: {plan_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
