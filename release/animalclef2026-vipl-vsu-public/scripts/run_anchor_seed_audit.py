#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


LABELED_DATASETS = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]


def resolve_default_teacher_config(dataset: str, repo_root: Path) -> tuple[list[str], list[Path]]:
    if dataset == "SalamanderID2025":
        return (
            ["mega", "miew"],
            [repo_root / "artifacts" / "training" / "experiments" / "ft_miew_arcface_masked_supcon_v1" / "checkpoints" / "last.pt"],
        )
    if dataset == "LynxID2025":
        return (
            ["mega"],
            [repo_root / "artifacts" / "training" / "experiments" / "ft_mega_arcface_distill_v1" / "checkpoints" / "best.pt"],
        )
    if dataset == "SeaTurtleID2022":
        return (
            ["mega", "miew"],
            [],
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.anchor_seed_audit import run_anchor_seed_audit

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--dataset", choices=LABELED_DATASETS, default="SalamanderID2025")
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    default_descriptor_sources, default_checkpoint_sources = resolve_default_teacher_config(
        bootstrap_args.dataset,
        repo_root,
    )

    parser = argparse.ArgumentParser(description="Audit anchor/seed quality before self-training.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset", choices=LABELED_DATASETS, default=bootstrap_args.dataset)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / f"{bootstrap_args.dataset.lower()}_anchor_seed_audit_20260402",
    )
    parser.add_argument("--teacher-descriptor-sources", nargs="*", default=default_descriptor_sources)
    parser.add_argument("--teacher-checkpoint-sources", nargs="*", type=Path, default=default_checkpoint_sources)
    parser.add_argument("--anchors", nargs="+", type=float, default=[0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--stability-delta", type=float, default=0.03)
    parser.add_argument("--min-seed-cluster-size", type=int, default=2)
    parser.add_argument("--max-seed-cluster-size", type=int, default=12)
    parser.add_argument("--min-mean-similarity", type=float, default=0.0)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    outputs = run_anchor_seed_audit(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        dataset=args.dataset,
        teacher_descriptor_sources=[str(value) for value in args.teacher_descriptor_sources],
        teacher_checkpoint_sources=[str(path.resolve()) for path in args.teacher_checkpoint_sources],
        anchors=args.anchors,
        stability_delta=args.stability_delta,
        min_seed_cluster_size=args.min_seed_cluster_size,
        max_seed_cluster_size=args.max_seed_cluster_size,
        min_mean_similarity=args.min_mean_similarity,
        val_identity_fraction=args.val_identity_fraction,
        split_seed=args.split_seed,
        device=args.device,
        num_workers=args.num_workers,
    )
    print(f"[anchor_seed_audit] summary: {outputs['summary_path']}")
    print(f"[anchor_seed_audit] anchor_summary: {outputs['anchor_summary_path']}")
    print(f"[anchor_seed_audit] qualitative: {outputs['qualitative_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
