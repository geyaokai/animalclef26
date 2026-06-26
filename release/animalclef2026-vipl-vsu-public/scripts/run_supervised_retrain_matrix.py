#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RETRAIN_PRESETS = {
    "miew_distill": {
        "experiment_prefix": "ft_miew_arcface_distill_rtv2",
        "student_backbone": "miew",
        "teacher_sources": ["mega", "miew"],
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "relation_distill_weight": 0.2,
        "feature_distill_weight": 0.05,
        "supcon_weight": 0.0,
        "salamander_subcenter_k": 1,
    },
    "mega_distill": {
        "experiment_prefix": "ft_mega_arcface_distill_rtv2",
        "student_backbone": "mega",
        "teacher_sources": ["mega", "miew"],
        "train_batch_size": 20,
        "eval_batch_size": 20,
        "relation_distill_weight": 0.2,
        "feature_distill_weight": 0.05,
        "supcon_weight": 0.0,
        "salamander_subcenter_k": 1,
    },
    "miew_masked_supcon": {
        "experiment_prefix": "ft_miew_arcface_masked_supcon_rtv2",
        "student_backbone": "miew",
        "teacher_sources": ["mega", "miew"],
        "train_batch_size": 32,
        "eval_batch_size": 32,
        "relation_distill_weight": 0.2,
        "feature_distill_weight": 0.05,
        "supcon_weight": 0.1,
        "salamander_subcenter_k": 1,
    },
    "lynx_mega_hist_nodistill": {
        "experiment_prefix": "ft_lynx_mega_hist_nodistill_rtv1",
        "student_backbone": "mega",
        "teacher_sources": [],
        "train_batch_size": 20,
        "eval_batch_size": 20,
        "relation_distill_weight": 0.0,
        "feature_distill_weight": 0.0,
        "supcon_weight": 0.0,
        "salamander_subcenter_k": 1,
    },
    "lynx_miew_hist_nodistill": {
        "experiment_prefix": "ft_lynx_miew_hist_nodistill_rtv1",
        "student_backbone": "miew",
        "teacher_sources": [],
        "train_batch_size": 24,
        "eval_batch_size": 24,
        "relation_distill_weight": 0.0,
        "feature_distill_weight": 0.0,
        "supcon_weight": 0.0,
        "salamander_subcenter_k": 1,
    },
}


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


def _format_fraction_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _format_dataset_tag(datasets: list[str] | None) -> str:
    if not datasets:
        return "all"
    return "-".join(sorted(datasets))


def _resolve_teacher_cache_dir(cache_root: Path, split_seed: int, val_identity_fraction: float, datasets: list[str] | None = None) -> Path:
    fraction_tag = _format_fraction_tag(val_identity_fraction)
    dataset_tag = _format_dataset_tag(datasets)
    legacy_candidates = [
        cache_root / f"shared_teacher_cache_seed{split_seed}_val{fraction_tag}_{dataset_tag}_full_v2",
        cache_root / f"shared_teacher_cache_seed{split_seed}_val{fraction_tag}_{dataset_tag}_v1",
    ]
    if dataset_tag == "all":
        legacy_candidates.extend(
            [
                cache_root / f"shared_teacher_cache_seed{split_seed}_val{fraction_tag}_full_v2",
                cache_root / f"shared_teacher_cache_seed{split_seed}_val{fraction_tag}_v1",
            ]
        )
    for candidate in legacy_candidates:
        if candidate.exists():
            return candidate
    return cache_root / f"shared_teacher_cache_seed{split_seed}_val{fraction_tag}_{dataset_tag}_rtv2"


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.supervised_training import LABELED_DATASETS, run_supervised_training

    parser = argparse.ArgumentParser(description="Run the supervised retraining matrix under the upgraded validation protocol.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-root", type=Path, default=repo_root / "artifacts" / "training" / "experiments")
    parser.add_argument("--cache-root", type=Path, default=repo_root / "artifacts" / "training" / "cache")
    parser.add_argument("--experiments", nargs="+", choices=sorted(RETRAIN_PRESETS), default=["miew_distill", "mega_distill", "miew_masked_supcon"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-identity-fraction", type=float, default=0.1)
    parser.add_argument("--datasets", nargs="+", choices=LABELED_DATASETS)
    parser.add_argument("--train-batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument(
        "--dataset-preprocess-override",
        action="append",
        default=[],
        help="Override preprocess mode as DATASET=MODE, e.g. LynxID2025=hist_norm_rgb",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    cache_root = args.cache_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    dataset_tag = _format_dataset_tag(args.datasets)
    dataset_preprocess_overrides = _parse_dataset_preprocess_overrides(args.dataset_preprocess_override)
    if not args.datasets and any(preset_name.startswith("lynx_") for preset_name in args.experiments):
        args.datasets = ["LynxID2025"]
        dataset_tag = _format_dataset_tag(args.datasets)
    if not dataset_preprocess_overrides and any(preset_name.endswith("hist_nodistill") for preset_name in args.experiments):
        dataset_preprocess_overrides = {"LynxID2025": "hist_norm_rgb"}

    plan_rows: list[dict[str, object]] = []
    for preset_name in args.experiments:
        preset = RETRAIN_PRESETS[preset_name]
        for split_seed in args.seeds:
            experiment_id = f"{preset['experiment_prefix']}_seed{split_seed}"
            if dataset_tag != "all":
                experiment_id = f"{experiment_id}_{dataset_tag}"
            output_dir = output_root / experiment_id
            preset_uses_distill = (
                float(preset["relation_distill_weight"]) > 0.0
                or float(preset["feature_distill_weight"]) > 0.0
            )
            teacher_cache_dir = (
                _resolve_teacher_cache_dir(
                    cache_root=cache_root,
                    split_seed=int(split_seed),
                    val_identity_fraction=float(args.val_identity_fraction),
                    datasets=args.datasets,
                )
                if preset_uses_distill
                else None
            )
            plan_rows.append(
                {
                    "experiment_id": experiment_id,
                    "preset": preset_name,
                    "device": args.device,
                    "split_seed": split_seed,
                    "datasets": args.datasets or LABELED_DATASETS,
                    "output_dir": str(output_dir),
                    "teacher_cache_dir": str(teacher_cache_dir) if teacher_cache_dir is not None else "",
                    "exists": (output_dir / "reports" / "summary.json").exists(),
                }
            )
            if args.skip_existing and (output_dir / "reports" / "summary.json").exists():
                print(f"[retrain_matrix] skip existing: {experiment_id}", flush=True)
                continue
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "experiment_id": experiment_id,
                            "preset": preset_name,
                            "device": args.device,
                            "split_seed": split_seed,
                            "output_dir": str(output_dir),
                            "teacher_cache_dir": str(teacher_cache_dir) if teacher_cache_dir is not None else "",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

            outputs = run_supervised_training(
                repo_root=repo_root,
                output_dir=output_dir,
                experiment_id=experiment_id,
                student_backbone=str(preset["student_backbone"]),
                teacher_sources=list(preset["teacher_sources"]),
                datasets=args.datasets,
                device=args.device,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size or int(preset["train_batch_size"]),
                eval_batch_size=args.eval_batch_size or int(preset["eval_batch_size"]),
                num_workers=args.num_workers,
                val_identity_fraction=args.val_identity_fraction,
                split_seed=int(split_seed),
                relation_distill_weight=float(preset["relation_distill_weight"]),
                feature_distill_weight=float(preset["feature_distill_weight"]),
                supcon_weight=float(preset["supcon_weight"]),
                salamander_subcenter_k=int(preset["salamander_subcenter_k"]),
                teacher_cache_dir=teacher_cache_dir,
                max_train_batches=args.max_train_batches,
                dataset_preprocess_overrides=dataset_preprocess_overrides,
                goal=(
                    f"Retrain `{preset_name}` under the upgraded protocol with identity-level validation seed `{split_seed}`, "
                    "explicit best_ari / best_recall1 / last checkpoint comparison, plus dataset-specific checkpoint saving and later multi-seed aggregation."
                ),
                resource_decision=(
                    f"Sequential retraining matrix on `{args.device}`. Keep old jobs untouched; use conservative batch sizing if free memory is limited."
                ),
                probe_reuse_note=(
                    "This retrain matrix reuses the historical recipe as the reference point but upgrades checkpoint policy. "
                    "If GPU free memory is visibly below the old formal runs, pass smaller train/eval batch sizes at launch."
                ),
            )
            print(f"[retrain_matrix] completed: {experiment_id} -> {outputs['summary_path']}", flush=True)

    plan_path = output_root / "retrain_matrix_plan.json"
    plan_path.write_text(json.dumps(plan_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[retrain_matrix] plan: {plan_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
