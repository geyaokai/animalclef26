#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import PATH_COLUMN
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path
    from animalclef_analysis.submission_baseline import _load_supervised_model_from_checkpoint
    from animalclef_analysis.supervised_training import extract_student_embeddings
    from animalclef_analysis.view_manifests import get_default_manifest_paths

    parser = argparse.ArgumentParser(description="Extract a standard val/test embedding bundle from one supervised checkpoint.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset", type=str, default="LynxID2025")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--val-metadata-path", type=Path, required=True)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    embeddings_dir = output_dir / "embeddings"
    reports_dir = output_dir / "reports"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset = str(args.dataset)
    val_df = pd.read_csv(args.val_metadata_path.resolve())
    val_df["image_id"] = val_df["image_id"].astype(str)
    val_df["dataset"] = val_df["dataset"].astype(str)
    val_df = val_df[val_df["dataset"] == dataset].copy().reset_index(drop=True)
    if val_df.empty:
        raise ValueError(f"No validation rows found for dataset={dataset} in {args.val_metadata_path.resolve()}")
    if PATH_COLUMN in val_df.columns:
        val_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in val_df.iterrows()]

    if args.test_manifest_path is not None:
        test_manifest_path = args.test_manifest_path.resolve()
    else:
        _train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
    test_df = pd.read_csv(test_manifest_path)
    test_df["image_id"] = test_df["image_id"].astype(str)
    test_df["dataset"] = test_df["dataset"].astype(str)
    test_df = test_df[test_df["dataset"] == dataset].copy().reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"No test rows found for dataset={dataset} in {test_manifest_path}")
    test_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in test_df.iterrows()]

    model, spec, checkpoint_config, checkpoint = _load_supervised_model_from_checkpoint(
        checkpoint_path=args.checkpoint_path.resolve(),
        device=str(args.device),
    )
    preprocess_config = checkpoint_config.get("resolved_preprocess_config")
    val_embeddings = extract_student_embeddings(
        df=val_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        horizontal_flip_tta=bool(args.horizontal_flip_tta),
        preprocess_config=preprocess_config,
    )
    test_embeddings = extract_student_embeddings(
        df=test_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=str(args.device),
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        horizontal_flip_tta=bool(args.horizontal_flip_tta),
        preprocess_config=preprocess_config,
    )

    np.save(embeddings_dir / "val_embeddings.npy", val_embeddings.astype(np.float32))
    np.save(embeddings_dir / "test_embeddings.npy", test_embeddings.astype(np.float32))
    val_df.to_csv(embeddings_dir / "val_metadata.csv", index=False)
    test_df.to_csv(embeddings_dir / "test_metadata.csv", index=False)

    summary = {
        "dataset": dataset,
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "student_backbone": checkpoint_config.get("student_backbone"),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "embedding_dim": int(val_embeddings.shape[1]) if val_embeddings.ndim == 2 and len(val_embeddings) else 0,
        "horizontal_flip_tta": bool(args.horizontal_flip_tta),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
