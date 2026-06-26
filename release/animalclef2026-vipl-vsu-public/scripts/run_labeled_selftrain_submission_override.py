#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_markdown_table(df: pd.DataFrame) -> list[str]:
    table_df = df.copy().fillna("")
    headers = [str(column) for column in table_df.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_df.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _resolve_threshold(checkpoint_path: Path, explicit_threshold: float | None) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    experiment_dir = checkpoint_path.resolve().parents[1]
    best_eval_path = experiment_dir / "tables" / "best_eval_v1.csv"
    if best_eval_path.exists():
        best_eval_df = pd.read_csv(best_eval_path)
        if not best_eval_df.empty and "threshold" in best_eval_df.columns:
            return float(best_eval_df.iloc[0]["threshold"])
    raise FileNotFoundError(
        f"Could not auto-resolve threshold from `{best_eval_path}`. "
        "Pass `--threshold` explicitly."
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import (
        PATH_COLUMN,
        apply_thresholds_to_df,
        build_submission,
    )
    from animalclef_analysis.labeled_selftrain import load_labeled_selftrain_model_from_checkpoint
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path
    from animalclef_analysis.supervised_training import extract_student_embeddings
    from animalclef_analysis.view_manifests import get_default_manifest_paths

    parser = argparse.ArgumentParser(
        description="Build a submission variant by overriding one dataset with a labeled_selftrain checkpoint."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--override-dataset", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--route-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--horizontal-flip-tta", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    embeddings_dir = output_dir / "embeddings"
    for path in [output_dir, tables_dir, reports_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.checkpoint_path.resolve()
    threshold = _resolve_threshold(checkpoint_path=checkpoint_path, explicit_threshold=args.threshold)
    sample_submission_path = args.sample_submission_path.resolve() if args.sample_submission_path else repo_root / "sample_submission.csv"
    if args.test_manifest_path is not None:
        test_manifest_path = args.test_manifest_path.resolve()
    else:
        _train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)

    base_pred_df = pd.read_csv(args.base_predictions.resolve())
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)

    test_df = pd.read_csv(test_manifest_path)
    test_df["image_id"] = test_df["image_id"].astype(str)
    test_df["dataset"] = test_df["dataset"].astype(str)
    test_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in test_df.iterrows()]

    dataset = str(args.override_dataset)
    dataset_df = test_df[test_df["dataset"] == dataset].copy().reset_index(drop=True)
    if dataset_df.empty:
        raise ValueError(f"No rows found for dataset={dataset} in {test_manifest_path}")

    model, spec, checkpoint_config, checkpoint = load_labeled_selftrain_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=args.device,
    )
    dataset_embeddings = extract_student_embeddings(
        df=dataset_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=args.device,
        batch_size=int(args.eval_batch_size),
        num_workers=int(args.num_workers),
        horizontal_flip_tta=bool(args.horizontal_flip_tta),
    )
    np.save(embeddings_dir / f"{dataset}_labeled_selftrain_test_embeddings.npy", dataset_embeddings.astype(np.float32))
    dataset_df.to_csv(embeddings_dir / f"{dataset}_labeled_selftrain_test_metadata.csv", index=False)

    override_pred_df = apply_thresholds_to_df(
        df=dataset_df,
        embeddings=dataset_embeddings,
        threshold_by_dataset={dataset: float(threshold)},
    )
    override_pred_df["route_name"] = str(args.route_name)
    override_pred_df["embedding_dim"] = int(dataset_embeddings.shape[1])
    override_pred_df["rerank_enabled"] = False
    override_pred_df["local_weight"] = 0.0

    kept_df = base_pred_df[base_pred_df["dataset"] != dataset].copy()
    merged_pred_df = pd.concat([kept_df, override_pred_df], ignore_index=True)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    route_rows = []
    for dataset_name, frame in merged_pred_df.groupby("dataset"):
        route_rows.append(
            {
                "dataset": str(dataset_name),
                "route_name": str(frame["route_name"].iloc[0]),
                "embedding_dim": int(frame["embedding_dim"].iloc[0]),
                "threshold": float(frame["chosen_threshold"].iloc[0]),
                "rerank_enabled": bool(frame["rerank_enabled"].iloc[0]) if "rerank_enabled" in frame.columns else False,
                "local_weight": float(frame["local_weight"].iloc[0]) if "local_weight" in frame.columns else 0.0,
            }
        )
    route_df = pd.DataFrame(route_rows).sort_values("dataset").reset_index(drop=True)
    route_df.to_csv(tables_dir / "route_config_v1.csv", index=False)

    summary_rows = []
    for dataset_name, frame in merged_pred_df.groupby("dataset"):
        counts = frame["pred_cluster_id"].value_counts()
        summary_rows.append(
            {
                "dataset": str(dataset_name),
                "samples": int(len(frame)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(frame["route_name"].iloc[0]),
                "embedding_dim": int(frame["embedding_dim"].iloc[0]),
                "threshold": float(frame["chosen_threshold"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=sample_submission_path,
        output_path=output_dir / "submission.csv",
    )

    config = {
        "base_predictions": str(args.base_predictions.resolve()),
        "override_dataset": dataset,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "route_name": str(args.route_name),
        "threshold": float(threshold),
        "device": str(args.device),
        "eval_batch_size": int(args.eval_batch_size),
        "num_workers": int(args.num_workers),
        "horizontal_flip_tta": bool(args.horizontal_flip_tta),
        "test_manifest_path": str(test_manifest_path),
        "sample_submission_path": str(sample_submission_path),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Labeled SelfTrain Submission Override",
        "",
        f"- Override dataset: `{dataset}`",
        f"- Route name: `{args.route_name}`",
        f"- Checkpoint path: `{checkpoint_path}`",
        f"- Checkpoint epoch: `{checkpoint.get('epoch')}`",
        f"- Threshold: `{float(threshold)}`",
        f"- Horizontal-flip TTA: `{bool(args.horizontal_flip_tta)}`",
        f"- Base predictions: `{args.base_predictions.resolve()}`",
        "",
        "## Route Summary",
        "",
        *_build_markdown_table(route_df),
        "",
        "## Test Cluster Summary",
        "",
        *_build_markdown_table(summary_df),
        "",
        "## Notes",
        "",
        f"- Student backbone: `{checkpoint_config['student_backbone']}`",
        f"- Input size: `{checkpoint_config['input_size']}`",
        f"- Embedding dim: `{checkpoint_config['embedding_dim']}`",
        f"- Eval manifest: `{test_manifest_path}`",
        "",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[labeled_selftrain_submission_override] submission: {output_dir / 'submission.csv'}")
    print(f"[labeled_selftrain_submission_override] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[labeled_selftrain_submission_override] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
