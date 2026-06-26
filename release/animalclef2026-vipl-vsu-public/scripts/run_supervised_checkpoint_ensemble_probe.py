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


def _resolve_component_names(checkpoint_paths: list[Path], component_names: list[str] | None) -> list[str]:
    if component_names is None:
        return [path.parent.parent.name for path in checkpoint_paths]
    if len(component_names) != len(checkpoint_paths):
        raise ValueError("component_names length must match checkpoint_paths length")
    return [str(value) for value in component_names]


def _resolve_weights(checkpoint_paths: list[Path], weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0] * len(checkpoint_paths)
    if len(weights) != len(checkpoint_paths):
        raise ValueError("weights length must match checkpoint_paths length")
    return [float(value) for value in weights]


def _fuse_same_dim_mean_l2(blocks: list[np.ndarray], weights: list[float], l2_normalize) -> np.ndarray:
    if not blocks:
        raise ValueError("Need at least one block for mean_l2 fusion")
    running = np.zeros_like(blocks[0], dtype=np.float32)
    for index, (block, weight) in enumerate(zip(blocks, weights, strict=True)):
        if block.shape != blocks[0].shape:
            raise ValueError(
                f"mean_l2 requires same shape for all blocks, but index 0 has {blocks[0].shape} and index {index} has {block.shape}"
            )
        running += l2_normalize(block.astype(np.float32, copy=False)) * float(weight)
    return l2_normalize(running.astype(np.float32, copy=False))


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import (
        PATH_COLUMN,
        apply_thresholds_to_df,
        build_submission,
        dataframe_to_markdown_table,
        fuse_embedding_blocks,
        l2_normalize,
        run_threshold_sweep,
    )
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path
    from animalclef_analysis.submission_baseline import _load_supervised_model_from_checkpoint
    from animalclef_analysis.supervised_training import extract_student_embeddings
    from animalclef_analysis.view_manifests import get_default_manifest_paths

    parser = argparse.ArgumentParser(
        description="Probe a single-dataset ensemble built from multiple supervised checkpoints."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--component-names", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--fusion-modes", nargs="+", default=["mean_l2", "concat_l2"])
    parser.add_argument("--val-metadata-path", type=Path, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--export-test-override", action="store_true")
    parser.add_argument("--route-name", type=str, default="")
    parser.add_argument("--base-predictions-path", type=Path)
    parser.add_argument("--sample-submission-path", type=Path)
    parser.add_argument("--test-manifest-path", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    checkpoint_paths = [path.resolve() for path in args.checkpoint_paths]
    component_names = _resolve_component_names(checkpoint_paths=checkpoint_paths, component_names=args.component_names)
    weights = _resolve_weights(checkpoint_paths=checkpoint_paths, weights=args.weights)
    fusion_modes = [str(value) for value in args.fusion_modes]
    supported_fusion_modes = {"mean_l2", "concat_l2"}
    unsupported = sorted(set(fusion_modes) - supported_fusion_modes)
    if unsupported:
        raise ValueError(f"Unsupported fusion modes: {unsupported}")

    dataset = str(args.dataset)
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    embeddings_dir = output_dir / "embeddings"
    for path in [output_dir, tables_dir, reports_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_df = pd.read_csv(args.val_metadata_path.resolve())
    val_df["image_id"] = val_df["image_id"].astype(str)
    if PATH_COLUMN in val_df.columns:
        val_df[PATH_COLUMN] = val_df[PATH_COLUMN].astype(str)
    val_df = val_df[val_df["dataset"].astype(str) == dataset].copy().reset_index(drop=True)
    if val_df.empty:
        raise ValueError(f"No rows found for dataset={dataset} in {args.val_metadata_path.resolve()}")

    component_rows: list[dict[str, object]] = []
    val_blocks: list[np.ndarray] = []
    for component_name, checkpoint_path, weight in zip(component_names, checkpoint_paths, weights, strict=True):
        model, spec, checkpoint_config, checkpoint = _load_supervised_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=str(args.device),
        )
        val_embeddings = extract_student_embeddings(
            df=val_df,
            repo_root=repo_root,
            model=model,
            spec=spec,
            device=str(args.device),
            batch_size=int(args.eval_batch_size),
            num_workers=int(args.num_workers),
        )
        np.save(embeddings_dir / f"{dataset}_{component_name}_val_embeddings.npy", val_embeddings.astype(np.float32))
        val_blocks.append(val_embeddings)
        component_rows.append(
            {
                "component": component_name,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "weight": float(weight),
                "embedding_dim": int(val_embeddings.shape[1]),
                "student_backbone": str(checkpoint_config.get("student_backbone", "")),
                "split_seed": checkpoint_config.get("split_seed"),
            }
        )

    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(tables_dir / "component_table_v1.csv", index=False)

    component_best_rows: list[dict[str, object]] = []
    for component_name, block in zip(component_names, val_blocks, strict=True):
        sweep_df, _pred_df = run_threshold_sweep(df=val_df, embeddings=block, thresholds=[float(value) for value in args.thresholds])
        best_row = sweep_df.sort_values(
            ["ari", "pairwise_f1", "nmi", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        component_best_rows.append(
            {
                "route": component_name,
                "fusion_mode": "single",
                "threshold": float(best_row["threshold"]),
                "ari": float(best_row["ari"]),
                "nmi": float(best_row["nmi"]),
                "pairwise_f1": float(best_row["pairwise_f1"]),
                "cluster_count": int(best_row["cluster_count"]),
                "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
            }
        )
    component_best_df = pd.DataFrame(component_best_rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "route"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    component_best_df.to_csv(tables_dir / "component_best_v1.csv", index=False)

    fusion_rows: list[dict[str, object]] = []
    best_payload: dict[str, object] | None = None
    best_sort_key: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, 1.0)
    for fusion_mode in fusion_modes:
        if fusion_mode == "mean_l2":
            fused_val_embeddings = _fuse_same_dim_mean_l2(val_blocks, weights=weights, l2_normalize=l2_normalize)
        else:
            fused_val_embeddings = fuse_embedding_blocks(val_blocks, weights=weights).astype(np.float32, copy=False)
        np.save(embeddings_dir / f"{dataset}_{fusion_mode}_val_embeddings.npy", fused_val_embeddings.astype(np.float32))
        sweep_df, prediction_df = run_threshold_sweep(
            df=val_df,
            embeddings=fused_val_embeddings,
            thresholds=[float(value) for value in args.thresholds],
        )
        sweep_df["fusion_mode"] = fusion_mode
        best_row = sweep_df.sort_values(
            ["ari", "pairwise_f1", "nmi", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        best_component_ari = float(component_best_df["ari"].max()) if not component_best_df.empty else float("-inf")
        best_component_pairwise_f1 = float(component_best_df["pairwise_f1"].max()) if not component_best_df.empty else float("-inf")
        row = {
            "route": f"{dataset}_multiseed",
            "fusion_mode": fusion_mode,
            "threshold": float(best_row["threshold"]),
            "ari": float(best_row["ari"]),
            "nmi": float(best_row["nmi"]),
            "pairwise_f1": float(best_row["pairwise_f1"]),
            "cluster_count": int(best_row["cluster_count"]),
            "singleton_cluster_ratio": float(best_row["singleton_cluster_ratio"]),
            "ari_delta_vs_best_single": round(float(best_row["ari"]) - best_component_ari, 6),
            "pairwise_f1_delta_vs_best_single": round(float(best_row["pairwise_f1"]) - best_component_pairwise_f1, 6),
        }
        fusion_rows.append(row)
        sort_key = (
            float(best_row["ari"]),
            float(best_row["pairwise_f1"]),
            float(best_row["nmi"]),
            -float(best_row["threshold"]),
        )
        if sort_key > best_sort_key:
            best_sort_key = sort_key
            best_payload = {
                "fusion_mode": fusion_mode,
                "fused_val_embeddings": fused_val_embeddings.astype(np.float32, copy=False),
                "sweep_df": sweep_df.copy(),
                "prediction_df": prediction_df.copy(),
                "best_row": row,
            }
        sweep_df.to_csv(tables_dir / f"{fusion_mode}_threshold_sweep_v1.csv", index=False)
        prediction_df.to_csv(tables_dir / f"{fusion_mode}_predictions_v1.csv", index=False)

    if best_payload is None:
        raise RuntimeError("No ensemble fusion mode was evaluated.")

    fusion_best_df = pd.DataFrame(fusion_rows).sort_values(
        ["ari", "pairwise_f1", "nmi", "fusion_mode"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    fusion_best_df.to_csv(tables_dir / "fusion_best_v1.csv", index=False)
    pd.DataFrame([best_payload["best_row"]]).to_csv(tables_dir / "best_config_v1.csv", index=False)
    val_df.to_csv(embeddings_dir / f"{dataset}_val_metadata.csv", index=False)

    export_paths: dict[str, str] = {}
    if args.export_test_override:
        if args.base_predictions_path is None:
            raise ValueError("--export-test-override requires --base-predictions-path")
        sample_submission_path = (
            args.sample_submission_path.resolve() if args.sample_submission_path else repo_root / "sample_submission.csv"
        )
        if args.test_manifest_path is not None:
            test_manifest_path = args.test_manifest_path.resolve()
        else:
            _train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
        test_df = pd.read_csv(test_manifest_path)
        test_df["image_id"] = test_df["image_id"].astype(str)
        test_df["dataset"] = test_df["dataset"].astype(str)
        test_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in test_df.iterrows()]
        dataset_test_df = test_df[test_df["dataset"] == dataset].copy().reset_index(drop=True)
        if dataset_test_df.empty:
            raise ValueError(f"No test rows found for dataset={dataset} in {test_manifest_path}")

        test_blocks: list[np.ndarray] = []
        for component_name, checkpoint_path in zip(component_names, checkpoint_paths, strict=True):
            model, spec, _config, _checkpoint = _load_supervised_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                device=str(args.device),
            )
            test_embeddings = extract_student_embeddings(
                df=dataset_test_df,
                repo_root=repo_root,
                model=model,
                spec=spec,
                device=str(args.device),
                batch_size=int(args.eval_batch_size),
                num_workers=int(args.num_workers),
            )
            np.save(embeddings_dir / f"{dataset}_{component_name}_test_embeddings.npy", test_embeddings.astype(np.float32))
            test_blocks.append(test_embeddings)

        if str(best_payload["fusion_mode"]) == "mean_l2":
            fused_test_embeddings = _fuse_same_dim_mean_l2(test_blocks, weights=weights, l2_normalize=l2_normalize)
        else:
            fused_test_embeddings = fuse_embedding_blocks(test_blocks, weights=weights).astype(np.float32, copy=False)
        np.save(embeddings_dir / f"{dataset}_best_test_embeddings.npy", fused_test_embeddings.astype(np.float32))
        dataset_test_df.to_csv(embeddings_dir / f"{dataset}_test_metadata.csv", index=False)

        chosen_threshold = float(best_payload["best_row"]["threshold"])
        override_pred_df = apply_thresholds_to_df(
            df=dataset_test_df,
            embeddings=fused_test_embeddings,
            threshold_by_dataset={dataset: chosen_threshold},
        )
        route_name = str(args.route_name) if str(args.route_name).strip() else f"{dataset.lower()}_multiseed_{best_payload['fusion_mode']}_v1"
        override_pred_df["route_name"] = route_name
        override_pred_df["embedding_dim"] = int(fused_test_embeddings.shape[1])
        override_pred_df["rerank_enabled"] = False
        override_pred_df["local_weight"] = 0.0

        base_pred_df = pd.read_csv(args.base_predictions_path.resolve())
        base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
        base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
        merged_pred_df = pd.concat(
            [base_pred_df[base_pred_df["dataset"] != dataset].copy(), override_pred_df],
            ignore_index=True,
        )
        merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
        route_rows: list[dict[str, object]] = []
        cluster_rows: list[dict[str, object]] = []
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
            counts = frame["pred_cluster_id"].value_counts()
            cluster_rows.append(
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
        route_df = pd.DataFrame(route_rows).sort_values("dataset").reset_index(drop=True)
        cluster_df = pd.DataFrame(cluster_rows).sort_values("dataset").reset_index(drop=True)
        route_df.to_csv(tables_dir / "route_config_v1.csv", index=False)
        cluster_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)
        build_submission(
            test_pred_df=merged_pred_df,
            sample_submission_path=sample_submission_path,
            output_path=output_dir / "submission.csv",
        )
        export_paths = {
            "submission_path": str((output_dir / "submission.csv").resolve()),
            "test_predictions_path": str((tables_dir / "test_predictions_v1.csv").resolve()),
            "route_config_path": str((tables_dir / "route_config_v1.csv").resolve()),
            "cluster_summary_path": str((tables_dir / "cluster_summary_v1.csv").resolve()),
            "test_embeddings_path": str((embeddings_dir / f"{dataset}_best_test_embeddings.npy").resolve()),
            "test_metadata_path": str((embeddings_dir / f"{dataset}_test_metadata.csv").resolve()),
        }

    summary_payload = {
        "dataset": dataset,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "component_names": component_names,
        "weights": weights,
        "fusion_modes": fusion_modes,
        "thresholds": [float(value) for value in args.thresholds],
        "best_config": best_payload["best_row"],
        "export_paths": export_paths,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Supervised Checkpoint Ensemble Probe",
        "",
        f"- Dataset: `{dataset}`",
        f"- Checkpoint count: `{len(checkpoint_paths)}`",
        f"- Fusion modes: `{fusion_modes}`",
        f"- Best config: `{best_payload['best_row']}`",
        "",
        "## Components",
        "",
        *_build_markdown_table(component_df),
        "",
        "## Best Single Checkpoints On This Validation Split",
        "",
        *_build_markdown_table(component_best_df),
        "",
        "## Ensemble Results",
        "",
        *_build_markdown_table(fusion_best_df),
        "",
        "## Notes",
        "",
        "- `single` rows mean: each checkpoint is re-run on the same validation split independently, then threshold-swept.",
        "- `mean_l2` means: L2-normalized same-dim embeddings are weighted-summed, then L2-normalized again.",
        "- `concat_l2` means: L2-normalized component embeddings are weighted, concatenated, and L2-normalized.",
    ]
    if export_paths:
        lines.extend(
            [
                "",
                "## Test Override",
                "",
                f"- Submission: `{export_paths['submission_path']}`",
                f"- Test predictions: `{export_paths['test_predictions_path']}`",
                f"- Test embeddings: `{export_paths['test_embeddings_path']}`",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[supervised_checkpoint_ensemble_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[supervised_checkpoint_ensemble_probe] component_best: {tables_dir / 'component_best_v1.csv'}")
    print(f"[supervised_checkpoint_ensemble_probe] fusion_best: {tables_dir / 'fusion_best_v1.csv'}")
    if export_paths:
        print(f"[supervised_checkpoint_ensemble_probe] submission: {output_dir / 'submission.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
