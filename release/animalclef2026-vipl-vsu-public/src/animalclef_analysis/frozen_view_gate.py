from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .descriptor_baselines import run_descriptor_baseline, run_descriptor_fusion_baseline
from .view_manifests import BODY_AXIS_VIEW_NAME, DEFAULT_VIEW_NAME, get_manifest_paths_for_view


SUPPORTED_DESCRIPTORS = ("mega", "miew", "fusion")


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def collect_run_metrics(run_dir: Path, *, view_name: str, descriptor_name: str) -> pd.DataFrame:
    best_df = pd.read_csv(run_dir / "tables" / "best_thresholds_v1.csv")
    recall_df = pd.read_csv(run_dir / "tables" / "val_recall_v1.csv")
    summary_path = run_dir / "reports" / "summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    merged = best_df.merge(recall_df, on="dataset", how="left")
    merged["view_name"] = view_name
    merged["descriptor"] = descriptor_name
    merged["artifact_dir"] = str(run_dir)
    merged["val_identity_fraction"] = float(summary_payload.get("val_identity_fraction", 0.0))
    merged["fallback_threshold"] = float(summary_payload.get("fallback_threshold", 0.0))
    ordered_columns = [
        "view_name",
        "descriptor",
        "dataset",
        "threshold",
        "ari",
        "nmi",
        "pairwise_precision",
        "pairwise_recall",
        "pairwise_f1",
        "recall_at_1",
        "recall_at_5",
        "cluster_count",
        "singleton_cluster_ratio",
        "samples",
        "fallback_threshold",
        "val_identity_fraction",
        "artifact_dir",
    ]
    return merged[ordered_columns].sort_values(["descriptor", "dataset"]).reset_index(drop=True)


def build_delta_table(metrics_df: pd.DataFrame, *, baseline_view: str) -> pd.DataFrame:
    baseline_df = metrics_df[metrics_df["view_name"] == baseline_view].copy()
    compare_df = metrics_df[metrics_df["view_name"] != baseline_view].copy()
    if baseline_df.empty or compare_df.empty:
        return pd.DataFrame(
            columns=[
                "descriptor",
                "dataset",
                "baseline_view",
                "candidate_view",
                "baseline_ari",
                "candidate_ari",
                "delta_ari",
                "baseline_recall_at_1",
                "candidate_recall_at_1",
                "delta_recall_at_1",
            ]
        )

    merged = compare_df.merge(
        baseline_df[["descriptor", "dataset", "ari", "recall_at_1"]].rename(
            columns={
                "ari": "baseline_ari",
                "recall_at_1": "baseline_recall_at_1",
            }
        ),
        on=["descriptor", "dataset"],
        how="left",
    )
    merged["candidate_view"] = merged["view_name"]
    merged["baseline_view"] = baseline_view
    merged["candidate_ari"] = merged["ari"]
    merged["candidate_recall_at_1"] = merged["recall_at_1"]
    merged["delta_ari"] = (merged["candidate_ari"] - merged["baseline_ari"]).round(6)
    merged["delta_recall_at_1"] = (
        merged["candidate_recall_at_1"] - merged["baseline_recall_at_1"]
    ).round(6)
    return merged[
        [
            "descriptor",
            "dataset",
            "baseline_view",
            "candidate_view",
            "baseline_ari",
            "candidate_ari",
            "delta_ari",
            "baseline_recall_at_1",
            "candidate_recall_at_1",
            "delta_recall_at_1",
        ]
    ].sort_values(["descriptor", "dataset", "candidate_view"]).reset_index(drop=True)


def write_frozen_view_gate_summary(
    output_path: Path,
    *,
    config: dict[str, Any],
    metrics_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    lines = [
        "# Frozen View Gate",
        "",
        "## Config",
        "",
        f"- Manifest root: `{config['manifest_root']}`",
        f"- Views: `{', '.join(config['views'])}`",
        f"- Descriptors: `{', '.join(config['descriptors'])}`",
        f"- Device: `{config['device']}`",
        f"- Num workers: `{config['num_workers']}`",
        f"- Val identity fraction: `{config['val_identity_fraction']}`",
        f"- Split seed: `{config['split_seed']}`",
        "",
        "## Metrics",
        "",
        dataframe_to_markdown_table(metrics_df),
        "",
        "## Delta vs Original",
        "",
        dataframe_to_markdown_table(delta_df),
        "",
        "## Reading Notes",
        "",
        f"- Main gate target: `{BODY_AXIS_VIEW_NAME}` on `SalamanderID2025`.",
        "- `delta_ari > 0` means the candidate view improves clustering on the local validation split.",
        "- `delta_recall_at_1` is a sanity check for neighbor quality before clustering.",
        "- `TexasHornedLizards` has no labeled validation split, so this gate remains inference-side only.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_frozen_view_gate(
    repo_root: Path,
    manifest_root: Path,
    output_dir: Path,
    *,
    views: list[str] | None = None,
    descriptors: list[str] | None = None,
    device: str = "cuda:0",
    num_workers: int = 4,
    val_identity_fraction: float = 0.1,
    split_seed: int = 42,
    thresholds: list[float] | None = None,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    manifest_root = manifest_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    tables_dir = output_dir / "tables"
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    selected_views = list(views or [DEFAULT_VIEW_NAME, BODY_AXIS_VIEW_NAME])
    selected_descriptors = list(descriptors or ["miew"])
    unsupported = [name for name in selected_descriptors if name not in SUPPORTED_DESCRIPTORS]
    if unsupported:
        raise ValueError(f"Unsupported descriptors: {unsupported}")

    metrics_frames: list[pd.DataFrame] = []
    written_paths: dict[str, Path] = {}
    need_fusion = "fusion" in selected_descriptors
    base_descriptors = [name for name in selected_descriptors if name in {"mega", "miew"}]
    if need_fusion:
        for required in ["mega", "miew"]:
            if required not in base_descriptors:
                base_descriptors.append(required)

    for view_name in selected_views:
        train_manifest_path, test_manifest_path = get_manifest_paths_for_view(manifest_root=manifest_root, view_name=view_name)
        view_output_dir = output_dir / view_name
        view_output_dir.mkdir(parents=True, exist_ok=True)

        base_output_dirs: dict[str, Path] = {}
        for descriptor_name in base_descriptors:
            descriptor_output_dir = view_output_dir / descriptor_name
            run_descriptor_baseline(
                repo_root=repo_root,
                output_dir=descriptor_output_dir,
                descriptor=descriptor_name,
                device=device,
                num_workers=num_workers,
                val_identity_fraction=val_identity_fraction,
                thresholds=thresholds,
                split_seed=split_seed,
                train_manifest_path=train_manifest_path,
                test_manifest_path=test_manifest_path,
            )
            base_output_dirs[descriptor_name] = descriptor_output_dir
            if descriptor_name in selected_descriptors:
                metrics_frames.append(
                    collect_run_metrics(
                        run_dir=descriptor_output_dir,
                        view_name=view_name,
                        descriptor_name=descriptor_name,
                    )
                )
            written_paths[f"{view_name}_{descriptor_name}_dir"] = descriptor_output_dir

        if need_fusion:
            fusion_output_dir = view_output_dir / "fusion"
            run_descriptor_fusion_baseline(
                repo_root=repo_root,
                output_dir=fusion_output_dir,
                source_dirs=[base_output_dirs["mega"], base_output_dirs["miew"]],
                component_names=["mega", "miew"],
                weights=[1.0, 1.0],
                thresholds=thresholds,
                split_seed=split_seed,
            )
            metrics_frames.append(
                collect_run_metrics(
                    run_dir=fusion_output_dir,
                    view_name=view_name,
                    descriptor_name="fusion",
                )
            )
            written_paths[f"{view_name}_fusion_dir"] = fusion_output_dir

    metrics_df = (
        pd.concat(metrics_frames, ignore_index=True)
        if metrics_frames
        else pd.DataFrame(
            columns=[
                "view_name",
                "descriptor",
                "dataset",
                "threshold",
                "ari",
                "nmi",
                "pairwise_precision",
                "pairwise_recall",
                "pairwise_f1",
                "recall_at_1",
                "recall_at_5",
                "cluster_count",
                "singleton_cluster_ratio",
                "samples",
                "fallback_threshold",
                "val_identity_fraction",
                "artifact_dir",
            ]
        )
    )
    metrics_path = tables_dir / "gate_metrics_v1.csv"
    metrics_df.to_csv(metrics_path, index=False)
    delta_df = build_delta_table(metrics_df=metrics_df, baseline_view=DEFAULT_VIEW_NAME)
    delta_path = tables_dir / "gate_deltas_v1.csv"
    delta_df.to_csv(delta_path, index=False)

    config = {
        "manifest_root": str(manifest_root),
        "views": selected_views,
        "descriptors": selected_descriptors,
        "device": device,
        "num_workers": num_workers,
        "val_identity_fraction": val_identity_fraction,
        "split_seed": split_seed,
        "thresholds": thresholds,
    }
    summary_path = reports_dir / "summary.md"
    write_frozen_view_gate_summary(
        summary_path,
        config=config,
        metrics_df=metrics_df,
        delta_df=delta_df,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                **config,
                "metrics_path": str(metrics_path),
                "delta_path": str(delta_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    written_paths["metrics_path"] = metrics_path
    written_paths["delta_path"] = delta_path
    written_paths["summary_path"] = summary_path
    return written_paths
