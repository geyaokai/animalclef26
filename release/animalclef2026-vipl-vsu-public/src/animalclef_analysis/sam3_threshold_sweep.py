from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sam3_probe import run_sam3_probe


SUMMARY_COLUMNS = [
    "threshold",
    "mask_threshold",
    "dataset",
    "split",
    "samples",
    "positive_masks",
    "positive_ratio",
    "mean_area_ratio",
    "mean_best_score",
]


def threshold_tag(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def summarize_probe_results(
    results_df: pd.DataFrame,
    threshold: float,
    mask_threshold: float,
) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary_df = (
        results_df.groupby(["dataset", "split"])
        .agg(
            samples=("image_id", "count"),
            positive_masks=("mask_count", lambda s: int((s > 0).sum())),
            positive_ratio=("mask_count", lambda s: round(float((s > 0).mean()), 4)),
            mean_area_ratio=("mask_area_ratio", lambda s: round(float(np.mean(s)), 4)),
            mean_best_score=("best_score", lambda s: round(float(np.mean(s)), 4)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )
    summary_df.insert(0, "mask_threshold", mask_threshold)
    summary_df.insert(0, "threshold", threshold)
    return summary_df[SUMMARY_COLUMNS]


def as_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def write_combined_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    run_records: list[dict[str, object]],
    samples_per_split: int,
    sample_seed: int,
) -> None:
    run_table = pd.DataFrame(run_records)
    lines = [
        "# SAM3 Threshold Sweep Summary",
        "",
        f"- Samples per dataset/split: `{samples_per_split}`",
        f"- Sample seed: `{sample_seed}`",
        "",
        "## Runs",
        "",
        as_markdown_table(run_table),
        "",
        "## Aggregate Results",
        "",
        as_markdown_table(summary_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_sam3_threshold_sweep(
    repo_root: Path,
    output_dir: Path,
    thresholds: list[float],
    datasets: list[str],
    samples_per_split: int = 8,
    sample_seed: int = 42,
    device: str = "cuda:0",
    mask_thresholds: list[float] | None = None,
) -> dict[str, Path]:
    if mask_thresholds is None:
        mask_thresholds = list(thresholds)
    if len(thresholds) != len(mask_thresholds):
        raise ValueError("thresholds and mask_thresholds must have the same length")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_frames: list[pd.DataFrame] = []
    run_records: list[dict[str, object]] = []
    for threshold, mask_threshold in zip(thresholds, mask_thresholds):
        run_tag = f"thr_{threshold_tag(threshold)}__mask_{threshold_tag(mask_threshold)}"
        run_dir = output_dir / run_tag
        outputs = run_sam3_probe(
            repo_root=repo_root,
            output_dir=run_dir,
            datasets=datasets,
            samples_per_split=samples_per_split,
            sample_seed=sample_seed,
            threshold=threshold,
            mask_threshold=mask_threshold,
            device=device,
        )
        results_df = pd.read_csv(outputs["results_path"])
        summary_frames.append(
            summarize_probe_results(
                results_df=results_df,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
        )
        run_records.append(
            {
                "run_tag": run_tag,
                "threshold": threshold,
                "mask_threshold": mask_threshold,
                "results_path": str(outputs["results_path"].relative_to(repo_root)),
                "summary_path": str(outputs["summary_path"].relative_to(repo_root)),
                "qualitative_dir": str(outputs["qualitative_dir"].relative_to(repo_root)),
            }
        )

    combined_summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame(columns=SUMMARY_COLUMNS)
    )
    combined_csv_path = tables_dir / "sam3_threshold_sweep_summary.csv"
    combined_summary.to_csv(combined_csv_path, index=False)

    summary_md_path = reports_dir / "summary.md"
    write_combined_summary(
        summary_df=combined_summary,
        output_path=summary_md_path,
        run_records=run_records,
        samples_per_split=samples_per_split,
        sample_seed=sample_seed,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "mask_thresholds": mask_thresholds,
                "datasets": datasets,
                "samples_per_split": samples_per_split,
                "sample_seed": sample_seed,
                "device": device,
                "runs": run_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv_path": combined_csv_path,
        "summary_md_path": summary_md_path,
        "output_dir": output_dir,
    }
