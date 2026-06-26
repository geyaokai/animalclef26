#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


SEED_PATTERN = re.compile(r"^(?P<family>.+)_seed(?P<seed>\d+)$")


def _parse_experiment_family(experiment_id: str) -> tuple[str, int]:
    match = SEED_PATTERN.match(experiment_id)
    if not match:
        raise ValueError(f"Experiment id does not match '<family>_seed<int>': {experiment_id}")
    return match.group("family"), int(match.group("seed"))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_mode(series: pd.Series) -> str:
    if series.empty:
        return ""
    mode = series.mode(dropna=True)
    if mode.empty:
        return ""
    return str(mode.iloc[0])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table

    parser = argparse.ArgumentParser(description="Summarize supervised retrain v2 results across seeds.")
    parser.add_argument("--experiments-root", type=Path, default=repo_root / "artifacts" / "training" / "experiments")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts" / "training" / "analysis" / "supervised_retrain_matrix_v2")
    parser.add_argument("--experiment-glob", type=str, default="*_rtv2_seed*")
    args = parser.parse_args()

    experiments_root = args.experiments_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    experiment_dirs = sorted(path for path in experiments_root.glob(args.experiment_glob) if path.is_dir())
    run_rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []

    for experiment_dir in experiment_dirs:
        summary_path = experiment_dir / "reports" / "summary.json"
        best_path = experiment_dir / "tables" / "best_checkpoints_v1.csv"
        training_log_path = experiment_dir / "tables" / "training_log_v1.csv"
        if not summary_path.exists() or not best_path.exists() or not training_log_path.exists():
            continue
        summary_payload = _read_json(summary_path)
        experiment_id = str(summary_payload["experiment_id"])
        family, split_seed = _parse_experiment_family(experiment_id)
        training_log_df = pd.read_csv(training_log_path)
        final_epoch = int(training_log_df["epoch"].max()) if not training_log_df.empty else 0
        run_rows.append(
            {
                "experiment_id": experiment_id,
                "family": family,
                "split_seed": split_seed,
                "student_backbone": str(summary_payload.get("student_backbone", "")),
                "teacher_cache_dir": str(summary_payload.get("teacher_cache_dir", "")),
                "final_epoch": final_epoch,
                "best_macro_epoch": int(summary_payload.get("best_epoch", -1)),
                "best_macro_ari": float(summary_payload.get("best_macro_ari", float("nan"))),
                "best_LynxID2025_epoch": int(summary_payload.get("best_metric_epochs", {}).get("LynxID2025", -1)),
                "best_LynxID2025_ari": float(summary_payload.get("best_metric_scores", {}).get("LynxID2025", float("nan"))),
                "best_SalamanderID2025_epoch": int(summary_payload.get("best_metric_epochs", {}).get("SalamanderID2025", -1)),
                "best_SalamanderID2025_ari": float(summary_payload.get("best_metric_scores", {}).get("SalamanderID2025", float("nan"))),
                "best_SeaTurtleID2022_epoch": int(summary_payload.get("best_metric_epochs", {}).get("SeaTurtleID2022", -1)),
                "best_SeaTurtleID2022_ari": float(summary_payload.get("best_metric_scores", {}).get("SeaTurtleID2022", float("nan"))),
            }
        )
        best_df = pd.read_csv(best_path)
        best_df["experiment_id"] = experiment_id
        best_df["family"] = family
        best_df["split_seed"] = split_seed
        best_rows.append(best_df)

    run_df = pd.DataFrame(run_rows).sort_values(["family", "split_seed"]).reset_index(drop=True) if run_rows else pd.DataFrame()
    best_long_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    if not run_df.empty:
        aggregate_rows: list[dict[str, object]] = []
        for family, family_df in run_df.groupby("family", dropna=False):
            row: dict[str, object] = {
                "family": family,
                "runs": int(len(family_df)),
                "student_backbone": _safe_mode(family_df["student_backbone"]),
                "seeds": ",".join(str(int(seed)) for seed in sorted(family_df["split_seed"].tolist())),
            }
            for metric in [
                "best_macro_ari",
                "best_LynxID2025_ari",
                "best_SalamanderID2025_ari",
                "best_SeaTurtleID2022_ari",
            ]:
                row[f"{metric}_mean"] = round(float(family_df[metric].mean()), 6)
                row[f"{metric}_std"] = round(float(family_df[metric].std(ddof=0)), 6)
            for epoch_metric in [
                "best_macro_epoch",
                "best_LynxID2025_epoch",
                "best_SalamanderID2025_epoch",
                "best_SeaTurtleID2022_epoch",
            ]:
                row[f"{epoch_metric}_mode"] = _safe_mode(family_df[epoch_metric].astype(str))
            aggregate_rows.append(row)
        aggregate_df = pd.DataFrame(aggregate_rows).sort_values("family").reset_index(drop=True)
    else:
        aggregate_df = pd.DataFrame()

    run_df.to_csv(tables_dir / "retrain_runs_v1.csv", index=False)
    best_long_df.to_csv(tables_dir / "retrain_best_checkpoints_long_v1.csv", index=False)
    aggregate_df.to_csv(tables_dir / "retrain_family_summary_v1.csv", index=False)

    lines = [
        "# Supervised Retrain Matrix V2 Summary",
        "",
        f"- experiments_root: `{experiments_root}`",
        f"- experiment_glob: `{args.experiment_glob}`",
        f"- loaded_runs: `{len(run_df)}`",
        "",
    ]
    if not run_df.empty:
        lines.extend(
            [
                "## Per-Run Summary",
                "",
                dataframe_to_markdown_table(run_df),
                "",
            ]
        )
    if not aggregate_df.empty:
        lines.extend(
            [
                "## Family Aggregate Summary",
                "",
                dataframe_to_markdown_table(aggregate_df),
                "",
                "## Reading Notes",
                "",
                "- 先看 `best_macro_ari_mean`，判断整条训练线跨 seed 是否稳定。",
                "- 再看 `best_LynxID2025_ari_mean / best_SalamanderID2025_ari_mean / best_SeaTurtleID2022_ari_mean`，判断各 dataset 的 backbone 归属是否稳定。",
                "- 再看各 `*_epoch_mode`，判断 dataset-specific best checkpoint 是否集中在相近 epoch，还是波动很大。",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[retrain_summary] summary: {output_dir / 'summary.md'}")
    print(f"[retrain_summary] runs: {tables_dir / 'retrain_runs_v1.csv'}")
    print(f"[retrain_summary] family_summary: {tables_dir / 'retrain_family_summary_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
