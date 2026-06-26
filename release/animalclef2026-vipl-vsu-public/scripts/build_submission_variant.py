#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import (
        apply_thresholds_to_df,
        build_submission,
        dataframe_to_markdown_table,
        load_cached_embedding_bundle,
    )

    parser = argparse.ArgumentParser(description="Build a submission variant by overriding one dataset route/threshold.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--override-dataset", type=str, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--route-name", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_pred_df = pd.read_csv(args.base_predictions)
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)

    bundle = load_cached_embedding_bundle(source_dir=args.source_dir.resolve())
    test_df = bundle.test_df.copy().reset_index(drop=True)
    test_df["image_id"] = test_df["image_id"].astype(str)
    dataset = str(args.override_dataset)
    dataset_df = test_df[test_df["dataset"] == dataset].copy().reset_index(drop=True)
    dataset_embeddings = bundle.test_embeddings[(test_df["dataset"] == dataset).to_numpy()]
    override_pred_df = apply_thresholds_to_df(
        df=dataset_df,
        embeddings=dataset_embeddings,
        threshold_by_dataset={dataset: float(args.threshold)},
    )
    override_pred_df["route_name"] = str(args.route_name)
    override_pred_df["embedding_dim"] = int(dataset_embeddings.shape[1])

    kept_df = base_pred_df[base_pred_df["dataset"] != dataset].copy()
    merged_pred_df = pd.concat([kept_df, override_pred_df], ignore_index=True)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    summary_rows = []
    for name, frame in merged_pred_df.groupby("dataset"):
        counts = frame["pred_cluster_id"].value_counts()
        summary_rows.append(
            {
                "dataset": name,
                "samples": int(len(frame)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(frame["route_name"].iloc[0]) if "route_name" in frame.columns else "",
                "chosen_threshold": float(frame["chosen_threshold"].iloc[0]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("dataset").reset_index(drop=True)
    summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=output_dir / "submission.csv",
    )

    config = {
        "base_predictions": str(args.base_predictions.resolve()),
        "override_dataset": dataset,
        "source_dir": str(args.source_dir.resolve()),
        "threshold": float(args.threshold),
        "route_name": str(args.route_name),
    }
    route_df = (
        merged_pred_df[
            [
                "dataset",
                "route_name",
                "embedding_dim",
                "chosen_threshold",
            ]
        ]
        .drop_duplicates(subset=["dataset"])
        .rename(columns={"chosen_threshold": "threshold"})
        .sort_values("dataset")
        .reset_index(drop=True)
    )
    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> optional rerank -> average-linkage clustering -> submission cluster label`.",
        "- Current route:",
    ]
    for row in route_df.itertuples(index=False):
        dataset_name = str(row.dataset)
        route_name = str(row.route_name)
        if dataset_name == dataset and "miew" in route_name.lower():
            desc = f"frozen `MiewID-msv3`, `B x 3 x 440 x 440 -> B x 2152`"
        elif dataset_name == dataset and "mega" in route_name.lower():
            desc = f"frozen `MegaDescriptor-L-384`, `B x 3 x 384 x 384 -> B x 1536`"
        elif dataset_name == dataset and "fusion" in route_name.lower():
            desc = "frozen early fusion, `Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`"
        elif route_name == "ft_mega_arcface_distill_v1":
            desc = "supervised `MegaDescriptor-L-384` student, `B x 3 x 384 x 384 -> B x 1536 -> B x 512`"
        elif route_name == "fusion_orb_rerank_v1":
            desc = "frozen early fusion, `Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`, then ORB rerank on top-K pairs"
        elif route_name == "fusion_v1":
            desc = "frozen early fusion, `Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`"
        else:
            desc = f"route `{route_name}`, embedding dim `{int(row.embedding_dim)}`"
        architecture_lines.append(
            f"  - `{dataset_name}`: {desc}, threshold `{float(row.threshold)}`."
        )

    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Submission Variant",
                "",
                f"- Override dataset: `{dataset}`",
                f"- Source dir: `{args.source_dir.resolve()}`",
                f"- Threshold: `{float(args.threshold)}`",
                f"- Route name: `{args.route_name}`",
                f"- Base predictions: `{args.base_predictions.resolve()}`",
                "",
                "## Architecture",
                "",
                *architecture_lines,
                "",
                "## Cluster Summary",
                "",
                dataframe_to_markdown_table(summary_df),
                "",
                "## Key Performance Tricks",
                "",
                "- `single-dataset override`: keep the rest of the submission fixed and modify only one dataset route or threshold, so score changes stay attributable.",
                f"- `current changed trick`: override `{dataset}` from the previous official route to `{args.route_name}` with threshold `{float(args.threshold)}`.",
                "- `submission-level attribution`: use this format for official variants so public/private gains can be tied back to one concrete architectural change.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"[submission_variant] submission: {output_dir / 'submission.csv'}")
    print(f"[submission_variant] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[submission_variant] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
