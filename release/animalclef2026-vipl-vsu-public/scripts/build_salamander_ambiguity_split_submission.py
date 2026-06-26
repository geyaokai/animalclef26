#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_BASE_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_lynx_seedsmooth_alpha0p15_onxgb_v1")
DEFAULT_SALAMANDER_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_v1")
DEFAULT_PROBE_DIR = Path("artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1")
SALAMANDER_DATASET = "SalamanderID2025"


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


def _summarize_clusters(pred_labels: np.ndarray) -> dict[str, float | int]:
    counts = pd.Series(pred_labels).value_counts()
    return {
        "clusters": int(counts.size),
        "singleton_clusters": int((counts == 1).sum()),
        "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
    }


def _split_clusters_into_singletons(
    pred_df: pd.DataFrame,
    *,
    cluster_ids: list[int],
    rule_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = pred_df.copy().reset_index(drop=True)
    result["base_pred_cluster_id"] = result["pred_cluster_id"].astype(int)
    result["base_cluster_label"] = result["cluster_label"].astype(str)
    result["ambiguity_overlay_enabled"] = False
    result["ambiguity_overlay_rule"] = ""
    changed_rows: list[dict[str, object]] = []

    next_cluster_id = int(result["pred_cluster_id"].astype(int).max()) + 1 if len(result) else 0
    for cluster_id in [int(value) for value in cluster_ids]:
        member_idx = result.index[result["pred_cluster_id"].astype(int).eq(cluster_id)].tolist()
        if len(member_idx) <= 1:
            continue
        member_idx = sorted(member_idx, key=lambda idx: str(result.at[idx, "image_id"]))
        anchor_idx = int(member_idx[0])
        anchor_image_id = str(result.at[anchor_idx, "image_id"])
        result.at[anchor_idx, "ambiguity_overlay_enabled"] = True
        result.at[anchor_idx, "ambiguity_overlay_rule"] = f"{rule_name}|split_cluster_{cluster_id}|keep_anchor"
        changed_rows.append(
            {
                "image_id": anchor_image_id,
                "base_pred_cluster_id": int(cluster_id),
                "overlay_pred_cluster_id": int(cluster_id),
                "overlay_action": "keep_anchor",
                "overlay_rule": rule_name,
            }
        )
        for idx in member_idx[1:]:
            new_cluster_id = int(next_cluster_id)
            next_cluster_id += 1
            result.at[idx, "pred_cluster_id"] = new_cluster_id
            result.at[idx, "cluster_label"] = f"cluster_{SALAMANDER_DATASET}_{new_cluster_id}"
            result.at[idx, "ambiguity_overlay_enabled"] = True
            result.at[idx, "ambiguity_overlay_rule"] = f"{rule_name}|split_cluster_{cluster_id}|singleton"
            changed_rows.append(
                {
                    "image_id": str(result.at[idx, "image_id"]),
                    "base_pred_cluster_id": int(cluster_id),
                    "overlay_pred_cluster_id": int(new_cluster_id),
                    "overlay_action": "singleton",
                    "overlay_rule": rule_name,
                }
            )
    changed_df = pd.DataFrame(changed_rows)
    return result, changed_df


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import build_submission

    parser = argparse.ArgumentParser(description="Build a Salamander ambiguity split submission overlay.")
    parser.add_argument("--base-submission-dir", type=Path, default=DEFAULT_BASE_SUBMISSION_DIR)
    parser.add_argument("--salamander-dir", type=Path, default=DEFAULT_SALAMANDER_DIR)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--split-cluster-ids", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-public-score", type=float, default=0.48755)
    parser.add_argument("--current-best-public-score", type=float, default=0.48876)
    parser.add_argument("--current-best-description", type=str, default="SeaTurtle cropped fusion w=0.3 t=0.7 overlay")
    parser.add_argument("--sample-submission-path", type=Path, default=repo_root / "sample_submission.csv")
    parser.add_argument("--submission-description", type=str, default="")
    args = parser.parse_args()

    base_submission_dir = (repo_root / args.base_submission_dir).resolve() if not args.base_submission_dir.is_absolute() else args.base_submission_dir.resolve()
    salamander_dir = (repo_root / args.salamander_dir).resolve() if not args.salamander_dir.is_absolute() else args.salamander_dir.resolve()
    probe_dir = (repo_root / args.probe_dir).resolve() if not args.probe_dir.is_absolute() else args.probe_dir.resolve()
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    base_pred_df = pd.read_csv(base_submission_dir / "tables" / "test_predictions_v1.csv")
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)

    salamander_pred_df = pd.read_csv(salamander_dir / "tables" / "salamander_test_predictions_v1.csv")
    salamander_pred_df["image_id"] = salamander_pred_df["image_id"].astype(str)
    salamander_pred_df["dataset"] = salamander_pred_df["dataset"].astype(str)
    salamander_pred_df["pred_cluster_id"] = salamander_pred_df["pred_cluster_id"].astype(int)
    salamander_pred_df["cluster_label"] = salamander_pred_df["cluster_label"].astype(str)
    salamander_pred_df["route_name"] = "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_split_overlay_v1"

    split_candidate_df = pd.read_csv(probe_dir / "tables" / "test_split_candidates_v1.csv")
    split_candidate_df["base_cluster_id"] = split_candidate_df["base_cluster_id"].astype(int)
    pair_df = pd.read_csv(probe_dir / "tables" / "test_pair_disagreement_v1.csv")
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)

    rule_name = "ambiguity_split_overlay_v1"
    overlay_df, changed_df = _split_clusters_into_singletons(
        salamander_pred_df,
        cluster_ids=[int(value) for value in args.split_cluster_ids],
        rule_name=rule_name,
    )
    overlay_df["route_name"] = "ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_split_overlay_v1"

    candidate_rows: list[pd.DataFrame] = []
    for cluster_id in [int(value) for value in args.split_cluster_ids]:
        cluster_meta = split_candidate_df[split_candidate_df["base_cluster_id"].astype(int).eq(cluster_id)].copy()
        if cluster_meta.empty:
            continue
        cluster_meta = cluster_meta.assign(requested_split_cluster_id=int(cluster_id))
        candidate_rows.append(cluster_meta)
    candidate_df = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    candidate_df.to_csv(tables_dir / "split_candidates_v1.csv", index=False)

    changed_df = changed_df.merge(
        overlay_df[
            [
                "image_id",
                "pred_cluster_id",
                "cluster_label",
                "path",
                "ambiguity_overlay_enabled",
                "ambiguity_overlay_rule",
            ]
        ].rename(
            columns={
                "pred_cluster_id": "final_pred_cluster_id",
                "cluster_label": "final_cluster_label",
            }
        ),
        on="image_id",
        how="left",
    )
    changed_df.to_csv(tables_dir / "changed_images_v1.csv", index=False)

    support_rows: list[pd.DataFrame] = []
    for cluster_id in [int(value) for value in args.split_cluster_ids]:
        support_df = pair_df[
            pair_df["base_cluster_left"].astype(int).eq(cluster_id)
            & pair_df["base_cluster_right"].astype(int).eq(cluster_id)
            & pair_df["vote_direction"].astype(str).eq("split")
        ].copy()
        support_df["requested_split_cluster_id"] = int(cluster_id)
        support_rows.append(support_df)
    support_df = pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame()
    support_df.to_csv(tables_dir / "split_support_pairs_v1.csv", index=False)

    base_stats = _summarize_clusters(salamander_pred_df["pred_cluster_id"].to_numpy(dtype=int))
    overlay_stats = _summarize_clusters(overlay_df["pred_cluster_id"].to_numpy(dtype=int))
    cluster_summary_df = pd.DataFrame(
        [
            {
                "dataset": SALAMANDER_DATASET,
                "base_clusters": int(base_stats["clusters"]),
                "overlay_clusters": int(overlay_stats["clusters"]),
                "base_singletons": int(base_stats["singleton_clusters"]),
                "overlay_singletons": int(overlay_stats["singleton_clusters"]),
                "changed_images": int(len(changed_df)),
                "changed_cluster_ids": "|".join(str(int(value)) for value in args.split_cluster_ids),
            }
        ]
    )
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    kept_df = base_pred_df[base_pred_df["dataset"].astype(str) != SALAMANDER_DATASET].copy()
    merged_pred_df = pd.concat([kept_df, overlay_df], ignore_index=True)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    overlay_df.to_csv(tables_dir / "salamander_test_predictions_v1.csv", index=False)

    submission_path = output_dir / "submission.csv"
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=args.sample_submission_path.resolve(),
        output_path=submission_path,
    )

    summary_json = {
        "base_submission_dir": str(base_submission_dir),
        "salamander_dir": str(salamander_dir),
        "probe_dir": str(probe_dir),
        "split_cluster_ids": [int(value) for value in args.split_cluster_ids],
        "submission_description": str(args.submission_description),
        "submission_path": str(submission_path),
        "test_predictions_path": str(tables_dir / "test_predictions_v1.csv"),
        "salamander_override_path": str(tables_dir / "salamander_test_predictions_v1.csv"),
        "changed_images_path": str(tables_dir / "changed_images_v1.csv"),
        "split_candidates_path": str(tables_dir / "split_candidates_v1.csv"),
        "split_support_pairs_path": str(tables_dir / "split_support_pairs_v1.csv"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    cluster_ids_text = ", ".join(str(int(value)) for value in args.split_cluster_ids)
    support_lines = (
        _build_markdown_table(
            support_df[
                [
                    "requested_split_cluster_id",
                    "image_id",
                    "neighbor_image_id",
                    "xgb_same_identity_prob",
                    "split_votes",
                    "ambiguity_score",
                    "conflict_methods",
                ]
            ]
        )
        if not support_df.empty
        else ["_No support pairs exported._"]
    )
    lines = [
        "# Submission Variant",
        "",
        f"- Baseline submission: `{base_submission_dir.name}`",
        f"- Baseline public score when this artifact family was created: `{float(args.baseline_public_score):.5f}`",
        f"- Note on current leaderboard state: current best public in Kaggle list is `{float(args.current_best_public_score):.5f}` from `{args.current_best_description}`, but its local artifact directory is not present in this workspace, so this overlay is built on the latest fully reproducible local best-official artifact.",
        f"- Override dataset: `{SALAMANDER_DATASET}`",
        f"- Only changed factor: keep the current `Salamander XGBoost + average-linkage @ 0.25` route fixed, and split exactly these A/B ambiguity clusters into smaller singleton pieces: `{cluster_ids_text}`.",
        "",
        "## Architecture",
        "",
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> pairwise fusion -> clustering -> submission cluster label`.",
        "- Current route:",
        f"  - `LynxID2025`: keep baseline route from `{base_submission_dir.name}` unchanged.",
        f"  - `SalamanderID2025`: keep `ft_miew_arcface_masked_supcon_v1_last_fusion_xgboost_v1`, threshold `0.25`, and apply a local ambiguity repair split overlay on clusters `{cluster_ids_text}`.",
        f"  - `SeaTurtleID2022`: keep baseline route unchanged.",
        f"  - `TexasHornedLizards`: keep baseline route unchanged.",
        "",
        "## Why These Splits",
        "",
        "- This candidate is not a global route swap.",
        "- It is a conservative local repair on top of the current official Salamander base graph.",
        "- The selected clusters come from the official-aligned ambiguity probe and were flagged as A/B-tier split risks.",
        "",
        "## Requested Split Candidates",
        "",
        *_build_markdown_table(candidate_df),
        "",
        "## Split Support Pairs",
        "",
        *support_lines,
        "",
        "## Changed Images",
        "",
        *_build_markdown_table(changed_df),
        "",
        "## Cluster Summary",
        "",
        *_build_markdown_table(cluster_summary_df),
        "",
        "## Validation",
        "",
        "- `submission.csv` row count matches `sample_submission.csv`.",
        "- Columns are exactly `image_id,cluster`.",
        "- `image_id` order matches `sample_submission.csv`.",
    ]
    if str(args.submission_description).strip():
        lines.extend(
            [
                "",
                "## Kaggle Submission",
                "",
                f"- Submission description: `{str(args.submission_description)}`",
            ]
        )
    (reports_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[salamander_ambiguity_split_submission] submission: {submission_path}")
    print(f"[salamander_ambiguity_split_submission] predictions: {tables_dir / 'test_predictions_v1.csv'}")
    print(f"[salamander_ambiguity_split_submission] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
