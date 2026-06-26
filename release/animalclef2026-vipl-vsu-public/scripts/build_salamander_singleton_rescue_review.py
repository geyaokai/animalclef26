#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_variant_seaturtle_origcrop_w0p3_on_gate2_v1")
DEFAULT_PAIR_FEATURES = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_v1/tables/test_pair_features_v1.csv")
DEFAULT_PAIR_JUDGMENTS = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_singleton_rescue_review_v1")


def _resolve_path(repo_root: Path, input_path: Path) -> Path:
    return input_path.resolve() if input_path.is_absolute() else (repo_root / input_path).resolve()


def _path_ref(base: Path, target: Path) -> str:
    return str(target.relative_to(base)).replace("\\", "/")


def _table_markdown(frame: pd.DataFrame, *, top_k: int = 12) -> str:
    if frame.empty:
        return "_空表。_"
    preview = frame.head(int(top_k)).copy()
    columns = list(preview.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in preview.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import create_pair_contact_sheet
    from animalclef_analysis.initial_audit import create_contact_sheet
    from animalclef_analysis.manual_review_workbench import resolve_predictions_path
    from animalclef_analysis.singleton_rescue_review import (
        SALAMANDER_DATASET,
        build_singleton_rescue_review,
        empty_merge_candidates,
        empty_pair_disagreement,
        load_manual_no_pairs,
    )

    parser = argparse.ArgumentParser(description="Build a manual review pack for Salamander singleton rescue candidates.")
    parser.add_argument("--base-predictions", type=Path, default=DEFAULT_BASE_PREDICTIONS)
    parser.add_argument("--pair-features", type=Path, default=DEFAULT_PAIR_FEATURES)
    parser.add_argument("--pair-judgments", type=Path, default=DEFAULT_PAIR_JUDGMENTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--singleton-singleton-min-prob", type=float, default=0.95)
    parser.add_argument("--attach-member-min-prob", type=float, default=0.90)
    parser.add_argument("--attach-min-support-count", type=int, default=2)
    parser.add_argument("--attach-min-mean-prob", type=float, default=0.90)
    parser.add_argument("--attach-min-max-prob", type=float, default=0.95)
    parser.add_argument("--top-support-pairs", type=int, default=8)
    args = parser.parse_args()

    predictions_path = resolve_predictions_path(repo_root, args.base_predictions)
    pair_features_path = _resolve_path(repo_root, args.pair_features)
    pair_judgments_path = _resolve_path(repo_root, args.pair_judgments)
    output_dir = _resolve_path(repo_root, args.output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    images_dir = output_dir / "images"
    for path in [output_dir, tables_dir, reports_dir, images_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(predictions_path)
    pair_feature_df = pd.read_csv(pair_features_path)
    manual_no_pairs = load_manual_no_pairs(pair_judgments_path, dataset=SALAMANDER_DATASET)
    result = build_singleton_rescue_review(
        pred_df,
        pair_feature_df,
        manual_no_pairs=manual_no_pairs,
        dataset=SALAMANDER_DATASET,
        singleton_singleton_min_prob=float(args.singleton_singleton_min_prob),
        attach_member_min_prob=float(args.attach_member_min_prob),
        attach_min_support_count=int(args.attach_min_support_count),
        attach_min_mean_prob=float(args.attach_min_mean_prob),
        attach_min_max_prob=float(args.attach_min_max_prob),
    )

    merge_candidate_df = result.merge_candidate_df if not result.merge_candidate_df.empty else empty_merge_candidates()
    pair_df = result.pair_df if not result.pair_df.empty else empty_pair_disagreement()
    merge_candidate_df.to_csv(tables_dir / "test_merge_candidates_v1.csv", index=False)
    pair_df.to_csv(tables_dir / "test_pair_disagreement_v1.csv", index=False)
    pd.DataFrame(
        columns=[
            "base_cluster_id",
            "base_cluster_size",
            "ambiguous_image_count",
            "ambiguous_pair_count",
            "max_split_votes",
            "mean_pair_probability",
            "max_pair_probability",
            "mean_ambiguity_score",
            "max_ambiguity_score",
            "mean_border_score",
            "max_conflict_ratio",
            "conflict_methods",
            "component_ids",
            "image_indices",
            "image_ids",
        ]
    ).to_csv(tables_dir / "test_split_candidates_v1.csv", index=False)
    result.stats_df.to_csv(tables_dir / "stats_v1.csv", index=False)
    result.rejected_candidate_df.to_csv(tables_dir / "rejected_candidates_v1.csv", index=False)

    salamander_pred_df = pred_df[pred_df["dataset"].astype(str).eq(SALAMANDER_DATASET)].copy()
    salamander_pred_df["image_id"] = salamander_pred_df["image_id"].astype(str)
    salamander_pred_df["pred_cluster_id"] = pd.to_numeric(
        salamander_pred_df["pred_cluster_id"], errors="coerce"
    ).fillna(-1).astype(int)
    salamander_pred_df["path"] = salamander_pred_df["path"].astype(str)
    image_path_lookup = dict(zip(salamander_pred_df["image_id"], salamander_pred_df["path"], strict=True))

    image_rows: list[dict[str, object]] = []
    markdown_lines = [
        "# Salamander Singleton Rescue Review",
        "",
        f"- 当前底座：`{predictions_path}`",
        f"- 支撑 pair：`{pair_features_path}`",
        f"- 人工 `no`：`{pair_judgments_path}`",
        f"- 数据集：`{SALAMANDER_DATASET}`",
        "",
        "## Rule",
        "",
        f"- `singleton-singleton`: 只看人工产出的 current singleton，要求 mutual top1 且 `xgb_same_identity_prob >= {float(args.singleton_singleton_min_prob):.2f}`，并且不命中已有人工 `no`。",
        f"- `singleton->cluster`: 只看人工产出的 current singleton，要求目标簇内至少 `{int(args.attach_min_support_count)}` 个成员分别满足 `xgb >= {float(args.attach_member_min_prob):.2f}`，且整体 `mean_xgb >= {float(args.attach_min_mean_prob):.2f}`、`max_xgb >= {float(args.attach_min_max_prob):.2f}`，并且对目标簇任意成员都不命中人工 `no`。",
        "",
        "## Stats",
        "",
        _table_markdown(result.stats_df, top_k=len(result.stats_df)),
        "",
        "## Workbench",
        "",
        "```bash",
        "cd /home/hechen/gyk/animalclef",
        "python scripts/launch_manual_review_workbench.py \\",
        f"  --base-predictions {predictions_path} \\",
        f"  --probe-dir {output_dir}",
        "```",
        "",
        "进入后切到 `merge` 任务即可，这一包没有新的 `split` 候选。",
        "",
        "## Accepted Candidates",
        "",
    ]

    if merge_candidate_df.empty:
        markdown_lines.append("_当前没有符合 strict singleton rescue 规则的 merge 候选。_")
    else:
        for row in merge_candidate_df.itertuples(index=False):
            cluster_pair_key = str(row.cluster_pair_key)
            left_cluster_id = int(row.left_cluster_id)
            right_cluster_id = int(row.right_cluster_id)
            candidate_kind = str(getattr(row, "candidate_kind", "merge"))
            cluster_members = salamander_pred_df[
                salamander_pred_df["pred_cluster_id"].isin([left_cluster_id, right_cluster_id])
            ].copy().sort_values(["pred_cluster_id", "image_id"]).reset_index(drop=True)
            cluster_members["review_cluster"] = cluster_members["pred_cluster_id"].apply(
                lambda value: f"cluster={int(value)}"
            )
            cluster_members["review_image"] = cluster_members["image_id"].apply(lambda value: f"image={value}")
            cluster_board_path = images_dir / f"merge_{left_cluster_id}_{right_cluster_id}_clusters.jpg"
            create_contact_sheet(
                df=cluster_members,
                repo_root=repo_root,
                output_path=cluster_board_path,
                title=f"Singleton Rescue | {cluster_pair_key}",
                caption_columns=["review_cluster", "review_image"],
                columns=4,
            )

            support_pairs = pair_df[pair_df["cluster_pair_key"].astype(str).eq(cluster_pair_key)].copy()
            support_pairs["query_path"] = support_pairs["image_id"].map(image_path_lookup)
            support_pairs["neighbor_path"] = support_pairs["neighbor_image_id"].map(image_path_lookup)
            support_pairs["identity"] = support_pairs["base_cluster_left"].astype(int).astype(str)
            support_pairs["neighbor_identity"] = support_pairs["base_cluster_right"].astype(int).astype(str)
            support_pairs["similarity"] = support_pairs["xgb_same_identity_prob"].astype(float)
            pair_board_path = images_dir / f"merge_{left_cluster_id}_{right_cluster_id}_pairs.jpg"
            if not support_pairs.empty:
                create_pair_contact_sheet(
                    rows_df=support_pairs.head(int(args.top_support_pairs)),
                    repo_root=repo_root,
                    output_path=pair_board_path,
                    title=f"Singleton Rescue Support | {cluster_pair_key}",
                    left_path_column="query_path",
                    right_path_column="neighbor_path",
                    caption_left="left",
                    caption_right="right",
                    columns=2,
                )

            image_rows.extend(
                [
                    {
                        "cluster_pair_key": cluster_pair_key,
                        "pred_cluster_id": int(member.pred_cluster_id),
                        "image_id": str(member.image_id),
                        "path": str(member.path),
                        "candidate_kind": candidate_kind,
                    }
                    for member in cluster_members.itertuples(index=False)
                ]
            )

            support_preview_df = support_pairs.loc[
                :,
                [
                    "image_id",
                    "neighbor_image_id",
                    "xgb_same_identity_prob",
                    "local_score",
                    "route_global_score",
                    "merge_votes",
                ],
            ]
            markdown_lines.extend(
                [
                    f"### `{cluster_pair_key}` | `{candidate_kind}`",
                    "",
                    f"- 预览：`{getattr(row, 'candidate_preview', '')}`",
                    f"- left/right cluster size: `{left_cluster_id}:{int(row.left_cluster_size)}` / `{right_cluster_id}:{int(row.right_cluster_size)}`",
                    f"- support pair count: `{int(row.support_pair_count)}`",
                    f"- mean/max xgb: `{float(row.mean_pair_probability):.6f}` / `{float(row.max_pair_probability):.6f}`",
                    f"- mean local: `{float(row.mean_border_score):.6f}`",
                    f"- original split cluster hint: `{getattr(row, 'origin_cluster_id', '')}`",
                    "",
                    f"![cluster-board]({_path_ref(output_dir, cluster_board_path)})",
                    "",
                ]
            )
            if pair_board_path.exists():
                markdown_lines.extend(
                    [
                        f"![pair-board]({_path_ref(output_dir, pair_board_path)})",
                        "",
                    ]
                )
            markdown_lines.extend(
                [
                    _table_markdown(support_preview_df, top_k=min(int(args.top_support_pairs), len(support_preview_df))),
                    "",
                ]
            )

    markdown_lines.extend(
        [
            "## Rejected Near Misses",
            "",
            _table_markdown(result.rejected_candidate_df, top_k=20),
            "",
        ]
    )

    image_member_df = pd.DataFrame(image_rows)
    image_member_df.to_csv(tables_dir / "candidate_members_v1.csv", index=False)
    (reports_dir / "summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    summary_json = {
        "predictions_path": str(predictions_path),
        "pair_features_path": str(pair_features_path),
        "pair_judgments_path": str(pair_judgments_path),
        "output_dir": str(output_dir),
        "accepted_candidate_count": int(len(merge_candidate_df)),
        "support_pair_row_count": int(len(pair_df)),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[singleton_rescue_review] output_dir: {output_dir}")
    print(f"[singleton_rescue_review] summary: {reports_dir / 'summary.md'}")
    print(f"[singleton_rescue_review] merge_candidates: {tables_dir / 'test_merge_candidates_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
