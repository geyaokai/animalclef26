#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_PROBE_DIR = Path("artifacts/analysis/salamander_ambiguity_map_probe_official_aligned_v1")
DEFAULT_OUTPUT_DIRNAME = "review_pack_v1"
DEFAULT_MERGE_KEYS = ["258|273", "9|18", "285|286", "12|13"]
DEFAULT_SPLIT_CLUSTER_IDS = [52, 223, 72, 138, 73, 27]

DEFAULT_MERGE_NOTES = {
    "258|273": ("A", "三方法一致，合并后总大小 9，是当前最像可试的小 merge。"),
    "9|18": ("B", "三方法一致，但属于更大的组件 48，风险高于 258|273。"),
    "285|286": ("C", "历史上已有 official 证伪记录，这里只做复核，不建议直接再提。"),
    "12|13": ("C", "票数高，但合并后总大小 19，明显过大，只适合做参考。"),
}

DEFAULT_SPLIT_NOTES = {
    52: ("A", "两图二分，最干净，最适合人工一眼判断。"),
    223: ("A", "两图二分，结构简单，适合 first shot 观察。"),
    72: ("B", "三图小簇，像是 2+1 或直接三分，需要人工判断桥图。"),
    138: ("B", "三图小簇，疑似某一张在桥接两个局部块。"),
    73: ("B", "三图簇里只有一条强 split 边，需要判断谁是异物。"),
    27: ("B", "两图二分，但优先级低于 52 和 223。"),
}


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_type: str
    candidate_key: str
    tier: str
    review_note: str


def _path_ref(base: Path, target: Path) -> str:
    return str(target.relative_to(base)).replace("\\", "/")


def _table_or_note(frame: pd.DataFrame, *, note: str, top_k: int = 8) -> str:
    if frame.empty:
        return note
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.head(top_k).iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _build_merge_candidates(keys: list[str]) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for key in keys:
        tier, review_note = DEFAULT_MERGE_NOTES.get(key, ("B", "人工补充说明"))
        candidates.append(
            ReviewCandidate(
                candidate_type="merge",
                candidate_key=str(key),
                tier=str(tier),
                review_note=str(review_note),
            )
        )
    return candidates


def _build_split_candidates(cluster_ids: list[int]) -> list[ReviewCandidate]:
    candidates: list[ReviewCandidate] = []
    for cluster_id in cluster_ids:
        tier, review_note = DEFAULT_SPLIT_NOTES.get(int(cluster_id), ("B", "人工补充说明"))
        candidates.append(
            ReviewCandidate(
                candidate_type="split",
                candidate_key=str(int(cluster_id)),
                tier=str(tier),
                review_note=str(review_note),
            )
        )
    return candidates


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import create_pair_contact_sheet
    from animalclef_analysis.initial_audit import create_contact_sheet

    parser = argparse.ArgumentParser(description="Build a manual review checklist and contact sheets for Salamander ambiguity candidates.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--merge-keys", nargs="+", default=DEFAULT_MERGE_KEYS)
    parser.add_argument("--split-cluster-ids", nargs="+", type=int, default=DEFAULT_SPLIT_CLUSTER_IDS)
    parser.add_argument("--top-support-pairs", type=int, default=8)
    args = parser.parse_args()

    probe_dir = (repo_root / args.probe_dir).resolve() if not args.probe_dir.is_absolute() else args.probe_dir.resolve()
    output_dir = (
        probe_dir / DEFAULT_OUTPUT_DIRNAME
        if args.output_dir is None
        else ((repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve())
    )
    image_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    for path in [output_dir, image_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.read_csv(probe_dir / "tables" / "test_best_predictions_v1.csv")
    prediction_df = prediction_df[prediction_df["method"].astype(str).eq("average_linkage")].copy().reset_index(drop=True)
    prediction_df["image_id"] = prediction_df["image_id"].astype(str)
    prediction_df["pred_cluster_id"] = prediction_df["pred_cluster_id"].astype(int)
    prediction_df["path"] = prediction_df["path"].astype(str)

    pair_df = pd.read_csv(probe_dir / "tables" / "test_pair_disagreement_v1.csv")
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    merge_candidate_df = pd.read_csv(probe_dir / "tables" / "test_merge_candidates_v1.csv")
    split_candidate_df = pd.read_csv(probe_dir / "tables" / "test_split_candidates_v1.csv")
    component_df = pd.read_csv(probe_dir / "tables" / "test_ambiguity_components_v1.csv")

    image_path_lookup = dict(zip(prediction_df["image_id"], prediction_df["path"], strict=True))

    merge_reviews = _build_merge_candidates([str(value) for value in args.merge_keys])
    split_reviews = _build_split_candidates([int(value) for value in args.split_cluster_ids])

    merge_review_rows: list[dict[str, object]] = []
    merge_member_rows: list[dict[str, object]] = []
    split_review_rows: list[dict[str, object]] = []
    split_member_rows: list[dict[str, object]] = []
    markdown_lines = [
        "# Salamander 歧义人工审图清单",
        "",
        f"- 基线分区：`average-linkage @ 0.25`，来自 `{probe_dir}`。",
        "- 这份清单只用于人工复核，不等价于已经应该提交 official。",
        "- `A` 档表示优先人工看；`B` 档表示值得看但风险更高；`C` 档表示主要做复核或反证。",
        "",
        "## Merge 候选",
        "",
    ]

    for review in merge_reviews:
        merge_row = merge_candidate_df[merge_candidate_df["cluster_pair_key"].astype(str).eq(review.candidate_key)]
        if merge_row.empty:
            continue
        row = merge_row.iloc[0]
        left_cluster_id, right_cluster_id = [int(value) for value in str(review.candidate_key).split("|")]
        cluster_members = prediction_df[
            prediction_df["pred_cluster_id"].isin([left_cluster_id, right_cluster_id])
        ].copy().sort_values(["pred_cluster_id", "image_id"]).reset_index(drop=True)
        cluster_members["review_cluster"] = cluster_members["pred_cluster_id"].apply(lambda value: f"cluster={int(value)}")
        cluster_members["review_image"] = cluster_members["image_id"].apply(lambda value: f"image={value}")
        cluster_board_path = image_dir / f"merge_{left_cluster_id}_{right_cluster_id}_clusters.jpg"
        create_contact_sheet(
            df=cluster_members.rename(columns={"path": "path"}),
            repo_root=repo_root,
            output_path=cluster_board_path,
            title=f"Merge Review | {left_cluster_id}|{right_cluster_id}",
            caption_columns=["review_cluster", "review_image"],
            columns=4,
        )

        support_pairs = pair_df[
            (
                (pair_df["base_cluster_left"].astype(int).eq(left_cluster_id) & pair_df["base_cluster_right"].astype(int).eq(right_cluster_id))
                | (pair_df["base_cluster_left"].astype(int).eq(right_cluster_id) & pair_df["base_cluster_right"].astype(int).eq(left_cluster_id))
            )
            & pair_df["vote_direction"].astype(str).eq("merge")
        ].copy()
        support_pairs["query_path"] = support_pairs["image_id"].map(image_path_lookup)
        support_pairs["neighbor_path"] = support_pairs["neighbor_image_id"].map(image_path_lookup)
        support_pairs["identity"] = support_pairs["base_cluster_left"].astype(int).astype(str)
        support_pairs["neighbor_identity"] = support_pairs["base_cluster_right"].astype(int).astype(str)
        support_pairs["similarity"] = support_pairs["xgb_same_identity_prob"].astype(float)
        support_pairs = support_pairs.sort_values(
            ["merge_votes", "ambiguity_score", "xgb_same_identity_prob"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        pair_board_path = image_dir / f"merge_{left_cluster_id}_{right_cluster_id}_pairs.jpg"
        if not support_pairs.empty:
            create_pair_contact_sheet(
                rows_df=support_pairs.head(int(args.top_support_pairs)),
                repo_root=repo_root,
                output_path=pair_board_path,
                title=f"Merge Support Pairs | {left_cluster_id}|{right_cluster_id}",
                left_path_column="query_path",
                right_path_column="neighbor_path",
                caption_left="left",
                caption_right="right",
                columns=2,
            )

        for member in cluster_members.itertuples(index=False):
            merge_member_rows.append(
                {
                    "cluster_pair_key": review.candidate_key,
                    "pred_cluster_id": int(member.pred_cluster_id),
                    "image_id": str(member.image_id),
                    "path": str(member.path),
                }
            )
        merge_review_rows.append(
            {
                "cluster_pair_key": review.candidate_key,
                "tier": review.tier,
                "review_note": review.review_note,
                "left_cluster_id": left_cluster_id,
                "right_cluster_id": right_cluster_id,
                "left_cluster_size": int(row["left_cluster_size"]),
                "right_cluster_size": int(row["right_cluster_size"]),
                "merged_total_size": int(row["merged_total_size"]),
                "support_pair_count": int(row["support_pair_count"]),
                "max_merge_votes": int(row["max_merge_votes"]),
                "mean_pair_probability": float(row["mean_pair_probability"]),
                "max_pair_probability": float(row["max_pair_probability"]),
                "mean_ambiguity_score": float(row["mean_ambiguity_score"]),
                "max_ambiguity_score": float(row["max_ambiguity_score"]),
                "conflict_methods": str(row["conflict_methods"]),
                "component_ids": str(row["component_ids"]),
            }
        )

        markdown_lines.extend(
            [
                f"### `{review.tier}` 档 Merge `{review.candidate_key}`",
                "",
                f"- 建议操作：把 base cluster `{left_cluster_id}` 与 `{right_cluster_id}` 合并。",
                f"- 人工备注：{review.review_note}",
                f"- 指标：合并后总大小 `{int(row['merged_total_size'])}`，支撑边 `{int(row['support_pair_count'])}`，最高投票 `{int(row['max_merge_votes'])}`，最高歧义分 `{float(row['max_ambiguity_score']):.6f}`。",
                f"- 冲突方法：`{str(row['conflict_methods'])}`；组件：`{str(row['component_ids'])}`。",
                "",
                f"![merge-cluster-board]({_path_ref(output_dir, cluster_board_path)})",
                "",
            ]
        )
        if pair_board_path.exists():
            markdown_lines.extend(
                [
                    f"![merge-pair-board]({_path_ref(output_dir, pair_board_path)})",
                    "",
                ]
            )
        support_view = support_pairs.loc[
            :,
            ["image_id", "neighbor_image_id", "xgb_same_identity_prob", "merge_votes", "ambiguity_score", "conflict_methods"],
        ]
        markdown_lines.extend(
            [
                _table_or_note(
                    support_view,
                    note="_没有可展示的支撑边。_",
                    top_k=min(int(args.top_support_pairs), len(support_view)),
                ),
                "",
            ]
        )

    markdown_lines.extend(
        [
            "## Split 候选",
            "",
        ]
    )

    for review in split_reviews:
        cluster_id = int(review.candidate_key)
        split_row = split_candidate_df[split_candidate_df["base_cluster_id"].astype(int).eq(cluster_id)]
        if split_row.empty:
            continue
        row = split_row.iloc[0]
        cluster_members = prediction_df[prediction_df["pred_cluster_id"].astype(int).eq(cluster_id)].copy()
        cluster_members = cluster_members.sort_values("image_id").reset_index(drop=True)
        ambiguous_ids = set(str(value) for value in str(row["image_ids"]).split("|") if value)
        cluster_members["review_flag"] = cluster_members["image_id"].apply(
            lambda value: "ambiguous" if str(value) in ambiguous_ids else "context"
        )
        cluster_members["review_image"] = cluster_members["image_id"].apply(lambda value: f"image={value}")
        cluster_board_path = image_dir / f"split_{cluster_id}_cluster.jpg"
        create_contact_sheet(
            df=cluster_members,
            repo_root=repo_root,
            output_path=cluster_board_path,
            title=f"Split Review | cluster={cluster_id}",
            caption_columns=["review_flag", "review_image"],
            columns=4,
        )

        support_pairs = pair_df[
            pair_df["base_cluster_left"].astype(int).eq(cluster_id)
            & pair_df["base_cluster_right"].astype(int).eq(cluster_id)
            & pair_df["vote_direction"].astype(str).eq("split")
        ].copy()
        support_pairs["query_path"] = support_pairs["image_id"].map(image_path_lookup)
        support_pairs["neighbor_path"] = support_pairs["neighbor_image_id"].map(image_path_lookup)
        support_pairs["identity"] = [f"cluster={cluster_id}"] * len(support_pairs)
        support_pairs["neighbor_identity"] = [f"cluster={cluster_id}"] * len(support_pairs)
        support_pairs["similarity"] = support_pairs["xgb_same_identity_prob"].astype(float)
        support_pairs = support_pairs.sort_values(
            ["split_votes", "ambiguity_score", "xgb_same_identity_prob"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        pair_board_path = image_dir / f"split_{cluster_id}_pairs.jpg"
        if not support_pairs.empty:
            create_pair_contact_sheet(
                rows_df=support_pairs.head(int(args.top_support_pairs)),
                repo_root=repo_root,
                output_path=pair_board_path,
                title=f"Split Support Pairs | cluster={cluster_id}",
                left_path_column="query_path",
                right_path_column="neighbor_path",
                caption_left="left",
                caption_right="right",
                columns=2,
            )

        for member in cluster_members.itertuples(index=False):
            split_member_rows.append(
                {
                    "base_cluster_id": cluster_id,
                    "image_id": str(member.image_id),
                    "path": str(member.path),
                    "review_flag": str(member.review_flag),
                }
            )
        split_review_rows.append(
            {
                "base_cluster_id": cluster_id,
                "tier": review.tier,
                "review_note": review.review_note,
                "base_cluster_size": int(row["base_cluster_size"]),
                "ambiguous_image_count": int(row["ambiguous_image_count"]),
                "ambiguous_pair_count": int(row["ambiguous_pair_count"]),
                "max_split_votes": int(row["max_split_votes"]),
                "mean_pair_probability": float(row["mean_pair_probability"]),
                "max_pair_probability": float(row["max_pair_probability"]),
                "mean_ambiguity_score": float(row["mean_ambiguity_score"]),
                "max_ambiguity_score": float(row["max_ambiguity_score"]),
                "conflict_methods": str(row["conflict_methods"]),
                "component_ids": str(row["component_ids"]),
                "image_ids": str(row["image_ids"]),
            }
        )

        markdown_lines.extend(
            [
                f"### `{review.tier}` 档 Split `cluster {cluster_id}`",
                "",
                f"- 建议操作：把当前 base cluster `{cluster_id}` 当成可疑混簇，重点判断是否需要拆开。",
                f"- 人工备注：{review.review_note}",
                f"- 指标：簇大小 `{int(row['base_cluster_size'])}`，歧义图像 `{int(row['ambiguous_image_count'])}`，歧义边 `{int(row['ambiguous_pair_count'])}`，最高 split 投票 `{int(row['max_split_votes'])}`，最高歧义分 `{float(row['max_ambiguity_score']):.6f}`。",
                f"- 冲突方法：`{str(row['conflict_methods'])}`；组件：`{str(row['component_ids'])}`。",
                "",
                f"![split-cluster-board]({_path_ref(output_dir, cluster_board_path)})",
                "",
            ]
        )
        if pair_board_path.exists():
            markdown_lines.extend(
                [
                    f"![split-pair-board]({_path_ref(output_dir, pair_board_path)})",
                    "",
                ]
            )
        support_view = support_pairs.loc[
            :,
            ["image_id", "neighbor_image_id", "xgb_same_identity_prob", "split_votes", "ambiguity_score", "conflict_methods"],
        ]
        markdown_lines.extend(
            [
                _table_or_note(
                    support_view,
                    note="_没有可展示的 split 支撑边。_",
                    top_k=min(int(args.top_support_pairs), len(support_view)),
                ),
                "",
            ]
        )

    pd.DataFrame(merge_review_rows).to_csv(tables_dir / "merge_review_candidates_v1.csv", index=False)
    pd.DataFrame(split_review_rows).to_csv(tables_dir / "split_review_candidates_v1.csv", index=False)
    pd.DataFrame(merge_member_rows).to_csv(tables_dir / "merge_review_members_v1.csv", index=False)
    pd.DataFrame(split_member_rows).to_csv(tables_dir / "split_review_members_v1.csv", index=False)
    component_df.head(20).to_csv(tables_dir / "component_top20_snapshot_v1.csv", index=False)

    (output_dir / "summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"[salamander_ambiguity_review_pack] summary: {output_dir / 'summary.md'}")
    print(f"[salamander_ambiguity_review_pack] images: {image_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
