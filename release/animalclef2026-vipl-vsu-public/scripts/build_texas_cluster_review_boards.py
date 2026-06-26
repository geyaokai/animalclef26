#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import itertools
import json
import os
from pathlib import Path

import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build HTML review boards for Texas constrained-merge and suspect-cluster inspection.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--metadata-path", type=Path, default=repo_root / "metadata.csv")
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=repo_root / "artifacts/submissions/kaggle_variant_texas_seeded_attach_on_059341_base_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v2/texas_pair_registry_v2.csv",
    )
    parser.add_argument(
        "--edge-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_constrained_seed_graph_probe_v1/tables/cluster_edge_candidates_v1.csv",
    )
    parser.add_argument(
        "--best-edge-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_constrained_seed_graph_probe_v1/tables/best_selected_edges_v1.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_cluster_review_boards_v1",
    )
    return parser.parse_args()


def to_rel(from_dir: Path, target: Path) -> str:
    return os.path.relpath(str(target.resolve()), str(from_dir.resolve()))


def split_members(raw_value: object) -> list[str]:
    text = str(raw_value or "").strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def build_pair_rows(
    image_ids: list[str],
    registry_df: pd.DataFrame,
    extra_score_lookup: dict[tuple[str, str], dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    pair_rows: list[dict[str, object]] = []
    extra_score_lookup = extra_score_lookup or {}
    for left, right in itertools.combinations(sorted(set(image_ids)), 2):
        pair_key = (left, right) if left <= right else (right, left)
        pair_df = registry_df[
            (
                (registry_df["image_id_a"].eq(left) & registry_df["image_id_b"].eq(right))
                | (registry_df["image_id_a"].eq(right) & registry_df["image_id_b"].eq(left))
            )
        ].copy()
        if pair_df.empty:
            score_meta = extra_score_lookup.get(pair_key, {})
            pair_rows.append(
                {
                    "image_id_a": left,
                    "image_id_b": right,
                    "constraint_type": "unlabeled",
                    "support_count": "",
                    "sources": "",
                    "notes": "",
                    "score_meta": score_meta,
                }
            )
            continue
        for row in pair_df.sort_values(["constraint_type", "support_count"], ascending=[True, False]).itertuples(index=False):
            score_meta = extra_score_lookup.get(pair_key, {})
            pair_rows.append(
                {
                    "image_id_a": left,
                    "image_id_b": right,
                    "constraint_type": str(row.constraint_type),
                    "support_count": str(row.support_count),
                    "sources": str(row.sources),
                    "notes": str(row.notes),
                    "score_meta": score_meta,
                }
            )
    return pair_rows


def render_image_panel(
    output_dir: Path,
    repo_root: Path,
    image_id: str,
    meta_row: pd.Series,
    pred_row: pd.Series,
) -> str:
    original_path = (repo_root / str(meta_row["path"])).resolve()
    aligned_path = (repo_root / str(pred_row["path"])).resolve()
    center_gray = pred_row.get("texas_center_body_gray_path_v1")
    center_square = pred_row.get("texas_center_body_square_path_v1")
    gray_path = (repo_root / str(center_gray)).resolve() if pd.notna(center_gray) and str(center_gray).strip() else None
    square_path = (repo_root / str(center_square)).resolve() if pd.notna(center_square) and str(center_square).strip() else None
    gray_low = pred_row.get("texas_center_body_gray_low_value_v1")
    gray_high = pred_row.get("texas_center_body_gray_high_value_v1")
    scale_factor = pred_row.get("texas_center_body_scale_factor_v1")
    sam_reason = pred_row.get("texas_center_body_sam_reason_v1")

    extra_blocks: list[str] = []
    if square_path is not None and square_path.exists():
        extra_blocks.append(
            f"""
            <div class="view-block">
              <div class="view-title">Center Body Square</div>
              <a href="{html.escape(to_rel(output_dir, square_path))}" target="_blank">
                <img src="{html.escape(to_rel(output_dir, square_path))}" alt="center square {html.escape(image_id)}" />
              </a>
              <div class="path-note">{html.escape(str(center_square))}</div>
            </div>
            """
        )
    if gray_path is not None and gray_path.exists():
        extra_blocks.append(
            f"""
            <div class="view-block">
              <div class="view-title">Center Body Gray</div>
              <a href="{html.escape(to_rel(output_dir, gray_path))}" target="_blank">
                <img src="{html.escape(to_rel(output_dir, gray_path))}" alt="center gray {html.escape(image_id)}" />
              </a>
              <div class="path-note">{html.escape(str(center_gray))}</div>
            </div>
            """
        )

    stats = [
        f"cluster {int(pred_row['pred_cluster_id'])}",
        f"is_seed={bool(pred_row.get('is_seed', False))}",
        f"pseudo_label={int(pred_row.get('pseudo_label_index', -1))}" if pd.notna(pred_row.get("pseudo_label_index")) else "pseudo_label=-",
        f"comp_size={int(pred_row.get('component_size', 0))}" if pd.notna(pred_row.get("component_size")) else "comp_size=-",
        f"scale={float(scale_factor):.3f}" if pd.notna(scale_factor) else "scale=-",
        f"gray=({float(gray_low):.1f}, {float(gray_high):.1f})" if pd.notna(gray_low) and pd.notna(gray_high) else "gray=-",
        f"sam={sam_reason}" if pd.notna(sam_reason) and str(sam_reason).strip() else "sam=-",
    ]

    return f"""
    <div class="image-card">
      <div class="image-header">
        <h3>{html.escape(image_id)}</h3>
        <div class="meta-chip">{html.escape(' | '.join(stats))}</div>
      </div>
      <div class="image-grid">
        <div class="view-block">
          <div class="view-title">Original</div>
          <a href="{html.escape(to_rel(output_dir, original_path))}" target="_blank">
            <img src="{html.escape(to_rel(output_dir, original_path))}" alt="original {html.escape(image_id)}" />
          </a>
          <div class="path-note">{html.escape(str(meta_row['path']))}</div>
        </div>
        <div class="view-block">
          <div class="view-title">Aligned / Model View</div>
          <a href="{html.escape(to_rel(output_dir, aligned_path))}" target="_blank">
            <img src="{html.escape(to_rel(output_dir, aligned_path))}" alt="aligned {html.escape(image_id)}" />
          </a>
          <div class="path-note">{html.escape(str(pred_row['path']))}</div>
        </div>
        {''.join(extra_blocks)}
      </div>
    </div>
    """


def render_pair_cards(
    pair_rows: list[dict[str, object]],
    output_dir: Path,
    repo_root: Path,
    meta_by_id: dict[str, pd.Series],
    pred_by_id: dict[str, pd.Series],
) -> str:
    cards: list[str] = []
    for row in pair_rows:
        left = str(row["image_id_a"])
        right = str(row["image_id_b"])
        left_img = (repo_root / str(pred_by_id[left]["path"])).resolve()
        right_img = (repo_root / str(pred_by_id[right]["path"])).resolve()
        score_meta = row.get("score_meta") or {}
        score_line = ""
        if score_meta:
            score_bits = []
            if "mean_top_score" in score_meta:
                score_bits.append(f"mean_top={float(score_meta['mean_top_score']):.6f}")
            if "max_score" in score_meta:
                score_bits.append(f"max={float(score_meta['max_score']):.6f}")
            if "support_count" in score_meta:
                score_bits.append(f"support={score_meta['support_count']}")
            score_line = f"<div><strong>Score</strong>: {html.escape(' | '.join(score_bits))}</div>"
        cards.append(
            f"""
            <div class="pair-card pair-{html.escape(str(row['constraint_type']))}">
              <div class="pair-title">
                <h3>{html.escape(left)} vs {html.escape(right)}</h3>
                <span class="badge">{html.escape(str(row['constraint_type']))}</span>
                <span class="badge muted">support {html.escape(str(row['support_count']) or '-')}</span>
              </div>
              <div class="pair-grid">
                <div class="pair-image">
                  <a href="{html.escape(to_rel(output_dir, left_img))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, left_img))}" alt="{html.escape(left)}" />
                  </a>
                  <div class="caption">{html.escape(left)} · cluster {int(pred_by_id[left]['pred_cluster_id'])}</div>
                </div>
                <div class="pair-image">
                  <a href="{html.escape(to_rel(output_dir, right_img))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, right_img))}" alt="{html.escape(right)}" />
                  </a>
                  <div class="caption">{html.escape(right)} · cluster {int(pred_by_id[right]['pred_cluster_id'])}</div>
                </div>
              </div>
              <div class="pair-notes">
                {score_line}
                <div><strong>Source</strong>: {html.escape(str(row['sources'] or '-'))}</div>
                <div><strong>Registry Note</strong>: {html.escape(str(row['notes'] or '-'))}</div>
                <div><strong>Original Paths</strong>: {html.escape(str(meta_by_id[left]['path']))} | {html.escape(str(meta_by_id[right]['path']))}</div>
              </div>
            </div>
            """
        )
    return "".join(cards)


def build_html_page(
    *,
    title: str,
    subtitle_chips: list[str],
    intro_lines: list[str],
    image_ids: list[str],
    pair_rows: list[dict[str, object]],
    output_dir: Path,
    repo_root: Path,
    meta_by_id: dict[str, pd.Series],
    pred_by_id: dict[str, pd.Series],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_cards = [
        render_image_panel(output_dir=output_dir, repo_root=repo_root, image_id=image_id, meta_row=meta_by_id[image_id], pred_row=pred_by_id[image_id])
        for image_id in image_ids
    ]
    table_rows = "".join(
        f"<tr><td>{html.escape(str(row['image_id_a']))} vs {html.escape(str(row['image_id_b']))}</td>"
        f"<td>{html.escape(str(row['constraint_type']))}</td>"
        f"<td>{html.escape(str(row['support_count']) or '-')}</td>"
        f"<td>{html.escape(str((row.get('score_meta') or {}).get('mean_top_score', '-')))}</td>"
        f"<td>{html.escape(str(row['sources'] or '-'))}</td></tr>"
        for row in pair_rows
    )
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #101418; color: #ecf0f1; }}
    main {{ max-width: 1680px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3, p {{ margin: 0; }}
    .hero, .section-card {{ background: #171d23; border: 1px solid #2a3642; border-radius: 14px; padding: 20px; margin-bottom: 24px; }}
    .meta-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .chip, .badge, .meta-chip {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: #243140; color: #d9e6f2; font-size: 13px; }}
    .badge {{ background: #334155; }}
    .badge.muted {{ background: #3b4652; color: #d0d7de; }}
    .intro {{ margin-top: 12px; color: #c7d2db; line-height: 1.6; }}
    .section-title {{ margin-bottom: 14px; }}
    .image-card, .pair-card {{ background: #171d23; border: 1px solid #2a3642; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    .image-header, .pair-title {{ display: flex; align-items: center; gap: 10px; justify-content: space-between; margin-bottom: 14px; flex-wrap: wrap; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .view-block, .pair-image {{ background: #0f1419; border-radius: 12px; padding: 12px; }}
    .view-title {{ font-size: 13px; color: #9fb3c8; margin-bottom: 8px; }}
    img {{ width: 100%; height: auto; max-height: 540px; object-fit: contain; background: #000; border-radius: 8px; }}
    .path-note, .caption, .pair-notes {{ font-size: 12px; color: #c5d1dd; margin-top: 8px; line-height: 1.5; word-break: break-all; }}
    .pair-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 10px; }}
    .pair-card.pair-must-link {{ border-color: #2e8b57; }}
    .pair-card.pair-cannot-link {{ border-color: #b94a48; }}
    .pair-card.pair-unlabeled {{ border-color: #5b7083; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #2a3642; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #243140; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <div class="meta-row">
        {''.join(f'<span class="chip">{html.escape(chip)}</span>' for chip in subtitle_chips)}
      </div>
      <div class="intro">
        {''.join(f'<p>{html.escape(line)}</p>' for line in intro_lines)}
      </div>
      <table>
        <thead>
          <tr><th>Pair</th><th>Constraint</th><th>Support</th><th>mean_top_score</th><th>Source</th></tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </section>

    <section class="section-card">
      <h2 class="section-title">Image Overview</h2>
      {''.join(image_cards)}
    </section>

    <section class="section-card">
      <h2 class="section-title">Pairwise Review</h2>
      {render_pair_cards(pair_rows=pair_rows, output_dir=output_dir, repo_root=repo_root, meta_by_id=meta_by_id, pred_by_id=pred_by_id)}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(args.metadata_path.resolve()).copy()
    metadata_df["image_id"] = metadata_df["image_id"].astype(str)
    metadata_df = metadata_df[metadata_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)

    pred_df = pd.read_csv(args.prediction_path.resolve()).copy()
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df = pred_df[pred_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)

    registry_df = pd.read_csv(args.registry_path.resolve()).copy()
    registry_df["image_id_a"] = registry_df["image_id_a"].astype(str)
    registry_df["image_id_b"] = registry_df["image_id_b"].astype(str)

    edge_df = pd.read_csv(args.edge_path.resolve()).copy()
    best_edge_df = pd.read_csv(args.best_edge_path.resolve()).copy()

    meta_by_id = {str(row["image_id"]): row for _, row in metadata_df.iterrows()}
    pred_by_id = {str(row["image_id"]): row for _, row in pred_df.iterrows()}

    best_edge = best_edge_df.iloc[0]
    best_edge_ids = sorted(set(split_members(best_edge["left_members"]) + split_members(best_edge["right_members"])))
    merge_pair_scores = {
        tuple(sorted([split_members(row["left_members"])[0], split_members(row["right_members"])[0]])): {
            "mean_top_score": float(row["mean_top_score"]),
            "max_score": float(row["max_score"]),
            "support_count": int(row["support_count"]),
        }
        for _, row in best_edge_df.iterrows()
        if len(split_members(row["left_members"])) == 1 and len(split_members(row["right_members"])) == 1
    }

    build_html_page(
        title="Texas Constrained Merge Review",
        subtitle_chips=[
            "target edge: 15416 <-> 15311",
            "source: constrained seed graph probe",
            f"mean_top={float(best_edge['mean_top_score']):.6f}",
            f"support={int(best_edge['support_count'])}",
        ],
        intro_lines=[
            "这页只看当前最值得提官方的 constrained merge 单边候选。",
            "重点判断 15416 和 15311 是否真应并到同一个 id。",
        ],
        image_ids=best_edge_ids,
        pair_rows=build_pair_rows(best_edge_ids, registry_df, extra_score_lookup=merge_pair_scores),
        output_dir=output_root / "merge_review",
        repo_root=repo_root,
        meta_by_id=meta_by_id,
        pred_by_id=pred_by_id,
    )

    neighbor_image_ids = ["15416", "15311", "15270", "15386", "15244", "15309", "15213", "15419"]
    neighbor_image_ids = [image_id for image_id in neighbor_image_ids if image_id in pred_by_id]
    neighbor_pair_scores: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in edge_df.iterrows():
        left_members = split_members(row["left_members"])
        right_members = split_members(row["right_members"])
        if len(left_members) != 1:
            continue
        if len(right_members) > 2:
            continue
        left = left_members[0]
        for right in right_members:
            pair_key = tuple(sorted([left, right]))
            if left in neighbor_image_ids and right in neighbor_image_ids:
                neighbor_pair_scores[pair_key] = {
                    "mean_top_score": float(row["mean_top_score"]),
                    "max_score": float(row["max_score"]),
                    "support_count": int(row["support_count"]),
                }

    build_html_page(
        title="Texas Merge Neighborhood Review",
        subtitle_chips=[
            "anchors: 15416 / 15311",
            "neighbors: 15270 / 15386 / 15244 / 15309 / 15213 / 15419",
            "goal: 判断是单对 merge 还是更大局部簇",
        ],
        intro_lines=[
            "这页看的是 15416 周围的纠缠邻域，而不只是单对 merge。",
            "如果你认为这里其实还能扩成更大的小簇，优先看 pairwise card 里的无标注高分边。",
        ],
        image_ids=neighbor_image_ids,
        pair_rows=build_pair_rows(neighbor_image_ids, registry_df, extra_score_lookup=neighbor_pair_scores),
        output_dir=output_root / "merge_neighborhood_review",
        repo_root=repo_root,
        meta_by_id=meta_by_id,
        pred_by_id=pred_by_id,
    )

    suspect_clusters = [43, 157, 227, 269]
    suspect_image_ids: list[str] = []
    for cluster_id in suspect_clusters:
        suspect_image_ids.extend(
            sorted(pred_df.loc[pred_df["pred_cluster_id"].eq(cluster_id), "image_id"].astype(str).tolist())
        )
    suspect_image_ids = list(dict.fromkeys(suspect_image_ids))
    suspect_pair_scores: dict[tuple[str, str], dict[str, object]] = {}
    for _, row in edge_df.iterrows():
        left_members = split_members(row["left_members"])
        right_members = split_members(row["right_members"])
        if len(left_members) != 1 or len(right_members) != 1:
            continue
        left = left_members[0]
        right = right_members[0]
        if left in suspect_image_ids and right in suspect_image_ids:
            suspect_pair_scores[tuple(sorted([left, right]))] = {
                "mean_top_score": float(row["mean_top_score"]),
                "max_score": float(row["max_score"]),
                "support_count": int(row["support_count"]),
            }

    build_html_page(
        title="Texas Suspect Cluster Board",
        subtitle_chips=[
            "clusters: 43 / 157 / 227 / 269",
            "purpose: 审核当前看起来可疑但 split proxy 不支持的簇",
            "focus: 有没有假并、尾巴图、或可追加 must-link",
        ],
        intro_lines=[
            "这页不是官方候选提交，而是为了人工判断这 4 个簇后续该不该加规则。",
            "43 / 157 / 227 没有内部 must-link；269 的核心 must-link 只有 15320 <-> 15384，15355 是 attach 尾巴图。",
        ],
        image_ids=suspect_image_ids,
        pair_rows=build_pair_rows(suspect_image_ids, registry_df, extra_score_lookup=suspect_pair_scores),
        output_dir=output_root / "suspect_cluster_board",
        repo_root=repo_root,
        meta_by_id=meta_by_id,
        pred_by_id=pred_by_id,
    )

    summary = {
        "merge_review": str((output_root / "merge_review" / "index.html").resolve()),
        "merge_neighborhood_review": str((output_root / "merge_neighborhood_review" / "index.html").resolve()),
        "suspect_cluster_board": str((output_root / "suspect_cluster_board" / "index.html").resolve()),
        "best_edge": best_edge.to_dict(),
        "neighbor_image_ids": neighbor_image_ids,
        "suspect_image_ids": suspect_image_ids,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[texas_cluster_review_boards] merge_review={output_root / 'merge_review' / 'index.html'}")
    print(f"[texas_cluster_review_boards] merge_neighborhood_review={output_root / 'merge_neighborhood_review' / 'index.html'}")
    print(f"[texas_cluster_review_boards] suspect_cluster_board={output_root / 'suspect_cluster_board' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
