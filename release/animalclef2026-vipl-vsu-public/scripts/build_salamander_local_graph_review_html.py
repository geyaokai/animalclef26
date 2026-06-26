#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import pandas as pd


def _choose_image_path(row: pd.Series) -> str:
    for column in ["source_global_path", "global_path", "trunk_path"]:
        value = str(row.get(column, "") or "")
        if value:
            return value
    return ""


def _to_rel_path(target_path: str, html_dir: Path, repo_root: Path) -> str:
    if not target_path:
        return ""
    absolute = (repo_root / target_path).resolve()
    return os.path.relpath(absolute, html_dir)


def _image_card(
    *,
    row: pd.Series,
    html_dir: Path,
    repo_root: Path,
    caption: str,
    css_class: str,
    extra_lines: list[str] | None = None,
) -> str:
    image_rel = _to_rel_path(_choose_image_path(row), html_dir=html_dir, repo_root=repo_root)
    lines = [f"<div class='card {css_class}'>"]
    if image_rel:
        lines.append(f"<img src='{html.escape(image_rel)}' loading='lazy' />")
    lines.append(f"<div class='caption'>{html.escape(caption)}</div>")
    if extra_lines:
        for line in extra_lines:
            lines.append(f"<div class='meta'>{html.escape(line)}</div>")
    lines.append("</div>")
    return "\n".join(lines)


def build_review_html(
    *,
    repo_root: Path,
    predictions_path: Path,
    pair_scores_path: Path,
    output_dir: Path,
    top_n: int,
    top_neighbors: int,
) -> Path:
    predictions_df = pd.read_csv(predictions_path)
    pair_df = pd.read_csv(pair_scores_path)

    predictions_df["image_id"] = predictions_df["image_id"].astype(str)
    predictions_df["identity"] = predictions_df["identity"].astype(str)
    predictions_df["cluster_label"] = predictions_df["cluster_label"].astype(str)
    row_by_image = predictions_df.set_index("image_id", drop=False)

    cluster_to_images = predictions_df.groupby("cluster_label")["image_id"].agg(list).to_dict()
    identity_to_images = predictions_df.groupby("identity")["image_id"].agg(list).to_dict()

    undirected_rows: list[dict[str, object]] = []
    for row in pair_df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        score = float(getattr(row, "local_only_score_v1", 0.0) or 0.0)
        global_score = float(getattr(row, "route_global_score", 0.0) or 0.0)
        same_identity = int(getattr(row, "same_identity", -1))
        decision = str(getattr(row, "yellow_veto_decision_v1", ""))
        for query, neighbor in [(left, right), (right, left)]:
            undirected_rows.append(
                {
                    "query_image_id": query,
                    "neighbor_image_id": neighbor,
                    "local_only_score_v1": score,
                    "route_global_score": global_score,
                    "same_identity": same_identity,
                    "yellow_veto_decision_v1": decision,
                }
            )
    neighbor_df = pd.DataFrame(undirected_rows)
    neighbor_df["query_image_id"] = neighbor_df["query_image_id"].astype(str)
    neighbor_df["neighbor_image_id"] = neighbor_df["neighbor_image_id"].astype(str)

    case_rows: list[dict[str, object]] = []
    for row in predictions_df.itertuples(index=False):
        image_id = str(row.image_id)
        identity = str(row.identity)
        cluster_label = str(row.cluster_label)
        true_mates = set(identity_to_images.get(identity, [])) - {image_id}
        pred_mates = set(cluster_to_images.get(cluster_label, [])) - {image_id}
        missed = sorted(true_mates - pred_mates)
        false_pos = sorted(pred_mates - true_mates)
        case_rows.append(
            {
                "image_id": image_id,
                "identity": identity,
                "cluster_label": cluster_label,
                "missed_count": len(missed),
                "false_pos_count": len(false_pos),
                "true_mate_count": len(true_mates),
                "pred_cluster_size_excl_self": len(pred_mates),
                "error_score": len(missed) * 10 + len(false_pos),
                "missed_ids": missed,
                "false_pos_ids": false_pos,
            }
        )
    case_df = pd.DataFrame(case_rows).sort_values(
        ["error_score", "missed_count", "false_pos_count", "pred_cluster_size_excl_self", "image_id"],
        ascending=[False, False, False, False, True],
    )
    selected_cases = case_df.head(int(top_n)).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path = output_dir / "index.html"

    sections: list[str] = [
        "<html><head><meta charset='utf-8'/>",
        "<title>Salamander Local Graph Error Review</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;background:#f5f5f5;color:#111;}",
        "h1,h2,h3{margin:8px 0;}",
        ".summary{margin-bottom:24px;padding:16px;background:#fff;border-radius:12px;}",
        ".case{margin:24px 0;padding:16px;background:#fff;border-radius:12px;}",
        ".grid{display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start;}",
        ".card{width:220px;padding:8px;border-radius:10px;background:#fafafa;border:3px solid #ddd;}",
        ".query{border-color:#222;}",
        ".good{border-color:#1b8a3d;}",
        ".missed{border-color:#c0392b;}",
        ".fp{border-color:#d35400;}",
        ".neighbor{border-color:#4a6cf7;}",
        "img{width:100%;height:160px;object-fit:contain;background:#000;border-radius:6px;}",
        ".caption{font-weight:700;margin-top:6px;font-size:13px;word-break:break-all;}",
        ".meta{font-size:12px;color:#444;margin-top:2px;word-break:break-word;}",
        "</style></head><body>",
        "<h1>Salamander Local Graph Error Review</h1>",
        f"<div class='summary'><div>Predictions: {html.escape(str(predictions_path))}</div><div>Pair scores: {html.escape(str(pair_scores_path))}</div><div>Cases shown: top {int(top_n)} error queries</div></div>",
    ]

    for rank, case in enumerate(selected_cases.itertuples(index=False), start=1):
        query_row = row_by_image.loc[str(case.image_id)]
        query_identity = str(case.identity)
        query_cluster = str(case.cluster_label)
        true_ids = set(identity_to_images.get(query_identity, [])) - {str(case.image_id)}
        pred_ids = set(cluster_to_images.get(query_cluster, [])) - {str(case.image_id)}
        missed_ids = sorted(true_ids - pred_ids)
        false_pos_ids = sorted(pred_ids - true_ids)

        query_neighbors = (
            neighbor_df[neighbor_df["query_image_id"].eq(str(case.image_id))]
            .sort_values(["local_only_score_v1", "route_global_score", "neighbor_image_id"], ascending=[False, False, True])
            .head(int(top_neighbors))
        )

        sections.append("<div class='case'>")
        sections.append(
            f"<h2>#{rank} image_id={html.escape(str(case.image_id))} | identity={html.escape(query_identity)} | "
            f"missed={int(case.missed_count)} | false_pos={int(case.false_pos_count)} | pred_cluster_size={int(case.pred_cluster_size_excl_self) + 1}</h2>"
        )
        sections.append("<h3>Query</h3><div class='grid'>")
        sections.append(
            _image_card(
                row=query_row,
                html_dir=output_dir,
                repo_root=repo_root,
                caption=f"{case.image_id}",
                css_class="query",
                extra_lines=[f"identity={query_identity}", f"cluster={query_cluster}"],
            )
        )
        sections.append("</div>")

        sections.append("<h3>Missed True Mates</h3><div class='grid'>")
        if missed_ids:
            for mate_id in missed_ids:
                mate_row = row_by_image.loc[mate_id]
                sections.append(
                    _image_card(
                        row=mate_row,
                        html_dir=output_dir,
                        repo_root=repo_root,
                        caption=mate_id,
                        css_class="missed",
                        extra_lines=[f"identity={query_identity}", f"pred_cluster={mate_row['cluster_label']}"],
                    )
                )
        else:
            sections.append("<div class='meta'>None</div>")
        sections.append("</div>")

        sections.append("<h3>False Positives In Predicted Cluster</h3><div class='grid'>")
        if false_pos_ids:
            for fp_id in false_pos_ids:
                fp_row = row_by_image.loc[fp_id]
                sections.append(
                    _image_card(
                        row=fp_row,
                        html_dir=output_dir,
                        repo_root=repo_root,
                        caption=fp_id,
                        css_class="fp",
                        extra_lines=[f"identity={fp_row['identity']}", f"pred_cluster={query_cluster}"],
                    )
                )
        else:
            sections.append("<div class='meta'>None</div>")
        sections.append("</div>")

        sections.append("<h3>Top Local Neighbors</h3><div class='grid'>")
        if not query_neighbors.empty:
            for neighbor in query_neighbors.itertuples(index=False):
                neighbor_row = row_by_image.loc[str(neighbor.neighbor_image_id)]
                css_class = "good" if int(neighbor.same_identity) == 1 else "neighbor"
                sections.append(
                    _image_card(
                        row=neighbor_row,
                        html_dir=output_dir,
                        repo_root=repo_root,
                        caption=str(neighbor.neighbor_image_id),
                        css_class=css_class,
                        extra_lines=[
                            f"identity={neighbor_row['identity']}",
                            f"local_score={float(neighbor.local_only_score_v1):.4f}",
                            f"global_score={float(neighbor.route_global_score):.4f}",
                            f"decision={str(neighbor.yellow_veto_decision_v1)}",
                            f"same_identity={int(neighbor.same_identity)}",
                        ],
                    )
                )
        else:
            sections.append("<div class='meta'>No candidate neighbors</div>")
        sections.append("</div></div>")

    sections.append("</body></html>")
    html_path.write_text("\n".join(sections), encoding="utf-8")
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Salamander local-graph error review HTML.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--pair-scores-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-neighbors", type=int, default=8)
    args = parser.parse_args()

    html_path = build_review_html(
        repo_root=args.repo_root.resolve(),
        predictions_path=args.predictions_path.resolve(),
        pair_scores_path=args.pair_scores_path.resolve(),
        output_dir=args.output_dir.resolve(),
        top_n=int(args.top_n),
        top_neighbors=int(args.top_neighbors),
    )
    print(f"[salamander_local_graph_review] html: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
