#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build an HTML review page for Salamander validation top-k neighbors.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--filter-mode", choices=["all", "miss_top1", "miss_top5", "miss_top10"], default="miss_top10")
    parser.add_argument("--max-queries", type=int, default=60)
    parser.add_argument("--sort-by", choices=["hardest", "easiest", "image_id"], default="hardest")
    return parser.parse_args()


def _resolve_path_column(frame: pd.DataFrame) -> str:
    for column in ["path", "preferred_path_v1", "normalized_path_v1", "global_path", "source_global_path"]:
        if column in frame.columns:
            return column
    raise KeyError("No image path column found in metadata frame.")


def _resolve_embedding_artifacts(experiment_dir: Path) -> tuple[Path, Path]:
    candidates = [
        (experiment_dir / "embeddings" / "val_metadata.csv", experiment_dir / "embeddings" / "val_embeddings.npy"),
        (
            experiment_dir / "embeddings" / "salamander_val_metadata.csv",
            experiment_dir / "embeddings" / "salamander_val_embeddings.npy",
        ),
    ]
    for metadata_path, embedding_path in candidates:
        if metadata_path.exists() and embedding_path.exists():
            return metadata_path, embedding_path
    raise FileNotFoundError(
        f"Could not find supported val metadata / embedding artifacts under: {experiment_dir / 'embeddings'}"
    )


def _compute_topk(embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    similarity = np.asarray(embeddings, dtype=np.float32) @ np.asarray(embeddings, dtype=np.float32).T
    np.fill_diagonal(similarity, -np.inf)
    width = min(int(top_k), max(1, len(similarity) - 1))
    topk = np.argpartition(-similarity, kth=width - 1, axis=1)[:, :width]
    sorted_topk = np.empty_like(topk)
    sorted_scores = np.empty((len(similarity), width), dtype=np.float32)
    for index in range(len(similarity)):
        ranked = sorted(topk[index].tolist(), key=lambda idx: similarity[index, idx], reverse=True)
        sorted_topk[index] = np.asarray(ranked, dtype=np.int32)
        sorted_scores[index] = similarity[index, sorted_topk[index]]
    return sorted_topk, sorted_scores


def _hit_at(neighbors: np.ndarray, labels: np.ndarray, index: int, k: int) -> bool:
    width = min(int(k), int(neighbors.shape[1]))
    if width <= 0:
        return False
    identity = str(labels[index])
    if not identity:
        return False
    return bool(np.any(labels[neighbors[index, :width]] == identity))


def _to_rel(from_dir: Path, target: Path) -> str:
    return str(Path(target.resolve().relative_to(target.resolve().anchor)) if False else Path(pd.io.common.os.path.relpath(str(target.resolve()), str(from_dir.resolve()))))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    experiment_dir = args.experiment_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "artifacts" / "analysis" / f"{experiment_dir.name}_topk_review_html"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path, embedding_path = _resolve_embedding_artifacts(experiment_dir)
    metadata_df = pd.read_csv(metadata_path)
    metadata_df["image_id"] = metadata_df["image_id"].astype(str)
    metadata_df["dataset"] = metadata_df["dataset"].astype(str)
    if "identity" in metadata_df.columns:
        metadata_df["identity"] = metadata_df["identity"].fillna("").astype(str)
    metadata_df = metadata_df[metadata_df["dataset"] == SALAMANDER_DATASET].reset_index(drop=True)
    embeddings = np.load(embedding_path).astype(np.float32)
    if len(metadata_df) != len(embeddings):
        full_metadata = pd.read_csv(metadata_path)
        full_metadata["dataset"] = full_metadata["dataset"].astype(str)
        mask = full_metadata["dataset"].eq(SALAMANDER_DATASET).to_numpy()
        embeddings = embeddings[mask]
    if len(metadata_df) != len(embeddings):
        raise ValueError("Salamander metadata and embedding rows do not match.")

    path_column = _resolve_path_column(metadata_df)
    labels = metadata_df["identity"].to_numpy(dtype=object)
    topk_indices, topk_scores = _compute_topk(embeddings=embeddings, top_k=args.top_k)

    query_rows: list[dict[str, object]] = []
    for index, row in enumerate(metadata_df.itertuples(index=False)):
        hit1 = _hit_at(topk_indices, labels, index, 1)
        hit5 = _hit_at(topk_indices, labels, index, 5)
        hit10 = _hit_at(topk_indices, labels, index, 10)
        neighbors = topk_indices[index]
        positive_ranks = [rank + 1 for rank, neighbor_idx in enumerate(neighbors) if labels[int(neighbor_idx)] == labels[index] and str(labels[index])]
        first_positive_rank = min(positive_ranks) if positive_ranks else 999
        query_rows.append(
            {
                "index": index,
                "image_id": str(row.image_id),
                "identity": str(getattr(row, "identity", "")),
                "path": str(getattr(row, path_column)),
                "hit_top1": hit1,
                "hit_top5": hit5,
                "hit_top10": hit10,
                "first_positive_rank": int(first_positive_rank),
            }
        )
    query_df = pd.DataFrame(query_rows)

    if args.filter_mode == "miss_top1":
        query_df = query_df[~query_df["hit_top1"]].copy()
    elif args.filter_mode == "miss_top5":
        query_df = query_df[~query_df["hit_top5"]].copy()
    elif args.filter_mode == "miss_top10":
        query_df = query_df[~query_df["hit_top10"]].copy()

    if args.sort_by == "hardest":
        query_df = query_df.sort_values(["first_positive_rank", "image_id"], ascending=[False, True])
    elif args.sort_by == "easiest":
        query_df = query_df.sort_values(["first_positive_rank", "image_id"], ascending=[True, True])
    else:
        query_df = query_df.sort_values("image_id", ascending=True)
    query_df = query_df.head(int(args.max_queries)).reset_index(drop=True)

    summary_rows = [
        ("queries_total", int(len(metadata_df))),
        ("shown_queries", int(len(query_df))),
        ("recall_at_1", round(float(np.mean([_hit_at(topk_indices, labels, i, 1) for i in range(len(metadata_df))])) if len(metadata_df) else 0.0, 6)),
        ("recall_at_5", round(float(np.mean([_hit_at(topk_indices, labels, i, 5) for i in range(len(metadata_df))])) if len(metadata_df) else 0.0, 6)),
        ("recall_at_10", round(float(np.mean([_hit_at(topk_indices, labels, i, 10) for i in range(len(metadata_df))])) if len(metadata_df) else 0.0, 6)),
        ("filter_mode", str(args.filter_mode)),
    ]

    cards: list[str] = []
    for item in query_df.itertuples(index=False):
        query_index = int(item.index)
        query_abs = (repo_root / str(item.path)).resolve()
        neighbor_blocks: list[str] = []
        for rank, neighbor_idx in enumerate(topk_indices[query_index].tolist(), start=1):
            neighbor_row = metadata_df.iloc[int(neighbor_idx)]
            neighbor_abs = (repo_root / str(neighbor_row[path_column])).resolve()
            same_identity = str(neighbor_row.get("identity", "")) == str(item.identity) and str(item.identity) != ""
            block_class = "neighbor hit" if same_identity else "neighbor miss"
            neighbor_blocks.append(
                f"""
                <div class="{block_class}">
                  <div class="neighbor-meta">
                    <span class="badge">rank {rank}</span>
                    <span class="badge">{'hit' if same_identity else 'miss'}</span>
                    <span class="badge">sim {float(topk_scores[query_index, rank - 1]):.4f}</span>
                  </div>
                  <a href="{html.escape(_to_rel(output_dir, neighbor_abs))}" target="_blank">
                    <img src="{html.escape(_to_rel(output_dir, neighbor_abs))}" alt="{html.escape(str(neighbor_row['image_id']))}" />
                  </a>
                  <div class="caption">
                    <div><strong>{html.escape(str(neighbor_row['image_id']))}</strong></div>
                    <div>identity: {html.escape(str(neighbor_row.get('identity', '')) or '-')}</div>
                    <div>path: {html.escape(str(neighbor_row[path_column]))}</div>
                  </div>
                </div>
                """
            )
        cards.append(
            f"""
            <section class="query-card">
              <div class="query-header">
                <div>
                  <h2>{html.escape(str(item.image_id))}</h2>
                  <div class="meta-line">
                    <span class="badge">identity {html.escape(str(item.identity) or '-')}</span>
                    <span class="badge">{'top1 hit' if bool(item.hit_top1) else 'top1 miss'}</span>
                    <span class="badge">{'top5 hit' if bool(item.hit_top5) else 'top5 miss'}</span>
                    <span class="badge">{'top10 hit' if bool(item.hit_top10) else 'top10 miss'}</span>
                    <span class="badge">first positive rank {int(item.first_positive_rank) if int(item.first_positive_rank) < 999 else 'none'}</span>
                  </div>
                </div>
              </div>
              <div class="query-layout">
                <div class="query-panel">
                  <div class="panel-title">Query</div>
                  <a href="{html.escape(_to_rel(output_dir, query_abs))}" target="_blank">
                    <img src="{html.escape(_to_rel(output_dir, query_abs))}" alt="{html.escape(str(item.image_id))}" />
                  </a>
                  <div class="caption">
                    <div>path: {html.escape(str(item.path))}</div>
                  </div>
                </div>
                <div class="neighbors-grid">
                  {''.join(neighbor_blocks)}
                </div>
              </div>
            </section>
            """
        )

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Salamander Top-{int(args.top_k)} Review</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
    .hero, .query-card {{ background: #1f2937; border: 1px solid #374151; border-radius: 16px; padding: 18px; margin-bottom: 20px; }}
    .badge {{ display: inline-block; margin: 4px 6px 0 0; padding: 6px 10px; border-radius: 999px; background: #374151; font-size: 12px; }}
    .meta-line {{ margin-top: 8px; }}
    .query-layout {{ display: grid; grid-template-columns: 320px 1fr; gap: 16px; }}
    .query-panel, .neighbor {{ background: #111827; border-radius: 12px; padding: 10px; }}
    .neighbors-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
    .neighbor.hit {{ border: 2px solid #16a34a; }}
    .neighbor.miss {{ border: 2px solid #7c2d12; }}
    .panel-title {{ font-size: 13px; color: #93c5fd; margin-bottom: 8px; }}
    .neighbor-meta {{ margin-bottom: 8px; }}
    .caption {{ font-size: 12px; line-height: 1.45; color: #d1d5db; word-break: break-all; margin-top: 8px; }}
    img {{ width: 100%; height: auto; max-height: 260px; object-fit: contain; background: #000; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #374151; padding: 8px 10px; text-align: left; }}
    th {{ background: #111827; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Salamander Validation Top-{int(args.top_k)} Review</h1>
      <div class="meta-line">
        {''.join(f'<span class="badge">{html.escape(str(key))}: {html.escape(str(value))}</span>' for key, value in summary_rows)}
      </div>
      <table>
        <thead><tr><th>image_id</th><th>identity</th><th>top1</th><th>top5</th><th>top10</th><th>first_positive_rank</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td>{html.escape(str(row.image_id))}</td><td>{html.escape(str(row.identity) or '-')}</td><td>{'hit' if bool(row.hit_top1) else 'miss'}</td><td>{'hit' if bool(row.hit_top5) else 'miss'}</td><td>{'hit' if bool(row.hit_top10) else 'miss'}</td><td>{int(row.first_positive_rank) if int(row.first_positive_rank) < 999 else 'none'}</td></tr>" for row in query_df.itertuples(index=False))}
        </tbody>
      </table>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    print(f"[salamander_topk_review_html] html: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
