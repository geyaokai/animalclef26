#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import itertools
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a small HTML review board for a Texas protected-conflict triplet.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--image-ids", nargs="+", required=True)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=repo_root / "metadata.csv",
    )
    parser.add_argument(
        "--prediction-path",
        type=Path,
        default=repo_root
        / "artifacts/submissions/kaggle_variant_texas_manual_graph_v2_on_059138_base_v1/tables/test_predictions_v1.csv",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v1/texas_pair_registry_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_protected_conflict_review_v1",
    )
    return parser.parse_args()


def relpath(from_dir: Path, target: Path) -> str:
    return str(target.resolve().relative_to(from_dir.resolve())) if str(target.resolve()).startswith(str(from_dir.resolve())) else str(Path("../") / Path(Path(target).resolve().relative_to(from_dir.resolve().parents[0])))


def safe_relpath(from_dir: Path, target: Path) -> str:
    return str(Path(Path(target).resolve().relative_to(Path.commonpath([from_dir.resolve(), target.resolve()]))))


def to_rel(from_dir: Path, target: Path) -> str:
    return str(Path(pd.io.common.os.path.relpath(str(target.resolve()), str(from_dir.resolve()))))


def pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_ids = [str(value) for value in args.image_ids]

    metadata_df = pd.read_csv(args.metadata_path.resolve())
    metadata_df["image_id"] = metadata_df["image_id"].astype(str)
    pred_df = pd.read_csv(args.prediction_path.resolve())
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    registry_df = pd.read_csv(args.registry_path.resolve())
    registry_df["image_id_a"] = registry_df["image_id_a"].astype(str)
    registry_df["image_id_b"] = registry_df["image_id_b"].astype(str)

    target_meta_df = metadata_df[metadata_df["image_id"].isin(image_ids)].copy()
    target_pred_df = pred_df[pred_df["image_id"].isin(image_ids)].copy()

    if len(target_meta_df) != len(image_ids):
        missing = sorted(set(image_ids) - set(target_meta_df["image_id"].tolist()))
        raise SystemExit(f"missing metadata rows for image_ids={missing}")
    if len(target_pred_df) != len(image_ids):
        missing = sorted(set(image_ids) - set(target_pred_df["image_id"].tolist()))
        raise SystemExit(f"missing prediction rows for image_ids={missing}")

    meta_by_id = {str(row.image_id): row for row in target_meta_df.itertuples(index=False)}
    pred_by_id = {str(row.image_id): row for row in target_pred_df.itertuples(index=False)}

    pair_rows: list[dict[str, str]] = []
    for left, right in itertools.combinations(sorted(image_ids), 2):
        sub = registry_df[
            (
                (registry_df["image_id_a"].eq(left) & registry_df["image_id_b"].eq(right))
                | (registry_df["image_id_a"].eq(right) & registry_df["image_id_b"].eq(left))
            )
        ].copy()
        if sub.empty:
            pair_rows.append(
                {
                    "pair_id": f"{left}_{right}",
                    "image_id_a": left,
                    "image_id_b": right,
                    "constraint_type": "unlabeled",
                    "support_count": "",
                    "sources": "",
                    "notes": "",
                }
            )
            continue
        for row in sub.sort_values(["constraint_type", "support_count"], ascending=[True, False]).itertuples(index=False):
            pair_rows.append(
                {
                    "pair_id": str(row.pair_id),
                    "image_id_a": left,
                    "image_id_b": right,
                    "constraint_type": str(row.constraint_type),
                    "support_count": str(row.support_count),
                    "sources": str(row.sources),
                    "notes": str(row.notes),
                }
            )

    current_cluster_ids = sorted({int(pred_by_id[image_id].pred_cluster_id) for image_id in image_ids})

    image_cards: list[str] = []
    for image_id in sorted(image_ids):
        meta_row = meta_by_id[image_id]
        pred_row = pred_by_id[image_id]
        original_path = (repo_root / str(meta_row.path)).resolve()
        aligned_path = (repo_root / str(pred_row.path)).resolve()
        image_cards.append(
            f"""
            <div class="image-card">
              <div class="image-header">
                <h3>{html.escape(image_id)}</h3>
                <div class="meta-chip">cluster {html.escape(str(pred_row.pred_cluster_id))}</div>
              </div>
              <div class="image-grid">
                <div class="view-block">
                  <div class="view-title">Original</div>
                  <a href="{html.escape(to_rel(output_dir, original_path))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, original_path))}" alt="original {html.escape(image_id)}" />
                  </a>
                  <div class="path-note">{html.escape(str(meta_row.path))}</div>
                </div>
                <div class="view-block">
                  <div class="view-title">Aligned / Model View</div>
                  <a href="{html.escape(to_rel(output_dir, aligned_path))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, aligned_path))}" alt="aligned {html.escape(image_id)}" />
                  </a>
                  <div class="path-note">{html.escape(str(pred_row.path))}</div>
                </div>
              </div>
            </div>
            """
        )

    pair_cards: list[str] = []
    for row in pair_rows:
        left = row["image_id_a"]
        right = row["image_id_b"]
        left_meta = meta_by_id[left]
        right_meta = meta_by_id[right]
        left_pred = pred_by_id[left]
        right_pred = pred_by_id[right]
        left_img = (repo_root / str(left_pred.path)).resolve()
        right_img = (repo_root / str(right_pred.path)).resolve()
        pair_cards.append(
            f"""
            <div class="pair-card pair-{html.escape(row['constraint_type'])}">
              <div class="pair-title">
                <h3>{html.escape(left)} vs {html.escape(right)}</h3>
                <span class="badge">{html.escape(row['constraint_type'])}</span>
                <span class="badge muted">support {html.escape(row['support_count']) or '-'}</span>
              </div>
              <div class="pair-grid">
                <div class="pair-image">
                  <a href="{html.escape(to_rel(output_dir, left_img))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, left_img))}" alt="{html.escape(left)}" />
                  </a>
                  <div class="caption">{html.escape(left)} · cluster {html.escape(str(left_pred.pred_cluster_id))}</div>
                </div>
                <div class="pair-image">
                  <a href="{html.escape(to_rel(output_dir, right_img))}" target="_blank">
                    <img src="{html.escape(to_rel(output_dir, right_img))}" alt="{html.escape(right)}" />
                  </a>
                  <div class="caption">{html.escape(right)} · cluster {html.escape(str(right_pred.pred_cluster_id))}</div>
                </div>
              </div>
              <div class="pair-notes">
                <div><strong>Source</strong>: {html.escape(row['sources'] or '-')}</div>
                <div><strong>Registry Note</strong>: {html.escape(row['notes'] or '-')}</div>
                <div><strong>Original Paths</strong>: {html.escape(str(left_meta.path))} | {html.escape(str(right_meta.path))}</div>
              </div>
            </div>
            """
        )

    triplet_payload = {
        "image_ids": sorted(image_ids),
        "current_cluster_ids": current_cluster_ids,
        "pair_rows": pair_rows,
    }
    (output_dir / "triplet_context_v1.json").write_text(json.dumps(triplet_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Texas Protected Conflict Review</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #101418; color: #ecf0f1; }}
    main {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0; }}
    .hero {{ background: #171d23; border: 1px solid #2a3642; border-radius: 14px; padding: 20px; margin-bottom: 24px; }}
    .meta-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }}
    .chip, .badge, .meta-chip {{ display: inline-block; padding: 6px 10px; border-radius: 999px; background: #243140; color: #d9e6f2; font-size: 13px; }}
    .badge {{ background: #334155; }}
    .badge.muted {{ background: #3b4652; color: #d0d7de; }}
    .section {{ margin-bottom: 28px; }}
    .image-card, .pair-card {{ background: #171d23; border: 1px solid #2a3642; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    .image-header, .pair-title {{ display: flex; align-items: center; gap: 10px; justify-content: space-between; margin-bottom: 14px; flex-wrap: wrap; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .view-block, .pair-image {{ background: #0f1419; border-radius: 12px; padding: 12px; }}
    .view-title {{ font-size: 13px; color: #9fb3c8; margin-bottom: 8px; }}
    img {{ width: 100%; height: auto; max-height: 540px; object-fit: contain; background: #000; border-radius: 8px; }}
    .path-note, .caption, .pair-notes {{ font-size: 12px; color: #c5d1dd; margin-top: 8px; line-height: 1.5; word-break: break-all; }}
    .pair-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-bottom: 10px; }}
    .pair-card.pair-must-link {{ border-color: #2e8b57; }}
    .pair-card.pair-cannot-link {{ border-color: #b94a48; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #2a3642; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #243140; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Texas Protected Conflict Review</h1>
      <div class="meta-row">
        <span class="chip">image_ids: {' | '.join(html.escape(x) for x in sorted(image_ids))}</span>
        <span class="chip">current cluster: {' | '.join(str(x) for x in current_cluster_ids)}</span>
        <span class="chip">conflict type: cannot-link inside must-link-supported cluster</span>
      </div>
      <table>
        <thead>
          <tr><th>Pair</th><th>Constraint</th><th>Support</th><th>Source</th></tr>
        </thead>
        <tbody>
          {''.join(f"<tr><td>{html.escape(row['image_id_a'])} vs {html.escape(row['image_id_b'])}</td><td>{html.escape(row['constraint_type'])}</td><td>{html.escape(row['support_count'] or '-')}</td><td>{html.escape(row['sources'] or '-')}</td></tr>" for row in pair_rows)}
        </tbody>
      </table>
    </section>

    <section class="section">
      <h2>Triplet Overview</h2>
      {''.join(image_cards)}
    </section>

    <section class="section">
      <h2>Pairwise Comparison</h2>
      {''.join(pair_cards)}
    </section>
  </main>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    print(f"[texas_conflict_review] html: {output_dir / 'index.html'}")
    print(f"[texas_conflict_review] json: {output_dir / 'triplet_context_v1.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
