#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PROBE_DIR = Path("/data1/gyk/animalclef_artifacts_parent/artifacts/analysis/salamander_graph_merge_overlay_probe_strict_v1")
DEFAULT_OUTPUT_DIRNAME = "merge_review_html"
DEFAULT_TOP_PAIRS = 6


def _to_rel(target: Path, base: Path) -> str:
    return os.path.relpath(target, base)


def _pick_image_path(row: pd.Series, repo_root: Path) -> Path:
    for column in [
        "body_axis_unsigned_rgb_v1_resolved_path_v1",
        "body_axis_unsigned_rgb_v1_export_path",
        "preferred_path_v1",
        "path",
    ]:
        value = row.get(column)
        if pd.isna(value):
            continue
        candidate = repo_root / str(value)
        if candidate.exists():
            return candidate
    return repo_root / str(row["path"])


def _pick_original_path(row: pd.Series, repo_root: Path) -> Path:
    for column in ["original_rgb_path_v1", "path"]:
        value = row.get(column)
        if pd.isna(value):
            continue
        candidate = repo_root / str(value)
        if candidate.exists():
            return candidate
    return repo_root / str(row["path"])


def _apply_pair_probability_as_score(base_score: np.ndarray, pair_df: pd.DataFrame, probability_col: str, blend_scale: float) -> np.ndarray:
    fused = base_score.copy().astype(np.float32, copy=True)
    for row in pair_df.itertuples(index=False):
        left_index = int(row.left_index)
        right_index = int(row.right_index)
        base_value = float(base_score[left_index, right_index])
        probability = float(getattr(row, probability_col))
        score = min(1.0, base_value + float(blend_scale) * probability * (1.0 - base_value))
        fused[left_index, right_index] = score
        fused[right_index, left_index] = score
    np.fill_diagonal(fused, 1.0)
    return fused


def _stage_image(source_path: Path, *, asset_dir: Path, cache: dict[str, Path], key: str) -> Path:
    resolved = source_path.resolve()
    cache_key = f"{key}|{resolved}"
    if cache_key in cache:
        return cache[cache_key]
    suffix = resolved.suffix or ".jpg"
    safe_key = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in key)
    staged_path = asset_dir / f"{safe_key}{suffix}"
    if not staged_path.exists():
        shutil.copy2(resolved, staged_path)
    cache[cache_key] = staged_path
    return staged_path


def _member_card(
    row: pd.Series,
    *,
    html_dir: Path,
    repo_root: Path,
    label: str,
    asset_dir: Path,
    asset_cache: dict[str, Path],
) -> str:
    original_path = _pick_original_path(row, repo_root=repo_root)
    display_path = _pick_image_path(row, repo_root=repo_root)
    image_id_key = str(row["image_id"])
    staged_original = _stage_image(original_path, asset_dir=asset_dir, cache=asset_cache, key=f"{image_id_key}_original")
    staged_display = _stage_image(display_path, asset_dir=asset_dir, cache=asset_cache, key=f"{image_id_key}_{label}_display")
    original_rel = html.escape(_to_rel(staged_original, html_dir))
    display_rel = html.escape(_to_rel(staged_display, html_dir))
    image_id = html.escape(str(row["image_id"]))
    orientation = html.escape(str(row.get("orientation", "")) or "-")
    date = html.escape(str(row.get("date", "")) or "-")
    return f"""
    <div class="member-card">
      <div class="member-head">
        <span class="badge">{html.escape(label)}</span>
        <span class="badge muted">{image_id}</span>
      </div>
      <a href="{display_rel}" target="_blank"><img src="{display_rel}" loading="lazy" alt="{image_id}"></a>
      <div class="meta">orientation: {orientation}</div>
      <div class="meta">date: {date}</div>
      <div class="links"><a href="{original_rel}" target="_blank">original</a> · <a href="{display_rel}" target="_blank">aligned/display</a></div>
    </div>
    """


def _pair_card(
    row: pd.Series,
    pred_by_id: dict[str, pd.Series],
    *,
    html_dir: Path,
    repo_root: Path,
    asset_dir: Path,
    asset_cache: dict[str, Path],
) -> str:
    left_id = str(row["image_id"])
    right_id = str(row["neighbor_image_id"])
    left_pred = pred_by_id[left_id]
    right_pred = pred_by_id[right_id]
    left_img = _stage_image(
        _pick_image_path(left_pred, repo_root=repo_root),
        asset_dir=asset_dir,
        cache=asset_cache,
        key=f"{left_id}_pair_display",
    )
    right_img = _stage_image(
        _pick_image_path(right_pred, repo_root=repo_root),
        asset_dir=asset_dir,
        cache=asset_cache,
        key=f"{right_id}_pair_display",
    )
    left_rel = html.escape(_to_rel(left_img, html_dir))
    right_rel = html.escape(_to_rel(right_img, html_dir))
    return f"""
    <div class="pair-card">
      <div class="pair-images">
        <div class="pair-side">
          <a href="{left_rel}" target="_blank"><img src="{left_rel}" loading="lazy" alt="{html.escape(left_id)}"></a>
          <div class="caption">{html.escape(left_id)} · base {int(row["base_cluster_left"])}</div>
        </div>
        <div class="pair-side">
          <a href="{right_rel}" target="_blank"><img src="{right_rel}" loading="lazy" alt="{html.escape(right_id)}"></a>
          <div class="caption">{html.escape(right_id)} · base {int(row["base_cluster_right"])}</div>
        </div>
      </div>
      <div class="pair-meta">
        <span class="badge">pair_score {float(row["pair_score"]):.6f}</span>
        <span class="badge">xgb {float(row["xgb_same_identity_prob"]):.6f}</span>
        <span class="badge">merge_votes {int(row["merge_votes"])}/{int(row["total_votes"])}</span>
        <span class="badge muted">local {float(row.get("local_score", 0.0)):.6f}</span>
        <span class="badge muted">inliers {int(row.get("inliers", 0))}</span>
      </div>
    </div>
    """


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.orb_rerank_baseline import cosine_score_matrix
    from animalclef_analysis.transductive_seed_refinement import cluster_labels_from_score_matrix

    parser = argparse.ArgumentParser(description="Build an HTML review page for Salamander consensus merge candidates.")
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-pairs", type=int, default=DEFAULT_TOP_PAIRS)
    parser.add_argument("--limit-candidates", type=int, default=0)
    args = parser.parse_args()

    probe_dir = args.probe_dir.resolve()
    output_dir = probe_dir / DEFAULT_OUTPUT_DIRNAME if args.output_dir is None else args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = output_dir / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_cache: dict[str, Path] = {}

    summary = json.loads((probe_dir / "reports" / "summary.json").read_text(encoding="utf-8"))
    route_dir = Path(summary["route_dir"]).resolve()
    xgb_variant_dir = Path(summary["xgb_variant_dir"]).resolve()
    blend_scale = float(summary.get("blend_scale", 1.0))

    pred_path = probe_dir / "tables" / "test_predictions_best_overall_v1.csv"
    merge_path = probe_dir / "tables" / "test_merge_candidates_best_overall_v1.csv"
    pair_path = probe_dir / "tables" / "test_pair_vote_summary_best_overall_v1.csv"
    predictions_df = pd.read_csv(pred_path)
    merge_df = pd.read_csv(merge_path)
    pair_df = pd.read_csv(pair_path)
    predictions_df["image_id"] = predictions_df["image_id"].astype(str)
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    pred_by_id = {str(row["image_id"]): row for _, row in predictions_df.iterrows()}

    if int(args.limit_candidates) > 0:
        merge_df = merge_df.head(int(args.limit_candidates)).copy()

    test_meta = pd.read_csv(route_dir / "embeddings" / "salamander_test_metadata.csv")
    test_embeddings = np.load(route_dir / "embeddings" / "salamander_test_embeddings.npy").astype(np.float32)
    test_meta["image_id"] = test_meta["image_id"].astype(str)
    route_score = cosine_score_matrix(test_embeddings)
    test_pair_features = pd.read_csv(xgb_variant_dir / "tables" / "test_pair_features_v1.csv")
    boosted_test_score = _apply_pair_probability_as_score(
        base_score=route_score,
        pair_df=test_pair_features,
        probability_col="xgb_same_identity_prob",
        blend_scale=blend_scale,
    )
    chosen_threshold = float(predictions_df["chosen_threshold"].iloc[0])
    base_labels = cluster_labels_from_score_matrix(
        score_matrix=boosted_test_score,
        threshold=chosen_threshold,
        score_space="unit_interval",
    )
    base_pred_df = test_meta.copy()
    base_pred_df["base_cluster_id"] = base_labels.astype(int)
    base_pred_by_cluster = {
        int(cluster_id): frame.sort_values("image_id").reset_index(drop=True)
        for cluster_id, frame in base_pred_df.groupby("base_cluster_id", sort=False)
    }

    sections: list[str] = []
    for rank, row in enumerate(merge_df.itertuples(index=False), start=1):
        left_cluster_id = int(row.left_cluster_id)
        right_cluster_id = int(row.right_cluster_id)
        left_members = base_pred_by_cluster.get(left_cluster_id, pd.DataFrame())
        right_members = base_pred_by_cluster.get(right_cluster_id, pd.DataFrame())
        support_pairs = pair_df[
            (
                (pair_df["base_cluster_left"].astype(int).eq(left_cluster_id) & pair_df["base_cluster_right"].astype(int).eq(right_cluster_id))
                | (pair_df["base_cluster_left"].astype(int).eq(right_cluster_id) & pair_df["base_cluster_right"].astype(int).eq(left_cluster_id))
            )
        ].copy()
        support_pairs = support_pairs.sort_values(
            ["pair_score", "xgb_same_identity_prob", "merge_votes"],
            ascending=[False, False, False],
        ).head(int(args.top_pairs))

        left_cards = "".join(
            _member_card(
                member,
                html_dir=output_dir,
                repo_root=repo_root,
                label=f"base {left_cluster_id}",
                asset_dir=asset_dir,
                asset_cache=asset_cache,
            )
            for _, member in left_members.iterrows()
        )
        right_cards = "".join(
            _member_card(
                member,
                html_dir=output_dir,
                repo_root=repo_root,
                label=f"base {right_cluster_id}",
                asset_dir=asset_dir,
                asset_cache=asset_cache,
            )
            for _, member in right_members.iterrows()
        )
        pair_cards = "".join(
            _pair_card(
                pair_row,
                pred_by_id,
                html_dir=output_dir,
                repo_root=repo_root,
                asset_dir=asset_dir,
                asset_cache=asset_cache,
            )
            for _, pair_row in support_pairs.iterrows()
        )

        sections.append(
            f"""
            <section class="candidate">
              <div class="candidate-head">
                <h2>#{rank} merge {html.escape(str(row.cluster_pair_key))}</h2>
                <div class="chips">
                  <span class="badge">support_pairs {int(row.support_pair_count)}</span>
                  <span class="badge">vote {int(row.max_merge_votes)} / 3</span>
                  <span class="badge">mean_pair_score {float(row.mean_pair_score):.6f}</span>
                  <span class="badge">mean_xgb {float(row.mean_pair_probability):.6f}</span>
                  <span class="badge muted">methods {html.escape(str(row.conflict_methods))}</span>
                </div>
              </div>
              <div class="cluster-grid">
                <div class="cluster-panel">
                  <h3>Base Cluster {left_cluster_id} · {len(left_members)} images</h3>
                  <div class="member-grid">{left_cards}</div>
                </div>
                <div class="cluster-panel">
                  <h3>Base Cluster {right_cluster_id} · {len(right_members)} images</h3>
                  <div class="member-grid">{right_cards}</div>
                </div>
              </div>
              <div class="support-panel">
                <h3>Top Support Pairs</h3>
                <div class="pair-grid">{pair_cards or '<div class=\"empty\">No support pairs.</div>'}</div>
              </div>
            </section>
            """
        )

    subtitle = [
        f"probe={probe_dir.name}",
        f"candidates={len(merge_df)}",
        f"threshold={chosen_threshold}",
        f"top_pairs={int(args.top_pairs)}",
    ]
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Salamander Consensus Merge Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }}
    .page {{ max-width: 1800px; margin: 0 auto; padding: 20px 24px 48px; }}
    h1, h2, h3 {{ margin: 0; }}
    .intro {{ margin: 12px 0 20px; color: #cbd5e1; }}
    .chips {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .badge {{ display:inline-block; padding:4px 8px; border-radius:999px; background:#1f2937; color:#e5e7eb; font-size:12px; }}
    .badge.muted {{ background:#374151; color:#cbd5e1; }}
    .candidate {{ border:1px solid #374151; border-radius:16px; padding:16px; margin:20px 0; background:#0f172a; }}
    .candidate-head {{ margin-bottom:14px; }}
    .cluster-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
    .cluster-panel, .support-panel {{ border:1px solid #1f2937; border-radius:12px; padding:12px; background:#111827; }}
    .member-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:12px; margin-top:12px; }}
    .member-card, .pair-card {{ border:1px solid #374151; border-radius:12px; padding:10px; background:#0b1220; }}
    .member-card img, .pair-side img {{ width:100%; border-radius:8px; display:block; }}
    .member-head {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }}
    .meta, .caption, .links {{ margin-top:6px; font-size:12px; color:#cbd5e1; }}
    .pair-grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(360px, 1fr)); gap:12px; margin-top:12px; }}
    .pair-images {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .pair-meta {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
    .empty {{ color:#94a3b8; padding:12px 0; }}
    a {{ color:#93c5fd; text-decoration:none; }}
    @media (max-width: 1100px) {{ .cluster-grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Salamander Consensus Merge Review</h1>
    <div class="intro">
      {' · '.join(html.escape(x) for x in subtitle)}<br>
      Strict best-overall merge candidates for manual HTML inspection.
    </div>
    {''.join(sections)}
  </div>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")
    manifest = {
        "probe_dir": str(probe_dir),
        "prediction_path": str(pred_path),
        "merge_candidate_path": str(merge_path),
        "pair_vote_path": str(pair_path),
        "html_path": str(html_path),
        "candidate_count": int(len(merge_df)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[salamander_consensus_merge_review_html] html: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
