#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import random
import shutil
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Salamander flip-invariant local-match review HTML.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=repo_root / "artifacts" / "manifests" / "sam_seg_trainprep_repaired_v1" / "tables" / "manifest_train_sam_trainprep_aligned_best_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "salamander_flip_invariant_review_v1",
    )
    parser.add_argument("--dataset", type=str, default="SalamanderID2025")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--nfeatures", type=int, default=512)
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--fast-threshold", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--ratio-test", type=float, default=0.75)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-pairs-per-identity", type=int, default=120)
    return parser.parse_args()


def _resolve_path_column(df: pd.DataFrame) -> str:
    for column in ["preferred_path_v1", "path", "recommended_model_input_path_v1", "original_rgb_path_v1"]:
        if column in df.columns:
            return column
    raise KeyError("No supported path column found.")


def _resolve_abs(repo_root: Path, rel_path: str) -> Path:
    path = (repo_root / str(rel_path)).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _stage_copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_hflip(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    with Image.open(src) as image:
        flipped = image.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(dst, quality=95)


def _to_rel(from_dir: Path, to_path: Path) -> str:
    return os.path.relpath(str(to_path), str(from_dir))


def _build_positive_pairs(df: pd.DataFrame, *, max_pairs_per_identity: int, sample_seed: int) -> list[tuple[int, int]]:
    rng = random.Random(int(sample_seed))
    rows: list[tuple[int, int]] = []
    for _identity, group_df in df.groupby("identity"):
        indices = group_df.index.to_list()
        if len(indices) < 2:
            continue
        pairs = list(combinations(indices, 2))
        if len(pairs) > int(max_pairs_per_identity):
            pairs = rng.sample(pairs, int(max_pairs_per_identity))
        rows.extend((int(left), int(right)) for left, right in pairs)
    return rows


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root / "src") not in sys.path:
        sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.orb_rerank_baseline import compute_local_match, extract_local_features

    manifest_path = args.manifest_path.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path, low_memory=False)
    df["dataset"] = df["dataset"].astype(str)
    df["split"] = df["split"].astype(str)
    df["image_id"] = df["image_id"].astype(str)
    df["identity"] = df["identity"].fillna("").astype(str)
    df = df[df["dataset"].eq(str(args.dataset)) & df["split"].eq(str(args.split)) & df["identity"].ne("")].copy().reset_index(drop=True)
    path_column = _resolve_path_column(df)
    df = df[df[path_column].fillna("").astype(str).ne("")].copy().reset_index(drop=True)
    if df.empty:
        raise SystemExit("No labeled Salamander rows found.")

    features = extract_local_features(
        df=df,
        repo_root=repo_root,
        nfeatures=int(args.nfeatures),
        max_side=int(args.max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        local_matcher="orb",
        hflip=False,
    )
    flipped_features = extract_local_features(
        df=df,
        repo_root=repo_root,
        nfeatures=int(args.nfeatures),
        max_side=int(args.max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        local_matcher="orb",
        hflip=True,
    )

    pair_rows: list[dict[str, object]] = []
    positive_pairs = _build_positive_pairs(
        df=df,
        max_pairs_per_identity=int(args.max_pairs_per_identity),
        sample_seed=int(args.sample_seed),
    )
    for left_index, right_index in positive_pairs:
        base = compute_local_match(
            left_feature=features[left_index],
            right_feature=features[right_index],
            ratio_test=float(args.ratio_test),
            ransac_threshold=float(args.ransac_threshold),
            min_inliers=int(args.min_inliers),
            local_matcher="orb",
        )
        flipped = compute_local_match(
            left_feature=features[left_index],
            right_feature=flipped_features[right_index],
            ratio_test=float(args.ratio_test),
            ransac_threshold=float(args.ransac_threshold),
            min_inliers=int(args.min_inliers),
            local_matcher="orb",
        )
        pair_rows.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "left_image_id": str(df.iloc[left_index]["image_id"]),
                "right_image_id": str(df.iloc[right_index]["image_id"]),
                "identity": str(df.iloc[left_index]["identity"]),
                "left_path": str(df.iloc[left_index][path_column]),
                "right_path": str(df.iloc[right_index][path_column]),
                "base_good_matches": int(base["good_matches"]),
                "base_inliers": int(base["inliers"]),
                "base_local_raw_score": float(base["local_raw_score"]),
                "flip_good_matches": int(flipped["good_matches"]),
                "flip_inliers": int(flipped["inliers"]),
                "flip_local_raw_score": float(flipped["local_raw_score"]),
            }
        )
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise SystemExit("No positive pairs were generated.")

    pair_df["delta_local_raw_score"] = pair_df["flip_local_raw_score"] - pair_df["base_local_raw_score"]
    pair_df["delta_inliers"] = pair_df["flip_inliers"] - pair_df["base_inliers"]
    pair_df["flip_better"] = (
        (pair_df["delta_local_raw_score"] > 0)
        | ((pair_df["delta_local_raw_score"] == 0) & (pair_df["delta_inliers"] > 0))
    )
    pair_df = pair_df.sort_values(
        ["flip_better", "delta_local_raw_score", "delta_inliers", "flip_local_raw_score", "left_image_id", "right_image_id"],
        ascending=[False, False, False, False, True, True],
    ).reset_index(drop=True)
    review_df = pair_df.head(int(args.top_n)).copy()

    pair_csv = output_dir / "flip_invariant_positive_pairs.csv"
    pair_df.to_csv(pair_csv, index=False)
    review_csv = output_dir / "flip_invariant_top_review_pairs.csv"
    review_df.to_csv(review_csv, index=False)

    staged_dir = output_dir / "staged_assets"
    cards: list[str] = []
    for row in review_df.itertuples(index=False):
        left_abs = _resolve_abs(repo_root, str(row.left_path))
        right_abs = _resolve_abs(repo_root, str(row.right_path))
        left_stage = staged_dir / "left" / f"{row.left_image_id}.jpg"
        right_stage = staged_dir / "right" / f"{row.right_image_id}.jpg"
        right_flip_stage = staged_dir / "right_hflip" / f"{row.right_image_id}.jpg"
        _stage_copy_or_link(left_abs, left_stage)
        _stage_copy_or_link(right_abs, right_stage)
        _stage_hflip(right_abs, right_flip_stage)
        left_rel = _to_rel(output_dir, left_stage)
        right_rel = _to_rel(output_dir, right_stage)
        right_flip_rel = _to_rel(output_dir, right_flip_stage)
        cards.append(
            f"""
            <section class="pair-card">
              <div class="pair-head">
                <div><strong>{html.escape(str(row.left_image_id))}</strong> vs <strong>{html.escape(str(row.right_image_id))}</strong></div>
                <div class="chips">
                  <span class="chip">identity {html.escape(str(row.identity))}</span>
                  <span class="chip {'good' if bool(row.flip_better) else 'muted'}">{'flip helps' if bool(row.flip_better) else 'no help'}</span>
                  <span class="chip">Δraw {float(row.delta_local_raw_score):.4f}</span>
                  <span class="chip">Δinliers {int(row.delta_inliers)}</span>
                </div>
              </div>
              <div class="grid">
                <div class="view-card">
                  <div class="title">left</div>
                  <a href="{html.escape(left_rel)}" target="_blank"><img src="{html.escape(left_rel)}" loading="lazy" /></a>
                  <div class="meta">{html.escape(str(row.left_image_id))}</div>
                </div>
                <div class="view-card">
                  <div class="title">right</div>
                  <a href="{html.escape(right_rel)}" target="_blank"><img src="{html.escape(right_rel)}" loading="lazy" /></a>
                  <div class="meta">{html.escape(str(row.right_image_id))}</div>
                  <div class="meta">base_raw={float(row.base_local_raw_score):.4f} | base_inliers={int(row.base_inliers)} | base_matches={int(row.base_good_matches)}</div>
                </div>
                <div class="view-card">
                  <div class="title">right_hflip</div>
                  <a href="{html.escape(right_flip_rel)}" target="_blank"><img src="{html.escape(right_flip_rel)}" loading="lazy" /></a>
                  <div class="meta">{html.escape(str(row.right_image_id))} hflip</div>
                  <div class="meta">flip_raw={float(row.flip_local_raw_score):.4f} | flip_inliers={int(row.flip_inliers)} | flip_matches={int(row.flip_good_matches)}</div>
                </div>
              </div>
              <div class="paths">
                <div>left_path: {html.escape(str(row.left_path))}</div>
                <div>right_path: {html.escape(str(row.right_path))}</div>
              </div>
            </section>
            """
        )

    summary = {
        "rows": int(len(df)),
        "positive_pairs": int(len(pair_df)),
        "flip_better_pairs": int(pair_df["flip_better"].sum()),
        "flip_better_ratio": round(float(pair_df["flip_better"].mean()) if len(pair_df) else 0.0, 6),
        "mean_delta_raw_score": round(float(pair_df["delta_local_raw_score"].mean()) if len(pair_df) else 0.0, 6),
        "median_delta_raw_score": round(float(pair_df["delta_local_raw_score"].median()) if len(pair_df) else 0.0, 6),
    }
    summary_rows = "".join(f"<tr><td>{html.escape(key)}</td><td>{html.escape(str(value))}</td></tr>" for key, value in summary.items())
    html_path = output_dir / "index.html"
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Salamander Flip-Invariant Review</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
    .hero, .pair-card {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 12px; }}
    .view-card {{ background: #020617; border: 1px solid #334155; border-radius: 12px; padding: 12px; }}
    .title {{ font-size: 14px; font-weight: 700; color: #93c5fd; margin-bottom: 8px; }}
    img {{ width: 100%; height: 320px; object-fit: contain; background: #000; border-radius: 8px; }}
    .chip {{ display: inline-block; padding: 4px 10px; margin-right: 8px; margin-top: 6px; border-radius: 999px; background: #1d4ed8; font-size: 12px; }}
    .chip.good {{ background: #166534; }}
    .chip.muted {{ background: #475569; }}
    .meta, .paths {{ font-size: 12px; line-height: 1.45; color: #cbd5e1; word-break: break-all; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    td, th {{ border: 1px solid #334155; padding: 8px 10px; text-align: left; }}
    th {{ background: #020617; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Salamander Flip-Invariant Local Match Review</h1>
      <p>这页只看同一 identity 的正样本 pair，目的是检查 `match(a, b)` 和 `match(a, hflip(b))` 的差别。</p>
      <table>
        <tbody>{summary_rows}</tbody>
      </table>
      <p>all_pairs: {html.escape(str(pair_csv))}</p>
      <p>top_review_pairs: {html.escape(str(review_csv))}</p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    html_path.write_text(html_text, encoding="utf-8")
    print(f"[flip_review] html: {html_path}")
    print(f"[flip_review] pairs: {pair_csv}")
    print(f"[flip_review] top_pairs: {review_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
