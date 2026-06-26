#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import os
import shutil
from pathlib import Path

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_preprocessing_cache_closure_v1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build Salamander preprocessing cache closure table and simple review HTML.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=str, default=SALAMANDER_DATASET)
    parser.add_argument("--sample-count", type=int, default=80)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def _load_manifest(path: Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if required_columns:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing columns in {path}: {missing}")
    for column in ["image_id", "dataset", "split", "path", "identity"]:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str)
    return frame


def _resolve_abs(repo_root: Path, rel_path: str) -> Path | None:
    rel_path = str(rel_path or "").strip()
    if not rel_path:
        return None
    absolute = (repo_root / rel_path).resolve()
    return absolute if absolute.exists() else None


def _to_rel(html_dir: Path, absolute_path: Path) -> str:
    return os.path.relpath(os.path.abspath(str(absolute_path)), str(html_dir.resolve()))


def _display_rel(base_dir: Path, target_path: Path) -> str:
    return os.path.relpath(str(target_path.resolve()), str(base_dir.resolve()))


def _stage_asset(*, output_dir: Path, absolute_path: Path | None, view_name: str, image_id: str) -> Path | None:
    if absolute_path is None or not absolute_path.exists():
        return None
    suffix = absolute_path.suffix or ".jpg"
    target = output_dir / "staged_assets" / view_name / f"{image_id}{suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    try:
        os.symlink(absolute_path, target)
    except OSError:
        shutil.copy2(absolute_path, target)
    return target


def _classify_aligned_cache(path_value: str) -> str:
    value = str(path_value or "")
    if not value:
        return "missing"
    if "sam_seg_trainprep_repaired_v1" in value:
        return "repaired_v1"
    if "sam_seg_trainprep_v1" in value:
        return "trainprep_v1"
    if "sam_masked_rgb_v1" in value:
        return "masked_fallback"
    if value.startswith("images/"):
        return "original_fallback"
    return "other"


def _render_summary_table(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "<p>Empty summary.</p>"
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in summary_df.columns)
    rows: list[str] = []
    for _, row in summary_df.iterrows():
        cells = "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in summary_df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def build_review_artifacts(*, repo_root: Path, output_dir: Path, dataset: str, sample_count: int, sample_seed: int) -> dict[str, Path]:
    manifests_root = repo_root / "artifacts" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    original_paths = [
        manifests_root / "v1" / "tables" / "manifest_train_original_only_v1.csv",
        manifests_root / "v1" / "tables" / "manifest_test_original_only_v1.csv",
    ]
    masked_paths = [
        manifests_root / "v1" / "tables" / "manifest_train_sam_masked_rgb_v1.csv",
        manifests_root / "v1" / "tables" / "manifest_test_sam_masked_rgb_v1.csv",
    ]
    aligned_paths = [
        manifests_root / "sam_seg_trainprep_repaired_v1" / "tables" / "manifest_train_sam_trainprep_aligned_best_v1.csv",
        manifests_root / "sam_seg_trainprep_repaired_v1" / "tables" / "manifest_test_sam_trainprep_aligned_best_v1.csv",
    ]

    original_df = pd.concat([_load_manifest(path) for path in original_paths], ignore_index=True)
    masked_df = pd.concat([_load_manifest(path) for path in masked_paths], ignore_index=True)
    aligned_df = pd.concat([_load_manifest(path) for path in aligned_paths], ignore_index=True)

    keys = ["image_id", "dataset", "split"]
    base_df = original_df[original_df["dataset"].eq(dataset)].copy().reset_index(drop=True)
    masked_df = masked_df[masked_df["dataset"].eq(dataset)].copy().reset_index(drop=True)
    aligned_df = aligned_df[aligned_df["dataset"].eq(dataset)].copy().reset_index(drop=True)

    base_df = base_df.rename(
        columns={
            "path": "original_manifest_path",
            "preferred_path_v1": "original_preferred_path_v1",
            "preprocess_variant_v1": "original_variant_v1",
        }
    )
    masked_df = masked_df.rename(
        columns={
            "path": "masked_only_manifest_path",
            "preferred_path_v1": "masked_only_preferred_path_v1",
            "preprocess_variant_v1": "masked_only_variant_v1",
        }
    )
    aligned_df = aligned_df.rename(
        columns={
            "path": "masked_aligned_manifest_path",
            "preferred_path_v1": "masked_aligned_preferred_path_v1",
            "preprocess_variant_v1": "masked_aligned_variant_v1",
        }
    )

    base_keep = keys + ["identity", "original_manifest_path", "original_preferred_path_v1", "original_variant_v1"]
    masked_keep = keys + [
        "masked_only_manifest_path",
        "masked_only_preferred_path_v1",
        "masked_only_variant_v1",
        "sam_masked_rgb_v1_applied",
        "sam_masked_rgb_v1_export_path",
        "sam_masked_rgb_v1_resolved_path_v1",
        "sam_masked_rgb_v1_resolved_variant_v1",
    ]
    aligned_keep = keys + [
        "masked_aligned_manifest_path",
        "masked_aligned_preferred_path_v1",
        "masked_aligned_variant_v1",
        "sam_trainprep_aligned_applied_v1",
        "sam_trainprep_aligned_path_v1",
        "sam_trainprep_aligned_resolved_path_v1",
        "sam_trainprep_masked_resolved_path_v1",
        "manifest_view_name_v1",
        "manifest_view_resolved_v1",
        "manifest_view_applied_v1",
    ]

    merged_df = base_df[base_keep].merge(masked_df[masked_keep], on=keys, how="left").merge(aligned_df[aligned_keep], on=keys, how="left")
    merged_df["identity"] = merged_df["identity"].fillna("").astype(str)
    merged_df["original_path"] = merged_df["original_preferred_path_v1"].fillna(merged_df["original_manifest_path"]).astype(str)
    merged_df["masked_only_path"] = (
        merged_df["masked_only_preferred_path_v1"]
        .fillna(merged_df["sam_masked_rgb_v1_resolved_path_v1"])
        .fillna(merged_df["masked_only_manifest_path"])
        .astype(str)
    )
    merged_df["masked_aligned_path"] = (
        merged_df["masked_aligned_preferred_path_v1"]
        .fillna(merged_df["sam_trainprep_aligned_resolved_path_v1"])
        .fillna(merged_df["masked_aligned_manifest_path"])
        .astype(str)
    )
    merged_df["original_exists"] = merged_df["original_path"].map(lambda value: _resolve_abs(repo_root, value) is not None)
    merged_df["masked_only_exists"] = merged_df["masked_only_path"].map(lambda value: _resolve_abs(repo_root, value) is not None)
    merged_df["masked_aligned_exists"] = merged_df["masked_aligned_path"].map(lambda value: _resolve_abs(repo_root, value) is not None)
    merged_df["masked_only_is_real_cache"] = merged_df["masked_only_path"].astype(str).ne(merged_df["original_path"].astype(str))
    merged_df["masked_aligned_is_real_cache"] = merged_df["masked_aligned_path"].astype(str).ne(merged_df["original_path"].astype(str))
    merged_df["masked_aligned_cache_source"] = merged_df["masked_aligned_path"].map(_classify_aligned_cache)
    merged_df["masked_aligned_matches_masked_only"] = (
        merged_df["masked_aligned_path"].astype(str).eq(merged_df["masked_only_path"].astype(str))
        & merged_df["masked_aligned_path"].astype(str).ne("")
    )
    merged_df = merged_df.sort_values(["split", "identity", "image_id"], ascending=[True, True, True]).reset_index(drop=True)

    closure_csv_path = output_dir / "salamander_preprocessing_cache_closure_v1.csv"
    merged_df.to_csv(closure_csv_path, index=False)

    summary_rows = [
        {"metric": "rows_total", "value": int(len(merged_df))},
        {"metric": "train_rows", "value": int(merged_df["split"].eq("train").sum())},
        {"metric": "test_rows", "value": int(merged_df["split"].eq("test").sum())},
        {"metric": "masked_only_exists", "value": int(merged_df["masked_only_exists"].sum())},
        {"metric": "masked_only_real_cache", "value": int(merged_df["masked_only_is_real_cache"].sum())},
        {"metric": "masked_aligned_exists", "value": int(merged_df["masked_aligned_exists"].sum())},
        {"metric": "masked_aligned_real_cache", "value": int(merged_df["masked_aligned_is_real_cache"].sum())},
        {"metric": "masked_aligned_equals_masked_only", "value": int(merged_df["masked_aligned_matches_masked_only"].sum())},
    ]
    cache_source_counts = (
        merged_df.groupby("masked_aligned_cache_source", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "masked_aligned_cache_source"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "summary_counts.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    cache_source_csv_path = output_dir / "aligned_cache_source_counts.csv"
    cache_source_counts.to_csv(cache_source_csv_path, index=False)

    summary_md_lines = [
        "# Salamander Preprocessing Cache Closure v1",
        "",
        f"- dataset: `{dataset}`",
        f"- rows_total: `{int(len(merged_df))}`",
        f"- train_rows: `{int(merged_df['split'].eq('train').sum())}`",
        f"- test_rows: `{int(merged_df['split'].eq('test').sum())}`",
        f"- masked_only_real_cache: `{int(merged_df['masked_only_is_real_cache'].sum())}`",
        f"- masked_aligned_real_cache: `{int(merged_df['masked_aligned_is_real_cache'].sum())}`",
        "",
        "## Key Notes",
        "",
        "- `original_only` 来自统一 `v1` manifest。",
        "- `masked_only` 当前正式名是 `sam_masked_rgb_v1`。",
        "- `masked_aligned` 当前主入口仍来自 `sam_seg_trainprep_repaired_v1` manifest，但其路径大多指向旧的 `sam_seg_trainprep_v1` 缓存。",
        "- `body_axis_unsigned_rgb_v1` 不在这份三视图 canonical closure 里。",
        "",
        "## Files",
        "",
        f"- closure_csv: `{_display_rel(repo_root, closure_csv_path)}`",
        f"- summary_counts: `{_display_rel(repo_root, summary_csv_path)}`",
        f"- aligned_cache_sources: `{_display_rel(repo_root, cache_source_csv_path)}`",
        "",
    ]
    summary_md_path = output_dir / "summary.md"
    summary_md_path.write_text("\n".join(summary_md_lines), encoding="utf-8")

    sampled_df = merged_df.copy()
    if len(sampled_df) > int(sample_count):
        sampled_df = sampled_df.sample(n=int(sample_count), random_state=int(sample_seed)).copy()
    sampled_df = sampled_df.sort_values(["split", "identity", "image_id"]).reset_index(drop=True)
    sampled_csv_path = output_dir / "sampled_rows.csv"
    sampled_df.to_csv(sampled_csv_path, index=False)

    cards: list[str] = []
    html_dir = output_dir
    for row in sampled_df.itertuples(index=False):
        original_abs = _resolve_abs(repo_root, str(row.original_path))
        masked_abs = _resolve_abs(repo_root, str(row.masked_only_path))
        aligned_abs = _resolve_abs(repo_root, str(row.masked_aligned_path))
        panels: list[str] = []
        for title, path_value, absolute_path, state_note, extra_note in [
            ("original_only", str(row.original_path), original_abs, "exists" if bool(row.original_exists) else "missing", ""),
            (
                "masked_only",
                str(row.masked_only_path),
                masked_abs,
                "exists" if bool(row.masked_only_exists) else "missing",
                "real_cache" if bool(row.masked_only_is_real_cache) else "fallback_to_original",
            ),
            (
                "masked_aligned",
                str(row.masked_aligned_path),
                aligned_abs,
                "exists" if bool(row.masked_aligned_exists) else "missing",
                str(row.masked_aligned_cache_source),
            ),
        ]:
            image_html = "<div class='missing'>missing</div>"
            if absolute_path is not None:
                staged_path = _stage_asset(
                    output_dir=output_dir,
                    absolute_path=absolute_path,
                    view_name=title,
                    image_id=str(row.image_id),
                )
                rel = _to_rel(html_dir, staged_path if staged_path is not None else absolute_path)
                image_html = (
                    f"<a href='{html.escape(rel)}' target='_blank' rel='noopener noreferrer'>"
                    f"<img src='{html.escape(rel)}' loading='lazy' alt='{html.escape(str(row.image_id))} {html.escape(title)}' />"
                    "</a>"
                )
            panel_lines = [
                "<div class='view-card'>",
                f"<div class='view-title'>{html.escape(title)}</div>",
                image_html,
                f"<div class='meta'><strong>state</strong>: {html.escape(state_note)}</div>",
                f"<div class='meta'><strong>note</strong>: {html.escape(extra_note or '-')}</div>",
                f"<div class='meta'><strong>path</strong>: {html.escape(path_value or '-')}</div>",
                "</div>",
            ]
            panels.append("\n".join(panel_lines))
        cards.append(
            "\n".join(
                [
                    "<section class='sample-card'>",
                    (
                        f"<div class='sample-head'><div><strong>{html.escape(str(row.image_id))}</strong></div>"
                        f"<div class='chips'>"
                        f"<span class='chip'>{html.escape(str(row.split))}</span>"
                        f"<span class='chip'>identity {html.escape(str(row.identity) or '-')}</span>"
                        f"<span class='chip'>aligned_source {html.escape(str(row.masked_aligned_cache_source))}</span>"
                        "</div></div>"
                    ),
                    "<div class='view-grid'>",
                    "".join(panels),
                    "</div>",
                    "</section>",
                ]
            )
        )

    summary_counts_html = _render_summary_table(summary_df)
    cache_source_html = _render_summary_table(cache_source_counts)
    html_text = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Salamander Preprocessing Cache Closure v1</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    main {{ max-width: 1800px; margin: 0 auto; padding: 24px; }}
    .hero, .sample-card {{ background: #111827; border: 1px solid #334155; border-radius: 14px; padding: 18px; margin-bottom: 18px; }}
    .hero h1 {{ margin-top: 0; }}
    .chips {{ margin-top: 8px; }}
    .chip {{ display: inline-block; padding: 4px 10px; margin-right: 8px; margin-top: 6px; border-radius: 999px; background: #1d4ed8; font-size: 12px; }}
    .view-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }}
    .view-card {{ background: #020617; border-radius: 12px; padding: 12px; border: 1px solid #334155; }}
    .view-title {{ font-size: 14px; font-weight: 700; margin-bottom: 8px; color: #93c5fd; }}
    img {{ width: 100%; height: 260px; object-fit: contain; background: #000; border-radius: 8px; }}
    .missing {{ height: 260px; display: flex; align-items: center; justify-content: center; background: #1f2937; color: #fca5a5; border-radius: 8px; }}
    .meta {{ font-size: 12px; line-height: 1.45; color: #cbd5e1; word-break: break-all; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #334155; padding: 8px 10px; text-align: left; }}
    th {{ background: #020617; }}
    .sample-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Salamander Preprocessing Cache Closure v1</h1>
      <div class="chips">
        <span class="chip">dataset {html.escape(dataset)}</span>
        <span class="chip">rows {int(len(merged_df))}</span>
        <span class="chip">sample_count {int(len(sampled_df))}</span>
      </div>
      <p>这页只收口三类视图：`original_only`、`masked_only`、`masked_aligned`。`body_axis_unsigned_rgb_v1` 不算进 canonical aligned。</p>
      <h2>Summary Counts</h2>
      {summary_counts_html}
      <h2>Aligned Cache Sources</h2>
      {cache_source_html}
      <p>full table: {html.escape(_display_rel(repo_root, closure_csv_path))}</p>
      <p>sample rows: {html.escape(_display_rel(repo_root, sampled_csv_path))}</p>
    </section>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html_text, encoding="utf-8")

    return {
        "closure_csv_path": closure_csv_path,
        "summary_md_path": summary_md_path,
        "summary_csv_path": summary_csv_path,
        "cache_source_csv_path": cache_source_csv_path,
        "sampled_csv_path": sampled_csv_path,
        "html_path": html_path,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    outputs = build_review_artifacts(
        repo_root=repo_root,
        output_dir=output_dir.resolve(),
        dataset=str(args.dataset),
        sample_count=int(args.sample_count),
        sample_seed=int(args.sample_seed),
    )
    for name, path in outputs.items():
        print(f"[salamander_preprocessing_cache_review] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
