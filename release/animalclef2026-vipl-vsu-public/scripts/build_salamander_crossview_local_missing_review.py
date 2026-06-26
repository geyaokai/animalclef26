#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
from pathlib import Path

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_CURRENT_PREDICTIONS = Path(
    "artifacts/submissions/kaggle_variant_salamander_top10_manual_graph_on_062817_bestpublic_v1/tables/test_predictions_v1.csv"
)
DEFAULT_REVIEWED_LABELS = Path(
    "artifacts/analysis/salamander_test_top10_yes_review_v1/result/merged/"
    "salamander_test_top10_yes_review_v1_merged_union_yes.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_crossview_local_missing_review_v1")
DEFAULT_SOURCES = {
    "aligned_orb": Path("artifacts/analysis/salamander_sam_orb_veto_v1/tables/test_aligned_local_scores_v1.csv"),
    "masked_orb": Path("artifacts/analysis/salamander_sam_orb_veto_v1/tables/test_masked_local_scores_v1.csv"),
    "aliked_lightglue": Path("artifacts/analysis/salamander_aliked_lightglue_probe_20260330/tables/test_local_match_scores_v1.csv"),
    "dualview_yellow": Path("artifacts/analysis/salamander_dualview_v5_local_graph_v1/tables/test_pair_local_scores_v1.csv"),
}


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _canonical_pair(left: object, right: object) -> tuple[str, str, str]:
    values = sorted([str(left), str(right)], key=lambda value: int(value) if value.isdigit() else value)
    return values[0], values[1], f"{values[0]}|{values[1]}"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _stage_image(repo_root: Path, output_dir: Path, image_id: str, rel_path: str) -> str:
    src = (repo_root / str(rel_path)).resolve()
    if not src.exists():
        return ""
    suffix = src.suffix.lower() or ".jpg"
    dst = output_dir / "staged_assets" / f"{image_id}{suffix}"
    _copy_or_link(src, dst)
    return os.path.relpath(dst, output_dir).replace("\\", "/")


def _load_predictions(path: Path, dataset: str) -> pd.DataFrame:
    pred_df = pd.read_csv(path, low_memory=False)
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    pred_df = pred_df[pred_df["dataset"].eq(str(dataset))].copy().reset_index(drop=True)
    if pred_df.empty:
        raise ValueError(f"No {dataset} rows found in {path}")
    pred_df["pred_cluster_id"] = pd.to_numeric(pred_df["pred_cluster_id"], errors="raise").astype(int)
    for column in ["path", "original_rgb_path_v1", "preferred_path_v1", "sam_trainprep_aligned_path_v1", "orientation", "date"]:
        if column not in pred_df.columns:
            pred_df[column] = ""
        pred_df[column] = pred_df[column].fillna("").astype(str)
    pred_df["review_image_path"] = pred_df["original_rgb_path_v1"]
    pred_df.loc[pred_df["review_image_path"].eq(""), "review_image_path"] = pred_df.loc[
        pred_df["review_image_path"].eq(""), "path"
    ]
    return pred_df


def _load_reviewed_keys(paths: list[Path]) -> set[str]:
    keys = set()
    for path in paths:
        if not path.exists():
            continue
        labels = pd.read_csv(path, low_memory=False)
        if not {"image_id", "neighbor_image_id"}.issubset(labels.columns):
            continue
        for row in labels.itertuples(index=False):
            _left, _right, key = _canonical_pair(getattr(row, "image_id"), getattr(row, "neighbor_image_id"))
            keys.add(key)
    return keys


def _source_score_columns(source_name: str) -> dict[str, str]:
    if source_name == "aligned_orb":
        return {
            "score": "aligned_local_score",
            "raw": "aligned_local_raw_score",
            "inliers": "aligned_inliers",
            "good": "aligned_good_matches",
        }
    if source_name == "masked_orb":
        return {
            "score": "masked_local_score",
            "raw": "masked_local_raw_score",
            "inliers": "masked_inliers",
            "good": "masked_good_matches",
        }
    if source_name == "aliked_lightglue":
        return {
            "score": "local_score",
            "raw": "local_raw_score",
            "inliers": "inliers",
            "good": "good_matches",
            "global": "global_score",
        }
    if source_name == "dualview_yellow":
        return {
            "score": "yellow_roi_local_score",
            "raw": "yellow_roi_local_raw_score",
            "inliers": "yellow_roi_inliers",
            "good": "yellow_roi_good_matches",
            "global": "route_global_score",
            "local_only": "local_only_score_v1",
        }
    raise KeyError(source_name)


def _aggregate_sources(
    *,
    source_paths: dict[str, Path],
    valid_image_ids: set[str],
) -> pd.DataFrame:
    by_key: dict[str, dict[str, object]] = {}
    for source_name, source_path in source_paths.items():
        if not source_path.exists():
            print(f"[warn] missing source: {source_name}: {source_path}")
            continue
        df = pd.read_csv(source_path, low_memory=False)
        if "dataset" in df.columns:
            df = df[df["dataset"].astype(str).eq(SALAMANDER_DATASET)].copy()
        if not {"image_id", "neighbor_image_id"}.issubset(df.columns):
            print(f"[warn] skip source without pair columns: {source_name}: {source_path}")
            continue
        df["image_id"] = df["image_id"].astype(str)
        df["neighbor_image_id"] = df["neighbor_image_id"].astype(str)
        df = df[df["image_id"].isin(valid_image_ids) & df["neighbor_image_id"].isin(valid_image_ids)].copy()
        columns = _source_score_columns(source_name)
        for row in df.itertuples(index=False):
            left, right, key = _canonical_pair(getattr(row, "image_id"), getattr(row, "neighbor_image_id"))
            item = by_key.setdefault(
                key,
                {
                    "pair_key": key,
                    "image_id": left,
                    "neighbor_image_id": right,
                    "source_names": set(),
                    "global_score_max": 0.0,
                    "local_score_max": 0.0,
                    "evidence_score": 0.0,
                    "max_inliers": 0,
                    "max_good_matches": 0,
                },
            )
            item["source_names"].add(source_name)
            score = _safe_float(getattr(row, columns["score"], 0.0))
            raw = _safe_float(getattr(row, columns.get("raw", ""), 0.0))
            inliers = int(round(_safe_float(getattr(row, columns.get("inliers", ""), 0), 0.0)))
            good = int(round(_safe_float(getattr(row, columns.get("good", ""), 0), 0.0)))
            global_score = _safe_float(getattr(row, columns.get("global", ""), 0.0))
            local_only = _safe_float(getattr(row, columns.get("local_only", ""), score))
            item[f"{source_name}_score"] = max(_safe_float(item.get(f"{source_name}_score", 0.0)), score)
            item[f"{source_name}_raw"] = max(_safe_float(item.get(f"{source_name}_raw", 0.0)), raw)
            item[f"{source_name}_inliers"] = max(int(item.get(f"{source_name}_inliers", 0)), inliers)
            item[f"{source_name}_good_matches"] = max(int(item.get(f"{source_name}_good_matches", 0)), good)
            item["global_score_max"] = max(_safe_float(item.get("global_score_max", 0.0)), global_score)
            item["local_score_max"] = max(_safe_float(item.get("local_score_max", 0.0)), score)
            item["max_inliers"] = max(int(item.get("max_inliers", 0)), inliers)
            item["max_good_matches"] = max(int(item.get("max_good_matches", 0)), good)
            # ALIKED is permissive, so require support from geometry or yellow/local-only before it dominates.
            evidence = max(score, local_only, min(1.0, raw * 3.0), min(1.0, inliers / 40.0))
            item["evidence_score"] = max(_safe_float(item.get("evidence_score", 0.0)), evidence)
    rows: list[dict[str, object]] = []
    for item in by_key.values():
        item = dict(item)
        item["source_names"] = "|".join(sorted(item["source_names"]))
        rows.append(item)
    return pd.DataFrame(rows)


def _select_candidates(
    *,
    pair_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    reviewed_keys: set[str],
    max_pairs: int,
    per_image_limit: int,
    min_evidence_score: float,
    require_orientation_change: bool,
) -> pd.DataFrame:
    cluster = dict(zip(pred_df["image_id"], pred_df["pred_cluster_id"], strict=True))
    meta = pred_df.set_index("image_id")
    df = pair_df.copy()
    df = df[~df["pair_key"].astype(str).isin(reviewed_keys)].copy()
    df["left_cluster"] = df["image_id"].map(cluster).astype(int)
    df["right_cluster"] = df["neighbor_image_id"].map(cluster).astype(int)
    df = df[df["left_cluster"].ne(df["right_cluster"])].copy()
    df["left_orientation"] = df["image_id"].map(meta["orientation"]).fillna("").astype(str)
    df["right_orientation"] = df["neighbor_image_id"].map(meta["orientation"]).fillna("").astype(str)
    df["orientation_changed"] = (
        df["left_orientation"].ne("")
        & df["right_orientation"].ne("")
        & df["left_orientation"].ne(df["right_orientation"])
    )
    if bool(require_orientation_change):
        df = df[df["orientation_changed"]].copy()
    df["left_date"] = df["image_id"].map(meta["date"]).fillna("").astype(str)
    df["right_date"] = df["neighbor_image_id"].map(meta["date"]).fillna("").astype(str)
    df = df[pd.to_numeric(df["evidence_score"], errors="coerce").fillna(0.0).ge(float(min_evidence_score))].copy()
    df["review_priority"] = (
        pd.to_numeric(df["evidence_score"], errors="coerce").fillna(0.0) * 1000.0
        + pd.to_numeric(df["local_score_max"], errors="coerce").fillna(0.0) * 100.0
        + pd.to_numeric(df["max_inliers"], errors="coerce").fillna(0.0)
        + df["orientation_changed"].astype(int) * 25.0
    )
    df = df.sort_values(
        ["review_priority", "evidence_score", "local_score_max", "max_inliers", "pair_key"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)

    counts: dict[str, int] = {}
    selected_rows = []
    for row in df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        if counts.get(left, 0) >= int(per_image_limit) or counts.get(right, 0) >= int(per_image_limit):
            continue
        selected_rows.append(row._asdict())
        counts[left] = counts.get(left, 0) + 1
        counts[right] = counts.get(right, 0) + 1
        if len(selected_rows) >= int(max_pairs):
            break
    return pd.DataFrame(selected_rows)


def _write_html(
    *,
    output_dir: Path,
    selected_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    repo_root: Path,
    session_name: str,
    pairs_per_page: int,
) -> None:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    meta = pred_df.set_index("image_id")
    image_rel_by_id = {}
    for image_id, row in meta.iterrows():
        image_rel_by_id[str(image_id)] = _stage_image(repo_root, output_dir, str(image_id), str(row["review_image_path"]))

    payload = selected_df.fillna("").to_dict(orient="records")
    for row in payload:
        left = str(row["image_id"])
        right = str(row["neighbor_image_id"])
        row["left_img"] = image_rel_by_id.get(left, "")
        row["right_img"] = image_rel_by_id.get(right, "")

    css = """
    :root{--bg:#f4efe4;--ink:#1f1b16;--muted:#71695f;--card:#fffaf1;--line:#d7c9b7;--yes:#147447;--warn:#b77425}
    body{margin:0;background:radial-gradient(circle at top left,#fff8df,var(--bg) 42%,#e8ddca);color:var(--ink);font-family:Georgia,'Times New Roman',serif}
    header{position:sticky;top:0;z-index:10;background:rgba(244,239,228,.97);border-bottom:1px solid var(--line);padding:14px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    h1{font-size:23px;margin:0} a{color:var(--ink)}
    .stats{color:var(--muted);font-size:14px}
    button,.button{border:1px solid var(--ink);background:var(--ink);color:white;border-radius:999px;padding:8px 12px;cursor:pointer;text-decoration:none}
    main{padding:18px 20px 40px}
    .page-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px}
    .page-link{display:block;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px;text-decoration:none;color:var(--ink)}
    .pair{background:var(--card);border:1px solid var(--line);border-radius:18px;margin:0 0 16px;padding:12px;box-shadow:0 8px 22px rgba(50,42,31,.07)}
    .pair.selected{border-color:var(--yes);box-shadow:0 0 0 3px rgba(20,116,71,.16)}
    .pair-head{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:10px}
    .yes-line{font-weight:700;color:var(--yes);cursor:pointer}
    .yes-box{width:22px;height:22px;accent-color:var(--yes);vertical-align:middle}
    .imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .img-card{background:#17130f;border-radius:12px;min-height:260px;display:flex;align-items:center;justify-content:center;overflow:hidden}
    img{max-width:100%;max-height:360px;object-fit:contain}
    .meta{font-size:13px;color:var(--muted);line-height:1.45;overflow-wrap:anywhere;margin-top:8px}
    .score{font-weight:700;color:var(--warn)}
    @media(max-width:760px){.imgs{grid-template-columns:1fr}.img-card{min-height:220px}}
    """
    script = f"""
    const sessionName = {json.dumps(session_name)};
    const allPairs = {json.dumps(payload, ensure_ascii=False)};
    const storageKey = sessionName + ':yes_pair_keys';
    function loadYesSet(){{try{{return new Set(JSON.parse(localStorage.getItem(storageKey)||'[]'));}}catch(e){{return new Set();}}}}
    function saveYesSet(s){{localStorage.setItem(storageKey, JSON.stringify([...s].sort()));}}
    function updateStats(){{const s=loadYesSet(); const el=document.getElementById('stats'); if(el) el.textContent=`${{s.size}} YES selected; unselected pairs are ignored, not used as strong NO`;}}
    function syncBoxes(){{const s=loadYesSet(); document.querySelectorAll('.yes-box').forEach(box=>{{box.checked=s.has(box.dataset.pairKey); box.closest('.pair').classList.toggle('selected', box.checked);}}); updateStats();}}
    document.addEventListener('change', e=>{{if(!e.target.classList.contains('yes-box')) return; const s=loadYesSet(); if(e.target.checked) s.add(e.target.dataset.pairKey); else s.delete(e.target.dataset.pairKey); saveYesSet(s); syncBoxes();}});
    function downloadYesCSV(){{const s=loadYesSet(); const rows=allPairs.filter(r=>s.has(r.pair_key)); const cols=['pair_key','image_id','neighbor_image_id','label','left_cluster','right_cluster','review_priority','evidence_score','local_score_max','max_inliers','max_good_matches','source_names','aligned_orb_score','masked_orb_score','aliked_lightglue_score','dualview_yellow_score','dualview_yellow_inliers','orientation_changed']; let csv=cols.join(',')+'\\n'; for(const r of rows){{r.label='yes'; csv += cols.map(c=>`"${{String(r[c]??'').replaceAll('"','""')}}"`).join(',')+'\\n';}} const blob=new Blob([csv],{{type:'text/csv;charset=utf-8'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=sessionName+'_yes.csv'; a.click(); URL.revokeObjectURL(a.href);}}
    function downloadCacheJSON(){{const s=loadYesSet(); const rows=allPairs.filter(r=>s.has(r.pair_key)).map(r=>Object.assign({{label:'yes'}}, r)); const blob=new Blob([JSON.stringify({{session_name:sessionName, yes_count:rows.length, rows}}, null, 2)],{{type:'application/json'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=sessionName+'_yes.json'; a.click(); URL.revokeObjectURL(a.href);}}
    window.addEventListener('DOMContentLoaded', syncBoxes);
    """

    def render_pair(row: dict[str, object], page_prefix: str) -> str:
        left = str(row["image_id"])
        right = str(row["neighbor_image_id"])
        left_img = html.escape(os.path.relpath(output_dir / str(row["left_img"]), pages_dir).replace("\\", "/"))
        right_img = html.escape(os.path.relpath(output_dir / str(row["right_img"]), pages_dir).replace("\\", "/"))
        pair_key = html.escape(str(row["pair_key"]))
        return f"""
        <section class="pair" id="pair-{pair_key}">
          <div class="pair-head">
            <label class="yes-line"><input class="yes-box" type="checkbox" data-pair-key="{pair_key}"> YES same identity</label>
            <div class="score">priority {float(row.get('review_priority', 0.0)):.2f} | evidence {float(row.get('evidence_score', 0.0)):.3f} | local {float(row.get('local_score_max', 0.0)):.3f} | inliers {int(float(row.get('max_inliers', 0) or 0))}</div>
          </div>
          <div class="imgs">
            <div><div class="img-card"><img src="{left_img}" loading="lazy"></div><div class="meta">left {html.escape(left)} | cluster {html.escape(str(row.get('left_cluster','')))} | orient {html.escape(str(row.get('left_orientation','')))} | date {html.escape(str(row.get('left_date','')))}</div></div>
            <div><div class="img-card"><img src="{right_img}" loading="lazy"></div><div class="meta">right {html.escape(right)} | cluster {html.escape(str(row.get('right_cluster','')))} | orient {html.escape(str(row.get('right_orientation','')))} | date {html.escape(str(row.get('right_date','')))}</div></div>
          </div>
          <div class="meta">pair {pair_key} | sources {html.escape(str(row.get('source_names','')))} | aligned {float(row.get('aligned_orb_score', 0) or 0):.3f} | masked {float(row.get('masked_orb_score', 0) or 0):.3f} | aliked {float(row.get('aliked_lightglue_score', 0) or 0):.3f} | yellow {float(row.get('dualview_yellow_score', 0) or 0):.3f} | global_max {float(row.get('global_score_max', 0) or 0):.3f}</div>
        </section>
        """

    page_size = max(1, int(pairs_per_page))
    chunks = [payload[start : start + page_size] for start in range(0, len(payload), page_size)]
    page_links = []
    for idx, rows in enumerate(chunks, start=1):
        page_name = f"page_{idx:03d}.html"
        page_links.append((page_name, len(rows), rows[0]["pair_key"] if rows else ""))
        body = "\n".join(render_pair(row, "../") for row in rows)
        page_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(session_name)} page {idx}</title><style>{css}</style></head>
        <body><header><h1>{html.escape(session_name)} page {idx}/{len(chunks)}</h1><a class="button" href="../index.html">Index</a><button onclick="downloadYesCSV()">Export YES CSV</button><button onclick="downloadCacheJSON()">Export YES JSON</button><span class="stats" id="stats"></span></header><main>{body}</main><script>{script}</script></body></html>"""
        (pages_dir / page_name).write_text(page_html, encoding="utf-8")

    links_html = "\n".join(
        f'<a class="page-link" href="pages/{html.escape(name)}">Page {idx:03d}<br><span class="stats">{count} pairs, first {html.escape(str(first))}</span></a>'
        for idx, (name, count, first) in enumerate(page_links, start=1)
    )
    index_html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(session_name)}</title><style>{css}</style></head>
    <body><header><h1>{html.escape(session_name)}</h1><button onclick="downloadYesCSV()">Export YES CSV</button><button onclick="downloadCacheJSON()">Export YES JSON</button><span class="stats" id="stats"></span></header>
    <main><p class="stats">Only review likely missed cross-cluster local matches. Click YES for same identity; unclicked pairs are ignored.</p><div class="page-grid">{links_html}</div></main><script>{script}</script></body></html>"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a focused Salamander cross-view local missing-merge review HTML.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--current-predictions", type=Path, default=DEFAULT_CURRENT_PREDICTIONS)
    parser.add_argument("--reviewed-labels", type=Path, nargs="*", default=[DEFAULT_REVIEWED_LABELS])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=360)
    parser.add_argument("--per-image-limit", type=int, default=3)
    parser.add_argument("--pairs-per-page", type=int, default=60)
    parser.add_argument("--min-evidence-score", type=float, default=0.10)
    parser.add_argument("--require-orientation-change", action="store_true")
    parser.add_argument("--session-name", type=str, default="salamander_crossview_local_missing_review_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = _resolve(repo_root, args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = _resolve(repo_root, args.current_predictions).resolve()
    reviewed_paths = [_resolve(repo_root, path).resolve() for path in args.reviewed_labels]
    pred_df = _load_predictions(predictions_path, SALAMANDER_DATASET)
    reviewed_keys = _load_reviewed_keys(reviewed_paths)
    source_paths = {name: _resolve(repo_root, path).resolve() for name, path in DEFAULT_SOURCES.items()}

    pair_df = _aggregate_sources(source_paths=source_paths, valid_image_ids=set(pred_df["image_id"]))
    selected_df = _select_candidates(
        pair_df=pair_df,
        pred_df=pred_df,
        reviewed_keys=reviewed_keys,
        max_pairs=int(args.max_pairs),
        per_image_limit=int(args.per_image_limit),
        min_evidence_score=float(args.min_evidence_score),
        require_orientation_change=bool(args.require_orientation_change),
    )
    pair_df.to_csv(tables_dir / "all_local_candidate_pairs_v1.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    selected_df.to_csv(tables_dir / "selected_crossview_local_missing_pairs_v1.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    _write_html(
        output_dir=output_dir,
        selected_df=selected_df,
        pred_df=pred_df,
        repo_root=repo_root,
        session_name=str(args.session_name),
        pairs_per_page=int(args.pairs_per_page),
    )

    summary = {
        "session_name": str(args.session_name),
        "predictions_path": str(predictions_path),
        "reviewed_label_paths": [str(path) for path in reviewed_paths],
        "output_dir": str(output_dir),
        "source_paths": {name: str(path) for name, path in source_paths.items()},
        "salamander_images": int(len(pred_df)),
        "current_clusters": int(pred_df["pred_cluster_id"].nunique()),
        "previous_reviewed_pairs": int(len(reviewed_keys)),
        "all_source_pairs": int(len(pair_df)),
        "selected_pairs": int(len(selected_df)),
        "max_pairs": int(args.max_pairs),
        "per_image_limit": int(args.per_image_limit),
        "min_evidence_score": float(args.min_evidence_score),
        "require_orientation_change": bool(args.require_orientation_change),
        "index_html": str(output_dir / "index.html"),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Salamander Cross-View Local Missing Review",
        "",
        f"- Current predictions: `{predictions_path}`",
        f"- Previous reviewed pairs excluded: `{len(reviewed_keys)}`",
        f"- Require orientation change: `{bool(args.require_orientation_change)}`",
        f"- All local source pairs: `{len(pair_df)}`",
        f"- Selected pairs: `{len(selected_df)}`",
        f"- Per image limit: `{int(args.per_image_limit)}`",
        f"- HTML: `{output_dir / 'index.html'}`",
        "",
        "Policy: only click/export `YES`; unclicked pairs are ignored and are not treated as strong no.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
