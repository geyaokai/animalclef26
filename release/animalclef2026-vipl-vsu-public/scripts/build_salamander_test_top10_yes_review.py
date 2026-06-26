#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
from pathlib import Path

import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_PAIR_FEATURES = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_yellow_v1/tables/test_pair_features_v1.csv")
DEFAULT_PREDICTIONS = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionxgb_yellow_v1/tables/test_predictions_v1.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_test_top10_yes_review_v1")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a simple Salamander test top10 YES-only pair review HTML.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--pair-features", type=Path, default=DEFAULT_PAIR_FEATURES)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=str, default=SALAMANDER_DATASET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--queries-per-page", type=int, default=25)
    parser.add_argument("--score-column", type=str, default="xgb_same_identity_prob")
    parser.add_argument("--session-name", type=str, default="salamander_test_top10_yes_review_v1")
    return parser.parse_args()


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _canonical_pair(left: object, right: object) -> tuple[str, str, str]:
    left_text = str(left)
    right_text = str(right)
    ordered = sorted([left_text, right_text])
    return ordered[0], ordered[1], f"{ordered[0]}|{ordered[1]}"


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
    return os.path.relpath(str(dst), str(output_dir))


def _load_prediction_metadata(predictions_path: Path, dataset: str) -> pd.DataFrame:
    pred_df = pd.read_csv(predictions_path, low_memory=False)
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    pred_df = pred_df[pred_df["dataset"].eq(str(dataset))].copy().reset_index(drop=True)
    if pred_df.empty:
        raise ValueError(f"No {dataset} rows found in {predictions_path}")
    if "original_rgb_path_v1" not in pred_df.columns:
        pred_df["original_rgb_path_v1"] = pred_df.get("path", "")
    for column in ["path", "original_rgb_path_v1", "cluster_label"]:
        if column not in pred_df.columns:
            pred_df[column] = ""
        pred_df[column] = pred_df[column].fillna("").astype(str)
    pred_df["pred_cluster_id"] = pd.to_numeric(pred_df["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    return pred_df


def _build_symmetric_topk(pair_df: pd.DataFrame, pred_df: pd.DataFrame, *, top_k: int, score_column: str) -> pd.DataFrame:
    if score_column not in pair_df.columns:
        raise KeyError(f"{score_column} not found in pair features.")
    required = {"image_id", "neighbor_image_id"}
    missing = required - set(pair_df.columns)
    if missing:
        raise KeyError(f"Pair features missing required columns: {sorted(missing)}")

    pair_df = pair_df.copy()
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    pair_df[score_column] = pd.to_numeric(pair_df[score_column], errors="coerce").fillna(0.0)
    for column in ["route_global_score", "fusion_global_score", "student_global_score", "local_score", "yellow_roi_local_score"]:
        if column in pair_df.columns:
            pair_df[column] = pd.to_numeric(pair_df[column], errors="coerce").fillna(0.0)

    cluster_lookup = dict(zip(pred_df["image_id"], pred_df["pred_cluster_id"], strict=False))
    rows: list[dict[str, object]] = []
    for row in pair_df.itertuples(index=False):
        left = str(row.image_id)
        right = str(row.neighbor_image_id)
        left_c, right_c, pair_key = _canonical_pair(left, right)
        canonical_left_cluster = int(cluster_lookup.get(left_c, -1))
        canonical_right_cluster = int(cluster_lookup.get(right_c, -1))
        for query_id, candidate_id in [(left, right), (right, left)]:
            query_cluster = int(cluster_lookup.get(query_id, -1))
            candidate_cluster = int(cluster_lookup.get(candidate_id, -1))
            rows.append(
                {
                    "query_image_id": query_id,
                    "candidate_image_id": candidate_id,
                    "pair_key": pair_key,
                    "canonical_image_id": left_c,
                    "canonical_neighbor_image_id": right_c,
                    "score": round(float(getattr(row, score_column)), 6),
                    "score_column": score_column,
                    "same_pred_cluster": bool(query_cluster == candidate_cluster and query_cluster >= 0),
                    "query_cluster_id": query_cluster,
                    "candidate_cluster_id": candidate_cluster,
                    "canonical_left_cluster_id": canonical_left_cluster,
                    "canonical_right_cluster_id": canonical_right_cluster,
                    "route_global_score": round(float(getattr(row, "route_global_score", 0.0)), 6),
                    "fusion_global_score": round(float(getattr(row, "fusion_global_score", 0.0)), 6),
                    "student_global_score": round(float(getattr(row, "student_global_score", 0.0)), 6),
                    "local_score": round(float(getattr(row, "local_score", 0.0)), 6),
                    "yellow_roi_local_score": round(float(getattr(row, "yellow_roi_local_score", 0.0)), 6),
                }
            )
    neighbor_df = pd.DataFrame(rows)
    neighbor_df = neighbor_df.sort_values(
        ["query_image_id", "score", "local_score", "candidate_image_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    neighbor_df["rank"] = neighbor_df.groupby("query_image_id").cumcount() + 1
    neighbor_df = neighbor_df[neighbor_df["rank"].le(int(top_k))].copy().reset_index(drop=True)
    return neighbor_df


def _write_html(
    *,
    output_dir: Path,
    pred_df: pd.DataFrame,
    review_df: pd.DataFrame,
    image_rel_lookup: dict[str, str],
    session_name: str,
    dataset: str,
    top_k: int,
    queries_per_page: int,
) -> Path:
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    candidate_by_query = {query_id: group.copy() for query_id, group in review_df.groupby("query_image_id", sort=False)}
    cluster_lookup = dict(zip(pred_df["image_id"], pred_df["pred_cluster_id"], strict=False))

    payload_columns = [
        "pair_key",
        "canonical_image_id",
        "canonical_neighbor_image_id",
        "query_image_id",
        "candidate_image_id",
        "rank",
        "score",
        "score_column",
        "same_pred_cluster",
        "query_cluster_id",
        "candidate_cluster_id",
        "canonical_left_cluster_id",
        "canonical_right_cluster_id",
        "route_global_score",
        "fusion_global_score",
        "student_global_score",
        "local_score",
        "yellow_roi_local_score",
    ]
    all_pairs_payload = review_df[payload_columns].to_dict(orient="records")
    query_ids = pred_df["image_id"].astype(str).tolist()
    page_size = max(1, int(queries_per_page))
    page_chunks = [query_ids[start : start + page_size] for start in range(0, len(query_ids), page_size)]

    base_css = """
    :root {{
      --bg: #f4efe4;
      --ink: #1f1b16;
      --muted: #70685e;
      --card: #fffaf1;
      --line: #d6cab8;
      --yes: #1e7b4d;
      --cross: #b9792b;
    }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }}
    header {{ position: sticky; top: 0; z-index: 20; padding: 14px 22px; background: rgba(244,239,228,.96); border-bottom: 1px solid var(--line); display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
    h1 {{ margin: 0; font-size: 24px; }}
    h2 {{ margin: 0 0 6px; font-size: 19px; }}
    a {{ color: var(--ink); }}
    button, .button {{ border: 1px solid var(--ink); background: var(--ink); color: white; padding: 8px 12px; border-radius: 999px; cursor: pointer; text-decoration: none; display: inline-block; }}
    input[type="search"] {{ padding: 8px 10px; min-width: 220px; border: 1px solid var(--line); border-radius: 999px; background: white; }}
    .stats {{ color: var(--muted); font-size: 14px; }}
    .page-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; margin: 22px; }}
    .page-link {{ padding: 14px; border: 1px solid var(--line); border-radius: 14px; background: var(--card); text-decoration: none; }}
    .query-card {{ display: grid; grid-template-columns: 210px 1fr; gap: 16px; margin: 18px 22px; padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: var(--card); box-shadow: 0 8px 22px rgba(50, 42, 31, .07); }}
    .sticky {{ position: sticky; top: 82px; }}
    .query-side img {{ width: 190px; height: 190px; object-fit: contain; background: #16130f; border-radius: 12px; }}
    .candidate-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(154px, 1fr)); gap: 12px; }}
    .candidate {{ position: relative; padding: 8px; border: 1px solid var(--line); border-radius: 14px; background: white; }}
    .candidate.cross-cluster {{ border-color: rgba(185, 121, 43, .55); }}
    .candidate.selected {{ border-color: var(--yes); box-shadow: 0 0 0 3px rgba(30, 123, 77, .18); }}
    .candidate img {{ width: 100%; height: 138px; object-fit: contain; background: #16130f; border-radius: 10px; }}
    .yes-line {{ display: flex; align-items: center; gap: 7px; margin-bottom: 6px; font-weight: 700; color: var(--yes); cursor: pointer; }}
    .yes-box {{ width: 20px; height: 20px; accent-color: var(--yes); }}
    .meta {{ color: var(--muted); font-size: 12px; line-height: 1.35; overflow-wrap: anywhere; }}
    .hidden {{ display: none; }}
    @media (max-width: 760px) {{
      .query-card {{ grid-template-columns: 1fr; }}
      .sticky {{ position: static; }}
      .query-side img {{ width: 100%; height: 220px; }}
    }}
    """
    base_css = base_css.replace("{{", "{").replace("}}", "}")

    def _download_script(pairs_payload: list[dict[str, object]]) -> str:
        return f"""
    const sessionName = {json.dumps(str(session_name), ensure_ascii=False)};
    const dataset = {json.dumps(str(dataset), ensure_ascii=False)};
    const pairs = {json.dumps(pairs_payload, ensure_ascii=False)};
    const storageKey = sessionName + ':yes_pair_keys';

    function loadYesSet() {{
      try {{
        return new Set(JSON.parse(localStorage.getItem(storageKey) || '[]'));
      }} catch (err) {{
        return new Set();
      }}
    }}

    function saveYesSet(yesSet) {{
      localStorage.setItem(storageKey, JSON.stringify([...yesSet].sort()));
    }}

    function updateStats() {{
      const stats = document.getElementById('stats');
      if (!stats) return;
      const yesSet = loadYesSet();
      const uniquePairs = new Set(pairs.map(row => row.pair_key));
      stats.textContent = `${{yesSet.size}} YES selected globally; this page/export covers ${{uniquePairs.size}} unique pairs; unselected exports as NO`;
    }}

    function syncBoxes() {{
      const yesSet = loadYesSet();
      document.querySelectorAll('.yes-box').forEach(box => {{
        const pairKey = box.dataset.pairKey;
        box.checked = yesSet.has(pairKey);
        box.closest('.candidate').classList.toggle('selected', box.checked);
      }});
      updateStats();
    }}

    document.querySelectorAll('.yes-box').forEach(box => {{
      box.addEventListener('change', () => {{
        const yesSet = loadYesSet();
        const pairKey = box.dataset.pairKey;
        if (box.checked) {{
          yesSet.add(pairKey);
        }} else {{
          yesSet.delete(pairKey);
        }}
        saveYesSet(yesSet);
        syncBoxes();
      }});
    }});

    const queryFilter = document.getElementById('query-filter');
    if (queryFilter) {{
      queryFilter.addEventListener('input', event => {{
        const needle = event.target.value.trim();
        document.querySelectorAll('.query-card').forEach(card => {{
          card.classList.toggle('hidden', needle && !card.id.includes(needle));
        }});
      }});
    }}

    function buildJudgments() {{
      const yesSet = loadYesSet();
      const byPair = new Map();
      for (const row of pairs) {{
        if (!byPair.has(row.pair_key)) {{
          byPair.set(row.pair_key, row);
        }}
      }}
      const judgments = [];
      let index = 1;
      for (const [pairKey, row] of [...byPair.entries()].sort()) {{
        const label = yesSet.has(pairKey) ? 'yes' : 'no';
        judgments.push({{
          judgment_id: `${{sessionName}}_${{String(index).padStart(6, '0')}}`,
          dataset,
          candidate_type: 'test_top10',
          candidate_key: String(row.query_image_id),
          pair_key: pairKey,
          image_id: String(row.canonical_image_id),
          neighbor_image_id: String(row.canonical_neighbor_image_id),
          base_cluster_left: Number(row.canonical_left_cluster_id),
          base_cluster_right: Number(row.canonical_right_cluster_id),
          xgb_same_identity_prob: Number(row.score),
          ambiguity_score: 0,
          label,
          note: `source=${{row.score_column}};top10_query=${{row.query_image_id}};rank=${{row.rank}};default_no_if_unselected`
        }});
        index += 1;
      }}
      return judgments;
    }}

    function downloadText(filename, text, mime) {{
      const blob = new Blob([text], {{ type: mime }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    const exportJson = document.getElementById('export-json');
    if (exportJson) {{
      exportJson.addEventListener('click', () => {{
        const payload = {{
          session_name: sessionName,
          pair_judgments: buildJudgments()
        }};
        downloadText(sessionName + '.json', JSON.stringify(payload, null, 2), 'application/json');
      }});
    }}

    const exportCsv = document.getElementById('export-csv');
    if (exportCsv) {{
      exportCsv.addEventListener('click', () => {{
        const judgments = buildJudgments();
        const columns = ['judgment_id','dataset','candidate_type','candidate_key','pair_key','image_id','neighbor_image_id','base_cluster_left','base_cluster_right','xgb_same_identity_prob','ambiguity_score','label','note'];
        const lines = [columns.join(',')];
        for (const row of judgments) {{
          lines.push(columns.map(column => JSON.stringify(row[column] ?? '')).join(','));
        }}
        downloadText(sessionName + '.csv', lines.join('\\n'), 'text/csv');
      }});
    }}

    syncBoxes();
        """

    page_infos: list[dict[str, object]] = []
    for page_index, page_query_ids in enumerate(page_chunks, start=1):
        cards: list[str] = []
        page_review_df = review_df[review_df["query_image_id"].astype(str).isin(page_query_ids)].copy()
        for query_id in page_query_ids:
            query_src = image_rel_lookup.get(str(query_id), "")
            query_src = os.path.relpath(str(output_dir / query_src), str(pages_dir)) if query_src else ""
            query_cluster = int(cluster_lookup.get(str(query_id), -1))
            group = candidate_by_query.get(str(query_id), pd.DataFrame())
            tiles: list[str] = []
            for row in group.itertuples(index=False):
                candidate_id = str(row.candidate_image_id)
                candidate_src = image_rel_lookup.get(candidate_id, "")
                candidate_src = os.path.relpath(str(output_dir / candidate_src), str(pages_dir)) if candidate_src else ""
                pair_key = str(row.pair_key)
                same_cluster_class = "same-cluster" if bool(row.same_pred_cluster) else "cross-cluster"
                tiles.append(
                    f"""
                    <article class="candidate {same_cluster_class}" data-pair-key="{html.escape(pair_key)}">
                      <label class="yes-line">
                        <input type="checkbox" class="yes-box" data-pair-key="{html.escape(pair_key)}" />
                        <span>YES</span>
                      </label>
                      <a href="{html.escape(candidate_src)}" target="_blank"><img src="{html.escape(candidate_src)}" loading="lazy" /></a>
                      <div class="meta">rank {int(row.rank)} | id {html.escape(candidate_id)} | cluster {int(row.candidate_cluster_id)}</div>
                      <div class="meta">score {float(row.score):.4f} | local {float(row.local_score):.4f} | yellow {float(row.yellow_roi_local_score):.4f}</div>
                      <div class="meta">{'same cluster' if bool(row.same_pred_cluster) else 'cross cluster'} | pair {html.escape(pair_key)}</div>
                    </article>
                    """
                )
            cards.append(
                f"""
                <section class="query-card" id="q-{html.escape(query_id)}">
                  <div class="query-side">
                    <div class="sticky">
                      <h2>query {html.escape(query_id)}</h2>
                      <p>cluster {query_cluster}</p>
                      <a href="{html.escape(query_src)}" target="_blank"><img src="{html.escape(query_src)}" loading="lazy" /></a>
                    </div>
                  </div>
                  <div class="candidate-grid">{''.join(tiles)}</div>
                </section>
                """
            )

        page_payload = page_review_df[payload_columns].to_dict(orient="records")
        page_name = f"page_{page_index:03d}.html"
        first_query = page_query_ids[0]
        last_query = page_query_ids[-1]
        prev_link = f"page_{page_index - 1:03d}.html" if page_index > 1 else ""
        next_link = f"page_{page_index + 1:03d}.html" if page_index < len(page_chunks) else ""
        nav = " ".join(
            part
            for part in [
                '<a class="button" href="../index.html">Index / global export</a>',
                f'<a class="button" href="{prev_link}">Prev</a>' if prev_link else "",
                f'<a class="button" href="{next_link}">Next</a>' if next_link else "",
            ]
            if part
        )
        page_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Salamander Top{int(top_k)} Review Page {page_index}</title>
  <style>{base_css}</style>
</head>
<body>
  <header>
    <h1>Page {page_index}/{len(page_chunks)}: query {html.escape(str(first_query))} - {html.escape(str(last_query))}</h1>
    {nav}
    <input id="query-filter" type="search" placeholder="filter query image_id on this page" />
    <span class="stats" id="stats"></span>
  </header>
  <main>{''.join(cards)}</main>
  <script>{_download_script(page_payload)}</script>
</body>
</html>
"""
        (pages_dir / page_name).write_text(page_html, encoding="utf-8")
        page_infos.append(
            {
                "page_index": page_index,
                "path": f"pages/{page_name}",
                "first_query": str(first_query),
                "last_query": str(last_query),
                "query_count": int(len(page_query_ids)),
                "review_rows": int(len(page_review_df)),
                "unique_pairs": int(page_review_df["pair_key"].nunique()),
            }
        )

    pd.DataFrame(page_infos).to_csv(output_dir / "page_manifest_v1.csv", index=False)
    page_links = "\n".join(
        f"""
        <a class="page-link" href="{html.escape(str(info['path']))}">
          <strong>Page {int(info['page_index']):03d}</strong><br />
          query {html.escape(str(info['first_query']))} - {html.escape(str(info['last_query']))}<br />
          {int(info['query_count'])} queries, {int(info['review_rows'])} rows, {int(info['unique_pairs'])} unique pairs
        </a>
        """
        for info in page_infos
    )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Salamander Test Top{int(top_k)} YES Review Index</title>
  <style>{base_css}</style>
</head>
<body>
  <header>
    <h1>Salamander Test Top{int(top_k)} YES Review Index</h1>
    <button id="export-json">Download ALL judgments JSON</button>
    <button id="export-csv">Download ALL CSV</button>
    <span class="stats" id="stats"></span>
  </header>
  <main><div class="page-grid">{page_links}</div></main>
  <script>{_download_script(all_pairs_payload)}</script>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(index_html, encoding="utf-8")
    return html_path


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    pair_features_path = _resolve_path(repo_root, args.pair_features).resolve()
    predictions_path = _resolve_path(repo_root, args.predictions).resolve()
    output_dir = _resolve_path(repo_root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = _load_prediction_metadata(predictions_path, dataset=str(args.dataset))
    pair_df = pd.read_csv(pair_features_path, low_memory=False)
    pair_df = pair_df[pair_df.get("dataset", str(args.dataset)).astype(str).eq(str(args.dataset))].copy()
    review_df = _build_symmetric_topk(
        pair_df=pair_df,
        pred_df=pred_df,
        top_k=int(args.top_k),
        score_column=str(args.score_column),
    )
    review_df.to_csv(output_dir / "review_pairs_top10_v1.csv", index=False)

    image_rel_lookup: dict[str, str] = {}
    for row in pred_df.itertuples(index=False):
        rel_path = str(getattr(row, "original_rgb_path_v1", "") or getattr(row, "path", ""))
        image_rel_lookup[str(row.image_id)] = _stage_image(repo_root, output_dir, str(row.image_id), rel_path)

    html_path = _write_html(
        output_dir=output_dir,
        pred_df=pred_df,
        review_df=review_df,
        image_rel_lookup=image_rel_lookup,
        session_name=str(args.session_name),
        dataset=str(args.dataset),
        top_k=int(args.top_k),
        queries_per_page=int(args.queries_per_page),
    )

    summary = {
        "dataset": str(args.dataset),
        "pair_features": str(pair_features_path),
        "predictions": str(predictions_path),
        "score_column": str(args.score_column),
        "top_k": int(args.top_k),
        "queries_per_page": int(args.queries_per_page),
        "page_count": int((pred_df["image_id"].nunique() + int(args.queries_per_page) - 1) // int(args.queries_per_page)),
        "query_images": int(pred_df["image_id"].nunique()),
        "review_rows": int(len(review_df)),
        "unique_pairs": int(review_df["pair_key"].nunique()),
        "html_path": str(html_path),
        "page_manifest_path": str(output_dir / "page_manifest_v1.csv"),
        "review_pairs_path": str(output_dir / "review_pairs_top10_v1.csv"),
        "export_policy": "Clicked YES exports yes; all unclicked unique pairs export no.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "\n".join(
            [
                "# Salamander Test Top10 YES Review",
                "",
                f"- dataset: `{args.dataset}`",
                f"- query images: `{summary['query_images']}`",
                f"- review rows: `{summary['review_rows']}`",
                f"- unique pairs: `{summary['unique_pairs']}`",
                f"- score column: `{args.score_column}`",
                f"- queries per page: `{args.queries_per_page}`",
                f"- page count: `{summary['page_count']}`",
                f"- index HTML: `{html_path}`",
                f"- page manifest: `{output_dir / 'page_manifest_v1.csv'}`",
                "",
                "Open paged HTML files from the index. Click `YES` for same-identity pairs. Unclicked pairs are exported as `no` from the index page.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[salamander_test_top10_yes_review] html: {html_path}")
    print(f"[salamander_test_top10_yes_review] pairs: {output_dir / 'review_pairs_top10_v1.csv'}")
    print(f"[salamander_test_top10_yes_review] summary: {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
