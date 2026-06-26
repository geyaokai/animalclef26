from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    return "".join(str(name).strip().lower().split())


def build_image_index(image_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image_index[normalize_name(path.name)] = path
    return image_index


def resolve_image_path(image_name: object, image_index: dict[str, Path]) -> Path | None:
    if pd.isna(image_name):
        return None
    return image_index.get(normalize_name(str(image_name)))


def relpath_or_empty(target: Path | None, html_parent: Path) -> str:
    if target is None:
        return ""
    return os.path.relpath(target, html_parent)


def format_score(score: object) -> str:
    if pd.isna(score):
        return ""
    text = str(score).strip()
    try:
        return f"{float(text):.2f}"
    except ValueError:
        if "+" in text and "e" not in text.lower():
            left, right = text.split("+", 1)
            if left and right.isdigit():
                try:
                    return f"{float(left) * (10 ** int(right)):.2f}"
                except ValueError:
                    return text
        return text


def get_rank_score(row: pd.Series, rank_idx: int) -> object:
    primary = f"Rank {rank_idx} - Score"
    if primary in row.index:
        return row[primary]
    fallback = f"Rank {rank_idx}"
    if fallback in row.index:
        return row[fallback]
    return ""


def order_group(role: str, result: str) -> tuple[int, str]:
    role_clean = str(role).strip()
    result_clean = str(result).strip()
    positive = {"Positive match", "Postive match"}
    if role_clean in {"Test", "Reference"} and result_clean in positive:
        return (0, "test_reference_positive")
    if role_clean in {"Test", "Reference"}:
        return (1, "test_reference_nonpositive")
    return (2, "unique")


def build_ordered_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    role = df["role_image"].astype(str).str.strip()
    result = df["Query Result"].astype(str).str.strip()
    order_keys = [order_group(r, q) for r, q in zip(role, result, strict=False)]
    df = df.copy()
    df["group_order"] = [item[0] for item in order_keys]
    df["group_name"] = [item[1] for item in order_keys]
    df["role_clean"] = role
    df["result_clean"] = result
    df["query_name_clean"] = df["query_origin_image_name"].astype(str).str.lower()
    df = df.sort_values(
        by=["group_order", "role_clean", "result_clean", "query_name_clean", "query_chip_id"],
        kind="stable",
    ).reset_index(drop=True)
    return df


def render_query_card(row: pd.Series, image_index: dict[str, Path], html_parent: Path) -> str:
    query_name = row["query_origin_image_name"]
    query_label = row["query_origin_label_hl"]
    query_relpath = relpath_or_empty(resolve_image_path(query_name, image_index), html_parent)
    image_html = (
        f'<a href="{html.escape(query_relpath)}" target="_blank" rel="noopener noreferrer">'
        f'<img src="{html.escape(query_relpath)}" loading="lazy" alt="{html.escape(str(query_name))}"></a>'
        if query_relpath
        else '<div class="img-missing">missing</div>'
    )
    return (
        '<div class="query-card">'
        '<div class="card-title">Query</div>'
        f"{image_html}"
        f'<div class="meta">{html.escape(str(query_name))}</div>'
        f'<div class="meta">{html.escape(str(query_label))}</div>'
        f'<div class="meta">chip: {html.escape(str(row["query_chip_id"]))}</div>'
        "</div>"
    )


def render_candidate_card(
    row: pd.Series,
    rank_idx: int,
    image_index: dict[str, Path],
    html_parent: Path,
) -> str:
    image_name = row.get(f"rank_{rank_idx}_origin_image_name", "")
    label = row.get(f"rank_{rank_idx}_origin_label_hl", "")
    candidate_chip = row.get(f"rank_{rank_idx}_id", row.get(f"Rank {rank_idx} - Chip", ""))
    score = format_score(get_rank_score(row, rank_idx))
    relpath = relpath_or_empty(resolve_image_path(image_name, image_index), html_parent)
    image_html = (
        f'<a href="{html.escape(relpath)}" target="_blank" rel="noopener noreferrer">'
        f'<img src="{html.escape(relpath)}" loading="lazy" alt="{html.escape(str(image_name))}"></a>'
        if relpath
        else '<div class="img-missing">missing</div>'
    )
    return (
        f'<button class="candidate-card" type="button" data-rank="{rank_idx}" '
        f'data-candidate-chip="{html.escape(str(candidate_chip))}" '
        f'data-candidate-name="{html.escape(str(image_name))}">'
        f'<div class="card-title">Rank {rank_idx}</div>'
        f"{image_html}"
        f'<div class="meta">{html.escape(str(image_name))}</div>'
        f'<div class="meta">{html.escape(str(label))}</div>'
        f'<div class="meta">chip: {html.escape(str(candidate_chip))}</div>'
        f'<div class="meta">score: {html.escape(score)}</div>'
        '<div class="select-badge">click to select</div>'
        "</button>"
    )


def build_row_html(row: pd.Series, image_index: dict[str, Path], html_parent: Path, row_idx: int) -> str:
    query_chip = row["query_chip_id"]
    role = row["role_clean"]
    result = row["result_clean"]
    group_name = row["group_name"]
    search_blob = " ".join(
        [
            str(row["Chip"]),
            str(query_chip),
            str(role),
            str(result),
            str(row["query_origin_image_name"]),
            str(row["query_origin_label_hl"]),
            *[str(row.get(f"rank_{idx}_origin_image_name", "")) for idx in range(1, 11)],
            *[str(row.get(f"rank_{idx}_origin_label_hl", "")) for idx in range(1, 11)],
        ]
    ).lower()

    query_card = render_query_card(row, image_index, html_parent)
    candidate_cards = "".join(
        render_candidate_card(row=row, rank_idx=rank_idx, image_index=image_index, html_parent=html_parent)
        for rank_idx in range(1, 11)
    )
    header_text = (
        f'#{row_idx + 1} | query_chip={query_chip} | role={role} | '
        f'result={result} | query={row["query_origin_image_name"]}'
    )
    return (
        f'<section class="query-row" data-query-chip="{html.escape(str(query_chip))}" '
        f'data-role="{html.escape(str(role))}" data-result="{html.escape(str(result))}" '
        f'data-group="{html.escape(str(group_name))}" data-search="{html.escape(search_blob)}">'
        f'<div class="row-header">{html.escape(header_text)}</div>'
        '<div class="row-toolbar">'
        '<button class="toolbar-btn clear-btn" type="button">Clear</button>'
        '<button class="toolbar-btn no-match-btn" type="button">Mark No Match</button>'
        '<button class="toolbar-btn reviewed-btn" type="button">Toggle Reviewed</button>'
        '<div class="selection-status">Selected: none</div>'
        "</div>"
        '<div class="row-grid">'
        f"{query_card}"
        f"{candidate_cards}"
        "</div>"
        "</section>"
    )


def render_html(df: pd.DataFrame, output_path: Path, image_root: Path) -> None:
    image_index = build_image_index(image_root)
    rows_html = [
        build_row_html(row=row, image_index=image_index, html_parent=output_path.parent, row_idx=row_idx)
        for row_idx, (_, row) in enumerate(df.iterrows())
    ]

    counts = df["group_name"].value_counts().to_dict()
    summary_json = html.escape(json.dumps(counts, ensure_ascii=False))
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TCU Texas HotSpotter Manual Annotation Board</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7fb;
      color: #111827;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 30;
      background: #ffffffee;
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #dbe3ef;
      padding: 12px 16px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .subtitle {{
      font-size: 13px;
      color: #475569;
      margin-bottom: 10px;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .controls input, .controls select, .controls button {{
      padding: 8px 10px;
      border: 1px solid #c7d2e0;
      border-radius: 8px;
      background: white;
      font-size: 14px;
    }}
    .controls button {{
      cursor: pointer;
      background: #eff6ff;
    }}
    .summary {{
      font-size: 13px;
      color: #334155;
    }}
    .legend {{
      margin-top: 8px;
      font-size: 12px;
      color: #475569;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }}
    .container {{
      padding: 16px;
    }}
    .query-row {{
      background: white;
      border: 1px solid #dbe3ef;
      border-radius: 12px;
      margin-bottom: 16px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }}
    .query-row[data-group="test_reference_positive"] .row-header {{
      background: #ecfdf5;
    }}
    .query-row[data-group="test_reference_nonpositive"] .row-header {{
      background: #fff7ed;
    }}
    .query-row[data-group="unique"] .row-header {{
      background: #eff6ff;
    }}
    .row-header {{
      padding: 10px 14px;
      border-bottom: 1px solid #dbe3ef;
      font-size: 14px;
      font-weight: 600;
    }}
    .row-toolbar {{
      padding: 10px 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      border-bottom: 1px solid #eef2f7;
      background: #fafcff;
    }}
    .toolbar-btn {{
      padding: 6px 10px;
      border: 1px solid #c7d2e0;
      border-radius: 8px;
      background: white;
      cursor: pointer;
    }}
    .selection-status {{
      font-size: 12px;
      color: #334155;
    }}
    .row-grid {{
      display: grid;
      grid-template-columns: repeat(11, minmax(150px, 1fr));
      gap: 10px;
      padding: 12px;
      align-items: start;
    }}
    .query-card, .candidate-card {{
      border: 1px solid #dbe3ef;
      border-radius: 10px;
      background: white;
      padding: 10px;
    }}
    .candidate-card {{
      cursor: pointer;
      text-align: left;
    }}
    .candidate-card.selected {{
      border: 2px solid #16a34a;
      background: #f0fdf4;
    }}
    .candidate-card.selected .select-badge {{
      background: #16a34a;
      color: white;
    }}
    .query-row.reviewed {{
      box-shadow: 0 0 0 3px #2563eb inset;
    }}
    .query-row.no-match .selection-status {{
      color: #b91c1c;
      font-weight: 700;
    }}
    .card-title {{
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 14px;
    }}
    img, .img-missing {{
      width: 100%;
      height: 180px;
      object-fit: contain;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
      background: #f8fafc;
    }}
    .img-missing {{
      display: flex;
      align-items: center;
      justify-content: center;
      color: #94a3b8;
      font-size: 13px;
    }}
    .meta {{
      margin-top: 6px;
      font-size: 12px;
      line-height: 1.35;
      color: #334155;
      word-break: break-word;
    }}
    .select-badge {{
      margin-top: 8px;
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: #e2e8f0;
      color: #334155;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 2200px) {{
      .row-grid {{ grid-template-columns: repeat(6, minmax(150px, 1fr)); }}
    }}
    @media (max-width: 1200px) {{
      .row-grid {{ grid-template-columns: repeat(3, minmax(150px, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .row-grid {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">TCU Texas HotSpotter Manual Annotation Board</div>
    <div class="subtitle">默认顺序：Test/Reference positive → Test/Reference non-positive → Unique。点击候选图即可标记为“同一个体”。</div>
    <div class="controls">
      <input id="searchBox" type="text" placeholder="Search chip / HL / image name">
      <select id="groupFilter">
        <option value="ALL">All groups</option>
        <option value="test_reference_positive">Test/Reference positive</option>
        <option value="test_reference_nonpositive">Test/Reference non-positive</option>
        <option value="unique">Unique</option>
      </select>
      <select id="reviewFilter">
        <option value="ALL">All review states</option>
        <option value="reviewed">Reviewed</option>
        <option value="unreviewed">Unreviewed</option>
      </select>
      <button id="exportJsonBtn" type="button">Export JSON</button>
      <button id="exportCsvBtn" type="button">Export CSV</button>
      <button id="clearStorageBtn" type="button">Clear All Saved State</button>
      <div class="summary"><span id="visibleCount"></span> / {len(df)} rows visible</div>
    </div>
    <div class="legend">
      <span>counts: {summary_json}</span>
      <span>绿色边框 = 你点选的同个体候选</span>
      <span>蓝色外框 = 已审核</span>
      <span>No Match = top10 都不是同个体</span>
    </div>
  </div>
  <div class="container">
    {''.join(rows_html)}
  </div>
  <script>
    const STORAGE_KEY = 'tcu_texas_hotspotter_manual_annotation_v1';
    const rows = Array.from(document.querySelectorAll('.query-row'));
    const searchBox = document.getElementById('searchBox');
    const groupFilter = document.getElementById('groupFilter');
    const reviewFilter = document.getElementById('reviewFilter');
    const visibleCount = document.getElementById('visibleCount');
    const exportJsonBtn = document.getElementById('exportJsonBtn');
    const exportCsvBtn = document.getElementById('exportCsvBtn');
    const clearStorageBtn = document.getElementById('clearStorageBtn');

    function loadState() {{
      try {{
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{{}}');
      }} catch (error) {{
        console.warn('Failed to parse storage, reset it.', error);
        return {{}};
      }}
    }}

    let state = loadState();

    function saveState() {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function ensureRowState(queryChip) {{
      if (!state[queryChip]) {{
        state[queryChip] = {{
          selected_ranks: [],
          selected_candidate_chips: [],
          selected_candidate_names: [],
          reviewed: false,
          no_match: false,
        }};
      }}
      return state[queryChip];
    }}

    function updateRowUI(row) {{
      const queryChip = row.dataset.queryChip;
      const rowState = ensureRowState(queryChip);
      const cards = Array.from(row.querySelectorAll('.candidate-card'));
      cards.forEach((card) => {{
        const rank = Number(card.dataset.rank);
        const isSelected = rowState.selected_ranks.includes(rank);
        card.classList.toggle('selected', isSelected);
      }});
      row.classList.toggle('reviewed', !!rowState.reviewed);
      row.classList.toggle('no-match', !!rowState.no_match);
      const status = row.querySelector('.selection-status');
      if (rowState.no_match) {{
        status.textContent = 'Selected: No Match';
      }} else if (rowState.selected_ranks.length === 0) {{
        status.textContent = 'Selected: none';
      }} else {{
        const parts = rowState.selected_ranks.map((rank, idx) => {{
          const chip = rowState.selected_candidate_chips[idx] || '';
          const name = rowState.selected_candidate_names[idx] || '';
          return `R${{rank}}:${{chip}}:${{name}}`;
        }});
        status.textContent = `Selected: ${{parts.join(' | ')}}`;
      }}
    }}

    function updateAllRows() {{
      rows.forEach(updateRowUI);
      applyFilter();
    }}

    function toggleCandidate(row, card) {{
      const queryChip = row.dataset.queryChip;
      const rowState = ensureRowState(queryChip);
      const rank = Number(card.dataset.rank);
      const chip = card.dataset.candidateChip || '';
      const name = card.dataset.candidateName || '';
      const existingIndex = rowState.selected_ranks.indexOf(rank);
      if (existingIndex >= 0) {{
        rowState.selected_ranks.splice(existingIndex, 1);
        rowState.selected_candidate_chips.splice(existingIndex, 1);
        rowState.selected_candidate_names.splice(existingIndex, 1);
      }} else {{
        rowState.selected_ranks.push(rank);
        rowState.selected_candidate_chips.push(chip);
        rowState.selected_candidate_names.push(name);
        const zipped = rowState.selected_ranks.map((value, idx) => ({{
          rank: value,
          chip: rowState.selected_candidate_chips[idx],
          name: rowState.selected_candidate_names[idx],
        }})).sort((a, b) => a.rank - b.rank);
        rowState.selected_ranks = zipped.map(item => item.rank);
        rowState.selected_candidate_chips = zipped.map(item => item.chip);
        rowState.selected_candidate_names = zipped.map(item => item.name);
      }}
      rowState.no_match = false;
      rowState.reviewed = true;
      saveState();
      updateRowUI(row);
    }}

    function clearRow(row) {{
      const queryChip = row.dataset.queryChip;
      state[queryChip] = {{
        selected_ranks: [],
        selected_candidate_chips: [],
        selected_candidate_names: [],
        reviewed: false,
        no_match: false,
      }};
      saveState();
      updateRowUI(row);
    }}

    function markNoMatch(row) {{
      const queryChip = row.dataset.queryChip;
      const rowState = ensureRowState(queryChip);
      rowState.selected_ranks = [];
      rowState.selected_candidate_chips = [];
      rowState.selected_candidate_names = [];
      rowState.no_match = true;
      rowState.reviewed = true;
      saveState();
      updateRowUI(row);
    }}

    function toggleReviewed(row) {{
      const queryChip = row.dataset.queryChip;
      const rowState = ensureRowState(queryChip);
      rowState.reviewed = !rowState.reviewed;
      saveState();
      updateRowUI(row);
    }}

    function applyFilter() {{
      const needle = searchBox.value.trim().toLowerCase();
      const groupNeedle = groupFilter.value;
      const reviewNeedle = reviewFilter.value;
      let count = 0;
      rows.forEach((row) => {{
        const queryChip = row.dataset.queryChip;
        const rowState = ensureRowState(queryChip);
        const searchBlob = row.dataset.search || '';
        const group = row.dataset.group || '';
        const reviewed = !!rowState.reviewed;
        const okSearch = !needle || searchBlob.includes(needle);
        const okGroup = groupNeedle === 'ALL' || group === groupNeedle;
        const okReview = reviewNeedle === 'ALL' || (reviewNeedle === 'reviewed' && reviewed) || (reviewNeedle === 'unreviewed' && !reviewed);
        const visible = okSearch && okGroup && okReview;
        row.classList.toggle('hidden', !visible);
        if (visible) count += 1;
      }});
      visibleCount.textContent = String(count);
    }}

    function buildExportRows() {{
      return rows.map((row) => {{
        const queryChip = row.dataset.queryChip;
        const rowState = ensureRowState(queryChip);
        const header = row.querySelector('.row-header').textContent;
        return {{
          query_chip_id: queryChip,
          role: row.dataset.role || '',
          result: row.dataset.result || '',
          group: row.dataset.group || '',
          reviewed: !!rowState.reviewed,
          no_match: !!rowState.no_match,
          selected_ranks: rowState.selected_ranks,
          selected_candidate_chips: rowState.selected_candidate_chips,
          selected_candidate_names: rowState.selected_candidate_names,
          header: header,
        }};
      }});
    }}

    function triggerDownload(filename, content, mimeType) {{
      const blob = new Blob([content], {{ type: mimeType }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.click();
      URL.revokeObjectURL(url);
    }}

    exportJsonBtn.addEventListener('click', () => {{
      const payload = {{
        storage_key: STORAGE_KEY,
        rows: buildExportRows(),
      }};
      triggerDownload('tcu_texas_manual_annotations.json', JSON.stringify(payload, null, 2), 'application/json');
    }});

    exportCsvBtn.addEventListener('click', () => {{
      const rowsForExport = buildExportRows();
      const header = [
        'query_chip_id', 'role', 'result', 'group', 'reviewed', 'no_match',
        'selected_ranks', 'selected_candidate_chips', 'selected_candidate_names', 'header'
      ];
      const lines = [header.join(',')];
      rowsForExport.forEach((item) => {{
        const values = [
          item.query_chip_id,
          item.role,
          item.result,
          item.group,
          String(item.reviewed),
          String(item.no_match),
          item.selected_ranks.join('|'),
          item.selected_candidate_chips.join('|'),
          item.selected_candidate_names.join('|'),
          item.header,
        ].map((value) => `"${{String(value).replaceAll('"', '""')}}"`);
        lines.push(values.join(','));
      }});
      triggerDownload('tcu_texas_manual_annotations.csv', lines.join('\\n'), 'text/csv;charset=utf-8;');
    }});

    clearStorageBtn.addEventListener('click', () => {{
      if (!confirm('Clear all saved annotations in localStorage?')) {{
        return;
      }}
      state = {{}};
      saveState();
      updateAllRows();
    }});

    rows.forEach((row) => {{
      row.querySelectorAll('.candidate-card').forEach((card) => {{
        card.addEventListener('click', () => toggleCandidate(row, card));
      }});
      row.querySelector('.clear-btn').addEventListener('click', () => clearRow(row));
      row.querySelector('.no-match-btn').addEventListener('click', () => markNoMatch(row));
      row.querySelector('.reviewed-btn').addEventListener('click', () => toggleReviewed(row));
    }});

    searchBox.addEventListener('input', applyFilter);
    groupFilter.addEventListener('change', applyFilter);
    reviewFilter.addEventListener('change', applyFilter);

    updateAllRows();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clickable manual annotation HTML board for TCU Texas HotSpotter review.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/analysis/tcu_texas_hotspotter_origin_view_v1/6._Output_from_HotSpotter_origin_view.csv"),
        help="Expanded origin-view csv with rank-10 candidates.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=Path("../external/datasets/tcu_texas_horned_lizard_v1/extracted/7. THL images - Original"),
        help="Directory containing the original Texas images.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("artifacts/analysis/tcu_texas_hotspotter_origin_view_v1/manual_annotation_board.html"),
        help="Output HTML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = build_ordered_dataframe(df)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    render_html(df=df, output_path=args.output_html, image_root=args.image_root)
    print(args.output_html)


if __name__ == "__main__":
    main()
