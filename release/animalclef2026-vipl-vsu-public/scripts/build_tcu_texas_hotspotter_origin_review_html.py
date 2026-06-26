from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

import pandas as pd


def normalize_name(name: str) -> str:
    """Build a tolerant key so csv names can match local image filenames."""
    return "".join(str(name).strip().lower().split())


def build_image_index(image_root: Path) -> dict[str, Path]:
    image_index: dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        image_index[normalize_name(path.name)] = path
    return image_index


def resolve_image_path(image_name: str, image_index: dict[str, Path]) -> Path | None:
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


def render_card(title: str, image_relpath: str, image_name: str, label_hl: str, score: str = "") -> str:
    title_html = html.escape(title)
    image_name_html = html.escape(str(image_name))
    label_html = html.escape(str(label_hl))
    score_html = html.escape(str(score)) if score else ""
    if image_relpath:
        image_tag = (
            f'<a href="{html.escape(image_relpath)}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{html.escape(image_relpath)}" loading="lazy" alt="{image_name_html}"></a>'
        )
    else:
        image_tag = '<div class="img-missing">missing</div>'
    score_block = f'<div class="meta">score: {score_html}</div>' if score else ""
    return (
        '<div class="card">'
        f'<div class="card-title">{title_html}</div>'
        f"{image_tag}"
        f'<div class="meta">{image_name_html}</div>'
        f'<div class="meta">{label_html}</div>'
        f"{score_block}"
        "</div>"
    )


def build_row_html(row: pd.Series, html_parent: Path, image_index: dict[str, Path]) -> str:
    query_name = row["query_origin_image_name"]
    query_label = row["query_origin_label_hl"]
    role = row["role_image"]
    result = row.get("Query Result", "")
    chip = row["Chip"]

    query_relpath = relpath_or_empty(resolve_image_path(query_name, image_index), html_parent)
    cards = [
        render_card(
            title="Query",
            image_relpath=query_relpath,
            image_name=query_name,
            label_hl=query_label,
        )
    ]

    for rank_idx in range(1, 6):
        rank_name = row.get(f"rank_{rank_idx}_origin_image_name", "")
        rank_label = row.get(f"rank_{rank_idx}_origin_label_hl", "")
        rank_score = row.get(f"Rank {rank_idx} - Score", "")
        rank_relpath = relpath_or_empty(resolve_image_path(rank_name, image_index), html_parent)
        cards.append(
            render_card(
                title=f"Rank {rank_idx}",
                image_relpath=rank_relpath,
                image_name=rank_name,
                label_hl=rank_label,
                score=format_score(rank_score),
            )
        )

    search_blob = " ".join(
        [
            str(chip),
            str(role),
            str(result),
            str(query_name),
            str(query_label),
            *[str(row.get(f"rank_{idx}_origin_image_name", "")) for idx in range(1, 6)],
            *[str(row.get(f"rank_{idx}_origin_label_hl", "")) for idx in range(1, 6)],
        ]
    ).lower()

    status_text = f"Chip={chip} | role={role} | result={result} | query={query_name} ({query_label})"
    return (
        f'<section class="query-row" data-role="{html.escape(str(role).strip())}" '
        f'data-search="{html.escape(search_blob)}">'
        f'<div class="row-header">{html.escape(status_text)}</div>'
        f'<div class="row-grid">{"".join(cards)}</div>'
        "</section>"
    )


def render_html(df: pd.DataFrame, output_path: Path, image_root: Path) -> None:
    image_index = build_image_index(image_root)
    rows_html = [
        build_row_html(row, output_path.parent, image_index)
        for _, row in df.iterrows()
    ]

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TCU Texas HotSpotter Origin Review</title>
  <style>
    body {{
      font-family: Arial, Helvetica, sans-serif;
      margin: 0;
      background: #f5f7fb;
      color: #111827;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: #ffffffee;
      backdrop-filter: blur(8px);
      border-bottom: 1px solid #dbe3ef;
      padding: 12px 16px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .controls {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .controls input, .controls select {{
      padding: 8px 10px;
      border: 1px solid #c7d2e0;
      border-radius: 8px;
      font-size: 14px;
      background: white;
    }}
    .summary {{
      font-size: 13px;
      color: #475569;
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
    .row-header {{
      padding: 10px 14px;
      background: #eff6ff;
      border-bottom: 1px solid #dbe3ef;
      font-size: 14px;
      font-weight: 600;
    }}
    .row-grid {{
      display: grid;
      grid-template-columns: repeat(6, minmax(180px, 1fr));
      gap: 12px;
      padding: 12px;
    }}
    .card {{
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      background: #fff;
      padding: 10px;
    }}
    .card-title {{
      font-weight: 700;
      margin-bottom: 8px;
      font-size: 14px;
    }}
    .card img, .img-missing {{
      width: 100%;
      height: 220px;
      object-fit: contain;
      background: #f8fafc;
      border-radius: 8px;
      border: 1px solid #e2e8f0;
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
      line-height: 1.4;
      word-break: break-word;
      color: #334155;
    }}
    .hidden {{
      display: none;
    }}
    @media (max-width: 1400px) {{
      .row-grid {{
        grid-template-columns: repeat(3, minmax(180px, 1fr));
      }}
    }}
    @media (max-width: 900px) {{
      .row-grid {{
        grid-template-columns: repeat(2, minmax(180px, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .row-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="topbar">
    <div class="title">TCU Texas HotSpotter Origin Review</div>
    <div class="controls">
      <input id="searchBox" type="text" placeholder="Search chip / HL / image name">
      <select id="roleFilter">
        <option value="ALL">All roles</option>
        <option value="Test">Test</option>
        <option value="Unique">Unique</option>
        <option value="Reference">Reference</option>
      </select>
      <div class="summary"><span id="visibleCount"></span> / {len(df)} rows visible</div>
    </div>
  </div>
  <div class="container">
    {''.join(rows_html)}
  </div>
  <script>
    const rows = Array.from(document.querySelectorAll('.query-row'));
    const searchBox = document.getElementById('searchBox');
    const roleFilter = document.getElementById('roleFilter');
    const visibleCount = document.getElementById('visibleCount');

    function updateFilter() {{
      const needle = searchBox.value.trim().toLowerCase();
      const roleNeedle = roleFilter.value;
      let count = 0;
      rows.forEach((row) => {{
        const searchBlob = row.dataset.search || '';
        const role = (row.dataset.role || '').trim();
        const okSearch = !needle || searchBlob.includes(needle);
        const okRole = roleNeedle === 'ALL' || role === roleNeedle || role === `${{roleNeedle}} `;
        const visible = okSearch && okRole;
        row.classList.toggle('hidden', !visible);
        if (visible) count += 1;
      }});
      visibleCount.textContent = String(count);
    }}

    searchBox.addEventListener('input', updateFilter);
    roleFilter.addEventListener('change', updateFilter);
    updateFilter();
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a local HTML review board for TCU Texas HotSpotter origin-level inspection."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("artifacts/analysis/tcu_texas_hotspotter_origin_view_v1/6._Output_from_HotSpotter_origin_view_review.csv"),
        help="Review csv with query and rank origin names / labels.",
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
        default=Path("artifacts/analysis/tcu_texas_hotspotter_origin_view_v1/review_board.html"),
        help="Output HTML file.",
    )
    parser.add_argument(
        "--only-nonpositive-test-reference",
        action="store_true",
        help="Keep only Test/Reference rows whose query result is not positive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    if args.only_nonpositive_test_reference:
        role = df["role_image"].astype(str).str.strip()
        result = df["Query Result"].astype(str).str.strip()
        positive = {"Positive match", "Postive match"}
        df = df[role.isin(["Test", "Reference"]) & (~result.isin(positive))].copy()
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    render_html(df=df, output_path=args.output_html, image_root=args.image_root)
    print(args.output_html)


if __name__ == "__main__":
    main()
