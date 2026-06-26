from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_TCU_TEXAS_CHIP_MANIFEST_PATH = Path(
    "artifacts/analysis/tcu_texas_dataset_v1/tables/tcu_texas_chip_manifest_v1.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/tcu_texas_official_hotspotter_review_v1")


def _path_ref(repo_root: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=repo_root.resolve()).replace("\\", "/")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            "| "
            + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _normalize_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _load_chip_manifest(chip_manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(chip_manifest_path).copy()
    required_columns = {
        "chip_id",
        "chip_path_v1",
        "external_identity_v1",
        "hotspotter_split_role_v1",
        "hotspotter_query_result_v1",
        "hotspotter_rank1_chip_id_v1",
        "hotspotter_rank1_score_v1",
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{chip_manifest_path} is missing required columns: {missing_columns}")
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].fillna("").astype(str)
    frame["chip_id"] = frame["chip_id"].map(lambda value: str(int(value)))
    frame["hotspotter_rank1_chip_id_v1"] = frame["hotspotter_rank1_chip_id_v1"].map(
        lambda value: "" if _normalize_text(value) == "" else str(int(float(value)))
    )
    return frame


def _resolve_review_pairs(chip_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_df = chip_df[
        chip_df["hotspotter_split_role_v1"].astype(str).eq("Test")
        & chip_df["hotspotter_query_result_v1"].astype(str).str.contains("Positive match", case=False, na=False)
        & chip_df["hotspotter_rank1_chip_id_v1"].astype(str).ne("")
    ].copy()
    if query_df.empty:
        raise ValueError("No official `Test + Positive match + rank1` rows found in the chip manifest.")

    rank_lookup = chip_df.add_prefix("neighbor_").rename(columns={"neighbor_chip_id": "neighbor_chip_id_lookup"})
    merged = query_df.merge(
        rank_lookup,
        left_on="hotspotter_rank1_chip_id_v1",
        right_on="neighbor_chip_id_lookup",
        how="left",
    )
    missing_rank1 = merged["neighbor_chip_id_lookup"].isna()
    if bool(missing_rank1.any()):
        missing_ids = merged.loc[missing_rank1, "hotspotter_rank1_chip_id_v1"].astype(str).tolist()[:5]
        raise ValueError(f"Rank1 chip ids missing from chip manifest: {missing_ids}")

    merged["image_id"] = merged["chip_id"].map(lambda value: f"tcu_chip_{int(value):04d}")
    merged["neighbor_image_id"] = merged["hotspotter_rank1_chip_id_v1"].map(lambda value: f"tcu_chip_{int(value):04d}")
    merged["pair_key"] = merged.apply(
        lambda row: "|".join(sorted([str(row["image_id"]), str(row["neighbor_image_id"])])),
        axis=1,
    )
    neighbor_rank1_lookup = chip_df.set_index("chip_id")["hotspotter_rank1_chip_id_v1"].to_dict()
    merged["reciprocal_top1"] = merged.apply(
        lambda row: str(neighbor_rank1_lookup.get(str(row["hotspotter_rank1_chip_id_v1"]), "")) == str(row["chip_id"]),
        axis=1,
    )
    merged["same_external_identity"] = (
        merged["external_identity_v1"].astype(str) == merged["neighbor_external_identity_v1"].astype(str)
    )
    merged["query_rank"] = range(1, len(merged) + 1)

    directed_columns = [
        "query_rank",
        "pair_key",
        "image_id",
        "neighbor_image_id",
        "chip_id",
        "hotspotter_rank1_chip_id_v1",
        "external_identity_v1",
        "neighbor_external_identity_v1",
        "same_external_identity",
        "reciprocal_top1",
        "hotspotter_rank1_score_v1",
        "hotspotter_query_result_v1",
        "hotspotter_split_role_v1",
        "chip_path_v1",
        "neighbor_chip_path_v1",
        "original_image_name_mapped_v1",
        "neighbor_original_image_name_mapped_v1",
        "original_match_path_v1",
        "neighbor_original_match_path_v1",
        "external_identity_image_count_v1",
        "neighbor_external_identity_image_count_v1",
    ]
    directed_df = merged[directed_columns].copy().rename(
        columns={
            "hotspotter_rank1_chip_id_v1": "neighbor_chip_id",
            "hotspotter_rank1_score_v1": "official_rank1_score",
            "hotspotter_query_result_v1": "official_query_result",
            "hotspotter_split_role_v1": "official_split_role",
        }
    )

    undirected_rows: list[dict[str, object]] = []
    grouped = directed_df.sort_values(
        ["pair_key", "reciprocal_top1", "official_rank1_score"],
        ascending=[True, False, False],
    ).groupby("pair_key", sort=True, dropna=False)
    for review_rank, (_, group_df) in enumerate(grouped, start=1):
        head = group_df.iloc[0]
        member_ids = sorted([str(head["image_id"]), str(head["neighbor_image_id"])])
        left_id, right_id = member_ids[0], member_ids[1]
        left_chip_id = f"{int(left_id.rsplit('_', 1)[-1]):d}"
        right_chip_id = f"{int(right_id.rsplit('_', 1)[-1]):d}"
        left_row = chip_df.loc[chip_df["chip_id"].astype(str) == left_chip_id].iloc[0]
        right_row = chip_df.loc[chip_df["chip_id"].astype(str) == right_chip_id].iloc[0]
        undirected_rows.append(
            {
                "review_rank": int(review_rank),
                "pair_key": str(head["pair_key"]),
                "left_image_id": left_id,
                "right_image_id": right_id,
                "left_chip_id": left_chip_id,
                "right_chip_id": right_chip_id,
                "left_identity": str(left_row["external_identity_v1"]),
                "right_identity": str(right_row["external_identity_v1"]),
                "same_external_identity": bool(str(left_row["external_identity_v1"]) == str(right_row["external_identity_v1"])),
                "directed_support_count": int(len(group_df)),
                "reciprocal_top1": bool(group_df["reciprocal_top1"].astype(bool).any()),
                "max_official_rank1_score": float(pd.to_numeric(group_df["official_rank1_score"], errors="coerce").fillna(0).max()),
                "left_chip_path": str(left_row["chip_path_v1"]),
                "right_chip_path": str(right_row["chip_path_v1"]),
                "left_original_name": str(left_row.get("original_image_name_mapped_v1", "")),
                "right_original_name": str(right_row.get("original_image_name_mapped_v1", "")),
                "left_original_path": str(left_row.get("original_match_path_v1", "")),
                "right_original_path": str(right_row.get("original_match_path_v1", "")),
                "left_identity_image_count": int(pd.to_numeric(left_row.get("external_identity_image_count_v1", 0), errors="coerce")),
                "right_identity_image_count": int(pd.to_numeric(right_row.get("external_identity_image_count_v1", 0), errors="coerce")),
            }
        )
    undirected_df = pd.DataFrame(undirected_rows).sort_values("review_rank").reset_index(drop=True)
    return directed_df, undirected_df


def _load_preview_image(repo_root: Path, rel_path: str, thumb_size: tuple[int, int]) -> Image.Image:
    text = _normalize_text(rel_path)
    if not text:
        return _placeholder_panel("missing", thumb_size)
    image_path = repo_root / text
    if not image_path.exists():
        return _placeholder_panel(f"missing\n{text}", thumb_size)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    return ImageOps.pad(rgb, thumb_size, method=Image.Resampling.BILINEAR, color=(22, 22, 22))


def _placeholder_panel(text: str, size: tuple[int, int]) -> Image.Image:
    image = Image.new("RGB", size, color=(238, 235, 229))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = str(text).splitlines() or [""]
    line_height = 14
    total_height = len(lines) * line_height
    start_y = max(12, (size[1] - total_height) // 2)
    for index, line in enumerate(lines):
        draw.text((12, start_y + index * line_height), line[:52], fill=(80, 80, 80), font=font)
    return image


def _draw_wrapped_text(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, text: str, font: ImageFont.ImageFont) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for index, line in enumerate(lines[:4]):
        draw.text((x, y + index * 14), line, fill=(45, 45, 45), font=font)


def _render_pair_board(
    *,
    repo_root: Path,
    row: pd.Series,
    output_path: Path,
    chip_thumb_size: tuple[int, int] = (360, 360),
    original_thumb_size: tuple[int, int] = (220, 220),
) -> None:
    margin = 18
    gap = 14
    section_gap = 20
    title_h = 42
    meta_h = 94
    canvas_width = margin * 2 + chip_thumb_size[0] * 2 + gap
    originals_y = margin + title_h + meta_h + chip_thumb_size[1] + section_gap
    canvas_height = originals_y + original_thumb_size[1] + 90

    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(247, 244, 238))
    draw = ImageDraw.Draw(canvas)
    body_font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    left_x = margin
    right_x = margin + chip_thumb_size[0] + gap
    chip_y = margin + title_h + meta_h

    left_chip = _load_preview_image(repo_root, str(row["left_chip_path"]), chip_thumb_size)
    right_chip = _load_preview_image(repo_root, str(row["right_chip_path"]), chip_thumb_size)
    left_orig = _load_preview_image(repo_root, str(row["left_original_path"]), original_thumb_size)
    right_orig = _load_preview_image(repo_root, str(row["right_original_path"]), original_thumb_size)

    canvas.paste(left_chip, (left_x, chip_y))
    canvas.paste(right_chip, (right_x, chip_y))
    canvas.paste(left_orig, (left_x, originals_y))
    canvas.paste(right_orig, (right_x, originals_y))

    title = (
        f"Review {int(row['review_rank']):02d} | "
        f"{row['left_image_id']} vs {row['right_image_id']} | "
        f"support={int(row['directed_support_count'])} | "
        f"reciprocal={bool(row['reciprocal_top1'])}"
    )
    draw.text((margin, margin), title, fill=(20, 20, 20), font=title_font)

    left_meta = (
        f"left chip={row['left_chip_id']} | id={row['left_identity']} | n={int(row['left_identity_image_count'])}\n"
        f"original={row['left_original_name'] or '-'}"
    )
    right_meta = (
        f"right chip={row['right_chip_id']} | id={row['right_identity']} | n={int(row['right_identity_image_count'])}\n"
        f"original={row['right_original_name'] or '-'}"
    )
    draw.multiline_text((left_x, margin + title_h), left_meta, fill=(40, 40, 40), font=body_font, spacing=3)
    draw.multiline_text((right_x, margin + title_h), right_meta, fill=(40, 40, 40), font=body_font, spacing=3)

    meta_line = (
        f"same_external_identity={bool(row['same_external_identity'])} | "
        f"max_rank1_score={float(row['max_official_rank1_score']):.2f} | "
        "question: biological same individual?"
    )
    _draw_wrapped_text(draw, margin, margin + title_h + 42, canvas_width - margin * 2, meta_line, body_font)
    draw.text((left_x, chip_y + chip_thumb_size[1] + 4), "chip", fill=(60, 60, 60), font=body_font)
    draw.text((right_x, chip_y + chip_thumb_size[1] + 4), "chip", fill=(60, 60, 60), font=body_font)
    draw.text((left_x, originals_y + original_thumb_size[1] + 4), "original", fill=(60, 60, 60), font=body_font)
    draw.text((right_x, originals_y + original_thumb_size[1] + 4), "original", fill=(60, 60, 60), font=body_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _build_overview_contact_sheet(
    *,
    repo_root: Path,
    review_df: pd.DataFrame,
    output_path: Path,
    columns: int = 4,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if review_df.empty:
        return
    font = ImageFont.load_default()
    title_height = 34
    caption_height = 54
    margin = 12
    gap = 12
    rows = math.ceil(len(review_df) / columns)
    cell_w = thumb_size[0]
    cell_h = thumb_size[1] + caption_height
    canvas = Image.new(
        "RGB",
        (
            margin * 2 + columns * cell_w + (columns - 1) * gap,
            margin * 2 + title_height + rows * cell_h + (rows - 1) * gap,
        ),
        color=(250, 249, 245),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), f"Official HotSpotter review pairs | n={len(review_df)}", fill=(20, 20, 20), font=font)
    start_y = margin + title_height
    for index, row in enumerate(review_df.itertuples(index=False), start=0):
        grid_x = index % columns
        grid_y = index // columns
        x = margin + grid_x * (cell_w + gap)
        y = start_y + grid_y * (cell_h + gap)
        left = _load_preview_image(repo_root, str(row.left_chip_path), thumb_size)
        right = _load_preview_image(repo_root, str(row.right_chip_path), thumb_size)
        pair_image = Image.new("RGB", (thumb_size[0] * 2 + 8, thumb_size[1]), color=(245, 242, 236))
        pair_image.paste(left, (0, 0))
        pair_image.paste(right, (thumb_size[0] + 8, 0))
        pair_thumb = ImageOps.contain(pair_image, thumb_size, method=Image.Resampling.BILINEAR)
        pair_thumb = ImageOps.pad(pair_thumb, thumb_size, color=(15, 15, 15))
        canvas.paste(pair_thumb, (x, y))
        caption = (
            f"{int(row.review_rank):02d} | {row.left_identity} vs {row.right_identity}\n"
            f"recip={bool(row.reciprocal_top1)} | same_tag={bool(row.same_external_identity)}"
        )
        draw.multiline_text((x, y + thumb_size[1] + 4), caption, fill=(40, 40, 40), font=font, spacing=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def build_tcu_texas_official_hotspotter_review(
    *,
    repo_root: Path,
    chip_manifest_path: Path = DEFAULT_TCU_TEXAS_CHIP_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    resolved_manifest_path = (chip_manifest_path if chip_manifest_path.is_absolute() else (repo_root / chip_manifest_path)).resolve()
    resolved_output_dir = (output_dir if output_dir.is_absolute() else (repo_root / output_dir)).resolve()
    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    qualitative_dir = resolved_output_dir / "qualitative"
    pair_board_dir = qualitative_dir / "pair_boards"
    for path in [resolved_output_dir, tables_dir, reports_dir, qualitative_dir, pair_board_dir]:
        path.mkdir(parents=True, exist_ok=True)

    chip_df = _load_chip_manifest(resolved_manifest_path)
    directed_df, review_df = _resolve_review_pairs(chip_df)

    board_rows: list[dict[str, object]] = []
    for row in review_df.itertuples(index=False):
        board_path = pair_board_dir / f"review_{int(row.review_rank):02d}_{row.left_chip_id}_{row.right_chip_id}.jpg"
        _render_pair_board(repo_root=repo_root, row=pd.Series(row._asdict()), output_path=board_path)
        board_rows.append(
            {
                "review_rank": int(row.review_rank),
                "pair_key": str(row.pair_key),
                "board_path": _path_ref(repo_root, board_path),
            }
        )
    board_index_df = pd.DataFrame(board_rows)
    review_with_boards_df = review_df.merge(board_index_df, on=["review_rank", "pair_key"], how="left")
    review_with_boards_df["human_label"] = ""
    review_with_boards_df["human_note"] = ""

    directed_path = tables_dir / "official_top1_directed_pairs_v1.csv"
    review_path = tables_dir / "official_top1_review_pairs_v1.csv"
    judgment_template_path = tables_dir / "official_top1_review_judgment_template_v1.csv"
    board_index_path = tables_dir / "official_top1_pair_board_index_v1.csv"
    directed_df.to_csv(directed_path, index=False)
    review_df.to_csv(review_path, index=False)
    review_with_boards_df.to_csv(judgment_template_path, index=False)
    board_index_df.to_csv(board_index_path, index=False)

    overview_path = qualitative_dir / "official_top1_review_overview_v1.jpg"
    _build_overview_contact_sheet(repo_root=repo_root, review_df=review_df, output_path=overview_path)

    summary = {
        "chip_manifest_path": _path_ref(repo_root, resolved_manifest_path),
        "directed_query_rows": int(len(directed_df)),
        "undirected_review_pairs": int(len(review_df)),
        "reciprocal_pairs": int(review_df["reciprocal_top1"].astype(bool).sum()) if not review_df.empty else 0,
        "same_external_identity_pairs": int(review_df["same_external_identity"].astype(bool).sum()) if not review_df.empty else 0,
        "cross_external_identity_pairs": int((~review_df["same_external_identity"].astype(bool)).sum()) if not review_df.empty else 0,
        "max_directed_support_count": int(review_df["directed_support_count"].max()) if not review_df.empty else 0,
        "directed_pairs_path": _path_ref(repo_root, directed_path),
        "review_pairs_path": _path_ref(repo_root, review_path),
        "judgment_template_path": _path_ref(repo_root, judgment_template_path),
        "board_index_path": _path_ref(repo_root, board_index_path),
        "overview_path": _path_ref(repo_root, overview_path),
        "pair_board_dir": _path_ref(repo_root, pair_board_dir),
    }

    summary_preview = review_df[
        [
            "review_rank",
            "left_identity",
            "right_identity",
            "same_external_identity",
            "reciprocal_top1",
            "directed_support_count",
            "max_official_rank1_score",
        ]
    ].head(12)
    summary_lines = [
        "# TCU Texas Official HotSpotter Review",
        "",
        "## Summary",
        "",
        f"- `chip_manifest_path`: `{summary['chip_manifest_path']}`",
        f"- `directed_query_rows`: `{summary['directed_query_rows']}`",
        f"- `undirected_review_pairs`: `{summary['undirected_review_pairs']}`",
        f"- `reciprocal_pairs`: `{summary['reciprocal_pairs']}`",
        f"- `same_external_identity_pairs`: `{summary['same_external_identity_pairs']}`",
        f"- `cross_external_identity_pairs`: `{summary['cross_external_identity_pairs']}`",
        f"- `max_directed_support_count`: `{summary['max_directed_support_count']}`",
        "",
        "## How To Review",
        "",
        "- 先看 `qualitative/official_top1_review_overview_v1.jpg`，快速扫一遍哪些 pair 很像。",
        "- 再按 `review_rank` 打开 `qualitative/pair_boards/` 里的逐对图板。",
        "- 人工结论直接回填到 `official_top1_review_judgment_template_v1.csv` 的 `human_label` 和 `human_note`。",
        "- `human_label` 建议只填：`yes`、`no`、`uncertain`。",
        "",
        "## Key Paths",
        "",
        f"- `directed_pairs_path`: `{summary['directed_pairs_path']}`",
        f"- `review_pairs_path`: `{summary['review_pairs_path']}`",
        f"- `judgment_template_path`: `{summary['judgment_template_path']}`",
        f"- `board_index_path`: `{summary['board_index_path']}`",
        f"- `overview_path`: `{summary['overview_path']}`",
        f"- `pair_board_dir`: `{summary['pair_board_dir']}`",
        "",
        "## Review Preview",
        "",
        _markdown_table(summary_preview),
        "",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "summary_path": summary_path,
        "directed_pairs_path": directed_path,
        "review_pairs_path": review_path,
        "judgment_template_path": judgment_template_path,
        "board_index_path": board_index_path,
        "overview_path": overview_path,
        "pair_board_dir": pair_board_dir,
    }
