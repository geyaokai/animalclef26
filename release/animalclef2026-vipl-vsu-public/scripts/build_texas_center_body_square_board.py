#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


def _save_image(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=92)
    return path.as_posix()


def _thumb(path: Path, size: tuple[int, int]) -> Image.Image:
    return ImageOps.pad(Image.open(path).convert("RGB"), size, color=(8, 8, 8))


def _draw_detail_board(records: pd.DataFrame, *, output_path: Path, thumb_size: tuple[int, int] = (180, 180), columns: int = 2) -> None:
    if records.empty:
        return
    panel_names = [
        ("original_path", "original"),
        ("aligned_path", "sam aligned"),
        ("scale_norm_path", "scale norm"),
        ("center_body_gray_path", "center square gray"),
    ]
    margin = 12
    header_h = 42
    label_h = 44
    panel_gap = 6
    panel_w, panel_h = thumb_size
    cell_w = len(panel_names) * panel_w + (len(panel_names) - 1) * panel_gap
    cell_h = panel_h + label_h
    rows = math.ceil(len(records) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), "Texas center body square qualitative board", fill=(20, 20, 20), font=font)
    for index, row in enumerate(records.itertuples(index=False)):
        grid_x = index % columns
        grid_y = index // columns
        x0 = margin + grid_x * (cell_w + margin)
        y0 = margin + header_h + grid_y * (cell_h + margin)
        for panel_index, (column, label) in enumerate(panel_names):
            x = x0 + panel_index * (panel_w + panel_gap)
            image = _thumb(Path(str(getattr(row, column))), thumb_size)
            canvas.paste(image, (x, y0))
            draw.text((x, y0 + panel_h + 2), label, fill=(40, 40, 40), font=font)
        sample_label = (
            f"id={row.image_id} | stage={row.sam_stage} | crop_side={row.square_side_px} | "
            f"subject_keep={row.foreground_ratio_of_subject:.3f}"
        )
        draw.text((x0, y0 + panel_h + 18), sample_label[:140], fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _draw_final_grid(records: pd.DataFrame, *, output_path: Path, thumb_size: tuple[int, int] = (112, 112), columns: int = 14) -> None:
    if records.empty:
        return
    margin = 8
    header_h = 34
    label_h = 20
    panel_w, panel_h = thumb_size
    rows = math.ceil(len(records) / columns)
    width = margin * 2 + columns * panel_w
    height = margin * 2 + header_h + rows * (panel_h + label_h)
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), "Texas final center body square gray overview (274 images)", fill=(20, 20, 20), font=font)
    for index, row in enumerate(records.itertuples(index=False)):
        grid_x = index % columns
        grid_y = index // columns
        x = margin + grid_x * panel_w
        y = margin + header_h + grid_y * (panel_h + label_h)
        image = _thumb(Path(str(row.center_body_gray_path)), thumb_size)
        canvas.paste(image, (x, y))
        label = f"{row.image_id}{'*' if row.sam_stage != 'none' else ''}"
        draw.text((x + 3, y + panel_h + 2), label, fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.qualitative_texas_views import (
        crop_texas_center_body_square,
        grayscale_normalize_image,
        scale_normalize_aligned_foreground,
    )
    from animalclef_analysis.sam_orb_veto import infer_mask_from_masked_rgb

    parser = argparse.ArgumentParser(description="Build Texas center-body-square qualitative boards from repaired SAM manifest.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=repo_root / "artifacts/manifests/sam_seg_trainprep_repaired_v1/tables/manifest_test_sam_trainprep_aligned_best_v1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=repo_root / "artifacts/preprocessing_qualitative/texas_center_body_square_repaired_v1")
    parser.add_argument("--detail-sample-count", type=int, default=48)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    views_dir = output_dir / "views"
    qualitative_dir = output_dir / "qualitative"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for directory in [views_dir, qualitative_dir, tables_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(args.manifest_path)
    texas_df = manifest_df[manifest_df["dataset"].astype(str).eq("TexasHornedLizards")].copy().reset_index(drop=True)
    records: list[dict[str, Any]] = []
    for row in texas_df.sort_values("image_id").itertuples(index=False):
        image_id = str(row.image_id)
        original_path = repo_root / str(row.original_rgb_path_v1)
        aligned_path = repo_root / str(row.path)
        with Image.open(original_path) as handle:
            original = handle.convert("RGB")
        with Image.open(aligned_path) as handle:
            aligned = handle.convert("RGB")
        aligned_mask = infer_mask_from_masked_rgb(aligned, nonzero_threshold=1)
        scale_rgb, scale_mask, scale_payload = scale_normalize_aligned_foreground(aligned, aligned_mask)
        center_rgb, center_mask, crop_payload = crop_texas_center_body_square(scale_rgb, scale_mask)
        center_gray_rgb, gray_payload = grayscale_normalize_image(center_rgb, focus_mask=center_mask)

        stem = f"{int(image_id):05d}" if image_id.isdigit() else image_id
        scale_path = views_dir / "scale_norm" / f"{stem}.jpg"
        center_path = views_dir / "center_body_square" / f"{stem}.jpg"
        gray_path = views_dir / "center_body_square_gray" / f"{stem}.jpg"
        aligned_copy_path = views_dir / "aligned_foreground" / f"{stem}.jpg"
        _save_image(aligned, aligned_copy_path)
        _save_image(scale_rgb, scale_path)
        _save_image(center_rgb, center_path)
        _save_image(center_gray_rgb, gray_path)

        records.append(
            {
                "image_id": image_id,
                "dataset": "TexasHornedLizards",
                "split": str(row.split),
                "original_path": original_path.as_posix(),
                "aligned_path": aligned_copy_path.as_posix(),
                "source_aligned_path": aligned_path.as_posix(),
                "scale_norm_path": scale_path.as_posix(),
                "center_body_square_path": center_path.as_posix(),
                "center_body_gray_path": gray_path.as_posix(),
                "sam_stage": str(getattr(row, "sam_trainprep_masked_fallback_stage_v1", "none") or "none"),
                "sam_reason": str(getattr(row, "sam_trainprep_masked_reason_v1", "") or ""),
                "foreground_ratio_of_subject": float(crop_payload["foreground_ratio_of_subject"]),
                "foreground_ratio_in_crop": float(crop_payload["foreground_ratio_in_crop"]),
                "square_side_px": int(crop_payload["square_side_px"]),
                "crop_fallback_reason": crop_payload.get("fallback_reason"),
                "gray_low_value": gray_payload.get("low_value"),
                "gray_high_value": gray_payload.get("high_value"),
                "scale_factor": scale_payload.get("scale_factor"),
                "major_extent_after_px": scale_payload.get("major_extent_after_px"),
                "crop_payload_json": json.dumps(crop_payload, ensure_ascii=False),
                "gray_payload_json": json.dumps(gray_payload, ensure_ascii=False),
                "scale_payload_json": json.dumps(scale_payload, ensure_ascii=False),
            }
        )

    records_df = pd.DataFrame(records)
    records_path = tables_dir / "texas_center_body_square_records_v1.csv"
    records_df.to_csv(records_path, index=False)

    fallback_df = records_df[records_df["sam_stage"].astype(str).ne("none")].copy()
    nonfallback_df = records_df[records_df["sam_stage"].astype(str).eq("none")].copy()
    sample_count = max(0, int(args.detail_sample_count) - len(fallback_df))
    if sample_count > 0 and len(nonfallback_df) > sample_count:
        sampled_nonfallback = nonfallback_df.sample(n=sample_count, random_state=int(args.sample_seed))
    else:
        sampled_nonfallback = nonfallback_df
    detail_df = pd.concat([fallback_df, sampled_nonfallback], ignore_index=True).sort_values(["sam_stage", "image_id"]).reset_index(drop=True)
    detail_records_path = tables_dir / "texas_center_body_square_detail_samples_v1.csv"
    detail_df.to_csv(detail_records_path, index=False)

    detail_board_path = qualitative_dir / "texas_center_body_square_detail_board_v1.jpg"
    full_grid_path = qualitative_dir / "texas_center_body_square_all_final_v1.jpg"
    _draw_detail_board(detail_df, output_path=detail_board_path)
    _draw_final_grid(records_df, output_path=full_grid_path)

    summary_lines = [
        "# Texas Center Body Square Qualitative Board",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{args.manifest_path}`",
        f"- Texas images: `{len(records_df)}`",
        f"- Detail board samples: `{len(detail_df)}`; includes all repaired fallback samples.",
        "",
        "## Outputs",
        "",
        f"- Detail board: `{detail_board_path}`",
        f"- Full final overview: `{full_grid_path}`",
        f"- Records table: `{records_path}`",
        f"- Detail sample table: `{detail_records_path}`",
        "",
        "## Stage Counts",
        "",
        records_df["sam_stage"].value_counts(dropna=False).rename_axis("sam_stage").reset_index(name="count").to_markdown(index=False),
        "",
        "## Crop Statistics",
        "",
        records_df[["foreground_ratio_of_subject", "foreground_ratio_in_crop", "square_side_px"]].describe().to_markdown(),
        "",
        "## Reading Notes",
        "",
        "- `*` in the full overview marks samples that were repaired by SAM prompt backoff.",
        "- Detail board columns are original, repaired SAM-aligned foreground, scale-normalized foreground, and final grayscale center body square.",
        "- Check whether the final column preserves dorsal black-dot patterns while suppressing head/tail/limbs.",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[texas_center_body_square_board] detail_board: {detail_board_path}")
    print(f"[texas_center_body_square_board] full_grid: {full_grid_path}")
    print(f"[texas_center_body_square_board] records: {records_path}")
    print(f"[texas_center_body_square_board] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
