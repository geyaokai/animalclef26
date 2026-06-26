#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_METADATA_PATH = Path("artifacts/manifests/v1/tables/metadata_enriched_v1.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_sam_prompt_backoff_probe_v1")
DEFAULT_DATASET = "TexasHornedLizards"
DEFAULT_PROMPTS = [
    "Texas horned lizard",
    "horned lizard",
    "lizard body",
    "lizard",
    "animal body",
    "animal",
]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def _make_contact_sheet(
    *,
    frame: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    columns: int = 2,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if frame.empty:
        return
    margin = 12
    header_h = 34
    label_h = 54
    panel_gap = 6
    panel_w, panel_h = thumb_size
    cell_w = panel_w * 3 + panel_gap * 2
    cell_h = panel_h + label_h
    rows = math.ceil(len(frame) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for idx, row in enumerate(frame.itertuples(index=False)):
        gx = idx % columns
        gy = idx // columns
        x = margin + gx * (cell_w + margin)
        y = start_y + gy * (cell_h + margin)
        original = ImageOps.pad(Image.open(repo_root / row.path).convert("RGB"), thumb_size, color=(10, 10, 10))
        overlay = ImageOps.pad(Image.open(repo_root / row.overlay_path).convert("RGB"), thumb_size, color=(10, 10, 10))
        masked = ImageOps.pad(Image.open(repo_root / row.masked_crop_path).convert("RGB"), thumb_size, color=(10, 10, 10))
        canvas.paste(original, (x, y))
        canvas.paste(overlay, (x + panel_w + panel_gap, y))
        canvas.paste(masked, (x + (panel_w + panel_gap) * 2, y))
        label = (
            f"{row.image_id} | {row.best_prompt}\n"
            f"mask={row.mask_count} area={row.mask_area_ratio:.3f} score={row.best_score:.3f}"
        )
        draw.multiline_text((x, y + panel_h + 4), label, fill=(30, 30, 30), font=font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.sam3_probe import (
        crop_to_union_mask,
        load_sam3,
        overlay_masks_on_image,
        run_single_inference,
    )

    parser = argparse.ArgumentParser(description="Rerun failed Texas SAM cases with prompt backoff.")
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    metadata_path = args.metadata_path if args.metadata_path.is_absolute() else (repo_root / args.metadata_path)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (repo_root / args.output_dir)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "masked_crops"
    qualitative_dir = output_dir / "qualitative"
    for path in [output_dir, tables_dir, reports_dir, overlays_dir, crops_dir, qualitative_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(metadata_path)
    metadata_df["image_id"] = metadata_df["image_id"].astype(str)
    metadata_df["dataset"] = metadata_df["dataset"].astype(str)
    metadata_df["path"] = metadata_df["path"].astype(str)
    failed_df = metadata_df[
        metadata_df["dataset"].eq(str(args.dataset))
        & metadata_df["sam_masked_rgb_v1_status"].astype(str).eq("skip")
    ].copy().sort_values("image_id").reset_index(drop=True)
    if int(args.limit) > 0:
        failed_df = failed_df.head(int(args.limit)).copy().reset_index(drop=True)

    resources = load_sam3(device=str(args.device))
    rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []

    for index, row in enumerate(failed_df.itertuples(index=False), start=1):
        image_path = repo_root / row.path
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            best_masks = None
            best_prompt = ""
            best_stats = {"mask_count": 0, "mask_area_ratio": 0.0, "best_score": 0.0}
            recovered = False
            for prompt_rank, prompt in enumerate(DEFAULT_PROMPTS, start=1):
                masks, stats = run_single_inference(
                    image=image,
                    prompt=prompt,
                    resources=resources,
                    threshold=float(args.threshold),
                    mask_threshold=float(args.mask_threshold),
                )
                attempt_rows.append(
                    {
                        "image_id": str(row.image_id),
                        "path": str(row.path),
                        "prompt_rank": int(prompt_rank),
                        "prompt": str(prompt),
                        "mask_count": int(stats.get("mask_count", 0)),
                        "mask_area_ratio": float(stats.get("mask_area_ratio", 0.0)),
                        "best_score": float(stats.get("best_score", 0.0)),
                    }
                )
                if masks is not None and int(stats.get("mask_count", 0)) > 0:
                    best_masks = masks
                    best_prompt = str(prompt)
                    best_stats = stats
                    recovered = True
                    break

            if best_masks is None:
                overlay = image.copy()
                masked_crop = image.copy()
            else:
                overlay = overlay_masks_on_image(image, best_masks)
                masked_crop = crop_to_union_mask(image, best_masks)

        overlay_rel = output_dir.relative_to(repo_root) / "overlays" / f"{row.image_id}.jpg"
        crop_rel = output_dir.relative_to(repo_root) / "masked_crops" / f"{row.image_id}.jpg"
        overlay.save(repo_root / overlay_rel, quality=92)
        masked_crop.save(repo_root / crop_rel, quality=92)
        rows.append(
            {
                "image_id": str(row.image_id),
                "split": str(row.split),
                "path": str(row.path),
                "original_reason": str(row.sam_masked_rgb_v1_reason),
                "recovered": bool(recovered),
                "best_prompt": str(best_prompt),
                "mask_count": int(best_stats.get("mask_count", 0)),
                "mask_area_ratio": float(best_stats.get("mask_area_ratio", 0.0)),
                "best_score": float(best_stats.get("best_score", 0.0)),
                "overlay_path": str(overlay_rel),
                "masked_crop_path": str(crop_rel),
            }
        )
        print(
            f"[texas_sam_prompt_backoff_probe] {index}/{len(failed_df)} | {row.image_id} | "
            f"recovered={recovered} | prompt={best_prompt or 'none'}",
            flush=True,
        )

    result_df = pd.DataFrame(rows).sort_values(["recovered", "image_id"], ascending=[False, True]).reset_index(drop=True)
    attempt_df = pd.DataFrame(attempt_rows).sort_values(["image_id", "prompt_rank"]).reset_index(drop=True)
    result_df.to_csv(tables_dir / "texas_failed_rerun_results_v1.csv", index=False)
    attempt_df.to_csv(tables_dir / "texas_failed_rerun_attempts_v1.csv", index=False)

    recovered_df = result_df[result_df["recovered"]].copy().reset_index(drop=True)
    unrecovered_df = result_df[~result_df["recovered"]].copy().reset_index(drop=True)
    _make_contact_sheet(
        frame=recovered_df.head(12),
        repo_root=repo_root,
        output_path=qualitative_dir / "texas_failed_recovered_v1.jpg",
        title="Texas failed SAM rerun | recovered | original / overlay / masked",
    )
    _make_contact_sheet(
        frame=unrecovered_df.head(12),
        repo_root=repo_root,
        output_path=qualitative_dir / "texas_failed_unrecovered_v1.jpg",
        title="Texas failed SAM rerun | unrecovered | original / overlay / masked",
    )

    prompt_summary_df = (
        recovered_df.groupby("best_prompt").size().reset_index(name="recovered_count").sort_values("recovered_count", ascending=False)
        if not recovered_df.empty
        else pd.DataFrame(columns=["best_prompt", "recovered_count"])
    )
    summary = {
        "dataset": str(args.dataset),
        "threshold": float(args.threshold),
        "mask_threshold": float(args.mask_threshold),
        "failed_input_count": int(len(failed_df)),
        "recovered_count": int(len(recovered_df)),
        "unrecovered_count": int(len(unrecovered_df)),
        "recovered_ratio": round(float(len(recovered_df) / max(len(failed_df), 1)), 6),
        "prompt_order": list(DEFAULT_PROMPTS),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Texas SAM Prompt Backoff Probe v1",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Metadata source: `{metadata_path.relative_to(repo_root)}`",
        f"- Threshold: `{float(args.threshold)}`",
        f"- Mask threshold: `{float(args.mask_threshold)}`",
        f"- Failed input count: `{int(len(failed_df))}`",
        f"- Recovered count: `{int(len(recovered_df))}`",
        f"- Unrecovered count: `{int(len(unrecovered_df))}`",
        f"- Recovered ratio: `{summary['recovered_ratio']:.4f}`",
        f"- Prompt order: `{ ' -> '.join(DEFAULT_PROMPTS) }`",
        "",
        "## Prompt Wins",
        "",
        _markdown_table(prompt_summary_df if not prompt_summary_df.empty else pd.DataFrame(columns=["best_prompt", "recovered_count"])),
        "",
        "## Top Recovered",
        "",
        _markdown_table(recovered_df.head(20)[["image_id", "best_prompt", "mask_count", "mask_area_ratio", "best_score"]]),
        "",
        "## Top Unrecovered",
        "",
        _markdown_table(unrecovered_df.head(20)[["image_id", "mask_count", "mask_area_ratio", "best_score"]]),
        "",
        f"![Recovered]({os.path.relpath((qualitative_dir / 'texas_failed_recovered_v1.jpg'), start=reports_dir).replace(os.sep, '/')})",
        "",
        f"![Unrecovered]({os.path.relpath((qualitative_dir / 'texas_failed_unrecovered_v1.jpg'), start=reports_dir).replace(os.sep, '/')})",
    ]
    (reports_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[texas_sam_prompt_backoff_probe] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
