from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .sam_orb_veto import infer_mask_from_masked_rgb


DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_aligned_crop_review_v1")
DEFAULT_MANIFEST_PATH = Path(
    "artifacts/manifests/sam_seg_trainprep_v1/tables/manifest_train_sam_trainprep_aligned_best_v1.csv"
)
DEFAULT_DATASET = "SalamanderID2025"
DEFAULT_SPLIT = "train"
DEFAULT_SAMPLE_COUNT = 24
DEFAULT_SAMPLE_SEED = 42
DEFAULT_COLUMNS = 2
DEFAULT_THUMB_SIZE = (220, 220)

CROP_AREA_RATIO = 0.75
VERTICAL_PADDING_RATIO = 0.06
HORIZONTAL_PADDING_RATIO = 0.03

COLOR_CROP75 = (255, 196, 0)

ALIGNED_VIEWS_ROOT = Path("artifacts/manifests/sam_seg_trainprep_v1/views/sam_masked_aligned_trainprep_v1")


def _infer_bbox_from_aligned(image: Image.Image) -> tuple[int, int, int, int] | None:
    mask = infer_mask_from_masked_rgb(image, nonzero_threshold=1)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _clip_box(
    left: float,
    top: float,
    right: float,
    bottom: float,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    left_i = max(0, min(int(round(left)), width - 1))
    top_i = max(0, min(int(round(top)), height - 1))
    right_i = max(left_i + 1, min(int(round(right)), width))
    bottom_i = max(top_i + 1, min(int(round(bottom)), height))
    return left_i, top_i, right_i, bottom_i


def _build_center_crop_box(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    area_ratio: float = CROP_AREA_RATIO,
    vertical_padding_ratio: float = VERTICAL_PADDING_RATIO,
    horizontal_padding_ratio: float = HORIZONTAL_PADDING_RATIO,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    image_w, image_h = image_size
    bbox_w = max(1, x1 - x0 + 1)
    bbox_h = max(1, y1 - y0 + 1)
    crop_scale = math.sqrt(float(area_ratio))
    crop_w = float(bbox_w) * crop_scale
    crop_h = float(bbox_h) * crop_scale
    pad_x = float(bbox_w) * float(horizontal_padding_ratio)
    pad_y = float(bbox_h) * float(vertical_padding_ratio)
    center_x = x0 + float(bbox_w) / 2.0
    center_y = y0 + float(bbox_h) / 2.0
    crop_left = center_x - crop_w / 2.0 - pad_x
    crop_right = center_x + crop_w / 2.0 + pad_x
    crop_top = center_y - crop_h / 2.0 - pad_y
    crop_bottom = center_y + crop_h / 2.0 + pad_y
    return _clip_box(
        crop_left,
        crop_top,
        crop_right,
        crop_bottom,
        width=image_w,
        height=image_h,
    )


def _draw_box_on_aligned(image: Image.Image, *, crop_box: tuple[int, int, int, int]) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(crop_box, outline=COLOR_CROP75, width=4)
    draw.text((crop_box[0] + 4, max(0, crop_box[1] - 12)), "crop75", fill=COLOR_CROP75, font=ImageFont.load_default())
    return overlay


def _identity_count_bucket(count: int) -> str:
    if count <= 1:
        return "singleton"
    if count == 2:
        return "pair"
    if count <= 4:
        return "small_3_4"
    return "multi_5_plus"


def _sample_review_rows(df: pd.DataFrame, *, sample_count: int, seed: int) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    identity_counts = working.groupby("identity", dropna=False)["image_id"].transform("size").astype(int)
    working["identity_image_count_fit"] = identity_counts
    working["identity_count_bucket_v1"] = identity_counts.map(_identity_count_bucket)

    selected_parts: list[pd.DataFrame] = []
    used_identities: set[str] = set()
    bucket_order = ["singleton", "pair", "small_3_4", "multi_5_plus"]
    per_bucket = max(1, math.ceil(sample_count / max(len(bucket_order), 1)))

    for bucket_index, bucket_name in enumerate(bucket_order):
        bucket_df = working[working["identity_count_bucket_v1"] == bucket_name].copy()
        if bucket_df.empty:
            continue
        bucket_df = bucket_df.sample(frac=1.0, random_state=seed + bucket_index).reset_index(drop=True)
        dedup_bucket = bucket_df.loc[~bucket_df["identity"].astype(str).isin(used_identities)].copy()
        if dedup_bucket.empty:
            dedup_bucket = bucket_df
        chosen = dedup_bucket.head(per_bucket).copy()
        selected_parts.append(chosen)
        used_identities.update(chosen["identity"].astype(str).tolist())

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=working.columns)
    if len(selected) < sample_count:
        remainder = (
            working.loc[~working["image_id"].astype(str).isin(selected["image_id"].astype(str).tolist())]
            .sample(frac=1.0, random_state=seed + 99)
            .reset_index(drop=True)
        )
        remainder = remainder.loc[~remainder["identity"].astype(str).isin(used_identities)].copy()
        fill = remainder.head(sample_count - len(selected))
        if not fill.empty:
            selected = pd.concat([selected, fill], ignore_index=True)
    return selected.head(sample_count).reset_index(drop=True)


def create_crop_review_contact_sheet(
    results_df: pd.DataFrame,
    *,
    repo_root: Path,
    output_path: Path,
    title: str,
    columns: int = DEFAULT_COLUMNS,
    thumb_size: tuple[int, int] = DEFAULT_THUMB_SIZE,
) -> None:
    if results_df.empty:
        return

    margin = 12
    header_h = 34
    label_h = 66
    panel_gap = 6
    panel_w, panel_h = thumb_size
    panels_per_cell = 2
    cell_w = panel_w * panels_per_cell + panel_gap * (panels_per_cell - 1)
    cell_h = panel_h + label_h
    rows = math.ceil(len(results_df) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(246, 246, 246))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for idx, row in enumerate(results_df.itertuples(index=False)):
        gx = idx % columns
        gy = idx // columns
        x = margin + gx * (cell_w + margin)
        y = start_y + gy * (cell_h + margin)
        panels = [
            ImageOps.pad(Image.open(repo_root / row.overlay_path).convert("RGB"), thumb_size, color=(10, 10, 10)),
            ImageOps.pad(Image.open(repo_root / row.crop75_path).convert("RGB"), thumb_size, color=(10, 10, 10)),
        ]
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (x + panel_index * (panel_w + panel_gap), y))
        label = (
            f"{row.image_id} | {row.identity} | {row.orientation}\n"
            f"bucket={row.identity_count_bucket_v1} | count={row.identity_image_count_fit}\n"
            f"view order: overlay / crop75"
        )
        draw.multiline_text((x, y + panel_h + 4), label, fill=(30, 30, 30), font=font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def _resolve_output_leaf(source_path: str, image_id: str) -> Path:
    source = Path(str(source_path))
    try:
        return source.relative_to(ALIGNED_VIEWS_ROOT)
    except ValueError:
        suffix = source.suffix if source.suffix else ".jpg"
        dataset = source.parts[-4] if len(source.parts) >= 4 else DEFAULT_DATASET
        split = source.parts[-3] if len(source.parts) >= 3 else DEFAULT_SPLIT
        identity = source.parts[-2] if len(source.parts) >= 2 else "unknown"
        return Path(dataset) / split / identity / f"{image_id}{suffix}"


def _write_summary(
    sampled_df: pd.DataFrame,
    *,
    manifest_path: Path,
    output_path: Path,
    qualitative_paths: list[Path],
) -> None:
    bucket_summary = (
        sampled_df.groupby("identity_count_bucket_v1", dropna=False)
        .agg(
            images=("image_id", "size"),
            identities=("identity", pd.Series.nunique),
            mean_count=("identity_image_count_fit", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# Salamander Aligned Crop Review",
        "",
        "## Config",
        "",
        f"- Source manifest: `{manifest_path}`",
        f"- Dataset / split: `{DEFAULT_DATASET}` / `{DEFAULT_SPLIT}`",
        f"- Sample count: `{len(sampled_df)}`",
        "- Filter: `sam_trainprep_aligned_applied_v1 == True` only",
        f"- Crop variant: `crop75`",
        f"- Target area ratio: `{CROP_AREA_RATIO}`",
        f"- Padding ratios: `horizontal={HORIZONTAL_PADDING_RATIO}`, `vertical={VERTICAL_PADDING_RATIO}`",
        "",
        "## How To Read",
        "",
        "- Panel order is `overlay / crop75`.",
        "- `overlay` shows the aligned image with the single crop box.",
        "- `crop75` keeps the central 75% area of the inferred segmented-body box, with light padding.",
        "- First judge whether the crop lands on the salamander body instead of black padding.",
        "- Then judge whether this simple body crop is already usable as a singleton local-view baseline.",
        "",
        "## Sample Bucket Summary",
        "",
        _dataframe_to_markdown_table(bucket_summary),
        "",
        "## Qualitative Sheets",
        "",
    ]
    for qualitative_path in qualitative_paths:
        rel = qualitative_path.relative_to(output_path.parent.parent)
        lines.extend([f"### `{qualitative_path.stem}`", "", f"![{qualitative_path.stem}]({rel.as_posix()})", ""])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_salamander_aligned_crop_review(
    *,
    repo_root: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    output_dir = (repo_root / output_dir).resolve() if not output_dir.is_absolute() else output_dir.resolve()
    manifest_path = (repo_root / manifest_path).resolve() if not manifest_path.is_absolute() else manifest_path.resolve()

    manifest_df = pd.read_csv(manifest_path, low_memory=False)
    subset = manifest_df[(manifest_df["dataset"].astype(str) == str(dataset)) & (manifest_df["split"].astype(str) == str(split))].copy()
    if "sam_trainprep_aligned_applied_v1" in subset.columns:
        subset = subset[subset["sam_trainprep_aligned_applied_v1"].fillna(False).astype(bool)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dataset={dataset}, split={split} in {manifest_path}")

    sampled_df = _sample_review_rows(subset, sample_count=int(sample_count), seed=int(sample_seed))
    if sampled_df.empty:
        raise ValueError("Sampling returned no rows")

    views_dir = output_dir / "views"
    overlays_dir = views_dir / "box_overlays"
    crop75_dir = views_dir / "crop75"
    tables_dir = output_dir / "tables"
    qualitative_dir = output_dir / "qualitative"
    reports_dir = output_dir / "reports"
    for path in [overlays_dir, crop75_dir, tables_dir, qualitative_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in sampled_df.itertuples(index=False):
        image_path = repo_root / str(row.sam_trainprep_aligned_resolved_path_v1)
        with Image.open(image_path) as handle:
            aligned_image = handle.convert("RGB")
        bbox = _infer_bbox_from_aligned(aligned_image)
        if bbox is None:
            continue
        crop75_box = _build_center_crop_box(bbox, aligned_image.size)

        overlay_image = _draw_box_on_aligned(aligned_image, crop_box=crop75_box)
        crop75 = aligned_image.crop(crop75_box)

        rel_stem = _resolve_output_leaf(str(row.sam_trainprep_aligned_resolved_path_v1), str(row.image_id))
        overlay_rel = output_dir.relative_to(repo_root) / "views" / "box_overlays" / rel_stem
        crop75_rel = output_dir.relative_to(repo_root) / "views" / "crop75" / rel_stem

        for rel_path in [overlay_rel, crop75_rel]:
            (repo_root / rel_path).parent.mkdir(parents=True, exist_ok=True)
        overlay_image.save(repo_root / overlay_rel, quality=92)
        crop75.save(repo_root / crop75_rel, quality=92)

        rows.append(
            {
                "image_id": str(row.image_id),
                "identity": str(getattr(row, "identity", "") or ""),
                "orientation": str(getattr(row, "orientation", "") or ""),
                "identity_image_count_fit": int(getattr(row, "identity_image_count_fit")),
                "identity_count_bucket_v1": str(getattr(row, "identity_count_bucket_v1")),
                "sam_trainprep_aligned_resolved_path_v1": str(row.sam_trainprep_aligned_resolved_path_v1),
                "overlay_path": overlay_rel.as_posix(),
                "crop75_path": crop75_rel.as_posix(),
                "crop75_box": str(crop75_box),
            }
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        raise ValueError("No valid crop results were produced")

    sampled_path = tables_dir / "sampled_crop_rows_v1.csv"
    results_df.to_csv(sampled_path, index=False)

    contact_sheet_path = qualitative_dir / "aligned_crop_candidates_train_v1.jpg"
    create_crop_review_contact_sheet(
        results_df,
        repo_root=repo_root,
        output_path=contact_sheet_path,
        title="Salamander aligned crop candidates | overlay / crop75",
    )

    summary_path = reports_dir / "summary.md"
    _write_summary(
        results_df,
        manifest_path=manifest_path,
        output_path=summary_path,
        qualitative_paths=[contact_sheet_path],
    )
    return {
        "summary_path": summary_path,
        "sampled_rows_path": sampled_path,
        "qualitative_dir": qualitative_dir,
    }
