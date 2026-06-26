from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .initial_audit import load_metadata
from .sam3_probe import (
    Sam3Resources,
    get_prompt_candidates_for_dataset,
    load_sam3,
    overlay_masks_on_image,
    run_single_inference_with_prompt_backoff,
    sample_rows_by_dataset,
)

try:  # pragma: no cover - optional in light test envs
    from scipy import ndimage
except ModuleNotFoundError:  # pragma: no cover
    ndimage = None


PROMPTS_BY_DATASET = {
    "LynxID2025": "lynx",
    "SalamanderID2025": "salamander body",
    "SeaTurtleID2022": "sea turtle",
    "TexasHornedLizards": "horned lizard body",
}

SKIP_DATASETS: set[str] = set()
DEFAULT_ALIGNED_CROP_PADDING_RATIO = 0.06
ALIGNED_CROP_PADDING_RATIO_OVERRIDES = {
    "TexasHornedLizards": 0.12,
}
DEFAULT_ROTATION_CANVAS_FILL_MODE = "edge"


@dataclass(frozen=True)
class OrientationDecision:
    status: str
    reason: str
    should_apply: bool


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def normalize_axis_angle_deg(angle_deg: float) -> float:
    return float(((angle_deg + 90.0) % 180.0) - 90.0)


def _largest_connected_component_numpy(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    best_component: list[tuple[int, int]] = []

    for y in range(height):
        for x in range(width):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] > 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            if len(component) > len(best_component):
                best_component = component

    largest = np.zeros_like(binary, dtype=np.uint8)
    for y, x in best_component:
        largest[y, x] = 1
    return largest


def extract_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return binary
    if ndimage is not None:
        labels, component_count = ndimage.label(binary)
        if component_count <= 1:
            return binary
        counts = np.bincount(labels.ravel())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        return (labels == largest_label).astype(np.uint8)
    return _largest_connected_component_numpy(binary)


def merge_masks(masks: np.ndarray | None) -> tuple[np.ndarray, dict[str, float]]:
    if masks is None or len(masks) == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return empty, {
            "mask_count": 0,
            "union_area_ratio": 0.0,
            "largest_component_ratio": 0.0,
            "foreground_pixels": 0.0,
        }
    union_mask = np.any(masks > 0, axis=0).astype(np.uint8)
    union_area = int(union_mask.sum())
    largest_mask = extract_largest_component(union_mask)
    largest_area = int(largest_mask.sum())
    return largest_mask, {
        "mask_count": int(masks.shape[0]),
        "union_area_ratio": round(float(union_area / union_mask.size), 6),
        "largest_component_ratio": round(float(largest_area / max(union_area, 1)), 6),
        "foreground_pixels": float(largest_area),
    }


def compute_body_axis(mask: np.ndarray) -> dict[str, float] | None:
    binary = (mask > 0).astype(np.uint8)
    coords_yx = np.column_stack(np.where(binary > 0))
    if len(coords_yx) < 2:
        return None

    coords_xy = np.stack([coords_yx[:, 1], coords_yx[:, 0]], axis=1).astype(np.float32)
    centroid = coords_xy.mean(axis=0)
    centered = coords_xy - centroid
    covariance = centered.T @ centered / max(len(coords_xy) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    major_vector = eigenvectors[:, 0]
    minor_vector = eigenvectors[:, 1]
    raw_angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0])))
    axis_angle = normalize_axis_angle_deg(raw_angle)
    axis_confidence = float((eigenvalues[0] - eigenvalues[1]) / max(eigenvalues[0] + eigenvalues[1], 1e-8))

    major_projection = centered @ major_vector
    minor_projection = centered @ minor_vector
    half_major = float(max(abs(major_projection.min()), abs(major_projection.max())))
    half_minor = float(max(abs(minor_projection.min()), abs(minor_projection.max())))
    axis_start = centroid - major_vector * half_major
    axis_end = centroid + major_vector * half_major
    bbox = mask_bbox(binary)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    bbox_area = max((x1 - x0 + 1) * (y1 - y0 + 1), 1)
    foreground_pixels = int(binary.sum())

    return {
        "centroid_x": round(float(centroid[0]), 4),
        "centroid_y": round(float(centroid[1]), 4),
        "axis_angle_deg": round(axis_angle, 4),
        "axis_confidence": round(axis_confidence, 6),
        "major_eigenvalue": round(float(eigenvalues[0]), 6),
        "minor_eigenvalue": round(float(eigenvalues[1]), 6),
        "axis_start_x": round(float(axis_start[0]), 4),
        "axis_start_y": round(float(axis_start[1]), 4),
        "axis_end_x": round(float(axis_end[0]), 4),
        "axis_end_y": round(float(axis_end[1]), 4),
        "major_extent_px": round(float(2.0 * half_major), 4),
        "minor_extent_px": round(float(2.0 * half_minor), 4),
        "foreground_area_ratio": round(float(foreground_pixels / binary.size), 6),
        "foreground_pixels": float(foreground_pixels),
        "bbox_fill_ratio": round(float(foreground_pixels / bbox_area), 6),
        "bbox_x0": float(x0),
        "bbox_y0": float(y0),
        "bbox_x1": float(x1),
        "bbox_y1": float(y1),
    }


def decide_orientation_application(
    axis_stats: dict[str, float] | None,
    *,
    min_foreground_pixels: int,
    min_area_ratio: float,
    max_area_ratio: float,
    min_axis_confidence: float,
    min_largest_component_ratio: float,
    largest_component_ratio: float,
) -> OrientationDecision:
    if axis_stats is None:
        return OrientationDecision(status="skip", reason="no_axis", should_apply=False)
    if axis_stats["foreground_pixels"] < min_foreground_pixels:
        return OrientationDecision(status="skip", reason="small_mask", should_apply=False)
    if axis_stats["foreground_area_ratio"] < min_area_ratio:
        return OrientationDecision(status="skip", reason="small_area_ratio", should_apply=False)
    if axis_stats["foreground_area_ratio"] > max_area_ratio:
        return OrientationDecision(status="skip", reason="large_area_ratio", should_apply=False)
    if largest_component_ratio < min_largest_component_ratio:
        return OrientationDecision(status="skip", reason="fragmented_mask", should_apply=False)
    if axis_stats["axis_confidence"] < min_axis_confidence:
        return OrientationDecision(status="skip", reason="low_axis_confidence", should_apply=False)
    return OrientationDecision(status="apply", reason="ok", should_apply=True)


def resolve_crop_padding_ratio(
    dataset: str,
    *,
    default_padding_ratio: float,
    padding_ratio_overrides: dict[str, float] | None = None,
) -> float:
    if padding_ratio_overrides is None:
        padding_ratio_overrides = ALIGNED_CROP_PADDING_RATIO_OVERRIDES
    return float(padding_ratio_overrides.get(dataset, default_padding_ratio))


def _rotate_image_and_mask(
    image: Image.Image,
    mask: np.ndarray,
    rotation_deg: float,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
    canvas_fill_mode: str = DEFAULT_ROTATION_CANVAS_FILL_MODE,
) -> tuple[Image.Image, np.ndarray]:
    image_rgb = image.convert("RGB")
    binary_mask = (mask > 0).astype(np.uint8)
    if canvas_fill_mode == "constant" or ndimage is None:
        mask_image = Image.fromarray(binary_mask * 255, mode="L")
        rotated_image = image_rgb.rotate(
            rotation_deg,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=background,
        )
        rotated_mask_image = mask_image.rotate(
            rotation_deg,
            resample=Image.Resampling.NEAREST,
            expand=True,
            fillcolor=0,
        )
        return rotated_image, (np.asarray(rotated_mask_image) > 0).astype(np.uint8)

    scipy_mode = {
        "edge": "nearest",
        "reflect": "reflect",
    }.get(canvas_fill_mode)
    if scipy_mode is None:
        raise ValueError(f"Unsupported canvas_fill_mode: {canvas_fill_mode}")

    image_array = np.asarray(image_rgb, dtype=np.float32)
    rotated_image_array = ndimage.rotate(
        image_array,
        rotation_deg,
        axes=(1, 0),
        reshape=True,
        order=1,
        mode=scipy_mode,
        prefilter=False,
    )
    rotated_mask_array = ndimage.rotate(
        binary_mask.astype(np.float32),
        rotation_deg,
        axes=(1, 0),
        reshape=True,
        order=0,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    rotated_image = Image.fromarray(np.clip(np.rint(rotated_image_array), 0, 255).astype(np.uint8), mode="RGB")
    rotated_mask = (rotated_mask_array > 0.5).astype(np.uint8)
    return rotated_image, rotated_mask


def rotation_to_horizontal(axis_angle_deg: float) -> float:
    return round(-float(axis_angle_deg), 4)


def rotation_to_vertical(axis_angle_deg: float) -> float:
    angle = float(axis_angle_deg)
    candidate_positive = 90.0 - angle
    candidate_negative = -90.0 - angle
    if abs(candidate_positive) <= abs(candidate_negative):
        return round(candidate_positive, 4)
    return round(candidate_negative, 4)


def crop_with_padding(mask: np.ndarray, padding_ratio: float = 0.06) -> tuple[int, int, int, int] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    height, width = mask.shape
    pad = max(4, int(round(max(x1 - x0 + 1, y1 - y0 + 1) * padding_ratio)))
    return (
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(width - 1, x1 + pad),
        min(height - 1, y1 + pad),
    )


def render_axis_overlay(
    image: Image.Image,
    mask: np.ndarray,
    axis_stats: dict[str, float] | None,
    decision: OrientationDecision,
) -> Image.Image:
    canvas = overlay_masks_on_image(image, np.expand_dims(mask.astype(np.uint8), axis=0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = mask_bbox(mask)
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 255, 255), width=3)
    if axis_stats is not None:
        draw.line(
            (
                axis_stats["axis_start_x"],
                axis_stats["axis_start_y"],
                axis_stats["axis_end_x"],
                axis_stats["axis_end_y"],
            ),
            fill=(255, 64, 64),
            width=4,
        )
        cx = axis_stats["centroid_x"]
        cy = axis_stats["centroid_y"]
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(255, 255, 0))
        text = (
            f"status={decision.status} | reason={decision.reason}\n"
            f"angle={axis_stats['axis_angle_deg']:.1f} | conf={axis_stats['axis_confidence']:.3f}"
        )
    else:
        text = f"status={decision.status} | reason={decision.reason}"
    draw.rectangle((8, 8, 8 + 6 * max(len(line) for line in text.splitlines()) + 12, 44), fill=(0, 0, 0))
    draw.multiline_text((14, 12), text, fill=(255, 255, 255), font=font, spacing=2)
    return canvas


def rotate_and_crop(
    image: Image.Image,
    mask: np.ndarray,
    rotation_deg: float,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
    padding_ratio: float = DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    keep_background: bool = True,
    canvas_fill_mode: str = DEFAULT_ROTATION_CANVAS_FILL_MODE,
) -> tuple[Image.Image, np.ndarray]:
    rotated_image, rotated_mask = _rotate_image_and_mask(
        image,
        mask,
        rotation_deg,
        background=background,
        canvas_fill_mode=canvas_fill_mode,
    )
    if keep_background:
        rotated_output = rotated_image
    else:
        arr = np.asarray(rotated_image).copy()
        arr[rotated_mask == 0] = np.array(background, dtype=np.uint8)
        rotated_output = Image.fromarray(arr)
    crop_box = crop_with_padding(rotated_mask, padding_ratio=padding_ratio)
    if crop_box is None:
        return rotated_output, rotated_mask
    x0, y0, x1, y1 = crop_box
    return rotated_output.crop((x0, y0, x1 + 1, y1 + 1)), rotated_mask[y0 : y1 + 1, x0 : x1 + 1]


def create_probe_contact_sheet(
    results_df: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    columns: int = 2,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if results_df.empty:
        return

    margin = 12
    header_h = 34
    label_h = 54
    panel_gap = 6
    panel_w, panel_h = thumb_size
    cell_w = panel_w * 4 + panel_gap * 3
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
            ImageOps.pad(Image.open(repo_root / row.path).convert("RGB"), thumb_size, color=(10, 10, 10)),
            ImageOps.pad(Image.open(repo_root / row.sam_overlay_path).convert("RGB"), thumb_size, color=(10, 10, 10)),
            ImageOps.pad(Image.open(repo_root / row.axis_overlay_path).convert("RGB"), thumb_size, color=(10, 10, 10)),
            ImageOps.pad(Image.open(repo_root / row.aligned_crop_path).convert("RGB"), thumb_size, color=(10, 10, 10)),
        ]
        for panel_index, panel in enumerate(panels):
            canvas.paste(panel, (x + panel_index * (panel_w + panel_gap), y))
        label = (
            f"{row.dataset} | {row.split} | {row.image_id}\n"
            f"status={row.orientation_status} | reason={row.orientation_reason}\n"
            f"angle={row.axis_angle_deg:.1f} | rot={row.rotation_applied_deg:.1f} | conf={row.axis_confidence:.3f}"
        )
        draw.multiline_text((x, y + panel_h + 4), label, fill=(30, 30, 30), font=font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def write_summary(
    results_df: pd.DataFrame,
    output_path: Path,
    *,
    datasets: list[str],
    model_path: Path,
    config: dict[str, Any],
) -> None:
    if results_df.empty:
        aggregate_df = pd.DataFrame(
            columns=[
                "dataset",
                "split",
                "samples",
                "positive_masks",
                "applied",
                "applied_ratio",
                "mean_area_ratio",
                "mean_axis_confidence",
                "median_abs_rotation_deg",
            ]
        )
        failure_df = pd.DataFrame(columns=["dataset", "orientation_reason", "count"])
    else:
        aggregate_df = (
            results_df.groupby(["dataset", "split"])
            .agg(
                samples=("image_id", "count"),
                positive_masks=("mask_count", lambda s: int((s > 0).sum())),
                applied=("orientation_status", lambda s: int((s == "apply").sum())),
                applied_ratio=("orientation_status", lambda s: round(float((s == "apply").mean()), 4)),
                mean_area_ratio=("foreground_area_ratio", lambda s: round(float(np.mean(s)), 4)),
                mean_axis_confidence=("axis_confidence", lambda s: round(float(np.mean(s)), 4)),
                median_abs_rotation_deg=("rotation_applied_deg", lambda s: round(float(np.median(np.abs(s))), 2)),
            )
            .reset_index()
            .sort_values(["dataset", "split"])
        )
        failure_df = (
            results_df.groupby(["dataset", "orientation_reason"])
            .size()
            .reset_index(name="count")
            .sort_values(["dataset", "count", "orientation_reason"], ascending=[True, False, True])
            .reset_index(drop=True)
        )

    lines = [
        "# Body Orientation Probe Summary",
        "",
        f"- Datasets: `{', '.join(datasets)}`",
        f"- Model source: `{model_path}`",
        f"- Target alignment: `major axis -> horizontal`",
        (
            f"- Aligned export: `{'rotate RGB + keep background' if config['keep_background'] else 'rotate masked crop'}`"
        ),
        f"- Rotation canvas fill: `{config['rotation_canvas_fill_mode']}`",
        (
            "- Crop padding: "
            f"`default={config['aligned_crop_padding_ratio']}`; "
            f"`overrides={config['aligned_crop_padding_ratio_overrides']}`"
        ),
        (
            "- Gate: "
            f"`min_foreground_pixels={config['min_foreground_pixels']}`, "
            f"`min_area_ratio={config['min_area_ratio']}`, "
            f"`max_area_ratio={config['max_area_ratio']}`, "
            f"`min_axis_confidence={config['min_axis_confidence']}`, "
            f"`min_largest_component_ratio={config['min_largest_component_ratio']}`"
        ),
        "",
        "## Aggregate Results",
        "",
        dataframe_to_markdown_table(aggregate_df),
        "",
        "## Failure Reasons",
        "",
        dataframe_to_markdown_table(failure_df),
        "",
        "## Panel Reading Guide",
        "",
        "- Panel 1: original image.",
        "- Panel 2: SAM3 mask overlay.",
        "- Panel 3: largest-component mask with PCA major axis and bbox.",
        (
            "- Panel 4: rotated aligned crop; by default it preserves RGB background "
            "inside the loose crop instead of zeroing non-mask pixels."
        ),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_body_orientation_probe(
    repo_root: Path,
    output_dir: Path,
    datasets: list[str],
    *,
    samples_per_split: int = 8,
    sample_seed: int = 42,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda:0",
    min_foreground_pixels: int = 1024,
    min_area_ratio: float = 0.015,
    max_area_ratio: float = 0.85,
    min_axis_confidence: float = 0.35,
    min_largest_component_ratio: float = 0.8,
    aligned_crop_padding_ratio: float = DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    aligned_crop_padding_ratio_overrides: dict[str, float] | None = None,
    keep_background: bool = True,
    rotation_canvas_fill_mode: str = DEFAULT_ROTATION_CANVAS_FILL_MODE,
) -> dict[str, Path]:
    metadata_df = load_metadata(repo_root / "metadata.csv")
    selected_datasets = [dataset for dataset in datasets if dataset not in SKIP_DATASETS]
    sampled_df = sample_rows_by_dataset(
        metadata_df=metadata_df[metadata_df["dataset"].isin(selected_datasets)],
        sample_seed=sample_seed,
        samples_per_split=samples_per_split,
        datasets=selected_datasets,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    sam_overlay_dir = output_dir / "sam_overlays"
    axis_overlay_dir = output_dir / "axis_overlays"
    aligned_crop_dir = output_dir / "aligned_crops"
    tables_dir = output_dir / "tables"
    qualitative_dir = output_dir / "qualitative"
    reports_dir = output_dir / "reports"
    for path in [sam_overlay_dir, axis_overlay_dir, aligned_crop_dir, tables_dir, qualitative_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    resources: Sam3Resources = load_sam3(device=device)
    rows: list[dict[str, Any]] = []
    total = len(sampled_df)
    for index, row in enumerate(sampled_df.itertuples(index=False), start=1):
        prompt_candidates = (
            ["horned lizard body", "Texas horned lizard body", "lizard body", "lizard", "animal body"]
            if str(row.dataset) == "TexasHornedLizards"
            else [PROMPTS_BY_DATASET[row.dataset]]
        )
        image_path = repo_root / row.path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            masks, stats = run_single_inference_with_prompt_backoff(
                image=image,
                prompts=prompt_candidates,
                resources=resources,
                threshold=threshold,
                mask_threshold=mask_threshold,
            )
            if masks is None:
                component_mask = np.zeros((image.height, image.width), dtype=np.uint8)
            else:
                component_mask, component_stats = merge_masks(masks)
                stats["largest_component_ratio"] = component_stats["largest_component_ratio"]
                stats["foreground_pixels"] = int(component_stats["foreground_pixels"])
                stats["union_area_ratio"] = component_stats["union_area_ratio"]
            if masks is None:
                stats["largest_component_ratio"] = 0.0
                stats["foreground_pixels"] = 0
                stats["union_area_ratio"] = 0.0

            axis_stats = compute_body_axis(component_mask)
            decision = decide_orientation_application(
                axis_stats,
                min_foreground_pixels=min_foreground_pixels,
                min_area_ratio=min_area_ratio,
                max_area_ratio=max_area_ratio,
                min_axis_confidence=min_axis_confidence,
                min_largest_component_ratio=min_largest_component_ratio,
                largest_component_ratio=float(stats["largest_component_ratio"]),
            )
            rotation_applied_deg = rotation_to_horizontal(axis_stats["axis_angle_deg"]) if axis_stats is not None else 0.0
            if not decision.should_apply:
                rotation_applied_deg = 0.0

            sam_overlay = overlay_masks_on_image(image, masks) if masks is not None else image.copy()
            axis_overlay = render_axis_overlay(image, component_mask, axis_stats, decision)
            if axis_stats is not None and decision.should_apply:
                crop_padding_ratio = resolve_crop_padding_ratio(
                    row.dataset,
                    default_padding_ratio=aligned_crop_padding_ratio,
                    padding_ratio_overrides=aligned_crop_padding_ratio_overrides,
                )
                aligned_crop, rotated_mask = rotate_and_crop(
                    image,
                    component_mask,
                    rotation_applied_deg,
                    padding_ratio=crop_padding_ratio,
                    keep_background=keep_background,
                    canvas_fill_mode=rotation_canvas_fill_mode,
                )
                aligned_foreground_ratio = round(float(rotated_mask.mean()), 6)
            else:
                aligned_crop = image.copy()
                aligned_foreground_ratio = round(float(component_mask.mean()), 6)
                crop_padding_ratio = resolve_crop_padding_ratio(
                    row.dataset,
                    default_padding_ratio=aligned_crop_padding_ratio,
                    padding_ratio_overrides=aligned_crop_padding_ratio_overrides,
                )

        sam_overlay_rel = output_dir.relative_to(repo_root) / "sam_overlays" / f"{row.image_id}.jpg"
        axis_overlay_rel = output_dir.relative_to(repo_root) / "axis_overlays" / f"{row.image_id}.jpg"
        aligned_crop_rel = output_dir.relative_to(repo_root) / "aligned_crops" / f"{row.image_id}.jpg"
        sam_overlay.save(repo_root / sam_overlay_rel, quality=92)
        axis_overlay.save(repo_root / axis_overlay_rel, quality=92)
        aligned_crop.save(repo_root / aligned_crop_rel, quality=92)

        rows.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "split": row.split,
                "orientation": row.orientation,
                "path": row.path,
                "prompt": str(stats.get("selected_prompt", prompt_candidates[0])),
                "mask_count": int(stats.get("mask_count", 0)),
                "mask_area_ratio": float(stats.get("mask_area_ratio", 0.0)),
                "union_area_ratio": float(stats.get("union_area_ratio", 0.0)),
                "best_score": float(stats.get("best_score", 0.0)),
                "largest_component_ratio": float(stats.get("largest_component_ratio", 0.0)),
                "foreground_pixels": float(axis_stats["foreground_pixels"]) if axis_stats is not None else 0.0,
                "foreground_area_ratio": float(axis_stats["foreground_area_ratio"]) if axis_stats is not None else 0.0,
                "bbox_fill_ratio": float(axis_stats["bbox_fill_ratio"]) if axis_stats is not None else 0.0,
                "axis_angle_deg": float(axis_stats["axis_angle_deg"]) if axis_stats is not None else 0.0,
                "axis_confidence": float(axis_stats["axis_confidence"]) if axis_stats is not None else 0.0,
                "major_extent_px": float(axis_stats["major_extent_px"]) if axis_stats is not None else 0.0,
                "minor_extent_px": float(axis_stats["minor_extent_px"]) if axis_stats is not None else 0.0,
                "rotation_applied_deg": float(rotation_applied_deg),
                "aligned_foreground_ratio": aligned_foreground_ratio,
                "aligned_crop_padding_ratio": float(crop_padding_ratio),
                "rotation_canvas_fill_mode": rotation_canvas_fill_mode,
                "orientation_status": decision.status,
                "orientation_reason": decision.reason,
                "sam_overlay_path": str(sam_overlay_rel),
                "axis_overlay_path": str(axis_overlay_rel),
                "aligned_crop_path": str(aligned_crop_rel),
            }
        )
        print(
            (
                f"[body_orientation_probe] {index}/{total} done | {row.dataset} | {row.image_id} | "
                f"masks={int(stats.get('mask_count', 0))} | status={decision.status} | reason={decision.reason}"
            ),
            flush=True,
        )

    results_df = pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    results_path = tables_dir / "body_orientation_probe_results.csv"
    results_df.to_csv(results_path, index=False)

    for dataset in selected_datasets:
        dataset_df = results_df[results_df["dataset"] == dataset]
        if dataset_df.empty:
            continue
        create_probe_contact_sheet(
            dataset_df,
            repo_root=repo_root,
            output_path=qualitative_dir / f"body_orientation_probe_{dataset}.jpg",
            title=f"Body Orientation Probe | {dataset} | original / sam / axis / aligned_rgb",
        )
        failure_df = dataset_df[dataset_df["orientation_status"] != "apply"]
        if not failure_df.empty:
            create_probe_contact_sheet(
                failure_df,
                repo_root=repo_root,
                output_path=qualitative_dir / f"body_orientation_failures_{dataset}.jpg",
                title=f"Body Orientation Probe Failures | {dataset}",
            )

    summary_config = {
        "samples_per_split": samples_per_split,
        "sample_seed": sample_seed,
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "device": device,
        "min_foreground_pixels": min_foreground_pixels,
        "min_area_ratio": min_area_ratio,
        "max_area_ratio": max_area_ratio,
        "min_axis_confidence": min_axis_confidence,
        "min_largest_component_ratio": min_largest_component_ratio,
        "aligned_crop_padding_ratio": aligned_crop_padding_ratio,
        "aligned_crop_padding_ratio_overrides": (
            aligned_crop_padding_ratio_overrides
            if aligned_crop_padding_ratio_overrides is not None
            else ALIGNED_CROP_PADDING_RATIO_OVERRIDES
        ),
        "keep_background": keep_background,
        "rotation_canvas_fill_mode": rotation_canvas_fill_mode,
    }
    write_summary(
        results_df=results_df,
        output_path=reports_dir / "summary.md",
        datasets=selected_datasets,
        model_path=resources.model_path,
        config=summary_config,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": selected_datasets,
                "model_path": str(resources.model_path),
                **summary_config,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "results_path": results_path,
        "summary_path": reports_dir / "summary.md",
        "qualitative_dir": qualitative_dir,
    }
