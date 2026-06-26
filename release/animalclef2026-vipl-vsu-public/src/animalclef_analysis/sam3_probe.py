from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .initial_audit import load_metadata


PROMPTS_BY_DATASET = {
    "LynxID2025": "lynx",
    "SalamanderID2025": "salamander",
    "SeaTurtleID2022": "sea turtle head",
    "TexasHornedLizards": "horned lizard",
}

PROMPT_CANDIDATES_BY_DATASET = {
    "LynxID2025": [
        "lynx",
        "wild cat",
        "cat",
        "animal",
    ],
    "SalamanderID2025": [
        "salamander",
        "salamander body",
        "animal",
    ],
    "SeaTurtleID2022": [
        "sea turtle",
        "sea turtle body",
        "turtle",
        "animal",
    ],
    "TexasHornedLizards": [
        "horned lizard",
        "Texas horned lizard",
        "lizard body",
        "lizard",
        "animal",
    ],
}

SKIP_DATASETS: set[str] = set()


@dataclass
class Sam3Resources:
    processor: Any
    model: Any
    device: str
    model_path: Path


def resolve_sam3_snapshot(cache_root: Path | None = None) -> Path:
    if cache_root is None:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--sam3"
    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = cache_root / "snapshots" / revision
        if snapshot.exists():
            return snapshot
    snapshots_dir = cache_root / "snapshots"
    candidates = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No local SAM3 snapshot found under {cache_root}")
    return candidates[0]


def load_sam3(device: str = "cuda:0") -> Sam3Resources:
    from transformers import Sam3Model, Sam3Processor

    model_path = resolve_sam3_snapshot()
    processor = Sam3Processor.from_pretrained(model_path)
    model = Sam3Model.from_pretrained(model_path)
    model = model.to(device)
    model.eval()
    return Sam3Resources(processor=processor, model=model, device=device, model_path=model_path)


def sample_rows_by_dataset(
    metadata_df: pd.DataFrame,
    sample_seed: int,
    samples_per_split: int,
    datasets: list[str],
) -> pd.DataFrame:
    rng = random.Random(sample_seed)
    rows: list[pd.DataFrame] = []
    for dataset in datasets:
        dataset_df = metadata_df[metadata_df["dataset"] == dataset].copy()
        for split in sorted(dataset_df["split"].unique()):
            split_df = dataset_df[dataset_df["split"] == split]
            if split_df.empty:
                continue
            size = min(samples_per_split, len(split_df))
            indices = rng.sample(list(split_df.index), k=size)
            rows.append(split_df.loc[indices].copy())
    if not rows:
        return pd.DataFrame(columns=metadata_df.columns)
    return pd.concat(rows, ignore_index=True)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def overlay_masks_on_image(image: Image.Image, masks: np.ndarray) -> Image.Image:
    image_rgba = image.convert("RGBA")
    colors = [
        (255, 0, 0, 110),
        (0, 180, 0, 110),
        (0, 90, 255, 110),
        (255, 180, 0, 110),
    ]
    for idx, mask in enumerate(masks):
        color = colors[idx % len(colors)]
        overlay = Image.new("RGBA", image.size, color=(0, 0, 0, 0))
        alpha = Image.fromarray((mask > 0).astype(np.uint8) * color[3], mode="L")
        fill = Image.new("RGBA", image.size, color=color[:3] + (0,))
        fill.putalpha(alpha)
        image_rgba = Image.alpha_composite(image_rgba, fill)
    return image_rgba.convert("RGB")


def crop_to_union_mask(image: Image.Image, masks: np.ndarray, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    union = np.any(masks > 0, axis=0)
    bbox = mask_bbox(union.astype(np.uint8))
    if bbox is None:
        return image.copy()
    x0, y0, x1, y1 = bbox
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~union] = np.array(background, dtype=np.uint8)
    cropped = Image.fromarray(arr).crop((x0, y0, x1 + 1, y1 + 1))
    return cropped


def run_single_inference(
    image: Image.Image,
    prompt: str,
    resources: Sam3Resources,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> tuple[np.ndarray | None, dict[str, object]]:
    import torch

    processor = resources.processor
    model = resources.model
    device = resources.device
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    processed = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]
    masks = processed.get("masks")
    boxes = processed.get("boxes")
    scores = processed.get("scores")
    if masks is None or len(masks) == 0:
        return None, {"mask_count": 0, "mask_area_ratio": 0.0, "best_score": 0.0, "boxes": []}

    mask_array = masks.detach().cpu().numpy().astype(np.uint8)
    union = np.any(mask_array > 0, axis=0)
    area_ratio = float(union.mean())
    best_score = float(scores.max().item()) if scores is not None and len(scores) else 0.0
    box_list = boxes.detach().cpu().numpy().tolist() if boxes is not None else []
    return mask_array, {
        "mask_count": int(mask_array.shape[0]),
        "mask_area_ratio": round(area_ratio, 6),
        "best_score": round(best_score, 6),
        "boxes": box_list,
    }


def get_prompt_candidates_for_dataset(dataset: str) -> list[str]:
    candidates = PROMPT_CANDIDATES_BY_DATASET.get(str(dataset))
    if candidates:
        return [str(value) for value in candidates]
    prompt = PROMPTS_BY_DATASET.get(str(dataset), str(dataset))
    return [str(prompt)]


def run_single_inference_with_prompt_backoff(
    image: Image.Image,
    prompts: str | list[str],
    resources: Sam3Resources,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> tuple[np.ndarray | None, dict[str, object]]:
    prompt_list = [str(prompts)] if isinstance(prompts, str) else [str(value) for value in prompts]
    if not prompt_list:
        raise ValueError("prompts must not be empty")

    best_failure_stats: dict[str, object] = {
        "mask_count": 0,
        "mask_area_ratio": 0.0,
        "best_score": 0.0,
        "boxes": [],
        "selected_prompt": "",
        "prompt_rank": 0,
        "attempted_prompt_count": len(prompt_list),
    }
    for prompt_rank, prompt in enumerate(prompt_list, start=1):
        masks, stats = run_single_inference(
            image=image,
            prompt=prompt,
            resources=resources,
            threshold=threshold,
            mask_threshold=mask_threshold,
        )
        stats = dict(stats)
        stats["selected_prompt"] = str(prompt)
        stats["prompt_rank"] = int(prompt_rank)
        stats["attempted_prompt_count"] = int(len(prompt_list))
        if masks is not None and int(stats.get("mask_count", 0)) > 0:
            return masks, stats
        if float(stats.get("best_score", 0.0)) >= float(best_failure_stats.get("best_score", 0.0)):
            best_failure_stats = stats
    return None, best_failure_stats


def create_triptych_contact_sheet(
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
    label_h = 40
    panel_gap = 6
    panel_w, panel_h = thumb_size
    cell_w = panel_w * 3 + panel_gap * 2
    cell_h = panel_h + label_h
    rows = math.ceil(len(results_df) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for idx, row in enumerate(results_df.itertuples(index=False)):
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
        label = f"{row.dataset} | {row.split} | masks={row.mask_count} | area={row.mask_area_ratio:.3f}"
        draw.text((x, y + panel_h + 4), label, fill=(30, 30, 30), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_summary(results_df: pd.DataFrame, output_path: Path, skipped: list[str], model_path: Path) -> None:
    if results_df.empty:
        summary_df = pd.DataFrame(columns=["dataset", "split", "samples", "positive_masks", "positive_ratio", "mean_area_ratio", "mean_best_score"])
    else:
        summary_df = (
            results_df.groupby(["dataset", "split"])
            .agg(
                samples=("image_id", "count"),
                positive_masks=("mask_count", lambda s: int((s > 0).sum())),
                positive_ratio=("mask_count", lambda s: round(float((s > 0).mean()), 4)),
                mean_area_ratio=("mask_area_ratio", lambda s: round(float(np.mean(s)), 4)),
                mean_best_score=("best_score", lambda s: round(float(np.mean(s)), 4)),
            )
            .reset_index()
            .sort_values(["dataset", "split"])
        )

    def as_markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in frame.iterrows()
        ]
        return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])

    lines = [
        "# SAM3 Probe Summary",
        "",
        f"- Model source: `{model_path}`",
        f"- Skipped datasets: `{', '.join(skipped) if skipped else 'none'}`",
        "",
        "## Aggregate Results",
        "",
        as_markdown_table(summary_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_sam3_probe(
    repo_root: Path,
    output_dir: Path,
    datasets: list[str],
    samples_per_split: int = 6,
    sample_seed: int = 42,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "cuda:0",
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
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "masked_crops"
    qualitative_dir = output_dir / "qualitative"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [overlays_dir, crops_dir, qualitative_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    resources = load_sam3(device=device)
    rows: list[dict[str, object]] = []
    total = len(sampled_df)
    for index, row in enumerate(sampled_df.itertuples(index=False), start=1):
        prompt_candidates = get_prompt_candidates_for_dataset(str(row.dataset))
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
                overlay = image.copy()
                masked_crop = image.copy()
            else:
                overlay = overlay_masks_on_image(image, masks)
                masked_crop = crop_to_union_mask(image, masks)

        overlay_rel = output_dir.relative_to(repo_root) / "overlays" / f"{row.image_id}.jpg"
        crop_rel = output_dir.relative_to(repo_root) / "masked_crops" / f"{row.image_id}.jpg"
        overlay_path = repo_root / overlay_rel
        crop_path = repo_root / crop_rel
        overlay.save(overlay_path, quality=92)
        masked_crop.save(crop_path, quality=92)
        rows.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "split": row.split,
                "orientation": row.orientation,
                "path": row.path,
                "prompt": str(stats.get("selected_prompt", prompt_candidates[0])),
                "mask_count": stats["mask_count"],
                "mask_area_ratio": stats["mask_area_ratio"],
                "best_score": stats["best_score"],
                "overlay_path": str(overlay_rel),
                "masked_crop_path": str(crop_rel),
            }
        )
        print(f"[sam3_probe] {index}/{total} done | {row.dataset} | {row.image_id} | masks={stats['mask_count']}")

    results_df = pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    results_path = tables_dir / "sam3_probe_results.csv"
    results_df.to_csv(results_path, index=False)

    for dataset in selected_datasets:
        dataset_results = results_df[results_df["dataset"] == dataset]
        if dataset_results.empty:
            continue
        create_triptych_contact_sheet(
            dataset_results,
            repo_root=repo_root,
            output_path=qualitative_dir / f"sam3_probe_{dataset}.jpg",
            title=f"SAM3 Probe | {dataset} | original / overlay / masked",
        )

    write_summary(
        results_df=results_df,
        output_path=reports_dir / "summary.md",
        skipped=sorted(SKIP_DATASETS),
        model_path=resources.model_path,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "datasets": selected_datasets,
                "samples_per_split": samples_per_split,
                "threshold": threshold,
                "mask_threshold": mask_threshold,
                "device": device,
                "model_path": str(resources.model_path),
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
