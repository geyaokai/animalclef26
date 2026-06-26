from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .body_orientation_probe import (
    ALIGNED_CROP_PADDING_RATIO_OVERRIDES,
    DEFAULT_ALIGNED_CROP_PADDING_RATIO,
    compute_body_axis,
    resolve_crop_padding_ratio,
    rotate_and_crop,
    rotation_to_horizontal,
)
from .sam3_probe import (
    PROMPTS_BY_DATASET,
    Sam3Resources,
    crop_to_union_mask,
    get_prompt_candidates_for_dataset,
    load_sam3,
    mask_bbox,
    run_single_inference_with_prompt_backoff,
)
from .sam_orb_veto import infer_mask_from_masked_rgb
from .view_manifests import PATH_COLUMN, dataframe_to_markdown_table


DEFAULT_SOURCE_MANIFEST_ROOT = Path("artifacts/manifests/v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/manifests/sam_seg_trainprep_v1")
DEFAULT_TARGET_DATASETS = ("SalamanderID2025", "TexasHornedLizards")
DEFAULT_TEXAS_FALLBACK_ENABLED = True
DEFAULT_TEXAS_FALLBACK_THRESHOLD = 0.35
DEFAULT_TEXAS_FALLBACK_MASK_THRESHOLD = 0.30
DEFAULT_TEXAS_FALLBACK_MIN_AREA_RATIO = 0.005
DEFAULT_TEXAS_FALLBACK_MAX_AREA_RATIO = 0.98
DEFAULT_TEXAS_FALLBACK_MIN_LARGEST_COMPONENT_RATIO = 0.40
DEFAULT_TEXAS_FALLBACK_DEVICE = "cuda:0"
DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS = 256
DEFAULT_ALIGNMENT_MIN_AREA_RATIO = 0.01
DEFAULT_ALIGNMENT_MAX_AREA_RATIO = 0.95
DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE = 0.18
DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE_OVERRIDES = {
    "TexasHornedLizards": 0.10,
}
DEFAULT_ALIGNMENT_PADDING_RATIO = DEFAULT_ALIGNED_CROP_PADDING_RATIO
DEFAULT_YOLO_FALLBACK_ENABLED = True
DEFAULT_YOLO_FALLBACK_MODEL = "yolov8s-worldv2.pt"
DEFAULT_YOLO_FALLBACK_CONF = 0.05
DEFAULT_YOLO_FALLBACK_IOU = 0.50
DEFAULT_YOLO_FALLBACK_IMGSZ = 640
DEFAULT_YOLO_FALLBACK_MAX_DET = 8
DEFAULT_GEOMETRIC_FALLBACK_ENABLED = True

MASKED_VIEW_NAME = "sam_masked_trainprep_v1"
ALIGNED_VIEW_NAME = "sam_masked_aligned_trainprep_v1"
TRAIN_SINGLE_MASKED_VIEW_NAME = "sam_trainprep_masked_best_v1"
TRAIN_SINGLE_ALIGNED_VIEW_NAME = "sam_trainprep_aligned_best_v1"
TRAIN_MULTIVIEW_NAME = "sam_trainprep_multiview_v1"

EXPORT_JOIN_KEYS = ["image_id", "dataset", "split", "original_rgb_path_v1"]


def _load_export_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in EXPORT_JOIN_KEYS:
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    return frame


def _merge_export_frames(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty and new_df.empty:
        return pd.DataFrame()
    if existing_df.empty:
        merged = new_df.copy()
    elif new_df.empty:
        merged = existing_df.copy()
    else:
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    for column in EXPORT_JOIN_KEYS:
        if column in merged.columns:
            merged[column] = merged[column].astype(str)
    if all(column in merged.columns for column in EXPORT_JOIN_KEYS):
        merged = merged.drop_duplicates(subset=EXPORT_JOIN_KEYS, keep="last")
        merged = merged.sort_values(EXPORT_JOIN_KEYS).reset_index(drop=True)
    return merged


def _pending_rows(reference_df: pd.DataFrame, existing_df: pd.DataFrame) -> pd.DataFrame:
    if reference_df.empty:
        return reference_df.copy()
    working = reference_df.copy()
    for column in EXPORT_JOIN_KEYS:
        if column in working.columns:
            working[column] = working[column].astype(str)
    if existing_df.empty:
        return working.reset_index(drop=True)
    pending = working.merge(
        existing_df[EXPORT_JOIN_KEYS].drop_duplicates(),
        on=EXPORT_JOIN_KEYS,
        how="left",
        indicator=True,
    )
    return pending[pending["_merge"] == "left_only"][reference_df.columns].reset_index(drop=True)


def _resolve_relative_view_path(original_path: str, *, suffix: str = ".jpg") -> Path:
    image_relative = Path(str(original_path)).relative_to("images")
    if suffix.startswith("."):
        return image_relative.with_suffix(suffix)
    return image_relative.with_name(f"{image_relative.stem}{suffix}")


def _load_base_metadata(source_manifest_root: Path) -> pd.DataFrame:
    metadata_path = source_manifest_root / "tables" / "metadata_enriched_v1.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    df["image_id"] = df["image_id"].astype(str)
    df["identity"] = df.get("identity", "").fillna("").astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["split"] = df["split"].astype(str)
    df["original_rgb_path_v1"] = df.get("original_rgb_path_v1", df["path"]).astype(str)
    return df


def _iter_fallback_attempts(threshold: float, mask_threshold: float) -> list[tuple[float, float]]:
    candidates = [
        (float(threshold), float(mask_threshold)),
        (min(float(threshold), 0.30), min(float(mask_threshold), 0.25)),
        (min(float(threshold), 0.25), min(float(mask_threshold), 0.20)),
    ]
    attempts: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for det_thr, seg_thr in candidates:
        key = (round(det_thr, 4), round(seg_thr, 4))
        if key in seen:
            continue
        seen.add(key)
        attempts.append((float(det_thr), float(seg_thr)))
    return attempts


def _yolo_prompt_candidates_for_dataset(dataset: str) -> list[str]:
    prompt_map = {
        "LynxID2025": ["lynx", "wild cat", "cat", "animal"],
        "SalamanderID2025": ["fire salamander", "salamander", "amphibian", "animal"],
        "SeaTurtleID2022": ["sea turtle", "turtle", "animal"],
        "TexasHornedLizards": ["Texas horned lizard", "horned lizard", "lizard", "reptile", "animal"],
    }
    return [str(value) for value in prompt_map.get(str(dataset), [str(dataset), "animal"])]


def _load_yolo_world(runtime: dict[str, Any], *, model_name: str, device: str | None) -> Any | None:
    cache_key = f"yolo::{model_name}::{device or 'default'}"
    if cache_key in runtime:
        return runtime[cache_key]
    try:
        from ultralytics import YOLO
    except Exception:
        runtime[cache_key] = None
        return None
    model_path = Path(model_name)
    model_arg = str(model_path if model_path.exists() else model_name)
    model = YOLO(model_arg)
    if device:
        try:
            model.to(device)
        except Exception:
            pass
    runtime[cache_key] = model
    return model


def _mask_from_box(
    *,
    image_size: tuple[int, int],
    xyxy: list[float],
    padding_ratio: float = 0.08,
) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    x0, y0, x1, y1 = [float(value) for value in xyxy]
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)
    pad_x = box_width * float(padding_ratio)
    pad_y = box_height * float(padding_ratio)
    left = max(0, int(np.floor(x0 - pad_x)))
    top = max(0, int(np.floor(y0 - pad_y)))
    right = min(width, int(np.ceil(x1 + pad_x)))
    bottom = min(height, int(np.ceil(y1 + pad_y)))
    mask = np.zeros((height, width), dtype=np.uint8)
    if right > left and bottom > top:
        mask[top:bottom, left:right] = 1
    return mask


def _geometric_center_mask(image_size: tuple[int, int], *, dataset: str) -> np.ndarray:
    width, height = int(image_size[0]), int(image_size[1])
    yy, xx = np.mgrid[0:height, 0:width]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    if str(dataset) in {"SalamanderID2025", "SeaTurtleID2022"}:
        radius_x = max(width * 0.42, 1.0)
        radius_y = max(height * 0.46, 1.0)
    else:
        radius_x = max(width * 0.40, 1.0)
        radius_y = max(height * 0.40, 1.0)
    ellipse = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
    return ellipse.astype(np.uint8)


def _crop_to_binary_mask(
    image: Image.Image,
    mask: np.ndarray,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    binary_mask = (np.asarray(mask) > 0)
    bbox = mask_bbox(binary_mask.astype(np.uint8))
    if bbox is None:
        return image.copy()
    x0, y0, x1, y1 = bbox
    arr = np.asarray(image.convert("RGB")).copy()
    arr[~binary_mask] = np.array(background, dtype=np.uint8)
    return Image.fromarray(arr).crop((x0, y0, x1 + 1, y1 + 1))


def _load_sam_resources(runtime: dict[str, Any], device: str) -> Sam3Resources:
    resources = runtime.get(device)
    if resources is not None:
        return resources
    try:
        resources = load_sam3(device=device)
        runtime[device] = resources
        return resources
    except Exception as exc:
        message = str(exc).lower()
        if "out of memory" not in message or device.lower() == "cpu":
            raise
        cpu_resources = runtime.get("cpu")
        if cpu_resources is None:
            cpu_resources = load_sam3(device="cpu")
            runtime["cpu"] = cpu_resources
        return cpu_resources


def _run_dataset_fallback_mask(
    *,
    repo_root: Path,
    output_dir: Path,
    runtime: dict[str, Any],
    dataset: str,
    original_path: str,
    threshold: float,
    mask_threshold: float,
    min_area_ratio: float,
    max_area_ratio: float,
    min_largest_component_ratio: float,
    device: str,
    yolo_fallback_enabled: bool = DEFAULT_YOLO_FALLBACK_ENABLED,
    yolo_model_name: str = DEFAULT_YOLO_FALLBACK_MODEL,
    yolo_conf: float = DEFAULT_YOLO_FALLBACK_CONF,
    yolo_iou: float = DEFAULT_YOLO_FALLBACK_IOU,
    yolo_imgsz: int = DEFAULT_YOLO_FALLBACK_IMGSZ,
    yolo_max_det: int = DEFAULT_YOLO_FALLBACK_MAX_DET,
    geometric_fallback_enabled: bool = DEFAULT_GEOMETRIC_FALLBACK_ENABLED,
) -> tuple[str, dict[str, Any]] | tuple[None, dict[str, Any]]:
    image_path = repo_root / original_path
    payload: dict[str, Any] = {
        "sam_trainprep_masked_status_v1": "skip",
        "sam_trainprep_masked_reason_v1": "no_mask",
        "sam_trainprep_masked_source_v1": "fallback",
        "sam_trainprep_masked_prompt_used_v1": "",
        "sam_trainprep_masked_prompt_rank_v1": 0,
        "sam_trainprep_masked_fallback_stage_v1": "sam",
        "sam_trainprep_masked_mask_count_v1": 0,
        "sam_trainprep_masked_union_area_ratio_v1": 0.0,
        "sam_trainprep_masked_largest_component_ratio_v1": 0.0,
        "sam_trainprep_masked_best_score_v1": 0.0,
        "sam_trainprep_masked_threshold_used_v1": 0.0,
        "sam_trainprep_masked_mask_threshold_used_v1": 0.0,
    }
    if not image_path.exists():
        payload["sam_trainprep_masked_reason_v1"] = "image_missing"
        return None, payload
    resources = _load_sam_resources(runtime=runtime, device=device)
    with Image.open(image_path) as image_handle:
        image = image_handle.convert("RGB")
        final_reason = "no_mask"
        masked_crop: Image.Image | None = None
        for attempt_index, (attempt_thr, attempt_mask_thr) in enumerate(
            _iter_fallback_attempts(threshold=threshold, mask_threshold=mask_threshold),
            start=1,
        ):
            masks, stats = run_single_inference_with_prompt_backoff(
                image=image,
                prompts=get_prompt_candidates_for_dataset(str(dataset)),
                resources=resources,
                threshold=float(attempt_thr),
                mask_threshold=float(attempt_mask_thr),
            )
            payload["sam_trainprep_masked_mask_count_v1"] = int(stats.get("mask_count", 0))
            payload["sam_trainprep_masked_best_score_v1"] = float(stats.get("best_score", 0.0))
            payload["sam_trainprep_masked_threshold_used_v1"] = float(attempt_thr)
            payload["sam_trainprep_masked_mask_threshold_used_v1"] = float(attempt_mask_thr)
            payload["sam_trainprep_masked_prompt_used_v1"] = str(stats.get("selected_prompt", "") or "")
            payload["sam_trainprep_masked_prompt_rank_v1"] = int(stats.get("prompt_rank", 0) or 0)
            if masks is None:
                final_reason = f"no_mask_try_{attempt_index}"
                continue
            union_mask = np.any(masks > 0, axis=0).astype(np.uint8)
            union_area = int(union_mask.sum())
            if union_area <= 0:
                final_reason = f"empty_union_try_{attempt_index}"
                continue
            inferred_mask = infer_mask_from_masked_rgb(crop_to_union_mask(image, masks).convert("RGB"), nonzero_threshold=1)
            largest_area = int(inferred_mask.sum())
            union_area_ratio = float(union_area / max(union_mask.size, 1))
            largest_component_ratio = float(largest_area / max(union_area, 1))
            payload["sam_trainprep_masked_union_area_ratio_v1"] = union_area_ratio
            payload["sam_trainprep_masked_largest_component_ratio_v1"] = largest_component_ratio
            if union_area_ratio < float(min_area_ratio):
                final_reason = f"small_area_ratio_try_{attempt_index}"
                continue
            if union_area_ratio > float(max_area_ratio):
                final_reason = f"large_area_ratio_try_{attempt_index}"
                continue
            if largest_component_ratio < float(min_largest_component_ratio):
                final_reason = f"fragmented_mask_try_{attempt_index}"
                continue
            masked_crop = crop_to_union_mask(image, masks)
            final_reason = "ok"
            payload["sam_trainprep_masked_fallback_stage_v1"] = "sam_prompt_backoff"
            break
        if masked_crop is None:
            if bool(yolo_fallback_enabled):
                model = _load_yolo_world(runtime, model_name=str(yolo_model_name), device=str(device))
                if model is not None:
                    try:
                        prompts = _yolo_prompt_candidates_for_dataset(str(dataset))
                        model.set_classes(prompts)
                        results = model.predict(
                            str(image_path),
                            conf=float(yolo_conf),
                            iou=float(yolo_iou),
                            imgsz=int(yolo_imgsz),
                            max_det=int(yolo_max_det),
                            verbose=False,
                        )
                        result = results[0] if results else None
                        boxes = getattr(result, "boxes", None)
                        if boxes is not None and len(boxes) > 0:
                            conf = boxes.conf.detach().cpu().numpy()
                            xyxy = boxes.xyxy.detach().cpu().numpy()
                            cls = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(conf))
                            best_index = int(np.argmax(conf))
                            yolo_mask = _mask_from_box(image_size=image.size, xyxy=[float(v) for v in xyxy[best_index].tolist()])
                            area_ratio = float(yolo_mask.mean())
                            if float(min_area_ratio) <= area_ratio <= float(max_area_ratio):
                                masked_crop = _crop_to_binary_mask(image, yolo_mask)
                                payload["sam_trainprep_masked_fallback_stage_v1"] = "yolo_world_box"
                                payload["sam_trainprep_masked_reason_v1"] = "yolo_box"
                                payload["sam_trainprep_masked_prompt_used_v1"] = prompts[int(cls[best_index])] if int(cls[best_index]) < len(prompts) else ""
                                payload["sam_trainprep_masked_prompt_rank_v1"] = int(cls[best_index]) + 1
                                payload["sam_trainprep_masked_mask_count_v1"] = 1
                                payload["sam_trainprep_masked_union_area_ratio_v1"] = area_ratio
                                payload["sam_trainprep_masked_largest_component_ratio_v1"] = 1.0
                                payload["sam_trainprep_masked_best_score_v1"] = float(conf[best_index])
                    except Exception as exc:
                        final_reason = f"{final_reason};yolo_error:{type(exc).__name__}"
            if masked_crop is None and bool(geometric_fallback_enabled):
                geometric_mask = _geometric_center_mask(image.size, dataset=str(dataset))
                masked_crop = _crop_to_binary_mask(image, geometric_mask)
                payload["sam_trainprep_masked_fallback_stage_v1"] = "geometric_center_mask"
                payload["sam_trainprep_masked_reason_v1"] = f"{final_reason};geometric_center_mask"
                payload["sam_trainprep_masked_prompt_used_v1"] = "geometric_center"
                payload["sam_trainprep_masked_prompt_rank_v1"] = 0
                payload["sam_trainprep_masked_mask_count_v1"] = 1
                payload["sam_trainprep_masked_union_area_ratio_v1"] = float(geometric_mask.mean())
                payload["sam_trainprep_masked_largest_component_ratio_v1"] = 1.0
                payload["sam_trainprep_masked_best_score_v1"] = 0.0
            if masked_crop is None:
                payload["sam_trainprep_masked_reason_v1"] = final_reason
                return None, payload
    export_rel = (
        output_dir.relative_to(repo_root)
        / "views"
        / MASKED_VIEW_NAME
        / "fallback"
        / str(dataset)
        / _resolve_relative_view_path(original_path)
    )
    export_abs = repo_root / export_rel
    export_abs.parent.mkdir(parents=True, exist_ok=True)
    masked_crop.save(export_abs, quality=95)
    payload["sam_trainprep_masked_status_v1"] = "apply"
    if str(payload.get("sam_trainprep_masked_reason_v1", "")) in {"", "no_mask"}:
        payload["sam_trainprep_masked_reason_v1"] = "ok"
    return export_rel.as_posix(), payload


def build_trainprep_masked_exports(
    *,
    repo_root: Path,
    output_dir: Path,
    base_df: pd.DataFrame,
    target_datasets: list[str],
    existing_export_df: pd.DataFrame | None = None,
    enable_texas_fallback: bool = DEFAULT_TEXAS_FALLBACK_ENABLED,
    texas_fallback_threshold: float = DEFAULT_TEXAS_FALLBACK_THRESHOLD,
    texas_fallback_mask_threshold: float = DEFAULT_TEXAS_FALLBACK_MASK_THRESHOLD,
    texas_fallback_min_area_ratio: float = DEFAULT_TEXAS_FALLBACK_MIN_AREA_RATIO,
    texas_fallback_max_area_ratio: float = DEFAULT_TEXAS_FALLBACK_MAX_AREA_RATIO,
    texas_fallback_min_largest_component_ratio: float = DEFAULT_TEXAS_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
    texas_fallback_device: str = DEFAULT_TEXAS_FALLBACK_DEVICE,
    yolo_fallback_enabled: bool = DEFAULT_YOLO_FALLBACK_ENABLED,
    yolo_fallback_model: str = DEFAULT_YOLO_FALLBACK_MODEL,
    yolo_fallback_conf: float = DEFAULT_YOLO_FALLBACK_CONF,
    yolo_fallback_iou: float = DEFAULT_YOLO_FALLBACK_IOU,
    yolo_fallback_imgsz: int = DEFAULT_YOLO_FALLBACK_IMGSZ,
    yolo_fallback_max_det: int = DEFAULT_YOLO_FALLBACK_MAX_DET,
    geometric_fallback_enabled: bool = DEFAULT_GEOMETRIC_FALLBACK_ENABLED,
) -> pd.DataFrame:
    candidate_df = base_df[base_df["dataset"].isin([str(x) for x in target_datasets])].copy().reset_index(drop=True)
    if candidate_df.empty:
        return pd.DataFrame()
    if existing_export_df is None:
        existing_export_df = pd.DataFrame()
    pending_df = _pending_rows(candidate_df, existing_export_df)
    if pending_df.empty:
        return pd.DataFrame()

    runtime: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    partial_path = output_dir / "tables" / "sam_trainprep_masked_exports_partial_v1.csv"
    ordered_df = pending_df.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    total = len(ordered_df)
    for index, row in enumerate(ordered_df.itertuples(index=False), start=1):
        original_path = str(row.original_rgb_path_v1)
        source_path = str(getattr(row, "sam_masked_rgb_v1_export_path", "") or "")
        source_path_exists = bool(source_path) and (repo_root / source_path).exists()
        payload = {
            "image_id": str(row.image_id),
            "dataset": str(row.dataset),
            "split": str(row.split),
            "identity": str(getattr(row, "identity", "") or ""),
            "original_rgb_path_v1": original_path,
            "sam_trainprep_masked_applied_v1": False,
            "sam_trainprep_masked_source_v1": "missing",
            "sam_trainprep_masked_reason_v1": str(getattr(row, "sam_masked_rgb_v1_reason", "missing_sam") or "missing_sam"),
            "sam_trainprep_masked_prompt_used_v1": "",
            "sam_trainprep_masked_prompt_rank_v1": 0,
            "sam_trainprep_masked_fallback_stage_v1": "none",
            "sam_trainprep_masked_path_v1": "",
            "sam_trainprep_masked_mask_count_v1": int(getattr(row, "sam_masked_rgb_v1_mask_count", 0) or 0),
            "sam_trainprep_masked_union_area_ratio_v1": float(getattr(row, "sam_masked_rgb_v1_union_area_ratio", 0.0) or 0.0),
            "sam_trainprep_masked_largest_component_ratio_v1": float(getattr(row, "sam_masked_rgb_v1_largest_component_ratio", 0.0) or 0.0),
            "sam_trainprep_masked_best_score_v1": float(getattr(row, "sam_masked_rgb_v1_best_score", 0.0) or 0.0),
            "sam_trainprep_masked_threshold_used_v1": 0.0,
            "sam_trainprep_masked_mask_threshold_used_v1": 0.0,
        }
        if bool(getattr(row, "sam_masked_rgb_v1_applied", False)) and source_path_exists:
            payload["sam_trainprep_masked_applied_v1"] = True
            payload["sam_trainprep_masked_source_v1"] = "base_manifest"
            payload["sam_trainprep_masked_reason_v1"] = "reuse_manifest"
            payload["sam_trainprep_masked_prompt_used_v1"] = str(getattr(row, "sam_masked_rgb_v1_prompt", "") or "")
            payload["sam_trainprep_masked_prompt_rank_v1"] = 0
            payload["sam_trainprep_masked_fallback_stage_v1"] = "base_manifest"
            payload["sam_trainprep_masked_path_v1"] = source_path
        elif bool(enable_texas_fallback):
            fallback_rel, fallback_payload = _run_dataset_fallback_mask(
                repo_root=repo_root,
                output_dir=output_dir,
                runtime=runtime,
                dataset=str(row.dataset),
                original_path=original_path,
                threshold=float(texas_fallback_threshold),
                mask_threshold=float(texas_fallback_mask_threshold),
                min_area_ratio=float(texas_fallback_min_area_ratio),
                max_area_ratio=float(texas_fallback_max_area_ratio),
                min_largest_component_ratio=float(texas_fallback_min_largest_component_ratio),
                device=str(texas_fallback_device),
                yolo_fallback_enabled=bool(yolo_fallback_enabled),
                yolo_model_name=str(yolo_fallback_model),
                yolo_conf=float(yolo_fallback_conf),
                yolo_iou=float(yolo_fallback_iou),
                yolo_imgsz=int(yolo_fallback_imgsz),
                yolo_max_det=int(yolo_fallback_max_det),
                geometric_fallback_enabled=bool(geometric_fallback_enabled),
            )
            payload.update(fallback_payload)
            if fallback_rel is not None:
                payload["sam_trainprep_masked_applied_v1"] = True
                payload["sam_trainprep_masked_path_v1"] = str(fallback_rel)
        rows.append(payload)
        print(
            (
                f"[sam_augmented_manifests] masked {index}/{total} | {row.dataset} | {row.image_id} | "
                f"source={payload['sam_trainprep_masked_source_v1']} | applied={int(payload['sam_trainprep_masked_applied_v1'])} | "
                f"stage={payload.get('sam_trainprep_masked_fallback_stage_v1', '')} | reason={payload['sam_trainprep_masked_reason_v1']}"
            ),
            flush=True,
        )
        if index % 25 == 0 or index == total:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def _alignment_min_axis_confidence(dataset: str) -> float:
    return float(DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE_OVERRIDES.get(str(dataset), DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE))


def build_trainprep_aligned_exports(
    *,
    repo_root: Path,
    output_dir: Path,
    masked_export_df: pd.DataFrame,
    existing_export_df: pd.DataFrame | None = None,
    min_foreground_pixels: int = DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS,
    min_area_ratio: float = DEFAULT_ALIGNMENT_MIN_AREA_RATIO,
    max_area_ratio: float = DEFAULT_ALIGNMENT_MAX_AREA_RATIO,
    padding_ratio: float = DEFAULT_ALIGNMENT_PADDING_RATIO,
    padding_ratio_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    if masked_export_df.empty:
        return pd.DataFrame()
    available_df = masked_export_df[masked_export_df["sam_trainprep_masked_applied_v1"].fillna(False)].copy().reset_index(drop=True)
    if available_df.empty:
        return pd.DataFrame()
    if existing_export_df is None:
        existing_export_df = pd.DataFrame()
    pending_df = _pending_rows(available_df, existing_export_df)
    if pending_df.empty:
        return pd.DataFrame()
    if padding_ratio_overrides is None:
        padding_ratio_overrides = ALIGNED_CROP_PADDING_RATIO_OVERRIDES

    rows: list[dict[str, Any]] = []
    partial_path = output_dir / "tables" / "sam_trainprep_aligned_exports_partial_v1.csv"
    ordered_df = pending_df.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    total = len(ordered_df)
    for index, row in enumerate(ordered_df.itertuples(index=False), start=1):
        masked_path = str(row.sam_trainprep_masked_path_v1)
        payload = {
            "image_id": str(row.image_id),
            "dataset": str(row.dataset),
            "split": str(row.split),
            "identity": str(getattr(row, "identity", "") or ""),
            "original_rgb_path_v1": str(row.original_rgb_path_v1),
            "sam_trainprep_aligned_applied_v1": False,
            "sam_trainprep_aligned_reason_v1": "missing_masked",
            "sam_trainprep_aligned_path_v1": "",
            "sam_trainprep_aligned_foreground_pixels_v1": 0.0,
            "sam_trainprep_aligned_foreground_area_ratio_v1": 0.0,
            "sam_trainprep_aligned_axis_angle_deg_v1": 0.0,
            "sam_trainprep_aligned_axis_confidence_v1": 0.0,
            "sam_trainprep_aligned_rotation_applied_deg_v1": 0.0,
            "sam_trainprep_aligned_padding_ratio_v1": float(resolve_crop_padding_ratio(str(row.dataset), default_padding_ratio=float(padding_ratio), padding_ratio_overrides=padding_ratio_overrides)),
        }
        masked_abs = repo_root / masked_path
        if not masked_abs.exists():
            payload["sam_trainprep_aligned_reason_v1"] = "masked_path_missing"
            rows.append(payload)
            continue
        with Image.open(masked_abs) as image_handle:
            masked_image = image_handle.convert("RGB")
            inferred_mask = infer_mask_from_masked_rgb(masked_image, nonzero_threshold=1)
            axis_stats = compute_body_axis(inferred_mask)
            if axis_stats is None:
                payload["sam_trainprep_aligned_reason_v1"] = "no_axis"
            else:
                payload["sam_trainprep_aligned_foreground_pixels_v1"] = float(axis_stats["foreground_pixels"])
                payload["sam_trainprep_aligned_foreground_area_ratio_v1"] = float(axis_stats["foreground_area_ratio"])
                payload["sam_trainprep_aligned_axis_angle_deg_v1"] = float(axis_stats["axis_angle_deg"])
                payload["sam_trainprep_aligned_axis_confidence_v1"] = float(axis_stats["axis_confidence"])
                if float(axis_stats["foreground_pixels"]) < float(min_foreground_pixels):
                    payload["sam_trainprep_aligned_reason_v1"] = "small_mask"
                elif float(axis_stats["foreground_area_ratio"]) < float(min_area_ratio):
                    payload["sam_trainprep_aligned_reason_v1"] = "small_area_ratio"
                elif float(axis_stats["foreground_area_ratio"]) > float(max_area_ratio):
                    payload["sam_trainprep_aligned_reason_v1"] = "large_area_ratio"
                elif float(axis_stats["axis_confidence"]) < _alignment_min_axis_confidence(str(row.dataset)):
                    payload["sam_trainprep_aligned_reason_v1"] = "low_axis_confidence"
                else:
                    rotation_deg = rotation_to_horizontal(float(axis_stats["axis_angle_deg"]))
                    aligned_rgb, _ = rotate_and_crop(
                        masked_image,
                        inferred_mask,
                        rotation_deg,
                        background=(0, 0, 0),
                        padding_ratio=float(payload["sam_trainprep_aligned_padding_ratio_v1"]),
                        keep_background=False,
                        canvas_fill_mode="constant",
                    )
                    export_rel = (
                        output_dir.relative_to(repo_root)
                        / "views"
                        / ALIGNED_VIEW_NAME
                        / _resolve_relative_view_path(str(row.original_rgb_path_v1))
                    )
                    export_abs = repo_root / export_rel
                    export_abs.parent.mkdir(parents=True, exist_ok=True)
                    aligned_rgb.save(export_abs, quality=95)
                    payload["sam_trainprep_aligned_applied_v1"] = True
                    payload["sam_trainprep_aligned_reason_v1"] = "ok"
                    payload["sam_trainprep_aligned_path_v1"] = export_rel.as_posix()
                    payload["sam_trainprep_aligned_rotation_applied_deg_v1"] = float(rotation_deg)
        rows.append(payload)
        print(
            (
                f"[sam_augmented_manifests] aligned {index}/{total} | {row.dataset} | {row.image_id} | "
                f"applied={int(payload['sam_trainprep_aligned_applied_v1'])} | reason={payload['sam_trainprep_aligned_reason_v1']}"
            ),
            flush=True,
        )
        if index % 100 == 0 or index == total:
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial_path, index=False)
    return pd.DataFrame(rows).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)


def create_trainprep_enriched_metadata(
    *,
    base_df: pd.DataFrame,
    masked_export_df: pd.DataFrame,
    aligned_export_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = base_df.copy().reset_index(drop=True)
    enriched["image_id"] = enriched["image_id"].astype(str)
    enriched["identity"] = enriched["identity"].fillna("").astype(str)
    masked_cols = [
        column
        for column in [
            "image_id",
            "dataset",
            "split",
            "original_rgb_path_v1",
            "sam_trainprep_masked_applied_v1",
            "sam_trainprep_masked_source_v1",
            "sam_trainprep_masked_reason_v1",
            "sam_trainprep_masked_prompt_used_v1",
            "sam_trainprep_masked_prompt_rank_v1",
            "sam_trainprep_masked_fallback_stage_v1",
            "sam_trainprep_masked_path_v1",
            "sam_trainprep_masked_mask_count_v1",
            "sam_trainprep_masked_union_area_ratio_v1",
            "sam_trainprep_masked_largest_component_ratio_v1",
            "sam_trainprep_masked_best_score_v1",
            "sam_trainprep_masked_threshold_used_v1",
            "sam_trainprep_masked_mask_threshold_used_v1",
        ]
        if column in masked_export_df.columns
    ]
    aligned_cols = [
        column
        for column in [
            "image_id",
            "dataset",
            "split",
            "original_rgb_path_v1",
            "sam_trainprep_aligned_applied_v1",
            "sam_trainprep_aligned_reason_v1",
            "sam_trainprep_aligned_path_v1",
            "sam_trainprep_aligned_foreground_pixels_v1",
            "sam_trainprep_aligned_foreground_area_ratio_v1",
            "sam_trainprep_aligned_axis_angle_deg_v1",
            "sam_trainprep_aligned_axis_confidence_v1",
            "sam_trainprep_aligned_rotation_applied_deg_v1",
            "sam_trainprep_aligned_padding_ratio_v1",
        ]
        if column in aligned_export_df.columns
    ]
    if masked_cols:
        enriched = enriched.merge(masked_export_df[masked_cols], on=["image_id", "dataset", "split", "original_rgb_path_v1"], how="left")
    if aligned_cols:
        enriched = enriched.merge(aligned_export_df[aligned_cols], on=["image_id", "dataset", "split", "original_rgb_path_v1"], how="left")

    default_values: dict[str, Any] = {
        "sam_trainprep_masked_applied_v1": False,
        "sam_trainprep_masked_source_v1": "missing",
        "sam_trainprep_masked_reason_v1": "missing_sam",
        "sam_trainprep_masked_prompt_used_v1": "",
        "sam_trainprep_masked_prompt_rank_v1": 0,
        "sam_trainprep_masked_fallback_stage_v1": "none",
        "sam_trainprep_masked_path_v1": "",
        "sam_trainprep_aligned_applied_v1": False,
        "sam_trainprep_aligned_reason_v1": "missing_masked",
        "sam_trainprep_aligned_path_v1": "",
    }
    for column, default_value in default_values.items():
        if column not in enriched.columns:
            enriched[column] = default_value
        else:
            enriched[column] = enriched[column].fillna(default_value)
    enriched["sam_trainprep_masked_applied_v1"] = enriched["sam_trainprep_masked_applied_v1"].fillna(False).astype(bool)
    enriched["sam_trainprep_aligned_applied_v1"] = enriched["sam_trainprep_aligned_applied_v1"].fillna(False).astype(bool)
    enriched["sam_trainprep_masked_resolved_path_v1"] = np.where(
        enriched["sam_trainprep_masked_applied_v1"] & enriched["sam_trainprep_masked_path_v1"].astype(str).ne(""),
        enriched["sam_trainprep_masked_path_v1"].astype(str),
        enriched["original_rgb_path_v1"].astype(str),
    )
    enriched["sam_trainprep_aligned_resolved_path_v1"] = np.where(
        enriched["sam_trainprep_aligned_applied_v1"] & enriched["sam_trainprep_aligned_path_v1"].astype(str).ne(""),
        enriched["sam_trainprep_aligned_path_v1"].astype(str),
        enriched["sam_trainprep_masked_resolved_path_v1"].astype(str),
    )
    return enriched


def _base_manifest_slice(enriched_df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "train":
        return enriched_df[enriched_df["recommended_train_keep_all_v1"].fillna(False)].copy().reset_index(drop=True)
    if split == "test":
        return enriched_df[enriched_df["split"].astype(str).eq("test")].copy().reset_index(drop=True)
    raise ValueError(f"Unsupported split: {split}")


def build_trainprep_single_view_manifest(
    *,
    enriched_df: pd.DataFrame,
    split: str,
    view_name: str,
    target_datasets: list[str],
) -> pd.DataFrame:
    if view_name not in {TRAIN_SINGLE_MASKED_VIEW_NAME, TRAIN_SINGLE_ALIGNED_VIEW_NAME}:
        raise ValueError(f"Unsupported view_name: {view_name}")
    manifest_df = _base_manifest_slice(enriched_df=enriched_df, split=split)
    target_mask = manifest_df["dataset"].astype(str).isin([str(x) for x in target_datasets])
    if view_name == TRAIN_SINGLE_MASKED_VIEW_NAME:
        resolved_path = manifest_df["sam_trainprep_masked_resolved_path_v1"].astype(str)
    else:
        resolved_path = manifest_df["sam_trainprep_aligned_resolved_path_v1"].astype(str)
    manifest_df[PATH_COLUMN] = np.where(target_mask, resolved_path, manifest_df["original_rgb_path_v1"].astype(str))
    manifest_df["preferred_path_v1"] = manifest_df[PATH_COLUMN]
    manifest_df["preprocess_variant_v1"] = np.where(target_mask, view_name, "original_only")
    manifest_df["manifest_view_name_v1"] = view_name
    manifest_df["manifest_view_requested_v1"] = view_name
    manifest_df["manifest_view_resolved_v1"] = manifest_df["preprocess_variant_v1"]
    manifest_df["manifest_view_applied_v1"] = target_mask.astype(bool)
    manifest_df["path"] = manifest_df[PATH_COLUMN]
    return manifest_df.sort_values(["dataset", "identity", "image_id"]).reset_index(drop=True)


def build_trainprep_multiview_manifest(
    *,
    enriched_df: pd.DataFrame,
    target_datasets: list[str],
) -> pd.DataFrame:
    base_df = _base_manifest_slice(enriched_df=enriched_df, split="train")
    rows: list[dict[str, Any]] = []
    target_set = {str(x) for x in target_datasets}
    for row in base_df.itertuples(index=False):
        common = pd.Series(row._asdict()).to_dict()
        source_image_id = str(common["image_id"])
        variants: list[tuple[str, str, bool]] = [("original", str(common["original_rgb_path_v1"]), False)]
        if str(common["dataset"]) in target_set:
            if bool(common.get("sam_trainprep_masked_applied_v1", False)) and str(common.get("sam_trainprep_masked_path_v1", "")):
                variants.append(("sam_masked", str(common["sam_trainprep_masked_path_v1"]), True))
            if bool(common.get("sam_trainprep_aligned_applied_v1", False)) and str(common.get("sam_trainprep_aligned_path_v1", "")):
                variants.append(("sam_aligned", str(common["sam_trainprep_aligned_path_v1"]), True))
        for rank, (variant_name, variant_path, is_augmented) in enumerate(variants):
            payload = dict(common)
            payload["source_image_id_v1"] = source_image_id
            payload["sam_trainprep_group_id_v1"] = source_image_id
            payload["sam_trainprep_variant_v1"] = variant_name
            payload["sam_trainprep_variant_rank_v1"] = int(rank)
            payload["sam_trainprep_is_augmented_v1"] = bool(is_augmented)
            payload["image_id"] = f"{source_image_id}__{variant_name}"
            payload[PATH_COLUMN] = str(variant_path)
            payload["preferred_path_v1"] = str(variant_path)
            payload["preprocess_variant_v1"] = variant_name
            payload["manifest_view_name_v1"] = TRAIN_MULTIVIEW_NAME
            payload["manifest_view_requested_v1"] = TRAIN_MULTIVIEW_NAME
            payload["manifest_view_resolved_v1"] = variant_name
            payload["manifest_view_applied_v1"] = bool(is_augmented)
            payload["path"] = str(variant_path)
            rows.append(payload)
    return pd.DataFrame(rows).sort_values(["dataset", "identity", "source_image_id_v1", "sam_trainprep_variant_rank_v1"]).reset_index(drop=True)


def _summarize_masked_exports(masked_export_df: pd.DataFrame) -> pd.DataFrame:
    if masked_export_df.empty:
        return pd.DataFrame(columns=["dataset", "split", "images", "applied", "applied_ratio", "base_manifest_count", "fallback_count", "geometric_count"])
    grouped = masked_export_df.groupby(["dataset", "split"])
    summary_df = grouped.agg(
        images=("image_id", "count"),
        applied=("sam_trainprep_masked_applied_v1", lambda s: int(np.sum(s))),
        applied_ratio=("sam_trainprep_masked_applied_v1", lambda s: round(float(np.mean(s)), 4)),
        base_manifest_count=("sam_trainprep_masked_fallback_stage_v1", lambda s: int(pd.Series(s).astype(str).eq("base_manifest").sum())),
        fallback_count=("sam_trainprep_masked_source_v1", lambda s: int(pd.Series(s).astype(str).eq("fallback").sum())),
        geometric_count=("sam_trainprep_masked_fallback_stage_v1", lambda s: int(pd.Series(s).astype(str).eq("geometric_center_mask").sum())),
    ).reset_index().sort_values(["dataset", "split"])
    return summary_df


def _summarize_aligned_exports(aligned_export_df: pd.DataFrame) -> pd.DataFrame:
    if aligned_export_df.empty:
        return pd.DataFrame(columns=["dataset", "split", "images", "applied", "applied_ratio"])
    summary_df = (
        aligned_export_df.groupby(["dataset", "split"])
        .agg(
            images=("image_id", "count"),
            applied=("sam_trainprep_aligned_applied_v1", lambda s: int(np.sum(s))),
            applied_ratio=("sam_trainprep_aligned_applied_v1", lambda s: round(float(np.mean(s)), 4)),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
    )
    return summary_df


def write_trainprep_summary(
    *,
    output_path: Path,
    config: dict[str, Any],
    manifest_table_df: pd.DataFrame,
    masked_summary_df: pd.DataFrame,
    aligned_summary_df: pd.DataFrame,
) -> None:
    lines = [
        "# SAM Segmentation Train Prep v1",
        "",
        "## Config",
        "",
        f"- Source manifest root: `{config['source_manifest_root']}`",
        f"- Output root: `{config['output_root']}`",
        f"- Target datasets: `{', '.join(config['target_datasets'])}`",
        f"- Texas fallback enabled: `{config['enable_texas_fallback']}`",
        (
            f"- Generic fallback gate: `threshold={config['texas_fallback_threshold']}`, "
            f"`mask_threshold={config['texas_fallback_mask_threshold']}`, "
            f"`min_area_ratio={config['texas_fallback_min_area_ratio']}`, "
            f"`max_area_ratio={config['texas_fallback_max_area_ratio']}`, "
            f"`min_largest_component_ratio={config['texas_fallback_min_largest_component_ratio']}`"
        ),
        (
            f"- YOLO fallback: `enabled={config['yolo_fallback_enabled']}`, "
            f"`model={config['yolo_fallback_model']}`, `conf={config['yolo_fallback_conf']}`, "
            f"`iou={config['yolo_fallback_iou']}`, `imgsz={config['yolo_fallback_imgsz']}`"
        ),
        f"- Geometric fallback enabled: `{config['geometric_fallback_enabled']}`",
        (
            f"- Mask-first alignment gate: `min_foreground_pixels={config['alignment_min_foreground_pixels']}`, "
            f"`min_area_ratio={config['alignment_min_area_ratio']}`, "
            f"`max_area_ratio={config['alignment_max_area_ratio']}`, "
            f"`min_axis_confidence_default={config['alignment_min_axis_confidence']}`, "
            f"`min_axis_confidence_overrides={config['alignment_min_axis_confidence_overrides']}`"
        ),
        "",
        "## Manifest Files",
        "",
        dataframe_to_markdown_table(manifest_table_df),
        "",
        "## Masked Export Summary",
        "",
        dataframe_to_markdown_table(masked_summary_df),
        "",
        "## Aligned Export Summary",
        "",
        dataframe_to_markdown_table(aligned_summary_df),
        "",
        "## Reading Notes",
        "",
        "- `sam_trainprep_masked_best_v1`：对目标数据集优先走已有 `SAM masked`；缺失时按 `dataset prompt -> animal -> YOLO-World bbox -> geometric center mask` 补救，目标是不再裸回退原图。",
        "- `sam_trainprep_aligned_best_v1`：对目标数据集优先走 `mask-first aligned`；若轴向不稳定则回退到 masked foreground view，而不是裸原图。",
        "- `sam_trainprep_multiview_v1`：训练专用 manifest；同一原图会展开成 `original / sam_masked / sam_aligned` 多条样本，`image_id` 已改成 view-specific 唯一键。",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_sam_augmented_manifests(
    *,
    repo_root: Path,
    source_manifest_root: Path = DEFAULT_SOURCE_MANIFEST_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_ROOT,
    target_datasets: list[str] | None = None,
    enable_texas_fallback: bool = DEFAULT_TEXAS_FALLBACK_ENABLED,
    texas_fallback_threshold: float = DEFAULT_TEXAS_FALLBACK_THRESHOLD,
    texas_fallback_mask_threshold: float = DEFAULT_TEXAS_FALLBACK_MASK_THRESHOLD,
    texas_fallback_min_area_ratio: float = DEFAULT_TEXAS_FALLBACK_MIN_AREA_RATIO,
    texas_fallback_max_area_ratio: float = DEFAULT_TEXAS_FALLBACK_MAX_AREA_RATIO,
    texas_fallback_min_largest_component_ratio: float = DEFAULT_TEXAS_FALLBACK_MIN_LARGEST_COMPONENT_RATIO,
    texas_fallback_device: str = DEFAULT_TEXAS_FALLBACK_DEVICE,
    yolo_fallback_enabled: bool = DEFAULT_YOLO_FALLBACK_ENABLED,
    yolo_fallback_model: str = DEFAULT_YOLO_FALLBACK_MODEL,
    yolo_fallback_conf: float = DEFAULT_YOLO_FALLBACK_CONF,
    yolo_fallback_iou: float = DEFAULT_YOLO_FALLBACK_IOU,
    yolo_fallback_imgsz: int = DEFAULT_YOLO_FALLBACK_IMGSZ,
    yolo_fallback_max_det: int = DEFAULT_YOLO_FALLBACK_MAX_DET,
    geometric_fallback_enabled: bool = DEFAULT_GEOMETRIC_FALLBACK_ENABLED,
    alignment_min_foreground_pixels: int = DEFAULT_ALIGNMENT_MIN_FOREGROUND_PIXELS,
    alignment_min_area_ratio: float = DEFAULT_ALIGNMENT_MIN_AREA_RATIO,
    alignment_max_area_ratio: float = DEFAULT_ALIGNMENT_MAX_AREA_RATIO,
    alignment_padding_ratio: float = DEFAULT_ALIGNMENT_PADDING_RATIO,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    source_manifest_root = source_manifest_root.resolve() if source_manifest_root.is_absolute() else (repo_root / source_manifest_root).resolve()
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if target_datasets is None:
        target_datasets = list(DEFAULT_TARGET_DATASETS)
    target_datasets = [str(x) for x in target_datasets]

    base_df = _load_base_metadata(source_manifest_root)
    masked_export_path = tables_dir / "sam_trainprep_masked_exports_v1.csv"
    aligned_export_path = tables_dir / "sam_trainprep_aligned_exports_v1.csv"
    masked_partial_export_path = tables_dir / "sam_trainprep_masked_exports_partial_v1.csv"
    aligned_partial_export_path = tables_dir / "sam_trainprep_aligned_exports_partial_v1.csv"
    existing_masked_df = _merge_export_frames(_load_export_frame(masked_export_path), _load_export_frame(masked_partial_export_path))
    existing_aligned_df = _merge_export_frames(_load_export_frame(aligned_export_path), _load_export_frame(aligned_partial_export_path))

    new_masked_df = build_trainprep_masked_exports(
        repo_root=repo_root,
        output_dir=output_dir,
        base_df=base_df,
        target_datasets=target_datasets,
        existing_export_df=existing_masked_df,
        enable_texas_fallback=enable_texas_fallback,
        texas_fallback_threshold=texas_fallback_threshold,
        texas_fallback_mask_threshold=texas_fallback_mask_threshold,
        texas_fallback_min_area_ratio=texas_fallback_min_area_ratio,
        texas_fallback_max_area_ratio=texas_fallback_max_area_ratio,
        texas_fallback_min_largest_component_ratio=texas_fallback_min_largest_component_ratio,
        texas_fallback_device=texas_fallback_device,
        yolo_fallback_enabled=yolo_fallback_enabled,
        yolo_fallback_model=yolo_fallback_model,
        yolo_fallback_conf=yolo_fallback_conf,
        yolo_fallback_iou=yolo_fallback_iou,
        yolo_fallback_imgsz=yolo_fallback_imgsz,
        yolo_fallback_max_det=yolo_fallback_max_det,
        geometric_fallback_enabled=geometric_fallback_enabled,
    )
    masked_export_df = _merge_export_frames(existing_masked_df, new_masked_df)
    new_aligned_df = build_trainprep_aligned_exports(
        repo_root=repo_root,
        output_dir=output_dir,
        masked_export_df=masked_export_df,
        existing_export_df=existing_aligned_df,
        min_foreground_pixels=alignment_min_foreground_pixels,
        min_area_ratio=alignment_min_area_ratio,
        max_area_ratio=alignment_max_area_ratio,
        padding_ratio=alignment_padding_ratio,
    )
    aligned_export_df = _merge_export_frames(existing_aligned_df, new_aligned_df)
    enriched_df = create_trainprep_enriched_metadata(
        base_df=base_df,
        masked_export_df=masked_export_df,
        aligned_export_df=aligned_export_df,
    )

    metadata_path = tables_dir / "metadata_enriched_trainprep_v1.csv"
    enriched_df.to_csv(metadata_path, index=False)
    masked_export_df.to_csv(masked_export_path, index=False)
    aligned_export_df.to_csv(aligned_export_path, index=False)

    manifest_rows: list[dict[str, Any]] = []
    outputs: dict[str, Path] = {
        "metadata_path": metadata_path,
        "masked_export_path": masked_export_path,
        "aligned_export_path": aligned_export_path,
    }
    manifest_specs = [
        ("train", TRAIN_SINGLE_MASKED_VIEW_NAME),
        ("test", TRAIN_SINGLE_MASKED_VIEW_NAME),
        ("train", TRAIN_SINGLE_ALIGNED_VIEW_NAME),
        ("test", TRAIN_SINGLE_ALIGNED_VIEW_NAME),
    ]
    for split, view_name in manifest_specs:
        manifest_df = build_trainprep_single_view_manifest(
            enriched_df=enriched_df,
            split=split,
            view_name=view_name,
            target_datasets=target_datasets,
        )
        manifest_path = tables_dir / f"manifest_{split}_{view_name}.csv"
        manifest_df.to_csv(manifest_path, index=False)
        outputs[f"{split}_{view_name}_path"] = manifest_path
        manifest_rows.append(
            {
                "manifest": manifest_path.name,
                "split": split,
                "view_name": view_name,
                "rows": int(len(manifest_df)),
                "path": str(manifest_path),
            }
        )
    multiview_train_df = build_trainprep_multiview_manifest(
        enriched_df=enriched_df,
        target_datasets=target_datasets,
    )
    multiview_train_path = tables_dir / f"manifest_train_{TRAIN_MULTIVIEW_NAME}.csv"
    multiview_train_df.to_csv(multiview_train_path, index=False)
    outputs[f"train_{TRAIN_MULTIVIEW_NAME}_path"] = multiview_train_path
    manifest_rows.append(
        {
            "manifest": multiview_train_path.name,
            "split": "train",
            "view_name": TRAIN_MULTIVIEW_NAME,
            "rows": int(len(multiview_train_df)),
            "path": str(multiview_train_path),
        }
    )

    manifest_table_df = pd.DataFrame(manifest_rows).sort_values(["split", "manifest"]).reset_index(drop=True)
    masked_summary_df = _summarize_masked_exports(masked_export_df)
    aligned_summary_df = _summarize_aligned_exports(aligned_export_df)
    summary_path = reports_dir / "summary.md"
    config = {
        "source_manifest_root": str(source_manifest_root),
        "output_root": str(output_dir),
        "target_datasets": target_datasets,
        "enable_texas_fallback": bool(enable_texas_fallback),
        "texas_fallback_threshold": float(texas_fallback_threshold),
        "texas_fallback_mask_threshold": float(texas_fallback_mask_threshold),
        "texas_fallback_min_area_ratio": float(texas_fallback_min_area_ratio),
        "texas_fallback_max_area_ratio": float(texas_fallback_max_area_ratio),
        "texas_fallback_min_largest_component_ratio": float(texas_fallback_min_largest_component_ratio),
        "texas_fallback_device": str(texas_fallback_device),
        "yolo_fallback_enabled": bool(yolo_fallback_enabled),
        "yolo_fallback_model": str(yolo_fallback_model),
        "yolo_fallback_conf": float(yolo_fallback_conf),
        "yolo_fallback_iou": float(yolo_fallback_iou),
        "yolo_fallback_imgsz": int(yolo_fallback_imgsz),
        "yolo_fallback_max_det": int(yolo_fallback_max_det),
        "geometric_fallback_enabled": bool(geometric_fallback_enabled),
        "alignment_min_foreground_pixels": int(alignment_min_foreground_pixels),
        "alignment_min_area_ratio": float(alignment_min_area_ratio),
        "alignment_max_area_ratio": float(alignment_max_area_ratio),
        "alignment_min_axis_confidence": float(DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE),
        "alignment_min_axis_confidence_overrides": dict(DEFAULT_ALIGNMENT_MIN_AXIS_CONFIDENCE_OVERRIDES),
        "alignment_padding_ratio": float(alignment_padding_ratio),
    }
    write_trainprep_summary(
        output_path=summary_path,
        config=config,
        manifest_table_df=manifest_table_df,
        masked_summary_df=masked_summary_df,
        aligned_summary_df=aligned_summary_df,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps({**config, **{key: str(value) for key, value in outputs.items()}}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    outputs["summary_path"] = summary_path
    return outputs
