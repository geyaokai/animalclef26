from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .body_orientation_probe import (
    compute_body_axis,
    decide_orientation_application,
    merge_masks,
    resolve_crop_padding_ratio,
    rotate_and_crop,
    rotation_to_horizontal,
    rotation_to_vertical,
)
from .initial_audit import load_metadata
from .qualitative_lynx_views import build_qualitative_lynx_view
from .qualitative_salamander_views import (
    SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES,
    generate_heuristic_end_middle_crops,
    scale_normalize_aligned_foreground as scale_normalize_salamander_aligned_foreground,
)
from .qualitative_texas_views import (
    TEXAS_YOLO_WORLD_PROMPT_CANDIDATES,
    build_texas_view_metadata,
    crop_texas_center_body_square,
    grayscale_normalize_image,
    scale_normalize_aligned_foreground as scale_normalize_texas_aligned_foreground,
)
from .sam3_probe import crop_to_union_mask, get_prompt_candidates_for_dataset, run_single_inference_with_prompt_backoff, sample_rows_by_dataset
from .sam_orb_veto import infer_mask_from_masked_rgb


DEFAULT_OUTPUT_ROOT = Path("artifacts/preprocessing_qualitative/v1")
DEFAULT_DATASETS = ("TexasHornedLizards", "SalamanderID2025", "LynxID2025")
DEFAULT_SAMPLES_PER_SPLIT = 4
DEFAULT_YOLO_MODEL = "yolov8s-worldv2.pt"
DEFAULT_YOLO_CONF = 0.15
DEFAULT_YOLO_IOU = 0.5
DEFAULT_YOLO_IMGSZ = 640
DEFAULT_YOLO_MAX_DET = 8
DEFAULT_TEXAS_AUGMENTOR_SAMPLES_PER_IMAGE = 2

DEFAULT_MIN_FOREGROUND_PIXELS = {
    "TexasHornedLizards": 256,
    "SalamanderID2025": 256,
}
DEFAULT_MIN_AREA_RATIO = {
    "TexasHornedLizards": 0.01,
    "SalamanderID2025": 0.01,
}
DEFAULT_MAX_AREA_RATIO = {
    "TexasHornedLizards": 0.95,
    "SalamanderID2025": 0.95,
}
DEFAULT_MIN_AXIS_CONFIDENCE = {
    "TexasHornedLizards": 0.10,
    "SalamanderID2025": 0.18,
}
DEFAULT_MIN_LARGEST_COMPONENT_RATIO = {
    "TexasHornedLizards": 0.40,
    "SalamanderID2025": 0.70,
}
DEFAULT_FORCE_VERTICAL_ALIGNMENT = {
    "TexasHornedLizards": True,
    "SalamanderID2025": True,
}


def parameter_explanation_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "parameter": "axis_confidence",
                "meaning": "PCA 主轴置信度，越高表示主体长轴越清楚，几何对齐越可信。",
            },
            {
                "parameter": "target_extent_ratio / target_major_extent_ratio",
                "meaning": "尺度归一化目标：主体长轴占输出画布长边的比例；本阶段会偏向更大的主体覆盖率。",
            },
            {
                "parameter": "scale_factor",
                "meaning": "主体在尺度归一化前后的等比例缩放系数；当前策略优先放大，尽量避免无必要缩小。",
            },
            {
                "parameter": "band_start_ratio / band_end_ratio",
                "meaning": "旧版 Texas 中段 band 参数；新版主规则已改为主体中心向外扩的 body square。",
            },
            {
                "parameter": "crop_start_ratio / crop_end_ratio",
                "meaning": "旧版 Salamander 端部/中段 crop 参数；新版主规则已改为主体中心向外扩的 trunk rectangle。",
            },
            {
                "parameter": "conf / iou",
                "meaning": "YOLO-World 推理阈值；conf 控制保留框置信度，iou 控制 NMS 去重强度。",
            },
            {
                "parameter": "yolo_imgsz / yolo_max_det",
                "meaning": "YOLO-World 输入边长与单图最大保留框数；前者影响小目标召回与时延，后者限制候选框规模。",
            },
            {
                "parameter": "sam_threshold / sam_mask_threshold",
                "meaning": "SAM 主体判定阈值；前者控制候选 mask 的前景接受门槛，后者控制二值化 mask 的保留强度。",
            },
            {
                "parameter": "gray_low/high_percentile",
                "meaning": "灰度归一化时用于拉伸动态范围的低/高百分位。",
            },
            {
                "parameter": "detector_padding_ratio",
                "meaning": "检测框外扩比例；用于给 head/body crop 额外留白，避免切掉关键边缘纹理。",
            },
            {
                "parameter": "orientation_target=vertical",
                "meaning": "SAM 主体对齐的目标方向为竖直，便于与主数据集的常见姿态分布保持一致。",
            },
            {
                "parameter": "forced_vertical_alignment",
                "meaning": "Texas / Salamander 的定性缓存分支优先统一到竖直方向；若原门控偏保守，会在记录里显示 forced_vertical:* 方便人工复核。",
            },
            {
                "parameter": "augment_probability / augment_factor_range",
                "meaning": "Augmentor 预览中每种增强操作的触发概率与强度范围，用于观察小样本是否会过强扰动纹理。",
            },
            {
                "parameter": "fallback_reason",
                "meaning": "某个分支未按理想路径执行时的回退原因，便于人工审阅失败模式。",
            },
        ]
    )


def _require_ultralytics() -> Any:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - runtime only
        raise RuntimeError("ultralytics is required for YOLO-World qualitative review.") from exc
    return YOLO


def _load_yolo_world_model(model_name: str, *, device: str | None = None) -> Any:
    YOLO = _require_ultralytics()
    try:
        model = YOLO(model_name)
    except Exception:
        if model_name.endswith("worldv2.pt"):
            fallback_name = model_name.replace("worldv2.pt", "world.pt")
            model = YOLO(fallback_name)
        else:
            raise
    if device:
        model.to(device)
    return model


def _load_sam_resources(device: str) -> Any:
    from .sam3_probe import load_sam3

    return load_sam3(device=device)


def _flatten_payload(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (list, tuple, dict)):
            flat[f"{prefix}_{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            flat[f"{prefix}_{key}"] = value
    return flat


def _merge_reasons(*values: Any) -> str:
    reasons: list[str] = []
    for value in values:
        if value in (None, "", "None"):
            continue
        if isinstance(value, str):
            candidates = value.replace("|", ";").split(";")
        else:
            candidates = [str(value)]
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned not in reasons:
                reasons.append(cleaned)
    return ";".join(reasons)


def _resolve_vertical_alignment_decision(
    *,
    dataset: str,
    decision: Any,
    axis_stats: dict[str, float] | None,
) -> tuple[bool, str]:
    if axis_stats is None:
        return False, str(decision.reason)
    if bool(decision.should_apply):
        return True, str(decision.reason)
    if bool(DEFAULT_FORCE_VERTICAL_ALIGNMENT.get(dataset, False)):
        return True, f"forced_vertical:{decision.reason}"
    return False, str(decision.reason)


def _extract_best_box(result: Any) -> dict[str, Any] | None:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(conf))
    best_index = int(np.argmax(conf))
    return {
        "xyxy": [float(value) for value in xyxy[best_index].tolist()],
        "confidence": float(conf[best_index]),
        "class_index": int(classes[best_index]),
    }


def _draw_box_overlay(image: Image.Image, detection: dict[str, Any] | None, *, label: str, color: tuple[int, int, int]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    if detection is None:
        return canvas
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x0, y0, x1, y1 = detection["xyxy"]
    draw.rectangle((x0, y0, x1, y1), outline=color, width=4)
    text = f"{label} {float(detection['confidence']):.2f}"
    text_w = max(42, 6 * len(text))
    draw.rectangle((x0, max(0, y0 - 18), x0 + text_w, y0), fill=color)
    draw.text((x0 + 4, max(0, y0 - 15)), text, fill=(255, 255, 255), font=font)
    return canvas


def _crop_from_detection(image: Image.Image, detection: dict[str, Any] | None, *, padding_ratio: float = 0.04) -> tuple[Image.Image, dict[str, Any]]:
    if detection is None:
        return image.convert("RGB").copy(), {"fallback_reason": "no_detection", "crop_box_xyxy": None}
    x0, y0, x1, y1 = [float(value) for value in detection["xyxy"]]
    width, height = image.size
    pad = float(max(x1 - x0, y1 - y0)) * float(padding_ratio)
    left = max(0, int(round(x0 - pad)))
    top = max(0, int(round(y0 - pad)))
    right = min(width, int(round(x1 + pad)))
    bottom = min(height, int(round(y1 + pad)))
    return image.crop((left, top, right, bottom)).convert("RGB"), {
        "fallback_reason": "",
        "crop_box_xyxy": [left, top, right, bottom],
        "box_confidence": round(float(detection["confidence"]), 6),
    }


def _run_yolo_prompt_backoff(
    model: Any,
    image: Image.Image,
    prompt_candidates: tuple[str, ...] | list[str],
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    best_failure = {
        "selected_prompt": "",
        "prompt_rank": 0,
        "attempted_prompt_count": int(len(prompt_candidates)),
        "confidence": 0.0,
        "iou": float(iou),
        "fallback_reason": "no_detection",
    }
    for prompt_rank, prompt in enumerate(prompt_candidates, start=1):
        try:
            world_model = getattr(model, "model", None)
            if world_model is not None and getattr(world_model, "clip_model", None) is not None:
                clip_model = world_model.clip_model
                if hasattr(clip_model, "device") and hasattr(world_model, "model"):
                    clip_model.device = next(world_model.model.parameters()).device
            model.set_classes([str(prompt)])
            results = model.predict(
                image,
                conf=float(conf),
                iou=float(iou),
                imgsz=int(imgsz),
                max_det=int(max_det),
                verbose=False,
            )
        except Exception as exc:
            best_failure = {
                "selected_prompt": str(prompt),
                "prompt_rank": int(prompt_rank),
                "attempted_prompt_count": int(len(prompt_candidates)),
                "confidence": 0.0,
                "iou": float(iou),
                "fallback_reason": f"detector_error:{type(exc).__name__}",
                "error_message": str(exc),
            }
            continue
        if not results:
            continue
        detection = _extract_best_box(results[0])
        if detection is not None:
            payload = {
                "selected_prompt": str(prompt),
                "prompt_rank": int(prompt_rank),
                "attempted_prompt_count": int(len(prompt_candidates)),
                "confidence": round(float(detection["confidence"]), 6),
                "iou": float(iou),
                "fallback_reason": "",
            }
            return detection, payload
        best_failure = {
            "selected_prompt": str(prompt),
            "prompt_rank": int(prompt_rank),
            "attempted_prompt_count": int(len(prompt_candidates)),
            "confidence": 0.0,
            "iou": float(iou),
            "fallback_reason": "no_detection",
        }
    return None, best_failure


def _save_image(image: Image.Image, output_dir: Path, dataset: str, view_name: str, split: str, image_id: str) -> str:
    target = output_dir / "views" / dataset / view_name / split / f"{image_id}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(target, quality=95)
    return target.relative_to(output_dir).as_posix()


def _save_image_relative_to_repo(
    image: Image.Image,
    output_dir: Path,
    repo_root: Path,
    dataset: str,
    view_name: str,
    split: str,
    image_id: str,
) -> str:
    relative = _save_image(image=image, output_dir=output_dir, dataset=dataset, view_name=view_name, split=split, image_id=image_id)
    return (output_dir.relative_to(repo_root) / relative).as_posix()


def _sam_subject_pipeline(
    *,
    image: Image.Image,
    dataset: str,
    sam_resources: Any,
    threshold: float,
    mask_threshold: float,
) -> dict[str, Any]:
    prompts = get_prompt_candidates_for_dataset(dataset)
    masks, sam_stats = run_single_inference_with_prompt_backoff(
        image=image,
        prompts=prompts,
        resources=sam_resources,
        threshold=float(threshold),
        mask_threshold=float(mask_threshold),
    )
    if masks is None:
        empty_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        return {
            "sam_stats": dict(sam_stats),
            "mask_stats": {"mask_count": 0, "union_area_ratio": 0.0, "largest_component_ratio": 0.0, "foreground_pixels": 0.0},
            "masked_image": image.convert("RGB").copy(),
            "masked_mask": empty_mask,
            "aligned_image": image.convert("RGB").copy(),
            "aligned_mask": empty_mask,
            "axis_stats": None,
            "orientation_reason": "no_mask",
            "rotation_applied_deg": 0.0,
            "subject_fallback_reason": "no_mask",
        }

    largest_mask, mask_stats = merge_masks(masks)
    masked_image = crop_to_union_mask(image, np.expand_dims(largest_mask.astype(np.uint8), axis=0)).convert("RGB")
    masked_mask = infer_mask_from_masked_rgb(masked_image, nonzero_threshold=1)
    axis_stats = None
    orientation_reason = "no_axis"
    rotation_applied_deg = 0.0
    aligned_image = masked_image.copy()
    aligned_mask = masked_mask.copy()

    try:
        axis_stats = compute_body_axis(largest_mask)
    except Exception:
        axis_stats = None

    decision = decide_orientation_application(
        axis_stats,
        min_foreground_pixels=int(DEFAULT_MIN_FOREGROUND_PIXELS[dataset]),
        min_area_ratio=float(DEFAULT_MIN_AREA_RATIO[dataset]),
        max_area_ratio=float(DEFAULT_MAX_AREA_RATIO[dataset]),
        min_axis_confidence=float(DEFAULT_MIN_AXIS_CONFIDENCE[dataset]),
        min_largest_component_ratio=float(DEFAULT_MIN_LARGEST_COMPONENT_RATIO[dataset]),
        largest_component_ratio=float(mask_stats.get("largest_component_ratio", 0.0)),
    )
    should_rotate, orientation_reason = _resolve_vertical_alignment_decision(
        dataset=dataset,
        decision=decision,
        axis_stats=axis_stats,
    )
    if should_rotate and axis_stats is not None:
        rotation_applied_deg = float(rotation_to_vertical(float(axis_stats["axis_angle_deg"])))
        aligned_image, aligned_mask = rotate_and_crop(
            image,
            largest_mask,
            rotation_applied_deg,
            background=(0, 0, 0),
            padding_ratio=resolve_crop_padding_ratio(dataset, default_padding_ratio=0.06),
            keep_background=False,
            canvas_fill_mode="constant",
        )
    return {
        "sam_stats": dict(sam_stats),
        "mask_stats": dict(mask_stats),
        "masked_image": masked_image,
        "masked_mask": masked_mask,
        "aligned_image": aligned_image.convert("RGB"),
        "aligned_mask": aligned_mask.astype(np.uint8),
        "axis_stats": axis_stats,
        "orientation_reason": orientation_reason,
        "rotation_applied_deg": rotation_applied_deg,
        "subject_fallback_reason": "" if masks is not None else "no_mask",
    }


def _run_texas_record(
    *,
    row: pd.Series,
    repo_root: Path,
    output_dir: Path,
    sam_resources: Any,
    yolo_model: Any,
    sam_threshold: float,
    sam_mask_threshold: float,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_max_det: int,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    split = str(row["split"])
    image_id = str(row["image_id"])
    source_path = str(row["path"])
    with Image.open(repo_root / source_path) as handle:
        image = handle.convert("RGB")

    subject = _sam_subject_pipeline(
        image=image,
        dataset=dataset,
        sam_resources=sam_resources,
        threshold=sam_threshold,
        mask_threshold=sam_mask_threshold,
    )
    scale_rgb, scale_mask, scale_payload = scale_normalize_texas_aligned_foreground(
        subject["aligned_image"],
        subject["aligned_mask"],
    )
    gray_rgb, gray_payload = grayscale_normalize_image(scale_rgb, focus_mask=scale_mask)
    center_body_rgb, _center_body_mask, center_body_payload = crop_texas_center_body_square(scale_rgb, scale_mask)

    detector_head, detector_head_payload = _run_yolo_prompt_backoff(
        yolo_model,
        subject["aligned_image"],
        TEXAS_YOLO_WORLD_PROMPT_CANDIDATES["head"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    detector_body, detector_body_payload = _run_yolo_prompt_backoff(
        yolo_model,
        subject["aligned_image"],
        TEXAS_YOLO_WORLD_PROMPT_CANDIDATES["body"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    hybrid_head, hybrid_head_payload = _run_yolo_prompt_backoff(
        yolo_model,
        scale_rgb,
        TEXAS_YOLO_WORLD_PROMPT_CANDIDATES["head"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    hybrid_body, hybrid_body_payload = _run_yolo_prompt_backoff(
        yolo_model,
        scale_rgb,
        TEXAS_YOLO_WORLD_PROMPT_CANDIDATES["body"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )

    detector_head_crop, detector_head_crop_payload = _crop_from_detection(subject["aligned_image"], detector_head)
    detector_body_crop, detector_body_crop_payload = _crop_from_detection(subject["aligned_image"], detector_body)
    hybrid_head_crop, hybrid_head_crop_payload = _crop_from_detection(scale_rgb, hybrid_head)
    hybrid_body_crop, hybrid_body_crop_payload = _crop_from_detection(scale_rgb, hybrid_body)

    record: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "image_id": image_id,
        "source_path": source_path,
    }
    view_map = {
        "original": image,
        "sam_masked": subject["masked_image"],
        "sam_aligned": subject["aligned_image"],
        "sam_aligned_scale_norm": scale_rgb,
        "gray_scale_norm": gray_rgb,
        "heuristic_center_body_square": center_body_rgb,
        "detector_head_overlay": _draw_box_overlay(subject["aligned_image"], detector_head, label="head", color=(255, 80, 80)),
        "detector_body_overlay": _draw_box_overlay(subject["aligned_image"], detector_body, label="body", color=(80, 180, 255)),
        "detector_head_crop": detector_head_crop,
        "detector_body_crop": detector_body_crop,
        "hybrid_head_crop": hybrid_head_crop,
        "hybrid_body_crop": hybrid_body_crop,
    }
    for view_name, view_image in view_map.items():
        record[f"{view_name}_path"] = _save_image_relative_to_repo(
            image=view_image,
            output_dir=output_dir,
            repo_root=repo_root,
            dataset=dataset,
            view_name=view_name,
            split=split,
            image_id=image_id,
        )

    record.update(
        _flatten_payload(dict(subject["sam_stats"]), "sam")
        | _flatten_payload(dict(subject["mask_stats"]), "mask")
        | _flatten_payload(scale_payload, "scale")
        | _flatten_payload(gray_payload, "gray")
        | _flatten_payload(center_body_payload, "center_body")
        | _flatten_payload(detector_head_payload, "detector_head")
        | _flatten_payload(detector_body_payload, "detector_body")
        | _flatten_payload(hybrid_head_payload, "hybrid_head")
        | _flatten_payload(hybrid_body_payload, "hybrid_body")
    )
    record["axis_angle_deg"] = float(subject["axis_stats"]["axis_angle_deg"]) if subject["axis_stats"] is not None else 0.0
    record["axis_confidence"] = float(subject["axis_stats"]["axis_confidence"]) if subject["axis_stats"] is not None else 0.0
    record["rotation_applied_deg"] = float(subject["rotation_applied_deg"])
    record["orientation_reason"] = str(subject["orientation_reason"])
    record["fallback_reason"] = _merge_reasons(
        subject["subject_fallback_reason"],
        scale_payload.get("fallback_reason"),
        gray_payload.get("fallback_reason"),
        center_body_payload.get("fallback_reason"),
        detector_head_payload.get("fallback_reason"),
        detector_body_payload.get("fallback_reason"),
        hybrid_head_payload.get("fallback_reason"),
        hybrid_body_payload.get("fallback_reason"),
    )
    record.update(
        build_texas_view_metadata(
            row=row.to_dict(),
            view_name="heuristic_center_body_square",
            grayscale_payload=gray_payload,
            scale_payload=scale_payload,
            crop_payload=center_body_payload,
            fallback_reason=record["fallback_reason"],
        )
    )
    return record


def _run_salamander_record(
    *,
    row: pd.Series,
    repo_root: Path,
    output_dir: Path,
    sam_resources: Any,
    yolo_model: Any,
    sam_threshold: float,
    sam_mask_threshold: float,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_max_det: int,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    split = str(row["split"])
    image_id = str(row["image_id"])
    source_path = str(row["path"])
    with Image.open(repo_root / source_path) as handle:
        image = handle.convert("RGB")

    subject = _sam_subject_pipeline(
        image=image,
        dataset=dataset,
        sam_resources=sam_resources,
        threshold=sam_threshold,
        mask_threshold=sam_mask_threshold,
    )
    scale_rgb, scale_mask, scale_payload = scale_normalize_salamander_aligned_foreground(
        subject["aligned_image"],
        subject["aligned_mask"],
        output_size=subject["aligned_image"].size,
    )
    heuristic_crops = generate_heuristic_end_middle_crops(scale_rgb, scale_mask, scale_payload=scale_payload)
    detector_head, detector_head_payload = _run_yolo_prompt_backoff(
        yolo_model,
        subject["aligned_image"],
        SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["head"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    detector_body, detector_body_payload = _run_yolo_prompt_backoff(
        yolo_model,
        subject["aligned_image"],
        SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["body"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    hybrid_head, hybrid_head_payload = _run_yolo_prompt_backoff(
        yolo_model,
        scale_rgb,
        SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["head"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )
    hybrid_body, hybrid_body_payload = _run_yolo_prompt_backoff(
        yolo_model,
        scale_rgb,
        SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES["body"],
        conf=yolo_conf,
        iou=yolo_iou,
        imgsz=yolo_imgsz,
        max_det=yolo_max_det,
    )

    detector_head_crop, _ = _crop_from_detection(subject["aligned_image"], detector_head)
    detector_body_crop, _ = _crop_from_detection(subject["aligned_image"], detector_body)
    hybrid_head_crop, _ = _crop_from_detection(scale_rgb, hybrid_head)
    hybrid_body_crop, _ = _crop_from_detection(scale_rgb, hybrid_body)

    record: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "image_id": image_id,
        "source_path": source_path,
    }
    view_map: dict[str, Image.Image] = {
        "original": image,
        "sam_masked": subject["masked_image"],
        "sam_aligned": subject["aligned_image"],
        "sam_aligned_scale_norm": scale_rgb,
        "heuristic_end_a": heuristic_crops["end_a"].rgb,
        "heuristic_middle": heuristic_crops["middle"].rgb,
        "heuristic_end_b": heuristic_crops["end_b"].rgb,
        "detector_head_overlay": _draw_box_overlay(subject["aligned_image"], detector_head, label="head", color=(255, 80, 80)),
        "detector_body_overlay": _draw_box_overlay(subject["aligned_image"], detector_body, label="body", color=(80, 180, 255)),
        "detector_head_crop": detector_head_crop,
        "detector_body_crop": detector_body_crop,
        "hybrid_head_crop": hybrid_head_crop,
        "hybrid_body_crop": hybrid_body_crop,
    }
    for view_name, view_image in view_map.items():
        record[f"{view_name}_path"] = _save_image_relative_to_repo(
            image=view_image,
            output_dir=output_dir,
            repo_root=repo_root,
            dataset=dataset,
            view_name=view_name,
            split=split,
            image_id=image_id,
        )
    record.update(
        _flatten_payload(dict(subject["sam_stats"]), "sam")
        | _flatten_payload(dict(subject["mask_stats"]), "mask")
        | _flatten_payload(scale_payload, "scale")
        | _flatten_payload(detector_head_payload, "detector_head")
        | _flatten_payload(detector_body_payload, "detector_body")
        | _flatten_payload(hybrid_head_payload, "hybrid_head")
        | _flatten_payload(hybrid_body_payload, "hybrid_body")
        | _flatten_payload(heuristic_crops["end_a"].metadata, "end_a")
        | _flatten_payload(heuristic_crops["middle"].metadata, "middle")
        | _flatten_payload(heuristic_crops["end_b"].metadata, "end_b")
    )
    record["axis_angle_deg"] = float(subject["axis_stats"]["axis_angle_deg"]) if subject["axis_stats"] is not None else 0.0
    record["axis_confidence"] = float(subject["axis_stats"]["axis_confidence"]) if subject["axis_stats"] is not None else 0.0
    record["rotation_applied_deg"] = float(subject["rotation_applied_deg"])
    record["orientation_reason"] = str(subject["orientation_reason"])
    record["fallback_reason"] = _merge_reasons(
        subject["subject_fallback_reason"],
        scale_payload.get("fallback_reason"),
        heuristic_crops["end_a"].metadata.get("fallback_reason"),
        heuristic_crops["middle"].metadata.get("fallback_reason"),
        heuristic_crops["end_b"].metadata.get("fallback_reason"),
        detector_head_payload.get("fallback_reason"),
        detector_body_payload.get("fallback_reason"),
        hybrid_head_payload.get("fallback_reason"),
        hybrid_body_payload.get("fallback_reason"),
    )
    return record


def _run_lynx_record(
    *,
    row: pd.Series,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    dataset = str(row["dataset"])
    split = str(row["split"])
    image_id = str(row["image_id"])
    source_path = str(row["path"])
    with Image.open(repo_root / source_path) as handle:
        image = handle.convert("RGB")

    gray_image, gray_payload = build_qualitative_lynx_view(image, mode="gray")
    hist_image, hist_payload = build_qualitative_lynx_view(image, mode="hist_norm")
    clahe_image, clahe_payload = build_qualitative_lynx_view(image, mode="clahe")
    record: dict[str, Any] = {
        "dataset": dataset,
        "split": split,
        "image_id": image_id,
        "source_path": source_path,
    }
    for view_name, view_image in {
        "original": image,
        "gray": gray_image.convert("RGB"),
        "gray_hist_norm": hist_image.convert("RGB"),
        "gray_clahe": clahe_image.convert("RGB"),
    }.items():
        record[f"{view_name}_path"] = _save_image_relative_to_repo(
            image=view_image,
            output_dir=output_dir,
            repo_root=repo_root,
            dataset=dataset,
            view_name=view_name,
            split=split,
            image_id=image_id,
        )
    record.update(
        _flatten_payload(gray_payload, "gray")
        | _flatten_payload(hist_payload, "hist")
        | _flatten_payload(clahe_payload, "clahe")
    )
    record["fallback_reason"] = _merge_reasons(
        gray_payload.get("fallback_reason"),
        hist_payload.get("fallback_reason"),
        clahe_payload.get("fallback_reason"),
    )
    return record


def _create_multi_view_contact_sheet(
    *,
    frame: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    view_columns: list[tuple[str, str]],
    columns: int = 1,
    thumb_size: tuple[int, int] = (180, 180),
) -> None:
    if frame.empty:
        return
    margin = 10
    header_h = 32
    label_h = 42
    panel_gap = 6
    panel_w, panel_h = thumb_size
    cell_w = len(view_columns) * panel_w + (len(view_columns) - 1) * panel_gap
    cell_h = panel_h + label_h
    rows = int(np.ceil(len(frame) / columns))
    canvas_w = margin * 2 + columns * cell_w + (columns - 1) * margin
    canvas_h = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for index, row in enumerate(frame.itertuples(index=False)):
        grid_x = index % columns
        grid_y = index // columns
        x0 = margin + grid_x * (cell_w + margin)
        y0 = start_y + grid_y * (cell_h + margin)
        for panel_index, (column, panel_title) in enumerate(view_columns):
            rel_path = getattr(row, column)
            panel_x = x0 + panel_index * (panel_w + panel_gap)
            with Image.open(repo_root / rel_path) as panel_image:
                thumb = ImageOps.pad(panel_image.convert("RGB"), thumb_size, color=(10, 10, 10))
            canvas.paste(thumb, (panel_x, y0))
            draw.text((panel_x, y0 + panel_h + 2), panel_title, fill=(40, 40, 40), font=font)
        label = f"{row.dataset} | {row.split} | {row.image_id}"
        draw.text((x0, y0 + panel_h + 20), label, fill=(30, 30, 30), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _build_texas_augmentor_preview(
    *,
    repo_root: Path,
    output_dir: Path,
    texas_df: pd.DataFrame,
    samples_per_image: int,
) -> pd.DataFrame:
    if texas_df.empty or int(samples_per_image) <= 0:
        return pd.DataFrame()
    try:
        import Augmentor
    except Exception:  # pragma: no cover - runtime only
        return pd.DataFrame()

    input_dir = output_dir / "augmentor_input" / "TexasHornedLizards"
    preview_dir = output_dir / "views" / "TexasHornedLizards" / "augmentor_preview" / "train"
    input_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    input_paths: list[Path] = []
    for row in texas_df.itertuples(index=False):
        source_rel = getattr(row, "gray_scale_norm_path", "")
        if not source_rel:
            continue
        source_path = repo_root / source_rel
        if not source_path.exists():
            continue
        target_path = input_dir / f"{row.image_id}.jpg"
        shutil.copy2(source_path, target_path)
        input_paths.append(target_path)
    if not input_paths:
        return pd.DataFrame()

    temp_root = Path(tempfile.mkdtemp(prefix="augmentor_texas_", dir=str(output_dir)))
    try:
        pipeline = Augmentor.Pipeline(source_directory=str(input_dir), output_directory=str(temp_root))
        pipeline.rotate(probability=0.7, max_left_rotation=12, max_right_rotation=12)
        pipeline.flip_left_right(probability=0.5)
        pipeline.random_contrast(probability=0.5, min_factor=0.85, max_factor=1.15)
        pipeline.random_brightness(probability=0.4, min_factor=0.9, max_factor=1.1)
        pipeline.sample(int(len(input_paths) * samples_per_image), multi_threaded=False)
        rows: list[dict[str, Any]] = []
        for index, aug_path in enumerate(sorted(temp_root.glob("*")), start=1):
            if not aug_path.is_file():
                continue
            final_path = preview_dir / aug_path.name
            shutil.move(str(aug_path), final_path)
            rows.append(
                {
                    "dataset": "TexasHornedLizards",
                    "augmentor_preview_path": final_path.relative_to(repo_root).as_posix(),
                    "rotate_probability": 0.7,
                    "rotate_max_left": 12,
                    "rotate_max_right": 12,
                    "flip_left_right_probability": 0.5,
                    "random_contrast_probability": 0.5,
                    "random_contrast_min_factor": 0.85,
                    "random_contrast_max_factor": 1.15,
                    "random_brightness_probability": 0.4,
                    "random_brightness_min_factor": 0.9,
                    "random_brightness_max_factor": 1.1,
                    "preview_index": index,
                }
            )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    return pd.DataFrame(rows)


def _write_summary(
    *,
    output_dir: Path,
    repo_root: Path,
    sample_df: pd.DataFrame,
    texas_df: pd.DataFrame,
    salamander_df: pd.DataFrame,
    lynx_df: pd.DataFrame,
    augmentor_df: pd.DataFrame,
    config: dict[str, Any],
) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    param_table = parameter_explanation_rows()
    summary_lines = [
        "# Stage-1 Qualitative Preprocessing Review",
        "",
        "## Goal",
        "",
        "- This stage does not replace the default manifests.",
        "- It exports dataset-specific cached views for qualitative review, fallback analysis, and later full-scale approval.",
        "",
        "## Scope",
        "",
        f"- Sampled datasets: `{', '.join(sorted(sample_df['dataset'].unique().tolist()))}`.",
        f"- Samples per split: `{int(config['samples_per_split'])}`.",
        f"- SAM thresholds: `threshold={float(config['sam_threshold']):.2f}`, `mask_threshold={float(config['sam_mask_threshold']):.2f}`.",
        f"- YOLO-World: `model={config['yolo_model_name']}`, `conf={float(config['yolo_conf']):.2f}`, `iou={float(config['yolo_iou']):.2f}`, `imgsz={int(config['yolo_imgsz'])}`.",
        "",
        "## Key Parameters",
        "",
    ]
    for row in param_table.itertuples(index=False):
        summary_lines.append(f"- `{row.parameter}`: {row.meaning}")

    summary_lines.extend(
        [
            "",
            "## Prompt Backoff",
            "",
            f"- `Texas whole`: `{list(TEXAS_YOLO_WORLD_PROMPT_CANDIDATES['whole'])}`.",
            f"- `Texas head`: `{list(TEXAS_YOLO_WORLD_PROMPT_CANDIDATES['head'])}`.",
            f"- `Texas body`: `{list(TEXAS_YOLO_WORLD_PROMPT_CANDIDATES['body'])}`.",
            f"- `Salamander whole`: `{list(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES['whole'])}`.",
            f"- `Salamander head`: `{list(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES['head'])}`.",
            f"- `Salamander body`: `{list(SALAMANDER_YOLOWORLD_PROMPT_CANDIDATES['body'])}`.",
            "",
        ]
    )

    sections = [
        (
            "Texas",
            texas_df,
            [
                ("original_path", "orig"),
                ("sam_masked_path", "sam"),
                ("sam_aligned_scale_norm_path", "scale"),
                ("gray_scale_norm_path", "gray"),
                ("heuristic_center_body_square_path", "body"),
                ("hybrid_head_crop_path", "hyb_head"),
            ],
            "Texas qualitative flow shows SAM 主体、竖直统一、偏大尺度、灰度稳定化，以及以主体中心向外扩的 body square crop. Detector overlays run on the SAM-aligned subject; hybrid detector crops run on the scale-normalized branch.",
        ),
        (
            "Salamander",
            salamander_df,
            [
                ("original_path", "orig"),
                ("sam_masked_path", "sam"),
                ("sam_aligned_scale_norm_path", "scale"),
                ("heuristic_end_a_path", "end_a"),
                ("heuristic_middle_path", "middle"),
                ("hybrid_head_crop_path", "hyb_head"),
            ],
            "Salamander qualitative flow shows SAM 主体、竖直统一、偏大尺度，以及 center-trimmed trunk partitions versus detector-guided head crops. Detector overlays run on the SAM-aligned subject; hybrid detector crops run on the scale-normalized branch.",
        ),
        (
            "Lynx",
            lynx_df,
            [
                ("original_path", "orig"),
                ("gray_path", "gray"),
                ("gray_hist_norm_path", "hist"),
                ("gray_clahe_path", "clahe"),
            ],
            "Lynx qualitative flow compares gray-only, histogram-normalized gray, and CLAHE-enhanced gray views.",
        ),
    ]
    qualitative_dir = output_dir / "qualitative"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    for section_name, section_df, view_columns, description in sections:
        if section_df.empty:
            continue
        sheet_path = qualitative_dir / f"{section_name.lower()}_qualitative_board_v1.jpg"
        _create_multi_view_contact_sheet(
            frame=section_df,
            repo_root=repo_root,
            output_path=sheet_path,
            title=f"{section_name} qualitative review",
            view_columns=view_columns,
        )
        relative_sheet = Path("..") / sheet_path.relative_to(output_dir)
        summary_lines.extend(
            [
                f"## {section_name}",
                "",
                f"- {description}",
                f"- Rows: `{len(section_df)}`.",
                "",
                f"![{section_name} qualitative]({relative_sheet.as_posix()})",
                "",
            ]
        )
        summary_columns = [column for column in section_df.columns if column in {"dataset", "split", "image_id", "axis_confidence", "scale_scale_factor", "fallback_reason"}]
        if summary_columns:
            preview_df = section_df.loc[:, summary_columns].head(8).copy()
            summary_lines.append(preview_df.to_markdown(index=False))
            summary_lines.append("")

    if not augmentor_df.empty:
        summary_lines.extend(
            [
                "## Texas Augmentor Preview",
                "",
                "- The preview uses `Augmentor.Pipeline` on the grayscale-normalized Texas branch to visualize low-risk augmentation settings before full-scale expansion.",
                f"- Generated preview rows: `{len(augmentor_df)}`.",
                "",
            ]
        )
        preview_cols = [column for column in augmentor_df.columns if column != "dataset"]
        summary_lines.append(augmentor_df.loc[:, preview_cols].head(8).to_markdown(index=False))
        summary_lines.append("")
    else:
        summary_lines.extend(
            [
                "## Texas Augmentor Preview",
                "",
                "- Augmentor preview was skipped or produced no rows in this run.",
                "",
            ]
        )

    summary_lines.extend(
        [
            "## Reading Notes",
            "",
            "- `Texas`: check whether `heuristic_center_body_square` suppresses limbs and head/tail while keeping the dorsal black-dot pattern stable.",
            "- `Salamander`: compare the center-trimmed trunk partitions against `hybrid_head_crop` to judge whether detector-based local views are semantically stronger.",
            "- `Lynx`: compare `gray_hist_norm` and `gray_clahe` to see whether brightness drift is reduced without crushing fur texture.",
            "",
        ]
    )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def _write_manifests(output_dir: Path, frame: pd.DataFrame) -> dict[str, Path]:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for dataset, dataset_df in frame.groupby("dataset", dropna=False):
        path = manifests_dir / f"{str(dataset).lower()}_view_manifest_v1.csv"
        dataset_df.sort_values(["split", "image_id"]).to_csv(path, index=False)
        manifest_paths[str(dataset)] = path
    return manifest_paths


def run_qualitative_preprocessing_review(
    *,
    repo_root: Path,
    output_dir: Path,
    metadata_path: Path | None = None,
    datasets: list[str] | None = None,
    samples_per_split: int = DEFAULT_SAMPLES_PER_SPLIT,
    sample_seed: int = 42,
    device: str = "cuda:0",
    sam_threshold: float = 0.5,
    sam_mask_threshold: float = 0.5,
    yolo_model_name: str = DEFAULT_YOLO_MODEL,
    yolo_conf: float = DEFAULT_YOLO_CONF,
    yolo_iou: float = DEFAULT_YOLO_IOU,
    yolo_imgsz: int = DEFAULT_YOLO_IMGSZ,
    yolo_max_det: int = DEFAULT_YOLO_MAX_DET,
    texas_augmentor_samples_per_image: int = DEFAULT_TEXAS_AUGMENTOR_SAMPLES_PER_IMAGE,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if metadata_path is None:
        metadata_path = repo_root / "metadata.csv"
    selected_datasets = list(DEFAULT_DATASETS if datasets is None else datasets)
    metadata_df = load_metadata(metadata_path)
    sample_df = sample_rows_by_dataset(
        metadata_df=metadata_df,
        sample_seed=int(sample_seed),
        samples_per_split=int(samples_per_split),
        datasets=selected_datasets,
    ).sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    if sample_df.empty:
        raise ValueError("No sampled rows available for the requested datasets.")

    sam_resources = _load_sam_resources(device=device)
    yolo_model = _load_yolo_world_model(yolo_model_name, device=device)

    texas_records: list[dict[str, Any]] = []
    salamander_records: list[dict[str, Any]] = []
    lynx_records: list[dict[str, Any]] = []
    for row in sample_df.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        dataset = str(row.dataset)
        if dataset == "TexasHornedLizards":
            texas_records.append(
                _run_texas_record(
                    row=row_series,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    sam_resources=sam_resources,
                    yolo_model=yolo_model,
                    sam_threshold=sam_threshold,
                    sam_mask_threshold=sam_mask_threshold,
                    yolo_conf=yolo_conf,
                    yolo_iou=yolo_iou,
                    yolo_imgsz=yolo_imgsz,
                    yolo_max_det=yolo_max_det,
                )
            )
        elif dataset == "SalamanderID2025":
            salamander_records.append(
                _run_salamander_record(
                    row=row_series,
                    repo_root=repo_root,
                    output_dir=output_dir,
                    sam_resources=sam_resources,
                    yolo_model=yolo_model,
                    sam_threshold=sam_threshold,
                    sam_mask_threshold=sam_mask_threshold,
                    yolo_conf=yolo_conf,
                    yolo_iou=yolo_iou,
                    yolo_imgsz=yolo_imgsz,
                    yolo_max_det=yolo_max_det,
                )
            )
        elif dataset == "LynxID2025":
            lynx_records.append(
                _run_lynx_record(
                    row=row_series,
                    repo_root=repo_root,
                    output_dir=output_dir,
                )
            )

    texas_df = pd.DataFrame(texas_records)
    salamander_df = pd.DataFrame(salamander_records)
    lynx_df = pd.DataFrame(lynx_records)
    combined_df = pd.concat([frame for frame in [texas_df, salamander_df, lynx_df] if not frame.empty], ignore_index=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not combined_df.empty:
        combined_df.to_csv(tables_dir / "qualitative_view_records_v1.csv", index=False)
    manifest_paths = _write_manifests(output_dir, combined_df) if not combined_df.empty else {}
    augmentor_df = _build_texas_augmentor_preview(
        repo_root=repo_root,
        output_dir=output_dir,
        texas_df=texas_df,
        samples_per_image=int(texas_augmentor_samples_per_image),
    )
    if not augmentor_df.empty:
        augmentor_df.to_csv(tables_dir / "texas_augmentor_preview_v1.csv", index=False)

    config = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "metadata_path": str(metadata_path),
        "datasets": selected_datasets,
        "samples_per_split": int(samples_per_split),
        "sample_seed": int(sample_seed),
        "device": str(device),
        "sam_threshold": float(sam_threshold),
        "sam_mask_threshold": float(sam_mask_threshold),
        "yolo_model_name": str(yolo_model_name),
        "yolo_conf": float(yolo_conf),
        "yolo_iou": float(yolo_iou),
        "yolo_imgsz": int(yolo_imgsz),
        "yolo_max_det": int(yolo_max_det),
        "texas_augmentor_samples_per_image": int(texas_augmentor_samples_per_image),
    }
    summary_path = _write_summary(
        output_dir=output_dir,
        repo_root=repo_root,
        sample_df=sample_df,
        texas_df=texas_df,
        salamander_df=salamander_df,
        lynx_df=lynx_df,
        augmentor_df=augmentor_df,
        config=config,
    )
    return {
        "summary_path": summary_path,
        "records_path": tables_dir / "qualitative_view_records_v1.csv",
        "manifest_paths": manifest_paths,
        "augmentor_preview_path": tables_dir / "texas_augmentor_preview_v1.csv",
    }


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DEFAULT_TEXAS_AUGMENTOR_SAMPLES_PER_IMAGE",
    "parameter_explanation_rows",
    "run_qualitative_preprocessing_review",
]
