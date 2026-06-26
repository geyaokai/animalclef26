from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from .body_orientation_probe import compute_body_axis, rotate_and_crop, rotation_to_horizontal
from .descriptor_baselines import dataframe_to_markdown_table
from .sam_orb_veto import infer_mask_from_masked_rgb

try:  # pragma: no cover - optional in light env
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_MANIFEST_PATH = Path("artifacts/manifests/v1/tables/manifest_test_body_axis_unsigned_rgb_v1.csv")
DEFAULT_REVIEW_INDEX_PATH = Path("artifacts/analysis/texas_orb_constraint_graph_v1/tables/auto_pair_review_index_v1.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_tail_prior_review_v1")


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _load_rgb(repo_root: Path, image_path: str) -> Image.Image:
    with Image.open(repo_root / str(image_path)) as image:
        return image.convert("RGB")


def _resize_with_pad(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (int(width), int(height)))
    canvas = Image.new("RGB", (int(width), int(height)), (246, 246, 246))
    offset = ((int(width) - fitted.width) // 2, (int(height) - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _placeholder_panel(text: str, *, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (int(width), int(height)), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width - 1, height - 1), outline=(188, 188, 188), width=2)
    draw.multiline_text((14, 14), text, fill=(60, 60, 60), font=_font(16), spacing=6)
    return canvas


def _overlay_mask(base_image: Image.Image, mask: np.ndarray, *, color: str, alpha: float = 0.45) -> Image.Image:
    rgb = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    binary = np.asarray(mask, dtype=np.uint8) > 0
    overlay = np.array(ImageColor.getrgb(color), dtype=np.float32)
    arr = rgb.astype(np.float32)
    arr[binary] = arr[binary] * (1.0 - float(alpha)) + overlay * float(alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _edge_slice(width: int, *, edge_fraction: float, min_columns: int) -> int:
    return max(int(min_columns), int(round(float(width) * float(edge_fraction))))


def infer_tail_side_from_mask(
    mask: np.ndarray,
    *,
    edge_fraction: float = 0.18,
    min_columns: int = 18,
    balance_margin: float = 0.08,
) -> dict[str, Any]:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    if binary.ndim != 2 or not binary.any():
        return {
            "tail_side": "missing",
            "tail_confidence": 0.0,
            "left_span": 0.0,
            "right_span": 0.0,
            "edge_width": 0,
        }
    profile = binary.sum(axis=0).astype(np.float32)
    width = int(binary.shape[1])
    edge_width = min(width, _edge_slice(width, edge_fraction=edge_fraction, min_columns=min_columns))
    left_span = float(np.median(profile[:edge_width])) if edge_width > 0 else 0.0
    right_span = float(np.median(profile[-edge_width:])) if edge_width > 0 else 0.0
    larger = max(left_span, right_span, 1e-6)
    confidence = float(abs(left_span - right_span) / larger)
    if confidence < float(balance_margin):
        tail_side = "uncertain"
    else:
        tail_side = "left" if left_span < right_span else "right"
    return {
        "tail_side": tail_side,
        "tail_confidence": round(confidence, 6),
        "left_span": round(left_span, 4),
        "right_span": round(right_span, 4),
        "edge_width": int(edge_width),
    }


def extract_black_pattern_mask(
    image: Image.Image,
    foreground_mask: np.ndarray,
    *,
    fallback_quantile: float = 0.28,
) -> tuple[np.ndarray, dict[str, Any]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    binary = (np.asarray(foreground_mask, dtype=np.uint8) > 0)
    if not binary.any():
        empty = np.zeros(binary.shape[:2], dtype=np.uint8)
        return empty, {"black_threshold": 0, "black_ratio": 0.0}
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    fg_values = gray[binary]
    if cv2 is not None and fg_values.size >= 16:
        threshold, _ = cv2.threshold(fg_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black_threshold = int(threshold)
    else:
        black_threshold = int(np.quantile(fg_values.astype(np.float32), float(fallback_quantile)))
    black_mask = ((gray <= black_threshold) & binary).astype(np.uint8)
    return black_mask, {
        "black_threshold": int(black_threshold),
        "black_ratio": round(float(black_mask.sum() / max(binary.sum(), 1)), 6),
    }


def _build_masked_aligned_preview(masked_image: Image.Image) -> tuple[Image.Image, np.ndarray, dict[str, Any]]:
    inferred_mask = infer_mask_from_masked_rgb(masked_image)
    axis_stats = compute_body_axis(inferred_mask)
    if axis_stats is None:
        return masked_image.copy(), inferred_mask.astype(np.uint8), {
            "axis_confidence": 0.0,
            "axis_angle_deg": 0.0,
            "rotation_applied_deg": 0.0,
            "alignment_status": "skip_no_axis",
        }
    rotation_applied_deg = rotation_to_horizontal(float(axis_stats["axis_angle_deg"]))
    aligned_rgb, aligned_mask = rotate_and_crop(
        masked_image,
        inferred_mask,
        rotation_applied_deg,
        background=(0, 0, 0),
        keep_background=False,
        canvas_fill_mode="constant",
    )
    return aligned_rgb, aligned_mask.astype(np.uint8), {
        "axis_confidence": float(axis_stats["axis_confidence"]),
        "axis_angle_deg": float(axis_stats["axis_angle_deg"]),
        "rotation_applied_deg": float(rotation_applied_deg),
        "alignment_status": "apply",
    }


def _tail_overlay(image: Image.Image, tail_payload: dict[str, Any]) -> Image.Image:
    side = str(tail_payload.get("tail_side", "missing"))
    edge_width = int(tail_payload.get("edge_width", 0))
    if image.width <= 0 or image.height <= 0 or edge_width <= 0:
        return image
    canvas = image.convert("RGBA")
    draw = ImageDraw.Draw(canvas, mode="RGBA")
    if side == "left":
        draw.rectangle((0, 0, edge_width, image.height), fill=(235, 87, 87, 84))
        draw.rectangle((image.width - edge_width, 0, image.width, image.height), fill=(66, 133, 244, 52))
    elif side == "right":
        draw.rectangle((0, 0, edge_width, image.height), fill=(66, 133, 244, 52))
        draw.rectangle((image.width - edge_width, 0, image.width, image.height), fill=(235, 87, 87, 84))
    else:
        draw.rectangle((0, 0, edge_width, image.height), fill=(246, 190, 0, 58))
        draw.rectangle((image.width - edge_width, 0, image.width, image.height), fill=(246, 190, 0, 58))
    return canvas.convert("RGB")


def _member_tags(image_id: str, *, pair_ids: set[str]) -> str:
    return "pair" if str(image_id) in pair_ids else "context"


def _member_preview_payload(repo_root: Path, row: pd.Series) -> dict[str, Any]:
    original_image = _load_rgb(repo_root=repo_root, image_path=str(row["path"]))
    sam_available = bool(row.get("sam_masked_rgb_v1_applied", False)) and str(row.get("sam_masked_rgb_v1_export_path", "")).strip()
    body_available = bool(row.get("body_axis_unsigned_rgb_v1_applied", False)) and str(row.get("body_axis_unsigned_rgb_v1_export_path", "")).strip()

    if sam_available:
        sam_image = _load_rgb(repo_root=repo_root, image_path=str(row["sam_masked_rgb_v1_export_path"]))
        aligned_masked, aligned_mask, alignment_payload = _build_masked_aligned_preview(sam_image)
        tail_payload = infer_tail_side_from_mask(aligned_mask)
        black_mask, black_payload = extract_black_pattern_mask(aligned_masked, aligned_mask)
        tail_image = _tail_overlay(aligned_masked, tail_payload)
        black_overlay = _overlay_mask(aligned_masked, black_mask, color="#00bcd4", alpha=0.55)
    else:
        sam_image = None
        aligned_masked = None
        alignment_payload = {
            "axis_confidence": 0.0,
            "axis_angle_deg": 0.0,
            "rotation_applied_deg": 0.0,
            "alignment_status": "no_sam",
        }
        tail_payload = {
            "tail_side": "missing",
            "tail_confidence": 0.0,
            "left_span": 0.0,
            "right_span": 0.0,
            "edge_width": 0,
        }
        black_payload = {"black_threshold": 0, "black_ratio": 0.0}
        tail_image = None
        black_overlay = None
    if body_available:
        body_image = _load_rgb(repo_root=repo_root, image_path=str(row["body_axis_unsigned_rgb_v1_export_path"]))
    else:
        fallback_body = str(row.get("body_axis_unsigned_rgb_v1_resolved_path_v1", "")).strip() or str(row["path"])
        body_image = _load_rgb(repo_root=repo_root, image_path=fallback_body)

    return {
        "original_image": original_image,
        "sam_image": sam_image,
        "body_image": body_image,
        "tail_image": tail_image,
        "black_overlay": black_overlay,
        "alignment_payload": alignment_payload,
        "tail_payload": tail_payload,
        "black_payload": black_payload,
        "sam_available": sam_available,
        "body_available": body_available,
    }


def _build_cluster_board(
    *,
    repo_root: Path,
    review_row: pd.Series,
    member_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    panel_w = 176
    panel_h = 132
    gap = 10
    margin = 18
    header_h = 92
    row_text_h = 54
    row_h = panel_h + row_text_h
    title_font = _font(22)
    body_font = _font(15)
    small_font = _font(13)
    width = margin * 2 + panel_w * 5 + gap * 4
    height = margin * 2 + header_h + len(member_df) * row_h + max(0, len(member_df) - 1) * gap
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Texas Tail Prior Review", fill=(20, 20, 20), font=title_font)
    header_lines = [
        f"cluster={int(review_row['base_cluster_id'])} | size={int(review_row['base_cluster_size'])} | review_rank={int(review_row['review_rank'])}",
        (
            f"auto_pair={review_row['image_id']} vs {review_row['neighbor_image_id']} | "
            f"orb_local={float(review_row['orb_local_score']):.3f} | "
            f"orb_inliers={int(review_row['orb_inliers'])} | "
            f"xgb={float(review_row['xgb_same_identity_prob']):.3f}"
        ),
        "Panels: original | sam_masked | body_aligned | tail_guess(red=tail, blue=opposite) | black_pattern(cyan)",
    ]
    for line_index, line in enumerate(header_lines):
        draw.text((margin, margin + 34 + line_index * 20), line, fill=(48, 48, 48), font=body_font)

    pair_ids = {str(review_row["image_id"]), str(review_row["neighbor_image_id"])}
    stat_rows: list[dict[str, Any]] = []
    y0 = margin + header_h
    for row_index, row in enumerate(member_df.itertuples(index=False)):
        y = y0 + row_index * (row_h + gap)
        payload = _member_preview_payload(repo_root=repo_root, row=pd.Series(row._asdict()))
        panels = [
            _resize_with_pad(payload["original_image"], width=panel_w, height=panel_h),
            _resize_with_pad(payload["sam_image"], width=panel_w, height=panel_h) if payload["sam_image"] is not None else _placeholder_panel("sam_mask\nmissing", width=panel_w, height=panel_h),
            _resize_with_pad(payload["body_image"], width=panel_w, height=panel_h),
            _resize_with_pad(payload["tail_image"], width=panel_w, height=panel_h) if payload["tail_image"] is not None else _placeholder_panel("tail_guess\nunavailable", width=panel_w, height=panel_h),
            _resize_with_pad(payload["black_overlay"], width=panel_w, height=panel_h) if payload["black_overlay"] is not None else _placeholder_panel("black_pattern\nunavailable", width=panel_w, height=panel_h),
        ]
        for panel_index, panel in enumerate(panels):
            x = margin + panel_index * (panel_w + gap)
            canvas.paste(panel, (x, y))
        member_tag = _member_tags(str(row.image_id), pair_ids=pair_ids)
        tail_side = str(payload["tail_payload"]["tail_side"])
        tail_conf = float(payload["tail_payload"]["tail_confidence"])
        row_caption = (
            f"{row.image_id} | {member_tag} | "
            f"sam={'ok' if payload['sam_available'] else 'miss'} | "
            f"body={'ok' if payload['body_available'] else 'fallback'} | "
            f"tail={tail_side}({tail_conf:.2f}) | "
            f"black_ratio={float(payload['black_payload']['black_ratio']):.3f}"
        )
        aux_caption = (
            f"axis_conf={float(payload['alignment_payload']['axis_confidence']):.3f} | "
            f"left_span={float(payload['tail_payload']['left_span']):.1f} | "
            f"right_span={float(payload['tail_payload']['right_span']):.1f}"
        )
        draw.text((margin, y + panel_h + 4), row_caption, fill=(34, 34, 34), font=body_font)
        draw.text((margin, y + panel_h + 26), aux_caption, fill=(92, 92, 92), font=small_font)

        stat_rows.append(
            {
                "base_cluster_id": int(review_row["base_cluster_id"]),
                "image_id": str(row.image_id),
                "member_tag": member_tag,
                "sam_available": bool(payload["sam_available"]),
                "body_available": bool(payload["body_available"]),
                "axis_confidence": round(float(payload["alignment_payload"]["axis_confidence"]), 6),
                "tail_side": tail_side,
                "tail_confidence": round(tail_conf, 6),
                "left_span": round(float(payload["tail_payload"]["left_span"]), 4),
                "right_span": round(float(payload["tail_payload"]["right_span"]), 4),
                "black_ratio": round(float(payload["black_payload"]["black_ratio"]), 6),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return pd.DataFrame(stat_rows)


def run_texas_tail_prior_review(
    *,
    repo_root: Path,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    review_index_path: Path = DEFAULT_REVIEW_INDEX_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    resolved_manifest_path = manifest_path if manifest_path.is_absolute() else (repo_root / manifest_path)
    resolved_review_index_path = review_index_path if review_index_path.is_absolute() else (repo_root / review_index_path)
    resolved_output_dir = output_dir if output_dir.is_absolute() else (repo_root / output_dir)
    resolved_manifest_path = resolved_manifest_path.resolve()
    resolved_review_index_path = resolved_review_index_path.resolve()
    resolved_output_dir = resolved_output_dir.resolve()

    tables_dir = resolved_output_dir / "tables"
    figures_dir = resolved_output_dir / "figures"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(resolved_manifest_path).copy()
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    manifest_df = manifest_df[manifest_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    review_index_df = pd.read_csv(resolved_review_index_path).copy()
    review_index_df["image_id"] = review_index_df["image_id"].astype(str)
    review_index_df["neighbor_image_id"] = review_index_df["neighbor_image_id"].astype(str)
    review_index_df["base_cluster_id"] = pd.to_numeric(review_index_df["base_cluster_id"], errors="coerce").fillna(-1).astype(int)
    review_index_df["cluster_image_ids"] = review_index_df["cluster_image_ids"].astype(str)

    figure_rows: list[dict[str, Any]] = []
    stat_frames: list[pd.DataFrame] = []
    for review_row in review_index_df.sort_values("review_rank").itertuples(index=False):
        member_ids = [value for value in str(review_row.cluster_image_ids).split("|") if value]
        member_df = manifest_df[manifest_df["image_id"].astype(str).isin(member_ids)].copy().reset_index(drop=True)
        figure_path = figures_dir / f"cluster_{int(review_row.base_cluster_id):03d}_review_{int(review_row.review_rank):02d}.jpg"
        stat_df = _build_cluster_board(
            repo_root=repo_root,
            review_row=pd.Series(review_row._asdict()),
            member_df=member_df,
            output_path=figure_path,
        )
        if not stat_df.empty:
            stat_df["review_rank"] = int(review_row.review_rank)
            stat_frames.append(stat_df)
        figure_rows.append(
            {
                "review_rank": int(review_row.review_rank),
                "base_cluster_id": int(review_row.base_cluster_id),
                "figure_path": _path_ref(reports_dir, figure_path),
                "pair_key": f"{review_row.image_id}|{review_row.neighbor_image_id}",
                "cluster_size": int(review_row.base_cluster_size),
            }
        )

    figure_index_df = pd.DataFrame(figure_rows).sort_values("review_rank").reset_index(drop=True)
    stat_df = pd.concat(stat_frames, ignore_index=True) if stat_frames else pd.DataFrame()
    figure_index_path = tables_dir / "cluster_figure_index_v1.csv"
    stat_path = tables_dir / "tail_prior_stats_v1.csv"
    figure_index_df.to_csv(figure_index_path, index=False)
    stat_df.to_csv(stat_path, index=False)

    summary_lines = [
        "# Texas Tail Prior Review v1",
        "",
        "## Goal",
        "",
        "- 先定性检查 Texas 的 `SAM mask`、`body aligned` 是否靠谱。",
        "- 再看一个很粗的 `tail-side` 先验值不值得信：当前规则只是按“对齐后更细的一端更像 tail”做的几何猜测。",
        "- 同时把 `black pattern` 粗提出来，帮助你判断后续是否该直接改成“黑斑点 ROI + ORB”而不是先做 tail cascade。",
        "",
        "## How To Read",
        "",
        "- 每个 cluster board 的 5 列依次是：`original`、`sam_masked`、`body_aligned`、`tail_guess`、`black_pattern`。",
        "- `tail_guess` 里红色区域是当前猜的 tail 端，蓝色区域是另一端；如果你觉得红色经常标错，这个 tail 先验就不该直接进主线。",
        "- `black_pattern` 里青色是当前粗提的黑纹区域；如果它已经能稳定抓住你人工真正看的斑点，那下一步就更应该走“黑纹 mask + ORB”。",
        "",
        "## Inputs",
        "",
        f"- `manifest_path`: `{_path_ref(repo_root, resolved_manifest_path)}`",
        f"- `review_index_path`: `{_path_ref(repo_root, resolved_review_index_path)}`",
        "",
        "## Figure Index",
        "",
        dataframe_to_markdown_table(figure_index_df) if not figure_index_df.empty else "_No figures generated._",
        "",
        "## Image Stats",
        "",
        dataframe_to_markdown_table(stat_df) if not stat_df.empty else "_No stats generated._",
        "",
    ]
    for row in figure_index_df.itertuples(index=False):
        summary_lines.extend(
            [
                f"## Review {int(row.review_rank)} | Cluster {int(row.base_cluster_id)}",
                "",
                f"- pair: `{row.pair_key}` | cluster_size: `{int(row.cluster_size)}`",
                "",
                f"![cluster_{int(row.base_cluster_id)}]({row.figure_path})",
                "",
            ]
        )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "summary_path": summary_path,
        "figure_index_path": figure_index_path,
        "stat_path": stat_path,
        "figures_dir": figures_dir,
    }
