#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ANALYSIS_DIR = Path("artifacts/analysis/salamander_yellow_orb_local_v1")
DEFAULT_OUTPUT_DIRNAME = "review_pack_v1"
SALAMANDER_DATASET = "SalamanderID2025"


def _path_ref(base: Path, target: Path) -> str:
    return str(target.relative_to(base)).replace("\\", "/")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask_overlay(mask_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(mask_path) as mask_image:
        mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    overlay[mask] = np.array([255, 220, 0], dtype=np.uint8)
    image = Image.fromarray(overlay, mode="RGB")
    if image.size != size:
        image = image.resize(size, Image.NEAREST)
    return image


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as mask_image:
        return (np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _blend_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    overlay = rgb.copy()
    tint = np.asarray(color, dtype=np.float32)
    base = overlay.astype(np.float32)
    base[mask_bool] = (1.0 - float(alpha)) * base[mask_bool] + float(alpha) * tint
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _resize_with_pad(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (int(width), int(height)))
    canvas = Image.new("RGB", (int(width), int(height)), (248, 248, 248))
    offset = ((int(width) - fitted.width) // 2, (int(height) - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _draw_points(image: Image.Image, points: np.ndarray, *, radius: int = 3) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x, y in np.asarray(points, dtype=np.float32):
        draw.ellipse(
            (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
            outline=(255, 40, 40),
            width=2,
        )
    return output


def _resize_mask(mask: np.ndarray, *, width: int, height: int) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255, mode="L")
    resized = image.resize((int(width), int(height)), Image.NEAREST)
    return (np.asarray(resized, dtype=np.uint8) > 0).astype(np.uint8)


def _build_focus_feature_map(
    *,
    focus_df: pd.DataFrame,
    repo_root: Path,
    max_side: int,
    fast_threshold: int,
    clahe_clip_limit: float,
    orb_features: int,
) -> dict[str, dict[str, object]]:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("yellow-band-orb review pack requires OpenCV in the active environment.") from exc

    from animalclef_analysis.salamander_yellow_orb_local import (
        DEFAULT_BAND_DILATE_RADIUS,
        DEFAULT_BAND_ERODE_RADIUS,
        DEFAULT_BAND_MIN_PIXELS,
        YELLOW_FOCUS_MASK_PATH_COLUMN,
        YELLOW_FOCUS_PATH_COLUMN,
        build_yellow_band_mask,
    )

    valid_df = focus_df[focus_df[YELLOW_FOCUS_PATH_COLUMN].fillna("").astype(str).ne("")].copy().reset_index(drop=True)
    if valid_df.empty:
        return {}
    result: dict[str, dict[str, object]] = {}
    detector = cv2.ORB_create(nfeatures=int(orb_features), fastThreshold=int(fast_threshold))
    for row in valid_df.itertuples(index=False):
        focus_path = repo_root / getattr(row, YELLOW_FOCUS_PATH_COLUMN)
        mask_path = repo_root / getattr(row, YELLOW_FOCUS_MASK_PATH_COLUMN)
        focus_image = _load_rgb(focus_path)
        focus_mask = _load_binary_mask(mask_path)
        band_mask = build_yellow_band_mask(
            focus_mask,
            dilate_radius=DEFAULT_BAND_DILATE_RADIUS,
            erode_radius=DEFAULT_BAND_ERODE_RADIUS,
            min_band_pixels=DEFAULT_BAND_MIN_PIXELS,
        )
        rgb = np.asarray(focus_image, dtype=np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape[:2]
        if max(height, width) > int(max_side):
            scale = float(max_side) / float(max(height, width))
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            gray = cv2.resize(gray, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            focus_mask = _resize_mask(focus_mask, width=resized_width, height=resized_height)
            band_mask = _resize_mask(band_mask, width=resized_width, height=resized_height)
        else:
            resized_width = int(width)
            resized_height = int(height)
        if float(clahe_clip_limit) > 0:
            clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        keypoints, descriptors = detector.detectAndCompute(gray, (band_mask.astype(np.uint8) * 255))
        if keypoints:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.empty((0, 2), dtype=np.float32)
            descriptors = None
        resized = Image.fromarray(rgb, mode="RGB")
        yellow_overlay = _blend_mask_overlay(resized, focus_mask, color=(255, 220, 0), alpha=0.42)
        band_overlay = _blend_mask_overlay(resized, band_mask, color=(0, 220, 140), alpha=0.45)
        keypoint_overlay = _draw_points(band_overlay, points, radius=3)
        result[str(row.image_id)] = {
            "point_count": int(len(points)),
            "cv_keypoints": keypoints,
            "descriptors": descriptors,
            "points": points,
            "width": int(resized_width),
            "height": int(resized_height),
            "rgb_array": rgb.astype(np.uint8, copy=False),
            "focus_preview": resized,
            "yellow_overlay": yellow_overlay,
            "band_overlay": band_overlay,
            "focus_keypoints": keypoint_overlay,
            "focus_mask": focus_mask.astype(np.uint8),
            "band_mask": band_mask.astype(np.uint8),
        }
    return result


def _compute_inlier_match_preview(
    left_info: dict[str, object],
    right_info: dict[str, object],
    *,
    ratio_test: float = 0.75,
    ransac_threshold: float = 5.0,
    max_lines: int = 48,
) -> tuple[Image.Image, dict[str, int]]:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("yellow-band-orb inlier preview requires OpenCV.") from exc

    left_descriptors = left_info.get("descriptors")
    right_descriptors = right_info.get("descriptors")
    left_keypoints = left_info.get("cv_keypoints") or []
    right_keypoints = right_info.get("cv_keypoints") or []
    left_base = np.asarray(left_info["band_overlay"], dtype=np.uint8)
    right_base = np.asarray(right_info["band_overlay"], dtype=np.uint8)

    stats = {"good_matches": 0, "inliers": 0}
    if left_descriptors is None or right_descriptors is None or len(left_keypoints) < 2 or len(right_keypoints) < 2:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(left_descriptors, right_descriptors, k=2)
    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < float(ratio_test) * second.distance:
            good_matches.append(first)
    stats["good_matches"] = int(len(good_matches))
    if len(good_matches) < 4:
        drawn = cv2.drawMatches(
            cv2.cvtColor(left_base, cv2.COLOR_RGB2BGR),
            left_keypoints,
            cv2.cvtColor(right_base, cv2.COLOR_RGB2BGR),
            right_keypoints,
            good_matches[: int(max_lines)],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return Image.fromarray(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)), stats

    left_points = np.float32([left_info["points"][match.queryIdx] for match in good_matches])
    right_points = np.float32([right_info["points"][match.trainIdx] for match in good_matches])
    try:
        _homography, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, float(ransac_threshold))
    except cv2.error:
        mask = None
    if mask is None:
        inlier_matches: list = []
    else:
        keep = mask.ravel().astype(bool)
        inlier_matches = [match for match, flag in zip(good_matches, keep, strict=True) if flag]
    stats["inliers"] = int(len(inlier_matches))
    drawn = cv2.drawMatches(
        cv2.cvtColor(left_base, cv2.COLOR_RGB2BGR),
        left_keypoints,
        cv2.cvtColor(right_base, cv2.COLOR_RGB2BGR),
        right_keypoints,
        inlier_matches[: int(max_lines)],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return Image.fromarray(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)), stats


def _select_examples(decision_df: pd.DataFrame, *, decision: str, top_n: int) -> pd.DataFrame:
    subset = decision_df[decision_df["yellow_veto_decision_v1"].astype(str).eq(str(decision))].copy()
    if subset.empty:
        return subset
    if "same_identity" in subset.columns:
        if decision == "support":
            subset = subset[subset["same_identity"].astype(int).eq(1)].copy()
        else:
            subset = subset[subset["same_identity"].astype(int).eq(0)].copy()
    if decision == "support":
        sort_columns = ["yellow_roi_local_score", "yellow_roi_inliers", "xgb_same_identity_prob"]
        ascending = [False, False, False]
    elif decision == "hard_veto":
        sort_columns = ["xgb_same_identity_prob", "yellow_patch_gray_corr_v1", "yellow_roi_local_score"]
        ascending = [False, True, True]
    else:
        sort_columns = ["xgb_same_identity_prob", "yellow_patch_gray_corr_v1", "yellow_patch_gray_absdiff_v1"]
        ascending = [False, True, False]
    available = [column for column in sort_columns if column in subset.columns]
    if available:
        subset = subset.sort_values(
            available,
            ascending=ascending[: len(available)],
        )
    return subset.head(int(top_n)).reset_index(drop=True)


def _build_pair_board(
    *,
    row: pd.Series,
    focus_lookup: pd.DataFrame,
    roi_lookup: pd.DataFrame,
    feature_map: dict[str, dict[str, object]],
    repo_root: Path,
    output_path: Path,
) -> None:
    from animalclef_analysis.sam_orb_veto import ALIGNED_PATH_COLUMN
    from animalclef_analysis.salamander_yellow_orb_local import (
        YELLOW_FOCUS_MASK_PATH_COLUMN,
        YELLOW_FOCUS_PATH_COLUMN,
    )

    panel_w = 240
    panel_h = 160
    margin = 16
    gap = 12
    caption_h = 92
    header_h = 52
    canvas_w = margin * 2 + panel_w * 4 + gap * 3
    match_panel_h = 240
    match_caption_h = 44
    canvas_h = header_h + margin + (panel_h + caption_h) * 2 + gap + match_panel_h + match_caption_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(22)
    text_font = _font(15)
    small_font = _font(13)

    decision = str(row["yellow_veto_decision_v1"])
    pair_title = (
        f"{decision} | {row['image_id']} vs {row['neighbor_image_id']} | "
        f"same_id={int(row['same_identity']) if 'same_identity' in row else 'na'}"
    )
    draw.text((margin, 14), pair_title, fill=(20, 20, 20), font=title_font)
    metrics_text = (
        f"xgb={float(row.get('xgb_same_identity_prob', 0.0)):.3f}  "
        f"roi_score={float(row.get('yellow_roi_local_score', 0.0)):.3f}  "
        f"roi_inliers={int(row.get('yellow_roi_inliers', 0))}  "
        f"patch_corr={float(row.get('yellow_patch_gray_corr_v1', 0.0)):.3f}  "
        f"patch_iou={float(row.get('yellow_patch_mask_iou_v1', 0.0)):.3f}"
    )
    draw.text((margin, 40), metrics_text, fill=(70, 70, 70), font=small_font)

    left_match_info = feature_map.get(str(row["image_id"]))
    right_match_info = feature_map.get(str(row["neighbor_image_id"]))

    for row_idx, image_id_col in enumerate(["image_id", "neighbor_image_id"]):
        image_id = str(row[image_id_col])
        y0 = header_h + margin + row_idx * (panel_h + caption_h)
        focus_row = focus_lookup.loc[focus_lookup["image_id"].astype(str).eq(image_id)].iloc[0]
        roi_row = roi_lookup.loc[roi_lookup["image_id"].astype(str).eq(image_id)].iloc[0]

        aligned_image = _load_rgb(repo_root / str(roi_row[ALIGNED_PATH_COLUMN]))
        focus_image = _load_rgb(repo_root / str(focus_row[YELLOW_FOCUS_PATH_COLUMN]))
        keypoint_info = feature_map.get(image_id)
        if keypoint_info is not None:
            focus_preview = keypoint_info["focus_preview"]
            yellow_overlay = keypoint_info["yellow_overlay"]
            band_overlay = keypoint_info["band_overlay"]
            keypoint_preview = keypoint_info["focus_keypoints"]
            point_count = int(keypoint_info["point_count"])
            band_pixels = int(np.asarray(keypoint_info["band_mask"], dtype=np.uint8).sum())
        else:
            focus_preview = focus_image.copy()
            yellow_overlay = _load_mask_overlay(repo_root / str(focus_row[YELLOW_FOCUS_MASK_PATH_COLUMN]), focus_image.size)
            band_overlay = focus_preview.copy()
            keypoint_preview = focus_preview.copy()
            point_count = 0
            band_pixels = 0

        preview_items = [
            ("aligned", _resize_with_pad(aligned_image, width=panel_w, height=panel_h)),
            ("yellow focus", _resize_with_pad(yellow_overlay, width=panel_w, height=panel_h)),
            ("yellow band", _resize_with_pad(band_overlay, width=panel_w, height=panel_h)),
            ("band + orb", _resize_with_pad(keypoint_preview, width=panel_w, height=panel_h)),
        ]
        for col_idx, (label, preview) in enumerate(preview_items):
            x0 = margin + col_idx * (panel_w + gap)
            canvas.paste(preview, (x0, y0))
            draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=(190, 190, 190), width=1)
            draw.text((x0, y0 + panel_h + 6), f"{label} | image={image_id}", fill=(25, 25, 25), font=text_font)
            if label == "band + orb":
                draw.text(
                    (x0, y0 + panel_h + 28),
                    f"orb_points={point_count}  band_pixels={band_pixels}",
                    fill=(90, 90, 90),
                    font=small_font,
                )
                draw.text(
                    (x0, y0 + panel_h + 46),
                    f"focus={int(focus_row['yellow_focus_width_v1'])}x{int(focus_row['yellow_focus_height_v1'])}",
                    fill=(90, 90, 90),
                    font=small_font,
                )

    match_y0 = header_h + margin + 2 * (panel_h + caption_h) + gap
    match_x0 = margin
    match_w = canvas_w - 2 * margin
    if left_match_info is not None and right_match_info is not None:
        match_preview, match_stats = _compute_inlier_match_preview(left_match_info, right_match_info)
        match_preview = _resize_with_pad(match_preview, width=match_w, height=match_panel_h)
        canvas.paste(match_preview, (match_x0, match_y0))
        draw.rectangle((match_x0, match_y0, match_x0 + match_w, match_y0 + match_panel_h), outline=(190, 190, 190), width=1)
        draw.text(
            (match_x0, match_y0 + match_panel_h + 8),
            f"band + inlier matches | good_matches={int(match_stats['good_matches'])}  inliers={int(match_stats['inliers'])}",
            fill=(25, 25, 25),
            font=text_font,
        )
    else:
        draw.rectangle((match_x0, match_y0, match_x0 + match_w, match_y0 + match_panel_h), outline=(190, 190, 190), width=1)
        draw.text((match_x0 + 12, match_y0 + 12), "band + inlier matches unavailable", fill=(120, 120, 120), font=text_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    parser = argparse.ArgumentParser(description="Build qualitative yellow ROI + ORB keypoint review boards for Salamander.")
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n-per-type", type=int, default=6)
    parser.add_argument("--orb-features", type=int, default=1024)
    parser.add_argument("--orb-max-side", type=int, default=512)
    parser.add_argument("--fast-threshold", type=int, default=7)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    args = parser.parse_args()

    analysis_dir = (repo_root / args.analysis_dir).resolve() if not args.analysis_dir.is_absolute() else args.analysis_dir.resolve()
    output_dir = (
        analysis_dir / DEFAULT_OUTPUT_DIRNAME
        if args.output_dir is None
        else ((repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve())
    )
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    for path in [output_dir, images_dir, tables_dir]:
        path.mkdir(parents=True, exist_ok=True)

    decision_df = pd.read_csv(analysis_dir / "tables" / "val_yellow_orb_decisions_v1.csv")
    focus_df = pd.read_csv(analysis_dir / "tables" / "yellow_focus_manifest_v1.csv")
    roi_df = pd.read_csv(analysis_dir / "tables" / "image_roi_manifest_v1.csv")
    decision_df["image_id"] = decision_df["image_id"].astype(str)
    decision_df["neighbor_image_id"] = decision_df["neighbor_image_id"].astype(str)
    focus_df["image_id"] = focus_df["image_id"].astype(str)
    roi_df["image_id"] = roi_df["image_id"].astype(str)

    focus_feature_map = _build_focus_feature_map(
        focus_df=focus_df[focus_df["dataset"].astype(str).eq(SALAMANDER_DATASET)].copy(),
        repo_root=repo_root,
        max_side=int(args.orb_max_side),
        fast_threshold=int(args.fast_threshold),
        clahe_clip_limit=float(args.clahe_clip_limit),
        orb_features=int(args.orb_features),
    )

    selected_frames = []
    markdown_lines = [
        "# Salamander Yellow ROI + ORB 定性图",
        "",
        f"- 来源：`{analysis_dir}`",
        f"- 数据：`val_yellow_orb_decisions_v1.csv`",
        f"- 每类展示：`{int(args.top_n_per_type)}` 对",
        "",
        "## 阅读说明",
        "",
        "- 每个 pair board 分四列：`aligned`、`yellow focus`、`yellow band`、`band + orb`。",
        "- `yellow focus` 是黄纹 mask 的局部 ROI；`yellow band` 是在 yellow mask 上做膨胀/侵蚀后得到的边界窄带。",
        "- `band + orb` 上的红点只在 `yellow band` 里提取；这比旧版更接近“只看黄黑交界”的直觉。",
        "- 每个 board 底部还会画一块 `band + inlier matches`，只显示真正通过几何一致性筛选的匹配线。",
        "",
    ]

    for decision in ["support", "hard_veto", "soft_veto"]:
        subset = _select_examples(decision_df, decision=decision, top_n=int(args.top_n_per_type))
        if subset.empty:
            continue
        decision_dir = images_dir / decision
        decision_dir.mkdir(parents=True, exist_ok=True)
        saved_rows: list[dict[str, object]] = []
        markdown_lines.extend([f"## {decision}", ""])
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            board_path = decision_dir / f"{rank:02d}_{row['image_id']}_{row['neighbor_image_id']}.jpg"
            _build_pair_board(
                row=row,
                focus_lookup=focus_df,
                roi_lookup=roi_df,
                feature_map=focus_feature_map,
                repo_root=repo_root,
                output_path=board_path,
            )
            saved_rows.append(
                {
                    "decision": decision,
                    "rank": rank,
                    "image_id": str(row["image_id"]),
                    "neighbor_image_id": str(row["neighbor_image_id"]),
                    "same_identity": int(row["same_identity"]) if "same_identity" in row else -1,
                    "xgb_same_identity_prob": float(row.get("xgb_same_identity_prob", 0.0)),
                    "yellow_roi_local_score": float(row.get("yellow_roi_local_score", 0.0)),
                    "yellow_roi_inliers": int(row.get("yellow_roi_inliers", 0)),
                    "yellow_patch_gray_corr_v1": float(row.get("yellow_patch_gray_corr_v1", 0.0)),
                    "yellow_patch_mask_iou_v1": float(row.get("yellow_patch_mask_iou_v1", 0.0)),
                    "left_band_orb_points": int(focus_feature_map[str(row["image_id"])]["point_count"]) if str(row["image_id"]) in focus_feature_map else 0,
                    "right_band_orb_points": int(focus_feature_map[str(row["neighbor_image_id"])]["point_count"]) if str(row["neighbor_image_id"]) in focus_feature_map else 0,
                    "board_path": str(board_path),
                }
            )
            markdown_lines.extend(
                [
                    f"### {decision} #{rank}: `{row['image_id']} vs {row['neighbor_image_id']}`",
                    "",
                    f"- `same_identity={int(row['same_identity']) if 'same_identity' in row else 'na'}`，`xgb={float(row.get('xgb_same_identity_prob', 0.0)):.3f}`，`yellow_roi_local_score={float(row.get('yellow_roi_local_score', 0.0)):.3f}`，`yellow_roi_inliers={int(row.get('yellow_roi_inliers', 0))}`，`patch_corr={float(row.get('yellow_patch_gray_corr_v1', 0.0)):.3f}`，`patch_iou={float(row.get('yellow_patch_mask_iou_v1', 0.0)):.3f}`。",
                    "",
                    f"![{decision}-{rank}]({_path_ref(output_dir, board_path)})",
                    "",
                ]
            )
        pd.DataFrame(saved_rows).to_csv(tables_dir / f"{decision}_examples_v1.csv", index=False)

    (output_dir / "summary.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"[yellow_orb_review_pack] summary: {output_dir / 'summary.md'}")
    print(f"[yellow_orb_review_pack] images: {images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
