#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_MANIFEST_PATH = Path("artifacts/manifests/v1/tables/metadata_enriched_v1.csv")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_sam_mask_orb_vs_aliked_review_v1")

SALAMANDER_DATASET = "SalamanderID2025"
SAM_MASKED_PATH_COLUMN = "sam_masked_rgb_v1_resolved_path_v1"

ORB_NFEATURES = 1024
ORB_MAX_SIDE = 768
ORB_FAST_THRESHOLD = 7
ORB_CLAHE_CLIP_LIMIT = 2.0
ORB_RATIO_TEST = 0.8
RANSAC_THRESHOLD = 5.0
MIN_INLIERS = 8


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _resize_with_pad(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (int(width), int(height)))
    canvas = Image.new("RGB", (int(width), int(height)), (248, 248, 248))
    offset = ((int(width) - fitted.width) // 2, (int(height) - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _resize_like(image: Image.Image, *, width: int, height: int) -> Image.Image:
    if image.size == (int(width), int(height)):
        return image.copy()
    return image.resize((int(width), int(height)), Image.Resampling.BILINEAR)


def _draw_points(
    image: Image.Image,
    points: np.ndarray,
    *,
    radius: int = 3,
    color: tuple[int, int, int] = (255, 40, 40),
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x, y in np.asarray(points, dtype=np.float32):
        draw.ellipse(
            (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
            outline=color,
            width=2,
        )
    return output


def _blend_mask_overlay(
    image: Image.Image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    alpha: float = 0.45,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_bool = np.asarray(mask, dtype=np.uint8) > 0
    base = rgb.astype(np.float32)
    tint = np.asarray(color, dtype=np.float32)
    base[mask_bool] = (1.0 - float(alpha)) * base[mask_bool] + float(alpha) * tint
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def _wrap_text_lines(text: str, *, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text).split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        try:
            width = font.getlength(candidate)
        except AttributeError:
            width = font.getbbox(candidate)[2]
        if width <= int(max_width):
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _feature_item(feature_dataset, index: int) -> dict[str, object]:
    item = feature_dataset[index]
    if isinstance(item, tuple):
        item = item[0]
    if isinstance(item, list):
        item = item[0]
    if not isinstance(item, dict):
        raise TypeError(f"Unsupported feature item type: {type(item)!r}")
    return item


def _make_pair_key(left_image_id: str, right_image_id: str) -> str:
    left = str(left_image_id)
    right = str(right_image_id)
    if left <= right:
        return f"{left}__{right}"
    return f"{right}__{left}"


def _build_image_dataset(metadata_df: pd.DataFrame, repo_root: Path):
    import torchvision.transforms as T
    from wildlife_tools.data.dataset import ImageDataset

    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    return ImageDataset(
        metadata=metadata_df[["path", "identity", "image_id"]].copy(),
        root=str(repo_root),
        transform=transform,
        col_path="path",
        col_label="identity",
        load_label=True,
    )


def _resolve_review_metadata(route_df: pd.DataFrame, manifest_path: Path, repo_root: Path) -> pd.DataFrame:
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path

    manifest_df = pd.read_csv(
        manifest_path,
        usecols=["image_id", "dataset", SAM_MASKED_PATH_COLUMN, "sam_masked_rgb_v1_applied"],
    )
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)

    resolved = route_df.copy()
    resolved["image_id"] = resolved["image_id"].astype(str)
    resolved["identity"] = resolved["identity"].fillna("").astype(str)
    resolved["dataset"] = resolved["dataset"].astype(str)
    resolved["original_path"] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in resolved.iterrows()]

    resolved = resolved.merge(
        manifest_df[manifest_df["dataset"] == SALAMANDER_DATASET].drop(columns=["dataset"]),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if resolved[SAM_MASKED_PATH_COLUMN].isna().any():
        missing_ids = resolved.loc[resolved[SAM_MASKED_PATH_COLUMN].isna(), "image_id"].tolist()
        raise ValueError(f"Missing SAM masked path for image ids: {missing_ids[:8]}")

    resolved["masked_path"] = resolved[SAM_MASKED_PATH_COLUMN].astype(str)
    resolved["sam_mask_applied"] = resolved["sam_masked_rgb_v1_applied"].fillna(False).astype(bool)
    resolved["path"] = resolved["masked_path"]
    resolved["recommended_model_input_path_v1"] = resolved["masked_path"]
    resolved["preferred_path_v1"] = resolved["masked_path"]

    missing_files = [path for path in resolved["masked_path"].tolist() if not (repo_root / path).exists()]
    if missing_files:
        raise FileNotFoundError(f"SAM masked files are missing, first examples: {missing_files[:4]}")
    return resolved.reset_index(drop=True)


def _build_yellow_focus_review_metadata(
    *,
    metadata_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from animalclef_analysis.salamander_yellow_orb_local import (
        YELLOW_FOCUS_MASK_PATH_COLUMN,
        YELLOW_FOCUS_PATH_COLUMN,
        build_yellow_focus_manifest,
    )
    from animalclef_analysis.sam_orb_veto import ALIGNED_PATH_COLUMN, MASKED_PATH_COLUMN

    roi_manifest_df = metadata_df[["image_id", "dataset", "split", "identity", "original_path", "masked_path"]].copy()
    roi_manifest_df["path"] = roi_manifest_df["original_path"].astype(str)
    roi_manifest_df[MASKED_PATH_COLUMN] = roi_manifest_df["masked_path"].astype(str)
    roi_manifest_df[ALIGNED_PATH_COLUMN] = ""

    focus_output_dir = output_dir / "yellow_focus_cache"
    focus_df = build_yellow_focus_manifest(
        roi_manifest_df=roi_manifest_df,
        repo_root=repo_root,
        output_dir=focus_output_dir,
    )
    merged = metadata_df.merge(
        focus_df[["image_id", "dataset", "yellow_focus_available_v1", YELLOW_FOCUS_PATH_COLUMN, YELLOW_FOCUS_MASK_PATH_COLUMN]],
        on=["image_id", "dataset"],
        how="left",
        validate="one_to_one",
    )
    merged["yellow_focus_available_v1"] = merged["yellow_focus_available_v1"].fillna(False).astype(bool)
    merged["focus_path"] = np.where(
        merged["yellow_focus_available_v1"].astype(bool),
        merged[YELLOW_FOCUS_PATH_COLUMN].fillna("").astype(str),
        merged["masked_path"].astype(str),
    )
    merged["focus_mask_path"] = merged[YELLOW_FOCUS_MASK_PATH_COLUMN].fillna("").astype(str)
    merged["path"] = merged["focus_path"]
    merged["recommended_model_input_path_v1"] = merged["focus_path"]
    merged["preferred_path_v1"] = merged["focus_path"]
    return merged.reset_index(drop=True), focus_df.reset_index(drop=True)


def _build_yellow_band_view_metadata(
    *,
    metadata_df: pd.DataFrame,
    repo_root: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    from animalclef_analysis.salamander_yellow_orb_local import (
        DEFAULT_BAND_DILATE_RADIUS,
        DEFAULT_BAND_ERODE_RADIUS,
        DEFAULT_BAND_MIN_PIXELS,
        build_yellow_band_mask,
    )

    band_views_dir = output_dir / "yellow_band_cache" / "views" / "yellow_band_rgb_v1"
    band_views_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    preview_by_image: dict[str, dict[str, object]] = {}
    for row in metadata_df.itertuples(index=False):
        image_id = str(row.image_id)
        focus_path = str(getattr(row, "focus_path", "") or "")
        focus_mask_path = str(getattr(row, "focus_mask_path", "") or "")
        band_path = ""
        band_available = False
        band_pixels = 0
        if focus_path and focus_mask_path:
            focus_image = _load_rgb(repo_root / focus_path)
            focus_mask = _load_binary_mask(repo_root / focus_mask_path)
            band_mask = build_yellow_band_mask(
                focus_mask,
                dilate_radius=DEFAULT_BAND_DILATE_RADIUS,
                erode_radius=DEFAULT_BAND_ERODE_RADIUS,
                min_band_pixels=DEFAULT_BAND_MIN_PIXELS,
            )
            rgb = np.asarray(focus_image, dtype=np.uint8)
            band_rgb = np.zeros_like(rgb, dtype=np.uint8)
            band_rgb[band_mask > 0] = rgb[band_mask > 0]
            relative_image_path = Path(str(row.original_path)).relative_to("images")
            export_rel = Path("views") / "yellow_band_rgb_v1" / relative_image_path
            export_abs = output_dir / "yellow_band_cache" / export_rel
            export_abs.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(band_rgb, mode="RGB").save(export_abs, quality=95)
            band_path = str((output_dir.relative_to(repo_root) / "yellow_band_cache" / export_rel).as_posix())
            band_available = True
            band_pixels = int(np.asarray(band_mask, dtype=np.uint8).sum())
            preview_by_image[image_id] = {
                "band_mask": np.asarray(band_mask, dtype=np.uint8),
                "band_overlay": _blend_mask_overlay(focus_image, band_mask, color=(0, 220, 140), alpha=0.45),
            }
        rows.append(
            {
                "image_id": image_id,
                "dataset": str(row.dataset),
                "yellow_band_available_v1": bool(band_available),
                "yellow_band_rgb_path_v1": str(band_path),
                "yellow_band_pixels_v1": int(band_pixels),
            }
        )
    band_df = pd.DataFrame(rows)
    merged = metadata_df.merge(
        band_df,
        on=["image_id", "dataset"],
        how="left",
        validate="one_to_one",
    )
    merged["yellow_band_available_v1"] = merged["yellow_band_available_v1"].fillna(False).astype(bool)
    return merged.reset_index(drop=True), preview_by_image


def _build_yellow_band_orb_features(
    *,
    metadata_df: pd.DataFrame,
    repo_root: Path,
    nfeatures: int,
    max_side: int,
    fast_threshold: int,
    clahe_clip_limit: float,
) -> tuple[list[object], dict[str, dict[str, object]]]:
    import cv2

    from animalclef_analysis.orb_rerank_baseline import OrbFeature
    from animalclef_analysis.salamander_yellow_orb_local import (
        DEFAULT_BAND_DILATE_RADIUS,
        DEFAULT_BAND_ERODE_RADIUS,
        DEFAULT_BAND_MIN_PIXELS,
        build_yellow_band_mask,
    )

    detector = cv2.ORB_create(nfeatures=int(nfeatures), fastThreshold=int(fast_threshold))
    features: list[object] = []
    preview_by_image: dict[str, dict[str, object]] = {}
    for row in metadata_df.itertuples(index=False):
        image_id = str(row.image_id)
        focus_path = str(getattr(row, "focus_path", "") or getattr(row, "masked_path", ""))
        focus_mask_path = str(getattr(row, "focus_mask_path", "") or "")
        if not focus_path:
            features.append(
                OrbFeature(
                    image_id=image_id,
                    matcher_name="orb",
                    point_count=0,
                    points=np.empty((0, 2), dtype=np.float32),
                    descriptors=None,
                    width=0,
                    height=0,
                )
            )
            continue

        focus_image = _load_rgb(repo_root / focus_path)
        if focus_mask_path:
            focus_mask = _load_binary_mask(repo_root / focus_mask_path)
        else:
            focus_mask = (np.any(np.asarray(focus_image, dtype=np.uint8) > 0, axis=2)).astype(np.uint8)
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
            focus_mask = np.asarray(Image.fromarray((focus_mask > 0).astype(np.uint8) * 255, mode="L").resize((resized_width, resized_height), Image.NEAREST), dtype=np.uint8) > 0
            band_mask = np.asarray(Image.fromarray((band_mask > 0).astype(np.uint8) * 255, mode="L").resize((resized_width, resized_height), Image.NEAREST), dtype=np.uint8) > 0
            width = int(resized_width)
            height = int(resized_height)
        if float(clahe_clip_limit) > 0:
            clahe = cv2.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        keypoints, descriptors = detector.detectAndCompute(gray, (np.asarray(band_mask, dtype=np.uint8) * 255))
        if keypoints:
            points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
        else:
            points = np.empty((0, 2), dtype=np.float32)
            descriptors = None
        preview_rgb = Image.fromarray(rgb, mode="RGB")
        band_overlay = _blend_mask_overlay(preview_rgb, np.asarray(band_mask, dtype=np.uint8), color=(0, 220, 140), alpha=0.45)
        focus_overlay = _blend_mask_overlay(preview_rgb, np.asarray(focus_mask, dtype=np.uint8), color=(255, 220, 0), alpha=0.42)
        preview_by_image[image_id] = {
            "focus_overlay": focus_overlay,
            "band_overlay": band_overlay,
            "band_points_overlay": _draw_points(band_overlay, points, radius=3, color=(255, 60, 60)),
        }
        features.append(
            OrbFeature(
                image_id=image_id,
                matcher_name="orb",
                point_count=int(len(points)),
                points=points,
                descriptors=descriptors,
                width=int(width),
                height=int(height),
            )
        )
    return features, preview_by_image


def _compute_lightglue_pair_rows(
    *,
    metadata_df: pd.DataFrame,
    feature_dataset,
    pair_index: list[tuple[int, int, float]],
    device: str,
    init_threshold: float,
    batch_size: int,
    ransac_threshold: float,
    min_inliers: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, np.ndarray]]:
    import cv2
    from wildlife_tools.similarity.pairwise.collectors import CollectAll
    from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue

    matcher = MatchLightGlue(
        features="aliked",
        init_threshold=float(init_threshold),
        device=str(device),
        batch_size=int(batch_size),
        num_workers=0,
        collector=CollectAll(),
        tqdm_silent=False,
    )
    pair_array = np.asarray([[left, right] for left, right, _ in pair_index], dtype=np.int32)
    raw_results = matcher(feature_dataset, feature_dataset, pairs=pair_array)
    global_score_map = {(int(left), int(right)): float(score) for left, right, score in pair_index}

    all_points_by_image: dict[str, np.ndarray] = {}
    for index, row in enumerate(metadata_df.itertuples(index=False)):
        item = _feature_item(feature_dataset, index)
        all_points_by_image[str(row.image_id)] = np.asarray(item["keypoints"], dtype=np.float32)

    rows: list[dict[str, object]] = []
    preview_by_pair: dict[str, dict[str, object]] = {}
    for item in raw_results:
        left_index = int(item["idx0"])
        right_index = int(item["idx1"])
        left_row = metadata_df.iloc[left_index]
        right_row = metadata_df.iloc[right_index]
        match_scores = np.asarray(item["scores"], dtype=np.float32)
        kpts0 = np.asarray(item["kpts0"], dtype=np.float32)
        kpts1 = np.asarray(item["kpts1"], dtype=np.float32)
        good_matches = int(len(match_scores))
        mean_match_score = float(match_scores.mean()) if len(match_scores) else 0.0

        if len(kpts0) < 4 or len(kpts1) < 4:
            inlier_mask = np.zeros((len(kpts0),), dtype=bool)
        else:
            try:
                _homography, mask = cv2.findHomography(kpts0, kpts1, cv2.RANSAC, float(ransac_threshold))
            except cv2.error:
                mask = None
            inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros((len(kpts0),), dtype=bool)
        inliers = int(inlier_mask.sum())

        left_all_points = all_points_by_image[str(left_row["image_id"])]
        right_all_points = all_points_by_image[str(right_row["image_id"])]
        if inliers < int(min_inliers):
            local_raw_score = 0.0
        else:
            local_raw_score = float(
                (inliers * max(mean_match_score, 1e-6))
                / max(1, min(len(left_all_points), len(right_all_points)))
            )

        pair_key = _make_pair_key(str(left_row["image_id"]), str(right_row["image_id"]))
        preview_by_pair[pair_key] = {
            "left_points": kpts0,
            "right_points": kpts1,
            "match_scores": match_scores,
            "inlier_mask": inlier_mask,
        }
        rows.append(
            {
                "pair_key": pair_key,
                "dataset": str(left_row["dataset"]),
                "left_index": left_index,
                "right_index": right_index,
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "identity": str(left_row["identity"]),
                "neighbor_identity": str(right_row["identity"]),
                "same_identity": bool(str(left_row["identity"]) == str(right_row["identity"])),
                "global_score": round(global_score_map[(left_index, right_index)], 6),
                "left_keypoints": int(len(left_all_points)),
                "right_keypoints": int(len(right_all_points)),
                "good_matches": good_matches,
                "inliers": inliers,
                "mean_match_score": round(mean_match_score, 6),
                "local_raw_score": round(local_raw_score, 6),
            }
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        pair_df["local_score"] = pd.Series(dtype=float)
        return pair_df, preview_by_pair, all_points_by_image

    nonzero = pair_df.loc[pair_df["local_raw_score"] > 0.0, "local_raw_score"].to_numpy(dtype=float)
    if len(nonzero) == 0:
        pair_df["local_score"] = 0.0
        return pair_df, preview_by_pair, all_points_by_image

    upper = max(float(np.quantile(nonzero, 0.95)), 1e-6)
    pair_df["local_score"] = np.round(np.clip(pair_df["local_raw_score"].to_numpy(dtype=float) / upper, 0.0, 1.0), 6)
    return pair_df, preview_by_pair, all_points_by_image


def _draw_orb_match_preview(
    *,
    left_feature,
    right_feature,
    left_rgb: Image.Image,
    right_rgb: Image.Image,
    ratio_test: float,
    ransac_threshold: float,
    max_lines: int = 64,
) -> tuple[Image.Image, dict[str, int]]:
    import cv2

    left_base = np.asarray(left_rgb.convert("RGB"), dtype=np.uint8)
    right_base = np.asarray(right_rgb.convert("RGB"), dtype=np.uint8)
    left_descriptors = left_feature.descriptors
    right_descriptors = right_feature.descriptors
    stats = {"good_matches": 0, "inliers": 0}

    if left_descriptors is None or right_descriptors is None:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats
    if len(left_descriptors) < 2 or len(right_descriptors) < 2:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats

    left_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in np.asarray(left_feature.points, dtype=np.float32)]
    right_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in np.asarray(right_feature.points, dtype=np.float32)]
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

    left_points = np.float32([left_feature.points[match.queryIdx] for match in good_matches])
    right_points = np.float32([right_feature.points[match.trainIdx] for match in good_matches])
    try:
        _homography, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, float(ransac_threshold))
    except cv2.error:
        mask = None
    keep = mask.ravel().astype(bool) if mask is not None else np.zeros((len(good_matches),), dtype=bool)
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


def _draw_aliked_match_preview(
    *,
    left_rgb: Image.Image,
    right_rgb: Image.Image,
    preview_info: dict[str, object] | None,
    max_lines: int = 64,
) -> tuple[Image.Image, dict[str, int]]:
    import cv2

    left_base = np.asarray(left_rgb.convert("RGB"), dtype=np.uint8)
    right_base = np.asarray(right_rgb.convert("RGB"), dtype=np.uint8)
    stats = {"good_matches": 0, "inliers": 0}
    if not preview_info:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats

    left_points = np.asarray(preview_info["left_points"], dtype=np.float32)
    right_points = np.asarray(preview_info["right_points"], dtype=np.float32)
    inlier_mask = np.asarray(preview_info["inlier_mask"], dtype=bool)
    stats["good_matches"] = int(len(left_points))
    stats["inliers"] = int(inlier_mask.sum())
    if len(left_points) == 0 or len(right_points) == 0:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats

    keep = inlier_mask if inlier_mask.any() else np.ones((len(left_points),), dtype=bool)
    selected_indices = np.flatnonzero(keep)[: int(max_lines)]
    selected_left = left_points[selected_indices]
    selected_right = right_points[selected_indices]
    if len(selected_left) == 0:
        blank = Image.new("RGB", (left_base.shape[1] + right_base.shape[1] + 24, max(left_base.shape[0], right_base.shape[0])), (250, 250, 250))
        return blank, stats

    left_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in selected_left]
    right_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in selected_right]
    matches = [cv2.DMatch(_queryIdx=index, _trainIdx=index, _imgIdx=0, _distance=0.0) for index in range(len(selected_left))]
    drawn = cv2.drawMatches(
        cv2.cvtColor(left_base, cv2.COLOR_RGB2BGR),
        left_keypoints,
        cv2.cvtColor(right_base, cv2.COLOR_RGB2BGR),
        right_keypoints,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return Image.fromarray(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)), stats


def _build_pair_board(
    *,
    row: pd.Series,
    category_title: str,
    paths_by_image: dict[str, dict[str, str]],
    orb_feature_by_image: dict[str, object],
    aliked_points_by_image: dict[str, np.ndarray],
    aliked_pair_preview_by_key: dict[str, dict[str, object]],
    repo_root: Path,
    output_path: Path,
    use_yellow_prior: bool,
    use_yellow_band_orb: bool,
    use_yellow_band_aliked: bool,
    orb_preview_by_image: dict[str, dict[str, object]] | None = None,
    aliked_preview_by_image: dict[str, dict[str, object]] | None = None,
) -> None:
    left_id = str(row["image_id"])
    right_id = str(row["neighbor_image_id"])
    pair_key = str(row["pair_key"])
    left_paths = paths_by_image[left_id]
    right_paths = paths_by_image[right_id]

    original_left = _load_rgb(repo_root / left_paths["original_path"])
    original_right = _load_rgb(repo_root / right_paths["original_path"])
    masked_left_full = _load_rgb(repo_root / left_paths["masked_path"])
    masked_right_full = _load_rgb(repo_root / right_paths["masked_path"])
    focus_left_full = _load_rgb(repo_root / left_paths["focus_path"]) if left_paths.get("focus_path") else masked_left_full
    focus_right_full = _load_rgb(repo_root / right_paths["focus_path"]) if right_paths.get("focus_path") else masked_right_full
    band_left_full = _load_rgb(repo_root / left_paths["band_path"]) if left_paths.get("band_path") else focus_left_full
    band_right_full = _load_rgb(repo_root / right_paths["band_path"]) if right_paths.get("band_path") else focus_right_full
    orb_match_left_full = focus_left_full if bool(use_yellow_prior) else masked_left_full
    orb_match_right_full = focus_right_full if bool(use_yellow_prior) else masked_right_full
    aliked_match_left_full = band_left_full if bool(use_yellow_band_aliked) else (focus_left_full if bool(use_yellow_prior) else masked_left_full)
    aliked_match_right_full = band_right_full if bool(use_yellow_band_aliked) else (focus_right_full if bool(use_yellow_prior) else masked_right_full)

    left_orb_feature = orb_feature_by_image[left_id]
    right_orb_feature = orb_feature_by_image[right_id]
    masked_left_orb = _resize_like(orb_match_left_full, width=left_orb_feature.width, height=left_orb_feature.height)
    masked_right_orb = _resize_like(orb_match_right_full, width=right_orb_feature.width, height=right_orb_feature.height)
    orb_left_preview = (orb_preview_by_image or {}).get(left_id, {})
    orb_right_preview = (orb_preview_by_image or {}).get(right_id, {})
    orb_left_points = orb_left_preview.get("band_points_overlay") if bool(use_yellow_band_orb) else None
    orb_right_points = orb_right_preview.get("band_points_overlay") if bool(use_yellow_band_orb) else None
    if orb_left_points is None:
        orb_left_points = _draw_points(masked_left_orb, left_orb_feature.points, radius=3, color=(255, 60, 60))
    if orb_right_points is None:
        orb_right_points = _draw_points(masked_right_orb, right_orb_feature.points, radius=3, color=(255, 60, 60))
    orb_match_preview, orb_match_stats = _draw_orb_match_preview(
        left_feature=left_orb_feature,
        right_feature=right_orb_feature,
        left_rgb=orb_left_preview.get("band_overlay", masked_left_orb) if bool(use_yellow_band_orb) else masked_left_orb,
        right_rgb=orb_right_preview.get("band_overlay", masked_right_orb) if bool(use_yellow_band_orb) else masked_right_orb,
        ratio_test=ORB_RATIO_TEST,
        ransac_threshold=RANSAC_THRESHOLD,
    )

    aliked_preview_size = 512
    aliked_left_preview = (aliked_preview_by_image or {}).get(left_id, {})
    aliked_right_preview = (aliked_preview_by_image or {}).get(right_id, {})
    aliked_left_base = aliked_left_preview.get("band_overlay", aliked_match_left_full) if bool(use_yellow_band_aliked) else aliked_match_left_full
    aliked_right_base = aliked_right_preview.get("band_overlay", aliked_match_right_full) if bool(use_yellow_band_aliked) else aliked_match_right_full
    masked_left_aliked = _resize_like(aliked_left_base, width=aliked_preview_size, height=aliked_preview_size)
    masked_right_aliked = _resize_like(aliked_right_base, width=aliked_preview_size, height=aliked_preview_size)
    aliked_left_points = _draw_points(masked_left_aliked, aliked_points_by_image.get(left_id, np.empty((0, 2), dtype=np.float32)), radius=2, color=(30, 120, 255))
    aliked_right_points = _draw_points(masked_right_aliked, aliked_points_by_image.get(right_id, np.empty((0, 2), dtype=np.float32)), radius=2, color=(30, 120, 255))
    aliked_match_preview, aliked_match_stats = _draw_aliked_match_preview(
        left_rgb=masked_left_aliked,
        right_rgb=masked_right_aliked,
        preview_info=aliked_pair_preview_by_key.get(pair_key),
    )

    margin = 18
    gap = 14
    panel_w = 360
    panel_h = 220
    match_h = 280
    canvas_w = margin * 2 + panel_w * 2 + gap
    title_font = _font(22)
    body_font = _font(16)
    info_lines = [
        f"left={left_id} ({row.get('identity', '')})    right={right_id} ({row.get('neighbor_identity', '')})",
        f"same_identity={row.get('same_identity', 'na')}    global_score={float(row.get('global_score', 0.0)):.3f}",
        (
            f"ORB local={float(row.get('orb_local_score', 0.0)):.3f} good={int(row.get('orb_good_matches', orb_match_stats['good_matches']))} "
            f"inliers={int(row.get('orb_inliers', orb_match_stats['inliers']))}    "
            f"ALIKED+LG local={float(row.get('aliked_local_score', 0.0)):.3f} good={int(row.get('aliked_good_matches', aliked_match_stats['good_matches']))} "
            f"inliers={int(row.get('aliked_inliers', aliked_match_stats['inliers']))}"
        ),
        f"delta(orb-aliked)={float(row.get('orb_minus_aliked', 0.0)):.3f}",
    ]
    if bool(use_yellow_prior):
        info_lines.append(
            f"yellow_prior=True    left_focus={bool(left_paths.get('focus_path'))}    right_focus={bool(right_paths.get('focus_path'))}"
        )
    if bool(use_yellow_band_orb):
        info_lines.append("orb_mode=yellow_band    aliked_mode=yellow_focus")
    if bool(use_yellow_band_aliked):
        info_lines.append("aliked_mode=yellow_band")
    header_lines: list[str] = []
    for line in info_lines:
        header_lines.extend(_wrap_text_lines(line, font=body_font, max_width=canvas_w - margin * 2))

    title_h = 38
    line_h = 26
    header_h = title_h + 14 + len(header_lines) * line_h + 24
    rows = [
        ("left original", "right original", original_left, original_right, panel_h),
        ("left sam masked", "right sam masked", masked_left_full, masked_right_full, panel_h),
    ]
    if bool(use_yellow_prior):
        rows.append(("left yellow focus", "right yellow focus", focus_left_full, focus_right_full, panel_h))
    if bool(use_yellow_band_orb):
        rows.append(
            (
                "left yellow band",
                "right yellow band",
                orb_left_preview.get("band_overlay", focus_left_full),
                orb_right_preview.get("band_overlay", focus_right_full),
                panel_h,
            )
        )
    elif bool(use_yellow_band_aliked):
        rows.append(
            (
                "left yellow band",
                "right yellow band",
                aliked_left_preview.get("band_overlay", band_left_full),
                aliked_right_preview.get("band_overlay", band_right_full),
                panel_h,
            )
        )
    rows.extend(
        [
            (
                (
                    "left yellow-band input + ORB points"
                    if bool(use_yellow_band_orb)
                    else ("left yellow-focus input + ORB points" if bool(use_yellow_prior) else "left sam masked + ORB points")
                ),
                (
                    "right yellow-band input + ORB points"
                    if bool(use_yellow_band_orb)
                    else ("right yellow-focus input + ORB points" if bool(use_yellow_prior) else "right sam masked + ORB points")
                ),
                orb_left_points,
                orb_right_points,
                panel_h,
            ),
        (
            f"ORB inlier matches | good={int(row.get('orb_good_matches', orb_match_stats['good_matches']))} inliers={int(row.get('orb_inliers', orb_match_stats['inliers']))}",
            "",
            orb_match_preview,
            None,
            match_h,
        ),
            (
                (
                    "left yellow-band input + ALIKED points"
                    if bool(use_yellow_band_aliked)
                    else ("left yellow-focus input + ALIKED points" if bool(use_yellow_prior) else "left sam masked + ALIKED points")
                ),
                (
                    "right yellow-band input + ALIKED points"
                    if bool(use_yellow_band_aliked)
                    else ("right yellow-focus input + ALIKED points" if bool(use_yellow_prior) else "right sam masked + ALIKED points")
                ),
                aliked_left_points,
                aliked_right_points,
                panel_h,
            ),
        (
            f"ALIKED + LightGlue inlier matches | good={int(row.get('aliked_good_matches', aliked_match_stats['good_matches']))} inliers={int(row.get('aliked_inliers', aliked_match_stats['inliers']))}",
            "",
            aliked_match_preview,
            None,
            match_h,
        ),
        ]
    )
    canvas_h = margin * 2 + header_h + sum(int(row_height) for *_unused, row_height in rows) + max(0, len(rows) - 1) * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), category_title, fill=(20, 20, 20), font=title_font)
    for index, line in enumerate(header_lines):
        draw.text((margin, margin + title_h + index * line_h), line, fill=(42, 42, 42), font=body_font)

    row_y = margin + header_h

    def draw_label(x: int, y: int, text: str) -> None:
        bbox = draw.textbbox((x, y), text, font=body_font)
        rect = (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4)
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=body_font)

    for left_label, right_label, left_image, right_image, row_height in rows:
        if right_image is None:
            board = _resize_with_pad(left_image, width=canvas_w - margin * 2, height=row_height)
            canvas.paste(board, (margin, row_y))
            draw_label(margin + 8, row_y + 8, left_label)
        else:
            left_board = _resize_with_pad(left_image, width=panel_w, height=row_height)
            right_board = _resize_with_pad(right_image, width=panel_w, height=row_height)
            canvas.paste(left_board, (margin, row_y))
            canvas.paste(right_board, (margin + panel_w + gap, row_y))
            draw_label(margin + 8, row_y + 8, left_label)
            draw_label(margin + panel_w + gap + 8, row_y + 8, right_label)
        row_y += row_height + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _prefix_columns(df: pd.DataFrame, prefix: str, keep: set[str]) -> pd.DataFrame:
    renamed = {}
    for column in df.columns:
        if column in keep:
            continue
        renamed[column] = f"{prefix}{column}"
    return df.rename(columns=renamed)


def _build_pair_comparison_table(orb_pair_df: pd.DataFrame, aliked_pair_df: pd.DataFrame) -> pd.DataFrame:
    keep_columns = {"pair_key", "image_id", "neighbor_image_id", "identity", "neighbor_identity", "same_identity", "global_score", "left_index", "right_index"}
    orb_prefixed = _prefix_columns(orb_pair_df, "orb_", keep_columns)
    aliked_prefixed = _prefix_columns(aliked_pair_df, "aliked_", keep_columns)
    merged = orb_prefixed.merge(
        aliked_prefixed.drop(columns=["image_id", "neighbor_image_id", "identity", "neighbor_identity", "same_identity", "global_score", "left_index", "right_index"]),
        on="pair_key",
        how="inner",
        validate="one_to_one",
    )
    merged["orb_minus_aliked"] = np.round(
        merged["orb_local_score"].to_numpy(dtype=float) - merged["aliked_local_score"].to_numpy(dtype=float),
        6,
    )
    merged["aliked_minus_orb"] = np.round(-merged["orb_minus_aliked"].to_numpy(dtype=float), 6)
    merged["combined_local_score"] = np.round(
        merged["orb_local_score"].to_numpy(dtype=float) + merged["aliked_local_score"].to_numpy(dtype=float),
        6,
    )
    return merged.sort_values(["same_identity", "global_score"], ascending=[False, False]).reset_index(drop=True)


def _select_examples(pair_df: pd.DataFrame, *, category: str, top_n: int) -> pd.DataFrame:
    margin = 0.15
    min_support = 0.15
    if category == "agreement_same_strong":
        subset = pair_df[
            pair_df["same_identity"].astype(bool)
            & (pair_df["orb_local_score"].astype(float) >= min_support)
            & (pair_df["aliked_local_score"].astype(float) >= min_support)
        ].copy()
        return subset.sort_values(["combined_local_score", "global_score"], ascending=[False, False]).head(int(top_n)).copy()
    if category == "same_orb_better":
        subset = pair_df[
            pair_df["same_identity"].astype(bool)
            & (pair_df["orb_minus_aliked"].astype(float) >= margin)
            & (pair_df["orb_local_score"].astype(float) >= min_support)
        ].copy()
        return subset.sort_values(["orb_minus_aliked", "orb_local_score", "orb_inliers"], ascending=[False, False, False]).head(int(top_n)).copy()
    if category == "same_aliked_better":
        subset = pair_df[
            pair_df["same_identity"].astype(bool)
            & (pair_df["aliked_minus_orb"].astype(float) >= margin)
            & (pair_df["aliked_local_score"].astype(float) >= min_support)
        ].copy()
        return subset.sort_values(["aliked_minus_orb", "aliked_local_score", "aliked_inliers"], ascending=[False, False, False]).head(int(top_n)).copy()
    if category == "false_orb_support":
        subset = pair_df[
            (~pair_df["same_identity"].astype(bool))
            & (pair_df["orb_minus_aliked"].astype(float) >= margin)
            & (pair_df["orb_local_score"].astype(float) >= min_support)
        ].copy()
        return subset.sort_values(["orb_local_score", "orb_minus_aliked", "global_score"], ascending=[False, False, False]).head(int(top_n)).copy()
    if category == "false_aliked_support":
        subset = pair_df[
            (~pair_df["same_identity"].astype(bool))
            & (pair_df["aliked_minus_orb"].astype(float) >= margin)
            & (pair_df["aliked_local_score"].astype(float) >= min_support)
        ].copy()
        return subset.sort_values(["aliked_local_score", "aliked_minus_orb", "global_score"], ascending=[False, False, False]).head(int(top_n)).copy()
    raise ValueError(f"Unknown category: {category}")


def _write_summary(
    *,
    output_path: Path,
    route_dir: Path,
    selected_tables: dict[str, pd.DataFrame],
    image_refs: dict[str, list[tuple[str, str, pd.Series]]],
    pair_df: pd.DataFrame,
    sam_applied_count: int,
    sample_count: int,
    use_yellow_prior: bool,
    yellow_focus_count: int,
    use_yellow_band_orb: bool,
    use_yellow_band_aliked: bool,
) -> None:
    lines = [
        "# Salamander SAM-Masked Local Matcher Review Pack",
        "",
        f"- Source route: `{route_dir.name}`",
        f"- Dataset: `{SALAMANDER_DATASET}`",
        f"- Validation samples: `{sample_count}`",
        f"- Candidate pairs: `{len(pair_df)}`",
        f"- SAM masked applied rows: `{sam_applied_count}/{sample_count}`",
        f"- Yellow prior mode: `{'enabled' if use_yellow_prior else 'disabled'}`",
        f"- ORB yellow-band mode: `{'enabled' if use_yellow_band_orb else 'disabled'}`",
        f"- ALIKED yellow-band mode: `{'enabled' if use_yellow_band_aliked else 'disabled'}`",
        f"- ORB params: `nfeatures={ORB_NFEATURES}`, `max_side={ORB_MAX_SIDE}`, `fast_threshold={ORB_FAST_THRESHOLD}`, `clahe={ORB_CLAHE_CLIP_LIMIT}`, `ratio_test={ORB_RATIO_TEST}`",
        f"- ALIKED + LightGlue view: `Resize(512,512)` on `{'yellow band rgb' if use_yellow_band_aliked else ('yellow focus crop' if use_yellow_prior else 'sam_masked_rgb_v1_resolved_path_v1')}`",
        "",
        "## How To Read",
        "",
        "- 每张图板第一行看原图，第二行看 SAM 去背景后的输入；如果开启黄色先验，会额外出现一行 `yellow focus`。若开启 `yellow band` 模式，会额外出现一行 `yellow band`，表示对应分支只在黄纹边界窄带里取证。",
        "- `orb_local_score` / `aliked_local_score` 越高，代表这条局部匹配分支更可能给这对样本额外加分；但这里先做定性观察，不直接宣称哪条路线整体更优。",
        "- 重点先看 `same_*_better`：它们回答“同个体时谁更稳”；再看 `false_*_support`：它们回答“谁更容易被假相似花纹骗到”。",
        "",
        "## Pair Score Snapshot",
        "",
        f"- Mean `orb_local_score`: `{pair_df['orb_local_score'].mean():.4f}`",
        f"- Mean `aliked_local_score`: `{pair_df['aliked_local_score'].mean():.4f}`",
        f"- Same-ID pairs: `{int(pair_df['same_identity'].astype(bool).sum())}`",
        f"- Different-ID pairs: `{int((~pair_df['same_identity'].astype(bool)).sum())}`",
        f"- Yellow focus available rows: `{yellow_focus_count}/{sample_count}`",
        "",
    ]

    section_titles = {
        "agreement_same_strong": "Same-ID Strong Agreement",
        "same_orb_better": "Same-ID ORB Better",
        "same_aliked_better": "Same-ID ALIKED+LightGlue Better",
        "false_orb_support": "False ORB Support",
        "false_aliked_support": "False ALIKED+LightGlue Support",
    }
    section_notes = {
        "agreement_same_strong": "两条局部路线都看对了，适合先建立肉眼基准：它们到底都在对齐什么区域。",
        "same_orb_better": "同个体里 ORB 更强，重点看它是不是更抓住了黄斑边界或稳定纹理。",
        "same_aliked_better": "同个体里 ALIKED+LightGlue 更强，重点看它是不是在姿态变化下保留了更多可解释连线。",
        "false_orb_support": "不同个体里 ORB 给出高支持，重点看它是不是只被局部黄纹相似性骗到了。",
        "false_aliked_support": "不同个体里 ALIKED+LightGlue 给出高支持，重点看深度局部匹配是否也会被相似花纹误导。",
    }

    for key in ["agreement_same_strong", "same_orb_better", "same_aliked_better", "false_orb_support", "false_aliked_support"]:
        selected_df = selected_tables.get(key)
        lines.extend([f"## {section_titles[key]}", ""])
        lines.append(f"- {section_notes[key]}")
        if selected_df is None or selected_df.empty:
            lines.extend(["- None.", ""])
            continue
        lines.append(f"- Selected rows: `{len(selected_df)}`")
        lines.append(f"- Table: `{key}_selected_pairs_v1.csv`")
        lines.append("")
        for title, rel_path, row in image_refs.get(key, []):
            caption = (
                f"`{title}` | `same_identity={row.get('same_identity', 'na')}` | "
                f"`global={float(row.get('global_score', 0.0)):.3f}` | "
                f"`orb_local={float(row.get('orb_local_score', 0.0)):.3f}` | "
                f"`aliked_local={float(row.get('aliked_local_score', 0.0)):.3f}` | "
                f"`orb_inliers={int(row.get('orb_inliers', 0))}` | "
                f"`aliked_inliers={int(row.get('aliked_inliers', 0))}`"
            )
            lines.append(f"- {caption}")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    parser = argparse.ArgumentParser(description="Build a SAM-masked qualitative review pack comparing ORB vs ALIKED + LightGlue on Salamander validation pairs.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--max-num-keypoints", type=int, default=256)
    parser.add_argument("--lightglue-init-threshold", type=float, default=0.1)
    parser.add_argument("--matcher-batch-size", type=int, default=32)
    parser.add_argument("--use-yellow-prior", action="store_true")
    parser.add_argument("--use-yellow-band-orb", action="store_true")
    parser.add_argument("--use-yellow-band-aliked", action="store_true")
    args = parser.parse_args()

    from animalclef_analysis.orb_rerank_baseline import (
        build_local_match_table,
        build_topk_pair_index,
        cosine_score_matrix,
        extract_orb_features,
    )
    from wildlife_tools.features.local import AlikedExtractor

    route_dir = args.route_dir.resolve()
    manifest_path = args.manifest_path.resolve()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, images_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    metadata_df = _resolve_review_metadata(route_df=metadata_df, manifest_path=manifest_path, repo_root=repo_root)
    match_metadata_df = metadata_df.copy()
    aliked_metadata_df = pd.DataFrame()
    yellow_focus_df = pd.DataFrame()
    yellow_focus_count = 0
    orb_preview_by_image: dict[str, dict[str, object]] = {}
    aliked_preview_by_image: dict[str, dict[str, object]] = {}
    if bool(args.use_yellow_prior):
        match_metadata_df, yellow_focus_df = _build_yellow_focus_review_metadata(
            metadata_df=metadata_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        aliked_metadata_df = match_metadata_df.copy()
        yellow_focus_count = int(match_metadata_df["yellow_focus_available_v1"].astype(bool).sum())
        yellow_focus_df.to_csv(tables_dir / "yellow_focus_manifest_v1.csv", index=False)
        (
            yellow_focus_df.groupby(["dataset", "split"])
            .agg(
                images=("image_id", "count"),
                focus_available=("yellow_focus_available_v1", lambda s: int(np.sum(s))),
                focus_available_ratio=("yellow_focus_available_v1", lambda s: round(float(np.mean(s)), 4)),
            )
            .reset_index()
        ).to_csv(tables_dir / "yellow_focus_summary_v1.csv", index=False)
    if bool(args.use_yellow_band_orb) and not bool(args.use_yellow_prior):
        raise ValueError("--use-yellow-band-orb requires --use-yellow-prior.")
    if bool(args.use_yellow_band_aliked) and not bool(args.use_yellow_prior):
        raise ValueError("--use-yellow-band-aliked requires --use-yellow-prior.")
    if bool(args.use_yellow_band_aliked):
        aliked_metadata_df, aliked_preview_by_image = _build_yellow_band_view_metadata(
            metadata_df=match_metadata_df,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        aliked_metadata_df = aliked_metadata_df.copy()
        aliked_metadata_df["path"] = aliked_metadata_df["yellow_band_rgb_path_v1"].fillna("").astype(str)
        aliked_metadata_df["recommended_model_input_path_v1"] = aliked_metadata_df["path"]
        aliked_metadata_df["preferred_path_v1"] = aliked_metadata_df["path"]
        aliked_metadata_df.to_csv(tables_dir / "yellow_band_manifest_v1.csv", index=False)
    elif aliked_metadata_df.empty:
        aliked_metadata_df = match_metadata_df.copy()
    embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    if len(match_metadata_df) != len(embeddings):
        raise ValueError("Route embeddings do not match validation metadata rows.")

    global_score = cosine_score_matrix(embeddings)
    pair_index = build_topk_pair_index(score_matrix=global_score, top_k=int(args.top_k), query_indices=None)

    if bool(args.use_yellow_band_orb):
        match_metadata_df = match_metadata_df.copy()
        match_metadata_df["focus_mask_path"] = match_metadata_df["yellow_orb_local_focus_mask_path_v1"].fillna("").astype(str)
        orb_features, orb_preview_by_image = _build_yellow_band_orb_features(
            metadata_df=match_metadata_df,
            repo_root=repo_root,
            nfeatures=ORB_NFEATURES,
            max_side=ORB_MAX_SIDE,
            fast_threshold=ORB_FAST_THRESHOLD,
            clahe_clip_limit=ORB_CLAHE_CLIP_LIMIT,
        )
    else:
        orb_features = extract_orb_features(
            df=match_metadata_df,
            repo_root=repo_root,
            nfeatures=ORB_NFEATURES,
            max_side=ORB_MAX_SIDE,
            fast_threshold=ORB_FAST_THRESHOLD,
            clahe_clip_limit=ORB_CLAHE_CLIP_LIMIT,
        )
    orb_pair_df = build_local_match_table(
        df=match_metadata_df,
        features=orb_features,
        pair_index=pair_index,
        ratio_test=ORB_RATIO_TEST,
        ransac_threshold=RANSAC_THRESHOLD,
        min_inliers=MIN_INLIERS,
        local_matcher="orb",
    )
    orb_pair_df["pair_key"] = orb_pair_df.apply(lambda row: _make_pair_key(str(row["image_id"]), str(row["neighbor_image_id"])), axis=1)

    image_dataset = _build_image_dataset(metadata_df=aliked_metadata_df, repo_root=repo_root)
    extractor = AlikedExtractor(device=str(args.device), max_num_keypoints=int(args.max_num_keypoints))
    feature_dataset = extractor(image_dataset)
    aliked_pair_df, aliked_pair_preview_by_key, aliked_points_by_image = _compute_lightglue_pair_rows(
        metadata_df=aliked_metadata_df,
        feature_dataset=feature_dataset,
        pair_index=pair_index,
        device=str(args.device),
        init_threshold=float(args.lightglue_init_threshold),
        batch_size=int(args.matcher_batch_size),
        ransac_threshold=RANSAC_THRESHOLD,
        min_inliers=MIN_INLIERS,
    )

    pair_df = _build_pair_comparison_table(orb_pair_df=orb_pair_df, aliked_pair_df=aliked_pair_df)
    pair_df.to_csv(tables_dir / "all_pairs_scored_v1.csv", index=False)

    paths_by_image = {
        str(row.image_id): {
            "original_path": str(row.original_path),
            "masked_path": str(row.masked_path),
            "focus_path": str(getattr(row, "focus_path", "") or ""),
            "focus_mask_path": str(getattr(row, "focus_mask_path", "") or ""),
            "band_path": str(getattr(row, "yellow_band_rgb_path_v1", "") or ""),
        }
        for row in aliked_metadata_df.itertuples(index=False)
    }
    orb_feature_by_image = {str(feature.image_id): feature for feature in orb_features}

    categories = [
        "agreement_same_strong",
        "same_orb_better",
        "same_aliked_better",
        "false_orb_support",
        "false_aliked_support",
    ]
    category_titles = {
        "agreement_same_strong": "sam masked qualitative review | same-id both local matchers look strong",
        "same_orb_better": "sam masked qualitative review | same-id ORB looks stronger than ALIKED + LightGlue",
        "same_aliked_better": "sam masked qualitative review | same-id ALIKED + LightGlue looks stronger than ORB",
        "false_orb_support": "sam masked qualitative review | different-id but ORB gives strong local support",
        "false_aliked_support": "sam masked qualitative review | different-id but ALIKED + LightGlue gives strong local support",
    }

    selected_tables: dict[str, pd.DataFrame] = {}
    image_refs: dict[str, list[tuple[str, str, pd.Series]]] = {}
    for key in categories:
        selected_df = _select_examples(pair_df=pair_df, category=key, top_n=int(args.top_n))
        selected_tables[key] = selected_df
        selected_df.to_csv(tables_dir / f"{key}_selected_pairs_v1.csv", index=False)
        refs: list[tuple[str, str, pd.Series]] = []
        for index, row in enumerate(selected_df.reset_index(drop=True).itertuples(index=False), start=1):
            row_series = pd.Series(row._asdict())
            left_id = str(row_series["image_id"])
            right_id = str(row_series["neighbor_image_id"])
            image_path = images_dir / key / f"{index:02d}_{left_id}_{right_id}.jpg"
            _build_pair_board(
                row=row_series,
                category_title=category_titles[key],
                paths_by_image=paths_by_image,
                orb_feature_by_image=orb_feature_by_image,
                aliked_points_by_image=aliked_points_by_image,
                aliked_pair_preview_by_key=aliked_pair_preview_by_key,
                repo_root=repo_root,
                output_path=image_path,
                use_yellow_prior=bool(args.use_yellow_prior),
                use_yellow_band_orb=bool(args.use_yellow_band_orb),
                use_yellow_band_aliked=bool(args.use_yellow_band_aliked),
                orb_preview_by_image=orb_preview_by_image,
                aliked_preview_by_image=aliked_preview_by_image,
            )
            refs.append((f"{left_id}_{right_id}", _path_ref(reports_dir, image_path), row_series))
        image_refs[key] = refs

    _write_summary(
        output_path=reports_dir / "summary.md",
        route_dir=route_dir,
        selected_tables=selected_tables,
        image_refs=image_refs,
        pair_df=pair_df,
        sam_applied_count=int(metadata_df["sam_mask_applied"].astype(bool).sum()),
        sample_count=len(metadata_df),
        use_yellow_prior=bool(args.use_yellow_prior),
        yellow_focus_count=int(yellow_focus_count),
        use_yellow_band_orb=bool(args.use_yellow_band_orb),
        use_yellow_band_aliked=bool(args.use_yellow_band_aliked),
    )

    print(f"[sam_mask_review] output_dir: {output_dir}")
    print(f"[sam_mask_review] table: {tables_dir / 'all_pairs_scored_v1.csv'}")
    print(f"[sam_mask_review] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
