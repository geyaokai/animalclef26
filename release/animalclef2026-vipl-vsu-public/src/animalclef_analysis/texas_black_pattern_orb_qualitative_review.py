from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont, ImageOps

from .descriptor_baselines import dataframe_to_markdown_table
from .manual_review_workbench import load_pair_judgments
from .texas_black_pattern_orb_local import (
    BLACK_PATTERN_ALIGNED_PATH_COLUMN,
    BLACK_PATTERN_MASK_PATH_COLUMN,
    BLACK_PATTERN_ORB_MASK_PATH_COLUMN,
    BLACK_PATTERN_ORB_REGION_KIND_COLUMN,
    DEFAULT_CLAHE_CLIP_LIMIT,
    DEFAULT_FAST_THRESHOLD,
    DEFAULT_MAX_SIDE,
    DEFAULT_MIN_INLIERS,
    DEFAULT_NFEATURES,
    DEFAULT_OUTPUT_DIR as DEFAULT_LOCAL_PROBE_DIR,
    DEFAULT_RANSAC_THRESHOLD,
    DEFAULT_RATIO_TEST,
    extract_texas_black_pattern_orb_features,
    merge_texas_black_pattern_orb_local_scores,
)
from .texas_orb_local_probe import TEXAS_DATASET

try:  # pragma: no cover - exercised in wildfusion
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


DEFAULT_REVIEW_SOURCE_DIR = Path("artifacts/analysis/texas_selftrain_review_orb_v1")
DEFAULT_PAIR_JUDGMENTS_PATH = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_black_pattern_orb_qualitative_review_v1")
DEFAULT_TOP_K_PER_CATEGORY = 12


@dataclass(frozen=True)
class CategorySpec:
    name: str
    title: str
    description: str
    sort_columns: tuple[str, ...]
    ascending: tuple[bool, ...]
    query: str


CATEGORY_SPECS = [
    CategorySpec(
        name="black_support_yes",
        title="Black ORB Supports Human YES",
        description="人工判 yes，且黑纹 ORB 给出高分，说明新先验确实抓到了同个体真正看的黑色花纹。",
        sort_columns=("black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"),
        ascending=(False, False, False),
        query="label == 'yes'",
    ),
    CategorySpec(
        name="black_miss_yes",
        title="Black ORB Misses Human YES",
        description="人工判 yes，但黑纹 ORB 仍然偏低，这些是当前黑纹提取或 ORB 仍会漏检的样本。",
        sort_columns=("black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"),
        ascending=(True, True, True),
        query="label == 'yes'",
    ),
    CategorySpec(
        name="black_false_support_no",
        title="Black ORB False Support On Human NO",
        description="人工判 no，但黑纹 ORB 仍然高分；这些样本代表新方案剩余的误导边界。",
        sort_columns=("black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"),
        ascending=(False, False, False),
        query="label == 'no'",
    ),
    CategorySpec(
        name="black_correct_reject_no",
        title="Black ORB Correctly Rejects Human NO",
        description="人工判 no，且黑纹 ORB 很低，说明它在这些负样本上已经更像你人工的黑纹拒识规则。",
        sort_columns=("black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"),
        ascending=(True, True, True),
        query="label == 'no'",
    ),
    CategorySpec(
        name="black_fixes_old_no",
        title="Black ORB Fixes Old ORB False Support",
        description="人工判 no，旧 ORB 偏高但黑纹 ORB 明显降下去；这是这条新方案最关键的改善证据。",
        sort_columns=("old_orb_minus_black", "old_orb_local_score", "black_orb_local_score"),
        ascending=(False, False, True),
        query="label == 'no'",
    ),
    CategorySpec(
        name="black_beats_old_yes",
        title="Black ORB Beats Old ORB On Human YES",
        description="人工判 yes，且黑纹 ORB 比旧 ORB 更强，说明新先验不只是更会拒识，也可能更会支持真实同个体。",
        sort_columns=("black_minus_old_orb", "black_orb_local_score", "black_orb_inliers"),
        ascending=(False, False, False),
        query="label == 'yes'",
    ),
]


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("Texas black-pattern qualitative review requires OpenCV in the active environment.")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _canonical_pair_key(left_image_id: object, right_image_id: object) -> str:
    left = str(left_image_id)
    right = str(right_image_id)
    return f"{left}|{right}" if left <= right else f"{right}|{left}"


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


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
    draw.multiline_text((12, 10), text, fill=(60, 60, 60), font=_font(15), spacing=4)
    return canvas


def _overlay_mask(base_image: Image.Image, mask: np.ndarray, *, color: str, alpha: float = 0.45) -> Image.Image:
    rgb = np.asarray(base_image.convert("RGB"), dtype=np.uint8)
    binary = np.asarray(mask, dtype=np.uint8) > 0
    overlay = np.array(ImageColor.getrgb(color), dtype=np.float32)
    arr = rgb.astype(np.float32)
    arr[binary] = arr[binary] * (1.0 - float(alpha)) + overlay * float(alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _load_probe_config(local_probe_dir: Path) -> dict[str, Any]:
    config = {
        "nfeatures": DEFAULT_NFEATURES,
        "max_side": DEFAULT_MAX_SIDE,
        "fast_threshold": DEFAULT_FAST_THRESHOLD,
        "clahe_clip_limit": DEFAULT_CLAHE_CLIP_LIMIT,
        "ratio_test": DEFAULT_RATIO_TEST,
        "ransac_threshold": DEFAULT_RANSAC_THRESHOLD,
        "min_inliers": DEFAULT_MIN_INLIERS,
    }
    summary_json_path = local_probe_dir / "reports" / "summary.json"
    if not summary_json_path.exists():
        return config
    payload = json.loads(summary_json_path.read_text(encoding="utf-8"))
    for key in list(config.keys()):
        if key in payload:
            config[key] = payload[key]
    return config


def load_texas_black_pattern_judged_pair_df(
    *,
    pair_judgments_path: Path,
    review_source_dir: Path,
    local_probe_dir: Path,
) -> tuple[pd.DataFrame, str]:
    session_name, judgments = load_pair_judgments(pair_judgments_path)
    judgment_df = pd.DataFrame(
        [item for item in judgments if str(item.get("dataset", "")) == TEXAS_DATASET and str(item.get("label", "")) in {"yes", "no"}]
    ).copy()
    if judgment_df.empty:
        raise ValueError(f"No judged Texas pairs found in {pair_judgments_path}")
    judgment_df["image_id"] = judgment_df["image_id"].astype(str)
    judgment_df["neighbor_image_id"] = judgment_df["neighbor_image_id"].astype(str)
    judgment_df["pair_key_canonical"] = [
        _canonical_pair_key(left, right)
        for left, right in zip(judgment_df["image_id"], judgment_df["neighbor_image_id"], strict=True)
    ]

    base_pair_df = pd.read_csv(review_source_dir / "tables" / "test_pair_disagreement_v1.csv").copy()
    base_pair_df["image_id"] = base_pair_df["image_id"].astype(str)
    base_pair_df["neighbor_image_id"] = base_pair_df["neighbor_image_id"].astype(str)
    local_pair_df = pd.read_csv(local_probe_dir / "tables" / "test_pair_local_scores_v1.csv").copy()
    local_pair_df["image_id"] = local_pair_df["image_id"].astype(str)
    local_pair_df["neighbor_image_id"] = local_pair_df["neighbor_image_id"].astype(str)

    enriched_pair_df = merge_texas_black_pattern_orb_local_scores(base_pair_df, local_pair_df)
    enriched_pair_df["pair_key_canonical"] = [
        _canonical_pair_key(left, right)
        for left, right in zip(enriched_pair_df["image_id"], enriched_pair_df["neighbor_image_id"], strict=True)
    ]
    enriched_pair_df = (
        enriched_pair_df.sort_values(
            ["black_orb_local_score", "black_orb_inliers", "black_orb_good_matches"],
            ascending=[False, False, False],
        )
        .drop_duplicates(subset=["pair_key_canonical"], keep="first")
        .reset_index(drop=True)
    )

    judged_pair_df = judgment_df.merge(enriched_pair_df, on="pair_key_canonical", how="left", suffixes=("_judgment", ""))
    if judged_pair_df["black_orb_local_score"].isna().any():
        missing = judged_pair_df.loc[
            judged_pair_df["black_orb_local_score"].isna(),
            ["image_id_judgment", "neighbor_image_id_judgment"],
        ].head(5)
        raise ValueError(f"Missing black ORB rows for some judged Texas pairs: {missing.to_dict(orient='records')}")
    if "image_id_judgment" in judged_pair_df.columns:
        judged_pair_df["judgment_image_id"] = judged_pair_df["image_id_judgment"].astype(str)
        judged_pair_df["judgment_neighbor_image_id"] = judged_pair_df["neighbor_image_id_judgment"].astype(str)
    judged_pair_df["old_orb_local_score"] = pd.to_numeric(
        judged_pair_df.get("orb_local_score", judged_pair_df.get("local_score", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    judged_pair_df["black_orb_local_score"] = pd.to_numeric(judged_pair_df["black_orb_local_score"], errors="coerce").fillna(0.0)
    judged_pair_df["miew_local_score"] = pd.to_numeric(judged_pair_df.get("miew_local_score", 0.0), errors="coerce").fillna(0.0)
    judged_pair_df["black_minus_old_orb"] = judged_pair_df["black_orb_local_score"] - judged_pair_df["old_orb_local_score"]
    judged_pair_df["old_orb_minus_black"] = judged_pair_df["old_orb_local_score"] - judged_pair_df["black_orb_local_score"]
    judged_pair_df["black_minus_miew"] = judged_pair_df["black_orb_local_score"] - judged_pair_df["miew_local_score"]
    return judged_pair_df.reset_index(drop=True), session_name


def summarize_judged_pairs(judged_pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_rows: list[dict[str, object]] = []
    for label, group in judged_pair_df.groupby("label", sort=True):
        label_rows.append(
            {
                "label": str(label),
                "pair_count": int(len(group)),
                "merge_pair_count": int(group["candidate_type"].astype(str).eq("merge").sum()),
                "split_pair_count": int(group["candidate_type"].astype(str).eq("split").sum()),
                "mean_black_orb_local_score": round(float(group["black_orb_local_score"].astype(float).mean()), 6),
                "median_black_orb_local_score": round(float(group["black_orb_local_score"].astype(float).median()), 6),
                "mean_old_orb_local_score": round(float(group["old_orb_local_score"].astype(float).mean()), 6),
                "mean_miew_local_score": round(float(group["miew_local_score"].astype(float).mean()), 6),
                "mean_black_orb_inliers": round(float(group["black_orb_inliers"].astype(float).mean()), 6),
                "mean_black_minus_old_orb": round(float(group["black_minus_old_orb"].astype(float).mean()), 6),
            }
        )
    label_summary_df = pd.DataFrame(label_rows).sort_values("label").reset_index(drop=True)

    threshold_rows: list[dict[str, object]] = []
    for threshold in [0.2, 0.4, 0.6, 0.8]:
        high_mask = judged_pair_df["black_orb_local_score"].astype(float).ge(float(threshold))
        low_mask = judged_pair_df["black_orb_local_score"].astype(float).le(float(1.0 - threshold))
        threshold_rows.append(
            {
                "black_orb_score_threshold": float(threshold),
                "high_score_pair_count": int(high_mask.sum()),
                "high_score_yes_ratio": round(float(judged_pair_df.loc[high_mask, "label"].astype(str).eq("yes").mean()) if high_mask.any() else 0.0, 6),
                "high_score_no_ratio": round(float(judged_pair_df.loc[high_mask, "label"].astype(str).eq("no").mean()) if high_mask.any() else 0.0, 6),
                "low_score_pair_count": int(low_mask.sum()),
                "low_score_yes_ratio": round(float(judged_pair_df.loc[low_mask, "label"].astype(str).eq("yes").mean()) if low_mask.any() else 0.0, 6),
                "low_score_no_ratio": round(float(judged_pair_df.loc[low_mask, "label"].astype(str).eq("no").mean()) if low_mask.any() else 0.0, 6),
            }
        )
    threshold_summary_df = pd.DataFrame(threshold_rows)

    delta_rows: list[dict[str, object]] = []
    for label in ["no", "yes"]:
        subset = judged_pair_df[judged_pair_df["label"].astype(str).eq(label)].copy()
        delta_rows.append(
            {
                "label": label,
                "old_orb_ge_0p6": int(subset["old_orb_local_score"].astype(float).ge(0.6).sum()),
                "black_orb_ge_0p6": int(subset["black_orb_local_score"].astype(float).ge(0.6).sum()),
                "old_high_black_low": int(
                    (subset["old_orb_local_score"].astype(float).ge(0.6) & subset["black_orb_local_score"].astype(float).le(0.2)).sum()
                ),
                "old_low_black_high": int(
                    (subset["old_orb_local_score"].astype(float).le(0.2) & subset["black_orb_local_score"].astype(float).ge(0.6)).sum()
                ),
                "mean_black_minus_old_orb": round(float(subset["black_minus_old_orb"].astype(float).mean()) if not subset.empty else 0.0, 6),
            }
        )
    delta_summary_df = pd.DataFrame(delta_rows)
    return label_summary_df, threshold_summary_df, delta_summary_df


def select_category_rows(
    judged_pair_df: pd.DataFrame,
    *,
    spec: CategorySpec,
    top_k: int,
) -> pd.DataFrame:
    subset = judged_pair_df.query(spec.query).copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(list(spec.sort_columns), ascending=list(spec.ascending)).reset_index(drop=True)
    return subset.head(int(top_k)).copy()


def _compute_inlier_match_preview(
    left_feature,
    right_feature,
    left_rgb: Image.Image,
    right_rgb: Image.Image,
    *,
    ratio_test: float,
    ransac_threshold: float,
    max_lines: int = 64,
) -> tuple[Image.Image, dict[str, int]]:
    _require_cv2()
    left_base = np.asarray(left_rgb.convert("RGB"), dtype=np.uint8)
    right_base = np.asarray(right_rgb.convert("RGB"), dtype=np.uint8)
    stats = {"good_matches": 0, "inliers": 0}
    left_descriptors = left_feature.descriptors
    right_descriptors = right_feature.descriptors
    left_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in np.asarray(left_feature.points, dtype=np.float32)]
    right_keypoints = [cv2.KeyPoint(float(x), float(y), 8) for x, y in np.asarray(right_feature.points, dtype=np.float32)]
    if left_descriptors is None or right_descriptors is None or len(left_descriptors) < 2 or len(right_descriptors) < 2:
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
    left_points = np.float32([left_feature.points[match.queryIdx] for match in good_matches])
    right_points = np.float32([right_feature.points[match.trainIdx] for match in good_matches])
    try:
        _homography, mask = cv2.findHomography(left_points, right_points, cv2.RANSAC, float(ransac_threshold))
    except cv2.error:
        mask = None
    keep = mask.ravel().astype(bool) if mask is not None else np.zeros(len(good_matches), dtype=bool)
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


def build_pair_board(
    *,
    row: pd.Series,
    view_lookup: dict[str, dict[str, Any]],
    feature_by_image: dict[str, object],
    config: dict[str, Any],
    repo_root: Path,
    output_path: Path,
    category_title: str,
) -> None:
    left_id = str(row["image_id"])
    right_id = str(row["neighbor_image_id"])
    left_meta = view_lookup[left_id]
    right_meta = view_lookup[right_id]

    def build_panels(meta: dict[str, Any]) -> list[Image.Image]:
        original = _load_rgb(repo_root / str(meta["path"])) if str(meta.get("path", "")).strip() else _placeholder_panel("original missing", width=180, height=130)
        sam_path = str(meta.get("sam_masked_rgb_path_v1", "") or "")
        sam_image = _load_rgb(repo_root / sam_path) if sam_path and (repo_root / sam_path).exists() else _placeholder_panel("sam missing", width=180, height=130)
        aligned_path = str(meta.get(BLACK_PATTERN_ALIGNED_PATH_COLUMN, "") or "")
        if aligned_path and (repo_root / aligned_path).exists():
            aligned_image = _load_rgb(repo_root / aligned_path)
            black_mask = _load_mask(repo_root / str(meta[BLACK_PATTERN_MASK_PATH_COLUMN]))
            orb_mask = _load_mask(repo_root / str(meta[BLACK_PATTERN_ORB_MASK_PATH_COLUMN]))
            overlay = _overlay_mask(aligned_image, black_mask, color="#00bcd4", alpha=0.45)
            overlay = _overlay_mask(overlay, orb_mask, color="#e91e63", alpha=0.35)
        else:
            aligned_image = _placeholder_panel("aligned missing", width=180, height=130)
            overlay = _placeholder_panel("prior missing", width=180, height=130)
        return [original, sam_image, aligned_image, overlay]

    left_panels = build_panels(left_meta)
    right_panels = build_panels(right_meta)

    left_feature = feature_by_image[left_id]
    right_feature = feature_by_image[right_id]
    left_aligned_path = str(left_meta.get(BLACK_PATTERN_ALIGNED_PATH_COLUMN, "") or "")
    right_aligned_path = str(right_meta.get(BLACK_PATTERN_ALIGNED_PATH_COLUMN, "") or "")
    if left_aligned_path and right_aligned_path and (repo_root / left_aligned_path).exists() and (repo_root / right_aligned_path).exists():
        left_aligned = _load_rgb(repo_root / left_aligned_path).resize((left_feature.width, left_feature.height), Image.Resampling.BILINEAR)
        right_aligned = _load_rgb(repo_root / right_aligned_path).resize((right_feature.width, right_feature.height), Image.Resampling.BILINEAR)
        match_preview, recomputed_stats = _compute_inlier_match_preview(
            left_feature=left_feature,
            right_feature=right_feature,
            left_rgb=left_aligned,
            right_rgb=right_aligned,
            ratio_test=float(config["ratio_test"]),
            ransac_threshold=float(config["ransac_threshold"]),
        )
    else:
        match_preview = _placeholder_panel("aligned match preview missing", width=760, height=240)
        recomputed_stats = {"good_matches": 0, "inliers": 0}

    margin = 18
    gap = 10
    panel_w = 180
    panel_h = 130
    panel_title_h = 24
    row_title_h = 28
    title_font = _font(22)
    body_font = _font(15)
    small_font = _font(13)

    panel_titles = ["original", "sam_masked", "masked_aligned", "black+orb"]
    total_panel_h = panel_h + panel_title_h
    row_h = row_title_h + total_panel_h
    match_h = 250
    width = margin * 2 + panel_w * 4 + gap * 3
    header_lines = [
        f"{left_id} vs {right_id} | label={row['label']} | candidate={row['candidate_type']} {row['candidate_key']}",
        f"black_orb={float(row.get('black_orb_local_score', 0.0)):.3f} | old_orb={float(row.get('old_orb_local_score', 0.0)):.3f} | miew={float(row.get('miew_local_score', 0.0)):.3f}",
        f"black-old={float(row.get('black_minus_old_orb', 0.0)):.3f} | black_inliers={int(row.get('black_orb_inliers', 0))} | old_inliers={int(row.get('orb_inliers', 0))}",
        f"xgb_prob={float(row.get('xgb_same_identity_prob', 0.0)):.3f} | ambiguity={float(row.get('ambiguity_score', 0.0)):.3f} | recomputed_inliers={int(recomputed_stats['inliers'])}",
    ]
    header_h = 38 + len(header_lines) * 22
    height = margin * 2 + header_h + row_h * 2 + gap + match_h + 24
    canvas = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), category_title, fill=(20, 20, 20), font=title_font)
    for idx, line in enumerate(header_lines):
        draw.text((margin, margin + 32 + idx * 22), line, fill=(42, 42, 42), font=body_font)

    current_y = margin + header_h
    for row_name, image_id, panels, meta, feature in [
        ("left", left_id, left_panels, left_meta, left_feature),
        ("right", right_id, right_panels, right_meta, right_feature),
    ]:
        row_title = (
            f"{row_name} {image_id} | black_ratio={float(meta.get('texas_black_pattern_black_ratio_reference_v1', 0.0)):.3f} | "
            f"region={str(meta.get(BLACK_PATTERN_ORB_REGION_KIND_COLUMN, 'none'))} | keypoints={int(feature.point_count)}"
        )
        draw.text((margin, current_y), row_title, fill=(50, 50, 50), font=small_font)
        panel_y = current_y + row_title_h
        for panel_idx, (panel_title, panel_image) in enumerate(zip(panel_titles, panels, strict=True)):
            x0 = margin + panel_idx * (panel_w + gap)
            draw.text((x0, panel_y), panel_title, fill=(60, 60, 60), font=small_font)
            canvas.paste(_resize_with_pad(panel_image, width=panel_w, height=panel_h), (x0, panel_y + panel_title_h))
        current_y += row_h

    current_y += gap
    draw.text((margin, current_y), "inlier match preview", fill=(60, 60, 60), font=small_font)
    canvas.paste(_resize_with_pad(match_preview, width=width - margin * 2, height=match_h), (margin, current_y + 18))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_summary_markdown(
    *,
    output_path: Path,
    repo_root: Path,
    session_name: str,
    pair_judgments_path: Path,
    review_source_dir: Path,
    local_probe_dir: Path,
    label_summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    delta_summary_df: pd.DataFrame,
    category_outputs: list[dict[str, object]],
) -> None:
    lines = [
        "# Texas Black Pattern ORB Qualitative Review v1",
        "",
        "## Goal",
        "",
        "- 对齐 Texas 当前人工 judgment，直接看“黑纹先验 + mask-first aligned ORB”是否更接近你的人工判法。",
        "- 当前重点不是重新聚类，而是定性确认这条新局部分支到底减少了多少旧 ORB 的误导，又保住了多少真实黑纹对应。",
        "",
        "## How To Read",
        "",
        "- 每张 pair board 的两行分别对应左图和右图，四列依次是：`original`、`sam_masked`、`masked_aligned`、`black+orb`。",
        "- `black+orb` 里青色是黑纹先验，洋红色是实际喂给 ORB 的局部区域；如果洋红色仍落到你人工不会看的区域，这条边就值得继续排查。",
        "- 最下面的 `inlier match preview` 是在新黑纹 ORB 特征上重算的内点连线，先看它连的是不是同类黑纹，再看分数。",
        "",
        "## Inputs",
        "",
        f"- `session_name`: `{session_name}`",
        f"- `pair_judgments_path`: `{_path_ref(repo_root, pair_judgments_path)}`",
        f"- `review_source_dir`: `{_path_ref(repo_root, review_source_dir)}`",
        f"- `local_probe_dir`: `{_path_ref(repo_root, local_probe_dir)}`",
        "",
        "## Label Summary",
        "",
        dataframe_to_markdown_table(label_summary_df),
        "",
        "## Black ORB Threshold Snapshot",
        "",
        dataframe_to_markdown_table(threshold_summary_df),
        "",
        "## Delta Vs Old ORB",
        "",
        dataframe_to_markdown_table(delta_summary_df),
        "",
    ]
    for category in category_outputs:
        lines.extend(
            [
                f"## {category['title']}",
                "",
                f"- {category['description']}",
                f"- `pair_count`: `{category['pair_count']}`",
                "",
                dataframe_to_markdown_table(category["table_preview"]),
                "",
            ]
        )
        for image_rel in category["embedded_images"]:
            lines.append(f"![{category['name']}]({image_rel})")
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_texas_black_pattern_orb_qualitative_review(
    *,
    repo_root: Path,
    local_probe_dir: Path,
    review_source_dir: Path,
    pair_judgments_path: Path,
    output_dir: Path,
    top_k_per_category: int = DEFAULT_TOP_K_PER_CATEGORY,
) -> dict[str, Path]:
    resolved_local_probe_dir = (local_probe_dir if local_probe_dir.is_absolute() else (repo_root / local_probe_dir)).resolve()
    resolved_review_source_dir = (review_source_dir if review_source_dir.is_absolute() else (repo_root / review_source_dir)).resolve()
    resolved_pair_judgments_path = (
        pair_judgments_path if pair_judgments_path.is_absolute() else (repo_root / pair_judgments_path)
    ).resolve()
    resolved_output_dir = (output_dir if output_dir.is_absolute() else (repo_root / output_dir)).resolve()

    tables_dir = resolved_output_dir / "tables"
    figures_dir = resolved_output_dir / "figures"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    judged_pair_df, session_name = load_texas_black_pattern_judged_pair_df(
        pair_judgments_path=resolved_pair_judgments_path,
        review_source_dir=resolved_review_source_dir,
        local_probe_dir=resolved_local_probe_dir,
    )
    label_summary_df, threshold_summary_df, delta_summary_df = summarize_judged_pairs(judged_pair_df)
    judged_pair_df.to_csv(tables_dir / "judged_pairs_enriched_v1.csv", index=False)
    label_summary_df.to_csv(tables_dir / "label_summary_v1.csv", index=False)
    threshold_summary_df.to_csv(tables_dir / "threshold_snapshot_v1.csv", index=False)
    delta_summary_df.to_csv(tables_dir / "delta_summary_v1.csv", index=False)

    view_df = pd.read_csv(resolved_local_probe_dir / "tables" / "image_feature_stats_v1.csv").copy()
    view_df["image_id"] = view_df["image_id"].astype(str)
    config = _load_probe_config(resolved_local_probe_dir)
    features = extract_texas_black_pattern_orb_features(
        view_df=view_df,
        repo_root=repo_root,
        nfeatures=int(config["nfeatures"]),
        max_side=int(config["max_side"]),
        fast_threshold=int(config["fast_threshold"]),
        clahe_clip_limit=float(config["clahe_clip_limit"]),
    )
    feature_by_image = {str(feature.image_id): feature for feature in features}
    view_lookup = {str(row["image_id"]): row for row in view_df.to_dict(orient="records")}

    category_outputs: list[dict[str, object]] = []
    for spec in CATEGORY_SPECS:
        subset = select_category_rows(judged_pair_df=judged_pair_df, spec=spec, top_k=int(top_k_per_category))
        category_dir = figures_dir / spec.name
        embedded_images: list[str] = []
        preview_columns = [
            "image_id",
            "neighbor_image_id",
            "candidate_type",
            "candidate_key",
            "label",
            "black_orb_local_score",
            "old_orb_local_score",
            "miew_local_score",
            "black_minus_old_orb",
            "black_orb_inliers",
        ]
        table_preview = subset.loc[:, [column for column in preview_columns if column in subset.columns]].head(8).copy() if not subset.empty else pd.DataFrame(columns=preview_columns)
        if not subset.empty:
            table_preview.to_csv(tables_dir / f"{spec.name}_pairs_v1.csv", index=False)
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                output_path = category_dir / f"{rank:02d}_{row.image_id}_{row.neighbor_image_id}.jpg"
                build_pair_board(
                    row=pd.Series(row._asdict()),
                    view_lookup=view_lookup,
                    feature_by_image=feature_by_image,
                    config=config,
                    repo_root=repo_root,
                    output_path=output_path,
                    category_title=spec.title,
                )
                if rank <= 3:
                    embedded_images.append(_path_ref(reports_dir, output_path))
        category_outputs.append(
            {
                "name": spec.name,
                "title": spec.title,
                "description": spec.description,
                "pair_count": int(len(subset)),
                "table_preview": table_preview,
                "embedded_images": embedded_images,
            }
        )

    summary = {
        "probe": resolved_output_dir.name,
        "dataset": TEXAS_DATASET,
        "session_name": session_name,
        "judged_pair_count": int(len(judged_pair_df)),
        "yes_pair_count": int(judged_pair_df["label"].astype(str).eq("yes").sum()),
        "no_pair_count": int(judged_pair_df["label"].astype(str).eq("no").sum()),
        "mean_black_yes": round(float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("yes"), "black_orb_local_score"].astype(float).mean()), 6),
        "mean_black_no": round(float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("no"), "black_orb_local_score"].astype(float).mean()), 6),
        "mean_old_yes": round(float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("yes"), "old_orb_local_score"].astype(float).mean()), 6),
        "mean_old_no": round(float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("no"), "old_orb_local_score"].astype(float).mean()), 6),
        "no_old_high_black_low": int(
            (
                judged_pair_df["label"].astype(str).eq("no")
                & judged_pair_df["old_orb_local_score"].astype(float).ge(0.6)
                & judged_pair_df["black_orb_local_score"].astype(float).le(0.2)
            ).sum()
        ),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(
        output_path=reports_dir / "summary.md",
        repo_root=repo_root,
        session_name=session_name,
        pair_judgments_path=resolved_pair_judgments_path,
        review_source_dir=resolved_review_source_dir,
        local_probe_dir=resolved_local_probe_dir,
        label_summary_df=label_summary_df,
        threshold_summary_df=threshold_summary_df,
        delta_summary_df=delta_summary_df,
        category_outputs=category_outputs,
    )
    return {
        "summary_path": reports_dir / "summary.md",
        "judged_pairs_path": tables_dir / "judged_pairs_enriched_v1.csv",
        "label_summary_path": tables_dir / "label_summary_v1.csv",
        "delta_summary_path": tables_dir / "delta_summary_v1.csv",
    }
