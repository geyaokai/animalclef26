#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


DEFAULT_SUBMISSION_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_OUTPUT_DIRNAME = "review_pack_v1"
SALAMANDER_DATASET = "SalamanderID2025"

ORB_NFEATURES = 1024
ORB_MAX_SIDE = 768
ORB_FAST_THRESHOLD = 7
ORB_CLAHE_CLIP_LIMIT = 2.0
RATIO_TEST = 0.8
RANSAC_THRESHOLD = 5.0
MIN_INLIERS = 8


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


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


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_like_feature(image: Image.Image, *, width: int, height: int) -> Image.Image:
    if image.size == (int(width), int(height)):
        return image.copy()
    return image.resize((int(width), int(height)), Image.Resampling.BILINEAR)


def _make_pair_key(left_image_id: str, right_image_id: str) -> str:
    left = str(left_image_id)
    right = str(right_image_id)
    if left <= right:
        return f"{left}__{right}"
    return f"{right}__{left}"


def _compute_inlier_match_preview(
    left_feature,
    right_feature,
    left_rgb: Image.Image,
    right_rgb: Image.Image,
    *,
    ratio_test: float = RATIO_TEST,
    ransac_threshold: float = RANSAC_THRESHOLD,
    max_lines: int = 64,
) -> tuple[Image.Image, dict[str, int]]:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("Current ORB review pack requires OpenCV in the active environment.") from exc

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


def _build_pair_board(
    *,
    row: pd.Series,
    category_title: str,
    feature_by_image: dict[str, object],
    path_by_image: dict[str, str],
    repo_root: Path,
    output_path: Path,
) -> None:
    left_id = str(row["image_id"])
    right_id = str(row["neighbor_image_id"])
    left_feature = feature_by_image[left_id]
    right_feature = feature_by_image[right_id]
    left_rgb = _resize_like_feature(_load_rgb(repo_root / path_by_image[left_id]), width=left_feature.width, height=left_feature.height)
    right_rgb = _resize_like_feature(_load_rgb(repo_root / path_by_image[right_id]), width=right_feature.width, height=right_feature.height)
    left_points = _draw_points(left_rgb, left_feature.points, radius=3)
    right_points = _draw_points(right_rgb, right_feature.points, radius=3)
    match_preview, match_stats = _compute_inlier_match_preview(
        left_feature=left_feature,
        right_feature=right_feature,
        left_rgb=left_rgb,
        right_rgb=right_rgb,
    )

    margin = 18
    gap = 14
    panel_w = 360
    panel_h = 220
    match_h = 320
    canvas_w = margin * 2 + panel_w * 2 + gap
    title_font = _font(22)
    body_font = _font(16)
    info_lines = [
        f"left={left_id} ({row.get('identity', '')})    right={right_id} ({row.get('neighbor_identity', '')})",
        f"same_identity={row.get('same_identity', 'na')}    global_score={float(row.get('global_score', 0.0)):.3f}    local_score={float(row.get('local_score', 0.0)):.3f}",
        f"left_keypoints={int(row.get('left_keypoints', left_feature.point_count))}    right_keypoints={int(row.get('right_keypoints', right_feature.point_count))}    good_matches={int(row.get('good_matches', match_stats['good_matches']))}    inliers={int(row.get('inliers', match_stats['inliers']))}",
    ]
    extra_lines = []
    if "transition_type" in row.index and pd.notna(row.get("transition_type")):
        extra_lines.extend(
            [
                f"transition={str(row.get('transition_type'))}",
                (
                    f"global_top1={str(row.get('global_top1_image_id', ''))} ({str(row.get('global_top1_identity', ''))})"
                    f" -> rerank_top1={str(row.get('reranked_top1_image_id', ''))} ({str(row.get('reranked_top1_identity', ''))})"
                ),
            ]
        )
    header_lines: list[str] = []
    for line in info_lines + extra_lines:
        header_lines.extend(_wrap_text_lines(line, font=body_font, max_width=canvas_w - margin * 2))
    title_h = 38
    line_h = 26
    header_h = title_h + 14 + len(header_lines) * line_h + 24
    canvas_h = margin * 2 + header_h + panel_h * 2 + match_h + gap * 2
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)

    draw.text((margin, margin), category_title, fill=(20, 20, 20), font=title_font)
    for index, line in enumerate(header_lines):
        draw.text((margin, margin + title_h + index * line_h), line, fill=(42, 42, 42), font=body_font)

    raw_left = _resize_with_pad(left_rgb, width=panel_w, height=panel_h)
    raw_right = _resize_with_pad(right_rgb, width=panel_w, height=panel_h)
    pt_left = _resize_with_pad(left_points, width=panel_w, height=panel_h)
    pt_right = _resize_with_pad(right_points, width=panel_w, height=panel_h)
    match_board = _resize_with_pad(match_preview, width=canvas_w - margin * 2, height=match_h)

    top_y = margin + header_h
    canvas.paste(raw_left, (margin, top_y))
    canvas.paste(raw_right, (margin + panel_w + gap, top_y))
    mid_y = top_y + panel_h + gap
    canvas.paste(pt_left, (margin, mid_y))
    canvas.paste(pt_right, (margin + panel_w + gap, mid_y))
    bot_y = mid_y + panel_h + gap
    canvas.paste(match_board, (margin, bot_y))

    def draw_label(x: int, y: int, text: str) -> None:
        bbox = draw.textbbox((x, y), text, font=body_font)
        rect = (bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4)
        draw.rectangle(rect, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255), font=body_font)

    draw_label(margin + 8, top_y + 8, "left image")
    draw_label(margin + panel_w + gap + 8, top_y + 8, "right image")
    draw_label(margin + 8, mid_y + 8, "left + ORB keypoints")
    draw_label(margin + panel_w + gap + 8, mid_y + 8, "right + ORB keypoints")
    draw_label(
        margin + 8,
        bot_y + 8,
        f"inlier matches actually used by ORB rerank | recomputed good_matches={int(match_stats['good_matches'])} inliers={int(match_stats['inliers'])}",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _select_support_same(pair_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    return (
        pair_df[pair_df["same_identity"].astype(bool)]
        .sort_values(["local_score", "inliers", "global_score"], ascending=[False, False, False])
        .head(int(top_n))
        .copy()
    )


def _select_support_false(pair_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    return (
        pair_df[~pair_df["same_identity"].astype(bool)]
        .sort_values(["local_score", "inliers", "global_score"], ascending=[False, False, False])
        .head(int(top_n))
        .copy()
    )


def _select_same_fail(pair_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    subset = pair_df[pair_df["same_identity"].astype(bool)].copy()
    subset = subset[subset["local_score"].astype(float) <= 0.02].copy()
    return subset.sort_values(["global_score", "inliers", "good_matches"], ascending=[False, True, True]).head(int(top_n)).copy()


def _select_corrected_pairs(pair_df: pd.DataFrame, transition_df: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    corrected = transition_df[transition_df["transition_type"].astype(str).eq("corrected")].copy()
    if corrected.empty:
        return corrected
    corrected = corrected.rename(
        columns={
            "global_score": "query_global_top1_score",
            "reranked_score": "query_reranked_top1_score",
        }
    )
    corrected["pair_key"] = corrected.apply(
        lambda row: _make_pair_key(str(row["image_id"]), str(row["reranked_top1_image_id"])),
        axis=1,
    )
    corrected = corrected.sort_values(["query_reranked_top1_score", "query_global_top1_score"], ascending=[False, False]).drop_duplicates(
        subset=["pair_key"],
        keep="first",
    )
    pair_lookup = pair_df.copy()
    pair_lookup["pair_key"] = pair_lookup.apply(
        lambda row: _make_pair_key(str(row["image_id"]), str(row["neighbor_image_id"])),
        axis=1,
    )
    merged = corrected.merge(
        pair_lookup[
            [
                "pair_key",
                "image_id",
                "neighbor_image_id",
                "identity",
                "neighbor_identity",
                "same_identity",
                "global_score",
                "left_keypoints",
                "right_keypoints",
                "good_matches",
                "inliers",
                "local_score",
            ]
        ],
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["image_id_y", "neighbor_image_id"]).rename(columns={"image_id_x": "query_image_id", "identity_x": "query_identity"})
    merged = merged.rename(columns={"image_id_y": "image_id"})
    return merged.sort_values(["local_score", "query_reranked_top1_score"], ascending=[False, False]).head(int(top_n)).copy()


def _write_summary(
    *,
    output_path: Path,
    submission_dir: Path,
    selected_tables: dict[str, pd.DataFrame],
    image_refs: dict[str, list[tuple[str, str, pd.Series]]],
) -> None:
    lines = [
        "# Current Pipeline ORB Review Pack",
        "",
        f"- Source submission route: `{submission_dir.name}`",
        f"- Dataset: `{SALAMANDER_DATASET}`",
        f"- ORB params: `nfeatures={ORB_NFEATURES}`, `max_side={ORB_MAX_SIDE}`, `fast_threshold={ORB_FAST_THRESHOLD}`, `clahe={ORB_CLAHE_CLIP_LIMIT}`",
        f"- Match params: `ratio_test={RATIO_TEST}`, `ransac_threshold={RANSAC_THRESHOLD}`, `min_inliers={MIN_INLIERS}`",
        "",
        "## How To Read",
        "",
        "- 每张图板都分三层：第一行是原图，第二行是当前 pipeline 真正提到的 ORB 点，第三行是通过 RANSAC 留下来的 inlier 连线。",
        "- `local_score` 是当前主线 ORB rerank 真正写回分数矩阵前使用的局部支持分数；`inliers` 越高，通常说明局部几何一致性越强。",
        "- 这套 review pack 看的不是 yellow-band 实验，而是当前 official/active 主线里那条 `fusion + ORB` 路线本身。",
        "",
    ]
    section_titles = {
        "support_same": "Strong Same-ID Supports",
        "support_false": "False Supports",
        "same_fail": "Same-ID Failures",
        "corrected": "Top-1 Corrections",
    }
    for key in ["support_same", "support_false", "same_fail", "corrected"]:
        df = selected_tables.get(key)
        lines.extend([f"## {section_titles[key]}", ""])
        if df is None or df.empty:
            lines.extend(["- None.", ""])
            continue
        lines.append(f"- Selected rows: `{len(df)}`")
        csv_name = f"{key}_selected_pairs_v1.csv"
        lines.append(f"- Table: `{csv_name}`")
        if key == "support_same":
            lines.append("- 先看连线是否沿着同一批黄斑/黑斑边界稳定落下；这代表当前 ORB support 在真正帮主路加分。")
        elif key == "support_false":
            lines.append("- 先看它是不是只在局部相似花纹上对上，而整体身体布局、黄斑拓扑并不一致；这就是假 support。")
        elif key == "same_fail":
            lines.append("- 先看两张是不是明显同个体，但连线几乎没有留下来；这代表当前 ORB 对姿态/尺度/局部遮挡不够稳。")
        elif key == "corrected":
            lines.append("- 先看这类 pair 的 inlier 连线是否比原来的 global top1 更可信；这就是 ORB 真正有用的地方。")
        lines.append("")
        for title, rel_path, row in image_refs.get(key, []):
            caption = (
                f"`{title}` | "
                f"`same_identity={row.get('same_identity', 'na')}` | "
                f"`global={float(row.get('global_score', 0.0)):.3f}` | "
                f"`local={float(row.get('local_score', 0.0)):.3f}` | "
                f"`good_matches={int(row.get('good_matches', 0))}` | "
                f"`inliers={int(row.get('inliers', 0))}`"
            )
            if "transition_type" in row.index and pd.notna(row.get("transition_type")):
                caption += (
                    f" | `transition={str(row.get('transition_type'))}`"
                )
            lines.append(f"- {caption}")
            lines.append(f"![{title}]({rel_path})")
            lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.orb_rerank_baseline import extract_orb_features, resolve_existing_image_rel_path

    parser = argparse.ArgumentParser(description="Build a qualitative review pack for the current Salamander ORB pipeline.")
    parser.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    parser.add_argument("--output-dirname", type=str, default=DEFAULT_OUTPUT_DIRNAME)
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args()

    submission_dir = (repo_root / args.submission_dir).resolve() if not args.submission_dir.is_absolute() else args.submission_dir.resolve()
    output_dir = submission_dir / args.output_dirname
    images_dir = output_dir / "images"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, images_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    val_metadata = pd.read_csv(submission_dir / "embeddings" / "salamander_val_metadata.csv")
    val_metadata["image_id"] = val_metadata["image_id"].astype(str)
    val_metadata["identity"] = val_metadata["identity"].fillna("").astype(str)
    val_metadata["dataset"] = val_metadata["dataset"].astype(str)
    val_metadata = val_metadata[val_metadata["dataset"] == SALAMANDER_DATASET].copy().reset_index(drop=True)

    pair_df = pd.read_csv(submission_dir / "tables" / "val_local_match_scores_v1.csv")
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    pair_df["identity"] = pair_df["identity"].fillna("").astype(str)
    pair_df["neighbor_identity"] = pair_df["neighbor_identity"].fillna("").astype(str)

    transition_df = pd.read_csv(submission_dir / "tables" / "val_top1_transitions_v1.csv")
    transition_df["image_id"] = transition_df["image_id"].astype(str)
    transition_df["identity"] = transition_df["identity"].fillna("").astype(str)
    transition_df["global_top1_image_id"] = transition_df["global_top1_image_id"].astype(str)
    transition_df["global_top1_identity"] = transition_df["global_top1_identity"].fillna("").astype(str)
    transition_df["reranked_top1_image_id"] = transition_df["reranked_top1_image_id"].astype(str)
    transition_df["reranked_top1_identity"] = transition_df["reranked_top1_identity"].fillna("").astype(str)

    resolved_metadata = val_metadata.copy()
    resolved_metadata["path"] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in resolved_metadata.iterrows()]
    path_by_image = dict(zip(resolved_metadata["image_id"].astype(str), resolved_metadata["path"].astype(str), strict=True))

    features = extract_orb_features(
        df=resolved_metadata,
        repo_root=repo_root,
        nfeatures=ORB_NFEATURES,
        max_side=ORB_MAX_SIDE,
        fast_threshold=ORB_FAST_THRESHOLD,
        clahe_clip_limit=ORB_CLAHE_CLIP_LIMIT,
    )
    feature_by_image = {str(feature.image_id): feature for feature in features}

    selected_tables = {
        "support_same": _select_support_same(pair_df, top_n=int(args.top_n)),
        "support_false": _select_support_false(pair_df, top_n=int(args.top_n)),
        "same_fail": _select_same_fail(pair_df, top_n=int(args.top_n)),
        "corrected": _select_corrected_pairs(pair_df, transition_df, top_n=int(args.top_n)),
    }

    category_titles = {
        "support_same": "current pipeline ORB | same-id strong support",
        "support_false": "current pipeline ORB | false support",
        "same_fail": "current pipeline ORB | same-id but local failed",
        "corrected": "current pipeline ORB | corrected top-1 pair",
    }

    image_refs: dict[str, list[tuple[str, str, pd.Series]]] = {}
    for key, selected_df in selected_tables.items():
        if selected_df.empty:
            selected_df.to_csv(tables_dir / f"{key}_selected_pairs_v1.csv", index=False)
            image_refs[key] = []
            continue
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
                feature_by_image=feature_by_image,
                path_by_image=path_by_image,
                repo_root=repo_root,
                output_path=image_path,
            )
            refs.append((f"{left_id}_{right_id}", _path_ref(reports_dir, image_path), row_series))
        image_refs[key] = refs

    _write_summary(
        output_path=reports_dir / "summary.md",
        submission_dir=submission_dir,
        selected_tables=selected_tables,
        image_refs=image_refs,
    )
    print(f"[current_orb_review] output_dir: {output_dir}")
    print(f"[current_orb_review] summary: {reports_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
