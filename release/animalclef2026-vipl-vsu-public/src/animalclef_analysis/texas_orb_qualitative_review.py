from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .descriptor_baselines import PATH_COLUMN, dataframe_to_markdown_table
from .manual_review_workbench import load_pair_judgments
from .orb_rerank_baseline import extract_local_features
from .texas_orb_local_probe import (
    DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    TEXAS_DATASET,
    load_aligned_texas_view_df,
    load_texas_reference_df,
    resolve_predictions_path,
)

try:  # pragma: no cover - exercised in wildfusion
    import cv2
except ModuleNotFoundError:  # pragma: no cover
    cv2 = None


DEFAULT_TEXAS_REVIEW_DIR = Path("artifacts/analysis/texas_selftrain_review_orb_v1")
DEFAULT_TEXAS_JUDGMENTS_PATH = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_orb_qualitative_review_v1")
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
        name="orb_support_yes",
        title="ORB Supports Human YES",
        description="人工判 yes，且 ORB 给出很强的局部支持。",
        sort_columns=("local_score", "orb_inliers", "orb_good_matches", "xgb_same_identity_prob"),
        ascending=(False, False, False, False),
        query="label == 'yes'",
    ),
    CategorySpec(
        name="orb_fail_yes",
        title="ORB Misses Human YES",
        description="人工判 yes，但 ORB 支持偏弱，说明局部匹配可能漏检。",
        sort_columns=("local_score", "orb_inliers", "orb_good_matches", "xgb_same_identity_prob"),
        ascending=(True, True, True, True),
        query="label == 'yes'",
    ),
    CategorySpec(
        name="orb_false_support_no",
        title="ORB False Support On Human NO",
        description="人工判 no，但 ORB 给出高分，这是最值得重点排查的误导样本。",
        sort_columns=("local_score", "orb_inliers", "orb_good_matches", "xgb_same_identity_prob"),
        ascending=(False, False, False, False),
        query="label == 'no'",
    ),
    CategorySpec(
        name="orb_correct_reject_no",
        title="ORB Correctly Rejects Human NO",
        description="人工判 no，且 ORB 分数很低，说明它在这些负样本上能有效拒识。",
        sort_columns=("local_score", "orb_inliers", "orb_good_matches", "xgb_same_identity_prob"),
        ascending=(True, True, True, True),
        query="label == 'no'",
    ),
    CategorySpec(
        name="orb_beats_miew_yes",
        title="ORB Beats Miew On Human YES",
        description="人工判 yes，且 ORB 明显高于原 miew 局部证据。",
        sort_columns=("orb_minus_miew", "orb_inliers", "local_score"),
        ascending=(False, False, False),
        query="label == 'yes'",
    ),
    CategorySpec(
        name="orb_beats_miew_no",
        title="ORB Beats Miew On Human NO",
        description="人工判 no，但 ORB 比 miew 更高，这类样本代表 ORB 可能把错误 merge 放大。",
        sort_columns=("orb_minus_miew", "orb_inliers", "local_score"),
        ascending=(False, False, False),
        query="label == 'no'",
    ),
]


def _require_cv2() -> None:
    if cv2 is None:
        raise ModuleNotFoundError("Texas ORB qualitative review requires OpenCV in the active environment.")


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


def load_texas_judged_pair_df(
    *,
    repo_root: Path,
    pair_judgments_path: Path,
    review_dir: Path,
) -> tuple[pd.DataFrame, str]:
    session_name, judgments = load_pair_judgments(pair_judgments_path)
    judgment_df = pd.DataFrame([item for item in judgments if str(item.get("dataset", "")) == TEXAS_DATASET]).copy()
    if judgment_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} judgments found in {pair_judgments_path}")
    judgment_df["image_id"] = judgment_df["image_id"].astype(str)
    judgment_df["neighbor_image_id"] = judgment_df["neighbor_image_id"].astype(str)
    judgment_df["label"] = judgment_df["label"].astype(str)
    judgment_df["pair_key_canonical"] = [
        _canonical_pair_key(left, right)
        for left, right in zip(judgment_df["image_id"], judgment_df["neighbor_image_id"], strict=True)
    ]

    pair_df = pd.read_csv(review_dir / "tables" / "test_pair_disagreement_v1.csv").copy()
    pair_df["image_id"] = pair_df["image_id"].astype(str)
    pair_df["neighbor_image_id"] = pair_df["neighbor_image_id"].astype(str)
    pair_df["pair_key_canonical"] = [
        _canonical_pair_key(left, right)
        for left, right in zip(pair_df["image_id"], pair_df["neighbor_image_id"], strict=True)
    ]
    pair_df = (
        pair_df.sort_values(["local_score", "orb_inliers", "orb_good_matches"], ascending=[False, False, False])
        .drop_duplicates(subset=["pair_key_canonical"], keep="first")
        .reset_index(drop=True)
    )
    merged = judgment_df.merge(pair_df, on="pair_key_canonical", how="left", suffixes=("_judgment", ""))
    if merged["local_score"].isna().any():
        missing = merged.loc[merged["local_score"].isna(), ["image_id_judgment", "neighbor_image_id_judgment"]].head(5)
        raise ValueError(f"Missing ORB review rows for some judged pairs, examples: {missing.to_dict(orient='records')}")
    if "image_id_judgment" in merged.columns:
        merged["judgment_image_id"] = merged["image_id_judgment"].astype(str)
        merged["judgment_neighbor_image_id"] = merged["neighbor_image_id_judgment"].astype(str)
    merged["orb_minus_miew"] = pd.to_numeric(merged["local_score"], errors="coerce") - pd.to_numeric(
        merged.get("miew_local_score", 0.0),
        errors="coerce",
    )
    return merged.reset_index(drop=True), session_name


def summarize_judged_pairs(judged_pair_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_summary_rows: list[dict[str, object]] = []
    for label, group in judged_pair_df.groupby("label", sort=True):
        label_summary_rows.append(
            {
                "label": str(label),
                "pair_count": int(len(group)),
                "merge_pair_count": int(group["candidate_type"].astype(str).eq("merge").sum()),
                "split_pair_count": int(group["candidate_type"].astype(str).eq("split").sum()),
                "mean_orb_local_score": round(float(group["local_score"].astype(float).mean()), 6),
                "median_orb_local_score": round(float(group["local_score"].astype(float).median()), 6),
                "mean_miew_local_score": round(float(group["miew_local_score"].astype(float).mean()), 6),
                "mean_orb_inliers": round(float(group["orb_inliers"].astype(float).mean()), 6),
                "mean_orb_good_matches": round(float(group["orb_good_matches"].astype(float).mean()), 6),
                "mean_xgb_prob": round(float(group["xgb_same_identity_prob"].astype(float).mean()), 6),
            }
        )
    label_summary_df = pd.DataFrame(label_summary_rows).sort_values("label").reset_index(drop=True)

    threshold_rows: list[dict[str, object]] = []
    thresholds = [0.2, 0.4, 0.6, 0.8]
    for threshold in thresholds:
        high_mask = judged_pair_df["local_score"].astype(float).ge(float(threshold))
        low_mask = judged_pair_df["local_score"].astype(float).le(float(1.0 - threshold))
        threshold_rows.append(
            {
                "orb_score_threshold": float(threshold),
                "high_score_pair_count": int(high_mask.sum()),
                "high_score_yes_ratio": round(
                    float(judged_pair_df.loc[high_mask, "label"].astype(str).eq("yes").mean()) if high_mask.any() else 0.0,
                    6,
                ),
                "high_score_no_ratio": round(
                    float(judged_pair_df.loc[high_mask, "label"].astype(str).eq("no").mean()) if high_mask.any() else 0.0,
                    6,
                ),
                "low_score_pair_count": int(low_mask.sum()),
                "low_score_yes_ratio": round(
                    float(judged_pair_df.loc[low_mask, "label"].astype(str).eq("yes").mean()) if low_mask.any() else 0.0,
                    6,
                ),
                "low_score_no_ratio": round(
                    float(judged_pair_df.loc[low_mask, "label"].astype(str).eq("no").mean()) if low_mask.any() else 0.0,
                    6,
                ),
            }
        )
    threshold_summary_df = pd.DataFrame(threshold_rows)
    return label_summary_df, threshold_summary_df


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


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _resize_with_pad(image: Image.Image, *, width: int, height: int) -> Image.Image:
    fitted = ImageOps.contain(image, (int(width), int(height)))
    canvas = Image.new("RGB", (int(width), int(height)), (246, 246, 246))
    offset = ((int(width) - fitted.width) // 2, (int(height) - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def _compute_inlier_match_preview(
    left_feature,
    right_feature,
    left_rgb: Image.Image,
    right_rgb: Image.Image,
    *,
    ratio_test: float = 0.85,
    ransac_threshold: float = 5.0,
    max_lines: int = 64,
) -> tuple[Image.Image, dict[str, int]]:
    _require_cv2()
    left_base = np.asarray(left_rgb.convert("RGB"), dtype=np.uint8)
    right_base = np.asarray(right_rgb.convert("RGB"), dtype=np.uint8)
    left_descriptors = left_feature.descriptors
    right_descriptors = right_feature.descriptors
    stats = {"good_matches": 0, "inliers": 0}

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
    left_rgb = _load_rgb(repo_root / path_by_image[left_id]).resize((left_feature.width, left_feature.height), Image.Resampling.BILINEAR)
    right_rgb = _load_rgb(repo_root / path_by_image[right_id]).resize((right_feature.width, right_feature.height), Image.Resampling.BILINEAR)
    match_preview, recomputed_stats = _compute_inlier_match_preview(
        left_feature=left_feature,
        right_feature=right_feature,
        left_rgb=left_rgb,
        right_rgb=right_rgb,
    )

    margin = 18
    gap = 12
    panel_w = 320
    panel_h = 220
    match_h = 280
    title_font = _font(22)
    body_font = _font(16)
    canvas_w = margin * 2 + panel_w * 2 + gap
    title_h = 34
    lines = [
        f"{left_id} vs {right_id} | label={row['label']} | candidate={row['candidate_type']} {row['candidate_key']}",
        (
            f"orb_local={float(row.get('local_score', 0.0)):.3f} | "
            f"miew_local={float(row.get('miew_local_score', 0.0)):.3f} | "
            f"orb_minus_miew={float(row.get('orb_minus_miew', 0.0)):.3f}"
        ),
        (
            f"orb_inliers={int(row.get('orb_inliers', 0))} | "
            f"orb_good={int(row.get('orb_good_matches', 0))} | "
            f"recomputed_inliers={int(recomputed_stats['inliers'])}"
        ),
        (
            f"xgb_prob={float(row.get('xgb_same_identity_prob', 0.0)):.3f} | "
            f"route_global={float(row.get('route_global_score', 0.0)):.3f} | "
            f"ambiguity={float(row.get('ambiguity_score', 0.0)):.3f}"
        ),
    ]
    header_h = title_h + 16 + len(lines) * 24
    canvas_h = margin * 2 + header_h + panel_h + gap + match_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), category_title, fill=(20, 20, 20), font=title_font)
    for index, line in enumerate(lines):
        draw.text((margin, margin + title_h + index * 24), line, fill=(42, 42, 42), font=body_font)

    top_y = margin + header_h
    canvas.paste(_resize_with_pad(left_rgb, width=panel_w, height=panel_h), (margin, top_y))
    canvas.paste(_resize_with_pad(right_rgb, width=panel_w, height=panel_h), (margin + panel_w + gap, top_y))
    bottom_y = top_y + panel_h + gap
    canvas.paste(_resize_with_pad(match_preview, width=canvas_w - margin * 2, height=match_h), (margin, bottom_y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_summary_markdown(
    *,
    output_path: Path,
    repo_root: Path,
    session_name: str,
    pair_judgments_path: Path,
    review_dir: Path,
    view_manifest_path: Path,
    label_summary_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    category_outputs: list[dict[str, object]],
) -> None:
    lines = [
        "# Texas ORB Qualitative Review v1",
        "",
        "## Goal",
        "",
        "- 对齐 Texas 当前人工 judgment，定性看 ORB 局部证据到底在支持什么、误导什么。",
        "- 当前重点不是重新聚类，而是理解 `ORB local_score / orb_inliers` 与人工黑色花纹判断的一致性边界。",
        "",
        "## Inputs",
        "",
        f"- `session_name`: `{session_name}`",
        f"- `pair_judgments_path`: `{_path_ref(repo_root, pair_judgments_path)}`",
        f"- `review_dir`: `{_path_ref(repo_root, review_dir)}`",
        f"- `view_manifest_path`: `{_path_ref(repo_root, view_manifest_path)}`",
        "",
        "## Label Summary",
        "",
        dataframe_to_markdown_table(label_summary_df),
        "",
        "## ORB Threshold Snapshot",
        "",
        dataframe_to_markdown_table(threshold_summary_df),
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


def run_texas_orb_qualitative_review(
    *,
    repo_root: Path,
    predictions_path: Path,
    review_dir: Path,
    pair_judgments_path: Path,
    output_dir: Path,
    view_manifest_path: Path = DEFAULT_TEXAS_VIEW_MANIFEST_PATH,
    top_k_per_category: int = DEFAULT_TOP_K_PER_CATEGORY,
    nfeatures: int = 2048,
    max_side: int = 768,
    fast_threshold: int = 12,
    clahe_clip_limit: float = 2.0,
) -> dict[str, Path]:
    resolved_predictions_path = resolve_predictions_path(repo_root=repo_root, value=predictions_path)
    resolved_review_dir = (review_dir if review_dir.is_absolute() else (repo_root / review_dir)).resolve()
    resolved_pair_judgments_path = (
        pair_judgments_path if pair_judgments_path.is_absolute() else (repo_root / pair_judgments_path)
    ).resolve()
    resolved_output_dir = (output_dir if output_dir.is_absolute() else (repo_root / output_dir)).resolve()
    resolved_view_manifest_path = (
        view_manifest_path if view_manifest_path.is_absolute() else (repo_root / view_manifest_path)
    ).resolve()

    tables_dir = resolved_output_dir / "tables"
    figures_dir = resolved_output_dir / "figures"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    judged_pair_df, session_name = load_texas_judged_pair_df(
        repo_root=repo_root,
        pair_judgments_path=resolved_pair_judgments_path,
        review_dir=resolved_review_dir,
    )
    label_summary_df, threshold_summary_df = summarize_judged_pairs(judged_pair_df=judged_pair_df)
    judged_pair_df.to_csv(tables_dir / "judged_pairs_enriched_v1.csv", index=False)
    label_summary_df.to_csv(tables_dir / "label_summary_v1.csv", index=False)
    threshold_summary_df.to_csv(tables_dir / "threshold_snapshot_v1.csv", index=False)

    reference_df = load_texas_reference_df(resolved_predictions_path)
    view_df = load_aligned_texas_view_df(
        repo_root=repo_root,
        reference_df=reference_df,
        manifest_path=resolved_view_manifest_path,
    )
    features = extract_local_features(
        df=view_df,
        repo_root=repo_root,
        nfeatures=int(nfeatures),
        max_side=int(max_side),
        fast_threshold=int(fast_threshold),
        clahe_clip_limit=float(clahe_clip_limit),
        local_matcher="orb",
    )
    feature_by_image = {str(feature.image_id): feature for feature in features}
    path_by_image = {str(row.image_id): str(getattr(row, PATH_COLUMN)) for row in view_df.itertuples(index=False)}

    category_outputs: list[dict[str, object]] = []
    for spec in CATEGORY_SPECS:
        subset = select_category_rows(judged_pair_df=judged_pair_df, spec=spec, top_k=int(top_k_per_category))
        category_dir = figures_dir / spec.name
        embedded_images: list[str] = []
        if not subset.empty:
            preview_columns = [
                "image_id",
                "neighbor_image_id",
                "candidate_type",
                "candidate_key",
                "label",
                "local_score",
                "miew_local_score",
                "orb_inliers",
                "orb_good_matches",
                "xgb_same_identity_prob",
            ]
            subset.loc[:, [column for column in preview_columns if column in subset.columns]].to_csv(
                tables_dir / f"{spec.name}_pairs_v1.csv",
                index=False,
            )
            for rank, row in enumerate(subset.itertuples(index=False), start=1):
                output_path = category_dir / f"{rank:02d}_{row.image_id}_{row.neighbor_image_id}.jpg"
                build_pair_board(
                    row=pd.Series(row._asdict()),
                    category_title=spec.title,
                    feature_by_image=feature_by_image,
                    path_by_image=path_by_image,
                    repo_root=repo_root,
                    output_path=output_path,
                )
                if rank <= 3:
                    embedded_images.append(_path_ref(reports_dir, output_path))
            table_preview = subset.loc[:, [column for column in preview_columns if column in subset.columns]].head(8).copy()
        else:
            table_preview = pd.DataFrame(columns=["image_id", "neighbor_image_id", "label", "local_score"])
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
        "mean_orb_yes": round(
            float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("yes"), "local_score"].astype(float).mean()),
            6,
        ),
        "mean_orb_no": round(
            float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("no"), "local_score"].astype(float).mean()),
            6,
        ),
        "mean_orb_inliers_yes": round(
            float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("yes"), "orb_inliers"].astype(float).mean()),
            6,
        ),
        "mean_orb_inliers_no": round(
            float(judged_pair_df.loc[judged_pair_df["label"].astype(str).eq("no"), "orb_inliers"].astype(float).mean()),
            6,
        ),
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_markdown(
        output_path=reports_dir / "summary.md",
        repo_root=repo_root,
        session_name=session_name,
        pair_judgments_path=resolved_pair_judgments_path,
        review_dir=resolved_review_dir,
        view_manifest_path=resolved_view_manifest_path,
        label_summary_df=label_summary_df,
        threshold_summary_df=threshold_summary_df,
        category_outputs=category_outputs,
    )
    return {
        "summary_path": reports_dir / "summary.md",
        "judged_pairs_path": tables_dir / "judged_pairs_enriched_v1.csv",
        "label_summary_path": tables_dir / "label_summary_v1.csv",
        "threshold_summary_path": tables_dir / "threshold_snapshot_v1.csv",
    }
