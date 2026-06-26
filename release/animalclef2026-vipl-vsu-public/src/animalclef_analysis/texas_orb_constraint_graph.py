from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .descriptor_baselines import dataframe_to_markdown_table
from .manual_cluster_overlay import apply_manual_cluster_overlay, load_manual_overlay_spec
from .manual_constraint_graph_compiler import compile_constraint_graph_to_overlay
from .manual_review_workbench import PAIR_LABEL_NO, PAIR_LABEL_YES, export_operations_spec, export_pair_judgments, load_pair_judgments
from .texas_orb_local_probe import TEXAS_DATASET, resolve_predictions_path
from .texas_unsupervised import summarize_cluster_labels


DEFAULT_REVIEW_DIR = Path("artifacts/analysis/texas_selftrain_review_orb_v1")
DEFAULT_PAIR_JUDGMENTS_PATH = Path("artifacts/analysis/manual_review_sessions/autosave/manual_pair_review_v1.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/texas_orb_constraint_graph_v1")
DEFAULT_MAX_ORB_LOCAL_SCORE = 0.4
DEFAULT_MAX_ORB_INLIERS = 4
DEFAULT_ORB_NEGATIVE_MODE = "both"
SUPPORTED_ORB_NEGATIVE_MODES = {"both", "either"}


def _path_ref(base: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base.resolve()).replace("\\", "/")


def _canonical_pair_key(left_image_id: object, right_image_id: object) -> str:
    left = str(left_image_id)
    right = str(right_image_id)
    return f"{left}|{right}" if left <= right else f"{right}|{left}"


def _resolve_texas_similarity_threshold(pred_df: pd.DataFrame) -> float:
    if "chosen_threshold" in pred_df.columns:
        distance_threshold = float(pd.to_numeric(pred_df["chosen_threshold"], errors="coerce").dropna().iloc[0])
        return round(1.0 - distance_threshold, 6)
    if "threshold" in pred_df.columns:
        distance_threshold = float(pd.to_numeric(pred_df["threshold"], errors="coerce").dropna().iloc[0])
        return round(1.0 - distance_threshold, 6)
    return 0.56


def _column_or_default(frame: pd.DataFrame, column_name: str, default_value: float | int) -> pd.Series:
    if column_name in frame.columns:
        return frame[column_name]
    return pd.Series(default_value, index=frame.index)


def select_texas_orb_auto_no_pairs(
    pair_df: pd.DataFrame,
    *,
    max_orb_local_score: float = DEFAULT_MAX_ORB_LOCAL_SCORE,
    max_orb_inliers: int = DEFAULT_MAX_ORB_INLIERS,
    mode: str = DEFAULT_ORB_NEGATIVE_MODE,
    min_auto_pairs_per_cluster: int = 1,
    exclude_pair_keys: set[str] | None = None,
) -> pd.DataFrame:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in SUPPORTED_ORB_NEGATIVE_MODES:
        raise ValueError(f"Unsupported ORB negative mode: {mode}. Expected one of {sorted(SUPPORTED_ORB_NEGATIVE_MODES)}")

    frame = pair_df.copy().reset_index(drop=True)
    if frame.empty:
        return frame
    frame["image_id"] = frame["image_id"].astype(str)
    frame["neighbor_image_id"] = frame["neighbor_image_id"].astype(str)
    frame["base_cluster_left"] = pd.to_numeric(frame["base_cluster_left"], errors="coerce").fillna(-1).astype(int)
    frame["base_cluster_right"] = pd.to_numeric(frame["base_cluster_right"], errors="coerce").fillna(-1).astype(int)
    frame["orb_local_score"] = pd.to_numeric(
        frame["orb_local_score"] if "orb_local_score" in frame.columns else _column_or_default(frame, "local_score", 0.0),
        errors="coerce",
    ).fillna(0.0)
    frame["orb_inliers"] = pd.to_numeric(_column_or_default(frame, "orb_inliers", 0), errors="coerce").fillna(0).astype(int)
    frame["orb_good_matches"] = pd.to_numeric(_column_or_default(frame, "orb_good_matches", 0), errors="coerce").fillna(0).astype(int)
    frame["xgb_same_identity_prob"] = pd.to_numeric(
        _column_or_default(frame, "xgb_same_identity_prob", 0.0),
        errors="coerce",
    ).fillna(0.0)
    frame["ambiguity_score"] = pd.to_numeric(_column_or_default(frame, "ambiguity_score", 0.0), errors="coerce").fillna(0.0)
    frame["pair_key_canonical"] = [
        _canonical_pair_key(left, right)
        for left, right in zip(frame["image_id"], frame["neighbor_image_id"], strict=True)
    ]
    same_cluster = frame["base_cluster_left"].eq(frame["base_cluster_right"])
    low_score = frame["orb_local_score"].le(float(max_orb_local_score))
    low_inliers = frame["orb_inliers"].le(int(max_orb_inliers))
    if resolved_mode == "both":
        veto_mask = same_cluster & low_score & low_inliers
    else:
        veto_mask = same_cluster & (low_score | low_inliers)
    if exclude_pair_keys:
        veto_mask &= ~frame["pair_key_canonical"].isin({str(value) for value in exclude_pair_keys})
    result = frame[veto_mask].copy()
    if result.empty:
        return result
    if int(min_auto_pairs_per_cluster) > 1:
        cluster_sizes = result.groupby("base_cluster_left").size()
        keep_clusters = cluster_sizes[cluster_sizes.ge(int(min_auto_pairs_per_cluster))].index.astype(int).tolist()
        result = result[result["base_cluster_left"].isin(keep_clusters)].copy()
    result["auto_rule_name"] = "texas_orb_auto_no_v1"
    result["auto_rule_detail"] = (
        f"same_cluster & orb_local<={float(max_orb_local_score):.3f}"
        + (" & " if resolved_mode == "both" else " | ")
        + f"orb_inliers<={int(max_orb_inliers)}"
    )
    return result.sort_values(
        ["base_cluster_left", "orb_local_score", "orb_inliers", "xgb_same_identity_prob", "image_id", "neighbor_image_id"],
        ascending=[True, True, True, True, True, True],
    ).reset_index(drop=True)


def auto_pairs_to_judgments(auto_pair_df: pd.DataFrame) -> list[dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    if auto_pair_df.empty:
        return judgments
    for row in auto_pair_df.itertuples(index=False):
        pair_key = _canonical_pair_key(row.image_id, row.neighbor_image_id)
        judgments.append(
            {
                "judgment_id": f"texas_orb_auto_split_{int(row.base_cluster_left)}_{str(row.image_id)}_{str(row.neighbor_image_id)}",
                "dataset": TEXAS_DATASET,
                "candidate_type": "split",
                "candidate_key": str(int(row.base_cluster_left)),
                "pair_key": pair_key,
                "image_id": str(row.image_id),
                "neighbor_image_id": str(row.neighbor_image_id),
                "base_cluster_left": int(row.base_cluster_left),
                "base_cluster_right": int(row.base_cluster_right),
                "xgb_same_identity_prob": round(float(row.xgb_same_identity_prob), 6),
                "ambiguity_score": round(float(row.ambiguity_score), 6),
                "label": PAIR_LABEL_NO,
                "note": (
                    f"auto_orb_no | orb_local_score={float(row.orb_local_score):.6f}"
                    f" | orb_inliers={int(row.orb_inliers)}"
                    f" | orb_good_matches={int(row.orb_good_matches)}"
                ),
            }
        )
    return judgments


def merge_manual_and_auto_texas_judgments(
    manual_judgments: list[dict[str, Any]],
    auto_judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = list(manual_judgments)
    manual_keys = {
        (
            str(item.get("dataset", "")),
            str(item.get("candidate_type", "")),
            str(item.get("candidate_key", "")),
            _canonical_pair_key(item.get("image_id", ""), item.get("neighbor_image_id", "")),
        )
        for item in manual_judgments
        if str(item.get("dataset", "")) == TEXAS_DATASET and str(item.get("candidate_type", "")) == "split"
    }
    for item in auto_judgments:
        key = (
            str(item.get("dataset", "")),
            str(item.get("candidate_type", "")),
            str(item.get("candidate_key", "")),
            _canonical_pair_key(item.get("image_id", ""), item.get("neighbor_image_id", "")),
        )
        if key in manual_keys:
            continue
        result.append(item)
    return result


def _filter_manual_yes_pair_keys(manual_judgments: list[dict[str, Any]]) -> set[str]:
    return {
        _canonical_pair_key(item.get("image_id", ""), item.get("neighbor_image_id", ""))
        for item in manual_judgments
        if str(item.get("dataset", "")) == TEXAS_DATASET and str(item.get("label", "")) == PAIR_LABEL_YES
    }


def _cluster_summary_row(pred_df: pd.DataFrame, *, variant_name: str) -> dict[str, Any]:
    labels = pred_df["pred_cluster_id"].to_numpy(dtype=int)
    summary = summarize_cluster_labels(labels)
    summary["variant"] = str(variant_name)
    summary["samples"] = int(len(pred_df))
    return summary


def _render_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_Empty table._"
    try:
        return frame.to_markdown(index=False)
    except (ImportError, ModuleNotFoundError):
        return "```text\n" + frame.to_string(index=False) + "\n```"


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


def _resolve_row_path(row: pd.Series | dict[str, Any]) -> str:
    if isinstance(row, pd.Series):
        payload = row.to_dict()
    else:
        payload = row
    for column in ["path", "recommended_model_input_path_v1", "preferred_path_v1"]:
        value = str(payload.get(column, "")).strip()
        if value:
            return value
    raise ValueError(f"Could not resolve image path from row keys: {sorted(payload.keys())}")


def _build_cluster_member_lookup(pred_df: pd.DataFrame) -> dict[int, list[dict[str, str]]]:
    payload: dict[int, list[dict[str, str]]] = {}
    for cluster_id, cluster_df in pred_df.groupby("pred_cluster_id", sort=True):
        members = (
            cluster_df[["image_id", "path"]]
            .astype({"image_id": str, "path": str})
            .sort_values("image_id")
            .to_dict(orient="records")
        )
        payload[int(cluster_id)] = [
            {
                "image_id": str(item["image_id"]),
                "path": str(item["path"]),
            }
            for item in members
        ]
    return payload


def _manual_pair_counts_by_cluster(manual_judgments: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    payload: dict[int, dict[str, int]] = {}
    for item in manual_judgments:
        if str(item.get("dataset", "")) != TEXAS_DATASET or str(item.get("candidate_type", "")) != "split":
            continue
        try:
            cluster_id = int(str(item.get("candidate_key", "")))
        except ValueError:
            continue
        entry = payload.setdefault(cluster_id, {"manual_yes_pairs": 0, "manual_no_pairs": 0})
        label = str(item.get("label", "")).strip().lower()
        if label == PAIR_LABEL_YES:
            entry["manual_yes_pairs"] += 1
        elif label == PAIR_LABEL_NO:
            entry["manual_no_pairs"] += 1
    return payload


def _changed_members_by_cluster(changed_df: pd.DataFrame) -> dict[int, list[str]]:
    if changed_df.empty:
        return {}
    frame = changed_df.copy()
    if "overlay_base_pred_cluster_id" not in frame.columns:
        return {}
    frame["overlay_base_pred_cluster_id"] = pd.to_numeric(frame["overlay_base_pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    frame["image_id"] = frame["image_id"].astype(str)
    payload: dict[int, list[str]] = {}
    for cluster_id, cluster_df in frame.groupby("overlay_base_pred_cluster_id", sort=True):
        payload[int(cluster_id)] = sorted(cluster_df["image_id"].astype(str).unique().tolist())
    return payload


def _build_auto_review_index(
    *,
    pred_df: pd.DataFrame,
    auto_pair_df: pd.DataFrame,
    manual_judgments: list[dict[str, Any]],
    orb_auto_changed_df: pd.DataFrame,
    manual_plus_orb_changed_df: pd.DataFrame,
    reports_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    if auto_pair_df.empty:
        return pd.DataFrame()
    cluster_members = _build_cluster_member_lookup(pred_df)
    manual_counts = _manual_pair_counts_by_cluster(manual_judgments)
    orb_auto_changed = _changed_members_by_cluster(orb_auto_changed_df)
    combo_changed = _changed_members_by_cluster(manual_plus_orb_changed_df)

    rows: list[dict[str, Any]] = []
    for review_rank, row in enumerate(
        auto_pair_df.sort_values(
            ["orb_local_score", "orb_inliers", "xgb_same_identity_prob", "base_cluster_left"],
            ascending=[True, True, False, True],
        ).itertuples(index=False),
        start=1,
    ):
        cluster_id = int(row.base_cluster_left)
        member_payload = cluster_members.get(cluster_id, [])
        member_ids = [str(item["image_id"]) for item in member_payload]
        other_members = [value for value in member_ids if value not in {str(row.image_id), str(row.neighbor_image_id)}]
        manual_payload = manual_counts.get(cluster_id, {"manual_yes_pairs": 0, "manual_no_pairs": 0})
        figure_path = figures_dir / f"review_{review_rank:02d}_cluster_{cluster_id}_{row.image_id}_{row.neighbor_image_id}.jpg"
        rows.append(
            {
                "review_rank": int(review_rank),
                "base_cluster_id": int(cluster_id),
                "base_cluster_size": int(len(member_ids)),
                "image_id": str(row.image_id),
                "neighbor_image_id": str(row.neighbor_image_id),
                "cluster_image_ids": "|".join(member_ids),
                "other_members": "|".join(other_members),
                "orb_local_score": round(float(row.orb_local_score), 6),
                "orb_inliers": int(row.orb_inliers),
                "orb_good_matches": int(row.orb_good_matches),
                "xgb_same_identity_prob": round(float(row.xgb_same_identity_prob), 6),
                "miew_local_score": round(float(row.miew_local_score), 6) if "miew_local_score" in auto_pair_df.columns else 0.0,
                "route_global_score": round(float(row.route_global_score), 6) if "route_global_score" in auto_pair_df.columns else 0.0,
                "ambiguity_score": round(float(row.ambiguity_score), 6),
                "manual_yes_pairs_in_cluster": int(manual_payload["manual_yes_pairs"]),
                "manual_no_pairs_in_cluster": int(manual_payload["manual_no_pairs"]),
                "orb_auto_only_moved_image_ids": "|".join(orb_auto_changed.get(cluster_id, [])),
                "manual_plus_orb_moved_image_ids": "|".join(combo_changed.get(cluster_id, [])),
                "figure_path": _path_ref(reports_dir, figure_path),
            }
        )
    return pd.DataFrame(rows).sort_values("review_rank").reset_index(drop=True)


def _thumb_with_border(image: Image.Image, *, width: int, height: int, border_color: tuple[int, int, int]) -> Image.Image:
    border = 4
    canvas = Image.new("RGB", (int(width), int(height)), border_color)
    inner = _resize_with_pad(image, width=int(width) - border * 2, height=int(height) - border * 2)
    canvas.paste(inner, (border, border))
    return canvas


def _build_auto_review_board(
    *,
    repo_root: Path,
    review_row: pd.Series,
    cluster_df: pd.DataFrame,
    output_path: Path,
) -> None:
    margin = 18
    gap = 12
    pair_w = 320
    pair_h = 240
    thumb_w = 170
    thumb_h = 152
    title_font = _font(22)
    body_font = _font(15)
    small_font = _font(13)

    left_rgb = _load_rgb(repo_root=repo_root, image_path=str(cluster_df[cluster_df["image_id"].astype(str).eq(str(review_row["image_id"]))].iloc[0]["path"]))
    right_rgb = _load_rgb(repo_root=repo_root, image_path=str(cluster_df[cluster_df["image_id"].astype(str).eq(str(review_row["neighbor_image_id"]))].iloc[0]["path"]))

    members = cluster_df.sort_values("image_id").reset_index(drop=True)
    context_columns = min(4, max(2, len(members)))
    context_rows = (len(members) + context_columns - 1) // context_columns
    header_lines = [
        f"cluster={int(review_row['base_cluster_id'])} | size={int(review_row['base_cluster_size'])} | review_rank={int(review_row['review_rank'])}",
        (
            f"pair={review_row['image_id']} vs {review_row['neighbor_image_id']} | "
            f"orb_local={float(review_row['orb_local_score']):.3f} | "
            f"orb_inliers={int(review_row['orb_inliers'])} | "
            f"orb_good={int(review_row['orb_good_matches'])}"
        ),
        (
            f"xgb_prob={float(review_row['xgb_same_identity_prob']):.3f} | "
            f"miew_local={float(review_row['miew_local_score']):.3f} | "
            f"route_global={float(review_row['route_global_score']):.3f} | "
            f"ambiguity={float(review_row['ambiguity_score']):.3f}"
        ),
        (
            f"manual_cluster_yes={int(review_row['manual_yes_pairs_in_cluster'])} | "
            f"manual_cluster_no={int(review_row['manual_no_pairs_in_cluster'])} | "
            f"orb_auto_move={str(review_row['orb_auto_only_moved_image_ids']) or '-'} | "
            f"manual_plus_orb_move={str(review_row['manual_plus_orb_moved_image_ids']) or '-'}"
        ),
    ]
    header_h = 42 + len(header_lines) * 22
    context_label_h = 34
    width = max(
        margin * 2 + pair_w * 2 + gap,
        margin * 2 + context_columns * thumb_w + (context_columns - 1) * gap,
    )
    height = margin * 2 + header_h + pair_h + gap + context_rows * (thumb_h + context_label_h) + max(0, context_rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (247, 247, 247))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), "Texas ORB Auto Split Review", fill=(18, 18, 18), font=title_font)
    for line_index, line in enumerate(header_lines):
        draw.text((margin, margin + 34 + line_index * 22), line, fill=(48, 48, 48), font=body_font)

    pair_y = margin + header_h
    canvas.paste(_thumb_with_border(left_rgb, width=pair_w, height=pair_h, border_color=(230, 132, 28)), (margin, pair_y))
    canvas.paste(
        _thumb_with_border(right_rgb, width=pair_w, height=pair_h, border_color=(214, 74, 76)),
        (margin + pair_w + gap, pair_y),
    )
    draw.text((margin, pair_y + pair_h - 22), f"{review_row['image_id']} | auto-pair", fill=(20, 20, 20), font=body_font)
    draw.text(
        (margin + pair_w + gap, pair_y + pair_h - 22),
        f"{review_row['neighbor_image_id']} | auto-pair",
        fill=(20, 20, 20),
        font=body_font,
    )

    orb_auto_moved = {value for value in str(review_row["orb_auto_only_moved_image_ids"]).split("|") if value}
    combo_moved = {value for value in str(review_row["manual_plus_orb_moved_image_ids"]).split("|") if value}
    pair_members = {str(review_row["image_id"]), str(review_row["neighbor_image_id"])}

    context_y = pair_y + pair_h + gap
    for member_index, member_row in enumerate(members.itertuples(index=False)):
        grid_x = member_index % context_columns
        grid_y = member_index // context_columns
        x = margin + grid_x * (thumb_w + gap)
        y = context_y + grid_y * (thumb_h + context_label_h + gap)
        image_id = str(member_row.image_id)
        if image_id in combo_moved:
            border_color = (94, 92, 230)
        elif image_id in orb_auto_moved:
            border_color = (33, 150, 243)
        elif image_id in pair_members:
            border_color = (230, 132, 28)
        else:
            border_color = (180, 180, 180)
        member_rgb = _load_rgb(repo_root=repo_root, image_path=str(_resolve_row_path({"path": member_row.path})))
        canvas.paste(_thumb_with_border(member_rgb, width=thumb_w, height=thumb_h, border_color=border_color), (x, y))
        tags: list[str] = []
        if image_id in pair_members:
            tags.append("pair")
        if image_id in orb_auto_moved:
            tags.append("orb_move")
        if image_id in combo_moved:
            tags.append("combo_move")
        label = image_id if not tags else f"{image_id} | {','.join(tags)}"
        draw.text((x, y + thumb_h + 4), label, fill=(36, 36, 36), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def _write_human_review_outputs(
    *,
    repo_root: Path,
    output_dir: Path,
    pred_df: pd.DataFrame,
    auto_pair_df: pd.DataFrame,
    manual_judgments: list[dict[str, Any]],
    resolved_base_predictions_path: Path,
    resolved_review_dir: Path,
    resolved_pair_judgments_path: Path,
    variant_summary_df: pd.DataFrame,
    resolved_graph_threshold: float,
    max_orb_local_score: float,
    max_orb_inliers: int,
    orb_negative_mode: str,
    session_name: str,
    variant_outputs_by_name: dict[str, dict[str, Path]],
) -> dict[str, Path]:
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures" / "auto_split_review"
    review_pack_dir = output_dir / "review_pack"
    review_pack_tables_dir = review_pack_dir / "tables"
    review_pack_reports_dir = review_pack_dir / "reports"
    for path in [figures_dir, review_pack_dir, review_pack_tables_dir, review_pack_reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    orb_auto_changed_df = pd.read_csv(variant_outputs_by_name["orb_auto_only"]["changed_path"])
    manual_plus_orb_changed_df = pd.read_csv(variant_outputs_by_name["manual_plus_orb"]["changed_path"])
    review_index_df = _build_auto_review_index(
        pred_df=pred_df,
        auto_pair_df=auto_pair_df,
        manual_judgments=manual_judgments,
        orb_auto_changed_df=orb_auto_changed_df,
        manual_plus_orb_changed_df=manual_plus_orb_changed_df,
        reports_dir=reports_dir,
        figures_dir=figures_dir,
    )
    review_index_path = tables_dir / "auto_pair_review_index_v1.csv"
    review_index_df.to_csv(review_index_path, index=False)

    if not review_index_df.empty:
        for review_row in review_index_df.itertuples(index=False):
            cluster_df = pred_df[pred_df["pred_cluster_id"].astype(int).eq(int(review_row.base_cluster_id))].copy().reset_index(drop=True)
            figure_path = reports_dir / str(review_row.figure_path)
            _build_auto_review_board(
                repo_root=repo_root,
                review_row=pd.Series(review_row._asdict()),
                cluster_df=cluster_df,
                output_path=figure_path,
            )

    split_rows: list[dict[str, Any]] = []
    for review_row in review_index_df.itertuples(index=False):
        split_rows.append(
            {
                "base_cluster_id": int(review_row.base_cluster_id),
                "base_cluster_size": int(review_row.base_cluster_size),
                "ambiguous_image_count": int(review_row.base_cluster_size),
                "ambiguous_pair_count": 1,
                "max_split_votes": 1,
                "mean_pair_probability": round(float(review_row.xgb_same_identity_prob), 6),
                "max_pair_probability": round(float(review_row.xgb_same_identity_prob), 6),
                "mean_ambiguity_score": round(float(review_row.ambiguity_score), 6),
                "max_ambiguity_score": round(float(review_row.ambiguity_score), 6),
                "mean_border_score": 0.0,
                "max_conflict_ratio": 0.0,
                "conflict_methods": "texas_orb_auto_no_v1",
                "component_ids": "",
                "image_indices": "",
                "image_ids": str(review_row.cluster_image_ids),
            }
        )
    review_pack_pair_path = review_pack_tables_dir / "test_pair_disagreement_v1.csv"
    review_pack_split_path = review_pack_tables_dir / "test_split_candidates_v1.csv"
    auto_pair_df.to_csv(review_pack_pair_path, index=False)
    pd.DataFrame(split_rows).to_csv(review_pack_split_path, index=False)

    review_command = (
        "source /home/hechen/miniconda3/etc/profile.d/conda.sh && "
        "conda activate wildfusion && "
        f"python scripts/launch_manual_review_workbench.py --base-predictions {str(_path_ref(repo_root, resolved_base_predictions_path))} "
        f"--probe-dir {str(_path_ref(repo_root, review_pack_dir))} --port 7862"
    )
    preview_columns = [
        "review_rank",
        "base_cluster_id",
        "base_cluster_size",
        "image_id",
        "neighbor_image_id",
        "orb_local_score",
        "orb_inliers",
        "xgb_same_identity_prob",
        "miew_local_score",
        "manual_no_pairs_in_cluster",
        "orb_auto_only_moved_image_ids",
        "manual_plus_orb_moved_image_ids",
    ]
    summary_lines = [
        "# Texas ORB Constraint Graph Probe v1",
        "",
        "这页是人工复核入口，不是机器日志。",
        "",
        "## 先看什么",
        "",
        "- 先按 `review_rank` 顺序看下面 6 个 board，它们就是当前默认规则自动切出来的全部 Texas ORB `cannot-link` 候选。",
        "- 如果你想继续在 UI 里逐对审差，直接用下面的 mini review pack 启动 workbench。",
        "- `manual_plus_orb_moved_image_ids` 比 `orb_auto_only_moved_image_ids` 更多时，说明这条 auto 负证据和你已有人工 judgment 发生了叠加效应。",
        "",
        "## Inputs",
        "",
        f"- `base_predictions_path`: `{_path_ref(repo_root, resolved_base_predictions_path)}`",
        f"- `review_dir`: `{_path_ref(repo_root, resolved_review_dir)}`",
        f"- `pair_judgments_path`: `{_path_ref(repo_root, resolved_pair_judgments_path)}`",
        f"- `manual_session_name`: `{session_name}`",
        "",
        "## Rule",
        "",
        f"- `graph_threshold`: `{float(resolved_graph_threshold):.6f}`",
        f"- `orb_negative_mode`: `{str(orb_negative_mode)}`",
        f"- `max_orb_local_score`: `{float(max_orb_local_score):.3f}`",
        f"- `max_orb_inliers`: `{int(max_orb_inliers)}`",
        f"- `auto_pair_count`: `{int(len(auto_pair_df))}`",
        f"- `auto_cluster_count`: `{int(auto_pair_df['base_cluster_left'].nunique()) if not auto_pair_df.empty else 0}`",
        "",
        "## Variant Summary",
        "",
        dataframe_to_markdown_table(variant_summary_df) if not variant_summary_df.empty else "_No variant rows generated._",
        "",
        "## Review Index",
        "",
        dataframe_to_markdown_table(review_index_df[preview_columns]) if not review_index_df.empty else "_No auto ORB split pairs._",
        "",
        "## Workbench",
        "",
        f"- `review_pack`: `{_path_ref(repo_root, review_pack_dir)}`",
        f"- `launch_command`: `{review_command}`",
        "",
    ]
    for review_row in review_index_df.itertuples(index=False):
        summary_lines.extend(
            [
                f"## Review {int(review_row.review_rank)} | Cluster {int(review_row.base_cluster_id)}",
                "",
                f"- pair: `{review_row.image_id}|{review_row.neighbor_image_id}`",
                f"- cluster_size: `{int(review_row.base_cluster_size)}` | other_members: `{str(review_row.other_members) or '-'}`",
                f"- orb_local: `{float(review_row.orb_local_score):.3f}` | orb_inliers: `{int(review_row.orb_inliers)}` | xgb_prob: `{float(review_row.xgb_same_identity_prob):.3f}` | miew_local: `{float(review_row.miew_local_score):.3f}`",
                f"- manual_cluster_yes/no: `{int(review_row.manual_yes_pairs_in_cluster)}/{int(review_row.manual_no_pairs_in_cluster)}`",
                f"- orb_auto_only_moved: `{str(review_row.orb_auto_only_moved_image_ids) or '-'}` | manual_plus_orb_moved: `{str(review_row.manual_plus_orb_moved_image_ids) or '-'}`",
                "",
                f"![review_{int(review_row.review_rank)}]({str(review_row.figure_path)})",
                "",
            ]
        )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    review_pack_summary_lines = [
        "# Texas ORB Auto Split Review Pack v1",
        "",
        "- 这个 mini pack 只保留当前自动命中的 6 个 ORB low-score same-cluster pair。",
        f"- `base_predictions_path`: `{_path_ref(repo_root, resolved_base_predictions_path)}`",
        f"- `pair_table`: `{_path_ref(repo_root, review_pack_pair_path)}`",
        f"- `split_candidates`: `{_path_ref(repo_root, review_pack_split_path)}`",
        "",
        "## Launch",
        "",
        f"- `{review_command}`",
        "",
    ]
    (review_pack_reports_dir / "summary.md").write_text("\n".join(review_pack_summary_lines) + "\n", encoding="utf-8")
    return {
        "summary_path": summary_path,
        "review_index_path": review_index_path,
        "review_pack_dir": review_pack_dir,
        "review_pack_summary_path": review_pack_reports_dir / "summary.md",
        "review_pack_pair_path": review_pack_pair_path,
        "review_pack_split_path": review_pack_split_path,
    }


def _variant_output_dir(output_dir: Path, variant_name: str) -> Path:
    return output_dir / "variants" / variant_name


def _write_variant_outputs(
    *,
    repo_root: Path,
    output_dir: Path,
    variant_name: str,
    pred_df: pd.DataFrame,
    operations: list[dict[str, Any]],
    candidate_summary_df: pd.DataFrame,
    component_summary_df: pd.DataFrame,
    edge_summary_df: pd.DataFrame,
) -> dict[str, Path]:
    variant_dir = _variant_output_dir(output_dir, variant_name)
    tables_dir = variant_dir / "tables"
    reports_dir = variant_dir / "reports"
    for path in [variant_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    spec_path = variant_dir / "compiled_overlay_spec.json"
    export_operations_spec(rule_name=f"texas_orb_constraint_graph_{variant_name}", operations=operations, output_path=spec_path)
    overlaid_pred_df, changed_df, operation_df = apply_manual_cluster_overlay(
        pred_df=pred_df,
        spec=load_manual_overlay_spec(spec_path),
    ) if operations else (pred_df.copy().reset_index(drop=True), pd.DataFrame(), pd.DataFrame())

    prediction_path = tables_dir / "test_predictions_v1.csv"
    changed_path = tables_dir / "changed_rows_v1.csv"
    operation_summary_path = tables_dir / "compiled_operations_v1.csv"
    candidate_summary_path = tables_dir / "candidate_summary_v1.csv"
    component_summary_path = tables_dir / "component_summary_v1.csv"
    edge_summary_path = tables_dir / "edge_summary_v1.csv"
    overlaid_pred_df.to_csv(prediction_path, index=False)
    changed_df.to_csv(changed_path, index=False)
    operation_df.to_csv(operation_summary_path, index=False)
    candidate_summary_df.to_csv(candidate_summary_path, index=False)
    component_summary_df.to_csv(component_summary_path, index=False)
    edge_summary_df.to_csv(edge_summary_path, index=False)

    base_summary = _cluster_summary_row(pred_df, variant_name="base")
    overlay_summary = _cluster_summary_row(overlaid_pred_df, variant_name=str(variant_name))
    summary_df = pd.DataFrame([base_summary, overlay_summary])
    summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    summary_lines = [
        f"# Texas ORB Constraint Graph Probe | {variant_name}",
        "",
        f"- `operations`: `{len(operations)}`",
        f"- `candidate_rows`: `{len(candidate_summary_df)}`",
        f"- `component_rows`: `{len(component_summary_df)}`",
        f"- `edge_rows`: `{len(edge_summary_df)}`",
        f"- `prediction_path`: `{_path_ref(repo_root, prediction_path)}`",
        "",
        "## Cluster Summary",
        "",
        _render_table(summary_df),
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "spec_path": spec_path,
        "prediction_path": prediction_path,
        "changed_path": changed_path,
        "candidate_summary_path": candidate_summary_path,
        "component_summary_path": component_summary_path,
        "edge_summary_path": edge_summary_path,
        "cluster_summary_path": tables_dir / "cluster_summary_v1.csv",
        "summary_path": reports_dir / "summary.md",
    }


def run_texas_orb_constraint_graph_probe(
    *,
    repo_root: Path,
    base_predictions_path: Path,
    review_dir: Path = DEFAULT_REVIEW_DIR,
    pair_judgments_path: Path = DEFAULT_PAIR_JUDGMENTS_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_orb_local_score: float = DEFAULT_MAX_ORB_LOCAL_SCORE,
    max_orb_inliers: int = DEFAULT_MAX_ORB_INLIERS,
    orb_negative_mode: str = DEFAULT_ORB_NEGATIVE_MODE,
    min_auto_pairs_per_cluster: int = 1,
    graph_threshold: float | None = None,
) -> dict[str, Path]:
    resolved_base_predictions_path = resolve_predictions_path(repo_root=repo_root, value=base_predictions_path)
    resolved_review_dir = (review_dir if review_dir.is_absolute() else (repo_root / review_dir)).resolve()
    resolved_pair_judgments_path = (
        pair_judgments_path if pair_judgments_path.is_absolute() else (repo_root / pair_judgments_path)
    ).resolve()
    resolved_output_dir = (output_dir if output_dir.is_absolute() else (repo_root / output_dir)).resolve()
    tables_dir = resolved_output_dir / "tables"
    reports_dir = resolved_output_dir / "reports"
    for path in [resolved_output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(resolved_base_predictions_path).copy()
    pred_df["dataset"] = pred_df["dataset"].astype(str)
    pred_df["image_id"] = pred_df["image_id"].astype(str)
    pred_df["pred_cluster_id"] = pd.to_numeric(pred_df["pred_cluster_id"], errors="coerce").fillna(-1).astype(int)
    pred_df = pred_df[pred_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if pred_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows found in {resolved_base_predictions_path}")

    pair_df = pd.read_csv(resolved_review_dir / "tables" / "test_pair_disagreement_v1.csv").copy()
    pair_df["dataset"] = pair_df["dataset"].astype(str)
    pair_df = pair_df[pair_df["dataset"].eq(TEXAS_DATASET)].copy().reset_index(drop=True)
    if pair_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} pair rows found in {resolved_review_dir}")

    session_name, manual_judgments = load_pair_judgments(resolved_pair_judgments_path)
    manual_yes_pair_keys = _filter_manual_yes_pair_keys(manual_judgments)
    auto_pair_df = select_texas_orb_auto_no_pairs(
        pair_df,
        max_orb_local_score=float(max_orb_local_score),
        max_orb_inliers=int(max_orb_inliers),
        mode=str(orb_negative_mode),
        min_auto_pairs_per_cluster=int(min_auto_pairs_per_cluster),
        exclude_pair_keys=manual_yes_pair_keys,
    )
    auto_pair_path = tables_dir / "auto_orb_no_pairs_v1.csv"
    auto_pair_df.to_csv(auto_pair_path, index=False)
    auto_judgments = auto_pairs_to_judgments(auto_pair_df)
    auto_judgments_path = export_pair_judgments(
        session_name="texas_orb_auto_no_v1",
        judgments=auto_judgments,
        output_path=tables_dir / "auto_pair_judgments_v1.json",
    )
    combined_judgments = merge_manual_and_auto_texas_judgments(manual_judgments=manual_judgments, auto_judgments=auto_judgments)
    combined_judgments_path = export_pair_judgments(
        session_name=f"{session_name}_plus_texas_orb_auto_no" if session_name else "texas_manual_plus_orb_auto_no_v1",
        judgments=combined_judgments,
        output_path=tables_dir / "combined_pair_judgments_v1.json",
    )

    resolved_graph_threshold = (
        float(graph_threshold)
        if graph_threshold is not None
        else _resolve_texas_similarity_threshold(pred_df=pred_df)
    )

    variant_rows: list[dict[str, Any]] = []
    variant_outputs_by_name: dict[str, dict[str, Path]] = {}
    outputs: dict[str, Path] = {
        "auto_pair_path": auto_pair_path,
        "auto_judgments_path": auto_judgments_path,
        "combined_judgments_path": combined_judgments_path,
    }
    variant_payloads = [
        ("manual_only", manual_judgments),
        ("orb_auto_only", auto_judgments),
        ("manual_plus_orb", combined_judgments),
    ]
    for variant_name, judgments in variant_payloads:
        operations, candidate_summary_df, component_summary_df, edge_summary_df = compile_constraint_graph_to_overlay(
            pred_df=pred_df,
            pair_df=pair_df,
            judgments=judgments,
            datasets=[TEXAS_DATASET],
            graph_threshold=float(resolved_graph_threshold),
            min_judged_pairs=1,
            min_no_pairs=1,
        )
        variant_outputs = _write_variant_outputs(
            repo_root=repo_root,
            output_dir=resolved_output_dir,
            variant_name=variant_name,
            pred_df=pred_df,
            operations=operations,
            candidate_summary_df=candidate_summary_df,
            component_summary_df=component_summary_df,
            edge_summary_df=edge_summary_df,
        )
        variant_outputs_by_name[variant_name] = variant_outputs
        outputs[f"{variant_name}_prediction_path"] = variant_outputs["prediction_path"]
        outputs[f"{variant_name}_summary_path"] = variant_outputs["summary_path"]
        cluster_summary_df = pd.read_csv(variant_outputs["cluster_summary_path"])
        overlay_row = cluster_summary_df[cluster_summary_df["variant"].astype(str).eq(variant_name)].iloc[0].to_dict()
        overlay_row["variant"] = variant_name
        overlay_row["operations"] = int(len(operations))
        overlay_row["candidate_rows"] = int(len(candidate_summary_df))
        overlay_row["component_rows"] = int(len(component_summary_df))
        overlay_row["edge_rows"] = int(len(edge_summary_df))
        variant_rows.append(overlay_row)

    variant_summary_df = pd.DataFrame(variant_rows).sort_values("variant").reset_index(drop=True)
    variant_summary_path = tables_dir / "variant_summary_v1.csv"
    variant_summary_df.to_csv(variant_summary_path, index=False)

    summary_payload = {
        "probe": resolved_output_dir.name,
        "base_predictions_path": _path_ref(repo_root, resolved_base_predictions_path),
        "review_dir": _path_ref(repo_root, resolved_review_dir),
        "pair_judgments_path": _path_ref(repo_root, resolved_pair_judgments_path),
        "auto_judgments_path": _path_ref(repo_root, auto_judgments_path),
        "combined_judgments_path": _path_ref(repo_root, combined_judgments_path),
        "graph_threshold": float(resolved_graph_threshold),
        "max_orb_local_score": float(max_orb_local_score),
        "max_orb_inliers": int(max_orb_inliers),
        "orb_negative_mode": str(orb_negative_mode),
        "min_auto_pairs_per_cluster": int(min_auto_pairs_per_cluster),
        "auto_pair_count": int(len(auto_pair_df)),
        "auto_cluster_count": int(auto_pair_df["base_cluster_left"].nunique()) if not auto_pair_df.empty else 0,
        "manual_session_name": session_name,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    human_review_outputs = _write_human_review_outputs(
        repo_root=repo_root,
        output_dir=resolved_output_dir,
        pred_df=pred_df,
        auto_pair_df=auto_pair_df,
        manual_judgments=manual_judgments,
        resolved_base_predictions_path=resolved_base_predictions_path,
        resolved_review_dir=resolved_review_dir,
        resolved_pair_judgments_path=resolved_pair_judgments_path,
        variant_summary_df=variant_summary_df,
        resolved_graph_threshold=float(resolved_graph_threshold),
        max_orb_local_score=float(max_orb_local_score),
        max_orb_inliers=int(max_orb_inliers),
        orb_negative_mode=str(orb_negative_mode),
        session_name=session_name,
        variant_outputs_by_name=variant_outputs_by_name,
    )
    outputs.update(human_review_outputs)
    outputs["variant_summary_path"] = variant_summary_path
    return outputs
