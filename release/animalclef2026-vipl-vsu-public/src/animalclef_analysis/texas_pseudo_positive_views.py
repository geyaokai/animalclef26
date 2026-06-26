from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


TEXAS_DATASET = "TexasHornedLizards"
DEFAULT_TEXAS_CENTER_BODY_MANIFEST = Path(
    "artifacts/manifests/texas_center_body_square_repaired_v1/tables/manifest_test_texas_center_body_square_gray_v1.csv"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/training/cache/texas_pseudo_positive_views_v1")

GROUP_COLUMN_CANDIDATES = [
    "trusted_group_id",
    "trusted_component_id",
    "component_id",
    "pseudo_identity",
    "trusted_identity",
    "group_id",
    "class_id",
]


def _markdown_table(df: pd.DataFrame, *, columns: list[str] | None = None, limit: int | None = None) -> str:
    if columns is not None:
        df = df.loc[:, columns].copy()
    if limit is not None:
        df = df.head(int(limit)).copy()
    if df.empty:
        return "_No rows._"
    header = "| " + " | ".join(df.columns.astype(str).tolist()) + " |"
    divider = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = [
        "| " + " | ".join("" if pd.isna(value) else str(value) for value in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _resolve_group_column(df: pd.DataFrame) -> str:
    for column in GROUP_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(
        "Trusted membership table is missing a group column. Expected one of: "
        + ", ".join(GROUP_COLUMN_CANDIDATES)
    )


def _resolve_source_path_column(df: pd.DataFrame) -> str:
    for column in ["recommended_model_input_path_v1", "preferred_path_v1", "path"]:
        if column in df.columns:
            return column
    raise ValueError("Manifest is missing a usable source-path column.")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


@dataclass(frozen=True)
class ViewRecipe:
    name: str
    family: str
    description: str
    payload: dict[str, Any]
    augmentor_hint: str
    gate_fn: Callable[[pd.Series], tuple[bool, str]]


def _always_enabled(_row: pd.Series) -> tuple[bool, str]:
    return True, "always_on"


def _flip_gate(row: pd.Series) -> tuple[bool, str]:
    foreground_ratio = float(row.get("texas_center_body_foreground_ratio_in_crop_v1", 0.0) or 0.0)
    axis_confidence = float(
        row.get("sam_trainprep_aligned_axis_confidence_v1", row.get("body_axis_unsigned_rgb_v1_axis_confidence", 0.0)) or 0.0
    )
    enabled = foreground_ratio >= 0.35 and axis_confidence >= 0.20
    reason = (
        f"foreground_ratio_in_crop={foreground_ratio:.3f}, axis_confidence={axis_confidence:.3f}"
        if enabled
        else f"gated_off: foreground_ratio_in_crop={foreground_ratio:.3f}, axis_confidence={axis_confidence:.3f}"
    )
    return enabled, reason


DEFAULT_VIEW_RECIPES: list[ViewRecipe] = [
    ViewRecipe(
        name="crop_jitter_tight_v1",
        family="crop",
        description="Keep the center-body crop but tighten the square slightly around the same center.",
        payload={
            "op": "crop_jitter",
            "crop_center_mode": "reuse_center_body_center",
            "side_scale": 0.96,
            "padding_mode": "edge",
        },
        augmentor_hint="Augmentor analogue: crop around the existing center region with a deterministic ~96% area.",
        gate_fn=_always_enabled,
    ),
    ViewRecipe(
        name="rotate_mild_pos5_v1",
        family="rotate",
        description="Apply a mild clockwise rotation while preserving the center-body framing.",
        payload={
            "op": "rotate",
            "degrees": 5.0,
            "resample": "bilinear",
            "fill_mode": "edge",
        },
        augmentor_hint="Augmentor analogue: Rotate with probability 1.0 and small right rotation (~5 degrees).",
        gate_fn=_always_enabled,
    ),
    ViewRecipe(
        name="rotate_mild_neg5_v1",
        family="rotate",
        description="Apply a mild counter-clockwise rotation while preserving the center-body framing.",
        payload={
            "op": "rotate",
            "degrees": -5.0,
            "resample": "bilinear",
            "fill_mode": "edge",
        },
        augmentor_hint="Augmentor analogue: Rotate with probability 1.0 and small left rotation (~5 degrees).",
        gate_fn=_always_enabled,
    ),
    ViewRecipe(
        name="scale_focus_in_v1",
        family="scale",
        description="Zoom slightly into the center-body crop to emphasize the dorsal black-dot field.",
        payload={
            "op": "scale",
            "scale_factor": 1.04,
            "anchor_mode": "center",
            "fill_mode": "edge",
        },
        augmentor_hint="Augmentor analogue: Zoom/scale slightly inward around center (~1.04x).",
        gate_fn=_always_enabled,
    ),
    ViewRecipe(
        name="horizontal_flip_gated_v1",
        family="flip",
        description="Mirror the center-body crop only when the subject fill and alignment quality are acceptable.",
        payload={
            "op": "horizontal_flip",
            "enabled_if": "foreground_ratio_in_crop>=0.35 and axis_confidence>=0.20",
        },
        augmentor_hint="Augmentor analogue: Flip left-right with probability 1.0 when gating passes.",
        gate_fn=_flip_gate,
    ),
]


def canonicalize_trusted_texas_membership(trusted_df: pd.DataFrame) -> pd.DataFrame:
    if "image_id" not in trusted_df.columns:
        raise ValueError("Trusted membership table must contain `image_id`.")
    group_column = _resolve_group_column(trusted_df)
    result = trusted_df.copy().reset_index(drop=True)
    if "dataset" not in result.columns:
        result["dataset"] = TEXAS_DATASET
    result["dataset"] = result["dataset"].fillna(TEXAS_DATASET).astype(str)
    result = result[result["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    result["image_id"] = result["image_id"].astype(str)
    result["trusted_group_id"] = result[group_column].astype(str)
    if "trusted_level" not in result.columns:
        result["trusted_level"] = "strong"
    else:
        result["trusted_level"] = result["trusted_level"].fillna("strong").astype(str)
    if "source_type" not in result.columns:
        result["source_type"] = ""
    else:
        result["source_type"] = result["source_type"].fillna("").astype(str)
    if "review_note" not in result.columns:
        result["review_note"] = ""
    else:
        result["review_note"] = result["review_note"].fillna("").astype(str)
    result = result.drop_duplicates(subset=["dataset", "image_id"]).reset_index(drop=True)
    group_sizes = result.groupby("trusted_group_id")["image_id"].transform("size").astype(int)
    result["trusted_group_size"] = group_sizes
    return result[
        [
            "dataset",
            "image_id",
            "trusted_group_id",
            "trusted_group_size",
            "trusted_level",
            "source_type",
            "review_note",
        ]
    ].copy()


def _build_base_view_rows(merged_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in merged_df.itertuples(index=False):
        rows.append(
            {
                "view_id": f"base::{row.image_id}",
                "dataset": row.dataset,
                "image_id": row.image_id,
                "trusted_group_id": row.trusted_group_id,
                "trusted_group_size": int(row.trusted_group_size),
                "trusted_level": row.trusted_level,
                "source_type": row.source_type,
                "review_note": row.review_note,
                "is_base_view": True,
                "view_recipe_name": "base_identity_v1",
                "view_family": "base",
                "view_description": "Center-body square gray repaired base view from the manifest.",
                "source_image_path": row.source_image_path,
                "materialized_view_path": row.source_image_path,
                "materialization_status": "ready",
                "transform_payload_json_v1": json.dumps({"op": "identity", "source": "manifest_base"}, ensure_ascii=False),
                "augmentor_hint_v1": "None; this is the base manifest view.",
                "gate_enabled_v1": True,
                "gate_reason_v1": "manifest_base",
                "manifest_view_name_v1": row.manifest_view_name_v1,
                "preprocess_variant_v1": row.preprocess_variant_v1,
            }
        )
    return pd.DataFrame(rows)


def _build_positive_view_rows(merged_df: pd.DataFrame, recipes: list[ViewRecipe]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in merged_df.iterrows():
        series = row[1]
        for recipe in recipes:
            enabled, gate_reason = recipe.gate_fn(series)
            rows.append(
                {
                    "view_id": f"pos::{series['image_id']}::{recipe.name}",
                    "dataset": str(series["dataset"]),
                    "image_id": str(series["image_id"]),
                    "trusted_group_id": str(series["trusted_group_id"]),
                    "trusted_group_size": int(series["trusted_group_size"]),
                    "trusted_level": str(series["trusted_level"]),
                    "source_type": str(series["source_type"]),
                    "review_note": str(series["review_note"]),
                    "is_base_view": False,
                    "view_recipe_name": recipe.name,
                    "view_family": recipe.family,
                    "view_description": recipe.description,
                    "source_image_path": str(series["source_image_path"]),
                    "materialized_view_path": "",
                    "materialization_status": "metadata_only",
                    "transform_payload_json_v1": json.dumps(recipe.payload, ensure_ascii=False, sort_keys=True),
                    "augmentor_hint_v1": recipe.augmentor_hint,
                    "gate_enabled_v1": bool(enabled),
                    "gate_reason_v1": gate_reason,
                    "manifest_view_name_v1": str(series["manifest_view_name_v1"]),
                    "preprocess_variant_v1": str(series["preprocess_variant_v1"]),
                }
            )
    return pd.DataFrame(rows)


def _build_pair_rows(positive_df: pd.DataFrame) -> pd.DataFrame:
    enabled_df = positive_df[positive_df["gate_enabled_v1"].map(_as_bool)].copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for row in enabled_df.itertuples(index=False):
        rows.append(
            {
                "pair_id": f"pair::{row.image_id}::{row.view_recipe_name}",
                "dataset": row.dataset,
                "image_id": row.image_id,
                "trusted_group_id": row.trusted_group_id,
                "trusted_group_size": int(row.trusted_group_size),
                "base_view_id": f"base::{row.image_id}",
                "positive_view_id": row.view_id,
                "pair_kind": "intra_image_pseudo_positive",
                "positive_recipe_name": row.view_recipe_name,
                "positive_view_family": row.view_family,
                "base_image_path": row.source_image_path,
                "positive_source_image_path": row.source_image_path,
                "positive_materialization_status": row.materialization_status,
                "positive_transform_payload_json_v1": row.transform_payload_json_v1,
                "augmentor_hint_v1": row.augmentor_hint_v1,
                "source_type": row.source_type,
                "trusted_level": row.trusted_level,
                "review_note": row.review_note,
            }
        )
    return pd.DataFrame(rows)


def build_texas_pseudo_positive_views(
    *,
    trusted_membership_path: Path,
    manifest_path: Path,
    output_dir: Path,
    recipes: list[ViewRecipe] | None = None,
) -> dict[str, Path]:
    trusted_df = pd.read_csv(trusted_membership_path)
    manifest_df = pd.read_csv(manifest_path)
    canonical_df = canonicalize_trusted_texas_membership(trusted_df)

    source_path_column = _resolve_source_path_column(manifest_df)
    manifest_df = manifest_df.copy().reset_index(drop=True)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    if "dataset" not in manifest_df.columns:
        manifest_df["dataset"] = TEXAS_DATASET
    manifest_df["dataset"] = manifest_df["dataset"].fillna(TEXAS_DATASET).astype(str)
    manifest_df = manifest_df[manifest_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    manifest_df["source_image_path"] = manifest_df[source_path_column].astype(str)
    if "manifest_view_name_v1" not in manifest_df.columns:
        manifest_df["manifest_view_name_v1"] = manifest_df.get("preprocess_variant_v1", "texas_center_body_square_gray_v1")
    if "preprocess_variant_v1" not in manifest_df.columns:
        manifest_df["preprocess_variant_v1"] = manifest_df["manifest_view_name_v1"]

    merged_df = canonical_df.merge(
        manifest_df,
        on=["dataset", "image_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "__manifest"),
    )
    if merged_df["source_image_path"].isna().any():
        missing = merged_df.loc[merged_df["source_image_path"].isna(), ["dataset", "image_id"]].head(5).to_dict(orient="records")
        raise ValueError(f"Trusted membership rows are missing from the manifest, examples: {missing}")

    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    selected_recipes = list(recipes or DEFAULT_VIEW_RECIPES)
    base_df = _build_base_view_rows(merged_df)
    positive_df = _build_positive_view_rows(merged_df, selected_recipes)
    views_df = pd.concat([base_df, positive_df], ignore_index=True)
    pairs_df = _build_pair_rows(positive_df)

    views_path = tables_dir / "pseudo_positive_views_v1.csv"
    pairs_path = tables_dir / "pseudo_positive_pairs_v1.csv"
    summary_path = reports_dir / "summary.md"
    views_df.to_csv(views_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)

    recipe_summary_df = (
        positive_df.groupby(["view_recipe_name", "view_family", "gate_enabled_v1"])
        .size()
        .reset_index(name="views")
        .sort_values(["view_recipe_name", "gate_enabled_v1"], ascending=[True, False])
        .reset_index(drop=True)
    )
    trusted_summary_df = (
        canonical_df.groupby(["trusted_group_id", "trusted_group_size", "trusted_level", "source_type"])
        .size()
        .reset_index(name="member_rows")
        .sort_values(["trusted_group_size", "trusted_group_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary_lines = [
        "# Texas Pseudo-Positive Views",
        "",
        "## Inputs",
        "",
        f"- Trusted membership: `{trusted_membership_path}`",
        f"- Base manifest: `{manifest_path}`",
        "- Base view source priority: `recommended_model_input_path_v1 -> preferred_path_v1 -> path`",
        "- Pairing rule: `base view` vs `derived positive view`; no aug-vs-aug pairs are emitted here.",
        "",
        "## Summary",
        "",
        f"- Trusted Texas images: `{int(len(canonical_df))}`",
        f"- Trusted groups: `{int(canonical_df['trusted_group_id'].nunique())}`",
        f"- Base views: `{int(len(base_df))}`",
        f"- Positive view metadata rows: `{int(len(positive_df))}`",
        f"- Enabled pseudo-positive pairs: `{int(len(pairs_df))}`",
        "",
        "## Trusted Groups",
        "",
        _markdown_table(trusted_summary_df, limit=20),
        "",
        "## Recipe Coverage",
        "",
        _markdown_table(recipe_summary_df, limit=20),
        "",
        "## Reading Notes",
        "",
        "- `pseudo_positive_views_v1.csv` includes both base rows and metadata-only positive-view rows.",
        "- `pseudo_positive_pairs_v1.csv` is the training-facing table: each row points from one base view to one enabled positive recipe for the same source image.",
        "- `augmentor_hint_v1` is descriptive only for now; it documents how a later materialization step could map each recipe into an Augmentor-style pipeline.",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "views_path": views_path,
        "pairs_path": pairs_path,
        "summary_path": summary_path,
    }

