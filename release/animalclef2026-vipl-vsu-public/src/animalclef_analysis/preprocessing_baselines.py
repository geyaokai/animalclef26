from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .initial_audit import load_metadata, sample_orientation_rows


SALAMANDER_ROTATION_TO_TOP_V1 = {
    "top": 0,
    "right": 90,
    "left": -90,
    "bottom": 180,
}


def rotation_angle_for_orientation(orientation: str) -> int:
    value = (orientation or "").strip().lower()
    return SALAMANDER_ROTATION_TO_TOP_V1.get(value, 0)


def build_salamander_orientation_manifest(
    metadata_df: pd.DataFrame,
    output_dir: Path,
    repo_root: Path,
) -> pd.DataFrame:
    salamander_df = metadata_df[metadata_df["dataset"] == "SalamanderID2025"].copy()
    rows: list[dict[str, object]] = []
    image_root = output_dir / "salamander_orientation_v1"
    for row in salamander_df.itertuples(index=False):
        rotation = rotation_angle_for_orientation(row.orientation)
        normalized_rel = (
            output_dir.relative_to(repo_root)
            / "salamander_orientation_v1"
            / Path(row.path).relative_to("images")
        )
        normalized_abs = repo_root / normalized_rel
        normalized_abs.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(repo_root / row.path) as image:
            image = image.convert("RGB")
            if rotation:
                normalized = image.rotate(rotation, expand=True)
            else:
                normalized = image.copy()
            normalized.save(normalized_abs, quality=95)
        rows.append(
            {
                "image_id": row.image_id,
                "identity": row.identity,
                "dataset": row.dataset,
                "species": row.species,
                "split": row.split,
                "orientation": row.orientation,
                "rotation_degrees_v1": rotation,
                "orientation_rule_v1": "to_top",
                "original_path": row.path,
                "normalized_path_v1": str(normalized_rel),
                "normalization_applied_v1": bool(rotation),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "identity", "image_id"]).reset_index(drop=True)


def build_duplicate_flags_from_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    usable = metrics_df[(metrics_df["sha1"].notna()) & (metrics_df["sha1"] != "")].copy()
    usable["image_id"] = usable["image_id"].astype(str)
    usable["duplicate_group_size"] = usable.groupby("sha1")["image_id"].transform("count")
    usable["is_exact_duplicate"] = usable["duplicate_group_size"] > 1
    usable = usable.sort_values(["sha1", "path", "image_id"]).reset_index(drop=True)
    usable["duplicate_rank"] = usable.groupby("sha1").cumcount()
    usable["is_duplicate_primary"] = usable["duplicate_rank"] == 0
    return usable[
        [
            "image_id",
            "path",
            "dataset",
            "split",
            "identity",
            "sha1",
            "duplicate_group_size",
            "duplicate_rank",
            "is_exact_duplicate",
            "is_duplicate_primary",
        ]
    ].rename(columns={"sha1": "exact_duplicate_sha1"})


def create_enriched_metadata(
    metadata_df: pd.DataFrame,
    orientation_manifest_df: pd.DataFrame,
    duplicate_flags_df: pd.DataFrame,
) -> pd.DataFrame:
    enriched = metadata_df.copy()
    enriched["preferred_path_v1"] = enriched["path"]
    enriched["preprocess_variant_v1"] = "original"
    enriched["rotation_degrees_v1"] = 0
    enriched["normalized_path_v1"] = ""
    enriched["normalization_applied_v1"] = False

    orientation_cols = orientation_manifest_df[
        [
            "image_id",
            "rotation_degrees_v1",
            "normalized_path_v1",
            "normalization_applied_v1",
        ]
    ].copy()
    enriched = enriched.merge(orientation_cols, on="image_id", how="left", suffixes=("", "_manifest"))
    has_manifest = enriched["normalized_path_v1_manifest"].fillna("") != ""
    enriched.loc[has_manifest, "rotation_degrees_v1"] = enriched.loc[has_manifest, "rotation_degrees_v1_manifest"]
    enriched.loc[has_manifest, "normalized_path_v1"] = enriched.loc[has_manifest, "normalized_path_v1_manifest"]
    enriched.loc[has_manifest, "normalization_applied_v1"] = enriched.loc[has_manifest, "normalization_applied_v1_manifest"]
    enriched.loc[has_manifest, "preferred_path_v1"] = enriched.loc[has_manifest, "normalized_path_v1_manifest"]
    enriched.loc[has_manifest, "preprocess_variant_v1"] = np.where(
        enriched.loc[has_manifest, "normalization_applied_v1_manifest"],
        "salamander_orientation_v1",
        "original",
    )
    enriched = enriched.drop(
        columns=[
            "rotation_degrees_v1_manifest",
            "normalized_path_v1_manifest",
            "normalization_applied_v1_manifest",
        ]
    )

    duplicate_cols = duplicate_flags_df[
        [
            "image_id",
            "exact_duplicate_sha1",
            "duplicate_group_size",
            "duplicate_rank",
            "is_exact_duplicate",
            "is_duplicate_primary",
        ]
    ]
    enriched = enriched.merge(duplicate_cols, on="image_id", how="left")
    enriched["duplicate_group_size"] = enriched["duplicate_group_size"].fillna(1).astype(int)
    enriched["duplicate_rank"] = enriched["duplicate_rank"].fillna(0).astype(int)
    enriched["is_exact_duplicate"] = enriched["is_exact_duplicate"].fillna(False)
    enriched["is_duplicate_primary"] = enriched["is_duplicate_primary"].fillna(True)
    enriched["exact_duplicate_sha1"] = enriched["exact_duplicate_sha1"].fillna("")
    enriched["local_label_available_v1"] = enriched["identity"].fillna("") != ""
    enriched["sea_turtle_duplicate_nonprimary_v1"] = (
        (enriched["dataset"] == "SeaTurtleID2022")
        & enriched["is_exact_duplicate"]
        & np.logical_not(enriched["is_duplicate_primary"])
    )
    enriched["recommended_train_keep_all_v1"] = (
        (enriched["split"] == "train") & enriched["local_label_available_v1"]
    )
    enriched["recommended_train_keep_dedup_v1"] = (
        enriched["recommended_train_keep_all_v1"] & (~enriched["sea_turtle_duplicate_nonprimary_v1"])
    )
    enriched["recommended_model_input_path_v1"] = enriched["preferred_path_v1"]
    return enriched


def create_orientation_preview(
    orientation_manifest_df: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    sample_seed: int = 42,
    columns: int = 3,
) -> None:
    preview_source = orientation_manifest_df.copy()
    preview_source["path"] = preview_source["original_path"]
    sampled = sample_orientation_rows(
        preview_source,
        dataset="SalamanderID2025",
        per_orientation=3,
        seed=sample_seed,
        orientations=["top", "right", "bottom", "left"],
    )
    if sampled.empty:
        return
    thumb_size = (220, 220)
    margin = 12
    header_h = 34
    label_h = 22
    rows = math.ceil(len(sampled) / columns)
    width = margin * 2 + columns * ((thumb_size[0] * 2) + margin) + (columns - 1) * margin
    height = margin * 2 + header_h + rows * (thumb_size[1] + label_h) + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), "Salamander Orientation Preview | original -> normalized", fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for idx, row in enumerate(sampled.itertuples(index=False)):
        gx = idx % columns
        gy = idx // columns
        x = margin + gx * (((thumb_size[0] * 2) + margin) + margin)
        y = start_y + gy * ((thumb_size[1] + label_h) + margin)
        with Image.open(repo_root / row.original_path) as image:
            original = ImageOps.pad(image.convert("RGB"), thumb_size, color=(10, 10, 10))
        with Image.open(repo_root / row.normalized_path_v1) as image:
            normalized = ImageOps.pad(image.convert("RGB"), thumb_size, color=(10, 10, 10))
        canvas.paste(original, (x, y))
        canvas.paste(normalized, (x + thumb_size[0] + margin, y))
        label = f"{row.orientation} | rot={int(row.rotation_degrees_v1)} | {row.image_id}"
        draw.text((x, y + thumb_size[1] + 3), label, fill=(30, 30, 30), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_preprocessing_summary(
    output_path: Path,
    orientation_manifest_df: pd.DataFrame,
    duplicate_flags_df: pd.DataFrame,
) -> None:
    rotation_counts = (
        orientation_manifest_df.groupby(["orientation", "rotation_degrees_v1"])
        .size()
        .reset_index(name="count")
        .sort_values(["count", "orientation"], ascending=[False, True])
    )
    seaturtle_duplicates = duplicate_flags_df[
        (duplicate_flags_df["dataset"] == "SeaTurtleID2022") & (duplicate_flags_df["is_exact_duplicate"])
    ].copy()
    duplicate_summary = (
        seaturtle_duplicates.groupby("identity")["image_id"]
        .count()
        .reset_index(name="duplicate_images")
        .sort_values(["duplicate_images", "identity"], ascending=[False, True])
    )

    def as_markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for _, values in frame.iterrows():
            rows.append("| " + " | ".join(str(values[column]) for column in columns) + " |")
        return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])

    lines = [
        "# Preprocessing Baselines v1",
        "",
        "## Salamander Orientation Normalization",
        "",
        "- Rule: rotate images to a common `top` orientation using metadata.",
        "- Mapping: `top -> 0`, `right -> +90`, `left -> -90`, `bottom -> 180`.",
        "- Output images are written to `artifacts/preprocessing_baselines/v1/salamander_orientation_v1/`.",
        "",
        as_markdown_table(rotation_counts),
        "",
        "## Sea Turtle Exact Duplicate Flags",
        "",
        "- Duplicate logic is exact file-content matching via the SHA1 hashes computed in the initial audit.",
        "- No original files were removed or overwritten.",
        "",
        as_markdown_table(duplicate_summary),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_preprocessing_baselines(
    repo_root: Path,
    audit_dir: Path,
    output_dir: Path,
    sample_seed: int = 42,
) -> dict[str, Path]:
    metadata_df = load_metadata(repo_root / "metadata.csv")
    metrics_df = pd.read_csv(audit_dir / "tables" / "image_metrics.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    qualitative_dir = output_dir / "qualitative"
    for path in [tables_dir, reports_dir, qualitative_dir]:
        path.mkdir(parents=True, exist_ok=True)

    orientation_manifest_df = build_salamander_orientation_manifest(
        metadata_df=metadata_df,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    duplicate_flags_df = build_duplicate_flags_from_metrics(metrics_df)
    enriched_metadata_df = create_enriched_metadata(
        metadata_df=metadata_df,
        orientation_manifest_df=orientation_manifest_df,
        duplicate_flags_df=duplicate_flags_df,
    )

    orientation_manifest_path = tables_dir / "salamander_orientation_manifest_v1.csv"
    duplicate_flags_path = tables_dir / "exact_duplicate_flags_v1.csv"
    enriched_metadata_path = tables_dir / "metadata_enriched_v1.csv"
    train_keep_all_path = tables_dir / "baseline_manifest_train_keep_all_v1.csv"
    train_dedup_path = tables_dir / "baseline_manifest_train_dedup_v1.csv"
    test_manifest_path = tables_dir / "baseline_manifest_test_v1.csv"
    orientation_manifest_df.to_csv(orientation_manifest_path, index=False)
    duplicate_flags_df.to_csv(duplicate_flags_path, index=False)
    enriched_metadata_df.to_csv(enriched_metadata_path, index=False)
    enriched_metadata_df[enriched_metadata_df["recommended_train_keep_all_v1"]].to_csv(
        train_keep_all_path, index=False
    )
    enriched_metadata_df[enriched_metadata_df["recommended_train_keep_dedup_v1"]].to_csv(
        train_dedup_path, index=False
    )
    enriched_metadata_df[enriched_metadata_df["split"] == "test"].to_csv(
        test_manifest_path, index=False
    )

    preview_path = qualitative_dir / "salamander_orientation_preview_v1.jpg"
    create_orientation_preview(
        orientation_manifest_df=orientation_manifest_df,
        repo_root=repo_root,
        output_path=preview_path,
        sample_seed=sample_seed,
    )

    summary_path = reports_dir / "summary.md"
    write_preprocessing_summary(
        output_path=summary_path,
        orientation_manifest_df=orientation_manifest_df,
        duplicate_flags_df=duplicate_flags_df,
    )
    summary_json = {
        "salamander_images": int(len(orientation_manifest_df)),
        "salamander_rotated_images": int(orientation_manifest_df["normalization_applied_v1"].sum()),
        "exact_duplicate_images": int(duplicate_flags_df["is_exact_duplicate"].sum()),
        "exact_duplicate_groups": int(
            duplicate_flags_df.loc[duplicate_flags_df["is_exact_duplicate"], "exact_duplicate_sha1"].nunique()
        ),
        "train_keep_all_rows": int(enriched_metadata_df["recommended_train_keep_all_v1"].sum()),
        "train_dedup_rows": int(enriched_metadata_df["recommended_train_keep_dedup_v1"].sum()),
        "test_rows": int((enriched_metadata_df["split"] == "test").sum()),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "orientation_manifest_path": orientation_manifest_path,
        "duplicate_flags_path": duplicate_flags_path,
        "enriched_metadata_path": enriched_metadata_path,
        "train_keep_all_path": train_keep_all_path,
        "train_dedup_path": train_dedup_path,
        "test_manifest_path": test_manifest_path,
        "preview_path": preview_path,
        "summary_path": summary_path,
    }
