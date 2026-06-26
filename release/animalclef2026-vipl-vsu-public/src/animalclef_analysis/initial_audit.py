from __future__ import annotations

import hashlib
import json
import math
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


REQUIRED_COLUMNS = [
    "image_id",
    "identity",
    "path",
    "date",
    "orientation",
    "species",
    "split",
    "dataset",
]


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    ensure_required_columns(df)
    df["image_id"] = df["image_id"].astype(str)
    return df


def build_count_table(
    df: pd.DataFrame, group_columns: list[str], count_name: str = "count"
) -> pd.DataFrame:
    if not group_columns:
        raise ValueError("group_columns must not be empty")
    return (
        df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name=count_name)
        .sort_values(group_columns + [count_name], ascending=[True] * len(group_columns) + [False])
        .reset_index(drop=True)
    )


def compute_sharpness(gray_array: np.ndarray) -> float:
    if gray_array.ndim != 2:
        raise ValueError("compute_sharpness expects a 2D grayscale array")
    if gray_array.shape[0] < 2 or gray_array.shape[1] < 2:
        return 0.0
    gray = gray_array.astype(np.float32)
    grad_x = np.diff(gray, axis=1)
    grad_y = np.diff(gray, axis=0)
    return float(np.var(grad_x) + np.var(grad_y))


def hash_file(file_path: Path) -> str:
    sha1 = hashlib.sha1()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            sha1.update(chunk)
    return sha1.hexdigest()


def analyze_image_file(image_path: Path, compute_hashes: bool = False) -> dict[str, object]:
    record: dict[str, object] = {
        "path_exists": image_path.exists(),
        "readable": False,
        "file_size_bytes": np.nan,
        "width": np.nan,
        "height": np.nan,
        "aspect_ratio": np.nan,
        "mode": "",
        "channels": np.nan,
        "sharpness": np.nan,
        "sha1": "",
        "error": "",
    }
    if not image_path.exists():
        record["error"] = "missing"
        return record

    try:
        record["file_size_bytes"] = image_path.stat().st_size
        with Image.open(image_path) as image:
            image.load()
            rgb = image.convert("RGB")
            gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
            record["width"] = int(rgb.width)
            record["height"] = int(rgb.height)
            record["aspect_ratio"] = round(rgb.width / rgb.height, 6) if rgb.height else np.nan
            record["mode"] = image.mode
            record["channels"] = len(rgb.getbands())
            record["sharpness"] = round(compute_sharpness(gray), 6)
            record["readable"] = True
        if compute_hashes:
            record["sha1"] = hash_file(image_path)
    except Exception as exc:  # pragma: no cover - exercised on real data
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def collect_image_metrics(
    df: pd.DataFrame,
    repo_root: Path,
    compute_hashes: bool = False,
    progress_every: int = 1000,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(df)
    for index, row in enumerate(df.itertuples(index=False), start=1):
        image_path = repo_root / row.path
        metrics = analyze_image_file(image_path, compute_hashes=compute_hashes)
        rows.append(
            {
                "image_id": row.image_id,
                "identity": row.identity,
                "path": row.path,
                "dataset": row.dataset,
                "split": row.split,
                "species": row.species,
                "orientation": row.orientation,
                **metrics,
            }
        )
        if progress_every and index % progress_every == 0:
            print(f"[audit] processed {index}/{total} images")
    return pd.DataFrame(rows)


def summarize_identity_distribution(df: pd.DataFrame) -> pd.DataFrame:
    train_df = df[df["identity"].fillna("") != ""].copy()
    counts = (
        train_df.groupby(["dataset", "identity"], dropna=False)
        .size()
        .reset_index(name="images_per_identity")
    )
    summary = (
        counts.groupby("dataset")["images_per_identity"]
        .agg(
            identities="count",
            min_images="min",
            median_images="median",
            mean_images="mean",
            max_images="max",
            singletons=lambda s: int((s == 1).sum()),
            p90_images=lambda s: float(np.percentile(s, 90)),
        )
        .reset_index()
    )
    summary["mean_images"] = summary["mean_images"].round(3)
    summary["singleton_ratio"] = (summary["singletons"] / summary["identities"]).round(4)
    summary["p90_images"] = summary["p90_images"].round(2)
    return summary.sort_values("dataset").reset_index(drop=True)


def summarize_image_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics_df.groupby(["dataset", "split"], dropna=False)
        .agg(
            images=("image_id", "count"),
            missing_files=("path_exists", lambda s: int((~s).sum())),
            unreadable_files=("readable", lambda s: int((~s).sum())),
            median_width=("width", "median"),
            median_height=("height", "median"),
            median_aspect_ratio=("aspect_ratio", "median"),
            median_file_size_kb=("file_size_bytes", lambda s: float(np.nanmedian(s) / 1024)),
            median_sharpness=("sharpness", "median"),
        )
        .reset_index()
        .sort_values(["dataset", "split"])
        .reset_index(drop=True)
    )
    numeric_columns = [
        "median_width",
        "median_height",
        "median_aspect_ratio",
        "median_file_size_kb",
        "median_sharpness",
    ]
    for column in numeric_columns:
        grouped[column] = grouped[column].round(3)
    return grouped


def detect_duplicate_hashes(metrics_df: pd.DataFrame) -> pd.DataFrame:
    usable = metrics_df[(metrics_df["sha1"] != "") & metrics_df["sha1"].notna()].copy()
    duplicate_hashes = usable.groupby("sha1").size().reset_index(name="duplicate_count")
    duplicate_hashes = duplicate_hashes[duplicate_hashes["duplicate_count"] > 1]
    if duplicate_hashes.empty:
        return duplicate_hashes
    merged = usable.merge(duplicate_hashes, on="sha1", how="inner")
    return merged.sort_values(["duplicate_count", "sha1", "dataset", "path"], ascending=[False, True, True, True])


def sample_random_rows(
    df: pd.DataFrame,
    filters: dict[str, object],
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    subset = df.copy()
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    if subset.empty:
        return subset
    size = min(sample_size, len(subset))
    return subset.sample(n=size, random_state=seed).reset_index(drop=True)


def sample_same_identity_rows(
    df: pd.DataFrame,
    dataset: str,
    identities: int,
    images_per_identity: int,
    seed: int,
) -> pd.DataFrame:
    subset = df[(df["dataset"] == dataset) & (df["identity"].fillna("") != "")].copy()
    counts = subset.groupby("identity").size()
    eligible = counts[counts >= images_per_identity].index.tolist()
    if not eligible:
        return pd.DataFrame(columns=df.columns)
    rng = np.random.default_rng(seed)
    picked = list(rng.choice(eligible, size=min(identities, len(eligible)), replace=False))
    rows: list[pd.DataFrame] = []
    for identity in picked:
        identity_df = subset[subset["identity"] == identity].sample(
            n=images_per_identity, random_state=seed
        )
        rows.append(identity_df)
    return pd.concat(rows, ignore_index=True)


def sample_orientation_rows(
    df: pd.DataFrame,
    dataset: str,
    per_orientation: int,
    seed: int,
    orientations: Iterable[str] | None = None,
) -> pd.DataFrame:
    subset = df[df["dataset"] == dataset].copy()
    if orientations is None:
        orientations = (
            subset["orientation"]
            .fillna("")
            .replace("", "missing")
            .value_counts()
            .head(4)
            .index.tolist()
        )
    rows: list[pd.DataFrame] = []
    for orientation in orientations:
        match_value = "" if orientation == "missing" else orientation
        orientation_df = subset[subset["orientation"] == match_value]
        if orientation_df.empty:
            continue
        rows.append(
            orientation_df.sample(
                n=min(per_orientation, len(orientation_df)),
                random_state=seed,
            )
        )
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(rows, ignore_index=True)


def _wrap_caption(parts: list[str], width: int = 28) -> str:
    compact = " | ".join(part for part in parts if part)
    wrapped = textwrap.wrap(compact, width=width)
    return "\n".join(wrapped[:3])


def create_contact_sheet(
    df: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    caption_columns: list[str],
    columns: int = 4,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if df.empty:
        return

    font = ImageFont.load_default()
    title_height = 36
    caption_height = 54
    margin = 12
    cell_width, image_height = thumb_size
    cell_height = image_height + caption_height
    rows = math.ceil(len(df) / columns)
    canvas_width = margin * 2 + columns * cell_width + (columns - 1) * margin
    canvas_height = (
        margin * 2
        + title_height
        + rows * cell_height
        + (rows - 1) * margin
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)

    start_y = margin + title_height
    for index, row in enumerate(df.itertuples(index=False)):
        grid_x = index % columns
        grid_y = index // columns
        x = margin + grid_x * (cell_width + margin)
        y = start_y + grid_y * (cell_height + margin)
        image_path = repo_root / row.path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            thumb = ImageOps.pad(image, thumb_size, color=(15, 15, 15))
        canvas.paste(thumb, (x, y))
        parts = [str(getattr(row, column, "")) for column in caption_columns]
        caption = _wrap_caption(parts)
        draw.multiline_text(
            (x, y + image_height + 4),
            caption,
            fill=(35, 35, 35),
            font=font,
            spacing=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def write_markdown_summary(
    metadata_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_path: Path,
    duplicate_count: int,
) -> None:
    dataset_split = build_count_table(metadata_df, ["dataset", "split"])
    orientation_counts = build_count_table(metadata_df, ["dataset", "orientation"])
    identity_summary = summarize_identity_distribution(metadata_df)
    image_summary = summarize_image_metrics(metrics_df)
    missing_files = int((~metrics_df["path_exists"]).sum())
    unreadable_files = int((~metrics_df["readable"]).sum())
    exact_duplicates = duplicate_count
    sharpest = metrics_df.sort_values("sharpness", ascending=False).head(5)[
        ["image_id", "dataset", "split", "sharpness", "path"]
    ]

    def as_markdown_table(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for _, values in frame.iterrows():
            cells = []
            for column in columns:
                value = values[column]
                if pd.isna(value):
                    cells.append("")
                else:
                    cells.append(str(value))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, separator, *rows])

    lines = [
        "# Initial Data Audit",
        "",
        "## Overview",
        "",
        f"- Total rows in `metadata.csv`: {len(metadata_df)}",
        f"- Unique `image_id`: {metadata_df['image_id'].nunique()}",
        f"- Missing files: {missing_files}",
        f"- Unreadable files: {unreadable_files}",
        f"- Exact duplicate hashes: {exact_duplicates}",
        "",
        "## Dataset / Split Counts",
        "",
        as_markdown_table(dataset_split),
        "",
        "## Identity Distribution",
        "",
        as_markdown_table(identity_summary),
        "",
        "## Image Property Summary",
        "",
        as_markdown_table(image_summary),
        "",
        "## Orientation Counts",
        "",
        as_markdown_table(orientation_counts),
        "",
        "## Highest Sharpness Samples",
        "",
        as_markdown_table(sharpest),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_initial_audit(
    repo_root: Path,
    output_dir: Path,
    compute_hashes: bool = True,
    sample_seed: int = 42,
) -> dict[str, Path]:
    metadata_path = repo_root / "metadata.csv"
    metadata_df = load_metadata(metadata_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    qualitative_dir = output_dir / "qualitative"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, qualitative_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_image_metrics(metadata_df, repo_root, compute_hashes=compute_hashes)
    dataset_split = build_count_table(metadata_df, ["dataset", "split"])
    dataset_species = build_count_table(metadata_df, ["dataset", "species"])
    orientation_counts = build_count_table(metadata_df, ["dataset", "orientation"])
    identity_summary = summarize_identity_distribution(metadata_df)
    image_summary = summarize_image_metrics(metrics_df)
    duplicates_df = detect_duplicate_hashes(metrics_df) if compute_hashes else pd.DataFrame()

    metadata_df.to_csv(tables_dir / "metadata_copy.csv", index=False)
    metrics_df.to_csv(tables_dir / "image_metrics.csv", index=False)
    dataset_split.to_csv(tables_dir / "dataset_split_counts.csv", index=False)
    dataset_species.to_csv(tables_dir / "dataset_species_counts.csv", index=False)
    orientation_counts.to_csv(tables_dir / "orientation_counts.csv", index=False)
    identity_summary.to_csv(tables_dir / "identity_summary.csv", index=False)
    image_summary.to_csv(tables_dir / "image_summary.csv", index=False)
    if compute_hashes:
        duplicates_df.to_csv(tables_dir / "duplicate_hashes.csv", index=False)

    report_path = reports_dir / "summary.md"
    write_markdown_summary(
        metadata_df=metadata_df,
        metrics_df=metrics_df,
        output_path=report_path,
        duplicate_count=int(duplicates_df["sha1"].nunique()) if not duplicates_df.empty else 0,
    )

    random_jobs = []
    for dataset in sorted(metadata_df["dataset"].unique()):
        for split in sorted(metadata_df[metadata_df["dataset"] == dataset]["split"].unique()):
            sample_df = sample_random_rows(
                metadata_df,
                filters={"dataset": dataset, "split": split},
                sample_size=12,
                seed=sample_seed,
            )
            random_jobs.append((dataset, split, sample_df))
    for dataset, split, sample_df in random_jobs:
        if sample_df.empty:
            continue
        create_contact_sheet(
            sample_df,
            repo_root=repo_root,
            output_path=qualitative_dir / f"random_{dataset}_{split}.jpg",
            title=f"Random Samples | {dataset} | {split}",
            caption_columns=["dataset", "split", "orientation", "identity"],
        )

    for dataset in sorted(metadata_df["dataset"].unique()):
        same_identity_df = sample_same_identity_rows(
            metadata_df, dataset=dataset, identities=4, images_per_identity=4, seed=sample_seed
        )
        if same_identity_df.empty:
            continue
        create_contact_sheet(
            same_identity_df,
            repo_root=repo_root,
            output_path=qualitative_dir / f"same_identity_{dataset}.jpg",
            title=f"Same Identity Audit | {dataset}",
            caption_columns=["identity", "orientation", "split"],
        )

    salamander_orientation_df = sample_orientation_rows(
        metadata_df,
        dataset="SalamanderID2025",
        per_orientation=4,
        seed=sample_seed,
    )
    if not salamander_orientation_df.empty:
        create_contact_sheet(
            salamander_orientation_df,
            repo_root=repo_root,
            output_path=qualitative_dir / "orientation_audit_SalamanderID2025.jpg",
            title="Orientation Audit | SalamanderID2025",
            caption_columns=["orientation", "identity", "split"],
        )

    summary_json = {
        "rows": int(len(metadata_df)),
        "unique_image_id": int(metadata_df["image_id"].nunique()),
        "missing_files": int((~metrics_df["path_exists"]).sum()),
        "unreadable_files": int((~metrics_df["readable"]).sum()),
        "duplicate_hash_groups": int(duplicates_df["sha1"].nunique()) if not duplicates_df.empty else 0,
        "datasets": dataset_split.to_dict(orient="records"),
    }
    (reports_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_dir": output_dir,
        "report_path": report_path,
        "summary_json_path": reports_dir / "summary.json",
        "tables_dir": tables_dir,
        "qualitative_dir": qualitative_dir,
    }
