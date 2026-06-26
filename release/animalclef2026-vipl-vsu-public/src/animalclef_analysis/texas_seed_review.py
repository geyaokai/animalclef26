from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .initial_audit import create_contact_sheet
from .texas_unsupervised import TEXAS_DATASET, dataframe_to_markdown_table


def _to_repo_relative(repo_root: Path, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    return os.path.relpath(path.resolve(), start=repo_root.resolve()).replace("\\", "/")


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    manifest_df = pd.read_csv(manifest_path)
    manifest_df["image_id"] = manifest_df["image_id"].astype(str)
    manifest_df["dataset"] = manifest_df["dataset"].astype(str)
    texas_df = manifest_df[(manifest_df["dataset"] == TEXAS_DATASET) & (manifest_df["split"].astype(str) == "test")].copy()
    if texas_df.empty:
        raise ValueError(f"No Texas test rows found in manifest: {manifest_path}")
    keep_columns = [
        "image_id",
        "dataset",
        "path",
        "original_rgb_path_v1",
        "texas_center_body_repaired_fallback_stage_v1",
        "texas_center_body_repaired_prompt_v1",
        "texas_center_body_foreground_ratio_in_crop_v1",
        "texas_center_body_foreground_ratio_of_subject_v1",
        "texas_center_body_gray_low_value_v1",
        "texas_center_body_gray_high_value_v1",
        "texas_center_body_square_side_px_v1",
    ]
    return texas_df[[column for column in keep_columns if column in texas_df.columns]].copy().reset_index(drop=True)


def _load_assignments(assignments_path: Path) -> pd.DataFrame:
    assignments_df = pd.read_csv(assignments_path)
    assignments_df["image_id"] = assignments_df["image_id"].astype(str)
    assignments_df["dataset"] = assignments_df["dataset"].astype(str)
    texas_df = assignments_df[assignments_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if texas_df.empty:
        raise ValueError(f"No Texas rows found in assignments: {assignments_path}")
    texas_df["seed_status"] = texas_df["seed_status"].fillna("uncertain").astype(str)
    texas_df["pseudo_identity"] = texas_df["pseudo_identity"].fillna("").astype(str)
    return texas_df


def _build_seed_class_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    if seed_df.empty:
        return pd.DataFrame(
            columns=[
                "pseudo_identity",
                "size",
                "mean_component_density",
                "repaired_fallback_seed_images",
                "fallback_prompts",
            ]
        )
    summary_df = (
        seed_df.groupby("pseudo_identity")
        .agg(
            size=("image_id", "count"),
            mean_component_density=("component_density", "mean"),
            repaired_fallback_seed_images=("texas_center_body_repaired_fallback_stage_v1", lambda values: int(pd.Series(values).ne("none").sum())),
            fallback_prompts=(
                "texas_center_body_repaired_prompt_v1",
                lambda values: "|".join(
                    sorted(
                        {
                            str(value)
                            for value in values
                            if str(value) and str(value).lower() != "nan"
                        }
                    )
                ),
            ),
        )
        .reset_index()
        .sort_values(["size", "pseudo_identity"], ascending=[False, True])
        .reset_index(drop=True)
    )
    summary_df["mean_component_density"] = summary_df["mean_component_density"].round(6)
    return summary_df


def _coalesce_suffix_columns(frame: pd.DataFrame, base_names: list[str]) -> pd.DataFrame:
    merged = frame.copy()
    for base_name in base_names:
        direct_exists = base_name in merged.columns
        left_name = f"{base_name}_x"
        right_name = f"{base_name}_y"
        if direct_exists:
            continue
        if left_name in merged.columns and right_name in merged.columns:
            merged[base_name] = merged[left_name].where(~pd.isna(merged[left_name]), merged[right_name])
            merged = merged.drop(columns=[left_name, right_name])
        elif left_name in merged.columns:
            merged = merged.rename(columns={left_name: base_name})
        elif right_name in merged.columns:
            merged = merged.rename(columns={right_name: base_name})
    return merged


def _write_cluster_contact_sheets(
    *,
    repo_root: Path,
    seed_df: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_rows: list[dict[str, object]] = []
    for cluster_index, (pseudo_identity, cluster_df) in enumerate(
        seed_df.groupby("pseudo_identity", sort=True),
        start=1,
    ):
        ordered_df = cluster_df.sort_values(
            ["texas_center_body_repaired_fallback_stage_v1", "image_id"],
            ascending=[True, True],
        ).reset_index(drop=True)
        contact_path = output_dir / f"{cluster_index:02d}_{pseudo_identity}_n{len(ordered_df)}.jpg"
        create_contact_sheet(
            ordered_df,
            repo_root=repo_root,
            output_path=contact_path,
            title=f"{pseudo_identity} | n={len(ordered_df)}",
            caption_columns=[
                "image_id",
                "texas_center_body_repaired_fallback_stage_v1",
                "component_density",
            ],
            columns=min(4, max(len(ordered_df), 1)),
        )
        contact_rows.append(
            {
                "pseudo_identity": pseudo_identity,
                "size": int(len(ordered_df)),
                "contact_sheet_path": _to_repo_relative(repo_root, contact_path),
            }
        )
    return contact_rows


def build_texas_seed_review_package(
    *,
    repo_root: Path,
    assignments_path: Path,
    candidate_pairs_path: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    assignments_path = assignments_path.resolve()
    candidate_pairs_path = candidate_pairs_path.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    qualitative_dir = output_dir / "qualitative"
    cluster_sheet_dir = qualitative_dir / "seed_clusters"
    for directory in [tables_dir, reports_dir, qualitative_dir, cluster_sheet_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    assignments_df = _load_assignments(assignments_path)
    manifest_df = _load_manifest(manifest_path)
    candidate_pairs_df = pd.read_csv(candidate_pairs_path) if candidate_pairs_path.exists() else pd.DataFrame()
    candidate_pairs_df["image_id"] = candidate_pairs_df.get("image_id", pd.Series(dtype=str)).astype(str)
    if "neighbor_image_id" in candidate_pairs_df.columns:
        candidate_pairs_df["neighbor_image_id"] = candidate_pairs_df["neighbor_image_id"].astype(str)

    enriched_df = assignments_df.merge(manifest_df, on=["image_id", "dataset"], how="left")
    missing_manifest = enriched_df["path_y"].isna().sum() if "path_y" in enriched_df.columns else 0
    if missing_manifest:
        missing = enriched_df.loc[enriched_df["path_y"].isna(), "image_id"].tolist()[:5]
        raise ValueError(f"Review package is missing manifest rows for Texas images: {missing}")
    if "path_y" in enriched_df.columns:
        enriched_df["path"] = enriched_df["path_y"]
        enriched_df = enriched_df.drop(columns=[column for column in ["path_x", "path_y"] if column in enriched_df.columns])
    enriched_df = _coalesce_suffix_columns(
        enriched_df,
        base_names=[
            "original_rgb_path_v1",
            "texas_center_body_repaired_fallback_stage_v1",
            "texas_center_body_repaired_prompt_v1",
            "texas_center_body_foreground_ratio_in_crop_v1",
            "texas_center_body_foreground_ratio_of_subject_v1",
            "texas_center_body_gray_low_value_v1",
            "texas_center_body_gray_high_value_v1",
            "texas_center_body_square_side_px_v1",
        ],
    )

    seed_df = enriched_df[enriched_df["seed_status"] == "seed"].copy().reset_index(drop=True)
    uncertain_df = enriched_df[enriched_df["seed_status"] != "seed"].copy().reset_index(drop=True)

    pseudo_assignments_path = tables_dir / "pseudo_assignments_v1.csv"
    pseudo_manifest_path = tables_dir / "pseudo_manifest_v1.csv"
    seed_class_summary_path = tables_dir / "seed_class_summary_v1.csv"
    size_distribution_path = tables_dir / "seed_class_size_distribution_v1.csv"
    candidate_pairs_output_path = tables_dir / "candidate_pairs_v1.csv"

    enriched_df.to_csv(pseudo_assignments_path, index=False)
    seed_df.to_csv(pseudo_manifest_path, index=False)
    seed_class_summary_df = _build_seed_class_summary(seed_df)
    seed_class_summary_df.to_csv(seed_class_summary_path, index=False)
    size_distribution_df = (
        seed_class_summary_df["size"].value_counts().rename_axis("cluster_size").reset_index(name="clusters").sort_values("cluster_size")
        if not seed_class_summary_df.empty
        else pd.DataFrame(columns=["cluster_size", "clusters"])
    )
    size_distribution_df.to_csv(size_distribution_path, index=False)
    candidate_pairs_df.to_csv(candidate_pairs_output_path, index=False)

    contact_rows = _write_cluster_contact_sheets(repo_root=repo_root, seed_df=seed_df, output_dir=cluster_sheet_dir)
    contact_sheet_index_df = pd.DataFrame(contact_rows)
    contact_sheet_index_path = tables_dir / "seed_cluster_contact_sheets_v1.csv"
    contact_sheet_index_df.to_csv(contact_sheet_index_path, index=False)

    if not seed_df.empty:
        overview_path = qualitative_dir / "seed_overview_v1.jpg"
        create_contact_sheet(
            seed_df.sort_values(["pseudo_identity", "image_id"]).reset_index(drop=True),
            repo_root=repo_root,
            output_path=overview_path,
            title=f"Texas pseudo seed overview | n={len(seed_df)}",
            caption_columns=["pseudo_identity", "image_id", "texas_center_body_repaired_fallback_stage_v1"],
            columns=6,
        )
    else:
        overview_path = qualitative_dir / "seed_overview_v1.jpg"

    summary_df = pd.DataFrame(
        [
            {
                "total_images": int(len(enriched_df)),
                "seed_images": int(len(seed_df)),
                "seed_clusters": int(seed_df["pseudo_identity"].nunique()) if not seed_df.empty else 0,
                "uncertain_images": int(len(uncertain_df)),
                "repaired_fallback_total": int(enriched_df["texas_center_body_repaired_fallback_stage_v1"].ne("none").sum()),
                "repaired_fallback_seed_images": int(seed_df["texas_center_body_repaired_fallback_stage_v1"].ne("none").sum()) if not seed_df.empty else 0,
            }
        ]
    )

    summary_lines = [
        "# Texas Pseudo Seed Review Package",
        "",
        "## Inputs",
        "",
        f"- Assignments: `{_to_repo_relative(repo_root, assignments_path)}`",
        f"- Candidate pairs: `{_to_repo_relative(repo_root, candidate_pairs_path)}`",
        f"- Manifest: `{_to_repo_relative(repo_root, manifest_path)}`",
        "",
        "## Summary",
        "",
        dataframe_to_markdown_table(summary_df),
        "",
        "## Seed Class Sizes",
        "",
        dataframe_to_markdown_table(size_distribution_df),
        "",
        "## Seed Class Summary",
        "",
        dataframe_to_markdown_table(seed_class_summary_df),
        "",
        "## Review Assets",
        "",
        f"- Pseudo assignments: `{_to_repo_relative(repo_root, pseudo_assignments_path)}`",
        f"- Pseudo manifest: `{_to_repo_relative(repo_root, pseudo_manifest_path)}`",
        f"- Seed class summary: `{_to_repo_relative(repo_root, seed_class_summary_path)}`",
        f"- Contact sheet index: `{_to_repo_relative(repo_root, contact_sheet_index_path)}`",
        f"- Overview board: `{_to_repo_relative(repo_root, overview_path)}`",
        "",
    ]
    summary_path = reports_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "dataset": TEXAS_DATASET,
                "assignments_path": _to_repo_relative(repo_root, assignments_path),
                "candidate_pairs_path": _to_repo_relative(repo_root, candidate_pairs_path),
                "manifest_path": _to_repo_relative(repo_root, manifest_path),
                "pseudo_assignments_path": _to_repo_relative(repo_root, pseudo_assignments_path),
                "pseudo_manifest_path": _to_repo_relative(repo_root, pseudo_manifest_path),
                "seed_class_summary_path": _to_repo_relative(repo_root, seed_class_summary_path),
                "contact_sheet_index_path": _to_repo_relative(repo_root, contact_sheet_index_path),
                "total_images": int(len(enriched_df)),
                "seed_images": int(len(seed_df)),
                "seed_clusters": int(seed_df["pseudo_identity"].nunique()) if not seed_df.empty else 0,
                "repaired_fallback_seed_images": int(seed_df["texas_center_body_repaired_fallback_stage_v1"].ne("none").sum()) if not seed_df.empty else 0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": summary_path,
        "pseudo_assignments_path": pseudo_assignments_path,
        "pseudo_manifest_path": pseudo_manifest_path,
        "seed_class_summary_path": seed_class_summary_path,
        "contact_sheet_index_path": contact_sheet_index_path,
    }
