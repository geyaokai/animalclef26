from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TCU_IDENTITY_PATTERN = re.compile(r"^(?P<year>\d{4})-?(?P<identity>HL\d+)[-_]?(?P<suffix>[A-Za-z]?)$", re.IGNORECASE)
CHIP_FILENAME_PATTERN = re.compile(r"^cid(?P<chip_id>\d+)_CHIP\(sz\d+\)\.(?P<ext>[A-Za-z0-9]+)$")
HOTSPOTTER_CHIP_PATTERN = re.compile(r"^Chip\s+(?P<chip_id>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class TcuTexasDatasetArtifacts:
    chip_manifest_path: Path
    original_manifest_path: Path
    chip_alignment_audit_path: Path
    original_coverage_audit_path: Path
    summary_path: Path


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def _to_repo_relative(repo_root: Path, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    path = Path(str(value))
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    return os.path.relpath(path.resolve(), start=repo_root.resolve()).replace("\\", "/")


def _normalize_lookup_key(value: object) -> str:
    text = str(value).strip()
    return re.sub(r"\s+", "", text).lower()


def _normalize_stem_key(value: object) -> str:
    stem = Path(str(value).strip()).stem
    return re.sub(r"[^A-Za-z0-9]", "", stem).lower()


def _parse_chip_filename(path: Path) -> dict[str, object]:
    match = CHIP_FILENAME_PATTERN.match(path.name)
    chip_id = str(int(match.group("chip_id"))) if match else ""
    return {
        "chip_filename": path.name,
        "chip_path": path,
        "chip_id": chip_id,
        "chip_parse_ok_v1": bool(match),
    }


def _parse_hotspotter_chip_id(value: object) -> str:
    text = str(value).strip()
    match = HOTSPOTTER_CHIP_PATTERN.match(text)
    if not match:
        return ""
    return str(int(match.group("chip_id")))


def _parse_identity_fields(image_name: object) -> dict[str, object]:
    text = str(image_name).strip()
    stem = re.sub(r"\s+", "", Path(text).stem)
    match = TCU_IDENTITY_PATTERN.match(stem)
    if not match:
        return {
            "capture_year_v1": "",
            "capture_token_v1": "",
            "external_identity_v1": "",
            "filename_parse_ok_v1": False,
        }
    identity_token = match.group("identity").upper()
    suffix = match.group("suffix").upper()
    capture_token = f"{identity_token}{suffix}"
    return {
        "capture_year_v1": match.group("year"),
        "capture_token_v1": capture_token,
        "external_identity_v1": identity_token,
        "filename_parse_ok_v1": True,
    }


def _build_original_match_lookup(original_dir: Path) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path]]:
    exact_lookup: dict[str, Path] = {}
    normalized_lookup: dict[str, Path] = {}
    stem_lookup: dict[str, Path] = {}
    for path in sorted(candidate for candidate in original_dir.iterdir() if candidate.is_file()):
        exact_lookup[path.name] = path
        normalized_key = _normalize_lookup_key(path.name)
        stem_key = _normalize_stem_key(path.name)
        # Keep only unique keys. If a collision appears, we drop the key and let audit expose ambiguity.
        normalized_lookup[normalized_key] = path if normalized_key not in normalized_lookup else Path("")
        stem_lookup[stem_key] = path if stem_key not in stem_lookup else Path("")
    normalized_lookup = {key: value for key, value in normalized_lookup.items() if str(value)}
    stem_lookup = {key: value for key, value in stem_lookup.items() if str(value)}
    return exact_lookup, normalized_lookup, stem_lookup


def _resolve_original_path(
    image_name: object,
    *,
    exact_lookup: dict[str, Path],
    normalized_lookup: dict[str, Path],
    stem_lookup: dict[str, Path],
) -> tuple[str, Path | None]:
    exact_key = str(image_name).strip()
    if exact_key in exact_lookup:
        return "exact", exact_lookup[exact_key]
    normalized_key = _normalize_lookup_key(image_name)
    if normalized_key in normalized_lookup:
        return "normalized_name", normalized_lookup[normalized_key]
    stem_key = _normalize_stem_key(image_name)
    if stem_key in stem_lookup:
        return "normalized_stem", stem_lookup[stem_key]
    return "missing", None


def _load_chip_files(chips_dir: Path) -> pd.DataFrame:
    rows = [_parse_chip_filename(path) for path in sorted(candidate for candidate in chips_dir.iterdir() if candidate.is_file())]
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No chip files found in: {chips_dir}")
    if frame["chip_id"].duplicated().any():
        duplicated = frame.loc[frame["chip_id"].duplicated(), "chip_id"].astype(str).tolist()[:5]
        raise ValueError(f"Duplicate chip ids in chips directory: {duplicated}")
    return frame


def _load_mapping(mapping_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(mapping_path).copy()
    frame["chip_id"] = frame["#ChipID"].map(lambda value: str(int(value)))
    frame["original_image_name_mapped_v1"] = frame["Image"].astype(str)
    return frame[["chip_id", "original_image_name_mapped_v1"]].copy()


def _load_hotspotter_output(hotspotter_output_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(hotspotter_output_path).copy()
    frame["chip_id"] = frame["Chip"].map(_parse_hotspotter_chip_id)
    keep_columns = {
        "Image": "hotspotter_split_role_v1",
        "Query Result": "hotspotter_query_result_v1",
        "Rank 1 - Chip": "hotspotter_rank1_chip_id_v1",
        "Rank 1 - Score": "hotspotter_rank1_score_v1",
        "Rank 2 - Chip": "hotspotter_rank2_chip_id_v1",
        "Rank 2 - Score": "hotspotter_rank2_score_v1",
    }
    result = frame[["chip_id", *keep_columns.keys()]].rename(columns=keep_columns)
    for chip_column in ["hotspotter_rank1_chip_id_v1", "hotspotter_rank2_chip_id_v1"]:
        result[chip_column] = result[chip_column].map(lambda value: "" if pd.isna(value) else str(int(value)))
    result["hotspotter_split_role_v1"] = result["hotspotter_split_role_v1"].fillna("").astype(str)
    result["hotspotter_query_result_v1"] = result["hotspotter_query_result_v1"].fillna("").astype(str)
    return result


def _load_original_files(original_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(candidate for candidate in original_dir.iterdir() if candidate.is_file()):
        identity_fields = _parse_identity_fields(path.name)
        rows.append(
            {
                "original_filename_v1": path.name,
                "original_path_v1": path,
                **identity_fields,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No original files found in: {original_dir}")
    if frame["original_filename_v1"].duplicated().any():
        duplicated = frame.loc[frame["original_filename_v1"].duplicated(), "original_filename_v1"].tolist()[:5]
        raise ValueError(f"Duplicate original filenames in original directory: {duplicated}")
    return frame


def _build_chip_manifest(
    *,
    repo_root: Path,
    chips_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    hotspotter_df: pd.DataFrame,
    exact_lookup: dict[str, Path],
    normalized_lookup: dict[str, Path],
    stem_lookup: dict[str, Path],
) -> pd.DataFrame:
    manifest_df = mapping_df.merge(chips_df, on="chip_id", how="outer", indicator=True)
    manifest_df["mapping_presence_v1"] = manifest_df["_merge"].map(
        {"both": "mapping_and_chip", "left_only": "mapping_only", "right_only": "chip_only"}
    )
    manifest_df = manifest_df.drop(columns=["_merge"])
    manifest_df = manifest_df.merge(hotspotter_df, on="chip_id", how="left")

    match_records: list[dict[str, object]] = []
    for row in manifest_df.itertuples(index=False):
        mapped_name = getattr(row, "original_image_name_mapped_v1", "")
        if not str(mapped_name).strip() or str(mapped_name).lower() == "nan":
            stage = "no_mapping"
            matched_path = None
        else:
            stage, matched_path = _resolve_original_path(
                mapped_name,
                exact_lookup=exact_lookup,
                normalized_lookup=normalized_lookup,
                stem_lookup=stem_lookup,
            )
        identity_fields = _parse_identity_fields(mapped_name)
        match_records.append(
            {
                "chip_id": str(getattr(row, "chip_id")),
                "original_match_stage_v1": stage,
                "original_match_ok_v1": bool(matched_path is not None),
                "original_match_filename_v1": matched_path.name if matched_path is not None else "",
                "original_match_path_v1": _to_repo_relative(repo_root, matched_path) if matched_path is not None else "",
                **identity_fields,
            }
        )
    match_df = pd.DataFrame(match_records)
    manifest_df = manifest_df.merge(match_df, on="chip_id", how="left")

    identity_counts = (
        manifest_df.loc[manifest_df["external_identity_v1"].astype(str).ne(""), "external_identity_v1"]
        .value_counts()
        .rename_axis("external_identity_v1")
        .reset_index(name="external_identity_image_count_v1")
    )
    manifest_df = manifest_df.merge(identity_counts, on="external_identity_v1", how="left")
    manifest_df["external_identity_image_count_v1"] = (
        pd.to_numeric(manifest_df["external_identity_image_count_v1"], errors="coerce").fillna(0).astype(int)
    )
    manifest_df["external_identity_multi_image_v1"] = manifest_df["external_identity_image_count_v1"].ge(2)
    manifest_df["supervised_warmup_candidate_v1"] = (
        manifest_df["filename_parse_ok_v1"].astype(bool) & manifest_df["external_identity_multi_image_v1"].astype(bool)
    )

    manifest_df["chip_path_v1"] = manifest_df["chip_path"].map(lambda value: _to_repo_relative(repo_root, value))
    manifest_df["chip_filename_v1"] = manifest_df["chip_filename"].fillna("").astype(str)
    manifest_df["original_image_name_mapped_v1"] = manifest_df["original_image_name_mapped_v1"].fillna("").astype(str)
    manifest_df["chip_parse_ok_v1"] = manifest_df["chip_parse_ok_v1"].fillna(False).astype(bool)
    manifest_df["filename_parse_ok_v1"] = manifest_df["filename_parse_ok_v1"].fillna(False).astype(bool)

    ordered_columns = [
        "chip_id",
        "chip_filename_v1",
        "chip_path_v1",
        "mapping_presence_v1",
        "chip_parse_ok_v1",
        "original_image_name_mapped_v1",
        "original_match_filename_v1",
        "original_match_path_v1",
        "original_match_stage_v1",
        "original_match_ok_v1",
        "capture_year_v1",
        "capture_token_v1",
        "external_identity_v1",
        "filename_parse_ok_v1",
        "external_identity_image_count_v1",
        "external_identity_multi_image_v1",
        "supervised_warmup_candidate_v1",
        "hotspotter_split_role_v1",
        "hotspotter_query_result_v1",
        "hotspotter_rank1_chip_id_v1",
        "hotspotter_rank1_score_v1",
        "hotspotter_rank2_chip_id_v1",
        "hotspotter_rank2_score_v1",
    ]
    return manifest_df[ordered_columns].sort_values("chip_id", key=lambda series: series.astype(int)).reset_index(drop=True)


def _build_original_manifest(*, repo_root: Path, original_df: pd.DataFrame, chip_manifest_df: pd.DataFrame) -> pd.DataFrame:
    coverage_rows = []
    for original_filename, group in chip_manifest_df.groupby("original_match_filename_v1", dropna=False, sort=True):
        if not str(original_filename).strip():
            continue
        coverage_rows.append(
            {
                "original_filename_v1": str(original_filename),
                "mapped_chip_count_v1": int(len(group)),
                "mapped_chip_ids_v1": "|".join(group["chip_id"].astype(str).tolist()),
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    if coverage_df.empty:
        coverage_df = pd.DataFrame(columns=["original_filename_v1", "mapped_chip_count_v1", "mapped_chip_ids_v1"])

    manifest_df = original_df.merge(coverage_df, on="original_filename_v1", how="left")
    manifest_df["mapped_chip_count_v1"] = pd.to_numeric(manifest_df["mapped_chip_count_v1"], errors="coerce").fillna(0).astype(int)
    manifest_df["mapped_chip_ids_v1"] = manifest_df["mapped_chip_ids_v1"].fillna("").astype(str)
    manifest_df["is_mapped_from_chip_v1"] = manifest_df["mapped_chip_count_v1"].ge(1)
    manifest_df["original_path_v1"] = manifest_df["original_path_v1"].map(lambda value: _to_repo_relative(repo_root, value))
    ordered_columns = [
        "original_filename_v1",
        "original_path_v1",
        "capture_year_v1",
        "capture_token_v1",
        "external_identity_v1",
        "filename_parse_ok_v1",
        "mapped_chip_count_v1",
        "mapped_chip_ids_v1",
        "is_mapped_from_chip_v1",
    ]
    return manifest_df[ordered_columns].sort_values("original_filename_v1").reset_index(drop=True)


def _build_summary_lines(
    *,
    chip_manifest_rel_path: str,
    original_manifest_rel_path: str,
    chip_alignment_audit_rel_path: str,
    original_coverage_audit_rel_path: str,
    counts_df: pd.DataFrame,
    alignment_df: pd.DataFrame,
    identity_df: pd.DataFrame,
    chip_audit_df: pd.DataFrame,
    original_audit_df: pd.DataFrame,
) -> list[str]:
    return [
        "# TCU Texas Dataset Audit",
        "",
        "## Outputs",
        "",
        f"- Chip manifest: `{chip_manifest_rel_path}`",
        f"- Original manifest: `{original_manifest_rel_path}`",
        f"- Chip alignment audit: `{chip_alignment_audit_rel_path}`",
        f"- Original coverage audit: `{original_coverage_audit_rel_path}`",
        "",
        "## Asset Counts",
        "",
        dataframe_to_markdown_table(counts_df),
        "",
        "## Chip To Original Alignment",
        "",
        dataframe_to_markdown_table(alignment_df),
        "",
        "## External Identity Multiplicity",
        "",
        dataframe_to_markdown_table(identity_df),
        "",
        "## Chip Audit Preview",
        "",
        dataframe_to_markdown_table(chip_audit_df),
        "",
        "## Original Audit Preview",
        "",
        dataframe_to_markdown_table(original_audit_df),
        "",
    ]


def build_tcu_texas_dataset_artifacts(
    *,
    repo_root: Path,
    chips_dir: Path,
    original_dir: Path,
    mapping_path: Path,
    hotspotter_output_path: Path,
    output_dir: Path,
) -> TcuTexasDatasetArtifacts:
    repo_root = repo_root.resolve()
    chips_dir = chips_dir.resolve()
    original_dir = original_dir.resolve()
    mapping_path = mapping_path.resolve()
    hotspotter_output_path = hotspotter_output_path.resolve()
    output_dir = output_dir.resolve()

    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    chips_df = _load_chip_files(chips_dir)
    mapping_df = _load_mapping(mapping_path)
    hotspotter_df = _load_hotspotter_output(hotspotter_output_path)
    original_df = _load_original_files(original_dir)
    exact_lookup, normalized_lookup, stem_lookup = _build_original_match_lookup(original_dir)

    chip_manifest_df = _build_chip_manifest(
        repo_root=repo_root,
        chips_df=chips_df,
        mapping_df=mapping_df,
        hotspotter_df=hotspotter_df,
        exact_lookup=exact_lookup,
        normalized_lookup=normalized_lookup,
        stem_lookup=stem_lookup,
    )
    original_manifest_df = _build_original_manifest(
        repo_root=repo_root,
        original_df=original_df,
        chip_manifest_df=chip_manifest_df,
    )

    chip_alignment_audit_df = chip_manifest_df[
        chip_manifest_df["original_match_stage_v1"].ne("exact")
        | chip_manifest_df["mapping_presence_v1"].ne("mapping_and_chip")
        | chip_manifest_df["filename_parse_ok_v1"].eq(False)
    ].copy()
    original_coverage_audit_df = original_manifest_df[
        original_manifest_df["mapped_chip_count_v1"].ne(1) | original_manifest_df["filename_parse_ok_v1"].eq(False)
    ].copy()

    chip_manifest_path = tables_dir / "tcu_texas_chip_manifest_v1.csv"
    original_manifest_path = tables_dir / "tcu_texas_original_manifest_v1.csv"
    chip_alignment_audit_path = tables_dir / "tcu_texas_chip_alignment_audit_v1.csv"
    original_coverage_audit_path = tables_dir / "tcu_texas_original_coverage_audit_v1.csv"

    chip_manifest_df.to_csv(chip_manifest_path, index=False)
    original_manifest_df.to_csv(original_manifest_path, index=False)
    chip_alignment_audit_df.to_csv(chip_alignment_audit_path, index=False)
    original_coverage_audit_df.to_csv(original_coverage_audit_path, index=False)

    counts_df = pd.DataFrame(
        [
            {
                "chip_rows": int(len(chip_manifest_df)),
                "original_rows": int(len(original_manifest_df)),
                "warmup_candidate_chips": int(chip_manifest_df["supervised_warmup_candidate_v1"].sum()),
                "multi_image_external_ids": int(
                    chip_manifest_df.loc[
                        chip_manifest_df["external_identity_multi_image_v1"].astype(bool), "external_identity_v1"
                    ].nunique()
                ),
            }
        ]
    )
    alignment_df = (
        chip_manifest_df["original_match_stage_v1"].value_counts(dropna=False).rename_axis("match_stage").reset_index(name="chips")
    )
    identity_df = (
        chip_manifest_df.loc[chip_manifest_df["external_identity_v1"].astype(str).ne(""), ["external_identity_v1", "external_identity_image_count_v1"]]
        .drop_duplicates()
        .sort_values(["external_identity_image_count_v1", "external_identity_v1"], ascending=[False, True])
        .head(15)
        .reset_index(drop=True)
    )
    summary_path = reports_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            _build_summary_lines(
                chip_manifest_rel_path=_to_repo_relative(repo_root, chip_manifest_path),
                original_manifest_rel_path=_to_repo_relative(repo_root, original_manifest_path),
                chip_alignment_audit_rel_path=_to_repo_relative(repo_root, chip_alignment_audit_path),
                original_coverage_audit_rel_path=_to_repo_relative(repo_root, original_coverage_audit_path),
                counts_df=counts_df,
                alignment_df=alignment_df,
                identity_df=identity_df,
                chip_audit_df=chip_alignment_audit_df.head(20).reset_index(drop=True),
                original_audit_df=original_coverage_audit_df.head(20).reset_index(drop=True),
            )
        ),
        encoding="utf-8",
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(
            {
                "chip_manifest_path": _to_repo_relative(repo_root, chip_manifest_path),
                "original_manifest_path": _to_repo_relative(repo_root, original_manifest_path),
                "chip_alignment_audit_path": _to_repo_relative(repo_root, chip_alignment_audit_path),
                "original_coverage_audit_path": _to_repo_relative(repo_root, original_coverage_audit_path),
                "chip_rows": int(len(chip_manifest_df)),
                "original_rows": int(len(original_manifest_df)),
                "alignment_stage_counts": alignment_df.to_dict(orient="records"),
                "warmup_candidate_chips": int(chip_manifest_df["supervised_warmup_candidate_v1"].sum()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return TcuTexasDatasetArtifacts(
        chip_manifest_path=chip_manifest_path,
        original_manifest_path=original_manifest_path,
        chip_alignment_audit_path=chip_alignment_audit_path,
        original_coverage_audit_path=original_coverage_audit_path,
        summary_path=summary_path,
    )
