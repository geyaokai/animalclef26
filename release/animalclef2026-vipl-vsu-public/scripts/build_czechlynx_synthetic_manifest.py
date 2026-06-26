#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build a training manifest for CzechLynx synthetic warmup.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=repo_root / "artifacts" / "external_data" / "czechlynx" / "extracted" / "CzechLynxDataset-Metadata-Synthetic.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "external_data" / "czechlynx" / "manifests",
    )
    parser.add_argument("--dataset-name", type=str, default="LynxID2025")
    parser.add_argument("--identity-prefix", type=str, default="CzechLynxSynthetic")
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.view_manifests import PATH_COLUMN

    metadata_path = args.metadata_path.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    source_df = pd.read_csv(metadata_path)
    if args.max_rows is not None and args.max_rows > 0:
        source_df = source_df.iloc[: args.max_rows].copy()

    manifest_df = pd.DataFrame(
        {
            "image_id": [f"{args.identity_prefix}_{index:05d}" for index in range(len(source_df))],
            "identity": args.identity_prefix + "_" + source_df["unique_name"].astype(str),
            "path": source_df["path"].astype(str).map(
                lambda value: str(Path("artifacts") / "external_data" / "czechlynx" / "extracted" / value)
            ),
            "date": "",
            "orientation": "",
            "species": "lynx",
            "split": "train",
            "dataset": args.dataset_name,
            "czechlynx_source_v1": source_df["source"].astype(str),
            "czechlynx_location_v1": source_df["location"].astype(str),
            "czechlynx_unique_name_v1": source_df["unique_name"].astype(str),
            "czechlynx_encounter_v1": source_df["encounter"],
        }
    )
    manifest_df[PATH_COLUMN] = manifest_df["path"]
    manifest_df["local_label_available_v1"] = True
    manifest_df["recommended_train_keep_all_v1"] = True
    manifest_df["recommended_train_keep_dedup_v1"] = True

    missing_paths = [path for path in manifest_df["path"].head(10) if not (repo_root / path).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Example synthetic paths are missing: {missing_paths[:3]}")
    if manifest_df["image_id"].duplicated().any():
        raise ValueError("Synthetic manifest image_id values are not unique")
    if manifest_df["identity"].nunique() < 2:
        raise ValueError("Synthetic manifest needs at least two identities for identity holdout")

    manifest_path = tables_dir / "manifest_train_czechlynx_synthetic_as_lynx_v1.csv"
    manifest_df.to_csv(manifest_path, index=False)

    per_identity = manifest_df.groupby("identity").size()
    summary = {
        "manifest_path": str(manifest_path),
        "rows": int(len(manifest_df)),
        "dataset_name": args.dataset_name,
        "identities": int(manifest_df["identity"].nunique()),
        "min_images_per_identity": int(per_identity.min()),
        "median_images_per_identity": float(per_identity.median()),
        "max_images_per_identity": int(per_identity.max()),
        "source_metadata_path": str(metadata_path),
        "label_policy": "Synthetic identities are prefixed and treated as an independent LynxID2025-compatible warmup label space.",
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "summary.md").write_text(
        "\n".join(
            [
                "# CzechLynx Synthetic Warmup Manifest",
                "",
                f"- Rows: `{summary['rows']}`",
                f"- Dataset name exposed to training: `{summary['dataset_name']}`",
                f"- Synthetic identities: `{summary['identities']}`",
                f"- Images per identity: min=`{summary['min_images_per_identity']}`, median=`{summary['median_images_per_identity']}`, max=`{summary['max_images_per_identity']}`",
                f"- Manifest: `{manifest_path}`",
                "",
                "## Label Policy",
                "",
                "- `CzechLynx_Synthetic` is not mapped to real `lynx_*` identities.",
                "- Each `synthetic_lynx_*` becomes `CzechLynxSynthetic_synthetic_lynx_*`.",
                "- The manifest uses `dataset=LynxID2025` only to reuse Lynx preprocessing and the existing supervised training head.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[czechlynx_synthetic_manifest] manifest: {manifest_path}")
    print(f"[czechlynx_synthetic_manifest] summary: {reports_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
