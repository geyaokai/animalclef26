#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Apply manual Texas pair overrides and export pair registry v2.")
    parser.add_argument(
        "--base-registry-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v1/texas_pair_registry_v1.csv",
    )
    parser.add_argument(
        "--override-path",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v2/manual_pair_overrides_v1.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts/analysis/texas_pair_registry_v2",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.base_registry_path.resolve())
    override_df = pd.read_csv(args.override_path.resolve())

    base_df["pair_id"] = base_df["pair_id"].astype(str)
    override_df["pair_id"] = override_df["pair_id"].astype(str)
    override_pair_ids = set(override_df["pair_id"].tolist())

    kept_df = base_df[~base_df["pair_id"].isin(override_pair_ids)].copy()
    override_df = override_df.copy()
    override_df["had_conflict"] = override_df.get("had_conflict", False)
    override_df = override_df[
        [
            "pair_id",
            "image_id_a",
            "image_id_b",
            "constraint_type",
            "support_count",
            "sources",
            "review_batches",
            "notes",
            "had_conflict",
        ]
    ]

    merged_df = pd.concat([kept_df, override_df], ignore_index=True)
    merged_df["image_id_a"] = pd.to_numeric(merged_df["image_id_a"], errors="coerce").fillna(-1).astype(int)
    merged_df["image_id_b"] = pd.to_numeric(merged_df["image_id_b"], errors="coerce").fillna(-1).astype(int)
    merged_df["support_count"] = pd.to_numeric(merged_df["support_count"], errors="coerce").fillna(0).astype(int)
    merged_df["constraint_type"] = merged_df["constraint_type"].astype(str)
    merged_df["sources"] = merged_df["sources"].astype(str)
    merged_df["review_batches"] = merged_df["review_batches"].astype(str)
    merged_df["notes"] = merged_df["notes"].astype(str)
    merged_df["had_conflict"] = merged_df["had_conflict"].fillna(False).astype(bool)
    merged_df = merged_df.sort_values(["constraint_type", "image_id_a", "image_id_b"], ascending=[True, True, True]).reset_index(drop=True)

    output_path = output_dir / "texas_pair_registry_v2.csv"
    merged_df.to_csv(output_path, index=False)

    summary = {
        "base_registry_path": str(args.base_registry_path.resolve()),
        "override_path": str(args.override_path.resolve()),
        "output_path": str(output_path),
        "base_rows": int(len(base_df)),
        "override_rows": int(len(override_df)),
        "output_rows": int(len(merged_df)),
        "must_link_rows": int(merged_df["constraint_type"].eq("must-link").sum()),
        "cannot_link_rows": int(merged_df["constraint_type"].eq("cannot-link").sum()),
        "overridden_pairs": sorted(override_pair_ids),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[texas_pair_registry_v2] output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
