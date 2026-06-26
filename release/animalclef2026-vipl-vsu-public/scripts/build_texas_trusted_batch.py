#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_class_indices(raw_values: list[str] | None) -> list[int] | None:
    if not raw_values:
        return None
    return [int(value) for value in raw_values]


def _parse_exclusions(raw_values: list[str] | None) -> dict[int, set[str]] | None:
    if not raw_values:
        return None
    parsed: dict[int, set[str]] = {}
    for item in raw_values:
        if ":" not in item:
            raise ValueError(f"Expected CLASS:IMAGE_ID exclusion, got: {item}")
        class_index_text, image_id = item.split(":", 1)
        parsed.setdefault(int(class_index_text), set()).add(str(image_id))
    return parsed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.texas_trusted_batch import (
        DEFAULT_APPROVED_CLASS_INDICES,
        DEFAULT_CLASS_EXCLUSIONS,
        compile_texas_trusted_batch,
    )

    parser = argparse.ArgumentParser(description="Compile trusted Texas components from manual yes-pairs and approved seed classes.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--manual-pairs-path",
        type=Path,
        default=repo_root
        / "artifacts"
        / "analysis"
        / "manual_review_sessions"
        / "merged"
        / "manual_pair_review_merged_trainable_pairs_20260414_221733.csv",
    )
    parser.add_argument(
        "--review-package-dir",
        type=Path,
        default=repo_root / "artifacts" / "training" / "cache" / "texas_pseudo_seed_centerbody_repaired_v1" / "review_package",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "texas_trusted_batch_v1",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=repo_root / "metadata.csv",
    )
    parser.add_argument(
        "--approved-class-indices",
        nargs="+",
        default=[str(value) for value in DEFAULT_APPROVED_CLASS_INDICES],
        help="1-based class indices in the review package contact-sheet order.",
    )
    parser.add_argument(
        "--exclude-member",
        action="append",
        default=[
            f"{class_index}:{image_id}"
            for class_index, image_ids in DEFAULT_CLASS_EXCLUSIONS.items()
            for image_id in sorted(image_ids)
        ],
        help="Explicit member exclusion in CLASS:IMAGE_ID format. Can be repeated.",
    )
    args = parser.parse_args()

    artifacts = compile_texas_trusted_batch(
        repo_root=args.repo_root.resolve(),
        manual_pairs_path=args.manual_pairs_path.resolve(),
        review_package_dir=args.review_package_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        approved_class_indices=_parse_class_indices(args.approved_class_indices),
        class_exclusions=_parse_exclusions(args.exclude_member),
        metadata_path=args.metadata_path.resolve(),
    )
    print(f"[texas_trusted_batch] trusted_membership: {artifacts.trusted_membership_path}")
    print(f"[texas_trusted_batch] trusted_pairs: {artifacts.trusted_pairs_path}")
    print(f"[texas_trusted_batch] trusted_components: {artifacts.trusted_components_path}")
    print(f"[texas_trusted_batch] summary: {artifacts.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
