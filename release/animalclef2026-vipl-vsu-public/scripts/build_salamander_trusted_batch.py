#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.salamander_trusted_batch import compile_salamander_trusted_batch

    parser = argparse.ArgumentParser(description="Compile Salamander trusted components from manual pair judgments.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--pair-judgments",
        type=Path,
        default=repo_root
        / "artifacts"
        / "analysis"
        / "manual_review_sessions"
        / "merged"
        / "manual_pair_review_merged_trainable_20260414_221733.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "artifacts" / "analysis" / "salamander_trusted_batch_v1",
    )
    parser.add_argument("--metadata-path", type=Path, default=repo_root / "metadata.csv")
    parser.add_argument("--dataset", type=str, default="SalamanderID2025")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    pair_judgments_path = args.pair_judgments if args.pair_judgments.is_absolute() else repo_root / args.pair_judgments
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    metadata_path = args.metadata_path if args.metadata_path.is_absolute() else repo_root / args.metadata_path

    artifacts = compile_salamander_trusted_batch(
        repo_root=repo_root,
        pair_judgments_path=pair_judgments_path.resolve(),
        output_dir=output_dir.resolve(),
        metadata_path=metadata_path.resolve(),
        dataset=str(args.dataset),
    )
    print(f"[salamander_trusted_batch] trusted_membership: {artifacts.trusted_membership_path}")
    print(f"[salamander_trusted_batch] clean_trusted_membership: {artifacts.clean_trusted_membership_path}")
    print(f"[salamander_trusted_batch] trusted_pairs: {artifacts.trusted_pairs_path}")
    print(f"[salamander_trusted_batch] clean_trusted_pairs: {artifacts.clean_trusted_pairs_path}")
    print(f"[salamander_trusted_batch] cannot_link_pairs: {artifacts.cannot_link_pairs_path}")
    print(f"[salamander_trusted_batch] trusted_components: {artifacts.trusted_components_path}")
    print(f"[salamander_trusted_batch] clean_trusted_components: {artifacts.clean_trusted_components_path}")
    print(f"[salamander_trusted_batch] conflict_pairs: {artifacts.conflict_pairs_path}")
    print(f"[salamander_trusted_batch] summary: {artifacts.summary_path}")
    print(f"[salamander_trusted_batch] review_html: {artifacts.review_html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
