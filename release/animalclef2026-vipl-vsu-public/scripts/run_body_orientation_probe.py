from __future__ import annotations

import argparse
from pathlib import Path

from src.animalclef_analysis.body_orientation_probe import PROMPTS_BY_DATASET, run_body_orientation_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run body-axis orientation probe on sampled AnimalCLEF images.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(PROMPTS_BY_DATASET),
        help="Datasets to sample and probe.",
    )
    parser.add_argument("--samples-per-split", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--min-foreground-pixels", type=int, default=1024)
    parser.add_argument("--min-area-ratio", type=float, default=0.015)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--min-axis-confidence", type=float, default=0.35)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.8)
    parser.add_argument("--aligned-crop-padding-ratio", type=float, default=0.06)
    parser.add_argument("--texas-aligned-crop-padding-ratio", type=float, default=0.12)
    parser.add_argument(
        "--rotation-canvas-fill-mode",
        type=str,
        default="edge",
        choices=["edge", "reflect", "constant"],
        help="How aligned RGB exports fill pixels introduced by rotation.",
    )
    parser.add_argument(
        "--mask-background",
        action="store_true",
        help="Zero out non-mask pixels in the aligned export instead of preserving RGB background.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_body_orientation_probe(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        datasets=args.datasets,
        samples_per_split=args.samples_per_split,
        sample_seed=args.sample_seed,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
        min_foreground_pixels=args.min_foreground_pixels,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        min_axis_confidence=args.min_axis_confidence,
        min_largest_component_ratio=args.min_largest_component_ratio,
        aligned_crop_padding_ratio=args.aligned_crop_padding_ratio,
        aligned_crop_padding_ratio_overrides={"TexasHornedLizards": args.texas_aligned_crop_padding_ratio},
        keep_background=not args.mask_background,
        rotation_canvas_fill_mode=args.rotation_canvas_fill_mode,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
