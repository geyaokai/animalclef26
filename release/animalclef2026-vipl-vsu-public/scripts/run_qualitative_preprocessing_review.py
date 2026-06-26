#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.qualitative_preprocessing_review import (
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_TEXAS_AUGMENTOR_SAMPLES_PER_IMAGE,
        run_qualitative_preprocessing_review,
    )

    parser = argparse.ArgumentParser(description="Run the stage-1 qualitative preprocessing review pipeline.")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--metadata-path", type=Path, default=repo_root / "metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=repo_root / DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["TexasHornedLizards", "SalamanderID2025", "LynxID2025"],
        help="Datasets to include in the qualitative review.",
    )
    parser.add_argument("--samples-per-split", type=int, default=4)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--sam-threshold", type=float, default=0.5)
    parser.add_argument("--sam-mask-threshold", type=float, default=0.5)
    parser.add_argument("--yolo-model-name", type=str, default="yolov8s-worldv2.pt")
    parser.add_argument("--yolo-conf", type=float, default=0.15)
    parser.add_argument("--yolo-iou", type=float, default=0.5)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-max-det", type=int, default=8)
    parser.add_argument(
        "--texas-augmentor-samples-per-image",
        type=int,
        default=DEFAULT_TEXAS_AUGMENTOR_SAMPLES_PER_IMAGE,
        help="How many Augmentor preview samples to draw per Texas grayscale input. Use 0 to disable.",
    )
    args = parser.parse_args()

    outputs = run_qualitative_preprocessing_review(
        repo_root=args.repo_root.resolve(),
        output_dir=args.output_dir.resolve(),
        metadata_path=args.metadata_path.resolve(),
        datasets=list(args.datasets),
        samples_per_split=int(args.samples_per_split),
        sample_seed=int(args.sample_seed),
        device=str(args.device),
        sam_threshold=float(args.sam_threshold),
        sam_mask_threshold=float(args.sam_mask_threshold),
        yolo_model_name=str(args.yolo_model_name),
        yolo_conf=float(args.yolo_conf),
        yolo_iou=float(args.yolo_iou),
        yolo_imgsz=int(args.yolo_imgsz),
        yolo_max_det=int(args.yolo_max_det),
        texas_augmentor_samples_per_image=int(args.texas_augmentor_samples_per_image),
    )
    print(f"[qualitative_preprocessing_review] summary: {outputs['summary_path']}")
    print(f"[qualitative_preprocessing_review] records: {outputs['records_path']}")
    if outputs["manifest_paths"]:
        for dataset, path in sorted(outputs["manifest_paths"].items()):
            print(f"[qualitative_preprocessing_review] manifest {dataset}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
