#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torchvision.transforms as T


DEFAULT_ROUTE_DIR = Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1")
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_official_wildfusion_probe_20260331")
DEFAULT_THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
SALAMANDER_DATASET = "SalamanderID2025"


@dataclass(frozen=True)
class SplitBundle:
    calib_df: pd.DataFrame
    calib_indices: np.ndarray
    eval_df: pd.DataFrame
    eval_indices: np.ndarray


class PrecomputedEmbeddingExtractor:
    def __init__(self, feature_by_image_id: dict[str, np.ndarray]) -> None:
        self.feature_by_image_id = feature_by_image_id

    def __call__(self, image_dataset):
        from wildlife_tools.data.dataset import FeatureDataset

        metadata = image_dataset.metadata.reset_index(drop=True).copy()
        features: list[np.ndarray] = []
        for image_id in metadata["image_id"].astype(str).tolist():
            if image_id not in self.feature_by_image_id:
                raise KeyError(f"Missing precomputed embedding for image_id={image_id}")
            features.append(self.feature_by_image_id[image_id])
        return FeatureDataset(
            features=np.stack(features).astype(np.float32),
            metadata=metadata,
            col_label="identity",
            load_label=True,
        )


class PairAwareCosineSimilarity:
    def __call__(self, query, database, pairs: np.ndarray | None = None) -> np.ndarray:
        query_features = np.asarray(query.features, dtype=np.float32)
        database_features = np.asarray(database.features, dtype=np.float32)
        if pairs is None:
            return query_features @ database_features.T
        score = np.full((len(query), len(database)), np.nan, dtype=np.float32)
        pair_array = np.asarray(pairs, dtype=np.int32)
        if len(pair_array) == 0:
            return score
        score[pair_array[:, 0], pair_array[:, 1]] = np.sum(
            query_features[pair_array[:, 0]] * database_features[pair_array[:, 1]],
            axis=1,
        )
        return score


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(
        ["ari", "pairwise_f1", "nmi", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]


def _build_image_dataset(metadata_df: pd.DataFrame, repo_root: Path):
    from wildlife_tools.data.dataset import ImageDataset

    transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])
    return ImageDataset(
        metadata=metadata_df.copy(),
        root=str(repo_root),
        transform=transform,
        col_path="path",
        col_label="identity",
        load_label=True,
    )


def _build_identity_split(metadata_df: pd.DataFrame, calib_identity_fraction: float, seed: int) -> SplitBundle:
    identity_counts = metadata_df.groupby("identity").size().sort_index()
    multi_ids = identity_counts[identity_counts > 1].index.tolist()
    single_ids = identity_counts[identity_counts == 1].index.tolist()
    rng = random.Random(seed)
    rng.shuffle(multi_ids)
    rng.shuffle(single_ids)

    calib_multi_count = min(len(multi_ids) - 1, max(5, int(round(len(multi_ids) * calib_identity_fraction))))
    calib_multi_count = max(1, calib_multi_count)
    calib_single_count = int(round(len(single_ids) * calib_identity_fraction))

    calib_ids = set(multi_ids[:calib_multi_count]) | set(single_ids[:calib_single_count])
    calib_mask = metadata_df["identity"].isin(calib_ids).to_numpy()
    eval_mask = ~calib_mask
    if calib_mask.sum() == 0 or eval_mask.sum() == 0:
        raise ValueError("Failed to build a non-empty calibration/eval split.")
    if metadata_df.loc[calib_mask, "identity"].nunique() < 2 or metadata_df.loc[eval_mask, "identity"].nunique() < 2:
        raise ValueError("Calibration/eval split needs at least two identities on both sides.")
    return SplitBundle(
        calib_df=metadata_df.loc[calib_mask].reset_index(drop=True),
        calib_indices=np.flatnonzero(calib_mask),
        eval_df=metadata_df.loc[eval_mask].reset_index(drop=True),
        eval_indices=np.flatnonzero(eval_mask),
    )


def _resolve_metadata_paths(metadata_df: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    from animalclef_analysis.orb_rerank_baseline import resolve_existing_image_rel_path

    resolved = metadata_df.copy()
    resolved["image_id"] = resolved["image_id"].astype(str)
    resolved["identity"] = resolved["identity"].fillna("").astype(str)
    resolved["path"] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in resolved.iterrows()]
    return resolved


def _candidate_pairs_to_array(pair_index: list[tuple[int, int, float]]) -> np.ndarray:
    if not pair_index:
        return np.empty((0, 2), dtype=np.int32)
    return np.asarray([[int(left), int(right)] for left, right, _score in pair_index], dtype=np.int32)


def _score_pairs_to_frame(
    pair_index: list[tuple[int, int, float]],
    pair_score_matrix: np.ndarray,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left_index, right_index, global_score in pair_index:
        fused_score = float(pair_score_matrix[left_index, right_index])
        left_row = metadata_df.iloc[left_index]
        right_row = metadata_df.iloc[right_index]
        rows.append(
            {
                "left_index": int(left_index),
                "right_index": int(right_index),
                "image_id": str(left_row["image_id"]),
                "neighbor_image_id": str(right_row["image_id"]),
                "identity": str(left_row["identity"]),
                "neighbor_identity": str(right_row["identity"]),
                "same_identity": bool(str(left_row["identity"]) == str(right_row["identity"])),
                "baseline_global_score": round(float(global_score), 6),
                "wildfusion_score": round(fused_score, 6),
                "score_delta": round(fused_score - float(global_score), 6),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from animalclef_analysis.descriptor_baselines import dataframe_to_markdown_table
    from animalclef_analysis.orb_rerank_baseline import (
        build_topk_pair_index,
        cosine_score_matrix,
        evaluate_threshold_sweep_from_score_matrix,
        recall_at_k_from_score_matrix,
        score_matrix_to_distance,
    )
    from wildlife_tools.features.local import AlikedExtractor
    from wildlife_tools.similarity.calibration import IsotonicCalibration
    from wildlife_tools.similarity.pairwise.collectors import CollectCounts
    from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
    from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion

    parser = argparse.ArgumentParser(description="Run an official wildlife_tools WildFusion probe on Salamander validation data.")
    parser.add_argument("--route-dir", type=Path, default=DEFAULT_ROUTE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--calib-identity-fraction", type=float, default=0.4)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--shortlist-k", type=int, default=10)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-num-keypoints", type=int, default=256)
    parser.add_argument("--lightglue-init-threshold", type=float, default=0.1)
    parser.add_argument("--lightglue-batch-size", type=int, default=32)
    parser.add_argument("--lightglue-count-threshold", type=float, default=0.2)
    args = parser.parse_args()

    route_dir = args.route_dir.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    metadata_df = pd.read_csv(route_dir / "embeddings" / "salamander_val_metadata.csv")
    metadata_df = _resolve_metadata_paths(metadata_df=metadata_df, repo_root=repo_root)
    embeddings = np.load(route_dir / "embeddings" / "salamander_val_embeddings.npy").astype(np.float32)
    if len(metadata_df) != len(embeddings):
        raise ValueError("Route embeddings do not match Salamander validation metadata rows.")

    split_bundle = _build_identity_split(
        metadata_df=metadata_df,
        calib_identity_fraction=float(args.calib_identity_fraction),
        seed=int(args.split_seed),
    )
    split_df = metadata_df[["image_id", "identity", "path"]].copy()
    split_df["probe_split"] = "eval"
    split_df.loc[split_bundle.calib_indices, "probe_split"] = "calib"
    split_df.to_csv(tables_dir / "split_assignments_v1.csv", index=False)

    feature_by_image_id = {
        str(image_id): embeddings[index].astype(np.float32, copy=False)
        for index, image_id in enumerate(metadata_df["image_id"].astype(str).tolist())
    }
    eval_embeddings = embeddings[split_bundle.eval_indices]
    eval_global_score = cosine_score_matrix(eval_embeddings)
    eval_labels = split_bundle.eval_df["identity"].to_numpy()
    eval_pair_index = build_topk_pair_index(score_matrix=eval_global_score, top_k=int(args.shortlist_k), query_indices=None)
    eval_pair_array = _candidate_pairs_to_array(eval_pair_index)

    calib_image_ds = _build_image_dataset(split_bundle.calib_df, repo_root=repo_root)
    eval_image_ds = _build_image_dataset(split_bundle.eval_df, repo_root=repo_root)

    global_pipeline = SimilarityPipeline(
        matcher=PairAwareCosineSimilarity(),
        extractor=PrecomputedEmbeddingExtractor(feature_by_image_id=feature_by_image_id),
        calibration=IsotonicCalibration(),
    )
    local_pipeline = SimilarityPipeline(
        matcher=MatchLightGlue(
            features="aliked",
            init_threshold=float(args.lightglue_init_threshold),
            device=str(args.device),
            batch_size=int(args.lightglue_batch_size),
            num_workers=0,
            collector=CollectCounts(thresholds=(float(args.lightglue_count_threshold),)),
            tqdm_silent=False,
        ),
        extractor=AlikedExtractor(device=str(args.device), max_num_keypoints=int(args.max_num_keypoints)),
        calibration=IsotonicCalibration(),
    )
    wildfusion = WildFusion(
        calibrated_pipelines=[global_pipeline, local_pipeline],
        priority_pipeline=global_pipeline,
    )

    wildfusion.fit_calibration(calib_image_ds, calib_image_ds)
    pair_score_matrix = wildfusion(eval_image_ds, eval_image_ds, pairs=eval_pair_array)

    fused_score = eval_global_score.copy().astype(np.float32, copy=True)
    for left_index, right_index, _baseline_score in eval_pair_index:
        candidate_score = float(pair_score_matrix[left_index, right_index])
        if not np.isfinite(candidate_score):
            continue
        fused_score[left_index, right_index] = candidate_score
        fused_score[right_index, left_index] = candidate_score
    np.fill_diagonal(fused_score, 1.0)

    baseline_sweep_df, baseline_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=eval_global_score,
        thresholds=thresholds,
    )
    wildfusion_sweep_df, wildfusion_pred_df = evaluate_threshold_sweep_from_score_matrix(
        df=split_bundle.eval_df,
        score_matrix=fused_score,
        thresholds=thresholds,
    )
    baseline_best = _pick_best_row(baseline_sweep_df)
    wildfusion_best = _pick_best_row(wildfusion_sweep_df)

    baseline_sweep_df.to_csv(tables_dir / "baseline_threshold_sweep_v1.csv", index=False)
    wildfusion_sweep_df.to_csv(tables_dir / "wildfusion_threshold_sweep_v1.csv", index=False)
    _score_pairs_to_frame(
        pair_index=eval_pair_index,
        pair_score_matrix=pair_score_matrix,
        metadata_df=split_bundle.eval_df,
    ).to_csv(tables_dir / "wildfusion_pair_scores_v1.csv", index=False)
    baseline_pred_df.to_csv(tables_dir / "baseline_predictions_v1.csv", index=False)
    wildfusion_pred_df.to_csv(tables_dir / "wildfusion_predictions_v1.csv", index=False)

    summary_rows = [
        {
            "route": "baseline_global",
            "threshold": float(baseline_best["threshold"]),
            "ari": float(baseline_best["ari"]),
            "nmi": float(baseline_best["nmi"]),
            "pairwise_f1": float(baseline_best["pairwise_f1"]),
            "cluster_count": int(baseline_best["cluster_count"]),
            "recall_at_1": float(recall_at_k_from_score_matrix(eval_global_score, eval_labels, k=1)),
            "recall_at_5": float(recall_at_k_from_score_matrix(eval_global_score, eval_labels, k=5)),
        },
        {
            "route": "official_wildfusion",
            "threshold": float(wildfusion_best["threshold"]),
            "ari": float(wildfusion_best["ari"]),
            "nmi": float(wildfusion_best["nmi"]),
            "pairwise_f1": float(wildfusion_best["pairwise_f1"]),
            "cluster_count": int(wildfusion_best["cluster_count"]),
            "recall_at_1": float(recall_at_k_from_score_matrix(fused_score, eval_labels, k=1)),
            "recall_at_5": float(recall_at_k_from_score_matrix(fused_score, eval_labels, k=5)),
        },
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df["ari_delta_vs_baseline"] = np.round(summary_df["ari"] - float(baseline_best["ari"]), 6)
    summary_df["pairwise_f1_delta_vs_baseline"] = np.round(summary_df["pairwise_f1"] - float(baseline_best["pairwise_f1"]), 6)
    summary_df.to_csv(tables_dir / "comparison_summary_v1.csv", index=False)

    summary = {
        "probe": "salamander_official_wildfusion_probe",
        "date": "2026-03-31",
        "dataset": SALAMANDER_DATASET,
        "route_dir": str(route_dir),
        "device": str(args.device),
        "val_image_count": int(len(metadata_df)),
        "calib_image_count": int(len(split_bundle.calib_df)),
        "eval_image_count": int(len(split_bundle.eval_df)),
        "calib_identity_count": int(split_bundle.calib_df["identity"].nunique()),
        "eval_identity_count": int(split_bundle.eval_df["identity"].nunique()),
        "shortlist_k": int(args.shortlist_k),
        "thresholds": [float(value) for value in thresholds],
        "lightglue_init_threshold": float(args.lightglue_init_threshold),
        "lightglue_count_threshold": float(args.lightglue_count_threshold),
        "max_num_keypoints": int(args.max_num_keypoints),
        "comparison_rows": summary_rows,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = [
        "# Salamander Official WildFusion Probe",
        "",
        "- Goal: test official `wildlife_tools.WildFusion` on top of the current best fine-tuned Salamander global branch.",
        f"- Route dir: `{route_dir}`",
        f"- Probe split: validation-only internal split with `calib_identity_fraction={float(args.calib_identity_fraction)}` and `seed={int(args.split_seed)}`.",
        f"- Calibration size: `{len(split_bundle.calib_df)}` images / `{split_bundle.calib_df['identity'].nunique()}` identities.",
        f"- Eval size: `{len(split_bundle.eval_df)}` images / `{split_bundle.eval_df['identity'].nunique()}` identities.",
        f"- Local matcher: `ALIKED + LightGlue` with `CollectCounts@{float(args.lightglue_count_threshold)}`, shortlist top-`{int(args.shortlist_k)}` candidate pairs.",
        "",
        "## Comparison",
        "",
        dataframe_to_markdown_table(
            summary_df[
                [
                    "route",
                    "threshold",
                    "ari",
                    "nmi",
                    "pairwise_f1",
                    "cluster_count",
                    "recall_at_1",
                    "recall_at_5",
                    "ari_delta_vs_baseline",
                    "pairwise_f1_delta_vs_baseline",
                ]
            ]
        ),
        "",
        "## Reading Note",
        "",
        "- This probe uses an internal calibration/eval split inside the Salamander validation fold, so it is only a direction check.",
        "- For clustering, only the shortlist pairs are replaced by official WildFusion scores; all other edges fall back to the baseline global cosine matrix.",
        f"- Best baseline threshold: `{float(baseline_best['threshold'])}`; best WildFusion threshold: `{float(wildfusion_best['threshold'])}`.",
    ]
    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[salamander_official_wildfusion_probe] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_official_wildfusion_probe] comparison: {tables_dir / 'comparison_summary_v1.csv'}")
    print(f"[salamander_official_wildfusion_probe] pairs: {tables_dir / 'wildfusion_pair_scores_v1.csv'}")
    print(f"[salamander_official_wildfusion_probe] distance_check: {score_matrix_to_distance(fused_score).shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
