#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


SALAMANDER_DATASET = "SalamanderID2025"
DEFAULT_OUTPUT_DIR = Path("artifacts/analysis/salamander_candidate_recall_audit_20260420")
DEFAULT_TOPK = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]


@dataclass(frozen=True)
class RouteSpec:
    name: str
    embeddings_path: Path
    metadata_path: Path


DEFAULT_ROUTES = [
    RouteSpec(
        name="frozen_miew",
        embeddings_path=Path("artifacts/descriptor_baselines/embed_miew_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/descriptor_baselines/embed_miew_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="frozen_mega",
        embeddings_path=Path("artifacts/descriptor_baselines/embed_mega_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/descriptor_baselines/embed_mega_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="frozen_fusion",
        embeddings_path=Path("artifacts/descriptor_baselines/embed_fusion_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/descriptor_baselines/embed_fusion_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_miew_distill",
        embeddings_path=Path("artifacts/training/experiments/ft_miew_arcface_distill_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_miew_arcface_distill_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_mega_distill",
        embeddings_path=Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_miew_maskedsupcon",
        embeddings_path=Path("artifacts/training/experiments/ft_miew_arcface_masked_supcon_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_miew_arcface_masked_supcon_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_salamander_subcenter",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_subcenter_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_subcenter_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="route_distill_last_fusionorb",
        embeddings_path=Path("artifacts/submissions/kaggle_variant_salamander_distill_last_fusionorb_v1/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/submissions/kaggle_variant_salamander_distill_last_fusionorb_v1/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="route_maskedsupcon_last_fusionorb",
        embeddings_path=Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/submissions/kaggle_variant_salamander_maskedsupcon_last_fusionorb_v1/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="dualview_v2",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v2/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v2/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="dualview_v3",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v3/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v3/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="dualview_v4_allidrep",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v4_allidrep/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v4_allidrep/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="dualview_v5_e30_val25",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v5_e30_val25/embeddings/salamander_val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_dualview_centertrunk_v5_e30_val25/embeddings/salamander_val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_salamander_miew_nodistill_e30",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_miew_recall_nodistill_e30_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_miew_recall_nodistill_e30_v1/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_salamander_miew_nodistill_e30_best_recall1",
        embeddings_path=Path("artifacts/analysis/ft_salamander_miew_recall_nodistill_e30_v1_best_recall1_eval/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/analysis/ft_salamander_miew_recall_nodistill_e30_v1_best_recall1_eval/embeddings/val_metadata.csv"),
    ),
    RouteSpec(
        name="ft_salamander_mega_nodistill_e30",
        embeddings_path=Path("artifacts/training/experiments/ft_salamander_mega_recall_nodistill_e30_v1/embeddings/val_embeddings.npy"),
        metadata_path=Path("artifacts/training/experiments/ft_salamander_mega_recall_nodistill_e30_v1/embeddings/val_metadata.csv"),
    ),
]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    embeddings = embeddings.astype(np.float32, copy=False)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return embeddings / norm


def filter_salamander_rows(embeddings: np.ndarray, metadata_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    df = metadata_df.copy()
    if "dataset" in df.columns:
        mask = df["dataset"].astype(str).eq(SALAMANDER_DATASET).to_numpy()
        if len(mask) == len(embeddings) and int(mask.sum()) > 0:
            df = df.loc[mask].reset_index(drop=True)
            embeddings = embeddings[mask]
    df["image_id"] = df["image_id"].astype(str)
    df["identity"] = df["identity"].fillna("").astype(str)
    return embeddings, df


def build_score_matrix(embeddings: np.ndarray) -> np.ndarray:
    normalized = normalize_embeddings(embeddings)
    score = normalized @ normalized.T
    np.fill_diagonal(score, -np.inf)
    return score.astype(np.float32, copy=False)


def first_hit_positions(score_matrix: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = pd.Series(labels).value_counts()
    valid_mask = np.array([counts[label] > 1 for label in labels], dtype=bool)
    order = np.argsort(-score_matrix, axis=1)
    first_hit = []
    for index in range(len(labels)):
        if not valid_mask[index]:
            continue
        positions = np.where(labels[order[index]] == labels[index])[0]
        if len(positions):
            first_hit.append(int(positions[0]) + 1)
    return np.array(first_hit, dtype=int), valid_mask


def summarize_first_hits(first_hit: np.ndarray, topk_values: list[int]) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "k@90_recall": int(np.quantile(first_hit, 0.9, method="higher")),
        "median_first_hit_k": int(np.quantile(first_hit, 0.5, method="higher")),
        "p95_first_hit_k": int(np.quantile(first_hit, 0.95, method="higher")),
    }
    for topk in topk_values:
        row[f"candidate_recall_at_top{topk}"] = round(float(np.mean(first_hit <= int(topk))), 6)
    return row


def compute_route_metrics(route: RouteSpec, repo_root: Path, topk_values: list[int]) -> tuple[dict[str, object], pd.DataFrame]:
    embeddings = np.load((repo_root / route.embeddings_path).resolve())
    metadata_df = pd.read_csv((repo_root / route.metadata_path).resolve())
    embeddings, metadata_df = filter_salamander_rows(embeddings=embeddings, metadata_df=metadata_df)
    score_matrix = build_score_matrix(embeddings)
    labels = metadata_df["identity"].to_numpy()
    first_hit, valid_mask = first_hit_positions(score_matrix=score_matrix, labels=labels)
    if len(first_hit) == 0:
        raise ValueError(f"No valid non-singleton Salamander rows for route: {route.name}")
    summary_row: dict[str, object] = {
        "route": route.name,
        "rows": int(len(metadata_df)),
        "identities": int(metadata_df["identity"].nunique()),
        "embeddings_path": str(route.embeddings_path),
        "metadata_path": str(route.metadata_path),
        "split_signature": __import__("hashlib").sha1("||".join(metadata_df["image_id"].tolist()).encode()).hexdigest()[:12],
    }
    summary_row.update(summarize_first_hits(first_hit=first_hit, topk_values=topk_values))
    distribution_df = metadata_df.loc[valid_mask, ["image_id", "identity"]].copy().reset_index(drop=True)
    distribution_df["route"] = route.name
    distribution_df["first_hit_k"] = first_hit
    return summary_row, distribution_df


def build_markdown_summary(summary_df: pd.DataFrame, output_dir: Path, topk_values: list[int]) -> str:
    topk_cols = [f"candidate_recall_at_top{value}" for value in topk_values]
    lines = [
        "# Salamander Candidate Recall Audit",
        "",
        "## Key Table",
        "",
        summary_df[["route", "rows", "identities", "split_signature", "k@90_recall", *topk_cols]].to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- `k@90_recall` = the smallest `top-k` such that at least `90%` of non-singleton Salamander validation queries see a same-identity hit inside the shortlist.",
        "- Routes with different `split_signature` are not on the exact same validation image set; compare them carefully.",
        f"- Output directory: `{output_dir}`",
        "",
    ]
    best_row = summary_df.sort_values(["k@90_recall", f"candidate_recall_at_top{topk_values[2]}"], ascending=[True, False]).iloc[0]
    lines.extend(
        [
            "## Takeaway",
            "",
            f"- Current best route by `k@90_recall`: `{best_row['route']}` with `k@90_recall={int(best_row['k@90_recall'])}`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Salamander candidate recall from cached validation embeddings.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--topk", nargs="+", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / DEFAULT_OUTPUT_DIR if args.output_dir is None else args.output_dir.resolve())
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    distribution_frames: list[pd.DataFrame] = []
    for route in DEFAULT_ROUTES:
        embeddings_path = (repo_root / route.embeddings_path).resolve()
        metadata_path = (repo_root / route.metadata_path).resolve()
        if not embeddings_path.exists() or not metadata_path.exists():
            continue
        summary_row, distribution_df = compute_route_metrics(route=route, repo_root=repo_root, topk_values=args.topk)
        summary_rows.append(summary_row)
        distribution_frames.append(distribution_df)

    if not summary_rows:
        raise FileNotFoundError("No valid Salamander candidate recall routes were found.")

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["k@90_recall", f"candidate_recall_at_top{args.topk[min(2, len(args.topk) - 1)]}"],
        ascending=[True, False],
    )
    distribution_df = pd.concat(distribution_frames, ignore_index=True)
    summary_df.to_csv(tables_dir / "candidate_recall_summary_v1.csv", index=False)
    distribution_df.to_csv(tables_dir / "first_hit_distribution_v1.csv", index=False)

    config = {
        "dataset": SALAMANDER_DATASET,
        "topk": list(args.topk),
        "route_count": int(len(summary_df)),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md = build_markdown_summary(summary_df=summary_df, output_dir=output_dir, topk_values=list(args.topk))
    (reports_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    print(f"[salamander_candidate_recall_audit] summary: {reports_dir / 'summary.md'}")
    print(f"[salamander_candidate_recall_audit] table: {tables_dir / 'candidate_recall_summary_v1.csv'}")
    print(f"[salamander_candidate_recall_audit] distribution: {tables_dir / 'first_hit_distribution_v1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
