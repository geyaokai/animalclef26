from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised in runtime env
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from .descriptor_baselines import PATH_COLUMN, build_identity_holdout_split, dataframe_to_markdown_table, load_manifests
from .initial_audit import create_contact_sheet
from .labeled_selftrain import (
    build_stable_pseudo_seed_bundle,
    build_teacher_embeddings,
    parse_teacher_sources,
)
from .supervised_training import collect_resource_snapshot, seed_everything


def _require_matplotlib() -> None:
    if plt is None:
        raise ModuleNotFoundError("matplotlib is required for anchor_seed_audit")


def _threshold_tag(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    if values.empty or float(weights.sum()) <= 0:
        return 0.0
    return float(np.average(values.astype(float).to_numpy(), weights=weights.astype(float).to_numpy()))


def _build_anchor_summary_row(
    *,
    anchor_threshold: float,
    cluster_summary_df: pd.DataFrame,
    teacher_anchor_metrics: dict[str, float],
    seed_image_count: int,
    target_size: int,
) -> dict[str, float | int]:
    accepted = cluster_summary_df[cluster_summary_df["accepted_as_seed"]].copy().reset_index(drop=True)
    seed_coverage_ratio = float(seed_image_count / target_size) if target_size else 0.0
    if accepted.empty:
        weighted_seed_purity = 0.0
        mean_seed_purity = 0.0
        min_seed_purity = 0.0
        mean_seed_similarity = 0.0
        pure_cluster_ratio = 0.0
        clean_seed_coverage = 0.0
    else:
        weights = accepted["size"].astype(float)
        weighted_seed_purity = _weighted_mean(accepted["purity_vs_truth"], weights)
        mean_seed_purity = float(accepted["purity_vs_truth"].mean())
        min_seed_purity = float(accepted["purity_vs_truth"].min())
        mean_seed_similarity = _weighted_mean(accepted["mean_similarity"], weights)
        pure_cluster_ratio = float(accepted["purity_vs_truth"].eq(1.0).mean())
        clean_seed_coverage = seed_coverage_ratio * weighted_seed_purity
    return {
        "anchor_threshold": round(float(anchor_threshold), 4),
        "seed_images": int(seed_image_count),
        "seed_coverage_ratio": round(seed_coverage_ratio, 6),
        "accepted_seed_clusters": int(len(accepted)),
        "weighted_seed_purity": round(weighted_seed_purity, 6),
        "mean_seed_purity": round(mean_seed_purity, 6),
        "min_seed_purity": round(min_seed_purity, 6),
        "mean_seed_similarity": round(mean_seed_similarity, 6),
        "pure_seed_cluster_ratio": round(pure_cluster_ratio, 6),
        "clean_seed_coverage": round(clean_seed_coverage, 6),
        "teacher_anchor_ari": float(teacher_anchor_metrics["ari"]),
        "teacher_anchor_pairwise_f1": float(teacher_anchor_metrics["pairwise_f1"]),
        "teacher_anchor_pairwise_precision": float(teacher_anchor_metrics["pairwise_precision"]),
        "teacher_anchor_pairwise_recall": float(teacher_anchor_metrics["pairwise_recall"]),
        "teacher_anchor_cluster_count": int(teacher_anchor_metrics["cluster_count"]),
        "teacher_anchor_singleton_cluster_ratio": float(teacher_anchor_metrics["singleton_cluster_ratio"]),
    }


def _build_seed_image_table(
    *,
    target_df: pd.DataFrame,
    pseudo_seed_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    seed_df = pseudo_seed_df[pseudo_seed_df["seed_status"] == "seed"].copy().reset_index(drop=True)
    if seed_df.empty:
        return pd.DataFrame()
    keep_columns = [
        "anchor_cluster_id",
        "size",
        "mean_similarity",
        "accepted_as_seed",
        "pseudo_identity",
        "purity_vs_truth",
    ]
    merged = seed_df.merge(
        cluster_summary_df.loc[:, keep_columns],
        on=["anchor_cluster_id", "pseudo_identity"],
        how="left",
        validate="many_to_one",
    )
    merged = merged.merge(
        target_df[["image_id", "identity", PATH_COLUMN]],
        on=["image_id", "identity", PATH_COLUMN],
        how="left",
        validate="one_to_one",
    )
    return merged.sort_values(
        ["purity_vs_truth", "mean_similarity", "pseudo_identity", "image_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def _sample_seed_cluster_rows(
    *,
    seed_image_df: pd.DataFrame,
    max_clusters: int,
    images_per_cluster: int,
) -> pd.DataFrame:
    if seed_image_df.empty:
        return pd.DataFrame(columns=seed_image_df.columns)
    ordered = (
        seed_image_df.groupby("pseudo_identity", as_index=False)
        .agg(
            size=("image_id", "count"),
            purity_vs_truth=("purity_vs_truth", "first"),
            mean_similarity=("mean_similarity", "first"),
        )
        .sort_values(["purity_vs_truth", "mean_similarity", "size", "pseudo_identity"], ascending=[True, True, False, True])
        .head(max_clusters)
    )
    rows: list[pd.DataFrame] = []
    for pseudo_identity in ordered["pseudo_identity"].tolist():
        cluster_df = seed_image_df[seed_image_df["pseudo_identity"] == pseudo_identity].copy().sort_values("image_id")
        rows.append(cluster_df.head(images_per_cluster))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=seed_image_df.columns)


def _sample_suspect_rows(
    *,
    seed_image_df: pd.DataFrame,
    max_images: int,
) -> pd.DataFrame:
    if seed_image_df.empty:
        return pd.DataFrame(columns=seed_image_df.columns)
    return (
        seed_image_df.sort_values(
            ["purity_vs_truth", "mean_similarity", "size", "pseudo_identity", "image_id"],
            ascending=[True, True, False, True, True],
        )
        .head(max_images)
        .reset_index(drop=True)
    )


def _write_anchor_plots(plots_dir: Path, anchor_summary_df: pd.DataFrame) -> dict[str, Path]:
    _require_matplotlib()
    plots_dir.mkdir(parents=True, exist_ok=True)
    x = anchor_summary_df["anchor_threshold"].astype(float).to_numpy()
    paths: dict[str, Path] = {}

    coverage_path = plots_dir / "anchor_seed_coverage.png"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, anchor_summary_df["seed_coverage_ratio"], marker="o", linewidth=2, label="seed coverage")
    ax.plot(x, anchor_summary_df["clean_seed_coverage"], marker="o", linewidth=2, label="clean seed coverage")
    ax.set_title("Anchor vs Seed Coverage")
    ax.set_xlabel("Anchor threshold")
    ax.set_ylabel("Coverage ratio")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(coverage_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["coverage"] = coverage_path

    purity_path = plots_dir / "anchor_seed_quality.png"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, anchor_summary_df["weighted_seed_purity"], marker="o", linewidth=2, label="weighted purity")
    ax.plot(x, anchor_summary_df["mean_seed_similarity"], marker="o", linewidth=2, label="mean similarity")
    ax.plot(x, anchor_summary_df["pure_seed_cluster_ratio"], marker="o", linewidth=2, label="pure cluster ratio")
    ax.set_title("Anchor vs Seed Quality")
    ax.set_xlabel("Anchor threshold")
    ax.set_ylabel("Quality")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(purity_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["quality"] = purity_path

    teacher_path = plots_dir / "anchor_teacher_metrics.png"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(x, anchor_summary_df["teacher_anchor_ari"], marker="o", linewidth=2, label="teacher ARI")
    ax.plot(x, anchor_summary_df["teacher_anchor_pairwise_f1"], marker="o", linewidth=2, label="teacher pairwise F1")
    ax.set_title("Anchor vs Teacher Target Metrics")
    ax.set_xlabel("Anchor threshold")
    ax.set_ylabel("Metric")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(teacher_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["teacher"] = teacher_path
    return paths


def _write_summary(
    *,
    output_dir: Path,
    config: dict[str, Any],
    anchor_summary_df: pd.DataFrame,
    recommended_anchor: float,
    plot_paths: dict[str, Path],
) -> Path:
    reports_dir = output_dir / "reports"
    summary_path = reports_dir / "summary.md"
    recommended_row = anchor_summary_df.loc[anchor_summary_df["anchor_threshold"].eq(recommended_anchor)].iloc[0]
    teacher_line = ", ".join(config["teacher_descriptor_sources"] + config["teacher_checkpoint_sources"])
    lines = [
        "# Anchor / Seed 审计报告",
        "",
        "## 审计目标",
        "",
        f"- `dataset`: `{config['dataset']}`",
        f"- `teacher_sources`: `{teacher_line}`",
        f"- `split_protocol`: `identity-level holdout, target split as pseudo-label audit domain`",
        f"- `target_images / target_ids`: `{config['target_images']} / {config['target_ids']}`",
        f"- `anchors`: `{config['anchors']}`",
        "",
        "## 先说结论",
        "",
        f"- 当前推荐先人工复查的 `anchor` 是 `{recommended_anchor}`。",
        f"- 原因不是它让 student 最强，而是它在当前 strongest embeddable teacher 下，`clean_seed_coverage={float(recommended_row['clean_seed_coverage']):.4f}`、`weighted_seed_purity={float(recommended_row['weighted_seed_purity']):.4f}`、`teacher_anchor_ari={float(recommended_row['teacher_anchor_ari']):.4f}` 的折中最好。",
        "",
        "## 指标怎么读",
        "",
        "- `anchor_threshold`：用这个阈值在 target 图上做 teacher 聚类。它是层次聚类的距离阈值；阈值越高，越容易继续合并，聚类会更激进、簇更大。",
        "- `seed_images`：被接纳成稳定 seed 的图像数。它决定 self-train 时有多少高置信伪标签样本可用。",
        "- `seed_coverage_ratio`：`seed_images / target_images`。它告诉你 target 域有多大比例被伪标签覆盖。",
        "- `weighted_seed_purity`：只看被接纳 seed cluster 的“加权纯度”。可以理解成：随机抽一张 seed 图，它落在正确同一只簇里的概率大概有多高。",
        "- `clean_seed_coverage`：`seed_coverage_ratio * weighted_seed_purity`。它是“覆盖多少、而且大概有多干净”的合成 proxy。",
        "- `mean_seed_similarity`：seed cluster 内部平均 teacher 相似度。越高说明 seed 内部越紧。",
        "- `pure_seed_cluster_ratio`：seed cluster 里有多少比例在本地真标签下是 `100% purity`。",
        "- `teacher_anchor_ari / pairwise_f1`：如果直接拿这个 teacher 和这个 anchor 去聚 target split，本地能得到多好的聚类结果。它不是 seed 指标，但能帮助判断 anchor 本身是否合理。",
        "",
        "## Anchor 总表",
        "",
        dataframe_to_markdown_table(anchor_summary_df),
        "",
        "## 推荐 Anchor",
        "",
        dataframe_to_markdown_table(pd.DataFrame([recommended_row])),
        "",
        "## 人工复查建议",
        "",
        "- 先看 `qualitative/anchor_<thr>/seed_clusters.jpg`：它展示被接纳 seed 的代表簇。",
        "- 再看 `qualitative/anchor_<thr>/suspect_seed_images.jpg`：这里优先放 purity 较低、相似度较低或簇更大的 seed 图。",
        "- 如果某个 anchor 的 coverage 很高，但 contact sheet 里肉眼看明显混入不同个体，就不要因为 coverage 漂亮就直接拿来 self-train。",
        "",
    ]
    if plot_paths:
        lines.extend(["## 图形读法", ""])
        if "coverage" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["coverage"], start=reports_dir))
            lines.extend(
                [
                    f"![Anchor seed coverage]({rel.as_posix()})",
                    "",
                    "- 先看 `seed coverage`，再看 `clean seed coverage`。如果两条线差很大，说明虽然 seed 多，但 purity 不够。",
                    "",
                ]
            )
        if "quality" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["quality"], start=reports_dir))
            lines.extend(
                [
                    f"![Anchor seed quality]({rel.as_posix()})",
                    "",
                    "- `weighted purity` 看干净程度，`mean similarity` 看簇内紧致度，`pure cluster ratio` 看完全无污染的小簇比例。",
                    "",
                ]
            )
        if "teacher" in plot_paths:
            rel = Path(os.path.relpath(plot_paths["teacher"], start=reports_dir))
            lines.extend(
                [
                    f"![Anchor teacher metrics]({rel.as_posix()})",
                    "",
                    "- 这张图帮助判断 anchor 本身是否荒谬：如果 teacher 在这个 anchor 下已经很差，这个 anchor 通常不适合继续做 seed。",
                    "",
                ]
            )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def run_anchor_seed_audit(
    *,
    repo_root: Path,
    output_dir: Path,
    dataset: str,
    teacher_descriptor_sources: list[str],
    teacher_checkpoint_sources: list[str],
    anchors: list[float],
    stability_delta: float,
    min_seed_cluster_size: int = 2,
    max_seed_cluster_size: int = 12,
    min_mean_similarity: float = 0.0,
    val_identity_fraction: float = 0.1,
    split_seed: int = 42,
    device: str = "cuda:0",
    num_workers: int = 4,
) -> dict[str, Path]:
    seed_everything(split_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    qualitative_dir = output_dir / "qualitative"
    embeddings_dir = output_dir / "embeddings"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, qualitative_dir, embeddings_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    train_df, _test_df = load_manifests(repo_root=repo_root)
    split_df = build_identity_holdout_split(
        train_df=train_df,
        val_identity_fraction=val_identity_fraction,
        seed=split_seed,
        datasets=[dataset],
    )
    fit_df = split_df[split_df["split_role_v1"] == "fit"].copy().reset_index(drop=True)
    target_df = split_df[split_df["split_role_v1"] == "val"].copy().reset_index(drop=True)
    fit_df.to_csv(tables_dir / "fit_manifest_v1.csv", index=False)
    target_df.to_csv(tables_dir / "target_manifest_v1.csv", index=False)

    teacher_specs = parse_teacher_sources(teacher_descriptor_sources, teacher_checkpoint_sources)
    teacher_fit_embeddings, teacher_target_embeddings, teacher_component_df = build_teacher_embeddings(
        repo_root=repo_root,
        fit_df=fit_df,
        target_df=target_df,
        teacher_specs=teacher_specs,
        device=device,
        num_workers=num_workers,
    )
    np.save(embeddings_dir / "teacher_fit_embeddings_v1.npy", teacher_fit_embeddings.astype(np.float32))
    np.save(embeddings_dir / "teacher_target_embeddings_v1.npy", teacher_target_embeddings.astype(np.float32))
    teacher_component_df.to_csv(tables_dir / "teacher_components_v1.csv", index=False)

    summary_rows: list[dict[str, float | int]] = []
    for anchor_threshold in sorted({round(float(value), 4) for value in anchors}):
        bundle = build_stable_pseudo_seed_bundle(
            target_df=target_df,
            teacher_embeddings=teacher_target_embeddings,
            anchor_threshold=anchor_threshold,
            stability_delta=stability_delta,
            min_seed_cluster_size=min_seed_cluster_size,
            max_seed_cluster_size=max_seed_cluster_size,
            min_mean_similarity=min_mean_similarity,
        )
        tag = _threshold_tag(anchor_threshold)
        anchor_dir = qualitative_dir / f"anchor_{tag}"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        seed_image_df = _build_seed_image_table(
            target_df=target_df,
            pseudo_seed_df=bundle.pseudo_seed_df,
            cluster_summary_df=bundle.cluster_summary_df,
        )
        seed_image_df.to_csv(tables_dir / f"seed_images_anchor_{tag}.csv", index=False)
        bundle.cluster_summary_df.to_csv(tables_dir / f"seed_clusters_anchor_{tag}.csv", index=False)
        bundle.pseudo_seed_df.to_csv(tables_dir / f"seed_assignments_anchor_{tag}.csv", index=False)
        bundle.threshold_summary_df.to_csv(tables_dir / f"threshold_summary_anchor_{tag}.csv", index=False)

        preview_df = _sample_seed_cluster_rows(seed_image_df=seed_image_df, max_clusters=8, images_per_cluster=4)
        if not preview_df.empty:
            create_contact_sheet(
                df=preview_df.rename(columns={PATH_COLUMN: "path"}),
                repo_root=repo_root,
                output_path=anchor_dir / "seed_clusters.jpg",
                title=f"Seed Clusters | {dataset} | anchor={anchor_threshold}",
                caption_columns=["pseudo_identity", "identity", "purity_vs_truth", "mean_similarity", "image_id"],
                columns=4,
            )
        suspect_df = _sample_suspect_rows(seed_image_df=seed_image_df, max_images=16)
        if not suspect_df.empty:
            create_contact_sheet(
                df=suspect_df.rename(columns={PATH_COLUMN: "path"}),
                repo_root=repo_root,
                output_path=anchor_dir / "suspect_seed_images.jpg",
                title=f"Suspect Seeds | {dataset} | anchor={anchor_threshold}",
                caption_columns=["pseudo_identity", "identity", "purity_vs_truth", "mean_similarity", "image_id"],
                columns=4,
            )
        summary_rows.append(
            _build_anchor_summary_row(
                anchor_threshold=anchor_threshold,
                cluster_summary_df=bundle.cluster_summary_df,
                teacher_anchor_metrics=bundle.teacher_anchor_metrics,
                seed_image_count=int(len(seed_image_df)),
                target_size=int(len(target_df)),
            )
        )

    anchor_summary_df = pd.DataFrame(summary_rows).sort_values("anchor_threshold").reset_index(drop=True)
    anchor_summary_df["recommended_rank"] = (
        anchor_summary_df.sort_values(
            ["clean_seed_coverage", "weighted_seed_purity", "teacher_anchor_ari", "seed_images"],
            ascending=[False, False, False, False],
        )
        .reset_index()
        .sort_values("index")
        .index
        + 1
    )
    anchor_summary_df.to_csv(tables_dir / "anchor_summary_v1.csv", index=False)
    recommended_row = anchor_summary_df.sort_values(
        ["clean_seed_coverage", "weighted_seed_purity", "teacher_anchor_ari", "seed_images"],
        ascending=[False, False, False, False],
    ).iloc[0]
    recommended_anchor = float(recommended_row["anchor_threshold"])
    plot_paths = _write_anchor_plots(reports_dir / "plots", anchor_summary_df)

    config = {
        "dataset": dataset,
        "anchors": [round(float(value), 4) for value in anchors],
        "stability_delta": stability_delta,
        "teacher_descriptor_sources": teacher_descriptor_sources,
        "teacher_checkpoint_sources": teacher_checkpoint_sources,
        "val_identity_fraction": val_identity_fraction,
        "split_seed": split_seed,
        "target_images": int(len(target_df)),
        "target_ids": int(target_df["identity"].nunique()),
        "resource_snapshot": collect_resource_snapshot(device),
        "recommended_anchor": recommended_anchor,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / "summary.json").write_text(json.dumps({"recommended_anchor": recommended_anchor}, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_path = _write_summary(
        output_dir=output_dir,
        config=config,
        anchor_summary_df=anchor_summary_df,
        recommended_anchor=recommended_anchor,
        plot_paths=plot_paths,
    )
    return {
        "summary_path": summary_path,
        "anchor_summary_path": tables_dir / "anchor_summary_v1.csv",
        "qualitative_dir": qualitative_dir,
    }
