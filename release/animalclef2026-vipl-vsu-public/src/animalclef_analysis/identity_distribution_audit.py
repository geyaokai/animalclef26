from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .descriptor_baselines import LABELED_DATASETS, build_identity_holdout_split, load_manifests
from .initial_audit import load_metadata


def build_identity_count_frame(df: pd.DataFrame, extra_group_columns: list[str] | None = None) -> pd.DataFrame:
    group_columns = ["dataset"]
    if extra_group_columns:
        group_columns.extend(extra_group_columns)
    group_columns.append("identity")
    counts = (
        df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="images_per_identity")
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    return counts


def build_distribution_table(
    counts_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    distribution = (
        counts_df.groupby(group_columns + ["images_per_identity"], dropna=False)
        .size()
        .reset_index(name="identity_count")
        .sort_values(group_columns + ["images_per_identity"])
        .reset_index(drop=True)
    )
    return distribution


def build_summary_table(
    counts_df: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    summary = (
        counts_df.groupby(group_columns, dropna=False)["images_per_identity"]
        .agg(
            identities="count",
            total_images="sum",
            min_images="min",
            median_images="median",
            mean_images="mean",
            max_images="max",
            singletons=lambda s: int((s == 1).sum()),
        )
        .reset_index()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    summary["singleton_ratio"] = (summary["singletons"] / summary["identities"]).round(4)
    summary["mean_images"] = summary["mean_images"].round(3)
    return summary


def prepend_overall_rows(
    counts_df: pd.DataFrame,
    extra_group_columns: list[str] | None = None,
) -> pd.DataFrame:
    columns = []
    if extra_group_columns:
        columns.extend(extra_group_columns)
    overall = counts_df.copy()
    overall["dataset"] = "ALL_LABELED_TRAIN"
    return pd.concat([overall, counts_df], ignore_index=True)


def _distribution_arrays(distribution_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = distribution_df["images_per_identity"].to_numpy(dtype=int)
    y = distribution_df["identity_count"].to_numpy(dtype=int)
    return x, y


def _load_matplotlib_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
        raise ModuleNotFoundError(
            "identity distribution plots require matplotlib. Run this script in an environment such as 'wildfusion'."
        ) from exc
    return plt


def _format_log_axis_as_integers(ax) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    ymax = ax.get_ylim()[1]
    candidate_ticks = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]
    ticks = [tick for tick in candidate_ticks if tick <= ymax * 1.02]
    if ymax > ticks[-1]:
        ticks.append(candidate_ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{int(value)}" if value >= 1 else ""))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_dual_scale_distribution(
    distribution_df: pd.DataFrame,
    title: str,
    subtitle: str,
    output_path: Path,
    color: str,
) -> None:
    plt = _load_matplotlib_pyplot()
    x, y = _distribution_arrays(distribution_df)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    for index, ax in enumerate(axes):
        ax.bar(x, y, color=color, width=0.9)
        ax.set_xlabel("images per identity")
        ax.set_ylabel("number of identities")
        ax.set_title("linear scale" if index == 0 else "log y-scale")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        if index == 1:
            ax.set_yscale("log")
            _format_log_axis_as_integers(ax)
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_dataset_distribution(
    distribution_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    plt = _load_matplotlib_pyplot()
    datasets = [dataset for dataset in summary_df["dataset"].tolist() if dataset != "ALL_LABELED_TRAIN"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    palette = ["#355070", "#6d597a", "#b56576", "#e56b6f"]
    for index, dataset in enumerate(datasets):
        ax = axes[index]
        current = distribution_df[distribution_df["dataset"] == dataset]
        summary = summary_df[summary_df["dataset"] == dataset].iloc[0]
        x, y = _distribution_arrays(current)
        ax.bar(x, y, color=palette[index % len(palette)], width=0.9)
        ax.set_yscale("log")
        ax.set_xlabel("images per identity")
        if index == 0:
            ax.set_ylabel("number of identities (log scale)")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        _format_log_axis_as_integers(ax)
        ax.set_title(
            f"{dataset}\n"
            f"ids={int(summary['identities'])}, "
            f"singletons={int(summary['singletons'])} ({summary['singleton_ratio']:.2%})"
        )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_overlay(
    distribution_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    plt = _load_matplotlib_pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
    role_order = ["fit", "val"]
    colors = {"fit": "#2a9d8f", "val": "#e76f51"}
    subtitle = []
    for role in role_order:
        row = summary_df[(summary_df["dataset"] == "ALL_LABELED_TRAIN") & (summary_df["split_role_v1"] == role)].iloc[0]
        subtitle.append(
            f"{role}: ids={int(row['identities'])}, images={int(row['total_images'])}, "
            f"singletons={int(row['singletons'])} ({row['singleton_ratio']:.2%})"
        )
    for index, ax in enumerate(axes):
        for role in role_order:
            current = distribution_df[
                (distribution_df["dataset"] == "ALL_LABELED_TRAIN")
                & (distribution_df["split_role_v1"] == role)
            ]
            x, y = _distribution_arrays(current)
            ax.step(x, y, where="mid", linewidth=1.8, color=colors[role], label=role)
            ax.scatter(x, y, s=12, color=colors[role])
        ax.set_xlabel("images per identity")
        ax.set_ylabel("number of identities")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend()
        ax.set_title("linear scale" if index == 0 else "log y-scale")
        if index == 1:
            ax.set_yscale("log")
            _format_log_axis_as_integers(ax)
    fig.suptitle(f"{title}\n" + " | ".join(subtitle), fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_split_by_dataset(
    distribution_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    plt = _load_matplotlib_pyplot()
    datasets = [dataset for dataset in summary_df["dataset"].unique().tolist() if dataset != "ALL_LABELED_TRAIN"]
    role_order = ["fit", "val"]
    colors = {"fit": "#2a9d8f", "val": "#e76f51"}
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for index, dataset in enumerate(datasets):
        ax = axes[index]
        for role in role_order:
            current = distribution_df[
                (distribution_df["dataset"] == dataset)
                & (distribution_df["split_role_v1"] == role)
            ]
            x, y = _distribution_arrays(current)
            ax.step(x, y, where="mid", linewidth=1.8, color=colors[role], label=role)
            ax.scatter(x, y, s=10, color=colors[role])
        fit_row = summary_df[(summary_df["dataset"] == dataset) & (summary_df["split_role_v1"] == "fit")].iloc[0]
        val_row = summary_df[(summary_df["dataset"] == dataset) & (summary_df["split_role_v1"] == "val")].iloc[0]
        ax.set_yscale("log")
        ax.set_xlabel("images per identity")
        if index == 0:
            ax.set_ylabel("number of identities (log scale)")
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        _format_log_axis_as_integers(ax)
        ax.set_title(
            f"{dataset}\n"
            f"fit ids={int(fit_row['identities'])}, val ids={int(val_row['identities'])}\n"
            f"fit sing={int(fit_row['singletons'])}, val sing={int(val_row['singletons'])}"
        )
        ax.legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    output_path: Path,
    overall_summary_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    config: dict[str, object],
    plot_paths: dict[str, Path],
) -> None:
    def frame_to_markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = [
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
            for _, row in frame.iterrows()
        ]
        return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])

    overall_row = overall_summary_df[overall_summary_df["dataset"] == "ALL_LABELED_TRAIN"].iloc[0]
    fit_overall = split_summary_df[
        (split_summary_df["dataset"] == "ALL_LABELED_TRAIN") & (split_summary_df["split_role_v1"] == "fit")
    ].iloc[0]
    val_overall = split_summary_df[
        (split_summary_df["dataset"] == "ALL_LABELED_TRAIN") & (split_summary_df["split_role_v1"] == "val")
    ].iloc[0]
    overall_plot_rel = os.path.relpath(plot_paths["overall"], start=output_path.parent)
    per_dataset_plot_rel = os.path.relpath(plot_paths["per_dataset"], start=output_path.parent)
    split_overall_plot_rel = os.path.relpath(plot_paths["split_overall"], start=output_path.parent)
    split_by_dataset_plot_rel = os.path.relpath(plot_paths["split_by_dataset"], start=output_path.parent)
    lines = [
        "# Identity Distribution Audit",
        "",
        "## 图怎么读",
        "",
        "- `x` 轴是 `images per identity`，表示一个 identity 在标注训练集中有多少张图。",
        "- `y` 轴是 `number of identities`，表示有多少个 identity 落在这个图片数上。",
        "- `linear scale` 适合先看主体分布，尤其先看 `x=1` 是否有明显尖峰，这对应 singleton。",
        "- `log y-scale` 会压缩前面的高峰，让右侧长尾更容易看见，适合判断是否存在少数高样本 identity 主导分布。",
        "",
        "## Overall Labeled Train Distribution",
        "",
        f"- Labeled training identities: `{int(overall_row['identities'])}`",
        f"- Labeled training images: `{int(overall_row['total_images'])}`",
        f"- Singleton identities: `{int(overall_row['singletons'])}` (`{overall_row['singleton_ratio']:.2%}`)",
        f"- Median images per identity: `{overall_row['median_images']}`",
        f"- Max images per identity: `{int(overall_row['max_images'])}`",
        "",
        f"![Overall labeled-train identity distribution]({overall_plot_rel})",
        "",
        "图 1 说明：左图是线性纵轴，所以前几个低图片数区间最显眼；右图是对数纵轴，适合看长尾。读这张图时先看 `x=1`，这里就是 singleton identity 的数量。",
        "",
        "## Baseline Holdout Split",
        "",
        f"- Split protocol: identity-level holdout within each labeled dataset (`{', '.join(config['datasets'])}`)",
        f"- Validation identity fraction: `{config['val_identity_fraction']}`",
        f"- Split seed: `{config['split_seed']}`",
        f"- Fit identities/images: `{int(fit_overall['identities'])}` / `{int(fit_overall['total_images'])}`",
        f"- Val identities/images: `{int(val_overall['identities'])}` / `{int(val_overall['total_images'])}`",
        f"- Val singleton identities: `{int(val_overall['singletons'])}` (`{val_overall['singleton_ratio']:.2%}`)",
        "",
        f"![Per-dataset labeled-train identity distribution]({per_dataset_plot_rel})",
        "",
        "图 2 说明：每个子图对应一个 dataset，全部使用对数纵轴。先比较各自 `x=1` 的高度，就能看出哪个数据集 singleton 更严重；再看右侧拖得多远，判断高样本 identity 的长尾有多长。",
        "",
        f"![Baseline split overall identity distribution]({split_overall_plot_rel})",
        "",
        "图 3 说明：`fit` 和 `val` 是按 identity 切分，所以同一个 identity 只会出现在一边。如果切分比较合理，两条曲线的形状应该大体相似，只是 `val` 的整体规模更小。",
        "",
        f"![Baseline split identity distribution by dataset]({split_by_dataset_plot_rel})",
        "",
        "图 4 说明：这是图 3 的分 dataset 版本。主要用来看某个 dataset 的验证集是否被切得更难，比如 singleton 明显偏多，或者高样本 identity 大多留在了 `fit` 里。",
        "",
        "## Overall Summary Table",
        "",
        frame_to_markdown(overall_summary_df),
        "",
        "## Split Summary Table",
        "",
        frame_to_markdown(split_summary_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_identity_distribution_audit(
    repo_root: Path,
    output_dir: Path,
    val_identity_fraction: float = 0.1,
    split_seed: int = 42,
) -> dict[str, Path]:
    metadata_df = load_metadata(repo_root / "metadata.csv")
    labeled_train_df = metadata_df[
        (metadata_df["split"] == "train")
        & (metadata_df["identity"].fillna("") != "")
        & (metadata_df["dataset"].isin(LABELED_DATASETS))
    ].copy()

    overall_counts_df = build_identity_count_frame(labeled_train_df)
    overall_counts_with_all_df = prepend_overall_rows(overall_counts_df)
    overall_distribution_df = build_distribution_table(overall_counts_with_all_df, group_columns=["dataset"])
    overall_summary_df = build_summary_table(overall_counts_with_all_df, group_columns=["dataset"])

    train_manifest_df, _ = load_manifests(repo_root=repo_root)
    split_df = build_identity_holdout_split(
        train_df=train_manifest_df,
        val_identity_fraction=val_identity_fraction,
        seed=split_seed,
        datasets=LABELED_DATASETS,
    )
    split_counts_df = build_identity_count_frame(split_df, extra_group_columns=["split_role_v1"])
    split_counts_with_all_df = split_counts_df.copy()
    split_counts_with_all_df["dataset"] = "ALL_LABELED_TRAIN"
    split_counts_with_all_df = pd.concat([split_counts_with_all_df, split_counts_df], ignore_index=True)
    split_distribution_df = build_distribution_table(
        split_counts_with_all_df,
        group_columns=["split_role_v1", "dataset"],
    )
    split_summary_df = build_summary_table(
        split_counts_with_all_df,
        group_columns=["split_role_v1", "dataset"],
    )

    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, plots_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    overall_counts_df.to_csv(tables_dir / "overall_identity_counts_labeled_train.csv", index=False)
    overall_distribution_df.to_csv(tables_dir / "overall_identity_distribution_labeled_train.csv", index=False)
    overall_summary_df.to_csv(tables_dir / "overall_identity_summary_labeled_train.csv", index=False)
    split_df.to_csv(tables_dir / "baseline_split_assignments_v1.csv", index=False)
    split_counts_df.to_csv(tables_dir / "baseline_split_identity_counts_v1.csv", index=False)
    split_distribution_df.to_csv(tables_dir / "baseline_split_identity_distribution_v1.csv", index=False)
    split_summary_df.to_csv(tables_dir / "baseline_split_identity_summary_v1.csv", index=False)

    overall_row = overall_summary_df[overall_summary_df["dataset"] == "ALL_LABELED_TRAIN"].iloc[0]
    plot_dual_scale_distribution(
        distribution_df=overall_distribution_df[overall_distribution_df["dataset"] == "ALL_LABELED_TRAIN"],
        title="Overall Identity Count Distribution | labeled train",
        subtitle=(
            f"ids={int(overall_row['identities'])}, images={int(overall_row['total_images'])}, "
            f"singletons={int(overall_row['singletons'])} ({overall_row['singleton_ratio']:.2%})"
        ),
        output_path=plots_dir / "overall_identity_count_distribution_labeled_train.png",
        color="#355070",
    )
    plot_per_dataset_distribution(
        distribution_df=overall_distribution_df,
        summary_df=overall_summary_df,
        title="Identity Count Distribution By Dataset | labeled train",
        output_path=plots_dir / "per_dataset_identity_count_distribution_labeled_train.png",
    )
    plot_split_overlay(
        distribution_df=split_distribution_df,
        summary_df=split_summary_df,
        title="Baseline Holdout Identity Count Distribution | overall",
        output_path=plots_dir / "baseline_split_identity_count_distribution_overall.png",
    )
    plot_split_by_dataset(
        distribution_df=split_distribution_df,
        summary_df=split_summary_df,
        title="Baseline Holdout Identity Count Distribution By Dataset",
        output_path=plots_dir / "baseline_split_identity_count_distribution_by_dataset.png",
    )
    plot_paths = {
        "overall": plots_dir / "overall_identity_count_distribution_labeled_train.png",
        "per_dataset": plots_dir / "per_dataset_identity_count_distribution_labeled_train.png",
        "split_overall": plots_dir / "baseline_split_identity_count_distribution_overall.png",
        "split_by_dataset": plots_dir / "baseline_split_identity_count_distribution_by_dataset.png",
    }

    report_config = {
        "val_identity_fraction": val_identity_fraction,
        "split_seed": split_seed,
        "datasets": LABELED_DATASETS,
    }
    write_markdown_report(
        output_path=reports_dir / "summary.md",
        overall_summary_df=overall_summary_df,
        split_summary_df=split_summary_df,
        config=report_config,
        plot_paths=plot_paths,
    )
    (reports_dir / "summary.json").write_text(
        json.dumps(report_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "report_path": reports_dir / "summary.md",
        "plots_dir": plots_dir,
        "tables_dir": tables_dir,
    }
