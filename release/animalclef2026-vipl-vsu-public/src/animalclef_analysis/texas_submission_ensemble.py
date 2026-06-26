from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import (
    PATH_COLUMN,
    build_submission,
    dataframe_to_markdown_table,
    ensure_metadata_alignment,
    fuse_embedding_blocks,
    load_cached_embedding_bundle,
)
from .texas_selftrain import evaluate_texas_thresholds, pick_best_texas_threshold
from .texas_unsupervised import TEXAS_DATASET, build_topk_indices


DEFAULT_BASE_PREDICTIONS = Path("artifacts/submissions/kaggle_mixed_baseline_v2/tables/test_predictions_v1.csv")
DEFAULT_SAMPLE_SUBMISSION = Path("sample_submission.csv")
DEFAULT_PSEUDO_ASSIGNMENTS = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1/tables/pseudo_assignments_v1.csv")
DEFAULT_CANDIDATE_PAIRS = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1/tables/candidate_pairs_v1.csv")
DEFAULT_TEACHER_ANCHOR_PREDICTIONS = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1/tables/test_predictions_best_v1.csv")
DEFAULT_TEACHER_TOPK_SOURCE_DIR = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1")
DEFAULT_TOP_K = 8


def _resolve_default_texas_proxy_root(
    *,
    repo_root: Path,
    source_dirs: list[Path],
    checkpoint_path: Path | None,
) -> Path | None:
    if checkpoint_path is not None:
        candidate = checkpoint_path.resolve().parents[1]
        if (candidate / "tables" / "pseudo_assignments_v1.csv").exists():
            return candidate
    if len(source_dirs) == 1:
        candidate = source_dirs[0].resolve()
        if (candidate / "tables" / "pseudo_assignments_v1.csv").exists():
            return candidate
    fallback = repo_root / DEFAULT_TEACHER_TOPK_SOURCE_DIR
    if (fallback / "tables" / "pseudo_assignments_v1.csv").exists():
        return fallback
    return None


def _resolve_texas_proxy_artifacts(
    *,
    repo_root: Path,
    source_dirs: list[Path],
    checkpoint_path: Path | None,
    pseudo_assignments_path: Path | None,
    candidate_pairs_path: Path | None,
    teacher_anchor_predictions_path: Path | None,
    teacher_topk_source_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    proxy_root = _resolve_default_texas_proxy_root(
        repo_root=repo_root,
        source_dirs=source_dirs,
        checkpoint_path=checkpoint_path,
    )
    default_pseudo_assignments = (
        proxy_root / "tables" / "pseudo_assignments_v1.csv"
        if proxy_root is not None
        else repo_root / DEFAULT_PSEUDO_ASSIGNMENTS
    )
    default_candidate_pairs = (
        proxy_root / "tables" / "candidate_pairs_v1.csv"
        if proxy_root is not None
        else repo_root / DEFAULT_CANDIDATE_PAIRS
    )
    default_teacher_anchor_predictions = (
        proxy_root / "tables" / "test_predictions_best_v1.csv"
        if proxy_root is not None
        else repo_root / DEFAULT_TEACHER_ANCHOR_PREDICTIONS
    )
    default_teacher_topk_source_dir = (
        proxy_root
        if proxy_root is not None and (proxy_root / "embeddings" / "test_embeddings.npy").exists()
        else repo_root / DEFAULT_TEACHER_TOPK_SOURCE_DIR
    )
    return (
        pseudo_assignments_path.resolve() if pseudo_assignments_path is not None else default_pseudo_assignments.resolve(),
        candidate_pairs_path.resolve() if candidate_pairs_path is not None else default_candidate_pairs.resolve(),
        teacher_anchor_predictions_path.resolve()
        if teacher_anchor_predictions_path is not None
        else default_teacher_anchor_predictions.resolve(),
        teacher_topk_source_dir.resolve() if teacher_topk_source_dir is not None else default_teacher_topk_source_dir.resolve(),
    )


def _reorder_texas_embeddings_to_reference(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(candidate_df) != len(embeddings):
        raise ValueError(
            f"Texas metadata row mismatch for {candidate_name}: df={len(candidate_df)} vs embeddings={len(embeddings)}"
        )
    join_columns = ["image_id", "dataset"]
    reference_view = reference_df[join_columns].copy().reset_index(drop=True)
    candidate_view = candidate_df[join_columns].copy().reset_index(drop=True)
    reference_view["image_id"] = reference_view["image_id"].astype(str)
    candidate_view["image_id"] = candidate_view["image_id"].astype(str)
    merged = reference_view.merge(
        candidate_view.assign(_embedding_row=np.arange(len(candidate_view), dtype=np.int32)),
        on=join_columns,
        how="left",
        validate="one_to_one",
    )
    if merged["_embedding_row"].isna().any():
        missing_rows = merged.loc[merged["_embedding_row"].isna(), join_columns].head(5).to_dict(orient="records")
        raise ValueError(f"Missing Texas image_id/dataset pairs in {candidate_name}, examples: {missing_rows}")
    reorder_index = merged["_embedding_row"].astype(int).to_numpy()
    reordered_df = candidate_df.iloc[reorder_index].reset_index(drop=True).copy()
    reordered_embeddings = embeddings[reorder_index]
    metadata_columns = [
        column
        for column in ["image_id", "dataset", "identity"]
        if column in reference_df.columns and column in reordered_df.columns
    ]
    ensure_metadata_alignment(
        reference_df=reference_df.loc[:, metadata_columns].reset_index(drop=True),
        candidate_df=reordered_df.loc[:, metadata_columns].reset_index(drop=True),
        split_name="texas_submission_variant",
        reference_name="reference_texas_metadata",
        candidate_name=candidate_name,
    )
    return reordered_df, reordered_embeddings


def _load_texas_source_block(
    *,
    source_dir: Path,
    reference_df: pd.DataFrame | None = None,
    component_name: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    bundle = load_cached_embedding_bundle(source_dir=source_dir.resolve(), name=component_name or source_dir.name)
    texas_df = bundle.test_df[bundle.test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    texas_df["image_id"] = texas_df["image_id"].astype(str)
    texas_embeddings = bundle.test_embeddings[(bundle.test_df["dataset"] == TEXAS_DATASET).to_numpy()]
    if reference_df is not None:
        texas_df, texas_embeddings = _reorder_texas_embeddings_to_reference(
            reference_df=reference_df,
            candidate_df=texas_df,
            embeddings=texas_embeddings,
            candidate_name=component_name or source_dir.name,
        )
    component_row = {
        "component": component_name or source_dir.name,
        "source_dir": str(source_dir.resolve()),
        "embedding_dim": int(texas_embeddings.shape[1]),
    }
    return texas_df, texas_embeddings, component_row


def load_texas_ensemble_blocks(
    *,
    source_dirs: list[Path],
    component_names: list[str] | None = None,
    weights: list[float] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if not source_dirs:
        raise ValueError("Need at least one Texas ensemble source dir")
    if component_names is not None and len(component_names) != len(source_dirs):
        raise ValueError("component_names length must match source_dirs")
    if weights is None:
        weights = [1.0] * len(source_dirs)
    if len(weights) != len(source_dirs):
        raise ValueError("weights length must match source_dirs")

    reference_df: pd.DataFrame | None = None
    blocks: list[np.ndarray] = []
    component_rows: list[dict[str, Any]] = []
    for index, (source_dir, weight) in enumerate(zip(source_dirs, weights, strict=True)):
        component_name = component_names[index] if component_names is not None else source_dir.name
        texas_df, texas_embeddings, component_row = _load_texas_source_block(
            source_dir=source_dir,
            reference_df=reference_df,
            component_name=component_name,
        )
        if reference_df is None:
            reference_df = texas_df.copy().reset_index(drop=True)
        blocks.append(texas_embeddings)
        component_row["weight"] = float(weight)
        component_rows.append(component_row)
    fused_embeddings = fuse_embedding_blocks(blocks, weights=weights).astype(np.float32, copy=False)
    return reference_df.copy().reset_index(drop=True), fused_embeddings, pd.DataFrame(component_rows)


def load_texas_checkpoint_block(
    *,
    repo_root: Path,
    checkpoint_path: Path,
    test_manifest_path: Path | None = None,
    component_name: str | None = None,
    device: str = "cuda:0",
    eval_batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    import torch

    from .orb_rerank_baseline import resolve_existing_image_rel_path
    from .supervised_training import build_eval_transform, load_student_backbone
    from .texas_selftrain import TexasSelfTrainModel, extract_texas_student_embeddings
    from .view_manifests import get_default_manifest_paths

    repo_root = repo_root.resolve()
    checkpoint_path = checkpoint_path.resolve()
    if test_manifest_path is None:
        _train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
    test_manifest_path = test_manifest_path.resolve()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config", {})
    student_backbone = str(checkpoint_config.get("student_backbone", "")).strip()
    if not student_backbone:
        raise ValueError(f"Texas checkpoint is missing config.student_backbone: {checkpoint_path}")
    embedding_dim = int(checkpoint_config.get("embedding_dim", 512))
    classification_head = str(checkpoint_config.get("classification_head", "linear"))

    backbone, backbone_spec = load_student_backbone(student_backbone, device=device)
    model = TexasSelfTrainModel(
        backbone=backbone,
        feature_dim=int(backbone_spec.feature_dim),
        embedding_dim=embedding_dim,
        teacher_dim=0,
        pseudo_class_count=0,
        classification_head=classification_head,
        arcface_scale=30.0,
        arcface_margin=0.3,
    ).to(device)
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Texas checkpoint is missing model_state_dict: {checkpoint_path}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    required_missing_prefixes = ("backbone.", "embedding_layer.", "embedding_bn.")
    required_missing_keys = [
        key for key in incompatible.missing_keys if key.startswith(required_missing_prefixes)
    ]
    if required_missing_keys:
        raise ValueError(
            f"Texas checkpoint is missing required encoder weights: {required_missing_keys[:8]}"
        )
    eval_transform = build_eval_transform(backbone_spec, dataset=TEXAS_DATASET)

    test_df = pd.read_csv(test_manifest_path)
    test_df["image_id"] = test_df["image_id"].astype(str)
    test_df["dataset"] = test_df["dataset"].astype(str)
    test_df[PATH_COLUMN] = [resolve_existing_image_rel_path(row, repo_root=repo_root) for _, row in test_df.iterrows()]
    texas_df = test_df[test_df["dataset"] == TEXAS_DATASET].copy().reset_index(drop=True)
    if texas_df.empty:
        raise ValueError(f"No {TEXAS_DATASET} rows found in {test_manifest_path}")

    texas_embeddings = extract_texas_student_embeddings(
        df=texas_df,
        repo_root=repo_root,
        model=model,
        transform=eval_transform,
        device=device,
        batch_size=int(eval_batch_size),
        num_workers=int(num_workers),
    ).astype(np.float32, copy=False)
    component_df = pd.DataFrame(
        [
            {
                "component": component_name or checkpoint_path.stem,
                "source_dir": str(checkpoint_path),
                "embedding_dim": int(texas_embeddings.shape[1]),
                "weight": 1.0,
            }
        ]
    )
    return texas_df, texas_embeddings, component_df


def _reorder_frame_to_reference(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    required_columns: list[str],
    candidate_name: str,
) -> pd.DataFrame:
    reference_ids = reference_df["image_id"].astype(str).tolist()
    frame = candidate_df.copy()
    frame["image_id"] = frame["image_id"].astype(str)
    lookup = frame.set_index("image_id", drop=False)
    missing = [image_id for image_id in reference_ids if image_id not in lookup.index]
    if missing:
        raise ValueError(f"Missing Texas image_ids in {candidate_name}, examples: {missing[:5]}")
    reordered = lookup.loc[reference_ids].reset_index(drop=True)
    available_columns = [column for column in required_columns if column in reference_df.columns and column in reordered.columns]
    if available_columns:
        ensure_metadata_alignment(
            reference_df=reference_df.loc[:, available_columns].reset_index(drop=True),
            candidate_df=reordered.loc[:, available_columns].reset_index(drop=True),
            split_name="texas_submission_variant",
            reference_name="reference_texas_metadata",
            candidate_name=candidate_name,
        )
    return reordered


def _load_aligned_pseudo_assignments(reference_df: pd.DataFrame, pseudo_assignments_path: Path) -> pd.DataFrame:
    pseudo_df = pd.read_csv(pseudo_assignments_path)
    pseudo_df = _reorder_frame_to_reference(
        reference_df=reference_df,
        candidate_df=pseudo_df,
        required_columns=["image_id", "dataset", PATH_COLUMN],
        candidate_name="pseudo_assignments",
    )
    for column in ["is_seed", "pseudo_label_index"]:
        if column not in pseudo_df.columns:
            raise ValueError(f"pseudo_assignments missing required column: {column}")
    pseudo_df["is_seed"] = pseudo_df["is_seed"].fillna(False).astype(bool)
    pseudo_df["pseudo_label_index"] = pseudo_df["pseudo_label_index"].fillna(-1).astype(int)
    return pseudo_df


def _load_aligned_anchor_labels(reference_df: pd.DataFrame, teacher_anchor_predictions_path: Path) -> np.ndarray:
    anchor_df = pd.read_csv(teacher_anchor_predictions_path)
    anchor_df = _reorder_frame_to_reference(
        reference_df=reference_df,
        candidate_df=anchor_df,
        required_columns=["image_id", "dataset", PATH_COLUMN],
        candidate_name="teacher_anchor_predictions",
    )
    if "pred_cluster_id" not in anchor_df.columns:
        raise ValueError("teacher_anchor_predictions missing pred_cluster_id")
    return anchor_df["pred_cluster_id"].to_numpy(dtype=int)


def _build_cluster_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, dataset_df in frame.groupby("dataset"):
        counts = dataset_df["pred_cluster_id"].value_counts()
        rows.append(
            {
                "dataset": str(dataset),
                "samples": int(len(dataset_df)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(dataset_df["route_name"].iloc[0]),
                "embedding_dim": int(dataset_df["embedding_dim"].iloc[0]),
                "threshold": float(dataset_df["chosen_threshold"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def _build_route_summary(frame: pd.DataFrame) -> pd.DataFrame:
    route_rows: list[dict[str, Any]] = []
    for dataset, dataset_df in frame.groupby("dataset"):
        route_rows.append(
            {
                "dataset": str(dataset),
                "route_name": str(dataset_df["route_name"].iloc[0]),
                "embedding_dim": int(dataset_df["embedding_dim"].iloc[0]),
                "threshold": float(dataset_df["chosen_threshold"].iloc[0]),
            }
        )
    return pd.DataFrame(route_rows).sort_values("dataset").reset_index(drop=True)


def _describe_route(
    *,
    dataset: str,
    route_name: str,
    embedding_dim: int,
    threshold: float,
    component_df: pd.DataFrame | None = None,
) -> str:
    if dataset == TEXAS_DATASET and component_df is not None and not component_df.empty:
        component_bits = [
            f"{row.component} `{int(row.embedding_dim)}-d` x weight `{float(row.weight):.2f}`"
            for row in component_df.itertuples(index=False)
        ]
        return (
            f"  - `{dataset}`: weighted `concat + L2 normalize` ensemble of "
            + ", ".join(component_bits)
            + f", producing `{embedding_dim}-d`; threshold `{threshold}`."
        )
    if route_name == "ft_mega_arcface_distill_v1":
        return (
            f"  - `{dataset}`: supervised `MegaDescriptor-L-384` student, "
            f"`B x 3 x 384 x 384 -> B x 1536 -> B x 512`, threshold `{threshold}`."
        )
    if route_name == "fusion_orb_rerank_v1":
        return (
            f"  - `{dataset}`: frozen early fusion, "
            f"`Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`, "
            f"then ORB rerank on top-K pairs, threshold `{threshold}`."
        )
    if route_name == "fusion_v1":
        return (
            f"  - `{dataset}`: frozen early fusion, "
            f"`Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`, threshold `{threshold}`."
        )
    return f"  - `{dataset}`: route `{route_name}`, embedding dim `{embedding_dim}`, threshold `{threshold}`."


def _write_markdown_report(
    output_path: Path,
    *,
    config: dict[str, Any],
    component_df: pd.DataFrame,
    best_threshold_df: pd.DataFrame,
    threshold_summary_df: pd.DataFrame,
    route_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
) -> None:
    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> optional local rerank -> average-linkage clustering -> submission cluster label`.",
        "- Current route:",
    ]
    for row in route_df.itertuples(index=False):
        component_view = component_df if str(row.dataset) == TEXAS_DATASET else None
        architecture_lines.append(
            _describe_route(
                dataset=str(row.dataset),
                route_name=str(row.route_name),
                embedding_dim=int(row.embedding_dim),
                threshold=float(row.threshold),
                component_df=component_view,
            )
        )

    lines = [
        "# Texas Ensemble Submission Variant",
        "",
        f"- Override dataset: `{TEXAS_DATASET}`",
        f"- Route name: `{config['route_name']}`",
        f"- Base predictions: `{config['base_predictions']}`",
        f"- Source dirs: `{config['source_dirs']}`",
        f"- Checkpoint path: `{config['checkpoint_path']}`",
        f"- Weights: `{config['weights']}`",
        f"- Candidate thresholds: `{config['thresholds']}`",
        "",
        "## Route Summary",
        "",
        dataframe_to_markdown_table(route_df),
        "",
        "## Texas Ensemble Components",
        "",
        dataframe_to_markdown_table(component_df),
        "",
        "## Selected Threshold",
        "",
        dataframe_to_markdown_table(best_threshold_df),
        "",
        "## Threshold Sweep",
        "",
        dataframe_to_markdown_table(threshold_summary_df),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Architecture",
        "",
        *architecture_lines,
        "",
        "## Notes",
        "",
        f"- Test manifest path: `{config['test_manifest_path']}`",
        f"- Device: `{config['device']}` | eval batch size `{config['eval_batch_size']}` | workers `{config['num_workers']}`",
        f"- Teacher anchor predictions: `{config['teacher_anchor_predictions_path']}`",
        f"- Teacher top-k source dir: `{config['teacher_topk_source_dir']}`",
        f"- Pseudo assignments: `{config['pseudo_assignments_path']}`",
        f"- Candidate pairs: `{config['candidate_pairs_path']}`",
        "",
        "## Key Performance Tricks",
        "",
        "- `single-dataset override`: keep `Lynx / Salamander / SeaTurtle` fixed and modify only `TexasHornedLizards`.",
        "- `low-cost ensemble`: reuse existing cached Texas embeddings instead of retraining a new student.",
        "- `proxy threshold selection`: pick the Texas threshold by pseudo-label agreement, mutual-topk pair retention, and cluster-shape stability before spending a Kaggle submission.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_texas_ensemble_submission_variant(
    *,
    repo_root: Path,
    output_dir: Path,
    route_name: str,
    source_dirs: list[Path] | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_component_name: str | None = None,
    component_names: list[str] | None = None,
    weights: list[float] | None = None,
    thresholds: list[float] | None = None,
    anchor_threshold: float = 0.44,
    base_predictions: Path | None = None,
    sample_submission_path: Path | None = None,
    test_manifest_path: Path | None = None,
    device: str = "cuda:0",
    eval_batch_size: int = 16,
    num_workers: int = 4,
    pseudo_assignments_path: Path | None = None,
    candidate_pairs_path: Path | None = None,
    teacher_anchor_predictions_path: Path | None = None,
    teacher_topk_source_dir: Path | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Path]:
    from .view_manifests import get_default_manifest_paths

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    source_dirs = [path.resolve() for path in (source_dirs or [])]
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.resolve()
    if source_dirs and checkpoint_path is not None:
        raise ValueError("Pass either source_dirs or checkpoint_path, not both")
    if not source_dirs and checkpoint_path is None:
        raise ValueError("Need at least one source_dir or one checkpoint_path")
    if base_predictions is None:
        base_predictions = repo_root / DEFAULT_BASE_PREDICTIONS
    if sample_submission_path is None:
        sample_submission_path = repo_root / DEFAULT_SAMPLE_SUBMISSION
    resolved_test_manifest_path: Path | None = None
    if checkpoint_path is not None:
        if test_manifest_path is None:
            _train_manifest_path, resolved_test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
        else:
            resolved_test_manifest_path = test_manifest_path.resolve()
    (
        pseudo_assignments_path,
        candidate_pairs_path,
        teacher_anchor_predictions_path,
        teacher_topk_source_dir,
    ) = _resolve_texas_proxy_artifacts(
        repo_root=repo_root,
        source_dirs=source_dirs,
        checkpoint_path=checkpoint_path,
        pseudo_assignments_path=pseudo_assignments_path,
        candidate_pairs_path=candidate_pairs_path,
        teacher_anchor_predictions_path=teacher_anchor_predictions_path,
        teacher_topk_source_dir=teacher_topk_source_dir,
    )
    if thresholds is None:
        thresholds = [0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48]

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    embeddings_dir = output_dir / "embeddings"
    for path in [tables_dir, reports_dir, embeddings_dir]:
        path.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is not None:
        texas_df, texas_embeddings, component_df = load_texas_checkpoint_block(
            repo_root=repo_root,
            checkpoint_path=checkpoint_path,
            test_manifest_path=resolved_test_manifest_path,
            component_name=checkpoint_component_name,
            device=device,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    else:
        texas_df, texas_embeddings, component_df = load_texas_ensemble_blocks(
            source_dirs=source_dirs,
            component_names=component_names,
            weights=weights,
        )
    texas_df["image_id"] = texas_df["image_id"].astype(str)
    pseudo_df = _load_aligned_pseudo_assignments(texas_df, pseudo_assignments_path.resolve())
    candidate_pair_df = pd.read_csv(candidate_pairs_path.resolve())
    teacher_anchor_labels = _load_aligned_anchor_labels(texas_df, teacher_anchor_predictions_path.resolve())
    _teacher_topk_df, teacher_topk_embeddings, _component = _load_texas_source_block(
        source_dir=teacher_topk_source_dir.resolve(),
        reference_df=texas_df,
        component_name="teacher_topk_reference",
    )
    teacher_topk_indices = build_topk_indices(teacher_topk_embeddings, top_k=top_k)

    threshold_summary_df, threshold_predictions_df = evaluate_texas_thresholds(
        metadata_df=pseudo_df,
        embeddings=texas_embeddings,
        thresholds=[float(value) for value in thresholds],
        anchor_threshold=float(anchor_threshold),
        candidate_pair_df=candidate_pair_df,
        teacher_anchor_labels=teacher_anchor_labels,
        teacher_topk_indices=teacher_topk_indices,
        top_k=top_k,
    )
    best_threshold_df = pick_best_texas_threshold(threshold_summary_df)
    best_threshold = float(best_threshold_df.iloc[0]["threshold"])
    chosen_predictions_df = threshold_predictions_df[
        np.isclose(threshold_predictions_df["threshold"].to_numpy(dtype=float), best_threshold, rtol=0.0, atol=1e-9)
    ].copy()
    chosen_predictions_df["route_name"] = str(route_name)
    chosen_predictions_df["embedding_dim"] = int(texas_embeddings.shape[1])

    base_pred_df = pd.read_csv(base_predictions.resolve())
    base_pred_df["image_id"] = base_pred_df["image_id"].astype(str)
    base_pred_df["dataset"] = base_pred_df["dataset"].astype(str)
    kept_df = base_pred_df[base_pred_df["dataset"] != TEXAS_DATASET].copy()
    merged_pred_df = pd.concat([kept_df, chosen_predictions_df], ignore_index=True)
    route_df = _build_route_summary(merged_pred_df)
    cluster_summary_df = _build_cluster_summary(merged_pred_df)

    np.save(embeddings_dir / "texas_ensemble_test_embeddings.npy", texas_embeddings.astype(np.float32))
    texas_df.to_csv(embeddings_dir / "texas_ensemble_test_metadata.csv", index=False)
    component_df.to_csv(tables_dir / "component_table_v1.csv", index=False)
    threshold_summary_df.to_csv(tables_dir / "threshold_sweep_v1.csv", index=False)
    threshold_predictions_df.to_csv(tables_dir / "threshold_predictions_v1.csv", index=False)
    best_threshold_df.to_csv(tables_dir / "best_threshold_v1.csv", index=False)
    merged_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)
    route_df.to_csv(tables_dir / "route_config_v1.csv", index=False)
    cluster_summary_df.to_csv(tables_dir / "cluster_summary_v1.csv", index=False)

    submission_path = output_dir / "submission.csv"
    build_submission(
        test_pred_df=merged_pred_df,
        sample_submission_path=sample_submission_path.resolve(),
        output_path=submission_path,
    )

    config = {
        "route_name": str(route_name),
        "base_predictions": str(base_predictions.resolve()),
        "source_dirs": [str(path.resolve()) for path in source_dirs],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else "",
        "weights": [float(value) for value in ([1.0] * max(1, len(component_df)) if weights is None else weights)],
        "thresholds": [float(value) for value in thresholds],
        "anchor_threshold": float(anchor_threshold),
        "teacher_anchor_predictions_path": str(teacher_anchor_predictions_path.resolve()),
        "teacher_topk_source_dir": str(teacher_topk_source_dir.resolve()),
        "pseudo_assignments_path": str(pseudo_assignments_path.resolve()),
        "candidate_pairs_path": str(candidate_pairs_path.resolve()),
        "top_k": int(top_k),
        "test_manifest_path": str(resolved_test_manifest_path) if resolved_test_manifest_path is not None else "",
        "device": str(device),
        "eval_batch_size": int(eval_batch_size),
        "num_workers": int(num_workers),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(
        reports_dir / "summary.md",
        config=config,
        component_df=component_df,
        best_threshold_df=best_threshold_df,
        threshold_summary_df=threshold_summary_df,
        route_df=route_df,
        cluster_summary_df=cluster_summary_df,
    )
    return {
        "submission_path": submission_path,
        "summary_path": reports_dir / "summary.md",
        "prediction_path": tables_dir / "test_predictions_v1.csv",
        "best_threshold_path": tables_dir / "best_threshold_v1.csv",
        "threshold_sweep_path": tables_dir / "threshold_sweep_v1.csv",
        "route_config_path": tables_dir / "route_config_v1.csv",
    }
