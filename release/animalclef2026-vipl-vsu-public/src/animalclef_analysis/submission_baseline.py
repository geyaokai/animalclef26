from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .descriptor_baselines import (
    PATH_COLUMN,
    apply_thresholds_to_df,
    build_average_linkage,
    build_submission,
    cluster_from_linkage,
    dataframe_to_markdown_table,
    ensure_metadata_alignment,
    load_cached_embedding_bundle,
)
from .view_manifests import get_default_manifest_paths
from .orb_rerank_baseline import (
    apply_local_rerank,
    build_local_match_table,
    build_topk_pair_index,
    cosine_score_matrix,
    extract_orb_features,
    score_matrix_to_distance,
)


DEFAULT_LYNX_CHECKPOINT = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/checkpoints/best.pt")
DEFAULT_LYNX_THRESHOLD_TABLE = Path("artifacts/training/experiments/ft_mega_arcface_distill_v1/tables/best_thresholds_v1.csv")
DEFAULT_FUSION_SOURCE_DIR = Path("artifacts/descriptor_baselines/embed_fusion_v1")
DEFAULT_ORB_SOURCE_DIR = Path("artifacts/descriptor_baselines/orb_rerank_v1")
DEFAULT_TEXAS_SELFTRAIN_SOURCE_DIR = Path("artifacts/training/experiments/ft_texas_miew_pseudo_v1")
DEFAULT_TEXAS_SELFTRAIN_THRESHOLD = 0.44


def _require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in runtime env
        raise ModuleNotFoundError("Submission baseline requires torch. Run in the 'wildfusion' environment.") from exc
    return torch


def _load_threshold_map(table_path: Path) -> dict[str, float]:
    table_df = pd.read_csv(table_path)
    return {
        str(row["dataset"]): float(row["threshold"])
        for _, row in table_df.iterrows()
    }


def _load_gate_map(table_path: Path) -> dict[str, dict[str, Any]]:
    table_df = pd.read_csv(table_path)
    gate_map: dict[str, dict[str, Any]] = {}
    for _, row in table_df.iterrows():
        gate_map[str(row["dataset"])] = {
            "enable_rerank": bool(row["enable_rerank"]),
            "chosen_local_weight": float(row["chosen_local_weight"]),
        }
    return gate_map


def _reorder_metadata_and_embeddings_to_reference(
    *,
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    embeddings: np.ndarray,
    split_name: str,
    candidate_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(candidate_df) != len(embeddings):
        raise ValueError(
            f"{split_name} metadata row mismatch for {candidate_name}: df={len(candidate_df)} vs embeddings={len(embeddings)}"
        )
    join_columns = ["image_id", "dataset"]
    audit_columns = join_columns + [PATH_COLUMN]
    reference_view = reference_df[audit_columns].copy().reset_index(drop=True)
    candidate_view = candidate_df[audit_columns].copy().reset_index(drop=True)
    reference_view["image_id"] = reference_view["image_id"].astype(str)
    candidate_view["image_id"] = candidate_view["image_id"].astype(str)
    merged = reference_view.merge(
        candidate_view.assign(_embedding_row=np.arange(len(candidate_view), dtype=np.int32)),
        on=join_columns,
        how="left",
        validate="one_to_one",
        suffixes=("_ref", ""),
    )
    if merged["_embedding_row"].isna().any():
        missing_rows = (
            merged.loc[merged["_embedding_row"].isna(), join_columns]
            .head(5)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"{split_name} missing image_id/dataset pairs in {candidate_name}, examples: {missing_rows}"
        )

    reorder_index = merged["_embedding_row"].astype(int).to_numpy()
    reordered_df = candidate_df.iloc[reorder_index].reset_index(drop=True).copy()
    reordered_embeddings = embeddings[reorder_index]
    metadata_columns = [column for column in ["image_id", "dataset", "identity"] if column in reference_df.columns and column in reordered_df.columns]
    ensure_metadata_alignment(
        reference_df=reference_df.loc[:, metadata_columns].reset_index(drop=True),
        candidate_df=reordered_df.loc[:, metadata_columns].reset_index(drop=True),
        split_name=split_name,
        reference_name="reference_metadata",
        candidate_name=candidate_name,
    )
    return reordered_df, reordered_embeddings


def _cluster_single_dataset_from_score_matrix(
    dataset_df: pd.DataFrame,
    score_matrix: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    if dataset_df["dataset"].nunique() != 1:
        raise ValueError("Expected a single dataset slice when clustering from a score matrix")
    dataset_name = str(dataset_df["dataset"].iloc[0])
    distance = score_matrix_to_distance(score_matrix)
    linkage_matrix = build_average_linkage(distance)
    pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), threshold)
    result = dataset_df.copy().reset_index(drop=True)
    result["chosen_threshold"] = float(threshold)
    result["pred_cluster_id"] = pred_labels
    result["cluster_label"] = [f"cluster_{dataset_name}_{int(label)}" for label in pred_labels]
    return result


def _load_supervised_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
) -> tuple[Any, Any, dict[str, Any], Any]:
    torch = _require_torch()
    from .supervised_training import SupervisedEmbeddingModel, load_student_backbone

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = dict(checkpoint["config"])
    backbone, spec = load_student_backbone(str(config["student_backbone"]), device=device)
    model = SupervisedEmbeddingModel(
        backbone=backbone,
        feature_dim=int(config["student_feature_dim"]),
        embedding_dim=int(config["embedding_dim"]),
        dataset_class_counts={str(key): int(value) for key, value in dict(config["fit_classes"]).items()},
        teacher_dim=int(config["teacher_dim"]),
        arcface_scale=float(config["arcface_scale"]),
        arcface_margin=float(config["arcface_margin"]),
        salamander_subcenter_k=int(config.get("salamander_subcenter_k", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, spec, config, checkpoint


def _write_markdown_report(
    output_path: Path,
    config: dict[str, Any],
    route_df: pd.DataFrame,
    cluster_summary_df: pd.DataFrame,
    artifact_df: pd.DataFrame,
) -> None:
    architecture_lines = [
        "- Overall system: `dataset-routed hybrid clustering pipeline`.",
        "- Global flow: `image -> dataset branch -> embedding -> optional local rerank -> average-linkage clustering -> submission cluster label`.",
        "- Branches:",
    ]
    for row in route_df.itertuples(index=False):
        rerank_text = (
            f"ORB rerank enabled, local_weight={float(row.local_weight):.2f}"
            if bool(row.rerank_enabled)
            else "no local rerank"
        )
        if str(row.dataset) == "LynxID2025":
            branch = (
                f"  - `{row.dataset}`: supervised `MegaDescriptor-L-384` student, "
                f"`B x 3 x 384 x 384 -> B x 1536 -> B x 512`, threshold `{float(row.threshold)}`; {rerank_text}."
            )
        elif str(row.dataset) == "TexasHornedLizards" and "ft_texas" in str(row.route_name).lower():
            branch = (
                f"  - `{row.dataset}`: Texas-only self-trained `MiewID` student, "
                f"`B x 3 x 440 x 440 -> B x 2152 -> B x 512`, "
                f"trained with pseudo `ArcFace` on consensus seed clusters plus frozen-teacher distillation, "
                f"threshold `{float(row.threshold)}`; {rerank_text}."
            )
        elif str(row.route_name).startswith("fusion"):
            branch = (
                f"  - `{row.dataset}`: frozen early fusion, "
                f"`Mega B x 1536 + Miew B x 2152 -> concat B x 3688 -> L2 normalize`, "
                f"threshold `{float(row.threshold)}`; {rerank_text}."
            )
        elif "miew" in str(row.route_name).lower():
            branch = (
                f"  - `{row.dataset}`: frozen `MiewID-msv3`, "
                f"`B x 3 x 440 x 440 -> B x 2152`, threshold `{float(row.threshold)}`; {rerank_text}."
            )
        elif "mega" in str(row.route_name).lower():
            branch = (
                f"  - `{row.dataset}`: frozen `MegaDescriptor-L-384`, "
                f"`B x 3 x 384 x 384 -> B x 1536`, threshold `{float(row.threshold)}`; {rerank_text}."
            )
        else:
            branch = (
                f"  - `{row.dataset}`: route `{row.route_name}`, embedding dim `{int(row.embedding_dim)}`, "
                f"threshold `{float(row.threshold)}`; {rerank_text}."
            )
        architecture_lines.append(branch)

    lines = [
        "# Kaggle Submission Baseline",
        "",
        "## Route Summary",
        "",
        dataframe_to_markdown_table(
            route_df[
                [
                    "dataset",
                    "route_name",
                    "embedding_dim",
                    "threshold",
                    "rerank_enabled",
                    "local_weight",
                ]
            ]
        ),
        "",
        "## Test Cluster Summary",
        "",
        dataframe_to_markdown_table(cluster_summary_df),
        "",
        "## Architecture",
        "",
        *architecture_lines,
        "",
        "## Key Artifacts",
        "",
        dataframe_to_markdown_table(artifact_df),
        "",
        "## Notes",
        "",
        f"- Sample submission path: `{config['sample_submission_path']}`",
        f"- Test manifest path: `{config['test_manifest_path']}`",
        f"- Fusion source dir: `{config['fusion_source_dir']}`",
        f"- Texas self-train source dir: `{config['texas_source_dir']}`",
        f"- Texas self-train threshold: `{config['texas_threshold']}`",
        f"- ORB source dir: `{config['orb_source_dir']}`",
        f"- Lynx checkpoint: `{config['lynx_checkpoint_path']}`",
        f"- Device: `{config['device']}`",
        f"- Eval batch size: `{config['eval_batch_size']}`",
        f"- Num workers: `{config['num_workers']}`",
        "",
        "## Key Performance Tricks",
        "",
        "- `dataset routing`: do not force one backbone onto all species; current route mixes supervised `Mega` for `Lynx`, reranked fusion for `Salamander`, frozen fusion for `SeaTurtle`, and Texas-only self-train for `TexasHornedLizards`.",
        "- `early fusion`: `Mega(1536) + Miew(2152) -> 3688` improves the local validation baseline over single descriptors.",
        "- `dataset-specific rerank gate`: only `SalamanderID2025` keeps ORB rerank; `Lynx` and `SeaTurtle` stay on plain global embeddings because validation showed no gain or harm.",
        "- `dataset-specific thresholds`: each dataset keeps its own clustering threshold rather than a single global cut.",
        "- `Texas pseudo self-train`: the Texas branch now uses a self-trained `Miew` student (`512-d`) because public LB showed a large gain over frozen threshold tuning.",
        "",
        "- This baseline uses supervised `ft_mega_arcface_distill_v1` only for `LynxID2025`.",
        "- `SalamanderID2025` uses cached `Mega+Miew` fusion embeddings plus ORB local rerank.",
        "- `SeaTurtleID2022` uses cached fusion embeddings without rerank.",
        "- `TexasHornedLizards` uses `ft_texas_miew_pseudo_v1` with threshold `0.44`, replacing the earlier frozen-threshold fallback route.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_submission_baseline(
    repo_root: Path,
    output_dir: Path,
    device: str = "cuda:0",
    eval_batch_size: int = 16,
    num_workers: int = 4,
    sample_submission_path: Path | None = None,
    test_manifest_path: Path | None = None,
    fusion_source_dir: Path | None = None,
    texas_source_dir: Path | None = None,
    texas_threshold: float = DEFAULT_TEXAS_SELFTRAIN_THRESHOLD,
    orb_source_dir: Path | None = None,
    lynx_checkpoint_path: Path | None = None,
    lynx_threshold_table_path: Path | None = None,
) -> dict[str, Path]:
    from .supervised_training import extract_student_embeddings

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    if sample_submission_path is None:
        sample_submission_path = repo_root / "sample_submission.csv"
    if test_manifest_path is None:
        _default_train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
    if fusion_source_dir is None:
        fusion_source_dir = repo_root / DEFAULT_FUSION_SOURCE_DIR
    if texas_source_dir is None:
        texas_source_dir = repo_root / DEFAULT_TEXAS_SELFTRAIN_SOURCE_DIR
    if orb_source_dir is None:
        orb_source_dir = repo_root / DEFAULT_ORB_SOURCE_DIR
    if lynx_checkpoint_path is None:
        lynx_checkpoint_path = repo_root / DEFAULT_LYNX_CHECKPOINT
    if lynx_threshold_table_path is None:
        lynx_threshold_table_path = repo_root / DEFAULT_LYNX_THRESHOLD_TABLE

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    embeddings_dir = output_dir / "embeddings"
    reports_dir = output_dir / "reports"
    for path in [tables_dir, embeddings_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(test_manifest_path)
    test_df["image_id"] = test_df["image_id"].astype(str)
    test_df["identity"] = test_df["identity"].fillna("").astype(str)

    fusion_bundle = load_cached_embedding_bundle(source_dir=fusion_source_dir, name="fusion")
    texas_bundle = load_cached_embedding_bundle(source_dir=texas_source_dir, name="ft_texas_pseudo_v1")
    fusion_test_df, fusion_test_embeddings = _reorder_metadata_and_embeddings_to_reference(
        reference_df=test_df,
        candidate_df=fusion_bundle.test_df,
        embeddings=fusion_bundle.test_embeddings,
        split_name="test",
        candidate_name="fusion_cached_metadata",
    )
    texas_reference_df = test_df[test_df["dataset"] == "TexasHornedLizards"].copy().reset_index(drop=True)
    texas_test_df, texas_test_embeddings = _reorder_metadata_and_embeddings_to_reference(
        reference_df=texas_reference_df,
        candidate_df=texas_bundle.test_df[texas_bundle.test_df["dataset"] == "TexasHornedLizards"].reset_index(drop=True),
        embeddings=texas_bundle.test_embeddings[(texas_bundle.test_df["dataset"] == "TexasHornedLizards").to_numpy()],
        split_name="test_texas",
        candidate_name="texas_selftrain_cached_metadata",
    )

    fusion_threshold_map = _load_threshold_map(fusion_source_dir / "tables" / "test_threshold_sources_v1.csv")
    rerank_threshold_map = _load_threshold_map(orb_source_dir / "tables" / "rerank_best_thresholds_v1.csv")
    gate_map = _load_gate_map(orb_source_dir / "tables" / "dataset_gate_v1.csv")
    lynx_threshold_map = _load_threshold_map(lynx_threshold_table_path)

    prediction_frames: list[pd.DataFrame] = []
    route_rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, str]] = []

    lynx_df = test_df[test_df["dataset"] == "LynxID2025"].copy().reset_index(drop=True)
    lynx_model, lynx_spec, lynx_config, _checkpoint = _load_supervised_model_from_checkpoint(
        checkpoint_path=lynx_checkpoint_path,
        device=device,
    )
    lynx_embeddings = extract_student_embeddings(
        df=lynx_df,
        repo_root=repo_root,
        model=lynx_model,
        spec=lynx_spec,
        device=device,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        preprocess_config=lynx_config.get("resolved_preprocess_config"),
    )
    np.save(embeddings_dir / "lynx_ft_mega_test_embeddings.npy", lynx_embeddings.astype(np.float32))
    lynx_df.to_csv(embeddings_dir / "lynx_ft_mega_test_metadata.csv", index=False)
    lynx_threshold = float(lynx_threshold_map["LynxID2025"])
    lynx_pred_df = apply_thresholds_to_df(
        df=lynx_df,
        embeddings=lynx_embeddings,
        threshold_by_dataset={"LynxID2025": lynx_threshold},
    )
    lynx_pred_df["route_name"] = "ft_mega_arcface_distill_v1"
    lynx_pred_df["embedding_dim"] = int(lynx_embeddings.shape[1])
    lynx_pred_df["rerank_enabled"] = False
    lynx_pred_df["local_weight"] = 0.0
    prediction_frames.append(lynx_pred_df)
    route_rows.append(
        {
            "dataset": "LynxID2025",
            "route_name": "ft_mega_arcface_distill_v1",
            "embedding_dim": int(lynx_embeddings.shape[1]),
            "threshold": lynx_threshold,
            "rerank_enabled": False,
            "local_weight": 0.0,
            "source": str(lynx_checkpoint_path),
        }
    )
    artifact_rows.extend(
        [
            {
                "artifact": "lynx_test_embeddings",
                "path": str(embeddings_dir / "lynx_ft_mega_test_embeddings.npy"),
            },
            {
                "artifact": "lynx_test_metadata",
                "path": str(embeddings_dir / "lynx_ft_mega_test_metadata.csv"),
            },
        ]
    )

    salamander_df = fusion_test_df[fusion_test_df["dataset"] == "SalamanderID2025"].copy().reset_index(drop=True)
    salamander_embeddings = fusion_test_embeddings[(fusion_test_df["dataset"] == "SalamanderID2025").to_numpy()]
    salamander_score = cosine_score_matrix(salamander_embeddings)
    salamander_gate = gate_map["SalamanderID2025"]
    if not bool(salamander_gate["enable_rerank"]):
        raise ValueError("Expected SalamanderID2025 rerank gate to be enabled for the submission baseline")
    salamander_features = extract_orb_features(
        df=salamander_df,
        repo_root=repo_root,
        nfeatures=1024,
        max_side=768,
        fast_threshold=7,
        clahe_clip_limit=2.0,
    )
    salamander_pairs = build_topk_pair_index(
        score_matrix=salamander_score,
        top_k=10,
        query_indices=None,
    )
    salamander_pair_df = build_local_match_table(
        df=salamander_df,
        features=salamander_features,
        pair_index=salamander_pairs,
        ratio_test=0.8,
        ransac_threshold=5.0,
        min_inliers=8,
    )
    salamander_pair_df["local_weight"] = float(salamander_gate["chosen_local_weight"])
    salamander_pair_df.to_csv(tables_dir / "salamander_local_match_scores_v1.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": "SalamanderID2025",
                "image_id": feature.image_id,
                "keypoints": feature.point_count,
                "width": feature.width,
                "height": feature.height,
            }
            for feature in salamander_features
        ]
    ).to_csv(tables_dir / "salamander_orb_keypoints_v1.csv", index=False)
    salamander_reranked_score = apply_local_rerank(
        global_score_matrix=salamander_score,
        pair_df=salamander_pair_df,
        local_weight=float(salamander_gate["chosen_local_weight"]),
    )
    salamander_threshold = float(rerank_threshold_map["SalamanderID2025"])
    salamander_pred_df = _cluster_single_dataset_from_score_matrix(
        dataset_df=salamander_df,
        score_matrix=salamander_reranked_score,
        threshold=salamander_threshold,
    )
    salamander_pred_df["route_name"] = "fusion_orb_rerank_v1"
    salamander_pred_df["embedding_dim"] = int(salamander_embeddings.shape[1])
    salamander_pred_df["rerank_enabled"] = True
    salamander_pred_df["local_weight"] = float(salamander_gate["chosen_local_weight"])
    prediction_frames.append(salamander_pred_df)
    route_rows.append(
        {
            "dataset": "SalamanderID2025",
            "route_name": "fusion_orb_rerank_v1",
            "embedding_dim": int(salamander_embeddings.shape[1]),
            "threshold": salamander_threshold,
            "rerank_enabled": True,
            "local_weight": float(salamander_gate["chosen_local_weight"]),
            "source": str(fusion_source_dir),
        }
    )
    artifact_rows.extend(
        [
            {
                "artifact": "salamander_local_match_scores",
                "path": str(tables_dir / "salamander_local_match_scores_v1.csv"),
            },
            {
                "artifact": "salamander_orb_keypoints",
                "path": str(tables_dir / "salamander_orb_keypoints_v1.csv"),
            },
        ]
    )

    for dataset in ["SeaTurtleID2022"]:
        dataset_df = fusion_test_df[fusion_test_df["dataset"] == dataset].copy().reset_index(drop=True)
        dataset_embeddings = fusion_test_embeddings[(fusion_test_df["dataset"] == dataset).to_numpy()]
        threshold = float(fusion_threshold_map[dataset])
        dataset_pred_df = apply_thresholds_to_df(
            df=dataset_df,
            embeddings=dataset_embeddings,
            threshold_by_dataset={dataset: threshold},
        )
        dataset_pred_df["route_name"] = "fusion_v1"
        dataset_pred_df["embedding_dim"] = int(dataset_embeddings.shape[1])
        dataset_pred_df["rerank_enabled"] = False
        dataset_pred_df["local_weight"] = 0.0
        prediction_frames.append(dataset_pred_df)
        route_rows.append(
            {
                "dataset": dataset,
                "route_name": "fusion_v1",
                "embedding_dim": int(dataset_embeddings.shape[1]),
                "threshold": threshold,
                "rerank_enabled": False,
                "local_weight": 0.0,
                "source": str(fusion_source_dir),
            }
        )

    texas_df = texas_test_df.copy().reset_index(drop=True)
    texas_embeddings = texas_test_embeddings.astype(np.float32, copy=False)
    np.save(embeddings_dir / "texas_ft_pseudo_test_embeddings.npy", texas_embeddings.astype(np.float32))
    texas_df.to_csv(embeddings_dir / "texas_ft_pseudo_test_metadata.csv", index=False)
    texas_pred_df = apply_thresholds_to_df(
        df=texas_df,
        embeddings=texas_embeddings,
        threshold_by_dataset={"TexasHornedLizards": float(texas_threshold)},
    )
    texas_pred_df["route_name"] = "ft_texas_miew_pseudo_v1"
    texas_pred_df["embedding_dim"] = int(texas_embeddings.shape[1])
    texas_pred_df["rerank_enabled"] = False
    texas_pred_df["local_weight"] = 0.0
    prediction_frames.append(texas_pred_df)
    route_rows.append(
        {
            "dataset": "TexasHornedLizards",
            "route_name": "ft_texas_miew_pseudo_v1",
            "embedding_dim": int(texas_embeddings.shape[1]),
            "threshold": float(texas_threshold),
            "rerank_enabled": False,
            "local_weight": 0.0,
            "source": str(texas_source_dir),
        }
    )
    artifact_rows.extend(
        [
            {
                "artifact": "texas_test_embeddings",
                "path": str(embeddings_dir / "texas_ft_pseudo_test_embeddings.npy"),
            },
            {
                "artifact": "texas_test_metadata",
                "path": str(embeddings_dir / "texas_ft_pseudo_test_metadata.csv"),
            },
        ]
    )

    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    prediction_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    route_df = pd.DataFrame(route_rows).sort_values("dataset").reset_index(drop=True)
    route_df.to_csv(tables_dir / "route_config_v1.csv", index=False)

    cluster_summary_rows: list[dict[str, Any]] = []
    for dataset, dataset_df in prediction_df.groupby("dataset"):
        counts = dataset_df["pred_cluster_id"].value_counts()
        cluster_summary_rows.append(
            {
                "dataset": dataset,
                "samples": int(len(dataset_df)),
                "clusters": int(counts.size),
                "singleton_clusters": int((counts == 1).sum()),
                "singleton_ratio": round(float((counts == 1).mean()) if len(counts) else 0.0, 6),
                "route_name": str(dataset_df["route_name"].iloc[0]),
                "embedding_dim": int(dataset_df["embedding_dim"].iloc[0]),
                "threshold": float(dataset_df["chosen_threshold"].iloc[0]),
            }
        )
    cluster_summary_df = pd.DataFrame(cluster_summary_rows).sort_values("dataset").reset_index(drop=True)
    cluster_summary_df.to_csv(tables_dir / "test_cluster_summary_v1.csv", index=False)

    submission_path = output_dir / "submission.csv"
    build_submission(
        test_pred_df=prediction_df,
        sample_submission_path=sample_submission_path,
        output_path=submission_path,
    )

    artifact_rows.extend(
        [
            {"artifact": "submission", "path": str(submission_path)},
            {"artifact": "test_predictions", "path": str(tables_dir / "test_predictions_v1.csv")},
            {"artifact": "route_config", "path": str(tables_dir / "route_config_v1.csv")},
            {"artifact": "test_cluster_summary", "path": str(tables_dir / "test_cluster_summary_v1.csv")},
        ]
    )
    artifact_df = pd.DataFrame(artifact_rows)
    artifact_df.to_csv(tables_dir / "artifacts_v1.csv", index=False)

    config = {
        "device": device,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "sample_submission_path": str(sample_submission_path),
        "test_manifest_path": str(test_manifest_path),
        "fusion_source_dir": str(fusion_source_dir),
        "texas_source_dir": str(texas_source_dir),
        "texas_threshold": float(texas_threshold),
        "orb_source_dir": str(orb_source_dir),
        "lynx_checkpoint_path": str(lynx_checkpoint_path),
        "lynx_threshold_table_path": str(lynx_threshold_table_path),
    }
    (reports_dir / "summary.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(
        output_path=reports_dir / "summary.md",
        config=config,
        route_df=route_df,
        cluster_summary_df=cluster_summary_df,
        artifact_df=artifact_df,
    )

    return {
        "submission_path": submission_path,
        "summary_path": reports_dir / "summary.md",
        "prediction_path": tables_dir / "test_predictions_v1.csv",
        "route_config_path": tables_dir / "route_config_v1.csv",
        "cluster_summary_path": tables_dir / "test_cluster_summary_v1.csv",
    }
