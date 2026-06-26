from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .initial_audit import create_contact_sheet
from .view_manifests import PATH_COLUMN, get_default_manifest_paths

try:  # pragma: no cover - exercised in the GPU environment
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T
except ModuleNotFoundError:  # pragma: no cover - allows light unit tests outside the training env
    torch = None
    DataLoader = Dataset = object
    T = None

try:  # pragma: no cover - exercised in the training env
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
except ModuleNotFoundError:  # pragma: no cover - allows module import in lighter envs
    fcluster = linkage = squareform = None

try:  # pragma: no cover - exercised in the training env
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix
except ModuleNotFoundError:  # pragma: no cover - allows module import in lighter envs
    adjusted_rand_score = normalized_mutual_info_score = pair_confusion_matrix = None


LABELED_DATASETS = ["LynxID2025", "SalamanderID2025", "SeaTurtleID2022"]
TEST_ONLY_DATASETS = ["TexasHornedLizards"]
DEFAULT_THRESHOLDS = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]


@dataclass(frozen=True)
class DescriptorSpec:
    name: str
    model_id: str
    family: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    default_batch_size: int


@dataclass(frozen=True)
class CachedEmbeddingBundle:
    name: str
    source_dir: Path
    weight: float
    val_embeddings: np.ndarray
    test_embeddings: np.ndarray
    val_df: pd.DataFrame
    test_df: pd.DataFrame


DESCRIPTOR_SPECS = {
    "mega": DescriptorSpec(
        name="mega",
        model_id="BVRA/MegaDescriptor-L-384",
        family="timm_hf",
        input_size=384,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        default_batch_size=16,
    ),
    "miew": DescriptorSpec(
        name="miew",
        model_id="conservationxlabs/miewid-msv3",
        family="transformers_hf",
        input_size=440,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        default_batch_size=12,
    ),
}


class ImagePathDataset(Dataset):
    def __init__(self, df: pd.DataFrame, repo_root: Path, image_size: int, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        _require_torch()
        self.df = df.reset_index(drop=True).copy()
        self.repo_root = repo_root
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[index]
        image_path = self.repo_root / row[PATH_COLUMN]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, str(row["image_id"])


def load_manifests(
    repo_root: Path,
    train_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_manifest_path is None:
        train_manifest_path, default_test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
        if test_manifest_path is None:
            test_manifest_path = default_test_manifest_path
    if test_manifest_path is None:
        _default_train_manifest_path, test_manifest_path = get_default_manifest_paths(repo_root=repo_root)
    train_df = pd.read_csv(train_manifest_path)
    test_df = pd.read_csv(test_manifest_path)
    for frame in [train_df, test_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        frame["identity"] = frame["identity"].fillna("")
    return train_df, test_df


def build_identity_holdout_split(
    train_df: pd.DataFrame,
    val_identity_fraction: float,
    seed: int,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    if not 0 < val_identity_fraction < 1:
        raise ValueError("val_identity_fraction must be between 0 and 1")
    if datasets is None:
        datasets = LABELED_DATASETS

    rows: list[pd.DataFrame] = []
    rng = random.Random(seed)
    for dataset in datasets:
        dataset_df = train_df[(train_df["dataset"] == dataset) & (train_df["identity"] != "")].copy()
        if dataset_df.empty:
            continue
        dataset_df["identity_total_images"] = (
            dataset_df.groupby("identity", dropna=False)["image_id"].transform("size").astype(int)
        )
        dataset_df["identity_holdout_eligible_v1"] = dataset_df["identity_total_images"] >= 2
        eligible_identities = sorted(
            dataset_df.loc[dataset_df["identity_holdout_eligible_v1"], "identity"].unique().tolist()
        )
        if len(eligible_identities) < 2:
            raise ValueError(
                f"Need at least 2 non-singleton identities to build holdout split for {dataset}; "
                f"found {len(eligible_identities)} eligible identities after excluding singletons from validation"
            )
        rng.shuffle(eligible_identities)
        val_count = max(1, int(round(len(eligible_identities) * val_identity_fraction)))
        val_count = min(val_count, len(eligible_identities) - 1)
        val_identities = set(eligible_identities[:val_count])
        dataset_df["split_role_v1"] = "fit"
        dataset_df.loc[dataset_df["identity"].isin(val_identities), "split_role_v1"] = "val"
        rows.append(dataset_df)
    if not rows:
        return pd.DataFrame(columns=list(train_df.columns) + ["split_role_v1"])
    return pd.concat(rows, ignore_index=True)


def recall_at_k(embeddings: np.ndarray, labels: np.ndarray, k: int) -> float:
    if len(embeddings) < 2:
        return 0.0
    counts = pd.Series(labels).value_counts()
    valid_mask = np.array([counts[label] > 1 for label in labels], dtype=bool)
    if not valid_mask.any():
        return 0.0
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    topk_indices = np.argpartition(-similarity, kth=min(k, len(embeddings) - 1) - 1, axis=1)[:, :k]
    hits = []
    for index, neighbors in enumerate(topk_indices):
        if not valid_mask[index]:
            continue
        hits.append(bool(np.any(labels[neighbors] == labels[index])))
    return round(float(np.mean(hits)), 6) if hits else 0.0


def build_neighbor_table(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    if len(df) < 2:
        return pd.DataFrame(
            columns=[
                "dataset",
                "image_id",
                "identity",
                "neighbor_rank",
                "neighbor_image_id",
                "neighbor_identity",
                "similarity",
                "same_identity",
            ]
        )
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    width = min(top_k, len(df) - 1)
    topk_indices = np.argpartition(-similarity, kth=width - 1, axis=1)[:, :width]
    rows: list[dict[str, object]] = []
    for index, row in enumerate(df.itertuples(index=False)):
        ranked = sorted(topk_indices[index].tolist(), key=lambda idx: similarity[index, idx], reverse=True)
        for rank, neighbor_idx in enumerate(ranked, start=1):
            neighbor = df.iloc[neighbor_idx]
            rows.append(
                {
                    "dataset": row.dataset,
                    "image_id": row.image_id,
                    "identity": row.identity,
                    "neighbor_rank": rank,
                    "neighbor_image_id": neighbor["image_id"],
                    "neighbor_identity": neighbor["identity"],
                    "similarity": round(float(similarity[index, neighbor_idx]), 6),
                    "same_identity": bool(row.identity == neighbor["identity"]),
                }
            )
    return pd.DataFrame(rows)


def pairwise_precision_recall(true_labels: np.ndarray, pred_labels: np.ndarray) -> tuple[float, float, float]:
    _require_clustering_deps()
    confusion = pair_confusion_matrix(true_labels, pred_labels)
    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]
    del true_negative
    precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    return round(precision, 6), round(recall, 6), round(f1, 6)


def summarize_cluster_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict[str, float]:
    _require_clustering_deps()
    precision, recall, f1 = pairwise_precision_recall(true_labels, pred_labels)
    counts = pd.Series(pred_labels).value_counts()
    singleton_ratio = float((counts == 1).mean()) if len(counts) else 0.0
    return {
        "ari": round(float(adjusted_rand_score(true_labels, pred_labels)), 6),
        "nmi": round(float(normalized_mutual_info_score(true_labels, pred_labels)), 6),
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "cluster_count": int(len(counts)),
        "singleton_cluster_ratio": round(singleton_ratio, 6),
    }


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def build_average_linkage(distance_matrix: np.ndarray) -> np.ndarray | None:
    _require_clustering_deps()
    if len(distance_matrix) < 2:
        return None
    condensed = squareform(distance_matrix, checks=False)
    return linkage(condensed, method="average")


def cluster_from_linkage(linkage_matrix: np.ndarray | None, sample_count: int, threshold: float) -> np.ndarray:
    _require_clustering_deps()
    if sample_count == 0:
        return np.array([], dtype=int)
    if sample_count == 1 or linkage_matrix is None:
        return np.zeros(sample_count, dtype=int)
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    _, normalized = np.unique(labels, return_inverse=True)
    return normalized.astype(int)


def run_threshold_sweep(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_clustering_deps()
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].reset_index(drop=True)
        dataset_embeddings = embeddings[df["dataset"] == dataset]
        distance = cosine_distance_matrix(dataset_embeddings)
        linkage_matrix = build_average_linkage(distance)
        labels_true = dataset_df["identity"].to_numpy()
        for threshold in thresholds:
            pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), threshold)
            metrics = summarize_cluster_metrics(true_labels=labels_true, pred_labels=pred_labels)
            rows.append(
                {
                    "dataset": dataset,
                    "threshold": threshold,
                    "samples": int(len(dataset_df)),
                    **metrics,
                }
            )
            frame = dataset_df[["image_id", "dataset", "identity", PATH_COLUMN]].copy()
            frame["threshold"] = threshold
            frame["pred_cluster_id"] = pred_labels
            prediction_frames.append(frame)
    sweep_df = pd.DataFrame(rows).sort_values(["dataset", "threshold"]).reset_index(drop=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return sweep_df, prediction_df


def pick_best_thresholds(sweep_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for dataset, dataset_df in sweep_df.groupby("dataset"):
        best = dataset_df.sort_values(
            ["ari", "nmi", "pairwise_f1", "threshold"],
            ascending=[False, False, False, True],
        ).iloc[0]
        rows.append(best)
    return pd.DataFrame(rows).reset_index(drop=True)


def apply_thresholds_to_df(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    threshold_by_dataset: dict[str, float],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].reset_index(drop=True)
        dataset_embeddings = embeddings[df["dataset"] == dataset]
        threshold = threshold_by_dataset[dataset]
        distance = cosine_distance_matrix(dataset_embeddings)
        linkage_matrix = build_average_linkage(distance)
        pred_labels = cluster_from_linkage(linkage_matrix, len(dataset_df), threshold)
        frame = dataset_df.copy()
        frame["chosen_threshold"] = threshold
        frame["pred_cluster_id"] = pred_labels
        frame["cluster_label"] = [f"cluster_{dataset}_{int(label)}" for label in pred_labels]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=df.columns)


def attach_embedding_rows(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    result["embedding_row"] = np.arange(len(result))
    return result


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


def _coerce_model_output(output: Any) -> torch.Tensor:
    _require_torch()
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and isinstance(output.pooler_output, torch.Tensor):
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and isinstance(output.last_hidden_state, torch.Tensor):
        return output.last_hidden_state[:, 0]
    if isinstance(output, (list, tuple)) and output:
        return _coerce_model_output(output[0])
    if isinstance(output, dict):
        for key in ["embeddings", "pooler_output", "last_hidden_state", "logits"]:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value[:, 0] if key == "last_hidden_state" and value.ndim == 3 else value
    raise TypeError(f"Unsupported model output type: {type(output)}")


def load_descriptor_model(descriptor: str, device: str) -> tuple[torch.nn.Module, DescriptorSpec]:
    _require_torch()
    spec = DESCRIPTOR_SPECS[descriptor]
    if spec.family == "timm_hf":
        import timm

        # Let timm resolve the Hugging Face checkpoint wrapper instead of manually
        # unpacking the training checkpoint, which can include optimizer/scheduler state.
        model = timm.create_model(
            f"hf-hub:{spec.model_id}",
            pretrained=True,
            num_classes=0,
        )
    elif spec.family == "transformers_hf":
        import importlib
        import sys

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        config_path = Path(hf_hub_download(repo_id=spec.model_id, filename="config.json"))
        weights_path = hf_hub_download(repo_id=spec.model_id, filename="model.safetensors")
        remote_module_path = next(
            (Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules").rglob("modeling_miewid.py")
        ).parent
        sys.path.insert(0, str((Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules").resolve()))
        package_base = ".".join(remote_module_path.relative_to((Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules")).parts)
        config_module = importlib.import_module(f"{package_base}.configuration_miewid")
        modeling_module = importlib.import_module(f"{package_base}.modeling_miewid")

        config_dict = json.loads(config_path.read_text(encoding="utf-8"))
        config_dict["pretrained"] = False
        config = config_module.MiewIdNetConfig(**config_dict)
        model = modeling_module.MiewIdNet(config)
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict, strict=True)
    else:  # pragma: no cover - guarded by static specs
        raise ValueError(f"Unsupported descriptor family: {spec.family}")

    model = model.to(device)
    model.eval()
    return model, spec


def extract_embeddings(
    df: pd.DataFrame,
    repo_root: Path,
    model: torch.nn.Module,
    spec: DescriptorSpec,
    device: str,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    _require_torch()
    dataset = ImagePathDataset(
        df=df,
        repo_root=repo_root,
        image_size=spec.input_size,
        mean=spec.mean,
        std=spec.std,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.startswith("cuda"))
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _image_ids in loader:
            images = images.to(device, non_blocking=True)
            output = model(images)
            tensor = _coerce_model_output(output)
            if tensor.ndim > 2:
                tensor = tensor.flatten(start_dim=1)
            rows.append(tensor.detach().cpu().numpy().astype(np.float32))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return l2_normalize(np.concatenate(rows, axis=0))


def write_embeddings(embeddings: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def dataframe_to_markdown_table(frame: pd.DataFrame) -> str:
    def _escape_markdown_cell(value: object) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace("|", "\\|")
        text = text.replace("\n", "<br>")
        return text

    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(_escape_markdown_cell(row[column]) for column in columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, separator, *rows]) if rows else "\n".join([header, separator])


def ensure_metadata_alignment(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    split_name: str,
    reference_name: str,
    candidate_name: str,
) -> None:
    if len(reference_df) != len(candidate_df):
        raise ValueError(
            f"{split_name} metadata row count mismatch: {reference_name}={len(reference_df)} vs "
            f"{candidate_name}={len(candidate_df)}"
        )
    shared_columns = [
        column
        for column in ["image_id", "dataset", "identity", PATH_COLUMN]
        if column in reference_df.columns and column in candidate_df.columns
    ]
    reference_view = reference_df[shared_columns].fillna("").astype(str).reset_index(drop=True)
    candidate_view = candidate_df[shared_columns].fillna("").astype(str).reset_index(drop=True)
    if reference_view.equals(candidate_view):
        return

    mismatch_mask = (reference_view != candidate_view).any(axis=1).to_numpy()
    mismatch_index = int(np.flatnonzero(mismatch_mask)[0])
    raise ValueError(
        f"{split_name} metadata order mismatch between {reference_name} and {candidate_name} at row {mismatch_index}: "
        f"{reference_view.iloc[mismatch_index].to_dict()} != {candidate_view.iloc[mismatch_index].to_dict()}"
    )


def load_cached_embedding_bundle(source_dir: Path, name: str | None = None, weight: float = 1.0) -> CachedEmbeddingBundle:
    source_dir = source_dir.resolve()
    embeddings_dir = source_dir / "embeddings"
    val_path = embeddings_dir / "val_embeddings.npy"
    test_path = embeddings_dir / "test_embeddings.npy"
    val_metadata_path = embeddings_dir / "val_metadata.csv"
    test_metadata_path = embeddings_dir / "test_metadata.csv"
    for path in [val_path, test_path, val_metadata_path, test_metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing cached embedding artifact: {path}")

    val_embeddings = np.load(val_path).astype(np.float32)
    test_embeddings = np.load(test_path).astype(np.float32)
    val_df = pd.read_csv(val_metadata_path)
    test_df = pd.read_csv(test_metadata_path)
    for frame in [val_df, test_df]:
        frame["image_id"] = frame["image_id"].astype(str)
        if "identity" in frame.columns:
            frame["identity"] = frame["identity"].fillna("")

    if len(val_df) != len(val_embeddings):
        raise ValueError(f"Validation embeddings do not match metadata rows for {source_dir}")
    if len(test_df) != len(test_embeddings):
        raise ValueError(f"Test embeddings do not match metadata rows for {source_dir}")

    return CachedEmbeddingBundle(
        name=name or source_dir.name,
        source_dir=source_dir,
        weight=float(weight),
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        val_df=val_df,
        test_df=test_df,
    )


def fuse_embedding_blocks(embeddings_list: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    if not embeddings_list:
        raise ValueError("Need at least one embedding block to fuse")
    if weights is None:
        weights = [1.0] * len(embeddings_list)
    if len(weights) != len(embeddings_list):
        raise ValueError("weights length must match embeddings_list length")

    sample_count = embeddings_list[0].shape[0]
    fused_blocks: list[np.ndarray] = []
    for index, (embeddings, weight) in enumerate(zip(embeddings_list, weights, strict=True)):
        if embeddings.ndim != 2:
            raise ValueError(f"Embedding block at index {index} must be 2D, got shape={embeddings.shape}")
        if embeddings.shape[0] != sample_count:
            raise ValueError(
                f"Embedding block row mismatch at index {index}: expected {sample_count}, got {embeddings.shape[0]}"
            )
        normalized = l2_normalize(embeddings.astype(np.float32, copy=False))
        fused_blocks.append(normalized * float(weight))
    fused = np.concatenate(fused_blocks, axis=1).astype(np.float32, copy=False)
    return l2_normalize(fused)


def build_fusion_component_table(bundles: list[CachedEmbeddingBundle]) -> pd.DataFrame:
    rows = [
        {
            "component": bundle.name,
            "source_dir": str(bundle.source_dir),
            "weight": bundle.weight,
            "val_dim": int(bundle.val_embeddings.shape[1]),
            "test_dim": int(bundle.test_embeddings.shape[1]),
        }
        for bundle in bundles
    ]
    return pd.DataFrame(rows)


def build_submission(test_pred_df: pd.DataFrame, sample_submission_path: Path, output_path: Path) -> None:
    sample_df = pd.read_csv(sample_submission_path)
    sample_df["image_id"] = sample_df["image_id"].astype(str)
    merged = sample_df[["image_id"]].merge(
        test_pred_df[["image_id", "cluster_label"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    if merged["cluster_label"].isna().any():
        missing = merged[merged["cluster_label"].isna()]["image_id"].tolist()[:5]
        raise ValueError(f"Missing cluster labels for test rows, examples: {missing}")
    merged = merged.rename(columns={"cluster_label": "cluster"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)


def sample_predicted_cluster_rows(
    df: pd.DataFrame,
    images_per_cluster: int,
    max_clusters: int,
    seed: int,
) -> pd.DataFrame:
    subset = df.copy()
    counts = subset.groupby("pred_cluster_id").size().sort_values(ascending=False)
    eligible = counts[counts >= 2].index.tolist()
    if not eligible:
        return pd.DataFrame(columns=subset.columns)
    rng = random.Random(seed)
    if len(eligible) > max_clusters:
        eligible = rng.sample(eligible, k=max_clusters)
    rows: list[pd.DataFrame] = []
    for cluster_id in eligible:
        cluster_df = subset[subset["pred_cluster_id"] == cluster_id]
        take = min(images_per_cluster, len(cluster_df))
        rows.append(cluster_df.sample(n=take, random_state=seed))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=subset.columns)


def create_pair_contact_sheet(
    rows_df: pd.DataFrame,
    repo_root: Path,
    output_path: Path,
    title: str,
    left_path_column: str,
    right_path_column: str,
    caption_left: str,
    caption_right: str,
    columns: int = 2,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    if rows_df.empty:
        return
    margin = 12
    header_h = 34
    label_h = 52
    pair_gap = 8
    panel_w, panel_h = thumb_size
    cell_w = panel_w * 2 + pair_gap
    cell_h = panel_h + label_h
    rows = math.ceil(len(rows_df) / columns)
    width = margin * 2 + columns * cell_w + (columns - 1) * margin
    height = margin * 2 + header_h + rows * cell_h + (rows - 1) * margin
    canvas = Image.new("RGB", (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((margin, margin), title, fill=(20, 20, 20), font=font)
    start_y = margin + header_h

    for idx, row in enumerate(rows_df.itertuples(index=False)):
        gx = idx % columns
        gy = idx // columns
        x = margin + gx * (cell_w + margin)
        y = start_y + gy * (cell_h + margin)
        left = ImageOps.pad(Image.open(repo_root / getattr(row, left_path_column)).convert("RGB"), thumb_size, color=(10, 10, 10))
        right = ImageOps.pad(Image.open(repo_root / getattr(row, right_path_column)).convert("RGB"), thumb_size, color=(10, 10, 10))
        canvas.paste(left, (x, y))
        canvas.paste(right, (x + panel_w + pair_gap, y))
        caption = (
            f"{caption_left}:{getattr(row, 'image_id')} ({getattr(row, 'identity')})\n"
            f"{caption_right}:{getattr(row, 'neighbor_image_id')} ({getattr(row, 'neighbor_identity')})\n"
            f"sim={getattr(row, 'similarity'):.3f}"
        )
        draw.multiline_text((x, y + panel_h + 4), caption, fill=(30, 30, 30), font=font, spacing=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def create_baseline_qualitative_outputs(
    val_pred_df: pd.DataFrame,
    neighbor_df: pd.DataFrame,
    repo_root: Path,
    qualitative_dir: Path,
    seed: int,
) -> None:
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(val_pred_df["dataset"].unique()):
        dataset_df = val_pred_df[val_pred_df["dataset"] == dataset].copy()
        cluster_preview = sample_predicted_cluster_rows(
            df=dataset_df,
            images_per_cluster=4,
            max_clusters=4,
            seed=seed,
        )
        if not cluster_preview.empty:
            create_contact_sheet(
                df=cluster_preview,
                repo_root=repo_root,
                output_path=qualitative_dir / f"predicted_clusters_{dataset}.jpg",
                title=f"Predicted Clusters | {dataset} | validation",
                caption_columns=["dataset", "identity", "pred_cluster_id", "image_id"],
                columns=4,
            )

        wrong_neighbors = neighbor_df[
            (neighbor_df["dataset"] == dataset)
            & (neighbor_df["neighbor_rank"] == 1)
            & (~neighbor_df["same_identity"])
        ].sort_values("similarity", ascending=False).head(8)
        if not wrong_neighbors.empty:
            joined = wrong_neighbors.merge(
                dataset_df[["image_id", PATH_COLUMN]],
                on="image_id",
                how="left",
            ).merge(
                dataset_df[["image_id", PATH_COLUMN]].rename(
                    columns={"image_id": "neighbor_image_id", PATH_COLUMN: "neighbor_path"}
                ),
                on="neighbor_image_id",
                how="left",
            ).rename(columns={PATH_COLUMN: "query_path"})
            create_pair_contact_sheet(
                rows_df=joined,
                repo_root=repo_root,
                output_path=qualitative_dir / f"hard_negatives_{dataset}.jpg",
                title=f"Hard Negatives | {dataset} | validation top-1 wrong neighbors",
                left_path_column="query_path",
                right_path_column="neighbor_path",
                caption_left="q",
                caption_right="nn",
            )


def write_markdown_report(
    output_path: Path,
    descriptor: str,
    spec: DescriptorSpec,
    val_metrics_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    threshold_sources_df: pd.DataFrame,
    fallback_threshold: float,
    config: dict[str, object],
) -> None:
    lines = [
        "# Descriptor Baseline Summary",
        "",
        f"- Descriptor: `{descriptor}`",
        f"- Model ID: `{spec.model_id}`",
        f"- Input size: `{spec.input_size}`",
        f"- Device: `{config['device']}`",
        f"- Validation identity fraction: `{config['val_identity_fraction']}`",
        f"- Threshold sweep: `{config['thresholds']}`",
        f"- Fallback threshold for unlabeled datasets: `{fallback_threshold}`",
        "",
        "## Best Validation Thresholds",
        "",
        dataframe_to_markdown_table(best_thresholds_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]),
        "",
        "## Threshold Sources For Test Clustering",
        "",
        dataframe_to_markdown_table(threshold_sources_df),
        "",
        "## Full Validation Sweep",
        "",
        dataframe_to_markdown_table(val_metrics_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_fusion_markdown_report(
    output_path: Path,
    fusion_name: str,
    component_df: pd.DataFrame,
    val_metrics_df: pd.DataFrame,
    best_thresholds_df: pd.DataFrame,
    threshold_sources_df: pd.DataFrame,
    fallback_threshold: float,
    config: dict[str, object],
) -> None:
    lines = [
        "# Fusion Baseline Summary",
        "",
        f"- Fusion name: `{fusion_name}`",
        f"- Components: `{config['component_names']}`",
        f"- Weights: `{config['weights']}`",
        f"- Fusion mode: `{config['fusion_mode']}`",
        f"- Validation identity fraction: `{config['val_identity_fraction']}`",
        f"- Threshold sweep: `{config['thresholds']}`",
        f"- Fallback threshold for unlabeled datasets: `{fallback_threshold}`",
        "",
        "## Fusion Components",
        "",
        dataframe_to_markdown_table(component_df),
        "",
        "## Best Validation Thresholds",
        "",
        dataframe_to_markdown_table(best_thresholds_df[["dataset", "threshold", "ari", "nmi", "pairwise_f1", "cluster_count", "singleton_cluster_ratio"]]),
        "",
        "## Threshold Sources For Test Clustering",
        "",
        dataframe_to_markdown_table(threshold_sources_df),
        "",
        "## Full Validation Sweep",
        "",
        dataframe_to_markdown_table(val_metrics_df),
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_cached_embedding_baseline(
    repo_root: Path,
    output_dir: Path,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    thresholds: list[float],
    split_seed: int,
    markdown_writer: Callable[[Path, pd.DataFrame, pd.DataFrame, pd.DataFrame, float], None],
    summary_payload: dict[str, object],
    split_df: pd.DataFrame | None = None,
    fit_df: pd.DataFrame | None = None,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    qualitative_dir = output_dir / "qualitative"
    for path in [embeddings_dir, tables_dir, reports_dir, qualitative_dir]:
        path.mkdir(parents=True, exist_ok=True)

    write_embeddings(val_embeddings, embeddings_dir / "val_embeddings.npy")
    write_embeddings(test_embeddings, embeddings_dir / "test_embeddings.npy")
    val_df.to_csv(embeddings_dir / "val_metadata.csv", index=False)
    test_df.to_csv(embeddings_dir / "test_metadata.csv", index=False)
    if split_df is not None:
        split_df.to_csv(tables_dir / "split_assignments_v1.csv", index=False)
    if fit_df is not None:
        fit_df.to_csv(tables_dir / "fit_manifest_v1.csv", index=False)
    val_df.to_csv(tables_dir / "val_manifest_v1.csv", index=False)
    if extra_tables:
        for filename, frame in extra_tables.items():
            frame.to_csv(tables_dir / filename, index=False)

    print("[descriptor_baseline] running validation threshold sweep", flush=True)
    sweep_df, _sweep_pred_df = run_threshold_sweep(df=val_df, embeddings=val_embeddings, thresholds=thresholds)
    sweep_df.to_csv(tables_dir / "val_threshold_sweep_v1.csv", index=False)
    best_thresholds_df = pick_best_thresholds(sweep_df)
    best_thresholds_df.to_csv(tables_dir / "best_thresholds_v1.csv", index=False)

    threshold_by_dataset = {
        row["dataset"]: float(row["threshold"])
        for _, row in best_thresholds_df.iterrows()
    }
    fallback_threshold = float(np.median(list(threshold_by_dataset.values())))
    threshold_sources = [
        {"dataset": dataset, "threshold": threshold, "source": "validation_best"}
        for dataset, threshold in sorted(threshold_by_dataset.items())
    ]
    for dataset in TEST_ONLY_DATASETS:
        threshold_by_dataset[dataset] = fallback_threshold
        threshold_sources.append(
            {"dataset": dataset, "threshold": fallback_threshold, "source": "validation_median_fallback"}
        )
    threshold_sources_df = pd.DataFrame(threshold_sources).sort_values("dataset").reset_index(drop=True)
    threshold_sources_df.to_csv(tables_dir / "test_threshold_sources_v1.csv", index=False)

    print("[descriptor_baseline] clustering val and test with chosen thresholds", flush=True)
    val_pred_df = apply_thresholds_to_df(df=val_df, embeddings=val_embeddings, threshold_by_dataset=threshold_by_dataset)
    test_pred_df = apply_thresholds_to_df(df=test_df, embeddings=test_embeddings, threshold_by_dataset=threshold_by_dataset)
    val_pred_df.to_csv(tables_dir / "val_predictions_v1.csv", index=False)
    test_pred_df.to_csv(tables_dir / "test_predictions_v1.csv", index=False)

    neighbor_df = build_neighbor_table(df=val_df[["image_id", "dataset", "identity", PATH_COLUMN]], embeddings=val_embeddings, top_k=5)
    neighbor_df.to_csv(tables_dir / "val_neighbors_v1.csv", index=False)
    recall_rows: list[dict[str, object]] = []
    for dataset in sorted(val_df["dataset"].unique()):
        dataset_mask = val_df["dataset"] == dataset
        dataset_embeddings = val_embeddings[dataset_mask]
        dataset_labels = val_df.loc[dataset_mask, "identity"].to_numpy()
        recall_rows.append(
            {
                "dataset": dataset,
                "recall_at_1": recall_at_k(dataset_embeddings, dataset_labels, k=1),
                "recall_at_5": recall_at_k(dataset_embeddings, dataset_labels, k=5),
            }
        )
    recall_df = pd.DataFrame(recall_rows)
    recall_df.to_csv(tables_dir / "val_recall_v1.csv", index=False)

    submission_path = tables_dir / "submission_v1.csv"
    build_submission(
        test_pred_df=test_pred_df,
        sample_submission_path=repo_root / "sample_submission.csv",
        output_path=submission_path,
    )

    print("[descriptor_baseline] writing qualitative artifacts", flush=True)
    create_baseline_qualitative_outputs(
        val_pred_df=val_pred_df,
        neighbor_df=neighbor_df,
        repo_root=repo_root,
        qualitative_dir=qualitative_dir,
        seed=split_seed,
    )

    markdown_writer(
        reports_dir / "summary.md",
        sweep_df,
        best_thresholds_df,
        threshold_sources_df,
        fallback_threshold,
    )
    payload = dict(summary_payload)
    payload["fallback_threshold"] = fallback_threshold
    (reports_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[descriptor_baseline] done | summary={reports_dir / 'summary.md'}", flush=True)

    return {
        "summary_path": reports_dir / "summary.md",
        "submission_path": submission_path,
        "threshold_sweep_path": tables_dir / "val_threshold_sweep_v1.csv",
        "qualitative_dir": qualitative_dir,
    }


def run_descriptor_baseline(
    repo_root: Path,
    output_dir: Path,
    descriptor: str,
    device: str = "cuda:0",
    batch_size: int | None = None,
    num_workers: int = 4,
    val_identity_fraction: float = 0.1,
    thresholds: list[float] | None = None,
    split_seed: int = 42,
    train_manifest_path: Path | None = None,
    test_manifest_path: Path | None = None,
) -> dict[str, Path]:
    _require_torch()
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    print(f"[descriptor_baseline] loading manifests | descriptor={descriptor}", flush=True)
    train_df, test_df = load_manifests(
        repo_root=repo_root,
        train_manifest_path=train_manifest_path,
        test_manifest_path=test_manifest_path,
    )
    split_df = build_identity_holdout_split(
        train_df=train_df,
        val_identity_fraction=val_identity_fraction,
        seed=split_seed,
    )
    val_df = split_df[split_df["split_role_v1"] == "val"].copy().reset_index(drop=True)
    fit_df = split_df[split_df["split_role_v1"] == "fit"].copy().reset_index(drop=True)

    print(
        f"[descriptor_baseline] split ready | val_images={len(val_df)} | fit_images={len(fit_df)} | test_images={len(test_df)}",
        flush=True,
    )
    print(f"[descriptor_baseline] loading model | descriptor={descriptor} | device={device}", flush=True)
    model, spec = load_descriptor_model(descriptor=descriptor, device=device)
    if batch_size is None:
        batch_size = spec.default_batch_size
    print(
        f"[descriptor_baseline] model ready | model_id={spec.model_id} | input_size={spec.input_size} | batch_size={batch_size}",
        flush=True,
    )

    print("[descriptor_baseline] extracting val embeddings", flush=True)
    val_embeddings = extract_embeddings(
        df=val_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print("[descriptor_baseline] extracting test embeddings", flush=True)
    test_embeddings = extract_embeddings(
        df=test_df,
        repo_root=repo_root,
        model=model,
        spec=spec,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    report_config = {
        "device": device,
        "val_identity_fraction": val_identity_fraction,
        "thresholds": thresholds,
    }
    return run_cached_embedding_baseline(
        repo_root=repo_root,
        output_dir=output_dir,
        val_df=val_df,
        test_df=test_df,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        thresholds=thresholds,
        split_seed=split_seed,
        markdown_writer=lambda output_path, val_metrics_df, best_thresholds_df, threshold_sources_df, fallback_threshold: write_markdown_report(
            output_path=output_path,
            descriptor=descriptor,
            spec=spec,
            val_metrics_df=val_metrics_df,
            best_thresholds_df=best_thresholds_df,
            threshold_sources_df=threshold_sources_df,
            fallback_threshold=fallback_threshold,
            config=report_config,
        ),
        summary_payload={
            "descriptor": descriptor,
            "model_id": spec.model_id,
            "input_size": spec.input_size,
            "device": device,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "val_identity_fraction": val_identity_fraction,
            "thresholds": thresholds,
        },
        split_df=split_df,
        fit_df=fit_df,
    )


def run_descriptor_fusion_baseline(
    repo_root: Path,
    output_dir: Path,
    source_dirs: list[Path],
    component_names: list[str] | None = None,
    weights: list[float] | None = None,
    thresholds: list[float] | None = None,
    split_seed: int = 42,
    fusion_mode: str = "concat_l2",
) -> dict[str, Path]:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    if fusion_mode != "concat_l2":
        raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")
    if component_names is not None and len(component_names) != len(source_dirs):
        raise ValueError("component_names length must match source_dirs length")
    if weights is not None and len(weights) != len(source_dirs):
        raise ValueError("weights length must match source_dirs length")

    bundles: list[CachedEmbeddingBundle] = []
    for index, source_dir in enumerate(source_dirs):
        bundle = load_cached_embedding_bundle(
            source_dir=source_dir,
            name=component_names[index] if component_names else None,
            weight=weights[index] if weights else 1.0,
        )
        bundles.append(bundle)

    reference = bundles[0]
    for bundle in bundles[1:]:
        ensure_metadata_alignment(
            reference_df=reference.val_df,
            candidate_df=bundle.val_df,
            split_name="val",
            reference_name=reference.name,
            candidate_name=bundle.name,
        )
        ensure_metadata_alignment(
            reference_df=reference.test_df,
            candidate_df=bundle.test_df,
            split_name="test",
            reference_name=reference.name,
            candidate_name=bundle.name,
        )

    val_embeddings = fuse_embedding_blocks(
        [bundle.val_embeddings for bundle in bundles],
        weights=[bundle.weight for bundle in bundles],
    )
    test_embeddings = fuse_embedding_blocks(
        [bundle.test_embeddings for bundle in bundles],
        weights=[bundle.weight for bundle in bundles],
    )
    component_df = build_fusion_component_table(bundles)
    split_path = reference.source_dir / "tables" / "split_assignments_v1.csv"
    fit_path = reference.source_dir / "tables" / "fit_manifest_v1.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else None
    fit_df = pd.read_csv(fit_path) if fit_path.exists() else None
    source_summary_path = reference.source_dir / "reports" / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8")) if source_summary_path.exists() else {}
    val_identity_fraction = source_summary.get("val_identity_fraction")

    report_config = {
        "component_names": component_df["component"].tolist(),
        "weights": component_df["weight"].tolist(),
        "fusion_mode": fusion_mode,
        "val_identity_fraction": val_identity_fraction,
        "thresholds": thresholds,
    }
    return run_cached_embedding_baseline(
        repo_root=repo_root,
        output_dir=output_dir,
        val_df=reference.val_df.copy(),
        test_df=reference.test_df.copy(),
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        thresholds=thresholds,
        split_seed=split_seed,
        markdown_writer=lambda output_path, val_metrics_df, best_thresholds_df, threshold_sources_df, fallback_threshold: write_fusion_markdown_report(
            output_path=output_path,
            fusion_name="+".join(component_df["component"].tolist()),
            component_df=component_df,
            val_metrics_df=val_metrics_df,
            best_thresholds_df=best_thresholds_df,
            threshold_sources_df=threshold_sources_df,
            fallback_threshold=fallback_threshold,
            config=report_config,
        ),
        summary_payload={
            "fusion_name": "+".join(component_df["component"].tolist()),
            "source_dirs": [str(path.resolve()) for path in source_dirs],
            "component_names": component_df["component"].tolist(),
            "weights": component_df["weight"].tolist(),
            "fusion_mode": fusion_mode,
            "thresholds": thresholds,
            "val_identity_fraction": val_identity_fraction,
        },
        split_df=split_df,
        fit_df=fit_df,
        extra_tables={"fusion_components_v1.csv": component_df},
    )


def _require_torch() -> None:
    if torch is None or T is None:
        raise ModuleNotFoundError(
            "descriptor baselines require torch and torchvision. Run this pipeline in the 'wildfusion' conda environment."
        )


def _require_clustering_deps() -> None:
    if fcluster is None or linkage is None or squareform is None or adjusted_rand_score is None:
        raise ModuleNotFoundError(
            "descriptor baselines require scipy and scikit-learn. Run this pipeline in the 'wildfusion' conda environment."
        )
