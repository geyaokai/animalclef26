from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageColor, ImageDraw, ImageFont

from ..descriptor_baselines import PATH_COLUMN
from ..orb_rerank_baseline import resolve_existing_image_rel_path
from .hotspotter_spatial import calc_diaglen_sqrd, homography_inliers

try:  # pragma: no cover
    from scipy.spatial import cKDTree
except ModuleNotFoundError:  # pragma: no cover
    cKDTree = None


@dataclass(frozen=True)
class HotspotterConfig:
    """Minimal HotSpotter vsmany defaults, mirroring the old source tree."""

    k: int = 4
    knorm: int = 1
    normalizer_rule: str = "last"
    n_shortlist: int = 50
    xy_thresh: float = 0.01
    scale_thresh_low: float = 0.5
    scale_thresh_high: float = 2.0
    min_n_inliers: int = 4
    just_affine: bool = False
    use_chip_extent: bool = False
    rootsift: bool = True
    use_adaptive_scale: bool = False
    scale_min: float = 0.0
    scale_max: float = 9001.0
    max_side: int | None = None
    max_features_per_image: int | None = None
    feature_selection_strategy: str = "largest_scale"
    can_match_sameimg: bool = False


@dataclass(frozen=True)
class HotspotterFeature:
    image_id: str
    rel_path: str
    width: int
    height: int
    keypoints: np.ndarray
    descriptors: np.ndarray

    @property
    def point_count(self) -> int:
        return int(self.keypoints.shape[0])


@dataclass(frozen=True)
class HotspotterIndex:
    descriptors: np.ndarray
    ax2_cx: np.ndarray
    ax2_fx: np.ndarray
    tree: object | None


@dataclass(frozen=True)
class HotspotterCandidate:
    query_index: int
    candidate_index: int
    score: float
    prescore: float
    match_count: int
    inlier_count: int
    fm: np.ndarray
    fs: np.ndarray
    fk: np.ndarray


@dataclass(frozen=True)
class HotspotterPrescoreCandidate:
    query_index: int
    candidate_index: int
    prescore: float
    match_count: int


@dataclass(frozen=True)
class HotspotterQueryResult:
    query_index: int
    prescored_candidates: list[HotspotterPrescoreCandidate]
    candidates: list[HotspotterCandidate]


@dataclass(frozen=True)
class HotspotterFeatureExtractionReport:
    image_count: int
    cache_hit_count: int
    cache_miss_count: int
    cache_write_count: int
    cache_dir: Path | None


def _import_pyhesaff():
    try:
        import pyhesaff
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "HotSpotter local matching requires pyhesaff in the active environment."
        ) from exc
    return pyhesaff


def rootsift_descriptors(descriptors: np.ndarray) -> np.ndarray:
    if descriptors.size == 0:
        return np.zeros((0, 128), dtype=np.float32)
    desc = np.asarray(descriptors, dtype=np.float32)
    desc_sum = np.clip(desc.sum(axis=1, keepdims=True), 1e-12, None)
    desc = np.sqrt(desc / desc_sum)
    norms = np.clip(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12, None)
    return (desc / norms).astype(np.float32, copy=False)


def _load_hesaff_image(image_path: Path, max_side: int | None) -> tuple[np.ndarray, int, int]:
    with Image.open(image_path) as image:
        gray = image.convert("L")
        width, height = gray.size
        if max_side is not None and int(max_side) > 0 and max(width, height) > int(max_side):
            scale = float(max_side) / float(max(width, height))
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            gray = gray.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
            width, height = gray.size
        return np.asarray(gray, dtype=np.uint8), int(width), int(height)


def _detect_hesaff_features(
    *,
    image_path: Path,
    image_array: np.ndarray,
    pyhesaff,
    config: HotspotterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    detect_kwargs = {
        "scale_min": float(config.scale_min),
        "scale_max": float(config.scale_max),
    }
    keypoints, descriptors = pyhesaff.detect_feats_in_image(image_array, **detect_kwargs)
    if bool(config.use_adaptive_scale) and config.max_side is None:
        try:
            keypoints = pyhesaff.adapt_scale(str(image_path), keypoints)
        except Exception:
            # The modern pyhesaff binding does not expose the same adaptive-scale
            # surface as legacy HotSpotter on every platform. Falling back to the
            # detected keypoints is still deterministic and keeps the pipeline live.
            pass
    return np.asarray(keypoints, dtype=np.float32), np.asarray(descriptors, dtype=np.uint8)


def _trim_features(keypoints: np.ndarray, descriptors: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    return _trim_features_with_strategy(
        keypoints=keypoints,
        descriptors=descriptors,
        limit=limit,
        strategy="largest_scale",
    )


def _trim_features_with_strategy(
    *,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    limit: int | None,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_strategy = str(strategy).strip().lower()
    if normalized_strategy == "none" or limit is None or int(limit) <= 0 or len(keypoints) <= int(limit):
        return keypoints, descriptors
    if normalized_strategy == "detection_order":
        keep_index = np.arange(int(limit), dtype=np.int32)
    elif normalized_strategy == "largest_scale":
        scale_column = np.sqrt(np.abs(keypoints[:, 2] * keypoints[:, 4]))
        keep_index = np.argsort(scale_column)[::-1][: int(limit)]
        keep_index = np.sort(keep_index)
    else:
        raise ValueError(f"Unsupported feature_selection_strategy: {strategy}")
    return keypoints[keep_index], descriptors[keep_index]


def _cache_config_uid(config: HotspotterConfig) -> str:
    payload = {
        "rootsift": bool(config.rootsift),
        "use_adaptive_scale": bool(config.use_adaptive_scale),
        "scale_min": float(config.scale_min),
        "scale_max": float(config.scale_max),
        "max_side": None if config.max_side is None else int(config.max_side),
        "max_features_per_image": None if config.max_features_per_image is None else int(config.max_features_per_image),
        "feature_selection_strategy": str(config.feature_selection_strategy),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _feature_cache_paths(cache_dir: Path, rel_path: str, config: HotspotterConfig) -> tuple[Path, Path]:
    rel_key = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    config_key = _cache_config_uid(config)
    stem = f"{rel_key}_{config_key}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _legacy_feature_cache_paths(cache_dir: Path, rel_path: str, config: HotspotterConfig) -> tuple[Path, Path]:
    rel_key = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()
    payload = {
        "rootsift": bool(config.rootsift),
        "use_adaptive_scale": bool(config.use_adaptive_scale),
        "scale_min": float(config.scale_min),
        "scale_max": float(config.scale_max),
        "max_side": None if config.max_side is None else int(config.max_side),
        "max_features_per_image": None if config.max_features_per_image is None else int(config.max_features_per_image),
    }
    legacy_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = f"{rel_key}_{legacy_key}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def _cache_source_signature(image_path: Path) -> tuple[int, int]:
    stat = image_path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _load_cached_feature(
    *,
    cache_dir: Path,
    image_path: Path,
    rel_path: str,
    image_id: str,
    config: HotspotterConfig,
) -> HotspotterFeature | None:
    candidate_paths = [_feature_cache_paths(cache_dir=cache_dir, rel_path=rel_path, config=config)]
    if str(config.feature_selection_strategy).strip().lower() == "largest_scale":
        candidate_paths.append(_legacy_feature_cache_paths(cache_dir=cache_dir, rel_path=rel_path, config=config))
    for npz_path, meta_path in candidate_paths:
        if not npz_path.exists() or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source_mtime_ns, source_size = _cache_source_signature(image_path)
            if int(meta.get("source_mtime_ns", -1)) != source_mtime_ns or int(meta.get("source_size", -1)) != source_size:
                continue
            bundle = np.load(npz_path)
            return HotspotterFeature(
                image_id=image_id,
                rel_path=rel_path,
                width=int(meta["width"]),
                height=int(meta["height"]),
                keypoints=np.asarray(bundle["keypoints"], dtype=np.float32),
                descriptors=np.asarray(bundle["descriptors"], dtype=np.float32),
            )
        except Exception:
            continue
    return None


def _write_cached_feature(
    *,
    cache_dir: Path,
    image_path: Path,
    feature: HotspotterFeature,
    config: HotspotterConfig,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _feature_cache_paths(cache_dir=cache_dir, rel_path=feature.rel_path, config=config)
    source_mtime_ns, source_size = _cache_source_signature(image_path)
    np.savez_compressed(
        npz_path,
        keypoints=np.asarray(feature.keypoints, dtype=np.float32),
        descriptors=np.asarray(feature.descriptors, dtype=np.float32),
    )
    meta_path.write_text(
        json.dumps(
            {
                "image_id": feature.image_id,
                "rel_path": feature.rel_path,
                "width": int(feature.width),
                "height": int(feature.height),
                "point_count": int(feature.point_count),
                "source_mtime_ns": int(source_mtime_ns),
                "source_size": int(source_size),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def extract_hesaff_features_with_report(
    df: pd.DataFrame,
    repo_root: Path,
    *,
    config: HotspotterConfig | None = None,
    path_column: str = PATH_COLUMN,
    cache_dir: Path | None = None,
) -> tuple[list[HotspotterFeature], HotspotterFeatureExtractionReport]:
    resolved_config = HotspotterConfig() if config is None else config
    resolved_cache_dir = None if cache_dir is None else cache_dir.resolve()
    pyhesaff = None
    features: list[HotspotterFeature] = []
    cache_hit_count = 0
    cache_miss_count = 0
    cache_write_count = 0
    for row in df.itertuples(index=False):
        rel_path = resolve_existing_image_rel_path(row, repo_root=repo_root)
        if getattr(row, path_column, None):
            rel_path = str(getattr(row, path_column))
        image_id = str(getattr(row, "image_id"))
        image_path = repo_root / rel_path
        cached_feature = None
        if resolved_cache_dir is not None:
            cached_feature = _load_cached_feature(
                cache_dir=resolved_cache_dir,
                image_path=image_path,
                rel_path=rel_path,
                image_id=image_id,
                config=resolved_config,
            )
        if cached_feature is not None:
            features.append(cached_feature)
            cache_hit_count += 1
            continue
        if pyhesaff is None:
            pyhesaff = _import_pyhesaff()
        cache_miss_count += 1
        image_array, width, height = _load_hesaff_image(image_path=image_path, max_side=resolved_config.max_side)
        keypoints, descriptors = _detect_hesaff_features(
            image_path=image_path,
            image_array=image_array,
            pyhesaff=pyhesaff,
            config=resolved_config,
        )
        keypoints, descriptors = _trim_features_with_strategy(
            keypoints=keypoints,
            descriptors=descriptors,
            limit=resolved_config.max_features_per_image,
            strategy=resolved_config.feature_selection_strategy,
        )
        normalized_descriptors = (
            rootsift_descriptors(descriptors)
            if resolved_config.rootsift
            else np.asarray(descriptors, dtype=np.float32)
        )
        feature = HotspotterFeature(
            image_id=image_id,
            rel_path=str(rel_path),
            width=int(width),
            height=int(height),
            keypoints=keypoints,
            descriptors=normalized_descriptors,
        )
        if resolved_cache_dir is not None:
            _write_cached_feature(
                cache_dir=resolved_cache_dir,
                image_path=image_path,
                feature=feature,
                config=resolved_config,
            )
            cache_write_count += 1
        features.append(feature)
    return features, HotspotterFeatureExtractionReport(
        image_count=int(len(features)),
        cache_hit_count=int(cache_hit_count),
        cache_miss_count=int(cache_miss_count),
        cache_write_count=int(cache_write_count),
        cache_dir=resolved_cache_dir,
    )


def extract_hesaff_features(
    df: pd.DataFrame,
    repo_root: Path,
    *,
    config: HotspotterConfig | None = None,
    path_column: str = PATH_COLUMN,
    cache_dir: Path | None = None,
) -> list[HotspotterFeature]:
    features, _report = extract_hesaff_features_with_report(
        df=df,
        repo_root=repo_root,
        config=config,
        path_column=path_column,
        cache_dir=cache_dir,
    )
    return features


def build_hotspotter_index(features: list[HotspotterFeature]) -> HotspotterIndex:
    descriptor_blocks: list[np.ndarray] = []
    ax2_cx: list[int] = []
    ax2_fx: list[int] = []
    for image_index, feature in enumerate(features):
        if feature.descriptors.size == 0:
            continue
        descriptor_blocks.append(feature.descriptors.astype(np.float32, copy=False))
        ax2_cx.extend([image_index] * len(feature.descriptors))
        ax2_fx.extend(range(len(feature.descriptors)))
    if descriptor_blocks:
        descriptors = np.vstack(descriptor_blocks).astype(np.float32, copy=False)
        tree = cKDTree(descriptors) if cKDTree is not None else None
    else:
        descriptors = np.zeros((0, 128), dtype=np.float32)
        tree = None
    return HotspotterIndex(
        descriptors=descriptors,
        ax2_cx=np.asarray(ax2_cx, dtype=np.int32),
        ax2_fx=np.asarray(ax2_fx, dtype=np.int32),
        tree=tree,
    )


def _ensure_2d_indices(indices: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int32)
    dist = np.asarray(distances, dtype=np.float32)
    if idx.ndim == 1:
        idx = idx[:, None]
    if dist.ndim == 1:
        dist = dist[:, None]
    return idx, dist


def query_neighbors(index: HotspotterIndex, query_desc: np.ndarray, num_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    if query_desc.size == 0 or index.descriptors.size == 0 or num_neighbors <= 0:
        return (
            np.zeros((len(query_desc), 0), dtype=np.int32),
            np.zeros((len(query_desc), 0), dtype=np.float32),
        )
    k = min(int(num_neighbors), int(len(index.descriptors)))
    if index.tree is not None:
        distances, indices = index.tree.query(query_desc.astype(np.float32, copy=False), k=k)
        return _ensure_2d_indices(indices, distances)
    query = np.asarray(query_desc, dtype=np.float32)
    base = np.asarray(index.descriptors, dtype=np.float32)
    query_norm = np.sum(query**2, axis=1, keepdims=True)
    base_norm = np.sum(base**2, axis=1)[None, :]
    dist_sq = np.clip(query_norm + base_norm - (2.0 * query @ base.T), 0.0, None)
    topk = np.argpartition(dist_sq, kth=k - 1, axis=1)[:, :k]
    ordered = np.take_along_axis(topk, np.argsort(np.take_along_axis(dist_sq, topk, axis=1), axis=1), axis=1)
    distances = np.sqrt(np.take_along_axis(dist_sq, ordered, axis=1))
    return ordered.astype(np.int32, copy=False), distances.astype(np.float32, copy=False)


def _compute_feature_scores(
    *,
    qfx2_dx: np.ndarray,
    qfx2_dist: np.ndarray,
    query_index: int,
    index: HotspotterIndex,
    config: HotspotterConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if qfx2_dx.size == 0:
        return (
            np.zeros((len(qfx2_dx), 0), dtype=np.float32),
            np.zeros((len(qfx2_dx), 0), dtype=bool),
        )
    k_width = min(int(config.k), qfx2_dx.shape[1])
    if k_width <= 0:
        return (
            np.zeros((len(qfx2_dx), 0), dtype=np.float32),
            np.zeros((len(qfx2_dx), 0), dtype=bool),
        )
    qfx2_nn = qfx2_dx[:, :k_width]
    qfx2_nndist = qfx2_dist[:, :k_width]
    if config.normalizer_rule != "last":
        raise ValueError(f"Unsupported normalizer rule: {config.normalizer_rule}")
    norm_col = min(qfx2_dx.shape[1] - 1, int(config.k + config.knorm - 1))
    norm_dist = qfx2_dist[:, [norm_col]]
    score = norm_dist - qfx2_nndist
    valid = np.ones_like(score, dtype=bool)
    if not config.can_match_sameimg:
        valid &= index.ax2_cx[qfx2_nn] != int(query_index)
    return score.astype(np.float32, copy=False), valid


def _build_chipmatches_for_query(
    *,
    query_index: int,
    query_feature: HotspotterFeature,
    index: HotspotterIndex,
    config: HotspotterConfig,
) -> dict[int, dict[str, np.ndarray]]:
    qfx2_dx, qfx2_dist = query_neighbors(index=index, query_desc=query_feature.descriptors, num_neighbors=config.k + config.knorm)
    qfx2_score, qfx2_valid = _compute_feature_scores(
        qfx2_dx=qfx2_dx,
        qfx2_dist=qfx2_dist,
        query_index=query_index,
        index=index,
        config=config,
    )
    cx2_fm: dict[int, list[list[int]]] = {}
    cx2_fs: dict[int, list[float]] = {}
    cx2_fk: dict[int, list[int]] = {}
    for qfx in range(len(qfx2_dx)):
        for rank_index in range(qfx2_score.shape[1]):
            if not bool(qfx2_valid[qfx, rank_index]):
                continue
            dx = int(qfx2_dx[qfx, rank_index])
            cx = int(index.ax2_cx[dx])
            fx = int(index.ax2_fx[dx])
            cx2_fm.setdefault(cx, []).append([int(qfx), fx])
            cx2_fs.setdefault(cx, []).append(float(qfx2_score[qfx, rank_index]))
            cx2_fk.setdefault(cx, []).append(int(rank_index))
    chipmatches: dict[int, dict[str, np.ndarray]] = {}
    for cx, fm_list in cx2_fm.items():
        chipmatches[cx] = {
            "fm": np.asarray(fm_list, dtype=np.int32).reshape(-1, 2),
            "fs": np.asarray(cx2_fs[cx], dtype=np.float32),
            "fk": np.asarray(cx2_fk[cx], dtype=np.int32),
        }
    return chipmatches


def _candidate_diag_sqrd(
    *,
    query_feature: HotspotterFeature,
    candidate_feature: HotspotterFeature,
    fm: np.ndarray,
    use_chip_extent: bool,
) -> float:
    if use_chip_extent:
        return float(candidate_feature.width**2 + candidate_feature.height**2)
    if len(fm) == 0:
        return 1.0
    x_m = candidate_feature.keypoints[fm[:, 1], 0].astype(np.float64, copy=False)
    y_m = candidate_feature.keypoints[fm[:, 1], 1].astype(np.float64, copy=False)
    if len(x_m) == 0:
        return 1.0
    return float(calc_diaglen_sqrd(x_m, y_m))


def query_hotspotter_all(
    *,
    features: list[HotspotterFeature],
    config: HotspotterConfig | None = None,
) -> list[HotspotterQueryResult]:
    resolved_config = HotspotterConfig() if config is None else config
    index = build_hotspotter_index(features)
    results: list[HotspotterQueryResult] = []
    for query_index, query_feature in enumerate(features):
        chipmatches = _build_chipmatches_for_query(
            query_index=query_index,
            query_feature=query_feature,
            index=index,
            config=resolved_config,
        )
        prescored = sorted(
            (
                (
                    candidate_index,
                    float(match_data["fs"].sum()),
                    match_data,
                )
                for candidate_index, match_data in chipmatches.items()
                if candidate_index != query_index
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        prescored_candidates = [
            HotspotterPrescoreCandidate(
                query_index=query_index,
                candidate_index=int(candidate_index),
                prescore=float(prescore),
                match_count=int(len(match_data["fm"])),
            )
            for candidate_index, prescore, match_data in prescored
        ]
        shortlisted = prescored[: int(resolved_config.n_shortlist)]
        candidates: list[HotspotterCandidate] = []
        for candidate_index, prescore, match_data in shortlisted:
            fm = match_data["fm"]
            fs = match_data["fs"]
            fk = match_data["fk"]
            verified_fm = fm
            verified_fs = fs
            verified_fk = fk
            inlier_count = int(len(fm))
            if len(fm) >= int(resolved_config.min_n_inliers):
                diag_sqrd = _candidate_diag_sqrd(
                    query_feature=query_feature,
                    candidate_feature=features[candidate_index],
                    fm=fm,
                    use_chip_extent=bool(resolved_config.use_chip_extent),
                )
                sv_tup = homography_inliers(
                    kpts1=query_feature.keypoints,
                    kpts2=features[candidate_index].keypoints,
                    fm=fm,
                    xy_thresh=float(resolved_config.xy_thresh),
                    max_scale=float(resolved_config.scale_thresh_high),
                    min_scale=float(resolved_config.scale_thresh_low),
                    dlen_sqrd2=diag_sqrd,
                    min_num_inliers=int(resolved_config.min_n_inliers),
                    just_affine=bool(resolved_config.just_affine),
                )
                if sv_tup is None:
                    verified_fm = np.zeros((0, 2), dtype=np.int32)
                    verified_fs = np.zeros((0,), dtype=np.float32)
                    verified_fk = np.zeros((0,), dtype=np.int32)
                    inlier_count = 0
                else:
                    _, inliers = sv_tup
                    verified_fm = fm[inliers]
                    verified_fs = fs[inliers]
                    verified_fk = fk[inliers]
                    inlier_count = int(len(inliers))
            score = float(verified_fs.sum()) if len(verified_fs) else 0.0
            candidates.append(
                HotspotterCandidate(
                    query_index=query_index,
                    candidate_index=int(candidate_index),
                    score=score,
                    prescore=float(prescore),
                    match_count=int(len(fm)),
                    inlier_count=int(inlier_count),
                    fm=verified_fm,
                    fs=verified_fs,
                    fk=verified_fk,
                )
            )
        candidates.sort(key=lambda item: (item.score, item.inlier_count, item.prescore), reverse=True)
        results.append(
            HotspotterQueryResult(
                query_index=query_index,
                prescored_candidates=prescored_candidates,
                candidates=candidates,
            )
        )
    return results


def prescore_results_to_dataframe(
    *,
    features: list[HotspotterFeature],
    query_results: list[HotspotterQueryResult],
    top_k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_result in query_results:
        query_feature = features[query_result.query_index]
        for rank, candidate in enumerate(query_result.prescored_candidates[: int(top_k)], start=1):
            candidate_feature = features[candidate.candidate_index]
            rows.append(
                {
                    "image_id": query_feature.image_id,
                    "neighbor_image_id": candidate_feature.image_id,
                    "left_index": int(query_result.query_index),
                    "right_index": int(candidate.candidate_index),
                    "rank": int(rank),
                    "local_prescore": float(candidate.prescore),
                    "good_matches": int(candidate.match_count),
                    "matcher_name": "hesaff_rootsift_lnbnn_presv",
                    "left_keypoints": int(query_feature.point_count),
                    "right_keypoints": int(candidate_feature.point_count),
                    "left_path": query_feature.rel_path,
                    "right_path": candidate_feature.rel_path,
                }
            )
    return pd.DataFrame(rows)


def rank_results_to_dataframe(
    *,
    features: list[HotspotterFeature],
    query_results: list[HotspotterQueryResult],
    top_k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_result in query_results:
        query_feature = features[query_result.query_index]
        for rank, candidate in enumerate(query_result.candidates[: int(top_k)], start=1):
            candidate_feature = features[candidate.candidate_index]
            rows.append(
                {
                    "image_id": query_feature.image_id,
                    "neighbor_image_id": candidate_feature.image_id,
                    "left_index": int(query_result.query_index),
                    "right_index": int(candidate.candidate_index),
                    "rank": int(rank),
                    "local_score": float(candidate.score),
                    "local_prescore": float(candidate.prescore),
                    "good_matches": int(candidate.match_count),
                    "inliers": int(candidate.inlier_count),
                    "matcher_name": "hesaff_rootsift_lnbnn_sv",
                    "left_keypoints": int(query_feature.point_count),
                    "right_keypoints": int(candidate_feature.point_count),
                    "left_path": query_feature.rel_path,
                    "right_path": candidate_feature.rel_path,
                }
            )
    return pd.DataFrame(rows)


def unique_pair_results_to_dataframe(
    *,
    features: list[HotspotterFeature],
    query_results: list[HotspotterQueryResult],
    top_k: int = 10,
) -> pd.DataFrame:
    ranking_df = rank_results_to_dataframe(features=features, query_results=query_results, top_k=top_k)
    if ranking_df.empty:
        return ranking_df
    pair_df = ranking_df.copy()
    swap_mask = pair_df["left_index"].to_numpy(dtype=int) > pair_df["right_index"].to_numpy(dtype=int)
    if swap_mask.any():
        left_index = pair_df.loc[swap_mask, "left_index"].copy()
        pair_df.loc[swap_mask, "left_index"] = pair_df.loc[swap_mask, "right_index"].to_numpy(dtype=int)
        pair_df.loc[swap_mask, "right_index"] = left_index.to_numpy(dtype=int)
        left_id = pair_df.loc[swap_mask, "image_id"].copy()
        pair_df.loc[swap_mask, "image_id"] = pair_df.loc[swap_mask, "neighbor_image_id"].astype(str).to_numpy()
        pair_df.loc[swap_mask, "neighbor_image_id"] = left_id.astype(str).to_numpy()
        left_path = pair_df.loc[swap_mask, "left_path"].copy()
        pair_df.loc[swap_mask, "left_path"] = pair_df.loc[swap_mask, "right_path"].astype(str).to_numpy()
        pair_df.loc[swap_mask, "right_path"] = left_path.astype(str).to_numpy()
        left_kpts = pair_df.loc[swap_mask, "left_keypoints"].copy()
        pair_df.loc[swap_mask, "left_keypoints"] = pair_df.loc[swap_mask, "right_keypoints"].to_numpy(dtype=int)
        pair_df.loc[swap_mask, "right_keypoints"] = left_kpts.to_numpy(dtype=int)
    pair_df = pair_df.sort_values(
        ["local_score", "inliers", "good_matches", "rank"],
        ascending=[False, False, False, True],
    )
    pair_df = pair_df.drop_duplicates(subset=["left_index", "right_index"], keep="first").reset_index(drop=True)
    return pair_df


def summarize_feature_counts(features: Iterable[HotspotterFeature]) -> pd.DataFrame:
    feature_list = list(features)
    return pd.DataFrame(
        {
            "image_id": [feature.image_id for feature in feature_list],
            "path": [feature.rel_path for feature in feature_list],
            "keypoints": [feature.point_count for feature in feature_list],
            "width": [feature.width for feature in feature_list],
            "height": [feature.height for feature in feature_list],
        }
    )


def _fit_preview_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max(width, height) <= max_side:
        return int(width), int(height)
    scale = float(max_side) / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def render_match_preview(
    *,
    repo_root: Path,
    left_feature: HotspotterFeature,
    right_feature: HotspotterFeature,
    fm: np.ndarray,
    fs: np.ndarray | None = None,
    max_lines: int = 40,
    image_max_side: int = 280,
) -> Image.Image:
    left_image = Image.open(repo_root / left_feature.rel_path).convert("RGB").resize(
        (left_feature.width, left_feature.height),
        Image.Resampling.BILINEAR,
    )
    right_image = Image.open(repo_root / right_feature.rel_path).convert("RGB").resize(
        (right_feature.width, right_feature.height),
        Image.Resampling.BILINEAR,
    )
    left_size = _fit_preview_size(left_feature.width, left_feature.height, image_max_side)
    right_size = _fit_preview_size(right_feature.width, right_feature.height, image_max_side)
    left_preview = left_image.resize(left_size, Image.Resampling.BILINEAR)
    right_preview = right_image.resize(right_size, Image.Resampling.BILINEAR)
    margin = 12
    canvas = Image.new(
        "RGB",
        (left_preview.width + right_preview.width + margin * 3, max(left_preview.height, right_preview.height) + margin * 2),
        color=(250, 248, 244),
    )
    canvas.paste(left_preview, (margin, margin))
    right_offset_x = left_preview.width + margin * 2
    canvas.paste(right_preview, (right_offset_x, margin))
    draw = ImageDraw.Draw(canvas)

    if len(fm) == 0:
        return canvas

    if fs is not None and len(fs) == len(fm):
        order = np.argsort(np.asarray(fs, dtype=np.float32))[::-1]
        fm_to_draw = fm[order][: int(max_lines)]
    else:
        fm_to_draw = fm[: int(max_lines)]
    palette = [
        ImageColor.getrgb(color)
        for color in ["#C13B2A", "#0077B6", "#2A9D8F", "#6A4C93", "#F4A261", "#264653", "#D62828", "#588157"]
    ]
    left_scale_x = float(left_preview.width) / float(max(left_feature.width, 1))
    left_scale_y = float(left_preview.height) / float(max(left_feature.height, 1))
    right_scale_x = float(right_preview.width) / float(max(right_feature.width, 1))
    right_scale_y = float(right_preview.height) / float(max(right_feature.height, 1))
    for index, match in enumerate(fm_to_draw):
        color = palette[index % len(palette)]
        qfx = int(match[0])
        dfx = int(match[1])
        left_point = left_feature.keypoints[qfx, 0:2]
        right_point = right_feature.keypoints[dfx, 0:2]
        left_xy = (
            margin + float(left_point[0]) * left_scale_x,
            margin + float(left_point[1]) * left_scale_y,
        )
        right_xy = (
            right_offset_x + float(right_point[0]) * right_scale_x,
            margin + float(right_point[1]) * right_scale_y,
        )
        draw.line([left_xy, right_xy], fill=color, width=2)
        radius = 3
        draw.ellipse((left_xy[0] - radius, left_xy[1] - radius, left_xy[0] + radius, left_xy[1] + radius), outline=color, width=2)
        draw.ellipse((right_xy[0] - radius, right_xy[1] - radius, right_xy[0] + radius, right_xy[1] + radius), outline=color, width=2)
    return canvas


def create_match_board(
    *,
    repo_root: Path,
    features: list[HotspotterFeature],
    rows: list[dict[str, object]],
    output_path: Path,
    title: str,
) -> None:
    margin = 20
    gap = 18
    header_height = 48
    row_panels: list[Image.Image] = []
    title_font = ImageFont.load_default()
    body_font = ImageFont.load_default()
    for row in rows:
        preview = render_match_preview(
            repo_root=repo_root,
            left_feature=features[int(row["left_index"])],
            right_feature=features[int(row["right_index"])],
            fm=np.asarray(row["fm"], dtype=np.int32),
            fs=np.asarray(row.get("fs", []), dtype=np.float32) if row.get("fs") is not None else None,
        )
        text_height = 44
        panel = Image.new("RGB", (preview.width, preview.height + text_height), color=(255, 255, 255))
        panel.paste(preview, (0, 0))
        draw = ImageDraw.Draw(panel)
        text = (
            f"{row['image_id']} ({row['identity']}) -> {row['neighbor_image_id']} ({row['neighbor_identity']}) | "
            f"pre_rank={row.get('pre_rank', '')} post_rank={row.get('post_rank', '')} | "
            f"pre={float(row.get('local_prescore', 0.0)):.3f} post={float(row.get('local_score', 0.0)):.3f} | "
            f"inliers={int(row.get('inliers', 0))}"
        )
        draw.text((8, preview.height + 8), text, fill=(40, 40, 40), font=body_font)
        row_panels.append(panel)
    if not row_panels:
        empty = Image.new("RGB", (800, 120), color=(255, 255, 255))
        draw = ImageDraw.Draw(empty)
        draw.text((margin, margin), f"{title}: empty", fill=(40, 40, 40), font=title_font)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.save(output_path)
        return
    board_width = max(panel.width for panel in row_panels) + margin * 2
    board_height = header_height + margin + sum(panel.height for panel in row_panels) + gap * (len(row_panels) - 1) + margin
    board = Image.new("RGB", (board_width, board_height), color=(246, 244, 239))
    draw = ImageDraw.Draw(board)
    draw.text((margin, margin), title, fill=(20, 20, 20), font=title_font)
    current_y = header_height
    for panel in row_panels:
        board.paste(panel, (margin, current_y))
        current_y += panel.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path)
