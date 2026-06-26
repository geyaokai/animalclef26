from __future__ import annotations

import numpy as np
import numpy.linalg as linalg


SV_DTYPE = np.float64


def compute_homog(x1_mn: np.ndarray, y1_mn: np.ndarray, x2_mn: np.ndarray, y2_mn: np.ndarray) -> np.ndarray:
    """Port of HotSpotter's normalized DLT homography solver."""
    num_pts = len(x1_mn)
    matrix = np.zeros((2 * num_pts, 9), dtype=SV_DTYPE)
    for index in range(num_pts):
        u2 = x2_mn[index]
        v2 = y2_mn[index]
        x1 = x1_mn[index]
        y1 = y1_mn[index]
        matrix[index * 2] = (0.0, 0.0, 0.0, -x1, -y1, -1.0, v2 * x1, v2 * y1, v2)
        matrix[index * 2 + 1] = (x1, y1, 1.0, 0.0, 0.0, 0.0, -u2 * x1, -u2 * y1, -u2)
    _, _, vh = linalg.svd(matrix, full_matrices=False)
    hvec = vh[-1]
    return np.vstack((hvec[0:3], hvec[3:6], hvec[6:9]))


def calc_diaglen_sqrd(x_m: np.ndarray, y_m: np.ndarray) -> float:
    x_extent_sqrd = float((x_m.max() - x_m.min()) ** 2)
    y_extent_sqrd = float((y_m.max() - y_m.min()) ** 2)
    return x_extent_sqrd + y_extent_sqrd


def split_kpts(kpts_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.asarray(kpts_t[0], dtype=SV_DTYPE)
    ys = np.asarray(kpts_t[1], dtype=SV_DTYPE)
    acd = np.asarray(kpts_t[2:5], dtype=SV_DTYPE)
    return xs, ys, acd


def normalize_xy_points(x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean_x = float(x_m.mean())
    mean_y = float(y_m.mean())
    std_x = float(x_m.std())
    std_y = float(y_m.std())
    sx = 1.0 / std_x if std_x > 0 else 1.0
    sy = 1.0 / std_y if std_y > 0 else 1.0
    transform = np.array(
        [
            (sx, 0.0, -mean_x * sx),
            (0.0, sy, -mean_y * sy),
            (0.0, 0.0, 1.0),
        ],
        dtype=SV_DTYPE,
    )
    return (x_m - mean_x) * sx, (y_m - mean_y) * sy, transform


def det_acd(acd: np.ndarray) -> np.ndarray:
    return acd[0] * acd[2]


def inv_acd(acd: np.ndarray, det: np.ndarray) -> np.ndarray:
    return np.array((acd[2], -acd[1], acd[0]), dtype=SV_DTYPE) / det


def dot_acd(acd1: np.ndarray, acd2: np.ndarray) -> np.ndarray:
    a = acd1[0] * acd2[0]
    c = (acd1[1] * acd2[0]) + (acd1[2] * acd2[1])
    d = acd1[2] * acd2[2]
    return np.array((a, c, d), dtype=SV_DTYPE)


def affine_inliers(
    x1_m: np.ndarray,
    y1_m: np.ndarray,
    acd1_m: np.ndarray,
    fx1_m: np.ndarray,
    x2_m: np.ndarray,
    y2_m: np.ndarray,
    acd2_m: np.ndarray,
    xy_thresh_sqrd: float,
    max_scale: float,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    best_inliers: np.ndarray = np.array([], dtype=np.int32)
    best_match_index: int | None = None
    best_count = 0
    det1_m = det_acd(acd1_m)
    det2_m = det_acd(acd2_m)
    inv1_m = inv_acd(acd1_m, det1_m)
    aff_list = dot_acd(acd2_m, inv1_m)
    det_aff_list = det_acd(aff_list)
    for match_index in range(len(x1_m)):
        aa, ac, ad = aff_list[:, match_index]
        adet = det_aff_list[match_index]
        x1_hypo = x1_m[match_index]
        y1_hypo = y1_m[match_index]
        x2_hypo = x2_m[match_index]
        y2_hypo = y2_m[match_index]
        x1_transformed = x2_hypo + aa * (x1_m - x1_hypo)
        y1_transformed = y2_hypo + ac * (x1_m - x1_hypo) + ad * (y1_m - y1_hypo)
        xy_err = (x1_transformed - x2_m) ** 2 + (y1_transformed - y2_m) ** 2
        scale_err = adet * det1_m / det2_m
        xy_inliers = xy_err < xy_thresh_sqrd
        scale_inliers = np.logical_and(scale_err > min_scale, scale_err < max_scale)
        hypothesis_inliers = np.where(np.logical_and(xy_inliers, scale_inliers))[0]
        unique_query_fx = np.unique(fx1_m[hypothesis_inliers])
        inlier_count = int(len(unique_query_fx))
        if inlier_count > best_count:
            best_count = inlier_count
            best_match_index = match_index
            best_inliers = hypothesis_inliers.astype(np.int32, copy=False)
    if best_match_index is None:
        return np.eye(3, dtype=SV_DTYPE), best_inliers
    aa, ac, ad = aff_list[:, best_match_index]
    x1 = x1_m[best_match_index]
    y1 = y1_m[best_match_index]
    x2 = x2_m[best_match_index]
    y2 = y2_m[best_match_index]
    xt = x2 - aa * x1
    yt = y2 - ac * x1 - ad * y1
    best_aff = np.array(
        [
            (aa, 0.0, xt),
            (ac, ad, yt),
            (0.0, 0.0, 1.0),
        ],
        dtype=SV_DTYPE,
    )
    return best_aff, best_inliers


def homography_inliers(
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    fm: np.ndarray,
    xy_thresh: float,
    max_scale: float,
    min_scale: float,
    dlen_sqrd2: float | None = None,
    min_num_inliers: int = 4,
    just_affine: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(fm) == 0:
        return None
    fx1_m = fm[:, 0].astype(np.int32, copy=False)
    fx2_m = fm[:, 1].astype(np.int32, copy=False)
    x1_m, y1_m, acd1_m = split_kpts(kpts1[fx1_m, :].T)
    x2_m, y2_m, acd2_m = split_kpts(kpts2[fx2_m, :].T)
    diaglen_sqrd = calc_diaglen_sqrd(x2_m, y2_m) if dlen_sqrd2 is None else float(dlen_sqrd2)
    xy_thresh_sqrd = diaglen_sqrd * float(xy_thresh)
    aff, aff_inliers = affine_inliers(
        x1_m=x1_m,
        y1_m=y1_m,
        acd1_m=acd1_m,
        fx1_m=fx1_m,
        x2_m=x2_m,
        y2_m=y2_m,
        acd2_m=acd2_m,
        xy_thresh_sqrd=xy_thresh_sqrd,
        max_scale=float(max_scale),
        min_scale=float(min_scale),
    )
    if just_affine:
        return aff, aff_inliers
    if len(aff_inliers) < int(min_num_inliers):
        return None
    x1_aff = x1_m[aff_inliers]
    y1_aff = y1_m[aff_inliers]
    x2_aff = x2_m[aff_inliers]
    y2_aff = y2_m[aff_inliers]
    x1_norm, y1_norm, t1 = normalize_xy_points(x1_aff, y1_aff)
    x2_norm, y2_norm, t2 = normalize_xy_points(x2_aff, y2_aff)
    h_prime = compute_homog(x1_norm, y1_norm, x2_norm, y2_norm)
    try:
        homography = linalg.solve(t2, h_prime).dot(t1)
    except linalg.LinAlgError:
        return None
    ((h11, h12, h13), (h21, h22, h23), (h31, h32, h33)) = homography
    x1_transformed = h11 * x1_m + h12 * y1_m + h13
    y1_transformed = h21 * x1_m + h22 * y1_m + h23
    z1_transformed = h31 * x1_m + h32 * y1_m + h33
    z1_transformed[z1_transformed == 0] = 1e-14
    xy_err = ((x1_transformed / z1_transformed) - x2_m) ** 2 + ((y1_transformed / z1_transformed) - y2_m) ** 2
    inliers = np.where(xy_err < xy_thresh_sqrd)[0].astype(np.int32, copy=False)
    return homography, inliers
