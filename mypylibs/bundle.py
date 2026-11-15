"""Sparse bundle adjustment with scipy.optimize.least_squares.

Cameras are world->camera: x_cam = R(rvec) X + t, projected with a fixed K.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rotate(points, rvecs):
    """Rotate each point by its Rodrigues vector (vectorised)."""
    theta = np.linalg.norm(rvecs, axis=1)[:, None]
    with np.errstate(invalid='ignore', divide='ignore'):
        v = np.nan_to_num(rvecs / theta)
    dot = np.sum(points * v, axis=1)[:, None]
    c, s = np.cos(theta), np.sin(theta)
    return c * points + s * np.cross(v, points) + dot * (1 - c) * v


def project(points, rvecs, tvecs, K):
    pc = rotate(points, rvecs) + tvecs
    z = pc[:, 2:3]
    uv = pc[:, :2] / z
    return np.column_stack([K[0, 0] * uv[:, 0] + K[0, 2], K[1, 1] * uv[:, 1] + K[1, 2]]), z.ravel()


def reprojection_errors(rvecs, tvecs, X, cam_idx, pt_idx, uv, K):
    proj, _ = project(X[pt_idx], rvecs[cam_idx], tvecs[cam_idx], K)
    return np.linalg.norm(proj - uv, axis=1)


def bundle_adjust(rvecs, tvecs, X, cam_idx, pt_idx, uv, K, free_cams,
                  huber_px=1.5, max_nfev=60):
    """Refine the cameras in `free_cams` and all points in X.

    rvecs, tvecs: (C,3); X: (P,3); observations (cam_idx[k], pt_idx[k], uv[k]).
    Cameras not in free_cams are held fixed (they anchor gauge and scale).
    Returns refined copies (rvecs, tvecs, X).
    """
    free_cams = np.asarray(sorted(set(free_cams)), int)
    n_free, n_pts = len(free_cams), len(X)
    slot = np.full(len(rvecs), -1, int)
    slot[free_cams] = np.arange(n_free)
    cslot = slot[cam_idx]

    def unpack(x):
        r, t = rvecs.copy(), tvecs.copy()
        cam = x[:6 * n_free].reshape(-1, 6)
        r[free_cams], t[free_cams] = cam[:, :3], cam[:, 3:]
        return r, t, x[6 * n_free:].reshape(-1, 3)

    def residuals(x):
        r, t, P = unpack(x)
        proj, _ = project(P[pt_idx], r[cam_idx], t[cam_idx], K)
        return (proj - uv).ravel()

    m = 2 * len(uv)
    A = lil_matrix((m, 6 * n_free + 3 * n_pts), dtype=int)
    rows = np.arange(len(uv))
    has_cam = cslot >= 0
    for s in range(6):
        A[2 * rows[has_cam], 6 * cslot[has_cam] + s] = 1
        A[2 * rows[has_cam] + 1, 6 * cslot[has_cam] + s] = 1
    for s in range(3):
        A[2 * rows, 6 * n_free + 3 * pt_idx + s] = 1
        A[2 * rows + 1, 6 * n_free + 3 * pt_idx + s] = 1

    x0 = np.hstack([np.hstack([rvecs[free_cams], tvecs[free_cams]]).ravel(), X.ravel()])
    res = least_squares(residuals, x0, jac_sparsity=A, method='trf', x_scale='jac',
                        loss='huber', f_scale=huber_px, max_nfev=max_nfev, verbose=0)
    return unpack(res.x)
