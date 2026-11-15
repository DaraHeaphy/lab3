"""Dense reconstruction from the recovered camera poses.

For each frame i paired with frame j = i + gap:
  * dense optical flow i->j and j->i (DIS), keep pixels whose forward-backward
    flow returns within `fb_px` (rejects occlusions and hand motion);
  * triangulate every surviving pixel with the known projection matrices;
  * keep points in front of both cameras, with low reprojection error and a
    minimum triangulation angle.
All per-pair points are then fused in a voxel grid: each occupied voxel's
position/colour is the mean of the samples that fell in it, and voxels hit by
fewer than `min_views` distinct frame pairs are dropped (single-pair noise).
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from mypylibs.sfm import triangulate


def _flow(g1, g2):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(g1, g2, None)


def pair_points(img_i, img_j, mask_i, P_i, P_j, C_i, C_j, cfg):
    gi = cv2.cvtColor(img_i, cv2.COLOR_BGR2GRAY)
    gj = cv2.cvtColor(img_j, cv2.COLOR_BGR2GRAY)
    ff = _flow(gi, gj)
    fb = _flow(gj, gi)
    h, w = gi.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    qx, qy = xs + ff[..., 0], ys + ff[..., 1]
    back = cv2.remap(fb, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=1e3)
    err = np.hypot(qx + back[..., 0] - xs, qy + back[..., 1] - ys)
    keep = (err < cfg['fb_px']) & (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
    if mask_i is not None:
        keep &= mask_i > 0
    v, u = np.nonzero(keep)
    if len(u) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    x1 = np.stack([u, v], 1).astype(np.float64)
    x2 = np.stack([qx[v, u], qy[v, u]], 1).astype(np.float64)
    X = triangulate(P_i, P_j, x1, x2)
    Xh = np.hstack([X, np.ones((len(X), 1))])
    p1, p2 = Xh @ P_i.T, Xh @ P_j.T
    e1 = np.linalg.norm(p1[:, :2] / p1[:, 2:3] - x1, axis=1)
    e2 = np.linalg.norm(p2[:, :2] / p2[:, 2:3] - x2, axis=1)
    r1, r2 = X - C_i, X - C_j
    cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    ok = ((p1[:, 2] > 0) & (p2[:, 2] > 0) & (e1 < cfg['tri_px']) & (e2 < cfg['tri_px'])
          & (ang >= cfg['min_angle_deg']))
    rgb = cv2.cvtColor(img_i, cv2.COLOR_BGR2RGB)[v[ok], u[ok]]
    return X[ok], rgb


def voxel_fuse(chunks, voxel, min_views):
    """chunks: list of (xyz, rgb) per frame pair. Returns averaged (xyz, rgb, count)."""
    xyz = np.concatenate([c[0] for c in chunks])
    rgb = np.concatenate([c[1] for c in chunks]).astype(np.float64)
    pair_id = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(chunks)])
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    mean = np.stack([np.bincount(inv, xyz[:, d], n) for d in range(3)], 1) / cnt[:, None]
    col = np.stack([np.bincount(inv, rgb[:, d], n) for d in range(3)], 1) / cnt[:, None]
    # Number of distinct frame pairs contributing to each voxel.
    views = np.bincount(np.unique(np.stack([inv, pair_id], 1), axis=0)[:, 0], minlength=n)
    keep = views >= min_views
    return mean[keep], col[keep].astype(np.uint8), views[keep]


def statistical_outlier_removal(xyz, rgb, k=16, n_std=2.0):
    if len(xyz) <= k:
        return xyz, rgb
    d, _ = cKDTree(xyz).query(xyz, k=k + 1)
    md = d[:, 1:].mean(1)
    ok = md <= md.mean() + n_std * md.std()
    return xyz[ok], rgb[ok]


def dense_cloud(frames, masks, recon, cfg, log=print):
    n = len(frames)
    gap = cfg['gap']
    chunks = []
    for i in range(0, n - gap, cfg.get('every', 1)):
        j = i + gap
        if not (recon.registered[i] and recon.registered[j]):
            continue
        X, rgb = pair_points(frames[i], frames[j], masks[i], recon.P(i), recon.P(j),
                             recon.center(i), recon.center(j), cfg)
        if len(X):
            chunks.append((X, rgb))
    if not chunks:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8)
    raw = sum(len(c[0]) for c in chunks)
    xyz, rgb, _ = voxel_fuse(chunks, cfg['voxel'], cfg['min_views'])
    xyz, rgb = statistical_outlier_removal(xyz, rgb)
    log(f'    dense: {raw} triangulated pixels from {len(chunks)} pairs -> {len(xyz)} fused points')
    return xyz, rgb
