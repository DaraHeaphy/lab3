"""Normalisation, alignment and evaluation metrics for the reconstruction."""
import cv2
import numpy as np


def similarity_frame(recon):
    """Return (s, Rw, c): aligned = s * Rw @ (X - c).

    * scale: median camera-to-point distance becomes 1 (monocular SfM has no
      metric scale, this makes runs comparable);
    * rotation: the normal of the plane best fitting the camera centres (the
      orbit plane) becomes +Y, oriented with the cameras' "up";
    * origin: centroid of the sparse points.
    """
    reg = np.nonzero(recon.registered)[0]
    C = np.stack([recon.center(i) for i in reg])
    X = recon.X[recon.has_X]
    c = np.median(X, axis=0)
    d = np.linalg.norm(X[None, ::max(1, len(X) // 2000)] - C[:, None], axis=2)
    s = 1.0 / np.median(d)
    _, _, vt = np.linalg.svd(C - C.mean(0))
    normal = vt[2]
    # Image "up" is -y of the camera, expressed in world coordinates.
    ups = [-cv2.Rodrigues(recon.rvecs[i])[0].T[:, 1] for i in reg]
    if normal @ np.mean(ups, 0) < 0:
        normal = -normal
    x_axis = vt[0] - (vt[0] @ normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.cross(x_axis, normal)
    Rw = np.stack([x_axis, normal, z_axis])
    return s, Rw, c


def apply(frame, X):
    s, Rw, c = frame
    return s * (X - c) @ Rw.T


def fit_circle(xy):
    """Algebraic (Kasa) circle fit: returns centre, radius."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ctr = sol[:2]
    return ctr, float(np.sqrt(sol[2] + ctr @ ctr))


def orbit_metrics(C):
    """C: aligned camera centres (Y up). Swept angle around the orbit centre,
    loop-closure gap (start-end distance / orbit radius) and planarity."""
    xz = C[:, [0, 2]]
    ctr, r = fit_circle(xz)
    ang = np.unwrap(np.arctan2(xz[:, 1] - ctr[1], xz[:, 0] - ctr[0]))
    swept = float(np.degrees(abs(ang[-1] - ang[0])))
    gap = float(np.linalg.norm(C[-1] - C[0]) / r)
    radial = np.linalg.norm(xz - ctr, axis=1)
    return {'swept_deg': swept, 'loop_gap': gap, 'orbit_radius': r,
            'orbit_radial_cv': float(radial.std() / radial.mean()),
            'orbit_height_std': float(C[:, 1].std() / r)}


def fit_sphere(xyz, iters=3, inlier_frac=0.15):
    """Robust algebraic sphere fit. Returns centre, radius, relative RMS of the
    inliers (RMS/radius) and the inlier fraction."""
    pts = xyz
    keep = np.ones(len(pts), bool)
    for _ in range(iters):
        P = pts[keep]
        A = np.column_stack([2 * P, np.ones(len(P))])
        b = (P ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        ctr = sol[:3]
        r = np.sqrt(sol[3] + ctr @ ctr)
        res = np.linalg.norm(pts - ctr, axis=1) - r
        keep = np.abs(res) < inlier_frac * r
    rms = np.sqrt(np.mean(res[keep] ** 2)) / r
    return ctr, float(r), float(rms), float(keep.mean())


def write_ply(path, xyz, rgb, comment=''):
    """Binary little-endian PLY with float xyz and uchar rgb."""
    n = len(xyz)
    header = ('ply\nformat binary_little_endian 1.0\n'
              + (f'comment {comment}\n' if comment else '')
              + f'element vertex {n}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\n'
              'end_header\n')
    data = np.empty(n, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                              ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    data['x'], data['y'], data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data['r'], data['g'], data['b'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())
