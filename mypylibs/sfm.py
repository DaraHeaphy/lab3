"""Incremental sequential Structure from Motion.

1. Pairwise: essential matrix + (R, t) for every pair of consecutive frames,
   plus the two-view reprojection error of the triangulated inliers.
2. Chaining: camera 0 is the world origin, camera 1 comes from E (unit
   baseline). Each following camera is registered with PnP-RANSAC against the
   points already triangulated (keeps a single, consistent scale), then the
   new tracks it closes are triangulated.
3. Drift control: every `ba_every` frames, a windowed bundle adjustment over the
   last `ba_window` (>=5) cameras; older cameras stay fixed and anchor the scale.
   A final global bundle adjustment refines everything.
"""
import cv2
import numpy as np

from mypylibs.bundle import bundle_adjust, reprojection_errors


def triangulate(P1s, P2s, x1, x2):
    """Linear (DLT) triangulation, batched; P1s/P2s are (n,3,4) or (3,4)."""
    P1s = np.broadcast_to(P1s, (len(x1), 3, 4))
    P2s = np.broadcast_to(P2s, (len(x1), 3, 4))
    A = np.stack([x1[:, :1] * P1s[:, 2] - P1s[:, 0],
                  x1[:, 1:] * P1s[:, 2] - P1s[:, 1],
                  x2[:, :1] * P2s[:, 2] - P2s[:, 0],
                  x2[:, 1:] * P2s[:, 2] - P2s[:, 1]], axis=1)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[:, -1]
    return Xh[:, :3] / Xh[:, 3:4]


def proj_err(P, X, x):
    Xh = np.hstack([X, np.ones((len(X), 1))])
    p = Xh @ P.T
    return np.linalg.norm(p[:, :2] / p[:, 2:3] - x, axis=1), p[:, 2]


def pair_reprojection_error(K, R, t, p1, p2):
    """Step 4: first camera at the origin, second at (R, t). Triangulate the
    matches and measure how well the 3D estimate reprojects into both frames."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t.reshape(3, 1)])
    X = triangulate(P1, P2, p1, p2)
    e1, z1 = proj_err(P1, X, p1)
    e2, z2 = proj_err(P2, X, p2)
    ok = (z1 > 0) & (z2 > 0)
    return float(np.mean((e1[ok] + e2[ok]) / 2)) if ok.any() else np.nan


class Reconstruction:
    def __init__(self, K, kps, track_of, tracks, cfg, log=print):
        self.K, self.kps, self.track_of, self.tracks = K, kps, track_of, tracks
        self.cfg, self.log = cfg, log
        n = len(kps)
        self.rvecs = np.zeros((n, 3))
        self.tvecs = np.zeros((n, 3))
        self.registered = np.zeros(n, bool)
        self.X = np.full((len(tracks), 3), np.nan)
        self.has_X = np.zeros(len(tracks), bool)
        self.stats = {}
        self.anchor = 0

    # ---- helpers -------------------------------------------------------
    def P(self, i):
        R, _ = cv2.Rodrigues(self.rvecs[i])
        return self.K @ np.hstack([R, self.tvecs[i].reshape(3, 1)])

    def center(self, i):
        R, _ = cv2.Rodrigues(self.rvecs[i])
        return -R.T @ self.tvecs[i]

    def observations(self, point_mask=None):
        cams, pts, uvs = [], [], []
        for tid, obs in enumerate(self.tracks):
            if not self.has_X[tid] or (point_mask is not None and not point_mask[tid]):
                continue
            for f, k in obs:
                if self.registered[f]:
                    cams.append(f)
                    pts.append(tid)
                    uvs.append(self.kps[f][k])
        return np.array(cams, int), np.array(pts, int), np.array(uvs, float).reshape(-1, 2)

    def mean_reprojection_error(self):
        c, p, uv = self.observations()
        return float(np.mean(reprojection_errors(self.rvecs, self.tvecs, self.X, c, p, uv, self.K)))

    # ---- triangulation -------------------------------------------------
    def triangulate_new(self, k):
        """Triangulate tracks seen in frame k that have no 3D point yet, using the
        earliest registered frame of the track (widest baseline)."""
        cfg = self.cfg
        tids = self.track_of[k]
        cand = np.nonzero((tids >= 0) & ~self.has_X[np.maximum(tids, 0)])[0]
        if len(cand) == 0:
            return 0
        firsts, x1, x2, tsel = [], [], [], []
        for kp in cand:
            tid = tids[kp]
            for f, j in self.tracks[tid]:
                if f < k and self.registered[f]:
                    firsts.append(f)
                    x1.append(self.kps[f][j])
                    x2.append(self.kps[k][kp])
                    tsel.append(tid)
                    break
        if not tsel:
            return 0
        firsts, x1, x2, tsel = np.array(firsts), np.array(x1), np.array(x2), np.array(tsel)
        P1s = np.stack([self.P(f) for f in firsts])
        P2 = self.P(k)
        X = triangulate(P1s, P2, x1, x2)
        e2, z2 = proj_err(P2, X, x2)
        Xh = np.hstack([X, np.ones((len(X), 1))])
        p1 = np.einsum('nij,nj->ni', P1s, Xh)
        z1 = p1[:, 2]
        e1 = np.linalg.norm(p1[:, :2] / p1[:, 2:3] - x1, axis=1)
        # Triangulation (parallax) angle between the two viewing rays.
        C1 = np.stack([self.center(f) for f in firsts])
        C2 = self.center(k)
        r1, r2 = X - C1, X - C2
        cosang = np.sum(r1 * r2, 1) / (np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        ok = ((z1 > 0) & (z2 > 0) & (e1 < cfg['tri_px']) & (e2 < cfg['tri_px'])
              & (ang >= cfg['min_angle_deg']))
        self.X[tsel[ok]] = X[ok]
        self.has_X[tsel[ok]] = True
        return int(ok.sum())

    # ---- registration ----------------------------------------------------
    def register(self, k, rel_R, rel_t):
        """PnP against existing 3D points; falls back to composing the pairwise
        essential-matrix pose (with the previous step length as scale)."""
        cfg = self.cfg
        Rp, _ = cv2.Rodrigues(self.rvecs[k - 1])
        step = (np.linalg.norm(self.center(k - 1) - self.center(k - 2))
                if k >= 2 and self.registered[k - 2] else 1.0)
        Rg = rel_R @ Rp
        tg = rel_R @ self.tvecs[k - 1] + step * rel_t
        rg, _ = cv2.Rodrigues(Rg)

        tids = self.track_of[k]
        sel = (tids >= 0) & self.has_X[np.maximum(tids, 0)]
        obj, img = self.X[tids[sel]], self.kps[k][sel]
        n_inl = 0
        if len(obj) >= 6:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, self.K, None, rg.copy(), tg.reshape(3, 1).copy(),
                useExtrinsicGuess=True, iterationsCount=300,
                reprojectionError=cfg['pnp_px'], confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= cfg['min_pnp_inliers']:
                inl = inl.ravel()
                rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
                self.rvecs[k], self.tvecs[k] = rvec.ravel(), tvec.ravel()
                self.registered[k] = True
                return len(inl), len(obj)
        self.rvecs[k], self.tvecs[k] = rg.ravel(), tg
        self.registered[k] = True
        self.log(f'    frame {k}: PnP failed ({len(obj)} 2D-3D), using E-composed pose')
        return n_inl, len(obj)

    # ---- bundle adjustment -----------------------------------------------
    def bundle(self, free_cams, point_mask=None, max_nfev=40):
        c, p, uv = self.observations(point_mask)
        if len(c) == 0:
            return
        used = np.unique(p)
        remap = np.full(len(self.X), -1, int)
        remap[used] = np.arange(len(used))
        free = [f for f in free_cams if f != self.anchor and self.registered[f]]
        r, t, Xu = bundle_adjust(self.rvecs, self.tvecs, self.X[used], c, remap[p], uv,
                                 self.K, free, huber_px=self.cfg['huber_px'], max_nfev=max_nfev)
        self.rvecs, self.tvecs = r, t
        self.X[used] = Xu

    def local_bundle(self, k):
        W = self.cfg['ba_window']
        window = [f for f in range(max(0, k - W + 1), k + 1) if self.registered[f]]
        seen = np.zeros(len(self.X), bool)
        for f in window:
            tids = self.track_of[f]
            tids = tids[tids >= 0]
            seen[tids[self.has_X[tids]]] = True
        self.bundle(window, seen, max_nfev=self.cfg['local_nfev'])

    def prune(self, max_px):
        c, p, uv = self.observations()
        e = reprojection_errors(self.rvecs, self.tvecs, self.X, c, p, uv, self.K)
        mean_e = np.bincount(p, e, minlength=len(self.X)) / np.maximum(np.bincount(p, minlength=len(self.X)), 1)
        bad = self.has_X & (mean_e > max_px)
        self.has_X[bad] = False
        self.X[bad] = np.nan
        return int(bad.sum())

    def transform(self, s, Rw, c):
        """Apply the similarity X' = s Rw (X - c) to points and cameras."""
        self.X = s * (self.X - c) @ Rw.T
        for i in range(len(self.rvecs)):
            R, _ = cv2.Rodrigues(self.rvecs[i])
            self.tvecs[i] = s * (R @ c + self.tvecs[i])
            self.rvecs[i] = cv2.Rodrigues(R @ Rw.T)[0].ravel()

    # ---- main loop -------------------------------------------------------
    def initialise(self):
        """Find the initial pair (s, m): the first start frame s and partner m
        whose essential matrix (through the tracks linking them) gives enough
        well-conditioned points. Leading frames with no parallax (object or
        camera not moving yet) are skipped and stay unregistered. Camera s is
        the world origin and |t_sm| = 1 fixes the (arbitrary) scale."""
        cfg, n = self.cfg, len(self.kps)
        for s in range(0, min(n - 1, cfg['init_max_start'] + 1)):
            ts = self.track_of[s]
            idx_s = np.nonzero(ts >= 0)[0]
            for m in range(s + 1, min(n, s + cfg['init_max_gap'] + 1)):
                pos_m = {tid: k for k, tid in enumerate(self.track_of[m]) if tid >= 0}
                pairs = [(k, pos_m[ts[k]]) for k in idx_s if ts[k] in pos_m]
                if len(pairs) < 50:
                    break
                pairs = np.array(pairs)
                ps, pm = self.kps[s][pairs[:, 0]], self.kps[m][pairs[:, 1]]
                E, inl = cv2.findEssentialMat(ps, pm, self.K, cv2.RANSAC, 0.999, 1.0)
                if E is None or E.shape != (3, 3):
                    continue
                _, R, t, _ = cv2.recoverPose(E, ps, pm, self.K, mask=inl)
                self.rvecs[s], self.tvecs[s] = 0, 0
                self.rvecs[m], self.tvecs[m] = cv2.Rodrigues(R)[0].ravel(), t.ravel()
                self.registered[:] = False
                self.registered[[s, m]] = True
                self.X[:], self.has_X[:] = np.nan, False
                n_pts = self.triangulate_new(m)
                if n_pts >= cfg['init_min_points']:
                    self.log(f'    init: frames {s}-{m}, {n_pts} points')
                    return s, m
        raise RuntimeError('could not initialise the reconstruction')

    def run(self, pair_R, pair_t):
        cfg, log = self.cfg, self.log
        n = len(self.kps)
        s, m = self.initialise()
        self.anchor = s
        # Frames between s and m: register by PnP.
        for k in range(s + 1, m):
            self.register(k, pair_R[k - 1], pair_t[k - 1])
            self.triangulate_new(k)
        pnp_inliers = []
        for k in range(m + 1, n):
            inl, tot = self.register(k, pair_R[k - 1], pair_t[k - 1])
            pnp_inliers.append(inl)
            self.triangulate_new(k)
            if cfg['ba_every'] and k % cfg['ba_every'] == 0:
                self.local_bundle(k)
        self.stats['pnp_inliers_median'] = float(np.median(pnp_inliers)) if pnp_inliers else 0
        self.stats['pnp_failures'] = int(sum(i == 0 for i in pnp_inliers))
        self.stats['registered'] = int(self.registered.sum())
        self.stats['reproj_before_global_ba'] = self.mean_reprojection_error()
        if cfg['global_ba']:
            self.bundle(range(n), max_nfev=cfg['global_nfev'])
            pruned = self.prune(cfg['prune_px'])
            self.bundle(range(n), max_nfev=cfg['global_nfev'])
            self.stats['pruned_points'] = pruned
        self.stats['reproj_final'] = self.mean_reprojection_error()
        self.stats['sparse_points'] = int(self.has_X.sum())
        log(f'    reprojection error: {self.stats["reproj_before_global_ba"]:.3f} px before '
            f'global BA -> {self.stats["reproj_final"]:.3f} px after, '
            f'{self.stats["sparse_points"]} sparse points, '
            f'{self.stats["pnp_failures"]} PnP failures')
