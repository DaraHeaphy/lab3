"""Keypoint detection, consecutive-frame matching and track building."""
import cv2
import numpy as np


def make_detector(name, n_features):
    name = name.lower()
    if name == 'sift':
        # Low contrast threshold: the scene is dimly lit (V ~ 60-150).
        return cv2.SIFT_create(nfeatures=n_features, contrastThreshold=0.02), cv2.NORM_L2
    if name == 'orb':
        return cv2.ORB_create(nfeatures=n_features), cv2.NORM_HAMMING
    if name == 'akaze':
        return cv2.AKAZE_create(threshold=1e-4), cv2.NORM_HAMMING
    raise ValueError(f'unknown detector {name!r}')


def detect(frames, masks, detector, clahe=False):
    """Detect and describe keypoints; optional CLAHE boosts the dim globe texture."""
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if clahe else None
    kps, descs = [], []
    for f, m in zip(frames, masks):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if eq is not None:
            g = eq.apply(g)
        k, d = detector.detectAndCompute(g, m)
        kps.append(np.array([p.pt for p in k], np.float64).reshape(-1, 2))
        descs.append(d)
    return kps, descs


def match(d1, d2, norm, ratio):
    """Lowe ratio test + mutual nearest neighbour. Returns (M,2) index pairs."""
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(norm)
    fwd = bf.knnMatch(d1, d2, k=2)
    bwd = {m.queryIdx: m.trainIdx for m in bf.match(d2, d1)}
    out = [(m.queryIdx, m.trainIdx) for m, n in (p for p in fwd if len(p) == 2)
           if m.distance < ratio * n.distance and bwd.get(m.trainIdx) == m.queryIdx]
    return np.array(out, int).reshape(-1, 2)


def essential_inliers(p1, p2, K, thresh):
    """RANSAC essential matrix between two views.

    Returns (E, R, t, epipolar inliers, cheirality inliers). Tracks are built
    from the epipolar inliers: when the camera barely translates (start/end of
    a video) the decomposition of E is degenerate and cheirality rejects
    everything, yet the matches themselves are still correct.
    """
    empty = np.zeros(len(p1), bool)
    if len(p1) < 8:
        return None, None, None, empty, empty
    E, inl = cv2.findEssentialMat(p1, p2, K, cv2.RANSAC, 0.999, thresh)
    if E is None or E.shape != (3, 3):
        return None, None, None, empty, empty
    inl = inl.ravel().astype(bool)
    _, R, t, pose_inl = cv2.recoverPose(E, p1, p2, K, mask=inl.astype(np.uint8).copy())
    return E, R, t.ravel(), inl, pose_inl.ravel() > 0


def select_keyframes(kps, descs, K, norm, ratio, ransac_px, target_px, min_inliers):
    """Adaptive keyframes: a frame becomes the next keyframe once the median
    displacement of its epipolar-inlier matches w.r.t. the last keyframe
    reaches `target_px` (enough parallax), or just before the matches with the
    last keyframe would drop below `min_inliers` (tracks about to break).
    Handles pauses (hand repositioning in b, still camera at the start/end of
    a) and bursts of fast motion without a fixed frame step."""
    kf = [0]
    i = 1
    while i < len(kps):
        m = match(descs[kf[-1]], descs[i], norm, ratio)
        p1, p2 = kps[kf[-1]][m[:, 0]], kps[i][m[:, 1]]
        _, _, _, inl, _ = essential_inliers(p1, p2, K, ransac_px)
        disp = np.median(np.linalg.norm(p2[inl] - p1[inl], axis=1)) if inl.sum() else 0
        if inl.sum() < min_inliers and i - 1 > kf[-1]:
            kf.append(i - 1)          # re-test frame i against the new keyframe
            continue
        if disp >= target_px or inl.sum() < min_inliers:
            kf.append(i)
        i += 1
    return kf


def build_tracks(n_frames, kps, pair_matches):
    """Link consecutive-frame matches into tracks.

    track_of[i][k] is the track id of keypoint k in frame i (-1 if unmatched).
    Returns (track_of, tracks) with tracks[tid] = list of (frame, kp index).
    """
    track_of = [np.full(len(k), -1, int) for k in kps]
    tracks = []
    for i, m in enumerate(pair_matches):
        for a, b in m:
            tid = track_of[i][a]
            if tid < 0:
                tid = len(tracks)
                tracks.append([(i, a)])
                track_of[i][a] = tid
            if track_of[i + 1][b] < 0:
                track_of[i + 1][b] = tid
                tracks[tid].append((i + 1, b))
    return track_of, tracks
