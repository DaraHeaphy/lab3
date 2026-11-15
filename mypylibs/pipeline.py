"""One full SfM run: frames -> features -> pairwise E -> incremental SfM + BA
-> dense fused cloud, for a given focal-length factor and (un)distortion."""
import time

import cv2
import numpy as np

from mypylibs import features as F
from mypylibs.dense import dense_cloud
from mypylibs.geometry import fit_sphere, orbit_metrics, similarity_frame
from mypylibs.masks import make_mask
from mypylibs.sfm import Reconstruction, pair_reprojection_error


def scaled_K(K, factor):
    Kf = K.copy()
    Kf[0, 0] *= factor
    Kf[1, 1] *= factor
    return Kf


def run(frames_raw, K, D, factor, undistort, cfg, log=print):
    t0 = time.time()
    Kf = scaled_K(K, factor)
    if undistort:
        # The (possibly wrong) Kf is used for undistortion too: the distortion
        # model is defined in normalised coordinates, so a wrong focal length
        # also mis-applies the radial correction.
        frames = [cv2.undistort(f, Kf, D, None, Kf) for f in frames_raw]
    else:
        frames = list(frames_raw)
    fcfg = cfg['features']
    feat_masks = [make_mask(fcfg.get('mask'), f) for f in frames]
    dense_masks = [make_mask(cfg['dense'].get('mask'), f) for f in frames]

    det, norm = F.make_detector(fcfg['detector'], fcfg['n_features'])
    kps, descs = F.detect(frames, feat_masks, det, fcfg.get('clahe', False))
    kf = F.select_keyframes(kps, descs, Kf, norm, fcfg['ratio'], fcfg['ransac_px'],
                            fcfg['keyframe_px'], fcfg['keyframe_min_inliers'])
    frames = [frames[i] for i in kf]
    kps, descs = [kps[i] for i in kf], [descs[i] for i in kf]
    dense_masks = [dense_masks[i] for i in kf]
    log(f'    {len(kf)} keyframes out of {len(feat_masks)} frames')

    # Pairwise essential matrices between consecutive frames.
    pair_matches, pair_R, pair_t, pair_err, n_inl = [], [], [], [], []
    for i in range(len(frames) - 1):
        m = F.match(descs[i], descs[i + 1], norm, fcfg['ratio'])
        p1, p2 = kps[i][m[:, 0]], kps[i + 1][m[:, 1]]
        E, R, t, inl, pose_inl = F.essential_inliers(p1, p2, Kf, fcfg['ransac_px'])
        if E is None:
            raise RuntimeError(f'essential matrix failed between frames {i} and {i + 1}')
        pair_matches.append(m[inl])
        pair_R.append(R)
        pair_t.append(t)
        n_inl.append(int(inl.sum()))
        pair_err.append(pair_reprojection_error(Kf, R, t, p1[pose_inl], p2[pose_inl])
                        if pose_inl.sum() >= 8 else np.nan)
    log(f'    pairwise E: median {np.median(n_inl):.0f} inliers, '
        f'two-view reprojection error {np.nanmean(pair_err):.3f} px')

    track_of, tracks = F.build_tracks(len(frames), kps, pair_matches)
    lengths = np.array([len(t) for t in tracks])
    recon = Reconstruction(Kf, kps, track_of, tracks, cfg['sfm'], log)
    recon.run(pair_R, pair_t)

    recon.transform(*similarity_frame(recon))
    xyz, rgb = dense_cloud(frames, dense_masks, recon, cfg['dense'], log)

    C = np.stack([recon.center(i) for i in np.nonzero(recon.registered)[0]])
    stats = dict(recon.stats)
    stats.update(orbit_metrics(C))
    stats.update({
        'pair_inliers_median': float(np.median(n_inl)),
        'pair_reproj': float(np.nanmean(pair_err)),
        'track_len_mean': float(lengths.mean()),
        'tracks_ge5': int((lengths >= 5).sum()),
        'dense_points': int(len(xyz)),
        'keyframes': len(kf),
    })
    if len(xyz) > 100:
        _, r, rms, frac = fit_sphere(xyz)
        stats.update({'sphere_rms_rel': rms, 'sphere_inliers': frac, 'sphere_radius': r})
    stats['seconds'] = time.time() - t0
    log(f'    orbit: swept {stats["swept_deg"]:.1f} deg, loop gap {stats["loop_gap"]:.3f} R; '
        f'sphere fit RMS {stats.get("sphere_rms_rel", np.nan) * 100:.2f}% of radius '
        f'({stats["seconds"]:.0f} s)')

    sparse = recon.X[recon.has_X]
    return xyz, rgb, C, sparse, stats
