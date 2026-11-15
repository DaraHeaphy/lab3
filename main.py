"""CS6524 Lab 3 - Structure from Motion.

For each input video (a: camera orbits the object, b: object spins in front of
the camera) runs the SfM pipeline six times: focal length x {0.8, 1.0, 1.2},
with and without undistortion. Writes out/[a,b]_ff[u,d].ply plus a metrics
table and figures.
"""
import csv
import os

import cv2
import numpy as np
import yaml
from os import path as opath

from mypylibs.geometry import write_ply
from mypylibs.pipeline import run
from mypylibs.plots import plot_overview, write_table
from mypylibs.video import read_video_frames

ROOT_IN = 'in'
ROOT_OUT = 'out'
FIG_DIR = opath.join('assets', 'latex')
VIDEOS = ['a', 'b']
FACTORS = [0.8, 1.0, 1.2]
UNDISTORT = [True, False]


def load_inputs():
    K = np.loadtxt(opath.join(ROOT_IN, 'K.csv'), delimiter=',')
    D = np.loadtxt(opath.join(ROOT_IN, 'D.csv'), delimiter=',').ravel()
    return K, D


def main():
    cv2.setRNGSeed(0)
    os.makedirs(ROOT_OUT, exist_ok=True)
    K, D = load_inputs()
    rows, results = [], {}
    for v in VIDEOS:
        with open(opath.join(ROOT_IN, f'{v}.yaml')) as f:
            cfg = yaml.safe_load(f)
        frames, _ = read_video_frames(opath.join(ROOT_IN, f'{v}.mp4'), step=cfg['frame_step'])
        print(f'video {v}: {len(frames)} frames (every {cfg["frame_step"]}), '
              f'{frames[0].shape[1]}x{frames[0].shape[0]}')
        for und in UNDISTORT:
            for fac in FACTORS:
                name = f'{v}_{round(fac * 10):02d}{"u" if und else "d"}'
                print(f'  {name}: f x {fac}, {"undistorted" if und else "distorted"}')
                cv2.setRNGSeed(0)
                xyz, rgb, C, sparse, stats = run(frames, K, D, fac, und, cfg)
                write_ply(opath.join(ROOT_OUT, f'{name}.ply'), xyz, rgb,
                          comment='SfM, Y up, unit = median camera-to-point distance')
                rows.append({'run': name, 'video': v, 'factor': fac, 'undistort': und, **stats})
                results[name] = (xyz, rgb, C, sparse)

    keys = list(rows[0].keys())
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(opath.join(ROOT_OUT, 'metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    plot_overview(results, rows, FIG_DIR)
    write_table(rows, FIG_DIR)


if __name__ == '__main__':
    main()
