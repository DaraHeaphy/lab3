"""One-off pre-processing of the raw phone videos into in/a.mp4 and in/b.mp4.

The lab2 calibration (K, D) was computed on frames resized so the long side is
640 px (1080x1920 -> 360x640, cv2.INTER_AREA). The scene videos must undergo
exactly the same resize, otherwise K does not apply. Frames are subsampled by
`step` only to keep the files under the 5 MB limit.

Usage (not run by main.py):
    python -m mypylibs.prepare_inputs RAW_A.mp4 RAW_B.mp4
"""
import shutil
import sys
from os import path as opath

import cv2

from mypylibs.video import preprocess

LAB2_OUT = opath.join('..', 'lab2', 'out')


def convert(src, dst, step=1):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer, i, n = None, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frame = preprocess(frame)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps / step, (w, h))
            writer.write(frame)
            n += 1
        i += 1
    cap.release()
    writer.release()
    print(f'{src} -> {dst}: {n} frames, {opath.getsize(dst) / 1e6:.2f} MB')


if __name__ == '__main__':
    raw_a, raw_b = sys.argv[1:3]
    convert(raw_a, opath.join('in', 'a.mp4'), step=2)
    convert(raw_b, opath.join('in', 'b.mp4'), step=2)
    shutil.copy(opath.join(LAB2_OUT, '01_K.csv'), opath.join('in', 'K.csv'))
    shutil.copy(opath.join(LAB2_OUT, '01_D.csv'), opath.join('in', 'D.csv'))
