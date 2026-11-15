import cv2

# Every video (lab2 calibration and these scenes) is resized so its long side
# is this many pixels, e.g. 1080x1920 -> 360x640. K is only valid at this
# resolution, so all videos must go through this same function.
PROCESS_LONG_SIDE = 640


def preprocess(frame, long_side=PROCESS_LONG_SIDE):
    h, w = frame.shape[:2]
    scale = long_side / max(h, w)
    if scale == 1:
        return frame
    size = (round(w * scale), round(h * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def read_video_frames(path, step=1, long_side=PROCESS_LONG_SIDE):
    """Read every `step`-th frame of a video, pre-processed.

    Returns (frames, frame_indices).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames, indices, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frames.append(preprocess(frame, long_side))
            indices.append(i)
        i += 1
    cap.release()
    if not frames:
        raise FileNotFoundError(path)
    return frames, indices
