import cv2
import numpy as np

# HSV hue ranges measured on the videos (OpenCV hue is 0..179):
#   floor (wood) 12-17, rug 13-17 (low S), skin 7-11,
#   globe land 18-58, globe ocean 75-120.
# So "hue >= 19 and saturated" isolates the globe from everything else.
GLOBE_HUE = (19, 125)
MIN_SAT = 40
MIN_VAL = 30


def globe_mask(frame):
    """Binary mask (uint8 0/255) of the globe: the largest saturated blob whose
    hue is land/ocean coloured, with its holes filled (labels, glare)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    m = ((h >= GLOBE_HUE[0]) & (h <= GLOBE_HUE[1]) & (s > MIN_SAT) & (v > MIN_VAL))
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros_like(m)
    big = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (lab == big).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(comp)
    cv2.drawContours(out, contours, -1, 255, -1)
    # Erode a little so features/pixels on the silhouette (where the background
    # shows through) are not used.
    return cv2.erode(out, np.ones((7, 7), np.uint8))


def _largest(m):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return np.zeros_like(m)
    return (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def cup_mask(frame):
    """Pink cup + toy lion on top, without the hand.

    Measured hues (OpenCV 0..179): cup pink 170-179/0-3 (wraps around red),
    lion body 18-30 (bright), skin 5-8 (bright, V > 200), rug 6-13 (dark,
    V < 135), floor 15-18. The lion's yellow is close to the floor, so lion
    blobs are only accepted where they touch the cup.
    """
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    h, s, v = [c.astype(int) for c in cv2.split(hsv)]
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r, r))
    pink = (((h >= 165) | (h <= 3)) & (s > 60) & (v > 50)).astype(np.uint8)
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, k(5))
    cup = _largest(cv2.morphologyEx(pink, cv2.MORPH_CLOSE, k(15)))
    if cup.sum() == 0:
        return cup
    lion = ((h >= 3) & (h <= 32) & (s > 80) & (v > 140)).astype(np.uint8)  # rug is darker
    near = cv2.dilate(cup, k(25))
    n, lab, _, _ = cv2.connectedComponentsWithStats(lion)
    touching = np.unique(lab[(near > 0) & (lion > 0)])
    lion = np.isin(lab, touching[touching > 0]).astype(np.uint8)
    obj = cv2.morphologyEx(cup | lion, cv2.MORPH_CLOSE, k(21))
    obj = _largest(obj)
    contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(obj)
    cv2.drawContours(out, contours, -1, 1, -1)
    # Hand and sleeve: only large blobs, so the red flower prints and lion
    # highlights (similar hue) are not cut out.
    skin = (((h >= 4) & (h <= 10) & (s > 90) & (v > 180)) | (v < 35)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(skin)
    big = np.nonzero(st[:, cv2.CC_STAT_AREA] > 600)[0]
    skin = np.isin(lab, big[big > 0]).astype(np.uint8)
    out[cv2.dilate(skin, k(11)) > 0] = 0
    return cv2.erode(out * 255, k(7))


def make_mask(kind, frame):
    if kind in (None, 'none'):
        return None
    if kind == 'globe':
        return globe_mask(frame)
    if kind == 'cup':
        return cup_mask(frame)
    if kind == 'cup_near':
        # Cup plus the rug around it: near, textured, rigid. Excludes the far
        # floor/wall, whose low-parallax points let PnP explain the orbit as a
        # pure rotation (camera translation collapsed to ~0 in testing).
        m = cup_mask(frame)
        return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (161, 161)))
    raise ValueError(f'unknown mask kind {kind!r}')
