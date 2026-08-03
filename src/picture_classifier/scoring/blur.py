"""Blur score via variance of Laplacian. Higher = sharper."""
from __future__ import annotations

import cv2

WORK_EDGE = 1024   # long edge the metric is measured at
MIN_REGION = 24    # a crop smaller than this (after scaling) is not worth measuring


def _scale_for(h: int, w: int) -> float:
    longest = max(h, w)
    return WORK_EDGE / longest if longest > WORK_EDGE else 1.0


def blur_score(img_path: str) -> float:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"failed to read {img_path}"
    h, w = img.shape
    scale = _scale_for(h, w)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def region_blur_score(img_path: str, bbox_xywh: list[int]) -> float | None:
    """Sharpness measured inside one box — the subject — instead of the whole
    frame. That is the honest question for a shot with a deliberately blurred
    background: a bokeh'd or panned photo reads as soft frame-wide even when the
    car itself is tack sharp.

    The crop is downscaled by the *frame's* factor, not its own, so the number
    stays comparable with `blur_score`. Returns None when the region is too
    small to say anything.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"failed to read {img_path}"
    h, w = img.shape
    x, y, bw, bh = bbox_xywh
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, x0 + int(bw)), min(h, y0 + int(bh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = img[y0:y1, x0:x1]
    scale = _scale_for(h, w)
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if min(crop.shape[:2]) < MIN_REGION:
        return None
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())
