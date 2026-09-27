"""Appearance descriptor for a detected subject crop — "does this look like the
same car?" rather than "is this the same car?".

Faces get ArcFace: a network trained so two photos of one person land next to
each other. No equivalent exists for consumer vehicle re-identification, so this
falls back to what is actually reliable from pixels: body colour and how that
colour is laid out over the shape.

What that means in practice:
  - Cars of clearly different colours separate well.
  - Two same-colour, same-shape cars will land in one group. That is a limit of
    the approach, not a bug to tune away.
  - Lighting changes (sun vs shade) move a car within colour space, so heavy
    exposure differences can split one car into two groups.

The descriptor is three L2-normalized blocks, weighted, then concatenated:
  1. hue-saturation histogram over the crop's centre (body colour)
  2. a coarse grid of mean Lab chroma (colour layout: dark roof, light body…)
  3. a coarse grid of mean lightness (tonal layout / rough shape shading)
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import imfile

CROP_SIZE = 96          # crops are normalized to this square before measuring
CENTRE_KEEP = 0.62      # fraction of the box used for the colour histogram
HS_BINS = (8, 8)        # hue x saturation bins
GRID = (4, 4)           # spatial grid for the layout blocks
BLOCK_WEIGHTS = (1.0, 0.7, 0.35)

DESCRIPTOR_LEN = HS_BINS[0] * HS_BINS[1] + 2 * GRID[0] * GRID[1] + GRID[0] * GRID[1]


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (float(np.linalg.norm(v)) + 1e-9)


def crop_of(img: np.ndarray, bbox_xywh: list[int]) -> np.ndarray | None:
    """Clamped BGR crop of one box, or None when it is too small to describe."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox_xywh
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, x0 + int(bw)), min(h, y0 + int(bh))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return img[y0:y1, x0:x1]


def describe(crop: np.ndarray) -> np.ndarray:
    """Appearance descriptor of one subject crop. Always DESCRIPTOR_LEN floats."""
    sq = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

    # 1. Body colour: hue x saturation over the centre, so the background
    #    showing around the box edges does not dominate the histogram. Weighting
    #    by saturation keeps grey asphalt from swamping a coloured body.
    m = int(CROP_SIZE * (1 - CENTRE_KEEP) / 2)
    centre = sq[m:CROP_SIZE - m, m:CROP_SIZE - m]
    hsv = cv2.cvtColor(centre, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)

    # 2 & 3. Layout: mean Lab over a coarse grid, chroma and lightness apart so
    #    each can be weighted on its own.
    lab = cv2.cvtColor(sq, cv2.COLOR_BGR2LAB).astype(np.float32)
    gh, gw = GRID
    cell = lab.reshape(gh, CROP_SIZE // gh, gw, CROP_SIZE // gw, 3).mean(axis=(1, 3))
    light = cell[:, :, 0].flatten() / 255.0
    chroma = (cell[:, :, 1:].reshape(-1) - 128.0) / 128.0

    w1, w2, w3 = BLOCK_WEIGHTS
    return np.concatenate([
        _l2(hist) * w1,
        _l2(chroma) * w2,
        _l2(light) * w3,
    ]).astype(np.float32)


def describe_box(img_path: str, bbox_xywh: list[int]) -> np.ndarray | None:
    img = imfile.imread(img_path)
    assert img is not None, f"failed to read {img_path}"
    crop = crop_of(img, bbox_xywh)
    return None if crop is None else describe(crop)
