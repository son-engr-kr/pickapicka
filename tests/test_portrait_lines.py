"""Checks for softening wrinkles and neck lines, and lifting dark circles.

    uv run python tests/test_portrait_lines.py

The face is built by hand, as in test_portrait: the 106 landmarks where the
model puts them, the alphas `analyze` would store over the face's crop, and
the cheek colour it would measure. Skin with pores, and on it a line on the
forehead, a round spot of the same depth on a cheek, a line on the neck and a
shadow under each eye. What has to hold is that a line is softened and a spot
of the same depth is not, that each slider keeps to its own region, that the
pores survive, and that a window gets the same pixels the whole render gives.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from pickapicka import portrait, reshape

N = 1000                 # a 400 px face: lines a few pixels wide
BOX = [0.3, 0.2, 0.4, 0.56]
CROP = [0.05, 0.05, 0.9, 0.9]
FW = BOX[2] * N
SKIN = (0.72, 0.55, 0.45)
LINE_SIGMA = 0.004 * FW  # a fine line, its profile's sigma
DEPTH = 0.08             # how much darker its middle is, as a fraction

FOREHEAD_LINE = (0.43, 0.57, 0.30)   # x from, x to, y
NECK_LINE = (0.45, 0.55, 0.86)
SPOT = (0.38, 0.62)


def _landmarks() -> np.ndarray:
    lm = np.full((106, 2), 0.5, np.float64)
    cx, cy, a, b = 0.5, 0.48, 0.19, 0.27
    for k in range(17):
        phi = math.pi / 2 * (k / 16)
        lm[reshape.CONTOUR_L[k]] = [cx - a * math.cos(phi), cy + b * math.sin(phi)]
        lm[reshape.CONTOUR_R[k]] = [cx + a * math.cos(phi), cy + b * math.sin(phi)]
    for ex, out, (outer, inner), lower, upper, pupils, brow in (
            (0.42, -1, (35, 39), (36, 33, 37), (41, 40, 42), (34, 38), portrait.BROW_L),
            (0.58, 1, (93, 89), (91, 87, 90), (96, 94, 95), (88, 92), portrait.BROW_R)):
        lm[outer], lm[inner] = [ex + 0.04 * out, 0.45], [ex - 0.04 * out, 0.45]
        for i, x in zip(lower, (0.02 * out, 0.0, -0.02 * out)):
            lm[i] = [ex + x, 0.458]
        for i, x in zip(upper, (0.02 * out, 0.0, -0.02 * out)):
            lm[i] = [ex + x, 0.442]
        lm[list(pupils)] = [ex, 0.45]
        for j, i in enumerate(brow):
            lm[i] = [ex - 0.045 + 0.09 * j / (len(brow) - 1), 0.39 - 0.006 * math.sin(math.pi * j / (len(brow) - 1))]
    nose = {72: (.5, .47), 73: (.5, .5), 74: (.5, .53), 86: (.5, .57), 80: (.5, .6),
            75: (.47, .46), 81: (.53, .46), 76: (.475, .55), 82: (.525, .55),
            77: (.47, .58), 83: (.53, .58), 78: (.48, .6), 84: (.52, .6), 79: (.49, .6), 85: (.51, .6)}
    for i, p in nose.items():
        lm[i] = p
    for j, i in enumerate(portrait.MOUTH):
        t = 2 * math.pi * j / len(portrait.MOUTH)
        lm[i] = [0.5 + 0.05 * math.cos(t), 0.67 + 0.02 * math.sin(t)]
    lm[52], lm[61] = [0.44, 0.67], [0.56, 0.67]
    return lm


def _over_crop(fill) -> np.ndarray:
    """An alpha over the crop at 200 px, drawn by `fill(alpha, to_px)`, where
    to_px maps frame fractions to the alpha's pixels."""
    a = np.zeros((200, 200), np.float32)
    to_px = lambda p: ((np.asarray(p) - CROP[:2]) / CROP[2:] * 200).round().astype(np.int32)
    fill(a, to_px)
    return cv2.GaussianBlur(a, (0, 0), 1.5)


def _face(rgb: np.ndarray) -> dict:
    lm = _landmarks()
    centre = ((BOX[0] + BOX[2] / 2 - CROP[0]) / CROP[2] * 200, (BOX[1] + BOX[3] / 2 - CROP[1]) / CROP[3] * 200)
    axes = (BOX[2] / CROP[2] * 100, BOX[3] / CROP[3] * 100)
    skin = _over_crop(lambda a, _: cv2.ellipse(a, (int(centre[0]), int(centre[1])),
                                               (int(axes[0]), int(axes[1])), 0, 0, 360, 1.0, -1))
    lines = _over_crop(lambda a, _: cv2.ellipse(a, (int(centre[0]), int(centre[1])),
                                                (int(axes[0] * 0.9), int(axes[1] * 0.9)), 0, 0, 360, 1.0, -1))
    neck = _over_crop(lambda a, px: cv2.rectangle(a, tuple(px((0.4, 0.78))), tuple(px((0.6, 0.94))), 1.0, -1))
    return {"box": BOX, "crop": CROP, "landmarks": lm.tolist(),
            "skin": skin, "skin_luma": 0.58, "ramps": {"teeth": None, "eyes": None},
            "lines": portrait._to_alpha8(lines), "neck": portrait._to_alpha8(neck), "neck_luma": 0.58,
            "circles": [portrait._cheek_ref(rgb, ref) for _, ref, _ in portrait._circle_polys(lm * N)]}


def _photo(line=True, neck=True, spot=True, circles=True) -> np.ndarray:
    rng = np.random.default_rng(5)
    img = np.empty((N, N, 3), np.float32)
    img[:] = SKIN
    img += rng.normal(0, 0.02, (N, N)).astype(np.float32)[..., None]     # pores
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    shade = np.ones((N, N), np.float32)
    for x0, x1, y, on in ((*FOREHEAD_LINE, line), (*NECK_LINE, neck)):
        if on:
            along = np.clip((xx - x0 * N) / (0.01 * N), 0, 1) * np.clip((x1 * N - xx) / (0.01 * N), 0, 1)
            shade -= DEPTH * along * np.exp(-((yy - y * N) ** 2) / (2 * LINE_SIGMA ** 2))
    if spot:
        shade -= DEPTH * np.exp(-((xx - SPOT[0] * N) ** 2 + (yy - SPOT[1] * N) ** 2) / (2 * LINE_SIGMA ** 2))
    if circles:
        for ex in (0.42, 0.58):
            m = np.zeros((N, N), np.float32)
            cv2.ellipse(m, (int(ex * N), int(0.478 * N)), (int(0.034 * N), int(0.012 * N)), 0, 0, 360, 1.0, -1)
            shade -= 0.15 * cv2.GaussianBlur(m, (0, 0), 0.004 * N)
    return np.clip(img * shade[..., None], 0, 1)


def _u8(img: np.ndarray) -> np.ndarray:
    return np.rint(img * 255).astype(np.uint8)


def _depth(img: np.ndarray, x0: float, x1: float, y: float) -> float:
    """How much darker the middle of a horizontal line is than the skin either
    side, on a copy smoothed past the pores."""
    lum = cv2.GaussianBlur(portrait._luma(img), (0, 0), 1.0)
    cols = slice(int((x0 + 0.02) * N), int((x1 - 0.02) * N))
    r = int(round(y * N))
    side = int(round(4 * LINE_SIGMA))
    return float(((lum[r - side, cols] + lum[r + side, cols]) / 2 - lum[r, cols]).mean())


def _spot_depth(img: np.ndarray) -> float:
    lum = cv2.GaussianBlur(portrait._luma(img), (0, 0), 1.0)
    x, y = int(SPOT[0] * N), int(SPOT[1] * N)
    side = int(round(4 * LINE_SIGMA))
    return float((lum[y, x - side] + lum[y, x + side] + lum[y - side, x] + lum[y + side, x]) / 4 - lum[y, x])


def _pores(img: np.ndarray, sl) -> float:
    lum = portrait._luma(img)
    return float((lum - cv2.GaussianBlur(lum, (0, 0), 1.0))[sl].std())


def test_normalize_carries_the_new_sliders() -> None:
    assert portrait.normalize({"wrinkles": 40}) == {"smooth": 0, "teeth": 0, "eyes": 0, "wrinkles": 40}
    assert portrait.normalize({"neck": 300})["neck"] == 100
    assert portrait.normalize({"dark_circles": -5}) is None


def test_a_line_is_softened_and_a_spot_of_its_depth_is_not() -> None:
    img = _photo()
    face = _face(_u8(img))
    out = portrait.apply_portrait(img, {"wrinkles": 100}, [face])
    line_kept = _depth(out, *FOREHEAD_LINE) / _depth(img, *FOREHEAD_LINE)
    spot_kept = _spot_depth(out) / _spot_depth(img)
    assert line_kept < 0.5, line_kept
    # A round spot is a blemish, not a line: it is left for the spot heal.
    # Not untouched, since the line measure reads the rim of a spot as a short
    # curved line (Frangi's own caveat), but it keeps most of its depth.
    assert spot_kept > 0.6 and spot_kept > 2 * line_kept, (line_kept, spot_kept)
    cheek = (slice(int(0.55 * N), int(0.6 * N)), slice(int(0.36 * N), int(0.4 * N)))
    assert _pores(out, cheek) > 0.9 * _pores(img, cheek)


def test_the_line_keeps_its_colour() -> None:
    """The band of every channel comes out, so the softened line goes back to
    the skin's colour rather than being brightened in its own."""
    img = _photo()
    out = portrait.apply_portrait(img, {"wrinkles": 100}, [_face(_u8(img))])
    x0, x1, y = FOREHEAD_LINE
    row = (int(y * N), slice(int((x0 + 0.02) * N), int((x1 - 0.02) * N)))
    lab = lambda im: cv2.cvtColor(cv2.GaussianBlur(im, (0, 0), 1.0), cv2.COLOR_RGB2LAB)[row].mean(axis=0)
    skin = cv2.cvtColor(np.float32([[SKIN]]), cv2.COLOR_RGB2LAB)[0, 0]
    chroma = lambda v: math.hypot(v[1], v[2])
    assert abs(chroma(lab(out)) - chroma(skin)) <= abs(chroma(lab(img)) - chroma(skin)) + 0.5


def test_each_slider_keeps_to_its_own_region() -> None:
    img = _photo()
    face = _face(_u8(img))
    neck_only = portrait.apply_portrait(img, {"neck": 100}, [face])
    assert _depth(neck_only, *NECK_LINE) < 0.5 * _depth(img, *NECK_LINE)
    assert abs(_depth(neck_only, *FOREHEAD_LINE) - _depth(img, *FOREHEAD_LINE)) < 1e-4
    face_only = portrait.apply_portrait(img, {"wrinkles": 100}, [face])
    assert abs(_depth(face_only, *NECK_LINE) - _depth(img, *NECK_LINE)) < 1e-4


def test_a_line_through_an_eye_or_a_lid_is_left_alone() -> None:
    """The eyes, the lids up to the brows and the nose are shapes, not lines."""
    img = _photo(line=False, neck=False, spot=False, circles=False)
    lid = (0.38, 0.46, 0.42)                 # across the left upper lid
    x0, x1, y = lid
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    along = ((xx > x0 * N) & (xx < x1 * N)).astype(np.float32)
    img = img * (1 - DEPTH * along * np.exp(-((yy - y * N) ** 2) / (2 * LINE_SIGMA ** 2)))[..., None]
    out = portrait.apply_portrait(img, {"wrinkles": 100}, [_face(_u8(img))])
    assert _depth(out, *lid) > 0.9 * _depth(img, *lid)


def test_dark_circles_are_lifted_to_the_cheek() -> None:
    img = _photo(line=False, neck=False, spot=False)
    face = _face(_u8(img))
    out = portrait.apply_portrait(img, {"dark_circles": 100}, [face])
    under = (slice(int(0.474 * N), int(0.482 * N)), slice(int(0.405 * N), int(0.435 * N)))
    skin_luma = float(portrait._luma(np.float32([[SKIN]]))[0, 0])
    before = float(portrait._luma(img)[under].mean())
    after = float(portrait._luma(out)[under].mean())
    assert skin_luma - before > 0.05
    assert abs(skin_luma - after) < 0.25 * (skin_luma - before), (skin_luma, before, after)
    assert _pores(out, under) > 0.85 * _pores(img, under)
    above = (slice(int(0.30 * N), int(0.36 * N)), slice(int(0.38 * N), int(0.46 * N)))
    assert np.array_equal(out[above], img[above])


def test_a_bright_under_eye_is_not_darkened() -> None:
    img = _photo(line=False, neck=False, spot=False, circles=False)
    yy = np.mgrid[0:N, 0:N][0]
    img = np.clip(img * np.where((yy > 0.46 * N) & (yy < 0.5 * N), 1.12, 1.0)[..., None], 0, 1).astype(np.float32)
    face = _face(_u8(_photo(line=False, neck=False, spot=False, circles=False)))
    out = portrait.apply_portrait(img, {"dark_circles": 100}, [face])
    under = (slice(int(0.474 * N), int(0.482 * N)), slice(int(0.405 * N), int(0.435 * N)))
    assert portrait._luma(out)[under].mean() >= portrait._luma(img)[under].mean() - 1e-3


def test_a_window_matches_the_whole_render() -> None:
    img = _photo()
    faces = [_face(_u8(img))]
    params = {"wrinkles": 90, "neck": 80, "dark_circles": 70, "smooth": 50}
    whole = portrait.apply_portrait(img, params, faces)
    pad = int(math.ceil(portrait.padding(params, faces, N)))
    for x0, y0, x1, y1 in ((450, 270, 560, 330), (390, 440, 470, 520), (450, 820, 560, 900)):
        px0, py0, px1, py1 = max(0, x0 - pad), max(0, y0 - pad), min(N, x1 + pad), min(N, y1 + pad)
        win = portrait.apply_portrait(img[py0:py1, px0:px1], params, faces,
                                      roi=(px0 / N, py0 / N, (px1 - px0) / N, (py1 - py0) / N))
        inner = win[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        assert np.abs(inner - whole[y0:y1, x0:x1]).max() < 1e-4, (x0, y0)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
