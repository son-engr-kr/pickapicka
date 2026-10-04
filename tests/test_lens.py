"""Checks for the lens-correction stage. Runnable with pytest or directly:

    uv run python tests/test_lens.py

Two properties carry the module and get most of the attention here:

  - a neutral correction must return the very same array. Every other stage in
    the pipeline keeps that promise, and this one is the only stage that would
    otherwise resample a frame for nothing.
  - the same parameters must produce the same *look* at any render size. The
    editor grades against a preview a few hundred pixels wide and ships a frame
    twenty times that; a coefficient that quietly meant "pixels" would make the
    two disagree, and the disagreement would only ever be noticed on the export.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pickapicka import lens


# ----- synthetic scenes ---------------------------------------------------
#
# All of these are defined as functions of the *normalized* frame coordinate, so
# the same call at two sizes samples one continuous image at two rates rather
# than producing two different pictures. That is what makes a cross-resolution
# comparison mean anything.

RIDGE_Y = 0.35    # where the test ridge sits, in half-diagonals
RIDGE_SIGMA = 0.05


def _norm_axes(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-centre offsets from the optical centre, in half-diagonals."""
    cx, cy, r = lens._centre(w, h)
    px = ((np.arange(w, dtype=np.float64) - cx) / r)[None, :]
    py = ((np.arange(h, dtype=np.float64) - cy) / r)[:, None]
    return px, py


def _grey(img: np.ndarray) -> np.ndarray:
    return np.repeat(img[..., None].astype(np.float32), 3, axis=2).copy()


def _ridge(w: int, h: int, params: dict | None = None) -> np.ndarray:
    """A straight horizontal ridge at Y = RIDGE_Y, or that ridge seen through
    the exact distortion `params` claims to correct.

    The distorted version is built by inverting the sample grid analytically —
    no resampling — so correcting it should give the straight ridge back and any
    residual bow is the module's error, not the test's.
    """
    px, py = _norm_axes(w, h)
    ys = py + 0.0 * px
    if params is not None:
        k, s = lens._distortion_coeff(params)
        v = np.hypot(px, py)
        u = v.copy()
        for _ in range(80):
            u -= (s * u * (1.0 + k * s * s * u * u) - v) / (s * (1.0 + 3.0 * k * s * s * u * u))
        # The correction magnifies the periphery, so a strong one has no inverse
        # out at the corners: the straight frame it came from did not reach that
        # far. Keep the test distortions inside the invertible range.
        assert np.abs(s * u * (1.0 + k * s * s * u * u) - v).max() < 1e-9, \
            "this distortion is too strong to build an exact inverse for"
        ys = ys * np.where(v > 0.0, u / np.maximum(v, 1e-30), 1.0)
    return _grey(np.exp(-((ys - RIDGE_Y) / RIDGE_SIGMA) ** 2))


def _ridge_curve(img: np.ndarray) -> np.ndarray:
    """The ridge's intensity-weighted Y per column, in half-diagonals: a
    sub-pixel measurement of how straight the line is, at any resolution."""
    h, w = img.shape[:2]
    _, py = _norm_axes(w, h)
    g = img[..., 1].astype(np.float64)
    return (g * py).sum(axis=0) / g.sum(axis=0)


def _bow(img: np.ndarray) -> float:
    c = _ridge_curve(img)
    return float(c.max() - c.min())


# Compact bumps: localized, smooth, and cheap enough to draw one per channel.
BLOB_R = 0.07                          # bump radius, in half-diagonals
BLOB_N = 48
CHANNEL_LEVEL = (0.85, 1.0, 0.7)       # the channels differ in level, not in shape


def _blob_centres() -> list[tuple[float, float]]:
    rng = np.random.default_rng(3)
    return [(rng.uniform(-0.72, 0.72), rng.uniform(-0.48, 0.48)) for _ in range(BLOB_N)]


def _blobs(w: int, h: int, ca: float = 0.0) -> np.ndarray:
    """Bumps at fixed normalized positions, with red scaled out by `ca` and blue
    in by the same amount about the optical centre — lateral chromatic
    aberration of a known size.

    Each channel is *drawn* at its own scale rather than resampled from a common
    one, so the aberration is exact: a resample would soften red and blue but
    not green, and an estimator asked to align a soft channel to a sharp one has
    been handed a different problem. Only each bump's own bounding box is
    touched, which keeps the cost of a 6 MP scene in milliseconds.
    """
    cx, cy, r = lens._centre(w, h)
    px = (np.arange(w, dtype=np.float32) - cx) / r
    py = (np.arange(h, dtype=np.float32) - cy) / r
    img = np.zeros((h, w, 3), dtype=np.float32)
    for ch, level in enumerate(CHANNEL_LEVEL):
        g = 1.0 + (ca if ch == 0 else -ca if ch == 2 else 0.0)
        for bx, by in _blob_centres():
            # The channel holds F(g p), so a bump of F centred at b shows up at
            # b / g with its radius scaled to match.
            x0 = max(0, int(np.floor(((bx - BLOB_R) / g) * r + cx)))
            x1 = min(w, int(np.ceil(((bx + BLOB_R) / g) * r + cx)) + 1)
            y0 = max(0, int(np.floor(((by - BLOB_R) / g) * r + cy)))
            y1 = min(h, int(np.ceil(((by + BLOB_R) / g) * r + cy)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            dx, dy = g * px[x0:x1] - bx, g * py[y0:y1] - by
            d2 = np.square(dy)[:, None] + np.square(dx)[None, :]
            img[y0:y1, x0:x1, ch] += level * np.clip(1.0 - d2 / BLOB_R ** 2, 0.0, None) ** 3
    return img


def _misalign(img: np.ndarray, margin: float = 0.04) -> tuple[float, float]:
    """How far red and blue are out of register with green, as the RMS of the
    difference once the known per-channel level is divided out. Exact for this
    scene — aligned channels are the same picture times a constant — so it needs
    no high-pass and no tuning, and it means the same thing at any size. The
    outer `margin` is dropped: a channel scaled outwards has nothing to read
    there and the frame's edge gets replicated instead.
    """
    h, w = img.shape[:2]
    b = int(margin * max(h, w))
    inner = img[b:h - b, b:w - b]
    green = inner[..., 1]
    return tuple(float(np.sqrt(np.mean((inner[..., ch] / CHANNEL_LEVEL[ch] - green) ** 2)))
                 for ch in (0, 2))


# ----- schema -------------------------------------------------------------

def test_defaults_are_neutral() -> None:
    assert lens.is_neutral(None)
    assert lens.is_neutral({})
    assert lens.is_neutral(lens.DEFAULT_LENS)
    assert lens.normalize({}) is None
    assert lens.normalize(lens.DEFAULT_LENS) is None
    assert lens.normalize("nonsense") is None
    assert lens.normalize(None) is None


def test_a_midpoint_on_its_own_is_not_a_correction() -> None:
    """Or the panel would look permanently dirty the moment it opened."""
    assert lens.normalize({"vignette_midpoint": 90}) is None
    assert lens.is_neutral({"vignette_midpoint": 0})
    assert not lens.is_neutral({"vignette_amount": -5})


def test_normalize_clamps_and_rounds() -> None:
    p = lens.normalize({"distortion": 999, "ca_red_cyan": -999,
                        "vignette_amount": 12.64, "vignette_midpoint": -3})
    # Tenths, as every slider holds (see the sliders module).
    assert p == {"distortion": 100, "ca_red_cyan": -100, "ca_blue_yellow": 0,
                 "ca_auto": False, "vignette_amount": 12.6, "vignette_midpoint": 0}
    # Garbage in a field leaves that field neutral instead of poisoning the dict.
    assert lens.normalize({"distortion": "wide", "ca_auto": 1}) == {
        **lens.DEFAULT_LENS, "ca_auto": True}


def test_is_geometric() -> None:
    assert not lens.is_geometric(None)
    assert not lens.is_geometric({})
    assert not lens.is_geometric({"vignette_amount": -100, "vignette_midpoint": 10})
    assert lens.is_geometric({"distortion": 1})
    assert lens.is_geometric({"ca_red_cyan": -1})
    assert lens.is_geometric({"ca_blue_yellow": 1})
    assert lens.is_geometric({"ca_auto": True})


# ----- neutral is free ----------------------------------------------------

def test_neutral_returns_the_same_array() -> None:
    """Byte-exact is not enough: a neutral correction must not resample at all,
    and the only way to be sure of that is to get the input array itself back."""
    img = _blobs(120, 90)
    for params in (None, {}, lens.DEFAULT_LENS, {"vignette_midpoint": 100},
                   {"distortion": 0, "ca_auto": False}):
        assert lens.apply_lens(img, params) is img, params


def test_apply_lens_insists_on_float_rgb() -> None:
    with pytest.raises(AssertionError):
        lens.apply_lens(np.zeros((8, 8, 3), np.uint8), {"distortion": 10})
    with pytest.raises(AssertionError):
        lens.apply_lens(np.zeros((8, 8), np.float32), {"distortion": 10})


def test_a_geometric_correction_refuses_a_window() -> None:
    """`is_geometric` exists so the caller can ask first; if it asks and ignores
    the answer, it gets a stack trace rather than a subtly wrong export."""
    img = _blobs(60, 40)
    with pytest.raises(AssertionError):
        lens.apply_lens(img, {"distortion": 10}, (0.1, 0.1, 0.3, 0.3))


# ----- distortion ---------------------------------------------------------

def test_barrel_bows_outward_and_the_slider_undoes_it() -> None:
    """Positive `distortion` corrects barrel, which is Lightroom's convention.

    The input here is the ridge as a barrel lens would have drawn it: the middle
    of a straight line sits further from the frame's midline than its ends do.
    """
    w, h = 801, 601
    barrel = _ridge(w, h, {"distortion": 50})
    curve = _ridge_curve(barrel)
    assert curve[w // 2] > curve[0] + 0.02, "not a barrel image to begin with"
    assert curve[w // 2] > curve[-1] + 0.02
    assert _bow(barrel) > 0.03

    fixed = lens.apply_lens(barrel, {"distortion": 50})
    assert _bow(fixed) < 0.001, "the line should come back straight"
    assert abs(float(_ridge_curve(fixed).mean()) - RIDGE_Y) < 0.001


def test_pincushion_bows_inward_and_the_slider_undoes_it() -> None:
    w, h = 801, 601
    pin = _ridge(w, h, {"distortion": -50})
    curve = _ridge_curve(pin)
    assert curve[w // 2] < curve[0] - 0.015, "not a pincushion image to begin with"
    assert curve[w // 2] < curve[-1] - 0.015

    fixed = lens.apply_lens(pin, {"distortion": -50})
    assert _bow(fixed) < 0.001
    assert abs(float(_ridge_curve(fixed).mean()) - RIDGE_Y) < 0.001


def test_the_two_directions_are_not_the_same_direction() -> None:
    """A straight line put through each slider bows the opposite way."""
    straight = _ridge(801, 601)
    assert _bow(straight) < 1e-6
    a = _ridge_curve(lens.apply_lens(straight, {"distortion": 80}))
    b = _ridge_curve(lens.apply_lens(straight, {"distortion": -80}))
    mid = 801 // 2
    assert a[mid] < a[0] and b[mid] > b[0]


def test_the_grid_never_leaves_the_frame() -> None:
    """The documented edge decision: scale to fit, so no corner is ever invented.

    A pincushion correction has to reach outwards, and without `_fit_scale` it
    would sample past the frame and hand the crop stage black corners it cannot
    see (`editing.geometry_matrix` only knows how to cut a *tilt* back).
    """
    w, h = 640, 480
    for distortion in (-100, -60, -1, 1, 60, 100):
        map_x, map_y = lens.distortion_map(w, h, {"distortion": distortion})
        assert map_x.min() >= -0.5 and map_x.max() <= w - 0.5, distortion
        assert map_y.min() >= -0.5 and map_y.max() <= h - 0.5, distortion
    # And a pincushion correction really does reach for the corner, so the scale
    # is doing work rather than hiding behind a wide margin.
    map_x, _ = lens.distortion_map(w, h, {"distortion": -100})
    assert map_x.max() > w - 2.0


def test_distortion_map_is_the_identity_when_neutral() -> None:
    map_x, map_y = lens.distortion_map(7, 5, None)
    assert map_x.dtype == np.float32 and map_y.dtype == np.float32
    assert np.allclose(map_x, np.arange(7, dtype=np.float32)[None, :], atol=1e-4)
    assert np.allclose(map_y, np.arange(5, dtype=np.float32)[:, None], atol=1e-4)


# ----- resolution independence -------------------------------------------

def _radial_profile(w: int, h: int, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """The sample grid as a pure function of normalized radius: for a row of
    output points, their radius and the radius they read from, both in
    half-diagonals. Free of the pixel count by construction — which is exactly
    the claim under test."""
    map_x, map_y = lens.distortion_map(w, h, params)
    cx, cy, r = lens._centre(w, h)
    row = h // 2
    px = (np.arange(w, dtype=np.float64) - cx) / r
    py = (row - cy) / r
    dest = np.hypot(px, py)
    src = np.hypot((map_x[row].astype(np.float64) - cx) / r,
                   (map_y[row].astype(np.float64) - cy) / r)
    # Only the outward half of the row, so radius increases along it and can be
    # interpolated against; and not the very centre, where the ratio is 0/0.
    keep = (px > 0.0) & (dest > 0.05)
    return dest[keep], (src / dest)[keep]


def test_the_sample_grid_is_the_same_function_at_any_size() -> None:
    """The strict half of the resolution-independence contract.

    The grid is compared as a function of normalized radius, not pixel by pixel,
    because two sizes cannot land a pixel centre on the same point of the frame.
    Interpolating the fine profile onto the coarse one's radii is the only
    approximation involved, and on a function this smooth it is worth ~1e-7.
    """
    params = {"distortion": 70}
    small_u, small_f = _radial_profile(500, 300, params)
    big_u, big_f = _radial_profile(5000, 3000, params)
    got = np.interp(small_u, big_u, big_f)
    assert np.abs(got - small_f).max() < 1e-5


def test_the_correction_looks_the_same_at_ten_times_the_size() -> None:
    """The half of the contract that a person would actually notice: render the
    same continuous scene small and large, correct both, and the geometry has to
    agree. Measured on the ridge's centroid curve, resampled to common columns.

    A tolerance is unavoidable and it is not the module's fault: the two renders
    sample the scene on different lattices to begin with, and one bilinear
    resample lands on different neighbours than the other. What must not creep
    in is a *systematic* difference, so the bound is a fifth of a small-frame
    pixel's worth of the half-diagonal — far below anything a coefficient
    accidentally living in pixels could hide under.
    """
    params = {"distortion": -70}
    small = lens.apply_lens(_ridge(500, 300, params), params)
    big = lens.apply_lens(_ridge(5000, 3000, params), params)
    fs = (np.arange(500) + 0.5) / 500.0
    fb = (np.arange(5000) + 0.5) / 5000.0
    got = np.interp(fs, fb, _ridge_curve(big))
    assert np.abs(got - _ridge_curve(small)).max() < 4e-4


def test_a_downscaled_big_render_matches_the_small_one() -> None:
    """The same claim in pixels, against the honest baseline: how far apart the
    two renders of the *uncorrected* scene already are."""
    params = {"distortion": 60, "vignette_amount": 50}
    small, big = _blobs(500, 333), _blobs(3000, 1998)
    down = cv2.resize(lens.apply_lens(big, params), (500, 333),
                      interpolation=cv2.INTER_AREA)
    got = np.abs(down - lens.apply_lens(small, params))
    base = np.abs(cv2.resize(big, (500, 333), interpolation=cv2.INTER_AREA) - small)
    assert got.mean() < max(3e-4, 4.0 * base.mean())
    assert got.max() < max(0.02, 2.0 * base.max())


# ----- chromatic aberration ----------------------------------------------

def test_ca_moves_red_and_blue_and_leaves_green_alone() -> None:
    """Green is the reference channel; if it moves, the fix has become a resize."""
    img = _blobs(1200, 800)
    out = lens.apply_lens(img, {"ca_red_cyan": 100, "ca_blue_yellow": -100})
    assert np.array_equal(out[..., 1], img[..., 1]), "green must be untouched"
    assert np.abs(out[..., 0] - img[..., 0]).max() > 0.01
    assert np.abs(out[..., 2] - img[..., 2]).max() > 0.01


def _channel_radius(img: np.ndarray, ch: int) -> float:
    """The intensity-weighted mean radius of a channel, in half-diagonals."""
    px, py = _norm_axes(*img.shape[1::-1])
    u = np.hypot(px, py)
    weight = img[..., ch].astype(np.float64)
    return float((weight * u).sum() / weight.sum())


def test_the_ca_sliders_scale_the_channel_they_name() -> None:
    """Positive samples from a larger radius, which pulls the channel's image
    inwards; negative pushes it out. Green stays where it is either way."""
    img = _blobs(1200, 800)
    r0, g0, b0 = (_channel_radius(img, c) for c in range(3))
    inward = lens.apply_lens(img, {"ca_red_cyan": 100, "ca_blue_yellow": 100})
    outward = lens.apply_lens(img, {"ca_red_cyan": -100, "ca_blue_yellow": -100})
    assert _channel_radius(inward, 0) < r0
    assert _channel_radius(inward, 2) < b0
    assert _channel_radius(outward, 0) > r0
    assert _channel_radius(outward, 2) > b0
    assert abs(_channel_radius(inward, 1) - g0) < 1e-12


def test_auto_ca_finds_a_known_aberration() -> None:
    """A synthetic frame whose red is scaled out and blue in by a known amount.
    The estimator should report the opposite scales, to within a few percent."""
    err = 0.002
    img = _blobs(1600, 1100, ca=err)
    a_r, a_b = lens.estimate_ca(img)
    assert abs(a_r + err) < 0.03 * err, (a_r, -err)
    assert abs(a_b - err) < 0.03 * err, (a_b, err)


def test_auto_ca_puts_the_channels_back_in_register() -> None:
    """The result that matters, rather than the number the estimator reports."""
    img = _blobs(1600, 1100, ca=0.002)
    before = _misalign(img)
    after = _misalign(lens.apply_lens(img, {"ca_auto": True}))
    assert max(before) > 0.005, "the test scene has to be misregistered first"
    assert after[0] < 0.05 * before[0], (before, after)
    assert after[1] < 0.05 * before[1], (before, after)


def test_auto_ca_holds_over_the_useful_range() -> None:
    """Small enough to be a real lens, large enough to be the end of a slider."""
    for err in (0.0005, 0.002, 0.008):
        a_r, a_b = lens.estimate_ca(_blobs(1200, 800, ca=err))
        assert abs(a_r + err) < 0.03 * err, (err, a_r)
        assert abs(a_b - err) < 0.03 * err, (err, a_b)


def test_auto_ca_survives_noise() -> None:
    rng = np.random.default_rng(0)
    img = _blobs(1200, 800, ca=0.002) + rng.normal(0, 0.01, (800, 1200, 3)).astype(np.float32)
    a_r, a_b = lens.estimate_ca(img)
    assert abs(a_r + 0.002) < 0.05 * 0.002
    assert abs(a_b - 0.002) < 0.05 * 0.002


def test_auto_and_manual_ca_add_up() -> None:
    """The estimate is a correction on top of the sliders, not instead of them."""
    img = _blobs(800, 560, ca=0.002)
    auto = lens.estimate_ca(img)
    both = lens._ca_gains({**lens.DEFAULT_LENS, "ca_auto": True,
                           "ca_red_cyan": 100, "ca_blue_yellow": -100}, img)
    assert abs(both[0] - (1.0 + lens._CA_MAX_SCALE + auto[0])) < 1e-12
    assert both[1] == 1.0
    assert abs(both[2] - (1.0 - lens._CA_MAX_SCALE + auto[1])) < 1e-12


def test_auto_ca_leaves_a_clean_frame_alone() -> None:
    """No aberration, no correction: the estimator must not invent one out of
    the difference between the channels' *content*."""
    img = _blobs(1600, 1100)
    assert not np.array_equal(img[..., 0], img[..., 1]), \
        "the channels must differ in level, or this proves nothing"
    a_r, a_b = lens.estimate_ca(img)
    assert abs(a_r) < 1e-5 and abs(a_b) < 1e-5


def test_auto_ca_agrees_across_render_sizes() -> None:
    """It is estimated on a fixed-size working copy, so a preview and an export
    reach nearly the same answer — nearly, because reducing 3000 px to 768 is
    not the same picture as reducing 1000 px to 768. `estimate_ca` documents
    this; the check is that the gap is small enough not to show."""
    err = 0.002
    small = lens.estimate_ca(_blobs(1000, 700, ca=err))
    big = lens.estimate_ca(_blobs(3000, 2100, ca=err))
    assert abs(small[0] - big[0]) < 0.02 * err, (small, big)
    assert abs(small[1] - big[1]) < 0.02 * err, (small, big)


# ----- lens vignetting ----------------------------------------------------

def _gain(params: dict, w: int = 600, h: int = 400) -> np.ndarray:
    flat = np.full((h, w, 3), 0.4, dtype=np.float32)
    return lens.apply_lens(flat, params)[..., 0] / 0.4


def test_vignette_correction_brightens_the_corners() -> None:
    g = _gain({"vignette_amount": 100})
    assert abs(g[200, 300] - 1.0) < 1e-3, "the centre is where the lens was honest"
    assert g[0, 0] > 2.0 and g[-1, -1] > 2.0
    assert g[0, -1] > 2.0 and g[-1, 0] > 2.0
    # Monotone outwards along a row: no ring, no plateau.
    row = g[200, 300:]
    assert np.all(np.diff(row) > 0)


def test_a_negative_amount_darkens_them_instead() -> None:
    g = _gain({"vignette_amount": -100})
    assert abs(g[200, 300] - 1.0) < 1e-3
    assert g[0, 0] < 0.5
    assert g[0, 0] > 0.0, "in stops, so it never reaches black"


def test_the_midpoint_moves_where_the_falloff_starts() -> None:
    """Low midpoint: the correction reaches most of the way in. High midpoint: it
    clings to the corners. The corner itself is the amount's business, not the
    midpoint's, so that is where the two agree."""
    def reach(mid: int) -> float:
        g = _gain({"vignette_amount": 100, "vignette_midpoint": mid})
        row = g[200, 300:]
        return float(np.argmax(row > 1.1)) / row.size

    reaches = [reach(m) for m in (0, 30, 50, 70, 100)]
    assert reaches == sorted(reaches), reaches
    assert reaches[0] < 0.2 and reaches[-1] > 0.5
    corners = [_gain({"vignette_amount": 100, "vignette_midpoint": m})[0, 0]
               for m in (0, 50, 100)]
    assert max(corners) - min(corners) < 0.05


def test_vignetting_on_a_window_matches_the_full_frame() -> None:
    """The point of `is_geometric` being False: the 1:1 editor view renders a
    window of the frame and it has to be the same pixels the export gets."""
    rng = np.random.default_rng(1)
    full = rng.random((400, 600, 3), dtype=np.float32)
    params = {"vignette_amount": 70, "vignette_midpoint": 35}
    ref = lens.apply_lens(full, params)
    x0, y0, ww, hh = 150, 100, 200, 120
    window = np.ascontiguousarray(full[y0:y0 + hh, x0:x0 + ww])
    got = lens.apply_lens(window, params, (x0 / 600, y0 / 400, ww / 600, hh / 400))
    assert np.abs(got - ref[y0:y0 + hh, x0:x0 + ww]).max() < 1e-6


def test_vignetting_does_not_resample() -> None:
    """A vignette-only correction is pointwise, so it must not go near the warp:
    a flat frame comes out flat and a hard edge stays hard."""
    img = _blobs(300, 200)
    out = lens.apply_lens(img, {"vignette_amount": 40})
    ratio = out[..., 1] / np.maximum(img[..., 1], 1e-6)
    # One gain per pixel, independent of the channel and of the neighbours.
    for ch in (0, 2):
        assert np.allclose(out[..., ch], img[..., ch] * ratio, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
