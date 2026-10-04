"""Lens corrections: distortion, lateral chromatic aberration, vignetting.

Lightroom's "Lens Corrections > Manual" tab, as three independent corrections of
the *optics*, not creative effects. They undo what the glass did:

  1. `distortion`        the radial polynomial. A lens does not project a
                         straight line to a straight line; this puts it back.
  2. chromatic aberration  the lens focuses red, green and blue at slightly
                         different magnifications, so an off-centre edge comes
                         out with a coloured fringe. The fix is a per-channel
                         radial scale about the optical centre — manual
                         (`ca_red_cyan`, `ca_blue_yellow`) or estimated from the
                         frame (`ca_auto`).
  3. lens vignetting     glass and the aperture pass less light towards the
                         corners. `vignette_amount` / `vignette_midpoint` add it
                         back.

Only numpy + OpenCV, like the rest of the pipeline.

Not the creative vignette
-------------------------
`editing._apply_vignette` is a different thing that happens to darken corners
too. That one is a *look*: it is applied last, after the crop, anchored to the
frame the viewer ends up with, and its whole point is to draw the eye inwards.
This one is a *correction*: it is applied first, to the frame as the sensor saw
it, and its whole point is to be invisible — a flat grey wall should come out
flat. The two are not redundant and are not interchangeable; corrected first,
then styled, is the same order Lightroom uses and the reason both exist.

Resolution independence
-----------------------
This is the contract the module lives or dies by: the same parameters must
produce the same *look* on a 1000 px preview and on a 6000 px export, because
the editor grades against the preview. So no coefficient here is ever in pixels.
Every one is expressed in normalized coordinates — a pixel's offset from the
optical centre divided by half the frame diagonal, so u = 1 at the frame's
corners whatever the sensor and whatever the render size. A pixel's normalized
position depends only on where it sits in the frame as a fraction (the
`(i + 0.5) / n` pixel-centre convention `editing._grid` uses, for the same
reason: endpoint sampling puts a window's grid half a pixel off from the matching
slice of the full frame's, and the 1:1 view then drifts from the export).

Cost
----
One resampling pass, never two. Distortion and CA are both radial scales about
the same centre, so they compose analytically into ONE sample grid per channel;
chaining two `cv2.remap` calls would cost a second pass over the megapixels and,
worse, interpolate twice — every warp softens, and softening a frame twice to
apply one correction is a real, visible loss. When CA is off, all three channels
share a grid and it is a single three-channel remap.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from .sliders import tenth

# ----- schema -------------------------------------------------------------

# Neutral everywhere. `vignette_midpoint` is the one field whose neutral is not
# zero: it says *where* the falloff sits and means nothing until an amount asks
# for one, so it never counts towards non-neutrality on its own.
DEFAULT_LENS: dict[str, Any] = {
    "distortion": 0,          # -100..100; + corrects barrel, - corrects pincushion
    "ca_red_cyan": 0,         # -100..100; red channel scaled out (+) or in (-)
    "ca_blue_yellow": 0,      # -100..100; blue channel scaled out (+) or in (-)
    "ca_auto": False,         # estimate the two above from the frame
    "vignette_amount": 0,     # -100..100; + brightens the corners
    "vignette_midpoint": 50,  # 0..100; how far in the falloff reaches
}

_RANGES: dict[str, tuple[float, float]] = {
    "distortion": (-100, 100),
    "ca_red_cyan": (-100, 100),
    "ca_blue_yellow": (-100, 100),
    "vignette_amount": (-100, 100),
    "vignette_midpoint": (0, 100),
}

FULL_ROI = (0.0, 0.0, 1.0, 1.0)

# How far the corner moves at +/-100, as a fraction of the half-diagonal. 25% is
# about a strong wide-angle zoom's barrel at the short end — past anything a
# rectilinear lens does, which is what the end of a slider is for. The cubic
# stays monotone as long as this is under 1/3.
_DISTORTION_K = 0.25

# Per-channel radial scale at +/-100. Lateral CA is small: a few pixels at the
# corner of a full-size frame, so a few tenths of a percent of the radius.
_CA_MAX_SCALE = 0.003

# Lens vignetting at +/-100, in stops at the corner, and how sharp the falloff
# is. The exponent is 2 at the neutral midpoint, matching the cos^4-ish shape of
# a real lens falloff (and the creative vignette's own quadratic).
_VIG_MAX_STOPS = 1.5
_VIG_EXP_MID = 2.0

# Automatic CA: the long edge the estimate is made at, and how many
# Gauss-Newton steps. See `estimate_ca`.
_CA_AUTO_EDGE = 768
_CA_AUTO_ITERS = 4
_CA_AUTO_LIMIT = 0.008   # nothing beyond this is lateral CA; refuse to "fix" it
_CA_AUTO_MIN_R = 0.25    # ignore the middle of the frame: no radius, no signal
_CA_AUTO_MARGIN = 0.02   # ...and the outermost band, which has no neighbours
_CA_AUTO_BAND = (1.5, 6.0)  # the scales the estimate is made at; see `_bandpass`

_EPS = 1e-5


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp into a full parameter dict, or None when the result would change
    nothing — the same contract as `editing.normalize_hsl`, so a lens panel the
    user opened and did not touch never makes an edit look non-neutral."""
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_LENS)
    out["ca_auto"] = bool(raw.get("ca_auto", False))
    for key, (lo, hi) in _RANGES.items():
        if raw.get(key) is None:
            continue
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            continue
        out[key] = tenth(min(hi, max(lo, val)))
    if is_neutral(out):
        return None
    return out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when the corrections would leave the pixels alone."""
    if not params:
        return True
    p = {**DEFAULT_LENS, **params}
    return (p["distortion"] == 0 and p["ca_red_cyan"] == 0
            and p["ca_blue_yellow"] == 0 and not p["ca_auto"]
            and p["vignette_amount"] == 0)


def is_geometric(params: dict[str, Any] | None) -> bool:
    """True when the corrections move pixels, rather than only changing their
    brightness.

    The caller needs this to know whether a window render is safe. A vignetting
    correction is pointwise: hand it a window plus the `roi` saying where that
    window sits and it lands exactly where it would in the full-frame render. A
    geometric correction is not: the pixels a window needs come from outside the
    window, and how far outside depends on the whole frame, so a window render
    has to go through the caller's own source-box machinery (`editing.
    geometry_source_box`) or not at all.
    """
    if not params:
        return False
    p = {**DEFAULT_LENS, **params}
    return bool(p["distortion"] != 0 or p["ca_red_cyan"] != 0
                or p["ca_blue_yellow"] != 0 or p["ca_auto"])


# ----- normalized geometry ------------------------------------------------
#
# The optical centre is the frame centre, and the unit of length is half the
# frame diagonal: a pixel at column j sits at X = (j - (w-1)/2) / (hypot(w,h)/2).
# Two things about that expression matter more than they look:
#
#   - the centre uses w-1 (the last pixel *index*) but the scale uses w (the
#     frame's extent). That is the `(j + 0.5) / w` pixel-centre convention
#     rearranged, and it is what makes X depend only on the aspect ratio, never
#     on the pixel count. Using hypot(w-1, h-1) instead would leave an O(1/w)
#     aspect-dependent term in every coefficient and quietly break the whole
#     resolution-independence contract for a half-pixel's worth of tidiness.
#   - u = 1 falls on the frame's geometric corner, so a corner *pixel centre* is
#     a hair inside it. Bounds below are taken at u = 1 regardless, which is
#     conservative by a fraction of a pixel and, again, independent of size.


def _centre(w: int, h: int) -> tuple[float, float, float]:
    """(cx, cy, R): the optical centre in pixel indices and the half-diagonal."""
    assert w > 0 and h > 0, "a frame needs a size"
    return (w - 1) / 2.0, (h - 1) / 2.0, math.hypot(w, h) / 2.0


def _fit_scale(k: float) -> float:
    """The uniform scale that keeps a distortion correction inside the frame.

    A sample grid r -> r * (1 + k r^2) with k > 0 reaches past the frame edge, so
    the corrected frame would have to invent pixels in its corners. Scaling the
    whole grid by s pulls it back: the binding constraint is the corner, u = 1,
    where s * (1 + k s^2) must not exceed 1. Newton from s = 1 on a cubic with
    one positive root.
    """
    if k <= 0.0:
        return 1.0
    s = 1.0
    for _ in range(40):
        s -= (k * s ** 3 + s - 1.0) / (3.0 * k * s * s + 1.0)
    assert abs(k * s ** 3 + s - 1.0) < 1e-12, "fit scale did not converge"
    return s


def _distortion_coeff(params: dict[str, Any] | None) -> tuple[float, float]:
    """(k, s) for a parameter dict: the cubic coefficient of the sample grid and
    the scale that keeps it in frame.

    Sign, stated once so the rest of the module can stop worrying about it. The
    slider follows Lightroom: **positive corrects barrel** (lines that bow away
    from the centre), **negative corrects pincushion** (lines that bow towards
    it). A barrel image is one whose magnification falls off with radius, so
    correcting it means magnifying the periphery, which means *sampling from a
    smaller* radius — k < 0. Hence the minus sign here.
    """
    p = {**DEFAULT_LENS, **(params or {})}
    k = -(p["distortion"] / 100.0) * _DISTORTION_K
    return k, _fit_scale(k)


def distortion_map(w: int, h: int,
                   params: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """The `cv2.remap` sample grid for the distortion correction of a `w` x `h`
    frame: for each output pixel, where to read the input.

    Two float32 arrays (x then y), in input pixel coordinates. This is the grid
    the green channel takes; red and blue ride a per-channel radial scale on top
    of it (see `apply_lens`). Neutral parameters give the identity grid rather
    than None, so a caller can compose or cache unconditionally.
    """
    k, s = _distortion_coeff(params)
    cx, cy, r = _centre(w, h)
    return _radial_grid(w, h, cx, cy, r, k, s, 1.0)


def _radial_grid(w: int, h: int, cx: float, cy: float, r: float,
                 k: float, s: float, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """The grid for one channel: source = dest * gain * s * (1 + k (s u)^2).

    `gain` is the channel's CA scale, folded in here rather than applied as a
    second warp — one interpolation per channel, which is the whole point.
    """
    px = (np.arange(w, dtype=np.float32) - np.float32(cx))
    py = (np.arange(h, dtype=np.float32) - np.float32(cy))
    # u^2 of every pixel, built by broadcasting two 1-D arrays: the only
    # full-frame temporary the grid needs besides the maps themselves.
    f = np.square(py / np.float32(r))[:, None] + np.square(px / np.float32(r))[None, :]
    f *= np.float32(k * s * s)
    f += np.float32(1.0)
    f *= np.float32(s * gain)
    map_x = f * px[None, :]
    map_x += np.float32(cx)
    map_y = f * py[:, None]
    map_y += np.float32(cy)
    return map_x, map_y


# ----- chromatic aberration -----------------------------------------------

def _ca_gains(params: dict[str, Any] | None,
              rgb: np.ndarray | None,
              ca: tuple[float, float] | None = None) -> tuple[float, float, float]:
    """(gain_r, gain_g, gain_b): the radial scale each channel is sampled at.

    Green is the reference and is never touched — it is the channel the lens is
    focused for, it carries most of the luminance, and moving all three would
    turn a fringe fix into a resize. Positive `ca_red_cyan` samples red from a
    larger radius, which pulls the red image inwards and kills a red fringe on
    the outward side of an off-centre edge; negative does the reverse and kills
    the cyan one. `ca_blue_yellow` says the same thing about blue against yellow.
    """
    p = {**DEFAULT_LENS, **(params or {})}
    a_r = (p["ca_red_cyan"] / 100.0) * _CA_MAX_SCALE
    a_b = (p["ca_blue_yellow"] / 100.0) * _CA_MAX_SCALE
    if p["ca_auto"]:
        if ca is None:
            assert rgb is not None, "automatic CA needs the frame to estimate from"
            ca = estimate_ca(rgb)
        auto_r, auto_b = ca
        a_r += auto_r
        a_b += auto_b
    return 1.0 + a_r, 1.0, 1.0 + a_b


def estimate_ca(rgb: np.ndarray) -> tuple[float, float]:
    """Estimate the radial scale that aligns red and blue to green.

    Returns the two *corrections* (the `a` in a sampling gain of 1 + a), so
    `_ca_gains` can add them to whatever the sliders say.

    The estimator. Lateral CA is, to first order, a pure magnification
    difference: the red image is the green one scaled about the optical centre.
    So there is exactly one unknown per channel, and one unknown is worth
    solving directly rather than searching for. Warping a channel C by a and
    linearising in a,

        C(p (1 + a)) ~= C(p) + a * (p - c) . grad C(p)

    and asking for that to equal green gives a weighted least squares with one
    parameter, w = (p - c) . grad C being the radial derivative times the radius:

        a = sum w (G - C) / sum w^2

    which is a Gauss-Newton step on the one-parameter radial-scale group -- a
    Lucas-Kanade tracker with the search space cut down to the only motion a
    lens can produce. It is iterated `_CA_AUTO_ITERS` times, re-warping the
    channel each round, because the linearisation is only good for shifts under
    about a pixel.

    Four details it does not work without:
      - both channels are band-passed first, and the band matters. Dropping the
        low frequencies is obvious enough: raw R and G differ by the *colour* of
        the scene far more than by CA, and a least squares fed that difference
        would chase the colour. Dropping the highest ones is the part that is
        easy to miss and was worth about a factor of two — detail near the
        sampling limit is where a resize aliases, where the two channels alias
        *differently* because one is scaled before it is sampled, and where a
        central-difference gradient is furthest from a real derivative. Aligning
        on it made the answer depend on the render size, which is the one thing
        this module is not allowed to do.
      - each band is divided by its own RMS, so a dim channel over a strong one
        does not read as a shift.
      - the middle of the frame is masked out. Lateral CA is zero at the centre
        by construction, so those pixels contribute noise to both sums and no
        signal to either.
      - so is a thin band around the outside. A channel scaled outwards reads
        from beyond the frame there and gets the edge pixel replicated instead;
        that strip is a couple of pixels wide and a full amplitude wrong, which
        is easily enough to bend the estimate.

    Cost is bounded: the estimate is made on a copy no longer than
    `_CA_AUTO_EDGE`, so it is a fixed handful of milliseconds plus one pass to
    reduce the frame (about 45 ms of it at 24 MP), and the answer is a
    normalized scale that then costs nothing to apply at full size. Being a scale rather than a pixel count is also what lets the estimate
    made on a preview be used on the export.

    The result is content-derived, and only approximately invariant to the size
    it was estimated at: a 6000 px frame reduced to 768 is not bit-identical to
    a 1000 px preview reduced to 768. The disagreement is small (see the tests)
    but it is not zero, so a caller who needs a preview and an export to match
    exactly should call this once and store the answer in the sliders.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "estimate_ca wants an RGB frame"
    h, w = rgb.shape[:2]
    scale = _CA_AUTO_EDGE / max(h, w)
    if scale < 1.0:
        # Low-pass to the target rate before resampling, but only for a mild
        # reduction: INTER_AREA averages the source pixels a destination one
        # covers, which past 2x is already the right anti-aliasing filter, and
        # blurring a 24 MP frame to throw most of it away costs more than the
        # whole rest of the estimate. Under 2x its box is barely wider than a
        # pixel and detail near the sampling limit folds back, which is exactly
        # the band the aligner must not be shown.
        if scale > 0.5:
            small = cv2.GaussianBlur(rgb, (0, 0), 0.5 / scale)
        else:
            small = rgb
        small = cv2.resize(small, (max(8, int(round(w * scale))),
                                   max(8, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = np.asarray(rgb, dtype=np.float32)
    sh, sw = small.shape[:2]
    cx, cy, r = _centre(sw, sh)

    px = (np.arange(sw, dtype=np.float32) - np.float32(cx))
    py = (np.arange(sh, dtype=np.float32) - np.float32(cy))
    u2 = np.square(py / np.float32(r))[:, None] + np.square(px / np.float32(r))[None, :]
    inset = np.zeros((sh, sw), dtype=bool)
    margin = max(2, int(round(_CA_AUTO_MARGIN * max(sh, sw))))
    inset[margin:sh - margin, margin:sw - margin] = True
    mask = ((u2 >= _CA_AUTO_MIN_R ** 2) & inset).astype(np.float32)

    green = _bandpass(small[..., 1])
    out = []
    for ch in (0, 2):
        a = 0.0
        for _ in range(_CA_AUTO_ITERS):
            gain = 1.0 + a
            chan = np.ascontiguousarray(small[..., ch])
            warped = _remap(chan, *_radial_grid(sw, sh, cx, cy, r, 0.0, 1.0, gain))
            band = _bandpass(warped)
            # w = (p - c) . grad C. Both in pixels of this working copy, which
            # is what leaves `a` dimensionless and therefore size-independent.
            # The gradient is taken before the mask, never after: differentiating
            # across a masked-off region reads its border as an edge steeper
            # than anything in the picture, and that one artefact outweighs the
            # whole frame.
            gy, gx = np.gradient(band)
            weight = (gx * px[None, :] + gy * py[:, None]) * mask
            denom = float(np.sum(weight * weight))
            if denom <= _EPS:
                break
            # Composing two scales is (1+a)(1+da); adding drops an a*da term
            # of order 1e-5 of a pixel, which is below what a remap can express.
            a += float(np.sum(weight * (green - band))) / denom
            a = float(np.clip(a, -_CA_AUTO_LIMIT, _CA_AUTO_LIMIT))
        out.append(a)
    return out[0], out[1]


def _bandpass(chan: np.ndarray) -> np.ndarray:
    """Structure at the scales the aligner can trust, normalized to unit RMS: a
    difference of Gaussians, so neither the scene's colour nor the sampling
    limit's aliasing gets a vote. See `estimate_ca`."""
    lo, hi = _CA_AUTO_BAND
    band = cv2.GaussianBlur(chan, (0, 0), lo) - cv2.GaussianBlur(chan, (0, 0), hi)
    rms = float(np.sqrt(np.mean(band * band)))
    return band / max(rms, _EPS)


# ----- vignetting ---------------------------------------------------------

def _vignette_gain(h: int, w: int, params: dict[str, Any],
                   roi: tuple[float, float, float, float]) -> np.ndarray:
    """The per-pixel multiplier that puts the lens's corner falloff back.

    gain = 2 ** (stops * u ** n): a symmetric correction *in stops*, so +100 adds
    `_VIG_MAX_STOPS` at the corner and -100 takes the same away. A linear
    multiplier would have driven the corners to literal zero at -100.

    `vignette_midpoint` sets n, from 1 (the falloff reaches most of the way to
    the centre) through 2 at the neutral 50 to 4 (it clings to the corners) —
    the same job as Lightroom's Midpoint: how far in the correction reaches.
    """
    x0, y0, rw, rh = roi
    # The full frame's aspect, recovered from the window's: a window is `rw` of
    # the frame wide and `w` pixels wide, so the frame is w / rw pixels wide.
    fw, fh = w / rw, h / rh
    r = math.hypot(fw, fh) / 2.0
    # Whole-frame pixel-centre coordinates, then offsets from the optical centre,
    # in half-diagonals. Same convention as `_centre`, expressed through `roi`.
    fx = (x0 + (np.arange(w, dtype=np.float32) + 0.5) / w * rw) * np.float32(fw)
    fy = (y0 + (np.arange(h, dtype=np.float32) + 0.5) / h * rh) * np.float32(fh)
    px = (fx - np.float32(fw / 2.0)) / np.float32(r)
    py = (fy - np.float32(fh / 2.0)) / np.float32(r)
    u2 = np.square(py)[:, None] + np.square(px)[None, :]

    n = _VIG_EXP_MID ** (1.0 + (params["vignette_midpoint"] - 50.0) / 50.0)
    stops = (params["vignette_amount"] / 100.0) * _VIG_MAX_STOPS
    return np.exp2(np.float32(stops) * np.power(u2, np.float32(n / 2.0)))


# ----- top level ----------------------------------------------------------

def _remap(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    # BORDER_REPLICATE, like `editing.apply_geometry`: the grid is kept inside
    # the frame by `_fit_scale`, so this only ever catches the last half pixel
    # at an edge (and the CA scale's fraction of one), where replicating is
    # invisible and a black border would not be.
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def apply_lens(rgb: np.ndarray, params: dict[str, Any] | None,
               roi: tuple[float, float, float, float] = FULL_ROI,
               ca: tuple[float, float] | None = None) -> np.ndarray:
    """Apply the lens corrections to a float32 RGB frame in [0,1].

    `rgb` must be a whole frame unless `is_geometric(params)` is False, in which
    case it may be a window and `roi` (x0, y0, w, h, in normalized whole-frame
    coordinates) says where that window sits — the same `roi` contract as
    `editing.render`. Neutral parameters return the input array itself, so a
    neutral correction costs nothing and resamples nothing.

    The result is not clipped: brightening the corners of an already-bright frame
    goes over 1.0, and the pipeline clips once at the end rather than at every
    stage (`editing.render`).

    Order: vignetting, then the warp. The falloff is a property of the lens as
    it exposed the sensor, so it is radial in the frame the sensor recorded;
    undoing it before the geometry is straightened is the only order in which it
    lands on the pixels it actually dimmed.

    The frame comes out the size it went in, always fully covered, with no
    invented pixels. Correcting barrel magnifies the periphery and simply throws
    the outermost ring away; correcting pincushion would have to reach outside
    the frame, so the whole grid is scaled to fit instead (`_fit_scale`). The
    alternative — leave the true curved outline and hand back black corners — was
    rejected because it pushes the problem downstream onto a crop stage that
    cannot see it: `editing.geometry_matrix` cuts a tilt back to the largest
    inscribed rectangle for exactly this reason, and it does so from the tilt
    angle alone. It has no idea what shape a lens correction left behind, so
    black corners here would survive all the way into the export. A scale is
    also cheap to be honest about: it is one normalized number, so it costs the
    same fraction of the field of view at every render size.

    `ca` is an `estimate_ca` the caller already has, for `ca_auto`: a scale,
    so one made on a photo's preview serves every render of it, and a drag
    need not estimate it again on every frame. Without it the frame is asked.
    """
    assert rgb.dtype == np.float32, "lens corrections work in float32"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "apply_lens wants an RGB frame"
    if is_neutral(params):
        return rgb
    p = {**DEFAULT_LENS, **params}
    geometric = is_geometric(p)
    assert not (geometric and roi != FULL_ROI), \
        "a geometric lens correction needs the whole frame; check is_geometric()"

    out = rgb
    h, w = rgb.shape[:2]
    if p["vignette_amount"] != 0:
        out = out * _vignette_gain(h, w, p, roi)[..., None]
    if not geometric:
        return out

    k, s = _distortion_coeff(p)
    cx, cy, r = _centre(w, h)
    gains = _ca_gains(p, rgb, ca)
    if gains[0] == 1.0 and gains[2] == 1.0:
        # One grid for all three channels: a single three-channel remap, which
        # is what OpenCV is fastest at and what most edits will hit.
        return _remap(out, *_radial_grid(w, h, cx, cy, r, k, s, 1.0))
    warped = np.empty_like(out)
    for ch, gain in enumerate(gains):
        warped[..., ch] = _remap(np.ascontiguousarray(out[..., ch]),
                                 *_radial_grid(w, h, cx, cy, r, k, s, gain))
    return warped
