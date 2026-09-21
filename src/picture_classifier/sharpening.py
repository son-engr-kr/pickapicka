"""Capture sharpening: the full Detail panel, not a single Amount slider.

Four parameters, the same four Lightroom offers, because between them they say
everything there is to say about an unsharp mask:

  - `amount`  (0..100)  how hard to push. Neutral at 0, and the whole stage is
                        a no-op there.
  - `radius`  (0..100)  the size of the edge being enhanced, 0.5..3.0 px.
                        Neutral at the midpoint (50), about 1.2 px.
  - `detail`  (0..100)  how much of the finest structure is admitted, and how
                        much overshoot an edge is allowed. Low means only real
                        edges are restored and nothing halos; high means fabric,
                        pores and grain all come up too.
  - `masking` (0..100)  an edge mask. 0 sharpens the whole frame; 100 sharpens
                        only where there is genuinely an edge, so a sky or a
                        cheek keeps its noise unamplified.

Only numpy + OpenCV, the same two the rest of the pipeline uses.

Structure of the operator, and why it is not one unsharp mask:

  y is split at two scales, sigma_f = 0.4 * sigma_r and sigma_r itself. That
  gives a mid band (structure at the radius scale — the edges the slider is
  named after) and a fine band (everything finer, which is texture and noise in
  equal measure). `detail` weights the fine band. So detail changes which
  frequencies are amplified rather than how much of one fixed mixture is, which
  is the difference between a second Amount slider and a real one.

  The result is then clamped, per pixel, into the range of luma values already
  present in the neighbourhood — a pixel may be pushed up as far as the
  brightest thing near it and no further. This is what stops halos at the
  source: on a step edge the local maximum *is* the bright plateau, so the
  transition can be made as steep as you like and never overshoot it. `detail`
  buys slack on top of that range, proportional to the local range itself, so
  turning detail up lets peaks exceed their neighbours (crisp, and eventually
  halos) while turning it down gives the strictly monotone edge that low detail
  means. The clamp is a smooth rational squash, not a hard cut, so nothing in
  the output has a crease in it that the input did not.

  `masking` scales the finished delta by a gradient-magnitude mask, thresholded
  with a smoothstep. Deliberately last: the clamp decides how big a delta is
  safe, and the mask then decides where it applies.

Everything is measured on luma and the delta is added back to all three
channels, so hue survives (`editing._apply_clarity` does the same).

Resolution: this stage is deliberately NOT resolution-independent
-----------------------------------------------------------------
The pipeline's central promise is that the preview and the export look the
same, and almost every stage keeps it by measuring its radius against
`frame_long` (see `editing._lowfreq`, `_apply_lens_blur`, `film._grain_field`).
Sharpening will not fit in that frame, and pretending otherwise breaks it in a
worse way than admitting it:

  - `radius` in absolute pixels means a 1000 px preview and a 6000 px export
    get visibly different sharpening. True, and this is the cost paid below.
  - `radius` scaled by `frame_long` means the export gets a 6x wider radius —
    an 8 px "sharpening" halo instead of a 1.3 px one. That is not a bigger
    version of the same look, it is a different operator: local contrast, which
    this pipeline already has two of (`clarity` and `texture`, both correctly
    frame-relative). Scaling would also be self-defeating, because the thing
    sharpening fights — the sensor's and the demosaic's ~1 px softness — does
    not get wider when the file does.

So sharpening is per-pixel, and `frame_long` is accepted and then deliberately
not used to scale anything. `editing._apply_denoise` made exactly this call
already, for exactly this reason (noise is a per-pixel quantity; a downscaled
preview has already averaged it away), and flagged itself as the pipeline's
documented exception. This is the second such stage and the last one that
should be.

What a user should expect, stated plainly: judge sharpening at 1:1 and it will
be what you exported. A fit-to-window preview of a 24 MP file has been
downscaled 6x, which has already thrown away the frequencies this stage works
on, so the preview will look softer than the export at any setting — the same
thing Lightroom tells you, and for the same reason.

`frame_long` stays in every public signature anyway so the caller treats this
stage like the others and nothing has to change shape if that judgement is ever
revisited.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ----- schema -------------------------------------------------------------

DEFAULT_SHARPEN: dict[str, int] = {
    "amount": 0,     # 0..100; neutral, and the only key that decides neutrality
    "radius": 50,    # 0..100 -> _RADIUS_MIN..._RADIUS_MAX px; midpoint is neutral
    "detail": 25,    # 0..100; Lightroom's default sits here, and so does this one
    "masking": 0,    # 0..100; 0 means the whole frame
}

_RANGES: dict[str, tuple[int, int]] = {
    "amount": (0, 100),
    "radius": (0, 100),
    "detail": (0, 100),
    "masking": (0, 100),
}

# The radius slider is geometric, not linear: radius is a scale, so equal slider
# steps should be equal ratios. It also puts the midpoint at ~1.2 px, which is
# where a demosaiced frame actually wants to be sharpened, instead of the 1.75 px
# a linear map would make the default.
_RADIUS_MIN = 0.5        # px of Gaussian sigma at slider 0
_RADIUS_MAX = 3.0        # ...and at slider 100
_FINE_FRAC = 0.4         # the fine/mid split, as a fraction of the radius
_FINE_SIGMA_MIN = 0.35   # below this a Gaussian is no longer a blur, just noise

_AMOUNT_MAX = 1.5        # high-pass gain at amount 100; matches the old one-slider stage
_FINE_WEIGHT = (0.15, 1.30)   # fine-band weight at detail 0 and at detail 100
_OVERSHOOT_MAX = 0.35    # slack past the local range at detail 100, as a fraction of it

_MASK_GRAD_MAX = 0.08    # luma per pixel; the gradient that still passes at masking 100
_MASK_SOFT = 0.25        # the mask's ramp opens at this fraction of the threshold
_MASK_BLUR_FRAC = 2.0    # mask blur sigma, in radii

_EPS = 1e-6


def normalize(raw: Any) -> dict[str, int] | None:
    """Clamp into a full parameter dict, or None when it would change nothing.

    Same contract as `editing.normalize_hsl` and `film.normalize`: a panel the
    user opened and did not touch must normalize to None, so it never makes an
    edit count as non-neutral and never lengthens the edit hash.

    A bare number is accepted as `amount` with everything else default. That is
    the old single-slider form, which is what is already stored in the database.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = {"amount": raw}
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_SHARPEN)
    for key, (lo, hi) in _RANGES.items():
        if raw.get(key) is None:
            continue
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            continue
        out[key] = int(round(min(hi, max(lo, val))))
    if is_neutral(out):
        return None
    return out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when the stage would leave the pixels alone.

    Only `amount` decides this. Radius, detail and masking describe *how* to
    sharpen; with nothing to apply they describe nothing.
    """
    if not params:
        return True
    return int(round(float(params.get("amount") or 0))) <= 0


# ----- geometry of the operator -------------------------------------------

def _sigmas(params: dict[str, Any]) -> tuple[float, float]:
    """The radius scale and the fine scale, both in pixels of the given array."""
    radius = float(params["radius"])
    sigma_r = _RADIUS_MIN * (_RADIUS_MAX / _RADIUS_MIN) ** (radius / 100.0)
    return sigma_r, max(_FINE_SIGMA_MIN, sigma_r * _FINE_FRAC)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _edge_mask(base: np.ndarray, sigma_r: float, masking: int,
               morph: np.ndarray) -> np.ndarray:
    """Where there is an edge worth sharpening, in [0,1].

    The gradient is read off `base`, the copy already blurred at the radius
    scale, for two reasons: it is free, and it is the right signal. A gradient
    measured on the raw pixels of a noisy sky is not small — shot noise has a
    gradient everywhere — so a mask built from it would open exactly over the
    grain it is supposed to protect. At the radius scale a step edge still
    reads ~0.33 luma/px while noise of the same amplitude reads under 0.01, and
    that gap is the whole mask.

    Then dilated by the same neighbourhood the clamp uses and blurred, so the
    mask covers the shoulders of an edge and not just its crest. A mask with
    hard sides would print its own outline into the frame.
    """
    gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
    gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3, scale=0.125)
    grad = cv2.magnitude(gx, gy)
    hi = _MASK_GRAD_MAX * (masking / 100.0)
    lo = hi * _MASK_SOFT
    mask = _smoothstep(np.clip((grad - lo) / max(hi - lo, _EPS), 0.0, 1.0))
    mask = cv2.dilate(mask, morph)
    return cv2.GaussianBlur(mask, (0, 0), max(0.6, _MASK_BLUR_FRAC * sigma_r))


def apply_sharpen(rgb: np.ndarray, params: dict[str, Any],
                  frame_long: float) -> np.ndarray:
    """Sharpen `rgb` (float32 RGB in [0,1]) and return a new array.

    The result may sit slightly outside [0,1] — an overshoot at a white edge is
    real and the caller clips once at the end, like every other stage here.

    `frame_long` is the long edge of the whole photo in this array's pixels. It
    is validated and then not used: see the module docstring on why this one
    stage is measured in pixels rather than in fractions of the frame.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected HxWx3 RGB, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"expected float32, got {rgb.dtype}"
    assert frame_long > 0.0, f"frame_long must be positive, got {frame_long}"
    p = {**DEFAULT_SHARPEN, **params}
    if is_neutral(p):
        return rgb

    amount = float(p["amount"]) / 100.0 * _AMOUNT_MAX
    detail = float(p["detail"]) / 100.0
    sigma_r, sigma_f = _sigmas(p)

    y = _luma(rgb)
    fine_base = cv2.GaussianBlur(y, (0, 0), sigma_f)
    base = cv2.GaussianBlur(y, (0, 0), sigma_r)

    # Two bands, not one: mid is the edge at the radius scale, fine is
    # everything below it. Weighting them separately is what makes `detail` a
    # frequency control instead of a second amount.
    w_fine = _FINE_WEIGHT[0] + (_FINE_WEIGHT[1] - _FINE_WEIGHT[0]) * detail
    delta = amount * ((fine_base - base) + w_fine * (y - fine_base))

    # Halo clamp. `morph` is the neighbourhood an edge is allowed to reach into;
    # 2 sigma covers the whole transition a blur at sigma_r produced.
    k = 2 * max(1, int(round(2.0 * sigma_r))) + 1
    morph = np.ones((k, k), np.uint8)
    local_hi = cv2.dilate(y, morph)
    local_lo = cv2.erode(y, morph)
    slack = (_OVERSHOOT_MAX * detail) * (local_hi - local_lo)
    room_up = local_hi - y + slack
    room_dn = y - local_lo + slack
    up = np.maximum(delta, 0.0)
    down = np.maximum(-delta, 0.0)
    # x*r/(r+x): equals x for small x, never reaches r, smooth and monotone
    # throughout. A hard clip at r would put a crease in a gradient.
    delta = (up * room_up / (room_up + up + _EPS)
             - down * room_dn / (room_dn + down + _EPS))

    if p["masking"] > 0:
        delta *= _edge_mask(base, sigma_r, int(p["masking"]), morph)

    return rgb + delta[..., None]


def padding(params: dict[str, Any] | None, frame_long: float) -> float:
    """Neighbourhood a window render needs, in pixels — mirrors
    `editing.effect_padding`'s `for_adj`, which is where this gets folded in.

    In absolute pixels, not a fraction of `frame_long`, for the same reason the
    radius is (and the same reason denoise's padding is). The chain reaches
    3 sigma for the base blur, 2 sigma more for the morphology, and 3 more mask
    blur sigmas on top of that when masking is on.
    """
    if is_neutral(params):
        return 0.0
    assert frame_long > 0.0, f"frame_long must be positive, got {frame_long}"
    p = {**DEFAULT_SHARPEN, **(params or {})}
    sigma_r, _ = _sigmas(p)
    reach = 3.0 * sigma_r + 2.0 * sigma_r
    if p["masking"] > 0:
        reach += 3.0 * _MASK_BLUR_FRAC * sigma_r
    return reach
