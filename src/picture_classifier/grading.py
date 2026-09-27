"""Colour grading: three-way wheels for shadows, midtones and highlights.

Each zone carries a hue, a saturation and a luminance, and every pixel is
graded by all three in proportion to how much of it belongs to each tonal
region. Two globals shape those regions: `blending` widens them until they
overlap, `balance` slides the split between shadow and highlight.

Design notes:
  - The whole stage is a function of one number, the pixel's luminance. Zone
    weights come from luminance, the tint added is a weighted sum of fixed
    vectors, and the luminance sliders compose into a per-channel gain and
    offset — so grading collapses into ONE table indexed by luminance, the same
    move `editing._wb_tone_lut` makes for tone, and for the same reason: the
    result is exact, and a table read costs a fraction of evaluating two
    smoothsteps and nine blends over every pixel of a 24 MP frame.
  - Adding colour must not move brightness. Every tint is a vector whose luma is
    zero (see `_tint_vector`), so `sat` changes hue and chroma while leaving
    `_LUMA . rgb` where it was, exactly, with no need to measure the luma and
    put it back afterwards.
  - The weights are a partition of unity by construction rather than by
    calibration (see `_zone_weights`), so no tone is graded twice and none is
    skipped, at any blending.
  - Pointwise, so nothing here has an opinion about resolution: one dict grades
    a 1000 px preview and a 6000 px export to the same values. No padding, no
    neighbourhood, no scale term.
  - Tinting a pixel that is already near white or black can push a channel out
    of [0,1]; the caller clips once at the end of the pipeline, as it does for
    the other colour stages. The `lum` sliders never leave the range on their
    own — they move towards white or black rather than scaling past either.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ----- schema -------------------------------------------------------------
#
# Stored sparsely, the way `editing.normalize_hsl` stores the colour mixer:
# only values that are off neutral survive, so a set of wheels nobody has
# dragged normalizes to None, never makes an edit look non-neutral, and keeps
# the edit hash short.

ZONES: tuple[str, ...] = ("shadows", "midtones", "highlights")
ZONE_KEYS: tuple[str, ...] = ("hue", "sat", "lum")

DEFAULT_GRADING: dict[str, Any] = {
    "shadows": {"hue": 0, "sat": 0, "lum": 0},
    "midtones": {"hue": 0, "sat": 0, "lum": 0},
    "highlights": {"hue": 0, "sat": 0, "lum": 0},
    "blending": 50,   # how far the three zones reach into each other
    "balance": 0,     # - hands the frame to the shadows, + to the highlights
}

# hue is an angle and wraps; the other two clamp.
_ZONE_RANGES: dict[str, tuple[float, float]] = {
    "sat": (0.0, 100.0),
    "lum": (-100.0, 100.0),
}
_GLOBAL_RANGES: dict[str, tuple[float, float]] = {
    "blending": (0.0, 100.0),
    "balance": (-100.0, 100.0),
}

_LUT_N = 8192   # luminance samples; see `_grading_lut` for why this many
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Where the zones sit on the tonal ramp. The two splits stay half the range
# apart, so a mid grey is entirely midtone at the default blending.
_SPLIT_LO = 0.25
_SPLIT_HI = 0.75
_HALF_MIN = 0.06        # half-width of a ramp at blending 0 (zones kept apart)
_HALF_MAX = 0.30        # ...and at 100, where all three reach the middle
_BALANCE_SHIFT = 0.20   # how far balance +/-100 slides both splits

_TINT_MAX = 0.22   # strongest channel excursion a sat of 100 may ask for
_LUM_MAX = 0.50    # how far towards white or black a lum of +/-100 goes

# Grading is memory-bound, not arithmetic-bound: run it in row bands so the
# luminance, the index and the gather stay in cache instead of making four
# round trips to main memory. Exact — the stage is pointwise — and worth about
# a third of the runtime on a 24 MP frame.
_TILE_ROWS = 128


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp a grading dict, dropping everything that changes nothing. Returns
    None when nothing is left.

    A zone with `sat` and `lum` both at zero is dropped whatever its hue: a hue
    on its own is a wheel position, not a change to the image, the same line
    `editing._MODIFIER_KEYS` draws for a motion angle with no motion. `hue` is
    likewise dropped when `sat` is zero, since there is then no tint for it to
    steer, and `blending`/`balance` are only stored when off their defaults.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {}
    for zone in ZONES:
        src = raw.get(zone)
        if not isinstance(src, dict):
            continue
        vals: dict[str, int] = {}
        for key, (lo, hi) in _ZONE_RANGES.items():
            val = int(round(_fnum(src.get(key), lo, hi, 0.0)))
            if val:
                vals[key] = val
        if not vals:
            continue
        # Wrapped, not clamped: 370 degrees is 10 degrees round the wheel.
        hue = int(round(_fnum(src.get("hue"), -1e6, 1e6, 0.0))) % 360
        if hue and "sat" in vals:
            vals["hue"] = hue
        out[zone] = vals
    if not out:
        return None
    for key, (lo, hi) in _GLOBAL_RANGES.items():
        default = DEFAULT_GRADING[key]
        val = int(round(_fnum(raw.get(key), lo, hi, float(default))))
        if val != default:
            out[key] = val
    return out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when the wheels would leave the pixels alone."""
    return normalize(params) is None


# ----- zone weights -------------------------------------------------------

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite ease on an already-clipped [0,1] ramp (as in `editing`)."""
    return t * t * (3.0 - 2.0 * t)


def _zone_weights(y: np.ndarray, blending: int = 50,
                  balance: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """How much of a pixel at luminance `y` belongs to each zone.

    Built from two monotone ramps that share one width — `a` handing over from
    shadows to midtones, `b` from midtones to highlights:

        shadows = 1 - a      midtones = a - b      highlights = b

    Three things fall out of writing it this way rather than as three separate
    windows. The weights sum to exactly 1 at every luminance, so no tone is
    graded twice and none is missed. `b` is `a` shifted right by a fixed half of
    the range, and a monotone function shifted right is never larger, so `a >= b`
    and the midtone weight cannot go negative however wide the ramps get. And
    the two controls become plain geometry: `blending` is the ramp width, so
    opening it lets the shadow and highlight zones reach into the middle while
    the midtone weight gives way, and `balance` slides both splits together, so
    a positive value hands more of the frame to the highlight wheel.

    Outside [0,1] the weights saturate, which is what a value that has drifted
    slightly out of range earlier in the pipeline should get.
    """
    half = _HALF_MIN + (blending / 100.0) * (_HALF_MAX - _HALF_MIN)
    shift = -(balance / 100.0) * _BALANCE_SHIFT
    span = 2.0 * half
    a = _smoothstep(np.clip((y - (_SPLIT_LO + shift)) / span + 0.5, 0.0, 1.0))
    b = _smoothstep(np.clip((y - (_SPLIT_HI + shift)) / span + 0.5, 0.0, 1.0))
    return 1.0 - a, a - b, b


# ----- the lookup ---------------------------------------------------------

def _tint_vector(hue: int) -> np.ndarray:
    """The chroma direction for a hue: its fully saturated colour, luma removed.

    Subtracting the luma is the whole of how `sat` avoids disturbing
    brightness. What is left points at the hue but has a luma of exactly zero,
    so adding any multiple of it moves the colour while `_LUMA . rgb` stays put
    — no measuring the old brightness and forcing it back, which is what leaves
    banding in the deep shadows where the ratio is ill-conditioned.

    Scaled so the largest channel is 1: every hue then pushes equally hard, the
    wheel has no strong and weak directions, and no hue can drive a channel
    further than `_TINT_MAX`.
    """
    hsv = np.array([[[float(hue % 360), 1.0, 1.0]]], dtype=np.float32)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    chroma = rgb - float(rgb @ _LUMA)
    return chroma / float(np.abs(chroma).max())


def _grading_lut(p: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray]:
    """Fold the three wheels into one table indexed by luminance.

    Every stage here is a per-channel affine map whose two coefficients depend
    on nothing but the pixel's luminance. A `lum` of k over a zone of weight w
    moves each channel towards white (`c + wk(1-c)`) or towards black
    (`c(1 + wk)`), which is `c * (1 - w|k|) + w*max(k,0)`; composing the three
    zones multiplies the gains and carries the offset through. A `sat` of s adds
    `w * s * v` for the zone's fixed chroma vector `v`. So the whole grade is

        out = gain(y) * rgb + offset(y)

    and sampling `gain` and `offset` on a luminance ramp loses nothing but the
    quantization of y. `_LUT_N` samples hold that to a quarter of an 8-bit level
    with every wheel at its limit and blending at 0, where the ramps are
    steepest — and a table this size stays in cache while the frame streams past
    it, so making it finer buys nothing measurable either way.

    Returns (gain, offset), with gain None when no zone moves luminance — then
    the grade is a pure offset and the multiply can be skipped entirely, which
    is the common case for a set of wheels used for colour.
    """
    y = np.linspace(0.0, 1.0, _LUT_N, dtype=np.float32)
    weights = _zone_weights(y, p.get("blending", DEFAULT_GRADING["blending"]),
                            p.get("balance", DEFAULT_GRADING["balance"]))
    gain = np.ones(_LUT_N, dtype=np.float32)
    bias = np.zeros(_LUT_N, dtype=np.float32)
    tint = np.zeros((_LUT_N, 3), dtype=np.float32)
    moves_luma = False
    for zone, w in zip(ZONES, weights):
        z = p.get(zone) or {}
        k = int(z.get("lum", 0)) / 100.0 * _LUM_MAX
        if k:
            moves_luma = True
            step = w * k
            factor = 1.0 - np.abs(step)
            gain *= factor
            bias = bias * factor + np.maximum(step, 0.0)
        sat = int(z.get("sat", 0))
        if sat:
            v = _tint_vector(int(z.get("hue", 0)))
            tint += (w * (sat / 100.0 * _TINT_MAX))[:, None] * v[None, :]
    offset = np.ascontiguousarray(bias[:, None] + tint, dtype=np.float32)
    return (gain if moves_luma else None), offset


def apply_grading(rgb: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Grade an RGB float32 image in [0,1]. Neutral params return it unchanged.

    The result may sit slightly outside [0,1] where a tint pushes a channel past
    white or black; the caller clips.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"want an RGB image, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"want float32 in [0,1], got {rgb.dtype}"
    p = normalize(params)
    if p is None:
        return rgb
    gain, offset = _grading_lut(p)
    out = np.empty_like(rgb)
    for r0 in range(0, rgb.shape[0], _TILE_ROWS):
        src = rgb[r0:r0 + _TILE_ROWS]
        dst = out[r0:r0 + _TILE_ROWS]
        # The weights read the luminance the pixel arrived with, so a lum slider
        # cannot walk a pixel out of its own zone as it brightens it.
        y = cv2.transform(src, _LUMA.reshape(1, 3))    # src @ _LUMA, quicker
        idx = (np.clip(y, 0.0, 1.0) * (_LUT_N - 1)).astype(np.int32)
        np.take(offset, idx, axis=0, out=dst)
        if gain is None:
            dst += src
        else:
            dst += src * np.take(gain, idx)[..., None]
    return out
