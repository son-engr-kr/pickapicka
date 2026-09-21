"""Colour-range and luminance-range masks: selections made of pixel *values*.

The masks in `editing` are geometry — an ellipse, a gradient, a painted stroke.
These two are the other half of Lightroom's mask system: they select by what the
pixels *are*, so "every mid-tone", or "everything that is this shade of blue",
is one click instead of an afternoon of brushwork.

Both produce exactly what `editing._radial_alpha` and friends produce — a
float32 alpha in [0,1] with the image's height and width — so the caller can
intersect one with a geometric alpha (a range mask refining a radial, the way
Lightroom does it) or use it standalone.

Only numpy + OpenCV, and deliberately no import of `editing`: that module will
import this one to wire the masks in, and a cycle would be the thanks we get.
`_smoothstep` and the work-size cap are therefore duplicated here, and
`RANGE_WORK_EDGE` is kept equal to `editing._MASK_WORK_EDGE` on purpose — both
alphas then live on the same grid and intersect without a resample.

Design notes:
  - Luminance is CIE L*, not a channel average: the slider then reads in
    perceptual lightness, so 50 is the mid-grey a photographer means by
    mid-grey, and the same conversion feeds the colour path.
  - The colour metric is a hue-dominant weighted Lab distance (see
    `_color_distance`). Plain RGB distance, and even plain Lab ΔE, call a dark
    red and a bright red different colours while calling a dark red and black
    the same one; that is the wrong answer for a *colour* selection.
  - The selector is built at a capped working resolution and the alpha
    upscaled. For geometry that is only a speed trick; here it decides what
    gets selected, so it gets its own argument under "working resolution".
  - The selector is blurred before it is thresholded, never the image: a range
    mask on a noisy frame otherwise dissolves into speckle, because sensor
    noise straddles the threshold pixel by pixel.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# ----- schema -------------------------------------------------------------
#
# Two parameter dicts, both neutral where they select the whole frame:
#
#   luminance  lo, hi          the L* window that is selected (0..100)
#              feather_lo      width in L* of the ramp below `lo`
#              feather_hi      width in L* of the ramp above `hi`
#
#   colour     samples         up to COLOR_SAMPLES_MAX eyedropped [r, g, b]
#              range           how far from a sample still counts (0..100)
#              feather         softness of that boundary (0..100)

LUMA_DEFAULT: dict[str, Any] = {"lo": 0, "hi": 100, "feather_lo": 10, "feather_hi": 10}
COLOR_DEFAULT: dict[str, Any] = {"samples": [], "range": 40, "feather": 25}

COLOR_SAMPLES_MAX = 5      # Lightroom's limit, and five is already plenty
MIN_LUMA_WINDOW = 1        # in L*; a zero-width window is not a selection
_EPS = 1e-4

# Even at feather 0 the boundary gets a hair of ramp. A true step on a value
# selection is an 8-bit contour: the source is quantized, so a hard threshold
# draws the quantization as a visible edge in the alpha.
_FEATHER_FLOOR = 0.5       # L* units
_COLOR_BAND_FLOOR = 1.0    # metric units

# `range` -> distance threshold in the metric's own units. Squared, because the
# interesting decisions all live at the tight end: 0..50 walks out from "this
# exact colour" to "this hue family", and the top half is for grabbing slop.
_COLOR_RANGE_MAX = 120.0   # a shade over the widest distance sRGB can produce,
                           # so range=100 really does select everything


# ----- working resolution -------------------------------------------------

# Both selectors are built with the frame downscaled to RANGE_WORK_EDGE on its
# long edge, and the alpha is upscaled back. For a geometric mask that is purely
# about cost. For a range mask it is a decision about *what gets selected*, so it
# needs its own argument:
#
#   1. Unlike geometry, a range mask changes with the resolution it is measured
#      at. Downscaling with area averaging low-passes the image, so fine detail
#      stops being selectable.
#
#   2. That is why the cap is not optional. Measured natively, the same `range`
#      value would select one thing on the 1500 px editor preview and another on
#      the 24 MP export, because at 24 MP the threshold sees per-pixel noise and
#      texture that the preview had already averaged away. The user tunes the
#      slider against the preview; a mask that then exports differently is not a
#      mask, it is a surprise. One fixed grid makes the selection identical for
#      every render at or above the cap.
#
#   3. It also *is* the noise robustness. Area averaging 24 MP down to 0.7 MP
#      divides sensor noise by about six before anything is thresholded, which no
#      amount of post-hoc smoothing of a speckled alpha would recover.
#
#   4. The price is real and worth stating: selection detail finer than the
#      working grid is gone. A colour range cannot pick single leaves out of a
#      sky or catch a one-pixel specular highlight. It is the right trade for a
#      slider-driven selection, and the caller intersects the result with a
#      geometric alpha built on the same grid anyway, so a sharper edge here
#      would only be resampled away there.
#
#   5. Below the cap the working size is the image size, so a 512 px thumbnail
#      selects a little more softly than the export does. Thumbnails are not
#      where a mask is judged, and forcing them up to the cap would only
#      interpolate detail that is not in the pixels.
RANGE_WORK_EDGE = 1024     # equal to editing._MASK_WORK_EDGE, deliberately

# Selector blur, as a fraction of the working long edge so a mask means the
# same thing at every render size. Two work pixels at the cap: it drops the
# alpha's pixel-to-pixel variation on a noisy flat field by about 25x, and
# spreads a hard selection boundary over ~5 work pixels, which is the same
# order as the INTER_LINEAR upscale of the alpha spreads it anyway.
#
# Rejected: cv2.bilateralFilter, the obvious edge-preserving choice. It is 4-12x
# slower and *worse* on noise (std 0.12 vs 0.04 in the same test), for the
# reason it exists: shot noise looks like an edge to it, so it carefully keeps
# it. Rejected: medianBlur, cheap and genuinely edge-preserving, but its output
# is piecewise constant, which is the one thing a smooth ramp must not be built
# on top of.
_SELECTOR_SIGMA_FRAC = 2.0 / RANGE_WORK_EDGE
_SELECTOR_SIGMA_FLOOR = 0.8


# ----- colour metric ------------------------------------------------------
#
# What a "colour selection" has to get right, and what the obvious metrics do:
#
#                                       RGB      Lab ΔE76   this
#   dark red  vs  bright red   (near)   194        62        19
#   dark red  vs  dark blue    (far)    113        68        34
#   dark red  vs  black        (far)     80        43        62
#
# RGB is hopeless: it puts a dark red nearer to black than to a bright red,
# because in RGB "red" mostly means "large first coordinate" and brightness
# swamps hue. Lab ΔE76 fixes the ordering of the first two but still ranks
# black closest of the three, because absolute chroma shrinks with lightness
# (dark red C*=41, bright red C*=85), so a dark saturated colour genuinely
# sits near the neutral axis in Lab.
#
# So: split the Lab chromatic difference into its chroma and hue parts the way
# CIE94 does, and weight the three terms by how much each one has to do with
# being the same colour.
#
#   dH  hue, the identity of a colour, weighted heaviest. Taken as CIE94's
#       ΔH*_ab, which is Δa,Δb with the chroma component removed; it scales
#       with sqrt(C1*C2) and so vanishes for near-neutral pixels. That is what
#       keeps a grey wall from speckling: hue angle is meaningless there, and
#       this formulation stops asking.
#   ΔS  saturation, as C*/(C* + C0) rather than raw chroma. Relative, so a
#       dark red and a bright red read as the same strength of colour, while
#       black still reads as no colour at all. C0 = 25 is roughly where sRGB
#       chroma stops being scarce.
#   ΔL  lightness, divided by _K_LIGHT. It cannot be dropped (white and black
#       are both hueless and both unsaturated, and they are not the same
#       colour) but it must stay small, or the metric turns back into the
#       brightness-dominated thing we are trying to escape. Brightness is what
#       the luminance-range mask is for.

_K_LIGHT = 4.0             # divisor on ΔL*
_K_HUE = 2.0               # divisor on ΔH*_ab
_SAT_HALF = 25.0           # C* at which relative saturation reads 50


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite ease on an already-clipped [0,1] ramp (same as editing's)."""
    return t * t * (3.0 - 2.0 * t)


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _inum(raw: Any, lo: float, hi: float, fallback: float) -> int:
    return int(round(_fnum(raw, lo, hi, fallback)))


# ----- normalization ------------------------------------------------------

def normalize_luma(raw: Any) -> dict[str, int] | None:
    """Clamp a luminance-range dict, or None when it selects every tone (which
    is not a mask at all and must not make an edit look non-neutral — the same
    reasoning as `editing.normalize_crop`)."""
    if not isinstance(raw, dict):
        return None
    lo = _fnum(raw.get("lo"), 0, 100, 0)
    hi = _fnum(raw.get("hi"), 0, 100, 100)
    if hi < lo:
        lo, hi = hi, lo
    lo = min(lo, 100 - MIN_LUMA_WINDOW)
    hi = max(hi, lo + MIN_LUMA_WINDOW)
    if lo <= _EPS and hi >= 100 - _EPS:
        return None
    return {
        "lo": int(round(lo)),
        "hi": int(round(hi)),
        "feather_lo": _inum(raw.get("feather_lo"), 0, 100, 10),
        "feather_hi": _inum(raw.get("feather_hi"), 0, 100, 10),
    }


def _normalize_samples(raw: Any) -> list[list[int]]:
    """Eyedropped colours as 8-bit RGB triples. Malformed entries are dropped,
    the way `editing._normalize_strokes` drops pointless strokes."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[list[int]] = []
    for item in raw:
        if len(out) >= COLOR_SAMPLES_MAX:
            break
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        out.append([_inum(c, 0, 255, 0) for c in item])
    return out


def normalize_color(raw: Any) -> dict[str, Any] | None:
    """Clamp a colour-range dict, or None when it is not a selection: nothing
    sampled, or a range wide enough to take the whole gamut."""
    if not isinstance(raw, dict):
        return None
    samples = _normalize_samples(raw.get("samples"))
    if not samples:
        return None
    rng = _inum(raw.get("range"), 0, 100, 40)
    if rng >= 100:
        return None
    return {
        "samples": samples,
        "range": rng,
        "feather": _inum(raw.get("feather"), 0, 100, 25),
    }


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when these parameters select everything, i.e. are not a mask."""
    if params is None:
        return True
    assert isinstance(params, dict), f"not a range-mask dict: {type(params)}"
    if "samples" in params:
        return normalize_color(params) is None
    assert "lo" in params or "hi" in params, f"unrecognized range params: {sorted(params)}"
    return normalize_luma(params) is None


# ----- the selector -------------------------------------------------------

def _work_size(h: int, w: int) -> tuple[int, int]:
    """The image size, capped at RANGE_WORK_EDGE on the long edge."""
    scale = RANGE_WORK_EDGE / max(h, w)
    if scale >= 1.0:
        return h, w
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _lab_selector(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab of the frame at the working resolution, pre-smoothed.

    INTER_AREA is the low-pass half of `_RESOLUTION_NOTE`; the Gaussian is the
    rest of it. Smoothing the *selector* and not the image is the whole point:
    the pixels the mask grades stay sharp, only the decision about which ones
    to grade is denoised.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected HxWx3 RGB, got {rgb.shape}"
    assert rgb.dtype in (np.uint8, np.float32), f"unsupported dtype {rgb.dtype}"
    h, w = rgb.shape[:2]
    sh, sw = _work_size(h, w)
    small = rgb if (sh, sw) == (h, w) else cv2.resize(rgb, (sw, sh),
                                                      interpolation=cv2.INTER_AREA)
    # cvtColor wants float32 in [0,1] to hand back L* in [0,100] and a,b in
    # [-127,127]; the 8-bit path would quantize L* to 255 steps for nothing.
    src = small.astype(np.float32) / 255.0 if small.dtype == np.uint8 else small
    # Float pixels on a 0..255 scale would silently convert to nonsense Lab and
    # select nothing. Checked on the reduced copy, so it costs a fraction of a ms.
    assert src.max() <= 1.0 + 1e-3, f"float RGB must be in [0,1], saw {src.max()}"
    lab = cv2.cvtColor(np.ascontiguousarray(src), cv2.COLOR_RGB2Lab)
    sigma = max(_SELECTOR_SIGMA_FLOOR, _SELECTOR_SIGMA_FRAC * max(sh, sw))
    return cv2.GaussianBlur(lab, (0, 0), sigma)


def _to_output(alpha: np.ndarray, h: int, w: int) -> np.ndarray:
    """Upscale a work-resolution alpha to h x w, float32 and writeable."""
    if alpha.shape[:2] != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(alpha, dtype=np.float32)


# ----- luminance range ----------------------------------------------------

def luma_alpha(rgb: np.ndarray, params: dict[str, Any],
               out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Alpha selecting the L* window in `params`: float32 in [0,1], shaped like
    `rgb` unless `out_hw` asks for another size (a caller that wants the alpha
    on the mask grid should pass it, and skip a round trip through full size).

    1 across [lo, hi], easing to 0 over `feather_lo` below and `feather_hi`
    above. The two ends are combined with a minimum, so a window narrower than
    its feathers peaks below 1 instead of double-counting the overlap.
    """
    for key in ("lo", "hi", "feather_lo", "feather_hi"):
        assert key in params, f"luma params missing {key}; normalize first"
    lo, hi = float(params["lo"]), float(params["hi"])
    assert hi > lo, f"empty luminance window [{lo}, {hi}]"
    flo = max(_FEATHER_FLOOR, float(params["feather_lo"]))
    fhi = max(_FEATHER_FLOOR, float(params["feather_hi"]))

    lum = _lab_selector(rgb)[..., 0]
    rise = _smoothstep(np.clip((lum - (lo - flo)) / flo, 0.0, 1.0))
    fall = _smoothstep(np.clip(((hi + fhi) - lum) / fhi, 0.0, 1.0))
    alpha = np.minimum(rise, fall)
    h, w = out_hw if out_hw is not None else rgb.shape[:2]
    return _to_output(alpha, h, w)


# ----- colour range -------------------------------------------------------

def _sat(chroma: np.ndarray) -> np.ndarray:
    """Chroma as a bounded relative saturation in [0,100]. See the metric note:
    absolute C* shrinks with lightness, so raw chroma would file a dark red
    next to black."""
    return 100.0 * chroma / (chroma + _SAT_HALF)


def _lab_parts(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    return lab[..., 0], a, b, chroma, _sat(chroma)


def _color_distance(lab: np.ndarray, sample_lab: np.ndarray) -> np.ndarray:
    """Squared hue-dominant Lab distance from every pixel of `lab` to one
    sampled Lab triple.

    Δa,Δb is split into its chroma and hue parts by the CIE94 identity
    Δa² + Δb² = ΔC² + ΔH², which costs one subtraction instead of two atan2s
    over the frame, and gives ΔH the sqrt(C1*C2) scaling that makes it fade out
    on neutrals.
    """
    lum, a, b, chroma, sat = _lab_parts(lab)
    slum, sa, sb, schroma, ssat = _lab_parts(sample_lab.reshape(1, 1, 3))
    d_ab2 = (a - sa) ** 2 + (b - sb) ** 2
    d_chroma = chroma - schroma
    d_hue2 = np.maximum(0.0, d_ab2 - d_chroma * d_chroma)
    return (((lum - slum) / _K_LIGHT) ** 2
            + (sat - ssat) ** 2
            + d_hue2 / (_K_HUE * _K_HUE))


def color_alpha(rgb: np.ndarray, params: dict[str, Any],
                out_hw: tuple[int, int] | None = None) -> np.ndarray:
    """Alpha selecting the sampled colours in `params`: float32 in [0,1],
    shaped like `rgb` unless `out_hw` asks for another size.

    Several samples union by keeping the nearest one per pixel — the best match
    across the set, which is what "add another colour to the selection" means.
    The ramp is centred on the `range` threshold so that feather changes how
    soft the boundary is without moving it.
    """
    assert "samples" in params and params["samples"], "colour params have no samples"
    for key in ("range", "feather"):
        assert key in params, f"colour params missing {key}; normalize first"
    thresh = (float(params["range"]) / 100.0) ** 2 * _COLOR_RANGE_MAX
    band = max(_COLOR_BAND_FLOOR, thresh * float(params["feather"]) / 100.0)

    lab = _lab_selector(rgb)
    samples = np.asarray(params["samples"], dtype=np.float32).reshape(-1, 1, 3) / 255.0
    sample_lab = cv2.cvtColor(np.ascontiguousarray(samples), cv2.COLOR_RGB2Lab)

    best = _color_distance(lab, sample_lab[0])
    for i in range(1, len(sample_lab)):
        np.minimum(best, _color_distance(lab, sample_lab[i]), out=best)
    dist = np.sqrt(best, out=best)
    alpha = 1.0 - _smoothstep(np.clip((dist - (thresh - band / 2.0)) / band, 0.0, 1.0))
    h, w = out_hw if out_hw is not None else rgb.shape[:2]
    return _to_output(alpha, h, w)
