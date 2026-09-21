"""Non-destructive photo grading: a small Lightroom-style edit pipeline.

An `edit` is a plain dict of scalar sliders, a master tone curve, and a list of
local-adjustment masks. `render` applies it to an RGB uint8 array; every field is
a no-op at its neutral value, so a neutral edit returns the input unchanged (the
same contract as `hdr.grade`).

Only numpy + OpenCV are used (both already required) — no scipy, no new heavy
dependency for the math. RAW decode helpers (rawpy) live in `raw.py` so this
module stays import-light and unit-testable without rawpy installed.

Design notes:
  - Work in float32 [0,1] internally; clip once at the end (`np.rint`, not a
    truncating cast, to avoid a systematic downward bias).
  - White balance, exposure, the four tone regions, global contrast and the
    master curve are all pointwise and per-channel, so they collapse into ONE
    256-entry lookup applied with `cv2.LUT` to the 8-bit source. That is the
    difference between ~340 ms and ~1 ms on a 2048 px frame, and it is exact.
    A mask grading an already-graded crop converts it back to 8 bits to take
    the same path; the round trip costs at most a level or two out of 255.
  - White balance runs before the tone LUT (correct the light before making
    luminance decisions); colour (vibrance/saturation) after tone; sharpen then
    vignette last.
  - Masks run after the global pass, each blending a locally graded copy through
    its alpha. Only the mask's bounding box is graded, and the alpha itself is
    built at a capped resolution, so cost stays flat in megapixels.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

from . import film as film_mod
from . import grading as grading_mod
from . import healing as healing_mod
from . import rangemask as rangemask_mod
from . import redeye as redeye_mod
from . import segment as segment_mod
from . import sharpening as sharpening_mod
from . import watermark as watermark_mod

# ----- schema -------------------------------------------------------------

# Every slider is neutral at 0; the curve is neutral as the identity diagonal.
DEFAULT_EDIT: dict[str, Any] = {
    "exposure": 0.0,   # EV stops
    "contrast": 0,
    "highlights": 0,
    "shadows": 0,
    "whites": 0,
    "blacks": 0,
    "temp": 0,         # warm (+) / cool (-)
    "tint": 0,         # magenta (+) / green (-)
    "vibrance": 0,
    "saturation": 0,
    "clarity": 0,
    "texture": 0,      # fine detail; - smooths skin, + finds pores and fabric
    "dehaze": 0,       # - puts atmosphere back, + cuts through it
    "denoise": 0,      # 0..100 (one-sided); the one stage judged at 1:1
    # Sharpening, four sliders. `sharpen` keeps its name and its 0..100 meaning
    # so an edit saved before the other three existed loads with its number in
    # the same place; see the note above _apply_sharpen about what did change.
    "sharpen": 0,           # amount, 0..100 (one-sided)
    "sharpen_radius": 50,   # 0.5..3.0 px of detail scale; neutral at the middle
    "sharpen_detail": 25,   # how much of the finest structure is admitted
    "sharpen_masking": 0,   # confine it to edges; 0 sharpens everything
    "blur": 0,         # defocus / bokeh, 0..100
    "motion": 0,       # directional blur amount, 0..100
    "motion_angle": 0, # its direction in degrees
    "glow": 0,         # highlight bloom, 0..100
    "pixelate": 0,     # mosaic block size, 0..100
    "vignette": 0,
    "tilt": 0.0,       # straighten, degrees; + levels a horizon drooping right
    "crop": None,      # {x, y, w, h} of the straightened frame; see "geometry"
    "curve": [[0.0, 0.0], [1.0, 1.0]],
    # Per-channel point curves, applied after the master one. See CURVE_KEYS.
    "curve_r": [[0.0, 0.0], [1.0, 1.0]],
    "curve_g": [[0.0, 0.0], [1.0, 1.0]],
    "curve_b": [[0.0, 0.0], [1.0, 1.0]],
    "hsl": None,       # colour mixer; see the "colour mixer" section below
    "grading": None,   # three-way colour wheels; see the grading module
    # Repairs, applied before anything tonal. See `_repair`.
    "healing": None,   # heal / clone / spot; see the healing module
    "redeye": None,    # red-eye and pet-eye; see the redeye module
    "masks": [],       # local adjustments; see the "masks" section below
    "film": None,      # film emulation chain; see the film module
    "watermark": None, # signature / shooting info; see the watermark module
}

# (min, max) clamp for each scalar. Exposure is the only float.
_RANGES: dict[str, tuple[float, float]] = {
    "exposure": (-2.0, 2.0),
    "contrast": (-100, 100),
    "highlights": (-100, 100),
    "shadows": (-100, 100),
    "whites": (-100, 100),
    "blacks": (-100, 100),
    "temp": (-100, 100),
    "tint": (-100, 100),
    "vibrance": (-100, 100),
    "saturation": (-100, 100),
    "clarity": (-100, 100),
    "texture": (-100, 100),
    "dehaze": (-100, 100),
    "denoise": (0, 100),
    "sharpen": (0, 100),
    "sharpen_radius": (0, 100),
    "sharpen_detail": (0, 100),
    "sharpen_masking": (0, 100),
    "blur": (0, 100),
    "motion": (0, 100),
    "motion_angle": (-180, 180),
    "glow": (0, 100),
    "pixelate": (0, 100),
    "vignette": (-100, 100),
    "tilt": (-45.0, 45.0),
}

# Kept out of the integer rounding in `normalize`: a tenth of a degree is the
# difference between a level horizon and an almost-level one.
_FLOAT_KEYS = frozenset({"exposure", "tilt"})

_EPS = 1e-4
_LUT_N = 1024  # tone-LUT sample count

# The master curve and the three per-channel ones. All four are the same shape —
# a list of monotone control points — and all four fold into the single
# per-channel lookup that `_wb_tone_lut` builds, so a channel curve costs nothing
# at render time beyond the table it is baked into. The master runs first: it
# says how bright a tone is, and the channel curves then say what colour it takes
# there, which is the order that makes a split-toned shadow behave.
CURVE_KEYS: tuple[str, ...] = ("curve", "curve_r", "curve_g", "curve_b")
_CHANNEL_CURVES: tuple[str, ...] = ("curve_r", "curve_g", "curve_b")

# (x0, y0, w, h) in normalized frame coordinates. Anything other than this means
# the array being graded is a window onto a larger photo — see `render`.
FULL_ROI = (0.0, 0.0, 1.0, 1.0)


# ----- geometry: straighten and crop --------------------------------------
#
# Two settings, applied in this order and before anything tonal:
#
#   tilt   degrees to turn the frame by. The result is cut back to the largest
#          rectangle of the original aspect that still lies inside the turned
#          image, so straightening never leaves blank corners and never has to
#          invent pixels at the edges.
#   crop   {x, y, w, h} in normalized coordinates *of the straightened frame*,
#          so a crop keeps meaning if the tilt is nudged afterwards.
#
# Everything downstream — masks, the vignette, frame-relative effects, the
# watermark — then treats the cropped result as the whole photo. That is what
# keeps one code path serving the thumbnail, the fit preview, a 1:1 window and
# the export: geometry decides what the frame *is*, and the rest of the pipeline
# only ever sees a frame.

MIN_CROP = 0.02   # keep a crop big enough to still be a picture


def normalize_crop(raw: Any) -> dict[str, float] | None:
    """Clamp a crop to the frame, or None when it selects everything (which is
    not a crop at all and must not make an edit look non-neutral)."""
    if not isinstance(raw, dict):
        return None
    try:
        x, y = float(raw.get("x", 0.0)), float(raw.get("y", 0.0))
        w, h = float(raw.get("w", 1.0)), float(raw.get("h", 1.0))
    except (TypeError, ValueError):
        return None
    w = min(1.0, max(MIN_CROP, w))
    h = min(1.0, max(MIN_CROP, h))
    x = min(1.0 - w, max(0.0, x))
    y = min(1.0 - h, max(0.0, y))
    if w > 1.0 - _EPS and h > 1.0 - _EPS:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def _tilt_scale(w: float, h: float, deg: float) -> float:
    """How much of a `w`x`h` frame survives a turn of `deg`, as a fraction of
    each side, if the result must keep the aspect and hold no blank corner.

    A `w*t` x `h*t` rectangle sits inside the turned frame when both of its
    half-extents, measured back in the frame's own axes, still fit:
        t*(w*cos + h*sin) <= w   and   t*(w*sin + h*cos) <= h
    """
    c, s = abs(math.cos(math.radians(deg))), abs(math.sin(math.radians(deg)))
    return min(w / (w * c + h * s), h / (w * s + h * c))


def geometry_is_neutral(edit: dict[str, Any] | None) -> bool:
    """True when the frame comes out the same shape and size it went in."""
    e = normalize(edit)
    return abs(e["tilt"]) < _EPS and e["crop"] is None


def geometry_matrix(w: int, h: int,
                    edit: dict[str, Any] | None) -> tuple[np.ndarray, tuple[int, int]]:
    """The 2x3 affine taking a pixel of the original frame to its place in the
    straightened, cropped one, plus that frame's size.

    Everything geometric is derived from this one function — the whole-frame
    path, the windowed path, and the transform the client is told about — so the
    three cannot drift apart.
    """
    e = normalize(edit)
    m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ow, oh = int(w), int(h)
    if abs(e["tilt"]) >= _EPS:
        m = np.asarray(cv2.getRotationMatrix2D((w / 2.0, h / 2.0), e["tilt"], 1.0))
        t = _tilt_scale(w, h, e["tilt"])
        ow, oh = max(1, int(round(w * t))), max(1, int(round(h * t)))
        m = m.copy()
        m[0, 2] -= (w - ow) // 2
        m[1, 2] -= (h - oh) // 2
    crop = e["crop"]
    if crop is not None:
        cw = max(1, int(round(ow * crop["w"])))
        ch = max(1, int(round(oh * crop["h"])))
        cx = min(int(round(ow * crop["x"])), ow - cw)
        cy = min(int(round(oh * crop["y"])), oh - ch)
        m = m.copy()
        m[0, 2] -= cx
        m[1, 2] -= cy
        ow, oh = cw, ch
    return m, (ow, oh)


def geometry_size(w: int, h: int, edit: dict[str, Any] | None) -> tuple[int, int]:
    """What the geometry stage will return for a `w`x`h` frame, without touching
    a pixel. The client lays the view out from this before the render lands."""
    return geometry_matrix(w, h, edit)[1]


def _is_translation(m: np.ndarray) -> bool:
    """True when the affine only shifts — a crop with no straightening, which can
    be taken as a slice instead of resampled."""
    return bool(np.allclose(m[:, :2], np.eye(2), atol=1e-9))


def apply_geometry(rgb: np.ndarray, edit: dict[str, Any] | None) -> np.ndarray:
    """Straighten and crop `rgb`, which must be a whole frame."""
    e = normalize(edit)
    if geometry_is_neutral(e):
        return rgb
    h, w = rgb.shape[:2]
    m, (ow, oh) = geometry_matrix(w, h, e)
    if _is_translation(m):
        x0, y0 = int(round(-m[0, 2])), int(round(-m[1, 2]))
        return np.ascontiguousarray(rgb[y0:y0 + oh, x0:x0 + ow])
    return cv2.warpAffine(rgb, m, (ow, oh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def geometry_source_box(w: int, h: int, edit: dict[str, Any] | None,
                        window: tuple[int, int, int, int],
                        pad: int = 0) -> tuple[int, int, int, int]:
    """Which axis-aligned patch of the *original* frame is needed to fill
    `window` (x, y, w, h, in output pixels) of the geometry-applied frame.

    This is what lets the 1:1 view grade in original coordinates without grading
    the whole frame: only the pixels the window can see, plus `pad` for the
    effects that reach into their neighbourhood.
    """
    m, _ = geometry_matrix(w, h, edit)
    inv = np.asarray(cv2.invertAffineTransform(m))
    wx, wy, ww, wh = window
    corners = np.array([[wx, wy], [wx + ww, wy], [wx, wy + wh], [wx + ww, wy + wh]],
                       dtype=np.float64)
    src = corners @ inv[:, :2].T + inv[:, 2]
    x0 = max(0, int(np.floor(src[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(src[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(src[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(src[:, 1].max())) + pad)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def geometry_window(patch: np.ndarray, patch_box: tuple[int, int, int, int],
                    w: int, h: int, edit: dict[str, Any] | None,
                    window: tuple[int, int, int, int]) -> np.ndarray:
    """Place an already-graded patch of the original frame into `window` of the
    geometry-applied frame. The counterpart of `geometry_source_box`."""
    m, _ = geometry_matrix(w, h, edit)
    bx, by = patch_box[0], patch_box[1]
    wx, wy, ww, wh = window
    a = m[:, :2]
    t = m[:, 2] + a @ np.array([float(bx), float(by)]) - np.array([float(wx), float(wy)])
    mm = np.hstack([a, t.reshape(2, 1)])
    if _is_translation(mm):
        x0, y0 = int(round(-mm[0, 2])), int(round(-mm[1, 2]))
        return np.ascontiguousarray(patch[y0:y0 + wh, x0:x0 + ww])
    return cv2.warpAffine(patch, mm, (ww, wh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def geometry_norm_matrix(w: int, h: int,
                         edit: dict[str, Any] | None) -> list[float]:
    """`geometry_matrix` expressed in normalized coordinates: it takes a point
    given as a fraction of the *original* frame to the same point as a fraction
    of the output frame. Six numbers, row-major.

    The editor draws mask guides with it. Masks are positioned against the
    original frame, so once a crop is on, a guide drawn straight onto the
    cropped preview would sit somewhere else entirely.
    """
    m, (ow, oh) = geometry_matrix(w, h, edit)
    return [m[0, 0] * w / ow, m[0, 1] * h / ow, m[0, 2] / ow,
            m[1, 0] * w / oh, m[1, 1] * h / oh, m[1, 2] / oh]


# ----- mask schema (local adjustments) ------------------------------------
#
# A mask is a shape plus its own slider set: `render` grades the whole frame
# with the global sliders, then blends a locally graded copy through the mask's
# alpha. Geometry is stored in normalized [0,1] image coordinates (x is a
# fraction of the width, y of the height) so one mask means the same framing on
# a thumbnail, on the editor preview and on the full-resolution export.
#
#   radial  ellipse: cx, cy, rx, ry, angle (degrees, applied in normalized space)
#   linear  gradient: full strength at (x1,y1), fading to nothing at (x2,y2)
#   brush   freehand strokes: [{radius, erase, points: [[x, y], ...]}, ...]
#
# Common to all three: `feather` softens the edge, `amount` scales the whole
# effect, `invert` flips inside/outside.

MASK_TYPES = ("radial", "linear", "brush", "auto", "range")

# An "auto" mask is the odd one out and the difference is worth stating up front.
# The other three are pure functions of geometry, which is why their alpha can be
# cached on the numbers that describe them. An automatic mask is a function of
# the *pixels*, and of two specific sets of pixels: the WHOLE frame, because a
# segmenter shown only a 1:1 window would find a different subject than the fit
# preview found, and the ORIGINAL frame, because a mask derived from graded
# pixels would crawl every time a slider moved.
#
# Neither of those is available where alphas are built — a window render only
# ever holds a window, already graded. So the field arrives from outside:
# `segment` computes it from the original whole frame, the caller caches it per
# photo, and everything here does is crop the window out and scale it. That also
# keeps onnxruntime out of this module's import path and leaves it testable
# without the model present.
#
# A "range" mask, and the range refinement any mask may carry, are the third
# kind of thing: also derived from the pixels, but cheaply and with pure numpy,
# so they are computed here from a whole-frame array the caller hands over as
# `src`. Two mechanisms rather than one, and the line between them is whether a
# model and a per-photo cache are involved.
#
# Both share one rule, and it is the important one: the selection is measured on
# the WHOLE frame at one fixed grid, never on the window being rendered. Measure
# a luminance range on a 1:1 window and it would select a different set of tones
# than the fit preview did, because the window's own histogram is not the
# frame's — and the slider was tuned against the preview.

# The sliders a mask may carry — everything except vignette (a frame-wide effect)
# and the master curve (kept global so the histogram stays readable).
LOCAL_KEYS: tuple[str, ...] = (
    "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    "temp", "tint", "vibrance", "saturation", "clarity", "texture",
    "dehaze", "denoise", "sharpen", "sharpen_radius", "sharpen_detail",
    "sharpen_masking",
    "blur", "motion", "motion_angle", "glow", "pixelate",
)

# Keys that describe *how* an effect looks rather than how much of it there is —
# on their own they change nothing, so they do not make a mask "active".
_MODIFIER_KEYS = frozenset({"motion_angle", "sharpen_radius", "sharpen_detail",
                            "sharpen_masking"})

MASK_MAX = 16              # masks per photo
STROKE_MAX = 400           # brush strokes per mask
STROKE_POINTS_MAX = 4000   # points per stroke
_MASK_WORK_EDGE = 1024     # long edge the alpha map is built at, then upscaled
_BRUSH_RADIUS_RANGE = (0.002, 0.5)  # as a fraction of the image width


def _default_mask(kind: str) -> dict[str, Any]:
    """A neutral mask of `kind`, centred and sized for a sane first drag."""
    m: dict[str, Any] = {
        "type": kind,
        "name": "",
        "enabled": True,
        "invert": False,
        "feather": 50,
        "amount": 100,
        "adj": _neutral_adj(),
        "range_luma": None,
        "range_color": None,
    }
    if kind == "radial":
        m.update(cx=0.5, cy=0.5, rx=0.25, ry=0.25, angle=0.0)
    elif kind == "linear":
        m.update(x1=0.5, y1=0.15, x2=0.5, y2=0.55, feather=100)
    elif kind == "auto":
        # Feather starts at zero here, unlike the drawn kinds: the model's own
        # edge is already soft in the right places — half-covered hair comes out
        # near 0.5 — and blurring that by default would throw the good edge away.
        m.update(group="subject", feather=0)
    elif kind == "range":
        # Nothing but the range selection: the base covers the whole frame, so
        # this kind is "everything that looks like this", with no shape at all.
        m.update(feather=0)
    else:
        m.update(strokes=[])
    return m


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _neutral_adj() -> dict[str, Any]:
    """A mask's sliders at their neutral values. Not all of those are zero —
    `sharpen_radius` is neutral at its midpoint — so this reads them off
    DEFAULT_EDIT rather than assuming."""
    return {k: DEFAULT_EDIT[k] for k in LOCAL_KEYS}


def _normalize_adj(raw: Any) -> dict[str, Any]:
    """Clamp a mask's slider dict to the local subset (missing keys stay neutral)."""
    out: dict[str, Any] = _neutral_adj()
    if not isinstance(raw, dict):
        return out
    for key in LOCAL_KEYS:
        if raw.get(key) is None:
            continue
        lo, hi = _RANGES[key]
        val = _fnum(raw[key], lo, hi, 0.0)
        out[key] = val if key == "exposure" else int(round(val))
    return out


def _normalize_strokes(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:STROKE_MAX]:
        if not isinstance(item, dict):
            continue
        pts: list[list[float]] = []
        for p in (item.get("points") or [])[:STROKE_POINTS_MAX]:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                pts.append([_fnum(p[0], -1.0, 2.0, 0.0), _fnum(p[1], -1.0, 2.0, 0.0)])
        if not pts:
            continue
        rlo, rhi = _BRUSH_RADIUS_RANGE
        out.append({
            "radius": _fnum(item.get("radius"), rlo, rhi, 0.05),
            "erase": bool(item.get("erase")),
            "points": pts,
        })
    return out


def normalize_mask(raw: Any) -> dict[str, Any] | None:
    """Coerce one mask dict into the canonical shape, or None when it is not a
    usable mask (unknown type, degenerate geometry, no brush strokes)."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("type")
    if kind not in MASK_TYPES:
        return None
    m = _default_mask(kind)
    name = raw.get("name")
    m["name"] = str(name)[:40] if isinstance(name, str) else ""
    m["enabled"] = bool(raw.get("enabled", True))
    m["invert"] = bool(raw.get("invert", False))
    m["feather"] = int(round(_fnum(raw.get("feather"), 0, 100, m["feather"])))
    m["amount"] = int(round(_fnum(raw.get("amount"), 0, 100, 100)))
    m["adj"] = _normalize_adj(raw.get("adj"))
    # Available on every kind, including "auto": "the subject, but only its
    # highlights" is one mask, and it is the reason this is a refinement rather
    # than a mask type of its own.
    m["range_luma"] = rangemask_mod.normalize_luma(raw.get("range_luma"))
    m["range_color"] = rangemask_mod.normalize_color(raw.get("range_color"))

    if kind == "radial":
        # Centres may sit off-frame (a corner ellipse), radii must stay positive.
        m["cx"] = _fnum(raw.get("cx"), -1.0, 2.0, 0.5)
        m["cy"] = _fnum(raw.get("cy"), -1.0, 2.0, 0.5)
        m["rx"] = _fnum(raw.get("rx"), 0.005, 3.0, 0.25)
        m["ry"] = _fnum(raw.get("ry"), 0.005, 3.0, 0.25)
        m["angle"] = _fnum(raw.get("angle"), -180.0, 180.0, 0.0)
    elif kind == "linear":
        m["x1"] = _fnum(raw.get("x1"), -1.0, 2.0, 0.5)
        m["y1"] = _fnum(raw.get("y1"), -1.0, 2.0, 0.15)
        m["x2"] = _fnum(raw.get("x2"), -1.0, 2.0, 0.5)
        m["y2"] = _fnum(raw.get("y2"), -1.0, 2.0, 0.55)
        if abs(m["x2"] - m["x1"]) < 1e-4 and abs(m["y2"] - m["y1"]) < 1e-4:
            return None  # zero-length gradient has no direction
    elif kind == "auto":
        group = raw.get("group")
        if group not in segment_mod.CLASS_GROUPS:
            return None  # a group the model cannot produce selects nothing
        m["group"] = group
    elif kind == "range":
        if m["range_luma"] is None and m["range_color"] is None:
            return None  # a range mask with no range selects everything
    elif kind == "brush":
        m["strokes"] = _normalize_strokes(raw.get("strokes"))
        if not m["strokes"]:
            return None
    else:
        raise AssertionError(f"unhandled mask type {kind!r}")
    return m


def _adj_is_neutral(adj: dict[str, Any]) -> bool:
    """True when a mask's sliders would leave its pixels alone.

    Measured against DEFAULT_EDIT rather than against zero: `sharpen_radius` is
    neutral at its midpoint, so a mask holding the default 50 there is doing
    nothing and must not count as active. The modifier keys are excluded anyway
    — they describe how an effect looks, not whether one was asked for — but
    reading the neutral off DEFAULT_EDIT is what keeps this correct the next time
    a slider arrives whose neutral is not zero.
    """
    return all(abs(float(adj.get(k, DEFAULT_EDIT[k]) or 0)
                   - float(DEFAULT_EDIT[k])) < _EPS
               for k in LOCAL_KEYS if k not in _MODIFIER_KEYS)


def mask_is_active(mask: dict[str, Any]) -> bool:
    """True when a normalized mask actually changes pixels."""
    return bool(mask["enabled"]) and mask["amount"] > 0 and not _adj_is_neutral(mask["adj"])


# ----- normalization ------------------------------------------------------

def _clean_curve(raw: Any) -> list[list[float]]:
    """Coerce a control-point list to sorted [x, y] floats in [0,1] with x=0 and
    x=1 endpoints present. Non-increasing x values are dropped (kept monotone)."""
    pts: list[tuple[float, float]] = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    x = min(1.0, max(0.0, float(item[0])))
                    y = min(1.0, max(0.0, float(item[1])))
                except (TypeError, ValueError):
                    continue
                pts.append((x, y))
    pts.sort(key=lambda p: p[0])

    xs: list[float] = []
    ys: list[float] = []
    for x, y in pts:
        if xs and x <= xs[-1] + 1e-6:
            continue  # dedupe / enforce strictly increasing x
        xs.append(x)
        ys.append(y)
    if not xs:
        return [[0.0, 0.0], [1.0, 1.0]]
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, ys[0])
    if xs[-1] < 1.0:
        xs.append(1.0)
        ys.append(ys[-1])
    return [[x, y] for x, y in zip(xs, ys)]


def normalize(edit: dict[str, Any] | None) -> dict[str, Any]:
    """Merge `edit` over DEFAULT_EDIT, clamp every scalar to its range and repair
    the curve. Always returns a full dict with all keys present."""
    out = dict(DEFAULT_EDIT)
    for key in CURVE_KEYS:
        out[key] = [list(p) for p in DEFAULT_EDIT[key]]
    out["masks"] = []
    out["watermark"] = None
    out["crop"] = None
    out["film"] = None
    out["hsl"] = None
    out["grading"] = None
    out["healing"] = None
    out["redeye"] = None
    if not edit:
        return out
    if edit.get("healing") is not None:
        out["healing"] = healing_mod.normalize(edit["healing"])
    if edit.get("redeye") is not None:
        out["redeye"] = redeye_mod.normalize(edit["redeye"])
    if edit.get("hsl") is not None:
        out["hsl"] = normalize_hsl(edit["hsl"])
    if edit.get("grading") is not None:
        out["grading"] = grading_mod.normalize(edit["grading"])
    if edit.get("film") is not None:
        out["film"] = film_mod.normalize(edit["film"])
    if edit.get("crop") is not None:
        out["crop"] = normalize_crop(edit["crop"])
    for key, (lo, hi) in _RANGES.items():
        if key not in edit or edit[key] is None:
            continue
        try:
            val = float(edit[key])
        except (TypeError, ValueError):
            continue
        val = min(hi, max(lo, val))
        out[key] = val if key in _FLOAT_KEYS else int(round(val))
    for key in CURVE_KEYS:
        if key in edit:
            out[key] = _clean_curve(edit[key])
    if isinstance(edit.get("masks"), (list, tuple)):
        masks = (normalize_mask(m) for m in edit["masks"][:MASK_MAX])
        out["masks"] = [m for m in masks if m is not None]
    if edit.get("watermark") is not None:
        wm = watermark_mod.normalize(edit["watermark"])
        # Only keep one that would actually print something, so a toggled-off
        # watermark never leaves an edit looking non-neutral.
        out["watermark"] = None if watermark_mod.is_neutral(wm) else wm
    return out


def _curve_is_identity(curve: list[list[float]]) -> bool:
    return all(abs(y - x) < _EPS for x, y in curve)


def merge_additive(base: dict[str, Any] | None,
                   overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Lay `overlay` on top of `base` instead of replacing it.

    This is what makes local presets composable: a preset holding a "blur the
    background" mask can be dropped onto a photo that already has a grade and
    two masks of its own, and all of it survives. The rules:

      - a slider the overlay leaves at neutral keeps the base's value, so a
        mask-only preset never silently flattens someone's exposure;
      - masks are appended, not swapped, up to MASK_MAX;
      - the curve and the watermark come from the overlay only when it actually
        carries one.
    """
    out = normalize(base)
    over = normalize(overlay)
    for key in _RANGES:
        if abs(float(over[key]) - float(DEFAULT_EDIT[key])) > _EPS:
            out[key] = over[key]
    for key in CURVE_KEYS:
        if not _curve_is_identity(over[key]):
            out[key] = [list(p) for p in over[key]]
    if over["watermark"] is not None:
        out["watermark"] = over["watermark"]
    if over["crop"] is not None:
        out["crop"] = dict(over["crop"])   # an aspect preset is worth carrying
    if over["film"] is not None:
        out["film"] = dict(over["film"])
    # Band by band, for the same reason the sliders merge one at a time: a
    # preset that only cools the blues must not wipe someone's reds.
    if over["hsl"] is not None:
        merged = {b: dict(v) for b, v in (out["hsl"] or {}).items()}
        for band, vals in over["hsl"].items():
            merged.setdefault(band, {}).update(vals)
        out["hsl"] = merged or None
    if over["grading"] is not None:
        # Zone by zone and key by key, for the same reason the mixer merges that
        # way: a preset that only warms the highlights must not clear the rest.
        merged = {k: (dict(v) if isinstance(v, dict) else v)
                  for k, v in (out["grading"] or {}).items()}
        for key, val in over["grading"].items():
            if isinstance(val, dict):
                merged.setdefault(key, {}).update(val)
            else:
                merged[key] = val
        out["grading"] = grading_mod.normalize(merged)
    # Repairs are appended, like masks, and for a concrete reason: sensor dust
    # lands in the same place on every frame a body shoots, so "remove the dust
    # spots" is exactly the kind of thing a preset should carry across a shoot.
    for key, mod in (("healing", healing_mod), ("redeye", redeye_mod)):
        if over[key] is None:
            continue
        base_ops = list((out[key] or {}).get("ops", []))
        out[key] = mod.normalize({"ops": base_ops + list(over[key]["ops"])})
    out["masks"] = (out["masks"] + over["masks"])[:MASK_MAX]
    return out


def is_neutral(edit: dict[str, Any] | None) -> bool:
    """True when `edit` (normalized or not) leaves pixels unchanged."""
    e = normalize(edit)
    for key in _RANGES:
        # A modifier with its effect at zero (an angle with no motion) is a
        # setting, not a change to the pixels.
        if key in _MODIFIER_KEYS:
            continue
        if abs(float(e[key]) - float(DEFAULT_EDIT[key])) > _EPS:
            return False
    if any(mask_is_active(m) for m in e["masks"]):
        return False
    if e["watermark"] is not None:
        return False
    if e["crop"] is not None:
        return False   # a crop changes every pixel's place, if not its value
    if e["film"] is not None:
        return False
    if e["hsl"] is not None:
        return False   # normalize_hsl returns None unless a band is off neutral
    if e["grading"] is not None:
        return False   # likewise: grading.normalize drops anything neutral
    if e["healing"] is not None or e["redeye"] is not None:
        return False   # both normalize to None unless they would change pixels
    return all(_curve_is_identity(e[k]) for k in CURVE_KEYS)


def edit_hash(edit: dict[str, Any] | None) -> str:
    """Stable short id of a non-neutral edit (used to key cached thumbnails).
    Neutral edits hash to '' so unedited photos keep their plain thumb name."""
    if is_neutral(edit):
        return ""
    payload = json.dumps(normalize(edit), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# ----- tone curve (monotone cubic, numpy-only) ----------------------------

def _pchip(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone-cubic interpolation evaluated at `xq`. Guarantees
    no overshoot for monotone data; flattens slope at local extrema otherwise."""
    n = len(xs)
    if n == 1:
        return np.full_like(xq, ys[0])
    h = np.diff(xs)
    delta = np.diff(ys) / h
    m = np.empty(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    idx = np.clip(np.searchsorted(xs, xq) - 1, 0, n - 2)
    t = (xq - xs[idx]) / h[idx]
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    yq = (h00 * ys[idx] + h10 * h[idx] * m[idx]
          + h01 * ys[idx + 1] + h11 * h[idx] * m[idx + 1])
    return np.clip(yq, 0.0, 1.0)


def _curve_lut(curve: list[list[float]], n: int = 256) -> np.ndarray:
    pts = np.asarray(curve, dtype=np.float64)
    return _pchip(pts[:, 0], pts[:, 1], np.linspace(0.0, 1.0, n))


# ----- tone LUT (exposure + regions + contrast + curve) -------------------

def _tone_lut(e: dict[str, Any]) -> np.ndarray:
    """Fold exposure, the four tone regions, contrast and the master curve into a
    single float LUT sampled on [0,1]. Returns None-equivalent identity when all
    of them are neutral (caller can skip)."""
    x = np.linspace(0.0, 1.0, _LUT_N)
    t = x * (2.0 ** float(e["exposure"]))

    # Additive tone regions, masked by where each acts on the tonal ramp.
    hi, sh = e["highlights"] / 100.0, e["shadows"] / 100.0
    wh, bl = e["whites"] / 100.0, e["blacks"] / 100.0
    if hi:
        t = t + hi * 0.5 * np.clip(t, 0, 1) ** 2
    if sh:
        t = t + sh * 0.5 * (1.0 - np.clip(t, 0, 1)) ** 2
    if wh:
        t = t + wh * 0.3 * np.clip((t - 0.75) / 0.25, 0, 1)
    if bl:
        t = t + bl * 0.3 * np.clip((0.25 - t) / 0.25, 0, 1)

    c = e["contrast"] / 100.0
    if c:
        t = (t - 0.5) * (1.0 + c) + 0.5

    t = np.clip(t, 0.0, 1.0)

    if not _curve_is_identity(e["curve"]):
        clut = _curve_lut(e["curve"], 256)
        t = np.interp(t, np.linspace(0.0, 1.0, 256), clut)

    return np.clip(t, 0.0, 1.0).astype(np.float32)


def _wb_gains(e: dict[str, Any]) -> tuple[float, float, float] | None:
    kt, ki = e["temp"] / 100.0, e["tint"] / 100.0
    if not kt and not ki:
        return None
    return (1.0 + 0.25 * kt, 1.0 - 0.20 * ki, 1.0 - 0.25 * kt)


def _wb_tone_lut(e: dict[str, Any]) -> np.ndarray | None:
    """White balance and tone folded into one 256-entry per-channel LUT.

    Both stages are pointwise and per-channel — a gain followed by a monotone
    map — so their composition is just another per-channel function of the
    original 8-bit value. Evaluating it as a lookup turns the single most
    expensive step in the pipeline (an interpolation over every float in the
    frame) into a table read: ~340 ms becomes ~1 ms at 2048 px, and the result
    is identical to float precision.

    Returns None when neither stage does anything.
    """
    gains = _wb_gains(e)
    tone_flat = _tone_is_neutral(e)
    channel = [None if _curve_is_identity(e[k]) else _curve_lut(e[k], 256)
               for k in _CHANNEL_CURVES]
    if gains is None and tone_flat and not any(c is not None for c in channel):
        return None
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    cols = []
    for c in range(3):
        v = np.clip(x * gains[c], 0.0, 1.0) if gains is not None else x
        if not tone_flat:
            v = np.interp(v, np.linspace(0.0, 1.0, _LUT_N), _tone_lut(e))
        if channel[c] is not None:
            v = np.interp(v, np.linspace(0.0, 1.0, 256), channel[c])
        cols.append(np.clip(v, 0.0, 1.0).astype(np.float32))
    return np.stack(cols, axis=-1).reshape(256, 1, 3)


def _tone_is_neutral(e: dict[str, Any]) -> bool:
    return (
        abs(float(e["exposure"])) < _EPS
        and not any(e[k] for k in ("contrast", "highlights", "shadows", "whites", "blacks"))
        and _curve_is_identity(e["curve"])
    )


# ----- colour / detail stages ---------------------------------------------

def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _lowfreq(chan: np.ndarray, frame_long: float,
             work_edge: int = 512, sigma: float = 8.0) -> np.ndarray:
    """A resolution-independent low-frequency component: blur on a downscaled copy
    then upscale back. Keeps clarity looking the same on a preview and on the
    full-res export (and bounds cost regardless of megapixels).

    `frame_long` is the long edge of the *whole photo* in the current array's
    pixels, so a zoomed-in crop gets the same radius as the full-frame render.
    """
    h, w = chan.shape[:2]
    scale = work_edge / frame_long
    if scale < 1.0:
        small = cv2.resize(chan, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(chan, (0, 0), sigma)


def _apply_clarity(rgb: np.ndarray, clarity: int, frame_long: float) -> np.ndarray:
    """Midtone local contrast via unsharp on luma; the delta is added back to each
    channel so hue is preserved. Midtone mask avoids halos in highlights/shadows."""
    amount = clarity / 100.0 * 0.8
    y = _luma(rgb)
    detail = y - _lowfreq(y, frame_long)
    mask = 1.0 - (2.0 * np.clip(y, 0, 1) - 1.0) ** 2
    delta = (amount * detail * mask)[..., None]
    return rgb + delta


# ----- colour mixer (HSL) -------------------------------------------------
#
# Eight hue bands, each with hue / saturation / luminance. Stored sparsely —
# only values that are actually off neutral — so a neutral mixer normalizes to
# None and never makes an edit look non-neutral, and the edit hash stays short.
#
# Global only, no per-mask version, and deliberately so: the mixer is a
# statement about a colour wherever it appears in the frame, and a mask already
# answers "only here" better than eight bands could. Lightroom draws the same
# line for the same reason.

HSL_BANDS: tuple[str, ...] = ("red", "orange", "yellow", "green",
                              "aqua", "blue", "purple", "magenta")
HSL_KEYS: tuple[str, ...] = ("hue", "sat", "lum")

# Where each band sits on the hue circle, in degrees.
_HSL_CENTRES = (0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 285.0, 320.0)
_HSL_HUE_SHIFT = 30.0    # degrees of hue rotation at +/-100
_HSL_LUM_MAX = 0.7       # how far towards white or black at +/-100
_HSL_GREY_GUARD = 0.12   # saturation below which the mixer lets go; see _apply_hsl


def normalize_hsl(raw: Any) -> dict[str, dict[str, int]] | None:
    """Clamp a colour-mixer dict, dropping neutral entries. Returns None when
    nothing is left, which is what keeps a switched-on-but-untouched mixer from
    counting as an edit."""
    if not isinstance(raw, dict):
        return None
    out: dict[str, dict[str, int]] = {}
    for band in HSL_BANDS:
        src = raw.get(band)
        if not isinstance(src, dict):
            continue
        vals = {}
        for key in HSL_KEYS:
            if src.get(key) is None:
                continue
            val = int(round(_fnum(src[key], -100.0, 100.0, 0.0)))
            if val:
                vals[key] = val
        if vals:
            out[band] = vals
    return out or None


def _hsl_luts(hsl: dict[str, dict[str, int]]) -> tuple[np.ndarray, ...]:
    """One 360-entry lookup per parameter, indexed by hue in degrees.

    The eight band values are interpolated around the circle with a smoothstep
    between neighbours rather than a per-band window. Two things fall out of
    that: a hue sitting on a band centre gets exactly that band's value, and the
    weights are a partition of unity everywhere else — so no hue is skipped, no
    hue is counted twice, and there is no seam between orange and yellow to find
    later. Closing the circle is just a matter of repeating the centres either
    side of 0 and 360.
    """
    hue = np.arange(360, dtype=np.float32)
    centres = np.asarray(_HSL_CENTRES, dtype=np.float32)
    xs = np.concatenate([centres - 360.0, centres, centres + 360.0])
    luts = []
    for key in HSL_KEYS:
        vals = np.asarray([hsl.get(b, {}).get(key, 0) for b in HSL_BANDS],
                          dtype=np.float32)
        ys = np.concatenate([vals, vals, vals])
        idx = np.searchsorted(xs, hue, side="right") - 1
        span = np.maximum(xs[idx + 1] - xs[idx], 1e-6)
        t = _smoothstep(np.clip((hue - xs[idx]) / span, 0.0, 1.0))
        luts.append(ys[idx] * (1.0 - t) + ys[idx + 1] * t)
    return tuple(luts)


def _apply_hsl(rgb: np.ndarray, hsl: dict[str, dict[str, int]]) -> np.ndarray:
    """Rotate, saturate and lighten each hue band.

    Every change is scaled by the pixel's own saturation, and that guard is the
    difference between a usable mixer and a noisy one. The hue of a near-grey
    pixel is numerically meaningless — a rounding error decides whether a patch
    of grey concrete is "blue" or "purple" — so pushing those pixels would
    speckle a flat wall with two different corrections. Fading out below
    `_HSL_GREY_GUARD` means the mixer only ever moves colours that are there.
    """
    hue_lut, sat_lut, lum_lut = _hsl_luts(hsl)
    # H, L, S with H in degrees; the 1-degree quantization of the lookup is far
    # below anything visible.
    hls = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2HLS)
    h, lum, sat = hls[..., 0], hls[..., 1], hls[..., 2]
    at = np.clip(h, 0.0, 359.0).astype(np.int32)
    grip = _smoothstep(np.clip(sat / _HSL_GREY_GUARD, 0.0, 1.0))

    shift = hue_lut[at] / 100.0 * _HSL_HUE_SHIFT
    hls[..., 0] = np.mod(h + shift * grip, 360.0)
    gain = sat_lut[at] / 100.0
    hls[..., 2] = np.clip(sat * (1.0 + gain * grip), 0.0, 1.0)
    k = lum_lut[at] / 100.0 * _HSL_LUM_MAX * grip
    # Towards white or towards black, never past either: a multiplier would
    # clip the brightest band members and flatten them into one another.
    hls[..., 1] = np.clip(np.where(k >= 0.0, lum + (1.0 - lum) * k,
                                   lum * (1.0 + k)), 0.0, 1.0)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


# ----- texture, dehaze and noise -----------------------------------------
#
# Three stages between tone and clarity, in the order a frame wants them: cut
# the haze first (it is a property of the light, not of the detail), then clean
# the noise, then decide how much detail to bring back. Amplifying detail before
# denoising only gives the denoiser more to chew on.

_TEXTURE_WORK_EDGE = 1024   # finer than clarity's 512 — texture is pores, not regions
_TEXTURE_SIGMA = 2.0
_TEXTURE_MAX = 0.9          # detail multiplier at +/-100


def _apply_texture(rgb: np.ndarray, texture: int, frame_long: float) -> np.ndarray:
    """High-frequency detail amplitude on luma.

    Texture and clarity are the same operation at two scales, and the scale is
    the whole point. Clarity's wide radius and midtone mask move *regional*
    contrast; texture's small radius moves the finest detail the frame holds.
    So negative texture smooths skin without the flat look a blur gives — the
    edges of a face live in the low frequencies and are left alone — and
    positive texture finds fabric and bark rather than darkening one side of
    the sky.
    """
    amount = texture / 100.0 * _TEXTURE_MAX
    y = _luma(rgb)
    detail = y - _lowfreq(y, frame_long, _TEXTURE_WORK_EDGE, _TEXTURE_SIGMA)
    return rgb + (amount * detail)[..., None]


_DEHAZE_WORK_EDGE = 512      # the transmission map is estimated on a copy this long
_DEHAZE_PATCH_FRAC = 0.015   # dark-channel window, as a fraction of the long edge
_DEHAZE_OMEGA = 0.92         # haze deliberately left in, so distance still reads
_DEHAZE_TMIN = 0.12          # floor on transmission; without it dense haze explodes


def _transmission(rgb: np.ndarray, frame_long: float) -> np.ndarray:
    """Dark-channel-prior transmission: how much of each pixel reached the lens
    directly rather than as scattered light.

    The prior is that in a haze-free patch of an outdoor photo at least one
    channel is nearly black, so whatever floor the darkest channel keeps over a
    small window measures the haze in front of it.

    Estimated on a downscaled copy and upscaled back, like `_lowfreq`, which
    bounds the cost regardless of megapixels.

    The low pass before the minimum is not cosmetic. A minimum filter does not
    commute with downsampling — one dark pixel between two leaves survives at
    full resolution and is averaged away in a preview — so reading the dark
    channel straight off the pixels makes the strength of the effect depend on
    the render size, which is exactly what this pipeline promises never to
    happen. Haze is a smooth veil, so estimating it from a low-passed copy at a
    frame-relative radius is both the physically right thing to measure and the
    thing that is the same at every resolution.
    """
    h, w = rgb.shape[:2]
    unit = min(float(_DEHAZE_WORK_EDGE), frame_long)
    scale = unit / frame_long
    if scale < 1.0:
        small = cv2.resize(rgb, (max(4, int(w * scale)), max(4, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    patch = max(3, int(round(_DEHAZE_PATCH_FRAC * unit)))
    patch |= 1   # odd, so the window is centred
    small = cv2.GaussianBlur(np.clip(small, 0.0, 1.0), (0, 0), patch / 2.0)
    dark = small.min(axis=2)
    dark = cv2.erode(dark, np.ones((patch, patch), np.uint8))
    # The min filter leaves blocks; a blur of the same order turns them back
    # into the gradient that haze actually is.
    dark = cv2.GaussianBlur(dark, (0, 0), patch / 2.0)
    t = np.clip(1.0 - _DEHAZE_OMEGA * dark, _DEHAZE_TMIN, 1.0)
    if t.shape != (h, w):
        t = cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR)
    return t


def _apply_dehaze(rgb: np.ndarray, dehaze: int, frame_long: float) -> np.ndarray:
    """Invert the haze model I = J*t + A*(1-t), or run it forwards for negative
    values, which puts atmosphere back into a frame that has none.

    The atmospheric light A is fixed at white rather than estimated from the
    frame. Estimating it is the textbook step and it is better on a single whole
    image — but A is a *global* statistic, so a 1:1 window would arrive at a
    different one than the full-frame preview and the two renders would not
    match. The contract here is that the preview is the export, so a slightly
    weaker dehaze identical at every zoom beats a better one that drifts. Fixing
    A also keeps the stage local, which is what lets `effect_padding` cover it.
    """
    amount = dehaze / 100.0
    t = _transmission(rgb, frame_long)[..., None]
    direct = (rgb - 1.0) / t + 1.0
    return rgb + amount * (direct - rgb)


_DENOISE_MAX_H = 12.0        # non-local-means strength at 100, in 8-bit levels
_DENOISE_CHROMA_SIGMA = 3.0  # chroma blur at 100, in pixels
_DENOISE_TEMPLATE = 5        # patch compared
_DENOISE_SEARCH = 11         # neighbourhood searched for similar patches


def _apply_denoise(rgb: np.ndarray, denoise: int) -> np.ndarray:
    """Luminance and chroma noise reduction, split because the two look nothing
    alike. Chroma noise is coloured blotches several pixels across, and the eye
    carries almost no chroma detail, so it can simply be blurred away. Luma
    noise has to go without the detail going with it — that part is non-local
    means, which averages a pixel with the pixels whose neighbourhoods resemble
    its own, so an edge is only ever averaged with other copies of that edge.

    This is the one stage deliberately *not* resolution-independent. Noise is a
    per-pixel quantity: a downscaled preview has already averaged most of it
    away and there is nothing left there to remove. The strength is in 8-bit
    levels at the array's own scale, so judge it at 1:1 — the same thing
    Lightroom asks of you, for the same reason.

    Non-local means needs 8-bit input, so luma makes a round trip through it.
    That costs at most a level out of 255, the tolerance the tone LUT already
    accepts.
    """
    amount = denoise / 100.0
    ycc = cv2.cvtColor(np.clip(rgb, 0.0, 1.0), cv2.COLOR_RGB2YCrCb)
    y8 = np.rint(ycc[..., 0] * 255.0).astype(np.uint8)
    y8 = cv2.fastNlMeansDenoising(y8, None, _DENOISE_MAX_H * amount,
                                  _DENOISE_TEMPLATE, _DENOISE_SEARCH)
    ycc[..., 0] = y8.astype(np.float32) / 255.0
    sigma = _DENOISE_CHROMA_SIGMA * amount
    if sigma > 0.1:
        ycc[..., 1:] = cv2.GaussianBlur(np.ascontiguousarray(ycc[..., 1:]),
                                        (0, 0), sigma)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)


# ----- creative effects ---------------------------------------------------
#
# All four size themselves off `frame_long` — the long edge of the whole photo
# measured in the pixels of the array being graded — so a thumbnail, the editor
# preview, a 1:1 zoomed crop and the export all get the same *look*.

_BLUR_MAX_FRAC = 0.05    # lens blur radius at 100, as a fraction of the long edge
_MOTION_MAX_FRAC = 0.07  # motion blur length at 100
_GLOW_SIGMA_FRAC = 0.012
_PIXELATE_MAX_FRAC = 0.045
_KERNEL_CAP = 12         # kernels are built at most this big; the image is scaled instead


def _at_reduced(img: np.ndarray, radius: float, fn) -> np.ndarray:
    """Run a kernel op with the image downscaled so `radius` never exceeds
    _KERNEL_CAP. A 300 px bokeh on a 24 MP file costs the same as a 12 px one."""
    scale = min(1.0, _KERNEL_CAP / max(radius, 1e-6))
    h, w = img.shape[:2]
    if scale < 1.0:
        sw, sh = max(4, int(round(w * scale))), max(4, int(round(h * scale)))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        out = fn(small, radius * scale)
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return fn(img, radius)


def _disc_kernel(radius: float) -> np.ndarray:
    r = max(1, int(round(radius)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return k / float(k.sum())


def _apply_lens_blur(rgb: np.ndarray, blur: int, frame_long: float) -> np.ndarray:
    """Defocus with a disc kernel — the shape a real aperture makes, so out-of-
    focus highlights turn into circles instead of the mush a Gaussian gives.
    Filtering squared values lets highlights bloom the way clipped ones do."""
    radius = blur / 100.0 * _BLUR_MAX_FRAC * frame_long
    if radius < 0.6:
        return rgb

    def run(img: np.ndarray, r: float) -> np.ndarray:
        k = _disc_kernel(r)
        lit = cv2.filter2D(np.clip(img, 0.0, None) ** 2, -1, k, borderType=cv2.BORDER_REPLICATE)
        return np.sqrt(np.clip(lit, 0.0, None))

    return _at_reduced(rgb, radius, run)


def _apply_motion_blur(rgb: np.ndarray, motion: int, angle: float,
                       frame_long: float) -> np.ndarray:
    """Directional smear — a panned car, or speed added after the fact."""
    length = motion / 100.0 * _MOTION_MAX_FRAC * frame_long
    if length < 1.0:
        return rgb
    rad = np.radians(angle)
    dx, dy = float(np.cos(rad)), float(np.sin(rad))

    def run(img: np.ndarray, half: float) -> np.ndarray:
        n = max(1, int(round(half)))
        size = 2 * n + 1
        k = np.zeros((size, size), dtype=np.float32)
        p0 = (int(round(n - dx * n)), int(round(n - dy * n)))
        p1 = (int(round(n + dx * n)), int(round(n + dy * n)))
        cv2.line(k, p0, p1, 1.0, 1, lineType=cv2.LINE_AA)
        total = float(k.sum())
        if total <= 0:
            return img
        return cv2.filter2D(img, -1, k / total, borderType=cv2.BORDER_REPLICATE)

    return _at_reduced(rgb, length / 2.0, run)


def _apply_glow(rgb: np.ndarray, glow: int, frame_long: float) -> np.ndarray:
    """Bloom: isolate the highlights, spread them, screen them back on. Screen
    blending means it only ever lifts, never muddies the shadows."""
    amount = glow / 100.0
    if amount <= 0:
        return rgb
    thresh = 0.62
    y = _luma(np.clip(rgb, 0.0, 1.0))
    weight = np.clip((y - thresh) / (1.0 - thresh), 0.0, 1.0)[..., None]
    bright = np.clip(rgb, 0.0, 1.0) * weight
    sigma = max(1.0, _GLOW_SIGMA_FRAC * frame_long)
    spread = _at_reduced(bright, sigma, lambda im, s: cv2.GaussianBlur(im, (0, 0), max(0.6, s)))
    return 1.0 - (1.0 - rgb) * (1.0 - amount * np.clip(spread, 0.0, 1.0))


def _apply_pixelate(rgb: np.ndarray, pixelate: int, frame_long: float,
                    origin_px: tuple[float, float]) -> np.ndarray:
    """Mosaic. `origin_px` is where this array starts inside the whole photo, so
    the block grid is anchored to the frame — a zoomed crop shows the same
    blocks in the same places as the full-frame render."""
    block = int(round(pixelate / 100.0 * _PIXELATE_MAX_FRAC * frame_long))
    if block < 2:
        return rgb
    h, w = rgb.shape[:2]
    ox = int(round(origin_px[0])) % block
    oy = int(round(origin_px[1])) % block
    padded = cv2.copyMakeBorder(rgb, oy, 0, ox, 0, cv2.BORDER_REPLICATE)
    ph, pw = padded.shape[:2]
    nh, nw = max(1, -(-ph // block)), max(1, -(-pw // block))
    small = cv2.resize(padded, (nw, nh), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (nw * block, nh * block), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(big[oy:oy + h, ox:ox + w])


def _apply_color(rgb: np.ndarray, vibrance: int, saturation: int) -> np.ndarray:
    """Saturate/desaturate by lerping each pixel toward its luma. Vibrance adds
    extra push weighted by (1 - current saturation), so already-vivid pixels
    (and skin) move less. No HSV round-trip, so no hue shift."""
    y = _luma(rgb)[..., None]
    sat_proxy = rgb.max(axis=2, keepdims=True) - rgb.min(axis=2, keepdims=True)
    factor = 1.0 + saturation / 100.0 + (vibrance / 100.0) * (1.0 - np.clip(sat_proxy, 0, 1))
    return y + factor * (rgb - y)


def _sharpen_params(e: dict[str, Any]) -> dict[str, int]:
    """The four sharpening sliders, in the shape the `sharpening` module wants."""
    return {"amount": e["sharpen"], "radius": e["sharpen_radius"],
            "detail": e["sharpen_detail"], "masking": e["sharpen_masking"]}


def _apply_sharpen(rgb: np.ndarray, e: dict[str, Any],
                   frame_long: float) -> np.ndarray:
    """Capture sharpening; the operator lives in `sharpening`.

    This replaced a bare unsharp mask — `rgb + amount * (rgb - blur(sigma=1))` —
    and the replacement is NOT the same picture. `sharpen` keeps its name and its
    scale so a saved edit loads unchanged, but at the same number the result now
    differs by about 6 levels on average and up to 40 on a hard edge, because the
    new operator clamps each pixel into the range of luma already present around
    it and the old one was free to overshoot. No setting of the other three
    reproduces the old look; the clamp is the point of it. That is a deliberate
    improvement to every edit that carries a sharpen value — halos it used to
    put there are gone — and it is recorded here because a non-destructive editor
    changing what a saved edit renders to is worth saying out loud.
    """
    return sharpening_mod.apply_sharpen(rgb, _sharpen_params(e), frame_long)


def _apply_vignette(rgb: np.ndarray, vignette: int, roi: tuple[float, float, float, float]) -> np.ndarray:
    """Radial corner darkening (-) or brightening (+); neutral at 0. The falloff
    is anchored to the whole frame, so a zoomed crop shows its true slice of the
    vignette rather than getting one of its own."""
    h, w = rgb.shape[:2]
    # Pixel centres, the same convention the masks use (`_grid`). linspace's
    # endpoint sampling put a crop's grid half a pixel off from the matching
    # slice of the full frame's, so the 1:1 view drifted from the export.
    gys, gxs = _grid(h, w, roi)
    ny, nx = 2.0 * gys - 1.0, 2.0 * gxs - 1.0
    d2 = (nx * nx + ny * ny) / 2.0  # 0 at centre, 1 at the corners
    factor = 1.0 + (vignette / 100.0) * d2
    return rgb * factor[..., None]


# ----- mask alpha ---------------------------------------------------------

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite ease on an already-clipped [0,1] ramp — no hard edge on the falloff."""
    return t * t * (3.0 - 2.0 * t)


def _work_size(h: int, w: int) -> tuple[int, int]:
    """Alpha-map resolution: the image size, capped at _MASK_WORK_EDGE. Building
    the mask small and upscaling keeps cost flat in megapixels and makes the
    feather look identical on the preview and on the export."""
    scale = _MASK_WORK_EDGE / max(h, w)
    if scale >= 1.0:
        return h, w
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def _grid(sh: int, sw: int, roi: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Normalized whole-frame coordinates of each pixel of an sh x sw array that
    covers `roi` of the frame. With the default roi this is just [0,1]."""
    x0, y0, rw, rh = roi
    ys = (y0 + (np.arange(sh, dtype=np.float32) + 0.5) / sh * rh)[:, None]
    xs = (x0 + (np.arange(sw, dtype=np.float32) + 0.5) / sw * rw)[None, :]
    return ys, xs


def _radial_alpha(m: dict[str, Any], sh: int, sw: int,
                  roi: tuple[float, float, float, float]) -> np.ndarray:
    gys, gxs = _grid(sh, sw, roi)
    ys = gys - m["cy"]
    xs = gxs - m["cx"]
    rad = np.radians(m["angle"])
    ca, sa = np.cos(rad, dtype=np.float32), np.sin(rad, dtype=np.float32)
    # Rotate in normalized space so the shape matches the ellipse the UI draws.
    u = (xs * ca + ys * sa) / m["rx"]
    v = (ys * ca - xs * sa) / m["ry"]
    dist = np.sqrt(u * u + v * v)
    f = max(m["feather"] / 100.0, 1e-3)
    return _smoothstep(np.clip((1.0 - dist) / f, 0.0, 1.0))


def _linear_alpha(m: dict[str, Any], sh: int, sw: int,
                  roi: tuple[float, float, float, float]) -> np.ndarray:
    vx, vy = m["x2"] - m["x1"], m["y2"] - m["y1"]
    denom = float(vx * vx + vy * vy)
    gys, gxs = _grid(sh, sw, roi)
    ys = gys - m["y1"]
    xs = gxs - m["x1"]
    t = np.clip((xs * vx + ys * vy) / denom, 0.0, 1.0)
    # Feather squeezes the ramp toward the midpoint: 100 = the full span, 0 = a line.
    f = max(m["feather"] / 100.0, 1e-3)
    return 1.0 - _smoothstep(np.clip((t - 0.5) / f + 0.5, 0.0, 1.0))


def _brush_alpha(m: dict[str, Any], sh: int, sw: int,
                 roi: tuple[float, float, float, float]) -> np.ndarray:
    """Replay the strokes in order: paint strokes union in, erase strokes take
    out, each softened by its own radius so feather scales with brush size."""
    alpha = np.zeros((sh, sw), dtype=np.float32)
    f = m["feather"] / 100.0
    x0, y0, rw, rh = roi
    # Stroke coords are whole-frame fractions; map them into this window.
    for stroke in m["strokes"]:
        r_px = max(1.0, stroke["radius"] * sw / rw)
        layer = np.zeros((sh, sw), dtype=np.uint8)
        pts = np.array([[(p[0] - x0) / rw * sw, (p[1] - y0) / rh * sh]
                        for p in stroke["points"]], dtype=np.int32)
        cv2.polylines(layer, [pts], False, 255, thickness=int(round(2 * r_px)),
                      lineType=cv2.LINE_AA)
        # OpenCV thick lines have flat ends and mitre-free joins, so stamp a disc
        # at every sample: that is what gives the stroke round caps and corners.
        # Discs closer together than a third of the radius add nothing the
        # polyline has not already covered, and a wide brush over a long stroke
        # is thousands of them — skip those.
        step = max(1.0, r_px * 0.33)
        last = None
        for px, py in pts:
            if last is not None and abs(px - last[0]) + abs(py - last[1]) < step:
                continue
            last = (px, py)
            cv2.circle(layer, (int(px), int(py)), int(round(r_px)), 255, -1,
                       lineType=cv2.LINE_AA)
        # Always cap the far end, however the decimation fell.
        cv2.circle(layer, (int(pts[-1][0]), int(pts[-1][1])), int(round(r_px)), 255, -1,
                   lineType=cv2.LINE_AA)
        soft = layer.astype(np.float32) / 255.0
        if f > 0:
            # A wide brush means a wide feather, and a big-sigma Gaussian at
            # full size costs more than everything else in the mask put
            # together. Blur a downscaled copy instead — same result to within
            # a thousandth, and it is what the creative effects already do.
            sigma = max(0.6, f * r_px * 0.8)
            soft = _at_reduced(soft, sigma,
                               lambda im, s: cv2.GaussianBlur(im, (0, 0), max(0.3, s)))
        if stroke["erase"]:
            alpha *= 1.0 - soft
        else:
            alpha = np.maximum(alpha, soft)
    return alpha


# Alpha maps depend only on a mask's *shape*, never on its sliders — so moving
# an exposure slider re-uses them. That matters because a painted brush costs
# 20-400 ms to rasterize (every stroke point is a filled circle), and it was
# being rebuilt on every keystroke of a drag. Bounded: the work size caps the
# long edge at _MASK_WORK_EDGE, so an entry is at most a few MB.
_ALPHA_CACHE: "OrderedDict[bytes, np.ndarray]" = OrderedDict()
def _window_of(field: np.ndarray, sh: int, sw: int,
               roi: tuple[float, float, float, float]) -> np.ndarray:
    """Crop `roi` out of a whole-frame alpha and scale it to sh x sw.

    Shared by the two pixel-derived mask sources. Both compute their field over
    the whole photo — that is what makes them agree between the fit preview and
    a 1:1 window — so both need the same last step.
    """
    fh, fw = field.shape[:2]
    x0 = max(0, min(fw - 1, int(round(roi[0] * fw))))
    y0 = max(0, min(fh - 1, int(round(roi[1] * fh))))
    x1 = max(x0 + 1, min(fw, int(round((roi[0] + roi[2]) * fw))))
    y1 = max(y0 + 1, min(fh, int(round((roi[1] + roi[3]) * fh))))
    window = np.ascontiguousarray(field[y0:y1, x0:x1], dtype=np.float32)
    if window.shape[:2] != (sh, sw):
        window = cv2.resize(window, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return window


def _range_alpha(mask: dict[str, Any], sh: int, sw: int,
                 roi: tuple[float, float, float, float],
                 src: np.ndarray | None) -> np.ndarray | None:
    """A mask's range refinement, or None when it carries none.

    `src` is the whole photo, ungraded. Both of those matter. Whole, because the
    selection has to be the same one at every zoom. Ungraded, because a selection
    measured on graded pixels would move every time a slider did — and a
    luminance range in particular would then chase the exposure that is being
    set through it.
    """
    if mask["range_luma"] is None and mask["range_color"] is None:
        return None
    assert src is not None, (
        "a mask with a range refinement needs the whole ungraded frame: "
        "pass src= to render()")
    # The selector's own grid is the whole frame's mask grid, so the field is
    # identical whatever window is being rendered out of it.
    gh, gw = _work_size(*src.shape[:2])
    alpha = None
    if mask["range_luma"] is not None:
        alpha = rangemask_mod.luma_alpha(src, mask["range_luma"], (gh, gw))
    if mask["range_color"] is not None:
        col = rangemask_mod.color_alpha(src, mask["range_color"], (gh, gw))
        # Both present means both conditions: this tone AND this colour.
        alpha = col if alpha is None else alpha * col
    return _window_of(alpha, sh, sw, roi)


_AUTO_FEATHER_FRAC = 0.02   # blur radius at feather 100, as a fraction of the edge


def _auto_alpha(mask: dict[str, Any], sh: int, sw: int,
                roi: tuple[float, float, float, float],
                fields: dict[str, np.ndarray] | None) -> np.ndarray:
    """Crop and scale a whole-frame automatic mask down to this render window.

    `fields` maps a group name to its alpha over the whole photo, at whatever
    resolution the caller computed it (the model's output is 256 px wide, so it
    is a small array however big the photo is). See the note on MASK_TYPES for
    why it has to come from outside rather than be computed here.
    """
    assert fields is not None and mask["group"] in fields, (
        f"the {mask['group']!r} automatic mask needs its field supplied by the "
        f"caller: pass auto={{'{mask['group']}': alpha}} to render()"
    )
    window = _window_of(fields[mask["group"]], sh, sw, roi)
    if mask["feather"]:
        sigma = mask["feather"] / 100.0 * _AUTO_FEATHER_FRAC * max(sh, sw)
        if sigma > 0.3:
            window = cv2.GaussianBlur(window, (0, 0), sigma)
    return np.clip(window, 0.0, 1.0)


_ALPHA_CACHE_MAX = 12
_ALPHA_LOCK = threading.Lock()


def _alpha_key(mask: dict[str, Any], sh: int, sw: int,
               roi: tuple[float, float, float, float]) -> bytes:
    """Everything that changes the alpha, and nothing that doesn't."""
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{mask['type']}|{mask['invert']}|{mask['feather']}|{mask['amount']}"
             f"|{sh}x{sw}|{roi}".encode())
    if mask["type"] == "radial":
        h.update(np.asarray([mask[k] for k in ("cx", "cy", "rx", "ry", "angle")],
                            dtype=np.float64).tobytes())
    elif mask["type"] == "linear":
        h.update(np.asarray([mask[k] for k in ("x1", "y1", "x2", "y2")],
                            dtype=np.float64).tobytes())
    elif mask["type"] in ("auto", "range"):
        raise AssertionError(
            f"{mask['type']} masks depend on the pixels and are not cached here")
    else:
        for s in mask["strokes"]:
            h.update(f"{s['radius']}|{s['erase']}".encode())
            h.update(np.asarray(s["points"], dtype=np.float64).tobytes())
    return h.digest()


def _mask_alpha_at(mask: dict[str, Any], sh: int, sw: int,
                   roi: tuple[float, float, float, float] = FULL_ROI,
                   auto: dict[str, np.ndarray] | None = None,
                   src: np.ndarray | None = None) -> np.ndarray:
    """Build a mask's alpha at exactly sh x sw, covering `roi` of the frame.

    The result is cached and shared — treat it as read-only. Masks whose alpha
    depends on the pixels are not cached at all: the cache is keyed on the
    numbers describing a mask, and those say nothing about which photo it is
    being applied to, so a hit would be plain wrong. Both such paths are a crop
    and a resize of a small field, which is cheaper than hashing the photo.
    """
    ranged = _range_alpha(mask, sh, sw, roi, src)

    if mask["type"] == "auto":
        alpha = _auto_alpha(mask, sh, sw, roi, auto)
    elif mask["type"] == "range":
        assert ranged is not None, "a range mask normalizes away without a range"
        alpha = np.ones((sh, sw), dtype=np.float32)
    else:
        alpha = None

    if alpha is not None:
        if ranged is not None:
            alpha = alpha * ranged
        if mask["invert"]:
            alpha = 1.0 - alpha
        if mask["amount"] != 100:
            alpha = alpha * (mask["amount"] / 100.0)
        return np.clip(alpha, 0.0, 1.0).astype(np.float32)

    if ranged is not None:
        # A drawn shape refined by a range: the shape is cacheable, the
        # refinement is not, so take the cached shape and narrow a copy of it.
        shape = _mask_alpha_at({**mask, "range_luma": None, "range_color": None},
                               sh, sw, roi)
        # Inversion has already been applied to the cached shape. Applying the
        # range after it is what "this shape, narrowed to these tones" means;
        # inverting the product instead would select the whole rest of the frame.
        return np.clip(shape * ranged, 0.0, 1.0).astype(np.float32)

    key = _alpha_key(mask, sh, sw, roi)
    with _ALPHA_LOCK:
        hit = _ALPHA_CACHE.get(key)
        if hit is not None:
            _ALPHA_CACHE.move_to_end(key)
            return hit

    if mask["type"] == "radial":
        alpha = _radial_alpha(mask, sh, sw, roi)
    elif mask["type"] == "linear":
        alpha = _linear_alpha(mask, sh, sw, roi)
    else:
        alpha = _brush_alpha(mask, sh, sw, roi)   # "auto" returned above
    if mask["invert"]:
        alpha = 1.0 - alpha
    if mask["amount"] != 100:
        alpha = alpha * (mask["amount"] / 100.0)
    alpha = alpha.astype(np.float32)
    alpha.flags.writeable = False   # it is shared; nobody may scribble on it

    with _ALPHA_LOCK:
        _ALPHA_CACHE[key] = alpha
        while len(_ALPHA_CACHE) > _ALPHA_CACHE_MAX:
            _ALPHA_CACHE.popitem(last=False)
    return alpha


def mask_alpha(mask: dict[str, Any], h: int, w: int,
               roi: tuple[float, float, float, float] = FULL_ROI,
               auto: dict[str, np.ndarray] | None = None,
               src: np.ndarray | None = None) -> np.ndarray:
    """Alpha map of a normalized mask for an h x w image: float32 in [0,1]."""
    sh, sw = _work_size(h, w)
    alpha = _mask_alpha_at(mask, sh, sw, roi, auto, src)
    if (sh, sw) != (h, w):
        alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return alpha


def _full_edit_from_adj(adj: dict[str, Any]) -> dict[str, Any]:
    """Wrap a mask's local sliders in a full edit dict so `_grade` can run them."""
    e = dict(DEFAULT_EDIT)
    e["curve"] = [list(p) for p in DEFAULT_EDIT["curve"]]
    e.update(adj)
    return e


def _apply_masks(img: np.ndarray, masks: list[dict[str, Any]],
                 roi: tuple[float, float, float, float],
                 auto: dict[str, np.ndarray] | None = None,
                 src: np.ndarray | None = None) -> np.ndarray:
    """Blend a locally graded copy through each mask, in order. `img` is float32
    RGB clipped to [0,1] and is modified in place."""
    h, w = img.shape[:2]
    sh, sw = _work_size(h, w)
    for m in masks:
        if not mask_is_active(m):
            continue
        small = _mask_alpha_at(m, sh, sw, roi, auto, src)
        hits = np.argwhere(small > 1.0 / 512.0)
        if not len(hits):
            continue
        # Grade only the mask's bounding box (padded by a work pixel, so the
        # upscaled falloff is fully covered) instead of the whole frame.
        (sy0, sx0), (sy1, sx1) = hits.min(axis=0), hits.max(axis=0) + 1
        y0, y1 = int(max(0, sy0 - 1) * h / sh), int(min(sh, sy1 + 1) * h / sh + 0.5)
        x0, x1 = int(max(0, sx0 - 1) * w / sw), int(min(sw, sx1 + 1) * w / sw + 0.5)
        y1, x1 = min(h, max(y1, y0 + 1)), min(w, max(x1, x0 + 1))

        if (sh, sw) == (h, w):
            alpha = small[y0:y1, x0:x1]
        else:
            alpha = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        sub = img[y0:y1, x0:x1]
        # The sub-window's own place in the frame, so frame-relative effects
        # (vignette is global-only, but blur/mosaic sizing is not) stay honest.
        sub_roi = (roi[0] + x0 / w * roi[2], roi[1] + y0 / h * roi[3],
                   (x1 - x0) / w * roi[2], (y1 - y0) / h * roi[3])
        adj = _full_edit_from_adj(m["adj"])
        # Hand the grade 8-bit pixels so its white-balance/tone step can be a
        # lookup instead of an interpolation over every float in the crop —
        # about five times faster, and the crop is on its way to an 8-bit
        # result anyway. Only worth the conversion when there is a tone stage
        # to accelerate.
        if _wb_tone_lut(adj) is not None:
            src = cv2.convertScaleAbs(sub, alpha=255.0)
        else:
            src = sub.copy()
        graded = np.clip(_grade(src, adj, sub_roi), 0.0, 1.0)
        img[y0:y1, x0:x1] = sub + (graded - sub) * alpha[..., None]
    return img


# ----- top-level render ---------------------------------------------------

def _repair(rgb: np.ndarray, e: dict[str, Any],
            roi: tuple[float, float, float, float]) -> np.ndarray:
    """Red-eye and healing, before anything tonal.

    These two repair the captured image; everything after them interprets it.
    Getting the order wrong is not cosmetic. A red pupil left in place is
    amplified by every stage that follows — vibrance finds it, the curve lifts
    it, a colour-range mask keys off it — and a heal applied after a grade
    diffuses from pixels a curve has already crushed, baking the grade into the
    repair so that changing the curve afterwards leaves a patch that no longer
    matches its surroundings.

    Both modules want float32, and the round trip back to 8 bits is on purpose:
    it lets `_grade`'s white-balance-and-tone step stay a `cv2.LUT` lookup rather
    than an interpolation over every float in the frame, which is the difference
    between about 1 ms and 340 ms on a 2048 px frame. The cost is at most a level
    out of 255 — the same tolerance the module already documents for a mask
    grading an already-graded crop.
    """
    if e["redeye"] is None and e["healing"] is None:
        return rgb
    work = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb.copy()
    # Fixed order so a render is reproducible. The two are independent in
    # practice: a heal over a corrected pupil would simply replace it.
    if e["redeye"] is not None:
        work = redeye_mod.apply_redeye(work, e["redeye"], roi)
    if e["healing"] is not None:
        work = healing_mod.apply_healing(work, e["healing"], roi)
    return np.rint(np.clip(work, 0.0, 1.0) * 255.0).astype(np.uint8)


def _grade(img: np.ndarray, e: dict[str, Any],
           roi: tuple[float, float, float, float] = FULL_ROI,
           seed: int = 0) -> np.ndarray:
    """Run the adjustment stages of a normalized edit over float32 RGB in [0,1].
    Shared by the global pass and by every local mask (masks reuse the same maths
    on their own slider values, so a local +1 EV means what a global +1 EV means).
    The result may sit slightly outside [0,1]; the caller clips."""
    h, w = img.shape[:2]
    # The whole photo's size in this array's pixels. Effects size themselves off
    # it so they look the same at any render resolution, and `origin` is where
    # this array starts inside that frame.
    frame_w, frame_h = w / roi[2], h / roi[3]
    frame_long = max(frame_w, frame_h)
    origin = (roi[0] * frame_w, roi[1] * frame_h)

    # White balance + tone. An 8-bit input takes the lookup path; a float one
    # (a mask grading an already-graded crop) falls back to interpolation.
    lut = _wb_tone_lut(e)
    if img.dtype == np.uint8:
        img = cv2.LUT(img, lut) if lut is not None else img.astype(np.float32) / 255.0
    else:
        gains = _wb_gains(e)
        if gains is not None:
            img = np.clip(img * np.array(gains, dtype=np.float32), 0.0, 1.0)
        if not _tone_is_neutral(e):
            xs = np.linspace(0.0, 1.0, _LUT_N).astype(np.float32)
            img = np.interp(np.clip(img, 0.0, 1.0), xs, _tone_lut(e)).astype(np.float32)
        # Same order as the lookup path: master tone, then the channel curves.
        cxs = np.linspace(0.0, 1.0, 256).astype(np.float32)
        for c, key in enumerate(_CHANNEL_CURVES):
            if not _curve_is_identity(e[key]):
                img[..., c] = np.interp(np.clip(img[..., c], 0.0, 1.0), cxs,
                                        _curve_lut(e[key], 256)).astype(np.float32)

    if e["dehaze"]:
        img = _apply_dehaze(img, e["dehaze"], frame_long)
    if e["denoise"]:
        img = _apply_denoise(img, e["denoise"])
    if e["texture"]:
        img = _apply_texture(img, e["texture"], frame_long)
    if e["clarity"]:
        img = _apply_clarity(img, e["clarity"], frame_long)
    # The mixer before vibrance and saturation: it decides what each colour is,
    # and those two then decide how much of all of them there is.
    if e["hsl"] is not None:
        img = _apply_hsl(img, e["hsl"])
    # The wheels after the mixer: the mixer says what a colour is, the wheels
    # then push a whole tonal region somewhere regardless of what was there.
    if e["grading"] is not None:
        img = grading_mod.apply_grading(img, e["grading"])
    if e["vibrance"] or e["saturation"]:
        img = _apply_color(img, e["vibrance"], e["saturation"])
    # Optical effects last, in the order a camera would produce them: defocus
    # and smear happen at the lens, bloom is light spilling, sharpen is capture,
    # vignette is falloff, and the mosaic is a deliberate post step on top.
    if e["blur"]:
        img = _apply_lens_blur(img, e["blur"], frame_long)
    if e["motion"]:
        img = _apply_motion_blur(img, e["motion"], e["motion_angle"], frame_long)
    if e["glow"]:
        img = _apply_glow(img, e["glow"], frame_long)
    if e["sharpen"]:
        img = _apply_sharpen(img, e, frame_long)
    if e["vignette"]:
        img = _apply_vignette(img, e["vignette"], roi)
    # The capture medium goes on after the frame-wide falloff and before the
    # mosaic, which is a deliberate post step rather than part of the photograph.
    if e["film"] is not None:
        img = film_mod.apply_film(np.clip(img, 0.0, 1.0), e["film"], roi, seed)
    if e["pixelate"]:
        img = _apply_pixelate(img, e["pixelate"], frame_long, origin)
    return img


def effect_padding(edit: dict[str, Any] | None, frame_long: float) -> float:
    """How many pixels of neighbourhood a render of one window needs.

    Blur, smear and bloom pull in pixels from outside the window, so cutting a
    crop exactly to the viewport would leave a seam at its edge. Only the
    effects actually in use matter: a plain tone edit needs a handful of pixels
    for clarity, not the worst-case radius, and at 1:1 that difference is most
    of the work.
    """
    e = normalize(edit)

    film_need = film_mod.padding(e.get("film"), frame_long)

    def for_adj(a: dict[str, Any]) -> float:
        need = 0.0
        if a.get("blur"):
            need = max(need, a["blur"] / 100.0 * _BLUR_MAX_FRAC * frame_long)
        if a.get("motion"):
            need = max(need, a["motion"] / 100.0 * _MOTION_MAX_FRAC * frame_long / 2.0)
        if a.get("glow"):
            need = max(need, _GLOW_SIGMA_FRAC * frame_long * 3.0)
        if a.get("clarity"):
            # _lowfreq blurs at sigma 8 on a 512-long working copy.
            need = max(need, 8.0 * 3.0 * frame_long / 512.0)
        if a.get("texture"):
            need = max(need, _TEXTURE_SIGMA * 3.0 * frame_long / _TEXTURE_WORK_EDGE)
        if a.get("dehaze"):
            # The min filter's window, plus the blur that smooths it, at scale.
            patch = _DEHAZE_PATCH_FRAC * _DEHAZE_WORK_EDGE
            need = max(need, patch * 2.0 * frame_long / _DEHAZE_WORK_EDGE)
        if a.get("denoise"):
            # Fixed in pixels, not in frame fractions: see _apply_denoise.
            need = max(need, float(_DENOISE_SEARCH + _DENOISE_TEMPLATE))
        if a.get("sharpen"):
            need = max(need, sharpening_mod.padding(
                {"amount": a["sharpen"],
                 "radius": a.get("sharpen_radius", DEFAULT_EDIT["sharpen_radius"]),
                 "detail": a.get("sharpen_detail", DEFAULT_EDIT["sharpen_detail"]),
                 "masking": a.get("sharpen_masking", DEFAULT_EDIT["sharpen_masking"])},
                frame_long))
        return need

    # `w` and `h` are not both known here — callers pass only the long edge —
    # and a heal's radius is normalized against the width, so handing the long
    # edge in for both is the conservative bound. Overestimating costs a slightly
    # larger patch; underestimating would show a seam.
    repair_need = max(healing_mod.padding(e["healing"], int(frame_long), int(frame_long)),
                      redeye_mod.padding(e["redeye"], int(frame_long), int(frame_long)))
    pad = max(for_adj(e), film_need, repair_need)
    for m in e["masks"]:
        if mask_is_active(m):
            pad = max(pad, for_adj(m["adj"]))
    # A few pixels for rounding and for the resampling at a patch's edge. The
    # stages with a real reach — blur, smear, bloom, clarity, texture, dehaze,
    # denoise, sharpening — report theirs in `for_adj`, so this is only the slack
    # on top. Raising it is not free: it changes the patch size, which moves the
    # grid a mask's alpha is built on, which costs byte-exactness on a pure crop
    # (see test_a_window_matches_the_same_part_of_the_whole_render).
    return pad + 4.0


def render(rgb: np.ndarray, edit: dict[str, Any] | None,
           roi: tuple[float, float, float, float] = FULL_ROI,
           meta: dict[str, Any] | None = None,
           with_watermark: bool = True,
           geometry: bool = True,
           auto: dict[str, np.ndarray] | None = None,
           src: np.ndarray | None = None) -> np.ndarray:
    """Apply `edit` to an RGB uint8 image and return a new RGB uint8 image.
    A neutral edit returns the input array unchanged (no copy).

    `roi` is (x0, y0, w, h) in normalized coordinates saying where `rgb` sits
    inside the whole photo — pass it when grading a crop (the 1:1 editor view)
    so masks, the vignette and every frame-relative effect land where they would
    in the full-frame render. The default says `rgb` *is* the whole photo.

    `geometry` runs the straighten-and-crop stage, so `rgb` must then be a whole
    frame. Pass False when the caller is handing over a window and will place it
    itself — that is the ROI path, which grades a patch of the original and warps
    it. It defaults to True because a forgotten True crops something already
    cropped, which is obvious, while a forgotten False would quietly drop the crop
    out of an export.

    The order is grade, then geometry, then watermark, and it matters:

      - the grade (masks included) is measured against the *original* frame, so
        cropping later does not drag a mask off the thing it was drawn on;
      - the watermark goes on afterwards and is placed against the frame that
        comes out, because a signature belongs on the picture you end up with.

    `auto` supplies the fields any automatic mask needs: a dict from group name
    (see `segment.CLASS_GROUPS`) to that group's alpha over the WHOLE photo,
    computed from the ORIGINAL pixels. It is the caller's job because only the
    caller knows which photo this is and can cache per photo; see the note on
    MASK_TYPES for why neither constraint can be met from in here. An active
    automatic mask whose field is missing raises rather than quietly grading
    nothing.

    `src` is the whole photo, ungraded, for any mask carrying a range
    refinement. It must be the SAME array for every render of a photo — the
    caller's cached preview decode is the right choice, not the full-resolution
    frame for an export and the preview for a preview, because the selector is
    measured on a fixed grid derived from whatever it is handed and two different
    inputs would select two slightly different things.

    `meta` is the photo's shooting info (see `exifinfo`), used to fill the
    watermark's tokens. `with_watermark=False` grades without stamping, for
    callers that measure the result rather than show it — focus peaking would
    read a signature's crisp lettering as the sharpest thing in the frame.
    """
    e = normalize(edit)
    if is_neutral(e):
        return rgb

    do_geom = geometry and not geometry_is_neutral(e)
    if do_geom:
        assert roi == FULL_ROI, \
            "geometry needs the whole frame; pass geometry=False for a window"
    stamp = e["watermark"] if with_watermark else None
    assert not (stamp is not None and not geometry and not geometry_is_neutral(e)), \
        "a window cannot be stamped here: only its caller knows where it sits " \
        "in the cropped frame. Stamp it yourself with watermark.render()."
    # Geometry excluded: on its own it leaves nothing for the tonal stages to do.
    tone_neutral = is_neutral({**e, "watermark": None, "crop": None, "tilt": 0.0})
    if tone_neutral and stamp is None and not do_geom:
        return rgb

    if tone_neutral:
        out = rgb
    else:
        # Hand `_grade` the uint8 original so it can take the lookup path.
        # Grain must be the same grain every time this photo is rendered, so it
        # is seeded from the photo rather than from chance.
        seed = film_mod.seed_for(str((meta or {}).get("file") or ""))
        img = _grade(_repair(rgb, e, roi), e, roi, seed)
        img = np.clip(img, 0.0, 1.0)
        if e["masks"]:
            img = _apply_masks(img, e["masks"], roi, auto, src)
        out = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if do_geom:
        out = apply_geometry(out, e)
    if stamp is not None:
        # FULL_ROI once cropped: `out` is now the whole of the frame that ships.
        out = watermark_mod.render(out, stamp, meta, FULL_ROI if do_geom else roi)
    return out


# ----- auto tone ----------------------------------------------------------

def auto_tone(rgb: np.ndarray) -> dict[str, Any]:
    """Suggest a starting edit from the luma histogram: set the black/white points
    from the 1st/99th percentiles and nudge exposure so the median lands near a
    pleasant midtone. Returns an edit dict (never saved implicitly)."""
    y = _luma(rgb.astype(np.float32) / 255.0)
    p1, p50, p99 = (float(v) for v in np.percentile(y, [1, 50, 99]))
    exposure = float(np.clip(np.log2(0.45 / max(p50, 1e-3)), -1.5, 1.5))
    blacks = int(np.clip(round(-p1 * 300), -100, 0))
    whites = int(np.clip(round((1.0 - p99) * 300), 0, 100))
    contrast = 10 if (p99 - p1) < 0.6 else 0
    return normalize({
        "exposure": exposure,
        "blacks": blacks,
        "whites": whites,
        "contrast": contrast,
    })


# ----- white-balance picker -----------------------------------------------
#
# The inverse of `_wb_gains`: click something that should be grey and get the
# temp/tint that make it grey. Solved in the same space the gains are applied
# in — the encoded 8-bit value, gamma and all — rather than in linear light.
# Linearizing first would answer a different question correctly: the sliders
# multiply the encoded value, so the pair that reads neutral there is the pair
# that reads neutral here.

_PICK_PATCH = 0.008   # sample half-width, as a fraction of the long edge
_PICK_FLOOR = 10 / 255.0   # below this a patch is noise wearing a colour
_PICK_CEIL = 250 / 255.0   # ...and above it, a clipped channel with no ratio left


def neutral_wb(rgb: np.ndarray, x: float, y: float) -> dict[str, Any] | None:
    """Solve for the temp/tint that turn the patch at (`x`, `y`) grey.

    Two gains, two constraints (R = G and G = B), one solution. Writing the
    slider values as kt, ki in [-1, 1], `_wb_gains` is (1 + kt/4, 1 - ki/5,
    1 - kt/4), so the red and blue constraint is

        r(1 + kt/4) = b(1 - kt/4)   =>   kt = 4(b - r)/(r + b)

    and both channels land on 2rb/(r + b), their harmonic mean; green is then
    whatever ki takes it to that same value.

    `x` and `y` are fractions of the *original* frame, which is the space masks
    are stored in and the space the editor's overlay reports — and the right
    one here, since white balance runs before geometry ever crops anything.

    Returns the pair plus `clamped`, which says the cast was further than the
    sliders reach: the gains stop at 1.25/0.75, about 1.7 stops between red and
    blue, and a deep tungsten frame wants more than that. Green is then aimed at
    the midpoint of the red and blue it could actually reach, so what is left is
    a smaller version of the same cast rather than a green one laid over it.

    Returns None when the patch cannot answer the question: near-black has no
    colour but the noise floor, and a clipped channel has no ratio left to read.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected RGB, got {rgb.shape}"
    h, w = rgb.shape[:2]
    rad = max(1, int(round(_PICK_PATCH * max(h, w))))
    cx = int(round(min(max(x, 0.0), 1.0) * (w - 1)))
    cy = int(round(min(max(y, 0.0), 1.0) * (h - 1)))
    patch = rgb[max(0, cy - rad):cy + rad + 1, max(0, cx - rad):cx + rad + 1]
    # The median, not the mean: a dust speck or one hot pixel inside the patch
    # would drag a mean somewhere the eye never agreed to.
    r, g, b = (float(v) / 255.0 for v in np.median(patch.reshape(-1, 3), axis=0))
    if min(r, g, b) < _PICK_FLOOR or max(r, g, b) > _PICK_CEIL:
        return None

    kt = 4.0 * (b - r) / (r + b)
    clamped = abs(kt) > 1.0
    kt = min(1.0, max(-1.0, kt))
    # Where red and blue actually ended up. Equal, and equal to the harmonic
    # mean, whenever kt was not clamped — so this is one path, not two.
    target = 0.5 * (r * (1.0 + 0.25 * kt) + b * (1.0 - 0.25 * kt))
    ki = 5.0 * (1.0 - target / g)
    clamped = clamped or abs(ki) > 1.0
    ki = min(1.0, max(-1.0, ki))
    return {"temp": int(round(kt * 100.0)), "tint": int(round(ki * 100.0)),
            "clamped": clamped}
