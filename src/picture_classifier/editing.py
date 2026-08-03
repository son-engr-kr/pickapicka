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
import threading
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np

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
    "sharpen": 0,      # 0..100 (one-sided)
    "blur": 0,         # defocus / bokeh, 0..100
    "motion": 0,       # directional blur amount, 0..100
    "motion_angle": 0, # its direction in degrees
    "glow": 0,         # highlight bloom, 0..100
    "pixelate": 0,     # mosaic block size, 0..100
    "vignette": 0,
    "curve": [[0.0, 0.0], [1.0, 1.0]],
    "masks": [],       # local adjustments; see the "masks" section below
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
    "sharpen": (0, 100),
    "blur": (0, 100),
    "motion": (0, 100),
    "motion_angle": (-180, 180),
    "glow": (0, 100),
    "pixelate": (0, 100),
    "vignette": (-100, 100),
}

_EPS = 1e-4
_LUT_N = 1024  # tone-LUT sample count

# (x0, y0, w, h) in normalized frame coordinates. Anything other than this means
# the array being graded is a window onto a larger photo — see `render`.
FULL_ROI = (0.0, 0.0, 1.0, 1.0)


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

MASK_TYPES = ("radial", "linear", "brush")

# The sliders a mask may carry — everything except vignette (a frame-wide effect)
# and the master curve (kept global so the histogram stays readable).
LOCAL_KEYS: tuple[str, ...] = (
    "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    "temp", "tint", "vibrance", "saturation", "clarity", "sharpen",
    "blur", "motion", "motion_angle", "glow", "pixelate",
)

# Keys that describe *how* an effect looks rather than how much of it there is —
# on their own they change nothing, so they do not make a mask "active".
_MODIFIER_KEYS = frozenset({"motion_angle"})

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
        "adj": {k: (0.0 if k == "exposure" else 0) for k in LOCAL_KEYS},
    }
    if kind == "radial":
        m.update(cx=0.5, cy=0.5, rx=0.25, ry=0.25, angle=0.0)
    elif kind == "linear":
        m.update(x1=0.5, y1=0.15, x2=0.5, y2=0.55, feather=100)
    else:
        m.update(strokes=[])
    return m


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _normalize_adj(raw: Any) -> dict[str, Any]:
    """Clamp a mask's slider dict to the local subset (missing keys stay neutral)."""
    out: dict[str, Any] = {k: (0.0 if k == "exposure" else 0) for k in LOCAL_KEYS}
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
    else:
        m["strokes"] = _normalize_strokes(raw.get("strokes"))
        if not m["strokes"]:
            return None
    return m


def _adj_is_neutral(adj: dict[str, Any]) -> bool:
    return all(abs(float(adj.get(k, 0) or 0)) < _EPS
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
    out["curve"] = [list(p) for p in DEFAULT_EDIT["curve"]]
    out["masks"] = []
    out["watermark"] = None
    if not edit:
        return out
    for key, (lo, hi) in _RANGES.items():
        if key not in edit or edit[key] is None:
            continue
        try:
            val = float(edit[key])
        except (TypeError, ValueError):
            continue
        val = min(hi, max(lo, val))
        out[key] = val if key == "exposure" else int(round(val))
    if "curve" in edit:
        out["curve"] = _clean_curve(edit["curve"])
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
    if not _curve_is_identity(over["curve"]):
        out["curve"] = [list(p) for p in over["curve"]]
    if over["watermark"] is not None:
        out["watermark"] = over["watermark"]
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
    return _curve_is_identity(e["curve"])


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
    if gains is None and tone_flat:
        return None
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    cols = []
    for c in range(3):
        v = np.clip(x * gains[c], 0.0, 1.0) if gains is not None else x
        if not tone_flat:
            v = np.interp(v, np.linspace(0.0, 1.0, _LUT_N), _tone_lut(e))
        cols.append(v.astype(np.float32))
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


def _apply_sharpen(rgb: np.ndarray, sharpen: int) -> np.ndarray:
    amount = sharpen / 100.0 * 1.5
    blur = cv2.GaussianBlur(rgb, (0, 0), 1.0)
    return rgb + amount * (rgb - blur)


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
    else:
        for s in mask["strokes"]:
            h.update(f"{s['radius']}|{s['erase']}".encode())
            h.update(np.asarray(s["points"], dtype=np.float64).tobytes())
    return h.digest()


def _mask_alpha_at(mask: dict[str, Any], sh: int, sw: int,
                   roi: tuple[float, float, float, float] = FULL_ROI) -> np.ndarray:
    """Build a mask's alpha at exactly sh x sw, covering `roi` of the frame.

    The result is cached and shared — treat it as read-only.
    """
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
        alpha = _brush_alpha(mask, sh, sw, roi)
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
               roi: tuple[float, float, float, float] = FULL_ROI) -> np.ndarray:
    """Alpha map of a normalized mask for an h x w image: float32 in [0,1]."""
    sh, sw = _work_size(h, w)
    alpha = _mask_alpha_at(mask, sh, sw, roi)
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
                 roi: tuple[float, float, float, float]) -> np.ndarray:
    """Blend a locally graded copy through each mask, in order. `img` is float32
    RGB clipped to [0,1] and is modified in place."""
    h, w = img.shape[:2]
    sh, sw = _work_size(h, w)
    for m in masks:
        if not mask_is_active(m):
            continue
        small = _mask_alpha_at(m, sh, sw, roi)
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

def _grade(img: np.ndarray, e: dict[str, Any],
           roi: tuple[float, float, float, float] = FULL_ROI) -> np.ndarray:
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

    if e["clarity"]:
        img = _apply_clarity(img, e["clarity"], frame_long)
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
        img = _apply_sharpen(img, e["sharpen"])
    if e["vignette"]:
        img = _apply_vignette(img, e["vignette"], roi)
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
        return need

    pad = for_adj(e)
    for m in e["masks"]:
        if mask_is_active(m):
            pad = max(pad, for_adj(m["adj"]))
    return pad + 4.0    # a few pixels for sharpen's 1 px kernel and rounding


def render(rgb: np.ndarray, edit: dict[str, Any] | None,
           roi: tuple[float, float, float, float] = FULL_ROI,
           meta: dict[str, Any] | None = None,
           with_watermark: bool = True) -> np.ndarray:
    """Apply `edit` to an RGB uint8 image and return a new RGB uint8 image.
    A neutral edit returns the input array unchanged (no copy).

    `roi` is (x0, y0, w, h) in normalized coordinates saying where `rgb` sits
    inside the whole photo — pass it when grading a crop (the 1:1 editor view)
    so masks, the vignette and every frame-relative effect land where they would
    in the full-frame render. The default says `rgb` *is* the whole photo.

    `meta` is the photo's shooting info (see `exifinfo`), used to fill the
    watermark's tokens. `with_watermark=False` grades without stamping, for
    callers that measure the result rather than show it — focus peaking would
    read a signature's crisp lettering as the sharpest thing in the frame.
    """
    e = normalize(edit)
    if is_neutral(e):
        return rgb

    stamp = e["watermark"] if with_watermark else None
    tone_neutral = is_neutral({**e, "watermark": None})
    if tone_neutral and stamp is None:
        return rgb   # watermark-only edit, and the caller doesn't want it

    if tone_neutral:
        out = rgb
    else:
        # Hand `_grade` the uint8 original so it can take the lookup path.
        img = _grade(rgb, e, roi)
        img = np.clip(img, 0.0, 1.0)
        if e["masks"]:
            img = _apply_masks(img, e["masks"], roi)
        out = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if stamp is not None:
        out = watermark_mod.render(out, stamp, meta, roi)
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
