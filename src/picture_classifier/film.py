"""Film emulation: the chain, not a filter.

A "film look" shipped as a colour LUT plus white noise reads as a filter because
that is what it is. Film is a sequence of physical effects and each one is
separately recognisable, so this builds them in the order the physics happens:

  1. `tone`      exposure onto the emulsion. The characteristic (Hurter-Driffield)
                 response is applied to *log* exposure, per channel, with a toe
                 and a shoulder — which is where film's highlight retention comes
                 from. Then a dye-density crosstalk matrix, in density space,
                 because each dye layer absorbs outside its own band. That matrix
                 is what produces the shadow/highlight colour crossover film is
                 recognised by, and it is the part a per-channel curve cannot do.
  2. `halation`  light that reaches the film base scatters and re-exposes the
                 emulsion from behind. The red-sensitive layer sits deepest, so
                 the bleed is strongest in red: warm glow around a bright edge.
  3. `grain`     silver halide crystals are discrete and randomly placed. Not
                 additive gaussian noise: the variance depends on local density,
                 the structure is spatially correlated, and the grain must not
                 change size when the image is resized — otherwise the thumbnail,
                 the fit preview, the 1:1 view and the export each look different.

Everything is measured against the whole frame and takes a `roi`, so one window
of a photo renders identically to the same part of the full frame. That is the
same contract the rest of the pipeline keeps, and the reason a 1:1 view can be
trusted while grading.

References behind the choices:
  Newson, Delon, Galerne, "A Stochastic Film Grain Model for
    Resolution-Independent Rendering", Computer Graphics Forum 2017.
  Norkin, Birkbeck, "Film Grain Synthesis for AV1", DCC 2018, and AV1 spec
    section 7.18.3 — autoregressive template plus a scaling function of luma.
"""
from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np

# A stock is nothing but a named set of these numbers.
DEFAULT_FILM: dict[str, Any] = {
    "enabled": False,
    "stock": "",          # which preset these came from, for the UI only
    "strength": 100,      # blend the whole chain against the ungraded input
    # --- characteristic curve -------------------------------------------
    "contrast": 45,       # slope of the straight-line section
    "toe": 40,            # how pronounced the shadow roll-off is
    "shoulder": 45,       # how pronounced the highlight roll-off is
    # --- colour ---------------------------------------------------------
    "crosstalk": 25,      # strength of the dye-density matrix
    "warmth": 0,          # -100 cool .. +100 warm, as a density offset
    "split": 0,           # crossover: which way shadows and highlights part
    # --- halation -------------------------------------------------------
    "halation": 30,
    "halation_radius": 35,
    # --- grain ----------------------------------------------------------
    "grain": 35,
    "grain_size": 40,
    "grain_rough": 55,    # how correlated the structure is
}

_RANGES: dict[str, tuple[float, float]] = {
    "strength": (0, 100),
    "contrast": (0, 100),
    "toe": (0, 100),
    "shoulder": (0, 100),
    "crosstalk": (0, 100),
    "warmth": (-100, 100),
    "split": (-100, 100),
    "halation": (0, 100),
    "halation_radius": (0, 100),
    "grain": (0, 100),
    "grain_size": (0, 100),
    "grain_rough": (0, 100),
}

_EPS = 1e-5
_LUT_N = 256

# Film sees light in stops around a mid grey, not in code values. 18% grey and
# six stops either side is about the latitude of a colour negative.
_MID_GREY = 0.18
_LATITUDE = 6.0


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp into a full dict, or None when the result would change nothing."""
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_FILM)
    out["enabled"] = bool(raw.get("enabled", False))
    stock = raw.get("stock")
    out["stock"] = str(stock)[:40] if isinstance(stock, str) else ""
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


def is_neutral(film: dict[str, Any] | None) -> bool:
    """True when the chain would leave the pixels alone."""
    if not film or not film.get("enabled"):
        return True
    f = {**DEFAULT_FILM, **film}
    if f["strength"] <= 0:
        return True
    # Every stage at zero: no response shaping, no scatter, no crystals.
    return (f["contrast"] == 0 and f["crosstalk"] == 0 and f["warmth"] == 0
            and f["split"] == 0 and f["halation"] == 0 and f["grain"] == 0)


# ----- 1. characteristic curve and dye crosstalk ---------------------------

def _density_curve(f: dict[str, Any]) -> np.ndarray:
    """The response as a 256-entry table: input tone in, density out.

    A generalised logistic in log-exposure. The toe and shoulder exponents move
    the two ends independently, which is what a plain gamma or a symmetric
    sigmoid cannot do, and the whole thing stays monotone so no tone inverts.
    """
    x = np.linspace(0.0, 1.0, _LUT_N, dtype=np.float64)
    # Into log exposure, in stops around mid grey. Anchoring on mid grey is the
    # point: the response has to reshape contrast *about* the middle, and a range
    # measured from an epsilon instead spans eleven decades, which puts mid grey
    # up at 0.94 and makes every stock a brightening filter.
    lin = np.clip(x, _EPS, 1.0) ** 2.2
    stops = np.log2(lin / _MID_GREY)
    t = np.clip((stops + _LATITUDE) / (2.0 * _LATITUDE), 0.0, 1.0)
    # Exponents at or above one, so the response is always an S and never an
    # inverse one: below one the middle *flattens*, which made a higher contrast
    # setting able to reduce contrast depending on where the toe happened to be.
    # At zero both are exactly one, the logistic collapses to t, and the curve is
    # the identity — which is what lets `contrast` blend cleanly from nothing.
    toe = 1.0 + (f["toe"] / 100.0) * 1.3
    sho = 1.0 + (f["shoulder"] / 100.0) * 1.3
    a = np.clip(t, _EPS, 1.0) ** toe
    b = np.clip(1.0 - t, _EPS, 1.0) ** sho
    s = a / (a + b)
    # Back out the same way, so an unchanged shape is the identity. With toe and
    # shoulder equal to 1 the logistic is t itself and this returns x exactly,
    # which is what makes "no curve" mean no curve.
    out_lin = _MID_GREY * np.exp2(s * 2.0 * _LATITUDE - _LATITUDE)
    resp = np.clip(out_lin, 0.0, 1.0) ** (1.0 / 2.2)
    # Contrast is how far towards that response to go. At zero the stage is the
    # identity, because a stock with no curve of its own must not touch the tone.
    k = f["contrast"] / 100.0
    return (resp * k + x * (1.0 - k)).astype(np.float32)


def _crosstalk_matrix(f: dict[str, Any]) -> np.ndarray:
    """The dye-density matrix. Off-diagonal terms are one layer absorbing light
    meant for another; `split` makes the leak asymmetric, which is what tips the
    shadows one way and the highlights the other."""
    c = (f["crosstalk"] / 100.0) * 0.22
    sp = f["split"] / 100.0 * 0.5
    m = np.array([
        [1.0,               c * (1.0 + sp),  c * (0.55 - sp)],
        [c * (0.75 - sp),   1.0,             c * (0.75 + sp)],
        [c * (0.5 + sp),    c * (1.0 - sp),  1.0],
    ], dtype=np.float32)
    # Keep the matrix neutral overall, or the whole image gains a cast that the
    # white balance then has to undo.
    return m / m.sum(axis=1, keepdims=True)


def _apply_tone(img: np.ndarray, f: dict[str, Any]) -> np.ndarray:
    """The curve and the matrix, both in density space. `img` is float [0,1].

    The curve is pointwise and per-channel, so it goes through a 256-entry
    cv2.LUT over an 8-bit copy rather than three np.interp passes over the
    floats: about 40x quicker on a 2048 px frame, for a rounding error of a level
    out of 255. Same trade the rest of the pipeline already makes.
    """
    curve = _density_curve(f)
    table = np.repeat((curve * 255.0).astype(np.uint8).reshape(256, 1), 3, axis=1)
    src8 = cv2.convertScaleAbs(img, alpha=255.0)
    dens = cv2.LUT(src8, table.reshape(256, 1, 3)).astype(np.float32) / 255.0
    m = _crosstalk_matrix(f)
    if not np.allclose(m, np.eye(3, dtype=np.float32), atol=1e-6):
        dens = dens @ m.T
    if f["warmth"]:
        # A density offset, so it acts like a filter over the lamp rather than a
        # gain on the code values: it moves the midtones and leaves the ends.
        w = f["warmth"] / 100.0 * 0.06
        dens = dens + np.array([w, w * 0.15, -w], dtype=np.float32)
    return np.clip(dens, 0.0, 1.0)


# ----- 2. halation ---------------------------------------------------------

# Red deepest, blue shallowest: the layer order is why the bleed is warm.
_HALATION_WEIGHT = np.array([1.0, 0.42, 0.16], dtype=np.float32)


def _apply_halation(img: np.ndarray, f: dict[str, Any],
                    frame_long: float) -> np.ndarray:
    amount = f["halation"] / 100.0
    if amount <= 0:
        return img
    # Radius against the frame, so the preview is the export.
    radius = max(1.0, (f["halation_radius"] / 100.0) * 0.035 * frame_long)
    lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # A soft knee rather than a hard threshold: film does not start scattering at
    # a particular code value, and a hard edge here shows as a contour.
    hot = np.clip((lum - 0.55) / 0.45, 0.0, 1.0) ** 1.5
    # Blurred at reduced resolution: a wide gaussian over a soft mask does not
    # need full resolution, and this is the expensive part.
    scale = min(1.0, 320.0 / max(1.0, radius * 6.0))
    if scale < 1.0:
        small = cv2.resize(hot, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), radius * scale)
        spread = cv2.resize(small, (hot.shape[1], hot.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    else:
        spread = cv2.GaussianBlur(hot, (0, 0), radius)
    glow = spread[..., None] * _HALATION_WEIGHT * (amount * 1.35)
    return img + glow * (1.0 - img)      # screen, so it cannot clip past white


# ----- 3. grain ------------------------------------------------------------

def _grain_field(f: dict[str, Any], shape: tuple[int, int],
                 roi: tuple[float, float, float, float], frame_w: float,
                 frame_h: float, seed: int) -> np.ndarray:
    """A grain field for one window of the frame.

    Built on a lattice defined in *frame* coordinates, then sampled for the
    window being rendered. Two consequences, both required:

      - the grain is the same grain wherever it is asked for, so a 1:1 window
        shows what the export will have in that spot;
      - crystal size is fixed against the frame, not against the render, so it
        does not turn to fine sand in a thumbnail and boulders at 1:1.
    """
    h, w = shape
    # The lattice is counted across the frame, not measured in render pixels.
    # Deriving a cell size in pixels and then clamping it — which is what this
    # did first — makes the crystals relatively coarser in a small render, so a
    # thumbnail, the fit view and a 1:1 crop each showed different grain.
    cells = max(8, int(round(700.0 / (0.35 + (f["grain_size"] / 100.0) * 2.2))))
    if frame_w >= frame_h:
        gw, gh = cells, max(8, int(round(cells * frame_h / max(frame_w, _EPS))))
    else:
        gh, gw = cells, max(8, int(round(cells * frame_w / max(frame_h, _EPS))))
    rng = np.random.default_rng(seed)
    lattice = rng.standard_normal((gh, gw), dtype=np.float32)
    # Correlate it. White noise is sensor noise; grain has structure, and the
    # AR-style neighbour mixing is what gives it a clump size.
    rough = f["grain_rough"] / 100.0
    if rough > 0:
        k = cv2.GaussianBlur(lattice, (0, 0), 0.4 + rough * 1.1)
        # Keep the variance up: blurring alone would just make it quieter.
        k /= max(1e-6, float(k.std()))
        lattice = k
    # The window, in lattice coordinates. Straight from the frame fractions, so
    # the same part of the photo always lands on the same crystals.
    x0, x1 = roi[0] * gw, (roi[0] + roi[2]) * gw
    y0, y1 = roi[1] * gh, (roi[1] + roi[3]) * gh
    map_x = np.linspace(x0, x1, w, endpoint=False, dtype=np.float32)
    map_y = np.linspace(y0, y1, h, endpoint=False, dtype=np.float32)
    grid_x = np.tile(map_x, (h, 1))
    grid_y = np.repeat(map_y[:, None], w, axis=1)
    return cv2.remap(lattice, grid_x, grid_y, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT)


def _apply_grain(img: np.ndarray, f: dict[str, Any],
                 roi: tuple[float, float, float, float],
                 frame_w: float, frame_h: float, seed: int) -> np.ndarray:
    amount = f["grain"] / 100.0
    if amount <= 0:
        return img
    field = _grain_field(f, img.shape[:2], roi, frame_w, frame_h, seed)
    lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    # Grain lives in the mid-densities: clear film has no crystals to see and
    # fully exposed film is packed solid. Peaks around the middle, and 4L(1-L)
    # is the cheapest curve with that shape.
    weight = (4.0 * lum * (1.0 - lum)).astype(np.float32)
    sigma = amount * 0.045
    return img + (field * weight)[..., None] * sigma


# ----- the chain -----------------------------------------------------------

def seed_for(key: str) -> int:
    """A stable seed per photo, so the grain does not crawl between renders."""
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(),
                          "little")


def apply_film(img: np.ndarray, film: dict[str, Any] | None,
               roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
               seed: int = 0) -> np.ndarray:
    """Run the chain over float32 RGB in [0,1]. Returns float32 in [0,1].

    `roi` says where `img` sits in the whole photo, so the halation radius and
    the grain lattice are measured against the frame rather than this array.
    """
    if is_neutral(film):
        return img
    f = {**DEFAULT_FILM, **(film or {})}
    h, w = img.shape[:2]
    frame_w, frame_h = w / max(roi[2], _EPS), h / max(roi[3], _EPS)
    frame_long = max(frame_w, frame_h)

    out = _apply_tone(img, f)
    out = _apply_halation(out, f, frame_long)
    out = _apply_grain(out, f, roi, frame_w, frame_h, seed)
    out = np.clip(out, 0.0, 1.0)

    s = f["strength"] / 100.0
    if s < 1.0:
        out = img * (1.0 - s) + out * s
    return out.astype(np.float32)


def padding(film: dict[str, Any] | None, frame_long: float) -> float:
    """Neighbourhood the chain reaches for, in pixels — halation is a blur, so a
    window cut exactly to size would show a seam at its edge."""
    if is_neutral(film):
        return 0.0
    f = {**DEFAULT_FILM, **(film or {})}
    if not f["halation"]:
        return 0.0
    return (f["halation_radius"] / 100.0) * 0.035 * frame_long * 3.0


# ----- stocks -------------------------------------------------------------
# Named sets of the numbers above. Deliberately not named after products: these
# are shapes that read like a class of film, not measured profiles of one.

STOCKS: dict[str, dict[str, Any]] = {
    "Neutral negative": {
        "contrast": 40, "toe": 45, "shoulder": 50, "crosstalk": 18,
        "warmth": 4, "split": 8, "halation": 22, "halation_radius": 32,
        "grain": 30, "grain_size": 42, "grain_rough": 55,
    },
    "Warm portrait": {
        "contrast": 34, "toe": 55, "shoulder": 58, "crosstalk": 30,
        "warmth": 18, "split": 22, "halation": 34, "halation_radius": 40,
        "grain": 24, "grain_size": 36, "grain_rough": 60,
    },
    "Cool consumer": {
        "contrast": 48, "toe": 35, "shoulder": 42, "crosstalk": 26,
        "warmth": -16, "split": -20, "halation": 26, "halation_radius": 30,
        "grain": 40, "grain_size": 48, "grain_rough": 50,
    },
    "Punchy slide": {
        "contrast": 70, "toe": 22, "shoulder": 28, "crosstalk": 14,
        "warmth": 6, "split": -8, "halation": 16, "halation_radius": 24,
        "grain": 16, "grain_size": 28, "grain_rough": 45,
    },
    "Push-processed": {
        "contrast": 62, "toe": 30, "shoulder": 34, "crosstalk": 34,
        "warmth": 10, "split": 30, "halation": 40, "halation_radius": 46,
        "grain": 70, "grain_size": 62, "grain_rough": 65,
    },
    # Deliberately not called "black and white": the chain has no saturation
    # stage, so it would be promising something it does not do. This is the
    # fine-grain high-contrast shape, colour left alone.
    "Fine grain": {
        "contrast": 52, "toe": 38, "shoulder": 44, "crosstalk": 0,
        "warmth": 0, "split": 0, "halation": 18, "halation_radius": 28,
        "grain": 34, "grain_size": 30, "grain_rough": 60,
    },
}


def stock(name: str) -> dict[str, Any] | None:
    """A stock as a full film dict, ready to store on an edit."""
    preset = STOCKS.get(name)
    if preset is None:
        return None
    return normalize({**DEFAULT_FILM, **preset, "enabled": True, "stock": name})
