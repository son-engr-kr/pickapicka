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
import threading
from collections import OrderedDict
from typing import Any

import cv2
import numpy as np
from .sliders import tenth

# A stock is nothing but a named set of these numbers.
DEFAULT_FILM: dict[str, Any] = {
    "enabled": False,
    "stock": "",          # which preset these came from, for the UI only
    "strength": 100,      # blend the whole chain against the ungraded input
    # --- characteristic curve -------------------------------------------
    "contrast": 69,       # how far towards the film response to go
    "toe": 60,            # how pronounced the shadow roll-off is
    "shoulder": 45,       # how pronounced the highlight roll-off is
    # --- colour ---------------------------------------------------------
    "crosstalk": 61,      # strength of the dye-density matrix
    "warmth": 32,         # -100 cool .. +100 warm, as a density offset
    "split": -33,         # crossover: which way shadows and highlights part
    # --- halation -------------------------------------------------------
    "halation": 66,
    "halation_radius": 29,
    # --- grain ----------------------------------------------------------
    "grain": 86,
    "grain_size": 0,      # 0 is the finest grain that survives being looked at
    "grain_rough": 49,    # how correlated the structure is
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
_LUMA_ROW = np.array([[0.2126, 0.7152, 0.0722]], dtype=np.float32)
_LUT_N = 256

# Film sees light in stops around a mid grey, not in code values. 18% grey and
# six stops either side is about the latitude of a colour negative.
_MID_GREY = 0.18
_LATITUDE = 6.0

# The preview size the grain lattice is calibrated against: at this width,
# size 0 is 2.5 px per crystal and size 100 is 8 px.
_GRAIN_FINE = 2400     # crystals across the frame at the fine end
_GRAIN_COARSE = 300    # ...and at the coarse, pushed end


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
        out[key] = tenth(min(hi, max(lo, val)))
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
    # Into log exposure, in stops around mid grey.
    lin = np.clip(x, _EPS, 1.0) ** 2.2
    stops = np.log2(lin / _MID_GREY)
    t = np.clip((stops + _LATITUDE) / (2.0 * _LATITUDE), 0.0, 1.0)
    # Where mid grey and diffuse white sit on that axis.
    t_grey = 0.5
    t_white = (np.log2(1.0 / _MID_GREY) + _LATITUDE) / (2.0 * _LATITUDE)

    # Toe below the pivot, shoulder above it, meeting at mid grey. One logistic
    # across the whole range cannot do this: it pivots wherever its own centre
    # falls, which was below mid grey, so it brightened the midtones and pushed
    # the upper ones past white — a sky came back as paper. Two segments pin
    # (0,0), mid grey and white, so nothing clips and nothing drifts.
    toe = 1.0 + (f["toe"] / 100.0) * 1.3
    sho = 1.0 + (f["shoulder"] / 100.0) * 1.3
    lower = t_grey * np.clip(t / t_grey, 0.0, 1.0) ** toe
    span = max(t_white - t_grey, _EPS)
    u = np.clip((t - t_grey) / span, 0.0, 1.0)
    upper = t_grey + span * (1.0 - (1.0 - u) ** sho)
    s = np.where(t <= t_grey, lower, upper)
    # Above diffuse white the response keeps rising, or specular highlights would
    # flatten into a solid patch.
    over = np.clip((t - t_white) / max(1.0 - t_white, _EPS), 0.0, 1.0)
    s = s + over * (1.0 - t_white)

    # Back out the same way, so exponents of one give exactly the identity.
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
        # Per pixel m . rgb, which is `dens @ m.T`: the same to a float32
        # rounding and thirty times quicker.
        dens = cv2.transform(dens, m)
    if f["warmth"]:
        # A density offset, so it acts like a filter over the lamp rather than a
        # gain on the code values: it moves the midtones and leaves the ends.
        w = f["warmth"] / 100.0 * 0.06
        dens = dens + np.array([w, w * 0.15, -w], dtype=np.float32)
    return np.clip(dens, 0.0, 1.0)


# ----- 2. halation ---------------------------------------------------------

# Red deepest, blue shallowest: the layer order is why the bleed is warm.
_HALATION_WEIGHT = np.array([1.0, 0.42, 0.16], dtype=np.float32)
# The widest blur run at the render's own size, in pixels. Wider, the mask is
# blurred at reduced size with the blur scaled to match: at full size a 27 px
# blur of a 2048 px preview cost 110 ms, and was most of the film look. Against
# the full-size blur the finished frame moves by at most 2 levels out of 255
# (halation at 100; a mean of 0.05), measured over the radius range on real
# photos. Above a 53 px blur this was already the path taken, the export's
# included.
_HALATION_BLUR_CAP = 20.0


def _apply_halation(img: np.ndarray, f: dict[str, Any],
                    frame_long: float) -> np.ndarray:
    amount = f["halation"] / 100.0
    if amount <= 0:
        return img
    # Radius against the frame, so the preview is the export.
    radius = max(1.0, (f["halation_radius"] / 100.0) * 0.035 * frame_long)
    lum = cv2.transform(img, _LUMA_ROW)
    # A soft knee, but a high one. Scatter comes off specular highlights and
    # bright edges, not off every light area: a knee at 0.55 catches a whole sky
    # and washes it to paper white, which is the difference between halation and
    # a veiling glare. The square keeps the ramp near the top end.
    hot = np.clip((lum - 0.80) / 0.20, 0.0, 1.0) ** 2.0
    # Blurred at reduced resolution: a wide gaussian over a soft mask does not
    # need full resolution, and this is the expensive part.
    scale = min(1.0, 320.0 / max(1.0, radius * 6.0), _HALATION_BLUR_CAP / radius)
    if scale < 1.0:
        small = cv2.resize(hot, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), radius * scale)
        spread = cv2.resize(small, (hot.shape[1], hot.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
    else:
        spread = cv2.GaussianBlur(hot, (0, 0), radius)
    glow = spread[..., None] * _HALATION_WEIGHT * (amount * 1.9)
    return img + glow * (1.0 - img)      # screen, so it cannot clip past white


# ----- 3. grain ------------------------------------------------------------

# Grain fields last asked for. A field depends only on the grain settings, the
# window and the seed, and a slider drag over anything else asks for the same
# one on every render: 40 ms of noise and resampling each time on a 2048 px
# preview. Entries are read-only and shared.
_GRAIN_CACHE: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
_GRAIN_CACHE_MAX = 4
_GRAIN_LOCK = threading.Lock()


def _grain_field(f: dict[str, Any], shape: tuple[int, int],
                 roi: tuple[float, float, float, float], frame_w: float,
                 frame_h: float, seed: int) -> np.ndarray:
    key = (f["grain_size"], f["grain_rough"], tuple(shape), tuple(roi),
           float(frame_w), float(frame_h), seed)
    with _GRAIN_LOCK:
        hit = _GRAIN_CACHE.get(key)
        if hit is not None:
            _GRAIN_CACHE.move_to_end(key)
            return hit
    field = _build_grain_field(f, shape, roi, frame_w, frame_h, seed)
    field.flags.writeable = False
    with _GRAIN_LOCK:
        _GRAIN_CACHE[key] = field
        while len(_GRAIN_CACHE) > _GRAIN_CACHE_MAX:
            _GRAIN_CACHE.popitem(last=False)
    return field


def _build_grain_field(f: dict[str, Any], shape: tuple[int, int],
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
    # Crystal size belongs to the negative, so it is counted across the frame and
    # never in render pixels. The range spans a fine modern emulsion to a coarse
    # pushed one; at a 1280 px view the middle of it is about a pixel per crystal,
    # which is what a 35 mm scan at that size actually looks like. The previous
    # range started at 2.5 px per crystal, i.e. already coarser than anything
    # natural, which is why the finest setting was the only one that read right.
    cells = max(8, int(round(_GRAIN_FINE
                             - (f["grain_size"] / 100.0) * (_GRAIN_FINE - _GRAIN_COARSE))))
    if frame_w >= frame_h:
        gw, gh = cells, max(8, int(round(cells * frame_h / max(frame_w, _EPS))))
    else:
        gh, gw = cells, max(8, int(round(cells * frame_w / max(frame_h, _EPS))))
    rng = np.random.default_rng(seed)
    lattice = rng.standard_normal((gh, gw), dtype=np.float32)
    # Correlate it. White noise is sensor noise; grain has structure, and the
    # neighbour mixing is what gives it a clump size.
    rough = f["grain_rough"] / 100.0
    if rough > 0:
        k = cv2.GaussianBlur(lattice, (0, 0), 0.4 + rough * 1.1)
        k /= max(1e-6, float(k.std()))   # blurring alone would just be quieter
        lattice = k
    # The window, in lattice coordinates. Straight from the frame fractions, so
    # the same part of the photo always lands on the same crystals.
    x0, x1 = roi[0] * gw, (roi[0] + roi[2]) * gw
    y0, y1 = roi[1] * gh, (roi[1] + roi[3]) * gh

    # Crystals finer than the output pixels are averaged, not point-sampled.
    # Point-sampling aliases, which is what forced the range to stop at 2.5 px per
    # crystal; area-averaging is what downsampling the export would do, so the
    # preview keeps predicting the file at any grain size.
    # The factor comes from the *frame*, not from this window. Deriving it from
    # the window makes the supersample step differ between a full render and a
    # crop of it, so the averaging lands on different phases and the 1:1 view
    # stops matching the export — it drifted by three levels before this.
    k = max(1, int(np.ceil(gw / max(frame_w, 1.0))),
            int(np.ceil(gh / max(frame_h, 1.0))))
    if k > 1:
        fine = _sample_lattice(lattice, x0, y0, x1, y1, w * k, h * k)
        return cv2.resize(fine, (w, h), interpolation=cv2.INTER_AREA)
    return _sample_lattice(lattice, x0, y0, x1, y1, w, h)


def _sample_lattice(lattice: np.ndarray, x0: float, y0: float, x1: float,
                    y1: float, w: int, h: int) -> np.ndarray:
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
    lum = cv2.transform(img, _LUMA_ROW)
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
        "contrast": 62, "toe": 52, "shoulder": 50, "crosstalk": 48,
        "warmth": 14, "split": -18, "halation": 52, "halation_radius": 30,
        "grain": 72, "grain_size": 14, "grain_rough": 50,
    },
    "Warm portrait": {
        "contrast": 56, "toe": 64, "shoulder": 60, "crosstalk": 58,
        "warmth": 38, "split": 26, "halation": 64, "halation_radius": 38,
        "grain": 62, "grain_size": 10, "grain_rough": 55,
    },
    "Cool consumer": {
        "contrast": 72, "toe": 44, "shoulder": 44, "crosstalk": 54,
        "warmth": -30, "split": -34, "halation": 56, "halation_radius": 26,
        "grain": 84, "grain_size": 22, "grain_rough": 48,
    },
    "Punchy slide": {
        "contrast": 86, "toe": 26, "shoulder": 30, "crosstalk": 34,
        "warmth": 12, "split": -12, "halation": 40, "halation_radius": 20,
        "grain": 46, "grain_size": 4, "grain_rough": 42,
    },
    "Push-processed": {
        "contrast": 80, "toe": 38, "shoulder": 38, "crosstalk": 66,
        "warmth": 22, "split": 40, "halation": 78, "halation_radius": 44,
        "grain": 96, "grain_size": 34, "grain_rough": 62,
    },
    # Overdone on purpose: a very long toe, halation at the stop and warmth well
    # up. Reads like a flared, hazy frame rather than a clean negative.
    "Heavy glow": {
        "contrast": 77, "toe": 86, "shoulder": 39, "crosstalk": 25,
        "warmth": 49, "split": 0, "halation": 100, "halation_radius": 35,
        "grain": 90, "grain_size": 13, "grain_rough": 67,
    },
    "Fine grain": {
        "contrast": 70, "toe": 46, "shoulder": 48, "crosstalk": 30,
        "warmth": 8, "split": -10, "halation": 44, "halation_radius": 22,
        "grain": 70, "grain_size": 0, "grain_rough": 45,
    },
}

def stock(name: str) -> dict[str, Any] | None:
    """A stock as a full film dict, ready to store on an edit."""
    preset = STOCKS.get(name)
    if preset is None:
        return None
    return normalize({**DEFAULT_FILM, **preset, "enabled": True, "stock": name})
