"""Red-eye and pet-eye repair: two different problems, two code paths.

This is a repair to the *capture*, not an interpretation of it, so the caller
must run it BEFORE the parametric grade — a red pupil left in place is amplified
by every stage that follows (vibrance finds it, the tone curve lifts it, a mask
built on colour keys off it), and a pupil corrected after a grade is corrected
against numbers the photograph no longer has.

Two corrections, deliberately not sharing one implementation:

  - `red` (people). Flash returns off a blood-rich retina, so the pupil glows
    red while the rest of the eye is untouched. The fix is pointwise: measure how
    far the red channel runs ahead of green and blue, and pull those pixels to a
    neutral built from the channels the flash did *not* contaminate. Crucially
    the specular catchlight is protected — it is bright in all three channels, so
    the redness measure ignores it anyway, and an explicit achromatic-brightness
    guard keeps a pink-tinged one as well. Killing the catchlight is what makes a
    corrected eye look dead, and it is the difference between this and a
    black disc painted over a pupil.

  - `pet` (animals). The tapetum lucidum is a mirror behind the retina, so the
    return is far stronger and rarely red: green, yellow, orange or blown to
    white, and covering most of the eye rather than just the pupil. A red-channel
    test finds none of it. The gate here is luminance, the neutral target is the
    full luma rather than the uncontaminated channels (on a blown reflection
    there are none), no highlight is preserved because the reflection *is* the
    highlight, and a synthetic catchlight is drawn back in because the real one
    was destroyed. That last part is Lightroom's behaviour too, and it is why
    pet-eye has a catchlight position control and red-eye does not.

What the automatic detection genuinely does, and what it does not:

  - `detect(rgb)` finds *human* eyes geometrically first, then tests each one for
    a red-eye pupil. The geometric gate is the load-bearing part. A colour test
    on its own cannot tell a red pupil from a red LED — both are a compact,
    saturated red blob — so anything not inside a located eye is never even
    looked at. That is what stops the failure this feature is infamous for: a
    red jumper, red lipstick and a brake light are rejected because no eye is
    there, and rejected *again* by the verifier if a user draws a region over
    one by hand (see `detect_in_region`).
  - The eye locator is MediaPipe FaceMesh, reusing the singleton and the eye
    landmark indices already in `scoring/eyes.py`, when mediapipe is installed
    (it is this project's optional `faces` extra). Without it the locator falls
    back to the frontal-face and eye Haar cascades that ship inside
    opencv-python. This is a capability choice made once at import, not an
    exception handler: both paths only ever *propose* an eye, and the same
    verifier decides. The cascade path is markedly weaker — it wants a frontal,
    unoccluded, reasonably large face — so a photo with no detected face simply
    gets no automatic correction, which is the right way to be wrong here.
  - Detection is not attempted for pets at all, and that is a deliberate refusal
    rather than an omission. There is no pet-eye landmark model available under
    this project's dependencies; opencv-python does bundle a frontal *cat* face
    cascade, but nothing that locates an eye inside it, nothing at all for dogs,
    and a false positive paints a dark disc onto an animal's face. Pet eye is
    therefore user-placed: `detect_in_region` verifies and tightens a circle the
    user has drawn, which is honest about needing one click per eye.

Coordinates are normalized against the whole frame — centres as fractions of the
frame, radii as a fraction of frame *width* (the convention `editing._brush_alpha`
already uses for brush radii) — so a correction placed on a fit preview lands in
exactly the same place in a 1:1 window and in the export. Every stage is
pointwise given the disc geometry, so a window render is identical to the same
slice of the full-frame render; nothing here adapts to what happens to be inside
the array it was handed.

Cost is bounded by the eye, not the frame: only the bounding box of each disc is
touched. numpy + OpenCV only.
"""
from __future__ import annotations

import importlib.util
from typing import Any

import cv2
import numpy as np

FULL_ROI = (0.0, 0.0, 1.0, 1.0)
_EPS = 1e-4

KINDS: tuple[str, ...] = ("red", "pet")

# Twelve faces' worth of eyes. A flash group shot is the only thing that gets
# near it, and the cap is what keeps a hostile edit dict from asking for 10^6
# discs.
CORRECTION_MAX = 24

# Below this a disc is sub-pixel on any sane frame: not a correction, a stray click.
_MIN_RADIUS = 5e-4
_MAX_RADIUS = 0.25

# ----- schema -------------------------------------------------------------

DEFAULT_CORRECTION: dict[str, Any] = {
    "kind": "red",       # "red" for people, "pet" for animals
    "cx": 0.5,           # centre, fraction of frame width
    "cy": 0.5,           # centre, fraction of frame height
    "r": 0.01,           # radius, fraction of frame width
    "amount": 100,       # 0..100, how much of the correction to blend in
    "darken": 50,        # 0..100, how far the corrected pupil goes towards black
    "catchlight": True,  # pet only: draw a synthetic highlight back in
    "cat_x": -0.30,      # its centre, in radii from the eye centre (-1..1)
    "cat_y": -0.35,
    "cat_size": 28,      # its radius, percent of the eye radius
}

DEFAULT_REDEYE: dict[str, Any] = {
    "enabled": False,
    "corrections": [],
}

_CORRECTION_RANGES: dict[str, tuple[float, float]] = {
    "cx": (0.0, 1.0),
    "cy": (0.0, 1.0),
    "r": (0.0, _MAX_RADIUS),
    "amount": (0.0, 100.0),
    "darken": (0.0, 100.0),
    # Kept inside the disc so a catchlight cannot be flung onto a cheek.
    "cat_x": (-0.70, 0.70),
    "cat_y": (-0.70, 0.70),
    "cat_size": (5.0, 60.0),
}
_CORRECTION_FLOATS: tuple[str, ...] = ("cx", "cy", "r", "cat_x", "cat_y")

# ----- tuning -------------------------------------------------------------

# Feather as a fraction of the radius. A hard circular edge is the tell of a
# cheap fix, and 0.35 is wide enough to read as an eye and narrow enough to
# leave the pupil fully corrected.
_EDGE_FEATHER = 0.35
_CAT_FEATHER = 0.55       # the synthetic highlight is softer still
_CAT_LEVEL = 0.94         # not 1.0: a clipped catchlight looks pasted on
# A catchlight is a reflection off the cornea, so it cannot spill onto the lid:
# it is confined to the eye by an almost-hard edge, which dims it only if the
# position and size the user chose actually push it over the rim.
_CAT_CONFINE = 0.12

# How far red has to run ahead of green and blue before it is treated as flash
# return. A brown iris sits around 0.10-0.18; a red-eye pupil is 0.3 upwards.
_RED_LO = 0.10
_RED_HI = 0.30

# Achromatic brightness (the min channel) above which a pixel is read as the
# specular catchlight and left alone.
_SPEC_LO = 0.55
_SPEC_HI = 0.85

# Luma band over which a pet-eye reflection is taken to be a reflection. A dark
# iris and the lid stay below it.
_PET_LO = 0.25
_PET_HI = 0.55

# darken=100 keeps this much of the pupil's own level. Not 0: a pupil is dark,
# not a hole.
_DARKEN_MAX = 0.82


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def normalize_correction(raw: Any) -> dict[str, Any] | None:
    """Clamp one correction, or None when it would change nothing."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in KINDS:
        return None
    out = dict(DEFAULT_CORRECTION)
    out["kind"] = kind
    for key, (lo, hi) in _CORRECTION_RANGES.items():
        if raw.get(key) is None:
            continue
        val = _fnum(raw[key], lo, hi, float(DEFAULT_CORRECTION[key]))
        out[key] = val if key in _CORRECTION_FLOATS else int(round(val))
    # A red pupil keeps its own catchlight, so the field would only be a lie
    # about what the correction does.
    out["catchlight"] = kind == "pet" and bool(raw.get("catchlight", True))
    if out["r"] < _MIN_RADIUS or out["amount"] <= 0:
        return None
    return out


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp into a full dict, or None when the result would change nothing.

    Corrections that could not do anything (no radius, no amount, an unknown
    kind) are dropped rather than carried, so a stray click in the UI never
    leaves a photo looking edited.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, Any] = {"enabled": bool(raw.get("enabled", False)),
                           "corrections": []}
    src = raw.get("corrections")
    if isinstance(src, (list, tuple)):
        for item in src[:CORRECTION_MAX]:
            c = normalize_correction(item)
            if c is not None:
                out["corrections"].append(c)
    if is_neutral(out):
        return None
    return out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when these parameters would leave every pixel alone."""
    if not params or not params.get("enabled"):
        return True
    src = params.get("corrections")
    if not isinstance(src, (list, tuple)):
        return True
    return not any(normalize_correction(c) is not None for c in src)


# ----- geometry -----------------------------------------------------------

def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite ease on an already-clipped [0,1] ramp. Mirrors
    `editing._smoothstep`; duplicated rather than imported because `editing`
    imports this module."""
    return t * t * (3.0 - 2.0 * t)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _place(c: dict[str, Any], shape: tuple[int, int],
           roi: tuple[float, float, float, float]
           ) -> tuple[float, float, float] | None:
    """Where a correction's disc sits in an array covering `roi`, in pixels.
    Returns None when the disc misses this window entirely."""
    h, w = shape
    frame_w = w / max(roi[2], _EPS)
    frame_h = h / max(roi[3], _EPS)
    px = (c["cx"] - roi[0]) * frame_w
    py = (c["cy"] - roi[1]) * frame_h
    rad = max(1.0, c["r"] * frame_w)
    reach = rad * (1.0 + _EDGE_FEATHER)
    if px + reach < 0 or py + reach < 0 or px - reach > w or py - reach > h:
        return None
    return px, py, rad


def _window(px: float, py: float, rad: float, h: int, w: int
            ) -> tuple[int, int, int, int]:
    """Pixel bounds of the disc plus its feather, clipped to the array. This is
    the reason cost scales with the eye and not the frame."""
    reach = rad * (1.0 + _EDGE_FEATHER) + 1.0
    x0 = max(0, int(np.floor(px - reach)))
    y0 = max(0, int(np.floor(py - reach)))
    x1 = min(w, int(np.ceil(px + reach)))
    y1 = min(h, int(np.ceil(py + reach)))
    return x0, y0, x1, y1


def _radial(hh: int, ww: int, cx: float, cy: float, rad: float,
            feather: float) -> np.ndarray:
    """Feathered disc alpha on pixel centres, 1 in the core, 0 outside."""
    ys = ((np.arange(hh, dtype=np.float32) + 0.5) - cy)[:, None]
    xs = ((np.arange(ww, dtype=np.float32) + 0.5) - cx)[None, :]
    dist = np.sqrt(xs * xs + ys * ys) / rad
    return _smoothstep(np.clip((1.0 - dist) / feather, 0.0, 1.0))


# ----- the two corrections ------------------------------------------------

def _correct_red(win: np.ndarray, disc: np.ndarray,
                 c: dict[str, Any]) -> np.ndarray:
    """Desaturate and darken flash return, keeping the catchlight.

    The weight is the product of three things, and each one is doing a job:
    the disc says where the user pointed, the redness gate says which of those
    pixels the flash actually contaminated (so an iris and lashes inside the
    circle survive), and the specular guard says which of *those* are the
    highlight that must not be touched.
    """
    r, g, b = win[..., 0], win[..., 1], win[..., 2]
    excess = r - np.maximum(g, b)
    redness = _smoothstep(np.clip((excess - _RED_LO) / (_RED_HI - _RED_LO),
                                 0.0, 1.0))
    achromatic = np.minimum(np.minimum(r, g), b)
    spec = _smoothstep(np.clip((achromatic - _SPEC_LO) / (_SPEC_HI - _SPEC_LO),
                               0.0, 1.0))
    weight = disc * redness * (1.0 - spec) * (c["amount"] / 100.0)
    # Green and blue were not contaminated, so they hold what the pupil's real
    # level was. Taking the lower of the two lands on a true neutral rather than
    # on grey with a red cast left in it.
    base = np.minimum(g, b) * (1.0 - _DARKEN_MAX * c["darken"] / 100.0)
    return win * (1.0 - weight[..., None]) + base[..., None] * weight[..., None]


def _correct_pet(win: np.ndarray, disc: np.ndarray, c: dict[str, Any],
                 cx: float, cy: float, rad: float) -> np.ndarray:
    """Neutralise a tapetal reflection of any hue, then draw a catchlight.

    No specular guard here, on purpose: on an animal the bright part *is* the
    thing being removed. Which is also why the highlight has to be put back by
    hand afterwards.

    The luminance band is absolute rather than measured off this array. Adapting
    it to what is in the window would read differently in a 1:1 view than in the
    export, and a repair that changes when you zoom is worse than one that needs
    `amount` nudged on a dark frame.
    """
    lum = _luma(win)
    bright = _smoothstep(np.clip((lum - _PET_LO) / (_PET_HI - _PET_LO),
                                0.0, 1.0))
    amount = c["amount"] / 100.0
    weight = disc * bright * amount
    # Straight to achromatic: a green or yellow reflection carries no colour
    # information about the eye, so there is nothing to preserve.
    base = lum * (1.0 - _DARKEN_MAX * c["darken"] / 100.0)
    out = win * (1.0 - weight[..., None]) + base[..., None] * weight[..., None]
    if not c["catchlight"]:
        return out
    hh, ww = win.shape[:2]
    glow = _radial(hh, ww,
                   cx + c["cat_x"] * rad, cy + c["cat_y"] * rad,
                   max(0.8, rad * c["cat_size"] / 100.0), _CAT_FEATHER)
    confine = _radial(hh, ww, cx, cy, rad, _CAT_CONFINE)
    k = (glow * confine * amount)[..., None]
    return out * (1.0 - k) + _CAT_LEVEL * k


def apply_redeye(rgb: np.ndarray, params: dict[str, Any] | None,
                 roi: tuple[float, float, float, float] = FULL_ROI
                 ) -> np.ndarray:
    """Repair every correction in `params`. float32 RGB [0,1] in and out.

    `roi` is (x0, y0, w, h) in normalized coordinates saying where `rgb` sits
    inside the whole photo, exactly as `editing.render` uses it: pass it when
    correcting a window so the discs land where they would in the full frame.

    Neutral parameters return the input array unchanged, with no copy.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected RGB, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"expected float32, got {rgb.dtype}"
    assert len(roi) == 4 and roi[2] > 0 and roi[3] > 0, f"bad roi {roi}"
    p = normalize(params)
    if p is None:
        return rgb
    h, w = rgb.shape[:2]
    out: np.ndarray | None = None
    for c in p["corrections"]:
        placed = _place(c, (h, w), roi)
        if placed is None:
            continue        # this eye is outside the window being rendered
        px, py, rad = placed
        x0, y0, x1, y1 = _window(px, py, rad, h, w)
        if x1 <= x0 or y1 <= y0:
            continue
        if out is None:
            out = rgb.copy()
        win = out[y0:y1, x0:x1]
        lx, ly = px - x0, py - y0
        disc = _radial(win.shape[0], win.shape[1], lx, ly, rad, _EDGE_FEATHER)
        if c["kind"] == "red":
            fixed = _correct_red(win, disc, c)
        else:
            fixed = _correct_pet(win, disc, c, lx, ly, rad)
        out[y0:y1, x0:x1] = np.clip(fixed, 0.0, 1.0)
    return rgb if out is None else out


def padding(params: dict[str, Any] | None, w: int, h: int) -> float:
    """Always 0.0: a window render needs no surrounding image at all.

    Stated explicitly rather than left to be inferred, since the caller folds
    this into `editing.effect_padding` alongside stages that genuinely do reach
    outside their window. Every pixel this module writes is a function of that
    pixel's own value and of the disc's position and radius, both of which are
    held in whole-frame normalized coordinates and reconstructed from `roi`.
    Nothing is convolved, nothing is sampled from a neighbour, and no threshold
    or reference is measured off the array that was handed over — that last part
    is a deliberate constraint, and it is why a disc straddling the window edge
    comes out identical to the same pixels of the full-frame render.

    `params`, `w` and `h` are accepted to match the signature the other stages
    use, and to leave room to answer differently if this ever grows a stage that
    reads a neighbourhood. Nothing is read from them today.
    """
    assert w > 0 and h > 0, f"bad frame size {w}x{h}"
    return 0.0


# ----- verification -------------------------------------------------------
#
# One function decides whether a candidate circle really holds a red pupil, and
# it is used by both the automatic pass and the user-drawn one. Everything it
# measures is a *relation* between a blob and its surroundings, because that is
# the only thing that separates an eye from a red object: a pupil's glow is
# bounded by iris and sclera, and a jumper, a lip and a tail light are not.

_DETECT_EXCESS = 0.15     # red-over-green/blue for a pixel to join the blob
_DETECT_PET_LUM = 0.40    # luma for a pixel to join a tapetal reflection
_BLOB_NEAR = 1.20         # blob centroid must sit this close in, in radii
_RING_IN, _RING_OUT = 1.40, 2.60
_PATCH_REACH = 2.80       # half-side of the patch, in radii
_MIN_FILL, _MAX_FILL = 0.004, 0.30
_MIN_CIRCULARITY = 0.45
_ASPECT_LO, _ASPECT_HI = 0.45, 2.20
_MAX_EXCESS = 0.88        # purer red than any retina returns: a light source
_RING_RATIO = 0.40        # blob redness must beat the ring by this much
_MAX_RING_EXCESS = 0.22   # skin is mildly red as well; this is above it, not below
_MIN_SCLERA = 0.06        # fraction of the ring that must look like an eye


def _patch(rgb: np.ndarray, px: float, py: float, rad: float
           ) -> tuple[np.ndarray, float, float, int, int] | None:
    """A square cut around a candidate, with the candidate's centre in it, its
    centre in patch coordinates and the patch's origin to get back out."""
    h, w = rgb.shape[:2]
    reach = int(np.ceil(rad * _PATCH_REACH))
    x0, y0 = int(round(px)) - reach, int(round(py)) - reach
    x1, y1 = x0 + 2 * reach, y0 + 2 * reach
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None    # not enough surround to judge it by; refuse to guess
    return rgb[y0:y1, x0:x1], px - x0, py - y0, x0, y0


def _blob(mask: np.ndarray, cx: float, cy: float, rad: float
          ) -> tuple[np.ndarray, tuple[float, float, float]] | None:
    """The connected component *under* the candidate centre, with its centroid
    and equivalent radius. None when nothing plausible is there.

    Taking the component the centre lands on, rather than the one with the
    nearest centroid, matters more than it looks: skin is mildly red too, so the
    mask usually also holds a large ring of eyelid and cheek whose centroid sits
    almost exactly on the pupil. Picking by centroid would grab that ring.
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    cxi = int(np.clip(round(cx), 0, mask.shape[1] - 1))
    cyi = int(np.clip(round(cy), 0, mask.shape[0] - 1))
    best = int(labels[cyi, cxi])
    if best == 0:
        # The click can miss the pupil by a little; look just far enough around
        # it to find one, and no further.
        rr = max(1, int(np.ceil(0.6 * rad)))
        y0, y1 = max(0, cyi - rr), min(mask.shape[0], cyi + rr + 1)
        x0, x1 = max(0, cxi - rr), min(mask.shape[1], cxi + rr + 1)
        sub = labels[y0:y1, x0:x1]
        nz = np.argwhere(sub > 0)
        if nz.size == 0:
            return None
        d = (nz[:, 0] + y0 - cy) ** 2 + (nz[:, 1] + x0 - cx) ** 2
        pick = nz[int(np.argmin(d))]
        best = int(sub[pick[0], pick[1]])
    if float(np.hypot(centroids[best][0] - cx, centroids[best][1] - cy)) > _BLOB_NEAR * rad:
        return None
    area = float(stats[best, cv2.CC_STAT_AREA])
    bw = float(stats[best, cv2.CC_STAT_WIDTH])
    bh = float(stats[best, cv2.CC_STAT_HEIGHT])
    if bw < 1 or bh < 1 or not _ASPECT_LO <= bw / bh <= _ASPECT_HI:
        return None
    sel = labels == best
    contours, _ = cv2.findContours(sel.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    perim = max(1e-6, float(cv2.arcLength(contours[0], True)))
    if 4.0 * np.pi * area / (perim * perim) < _MIN_CIRCULARITY:
        return None
    bx, by = centroids[best]
    return sel, (float(bx), float(by), float(np.sqrt(area / np.pi)))


def _ring(shape: tuple[int, int], cx: float, cy: float, rad: float
          ) -> np.ndarray:
    ys = ((np.arange(shape[0], dtype=np.float32) + 0.5) - cy)[:, None]
    xs = ((np.arange(shape[1], dtype=np.float32) + 0.5) - cx)[None, :]
    d = np.sqrt(xs * xs + ys * ys) / rad
    return (d >= _RING_IN) & (d <= _RING_OUT)


def _verify_red(rgb: np.ndarray, px: float, py: float, rad: float
                ) -> tuple[float, float, float] | None:
    """Is there a red-eye pupil at this candidate? Returns a tightened
    (px, py, radius) or None.

    The tests, in the order they earn their keep:
      1. the red blob is compact, roughly round, and small against its patch —
         a jumper fills the patch, lips are wide, neither is round;
      2. the surround is *not* red — this is what a red object cannot fake, and
         what a pupil always satisfies;
      3. the red is not purer than a retina can return — a tail light or an LED
         clips one channel and leaves the other two at nothing;
      4. some of the surround looks like an eye: bright and near-neutral, i.e.
         sclera. A red blob on dark bodywork fails here.
    None of this is sufficient on its own; see the module docstring on why the
    geometric gate in `detect` is the part that matters.
    """
    cut = _patch(rgb, px, py, rad)
    if cut is None:
        return None
    patch, cx, cy, ox, oy = cut
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    excess = r - np.maximum(g, b)
    found = _blob(excess > _DETECT_EXCESS, cx, cy, rad)
    if found is None:
        return None
    sel, (bx, by, brad) = found
    fill = float(sel.mean())
    if not _MIN_FILL <= fill <= _MAX_FILL:
        return None
    blob_excess = float(excess[sel].mean())
    # The floor is the point where the correction itself starts to bite: below
    # it there would be nothing to fix even if this were an eye.
    if blob_excess < _RED_HI or blob_excess > _MAX_EXCESS:
        return None
    ring = _ring(patch.shape[:2], bx, by, brad)
    if ring.sum() < 16:
        return None
    ring_excess = float(np.clip(excess[ring], 0.0, None).mean())
    if ring_excess > _MAX_RING_EXCESS or ring_excess > _RING_RATIO * blob_excess:
        return None
    lum = _luma(patch)
    chroma = patch.max(axis=2) - patch.min(axis=2)
    sclera = float(((lum > 0.30) & (chroma < 0.22))[ring].mean())
    if sclera < _MIN_SCLERA:
        return None
    return bx + ox, by + oy, brad


def _verify_pet(rgb: np.ndarray, px: float, py: float, rad: float
                ) -> tuple[float, float, float] | None:
    """Is there a tapetal reflection at this candidate? Returns a tightened
    (px, py, radius) or None.

    Hue is deliberately not tested: the return can be green, yellow, orange or
    blown white. What is tested is that a bright blob sits inside a much darker
    surround, which is what an eye in a face looks like and what a bright object
    on a bright background does not.
    """
    cut = _patch(rgb, px, py, rad)
    if cut is None:
        return None
    patch, cx, cy, ox, oy = cut
    lum = _luma(patch)
    found = _blob(lum > _DETECT_PET_LUM, cx, cy, rad)
    if found is None:
        return None
    sel, (bx, by, brad) = found
    fill = float(sel.mean())
    if not _MIN_FILL <= fill <= _MAX_FILL:
        return None
    ring = _ring(patch.shape[:2], bx, by, brad)
    if ring.sum() < 16:
        return None
    blob_lum = float(lum[sel].mean())
    # The eye has to be darker around the reflection than the reflection is,
    # by a clear margin. A white shirt or a bright sky fails this.
    if float(lum[ring].mean()) > 0.60 * blob_lum:
        return None
    return bx + ox, by + oy, brad


_VERIFIERS = {"red": _verify_red, "pet": _verify_pet}


def _as_correction(rgb: np.ndarray, kind: str,
                   found: tuple[float, float, float]) -> dict[str, Any]:
    """Pixel findings back into the normalized schema."""
    h, w = rgb.shape[:2]
    bx, by, brad = found
    return {
        "kind": kind,
        "cx": float(bx / w),
        "cy": float(by / h),
        # A shade wider than the blob, so the feather starts outside the glow
        # rather than through it.
        "r": float(min(_MAX_RADIUS, max(_MIN_RADIUS, brad * 1.25 / w))),
        "amount": 100,
        "darken": 50 if kind == "red" else 70,
        "catchlight": kind == "pet",
        "cat_x": DEFAULT_CORRECTION["cat_x"],
        "cat_y": DEFAULT_CORRECTION["cat_y"],
        "cat_size": DEFAULT_CORRECTION["cat_size"],
    }


def detect_in_region(rgb: np.ndarray, cx: float, cy: float, r: float,
                     kind: str = "red") -> dict[str, Any] | None:
    """Verify and tighten a circle the user drew, in normalized coordinates.

    This is the only path available for pet eye (see the module docstring), and
    it doubles as the honest test of the verifier: a user *can* draw a circle
    over a red jumper, and the answer must still be None.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected RGB, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"expected float32, got {rgb.dtype}"
    assert kind in KINDS, f"unknown kind {kind!r}"
    h, w = rgb.shape[:2]
    found = _VERIFIERS[kind](rgb, cx * w, cy * h, max(1.5, r * w))
    if found is None:
        return None
    return _as_correction(rgb, kind, found)


# ----- eye localisation ---------------------------------------------------

_HAS_MEDIAPIPE = importlib.util.find_spec("mediapipe") is not None
_DETECT_LONG_EDGE = 1280   # what scoring/eyes.py already works at
_CASCADES: dict[str, cv2.CascadeClassifier] = {}


def _cascade(name: str) -> cv2.CascadeClassifier:
    if name not in _CASCADES:
        path = cv2.data.haarcascades + name
        clf = cv2.CascadeClassifier(path)
        assert not clf.empty(), f"opencv is missing its bundled cascade {path}"
        _CASCADES[name] = clf
    return _CASCADES[name]


def _mesh_eyes(rgb_u8: np.ndarray) -> list[tuple[float, float, float]]:
    """Eye centres from the FaceMesh landmarks `scoring/eyes.py` already loads.

    That module's singleton is reused rather than a second FaceMesh built here:
    the model costs real memory and a second copy would be loaded on any machine
    that also scores eye-openness. Its six indices per eye are the corners and
    lids, so the span between the two corners gives both the centre and a
    sensible pupil radius without needing the iris refinement.
    """
    from .scoring import eyes as eyes_mod
    result = eyes_mod._get_face_mesh().process(rgb_u8)
    if not result.multi_face_landmarks:
        return []
    h, w = rgb_u8.shape[:2]
    out = []
    for face in result.multi_face_landmarks:
        for idx in (eyes_mod.LEFT_EYE, eyes_mod.RIGHT_EYE):
            lm = face.landmark
            x0, y0 = lm[idx[0]].x * w, lm[idx[0]].y * h
            x1, y1 = lm[idx[3]].x * w, lm[idx[3]].y * h
            span = float(np.hypot(x1 - x0, y1 - y0))
            if span < 6.0:
                continue      # an eye this small has no pupil to correct
            out.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, span * 0.22))
    return out


def _cascade_eyes(gray: np.ndarray) -> list[tuple[float, float, float]]:
    """Eye boxes from the frontal-face and eye cascades bundled with OpenCV.

    Weaker than FaceMesh in every way — frontal, unoccluded, large faces only —
    but it needs no optional dependency and no download, and a proposal that is
    wrong is thrown out by the verifier. Eyes are searched only in the upper
    part of a face box, which is what stops the eye cascade matching nostrils
    and the corners of a mouth.
    """
    faces = _cascade("haarcascade_frontalface_default.xml").detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(64, 64))
    out = []
    for fx, fy, fw, fh in faces:
        band = gray[fy:fy + int(fh * 0.62), fx:fx + fw]
        eye_min = max(12, int(fw * 0.10))
        eyes = _cascade("haarcascade_eye.xml").detectMultiScale(
            band, scaleFactor=1.15, minNeighbors=6,
            minSize=(eye_min, eye_min))
        for ex, ey, ew, eh in eyes:
            out.append((float(fx + ex + ew / 2.0), float(fy + ey + eh / 2.0),
                        float(ew) * 0.25))
    return out


def _eye_candidates(rgb_u8: np.ndarray) -> list[tuple[float, float, float]]:
    """Propose eyes as (x, y, radius) in `rgb_u8` pixels — the one seam between
    a learned model and the rest of this module. Whatever comes out of here is
    only ever a proposal; `_verify_red` decides."""
    if _HAS_MEDIAPIPE:
        return _mesh_eyes(rgb_u8)
    return _cascade_eyes(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY))


def detect(rgb: np.ndarray) -> list[dict[str, Any]]:
    """Find human red-eye automatically. float32 RGB [0,1] in, corrections out.

    Deterministic: no sampling, no randomness, and the result is ordered by
    position so the same photo always yields the same list.

    Returns [] when no eye is located, which is the common case for anything
    that is merely red — see the module docstring for what this can and cannot
    do, and note that pets are never detected here.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected RGB, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"expected float32, got {rgb.dtype}"
    h, w = rgb.shape[:2]
    u8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    scale = min(1.0, _DETECT_LONG_EDGE / max(h, w))
    small = u8 if scale >= 1.0 else cv2.resize(
        u8, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    cands = _eye_candidates(small)

    inv = 1.0 / scale
    out = []
    for px, py, rad in cands:
        found = _verify_red(rgb, px * inv, py * inv, max(1.5, rad * inv))
        if found is not None:
            out.append(_as_correction(rgb, "red", found))
    out.sort(key=lambda c: (round(c["cy"], 4), round(c["cx"], 4)))
    return out[:CORRECTION_MAX]
