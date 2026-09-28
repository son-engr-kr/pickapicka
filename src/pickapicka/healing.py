"""Content removal that carries no licence with it: heal, clone and spot.

Three tools, all built on OpenCV's own algorithms and the arithmetic in this
file. Nothing here downloads a model or borrows weights, so an MIT-licensed
build stays MIT and the photographer using it inherits no restriction. That is
the whole point of this module: it is the *default* path of the removal feature
(see `docs/roadmap-parity.md`, "Decisions taken"), and a heavier model behind a
consent dialog is somebody else's file.

  heal   fill a region with what surrounds it (`cv2.inpaint`)
  clone  copy from an explicit source offset, through a feathered edge
  spot   a one-click circular heal, for dust and blemishes

Where this approach fails, plainly
----------------------------------
`cv2.inpaint` is a *diffusion* method: it walks inward from the hole's rim and
extends the colour and the gradient it finds there. It has no idea what a brick
wall or a head of hair looks like, and it cannot invent one. So:

  - Dust, sensor spots, lint, skin blemishes, a stray hair, a power line against
    a sky, a sign in a smooth wall — excellent. Indistinguishable in practice.
  - A region crossing a strong edge — usable up to a couple of dozen pixels; the
    edge is extended straight, which is right more often than not.
  - Anything asked to reproduce *texture* (fabric, foliage, gravel, hair) over
    more than a few pixels — it smears. It produces a soft, directional blur
    that reads as a smudge, and no parameter fixes that; it is what the
    algorithm is. Our own measurements: RMSE in the hole is ~0.005 on a smooth
    gradient and ~0.12 on a fine woven texture at the same 45 px radius, a
    twenty-fold difference.
  - A large region over anything at all — same story, worse.

What to reach for instead, in order: `clone`, which copies real texture from a
place the user picks and is the honest answer whenever heal smears; then the
opt-in large-region model on the roadmap, which is the only thing that can
*invent* plausible texture and is deliberately not the default because its
licence would follow the user.

Notes for the caller
--------------------
  - Healing runs BEFORE the parametric grade. A heal repairs the captured image;
    the grade is an interpretation of it. Sharpening a smudge or pushing a clone
    seam through a contrast curve is one thing; healing pixels that a curve has
    already crushed is another, and the second one bakes the grade into the
    repair. So: heal, then `editing._grade`, then the masks.
  - Everything is stored in normalized [0,1] frame coordinates, exactly as
    `editing`'s brush strokes are, so one heal means the same repair on a
    thumbnail, on the 1:1 preview and on the full-resolution export.
  - Cost scales with the *region*, never with the image: each operation touches
    its own bounding box plus a small margin. A 200x200 heal costs the same
    8 ms on a 1 MP file and on a 36 MP one.
  - Deterministic by construction — no randomness anywhere — because the caller
    caches renders by parameter hash.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ----- schema -------------------------------------------------------------
#
# A healing parameter block is a list of operations applied in order:
#
#   {"ops": [{"kind": "spot", "points": [[0.31, 0.22]], "radius": 0.004,
#             "feather": 50, "opacity": 100, "method": "ns", "enabled": True},
#            {"kind": "clone", "points": [[0.5, 0.5], [0.55, 0.52]],
#             "radius": 0.02, "dx": 0.08, "dy": -0.03, "feather": 60, ...}]}
#
#   kind      "heal" | "clone" | "spot"
#   points    the region's path in normalized frame coordinates (x a fraction of
#             the width, y of the height); a "spot" carries exactly one
#   radius    the brush radius, a fraction of the frame WIDTH — so the region is
#             a circle in *pixels*, which is what `editing._brush_alpha` does too
#   dx, dy    clone only: where to read from, as a normalized frame offset
#   feather   0..100, how wide the edge ramp is relative to the radius
#   opacity   0..100, how much of the repair is blended in
#   method    "ns" | "telea", the diffusion used by heal and spot
#   enabled   a switched-off operation is kept (the UI toggles it) but does nothing

FULL_ROI = (0.0, 0.0, 1.0, 1.0)

KINDS: tuple[str, ...] = ("heal", "clone", "spot")

# Navier-Stokes is the default. Measured against ground truth over four
# synthetic backgrounds (smooth gradient, fine weave, hard edge, skin) at seven
# hole sizes from 2 to 45 px: NS wins on the mean at every inpaint radius, and
# it is *flat* in that radius (0.0298-0.0312 RMSE) where TELEA degrades as the
# radius grows (0.0321-0.0340). TELEA is faster on a whole frame, which we never
# do, and it wins only on a few large regions where both are already unusable.
# So NS by default, TELEA kept selectable because on some real edges it is
# crisper and this is cheap enough to offer.
METHODS: dict[str, int] = {"ns": cv2.INPAINT_NS, "telea": cv2.INPAINT_TELEA}

OP_MAX = 200               # healing operations per photo
OP_POINTS_MAX = 4000       # points in one operation's path (as STROKE_POINTS_MAX)

# Dust is small: 0.0005 of the width is 3 px on a 6000 px file, and a spot tool
# that cannot be that small is not a spot tool. The ceiling is the licence
# decision made concrete — a quarter of the frame width is far past where
# diffusion stops working, and larger removals are the opt-in model's job.
_RADIUS_RANGE = (0.0005, 0.25)
_OFFSET_RANGE = (-1.0, 1.0)     # a clone may read from anywhere in the frame

_FEATHER_FRAC = 0.5        # ramp width at feather=100, as a fraction of the radius
_INPAINT_RADIUS = 3        # px of rim the diffusion reads; OpenCV's own advice, and
                           # NS is insensitive to it (see METHODS)
# Pixels of real image kept around the region so the windowed inpaint gives bit
# -identical results to a whole-frame one. Both OpenCV inpainters march inward
# from the rim reading a neighbourhood of `_INPAINT_RADIUS`, so the influence
# reaches at most that far outside the hole. Measured: 4 px of margin already
# agrees to within 1/255 and 8 px agrees exactly, at hole radii 3, 12 and 40.
# Twice the radius plus four is 10 — cheap, and past where it stops mattering.
_INPAINT_MARGIN = 2 * _INPAINT_RADIUS + 4

_EPS = 1e-4

DEFAULT_HEALING: dict[str, Any] = {"ops": []}


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    """Clamp one number out of untrusted JSON (same coercion as `editing._fnum`)."""
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite ease on an already-clipped [0,1] ramp — no hard edge on the falloff."""
    return t * t * (3.0 - 2.0 * t)


# ----- normalization ------------------------------------------------------

def normalize_op(raw: Any) -> dict[str, Any] | None:
    """Coerce one operation into the canonical shape, or None when it cannot
    change a pixel (unknown kind, no path, zero opacity, a clone reading from
    where it writes)."""
    if not isinstance(raw, dict):
        return None
    kind = raw.get("kind")
    if kind not in KINDS:
        return None

    pts: list[list[float]] = []
    for p in (raw.get("points") or [])[:OP_POINTS_MAX]:
        if isinstance(p, (list, tuple)) and len(p) == 2:
            # A region may hang off the frame; only its overlap gets repaired.
            pts.append([_fnum(p[0], -1.0, 2.0, 0.0), _fnum(p[1], -1.0, 2.0, 0.0)])
    if kind == "spot":
        pts = pts[:1]      # a spot is a circle, and a circle has one centre
    if not pts:
        return None

    rlo, rhi = _RADIUS_RANGE
    op: dict[str, Any] = {
        "kind": kind,
        "enabled": bool(raw.get("enabled", True)),
        "points": pts,
        "radius": _fnum(raw.get("radius"), rlo, rhi, 0.01),
        "feather": int(round(_fnum(raw.get("feather"), 0, 100, 50))),
        "opacity": int(round(_fnum(raw.get("opacity"), 0, 100, 100))),
        "method": raw["method"] if raw.get("method") in METHODS else "ns",
        "dx": 0.0,
        "dy": 0.0,
    }
    if op["opacity"] == 0:
        return None
    if kind == "clone":
        op["dx"] = _fnum(raw.get("dx"), _OFFSET_RANGE[0], _OFFSET_RANGE[1], 0.0)
        op["dy"] = _fnum(raw.get("dy"), _OFFSET_RANGE[0], _OFFSET_RANGE[1], 0.0)
        if abs(op["dx"]) < _EPS and abs(op["dy"]) < _EPS:
            return None    # copying a region onto itself is not an edit
    return op


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp a healing block and drop the operations that would do nothing.

    Returns None when nothing usable is left, which is what keeps an opened-but
    -untouched heal panel from counting as an edit. A bare list of operations is
    accepted as a convenience for callers holding just that.
    """
    if isinstance(raw, (list, tuple)):
        raw = {"ops": raw}
    if not isinstance(raw, dict):
        return None
    ops: list[dict[str, Any]] = []
    for item in (raw.get("ops") or [])[:OP_MAX]:
        op = normalize_op(item)
        if op is not None:
            ops.append(op)
    return {"ops": ops} if ops else None


def op_is_active(op: dict[str, Any]) -> bool:
    """True when a normalized operation actually changes pixels."""
    return bool(op["enabled"]) and op["opacity"] > 0


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when `params` (normalized or not) leaves pixels unchanged."""
    p = normalize(params)
    return p is None or not any(op_is_active(op) for op in p["ops"])


# ----- geometry -----------------------------------------------------------

def _op_geometry(op: dict[str, Any], frame_w: float) -> tuple[float, float]:
    """An operation's brush radius and edge collar, in pixels of the full frame.

    The collar is the band *outside* the drawn region across which the repair
    fades back to the original. It is always at least a pixel wide: a heal joins
    its surroundings by construction, but a clone with a truly hard edge shows
    every rounding step of the rasterized circle.
    """
    r_px = max(0.5, op["radius"] * frame_w)
    return r_px, max(1.0, op["feather"] / 100.0 * _FEATHER_FRAC * r_px)


def _pixel_path(op: dict[str, Any], frame_w: float, frame_h: float,
                ox: float, oy: float) -> np.ndarray:
    """The operation's path in the pixel coordinates of a window whose top-left
    corner sits at (ox, oy) in a frame_w x frame_h frame."""
    return np.asarray([[p[0] * frame_w - ox, p[1] * frame_h - oy]
                       for p in op["points"]], dtype=np.float64)


def _reach_bbox(path: np.ndarray, reach: float,
                w: int, h: int) -> tuple[int, int, int, int]:
    """Integer box containing every pixel within `reach` of `path`, clipped to
    a w x h array. Returned as (x0, y0, x1, y1) with x1/y1 exclusive."""
    x0 = max(0, int(math.floor(path[:, 0].min() - reach)))
    y0 = max(0, int(math.floor(path[:, 1].min() - reach)))
    x1 = min(w, int(math.ceil(path[:, 0].max() + reach)) + 1)
    y1 = min(h, int(math.ceil(path[:, 1].max() + reach)) + 1)
    return x0, y0, max(x0, x1), max(y0, y1)


def region_bbox(op: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int]:
    """The box of pixels an operation *writes*, in a w x h frame.

    Returned as (x0, y0, x1, y1) with the far corner exclusive, clipped to the
    frame, so it slices an array directly. An empty box (x1 == x0) means the
    operation misses the frame entirely. The read margin the diffusion needs is
    not included — see `read_bbox` for that.
    """
    r_px, collar = _op_geometry(op, float(w))
    return _reach_bbox(_pixel_path(op, float(w), float(h), 0.0, 0.0),
                       r_px + collar + 0.5, w, h)


def read_bbox(op: dict[str, Any], w: int, h: int) -> tuple[int, int, int, int]:
    """The box of pixels an operation *reads*: its written box, the margin the
    diffusion needs around it, and for a clone the source box as well.

    This is what a caller rendering a window has to include for the window to
    match the full-frame render. Same (x0, y0, x1, y1) convention.
    """
    x0, y0, x1, y1 = region_bbox(op, w, h)
    if x1 <= x0:
        return x0, y0, x1, y1
    x0, y0 = max(0, x0 - _INPAINT_MARGIN), max(0, y0 - _INPAINT_MARGIN)
    x1, y1 = min(w, x1 + _INPAINT_MARGIN), min(h, y1 + _INPAINT_MARGIN)
    if op["kind"] == "clone":
        dx, dy = int(round(op["dx"] * w)), int(round(op["dy"] * h))
        x0, y0 = max(0, min(x0, x0 + dx)), max(0, min(y0, y0 + dy))
        x1, y1 = min(w, max(x1, x1 + dx)), min(h, max(y1, y1 + dy))
    return x0, y0, x1, y1


def padding(params: dict[str, Any] | None, w: int, h: int) -> float:
    """Pixels of surrounding image a windowed render has to be handed so a heal
    near the window's edge comes out the same as it does in the whole-frame one.

    Mirrors `editing.effect_padding`, and exists for the same reason: the 1:1
    editor view renders a patch of the original, and an operation whose read
    region crosses the patch boundary would otherwise diffuse from pixels that
    were never passed in — so the preview and the export would disagree. 0.0
    when no operation is active.

    Deliberately *not* `read_bbox` minus `region_bbox`. Both of those are clipped
    to the frame, so an operation sitting against the left edge would report
    needing nothing, while the number returned here is used to widen the window
    on every side. `test_padding_agrees_with_read_bbox` ties the two together
    where the clip does not bite.
    """
    p = normalize(params)
    if p is None:
        return 0.0
    pad = 0.0
    for op in p["ops"]:
        if not op_is_active(op):
            continue
        # The diffusion reads `_INPAINT_MARGIN` beyond the region it repairs. A
        # clone additionally reads a whole region-sized patch at its source
        # offset; strictly it needs no margin on top of that, but `_apply_op`
        # forms one box for both kinds and ten generous pixels cost nothing.
        reach = float(_INPAINT_MARGIN)
        if op["kind"] == "clone":
            reach += max(abs(op["dx"]) * w, abs(op["dy"]) * h)
        pad = max(pad, reach)
    return pad


# ----- the region mask ----------------------------------------------------

def _rasterize(layer: np.ndarray, path: np.ndarray, r_px: float) -> None:
    """Stamp the region into a binary `layer`: a thick polyline plus a disc at
    every sample. The polyline alone has flat ends and no join at a corner, so
    the discs are what give a dragged heal round caps — the same construction
    `editing._brush_alpha` uses, and for the same reason. Samples closer than a
    third of the radius add nothing the polyline has not covered, and a wide
    brush over a long drag is thousands of them, so those are skipped.

    Nothing is anti-aliased here on purpose: `cv2.inpaint` wants a hard yes/no
    about every pixel, and the soft edge comes from the distance ramp below.
    """
    ip = np.rint(path).astype(np.int32)
    r = max(1, int(round(r_px)))
    if len(ip) > 1:
        cv2.polylines(layer, [ip], False, 255, thickness=2 * r)
    step = max(1.0, r * 0.33)
    last: tuple[int, int] | None = None
    for px, py in ip:
        if last is not None and abs(px - last[0]) + abs(py - last[1]) < step:
            continue
        last = (int(px), int(py))
        cv2.circle(layer, last, r, 255, -1)
    cv2.circle(layer, (int(ip[-1][0]), int(ip[-1][1])), r, 255, -1)


def _fill_and_alpha(core: np.ndarray, collar: float) -> tuple[np.ndarray, np.ndarray]:
    """From the drawn region, the mask to repair and the alpha to blend it with.

    One exact Euclidean distance transform does both jobs. `dist` is 0 inside
    the region and grows outward, so:

      - the repaired mask is everything within `collar` of the region. The
        region is *dilated* rather than used as drawn, deliberately: the alpha
        has to reach full strength at the drawn edge or a rim of the dust the
        user painted over survives the blend, which is the one artefact nobody
        forgives in a spot tool.
      - the alpha is 1 across the drawn region and eases to 0 half a pixel past
        the collar, smoothstepped so there is no crease where the ramp starts.
    """
    dist = cv2.distanceTransform(255 - core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fill = (dist <= collar).astype(np.uint8)
    alpha = _smoothstep(np.clip((collar + 0.5 - dist) / collar, 0.0, 1.0))
    return fill, alpha


# ----- the three tools ----------------------------------------------------

def _shifted_read(img: np.ndarray, box: tuple[int, int, int, int],
                  dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Read `box` from `img` displaced by (dy, dx), with a validity mask.

    Where the source falls outside `img` the result is left at zero and marked
    invalid, and the caller multiplies the alpha by that: a clone copies what is
    actually there and never mirrors or stretches the edge to cover a gap. The
    read is a copy, so an overlapping clone still sees the pixels as they were
    before this operation wrote anything.
    """
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0, img.shape[2]), dtype=img.dtype)
    valid = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    sy0, sx0 = max(0, y0 + dy), max(0, x0 + dx)
    sy1, sx1 = min(h, y1 + dy), min(w, x1 + dx)
    if sy1 <= sy0 or sx1 <= sx0:
        return out, valid
    ty, tx = sy0 - dy - y0, sx0 - dx - x0
    out[ty:ty + sy1 - sy0, tx:tx + sx1 - sx0] = img[sy0:sy1, sx0:sx1]
    valid[ty:ty + sy1 - sy0, tx:tx + sx1 - sx0] = 1.0
    return out, valid


def _apply_op(img: np.ndarray, op: dict[str, Any], frame_w: float, frame_h: float,
              ox: float, oy: float) -> None:
    """Apply one operation to `img` in place. `img` is the window; (ox, oy) is
    where its top-left corner sits in the frame_w x frame_h frame."""
    h, w = img.shape[:2]
    r_px, collar = _op_geometry(op, frame_w)
    path = _pixel_path(op, frame_w, frame_h, ox, oy)

    # The write box, then the read box: real pixels around the region for the
    # diffusion to march in from. Clipped to the window, so an operation near
    # its edge simply gets less context — that is the cost of rendering a crop,
    # and `read_bbox` tells a caller how to avoid paying it.
    wx0, wy0, wx1, wy1 = _reach_bbox(path, r_px + collar + 0.5, w, h)
    if wx1 <= wx0 or wy1 <= wy0:
        return                                  # this operation misses the window
    box = (max(0, wx0 - _INPAINT_MARGIN), max(0, wy0 - _INPAINT_MARGIN),
           min(w, wx1 + _INPAINT_MARGIN), min(h, wy1 + _INPAINT_MARGIN))
    x0, y0, x1, y1 = box

    core = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    _rasterize(core, path - np.asarray([x0, y0], dtype=np.float64), r_px)
    if not core.any():
        return                                  # the region rounded away to nothing
    fill, alpha = _fill_and_alpha(core, collar)
    if op["opacity"] != 100:
        alpha = alpha * (op["opacity"] / 100.0)

    patch = img[y0:y1, x0:x1]
    if op["kind"] == "clone":
        src, valid = _shifted_read(img, box, int(round(op["dy"] * frame_h)),
                                   int(round(op["dx"] * frame_w)))
        alpha = alpha * valid
    else:
        # cv2.inpaint takes 8-bit only. The round trip is confined to the filled
        # pixels — everything else keeps its float value untouched — and the
        # caller's source is an 8-bit frame anyway, so nothing is actually lost.
        u8 = np.rint(np.clip(patch, 0.0, 1.0) * 255.0).astype(np.uint8)
        filled = cv2.inpaint(u8, fill, _INPAINT_RADIUS, METHODS[op["method"]])
        src = np.where(fill[..., None] > 0, filled.astype(np.float32) / 255.0, patch)
    img[y0:y1, x0:x1] = patch + (src - patch) * alpha[..., None]


def apply_healing(rgb: np.ndarray, params: dict[str, Any],
                  roi: tuple[float, float, float, float] = FULL_ROI,
                  inplace: bool = False) -> np.ndarray:
    """Apply a healing block to a float32 RGB image in [0,1] and return a new one.
    Neutral parameters return the input array unchanged (no copy).

    `roi` is (x0, y0, w, h) in normalized coordinates saying where `rgb` sits
    inside the whole photo, exactly as `editing.render` uses it — pass it when
    healing a crop (the 1:1 editor view) so every operation lands where it would
    in the full-frame render.

    `inplace` writes into `rgb` instead of a copy. Pass it when the caller owns
    the buffer, which `editing.render` does: the repair itself costs what the
    region costs, but duplicating a 24 MP float frame to protect an array nobody
    else holds costs more than the repair, and it is the one part of this module
    that would scale with the image.

    Two honest caveats about the ROI path, both of them about pixels the window
    does not contain. A heal whose region reaches the window's edge diffuses from
    less context than the full frame would give it, and a clone whose source lies
    outside the window has nothing to read there and leaves those pixels alone.
    Both go away if the caller widens the window to `read_bbox`; neither is
    silently papered over here.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected an RGB image, got {rgb.shape}"
    assert rgb.dtype == np.float32, f"expected float32 in [0,1], got {rgb.dtype}"
    assert len(roi) == 4 and roi[2] > 0 and roi[3] > 0, f"degenerate roi {roi}"

    p = normalize(params)
    if p is None:
        return rgb
    ops = [op for op in p["ops"] if op_is_active(op)]
    if not ops:
        return rgb

    h, w = rgb.shape[:2]
    frame_w, frame_h = w / roi[2], h / roi[3]
    ox, oy = roi[0] * frame_w, roi[1] * frame_h
    out = rgb if inplace else rgb.copy()
    for op in ops:
        _apply_op(out, op, frame_w, frame_h, ox, oy)
    return out
