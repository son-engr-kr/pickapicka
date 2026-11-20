"""Content removal that carries no licence with it: heal, clone and spot.

Three tools, all built on OpenCV's own algorithms and the arithmetic in this
file. Nothing here downloads a model or borrows weights, so an MIT-licensed
build stays MIT and the photographer using it inherits no restriction. That is
the whole point of this module: it is the *default* path of the removal feature
(see `docs/roadmap-parity.md`, "Decisions taken"), and a heavier model behind a
consent dialog is somebody else's file.

  heal   fill a region with what surrounds it
  clone  copy from an explicit source offset, through a feathered edge
  spot   a one-click circular heal, for dust and blemishes

A heal or a spot fills its region one of two ways. Method "patch", what the
editor makes, is the healing brush's: real pixels copied from a nearby source
(found once by `find_source` and stored as the op's offset, as a clone's is)
and blended in the gradient domain, so the copy keeps its own grain while its
tones are bent to meet the region's edge exactly (Perez, Gangnet and Blake,
"Poisson Image Editing", SIGGRAPH 2003: the "seamless cloning" of section 3,
guidance field the source's own gradient). Methods "ns" and "telea" are
OpenCV's diffusion (`cv2.inpaint`), what every heal used before; an op made
that way still renders as it did. Diffusion has no grain to give, and
Navier-Stokes carries the rim's noise inward along the pixel grid, so on a
grainy sky it left a smooth disc crossed by horizontal and vertical streaks.

A heal or a spot may instead carry method "ai": its pixels were made once by a
generative model (`aifill`, opt-in by use, MIT-licensed weights) and are handed
in by the caller as a patch, which this module only places and blends through
the same alpha a diffusion heal uses. Nothing here runs or fetches the model.

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
#   method    "ns" | "telea", the diffusion used by heal and spot, or "ai":
#             a patch made by `aifill`, which the op then names in
#   fill      {"id": <64 hex>, "box": [x0, y0, x1, y1], "model": "aifill"}: the
#             patch's name, where it sits in frame fractions, and what made it
#             (ai only)
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
PATCH_METHOD = "patch"     # a source patch, blended seamlessly; see `find_source`
AI_METHOD = "ai"           # a patch made by a model; see `aifill` and `genfill`
FILL_MODELS = ("aifill", "genfill")
_AI_KINDS = ("heal", "spot")

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
        "method": raw["method"] if raw.get("method") in METHODS or (
            raw.get("method") == PATCH_METHOD and kind in _AI_KINDS) else "ns",
        "dx": 0.0,
        "dy": 0.0,
    }
    if op["opacity"] == 0:
        return None
    if raw.get("method") == AI_METHOD and kind in _AI_KINDS:
        # Until its patch has been made it cannot change a pixel; a stroke the
        # caller has not filled yet is not an edit.
        fill = _normalize_fill(raw.get("fill"))
        if fill is None:
            return None
        op["method"] = AI_METHOD
        op["fill"] = fill
    if kind == "clone" or op["method"] == PATCH_METHOD:
        op["dx"] = _fnum(raw.get("dx"), _OFFSET_RANGE[0], _OFFSET_RANGE[1], 0.0)
        op["dy"] = _fnum(raw.get("dy"), _OFFSET_RANGE[0], _OFFSET_RANGE[1], 0.0)
        if abs(op["dx"]) < _EPS and abs(op["dy"]) < _EPS:
            return None    # copying a region onto itself is not an edit
    return op


def _normalize_fill(raw: Any) -> dict[str, Any] | None:
    """An AI heal's patch reference, or None when it is not a usable one."""
    if not isinstance(raw, dict):
        return None
    fid, box = raw.get("id"), raw.get("box")
    if not (isinstance(fid, str) and len(fid) == 64 and all(c in "0123456789abcdef" for c in fid)):
        return None
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    b = [_fnum(v, 0.0, 1.0, -1.0) for v in box]
    if min(b) < 0.0 or b[2] <= b[0] or b[3] <= b[1]:
        return None
    out: dict[str, Any] = {"id": fid, "box": b}
    if raw.get("model") in FILL_MODELS:
        out["model"] = raw["model"]      # which model made it, for the list; not read by the render
    return out


def fill_ids(params: Any) -> list[str]:
    """The patches the active AI heals in a healing block need."""
    p = normalize(params)
    if p is None:
        return []
    return [op["fill"]["id"] for op in p["ops"] if op_is_active(op) and op["method"] == AI_METHOD]


def without_fills(params: Any) -> dict[str, Any] | None:
    """The block with its AI heals taken out, for an edit going to another
    photo: a fill is that photo's own pixels, pasted anywhere else it is a
    piece of the wrong picture."""
    p = normalize(params)
    if p is None:
        return None
    return normalize({"ops": [op for op in p["ops"] if op["method"] != AI_METHOD]})


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


def region_mask(op: dict[str, Any], w: int, h: int) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """The box an operation writes in a w x h frame (as `region_bbox`) and,
    over it, which pixels it writes: everything its alpha reaches."""
    x0, y0, x1, y1 = region_bbox(op, w, h)
    if x1 <= x0 or y1 <= y0:
        return (x0, y0, x1, y1), np.zeros((max(0, y1 - y0), max(0, x1 - x0)), bool)
    r_px, collar = _op_geometry(op, float(w))
    core = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _rasterize(core, _pixel_path(op, float(w), float(h), float(x0), float(y0)), r_px)
    _, alpha = _fill_and_alpha(core, collar)
    return (x0, y0, x1, y1), alpha > 0.0


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
    if _reads_offset(op):
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
        if _reads_offset(op):
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


def _reads_offset(op: dict[str, Any]) -> bool:
    """A clone, or a heal filled from a source patch: both read a region-sized
    patch at the op's offset."""
    return op["kind"] == "clone" or op["method"] == PATCH_METHOD


# ----- seamless patch -----------------------------------------------------

def _laplacian(fill: np.ndarray) -> tuple[Any, list[tuple[np.ndarray, np.ndarray, np.ndarray]], tuple[np.ndarray, np.ndarray]]:
    """The 5-point Laplace system over the pixels of `fill`, factorized.

    Every pixel of `fill` is an unknown; a neighbour outside it (inside the
    array) is a fixed boundary value, and a neighbour past the array's edge is
    left out (a zero-flux edge). Returns the solver, and for each of the four
    directions the unknowns that have a boundary neighbour that way, with that
    neighbour's coordinates, so a caller builds the right-hand side for any
    boundary values without rebuilding the matrix."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    h, w = fill.shape
    ys, xs = np.nonzero(fill)
    n = ys.size
    idx = np.full((h, w), -1, dtype=np.int64)
    idx[ys, xs] = np.arange(n)
    rows, cols, data = [np.arange(n)], [np.arange(n)], [np.zeros(n)]
    diag = data[0]
    bound = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        diag += inside
        ny_c, nx_c = np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)
        j = np.where(inside, idx[ny_c, nx_c], -1)
        own = inside & (j >= 0)
        rows.append(np.arange(n)[own]); cols.append(j[own]); data.append(-np.ones(own.sum()))
        edge = inside & (j < 0)
        bound.append((np.nonzero(edge)[0], ny[edge], nx[edge]))
    a = sp.csc_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    return spla.factorized(a), bound, (ys, xs)


def _membrane(edge: np.ndarray, system: Any) -> np.ndarray:
    """The harmonic function over the fill that equals `edge` (h, w, 3) on its
    boundary, as an (h, w, 3) array that is zero outside the fill."""
    solve, bound, (ys, xs) = system
    out = np.zeros_like(edge)
    for c in range(edge.shape[2]):
        rhs = np.zeros(ys.size)
        for k, by, bx in bound:
            np.add.at(rhs, k, edge[by, bx, c])
        out[ys, xs, c] = solve(rhs)
    return out


# Unknowns past which the membrane is solved on a smaller grid. A direct solve
# grows faster than the region (measured: 0.06 s at 20k unknowns, 0.6 s at
# 130k, 3.9 s at 500k); a membrane is harmonic, smooth away from its edge, so a
# grid a few times coarser loses only the edge's pixel-scale noise, which the
# op's feathered collar blends over. A dust spot never gets here.
_MEMBRANE_MAX = 100_000


def _seamless(patch: np.ndarray, src: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """`src` blended into `patch` over `fill` (Perez et al. 2003, seamless
    cloning): inside the fill the result keeps the source's gradient and meets
    the patch on the fill's edge, which is the source plus the membrane that
    takes the source's edge to the patch's."""
    diff = (patch - src).astype(np.float32)
    n = int(fill.sum())
    if n <= _MEMBRANE_MAX:
        m = _membrane(diff.astype(np.float64), _laplacian(fill))
    else:
        h, w = fill.shape
        f = math.sqrt(n / _MEMBRANE_MAX)
        size = (max(2, round(w / f)), max(2, round(h / f)))
        outside = (fill == 0).astype(np.float32)
        # The edge values on the coarse grid: an average of the outside pixels
        # only, since the difference inside the fill means nothing.
        wsum = cv2.resize(outside, size, interpolation=cv2.INTER_AREA)
        dsum = cv2.resize(diff * outside[..., None], size, interpolation=cv2.INTER_AREA)
        coarse_fill = (wsum < 1e-6).astype(np.uint8)
        coarse_edge = dsum / np.maximum(wsum, 1e-6)[..., None]
        m = _membrane(coarse_edge.astype(np.float64), _laplacian(coarse_fill))
        m = np.where(coarse_fill[..., None] > 0, m, coarse_edge)
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    out = src + m.astype(np.float32)
    return np.where(fill[..., None] > 0, out, patch)


# Where `find_source` looks: rings at these distances, in units of the region's
# reach (radius plus collar), at this many angles each. The nearest ring that
# clears the region and its surround is 2 reaches out; sky and skin change
# slowly enough that 4.5 is still the same stuff.
_SOURCE_RINGS = (2.2, 2.9, 3.7, 4.5)
_SOURCE_ANGLES = 16
_SOURCE_WORK_R = 16.0      # px: the search runs on a copy scaled so the radius is about this


def find_source(rgb: np.ndarray, op: dict[str, Any],
                roi: tuple[float, float, float, float] = FULL_ROI) -> tuple[float, float] | None:
    """Where a heal or a spot should copy from, as the (dx, dy) frame-fraction
    offset of its source, or None when no candidate lies wholly in the image.

    The healing brush's choice, made once when the repair is placed: candidates
    on rings around the region, each scored by two things the blend cannot fix.
    The band just outside the region has to look like the band outside the
    candidate (the blend takes care of a difference in level, so it is compared
    with its mean taken out); and the candidate's inside has to be as plain as
    its own surround says it should be: what is left after the membrane of its
    own edge is taken out is a blob, an edge, another speck of dust. The search
    runs on a copy scaled so the radius is about 16 px, which a dust spot's
    surround does not need more than.

    `rgb` is float32 in [0,1], the frame (or the window `roi` of it) the repair
    applies to, with the repairs before this one already done."""
    assert rgb.ndim == 3 and rgb.dtype == np.float32, f"expected float32 RGB, got {rgb.dtype} {rgb.shape}"
    p = normalize_op(dict(op, method="ns", dx=0.0, dy=0.0))
    assert p is not None and p["kind"] in _AI_KINDS, f"not a heal or a spot: {op!r}"
    h, w = rgb.shape[:2]
    frame_w, frame_h = w / roi[2], h / roi[3]
    ox, oy = roi[0] * frame_w, roi[1] * frame_h
    r_px, collar = _op_geometry(p, frame_w)
    k = min(1.0, _SOURCE_WORK_R / r_px)
    img = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))),
                     interpolation=cv2.INTER_AREA) if k < 1.0 else rgb
    hh, ww = img.shape[:2]
    path = _pixel_path(p, frame_w, frame_h, ox, oy) * k
    r, col = r_px * k, collar * k
    band = max(2.0, 0.5 * r)
    x0, y0, x1, y1 = _reach_bbox(path, r + col + band + 1, ww, hh)
    core = np.zeros((y1 - y0, x1 - x0), np.uint8)
    _rasterize(core, path - np.asarray([x0, y0], dtype=np.float64), r)
    dist = cv2.distanceTransform(255 - core, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fill = (dist <= col).astype(np.uint8)
    ring = (dist > col) & (dist <= col + band)
    system = _laplacian(fill)
    target = img[y0:y1, x0:x1].astype(np.float64)
    t_ring = target[ring] - target[ring].mean(0)
    # Reach in pixels of the region's own box: the farthest pixel it covers.
    reach = max(x1 - x0, y1 - y0) / 2.0
    best, best_cost = None, math.inf
    for d in _SOURCE_RINGS:
        for i in range(_SOURCE_ANGLES):
            a = 2 * math.pi * i / _SOURCE_ANGLES
            dx, dy = int(round(math.cos(a) * d * reach)), int(round(math.sin(a) * d * reach))
            if x0 + dx < 0 or y0 + dy < 0 or x1 + dx > ww or y1 + dy > hh:
                continue
            src = img[y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float64)
            s_ring = src[ring] - src[ring].mean(0)
            ring_cost = float(((t_ring - s_ring) ** 2).sum(1).mean())
            own = src - _membrane(np.where(fill[..., None] > 0, 0.0, src), system)
            inside_cost = float((own[fill > 0] ** 2).sum(1).mean())
            cost = ring_cost + inside_cost
            if cost < best_cost:
                best, best_cost = (dx, dy), cost
    if best is None:
        return None
    return best[0] / k / frame_w, best[1] / k / frame_h


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


def _place_patch(patch: np.ndarray, box: list[float], frame_w: float, frame_h: float,
                 ox: float, oy: float, region: tuple[int, int, int, int]) -> np.ndarray:
    """An AI heal's patch (uint8, made at some full resolution over `box`, in
    frame fractions) resampled onto `region` (x0, y0, x1, y1) of a window at
    (ox, oy) in a frame_w x frame_h frame, as float32 in [0,1].

    At the patch's own scale, and a window cut at whole pixels of it, this is
    an exact copy. Smaller, it is first shrunk by area, so a preview of a large
    fill does not alias; what is left of the scale is a bilinear resample that
    depends only on the frame's scale, not on where a window starts."""
    x0, y0, x1, y1 = region
    ph, pw = patch.shape[:2]
    sx = (box[2] - box[0]) * frame_w / pw
    sy = (box[3] - box[1]) * frame_h / ph
    src = patch.astype(np.float32) / 255.0
    if sx < 0.5 or sy < 0.5:
        nw, nh = max(1, int(round(pw * sx))), max(1, int(round(ph * sy)))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        sx, sy = sx * pw / nw, sy * ph / nh
    # Pixel centres: patch pixel u sits at frame position box[0] * frame_w + (u + 0.5) * sx - 0.5.
    tx = box[0] * frame_w + 0.5 * sx - 0.5 - ox - x0
    ty = box[1] * frame_h + 0.5 * sy - 0.5 - oy - y0
    m = np.float32([[sx, 0.0, tx], [0.0, sy, ty]])
    return cv2.warpAffine(src, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _apply_op(img: np.ndarray, op: dict[str, Any], frame_w: float, frame_h: float,
              ox: float, oy: float, fills: dict[str, np.ndarray] | None = None) -> None:
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
    if op["method"] == AI_METHOD:
        fid = op["fill"]["id"]
        assert fills is not None and fid in fills, f"AI heal {fid[:12]} needs its patch from the caller"
        src = _place_patch(fills[fid], op["fill"]["box"], frame_w, frame_h, ox, oy, box)
    elif _reads_offset(op):
        src, valid = _shifted_read(img, box, int(round(op["dy"] * frame_h)),
                                   int(round(op["dx"] * frame_w)))
        alpha = alpha * valid
        if op["kind"] != "clone":
            src = _seamless(patch, src, fill)
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
                  inplace: bool = False,
                  fills: dict[str, np.ndarray] | None = None) -> np.ndarray:
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

    `fills` maps an AI heal's patch name to the patch (`aifill.make_fill`), and
    must hold every one the active operations name.
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
        _apply_op(out, op, frame_w, frame_h, ox, oy, fills)
    return out
