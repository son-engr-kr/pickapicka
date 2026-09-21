"""Lightroom-style Transform: perspective correction and automatic Upright.

The panel's two halves:

  manual   `vertical` / `horizontal` keystone, `rotate`, `aspect`, `scale` and
           `offset_x` / `offset_y`. Every one is neutral at its default, and the
           neutral set is *exactly* the identity, so a switched-on but untouched
           panel costs nothing and resamples nothing.
  upright  `upright` asks for the correction to be estimated from the image's
           own lines. It is a request, not a parameter: `resolve_upright` turns
           it into the manual parameters above, which is what the editor then
           puts on its sliders. Nothing about the answer is hidden from the user
           or from a test, and the user can overrule any of it.

Only numpy + OpenCV, like `editing`. Line segments come from `cv2.HoughLinesP`
and not from `cv2.createFastLineDetector`, because the latter lives in
opencv-contrib's `ximgproc` and this project depends on plain `opencv-python`
(`hasattr(cv2, "createFastLineDetector")` is False here). `createLineSegmentDetector`
does exist in this build but has been present, absent and legally encumbered
across OpenCV 4.x releases, so it is not something to build a feature on.
Hough over Canny is the boring choice that is always there.

Coordinates
-----------
Everything public is in NORMALIZED coordinates: a point is a fraction of the
frame, (0,0) the top-left corner and (1,1) the bottom-right. `norm_matrix`
returns a 3x3 homography between those, so one matrix serves a 400 px thumbnail,
the fit preview and a 24 MP export alike. `estimate_upright` honours the same
contract by always measuring on a copy resampled to a fixed `EST_LONG`, so the
size it was handed cannot leak into the answer.

Resolution independence is not aspect independence. A rotation is only a
rotation when the axes are equally scaled, and a vanishing point is a physical
place, so the frame's aspect (w/h) is an argument -- `frame_aspect`. Pass it and
the matrix is exact; leave it at 1.0 and you get a square frame's answer.
Internally every control is defined in one centered isotropic space,
x in [-A, A] and y in [-1, 1] with A = w/h and y pointing down, and
`norm_matrix` conjugates that back into normalized coordinates.

The manual controls compose in this order, innermost (applied to the source
point) first:

    rotate -> keystone -> aspect -> scale -> offset

Rotate before keystone because that is the order a person works in and the order
Upright estimates in: level the picture, then straighten what is still leaning.
Aspect after the keystone because its job is to undo the stretch the keystone
left. Scale then offset last, because those two choose which part of the warped
picture the frame ends up on.

How it composes with editing.geometry
-------------------------------------
This stage runs BEFORE `editing`'s straighten-and-crop, the order Lightroom
uses: warp the picture, then choose the rectangle out of it. It maps the unit
square onto the unit square and, applied to pixels, keeps the frame's size, so
`editing.geometry_matrix` downstream sees exactly the frame it expects and
needs no changes. In normalized coordinates the full geometric chain is

    n_out = G @ T @ n_src

with T from `norm_matrix` here and G from `editing.geometry_norm_matrix`
reshaped to 3x3 (its six numbers are row-major; append [0, 0, 1]). Multiplying
the two and warping once is worth doing for the export: two resamples of a
24 MP frame cost twice as much and blur twice as much as one.

`rotate` versus editing's `tilt`
-------------------------------
They do the same thing and `rotate` is the worse of the two on its own, so say
it plainly: **for straightening, keep using `tilt`.** It shares this module's
sign convention exactly (a horizon drooping right is levelled by a positive
value), and it cuts the frame back to the largest inscribed rectangle instead of
asking for a zoom, which is a strictly better answer -- no invented edge, no
second parameter to get wrong. A caller that wires up this panel should map
Upright's "level" mode onto `tilt` and leave `rotate` at zero.

`rotate` exists anyway for two reasons that `tilt` cannot cover. It has to be
*inside* the homography: Upright measures the roll first and then measures the
convergence of the levelled picture, so the keystone it solves is only correct
when composed onto that rotation, and the two cannot be split across two stages
without `tilt`'s crop landing between them and changing the frame. And the fit
is a property of the whole warp at once -- `fit_scale` needs the roll and the
keystone in one quad, because zooming to cover a rotated frame and then zooming
again to cover a bent one over-zooms. So: one control, two implementations, and
the duplicate is deliberate and confined to the case where a keystone is also
in play.

Blank corners
-------------
A keystone invents edges. `editing`'s `tilt` never does, because it cuts the
output back to the largest inscribed rectangle of the original aspect
(`_tilt_scale`). That exact answer does not transfer: a homography's inscribed
rectangle depends on all seven controls at once, so the output size would change
under every nudge of every slider, and the crop rectangle the user drew next
would mean something different each time. The frame size is therefore held
constant and the same inscribed-rectangle computation is reported as a
parameter instead -- `fit_scale` returns the smallest `scale` that covers the
frame. `estimate_upright` always fills it in, so an automatic correction never
leaves a blank corner, and a manual one is left exactly as asked with
`apply_transform` filling the uncovered area with black. Black, not
BORDER_REPLICATE: an invented edge should be visible, not smeared into
something that looks like a photograph.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

# ----- schema -------------------------------------------------------------

UPRIGHT_MODES: tuple[str, ...] = ("off", "level", "vertical", "full", "auto")

DEFAULT_TRANSFORM: dict[str, Any] = {
    "vertical": 0.0,    # keystone about the horizontal axis; + widens the top
    "horizontal": 0.0,  # keystone about the vertical axis; + widens the left
    "rotate": 0.0,      # degrees; same sign as editing's `tilt`
    "aspect": 0.0,      # + stretches the frame across, - stretches it tall
    "scale": 100.0,     # zoom; > 100 pushes invented edges out of frame
    "offset_x": 0.0,    # + moves the picture right within the frame
    "offset_y": 0.0,    # + moves the picture down
    "upright": "off",   # see UPRIGHT_MODES; resolved by `resolve_upright`
}

# All floats, unlike `editing`'s mostly-integer sliders: every one of these can
# be produced by `estimate_upright`, and rounding an estimated keystone to a
# whole slider step throws away a correction the estimator actually measured.
# The UI can still step in ones.
_RANGES: dict[str, tuple[float, float]] = {
    "vertical": (-100.0, 100.0),
    "horizontal": (-100.0, 100.0),
    "rotate": (-45.0, 45.0),
    "aspect": (-100.0, 100.0),
    # The slider reads 0..200, but a scale of 0 collapses the frame to a point
    # and its matrix is singular, so the low end is held at 10.
    "scale": (10.0, 200.0),
    "offset_x": (-100.0, 100.0),
    "offset_y": (-100.0, 100.0),
}

_EPS = 1e-4

# Keystone coefficient at slider +-100. The homography's denominator is
# 1 + a*x + b*y, and on the frame's own corners |a*x| and |b*y| are each at most
# this, leaving 1 - 2*0.35 = 0.3 in the worst case. That bound does NOT survive
# the rotation in front of it: a corner of a 3:1 frame turned by 45 degrees
# lands at |y| = 2.8, not 1, and the denominator there can reach zero. So the
# combination of a wide frame, a large rotation and both keystones near their
# limits is a warp that folds the frame through infinity. It is not clamped into
# something plausible -- `is_degenerate` names it and `norm_matrix` refuses it.
KEYSTONE_MAX = 0.35
ASPECT_MAX = 1.3      # stretch factor at aspect +-100
OFFSET_MAX = 1.0      # offset +-100 shifts by half the frame
SCALE_MIN, SCALE_MAX = _RANGES["scale"]


def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def _all_neutral(p: dict[str, Any]) -> bool:
    return (p["upright"] == "off"
            and all(abs(p[k] - DEFAULT_TRANSFORM[k]) < _EPS for k in _RANGES))


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp a transform dict, or None when it asks for nothing -- which is what
    keeps a switched-on but untouched panel from counting as an edit. Same
    contract as `editing.normalize_hsl`.

    An `upright` request other than "off" is never neutral even though it names
    no numbers, because it is a request to go and find some.
    """
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_TRANSFORM)
    for key, (lo, hi) in _RANGES.items():
        if raw.get(key) is None:
            continue
        out[key] = _fnum(raw[key], lo, hi, DEFAULT_TRANSFORM[key])
    mode = raw.get("upright")
    out["upright"] = mode if mode in UPRIGHT_MODES else "off"
    return None if _all_neutral(out) else out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when this transform leaves the frame alone."""
    return normalize(params) is None


def _params(raw: Any) -> dict[str, Any]:
    """The full clamped dict, neutral included."""
    n = normalize(raw)
    return dict(DEFAULT_TRANSFORM) if n is None else n


# ----- the matrix ---------------------------------------------------------

def _to_centered(a: float) -> np.ndarray:
    """Normalized (fraction of the frame) -> centered isotropic, y down."""
    return np.array([[2.0 * a, 0.0, -a], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]])


def _from_centered(a: float) -> np.ndarray:
    """Its exact inverse, written out rather than inverted numerically."""
    return np.array([[1.0 / (2.0 * a), 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])


def _frame_corners(a: float) -> np.ndarray:
    """The frame's own corners in centered coordinates, clockwise from top-left."""
    return np.array([[-a, -1.0], [a, -1.0], [a, 1.0], [-a, 1.0]])


def _rotation(deg: float) -> np.ndarray:
    """Turns a line of slope +tan(deg) flat, which is `editing.tilt`'s sign: a
    horizon drooping to the right is levelled by a positive value."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _keystone(vertical: float, horizontal: float, a: float) -> np.ndarray:
    """The two-point perspective. Its third row is what makes this a homography
    rather than an affine: the point (x, y) leaves at (x, y) / (ax + by + 1), so
    one edge of the frame is magnified and the opposite one shrunk.

    `horizontal`'s coefficient is divided by the aspect so that a slider value
    means the same fraction of the frame's *own* width whatever its shape;
    without that a 3:1 panorama would fold at half the slider's travel.
    """
    return np.array([[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [KEYSTONE_MAX * horizontal / 100.0 / a,
                      KEYSTONE_MAX * vertical / 100.0, 1.0]])


def _stretch(aspect: float) -> np.ndarray:
    s = ASPECT_MAX ** (aspect / 100.0)
    return np.diag([s, 1.0 / s, 1.0])


def _pre_scale(p: dict[str, Any], a: float) -> np.ndarray:
    """Everything the frame's contents go through before the zoom and the shift:
    stretch @ keystone @ rotate, in centered coordinates."""
    return _stretch(p["aspect"]) @ _keystone(p["vertical"], p["horizontal"], a) \
        @ _rotation(p["rotate"])


def _offset(p: dict[str, Any], a: float) -> np.ndarray:
    return np.array([OFFSET_MAX * a * p["offset_x"] / 100.0,
                     OFFSET_MAX * p["offset_y"] / 100.0])


def _hom(pts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Apply a 3x3 to an (N,2) array of points and divide through."""
    q = np.column_stack([pts, np.ones(len(pts))]) @ m.T
    assert (np.abs(q[:, 2]) > 1e-9).all(), "the warp folds the frame through infinity"
    return q[:, :2] / q[:, 2:3]


def _corner_weights(p: dict[str, Any], a: float) -> np.ndarray:
    """The homographic denominator at each of the frame's four corners. All four
    must stay positive: a corner whose weight has crossed zero has been mapped
    through infinity and is now behind the camera."""
    return np.column_stack([_frame_corners(a), np.ones(4)]) @ _pre_scale(p, a)[2]


def is_degenerate(params: dict[str, Any] | None,
                  frame_aspect: float = 1.0) -> bool:
    """True when these controls do not describe a picture: some corner of the
    frame has been pushed through infinity, which happens for a wide frame with
    a large rotation and both keystones near their limits (see KEYSTONE_MAX).

    `norm_matrix` and `fit_scale` both refuse such a set outright rather than
    quietly clamping it into something plausible. This predicate exists so the
    UI can stop short of it -- hold the slider, grey out the combination -- and
    not so anything can carry on regardless.
    """
    return bool((_corner_weights(_params(params), float(frame_aspect)) <= 1e-6).any())


def norm_matrix(params: dict[str, Any] | None,
                frame_aspect: float = 1.0) -> np.ndarray:
    """The 3x3 homography taking a point given as a fraction of the source frame
    to the same point as a fraction of the output frame.

    `frame_aspect` is the source frame's w/h. It is not a resolution -- every
    render size of one photo shares it -- but it is needed for a rotation to be
    a rotation. Normalized in and normalized out, so the same matrix applies at
    any render size, and neutral parameters give exactly `np.eye(3)`.

    An unresolved `upright` request is refused rather than ignored: this
    function cannot see the image, and silently dropping the automatic
    correction would be the worst of the available answers.
    """
    p = _params(params)
    assert p["upright"] == "off", \
        f"upright={p['upright']!r} needs resolve_upright() first; norm_matrix " \
        "cannot see the image"
    a = float(frame_aspect)
    assert a > 0.0, f"frame aspect must be positive, got {a}"
    if _all_neutral(p):
        return np.eye(3)
    w = _corner_weights(p, a)
    assert (w > 1e-6).all(), \
        f"these controls fold the frame through infinity (corner weights {w}); " \
        "see is_degenerate"
    m = _pre_scale(p, a)
    z = p["scale"] / 100.0
    t = _offset(p, a)
    m = np.array([[z, 0.0, t[0]], [0.0, z, t[1]], [0.0, 0.0, 1.0]]) @ m
    m = _from_centered(a) @ m @ _to_centered(a)
    return m / m[2, 2]


def fit_scale(params: dict[str, Any] | None, frame_aspect: float = 1.0) -> float:
    """The smallest `scale` at which the warped picture still covers the whole
    frame, in slider units. This is `editing._tilt_scale`'s inscribed-rectangle
    question asked of a homography, and reported as a parameter rather than
    taken silently, because a frame that resized itself on every slider nudge
    would move the crop the user drew afterwards.

    The warped frame is a convex quadrilateral. Zooming by s about the centre and
    then shifting by t puts its edge line at `n.x = s*g + n.t`, so covering a
    corner f of the frame needs `s >= n.(f - t) / g` -- one lower bound per edge
    per corner, and the largest of them is the answer. The frame's centre is a
    fixed point of every control, so it is always inside the quad and g is
    always negative; the binding corner of each edge is therefore the one
    furthest outside it, hence the `min`.

    Clamped to the slider's range, so an extreme warp can come back at
    SCALE_MAX without actually fitting -- there is no zoom the slider can
    express that would cover a 3:1 frame turned by 44 degrees.
    """
    p = _params(params)
    a = float(frame_aspect)
    assert a > 0.0, f"frame aspect must be positive, got {a}"
    w = _corner_weights(p, a)
    assert (w > 1e-6).all(), \
        f"these controls fold the frame through infinity (corner weights {w}); " \
        "see is_degenerate"
    quad = _hom(_frame_corners(a), _pre_scale(p, a))
    t = _offset(p, a)
    corners = _frame_corners(a)
    centre = quad.mean(axis=0)
    want = 1.0
    for i in range(4):
        r0, r1 = quad[i], quad[(i + 1) % 4]
        d = r1 - r0
        n = np.array([-d[1], d[0]])
        g = float(n @ r0)
        if n @ centre < g:            # orient the normal inwards
            n, g = -n, -g
        assert g < 0.0, f"the frame's centre fell outside its own image: {g}"
        want = max(want, float(((corners - t) @ n).min() / g))
    return min(SCALE_MAX, 100.0 * want)


def apply_transform(rgb: np.ndarray, params: dict[str, Any] | None) -> np.ndarray:
    """Warp a whole frame, resolving an `upright` request first. The output has
    the input's size; uncovered area is black.

    A convenience for tests and for a one-off render. The real pipeline should
    fold `norm_matrix` into the geometry stage's own matrix and warp once.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"want HxWx3, got {rgb.shape}"
    p = resolve_upright(rgb, params)
    if _all_neutral(p):
        return rgb
    h, w = rgb.shape[:2]
    n = norm_matrix(p, w / h)
    # Normalized -> pixels on both sides: diag(w,h,1) @ N @ diag(1/w,1/h,1).
    px = np.diag([float(w), float(h), 1.0]) @ n @ np.diag([1.0 / w, 1.0 / h, 1.0])
    return cv2.warpPerspective(rgb, px, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


# ----- Upright: estimating the correction from the picture's own lines -----
#
# The measurement runs on a copy resampled to EST_LONG on its long side, always,
# up or down. Two things fall out of that: the cost is flat in megapixels, and
# the detector always sees the same scene at the same scale, so the estimate
# cannot depend on the size it was handed. Segment endpoints are then converted
# to centered isotropic coordinates, which is a uniform scale and a shift -- no
# resolution survives into the answer.

EST_LONG = 1024        # long side the estimate is always measured at
MIN_LEN_FRAC = 0.12    # shortest segment worth trusting, as a fraction of it
AXIS_TOL = 25.0        # how far off an axis a segment may lean and still count
MIN_VP_SEGMENTS = 6    # a vanishing point from fewer lines has no redundancy
MIN_EVIDENCE = 1.5     # agreeing line, summed, as a multiple of half the frame
INLIER_DEG = 6.0       # width of the agreement window for the roll
MIN_INLIER = 0.45      # weight fraction that must fall inside it
VP_COND = 3.0          # how much the second eigenvalue must beat the first by
VP_FLOOR = 0.02        # ...and in absolute terms, per unit of segment weight
VP_MARGIN = 1.5        # a vanishing point nearer than this is not believable
AUTO_DAMP = 0.5        # "auto" is "full" held back by this much


def _estimation_gray(rgb: np.ndarray) -> np.ndarray:
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"want HxWx3, got {rgb.shape}"
    assert rgb.dtype == np.uint8, f"want uint8, got {rgb.dtype}"
    h, w = rgb.shape[:2]
    assert min(w, h) >= 32, f"too small to find lines in: {w}x{h}"
    k = EST_LONG / max(w, h)
    size = (max(1, int(round(w * k))), max(1, int(round(h * k))))
    small = cv2.resize(rgb, size,
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _segments(gray: np.ndarray, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Line segments as (N,4) endpoints in centered isotropic coordinates, with
    their lengths as weights. Canny's thresholds come off the image's median so
    the same call works on a bright wall and a night street."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    med = float(np.median(blur))
    edges = cv2.Canny(blur, int(max(0.0, 0.66 * med)), int(min(255.0, 1.33 * med)),
                      L2gradient=True)
    least = MIN_LEN_FRAC * EST_LONG
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, int(least * 0.5),
                            minLineLength=int(least),
                            maxLineGap=int(EST_LONG * 0.01))
    if lines is None:
        return np.zeros((0, 4)), np.zeros(0)
    seg = lines.reshape(-1, 4).astype(np.float64)
    h, w = gray.shape[:2]
    seg[:, 0::2] = (2.0 * seg[:, 0::2] / w - 1.0) * a
    seg[:, 1::2] = 2.0 * seg[:, 1::2] / h - 1.0
    wt = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    keep = wt > 1e-9
    return seg[keep], wt[keep]


def _axis_angles(seg: np.ndarray) -> np.ndarray:
    """Each segment's direction in degrees, folded into [-90, 90) -- a line has
    no arrowhead, so the two directions are the same line."""
    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0]))
    return (ang + 90.0) % 180.0 - 90.0


def _families(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Which segments lean near enough to each axis to be evidence about it.
    A segment steeper than AXIS_TOL off both axes is a diagonal and says nothing
    about either, so it takes part in neither family."""
    ang = _axis_angles(seg)
    return np.abs(ang) <= AXIS_TOL, np.abs(ang) >= 90.0 - AXIS_TOL


def _wmedian(v: np.ndarray, wt: np.ndarray) -> float:
    order = np.argsort(v)
    c = np.cumsum(wt[order])
    return float(v[order][np.searchsorted(c, 0.5 * c[-1])])


def _family_angle(ang: np.ndarray, wt: np.ndarray) -> tuple[float, float] | None:
    """One family's level reference: the weighted median angle refined over its
    inliers, plus the length of line that agreed. None when the family is not
    evidence, which is the honest answer for a picture of leaves.

    The median rather than the mean because converging lines spread out
    symmetrically about the true roll -- the middle of them is still level even
    when the spread is wide.

    Both gates are on length, not on how many pieces the detector happened to
    cut the lines into. A lone clean horizon comes back from Hough as one or two
    frame-wide segments and is overwhelming evidence about the roll; six 120 px
    scraps pointing every which way are none. Counting segments gets that
    backwards, and it is the one thing an angle estimate is allowed to be sure
    about from very few lines -- it is a single parameter.
    """
    if not len(ang):
        return None
    med = _wmedian(ang, wt)
    keep = np.abs(ang - med) <= INLIER_DEG
    agreed = float(wt[keep].sum())
    if agreed < MIN_EVIDENCE or agreed < MIN_INLIER * float(wt.sum()):
        return None
    return float(np.average(ang[keep], weights=wt[keep])), agreed


def _roll(seg: np.ndarray, wt: np.ndarray) -> float:
    """How far the picture is turned, in `rotate`'s own units and sign.

    Both families vote: a near-horizontal segment should be at 0 and a
    near-vertical one at 90, so each says the same thing about the roll. Each is
    tested for self-agreement on its own and only the ones that pass are
    averaged, weighted by how much line agreed -- an interior with one clean
    horizontal family and a mess of near-vertical clutter still gets levelled.
    """
    ang = _axis_angles(seg)
    flat, steep = _families(seg)
    # A near-vertical segment is levelled by the same turn as a near-horizontal
    # one once its 90 degrees are taken off, so both vote on one quantity.
    upright = ang[steep]
    votes = []
    for a, w in ((ang[flat], wt[flat]),
                 (np.where(upright > 0, upright - 90.0, upright + 90.0), wt[steep])):
        got = _family_angle(a, w)
        if got is not None:
            votes.append(got)
    if not votes:
        return 0.0
    total = sum(v[1] for v in votes)
    return sum(v[0] * v[1] for v in votes) / total


def _vanishing_point(seg: np.ndarray, wt: np.ndarray, axis: int,
                     margin: float) -> np.ndarray | None:
    """Where a family of lines meets, or None when it does not usefully meet.

    Each segment gives a homogeneous line l, scaled so that l.p is a real
    distance; the meeting point is the null vector of sum(w * l l'). Three ways
    that can fail to mean anything, and all three end in no correction:
      - the null space is not one-dimensional, so the point is not determined;
      - the point is at infinity, so the lines are already parallel and there is
        nothing to fix;
      - the point is inside the frame or just outside it, which is not a
        perspective but a fan of radial clutter.

    Unlike the roll, this one does keep a minimum segment *count*: two lines fix
    a meeting point exactly, with nothing left over to tell you whether it is
    the right one, so the conditioning test below would have nothing to measure.
    """
    if len(seg) < MIN_VP_SEGMENTS:
        return None
    p1 = np.column_stack([seg[:, 0], seg[:, 1], np.ones(len(seg))])
    p2 = np.column_stack([seg[:, 2], seg[:, 3], np.ones(len(seg))])
    lines = np.cross(p1, p2)
    norm = np.linalg.norm(lines[:, :2], axis=1)
    assert (norm > 1e-12).all(), "a zero-length segment made it through"
    lines = lines / norm[:, None]
    m = np.einsum("n,ni,nj->ij", wt, lines, lines)
    ev, vec = np.linalg.eigh(m)
    if ev[1] < max(VP_COND * ev[0], VP_FLOOR * float(wt.sum())):
        return None
    p = vec[:, 0]
    if abs(p[2]) < 1e-9 * float(np.linalg.norm(p[:2])):
        return None
    p = p / p[2]
    if abs(p[axis]) < margin:
        return None
    return p


def estimate_upright(rgb: np.ndarray, mode: str) -> dict[str, Any]:
    """The manual parameters the given Upright mode implies for this image.

    The result is an ordinary transform dict with `upright` back at "off", so it
    is inspectable, adjustable, storable and diffable -- the user sees the
    sliders move, not a black box. `scale` comes back filled in from
    `fit_scale`, so an automatic correction never leaves a blank corner; zero it
    out if you would rather see the invented edges.

    The modes follow Lightroom's, which nest:
      level     roll only.
      vertical  roll, then the vertical convergence.
      full      roll, both convergences, and the aspect they leave behind.
      auto      full, held back by AUTO_DAMP on everything but the roll -- the
                restrained version, for when a full correction looks staged.

    When the picture has no usable lines this returns the neutral dict. That is
    a real answer and not a failure: the correct correction for a photograph of
    fog is none.
    """
    assert mode in UPRIGHT_MODES, f"unknown upright mode {mode!r}"
    out = dict(DEFAULT_TRANSFORM)
    if mode == "off":
        return out
    h, w = rgb.shape[:2]
    a = w / h
    seg, wt = _segments(_estimation_gray(rgb), a)
    if not len(seg):
        return out
    out["rotate"] = _roll(seg, wt)

    if mode != "level":
        # Level first, then measure the convergence of what is left. The two
        # keystones are solved independently rather than as one 2x2 system: a
        # vertical vanishing point sits near the frame's own x centre, so its
        # contribution to the horizontal coefficient is second order, and
        # dropping it costs far less than the ill-conditioned solve would.
        turned = _rotate_points(seg, out["rotate"])
        flat, steep = _families(turned)
        # b = -1/py: the keystone that sends the vertical family's meeting point
        # off to infinity, which is what "these lines are parallel" means.
        vp_v = _vanishing_point(turned[steep], wt[steep], 1, VP_MARGIN)
        if vp_v is not None:
            out["vertical"] = 100.0 * (-1.0 / vp_v[1]) / KEYSTONE_MAX
        if mode in ("full", "auto"):
            # The horizontal vanishing point is off to the side, so its margin
            # is in half-widths where the vertical one's was in half-heights.
            vp_h = _vanishing_point(turned[flat], wt[flat], 0, VP_MARGIN * a)
            if vp_h is not None:
                out["horizontal"] = 100.0 * (-a / vp_h[0]) / KEYSTONE_MAX
            out["aspect"] = _aspect_after(out, a)
        if mode == "auto":
            for key in ("vertical", "horizontal", "aspect"):
                out[key] *= AUTO_DAMP

    for key, (lo, hi) in _RANGES.items():
        out[key] = min(hi, max(lo, out[key]))
    if _all_neutral(out):
        return out
    out["scale"] = fit_scale(out, a)
    return out


def _rotate_points(seg: np.ndarray, deg: float) -> np.ndarray:
    """Level a set of segment endpoints, so the convergence is measured on a
    picture that is already straight."""
    r = _rotation(deg)[:2, :2]
    out = seg.copy()
    out[:, 0:2] = seg[:, 0:2] @ r.T
    out[:, 2:4] = seg[:, 2:4] @ r.T
    return out


def _aspect_after(p: dict[str, Any], a: float) -> float:
    """The stretch that undoes what the keystone did to the frame's proportions.

    A keystone magnifies one edge and shrinks the opposite one; the two sides
    running between them grow with it, and by a different amount, which is why a
    corrected building looks too tall. Compare the mean length of the warped
    frame's two horizontal edges against its two vertical ones and stretch by
    the square root of the ratio, which makes the two magnifications equal.
    Small in practice -- a couple of slider points for a strong correction --
    but it is the difference between a plausible building and a stretched one.
    """
    quad = _hom(_frame_corners(a),
                _keystone(p["vertical"], p["horizontal"], a)
                @ _rotation(p["rotate"]))
    def side(i: int, j: int) -> float:
        return float(np.linalg.norm(quad[i] - quad[j]))
    across = (side(0, 1) + side(3, 2)) / (2.0 * 2.0 * a)
    down = (side(0, 3) + side(1, 2)) / (2.0 * 2.0)
    return 100.0 * math.log(math.sqrt(down / across)) / math.log(ASPECT_MAX)


def resolve_upright(rgb: np.ndarray, params: dict[str, Any] | None) -> dict[str, Any]:
    """Fold an `upright` request into the manual parameters, leaving the ones the
    user already set alone: the estimate is added to them, so nudging `vertical`
    after asking for Upright adjusts the automatic answer instead of replacing
    it. A dict already resolved (or neutral) comes back untouched."""
    p = _params(params)
    if p["upright"] == "off":
        return p
    est = estimate_upright(rgb, p["upright"])
    out = dict(p)
    out["upright"] = "off"
    for key in ("vertical", "horizontal", "rotate", "aspect", "offset_x", "offset_y"):
        out[key] = p[key] + est[key]
    # The user's own scale multiplies the fitted one, so "Upright plus a bit more
    # zoom" is expressible and "Upright" alone still just covers the frame.
    out["scale"] = est["scale"] * p["scale"] / 100.0
    for key, (lo, hi) in _RANGES.items():
        out[key] = min(hi, max(lo, out[key]))
    return out
