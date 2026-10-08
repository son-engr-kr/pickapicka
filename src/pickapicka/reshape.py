"""Face reshaping: eyes, face width, jaw, chin, nose and mouth, as warps.

What a reshape is
-----------------
Six sliders, each -100..100 and neutral at 0, stored in the edit's portrait
panel beside the skin sliders and applied to every face `portrait.analyze`
found. Nothing is baked: a warp is rebuilt from the face's 106 landmarks and
the slider values on every render, so it can be changed or taken off at any
time and is resolution independent, like every other stage.

The warps
---------
Every warp is built from Gustafsson's local warps ("Interactive Image
Warping", Helsinki University of Technology, 1993, section 4.4): a circular
area of influence round a centre, outside which nothing moves. All of them are
inverse maps, so each output pixel x reads the source at u(x).

A local translation moves the content at c by t. The thesis prints its
weight with |m - c|^2 in the denominator, and the source code in its Appendix
B uses |x - m|^2 (m = c + t). The code is the one the text describes: only
with |x - m| does the weight reach 1 at x = m, so that the grabbed point lands
exactly where it was dragged, and only with it does the weight fall smoothly to
0 at the rim, as (1 - r^2/R^2)^2 for a small t. The printed form stays near 1
right up to the rim for a small t and then drops, which is a hard edge. So:

    a(x) = ((R^2 - |x - c|^2) / ((R^2 - |x - c|^2) + |x - m|^2))^2
    u(x) = x - a(x) t                      inside R, x elsewhere

A local scaling enlarges round c (the thesis's f_s, applied in the unit
circle of an ellipse so that it can follow an eye's shape):

    u = c + (1 - (rho - 1)^2 a) (x - c),   rho = elliptical radius / R

which magnifies 1 / (1 - a) at the centre, is the identity with a matching
slope at the rim, and is one-to-one for |a| < 1.

The thesis composes its warps one after another, because each is placed by
hand on an image already warped by the ones before. Here every control point
is a landmark found on the unwarped face, which is the case it says calls for
combining the warps rather than composing them (section 4.5), so their
displacements are added. A jaw line is moved by a chain of translations, one
at each contour landmark, with radii wider than the spacing so that the line
bends smoothly rather than in dents; the overlapping weights would then add
up to several times the requested move, so each chain is divided by its own
overlap at its strongest point.

What each slider moves
----------------------
- eye_size: a scaling of each eye, on an ellipse along the eye's corners.
- face_width: the cheeks between the cheekbone and the jaw, horizontally in the
  face's own frame (so a tilted head is slimmed across its face, not across
  the photo).
- jaw_width: the jaw from below the mouth to beside the chin, towards the
  middle of the mouth, which narrows it into a V rather than pulling it in flat.
- chin_length: the chin along the face's vertical.
- nose_width: the wings of the nose, towards its middle.
- mouth_width: the corners of the mouth, along the line between them.

Every size is a fraction of a distance on the face itself (between the eyes,
an eye's width, the nose's, the mouth's), so a headshot and a face in a group
photo get the same shape change. The landmark groups were read off a plot of
the model's numbered output on a real face, and checked on six (the jaw line
descends to the chin on both sides, the corners sit outside the middles).

The strengths were tuned by eye on real faces, for the end of each slider to
be a strong but still believable change. Measured there at full strength, as
how far a landmark lands from where it was: the cheek line 0.071 of the
distance between the eyes for face_width, the jaw 0.085 for jaw_width, the
chin 0.095 for chin_length, each nose wing 0.045, each mouth corner 0.086, and
an eye 1.33 times its size at its centre and 1.057 corner to corner. The chains
land at 80-90% of the move they ask for because their weights are divided by
their overlap at the peak, where the neighbours ask for a little less. The
smallest Jacobian determinant of the map, over six faces, every slider alone
at both ends and all of them together, was 0.57 (in the ring round an enlarged
eye, which a scaling compresses by design): nothing folds.

Where it runs, and what it costs
--------------------------------
Last in the grade, after the local masks and before the crop. The skin
retouching, the heals, the masks and the automatic masks are all placed on the
unwarped frame (that is where the landmarks and the segmentation were found),
so the warp moves their finished result rather than them. What does not follow
is something drawn on the warped preview: a heal or a brush placed in a region
the warp moves lands up to the warp's displacement away from where it was
drawn. The displacement is a few percent of the face at full strength.

The resampling is Lanczos (8x8). Most pixels move by a fraction of a pixel and
the ones beside them are not resampled at all, so any softening shows as a patch
of blurrier skin. Measured on a real cheek at a half-pixel shift, bilinear kept
74% of the detail finer than 1 px, bicubic 92% and Lanczos 94%.

Only the boxes round the faces are warped, so the cost follows the faces and
not the megapixels, and a window render gets identical pixels given `padding`.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from . import faceparams

SHAPE_KEYS = ("eye_size", "face_width", "jaw_width", "chin_length", "nose_width", "mouth_width")
SHAPE_RANGE = (-100, 100)

FULL_ROI = (0.0, 0.0, 1.0, 1.0)

# InsightFace 2d106det, read off its output on real faces. The jaw line runs on
# each side from the temple at eye level down to the chin, which is 0 on both.
CONTOUR_L = (1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0)
CONTOUR_R = (17, 25, 26, 27, 28, 29, 30, 31, 32, 18, 19, 20, 21, 22, 23, 24, 0)
# The eye's outline without 34/38 and 88/92, which sit on the pupil and follow
# the gaze rather than the eye.
EYE_RING_L = (33, 35, 36, 37, 39, 40, 41, 42)
EYE_RING_R = (87, 89, 90, 91, 93, 94, 95, 96)
EYE_CORNERS_L = (35, 39)          # outer, inner
EYE_CORNERS_R = (93, 89)
NOSE_WING_L = (76, 77, 78)        # the side, the outermost point, the nostril's foot
NOSE_WING_R = (82, 83, 84)
MOUTH_CORNERS = (52, 61)

# Strengths at +-100 and the reach of each warp, as fractions of a distance on
# the face: `s` is the distance between the eye centres. See "What each slider
# moves" for what they come to on a real face.
_EYE_SCALE = 0.25          # Gustafsson's a: 1.33x at the centre of the eye
_EYE_RADII = (1.0, 0.75)   # of the eye's width, along and across it
_FACE_MOVE = 0.08          # of s, the cheek line at full strength
_JAW_MOVE = 0.10           # of s
_CHIN_MOVE = 0.12          # of s
_CONTOUR_RADIUS = 0.45     # of s
_NOSE_MOVE = 0.08          # of the nose's width, each wing
_NOSE_RADIUS = 0.5         # of the nose's width
_MOUTH_MOVE = 0.08         # of the mouth's width, each corner
_MOUTH_RADIUS = 0.45       # of the mouth's width

# Where along the jaw line (0 at the temple, 16 at the chin) each slider acts:
# a smooth bump between two positions, peaking at a third.
_FACE_SPAN = (2, 7, 12)
_JAW_SPAN = (8, 12, 16)
_CHIN_SPAN = (13, 16, 19)  # past 16 it is mirrored: the chin's neighbours on both sides

_LANCZOS_REACH = 4.0       # pixels either side the resampling kernel reads
_SUBPIXEL = 32             # cv2.INTER_TAB_SIZE: the steps remap resolves a pixel into


def is_active(params: dict[str, Any] | None) -> bool:
    """Whether any face, by the panel or by its own settings, is reshaped."""
    return faceparams.anything(params, SHAPE_KEYS)


# ----- planning: landmarks and sliders to primitive warps -------------------

def _bump(pos: float, span: tuple[float, float, float]) -> float:
    """A raised-cosine bump that is 0 at span[0] and span[2] and 1 at span[1]."""
    lo, peak, hi = span
    if pos <= lo or pos >= hi:
        return 0.0
    t = (pos - lo) / (peak - lo) if pos <= peak else (hi - pos) / (hi - peak)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.zeros(2)


def _chain_overlap(centres: np.ndarray, radii: np.ndarray, at: int) -> float:
    """How much the chain's weights add up to at its centre `at`, for a small
    move: the factor its displacements are divided by."""
    r2 = ((centres - centres[at]) ** 2).sum(axis=1)
    k = np.clip(1.0 - r2 / radii ** 2, 0.0, None) ** 2
    return max(1.0, float(k.sum()))


def plan(face: dict[str, Any], params: dict[str, Any],
         frame_w: float, frame_h: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """One face's warps, in frame pixels.

    Returns (moves, scalings): `moves` rows are [cx, cy, R, tx, ty], already
    divided by their chain's overlap; `scalings` rows are [cx, cy, ux, uy, Rx,
    Ry, a], (ux, uy) the ellipse's long axis."""
    lm = np.asarray(face["landmarks"], np.float64) * [frame_w, frame_h]
    e_l, e_r = lm[list(EYE_RING_L)].mean(axis=0), lm[list(EYE_RING_R)].mean(axis=0)
    s = float(np.hypot(*(e_r - e_l)))
    ex = _unit(e_r - e_l)                 # the face's horizontal, image left to right
    ey = np.array([-ex[1], ex[0]])        # its vertical, pointing at the chin
    v = {k: float(params.get(k) or 0.0) / 100.0 for k in SHAPE_KEYS}
    moves: list[np.ndarray] = []
    scalings: list[np.ndarray] = []

    def chain(rows: list[tuple[np.ndarray, float, np.ndarray, float]]) -> None:
        """Add a chain of (centre, radius, move, weight) normalized by its
        overlap at the most heavily weighted centre."""
        rows = [r for r in rows if r[3] > 0.0 and float(np.hypot(*r[2])) > 1e-6]
        if not rows:
            return
        c = np.array([r[0] for r in rows])
        rad = np.array([r[1] for r in rows])
        z = _chain_overlap(c, rad, int(np.argmax([r[3] for r in rows])))
        for cc, rr, t, _ in rows:
            moves.append(np.array([cc[0], cc[1], rr, t[0] / z, t[1] / z]))

    if v["eye_size"]:
        for ring, (outer, inner) in ((EYE_RING_L, EYE_CORNERS_L), (EYE_RING_R, EYE_CORNERS_R)):
            axis = lm[inner] - lm[outer]
            w = float(np.hypot(*axis))
            u = _unit(axis)
            c = lm[list(ring)].mean(axis=0)
            scalings.append(np.array([c[0], c[1], u[0], u[1], _EYE_RADII[0] * w,
                                      _EYE_RADII[1] * w, _EYE_SCALE * v["eye_size"]]))

    radius = _CONTOUR_RADIUS * s
    mouth = lm[list(MOUTH_CORNERS)].mean(axis=0)
    for side, out_dir in ((CONTOUR_L, -ex), (CONTOUR_R, ex)):
        rows = []
        for pos, idx in enumerate(side):
            # A slider at + widens: the cheek line moves out along the face's
            # horizontal, the jaw away from the mouth.
            p = lm[idx]
            t = np.zeros(2)
            wsum = 0.0
            if v["face_width"]:
                w = _bump(pos, _FACE_SPAN)
                t = t + v["face_width"] * _FACE_MOVE * s * w * out_dir
                wsum += w
            if v["jaw_width"]:
                w = _bump(pos, _JAW_SPAN)
                t = t + v["jaw_width"] * _JAW_MOVE * s * w * _unit(p - mouth)
                wsum += w
            if v["chin_length"]:
                # The chin and its neighbours down the face; mirrored round 16
                # so that both sides' neighbours of the chin take part.
                w = max(_bump(pos, _CHIN_SPAN), _bump(32 - pos, _CHIN_SPAN))
                if idx == side[-1]:
                    w = 0.5 * w           # the chin is in both chains
                t = t + v["chin_length"] * _CHIN_MOVE * s * w * ey
                wsum += w
            rows.append((p, radius, t, wsum))
        chain(rows)

    if v["nose_width"]:
        nose_w = float(np.hypot(*(lm[NOSE_WING_R[1]] - lm[NOSE_WING_L[1]])))
        for wing, out_dir in ((NOSE_WING_L, -ex), (NOSE_WING_R, ex)):
            chain([(lm[i], _NOSE_RADIUS * nose_w,
                    v["nose_width"] * _NOSE_MOVE * nose_w * wt * out_dir, wt)
                   for i, wt in zip(wing, (0.6, 1.0, 0.8))])

    if v["mouth_width"]:
        a, b = lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]]
        mouth_w = float(np.hypot(*(b - a)))
        u = _unit(b - a)
        for corner, out_dir in ((a, -u), (b, u)):
            chain([(corner, _MOUTH_RADIUS * mouth_w,
                    v["mouth_width"] * _MOUTH_MOVE * mouth_w * out_dir, 1.0)])
    return moves, scalings


# ----- the field -----------------------------------------------------------

def _support(moves: list[np.ndarray], scalings: list[np.ndarray]) -> list[tuple[float, float, float, float]]:
    """Each primitive's bounding box in frame pixels, (x0, y0, x1, y1)."""
    out = []
    for m in moves:
        cx, cy, r = m[0], m[1], m[2]
        out.append((cx - r, cy - r, cx + r, cy + r))
    for sc in scalings:
        cx, cy, r = sc[0], sc[1], max(sc[4], sc[5])
        out.append((cx - r, cy - r, cx + r, cy + r))
    return out


def field(moves: list[np.ndarray], scalings: list[np.ndarray],
          xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Source minus destination, (dx, dy), over the grid of frame positions
    `xs` (columns) by `ys` (rows)."""
    dx = np.zeros((len(ys), len(xs)), np.float64)
    dy = np.zeros_like(dx)
    for m in moves:
        cx, cy, r, tx, ty = (float(v) for v in m)
        c0, c1 = np.searchsorted(xs, cx - r), np.searchsorted(xs, cx + r)
        r0, r1 = np.searchsorted(ys, cy - r), np.searchsorted(ys, cy + r)
        if c1 <= c0 or r1 <= r0:
            continue
        gx = xs[c0:c1][None, :]
        gy = ys[r0:r1][:, None]
        edge = r * r - ((gx - cx) ** 2 + (gy - cy) ** 2)
        msq = (gx - cx - tx) ** 2 + (gy - cy - ty) ** 2
        a = np.where(edge > 0.0, edge / np.maximum(edge + msq, 1e-12), 0.0) ** 2
        dx[r0:r1, c0:c1] -= a * tx
        dy[r0:r1, c0:c1] -= a * ty
    for sc in scalings:
        cx, cy, ux, uy, rx, ry, amt = (float(v) for v in sc)
        reach = max(rx, ry)
        c0, c1 = np.searchsorted(xs, cx - reach), np.searchsorted(xs, cx + reach)
        r0, r1 = np.searchsorted(ys, cy - reach), np.searchsorted(ys, cy + reach)
        if c1 <= c0 or r1 <= r0:
            continue
        px = xs[c0:c1][None, :] - cx
        py = ys[r0:r1][:, None] - cy
        rho = np.sqrt(((px * ux + py * uy) / rx) ** 2 + ((py * ux - px * uy) / ry) ** 2)
        k = np.where(rho < 1.0, -((rho - 1.0) ** 2) * amt, 0.0)
        dx[r0:r1, c0:c1] += k * px
        dy[r0:r1, c0:c1] += k * py
    return dx, dy


def _frame(shape: tuple[int, ...], roi: tuple[float, float, float, float]):
    """Whole-frame size in this array's pixels, and where the array starts."""
    h, w = shape[:2]
    fw, fh = w / roi[2], h / roi[3]
    return fw, fh, roi[0] * fw, roi[1] * fh


def _merged_boxes(boxes: list[tuple[float, float, float, float]]) -> list[list[float]]:
    """Union of overlapping boxes, so overlapping faces are warped in one pass."""
    out: list[list[float]] = []
    for b in sorted(boxes):
        b = list(b)
        merged = True
        while merged:
            merged = False
            for o in out:
                if b[0] <= o[2] and o[0] <= b[2] and b[1] <= o[3] and o[1] <= b[3]:
                    b = [min(b[0], o[0]), min(b[1], o[1]), max(b[2], o[2]), max(b[3], o[3])]
                    out.remove(o)
                    merged = True
                    break
        out.append(b)
    return out


def apply_reshape(img: np.ndarray, params: dict[str, Any] | None,
                  faces: list[dict[str, Any]] | None,
                  roi: tuple[float, float, float, float] = FULL_ROI) -> np.ndarray:
    """Warp every face in float32 RGB [0,1]. `roi` is where `img` sits in the
    frame, the same contract as `editing.render`."""
    assert img.dtype == np.float32, "apply_reshape works in float32"
    if not is_active(params) or not faces:
        return img
    frame_w, frame_h, ox, oy = _frame(img.shape, roi)
    h, w = img.shape[:2]
    per_face = []
    for face in faces:
        if face["box"][2] * frame_w < 16:
            continue            # a few pixels on a thumbnail: nothing to see
        moves, scalings = plan(face, faceparams.for_face(params, face, frame_w, frame_h), frame_w, frame_h)
        boxes = _support(moves, scalings)
        if boxes:
            per_face.append((moves, scalings, (min(b[0] for b in boxes), min(b[1] for b in boxes),
                                               max(b[2] for b in boxes), max(b[3] for b in boxes))))
    if not per_face:
        return img
    out = img
    for box in _merged_boxes([p[2] for p in per_face]):
        # The box in this array's pixels, clipped to it.
        x0, y0 = max(0, int(math.floor(box[0] - ox))), max(0, int(math.floor(box[1] - oy)))
        x1, y1 = min(w, int(math.ceil(box[2] - ox)) + 1), min(h, int(math.ceil(box[3] - oy)) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue            # this face is outside the window
        xs = np.arange(x0, x1, dtype=np.float64) + ox
        ys = np.arange(y0, y1, dtype=np.float64) + oy
        moves = [m for p in per_face for m in p[0]]
        scalings = [sc for p in per_face for sc in p[1]]
        dx, dy = field(moves, scalings, xs, ys)
        moved = (dx != 0.0) | (dy != 0.0)
        if not moved.any():
            continue
        # remap resolves a position to 1/32 px (cv2.INTER_TAB_SIZE) by rounding
        # it. Rounded here first, a position is an exact float32 whatever the
        # array's offset, so a window and the whole frame round it alike; left
        # to remap, 300.37 and 140.37 can fall either side of the same step.
        map_x = (np.arange(x0, x1, dtype=np.float64)[None, :]
                 + np.round(dx * _SUBPIXEL) / _SUBPIXEL).astype(np.float32)
        map_y = (np.arange(y0, y1, dtype=np.float64)[:, None]
                 + np.round(dy * _SUBPIXEL) / _SUBPIXEL).astype(np.float32)
        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
        np.clip(warped, 0.0, 1.0, out=warped)
        if out is img:
            out = img.copy()
        region = out[y0:y1, x0:x1]
        region[moved] = warped[moved]
    return out


def padding(params: dict[str, Any] | None, faces: list[dict[str, Any]] | None,
            frame_long: float) -> float:
    """Pixels of neighbourhood a window render needs for the warp: how far any
    pixel reads from, plus the resampling kernel's reach.

    Measured from the field itself, on a grid over each face. Only the long
    edge is known here, so the landmarks are scaled by it on both axes, which
    makes the face larger than it is on the short one; the result is then
    taken half again as large, since a coarse grid can fall beside the peak."""
    if not is_active(params) or not faces:
        return 0.0
    worst = 0.0
    for face in faces:
        moves, scalings = plan(face, faceparams.for_face(params, face, frame_long, frame_long),
                               frame_long, frame_long)
        boxes = _support(moves, scalings)
        if not boxes:
            continue
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        dx, dy = field(moves, scalings, np.linspace(x0, x1, 96), np.linspace(y0, y1, 96))
        worst = max(worst, float(np.sqrt(dx * dx + dy * dy).max()))
    return 1.5 * worst + _LANCZOS_REACH + 2.0
