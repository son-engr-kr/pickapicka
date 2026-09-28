"""Unit checks for the healing tools. Runnable with pytest or directly:

    uv run python tests/test_healing.py
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from pickapicka import healing


def _weave(h: int = 240, w: int = 320, seed: int = 3) -> np.ndarray:
    """A fine woven texture with noise — the hard case for a diffusion inpaint,
    and the one a dust spot actually has to survive."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    t = 0.5 + 0.16 * np.sin(xx / 3.0) * np.sin(yy / 3.5) + 0.09 * np.sin((xx + yy) / 7.0)
    img = np.stack([t, t * 0.95, t * 0.88], -1)
    img += rng.normal(0, 0.015, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _sky(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """A smooth gradient with a little grain — the case diffusion is good at."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = 0.35 + 0.4 * yy / h + 0.1 * xx / w
    img = np.stack([g * 0.85, g * 0.92, g], -1)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _analytic(h: int, w: int) -> np.ndarray:
    """A pattern defined in *normalized* coordinates, so the same picture exists
    at any resolution. Needed to compare two render sizes at all."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = (xx + 0.5) / w, (yy + 0.5) / h
    t = 0.5 + 0.2 * np.sin(u * 11.0) * np.sin(v * 9.0)
    return np.clip(np.stack([t, t * 0.9, t * 0.8], -1), 0.0, 1.0).astype(np.float32)


def _spot(cx: float, cy: float, radius: float, **kw) -> dict:
    return {"kind": "spot", "points": [[cx, cy]], "radius": radius, **kw}


def _dot(img: np.ndarray, cx: float, cy: float, r_px: int,
         colour=(1.0, 0.15, 0.15)) -> np.ndarray:
    """Paint a known defect and hand back the damaged copy."""
    out = img.copy()
    h, w = img.shape[:2]
    cv2.circle(out, (int(round(cx * w)), int(round(cy * h))), r_px, colour, -1)
    return out


# ----- schema and normalization -------------------------------------------

def test_neutral_forms_normalize_to_none() -> None:
    for raw in (None, {}, [], healing.DEFAULT_HEALING, {"ops": []}, {"ops": None},
                "nonsense", {"ops": [None, 7, {}, {"kind": "wat"}]}):
        assert healing.normalize(raw) is None, f"expected None for {raw!r}"
        assert healing.is_neutral(raw)


def test_normalize_drops_operations_that_do_nothing() -> None:
    # No path, zero opacity, and a clone reading from where it writes.
    assert healing.normalize_op({"kind": "heal", "points": []}) is None
    assert healing.normalize_op(_spot(0.5, 0.5, 0.01, opacity=0)) is None
    assert healing.normalize_op({"kind": "clone", "points": [[0.5, 0.5]],
                                 "radius": 0.02, "dx": 0.0, "dy": 0.0}) is None
    # A non-zero offset is a real clone.
    assert healing.normalize_op({"kind": "clone", "points": [[0.5, 0.5]],
                                 "radius": 0.02, "dx": 0.1, "dy": 0.0}) is not None


def test_normalize_clamps_and_caps() -> None:
    op = healing.normalize_op({"kind": "heal", "points": [[0.5, 0.5], [0.6, 0.5]],
                               "radius": 99.0, "feather": -40, "opacity": 500,
                               "method": "made-up", "dx": 9.0})
    assert op["radius"] == healing._RADIUS_RANGE[1]
    assert op["feather"] == 0 and op["opacity"] == 100
    assert op["method"] == "ns", "an unknown method must fall back to the default"
    assert op["dx"] == 0.0, "only a clone carries a source offset"

    clone = healing.normalize_op({"kind": "clone", "points": [[0.5, 0.5]],
                                 "radius": 0.02, "dx": 9.0, "dy": -9.0})
    assert (clone["dx"], clone["dy"]) == healing._OFFSET_RANGE[::-1]

    many = healing.normalize({"ops": [_spot(0.5, 0.5, 0.01)] * (healing.OP_MAX + 50)})
    assert len(many["ops"]) == healing.OP_MAX

    long_path = healing.normalize_op({"kind": "heal", "radius": 0.01,
                                      "points": [[0.5, 0.5]] * (healing.OP_POINTS_MAX + 10)})
    assert len(long_path["points"]) == healing.OP_POINTS_MAX


def test_spot_keeps_exactly_one_centre() -> None:
    op = healing.normalize_op({"kind": "spot", "radius": 0.01,
                              "points": [[0.2, 0.3], [0.7, 0.8]]})
    assert op["points"] == [[0.2, 0.3]]


def test_normalize_is_idempotent() -> None:
    once = healing.normalize({"ops": [_spot(0.3, 0.4, 0.02, feather=30),
                                      {"kind": "clone", "points": [[0.5, 0.5]],
                                       "radius": 0.03, "dx": 0.1, "dy": 0.05}]})
    assert healing.normalize(once) == once


def test_is_neutral_tracks_enabled() -> None:
    off = {"ops": [_spot(0.5, 0.5, 0.02, enabled=False)]}
    assert healing.is_neutral(off)
    assert healing.normalize(off) is not None, "a switched-off operation is kept"
    assert not healing.is_neutral({"ops": [_spot(0.5, 0.5, 0.02)]})


def test_neutral_apply_returns_the_input_array() -> None:
    img = _weave()
    for params in (None, {}, {"ops": []}, {"ops": [_spot(0.5, 0.5, 0.02, opacity=0)]},
                   {"ops": [_spot(0.5, 0.5, 0.02, enabled=False)]}):
        assert healing.apply_healing(img, params) is img


# ----- does it actually repair anything -----------------------------------

def test_spot_removes_a_known_defect() -> None:
    """Construct the defect, so "better" can be measured against the truth
    rather than against how it looks."""
    clean = _weave()
    h, w = clean.shape[:2]
    r_px = 5
    damaged = _dot(clean, 0.5, 0.5, r_px)
    out = healing.apply_healing(
        damaged, {"ops": [_spot(0.5, 0.5, r_px / w, feather=40)]})

    hole = np.zeros((h, w), np.uint8)
    cv2.circle(hole, (w // 2, h // 2), r_px, 255, -1)
    sel = hole > 0
    before = np.sqrt(((damaged - clean)[sel] ** 2).mean())
    after = np.sqrt(((out - clean)[sel] ** 2).mean())
    assert before > 0.25, f"the defect was not loud enough to be a test: {before}"
    assert after < before / 6, f"spot barely helped: {before:.3f} -> {after:.3f}"
    # And it must not have wandered: the frame outside the region is untouched.
    far = cv2.dilate(hole, np.ones((9, 9), np.uint8)) == 0
    assert np.array_equal(out[far], damaged[far])


def test_spot_on_a_smooth_background_is_nearly_perfect() -> None:
    """The case this module is actually for: dust on sky."""
    clean = _sky()
    w = clean.shape[1]
    damaged = _dot(clean, 0.4, 0.35, 4, colour=(0.05, 0.05, 0.05))
    out = healing.apply_healing(damaged, {"ops": [_spot(0.4, 0.35, 4 / w)]})
    assert np.abs(out - clean).max() < 0.04, \
        f"a dust spot on a gradient should vanish, worst pixel {np.abs(out - clean).max():.3f}"


def test_heal_follows_a_dragged_path() -> None:
    """A power line against a sky: a stroke, not a circle."""
    clean = _sky(200, 400)
    h, w = clean.shape[:2]
    damaged = clean.copy()
    cv2.line(damaged, (40, 70), (360, 110), (0.02, 0.02, 0.02), 3)
    params = {"ops": [{"kind": "heal", "radius": 4.0 / w, "feather": 50,
                       "points": [[x / w, (70 + (x - 40) * 40 / 320) / h]
                                  for x in range(40, 361, 8)]}]}
    out = healing.apply_healing(damaged, params)
    assert np.abs(out - clean).max() < 0.06, \
        f"the line survived the heal: worst pixel {np.abs(out - clean).max():.3f}"


def test_heal_is_deterministic() -> None:
    """The caller keys a cache on the parameters, so equal parameters must mean
    equal pixels — not merely similar ones."""
    img = _weave()
    params = {"ops": [_spot(0.4, 0.4, 0.02, method="ns"),
                      {"kind": "clone", "points": [[0.6, 0.5], [0.65, 0.55]],
                       "radius": 0.03, "dx": -0.15, "dy": 0.1, "feather": 70},
                      {"kind": "heal", "points": [[0.2, 0.7], [0.3, 0.75]],
                       "radius": 0.02, "method": "telea"}]}
    first = healing.apply_healing(img, params)
    for _ in range(4):
        assert np.array_equal(healing.apply_healing(img, params), first)
    assert not np.array_equal(first, img)


def test_both_methods_run_and_differ() -> None:
    img = _weave()
    ns = healing.apply_healing(img, {"ops": [_spot(0.5, 0.5, 0.06, method="ns")]})
    telea = healing.apply_healing(img, {"ops": [_spot(0.5, 0.5, 0.06, method="telea")]})
    assert not np.array_equal(ns, telea), "the two inpainters cannot be the same code"
    assert healing.METHODS["ns"] == cv2.INPAINT_NS


def test_operations_apply_in_order() -> None:
    """The second operation sees the first one's result, which is what makes a
    clone over a heal behave the way a layer stack does."""
    img = _weave()
    a = _spot(0.4, 0.5, 0.05)
    b = _spot(0.45, 0.5, 0.05)
    assert not np.array_equal(healing.apply_healing(img, {"ops": [a, b]}),
                              healing.apply_healing(img, {"ops": [b, a]}))


# ----- clone --------------------------------------------------------------

def _two_tone(h: int = 400, w: int = 400) -> np.ndarray:
    """Flat 0.2 everywhere with a flat 0.8 block, so a seam is unambiguous."""
    img = np.full((h, w, 3), 0.2, dtype=np.float32)
    img[150:250, 250:350] = 0.8
    return img


def test_clone_copies_from_the_source_offset() -> None:
    img = _two_tone()
    h, w = img.shape[:2]
    # Destination centred at (160, 200) px, source 140 px to its right, inside
    # the bright block; both discs are well clear of the block's own edges.
    op = {"kind": "clone", "points": [[160 / w, 200 / h]], "radius": 20 / w,
          "dx": 140 / w, "dy": 0.0, "feather": 0}
    out = healing.apply_healing(img, {"ops": [op]})
    assert abs(float(out[200, 160, 0]) - 0.8) < 1e-5, \
        f"the clone did not bring the source colour: {out[200, 160, 0]}"
    assert abs(float(out[200, 40, 0]) - 0.2) < 1e-6, "the clone leaked outside its region"
    # The source itself must be left alone.
    assert abs(float(out[200, 300, 0]) - 0.8) < 1e-6


def test_clone_reading_off_the_frame_writes_nothing_there() -> None:
    """No mirroring, no edge stretching: where there is nothing to copy, the
    original stays. Half the destination is covered, half is not."""
    img = _two_tone()
    h, w = img.shape[:2]
    op = {"kind": "clone", "points": [[10 / w, 200 / h]], "radius": 20 / w,
          "dx": -30 / w, "dy": 0.0, "feather": 0}
    out = healing.apply_healing(img, {"ops": [op]})
    assert np.isfinite(out).all()
    assert np.abs(out[:, :2] - img[:, :2]).max() < 1e-6


def test_clone_source_is_read_before_the_write() -> None:
    """An overlapping clone must not smear its own output across itself, or the
    result would depend on the order numpy happened to touch the rows."""
    img = _weave(120, 160)
    op = {"kind": "clone", "points": [[0.5, 0.5]], "radius": 0.15,
          "dx": 0.02, "dy": 0.0, "feather": 0}
    once = healing.apply_healing(img, {"ops": [op]})
    twice = healing.apply_healing(img, {"ops": [op]})
    assert np.array_equal(once, twice)
    h, w = img.shape[:2]
    shift = int(round(0.02 * w))
    assert abs(float(once[h // 2, w // 2, 0]) - float(img[h // 2, w // 2 + shift, 0])) < 1e-6


# ----- feather and opacity ------------------------------------------------

def test_feather_leaves_no_hard_seam() -> None:
    """A clone across a 0.6 step is the worst case for a seam. The alpha is a
    smoothstep over the collar, whose slope peaks at 1.5/collar, so the output
    gradient cannot exceed 1.5 * step / collar. Check the bound, and check that
    a hard edge really is the thing being avoided."""
    img = _two_tone()
    h, w = img.shape[:2]
    base = {"kind": "clone", "points": [[160 / w, 200 / h]], "radius": 20 / w,
            "dx": 140 / w, "dy": 0.0}

    def peak_gradient(feather: int) -> float:
        """The largest step between neighbouring pixels. A forward difference,
        not `np.gradient`: a central difference spreads a one-pixel step over
        two and would report a hard seam as half its real height."""
        out = healing.apply_healing(img, {"ops": [dict(base, feather=feather)]})[..., 0]
        win = out[150:250, 110:210].astype(np.float64)
        return float(max(np.abs(np.diff(win, axis=0)).max(),
                         np.abs(np.diff(win, axis=1)).max()))

    r_px, collar = healing._op_geometry(healing.normalize_op(dict(base, feather=100)),
                                        float(w))
    assert collar > 5.0, "the test needs a collar wide enough to measure"
    soft, hard = peak_gradient(100), peak_gradient(0)
    assert soft <= 1.5 * 0.6 / collar * 1.05, f"the feathered edge is too steep: {soft}"
    # feather=0 is not a raw copy either: the collar has a one-pixel floor, so
    # the steepest a "hard" edge can be is half the step. That floor is the
    # difference between a clone and a rectangle of pasted pixels.
    assert abs(hard - 0.5 * 0.6) < 0.01, f"the hard edge is not the 1 px floor: {hard}"
    assert soft < hard / 3.0, f"feather made little difference: hard {hard}, soft {soft}"


def test_feather_still_replaces_the_whole_drawn_region() -> None:
    """The point of dilating the mask by the collar: a feathered spot must not
    leave a rim of the dust it was drawn over."""
    clean = _sky()
    w = clean.shape[1]
    r_px = 6
    damaged = _dot(clean, 0.5, 0.5, r_px, colour=(0.0, 0.0, 0.0))
    out = healing.apply_healing(
        damaged, {"ops": [_spot(0.5, 0.5, r_px / w, feather=100)]})
    assert np.abs(out - clean).max() < 0.05, \
        f"a rim of the defect survived: worst pixel {np.abs(out - clean).max():.3f}"


def test_opacity_blends() -> None:
    img = _two_tone()
    h, w = img.shape[:2]
    base = {"kind": "clone", "points": [[160 / w, 200 / h]], "radius": 20 / w,
            "dx": 140 / w, "dy": 0.0, "feather": 0}
    for opacity in (25, 50, 75, 100):
        out = healing.apply_healing(img, {"ops": [dict(base, opacity=opacity)]})
        want = 0.2 + (opacity / 100.0) * 0.6
        assert abs(float(out[200, 160, 0]) - want) < 1e-5, \
            f"opacity {opacity} gave {out[200, 160, 0]}, wanted {want}"


# ----- normalized coordinates and the ROI path ----------------------------

def test_the_same_heal_lands_in_the_same_place_at_two_resolutions() -> None:
    """Normalized parameters exist so a heal survives being re-rendered. What
    can be checked is *placement*, not equality: at 2x the region covers four
    times as many pixels, the region mask rounds to a different set of them and
    the diffusion solves a different-sized problem, so the pixels legitimately
    differ. The centre of the change is what must not move — to within the
    pixel the region was rasterized to.

    Placement is measured from the *extent* of the changed pixels, not from a
    weighted centroid of the change: how wrong the diffusion is varies across
    the region with the texture under it, so a weighted centroid tracks the
    background rather than the heal."""
    params = {"ops": [_spot(0.375, 0.6, 0.05, feather=50)]}

    def placement(h: int, w: int) -> tuple[float, float]:
        img = _analytic(h, w)
        out = healing.apply_healing(img, params)
        ys, xs = np.nonzero(np.abs(out - img).sum(axis=2) > 1e-4)
        assert len(xs), "nothing changed"
        return (float(xs.min() + xs.max() + 1) / 2 / w,
                float(ys.min() + ys.max() + 1) / 2 / h)

    small, big = placement(300, 400), placement(600, 800)
    assert abs(small[0] - big[0]) < 2.0 / 400, f"x drifted: {small[0]} vs {big[0]}"
    assert abs(small[1] - big[1]) < 2.0 / 300, f"y drifted: {small[1]} vs {big[1]}"
    assert abs(small[0] - 0.375) < 0.01 and abs(small[1] - 0.6) < 0.01


def test_region_bbox_follows_the_frame_size() -> None:
    op = healing.normalize_op(_spot(0.5, 0.5, 0.05, feather=0))
    x0, y0, x1, y1 = healing.region_bbox(op, 400, 300)
    assert x0 < 200 < x1 and y0 < 150 < y1
    r_px = 0.05 * 400
    assert x1 - x0 <= 2 * (r_px + 2) + 2, f"the box is bigger than the region: {x1 - x0}"
    # Twice the frame, twice the box — bar the parts that do not scale: the
    # one-pixel minimum collar and the rounding out to whole pixels.
    bx0, by0, bx1, by1 = healing.region_bbox(op, 800, 600)
    assert abs((bx1 - bx0) - 2 * (x1 - x0)) <= 6

    off = healing.normalize_op(_spot(-0.5, -0.5, 0.02))
    assert healing.region_bbox(off, 400, 300)[2] == 0, "an off-frame region has no box"


def test_read_bbox_covers_the_margin_and_the_clone_source() -> None:
    op = healing.normalize_op({"kind": "clone", "points": [[0.5, 0.5]],
                               "radius": 0.05, "dx": 0.2, "dy": -0.1})
    w, h = 400, 300
    rx0, ry0, rx1, ry1 = healing.region_bbox(op, w, h)
    kx0, ky0, kx1, ky1 = healing.read_bbox(op, w, h)
    assert kx0 <= rx0 - healing._INPAINT_MARGIN and ky1 >= ry1 + healing._INPAINT_MARGIN
    assert kx1 >= rx1 + int(0.2 * w), "the source is to the right and must be inside"
    assert ky0 <= ry0 - int(0.1 * h)


def test_padding_is_zero_when_nothing_is_active() -> None:
    for params in (None, {}, {"ops": []}, {"ops": [_spot(0.5, 0.5, 0.02, enabled=False)]},
                   {"ops": [_spot(0.5, 0.5, 0.02, opacity=0)]}):
        assert healing.padding(params, 600, 400) == 0.0


def test_padding_reports_the_margin_and_the_clone_offset() -> None:
    """A heal needs the diffusion's read margin and nothing more, whatever its
    size; a clone needs to reach its source as well."""
    w, h = 600, 400
    small = healing.padding({"ops": [_spot(0.5, 0.5, 0.005)]}, w, h)
    big = healing.padding({"ops": [{"kind": "heal", "radius": 0.1, "feather": 100,
                                    "points": [[0.3, 0.5], [0.7, 0.5]]}]}, w, h)
    assert small == big == float(healing._INPAINT_MARGIN), \
        f"a heal's padding must not follow its radius: {small} vs {big}"

    clone = healing.padding({"ops": [{"kind": "clone", "points": [[0.5, 0.5]],
                                      "radius": 0.02, "dx": -0.1, "dy": 0.05}]}, w, h)
    assert clone == float(healing._INPAINT_MARGIN) + 0.1 * w
    # The widest operation sets the padding for the whole block.
    both = healing.padding({"ops": [_spot(0.2, 0.2, 0.01),
                                    {"kind": "clone", "points": [[0.5, 0.5]],
                                     "radius": 0.02, "dx": -0.1, "dy": 0.05}]}, w, h)
    assert both == clone


def test_padding_agrees_with_read_bbox() -> None:
    """`padding` is computed straight from the reach rather than by subtracting
    the two boxes, so it has to be checked against them — away from the frame
    edge, where the clip on both boxes does not bite."""
    w, h = 600, 400
    for raw in (_spot(0.5, 0.5, 0.03, feather=80),
                {"kind": "heal", "radius": 0.02, "points": [[0.45, 0.5], [0.55, 0.55]]},
                {"kind": "clone", "points": [[0.5, 0.5]], "radius": 0.02,
                 "dx": -0.1, "dy": 0.05}):
        op = healing.normalize_op(raw)
        rx0, ry0, rx1, ry1 = healing.region_bbox(op, w, h)
        kx0, ky0, kx1, ky1 = healing.read_bbox(op, w, h)
        assert 0 < kx0 and 0 < ky0 and kx1 < w and ky1 < h, "the test needs slack"
        gap = max(rx0 - kx0, ry0 - ky0, kx1 - rx1, ky1 - ry1)
        assert abs(gap - healing.padding({"ops": [op]}, w, h)) <= 1.0, \
            f"padding {healing.padding({'ops': [op]}, w, h)} disagrees with read_bbox {gap}"


def test_a_window_padded_by_padding_matches_the_whole_frame() -> None:
    """The invariant `padding` exists for, and the number is incidental: hand a
    window that much surrounding image and it reproduces the full-frame render.

    The operations are placed against the window's top-left corner on purpose —
    a spot whose region abuts it, so the diffusion would otherwise be starved on
    two sides, and a clone whose source sits outside the window entirely, which
    without the padding copies nothing at all.
    """
    img = _weave(400, 600)
    h, w = img.shape[:2]
    wx0, wy0, ww, wh = 200, 120, 240, 180
    params = {"ops": [
        _spot(231 / w, 151 / h, 25 / w, feather=50),
        {"kind": "clone", "points": [[(wx0 + 30) / w, (wy0 + 150) / h]],
         "radius": 18 / w, "dx": -60 / w, "dy": 0.0, "feather": 50}]}
    full = healing.apply_healing(img, params)
    want = full[wy0:wy0 + wh, wx0:wx0 + ww]

    def windowed(pad: int) -> float:
        x0, y0 = wx0 - pad, wy0 - pad
        pw, ph = ww + 2 * pad, wh + 2 * pad
        assert x0 >= 0 and y0 >= 0 and x0 + pw <= w and y0 + ph <= h
        out = healing.apply_healing(img[y0:y0 + ph, x0:x0 + pw], params,
                                    roi=(x0 / w, y0 / h, pw / w, ph / h))
        inner = out[wy0 - y0:wy0 - y0 + wh, wx0 - x0:wx0 - x0 + ww]
        return float(np.abs(inner - want).max())

    pad = healing.padding(params, w, h)
    assert windowed(int(round(pad))) <= 2.0 / 255, \
        f"padding {pad} was not enough: off by {windowed(int(round(pad))) * 255:.1f}/255"
    # And the padding is doing real work, not covering for a heal that never
    # reached the boundary in the first place.
    assert windowed(0) > 8.0 / 255, \
        f"the unpadded window already agreed, so this pins nothing: {windowed(0) * 255:.1f}/255"


def test_roi_puts_a_heal_in_the_right_place_in_a_window() -> None:
    """Healing a window with its roi must reproduce that part of the full-frame
    heal. A tolerance is needed and it is not a fudge: roi is a ratio of floats,
    so `w / roi[2]` recovers the frame width to within an ulp rather than
    exactly, and a region centre landing within a hair of a pixel boundary can
    round to the neighbouring pixel. One 8-bit level is the size of that error;
    a misplaced heal would be off by tens."""
    img = _weave(400, 600)
    h, w = img.shape[:2]
    params = {"ops": [_spot(0.45, 0.5, 0.03, feather=60),
                      {"kind": "clone", "points": [[0.5, 0.45], [0.53, 0.47]],
                       "radius": 0.02, "dx": -0.08, "dy": 0.02, "feather": 50}]}
    full = healing.apply_healing(img, params)

    px0, py0, pw, ph = 180, 100, 300, 200
    roi = (px0 / w, py0 / h, pw / w, ph / h)
    win = healing.apply_healing(img[py0:py0 + ph, px0:px0 + pw], params, roi=roi)
    diff = np.abs(win - full[py0:py0 + ph, px0:px0 + pw])
    assert diff.max() < 1.5 / 255, f"the window heal drifted: max {diff.max():.5f}"
    # And it is not trivially equal because nothing happened.
    assert np.abs(win - img[py0:py0 + ph, px0:px0 + pw]).max() > 0.05


def test_an_operation_outside_the_window_does_nothing_to_it() -> None:
    img = _weave(200, 200)
    roi = (0.0, 0.0, 0.25, 0.25)
    out = healing.apply_healing(img, {"ops": [_spot(0.8, 0.8, 0.02)]}, roi=roi)
    assert np.array_equal(out, img)


# ----- cost ---------------------------------------------------------------

def test_the_windowed_inpaint_matches_a_whole_frame_one() -> None:
    """Why `_INPAINT_MARGIN` is what it is: both inpainters read at most
    `_INPAINT_RADIUS` outside the hole, so a collar of real pixels that wide is
    all the context the full frame would have given. If this ever fails, the
    margin is too small and every heal in the app is subtly resolution-dependent.
    """
    img = np.rint(np.clip(_weave(600, 600), 0, 1) * 255).astype(np.uint8)
    for r in (3, 12, 40):
        mask = np.zeros((600, 600), np.uint8)
        cv2.circle(mask, (300, 300), r, 255, -1)
        src = img.copy()
        src[mask > 0] = (230, 40, 40)
        for flag in healing.METHODS.values():
            full = cv2.inpaint(src, mask, healing._INPAINT_RADIUS, flag)
            m = healing._INPAINT_MARGIN
            sl = (slice(300 - r - m, 300 + r + m + 1),) * 2
            win = cv2.inpaint(src[sl], mask[sl], healing._INPAINT_RADIUS, flag)
            hole = mask[sl] > 0
            gap = np.abs(win.astype(int) - full[sl].astype(int))[hole].max()
            assert gap == 0, f"margin {m} is too thin at r={r}: off by {gap}/255"


def test_cost_scales_with_the_region_not_the_image() -> None:
    """The same 200 px heal on a 1 MP frame and on a 36 MP one. The repair is
    bounded by its own bounding box, so the two must cost the same; only the
    defensive copy of the frame scales, which is what `inplace` is for."""
    params = {"ops": [{"kind": "heal", "radius": 100.0 / 1000, "feather": 50,
                       "points": [[0.5, 0.5]]}]}
    timings = {}
    for edge in (1000, 6000):
        img = _analytic(edge, edge)
        # The radius is a fraction of the width, so hold the *pixel* size fixed.
        p = {"ops": [dict(params["ops"][0], radius=100.0 / edge)]}
        healing.apply_healing(img, p, inplace=True)          # warm the caches
        best = min(_time(lambda: healing.apply_healing(img, p, inplace=True))
                   for _ in range(3))
        timings[edge] = best
    ratio = timings[6000] / timings[1000]
    assert ratio < 3.0, f"cost followed the image, not the region: {timings}, {ratio:.1f}x"


def _time(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if fails else 0)
