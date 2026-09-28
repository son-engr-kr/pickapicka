"""Checks for the Transform stage: keystone, rotate, aspect, scale, offset and
automatic Upright.

    uv run python tests/test_transform.py

Two properties matter more than the rest. The matrix is in normalized
coordinates, so it must say the same thing at every render size -- including
when Upright *estimated* it, which is why the estimate is measured at two
resolutions and the two are compared. And Upright must degrade honestly: an
image with no usable lines has to come back neutral rather than warped by a
guess, so a frame of pure noise is checked directly.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from pickapicka import editing, transform


# ----- fixtures -----------------------------------------------------------

def _grid(w: int = 1600, h: int = 1200, roll: float = 0.0, n: int = 9) -> np.ndarray:
    """A wall of horizontal and vertical lines, optionally turned by `roll`
    degrees. Structure a line detector can measure, and nothing else."""
    img = np.full((h, w, 3), 30, np.uint8)
    c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))

    def turn(x: float, y: float) -> tuple[int, int]:
        x, y = x - w / 2, y - h / 2
        return int(round(x * c - y * s + w / 2)), int(round(x * s + y * c + h / 2))

    for i in range(1, n):
        y, x = h * i / n, w * i / n
        cv2.line(img, turn(-w, y), turn(2 * w, y), (220, 220, 220), 3)
        cv2.line(img, turn(x, -h), turn(x, 2 * h), (220, 220, 220), 3)
    return img


def _block(w: int = 400, h: int = 300) -> np.ndarray:
    """A bright square in the middle of a dark frame: measurable position and
    measurable extent, which is all the manual controls need to be pinned down.
    """
    img = np.zeros((h, w, 3), np.uint8)
    img[h // 2 - h // 10:h // 2 + h // 10, w // 2 - h // 10:w // 2 + h // 10] = 255
    return img


def _bbox(img: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, width, height) of the bright part, in normalized coordinates."""
    ys, xs = np.nonzero(img[..., 0] > 128)
    assert len(xs), "nothing bright left to measure"
    h, w = img.shape[:2]
    return ((xs.min() + xs.max() + 1) / 2 / w, (ys.min() + ys.max() + 1) / 2 / h,
            (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h)


def _at(m: np.ndarray, u: float, v: float) -> tuple[float, float]:
    q = m @ np.array([u, v, 1.0])
    return q[0] / q[2], q[1] / q[2]


def _family_spread(img: np.ndarray, vertical: bool = True) -> float:
    """How much the near-vertical (or near-horizontal) lines disagree about
    their direction, in degrees. Zero when they are parallel."""
    seg, _ = transform._segments(
        transform._estimation_gray(img), img.shape[1] / img.shape[0])
    ang = transform._axis_angles(seg)
    if vertical:
        fam = ang[np.abs(ang) >= 90.0 - transform.AXIS_TOL]
        fam = np.where(fam < 0, fam + 90.0, fam - 90.0)
    else:
        fam = ang[np.abs(ang) <= transform.AXIS_TOL]
    assert len(fam) >= 4, f"only {len(fam)} lines to measure"
    return float(np.percentile(fam, 90) - np.percentile(fam, 10))


# ----- schema -------------------------------------------------------------

def test_defaults_are_no_transform() -> None:
    assert transform.normalize(transform.DEFAULT_TRANSFORM) is None
    assert transform.normalize({}) is None
    assert transform.normalize(None) is None
    assert transform.normalize("nonsense") is None
    assert transform.is_neutral(None) and transform.is_neutral({})
    assert transform.DEFAULT_TRANSFORM["scale"] == 100.0
    assert transform.DEFAULT_TRANSFORM["upright"] == "off"


def test_a_transform_of_nothing_is_not_a_transform() -> None:
    """Or the panel would look permanently dirty the moment it was opened."""
    assert transform.normalize({"vertical": 0, "rotate": 0.0, "scale": 100}) is None
    assert transform.is_neutral({"scale": 100.0, "offset_x": 0})
    assert not transform.is_neutral({"vertical": 1})


def test_values_are_clamped_and_junk_falls_back() -> None:
    p = transform.normalize({"vertical": 999, "horizontal": -999, "rotate": 90,
                             "aspect": 500, "scale": 0, "offset_x": "nope"})
    assert p["vertical"] == 100.0 and p["horizontal"] == -100.0
    assert p["rotate"] == 45.0 and p["aspect"] == 100.0
    assert p["offset_x"] == 0.0, "unparseable values fall back to neutral"
    # The slider reads 0..200 but a scale of 0 is a singular matrix.
    assert p["scale"] == transform.SCALE_MIN > 0.0


def test_rotate_keeps_its_decimals() -> None:
    """A tenth of a degree is the difference between level and almost level, the
    same reason `editing` keeps `tilt` out of its integer rounding."""
    assert transform.normalize({"rotate": 3.7})["rotate"] == 3.7


def test_an_upright_request_is_never_neutral() -> None:
    """It names no numbers, but it is a request to go and find some."""
    for mode in ("level", "vertical", "full", "auto"):
        assert not transform.is_neutral({"upright": mode}), mode
        assert transform.normalize({"upright": mode})["upright"] == mode
    assert transform.is_neutral({"upright": "off"})
    assert transform.normalize({"upright": "banana"}) is None, "unknown mode is off"


# ----- the matrix ---------------------------------------------------------

def test_neutral_is_exactly_the_identity() -> None:
    for aspect in (1.0, 4 / 3, 3.0, 0.5):
        m = transform.norm_matrix({}, aspect)
        assert np.array_equal(m, np.eye(3)), f"aspect {aspect}: {m}"


def test_neutral_does_not_touch_a_pixel() -> None:
    """Not "the same pixels" -- the same array. A neutral panel must not cost a
    24 MP resample, and must not cost a rounding error either."""
    img = _block()
    assert transform.apply_transform(img, {}) is img
    assert transform.apply_transform(img, {"scale": 100.0}) is img
    assert transform.apply_transform(img, {"upright": "off"}) is img


def test_the_matrix_is_normalized_in_and_normalized_out() -> None:
    """The whole contract: it takes fractions of the frame to fractions of the
    frame, so the render size never enters into it."""
    p = {"vertical": 30.0, "horizontal": -20.0, "rotate": 4.0, "scale": 130.0}
    m = transform.norm_matrix(p, 4 / 3)
    for w, h in ((400, 300), (1600, 1200), (6000, 4500)):
        px = np.diag([w, h, 1.0]) @ m @ np.diag([1 / w, 1 / h, 1.0])
        px = px / px[2, 2]
        for u, v in ((0.0, 0.0), (0.5, 0.5), (0.3, 0.8), (1.0, 1.0)):
            want = _at(m, u, v)
            got = _at(px, u * w, v * h)
            assert abs(got[0] / w - want[0]) < 1e-9, f"{w}x{h} at ({u},{v})"
            assert abs(got[1] / h - want[1]) < 1e-9, f"{w}x{h} at ({u},{v})"


def test_a_symmetric_warp_leaves_the_centre_alone() -> None:
    """Every control is defined about the frame's middle, so the middle is the
    one point none of them may move."""
    for p in ({"vertical": 60.0}, {"horizontal": -45.0}, {"rotate": 15.0},
              {"aspect": 70.0}, {"scale": 180.0},
              {"vertical": 30.0, "rotate": -8.0, "aspect": 20.0, "scale": 140.0}):
        cx, cy = _at(transform.norm_matrix(p, 4 / 3), 0.5, 0.5)
        assert abs(cx - 0.5) < 1e-9 and abs(cy - 0.5) < 1e-9, f"{p} -> {cx},{cy}"


def test_an_unresolved_upright_is_refused_not_ignored() -> None:
    """Silently dropping the automatic correction is the worst answer available,
    so the matrix refuses to be built from a request it cannot evaluate."""
    try:
        transform.norm_matrix({"upright": "full"}, 4 / 3)
    except AssertionError:
        return
    raise AssertionError("expected a refusal for an unresolved upright request")


# ----- what each manual control does --------------------------------------

def test_vertical_keystone_widens_the_top() -> None:
    m = transform.norm_matrix({"vertical": 50.0}, 4 / 3)
    top = _at(m, 1.0, 0.0)[0] - _at(m, 0.0, 0.0)[0]
    bottom = _at(m, 1.0, 1.0)[0] - _at(m, 0.0, 1.0)[0]
    assert top > 1.0 > bottom, f"top {top:.3f}, bottom {bottom:.3f}"
    # ...and the negative direction does the opposite, not nothing.
    n = transform.norm_matrix({"vertical": -50.0}, 4 / 3)
    assert _at(n, 1.0, 0.0)[0] - _at(n, 0.0, 0.0)[0] < 1.0


def test_horizontal_keystone_widens_the_left() -> None:
    m = transform.norm_matrix({"horizontal": 50.0}, 4 / 3)
    left = _at(m, 0.0, 1.0)[1] - _at(m, 0.0, 0.0)[1]
    right = _at(m, 1.0, 1.0)[1] - _at(m, 1.0, 0.0)[1]
    assert left > 1.0 > right, f"left {left:.3f}, right {right:.3f}"
    n = transform.norm_matrix({"horizontal": -50.0}, 4 / 3)
    assert _at(n, 0.0, 1.0)[1] - _at(n, 0.0, 0.0)[1] < 1.0


def test_a_keystone_makes_converging_lines_parallel() -> None:
    """The point of the control. Bend a grid, then straighten it back and check
    the lines it invented agreement on a direction again."""
    base = _grid()
    for bend, mode in ((-40.0, True), (30.0, True)):
        p = {"vertical": bend}
        p["scale"] = transform.fit_scale(p, 4 / 3)
        bent = transform.apply_transform(base, p)
        before = _family_spread(bent, vertical=mode)
        assert before > 8.0, f"the test image is not bent enough: {before:.2f}deg"
        back = transform.estimate_upright(bent, "vertical")
        after = _family_spread(transform.apply_transform(bent, back), vertical=mode)
        assert after < 1.0, f"bend {bend}: {before:.2f}deg -> {after:.2f}deg"


def test_scale_zooms_in_about_the_centre() -> None:
    small = _bbox(transform.apply_transform(_block(), {}))
    big = _bbox(transform.apply_transform(_block(), {"scale": 150.0}))
    assert abs(big[0] - 0.5) < 0.01 and abs(big[1] - 0.5) < 0.01, "drifted off centre"
    assert big[2] > small[2] * 1.4 and big[3] > small[3] * 1.4, f"{small} -> {big}"
    tiny = _bbox(transform.apply_transform(_block(), {"scale": 50.0}))
    assert tiny[2] < small[2] * 0.6


def test_the_offsets_move_the_picture_the_way_they_read() -> None:
    base = _bbox(transform.apply_transform(_block(), {}))
    right = _bbox(transform.apply_transform(_block(), {"offset_x": 40.0}))
    down = _bbox(transform.apply_transform(_block(), {"offset_y": 40.0}))
    assert right[0] > base[0] + 0.1, f"offset_x did not move it right: {right[0]}"
    assert abs(right[1] - base[1]) < 0.01, "offset_x moved it vertically too"
    assert down[1] > base[1] + 0.1, f"offset_y did not move it down: {down[1]}"
    assert abs(down[0] - base[0]) < 0.01, "offset_y moved it horizontally too"
    left = _bbox(transform.apply_transform(_block(), {"offset_x": -40.0}))
    assert left[0] < base[0] - 0.1


def test_aspect_stretches_across_and_squeezes_tall() -> None:
    base = _bbox(transform.apply_transform(_block(), {}))
    wide = _bbox(transform.apply_transform(_block(), {"aspect": 100.0}))
    tall = _bbox(transform.apply_transform(_block(), {"aspect": -100.0}))
    assert wide[2] > base[2] * 1.1 and wide[3] < base[3] * 0.95, f"{base} -> {wide}"
    assert tall[3] > base[3] * 1.1 and tall[2] < base[2] * 0.95, f"{base} -> {tall}"


def test_rotate_has_the_same_sign_as_editings_tilt() -> None:
    """The caller has both. If they disagreed about which way is positive, one of
    the two panels would fight the person using it."""
    for deg in (3.0, -7.0, 12.0):
        mine = _grid(roll=deg)
        # Zoomed to the fit: the blank wedges a bare rotation leaves have straight
        # borders of their own, and the detector would measure those too.
        levelled = transform.apply_transform(
            mine, {"rotate": deg, "scale": transform.fit_scale({"rotate": deg}, 4 / 3)})
        assert _family_spread(levelled, vertical=False) < 1.0
        seg, _ = transform._segments(transform._estimation_gray(levelled), 4 / 3)
        ang = transform._axis_angles(seg)
        flat = ang[np.abs(ang) <= transform.AXIS_TOL]
        assert abs(float(np.median(flat))) < 0.5, \
            f"rotate {deg} left {np.median(flat):.2f}deg -- wrong sign?"


# ----- blank corners ------------------------------------------------------

def test_fit_scale_is_the_same_answer_editing_gives_a_tilt() -> None:
    """A rotation is the one case both modules can compute, so it is the one case
    where they must not disagree: `fit_scale` is `_tilt_scale` inverted."""
    checked = 0
    for deg in (1.0, 5.0, 12.0, 30.0, 44.0):
        for w, h in ((4000, 3000), (3000, 4000), (6000, 2000)):
            mine = transform.fit_scale({"rotate": deg}, w / h)
            theirs = 100.0 / editing._tilt_scale(w, h, deg)
            if theirs > transform.SCALE_MAX:
                # `tilt` can shrink the frame as far as it likes; a scale slider
                # stops at 200. Turning a 3:1 frame by 30 degrees needs 237.
                assert mine == transform.SCALE_MAX, f"{deg}deg {w}x{h}: {mine}"
                continue
            assert abs(mine - theirs) < 1e-6, f"{deg}deg {w}x{h}: {mine} vs {theirs}"
            checked += 1
    assert checked >= 10, f"only {checked} cases were actually compared"
    assert transform.fit_scale({}, 4 / 3) == 100.0, "neutral needs no zoom"


def test_scale_pushes_the_invented_edges_out_of_frame() -> None:
    """A keystone invents edges; `apply_transform` fills them with black so they
    are visible, and `fit_scale` says how much zoom removes them."""
    white = np.full((1200, 1600, 3), 255, np.uint8)
    for p in ({"vertical": 45.0}, {"horizontal": -30.0},
              {"rotate": 9.0, "vertical": 25.0}):
        loose = transform.apply_transform(white, p)
        assert loose.min() < 40, f"{p}: expected visible blank area"
        assert (loose < 40).mean() > 0.01, f"{p}: barely any blank area to fix"
        fitted = dict(p, scale=transform.fit_scale(p, 4 / 3))
        tight = transform.apply_transform(white, fitted)
        # The fit covers the frame exactly, so the outermost row of pixels can
        # still catch a sample from beyond the edge; judge the interior.
        assert tight[2:-2, 2:-2].min() > 200, \
            f"{p}: scale {fitted['scale']:.1f} still left a blank corner " \
            f"(min {tight[2:-2, 2:-2].min()})"


def test_an_offset_costs_the_zoom_it_should() -> None:
    """Shifting by half a frame needs exactly 2x to cover it again, which is the
    whole of the scale slider -- so the two ranges are matched deliberately."""
    assert abs(transform.fit_scale({"offset_x": 100.0}, 4 / 3) - 200.0) < 1e-9
    assert abs(transform.fit_scale({"offset_y": -100.0}, 4 / 3) - 200.0) < 1e-9
    assert abs(transform.fit_scale({"offset_x": 50.0}, 4 / 3) - 150.0) < 1e-9


def test_a_warp_that_folds_the_frame_is_refused() -> None:
    """A wide frame, a big rotation and both keystones at once can push a corner
    through infinity. That is not a picture, so it is named and refused rather
    than clamped into something that looks like one."""
    folded = {"vertical": -100.0, "horizontal": -100.0, "rotate": -45.0}
    assert transform.is_degenerate(folded, 3.0)
    assert not transform.is_degenerate(folded, 1.0), "only the wide frame folds"
    assert not transform.is_degenerate({}, 3.0)
    assert not transform.is_degenerate({"vertical": 100.0, "horizontal": 100.0}, 3.0), \
        "the keystones alone are inside the safe bound"
    for fn in (transform.norm_matrix, transform.fit_scale):
        try:
            fn(folded, 3.0)
        except AssertionError:
            continue
        raise AssertionError(f"{fn.__name__} accepted a folded frame")


# ----- Upright ------------------------------------------------------------

def test_upright_level_straightens_a_rolled_scene() -> None:
    for roll in (3.0, -7.0, 12.0, 0.0):
        est = transform.estimate_upright(_grid(roll=roll), "level")
        assert abs(est["rotate"] - roll) < 0.3, \
            f"rolled {roll}, estimated {est['rotate']:.3f}"
        assert est["vertical"] == est["horizontal"] == 0.0, "level is roll only"
        fixed = transform.apply_transform(_grid(roll=roll), {"upright": "level"})
        assert _family_spread(fixed, vertical=False) < 1.0


def test_upright_on_noise_returns_no_correction() -> None:
    """The honest answer for an image with no lines in it. A wild guess here is
    worse than doing nothing, because the user cannot tell it apart from a real
    correction until the export."""
    noise = np.random.default_rng(7).integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    for mode in ("level", "vertical", "full", "auto"):
        est = transform.estimate_upright(noise, mode)
        assert transform.is_neutral(est), f"{mode} invented {est}"
    assert transform.apply_transform(noise, {"upright": "full"}) is noise


def test_upright_on_a_flat_frame_returns_no_correction() -> None:
    """No edges at all, rather than edges that mean nothing."""
    flat = np.full((900, 1200, 3), 128, np.uint8)
    assert transform.is_neutral(transform.estimate_upright(flat, "full"))


def test_a_lone_horizon_is_enough_to_level_by() -> None:
    """Hough hands back one clean horizon as one or two frame-wide segments. That
    is overwhelming evidence about the roll and must not be thrown away for being
    only two pieces -- the gate is on how much line agrees, not how many bits the
    detector cut it into."""
    for roll in (6.0, -11.0):
        img = np.full((1200, 1600, 3), 30, np.uint8)
        yy, xx = np.mgrid[0:1200, 0:1600]
        img[yy > 600 + (xx - 800) * math.tan(math.radians(roll))] = 200
        est = transform.estimate_upright(img, "level")
        assert abs(est["rotate"] - roll) < 0.3, f"rolled {roll}, got {est['rotate']}"


def test_a_radial_fan_is_not_a_perspective() -> None:
    """Lines meeting inside the frame are clutter -- spokes, a sunburst, a pile of
    sticks -- not converging verticals, and a keystone built on that meeting
    point would be violent nonsense."""
    img = np.full((1200, 1600, 3), 30, np.uint8)
    for k in range(24):
        a = math.pi * k / 24
        cv2.line(img, (800, 600),
                 (int(800 + 2000 * math.cos(a)), int(600 + 2000 * math.sin(a))),
                 (220, 220, 220), 3)
    assert transform.is_neutral(transform.estimate_upright(img, "full"))


def test_curves_are_not_lines() -> None:
    """A wavy edge has a different direction everywhere along it, so the family
    cannot agree with itself and no roll comes out of it."""
    img = np.full((1200, 1600, 3), 30, np.uint8)
    for i in range(1, 9):
        pts = np.array([[x, 1200 * i / 9 + 90 * math.sin(math.pi * x / 1600)]
                        for x in range(0, 1601, 20)], np.int32)
        cv2.polylines(img, [pts], False, (220, 220, 220), 3)
    assert transform.estimate_upright(img, "level")["rotate"] == 0.0


def test_a_scene_built_of_diagonals_is_levelled_onto_them() -> None:
    """A known and accepted limitation, recorded so that changing it is a
    decision rather than a surprise. Upright has no idea what a building is; it
    finds the strongest agreeing family within AXIS_TOL of an axis and calls it
    level. A picture made entirely of consistent 20-degree diagonals therefore
    gets turned by 20 degrees, and it is right to by its own lights. The
    defences against this are the tolerance itself -- a 30-degree lattice is
    ignored -- and the fact that the user can see the number and undo it."""
    def lattice(deg: float) -> np.ndarray:
        img = np.full((1200, 1600, 3), 30, np.uint8)
        for i in range(-20, 40):
            for d in (deg, -deg):
                x0 = i * 80
                cv2.line(img, (x0, 0),
                         (int(x0 + 1200 * math.tan(math.radians(d))), 1200),
                         (220, 220, 220), 3)
        return img

    inside = transform.estimate_upright(lattice(20.0), "level")["rotate"]
    assert abs(abs(inside) - 20.0) < 1.0, f"expected the known bias, got {inside}"
    outside = transform.estimate_upright(lattice(38.0), "level")["rotate"]
    assert outside == 0.0, \
        f"a lattice past AXIS_TOL should be invisible, got {outside}"


def test_the_estimate_is_the_same_at_any_resolution() -> None:
    """The property the normalized matrix exists for. The estimate is measured at
    a fixed EST_LONG whatever it is handed, so the size cannot leak in: what
    little movement is left comes from resampling the test image itself."""
    p = {"vertical": -35.0, "rotate": 6.0}
    p["scale"] = transform.fit_scale(p, 4 / 3)
    big = transform.apply_transform(_grid(3200, 2400, n=13), p)
    ref = None
    for w, h in ((3200, 2400), (1600, 1200), (800, 600), (2400, 1800)):
        img = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
        m = transform.norm_matrix(transform.estimate_upright(img, "full"), 4 / 3)
        if ref is None:
            ref = m
            continue
        assert np.abs(m - ref).max() < 1e-2, \
            f"{w}x{h} disagrees by {np.abs(m - ref).max():.4f}:\n{m}\nvs\n{ref}"
        # Matrix entries are not all the same size, so also compare the two by
        # what they actually do: where do they send the frame? Anything under a
        # percent of the frame is invisible.
        probe = [(u, v) for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)]
        drift = max(math.dist(_at(m, u, v), _at(ref, u, v)) for u, v in probe)
        assert drift < 0.01, f"{w}x{h} moves the frame by {drift:.4f} of it"


def test_upright_never_leaves_a_blank_corner() -> None:
    """An automatic correction the user did not ask for in detail must not hand
    back a picture with holes in it, so the fit is part of the estimate."""
    p = {"vertical": -40.0, "rotate": 5.0}
    p["scale"] = transform.fit_scale(p, 4 / 3)
    bent = transform.apply_transform(np.full((1200, 1600, 3), 255, np.uint8),
                                     dict(p, scale=200.0))
    est = transform.estimate_upright(_grid(), "full")
    assert est["scale"] == 100.0, "a straight grid needs no zoom"
    est = transform.estimate_upright(transform.apply_transform(_grid(), p), "full")
    assert est["scale"] > 100.0, "a correction that invents edges needs zoom"
    out = transform.apply_transform(bent, est)
    assert out[2:-2, 2:-2].min() > 200, f"blank corner left (min {out.min()})"


def test_the_upright_modes_nest() -> None:
    """level < vertical < full, and auto is full held back -- so a user stepping
    through them sees more correction, not a different one."""
    p = {"vertical": -35.0, "rotate": 5.0}
    p["scale"] = transform.fit_scale(p, 4 / 3)
    bent = transform.apply_transform(_grid(), p)
    est = {m: transform.estimate_upright(bent, m) for m in transform.UPRIGHT_MODES}
    assert transform.is_neutral(est["off"])
    assert est["level"]["vertical"] == 0.0
    assert est["vertical"]["vertical"] > 5.0
    assert est["vertical"]["aspect"] == 0.0, "aspect belongs to full"
    assert est["full"]["aspect"] != 0.0
    assert abs(est["full"]["vertical"] - est["vertical"]["vertical"]) < 0.5, \
        "full and vertical should agree about the vertical convergence"
    for key in ("vertical", "aspect"):
        assert abs(est["auto"][key] - est["full"][key] * transform.AUTO_DAMP) < 1e-6
    assert abs(est["auto"]["rotate"] - est["full"]["rotate"]) < 1e-9, \
        "auto holds back the perspective, not the level"


def test_upright_adds_to_what_the_user_already_set() -> None:
    """So nudging a slider after asking for Upright adjusts the automatic answer
    rather than throwing it away."""
    grid = _grid(roll=8.0)
    auto = transform.resolve_upright(grid, {"upright": "level"})
    nudged = transform.resolve_upright(grid, {"upright": "level", "rotate": 2.0})
    assert abs(nudged["rotate"] - (auto["rotate"] + 2.0)) < 1e-9
    assert nudged["upright"] == "off", "the request has been spent"
    # An already-resolved dict is left exactly as it is.
    manual = {"vertical": 12.0, "scale": 110.0}
    assert transform.resolve_upright(grid, manual)["vertical"] == 12.0


def test_estimate_upright_refuses_a_mode_it_does_not_know() -> None:
    try:
        transform.estimate_upright(_block(), "sideways")
    except AssertionError:
        return
    raise AssertionError("expected a refusal for an unknown upright mode")


# ----- how it composes with editing.geometry ------------------------------

def test_it_composes_with_the_geometry_stage_in_that_order() -> None:
    """Transform runs BEFORE straighten-and-crop, so the full chain in
    normalized coordinates is G @ T. Verified on pixels, because that is the
    claim the caller has to be able to trust when it folds the two matrices into
    one warp."""
    W, H = 800, 600
    img = np.zeros((H, W, 3), np.uint8)
    img[290:310, 190:210] = 255                     # a mark at (0.25, 0.5)
    p = {"vertical": 30.0, "rotate": 4.0, "scale": 120.0}
    edit = {"tilt": 6.0, "crop": {"x": .1, "y": .1, "w": .7, "h": .7}}

    staged = editing.apply_geometry(transform.apply_transform(img, p), edit)
    got = _bbox(staged)

    t = transform.norm_matrix(p, W / H)
    g = np.asarray(editing.geometry_norm_matrix(W, H, edit)).reshape(2, 3)
    chain = np.vstack([g, [0.0, 0.0, 1.0]]) @ t
    want = _at(chain, 0.25, 0.5)
    assert abs(got[0] - want[0]) < 0.01 and abs(got[1] - want[1]) < 0.01, \
        f"pixels put the mark at {got[:2]}, G @ T said {want}"


def test_the_transform_stage_hands_on_the_size_it_was_given() -> None:
    """Which is what lets `geometry_matrix` downstream stay exactly as it is."""
    for w, h in ((400, 300), (1600, 1200), (501, 333)):
        img = np.zeros((h, w, 3), np.uint8)
        out = transform.apply_transform(img, {"vertical": 40.0, "rotate": 3.0})
        assert out.shape == img.shape, f"{w}x{h} came back {out.shape}"


# ----- cost ---------------------------------------------------------------

def test_the_cost_is_bounded_in_megapixels() -> None:
    """The estimate always measures at EST_LONG, so a 24 MP frame costs what a
    2 MP one does plus the downscale. Reported rather than asserted tightly,
    since a CI machine's absolute timings are not a contract."""
    frame = cv2.resize(_grid(1600, 1200, roll=5.0), (6000, 4000),
                       interpolation=cv2.INTER_LINEAR)
    assert frame.shape[0] * frame.shape[1] >= 24_000_000

    t0 = time.perf_counter()
    est = transform.estimate_upright(frame, "full")
    t1 = time.perf_counter()
    out = transform.apply_transform(frame, est)
    t2 = time.perf_counter()
    print(f"    24 MP: estimate {1e3 * (t1 - t0):.0f} ms, warp {1e3 * (t2 - t1):.0f} ms")
    assert out.shape == frame.shape
    assert abs(est["rotate"] - 5.0) < 0.5, f"lost the roll at 24 MP: {est['rotate']}"
    assert t1 - t0 < 2.0, "the estimate is not supposed to scale with the frame"


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
