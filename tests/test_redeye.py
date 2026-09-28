"""Unit checks for red-eye and pet-eye repair. Runnable with pytest or directly:

    uv run python tests/test_redeye.py

The images here are synthetic on purpose: a real red eye cannot be checked into
a repository, and the measurements that matter (is the catchlight still there, is
the pupil neutral, is the edge feathered) are all relative and hold on a drawn
eye exactly as they do on a photographed one.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pickapicka import redeye

SIZE = 200          # every fixture is this square, eye at the centre
SKIN = (0.78, 0.60, 0.50)
SCLERA = (0.92, 0.90, 0.88)
IRIS = (0.32, 0.22, 0.16)
PUPIL_RED = (0.78, 0.18, 0.16)
CATCH = (0.98, 0.96, 0.93)


def _fill(colour: tuple[float, float, float]) -> np.ndarray:
    img = np.empty((SIZE, SIZE, 3), dtype=np.float32)
    img[:] = colour
    return img


def _ellipse(img: np.ndarray, centre, axes, colour) -> None:
    cv2.ellipse(img, centre, axes, 0, 0, 360, colour, -1, cv2.LINE_AA)


def _disc(img: np.ndarray, centre, radius, colour) -> None:
    cv2.circle(img, centre, radius, colour, -1, cv2.LINE_AA)


def _red_eye(catchlight: bool = True, pupil=PUPIL_RED) -> np.ndarray:
    """An eye in a face: skin, sclera, iris, a glowing pupil, a catchlight."""
    img = _fill(SKIN)
    c = SIZE // 2
    _ellipse(img, (c, c), (34, 17), SCLERA)
    _disc(img, (c, c), 15, IRIS)
    _disc(img, (c, c), 8, pupil)
    if catchlight:
        _disc(img, (c - 3, c - 4), 3, CATCH)
    return img


def _pet_eye() -> np.ndarray:
    """A cat's eye: dark fur, and a green-yellow tapetal return filling it."""
    img = _fill((0.10, 0.09, 0.08))
    c = SIZE // 2
    _ellipse(img, (c, c), (22, 20), (0.20, 0.18, 0.14))
    _disc(img, (c, c), 13, (0.55, 0.86, 0.26))
    return img


def _red_jumper() -> np.ndarray:
    """Red wool: red everywhere, so there is no surround to be redder than."""
    rng = np.random.default_rng(3)
    img = _fill((0.62, 0.11, 0.13))
    img += rng.normal(0.0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _red_lips() -> np.ndarray:
    """Lipstick: red, but wide, and darker than the skin around it."""
    img = _fill(SKIN)
    c = SIZE // 2
    _ellipse(img, (c, c), (52, 15), (0.58, 0.14, 0.18))
    return img


def _red_light() -> np.ndarray:
    """A brake light: a compact, round, very saturated red disc on bodywork."""
    img = _fill((0.14, 0.14, 0.15))
    c = SIZE // 2
    _disc(img, (c, c), 12, (1.0, 0.03, 0.04))
    return img


def _white_shirt() -> np.ndarray:
    """Bright and round but with no dark surround: not a tapetal reflection."""
    img = _fill((0.82, 0.82, 0.80))
    _disc(img, (SIZE // 2, SIZE // 2), 13, (0.95, 0.95, 0.94))
    return img


def _correction(**kw) -> dict:
    # A user circles the pupil a little generously, which is why the radius
    # here is wider than the 8 px pupil the fixtures draw.
    c = {"kind": "red", "cx": 0.5, "cy": 0.5, "r": 11.0 / SIZE,
         "amount": 100, "darken": 50}
    c.update(kw)
    return {"enabled": True, "corrections": [c]}


def _ring_mask(cx: float, cy: float, r0: float, r1: float) -> np.ndarray:
    ys = (np.arange(SIZE, dtype=np.float32) + 0.5 - cy)[:, None]
    xs = (np.arange(SIZE, dtype=np.float32) + 0.5 - cx)[None, :]
    d = np.sqrt(xs * xs + ys * ys)
    return (d >= r0) & (d <= r1)


# ----- schema -------------------------------------------------------------

def test_neutral_normalizes_to_none() -> None:
    assert redeye.normalize(None) is None
    assert redeye.normalize({}) is None
    assert redeye.normalize(redeye.DEFAULT_REDEYE) is None
    assert redeye.normalize({"enabled": True, "corrections": []}) is None
    # Switched on but every correction a no-op is still nothing.
    assert redeye.normalize({"enabled": True, "corrections": [
        {"kind": "red", "cx": 0.5, "cy": 0.5, "r": 0.02, "amount": 0},
        {"kind": "red", "cx": 0.5, "cy": 0.5, "r": 0.0, "amount": 100},
        {"kind": "spaceship", "cx": 0.5, "cy": 0.5, "r": 0.02},
        "not a dict",
    ]}) is None
    # A correction with something to do survives, and switching off hides it.
    live = {"enabled": True, "corrections": [{"kind": "red", "cx": 0.4,
                                              "cy": 0.6, "r": 0.02}]}
    assert redeye.normalize(live) is not None
    assert redeye.normalize({**live, "enabled": False}) is None


def test_is_neutral() -> None:
    assert redeye.is_neutral(None)
    assert redeye.is_neutral({})
    assert redeye.is_neutral(redeye.DEFAULT_REDEYE)
    assert not redeye.is_neutral(_correction())


def test_normalize_clamps_and_caps() -> None:
    p = redeye.normalize({"enabled": True, "corrections": [
        {"kind": "red", "cx": 9.0, "cy": -4.0, "r": 5.0, "amount": 900,
         "darken": -50, "cat_x": -9, "cat_size": 999},
    ]})
    c = p["corrections"][0]
    assert (c["cx"], c["cy"]) == (1.0, 0.0)
    assert c["r"] == redeye._MAX_RADIUS
    assert (c["amount"], c["darken"]) == (100, 0)
    assert c["cat_x"] == -0.70 and c["cat_size"] == 60
    # A red correction never claims to draw a catchlight; a pet one may.
    assert c["catchlight"] is False
    pet = redeye.normalize_correction({"kind": "pet", "cx": 0.5, "cy": 0.5,
                                       "r": 0.02})
    assert pet["catchlight"] is True
    many = redeye.normalize({"enabled": True, "corrections": [
        {"kind": "red", "cx": 0.5, "cy": 0.5, "r": 0.02}] * 200})
    assert len(many["corrections"]) == redeye.CORRECTION_MAX


def test_neutral_apply_is_identity() -> None:
    img = _red_eye()
    for params in (None, {}, redeye.DEFAULT_REDEYE,
                   {"enabled": True, "corrections": []}):
        out = redeye.apply_redeye(img, params)
        assert out is img, f"neutral params copied the frame: {params}"
    # A disc that misses the window entirely must not touch it either.
    off = redeye.apply_redeye(img, _correction(cx=0.5, cy=0.5, r=0.02),
                              roi=(0.9, 0.9, 0.1, 0.1))
    assert off is img


# ----- the red-eye correction ---------------------------------------------

def test_red_pupil_is_desaturated_and_darkened() -> None:
    img = _red_eye(catchlight=False)
    out = redeye.apply_redeye(img, _correction())
    core = _ring_mask(SIZE / 2, SIZE / 2, 0.0, 5.0)
    before, after = img[core], out[core]
    spread_before = float((before.max(axis=1) - before.min(axis=1)).mean())
    spread_after = float((after.max(axis=1) - after.min(axis=1)).mean())
    assert spread_before > 0.5, "fixture is not red to begin with"
    assert spread_after < 0.1 * spread_before, f"still saturated: {spread_after}"
    assert after.mean() < 0.5 * before.mean(), "pupil was not darkened"
    # Nothing outside the eye moved.
    far = _ring_mask(SIZE / 2, SIZE / 2, 40.0, 90.0)
    assert np.allclose(out[far], img[far], atol=1e-6)


def test_corrected_pupil_is_colour_neutral() -> None:
    out = redeye.apply_redeye(_red_eye(catchlight=False), _correction())
    core = out[_ring_mask(SIZE / 2, SIZE / 2, 0.0, 6.0)]
    spread = core.max(axis=1) - core.min(axis=1)
    assert float(spread.max()) < 0.02, \
        f"grey with a red cast left in it: spread {float(spread.max())}"


def test_catchlight_survives() -> None:
    img = _red_eye(catchlight=True)
    out = redeye.apply_redeye(img, _correction())
    cx, cy = SIZE / 2 - 3, SIZE / 2 - 4
    spot = _ring_mask(cx, cy, 0.0, 2.0)
    around = _ring_mask(cx, cy, 5.0, 7.5) & _ring_mask(SIZE / 2, SIZE / 2, 0.0, 8.0)
    spot_lum = float(redeye._luma(out)[spot].mean())
    around_lum = float(redeye._luma(out)[around].mean())
    assert spot_lum > around_lum + 0.25, \
        f"catchlight lost: {spot_lum:.3f} vs {around_lum:.3f} around it"
    # And it kept most of its own brightness rather than merely staying relatively bright.
    assert spot_lum > 0.8 * float(redeye._luma(img)[spot].mean())


def test_edge_is_feathered() -> None:
    # A flat red field, so the only thing shaping the result is the disc alpha.
    img = _fill((0.80, 0.14, 0.12))
    rad = 40.0
    out = redeye.apply_redeye(img, _correction(r=rad / SIZE))
    row = SIZE // 2
    delta = np.abs(out[row] - img[row]).sum(axis=1)
    peak = float(delta.max())
    assert peak > 0.5, "nothing was corrected"
    step = float(np.abs(np.diff(delta)).max())
    assert step < 0.15 * peak, f"hard edge: one pixel moved {step / peak:.2f} of the drop"
    band = (delta > 0.05 * peak) & (delta < 0.95 * peak)
    # The transition runs over the outer _EDGE_FEATHER of the radius, twice
    # (once each side); a hard edge would put a handful of pixels here at most.
    assert int(band.sum()) > 12, f"transition only {int(band.sum())} px wide"


def test_amount_and_darken_scale_monotonically() -> None:
    img = _red_eye(catchlight=False)
    core = _ring_mask(SIZE / 2, SIZE / 2, 0.0, 5.0)

    def core_lum(**kw) -> float:
        out = redeye.apply_redeye(img, _correction(**kw))
        return float(redeye._luma(out)[core].mean())

    by_amount = [core_lum(amount=a) for a in (1, 25, 50, 75, 100)]
    assert all(b < a for a, b in zip(by_amount, by_amount[1:])), by_amount
    by_darken = [core_lum(darken=d) for d in (0, 25, 50, 75, 100)]
    assert all(b < a for a, b in zip(by_darken, by_darken[1:])), by_darken


def test_pet_amount_and_darken_scale_monotonically() -> None:
    img = _pet_eye()
    core = _ring_mask(SIZE / 2, SIZE / 2, 0.0, 8.0)

    def core_lum(**kw) -> float:
        out = redeye.apply_redeye(img, _correction(
            kind="pet", r=18.0 / SIZE, catchlight=False, **kw))
        return float(redeye._luma(out)[core].mean())

    by_amount = [core_lum(amount=a, darken=70) for a in (1, 25, 50, 75, 100)]
    assert all(b < a for a, b in zip(by_amount, by_amount[1:])), by_amount
    by_darken = [core_lum(darken=d) for d in (0, 25, 50, 75, 100)]
    assert all(b < a for a, b in zip(by_darken, by_darken[1:])), by_darken


def test_both_eyes_of_a_pair_are_corrected() -> None:
    img = _fill(SKIN)
    for cx in (60, 140):
        _ellipse(img, (cx, 100), (26, 14), SCLERA)
        _disc(img, (cx, 100), 12, IRIS)
        _disc(img, (cx, 100), 7, PUPIL_RED)
    params = {"enabled": True, "corrections": [
        {"kind": "red", "cx": cx / SIZE, "cy": 0.5, "r": 10.0 / SIZE,
         "amount": 100, "darken": 50} for cx in (60, 140)]}
    out = redeye.apply_redeye(img, params)
    for cx in (60, 140):
        core = out[_ring_mask(cx, 100, 0.0, 4.0)]
        assert float((core.max(axis=1) - core.min(axis=1)).max()) < 0.02, \
            f"the eye at x={cx} was left red"


def test_work_stays_inside_the_eye() -> None:
    """One small correction on a big frame touches only the disc's own box —
    this is the claim that cost scales with the eye and not the megapixels."""
    big = np.empty((900, 1200, 3), dtype=np.float32)
    big[:] = (0.80, 0.14, 0.12)
    r = 0.01                      # 12 px on this frame
    out = redeye.apply_redeye(big, _correction(cx=0.25, cy=0.75, r=r))
    changed = np.argwhere(np.abs(out - big).max(axis=2) > 1e-6)
    assert changed.size > 0, "nothing was corrected"
    reach = r * 1200 * (1.0 + redeye._EDGE_FEATHER) + 1.0
    ys, xs = changed[:, 0], changed[:, 1]
    assert xs.min() >= 0.25 * 1200 - reach - 1 and xs.max() <= 0.25 * 1200 + reach
    assert ys.min() >= 0.75 * 900 - reach - 1 and ys.max() <= 0.75 * 900 + reach


def test_bad_input_fails_loudly() -> None:
    params = _correction()
    with pytest.raises(AssertionError):
        redeye.apply_redeye((_red_eye() * 255).astype(np.uint8), params)
    with pytest.raises(AssertionError):
        redeye.apply_redeye(_red_eye()[..., 0], params)
    with pytest.raises(AssertionError):
        redeye.apply_redeye(_red_eye(), params, roi=(0.0, 0.0, 0.0, 1.0))
    with pytest.raises(AssertionError):
        redeye.detect_in_region(_red_eye(), 0.5, 0.5, 0.05, "cyclops")


# ----- the pet-eye correction ---------------------------------------------

def test_pet_eye_kills_a_green_reflection_a_red_test_would_miss() -> None:
    img = _pet_eye()
    core = _ring_mask(SIZE / 2, SIZE / 2, 0.0, 8.0)
    params = _correction(kind="pet", r=17.0 / SIZE, darken=100,
                         catchlight=False)
    # The red path is blind to it: green ahead of red means no redness at all.
    as_red = redeye.apply_redeye(img, _correction(r=17.0 / SIZE, darken=100))
    assert np.allclose(as_red[core], img[core], atol=1e-6), \
        "the red-channel path must not be what fixes a green reflection"
    out = redeye.apply_redeye(img, params)
    before, after = img[core], out[core]
    assert float(redeye._luma(out)[core].mean()) < \
        0.25 * float(redeye._luma(img)[core].mean()), "reflection not darkened"
    spread_after = float((after.max(axis=1) - after.min(axis=1)).mean())
    spread_before = float((before.max(axis=1) - before.min(axis=1)).mean())
    assert spread_after < 0.1 * spread_before, f"still green: {spread_after}"


def test_pet_catchlight_lands_where_asked() -> None:
    img = _pet_eye()
    rad, inner = 18.0, 13.0
    out = redeye.apply_redeye(img, _correction(
        kind="pet", r=rad / SIZE, darken=80, catchlight=True,
        cat_x=0.5, cat_y=-0.5, cat_size=25))
    lum = redeye._luma(out)
    disc = _ring_mask(SIZE / 2, SIZE / 2, 0.0, inner)
    masked = np.where(disc, lum, -1.0)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    assert x > SIZE / 2 and y < SIZE / 2, \
        f"catchlight asked for up-right, brightest pixel at ({x}, {y})"
    assert float(masked.max()) > 0.8, "catchlight is not bright"
    # Moving it to the other side moves the highlight with it.
    other = redeye.apply_redeye(img, _correction(
        kind="pet", r=rad / SIZE, darken=80, catchlight=True,
        cat_x=-0.5, cat_y=0.5, cat_size=25))
    oy, ox = np.unravel_index(
        int(np.argmax(np.where(disc, redeye._luma(other), -1.0))), disc.shape)
    assert ox < SIZE / 2 and oy > SIZE / 2, f"mirrored request landed at ({ox}, {oy})"
    dark = redeye.apply_redeye(img, _correction(
        kind="pet", r=rad / SIZE, darken=80, catchlight=False))
    assert float(redeye._luma(dark)[disc].max()) < 0.5, \
        "without the toggle there should be no highlight to find"


def test_pet_catchlight_stays_inside_the_eye() -> None:
    """Asked for at the rim and as big as it goes, it must not spill onto fur."""
    img = _pet_eye()
    rad = 16.0
    out = redeye.apply_redeye(img, _correction(
        kind="pet", r=rad / SIZE, darken=90, catchlight=True,
        cat_x=0.7, cat_y=0.7, cat_size=60))
    outside = _ring_mask(SIZE / 2, SIZE / 2, rad + 2.0, rad + 12.0)
    assert float(redeye._luma(out)[outside].max()) < 0.3, \
        "the catchlight leaked past the eye"


# ----- the roi contract ---------------------------------------------------

def test_roi_window_matches_the_full_frame() -> None:
    img = _red_eye()
    x0, y0, side = 60, 40, 120
    for kind, extra in (("red", {}), ("pet", {"catchlight": True})):
        params = _correction(kind=kind, cx=0.5, cy=0.5, r=10.0 / SIZE,
                             darken=60, **extra)
        full = redeye.apply_redeye(img, params)
        win = np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side])
        roi = (x0 / SIZE, y0 / SIZE, side / SIZE, side / SIZE)
        out = redeye.apply_redeye(win, params, roi)
        assert np.allclose(out, full[y0:y0 + side, x0:x0 + side], atol=1e-6), \
            f"{kind}: the window render drifted from the full frame"


def test_padding_is_zero_and_a_straddled_edge_proves_it() -> None:
    """The claim `padding` makes: a window needs no surrounding image, even when
    a disc hangs off its edge."""
    assert redeye.padding(_correction(), SIZE, SIZE) == 0.0
    assert redeye.padding(None, SIZE, SIZE) == 0.0
    img = _red_eye()
    for kind, extra in (("red", {}), ("pet", {"catchlight": True})):
        params = _correction(kind=kind, cx=0.5, cy=0.5, r=14.0 / SIZE,
                             darken=60, **extra)
        full = redeye.apply_redeye(img, params)
        # The window's right edge runs through the middle of the disc and its
        # top edge cuts across it, so most of the disc is outside the window.
        x0, y0, side = 40, 90, 60
        reach = 14.0 * (1.0 + redeye._EDGE_FEATHER)
        assert x0 + side < SIZE / 2 + reach and y0 > SIZE / 2 - reach, \
            "this window is meant to clip the disc, not contain it"
        win = np.ascontiguousarray(img[y0:y0 + side, x0:x0 + side])
        roi = (x0 / SIZE, y0 / SIZE, side / SIZE, side / SIZE)
        out = redeye.apply_redeye(win, params, roi)
        assert not np.allclose(out, win, atol=1e-6), \
            "the clipped part of the disc was not corrected at all"
        assert np.allclose(out, full[y0:y0 + side, x0:x0 + side], atol=1e-6), \
            f"{kind}: a disc clipped by the window drifted from the full frame"


# ----- detection ----------------------------------------------------------
#
# Note on what is tested here. Locating a face is a learned model's job, and no
# drawn fixture will trigger either FaceMesh or a Haar cascade, so the automatic
# pass cannot be shown finding an eye in a synthetic frame — `detect` on any
# fixture below correctly returns nothing, because it finds no face. What *is*
# testable, and is where all the rejection power lives, is the verifier: it runs
# on a user-drawn circle through `detect_in_region`, and the same function
# decides every automatic candidate.

def _all_fixtures() -> dict[str, np.ndarray]:
    return {"red eye": _red_eye(), "pet eye": _pet_eye(),
            "red jumper": _red_jumper(), "red lips": _red_lips(),
            "red light": _red_light(), "white shirt": _white_shirt()}


def test_detection_does_not_fire_on_things_that_are_merely_red() -> None:
    for name, img in (("red jumper", _red_jumper()), ("red lips", _red_lips()),
                      ("red light", _red_light())):
        assert redeye.detect(img) == [], f"automatic pass fired on a {name}"
        # And it stays rejected even when a user points straight at it, which
        # is the only way the verifier ever sees such a thing.
        for r in (6.0, 12.0, 20.0):
            got = redeye.detect_in_region(img, 0.5, 0.5, r / SIZE, "red")
            assert got is None, f"verifier accepted a {name} at r={r}: {got}"


def test_detection_does_not_fire_on_a_bright_thing_with_no_dark_surround() -> None:
    img = _white_shirt()
    assert redeye.detect_in_region(img, 0.5, 0.5, 13.0 / SIZE, "pet") is None
    assert redeye.detect(img) == []


def test_verifier_finds_a_red_pupil_and_a_tapetal_reflection() -> None:
    eye = redeye.detect_in_region(_red_eye(), 0.5, 0.5, 8.0 / SIZE, "red")
    assert eye is not None, "the verifier missed an actual red pupil"
    assert abs(eye["cx"] - 0.5) < 0.02 and abs(eye["cy"] - 0.5) < 0.02
    assert 0.03 < eye["r"] * SIZE < 16.0, f"radius nonsense: {eye['r'] * SIZE}"
    assert eye["kind"] == "red" and eye["catchlight"] is False
    # A slightly off-centre click still lands on the pupil.
    near = redeye.detect_in_region(_red_eye(), 0.5 + 4.0 / SIZE, 0.5,
                                   8.0 / SIZE, "red")
    assert near is not None and abs(near["cx"] - 0.5) < 0.02

    pet = redeye.detect_in_region(_pet_eye(), 0.5, 0.5, 13.0 / SIZE, "pet")
    assert pet is not None, "the verifier missed a tapetal reflection"
    assert pet["kind"] == "pet" and pet["catchlight"] is True
    # The two paths are genuinely different: neither finds the other's eye.
    assert redeye.detect_in_region(_pet_eye(), 0.5, 0.5, 13.0 / SIZE, "red") is None


def test_what_the_verifier_returns_is_a_usable_correction() -> None:
    # No catchlight in this one, so the measured core is nothing but corrected
    # pupil: a preserved catchlight is a colour of its own and is checked above.
    img = _red_eye(catchlight=False)
    found = redeye.detect_in_region(img, 0.5, 0.5, 8.0 / SIZE, "red")
    params = redeye.normalize({"enabled": True, "corrections": [found]})
    assert params is not None, "detection produced a correction normalize drops"
    out = redeye.apply_redeye(img, params)
    core = out[_ring_mask(SIZE / 2, SIZE / 2, 0.0, 5.0)]
    assert float((core.max(axis=1) - core.min(axis=1)).max()) < 0.02


def test_detection_is_deterministic() -> None:
    for name, img in _all_fixtures().items():
        first = redeye.detect(img)
        assert redeye.detect(img) == first, f"detect wandered on a {name}"
        a = redeye.detect_in_region(img, 0.5, 0.5, 10.0 / SIZE, "red")
        b = redeye.detect_in_region(img, 0.5, 0.5, 10.0 / SIZE, "pet")
        assert redeye.detect_in_region(img, 0.5, 0.5, 10.0 / SIZE, "red") == a
        assert redeye.detect_in_region(img, 0.5, 0.5, 10.0 / SIZE, "pet") == b


def _pair_of_eyes() -> np.ndarray:
    """Two eyes in a face-like arrangement, plus a red blemish on a cheek."""
    img = _fill(SKIN)
    for cx in (70, 130):
        _ellipse(img, (cx, 80), (24, 13), SCLERA)
        _disc(img, (cx, 80), 11, IRIS)
        _disc(img, (cx, 80), 6, PUPIL_RED)
        _disc(img, (cx - 2, 77), 2, CATCH)
    _disc(img, (100, 140), 7, (0.72, 0.24, 0.24))   # a spot, not an eye
    return img


def test_detect_runs_the_verifier_over_the_candidates(monkeypatch) -> None:
    """The automatic pass end to end, with the one part that cannot fire on a
    drawing replaced by a stub.

    No Haar cascade or FaceMesh will ever find a face in a synthetic fixture, so
    the localiser is stubbed to propose both eyes *and* the cheek spot. What is
    being checked is the real thing: that every proposal goes through the
    verifier, that the eyes come back as usable corrections in whole-frame
    normalized coordinates, and that the spot does not.
    """
    img = _pair_of_eyes()
    proposals = [(130.0, 80.0, 7.0), (70.0, 80.0, 7.0), (100.0, 140.0, 7.0)]
    monkeypatch.setattr(redeye, "_eye_candidates", lambda u8: list(proposals))
    got = redeye.detect(img)
    assert len(got) == 2, f"expected two eyes and no cheek spot, got {got}"
    # Sorted by position, so the left eye comes first at equal height.
    assert got[0]["cx"] < got[1]["cx"]
    for c, want in zip(got, (70, 130)):
        assert abs(c["cx"] * SIZE - want) < 2.0 and abs(c["cy"] * SIZE - 80) < 2.0
        assert c["kind"] == "red" and c["amount"] > 0
    out = redeye.apply_redeye(img, {"enabled": True, "corrections": got})
    for cx in (70, 130):
        # Everything but the catchlight, which is meant to keep its own colour.
        core = out[_ring_mask(cx, 80, 0.0, 4.0)
                   & ~_ring_mask(cx - 2, 77, 0.0, 3.5)]
        assert float((core.max(axis=1) - core.min(axis=1)).max()) < 0.02, \
            f"the eye at x={cx} is still red after its own correction"


def test_the_verifier_and_not_just_the_face_gate_does_the_rejecting(monkeypatch) -> None:
    """Hand the verifier the red things directly, as if a face had been found
    right on top of them. Every one must still be refused — otherwise the
    geometric gate would be the only thing standing between a red jumper and a
    grey patch."""
    for name, img in (("red jumper", _red_jumper()), ("red lips", _red_lips()),
                      ("red light", _red_light()),
                      ("white shirt", _white_shirt())):
        monkeypatch.setattr(redeye, "_eye_candidates",
                            lambda u8: [(100.0, 100.0, r) for r in (5.0, 9.0, 16.0)])
        assert redeye.detect(img) == [], f"the verifier accepted a {name}"


def test_detection_needs_a_surround_to_judge_by() -> None:
    # An eye jammed against the frame edge has no ring to measure, and the
    # verifier refuses rather than guessing.
    img = _red_eye()
    assert redeye.detect_in_region(img, 0.004, 0.5, 8.0 / SIZE, "red") is None


# Delegated rather than hand-rolled, the way test_lens and test_segment do it:
# the detector tests take pytest's monkeypatch to stub `_eye_candidates`, and a
# loop over globals() cannot supply one. It used to skip those four and still
# print "24 checks passed", which reads as a full pass of a suite of 26.
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
