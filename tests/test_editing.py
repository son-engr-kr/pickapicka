"""Unit checks for the editing pipeline. Runnable with pytest or directly:

    uv run python tests/test_editing.py
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from picture_classifier import editing


def _sample() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


def test_neutral_is_identity() -> None:
    img = _sample()
    for edit in (None, {}, editing.DEFAULT_EDIT, {"exposure": 0, "curve": [[0, 0], [1, 1]]}):
        out = editing.render(img, edit)
        assert out.dtype == np.uint8 and out.shape == img.shape
        assert np.array_equal(out, img), f"neutral edit changed pixels: {edit}"


def test_is_neutral() -> None:
    assert editing.is_neutral(None)
    assert editing.is_neutral({})
    assert editing.is_neutral({"curve": [[0, 0], [0.5, 0.5], [1, 1]]})
    assert not editing.is_neutral({"exposure": 0.5})
    assert not editing.is_neutral({"saturation": 20})
    assert not editing.is_neutral({"curve": [[0, 0], [0.5, 0.6], [1, 1]]})


def test_exposure_direction() -> None:
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert editing.render(img, {"exposure": 1.0}).mean() > img.mean()
    assert editing.render(img, {"exposure": -1.0}).mean() < img.mean()


def test_saturation_extremes() -> None:
    img = _sample()
    gray = editing.render(img, {"saturation": -100}).astype(np.float32)
    # Full desaturation collapses channels toward a common luma.
    spread = gray.max(axis=2) - gray.min(axis=2)
    assert spread.mean() < 2.0
    more = editing.render(img, {"saturation": 100}).astype(np.float32)
    orig_spread = img.astype(np.float32).max(axis=2) - img.astype(np.float32).min(axis=2)
    assert (more.max(axis=2) - more.min(axis=2)).mean() > orig_spread.mean()


def test_curve_lut_monotonic() -> None:
    lut = editing._curve_lut([[0, 0], [0.3, 0.5], [0.7, 0.6], [1, 1]], 256)
    assert lut[0] <= 1e-6 and lut[-1] >= 1 - 1e-6
    assert np.all(np.diff(lut) >= -1e-6), "monotone curve produced a non-monotone LUT"
    ident = editing._curve_lut([[0, 0], [1, 1]], 256)
    assert np.allclose(ident, np.linspace(0, 1, 256), atol=1e-6)


def test_curve_brighten_then_render() -> None:
    img = np.full((16, 16, 3), 128, dtype=np.uint8)
    # A curve lifting the midtone should brighten a mid-gray image.
    out = editing.render(img, {"curve": [[0, 0], [0.5, 0.7], [1, 1]]})
    assert out.mean() > img.mean()


def test_auto_tone_shape() -> None:
    dark = (_sample().astype(np.float32) * 0.3).astype(np.uint8)
    edit = editing.auto_tone(dark)
    assert set(editing.DEFAULT_EDIT).issubset(edit)
    for k, (lo, hi) in editing._RANGES.items():
        assert lo - 1e-6 <= float(edit[k]) <= hi + 1e-6
    # A dark image should suggest a positive exposure lift.
    assert edit["exposure"] > 0


def _cast(rgb: tuple[float, float, float], level: float = 0.55) -> np.ndarray:
    """A flat patch of one colour, as the picker will meet it."""
    px = np.array(rgb, dtype=np.float32) * level * 255.0
    return np.clip(np.broadcast_to(px, (48, 64, 3)), 0, 255).astype(np.uint8)


def test_neutral_wb_leaves_grey_alone() -> None:
    wb = editing.neutral_wb(_cast((1.0, 1.0, 1.0)), 0.5, 0.5)
    assert wb == {"temp": 0, "tint": 0, "clamped": False}


def test_neutral_wb_cancels_a_cast() -> None:
    # A warm indoor cast, and the pair the picker returns must actually render
    # it grey — the round trip through the pipeline is the only claim worth
    # testing, since the gains are multiplicative and not symmetric in the
    # slider value.
    img = _cast((1.18, 1.05, 0.82))
    wb = editing.neutral_wb(img, 0.5, 0.5)
    assert wb is not None and wb["temp"] < 0 and not wb["clamped"]
    out = editing.render(img, {"temp": wb["temp"], "tint": wb["tint"]})
    px = out[24, 32].astype(int)
    assert int(px.max() - px.min()) <= 2, f"still cast: {px}"


def test_neutral_wb_reports_what_it_cannot_reach() -> None:
    # Deep tungsten: further than 1.25/0.75 goes, so the answer is pinned and
    # says so rather than pretending.
    wb = editing.neutral_wb(_cast((1.45, 1.1, 0.68)), 0.5, 0.5)
    assert wb is not None and wb["clamped"]
    assert wb["temp"] == -100


def test_neutral_wb_needs_a_readable_patch() -> None:
    assert editing.neutral_wb(np.zeros((16, 16, 3), np.uint8), 0.5, 0.5) is None
    assert editing.neutral_wb(np.full((16, 16, 3), 255, np.uint8), 0.5, 0.5) is None


def test_neutral_wb_reads_where_it_is_pointed() -> None:
    img = np.concatenate([_cast((1.2, 1.0, 0.8)), _cast((0.8, 1.0, 1.2))], axis=1)
    warm = editing.neutral_wb(img, 0.25, 0.5)
    cool = editing.neutral_wb(img, 0.75, 0.5)
    assert warm["temp"] < 0 < cool["temp"]


def test_neutral_wb_survives_a_speck() -> None:
    # One blown pixel inside the sampled patch must not move the answer: the
    # patch is reduced by its median for exactly this case.
    img = _cast((1.18, 1.05, 0.82))
    clean = editing.neutral_wb(img, 0.5, 0.5)
    img[24, 32] = (255, 0, 0)
    assert editing.neutral_wb(img, 0.5, 0.5) == clean


def test_edit_hash() -> None:
    assert editing.edit_hash(None) == ""
    assert editing.edit_hash({}) == ""
    h1 = editing.edit_hash({"exposure": 0.5})
    h2 = editing.edit_hash({"exposure": 0.5})
    h3 = editing.edit_hash({"exposure": 0.6})
    assert h1 and h1 == h2 and h1 != h3


def test_normalize_clamps_and_repairs() -> None:
    e = editing.normalize({"exposure": 99, "saturation": -999, "sharpen": -5,
                           "curve": [[2, 2], [-1, 0.5]]})
    assert e["exposure"] == 2.0 and e["saturation"] == -100 and e["sharpen"] == 0
    xs = [p[0] for p in e["curve"]]
    assert xs[0] == 0.0 and xs[-1] == 1.0 and xs == sorted(xs)


def test_full_stack_runs() -> None:
    img = _sample()
    edit = {"exposure": 0.4, "contrast": 20, "highlights": -30, "shadows": 25,
            "whites": 10, "blacks": -10, "temp": 15, "tint": -8, "vibrance": 30,
            "saturation": 10, "clarity": 40, "sharpen": 50, "vignette": -20,
            "curve": [[0, 0], [0.25, 0.2], [0.75, 0.85], [1, 1]]}
    out = editing.render(img, edit)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert not np.array_equal(out, img)


# ----- local adjustment masks ---------------------------------------------

def _radial(**over) -> dict:
    m = {"type": "radial", "cx": 0.25, "cy": 0.5, "rx": 0.15, "ry": 0.2,
         "feather": 40, "adj": {"exposure": 1.0}}
    m.update(over)
    return m


def test_mask_normalize_rejects_unusable() -> None:
    assert editing.normalize_mask({"type": "nope"}) is None
    assert editing.normalize_mask({"type": "brush", "strokes": []}) is None
    # A zero-length gradient has no direction to ramp along.
    assert editing.normalize_mask({"type": "linear", "x1": 0.5, "y1": 0.5,
                                   "x2": 0.5, "y2": 0.5}) is None
    assert editing.normalize_mask("not a dict") is None


def test_mask_normalize_clamps() -> None:
    m = editing.normalize_mask({"type": "radial", "rx": 0, "ry": 99, "feather": 500,
                                "amount": -20, "angle": 999, "name": "x" * 99,
                                "adj": {"exposure": 9, "vignette": 50, "bogus": 1}})
    assert m["rx"] >= 0.005 and m["ry"] <= 3.0
    assert m["feather"] == 100 and m["amount"] == 0
    assert -180 <= m["angle"] <= 180
    assert len(m["name"]) == 40
    assert m["adj"]["exposure"] == 2.0
    # Only local keys survive: vignette and the curve stay global.
    assert set(m["adj"]) == set(editing.LOCAL_KEYS)


def test_normalize_is_idempotent_for_masks() -> None:
    once = editing.normalize({"masks": [_radial(), {"type": "brush", "feather": 20,
                                                    "strokes": [{"radius": 0.05,
                                                                 "points": [[0.2, 0.2], [0.5, 0.3]]}]}]})
    assert editing.normalize(once) == once


def test_mask_neutrality_and_hash() -> None:
    assert editing.is_neutral({"masks": [_radial(adj={})]})       # shape but no sliders
    assert editing.is_neutral({"masks": [_radial(amount=0)]})     # zero strength
    assert editing.is_neutral({"masks": [_radial(enabled=False)]})
    assert not editing.is_neutral({"masks": [_radial()]})
    assert editing.edit_hash({"masks": [_radial()]}) != editing.edit_hash({"masks": [_radial(cx=0.75)]})


def test_mask_is_local() -> None:
    img = np.full((200, 300, 3), 100, dtype=np.uint8)
    out = editing.render(img, {"masks": [_radial()]})
    assert out[100, 75].mean() > 140, "inside the ellipse should be brightened"
    assert np.array_equal(out[100, 280], img[100, 280]), "far outside must be untouched"


def test_mask_invert_flips_the_area() -> None:
    img = np.full((200, 300, 3), 100, dtype=np.uint8)
    plain = editing.render(img, {"masks": [_radial()]})
    flipped = editing.render(img, {"masks": [_radial(invert=True)]})
    assert plain[100, 75].mean() > flipped[100, 75].mean()
    assert flipped[100, 280].mean() > plain[100, 280].mean()


def test_linear_mask_ramps_along_its_axis() -> None:
    img = np.full((200, 300, 3), 100, dtype=np.uint8)
    mask = {"type": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 1.0,
            "adj": {"exposure": -1.0}}
    out = editing.render(img, {"masks": [mask]}).astype(np.float32)
    top, mid, bottom = out[2, 150].mean(), out[100, 150].mean(), out[197, 150].mean()
    assert top < mid < bottom, f"expected a monotone ramp, got {top}/{mid}/{bottom}"
    assert bottom == 100


def test_brush_mask_paints_only_the_stroke() -> None:
    img = np.full((200, 300, 3), 100, dtype=np.uint8)
    mask = {"type": "brush", "feather": 30, "adj": {"exposure": 1.0},
            "strokes": [{"radius": 0.04, "points": [[0.2, 0.25], [0.8, 0.25]]}]}
    out = editing.render(img, {"masks": [mask]})
    assert out[50, 150].mean() > 140, "on the stroke"
    assert np.array_equal(out[170, 150], img[170, 150]), "far from the stroke"


def test_a_brush_click_paints_exactly_the_brush() -> None:
    """One click is a disc of the brush radius — no more, and round.

    The overlay preview mirrors this on canvas; it used to stroke the circle
    with a pen as wide as the brush's diameter, which painted a blob of twice
    the radius and made a click look nothing like what was applied.
    """
    h, w = 800, 1200
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    radius = 0.06                       # a fraction of the width, as stored
    mask = {"type": "brush", "feather": 0, "adj": {"exposure": 2.0},
            "strokes": [{"radius": radius, "points": [[0.5, 0.5]]}]}
    out = editing.render(img, {"masks": [mask]})

    lit = out[..., 0] > 80
    xs = np.where(lit[h // 2])[0]
    ys = np.where(lit[:, w // 2])[0]
    expected = radius * w
    rx = (xs[-1] - xs[0]) / 2
    ry = (ys[-1] - ys[0]) / 2
    assert abs(rx - expected) <= 2, f"horizontal radius {rx}, expected {expected}"
    assert abs(ry - expected) <= 2, f"vertical radius {ry}, expected {expected}"
    # Centred on the click, not offset by the path-drawing code.
    assert abs((xs[0] + xs[-1]) / 2 - w / 2) <= 2
    assert abs((ys[0] + ys[-1]) / 2 - h / 2) <= 2


def test_mask_alpha_is_resolution_independent() -> None:
    small = editing.mask_alpha(editing.normalize_mask(_radial()), 100, 150)
    large = editing.mask_alpha(editing.normalize_mask(_radial()), 800, 1200)
    assert small.shape == (100, 150) and large.shape == (800, 1200)
    assert 0.0 <= float(large.min()) and float(large.max()) <= 1.0
    # The covered fraction of the frame must not depend on the render size.
    assert abs(small.mean() - large.mean()) < 0.01


def test_masks_compose_in_order() -> None:
    img = np.full((120, 160, 3), 100, dtype=np.uint8)
    a = _radial(cx=0.5, cy=0.5, rx=0.4, ry=0.4, feather=0, adj={"exposure": 1.0})
    b = _radial(cx=0.5, cy=0.5, rx=0.4, ry=0.4, feather=0, adj={"exposure": -1.0})
    both = editing.render(img, {"masks": [a, b]})
    # The second mask grades the first one's output, landing back near the start.
    assert abs(int(both[60, 80].mean()) - 100) <= 2


# ----- creative effects ---------------------------------------------------

def _detailed(h: int = 300, w: int = 400) -> np.ndarray:
    """A busy but smooth image: blur has something to destroy, and downscaling
    it for the ROI comparisons behaves."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (120 + 90 * np.sin(x / 7.0) * np.cos(y / 5.0)).clip(0, 255)
    img = np.dstack([base, base * 0.85, base * 0.7]).astype(np.uint8)
    img[h // 3:h // 2, w // 3:w // 2] = 252   # a highlight patch for glow
    return img


def _detail_energy(img: np.ndarray) -> float:
    import cv2
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


def test_effects_are_neutral_at_zero() -> None:
    img = _detailed()
    for key in ("blur", "motion", "glow", "pixelate"):
        assert np.array_equal(editing.render(img, {key: 0}), img), key
    # An angle on its own is a modifier, not an effect: nothing should change.
    assert editing.is_neutral({"motion_angle": 90})
    assert np.array_equal(editing.render(img, {"motion_angle": 90}), img)


def test_defocus_destroys_detail() -> None:
    img = _detailed()
    out = editing.render(img, {"blur": 60})
    assert _detail_energy(out) < _detail_energy(img) * 0.2


def test_motion_blur_is_directional() -> None:
    import cv2

    def xdetail(im: np.ndarray) -> float:
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return float(np.abs(np.diff(g, axis=1)).mean())

    # Vertical stripes: all the detail lies along x and none along y, so a smear
    # along x must erase it and a smear along y must leave it alone.
    h, w = 300, 400
    xx = np.mgrid[0:h, 0:w][1]
    stripes = np.where((xx // 4) % 2 == 0, 40, 215).astype(np.uint8)
    img = np.dstack([stripes, stripes, stripes])
    base = xdetail(img)

    along = xdetail(editing.render(img, {"motion": 80, "motion_angle": 0}))
    across = xdetail(editing.render(img, {"motion": 80, "motion_angle": 90}))
    assert along < base * 0.1, f"a horizontal smear must erase vertical stripes: {base} -> {along}"
    assert across > base * 0.8, f"a vertical smear must preserve them: {base} -> {across}"


def test_glow_only_lifts() -> None:
    img = _detailed()
    out = editing.render(img, {"glow": 70})
    assert (out.astype(int) >= img.astype(int) - 1).all(), "screen blend must never darken"
    assert out.mean() > img.mean()


def _block_size(out: np.ndarray) -> int:
    """Width of the first run of identical columns along the top row."""
    row = out[0]
    n = 1
    while n < row.shape[0] and np.array_equal(row[n], row[0]):
        n += 1
    return n


def test_pixelate_makes_flat_blocks() -> None:
    img = _detailed()
    out = editing.render(img, {"pixelate": 50})
    # Blockiness is the point, so measure blocks rather than detail energy —
    # hard block edges carry plenty of Laplacian energy of their own.
    block = _block_size(out)
    assert block >= 2, "no blocks formed"
    assert len(np.unique(out[:block, :block].reshape(-1, 3), axis=0)) == 1, "block not flat"
    # Bigger amount, bigger blocks.
    assert _block_size(editing.render(img, {"pixelate": 100})) > block


def test_effects_work_through_a_mask() -> None:
    img = _detailed()
    mask = {"type": "radial", "cx": 0.25, "cy": 0.5, "rx": 0.15, "ry": 0.2,
            "feather": 0, "adj": {"blur": 70}}
    out = editing.render(img, {"masks": [mask]})
    h, w = img.shape[:2]
    inside = (slice(int(h * 0.45), int(h * 0.55)), slice(int(w * 0.2), int(w * 0.3)))
    assert _detail_energy(out[inside]) < _detail_energy(img[inside]) * 0.35
    assert np.array_equal(out[:, int(w * 0.8):], img[:, int(w * 0.8):]), "outside untouched"


# ----- ROI rendering (the 1:1 editor view) --------------------------------

def test_roi_default_is_the_whole_frame() -> None:
    img = _detailed()
    edit = {"exposure": 0.4, "vignette": -50}
    assert np.array_equal(editing.render(img, edit),
                          editing.render(img, edit, roi=(0.0, 0.0, 1.0, 1.0)))


def test_roi_matches_the_same_window_of_the_full_render() -> None:
    """Grading a crop with its roi must reproduce that part of the full render —
    this is what makes the 1:1 zoom trustworthy rather than merely plausible."""
    img = _detailed(600, 800)
    h, w = img.shape[:2]
    edit = {"exposure": 0.3, "vignette": -60,
            "masks": [{"type": "radial", "cx": 0.35, "cy": 0.45, "rx": 0.2, "ry": 0.25,
                       "feather": 50, "adj": {"exposure": 0.8, "saturation": 40}},
                      {"type": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.6,
                       "adj": {"temp": -30}}]}
    full = editing.render(img, edit)
    x0, y0, rw, rh = 0.25, 0.30, 0.30, 0.30
    px0, py0 = int(round(x0 * w)), int(round(y0 * h))
    pw, ph = int(round(rw * w)), int(round(rh * h))
    crop = editing.render(img[py0:py0 + ph, px0:px0 + pw], edit, roi=(x0, y0, rw, rh))
    diff = np.abs(crop.astype(int) - full[py0:py0 + ph, px0:px0 + pw].astype(int))
    assert diff.max() <= 2, f"roi render drifted from the full frame: max {diff.max()}"


def test_roi_brush_strokes_land_in_the_window() -> None:
    img = _detailed(400, 600)
    mask = {"type": "brush", "feather": 0, "adj": {"exposure": 1.5},
            "strokes": [{"radius": 0.05, "points": [[0.3, 0.5], [0.7, 0.5]]}]}
    h, w = img.shape[:2]
    x0, y0, rw, rh = 0.25, 0.35, 0.5, 0.3
    px0, py0 = int(round(x0 * w)), int(round(y0 * h))
    pw, ph = int(round(rw * w)), int(round(rh * h))
    full = editing.render(img, {"masks": [mask]})
    crop = editing.render(img[py0:py0 + ph, px0:px0 + pw], {"masks": [mask]},
                          roi=(x0, y0, rw, rh))
    diff = np.abs(crop.astype(int) - full[py0:py0 + ph, px0:px0 + pw].astype(int))
    assert diff.mean() < 1.0, f"brush stroke misplaced in the window: {diff.mean()}"


# ----- additive preset merge ----------------------------------------------

# ----- optics (lens and perspective) ---------------------------------------

def _optics_photo() -> np.ndarray:
    rng = np.random.default_rng(11)
    img = rng.integers(40, 220, (60, 90, 3), dtype=np.uint8)
    img[:, ::9] = 250            # vertical lines, so a warp visibly moves them
    return img


def test_optics_normalize_to_none_when_untouched() -> None:
    e = editing.normalize({"lens": {"distortion": 0, "vignette_midpoint": 70},
                           "transform": {"scale": 100, "upright": "off"}})
    assert e["lens"] is None and e["transform"] is None
    assert editing.is_neutral(e) and editing.optics_key(e) == ""
    lensed = {"lens": {"distortion": 30}}
    assert not editing.is_neutral(lensed)
    assert editing.optics_key(lensed) and editing.optics_key(lensed) != editing.optics_key(
        {"lens": {"distortion": 31}})
    # The grade does not move the key: it is what a cache of corrected frames keys on.
    assert editing.optics_key({**lensed, "exposure": 1.0}) == editing.optics_key(lensed)


def test_optics_run_first_and_masks_sit_on_the_corrected_frame() -> None:
    """Rendering with the optics is the same as correcting first and rendering
    the rest, which is what places a mask on the corrected picture."""
    img = _optics_photo()
    edit = {"lens": {"distortion": 40, "vignette_amount": 30},
            "transform": {"vertical": 25, "scale": 115},
            "exposure": 0.4,
            "masks": [{"type": "radial", "cx": 0.3, "cy": 0.4, "rx": 0.2, "ry": 0.3,
                       "adj": {"exposure": -1.0}}]}
    whole = editing.render(img, edit)
    corrected = editing.apply_optics(img, edit)
    assert not np.array_equal(corrected, img)
    rest = editing.render(corrected, edit, optics=False)
    assert np.array_equal(whole, rest)


def test_a_window_needs_the_frame_corrected_first() -> None:
    img = _optics_photo()
    edit = {"lens": {"distortion": 40}, "exposure": 0.3}
    try:
        editing.render(img[:30, :45], edit, roi=(0.0, 0.0, 0.5, 0.5), geometry=False)
    except AssertionError as exc:
        assert "whole frame" in str(exc)
    else:
        raise AssertionError("rendered a lens correction on a window")
    # Correct the whole frame first, then any window of it renders.
    corrected = editing.apply_optics(img, edit)
    win = editing.render(corrected[:30, :45], edit, roi=(0.0, 0.0, 0.5, 0.5),
                         geometry=False, optics=False)
    assert win.shape == (30, 45, 3)


def test_optics_keep_the_frame_size() -> None:
    img = _optics_photo()
    out = editing.apply_optics(img, {"lens": {"distortion": -50, "ca_red_cyan": 40},
                                     "transform": {"horizontal": 30, "aspect": 10}})
    assert out.shape == img.shape and out.dtype == np.uint8


def test_a_preset_brings_its_lens_correction() -> None:
    merged = editing.merge_additive({"exposure": 0.5}, {"lens": {"distortion": 12}})
    assert merged["lens"]["distortion"] == 12 and merged["exposure"] == 0.5


def test_merge_additive_appends_repairs() -> None:
    """Heals and red-eye fixes from a preset are added to the photo's own. The
    red-eye half used to raise a KeyError: it was read as "ops"."""
    spot = {"kind": "spot", "points": [[0.3, 0.3]], "radius": 0.01}
    eye = {"kind": "red", "cx": 0.5, "cy": 0.4, "r": 0.02}
    base = {"healing": {"ops": [spot]},
            "redeye": {"enabled": True, "corrections": [eye]}}
    over = {"healing": {"ops": [{**spot, "points": [[0.6, 0.6]]}]},
            "redeye": {"enabled": True, "corrections": [{**eye, "cx": 0.6}]}}
    out = editing.merge_additive(base, over)
    assert len(out["healing"]["ops"]) == 2
    assert [c["cx"] for c in out["redeye"]["corrections"]] == [0.5, 0.6]
    assert editing.merge_additive({}, {"redeye": base["redeye"]})["redeye"]["enabled"]


def test_merge_keeps_the_base_where_the_overlay_is_neutral() -> None:
    """A mask-only preset must not flatten the grade already on the photo."""
    base = {"exposure": 0.5, "contrast": 20, "temp": -15}
    overlay = {"masks": [{"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.3,
                          "ry": 0.3, "adj": {"blur": 50}}]}
    out = editing.merge_additive(base, overlay)
    assert out["exposure"] == 0.5 and out["contrast"] == 20 and out["temp"] == -15
    assert len(out["masks"]) == 1


def test_merge_lets_the_overlay_win_where_it_is_set() -> None:
    out = editing.merge_additive({"exposure": 0.5, "contrast": 20},
                                 {"contrast": -40, "saturation": 30})
    assert out["exposure"] == 0.5, "untouched by the overlay"
    assert out["contrast"] == -40, "overlay wins"
    assert out["saturation"] == 30


def test_merge_appends_masks_and_caps_them() -> None:
    mask = {"type": "radial", "cx": 0.4, "cy": 0.4, "rx": 0.2, "ry": 0.2,
            "adj": {"exposure": 0.5}}
    line = {"type": "linear", "x1": 0.5, "y1": 0.0, "x2": 0.5, "y2": 0.6,
            "adj": {"blur": 30}}
    out = editing.merge_additive({"masks": [mask]}, {"masks": [line]})
    assert [m["type"] for m in out["masks"]] == ["radial", "linear"], "order preserved"
    flood = editing.merge_additive({"masks": [mask] * 10}, {"masks": [line] * 20})
    assert len(flood["masks"]) == editing.MASK_MAX


def test_merge_carries_curve_and_watermark_only_when_present() -> None:
    curved = [[0.0, 0.0], [0.5, 0.7], [1.0, 1.0]]
    base = {"curve": curved, "watermark": {"enabled": True, "name": "Base"}}
    # An overlay with neither keeps the base's.
    kept = editing.merge_additive(base, {"saturation": 10})
    assert kept["curve"] == curved
    assert kept["watermark"]["name"] == "Base"
    # An overlay carrying them wins.
    other = [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]]
    won = editing.merge_additive(base, {"curve": other,
                                        "watermark": {"enabled": True, "name": "Over"}})
    assert won["curve"] == other
    assert won["watermark"]["name"] == "Over"


def test_merge_of_nothing_is_the_base() -> None:
    base = {"exposure": 0.3, "masks": [{"type": "radial", "cx": 0.5, "cy": 0.5,
                                        "rx": 0.2, "ry": 0.2, "adj": {"blur": 20}}]}
    assert editing.merge_additive(base, None) == editing.normalize(base)
    assert editing.merge_additive(None, base) == editing.normalize(base)
    assert editing.is_neutral(editing.merge_additive(None, None))


def test_stacking_local_presets_composes() -> None:
    """The point of the feature: two local presets applied in turn give both."""
    from picture_classifier import presets
    by_id = {p["id"]: p["edit"] for p in presets.list_builtins()}
    edit = editing.merge_additive(None, by_id["car-bokeh"])
    edit = editing.merge_additive(edit, by_id["car-plate"])
    assert len(edit["masks"]) == 2
    kinds = {m["name"] for m in edit["masks"]}
    assert kinds == {"Bokeh", "Plate"}, "stacked presets keep distinct names"
    img = _detailed(200, 300)
    out = editing.render(img, edit)
    assert out.shape == img.shape and not np.array_equal(out, img)


# ----- texture, dehaze and denoise ----------------------------------------

def test_new_sliders_are_neutral_at_zero() -> None:
    img = _sample()
    for key in ("texture", "dehaze", "denoise"):
        assert editing.is_neutral({key: 0}), f"{key}=0 should be neutral"
        assert not editing.is_neutral({key: 30}), f"{key}=30 should not be neutral"
        assert np.array_equal(editing.render(img, {key: 0}), img)


def test_texture_direction() -> None:
    # A frame with fine detail: high-frequency amplitude is what texture moves.
    rng = np.random.default_rng(1)
    base = np.full((128, 128, 3), 128, dtype=np.uint8)
    noise = rng.integers(-30, 31, size=(128, 128, 1))
    img = np.clip(base.astype(int) + noise, 0, 255).astype(np.uint8)

    def detail(a: np.ndarray) -> float:
        y = editing._luma(a.astype(np.float32) / 255.0)
        return float((y - editing._lowfreq(y, 128.0, 1024, 2.0)).std())

    assert detail(editing.render(img, {"texture": 100})) > detail(img) * 1.2
    assert detail(editing.render(img, {"texture": -100})) < detail(img) * 0.6


def test_texture_keeps_regional_contrast() -> None:
    """Texture must not act like clarity: a soft edge is low-frequency, so
    smoothing texture all the way must leave it standing."""
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    img[:, :64] = 70
    img[:, 64:] = 190
    out = editing.render(img, {"texture": -100})
    step = float(out[:, 80:].mean()) - float(out[:, :48].mean())
    assert step > 100, f"the edge was flattened: step {step}"


def test_dehaze_direction() -> None:
    """Positive dehaze deepens the blacks a veil has lifted; negative lifts them."""
    rng = np.random.default_rng(2)
    scene = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    hazy = np.clip(scene.astype(np.float32) * 0.45 + 130.0, 0, 255).astype(np.uint8)

    cleared = editing.render(hazy, {"dehaze": 80})
    assert cleared.mean() < hazy.mean()
    # Contrast is the point, not brightness: the histogram should spread out.
    assert cleared.std() > hazy.std()

    more_haze = editing.render(hazy, {"dehaze": -80})
    assert more_haze.std() < hazy.std()


def _hazy_scene(edge: int) -> np.ndarray:
    """A photograph-like frame: smooth gradients and a few solid shapes, under a
    haze veil. Scale invariance is a claim about photographs — per-pixel white
    noise survives no resampling at all, so it would test the resampler, not the
    operator."""
    yy, xx = np.mgrid[0:edge, 0:edge].astype(np.float32) / edge
    scene = np.stack([0.15 + 0.7 * yy, 0.2 + 0.6 * xx, 0.5 - 0.3 * yy], axis=2)
    q = edge // 4
    scene[q:2 * q, q:3 * q] = 0.05          # something genuinely dark
    scene[2 * q:3 * q, 2 * q:3 * q] = 0.9   # and something bright
    hazy = scene * 0.5 + 0.45               # I = J*t + A*(1-t), t = 0.5, A = 0.9
    return np.rint(np.clip(hazy, 0, 1) * 255.0).astype(np.uint8)


def test_dehaze_is_resolution_independent() -> None:
    """The same photo at two render sizes must get the same look. Two things buy
    that: the atmospheric light is fixed rather than estimated per frame, and the
    dark channel is read off a low-passed copy at a frame-relative radius."""
    big = _hazy_scene(400)
    small = cv2.resize(big, (100, 100), interpolation=cv2.INTER_AREA)

    edit = {"dehaze": 70}
    from_big = cv2.resize(editing.render(big, edit), (100, 100),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
    from_small = editing.render(small, edit).astype(np.float32)
    assert abs(from_big.mean() - from_small.mean()) < 4.0
    assert np.abs(from_big - from_small).mean() < 6.0


def test_denoise_removes_noise_and_keeps_the_edge() -> None:
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    img[:, :80] = 80
    img[:, 80:] = 175
    rng = np.random.default_rng(4)
    noisy = np.clip(img.astype(np.float32)
                    + rng.normal(0.0, 12.0, img.shape), 0, 255).astype(np.uint8)

    out = editing.render(noisy, {"denoise": 100}).astype(np.float32)

    def flat_noise(a: np.ndarray) -> float:
        # Standard deviation inside one flat half, away from the edge.
        return float(a[20:140, 10:60].std())

    assert flat_noise(out) < flat_noise(noisy.astype(np.float32)) * 0.7
    step = float(out[:, 100:150].mean()) - float(out[:, 10:60].mean())
    assert step > 85, f"denoise ate the edge: step {step}"


def test_new_sliders_work_on_a_mask() -> None:
    for key in ("texture", "dehaze", "denoise"):
        assert key in editing.LOCAL_KEYS
    rng = np.random.default_rng(5)
    img = rng.integers(60, 200, size=(96, 96, 3), dtype=np.uint8)
    mask = editing._default_mask("radial")
    mask["adj"]["dehaze"] = 90
    out = editing.render(img, {"masks": [mask]})
    assert not np.array_equal(out, img)
    # A radial mask centred in the frame leaves the corners alone.
    assert np.array_equal(out[:6, :6], img[:6, :6])


def test_padding_covers_the_new_stages() -> None:
    for key in ("texture", "dehaze", "denoise"):
        assert editing.effect_padding({key: 100}, 4000.0) > \
            editing.effect_padding({}, 4000.0), f"{key} asked for no padding"


# ----- colour mixer -------------------------------------------------------

def _patch(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.full((32, 32, 3), rgb, dtype=np.uint8)


def test_hsl_neutral_forms() -> None:
    img = _sample()
    for raw in (None, {}, {"red": {}}, {"red": {"hue": 0, "sat": 0, "lum": 0}},
                {"nonsense": {"sat": 50}}):
        assert editing.normalize_hsl(raw) is None, f"should normalize away: {raw}"
        assert editing.is_neutral({"hsl": raw}), f"should be neutral: {raw}"
        assert np.array_equal(editing.render(img, {"hsl": raw}), img)
    assert not editing.is_neutral({"hsl": {"blue": {"sat": -40}}})


def test_hsl_stores_sparsely() -> None:
    """Only values off neutral survive, so the edit hash stays short and a
    switched-on mixer nobody touched is still a neutral edit."""
    got = editing.normalize_hsl({"red": {"hue": 0, "sat": 30, "lum": 0},
                                 "green": {"sat": 0}, "blue": {"lum": 200}})
    assert got == {"red": {"sat": 30}, "blue": {"lum": 100}}


def test_hsl_saturation_hits_only_its_band() -> None:
    red, blue = _patch((200, 40, 40)), _patch((40, 60, 200))
    edit = {"hsl": {"red": {"sat": -100}}}

    def spread(a: np.ndarray) -> float:
        f = a.astype(np.float32)
        return float((f.max(axis=2) - f.min(axis=2)).mean())

    assert spread(editing.render(red, edit)) < spread(red) * 0.35
    # The blue patch is four bands away and must come back untouched.
    assert np.array_equal(editing.render(blue, edit), blue)


def test_hsl_luminance_direction() -> None:
    green = _patch((40, 170, 60))
    up = editing.render(green, {"hsl": {"green": {"lum": 100}}})
    down = editing.render(green, {"hsl": {"green": {"lum": -100}}})
    assert up.mean() > green.mean() + 10
    assert down.mean() < green.mean() - 10
    # Neither end may clip a band flat: the hue has to survive the move.
    for out in (up, down):
        f = out.astype(np.float32)
        assert (f.max(axis=2) - f.min(axis=2)).mean() > 4.0


def test_hsl_hue_rotation_direction() -> None:
    orange = _patch((220, 120, 40))
    turned = editing.render(orange, {"hsl": {"orange": {"hue": 100}}})
    hls = cv2.cvtColor(orange.astype(np.float32) / 255.0, cv2.COLOR_RGB2HLS)
    hls2 = cv2.cvtColor(turned.astype(np.float32) / 255.0, cv2.COLOR_RGB2HLS)
    moved = float(hls2[..., 0].mean() - hls[..., 0].mean())
    assert 15.0 < moved < 40.0, f"hue moved {moved} degrees, expected about 30"


def test_hsl_leaves_greys_alone() -> None:
    """A near-grey pixel's hue is numerical noise, so the mixer must let go of
    it — otherwise a flat wall speckles into two different corrections."""
    grey = _patch((128, 128, 128))
    for band in editing.HSL_BANDS:
        out = editing.render(grey, {"hsl": {band: {"sat": 100, "lum": 80, "hue": 100}}})
        assert np.abs(out.astype(np.int16) - 128).max() <= 2, f"{band} moved a grey"


def test_hsl_bands_blend_without_a_seam() -> None:
    """Adjacent bands are interpolated, so a hue between two centres gets a mix
    of both rather than a step. Walking the hue circle must stay smooth."""
    hue_lut, sat_lut, _ = editing._hsl_luts({"orange": {"sat": 100}})
    assert sat_lut[30] > 99.0, "a band centre must keep its own full value"
    assert sat_lut[60] < 1.0, "the neighbouring centre must be unaffected"
    ring = np.concatenate([sat_lut, sat_lut[:1]])
    assert np.abs(np.diff(ring)).max() < 6.0, "found a step between bands"
    # Every parameter is zero when nothing is set, all the way round.
    assert np.abs(hue_lut).max() < 1e-6


def test_hsl_partition_of_unity() -> None:
    """Setting every band to the same value must move every hue by that value —
    the check that the weights sum to one everywhere, with no gap or overlap."""
    same = {b: {"sat": 60} for b in editing.HSL_BANDS}
    _, sat_lut, _ = editing._hsl_luts(editing.normalize_hsl(same))
    assert np.allclose(sat_lut, 60.0, atol=1e-4)


def test_hsl_merges_band_by_band() -> None:
    base = {"hsl": {"red": {"sat": 40}}}
    over = {"hsl": {"blue": {"lum": -30}}}
    merged = editing.merge_additive(base, over)
    assert merged["hsl"] == {"red": {"sat": 40}, "blue": {"lum": -30}}
    # An overlay touching the same band wins on that key only.
    again = editing.merge_additive(base, {"hsl": {"red": {"hue": 10}}})
    assert again["hsl"] == {"red": {"sat": 40, "hue": 10}}


def test_hsl_is_pointwise() -> None:
    """The mixer maps a colour to a colour with no reference to a pixel's
    neighbours. That is the strongest form of the guarantee the rest of the
    pipeline works for: it needs no padding, and the same pixel value comes out
    the same in a thumbnail, a 1:1 window and the export.

    Note this is *not* testable by downscaling one render and comparing — a
    nonlinear pointwise map never commutes with averaging. Nearest-neighbour
    replication is, because it produces the same values.
    """
    rng = np.random.default_rng(6)
    small = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(small, 5, axis=0), 5, axis=1)
    edit = {"hsl": {"aqua": {"sat": 70, "lum": -40}}}
    assert np.array_equal(editing.render(small, edit), editing.render(big, edit)[::5, ::5])
    assert editing.effect_padding(edit, 4000.0) == editing.effect_padding({}, 4000.0)


# ----- per-channel tone curves --------------------------------------------

_LIFT_BLUE = [[0.0, 0.12], [0.5, 0.55], [1.0, 1.0]]


def test_channel_curves_neutral_and_not() -> None:
    img = _sample()
    for key in editing._CHANNEL_CURVES:
        assert editing.is_neutral({key: [[0, 0], [1, 1]]})
        assert editing.is_neutral({key: [[0, 0], [0.4, 0.4], [1, 1]]})
        assert np.array_equal(editing.render(img, {key: [[0, 0], [1, 1]]}), img)
        assert not editing.is_neutral({key: _LIFT_BLUE})


def test_channel_curve_touches_only_its_channel() -> None:
    grey = _patch((120, 120, 120))
    out = editing.render(grey, {"curve_b": _LIFT_BLUE})
    assert np.array_equal(out[..., 0], grey[..., 0]), "the red channel moved"
    assert np.array_equal(out[..., 1], grey[..., 1]), "the green channel moved"
    # _LIFT_BLUE maps 0.47 to about 0.53, so mid-grey lifts by roughly 14 levels.
    assert out[..., 2].mean() > grey[..., 2].mean() + 10


def test_channel_curves_compose_with_the_master() -> None:
    """The master says how bright a tone is and the channel curves say what
    colour it takes there, so the two must compose rather than one winning."""
    grey = _patch((110, 110, 110))
    darken = [[0.0, 0.0], [0.5, 0.32], [1.0, 1.0]]
    both = editing.render(grey, {"curve": darken, "curve_b": _LIFT_BLUE})
    master_only = editing.render(grey, {"curve": darken})
    chan_only = editing.render(grey, {"curve_b": _LIFT_BLUE})
    # Darker than the channel curve alone, bluer than the master alone.
    assert both[..., 0].mean() < chan_only[..., 0].mean() - 10
    assert both[..., 2].mean() > master_only[..., 2].mean() + 10


def test_channel_curves_are_free_at_render_time() -> None:
    """They fold into the same per-channel lookup white balance and tone use, so
    the render path must be a table read, not an extra pass."""
    e = editing.normalize({"curve_r": _LIFT_BLUE})
    lut = editing._wb_tone_lut(e)
    assert lut is not None and lut.shape == (256, 1, 3)
    # Only the red column moved; the other two are still the identity ramp.
    ident = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    assert np.abs(lut[:, 0, 0] - ident).max() > 0.05
    assert np.allclose(lut[:, 0, 1], ident, atol=1e-6)
    assert np.allclose(lut[:, 0, 2], ident, atol=1e-6)


def test_channel_curve_paths_agree() -> None:
    """The 8-bit lookup path and the float path must produce the same picture —
    otherwise a mask grading an already-graded crop would silently drop them."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    e = editing.normalize({"curve": [[0, 0], [0.5, 0.6], [1, 1]],
                           "curve_r": _LIFT_BLUE,
                           "curve_b": [[0, 0], [0.5, 0.4], [1, 1]]})
    from_u8 = editing._grade(img, e)
    from_float = editing._grade(img.astype(np.float32) / 255.0, e)
    assert np.abs(from_u8 - from_float).max() < 0.01


def test_channel_curves_merge_and_hash() -> None:
    merged = editing.merge_additive({"curve_r": _LIFT_BLUE}, {"curve_b": _LIFT_BLUE})
    assert not editing._curve_is_identity(merged["curve_r"])
    assert not editing._curve_is_identity(merged["curve_b"])
    assert editing.edit_hash({"curve_g": _LIFT_BLUE}) != ""
    assert editing.edit_hash({"curve_g": [[0, 0], [1, 1]]}) == ""


# ----- automatic masks ----------------------------------------------------

def _auto_mask(group: str = "subject", **kw) -> dict:
    m = editing._default_mask("auto")
    m["group"] = group
    m["adj"]["exposure"] = 1.0
    m.update(kw)
    return m


def _half_field(h: int = 64, w: int = 64) -> np.ndarray:
    """A field selecting the left half of the frame, hard-edged so the boundary
    is easy to measure."""
    f = np.zeros((h, w), dtype=np.float32)
    f[:, : w // 2] = 1.0
    return f


def test_auto_mask_defaults_to_no_feather() -> None:
    """The model's own edge is already soft where it should be — half-covered
    hair comes out near 0.5 — so blurring it by default would discard that."""
    m = editing._default_mask("auto")
    assert m["type"] == "auto" and m["group"] == "subject" and m["feather"] == 0


def test_auto_mask_normalization() -> None:
    for group in editing.segment_mod.CLASS_GROUPS:
        got = editing.normalize_mask({"type": "auto", "group": group})
        assert got is not None and got["group"] == group
    # A group the model cannot produce selects nothing, so it is not a mask.
    assert editing.normalize_mask({"type": "auto", "group": "hat"}) is None
    assert editing.normalize_mask({"type": "auto"}) is None


def test_auto_mask_without_its_field_raises() -> None:
    """Fail loudly rather than grade nothing: a silently skipped mask looks like
    the edit was saved wrong."""
    img = _sample()
    with pytest.raises(AssertionError, match="automatic mask needs its field"):
        editing.render(img, {"masks": [_auto_mask()]})


def test_auto_mask_grades_only_inside_the_field() -> None:
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    out = editing.render(img, {"masks": [_auto_mask()]},
                         auto={"subject": _half_field()})
    left, right = out[:, :24].mean(), out[:, 40:].mean()
    assert left > 130, f"the selected half was not graded: {left}"
    assert abs(float(right) - 100.0) < 1.0, f"the other half moved: {right}"


def test_auto_mask_inverts() -> None:
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    out = editing.render(img, {"masks": [_auto_mask(invert=True)]},
                         auto={"subject": _half_field()})
    assert out[:, 40:].mean() > 130 and abs(float(out[:, :24].mean()) - 100.0) < 1.0


def test_auto_mask_amount_scales() -> None:
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    field = {"subject": _half_field()}
    full = editing.render(img, {"masks": [_auto_mask()]}, auto=field)
    half = editing.render(img, {"masks": [_auto_mask(amount=50)]}, auto=field)
    assert 100 < half[:, :24].mean() < full[:, :24].mean()


def test_auto_mask_field_is_cropped_to_the_roi() -> None:
    """A 1:1 window gets the matching piece of the whole-frame field, which is
    what makes an automatic mask land on the same pixels at every zoom."""
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    # The right half of the frame: entirely outside the field's left-half
    # selection, so nothing in this window may be graded.
    right = editing.render(img, {"masks": [_auto_mask()]}, roi=(0.5, 0.0, 0.5, 1.0),
                           geometry=False, auto={"subject": _half_field()})
    assert abs(float(right.mean()) - 100.0) < 1.0
    left = editing.render(img, {"masks": [_auto_mask()]}, roi=(0.0, 0.0, 0.5, 1.0),
                          geometry=False, auto={"subject": _half_field()})
    assert left.mean() > 130


def test_auto_mask_field_resolution_does_not_matter() -> None:
    """The field is the model's small output, so it will nearly always be a
    different size from the frame being rendered. That must not shift it."""
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    coarse = editing.render(img, {"masks": [_auto_mask()]},
                            auto={"subject": _half_field(16, 16)})
    fine = editing.render(img, {"masks": [_auto_mask()]},
                          auto={"subject": _half_field(256, 256)})
    assert abs(float(coarse[:, :40].mean()) - float(fine[:, :40].mean())) < 3.0
    assert abs(float(coarse[:, 88:].mean()) - 100.0) < 1.0


def test_auto_mask_feather_softens_the_edge() -> None:
    img = np.full((128, 128, 3), 100, dtype=np.uint8)
    field = {"subject": _half_field(128, 128)}

    def edge_width(out: np.ndarray) -> int:
        col = out[64, :, 0].astype(np.int16)
        mid = (int(col.max()) + int(col.min())) // 2
        band = np.abs(col - mid) < (int(col.max()) - int(col.min())) * 0.35
        return int(band.sum())

    hard = editing.render(img, {"masks": [_auto_mask()]}, auto=field)
    soft = editing.render(img, {"masks": [_auto_mask(feather=100)]}, auto=field)
    assert edge_width(soft) > edge_width(hard)


def test_auto_masks_are_not_cached_by_geometry() -> None:
    """The other kinds hash their numbers; an automatic mask is a function of the
    caller's array, which those numbers say nothing about."""
    with pytest.raises(AssertionError, match="not cached"):
        editing._alpha_key(_auto_mask(), 64, 64, editing.FULL_ROI)


def test_auto_mask_neutral_adj_is_inactive() -> None:
    m = editing._default_mask("auto")
    assert not editing.mask_is_active(m)
    assert editing.is_neutral({"masks": [m]})


# ----- colour grading (wiring; the operator is tested in test_grading.py) ---

def test_grading_neutral_forms() -> None:
    img = _sample()
    for raw in (None, {}, {"shadows": {}}, {"shadows": {"sat": 0, "lum": 0}},
                {"shadows": {"hue": 200}}):   # a hue with no strength does nothing
        assert editing.is_neutral({"grading": raw}), f"should be neutral: {raw}"
        assert np.array_equal(editing.render(img, {"grading": raw}), img)
    assert not editing.is_neutral({"grading": {"shadows": {"hue": 200, "sat": 30}}})


def _three_bands() -> np.ndarray:
    """Dark, mid and bright in one frame. A flat mid-tone patch is 100% midtones
    by design, so it cannot exercise the shadow or highlight wheels at all."""
    img = np.zeros((30, 96, 3), dtype=np.uint8)
    img[:, :32] = (40, 36, 32)
    img[:, 32:64] = (120, 110, 100)
    img[:, 64:] = (215, 205, 195)
    return img


def test_grading_runs_after_the_mixer() -> None:
    """The mixer says what a colour is; the wheels then push a whole tonal
    region regardless. Both present must give something different from either."""
    img = _three_bands()
    hsl = {"orange": {"sat": 60}}
    grade = {"shadows": {"hue": 220, "sat": 40}, "highlights": {"hue": 40, "sat": 40}}
    both = editing.render(img, {"hsl": hsl, "grading": grade}).astype(np.int16)
    only_hsl = editing.render(img, {"hsl": hsl}).astype(np.int16)
    only_grade = editing.render(img, {"grading": grade}).astype(np.int16)
    assert np.abs(both - only_hsl).max() > 2
    assert np.abs(both - only_grade).max() > 2


def test_grading_merges_zone_by_zone() -> None:
    merged = editing.merge_additive(
        {"grading": {"shadows": {"hue": 200, "sat": 30}}},
        {"grading": {"highlights": {"hue": 40, "sat": 20}}})
    assert merged["grading"]["shadows"]["sat"] == 30
    assert merged["grading"]["highlights"]["sat"] == 20


def test_grading_hashes_and_round_trips() -> None:
    g = {"midtones": {"hue": 90, "sat": 25, "lum": -10}, "blending": 70}
    assert editing.edit_hash({"grading": g}) != ""
    assert editing.edit_hash({"grading": {"midtones": {"hue": 90}}}) == ""
    assert editing.normalize({"grading": g})["grading"]["midtones"]["lum"] == -10


def test_grading_is_pointwise() -> None:
    """No neighbourhood, so it needs no padding and cannot drift with zoom."""
    rng = np.random.default_rng(8)
    small = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
    big = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)
    edit = {"grading": {"shadows": {"hue": 210, "sat": 40, "lum": 15}}}
    assert np.array_equal(editing.render(small, edit), editing.render(big, edit)[::4, ::4])
    assert editing.effect_padding(edit, 4000.0) == editing.effect_padding({}, 4000.0)


# ----- sharpening (wiring; the operator is tested in test_sharpening.py) ----

def test_sharpen_modifiers_are_not_changes() -> None:
    """Radius, detail and masking say how sharpening looks, not whether any was
    asked for. On their own they must leave an edit neutral — otherwise every
    photo would open looking edited."""
    img = _sample()
    for key in ("sharpen_radius", "sharpen_detail", "sharpen_masking"):
        assert editing.is_neutral({key: 90}), key
        assert np.array_equal(editing.render(img, {key: 90}), img)
    assert not editing.is_neutral({"sharpen": 20})


def test_a_legacy_sharpen_edit_still_loads() -> None:
    """`sharpen` predates the other three, so an edit saved with only that key
    must come back with its number where it was and the rest at their neutrals."""
    e = editing.normalize({"sharpen": 60})
    assert e["sharpen"] == 60
    assert e["sharpen_radius"] == editing.DEFAULT_EDIT["sharpen_radius"]
    assert e["sharpen_detail"] == editing.DEFAULT_EDIT["sharpen_detail"]
    assert e["sharpen_masking"] == 0


def test_a_default_mask_is_inactive_despite_a_non_zero_neutral() -> None:
    """sharpen_radius is neutral at its midpoint, so a mask's neutral slider set
    is not all zeroes — and `mask_is_active` must measure against the neutrals
    rather than against zero."""
    for kind in ("radial", "linear"):
        m = editing._default_mask(kind)
        assert m["adj"]["sharpen_radius"] == editing.DEFAULT_EDIT["sharpen_radius"]
        assert not editing.mask_is_active(m), kind


def test_sharpening_asks_for_padding_and_a_window_still_matches() -> None:
    """Sharpening reads a neighbourhood, so a window render has to be handed
    enough surrounding pixels to agree with the whole render. This is the
    invariant the 1:1 view rests on."""
    rng = np.random.default_rng(11)
    img = rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)
    edit = {"sharpen": 90, "sharpen_radius": 70, "sharpen_detail": 80,
            "sharpen_masking": 30, "exposure": 0.2}
    assert editing.effect_padding(edit, 400.0) > editing.effect_padding({}, 400.0)

    full = editing.render(img, edit)
    win = (80, 60, 200, 140)
    pad = int(round(editing.effect_padding(edit, 400.0))) + 2
    box = editing.geometry_source_box(400, 300, edit, win, pad)
    bx, by, bw, bh = box
    graded = editing.render(img[by:by + bh, bx:bx + bw], edit,
                            roi=(bx / 400, by / 300, bw / 400, bh / 300),
                            with_watermark=False, geometry=False)
    got = editing.geometry_window(graded, box, 400, 300, edit, win)
    want = full[win[1]:win[1] + win[3], win[0]:win[0] + win[2]]
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.max() <= 1, f"the window disagrees with the whole render by {diff.max()}"


# ----- range masks and range refinements ----------------------------------

def _bands_frame() -> np.ndarray:
    """Dark left, mid middle, bright right — a frame whose tones are separable
    so a luminance range's effect can be read off by column."""
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    img[:, :32] = 30
    img[:, 32:64] = 128
    img[:, 64:] = 225
    return img


def test_range_mask_needs_a_range() -> None:
    """Selecting every tone is not a mask, so it must not survive normalization
    or make an edit look non-neutral — the same rule normalize_crop follows."""
    assert editing.normalize_mask({"type": "range"}) is None
    assert editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 100}}) is None
    got = editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 40}})
    assert got is not None and got["range_luma"] is not None


def test_every_mask_kind_carries_the_refinement_fields() -> None:
    for kind in editing.MASK_TYPES:
        m = editing._default_mask(kind)
        assert m["range_luma"] is None and m["range_color"] is None, kind


def test_range_mask_selects_its_tones() -> None:
    img = _bands_frame()
    m = editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 35},
                                "adj": {"exposure": 1.0}})
    out = editing.render(img, {"masks": [m]}, src=img)
    assert out[:, :24].mean() > 45, "the dark band was not selected"
    assert abs(float(out[:, 72:].mean()) - 225.0) < 2.0, "the bright band moved"


def test_a_refinement_narrows_a_drawn_shape() -> None:
    """"This shape, but only these tones" — the reason a range is a refinement
    on every kind rather than a mask type of its own."""
    img = _bands_frame()
    shape = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.9, "ry": 0.9,
             "feather": 0, "adj": {"exposure": 1.0}}
    plain = editing.render(img, {"masks": [editing.normalize_mask(shape)]}, src=img)
    refined = editing.render(
        img, {"masks": [editing.normalize_mask({**shape,
                                                "range_luma": {"lo": 0, "hi": 35}})]},
        src=img)
    # The radial covers all three bands; the refinement should keep only the dark.
    assert plain[:, 72:].mean() > 225.0
    assert abs(float(refined[:, 72:].mean()) - 225.0) < 2.0
    assert refined[:, :24].mean() > 45


def test_a_refinement_without_the_frame_raises() -> None:
    img = _bands_frame()
    m = editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 35},
                                "adj": {"exposure": 1.0}})
    with pytest.raises(AssertionError, match="whole ungraded frame"):
        editing.render(img, {"masks": [m]})


def test_pixel_derived_masks_are_not_cached_by_geometry() -> None:
    for kind, extra in (("auto", {"group": "subject"}),
                        ("range", {"range_luma": {"lo": 0, "hi": 40}})):
        m = editing.normalize_mask({"type": kind, **extra})
        with pytest.raises(AssertionError, match="not cached"):
            editing._alpha_key(m, 64, 64, editing.FULL_ROI)


def test_a_refinement_selects_the_same_tones_in_a_window() -> None:
    """The selector is measured on the whole frame, so a 1:1 window must select
    the same tones the fit preview did. Measured on the window it would not: the
    window's own histogram is not the frame's."""
    img = _bands_frame()
    m = editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 35},
                                "adj": {"exposure": 1.0}})
    edit = {"masks": [m]}
    full = editing.render(img, edit, src=img)
    # A window over the bright band only. Its own tones span nothing near L* 35,
    # so nothing in it may be selected.
    win = img[:, 64:]
    got = editing.render(win, edit, roi=(64 / 96, 0.0, 32 / 96, 1.0),
                         geometry=False, src=img)
    want = full[:, 64:]
    assert np.abs(got.astype(int) - want.astype(int)).max() <= 1


def test_a_refinement_on_an_automatic_mask() -> None:
    """The combination that motivated making this a refinement: the subject, but
    only part of its tonal range."""
    img = _bands_frame()
    m = editing.normalize_mask({"type": "auto", "group": "subject",
                                "range_luma": {"lo": 0, "hi": 35},
                                "adj": {"exposure": 1.0}})
    assert m is not None
    field = np.ones(img.shape[:2], dtype=np.float32)   # "all subject"
    out = editing.render(img, {"masks": [m]}, auto={"subject": field}, src=img)
    assert out[:, :24].mean() > 45, "the dark part of the subject was not graded"
    assert abs(float(out[:, 72:].mean()) - 225.0) < 2.0, "the bright part moved"


def test_a_refinement_after_another_mask_selects_from_the_frame() -> None:
    """A range is measured on the whole ungraded frame whatever masks come
    before it. The mask loop once reused that frame's name for the crop it had
    just graded, so a refined mask after any active mask selected its tones out
    of the previous mask's crop: in the preview, the thumbnails and the export.
    """
    img = _bands_frame()
    # A small shape over the bright band, well away from the dark one.
    first = editing.normalize_mask({"type": "radial", "cx": 0.9, "cy": 0.2, "rx": 0.06,
                                    "ry": 0.1, "feather": 0, "adj": {"exposure": 0.5}})
    ranged = editing.normalize_mask({"type": "range", "range_luma": {"lo": 0, "hi": 35},
                                     "adj": {"exposure": 1.0}})
    alone = editing.render(img, {"masks": [ranged]}, src=img)
    both = editing.render(img, {"masks": [first, ranged]}, src=img)
    # Outside the first mask's box only the range mask acts, so the two agree.
    assert np.array_equal(both[:, :64], alone[:, :64])
    assert both[:, :24].mean() > 45, "the dark band was not selected"


# ----- repairs: healing and red-eye (operators tested in their own suites) --

def _spot_op(cx: float = 0.5, cy: float = 0.5, r: float = 0.06) -> dict:
    return {"kind": "spot", "points": [[cx, cy]], "radius": r,
            "feather": 40, "opacity": 100}


def _red_pupil(size: int = 96) -> np.ndarray:
    """A grey frame with one saturated red disc in it."""
    img = np.full((size, size, 3), 110, dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (xx - size // 2) ** 2 + (yy - size // 2) ** 2 <= (size // 12) ** 2
    img[disc] = (200, 30, 30)
    return img


def test_repairs_normalize_away_when_empty() -> None:
    img = _sample()
    for key in ("healing", "redeye"):
        for raw in (None, {}, {"ops": []}):
            assert editing.is_neutral({key: raw}), f"{key}={raw}"
            assert np.array_equal(editing.render(img, {key: raw}), img)


def test_healing_changes_only_its_region() -> None:
    rng = np.random.default_rng(12)
    img = rng.integers(90, 140, (96, 96, 3), dtype=np.uint8)
    img[44:52, 44:52] = 250                      # a bright defect in the middle
    out = editing.render(img, {"healing": {"ops": [_spot_op()]}})
    assert out[44:52, 44:52].mean() < 200, "the defect survived"
    # A corner well outside the spot must be untouched.
    assert np.array_equal(out[:8, :8], img[:8, :8])


def test_a_repair_runs_before_the_grade() -> None:
    """Order is the point of this stage. Healing first then grading is not the
    same picture as grading first then healing, and only the first is right:
    the second diffuses from pixels a curve has already crushed."""
    rng = np.random.default_rng(13)
    img = rng.integers(90, 140, (96, 96, 3), dtype=np.uint8)
    img[44:52, 44:52] = 250
    heal = {"ops": [_spot_op()]}

    both = editing.render(img, {"healing": heal, "contrast": 60})
    # Grading the already-healed frame is what render does; doing it the other
    # way round has to differ somewhere, or the ordering claim is empty.
    healed_then_graded = editing.render(editing.render(img, {"healing": heal}),
                                        {"contrast": 60})
    graded_then_healed = editing.render(editing.render(img, {"contrast": 60}),
                                        {"healing": heal})
    assert np.abs(both.astype(int) - healed_then_graded.astype(int)).max() <= 2
    assert np.abs(both.astype(int) - graded_then_healed.astype(int)).max() > 2


def test_redeye_neutralizes_a_red_disc() -> None:
    img = _red_pupil()
    # `enabled` is an explicit toggle, as the watermark's is: detection proposes
    # corrections, and being able to switch the whole set off is the point.
    edit = {"redeye": {"enabled": True,
                       "corrections": [{"kind": "red", "cx": 0.5, "cy": 0.5,
                                        "r": 0.12, "amount": 100, "darken": 60}]}}
    out = editing.render(img, edit)
    c = out[46:50, 46:50].astype(np.float32)
    spread = float(c.max(axis=2).mean() - c.min(axis=2).mean())
    assert spread < 30, f"the pupil is still coloured: channel spread {spread}"
    assert np.array_equal(out[:6, :6], img[:6, :6]), "outside the disc moved"


def test_redeye_needs_its_toggle() -> None:
    """A set of corrections with the toggle off changes nothing. Worth pinning:
    an edit that silently did nothing would read as a broken feature."""
    c = [{"kind": "red", "cx": 0.5, "cy": 0.5, "r": 0.12, "amount": 100}]
    assert editing.is_neutral({"redeye": {"corrections": c}})
    assert not editing.is_neutral({"redeye": {"enabled": True, "corrections": c}})


def test_repairs_ask_for_padding_only_when_active() -> None:
    base = editing.effect_padding({}, 4000.0)
    assert editing.effect_padding({"healing": {"ops": [_spot_op()]}}, 4000.0) > base
    # Red-eye reads nothing outside its own disc, so it asks for nothing.
    red = {"redeye": {"enabled": True,
                      "corrections": [{"kind": "red", "cx": .5, "cy": .5,
                                       "r": .1, "amount": 100}]}}
    assert not editing.is_neutral(red), "the fixture has to be active to mean anything"
    assert editing.effect_padding(red, 4000.0) == base


def test_repairs_append_when_presets_stack() -> None:
    """Sensor dust lands in the same place on every frame a body shoots, so a
    preset carrying heals must add to a photo's own rather than replace them."""
    a = {"healing": {"ops": [_spot_op(0.2, 0.2)]}}
    b = {"healing": {"ops": [_spot_op(0.8, 0.8)]}}
    merged = editing.merge_additive(a, b)
    assert len(merged["healing"]["ops"]) == 2


def test_repair_hashes_into_the_edit() -> None:
    e = {"healing": {"ops": [_spot_op()]}}
    assert editing.edit_hash(e) != ""
    assert editing.edit_hash({"healing": {"ops": []}}) == ""


# Last in the file, and it has to stay last: it collects the test functions out
# of globals(), so anything defined below it would not exist yet and would be
# silently skipped. It sat mid-file for a while and ran 37 of 97 while printing
# a pass. `assert_collected` is the guard that makes that impossible to repeat.
# ----- the quick paths agree with the formulas they replaced ---------------

def _graded_like(h: int = 301, w: int = 457) -> np.ndarray:
    """Float RGB the way it looks mid-pipeline: mostly in [0,1], a little over
    and under after a push, with exact half-level steps in it."""
    rng = np.random.default_rng(5)
    img = rng.uniform(-0.03, 1.03, (h, w, 3)).astype(np.float32)
    img[0, :, :] = (np.arange(w) % 511)[:, None].astype(np.float32) / 510.0
    return img


def test_luma_is_the_weighted_sum() -> None:
    img = _graded_like()
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    for arr in (img, img[20:200, 33:300]):          # a view, as masks hand in
        assert np.abs(editing._luma(arr) - arr @ weights).max() < 1e-6


def test_colour_is_the_per_pixel_formula() -> None:
    img = _graded_like()
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    for vib, sat in ((20, 5), (-40, 30), (0, -100), (100, 0)):
        y = (img @ weights)[..., None]
        proxy = img.max(axis=2, keepdims=True) - img.min(axis=2, keepdims=True)
        factor = 1.0 + sat / 100.0 + (vib / 100.0) * (1.0 - np.clip(proxy, 0, 1))
        want = y + factor * (img - y)
        got = editing._apply_color(img, vib, sat)
        assert got.shape == img.shape and got.dtype == np.float32
        assert np.abs(got - want).max() < 1e-5, (vib, sat)


def test_to_u8_is_rint_of_the_clipped_value() -> None:
    img = _graded_like()
    want = np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    assert np.array_equal(editing._to_u8(img), want)
    assert np.array_equal(editing._to_u8(img[5:50, 7:90]), want[5:50, 7:90])


def test_the_frame_before_the_grade_is_kept_between_renders() -> None:
    """A slider drag re-renders one photo over and over; what comes before the
    grade (repairs, skin smoothing, the look) does not move with the slider."""
    real = editing._repair
    calls = []
    editing._repair = lambda *a, **k: calls.append(1) or real(*a, **k)
    try:
        _check_the_pregrade_cache(calls)
    finally:
        editing._repair = real


def _check_the_pregrade_cache(calls: list) -> None:
    img = _sample()
    spot = {"healing": {"ops": [_spot_op()]}}
    first = editing.render(img, {**spot, "exposure": 0.3}, cache_key="p.jpg|corrected")
    second = editing.render(img, {**spot, "exposure": 0.6}, cache_key="p.jpg|corrected")
    assert len(calls) == 1, "the repairs ran again for a slider that cannot change them"
    assert np.array_equal(first, editing.render(img, {**spot, "exposure": 0.3}))
    assert np.array_equal(second, editing.render(img, {**spot, "exposure": 0.6}))
    calls.clear()

    # The same name on a different array (the photo decoded afresh) is a miss.
    editing.render(img.copy(), {**spot, "exposure": 0.6}, cache_key="p.jpg|corrected")
    assert len(calls) == 1
    # So is a change to what the stages themselves read.
    moved = {"healing": {"ops": [_spot_op(cx=0.3)]}}
    got = editing.render(img, {**moved, "exposure": 0.6}, cache_key="p.jpg|corrected")
    assert np.array_equal(got, editing.render(img, {**moved, "exposure": 0.6}))


def _strokes(n: int, pts: int, seed: int = 0, erase_every: int = 3) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        x0, y0 = rng.uniform(0.2, 0.8, 2)
        out.append({"radius": 0.03, "erase": i % erase_every == erase_every - 1,
                    "points": [[float(x0 + 0.2 * np.sin(t / 9 + i)), float(y0 + 0.1 * np.cos(t / 7))]
                               for t in range(pts)]})
    return out


def _clear_brush_caches() -> None:
    for cache in (editing._ALPHA_CACHE, editing._BRUSH_PREFIX, editing._DIGEST_MEMO,
                  editing._STROKES_MEMO):
        cache.clear()


def test_painting_replays_only_the_stroke_being_painted() -> None:
    """While a stroke is painted every draft carries the finished strokes and
    a longer last one. The finished ones are not drawn again, and the result is
    the one a full replay gives."""
    import cv2
    strokes = _strokes(6, 40)

    def draft(end: int) -> np.ndarray:
        # A fresh copy each time, as each request parses the edit afresh.
        live = [dict(s, points=list(s["points"])) for s in strokes[:-1]] \
            + [dict(strokes[-1], points=strokes[-1]["points"][:end])]
        m = editing.normalize_mask({"type": "brush", "feather": 40, "strokes": live})
        return editing._brush_alpha(m, 120, 180, editing.FULL_ROI)

    drawn = []
    real = cv2.polylines
    editing.cv2.polylines = lambda *a, **k: drawn.append(1) or real(*a, **k)
    try:
        _clear_brush_caches()
        draft(10)
        draft(20)
        drawn.clear()
        got = draft(30).copy()
        assert len(drawn) == 1, f"drew {len(drawn)} strokes for the one being painted"
        _clear_brush_caches()
        assert np.array_equal(got, draft(30)), "the cached replay differs from a full one"
    finally:
        editing.cv2.polylines = real


def test_a_draft_and_the_preview_share_the_photos_mask_grid() -> None:
    """Rendered with the photo's size, a draft and the settled preview build
    their masks on the same grid, so the settle reuses what the drafts drew."""
    import cv2
    rng = np.random.default_rng(2)
    preview = rng.integers(0, 256, (1366, 2048, 3), dtype=np.uint8)
    draft = cv2.resize(preview, (1100, 733), interpolation=cv2.INTER_AREA)
    assert editing._work_size(733, 1100) != editing._work_size(1366, 2048)   # what went wrong
    brush = {"type": "brush", "feather": 40, "strokes": _strokes(3, 30), "adj": {"exposure": 0.5}}
    edit = {"masks": [brush]}
    drawn = []
    real = cv2.polylines
    editing.cv2.polylines = lambda *a, **k: drawn.append(1) or real(*a, **k)
    try:
        _clear_brush_caches()
        editing.render(draft, edit, frame_size=(7028, 4688))
        drawn.clear()
        editing.render(preview, edit, frame_size=(7028, 4688))
        assert drawn == [], f"the settle drew {len(drawn)} strokes again"
    finally:
        editing.cv2.polylines = real
    with pytest.raises(AssertionError, match="not the shape"):
        editing.render(preview, edit, frame_size=(4688, 7028))


def test_normalized_strokes_are_the_same_whoever_asks() -> None:
    raw = _strokes(4, 30) + [{"radius": 9, "points": [[5, -3], "x", [0.2]]}, "junk"]
    _clear_brush_caches()
    first = editing._normalize_strokes(raw)
    again = editing._normalize_strokes(raw)
    assert again is first
    assert editing._normalize_strokes(first) is first
    _clear_brush_caches()
    assert editing._normalize_strokes(raw) == first
    assert first[-1]["points"] == [[2.0, -1.0]] and first[-1]["radius"] == 0.5


def _detailed(h: int = 1200, w: int = 1800) -> np.ndarray:
    """Edges, gradients and highlights at several scales, so every spatial stage
    has something to act on at the band edges."""
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 110 + 60 * np.sin(xx / 37.0) * np.cos(yy / 23.0) + 40 * np.sign(np.sin(xx / 5.0 + yy / 9.0))
    img = np.stack([base, np.roll(base, 7, axis=0) * 0.9, np.roll(base, -11, axis=1) * 1.1], axis=2)
    img += rng.normal(0, 6, img.shape)
    img[500:540, 800:900] = 250
    return np.clip(img, 0, 255).astype(np.uint8)


def test_banded_renders_agree_with_the_whole_render() -> None:
    img = _detailed()
    basic = {"exposure": 0.3, "contrast": 20, "shadows": 30, "vibrance": 25, "temp": 10}
    radial = {"type": "radial", "cx": 0.45, "cy": 0.43, "rx": 0.3, "ry": 0.2, "feather": 60,
              "adj": {"exposure": 0.5, "clarity": 30}}
    cases = [
        ("pointwise", basic, 0),
        ("hsl, grading, vignette, radial", {**basic, "hsl": {"blue": {"hue": -10, "sat": 20}},
                                            "grading": {"shadows": {"hue": 220, "sat": 20}},
                                            "vignette": -20, "masks": [radial]}, 3),
        ("clarity, texture, sharpen", {**basic, "clarity": 30, "texture": 20, "sharpen": 40}, 3),
        ("film", {**basic, "film": editing.film_mod.stock("Warm portrait")}, 3),
        ("lens, tilt, crop, watermark", {**basic, "lens": {"distortion": 20},
                                          "tilt": 2.0, "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
                                          "watermark": {"enabled": True, "name": "Test"}}, 3),
    ]
    for name, edit, tol in cases:
        e = editing.normalize(edit)
        pad = int(np.ceil(editing.effect_padding(e, float(max(img.shape[:2])))))
        assert editing._band_count(e, img.shape[0], pad, editing.BAND_WORKERS) >= 2, name
        whole = editing.render(img, edit, meta={"file": "b"})
        banded = editing.render_bands(img, edit, meta={"file": "b"})
        assert banded.shape == whole.shape, name
        d = np.abs(banded.astype(int) - whole.astype(int))
        assert d.max() <= tol, (name, d.max())
        assert d.mean() < 0.05, (name, d.mean())


def test_edits_that_would_seam_are_rendered_whole() -> None:
    img = _detailed(900, 1300)
    brush = {"type": "brush", "feather": 40, "strokes": _strokes(3, 30), "adj": {"exposure": 0.5}}
    for edit in ({"dehaze": 30}, {"denoise": 30},
                 {"masks": [{"type": "linear", "x1": 0.5, "y1": 0, "x2": 0.5, "y2": 0.5,
                             "adj": {"dehaze": 40}}]},
                 {"exposure": 0.2, "masks": [brush]}):
        e = editing.normalize(edit)
        assert editing._band_count(e, img.shape[0], 10, editing.BAND_WORKERS) == 1, edit
        assert np.array_equal(editing.render_bands(img, edit), editing.render(img, edit)), edit


def _main() -> None:
    import re
    from pathlib import Path
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    declared = len(re.findall(r"^def test_", Path(__file__).read_text(), re.M))
    assert len(fns) == declared, (
        f"collected {len(fns)} of {declared} tests — the runner has to be the "
        f"last thing in the file")
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
