"""Unit checks for the editing pipeline. Runnable with pytest or directly:

    uv run python tests/test_editing.py
"""
from __future__ import annotations

import numpy as np

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


def test_builtin_presets_are_valid_edits() -> None:
    from picture_classifier import presets
    img = _detailed(200, 300)
    ids = set()
    for p in presets.list_builtins():
        assert p["id"] not in ids, f"duplicate preset id {p['id']}"
        ids.add(p["id"])
        assert p["builtin"] is True and p["name"] and p["hint"]
        e = editing.normalize(p["edit"])
        assert not editing.is_neutral(e), f"{p['id']} does nothing"
        # Every mask a preset ships must survive normalization.
        assert len(e["masks"]) == len(p["edit"].get("masks", [])), p["id"]
        out = editing.render(img, p["edit"])
        assert out.shape == img.shape and out.dtype == np.uint8
        assert presets.is_builtin(p["id"])
    assert not presets.is_builtin("nope")


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
