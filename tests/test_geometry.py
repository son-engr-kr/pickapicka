"""Checks for the crop-and-straighten stage.

    uv run python tests/test_geometry.py

The pipeline treats the cropped result as the whole photo, so the property that
matters most is that `geometry_size` and `apply_geometry` never disagree: the
client lays the view out from the first and sees the second.
"""
from __future__ import annotations

import math

import numpy as np

from picture_classifier import editing


# ----- schema -------------------------------------------------------------

def test_defaults_are_no_geometry() -> None:
    e = editing.normalize({})
    assert e["tilt"] == 0.0 and e["crop"] is None
    assert editing.geometry_is_neutral(e)
    assert editing.geometry_is_neutral(None)


def test_a_crop_of_everything_is_not_a_crop() -> None:
    """Or the editor would look permanently dirty the moment the tool opened."""
    assert editing.normalize_crop({"x": 0, "y": 0, "w": 1, "h": 1}) is None
    assert editing.normalize({"crop": {"x": 0, "y": 0, "w": 1, "h": 1}})["crop"] is None
    assert editing.is_neutral({"crop": {"x": 0, "y": 0, "w": 1, "h": 1}})


def test_crop_is_clamped_into_the_frame() -> None:
    c = editing.normalize_crop({"x": -0.5, "y": 0.9, "w": 2.0, "h": 0.4})
    assert c == {"x": 0.0, "y": 0.6, "w": 1.0, "h": 0.4}, c
    # Never smaller than MIN_CROP, and never hanging off the right edge.
    tiny = editing.normalize_crop({"x": 0.99, "y": 0.99, "w": 0.0, "h": 0.0})
    assert tiny["w"] == editing.MIN_CROP and tiny["h"] == editing.MIN_CROP
    assert tiny["x"] + tiny["w"] <= 1.0 + 1e-9
    assert editing.normalize_crop("nonsense") is None
    assert editing.normalize_crop({"x": "a"}) is None


def test_tilt_is_a_float_and_clamped() -> None:
    assert editing.normalize({"tilt": 3.7})["tilt"] == 3.7, "must not round to 4"
    assert editing.normalize({"tilt": 999})["tilt"] == 45.0
    assert editing.normalize({"tilt": -999})["tilt"] == -45.0


def test_geometry_makes_an_edit_non_neutral() -> None:
    assert not editing.is_neutral({"tilt": 0.5})
    assert not editing.is_neutral({"crop": {"x": 0, "y": 0, "w": 0.5, "h": 1}})
    assert not editing.geometry_is_neutral({"tilt": 0.5})
    assert editing.geometry_is_neutral({"exposure": 1.0, "vignette": -50})


def test_hash_notices_geometry() -> None:
    """A cached thumbnail must not outlive a re-crop."""
    a = editing.edit_hash({"crop": {"x": 0, "y": 0, "w": 0.5, "h": 0.5}})
    b = editing.edit_hash({"crop": {"x": 0.1, "y": 0, "w": 0.5, "h": 0.5}})
    c = editing.edit_hash({"tilt": 1.0})
    d = editing.edit_hash({"tilt": 1.5})
    assert a and b and c and d
    assert len({a, b, c, d}) == 4


def test_a_preset_can_carry_a_crop() -> None:
    base = {"exposure": 0.5, "crop": {"x": 0, "y": 0, "w": 0.9, "h": 0.9}}
    over = {"crop": {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}, "tilt": 2.0}
    merged = editing.merge_additive(base, over)
    assert merged["crop"]["w"] == 0.5 and merged["tilt"] == 2.0
    assert merged["exposure"] == 0.5, "the overlay's crop must not flatten the grade"
    # A preset with no crop leaves the photo's own alone.
    kept = editing.merge_additive(base, {"contrast": 20})
    assert kept["crop"]["w"] == 0.9


# ----- sizes --------------------------------------------------------------

def _img(w: int = 400, h: int = 300) -> np.ndarray:
    rng = np.random.default_rng(4)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_size_and_pixels_always_agree() -> None:
    """The client lays out from geometry_size and then displays apply_geometry;
    a disagreement is a visibly wrong frame."""
    for w, h in ((400, 300), (300, 400), (401, 301), (500, 500)):
        for edit in ({}, {"tilt": 0.4}, {"tilt": 7.0}, {"tilt": -13.5}, {"tilt": 45.0},
                     {"crop": {"x": .1, "y": .2, "w": .5, "h": .6}},
                     {"tilt": 5.0, "crop": {"x": .05, "y": .05, "w": .8, "h": .3}},
                     {"tilt": -22.0, "crop": {"x": .3, "y": .3, "w": .5, "h": .5}}):
            out = editing.apply_geometry(_img(w, h), edit)
            want = editing.geometry_size(w, h, edit)
            got = (out.shape[1], out.shape[0])
            assert got == want, f"{w}x{h} {edit}: pixels {got} != arithmetic {want}"


def test_neutral_geometry_is_a_passthrough() -> None:
    img = _img()
    assert editing.apply_geometry(img, {}) is img, "must not copy for nothing"
    assert editing.apply_geometry(img, {"exposure": 1.0}) is img


def test_straighten_keeps_the_aspect() -> None:
    for deg in (1, 5, 12, 30, 44):
        w, h = editing.geometry_size(4000, 3000, {"tilt": deg})
        assert abs(w / h - 4000 / 3000) < 0.002, f"{deg}deg drifted to {w/h}"


def test_straighten_shrinks_monotonically() -> None:
    sizes = [editing.geometry_size(4000, 3000, {"tilt": d})[0]
             for d in (0, 2, 5, 10, 20, 40)]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[0] == 4000 and sizes[-1] < 4000


def test_straighten_leaves_no_blank_corner() -> None:
    """The whole point of cutting back to an inscribed rectangle."""
    flat = np.full((300, 400, 3), 210, np.uint8)
    for deg in (1.0, 6.0, 15.0, -25.0, 45.0):
        out = editing.apply_geometry(flat, {"tilt": deg})
        assert out.min() >= 195, f"{deg}deg left a dark edge: min {out.min()}"


def test_crop_takes_the_requested_pixels() -> None:
    """A marked corner has to come out where the crop says it should."""
    img = np.zeros((300, 400, 3), np.uint8)
    img[150:170, 200:220] = 255                     # a patch at (0.5, 0.5)
    out = editing.apply_geometry(img, {"crop": {"x": .5, "y": .5, "w": .5, "h": .5}})
    assert out.shape[:2] == (150, 200)
    assert out[0:20, 0:20].min() == 255, "the patch should now be the top-left"
    assert out[100:, 100:].max() == 0


def test_tilt_levels_a_horizon_of_the_same_sign() -> None:
    """The slider's sign has to be the one a person would reach for: a horizon
    drooping to the right is levelled by a positive value."""
    def horizon(deg: float, w: int = 600, h: int = 400) -> np.ndarray:
        img = np.full((h, w, 3), 40, np.uint8)
        yy, xx = np.mgrid[0:h, 0:w]
        img[yy > h / 2 + (xx - w / 2) * math.tan(math.radians(deg))] = 200
        return img

    def slope(img: np.ndarray) -> float:
        g = img[..., 0].astype(float)
        h, w = g.shape
        xs, ys = [], []
        for x in range(w // 6, w - w // 6, 5):
            edge = np.where(np.diff(g[:, x]) > 60)[0]
            if len(edge):
                xs.append(x)
                ys.append(edge[0])
        return math.degrees(math.atan(np.polyfit(xs, ys, 1)[0]))

    for off in (4.0, -6.0, 9.0):
        levelled = editing.apply_geometry(horizon(off), {"tilt": off})
        assert abs(slope(levelled)) < 0.4, \
            f"tilt {off} left {slope(levelled):.2f}deg — wrong sign?"


# ----- how it composes with the rest of the pipeline ----------------------

def test_render_applies_geometry_by_default() -> None:
    img = _img()
    out = editing.render(img, {"crop": {"x": 0, "y": 0, "w": .5, "h": .5}})
    assert out.shape[:2] == (150, 200)


def test_render_can_be_told_the_geometry_is_already_done() -> None:
    """The ROI path hands over a window of an already-cropped base; applying the
    crop again would cut a crop out of a crop."""
    img = _img()
    edit = {"crop": {"x": 0, "y": 0, "w": .5, "h": .5}, "exposure": 0.4}
    once = editing.render(img, edit, geometry=False)
    assert once.shape[:2] == img.shape[:2], "should have graded, not cropped"


def test_a_window_may_not_ask_for_geometry() -> None:
    """Silently cropping a window would put every mask in the wrong place, so it
    is refused rather than guessed at."""
    img = _img()
    try:
        editing.render(img, {"crop": {"x": 0, "y": 0, "w": .5, "h": .5}},
                       roi=(0.1, 0.1, 0.5, 0.5))
    except AssertionError:
        return
    raise AssertionError("expected an assertion for geometry on a partial ROI")


def test_the_grade_runs_before_the_geometry() -> None:
    """The order the whole feature turns on: adjustments are measured against the
    original frame and the crop is taken out of the result. Grading the crop
    instead would move every mask and re-centre the vignette the moment someone
    cropped."""
    img = np.full((300, 400, 3), 140, np.uint8)
    # Keep only the left half. A vignette applied to the *original* frame is dark
    # at the original's left edge and bright at its centre — which is now the
    # right edge of the crop, so the output brightens left to right.
    out = editing.render(img, {"crop": {"x": 0, "y": 0, "w": 0.5, "h": 1.0},
                               "vignette": -90})
    h, w = out.shape[:2]
    left = out[h // 2, : w // 8].mean()
    right = out[h // 2, -w // 8:].mean()
    assert right > left + 15, \
        f"vignette re-centred on the crop (left {left:.0f}, right {right:.0f})"


def test_a_window_cannot_be_stamped_by_render() -> None:
    """Only the caller knows where its window sits in the cropped frame, so
    render refuses to guess rather than putting the signature in the wrong place.
    """
    img = _img()
    edit = {"crop": {"x": 0, "y": 0, "w": .5, "h": .5},
            "watermark": {"enabled": True, "name": "X"}}
    try:
        editing.render(img, edit, roi=(0.1, 0.1, 0.4, 0.4), geometry=False)
    except AssertionError:
        return
    raise AssertionError("expected a refusal to stamp a window")


def test_a_crop_never_resamples() -> None:
    """With no straighten there is nothing to interpolate, so the crop has to be
    the original's own pixels — not a resampled copy of them."""
    img = _img(400, 300)
    out = editing.apply_geometry(img, {"crop": {"x": .25, "y": .25, "w": .5, "h": .5}})
    assert np.array_equal(out, img[75:225, 100:300]), "crop went through a resample"


def test_watermark_lands_on_the_cropped_frame() -> None:
    """Signing the uncropped frame would put the signature outside the picture."""
    img = np.full((400, 600, 3), 90, np.uint8)
    wm = {"enabled": True, "style": "bar", "name": "Someone",
          "position": "bottom-center"}
    out = editing.render(img, {"crop": {"x": 0, "y": 0, "w": 1.0, "h": 0.5},
                               "watermark": wm}, meta={})
    h = out.shape[0]
    assert h == 200
    changed = (out != 90).any(axis=(1, 2))
    assert changed[int(h * 0.75):].any(), "nothing was stamped in the lower half"
    assert not changed[: int(h * 0.5)].any(), "stamped above the middle"


def test_masks_stay_on_the_original_frame() -> None:
    """The complaint this ordering exists to answer: cropping must not drag a
    mask off the thing it was drawn on, nor change its size."""
    flat = np.full((600, 800, 3), 200, np.uint8)
    mask = {"type": "radial", "cx": 0.45, "cy": 0.30, "rx": 0.10, "ry": 0.10,
            "feather": 0, "adj": {"exposure": -3.0}}

    def bbox(img):
        ys, xs = np.nonzero(img[..., 0] < 150)
        assert len(xs), "the mask left no mark"
        return (xs.min() / img.shape[1], xs.max() / img.shape[1],
                ys.min() / img.shape[0], ys.max() / img.shape[0])

    base = bbox(editing.render(flat, {"masks": [mask]}))
    # Crops that all contain the mask outright, so nothing is clipped and any
    # movement is the pipeline's fault rather than the crop's edge.
    for crop in ({"x": .10, "y": .05, "w": .70, "h": .70},
                 {"x": .30, "y": .15, "w": .40, "h": .40},
                 {"x": .00, "y": .00, "w": .60, "h": .55},
                 {"x": .25, "y": .10, "w": .70, "h": .35}):
        out = editing.render(flat, {"masks": [mask], "crop": crop})
        x0, x1, y0, y1 = bbox(out)
        back = (crop["x"] + x0 * crop["w"], crop["x"] + x1 * crop["w"],
                crop["y"] + y0 * crop["h"], crop["y"] + y1 * crop["h"])
        drift = max(abs(back[i] - base[i]) * (800 if i < 2 else 600) for i in range(4))
        assert drift < 2.0, f"crop {crop} moved the mask by {drift:.1f}px"


def test_a_window_matches_the_same_part_of_the_whole_render() -> None:
    """What the 1:1 view rests on. The window is mapped back through the geometry,
    that patch of the original is graded, and the result is warped into place; it
    has to come out as the whole render's own pixels."""
    img = np.random.default_rng(9).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    cases = {
        "crop": {"crop": {"x": .2, "y": .15, "w": .5, "h": .6}, "exposure": .3,
                 "vignette": -40,
                 "masks": [{"type": "radial", "cx": .35, "cy": .4, "rx": .2,
                            "ry": .18, "feather": 40, "adj": {"exposure": -1.0}}]},
        "tilt": {"tilt": 6.0, "exposure": .2,
                 "masks": [{"type": "linear", "x1": .2, "y1": .2, "x2": .8,
                            "y2": .7, "feather": 50, "adj": {"exposure": -.8}}]},
        "both": {"tilt": -9.0, "crop": {"x": .1, "y": .1, "w": .6, "h": .6},
                 "contrast": 25,
                 "masks": [{"type": "radial", "cx": .5, "cy": .5, "rx": .25,
                            "ry": .25, "feather": 30, "adj": {"exposure": -1.2}}]},
    }
    for label, edit in cases.items():
        full = editing.render(img, edit)
        oh, ow = full.shape[:2]
        win = (int(ow * .2), int(oh * .25), max(1, int(ow * .5)), max(1, int(oh * .45)))
        pad = int(round(editing.effect_padding(edit, 400))) + 2
        box = editing.geometry_source_box(400, 300, edit, win, pad)
        bx, by, bw, bh = box
        graded = editing.render(img[by:by + bh, bx:bx + bw], edit,
                                roi=(bx / 400, by / 300, bw / 400, bh / 300),
                                with_watermark=False, geometry=False)
        got = editing.geometry_window(graded, box, 400, 300, edit, win)
        want = full[win[1]:win[1] + win[3], win[0]:win[0] + win[2]]
        assert got.shape == want.shape, f"{label}: {got.shape} vs {want.shape}"
        diff = np.abs(got.astype(int) - want.astype(int))
        # A pure crop is exact. A straighten resamples, and warping a patch cannot
        # land bit-identically on warping the whole frame near the patch's edge.
        limit = 0 if label == "crop" else 12
        assert diff.max() <= limit, f"{label}: max diff {diff.max()} > {limit}"


def test_the_normalized_matrix_agrees_with_the_pixels() -> None:
    """The editor places mask guides with this, so it has to say the same thing
    the renderer does about where a point ends up."""
    W, H = 400, 300
    for edit in ({"crop": {"x": .2, "y": .1, "w": .5, "h": .6}},
                 {"tilt": 8.0},
                 {"tilt": -5.0, "crop": {"x": .15, "y": .2, "w": .6, "h": .5}}):
        m, (ow, oh) = editing.geometry_matrix(W, H, edit)
        n = editing.geometry_norm_matrix(W, H, edit)
        for u, v in ((0.0, 0.0), (0.5, 0.5), (0.3, 0.8), (1.0, 1.0)):
            px = m @ np.array([u * W, v * H, 1.0])
            want = (px[0] / ow, px[1] / oh)
            got = (n[0] * u + n[1] * v + n[2], n[3] * u + n[4] * v + n[5])
            assert abs(got[0] - want[0]) < 1e-9 and abs(got[1] - want[1]) < 1e-9, \
                f"{edit} at ({u},{v}): {got} vs {want}"


def test_the_source_box_covers_the_window() -> None:
    """If the patch were too small the window would come out with a blank edge."""
    W, H = 400, 300
    for edit in ({"crop": {"x": .2, "y": .1, "w": .5, "h": .6}},
                 {"tilt": 12.0},
                 {"tilt": -20.0, "crop": {"x": .1, "y": .1, "w": .7, "h": .7}}):
        _, (ow, oh) = editing.geometry_matrix(W, H, edit)
        win = (ow // 4, oh // 4, max(1, ow // 3), max(1, oh // 3))
        box = editing.geometry_source_box(W, H, edit, win)
        bx, by, bw, bh = box
        # A patch that is entirely marked: anything unmarked in the placed window
        # is a pixel the box failed to supply.
        patch = np.full((bh, bw, 3), 255, np.uint8)
        placed = editing.geometry_window(patch, box, W, H, edit, win)
        assert placed.shape[:2] == (win[3], win[2]), \
            f"{edit}: window came out {placed.shape[:2]}, wanted {(win[3], win[2])}"
        assert placed.min() > 250, \
            f"{edit}: the patch did not cover the window (min {placed.min()})"


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
