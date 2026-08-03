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


def test_geometry_runs_before_the_grade() -> None:
    """Order matters: grading first and cropping after would let the vignette and
    the masks be positioned against a frame the viewer never sees."""
    img = np.full((300, 400, 3), 120, np.uint8)
    edit = {"crop": {"x": 0, "y": 0, "w": 0.5, "h": 1.0}, "vignette": -90}
    out = editing.render(img, edit)
    h, w = out.shape[:2]
    # The vignette darkens the corners of whatever it is given. Applied to the
    # crop, the crop's own centre column stays bright.
    centre = out[h // 2, w // 2].mean()
    corner = out[2, 2].mean()
    assert centre > corner + 15, \
        f"vignette does not follow the crop (centre {centre:.0f}, corner {corner:.0f})"


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


def test_masks_are_relative_to_the_cropped_frame() -> None:
    """Documented behaviour, pinned so it cannot drift silently: a mask is placed
    in the frame the crop produced, so crop first and mask second."""
    img = np.full((300, 400, 3), 128, np.uint8)
    mask = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.3, "ry": 0.3,
            "feather": 0, "adj": {"exposure": -2.0}}
    out = editing.render(img, {"crop": {"x": 0.5, "y": 0, "w": 0.5, "h": 1.0},
                               "masks": [mask]})
    h, w = out.shape[:2]
    # Darkened at the centre of the *crop*, not at the centre of the original.
    assert out[h // 2, w // 2].mean() < 90, "mask did not land in the crop's centre"
    assert out[5, 5].mean() > 110, "mask spilled to the crop's corner"


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
