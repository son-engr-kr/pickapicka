"""Checks for the skin tone hold.

    uv run python tests/test_skintone.py

The colour rule is checked on a*b* values directly, and the render on the
hand-built face of the portrait checks, so neither the face model nor the
segmenter is needed. What has to hold: a warm edit's extra chroma on skin is
eased and never past its limit, lightness never moves, black and white and a
cast being taken off go through, the yellow side is held tighter than the
red, nothing outside the skin changes, and a window gets the pixels the whole
render gives it.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from pickapicka import editing, portrait, skintone
from test_portrait import N, _face, _skin


def _lab(c: float, h: float) -> np.ndarray:
    return np.array([[c * math.cos(math.radians(h)), c * math.sin(math.radians(h))]], np.float32)


def _ch(ab: np.ndarray) -> tuple[float, float]:
    a, b = float(ab[0, 0]), float(ab[0, 1])
    return math.hypot(a, b), math.degrees(math.atan2(b, a))


def test_the_knee_is_smooth_and_never_reaches_its_limit() -> None:
    x = np.linspace(0, 60, 601, dtype=np.float32)
    y = skintone.knee(x, 5.0)
    assert np.all(np.diff(y) >= 0) and float(y.max()) <= 5.0
    assert np.all(np.diff(y) <= np.diff(x) + 1e-6)          # slope at most 1
    assert abs(float(skintone.knee(np.float32(0.5), 5.0)) - 0.5) < 0.01


def test_a_warm_edit_keeps_some_warmth_and_no_more_than_the_limit() -> None:
    ref = _lab(22.0, 51.0)                                  # ColorChecker light skin
    for extra in (6.0, 18.0, 30.0):
        graded = _lab(22.0 + extra, 54.0)
        c, _ = _ch(skintone.hold(graded, ref))
        assert 22.0 < c < 22.0 + skintone.CHROMA_LIMIT + 0.5, (extra, c)
    c_small, _ = _ch(skintone.hold(_lab(23.0, 51.0), ref))
    assert abs(c_small - 23.0) < 0.1                        # a small change goes through


def test_black_and_white_and_a_fade_go_through() -> None:
    ref = _lab(22.0, 51.0)
    for graded in (np.zeros((1, 2), np.float32), _lab(9.0, 51.0)):
        assert np.abs(skintone.hold(graded, ref) - graded).max() < 1e-4


def test_a_cast_being_taken_off_goes_through() -> None:
    # A shaded face gone magenta: turning it towards the preferred hue is free,
    # turning it past that is not.
    ref = _lab(14.0, 0.0)
    _, h = _ch(skintone.hold(_lab(14.0, 40.0), ref))
    assert abs(h - 40.0) < 0.5
    _, h = _ch(skintone.hold(_lab(14.0, 80.0), ref))
    assert skintone.PREFERRED_HUE < h < 70.0


def test_a_turned_hue_keeps_its_chroma() -> None:
    # A film look turning skin towards yellow: held back as a turn, not as
    # a turn with the skin's colour drained out of it.
    ref = _lab(22.0, 50.0)
    c, h = _ch(skintone.hold(_lab(22.0, 80.0), ref))
    assert abs(c - 22.0) < 0.1 and 50.0 < h < 60.0


def test_yellow_is_held_tighter_than_red() -> None:
    ref = _lab(24.0, skintone.PREFERRED_HUE)
    _, yellow = _ch(skintone.hold(_lab(24.0, skintone.PREFERRED_HUE + 25), ref))
    _, red = _ch(skintone.hold(_lab(24.0, skintone.PREFERRED_HUE - 25), ref))
    assert yellow - skintone.PREFERRED_HUE < skintone.PREFERRED_HUE - red


def test_near_grey_skin_does_not_turn_noisy() -> None:
    # A specular highlight: a*b* is noise round a grey, and its direction says
    # nothing. Split along it, a warm shift would scatter with the noise.
    rng = np.random.default_rng(0)
    ref = (2.0 + rng.normal(0, 1, (5000, 2))).astype(np.float32)
    kept = skintone.hold(ref + np.float32([8.0, 14.0]), ref) - ref
    assert float(kept.std(axis=0).max()) < 0.5


def _warm_face() -> tuple[np.ndarray, list[dict], dict]:
    img = np.rint(_skin() * 255).astype(np.uint8)
    face = _face()
    face["neck"] = np.zeros((96, 96), np.uint8)
    return img, [face], {"body": np.zeros((64, 64), np.float32)}


def _skin_lab(out: np.ndarray) -> np.ndarray:
    patch = out[280:320, 280:320].astype(np.float32) / 255.0
    return cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).reshape(-1, 3).mean(axis=0)


def test_render_holds_the_skin_and_leaves_the_rest() -> None:
    img, faces, auto = _warm_face()
    warm = editing.render(img, {"temp": 80}, faces=faces)
    held = editing.render(img, {"temp": 80, "portrait": {"skin_tone": 100}}, faces=faces, auto=auto)
    before, w, h = _skin_lab(img), _skin_lab(warm), _skin_lab(held)
    c = lambda lab: math.hypot(lab[1], lab[2])
    assert c(before) < c(h) < c(w) and c(h) - c(before) < skintone.CHROMA_LIMIT + 0.5
    assert abs(h[0] - w[0]) < 0.5                           # lightness is the edit's
    corner = (slice(0, 20), slice(0, 20))                    # outside the face's crop
    assert np.array_equal(held[corner], warm[corner])


def test_render_needs_the_faces_and_the_body_field() -> None:
    img, faces, auto = _warm_face()
    edit = {"temp": 40, "portrait": {"skin_tone": 100}}
    for kw in ({"auto": auto}, {"faces": faces}):
        try:
            editing.render(img, edit, **kw)
        except AssertionError:
            continue
        raise AssertionError(f"held skin tones without {set(('faces', 'auto')) - set(kw)}")
    assert editing.auto_groups(edit) == {"body"}
    # With no colour edit there is nothing to hold, and nothing is asked for.
    assert editing.auto_groups({"exposure": 0.5, "portrait": {"skin_tone": 100}}) == set()
    plain = editing.render(img, {"exposure": 0.5}, faces=faces)
    assert np.array_equal(editing.render(img, {"exposure": 0.5, "portrait": {"skin_tone": 100}},
                                         faces=faces), plain)


def test_a_face_of_its_own_can_be_left_out() -> None:
    img, faces, auto = _warm_face()
    at = [faces[0]["box"][0] + faces[0]["box"][2] / 2, faces[0]["box"][1] + faces[0]["box"][3] / 2]
    panel = portrait.normalize({"skin_tone": 100, "faces": [{"at": at, "skin_tone": 0}]})
    warm = editing.render(img, {"temp": 80}, faces=faces)
    out = editing.render(img, {"temp": 80, "portrait": panel}, faces=faces, auto=auto)
    assert np.array_equal(out, warm)


def test_a_window_matches_the_whole_render() -> None:
    img, faces, auto = _warm_face()
    edit = {"temp": 70, "saturation": 20, "portrait": {"skin_tone": 100}}
    whole = editing.render(img, edit, faces=faces, auto=auto)
    x0, y0, x1, y1 = 220, 250, 380, 410
    win = editing.render(img[y0:y1, x0:x1], edit, faces=faces, auto=auto,
                         roi=(x0 / N, y0 / N, (x1 - x0) / N, (y1 - y0) / N), geometry=False)
    assert int(np.abs(win.astype(int) - whole[y0:y1, x0:x1].astype(int)).max()) <= 1


def test_the_panel_without_it_normalizes_as_before() -> None:
    assert portrait.normalize({"smooth": 30}) == {"smooth": 30, "teeth": 0, "eyes": 0}
    assert portrait.normalize({"skin_tone": 60})["skin_tone"] == 60
    assert portrait.skin_tone_is_neutral({"smooth": 30})


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
