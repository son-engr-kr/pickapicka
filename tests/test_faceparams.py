"""Checks for a face's own portrait settings.

    uv run pytest tests/test_faceparams.py

Two faces built by hand side by side (the reshape tests' face, scaled and
moved), so that what one face's own settings do can be told apart from what
the panel does to the other. What has to hold: a face with its own entry
gets the entry and nothing of the panel, the other face gets the panel; the
entry finds its face by place; and an edit that goes to another photo leaves
the entries behind and keeps the target's.
"""
from __future__ import annotations

import math

import numpy as np

from pickapicka import editing, faceparams, portrait, reshape

N = 800


def _one_face(cx: float) -> dict:
    """The reshape tests' face, half size, centred at (cx, 0.5)."""
    import sys
    sys.path.insert(0, "tests")
    from test_reshape import _face
    f = _face()
    lm = (np.asarray(f["landmarks"]) - 0.5) * 0.5 + [cx, 0.5]
    bx, by, bw, bh = f["box"]
    box = [(bx - 0.5) * 0.5 + cx, (by - 0.5) * 0.5 + 0.5, bw * 0.5, bh * 0.5]
    return {"box": box, "landmarks": lm.tolist()}


A, B = _one_face(0.27), _one_face(0.73)


def _centre(f):
    return [f["box"][0] + f["box"][2] / 2, f["box"][1] + f["box"][3] / 2]


def _photo() -> np.ndarray:
    import cv2
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32) / N
    img = np.dstack([0.3 + 0.4 * xx, 0.3 + 0.4 * yy, np.full_like(xx, 0.5)])
    img += cv2.GaussianBlur(rng.normal(0, 0.08, (N, N)).astype(np.float32), (0, 0), 1.2)[..., None]
    return np.clip(img, 0, 1).astype(np.float32)


def _half(img: np.ndarray, side: str) -> np.ndarray:
    return img[:, : N // 2] if side == "a" else img[:, N // 2:]


def test_normalize_keeps_entries_and_their_place() -> None:
    p = portrait.normalize({"face_width": -50, "faces": [
        {"at": [0.73, 0.5], "face_width": 0},          # left out of the slimming
        {"at": [2.0, 0.5], "eye_size": 10},            # off the frame: dropped
        {"at": [0.3, 0.5], "eye_size": 400}]})
    assert p["faces"] == [{"at": [0.73, 0.5]}, {"at": [0.3, 0.5], "eye_size": 100}]
    assert portrait.normalize({"faces": [{"at": [0.5, 0.5]}]}) is None       # nothing set anywhere
    assert portrait.normalize({"faces": [{"at": [0.5, 0.5], "smooth": 20}]})["faces"][0]["smooth"] == 20


def test_an_entry_finds_its_face_by_place() -> None:
    params = {"smooth": 10, "faces": [{"at": _centre(B), "smooth": 60}]}
    assert faceparams.for_face(params, B, N, N)["smooth"] == 60
    assert faceparams.for_face(params, A, N, N)["smooth"] == 10
    # Moved by less than half its width: still that face's.
    near = {"faces": [{"at": [_centre(B)[0] + 0.4 * B["box"][2], _centre(B)[1]], "smooth": 60}]}
    assert faceparams.for_face(near, B, N, N)["smooth"] == 60
    far = {"faces": [{"at": [_centre(B)[0] + 0.6 * B["box"][2], _centre(B)[1]], "smooth": 60}]}
    assert faceparams.for_face(far, B, N, N).get("smooth", 0) == 0
    # Two entries near one face: the nearer wins.
    two = {"faces": [{"at": [_centre(B)[0] + 0.3 * B["box"][2], _centre(B)[1]], "smooth": 1},
                     {"at": _centre(B), "smooth": 2}]}
    assert faceparams.for_face(two, B, N, N)["smooth"] == 2


def test_a_face_with_its_own_settings_is_left_out_of_the_panels_reshape() -> None:
    img = _photo()
    both = reshape.apply_reshape(img, {"face_width": -100, "eye_size": 60}, [A, B])
    only_a = reshape.apply_reshape(img, {"face_width": -100, "eye_size": 60,
                                         "faces": [{"at": _centre(B)}]}, [A, B])
    assert not np.array_equal(_half(both, "b"), _half(img, "b"))
    assert np.array_equal(_half(only_a, "b"), _half(img, "b"))
    assert np.array_equal(_half(only_a, "a"), _half(both, "a"))


def test_one_face_reshaped_by_its_own_settings_alone() -> None:
    img = _photo()
    out = reshape.apply_reshape(img, {"faces": [{"at": _centre(A), "jaw_width": -100}]}, [A, B])
    assert reshape.is_active({"faces": [{"at": _centre(A), "jaw_width": -100}]})
    assert not np.array_equal(_half(out, "a"), _half(img, "a"))
    assert np.array_equal(_half(out, "b"), _half(img, "b"))


def test_a_window_matches_the_whole_render_with_entries() -> None:
    img, faces = _photo(), [A, B]
    params = {"chin_length": 80, "faces": [{"at": _centre(B), "face_width": -100, "chin_length": -40}]}
    whole = reshape.apply_reshape(img, params, faces)
    pad = int(math.ceil(reshape.padding(params, faces, N)))
    x0, y0, x1, y1 = 520, 360, 680, 520                   # B's jaw
    px0, py0, px1, py1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    win = reshape.apply_reshape(img[py0:py1, px0:px1], params, faces,
                                roi=(px0 / N, py0 / N, (px1 - px0) / N, (py1 - py0) / N))
    assert np.abs(win[pad:pad + y1 - y0, pad:pad + x1 - x0] - whole[y0:y1, x0:x1]).max() < 1e-5


def test_the_skin_cache_key_follows_an_entry() -> None:
    a = portrait.tone_params({"smooth": 10, "faces": [{"at": [0.7, 0.5], "smooth": 20}]})
    b = portrait.tone_params({"smooth": 10, "faces": [{"at": [0.7, 0.5], "smooth": 30}]})
    c = portrait.tone_params({"smooth": 10, "faces": [{"at": [0.7, 0.5], "smooth": 20, "eye_size": 40}]})
    assert a != b and a == c                 # the shape is not the skin stage's
    assert not portrait.tone_is_neutral({"faces": [{"at": [0.7, 0.5], "smooth": 20}]})
    assert portrait.tone_is_neutral({"faces": [{"at": [0.7, 0.5], "eye_size": 20}]})


def test_entries_stay_with_their_photo() -> None:
    from pickapicka.server import _keep_own
    base = {"portrait": {"smooth": 5, "faces": [{"at": [0.2, 0.5], "smooth": 40}]}}
    preset = {"portrait": {"smooth": 30, "faces": [{"at": [0.8, 0.5], "eye_size": 50}]}}
    added = editing.merge_additive(base, preset)
    assert added["portrait"]["smooth"] == 30
    assert added["portrait"]["faces"] == [{"at": [0.2, 0.5], "smooth": 40}]
    replaced = _keep_own(editing.normalize(preset), base)
    assert replaced["portrait"]["smooth"] == 30
    assert replaced["portrait"]["faces"] == [{"at": [0.2, 0.5], "smooth": 40}]
    assert portrait.without_faces(preset["portrait"]) == portrait.normalize({"smooth": 30})


def _skinned(f: dict) -> dict:
    """A face for the skin stages: its box as the crop, all of it skin."""
    return {**f, "crop": list(f["box"]), "skin": np.ones((32, 32), np.float32), "skin_luma": 0.5,
            "ramps": {"teeth": None, "eyes": None}}


def test_only_the_face_with_its_own_smoothing_is_smoothed() -> None:
    img = _photo()
    faces = [_skinned(A), _skinned(B)]
    out = portrait.apply_portrait(img, {"faces": [{"at": _centre(B), "smooth": 100}]}, faces)
    assert np.array_equal(_half(out, "a"), _half(img, "a"))
    assert not np.array_equal(_half(out, "b"), _half(img, "b"))
    # And the other way round: the panel smooths A, B's own zero keeps it out.
    out = portrait.apply_portrait(img, {"smooth": 100, "faces": [{"at": _centre(B)}]}, faces)
    assert not np.array_equal(_half(out, "a"), _half(img, "a"))
    assert np.array_equal(_half(out, "b"), _half(img, "b"))
