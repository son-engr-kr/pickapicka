"""Checks for face reshaping.

    uv run python tests/test_reshape.py

The face is built by hand: the 106 landmarks laid out where the model puts
them (a jaw line down each side to the chin, eyes, nose wings, mouth corners),
which is all the warp reads. What has to hold is that each slider moves the
part it names in the direction it says, that nothing folds over even with
every slider at its end, that the photo away from the face is untouched, and
that a window gets the same pixels the whole render gives it.
"""
from __future__ import annotations

import math

import numpy as np

from pickapicka import editing, portrait, reshape

N = 600


def _face() -> dict:
    lm = np.full((106, 2), 0.5, np.float64)
    cx, cy, a, b = 0.5, 0.48, 0.19, 0.27
    for k in range(17):
        # Down each side from the temple at eye level to the chin.
        phi = math.pi / 2 * (k / 16)
        lm[reshape.CONTOUR_L[k]] = [cx - a * math.cos(phi), cy + b * math.sin(phi)]
        lm[reshape.CONTOUR_R[k]] = [cx + a * math.cos(phi), cy + b * math.sin(phi)]
    for ring, (outer, inner), ex, out in ((reshape.EYE_RING_L, reshape.EYE_CORNERS_L, 0.42, -1),
                                          (reshape.EYE_RING_R, reshape.EYE_CORNERS_R, 0.58, 1)):
        lm[outer] = [ex + 0.04 * out, 0.45]
        lm[inner] = [ex - 0.04 * out, 0.45]
        rest = [i for i in ring if i not in (outer, inner)]
        for j, i in enumerate(rest):
            t = 2 * math.pi * (j + 0.5) / len(rest)
            lm[i] = [ex + 0.03 * math.cos(t), 0.45 + 0.012 * math.sin(t)]
    for idx, x in ((reshape.NOSE_WING_L, 0.47), (reshape.NOSE_WING_R, 0.53)):
        for i, dy in zip(idx, (-0.03, 0.0, 0.02)):
            lm[i] = [x + (0.004 if x < 0.5 else -0.004) * (dy != 0.0), 0.58 + dy]
    lm[reshape.MOUTH_CORNERS[0]] = [0.44, 0.67]
    lm[reshape.MOUTH_CORNERS[1]] = [0.56, 0.67]
    return {"box": [0.3, 0.2, 0.4, 0.56], "landmarks": lm.tolist()}


def _photo() -> np.ndarray:
    """Texture everywhere, so that a moved pixel shows."""
    import cv2
    rng = np.random.default_rng(3)
    img = np.empty((N, N, 3), np.float32)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32) / N
    img[..., 0], img[..., 1], img[..., 2] = 0.3 + 0.4 * xx, 0.3 + 0.4 * yy, 0.5
    img += cv2.GaussianBlur(rng.normal(0, 0.08, (N, N)).astype(np.float32), (0, 0), 1.2)[..., None]
    return np.clip(img, 0, 1)


def _landed(params: dict, idx: int, face: dict | None = None) -> np.ndarray:
    """Where landmark `idx` ends up, in pixels: the x with x + D(x) at it."""
    face = face or _face()
    moves, scalings = reshape.plan(face, params, N, N)
    p = np.asarray(face["landmarks"][idx]) * N
    x = p.copy()
    for _ in range(80):
        dx, dy = reshape.field(moves, scalings, np.array([x[0]]), np.array([x[1]]))
        x = p - np.array([dx[0, 0], dy[0, 0]])
    return x


def _at(idx: int) -> np.ndarray:
    return np.asarray(_face()["landmarks"][idx]) * N


ALL_IN = {"eye_size": 100, "face_width": -100, "jaw_width": -100, "chin_length": 100,
          "nose_width": -100, "mouth_width": -100}
ALL_OUT = {k: -v for k, v in ALL_IN.items()}


def test_normalize_keeps_old_panels_as_they_were() -> None:
    assert portrait.normalize({"smooth": 20}) == {"smooth": 20, "teeth": 0, "eyes": 0}
    assert portrait.normalize({"eye_size": 30}) == {"smooth": 0, "teeth": 0, "eyes": 0, "eye_size": 30}
    assert portrait.normalize({"jaw_width": -300})["jaw_width"] == -100
    assert portrait.normalize({"chin_length": 0}) is None


def test_a_neutral_shape_is_a_no_op() -> None:
    img = _photo()
    assert reshape.apply_reshape(img, {"smooth": 40}, [_face()]) is img
    assert reshape.apply_reshape(img, {"eye_size": 50}, []) is img
    assert reshape.padding({"smooth": 40}, [_face()], N) == 0.0


def test_each_slider_moves_what_it_names() -> None:
    l_cheek, r_cheek = reshape.CONTOUR_L[7], reshape.CONTOUR_R[7]
    assert _landed({"face_width": -100}, l_cheek)[0] > _at(l_cheek)[0] + 3
    assert _landed({"face_width": -100}, r_cheek)[0] < _at(r_cheek)[0] - 3
    assert _landed({"face_width": 100}, l_cheek)[0] < _at(l_cheek)[0] - 3

    jaw = reshape.CONTOUR_L[12]
    mouth = (_at(52) + _at(61)) / 2
    before = np.linalg.norm(_at(jaw) - mouth)
    assert np.linalg.norm(_landed({"jaw_width": -100}, jaw) - mouth) < before - 3

    assert _landed({"chin_length": 100}, 0)[1] > _at(0)[1] + 3
    assert _landed({"chin_length": -100}, 0)[1] < _at(0)[1] - 3

    wings = (reshape.NOSE_WING_L[1], reshape.NOSE_WING_R[1])
    width = lambda p: _landed(p, wings[1])[0] - _landed(p, wings[0])[0]
    assert width({"nose_width": -100}) < (_at(wings[1]) - _at(wings[0]))[0] - 2
    assert width({"nose_width": 100}) > (_at(wings[1]) - _at(wings[0]))[0] + 2

    corners = reshape.MOUTH_CORNERS
    mouth_w = lambda p: _landed(p, corners[1])[0] - _landed(p, corners[0])[0]
    assert mouth_w({"mouth_width": 100}) > (_at(corners[1]) - _at(corners[0]))[0] + 3

    outer, inner = reshape.EYE_CORNERS_L
    eye_w = lambda p: np.linalg.norm(_landed(p, inner) - _landed(p, outer))
    assert eye_w({"eye_size": 100}) > np.linalg.norm(_at(inner) - _at(outer)) * 1.03
    assert eye_w({"eye_size": -100}) < np.linalg.norm(_at(inner) - _at(outer)) * 0.97


def test_one_slider_leaves_the_other_features_alone() -> None:
    for idx in (reshape.MOUTH_CORNERS[0], reshape.NOSE_WING_L[1], reshape.EYE_CORNERS_L[0]):
        assert np.abs(_landed({"chin_length": 100}, idx) - _at(idx)).max() < 0.5
    assert np.abs(_landed({"eye_size": 100}, 0) - _at(0)).max() < 0.5


def test_nothing_folds_with_every_slider_at_its_end() -> None:
    for params in (ALL_IN, ALL_OUT):
        moves, scalings = reshape.plan(_face(), params, N, N)
        xs = ys = np.arange(0.0, N, 0.5)
        dx, dy = reshape.field(moves, scalings, xs, ys)
        g = 0.5
        det = (1 + np.gradient(dx, g, axis=1)) * (1 + np.gradient(dy, g, axis=0)) \
            - np.gradient(dx, g, axis=0) * np.gradient(dy, g, axis=1)
        assert det.min() > 0.3, det.min()


def test_away_from_the_face_nothing_changes() -> None:
    img = _photo()
    out = reshape.apply_reshape(img, ALL_IN, [_face()])
    assert not np.array_equal(out, img)
    for corner in ((slice(0, 40), slice(0, 40)), (slice(-40, None), slice(-40, None))):
        assert np.array_equal(out[corner], img[corner])


def test_a_window_matches_the_whole_render() -> None:
    img, faces = _photo(), [_face()]
    whole = reshape.apply_reshape(img, ALL_IN, faces)
    pad = int(math.ceil(reshape.padding(ALL_IN, faces, N)))
    for x0, y0, x1, y1 in ((160, 330, 300, 470), (240, 230, 360, 300)):   # a jaw, the eyes
        px0, py0, px1, py1 = max(0, x0 - pad), max(0, y0 - pad), min(N, x1 + pad), min(N, y1 + pad)
        win = reshape.apply_reshape(img[py0:py1, px0:px1], ALL_IN, faces,
                                    roi=(px0 / N, py0 / N, (px1 - px0) / N, (py1 - py0) / N))
        inner = win[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
        assert np.abs(inner - whole[y0:y1, x0:x1]).max() < 1e-5


def test_padding_covers_the_farthest_read() -> None:
    for params in (ALL_IN, ALL_OUT):
        moves, scalings = reshape.plan(_face(), params, N, N)
        xs = ys = np.arange(0.0, N, 1.0)
        dx, dy = reshape.field(moves, scalings, xs, ys)
        assert np.sqrt(dx * dx + dy * dy).max() + 4 <= reshape.padding(params, [_face()], N)


def test_render_reshapes_with_the_faces_and_needs_them() -> None:
    img = np.rint(_photo() * 255).astype(np.uint8)
    edit = {"portrait": {"face_width": -80}}
    try:
        editing.render(img, edit)
    except AssertionError as exc:
        assert "faces" in str(exc)
    else:
        raise AssertionError("reshaped a face without being told where it is")
    assert np.array_equal(editing.render(img, edit, faces=[]), img)
    out = editing.render(img, edit, faces=[_face()])
    assert not np.array_equal(out, img)
    assert np.array_equal(out[:40, :40], img[:40, :40])


def test_a_window_of_the_render_matches_with_effect_padding() -> None:
    """Through `editing.render`, with a grade before the warp: the padding has
    to cover the grade's reach and the warp's on top of it."""
    img = np.rint(_photo() * 255).astype(np.uint8)
    faces = [_face()]
    edit = {"clarity": 40, "portrait": {"jaw_width": -100, "chin_length": 60}}
    whole = editing.render(img, edit, faces=faces)
    pad = int(math.ceil(editing.effect_padding(edit, N, faces)))
    x0, y0, x1, y1 = 170, 350, 290, 470
    px0, py0, px1, py1 = max(0, x0 - pad), max(0, y0 - pad), min(N, x1 + pad), min(N, y1 + pad)
    win = editing.render(img[py0:py1, px0:px1], edit, faces=faces, geometry=False,
                         roi=(px0 / N, py0 / N, (px1 - px0) / N, (py1 - py0) / N))
    diff = np.abs(win[y0 - py0:y1 - py0, x0 - px0:x1 - px0].astype(int) - whole[y0:y1, x0:x1].astype(int))
    assert diff.max() <= 1, diff.max()


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
