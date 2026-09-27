"""Checks for skin smoothing.

    uv run python tests/test_portrait.py

The face here is built by hand: a box, landmarks with the eye, brow and mouth
groups where a face has them, and a skin alpha. That tests the smoothing
without the face model, which `analyze` needs and these checks do not. What
has to hold is that it evens skin out, leaves the features and everything
outside the face alone, and gives a window the same pixels the whole render
gives it.
"""
from __future__ import annotations

import numpy as np

from picture_classifier import editing, portrait

# A 300 px face: at much less the pores are sub-pixel and there is nothing
# for a texture split to keep.
N = 600


def _face() -> dict:
    lm = np.full((106, 2), 0.5, np.float32)
    groups = {portrait.EYE_L: (0.40, 0.42), portrait.EYE_R: (0.60, 0.42),
              portrait.BROW_L: (0.40, 0.35), portrait.BROW_R: (0.60, 0.35),
              portrait.MOUTH: (0.50, 0.66)}
    rng = np.random.default_rng(1)
    for idx, (cx, cy) in groups.items():
        lm[list(idx)] = [cx, cy] + rng.uniform(-0.03, 0.03, (len(idx), 2))
    # Skin as an oval inside the face box, over the crop, the shape the
    # segmenter gives a face.
    import cv2
    skin = np.zeros((96, 96), np.float32)
    box, crop = [0.25, 0.22, 0.50, 0.56], [0.05, 0.05, 0.9, 0.9]
    cx = (box[0] + box[2] / 2 - crop[0]) / crop[2] * 96
    cy = (box[1] + box[3] / 2 - crop[1]) / crop[3] * 96
    cv2.ellipse(skin, (int(cx), int(cy)), (int(box[2] / crop[2] * 48), int(box[3] / crop[3] * 48)),
                0, 0, 360, 1.0, -1)
    return {"box": box, "crop": crop,
            "landmarks": lm.tolist(), "skin": cv2.GaussianBlur(skin, (0, 0), 1.5),
            "skin_luma": 0.58,
            "ramps": {"teeth": [55.0, 80.0], "eyes": [60.0, 88.0]}}


def _skin() -> np.ndarray:
    """Pores (fine noise) over blotches (mid-scale), on a skin tone. The
    amplitudes are a real cheek's, measured at 0.33 luma (0.014 for the
    blotches, 0.016 for the pores) and scaled to this brighter skin."""
    import cv2
    rng = np.random.default_rng(7)
    base = np.empty((N, N, 3), np.float32)
    base[:] = (0.72, 0.55, 0.45)
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (N, N)).astype(np.float32), (0, 0), 9.0)
    blotch *= 0.014 * 0.58 / 0.33 / blotch.std()
    pores = rng.normal(0, 0.016 * 0.58 / 0.33, (N, N)).astype(np.float32)
    return np.clip(base + (blotch + pores)[..., None], 0, 1).astype(np.float32)


def _band(img: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import cv2
    y = img.mean(axis=2)
    return cv2.GaussianBlur(y, (0, 0), lo) - cv2.GaussianBlur(y, (0, 0), hi)


def test_normalize() -> None:
    assert portrait.normalize({"smooth": 0}) is None
    assert portrait.normalize({"smooth": 250}) == {"smooth": 100, "teeth": 0, "eyes": 0}
    assert portrait.normalize({"teeth": 40}) == {"smooth": 0, "teeth": 40, "eyes": 0}
    assert portrait.normalize("nonsense") is None


def test_blotches_even_out_and_pores_stay() -> None:
    img, face = _skin(), _face()
    out = portrait.apply_portrait(img, {"smooth": 100}, [face])
    cheek = (slice(310, 370), slice(190, 250))      # skin, clear of the features
    face_px = face["box"][2] * N
    sigma = portrait._TEX_SIGMA * face_px
    mid_before = _band(img, sigma, 4.6 * sigma)[cheek].std()
    mid_after = _band(out, sigma, 4.6 * sigma)[cheek].std()
    # "Pores" is what is finer than the operator's own texture split, which is
    # a fraction of the face width; measured at any coarser scale, part of what
    # is counted is the band the smoothing is meant to take out.
    fine_before = (img - _blur(img, sigma))[cheek].std()
    fine_after = (out - _blur(out, sigma))[cheek].std()
    assert mid_after < 0.6 * mid_before, (mid_before, mid_after)
    assert fine_after > 0.9 * fine_before, (fine_before, fine_after)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    return cv2.GaussianBlur(img, (0, 0), sigma)


def test_features_and_the_background_are_left_alone() -> None:
    img, face = _skin(), _face()
    out = portrait.apply_portrait(img, {"smooth": 100}, [face])
    eye = (slice(int(0.41 * N), int(0.43 * N)), slice(int(0.39 * N), int(0.41 * N)))
    assert np.abs(out[eye] - img[eye]).max() < 0.02
    corner = (slice(0, 20), slice(0, 20))            # outside the face's crop
    assert np.array_equal(out[corner], img[corner])


def test_a_window_matches_the_whole_render() -> None:
    img, faces = _skin(), [_face()]
    whole = portrait.apply_portrait(img, {"smooth": 80}, faces)
    pad = int(np.ceil(portrait.padding({"smooth": 80}, faces, N)))
    x0, y0, x1, y1 = 220, 250, 380, 410
    px0, py0, px1, py1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    win = portrait.apply_portrait(img[py0:py1, px0:px1], {"smooth": 80}, faces,
                                  roi=(px0 / N, py0 / N, (px1 - px0) / N, (py1 - py0) / N))
    inner = win[pad:pad + (y1 - y0), pad:pad + (x1 - x0)]
    assert np.abs(inner - whole[y0:y1, x0:x1]).max() < 1e-4


def _smile() -> tuple[np.ndarray, dict]:
    """Yellowish teeth in the inner-mouth polygon, a dark gap under them, and
    red lips round it."""
    import cv2
    img = _skin()
    face = _face()
    lm = np.asarray(face["landmarks"], np.float32)
    inner = np.array([[0.42, 0.64], [0.46, 0.63], [0.50, 0.63], [0.54, 0.63], [0.58, 0.64],
                      [0.54, 0.69], [0.50, 0.70], [0.46, 0.69]], np.float32)
    lm[list(portrait.MOUTH_INNER)] = inner
    face["landmarks"] = lm.tolist()
    cv2.ellipse(img, (300, 400), (60, 28), 0, 0, 360, (0.62, 0.22, 0.24), -1)   # lips
    cv2.fillPoly(img, [(inner * N).astype(np.int32)], (0.20, 0.08, 0.08))       # the dark of the mouth
    # Upper teeth at L 78, a 4, b 22: a real smile measured a 6, b 21.
    teeth = cv2.cvtColor(np.float32([[[78.0, 4.0, 22.0]]]), cv2.COLOR_LAB2RGB)[0, 0]
    cv2.rectangle(img, (256, 378), (344, 396), tuple(float(v) for v in teeth), -1)
    return img, face


def _b_star(px: np.ndarray) -> float:
    import cv2
    return float(cv2.cvtColor(px.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[..., 2].mean())


def test_teeth_lose_their_yellow_and_the_rest_of_the_mouth_does_not_move() -> None:
    img, face = _smile()
    out = portrait.apply_portrait(img, {"teeth": 100}, [face])
    teeth = (slice(382, 392), slice(280, 320))
    gap = (slice(405, 410), slice(290, 310))
    lip = (slice(420, 424), slice(290, 310))
    assert _b_star(out[teeth]) < 0.6 * _b_star(img[teeth])
    assert out[teeth].mean() > img[teeth].mean()
    assert np.abs(out[gap] - img[gap]).max() < 0.01
    assert np.abs(out[lip] - img[lip]).max() < 0.01


def test_whitening_in_a_window_matches_the_whole_render() -> None:
    """The brightness ramp is stored per face, so half a mouth in a 1:1 window
    whitens exactly as the whole mouth does."""
    img, face = _smile()
    params = {"teeth": 80}
    whole = portrait.apply_portrait(img, params, [face])
    x0, y0, x1, y1 = 300, 360, 380, 420          # the right half of the mouth
    pad = int(np.ceil(portrait.padding(params, [face], N)))
    px0, py0 = x0 - pad, y0 - pad
    win = portrait.apply_portrait(img[py0:y1 + pad, px0:x1 + pad], params, [face],
                                  roi=(px0 / N, py0 / N, (x1 - x0 + 2 * pad) / N, (y1 - y0 + 2 * pad) / N))
    assert np.abs(win[pad:pad + y1 - y0, pad:pad + x1 - x0] - whole[y0:y1, x0:x1]).max() < 1e-4


def _with_spots(spots) -> np.ndarray:
    """The test skin, uint8, with spots planted as (x, y, dL, da) in CIELAB
    units, each 1% of the face width across."""
    import cv2
    img = _skin()
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    face_px = _face()["box"][2] * N
    yy, xx = np.mgrid[0:N, 0:N]
    for x, y, dl, da in spots:
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (0.01 * face_px) ** 2)).astype(np.float32)
        lab[..., 0] += dl * g
        lab[..., 1] += da * g
    return np.rint(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255).astype(np.uint8)


def test_blemishes_are_found_where_they_are_and_nowhere_else() -> None:
    """A dark spot and a red one of a visible contrast (8 L, 12 a*) are found;
    the skin's own pores and blotches, at a real cheek's amplitude, are not."""
    face = _face()
    assert portrait.find_blemishes(_with_spots([]), [face]) == []
    planted = [(215, 330, -8.0, 0.0), (385, 345, 0.0, 12.0)]
    ops = portrait.find_blemishes(_with_spots(planted), [face])
    assert len(ops) == 2, ops
    for x, y, _, _ in planted:
        assert any(abs(o["points"][0][0] * N - x) < 4 and abs(o["points"][0][1] * N - y) < 4 for o in ops)
    assert all(o["kind"] == "spot" and 0 < o["radius"] < 0.05 for o in ops)


def test_a_spot_on_a_feature_is_not_a_blemish() -> None:
    """Dark and round, but in an eye: a pupil, not a blemish."""
    face = _face()
    eye = np.asarray(face["landmarks"])[list(portrait.EYE_L)].mean(axis=0) * N
    assert portrait.find_blemishes(_with_spots([(eye[0], eye[1], -20.0, 0.0)]), [face]) == []


def test_render_needs_the_faces_and_smooths_with_them() -> None:
    img = np.rint(_skin() * 255).astype(np.uint8)
    edit = {"portrait": {"smooth": 60}}
    assert not editing.is_neutral(edit)
    try:
        editing.render(img, edit)
    except AssertionError as exc:
        assert "faces" in str(exc)
    else:
        raise AssertionError("smoothed skin without being told where the faces are")
    assert np.array_equal(editing.render(img, edit, faces=[]), img)
    out = editing.render(img, edit, faces=[_face()])
    assert not np.array_equal(out, img)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
