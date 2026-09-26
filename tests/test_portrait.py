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
    return {"box": [0.25, 0.22, 0.50, 0.56], "crop": [0.05, 0.05, 0.9, 0.9],
            "landmarks": lm.tolist(), "skin": np.ones((96, 96), np.float32),
            "skin_luma": 0.58}


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
    assert portrait.normalize({"smooth": 250}) == {"smooth": 100}
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
