"""Unit checks for the colour-grading wheels. Runnable with pytest or directly:

    uv run python tests/test_grading.py
"""
from __future__ import annotations

import numpy as np

from picture_classifier import grading

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _grey(value: float, size: int = 8) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float32)


def _ramp(h: int = 64, w: int = 64) -> np.ndarray:
    """A vertical black-to-white grey ramp: every zone gets some of the frame."""
    col = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    return np.repeat(np.repeat(col, w, axis=1), 3, axis=2)


def _patches() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One grey the default blending gives to each zone outright — the ramps
    have finished handing over by 0.07 and have not started again until 0.43, so
    the other two weights at these tones are not small but exactly zero."""
    return _grey(0.05), _grey(0.50), _grey(0.95)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def _sample(seed: int = 0, h: int = 32, w: int = 48) -> np.ndarray:
    return np.random.default_rng(seed).random((h, w, 3), dtype=np.float32)


# ----- schema -------------------------------------------------------------

def test_neutral_is_identity() -> None:
    img = _sample()
    for params in (None, {}, "junk", grading.DEFAULT_GRADING,
                   {"blending": 100, "balance": -50},
                   {"shadows": {"hue": 210}, "highlights": {"hue": 40}}):
        assert grading.is_neutral(params), f"should be neutral: {params}"
        out = grading.apply_grading(img, params)
        assert out.dtype == np.float32 and out.shape == img.shape
        assert np.array_equal(out, img), f"neutral grading changed pixels: {params}"
    assert not grading.is_neutral({"shadows": {"hue": 210, "sat": 20}})


def test_hue_alone_is_not_a_change() -> None:
    """A wheel dragged round its rim but never out from the centre is a setting,
    not an edit — the same line drawn for a motion angle with no motion."""
    assert grading.normalize({"midtones": {"hue": 300, "sat": 0, "lum": 0}}) is None
    # ...and a hue is dropped when sat is zero, since nothing is left to steer.
    assert grading.normalize({"midtones": {"hue": 300, "lum": 20}}) == {
        "midtones": {"lum": 20}}


def test_normalize_stores_sparsely() -> None:
    """Only what is off neutral survives, so the edit hash stays short and the
    globals do not appear until someone moves them."""
    got = grading.normalize({"shadows": {"hue": 0, "sat": 30, "lum": 0},
                             "midtones": {"hue": 120, "sat": 0, "lum": 0},
                             "highlights": {"hue": 40, "sat": 15, "lum": -8},
                             "blending": 50, "balance": 0})
    assert got == {"shadows": {"sat": 30},
                   "highlights": {"sat": 15, "lum": -8, "hue": 40}}
    assert grading.normalize({"shadows": {"sat": 30}, "balance": 25}) == {
        "shadows": {"sat": 30}, "balance": 25}


def test_normalize_clamps_wraps_and_is_idempotent() -> None:
    got = grading.normalize({"shadows": {"hue": 370, "sat": 500, "lum": -900},
                             "midtones": {"hue": -30, "sat": 12},
                             "blending": 900, "balance": -900})
    assert got == {"shadows": {"hue": 10, "sat": 100, "lum": -100},
                   "midtones": {"hue": 330, "sat": 12},
                   "blending": 100, "balance": -100}
    assert grading.normalize(got) == got
    # Junk in a field falls back to that field's neutral rather than exploding.
    assert grading.normalize({"shadows": {"sat": "x", "lum": None}}) is None


# ----- zone weights -------------------------------------------------------

def test_weights_are_a_smooth_partition_of_unity() -> None:
    """The three zones must account for every luminance exactly once: sum to 1
    everywhere, never negative, and never step. Anything else and a smooth
    gradient shows a band where the zones hand over."""
    y = np.linspace(0.0, 1.0, 2001, dtype=np.float32)
    for blending in (0, 25, 50, 75, 100):
        for balance in (-100, -40, 0, 40, 100):
            ws = grading._zone_weights(y, blending, balance)
            total = sum(ws)
            assert np.allclose(total, 1.0, atol=1e-6), (blending, balance)
            for w in ws:
                assert w.min() >= -1e-7, f"negative weight at {blending}/{balance}"
                assert w.max() <= 1.0 + 1e-6
                # 2000 steps of luminance: a real edge would jump far more.
                assert np.abs(np.diff(w)).max() < 0.02, (blending, balance)


def test_mid_grey_is_all_midtones_at_the_defaults() -> None:
    """Untouched globals must give three clean zones, or every wheel bleeds into
    every tone and the tool stops being three-way."""
    ws = grading._zone_weights(np.array([0.5], dtype=np.float32))
    assert ws[1][0] > 0.999
    assert ws[0][0] < 1e-6 and ws[2][0] < 1e-6


def test_balance_hands_the_frame_to_one_end() -> None:
    """Positive balance moves the split down, so more of the frame falls under
    the highlight wheel; negative gives it to the shadows."""
    y = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    share = []
    for balance in (-100, -50, 0, 50, 100):
        shadows, _, highlights = grading._zone_weights(y, 50, balance)
        share.append((float(shadows.mean()), float(highlights.mean())))
    highs = [h for _, h in share]
    lows = [s for s, _ in share]
    assert highs == sorted(highs) and highs[0] < highs[-1] - 0.1
    assert lows == sorted(lows, reverse=True) and lows[0] > lows[-1] + 0.1


def test_blending_widens_the_overlap() -> None:
    """Blending is the width of the hand-over. At 0 the zones stay out of each
    other's way; wound up, all three reach a mid grey at once."""
    y = np.linspace(0.0, 1.0, 4001, dtype=np.float32)
    widths = []
    for blending in (0, 25, 50, 75, 100):
        shadows, _, highlights = grading._zone_weights(y, blending, 0)
        widths.append(int(np.count_nonzero((shadows > 0.01) & (shadows < 0.99))))
        three = (shadows > 0.01) & (highlights > 0.01)
        if blending == 0:
            assert not three.any(), "zones should be disjoint at blending 0"
        if blending == 100:
            assert three.any(), "zones should overlap at blending 100"
    assert widths == sorted(widths) and widths[0] * 3 < widths[-1]


# ----- what the wheels do -------------------------------------------------

def test_each_zone_reaches_only_its_own_tones() -> None:
    dark, mid, bright = _patches()
    cases = {"shadows": (dark, (mid, bright)),
             "midtones": (mid, (dark, bright)),
             "highlights": (bright, (dark, mid))}
    for zone, (target, others) in cases.items():
        params = {zone: {"hue": 210, "sat": 100}}
        moved = grading.apply_grading(target, params)
        assert np.abs(moved - target).max() > 0.05, f"{zone} did not grade its own tones"
        for other in others:
            out = grading.apply_grading(other, params)
            assert np.abs(out - other).max() < 1e-6, f"{zone} leaked into another zone"


def test_saturation_leaves_brightness_alone() -> None:
    """A sat push is a move in the plane of constant luma, so it changes colour
    while the frame's brightness stays where the photographer put it."""
    img = _sample(1, 64, 64)
    for zone in grading.ZONES:
        for hue in (0, 45, 120, 210, 300):
            out = grading.apply_grading(img, {zone: {"hue": hue, "sat": 100}})
            assert np.abs(_luma(out) - _luma(img)).max() < 1e-5, (zone, hue)
            # ...and it does have to actually do something.
            assert np.abs(out - img).max() > 0.02, (zone, hue)


def test_saturation_adds_chroma_in_the_hue_it_was_asked_for() -> None:
    """A blue push must leave a blue cast, and the opposite hue the opposite
    one, so the wheel's angle means what the UI draws."""
    grey = _grey(0.5)
    blue = grading.apply_grading(grey, {"midtones": {"hue": 240, "sat": 100}})
    yellow = grading.apply_grading(grey, {"midtones": {"hue": 60, "sat": 100}})
    assert blue[..., 2].mean() > blue[..., 0].mean() + 0.1
    assert yellow[..., 2].mean() < yellow[..., 0].mean() - 0.1
    for out in (blue, yellow):
        spread = out.max(axis=2) - out.min(axis=2)
        assert spread.mean() > 0.1, "no chroma was added at all"


def test_lum_brightens_and_darkens_only_its_zone() -> None:
    dark, mid, bright = _patches()
    for zone, target in zip(grading.ZONES, (dark, mid, bright)):
        up = grading.apply_grading(target, {zone: {"lum": 100}})
        down = grading.apply_grading(target, {zone: {"lum": -100}})
        assert up.mean() > target.mean() + 0.02, zone
        assert down.mean() < target.mean() - 0.02, zone
        # A lum move stays in gamut: towards white or black, never past either.
        for out in (up, down):
            assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6, zone
        for other in (dark, mid, bright):
            if other is target:
                continue
            for params in ({zone: {"lum": 100}}, {zone: {"lum": -100}}):
                out = grading.apply_grading(other, params)
                assert np.abs(out - other).max() < 1e-6, zone


def test_lum_does_not_shift_the_hue() -> None:
    """Brightening a zone is a tonal move; the colour that is there must survive
    it, or the lum slider quietly becomes a saturation slider."""
    rgb = np.zeros((4, 4, 3), dtype=np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.5, 0.3, 0.2
    for lum in (100, -100):
        out = grading.apply_grading(rgb, {"midtones": {"lum": lum}})
        order = np.argsort(out.reshape(-1, 3)[0])
        assert list(order) == [2, 1, 0], f"channel order changed at lum {lum}"


def test_wheels_compose_over_the_whole_ramp() -> None:
    """All three wheels at once on a black-to-white ramp: the grade must stay
    monotone in luminance, or a gradient develops a reversal."""
    ramp = _ramp(256, 4)
    out = grading.apply_grading(ramp, {
        "shadows": {"hue": 210, "sat": 60, "lum": 20},
        "midtones": {"hue": 90, "sat": 25, "lum": -10},
        "highlights": {"hue": 40, "sat": 50, "lum": 15},
        "blending": 70, "balance": 15})
    y = _luma(out)[:, 0]
    assert np.all(np.diff(y) > -1e-6), "grading put a reversal in a grey ramp"


# ----- the lookup and its guarantees --------------------------------------

def _reference(rgb: np.ndarray, params: dict) -> np.ndarray:
    """The grade written out the slow, obvious way: weights per pixel, then each
    zone's luminance move and tint applied in turn. `apply_grading` folds all of
    this into one gain and one offset per luminance; this is what that has to
    agree with."""
    p = grading.normalize(params)
    ws = grading._zone_weights(_luma(rgb), p.get("blending", 50), p.get("balance", 0))
    out = rgb.astype(np.float32)
    for zone, w in zip(grading.ZONES, ws):
        k = (p.get(zone) or {}).get("lum", 0) / 100.0 * grading._LUM_MAX
        if k > 0:
            out = out + (w * k)[..., None] * (1.0 - out)
        elif k < 0:
            out = out * (1.0 + (w * k)[..., None])
    for zone, w in zip(grading.ZONES, ws):
        z = p.get(zone) or {}
        if z.get("sat"):
            v = grading._tint_vector(z.get("hue", 0))
            out = out + (w * (z["sat"] / 100.0 * grading._TINT_MAX))[..., None] * v
    return out


def test_lookup_matches_a_direct_evaluation() -> None:
    """The table is a shortcut, not an approximation: quantizing the luminance
    index is the only difference, and it has to stay under an 8-bit level even
    with every wheel at its limit and blending at 0, where the ramps that feed
    it are at their steepest."""
    img = _sample(2, 96, 96)
    for params in ({"shadows": {"hue": 210, "sat": 80, "lum": -30}},
                   {"midtones": {"hue": 30, "sat": 40, "lum": 25},
                    "blending": 0},
                   {"shadows": {"hue": 300, "sat": 60, "lum": 40},
                    "midtones": {"hue": 150, "sat": 30, "lum": -20},
                    "highlights": {"hue": 60, "sat": 90, "lum": 35},
                    "blending": 100, "balance": -60},
                   {"shadows": {"hue": 210, "sat": 100, "lum": -100},
                    "midtones": {"hue": 90, "sat": 100, "lum": 100},
                    "highlights": {"hue": 40, "sat": 100, "lum": 100},
                    "blending": 0}):
        err = float(np.abs(grading.apply_grading(img, params) - _reference(img, params)).max())
        assert err < 0.5 / 255.0, f"{params}: off by {err * 255} of a level"


def test_tint_vectors_are_luma_free_and_evenly_strong() -> None:
    """Every hue on the wheel pushes the same distance and none of them carries
    any brightness with it."""
    for hue in range(0, 360, 3):
        v = grading._tint_vector(hue)
        assert abs(float(v @ _LUMA)) < 1e-6, f"hue {hue} carries luma"
        assert abs(float(np.abs(v).max()) - 1.0) < 1e-6, f"hue {hue} is off scale"
    assert np.allclose(grading._tint_vector(370), grading._tint_vector(10))


def test_grading_is_pointwise() -> None:
    """The strongest form of the resolution guarantee the pipeline works for: a
    colour maps to a colour with no reference to a pixel's neighbours, so the
    preview and the export agree exactly and no padding is needed.

    Note this is not testable by downscaling one render and comparing — a
    nonlinear pointwise map never commutes with averaging. Nearest-neighbour
    replication is, because it produces the same values.
    """
    small = _sample(3, 24, 24)
    big = np.repeat(np.repeat(small, 5, axis=0), 5, axis=1)
    params = {"shadows": {"hue": 220, "sat": 70, "lum": -25},
              "highlights": {"hue": 45, "sat": 55, "lum": 20},
              "blending": 65, "balance": 20}
    assert np.array_equal(grading.apply_grading(small, params),
                          grading.apply_grading(big, params)[::5, ::5])


def test_row_bands_do_not_show() -> None:
    """The stage runs in row bands for cache reasons; a frame taller than one
    band must come out exactly as its bands do separately."""
    img = _sample(4, grading._TILE_ROWS * 2 + 7, 16)
    params = {"midtones": {"hue": 180, "sat": 50, "lum": 30}}
    out = grading.apply_grading(img, params)
    for r0 in (0, grading._TILE_ROWS, grading._TILE_ROWS * 2):
        band = grading.apply_grading(img[r0:r0 + grading._TILE_ROWS], params)
        assert np.array_equal(out[r0:r0 + band.shape[0]], band)


def test_apply_rejects_the_wrong_kind_of_array() -> None:
    params = {"shadows": {"sat": 40}}
    for bad in (np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4, 4), dtype=np.float32),
                np.zeros((4, 4, 3), dtype=np.float64),
                np.zeros((4, 4, 3), dtype=np.uint8)):
        try:
            grading.apply_grading(bad, params)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad.shape} {bad.dtype}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
