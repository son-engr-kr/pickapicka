"""Unit checks for the colour-range and luminance-range masks. Runnable with
pytest or directly:

    uv run python tests/test_rangemask.py
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from picture_classifier import rangemask


# Three colours that make the point about metrics: a dark red, a bright red of
# the same hue, and a dark blue. A colour selection sampled on the first must
# rank the second nearer than the third.
DARK_RED = (80, 0, 0)
BRIGHT_RED = (255, 60, 60)
DARK_BLUE = (0, 0, 80)
BLACK = (0, 0, 0)


def _patches(colors, size: int = 48) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """An image of uniform square patches, plus the centre of each one. Sampling
    centres keeps the selector's edge blur out of the assertions."""
    img = np.zeros((size, size * len(colors), 3), dtype=np.uint8)
    centres = []
    for i, c in enumerate(colors):
        img[:, i * size:(i + 1) * size] = c
        centres.append((size // 2, i * size + size // 2))
    return img, centres


def _lstar(grey: np.ndarray | int) -> np.ndarray:
    """L* of an 8-bit neutral, straight from OpenCV so the tests never carry a
    hand-copied constant."""
    g = np.asarray(grey, dtype=np.float32).reshape(-1, 1, 1).repeat(3, axis=2) / 255.0
    return cv2.cvtColor(g, cv2.COLOR_RGB2Lab)[:, 0, 0]


def _grey_ramp(h: int = 64, w: int = 256) -> np.ndarray:
    """A column-wise 0..255 neutral ramp: column j has grey level j*255/(w-1)."""
    col = np.linspace(0, 255, w, dtype=np.float32).round().astype(np.uint8)
    return np.repeat(col[None, :, None], 3, axis=2).repeat(h, axis=0)


# ----- normalization ------------------------------------------------------

def test_normalize_luma_rejects_select_everything() -> None:
    for raw in (None, "nope", 3, {}, rangemask.LUMA_DEFAULT,
                {"lo": 0, "hi": 100}, {"lo": -50, "hi": 500},
                {"lo": None, "hi": None, "feather_lo": 30}):
        assert rangemask.normalize_luma(raw) is None, f"should be neutral: {raw}"


def test_normalize_luma_clamps() -> None:
    m = rangemask.normalize_luma({"lo": 70, "hi": 30, "feather_lo": -5,
                                  "feather_hi": 500})
    assert m == {"lo": 30, "hi": 70, "feather_lo": 0, "feather_hi": 100}
    # A zero-width window is not a selection; it opens to the minimum instead.
    z = rangemask.normalize_luma({"lo": 50, "hi": 50})
    assert z is not None and z["hi"] - z["lo"] >= rangemask.MIN_LUMA_WINDOW
    top = rangemask.normalize_luma({"lo": 100, "hi": 100})
    assert top is not None and top["hi"] - top["lo"] >= rangemask.MIN_LUMA_WINDOW
    assert rangemask.normalize_luma({"lo": "x", "hi": 40}) == {
        "lo": 0, "hi": 40, "feather_lo": 10, "feather_hi": 10}


def test_normalize_color_rejects_select_everything() -> None:
    for raw in (None, "nope", {}, rangemask.COLOR_DEFAULT,
                {"samples": []}, {"samples": [[1, 2]]}, {"samples": "red"},
                {"samples": [[10, 20, 30]], "range": 100},
                {"samples": [[10, 20, 30]], "range": 999}):
        assert rangemask.normalize_color(raw) is None, f"should be neutral: {raw}"


def test_normalize_color_clamps() -> None:
    m = rangemask.normalize_color({"samples": [[300, -5, 20.6], [1, 2]] + [[0, 0, 0]] * 9,
                                   "range": -3, "feather": 1e9})
    assert m is not None
    assert m["samples"][0] == [255, 0, 21]
    assert len(m["samples"]) == rangemask.COLOR_SAMPLES_MAX
    assert m["range"] == 0 and m["feather"] == 100


def test_is_neutral() -> None:
    assert rangemask.is_neutral(None)
    assert rangemask.is_neutral(rangemask.LUMA_DEFAULT)
    assert rangemask.is_neutral(rangemask.COLOR_DEFAULT)
    assert not rangemask.is_neutral({"lo": 20, "hi": 60})
    assert not rangemask.is_neutral({"samples": [list(DARK_RED)], "range": 30})


# ----- the alpha contract -------------------------------------------------

def test_alpha_contract() -> None:
    """Same shape, dtype and range as editing's geometric alphas — `_apply_masks`
    multiplies these together and indexes them with the image's own slices."""
    img = _grey_ramp(70, 130)
    luma = rangemask.normalize_luma({"lo": 30, "hi": 70})
    color = rangemask.normalize_color({"samples": [[128, 128, 128]], "range": 40})
    for alpha in (rangemask.luma_alpha(img, luma), rangemask.color_alpha(img, color)):
        assert alpha.dtype == np.float32, alpha.dtype
        assert alpha.shape == img.shape[:2], alpha.shape
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0
        assert alpha.flags.writeable and alpha.flags.c_contiguous


def test_alpha_accepts_float_input_and_out_hw() -> None:
    img = _grey_ramp(64, 256)
    luma = rangemask.normalize_luma({"lo": 30, "hi": 70})
    from_u8 = rangemask.luma_alpha(img, luma)
    from_float = rangemask.luma_alpha(img.astype(np.float32) / 255.0, luma)
    assert np.abs(from_u8 - from_float).max() < 1e-3

    small = rangemask.luma_alpha(img, luma, out_hw=(16, 64))
    assert small.shape == (16, 64) and small.dtype == np.float32
    # Same selection, just carried on a coarser grid.
    assert abs(float(small.mean()) - float(from_u8.mean())) < 0.02


# ----- luminance range ----------------------------------------------------

def test_luma_window_selects_its_band() -> None:
    greys = [20, 90, 150, 235]
    img, centres = _patches([(g, g, g) for g in greys])
    target = float(_lstar(greys[2])[0])
    params = rangemask.normalize_luma({"lo": target - 5, "hi": target + 5,
                                       "feather_lo": 2, "feather_hi": 2})
    alpha = rangemask.luma_alpha(img, params)
    picked = [float(alpha[y, x]) for y, x in centres]
    assert picked[2] > 0.98, picked
    assert max(picked[:2] + picked[3:]) < 0.02, picked


def test_luma_feather_width_and_placement() -> None:
    """The ramp must span exactly `feather_lo` in L*, sitting below `lo`."""
    img = _grey_ramp(32, 256)
    lum = _lstar(np.arange(256))
    for flo in (5, 20, 40):
        params = rangemask.normalize_luma({"lo": 70, "hi": 100, "feather_lo": flo,
                                           "feather_hi": 0})
        row = rangemask.luma_alpha(img, params).mean(axis=0)
        # smoothstep hits 0.5 at the middle of its ramp and 0.05/0.95 at
        # 0.1273/0.8727 of it, so those crossings pin both width and placement.
        mid = float(np.interp(0.5, row, lum))
        lo_x = float(np.interp(0.05, row, lum))
        hi_x = float(np.interp(0.95, row, lum))
        assert abs(mid - (70 - flo / 2)) < 1.5, (flo, mid)
        assert abs((hi_x - lo_x) - 0.745 * flo) < max(1.5, 0.2 * flo), (flo, lo_x, hi_x)


def test_luma_no_banding_on_a_gradient() -> None:
    """A smooth gradient must give a smooth alpha: monotone through each ramp and
    no step big enough to read as a contour."""
    img = _grey_ramp(16, 1024)
    params = rangemask.normalize_luma({"lo": 45, "hi": 55, "feather_lo": 25,
                                       "feather_hi": 25})
    row = rangemask.luma_alpha(img, params).mean(axis=0)
    peak = int(np.argmax(row))
    rise, fall = row[:peak + 1], row[peak:]
    assert np.all(np.diff(rise) >= -1e-4), "rising edge is not monotone"
    assert np.all(np.diff(fall) <= 1e-4), "falling edge is not monotone"
    assert float(np.abs(np.diff(row)).max()) < 0.02, float(np.abs(np.diff(row)).max())
    assert row[peak] > 0.99 and row[0] < 0.01 and row[-1] < 0.01


def test_luma_window_is_stable_across_resolutions() -> None:
    """The point of the capped working grid: the same photo selects the same
    thing at preview size and at export size."""
    base = _grey_ramp(400, 600)
    params = rangemask.normalize_luma({"lo": 40, "hi": 60})
    big = cv2.resize(base, (3000, 2000), interpolation=cv2.INTER_LINEAR)
    frac_small = float(rangemask.luma_alpha(base, params).mean())
    frac_big = float(rangemask.luma_alpha(big, params).mean())
    assert abs(frac_small - frac_big) < 0.01, (frac_small, frac_big)


# ----- colour range -------------------------------------------------------

def test_color_selects_sample_and_rejects_another() -> None:
    img, centres = _patches([(40, 150, 60), (240, 140, 40), (120, 170, 230)])
    params = rangemask.normalize_color({"samples": [[40, 150, 60]], "range": 30,
                                        "feather": 25})
    alpha = rangemask.color_alpha(img, params)
    picked = [float(alpha[y, x]) for y, x in centres]
    assert picked[0] > 0.99, picked
    assert max(picked[1:]) < 0.01, picked


def test_color_metric_beats_rgb_distance() -> None:
    """Sampled on a dark red, a colour selection must reach the bright red of the
    same hue before it reaches a dark blue or black.

    RGB Euclidean distance gets this exactly backwards — it is the reason this
    module converts to Lab and splits out hue at all.
    """
    rgb_d = {name: float(np.linalg.norm(np.array(c, float) - np.array(DARK_RED, float)))
             for name, c in (("bright", BRIGHT_RED), ("blue", DARK_BLUE), ("black", BLACK))}
    assert rgb_d["bright"] > rgb_d["blue"] > rgb_d["black"], rgb_d

    img, centres = _patches([DARK_RED, BRIGHT_RED, DARK_BLUE, BLACK])
    # One range setting has to admit the bright red and refuse the other two.
    params = rangemask.normalize_color({"samples": [list(DARK_RED)], "range": 45,
                                        "feather": 30})
    a = [float(rangemask.color_alpha(img, params)[y, x]) for y, x in centres]
    assert a[0] > 0.99, a
    assert a[1] > 0.9, a          # same hue, four stops brighter: selected
    assert a[2] < 0.1, a          # same darkness, other side of the wheel: not
    assert a[3] < 0.1, a          # no colour at all: not

    # And the ordering holds regardless of where the threshold is put.
    wide = rangemask.normalize_color({"samples": [list(DARK_RED)], "range": 100 - 1,
                                      "feather": 0})
    full = rangemask.color_alpha(img, wide)
    assert float(full[centres[1]]) >= float(full[centres[2]]) >= float(full[centres[3]])


def test_color_multiple_samples_union() -> None:
    green, orange, blue = (40, 150, 60), (240, 140, 40), (120, 170, 230)
    img, centres = _patches([green, orange, blue])
    p_one = rangemask.normalize_color({"samples": [list(green)], "range": 30})
    p_two = rangemask.normalize_color({"samples": [list(green), list(orange)],
                                       "range": 30})
    one = rangemask.color_alpha(img, p_one)
    two = rangemask.color_alpha(img, p_two)
    # The union only ever adds.
    assert np.all(two >= one - 1e-6)
    picked = [float(two[y, x]) for y, x in centres]
    assert picked[0] > 0.99 and picked[1] > 0.99, picked
    assert picked[2] < 0.01, picked
    assert float(one[centres[1]]) < 0.01


def test_color_range_widens_monotonically() -> None:
    img, centres = _patches([(40, 150, 60), (120, 170, 230)])
    got = []
    for rng in (0, 20, 40, 60, 80, 99):
        params = rangemask.normalize_color({"samples": [[40, 150, 60]], "range": rng,
                                            "feather": 30})
        got.append(float(rangemask.color_alpha(img, params)[centres[1]]))
    assert got[0] < 0.01 and got[-1] > 0.99, got
    assert all(b >= a - 1e-6 for a, b in zip(got, got[1:])), got


# ----- noise --------------------------------------------------------------

def _noisy(color, h: int = 683, w: int = 1024, sigma: float = 6.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    flat = np.full((h, w, 3), color, dtype=np.float32)
    return np.clip(flat + rng.normal(0, sigma, (h, w, 3)), 0, 255).astype(np.uint8)


def _raw_lab(img: np.ndarray) -> np.ndarray:
    """Lab with no working-resolution reduction and no selector blur: what the
    alpha would look like if the mask trusted every pixel."""
    return cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)


def _speckle(alpha: np.ndarray) -> float:
    """Mean pixel-to-pixel jump. A smooth selection is near zero here; a
    speckled one is not."""
    return float(np.abs(np.diff(alpha, axis=1)).mean())


def _roughness(alpha: np.ndarray) -> float:
    """Pixel-to-pixel jump measured against the variation's own size. Adjacent
    samples of white noise differ by 2/sqrt(pi) = 1.13 standard deviations, so
    this reads ~1.1 for true speckle and falls toward 0 as the variation spreads
    over more pixels. Scale-free, which the raw jump is not: how *much* the
    alpha wanders depends on how narrow a band the parameters asked for, but
    whether it wanders per pixel is the actual question."""
    return _speckle(alpha) / max(float(alpha.std()), 1e-6)


def test_luma_no_speckle_on_noise() -> None:
    img = _noisy((128, 128, 128))
    lo, flo = 56, 8
    params = rangemask.normalize_luma({"lo": lo, "hi": 100, "feather_lo": flo,
                                       "feather_hi": 0})
    alpha = rangemask.luma_alpha(img, params)
    # Mid-ramp is the worst case: that is where noise has the most leverage.
    assert 0.2 < float(alpha.mean()) < 0.9, float(alpha.mean())

    t = np.clip((_raw_lab(img)[..., 0] - (lo - flo)) / flo, 0, 1)
    naive = t * t * (3 - 2 * t)
    assert _roughness(naive) > 1.0, _roughness(naive)   # the thing to avoid
    assert _speckle(alpha) < _speckle(naive) / 10.0, (_speckle(alpha), _speckle(naive))
    assert _roughness(alpha) < 0.4, _roughness(alpha)
    assert float(alpha.std()) < 0.07, float(alpha.std())


def test_color_no_speckle_on_noise() -> None:
    color = (180, 60, 60)
    img = _noisy(color)
    # A range that lands this patch mid-ramp, where noise has the most leverage.
    params = rangemask.normalize_color({"samples": [[150, 60, 60]], "range": 23,
                                        "feather": 40})
    alpha = rangemask.color_alpha(img, params)
    assert 0.1 < float(alpha.mean()) < 0.95, float(alpha.mean())

    thresh = (params["range"] / 100.0) ** 2 * rangemask._COLOR_RANGE_MAX
    band = max(rangemask._COLOR_BAND_FLOOR, thresh * params["feather"] / 100.0)
    sample = np.asarray([params["samples"][0]], np.float32).reshape(1, 1, 3) / 255.0
    d = np.sqrt(rangemask._color_distance(
        _raw_lab(img), cv2.cvtColor(sample, cv2.COLOR_RGB2Lab)))
    t = np.clip((d - (thresh - band / 2)) / band, 0, 1)
    naive = 1.0 - t * t * (3 - 2 * t)
    assert _roughness(naive) > 1.0, _roughness(naive)
    assert _speckle(alpha) < _speckle(naive) / 10.0, (_speckle(alpha), _speckle(naive))
    assert _roughness(alpha) < 0.4, _roughness(alpha)
    # A looser amplitude bound than the luminance case, honestly: the colour
    # metric reads the noisy a,b channels too, and this band is 2.5 units wide.
    # Note also that this frame is exactly the working size, so the area-average
    # reduction a real 24 MP file gets is not in play here.
    assert float(alpha.std()) < 0.12, float(alpha.std())


# ----- cost ---------------------------------------------------------------

def _time(fn, *args, **kw) -> float:
    fn(*args, **kw)                       # warm the allocator and OpenCV's pools
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        fn(*args, **kw)
        best = min(best, time.perf_counter() - t0)
    return best


def test_cost_is_flat_in_megapixels() -> None:
    """Four times the pixels must not cost four times as much: everything except
    the two resamples happens on the capped working grid."""
    params = rangemask.normalize_color({"samples": [[120, 170, 230], [40, 150, 60]],
                                        "range": 35, "feather": 25})
    six = _grey_ramp(2000, 3000)
    twentyfour = _grey_ramp(4000, 6000)
    t6 = _time(rangemask.color_alpha, six, params)
    t24 = _time(rangemask.color_alpha, twentyfour, params)
    print(f"colour range: 6 MP {t6 * 1000:.0f} ms, 24 MP {t24 * 1000:.0f} ms")
    assert t24 < 2.6 * t6, (t6, t24)      # 4x pixels, well under 4x the time
    assert t24 < 1.5, t24                 # generous: the real figure is ~5x lower

    # Asked for the alpha on the mask grid, the tail of O(pixels) work is only
    # the read of the source.
    work = rangemask._work_size(4000, 6000)
    t24_work = _time(rangemask.color_alpha, twentyfour, params, out_hw=work)
    print(f"colour range at work size: 24 MP {t24_work * 1000:.0f} ms")
    assert t24_work < 2.6 * t6, (t6, t24_work)


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        getattr(mod, name)()
        print(f"ok  {name}")
