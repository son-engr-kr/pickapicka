"""Unit checks for the Detail / Sharpening stage. Runnable with pytest or directly:

    uv run python tests/test_sharpening.py
"""
from __future__ import annotations

import cv2
import numpy as np

from pickapicka import sharpening


# ----- builders -----------------------------------------------------------

def _sample(edge: int = 96) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((edge, edge, 3), dtype=np.float32)


def _ripple(period: float, edge: int = 128, amp: float = 0.10) -> np.ndarray:
    """A sine grating: one spatial frequency and nothing else, which is how you
    ask an operator what it does to a given scale."""
    xs = np.arange(edge, dtype=np.float32)
    y = 0.5 + amp * np.sin(2.0 * np.pi * xs / period)[None, :].repeat(edge, 0)
    return np.repeat(y[..., None], 3, axis=2)


def _texture(edge: int = 192, amp: float = 0.08) -> np.ndarray:
    """Broadband detail — noise with a little correlation, so it has structure at
    every scale rather than only at the pixel."""
    rng = np.random.default_rng(1)
    n = cv2.GaussianBlur(rng.normal(0.0, 1.0, (edge, edge)).astype(np.float32), (0, 0), 0.7)
    y = np.clip(0.5 + amp * (n / n.std()), 0.0, 1.0)
    return np.repeat(y[..., None], 3, axis=2)


def _edge(lo: float = 0.3, hi: float = 0.7, width: int = 96,
          blur: float = 1.3) -> np.ndarray:
    """A blurred step: the thing sharpening is for, and the thing halos live on."""
    img = np.full((48, width, 3), lo, np.float32)
    img[:, width // 2:] = hi
    return cv2.GaussianBlur(img, (0, 0), blur)


def _band(img: np.ndarray, s1: float, s2: float | None = None) -> float:
    """Energy in one octave of the image: above `s1`, or between `s1` and `s2`."""
    y = sharpening._luma(img)
    low = cv2.GaussianBlur(y, (0, 0), s1)
    if s2 is None:
        return float((y - low).std())
    return float((low - cv2.GaussianBlur(y, (0, 0), s2)).std())


def _run(img: np.ndarray, **params) -> np.ndarray:
    return sharpening.apply_sharpen(img, params, float(max(img.shape[:2])))


# ----- neutrality and normalization ---------------------------------------

def test_neutral_is_identity() -> None:
    img = _sample()
    for params in ({}, dict(sharpening.DEFAULT_SHARPEN), {"amount": 0},
                   {"amount": 0, "radius": 100, "detail": 100, "masking": 100}):
        out = sharpening.apply_sharpen(img, params, 96.0)
        assert out is img, f"neutral params copied or changed the array: {params}"


def test_is_neutral() -> None:
    assert sharpening.is_neutral(None)
    assert sharpening.is_neutral({})
    assert sharpening.is_neutral(sharpening.DEFAULT_SHARPEN)
    # Only amount decides: the other three describe how to sharpen, and with
    # nothing to apply they describe nothing.
    assert sharpening.is_neutral({"radius": 90, "detail": 100, "masking": 60})
    assert not sharpening.is_neutral({"amount": 1})


def test_normalize_neutral_forms() -> None:
    for raw in (None, {}, 0, "nonsense", {"amount": 0}, {"detail": 100},
                {"amount": None}, {"amount": -20}, dict(sharpening.DEFAULT_SHARPEN)):
        assert sharpening.normalize(raw) is None, f"should normalize away: {raw}"


def test_normalize_fills_and_clamps() -> None:
    got = sharpening.normalize({"amount": 250, "radius": -5, "masking": 61.4})
    assert got == {"amount": 100, "radius": 0, "detail": 25, "masking": 61}
    # A key the caller never sent keeps its default rather than dropping to zero.
    assert sharpening.normalize({"amount": 30})["radius"] == \
        sharpening.DEFAULT_SHARPEN["radius"]
    # Garbage in one field does not take the rest of the dict with it.
    assert sharpening.normalize({"amount": 40, "detail": "x"})["detail"] == 25


def test_normalize_accepts_the_old_single_slider() -> None:
    """`{"sharpen": 40}` is what is already in the database."""
    assert sharpening.normalize(40) == {**sharpening.DEFAULT_SHARPEN, "amount": 40}
    assert sharpening.normalize(True) is None    # a bool is not an amount


def test_bad_input_fails_loudly() -> None:
    active = {"amount": 50}
    for bad in (np.zeros((8, 8), np.float32), np.zeros((8, 8, 3), np.float64)):
        try:
            sharpening.apply_sharpen(bad, active, 8.0)
        except AssertionError:
            continue
        raise AssertionError(f"accepted a bad array: {bad.shape} {bad.dtype}")


# ----- what the four sliders do -------------------------------------------

def test_amount_direction() -> None:
    """More amount, more high-frequency energy, monotonically."""
    tex = _texture()
    energies = [_band(_run(tex, amount=a), 1.2) for a in (0, 20, 50, 80, 100)]
    assert np.all(np.diff(energies) > 0), f"not monotone in amount: {energies}"
    assert energies[-1] > energies[0] * 1.2


def test_hue_is_preserved() -> None:
    """Luma is sharpened and the delta goes back to all three channels, so a
    pixel's colour is only ever made brighter or darker, never recoloured."""
    rng = np.random.default_rng(2)
    img = np.clip(rng.normal(0.5, 0.12, (96, 96, 3)).astype(np.float32), 0.05, 0.95)
    out = _run(img, amount=100, detail=100)
    delta = out - img
    # The same delta in every channel means channel differences are untouched.
    assert np.abs(delta - delta.mean(axis=2, keepdims=True)).max() < 1e-6
    assert np.abs((out[..., 0] - out[..., 1]) - (img[..., 0] - img[..., 1])).max() < 1e-6


def test_radius_sets_the_spatial_scale() -> None:
    """A small radius is blind to a wide feature; a large one is not. That is the
    whole difference between radius and amount."""
    fine, coarse = _ripple(3.0), _ripple(16.0)

    def gain(img: np.ndarray, radius: int) -> float:
        return _band(_run(img, amount=70, radius=radius, detail=60), 1.2) / _band(img, 1.2)

    assert gain(coarse, 100) > gain(coarse, 0) * 1.1, "a wide radius ignored a wide feature"
    # And the small radius must be genuinely narrow: it barely sees the 16 px
    # grating while still working on the 3 px one.
    assert gain(fine, 0) > gain(coarse, 0) * 1.1
    # Reach grows with radius, so a windowed render needs more of its surround.
    assert sharpening.padding({"amount": 50, "radius": 100}, 4000.0) > \
        sharpening.padding({"amount": 50, "radius": 0}, 4000.0)


def test_detail_changes_the_frequency_mix() -> None:
    """detail=100 must admit more of the finest structure than detail=0 — and do
    it by shifting the mix, not by quietly acting as a second amount slider."""
    tex = _texture()
    fine0, mid0 = _band(tex, 0.9), _band(tex, 0.9, 3.0)

    low = _run(tex, amount=70, detail=0)
    high = _run(tex, amount=70, detail=100)
    fine_lo, mid_lo = _band(low, 0.9) / fine0, _band(low, 0.9, 3.0) / mid0
    fine_hi, mid_hi = _band(high, 0.9) / fine0, _band(high, 0.9, 3.0) / mid0

    assert fine_hi > fine_lo * 1.15, f"detail admitted no fine structure: {fine_lo} -> {fine_hi}"
    # The mix, not the volume: fine gains relative to mid.
    assert fine_hi / mid_hi > (fine_lo / mid_lo) * 1.05, \
        f"detail only scaled the amount: {fine_lo / mid_lo} -> {fine_hi / mid_hi}"


def test_detail_zero_still_sharpens_an_edge() -> None:
    """Low detail is not low amount. It restores a soft edge — that is what
    sharpening is for — and only declines to build texture on top of it."""
    img = _edge()
    def slope(a: np.ndarray) -> float:
        return float(np.abs(np.diff(sharpening._luma(a)[24])).max())
    assert slope(_run(img, amount=70, detail=0)) > slope(img) * 1.1


def test_masking_protects_flat_areas_and_keeps_the_edge() -> None:
    """The mask is the difference between sharpening a photograph and sharpening
    the noise in its sky. Half noise, half a real edge, one array."""
    rng = np.random.default_rng(3)
    flat = np.clip(np.full((128, 128, 3), 0.45, np.float32)
                   + rng.normal(0.0, 0.03, (128, 128, 3)).astype(np.float32), 0.0, 1.0)
    step = np.full((128, 128, 3), 0.35, np.float32)
    step[:, 64:] = 0.8
    comp = np.concatenate([flat, cv2.GaussianBlur(step, (0, 0), 1.2)], axis=1)

    def moved(params: dict) -> tuple[float, float]:
        out = sharpening.apply_sharpen(comp, params, 256.0)
        d = np.abs(sharpening._luma(out) - sharpening._luma(comp))
        return float(d[:, 10:110].mean()), float(d[:, 180:205].max())

    open_flat, open_edge = moved({"amount": 80, "detail": 50, "masking": 0})
    assert open_flat > 0.0, "with no mask the flat half should be sharpened too"

    masked_flat, masked_edge = moved({"amount": 80, "detail": 50, "masking": 100})
    assert masked_flat < open_flat * 0.05, \
        f"masking left the flat area alone? {open_flat} -> {masked_flat}"
    assert masked_edge > open_edge * 0.7, \
        f"masking ate the edge as well: {open_edge} -> {masked_edge}"

    # And it is a ramp, not a switch: half masking protects less than full.
    half_flat, _ = moved({"amount": 80, "detail": 50, "masking": 20})
    assert masked_flat <= half_flat <= open_flat


def test_hard_edge_does_not_halo() -> None:
    """The overshoot bound, measured on the case that actually produces halos: a
    nearly hard step, where a plain unsharp mask throws a fifth of the step's
    height into a bright line beside it. The local-range clamp is what stops it,
    and at detail 0 the bound is exact rather than merely small."""
    img = _edge(0.3, 0.7, blur=0.5)
    row = sharpening._luma(img)[24]
    step = float(row.max() - row.min())

    def overshoot_of(out: np.ndarray) -> float:
        r = sharpening._luma(out)[24]
        return max(float(r.max() - row.max()), float(row.min() - r.min())) / step

    def overshoot(**params) -> float:
        return overshoot_of(_run(img, **params))

    assert overshoot(amount=70, detail=25) < 0.07
    assert overshoot(amount=100, detail=50) < 0.11
    # detail=0 asks for no overshoot at all and gets it, to the last bit.
    assert overshoot(amount=100, detail=0) < 1e-6

    # And it must beat the unclamped unsharp mask it replaces at matched gain —
    # otherwise the whole two-band-plus-clamp construction is not paying for
    # itself. This is editing._apply_sharpen, the one-slider stage.
    plain = img + 1.5 * (img - cv2.GaussianBlur(img, (0, 0), 1.0))
    assert overshoot(amount=100, detail=50) < overshoot_of(plain) * 0.6


def test_output_may_leave_the_unit_range() -> None:
    """The caller clips. A white edge overshooting past 1.0 is real, and swallowing
    it here would make the stage lie about what it did."""
    img = _edge(0.02, 0.99)
    out = _run(img, amount=100, detail=100)
    assert out.max() > 1.0 or out.min() < 0.0


# ----- padding ------------------------------------------------------------

def test_padding_is_zero_only_when_neutral() -> None:
    for neutral in (None, {}, {"amount": 0}, dict(sharpening.DEFAULT_SHARPEN)):
        assert sharpening.padding(neutral, 4000.0) == 0.0, f"asked for padding: {neutral}"
    assert sharpening.padding({"amount": 1}, 4000.0) > 0.0
    # Masking reads a mask through a blur, so it reaches further than the kernel.
    assert sharpening.padding({"amount": 50, "masking": 50}, 4000.0) > \
        sharpening.padding({"amount": 50, "masking": 0}, 4000.0)


def test_padding_is_in_pixels_not_frame_fractions() -> None:
    """The deliberate exception, the same one denoise makes: this stage is
    measured in pixels, so its padding must not grow with the file."""
    active = {"amount": 60, "radius": 70, "masking": 40}
    assert sharpening.padding(active, 1000.0) == sharpening.padding(active, 24000.0)


def test_padding_covers_the_real_reach() -> None:
    """Sharpening one window of a frame must match sharpening the whole frame and
    cutting the window out — otherwise a 1:1 tile shows a seam at its border."""
    img = _texture(edge=192)
    params = {"amount": 90, "radius": 100, "detail": 60, "masking": 30}
    pad = int(np.ceil(sharpening.padding(params, 192.0)))
    assert pad < 96, "padding is too greedy to be worth having"

    whole = sharpening.apply_sharpen(img, params, 192.0)[64:128, 64:128]
    padded = sharpening.apply_sharpen(
        np.ascontiguousarray(img[64 - pad:128 + pad, 64 - pad:128 + pad]), params, 192.0)
    window = padded[pad:pad + 64, pad:pad + 64]
    assert np.abs(whole - window).max() < 1e-4, \
        f"window drifted from the full render by {np.abs(whole - window).max()}"


if __name__ == "__main__":
    import sys
    mod = sys.modules[__name__]
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        getattr(mod, name)()
        print(f"ok  {name}")
