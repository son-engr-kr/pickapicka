"""Checks for the film chain.

    uv run python tests/test_film.py

These are mostly properties rather than values, because the numbers are a matter
of taste but the properties are what separate a film chain from a filter:

  - the response is the identity when it is not asked for, so "neutral" is
    neutral and the stage can live in the pipeline permanently;
  - the response reshapes contrast about mid grey rather than lifting the whole
    frame (the first version got this wrong: measuring log exposure from an
    epsilon put mid grey at 0.94 and made every stock a brightening filter);
  - halation bleeds warm, because the red-sensitive layer sits deepest;
  - grain is correlated, strongest in the mid-densities, the same size whatever
    resolution it is rendered at, and the same grain every time — otherwise the
    thumbnail, the fit preview, the 1:1 view and the export each show a different
    photograph.
"""
from __future__ import annotations

import numpy as np

from pickapicka import editing, film


# ----- schema -------------------------------------------------------------

def test_off_is_neutral() -> None:
    assert film.is_neutral(None)
    assert film.is_neutral({})
    assert film.is_neutral({"enabled": False, "grain": 90})
    assert film.is_neutral({"enabled": True, "strength": 0, "grain": 90})
    # Every stage at zero: nothing to do, whatever the switch says.
    assert film.is_neutral({"enabled": True, "contrast": 0, "crosstalk": 0,
                            "warmth": 0, "split": 0, "halation": 0, "grain": 0})
    assert not film.is_neutral({"enabled": True, "grain": 5})


def test_normalize_clamps_and_drops_a_no_op() -> None:
    f = film.normalize({"enabled": True, "grain": 999, "warmth": -999,
                        "contrast": "x", "stock": "a" * 200})
    assert f["grain"] == 100 and f["warmth"] == -100
    assert f["contrast"] == film.DEFAULT_FILM["contrast"], "bad value keeps the default"
    assert len(f["stock"]) == 40
    assert film.normalize({"enabled": True, "contrast": 0, "crosstalk": 0,
                           "warmth": 0, "split": 0, "halation": 0,
                           "grain": 0}) is None
    assert film.normalize("nonsense") is None


def test_every_stock_is_usable_and_distinct() -> None:
    seen = set()
    for name in film.STOCKS:
        st = film.stock(name)
        assert st is not None and st["enabled"] and st["stock"] == name
        key = tuple(sorted((k, v) for k, v in st.items() if k != "stock"))
        assert key not in seen, f"{name} is a duplicate of another stock"
        seen.add(key)
    assert film.stock("no such stock") is None


def test_an_edit_carries_film() -> None:
    st = film.stock("Punchy slide")
    assert not editing.is_neutral({"film": st})
    assert editing.is_neutral({"film": {"enabled": False}})
    assert editing.normalize({"film": {"enabled": False}})["film"] is None
    # The hash has to notice, or a cached thumbnail would outlive the change.
    a = editing.edit_hash({"film": film.stock("Punchy slide")})
    b = editing.edit_hash({"film": film.stock("Warm portrait")})
    assert a and b and a != b


# ----- the response curve -------------------------------------------------

def test_no_curve_is_exactly_the_identity() -> None:
    """The stage sits in the pipeline permanently, so it has to cost nothing when
    it is not wanted."""
    c = film._density_curve({**film.DEFAULT_FILM, "contrast": 0})
    x = np.linspace(0.0, 1.0, len(c))
    assert np.abs(c - x).max() < 1e-6


def test_the_curve_is_monotone_and_spans_the_range() -> None:
    for name in film.STOCKS:
        c = film._density_curve({**film.DEFAULT_FILM, **film.STOCKS[name]})
        assert np.all(np.diff(c) >= -1e-6), f"{name} inverts somewhere"
        assert c[0] < 0.12 and c[-1] > 0.95, f"{name} does not span the range"


def test_the_curve_reshapes_about_mid_grey() -> None:
    """Not a brightening filter. Mid grey has to stay near mid grey, or every
    stock just lifts the frame — which is exactly what measuring log exposure
    from an epsilon did, putting mid grey at 0.94."""
    for name in film.STOCKS:
        c = film._density_curve({**film.DEFAULT_FILM, **film.STOCKS[name]})
        mid = float(np.interp(0.5, np.linspace(0.0, 1.0, len(c)), c))
        assert abs(mid - 0.5) < 0.06, f"{name} moved mid grey to {mid:.3f}"


def test_more_contrast_steepens_the_middle() -> None:
    soft = film._density_curve({**film.DEFAULT_FILM, "contrast": 10,
                                "toe": 20, "shoulder": 20})
    hard = film._density_curve({**film.DEFAULT_FILM, "contrast": 90,
                                "toe": 20, "shoulder": 20})
    # Slope across the middle third.
    lo, hi = len(soft) // 3, 2 * len(soft) // 3
    assert (hard[hi] - hard[lo]) > (soft[hi] - soft[lo])


def test_crosstalk_matrix_stays_neutral_overall() -> None:
    """A matrix whose rows do not sum to one gives the whole frame a cast that
    the white balance then has to fight."""
    for c in (0, 25, 60, 100):
        for sp in (-100, 0, 100):
            m = film._crosstalk_matrix({**film.DEFAULT_FILM,
                                        "crosstalk": c, "split": sp})
            assert np.allclose(m.sum(axis=1), 1.0, atol=1e-6)


def test_crosstalk_mixes_the_channels() -> None:
    off = film._crosstalk_matrix({**film.DEFAULT_FILM, "crosstalk": 0})
    on = film._crosstalk_matrix({**film.DEFAULT_FILM, "crosstalk": 80})
    assert np.allclose(off, np.eye(3), atol=1e-6)
    assert not np.allclose(on, np.eye(3), atol=1e-3)


# ----- halation -----------------------------------------------------------

def test_halation_bleeds_warm() -> None:
    """The red layer sits deepest in the emulsion, so the scatter is reddest.
    That warm fringe is the recognisable part."""
    img = np.zeros((400, 400, 3), np.uint8)
    img[150:250, 150:250] = 255
    f = {**film.DEFAULT_FILM, "enabled": True, "contrast": 0, "crosstalk": 0,
         "grain": 0, "halation": 90, "halation_radius": 45}
    out = editing.render(img, {"film": f}, meta={"file": "h"}).astype(int)
    ring = out[200, 251:262].mean(axis=0)          # just outside the square
    assert ring[0] > ring[1] > ring[2], f"not warm: {ring}"
    assert ring[0] > 8, f"too faint to see: {ring}"


def test_halation_scales_with_the_frame() -> None:
    """Sized against the frame, so the preview is the export."""
    def spread(n: int) -> int:
        img = np.zeros((n, n, 3), np.uint8)
        q = n // 4
        img[q:3 * q, q:3 * q] = 255
        f = {**film.DEFAULT_FILM, "enabled": True, "contrast": 0, "crosstalk": 0,
             "grain": 0, "halation": 90, "halation_radius": 60}
        out = editing.render(img, {"film": f}, meta={"file": "h"})
        row = out[n // 2, 3 * q:, 0].astype(int)   # to the right of the square
        lit = np.nonzero(row > 3)[0]
        return int(lit.max()) if len(lit) else 0
    small, large = spread(200), spread(400)
    assert large > small * 1.5, f"did not scale: {small} vs {large}"


def test_halation_needs_neighbours() -> None:
    """It is a blur, so a window cut exactly to size would show a seam."""
    st = film.stock("Push-processed")
    assert film.padding(st, 4000.0) > 10.0
    assert film.padding(None, 4000.0) == 0.0
    assert editing.effect_padding({"film": st}, 4000.0) > 10.0


# ----- grain --------------------------------------------------------------

_GRAIN = {**film.DEFAULT_FILM, "enabled": True, "contrast": 0, "crosstalk": 0,
          "warmth": 0, "split": 0, "halation": 0,
          "grain": 70, "grain_size": 40, "grain_rough": 55}


def _grain_only(h: int, w: int, seed_key: str = "a") -> np.ndarray:
    flat = np.full((h, w, 3), 128, np.uint8)
    out = editing.render(flat, {"film": _GRAIN}, meta={"file": seed_key})
    return out[..., 1].astype(np.float32) - 128.0


def test_grain_is_correlated_not_white_noise() -> None:
    n = _grain_only(600, 800)
    n = (n - n.mean()) / (n.std() + 1e-9)
    lag1 = float((n[:, :-1] * n[:, 1:]).mean())
    lag16 = float((n[:, :-16] * n[:, 16:]).mean())
    assert lag1 > 0.25, f"white noise, not grain: lag-1 correlation {lag1:.3f}"
    assert lag16 < 0.15, f"correlated over far too long a distance: {lag16:.3f}"


def test_grain_is_strongest_in_the_mid_densities() -> None:
    """Clear film has no crystals to see and solid black has no gaps between
    them; the visible grain lives in between."""
    ramp = np.repeat(np.tile(np.linspace(0, 255, 800, dtype=np.uint8),
                             (400, 1))[..., None], 3, axis=2)
    with_g = editing.render(ramp, {"film": _GRAIN}, meta={"file": "a"}).astype(np.float32)
    without = editing.render(ramp, {"film": {**_GRAIN, "grain": 0}},
                             meta={"file": "a"}).astype(np.float32)
    res = (with_g - without)[..., 1]
    bands = [float(res[:, i * 100:(i + 1) * 100].std()) for i in range(8)]
    assert max(bands[2:6]) > max(bands[0], bands[7]) * 1.4, bands


def test_grain_size_does_not_follow_the_render_resolution() -> None:
    """The property the whole lattice design exists for. Deriving a cell size in
    render pixels and clamping it — the first attempt — made a thumbnail's grain
    relatively coarser than a 1:1 view's."""
    lengths = []
    # Measured where the grain is actually resolvable. Below roughly two output
    # pixels per crystal the correlation lag cannot shrink any further — the
    # measurement floors out, not the model — so testing at 400 px would be
    # checking the ruler rather than the thing.
    for scale in (1.0, 2.0, 3.0):
        h, w = int(600 * scale), int(800 * scale)
        n = _grain_only(h, w)
        n = (n - n.mean()) / (n.std() + 1e-9)
        lag = next((d for d in range(1, 80)
                    if float((n[:, :-d] * n[:, d:]).mean()) < 0.5), 80)
        lengths.append(lag / w)          # as a fraction of the frame
    assert max(lengths) / min(lengths) < 1.35, lengths


def _grain_stats(f: dict, w: int, h: int) -> tuple[float, float]:
    """Amplitude in levels, and structure size as a fraction of the frame."""
    flat = np.full((h, w, 3), 128, np.uint8)
    n = editing.render(flat, {"film": f}, meta={"file": "a"})[..., 1]
    n = n.astype(np.float32) - 128.0
    sigma = float(n.std())
    nn = (n - n.mean()) / (n.std() + 1e-9)
    lag = next((d for d in range(1, 90)
                if float((nn[:, :-d] * nn[:, d:]).mean()) < 0.5), 90)
    return sigma, lag / w


def test_finer_than_a_pixel_is_averaged_rather_than_aliased() -> None:
    """Crystals smaller than an output pixel are the natural case, not an error: a
    35 mm negative viewed at 1280 px has sub-pixel grain, which is why the finest
    setting was the only one that read as film. Point-sampling the lattice there
    aliases, which is what forced the range to stop at 2.5 px per crystal.
    Area-averaged instead, fine grain simply gets quieter in a small render — what
    a small print does — and no resolution shows structure the export lacks."""
    fine = [_grain_stats({**_GRAIN, "grain_size": 0}, int(800 * s), int(600 * s))[0]
            for s in (1.0, 2.0, 3.0)]
    assert fine == sorted(fine), f"not monotone with resolution: {fine}"
    assert fine[-1] > fine[0] * 1.15, f"nothing is being averaged: {fine}"
    # A coarse setting has nothing to average away, so it must not be attenuated.
    coarse = [_grain_stats({**_GRAIN, "grain_size": 100},
                           int(800 * s), int(600 * s))[0] for s in (1.0, 3.0)]
    assert abs(coarse[0] - coarse[1]) < 0.6, f"coarse grain attenuated: {coarse}"


def test_the_size_range_reaches_a_natural_scale() -> None:
    """What prompted the rework: the old range began at 2.5 px per crystal, already
    coarser than a real scan at a normal viewing size, so the only natural-looking
    setting was pinned at one end of the slider. About a pixel per crystal has to
    fall somewhere in the middle."""
    def px_per_crystal(sz: int) -> float:
        cells = film._GRAIN_FINE - (sz / 100.0) * (film._GRAIN_FINE - film._GRAIN_COARSE)
        return 1280.0 / cells
    assert px_per_crystal(0) < 0.7
    assert 0.7 < px_per_crystal(50) < 1.6, \
        f"a pixel per crystal is not mid-slider: {px_per_crystal(50):.2f}"
    assert px_per_crystal(100) > 3.0


def test_a_window_of_fine_grain_still_matches_the_export() -> None:
    """The averaging factor has to come from the frame. Derived from the window it
    differs between a full render and a crop of it, the averaging lands on
    different phases, and the 1:1 view stops being the export — three levels out
    when this was wrong."""
    src = np.random.default_rng(11).integers(0, 256, (600, 800, 3)).astype(np.uint8)
    f = {**_GRAIN, "grain_size": 0}
    full = editing.render(src, {"film": f}, meta={"file": "w"})
    px0, py0, pw, ph = 200, 180, 320, 210
    pad = int(round(editing.effect_padding({"film": f}, 800))) + 2
    ax0, ay0 = max(0, px0 - pad), max(0, py0 - pad)
    ax1, ay1 = min(800, px0 + pw + pad), min(600, py0 + ph + pad)
    win = editing.render(src[ay0:ay1, ax0:ax1], {"film": f},
                         roi=(ax0 / 800, ay0 / 600,
                              (ax1 - ax0) / 800, (ay1 - ay0) / 600),
                         meta={"file": "w"})
    got = win[py0 - ay0:py0 - ay0 + ph, px0 - ax0:px0 - ax0 + pw].astype(int)
    want = full[py0:py0 + ph, px0:px0 + pw].astype(int)
    assert np.abs(got - want).max() == 0, "fine grain drifted in a window"


def test_grain_is_the_same_grain_every_time() -> None:
    a = _grain_only(300, 400, "photo-one")
    b = _grain_only(300, 400, "photo-one")
    c = _grain_only(300, 400, "photo-two")
    assert np.array_equal(a, b), "the grain crawls between renders"
    assert not np.array_equal(a, c), "every photo got the same grain"


# ----- the chain in the pipeline -----------------------------------------

def test_a_window_matches_the_same_part_of_the_full_render() -> None:
    """What makes the 1:1 view trustworthy: the grain and the halation in a
    zoomed window have to be the ones the export will have, in the same place."""
    src = np.random.default_rng(3).integers(0, 256, (600, 800, 3)).astype(np.uint8)
    st = film.stock("Push-processed")
    full = editing.render(src, {"film": st}, meta={"file": "p"})
    px0, py0, pw, ph = 200, 180, 320, 210
    pad = int(round(editing.effect_padding({"film": st}, 800))) + 2
    ax0, ay0 = max(0, px0 - pad), max(0, py0 - pad)
    ax1, ay1 = min(800, px0 + pw + pad), min(600, py0 + ph + pad)
    win = editing.render(src[ay0:ay1, ax0:ax1], {"film": st},
                         roi=(ax0 / 800, ay0 / 600,
                              (ax1 - ax0) / 800, (ay1 - ay0) / 600),
                         meta={"file": "p"})
    got = win[py0 - ay0:py0 - ay0 + ph, px0 - ax0:px0 - ax0 + pw].astype(int)
    want = full[py0:py0 + ph, px0:px0 + pw].astype(int)
    d = np.abs(got - want)
    # Grain and the response curve are exact — checked separately below. Halation
    # blurs at reduced resolution, and the same gaussian over two differently
    # sized windows cannot land bit-identically, so one level on a stray pixel is
    # the honest bound. Anything more would be a placement error, not rounding.
    assert d.max() <= 1, f"max {d.max()}"
    assert d.mean() < 0.001, f"mean {d.mean()}"


def test_grain_and_the_curve_are_exact_in_a_window() -> None:
    """The two stages that have no excuse: no resampling, so a window has to be
    the full render's own pixels."""
    src = np.random.default_rng(4).integers(0, 256, (400, 600, 3)).astype(np.uint8)
    px0, py0, pw, ph = 150, 120, 240, 160
    for label, f in (
        ("grain", {**film.DEFAULT_FILM, "enabled": True, "contrast": 0,
                   "crosstalk": 0, "halation": 0, "grain": 80}),
        ("curve", {**film.DEFAULT_FILM, "enabled": True, "halation": 0,
                   "grain": 0}),
    ):
        full = editing.render(src, {"film": f}, meta={"file": "q"})
        pad = int(round(editing.effect_padding({"film": f}, 600))) + 2
        ax0, ay0 = max(0, px0 - pad), max(0, py0 - pad)
        ax1, ay1 = min(600, px0 + pw + pad), min(400, py0 + ph + pad)
        win = editing.render(src[ay0:ay1, ax0:ax1], {"film": f},
                             roi=(ax0 / 600, ay0 / 400,
                                  (ax1 - ax0) / 600, (ay1 - ay0) / 400),
                             meta={"file": "q"})
        got = win[py0 - ay0:py0 - ay0 + ph, px0 - ax0:px0 - ax0 + pw].astype(int)
        want = full[py0:py0 + ph, px0:px0 + pw].astype(int)
        assert np.abs(got - want).max() == 0, f"{label} drifted in a window"


def test_strength_fades_the_whole_chain() -> None:
    src = np.random.default_rng(5).integers(0, 256, (200, 300, 3)).astype(np.uint8)
    st = film.stock("Push-processed")
    full = editing.render(src, {"film": st}, meta={"file": "s"}).astype(int)
    half = editing.render(src, {"film": {**st, "strength": 50}},
                          meta={"file": "s"}).astype(int)
    none = editing.render(src, {"film": {**st, "strength": 0}}, meta={"file": "s"})
    d_full = np.abs(full - src.astype(int)).mean()
    d_half = np.abs(half - src.astype(int)).mean()
    assert d_half < d_full * 0.75, f"{d_half:.2f} vs {d_full:.2f}"
    assert np.array_equal(none, src), "strength 0 should be untouched"


def test_a_preset_can_carry_a_stock() -> None:
    base = {"exposure": 0.4}
    merged = editing.merge_additive(base, {"film": film.stock("Warm portrait")})
    assert merged["film"]["stock"] == "Warm portrait"
    assert merged["exposure"] == 0.4, "the film must not flatten the grade"
    kept = editing.merge_additive({"film": film.stock("Punchy slide")},
                                  {"contrast": 20})
    assert kept["film"]["stock"] == "Punchy slide"


# ----- the quick paths agree with the formulas they replaced ---------------

def _speculars(h: int = 600, w: int = 900) -> np.ndarray:
    """Mid-grey with bright highlights of several sizes: what halation is for."""
    rng = np.random.default_rng(3)
    img = np.clip(0.35 + 0.05 * rng.standard_normal((h, w, 3)), 0, 1).astype(np.float32)
    for (y, x, r) in ((100, 150, 6), (300, 450, 30), (450, 700, 70), (150, 750, 2)):
        yy, xx = np.ogrid[:h, :w]
        img[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 0.98
    return img


def test_crosstalk_is_the_matrix_product() -> None:
    f = {**film.DEFAULT_FILM, **film.stock("Warm portrait"), "warmth": 0}
    img = _speculars()
    got = film._apply_tone(img, f)
    curve = film._density_curve(f)
    table = np.repeat((curve * 255.0).astype(np.uint8).reshape(256, 1), 3, axis=1)
    import cv2
    dens = cv2.LUT(cv2.convertScaleAbs(img, alpha=255.0), table.reshape(256, 1, 3))
    want = np.clip((dens.astype(np.float32) / 255.0) @ film._crosstalk_matrix(f).T, 0.0, 1.0)
    assert np.abs(got - want).max() < 1e-6


def test_the_capped_halation_blur_stays_within_two_levels() -> None:
    """Wide halation blurs a reduced copy of the highlight mask. Against the
    blur at full size the finished frame may move by 2 levels out of 255."""
    import cv2
    img = _speculars()
    for radius_param in (25, 60, 100):
        f = {**film.DEFAULT_FILM, "enabled": True, "halation": 100,
             "halation_radius": radius_param}
        frame_long = float(max(img.shape[:2]))
        got = film._apply_halation(img, f, frame_long)
        radius = max(1.0, (radius_param / 100.0) * 0.035 * frame_long)
        lum = img @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        hot = np.clip((lum - 0.80) / 0.20, 0.0, 1.0) ** 2.0
        spread = cv2.GaussianBlur(hot, (0, 0), radius)
        want = img + spread[..., None] * film._HALATION_WEIGHT * 1.9 * (1.0 - img)
        to8 = lambda a: np.rint(np.clip(a, 0, 1) * 255).astype(int)
        d = np.abs(to8(got) - to8(want))
        assert d.max() <= 2, (radius_param, d.max())
        assert d.mean() < 0.1, (radius_param, d.mean())


def test_the_grain_cache_hands_back_what_it_would_build() -> None:
    f = {**film.DEFAULT_FILM, **_GRAIN}
    args = ((240, 320), (0.1, 0.2, 0.5, 0.5), 640.0, 480.0, 11)
    first = film._grain_field(f, *args)
    again = film._grain_field(f, *args)
    assert again is first
    assert np.array_equal(first, film._build_grain_field(f, *args))
    assert not first.flags.writeable
    other = film._grain_field(f, *args[:-1], 12)
    assert not np.array_equal(other, first)
    coarser = film._grain_field({**f, "grain_size": 80}, *args)
    assert not np.array_equal(coarser, first)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
