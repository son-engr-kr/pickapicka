"""Process versions, and what version 2 changed.

An edit saved before process version 2 carries no `pv` and renders exactly as
it was made. Version 2 puts exposure in stops of light and makes the colour
mixer's luminance multiply the light by chroma, so that it stops lifting dark,
nearly grey shadows (Blue +10 used to lift a blue-grey shadow three times as
far as the sky).

    uv run pytest tests/test_process_version.py
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from pickapicka import editing

SHADOW = (30, 34, 45)     # a dark, nearly grey, bluish shadow
MID_SKY = (95, 145, 215)  # the calibration colour of `_HSL_LUM_EV`
GREY = (128, 128, 128)


def _px(rgb) -> np.ndarray:
    return (np.array(rgb, np.float32) / 255.0).reshape(1, 1, 3)


def _lab(px: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0]


def _stops(before: np.ndarray, after: np.ndarray) -> float:
    w = np.array([0.2126, 0.7152, 0.0722], np.float32)
    lin = lambda p: float((editing._srgb_to_linear(p[0, 0]) * w).sum())
    return float(np.log2(lin(after) / lin(before)))


def _encoded_at(edit: dict, value: float) -> float:
    lut = editing._tone_lut(editing.normalize(edit))
    return float(np.interp(value, np.linspace(0.0, 1.0, editing._LUT_N), lut))


def test_an_old_edit_keeps_process_1_and_its_hash() -> None:
    old = {"exposure": 0.5, "contrast": 10}
    e = editing.normalize(old)
    assert "pv" not in e and editing.process_version(e) == 1
    assert editing.edit_hash(old) != editing.edit_hash({**old, "pv": 2})
    # Process 1's exposure: the encoded value doubled per unit.
    assert _encoded_at({"exposure": 0.5}, 0.4) == pytest.approx(0.4 * 2 ** 0.5, abs=1e-3)


def test_an_unknown_process_version_is_refused() -> None:
    with pytest.raises(AssertionError):
        editing.normalize({"pv": 3, "exposure": 1})


def test_process_2_exposure_is_in_stops() -> None:
    grey = float(editing._linear_to_srgb(np.float32(0.18)))
    for ev in (1.0, -1.0, 0.5):
        got = float(editing._srgb_to_linear(np.float32(_encoded_at({"pv": 2, "exposure": ev}, grey))))
        assert got == pytest.approx(0.18 * 2 ** ev, rel=2e-3)


def test_process_2_luminance_leaves_shadows_and_greys_alone() -> None:
    for pv, most in ((1, None), (2, 0.5)):
        px = _px(SHADOW)
        out = editing._apply_hsl(px.copy(), {"blue": {"lum": 10}}, pv)
        de = float(np.linalg.norm(_lab(out) - _lab(px)))
        if most is None:
            assert de > 5.0, "process 1 changed; old edits would no longer render as made"
        else:
            assert de < most, f"Blue +10 still moves a blue-grey shadow by ΔE {de:.2f}"
    grey = _px(GREY)
    assert np.allclose(editing._apply_hsl(grey.copy(), {"blue": {"lum": 100}}, 2), grey, atol=1e-6)


def test_process_2_luminance_is_calibrated_on_a_mid_sky() -> None:
    px = _px(MID_SKY)
    up = _stops(px, editing._apply_hsl(px.copy(), {"blue": {"lum": 100}}, 2))
    down = _stops(px, editing._apply_hsl(px.copy(), {"blue": {"lum": -100}}, 2))
    # About process 1's +100 (+0.91 stops) either way; see `_HSL_LUM_EV`.
    assert (abs(up) + abs(down)) / 2 == pytest.approx(0.91, abs=0.03)
    assert up > 0 > down


def test_upgrade_keeps_mid_grey_and_converts_masks() -> None:
    old = {"exposure": 0.5, "contrast": 12,
           "masks": [{"type": "radial", "adj": {"exposure": -0.5}}]}
    new = editing.upgrade(old)
    assert new["pv"] == 2 and new["contrast"] == 12
    grey = float(editing._linear_to_srgb(np.float32(0.18)))
    assert _encoded_at(new, grey) == pytest.approx(_encoded_at(old, grey), abs=1 / 255)
    assert new["exposure"] == pytest.approx(1.09, abs=0.01)
    mask_new = {"pv": 2, "exposure": new["masks"][0]["adj"]["exposure"]}
    assert _encoded_at(mask_new, grey) == pytest.approx(_encoded_at({"exposure": -0.5}, grey), abs=1 / 255)
    assert editing.upgrade(new) == new


def test_auto_tone_lands_the_median_where_process_1_did() -> None:
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(70, 25, (120, 160, 3)), 0, 255).astype(np.uint8)
    auto = editing.auto_tone(img)
    assert auto["pv"] == 2
    y = editing._luma(img.astype(np.float32) / 255.0)
    p50 = float(np.percentile(y, 50))
    old_ev = float(np.clip(np.log2(0.45 / p50), -1.5, 1.5))
    assert _encoded_at({"pv": 2, "exposure": auto["exposure"]}, p50) == pytest.approx(
        _encoded_at({"exposure": old_ev}, p50), abs=1 / 255)


def test_sliders_hold_tenths_and_whole_values_stay_ints() -> None:
    e = editing.normalize({"contrast": 12.34, "shadows": 7.0, "hsl": {"blue": {"lum": -3.25}},
                           "masks": [{"type": "radial", "feather": 33.33, "adj": {"clarity": 4.44}}]})
    assert e["contrast"] == 12.3 and e["shadows"] == 7 and type(e["shadows"]) is int
    assert e["hsl"]["blue"]["lum"] == pytest.approx(-3.2, abs=0.051)
    assert e["masks"][0]["feather"] == 33.3 and e["masks"][0]["adj"]["clarity"] == 4.4


def test_a_preset_on_a_bare_photo_brings_its_process_version() -> None:
    preset = {"pv": 2, "exposure": 0.3}
    assert editing.merge_additive(None, preset)["pv"] == 2
    assert "pv" not in editing.merge_additive({"pv": 2}, {"exposure": 0.3}), \
        "an old preset on a bare photo is still the maths it was made with"
    old_photo = {"exposure": 0.2, "contrast": 5}
    assert "pv" not in editing.merge_additive(old_photo, preset)


def test_a_mask_grades_at_its_edits_process_version() -> None:
    grey = float(editing._linear_to_srgb(np.float32(0.18)))
    img = np.full((40, 60, 3), round(grey * 255), np.uint8)
    mask = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 3.0, "ry": 3.0, "feather": 0,
            "adj": {"exposure": 1.0}}
    for pv, want in ((2, 0.36), (1, None)):
        edit = {"masks": [mask], **({"pv": 2} if pv == 2 else {})}
        got = editing.render(img, edit)[20, 30, 1] / 255.0
        if want is None:   # process 1: the encoded value doubled
            assert got == pytest.approx(min(1.0, round(grey * 255) / 255.0 * 2), abs=2 / 255)
        else:              # process 2: the light doubled
            assert float(editing._srgb_to_linear(np.float32(got))) == pytest.approx(want, rel=0.03)
