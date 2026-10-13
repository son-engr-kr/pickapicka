"""The looks that ship with the app (looks.py).

    uv run pytest tests/test_looks.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from pickapicka import editing, looks, presets
from pickapicka.scorer import run_scoring
from pickapicka.server import OpenProjectPayload, PresetApplyPayload, create_app


def _photo() -> np.ndarray:
    """A sky-to-ground ramp with colour in every band, small."""
    rng = np.random.default_rng(0)
    y = np.linspace(0, 1, 48)[:, None, None]
    hue = np.linspace(0, 1, 64)[None, :, None]
    rgb = np.concatenate([0.5 + 0.4 * np.sin(2 * np.pi * (hue + k / 3)) * (1 - 0.5 * y) for k in range(3)], axis=2)
    rgb = np.clip(rgb * (0.4 + 0.6 * (1 - y)) + rng.normal(0, 0.01, rgb.shape), 0, 1)
    return (rgb * 255).astype(np.uint8)


def test_twenty_looks_in_six_groups() -> None:
    assert len(looks.LOOKS) == 20
    assert {lk["group"] for lk in looks.LOOKS} == set(looks.GROUPS)
    for lk in looks.LOOKS:
        assert lk["builtin"] and lk["id"].startswith("look-") and lk["hint"]
        assert lk["parts"] and set(lk["parts"]) <= set(presets.PART_NAMES)
        assert presets.KEEP_LIGHT in lk["parts"] and "light" not in lk["parts"]


def test_a_look_leaves_the_photos_own_corrections_alone() -> None:
    """Exposure is never a look's; white balance only a black and white one's,
    and sharpening only the looks that carry Detail & effects."""
    base = editing.normalize({"pv": editing.PROCESS_VERSION, "exposure": 0.7, "contrast": 12,
                              "temp": 20, "tint": -5, "sharpen": 30, "crop": {"x": 0.1, "y": 0, "w": 0.8, "h": 1}})
    for lk in looks.LOOKS:
        out = presets.apply(base, lk)
        assert (out["exposure"], out["contrast"], out["crop"]) == (0.7, 12, base["crop"]), lk["name"]
        mono = lk["group"] == "Black & white"
        assert (out["temp"], out["tint"]) == ((0, 0) if mono else (20, -5)), lk["name"]
        assert out["sharpen"] == (lk["edit"]["sharpen"] if "detail" in lk["parts"] else 30), lk["name"]


def test_every_look_renders_and_changes_the_photo() -> None:
    rgb = _photo()
    for lk in looks.LOOKS:
        out = editing.render(rgb, presets.apply(None, lk))
        assert out.shape == rgb.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - rgb).mean() > 2, f"{lk['name']} hardly changes anything"


def test_the_black_and_white_looks_leave_no_colour() -> None:
    rgb = _photo()
    for lk in looks.LOOKS:
        if lk["group"] != "Black & white":
            continue
        out = editing.render(rgb, presets.apply(None, lk)).astype(int)
        assert (out.max(axis=2) - out.min(axis=2)).max() <= 1, lk["name"]


def _app(tmp_path: Path):
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    photos.mkdir()
    Image.fromarray(_photo()).save(photos / "a.jpg")
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    app = create_app()
    route = lambda path, method: next(r.endpoint for r in app.routes
                                      if getattr(r, "path", None) == path and method in r.methods)
    route("/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while route("/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return route


def test_the_looks_are_listed_after_your_own_and_cannot_be_deleted(tmp_path) -> None:
    route = _app(tmp_path)
    from pickapicka.server import PresetSavePayload
    route("/api/presets", "POST")(PresetSavePayload(name="mine", edit={"vibrance": 20}, parts=["color"]))
    listed = route("/api/presets", "GET")()["presets"]
    assert listed[0]["name"] == "mine" and not listed[0].get("builtin")
    assert [p["id"] for p in listed[1:]] == [lk["id"] for lk in looks.LOOKS]
    with pytest.raises(HTTPException) as exc:
        route("/api/presets/{preset_id}", "DELETE")("look-noir")
    assert exc.value.status_code == 400

    applied = route("/api/presets/apply", "POST")(PresetApplyPayload(
        rel_path="a.jpg", edit={"exposure": 0.5}, preset_id="look-cinematic", amount=50))
    assert applied["edit"]["exposure"] == 0.5 and applied["edit"]["grading"] is not None


# ----- Keep brightness ----------------------------------------------------

def _scene(median: float, white: float = 0.0) -> np.ndarray:
    """A textured frame whose encoded luma has about this median, with
    `white` of it near white."""
    rng = np.random.default_rng(1)
    base = np.clip(rng.lognormal(np.log(median), 0.5, (60, 80)), 0, 1)
    rgb = np.stack([base * 1.05, base, base * 0.9], axis=2)
    if white:
        rgb[: int(60 * white)] = 0.97
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _median(rgb: np.ndarray) -> float:
    return float(np.median((rgb.astype(np.float32) / 255) @ np.array([0.2126, 0.7152, 0.0722])))


def _kept(rgb: np.ndarray, base: dict | None, lk: dict) -> dict:
    return presets.keep_light(base, presets.apply(base, lk), lambda e: editing.render(rgb, e))


@pytest.mark.parametrize("median", [0.06, 0.2, 0.45])
def test_keep_brightness_puts_the_midtone_back_whatever_it_was(median) -> None:
    """A night frame, an unedited RAW and a frame after Auto: each keeps its own."""
    rgb = _scene(median)
    before = _median(editing.render(rgb, None))
    for key in ("seventies", "heritage", "noir", "pastel"):
        lk = next(x for x in looks.LOOKS if x["id"] == f"look-{key}")
        loose = _median(editing.render(rgb, presets.apply(None, lk)))
        kept = _median(editing.render(rgb, _kept(rgb, None, lk)))
        assert abs(kept - before) <= max(0.012, 0.25 * abs(loose - before)), (key, median, before, loose, kept)


def test_keep_brightness_does_not_blow_out_what_the_look_kept() -> None:
    """A darkening look on a frame with a bright sky: bringing the midtone back
    must not push the sky into white."""
    rgb = _scene(0.3, white=0.25)
    dark = presets.make({"pv": editing.PROCESS_VERSION,
                         "curve": [[0, 0], [0.5, 0.3], [0.97, 0.9], [1, 0.92]]}, ["curve"])
    dark = {**dark, "id": "dark", "parts": dark["parts"] + [presets.KEEP_LIGHT]}
    look_only = editing.render(rgb, presets.apply(None, dark))
    kept = editing.render(rgb, _kept(rgb, None, dark))
    blown = lambda a: float(((a.astype(np.float32) / 255) @ np.array([0.2126, 0.7152, 0.0722]) >= 0.985).mean())
    assert blown(kept) <= blown(look_only) + 0.004
    assert _median(kept) > _median(look_only)        # it still came back as far as it could


def test_keep_brightness_leaves_a_preset_that_moves_nothing() -> None:
    rgb = _scene(0.3)
    same = {"id": "x", "parts": ["mixer", presets.KEEP_LIGHT],
            "edit": {"pv": editing.PROCESS_VERSION, "hsl": {"purple": {"hue": 10}}}}
    out = _kept(rgb, None, same)
    assert out["curve"] == editing.DEFAULT_EDIT["curve"]


def test_the_server_keeps_an_unedited_photos_brightness(tmp_path) -> None:
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    photos.mkdir()
    rgb = _scene(0.2)
    Image.fromarray(rgb).save(photos / "a.jpg", quality=98)
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    app = create_app()
    route = lambda path, method: next(r.endpoint for r in app.routes
                                      if getattr(r, "path", None) == path and method in r.methods)
    route("/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while route("/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    out = route("/api/presets/apply", "POST")(PresetApplyPayload(rel_path="a.jpg", preset_id="look-seventies"))
    assert out["edit"]["exposure"] == 0
    with Image.open(photos / "a.jpg") as im:
        a = np.asarray(im.convert("RGB"))
    assert abs(_median(editing.render(a, out["edit"])) - _median(a)) < 0.015
