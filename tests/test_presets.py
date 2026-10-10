"""Checks for presets: what one stores, and what applying one does to an edit
that is already there. Runnable with pytest or directly:

    uv run pytest tests/test_presets.py
"""
from __future__ import annotations

import numpy as np
import pytest

from pickapicka import editing, presets

_LOOK = {
    "pv": editing.PROCESS_VERSION,         # made in today's editor
    "exposure": 0.5, "contrast": 0, "clarity": 30, "vignette": -20, "motion_angle": 40,
    "hsl": {"red": {"sat": 20}, "blue": {"hue": -10}},
    "grading": {"shadows": {"hue": 220, "sat": 30}},
    "curve": [[0, 0.05], [0.5, 0.55], [1, 1]],
    "masks": [{"type": "auto", "group": "background", "name": "Background",
               "adj": {"exposure": -1.0, "blur": 30}}],
    "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8},
    "watermark": {"enabled": True, "name": "Me"},
}
_PHOTO = {
    "exposure": 0.3, "contrast": 20, "clarity": 10, "temp": 15,
    "crop": {"x": 0.2, "y": 0.0, "w": 0.6, "h": 1.0},
    "healing": {"ops": [{"kind": "spot", "points": [[0.5, 0.5]], "radius": 0.01}]},
    "masks": [{"type": "radial", "name": "Mine", "adj": {"exposure": -0.5}}],
}


def _preset(parts: list[str], edit: dict = _LOOK, pid: str = "p1") -> dict:
    return {"id": pid, "name": "Look", **presets.make(edit, parts)}


def test_modified_ticks_everything_the_edit_sets() -> None:
    """Crop and watermark included: a part that was changed is modified, and
    leaving them out read as the dialog having missed them."""
    got = presets.modified_parts(_LOOK)
    assert set(got) == {"light", "detail", "creative", "curve", "mixer", "grading", "layers",
                        "crop", "watermark"}
    assert "color" not in got and presets.AUTO_LIGHT not in got
    assert presets.modified_parts(None) == []
    assert presets.modified_parts({"tilt": 2.0}) == ["crop"], "a straighten is the crop part"
    assert presets.modified_parts({"lens": {"distortion": 10}}) == ["lens"]


def test_a_preset_stores_its_parts_only() -> None:
    p = presets.make(_LOOK, ["light", "mixer"])
    assert p["parts"] == ["light", "mixer"]
    keys = set(p["edit"]) - {"pv"}
    assert keys == set(presets.PARTS["light"]) | {"hsl"}
    with pytest.raises(AssertionError, match="unknown preset parts"):
        presets.make(_LOOK, ["light", "healing"])
    with pytest.raises(AssertionError, match="at least one part"):
        presets.make(_LOOK, [])


def test_a_faces_own_settings_and_a_layers_tag_are_not_stored() -> None:
    edit = {"portrait": {"smooth": 40, "faces": [{"at": [0.4, 0.4], "smooth": 80}]},
            "masks": [{"type": "radial", "preset": "other", "adj": {"exposure": 1}}]}
    p = presets.make(edit, ["portrait", "layers"])
    assert "faces" not in p["edit"]["portrait"]
    assert "preset" not in p["edit"]["masks"][0]


def test_applying_sets_the_ticked_parts_and_leaves_the_rest() -> None:
    out = presets.apply(_PHOTO, _preset(["light", "detail"]), 100)
    assert out["exposure"] == 0.5
    assert out["contrast"] == 0, "a ticked zero is written, not read as 'leave it'"
    assert out["clarity"] == 30 and out["vignette"] == -20
    assert out["temp"] == 15, "color was not ticked"
    assert out["crop"] == editing.normalize(_PHOTO)["crop"], "crop was not ticked"
    assert out["healing"] == editing.normalize(_PHOTO)["healing"], "heals are the photo's"
    assert out["hsl"] is None and out["masks"] == editing.normalize(_PHOTO)["masks"]


def test_values_are_absolute_not_added() -> None:
    out = presets.apply({"exposure": 0.3}, _preset(["light"]), 100)
    assert out["exposure"] == 0.5


def test_layers_are_added_tagged_and_replaced_on_reapply() -> None:
    p = _preset(["layers"])
    once = presets.apply(_PHOTO, p, 100)
    assert [(m["name"], m.get("preset")) for m in once["masks"]] == [("Mine", None), ("Background", "p1")]
    twice = presets.apply(once, p, 100)
    assert twice["masks"] == once["masks"], "the same preset again must not stack"
    other = presets.apply(once, _preset(["layers"], pid="p2"), 100)
    assert [m.get("preset") for m in other["masks"]] == [None, "p1", "p2"]


def test_amount_moves_each_value_from_the_photos_towards_the_presets() -> None:
    p = _preset(["light", "detail", "creative", "mixer", "layers"])
    half = presets.apply(_PHOTO, p, 50)
    assert half["exposure"] == pytest.approx(0.4)
    assert half["contrast"] == 10 and half["clarity"] == 20
    assert half["hsl"]["red"]["sat"] == 10
    assert half["masks"][1]["adj"]["exposure"] == pytest.approx(-0.5)
    assert half["masks"][1]["adj"]["blur"] == 15
    assert half["motion_angle"] == 40, "an angle is not a quantity: set as stored"
    double = presets.apply(_PHOTO, p, 200)
    assert double["exposure"] == pytest.approx(0.7)
    assert double["vignette"] == -40
    assert presets.apply(_PHOTO, p, 0) == editing.normalize(_PHOTO)
    with pytest.raises(AssertionError, match="outside"):
        presets.apply(_PHOTO, p, 250)


def test_a_curve_between_two_curves_is_between_them() -> None:
    p = _preset(["curve"])
    full = presets.apply(None, p, 100)
    assert full["curve"] == editing.normalize(_LOOK)["curve"]
    half = presets.apply(None, p, 50)
    lut_id = editing._curve_lut([[0, 0], [1, 1]])
    lut_p = editing._curve_lut(full["curve"])
    lut_h = editing._curve_lut(half["curve"])
    mid = (lut_id + lut_p) / 2
    assert np.abs(lut_h - mid).max() < 0.02


def test_a_look_and_a_film_scale_their_strength_no_further_than_full() -> None:
    edit = {"lut": {"key": "a" * editing._LUT_KEY_LEN, "name": "x", "amount": 80},
            "film": {"enabled": True, "strength": 90, "grain": 40}}
    p = _preset(["look", "film"], edit)
    half = presets.apply(None, p, 50)
    assert half["lut"]["amount"] == 40 and half["film"]["strength"] == 45
    over = presets.apply(None, p, 150)
    assert over["lut"]["amount"] == 80 and over["film"]["strength"] == 90


def test_auto_light_sets_the_photos_own_tone_after_the_stored_light() -> None:
    p = _preset(["light", presets.AUTO_LIGHT])
    auto = {"exposure": 1.2, "contrast": 10, "whites": 5, "blacks": -8}
    out = presets.apply(_PHOTO, p, 100, auto=auto)
    assert (out["exposure"], out["contrast"], out["whites"], out["blacks"]) == (1.2, 10, 5, -8)
    assert out["clarity"] == 10, "detail was not ticked"
    with pytest.raises(AssertionError, match="auto tone"):
        presets.apply(_PHOTO, p, 100)
    with pytest.raises(AssertionError, match="auto tone"):
        presets.apply(_PHOTO, _preset(["light"]), 100, auto=auto)


def test_the_process_is_the_photos_unless_it_has_none_yet() -> None:
    p = _preset(["light"])
    assert editing.process_version(p["edit"]) == editing.PROCESS_VERSION
    assert editing.process_version(presets.apply(None, p, 100)) == editing.PROCESS_VERSION
    old = {"exposure": 0.2}                 # no pv: an edit made with process 1
    assert editing.process_version(presets.apply(old, p, 100)) == 1


def test_a_preset_with_every_part_renders() -> None:
    parts = [p for p in presets.PARTS]
    p = _preset(parts)
    img = np.random.default_rng(3).integers(0, 256, (90, 120, 3), dtype=np.uint8)
    field = np.zeros((32, 32), np.float32)
    field[:, :16] = 1
    out = editing.render(img, presets.apply(_PHOTO, p, 100), auto={"background": field})
    assert out.dtype == np.uint8 and not np.array_equal(out, img)


# ----- the endpoints: save, apply on one photo, apply to many ---------------

def _project(tmp_path):
    import json
    from PIL import Image
    from pickapicka import db
    rng = np.random.default_rng(0)
    for name in ("a.jpg", "b.jpg"):
        Image.fromarray(rng.integers(0, 256, (60, 90, 3), dtype=np.uint8)).save(tmp_path / name, quality=95)
    data = db.init_db(tmp_path, "")
    data["photos"] = [{"rel_path": n, "scene": "(none)", "width": 90, "height": 60}
                      for n in ("a.jpg", "b.jpg")]
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def test_the_endpoints_save_apply_and_bulk_apply(tmp_path, monkeypatch) -> None:
    from pickapicka import server
    app = server.create_app(_project(tmp_path))
    look = {"pv": editing.PROCESS_VERSION, "exposure": 0.5, "clarity": 30,
            "masks": [{"type": "auto", "group": "skin", "name": "Skin", "adj": {"exposure": 0.4}}]}
    inspect = _route(app, "/api/presets/inspect", "POST")(server.PresetInspectPayload(edit=look))
    assert inspect["modified"] == ["light", "detail", "layers"]
    assert inspect["photo_own"] == ["crop", "lens", "watermark"]
    saved = _route(app, "/api/presets", "POST")(
        server.PresetSavePayload(name="Look", edit=look, parts=["light", "layers", presets.AUTO_LIGHT]))
    pid = saved["preset"]["id"]
    assert saved["preset"]["parts"] == ["light", "layers", presets.AUTO_LIGHT]

    # The photo has no skin, by the field the stand-in segmenter returns.
    groups_seen = []

    def fields(rel, edit):
        groups = {g for m in editing.normalize(edit)["masks"] for g in editing.mask_groups(m)}
        groups_seen.append(groups)
        return {g: np.full((16, 16), 0.3, np.float32) for g in groups}
    monkeypatch.setattr(server.AppContext, "auto_fields", lambda self, rel, edit: fields(rel, edit))
    # Auto light reads the photo's own auto tone, from the decode Auto reads.
    toned = []
    monkeypatch.setattr(server.editing, "auto_tone",
                        lambda rgb: toned.append(rgb.shape) or
                        {"exposure": 1.1, "contrast": 10, "whites": 4, "blacks": -6})
    got = _route(app, "/api/presets/apply", "POST")(
        server.PresetApplyPayload(rel_path="a.jpg", edit={"clarity": 10}, preset_id=pid, amount=100))
    assert got["skipped"] == ["Skin"] and got["edit"]["masks"] == []
    assert got["edit"]["clarity"] == 10, "detail was not in the preset"
    assert (got["edit"]["exposure"], got["edit"]["blacks"]) == (1.1, -6)
    assert toned == [(60, 90, 3)]
    assert groups_seen == [{"skin"}]

    res = _route(app, "/api/edit/bulk", "POST")(
        server.EditBulkPayload(rel_paths=["a.jpg", "b.jpg"], preset_id=pid, amount=50))
    assert res["updated"] == 2 and res["skipped"] == {"a.jpg": ["Skin"], "b.jpg": ["Skin"]}
    with pytest.raises(server.HTTPException):
        _route(app, "/api/presets/apply", "POST")(
            server.PresetApplyPayload(rel_path="a.jpg", preset_id="nope"))


def test_a_preset_from_before_parts_carries_what_its_edit_sets() -> None:
    """Saved as a bare edit, it set all of that edit when applied, so it becomes
    a preset of every part the edit sets, the opt-in ones too."""
    old = {"id": "73b4fc81", "name": "test",
           "edit": {"pv": editing.PROCESS_VERSION, "exposure": 0.4, "curve_b": [[0, 0.1], [1, 1]],
                    "crop": {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}, "sharpen_radius": 60}}
    got = presets.upgrade(old)
    assert got["parts"] == ["light", "detail", "curve", "crop"]
    assert (got["id"], got["name"]) == ("73b4fc81", "test")
    out = presets.apply({"contrast": 30, "temp": 8}, got, 100)
    assert out["exposure"] == 0.4 and out["contrast"] == 0 and out["temp"] == 8
    assert presets.upgrade(got) is got, "today's shape is left alone"
    empty = presets.upgrade({"id": "x", "name": "nothing", "edit": {}})
    assert empty["parts"] == [] and presets.apply({"temp": 8}, empty, 100)["temp"] == 8


def test_the_list_reads_an_old_preset_in_todays_shape(tmp_path) -> None:
    from pickapicka import server, userstate
    data = userstate._load()
    data["presets"] = [{"id": "73b4fc81", "name": "test", "edit": {"exposure": 0.4}}]
    userstate._save(data)
    app = server.create_app(_project(tmp_path))
    listed = _route(app, "/api/presets", "GET")()["presets"]
    assert listed[0]["parts"] == ["light"]
    got = _route(app, "/api/presets/apply", "POST")(
        server.PresetApplyPayload(rel_path="a.jpg", edit={"temp": 5}, preset_id="73b4fc81"))
    assert got["edit"]["exposure"] == 0.4 and got["edit"]["temp"] == 5
    assert "parts" not in userstate.list_presets()[0], "the file is left as it was"
