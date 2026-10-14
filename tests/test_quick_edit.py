"""Checks for quick edit, the culling screen's Quick Develop:

    uv run pytest tests/test_quick_edit.py

A step is relative on every photo named, a photo's other settings stay, a
new edit starts at today's process, a slider stepped back to nothing leaves
the photo unedited, auto sets only what auto tone sets, and what each photo
was comes back for Undo.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from pickapicka import db, editing, presets, server
from test_presets import _project, _route

THIRD = 1 / 3


def _app(tmp_path, edits: dict):
    path = _project(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for p in data["photos"]:
        if p["rel_path"] in edits:
            p.update(edit=edits[p["rel_path"]], edited_at="2026-10-01T09:00:00")
    path.write_text(json.dumps(data), encoding="utf-8")
    return server.create_app(path), path


def _quick(app, rels, **kw):
    return _route(app, "/api/edit/quick", "POST")(server.QuickEditPayload(rel_paths=rels, **kw))


def test_a_step_is_relative_and_leaves_the_rest(tmp_path) -> None:
    app, path = _app(tmp_path, {"a.jpg": {"pv": editing.PROCESS_VERSION, "exposure": 0.5, "temp": 12}})
    got = _quick(app, ["a.jpg", "b.jpg"], by={"exposure": THIRD})["photos"]
    assert got["a.jpg"]["edit"]["exposure"] == pytest.approx(0.5 + THIRD)
    assert got["a.jpg"]["edit"]["temp"] == 12
    assert got["b.jpg"]["edit"]["exposure"] == pytest.approx(THIRD)
    assert got["b.jpg"]["edit"]["pv"] == editing.PROCESS_VERSION, "a new edit is today's"
    for _ in range(3):
        got = _quick(app, ["b.jpg"], by={"exposure": THIRD})["photos"]
    assert got["b.jpg"]["edit"]["exposure"] == 4 * THIRD
    stored = {p["rel_path"]: p for p in db.load(path)["photos"]}
    assert stored["a.jpg"]["edited_at"] != "2026-10-01T09:00:00"


def test_back_to_nothing_is_unedited_and_the_ends_hold(tmp_path) -> None:
    app, _ = _app(tmp_path, {})
    _quick(app, ["a.jpg"], by={"exposure": THIRD, "contrast": 5})
    got = _quick(app, ["a.jpg"], by={"exposure": -THIRD, "contrast": -5})["photos"]["a.jpg"]
    assert "edit" not in got and "edited_at" not in got
    got = _quick(app, ["a.jpg"], by={"exposure": 9.0, "blacks": -300})["photos"]["a.jpg"]["edit"]
    assert (got["exposure"], got["blacks"]) == (editing._RANGES["exposure"][1], -100)
    got = _quick(app, ["a.jpg"], to={"exposure": 0})["photos"]["a.jpg"]["edit"]
    assert got["exposure"] == 0 and got["blacks"] == -100


def test_auto_sets_what_auto_tone_sets_and_nothing_else(tmp_path, monkeypatch) -> None:
    app, _ = _app(tmp_path, {"a.jpg": {"exposure": 0.4, "highlights": -30, "temp": 8}})
    monkeypatch.setattr(server.editing, "auto_tone", lambda rgb: editing.normalize(
        {"pv": editing.PROCESS_VERSION, "exposure": 1.1, "contrast": 10, "whites": 4, "blacks": -6}))
    got = _quick(app, ["a.jpg"], auto=True)["photos"]["a.jpg"]["edit"]
    assert {k: got[k] for k in presets.AUTO_KEYS} == {
        "exposure": 1.1, "contrast": 10, "whites": 4, "blacks": -6}
    assert (got["highlights"], got["temp"]) == (-30, 8)
    assert got["pv"] == editing.PROCESS_VERSION, "a process-1 edit is updated first"


def test_undo_puts_each_photo_back(tmp_path) -> None:
    own = {"exposure": 0.4}                       # an older shape, as stored
    app, path = _app(tmp_path, {"a.jpg": own})
    done = _quick(app, ["a.jpg", "b.jpg"], by={"exposure": -1.0})
    _route(app, "/api/edit/restore", "POST")(server.RestoreEditsPayload(
        photos=[server.EditEntry(**b) for b in done["before"]]))
    stored = {p["rel_path"]: p for p in db.load(path)["photos"]}
    assert stored["a.jpg"]["edit"] == own and stored["a.jpg"]["edited_at"] == "2026-10-01T09:00:00"
    assert "edit" not in stored["b.jpg"]


def test_only_sliders_and_something_to_do(tmp_path) -> None:
    app, _ = _app(tmp_path, {})
    for kw in ({"by": {"crop": 1.0}}, {"by": {"motion_angle": 5.0}}, {}):
        with pytest.raises(server.HTTPException):
            _quick(app, ["a.jpg"], **kw)
    with pytest.raises(server.HTTPException):
        _quick(app, ["nope.jpg"], by={"exposure": THIRD})


def test_the_viewer_decodes_a_photo_once(tmp_path, monkeypatch) -> None:
    ctx = server.AppContext()
    ctx.load_db(_project(tmp_path))
    calls = []
    real = server.AppContext._decode_scaled

    def counted(self, rel, edge, quick=False):
        calls.append((rel, edge))
        return real(self, rel, edge, quick)
    monkeypatch.setattr(server.AppContext, "_decode_scaled", counted)
    first = ctx.decode_view("a.jpg", 64)
    assert ctx.decode_view("a.jpg", 64) is first and calls == [("a.jpg", 64)]
    ctx.decode_view("a.jpg", None)
    ctx.decode_view("a.jpg", None)
    assert calls.count(("a.jpg", None)) == 2, "a full-size decode is not kept"
    assert np.asarray(first).dtype == np.uint8


def test_straighten_steps_and_level_from_the_photos_lines(tmp_path, monkeypatch) -> None:
    keystone = {"vertical": 20.0}
    app, _ = _app(tmp_path, {"a.jpg": {"pv": editing.PROCESS_VERSION, "tilt": 1.0,
                                       "transform": keystone, "exposure": 0.3}})
    got = _quick(app, ["a.jpg"], by={"tilt": -0.5})["photos"]["a.jpg"]["edit"]
    assert got["tilt"] == 0.5 and got["transform"]["vertical"] == 20.0
    assert _quick(app, ["a.jpg"], by={"tilt": 90})["photos"]["a.jpg"]["edit"]["tilt"] == 45
    rolls = {}
    monkeypatch.setattr(server.transform_mod, "estimate_upright",
                        lambda rgb, mode: {"rotate": rolls.pop("next")})
    rolls["next"] = 2.345
    done = _quick(app, ["a.jpg"], level=True)
    edit = done["photos"]["a.jpg"]["edit"]
    assert done["levelled"] == {"a.jpg": 2.35}
    assert edit["tilt"] == 2.35 and edit["transform"] is None and edit["exposure"] == 0.3
    rolls["next"] = 0.0                       # fog: nothing to go by, left as it was
    done = _quick(app, ["a.jpg"], level=True)
    assert done["levelled"] == {"a.jpg": None} and done["photos"]["a.jpg"]["edit"]["tilt"] == 2.35
