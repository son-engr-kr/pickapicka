"""A photo's saved edit slots as versions: exported a file each, and filled for
many photos at once by Apply to many.

    uv run pytest tests/test_versions.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from pickapicka import db
from pickapicka.scorer import run_scoring
from pickapicka.server import (EditBulkPayload, EditSlotPayload, ExportPayload, ExportSettings,
                               OpenProjectPayload, _export_versions, create_app)

BRIGHT = {"exposure": 1.0}
GREY = {"saturation": -100}
# The photos as the scan names them, with the system's separator.
A0, A1 = str(Path("a", "a0.jpg")), str(Path("a", "a1.jpg"))


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _wait(state: dict) -> dict:
    deadline = time.monotonic() + 30
    while state["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert state.get("error") is None, state["error"]
    return state


def _app(tmp_path: Path, slots: dict[str, list]):
    """Picks a0 and a1, with `slots` (rel → edit_slots) laid on them."""
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i in range(2):
        (photos / "a").mkdir(parents=True, exist_ok=True)
        rgb = np.random.default_rng(i).integers(60, 200, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(photos / "a" / f"a{i}.jpg", quality=95)
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    data = db.load(proj / "picks.json")
    for p in data["photos"]:
        p["decision"] = "pick"
        if p["rel_path"] in slots:
            p["edit_slots"] = slots[p["rel_path"]]
    db.save(proj / "picks.json", data)
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    _wait(_route(app, "/api/state", "GET")()["opening"])
    return app, proj


def _slot(edit: dict, name: str = "") -> dict:
    return {"edit": edit, "saved_at": "2026-10-13T12:00:00", "name": name}


def _export(app, out: Path, mode: str, versions: bool) -> dict:
    _route(app, "/api/export/picks", "POST")(ExportPayload(
        target_dir=str(out), mode=mode, which="picks",
        settings=ExportSettings(metadata="none", versions=versions)))
    return _wait(_route(app, "/api/export/status", "GET")())["result"]


def test_the_versions_are_the_slots_that_differ() -> None:
    photo = {"edit": BRIGHT, "edit_slots": [
        _slot(GREY, "bw"), None, _slot(BRIGHT), _slot(GREY, "again"), _slot({}, "bw"), None]}
    # Slot 3 is the photo's own edit and slot 4 is slot 1 again; slot 5, the
    # photo with nothing on it, is a version, and its name is taken.
    assert [t for t, _ in _export_versions(photo)] == ["bw", "bw5"]
    assert _export_versions({"edit_slots": [_slot({})]}) == []    # unedited both


def test_an_export_writes_each_version_beside_its_photo(tmp_path) -> None:
    app, _ = _app(tmp_path, {A0: [_slot(GREY, "bw"), _slot(BRIGHT)]})
    preview = _route(app, "/api/export/picks/preview", "GET")(which="picks", scenes=None)
    assert preview["count"] == 2 and preview["versions"] == 2

    r = _export(app, tmp_path / "plain", "folder", versions=False)
    assert r["versions"] == 0
    assert sorted(f.name for f in (tmp_path / "plain" / "a").iterdir()) == ["a0.jpg", "a1.jpg"]

    r = _export(app, tmp_path / "out", "folder", versions=True)
    assert r["copied"] == 2 and r["versions"] == 2
    got = {f.name: np.asarray(Image.open(f), dtype=float) for f in (tmp_path / "out" / "a").iterdir()}
    assert sorted(got) == ["a0.jpg", "a0_bw.jpg", "a0_slot2.jpg", "a1.jpg"]
    bw = got["a0_bw.jpg"]
    assert np.abs(bw - bw.mean(axis=2, keepdims=True)).max() < 3          # grey
    assert got["a0_slot2.jpg"].mean() > got["a0.jpg"].mean() + 20          # a stop up


def test_a_version_never_lands_on_another_file(tmp_path) -> None:
    app, _ = _app(tmp_path, {A0: [_slot(GREY, "bw")]})
    out = tmp_path / "out"
    _export(app, out, "flat", versions=True)
    _export(app, out, "flat", versions=True)
    assert sorted(f.name for f in out.iterdir()) == [
        "a0.jpg", "a0_1.jpg", "a0_bw.jpg", "a0_bw_1.jpg", "a1.jpg", "a1_1.jpg"]


def test_apply_to_many_can_fill_a_slot_and_leave_the_edits(tmp_path) -> None:
    app, proj = _app(tmp_path, {A1: [_slot(BRIGHT, "old")]})
    bulk = _route(app, "/api/edit/bulk", "POST")
    r = bulk(EditBulkPayload(rel_paths=[A0, A1], edit=GREY, mode="add",
                             slot=0, slot_name="bw"))
    photos = {p["rel_path"]: p for p in db.load(proj / "picks.json")["photos"]}
    for rel in (A0, A1):
        assert "edit" not in photos[rel]                              # renders as before
        assert photos[rel]["edit_slots"][0]["name"] == "bw"
        assert photos[rel]["edit_slots"][0]["edit"]["saturation"] == -100
        assert r["photos"][rel]["edit_slots"][0]["name"] == "bw"

    bulk(EditBulkPayload(rel_paths=[A0, A1], edit={}, mode="replace", slot=0))
    photos = {p["rel_path"]: p for p in db.load(proj / "picks.json")["photos"]}
    assert all("edit_slots" not in p for p in photos.values())


def test_naming_a_slot_keeps_when_it_was_filled(tmp_path) -> None:
    app, proj = _app(tmp_path, {A0: [_slot(GREY)]})
    save = _route(app, "/api/edit/slot", "POST")
    stored = db.load(proj / "picks.json")["photos"][0]["edit_slots"][0]
    first = save(EditSlotPayload(rel_path=A0, slot=0, edit=stored["edit"]))["slots"][0]
    named = save(EditSlotPayload(rel_path=A0, slot=0, edit=first["edit"], name="bw"))["slots"][0]
    assert named["name"] == "bw" and named["saved_at"] == first["saved_at"]
    moved = save(EditSlotPayload(rel_path=A0, slot=0, edit=BRIGHT, name="bw"))["slots"][0]
    assert moved["saved_at"] != first["saved_at"]
