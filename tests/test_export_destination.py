"""Exporting into the photo folder, and the rule that keeps an export there
out of every scan.

    uv run pytest tests/test_export_destination.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from pickapicka import db, folderinfo, projects, relink
from pickapicka.scorer import run_scoring, walk_files
from pickapicka.server import ExportPayload, ExportSettings, OpenProjectPayload, create_app


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


def _jpeg(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(path)


def _open(proj: Path):
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    return app, _wait(_route(app, "/api/state", "GET")()["opening"])


def _app(tmp_path: Path):
    """Scenes a/ and b/, a0 and b0 picked."""
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    for i, rel in enumerate(["a/a0.jpg", "a/a1.jpg", "b/b0.jpg"]):
        _jpeg(photos / rel, i)
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    data = db.load(proj / "picks.json")
    for p in data["photos"]:
        p["decision"] = "pick" if p["rel_path"].endswith("0.jpg") else None
    db.save(proj / "picks.json", data)
    app, _ = _open(proj)
    return app, photos, proj


def _export(app, target: str, **settings) -> dict:
    _route(app, "/api/export/picks", "POST")(ExportPayload(
        target_dir=target, mode="flat", settings=ExportSettings(metadata="none", **settings)))
    return _wait(_route(app, "/api/export/status", "GET")())["result"]


def test_the_photo_folder_choice_is_a_dated_folder_in_its_exports(tmp_path) -> None:
    app, photos, _ = _app(tmp_path)
    preview = _route(app, "/api/export/picks/preview", "GET")
    info = preview(which="picks", scenes=None)
    down, inside = Path(info["targets"]["downloads"]), Path(info["targets"]["photos"])
    assert down.parent == Path.home() / "Downloads"
    assert inside.parent == photos / projects.EXPORTS_DIR and inside.name.startswith("trip_")
    assert info["default_target"] == str(down)        # until it is chosen once

    result = _export(app, str(inside), destination="photos")
    assert result["copied"] == 2
    assert sorted(f.name for f in inside.iterdir()) == [projects.EXPORT_MARK, "a0.jpg", "b0.jpg"]
    assert projects.is_export_dir(inside)
    # The choice is remembered for the next export.
    assert preview(which="picks", scenes=None)["default_target"].startswith(str(photos / projects.EXPORTS_DIR))


def test_an_export_in_the_photo_folder_is_left_out_when_the_project_opens(tmp_path) -> None:
    app, photos, proj = _app(tmp_path)
    inside = _route(app, "/api/export/picks/preview", "GET")(which="picks", scenes=None)["targets"]["photos"]
    _export(app, inside)
    # Moved or renamed, it stays out: the rule is the folder's own mark.
    Path(inside).rename(photos / "keepers")

    app, opening = _open(proj)
    assert opening["files_changed"] is False
    changes = _route(app, "/api/changes", "GET")()
    assert changes["added"] == 0 and changes["missing"] == []
    assert {r["name"] for r in _route(app, "/api/folders", "GET")()["folders"]} == {"a", "b"}


def test_a_folder_chosen_inside_the_photo_folder_is_marked_too(tmp_path) -> None:
    app, photos, _ = _app(tmp_path)
    _export(app, str(photos / "for-print"))
    assert projects.is_export_dir(photos / "for-print")
    # Outside the photo folder nothing is marked: a project may be made of it.
    _export(app, str(tmp_path / "out"))
    assert not projects.is_export_dir(tmp_path / "out")
    # And an earlier export may be written into again.
    _export(app, str(photos / "for-print"))


def test_an_export_that_would_hide_photos_is_refused(tmp_path) -> None:
    app, photos, _ = _app(tmp_path)
    start = _route(app, "/api/export/picks", "POST")
    for target in (photos, photos / "a"):
        with pytest.raises(HTTPException) as exc:
            start(ExportPayload(target_dir=str(target), mode="flat"))
        assert exc.value.status_code == 400
        assert not projects.is_export_dir(target)
    # Photos not yet in the project count as photos as well.
    _jpeg(photos / "new" / "n0.jpg", 9)
    with pytest.raises(HTTPException):
        start(ExportPayload(target_dir=str(photos / "new"), mode="flat"))


def test_every_scan_of_a_folder_passes_over_an_export(tmp_path) -> None:
    root = tmp_path / "shoot"
    _jpeg(root / "a" / "x.jpg", 0)
    _jpeg(root / "Exports" / "run1" / "x.jpg", 1)
    (root / "Exports" / "run1").mkdir(exist_ok=True)
    projects.mark_export_dir(root / "Exports" / "run1")

    assert [f.relative_to(root) for f in walk_files(root)] == [Path("a") / "x.jpg"]
    # A project made of the export folder itself still finds its photos.
    assert sorted(f.name for f in walk_files(root / "Exports" / "run1")) == [projects.EXPORT_MARK, "x.jpg"]
    assert folderinfo.inspect(root)["photos"] == 1
    # Re-linking matches by file name, and must not take the export for x.jpg.
    assert relink._basename_index(root)["x.jpg"] == [root / "a" / "x.jpg"]
