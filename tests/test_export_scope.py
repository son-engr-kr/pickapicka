"""Exporting one scene, or a few, and more than the picks.

    uv run pytest tests/test_export_scope.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from pickapicka import db
from pickapicka.scorer import run_scoring
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


def _app(tmp_path: Path):
    """Scenes a/ and b/ (by folder): a0 pick, a1 reject, b0 pick, b1 undecided."""
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    for i, rel in enumerate(["a/a0.jpg", "a/a1.jpg", "b/b0.jpg", "b/b1.jpg"]):
        (photos / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(photos / rel)
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    data = db.load(proj / "picks.json")
    marks = {"a0.jpg": "pick", "a1.jpg": "reject", "b0.jpg": "pick"}
    for p in data["photos"]:
        p["decision"] = marks.get(Path(p["rel_path"]).name)
    db.save(proj / "picks.json", data)
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    _wait(_route(app, "/api/state", "GET")()["opening"])
    return app


def test_the_count_and_folder_follow_the_scenes_and_photos_chosen(tmp_path) -> None:
    app = _app(tmp_path)
    preview = _route(app, "/api/export/picks/preview", "GET")
    assert preview(which="picks", scenes=None)["count"] == 2
    one = preview(which="picks", scenes=["a"])
    assert one["count"] == 1
    assert Path(one["default_target"]).parent == Path.home() / "Downloads"
    assert Path(one["default_target"]).name.startswith("trip_a_")
    assert Path(preview(which="picks", scenes=["a", "b"])["default_target"]).name.startswith("trip_2")
    assert preview(which="kept", scenes=["a"])["count"] == 1      # the reject stays behind
    assert preview(which="all", scenes=["a"])["count"] == 2
    assert preview(which="kept", scenes=None)["count"] == 3


def test_a_scene_exports_on_its_own(tmp_path) -> None:
    app = _app(tmp_path)
    out = tmp_path / "out"
    started = _route(app, "/api/export/picks", "POST")(ExportPayload(
        target_dir=str(out), mode="flat", scenes=["b"], which="kept",
        settings=ExportSettings(metadata="none")))
    assert started["total"] == 2
    _wait(_route(app, "/api/export/status", "GET")())
    assert sorted(f.name for f in out.iterdir()) == ["b0.jpg", "b1.jpg"]
