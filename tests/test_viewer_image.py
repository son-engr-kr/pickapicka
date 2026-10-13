"""The viewer's full image (/img), with its edit and without it.

    uv run pytest tests/test_viewer_image.py
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
from fastapi.responses import FileResponse
from PIL import Image

from pickapicka.scorer import run_scoring
from pickapicka.server import OpenProjectPayload, create_app


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _open(tmp_path: Path, edit: dict) -> tuple[object, Path]:
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    photos.mkdir()
    Image.fromarray(np.full((40, 60, 3), 80, np.uint8)).save(photos / "a.jpg", quality=95)
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    data = json.loads((proj / "picks.json").read_text())
    data["photos"][0].update(edit=edit, edited_at="2026-10-13T12:00:00")
    (proj / "picks.json").write_text(json.dumps(data))
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while _route(app, "/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return app, photos / "a.jpg"


def test_the_original_leaves_the_edit_out(tmp_path) -> None:
    app, path = _open(tmp_path, {"exposure": 1.0, "crop": {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}})
    get = _route(app, "/img/{rel_path:path}", "GET")

    edited = get("a.jpg", edge=None, original=False)
    with Image.open(io.BytesIO(edited.body)) as im:
        assert im.size == (30, 20)                      # cropped
        assert np.asarray(im).mean() > 120              # a stop brighter than 80

    original = get("a.jpg", edge=None, original=True)
    assert isinstance(original, FileResponse) and Path(original.path) == path
