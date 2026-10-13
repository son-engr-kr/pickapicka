"""The info panel's larger faces come as larger crops.

    uv run pytest tests/test_face_detail.py
"""
from __future__ import annotations

import io
import time

import numpy as np
from PIL import Image

from pickapicka import db
from pickapicka.scorer import run_scoring
from pickapicka.server import FACE_DETAIL_EDGE, FACE_LONG_EDGE, OpenProjectPayload, create_app


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def test_a_detail_crop_is_larger_and_cached_apart(tmp_path) -> None:
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    photos.mkdir()
    Image.fromarray(np.random.default_rng(0).integers(0, 256, (1000, 1600, 3), dtype=np.uint8)).save(
        photos / "a.jpg")
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    data = db.load(proj / "picks.json")
    # A face 400 px across: its crop is 680 px, padding and all.
    data["photos"][0]["faces"] = [{"bbox_xywh": [600, 300, 400, 400], "person_id": None}]
    db.save(proj / "picks.json", data)
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while _route(app, "/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)

    face = _route(app, "/face/{rel_path:path}", "GET")

    def size(**kw) -> tuple[int, int]:
        res = face("a.jpg", idx=0, **kw)
        with Image.open(io.BytesIO(res.body)) as im:
            return im.size

    assert size() == (FACE_LONG_EDGE, FACE_LONG_EDGE)
    assert size(detail=True) == (680, 680) and 680 < FACE_DETAIL_EDGE
    assert size() == (FACE_LONG_EDGE, FACE_LONG_EDGE)       # the usual crop is still its own
