"""A photo's own size, which for a RAW is not the frame it was scored on.

    uv run pytest tests/test_pixel_size.py

A RAW is scored on its cached preview (a half-size decode fitted into 2560
px), and its faces are boxed in that frame. The size it decodes and exports
at is read from the RAW and kept beside it, as `full_size`. The RAWs here are
stand-ins, so libraw's reads are replaced with what a Sony a7C II file gives.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from pickapicka import db, exifinfo, raw
from pickapicka.scorer import pixel_size, run_scoring
from pickapicka.server import EditPreviewPayload, OpenProjectPayload, create_app

FULL = (7028, 4688)        # what libraw makes of an a7C II ARW, landscape


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _fake_libraw(monkeypatch, reads: list[Path] | None = None) -> None:
    def preview(path: Path, max_edge: int) -> np.ndarray:
        w = max_edge
        return np.full((round(w * FULL[1] / FULL[0]), w, 3), 120, np.uint8)

    def size(path: Path) -> tuple[int, int]:
        if reads is not None:
            reads.append(path)
        return FULL

    monkeypatch.setattr(raw, "decode_preview", preview)
    monkeypatch.setattr(raw, "read_capture_time", lambda path: None)
    monkeypatch.setattr(raw, "output_size", size)
    monkeypatch.setattr(exifinfo, "read_raw", lambda path: dict(exifinfo.EMPTY))


def _project(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    (photos / "a").mkdir(parents=True)
    for name in ("DSC01.ARW", "DSC02.ARW"):
        (photos / "a" / name).write_bytes(b"raw")
    Image.new("RGB", (64, 48), (90, 90, 90)).save(photos / "a" / "j.jpg")
    (proj / ".cache").mkdir(parents=True)
    _fake_libraw(monkeypatch)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    return photos, proj


def _open(proj: Path):
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    state = _route(app, "/api/state", "GET")
    while state()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert state()["opening"]["error"] is None, state()["opening"]["error"]
    return app


def test_libraw_sizes_its_output_this_way() -> None:
    def sizes(flip: int, aspect: float = 1.0, w: int = 7028, h: int = 4688):
        return SimpleNamespace(width=w, height=h, flip=flip, pixel_aspect=aspect)

    assert raw.sized_output(sizes(0)) == (7028, 4688)
    assert raw.sized_output(sizes(3)) == (7028, 4688)        # half a turn keeps the shape
    assert raw.sized_output(sizes(5)) == (4688, 7028)        # a quarter turn either way
    assert raw.sized_output(sizes(6)) == (4688, 7028)
    # Pixels taller than wide are stretched in height, wider than tall in width.
    assert raw.sized_output(sizes(0, 0.5, 100, 50)) == (100, 100)
    assert raw.sized_output(sizes(0, 2.0, 50, 100)) == (100, 100)
    assert raw.sized_output(sizes(6, 2.0, 50, 100)) == (100, 100)


def test_a_raw_keeps_its_own_size_beside_the_frame_it_was_scored_on(tmp_path, monkeypatch) -> None:
    _, proj = _project(tmp_path, monkeypatch)
    photos = {Path(p["rel_path"]).name: p for p in db.load(proj / "picks.json")["photos"]}
    arw = photos["DSC01.ARW"]
    assert (arw["width"], arw["height"]) == (2560, 1708)     # the preview: faces are boxed in it
    assert arw["full_size"] == list(FULL)
    assert pixel_size(arw) == FULL
    jpg = photos["j.jpg"]
    assert "full_size" not in jpg and pixel_size(jpg) == (64, 48)
    assert pixel_size({"type": "raw", "width": 2560, "height": 1708}) is None


def test_a_project_scored_before_gets_its_raw_sizes_once_on_opening(tmp_path, monkeypatch) -> None:
    photos, proj = _project(tmp_path, monkeypatch)
    data = db.load(proj / "picks.json")
    for p in data["photos"]:
        p.pop("full_size", None)
    db.save(proj / "picks.json", data)
    (photos / "a" / "DSC02.ARW").unlink()          # not on disk: left for a later open
    reads: list[Path] = []
    _fake_libraw(monkeypatch, reads)

    _open(proj)
    after = {Path(p["rel_path"]).name: p for p in db.load(proj / "picks.json")["photos"]}
    assert after["DSC01.ARW"]["full_size"] == list(FULL)
    assert "full_size" not in after["DSC02.ARW"]
    assert [f.name for f in reads] == ["DSC01.ARW"]

    reads.clear()
    _open(proj)
    assert reads == []                              # saved, so not read again


def test_the_editor_measures_a_raw_by_its_own_size(tmp_path, monkeypatch) -> None:
    """The crop's pixel count and the 1:1 view are laid out from X-Frame-Size,
    which used to be the preview's 2560 px."""
    _, proj = _project(tmp_path, monkeypatch)
    app = _open(proj)
    preview = _route(app, "/api/edit/preview", "POST")
    rel = str(Path("a") / "DSC01.ARW")
    res = preview(EditPreviewPayload(rel_path=rel, edit={}))
    assert res.headers["X-Frame-Size"] == "7028x4688"
    assert res.headers["X-Frame-Full"] == "7028x4688"
    res = preview(EditPreviewPayload(rel_path=rel, edit={"crop": {"x": 0, "y": 0, "w": 0.5, "h": 0.5}}))
    assert res.headers["X-Frame-Size"] == "3514x2344"


def test_the_editor_reads_the_size_of_a_raw_the_db_has_none_for(tmp_path, monkeypatch) -> None:
    """A RAW that was not on disk when its project was opened."""
    _, proj = _project(tmp_path, monkeypatch)
    app = _open(proj)
    rel = str(Path("a") / "DSC02.ARW")
    # An unedited photo goes out as the project's own dict, so this is the one
    # the server keeps.
    photo = next(p for p in _route(app, "/api/db", "GET")()["photos"] if p["rel_path"] == rel)
    photo.pop("full_size")
    res = _route(app, "/api/edit/preview", "POST")(EditPreviewPayload(rel_path=rel, edit={}))
    assert res.headers["X-Frame-Size"] == "7028x4688"
    assert photo["full_size"] == list(FULL)
