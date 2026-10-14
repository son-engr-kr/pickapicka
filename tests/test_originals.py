"""Deleting the originals of photos from disk (originals.py, /api/originals).

Which files a photo's originals are, as scoring pairs them; that they go to
the Trash and the photos leave the project; that a small copy keeps the photo,
its capture time and its face boxes on the faces; and that a drive with no
Trash is never deleted from without the user having agreed to it. The Trash
is a folder of the test's own, so nothing reaches the real one.

    uv run pytest tests/test_originals.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickapicka import db, originals
from pickapicka.scorer import run_scoring, shot_files
from pickapicka.server import (DeleteOriginalsPayload, OpenProjectPayload, OriginalsPayload,
                               create_app)

TAKEN = "2026:10:03 10:00:0{}"


def _jpeg(path: Path, seed: int, taken: str | None = None, size=(96, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken   # DateTimeOriginal
    Image.fromarray(rgb).save(path, quality=95, exif=exif)


@pytest.fixture
def trash(tmp_path, monkeypatch) -> Path:
    """send2trash, into a folder of the test's own."""
    bin_ = tmp_path / "trash"
    bin_.mkdir()

    def send(path) -> None:
        # A rename, which a busy file on Windows refuses whole. shutil.move
        # copied first, and its retry after a refused delete trashed it twice.
        path = Path(path)
        path.rename(bin_ / f"{len(list(bin_.iterdir()))}-{path.name}")
    monkeypatch.setattr(originals, "send2trash", send)
    return bin_


def _project(tmp_path: Path, n: int = 3) -> tuple[Path, Path]:
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i in range(n):
        _jpeg(photos / f"a{i}.jpg", i, TAKEN.format(i), size=(1600, 1000))
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    return photos, proj


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _wait(state: dict) -> dict:
    deadline = time.monotonic() + 30
    while state["running"]:
        assert time.monotonic() < deadline, "the delete never finished"
        time.sleep(0.02)
    assert state.get("error") is None, state["error"]
    return state


def _open(proj: Path):
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while _route(app, "/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return app


def _delete(app, rels: list[str], **kw) -> dict:
    _route(app, "/api/originals/delete", "POST")(DeleteOriginalsPayload(rel_paths=rels, **kw))
    return _wait(_route(app, "/api/originals/status", "GET")())["result"]


# ----- which files ---------------------------------------------------------

def test_a_raw_goes_with_its_camera_jpeg_and_an_hdr_with_its_frames(tmp_path) -> None:
    photos = tmp_path / "shoot"
    for name in ("IMG_1.jpg", "IMG_2.jpg", "B_1.jpg", "B_2.jpg"):
        _jpeg(photos / name, 0)
    (photos / "IMG_1.arw").write_bytes(b"raw")
    data = {"photo_root": str(photos), "brackets": [{"members": ["B_1.jpg", "B_2.jpg"]}],
            "photos": [
                {"rel_path": "IMG_1.arw", "type": "raw", "src": "IMG_1.arw"},
                {"rel_path": "IMG_2.jpg"},
                {"rel_path": "__hdr__/B_1-hdr-x.jpg", "type": "hdr", "members": ["B_1.jpg", "B_2.jpg"]},
                {"rel_path": "gone.jpg"},
            ]}
    got = shot_files(data, tmp_path / "proj" / "picks.json", tmp_path / "proj", data["photos"])
    assert {k: sorted(f.name for f in v) for k, v in got.items()} == {
        "IMG_1.arw": ["IMG_1.arw", "IMG_1.jpg"],
        "IMG_2.jpg": ["IMG_2.jpg"],
        "__hdr__/B_1-hdr-x.jpg": ["B_1.jpg", "B_2.jpg"],
        "gone.jpg": [],
    }


# ----- to the Trash --------------------------------------------------------

def test_deleted_originals_go_to_the_trash_and_the_photos_leave(tmp_path, trash) -> None:
    photos, proj = _project(tmp_path)
    app = _open(proj)
    plan = _route(app, "/api/originals/plan", "POST")(OriginalsPayload(rel_paths=["a0.jpg", "a1.jpg"]))
    assert plan["per"]["a0.jpg"]["files"] == 1 and plan["per"]["a0.jpg"]["bytes"] > 0
    size = sum(plan["per"][r]["bytes"] for r in ("a0.jpg", "a1.jpg"))

    r = _delete(app, ["a0.jpg", "a1.jpg"], mode="trash")
    assert r["done"] == 2 and not r["failed"] and r["freed"] == size
    assert sorted(p.name for p in photos.iterdir()) == ["a2.jpg"]
    assert sorted(p.name.split("-", 1)[1] for p in trash.iterdir()) == ["a0.jpg", "a1.jpg"]
    assert [p["rel_path"] for p in db.load(proj / "picks.json")["photos"]] == ["a2.jpg"]


def test_a_drive_with_no_trash_is_deleted_from_only_when_agreed(tmp_path, trash, monkeypatch) -> None:
    photos, proj = _project(tmp_path)
    monkeypatch.setattr(originals, "has_trash", lambda path: False)
    app = _open(proj)
    assert _route(app, "/api/originals/plan", "POST")(
        OriginalsPayload(rel_paths=["a0.jpg"]))["per"]["a0.jpg"]["no_trash"] == 1

    r = _delete(app, ["a0.jpg"], mode="trash")
    assert r["done"] == 0 and "no Trash" in r["failed"][0]["error"]
    assert (photos / "a0.jpg").is_file()

    r = _delete(app, ["a0.jpg"], mode="trash", permanent_ok=True)
    assert r["done"] == 1
    assert not (photos / "a0.jpg").exists() and not list(trash.iterdir())


# ----- a small copy instead ------------------------------------------------

def test_a_small_copy_keeps_the_photo_its_time_and_its_boxes(tmp_path, trash) -> None:
    photos, proj = _project(tmp_path)
    db_path = proj / "picks.json"
    data = json.loads(db_path.read_text())
    p0 = next(p for p in data["photos"] if p["rel_path"] == "a0.jpg")
    p0.update(decision="reject", faces=[{"bbox_xywh": [480, 320, 96, 64]}])
    db_path.write_text(json.dumps(data))
    app = _open(proj)

    r = _delete(app, ["a0.jpg"], mode="small", long_edge=1280)
    assert r["done"] == 1 and not r["failed"] and r["freed"] > 0, r
    with Image.open(photos / "a0.jpg") as im:
        assert im.size == (1280, 800)
        assert im.getexif().get_ifd(0x8769)[36867] == TAKEN.format(0)
    assert [p.name.split("-", 1)[1] for p in trash.iterdir()] == ["a0.jpg"]
    assert not list(photos.glob(".*"))     # no staged copy left behind
    after = next(p for p in db.load(db_path)["photos"] if p["rel_path"] == "a0.jpg")
    assert after["decision"] == "reject"
    assert (after["width"], after["height"]) == (1280, 800)
    assert after["faces"][0]["bbox_xywh"] == [384, 256, 77, 51]


def test_the_boxes_follow_the_copy_down(tmp_path) -> None:
    photo = {"width": 6000, "height": 4000,
             "faces": [{"bbox_xywh": [3000, 2000, 600, 400]}],
             "objects": [{"bbox_xywh": [0, 0, 6000, 4000]}]}
    originals.scale_geometry(photo, (2048, 1365))
    assert (photo["width"], photo["height"]) == (2048, 1365)
    assert photo["faces"][0]["bbox_xywh"] == [1024, 682, 205, 136]
    assert photo["objects"][0]["bbox_xywh"] == [0, 0, 2048, 1365]
    with pytest.raises(AssertionError):
        originals.check_shape({"width": 6000, "height": 4000}, (1365, 2048))   # turned


def test_a_raw_becomes_its_small_jpeg(tmp_path, trash, monkeypatch) -> None:
    """The RAW and its camera JPEG both go; the photo is then the copy, which
    takes the camera JPEG's name, and keeps its decision and its face."""
    photos, proj = _project(tmp_path, n=2)
    (photos / "a0.arw").write_bytes(b"raw")
    db_path = proj / "picks.json"
    data = json.loads(db_path.read_text())
    p0 = next(p for p in data["photos"] if p["rel_path"] == "a0.jpg")
    p0.update(rel_path="a0.arw", type="raw", src="a0.arw", decision="reject",
              faces=[{"bbox_xywh": [480, 320, 96, 64]}])
    db_path.write_text(json.dumps(data))

    from pickapicka import metadata, raw
    monkeypatch.setattr(raw, "decode_preview",
                        lambda path, edge: np.full((round(edge / 1.6), edge, 3), 128, np.uint8))
    monkeypatch.setattr(metadata, "read_source", lambda path, is_raw: (Image.Exif(), None))
    # Opening reads the RAW's own size, which this stand-in file has none of.
    monkeypatch.setattr(raw, "output_size", lambda path: (6400, 4000))
    app = _open(proj)
    assert next(p for p in db.load(db_path)["photos"] if p["rel_path"] == "a0.arw")["full_size"] == [6400, 4000]

    r = _delete(app, ["a0.arw"], mode="small", long_edge=1280)
    assert r["done"] == 1, r
    assert sorted(p.name.split("-", 1)[1] for p in trash.iterdir()) == ["a0.arw", "a0.jpg"]
    with Image.open(photos / "a0.jpg") as im:
        assert im.size == (1280, 800)
        # A Sony RAW's preview has no capture time; the stored one is written.
        assert im.getexif().get_ifd(0x8769)[36867] == TAKEN.format(0)
    after = {p["rel_path"]: p for p in db.load(db_path)["photos"]}
    assert "a0.arw" not in after
    p = after["a0.jpg"]
    assert "type" not in p and "src" not in p and "full_size" not in p and p["decision"] == "reject"
    assert (p["width"], p["height"]) == (1280, 800) and p["faces"][0]["bbox_xywh"] == [384, 256, 77, 51]
