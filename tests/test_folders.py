"""Folders inside a shoot that are not part of it, and what a re-score keeps.

A folder of video stills made inside the photo folder was scored into the
project the next time it was opened, the open re-scored and regrouped faces on
its own, and the re-score dropped the project's time-gap scenes for folder ones.
Opening now only reports the change, a project can leave folders out, and a
re-score keeps the scene grouping, the ignored folders and the edit slots.

    uv run pytest tests/test_folders.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from pickapicka import db, userstate
from pickapicka.scorer import drop_folders, run_scoring
from pickapicka.server import FoldersPayload, OpenProjectPayload, create_app

TIME_GAP = {"mode": "time_gap", "gap_minutes": 5}


def _jpeg(path: Path, seed: int, taken: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken   # DateTimeOriginal
    Image.fromarray(rgb).save(path, quality=90, exif=exif)


def _shoot(tmp_path: Path) -> tuple[Path, Path]:
    """Two bursts an hour apart, scored into a time-gap project."""
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i in range(3):
        _jpeg(photos / f"a{i}.jpg", i, f"2026:10:03 10:00:0{i}")
        _jpeg(photos / f"b{i}.jpg", 10 + i, f"2026:10:03 11:00:0{i}")
    proj.mkdir()
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False, scene_grouping=TIME_GAP)
    return photos, proj


def _scenes(data: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in data["photos"]:
        out.setdefault(p["scene"], []).append(p["rel_path"])
    return {k: sorted(v) for k, v in out.items()}


def test_a_rescore_keeps_time_scenes_ignored_folders_and_edit_slots(tmp_path) -> None:
    photos, proj = _shoot(tmp_path)
    db_path = proj / "picks.json"
    first = db.load(db_path)
    assert first["scene_grouping"] == TIME_GAP
    assert sorted(_scenes(first).values()) == [["a0.jpg", "a1.jpg", "a2.jpg"], ["b0.jpg", "b1.jpg", "b2.jpg"]]

    slot = {"edit": {"exposure": 0.5}, "saved_at": "2026-10-03T12:00:00"}
    for p in first["photos"]:
        if p["rel_path"] == "a1.jpg":
            p.update(decision="pick", edit_slots=[slot, None])
    first["ignored_dirs"] = ["trailer"]
    db.save(db_path, first)
    _jpeg(photos / "trailer" / "work" / "still.jpg", 99)

    run_scoring(photos, "", db_path, with_faces=False, progress_cb=lambda *a: None, project_dir=proj)
    again = db.load(db_path)
    assert again["scene_grouping"] == TIME_GAP, "the re-score dropped the time-gap grouping"
    assert _scenes(again) == _scenes(first), "the re-score regrouped by folder"
    assert again["ignored_dirs"] == ["trailer"]
    a1 = next(p for p in again["photos"] if p["rel_path"] == "a1.jpg")
    assert a1["decision"] == "pick"
    assert a1["edit_slots"] == [slot, None], "the re-score dropped the edit slots"


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _open(app, proj: Path) -> dict:
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    state = _route(app, "/api/state", "GET")
    deadline = time.monotonic() + 30
    while state()["opening"]["running"]:
        assert time.monotonic() < deadline, "the open never finished"
        time.sleep(0.02)
    opening = state()["opening"]
    assert opening["error"] is None, opening["error"]
    return opening


def test_opening_a_changed_folder_asks_instead_of_rescoring(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(userstate, "remember_open", lambda *a, **k: None)
    photos, proj = _shoot(tmp_path)
    db_path = proj / "picks.json"
    before = db_path.read_text(encoding="utf-8")
    _jpeg(photos / "trailer" / "work" / "still1.jpg", 98)
    _jpeg(photos / "trailer" / "out" / "still2.jpg", 99)

    app = create_app()
    assert _open(app, proj)["files_changed"]
    assert db_path.read_text(encoding="utf-8") == before, "the open re-scored on its own"

    report = _route(app, "/api/folders", "GET")()
    rows = {r["name"]: r for r in report["folders"]}
    assert rows["trailer"] | {} == {"name": "trailer", "on_disk": 2, "new": 2, "missing": 0,
                                    "photos": 0, "marked": 0, "ignored": False}
    assert rows[None]["on_disk"] == 6 and rows[None]["new"] == 0
    assert (report["added"], report["missing"]) == (2, 0)

    report = _route(app, "/api/folders", "POST")(FoldersPayload(ignored=["trailer"]))
    assert (report["added"], report["missing"]) == (0, 0)
    assert json.loads(db_path.read_text(encoding="utf-8"))["ignored_dirs"] == ["trailer"]
    assert not _open(create_app(), proj)["files_changed"], "an ignored folder still counts as a change"

    report = _route(app, "/api/folders", "POST")(FoldersPayload(ignored=[]))
    assert report["added"] == 2, "a folder taken back in should be offered to the next re-score"


def test_leaving_a_folder_out_drops_its_photos_and_repairs_people(tmp_path) -> None:
    photos, proj = _shoot(tmp_path)
    db_path = proj / "picks.json"
    _jpeg(photos / "trailer" / "s0.jpg", 50)
    _jpeg(photos / "trailer" / "s1.jpg", 51)
    run_scoring(photos, "", db_path, with_faces=False, progress_cb=lambda *a: None, project_dir=proj)
    data = db.load(db_path)
    face = {"bbox_xywh": [1, 1, 8, 8], "ear": None, "det_score": 0.9, "embedding_idx": 0}
    by_rel = {p["rel_path"]: p for p in data["photos"]}
    by_rel["trailer/s0.jpg"]["faces"] = [dict(face, person_id="p0"), dict(face, person_id="p1")]
    by_rel["b2.jpg"]["faces"] = [dict(face, person_id="p0")]
    data["people"] = [
        {"id": "p0", "label": "Person 1", "priority": 1, "excluded": False, "count": 2,
         "ref": {"rel_path": "trailer/s0.jpg", "face_idx": 0}},
        {"id": "p1", "label": "Person 2", "priority": 2, "excluded": False, "count": 1,
         "ref": {"rel_path": "trailer/s0.jpg", "face_idx": 1}},
    ]

    assert drop_folders(data, {"trailer"}) == 2
    assert sorted(p["rel_path"] for p in data["photos"]) == sorted(
        f"{s}{i}.jpg" for s in "ab" for i in range(3))
    assert data["people"] == [{"id": "p0", "label": "Person 1", "priority": 1, "excluded": False,
                               "count": 1, "ref": {"rel_path": "b2.jpg", "face_idx": 0}}]
