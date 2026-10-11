"""Photos added to or gone from the folder while the project is open.

New photos are scored and added on their own, joining the people already
there, rather than by a re-score of everything that resets their names. A
photo whose original is gone is reported, still shown from its thumbnail,
refused for editing with a 404 that says so (it was a 500), and can be taken
out of the project.

    uv run pytest tests/test_changes.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pickapicka import cluster, db
from pickapicka.scorer import drop_photos, embeddings_path, find_changes, fold_in, run_scoring
from pickapicka.server import (DropPhotosPayload, EditPreviewPayload, HTTPException,
                               OpenProjectPayload, create_app)

TIME_GAP = {"mode": "time_gap", "gap_minutes": 5}


def _jpeg(path: Path, seed: int, taken: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    exif = Image.Exif()
    if taken:
        exif.get_ifd(0x8769)[36867] = taken   # DateTimeOriginal
    Image.fromarray(rgb).save(path, quality=90, exif=exif)


def _project(tmp_path: Path, n: int = 3, grouping=None) -> tuple[Path, Path]:
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i in range(n):
        _jpeg(photos / f"a{i}.jpg", i, f"2026:10:03 10:00:0{i}")
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False, scene_grouping=grouping)
    return photos, proj


# ----- finding what changed -------------------------------------------------

def test_changes_find_new_shots_missing_photos_and_raws_for_jpegs(tmp_path) -> None:
    photos, proj = _project(tmp_path)
    data = db.load(proj / "picks.json")
    assert find_changes(data, proj / "picks.json", proj) == {"shots": [], "with_existing": 0, "missing": []}

    _jpeg(photos / "b0.jpg", 10)                       # new
    (photos / "a1.jpg").unlink()                       # gone
    (photos / "a2.cr3").write_bytes(b"raw")            # a RAW for a JPEG photo
    _jpeg(photos / "trailer" / "still.jpg", 11)        # in a folder left out
    data["ignored_dirs"] = ["trailer"]
    ch = find_changes(data, proj / "picks.json", proj)
    assert [(s, r, j.name) for s, r, j in ch["shots"]] == [("(none)", None, "b0.jpg")]
    assert ch["with_existing"] == 1
    assert ch["missing"] == ["a1.jpg"]


# ----- folding new photos in ------------------------------------------------

def test_fold_in_appends_embeddings_and_regroups_by_time(tmp_path) -> None:
    _, proj = _project(tmp_path, grouping=TIME_GAP)
    db_path = proj / "picks.json"
    np.save(embeddings_path(db_path), np.ones((2, 512), dtype=np.float32))
    data = db.load(db_path)
    data["photos"][0]["decision"] = "pick"
    new_photo = {**data["photos"][0], "rel_path": "late.jpg", "decision": None,
                 "captured_at": "2026-10-03T12:00:00",
                 "faces": [{"embedding_idx": 0, "person_id": None}]}
    new = {"photos": [new_photo], "brackets": [],
           "embeddings": np.full((1, 512), 2.0, dtype=np.float32), "with_existing": 0}
    refs = fold_in(data, new, db_path)
    emb = np.load(embeddings_path(db_path))
    assert emb.shape == (3, 512) and emb[2, 0] == 2.0
    assert data["photos"][-1]["faces"][0]["embedding_idx"] == 2   # past the rows there
    assert refs == {"faces": [(3, 0)], "objects": []}
    # Two hours after the burst: a scene of its own by time, the burst untouched.
    scenes = {p["rel_path"]: p["scene"] for p in data["photos"]}
    assert scenes["late.jpg"] != scenes["a0.jpg"] == scenes["a1.jpg"] == scenes["a2.jpg"]
    assert data["photos"][0]["decision"] == "pick"
    assert data["photos"][-1]["auto_suggestion"] == "review"   # alone in its scene


# ----- people ---------------------------------------------------------------

def _people_data(n_old: int, vectors: np.ndarray) -> dict:
    """Faces 0..n_old-1 grouped as person p0 (renamed 'Mum'); the rest new."""
    photos = [{"rel_path": f"{i}.jpg", "faces": [{"embedding_idx": i, "person_id": "p0" if i < n_old else None}]}
              for i in range(len(vectors))]
    return {"photos": photos, "clustered_at": "2026-10-01",
            "people": [{"id": "p0", "label": "Mum", "priority": 1, "excluded": False, "count": n_old,
                        "ref": {"rel_path": "0.jpg", "face_idx": 0}}]}


def test_new_faces_join_their_person_and_new_people_follow(tmp_path) -> None:
    rng = np.random.default_rng(1)
    mum, stranger = rng.normal(size=512), rng.normal(size=512)
    near = lambda v: v + rng.normal(scale=0.05, size=512)   # noqa: E731
    vectors = np.stack([near(mum) for _ in range(3)] + [near(mum)] + [near(stranger) for _ in range(3)])
    data = _people_data(3, vectors)
    joined = cluster.add_faces(data, vectors.astype(np.float32), [(i, 0) for i in range(3, 7)])
    assert joined == 4
    ids = [p["faces"][0]["person_id"] for p in data["photos"]]
    assert ids[:4] == ["p0"] * 4 and ids[4:] == ["p1"] * 3
    mum_entry, new_entry = data["people"]
    assert (mum_entry["label"], mum_entry["count"]) == ("Mum", 4)   # name kept, count grown
    assert (new_entry["id"], new_entry["label"], new_entry["priority"], new_entry["count"]) == \
        ("p1", "Person 2", 2, 3)


def test_a_project_never_grouped_is_left_for_its_first_grouping() -> None:
    vectors = np.ones((2, 512), dtype=np.float32)
    data = _people_data(1, vectors)
    data["clustered_at"] = None
    assert cluster.add_faces(data, vectors, [(1, 0)]) == 0
    assert data["photos"][1]["faces"][0]["person_id"] is None


def test_dropping_photos_repairs_people(tmp_path) -> None:
    rng = np.random.default_rng(2)
    vectors = np.stack([rng.normal(size=512) for _ in range(3)])
    data = _people_data(3, vectors)
    for p in data["photos"]:
        p["scene"] = "s"
        p["scores"] = {"blur": 1.0, "brightness": 0.5, "eye_open": None}
    data["people"][0]["ref"] = {"rel_path": "0.jpg", "face_idx": 0}
    assert drop_photos(data, {"0.jpg"}) == 1
    assert data["people"][0]["count"] == 2
    assert data["people"][0]["ref"]["rel_path"] != "0.jpg"


# ----- the open project -----------------------------------------------------

def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _wait(state: dict, what: str) -> dict:
    deadline = time.monotonic() + 30
    while state["running"]:
        assert time.monotonic() < deadline, f"{what} never finished"
        time.sleep(0.02)
    assert state.get("error") is None, state["error"]
    return state


def _open(tmp_path: Path):
    photos, proj = _project(tmp_path)
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    _wait(_route(app, "/api/state", "GET")()["opening"], "the open")
    return app, photos, proj


def test_new_photos_are_added_without_a_rescore(tmp_path) -> None:
    app, photos, proj = _open(tmp_path)
    db_path = proj / "picks.json"
    data = db.load(db_path)
    scores_before = {p["rel_path"]: p["scores"]["blur"] for p in data["photos"]}
    # A mark made in the app, as the grid would make it.
    get_db = _route(app, "/api/db", "GET")
    get_db()["photos"][0]["decision"] = "pick"

    _jpeg(photos / "b0.jpg", 20, "2026:10:03 10:00:09")
    _jpeg(photos / "b1.jpg", 21, "2026:10:03 10:00:10")
    assert _route(app, "/api/changes", "GET")()["added"] == 2
    _route(app, "/api/photos/add", "POST")()
    state = _wait(_route(app, "/api/score/status", "GET")(), "the add")
    assert state["added"] == 2

    after = db.load(db_path)
    rels = sorted(p["rel_path"] for p in after["photos"])
    assert rels == ["a0.jpg", "a1.jpg", "a2.jpg", "b0.jpg", "b1.jpg"]
    by_rel = {p["rel_path"]: p for p in after["photos"]}
    assert all(by_rel[r]["scores"]["blur"] == b for r, b in scores_before.items()), \
        "the photos already there were scored again"
    assert sum(p.get("decision") == "pick" for p in after["photos"]) == 1, \
        "a mark made while adding was lost"
    assert _route(app, "/api/changes", "GET")()["added"] == 0


def test_a_missing_original_is_reported_shown_and_refused_for_editing(tmp_path) -> None:
    app, photos, proj = _open(tmp_path)
    thumb = _route(app, "/thumb/{rel_path:path}", "GET")
    assert thumb("a1.jpg").status_code == 200       # makes the cached thumbnail
    (photos / "a1.jpg").unlink()

    ch = _route(app, "/api/changes", "GET")()
    assert (ch["missing"], ch["missing_marked"]) == (["a1.jpg"], 0)
    assert thumb("a1.jpg").status_code == 200       # from the cache, original gone
    with pytest.raises(HTTPException):
        thumb("a0.jpg-never-made")

    # Editing a photo whose original is gone: a 404 that says so, not a 500.
    from pickapicka.server import OriginalMissing
    with pytest.raises(OriginalMissing):
        _route(app, "/api/edit/preview", "POST")(EditPreviewPayload(rel_path="a1.jpg", edit={}))
    handler = app.exception_handlers[OriginalMissing]
    resp = handler(None, OriginalMissing("x"))
    assert resp.status_code == 404 and b'"missing":true' in resp.body

    out = _route(app, "/api/photos/drop", "POST")(DropPhotosPayload(rel_paths=["a1.jpg", "a0.jpg"]))
    assert out == {"dropped": 1, "kept": 1}         # a0's original is there: kept
    assert sorted(p["rel_path"] for p in db.load(proj / "picks.json")["photos"]) == ["a0.jpg", "a2.jpg"]
