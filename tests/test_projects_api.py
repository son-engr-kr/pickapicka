"""The project-management routes, called directly on a built app.

    uv run pytest tests/test_projects_api.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pickapicka import db, server, userstate


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(userstate, "CONFIG_DIR", tmp_path / "app-data")
    monkeypatch.setattr(userstate, "STATE_FILE", tmp_path / "app-data" / "state.json")
    return server.create_app()


def _route(app, path: str, method: str):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in r.methods:
            return r.endpoint
    raise KeyError(path)


def _project(d: Path, photo_root: Path, photos=None) -> Path:
    (d / ".cache").mkdir(parents=True)
    data = db.init_db(photo_root, "")
    data["photos"] = photos or []
    db.save(d / "picks.json", data)
    return d


def test_move_into_the_photos_and_find_it_after_the_folder_moves(app, tmp_path) -> None:
    shoot = tmp_path / "trip"
    shoot.mkdir()
    (shoot / "a.jpg").write_bytes(b"x")
    ws = tmp_path / "ws"
    pdir = _project(ws / "trip", shoot, [{"rel_path": "a.jpg", "decision": "pick"}])
    userstate.add_workspace(ws)

    out = _route(app, "/api/projects/move", "POST")(server.MoveProjectsPayload(
        project_dirs=[str(pdir)], location="photos"))
    kept = (shoot / ".pickapicka" / "trip").resolve()
    assert out["results"][0]["moved_to"] == str(kept)

    listed = _route(app, "/api/workspaces/projects", "GET")(workspace=str(ws))
    assert listed["projects"] == []
    assert [p["project_dir"] for p in listed["with_photos"]] == [str(kept)]

    # The photo folder moves, project and all: listed as missing, then found.
    moved = tmp_path / "archive"
    shoot.rename(moved)
    card = _route(app, "/api/workspaces/projects", "GET")(workspace=str(ws))["with_photos"][0]
    assert card["project_missing"]
    found = _route(app, "/api/project/relink", "POST")(server.RelinkPayload(
        project_dir=str(kept), new_photo_dir=str(moved)))
    new_dir = moved.resolve() / ".pickapicka" / "trip"
    assert found == {"located": True, "project_dir": str(new_dir)}
    assert db.load(new_dir / "picks.json")["photo_root"] == str(moved.resolve())
    assert userstate.known_projects() == [str(new_dir)]


def test_merge_preview(app, tmp_path) -> None:
    photos = tmp_path / "photos"
    for n in ("a/1.jpg", "b/2.jpg", "c/3.jpg"):
        (photos / n).parent.mkdir(parents=True, exist_ok=True)
        (photos / n).write_bytes(b"x")
    a = _project(tmp_path / "ws" / "A", photos / "a", [{"rel_path": "1.jpg", "rating": 5}])
    b = _project(tmp_path / "ws" / "B", photos / "b")
    plan = _route(app, "/api/projects/merge/preview", "POST")(server.MergePayload(
        project_dirs=[str(a), str(b)]))
    assert plan["root"] == str(photos.resolve())
    assert plan["ignored_dirs"] == ["c"]
    assert plan["carried"] == 1


def test_moving_in_needs_the_photo_folder(app, tmp_path) -> None:
    """A project whose photos are gone is not moved 'into' them: that would
    make the folder up at its old path, with no photos in it."""
    pdir = _project(tmp_path / "ws" / "trip", tmp_path / "unplugged-drive" / "trip")
    out = _route(app, "/api/projects/move", "POST")(server.MoveProjectsPayload(
        project_dirs=[str(pdir)], location="photos"))
    assert "photo folder not found" in out["results"][0]["error"]
    assert not (tmp_path / "unplugged-drive").exists()
    assert (pdir / "picks.json").is_file()


def test_a_project_lost_with_its_photos_can_be_dropped_from_the_list(app, tmp_path) -> None:
    lost = tmp_path / "gone" / ".pickapicka" / "trip"
    userstate.remember_project(lost)
    out = _route(app, "/api/projects/delete", "POST")(server.DeleteProjectPayload(
        project_dir=str(lost)))
    assert out["forgotten"]
    assert userstate.known_projects() == []


def test_merge_preview_says_why_it_cannot(app, tmp_path) -> None:
    a = _project(tmp_path / "ws" / "A", tmp_path / "photos")
    (tmp_path / "photos").mkdir()
    b = _project(tmp_path / "ws" / "B", tmp_path / "elsewhere")
    with pytest.raises(server.HTTPException) as e:
        _route(app, "/api/projects/merge/preview", "POST")(server.MergePayload(
            project_dirs=[str(a), str(b)]))
    assert e.value.status_code == 400 and "not found" in e.value.detail
