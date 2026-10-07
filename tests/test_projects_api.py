"""The project-management routes, called directly on a built app.

    uv run pytest tests/test_projects_api.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pickapicka import db, server, userstate


@pytest.fixture
def app():
    return server.create_app()   # app data is a fresh tmp dir (conftest)


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


def _ws(app, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    userstate.add_workspace(path)
    return path.resolve()


def test_merging_a_workspace_moves_its_projects_and_removes_the_folder(app, tmp_path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    old = _ws(app, tmp_path / "Old")
    new = _ws(app, tmp_path / "New")   # current
    _project(old / "trip", photos)
    _project(old / "Wedding", photos)
    _project(old / "gone.deleted-20261001-1200", photos)
    (old / ".DS_Store").write_bytes(b"x")
    _project(new / "wedding", photos)   # same name but for case
    userstate.remember_open(old / "trip" / "picks.json", photos, "", kind="project",
                            project_dir=old / "trip")

    pre = _route(app, "/api/workspaces/merge/preview", "POST")(server.MergeWorkspacesPayload(
        source=str(old), target=str(new)))
    assert (pre["projects"], pre["deleted"], pre["others"], pre["open"]) == (2, 1, [], None)
    assert pre["renamed"] == [["Wedding", "Wedding (Old)"]]

    out = _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
        source=str(old), target=str(new)))
    assert all(r.get("moved") for r in out["results"])
    assert out["removed"] and not old.exists()
    assert sorted(c.name for c in new.iterdir()) == [
        "Wedding (Old)", "gone.deleted-20261001-1200", "trip", "wedding"]
    assert out["workspaces"] == [str(new)] and out["current"] == str(new)
    assert userstate.get_recents()[0]["project_dir"] == str(new / "trip")
    # The deleted one is still hidden, and was not added to any list.
    assert str(new / "gone.deleted-20261001-1200") not in userstate.known_projects()


def test_a_workspace_with_other_files_keeps_its_folder(app, tmp_path) -> None:
    old = _ws(app, tmp_path / "Old")
    new = _ws(app, tmp_path / "New")
    _project(old / "trip", tmp_path)
    (old / "notes.txt").write_text("mine")
    pre = _route(app, "/api/workspaces/merge/preview", "POST")(server.MergeWorkspacesPayload(
        source=str(old), target=str(new)))
    assert pre["others"] == ["notes.txt"]
    out = _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
        source=str(old), target=str(new)))
    assert not out["removed"] and (old / "notes.txt").is_file()
    assert (new / "trip" / "picks.json").is_file()
    assert out["workspaces"] == [str(new)]   # no projects left in it, so off the list


def test_a_workspace_is_not_merged_into_itself(app, tmp_path) -> None:
    ws = _ws(app, tmp_path / "W")
    with pytest.raises(server.HTTPException) as e:
        _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
            source=str(ws), target=str(ws)))
    assert e.value.status_code == 400


def test_a_project_that_cannot_move_keeps_the_old_workspace(app, tmp_path, monkeypatch) -> None:
    from pickapicka import projects
    old = _ws(app, tmp_path / "Old")
    new = _ws(app, tmp_path / "New")
    _project(old / "a", tmp_path)
    _project(old / "b", tmp_path)
    real = projects.move_project

    def fail_b(src, dst):
        if src.name == "b":
            raise OSError("permission denied")
        return real(src, dst)
    monkeypatch.setattr(projects, "move_project", fail_b)
    out = _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
        source=str(old), target=str(new)))
    assert [r.get("error") for r in out["results"]] == [None, "permission denied"]
    assert (old / "b" / "picks.json").is_file() and (new / "a" / "picks.json").is_file()
    assert not out["removed"] and str(old) in out["workspaces"]


def test_projects_move_to_another_workspace_and_the_current_one_stays(app, tmp_path) -> None:
    a = _ws(app, tmp_path / "A")
    b = _ws(app, tmp_path / "B")
    _ws(app, tmp_path / "Here")            # current
    pdir = _project(a / "trip", tmp_path)
    elsewhere = (tmp_path / "Elsewhere").resolve()   # not listed yet
    out = _route(app, "/api/projects/move", "POST")(server.MoveProjectsPayload(
        project_dirs=[str(pdir)], location="workspace", workspace_dir=str(elsewhere)))
    assert out["results"][0]["moved_to"] == str(elsewhere / "trip")
    assert out["current"] == str((tmp_path / "Here").resolve())
    assert str(elsewhere) in out["workspaces"]
    # Already there: skipped, not an error.
    out = _route(app, "/api/projects/move", "POST")(server.MoveProjectsPayload(
        project_dirs=[str(elsewhere / "trip")], location="workspace", workspace_dir=str(elsewhere)))
    assert out["results"][0]["skipped"] == "already there"
    # A name taken there is refused, nothing overwritten.
    _project(b / "trip", tmp_path)
    out = _route(app, "/api/projects/move", "POST")(server.MoveProjectsPayload(
        project_dirs=[str(b / "trip")], location="workspace", workspace_dir=str(elsewhere)))
    assert "already exists" in out["results"][0]["error"]
    assert (b / "trip" / "picks.json").is_file()


def test_merging_into_another_workspace_keeps_the_current_one(app, tmp_path) -> None:
    a = _ws(app, tmp_path / "A")
    here = _ws(app, tmp_path / "Here")     # current
    _project(a / "trip", tmp_path)
    new = (tmp_path / "New").resolve()
    out = _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
        source=str(a), target=str(new)))
    assert out["current"] == str(here) and sorted(out["workspaces"]) == sorted([str(here), str(new)])
    assert (new / "trip" / "picks.json").is_file() and not a.exists()
    # Merging the current one away makes its target current.
    _project(here / "beach", tmp_path)
    out = _route(app, "/api/workspaces/merge", "POST")(server.MergeWorkspacesPayload(
        source=str(here), target=str(new)))
    assert out["current"] == str(new) and out["workspaces"] == [str(new)]


# ----- finding projects again after a move ---------------------------------

def test_a_moved_workspace_is_located_and_its_projects_refiled(app, tmp_path) -> None:
    old = _ws(app, tmp_path / "Old")
    _ws(app, tmp_path / "Other")
    userstate.set_current_workspace(old)
    photos = tmp_path / "photos"
    photos.mkdir()
    _project(old / "trip", photos)
    _project(old / "lost", tmp_path / "unplugged")
    userstate.remember_open(old / "trip" / "picks.json", photos, "", kind="project",
                            project_dir=old / "trip")
    userstate.set_view(str(old / "trip"), {"page": 4})
    new = tmp_path / "Moved"
    old.rename(new)
    new = new.resolve()

    ws = _route(app, "/api/workspaces", "GET")()
    assert ws["missing"] == [str(old)]
    out = _route(app, "/api/workspaces/relocate", "POST")(server.RelocateWorkspacePayload(
        old=str(old), new=str(new)))
    assert (out["projects"], out["photos_missing"]) == (2, 1)
    # In the old one's place in the list, and current as it was.
    assert out["workspaces"] == [str(tmp_path.resolve() / "Other"), str(new)]
    assert out["current"] == str(new)
    assert userstate.get_recents()[0]["project_dir"] == str(new / "trip")
    assert userstate.get_view(str(new / "trip"))["page"] == 4


def test_one_relink_offers_the_projects_that_moved_with_it(app, tmp_path) -> None:
    old, new = tmp_path / "A" / "Photos", tmp_path / "B" / "Archive"
    _shoot_files = lambda root, *names: [  # noqa: E731
        (root / n).parent.mkdir(parents=True, exist_ok=True) or (root / n).write_bytes(b"x")
        for n in names]
    _shoot_files(new, "2024/x/a.jpg", "2024/y/b.jpg")
    ws = _ws(app, tmp_path / "ws")
    x = _project(ws / "x", old / "2024" / "x", [{"rel_path": "a.jpg"}])
    y = _project(ws / "y", old / "2024" / "y", [{"rel_path": "b.jpg", "decision": "pick"}])

    rep = _route(app, "/api/project/relink", "POST")(server.RelinkPayload(
        project_dir=str(x), new_photo_dir=str(new / "2024" / "x")))
    assert rep["moved"] == [str(old), str(new)]
    assert [o["name"] for o in rep["others"]] == ["y"]

    out = _route(app, "/api/projects/relink-many", "POST")(server.RelinkManyPayload(
        items=[server.RelinkItem(**{k: rep["others"][0][k] for k in ("project_dir", "new_photo_dir")})]))
    assert out["results"][0]["resolved"] == 1
    data = db.load(y / "picks.json")
    assert data["photo_root"] == str((new / "2024" / "y").resolve())
    assert data["photos"][0]["decision"] == "pick"


def test_find_lists_new_projects_and_refiles_moved_ones(app, tmp_path) -> None:
    from pickapicka import projects
    drive_a, drive_b = tmp_path / "A", tmp_path / "B"
    moved = _project(drive_a / "trip" / ".pickapicka" / "trip", drive_a / "trip")
    pid = projects.ensure_id(moved)
    userstate.remember_project(moved.resolve())
    userstate.note_project_id(moved.resolve(), pid)
    userstate.remember_open(moved / "picks.json", drive_a / "trip", "", kind="project",
                            project_dir=moved.resolve())
    # The photo folder goes to the other drive, project and all.
    (drive_b).mkdir()
    (drive_a / "trip").rename(drive_b / "trip")
    _project(drive_b / "hike" / ".pickapicka" / "hike", drive_b / "hike")
    # A different project that happens to share the name: not taken for the moved one.
    _project(drive_b / "x" / ".pickapicka" / "trip", drive_b / "x")

    out = _route(app, "/api/projects/find", "POST")(server.FindProjectsPayload(root=str(drive_b)))
    status = {Path(r["project_dir"]).relative_to(drive_b.resolve()).parts[0]: r["status"]
              for r in out["results"]}
    assert status == {"hike": "new", "trip": "moved", "x": "new"}
    known = userstate.known_projects()
    assert str(moved.resolve()) not in known
    new_trip = drive_b.resolve() / "trip" / ".pickapicka" / "trip"
    assert str(new_trip) in known
    assert userstate.get_recents()[0]["project_dir"] == str(new_trip)
    assert db.load(new_trip / "picks.json")["photo_root"] == str(drive_b.resolve() / "trip")
    # Looking again finds them listed.
    again = _route(app, "/api/projects/find", "POST")(server.FindProjectsPayload(root=str(drive_b)))
    assert {r["status"] for r in again["results"]} == {"listed"}
