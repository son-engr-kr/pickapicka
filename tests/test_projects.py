"""Where projects live, moving them, and merging them.

    uv run pytest tests/test_projects.py
"""
from __future__ import annotations

from pathlib import Path

import pytest

from pickapicka import db, projects
from pickapicka.scorer import walk_files


def _project(d: Path, photo_root: Path, photos: list[dict] | None = None, **extra) -> Path:
    (d / ".cache").mkdir(parents=True)
    data = db.init_db(photo_root, extra.pop("jpeg_subdir", ""))
    data["photos"] = photos or []
    data.update(extra)
    db.save(d / "picks.json", data)
    return d


def _shoot(root: Path, *names: str) -> Path:
    for n in names:
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        (root / n).write_bytes(b"x")
    return root


def test_a_project_kept_with_its_photos_follows_the_folder(tmp_path) -> None:
    shoot = _shoot(tmp_path / "trip", "a.jpg")
    pdir = _project(shoot / ".pickapicka" / "trip", shoot)
    moved = tmp_path / "archive" / "trip"
    moved.parent.mkdir()
    shoot.rename(moved)
    pdir = moved / ".pickapicka" / "trip"
    assert projects.photo_root_of(pdir, db.load(pdir / "picks.json")) == moved
    assert projects.heal_photo_root(pdir)
    assert db.load(pdir / "picks.json")["photo_root"] == str(moved.resolve())
    assert not projects.heal_photo_root(pdir)


def test_a_workspace_project_keeps_its_stored_photo_folder(tmp_path) -> None:
    pdir = _project(tmp_path / "ws" / "trip", tmp_path / "elsewhere")
    assert not projects.heal_photo_root(pdir)
    assert projects.photo_root_of(pdir, db.load(pdir / "picks.json")) == tmp_path / "elsewhere"


def test_moving_there_and_back(tmp_path) -> None:
    shoot = _shoot(tmp_path / "trip", "a.jpg")
    ws_dir = _project(tmp_path / "ws" / "trip", shoot)
    (ws_dir / ".cache" / "thumb.jpg").write_bytes(b"t")
    inside = projects.target_dir("trip", shoot, "photos", tmp_path / "ws")
    projects.move_project(ws_dir, inside)
    assert not ws_dir.exists()
    assert (inside / ".cache" / "thumb.jpg").is_file()
    back = projects.target_dir("trip", shoot, "workspace", tmp_path / "ws")
    projects.move_project(inside, back)
    # An emptied .pickapicka is not left behind in the photo folder.
    assert not (shoot / ".pickapicka").exists()
    assert (back / "picks.json").is_file()


def test_moving_onto_an_existing_project_is_refused(tmp_path) -> None:
    a = _project(tmp_path / "ws" / "trip", tmp_path)
    b = _project(tmp_path / "other" / "trip", tmp_path)
    with pytest.raises(AssertionError):
        projects.move_project(a, b)


def test_scanning_skips_every_projects_folder(tmp_path) -> None:
    shoot = _shoot(tmp_path / "trip", "a.jpg", "day2/b.jpg",
                   ".pickapicka/trip/.cache/thumbs/a.jpg",
                   ".pickapicka/other/.cache/thumbs/b.jpg")
    found = {str(p.relative_to(shoot)) for p in walk_files(shoot)}
    assert found == {"a.jpg", str(Path("day2/b.jpg"))}


def test_merge_carries_marks_under_the_common_folder(tmp_path) -> None:
    photos = tmp_path / "photos"
    _shoot(photos, "day1/JPEG/a.jpg", "day2/b.jpg", "day2/c.cr3", "unrelated/z.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "day1", jpeg_subdir="JPEG", photos=[
        {"rel_path": "a.jpg", "decision": "pick", "decided_at": "2026-01-01"},
    ], luts={"k1": {"name": "warm"}})
    b = _project(tmp_path / "ws" / "B", photos / "day2", photos=[
        {"rel_path": "b.jpg", "rating": 4},
        {"rel_path": "c.cr3", "type": "raw", "src": "c.cr3", "label": "red"},
        {"rel_path": "unmarked.jpg"},
    ])
    plan = projects.merge_plan([a, b])
    assert plan["root"] == photos.resolve()
    assert plan["ignored_dirs"] == ["unrelated"]
    assert plan["photos"] == 3 and plan["source_photos"] == 4
    seed = projects.merge_seed(plan)
    assert {p["rel_path"]: p for p in seed["photos"]} == {
        str(Path("day1/JPEG/a.jpg")): {"rel_path": str(Path("day1/JPEG/a.jpg")),
                                      "decision": "pick", "decided_at": "2026-01-01"},
        str(Path("day2/b.jpg")): {"rel_path": str(Path("day2/b.jpg")), "rating": 4},
        str(Path("day2/c.cr3")): {"rel_path": str(Path("day2/c.cr3")), "label": "red"},
    }
    assert seed["scored_at"] is None and seed["jpeg_subdir"] == "" and seed["raw_subdir"] == ""
    assert seed["luts"] == {"k1": {"name": "warm"}}
    assert seed["ignored_dirs"] == ["unrelated"]


def test_merge_keeps_the_later_call_on_a_shared_photo(tmp_path) -> None:
    photos = _shoot(tmp_path / "photos", "a.jpg")
    a = _project(tmp_path / "ws" / "A", photos, photos=[
        {"rel_path": "a.jpg", "decision": "reject", "decided_at": "2026-02-01"}])
    b = _project(tmp_path / "ws" / "B", photos, photos=[
        {"rel_path": "a.jpg", "decision": "pick", "decided_at": "2026-01-01"}])
    plan = projects.merge_plan([a, b])
    assert plan["conflicts"] == 1
    assert plan["ignored_dirs"] == []   # one source covers the whole root
    assert plan["marks"]["a.jpg"]["decision"] == "reject"


def test_merge_moves_a_jpeg_mark_to_the_raw_that_now_wins(tmp_path) -> None:
    """A project that scanned only its JPEG folder, merged over a root that
    also holds the RAWs: scoring will pick the RAW, so the mark goes there."""
    photos = _shoot(tmp_path / "photos", "day1/JPEG/a.jpg", "day1/RAW/a.cr3", "day2/b.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "day1", jpeg_subdir="JPEG", photos=[
        {"rel_path": "a.jpg", "decision": "pick"}])
    b = _project(tmp_path / "ws" / "B", photos / "day2")
    plan = projects.merge_plan([a, b])
    assert plan["marks"][str(Path("day1/RAW/a.cr3"))] == {"decision": "pick"}
    assert plan["photos"] == 2


def test_merge_carries_an_hdr_mark_to_the_bracket_detected_again(tmp_path) -> None:
    from pickapicka import hdr
    photos = _shoot(tmp_path / "photos", "a/1.jpg", "a/2.jpg", "a/3.jpg", "b/x.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "a", photos=[
        {"rel_path": hdr.merged_rel_path(["1.jpg", "2.jpg", "3.jpg"]), "type": "hdr",
         "members": ["1.jpg", "2.jpg", "3.jpg"], "decision": "pick"}])
    b = _project(tmp_path / "ws" / "B", photos / "b")
    plan = projects.merge_plan([a, b])
    want = hdr.merged_rel_path([str(Path("a") / n) for n in ("1.jpg", "2.jpg", "3.jpg")])
    assert plan["marks"] == {want: {"decision": "pick"}}


def test_merge_settings(tmp_path) -> None:
    photos = _shoot(tmp_path / "photos", "a/1.jpg", "b/2.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "a", subject_classes=["car"],
                 detect_faces=False, hdr_look={"strength": 1}, scene_grouping={"mode": "folder"})
    b = _project(tmp_path / "ws" / "B", photos / "b", subject_classes=["dog"],
                 detect_faces=False, hdr_look={"strength": 2})
    plan = projects.merge_plan([a, b])
    st = plan["settings"]
    assert st["subject_classes"] == ["car", "dog"]   # what either detected
    assert st["detect_faces"] is False
    assert "hdr_look" not in st                      # they disagree: the default
    # By folder would make each project one scene under the merged root.
    assert st["scene_grouping"] == {"mode": "time_gap", "gap_minutes": 30}
    assert len(plan["notes"]) == 2


def test_merge_keeps_time_gap_and_a_whole_root_folder_grouping(tmp_path) -> None:
    photos = _shoot(tmp_path / "photos", "a/1.jpg", "b/2.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "a",
                 scene_grouping={"mode": "time_gap", "gap_minutes": 12})
    b = _project(tmp_path / "ws" / "B", photos / "b",
                 scene_grouping={"mode": "time_gap", "gap_minutes": 12})
    assert projects.merge_plan([a, b])["settings"]["scene_grouping"] == \
        {"mode": "time_gap", "gap_minutes": 12}
    whole = _project(tmp_path / "ws" / "W", photos)
    sub = _project(tmp_path / "ws" / "S", photos / "a")
    plan = projects.merge_plan([whole, sub])
    assert plan["settings"]["scene_grouping"] == {"mode": "folder"} and plan["notes"] == []


def test_merge_left_out_folders(tmp_path) -> None:
    photos = _shoot(tmp_path / "photos", "a/1.jpg", "a/stills/s.jpg", "b/2.jpg", "c/3.jpg")
    whole = _project(tmp_path / "ws" / "W", photos, ignored_dirs=["c", "b"])
    sub = _project(tmp_path / "ws" / "A", photos / "a", ignored_dirs=["stills"])
    sub_b = _project(tmp_path / "ws" / "B", photos / "b")
    plan = projects.merge_plan([whole, sub, sub_b])
    # c stays out as W had it; b is B's own photos, so it comes back in.
    assert plan["ignored_dirs"] == ["c"]
    # a/stills cannot be left out of the merged root: said, not hidden.
    assert plan["still_scanned"] == [str(Path("a/stills"))]


def test_merge_refuses_what_it_cannot_do(tmp_path) -> None:
    photos = _shoot(tmp_path / "photos", "a/1.jpg")
    a = _project(tmp_path / "ws" / "A", photos / "a")
    gone = _project(tmp_path / "ws" / "G", tmp_path / "moved-away")
    with pytest.raises(projects.MergeError, match="not found"):
        projects.merge_plan([a, gone])
    with pytest.raises(projects.MergeError, match="at least two"):
        projects.merge_plan([a, a])


def test_a_failed_move_leaves_no_half_copy(tmp_path, monkeypatch) -> None:
    import shutil
    shoot = _shoot(tmp_path / "trip", "a.jpg")
    src = _project(tmp_path / "ws" / "trip", shoot)
    dst = projects.target_dir("trip", shoot, "photos", tmp_path / "ws")

    def half_copy(a, b):
        Path(b).mkdir(parents=True)
        (Path(b) / "picks.json").write_text("{}")
        raise OSError("disk full")
    monkeypatch.setattr(shutil, "move", half_copy)
    with pytest.raises(OSError, match="disk full"):
        projects.move_project(src, dst)
    assert (src / "picks.json").is_file()
    assert not (shoot / ".pickapicka").exists()


def test_a_photo_folder_holding_kept_projects_still_reads_as_photos(tmp_path) -> None:
    from pickapicka import folderinfo
    shoot = _shoot(tmp_path / "trip", "a.jpg")
    _project(shoot / ".pickapicka" / "trip", shoot)
    info = folderinfo.inspect(shoot)
    assert info["photos"] == 1 and info["projects"] == 0


def test_relinking_ignores_kept_projects_thumbnails(tmp_path) -> None:
    from pickapicka import relink
    new = _shoot(tmp_path / "new", "day/a.jpg", ".pickapicka/other/.cache/thumbs/a.jpg")
    pdir = _project(tmp_path / "ws" / "p", tmp_path / "old", photos=[{"rel_path": "x/a.jpg"}])
    rep = relink.relink_project(pdir / "picks.json", new)
    assert rep["rematched"] == 1
    assert db.load(pdir / "picks.json")["photos"][0]["rel_path"] == str(Path("day/a.jpg"))


def test_project_moved_refiles_recents_and_view(tmp_path) -> None:
    from pickapicka import userstate
    old, new = tmp_path / "ws" / "trip", tmp_path / "trip" / ".pickapicka" / "trip"
    userstate.remember_open(old / "picks.json", tmp_path / "trip", "",
                            kind="project", project_dir=old)
    userstate.set_view(str(old), {"page": 3})
    userstate.remember_project(old)
    userstate.project_moved(old, new)
    r = userstate.get_recents()[0]
    assert r["project_dir"] == str(new) and r["db_path"] == str(new / "picks.json")
    assert userstate.get_view(str(new))["page"] == 3
    assert userstate.get_view(str(old)) == {}
    assert userstate.known_projects() == [str(new)]
    assert userstate.get_last_db_path() == new / "picks.json"
