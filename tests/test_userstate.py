"""Checks for the remembered per-project view.

    uv run python tests/test_userstate.py

Every test runs against a throwaway directory, so the real application-data
directory is never touched. CONFIG_DIR is read at import time, so it is
repointed here.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path


def _fresh(home: Path):
    """A userstate module writing inside `home`."""
    from pickapicka import userstate
    importlib.reload(userstate)
    userstate.CONFIG_DIR = home / "app-data"
    userstate.STATE_FILE = userstate.CONFIG_DIR / "state.json"
    return userstate


# ----- keys ---------------------------------------------------------------

def test_key_matches_how_a_project_is_filed(tmp_path) -> None:
    """A project must key the same way `remember_open` files it, or a project
    would be reopened at another one's page."""
    u = _fresh(tmp_path)
    proj, db = tmp_path / "MyProject", tmp_path / "MyProject" / "picks.json"
    assert u.view_key(db, proj) == str(proj)
    # Legacy layout: no project dir, so the db path identifies it.
    assert u.view_key(db) == str(db)
    assert u.view_key(db, None) == str(db)


# ----- round trip ---------------------------------------------------------

def test_unseen_project_has_no_view(tmp_path) -> None:
    u = _fresh(tmp_path)
    assert u.get_view("/nowhere") == {}, "must read as 'use the defaults'"


def test_view_round_trips(tmp_path) -> None:
    u = _fresh(tmp_path)
    u.set_view("/p", {"filter": "edited", "page_size": 8, "page": 3, "scene": "JPEG"})
    got = u.get_view("/p")
    assert (got["filter"], got["page_size"], got["page"], got["scene"]) == \
        ("edited", 8, 3, "JPEG")
    assert got["saved_at"], "needed to trim the oldest later"


def test_projects_do_not_share_a_view(tmp_path) -> None:
    u = _fresh(tmp_path)
    u.set_view("/a", {"filter": "pick", "page_size": 4, "page": 1, "scene": "s"})
    u.set_view("/b", {"filter": "reject", "page_size": 1, "page": 0, "scene": "t"})
    assert u.get_view("/a")["filter"] == "pick"
    assert u.get_view("/b")["filter"] == "reject"


def test_saving_again_replaces_rather_than_accumulates(tmp_path) -> None:
    u = _fresh(tmp_path)
    for page in (1, 2, 3):
        u.set_view("/p", {"filter": "all", "page_size": 4, "page": page, "scene": "s"})
    assert u.get_view("/p")["page"] == 3
    saved = json.loads(u.STATE_FILE.read_text(encoding="utf-8"))
    assert list(saved["views"]) == ["/p"]


def test_view_survives_alongside_the_rest_of_the_file(tmp_path) -> None:
    """The view map must not disturb recents or presets, which share the file."""
    u = _fresh(tmp_path)
    u.remember_open(tmp_path / "picks.json", tmp_path, "JPEG")
    u.save_preset("Punchy", {"exposure": 0.3})
    u.set_view("/p", {"filter": "pick", "page_size": 2, "page": 0, "scene": "s"})
    assert len(u.get_recents()) == 1
    assert [p["name"] for p in u.list_presets()] == ["Punchy"]
    assert u.get_view("/p")["filter"] == "pick"
    # …and the reverse order, in case one writer clobbers the other.
    u.save_preset("Flat", {"contrast": -0.2})
    assert u.get_view("/p")["filter"] == "pick"
    assert len(u.list_presets()) == 2


# ----- bounds and cleanup -------------------------------------------------

def test_the_map_stays_bounded(tmp_path) -> None:
    u = _fresh(tmp_path)
    for i in range(u.MAX_VIEWS + 12):
        u.set_view(f"/p{i:03d}", {"filter": "all", "page_size": 4,
                                  "page": i, "scene": "s"})
    saved = json.loads(u.STATE_FILE.read_text(encoding="utf-8"))
    assert len(saved["views"]) == u.MAX_VIEWS
    # The most recently left project is the one that must still be there.
    last = f"/p{u.MAX_VIEWS + 11:03d}"
    assert u.get_view(last)["page"] == u.MAX_VIEWS + 11
    assert u.get_view("/p000") == {}, "oldest should have been dropped"


def test_forgetting_a_project_drops_its_view(tmp_path) -> None:
    u = _fresh(tmp_path)
    db = tmp_path / "picks.json"
    u.remember_open(db, tmp_path, "JPEG")
    u.set_view(str(db), {"filter": "pick", "page_size": 4, "page": 2, "scene": "s"})
    u.forget(db)
    assert u.get_recents() == []
    assert u.get_view(str(db)) == {}, "a removed project must not keep a page"


def test_a_corrupt_state_file_reads_as_empty(tmp_path) -> None:
    u = _fresh(tmp_path)
    u.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    u.STATE_FILE.write_text("{ not json", encoding="utf-8")
    assert u.get_view("/p") == {}
    # …and writing recovers rather than staying broken.
    u.set_view("/p", {"filter": "all", "page_size": 4, "page": 0, "scene": "s"})
    assert u.get_view("/p")["page"] == 0


def _main() -> None:
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
