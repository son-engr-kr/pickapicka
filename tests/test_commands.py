"""The command line an assistant works a project with (commands.py, link.py),
run against a small scored project with no app running, and the app's side
of it: counting outside changes, and the note of where it listens.

    uv run pytest tests/test_commands.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from PIL import Image

from pickapicka import cli, db, link, server, skill, userstate
from pickapicka.scorer import run_scoring


def _jpeg(path: Path, seed: int, taken: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.random.default_rng(seed).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[36867] = taken   # DateTimeOriginal
    Image.fromarray(rgb).save(path, quality=90, exif=exif)


@pytest.fixture
def proj(tmp_path: Path) -> Path:
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i in range(4):
        _jpeg(photos / f"a{i}.jpg", i, f"2026:10:03 10:00:0{i}")
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    return proj


def _run(*args: str, ok: bool = True) -> dict:
    r = CliRunner().invoke(cli.main, list(args))
    if not ok:
        assert r.exit_code == 1, r.output
        return json.loads(r.stderr)
    assert r.exit_code == 0, (r.output, r.stderr, r.exception)
    return json.loads(r.stdout)


def _photo(proj: Path, rel: str) -> dict:
    return next(p for p in db.load(proj / "picks.json")["photos"] if p["rel_path"] == rel)


def test_photos_and_scenes(proj: Path) -> None:
    out = _run("photos", "-p", str(proj))
    assert out["count"] == 4
    assert [p["rel_path"] for p in out["photos"]] == ["a0.jpg", "a1.jpg", "a2.jpg", "a3.jpg"]
    assert set(out["photos"][0]["scores"]) == {"badness", "blur_pct", "exposure_zscore", "eye_open"}
    best = _run("photos", "-p", str(proj), "--sort", "badness", "--limit", "1")
    assert best["count"] == 4 and best["shown"] == 1
    assert sum(s["photos"] for s in _run("scenes", "-p", str(proj))["scenes"]) == 4


def test_decide_mark_and_undo(proj: Path) -> None:
    out = _run("decide", "pick", "a0.jpg", "a1.jpg", "-p", str(proj))
    assert out == {"updated": 2, "decision": "pick", "project": str(proj.resolve()), "in_app": False}
    _run("mark", "a0.jpg", "--rating", "4", "--label", "green", "-p", str(proj))
    assert (_photo(proj, "a0.jpg")["rating"], _photo(proj, "a0.jpg")["label"]) == (4, "green")
    assert _run("photos", "-p", str(proj), "--decision", "pick", "--rating", "4")["count"] == 1

    assert _run("undo", "-p", str(proj))["undone"] == "mark"
    assert "rating" not in _photo(proj, "a0.jpg") and "label" not in _photo(proj, "a0.jpg")
    assert _run("undo", "-p", str(proj))["undone"] == "decide pick"
    assert _photo(proj, "a0.jpg")["decision"] is None
    assert _run("undo", "-p", str(proj), ok=False)["status"] == 404


def test_a_photo_not_in_the_project_changes_nothing(proj: Path) -> None:
    err = _run("decide", "reject", "a0.jpg", "nope.jpg", "-p", str(proj), ok=False)
    assert err["error"]["rel_paths"] == ["nope.jpg"]
    assert _photo(proj, "a0.jpg")["decision"] is None


def test_quick_edit_and_undo_put_the_edit_back(proj: Path) -> None:
    out = _run("quick", "a2.jpg", "--by", "exposure=0.5,contrast=10", "-p", str(proj))
    assert out["photos"] == [{"rel_path": "a2.jpg", "edited": True, "versions": 0}]
    edit = _photo(proj, "a2.jpg")["edit"]
    assert (edit["exposure"], edit["contrast"]) == (0.5, 10)
    assert _run("quick", "a2.jpg", "--by", "nonsense=1", "-p", str(proj), ok=False)["status"] == 400
    _run("undo", "-p", str(proj))
    assert "edit" not in _photo(proj, "a2.jpg")


def test_a_version_goes_into_its_slot(proj: Path) -> None:
    _run("edit", "a1.jpg", "--add", '{"vignette": -20}', "--slot", "2", "--name", "dark", "-p", str(proj))
    p = _photo(proj, "a1.jpg")
    assert "edit" not in p
    assert p["edit_slots"][2]["name"] == "dark" and p["edit_slots"][2]["edit"]["vignette"] == -20


def test_apply_a_look_by_name(proj: Path) -> None:
    out = _run("apply", "pastel", "a3.jpg", "-p", str(proj))
    assert out["preset"] == "Pastel"
    assert _photo(proj, "a3.jpg")["edit"]["curve"] != [[0.0, 0.0], [1.0, 1.0]]
    _run("undo", "-p", str(proj))
    assert "edit" not in _photo(proj, "a3.jpg")


def test_images_to_look_at(proj: Path, tmp_path: Path) -> None:
    out = _run("thumb", "a0.jpg", "-p", str(proj), "--out", str(tmp_path / "t"))
    assert Image.open(out["files"][0]["file"]).size[0] > 0
    v = _run("view", "a0.jpg", "-p", str(proj), "--out", str(tmp_path / "v.jpg"), "--edge", "32")
    assert max(v["size"]) == 32 and v["shows"] == "original"
    s = _run("sheet", "-p", str(proj), "--out", str(tmp_path / "s"), "--cols", "2", "--rows", "1")
    assert [(x["from"], x["to"]) for x in s["sheets"]] == [(1, 2), (3, 4)]
    assert [p["n"] for p in s["photos"]] == [1, 2, 3, 4]


def test_export_leaves_the_persons_settings(proj: Path, tmp_path: Path) -> None:
    before = userstate.get_export_settings()
    _run("decide", "pick", "a0.jpg", "-p", str(proj))
    out = _run("export", "-p", str(proj), "--to", str(tmp_path / "out"), "--mode", "flat",
               "--long-edge", "64")
    assert out["copied"] == 1
    assert max(Image.open(tmp_path / "out" / "a0.jpg").size) == 64
    assert userstate.get_export_settings() == before


def test_naming_a_project(proj: Path) -> None:
    err = _run("photos", "-p", "no-such-project", ok=False)
    assert err["status"] == 404
    assert _run("photos", "-p", str(proj / "picks.json"))["count"] == 4


def test_no_project_open_and_none_named() -> None:
    assert _run("status") == {"app": {"running": False}}
    assert _run("photos", ok=False)["status"] == 400


def test_the_app_counts_outside_changes_only(proj: Path) -> None:
    app = server.create_app()
    app.state.ctx.load_project(proj)
    c = TestClient(app)
    c.post("/api/decide", json={"rel_path": "a0.jpg", "decision": "pick"})
    assert c.get("/api/state").json()["outside_rev"] == 0
    c.post("/api/decide", json={"rel_path": "a0.jpg", "decision": "pick"},
           headers={link.CHANGE_HEADER: "1"})
    assert c.get("/api/state").json()["outside_rev"] == 1
    # A refusal is not a change.
    c.post("/api/decide", json={"rel_path": "nope.jpg", "decision": "pick"},
           headers={link.CHANGE_HEADER: "1"})
    assert c.get("/api/state").json()["outside_rev"] == 1


def test_a_stale_running_note_is_no_app() -> None:
    link.note_running("http://127.0.0.1:9")   # nothing listens on the discard port
    assert link.find_running() is None
    link.forget_running()
    assert not (userstate.CONFIG_DIR / link.RUNNING_FILE).exists()


def test_the_skill_names_the_command(tmp_path: Path) -> None:
    text = skill.render("/Applications/Pickapicka.app/Contents/MacOS/pickapicka")
    assert text.startswith("---\nname: pickapicka\n")
    assert "The command is `/Applications/Pickapicka.app/Contents/MacOS/pickapicka`" in text
    assert "—" not in text
    assert skill.install(tmp_path / "s").read_text(encoding="utf-8").startswith("---")


@pytest.fixture
def claude() -> Path:
    """The test's own Claude Code folder (conftest), not yet made."""
    assert not skill.CLAUDE_DIR.exists()
    return skill.CLAUDE_DIR


def test_the_app_tells_claude_code_as_it_opens(claude: Path) -> None:
    assert skill.refresh()["found"] == "no_claude"
    assert not claude.exists()
    claude.mkdir()
    assert skill.refresh()["found"] == "written"
    dst = claude / "skills" / "pickapicka" / "SKILL.md"
    assert dst.read_text(encoding="utf-8") == skill.render()
    assert skill.refresh()["found"] == "current"
    dst.write_text(skill.render().replace("Pickapicka 0.", "Pickapicka 0.0."), encoding="utf-8")
    assert skill.refresh()["found"] == "written"   # an older version's is replaced


def test_a_skill_of_their_own_is_left(claude: Path) -> None:
    dst = claude / "skills" / "pickapicka" / "SKILL.md"
    dst.parent.mkdir(parents=True)
    dst.write_text("my own notes", encoding="utf-8")
    assert skill.refresh()["found"] == "theirs"
    assert skill.remove()["found"] == "off"
    assert dst.read_text(encoding="utf-8") == "my own notes"


def test_switched_off_the_skill_goes(claude: Path) -> None:
    claude.mkdir()
    app = server.create_app()
    assert app.state.assistant["found"] == "not_checked"   # only serve() writes it
    assert not (claude / "skills").exists()
    c = TestClient(app)
    assert c.post("/api/assistant", json={"on": True}).json()["found"] == "written"
    out = c.post("/api/assistant", json={"on": False}).json()
    assert (out["on"], out["found"]) == (False, "off")
    assert not (claude / "skills" / "pickapicka").exists()
    assert skill.refresh()["found"] == "off"   # and stays gone as the app opens again
    assert not (claude / "skills" / "pickapicka").exists()
