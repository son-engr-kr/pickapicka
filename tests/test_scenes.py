"""Checks for scene grouping.

    uv run python tests/test_scenes.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


def test_by_folder_splits_on_either_separator() -> None:
    """A project scored on Windows stores rel_paths with backslashes."""
    from pickapicka import scenes
    photos = [{"rel_path": "Ceremony/a.jpg"}, {"rel_path": "Reception\\\\b.jpg"},
              {"rel_path": "loose.jpg"}]
    scenes.group_by_folder(photos)
    assert [p["scene"] for p in photos] == ["Ceremony", "Reception", "(none)"]


def test_time_gap_starts_a_scene_after_a_long_pause() -> None:
    from pickapicka import scenes
    t0 = datetime(2026, 5, 1, 14, 0, 0)
    offsets = [0, 2, 5, 50, 52, 200]          # minutes
    photos = [{"rel_path": f"{i}.jpg", "captured_at": (t0 + timedelta(minutes=m)).isoformat()}
              for i, m in enumerate(offsets)] + [{"rel_path": "x.jpg", "type": "raw"}]
    scenes.group_by_time_gap(photos, Path("/nonexistent"), gap_minutes=30)
    assert [p["scene"] for p in photos] == ["Scene 01"] * 3 + ["Scene 02"] * 2 + ["Scene 03", "(no_time)"]
    scenes.group_by_time_gap(photos, Path("/nonexistent"), gap_minutes=60)
    assert [p["scene"] for p in photos][:6] == ["Scene 01"] * 5 + ["Scene 02"]


def test_folder_then_time_splits_each_folder_on_its_own_pauses() -> None:
    """A folder per part of a shoot that still runs for hours: split inside
    it, named after it; a folder that stays one scene keeps its plain name."""
    from pickapicka import scenes
    t0 = datetime(2026, 5, 1, 14, 0, 0)

    def at(rel: str, minutes: int | None) -> dict:
        p = {"rel_path": rel, "type": "raw"}   # raw: no file to read a time from
        if minutes is not None:
            p["captured_at"] = (t0 + timedelta(minutes=minutes)).isoformat()
        return p
    photos = [at("ceremony/a.jpg", 0), at("ceremony/b.jpg", 3),
              at("reception/c.jpg", 60), at("reception/d.jpg", 62), at("reception/e.jpg", 150),
              at("reception/f.jpg", None), at("loose.jpg", 5), at("Reception\\g.jpg", 151)]
    scenes.group_by_folder_time(photos, Path("/nonexistent"), gap_minutes=30)
    assert [p["scene"] for p in photos] == [
        "ceremony", "ceremony",
        "reception · 1", "reception · 1", "reception · 2", "reception · no time",
        "(none)", "Reception"]
    # The same through regroup, and a gap long enough keeps reception whole
    # but for the photo with no time.
    scenes.regroup(photos, Path("/nonexistent"), "folder_time", 120)
    assert [p["scene"] for p in photos][2:6] == ["reception · 1"] * 3 + ["reception · no time"]


def test_a_rescore_keeps_folder_then_time(tmp_path) -> None:
    import numpy as np
    from PIL import Image

    from pickapicka import db
    from pickapicka.scorer import run_scoring
    photos, proj = tmp_path / "shoot", tmp_path / "proj"
    for i, (folder, stamp) in enumerate([("a", "10:00:00"), ("a", "10:01:00"), ("a", "13:00:00"),
                                         ("b", "10:00:30")]):
        (photos / folder).mkdir(parents=True, exist_ok=True)
        exif = Image.Exif()
        exif.get_ifd(0x8769)[36867] = f"2026:10:03 {stamp}"
        rgb = np.random.default_rng(i).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        Image.fromarray(rgb).save(photos / folder / f"{i}.jpg", exif=exif)
    proj.mkdir()
    grouping = {"mode": "folder_time", "gap_minutes": 30}
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False, scene_grouping=grouping)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj)
    data = db.load(proj / "picks.json")
    assert data["scene_grouping"] == grouping
    assert sorted({p["scene"] for p in data["photos"]}) == ["a · 1", "a · 2", "b"]


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
