"""Checks for scene grouping.

    uv run python tests/test_scenes.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path


def test_by_folder_splits_on_either_separator() -> None:
    """A project scored on Windows stores rel_paths with backslashes."""
    from picture_classifier import scenes
    photos = [{"rel_path": "Ceremony/a.jpg"}, {"rel_path": "Reception\\\\b.jpg"},
              {"rel_path": "loose.jpg"}]
    scenes.group_by_folder(photos)
    assert [p["scene"] for p in photos] == ["Ceremony", "Reception", "(none)"]


def test_time_gap_starts_a_scene_after_a_long_pause() -> None:
    from picture_classifier import scenes
    t0 = datetime(2026, 5, 1, 14, 0, 0)
    offsets = [0, 2, 5, 50, 52, 200]          # minutes
    photos = [{"rel_path": f"{i}.jpg", "captured_at": (t0 + timedelta(minutes=m)).isoformat()}
              for i, m in enumerate(offsets)] + [{"rel_path": "x.jpg", "type": "raw"}]
    scenes.group_by_time_gap(photos, Path("/nonexistent"), gap_minutes=30)
    assert [p["scene"] for p in photos] == ["Scene 01"] * 3 + ["Scene 02"] * 2 + ["Scene 03", "(no_time)"]
    scenes.group_by_time_gap(photos, Path("/nonexistent"), gap_minutes=60)
    assert [p["scene"] for p in photos][:6] == ["Scene 01"] * 5 + ["Scene 02"]


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
