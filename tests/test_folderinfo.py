"""Checks for the folder description the setup screens show.

    uv run python tests/test_folderinfo.py

What the setup screens say about a folder decides whether someone gets
warned off choosing their photo folder as the workspace, and whether the
wizard lets them through, so the counts have to be right about what scanning
would find. The files are empty: only names and extensions are looked at.
"""
from __future__ import annotations

from pathlib import Path


def _touch(root: Path, *rels: str) -> None:
    for rel in rels:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()


def _project(root: Path, name: str) -> Path:
    """A project folder the way create_project leaves it, thumbnails and all."""
    d = root / name
    _touch(d, "picks.json", ".cache/thumbs/a.jpg", ".cache/thumbs/b.jpg")
    return d


def test_counts_photos_raws_and_scenes(tmp_path) -> None:
    from picture_classifier import folderinfo
    _touch(tmp_path, "Scene_1/a.JPG", "Scene_1/a.CR3", "Scene_2/b.jpeg",
           "Scene_2/deep/c.png", "loose.jpg", "notes.txt", "Scene_1/._a.JPG")

    info = folderinfo.inspect(tmp_path)

    assert info["exists"] and info["absolute"]
    assert info["photos"] == 4          # a.JPG, b.jpeg, c.png, loose.jpg
    assert info["raws"] == 1
    assert info["loose"] == 1
    # Nesting deeper does not make another scene: a scene is the first level.
    assert info["subfolders"] == 2
    assert info["projects"] == 0 and not info["truncated"]


def test_project_thumbnails_are_not_photos(tmp_path) -> None:
    """A workspace holds projects full of cached JPEGs. Counting those as photos
    would warn someone off the very folder they meant to pick."""
    from picture_classifier import folderinfo
    _project(tmp_path, "wedding")
    _project(tmp_path, "trip")

    info = folderinfo.inspect(tmp_path)

    assert info["projects"] == 2
    assert info["photos"] == 0 and info["raws"] == 0


def test_legacy_caches_beside_photos_are_skipped(tmp_path) -> None:
    from picture_classifier import folderinfo
    _touch(tmp_path, "a.jpg", "picks.json", "picks.json.thumbs/a.jpg",
           "picks.json.peaks/a.png", "picks.json.hdr/m.jpg")

    info = folderinfo.inspect(tmp_path)

    assert info["photos"] == 1


def test_a_project_folder_is_flagged_and_not_counted(tmp_path) -> None:
    from picture_classifier import folderinfo
    d = _project(tmp_path, "wedding")

    info = folderinfo.inspect(d)

    assert info["is_project"]
    assert info["photos"] == 0


def test_a_folder_holding_the_workspace_is_flagged(tmp_path) -> None:
    """Photos chosen at or above the workspace would put the project among
    them, which create refuses; the wizard says so before that."""
    from picture_classifier import folderinfo
    ws = tmp_path / "Projects"
    ws.mkdir()
    _touch(tmp_path, "shoot/a.jpg")

    assert folderinfo.inspect(tmp_path, workspace=ws)["contains_workspace"]
    assert folderinfo.inspect(ws, workspace=ws)["contains_workspace"]
    assert not folderinfo.inspect(tmp_path / "shoot", workspace=ws)["contains_workspace"]


def test_missing_and_relative_paths(tmp_path) -> None:
    from picture_classifier import folderinfo
    gone = folderinfo.inspect(tmp_path / "nope")
    assert gone["absolute"] and not gone["exists"]
    # Relative to whatever directory the server started in, which is never
    # what someone typing "photos" means.
    rel = folderinfo.inspect(Path("photos"))
    assert not rel["absolute"] and not rel["exists"]


def test_counting_stops_at_the_limit(tmp_path) -> None:
    from picture_classifier import folderinfo
    _touch(tmp_path, *[f"s/{i}.jpg" for i in range(12)])

    info = folderinfo.inspect(tmp_path, limit=5)

    assert info["truncated"]
    assert info["photos"] == 5


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
