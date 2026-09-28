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
    from pickapicka import folderinfo
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
    from pickapicka import folderinfo
    _project(tmp_path, "wedding")
    _project(tmp_path, "trip")

    info = folderinfo.inspect(tmp_path)

    assert info["projects"] == 2
    assert info["photos"] == 0 and info["raws"] == 0


def test_legacy_caches_beside_photos_are_skipped(tmp_path) -> None:
    from pickapicka import folderinfo
    _touch(tmp_path, "a.jpg", "picks.json", "picks.json.thumbs/a.jpg",
           "picks.json.peaks/a.png", "picks.json.hdr/m.jpg")

    info = folderinfo.inspect(tmp_path)

    assert info["photos"] == 1


def test_a_project_folder_is_flagged_and_not_counted(tmp_path) -> None:
    from pickapicka import folderinfo
    d = _project(tmp_path, "wedding")

    info = folderinfo.inspect(d)

    assert info["is_project"]
    assert info["photos"] == 0


def test_a_folder_holding_the_workspace_is_flagged(tmp_path) -> None:
    """Photos chosen at or above the workspace would put the project among
    them, which create refuses; the wizard says so before that."""
    from pickapicka import folderinfo
    ws = tmp_path / "Projects"
    ws.mkdir()
    _touch(tmp_path, "shoot/a.jpg")

    assert folderinfo.inspect(tmp_path, workspace=ws)["contains_workspace"]
    assert folderinfo.inspect(ws, workspace=ws)["contains_workspace"]
    assert not folderinfo.inspect(tmp_path / "shoot", workspace=ws)["contains_workspace"]


def test_missing_and_relative_paths(tmp_path) -> None:
    from pickapicka import folderinfo
    gone = folderinfo.inspect(tmp_path / "nope")
    assert gone["absolute"] and not gone["exists"]
    # Relative to whatever directory the server started in, which is never
    # what someone typing "photos" means.
    rel = folderinfo.inspect(Path("photos"))
    assert not rel["absolute"] and not rel["exists"]


def test_counting_stops_at_the_limit(tmp_path) -> None:
    from pickapicka import folderinfo
    _touch(tmp_path, *[f"s/{i}.jpg" for i in range(12)])

    info = folderinfo.inspect(tmp_path, limit=5)

    assert info["truncated"]
    assert info["photos"] == 5


def _jpeg(path: Path, stamp: str | None) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if stamp:
        exif.get_ifd(0x8769)[0x9003] = stamp
    Image.new("RGB", (8, 8)).save(path, exif=exif.tobytes())


def _tiff_raw(path: Path, stamp: str) -> None:
    """A TIFF-based RAW as far as its head goes, which is all a time read needs."""
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = stamp
    tmp = path.with_suffix(".tif")
    Image.new("RGB", (8, 8)).save(tmp, exif=exif.tobytes())
    tmp.rename(path)


def test_shots_pair_raw_and_jpeg_as_scoring_does(tmp_path) -> None:
    from datetime import datetime

    from pickapicka import folderinfo
    _jpeg(tmp_path / "day1" / "DSC1.JPG", "2026:05:01 10:00:00")
    _tiff_raw(tmp_path / "day1" / "DSC1.ARW", "2026:05:01 09:00:00")   # one shot, RAW's time
    _jpeg(tmp_path / "day1" / "DSC2.jpg", "2026:05:01 10:05:00")
    _jpeg(tmp_path / "day2" / "DSC1.JPG", "2026:05:02 08:00:00")        # same stem, other folder
    _jpeg(tmp_path / "loose.jpg", None)                                 # no capture time
    _project(tmp_path, "proj")                                          # never counted

    got = folderinfo.shots(tmp_path)
    assert got["shots"] == 4
    assert got["untimed"] == 1
    assert got["folders"] == [{"name": "(none)", "count": 1}, {"name": "day1", "count": 2},
                              {"name": "day2", "count": 1}]
    want = [datetime(2026, 5, 1, 9, 0), datetime(2026, 5, 1, 10, 5), datetime(2026, 5, 2, 8, 0)]
    assert got["times"] == [(t - datetime(1970, 1, 1)).total_seconds() for t in want]


def test_shots_find_raws_in_their_own_subfolder(tmp_path) -> None:
    from pickapicka import folderinfo
    _jpeg(tmp_path / "JPEG" / "a" / "DSC1.JPG", "2026:05:01 10:00:00")
    _tiff_raw(tmp_path / "RAW" / "a" / "DSC1.ARW", "2026:05:01 09:00:00")
    _tiff_raw(tmp_path / "RAW" / "a" / "DSC2.ARW", "2026:05:01 09:30:00")   # RAW only
    got = folderinfo.shots(tmp_path / "JPEG", tmp_path / "RAW")
    assert got["shots"] == 2 and got["folders"] == [{"name": "a", "count": 2}]
    assert len(got["times"]) == 2 and got["times"][1] - got["times"][0] == 30 * 60


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
