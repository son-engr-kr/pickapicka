"""How a scene's files become shots: a RAW and its JPEG are one, and two
different photos are never folded into one.

    uv run pytest tests/test_pairing.py
"""
from __future__ import annotations

from pickapicka import folderinfo
from pickapicka.scorer import pair_group


def test_same_name_in_two_subfolders_of_a_scene_is_two_photos(tmp_path) -> None:
    """Two bodies, or a wrapped counter: both files used to share one key and
    the first was dropped from the project without a word."""
    s1, s2 = tmp_path / "w" / "S1" / "IMG_1.jpg", tmp_path / "w" / "S2" / "IMG_1.jpg"
    assert sorted(pair_group([s1, s2], [])) == [(None, s1), (None, s2)]
    # A RAW still takes its own JPEG, by folder or by a folder of the same name.
    r1 = tmp_path / "w" / "S1" / "IMG_1.cr3"
    rb = tmp_path / "w" / "RAW" / "S2" / "IMG_1.cr3"
    jb = tmp_path / "w" / "JPEG" / "S2" / "IMG_1.jpg"
    assert sorted(pair_group([s1, jb], [r1, rb]), key=str) == sorted([(r1, s1), (rb, jb)], key=str)
    # The common case is unchanged: one of each is one shot.
    assert pair_group([s1], [rb]) == [(rb, s1)]


def test_the_wizard_counts_them_as_scoring_does(tmp_path) -> None:
    for n in ("w/S1/IMG_1.jpg", "w/S2/IMG_1.jpg"):
        (tmp_path / n).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / n).write_bytes(b"x")
    got = folderinfo.shots(tmp_path)
    assert got["shots"] == 2 and got["folders"] == [{"name": "w", "count": 2}]
