"""Unit checks for per-photo edit slots. Runnable with pytest or directly:

    uv run python tests/test_slots.py

Only the rack bookkeeping is covered here — the HTTP layer around it is a thin
wrapper, but the bookkeeping has the two details worth pinning: a rack saved
under a different `EDIT_SLOTS` must still load, and an emptied rack must leave
no trace on the photo.
"""
from __future__ import annotations

from picture_classifier.server import EDIT_SLOTS, _write_slot


def _entry(tag: str) -> dict:
    return {"edit": {"temp": -20}, "saved_at": "2026-09-20T12:00:00", "name": tag}


def test_first_write_creates_the_rack() -> None:
    photo: dict = {}
    rack = _write_slot(photo, 2, _entry("a"))
    assert len(rack) == EDIT_SLOTS
    assert rack[2]["name"] == "a"
    assert all(rack[i] is None for i in range(EDIT_SLOTS) if i != 2)
    assert photo["edit_slots"] is rack


def test_writes_do_not_disturb_their_neighbours() -> None:
    photo: dict = {}
    _write_slot(photo, 0, _entry("a"))
    rack = _write_slot(photo, 5, _entry("b"))
    assert rack[0]["name"] == "a" and rack[5]["name"] == "b"


def test_clearing_the_last_entry_removes_the_key() -> None:
    # An untouched photo must stay exactly as small in picks.json as it was
    # before slots existed — six nulls on every photo is not free.
    photo: dict = {"rel_path": "a.jpg"}
    _write_slot(photo, 1, _entry("a"))
    assert "edit_slots" in photo
    assert _write_slot(photo, 1, None) == []
    assert "edit_slots" not in photo
    assert photo == {"rel_path": "a.jpg"}


def test_clearing_one_of_several_keeps_the_rack() -> None:
    photo: dict = {}
    _write_slot(photo, 0, _entry("a"))
    _write_slot(photo, 3, _entry("b"))
    rack = _write_slot(photo, 0, None)
    assert rack[0] is None and rack[3]["name"] == "b"
    assert "edit_slots" in photo


def test_a_short_rack_is_padded() -> None:
    # Written when the cap was smaller: what is there has to survive.
    photo = {"edit_slots": [_entry("old"), None]}
    rack = _write_slot(photo, 4, _entry("new"))
    assert len(rack) == EDIT_SLOTS
    assert rack[0]["name"] == "old" and rack[4]["name"] == "new"


def test_a_long_rack_is_cut() -> None:
    # ...and written when it was larger: the tail past the cap goes, rather than
    # the endpoint raising on an index it cannot address anyway.
    photo = {"edit_slots": [_entry(str(i)) for i in range(EDIT_SLOTS + 3)]}
    rack = _write_slot(photo, 0, None)
    assert len(rack) == EDIT_SLOTS
    assert rack[EDIT_SLOTS - 1]["name"] == str(EDIT_SLOTS - 1)


def test_a_junk_rack_does_not_take_the_photo_with_it() -> None:
    for junk in (None, "", 0, {}):
        photo = {"edit_slots": junk}
        rack = _write_slot(photo, 0, _entry("a"))
        assert len(rack) == EDIT_SLOTS and rack[0]["name"] == "a"


# Last in the file, and it has to stay last: it collects the test functions out
# of globals(), so anything defined below it would not exist yet and would be
# silently skipped.
def _main() -> None:
    import re
    from pathlib import Path
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    declared = len(re.findall(r"^def test_", Path(__file__).read_text(encoding="utf-8"), re.M))
    assert len(fns) == declared, (
        f"collected {len(fns)} of {declared} tests — the runner has to be the "
        f"last thing in the file")
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
