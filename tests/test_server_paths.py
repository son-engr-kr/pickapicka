"""The image routes' guard against paths that climb out of the project.

    uv run pytest tests/test_server_paths.py
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from pickapicka.server import _lexical, _within_roots


def _ctx(root: Path) -> SimpleNamespace:
    return SimpleNamespace(jpeg_root=root, hdr_root=None, photo_root=root, raw_cache_root=None)


def test_a_linked_photo_inside_the_folder_is_served(tmp_path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "a.jpg").write_bytes(b"x")
    shoot = tmp_path / "shoot" / "day1"
    shoot.mkdir(parents=True)
    try:
        os.symlink(elsewhere / "a.jpg", shoot / "a.jpg")
    except OSError:
        pytest.skip("this system does not let the tests make symlinks")
    assert _within_roots(_ctx(tmp_path / "shoot"), shoot / "a.jpg")


def test_climbing_out_is_refused(tmp_path) -> None:
    root = tmp_path / "shoot"
    root.mkdir()
    (tmp_path / "secret.txt").write_text("no")
    assert not _within_roots(_ctx(root), root / ".." / "secret.txt")
    assert not _within_roots(_ctx(root), root / "day1" / ".." / ".." / "secret.txt")
    assert _within_roots(_ctx(root), root / "day1" / ".." / "a.jpg")


def test_a_root_reached_through_a_link_still_matches(tmp_path) -> None:
    real = tmp_path / "real"
    (real / "day1").mkdir(parents=True)
    try:
        os.symlink(real, tmp_path / "alias", target_is_directory=True)
    except OSError:
        pytest.skip("this system does not let the tests make symlinks")
    # Stored resolved, asked for through the alias, and the other way round.
    assert _within_roots(_ctx(real), _lexical(tmp_path / "alias" / "day1" / "a.jpg").resolve())
    assert _within_roots(_ctx(tmp_path / "alias"), real / "day1" / "a.jpg")
