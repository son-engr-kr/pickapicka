"""Swapping a finished file into place, on Windows too.

Everything the app writes that something else may be reading (thumbnails,
RAW previews, picks.json, state.json) is written to a temp file and renamed
over the old one, so a reader never sees half a file. On macOS and Linux the
rename always succeeds. On Windows it fails with PermissionError while any
other thread has the old file open, which is exactly when the grid is
streaming the thumbnail being rebuilt, and a read of the old file fails the
same way while the rename is under way. Both are retried for a moment: the
other side's hold is one read of a small file.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

_TRIES = 100
_WAIT = 0.02    # seconds; 2 s in all


def replace(src: str | Path, dst: str | Path) -> None:
    """`os.replace`, retried while Windows reports the target busy."""
    for attempt in range(_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if os.name != "nt" or attempt == _TRIES - 1:
                raise
            time.sleep(_WAIT)


def read_text(path: Path) -> str:
    """`path.read_text(encoding="utf-8")`, retried the same way."""
    return read_stamped(path)[0].decode("utf-8")


def read_bytes(path: str | Path) -> bytes:
    """The whole file, retried the same way."""
    return read_stamped(path)[0]


def read_stamped(path: str | Path) -> tuple[bytes, float]:
    """The whole file and its modification time, both from one open, so the
    time is that of the bytes returned even if the file is replaced meanwhile."""
    for attempt in range(_TRIES):
        try:
            with open(path, "rb") as fh:
                return fh.read(), os.fstat(fh.fileno()).st_mtime
        except PermissionError:
            if os.name != "nt" or attempt == _TRIES - 1:
                raise
            time.sleep(_WAIT)
    raise AssertionError("unreachable")
