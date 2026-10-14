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
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

_TRIES = 100
_WAIT = 0.02    # seconds; 2 s in all


def replace(src: str | Path, dst: str | Path) -> None:
    """`os.replace`, retried while Windows reports the target busy."""
    retry_busy(lambda: os.replace(src, dst))


def write_text(path: Path, text: str) -> None:
    """`text` as the whole of `path`, by a temp file renamed over it. The temp
    file is this thread's own, as the thumbnails' and the RAW previews' are:
    the server saves on many threads, and with one name for every writer the
    second to finish found the first had already renamed it away."""
    tmp = path.parent / f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        tmp.write_text(text, encoding="utf-8")
        replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def retry_busy(op: Callable[[], T]) -> T:
    """Run `op` on a file, retried while Windows reports the file busy: the
    same wait as a rename gets, for a delete or a move to the Trash."""
    for attempt in range(_TRIES):
        try:
            return op()
        except PermissionError:
            if os.name != "nt" or attempt == _TRIES - 1:
                raise
            time.sleep(_WAIT)
    raise AssertionError("unreachable")


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
