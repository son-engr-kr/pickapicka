"""RAW decode helpers (rawpy / libraw).

Kept separate from `editing` so the pipeline stays import-light and testable
without rawpy installed. RAW files aren't browser- or PIL-readable, so every
place that needs pixels goes through here:
  - the editor's base and the export bake `decode_raw`;
  - the grid thumbnail, and scoring (blur/exposure/faces, all path +
    cv2.imread based), read a cached preview JPEG written once per RAW.

Both come out of the same libraw settings, which is the point. The cached
preview used to be the JPEG the camera embeds in the file — free to read, but
it is the *camera's* rendering: its picture style, its tone curve, its own
brightening. A grid built from it showed a photo nothing else in the app could
reproduce. Open the editor and the frame dropped by most of a stop, and an edit
graded against that base exported to something different again. A half-size
decode costs real time at scan, once, and buys "what you see is the JPEG you
get" everywhere after.

`decode_raw` output is already oriented by libraw — never `exif_transpose` it
(unlike the embedded JPEG, which does carry EXIF orientation, and which
`exifinfo` still reads for the metadata strings it holds).
"""
from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from . import fsutil

RAW_EXTS = {
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf",
    ".raf", ".rw2", ".orf", ".pef", ".srw", ".dng", ".raw", ".3fr",
    ".erf", ".kdc", ".mos", ".mrw", ".x3f",
}


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS and not path.name.startswith("._")


def fit_within(arr: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the long edge is <= max_edge (no-op if already smaller)."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_edge:
        return arr
    s = max_edge / max(h, w)
    return cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def decode_raw(path: Path, half_size: bool = False) -> np.ndarray:
    """Full (or half) oriented RGB uint8 from a RAW file: camera white balance,
    8-bit sRGB, no auto-brightness (deterministic base so preview == export)."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=8,
            output_color=rawpy.ColorSpace.sRGB, half_size=half_size,
        )
    return np.ascontiguousarray(rgb)


def decode_preview(path: Path, max_edge: int) -> np.ndarray:
    """Preview-sized RGB from the same settings `decode_raw` uses.

    Half-size, which bins each 2x2 sensor block instead of demosaicing it: a few
    times faster, and tone-for-tone identical, which is all that matters at a
    size no larger than `max_edge`. Half of any RAW worth shooting still clears
    the 1280 px thumbnails and the 1024 px the blur metric measures at.
    """
    return fit_within(decode_raw(path, half_size=True), max_edge)


# Bumped when the preview rendering changes, so caches written by an older
# build are rewritten rather than trusted. Compared against mtime, which keeps
# the filename stable: nothing that reads a cached preview by path (clustering,
# thumbnails) can find itself pointed at a file that was never written.
PREVIEW_EPOCH = 1786298816.0  # 2026-08-09, the switch off the embedded JPEG


def raw_cache_path(cache_root: Path, ident: str) -> Path:
    """Stable location of a RAW's cached preview JPEG (keyed by its identity)."""
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{digest}.jpg"


def cache_is_fresh(raw_path: Path, cache_root: Path, ident: str) -> bool:
    dst = raw_cache_path(cache_root, ident)
    return (dst.exists()
            and dst.stat().st_mtime >= max(raw_path.stat().st_mtime, PREVIEW_EPOCH))


def ensure_cache(raw_path: Path, cache_root: Path, ident: str, max_edge: int = 2560) -> Path:
    """Write a preview JPEG for a RAW (once, refreshed when the RAW is newer or
    the rendering has changed) so path-based code — thumbs, blur, exposure, face
    detection — can read it.

    Written to a temp file and renamed into place: the grid asks for a preview
    it does not have from several tiles at once, and a reader must never get a
    JPEG that is still being written.
    """
    dst = raw_cache_path(cache_root, ident)
    if cache_is_fresh(raw_path, cache_root, ident):
        return dst
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp.jpg")
    try:
        Image.fromarray(decode_preview(raw_path, max_edge)).save(tmp, "JPEG", quality=92)
        fsutil.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst


# What rawpy builds from a stored 0: `datetime.fromtimestamp(0)`, local time.
_UNKNOWN_TIME = datetime.fromtimestamp(0)


def capture_time(rawpy_handle: Any) -> datetime | None:
    """Capture time off an open libraw handle, or None when the file omits it.

    This used to be read as DateTimeOriginal from the embedded JPEG's EXIF, and
    Sony writes a preview with no such tag: every ARW came back without a time,
    which left "group scenes by time gap" with nothing to group on and the
    watermark's {date} blank. libraw parses the maker's own record instead, for
    every format it supports.

    A file that does not say reports 0, which rawpy hands over as the Unix
    epoch — the one datetime here that means unknown rather than 1970.
    It is compared as a datetime: `stamp.timestamp()` raises OSError on
    Windows for the epoch in any timezone east of UTC, which failed the scan
    of every folder holding such a file.
    """
    stamp = rawpy_handle.other.timestamp
    return None if stamp <= _UNKNOWN_TIME else stamp


# Most RAW formats are TIFF underneath: ARW, NEF, CR2, DNG and others open with
# a TIFF header whose first IFD points at the Exif IFD, a few kilobytes in.
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
_HEAD_BYTES = 512 * 1024
_DATETIME_ORIGINAL = 0x9003


def head_capture_time(path: Path) -> datetime | None:
    """DateTimeOriginal read from the head of a TIFF-based RAW, or None when
    the file does not start with a TIFF header or the tag is not in its first
    512 KB.

    libraw opens the whole file to answer the same question: 478 ms an ARW
    against 0.5 ms for this, measured on 118 Sony ARW files, which gave the
    same time for every one of them. It exists for the new-project wizard, which
    reads every shot's time before anything is scored; a caller getting None
    asks `read_capture_time` instead, which knows every format libraw does
    (CR3 and RAF are not TIFF).
    """
    import warnings
    with open(path, "rb") as fh:
        head = fh.read(_HEAD_BYTES)
    if head[:4] not in _TIFF_MAGIC:
        return None
    exif = Image.Exif()
    # A head cut short of the Exif IFD reads as the tag being absent, with a
    # warning about the truncation; that is the None this returns anyway.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        exif.load(head)
        value = exif.get_ifd(0x8769).get(_DATETIME_ORIGINAL)
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None      # a malformed stamp; libraw may still read the maker's own


def read_capture_time(path: Path) -> datetime | None:
    """`capture_time` for a path. Opens the RAW for its metadata only — no
    unpack, no decode, so this is cheap next to a preview."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        return capture_time(r)
