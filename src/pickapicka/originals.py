"""Deleting the originals of photos from disk: to the Trash, or swapped for a
small JPEG of themselves.

Lightroom's "Remove from Disk" is the model. The files go to the Trash (the
Recycle Bin on Windows), not straight to nothing, so a slip can still be
taken back out of it; the disk space comes back when the Trash is emptied.
Which files are a photo's originals is the folder's business, worked out the
way scoring pairs them (`scorer.shot_files`): a RAW goes with the JPEG the
camera wrote beside it, an HDR result with its bracket's frames.

A drive with no Recycle Bin (a USB stick, a network share) is the exception
on Windows. The shell call send2trash makes has no confirmation to show, so
on such a drive it deletes for good without a word. `has_trash` finds those
drives first, so that the user is told, and agrees, before any file on one
goes. On macOS every local volume has a Trash, and a share without one makes
the move fail rather than delete.

A small copy keeps the photo in the project for the record: its decision,
its marks, the shot as it was, at a fraction of the size. It is written
beside the original with the capture time and camera details carried over,
since scenes are grouped by capture time, and only once it is on disk does
the original go to the Trash and the copy take its name.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from send2trash import send2trash

from . import exporting, fsutil, metadata as metadata_mod
from .raw import fit_within

# The long edges a small copy can have. 2048 px is the default the dialog
# offers: a full screen at 1080p and more, where a camera's 6000 is not.
SMALL_EDGES = (1280, 2048, 3072)
# JPEG quality for a small copy, lower than an export's: it is kept for the
# record, not to print. Measured on 12 frames of a 33 MP Sony shoot (18.7 MB
# JPEGs): 0.26 MB median at 2048 px (0.08 to 0.61), 0.13 at 1280, 0.48 at 3072.
SMALL_QUALITY = 75
_DRIVE_FIXED = 3     # GetDriveTypeW: a local disk, the kind with a Recycle Bin


def has_trash(path: Path) -> bool:
    """Whether deleting `path` can go to a Trash rather than for good.

    Windows keeps a Recycle Bin on fixed drives only: GetDriveTypeW says
    removable for a USB stick and remote for a network share, mapped or not.
    """
    if sys.platform != "win32":
        return True
    import ctypes
    drive = os.path.splitdrive(os.path.abspath(path))[0]
    return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == _DRIVE_FIXED


def discard(path: Path, *, permanent: bool) -> None:
    """Send one original to the Trash, or delete it for good when the user
    agreed to that for a drive with no Trash (see `has_trash`). Retried for a
    moment on Windows while the grid is reading the file, which both refuse."""
    assert path.is_file(), f"not a file: {path}"
    fsutil.retry_busy(path.unlink if permanent else lambda: send2trash(path))


def read_small(path: Path, long_edge: int) -> tuple[np.ndarray, Image.Exif, bytes | None]:
    """A JPEG's pixels upright and at most `long_edge` long, with its EXIF and
    colour profile. Decoded at the smallest DCT scale that still covers the
    size, as thumbnails are."""
    with Image.open(path) as img:
        img.draft("RGB", (long_edge, long_edge))
        exif, icc = img.getexif(), img.info.get("icc_profile")
        rgb = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
    return fit_within(rgb, long_edge), exif, icc


def write_small(rgb: np.ndarray, dst: Path, *, src_exif: Image.Exif | None, icc: bytes | None,
                captured_at: str, from_raw: bool) -> None:
    """Write a small copy with the metadata an export with "all" keeps.

    `captured_at` is EXIF's "YYYY:MM:DD HH:MM:SS", for a RAW whose embedded
    preview has no DateTimeOriginal (Sony's), as `metadata.build_exif` says."""
    exif = metadata_mod.build_exif(src_exif, size=(rgb.shape[1], rgb.shape[0]),
                                   captured_at=captured_at, mode="all", srgb=from_raw)
    exporting.write(rgb, dst, fmt="jpeg", quality=SMALL_QUALITY, exif=exif, icc=icc)


def staged(dst: Path) -> Path:
    """Where a small copy is written before it takes `dst`'s name: beside it,
    hidden, so it is on the same volume and the final rename is atomic."""
    return dst.with_name(f".{dst.stem}.small{dst.suffix}")


def put_in_place(staged_copy: Path, originals: list[Path], dst: Path, *, permanent: bool) -> None:
    """Send `originals` to the Trash and move the staged copy to `dst`, which
    is one of them or a name none of them has. A failure part way through
    leaves the staged copy where it is, for the caller to report."""
    for f in originals:
        discard(f, permanent=permanent)
    fsutil.replace(staged_copy, dst)


def check_shape(photo: dict[str, Any], new_size: tuple[int, int]) -> None:
    """A small copy of `new_size` (width, height) must be the frame the photo
    was scored on, smaller: the same shape to a pixel's rounding, and upright
    the same way."""
    w, h = int(photo["width"]), int(photo["height"])
    nw, nh = new_size
    assert abs(nw / nh - w / h) < 2.0 / min(nw, nh), f"copy {nw}x{nh} is not the shape of {w}x{h}"


def scale_geometry(photo: dict[str, Any], new_size: tuple[int, int]) -> None:
    """Rescale what a photo stores in pixels of the frame it was scored on to
    a small copy of `new_size` (width, height): the frame's own size, and the
    face and subject boxes. A face is cut out of the file by its box, so a box
    left in the old frame's pixels would cut the wrong part of the copy."""
    check_shape(photo, new_size)
    w, h = int(photo["width"]), int(photo["height"])
    nw, nh = new_size
    sx, sy = nw / w, nh / h
    for item in (photo.get("faces") or []) + (photo.get("objects") or []):
        x, y, bw, bh = item["bbox_xywh"]
        item["bbox_xywh"] = [round(x * sx), round(y * sy), round(bw * sx), round(bh * sy)]
    photo["width"], photo["height"] = nw, nh
