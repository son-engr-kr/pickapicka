"""How an export is written: format, size, file name, and when to copy instead.

The pixels come from `editing.render`, the same call the editor previews with,
so an export is the edit as graded. This module decides what happens to those
pixels on the way to disk. `metadata` decides what the file says about itself.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
from PIL import Image

from .raw import fit_within

Format = Literal["jpeg", "tiff"]

EXTENSIONS: dict[str, str] = {"jpeg": ".jpg", "tiff": ".tif"}

# Tokens a file-name template may use. Anything else in braces is refused
# rather than written out literally, since "{nmae}" is a typo and not a name.
NAME_TOKENS = ("name", "seq", "date", "time", "scene", "project")
_TOKEN = re.compile(r"\{([^{}]*)\}")

# Below this a long edge is a thumbnail, not an export; above it no camera.
LONG_EDGE_MIN, LONG_EDGE_MAX = 64, 20000

_COPYABLE = {".jpg", ".jpeg"}


def check_template(template: str) -> None:
    """Raise ValueError naming what is wrong with a file-name template."""
    if not template.strip():
        raise ValueError("the file name template is empty")
    unknown = [t for t in _TOKEN.findall(template) if t not in NAME_TOKENS]
    if unknown:
        raise ValueError(
            "unknown token " + ", ".join("{" + t + "}" for t in unknown)
            + "; use " + ", ".join("{" + t + "}" for t in NAME_TOKENS))
    if _TOKEN.sub("", template).count("{") or _TOKEN.sub("", template).count("}"):
        raise ValueError("unmatched brace in the file name template")


def seq_width(total: int) -> int:
    """Digits for {seq}: at least three, so 001 sorts before 010."""
    return max(3, len(str(total)))


def render_name(template: str, *, name: str, seq: int, width: int,
                captured_at: str, scene: str, project: str) -> str:
    """The file name a template gives one photo, without its extension.

    `captured_at` is EXIF's "YYYY:MM:DD HH:MM:SS". A photo with no capture time
    gets empty {date} and {time}, which is what it has.
    """
    date, _, clock = captured_at.partition(" ")
    values = {
        "name": name,
        "seq": str(seq).zfill(width),
        "date": date.replace(":", "-") if date else "",
        "time": clock.replace(":", "") if clock else "",
        "scene": scene,
        "project": project,
    }
    return _TOKEN.sub(lambda m: values[m.group(1)], template)


def can_copy(source: Path, *, needs_render: bool, settings: Any) -> bool:
    """Whether the original file already is the export, byte for byte.

    Only an untouched JPEG exported as a full-size JPEG with all its metadata
    qualifies. Any other setting changes the file, and copying would ignore it,
    for instance by leaking a location the export was told to remove.
    """
    return (not needs_render
            and source.suffix.lower() in _COPYABLE
            and settings.format == "jpeg"
            and settings.long_edge is None
            and settings.metadata == "all")


def resize(rgb: np.ndarray, long_edge: int | None) -> np.ndarray:
    """Shrink to `long_edge` on the longer side. Never enlarges."""
    return rgb if long_edge is None else fit_within(rgb, long_edge)


def write(rgb: np.ndarray, dst: Path | BinaryIO, *, fmt: Format, quality: int,
          exif: bytes | None, icc: bytes | None, xmp: bytes | None = None) -> None:
    extra: dict[str, Any] = {}
    if exif:
        extra["exif"] = exif
    if icc:
        extra["icc_profile"] = icc
    img = Image.fromarray(rgb)
    if fmt == "jpeg":
        if xmp:
            extra["xmp"] = xmp
        img.save(dst, "JPEG", quality=quality, **extra)
    else:
        # Uncompressed. Pillow's compressed TIFF goes through libtiff, which
        # cannot write the Exif and GPS sub-IFDs ("Error setting from
        # dictionary"), so LZW would cost the capture time and the lens. Without
        # compression every lab and editor can open it, at 3 bytes a pixel.
        if xmp:
            # A TIFF's XMP is tag 700 among the others. Given as tiffinfo it
            # replaces the EXIF tags instead of joining them, so it goes in with
            # them.
            tags = Image.Exif()
            if exif:
                tags.load(exif)
            tags[_TIFF_XMP] = xmp
            extra["exif"] = tags.tobytes()
        img.save(dst, "TIFF", **extra)


_TIFF_XMP = 700
