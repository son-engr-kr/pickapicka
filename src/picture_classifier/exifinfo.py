"""Shooting metadata: what the watermark stamps and the viewer shows.

One dict per photo, read once at scoring time and cached in the db. RAW files
are not PIL-readable, but their embedded JPEG preview carries the same EXIF —
the trick `raw.read_capture_time` already relies on.

Values are stored both raw (numbers) and preformatted (`focal`, `aperture`, …)
so the watermark, the UI and any export all print them identically.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image

from . import cameras

_EXIF_IFD = 0x8769
_TAG = {name: tag for tag, name in ExifTags.TAGS.items()}

EMPTY: dict[str, Any] = {
    "make": "", "model": "", "camera": "", "lens": "",
    "focal_mm": None, "fnumber": None, "iso": None, "exposure_s": None,
    "focal": "", "aperture": "", "shutter": "", "iso_text": "",
    "captured_at": "",
}


def _num(value: Any) -> float | None:
    """EXIF rationals arrive as Fraction/IFDRational/tuple depending on the tag."""
    if value is None:
        return None
    try:
        if isinstance(value, tuple) and len(value) == 2:
            return float(value[0]) / float(value[1] or 1)
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def format_focal(mm: float | None) -> str:
    if not mm:
        return ""
    return f"{mm:.0f}mm" if abs(mm - round(mm)) < 0.05 else f"{mm:.1f}mm"


def format_aperture(f: float | None) -> str:
    if not f:
        return ""
    # f/1.4 keeps the decimal, f/8 does not — the way lenses are actually named.
    return f"f/{f:.1f}".rstrip("0").rstrip(".") if f < 10 else f"f/{f:.0f}"


def format_shutter(seconds: float | None) -> str:
    if not seconds:
        return ""
    if seconds >= 1.0:
        return f"{seconds:.0f}s" if abs(seconds - round(seconds)) < 0.05 else f"{seconds:.1f}s"
    # Sub-second speeds are written as a fraction, the way a camera dial reads.
    return f"1/{round(1 / seconds)}s"


def format_iso(iso: int | None) -> str:
    return f"ISO {iso}" if iso else ""


def _from_pil(img: Image.Image) -> dict[str, Any]:
    info = dict(EMPTY)
    exif = img.getexif()
    if not exif:
        return info
    ifd = exif.get_ifd(_EXIF_IFD) or {}

    make = str(exif.get(_TAG["Make"], "") or "").strip()
    model = str(exif.get(_TAG["Model"], "") or "").strip()
    lens = str(ifd.get(_TAG.get("LensModel"), "") or "").strip()
    if not lens:
        lens = str(exif.get(0xA434, "") or "").strip()  # LensModel in IFD0 on some bodies

    focal = _num(ifd.get(_TAG["FocalLength"]))
    fnum = _num(ifd.get(_TAG["FNumber"]))
    exposure = _num(ifd.get(_TAG["ExposureTime"]))
    iso_raw = ifd.get(_TAG["ISOSpeedRatings"]) or ifd.get(0x8833)
    if isinstance(iso_raw, (list, tuple)):
        iso_raw = iso_raw[0] if iso_raw else None
    try:
        iso = int(iso_raw) if iso_raw else None
    except (TypeError, ValueError):
        iso = None

    info.update(
        make=cameras.pretty_make(make),
        model=model,
        camera=cameras.pretty_model(model, make),
        lens=" ".join(lens.split()),
        focal_mm=focal, fnumber=fnum, iso=iso, exposure_s=exposure,
        focal=format_focal(focal),
        aperture=format_aperture(fnum),
        shutter=format_shutter(exposure),
        iso_text=format_iso(iso),
        captured_at=str(ifd.get(_TAG["DateTimeOriginal"], "") or "").strip(),
    )
    return info


def read(path: Path) -> dict[str, Any]:
    """Shooting info for a JPEG/PNG. Missing or unreadable EXIF gives EMPTY."""
    try:
        with Image.open(path) as img:
            return _from_pil(img)
    except (OSError, ValueError, AttributeError, KeyError):
        return dict(EMPTY)


def read_raw(path: Path) -> dict[str, Any]:
    """Shooting info for a RAW, via its embedded JPEG preview."""
    import rawpy
    try:
        with rawpy.imread(str(path)) as r:
            try:
                thumb = r.extract_thumb()
            except Exception:
                thumb = None
        if thumb is not None and thumb.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(thumb.data)) as img:
                return _from_pil(img)
    except Exception:
        pass
    return dict(EMPTY)


def read_any(path: Path, is_raw: bool = False) -> dict[str, Any]:
    return read_raw(path) if is_raw else read(path)


def is_empty(info: dict[str, Any] | None) -> bool:
    return not info or not any(
        info.get(k) for k in ("camera", "lens", "focal", "aperture", "shutter", "iso_text")
    )
