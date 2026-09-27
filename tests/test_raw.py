"""Checks for reading RAW files without decoding them, Fujifilm RAF included.

    uv run pytest tests/test_raw.py

The files are built here: a RAF is a header giving the offset and length of
an embedded JPEG, and that is all a capture-time read looks at.
"""
from __future__ import annotations

import io
import struct
from datetime import datetime
from pathlib import Path

from PIL import Image

from picture_classifier import cameras, exifinfo, raw


def _jpeg(stamp: str | None = None, lens: str | None = None) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "FUJIFILM"
    exif[0x0110] = "X-T5"
    if stamp:
        exif.get_ifd(0x8769)[0x9003] = stamp
    if lens:
        exif.get_ifd(0x8769)[0xA434] = lens
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (90, 120, 150)).save(buf, "JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _raf(path: Path, jpeg: bytes) -> Path:
    """The parts of a RAF header a reader uses: the magic, and the JPEG's
    offset and length, big-endian at bytes 84 and 88."""
    head = bytearray(160)
    head[:16] = b"FUJIFILMCCD-RAW "
    head[16:28] = b"0201FF383501"
    struct.pack_into(">II", head, 84, len(head), len(jpeg))
    path.write_bytes(bytes(head) + jpeg + b"\x00" * 4096)   # raw data follows
    return path


def test_raf_capture_time_comes_from_the_embedded_jpeg(tmp_path) -> None:
    f = _raf(tmp_path / "DSCF0021.RAF", _jpeg("2022:11:20 15:38:30"))
    assert raw.is_raw(f)
    assert raw.head_capture_time(f) == datetime(2022, 11, 20, 15, 38, 30)
    # And the reader the scan and the wizard use answers from it, without libraw
    # (which could not open this file: there is no sensor data in it).
    assert raw.read_capture_time(f) == datetime(2022, 11, 20, 15, 38, 30)


def test_raf_without_a_time_says_none(tmp_path) -> None:
    f = _raf(tmp_path / "DSCF0022.RAF", _jpeg(None))
    assert raw.head_capture_time(f) is None


def test_neither_tiff_nor_raf_is_left_to_libraw(tmp_path) -> None:
    f = tmp_path / "IMG_0001.CR3"
    f.write_bytes(b"\x00\x00\x00\x18ftypcrx " + b"\x00" * 64)
    assert raw.head_capture_time(f) is None


def test_exif_text_loses_its_nul_padding(tmp_path) -> None:
    """Fujifilm pads LensModel to a fixed width with NULs, which reached the
    watermark and the photo info as a run of control characters."""
    f = tmp_path / "a.jpg"
    f.write_bytes(_jpeg("2022:11:20 15:38:30", "XF18mmF2 R" + "\x00" * 43))
    info = exifinfo.read(f)
    assert info["lens"] == "XF18mmF2 R"
    assert info["captured_at"] == "2022:11:20 15:38:30"


def test_fujifilm_bodies_read_alike() -> None:
    """Listed bodies print bare, unlisted ones gain the make: both X-T5 and
    X-S10 are common enough that they should read the same way."""
    for model in ("X-T5", "X-S10", "X-T30 II", "X100V", "GFX 100", "GFX50S II"):
        assert cameras.pretty_model(model, "FUJIFILM") == model
    assert cameras.pretty_model("X-PRO2", "FUJIFILM") == "X-Pro2"
