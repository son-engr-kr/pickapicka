"""Stars and colour labels: kept across a re-score, and written into exports
where Lightroom (XMP) and Windows (EXIF) read them.

    uv run pytest tests/test_marks.py
"""
from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from picture_classifier import exporting, metadata, scorer

SONY_XMP = ("<?xpacket begin='﻿' id='W5M0MpCehiHzreSzNTczkc9d'?>\n"
            "<x:xmpmeta xmlns:x='adobe:ns:meta/' x:xmptk=''>\n"
            "<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>\n"
            " <rdf:Description rdf:about=''\n  xmlns:xmp='http://ns.adobe.com/xap/1.0/'>\n"
            "  <xmp:Rating>0</xmp:Rating>\n </rdf:Description>\n</rdf:RDF>\n</x:xmpmeta>\n"
            + " " * 2000 + "\n<?xpacket end='w'?>")


def _jpeg(tmp_path: Path, xmp: str | None = None) -> Path:
    rng = np.random.default_rng(1)
    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0110] = "ILCE-7CM2"
    Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)).save(
        buf, "JPEG", quality=92, exif=exif.tobytes(), xmp=xmp.encode("utf-8") if xmp else None)
    f = tmp_path / "DSC0001.JPG"
    f.write_bytes(buf.getvalue())
    return f


def _xmp_segments(data: bytes) -> list[str]:
    out, i = [], 2
    while data[i + 1] not in (0xDA, 0xD9):
        size = struct.unpack(">H", data[i + 2:i + 4])[0]
        if data[i + 1] == 0xE1 and data[i + 4:i + 4 + len(metadata.XMP_ID)] == metadata.XMP_ID:
            out.append(data[i + 4 + len(metadata.XMP_ID):i + 2 + size].decode("utf-8"))
        i += 2 + size
    return out


def _image_data(data: bytes) -> bytes:
    return data[data.index(b"\xff\xda"):]


def test_a_copy_without_xmp_gets_one_and_its_pixels_stay(tmp_path) -> None:
    f = _jpeg(tmp_path)
    before = f.read_bytes()
    metadata.mark_jpeg(f, 4, "green")
    after = f.read_bytes()
    assert _image_data(after) == _image_data(before)
    (packet,) = _xmp_segments(after)
    assert 'xmp:Rating="4"' in packet and 'xmp:Label="Green"' in packet
    assert Image.open(f).getexif()[0x0110] == "ILCE-7CM2"


def test_sonys_packet_is_edited_not_doubled(tmp_path) -> None:
    f = _jpeg(tmp_path, SONY_XMP)
    before = f.read_bytes()
    metadata.mark_jpeg(f, 5, None)
    after = f.read_bytes()
    assert _image_data(after) == _image_data(before)
    (packet,) = _xmp_segments(after)
    assert "<xmp:Rating>5</xmp:Rating>" in packet and "<xmp:Rating>0" not in packet


def test_an_attribute_rating_is_rewritten_and_a_label_added(tmp_path) -> None:
    f = _jpeg(tmp_path, metadata.build_xmp(2, None).decode("utf-8"))
    metadata.mark_jpeg(f, 3, "red")
    (packet,) = _xmp_segments(f.read_bytes())
    assert 'xmp:Rating="3"' in packet and 'xmp:Rating="2"' not in packet
    assert 'xmp:Label="Red"' in packet


def test_nothing_to_say_writes_nothing(tmp_path) -> None:
    assert metadata.build_xmp(0, None) is None
    f = _jpeg(tmp_path)
    before = f.read_bytes()
    metadata.mark_jpeg(f, 0, None)
    assert f.read_bytes() == before


def test_rendered_exports_carry_both(tmp_path) -> None:
    rgb = np.full((24, 32, 3), 120, np.uint8)
    src = Image.Exif()
    src[0x0110] = "X-T5"
    src.get_ifd(0x8769)[0x9003] = "2026:05:01 10:00:00"
    exif = metadata.build_exif(src, size=(32, 24), captured_at="", mode="all", srgb=False, rating=4)
    xmp = metadata.build_xmp(4, "blue")
    for fmt in ("jpeg", "tiff"):
        f = tmp_path / f"out.{fmt}"
        exporting.write(rgb, f, fmt=fmt, quality=90, exif=exif, icc=None, xmp=xmp)
        img = Image.open(f)
        tags = img.getexif()
        assert tags[0x0110] == "X-T5" and tags.get_ifd(0x8769)[0x9003] == "2026:05:01 10:00:00", fmt
        assert tags[0x4746] == 4 and tags[0x4749] == 75, fmt
        packet = img.info.get("xmp") if fmt == "jpeg" else bytes(img.tag_v2[700])
        assert b'xmp:Rating="4"' in packet and b'xmp:Label="Blue"' in packet, fmt


def test_a_rescore_keeps_stars_and_labels() -> None:
    prior = {"photos": [
        {"rel_path": "a.jpg", "decision": "pick", "decided_at": "t", "rating": 4, "label": "red"},
        {"rel_path": "b.jpg", "decision": None},
    ]}
    kept = scorer._existing_decisions(prior)
    assert kept["a.jpg"] == {"decision": "pick", "decided_at": "t", "rating": 4, "label": "red"}
    assert kept["b.jpg"] == {}
