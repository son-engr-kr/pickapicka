"""Checks for the EXIF and colour profile an export carries.

    uv run python tests/test_metadata.py

The source files are built here with the value types real cameras write:
rationals, byte-typed GPS fields, and a MakerNote of the size a Sony body
leaves (38 KB) or larger. The case that matters most is that the JPEG can be
saved at all, since a JPEG's EXIF segment is capped at 64 KB.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import ExifTags, Image
from PIL.TiffImagePlugin import IFDRational

B = ExifTags.Base
EXIF, GPS = ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo


def _camera_jpeg(path: Path, *, makernote: int = 38_000, icc: bytes | None = None,
                 orientation: int = 6) -> None:
    """A JPEG with the metadata a camera writes."""
    ex = Image.Exif()
    ex[B.Make], ex[B.Model], ex[B.Orientation] = "SONY", "ILCE-7CM2", orientation
    ex[B.Copyright], ex[B.Artist] = "(c) Someone", "Someone"
    ex[B.XResolution] = ex[B.YResolution] = IFDRational(350)
    ex[B.ResolutionUnit] = 2
    e = ex.get_ifd(EXIF)
    e[B.DateTimeOriginal] = "2026:08:02 17:00:18"
    e[B.ExposureTime] = IFDRational(1, 250)
    e[B.FNumber] = IFDRational(56, 10)
    e[B.ISOSpeedRatings] = 400
    e[B.FocalLength] = IFDRational(700, 10)
    e[B.LensModel] = "E 70-300mm F4.5-6.3 A047"
    e[B.ExifImageWidth], e[B.ExifImageHeight] = 7008, 4672
    e[B.MakerNote] = b"\x00" * makernote
    g = ex.get_ifd(GPS)
    g[0], g[1], g[3], g[5] = b"\x02\x03\x00\x00", "N", "W", b"\x00"
    g[2] = (IFDRational(42), IFDRational(20), IFDRational(1234, 100))
    g[4] = (IFDRational(71), IFDRational(5), IFDRational(5678, 100))
    extra = {"icc_profile": icc} if icc else {}
    Image.new("RGB", (64, 48), (120, 80, 40)).save(path, "JPEG", exif=ex.tobytes(), **extra)


def _reopen(exif: bytes | None, icc: bytes | None = None) -> Image.Image:
    buf = io.BytesIO()
    extra = {}
    if exif:
        extra["exif"] = exif
    if icc:
        extra["icc_profile"] = icc
    Image.new("RGB", (32, 20)).save(buf, "JPEG", **extra)
    return Image.open(io.BytesIO(buf.getvalue()))


def test_all_keeps_the_shot_and_rewrites_the_file_tags(tmp_path) -> None:
    from picture_classifier import metadata
    src = tmp_path / "a.jpg"
    _camera_jpeg(src)
    exif, icc = metadata.read_source(src, is_raw=False)

    out = _reopen(metadata.build_exif(exif, size=(2048, 1365), captured_at="",
                                      mode="all", srgb=False)).getexif()
    e = out.get_ifd(EXIF)

    assert out[B.Make] == "SONY" and out[B.Model] == "ILCE-7CM2"
    assert out[B.Copyright] == "(c) Someone"
    assert e[B.DateTimeOriginal] == "2026:08:02 17:00:18"
    assert float(e[B.FNumber]) == 5.6 and e[B.ISOSpeedRatings] == 400
    assert e[B.LensModel] == "E 70-300mm F4.5-6.3 A047"
    # The pixels are exported upright, so the flag that says "rotate me" goes.
    assert out[B.Orientation] == 1
    assert (e[B.ExifImageWidth], e[B.ExifImageHeight]) == (2048, 1365)
    assert out[B.Software].startswith("Picture Classifier ")
    assert B.MakerNote not in e
    g = out.get_ifd(GPS)
    assert g[1] == "N" and [float(v) for v in g[2]] == [42.0, 20.0, 12.34]


def test_a_makernote_too_big_for_a_jpeg_does_not_stop_the_export(tmp_path) -> None:
    """70 KB of MakerNote makes the EXIF block too long for a JPEG, and Pillow
    refuses to save it. Leaving the MakerNote out is what makes this work."""
    from picture_classifier import metadata
    src = tmp_path / "a.jpg"
    _camera_jpeg(src, makernote=38_000)
    exif, _ = metadata.read_source(src, is_raw=False)
    exif.get_ifd(EXIF)[B.MakerNote] = b"\x00" * 70_000

    built = metadata.build_exif(exif, size=(10, 10), captured_at="", mode="all", srgb=False)

    assert len(built) < 4_000
    _reopen(built)   # would raise "EXIF data is too long"


def test_no_location_drops_gps_only(tmp_path) -> None:
    from picture_classifier import metadata
    src = tmp_path / "a.jpg"
    _camera_jpeg(src)
    exif, _ = metadata.read_source(src, is_raw=False)

    out = _reopen(metadata.build_exif(exif, size=(10, 10), captured_at="",
                                      mode="no_location", srgb=False)).getexif()

    assert not out.get_ifd(GPS)
    assert out.get_ifd(EXIF)[B.DateTimeOriginal] == "2026:08:02 17:00:18"


def test_none_writes_nothing(tmp_path) -> None:
    from picture_classifier import metadata
    src = tmp_path / "a.jpg"
    _camera_jpeg(src)
    exif, _ = metadata.read_source(src, is_raw=False)
    assert metadata.build_exif(exif, size=(10, 10), captured_at="x",
                               mode="none", srgb=False) is None


def test_capture_time_fills_in_only_when_missing(tmp_path) -> None:
    """Sony RAW previews leave DateTimeOriginal out; the time libraw read is
    used then, and never over one the file does have."""
    from picture_classifier import metadata
    src = tmp_path / "a.jpg"
    _camera_jpeg(src)
    exif, _ = metadata.read_source(src, is_raw=False)
    kept = _reopen(metadata.build_exif(exif, size=(10, 10), captured_at="2001:01:01 00:00:00",
                                       mode="all", srgb=False)).getexif()
    assert kept.get_ifd(EXIF)[B.DateTimeOriginal] == "2026:08:02 17:00:18"

    filled = _reopen(metadata.build_exif(None, size=(10, 10), captured_at="2001:01:01 00:00:00",
                                         mode="all", srgb=True)).getexif()
    e = filled.get_ifd(EXIF)
    assert e[B.DateTimeOriginal] == "2001:01:01 00:00:00"
    assert e[B.ColorSpace] == 1


def test_the_source_profile_is_carried_and_none_is_not_invented(tmp_path) -> None:
    from picture_classifier import metadata
    tagged, bare = tmp_path / "t.jpg", tmp_path / "b.jpg"
    profile = metadata.srgb_profile()
    _camera_jpeg(tagged, icc=profile)
    _camera_jpeg(bare)

    assert metadata.read_source(tagged, is_raw=False)[1] == profile
    assert metadata.read_source(bare, is_raw=False)[1] is None


def _main() -> None:
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
