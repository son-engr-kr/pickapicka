"""Checks for how an export is written: names, sizes, formats, and copying.

    uv run python tests/test_exporting.py

The copy rule matters most. Copying an original is lossless and fast, but it
ignores every setting, so it must only happen when no setting would have
changed the file. Otherwise a "remove location" export would quietly ship GPS.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def _settings(**over):
    base = dict(format="jpeg", quality=95, long_edge=None, metadata="all",
                name_template="{name}")
    return SimpleNamespace(**{**base, **over})


# ----- file names ---------------------------------------------------------

def test_every_token_renders() -> None:
    from pickapicka import exporting
    name = exporting.render_name(
        "{project}_{scene}_{date}_{time}_{seq}_{name}", name="DSC0001", seq=7,
        width=3, captured_at="2026:08:02 17:00:18", scene="Scene_1", project="wedding")
    assert name == "wedding_Scene_1_2026-08-02_170018_007_DSC0001"


def test_no_capture_time_leaves_date_and_time_empty() -> None:
    from pickapicka import exporting
    assert exporting.render_name("{date}{name}", name="a", seq=1, width=3,
                                 captured_at="", scene="", project="") == "a"


def test_seq_is_padded_to_sort() -> None:
    from pickapicka import exporting
    assert exporting.seq_width(12) == 3
    assert exporting.seq_width(1234) == 4


def test_bad_templates_are_refused_with_a_reason() -> None:
    from pickapicka import exporting
    for bad, says in (("", "empty"), ("{nmae}", "unknown token {nmae}"),
                      ("{name", "unmatched"), ("name}", "unmatched")):
        try:
            exporting.check_template(bad)
        except ValueError as exc:
            assert says in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} was accepted")
    exporting.check_template("{date}-{seq} {name}")


# ----- when the original is the export ------------------------------------

def test_an_untouched_jpeg_at_default_settings_is_copied() -> None:
    from pickapicka import exporting
    assert exporting.can_copy(Path("a.JPG"), needs_render=False, settings=_settings())


def test_any_setting_that_changes_the_file_forces_a_write() -> None:
    from pickapicka import exporting
    src = Path("a.jpg")
    assert not exporting.can_copy(src, needs_render=True, settings=_settings())
    for over in ({"format": "tiff"}, {"long_edge": 2048},
                 {"metadata": "no_location"}, {"metadata": "none"}):
        assert not exporting.can_copy(src, needs_render=False, settings=_settings(**over)), over
    # A PNG exported as JPEG has to be converted.
    assert not exporting.can_copy(Path("a.png"), needs_render=False, settings=_settings())


# ----- size and format ----------------------------------------------------

def test_resize_shrinks_the_long_edge_and_never_enlarges() -> None:
    from pickapicka import exporting
    img = np.zeros((400, 600, 3), np.uint8)
    assert exporting.resize(img, 300).shape[:2] == (200, 300)
    assert exporting.resize(img, 1000) is img
    assert exporting.resize(img, None) is img


def test_both_formats_carry_exif_and_profile() -> None:
    from pickapicka import exporting, metadata
    rgb = np.full((20, 30, 3), 128, np.uint8)
    exif = metadata.build_exif(None, size=(30, 20), captured_at="2026:01:02 03:04:05",
                               mode="all", srgb=True)
    for fmt in ("jpeg", "tiff"):
        buf = io.BytesIO()
        exporting.write(rgb, buf, fmt=fmt, quality=90, exif=exif, icc=metadata.srgb_profile())
        img = Image.open(io.BytesIO(buf.getvalue()))
        assert img.format == {"jpeg": "JPEG", "tiff": "TIFF"}[fmt]
        assert img.size == (30, 20)
        assert img.info.get("icc_profile") == metadata.srgb_profile()
        dto = img.getexif().get_ifd(0x8769).get(0x9003)
        assert dto == "2026:01:02 03:04:05", (fmt, dto)


def test_tiff_is_lossless() -> None:
    from pickapicka import exporting
    rgb = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    buf = io.BytesIO()
    exporting.write(rgb, buf, fmt="tiff", quality=95, exif=None, icc=None)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(buf.getvalue()))), rgb)


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
