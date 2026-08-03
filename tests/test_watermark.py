"""Unit checks for camera naming, EXIF extraction and watermark rendering.

    uv run python tests/test_watermark.py
"""
from __future__ import annotations

import numpy as np

from picture_classifier import cameras, editing, exifinfo, watermark


# ----- camera names -------------------------------------------------------

def test_sony_a7c2_typography() -> None:
    """The body this feature was asked for, spelled the way Sony spells it."""
    assert cameras.pretty_model("ILCE-7CM2", "SONY") == "α7C II"
    assert cameras.pretty_model("ilce-7cm2", "sony") == "α7C II", "case-insensitive"


def test_known_bodies_map_to_marketing_names() -> None:
    cases = {
        ("ILCE-7M4", "SONY"): "α7 IV",
        ("ILCE-7RM5", "SONY"): "α7R V",
        ("ILCE-1", "SONY"): "α1",
        ("NIKON Z 9", "NIKON CORPORATION"): "Z 9",
        ("NIKON Z 6_2", "NIKON CORPORATION"): "Z 6II",
        ("X-T5", "FUJIFILM"): "X-T5",
        ("Canon EOS R5", "Canon"): "EOS R5",
        ("DC-S5M2", "Panasonic"): "LUMIX S5II",
        ("OM-1MarkII", "OM Digital Solutions"): "OM-1 Mark II",
    }
    for (model, make), want in cases.items():
        assert cameras.pretty_model(model, make) == want, model


def test_unknown_body_still_reads_sensibly() -> None:
    # Unknown model gets the make prefixed…
    assert cameras.pretty_model("XYZ-99", "SONY") == "Sony XYZ-99"
    # …but never twice.
    assert cameras.pretty_model("Canon EOS 80D", "Canon") == "Canon EOS 80D"
    assert cameras.pretty_model("", "Sony") == ""
    assert cameras.pretty_model(None) == ""


def test_preset_list_is_unique_and_sorted() -> None:
    presets = cameras.preset_list()
    assert len(presets) > 50
    names = [p["name"] for p in presets]
    assert len(names) == len(set(names)), "duplicate display names"
    assert names == sorted(names, key=str.lower)
    assert "α7C II" in names


# ----- EXIF formatting ----------------------------------------------------

def test_value_formatting() -> None:
    assert exifinfo.format_focal(85.0) == "85mm"
    assert exifinfo.format_focal(23.5) == "23.5mm"
    assert exifinfo.format_focal(None) == ""
    assert exifinfo.format_aperture(1.4) == "f/1.4"
    assert exifinfo.format_aperture(2.8) == "f/2.8"
    assert exifinfo.format_aperture(8.0) == "f/8"
    assert exifinfo.format_shutter(1 / 250) == "1/250s"
    assert exifinfo.format_shutter(0.5) == "1/2s"
    assert exifinfo.format_shutter(2.0) == "2s"
    assert exifinfo.format_iso(400) == "ISO 400"
    assert exifinfo.format_iso(None) == ""


def test_reads_exif_round_trip(tmp_path) -> None:
    from PIL import Image
    img = Image.new("RGB", (200, 120), (90, 90, 90))
    exif = img.getexif()
    exif[0x010F] = "SONY"
    exif[0x0110] = "ILCE-7CM2"
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "2026:08:02 14:30:11"
    sub[0x920A] = 85.0
    sub[0x829D] = 1.4
    sub[0x829A] = 1 / 250
    sub[0x8827] = 400
    sub[0xA434] = "FE 85mm F1.4 GM"
    path = tmp_path / "shot.jpg"
    img.save(path, "JPEG", exif=exif)

    info = exifinfo.read(path)
    assert info["camera"] == "α7C II"
    assert info["lens"] == "FE 85mm F1.4 GM"
    assert (info["focal"], info["aperture"], info["shutter"], info["iso_text"]) == (
        "85mm", "f/1.4", "1/250s", "ISO 400")
    assert not exifinfo.is_empty(info)


def test_missing_exif_is_empty_not_an_error(tmp_path) -> None:
    from PIL import Image
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (64, 64), (10, 10, 10)).save(path, "JPEG")
    info = exifinfo.read(path)
    assert exifinfo.is_empty(info)
    assert info["camera"] == ""
    assert exifinfo.is_empty(exifinfo.read(tmp_path / "does-not-exist.jpg"))


# ----- templates ----------------------------------------------------------

_VALUES = {
    "name": "Hyoungseo Son", "camera": "α7C II", "lens": "FE 85mm F1.4 GM",
    "focal": "85mm", "aperture": "f/1.4", "shutter": "1/250s", "iso": "ISO 400",
    "date": "2026-08-02", "time": "14:30:11", "file": "DSC01234",
}


def test_template_fills_tokens() -> None:
    assert watermark.render_line("{name}", _VALUES) == "Hyoungseo Son"
    assert watermark.render_line("{camera} · {lens}", _VALUES) == "α7C II · FE 85mm F1.4 GM"
    assert watermark.render_line("© {name}", _VALUES) == "© Hyoungseo Son"
    assert watermark.render_line("", _VALUES) == ""


def test_empty_tokens_take_their_separator_with_them() -> None:
    """A lens-less shot must not print 'α7C II ·  · ISO 400'."""
    sparse = dict(_VALUES, lens="", shutter="", iso="")
    assert watermark.render_line("{camera} · {lens}", sparse) == "α7C II"
    assert watermark.render_line("{focal} · {aperture} · {shutter} · {iso}", sparse) == "85mm · f/1.4"
    blank = {k: "" for k in _VALUES}
    assert watermark.render_line("{camera} · {lens}", blank) == ""


def test_unknown_token_is_dropped() -> None:
    assert watermark.render_line("{name} {nope}", _VALUES) == "Hyoungseo Son"


# ----- normalization ------------------------------------------------------

def test_normalize_clamps_and_defaults() -> None:
    w = watermark.normalize({"style": "nonsense", "position": "middle",
                             "size": 9999, "opacity": -5, "margin": -20,
                             "color": "red", "name": "x" * 500})
    assert w["style"] in watermark.STYLES
    assert w["position"] in watermark.POSITIONS
    assert w["size"] == 300 and w["opacity"] == 0 and w["margin"] == 0
    assert w["color"] == "#ffffff"
    assert len(w["name"]) == 160
    assert watermark.normalize("garbage")["style"] == watermark.DEFAULT_WATERMARK["style"]


def test_neutrality() -> None:
    assert watermark.is_neutral(None)
    assert watermark.is_neutral({"enabled": False, "line1": "{name}"})
    assert watermark.is_neutral({"enabled": True, "opacity": 0, "line1": "{name}"})
    assert watermark.is_neutral({"enabled": True, "line1": "", "line2": "", "line3": ""})
    assert not watermark.is_neutral({"enabled": True, "line1": "{name}"})


# ----- rendering ----------------------------------------------------------

def _photo(h: int = 400, w: int = 600) -> np.ndarray:
    return np.full((h, w, 3), 90, dtype=np.uint8)


_META = {"camera": "α7C II", "lens": "FE 85mm F1.4 GM", "focal": "85mm",
         "aperture": "f/1.4", "shutter": "1/250s", "iso_text": "ISO 400",
         "captured_at": "2026:08:02 14:30:11"}


def test_every_style_marks_the_photo() -> None:
    img = _photo()
    for style in watermark.STYLES:
        out = watermark.render(img, {"enabled": True, "style": style, "name": "Someone"}, _META)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out, img), style


def test_position_moves_the_mark() -> None:
    img = _photo()
    base = {"enabled": True, "style": "minimal", "name": "Someone"}
    top = watermark.render(img, {**base, "position": "top-left"}, _META)
    bottom = watermark.render(img, {**base, "position": "bottom-right"}, _META)
    h = img.shape[0]
    changed_top = (top != img).any(axis=(1, 2))
    changed_bottom = (bottom != img).any(axis=(1, 2))
    assert changed_top[: h // 2].any() and not changed_top[h // 2:].any()
    assert changed_bottom[h // 2:].any() and not changed_bottom[: h // 2].any()


def test_neutral_watermark_returns_the_input_untouched() -> None:
    img = _photo()
    assert watermark.render(img, None, _META) is img
    assert watermark.render(img, {"enabled": False}, _META) is img


def test_scales_with_the_frame() -> None:
    """The same watermark must cover the same fraction of a small and a large
    render — that is what keeps the preview honest about the export."""
    wm = {"enabled": True, "style": "minimal", "name": "Someone"}
    small = watermark.render(_photo(400, 600), wm, _META)
    large = watermark.render(_photo(800, 1200), wm, _META)
    frac_small = (small != 90).any(axis=2).mean()
    frac_large = (large != 90).any(axis=2).mean()
    assert abs(frac_small - frac_large) < frac_small * 0.35, f"{frac_small} vs {frac_large}"


def test_roi_render_matches_the_full_frame_window() -> None:
    """Zoomed to 1:1 the watermark must land where the full render puts it."""
    img = _photo(800, 1200)
    x0, y0, pw, ph = 300, 400, 600, 400
    roi = (x0 / 1200, y0 / 800, pw / 1200, ph / 800)
    for style in watermark.STYLES:
        wm = {"enabled": True, "style": style, "name": "Someone",
              "position": "bottom-center"}
        full = watermark.render(img, wm, _META)
        crop = watermark.render(img[y0:y0 + ph, x0:x0 + pw], wm, _META, roi=roi)
        diff = np.abs(crop.astype(int) - full[y0:y0 + ph, x0:x0 + pw].astype(int))
        assert diff.max() == 0, f"{style} drifted in the ROI render: {diff.max()}"


def test_camera_override_beats_exif() -> None:
    img = _photo()
    wm = {"enabled": True, "style": "minimal", "line1": "{camera}",
          "line2": "", "line3": "", "camera": "Hasselblad 500C/M"}
    assert watermark.lines_for(watermark.normalize(wm), _META) == ["Hasselblad 500C/M"]
    wm_auto = {**wm, "camera": ""}
    assert watermark.lines_for(watermark.normalize(wm_auto), _META) == ["α7C II"]
    assert not np.array_equal(watermark.render(img, wm, _META), img)


# ----- integration with the edit pipeline ---------------------------------

def test_edit_carries_the_watermark() -> None:
    img = _photo()
    wm = {"enabled": True, "style": "minimal", "name": "Someone"}
    assert not editing.is_neutral({"watermark": wm})
    assert editing.is_neutral({"watermark": {"enabled": False}})
    # A watermark-only edit still renders…
    assert not np.array_equal(editing.render(img, {"watermark": wm}, meta=_META), img)
    # …but thumbnails opt out, where it would only be noise.
    assert np.array_equal(
        editing.render(img, {"watermark": wm}, meta=_META, with_watermark=False), img)
    # Grading still happens on the thumbnail path.
    assert not np.array_equal(
        editing.render(img, {"exposure": 0.5, "watermark": wm}, meta=_META,
                       with_watermark=False), img)


def test_normalize_drops_a_watermark_that_prints_nothing() -> None:
    assert editing.normalize({"watermark": {"enabled": False}})["watermark"] is None
    kept = editing.normalize({"watermark": {"enabled": True, "name": "A"}})["watermark"]
    assert kept is not None and kept["name"] == "A"
    # The hash has to notice it, or a cached thumb would outlive the change.
    h1 = editing.edit_hash({"watermark": {"enabled": True, "name": "A"}})
    h2 = editing.edit_hash({"watermark": {"enabled": True, "name": "B"}})
    assert h1 and h1 != h2


def _main() -> None:
    import inspect
    import tempfile
    from pathlib import Path
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
