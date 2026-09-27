"""What an exported file says about itself: its EXIF and its colour profile.

Every rendered export used to be written bare. That was every RAW, since a RAW
is always rendered, and every edited photo. Capture time, camera, lens,
exposure and GPS were dropped, so a delivery that mixed edited and untouched
frames no longer sorted by when it was shot.

The EXIF is rebuilt tag by tag from a fixed list rather than copied wholesale.
Two things about copying whole were found on real files:

- Pillow cannot always re-serialise what it parsed. A Sony A7C II JPEG raises
  `struct.error` from `Exif.tobytes()` once its thumbnail IFD has been
  loaded, and other bodies will have their own oddities. A fixed list of
  standard tags, assigned into a fresh `Exif`, gets Pillow's declared types
  for every one of them, so what is written is always writable.
- The MakerNote is vendor-private and addressed by offsets into the original
  file, which are wrong wherever it lands in a new one. It is also the bulk of
  the block (38 KB of that Sony's 49 KB), and a JPEG's APP1 segment cannot
  exceed 64 KB. It is left out, as is the embedded thumbnail, which would
  show the unedited frame.

What is kept is everything that describes the shot, plus copyright and author.
File-structure tags are rewritten to describe the new file: orientation is 1
because the pixels are already upright, and the pixel dimensions are the
exported size.

The colour profile is whatever the original carried, since the pipeline never
converts colour. The numbers it outputs are still in the source's space, so
that source's profile is the correct label for them. A RAW is decoded straight
to sRGB, so it gets an sRGB profile. A file that came without one is left
without one rather than assumed to be sRGB.
"""
from __future__ import annotations

import io
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import ExifTags, Image, ImageCms

from . import __version__

Base = ExifTags.Base

MetadataMode = Literal["all", "no_location", "none"]

SOFTWARE = f"Picture Classifier {__version__}"

# Carried from the original, in IFD0. Software and Orientation are written
# fresh instead; the resolution tags are the print size a lab will read.
_IFD0_KEEP = (
    Base.ImageDescription, Base.Make, Base.Model, Base.DateTime, Base.Artist,
    Base.Copyright, Base.XResolution, Base.YResolution, Base.ResolutionUnit,
)

# Carried from the Exif sub-IFD: the exposure, the lens, and the times.
_EXIF_KEEP = (
    Base.ExifVersion, Base.ExposureTime, Base.FNumber, Base.ExposureProgram,
    Base.ISOSpeedRatings, Base.SensitivityType, Base.RecommendedExposureIndex,
    Base.ISOSpeed, Base.DateTimeOriginal, Base.DateTimeDigitized,
    Base.OffsetTime, Base.OffsetTimeOriginal, Base.OffsetTimeDigitized,
    Base.SubsecTime, Base.SubsecTimeOriginal, Base.SubsecTimeDigitized,
    Base.ShutterSpeedValue, Base.ApertureValue, Base.BrightnessValue,
    Base.ExposureBiasValue, Base.MaxApertureValue, Base.SubjectDistance,
    Base.MeteringMode, Base.LightSource, Base.Flash, Base.FocalLength,
    Base.ColorSpace, Base.SensingMethod, Base.FileSource, Base.SceneType,
    Base.CustomRendered, Base.ExposureMode, Base.WhiteBalance,
    Base.DigitalZoomRatio, Base.FocalLengthIn35mmFilm, Base.SceneCaptureType,
    Base.GainControl, Base.Contrast, Base.Saturation, Base.Sharpness,
    Base.SubjectDistanceRange, Base.CameraOwnerName, Base.BodySerialNumber,
    Base.LensSpecification, Base.LensMake, Base.LensModel, Base.LensSerialNumber,
)

_SRGB = 1   # ExifColorSpace value for sRGB


@lru_cache(maxsize=1)
def srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def read_source(path: Path, is_raw: bool) -> tuple[Image.Exif | None, bytes | None]:
    """EXIF and ICC profile of the file a photo came from.

    A RAW is not PIL-readable. Its EXIF comes from the JPEG preview it embeds,
    which is where `exifinfo` reads the strings from as well. Its profile is
    sRGB, the space `raw.decode_raw` renders into.
    """
    if is_raw:
        import rawpy
        with rawpy.imread(str(path)) as r:
            try:
                thumb = r.extract_thumb()
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                return None, srgb_profile()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            return None, srgb_profile()
        with Image.open(io.BytesIO(thumb.data)) as im:
            return im.getexif(), srgb_profile()
    with Image.open(path) as im:
        return im.getexif(), im.info.get("icc_profile")


def build_exif(
    src: Image.Exif | None,
    *,
    size: tuple[int, int],
    captured_at: str,
    mode: MetadataMode,
    srgb: bool,
    rating: int = 0,
) -> bytes | None:
    """EXIF for an export of `size` (width, height), or None for `mode="none"`.

    `captured_at` is the capture time in EXIF's "YYYY:MM:DD HH:MM:SS" form, as
    `exifinfo` caches it. It fills DateTimeOriginal when the source has none,
    which is the case for Sony RAW, whose preview leaves the tag out; libraw
    supplies it instead.
    """
    if mode == "none":
        return None
    out = Image.Exif()
    exif_ifd = out.get_ifd(ExifTags.IFD.Exif)
    if src is not None:
        for tag in _IFD0_KEEP:
            if tag in src:
                out[tag] = src[tag]
        src_exif = src.get_ifd(ExifTags.IFD.Exif)
        for tag in _EXIF_KEEP:
            if tag in src_exif:
                exif_ifd[tag] = src_exif[tag]
        if mode == "all":
            src_gps = src.get_ifd(ExifTags.IFD.GPSInfo)
            if src_gps:
                out.get_ifd(ExifTags.IFD.GPSInfo).update(src_gps)
    if Base.DateTimeOriginal not in exif_ifd and captured_at:
        exif_ifd[Base.DateTimeOriginal] = captured_at
    if srgb:
        exif_ifd[Base.ColorSpace] = _SRGB
    out[Base.Software] = SOFTWARE
    out[Base.Orientation] = 1
    if rating:
        # Windows' own pair; XMP (build_xmp) is what Lightroom and Bridge read.
        out[_EXIF_RATING] = rating
        out[_EXIF_RATING_PERCENT] = _RATING_PERCENT[rating]
    exif_ifd[Base.ExifImageWidth], exif_ifd[Base.ExifImageHeight] = size
    return out.tobytes()


# ----- stars and colour labels -------------------------------------------

_EXIF_RATING = 0x4746
_EXIF_RATING_PERCENT = 0x4749
_RATING_PERCENT = {1: 1, 2: 25, 3: 50, 4: 75, 5: 99}   # what Windows writes for 1-5 stars
# Lightroom's label names, which is what its XMP holds and other apps match on.
LABELS = {"red": "Red", "yellow": "Yellow", "green": "Green", "blue": "Blue", "purple": "Purple"}
XMP_ID = b"http://ns.adobe.com/xap/1.0/\x00"
_XMP_NS = "http://ns.adobe.com/xap/1.0/"


def build_xmp(rating: int, label: str | None) -> bytes | None:
    """An XMP packet carrying a star rating and a colour label, or None when
    there is neither to say."""
    attrs = (f' xmp:Rating="{rating}"' if rating else "") + (f' xmp:Label="{LABELS[label]}"' if label else "")
    if not attrs:
        return None
    return ('<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>'
            '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            f'<rdf:Description rdf:about="" xmlns:xmp="{_XMP_NS}"{attrs}/>'
            '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>').encode("utf-8")


def _set_property(packet: str, name: str, value: str) -> str:
    """Set xmp:`name` in an XMP packet, in whichever form the file already
    uses (Sony writes <xmp:Rating>0</xmp:Rating>, others an attribute), or as
    a Description of its own when the packet does not have it."""
    element = re.compile(rf"(<xmp:{name}>)[^<]*(</xmp:{name}>)")
    if element.search(packet):
        return element.sub(lambda m: m.group(1) + value + m.group(2), packet, count=1)
    attribute = re.compile(rf"""(xmp:{name}=)(["'])[^"']*\2""")
    if attribute.search(packet):
        return attribute.sub(lambda m: f"{m.group(1)}{m.group(2)}{value}{m.group(2)}", packet, count=1)
    end = packet.index("</rdf:RDF>")
    return (packet[:end] + f'<rdf:Description rdf:about="" xmlns:xmp="{_XMP_NS}" xmp:{name}="{value}"/>'
            + packet[end:])


def mark_jpeg(path: Path, rating: int, label: str | None) -> None:
    """Write a star rating and colour label into a JPEG's XMP in place, the
    image data untouched. For an export that is a copy of the original: the
    original is never passed here. An existing XMP packet is edited, since a
    second one would be ignored; a file without one gets one."""
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8", f"not a JPEG: {path}"
    i = 2
    while True:
        assert data[i] == 0xFF, f"a JPEG segment was expected at byte {i} of {path}"
        marker = data[i + 1]
        if marker in (0xDA, 0xD9):
            xmp_at = None
            break
        size = struct.unpack(">H", data[i + 2:i + 4])[0]
        if marker == 0xE1 and data[i + 4:i + 4 + len(XMP_ID)] == XMP_ID:
            xmp_at = (i, size)
            break
        i += 2 + size
    if xmp_at is None:
        packet = build_xmp(rating, label)
        if packet is None:
            return
        body = XMP_ID + packet
        segment = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
        path.write_bytes(data[:2] + segment + data[2:])
        return
    start, size = xmp_at
    packet = data[start + 4 + len(XMP_ID):start + 2 + size].decode("utf-8")
    if rating:
        packet = _set_property(packet, "Rating", str(rating))
    if label:
        packet = _set_property(packet, "Label", LABELS[label])
    body = XMP_ID + packet.encode("utf-8")
    assert len(body) + 2 <= 0xFFFF, f"the XMP of {path} would outgrow its JPEG segment"
    segment = b"\xff\xe1" + struct.pack(">H", len(body) + 2) + body
    path.write_bytes(data[:start] + segment + data[start + 2 + size:])
