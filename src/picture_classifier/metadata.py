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
    exif_ifd[Base.ExifImageWidth], exif_ifd[Base.ExifImageHeight] = size
    return out.tobytes()
