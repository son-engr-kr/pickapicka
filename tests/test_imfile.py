"""Image files open on any path, and nothing bypasses the module that makes
sure they do.

The first test only proves something on Windows, where cv2.imread cannot open
a path outside the ANSI code page; that is why CI runs the suite there too.
"""
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from picture_classifier import imfile
from picture_classifier.scoring import blur, exposure

PKG = Path(__file__).resolve().parents[1] / "src" / "picture_classifier"


def test_reads_and_writes_under_a_korean_folder(tmp_path):
    d = tmp_path / "바탕 화면" / "그리스"
    d.mkdir(parents=True)
    rgb = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)
    Image.fromarray(rgb).save(d / "사진.jpg", quality=95)

    img = imfile.imread(d / "사진.jpg")
    assert img is not None and img.shape == (48, 64, 3)
    # What scoring runs on each photo, with the path as scoring passes it.
    assert blur.blur_score(str(d / "사진.jpg")) >= 0.0
    assert 0.0 <= exposure.brightness(str(d / "사진.jpg")) <= 255.0

    imfile.imwrite(d / "결과.png", img)
    assert np.array_equal(imfile.imread(d / "결과.png"), img)


def test_same_pixels_as_imread_with_exif_rotation(tmp_path):
    img = Image.new("RGB", (60, 40))
    img.putdata([(x * 4, y * 6, 100) for y in range(40) for x in range(60)])
    exif = Image.Exif()
    exif[0x0112] = 6   # rotate 90° clockwise to display
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif.tobytes(), quality=92)
    for flag in (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED):
        assert np.array_equal(imfile.imread(path, flag), cv2.imread(str(path), flag))


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        imfile.imread(tmp_path / "nope.jpg")


def test_no_direct_opencv_file_io_in_the_package():
    direct = re.compile(r"\bcv2\.(imread|imwrite)\(")
    offenders = [
        f"{p.relative_to(PKG)}:{n}"
        for p in PKG.rglob("*.py") if p.name != "imfile.py"
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if direct.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, f"use imfile.imread / imfile.imwrite instead: {offenders}"
