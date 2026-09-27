"""Reading and writing image files with OpenCV, on any path.

`cv2.imread` and `cv2.imwrite` hand the path to the C runtime as a narrow
string, which Windows reads in the ANSI code page. A path with characters
outside it, such as a Korean folder name or a OneDrive "바탕 화면" (Desktop),
then fails to open and imread returns None. Python opens the file here instead
and OpenCV only sees the bytes. Decoding from memory gives the same pixels as
imread, EXIF orientation included.

Use these, never cv2.imread / cv2.imwrite: tests/test_imfile.py fails on any
direct call left in the package.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """`cv2.imread`, except that a missing file raises. None still means the
    file is there but is not an image OpenCV can decode."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, img: np.ndarray, params: list[int] | tuple[int, ...] = ()) -> None:
    """`cv2.imwrite`; the format comes from the file extension, as there."""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img, list(params))
    assert ok, f"could not encode an image as {ext!r} for {path}"
    buf.tofile(path)
