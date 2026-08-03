"""RAW decode helpers (rawpy / libraw).

Kept separate from `editing` so the pipeline stays import-light and testable
without rawpy installed. RAW files aren't browser- or PIL-readable, so:
  - the grid thumbnail and the editor's decoded base come from a fast embedded
    preview (or a half-size decode);
  - scoring (blur/exposure/faces, all path + cv2.imread based) reads a cached
    preview JPEG written once per RAW at scan time.

`decode_raw` output is already oriented by libraw — never `exif_transpose` it
(unlike the embedded JPEG thumbnail, which does carry EXIF orientation).
"""
from __future__ import annotations

import hashlib
import io
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

RAW_EXTS = {
    ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".arw", ".sr2", ".srf",
    ".raf", ".rw2", ".orf", ".pef", ".srw", ".dng", ".raw", ".3fr",
    ".erf", ".kdc", ".mos", ".mrw", ".x3f",
}


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTS and not path.name.startswith("._")


def fit_within(arr: np.ndarray, max_edge: int) -> np.ndarray:
    """Downscale so the long edge is <= max_edge (no-op if already smaller)."""
    h, w = arr.shape[:2]
    if max(h, w) <= max_edge:
        return arr
    s = max_edge / max(h, w)
    return cv2.resize(arr, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def decode_raw(path: Path, half_size: bool = False) -> np.ndarray:
    """Full (or half) oriented RGB uint8 from a RAW file: camera white balance,
    8-bit sRGB, no auto-brightness (deterministic base so preview == export)."""
    import rawpy
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(
            use_camera_wb=True, no_auto_bright=True, output_bps=8,
            output_color=rawpy.ColorSpace.sRGB, half_size=half_size,
        )
    return np.ascontiguousarray(rgb)


def decode_thumb(path: Path, max_edge: int) -> np.ndarray:
    """Fast preview RGB: the embedded JPEG thumbnail when present (needs
    exif_transpose — it's a real JPEG), else a half-size RAW decode."""
    import rawpy
    try:
        with rawpy.imread(str(path)) as r:
            try:
                th = r.extract_thumb()
            except Exception:
                th = None
            if th is not None and th.format == rawpy.ThumbFormat.JPEG:
                im = ImageOps.exif_transpose(Image.open(io.BytesIO(th.data)).convert("RGB"))
                return fit_within(np.asarray(im), max_edge)
            if th is not None and th.format == rawpy.ThumbFormat.BITMAP:
                return fit_within(np.ascontiguousarray(th.data), max_edge)
    except Exception:
        pass
    return fit_within(decode_raw(path, half_size=True), max_edge)


def raw_cache_path(cache_root: Path, ident: str) -> Path:
    """Stable location of a RAW's cached preview JPEG (keyed by its identity)."""
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{digest}.jpg"


def ensure_cache(raw_path: Path, cache_root: Path, ident: str, max_edge: int = 2560) -> Path:
    """Write a preview JPEG for a RAW (once, refreshed when the RAW is newer) so
    path-based code — thumbs, blur, exposure, face detection — can read it."""
    dst = raw_cache_path(cache_root, ident)
    if dst.exists() and dst.stat().st_mtime >= raw_path.stat().st_mtime:
        return dst
    cache_root.mkdir(parents=True, exist_ok=True)
    Image.fromarray(decode_thumb(raw_path, max_edge)).save(dst, "JPEG", quality=92)
    return dst


def read_capture_time(path: Path) -> datetime | None:
    """Capture time from the RAW's embedded JPEG preview EXIF (best effort)."""
    import rawpy
    try:
        with rawpy.imread(str(path)) as r:
            try:
                th = r.extract_thumb()
            except Exception:
                th = None
        if th is not None and th.format == rawpy.ThumbFormat.JPEG:
            with Image.open(io.BytesIO(th.data)) as im:
                ifd = im.getexif().get_ifd(0x8769)
                raw = ifd.get(0x9003)  # DateTimeOriginal
            if raw:
                return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None
