"""AI fill: a heal whose pixels are made by a generative inpainting model.

Why a model, and which
----------------------
`healing`'s heal is diffusion (`cv2.inpaint`): it extends the colour round a
hole inward, and on skin that is a smudge. Measured on 60 holes cut into five
real faces (cheek, forehead, under the eye, across the jaw line; 1%, 3% and 6%
of the face width), with the original pixels as the truth: the fine texture
inside the fill (the band finer than a 1 px blur, over the same in a ring round
the hole) came to 0.31 of the skin's own for cv2.inpaint, 0.95 for this model,
and 0.96 for the truth; the RMSE against the truth was 10.2 and 7.5 levels.
The model also carries an edge through the hole (a jaw line, a crease) where
the diffusion leaves a star of smeared spokes.

The model is MI-GAN (Sargsyan et al., "MI-GAN: A Simple Baseline for Image
Inpainting on Mobile Devices", ICCV 2023), the 512 px Places2 one, as the
authors' own ONNX pipeline. It is the only inpainting model found whose code
AND weights are under a permissive licence: the repository's LICENSE and
LICENSE-WEIGHTS are both MIT (Picsart AI Research), so the default path
carries no restriction to the photographer, as `docs/roadmap-parity.md`
requires. It is 28 MB, and a fill took 150 to 200 ms on an M5 Pro's CPU for
strokes from 11 to 300 px in radius on a 19 MP frame (the model always works
at 512 px; what grows with the stroke is only the crop round it).

The pipeline takes a uint8 RGB image and a uint8 mask (255 known, 0 to fill),
crops a square round the hole (the hole's box plus 256 px, at least 512),
resizes it to 512, inpaints and pastes the result back. So a hole up to about
256 px across is filled at its own resolution, and a larger one at less: the
fill of a 600 px hole is made at 512 / 856 of the photo's resolution and comes
out softer than the skin beside it.

What is stored
--------------
A fill is pixels, not a parameter, so it cannot live in the edit's dict. It is
made once, on request, from the full-resolution frame as the lens corrections
leave it (before any other repair: a fill does not depend on the heals made
before it, so editing one of those does not make it stale), and kept as a PNG
named by the hash of its pixels. The edit's heal operation carries that name
and where the patch sits in the frame, so the render composites it through the
heal's own feathered alpha at any resolution, exactly where a diffusion heal
would have written. Deterministic: the model has no noise input, and the same
photo and stroke give the same patch.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.request
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import healing as healing_mod
from . import paths

MODEL_NAME = "migan_pipeline_v2.onnx"
# The authors' own upload. Before a release, mirror it to this repository's
# release assets (as the segmentation model is) so a fresh install does not
# depend on someone else's link; the pin below is what makes that safe.
MODEL_URL = "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx"
MODEL_SHA256 = "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b"
MODEL_SIZE = 28079181   # bytes
MODEL_LICENCE = "MIT (Picsart AI Research): code and weights"

# How much of the photo round the hole the model is handed: the pipeline crops
# its own square of the hole's box plus 256 px, and this is enough to hold it.
_CONTEXT = 256

_MODEL_DIR = paths.MODEL_DIR
_session = None
_SESSION_LOCK = threading.Lock()


# ----- model file ---------------------------------------------------------

def model_path() -> Path:
    return _MODEL_DIR / MODEL_NAME


def is_model_ready() -> bool:
    p = model_path()
    return p.is_file() and p.stat().st_size == MODEL_SIZE


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(progress_cb: Callable[[int, int], None] | None = None) -> Path:
    """Download the model on first use, checked against its pinned SHA-256
    before it is moved into place (as `segment.ensure_model`)."""
    dest = model_path()
    if is_model_ready():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp, tmp.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or MODEL_SIZE)
        done = 0
        while chunk := resp.read(256 * 1024):
            out.write(chunk)
            done += len(chunk)
            if progress_cb is not None:
                progress_cb(done, total)
    got = _digest(tmp)
    if got != MODEL_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"model checksum mismatch for {MODEL_NAME}: expected {MODEL_SHA256}, got {got}")
    shutil.move(str(tmp), str(dest))
    return dest


def _get_session():
    global _session
    with _SESSION_LOCK:
        if _session is None:
            import onnxruntime as ort
            _session = ort.InferenceSession(str(ensure_model()), providers=["CPUExecutionProvider"])
        return _session


# ----- making a fill ------------------------------------------------------

def inpaint(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """The model on a uint8 RGB array and a boolean hole of the same size."""
    assert rgb.dtype == np.uint8 and rgb.ndim == 3, "inpaint wants uint8 RGB"
    assert hole.shape == rgb.shape[:2], "the hole must cover the image"
    img = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    mask = np.where(hole, 0, 255).astype(np.uint8)[None, None]
    out = _get_session().run(None, {"image": img, "mask": mask})[0][0]
    return np.ascontiguousarray(out.transpose(1, 2, 0))


def make_fill(frame: np.ndarray, op: dict[str, Any]) -> tuple[np.ndarray, list[float]]:
    """The patch an AI heal writes, made from the WHOLE uint8 `frame`.

    Returns the patch (uint8 RGB over the operation's written box, at the
    frame's resolution) and that box as [x0, y0, x1, y1] fractions of the
    frame. `op` is a heal or spot operation as `healing.normalize_op` gives it,
    without its fill.
    """
    assert frame.dtype == np.uint8 and frame.ndim == 3, "make_fill wants the uint8 frame"
    h, w = frame.shape[:2]
    (x0, y0, x1, y1), hole = healing_mod.region_mask(op, w, h)
    assert x1 > x0 and y1 > y0, "the stroke misses the photo"
    m = max(_CONTEXT, x1 - x0, y1 - y0)
    cx0, cy0 = max(0, x0 - m), max(0, y0 - m)
    cx1, cy1 = min(w, x1 + m), min(h, y1 + m)
    crop = np.ascontiguousarray(frame[cy0:cy1, cx0:cx1])
    holes = np.zeros(crop.shape[:2], bool)
    holes[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0] = hole
    out = inpaint(crop, holes)
    patch = np.ascontiguousarray(out[y0 - cy0:y1 - cy0, x0 - cx0:x1 - cx0])
    return patch, [x0 / w, y0 / h, x1 / w, y1 / h]


def fill_id(patch: np.ndarray, box: list[float]) -> str:
    """The name a patch is kept under: the hash of its pixels and its place."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(patch).tobytes())
    h.update(repr((patch.shape, [round(v, 9) for v in box])).encode())
    return h.hexdigest()
