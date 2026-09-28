"""Person and part segmentation: the automatic masks.

Why this model. MediaPipe's multiclass selfie segmenter is Apache-2.0, 16 MB,
and returns six classes in one forward pass — background, hair, body skin, face
skin, clothes and accessories. That is very nearly the exact set of masks a
portrait retoucher reaches for, from one model rather than five.

Why ONNX rather than running the `.tflite` through mediapipe. `onnxruntime` is
a hard dependency of this project; `mediapipe` is an optional extra used only by
eye-open scoring. Converting once means the automatic masks work on a base
install, on every platform the app ships to, instead of only where a mediapipe
wheel exists. The conversion is `tf2onnx --tflite`, recorded in
`docs/roadmap-parity.md`, and the result is hosted as a release asset of this
repository — the same "download once into the platform cache, then offline
forever" shape as `scoring/objects.py` and insightface's buffalo_l.

What this module deliberately does not do: threshold. A mask wants a soft edge,
so `class_mask` returns the summed probability of the requested classes,
upscaled from the model's 256x256 output as a *probability* field. Upscaling a
thresholded mask instead would give a staircase edge that no amount of feathering
afterwards can repair, because the information is already gone.

Resolution independence comes free here, and it is worth being explicit about
why: the network always sees a 256x256 letterbox of the whole frame, whatever
resolution the caller is rendering at. So the alpha for a given photo is the same
field every time, and a thumbnail, the fit preview, a 1:1 window and the export
all agree — which is the contract the rest of the pipeline is built on.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import paths

# ----- the model ----------------------------------------------------------

MODEL_NAME = "selfie_multiclass_256x256.onnx"
MODEL_URL = (
    "https://github.com/son-engr-kr/pickapicka/releases/download/"
    "models-0.1.0/selfie_multiclass_256x256.onnx"
)
MODEL_SHA256 = "46532ed4d2f36a60038b729042d8f01acf4f6e7a6a0b016ddc0e6782c217ce66"
MODEL_SIZE = 16454469   # bytes, per the release asset
INPUT_SIZE = (256, 256)  # the exported graph is fixed at this size

_MODEL_DIR = paths.MODEL_DIR
_session = None
_SESSION_LOCK = threading.Lock()

# Index order is the model's own output channel order and must not be reordered.
CLASSES: tuple[str, ...] = (
    "background", "hair", "body-skin", "face-skin", "clothes", "accessories",
)

# What the editor actually offers. Nobody asks for "body skin union face skin" —
# they ask for the subject, or for skin, so the groups are the vocabulary and the
# six raw classes are an implementation detail.
CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "subject": ("hair", "body-skin", "face-skin", "clothes", "accessories"),
    "background": ("background",),
    "skin": ("body-skin", "face-skin"),
    "face": ("face-skin",),
    "hair": ("hair",),
    "clothes": ("clothes",),
}


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
    """Download the ONNX model on first use. Returns its path.

    The download lands in a temp file, is checked against the pinned SHA-256,
    and only then moved into place — an interrupted or tampered download can
    never leave a model the app would go on to trust.
    """
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
        raise RuntimeError(
            f"model checksum mismatch for {MODEL_NAME}: expected {MODEL_SHA256}, got {got}"
        )
    shutil.move(str(tmp), str(dest))
    return dest


def _get_session():
    global _session
    with _SESSION_LOCK:
        if _session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            _session = ort.InferenceSession(
                str(ensure_model()), so, providers=["CPUExecutionProvider"],
            )
    return _session


# ----- pre / post processing ----------------------------------------------

def _letterbox(rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Fit the frame into a 256x256 canvas without changing its proportions.

    Stretching to square would be cheaper and it is what the simplest MediaPipe
    path does, but a 3:2 landscape frame stretched to 1:1 presents the model with
    a person 1.5x too wide — a shape it never saw in training. Letterboxing costs
    some of the 256 px budget and keeps the geometry honest, which is also what
    `scoring/objects.py` already does for detection.

    Returns the canvas and the (x, y, w, h) box the image occupies inside it, so
    the probabilities can be cropped back out.
    """
    ih, iw = INPUT_SIZE
    h, w = rgb.shape[:2]
    ratio = min(ih / h, iw / w)
    nh, nw = max(1, int(round(h * ratio))), max(1, int(round(w * ratio)))
    ox, oy = (iw - nw) // 2, (ih - nh) // 2
    canvas = np.zeros((ih, iw, 3), dtype=np.float32)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, (ox, oy, nw, nh)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Turn the model's raw scores into probabilities over the class axis.

    The exported graph stops at the logits — the scores do not sum to 1 and are
    freely negative, which was worth checking rather than assuming, since summing
    raw logits for a class group would produce an "alpha" outside [0,1] that
    happened to look plausible in the middle of a region and wrong at its edge.
    Subtracting the row maximum first is the standard guard against overflow in
    `exp`; it does not change the result.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted, dtype=np.float32)
    return e / e.sum(axis=-1, keepdims=True)


def probabilities(rgb: np.ndarray) -> np.ndarray:
    """Per-class probability field for `rgb`, as (H, W, len(CLASSES)) float32.

    `rgb` is uint8 or float32 RGB and is taken to be the whole photo. The result
    is at the input's own height and width, so the caller can sum whichever
    classes it wants and use the result directly as an alpha.
    """
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected HxWx3 RGB, got {rgb.shape}"
    src = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else rgb
    canvas, (ox, oy, nw, nh) = _letterbox(np.clip(src, 0.0, 1.0))

    sess = _get_session()
    blob = canvas[None]   # NHWC, which is what a tflite export carries over
    out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]
    logits = np.asarray(out, dtype=np.float32)
    if logits.ndim == 4:
        logits = logits[0]
    # Guard the class-order assumption rather than trusting it silently: a model
    # swapped underneath this code would otherwise recolour hair as clothes.
    assert logits.shape[-1] == len(CLASSES), \
        f"model returned {logits.shape[-1]} classes, expected {len(CLASSES)}"
    probs = _softmax(logits)

    inside = probs[oy:oy + nh, ox:ox + nw]
    h, w = rgb.shape[:2]
    return cv2.resize(inside, (w, h), interpolation=cv2.INTER_LINEAR)


_CACHE_MAX = 6
_CACHE: "OrderedDict[tuple[Any, int, int], np.ndarray]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def probabilities_cached(rgb: np.ndarray, key: Any) -> np.ndarray:
    """`probabilities`, memoized on a caller-supplied key.

    One forward pass per photo is cheap but not free, and the editor asks for a
    mask again on every slider drag. `key` should identify the photo (its path is
    the obvious choice); the render size is folded in because the result is
    returned at the input's own size.
    """
    ck = (key, rgb.shape[0], rgb.shape[1])
    with _CACHE_LOCK:
        hit = _CACHE.get(ck)
        if hit is not None:
            _CACHE.move_to_end(ck)
            return hit
    probs = probabilities(rgb)
    with _CACHE_LOCK:
        _CACHE[ck] = probs
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return probs


def class_mask(rgb: np.ndarray, group: str, key: Any = None) -> np.ndarray:
    """Soft alpha in [0,1] for one of `CLASS_GROUPS`, as float32 (H, W).

    The summed probability *is* the alpha — no threshold, no morphology. A
    threshold would throw away exactly the information a mask edge is made of,
    and hair is the case that proves it: at the boundary the model is genuinely
    50/50 between hair and background, and that is the correct alpha for a strand
    thinner than a pixel.
    """
    assert group in CLASS_GROUPS, \
        f"unknown group {group!r}; expected one of {sorted(CLASS_GROUPS)}"
    probs = probabilities(rgb) if key is None else probabilities_cached(rgb, key)
    idx = [CLASSES.index(c) for c in CLASS_GROUPS[group]]
    alpha = probs[..., idx].sum(axis=2)
    return np.clip(alpha, 0.0, 1.0).astype(np.float32)
