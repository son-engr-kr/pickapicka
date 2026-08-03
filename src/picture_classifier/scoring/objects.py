"""COCO object detection via YOLOX-tiny ONNX, run on onnxruntime.

Why YOLOX: it is Apache-2.0 and Megvii publish the ONNX export directly, so it
drops onto the `onnxruntime` this project already depends on — no torch, and no
AGPL entanglement the way YOLOv8/v11 would bring.

The model file (~20 MB) is fetched on first use into
`~/.picture-classifier/models/`, the same "download once, reuse forever" shape
as insightface's buffalo_l. Everything after that is offline.

The exported graph is the raw head: boxes come out in per-stride grid units and
have to be decoded here (`_decode`), then NMS'd. That mirrors YOLOX's own
`demo/ONNXRuntime` reference implementation.
"""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Handy presets for the project's "what am I shooting?" setting.
VEHICLE_CLASSES: tuple[str, ...] = ("car", "truck", "bus", "motorcycle")
CLASS_PRESETS: dict[str, tuple[str, ...]] = {
    "vehicle": VEHICLE_CLASSES,
    "person": ("person",),
    "pet": ("dog", "cat", "bird", "horse"),
    "bike": ("bicycle", "motorcycle"),
}

def resolve_classes(spec: str | Iterable[str] | None) -> list[str]:
    """Turn a preset name ('vehicle'), a comma-separated string ('car,truck') or
    a list into a validated list of COCO class names. Unknown names are dropped."""
    if not spec:
        return []
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    out: list[str] = []
    for raw in names:
        name = str(raw).strip().lower()
        if not name:
            continue
        for c in CLASS_PRESETS.get(name, (name,)):
            if c in COCO_CLASSES and c not in out:
                out.append(c)
    return out


MODEL_NAME = "yolox_tiny.onnx"
MODEL_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
    "0.1.1rc0/yolox_tiny.onnx"
)
MODEL_SHA256 = "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"
MODEL_SIZE = 20219662  # bytes, per the release asset
INPUT_SIZE = (416, 416)  # yolox_tiny/nano are exported at 416
STRIDES = (8, 16, 32)

_CONF_DEFAULT = 0.35
_NMS_IOU = 0.45
_MODEL_DIR = Path.home() / ".picture-classifier" / "models"

_session = None


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
    if _session is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _session = ort.InferenceSession(
            str(ensure_model()), so, providers=["CPUExecutionProvider"],
        )
    return _session


# ----- pre / post processing ----------------------------------------------

def _preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Letterbox onto a 114-gray canvas, keeping aspect. YOLOX's standard export
    takes raw BGR 0..255 (no mean/std), so there is no normalization here."""
    ih, iw = INPUT_SIZE
    h, w = img.shape[:2]
    ratio = min(ih / h, iw / w)
    nh, nw = int(round(h * ratio)), int(round(w * ratio))
    canvas = np.full((ih, iw, 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    blob = canvas.transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(blob), ratio


def _decode(pred: np.ndarray) -> np.ndarray:
    """Turn the raw head into absolute boxes in input-image pixels.

    `pred` is (n_anchors, 85): [cx, cy, w, h, obj, 80 class scores], where cx/cy
    are offsets within their cell and w/h are log-scale, both in stride units.
    """
    grids, expanded = [], []
    for stride in STRIDES:
        gh, gw = INPUT_SIZE[0] // stride, INPUT_SIZE[1] // stride
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        expanded.append(np.full((gh * gw, 1), stride))
    grid = np.concatenate(grids, axis=0)
    strides = np.concatenate(expanded, axis=0)
    out = pred.copy()
    out[:, 0:2] = (pred[:, 0:2] + grid) * strides
    out[:, 2:4] = np.exp(pred[:, 2:4]) * strides
    return out


def _to_boxes(decoded: np.ndarray, ratio: float, conf: float,
              keep: set[int] | None) -> list[tuple[int, float, list[int]]]:
    """Confidence-filter, NMS per class, and map back to original pixels."""
    scores = decoded[:, 4:5] * decoded[:, 5:]          # objectness x class prob
    cls_ids = scores.argmax(axis=1)
    cls_scores = scores.max(axis=1)
    mask = cls_scores >= conf
    if keep is not None:
        mask &= np.isin(cls_ids, list(keep))
    if not mask.any():
        return []

    cxcywh = decoded[mask, :4] / ratio
    cls_ids, cls_scores = cls_ids[mask], cls_scores[mask]
    xywh = np.empty_like(cxcywh)
    xywh[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    xywh[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    xywh[:, 2:] = cxcywh[:, 2:]

    out: list[tuple[int, float, list[int]]] = []
    for cid in np.unique(cls_ids):
        sel = cls_ids == cid
        boxes = xywh[sel].tolist()
        confs = cls_scores[sel].astype(np.float32).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes, confs, conf, _NMS_IOU)
        for i in np.array(idxs).reshape(-1):
            x, y, w, h = boxes[int(i)]
            out.append((int(cid), float(confs[int(i)]),
                        [int(round(x)), int(round(y)), int(round(w)), int(round(h))]))
    out.sort(key=lambda o: o[1], reverse=True)
    return out


# ----- public API ---------------------------------------------------------

def detect(img_path: str, classes: Iterable[str] | None = None,
           conf: float = _CONF_DEFAULT) -> tuple[list[dict[str, Any]], int, int]:
    """Detect objects in `img_path`.

    Returns (objects, original_width, original_height). Each object is
    `{"cls": name, "score": float, "bbox_xywh": [x, y, w, h]}` in original-image
    pixels. `classes` restricts detection to those COCO names (None = all).
    """
    img = cv2.imread(img_path)
    assert img is not None, f"failed to read {img_path}"
    h, w = img.shape[:2]
    keep = None
    if classes is not None:
        keep = {COCO_CLASSES.index(c) for c in classes if c in COCO_CLASSES}
        if not keep:
            return [], w, h

    blob, ratio = _preprocess(img)
    session = _get_session()
    pred = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    found = _to_boxes(_decode(pred), ratio, conf, keep)

    objects: list[dict[str, Any]] = []
    for cid, score, (bx, by, bw, bh) in found:
        # Clamp to the frame: a box may hang off the edge after decoding.
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(w, bx + bw), min(h, by + bh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        objects.append({
            "cls": COCO_CLASSES[cid],
            "score": round(score, 4),
            "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
        })
    return objects, w, h
