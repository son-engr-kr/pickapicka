"""HDR bracket detection (EXIF-only) and exposure-fusion merge.

Detects auto-exposure-bracket (AEB) runs among JPEGs from EXIF alone, then
fuses each run into one image with OpenCV exposure fusion (Mertens). No new
dependency — opencv-python and pillow are already required.

A bracket is a run of 3-9 consecutive frames that:
  - are close in time, with the allowed gap scaling with the longer of the two
    frames' exposure times (a long exposure forces a multi-second gap);
  - cycle through a set of exposures and then restart — the sweep repeating
    (EV-compensation, or shutter speed) marks the boundary to the next bracket.
Detection is EXIF-only; it never opens the image pixels.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

# EXIF tag ids inside the Exif sub-IFD (0x8769). Numeric ids are used directly
# so the code does not depend on PIL's tag-name table.
_EXIF_IFD = 0x8769
_DATETIME_ORIGINAL = 0x9003
_SUBSEC_ORIGINAL = 0x9291
_EXPOSURE_TIME = 0x829A
_EXPOSURE_BIAS = 0x9204
_ISO = 0x8827

# Synthetic rel_path prefix for a merged result. These entries are addressed
# against the HDR output directory instead of the JPEG root.
HDR_PREFIX = "__hdr__"

# ----- bracket-detection tuning ------------------------------------------
MIN_BRACKET = 3
MAX_BRACKET = 9
BASE_GAP_S = 2.0           # allowed inter-frame gap at negligible exposure time
GAP_EXPOSURE_FACTOR = 2.5  # ...plus this multiple of the longer exposure time.
#   Scaled off the *longer* of the two frames because cameras timestamp frames
#   inconsistently (start vs end of exposure), and long-exposure noise
#   reduction shoots a second dark frame that ~doubles the time to the next.
_EV_STEP_PER_STOP = 3      # round exposures to 1/3-stop bins for sweep matching

# ----- merge "look" (post-fusion grade) ----------------------------------
# Mertens fusion is faithful but flat; this grade gives the punchy, bright
# real-estate finish. Every field is a no-op at its neutral value.
DEFAULT_LOOK: dict[str, float] = {
    "shadows": 0.22,      # lift dark areas — the shadowless real-estate look
    "brightness": 1.05,   # >1 lifts midtones (gamma) — clean, bright tone
    "clarity": 1.6,       # CLAHE clip limit — local mid-tone contrast / punch
    "saturation": 1.15,   # HSV saturation multiplier
    "sharpen": 0.45,      # unsharp-mask amount
}


# ----- EXIF ---------------------------------------------------------------

def read_exposure_exif(image_path: Path) -> dict[str, Any] | None:
    """Best-effort read of capture time + exposure tags. None if the file has
    no usable DateTimeOriginal (without it a frame cannot be ordered)."""
    try:
        with Image.open(image_path) as img:
            ifd = img.getexif().get_ifd(_EXIF_IFD)
    except (OSError, ValueError, AttributeError):
        return None
    raw_dt = ifd.get(_DATETIME_ORIGINAL)
    if not raw_dt:
        return None
    try:
        dt = datetime.strptime(str(raw_dt).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    subsec = ifd.get(_SUBSEC_ORIGINAL)
    if subsec is not None:
        digits = str(subsec).strip()
        if digits.isdigit():
            dt = dt + timedelta(seconds=float("0." + digits))

    exp = ifd.get(_EXPOSURE_TIME)
    bias = ifd.get(_EXPOSURE_BIAS)
    iso = ifd.get(_ISO)
    if isinstance(iso, (tuple, list)):
        iso = iso[0] if iso else None
    return {
        "exposure_time": float(exp) if exp else None,
        "ev_bias": float(bias) if bias is not None else None,
        "iso": float(iso) if iso else None,
        "time": dt,
    }


# ----- bracket detection --------------------------------------------------

def _segment_run(run: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split a time-run into individual brackets by detecting where the AEB
    sweep restarts — i.e. an exposure value the current bracket already used
    repeats. EV-compensation is the primary signal (it resets to 0 each
    bracket); shutter speed is the fallback when EV-comp is left fixed.

    Returns [] for a run shot at a single setting (a burst, not a bracket)."""
    biases = [r["ev_bias"] for r in run]
    times = [r["exposure_time"] for r in run]
    have_bias = [b for b in biases if b is not None]

    if len(have_bias) >= 2 and len(set(have_bias)) > 1:
        def signature(r: dict[str, Any]) -> Any:
            return round((r["ev_bias"] or 0.0) * _EV_STEP_PER_STOP)
    elif (
        sum(bool(t) for t in times) >= 2
        and len({round(math.log2(t) * _EV_STEP_PER_STOP) for t in times if t}) > 1
    ):
        def signature(r: dict[str, Any]) -> Any:
            t = r["exposure_time"]
            return round(math.log2(t) * _EV_STEP_PER_STOP) if t else None
    else:
        return []  # one exposure setting throughout — a burst, not a bracket

    segments: list[list[dict[str, Any]]] = [[run[0]]]
    used = {signature(run[0])}
    for rec in run[1:]:
        sig = signature(rec)
        if sig in used:
            segments.append([rec])
            used = {sig}
        else:
            segments[-1].append(rec)
            used.add(sig)
    return segments


def detect_brackets(jpeg_root: Path, rel_paths: Iterable[str]) -> list[list[str]]:
    """Group `rel_paths` into AEB brackets. Returns a list of member-lists, each
    of length 3-9. Frames not part of any bracket are simply omitted."""
    recs: list[dict[str, Any]] = []
    for rp in rel_paths:
        info = read_exposure_exif(jpeg_root / rp)
        if info is None:
            continue
        info["rel_path"] = rp
        recs.append(info)
    recs.sort(key=lambda r: r["time"])

    # Time-runs: a new run starts whenever the gap is larger than the dead time
    # the two frames' exposures could explain. The *longer* exposure is used
    # because the timestamp may mark the start or the end of the exposure.
    runs: list[list[dict[str, Any]]] = []
    for rec in recs:
        if runs:
            prev = runs[-1][-1]
            gap = (rec["time"] - prev["time"]).total_seconds()
            longer = max(prev["exposure_time"] or 0.0, rec["exposure_time"] or 0.0)
            if gap <= BASE_GAP_S + GAP_EXPOSURE_FACTOR * longer:
                runs[-1].append(rec)
                continue
        runs.append([rec])

    brackets: list[list[str]] = []
    for run in runs:
        if len(run) < MIN_BRACKET:
            continue
        for seg in _segment_run(run):
            if MIN_BRACKET <= len(seg) <= MAX_BRACKET:
                brackets.append([r["rel_path"] for r in seg])
    return brackets


# ----- identity & paths ---------------------------------------------------

def bracket_id(members: list[str]) -> str:
    """Stable id derived from the member set — survives re-scoring, changes
    when membership changes (so a decision never carries to a different shot)."""
    digest = hashlib.sha1("\n".join(sorted(members)).encode("utf-8")).hexdigest()
    return digest[:10]


def merged_filename(members: list[str]) -> str:
    """On-disk name of the merged JPEG (lives in the HDR output directory)."""
    return f"{Path(members[0]).stem}-hdr-{bracket_id(members)}.jpg"


def merged_rel_path(members: list[str]) -> str:
    """Synthetic rel_path used as the merged result's identity in the db."""
    return f"{HDR_PREFIX}/{merged_filename(members)}"


def is_hdr_rel(rel_path: str) -> bool:
    return rel_path.startswith(HDR_PREFIX + "/")


def base_frame(members: list[str], jpeg_root: Path) -> str:
    """The member shot at (closest to) neutral exposure — the '0 EV' frame a
    merged result is compared against. Falls back to the middle frame when no
    EV-compensation is recorded."""
    best, best_bias = None, None
    for m in members:
        info = read_exposure_exif(jpeg_root / m)
        if info is None or info["ev_bias"] is None:
            continue
        bias = abs(info["ev_bias"])
        if best_bias is None or bias < best_bias:
            best, best_bias = m, bias
    return best if best is not None else members[len(members) // 2]


# ----- merge --------------------------------------------------------------

def fuse(member_paths: list[Path], max_edge: int | None = None) -> np.ndarray:
    """Align the bracket frames and exposure-fuse them (Mertens). Returns an
    RGB uint8 array. `max_edge`, if set, downscales the inputs first — used to
    make previews fast.

    Mertens fusion produces a ready-to-view LDR image directly — no camera
    response curve, exposure times, or tone-mapping needed.
    """
    assert len(member_paths) >= 2, "a bracket needs at least 2 frames"
    imgs: list[np.ndarray] = []
    for p in member_paths:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_edge and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            imgs.append(np.ascontiguousarray(np.asarray(im)))

    # AlignMTB requires identical dimensions; crop to the common size.
    h = min(a.shape[0] for a in imgs)
    w = min(a.shape[1] for a in imgs)
    imgs = [a[:h, :w] for a in imgs]

    cv2.createAlignMTB().process(imgs, imgs)
    fused = cv2.createMergeMertens().process(imgs)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def grade(img: np.ndarray, look: dict[str, float] | None) -> np.ndarray:
    """Apply a real-estate 'look' to a fused RGB uint8 image: lifted shadows
    (the shadowless feel), a gamma brightness lift, CLAHE local contrast,
    saturation and unsharp sharpening. Each step is a no-op at its neutral
    value, so an empty look returns the image unchanged."""
    look = look or {}
    shadows = float(look.get("shadows", 0.0))
    brightness = float(look.get("brightness", 1.0))
    clarity = float(look.get("clarity", 0.0))
    saturation = float(look.get("saturation", 1.0))
    sharpen = float(look.get("sharpen", 0.0))
    out = np.ascontiguousarray(img)

    # Tone: lift shadows toward the 'no deep shadows' look, then a gentle gamma
    # brightening — both folded into one 256-entry lookup table.
    if shadows > 1e-3 or abs(brightness - 1.0) > 1e-3:
        x = np.linspace(0.0, 1.0, 256)
        x = x + shadows * (1.0 - x) ** 2
        x = np.clip(x, 0.0, 1.0) ** (1.0 / max(brightness, 1e-3))
        out = np.clip(x * 255.0, 0, 255).astype(np.uint8)[out]

    # Local contrast — CLAHE on the L channel of LAB.
    if clarity > 1e-3:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(
            clipLimit=clarity, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Saturation.
    if abs(saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Crispness — unsharp mask.
    if sharpen > 1e-3:
        blur = cv2.GaussianBlur(out, (0, 0), 1.6)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)

    return out


def merge_bracket(
    member_paths: list[Path],
    out_path: Path,
    look: dict[str, float] | None = None,
) -> None:
    """Align + exposure-fuse the bracket, apply the real-estate look, save JPEG.
    `look=None` uses DEFAULT_LOOK."""
    out = grade(fuse(member_paths), DEFAULT_LOOK if look is None else look)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path, "JPEG", quality=95)
