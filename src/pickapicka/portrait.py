"""Portrait retouching: skin, teeth and eyes, and the face geometry they stand on.

What a face needs that a frame-wide slider cannot give it
---------------------------------------------------------
`editing`'s texture slider already smooths skin at negative values, and it is
the wrong tool for a portrait in two ways. It is sized to the frame, so a face
filling a headshot and a face in a group photo get the same radius, which is
nothing on the first and a smear on the second. And it cannot tell skin from
eyes, lips, brows or a beard, so turning it down far enough to even out a cheek
softens everything a viewer looks at first.

So every radius here is a fraction of the face's own width, and the smoothing
is confined to that face's skin with the eyes, brows and mouth cut out.

The skin and the features
-------------------------
The skin comes from the same MediaPipe multiclass segmenter the automatic masks
use (`segment`), run on a crop around each face rather than on the frame. On
the whole frame a face is thirty-odd pixels of the model's 256 and the mask
comes back blocky. On a crop with head-and-shoulders context, which is what the
selfie model was trained on, the edge follows the jaw. Its face-skin class
still covers the eyes, brows and lips. Those are cut out with polygons from
InsightFace's 106-point landmark model, which ships in the buffalo_l pack the
app already downloads for face grouping, so nothing new is fetched. The index
groups below were read off a plot of the model's output on a real face.

The smoothing
-------------
Frequency separation with an edge-aware low band:

    texture = I - G(I, s_t)          what is finer than pores
    low     = G(I, s_t)
    even    = guided(low, radius r, eps)
    out     = low + a * (even - low) + texture

`texture` is left alone, so pores and grain survive and the skin does not go
to plastic. `even` is He, Sun and Tang's guided filter (ECCV 2010), self-guided
on luma, which averages where the local variance is under `eps` and keeps
edges where it is over it. So a blotch or a fine line is evened out and a
nostril or the edge of the jaw is not. The constants were chosen by eye on a
643 px face at full resolution: s_t at 0.6% of the face width kept the pores
that 0.25% lost, r at 2.8% evened the blotches without flattening the cheek's
shape, and eps at 0.03 squared left the nostrils crisp.

On that face the blotches measured 0.014 (standard deviation of the band
between the two scales, on a cheek at 0.33 luma) against 0.016 for the pores.
A brighter exposure of the same skin scales both up, so an eps fixed in
absolute terms would smooth a well-exposed face less than an underexposed one.
eps is therefore a contrast relative to the skin's own mean luma, as `analyze`
measures it over the whole face-skin alpha: 0.223 on that face, so the 0.03
chosen by eye is 13.5%. Measured on the same cheek at full strength, the band
finer than s_t keeps 97% of its amplitude.

Teeth and eyes
--------------
Both are whitening, of different things, in CIELAB: the pixels that are the
thing (bright for the region, and not saturated) are pulled towards neutral in
a and b and lifted in L. The region is the inner-mouth polygon for teeth and
the eye hulls for eyes. "Bright for the region" is a ramp between two
percentiles of L inside it, measured once by `analyze` on the whole frame and
stored, because measured at render time on a 1:1 window showing half a mouth
it would be a different number. On a real smile the teeth ramp picked the
upper teeth and left the shadowed lower ones and the gums alone. Some warmth is
kept (a third of the chroma for teeth), because fully neutral teeth read as
grey.

What "not the target" means differs. Teeth are told apart from lips, gums and
tongue by redness (a*), not by chroma: those three are red, and teeth are
yellow. A chroma limit would have spared exactly the teeth that most need
whitening. Measured on a real smile, the teeth sat at a* 6, b* 21, chroma 22,
in line with the dental shade guides (VITA A2 is about L 77, a 1.5, b 20). For
eyes a chroma limit is the right test: the whites are near-neutral, while
eyelid skin and the coloured reflections in glasses are not.

Blemishes
---------
`find_blemishes` does not retouch anything itself. It returns spot heals for
the healing module, which the editor adds to the photo's list, so every one is
visible and can be removed. A blemish is a spot that is darker or redder than
the skin around it, at a size between 0.6% and 1.6% of the face width, found as
a peak of the difference of Gaussians in L and in a*. Its strength is measured
against the robust spread of the same band over that face's skin, so pores and
noise set the floor. The first cut found folds, glasses rims, nostrils, eye
bags and the edge of the jaw on two real, clear-skinned faces. Three changes
took that down to two small dark spots on the cheeks, which look real, and one
blue reflection of a light in a lens, which is not. The changes: the spot must
be round (both Hessian eigenvalues of one sign, the smaller over 0.4 of the
larger, which a crease or a rim is not); the nose and a wide zone round the
eyes are left out; the skin is eroded by 3% of the face width; and a spot
bluer than the skin (b* more than 6 below its median) is a reflection, since
skin gone wrong goes darker or redder. The threshold of 6 is where the last
false one on clear skin went. What is not measured is recall on real acne,
since neither face had any; the tests show it finds planted spots of a
visible contrast.

Everything is local to a box around each face, so the cost follows the faces
and not the megapixels, and a window render gets identical pixels given
`padding` around it.
"""
from __future__ import annotations

import math
import threading
from typing import Any

import cv2
import numpy as np
from .sliders import tenth

# ----- schema -------------------------------------------------------------

DEFAULT_PORTRAIT: dict[str, Any] = {
    "smooth": 0,       # 0..100, skin smoothing
    "teeth": 0,        # 0..100, teeth whitening
    "eyes": 0,         # 0..100, whitening the whites of the eyes
}
_RANGES = {"smooth": (0, 100), "teeth": (0, 100), "eyes": (0, 100)}

FULL_ROI = (0.0, 0.0, 1.0, 1.0)

# InsightFace 2d106det index groups, read off its output on a real face.
EYE_L, EYE_R = tuple(range(33, 43)), tuple(range(87, 97))
BROW_L, BROW_R = tuple(range(43, 52)), tuple(range(97, 106))
MOUTH = tuple(range(52, 72))
# The inner lip line, in order round the opening: what is inside is teeth, gums,
# tongue and the dark of the mouth.
MOUTH_INNER = (65, 66, 62, 70, 69, 57, 60, 54)
# How far each feature's hull is grown before it is cut out of the skin, as a
# fraction of its own size: eyes get the most, for lashes and the lid crease.
_FEATURES = ((EYE_L, 0.30), (EYE_R, 0.30), (BROW_L, 0.15), (BROW_R, 0.15), (MOUTH, 0.10))
_FEATURE_FEATHER = 0.012   # the cut's soft edge, as a fraction of the face width

_TEX_SIGMA = 0.006         # texture split, fraction of the face width
_EVEN_RADIUS = 0.028       # guided-filter radius, fraction of the face width
_EVEN_CONTRAST = 0.135    # evened below this contrast, relative to the skin's mean

# Whitening, per target: the L percentiles the "bright for the region" ramp
# runs between, the chroma past which a pixel is not the target, how much
# chroma is kept at full strength, and the L lift. Tuned by eye on real faces.
_WHITEN = {
    "teeth": {"pct": (35, 70), "red_max": 20.0, "keep": 0.35, "lift": 8.0},
    "eyes": {"pct": (50, 85), "chroma_max": 28.0, "keep": 0.30, "lift": 6.0},
}
_WHITEN_FEATHER = 0.004    # the region's soft edge, fraction of the face width

# Analysis. A face smaller than this is not worth retouching and its landmarks
# are not reliable enough to cut features out by.
DETECT_LONG_EDGE = 1600
MIN_FACE_FRAC = 0.04       # of the frame's long edge
MIN_SCORE = 0.5
_CONTEXT = 0.6             # crop margin for segmentation, fraction of the face size
_SKIN_EDGE = 384           # the stored skin alpha's long edge


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp, or None when nothing would change (the house contract)."""
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_PORTRAIT)
    for key, (lo, hi) in _RANGES.items():
        try:
            out[key] = tenth(min(hi, max(lo, float(raw.get(key, 0) or 0))))
        except (TypeError, ValueError):
            out[key] = 0
    return None if is_neutral(out) else out


def is_neutral(params: dict[str, Any] | None) -> bool:
    return not params or all(not params.get(k) for k in DEFAULT_PORTRAIT)


# ----- analysis (the caller's job: heavy, cache it per photo) --------------

_APP = None
_APP_LOCK = threading.Lock()


def _face_app():
    global _APP
    with _APP_LOCK:
        if _APP is not None:
            return _APP
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"],
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _APP = app
        return _APP


def analyze(rgb: np.ndarray) -> list[dict[str, Any]]:
    """Every retouchable face in a WHOLE uint8 frame.

    Each is {box, crop, landmarks, skin, skin_luma}: `box` and `crop` are
    [x, y, w, h] and `landmarks` is 106 [x, y] pairs, all as fractions of the
    frame, so they mean the same thing on a thumbnail, the preview and the
    export. `skin` is the face-skin alpha over `crop`, at most `_SKIN_EDGE` on
    its long side, and `skin_luma` its mean luma, which sets how much contrast
    counts as a blotch.
    """
    from . import segment
    assert rgb.dtype == np.uint8 and rgb.ndim == 3, "analyze wants the uint8 frame"
    h, w = rgb.shape[:2]
    k = min(1.0, DETECT_LONG_EDGE / max(h, w))
    small = rgb if k >= 1.0 else cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    found = _face_app().get(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    faces = []
    for f in sorted(found, key=lambda f: (f.bbox[0], f.bbox[1])):
        x1, y1, x2, y2 = (float(v) / k for v in f.bbox)
        fw, fh = x2 - x1, y2 - y1
        if f.det_score < MIN_SCORE or max(fw, fh) < MIN_FACE_FRAC * max(w, h):
            continue
        m = _CONTEXT * max(fw, fh)
        cx0, cy0 = max(0, int(x1 - m)), max(0, int(y1 - m))
        cx1, cy1 = min(w, int(math.ceil(x2 + m))), min(h, int(math.ceil(y2 + m)))
        crop = np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1])
        skin = segment.probabilities(crop)[..., segment.CLASSES.index("face-skin")]
        luma = _luma(crop.astype(np.float32) / 255.0)
        skin_luma = float((luma * skin).sum() / max(float(skin.sum()), 1e-6))
        s = min(1.0, _SKIN_EDGE / max(skin.shape))
        if s < 1.0:
            skin = cv2.resize(skin, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        lm = f.landmark_2d_106 / k
        ramps = {what: _region_ramp(rgb, _whiten_polys(what, lm), _WHITEN[what]["pct"])
                 for what in _WHITEN}
        faces.append({
            "box": [x1 / w, y1 / h, fw / w, fh / h],
            "crop": [cx0 / w, cy0 / h, (cx1 - cx0) / w, (cy1 - cy0) / h],
            "landmarks": (lm / [w, h]).tolist(),
            "skin": skin.astype(np.float32),
            "skin_luma": skin_luma,
            "ramps": ramps,
        })
    return faces


# ----- applying -----------------------------------------------------------

def _whiten_polys(what: str, lm: np.ndarray) -> list[np.ndarray]:
    """The region's polygons, in whatever units `lm` is in."""
    if what == "teeth":
        return [lm[list(MOUTH_INNER)]]
    polys = []
    for idx in (EYE_L, EYE_R):
        pts = lm[list(idx)]
        c = pts.mean(axis=0)
        polys.append(cv2.convexHull(((pts - c) * 1.05 + c).astype(np.float32)).reshape(-1, 2))
    return polys


def _region_ramp(rgb: np.ndarray, polys: list[np.ndarray],
                 pct: tuple[float, float]) -> list[float] | None:
    """Two L percentiles inside `polys` on the whole frame, or None when the
    region is too small to measure (a closed mouth has no inner opening)."""
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for poly in polys:
        cv2.fillPoly(mask, [poly.round().astype(np.int32)], 1)
    if int(mask.sum()) < 12:
        return None
    ys, xs = np.nonzero(mask)
    pix = rgb[ys, xs].astype(np.float32)[None] / 255.0
    L = cv2.cvtColor(pix, cv2.COLOR_RGB2LAB)[0, :, 0]
    lo, hi = (float(v) for v in np.percentile(L, pct))
    return [lo, max(hi, lo + 1.0)]


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _whiten(reg: np.ndarray, alpha: np.ndarray, amount: float, ramp: list[float],
            cfg: dict[str, float]) -> np.ndarray:
    """Pull the bright, unsaturated pixels under `alpha` towards neutral and up."""
    lab = cv2.cvtColor(np.clip(reg, 0.0, 1.0), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    if "red_max" in cfg:
        off = _smoothstep(cfg["red_max"] * 0.6, cfg["red_max"], a)
    else:
        off = _smoothstep(cfg["chroma_max"] * 0.6, cfg["chroma_max"], np.hypot(a, b))
    w = alpha * _smoothstep(ramp[0], ramp[1], L) * (1.0 - off)
    k = amount * w
    cut = 1.0 - k * (1.0 - cfg["keep"])
    lab[..., 1] = a * cut
    lab[..., 2] = np.where(b > 0, b * cut, b)      # yellow goes; blue is not "yellowed"
    lab[..., 0] = np.minimum(100.0, L + k * cfg["lift"])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _guided(p: np.ndarray, guide: np.ndarray, r: int, eps: float) -> np.ndarray:
    """He et al.'s guided filter with a gray guide, from box filters. `p` is
    (H, W, 3); the guide is (H, W)."""
    size = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, -1, size, borderType=cv2.BORDER_REFLECT)

    mi = box(guide)
    var = box(guide * guide) - mi * mi
    mp = box(p)
    cov = box(guide[..., None] * p) - mi[..., None] * mp
    a = cov / (var[..., None] + eps)
    b = mp - a * mi[..., None]
    return box(a) * guide[..., None] + box(b)


def _luma(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32))


def _frame(shape: tuple[int, ...], roi: tuple[float, float, float, float]):
    """Whole-frame size in this array's pixels, and where the array starts."""
    h, w = shape[:2]
    fw, fh = w / roi[2], h / roi[3]
    return fw, fh, roi[0] * fw, roi[1] * fh


def padding(params: dict[str, Any] | None, faces: list[dict[str, Any]] | None,
            frame_w: float) -> float:
    """Pixels of neighbourhood a window render needs so that its pixels match
    the whole render's: the reach of the widest face's filters."""
    if is_neutral(params) or not faces:
        return 0.0
    widest = max(f["box"][2] for f in faces) * frame_w
    return widest * (_EVEN_RADIUS * 2.0 + _TEX_SIGMA * 3.0) + 2.0


def _skin_alpha(face: dict[str, Any], region: tuple[int, int, int, int],
                frame_w: float, frame_h: float, ox: float, oy: float,
                face_px: float) -> np.ndarray:
    """The face's skin, minus its features, over `region` (x0, y0, x1, y1 in
    this array's pixels)."""
    x0, y0, x1, y1 = region
    rw, rh = x1 - x0, y1 - y0
    cx, cy, cw, ch = face["crop"]
    # Where the stored alpha lands in this array, then resampled onto the region.
    ax, ay = cx * frame_w - ox - x0, cy * frame_h - oy - y0
    sw, sh = cw * frame_w / face["skin"].shape[1], ch * frame_h / face["skin"].shape[0]
    m = np.float32([[sw, 0, ax], [0, sh, ay]])
    skin = cv2.warpAffine(face["skin"], m, (rw, rh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    lm = np.asarray(face["landmarks"], np.float32) * [frame_w, frame_h] - [ox + x0, oy + y0]
    feat = np.zeros((rh, rw), np.float32)
    for idx, grow in _FEATURES:
        pts = lm[list(idx)]
        c = pts.mean(axis=0)
        hull = cv2.convexHull(((pts - c) * (1.0 + grow) + c).round().astype(np.int32))
        cv2.fillConvexPoly(feat, hull, 1.0, cv2.LINE_AA)
    feat = cv2.GaussianBlur(feat, (0, 0), max(0.5, _FEATURE_FEATHER * face_px))
    return np.clip(skin * (1.0 - np.clip(feat * 1.6, 0.0, 1.0)), 0.0, 1.0)


def _skin_mean(face: dict[str, Any]) -> float:
    """The face's mean skin luma, recorded by `analyze`."""
    return max(0.02, float(face["skin_luma"]))


def apply_portrait(img: np.ndarray, params: dict[str, Any] | None,
                   faces: list[dict[str, Any]] | None,
                   roi: tuple[float, float, float, float] = FULL_ROI) -> np.ndarray:
    """Smooth each face's skin in float32 RGB [0,1]. `roi` is where `img` sits
    in the frame, the same contract as `editing.render`."""
    assert img.dtype == np.float32, "apply_portrait works in float32"
    if is_neutral(params) or not faces:
        return img
    p = {**DEFAULT_PORTRAIT, **params}
    amount = p["smooth"] / 100.0
    frame_w, frame_h, ox, oy = _frame(img.shape, roi)
    h, w = img.shape[:2]
    out = img
    for face in faces:
        face_px = face["box"][2] * frame_w
        if face_px < 8:
            continue            # a face a few pixels wide on a thumbnail: nothing to smooth
        cx, cy, cw, ch = face["crop"]
        x0 = max(0, int(math.floor(cx * frame_w - ox)))
        y0 = max(0, int(math.floor(cy * frame_h - oy)))
        x1 = min(w, int(math.ceil((cx + cw) * frame_w - ox)))
        y1 = min(h, int(math.ceil((cy + ch) * frame_h - oy)))
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue            # this face is outside the window
        if out is img:
            out = img.copy()
        region = (x0, y0, x1, y1)
        if amount > 0:
            _smooth_face(out, face, region, frame_w, frame_h, ox, oy, face_px, amount)
        lm = (np.asarray(face["landmarks"], np.float32) * [frame_w, frame_h]
              - [ox + x0, oy + y0])
        for what in _WHITEN:
            ramp = face["ramps"].get(what)
            if not p[what] or ramp is None:
                continue
            alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
            for poly in _whiten_polys(what, lm):
                cv2.fillPoly(alpha, [poly.round().astype(np.int32)], 1.0, cv2.LINE_AA)
            alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.5, _WHITEN_FEATHER * face_px))
            if float(alpha.max()) <= 0.0:
                continue
            reg = out[y0:y1, x0:x1]
            out[y0:y1, x0:x1] = _whiten(reg, alpha, p[what] / 100.0, ramp, _WHITEN[what])
    return out


def _smooth_face(out: np.ndarray, face: dict[str, Any], region: tuple[int, int, int, int],
                 frame_w: float, frame_h: float, ox: float, oy: float,
                 face_px: float, amount: float) -> None:
    """Smooth one face's skin in place, over `region` of `out`."""
    x0, y0, x1, y1 = region
    alpha = _skin_alpha(face, region, frame_w, frame_h, ox, oy, face_px)
    if float(alpha.max()) <= 0.0:
        return
    reg = out[y0:y1, x0:x1]
    low = cv2.GaussianBlur(reg, (0, 0), max(0.5, _TEX_SIGMA * face_px))
    guide = _luma(low)
    # Measured on the face's own skin, from its stored alpha, so a window
    # showing part of the face uses the same number the whole render does.
    eps = (_EVEN_CONTRAST * _skin_mean(face)) ** 2
    even = _guided(low, guide, max(1, int(round(_EVEN_RADIUS * face_px))), eps)
    smoothed = reg + (amount * (even - low))
    out[y0:y1, x0:x1] = reg + (smoothed - reg) * alpha[..., None]


# ----- blemishes ----------------------------------------------------------

_BLEMISH_SCALES = (0.006, 0.010, 0.016)   # DoG sigmas, fractions of the face width
_BLEMISH_Z = 6.0          # strength against the skin's own spread in that band
_BLEMISH_ROUND = 0.4      # smaller over larger Hessian eigenvalue, at least
_BLEMISH_ERODE = 0.03     # how far inside the skin's edge to look, of the face width
_BLEMISH_MAX = 40         # per face
# Left out besides the features: the nose (nostrils and the creases of its
# wings are dark and round enough to pass) and a wide zone round each eye (the
# lid crease and the shadow of a bag under it).
_BLEMISH_SKIP = ((tuple(range(76, 86)), 1.6), (EYE_L, 1.9), (EYE_R, 1.9))
_HEAL_OVER = 2.5          # a spot heal's radius, in DoG sigmas: the blob and its rim
_BLEMISH_BLUE = -6.0      # b* below the skin's median past which a spot is not skin


def _roundness(ch: np.ndarray, sigma: float, sign: float) -> np.ndarray:
    """0 unless both Hessian eigenvalues have the sign of a spot (`sign` +1
    for a minimum, -1 for a maximum); then their ratio, smaller over larger."""
    g = cv2.GaussianBlur(ch, (0, 0), sigma)
    xx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    yy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    xy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    tr, det = (xx + yy) * sign, xx * yy - xy * xy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1, l2 = tr / 2.0 + disc, tr / 2.0 - disc
    return np.where((l1 > 0) & (l2 > 0), l2 / np.maximum(l1, 1e-9), 0.0)


def find_blemishes(rgb: np.ndarray, faces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Spot heals for the blemishes on every face, on a WHOLE uint8 frame.

    Returned in the healing module's op format, positions and radius as
    fractions of the frame (the radius of its width), strongest first.
    """
    assert rgb.dtype == np.uint8 and rgb.ndim == 3, "find_blemishes wants the uint8 frame"
    H, W = rgb.shape[:2]
    ops: list[dict[str, Any]] = []
    for face in faces:
        fw = face["box"][2] * W
        if fw < 60:
            continue            # too small for a blemish to be more than a pixel
        # The context crop rather than the face box, so every filter has real
        # pixels round the skin rather than a reflected border.
        cx, cy, cw, ch = face["crop"]
        x, y = max(0, int(cx * W)), max(0, int(cy * H))
        x1, y1 = min(W, int(math.ceil((cx + cw) * W))), min(H, int(math.ceil((cy + ch) * H)))
        alpha = _skin_alpha(face, (x, y, x1, y1), W, H, 0.0, 0.0, fw)
        inner = (alpha > 0.85).astype(np.uint8)
        lm = np.asarray(face["landmarks"], np.float32) * [W, H] - [x, y]
        for idx, grow in _BLEMISH_SKIP:
            pts = lm[list(idx)]
            c = pts.mean(axis=0)
            cv2.fillConvexPoly(inner, cv2.convexHull(((pts - c) * grow + c).round().astype(np.int32)), 0)
        r = max(1, int(round(_BLEMISH_ERODE * fw)))
        # Outside the crop counts as not skin; erode's default treats it as skin.
        inner = cv2.erode(inner, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)),
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        # Found inside the face box only: the crop's margin is context for the
        # filters, not a place to look (neck, shoulders and hairline).
        box = np.zeros_like(inner)
        bx0, by0 = int(face["box"][0] * W) - x, int(face["box"][1] * H) - y
        box[max(0, by0):by0 + int(face["box"][3] * H), max(0, bx0):bx0 + int(fw)] = 1
        sel = (inner > 0) & (box > 0)
        if int(sel.sum()) < 50:
            continue
        lab = cv2.cvtColor(rgb[y:y1, x:x1].astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        b_med = float(np.median(B[sel]))
        found = []
        for frac in _BLEMISH_SCALES:
            s = frac * fw
            dl = cv2.GaussianBlur(L, (0, 0), s) - cv2.GaussianBlur(L, (0, 0), 2.5 * s)
            da = cv2.GaussianBlur(A, (0, 0), s) - cv2.GaussianBlur(A, (0, 0), 2.5 * s)
            zs = []
            for band, sign in ((dl, -1.0), (da, 1.0)):
                med = float(np.median(band[sel]))
                spread = 1.4826 * float(np.median(np.abs(band[sel] - med))) + 1e-6
                zs.append(sign * (band - med) / spread)
            z = np.maximum(*zs)
            z[~sel] = 0.0
            round_ = np.maximum(_roundness(L, s, 1.0), _roundness(A, s, -1.0))
            peak = (z == cv2.dilate(z, np.ones((3, 3), np.uint8))) & (z > _BLEMISH_Z) \
                & (round_ > _BLEMISH_ROUND)
            # A blemish is skin gone darker or redder, never bluer: a spot well
            # below the skin's b* is a reflection in glasses or a light.
            bs = cv2.GaussianBlur(B, (0, 0), s)
            for py, px in zip(*np.nonzero(peak)):
                if bs[py, px] - b_med < _BLEMISH_BLUE:
                    continue
                found.append((float(z[py, px]), float(px + x), float(py + y), s))
        # Strongest first, one per place: a larger spot absorbs the smaller
        # detections inside it.
        found.sort(reverse=True)
        kept: list[tuple[float, float, float, float]] = []
        for z_, px, py, s in found:
            if all(math.hypot(px - qx, py - qy) > _HEAL_OVER * max(s, qs) for _, qx, qy, qs in kept):
                kept.append((z_, px, py, s))
        for _, px, py, s in kept[:_BLEMISH_MAX]:
            ops.append({"kind": "spot", "points": [[px / W, py / H]],
                        "radius": _HEAL_OVER * s / W, "feather": 50, "opacity": 100,
                        "method": "ns", "enabled": True})
    return ops
