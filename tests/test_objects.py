"""Unit checks for subject detection and appearance grouping. Runnable with
pytest or directly:

    uv run python tests/test_objects.py

The YOLOX model is only needed by `test_detect_on_a_real_photo`, which skips
itself when the model has not been downloaded yet.
"""
from __future__ import annotations

import numpy as np

from picture_classifier.scoring import appearance, blur, objects


# ----- class resolution ---------------------------------------------------

def test_resolve_classes() -> None:
    assert objects.resolve_classes("vehicle") == list(objects.VEHICLE_CLASSES)
    assert objects.resolve_classes("car,truck") == ["car", "truck"]
    assert objects.resolve_classes(" Car , CAR ") == ["car"], "dedupe + case-insensitive"
    assert objects.resolve_classes(["dog", "not-a-class"]) == ["dog"]
    assert objects.resolve_classes(None) == []
    assert objects.resolve_classes("") == []


def test_coco_class_list() -> None:
    assert len(objects.COCO_CLASSES) == 80
    assert objects.COCO_CLASSES[2] == "car", "class index order must match the model"
    for c in objects.VEHICLE_CLASSES:
        assert c in objects.COCO_CLASSES


# ----- decode / postprocess ----------------------------------------------

def _fake_pred(cx: float, cy: float, w: float, h: float, cls_id: int) -> np.ndarray:
    """One synthetic YOLOX head output with a single confident anchor.

    The anchor is placed on the stride-8 grid cell containing (cx, cy) and given
    the offsets that decode back to exactly that box, so the test pins the grid
    maths rather than restating it.
    """
    n = sum((objects.INPUT_SIZE[0] // s) * (objects.INPUT_SIZE[1] // s)
            for s in objects.STRIDES)
    pred = np.zeros((n, 85), dtype=np.float32)
    pred[:, 4] = 0.001  # everything else is background
    stride = objects.STRIDES[0]
    gw = objects.INPUT_SIZE[1] // stride
    gx, gy = int(cx // stride), int(cy // stride)
    row = gy * gw + gx
    pred[row, 0] = cx / stride - gx
    pred[row, 1] = cy / stride - gy
    pred[row, 2] = np.log(w / stride)
    pred[row, 3] = np.log(h / stride)
    pred[row, 4] = 0.99
    pred[row, 5 + cls_id] = 0.99
    return pred


def test_decode_recovers_the_box() -> None:
    pred = _fake_pred(200.0, 100.0, 64.0, 32.0, cls_id=2)
    decoded = objects._decode(pred)
    boxes = objects._to_boxes(decoded, ratio=1.0, conf=0.5, keep=None)
    assert len(boxes) == 1
    cid, score, (x, y, w, h) = boxes[0]
    assert objects.COCO_CLASSES[cid] == "car"
    assert score > 0.9
    # decode returns a centre box; _to_boxes converts to top-left xywh
    assert abs(x - (200 - 32)) <= 1 and abs(y - (100 - 16)) <= 1
    assert abs(w - 64) <= 1 and abs(h - 32) <= 1


def test_decode_maps_back_through_the_letterbox_ratio() -> None:
    pred = _fake_pred(200.0, 100.0, 64.0, 32.0, cls_id=2)
    boxes = objects._to_boxes(objects._decode(pred), ratio=0.5, conf=0.5, keep=None)
    _cid, _score, (x, y, w, h) = boxes[0]
    # A 0.5 letterbox scale means the original image was twice as big.
    assert abs(w - 128) <= 2 and abs(h - 64) <= 2
    assert abs(x - (400 - 64)) <= 2 and abs(y - (200 - 32)) <= 2


def test_class_filter_drops_everything_else() -> None:
    pred = _fake_pred(200.0, 100.0, 64.0, 32.0, cls_id=16)  # dog
    keep = {objects.COCO_CLASSES.index("car")}
    assert objects._to_boxes(objects._decode(pred), 1.0, 0.5, keep) == []
    keep = {objects.COCO_CLASSES.index("dog")}
    assert len(objects._to_boxes(objects._decode(pred), 1.0, 0.5, keep)) == 1


def test_preprocess_letterboxes_without_distorting() -> None:
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    blob, ratio = objects._preprocess(img)
    assert blob.shape == (1, 3, *objects.INPUT_SIZE)
    assert abs(ratio - objects.INPUT_SIZE[1] / 600) < 1e-6
    # The unused part of the canvas stays the 114 grey YOLOX pads with.
    canvas = blob[0].transpose(1, 2, 0)
    filled_h = int(round(300 * ratio))
    assert canvas[filled_h + 2, 0, 0] == 114


# ----- region sharpness ---------------------------------------------------

def test_region_blur_prefers_the_sharp_region(tmp_path) -> None:
    import cv2
    h, w = 400, 600
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    # Left half sharp noise, right half heavily blurred.
    img[:, w // 2:] = cv2.GaussianBlur(img[:, w // 2:], (0, 0), 6)
    path = tmp_path / "half.png"
    cv2.imwrite(str(path), img)
    sharp = blur.region_blur_score(str(path), [10, 10, w // 2 - 20, h - 20])
    soft = blur.region_blur_score(str(path), [w // 2 + 10, 10, w // 2 - 20, h - 20])
    assert sharp is not None and soft is not None
    assert sharp > soft * 5, f"sharp={sharp} soft={soft}"
    # A degenerate box is reported as unmeasurable rather than as zero sharpness.
    assert blur.region_blur_score(str(path), [0, 0, 1, 1]) is None


# ----- appearance descriptor ---------------------------------------------

def _swatch(bgr: tuple[int, int, int], size: int = 120) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    rng = np.random.default_rng(1)
    return np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_descriptor_shape_and_finiteness() -> None:
    d = appearance.describe(_swatch((30, 30, 200)))
    assert d.shape == (appearance.DESCRIPTOR_LEN,)
    assert np.isfinite(d).all()


def test_same_colour_is_closer_than_different_colour() -> None:
    red_a = appearance.describe(_swatch((30, 30, 200)))
    red_b = appearance.describe(_swatch((36, 34, 205)))
    blue = appearance.describe(_swatch((200, 40, 30)))
    assert _cos(red_a, red_b) > _cos(red_a, blue), "colour must dominate the descriptor"
    assert _cos(red_a, red_b) > 0.9


def test_crop_of_clamps_and_rejects_tiny_boxes() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = appearance.crop_of(img, [180, 60, 100, 100])  # runs off the right edge
    assert crop is not None and crop.shape[:2] == (40, 20)
    assert appearance.crop_of(img, [0, 0, 4, 4]) is None, "too small to describe"
    # Clamping can also leave a sliver that is too thin to be worth describing.
    assert appearance.crop_of(img, [180, 90, 100, 100]) is None


# ----- scoring integration ------------------------------------------------

def test_subject_scores_use_the_biggest_box(tmp_path) -> None:
    import cv2
    from picture_classifier import scorer
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
    path = tmp_path / "p.png"
    cv2.imwrite(str(path), img)
    objs = [
        {"cls": "car", "score": 0.99, "bbox_xywh": [10, 10, 40, 40]},      # tiny, confident
        {"cls": "car", "score": 0.60, "bbox_xywh": [150, 100, 300, 200]},  # the hero car
    ]
    s = scorer._subject_scores(path, objs, 600, 400)
    assert s["subject_count"] == 2
    # 300x200 of 600x400 = a quarter of the frame, and it is dead centre.
    assert abs(s["subject_area"] - 0.25) < 1e-4
    assert s["subject_center"] < 0.01
    assert s["subject_blur"] is not None

    off = scorer._subject_scores(
        path, [{"cls": "car", "score": 0.9, "bbox_xywh": [0, 0, 60, 40]}], 600, 400)
    assert off["subject_center"] > 0.5, "a corner subject should read as off-centre"

    empty = scorer._subject_scores(path, [], 600, 400)
    assert empty["subject_area"] == 0.0 and empty["subject_blur"] is None


def _photo(name: str, blur_v: float, **subject) -> dict:
    scores = {"blur": blur_v, "brightness": 128.0, "eye_open": None,
              "blur_pct": None, "exposure_zscore": None, "badness": None,
              "subject_count": 0, "subject_area": None, "subject_center": None,
              "subject_blur": None}
    scores.update(subject)
    return {"rel_path": name, "scene": "s", "scores": scores}


def test_missing_subject_is_penalized() -> None:
    from picture_classifier import scorer
    # The car-less frame is the *sharper* of the two, so it wins the blur rank
    # outright. Having no subject still has to sink it.
    items = [
        _photo("with.jpg", 500.0, subject_area=0.3, subject_center=0.1, subject_blur=500.0),
        _photo("without.jpg", 900.0, subject_area=0.0, subject_center=0.0, subject_blur=None),
    ]
    scorer.apply_scene_suggestions(items)
    with_s, without = items[0]["scores"]["badness"], items[1]["scores"]["badness"]
    assert without > with_s, f"no-subject {without} should be worse than {with_s}"
    assert items[0]["auto_suggestion"] == "pick"
    assert items[1]["auto_suggestion"] == "reject"


def test_subject_sharpness_outranks_frame_sharpness() -> None:
    from picture_classifier import scorer
    # "bokeh" is soft frame-wide but tack sharp on the subject; "flat" is the
    # reverse. The subject-aware ranking must prefer the bokeh shot.
    items = [
        _photo("bokeh.jpg", 40.0, subject_area=0.3, subject_center=0.1, subject_blur=900.0),
        _photo("flat.jpg", 800.0, subject_area=0.3, subject_center=0.1, subject_blur=60.0),
    ]
    scorer.apply_scene_suggestions(items)
    assert items[0]["scores"]["badness"] < items[1]["scores"]["badness"]


def test_subject_terms_are_off_without_detection() -> None:
    from picture_classifier import scorer
    items = [_photo("a.jpg", 100.0), _photo("b.jpg", 900.0)]
    scorer.apply_scene_suggestions(items)
    # No subject data anywhere: badness is the plain blur + exposure formula.
    assert items[1]["scores"]["badness"] < items[0]["scores"]["badness"]
    assert items[1]["scores"]["badness"] < 0.6


# ----- focus peaking ------------------------------------------------------

def _peak_coverage(gray, tmp_path, level: str = "normal") -> float:
    """Fraction of the frame the peaking overlay marks."""
    import cv2
    from picture_classifier.server import _ensure_peak
    src = tmp_path / f"{level}-src.jpg"
    cv2.imwrite(str(src), gray)
    out = _ensure_peak(src, tmp_path, f"{level}-src", "", level)
    alpha = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)[..., 3]
    return float((alpha > 0).mean())


def _textured(h: int = 400, w: int = 600) -> np.ndarray:
    """Structured detail — the kind of edge content a real photo has."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    v = 128 + 90 * np.sign(np.sin(x / 9.0)) * np.sign(np.cos(y / 11.0))
    return np.clip(v, 0, 255).astype(np.uint8)


def test_peaking_ignores_a_blurred_frame(tmp_path) -> None:
    """The whole point: a frame with nothing in focus must light up nothing.

    The previous detector thresholded on the frame's own percentile, so it
    always marked something — in fact it marked *more* on a blurred photo than
    a sharp one, because smooth gradients cross a relative threshold easily.
    """
    import cv2
    sharp = _textured()
    assert _peak_coverage(sharp, tmp_path) > 0.02
    for sigma in (4.0, 8.0):
        blurred = cv2.GaussianBlur(sharp, (0, 0), sigma)
        assert _peak_coverage(blurred, tmp_path) < 0.001, f"sigma {sigma}"


def test_peaking_follows_focus_not_contrast(tmp_path) -> None:
    """A soft edge with huge contrast must lose to a crisp low-contrast one."""
    import cv2
    h, w = 300, 200
    soft = np.zeros((h, w), np.float32)
    soft[:, w // 2:] = 255.0                       # maximum contrast…
    soft = cv2.GaussianBlur(soft, (0, 0), 5.0)     # …but thoroughly defocused
    crisp = np.zeros((h, w), np.float32)
    crisp[:, w // 2:] = 45.0                       # faint, but a hard edge
    assert _peak_coverage(soft.astype(np.uint8), tmp_path) == 0.0
    assert _peak_coverage(crisp.astype(np.uint8), tmp_path) > 0.0


def test_peaking_lands_on_the_sharp_half(tmp_path) -> None:
    import cv2
    from picture_classifier.server import _ensure_peak
    img = _textured(400, 600)
    img[:, 300:] = cv2.GaussianBlur(img[:, 300:], (0, 0), 5.0)   # right half soft
    src = tmp_path / "half.jpg"
    cv2.imwrite(str(src), img)
    alpha = cv2.imread(str(_ensure_peak(src, tmp_path, "half", "", "normal")),
                       cv2.IMREAD_UNCHANGED)[..., 3]
    left, right = (alpha[:, :300] > 0).mean(), (alpha[:, 320:] > 0).mean()
    assert left > 0.05 and right < 0.005, f"left {left} right {right}"


def test_peak_levels_are_ordered(tmp_path) -> None:
    from picture_classifier.server import PEAK_LEVELS
    img = _textured()
    cov = {lvl: _peak_coverage(img, tmp_path, lvl) for lvl in PEAK_LEVELS}
    assert cov["tight"] <= cov["normal"] <= cov["loose"], cov


def test_derived_caches_are_never_served_half_written(tmp_path) -> None:
    """Readers must never see a partially written cache file.

    The grid asks for /thumb and /peak at the same instant and both build the
    thumbnail; when the builder wrote straight to the served path, one request
    truncated the file the other was streaming. Starlette surfaced that as
    "Response content shorter than Content-Length" and the tile broke.
    """
    import os
    import threading
    import cv2
    from picture_classifier.server import _atomic_write, _ensure_thumb

    src = tmp_path / "src.jpg"
    cv2.imwrite(str(src), _textured(600, 900))
    thumbs = tmp_path / "thumbs"

    sizes: list[int] = []
    errors: list[Exception] = []
    done = threading.Event()
    dst = thumbs / "src.jpg"

    def build() -> None:
        try:
            for _ in range(15):
                # Force a rebuild each round: this is the writer racing readers.
                _ensure_thumb(src, thumbs, "src.jpg")
                if dst.exists():
                    os.utime(src, None)      # make the cache look stale again
        except Exception as exc:            # pragma: no cover - failure path
            errors.append(exc)

    def read() -> None:
        try:
            while not done.is_set():
                if dst.exists():
                    # Size then read, the way a FileResponse does it.
                    declared = dst.stat().st_size
                    got = len(dst.read_bytes())
                    sizes.append(declared - got)
        except Exception as exc:            # pragma: no cover - failure path
            errors.append(exc)

    readers = [threading.Thread(target=read) for _ in range(4)]
    builders = [threading.Thread(target=build) for _ in range(4)]
    for t in readers + builders:
        t.start()
    for t in builders:
        t.join()
    done.set()
    for t in readers:
        t.join()
    assert not errors, errors
    assert sizes, "the reader never observed the file"
    assert all(d == 0 for d in sizes), "a reader saw a partly written thumbnail"

    # The temp files the writer used must not be left lying around.
    assert not list(thumbs.glob(".*tmp*")), "temp files leaked into the cache"

    # And the helper itself only publishes on success.
    target = tmp_path / "atomic.txt"
    try:
        with _atomic_write(target) as tmp:
            tmp.write_text("partial")
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not target.exists(), "a failed build must not publish a file"


def test_peak_cache_is_versioned(tmp_path) -> None:
    """A detector change must not keep serving overlays built by the old one."""
    import cv2
    from picture_classifier.server import _ensure_peak, _PEAK_VERSION
    src = tmp_path / "c.jpg"
    cv2.imwrite(str(src), _textured(120, 160))
    out = _ensure_peak(src, tmp_path, "c", "", "normal")
    assert _PEAK_VERSION in out.name
    assert "normal" in out.name


# ----- download / export naming -------------------------------------------

def test_download_never_overwrites(tmp_path) -> None:
    """Two photos with the same filename must both survive the trip."""
    from picture_classifier.server import _unique_name
    (tmp_path / "DSC01.jpg").write_bytes(b"already here")
    seen: set[str] = set()
    first = _unique_name(tmp_path, "DSC01.jpg", seen)
    second = _unique_name(tmp_path, "DSC01.jpg", seen)
    assert first == "DSC01_1.jpg", "an existing file on disk must not be clobbered"
    assert second == "DSC01_2.jpg", "nor one claimed earlier in the same run"
    assert _unique_name(tmp_path, "other.jpg", seen) == "other.jpg"


def test_only_raw_and_edited_photos_are_re_encoded() -> None:
    """A plain JPEG is byte-copied: re-encoding it would lose quality for nothing."""
    from pathlib import Path
    from types import SimpleNamespace

    from picture_classifier.server import ExportSettings, _baked_name, _needs_render
    ctx = SimpleNamespace(source_path=lambda rel: Path(rel))
    default = ExportSettings()
    plain = {"rel_path": "a/DSC01.jpg"}
    assert not _needs_render(plain)
    assert _baked_name(ctx, plain, "a/DSC01.jpg", default) == "DSC01.jpg"

    edited = {"rel_path": "a/DSC01.jpg", "edit": {"exposure": 0.5}}
    assert _needs_render(edited)
    assert _baked_name(ctx, edited, "a/DSC01.jpg", default) == "DSC01.jpg"

    raw = {"rel_path": "a/DSC01.ARW", "type": "raw"}
    assert _needs_render(raw)
    assert _baked_name(ctx, raw, "a/DSC01.ARW", default) == "DSC01.jpg", "RAW has to come out as JPEG"

    # An edit that does nothing is not a reason to re-encode.
    assert not _needs_render({"rel_path": "a.jpg", "edit": {"exposure": 0}})

    # A copied original keeps its own extension, case and all; a format the
    # original is not in means writing it fresh.
    assert _baked_name(ctx, plain, "a/DSC01.JPG", default) == "DSC01.JPG"
    tiff = ExportSettings(format="tiff")
    assert _baked_name(ctx, plain, "a/DSC01.jpg", tiff) == "DSC01.tif"
    # A merged HDR result carries its 0 EV frame's metadata, so it is never a copy.
    merged = {"rel_path": "hdr::m.jpg", "type": "hdr", "base": "a/DSC02.jpg"}
    assert _baked_name(ctx, merged, "hdr::m.jpg", default, stem="m") == "m.jpg"


# ----- optional: the real model -------------------------------------------

def test_detect_on_a_real_photo() -> None:
    if not objects.is_model_ready():
        print("  skip test_detect_on_a_real_photo (model not downloaded)")
        return
    import cv2
    # A blue rectangle is not a car; this only pins the plumbing (shapes, types,
    # clamping), not detection quality.
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (100, 200), (400, 380), (120, 60, 40), -1)
    path = "/tmp/pcls-detect-smoke.jpg"
    cv2.imwrite(path, img)
    found, w, h = objects.detect(path, classes=["car"])
    assert (w, h) == (640, 480)
    for o in found:
        assert o["cls"] == "car"
        x, y, bw, bh = o["bbox_xywh"]
        assert 0 <= x < w and 0 <= y < h and bw > 0 and bh > 0
        assert x + bw <= w and y + bh <= h


def _main() -> None:
    import inspect
    import tempfile
    from pathlib import Path
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
