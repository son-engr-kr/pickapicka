"""What a lens or perspective drag's drafts reuse, and what stays exact.

A draft is a small render sent while the pointer moves. It used to rebuild the
whole corrected preview, find the faces again and estimate the chromatic
aberration again on every frame; 250 ms a frame with skin smoothing on. It now
reuses what the last settled render found, and the settled render that follows
the drag is what it always was.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pickapicka import db, editing, lens
from pickapicka.server import AppContext


def _project(tmp_path: Path) -> AppContext:
    rng = np.random.default_rng(0)
    # Detail with a colour fringe that grows with the radius, so auto CA has
    # something to find.
    h, w = 300, 450
    yy, xx = np.mgrid[0:h, 0:w]
    base = (128 + 90 * np.sign(np.sin(xx / 7.0)) * np.sign(np.cos(yy / 9.0))).astype(np.float32)
    rgb = np.stack([np.roll(base, 1, axis=1), base, np.roll(base, -1, axis=1)], axis=2)
    rgb = np.clip(rgb + rng.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "a.jpg", quality=95)
    data = db.init_db(tmp_path, "")
    data["photos"] = [{"rel_path": "a.jpg", "scene": "(none)", "width": w, "height": h}]
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    ctx = AppContext()
    ctx.load_db(path)
    return ctx


def test_the_ca_estimate_is_made_once_and_matches_the_frames_own(tmp_path, monkeypatch):
    ctx = _project(tmp_path)
    edit = {"lens": {"ca_auto": True, "distortion": 10}}
    base = ctx.get_decoded_base("a.jpg")
    want = editing.apply_optics(base, edit)          # estimated from the frame itself
    calls = []
    real = lens.estimate_ca
    monkeypatch.setattr(lens, "estimate_ca", lambda rgb: calls.append(1) or real(rgb))
    got = ctx.get_corrected_base("a.jpg", edit)
    assert np.array_equal(got, want), "the settled frame changed"
    for d in (20, 30, 40):
        ctx.get_corrected_base("a.jpg", {"lens": {"ca_auto": True, "distortion": d}})
    assert len(calls) == 1, f"estimated {len(calls)} times for one photo"
    assert ctx.ca_for("a.jpg", {"lens": {"distortion": 10}}) is None


def test_a_precomputed_estimate_is_the_same_correction():
    rng = np.random.default_rng(1)
    img = rng.random((120, 180, 3), dtype=np.float32)
    params = {"ca_auto": True, "distortion": 15}
    est = lens.estimate_ca(img)
    assert np.array_equal(lens.apply_lens(img, params, ca=est), lens.apply_lens(img, params))


def test_drafts_keep_the_settled_optics_for_their_analysis(tmp_path):
    ctx = _project(tmp_path)
    mask = {"type": "radial", "cx": 0.5, "cy": 0.5, "rx": 0.2, "ry": 0.2, "adj": {"exposure": 1}}
    settled = {"lens": {"distortion": 10}, "masks": [mask]}
    got = ctx.analysis_edit("a.jpg", settled, draft=False)
    assert got["lens"]["distortion"] == 10
    # A drag moves the lens; its drafts are analysed at the settled optics,
    # with everything else as the draft has it.
    dragging = {"lens": {"distortion": 30}, "masks": [mask], "portrait": {"smooth": 40}}
    got = ctx.analysis_edit("a.jpg", dragging, draft=True)
    assert got["lens"]["distortion"] == 10
    assert got["portrait"]["smooth"] == 40 and len(got["masks"]) == 1
    # The settled render at the end of the drag is analysed at its own optics.
    assert ctx.analysis_edit("a.jpg", dragging, draft=False)["lens"]["distortion"] == 30
    assert ctx.analysis_edit("a.jpg", dragging, draft=True)["lens"]["distortion"] == 30


def test_a_first_draft_uses_its_own_optics(tmp_path):
    ctx = _project(tmp_path)
    got = ctx.analysis_edit("a.jpg", {"transform": {"vertical": 12}}, draft=True)
    assert got["transform"]["vertical"] == 12


def test_a_fit_sized_base_is_the_corrected_frame_scaled(tmp_path):
    """A settled render the size the canvas shows is scaled after the optics,
    so it is the corrected frame, only smaller, and it is made once."""
    import cv2
    ctx = _project(tmp_path)
    edit = {"lens": {"distortion": 25}}
    corrected = ctx.get_corrected_base("a.jpg", edit)
    fit = ctx.get_fit_base("a.jpg", edit, 300)
    h, w = corrected.shape[:2]
    want = cv2.resize(corrected, (300, int(h * 300 / w)), interpolation=cv2.INTER_AREA)
    assert np.array_equal(fit, want)
    assert ctx.get_fit_base("a.jpg", edit, 300) is fit
    # Asked for more than the preview holds, it is the preview.
    assert ctx.get_fit_base("a.jpg", edit, 4000) is corrected


def test_the_quick_preview_decode_is_the_full_one_smaller(tmp_path):
    """The editor's base is decoded at a reduced DCT scale: same orientation,
    same size, and close to what a full decode resized gives."""
    ctx = _project(tmp_path)
    # Rotated by its EXIF, which must still be applied after the scaled decode.
    with Image.open(tmp_path / "a.jpg") as im:
        exif = Image.Exif()
        exif[0x0112] = 6
        im.save(tmp_path / "a.jpg", exif=exif.tobytes(), quality=95)
    full = ctx._decode_scaled("a.jpg", 120)
    quick = ctx._decode_scaled("a.jpg", 120, quick=True)
    assert quick.shape == full.shape and quick.shape[0] > quick.shape[1]
    d = np.abs(quick.astype(int) - full.astype(int))
    assert d.mean() < 3.0, d.mean()


def test_a_reload_forgets_both(tmp_path):
    ctx = _project(tmp_path)
    ctx.analysis_edit("a.jpg", {"lens": {"distortion": 10}}, draft=False)
    ctx.ca_for("a.jpg", {"lens": {"ca_auto": True}})
    ctx.reload_data()
    assert not ctx.settled_optics and not ctx.ca_cache
