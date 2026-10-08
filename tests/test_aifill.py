"""Checks for AI fill: the heal whose pixels a model made. Runnable with
pytest or directly:

    uv run python tests/test_aifill.py

Most of it does not need the model: the operation's schema, where a patch is
made from and where it lands at any resolution, that it never travels to
another photo, and that a merged project keeps the files its edits name. The
model's own `inpaint` is swapped for a stand-in there. The checks that run the
real model skip themselves until it has been downloaded, as test_segment does.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pickapicka import aifill, editing, healing, projects

needs_model = pytest.mark.skipif(not aifill.is_model_ready(), reason="AI fill model not downloaded")

W, H = 640, 480
FID = "ab" * 32


def _frame() -> np.ndarray:
    rng = np.random.default_rng(2)
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.stack([xx * 200 // W + 30, yy * 200 // H + 30, np.full_like(xx, 120)], axis=2)
    img = img + rng.integers(-12, 13, (H, W, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def _stroke(**kw) -> dict:
    return {"kind": "heal", "points": [[0.40, 0.45], [0.48, 0.47]], "radius": 0.02,
            "feather": 50, "opacity": 100, "enabled": True, **kw}


def _made(frame: np.ndarray, monkeypatch, value=(250, 20, 200)) -> tuple[dict, np.ndarray]:
    """An AI heal made with a stand-in model that paints the hole one colour."""
    def fake(rgb, hole):
        out = rgb.copy()
        out[hole] = value
        return out
    monkeypatch.setattr(aifill, "inpaint", fake)
    op = healing.normalize_op(_stroke())
    patch, box = aifill.make_fill(frame, op)
    fid = aifill.fill_id(patch, box)
    return {**op, "method": "ai", "fill": {"id": fid, "box": box}}, patch


# ----- the schema ----------------------------------------------------------

def test_an_ai_heal_needs_its_patch_to_be_an_edit() -> None:
    assert healing.normalize_op(_stroke(method="ai")) is None
    assert healing.normalize_op(_stroke(method="ai", fill={"id": "x", "box": [0, 0, 1, 1]})) is None
    assert healing.normalize_op(_stroke(method="ai", fill={"id": FID, "box": [0.5, 0, 0.4, 1]})) is None
    op = healing.normalize_op(_stroke(method="ai", fill={"id": FID, "box": [0.3, 0.4, 0.5, 0.5]}))
    assert op["method"] == "ai" and op["fill"] == {"id": FID, "box": [0.3, 0.4, 0.5, 0.5]}
    # A clone copies real pixels; it is never a fill.
    clone = healing.normalize_op(_stroke(kind="clone", dx=0.1, method="ai",
                                         fill={"id": FID, "box": [0.3, 0.4, 0.5, 0.5]}))
    assert clone["method"] == "ns" and "fill" not in clone


def test_fill_ids_and_taking_fills_out() -> None:
    ai = _stroke(method="ai", fill={"id": FID, "box": [0.3, 0.4, 0.5, 0.5]})
    block = {"ops": [_stroke(), ai, {**ai, "enabled": False}]}
    assert healing.fill_ids(block) == [FID]
    stripped = healing.without_fills(block)
    assert len(stripped["ops"]) == 1 and stripped["ops"][0]["method"] == "ns"
    assert healing.without_fills({"ops": [ai]}) is None


# ----- making one ------------------------------------------------------------

def test_the_patch_covers_what_the_heal_writes_and_only_the_hole_changes(monkeypatch) -> None:
    frame = _frame()
    op, patch = _made(frame, monkeypatch)
    (x0, y0, x1, y1), hole = healing.region_mask(healing.normalize_op(_stroke()), W, H)
    assert patch.shape[:2] == (y1 - y0, x1 - x0)
    assert op["fill"]["box"] == [x0 / W, y0 / H, x1 / W, y1 / H]
    assert np.array_equal(patch[~hole], frame[y0:y1, x0:x1][~hole])
    assert (patch[hole] == (250, 20, 200)).all()


def test_the_model_is_handed_room_round_the_hole(monkeypatch) -> None:
    seen = {}

    def fake(rgb, hole):
        seen["shape"], seen["hole"] = rgb.shape, hole.copy()
        return rgb.copy()
    monkeypatch.setattr(aifill, "inpaint", fake)
    big = np.zeros((1200, 1600, 3), np.uint8)
    aifill.make_fill(big, healing.normalize_op(_stroke(points=[[0.5, 0.5]], radius=0.01)))
    ys, xs = np.nonzero(seen["hole"])
    assert xs.min() >= aifill._CONTEXT and seen["shape"][1] - xs.max() > aifill._CONTEXT
    assert ys.min() >= aifill._CONTEXT and seen["shape"][0] - ys.max() > aifill._CONTEXT


def test_the_name_is_the_pixels() -> None:
    p = np.zeros((4, 5, 3), np.uint8)
    q = p.copy()
    q[1, 1, 0] = 1
    box = [0.1, 0.1, 0.2, 0.2]
    assert aifill.fill_id(p, box) == aifill.fill_id(p.copy(), list(box))
    assert aifill.fill_id(p, box) != aifill.fill_id(q, box)
    assert aifill.fill_id(p, box) != aifill.fill_id(p, [0.1, 0.1, 0.2, 0.3])


# ----- placing one -------------------------------------------------------------

def test_the_fill_lands_where_the_heal_would_have_written(monkeypatch) -> None:
    frame = _frame()
    op, patch = _made(frame, monkeypatch)
    img = frame.astype(np.float32) / 255.0
    out = healing.apply_healing(img, {"ops": [op]}, fills={op["fill"]["id"]: patch})
    (x0, y0, x1, y1), hole = healing.region_mask(healing.normalize_op(_stroke()), W, H)
    core = np.zeros((H, W), bool)
    core[y0:y1, x0:x1] = hole
    centre = (int(0.44 * H * 0 + 0.46 * H), int(0.44 * W))
    assert np.abs(out[centre] - np.float32([250, 20, 200]) / 255).max() < 1e-6
    assert np.array_equal(out[~core], img[~core])


def test_a_window_matches_the_whole_render(monkeypatch) -> None:
    frame = _frame()
    op, patch = _made(frame, monkeypatch)
    fills = {op["fill"]["id"]: patch}
    img = frame.astype(np.float32) / 255.0
    whole = healing.apply_healing(img, {"ops": [op]}, fills=fills)
    x0, y0, x1, y1 = 260, 200, 300, 240                 # through the stroke
    # Padded as the editor pads a 1:1 window: the heal's own alpha is a
    # distance from its stroke, which needs the stroke round the window.
    pad = int(math.ceil(editing.effect_padding({"healing": {"ops": [op]}}, W))) + 16
    px0, py0, px1, py1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    win = healing.apply_healing(img[py0:py1, px0:px1], {"ops": [op]},
                                roi=(px0 / W, py0 / H, (px1 - px0) / W, (py1 - py0) / H), fills=fills)
    assert np.abs(win[pad:pad + y1 - y0, pad:pad + x1 - x0] - whole[y0:y1, x0:x1]).max() < 1e-6


def test_at_half_size_it_lands_in_the_same_place(monkeypatch) -> None:
    frame = _frame()
    op, patch = _made(frame, monkeypatch)
    import cv2
    small = cv2.resize(frame, (W // 2, H // 2), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    out = healing.apply_healing(small, {"ops": [op]}, fills={op["fill"]["id"]: patch})
    # The stroke's middle, at half size, is the fill's colour.
    cy, cx = int(0.46 * H / 2), int(0.44 * W / 2)
    assert np.abs(out[cy, cx] - np.float32([250, 20, 200]) / 255).max() < 0.02


def test_rendering_without_the_patch_is_an_error(monkeypatch) -> None:
    frame = _frame()
    op, _ = _made(frame, monkeypatch)
    with pytest.raises(AssertionError, match="patch"):
        editing.render(frame, {"healing": {"ops": [op]}})


def test_render_composites_the_patch(monkeypatch) -> None:
    frame = _frame()
    op, patch = _made(frame, monkeypatch)
    out = editing.render(frame, {"healing": {"ops": [op]}}, fills={op["fill"]["id"]: patch})
    assert tuple(out[int(0.46 * H), int(0.44 * W)]) == (250, 20, 200)


# ----- it stays on its own photo -------------------------------------------------

def test_a_preset_or_an_add_never_carries_a_fill(monkeypatch) -> None:
    frame = _frame()
    op, _ = _made(frame, monkeypatch)
    base = {"exposure": 0.3, "healing": {"ops": [_stroke()]}}
    merged = editing.merge_additive(base, {"contrast": 10, "healing": {"ops": [op, _stroke(points=[[0.1, 0.1]])]}})
    methods = [o["method"] for o in merged["healing"]["ops"]]
    assert methods == ["ns", "ns"] and merged["contrast"] == 10


def test_replacing_an_edit_keeps_the_photos_own_fills(monkeypatch) -> None:
    from pickapicka.server import _keep_own
    frame = _frame()
    mine, _ = _made(frame, monkeypatch, value=(1, 2, 3))
    theirs, _ = _made(frame, monkeypatch, value=(9, 9, 9))
    assert mine["fill"]["id"] != theirs["fill"]["id"]
    incoming = editing.normalize({"exposure": 1.0, "healing": {"ops": [theirs, _stroke()]}})
    out = _keep_own(incoming, {"healing": {"ops": [mine]}})
    ids = healing.fill_ids(out["healing"])
    assert ids == [mine["fill"]["id"]] and out["exposure"] == 1.0
    assert len(out["healing"]["ops"]) == 2


def test_a_merged_project_takes_the_fills_its_edits_name(tmp_path) -> None:
    a, b, dest = tmp_path / "a", tmp_path / "b", tmp_path / "merged"
    ids = ["1" * 64, "2" * 64, "3" * 64]
    for d, fid in ((a, ids[0]), (b, ids[1]), (b, ids[2])):
        (d / projects.FILLS_DIR).mkdir(parents=True, exist_ok=True)
        (d / projects.FILLS_DIR / f"{fid}.png").write_bytes(fid.encode())
    ai = lambda fid: {"kind": "spot", "points": [[0.5, 0.5]], "radius": 0.01, "method": "ai",
                      "fill": {"id": fid, "box": [0.4, 0.4, 0.6, 0.6]}}
    marks = {"x.jpg": {"edit": {"healing": {"ops": [ai(ids[0])]}}},
             "y.jpg": {"edit_slots": [None, {"healing": {"ops": [ai(ids[1])]}}], "rating": 3}}
    assert projects.carry_fills([a, b], marks, dest) == 2
    assert sorted(p.stem for p in (dest / projects.FILLS_DIR).iterdir()) == ids[:2]


# ----- the real model -------------------------------------------------------------

@needs_model
def test_the_model_fills_a_hole_in_texture_with_texture() -> None:
    """Noise with a hole cut in it: the fill has the texture's fine detail,
    where a diffusion heal would leave a flat smudge."""
    import cv2
    rng = np.random.default_rng(4)
    tex = np.clip(128 + cv2.GaussianBlur(rng.normal(0, 40, (H, W)), (0, 0), 1.2), 0, 255).astype(np.uint8)
    frame = np.dstack([tex, tex, tex])
    op = healing.normalize_op(_stroke(points=[[0.5, 0.5]], radius=0.04))
    patch, box = aifill.make_fill(frame, op)
    (x0, y0, x1, y1), hole = healing.region_mask(op, W, H)
    hp = lambda a: (a.astype(np.float32) - cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.0))
    filled = hp(patch[..., 0])[hole].std()
    truth = hp(frame[y0:y1, x0:x1, 0])[hole].std()
    assert filled > 0.5 * truth, (filled, truth)


@needs_model
def test_the_model_is_deterministic() -> None:
    frame = _frame()
    op = healing.normalize_op(_stroke())
    p1, b1 = aifill.make_fill(frame, op)
    p2, b2 = aifill.make_fill(frame, op)
    assert np.array_equal(p1, p2) and b1 == b2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
