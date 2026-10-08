"""Checks for generative fill (the diffusion model). Runnable with pytest or
directly:

    uv run pytest tests/test_genfill.py

The sampler's numbers are checked against what diffusers' LCMScheduler gives
(the timesteps verbatim; the step by the property it has to have), the colour
match by a drift it has to remove, and the crop round a stroke with a stand-in
for the model. The check that runs the model itself skips until its 1.9 GB
are downloaded.
"""
from __future__ import annotations

import numpy as np
import pytest

from pickapicka import genfill, healing

needs_model = pytest.mark.skipif(not genfill.is_installed(), reason="generative fill model not downloaded")


def test_the_timesteps_are_lcms() -> None:
    # diffusers' LCMScheduler(set_timesteps(4)) on the SD 1.5 config gives these.
    assert genfill.timesteps(4).tolist() == [999, 759, 499, 259]
    assert genfill.timesteps(2).tolist() == [999, 499]


def test_a_step_recovers_the_clean_latent_from_its_own_noise() -> None:
    """With the true noise as the model's prediction, x0 comes back exactly,
    and at these timesteps c_out is all but 1, so the denoised latent is x0."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((1, 4, 8, 8))
    eps = rng.standard_normal((1, 4, 8, 8))
    for t, nxt in ((999, 759), (259, None)):
        a = genfill._alphas_cumprod()[t]
        x = np.sqrt(a) * x0 + np.sqrt(1 - a) * eps
        noise = rng.standard_normal(x.shape)
        out, den = genfill.lcm_step(x.astype(np.float32), eps.astype(np.float32), t, nxt,
                                    None if nxt is None else noise.astype(np.float32))
        assert np.abs(den - x0).max() < 1e-4
        if nxt is None:
            assert np.array_equal(out, den)
        else:
            an = genfill._alphas_cumprod()[nxt]
            assert np.abs(out - (np.sqrt(an) * den + np.sqrt(1 - an) * noise)).max() < 1e-4


def test_the_colour_match_takes_out_a_drift_and_keeps_what_was_drawn() -> None:
    rng = np.random.default_rng(1)
    src = rng.integers(60, 190, (genfill.SIZE, genfill.SIZE, 3)).astype(np.uint8)
    hole = np.zeros((genfill.SIZE, genfill.SIZE), bool)
    hole[200:300, 220:320] = True
    drawn = src.astype(int).copy()
    drawn[hole] = (120, 90, 70)                      # what the model drew in the hole
    out = np.clip(drawn + (14, -6, 9), 0, 255).astype(np.uint8)   # and its drift everywhere
    fixed = genfill.match_colour(out, src, hole)
    assert np.abs(fixed[hole].astype(int) - (120, 90, 70)).max() <= 1
    assert np.array_equal(fixed[~hole], out[~hole])  # outside the hole is the caller's to drop


def _stroke(**kw) -> dict:
    return healing.normalize_op({"kind": "heal", "points": [[0.40, 0.45], [0.48, 0.47]],
                                 "radius": 0.02, "feather": 50, **kw})


def test_the_patch_keeps_the_aifill_contract(monkeypatch) -> None:
    seen = {}

    def fake(rgb, hole, seed, model_dir=None):
        seen["shape"], seen["hole"], seen["seed"] = rgb.shape, hole.copy(), seed
        out = rgb.copy()
        out[hole] = (250, 20, 200)
        return out
    monkeypatch.setattr(genfill, "inpaint", fake)
    monkeypatch.setattr(genfill, "_COLOUR_MATCH", False)
    frame = np.random.default_rng(2).integers(0, 255, (900, 1200, 3)).astype(np.uint8)
    op = _stroke()
    patch, box = genfill.make_fill(frame, op)
    (x0, y0, x1, y1), hole = healing.region_mask(op, 1200, 900)
    assert seen["shape"] == (genfill.SIZE, genfill.SIZE, 3)
    assert box == [x0 / 1200, y0 / 900, x1 / 1200, y1 / 900]
    assert patch.shape[:2] == (y1 - y0, x1 - x0)
    assert np.array_equal(patch[~hole], frame[y0:y1, x0:x1][~hole])
    assert (np.abs(patch[hole].astype(int) - (250, 20, 200)) <= 2).mean() > 0.95
    # The hole given to the model covers the stroke, grown a little.
    assert seen["hole"].sum() > hole.sum() * (genfill.SIZE / 900) ** 2


def test_the_same_stroke_gets_the_same_noise() -> None:
    a, b = _stroke(), _stroke()
    assert genfill.seed_for(a, (900, 1200, 3)) == genfill.seed_for(b, (900, 1200, 3))
    assert genfill.seed_for(a, (900, 1200, 3)) != genfill.seed_for(_stroke(radius=0.03), (900, 1200, 3))


def test_nothing_is_installed_in_an_empty_folder(tmp_path) -> None:
    assert not genfill.is_installed(tmp_path)


@needs_model
def test_the_model_fills_a_hole_and_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:600, 0:800]
    frame = np.clip(np.stack([xx // 4, yy // 3, np.full_like(xx, 140)], axis=2)
                    + rng.integers(-8, 9, (600, 800, 3)), 0, 255).astype(np.uint8)
    op = _stroke(points=[[0.5, 0.5]], radius=0.05)
    p1, b1 = genfill.make_fill(frame, op)
    p2, b2 = genfill.make_fill(frame, op)
    assert np.array_equal(p1, p2) and b1 == b2
    (x0, y0, x1, y1), hole = healing.region_mask(op, 800, 600)
    # The fill continues the gradient round it rather than leaving a blot.
    err = np.abs(p1[hole].astype(int) - frame[y0:y1, x0:x1][hole].astype(int)).mean()
    assert err < 25, err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
