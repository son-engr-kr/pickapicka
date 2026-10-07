"""Generative fill: an AI heal made by a diffusion model, for regions MI-GAN
cannot carry.

What it is for
--------------
`aifill`'s MI-GAN keeps a skin's pores in a small hole, and is the default. On
a large one it runs out: a hole 6% of the face across under an eye came back
with dark dashes where the creases ran into it, while this model drew the fold
of the lid and the skin under it. So this is the second, heavier tool, for
when MI-GAN's fill does not hold up, and an opt-in download.

The model
---------
Stable Diffusion 1.5 inpainting (Runway; the stable-diffusion-v1-5 mirror on
Hugging Face) with the LCM-LoRA for SD 1.5 (Luo et al., "LCM-LoRA: A
Universal Stable-Diffusion Acceleration Module", 2023) fused into its UNet,
so four steps do what fifty did, at a guidance of 1: one UNet call per step,
and no unconditional pass. The prompt is empty, and the empty prompt's text
embedding is computed once at export and shipped as an array, so neither the
tokenizer nor the text encoder (half a gigabyte) is needed; on faces an empty
prompt and "close-up photo of skin" gave fills that could not be told apart.

The licences are CreativeML OpenRAIL-M (the inpainting model) and OpenRAIL++
(the LoRA). Both allow commercial use, and both carry use restrictions that
whoever passes the weights on must pass on and hold their users to (section 4
of the licence), which is why it is a download the person asks for, after its
terms, and never part of the installer.

The sampling
------------
The inpainting UNet takes nine channels: the noisy latent (4), the hole at
latent size (1) and the latent of the image with the hole blanked (4). The
latents are the VAE's, times 0.18215. The schedule and the step are LCM's
(Luo et al., "Latent Consistency Models", 2023, the multistep sampler), as
the authors' code and diffusers' LCMScheduler implement it:

    t           999, 759, 499, 259     (4 of the 50-step schedule k*i - 1, k = 20)
    x0          (x - sqrt(1 - a_t) eps) / sqrt(a_t)
    c_skip      0.25 / ((10 t)^2 + 0.25)
    c_out       10 t / sqrt((10 t)^2 + 0.25)
    denoised    c_out x0 + c_skip x
    next x      sqrt(a_t') denoised + sqrt(1 - a_t') noise,  and the last
                step's denoised is the answer

with a_t the scaled-linear schedule's cumulative product (beta 0.00085 to
0.012 over 1000 steps). The hole is taken to latent size by max-pooling, not
by the nearest sample the reference pipeline uses, so a stroke thinner than
eight pixels still reaches the latent.

The noise is seeded from the stroke itself, so a fill can be made again and
come out the same.

The cost, measured on an M5 Pro's CPU through onnxruntime 1.25: a fill takes
about 15 s once the model is loaded and 22 s the first time (loading the
graphs and their first runs). Of the 15, the VAE decoder is 5, its encoder 2
and each of the four UNet calls 1 to 2.5; the UNet's time varies from call to
call, and no thread count steadied it. While a fill runs the process peaks
at 10 to 12.5 GB of resident memory (macOS, over repeated runs), most of it
the UNet's float32 weights and its first run's working memory; onnxruntime's
memory arena, its memory patterns and weight prepacking each made the peak
higher when switched off, and loading one graph at a time did not lower it.
Whether the model then stays loaded between fills is
`modelstore.keep_loaded`: by default only on a machine with 2.5 times that.
Against the torch reference the exported graphs agree to 2e-4 of their
output's range and the step to 4e-5 (diffusers 0.41, LCMScheduler).

The download is 1.9 GB: the weights are stored as float16 and cast to float32
in the graph, which onnxruntime folds when it loads them, so it computes in
float32 (its float16 kernels on the CPU measured half as fast).
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import healing as healing_mod
from . import paths

NAME = "sd15-inpaint-lcm"
LATENT_SCALE = 0.18215
STEPS = 4
_TRAIN_STEPS = 1000
_ORIGINAL_STEPS = 50
_BETAS = (0.00085, 0.012)
_TIMESTEP_SCALING = 10.0
_SIGMA_DATA = 0.5
SIZE = 512                 # what the model works at
_CONTEXT = 1.6             # the crop is this much wider than the hole's box, at least SIZE

MODEL_DIR = paths.MODEL_DIR / NAME
GRAPHS = ("unet", "vae_encoder", "vae_decoder")
EMBEDDING = "empty_prompt.npy"

_sessions: dict[str, Any] = {}
_LOCK = threading.Lock()


def _alphas_cumprod() -> np.ndarray:
    betas = np.linspace(_BETAS[0] ** 0.5, _BETAS[1] ** 0.5, _TRAIN_STEPS, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def timesteps(steps: int = STEPS) -> np.ndarray:
    """LCM's inference timesteps: `steps` evenly spaced indices into the
    distillation schedule k*i - 1 (i = 1..50, k = 20), from the noisy end."""
    k = _TRAIN_STEPS // _ORIGINAL_STEPS
    origin = (np.arange(1, _ORIGINAL_STEPS + 1) * k - 1)[::-1]
    idx = np.floor(np.linspace(0, _ORIGINAL_STEPS, num=steps, endpoint=False)).astype(np.int64)
    return origin[idx]


def lcm_step(x: np.ndarray, eps: np.ndarray, t: int, t_next: int | None,
             noise: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """One multistep LCM step: the next latent and this step's denoised one."""
    ac = _alphas_cumprod()
    a_t = ac[t]
    x0 = (x - np.sqrt(1.0 - a_t) * eps) / np.sqrt(a_t)
    st = t * _TIMESTEP_SCALING
    c_skip = _SIGMA_DATA ** 2 / (st ** 2 + _SIGMA_DATA ** 2)
    c_out = st / np.sqrt(st ** 2 + _SIGMA_DATA ** 2)
    denoised = c_out * x0 + c_skip * x
    if t_next is None:
        return denoised.astype(np.float32), denoised.astype(np.float32)
    a_n = ac[t_next]
    nxt = np.sqrt(a_n) * denoised + np.sqrt(1.0 - a_n) * noise
    return nxt.astype(np.float32), denoised.astype(np.float32)


# ----- the model files ------------------------------------------------------

def is_installed(model_dir: Path | None = None) -> bool:
    d = model_dir or MODEL_DIR
    return all((d / f"{g}.onnx").is_file() and (d / f"{g}.onnx.data").is_file() for g in GRAPHS) \
        and (d / EMBEDDING).is_file()


def _session(name: str, model_dir: Path | None = None):
    import onnxruntime as ort
    d = model_dir or MODEL_DIR
    key = f"{d}/{name}"
    with _LOCK:
        if key not in _sessions:
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _sessions[key] = ort.InferenceSession(str(d / f"{name}.onnx"), so, providers=["CPUExecutionProvider"])
        return _sessions[key]


def release() -> None:
    """Drop the loaded sessions (the UNet holds gigabytes), e.g. before the
    files are deleted."""
    with _LOCK:
        _sessions.clear()


# ----- inpainting a 512 square --------------------------------------------

def inpaint(rgb: np.ndarray, hole: np.ndarray, seed: int, model_dir: Path | None = None) -> np.ndarray:
    """The model on a SIZE x SIZE uint8 RGB square and a boolean hole of the
    same size; returns the square with the model's output everywhere (the
    caller keeps only the hole)."""
    assert rgb.shape == (SIZE, SIZE, 3) and rgb.dtype == np.uint8, "inpaint wants a 512 square, uint8"
    assert hole.shape == (SIZE, SIZE), "the hole must cover the square"
    image = (rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    masked = image * (~hole)[None, None].astype(np.float32)
    lat_hole = hole.reshape(SIZE // 8, 8, SIZE // 8, 8).max(axis=(1, 3)).astype(np.float32)[None, None]
    enc = _session("vae_encoder", model_dir)
    masked_lat = enc.run(None, {"image": masked})[0] * LATENT_SCALE
    emb = np.load((model_dir or MODEL_DIR) / EMBEDDING).astype(np.float32)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, 4, SIZE // 8, SIZE // 8)).astype(np.float32)
    unet = _session("unet", model_dir)
    ts = timesteps()
    for i, t in enumerate(ts):
        model_in = np.concatenate([x, lat_hole, masked_lat], axis=1).astype(np.float32)
        eps = unet.run(None, {"sample": model_in, "timestep": np.array([t], np.int64),
                              "encoder_hidden_states": emb})[0]
        last = i == len(ts) - 1
        noise = None if last else rng.standard_normal(x.shape).astype(np.float32)
        x, _ = lcm_step(x, eps, int(t), None if last else int(ts[i + 1]), noise)
    out = _session("vae_decoder", model_dir).run(None, {"latent": (x / LATENT_SCALE).astype(np.float32)})[0][0]
    return np.clip((out.transpose(1, 2, 0) + 1.0) * 127.5 + 0.5, 0, 255).astype(np.uint8)


_COLOUR_MATCH = True
_MATCH_RING = 24           # px at the model's size: the known band round the hole read
_MATCH_SIGMA = 16.0        # px: how smooth the correction is


def match_colour(out: np.ndarray, src: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Take out of the hole the slow colour drift the model's VAE leaves.

    Round the hole the model redraws pixels it was given, and the difference
    there is its drift, not content. That difference, spread smoothly over the
    hole (a normalized convolution of it over the ring), is subtracted inside
    the hole, so the fill meets its rim in the photo's own colour while what
    the model drew inside is kept."""
    ring = (cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * _MATCH_RING + 1, 2 * _MATCH_RING + 1))) > 0) & ~hole
    if not ring.any():
        return out
    diff = out.astype(np.float32) - src.astype(np.float32)
    w = ring.astype(np.float32)
    num = cv2.GaussianBlur(diff * w[..., None], (0, 0), _MATCH_SIGMA)
    den = cv2.GaussianBlur(w, (0, 0), _MATCH_SIGMA)[..., None]
    # Far inside a large hole the ring's weight thins out; spread the nearest
    # known drift there rather than dividing by almost nothing.
    drift = num / np.maximum(den, 1e-3)
    fixed = out.astype(np.float32)
    fixed[hole] -= drift[hole]
    return np.clip(fixed + 0.5, 0, 255).astype(np.uint8)


def seed_for(op: dict[str, Any], frame_shape: tuple[int, ...]) -> int:
    """The same stroke on the same frame size gets the same noise."""
    key = json.dumps({"points": op["points"], "radius": op["radius"], "kind": op["kind"],
                      "shape": list(frame_shape[:2])}, sort_keys=True)
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")


def make_fill(frame: np.ndarray, op: dict[str, Any], seed: int | None = None,
              model_dir: Path | None = None) -> tuple[np.ndarray, list[float]]:
    """The patch a generative heal writes, made from the WHOLE uint8 `frame`;
    the same contract as `aifill.make_fill`."""
    assert frame.dtype == np.uint8 and frame.ndim == 3, "make_fill wants the uint8 frame"
    h, w = frame.shape[:2]
    (x0, y0, x1, y1), hole = healing_mod.region_mask(op, w, h)
    assert x1 > x0 and y1 > y0, "the stroke misses the photo"
    # A square round the hole, at least SIZE, clamped into the frame.
    side = int(max(SIZE, _CONTEXT * max(x1 - x0, y1 - y0)))
    side = min(side, w, h)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    sx0 = int(np.clip(cx - side // 2, 0, w - side))
    sy0 = int(np.clip(cy - side // 2, 0, h - side))
    square = frame[sy0:sy0 + side, sx0:sx0 + side]
    holes = np.zeros((side, side), bool)
    holes[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0] = hole
    small = cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR)
    small_hole = cv2.resize(holes.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
    # Grown a little at the model's size, so its output reaches past the
    # hole's rim and the heal's feathered edge blends into made pixels.
    small_hole = cv2.dilate(small_hole.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out = inpaint(np.ascontiguousarray(small), small_hole,
                  seed if seed is not None else seed_for(op, frame.shape), model_dir)
    if _COLOUR_MATCH:
        out = match_colour(out, small, small_hole)
    big = cv2.resize(out, (side, side), interpolation=cv2.INTER_LANCZOS4) if side != SIZE else out
    patch = frame[y0:y1, x0:x1].copy()
    made = big[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]
    patch[hole] = made[hole]
    return patch, [x0 / w, y0 / h, x1 / w, y1 / h]
