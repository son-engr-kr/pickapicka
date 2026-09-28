"""Synthesize the chime the app plays as it opens.

    uv run python packaging/make_launch_sound.py

Writes src/pickapicka/web/sounds/launch.wav. Made from sine partials rather than
a sample, so there is no licence to track and changing it means editing this.
A soft Ebmaj9 chord swells over about 140 ms and settles, with one bell on top.
The reverb is seeded, so the same script writes the same file.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000
OUT = Path(__file__).resolve().parents[1] / "src" / "pickapicka" / "web" / "sounds" / "launch.wav"
# A notch under the -3 dBFS of the first draft, so it sits nearer the level of
# the system's own alert sounds.
PEAK = 0.5


def hz(midi: int) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def bell(f: float, dur: float, amp: float, decay: float, bright: float) -> np.ndarray:
    """A struck bell: inharmonic partials, the higher ones dying faster."""
    t = np.arange(int(SR * dur)) / SR
    partials = [(1.0, 1.0), (2.0, 0.45 * bright), (3.01, 0.22 * bright),
                (4.17, 0.12 * bright), (5.43, 0.06 * bright)]
    y = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t * decay * (0.6 + 0.5 * r))
            for r, a in partials)
    return amp * y * np.clip(t / 0.004, 0, 1)


def pad(f: float, dur: float, amp: float, attack: float, decay: float) -> np.ndarray:
    """A soft sustained tone with a slightly detuned second voice."""
    t = np.arange(int(SR * dur)) / SR
    y = sum(a * np.sin(2 * np.pi * f * m * t + p) for m, a, p in [(1, 1, 0), (2, .3, .5), (3, .12, 1.1)])
    y = y + 0.5 * np.sin(2 * np.pi * f * 1.004 * t)
    env = np.clip(t / attack, 0, 1) ** 2 * np.exp(-np.maximum(t - attack, 0) * decay)
    return amp * y * env


def place(buf: np.ndarray, sig: np.ndarray, at: float, pan: float) -> None:
    # Taper every note's tail, so none of them ends in a click.
    tail = min(len(sig), int(SR * 0.08))
    sig = sig.copy()
    sig[-tail:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
    i = int(SR * at)
    n = min(len(sig), buf.shape[0] - i)
    buf[i:i + n, 0] += sig[:n] * np.sqrt((1 - pan) / 2)
    buf[i:i + n, 1] += sig[:n] * np.sqrt((1 + pan) / 2)


def reverb(buf: np.ndarray, seconds: float = 1.6, mix: float = 0.25) -> np.ndarray:
    """Convolution with decaying noise: a small, bright room."""
    rng = np.random.default_rng(7)
    n = int(SR * seconds)
    # This chime was picked from three drafts made in one run, and the first
    # draft's reverb drew from the stream before this one did. Skipping those
    # draws keeps the file identical to the one that was listened to and chosen.
    rng.standard_normal(2 * n)
    t = np.arange(n) / SR
    out = buf.copy()
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-t * 4.5)
        ir[: int(SR * 0.012)] = 0
        size = len(buf) + n
        wet = np.fft.irfft(np.fft.rfft(buf[:, ch], size) * np.fft.rfft(ir, size))[: len(buf)]
        out[:, ch] = buf[:, ch] + mix * wet / np.max(np.abs(wet)) * np.max(np.abs(buf[:, ch]))
    return out


def main() -> None:
    buf = np.zeros((int(SR * 2.6), 2))
    for m, pan in [(51, 0), (58, -.3), (63, .3), (67, -.15), (70, .15)]:
        place(buf, pad(hz(m), 2.4, amp=.5, attack=.14, decay=1.3), 0.0, pan)
    place(buf, bell(hz(82), 2.2, amp=.6, decay=1.8, bright=.6), 0.10, .1)
    buf = reverb(buf)
    fade = int(SR * 0.25)
    buf[-fade:] *= np.linspace(1, 0, fade)[:, None]
    buf *= PEAK / np.max(np.abs(buf))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype("<i2").tobytes())
    print(f"{OUT}  {len(buf) / SR:.2f} s")


if __name__ == "__main__":
    main()
