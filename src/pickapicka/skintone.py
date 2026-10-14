"""Skin tone protection: warm or grade a photo without turning its faces orange.

What goes wrong
---------------
Every colour edit treats skin like any other surface, and skin is the one
surface a viewer judges against a narrow memory of what it should be. On the
ColorChecker's light-skin patch, Temperature +60 takes CIELAB chroma from 21.7
to 39.3 and moves hue by only 3 degrees: the orange face is over-saturation,
not a new hue. The looks do it as well: Peach and Golden hour raise the skin's
chroma by over a third, with no Temperature at all, and Gold film and
Faded 70s turn a bright cheek 20 to 32 degrees towards yellow. So this is a
stage after every colour edit, whichever of them moved the skin.

What the skin is held to
------------------------
Each skin pixel is held to itself: the same render with every colour edit
left out (`editing.colour_neutral`). Not to a fixed "skin colour", which is
what most published skin enhancers pull towards. Measured skin is too varied
for one: across the 14,533 spectra of the ISSA database (Lu et al., Sci Data
2025) ethnicity explains half the variance of L* but only 13% of hue, body
site explains 39% of hue (face 53-56 degrees, neck and arms 62-67), and skin
in photographs spreads wider still. A band fitted to any of that would
"correct" faces that were fine. Held to itself, a forehead stays as yellowish
and a cheek as red as it was, and a neck keeps its own hue beside the face.
Kodak's colour shields (US6850342B2) and Lightroom's Vibrance protect skin
the same way, against the edit rather than towards a target.

How far it may move
-------------------
The research on preferred skin colour agrees on three things that shape the
rule. Hue is tolerated less than chroma: Zeng and Luo (CIC 2010, CRA 2013),
Sanger et al. (CIC 1994), Chen et al. (JIST 2013) and Cao et al. (CIC 2020)
all find it. The tolerated region is an ellipse whose long axis runs from
redder and paler to yellower and more saturated (62 degrees in a*b* for Zeng,
68.7 for Cao, against a centre near 46-49). And a yellowing skin loses
quality fast (Braun, US7190831B2), faster than a reddening one. Cao, Luo and
Liu's ellipse liked by half their 90 observers reaches about 14.6 in chroma
and 10.2 across hue from its centre (C* 24.7, h 46.1).

So the change is split per pixel into its chroma part and its hue part,
the hue measured as CIE's metric hue difference (a distance in chroma units,
so that a pale or very dark skin, at C* 10, is not judged by a hue angle that
a small shift swings by 11 degrees), and each is passed through a knee whose
slope is 1 at the start, so a small change goes through almost whole, and
which eases towards a limit it never reaches. Chroma may fall freely, which
is what a black and white or a faded look does to skin; the yellow side of hue
is held tighter than the red; and a turn towards the hue people prefer skin
at, as far as that hue, is free, since that is a cast coming off (a face in
shade goes magenta, and warming it is the fix).

    C, h    = chroma and hue of the reference pixel; C', h' of the graded one
    C_out   = C + knee(C' - C)          when it rises, else C'
    turn    = h' - h, less the part of it that goes towards the preferred hue
    dH      = 2 sqrt(C C_out) sin(turn / 2), knee'd, turned back into an angle
    out     = (L of the graded pixel, C_out at the eased hue)

Warming is the edit this lets through most: Temperature moves the
light-skin patch at 58 degrees in a*b*, close to the long axis of the
tolerance, which is why a little warmth on a face looks right and only the
amount goes wrong. A film look's drift goes across that axis.

Lightness is never touched, as in nearly every published method (LSI's
US7821580B2 gives the reason: change it and a skin's race may be corrupted).
The edit's exposure and tone stay the photo's.

What is not established
-----------------------
No study measures how far a skin may move from what it was before it looks
wrong; the numbers above describe which skin colours people like, around a
preferred centre. Using them for a change from the pixel's own colour is a
transfer, and so the limits and the yellow side's margin here were set by
eye on real portraits, within those bounds (see the values below and
docs/editing.md).
"""
from __future__ import annotations

import cv2
import numpy as np

# Below this chroma a pixel's direction in a*b* says nothing (a specular
# highlight, a near-grey patch), so all of its change counts as chroma. Measured
# skin does not go under it: the darkest in ISSA, L* under 35, sits at C* 10-13.
GREY_GUARD = 10.0

# The hue people prefer skin at, the same for every skin group: 46.1 degrees in
# Cao, Luo and Liu (CIC 2020, D65), 46 in Peng et al. (CIC 2020) for Caucasian,
# Oriental, South Asian and African faces. A hue change towards it is let
# through, since that is a cast being taken off, not one being put on.
PREFERRED_HUE = 46.1

# How far each part of the change may go, in CIELAB units (C*ab, and the
# across-hue distance): the limit the knee eases towards. Cao et al.'s
# half-liked ellipse bounds them (14.6 in chroma, 10.2 across hue), and at
# those a face under Temperature +80 was as orange as without the hold, on
# real portraits from a hiking shoot. These were chosen by eye on the same
# frames: 8 still left a peach face, 4 to 5 kept it natural with the scene
# around it warm, and the yellow side at half the red one.
CHROMA_LIMIT = 5.0
HUE_LIMIT_RED, HUE_LIMIT_YELLOW = 5.0, 2.5


def knee(x: np.ndarray, limit: float) -> np.ndarray:
    """`limit * tanh(x / limit)` for `x` >= 0: slope 1 at 0, so a small change
    goes through almost whole, and never past `limit`."""
    return (limit * np.tanh(x / limit)).astype(np.float32)


def _signed_knee(x: np.ndarray, up: float, down: float) -> np.ndarray:
    return np.where(x > 0.0, knee(np.maximum(x, 0.0), up), -knee(np.maximum(-x, 0.0), down))


def _wrap(angle: np.ndarray) -> np.ndarray:
    """An angle difference in radians, into (-pi, pi]."""
    return np.pi - np.mod(np.pi - angle, 2.0 * np.pi)


def hold(graded_ab: np.ndarray, ref_ab: np.ndarray) -> np.ndarray:
    """The a*b* that `graded_ab` keeps of its change from `ref_ab`, both (..., 2)."""
    c_ref = np.hypot(ref_ab[..., 0], ref_ab[..., 1])
    c_out = np.hypot(graded_ab[..., 0], graded_ab[..., 1])
    h_ref = np.arctan2(ref_ab[..., 1], ref_ab[..., 0])
    turn = _wrap(np.arctan2(graded_ab[..., 1], graded_ab[..., 0]) - h_ref)  # + is towards yellow
    rise = c_out - c_ref
    c_kept = np.where(rise > 0.0, c_ref + knee(np.maximum(rise, 0.0), CHROMA_LIMIT), c_out)
    # The turn towards the preferred hue, as far as it and no further, is free.
    to_pref = _wrap(np.radians(PREFERRED_HUE) - h_ref)
    free = np.where(turn * to_pref > 0.0, np.sign(turn) * np.minimum(np.abs(turn), np.abs(to_pref)), 0.0)
    rest = turn - free
    # The rest is eased as a distance, CIE's metric hue difference
    # 2 sqrt(C1 C2) sin(dh / 2), so that the same turn counts for less on a
    # paler skin, and turned back into an angle.
    span = 2.0 * np.sqrt(c_ref * c_kept)
    dist = span * np.sin(np.abs(rest) / 2.0)
    eased = np.where(rest > 0.0, knee(dist, HUE_LIMIT_YELLOW), knee(dist, HUE_LIMIT_RED))
    rest_kept = np.sign(rest) * 2.0 * np.arcsin(np.clip(eased / np.maximum(span, 1e-6), 0.0, 1.0))
    h_kept = h_ref + free + rest_kept
    polar = np.stack([c_kept * np.cos(h_kept), c_kept * np.sin(h_kept)], axis=-1)
    # Near grey there is no hue to turn from: the whole change counts as
    # chroma, eased by its length.
    d = graded_ab - ref_ab
    size = np.hypot(d[..., 0], d[..., 1])
    whole = ref_ab + d * (knee(size, CHROMA_LIMIT) / np.maximum(size, 1e-6))[..., None]
    g = np.clip(c_ref / GREY_GUARD, 0.0, 1.0)
    g = (g * g * (3.0 - 2.0 * g))[..., None]
    return (g * polar + (1.0 - g) * whole).astype(np.float32)


def protect(graded: np.ndarray, ref: np.ndarray,
            box: tuple[int, int, int, int], weight: np.ndarray) -> np.ndarray:
    """`graded` with the skin inside `box` (x0, y0, x1, y1) held to `ref`.

    Both are float32 RGB in [0,1] of the same shape: the render as edited and
    with its colour edits left out. `weight` covers `box` and says how much of
    the hold each pixel gets (the skin's alpha times the slider); 0 leaves a
    pixel exactly as graded, and only the others are converted and held.
    Written into `graded`, which is returned.
    """
    assert graded.shape == ref.shape and graded.dtype == np.float32
    x0, y0, x1, y1 = box
    assert weight.shape == (y1 - y0, x1 - x0)
    on = weight > 0.0
    g = graded[y0:y1, x0:x1][on].reshape(-1, 1, 3)
    lab_g = cv2.cvtColor(g, cv2.COLOR_RGB2LAB)
    lab_r = cv2.cvtColor(np.ascontiguousarray(ref[y0:y1, x0:x1][on]).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    lab_g[..., 1:] = hold(lab_g[..., 1:], lab_r[..., 1:])
    rgb = np.clip(cv2.cvtColor(lab_g, cv2.COLOR_LAB2RGB), 0.0, 1.0)
    win = graded[y0:y1, x0:x1]
    win[on] = (g + weight[on][:, None, None] * (rgb - g)).reshape(-1, 3)
    return graded
