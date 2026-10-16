"""Presets: a named set of parts of an edit, laid onto another photo's edit.

What a preset is, and what applying one does, follows what Lightroom, Capture
One, darktable and DxO PhotoLab agree on (see docs/editing.md, "Presets"):

  - A preset carries the parts that were ticked when it was saved, and only
    those. Applying it sets each of them to the stored value, a zero included,
    and leaves every other part of the photo's edit as it was. Whether a part
    is in the preset is what decides it, never whether its value is zero: the
    old "add" mode read a zero as "leave it", so no preset could take clarity
    back to nothing.
  - Values are absolute. A preset at +0.5 EV puts the photo at +0.5 EV, not at
    its own exposure plus half a stop; the Amount slider is the one relative
    control, and it moves each value from where the photo had it towards the
    preset's.
  - What belongs to one photo stays out. Heals, red eye and a face's own
    portrait settings are never stored. Crop, lens and the watermark are parts
    like the rest, and "Modified" ticks them when they are set, as it says;
    the save dialog marks them as usually the photo's own.
  - A preset's layers are added to the photo's, each tagged with the preset's
    id, and applying the same preset again replaces those rather than stacking
    a second set. Lightroom stacks them, and that has been a standing
    complaint against it since 2021.

`auto_light` is a part with no values: on apply, the photo's own auto tone
(`editing.auto_tone`) sets exposure, contrast, whites and blacks, so one preset
suits frames exposed differently. It runs after the other parts, so it wins
over a stored Light part for those four sliders.

`keep_light` is the other part with no values, and the opposite promise: the
photo keeps its own brightness. A look's curves and colours move a frame's
midtone, and a fixed curve cannot know where that midtone is; a night frame's
sits at a tenth of a day frame's. So the photo is measured, before and after,
and the preset's master curve is bent until its median comes back
(`keep_light`). Measured against the photo itself, not against a target, a
night frame stays a night frame and a snowfield stays bright.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np

from . import editing, grading as grading_mod, portrait as portrait_mod

# The parts of an edit a preset can carry, and the edit keys each one holds.
# The slider parts follow the editor's sections (EDIT_SCHEMA in app.js).
PARTS: dict[str, tuple[str, ...]] = {
    "light": ("exposure", "contrast", "highlights", "shadows", "whites", "blacks"),
    "color": ("temp", "tint", "vibrance", "saturation", "clean"),
    "detail": ("texture", "clarity", "dehaze", "denoise", "sharpen", "sharpen_radius",
               "sharpen_detail", "sharpen_masking", "vignette"),
    "creative": ("blur", "motion", "motion_angle", "glow", "pixelate"),
    "curve": editing.CURVE_KEYS,
    "mixer": ("hsl", "points"),
    "grading": ("grading",),
    "look": ("lut",),
    "film": ("film",),
    "layers": ("masks",),
    "portrait": ("portrait",),
    "crop": ("crop", "tilt"),
    "lens": ("lens", "transform"),
    "watermark": ("watermark",),
}
AUTO_LIGHT = "auto_light"
KEEP_LIGHT = "keep_light"
PART_NAMES: tuple[str, ...] = (*PARTS, AUTO_LIGHT, KEEP_LIGHT)
# Parts with no values: what they do is worked out from each photo on apply.
FROM_THE_PHOTO: tuple[str, ...] = (AUTO_LIGHT, KEEP_LIGHT)
# Parts that are usually the photo's own (its framing, its lens, its
# signature) rather than the look's. Ticked like any other when set; the save
# dialog says this about them. Leaving them out of "Modified" was tried, and
# read as a part that had been changed not counting as modified.
PHOTO_OWN: tuple[str, ...] = ("crop", "lens", "watermark")
AMOUNT_RANGE = (0, 200)
AUTO_KEYS: tuple[str, ...] = ("exposure", "contrast", "whites", "blacks")

_SLIDER_PARTS = ("light", "color", "detail", "creative")
_SLIDER_KEYS = frozenset(k for p in _SLIDER_PARTS for k in PARTS[p])
assert _SLIDER_KEYS == set(editing._RANGES) - {"tilt"}, \
    "every slider belongs to exactly one preset part"
# Settings that say how an effect looks, not how much of it there is (a blur's
# angle, sharpening's radius): Amount cannot scale them, so they are set as
# stored whenever the preset is applied at all.
_UNSCALED = editing._MODIFIER_KEYS
_CURVE_SAMPLES = 9          # control points a curve between two curves is given


def _part_set(edit: dict[str, Any], part: str) -> bool:
    """Whether a normalized edit's `part` differs from a fresh edit's."""
    neutral = editing.normalize(None)
    if part == "layers":
        return bool(edit["masks"])
    return any(edit[k] != neutral[k] for k in PARTS[part])


def modified_parts(edit: dict[str, Any] | None) -> list[str]:
    """The parts "Modified" ticks: every one the edit sets. Auto light is never
    set by an edit, so never ticked by this."""
    e = editing.normalize(edit)
    return [p for p in PARTS if _part_set(e, p)]


# Not parts a preset can carry, but work on a photo all the same, which
# replacing or clearing its edit takes away with the rest.
REPAIRS: tuple[str, ...] = ("healing", "redeye")


def replaced_parts(before: dict[str, Any] | None, after: dict[str, Any] | None) -> list[str]:
    """The parts `before` sets (`modified_parts`) that are not the same in
    `after`, and its repairs likewise: what a change takes from a photo's edit.
    A layer counts when one of the photo's own is gone or changed; layers
    added beside them take nothing."""
    b, a = editing.normalize(before), editing.normalize(after)
    out = []
    for part in PARTS:
        if not _part_set(b, part):
            continue
        if part == "layers":
            changed = any(m not in a["masks"] for m in b["masks"])
        elif part == "mixer":
            # The picked colours as the layers: one of the photo's own gone or
            # changed counts, one added beside them does not.
            changed = b["hsl"] != a["hsl"] or any(
                q not in (a["points"] or []) for q in (b["points"] or []))
        else:
            changed = any(b[k] != a[k] for k in PARTS[part])
        if changed:
            out.append(part)
    return out + [k for k in REPAIRS if b[k] is not None and b[k] != a[k]]


def replaced_by(base: dict[str, Any] | None, preset: dict[str, Any],
                amount: float = 100) -> list[str]:
    """What laying `preset` on `base` replaces (`replaced_parts`), worked out
    without the photo, as the parts with no values need it: Keep brightness
    only bends the curve under what it lands on, and Auto light counts as
    replacing Light wherever the edit has one of the sliders it sets."""
    laid = {**preset, "parts": [p for p in preset["parts"] if p not in FROM_THE_PHOTO]}
    b = editing.normalize(base)
    out = set(replaced_parts(b, apply(b, laid, amount)))
    if (AUTO_LIGHT in preset["parts"] and amount > 0
            and any(b[k] != editing.DEFAULT_EDIT[k] for k in AUTO_KEYS)):
        out.add("light")
    return [p for p in (*PARTS, *REPAIRS) if p in out]


def make(edit: dict[str, Any] | None, parts: list[str]) -> dict[str, Any]:
    """What is stored for a preset: the ticked parts, and the edit with only
    their keys. A face's own portrait settings and a layer's tag from another
    preset are dropped; heals and red eye are not a part at all."""
    unknown = set(parts) - set(PART_NAMES)
    assert not unknown, f"unknown preset parts: {sorted(unknown)}"
    assert parts, "a preset carries at least one part"
    e = editing.normalize(edit)
    stored: dict[str, Any] = {"pv": editing.process_version(e)}
    for part in PARTS:
        if part not in parts:
            continue
        for k in PARTS[part]:
            stored[k] = copy.deepcopy(e[k])
    if "portrait" in stored:
        stored["portrait"] = portrait_mod.without_faces(stored["portrait"])
    if "masks" in stored:
        for m in stored["masks"]:
            m.pop("preset", None)
    for q in stored.get("points") or []:
        q.pop("preset", None)
    return {"parts": [p for p in PART_NAMES if p in parts], "edit": stored}


def upgrade(stored: dict[str, Any]) -> dict[str, Any]:
    """A stored preset in today's shape. One saved before presets had parts
    is a whole edit, which is what it carried: it becomes a preset of every
    part that edit sets, the opt-in ones included, as applying it used to set
    them too. Read-only: what is on disk is left as it was."""
    if "parts" in stored:
        return stored
    e = editing.normalize(stored.get("edit"))
    parts = modified_parts(e)
    if not parts:
        return {**stored, "parts": [], "edit": {}}
    return {**stored, **make(e, parts)}


def _lerp(b: float, p: float, a: float) -> float:
    return b + a * (p - b)


def _blend_curve(b: list[list[float]], p: list[list[float]], a: float) -> list[list[float]]:
    """A curve `a` of the way from `b` to `p`, as the curves themselves, not as
    their control points (two curves need not share a single x)."""
    if a == 1:
        return copy.deepcopy(p)
    lb, lp = editing._curve_lut(b), editing._curve_lut(p)
    n = len(lb)
    xs = np.linspace(0.0, 1.0, _CURVE_SAMPLES)
    idx = np.rint(xs * (n - 1)).astype(int)
    ys = np.clip(lb[idx] + a * (lp[idx] - lb[idx]), 0.0, 1.0)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def _blend_hsl(b: dict | None, p: dict | None, a: float) -> dict | None:
    out: dict[str, dict[str, float]] = {}
    for band in editing.HSL_BANDS:
        vals = {}
        for key in editing.HSL_KEYS:
            vb = ((b or {}).get(band) or {}).get(key, 0)
            vp = ((p or {}).get(band) or {}).get(key, 0)
            vals[key] = _lerp(vb, vp, a)
        out[band] = vals
    return editing.normalize_hsl(out)


def _scaled_points(points: list | None, preset_id: str | None, a: float) -> list:
    """A preset's picked colours as they land: their values `a` of the way from
    neutral, as a layer's sliders are, tagged with the preset when it has one."""
    out = []
    for q in copy.deepcopy(points or []):
        for key in editing.HSL_KEYS:
            if key in q:
                q[key] = _lerp(0, q[key], a)
        if preset_id is not None:
            q["preset"] = preset_id
        out.append(q)
    return out


def _blend_grading(b: dict | None, p: dict | None, a: float) -> dict | None:
    """Strength and lightness move towards the preset's; a zone's hue is the
    preset's (a hue halfway between two is not a weaker version of either)."""
    nb = grading_mod.normalize(b) or {}
    np_ = grading_mod.normalize(p) or {}
    out: dict[str, Any] = {}
    for zone in grading_mod.ZONES:
        zb, zp = nb.get(zone) or {}, np_.get(zone) or {}
        out[zone] = {"sat": _lerp(zb.get("sat", 0), zp.get("sat", 0), a),
                     "lum": _lerp(zb.get("lum", 0), zp.get("lum", 0), a),
                     "hue": zp.get("hue", zb.get("hue", 0)) if a > 0 else zb.get("hue", 0)}
    # Stored only when off their defaults, so a missing one is its default.
    for key in ("balance", "blending"):
        d = grading_mod.DEFAULT_GRADING[key]
        out[key] = _lerp(nb.get(key, d), np_.get(key, d), a)
    return grading_mod.normalize(out)


def _blend_portrait(b: dict | None, p: dict | None, a: float) -> dict | None:
    """The panel's values move towards the preset's; the photo's own faces'
    settings stay, as the preset has none."""
    nb = b or {}
    out: dict[str, Any] = {k: _lerp(nb.get(k, 0), (p or {}).get(k, 0), a)
                           for k in portrait_mod.DEFAULT_PORTRAIT}
    if nb.get("faces"):
        out["faces"] = copy.deepcopy(nb["faces"])
    return portrait_mod.normalize(out)


def _scaled_layer(m: dict[str, Any], preset_id: str, a: float) -> dict[str, Any]:
    """A preset's layer as it lands: tagged, with its sliders `a` of the way
    from neutral, as a layer the photo did not have starts from nothing."""
    out = copy.deepcopy(m)
    out["preset"] = preset_id
    adj = out["adj"]
    for k in editing.LOCAL_KEYS:
        if k in _UNSCALED:
            continue
        adj[k] = _lerp(editing.DEFAULT_EDIT[k], adj[k], a)
    # The rest as the photo's own parts scale (see `apply`), from neutral.
    for k in editing.CURVE_KEYS:
        adj[k] = _blend_curve(editing.DEFAULT_EDIT[k], adj[k], a)
    adj["hsl"] = _blend_hsl(None, adj["hsl"], a)
    adj["points"] = _scaled_points(adj["points"], None, a)
    adj["grading"] = _blend_grading(None, adj["grading"], a)
    for k, strength in (("lut", "amount"), ("film", "strength")):
        if adj[k] is not None:
            adj[k][strength] = adj[k][strength] * min(a, 1.0)
    return out


def apply(base: dict[str, Any] | None, preset: dict[str, Any], amount: float = 100,
          auto: dict[str, Any] | None = None) -> dict[str, Any]:
    """`base` with `preset` laid on it at `amount` percent (0-200), normalized.

    At 100 each ticked part takes the stored value; at 0 the edit is `base`
    as it was; in between, and up to 200, each value moves along the line from
    the photo's to the preset's (clamped to its slider). What cannot be scaled
    (a look, a film stock, a crop, the settings in `_UNSCALED`) is set as stored
    whenever the amount is above 0. A look's and a film's own strength scale up
    to 100 and no further, as those are blends.

    `auto` is the photo's auto tone, `editing.auto_tone` of it, which the
    caller computes when the preset carries `auto_light`.
    """
    lo, hi = AMOUNT_RANGE
    assert lo <= amount <= hi, f"amount {amount} outside {AMOUNT_RANGE}"
    parts = preset["parts"]
    assert (AUTO_LIGHT in parts) == (auto is not None), \
        "auto light needs the photo's auto tone, and only it does"
    a = amount / 100.0
    b = editing.normalize(base)
    p = editing.normalize(preset["edit"])
    out = copy.deepcopy(b)
    on = a > 0
    for part in parts:
        if part in FROM_THE_PHOTO:
            continue
        if part in _SLIDER_PARTS:
            for k in PARTS[part]:
                out[k] = (p[k] if on else b[k]) if k in _UNSCALED else _lerp(b[k], p[k], a)
        elif part == "curve":
            for k in editing.CURVE_KEYS:
                out[k] = _blend_curve(b[k], p[k], a)
        elif part == "mixer":
            out["hsl"] = _blend_hsl(b["hsl"], p["hsl"], a)
            # The photo's own picked colours stay, and the preset's are added
            # beside them, as its layers are: every look carries the mixer, and
            # applying one must not take away colours picked on this photo.
            pid = preset["id"]
            keep = [q for q in (b["points"] or []) if q.get("preset") != pid]
            added = _scaled_points(p["points"], pid, a) if on else []
            out["points"] = (keep + added)[:editing.POINTS_MAX] or None
        elif part == "grading":
            out["grading"] = _blend_grading(b["grading"], p["grading"], a)
        elif part == "portrait":
            out["portrait"] = _blend_portrait(b["portrait"], p["portrait"], a)
        elif part == "look":
            lut = copy.deepcopy(p["lut"]) if on else b["lut"]
            if on and lut is not None:
                lut["amount"] = lut["amount"] * min(a, 1.0)
            out["lut"] = lut
        elif part == "film":
            film = copy.deepcopy(p["film"]) if on else b["film"]
            if on and film is not None:
                film["strength"] = film["strength"] * min(a, 1.0)
            out["film"] = film
        elif part == "layers":
            pid = preset["id"]
            keep = [m for m in b["masks"] if m.get("preset") != pid]
            added = [_scaled_layer(m, pid, a) for m in p["masks"]] if on else []
            out["masks"] = (keep + added)[:editing.MASK_MAX]
        else:   # crop, lens, watermark: a place or a stamp, set as stored
            for k in PARTS[part]:
                out[k] = copy.deepcopy(p[k] if on else b[k])
    if auto is not None:
        for k in AUTO_KEYS:
            out[k] = _lerp(b[k], auto[k], a)
    # The maths is the photo's, unless it has nothing on it yet: then the
    # preset's, as editing.merge_additive decides it.
    if editing.is_empty(base):
        out.pop("pv", None)
        if "pv" in p:
            out["pv"] = p["pv"]
    return editing.normalize(out)


# ----- keep the photo's brightness ----------------------------------------
#
# The correction is a gamma under the master curve, C(x ** g): it moves the
# midtone and leaves the curve's own black and white where the look put them,
# so a faded black stays faded. Solved by bisection on renders of a small copy
# of the photo; a few milliseconds each at this size.
#
# Except where the look's black or white is beyond the photo's midtone: a
# fade to 0.07 is above the whole of a night frame, whose median sits near
# 0.06, and no gamma brings back a midtone the floor is above. So the ends of
# every curve, the master and the three channels, are first fitted to the
# photo (`_fit_ends`): a lifted black may come up by at most half the frame's
# median, and a lowered white come down by at most half of what lies above it.
# A night frame gets a gentler fade, in proportion. What the film stage does
# after the curves is not undone: film's own black is a few levels above zero.

KEEP_EDGE = 256                    # long edge of the copy the photo is measured on
_KEEP_GAMMA = (0.5, 2.0)           # about a stop either way at a mid grey
_KEEP_STEPS = 8                    # the median lands within a few hundredths of a stop
_KEEP_TOLERANCE = 0.003            # share of the frame the correction may newly clip
_BLOWN, _BLOCKED = 0.985, 0.02     # encoded luma counted as white, and as black
_NOTHING = 1.5 / 255.0             # a median this low has no midtone to keep
# Where the corrected curve is sampled: closer together at the ends, where a
# look keeps its fade and its roll-off.
_KEEP_XS = (0.0, 0.06, 0.14, 0.25, 0.38, 0.52, 0.68, 0.84, 1.0)
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _light(rgb: np.ndarray) -> tuple[float, float, float]:
    """A rendered frame's median encoded luma, and its shares blown and blocked."""
    y = (rgb.astype(np.float32) / 255.0) @ _LUMA
    return float(np.median(y)), float((y >= _BLOWN).mean()), float((y <= _BLOCKED).mean())


_END_SHARE = 0.5                   # of the midtone, how far a black may lift
_END_REACH = 0.35                  # how far into the tones the eased ends reach


def _fit_ends(curve: list[list[float]], median: float) -> list[list[float]]:
    """`curve` with its lifted black and lowered white eased to suit a frame of
    this median encoded luma, sampled at _KEEP_XS; the curve itself when its
    ends already suit it. The easing fades out over `_END_REACH` of the tones,
    and keeps the curve rising: it takes away more where the curve is lower."""
    lut = editing._curve_lut(curve, 1024)
    floor_cut = max(0.0, float(lut[0]) - _END_SHARE * median)
    ceil_cut = max(0.0, (1.0 - _END_SHARE * (1.0 - median)) - float(lut[-1]))
    if floor_cut < 1e-3 and ceil_cut < 1e-3:
        return curve
    grid = np.linspace(0.0, 1.0, 1024)
    lo = np.clip(1.0 - grid / _END_REACH, 0.0, 1.0) ** 2
    hi = np.clip((grid - (1.0 - _END_REACH)) / _END_REACH, 0.0, 1.0) ** 2
    eased = np.clip(lut - floor_cut * lo + ceil_cut * hi, 0.0, 1.0)
    return [[x, float(np.interp(x, grid, eased))] for x in _KEEP_XS]


def _under_gamma(curve: list[list[float]], g: float) -> list[list[float]]:
    lut = editing._curve_lut(curve, 1024)
    return [[x, float(np.interp(x ** g, np.linspace(0.0, 1.0, 1024), lut))] for x in _KEEP_XS]


def keep_light(reference: dict[str, Any] | None, edit: dict[str, Any],
               render) -> dict[str, Any]:
    """`edit`, a preset laid on `reference`, with its master curve bent so the
    photo's median brightness is what it was in `reference`.

    `render(edit)` is the photo rendered small, as uint8 RGB. The correction
    never blows out or blocks up more of the frame than the photo had or the
    look itself does, whichever is more (beyond `_KEEP_TOLERANCE`): a night
    sky the look's fade lifted may go back to the black it was, and no
    further. Where the midtone cannot come all the way back within that, it
    comes as far as it can. A frame with no midtone to measure, all but black
    or all but white, is left as the preset made it.
    """
    e = editing.normalize(edit)
    target, hi0, lo0 = _light(render(reference))
    if target <= _NOTHING or target >= _BLOWN:
        return e
    e = {**e, **{k: _fit_ends(e[k], target) for k in editing.CURVE_KEYS}}

    def at(g: float) -> dict[str, Any]:
        return {**e, "curve": _under_gamma(e["curve"], g)}

    m1, hi1, lo1 = _light(render(e))
    if abs(m1 - target) < 0.002:
        return e
    lo, hi = _KEEP_GAMMA
    for _ in range(_KEEP_STEPS):          # a larger g darkens
        g = (lo * hi) ** 0.5
        if _light(render(at(g)))[0] > target:
            lo = g
        else:
            hi = g
    g = (lo * hi) ** 0.5
    # Clipping guard: back towards 1 until the correction adds no more blown
    # (when it brightens) or blocked (when it darkens) than the look had.
    k = 1 if g < 1 else 2
    limit = (max(hi0, hi1) if k == 1 else max(lo0, lo1)) + _KEEP_TOLERANCE
    if _light(render(at(g)))[k] > limit:
        near, far = 1.0, g
        for _ in range(_KEEP_STEPS):
            mid = (near * far) ** 0.5
            if _light(render(at(mid)))[k] > limit:
                far = mid
            else:
                near = mid
        g = near
    return editing.normalize(at(g) if abs(g - 1) > 1e-3 else e)
