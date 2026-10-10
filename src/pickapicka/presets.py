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
    "color": ("temp", "tint", "vibrance", "saturation"),
    "detail": ("texture", "clarity", "dehaze", "denoise", "sharpen", "sharpen_radius",
               "sharpen_detail", "sharpen_masking", "vignette"),
    "creative": ("blur", "motion", "motion_angle", "glow", "pixelate"),
    "curve": editing.CURVE_KEYS,
    "mixer": ("hsl",),
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
PART_NAMES: tuple[str, ...] = (*PARTS, AUTO_LIGHT)
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
        if part == AUTO_LIGHT:
            continue
        if part in _SLIDER_PARTS:
            for k in PARTS[part]:
                out[k] = (p[k] if on else b[k]) if k in _UNSCALED else _lerp(b[k], p[k], a)
        elif part == "curve":
            for k in editing.CURVE_KEYS:
                out[k] = _blend_curve(b[k], p[k], a)
        elif part == "mixer":
            out["hsl"] = _blend_hsl(b["hsl"], p["hsl"], a)
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
