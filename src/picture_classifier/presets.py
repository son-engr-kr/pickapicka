"""Built-in edit presets that ship with the app.

These sit alongside the user's own presets (`userstate`) but are read-only: the
UI offers them as starting points, and saving over one creates a normal user
preset instead. Each is a plain edit dict, so anything the editor can do a
preset can carry — including local masks.

The car set is tuned for the two situations a car shoot actually produces: a
parked hero car that wants paint contrast and clean highlights, and a moving car
that wants the subject held sharp while everything else smears.
"""
from __future__ import annotations

from typing import Any

# A radial mask centred on the frame, sized for a car filling most of it.
# `invert` means "everything except the car", which is what the background
# treatments below act on.
def _around_subject(name: str, adj: dict[str, Any], *, feather: int = 60,
                    rx: float = 0.42, ry: float = 0.34) -> dict[str, Any]:
    # Named per preset: these stack, and two masks both called "Background"
    # would be indistinguishable in the mask list.
    return {
        "type": "radial", "name": name, "enabled": True, "invert": True,
        "feather": feather, "amount": 100,
        "cx": 0.5, "cy": 0.55, "rx": rx, "ry": ry, "angle": 0.0,
        "adj": adj,
    }


BUILTIN_PRESETS: list[dict[str, Any]] = [
    {
        "id": "car-gloss",
        "name": "Car · Glossy paint",
        "group": "Car",
        "hint": "Deep paint, controlled highlights — the parked hero shot.",
        "edit": {
            "contrast": 22, "highlights": -35, "shadows": 12, "whites": -8,
            "blacks": -18, "clarity": 28, "vibrance": 18, "saturation": 6,
            "sharpen": 30,
        },
    },
    {
        "id": "car-studio",
        "name": "Car · Studio white",
        "group": "Car",
        "hint": "Clean bright backdrop, neutral body colour.",
        "edit": {
            "exposure": 0.25, "contrast": 10, "highlights": -20, "whites": 22,
            "blacks": -12, "clarity": 14, "saturation": -6, "sharpen": 25,
        },
    },
    {
        "id": "car-night",
        "name": "Car · Night neon",
        "group": "Car",
        "hint": "Lifted shadows, cool cast, headlights blooming.",
        "edit": {
            "exposure": -0.15, "contrast": 18, "highlights": -25, "shadows": 32,
            "blacks": -22, "temp": -22, "tint": 8, "vibrance": 28,
            "glow": 35, "clarity": 18, "vignette": -25,
        },
    },
    {
        "id": "car-sunset",
        "name": "Car · Golden hour",
        "group": "Car",
        "hint": "Warm low sun with a soft bloom off the bodywork.",
        "edit": {
            "exposure": 0.1, "contrast": 14, "highlights": -30, "shadows": 20,
            "temp": 26, "tint": 6, "vibrance": 22, "glow": 22, "clarity": 12,
            "vignette": -15,
        },
    },
    {
        "id": "car-speed",
        "name": "Car · Speed pan",
        "group": "Car",
        "hint": "Horizontal smear everywhere but the car — fakes a panned shot.",
        "edit": {
            "contrast": 12, "clarity": 20, "sharpen": 35,
            "masks": [_around_subject("Speed", {"motion": 45, "motion_angle": 0})],
        },
    },
    {
        "id": "car-bokeh",
        "name": "Car · Isolate (bokeh)",
        "group": "Car",
        "hint": "Throws the background out of focus and darkens it slightly.",
        "edit": {
            "contrast": 10, "clarity": 16, "sharpen": 28,
            "masks": [_around_subject("Bokeh", {"blur": 38, "exposure": -0.3})],
        },
    },
    {
        "id": "car-plate",
        "name": "Car · Blur a plate",
        "group": "Car",
        "hint": "A small mosaic patch — drag it onto the number plate.",
        # A radial rather than a brush: an unpainted brush mask normalizes away
        # to nothing, and a shape you can drag straight onto the plate is less
        # work than painting one anyway.
        "edit": {
            "masks": [{
                "type": "radial", "name": "Plate", "enabled": True, "invert": False,
                "feather": 10, "amount": 100,
                "cx": 0.5, "cy": 0.72, "rx": 0.12, "ry": 0.05, "angle": 0.0,
                "adj": {"pixelate": 55},
            }],
        },
    },
]


def list_builtins() -> list[dict[str, Any]]:
    """Built-in presets in API shape, tagged so the UI can mark them read-only."""
    return [
        {"id": p["id"], "name": p["name"], "group": p.get("group", ""),
         "hint": p.get("hint", ""), "edit": p["edit"], "builtin": True}
        for p in BUILTIN_PRESETS
    ]


def is_builtin(preset_id: str) -> bool:
    return any(p["id"] == preset_id for p in BUILTIN_PRESETS)
