"""Built-in edit presets that ship with the app.

These sit alongside the user's own presets (`userstate`) but are read-only: the
UI offers them as starting points, and saving over one creates a normal user
preset instead. Each is a plain edit dict, so anything the editor can do a
preset can carry — including the colour mixer, the grading wheels, the channel
curves and local masks.

Six groups, and the split is by what you are trying to do rather than by which
sliders move:

  Portrait  the skin sets. Mostly white balance and the orange/yellow bands,
            because that is where a face that came out sallow actually lives.
  Look      a colour identity laid over a correct photo — what a .cube LUT
            would be, expressed parametrically so every number stays editable.
  Mono      three ways of losing the colour, which are three different pictures.
  Scene     subject-led starting points: landscape, interior, food, snow.
  Fix       one problem each, named after the problem.
  Car       the two situations a car shoot produces — a parked hero car that
            wants paint contrast, and a moving one that wants the subject held
            sharp while everything else smears.

Two things worth knowing before adding to this file:

  - **Saturation runs after the mixer and the wheels**, so a toned monochrome
    cannot be built from `saturation: -100` plus a grade — the grade is what the
    saturation then removes. `Mono · Sepia` gets its colour back from the film
    stage instead, which is the one colour stage that runs *after* saturation.
  - **The white-balance sliders are gains, not Kelvin.** They stop at 1.25/0.75
    per channel, so the tungsten and fluorescent presets here are as far as a
    preset can honestly go; the eyedropper beside them is what finds the exact
    pair for one photo.
"""
from __future__ import annotations

from typing import Any

from . import film as film_mod


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


def _segment(name: str, group: str, adj: dict[str, Any],
             *, feather: int = 0) -> dict[str, Any]:
    """An automatic mask on one of the segmenter's groups. Feather stays at zero
    by default for the reason `editing._default_mask` gives: the model's own edge
    is already soft where it should be, and blurring it throws that away."""
    return {
        "type": "auto", "name": name, "group": group, "enabled": True,
        "invert": False, "feather": feather, "amount": 100, "adj": adj,
    }


BUILTIN_PRESETS: list[dict[str, Any]] = [
    # ----- Portrait -------------------------------------------------------
    {
        "id": "skin-bright",
        "name": "Bright skin",
        "group": "Portrait",
        "hint": "Cools the light and takes the yellow out of the skin itself.",
        "edit": {
            "exposure": 0.12, "shadows": 18, "whites": 10, "blacks": 8,
            "temp": -18, "tint": 6, "vibrance": 8, "texture": -12,
            "hsl": {
                "red": {"sat": -6},
                "orange": {"sat": -10, "lum": 16},
                "yellow": {"sat": -28, "lum": 10},
            },
        },
    },
    {
        "id": "skin-airy",
        "name": "Airy & fair",
        "group": "Portrait",
        "hint": "High key: lifted blacks, low contrast, a cool highlight.",
        "edit": {
            "exposure": 0.28, "contrast": -12, "highlights": -15, "shadows": 26,
            "whites": 14, "blacks": 12, "temp": -14, "tint": 4,
            "vibrance": 6, "texture": -18,
            "hsl": {
                "orange": {"sat": -14, "lum": 22},
                "yellow": {"sat": -35, "lum": 12},
            },
            "grading": {"highlights": {"hue": 205, "sat": 10}},
        },
    },
    {
        "id": "skin-clean",
        "name": "Clean beauty",
        "group": "Portrait",
        "hint": "Pores smoothed, edges kept. Sharpening held off the skin.",
        "edit": {
            "highlights": -12, "shadows": 12, "clarity": -8, "texture": -28,
            "denoise": 12, "sharpen": 24, "sharpen_radius": 40,
            "sharpen_masking": 65,
            "hsl": {
                "red": {"sat": -10},
                "orange": {"lum": 12},
                "yellow": {"sat": -18},
            },
        },
    },
    {
        "id": "skin-glow",
        "name": "Warm glow",
        "group": "Portrait",
        "hint": "The other direction: warm light, soft bloom, held highlights.",
        "edit": {
            "highlights": -22, "shadows": 18, "temp": 8, "vibrance": 10,
            "texture": -10, "glow": 20,
            "hsl": {"orange": {"sat": 6, "lum": 8}},
            "grading": {"shadows": {"hue": 28, "sat": 12},
                        "highlights": {"hue": 45, "sat": 10}},
        },
    },
    {
        "id": "skin-backlit",
        "name": "Backlit rescue",
        "group": "Portrait",
        "hint": "Face out of the shadow without blowing the window behind it.",
        "edit": {
            "exposure": 0.35, "highlights": -50, "shadows": 45, "whites": -10,
            "blacks": -8, "temp": -6, "clarity": 10, "dehaze": 14,
        },
    },
    {
        "id": "skin-tungsten",
        "name": "Tungsten indoors",
        "group": "Portrait",
        "hint": "Household lamps. Strong, and still short of a deep cast — "
                "finish it with the eyedropper.",
        "edit": {
            "exposure": 0.15, "shadows": 20, "temp": -38, "tint": 12,
            "vibrance": 8,
            "hsl": {"orange": {"lum": 12}, "yellow": {"sat": -30, "lum": 8}},
        },
    },
    {
        "id": "skin-fluorescent",
        "name": "Fluorescent indoors",
        "group": "Portrait",
        "hint": "Office strip lights: the cast is green as much as yellow.",
        "edit": {
            "shadows": 16, "temp": -12, "tint": 26,
            "hsl": {
                "orange": {"lum": 10},
                "yellow": {"sat": -20, "lum": 6},
                "green": {"sat": -25},
            },
        },
    },
    {
        "id": "skin-local",
        "name": "Skin only (local)",
        "group": "Portrait",
        "hint": "Cools and smooths the skin alone — the background keeps its "
                "own light. Needs the segmentation model.",
        "edit": {
            "masks": [_segment("Skin", "skin", {
                "exposure": 0.18, "temp": -14, "tint": 5, "texture": -22,
            })],
        },
    },

    # ----- Look -----------------------------------------------------------
    {
        "id": "look-teal-orange",
        "name": "Teal & orange",
        "group": "Look",
        "hint": "Cold shadows against a warm subject. The blockbuster grade.",
        "edit": {
            "contrast": 16, "highlights": -18, "shadows": 10, "saturation": -4,
            "vibrance": 12,
            "grading": {
                "shadows": {"hue": 195, "sat": 26},
                "midtones": {"hue": 25, "sat": 6},
                "highlights": {"hue": 32, "sat": 18},
                "blending": 55,
            },
        },
    },
    {
        "id": "look-faded",
        "name": "Faded matte",
        "group": "Look",
        "hint": "Blacks lifted off the floor, whites pulled off the ceiling.",
        "edit": {
            "contrast": -6, "saturation": -14,
            "curve": [[0.0, 0.08], [0.25, 0.30], [0.75, 0.78], [1.0, 0.94]],
            "grading": {"shadows": {"hue": 210, "sat": 14},
                        "highlights": {"hue": 40, "sat": 8}},
        },
    },
    {
        "id": "look-bleach",
        "name": "Bleach bypass",
        "group": "Look",
        "hint": "Silver left in the print: hard contrast, colour nearly gone.",
        "edit": {
            "contrast": 38, "whites": 14, "blacks": -18, "saturation": -48,
            "clarity": 26, "sharpen": 25,
        },
    },
    {
        "id": "look-moody",
        "name": "Moody blue",
        "group": "Look",
        "hint": "Cold, closed down, weight in the shadows.",
        "edit": {
            "exposure": -0.1, "contrast": 14, "highlights": -28, "shadows": -8,
            "blacks": -14, "temp": -26, "tint": -4, "saturation": -10,
            "vignette": 18,
            "grading": {"shadows": {"hue": 222, "sat": 22},
                        "highlights": {"hue": 200, "sat": 8}},
        },
    },
    {
        "id": "look-golden",
        "name": "Golden hour",
        "group": "Look",
        "hint": "Low warm sun, wherever the shot was actually taken.",
        "edit": {
            "highlights": -25, "shadows": 18, "temp": 22, "tint": 6,
            "vibrance": 18, "glow": 12,
            "grading": {"shadows": {"hue": 20, "sat": 10},
                        "highlights": {"hue": 40, "sat": 20}},
        },
    },
    {
        "id": "look-cross",
        "name": "Cross process",
        "group": "Look",
        "hint": "Channel curves pulling apart — E-6 chemistry on C-41 film.",
        "edit": {
            "contrast": 10, "saturation": 8,
            "curve_r": [[0.0, 0.04], [0.5, 0.56], [1.0, 1.0]],
            "curve_g": [[0.0, 0.0], [0.5, 0.48], [1.0, 1.0]],
            "curve_b": [[0.0, 0.14], [0.5, 0.46], [1.0, 0.88]],
        },
    },
    {
        "id": "look-cine-matte",
        "name": "Cinematic matte",
        "group": "Look",
        "hint": "Flat toe, split tone, a little fall-off at the corners.",
        "edit": {
            "contrast": 12, "highlights": -20, "shadows": 14, "saturation": -6,
            "vignette": 14,
            "curve": [[0.0, 0.06], [0.5, 0.48], [1.0, 0.96]],
            "grading": {
                "shadows": {"hue": 205, "sat": 18},
                "midtones": {"hue": 30, "sat": 6},
                "highlights": {"hue": 45, "sat": 10},
                "balance": 10,
            },
        },
    },
    {
        "id": "look-vintage",
        "name": "Vintage warm",
        "group": "Look",
        "hint": "Haze put back, colours aged, corners drawn in.",
        "edit": {
            "temp": 14, "saturation": -16, "dehaze": -18, "glow": 10,
            "vignette": 20,
            "curve": [[0.0, 0.10], [0.5, 0.52], [1.0, 0.92]],
            "hsl": {
                "orange": {"sat": 8},
                "green": {"hue": 10, "sat": -18},
                "blue": {"sat": -20, "lum": -6},
            },
        },
    },
    {
        "id": "look-neon",
        "name": "Neon night",
        "group": "Look",
        "hint": "Lifted shadows, blooming signs, magenta in the highlights.",
        "edit": {
            "exposure": -0.12, "contrast": 18, "highlights": -30, "shadows": 30,
            "blacks": -20, "temp": -18, "vibrance": 30, "clarity": 14,
            "glow": 30,
            "grading": {"shadows": {"hue": 250, "sat": 20},
                        "highlights": {"hue": 320, "sat": 14}},
        },
    },
    {
        "id": "look-pastel",
        "name": "Pastel",
        "group": "Look",
        "hint": "Everything a little washed and a little lighter.",
        "edit": {
            "exposure": 0.15, "contrast": -18, "whites": -8, "blacks": 16,
            "saturation": -22, "vibrance": 12,
            "hsl": {
                "orange": {"lum": 12},
                "green": {"sat": -18, "lum": 12},
                "blue": {"sat": -20, "lum": 10},
            },
            "grading": {"shadows": {"hue": 190, "sat": 10},
                        "highlights": {"hue": 320, "sat": 10}},
        },
    },

    # ----- Mono -----------------------------------------------------------
    {
        "id": "mono-classic",
        "name": "Classic",
        "group": "Mono",
        "hint": "Straight black and white with the contrast a print wants.",
        "edit": {
            "contrast": 20, "saturation": -100, "clarity": 14, "sharpen": 25,
        },
    },
    {
        "id": "mono-highkey",
        "name": "High key",
        "group": "Mono",
        "hint": "Bright, open, almost no black in the frame.",
        "edit": {
            "exposure": 0.3, "contrast": -12, "shadows": 20, "whites": 20,
            "blacks": 18, "saturation": -100,
        },
    },
    {
        "id": "mono-noir",
        "name": "Noir",
        "group": "Mono",
        "hint": "Crushed blacks, hard light, heavy corners.",
        "edit": {
            "contrast": 42, "whites": 12, "blacks": -32, "saturation": -100,
            "clarity": 28, "sharpen": 30, "vignette": 32,
        },
    },
    {
        "id": "mono-sepia",
        "name": "Sepia",
        "group": "Mono",
        "hint": "Toned by the film stage, which is the only colour that "
                "survives a saturation of -100.",
        "edit": {
            "contrast": 12, "saturation": -100, "clarity": 10,
            "film": film_mod.stock("Warm portrait"),
        },
    },

    # ----- Scene ----------------------------------------------------------
    {
        "id": "scene-landscape",
        "name": "Landscape pop",
        "group": "Scene",
        "hint": "Depth in the sky, separation in the foliage.",
        "edit": {
            "contrast": 14, "highlights": -22, "shadows": 16, "whites": 12,
            "blacks": -12, "clarity": 22, "dehaze": 10, "vibrance": 24,
            "saturation": -4, "sharpen": 30,
            "hsl": {"green": {"hue": -8, "sat": 10}, "blue": {"sat": 12, "lum": -10}},
        },
    },
    {
        "id": "scene-sky",
        "name": "Deepen the sky",
        "group": "Scene",
        "hint": "A gradient down from the top — drag it to the horizon.",
        "edit": {
            "masks": [{
                "type": "linear", "name": "Sky", "enabled": True, "invert": False,
                "feather": 100, "amount": 100,
                "x1": 0.5, "y1": 0.05, "x2": 0.5, "y2": 0.45,
                "adj": {"exposure": -0.4, "temp": -10, "clarity": 15,
                        "saturation": 12},
            }],
        },
    },
    {
        "id": "scene-interior",
        "name": "Interior",
        "group": "Scene",
        "hint": "Windows held, room lifted, the lamp cast taken out.",
        "edit": {
            "exposure": 0.2, "highlights": -35, "shadows": 30, "whites": 8,
            "temp": -20, "tint": 6, "clarity": 12, "sharpen": 25,
        },
    },
    {
        "id": "scene-food",
        "name": "Food",
        "group": "Scene",
        "hint": "Warm, textured, green pulled back so the plate stays the subject.",
        "edit": {
            "contrast": 12, "highlights": -18, "shadows": 12, "temp": 6,
            "clarity": 16, "texture": 15, "vibrance": 20, "sharpen": 28,
            "vignette": 12,
            "hsl": {"orange": {"sat": 10, "lum": 6}, "yellow": {"sat": 8},
                    "green": {"sat": -8}},
        },
    },
    {
        "id": "scene-snow",
        "name": "Snow & beach",
        "group": "Scene",
        "hint": "Undoes the grey the meter makes of a bright scene.",
        "edit": {
            "exposure": 0.25, "highlights": -25, "whites": 15, "blacks": 6,
            "temp": -10, "tint": 4, "saturation": -6, "clarity": 10,
        },
    },

    # ----- Fix ------------------------------------------------------------
    {
        "id": "fix-underexposed",
        "name": "Underexposed",
        "group": "Fix",
        "hint": "A stop back, with the noise it uncovers dealt with.",
        "edit": {
            "exposure": 0.8, "contrast": 8, "shadows": 35, "whites": 10,
            "blacks": -8, "denoise": 20,
        },
    },
    {
        "id": "fix-overexposed",
        "name": "Overexposed",
        "group": "Fix",
        "hint": "Highlights pulled down as far as they still hold detail.",
        "edit": {
            "exposure": -0.5, "contrast": 8, "highlights": -55, "shadows": 15,
            "whites": -25,
        },
    },
    {
        "id": "fix-haze",
        "name": "Hazy day",
        "group": "Fix",
        "hint": "Through the atmosphere, with the black point put back.",
        "edit": {
            "contrast": 12, "blacks": -14, "dehaze": 35, "clarity": 14,
            "vibrance": 12,
        },
    },
    {
        "id": "fix-noise",
        "name": "High ISO",
        "group": "Fix",
        "hint": "Denoise with sharpening masked to the edges. Judge it at 1:1.",
        "edit": {
            "contrast": 6, "texture": -10, "denoise": 45, "sharpen": 22,
            "sharpen_detail": 15, "sharpen_masking": 70,
        },
    },
    {
        "id": "fix-flat",
        "name": "Flat JPEG",
        "group": "Fix",
        "hint": "For a camera-neutral file that came out lifeless.",
        "edit": {
            "contrast": 16, "whites": 12, "blacks": -12, "clarity": 12,
            "vibrance": 14, "sharpen": 24,
        },
    },

    # ----- Car ------------------------------------------------------------
    {
        "id": "car-gloss",
        "name": "Glossy paint",
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
        "name": "Studio white",
        "group": "Car",
        "hint": "Clean bright backdrop, neutral body colour.",
        "edit": {
            "exposure": 0.25, "contrast": 10, "highlights": -20, "whites": 22,
            "blacks": -12, "clarity": 14, "saturation": -6, "sharpen": 25,
        },
    },
    {
        "id": "car-night",
        "name": "Night neon",
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
        "name": "Golden hour",
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
        "name": "Speed pan",
        "group": "Car",
        "hint": "Horizontal smear everywhere but the car — fakes a panned shot.",
        "edit": {
            "contrast": 12, "clarity": 20, "sharpen": 35,
            "masks": [_around_subject("Speed", {"motion": 45, "motion_angle": 0})],
        },
    },
    {
        "id": "car-bokeh",
        "name": "Isolate (bokeh)",
        "group": "Car",
        "hint": "Throws the background out of focus and darkens it slightly.",
        "edit": {
            "contrast": 10, "clarity": 16, "sharpen": 28,
            "masks": [_around_subject("Bokeh", {"blur": 38, "exposure": -0.3})],
        },
    },
    {
        "id": "car-plate",
        "name": "Blur a plate",
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
