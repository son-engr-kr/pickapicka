"""Looks that ship with the app: twenty presets, in six groups.

They are presets like any the user saves (see `presets`), read-only, and
listed after the user's own. What they are for is the kind of colour an
Instagram filter gives a photo, whether a landscape or a portrait: CSSgram's
reconstructions of those filters (github.com/una/CSSgram) were the reference
for what each kind of look does, as Instagram publishes none of its own.

Every look carries Keep brightness (`presets.keep_light`): applied, it is
measured on the photo and its tone curve bent until the photo's median
brightness is what it was, without blowing out or blocking up more of the
frame than the look itself does. Measured on 80 picks of a hiking shoot, the
same curves without it moved the midtone by up to a fifth of a stop, and by
over a stop when the curves did not keep near the diagonal through the
midtones; with it, a look no longer has a say in the exposure at all.

A preset sets the parts it carries, as stored, and leaves the rest of the
photo's edit alone. So a look is made of the parts that are a look rather
than a correction of this photo: the tone curves, the colour mixer and the
grading wheels, and film where it is film. None carries Light, so the
photo's exposure stays as it was, and only the two black and white looks
carry Color, for Saturation (which also puts Temperature and Tint at zero;
in black and white those only move the greys). Warmth, coolness and a
lifted, faded black are the wheels and the curves.

Detail & effects is carried by the five looks that need its clarity,
texture or vignette (Moody, Heritage, Noir, Anime, Comic), which means they
also set sharpening and noise reduction to theirs, none but Comic's. Creative
by the five with a glow, which also takes away a defocus or motion blur.

Film looks ease the stock's characteristic curve (`_film`), and a look's
master curve stays close to the diagonal through the midtones even so: Keep
brightness can bring any midtone back, but the less it has to bend the curve,
the more of the look's own shape is left.

The stage order decides two things here: the wheels run before Saturation,
so a black and white look cannot be toned by them (film, after the grade,
could), and the mixer runs before the wheels, so a band it turns is then
graded as the colour it became.
"""
from __future__ import annotations

from typing import Any

from . import editing, film as film_mod, presets

GROUPS: tuple[str, ...] = ("Bright & soft", "Vivid", "Moody", "Film", "Black & white", "Comic")

_ALL_BANDS = editing.HSL_BANDS


def _mix(every: dict[str, float] | None = None, **bands: dict[str, float]) -> dict[str, Any]:
    """A colour mixer: `every` for all eight bands, then each band's own on top."""
    out: dict[str, dict[str, float]] = {}
    for band in _ALL_BANDS:
        vals = dict(every or {})
        for k, v in bands.get(band, {}).items():
            vals[k] = vals.get(k, 0) + v
        if vals:
            out[band] = vals
    return out


def _wheels(shadows: tuple[float, float] | None = None, midtones: tuple[float, float] | None = None,
            highlights: tuple[float, float] | None = None, **globals_: float) -> dict[str, Any]:
    """Grading wheels, each zone as (hue, sat)."""
    out: dict[str, Any] = dict(globals_)
    for zone, v in (("shadows", shadows), ("midtones", midtones), ("highlights", highlights)):
        if v is not None:
            out[zone] = {"hue": v[0], "sat": v[1], "lum": 0}
    return out


def _film(stock: str, strength: float) -> dict[str, Any]:
    """A film stock with its characteristic curve eased (contrast 40, toe 30):
    the stocks' own toe sank a dark frame's midtones by a third of a stop,
    and a look keeps those where they were. The colour, halation and grain
    are the stock's."""
    f = film_mod.stock(stock)
    assert f is not None, f"no film stock {stock}"
    return {**f, "strength": strength, "contrast": 40, "toe": 30}


def _look(key: str, name: str, group: str, hint: str, **edit: Any) -> dict[str, Any]:
    assert group in GROUPS, group
    edit = {"pv": editing.PROCESS_VERSION, **edit}
    parts = presets.modified_parts(edit)
    assert "light" not in parts, f"{name}: a look leaves the photo's exposure alone"
    parts.append(presets.KEEP_LIGHT)
    return {"id": f"look-{key}", "name": name, "group": group, "hint": hint,
            "builtin": True, **presets.make(edit, parts)}


LOOKS: list[dict[str, Any]] = [
    # ----- Bright & soft ---------------------------------------------------
    _look("pastel", "Pastel", "Bright & soft",
          "Light and powdery: pale luminous colour, soft blacks, mint greens, lavender shadows, a glow.",
          # The whites stay white: dimmed, with the midtone kept, it read grey.
          curve=[[0, 0.07], [0.2, 0.22], [0.5, 0.52], [0.85, 0.87], [1, 1]],
          hsl=_mix({"sat": -35, "lum": 8}, green={"hue": 30}, yellow={"hue": 15}, blue={"hue": 12}),
          grading=_wheels((260, 22), (330, 8), (25, 16)),
          glow=25),
    _look("airy", "Airy", "Bright & soft",
          "Light and open, like a Japanese film snap: soft contrast, cyan shadows, pale minty greens.",
          curve=[[0, 0.08], [0.2, 0.23], [0.5, 0.52], [0.85, 0.85], [1, 0.95]],
          hsl=_mix(green={"hue": 28, "sat": -45, "lum": 12}, yellow={"hue": 18, "sat": -40, "lum": 8},
                   orange={"sat": -12, "lum": 8}, red={"sat": -15},
                   blue={"sat": -25, "lum": 12}, aqua={"sat": -25, "lum": 10}),
          grading=_wheels((190, 24), (180, 6), (50, 8)),
          glow=14),
    _look("peach", "Peach", "Bright & soft",
          "Warm peachy skin and a soft glow, greens turned olive, for portraits.",
          curve=[[0, 0.05], [0.2, 0.22], [0.85, 0.85], [1, 0.95]],
          curve_r=[[0, 0.01], [0.5, 0.53], [1, 1]],
          curve_b=[[0, 0.03], [0.5, 0.47], [1, 0.93]],
          hsl=_mix(orange={"hue": -6, "sat": -8, "lum": 10}, red={"hue": 5, "lum": 6},
                   green={"hue": -18, "sat": -40}, yellow={"hue": -10, "sat": -20}),
          grading=_wheels((10, 8), (18, 12), (25, 26)),
          glow=14),
    _look("cotton", "Cotton candy", "Bright & soft",
          "Dreamy pink and blue: pink highlights, blue shadows, mint greens.",
          curve=[[0, 0.07], [0.2, 0.23], [0.85, 0.84], [1, 0.93]],
          curve_r=[[0, 0.03], [1, 1]],
          curve_b=[[0, 0.08], [0.25, 0.27], [1, 0.98]],
          hsl=_mix({"sat": -25}, green={"hue": 35, "sat": -15}, yellow={"hue": 20}, blue={"hue": 18}),
          grading=_wheels((225, 28), (300, 10), (335, 26))),
    # ----- Vivid -----------------------------------------------------------
    _look("vivid", "Vivid", "Vivid",
          "Colour and a little punch, with cool shadows: a sunny day as it felt.",
          curve=[[0, 0], [0.1, 0.093], [0.5, 0.5], [0.75, 0.775], [1, 1]],
          hsl=_mix({"sat": 25}, blue={"sat": 15, "lum": -6}, green={"sat": 10}),
          grading=_wheels((200, 10), None, (40, 6))),
    _look("clear", "Clear", "Vivid",
          "Cool and refreshing: cyan-blue sky and water, fresh mint greens, clean whites.",
          curve=[[0, 0.03], [0.2, 0.21], [0.8, 0.83], [1, 1]],
          hsl=_mix(blue={"hue": -8, "sat": 30, "lum": -12}, aqua={"hue": -12, "sat": 30},
                   green={"hue": 25, "sat": -10, "lum": 8}, yellow={"hue": 20, "sat": -25, "lum": 5},
                   orange={"sat": -5, "lum": 6}),
          grading=_wheels((195, 22), (190, 8), (185, 6))),
    _look("golden", "Golden hour", "Vivid",
          "Late-afternoon warmth over everything, gold foliage, a soft glow.",
          curve=[[0, 0.02], [0.14, 0.14], [0.85, 0.86], [1, 1]],
          hsl=_mix(orange={"sat": 15}, yellow={"hue": -10, "sat": 15},
                   green={"hue": -20, "sat": -10}, blue={"sat": -20}),
          grading=_wheels((15, 8), (30, 14), (38, 28)),
          glow=22),
    _look("autumn", "Autumn", "Vivid",
          "Rich fall colour: reds and golds deepened, greens turned towards yellow.",
          curve=[[0, 0.01], [0.12, 0.115], [0.8, 0.82], [1, 1]],
          hsl=_mix(red={"hue": 5, "sat": 15}, orange={"hue": -5, "sat": 30},
                   yellow={"hue": -18, "sat": 25}, green={"hue": -25, "sat": -15},
                   blue={"sat": 10, "lum": -8}),
          grading=_wheels((20, 10), None, (35, 14))),
    # ----- Moody -----------------------------------------------------------
    _look("cinematic", "Cinematic", "Moody",
          "Teal and orange, as in films: teal shadows, warm skin, greens and blues pulled to teal.",
          curve=[[0, 0.04], [0.2, 0.19], [0.8, 0.8], [1, 0.95]],
          hsl=_mix(blue={"hue": -30}, aqua={"sat": 10}, orange={"sat": 12, "lum": 5},
                   green={"hue": 40, "sat": -40}, yellow={"hue": -18, "sat": -25}),
          grading=_wheels((188, 32), None, (35, 26))),
    _look("moody", "Moody", "Moody",
          "Dark and quiet: matte blacks, dimmed highlights, muted greens and blues, warm skin.",
          curve=[[0, 0.08], [0.2, 0.2], [0.6, 0.57], [0.85, 0.78], [1, 0.88]],
          hsl=_mix(green={"hue": -15, "sat": -55, "lum": -5}, yellow={"hue": -10, "sat": -50, "lum": -4},
                   blue={"sat": -45, "lum": -10}, aqua={"sat": -40}, orange={"sat": -12, "lum": 3},
                   red={"sat": -15}),
          grading=_wheels((200, 22), (190, 6), (38, 14)),
          vignette=-12),
    _look("heritage", "Heritage", "Moody",
          "Weighty and rich, like an old painting: deep browns, subdued colour, darkened edges.",
          curve=[[0, 0.02], [0.14, 0.135], [0.6, 0.59], [0.85, 0.81], [1, 0.92]],
          hsl=_mix({"sat": -30}, red={"sat": 30, "lum": -6}, orange={"sat": 25},
                   yellow={"hue": -10}, green={"hue": -25, "sat": -25, "lum": -4},
                   aqua={"sat": -30}, blue={"sat": -35, "lum": -10}),
          grading=_wheels((25, 28), (35, 14), (42, 16), balance=-20),
          clarity=15, vignette=-15),
    _look("forest", "Forest", "Moody",
          "Deep, cool greens turned towards teal, for woods and shade.",
          curve=[[0, 0.06], [0.2, 0.2], [0.85, 0.80], [1, 0.92]],
          hsl=_mix(green={"hue": 45, "sat": -35, "lum": -5}, yellow={"hue": 40, "sat": -40, "lum": -4},
                   aqua={"sat": -15, "lum": -8}, orange={"sat": -10}, red={"sat": -10}),
          grading=_wheels((175, 24), (170, 6), (45, 12))),
    # ----- Film ------------------------------------------------------------
    _look("portra", "Portrait film", "Film",
          "A warm colour negative, kind to skin: soft greens, faint teal shadows, its grain.",
          curve=[[0, 0.05], [0.2, 0.22], [1, 0.96]],
          hsl=_mix(orange={"sat": -10, "lum": 10}, green={"hue": -18, "sat": -30},
                   yellow={"hue": -10, "sat": -15}, blue={"hue": -8, "sat": -15}),
          grading=_wheels((185, 10), None, (35, 12)),
          film=_film("Warm portrait", 100)),
    _look("daylight", "Daylight film", "Film",
          "A cool daylight negative: pastel colour, greens leaning to cyan, a faint magenta in the whites.",
          curve=[[0, 0.06], [0.2, 0.22], [0.85, 0.85], [1, 0.95]],
          hsl=_mix({"sat": -15}, green={"hue": 30, "sat": -25, "lum": 6}, yellow={"hue": 15, "sat": -20},
                   blue={"hue": -8}, orange={"sat": -8, "lum": 6}),
          grading=_wheels((175, 18), (160, 5), (330, 10)),
          film=_film("Cool consumer", 80)),
    _look("gold", "Gold film", "Film",
          "A warm consumer film: golden light, strong reds, a nostalgic yellow cast.",
          curve=[[0, 0.03], [0.2, 0.2], [0.85, 0.87], [1, 0.98]],
          hsl=_mix(yellow={"hue": -8, "sat": 20}, orange={"sat": 15}, red={"sat": 15},
                   green={"hue": -15, "sat": -10}, blue={"hue": 5, "sat": -20}),
          grading=_wheels((35, 10), (42, 12), (48, 26)),
          film=_film("Neutral negative", 80)),
    _look("seventies", "Faded 70s", "Film",
          "An old print: faded blacks, yellowed whites, green-blue shadows, tired colour.",
          curve=[[0, 0.07], [0.18, 0.185], [0.5, 0.5], [0.85, 0.84], [1, 0.90]],
          curve_r=[[0, 0.04], [0.2, 0.2], [1, 1]],
          curve_b=[[0, 0.10], [0.25, 0.25], [1, 0.86]],
          hsl=_mix({"sat": -30}, blue={"sat": -30}),
          grading=_wheels((160, 10), None, (45, 15))),
    # ----- Black & white ---------------------------------------------------
    _look("noir", "Noir", "Black & white",
          "Hard black and white: bright whites, dark skies, gritty detail, dark corners.",
          saturation=-100,
          curve=[[0, 0], [0.08, 0.072], [0.3, 0.3], [0.6, 0.64], [0.85, 0.92], [1, 1]],
          hsl=_mix(blue={"lum": -20}, orange={"lum": 8}),
          clarity=25, texture=10, vignette=-15),
    _look("silver", "Silver", "Black & white",
          "Soft black and white with open shadows and luminous skin.",
          saturation=-100,
          curve=[[0, 0.03], [0.14, 0.15], [0.85, 0.86], [1, 0.97]],
          hsl=_mix(orange={"lum": 10}, red={"lum": 8}, blue={"lum": -8})),
    # ----- Comic -----------------------------------------------------------
    _look("anime", "Anime", "Comic",
          "Like an animated film: bright, saturated sky blues, fresh greens, smooth surfaces, a soft glow.",
          curve=[[0, 0.03], [0.14, 0.145], [0.5, 0.51], [0.85, 0.87], [1, 1]],
          hsl=_mix(blue={"hue": -18, "sat": 60, "lum": -12}, aqua={"hue": -18, "sat": 55},
                   green={"hue": -12, "sat": 40, "lum": 6}, yellow={"sat": 30},
                   orange={"sat": -5, "lum": 10}, red={"sat": 10}),
          grading=_wheels((225, 26), None, (55, 12)),
          texture=-50, clarity=18, glow=35),
    _look("comic", "Comic", "Comic",
          "Bold like a comic book: hard contrast, loud colour, flat surfaces with inked edges.",
          # No stepped curve for flat bands of tone: on R, G and B apart it
          # bends their ratios, and a sky's gradient came out as a rainbow.
          curve=[[0, 0], [0.1, 0.088], [0.3, 0.3], [0.6, 0.625], [0.85, 0.88], [1, 0.99]],
          hsl=_mix({"sat": 22}, orange={"sat": -15, "lum": 5}, red={"sat": -5}),
          texture=-60, clarity=45, sharpen=50, sharpen_masking=50),
]

assert len({lk["id"] for lk in LOOKS}) == len(LOOKS)
assert len(LOOKS) == 20
