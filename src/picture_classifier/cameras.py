"""EXIF camera model strings → the name the manufacturer actually prints.

EXIF stores internal codenames: a Sony a7C II reports `ILCE-7CM2`, a Nikon Z6 II
reports `NIKON Z 6_2`. Stamping those on a photo looks like a database dump, so
this maps them to the marketing name with the real typography — the Greek alpha
in Sony's α7C II, Roman numerals, the spacing in Nikon's `Z 9`.

Unknown bodies fall through `pretty_model`, which just strips a redundant make
prefix and tidies whitespace, so a camera missing from the table still reads
sensibly rather than being dropped.
"""
from __future__ import annotations

# EXIF Model (upper-cased, trimmed) -> display name.
#
# Sony's whole line uses the α glyph (U+03B1). Canon and Fujifilm already report
# print-ready strings, so they mostly appear here to strip the make prefix or fix
# spacing. Add to this table freely: an entry is one line and costs nothing.
_MODELS: dict[str, str] = {
    # ----- Sony (ILCE = Interchangeable Lens Camera E-mount) -----
    "ILCE-7CM2": "α7C II",
    "ILCE-7CR": "α7CR",
    "ILCE-7C": "α7C",
    "ILCE-7M4": "α7 IV",
    "ILCE-7M3": "α7 III",
    "ILCE-7M2": "α7 II",
    "ILCE-7": "α7",
    "ILCE-7RM5": "α7R V",
    "ILCE-7RM4": "α7R IV",
    "ILCE-7RM4A": "α7R IVA",
    "ILCE-7RM3": "α7R III",
    "ILCE-7RM3A": "α7R IIIA",
    "ILCE-7RM2": "α7R II",
    "ILCE-7SM3": "α7S III",
    "ILCE-7SM2": "α7S II",
    "ILCE-7S": "α7S",
    "ILCE-9M3": "α9 III",
    "ILCE-9M2": "α9 II",
    "ILCE-9": "α9",
    "ILCE-1M2": "α1 II",
    "ILCE-1": "α1",
    "ILCE-6700": "α6700",
    "ILCE-6600": "α6600",
    "ILCE-6400": "α6400",
    "ILCE-6100": "α6100",
    "ILCE-6000": "α6000",
    "ZV-E1": "ZV-E1",
    "ZV-E10M2": "ZV-E10 II",
    "ZV-E10": "ZV-E10",
    "ILME-FX3": "FX3",
    "ILME-FX30": "FX30",
    "DSC-RX100M7": "RX100 VII",
    "DSC-RX1RM3": "RX1R III",

    # ----- Canon -----
    "CANON EOS R5 MARK II": "EOS R5 Mark II",
    "CANON EOS R5": "EOS R5",
    "CANON EOS R6 MARK II": "EOS R6 Mark II",
    "CANON EOS R6": "EOS R6",
    "CANON EOS R8": "EOS R8",
    "CANON EOS R7": "EOS R7",
    "CANON EOS R3": "EOS R3",
    "CANON EOS R1": "EOS R1",
    "CANON EOS R10": "EOS R10",
    "CANON EOS R50": "EOS R50",
    "CANON EOS RP": "EOS RP",
    "CANON EOS R": "EOS R",
    "CANON EOS 5D MARK IV": "EOS 5D Mark IV",
    "CANON EOS 6D MARK II": "EOS 6D Mark II",
    "CANON EOS 90D": "EOS 90D",

    # ----- Nikon -----
    "NIKON Z 9": "Z 9",
    "NIKON Z 8": "Z 8",
    "NIKON Z 7_2": "Z 7II",
    "NIKON Z 7": "Z 7",
    "NIKON Z 6_3": "Z 6III",
    "NIKON Z 6_2": "Z 6II",
    "NIKON Z 6": "Z 6",
    "NIKON Z 5": "Z 5",
    "NIKON Z 50": "Z 50",
    "NIKON Z FC": "Z fc",
    "NIKON Z F": "Zf",
    "NIKON D850": "D850",
    "NIKON D780": "D780",

    # ----- Fujifilm -----
    "X-T5": "X-T5",
    "X-T4": "X-T4",
    "X-T3": "X-T3",
    "X-H2S": "X-H2S",
    "X-H2": "X-H2",
    "X-S20": "X-S20",
    "X-PRO3": "X-Pro3",
    "X-E4": "X-E4",
    "X100VI": "X100VI",
    "X100V": "X100V",
    "GFX100S II": "GFX100S II",
    "GFX100 II": "GFX100 II",
    "GFX100S": "GFX100S",

    # ----- Panasonic LUMIX (EXIF drops the LUMIX brand) -----
    "DC-S5M2": "LUMIX S5II",
    "DC-S5M2X": "LUMIX S5IIX",
    "DC-S1R": "LUMIX S1R",
    "DC-S1H": "LUMIX S1H",
    "DC-GH7": "LUMIX GH7",
    "DC-GH6": "LUMIX GH6",
    "DC-GH5M2": "LUMIX GH5 II",
    "DC-G9M2": "LUMIX G9 II",

    # ----- OM System / Olympus -----
    "OM-1MARKII": "OM-1 Mark II",
    "OM-1": "OM-1",
    "OM-5": "OM-5",
    "E-M1MARKIII": "OM-D E-M1 Mark III",
    "E-M5MARKIII": "OM-D E-M5 Mark III",

    # ----- Leica / Ricoh / Hasselblad -----
    "LEICA M11": "Leica M11",
    "LEICA Q3": "Leica Q3",
    "LEICA SL3": "Leica SL3",
    "RICOH GR III": "GR III",
    "RICOH GR IIIX": "GR IIIx",
    "GR III": "GR III",
    "GR IIIX": "GR IIIx",
    "X2D 100C": "Hasselblad X2D 100C",

    # ----- Phones (they turn up in mixed folders) -----
    "IPHONE 16 PRO": "iPhone 16 Pro",
    "IPHONE 15 PRO MAX": "iPhone 15 Pro Max",
    "IPHONE 15 PRO": "iPhone 15 Pro",
    "IPHONE 14 PRO": "iPhone 14 Pro",
}

# Makes worth keeping in front of the model when the model alone is ambiguous.
_MAKE_ALIASES = {
    "SONY": "Sony",
    "CANON": "Canon",
    "NIKON": "Nikon",
    "NIKON CORPORATION": "Nikon",
    "FUJIFILM": "Fujifilm",
    "PANASONIC": "Panasonic",
    "OM DIGITAL SOLUTIONS": "OM System",
    "OLYMPUS": "Olympus",
    "OLYMPUS CORPORATION": "Olympus",
    "LEICA CAMERA AG": "Leica",
    "RICOH IMAGING COMPANY, LTD.": "Ricoh",
    "APPLE": "Apple",
    "HASSELBLAD": "Hasselblad",
    "DJI": "DJI",
    "GOPRO": "GoPro",
}


def pretty_model(model: str | None, make: str | None = None) -> str:
    """Display name for an EXIF model string. Falls back to the cleaned-up model
    when the body isn't in the table, and never returns the make twice."""
    raw = (model or "").strip()
    if not raw:
        return ""
    hit = _MODELS.get(raw.upper())
    if hit:
        return hit
    brand = _MAKE_ALIASES.get((make or "").strip().upper(), (make or "").strip())
    # "Canon Canon EOS 80D" is a real EXIF pairing; don't repeat the make.
    if brand and raw.upper().startswith(brand.upper()):
        return " ".join(raw.split())
    return " ".join(f"{brand} {raw}".split()) if brand else " ".join(raw.split())


def pretty_make(make: str | None) -> str:
    m = (make or "").strip()
    return _MAKE_ALIASES.get(m.upper(), m)


def preset_list() -> list[dict[str, str]]:
    """Every known body as {exif, name}, for the watermark's camera picker.
    Sorted by display name so the picker reads like a catalogue."""
    seen: dict[str, str] = {}
    for exif, name in _MODELS.items():
        seen.setdefault(name, exif)
    return [{"name": name, "exif": exif}
            for name, exif in sorted(seen.items(), key=lambda kv: kv[0].lower())]
