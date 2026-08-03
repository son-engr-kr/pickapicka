"""Signature / shooting-info watermarks drawn onto a rendered photo.

A watermark is part of the edit dict, so it rides along with presets, bulk
apply, the live preview and the export without any separate plumbing.

Text comes from templates with tokens — `{name} · {camera} · {focal}` — filled
from the photo's EXIF (see `exifinfo`). Empty tokens collapse together with
their separator, so a lens-less shot doesn't print "α7C II ·  · ISO 400".

Everything is sized as a fraction of the frame's long edge, so the same
watermark looks identical on the editor preview and on a 60 MP export. Drawing
is ROI-aware: pass the window being rendered and the layout is still computed
for the whole frame, then clipped — that is what lets the 1:1 zoom show the real
watermark instead of a differently-scaled one.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STYLES: tuple[str, ...] = ("minimal", "bar", "plate", "corner", "filmstrip")
POSITIONS: tuple[str, ...] = (
    "bottom-left", "bottom-center", "bottom-right",
    "top-left", "top-center", "top-right",
)

DEFAULT_WATERMARK: dict[str, Any] = {
    "enabled": False,
    "style": "minimal",
    "position": "bottom-right",
    "name": "",
    "camera": "",          # override; empty means "use the photo's EXIF"
    "line1": "{name}",
    "line2": "{camera} · {lens}",
    "line3": "{focal} · {aperture} · {shutter} · {iso}",
    "size": 100,           # 100 = the default scale below
    "opacity": 90,
    "color": "#ffffff",
    "margin": 100,         # 100 = the default inset below
}

TOKENS: tuple[str, ...] = (
    "name", "camera", "lens", "focal", "aperture", "shutter", "iso",
    "date", "time", "file",
)

# All relative to the frame's long edge.
_BASE_TEXT = 0.018     # cap height of the secondary lines at size=100
_TITLE_SCALE = 1.45    # line 1 is the signature, so it leads
_BASE_MARGIN = 0.035
_LINE_GAP = 0.42       # of a line height

_TOKEN_RE = re.compile(r"\{(\w+)\}")
_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

# Preferred system faces, then Pillow's bundled scalable font. Nothing is
# downloaded and nothing extra is shipped; the fallback always exists.
_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "regular": (
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ),
    "bold": (
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
}
_BOLD_INDEX = {"bold": 1}  # .ttc collections: index 1 is usually the bold face


def _font(weight: str, px: int) -> ImageFont.FreeTypeFont:
    px = max(6, int(px))
    key = (weight, px)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    font: ImageFont.FreeTypeFont | None = None
    for path in _FONT_CANDIDATES[weight]:
        if not Path(path).is_file():
            continue
        try:
            idx = _BOLD_INDEX.get(weight, 0) if path.endswith(".ttc") else 0
            font = ImageFont.truetype(path, px, index=idx)
            break
        except (OSError, ValueError):
            continue
    if font is None:
        font = ImageFont.load_default(size=px)
    _FONT_CACHE[key] = font
    return font


# ----- normalization ------------------------------------------------------

def _clamp_int(raw: Any, lo: int, hi: int, fallback: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(raw)))))
    except (TypeError, ValueError):
        return fallback


def _clean_color(raw: Any) -> str:
    s = str(raw or "").strip()
    return s if re.fullmatch(r"#[0-9a-fA-F]{6}", s) else "#ffffff"


def normalize(raw: Any) -> dict[str, Any]:
    """Coerce a watermark dict into the canonical shape (always all keys)."""
    out = dict(DEFAULT_WATERMARK)
    if not isinstance(raw, dict):
        return out
    out["enabled"] = bool(raw.get("enabled"))
    style = str(raw.get("style", "") or "").lower()
    out["style"] = style if style in STYLES else DEFAULT_WATERMARK["style"]
    pos = str(raw.get("position", "") or "").lower()
    out["position"] = pos if pos in POSITIONS else DEFAULT_WATERMARK["position"]
    for key in ("name", "camera", "line1", "line2", "line3"):
        val = raw.get(key)
        out[key] = str(val)[:160] if isinstance(val, str) else DEFAULT_WATERMARK[key]
    out["size"] = _clamp_int(raw.get("size"), 30, 300, 100)
    out["opacity"] = _clamp_int(raw.get("opacity"), 0, 100, 90)
    out["margin"] = _clamp_int(raw.get("margin"), 0, 300, 100)
    out["color"] = _clean_color(raw.get("color"))
    return out


def is_neutral(wm: dict[str, Any] | None) -> bool:
    """True when the watermark would not put anything on the photo."""
    w = normalize(wm)
    if not w["enabled"] or w["opacity"] <= 0:
        return True
    return not any(w[k].strip() for k in ("line1", "line2", "line3"))


# ----- text ---------------------------------------------------------------

def _values(wm: dict[str, Any], meta: dict[str, Any] | None) -> dict[str, str]:
    m = meta or {}
    captured = str(m.get("captured_at") or "")
    date, _, time = captured.partition(" ")
    return {
        "name": wm["name"].strip(),
        # An explicit camera name wins: EXIF codenames are ugly and the user may
        # want to credit a body the file doesn't know about.
        "camera": (wm["camera"].strip() or str(m.get("camera") or "")),
        "lens": str(m.get("lens") or ""),
        "focal": str(m.get("focal") or ""),
        "aperture": str(m.get("aperture") or ""),
        "shutter": str(m.get("shutter") or ""),
        "iso": str(m.get("iso_text") or ""),
        "date": date.replace(":", "-"),
        "time": time,
        "file": str(m.get("file") or ""),
    }


def render_line(template: str, values: dict[str, str]) -> str:
    """Fill a template, dropping separators left dangling by an empty token.

    The rules, in the order they matter:
      - `{camera} · {lens}` with no lens gives "α7C II", not "α7C II · ".
      - a line whose tokens are all empty disappears entirely, so an EXIF-less
        photo doesn't get stamped with a lone "·".
      - literal text is kept when it is doing work: "© {name}" keeps the ©, but
        a trailing separator with nothing after it is dropped. "Has letters or
        digits" is the test — punctuation alone is a separator.
    """
    if not template:
        return ""
    parts = _TOKEN_RE.split(template)   # [text, token, text, token, ...]
    has_token = len(parts) > 1
    if has_token and not any(values.get(parts[i], "") for i in range(1, len(parts), 2)):
        return ""

    out: list[str] = []
    pending = ""
    for i, part in enumerate(parts):
        if i % 2 == 0:
            pending += part
            continue
        text = values.get(part, "")
        if not text:
            continue            # the token's leading separator goes with it
        out.append(pending)
        out.append(text)
        pending = ""
    if any(ch.isalnum() for ch in pending):
        out.append(pending)     # a real trailing word, not a dangling separator
    return " ".join("".join(out).split())


def lines_for(wm: dict[str, Any], meta: dict[str, Any] | None) -> list[str]:
    values = _values(wm, meta)
    return [s for s in (render_line(wm[k], values) for k in ("line1", "line2", "line3")) if s]


# ----- drawing ------------------------------------------------------------

def _anchor(position: str) -> tuple[str, str]:
    vertical, _, horizontal = position.partition("-")
    return vertical, horizontal


def _hex_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _draw_block(
    draw: ImageDraw.ImageDraw, lines: list[str], fonts: list[ImageFont.FreeTypeFont],
    x: float, y: float, align: str, rgb: tuple[int, int, int], alpha: int,
    shadow: bool,
) -> None:
    """Lines stacked from `y` downward, aligned around `x`."""
    for text, font in zip(lines, fonts):
        w = draw.textlength(text, font=font)
        tx = x - w if align == "right" else (x - w / 2 if align == "center" else x)
        if shadow:
            # A soft dark offset keeps white text legible over a white car.
            draw.text((tx + max(1, font.size * 0.05), y + max(1, font.size * 0.05)),
                      text, font=font, fill=(0, 0, 0, int(alpha * 0.55)))
        draw.text((tx, y), text, font=font, fill=(*rgb, alpha))
        y += font.size * (1.0 + _LINE_GAP)


def _measure(lines: list[str], fonts: list[ImageFont.FreeTypeFont],
             draw: ImageDraw.ImageDraw) -> tuple[float, float]:
    width = max((draw.textlength(t, font=f) for t, f in zip(lines, fonts)), default=0.0)
    height = sum(f.size * (1.0 + _LINE_GAP) for f in fonts) - (
        fonts[-1].size * _LINE_GAP if fonts else 0.0)
    return width, height


def render(
    rgb_img: np.ndarray,
    wm: dict[str, Any] | None,
    meta: dict[str, Any] | None = None,
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
) -> np.ndarray:
    """Draw the watermark onto an RGB uint8 array and return a new array.

    `roi` says which window of the frame `rgb_img` is, so the layout is computed
    against the whole photo and only the visible part lands on the crop.
    """
    w = normalize(wm)
    if is_neutral(w):
        return rgb_img
    lines = lines_for(w, meta)
    if not lines:
        return rgb_img

    h_px, w_px = rgb_img.shape[:2]
    frame_w, frame_h = w_px / roi[2], h_px / roi[3]
    frame_long = max(frame_w, frame_h)
    off_x, off_y = roi[0] * frame_w, roi[1] * frame_h

    scale = w["size"] / 100.0
    body = max(7.0, _BASE_TEXT * frame_long * scale)
    fonts = [_font("bold", int(body * _TITLE_SCALE))] + [
        _font("regular", int(body)) for _ in lines[1:]]
    if w["style"] == "filmstrip":
        fonts = [_font("regular", int(body * (1.15 if i == 0 else 1.0)))
                 for i in range(len(lines))]

    overlay = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # Coordinates are computed in frame space, then shifted onto this window.
    text_w, text_h = _measure(lines, fonts, draw)
    margin = _BASE_MARGIN * frame_long * (w["margin"] / 100.0)
    vertical, horizontal = _anchor(w["position"])
    rgb = _hex_rgb(w["color"])
    alpha = int(round(255 * w["opacity"] / 100.0))

    x = {"left": margin, "center": frame_w / 2, "right": frame_w - margin}[horizontal]
    y = margin if vertical == "top" else frame_h - margin - text_h
    align = {"left": "left", "center": "center", "right": "right"}[horizontal]

    if w["style"] == "bar":
        # A gradient scrim across the full width — always legible, and it reads
        # as a deliberate lower third rather than floating text. Built directly
        # in window space so a zoomed crop just sees its slice of the ramp.
        band_h = text_h + margin * 1.6
        top = 0.0 if vertical == "top" else frame_h - band_h
        rows = off_y + np.arange(h_px, dtype=np.float32)          # frame-space y
        t = np.clip((rows - top) / max(band_h, 1e-6), 0.0, 1.0)
        if vertical == "top":
            t = 1.0 - t
        t[(rows < top) | (rows > top + band_h)] = 0.0
        col = np.zeros((h_px, w_px, 4), dtype=np.uint8)
        col[..., 3] = np.repeat((t * 165 * (w["opacity"] / 100.0)).astype(np.uint8)[:, None],
                                w_px, axis=1)
        overlay.alpha_composite(Image.fromarray(col, "RGBA"))
        y = top + (band_h - text_h) / 2
    elif w["style"] == "corner":
        # A short accent rule above the signature — the thing that makes it look
        # placed rather than typed on.
        rule_w = max(body * 2.2, text_w * 0.18)
        ry = y - body * 0.75
        rx0 = x - (rule_w if align == "right" else (rule_w / 2 if align == "center" else 0))
        draw.line([(rx0 - off_x, ry - off_y), (rx0 + rule_w - off_x, ry - off_y)],
                  fill=(*rgb, alpha), width=max(2, int(body * 0.16)))
    elif w["style"] == "plate":
        pad = body * 0.7
        px0 = x - (text_w if align == "right" else (text_w / 2 if align == "center" else 0))
        box = (px0 - pad - off_x, y - pad * 0.8 - off_y,
               px0 + text_w + pad - off_x, y + text_h + pad * 0.8 - off_y)
        plate = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
        ImageDraw.Draw(plate).rounded_rectangle(
            box, radius=body * 0.55, fill=(0, 0, 0, int(120 * w["opacity"] / 100.0)))
        overlay.alpha_composite(plate)
    elif w["style"] == "filmstrip":
        # Centred, spaced-out caption between two hairlines, like a print rebate.
        lines = [" ".join(t.upper()) if i == 0 else t for i, t in enumerate(lines)]
        text_w, text_h = _measure(lines, fonts, draw)
        x, align = frame_w / 2, "center"
        y = margin if vertical == "top" else frame_h - margin - text_h
        rule_y = y - body * 1.0
        rule_half = max(text_w / 2 + body * 1.6, frame_w * 0.12)
        draw.line([(frame_w / 2 - rule_half - off_x, rule_y - off_y),
                   (frame_w / 2 + rule_half - off_x, rule_y - off_y)],
                  fill=(*rgb, int(alpha * 0.75)), width=max(2, int(body * 0.12)))

    shadow = w["style"] in ("minimal", "corner", "filmstrip")
    _draw_block(draw, lines, fonts, x - off_x, y - off_y, align, rgb, alpha, shadow)

    base = Image.fromarray(rgb_img).convert("RGBA")
    base.alpha_composite(overlay)
    return np.asarray(base.convert("RGB"))
