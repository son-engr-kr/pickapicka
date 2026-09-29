"""Draw the app icon: the brand squircle with its conic gradient, and a mark on it.

    uv run --with numpy,pillow python packaging/make_icon.py
    uv run --with numpy,pillow python packaging/make_icon.py --variant p
    uv run --with numpy,pillow python packaging/make_icon.py --sheet DIR

The first writes the shipped icon: packaging/icons/pickapicka.icns (the macOS
bundle, built with iconutil), packaging/icons/pickapicka.ico (the Windows exe
and installer), packaging/icons/pickapicka-1024.png, and the page's
src/pickapicka/web/icons/favicon-32.png and apple-touch-icon.png. --variant
picks another mark for the same files. --sheet writes every candidate to DIR
instead, each as a 1024 px master and all of them side by side at the sizes
they are really shown at, and touches nothing in the repo.

The squircle and its colours are the ones the page draws (style.css, .tb-mark
and .intro-mark), so the Dock icon and the mark in the app's own top bar are the
same thing. Every mark is drawn from rectangles, circles and strokes as signed
distance fields. The name under the mark is set in Poppins SemiBold, kept with
its licence (SIL OFL 1.1, no reserved name) in src/pickapicka/web/fonts, where
the page sets the name in it too; so the script needs nothing from the system
it runs on but iconutil for the .icns.

The shipped mark, "pickword", is a viewfinder with a tick in it (culling is
picking) and the name beneath. The name is legible from about 100 px of
squircle up; below that it would be a grey smear, so the smaller images in the
.icns and the .ico are the mark alone, drawn larger. The Dock, Launchpad and
the Start menu print the name under the icon anyway.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "packaging" / "icons"
WEB_ICON_DIR = ROOT / "src" / "pickapicka" / "web" / "icons"

VARIANT = "pickword"
FONT = ROOT / "src" / "pickapicka" / "web" / "fonts" / "Poppins-SemiBold.ttf"
NAME = "Pickapicka"

# conic-gradient(from 210deg, --pick, --review, --reject, --accent, --pick),
# interpolated in sRGB as CSS does for hex colours.
FROM_DEG = 210.0
STOPS = np.array([
    (0x46, 0xA7, 0x58),   # --pick
    (0xD4, 0xA0, 0x2C),   # --review
    (0xE0, 0x52, 0x52),   # --reject
    (0x5B, 0x8D, 0xEF),   # --accent
    (0x46, 0xA7, 0x58),   # --pick again, so the seam is continuous
], dtype=np.float32) / 255.0

# Where the squircle sits on the canvas, as fractions of the canvas side.
#   mac:   Apple's Big Sur grid, an 824 px body with 100 px to spare on a 1024
#          canvas and a radius of 185, with the soft shadow that margin is for.
#   tight: the same body with almost no margin and no shadow. At 16 and 32 px a
#          100 px margin throws away a fifth of the pixels, and Windows draws
#          its taskbar and Explorer icons close to the edge, so the .ico and the
#          favicon use this.
#   bleed: a full, opaque square for apple-touch-icon; Apple rounds it itself.
LAYOUTS = {
    "mac":   {"margin": 100 / 1024, "radius": 185 / 824, "shadow": True},
    "tight": {"margin": 24 / 1024, "radius": 185 / 824, "shadow": False},
    "bleed": {"margin": 0.0, "radius": 0.0, "shadow": False},
}
BODY_SHADOW = {"offset": 10 / 1024, "sigma": 12 / 1024, "opacity": 0.32}
MARK_SHADOW = {"offset": 5 / 824, "sigma": 9 / 824, "opacity": 0.28}
MAX_RENDER = 2048          # the supersampled canvas never goes past this side


# ----- signed distance fields, in squircle units (the body's side is 1) -----

def sd_box(x, y, cx, cy, hw, hh, r=0.0):
    qx = np.abs(x - cx) - (hw - r)
    qy = np.abs(y - cy) - (hh - r)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    return outside + np.minimum(np.maximum(qx, qy), 0) - r


def sd_circle(x, y, cx, cy, r):
    return np.hypot(x - cx, y - cy) - r


def sd_segment(x, y, ax, ay, bx, by, r):
    """A stroke from a to b with round caps, r being half its width."""
    px, py, dx, dy = x - ax, y - ay, bx - ax, by - ay
    h = np.clip((px * dx + py * dy) / (dx * dx + dy * dy), 0.0, 1.0)
    return np.hypot(px - dx * h, py - dy * h) - r


def union(*ds):
    return np.minimum.reduce(ds)


def minus(a, b):
    return np.maximum(a, -b)


def coverage(d, px):
    """How much of each pixel the shape covers, px being one pixel's size."""
    return np.clip(0.5 - d / px, 0.0, 1.0)


# ----- the marks ------------------------------------------------------------

def upper_p(u, v, left, top, height, stem, bowl_h, bowl_w, stroke):
    """A geometric capital P: a stem, and a bowl made of a bar and a half circle."""
    r = bowl_h / 2
    right = left + bowl_w
    s = sd_box(u, v, left + stem / 2, top + height / 2, stem / 2, height / 2)
    outer = union(sd_box(u, v, (left + right - r) / 2, top + r, (right - r - left) / 2, r),
                  sd_circle(u, v, right - r, top + r, r))
    hole = union(sd_box(u, v, (left + stem + right - r) / 2, top + r,
                        (right - r - left - stem) / 2, r - stroke),
                 sd_circle(u, v, right - r, top + r, r - stroke))
    return union(s, minus(outer, hole))


def lower_p(u, v, left, top, bowl, stem, stroke, descender):
    """A geometric small p: a round bowl with the stem running down its left side."""
    r = bowl / 2
    ring = minus(sd_circle(u, v, left + r, top + r, r),
                 sd_circle(u, v, left + r, top + r, r - stroke))
    s = sd_box(u, v, left + stem / 2, top + (bowl + descender) / 2, stem / 2, (bowl + descender) / 2)
    return union(ring, s)


def mark_p(u, v):
    height, stem, stroke = 0.56, 0.125, 0.11
    bowl_h, bowl_w = 0.345, 0.37
    # The bowl carries the weight up and to the right, so the box sits a touch
    # right of centre and a touch high to look centred.
    left = 0.5 - bowl_w / 2 + 0.012
    top = 0.5 - height / 2 - 0.004
    return upper_p(u, v, left, top, height, stem, bowl_h, bowl_w, stroke)


def mark_pp(u, v):
    # Lighter than the single P, so the two counters stay open at 64 px.
    bowl, stem, stroke, desc, gap = 0.30, 0.082, 0.076, 0.20, 0.04
    width = 2 * bowl + gap
    left = 0.5 - width / 2
    # The bowls are the weight; centre them a little above the middle and let
    # the descenders hang.
    top = 0.5 - (bowl + desc) / 2 - 0.03
    return union(lower_p(u, v, left, top, bowl, stem, stroke, desc),
                 lower_p(u, v, left + bowl + gap, top, bowl, stem, stroke, desc))


def mark_pick(u, v):
    lo, hi, arm, w = 0.215, 0.785, 0.15, 0.030
    corners = []
    for cx, sx in ((lo, 1), (hi, -1)):
        for cy, sy in ((lo, 1), (hi, -1)):
            corners.append(sd_segment(u, v, cx, cy, cx + sx * arm, cy, w))
            corners.append(sd_segment(u, v, cx, cy, cx, cy + sy * arm, w))
    # The tick's weight is at its elbow, low and left; the box it spans is set
    # so that elbow sits just under the middle.
    cw = 0.052
    a, b, c = (0.345, 0.515), (0.455, 0.625), (0.665, 0.395)
    tick = union(sd_segment(u, v, *a, *b, cw), sd_segment(u, v, *b, *c, cw))
    return union(*corners, tick)


def scaled(mark, s, cx, cy):
    """A mark drawn at s times its size, centred on (cx, cy) of the squircle."""
    return lambda u, v: s * mark((u - cx) / s + 0.5, (v - cy) / s + 0.5)


# The mark above the name, in squircle units: the pair is centred as one block,
# a touch high, as a mark over a word looks centred.
WORD_MARK_SCALE = 0.70
WORD_MARK_CY = 0.385
WORD_BASELINE = 0.805
WORD_WIDTH = 0.64
WORD_MIN_BODY = 100        # px of squircle below which the name is left out


def word_alpha(n, left, side):
    """The name, as coverage on the n x n canvas whose squircle starts at left."""
    probe = ImageFont.truetype(str(FONT), 100)
    size = max(1, round(100 * WORD_WIDTH * side / probe.getlength(NAME)))
    font = ImageFont.truetype(str(FONT), size)
    img = Image.new("L", (n, n), 0)
    ImageDraw.Draw(img).text((left + side / 2, left + WORD_BASELINE * side), NAME,
                             fill=255, font=font, anchor="ms")
    return np.asarray(img, dtype=np.float32) / 255.0


def glyph_alpha(variant, u, v, px, n, left, side, body_px):
    """The mark's coverage; body_px is the squircle's side in the final image."""
    if variant == "pickword":
        if body_px < WORD_MIN_BODY:
            return coverage(mark_pick(u, v), px)
        mark = scaled(mark_pick, WORD_MARK_SCALE, 0.5, WORD_MARK_CY)
        return np.maximum(coverage(mark(u, v), px), word_alpha(n, left, side))
    return coverage(MARKS[variant](u, v), px)


MARKS = {"plain": None, "p": mark_p, "pp": mark_pp, "pick": mark_pick, "pickword": "pickword"}


# ----- drawing --------------------------------------------------------------

def conic(u, v):
    # Clockwise from 12 o'clock, like CSS; y grows downwards on the canvas.
    ang = np.degrees(np.arctan2(u - 0.5, -(v - 0.5)))
    t = ((ang - FROM_DEG) % 360.0) / 360.0 * (len(STOPS) - 1)
    i = np.minimum(np.floor(t).astype(np.int32), len(STOPS) - 2)
    f = (t - i)[..., None]
    return STOPS[i] * (1 - f) + STOPS[i + 1] * f


def blur(a, sigma_px):
    img = Image.fromarray(np.round(a * 255).astype(np.uint8), "L")
    return np.asarray(img.filter(ImageFilter.GaussianBlur(sigma_px)), dtype=np.float32) / 255.0


def shift_down(a, by):
    if by <= 0:
        return a
    out = np.zeros_like(a)
    out[by:] = a[:-by]
    return out


def over(dst_rgb, dst_a, src_rgb, src_a):
    """Porter-Duff source over, premultiplied colour."""
    a = src_a[..., None]
    return src_rgb * a + dst_rgb * (1 - a), src_a + dst_a * (1 - src_a)


def render(variant: str, size: int, layout: str = "mac") -> Image.Image:
    lay = LAYOUTS[layout]
    ss = max(1, min(8, MAX_RENDER // size))
    n = size * ss
    left = n * lay["margin"]
    side = n - 2 * left
    grid = (np.arange(n, dtype=np.float32) + 0.5 - left) / side
    u, v = np.meshgrid(grid, grid)
    px = 1.0 / side

    body = coverage(sd_box(u, v, 0.5, 0.5, 0.5, 0.5, lay["radius"]), px)
    rgb = np.zeros((n, n, 3), np.float32)
    alpha = np.zeros((n, n), np.float32)
    if lay["shadow"]:
        sh = shift_down(blur(body, BODY_SHADOW["sigma"] * n), round(BODY_SHADOW["offset"] * n))
        rgb, alpha = over(rgb, alpha, np.zeros_like(rgb), sh * BODY_SHADOW["opacity"])
    rgb, alpha = over(rgb, alpha, conic(u, v), body)

    if MARKS[variant] is not None:
        # Whether the name fits is a question of the final image, not of the
        # supersampled canvas it is drawn on.
        glyph = glyph_alpha(variant, u, v, px, n, left, side, size * (1 - 2 * lay["margin"]))
        sh = shift_down(blur(glyph, MARK_SHADOW["sigma"] * side), round(MARK_SHADOW["offset"] * side))
        rgb, alpha = over(rgb, alpha, np.zeros_like(rgb), sh * MARK_SHADOW["opacity"] * body)
        rgb, alpha = over(rgb, alpha, np.ones_like(rgb), glyph * body)

    straight = np.where(alpha[..., None] > 0, rgb / np.maximum(alpha[..., None], 1e-6), 0)
    out = np.dstack([straight, alpha])
    img = Image.fromarray(np.round(np.clip(out, 0, 1) * 255).astype(np.uint8), "RGBA")
    if layout == "bleed":
        img = img.convert("RGB")
    return img if ss == 1 else img.resize((size, size), Image.LANCZOS)


# ----- outputs --------------------------------------------------------------

ICONSET = [(16, 1), (16, 2), (32, 1), (32, 2), (128, 1), (128, 2),
           (256, 1), (256, 2), (512, 1), (512, 2)]
ICO_SIZES = [16, 24, 32, 48, 64, 128, 256]


def write_icns(variant: str, out: Path) -> None:
    assert shutil.which("iconutil"), "iconutil ships with macOS; build the .icns there"
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "pickapicka.iconset"
        iconset.mkdir()
        for pt, scale in ICONSET:
            name = f"icon_{pt}x{pt}{'@2x' if scale == 2 else ''}.png"
            render(variant, pt * scale, "mac").save(iconset / name)
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)], check=True)


def write_ico(variant: str, out: Path) -> None:
    # Each size drawn at its own resolution rather than scaled from 256 by
    # Pillow, so the 16 and 32 are as sharp as they can be.
    frames = [render(variant, s, "tight") for s in ICO_SIZES]
    frames[-1].save(out, format="ICO", sizes=[(s, s) for s in ICO_SIZES],
                    append_images=frames[:-1])


def write_shipped(variant: str) -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    WEB_ICON_DIR.mkdir(parents=True, exist_ok=True)
    write_icns(variant, ICON_DIR / "pickapicka.icns")
    write_ico(variant, ICON_DIR / "pickapicka.ico")
    render(variant, 1024, "mac").save(ICON_DIR / "pickapicka-1024.png")
    render(variant, 32, "tight").save(WEB_ICON_DIR / "favicon-32.png")
    render(variant, 180, "bleed").save(WEB_ICON_DIR / "apple-touch-icon.png")
    for p in (ICON_DIR / "pickapicka.icns", ICON_DIR / "pickapicka.ico",
              ICON_DIR / "pickapicka-1024.png", WEB_ICON_DIR / "favicon-32.png",
              WEB_ICON_DIR / "apple-touch-icon.png"):
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size:,} bytes")


SHEET_SIZES = [256, 128, 64, 32, 16]
SHEET_BGS = [(0x1E, 0x1E, 0x1E), (0xF2, 0xF2, 0xF2)]


def write_sheet(out_dir: Path) -> None:
    """Every candidate at the sizes it is really shown at, on dark and on light."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pad, label_w = 24, 190
    block_w = sum(SHEET_SIZES) + pad * (len(SHEET_SIZES) + 1)
    row_h = SHEET_SIZES[0] + 2 * pad
    sheet = Image.new("RGB", (label_w + block_w * len(SHEET_BGS), row_h * len(MARKS)), (0x80, 0x80, 0x80))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(FONT), 22)
    for r, variant in enumerate(MARKS):
        render(variant, 1024, "mac").save(out_dir / f"icon-{variant}.png")
        icons = {s: render(variant, s, "mac") for s in SHEET_SIZES}
        y0 = r * row_h
        draw.text((pad, y0 + row_h / 2), f"{chr(65 + r)}  {variant}", fill=(255, 255, 255),
                  font=font, anchor="lm")
        for b, bg in enumerate(SHEET_BGS):
            x = label_w + b * block_w
            draw.rectangle([x, y0, x + block_w - 1, y0 + row_h - 1], fill=bg)
            x += pad
            for s in SHEET_SIZES:
                sheet.paste(icons[s], (x, y0 + (row_h - s) // 2), icons[s])
                x += s + pad
    sheet.save(out_dir / "icon-candidates.png")
    print(out_dir / "icon-candidates.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", choices=list(MARKS), default=VARIANT)
    ap.add_argument("--sheet", type=Path, help="write every candidate here instead")
    args = ap.parse_args()
    if args.sheet:
        write_sheet(args.sheet)
    else:
        write_shipped(args.variant)


if __name__ == "__main__":
    main()
