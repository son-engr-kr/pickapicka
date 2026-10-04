"""Colour LUTs: an imported look, and one fitted from a reference frame.

Two things live here, and they share one representation on purpose.

  1. `.cube` import — Adobe/Iridas 1D and 3D lookup tables, the format every
     creative profile ships in (Lightroom's, Evoto's colour looks, a colourist's
     show LUT). Parsed strictly: a file that is not a LUT is an error, not
     something to guess at.
  2. Colour match — given a frame from one camera body and a frame from another,
     fit a transform that moves the first body's colour towards the second's.
     The fit is baked into a table, so the render path is the same table lookup
     as an imported look and the reference frame is never needed again.

That shared representation is the whole design. A look is `size**dim * 3`
numbers plus a domain, and *everything* — an imported film print emulation, a
fitted body match — reduces to it. One apply path, one cache, one thing to make
fast, and a look re-applies deterministically from data alone.

Applying it fast is the interesting part. A 33^3 table over a 24 MP frame is
24 million trilinear interpolations; eight fancy-index gathers of a (P, 3) float
array in numpy moves gigabytes of temporaries and takes seconds. But trilinear
is bilinear in two axes followed by a lerp in the third, and bilinear sampling
of a small table is exactly what `cv2.remap` does in one SIMD, multi-threaded
pass. Laying the table out as an (N*N, N) image — row `b*N + g`, column `r`,
which is precisely the order a .cube stores its entries in — turns the whole
operation into two remaps and one lerp, and a 1D table into three. See
`_apply_3d`: 286 ms for 24 MP against 7.0 s for the eight-gather version, and
233 ms for a 1D table against 640 ms.

What it costs is precision: OpenCV carries the interpolation weight in 1/32ths
of a cell, so the error scales with cell width. Every table is therefore refined
once, up front, to at least 32 cells per axis (`_grid_2d`), which is exact —
piecewise-trilinear resampled on a refinement of its own grid is the same
function — and puts a 2-node colour matrix and a 64-node show LUT alike within
0.17 of a code value out of 255. Same trade `editing._wb_tone_lut` and
`film._apply_tone` already make, at a tenth of the size.

The table is stored as base64 uint16 rather than a JSON list of floats: a 33^3
LUT is 107811 numbers, and 1/65535 is two orders of magnitude finer than the
8-bit output. `table_key` gives a caller a content hash, so a look shared by a
whole shoot can be stored once and referenced instead of copied into every edit.
"""
from __future__ import annotations

import base64
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from .sliders import tenth

# ----- schema -------------------------------------------------------------

# A look is a table plus how much of it to use. Neutral is "no table".
DEFAULT_LUT: dict[str, Any] = {
    "enabled": False,
    "name": "",              # the .cube TITLE, a file name, or "Colour match"
    "amount": 100,           # 0..100, blends the result against the input
    "dim": 3,                # 1 (per-channel curves) or 3 (a colour cube)
    "size": 0,               # entries per axis; 0 is "no table", i.e. neutral
    "domain_min": [0.0, 0.0, 0.0],
    "domain_max": [1.0, 1.0, 1.0],
    "table": "",             # base64 uint16, size**dim * 3 values; see pack_table
}

MAX_3D_SIZE = 64        # 64^3 is 786k entries; past that a .cube is a data file
MAX_1D_SIZE = 65536     # the Iridas ceiling for a 1D table
NAME_MAX = 60

_EPS = 1e-6
# Baked-fit grid. 17 nodes hold the fitted model to about a fifth of a code
# value on average and a few code values at worst (test_a_fit_bakes_faithfully),
# and keep the stored look at ~38 KB of base64 rather than ~280 KB for 33 nodes.
FIT_SIZE = 17
_STAT_EDGE = 512        # frames are reduced to this before any statistic
_HIST_BINS = 1024       # bins per channel for the marginal match
_CDF_EDGE = 1e-4        # CDF mass below which a bin counts as unpopulated
_SMOOTH_PASSES = 2      # box passes over the marginal map; see _marginal_maps
_TILE_PIXELS = 4_000_000
_REMAP_CELLS = 32       # cells per axis a cube is refined to; see _grid_2d


# ----- table packing ------------------------------------------------------

def pack_table(table: np.ndarray) -> str:
    """(n, 3) floats in [0,1] -> base64 uint16. Values are clamped, which is the
    one place this module discards information: the pipeline is [0,1], so a LUT
    that maps past white cannot be represented and is clipped here rather than
    half-way down the render where it would be a mystery."""
    arr = np.asarray(table, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 3, f"table must be (n, 3), got {arr.shape}"
    assert np.isfinite(arr).all(), "table holds a non-finite value"
    q = np.rint(np.clip(arr, 0.0, 1.0) * 65535.0).astype("<u2")
    return base64.b64encode(q.tobytes()).decode("ascii")


@lru_cache(maxsize=4)
def _unpack_cached(blob: str, count: int) -> np.ndarray:
    raw = base64.b64decode(blob, validate=True)
    assert len(raw) == count * 3 * 2, (
        f"table holds {len(raw) // 2} values, the declared size needs {count * 3}")
    arr = np.frombuffer(raw, dtype="<u2").astype(np.float32) / 65535.0
    out = arr.reshape(count, 3)
    out.flags.writeable = False
    return out


def unpack_table(blob: str, count: int) -> np.ndarray:
    """base64 uint16 -> a read-only (count, 3) float32 array in [0,1].

    Cached, because `is_neutral` and every tile of every render ask for the same
    table and base64-decoding 200 KB per tile would show up in a profile.
    """
    assert isinstance(blob, str) and blob, "no table to unpack"
    return _unpack_cached(blob, int(count))


def entry_count(dim: int, size: int) -> int:
    """How many (r,g,b) entries a table of this shape holds."""
    assert dim in (1, 3), f"dim must be 1 or 3, got {dim!r}"
    return size if dim == 1 else size ** 3


def table_key(params: dict[str, Any] | None) -> str:
    """A short content hash of the look, or '' when there is none.

    Offered so a caller can store one shared look once — keyed by this — and put
    the key on each photo's edit instead of 39 KB of base64 per photo. The shape
    and the domain are in it as well as the table: the same numbers read over a
    different input range are a different look, and two of those must not share
    a key when the key is what an edit stores. The name and amount are not, since
    renaming or dialling back a look does not change what its table is.
    """
    if not params or not params.get("table"):
        return ""
    p = {**DEFAULT_LUT, **params}
    head = f"{p['dim']}:{p['size']}:{p['domain_min']}:{p['domain_max']}:"
    return hashlib.sha1((head + params["table"]).encode("ascii")).hexdigest()[:16]


# ----- .cube parsing ------------------------------------------------------

def _as_float(token: str, lineno: int, what: str) -> float:
    # No try/except: a token that is not a number is a malformed file, and the
    # ValueError names the token better than a rewritten message would.
    val = float(token)
    assert np.isfinite(val), f"line {lineno}: {what} is not a finite number ({token!r})"
    return val


def parse_cube(text: str) -> dict[str, Any]:
    """Parse an Adobe/Iridas `.cube` file.

    Understands `TITLE`, `LUT_1D_SIZE`, `LUT_3D_SIZE`, `DOMAIN_MIN`,
    `DOMAIN_MAX`, `#` comments and arbitrary whitespace. Returns

        {"dim": 1|3, "size": n, "title": str,
         "domain_min": [r,g,b], "domain_max": [r,g,b],
         "table": float32 (entries, 3)}

    with the entries in file order: for a 3D table red varies fastest, so entry
    `r + n*g + n*n*b` is the node at (r, g, b). Nothing is repaired — a wrong
    entry count, a missing size, a bad domain or a video-range flag all raise
    with the line at fault, because a LUT that is quietly patched up renders a
    photograph wrongly and nothing says so.
    """
    assert isinstance(text, str), f"parse_cube wants the file's text, got {type(text).__name__}"
    title = ""
    dim = 0
    size = 0
    dmin = [0.0, 0.0, 0.0]
    dmax = [1.0, 1.0, 1.0]
    rows: list[tuple[float, float, float]] = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        key = parts[0].upper()

        if key == "TITLE":
            title = line[len(parts[0]):].strip().strip('"')[:NAME_MAX]
            continue
        if key in ("LUT_1D_SIZE", "LUT_3D_SIZE"):
            assert dim == 0, f"line {lineno}: {key} after a LUT size was already declared"
            assert len(parts) == 2, f"line {lineno}: {key} takes one integer, got {line!r}"
            size = int(parts[1])
            dim = 1 if key == "LUT_1D_SIZE" else 3
            cap = MAX_1D_SIZE if dim == 1 else MAX_3D_SIZE
            assert 2 <= size <= cap, f"line {lineno}: {key} must be 2..{cap}, got {size}"
            continue
        if key in ("DOMAIN_MIN", "DOMAIN_MAX"):
            assert len(parts) == 4, f"line {lineno}: {key} takes three numbers, got {line!r}"
            vals = [_as_float(p, lineno, key) for p in parts[1:]]
            if key == "DOMAIN_MIN":
                dmin = vals
            else:
                dmax = vals
            continue
        if key in ("LUT_IN_VIDEO_RANGE", "LUT_OUT_VIDEO_RANGE"):
            # Legal in the spec, not implemented here. Applying it as if it were
            # full-range would crush or clip the ends of the frame silently.
            assert False, f"line {lineno}: video-range LUTs are not supported ({key})"

        assert len(parts) == 3, (
            f"line {lineno}: expected three numbers or a known keyword, got {line!r}")
        rows.append(tuple(_as_float(p, lineno, "table entry") for p in parts))

    assert dim != 0, "no LUT_1D_SIZE or LUT_3D_SIZE — this is not a .cube LUT"
    want = entry_count(dim, size)
    assert len(rows) == want, (
        f"LUT_{dim}D_SIZE {size} needs {want} entries, the file has {len(rows)}")
    for c in range(3):
        assert dmax[c] - dmin[c] > _EPS, (
            f"DOMAIN_MAX must exceed DOMAIN_MIN on every channel; "
            f"channel {c} is [{dmin[c]}, {dmax[c]}]")

    return {
        "dim": dim,
        "size": size,
        "title": title,
        "domain_min": dmin,
        "domain_max": dmax,
        "table": np.asarray(rows, dtype=np.float32),
    }


def load_cube(path: str | Path) -> dict[str, Any]:
    """Read and parse a `.cube` file. Latin-1 because the format is ASCII with
    the occasional stray byte in a TITLE, and a decode error there should not
    stop a valid table from loading."""
    text = Path(path).read_text(encoding="latin-1")
    cube = parse_cube(text)
    if not cube["title"]:
        cube["title"] = Path(path).stem[:NAME_MAX]
    return cube


def params_from_cube(cube: dict[str, Any], amount: int = 100,
                     name: str | None = None) -> dict[str, Any]:
    """A parsed cube as the parameter dict an edit stores."""
    return _params(name if name is not None else cube["title"], amount,
                   cube["dim"], cube["size"], cube["domain_min"],
                   cube["domain_max"], cube["table"])


# ----- schema handling ----------------------------------------------------

def _fnum(raw: Any, lo: float, hi: float, fallback: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return fallback


def identity_table(dim: int, size: int) -> np.ndarray:
    """The table that changes nothing, as (entries, 3) float32.

    Also the node grid a fit is evaluated on, since the identity table *is* the
    list of node coordinates. Red varies fastest, matching the .cube order, so
    entry `r + n*g + n*n*b` is the node at (r, g, b).
    """
    assert dim in (1, 3), f"dim must be 1 or 3, got {dim!r}"
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
    if dim == 1:
        return np.repeat(ramp[:, None], 3, axis=1)
    return np.stack([np.tile(ramp, size * size),
                     np.tile(np.repeat(ramp, size), size),
                     np.repeat(ramp, size * size)], axis=1)


def _params(name: str, amount: int, dim: int, size: int,
            domain_min: list[float], domain_max: list[float],
            table: np.ndarray) -> dict[str, Any]:
    """Assemble a full parameter dict and check it holds together.

    Deliberately not routed through `normalize`: `normalize` answers "is this
    worth storing" and returns None for a look that does nothing, which is the
    wrong answer for a constructor. An identity .cube imports as an identity
    LUT; it is `normalize` at the storage boundary that decides to drop it.
    """
    want = entry_count(dim, size)
    assert table.shape == (want, 3), (
        f"a {dim}D LUT of size {size} needs {want} entries, got {table.shape[0]}")
    return {
        "enabled": True,
        "name": str(name)[:NAME_MAX],
        "amount": tenth(_fnum(amount, 0.0, 100.0, 100.0)),
        "dim": dim,
        "size": size,
        "domain_min": [float(v) for v in domain_min],
        "domain_max": [float(v) for v in domain_max],
        "table": pack_table(table),
    }


def normalize(raw: Any) -> dict[str, Any] | None:
    """Clamp a LUT dict into shape, or None when it would change nothing.

    Same contract as `editing.normalize_hsl`: a switched-on-but-empty LUT, a
    zero amount and an identity table all normalize to None, so a look that is
    loaded but doing nothing never makes an edit count as edited.

    Scalars are clamped in the lenient house style. The table is not: a declared
    size that disagrees with the number of values it holds is a corrupt look, and
    there is no honest way to repair one, so it asserts.
    """
    if not isinstance(raw, dict):
        return None
    out = dict(DEFAULT_LUT)
    out["domain_min"] = list(DEFAULT_LUT["domain_min"])
    out["domain_max"] = list(DEFAULT_LUT["domain_max"])
    out["enabled"] = bool(raw.get("enabled", False))
    name = raw.get("name")
    out["name"] = str(name)[:NAME_MAX] if isinstance(name, str) else ""
    out["amount"] = tenth(_fnum(raw.get("amount"), 0.0, 100.0, 100.0))

    blob = raw.get("table") or ""
    if not isinstance(blob, str) or not blob:
        return None

    dim = raw.get("dim", 3)
    assert dim in (1, 3), f"dim must be 1 or 3, got {dim!r}"
    size = int(raw.get("size", 0))
    cap = MAX_1D_SIZE if dim == 1 else MAX_3D_SIZE
    assert 2 <= size <= cap, f"a {dim}D LUT needs size 2..{cap}, got {size}"
    out["dim"] = dim
    out["size"] = size
    out["table"] = blob
    unpack_table(blob, entry_count(dim, size))   # asserts the length agrees

    for key, default in (("domain_min", 0.0), ("domain_max", 1.0)):
        src = raw.get(key)
        if isinstance(src, (list, tuple)) and len(src) == 3:
            out[key] = [_fnum(v, -16.0, 16.0, default) for v in src]
    for c in range(3):
        assert out["domain_max"][c] - out["domain_min"][c] > _EPS, (
            f"domain_max must exceed domain_min on channel {c}")

    if is_neutral(out):
        return None
    return out


def is_neutral(params: dict[str, Any] | None) -> bool:
    """True when applying this would leave the pixels alone.

    Includes the identity table, so an unmodified template .cube — or a colour
    match fitted between two frames that already agree — costs nothing at render
    time and is byte-exact rather than merely close.
    """
    if not params or not params.get("enabled"):
        return True
    p = {**DEFAULT_LUT, **params}
    if p["amount"] <= 0 or not p["table"] or p["size"] < 2:
        return True
    if p["domain_min"] != [0.0, 0.0, 0.0] or p["domain_max"] != [1.0, 1.0, 1.0]:
        return False   # a domain remap is not the identity even with a ramp
    table = unpack_table(p["table"], entry_count(p["dim"], p["size"]))
    ident = identity_table(p["dim"], p["size"])
    # 1.5 quantization steps: an identity written out as uint16 and read back.
    return bool(np.abs(table - ident).max() < 1.5 / 65535.0)


# ----- applying -----------------------------------------------------------

def _coords(rgb: np.ndarray, params: dict[str, Any], size: int) -> np.ndarray:
    """Frame values -> table coordinates in [0, size-1], per channel."""
    dmin = np.asarray(params["domain_min"], dtype=np.float32)
    dmax = np.asarray(params["domain_max"], dtype=np.float32)
    x = (rgb - dmin) / (dmax - dmin)
    return np.clip(x, 0.0, 1.0) * np.float32(size - 1)


@lru_cache(maxsize=4)
def _grid_1d(blob: str, size: int) -> tuple[tuple[np.ndarray, ...], int]:
    """The three curves as (1, N, 1) single-row images, plus the N they ended up
    with. Refined for the same reason `_grid_2d` refines: the remap weight is
    carried in 1/32ths of a cell, so a four-entry curve needs finer cells."""
    table = unpack_table(blob, size)
    factor = -(-_REMAP_CELLS // (size - 1))
    if factor > 1:
        n2 = (size - 1) * factor + 1
        xs = np.linspace(0.0, size - 1, n2)
        src = np.arange(size, dtype=np.float64)
        # A piecewise-linear curve resampled on a refinement of its own knots is
        # the same curve, so this costs accuracy nothing.
        table = np.stack([np.interp(xs, src, table[:, c]) for c in range(3)],
                         axis=1).astype(np.float32)
        size = n2
    curves = tuple(np.ascontiguousarray(table[:, c]).reshape(1, size, 1)
                   for c in range(3))
    return curves, size


def _apply_1d(rgb: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Three independent monotone curves, one `cv2.remap` each.

    The obvious version — floor, gather the two neighbours, lerp — is two fancy
    indexes and four full-frame temporaries per channel, and measured 640 ms on
    24 MP against 245 ms for this. A one-row image sampled with INTER_LINEAR is
    the same linear interpolation with the loop in OpenCV.
    """
    curves, size = _grid_1d(params["table"], params["size"])
    h, w = rgb.shape[:2]
    band = max(1, min(h, _TILE_PIXELS // max(1, w)))
    out = np.empty_like(rgb)
    for y in range(0, h, band):
        pos = _coords(rgb[y:y + band], params, size)
        row = np.zeros(pos.shape[:2], dtype=np.float32)
        for c in range(3):
            got = cv2.remap(curves[c], np.ascontiguousarray(pos[..., c]), row,
                            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out[y:y + band, :, c] = got if got.ndim == 2 else got[..., 0]
    return out


def _resample(table: np.ndarray, size: int, factor: int) -> np.ndarray:
    """Refine a cube by an integer factor, exactly.

    Because the new nodes fall on the old grid's own trilinear surface and every
    new cell lies inside one old cell, interpolating the refined table gives back
    the same function to the last bit — this adds resolution, not detail. It is
    the eight-corner gather, which is affordable here and nowhere else: a few
    tens of thousands of nodes once per LUT against 24 million pixels per frame.
    """
    n2 = (size - 1) * factor + 1
    pos = identity_table(3, n2).astype(np.float64) * (size - 1)
    i0 = np.clip(np.floor(pos), 0, size - 2).astype(np.int64)
    frac = pos - i0
    base = i0[:, 0] + size * i0[:, 1] + size * size * i0[:, 2]
    out = np.zeros((n2 ** 3, 3), dtype=np.float64)
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = np.ones(len(pos))
                for axis, d in enumerate((dr, dg, db)):
                    w = w * (frac[:, axis] if d else 1.0 - frac[:, axis])
                out += w[:, None] * table[base + dr + size * dg + size * size * db]
    return out.astype(np.float32)


@lru_cache(maxsize=4)
def _grid_2d(blob: str, size: int) -> tuple[np.ndarray, int]:
    """The cube as an (N*N, N, 3) image — row `b*N + g`, column `r` — plus the N
    it ended up with.

    No transpose and no copy for the layout itself: a .cube stores entry
    `r + N*g + N*N*b`, so the reshape already lays red along the columns and
    (g, b) down the rows.

    The refinement in front of it is there because `cv2.remap` carries its
    interpolation weight in 1/32ths of a cell, so the error scales with how wide
    a cell is. A 33-node cube has cells 1/32 of the range wide and lands within a
    quarter of a code value; an 8-node one is four times as coarse and a 2-node
    one — a plain colour matrix, which people do ship — was out by four code
    values, which is visible. Refining every cube to at least 32 cells per axis
    first puts all of them on the same quarter-code-value footing, for a
    few-hundred-KB table built once and cached.
    """
    table = unpack_table(blob, size ** 3)
    factor = -(-_REMAP_CELLS // (size - 1))      # ceil
    if factor > 1:
        table = _resample(table, size, factor)
        size = (size - 1) * factor + 1
    return np.ascontiguousarray(table.reshape(size * size, size, 3)), size


def _apply_3d(rgb: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Trilinear interpolation as two `cv2.remap` passes and a lerp.

    Bilinear in (r, g) at the two bracketing blue planes, then linear between
    them. The row coordinate is `b0*N + g`, and because `g0` is clipped to
    `N-2` the bilinear pair `g0, g0+1` never crosses out of a blue plane's block
    of rows — which is what makes packing the cube into one 2D image legitimate
    rather than an approximation.

    Tiled by rows so the coordinate maps and the two intermediates stay bounded
    regardless of megapixels.
    """
    tab, size = _grid_2d(params["table"], params["size"])
    h, w = rgb.shape[:2]
    band = max(1, min(h, _TILE_PIXELS // max(1, w)))
    out = np.empty_like(rgb)
    for y in range(0, h, band):
        chunk = rgb[y:y + band]
        pos = _coords(chunk, params, size)
        b0 = np.clip(np.floor(pos[..., 2]), 0, size - 2)
        fb = (pos[..., 2] - b0)[..., None]
        map_r = np.ascontiguousarray(pos[..., 0])
        row_g = np.ascontiguousarray(pos[..., 1])
        lo = cv2.remap(tab, map_r, b0 * size + row_g,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        hi = cv2.remap(tab, map_r, (b0 + 1.0) * size + row_g,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out[y:y + band] = lo + (hi - lo) * fb
    return out


def apply_lut(rgb: np.ndarray, params: dict[str, Any] | None) -> np.ndarray:
    """Apply a look to float32 RGB in [0,1]. Returns a new array of the same shape.

    A neutral LUT returns the input array itself, so it is a byte-exact no-op and
    the stage can sit in the pipeline permanently.
    """
    assert isinstance(rgb, np.ndarray), "apply_lut wants an array"
    assert rgb.dtype == np.float32, f"apply_lut works in float32, got {rgb.dtype}"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected (h, w, 3), got {rgb.shape}"
    if is_neutral(params):
        return rgb
    p = {**DEFAULT_LUT, **params}
    graded = _apply_1d(rgb, p) if p["dim"] == 1 else _apply_3d(rgb, p)
    amount = p["amount"] / 100.0
    if amount < 1.0:
        graded = rgb + (graded - rgb) * np.float32(amount)
    return np.clip(graded, 0.0, 1.0)


# ----- colour match -------------------------------------------------------
#
# The model: a linear Monge-Kantorovich colour transport, then a per-channel
# marginal (histogram) match, baked together into one 3D table.
#
# Why these two and in this order. The frames are not registered — they are
# different photographs, possibly of different subjects — so there are no pixel
# correspondences and nothing to regress against. Only the *distributions* are
# comparable, which rules out the polynomial fit outright: a polynomial needs
# pairs.
#
# Of what is left:
#   - Reinhard mean/std in a decorrelated space is the two-moment special case of
#     what is done here, and it cannot express a crossover — the thing that
#     actually separates two camera bodies, where the shadows part one way and
#     the highlights the other.
#   - Per-channel histogram matching alone captures that crossover but is blind
#     to the correlation between channels, so it cannot express the hue rotation
#     that a different colour matrix produces.
# The linear transport step fixes the full 3x3 covariance, which is exactly the
# cross-channel structure the marginals cannot see; the marginal match then
# fixes the shape of each channel's response, which two moments cannot see. They
# are complementary, and running the linear step first means the marginals it is
# handed are already close, so the 1D maps stay gentle and near-monotone-linear
# instead of doing all the work themselves.
#
# Baking to a table is what makes any of this storable. The composition of a
# matrix and three curves is not a matrix and is not three curves, and a table
# does not care: 17^3 nodes hold it to a third of a code value, re-apply
# deterministically, and go down the same fast path as an imported .cube.


def _stats_view(img: np.ndarray) -> np.ndarray:
    """A frame reduced to a fixed working size, as (n, 3) float64 in [0,1].

    Fixed size, area-averaged, no sampling: the fit must be deterministic, and a
    reference and a source of different pixel dimensions must not weight their
    statistics differently just because one camera has more photosites.
    """
    assert isinstance(img, np.ndarray), "colour match wants arrays"
    assert img.dtype == np.float32, f"colour match works in float32, got {img.dtype}"
    assert img.ndim == 3 and img.shape[2] == 3, f"expected (h, w, 3), got {img.shape}"
    h, w = img.shape[:2]
    assert h >= 2 and w >= 2, f"frame is too small to fit anything from: {img.shape}"
    scale = _STAT_EDGE / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(2, int(round(w * scale))),
                               max(2, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.0, 1.0).reshape(-1, 3).astype(np.float64)


def _psd_sqrt(m: np.ndarray, power: float) -> np.ndarray:
    """`m ** power` for a symmetric positive-semidefinite matrix, via its
    eigendecomposition. The floor keeps a near-degenerate channel — a frame with
    almost no blue variation — from turning into an inverse of infinity."""
    w, v = np.linalg.eigh(m)
    floor = max(w.max(), _EPS) * 1e-6
    return (v * np.power(np.maximum(w, floor), power)) @ v.T


def _transport_matrix(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The closed-form linear map taking the source's (mean, covariance) onto the
    reference's. Pitie & Kokaram's Monge-Kantorovich solution: of all linear maps
    that match both moments this is the one of least displacement, so it does not
    add a rotation nobody asked for."""
    mu_s = src.mean(axis=0)
    mu_r = ref.mean(axis=0)
    cov_s = np.cov(src, rowvar=False)
    cov_r = np.cov(ref, rowvar=False)
    assert np.trace(cov_s) > 1e-9, "the source frame has no colour variation to match"
    assert np.trace(cov_r) > 1e-9, "the reference frame has no colour variation to match from"
    s_half = _psd_sqrt(cov_s, 0.5)
    s_inv_half = _psd_sqrt(cov_s, -0.5)
    mid = _psd_sqrt(s_half @ cov_r @ s_half, 0.5)
    return s_inv_half @ mid @ s_inv_half, mu_s, mu_r


def _marginal_maps(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Three monotone [0,1] -> [0,1] curves matching each channel's CDF, sampled
    at `_HIST_BINS` points.

    Smoothed before it is used. A raw CDF match is a staircase wherever a
    histogram is sparse — the top two stops of a frame with no highlights are a
    handful of pixels deciding the shape of the whole shoulder — and that
    staircase is visible as banding in a sky. The box filter is wide enough to
    average over that and narrow enough to keep a real crossover.

    What is smoothed is the map's *deviation from the identity*, not the map. A
    box filter over the map itself has to invent values past both ends, and
    whichever way it invents them it bends the last half-window: replicating the
    edge lifted the blacks by two code values even when the two frames were the
    same frame. The deviation really is flat outside the range, so padding it
    with its edge value is not an invention, and a fit between two identical
    frames comes back as the identity to a rounding error.

    Two passes, not one — the box run twice is a triangular kernel, which throws
    away far more of the ripple for the same width. On a frame with a spiky
    histogram one pass left the map's slope swinging between 0 and 3.9, which is
    both a noise amplifier and too curved for a 17-node grid to hold; two passes
    bring that to 2.4 and cut the baked table's worst error from 10 code values
    to 3.5, while the match itself measures the same.
    """
    nb = _HIST_BINS
    centres = (np.arange(nb, dtype=np.float64) + 0.5) / nb
    win = max(3, nb // 16)
    kernel = np.full(win, 1.0 / win)
    out = np.empty((nb, 3), dtype=np.float64)
    for c in range(3):
        hs = np.histogram(src[:, c], bins=nb, range=(0.0, 1.0))[0]
        hr = np.histogram(ref[:, c], bins=nb, range=(0.0, 1.0))[0]
        cs = np.cumsum(hs) / max(hs.sum(), 1)
        cr = np.cumsum(hr) / max(hr.sum(), 1)
        dev = np.interp(cs, cr, centres) - centres
        # Outside the source's populated range the CDF is flat and the match is
        # meaningless: an empty bin can be sent anywhere and still match. Left
        # alone it does real damage, because the look is fitted on one frame and
        # then applied to the rest of the shoot — a fitting frame with no true
        # blacks produced a map that crushed every later frame's shadows onto
        # its own darkest tone. Holding the deviation flat past the ends instead
        # extrapolates the correction the populated range actually asked for.
        lo = int(np.searchsorted(cs, _CDF_EDGE, side="left"))
        hi = int(np.searchsorted(cs, 1.0 - _CDF_EDGE, side="left"))
        lo = min(lo, nb - 1)
        hi = min(max(hi, lo), nb - 1)
        dev[:lo] = dev[lo]
        dev[hi:] = dev[hi]
        for _ in range(_SMOOTH_PASSES):
            pad = np.concatenate([np.full(win, dev[0]), dev, np.full(win, dev[-1])])
            dev = np.convolve(pad, kernel, mode="same")[win:win + nb]
        m = centres + dev
        # Monotone by construction before smoothing; keep it so after.
        out[:, c] = np.maximum.accumulate(np.clip(m, 0.0, 1.0))
    return out


def _apply_model(grid: np.ndarray, matrix: np.ndarray, mu_s: np.ndarray,
                 mu_r: np.ndarray, maps: np.ndarray) -> np.ndarray:
    """The fitted transform evaluated directly, on (n, 3) float64 in [0,1]."""
    moved = np.clip((grid - mu_s) @ matrix.T + mu_r, 0.0, 1.0)
    nb = maps.shape[0]
    centres = (np.arange(nb, dtype=np.float64) + 0.5) / nb
    # The curve is sampled at bin centres, so it says nothing about the half bin
    # at either end. Extending it by its own deviation there rather than by its
    # value keeps the identity exactly the identity all the way to 0 and 1.
    xs = np.concatenate([[0.0], centres, [1.0]])
    out = np.empty_like(moved)
    for c in range(3):
        lo = maps[0, c] - centres[0]
        hi = maps[-1, c] - centres[-1] + 1.0
        ys = np.concatenate([[lo], maps[:, c], [hi]])
        out[:, c] = np.interp(moved[:, c], xs, np.clip(ys, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0)


def fit_from_reference(source: np.ndarray, reference: np.ndarray,
                       size: int = FIT_SIZE, amount: int = 100,
                       name: str = "Colour match") -> dict[str, Any]:
    """Fit a look that moves `source`'s colour towards `reference`'s.

    Both are float32 RGB in [0,1] and need not be the same size, the same scene,
    or registered — the fit is statistical. The result is an ordinary LUT
    parameter dict: store it, re-apply it to the rest of that body's frames with
    `apply_lut`, dial it back with `amount`. The reference frame is not needed
    again, and is deliberately not referenced by the result.

    What it is for: a shoot covered with two bodies, or one body across a firmware
    or profile change, where the two sets of frames should print alike. Give it
    one frame from each of the same subject under the same light and it will
    reproduce the other body's white balance, contrast shape and colour matrix
    closely.

    What it cannot do, and will do wrong if asked:

      - It matches distributions, not content. Hand it a source of a grey street
        and a reference of a red sunset and it will happily paint the street red;
        nothing in the method can tell a colour cast from a red subject. Frames
        of the same subject under the same light are the working case, frames of
        different scenes are a guess, and the further apart the content the more
        confidently wrong the answer.
      - It transfers exposure and contrast along with colour, because a camera
        profile difference is partly tonal and separating the two from marginals
        alone is not possible. If only the colour should move, that is a job for
        white balance, not for this.
      - It is one global map. Two bodies metering differently under mixed
        lighting need different corrections in the tungsten and the daylight
        parts of the frame, and a single table cannot hold both.
      - It cannot invert clipping. Detail the source blew out is gone, and the
        match will map the flat patch to whatever the reference has there.

    `amount` exists because of the first limitation: when the content differs,
    half the match is usually right and all of it is not.
    """
    assert 2 <= size <= MAX_3D_SIZE, f"fit size must be 2..{MAX_3D_SIZE}, got {size}"
    src = _stats_view(source)
    ref = _stats_view(reference)
    matrix, mu_s, mu_r = _transport_matrix(src, ref)
    maps = _marginal_maps(np.clip((src - mu_s) @ matrix.T + mu_r, 0.0, 1.0), ref)
    grid = identity_table(3, size).astype(np.float64)
    table = _apply_model(grid, matrix, mu_s, mu_r, maps)
    return _params(name, amount, 3, size, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], table)
