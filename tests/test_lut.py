"""Checks for colour LUTs and the colour match.

    uv run python tests/test_lut.py

Two things are being defended here.

The first is that a LUT is data someone else wrote, so parsing has to be strict:
a file with the wrong number of entries or no declared size is an error, not
something to patch up, because a quietly repaired LUT renders a photograph
wrongly and nothing says so. Every malformed case below asserts.

The second is that the fast apply path is still the right answer. `_apply_3d`
trades exact float weights for `cv2.remap`'s fixed-point ones, so the tests
measure that trade rather than assuming it: trilinear against a hand-written
eight-corner reference, and a smooth gradient checked for the banding that a
nearest-neighbour lookup would leave.
"""
from __future__ import annotations

import numpy as np

from picture_classifier import editing, lut


# ----- helpers ------------------------------------------------------------

def _cube_text(size: int, fn, title: str = "Test look",
               domain: tuple[list[float], list[float]] | None = None) -> str:
    """Write a 3D .cube the way a real one is written: red varying fastest."""
    lines = [f'TITLE "{title}"', "# a comment, and a blank line follow", "",
             f"LUT_3D_SIZE {size}"]
    if domain is not None:
        lines.append("DOMAIN_MIN " + " ".join(f"{v:g}" for v in domain[0]))
        lines.append("DOMAIN_MAX " + " ".join(f"{v:g}" for v in domain[1]))
    for b in range(size):
        for g in range(size):
            for r in range(size):
                out = fn(r / (size - 1), g / (size - 1), b / (size - 1))
                lines.append("  {:.6f}   {:.6f} {:.6f}  ".format(*out))
    return "\n".join(lines) + "\n"


def _params(table: np.ndarray, size: int, dim: int = 3, amount: int = 100,
            domain: tuple[list[float], list[float]] | None = None) -> dict:
    lo, hi = domain if domain is not None else ([0.0] * 3, [1.0] * 3)
    return lut._params("t", amount, dim, size, lo, hi, table)


def _reference_trilinear(rgb: np.ndarray, params: dict) -> np.ndarray:
    """Trilinear interpolation written out corner by corner, in float64.

    Deliberately the slow, obvious version: the point of `_apply_3d` is that it
    is not this, so this is what it has to agree with.
    """
    n = params["size"]
    table = lut.unpack_table(params["table"], n ** 3).astype(np.float64)
    pos = np.clip(rgb.astype(np.float64), 0.0, 1.0) * (n - 1)
    i0 = np.clip(np.floor(pos), 0, n - 2).astype(np.int64)
    frac = pos - i0
    base = i0[..., 0] + n * i0[..., 1] + n * n * i0[..., 2]
    out = np.zeros(rgb.shape, dtype=np.float64)
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = np.ones(rgb.shape[:2])
                for axis, d in enumerate((dr, dg, db)):
                    w = w * (frac[..., axis] if d else 1.0 - frac[..., axis])
                out += w[..., None] * table[base + dr + n * dg + n * n * db]
    return out


def _asserts(fn) -> str:
    """Run `fn`, require an AssertionError, hand back its message."""
    try:
        fn()
    except AssertionError as exc:
        return str(exc)
    raise AssertionError("expected an AssertionError, nothing was raised")


def _scene(seed: int, h: int = 240, w: int = 320) -> np.ndarray:
    """A frame with gradients, a few flat patches and some noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([xx / w, yy / h,
                    0.35 + 0.3 * np.cos(xx / 37.0) * np.sin(yy / 29.0)], axis=-1)
    img[20:60, 20:80] = (0.82, 0.74, 0.61)
    img[120:180, 200:300] = (0.11, 0.14, 0.22)
    img += rng.normal(0.0, 0.015, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ----- parsing ------------------------------------------------------------

def test_parse_a_3d_cube() -> None:
    size = 5
    text = _cube_text(size, lambda r, g, b: (g, b, r))   # a channel rotation
    cube = lut.parse_cube(text)
    assert cube["dim"] == 3 and cube["size"] == size
    assert cube["title"] == "Test look"
    assert cube["domain_min"] == [0.0, 0.0, 0.0]
    assert cube["domain_max"] == [1.0, 1.0, 1.0]
    assert cube["table"].shape == (size ** 3, 3)
    # Red fastest: entry r + n*g + n*n*b is the node at (r, g, b).
    r, g, b = 1, 2, 3
    node = cube["table"][r + size * g + size * size * b]
    assert np.allclose(node, [g / 4, b / 4, r / 4]), node


def test_parse_a_1d_cube() -> None:
    text = "\n".join(["# a 1D contrast curve", "LUT_1D_SIZE 4"]
                     + ["0.0 0.0 0.0", "0.2 0.25 0.3",
                        "0.7 0.75 0.8", "1.0 1.0 1.0"])
    cube = lut.parse_cube(text)
    assert cube["dim"] == 1 and cube["size"] == 4
    assert cube["title"] == "", "no TITLE means no title, not a guessed one"
    assert cube["table"].shape == (4, 3)
    assert np.allclose(cube["table"][1], [0.2, 0.25, 0.3])


def test_parse_tolerates_the_formatting_a_real_file_has() -> None:
    text = ("\r\n\r\n"
            "  # leading comment\r\n"
            '\ttitle   "Odd \'un"  \r\n'
            "lut_3d_size\t2\r\n"
            "DOMAIN_MIN  0 0 0\r\n"
            "DOMAIN_MAX 1 1 1\r\n"
            + "".join(f"\t{i % 2}\t {i % 2} \t{i % 2}\r\n" for i in range(8))
            + "  # trailing comment  \r\n")
    cube = lut.parse_cube(text)
    assert cube["size"] == 2 and cube["title"] == "Odd 'un"
    assert cube["table"].shape == (8, 3)


def test_parse_reads_a_domain() -> None:
    text = _cube_text(2, lambda r, g, b: (r, g, b),
                      domain=([0.0, 0.0, 0.0], [2.0, 2.0, 4.0]))
    cube = lut.parse_cube(text)
    assert cube["domain_max"] == [2.0, 2.0, 4.0]


def test_malformed_cubes_all_assert() -> None:
    good = _cube_text(3, lambda r, g, b: (r, g, b))

    # One entry short.
    short = "\n".join(good.strip().splitlines()[:-1]) + "\n"
    msg = _asserts(lambda: lut.parse_cube(short))
    assert "27 entries" in msg and "26" in msg, msg

    # One entry too many.
    msg = _asserts(lambda: lut.parse_cube(good + "0.5 0.5 0.5\n"))
    assert "27 entries" in msg and "28" in msg, msg

    # No size declared at all.
    msg = _asserts(lambda: lut.parse_cube("0.0 0.0 0.0\n1.0 1.0 1.0\n"))
    assert "not a .cube" in msg, msg

    # A size out of range, and a size that is not a number.
    assert "must be 2..64" in _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 1\n"))
    assert "must be 2..64" in _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 99\n"))
    assert "must be 2..65536" in _asserts(lambda: lut.parse_cube("LUT_1D_SIZE 0\n"))

    # Two sizes.
    msg = _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 2\nLUT_1D_SIZE 4\n"))
    assert "already declared" in msg, msg

    # A row with the wrong number of numbers, and a typo'd keyword.
    msg = _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 2\n0.1 0.2\n"))
    assert "line 2" in msg and "three numbers" in msg, msg
    assert "line 1" in _asserts(lambda: lut.parse_cube("LUT_3D_SIZ 2\n"))

    # A degenerate domain, and a domain with the wrong arity.
    flat = _cube_text(2, lambda r, g, b: (r, g, b),
                      domain=([0.0, 0.0, 0.0], [1.0, 0.0, 1.0]))
    assert "DOMAIN_MAX must exceed" in _asserts(lambda: lut.parse_cube(flat))
    msg = _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 2\nDOMAIN_MIN 0 0\n"))
    assert "three numbers" in msg, msg

    # Not a LUT at all: the first line that is neither a keyword nor three
    # numbers is where it stops, and it says so.
    msg = _asserts(lambda: lut.parse_cube("<?xml version='1.0'?>\n"))
    assert "line 1" in msg and "known keyword" in msg, msg
    assert "not a .cube" in _asserts(lambda: lut.parse_cube("# just a comment\n"))

    # Declared but unimplemented: better to refuse than to render it wrongly.
    msg = _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 2\nLUT_IN_VIDEO_RANGE\n"))
    assert "video-range" in msg, msg

    # A non-finite entry.
    msg = _asserts(lambda: lut.parse_cube("LUT_3D_SIZE 2\n" + "nan nan nan\n" * 8))
    assert "not a finite number" in msg, msg


def test_load_cube_names_an_untitled_file(tmp_path) -> None:
    path = tmp_path / "Kodak 2383 D55.cube"
    path.write_text("LUT_3D_SIZE 2\n" + "0 0 0\n" * 8)
    assert lut.load_cube(path)["title"] == "Kodak 2383 D55"


# ----- schema -------------------------------------------------------------

def test_the_empty_lut_is_neutral() -> None:
    assert lut.is_neutral(None)
    assert lut.is_neutral({})
    assert lut.is_neutral(lut.DEFAULT_LUT)
    assert lut.normalize(None) is None
    assert lut.normalize("nonsense") is None
    assert lut.normalize({}) is None
    assert lut.normalize({"enabled": True}) is None, "enabled with no table is nothing"


def test_normalize_clamps_and_round_trips() -> None:
    table = lut.identity_table(3, 3).copy()
    table[0] = (0.5, 0.5, 0.5)          # so it is not the identity
    p = _params(table, 3, amount=100)
    p["amount"] = 400
    p["name"] = "x" * 500
    got = lut.normalize(p)
    assert got is not None
    assert got["amount"] == 100 and len(got["name"]) == lut.NAME_MAX
    assert lut.normalize({**p, "amount": -5}) is None, "amount 0 is neutral"
    assert lut.normalize({**p, "enabled": False}) is None
    # A full round trip leaves the dict alone.
    assert lut.normalize(got) == got


def test_an_identity_table_is_neutral_but_a_domain_is_not() -> None:
    ident = _params(lut.identity_table(3, 9), 9)
    assert lut.is_neutral(ident), "a template LUT should cost nothing"
    assert lut.normalize(ident) is None
    assert lut.is_neutral(_params(lut.identity_table(1, 64), 64, dim=1))
    # Same ramp, but reading only the bottom half of the range: not the identity.
    shifted = _params(lut.identity_table(3, 9), 9,
                      domain=([0.0] * 3, [0.5, 0.5, 0.5]))
    assert not lut.is_neutral(shifted)


def test_a_corrupt_table_asserts_rather_than_being_repaired() -> None:
    good = _params(lut.identity_table(3, 3) * 0.5, 3)
    msg = _asserts(lambda: lut.normalize({**good, "size": 4}))
    assert "values" in msg and "192" in msg, msg
    assert "dim must be 1 or 3" in _asserts(lambda: lut.normalize({**good, "dim": 2}))
    assert "size 2..64" in _asserts(lambda: lut.normalize({**good, "size": 1}))


def test_table_key_identifies_the_look() -> None:
    a = _params(lut.identity_table(3, 5) ** 0.9, 5)
    b = _params(lut.identity_table(3, 5) ** 0.8, 5)
    assert lut.table_key(a) and lut.table_key(a) == lut.table_key(dict(a))
    assert lut.table_key(a) != lut.table_key(b)
    assert lut.table_key(None) == "" and lut.table_key({}) == ""
    # The same numbers read over another input range are another look, and the
    # key is what an edit stores, so the two must not collide.
    wide = _params(lut.identity_table(3, 5) ** 0.9, 5, domain=([0.0] * 3, [2.0] * 3))
    assert lut.table_key(wide) != lut.table_key(a)
    # The name and the amount are not part of what the table is.
    assert lut.table_key({**a, "name": "other", "amount": 40}) == lut.table_key(a)


# ----- riding in an edit --------------------------------------------------

def _look():
    params = lut.normalize(_params(lut.identity_table(3, 9) ** 0.6, 9))
    key = lut.table_key(params)
    return key, params


def _photo() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)


def test_an_edit_stores_a_reference_not_the_table() -> None:
    key, _ = _look()
    e = editing.normalize({"lut": {"key": key, "name": "Gamma", "amount": 140}})
    assert e["lut"] == {"key": key, "name": "Gamma", "amount": 100}
    assert editing.normalize({"lut": {"key": key, "amount": 0}})["lut"] is None
    assert editing.normalize({"lut": {"key": "../../etc", "amount": 50}})["lut"] is None
    assert not editing.is_neutral({"lut": {"key": key, "amount": 30}})
    assert editing.is_neutral({"lut": {"key": key, "amount": 0}})
    a = editing.edit_hash({"lut": {"key": key, "amount": 30}})
    assert a and a != editing.edit_hash({"lut": {"key": key, "amount": 60}})


def test_the_look_runs_under_the_sliders() -> None:
    """Look first, then the grade, so a colour match fitted on ungraded pixels
    is handed ungraded pixels. Exposure after a gamma curve is not the same as
    the curve after exposure, which is what makes the order visible."""
    key, params = _look()
    img = _photo()
    looked = np.rint(lut.apply_lut(img.astype(np.float32) / 255.0, params) * 255.0).astype(np.uint8)
    want = editing.render(looked, {"exposure": 0.7})
    got = editing.render(img, {"exposure": 0.7, "lut": {"key": key, "amount": 100}},
                         luts={key: params})
    assert np.array_equal(got, want)
    wrong = np.rint(lut.apply_lut(editing.render(img, {"exposure": 0.7}).astype(np.float32) / 255.0,
                                  params) * 255.0).astype(np.uint8)
    assert not np.array_equal(got, wrong)


def test_amount_blends_the_look() -> None:
    key, params = _look()
    img = _photo()
    half = editing.render(img, {"lut": {"key": key, "amount": 50}}, luts={key: params})
    full = editing.render(img, {"lut": {"key": key, "amount": 100}}, luts={key: params})
    mid = (img.astype(np.float32) + full.astype(np.float32)) / 2.0
    assert np.abs(half.astype(np.float32) - mid).max() <= 1.0


def test_a_missing_table_is_an_error_not_a_blank_look() -> None:
    key, _ = _look()
    try:
        editing.render(_photo(), {"lut": {"key": key, "amount": 100}})
    except AssertionError as exc:
        assert key in str(exc)
    else:
        raise AssertionError("rendered a look without its table")


def test_a_preset_added_on_top_brings_its_look() -> None:
    key, _ = _look()
    merged = editing.merge_additive({"exposure": 0.5}, {"lut": {"key": key, "amount": 70}})
    assert merged["exposure"] == 0.5 and merged["lut"]["key"] == key
    kept = editing.merge_additive({"lut": {"key": key, "amount": 70}}, {"contrast": 10})
    assert kept["lut"]["amount"] == 70


# ----- applying -----------------------------------------------------------

def test_the_identity_lut_is_a_byte_exact_no_op() -> None:
    img = _scene(1)
    for params in (None, {}, lut.DEFAULT_LUT,
                   _params(lut.identity_table(3, 17), 17),
                   _params(lut.identity_table(1, 256), 256, dim=1)):
        out = lut.apply_lut(img, params)
        assert out is img, "a neutral LUT should hand the array straight back"
        assert np.array_equal(out, img)


def test_an_identity_table_is_still_a_no_op_through_the_fast_path() -> None:
    """The public path short-circuits on a neutral LUT, so check the interpolator
    itself: `cv2.remap` carries the bilinear weights in 1/32ths, which is the one
    inexactness in the whole module and is worth pinning a number to."""
    img = _scene(2)
    ident = {**lut.DEFAULT_LUT, **_params(lut.identity_table(3, 33), 33)}
    err = np.abs(lut._apply_3d(img, ident) - img).max()
    assert err * 255 < 0.2, f"{err * 255:.3f} code values"
    ident1 = {**lut.DEFAULT_LUT, **_params(lut.identity_table(1, 1024), 1024, dim=1)}
    err1 = np.abs(lut._apply_1d(img, ident1) - img).max()
    assert err1 * 255 < 0.05, f"{err1 * 255:.4f} code values"


def test_trilinear_matches_a_corner_by_corner_reference() -> None:
    size = 17
    grid = lut.identity_table(3, size).astype(np.float64)
    # Something with curvature and cross-channel mixing in it.
    table = np.clip(grid ** 1.3 @ np.array([[0.9, 0.08, 0.02],
                                            [0.05, 1.0, 0.05],
                                            [0.03, 0.1, 0.87]]).T
                    + 0.04 * np.sin(grid[:, ::-1] * 5.0), 0.0, 1.0)
    params = {**lut.DEFAULT_LUT, **_params(table, size)}
    rng = np.random.default_rng(4)
    img = rng.random((120, 160, 3), dtype=np.float32)
    err = np.abs(lut._apply_3d(img, params) - _reference_trilinear(img, params))
    assert err.max() * 255 < 0.5, f"worst {err.max() * 255:.3f} code values"
    assert err.mean() * 255 < 0.1, f"mean {err.mean() * 255:.3f} code values"


def test_a_known_lut_hits_hand_computed_colours() -> None:
    """A 2-node cube holding a pure channel rotation. With only the eight corners
    to work from, trilinear interpolation of a linear map is exact everywhere, so
    these are values worked out on paper rather than tolerances."""
    text = _cube_text(2, lambda r, g, b: (g, b, r))
    params = {**lut.DEFAULT_LUT, **lut.params_from_cube(lut.parse_cube(text))}
    probe = np.array([[[0.2, 0.4, 0.8], [1.0, 0.0, 0.0], [0.0, 0.5, 1.0],
                       [0.25, 0.25, 0.25], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
                     dtype=np.float32)
    want = np.array([[[0.4, 0.8, 0.2], [0.0, 0.0, 1.0], [0.5, 1.0, 0.0],
                      [0.25, 0.25, 0.25], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]],
                    dtype=np.float32)
    got = lut.apply_lut(probe, params)
    assert np.abs(got - want).max() * 255 < 0.5, got - want

    # A 3-node cube that only lifts the midpoint of every channel: the node is
    # hit exactly, and half way to it is half the lift.
    def mid(r, g, b):
        return tuple(min(1.0, v + (0.25 if abs(v - 0.5) < 1e-6 else 0.0))
                     for v in (r, g, b))
    cube = lut.parse_cube(_cube_text(3, mid))
    p3 = {**lut.DEFAULT_LUT, **lut.params_from_cube(cube)}
    probe = np.array([[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.0, 0.0, 0.0]]],
                     dtype=np.float32)
    got = lut.apply_lut(probe, p3)
    assert np.abs(got[0, 0] - 0.75).max() * 255 < 0.5, got[0, 0]
    assert np.abs(got[0, 1] - 0.375).max() * 255 < 0.5, got[0, 1]
    assert np.abs(got[0, 2] - 0.0).max() * 255 < 0.5, got[0, 2]


def test_a_1d_lut_is_three_curves() -> None:
    text = "LUT_1D_SIZE 3\n0 0 0\n0.75 0.5 0.25\n1 1 1\n"
    params = lut.params_from_cube(lut.parse_cube(text))
    probe = np.array([[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]], dtype=np.float32)
    got = lut.apply_lut(probe, params)
    assert np.abs(got[0, 0] - [0.75, 0.5, 0.25]).max() * 255 < 0.5, got[0, 0]
    assert np.abs(got[0, 1] - [0.375, 0.25, 0.125]).max() * 255 < 0.5, got[0, 1]


def test_a_domain_rescales_the_input() -> None:
    """DOMAIN_MAX 0.5 means the table's last node is reached at half scale, and
    everything above it clamps there."""
    text = _cube_text(2, lambda r, g, b: (r, g, b),
                      domain=([0.0] * 3, [0.5] * 3))
    params = lut.params_from_cube(lut.parse_cube(text))
    probe = np.array([[[0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]]],
                     dtype=np.float32)
    got = lut.apply_lut(probe, params)
    assert np.abs(got[0, 0] - 0.5).max() * 255 < 0.5, got[0, 0]
    assert np.abs(got[0, 1] - 1.0).max() * 255 < 0.5, got[0, 1]
    assert np.abs(got[0, 2] - 1.0).max() * 255 < 0.5, "past the domain, clamped"


def test_a_gradient_comes_through_without_banding() -> None:
    """The reason trilinear is not optional.

    A ramp through a 17-node LUT crosses a node every sixteenth of the range.
    Nearest neighbour would hold the whole interval at one value and then jump,
    so the ramp would come out with seventeen distinct levels and a step of a
    dozen code values at each of them — which is exactly the contouring a sky
    shows. The nearest-neighbour numbers are computed here too, because "no
    banding" only means something next to what banding measures.
    """
    ramp = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    img = np.repeat(ramp[None, :, None], 3, axis=2).copy()
    size = 17
    grid = lut.identity_table(3, size).astype(np.float64)
    # Curved enough that the nodes are nowhere near collinear, and monotone all
    # the way to white so nothing is flat for a reason other than the lookup.
    table = grid ** 1.6 * 0.9 + grid * 0.1
    params = _params(table, size)
    out = lut.apply_lut(img, params)[0, :, 0]

    steps = np.diff(out)
    assert np.all(steps >= -1e-6), "a monotone LUT must not invert a ramp"
    # Largest jump between neighbouring samples, in code values out of 255. The
    # LUT's own slope reaches 1.5, so a sample is worth 1.5/1023 of the range;
    # anything much above that is interpolation error, not the curve.
    assert steps.max() * 255 < 0.6, f"largest step {steps.max() * 255:.3f}"
    levels = len(np.unique(np.rint(out * 255.0)))

    nearest_idx = np.rint(np.clip(img[..., 0], 0.0, 1.0) * (size - 1)).astype(int)
    nearest = table[nearest_idx * (1 + size + size * size), 0][0]
    assert len(np.unique(np.rint(nearest * 255.0))) == size, "the baseline holds"
    assert np.abs(np.diff(nearest)).max() * 255 > 10.0, "...and jumps between"
    assert levels > 240, f"only {levels} distinct levels out of 256"


def test_amount_blends_monotonically() -> None:
    img = _scene(3)
    size = 9
    grid = lut.identity_table(3, size).astype(np.float64)
    table = np.clip(grid ** 0.75 + np.array([0.06, 0.0, -0.04]), 0.0, 1.0)
    full = lut.apply_lut(img, _params(table, size, amount=100))
    prev = img
    for amount in (10, 25, 50, 75, 100):
        out = lut.apply_lut(img, _params(table, size, amount=amount))
        want = img + (full - img) * (amount / 100.0)
        assert np.abs(out - want).max() < 2e-3, amount
        # Every step moves further from the input and closer to the full look.
        assert np.abs(out - img).mean() > np.abs(prev - img).mean() - 1e-9
        assert np.abs(out - full).mean() <= np.abs(prev - full).mean() + 1e-9
        prev = out
    assert lut.apply_lut(img, _params(table, size, amount=0)) is img


def test_apply_is_tile_independent() -> None:
    """Cost is bounded by tiling the frame; the seam must not be visible, so a
    tall frame has to give the same answer as the band it is made of."""
    img = _scene(5, h=600, w=64)
    size = 13
    grid = lut.identity_table(3, size).astype(np.float64)
    params = _params(np.clip(grid ** 1.2 * 1.05, 0, 1), size)
    whole = lut.apply_lut(img, params)
    halves = np.concatenate([lut.apply_lut(img[:300].copy(), params),
                             lut.apply_lut(img[300:].copy(), params)])
    assert np.array_equal(whole, halves)


def test_apply_refuses_the_wrong_kind_of_array() -> None:
    params = _params(lut.identity_table(3, 5) * 0.5, 5)
    assert "float32" in _asserts(
        lambda: lut.apply_lut(np.zeros((4, 4, 3), np.uint8), params))
    assert "(h, w, 3)" in _asserts(
        lambda: lut.apply_lut(np.zeros((4, 4), np.float32), params))


# ----- colour match -------------------------------------------------------

def _shift(img: np.ndarray) -> np.ndarray:
    """A stand-in for the other camera body: a colour matrix, a per-channel gamma
    and a per-channel gain — the three things a different profile actually does."""
    matrix = np.array([[1.0, 0.07, -0.04],
                       [-0.05, 1.0, 0.06],
                       [0.03, -0.08, 1.0]], dtype=np.float32)
    out = np.clip(img @ matrix.T, 0.0, 1.0) ** np.array([0.88, 1.0, 1.18], np.float32)
    return np.clip(out * np.array([1.09, 1.0, 0.88], np.float32), 0.0, 1.0).astype(np.float32)


def test_a_fit_recovers_a_known_shift() -> None:
    ref = _scene(11)
    src = _shift(ref)
    params = lut.fit_from_reference(src, ref)
    out = lut.apply_lut(src, params)
    before = np.abs(src - ref).mean() * 255
    after = np.abs(out - ref).mean() * 255
    assert before > 8.0, f"the test shift is too small to measure: {before:.2f}"
    assert after < before * 0.35, f"{before:.2f} -> {after:.2f} code values"
    # And the per-channel means, which is what a cast looks like on a histogram.
    for c in range(3):
        d0 = abs(src[..., c].mean() - ref[..., c].mean()) * 255
        d1 = abs(out[..., c].mean() - ref[..., c].mean()) * 255
        assert d1 < max(d0 * 0.2, 0.5), f"channel {c}: {d0:.2f} -> {d1:.2f}"


def test_a_fit_transfers_to_another_frame_of_the_same_shift() -> None:
    """The point of the whole exercise: fitted on one pair, applied to the rest of
    the shoot. Fitting and then only ever applying to the fitting frame would
    prove nothing."""
    params = lut.fit_from_reference(_shift(_scene(11)), _scene(11))
    other = _scene(12)
    before = np.abs(_shift(other) - other).mean() * 255
    after = np.abs(lut.apply_lut(_shift(other), params) - other).mean() * 255
    assert after < before * 0.5, f"{before:.2f} -> {after:.2f} code values"


def test_a_fit_between_matching_frames_is_the_identity() -> None:
    ref = _scene(13)
    params = lut.fit_from_reference(ref, ref)
    table = lut.unpack_table(params["table"], lut.FIT_SIZE ** 3)
    dev = np.abs(table - lut.identity_table(3, lut.FIT_SIZE)).max()
    assert dev * 255 < 0.05, f"{dev * 255:.4f} code values off the identity"
    assert lut.is_neutral(params), "nothing to correct should cost nothing"


def test_a_fit_is_deterministic_and_ignores_frame_size() -> None:
    ref, src = _scene(14), _shift(_scene(14))
    first = lut.fit_from_reference(src, ref)
    assert first == lut.fit_from_reference(src, ref), "not reproducible"
    assert first == lut.fit_from_reference(src.copy(), ref.copy())
    # Statistics are taken at a fixed working size, so twice the pixels of the
    # same picture is the same fit to within resampling.
    big = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)
    big_ref = np.repeat(np.repeat(ref, 2, axis=0), 2, axis=1)
    second = lut.fit_from_reference(big, big_ref)
    a = lut.unpack_table(first["table"], lut.FIT_SIZE ** 3)
    b = lut.unpack_table(second["table"], lut.FIT_SIZE ** 3)
    assert np.abs(a - b).max() * 255 < 3.0, np.abs(a - b).max() * 255


def test_a_fit_round_trips_through_normalize_and_apply() -> None:
    import json

    ref, src = _scene(15), _shift(_scene(15))
    params = lut.fit_from_reference(src, ref)
    stored = json.loads(json.dumps(lut.normalize(params)))
    assert lut.normalize(stored) == stored, "a stored fit must be a fixed point"
    assert np.array_equal(lut.apply_lut(src, params),
                          lut.apply_lut(src, lut.normalize(stored)))
    # And it lives in an edit like any other look.
    assert not lut.is_neutral(stored)
    assert lut.table_key(stored) == lut.table_key(params)


def test_a_fit_bakes_faithfully() -> None:
    """The table has to hold the model it was fitted from. Checked against the
    model evaluated directly, which is the thing the table approximates."""
    ref, src = _scene(16), _shift(_scene(16))
    sv, rv = lut._stats_view(src), lut._stats_view(ref)
    matrix, mu_s, mu_r = lut._transport_matrix(sv, rv)
    maps = lut._marginal_maps(np.clip((sv - mu_s) @ matrix.T + mu_r, 0, 1), rv)
    params = lut.fit_from_reference(src, ref)

    rng = np.random.default_rng(9)
    probe = rng.random((200, 200, 3), dtype=np.float32)
    direct = lut._apply_model(probe.reshape(-1, 3).astype(np.float64),
                              matrix, mu_s, mu_r, maps).reshape(probe.shape)
    err = np.abs(lut.apply_lut(probe, params) - direct) * 255
    assert err.mean() < 0.5, f"mean {err.mean():.3f} code values"
    assert err.max() < 6.0, f"worst {err.max():.3f} code values"


def test_a_fit_is_a_smooth_map() -> None:
    """A fitted table that is not smooth bands, whatever the interpolation does,
    so the marginal maps are held monotone and their slope bounded."""
    ref, src = _scene(17), _shift(_scene(17))
    sv, rv = lut._stats_view(src), lut._stats_view(ref)
    matrix, mu_s, mu_r = lut._transport_matrix(sv, rv)
    maps = lut._marginal_maps(np.clip((sv - mu_s) @ matrix.T + mu_r, 0, 1), rv)
    slope = np.diff(maps, axis=0) * lut._HIST_BINS
    assert slope.min() >= -1e-9, "the map inverts somewhere"
    assert slope.max() < 4.0, f"slope up to {slope.max():.2f} will amplify noise"

    ramp = np.linspace(0.0, 1.0, 1024, dtype=np.float32)
    img = np.repeat(ramp[None, :, None], 3, axis=2).copy()
    out = lut.apply_lut(img, lut.fit_from_reference(src, ref))
    assert np.abs(np.diff(out, axis=1)).max() * 255 < 4.0


def test_a_fit_refuses_a_frame_with_nothing_in_it() -> None:
    flat = np.full((64, 64, 3), 0.5, dtype=np.float32)
    scene = _scene(18)
    assert "no colour variation" in _asserts(
        lambda: lut.fit_from_reference(flat, scene))
    assert "no colour variation" in _asserts(
        lambda: lut.fit_from_reference(scene, flat))
    assert "float32" in _asserts(
        lambda: lut.fit_from_reference(scene.astype(np.float64), scene))


def test_a_fit_of_different_scenes_stays_in_range() -> None:
    """The honest failure mode. Two unrelated frames still produce a usable
    table — monotone, in range, no invented values — it is just aimed at the
    wrong colour. That it does not blow up is worth checking, because `amount` is
    the only defence and it needs something sane to blend towards."""
    params = lut.fit_from_reference(_scene(19), _scene(20) ** 0.5 * 0.8)
    table = lut.unpack_table(params["table"], lut.FIT_SIZE ** 3)
    assert np.isfinite(table).all()
    assert table.min() >= 0.0 and table.max() <= 1.0
    out = lut.apply_lut(_scene(19), params)
    assert np.isfinite(out).all() and out.min() >= 0.0 and out.max() <= 1.0


# ----- fitting into the pipeline ------------------------------------------

def test_a_look_can_ride_along_with_an_edit() -> None:
    """The parameter dict has to survive the trip a stored look takes: JSON, a
    hash, and back."""
    import json

    params = lut.normalize(_params(lut.identity_table(3, 5) ** 0.9, 5))
    assert params is not None
    text = json.dumps({"lut": params}, sort_keys=True)
    assert lut.normalize(json.loads(text)["lut"]) == params
    # The stored size is worth knowing about: this is per-photo if a caller
    # copies it rather than keying on table_key().
    assert len(json.dumps(lut.normalize(
        lut.fit_from_reference(_shift(_scene(21)), _scene(21))))) < 60_000
    assert editing.edit_hash({"exposure": 0.3})   # the module still imports


def _main() -> None:
    import tempfile
    from pathlib import Path

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in fns:
            fn(Path(tmp)) if fn.__code__.co_argcount else fn()
            print(f"  ok  {name}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
