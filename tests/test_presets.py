"""Unit checks for the built-in preset library. Runnable with pytest or directly:

    uv run python tests/test_presets.py

The point of these is not that the numbers are tasteful — that is a judgement —
but that every preset is a *valid, non-neutral, renderable* edit and that the
groups do the thing their name promises. A preset whose values all normalize
away still appears in the picker and then does nothing when chosen, which is the
failure mode worth a test.
"""
from __future__ import annotations

import numpy as np

from picture_classifier import editing, film as film_mod, presets

# Every group the library is meant to ship, so deleting one is a failing test
# rather than a quiet hole in the picker.
GROUPS = {"Portrait", "Look", "Mono", "Scene", "Fix", "Car"}


def _frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(30, 225, size=(72, 96, 3), dtype=np.uint8)


def _sallow() -> np.ndarray:
    """A yellow-cast frame: a white wall on the left, skin on the right. Both
    carry the same cast, which is what a photo under one bad light looks like."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = (232, 226, 196)   # what should be a white wall
    img[:, 32:] = (206, 168, 124)   # skin, gone sallow under it
    return img


def _ids() -> list[str]:
    return [p["id"] for p in presets.BUILTIN_PRESETS]


def _by_id(preset_id: str) -> dict:
    return presets.BUILTIN_PRESETS[_ids().index(preset_id)]


def test_ids_are_unique() -> None:
    ids = _ids()
    assert len(ids) == len(set(ids)), "two presets share an id"
    assert all(presets.is_builtin(i) for i in ids)
    assert not presets.is_builtin("nothing-by-this-name")


def test_every_preset_is_described() -> None:
    for p in presets.BUILTIN_PRESETS:
        assert p["name"] and p["hint"], f"{p['id']} has nothing to show the user"
        assert p["group"] in GROUPS, f"{p['id']} is in an unknown group"
    assert {p["group"] for p in presets.BUILTIN_PRESETS} == GROUPS


def test_api_shape_matches_the_stored_one() -> None:
    listed = presets.list_builtins()
    assert len(listed) == len(presets.BUILTIN_PRESETS)
    for out, src in zip(listed, presets.BUILTIN_PRESETS):
        assert out["builtin"] is True
        assert out["id"] == src["id"] and out["edit"] == src["edit"]


def test_no_preset_normalizes_away() -> None:
    # A preset that survives normalize() as a neutral edit is a dead entry in
    # the picker: it applies, reports nothing changed, and looks broken.
    for p in presets.BUILTIN_PRESETS:
        assert not editing.is_neutral(editing.normalize(p["edit"])), \
            f"{p['id']} normalizes to a neutral edit"


def test_every_preset_renders() -> None:
    img = _frame()
    # One alpha for every automatic mask any preset asks for: `render` demands
    # the field from its caller rather than reaching for the model itself.
    auto = {g: np.ones(img.shape[:2], dtype=np.float32)
            for p in presets.BUILTIN_PRESETS
            for m in editing.normalize(p["edit"])["masks"] if m["type"] == "auto"
            for g in [m["group"]]}
    for p in presets.BUILTIN_PRESETS:
        out = editing.render(img, p["edit"], auto=auto)
        assert out.shape == img.shape and out.dtype == np.uint8, p["id"]


def test_portrait_presets_take_the_yellow_out() -> None:
    img = _sallow()
    wall_before = img[32, 16].astype(int)
    skin_before = img[32, 48].astype(int)
    for pid in ("skin-bright", "skin-airy", "skin-tungsten"):
        out = editing.render(img, _by_id(pid)["edit"])
        wall, skin = out[32, 16].astype(int), out[32, 48].astype(int)
        # Yellow is red and green over blue, so the cast is the red-blue gap.
        assert wall[0] - wall[2] < wall_before[0] - wall_before[2], pid
        assert skin[0] - skin[2] < skin_before[0] - skin_before[2], pid
        # ...and "bright" has to mean brighter, not merely cooler.
        assert skin.mean() > skin_before.mean(), pid


def test_mono_presets_are_actually_grey() -> None:
    img = _frame()
    for pid in ("mono-classic", "mono-highkey", "mono-noir"):
        out = editing.render(img, _by_id(pid)["edit"]).astype(int)
        assert int(out.max(axis=2).mean() - out.min(axis=2).mean()) <= 1, pid


def test_sepia_is_toned_by_the_film_stage() -> None:
    # The one that would silently stop working if the stock were renamed, or if
    # film ever moved to before saturation: everything else that could tint it
    # runs first and would be desaturated away.
    assert _by_id("mono-sepia")["edit"]["film"] is not None, "stock not found"
    out = editing.render(np.full((48, 48, 3), 128, np.uint8),
                         _by_id("mono-sepia")["edit"])
    px = out[24, 24].astype(int)
    assert px[0] > px[2] + 4, f"sepia came out neutral: {px}"


def test_the_stock_the_sepia_preset_names_exists() -> None:
    assert film_mod.stock("Warm portrait") is not None


def test_local_presets_carry_a_usable_mask() -> None:
    for pid in ("skin-local", "scene-sky", "car-speed", "car-bokeh", "car-plate"):
        masks = editing.normalize(_by_id(pid)["edit"])["masks"]
        assert len(masks) == 1, f"{pid} lost its mask in normalize()"
        assert editing.mask_is_active(masks[0]), f"{pid}'s mask does nothing"


# Last in the file, and it has to stay last: it collects the test functions out
# of globals(), so anything defined below it would not exist yet and would be
# silently skipped.
def _main() -> None:
    import re
    from pathlib import Path
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    declared = len(re.findall(r"^def test_", Path(__file__).read_text(encoding="utf-8"), re.M))
    assert len(fns) == declared, (
        f"collected {len(fns)} of {declared} tests — the runner has to be the "
        f"last thing in the file")
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
