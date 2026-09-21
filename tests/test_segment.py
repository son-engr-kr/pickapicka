"""Unit checks for automatic-mask segmentation. Runnable with pytest or directly:

    uv run python tests/test_segment.py

The tests that need the ONNX model skip themselves when it has not been
downloaded yet, the same way `test_objects.py` treats YOLOX. The ones that do
not — the class table, the group vocabulary, letterbox geometry and the softmax —
run everywhere, and between them they cover the parts most likely to break
silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from picture_classifier import segment


def _sample(h: int = 240, w: int = 360) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


needs_model = pytest.mark.skipif(
    not segment.is_model_ready(),
    reason="segmentation model not downloaded",
)


# ----- the class table ----------------------------------------------------

def test_class_table_matches_the_model() -> None:
    """Six classes in a fixed order. The order is the model's, not ours, so it
    must never be sorted or reordered for convenience."""
    assert len(segment.CLASSES) == 6
    assert segment.CLASSES[0] == "background"
    assert len(set(segment.CLASSES)) == 6


def test_groups_only_name_real_classes() -> None:
    for group, members in segment.CLASS_GROUPS.items():
        assert members, f"{group} is empty"
        for c in members:
            assert c in segment.CLASSES, f"{group} names {c!r}, which is not a class"


def test_subject_and_background_partition_the_frame() -> None:
    """Every class belongs to exactly one of the two, so the two alphas will sum
    to 1 and an inverted subject mask is the background mask."""
    subject = set(segment.CLASS_GROUPS["subject"])
    background = set(segment.CLASS_GROUPS["background"])
    assert not (subject & background)
    assert subject | background == set(segment.CLASSES)


def test_unknown_group_asserts() -> None:
    with pytest.raises(AssertionError):
        segment.class_mask(_sample(), "hat")


# ----- letterbox geometry -------------------------------------------------

def test_letterbox_keeps_proportions() -> None:
    """A stretched person is a shape the model never saw in training, so the
    frame goes in with its aspect intact and padding around it."""
    canvas, (ox, oy, nw, nh) = segment._letterbox(
        np.zeros((200, 400, 3), dtype=np.float32))
    assert canvas.shape == (*segment.INPUT_SIZE, 3)
    assert abs((nw / nh) - 2.0) < 0.02, f"aspect changed: {nw}x{nh}"
    assert nw == segment.INPUT_SIZE[1], "the long edge should fill the canvas"
    assert oy > 0 and ox == 0, "a landscape frame should be padded top and bottom"


def test_letterbox_fills_a_square() -> None:
    canvas, (ox, oy, nw, nh) = segment._letterbox(
        np.zeros((300, 300, 3), dtype=np.float32))
    assert (ox, oy) == (0, 0)
    assert (nw, nh) == segment.INPUT_SIZE


def test_letterbox_survives_an_extreme_aspect() -> None:
    """A panorama must not round its short edge away to zero."""
    _, (_, _, nw, nh) = segment._letterbox(np.zeros((8, 4000, 3), dtype=np.float32))
    assert nh >= 1 and nw >= 1


# ----- softmax ------------------------------------------------------------

def test_softmax_normalizes() -> None:
    """The exported graph stops at the logits, so this module has to finish the
    job — summing raw logits would give an alpha outside [0,1]."""
    rng = np.random.default_rng(1)
    logits = rng.normal(0.0, 4.0, (7, 11, 6)).astype(np.float32)
    probs = segment._softmax(logits)
    assert np.abs(probs.sum(axis=-1) - 1.0).max() < 1e-5
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    # Order is preserved: the largest logit stays the largest probability.
    assert np.array_equal(logits.argmax(axis=-1), probs.argmax(axis=-1))


def test_softmax_survives_large_logits() -> None:
    """Subtracting the row max is what stops `exp` overflowing; prove it."""
    logits = np.full((2, 2, 6), 1000.0, dtype=np.float32)
    logits[..., 3] = 1200.0
    probs = segment._softmax(logits)
    assert np.isfinite(probs).all()
    assert np.abs(probs.sum(axis=-1) - 1.0).max() < 1e-5
    assert probs[0, 0, 3] > 0.99


# ----- inference ----------------------------------------------------------

@needs_model
def test_probabilities_shape_and_normalization() -> None:
    img = _sample()
    probs = segment.probabilities(img)
    assert probs.shape == (*img.shape[:2], len(segment.CLASSES))
    assert probs.dtype == np.float32
    assert np.abs(probs.sum(axis=2) - 1.0).max() < 1e-3


@needs_model
def test_class_mask_alpha_contract() -> None:
    """The alpha has to satisfy exactly what editing._apply_masks expects, or it
    cannot be used as a mask at all."""
    img = _sample()
    for group in segment.CLASS_GROUPS:
        alpha = segment.class_mask(img, group)
        assert alpha.dtype == np.float32
        assert alpha.shape == img.shape[:2]
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0


@needs_model
def test_subject_and_background_alphas_sum_to_one() -> None:
    img = _sample()
    total = segment.class_mask(img, "subject") + segment.class_mask(img, "background")
    assert np.abs(total - 1.0).max() < 1e-3


@needs_model
def test_resolution_independent() -> None:
    """The network always sees a 256x256 letterbox of the whole frame, so the
    alpha field is the same whatever size the caller renders at. This is the
    property that lets an automatic mask be trusted at 1:1 while grading."""
    import cv2
    big = _sample(600, 900)
    # cv2.resize takes (width, height), so this is half of 900x600.
    small = cv2.resize(big, (450, 300), interpolation=cv2.INTER_AREA)
    a_big = cv2.resize(segment.class_mask(big, "subject"), (450, 300),
                       interpolation=cv2.INTER_AREA)
    a_small = segment.class_mask(small, "subject")
    # Not byte-exact: the two inputs differ, having been resampled to 256
    # from different starting sizes. The field has to agree, not the bits.
    assert np.abs(a_big - a_small).mean() < 0.02


@needs_model
def test_deterministic() -> None:
    img = _sample()
    assert np.array_equal(segment.class_mask(img, "skin"),
                          segment.class_mask(img, "skin"))


@needs_model
def test_cache_returns_the_same_field() -> None:
    img = _sample()
    once = segment.probabilities_cached(img, "photo-a")
    twice = segment.probabilities_cached(img, "photo-a")
    assert np.array_equal(once, twice)
    assert once is twice, "a cache hit should not recompute"


@needs_model
def test_float_and_uint8_inputs_agree() -> None:
    img = _sample()
    from_u8 = segment.class_mask(img, "subject")
    from_float = segment.class_mask(img.astype(np.float32) / 255.0, "subject")
    assert np.abs(from_u8 - from_float).max() < 1e-3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
