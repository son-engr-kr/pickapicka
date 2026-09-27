"""The editor's per-photo cache of decoded arrays.

What it has to get right is what the editor felt: opening a photo decoded it
twice (the preview and the hold-to-compare original are requested together),
and going back one photo decoded it again once a lens drag had filled the cache
with corrected copies.
"""
import threading
import time

import numpy as np
import pytest

from picture_classifier.server import PhotoArrays


def _arr(v: int = 0) -> np.ndarray:
    return np.full((2, 2), v, dtype=np.uint8)


def test_two_requests_for_one_decode_build_it_once():
    cache = PhotoArrays(photos=4, per_photo=4)
    builds = []

    def slow_decode():
        builds.append(1)
        time.sleep(0.2)
        return _arr(7)

    got = []
    threads = [threading.Thread(target=lambda: got.append(cache.get("a.jpg", "base", slow_decode)))
               for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(builds) == 1
    assert len(got) == 3 and all(g is got[0] for g in got)


def test_going_back_a_photo_keeps_its_decode():
    cache = PhotoArrays(photos=4, per_photo=4)
    decodes = []

    def decode(name):
        decodes.append(name)
        return _arr()

    cache.get("a", "base", lambda: decode("a"))
    for i in range(6):   # a lens drag on a: one corrected copy per optics setting
        cache.get("a", f"optics:{i}", _arr)
    for name in ("b", "c"):
        cache.get(name, "base", lambda n=name: decode(n))
        cache.get(name, "draft@1100", _arr)
    cache.get("a", "base", lambda: decode("a"))
    assert decodes == ["a", "b", "c"], decodes


def test_derived_variants_go_before_the_decode():
    cache = PhotoArrays(photos=4, per_photo=3)
    cache.get("a", "base", lambda: _arr(1))
    for i in range(5):
        cache.get("a", f"optics:{i}", lambda: _arr(2))
    rebuilt = []
    cache.get("a", "base", lambda: rebuilt.append(1) or _arr(1))
    cache.get("a", "optics:4", lambda: rebuilt.append(2) or _arr(2))
    assert rebuilt == []          # the decode and the newest copy survived
    cache.get("a", "optics:0", lambda: rebuilt.append(3) or _arr(2))
    assert rebuilt == [3]         # an old copy did not


def test_photos_go_least_recently_used_first():
    cache = PhotoArrays(photos=2, per_photo=2)
    cache.get("a", "base", _arr)
    cache.get("b", "base", _arr)
    cache.get("a", "base", _arr)       # a is now the most recent
    cache.get("c", "base", _arr)       # so b goes
    rebuilt = []
    cache.get("a", "base", lambda: rebuilt.append("a") or _arr())
    cache.get("b", "base", lambda: rebuilt.append("b") or _arr())
    assert rebuilt == ["b"]


def test_max_arrays_caps_the_whole_cache():
    cache = PhotoArrays(photos=2, per_photo=2, max_arrays=2)
    cache.get("a", "base", _arr)
    cache.get("a", "optics:x", _arr)
    cache.get("b", "base", _arr)
    total = sum(len(e) for e in cache._entries.values())
    assert total == 2


def test_a_failed_build_reaches_the_waiters_and_is_not_kept():
    cache = PhotoArrays(photos=4, per_photo=4)
    started = threading.Event()

    def broken():
        started.set()
        time.sleep(0.1)
        raise OSError("disk went away")

    errors = []

    def waiter():
        started.wait()
        try:
            cache.get("a", "base", broken)
        except OSError as exc:
            errors.append(str(exc))

    t = threading.Thread(target=waiter)
    t.start()
    with pytest.raises(OSError):
        cache.get("a", "base", broken)
    t.join()
    assert errors == ["disk went away"]
    assert np.array_equal(cache.get("a", "base", lambda: _arr(3)), _arr(3))


def test_a_build_that_straddles_clear_is_not_kept():
    cache = PhotoArrays(photos=4, per_photo=4)
    cache_cleared = threading.Event()

    def decode_old_pixels():
        cache_cleared.wait()
        return _arr(1)

    t = threading.Thread(target=lambda: cache.get("a", "base", decode_old_pixels))
    t.start()
    time.sleep(0.05)
    cache.clear()          # a re-score replaced the file meanwhile
    cache_cleared.set()
    t.join()
    assert np.array_equal(cache.get("a", "base", lambda: _arr(2)), _arr(2))
