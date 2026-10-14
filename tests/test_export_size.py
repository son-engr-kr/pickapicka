"""The export's file size limit, Lightroom's Limit File Size:

    uv run pytest tests/test_export_size.py

Only the JPEG quality moves: the highest that fits is found, a photo already
under the limit is written as without one, a photo no quality can hold is
left out and named, and an original over the limit is written fresh rather
than copied.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from pickapicka import exporting
from pickapicka.server import ExportPayload, ExportSettings
from test_versions import A0, A1, _app, _export, _route, _wait

NOISE = np.random.default_rng(0).integers(0, 256, (300, 400, 3), dtype=np.uint8)


def _size(q: int) -> int:
    buf = io.BytesIO()
    exporting.write(NOISE, buf, fmt="jpeg", quality=q, exif=None, icc=None)
    return len(buf.getvalue())


def test_the_highest_quality_that_fits_is_found() -> None:
    limit = (_size(60) + _size(61)) // 2           # between two steps
    buf = io.BytesIO()
    q = exporting.write_within(NOISE, buf, quality=95, max_bytes=limit, exif=None, icc=None)
    assert len(buf.getvalue()) <= limit
    best = max(k for k in range(1, 96) if _size(k) <= limit)
    assert best - 2 <= q <= best, (q, best)
    assert Image.open(io.BytesIO(buf.getvalue())).size == (400, 300), "never resized"


def test_a_photo_under_the_limit_is_written_as_without_one() -> None:
    buf = io.BytesIO()
    assert exporting.write_within(NOISE, buf, quality=90, max_bytes=_size(90),
                                  exif=None, icc=None) == 90
    assert len(buf.getvalue()) == _size(90)


def test_what_no_quality_can_hold_is_not_written(tmp_path) -> None:
    dst = tmp_path / "a.jpg"
    with pytest.raises(exporting.TooLarge):
        exporting.write_within(NOISE, dst, quality=95, max_bytes=_size(1) - 1, exif=None, icc=None)
    assert not dst.exists()


def test_an_original_over_the_limit_is_not_copied(tmp_path) -> None:
    src = tmp_path / "a.jpg"
    Image.fromarray(NOISE).save(src, quality=95)
    size = src.stat().st_size
    full = dict(format="jpeg", long_edge=None, metadata="all")
    assert exporting.can_copy(src, needs_render=False,
                              settings=SimpleNamespace(**full, max_mb=size / exporting.MB))
    assert exporting.can_copy(src, needs_render=False, settings=SimpleNamespace(**full, max_mb=None))
    assert not exporting.can_copy(src, needs_render=False,
                                  settings=SimpleNamespace(**full, max_mb=(size - 1) / exporting.MB))


def test_the_limit_is_on_by_default_at_12_mb() -> None:
    assert ExportSettings().max_mb == 12
    assert ExportSettings(**{"quality": 90}).max_mb == 12, "settings saved before it get it"
    assert ExportSettings(max_mb=None).max_mb is None
    with pytest.raises(ValueError):
        ExportSettings(max_mb=0)


def test_an_export_names_what_it_left_out(tmp_path) -> None:
    app, _ = _app(tmp_path, {})
    tiny = ExportSettings(metadata="none", max_mb=100 / exporting.MB)      # 100 bytes
    r = _export_with(app, tmp_path / "out", tiny)
    assert r["copied"] == 0 and sorted(r["too_large"]) == [A0, A1]
    assert not [f for f in (tmp_path / "out").rglob("*") if f.is_file()]
    r = _export(app, tmp_path / "ok", "folder", versions=False)
    assert r["copied"] == 2 and r["too_large"] == [] and r["lowered"] == 0


def _export_with(app, out, settings: ExportSettings) -> dict:
    _route(app, "/api/export/picks", "POST")(ExportPayload(
        target_dir=str(out), mode="folder", which="picks", settings=settings))
    return _wait(_route(app, "/api/export/status", "GET")())["result"]
