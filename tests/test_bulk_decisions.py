"""Deciding many photos at once, and undoing it to what each one was.

    uv run pytest tests/test_bulk_decisions.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from pickapicka import db
from pickapicka.scorer import run_scoring
from pickapicka.server import (
    BulkDecidePayload, DecidePayload, DecisionEntry, OpenProjectPayload,
    RestoreDecisionsPayload, create_app,
)


def _route(app, path: str, method: str):
    return next(r.endpoint for r in app.routes
                if getattr(r, "path", None) == path and method in r.methods)


def _app(tmp_path: Path):
    photos, proj = tmp_path / "shoot", tmp_path / "trip"
    photos.mkdir()
    for i in range(3):
        Image.fromarray(np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(
            photos / f"a{i}.jpg")
    (proj / ".cache").mkdir(parents=True)
    run_scoring(photos, "", proj / "picks.json", with_faces=False, progress_cb=lambda *a: None,
                project_dir=proj, subject_classes=[], detect_faces=False)
    app = create_app()
    _route(app, "/api/project/open", "POST")(OpenProjectPayload(project_dir=str(proj)))
    deadline = time.monotonic() + 30
    while _route(app, "/api/state", "GET")()["opening"]["running"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return app, proj


def test_undoing_a_bulk_decision_puts_back_what_each_photo_had(tmp_path) -> None:
    app, proj = _app(tmp_path)
    decide = _route(app, "/api/decide", "POST")
    first = decide(DecidePayload(rel_path="a0.jpg", decision="reject"))
    decide(DecidePayload(rel_path="a1.jpg", decision="pick"))
    before = {p["rel_path"]: (p["decision"], p.get("decided_at"))
              for p in db.load(proj / "picks.json")["photos"]}

    out = _route(app, "/api/decide/bulk", "POST")(
        BulkDecidePayload(rel_paths=["a0.jpg", "a1.jpg", "a2.jpg"], decision=None))
    assert out == {"updated": 3, "decided_at": None}
    assert {p["decision"] for p in db.load(proj / "picks.json")["photos"]} == {None}

    _route(app, "/api/decide/restore", "POST")(RestoreDecisionsPayload(decisions=[
        DecisionEntry(rel_path=r, decision=d, decided_at=t) for r, (d, t) in before.items()]))
    after = {p["rel_path"]: (p["decision"], p.get("decided_at"))
             for p in db.load(proj / "picks.json")["photos"]}
    assert after == before
    assert after["a0.jpg"] == ("reject", first["decided_at"])
    assert after["a2.jpg"] == (None, None)


def test_a_bulk_decision_says_when_it_was_made(tmp_path) -> None:
    app, proj = _app(tmp_path)
    out = _route(app, "/api/decide/bulk", "POST")(
        BulkDecidePayload(rel_paths=["a0.jpg", "a2.jpg"], decision="reject"))
    stamps = {p["decided_at"] for p in db.load(proj / "picks.json")["photos"] if p["decision"] == "reject"}
    assert stamps == {out["decided_at"]} and out["decided_at"]


def test_a_restore_naming_an_unknown_photo_changes_nothing(tmp_path) -> None:
    app, proj = _app(tmp_path)
    with pytest.raises(HTTPException):
        _route(app, "/api/decide/restore", "POST")(RestoreDecisionsPayload(decisions=[
            DecisionEntry(rel_path="a0.jpg", decision="pick", decided_at=None),
            DecisionEntry(rel_path="gone.jpg", decision="pick", decided_at=None)]))
    assert {p["decision"] for p in db.load(proj / "picks.json")["photos"]} == {None}
