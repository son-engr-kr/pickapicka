"""Writes from many threads at once: the server answers requests on a thread
pool, so two can save state.json, or picks.json, at the same moment.

    uv run pytest tests/test_state_writes.py

With one temp file name for every writer, the second rename found the first
had already moved the file away (FileNotFoundError on state.json.tmp, seen in
the app), and two changes to state.json read before either was written lost
one of them.
"""
from __future__ import annotations

import json
import threading

from pickapicka import db, userstate

THREADS, EACH = 8, 40


def _hammer(work) -> list[BaseException]:
    errors: list[BaseException] = []
    start = threading.Barrier(THREADS)

    def run(t: int) -> None:
        start.wait()
        for i in range(EACH):
            try:
                work(t, i)
            except BaseException as exc:      # collected, and failed on below
                errors.append(exc)
                return

    threads = [threading.Thread(target=run, args=(t,)) for t in range(THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


def test_state_json_takes_every_change_from_every_thread() -> None:
    errors = _hammer(lambda t, i: userstate.set_view(f"/p{t}", {"page": i}))
    assert not errors, errors[0]
    views = json.loads(userstate.STATE_FILE.read_text(encoding="utf-8"))["views"]
    assert {k: v["page"] for k, v in views.items()} == {f"/p{t}": EACH - 1 for t in range(THREADS)}
    assert not list(userstate.CONFIG_DIR.glob("*.tmp")), "a temp file was left behind"


def test_picks_json_survives_writers_at_once(tmp_path) -> None:
    path = tmp_path / "picks.json"
    data = db.init_db(tmp_path, "")
    errors = _hammer(lambda t, i: db.save(path, {**data, "scored_at": f"{t}-{i}"}))
    assert not errors, errors[0]
    assert db.load(path)["version"] == db.DB_VERSION
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))
