"""Checks for what the app does as it opens: the intro address and the chime.

    uv run python tests/test_launch.py

Nothing here plays a sound: each platform's player is replaced by a recorder.
"""
from __future__ import annotations

import sys
import threading
import types
import wave
from pathlib import Path

import pytest

from pickapicka import launch

INDEX_HTML = Path(launch.__file__).parent / "web" / "index.html"


# ----- the sound file -------------------------------------------------------

def test_the_chime_ships_and_stays_short_and_quiet() -> None:
    """A chime someone swapped for a long or loud file would hold up every launch."""
    with wave.open(str(launch.SOUND)) as w:
        seconds = w.getnframes() / w.getframerate()
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        frames = w.readframes(w.getnframes())
    assert 1.0 < seconds < 3.0, seconds
    peak = max(abs(int.from_bytes(frames[i:i + 2], "little", signed=True))
               for i in range(0, len(frames), 2))
    assert peak / 32767 <= 0.51, f"peak {peak / 32767:.2f}, where make_launch_sound.py writes 0.5"


# ----- the intro address ----------------------------------------------------

def test_the_page_looks_for_the_query_the_app_opens_it_with() -> None:
    """The two halves of the intro live in different languages; this is what
    ties them together."""
    assert launch.intro_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765/?launch"
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert f'.has("{launch.INTRO_QUERY}")' in html


# The server module is imported inside the two tests that need it: importing
# FastAPI before pytest.main starts is what makes pytest warn it cannot rewrite
# anyio's asserts.

def test_opening_the_browser_on_a_launch_opens_the_intro(monkeypatch) -> None:
    from pickapicka import server
    opened: list[str] = []
    done = threading.Event()

    def fake_open(url: str) -> None:
        opened.append(url)
        done.set()

    class FakeServer:
        def __init__(self, config) -> None:
            self.config = config

        def run(self) -> None:   # serve() would block here for the app's lifetime
            pass

    import time
    import uvicorn
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", fake_open)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(server, "_is_pickapicka_running", lambda host, port: False)
    monkeypatch.setattr(server, "_pick_free_port", lambda host, port: port)

    server.serve(None, "127.0.0.1", 8765, open_browser=True)

    assert done.wait(5), "the browser was never opened"
    assert opened == ["http://127.0.0.1:8765/?launch"]


def test_the_sound_route_plays_the_chime(monkeypatch) -> None:
    from pickapicka import server
    calls: list[bool] = []
    monkeypatch.setattr(launch, "play_sound", lambda: calls.append(True) or True)
    app = server.create_app(None)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/api/launch-sound")
    assert route.methods == {"POST"}
    assert route.endpoint() == {"played": True}
    assert calls == [True]


# ----- each platform's player ---------------------------------------------

class _NowThread:
    """threading.Thread, but the target runs on start()."""

    def __init__(self, target, args=(), kwargs=None, daemon=None) -> None:
        self.run = lambda: target(*args, **(kwargs or {}))

    def start(self) -> None:
        self.run()


def test_macos_plays_it_with_afplay(monkeypatch) -> None:
    ran: list[tuple] = []
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(launch.threading, "Thread", _NowThread)
    monkeypatch.setattr(launch.subprocess, "run", lambda cmd, **kw: ran.append((cmd, kw)))

    assert launch.can_play() and launch.play_sound() is True
    assert ran == [(["afplay", str(launch.SOUND)], {"check": True})]


def test_windows_plays_it_with_winsound_and_does_not_wait(monkeypatch) -> None:
    played: list[tuple] = []
    fake = types.SimpleNamespace(SND_FILENAME=0x20000, SND_ASYNC=0x1,
                                 PlaySound=lambda path, flags: played.append((path, flags)))
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winsound", fake)

    assert launch.can_play() and launch.play_sound() is True
    assert played == [(str(launch.SOUND), 0x20000 | 0x1)]


def test_linux_stays_silent(monkeypatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("nothing should run"))

    assert not launch.can_play()
    assert launch.play_sound() is False


# Handed to pytest, like test_redeye: most of these take monkeypatch, which a
# loop over globals() cannot supply.
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
