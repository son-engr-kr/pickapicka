"""Checks for the installed app's open tabs: counted, handed over, and the app
quitting once the last has gone (tabs.py, /api/tabs, macapp.py).

    uv run pytest tests/test_tabs.py

The Mac checks start a real server with AppKit's loop on its main thread, as
the app runs, in a process of its own with HOME pointed at a throwaway folder.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest
from websockets.sync.client import connect

from pickapicka import launch, tabs as tabs_mod

APP_JS = Path(launch.__file__).parent / "web" / "app.js"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# ----- counting -----------------------------------------------------------------

def test_the_launch_tab_gets_longer_to_arrive_than_a_reload() -> None:
    clock = Clock()
    tabs = tabs_mod.Tabs(clock)
    clock.now += tabs_mod.LAST_TAB_GRACE + 1
    assert not tabs.all_gone(), "the browser is still starting"
    clock.now += tabs_mod.FIRST_TAB_GRACE
    assert tabs.all_gone(), "no page ever came"


def test_a_reload_is_not_a_quit() -> None:
    clock = Clock()
    tabs = tabs_mod.Tabs(clock)
    tabs.opened()
    clock.now += 3600
    assert not tabs.all_gone()
    tabs.closed()
    clock.now += tabs_mod.LAST_TAB_GRACE - 1
    assert not tabs.all_gone()
    tabs.opened()                      # the reloaded page
    clock.now += tabs_mod.LAST_TAB_GRACE
    assert not tabs.all_gone()
    tabs.closed()
    clock.now += tabs_mod.LAST_TAB_GRACE
    assert tabs.all_gone()


def test_the_last_of_several_tabs_is_the_one_that_counts() -> None:
    clock = Clock()
    tabs = tabs_mod.Tabs(clock)
    tabs.opened()
    tabs.opened()
    tabs.closed()
    clock.now += tabs_mod.FIRST_TAB_GRACE
    assert tabs.count() == 1 and not tabs.all_gone()


def test_a_running_task_holds_the_quit_until_it_finishes() -> None:
    clock = Clock()
    tabs = tabs_mod.Tabs(clock)
    tabs.opened()
    tabs.closed()
    busy = {"left": 3}
    stopped = []

    def sleep(seconds: float) -> None:
        clock.now += seconds
        busy["left"] -= 1

    clock.now += tabs_mod.LAST_TAB_GRACE
    tabs_mod.watch(tabs, busy=lambda: busy["left"] > 0, stop=lambda: stopped.append(clock.now), sleep=sleep)
    assert stopped and busy["left"] == 0


# ----- the socket -----------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.02)


@contextmanager
def _serving():
    """A real server on a thread, as serve() runs one, and where its tabs connect."""
    import uvicorn

    from pickapicka.server import create_app
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=_free_port(),
                                           log_level="warning"))
    app.state.server = server
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait(lambda: server.started)
    try:
        yield app, f"ws://127.0.0.1:{server.config.port}/api/tabs"
    finally:
        server.should_exit = True
        thread.join(10)
        assert not thread.is_alive()


def test_each_open_page_is_counted_while_its_socket_is_open() -> None:
    with _serving() as (app, url):
        tabs = app.state.tabs
        with connect(url):
            _wait(lambda: tabs.count() == 1)
            with connect(url):
                _wait(lambda: tabs.count() == 2)
            _wait(lambda: tabs.count() == 1)
        _wait(lambda: tabs.count() == 0)


def test_the_tab_a_launch_opens_takes_over_from_the_open_ones() -> None:
    with _serving() as (app, url):
        with connect(url) as old, connect(url) as older:
            _wait(lambda: app.state.tabs.count() == 2)
            with connect(url + "?claim=true") as launched:
                assert json.loads(old.recv(timeout=5)) == {"type": "superseded"}
                assert json.loads(older.recv(timeout=5)) == {"type": "superseded"}
                # A tab opened by hand is one more tab, and takes nothing over.
                with connect(url):
                    _wait(lambda: app.state.tabs.count() == 4)
                    for ws in (old, older, launched):
                        with pytest.raises(TimeoutError):
                            ws.recv(timeout=0.3)


def test_the_installed_app_stops_once_its_last_tab_has_gone(monkeypatch) -> None:
    monkeypatch.setattr(tabs_mod, "LAST_TAB_GRACE", 0.3)
    monkeypatch.setattr(tabs_mod, "POLL", 0.05)
    with _serving() as (app, url):
        with connect(url):
            _wait(lambda: app.state.tabs.count() == 1)
            threading.Thread(target=app.state.watch_tabs, daemon=True).start()
            time.sleep(0.6)
            assert not app.state.server.should_exit, "a tab is open"
        _wait(lambda: app.state.server.should_exit)


# ----- the page ---------------------------------------------------------------------

def test_the_page_claims_the_app_with_the_query_the_app_opens_it_with() -> None:
    assert launch.show_url("http://127.0.0.1:8765") == "http://127.0.0.1:8765/?show"
    js = APP_JS.read_text(encoding="utf-8")
    assert f'const TAB_SHOW_QUERY = "{launch.SHOW_QUERY}";' in js
    assert f'["{launch.INTRO_QUERY}", TAB_SHOW_QUERY]' in js


# ----- the Mac app's loop -------------------------------------------------------------

def _start_app(tmp_path: Path) -> tuple[subprocess.Popen, str]:
    port = _free_port()
    env = {**os.environ, "HOME": str(tmp_path)}
    proc = subprocess.Popen(
        [sys.executable, "-c", "from pickapicka.cli import main; main()",
         "serve", "--as-app", "--port", str(port)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"http://127.0.0.1:{port}"
    end = time.monotonic() + 60
    while True:
        assert proc.poll() is None, proc.stdout.read()
        assert time.monotonic() < end, "the app never answered"
        try:
            with urllib.request.urlopen(f"{url}/api/state", timeout=1):
                return proc, url
        except OSError:
            time.sleep(0.2)


mac_only = pytest.mark.skipif(sys.platform != "darwin", reason="AppKit is macOS's")


@mac_only
def test_the_mac_app_ends_when_its_server_is_quit(tmp_path) -> None:
    proc, url = _start_app(tmp_path)
    req = urllib.request.Request(f"{url}/api/quit", data=json.dumps({}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        assert json.load(r) == {"quitting": True}
    assert proc.wait(timeout=30) == 0, proc.stdout.read()


@mac_only
def test_the_mac_app_ends_on_a_signal_inside_appkits_loop(tmp_path) -> None:
    proc, _url = _start_app(tmp_path)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 0, proc.stdout.read()
