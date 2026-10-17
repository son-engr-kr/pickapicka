"""How many of the app's pages are open, so the installed app can quit once the
last one is closed.

The page is only a view of the server, so a server left running after its last
tab closed was an app nobody could see, still holding the port. Each page keeps
a WebSocket open to the server while it is open (app.js, connectTab), and the
server counts them here. A reload closes the socket and opens a new one a
moment later, so the app waits LAST_TAB_GRACE before it counts the tabs as
gone. At launch the browser can take a while to start and load the page, so the
first tab gets FIRST_TAB_GRACE.

Only the app started as an app quits this way (serve(as_app=True)). A server
started from a terminal, or by the tests, runs until it is stopped.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable

FIRST_TAB_GRACE = 60.0   # seconds from start for the page the launch opened
LAST_TAB_GRACE = 15.0    # seconds after the last tab closed, so a reload is not a quit
POLL = 1.0               # how often the watcher looks


class Tabs:
    """The open pages, counted from the sockets they hold."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._open = 0
        self._seen = False
        self._empty_since = clock()   # start counts as the moment there were none

    def opened(self) -> None:
        with self._lock:
            self._open += 1
            self._seen = True

    def closed(self) -> None:
        with self._lock:
            assert self._open > 0, "a tab closed that was never counted open"
            self._open -= 1
            if self._open == 0:
                self._empty_since = self._clock()

    def count(self) -> int:
        with self._lock:
            return self._open

    def all_gone(self) -> bool:
        """No tab, and for long enough that none is coming back."""
        with self._lock:
            if self._open:
                return False
            grace = LAST_TAB_GRACE if self._seen else FIRST_TAB_GRACE
            return self._clock() - self._empty_since >= grace


def watch(tabs: Tabs, busy: Callable[[], bool], stop: Callable[[], None],
          sleep: Callable[[float], None] = time.sleep) -> None:
    """Stop the app once every tab has gone. Not while something is still
    running (an export, scoring): that waits until it finishes, and the app
    quits then if no tab has come back."""
    while not (tabs.all_gone() and not busy()):
        sleep(POLL)
    stop()
