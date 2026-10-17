"""The Mac app's event loop, so that macOS can reach the running app.

Opened again from the Finder, the Dock, Launchpad or Spotlight while it runs,
an app is not started a second time: macOS asks the running one to come
forward, through the AppKit event loop on its main thread. This app's main
thread ran the web server and nothing else, so macOS found nobody to ask, said
"The application "Pickapicka" is not open anymore" (LaunchServices error -600)
and opened nothing.

For the app as macOS opens it (app_entry.py), the main thread now runs AppKit
and the server runs on a thread of its own. Opening the app again points the
browser at it, Quit in the Dock or the app menu stops it as the page's Quit
button does, and the loop ends when the server stops, whatever stopped it.

An app with no window of its own counts as out of sight, and macOS naps one
(App Nap) once it has been out of sight for a while: its threads drop to
background priority and its timers slip. The browser tab is this app's window,
so it opts out for as long as it runs. Without that, its priority fell from 46
to 4 within 45 seconds of starting, and the quit after the last tab came up to
70 seconds late; 0.17.0, with no AppKit, was never napped.
"""
from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSEvent,
    NSEventTypeApplicationDefined,
    NSMenu,
    NSMenuItem,
    NSObject,
    NSTerminateLater,
)
from Foundation import NSActivityUserInitiatedAllowingIdleSystemSleep, NSProcessInfo
from PyObjCTools import AppHelper, MachSignals

if TYPE_CHECKING:
    import uvicorn

APP_NAME = "Pickapicka"


def prepare() -> None:
    """Connect to the window server at once, before the imports that take
    seconds. Until an app has, macOS cannot bring it forward, and a second
    click in that time would still be told it is not open."""
    NSApplication.sharedApplication()


class _Delegate(NSObject):
    """What AppKit asks of the app. `reopen` and `quit` are set by run()."""

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _app, _flag) -> bool:
        # Opening the browser waits on osascript; the loop need not.
        threading.Thread(target=self.reopen, name="reopen", daemon=True).start()
        return False   # the app has no window of its own to bring forward

    def applicationShouldTerminate_(self, _app) -> int:
        # Quit in the Dock or the menu, or the Mac logging out. The answer
        # waits for the server to stop (_end).
        self.quitting = True
        self.quit()
        return NSTerminateLater


def _menu() -> NSMenu:
    """The app menu, with Quit (Cmd-Q) in it, for when the app is in front."""
    bar = NSMenu.alloc().init()
    item = NSMenuItem.alloc().init()
    bar.addItem_(item)
    menu = NSMenu.alloc().initWithTitle_(APP_NAME)
    menu.addItemWithTitle_action_keyEquivalent_(f"Quit {APP_NAME}", "terminate:", "q")
    item.setSubmenu_(menu)
    return bar


def _end(delegate: _Delegate) -> None:
    """The server has stopped: end the loop, on the main thread."""
    app = NSApplication.sharedApplication()
    if delegate.quitting:
        # macOS asked, and waits on the answer. Yes ends the process here, in
        # AppKit, which lets a logout carry on; no would cancel the logout.
        app.replyToApplicationShouldTerminate_(True)
        return
    app.stop_(None)
    # stop: takes effect once the loop handles an event, so give it one.
    app.postEvent_atStart_(
        NSEvent.otherEventWithType_location_modifierFlags_timestamp_windowNumber_context_subtype_data1_data2_(
            NSEventTypeApplicationDefined, (0, 0), 0, 0, 0, None, 0, 0, 0),
        True)


def run(server: uvicorn.Server, *, reopen: Callable[[], None], quit: Callable[[], None]) -> None:
    """Serve on a thread and run AppKit on this one until the server stops.

    When macOS asked the app to quit, AppKit ends the process once the server
    has stopped, and this does not return. Otherwise it returns then, and
    raises what the server raised."""
    app = NSApplication.sharedApplication()
    # The bundle makes it a regular app already; run from a terminal, it is not.
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    app.setMainMenu_(_menu())
    delegate = _Delegate.alloc().init()
    delegate.reopen = reopen
    delegate.quit = quit
    delegate.quitting = False
    app.setDelegate_(delegate)
    # Ctrl-C, and kill, when started from a terminal. Python's own handlers
    # wait for Python code to run on this thread, which inside AppKit's loop
    # happens only when an event calls into it.
    for sig in (signal.SIGINT, signal.SIGTERM):
        MachSignals.signal(sig, lambda _signum: quit())
    # No App Nap (see above); the Mac may still sleep when idle.
    awake = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
        NSActivityUserInitiatedAllowingIdleSystemSleep, "Serving the Pickapicka page in the browser")
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="server") as pool:
        served = pool.submit(server.run)
        served.add_done_callback(lambda _f: AppHelper.callAfter(_end, delegate))
        app.run()
    NSProcessInfo.processInfo().endActivity_(awake)
    served.result()
