"""Entry point for the bundled app: the macOS .app and the Windows installer.

When launched from Finder/Dock or the Start menu there are no CLI arguments, so
we default to `serve --open`. When run from the command line, all the regular
`pickapicka` subcommands still work. Opened as an app, it serves as one
(`serve --as-app`): it quits when its last tab closes, and on macOS it answers
the Dock (macapp.py).

Eagerly imports the package's heavy submodules so that PyInstaller's static
analysis pulls in their transitive native dependencies (opencv, onnxruntime,
insightface, scikit-learn, fastapi, etc.) into the bundle.
"""
from __future__ import annotations

import os
import sys

from pickapicka import fsutil, paths

LOG_NAME = "app.log"
BUNDLE_ID = "kr.son-engr.pickapicka"   # packaging/pickapicka.spec
LOG_MAX_BYTES = 2_000_000   # past this the last run's log is kept as app.log.1


def give_output_somewhere() -> None:
    """Point a missing stdout and stderr at a log file.

    A windowed Windows build (console=False) starts with sys.stdout and
    sys.stderr set to None, because there is no console for them. Anything
    that writes to them or asks them a question then crashes. uvicorn's log
    formatter calls sys.stdout.isatty(), so the app died at startup with
    "Unable to configure formatter 'default'". InsightFace's model download
    writes a tqdm progress bar to stderr and would have died on the first
    scoring. A file answers both, and is where to look when the app misbehaves
    with no console to show why.

    Appended to, so a second launch that only points the browser at the running
    one does not wipe that one's log; set aside once it grows past
    LOG_MAX_BYTES. On Windows the running one holds it open, so the rename
    then fails; that launch appends instead, and the next first launch sets it
    aside. Raised, it would have stopped the app from opening at all.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = paths.DATA_DIR / LOG_NAME
    if log.is_file() and log.stat().st_size > LOG_MAX_BYTES:
        try:
            fsutil.replace(log, log.with_name(LOG_NAME + ".1"))
        except PermissionError:
            pass
    out = open(log, "a", encoding="utf-8", buffering=1)   # line-buffered, so a crash still lands
    if sys.stdout is None:
        sys.stdout = out
    if sys.stderr is None:
        sys.stderr = out


def opened_by_launchservices() -> bool:
    """Whether macOS opened this process as the app: from the Finder, the Dock,
    Launchpad or Spotlight, or with `open`, as the updater starts it again
    (updater.py). LaunchServices puts the bundle's id in __CFBundleIdentifier
    for what it opens; a command typed in Terminal carries Terminal's."""
    return sys.platform == "darwin" and os.environ.get("__CFBundleIdentifier") == BUNDLE_ID


def opened_as_app() -> bool:
    """No arguments is a launch from the Start menu or the Finder. macOS also
    opens the app with arguments, when the updater asks it to."""
    return len(sys.argv) == 1 or opened_by_launchservices()


# Before the imports below, so nothing they do at import time meets a None.
give_output_somewhere()

# Before the imports below too, which take seconds (macapp.prepare says why).
if opened_by_launchservices():
    from pickapicka import macapp
    macapp.prepare()

# Static-analysis anchors: do NOT remove. These trigger PyInstaller to bundle
# the entire dependency graph. Lazy-imported in cli.py at runtime, but bundled
# here at build time. Absolute imports — PyInstaller runs this script as
# __main__ with no package context, so relative imports would fail.
from pickapicka import cli, cluster, db, editing, hdr, raw, scenes, scorer, server, userstate  # noqa: E402, F401
from pickapicka.scoring import blur, exposure, faces  # noqa: E402, F401
import rawpy  # noqa: E402, F401  # anchor the native libraw dependency into the bundle
# The command line's app built in the same process (link._local) goes through
# FastAPI's TestClient, imported only when a command runs.
from fastapi import testclient  # noqa: E402, F401


def run() -> None:
    as_app = opened_as_app()
    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "--open"])
    if as_app and sys.argv[1] == "serve" and "--as-app" not in sys.argv:
        sys.argv.append("--as-app")
    cli.main()


if __name__ == "__main__":
    run()
