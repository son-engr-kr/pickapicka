"""Entry point for the bundled app: the macOS .app and the Windows installer.

When launched from Finder/Dock or the Start menu there are no CLI arguments, so
we default to `serve --open`. When run from the command line, all the regular
`pickapicka` subcommands still work.

Eagerly imports the package's heavy submodules so that PyInstaller's static
analysis pulls in their transitive native dependencies (opencv, onnxruntime,
insightface, scikit-learn, fastapi, etc.) into the bundle.
"""
from __future__ import annotations

import sys

from pickapicka import paths

LOG_NAME = "app.log"
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
    LOG_MAX_BYTES.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    log = paths.DATA_DIR / LOG_NAME
    if log.is_file() and log.stat().st_size > LOG_MAX_BYTES:
        log.replace(log.with_name(LOG_NAME + ".1"))
    out = open(log, "a", encoding="utf-8", buffering=1)   # line-buffered, so a crash still lands
    if sys.stdout is None:
        sys.stdout = out
    if sys.stderr is None:
        sys.stderr = out


# Before the imports below, so nothing they do at import time meets a None.
give_output_somewhere()

# Static-analysis anchors: do NOT remove. These trigger PyInstaller to bundle
# the entire dependency graph. Lazy-imported in cli.py at runtime, but bundled
# here at build time. Absolute imports — PyInstaller runs this script as
# __main__ with no package context, so relative imports would fail.
from pickapicka import cli, cluster, db, editing, hdr, raw, scenes, scorer, server, userstate  # noqa: E402, F401
from pickapicka.scoring import blur, exposure, faces  # noqa: E402, F401
import rawpy  # noqa: E402, F401  # anchor the native libraw dependency into the bundle


def run() -> None:
    if len(sys.argv) == 1:
        sys.argv.extend(["serve", "--open"])
    cli.main()


if __name__ == "__main__":
    run()
