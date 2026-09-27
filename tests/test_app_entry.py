"""Checks for the bundled app's entry point.

    uv run python tests/test_app_entry.py

A windowed Windows build starts with sys.stdout and sys.stderr set to None,
and 0.6.0 shipped crashing on that at startup (uvicorn asks stdout isatty()).
Each check runs in a fresh interpreter with both set to None, the way the
Windows build starts, and with HOME pointed at a throwaway directory so the
log lands there.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _run_windowed(code: str, home: Path) -> subprocess.CompletedProcess:
    prelude = "import sys\nsys.stdout = None\nsys.stderr = None\n"
    env = {**os.environ, "HOME": str(home), "XDG_DATA_HOME": str(home / "data")}
    return subprocess.run([sys.executable, "-c", prelude + code], env=env,
                          capture_output=True, text=True, timeout=120)


def test_a_windowed_start_gets_a_log_instead_of_none() -> None:
    with tempfile.TemporaryDirectory() as d:
        out = _run_windowed(
            "from picture_classifier import app_entry, paths\n"
            "print('to stdout'); sys.stderr.write('to stderr\\n')\n"
            "assert sys.stdout.isatty() is False   # what uvicorn asks\n"
            "sys.__stdout__.write(str(paths.DATA_DIR / app_entry.LOG_NAME))\n",
            Path(d))
        assert out.returncode == 0, out.stderr
        log = Path(out.stdout.strip())
        assert log.is_file()
        text = log.read_text(encoding="utf-8")
        assert "to stdout" in text and "to stderr" in text


def test_uvicorn_logging_configures_once_output_exists() -> None:
    """The exact call that failed: uvicorn building its default formatter."""
    with tempfile.TemporaryDirectory() as d:
        out = _run_windowed(
            "from picture_classifier import app_entry\n"
            "import logging.config, uvicorn.config\n"
            "logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)\n",
            Path(d))
        assert out.returncode == 0, out.stderr


def test_a_console_start_is_left_alone() -> None:
    out = subprocess.run([sys.executable, "-c",
                          "import sys; before = sys.stdout\n"
                          "from picture_classifier import app_entry\n"
                          "assert sys.stdout is before"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr


def _main() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
