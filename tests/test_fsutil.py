"""Swapping a finished file in while something is reading the old one.

Only Windows refuses that rename, so the first test only proves something
there; CI runs the suite on Windows for it.
"""
import re
import threading
import time
from pathlib import Path

from pickapicka import fsutil

PKG = Path(__file__).resolve().parents[1] / "src" / "pickapicka"


def test_replace_waits_out_a_reader(tmp_path):
    dst = tmp_path / "thumb.jpg"
    dst.write_bytes(b"old" * 1000)
    opened = threading.Event()

    def reader():
        with open(dst, "rb") as fh:
            opened.set()
            time.sleep(0.2)          # a slow stream of the old file
            fh.read()

    t = threading.Thread(target=reader)
    t.start()
    opened.wait()
    new = tmp_path / ".thumb.jpg.tmp"
    new.write_bytes(b"new")
    fsutil.replace(new, dst)
    t.join()
    assert dst.read_bytes() == b"new" and not new.exists()


def test_no_bare_replace_in_the_package():
    bare = re.compile(r"\bos\.replace\(|\btmp\.replace\(")
    offenders = [
        f"{p.relative_to(PKG)}:{n}"
        for p in PKG.rglob("*.py") if p.name != "fsutil.py"
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if bare.search(line) and not line.lstrip().startswith("#")
    ]
    assert not offenders, f"use fsutil.replace instead: {offenders}"
