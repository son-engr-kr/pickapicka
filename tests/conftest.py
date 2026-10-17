"""Shared test setup.

Every test gets an application-data directory of its own. userstate takes its
paths at import time, and opening a project files it under them (recents,
known projects, their ids); without this, a test run filed its temporary
projects into the real state.json.

And a Claude Code folder of its own: the skill the app writes there (skill.py)
would otherwise land in the real ~/.claude, as it once did from a test.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pickapicka import skill, userstate


@pytest.fixture(autouse=True)
def _own_app_data(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("app-data")
    monkeypatch.setattr(userstate, "CONFIG_DIR", home)
    monkeypatch.setattr(userstate, "STATE_FILE", home / "state.json")
    monkeypatch.setattr(userstate, "LUT_DIR", home / "luts")
    monkeypatch.setattr(skill, "CLAUDE_DIR", home / ".claude")
    monkeypatch.setattr(skill, "DEFAULT_DIR", home / ".claude" / "skills" / "pickapicka")


@pytest.fixture
def unplugged() -> tuple[Path, str]:
    """Where a drive that is not connected would be, and its name."""
    if os.name == "nt":
        letter = next(f"{c}:" for c in "ZYXWVUTS" if not Path(f"{c}:\\").exists())
        return Path(f"{letter}\\"), letter
    return Path("/Volumes/pickapicka-unplugged"), "pickapicka-unplugged"
