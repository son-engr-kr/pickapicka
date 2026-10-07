"""Shared test setup.

Every test gets an application-data directory of its own. userstate takes its
paths at import time, and opening a project files it under them (recents,
known projects, their ids); without this, a test run filed its temporary
projects into the real state.json.
"""
from __future__ import annotations

import pytest

from pickapicka import userstate


@pytest.fixture(autouse=True)
def _own_app_data(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("app-data")
    monkeypatch.setattr(userstate, "CONFIG_DIR", home)
    monkeypatch.setattr(userstate, "STATE_FILE", home / "state.json")
    monkeypatch.setattr(userstate, "LUT_DIR", home / "luts")
