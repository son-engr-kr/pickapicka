"""User-level state in `~/.picture-classifier/state.json`: recents, last db, and
app-global edit presets."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".picture-classifier"
STATE_FILE = CONFIG_DIR / "state.json"
MAX_RECENTS = 10


def _load() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"recents": [], "last_db_path": None}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"recents": [], "last_db_path": None}


def _save(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_FILE)


def get_recents() -> list[dict[str, Any]]:
    return _load().get("recents", [])


def get_last_db_path() -> Path | None:
    p = _load().get("last_db_path")
    return Path(p) if p else None


def remember_open(
    db_path: Path,
    photo_dir: Path,
    jpeg_subdir: str,
    *,
    kind: str = "legacy",
    project_dir: Path | None = None,
) -> None:
    data = _load()
    entry: dict[str, Any] = {
        "kind": kind,
        "db_path": str(db_path),
        "photo_dir": str(photo_dir),
        "jpeg_subdir": jpeg_subdir,
        "opened_at": datetime.now().isoformat(),
    }
    if project_dir is not None:
        entry["project_dir"] = str(project_dir)
        entry["name"] = project_dir.name
    else:
        entry["name"] = photo_dir.name

    def _key(r: dict[str, Any]) -> str:
        return r.get("project_dir") or r.get("db_path") or ""
    new_key = entry.get("project_dir") or entry["db_path"]
    recents = [r for r in data.get("recents", []) if _key(r) != new_key]
    recents.insert(0, entry)
    data["recents"] = recents[:MAX_RECENTS]
    data["last_db_path"] = str(db_path)
    _save(data)


def forget(key: Path) -> None:
    """Remove a recent entry. `key` may be a db_path or project_dir."""
    data = _load()
    s = str(key)
    data["recents"] = [
        r for r in data.get("recents", [])
        if r.get("db_path") != s and r.get("project_dir") != s
    ]
    if data.get("last_db_path") == s:
        data["last_db_path"] = None
    _save(data)


# ----- edit presets (app-global) ------------------------------------------

def list_presets() -> list[dict[str, Any]]:
    return _load().get("presets", [])


def save_preset(name: str, edit: dict[str, Any]) -> dict[str, Any]:
    """Create or update (by name) a global edit preset. Returns the stored one."""
    data = _load()
    presets = data.get("presets", [])
    name = name.strip() or "Preset"
    existing = next((p for p in presets if p.get("name") == name), None)
    if existing is not None:
        existing["edit"] = edit
        preset = existing
    else:
        preset = {"id": uuid.uuid4().hex[:8], "name": name, "edit": edit}
        presets.append(preset)
    data["presets"] = presets
    _save(data)
    return preset


def delete_preset(preset_id: str) -> None:
    data = _load()
    data["presets"] = [p for p in data.get("presets", []) if p.get("id") != preset_id]
    _save(data)


# ----- workspaces ---------------------------------------------------------
# A workspace is a folder that holds project subfolders (each a project dir with
# its own picks.json). DaVinci-Resolve-style: pick a workspace, create projects
# by name inside it, list projects per workspace.

DEFAULT_WORKSPACE = Path.home() / "PictureClassifier-Projects"

# Deleting a project only renames its folder — the photos are never touched and
# the project data stays on disk. A folder carrying this marker is skipped when
# the workspace is listed, so "undo" is just renaming it back.
DELETED_MARKER = ".deleted-"


def is_deleted_project(name: str) -> bool:
    return DELETED_MARKER in name


def deleted_project_path(project_dir: Path, stamp: str) -> Path:
    """Free path to rename `project_dir` to when it is deleted at `stamp`."""
    target = project_dir.with_name(f"{project_dir.name}{DELETED_MARKER}{stamp}")
    n = 2
    while target.exists():
        target = project_dir.with_name(f"{project_dir.name}{DELETED_MARKER}{stamp}-{n}")
        n += 1
    return target


def get_workspaces() -> list[str]:
    return _load().get("workspaces", [])


def get_current_workspace() -> str | None:
    data = _load()
    cur = data.get("current_workspace")
    if cur:
        return cur
    ws = data.get("workspaces", [])
    return ws[0] if ws else None


def add_workspace(path: Path) -> None:
    """Register (or bump) a workspace and make it current."""
    data = _load()
    p = str(path)
    ws = [w for w in data.get("workspaces", []) if w != p]
    ws.insert(0, p)
    data["workspaces"] = ws
    data["current_workspace"] = p
    _save(data)


def set_current_workspace(path: Path) -> None:
    data = _load()
    p = str(path)
    if p not in data.get("workspaces", []):
        data.setdefault("workspaces", []).insert(0, p)
    data["current_workspace"] = p
    _save(data)


def forget_workspace(path: Path) -> None:
    data = _load()
    p = str(path)
    data["workspaces"] = [w for w in data.get("workspaces", []) if w != p]
    if data.get("current_workspace") == p:
        data["current_workspace"] = data["workspaces"][0] if data["workspaces"] else None
    _save(data)
