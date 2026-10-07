"""User-level state in `state.json`: recents, last db, and app-global edit
presets. Lives in the platform's application-data directory — see `paths`."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import fsutil, paths

CONFIG_DIR = paths.DATA_DIR
STATE_FILE = CONFIG_DIR / "state.json"
MAX_RECENTS = 10


def _load() -> dict[str, Any]:
    if not STATE_FILE.is_file():
        return {"recents": [], "last_db_path": None}
    try:
        return json.loads(fsutil.read_text(STATE_FILE))
    except (OSError, json.JSONDecodeError):
        return {"recents": [], "last_db_path": None}


def _save(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    fsutil.replace(tmp, STATE_FILE)


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
    data.get("views", {}).pop(s, None)   # nothing left to reopen it at
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


# ----- export settings (app-global) ---------------------------------------
# The last export's format, size, metadata and naming, so the dialog opens on
# what was used before. Global rather than per project: how someone delivers
# is a habit of theirs, not a property of one shoot. The target folder is not
# kept, since that one is per project.

def get_export_settings() -> dict[str, Any]:
    """The last export's settings, or {} before the first export."""
    return _load().get("export_settings", {})


def set_export_settings(settings: dict[str, Any]) -> None:
    data = _load()
    data["export_settings"] = settings
    _save(data)


# ----- updates (app-global) --------------------------------------------------
# Whether the app looks for a new version by itself (updater.py). On unless
# turned off in Preferences; "Check now" works either way.

def get_update_auto() -> bool:
    return bool(_load().get("update_auto", True))


def set_update_auto(on: bool) -> None:
    data = _load()
    data["update_auto"] = bool(on)
    _save(data)


# ----- look library (app-global) -------------------------------------------
# Imported .cube files and fitted colour matches, one JSON file each, named by
# `lut.table_key`. Global like presets, so a look imported once is there in
# every project. A project also keeps its own copy of each look its edits use
# (the "luts" map in picks.json), so it still renders if this library is
# cleared or the project moves to another machine.

LUT_DIR = CONFIG_DIR / "luts"


def _lut_path(key: str) -> Path:
    assert len(key) == 16 and all(c in "0123456789abcdef" for c in key), \
        f"not a look key: {key!r}"
    return LUT_DIR / f"{key}.json"


def list_luts() -> list[dict[str, Any]]:
    """Every look in the library, by name, without its table."""
    if not LUT_DIR.is_dir():
        return []
    out = []
    for f in LUT_DIR.glob("*.json"):
        p = json.loads(f.read_text(encoding="utf-8"))
        out.append({"key": f.stem, "name": p.get("name", ""),
                    "dim": p.get("dim"), "size": p.get("size")})
    return sorted(out, key=lambda e: e["name"].lower())


def save_lut(key: str, params: dict[str, Any]) -> None:
    LUT_DIR.mkdir(parents=True, exist_ok=True)
    dst = _lut_path(key)
    tmp = dst.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(params), encoding="utf-8")
    fsutil.replace(tmp, dst)


def load_lut(key: str) -> dict[str, Any] | None:
    path = _lut_path(key)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def delete_lut(key: str) -> None:
    _lut_path(key).unlink(missing_ok=True)


# ----- remembered view, per project ---------------------------------------
# Which filter, layout and page a project was last left on, so reopening it puts
# you back where you were instead of on page 1 of everything. A user preference
# rather than anything about the photos, so it lives here and not in the db.

MAX_VIEWS = 50


def view_key(db_path: Path, project_dir: Path | None = None) -> str:
    """Identify a project the way `remember_open` files it, so the two agree on
    what counts as the same project."""
    return str(project_dir if project_dir is not None else db_path)


def get_view(key: str) -> dict[str, Any]:
    """The saved view, or {} for a project opened for the first time."""
    return _load().get("views", {}).get(key, {})


def set_view(key: str, view: dict[str, Any]) -> None:
    data = _load()
    views = data.setdefault("views", {})
    views[key] = {**view, "saved_at": datetime.now().isoformat()}
    if len(views) > MAX_VIEWS:
        # Drop the least recently left. A page number in a project untouched for
        # that long is not worth keeping the file big for.
        oldest = sorted(views, key=lambda k: views[k].get("saved_at") or "")
        for stale in oldest[: len(views) - MAX_VIEWS]:
            del views[stale]
    _save(data)


# ----- workspaces ---------------------------------------------------------
# A workspace is a folder that holds project subfolders (each a project dir with
# its own picks.json). DaVinci-Resolve-style: pick a workspace, create projects
# by name inside it, list projects per workspace.

DEFAULT_WORKSPACE = Path.home() / "Pickapicka-Projects"

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


def include_workspace(path: Path) -> None:
    """List a workspace without making it current, as add_workspace would:
    one projects were just moved into, while the person stays where they are."""
    data = _load()
    p = str(path)
    ws = data.setdefault("workspaces", [])
    if p not in ws:
        ws.append(p)
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


# ----- where projects live ---------------------------------------------------
# New projects go in the current workspace ("workspace") or inside their photo
# folder's `.pickapicka/` ("photos"); see `projects`. A project kept with its
# photos is in no workspace, so it is listed from `known_projects`, which every
# create and open adds to.

def get_project_location() -> str:
    return _load().get("project_location", "workspace")


def set_project_location(location: str) -> None:
    assert location in ("workspace", "photos"), f"unknown project location: {location!r}"
    data = _load()
    data["project_location"] = location
    _save(data)


def known_projects() -> list[str]:
    return _load().get("known_projects", [])


def remember_project(project_dir: Path) -> None:
    data = _load()
    p = str(project_dir)
    known = data.setdefault("known_projects", [])
    if p not in known:
        known.append(p)
        _save(data)


def forget_project(project_dir: Path) -> None:
    data = _load()
    p = str(project_dir)
    data["known_projects"] = [k for k in data.get("known_projects", []) if k != p]
    _save(data)


def project_moved(old: Path, new: Path) -> None:
    """Refile everything keyed by a project's folder under its new one: the
    recents entry, the remembered view and the known-projects list. Without
    this a moved project reopens on page 1 and lingers in recents at a path
    that no longer exists."""
    data = _load()
    o, n = str(old), str(new)
    o_db, n_db = str(old / "picks.json"), str(new / "picks.json")
    for r in data.get("recents", []):
        if r.get("project_dir") == o:
            r["project_dir"] = n
            r["db_path"] = n_db
            r["name"] = new.name
    if data.get("last_db_path") == o_db:
        data["last_db_path"] = n_db
    views = data.get("views", {})
    if o in views:
        views[n] = views.pop(o)
    # The new path may be known already (opened from its new place first).
    known = [k for k in data.get("known_projects", []) if k not in (o, n)]
    data["known_projects"] = known + [n]
    _save(data)
