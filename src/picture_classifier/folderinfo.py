"""What a folder holds, described before anything is created in it.

Setting up asks for two folders that are easy to confuse: the photos of a shoot,
and the projects folder (the workspace) the app writes its own files into.
Counting what is already inside lets the setup screens say so, with "412 photos
in 8 subfolders" or "this folder already holds photos", instead of a wrong pick
failing at create time or, worse, succeeding and mixing project files in with
the photos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import raw
from .scorer import _is_supported

# Files looked at before stopping. Someone will pick a whole drive, and the
# useful answer there ("lots of photos") is already in a partial count.
SCAN_LIMIT = 20_000

# Cache folders a legacy picks.json keeps beside the photos it describes.
_LEGACY_CACHE_SUFFIXES = (".thumbs", ".faces", ".peaks", ".hdr", ".rawcache")


def is_project_dir(d: Path) -> bool:
    """A project folder as `create_project` makes it: picks.json and .cache/."""
    return (d / "picks.json").is_file() and (d / ".cache").is_dir()


def inspect(
    root: Path, workspace: Path | None = None, limit: int = SCAN_LIMIT,
) -> dict[str, Any]:
    """Count the photos under `root` and note how it relates to `workspace`.

    Project folders are counted but not descended into: what they hold is the
    app's own thumbnails, which would otherwise read as photos. RAW and
    JPEG/PNG files are counted separately, as files rather than shots, since a
    RAW+JPEG pair is one shot but two files.
    """
    root = root.expanduser()
    info: dict[str, Any] = {
        "path": str(root),
        "absolute": root.is_absolute(),
        "exists": root.is_absolute() and root.is_dir(),
        "photos": 0,           # JPEG / PNG files
        "raws": 0,             # RAW files
        "loose": 0,            # photos directly in `root`, in no subfolder
        "subfolders": 0,       # first-level subfolders holding any photo
        "projects": 0,         # project folders directly in `root`
        "is_project": False,
        "contains_workspace": False,
        "truncated": False,
    }
    if not info["exists"]:
        return info
    root = root.resolve()
    info["path"] = str(root)
    if workspace is not None:
        ws = workspace.expanduser().resolve()
        # Every project would land inside this folder, which create refuses.
        info["contains_workspace"] = ws.is_relative_to(root)
    if is_project_dir(root):
        info["is_project"] = True
        return info

    scenes: set[str] = set()
    seen = 0
    for dirpath, dirnames, filenames in root.walk():
        keep = []
        for d in dirnames:
            if is_project_dir(dirpath / d):
                if dirpath == root:
                    info["projects"] += 1
                continue
            if d.startswith("picks.json") and d.endswith(_LEGACY_CACHE_SUFFIXES):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in filenames:
            seen += 1
            if seen > limit:
                info["truncated"] = True
                break
            f = dirpath / fn
            if _is_supported(f):
                info["photos"] += 1
            elif raw.is_raw(f):
                info["raws"] += 1
            else:
                continue
            if dirpath == root:
                info["loose"] += 1
            else:
                scenes.add(f.relative_to(root).parts[0])
        if info["truncated"]:
            break
    info["subfolders"] = len(scenes)
    return info
