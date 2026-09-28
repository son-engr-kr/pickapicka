"""What a folder holds, described before anything is created in it.

Setting up asks for two folders that are easy to confuse: the photos of a shoot,
and the projects folder (the workspace) the app writes its own files into.
Counting what is already inside lets the setup screens say so, with "412 photos
in 8 subfolders" or "this folder already holds photos", instead of a wrong pick
failing at create time or, worse, succeeding and mixing project files in with
the photos.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from . import raw, scenes
from .scorer import _is_supported

# Files looked at before stopping. Someone will pick a whole drive, and the
# useful answer there ("lots of photos") is already in a partial count.
SCAN_LIMIT = 20_000

# Cache folders a legacy picks.json keeps beside the photos it describes.
_LEGACY_CACHE_SUFFIXES = (".thumbs", ".faces", ".peaks", ".hdr", ".rawcache")


# Shots read for a grouping preview before stopping. Reading a capture time is
# a few milliseconds a file, so this is seconds, not minutes.
SHOT_LIMIT = 6000
_TIME_WORKERS = 8


def _skip_dir(parent: Path, name: str) -> bool:
    """A folder that holds the app's own files, not photos."""
    return is_project_dir(parent / name) or (
        name.startswith("picks.json") and name.endswith(_LEGACY_CACHE_SUFFIXES))


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

    subdirs: set[str] = set()
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
                subdirs.add(f.relative_to(root).parts[0])
        if info["truncated"]:
            break
    info["subfolders"] = len(subdirs)
    return info


_WALL_EPOCH = datetime(1970, 1, 1)


def _wall_seconds(t: datetime) -> float:
    """A camera's wall-clock time as seconds, read as if it were UTC.

    Capture times carry no timezone, and only the gaps between them matter for
    grouping, so no timezone is applied: the client formats these back as UTC
    and shows the clock the camera showed. `t.timestamp()` would apply this
    machine's zone instead, and raises OSError on Windows for a camera whose
    clock was never set and stamps 1970 or earlier.
    """
    return (t - _WALL_EPOCH).total_seconds()


def shots(
    root: Path, raw_root: Path | None = None, limit: int = SHOT_LIMIT,
) -> dict[str, Any]:
    """The shots under `root` the way scoring will count them, with their
    capture times, for the new-project wizard to preview scene grouping on.

    A shot is what scoring makes one photo of: files with the same stem in the
    same first-level folder, so a RAW and its JPEG are one shot, and its time
    is the RAW's when it has one (`scorer` pairs them on exactly that key and
    takes the RAW's time from libraw). RAWs are looked for under `raw_root`
    when the project keeps them in a subfolder of their own, as scoring does.
    HDR brackets are merged later, at scoring, and are counted here as their
    separate frames.

    Returns {shots, times, untimed, folders, truncated}: `times` are the known
    capture times as sorted epoch seconds, `folders` the shot count per
    first-level folder ("(none)" for loose files), both as scoring would see
    them.
    """
    root = root.expanduser().resolve()
    assert root.is_dir(), f"not a folder: {root}"
    raw_root = root if raw_root is None else raw_root.expanduser().resolve()
    by_key: dict[tuple[str, str], dict[str, Path]] = {}
    truncated = False
    # One walk when the RAWs sit among the JPEGs, two when they have a folder.
    walks = [(root, ("jpeg", "raw"))] if raw_root == root else \
        [(root, ("jpeg",))] + ([(raw_root, ("raw",))] if raw_root.is_dir() else [])
    for base, wanted in walks:
        for dirpath, dirnames, filenames in base.walk():
            dirnames[:] = [d for d in dirnames if not _skip_dir(dirpath, d)]
            for fn in filenames:
                f = dirpath / fn
                kind = "jpeg" if _is_supported(f) else "raw" if raw.is_raw(f) else None
                if kind not in wanted:
                    continue
                rel = f.relative_to(base)
                scene = rel.parts[0] if len(rel.parts) > 1 else "(none)"
                key = (scene, f.stem.lower())
                if key not in by_key and len(by_key) >= limit:
                    truncated = True
                    break
                by_key.setdefault(key, {})[kind] = f
            if truncated:
                break
        if truncated:
            break

    def when(files: dict[str, Path]) -> float | None:
        if "raw" in files:
            # From the file's head where the format allows (raw.head_capture_time):
            # libraw unpacks the whole file for it, half a second a frame.
            t = raw.read_capture_time(files["raw"])
        else:
            t = scenes.read_capture_time(files["jpeg"])
        return _wall_seconds(t) if t is not None else None

    keys = sorted(by_key)
    with ThreadPoolExecutor(max_workers=_TIME_WORKERS) as pool:
        times = list(pool.map(lambda k: when(by_key[k]), keys))
    timed = sorted(t for t in times if t is not None)
    counts = Counter(k[0] for k in keys)
    return {
        "shots": len(keys),
        "times": timed,
        "untimed": len(keys) - len(timed),
        "folders": [{"name": n, "count": c} for n, c in sorted(counts.items())],
        "truncated": truncated,
    }
