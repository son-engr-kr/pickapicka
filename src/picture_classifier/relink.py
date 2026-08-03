"""Re-resolve a project's photo paths after its source folder moves or is renamed.

Fast path: if the project's existing relative structure still resolves under the
new root (a plain move/rename), just repoint `photo_root`. Fallback: for files
that no longer resolve, match by basename within the new tree (only when the
name is unique) and rewrite that photo's rel_path / src. Decisions and edits ride
along untouched because we mutate the photo dicts in place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from . import db


def _basename_index(root: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = {}
    if not root.is_dir():
        return idx
    for dirpath, _dirs, files in root.walk():
        for fn in files:
            if fn.startswith("._"):
                continue
            idx.setdefault(fn.lower(), []).append(dirpath / fn)
    return idx


def relink_project(db_path: Path, new_photo_root: Path) -> dict[str, Any]:
    """Point a project at `new_photo_root`, remapping files by basename where the
    relative path no longer resolves. Returns a match report."""
    data = db.load(db_path)
    jpeg_subdir = data.get("jpeg_subdir", "")
    raw_subdir = data.get("raw_subdir", "")
    new_jpeg_root = new_photo_root / jpeg_subdir if jpeg_subdir else new_photo_root
    new_raw_root = new_photo_root / raw_subdir if raw_subdir else new_jpeg_root

    total = resolved = rematched = 0
    unmatched: list[str] = []
    jpeg_idx: dict[str, list[Path]] | None = None
    raw_idx: dict[str, list[Path]] | None = None

    for p in data.get("photos", []):
        t = p.get("type")
        if t == "hdr":
            continue  # the merged result lives inside the project, not the photo folder
        total += 1
        if t == "raw":
            src = p.get("src", p["rel_path"])
            if (new_photo_root / src).is_file():
                resolved += 1
                continue
            if raw_idx is None:
                raw_idx = _basename_index(new_raw_root)
            cands = raw_idx.get(Path(src).name.lower(), [])
            if len(cands) == 1:
                p["src"] = str(cands[0].relative_to(new_photo_root))
                p["rel_path"] = str(cands[0].relative_to(new_raw_root))
                rematched += 1
            else:
                unmatched.append(Path(src).name)
        else:
            rel = p["rel_path"]
            if (new_jpeg_root / rel).is_file():
                resolved += 1
                continue
            if jpeg_idx is None:
                jpeg_idx = _basename_index(new_jpeg_root)
            cands = jpeg_idx.get(Path(rel).name.lower(), [])
            if len(cands) == 1:
                p["rel_path"] = str(cands[0].relative_to(new_jpeg_root))
                rematched += 1
            else:
                unmatched.append(Path(rel).name)

    data["photo_root"] = str(new_photo_root)
    db.save(db_path, data)
    return {
        "total": total,
        "resolved": resolved,
        "rematched": rematched,
        "unmatched": len(unmatched),
        "unmatched_names": unmatched[:20],
    }
