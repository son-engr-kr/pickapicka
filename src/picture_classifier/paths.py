"""Where the app keeps its own files.

Per-platform convention, via `platformdirs`:

    macOS    ~/Library/Application Support/picture-classifier
             ~/Library/Caches/picture-classifier
    Windows  %LOCALAPPDATA%\\picture-classifier
             %LOCALAPPDATA%\\picture-classifier\\Cache
    Linux    ~/.local/share/picture-classifier   (or $XDG_DATA_HOME)
             ~/.cache/picture-classifier         (or $XDG_CACHE_HOME)

Two directories, split by whether losing the contents costs anything:

- `DATA_DIR` holds state that cannot be regenerated — recents, workspaces, the
  remembered per-project view, saved presets. Backed up by Time Machine.
- `CACHE_DIR` holds downloaded model weights, which re-download on demand. On
  macOS this keeps a 20 MB blob out of every backup.

Everything used to live in `~/.picture-classifier`. That is a Python-CLI
convention, and this ships as an installed app on three platforms — a dotfile in
the Windows user profile is nobody's convention. `migrate_legacy()` moves an old
install across on first run.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import platformdirs

APP_NAME = "picture-classifier"

# appauthor=False keeps Windows at %LOCALAPPDATA%\picture-classifier rather than
# nesting it under a vendor directory nobody would recognise.
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
CACHE_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False))

MODEL_DIR = CACHE_DIR / "models"

LEGACY_DIR = Path.home() / ".picture-classifier"

# What moved where, when the old layout is found. Model weights go to the cache
# directory, not alongside the state, which is the one change of shape.
_LEGACY_MOVES = (
    ("state.json", DATA_DIR),
    ("models", CACHE_DIR),
)


def migrate_legacy() -> list[str]:
    """Move a `~/.picture-classifier` install to the per-platform directories.

    Idempotent, and never overwrites: anything already present at the
    destination wins, and the source is left where it is so it can be looked at.
    Returns one description per item moved, for the caller to report.
    """
    if not LEGACY_DIR.is_dir():
        return []

    moved: list[str] = []
    for name, dest_dir in _LEGACY_MOVES:
        src, dst = LEGACY_DIR / name, dest_dir / name
        if not src.exists() or dst.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(f"{src} -> {dst}")

    # Only clear the old directory once it is genuinely empty; anything left
    # behind is something this function chose not to touch, and deleting it
    # would be the one irreversible thing here.
    if not any(LEGACY_DIR.iterdir()):
        LEGACY_DIR.rmdir()

    return moved
