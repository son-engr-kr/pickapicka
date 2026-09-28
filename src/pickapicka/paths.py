"""Where the app keeps its own files.

Per-platform convention, via `platformdirs`:

    macOS    ~/Library/Application Support/pickapicka
             ~/Library/Caches/pickapicka
    Windows  %LOCALAPPDATA%\\pickapicka
             %LOCALAPPDATA%\\pickapicka\\Cache
    Linux    ~/.local/share/pickapicka   (or $XDG_DATA_HOME)
             ~/.cache/pickapicka         (or $XDG_CACHE_HOME)

Two directories, split by whether losing the contents costs anything:

- `DATA_DIR` holds state that cannot be regenerated — recents, workspaces, the
  remembered per-project view, saved presets. Backed up by Time Machine.
- `CACHE_DIR` holds downloaded model weights, which re-download on demand. On
  macOS this keeps a 20 MB blob out of every backup.

Everything used to live in `~/.picture-classifier`. That is a Python-CLI
convention, and this ships as an installed app on three platforms — a dotfile in
the Windows user profile is nobody's convention. Then the app was renamed from
Picture Classifier, which named the per-platform directories too. On first run
`migrate_legacy()` moves either kind of old install across.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import platformdirs

APP_NAME = "pickapicka"

# appauthor=False keeps Windows at %LOCALAPPDATA%\pickapicka rather than
# nesting it under a vendor directory nobody would recognise.
DATA_DIR = Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
CACHE_DIR = Path(platformdirs.user_cache_dir(APP_NAME, appauthor=False))

MODEL_DIR = CACHE_DIR / "models"

# Where 0.6.0 through 0.9.0 kept them, under the app's old name.
PREVIOUS_NAME = "picture-classifier"
PREVIOUS_DATA_DIR = Path(platformdirs.user_data_dir(PREVIOUS_NAME, appauthor=False))
PREVIOUS_CACHE_DIR = Path(platformdirs.user_cache_dir(PREVIOUS_NAME, appauthor=False))

LEGACY_DIR = Path.home() / ".picture-classifier"

# What moved where, when the old layout is found. Model weights go to the cache
# directory, not alongside the state, which is the one change of shape.
_LEGACY_MOVES = (
    ("state.json", DATA_DIR),
    ("models", CACHE_DIR),
)


def migrate_legacy() -> list[str]:
    """Move an old install to the per-platform directories.

    Two kinds: the directories under the app's old name, and before them a
    `~/.picture-classifier` dotfile. Idempotent, and never overwrites: anything
    already present at the destination wins, and the source is left where it is
    so it can be looked at. Returns one description per item moved, for the
    caller to report.
    """
    moved: list[str] = []
    # Data before cache: on Windows the old cache directory is inside the old
    # data directory, so it moves with it and the second pass finds nothing.
    for old_dir, new_dir in ((PREVIOUS_DATA_DIR, DATA_DIR),
                             (PREVIOUS_CACHE_DIR, CACHE_DIR)):
        moved += _move_contents(old_dir, new_dir)

    if not LEGACY_DIR.is_dir():
        return moved

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


def _move_contents(src_dir: Path, dest_dir: Path) -> list[str]:
    """Move each entry of `src_dir` into `dest_dir`, skipping any already there.

    Entry by entry rather than the directory as a whole, because the bundled app
    has usually created `dest_dir` for its log before this runs.
    """
    if not src_dir.is_dir():
        return []
    moved: list[str] = []
    for src in sorted(src_dir.iterdir()):
        dst = dest_dir / src.name
        if dst.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(f"{src} -> {dst}")
    # As for the dotfile: only an emptied directory goes.
    if not any(src_dir.iterdir()):
        src_dir.rmdir()
    return moved
