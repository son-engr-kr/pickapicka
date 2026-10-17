"""The Claude Code skill that teaches an assistant the command line, written
with the path of the command that wrote it, so it works for the installed app
as well, whose command is not on PATH.

The app writes it as it opens (`refresh`) when Claude Code is on the machine,
so there is nothing to set up: an assistant started afterwards already knows
the commands, and an update brings the skill up to date with it. Preferences
switches that off, which takes the skill away again. `pickapicka install-skill`
writes it by hand.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__, userstate

CLAUDE_DIR = Path.home() / ".claude"
DEFAULT_DIR = CLAUDE_DIR / "skills" / "pickapicka"
# On the first line after the front matter of every skill this writes. A
# SKILL.md there without it is the person's own, and is left alone.
MARK = "<!-- Written by Pickapicka"

TEXT = """\
---
name: pickapicka
description: Cull, edit and export a photo shoot with Pickapicka, the photo culling and editing app installed on this machine. Looks at the photos (contact sheets, full views, 100% crops), picks and rejects, rates, applies looks and quick edits, and exports, all shown live in the app's window when it is open. Use when the person wants to sort, choose, cull, grade, edit or export their photos, asks about "this photo" in Pickapicka, or mentions Pickapicka.
---

<!-- Written by Pickapicka {version}, as it opened or by `pickapicka install-skill`. It is replaced when Pickapicka is updated. -->

# Working a shoot with Pickapicka

The command is `{cmd}`, written PK below. Every command prints one JSON object,
and `PK COMMAND --help` says exactly what it does and what each field means. A
command that fails prints `{{"error", "status"}}` on stderr: read it, it says why.

The person may not be a programmer. Talk with them in their language and in plain
words, run every command yourself, and never ask them to type one or to edit a file.
Never edit a project's picks.json yourself: the app holds it in memory and would
write over you.

## 1. Where things are

Run `PK status`.

- `app.running` and a `project`: the app is open on that project. Changes you make
  show in its window at once (each change answers `"in_app": true`). `screen` is
  where the person is: `screen.photo` is the photo their cursor is on, so "this
  photo" means it, and `screen.scene` is "this scene".
- No project, or the app is closed: `PK projects` lists them. Ask which one if it is
  not clear, and give every command `-p NAME`. Changes are saved to the project and
  are there when the person opens it.

## 2. Looking at photos

You can see the photos: these commands write image files, which you then read.

- `PK scenes`: the scenes, and how far each is culled.
- `PK sheet --scene SCENE`: contact sheets, twelve photos to an image, each numbered,
  with its stars and decision under it. The JSON maps each number to its `rel_path`.
- `PK photos --scene SCENE --sort badness`: the scorer's view of the same photos:
  `scores.blur_pct` (sharpness rank in the scene, 1 the sharpest), `scores.badness`
  (lower is better), `eye_open` (below 0.18 is closed), `suggested`.
- `PK view REL`: one photo, larger. `--crop X,Y,W,H` (fractions of the frame) shows
  a region at full resolution, for focus and eyes. `--original` shows it as shot.

The scores are a hint, not the decision: they rank sharpness and brightness within
a scene and know nothing of a moment, an expression or a composition. Look before
you decide, and look closely (`--crop` on the faces) before rejecting a photo for
softness or closed eyes.

## 3. Culling

`PK decide pick|review|reject|none REL...` and `PK mark REL... --rating N --label COLOUR`.
Pass many photos at once; `-` reads them from stdin, one per line. When you cull for
the person, say what you did and why in a sentence or two per scene, and how many
picks you are leaving them with. Ask before rejecting whole scenes.

## 4. Editing

- `PK presets` lists the person's presets and the looks that come with the app.
  Try one first with `PK view REL --preset NAME`, look at it, then
  `PK apply NAME REL...`. Looks keep each photo's brightness.
- `PK quick REL... --by exposure=0.33 --auto --level`: sliders moved on each photo
  from where it is (`PK sliders` lists them), auto tone, horizon levelled.
- `PK edit REL... --set JSON` writes a whole edit (`PK photo REL` shows one under
  "edit"); try it with `PK view REL --edit JSON` first.
- `--slot 0..5` on `apply` and `edit` saves the result as a version beside the
  photo's own edit instead of replacing it: the way to offer the person choices.

## 5. Export and undo

`PK export` writes the picks, edits applied and RAW rendered, with the person's
last export settings, and waits until done; its JSON says where (`target_dir`).
`--which kept|all`, `--scene`, `--long-edge`, `--to FOLDER` change that.
Show the person the folder (`open TARGET_DIR` on macOS).

`PK undo` puts back what the last command-line change replaced, in that project.
Twenty are kept. Offer it when the person does not like a change.
"""


def command_path() -> str:
    """How to run the command that is running now: its name when PATH finds
    this same one, else its full path (the installed app's is inside it)."""
    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve()
    else:
        here = Path(sys.argv[0]).resolve()
    on_path = shutil.which("pickapicka")
    if on_path and Path(on_path).resolve() == here:
        return "pickapicka"
    return str(here)


def render(cmd: str | None = None) -> str:
    return TEXT.format(cmd=cmd or command_path(), version=__version__)


def install(folder: Path | None = None) -> Path:
    folder = folder or DEFAULT_DIR
    folder.mkdir(parents=True, exist_ok=True)
    dst = folder / "SKILL.md"
    dst.write_text(render(), encoding="utf-8")
    return dst


def _unusable() -> str | None:
    """Why this app's command cannot be what an assistant runs, if it cannot.
    The Windows app is built without a console, so its command prints nothing."""
    if getattr(sys, "frozen", False) and os.name == "nt":
        return "The Windows app's command prints nothing. Install with uv tool to use it."
    return None


def _ours(dst: Path) -> bool:
    return MARK in dst.read_text(encoding="utf-8")


def refresh() -> dict[str, Any]:
    """Write the skill if it is wanted, Claude Code is here, and what is there
    is not already this: called as the app opens. Says what it found."""
    dst = DEFAULT_DIR / "SKILL.md"
    if not userstate.get_assistant_skill():
        return status("off")
    if _unusable():
        return status("unusable")
    if not CLAUDE_DIR.is_dir():
        return status("no_claude")
    if dst.is_file() and not _ours(dst):
        return status("theirs")
    if dst.is_file() and dst.read_text(encoding="utf-8") == render():
        return status("current")
    try:
        install()
    except OSError as exc:
        # A Claude folder this app may not write to is no reason not to open:
        # said in Preferences, where the switch for it is.
        print(f"Could not write the Claude Code skill: {exc}", file=sys.stderr)
        return status("failed", str(exc))
    return status("written")


def remove() -> dict[str, Any]:
    """Take away the skill this app wrote, and its folder if nothing else is
    in it. One the person wrote is left."""
    dst = DEFAULT_DIR / "SKILL.md"
    if dst.is_file() and _ours(dst):
        dst.unlink()
        if not any(DEFAULT_DIR.iterdir()):
            DEFAULT_DIR.rmdir()
    return status("off")


def status(found: str, error: str | None = None) -> dict[str, Any]:
    """What Preferences says: whether it is on, and what `refresh` found."""
    return {"on": userstate.get_assistant_skill(), "found": found, "error": error,
            "unusable": _unusable(), "path": str(DEFAULT_DIR / "SKILL.md"),
            "command": command_path()}
