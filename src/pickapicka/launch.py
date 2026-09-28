"""What the app does as it opens: point the browser at the intro, play the chime.

The chime is played by this process, not by the page. Browsers will not start
audio before the user has clicked or pressed something on the page (Chrome's
and Safari's autoplay policies), and a launch is exactly the moment when nobody
has. So the page asks for it (POST /api/launch-sound) as its intro starts, and
the sound comes from the platform's own player.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

SOUND = Path(__file__).parent / "web" / "sounds" / "launch.wav"

# The page plays its intro when it is opened at this address. Only the app's
# own launch opens it, so a reload or a second tab goes straight to work.
INTRO_QUERY = "launch"


def intro_url(base: str) -> str:
    return f"{base}/?{INTRO_QUERY}"


def can_play() -> bool:
    """Whether this platform has a player every install can count on."""
    return sys.platform in ("darwin", "win32")


def play_sound() -> bool:
    """Start the chime and return without waiting for it to finish.

    False, and silence, on Linux: no one player is on every desktop there.
    """
    if sys.platform == "darwin":
        # afplay ships with macOS. A thread waits on it so the finished
        # process is reaped rather than left behind.
        threading.Thread(target=subprocess.run, args=(["afplay", str(SOUND)],),
                         kwargs={"check": True}, daemon=True).start()
        return True
    if sys.platform == "win32":
        import winsound
        winsound.PlaySound(str(SOUND), winsound.SND_FILENAME | winsound.SND_ASYNC)
        return True
    return False
