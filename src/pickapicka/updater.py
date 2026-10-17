"""Keeping an installed app current: a check against GitHub Releases, the new
installer fetched in the background, and one click to put it in.

The installers are the ones the Releases page offers (the macOS .pkg and the
Windows setup.exe), installed the way a person would, only without a browser
in between. A file the app downloads itself carries no quarantine flag (macOS)
and no mark of the web (Windows), so the unsigned-app warnings that a browser
download brings never appear. What is left is the system's own consent to
write into /Applications or Program Files: the password dialog, or the UAC
prompt.

Sparkle, the usual macOS updater, draws its dialogs with AppKit and needs an
AppKit event loop on the app's main thread. This app's main thread runs the web
server and its only windows are browser tabs, so the update is offered in the
page instead, and put in by the platform's own installer.

Installing cannot happen inside the app, because the files being replaced are
the ones it is running from. So the app writes a short script, starts it
detached and quits; the script waits for it to be gone, runs the installer,
notes how that went and opens the app again. The page, which stays open, finds
the new server and reloads.

Every download is checked against the SHA-256 digest GitHub publishes for the
release asset, and a file that does not match is thrown away, never run. The
connection itself is checked by the system's certificate verifier (truststore,
set up in cli.main), not by Python's own reading of the Windows store.

A check that fails is tried again a quarter of an hour later, not at the next
six-hourly one, and one that keeps failing is shown in the top bar: an update
the app silently cannot fetch looks to the user exactly like no update at all.

Other installs (uv tool, Homebrew, a checkout) are told of a new version and
given the command that updates them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__, fsutil, paths

REPO = "son-engr-kr/pickapicka"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
FIRST_CHECK_DELAY = 10.0        # seconds after starting: after the intro, not during it
CHECK_INTERVAL = 6 * 3600.0     # and again while it keeps running
# After a failed check. A start before the network is up (a laptop just woken)
# heals in minutes, and GitHub's limit of 60 unauthenticated API calls an hour
# per address, which a shared office or campus address can use up, resets
# within the hour; four tries an hour stay far below it.
RETRY_INTERVAL = 15 * 60.0
# Failed checks in a row before the top bar says so. One or two is a moment
# offline, which is ordinary; three is half an hour of not knowing.
FAILURES_SHOWN = 3
UPDATE_DIR = paths.CACHE_DIR / "updates"
# Written as the app hands over to the installer, read by the version that
# comes up afterwards: whether it is that new version is the answer.
PENDING_FILE = paths.DATA_DIR / "update-pending.json"
EXIT_FILE_NAME = "installer-exit.txt"
# Where the .pkg puts the app: pkgbuild --install-location in build-macos.yml,
# with relocation off, so it is always here rather than wherever a copy was.
MAC_INSTALLED_APP = Path("/Applications/Pickapicka.app")
ASSET_SUFFIX = {"macos-app": "-macos-arm64.pkg", "windows-app": "-windows-x64.exe"}
UPGRADE_COMMAND = {"homebrew": "brew upgrade pickapicka", "uv-tool": "uv tool upgrade pickapicka"}
# Inno Setup: a progress window and nothing else, no restart, no questions.
# Its own "launch Pickapicka" step is skipped when silent; the script relaunches.
INNO_SILENT = ["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-"]
API_TIMEOUT = 15
DOWNLOAD_TIMEOUT = 60
CHUNK = 1 << 20


def parse_version(text: str) -> tuple[int, ...]:
    m = re.fullmatch(r"v?(\d+(?:\.\d+)*)", text.strip())
    assert m, f"not a release version: {text!r}"
    return tuple(int(x) for x in m.group(1).split("."))


def install_kind() -> str:
    """How this copy was installed, which decides how it can be updated."""
    if getattr(sys, "frozen", False):
        return {"darwin": "macos-app", "win32": "windows-app"}.get(sys.platform, "other")
    prefix = Path(sys.prefix)
    # uv writes a receipt into every tool environment it makes.
    if (prefix / "uv-receipt.toml").is_file():
        return "uv-tool"
    # The formula installs a venv under the Cellar (homebrew-pickapicka).
    if "Cellar" in prefix.parts:
        return "homebrew"
    return "other"


def pick_asset(kind: str, assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    suffix = ASSET_SUFFIX.get(kind)
    if suffix is None:
        return None
    found = [a for a in assets if a["name"].endswith(suffix)]
    assert len(found) <= 1, f"more than one {suffix} in the release: {[a['name'] for a in found]}"
    return found[0] if found else None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def running_mac_bundle() -> Path:
    """The .app this process runs from: .../Pickapicka.app/Contents/MacOS/pickapicka."""
    bundle = Path(sys.executable).resolve().parents[2]
    assert bundle.suffix == ".app", f"not inside an app bundle: {sys.executable}"
    return bundle


# ----- the scripts that do the installing ---------------------------------

# The .pkg's path and the dialog's words come in as arguments, and AppleScript
# quotes the path for the shell itself (quoted form of), so nothing in either is
# ever read as shell syntax.
MAC_APPLESCRIPT = (
    "on run argv",
    ('do shell script "/usr/sbin/installer -pkg " & quoted form of (item 1 of argv) & " -target /"'
     " with prompt (item 2 of argv) with administrator privileges"),
    "end run",
)


def mac_install_script(*, pkg: Path, pid: int, port: int, exit_file: Path,
                       installed_app: Path, running_app: Path, prompt: str) -> str:
    """Wait for the app to quit, install the .pkg as an administrator (the
    system's password dialog, with our own words in it), and open the app:
    the new one, or the old one again when the install did not happen."""
    q = shlex.quote
    osa = " ".join(f"-e {q(line)}" for line in MAC_APPLESCRIPT)
    return f"""#!/bin/sh
# Written by Pickapicka's updater (updater.py), run once, after the app quits.
while kill -0 {pid} 2>/dev/null; do sleep 0.2; done
message=$(/usr/bin/osascript {osa} {q(str(pkg))} {q(prompt)} 2>&1)
code=$?
printf '%s\\n%s\\n' "$code" "$message" > {q(str(exit_file))}
if [ "$code" -eq 0 ]; then app={q(str(installed_app))}; else app={q(str(running_app))}; fi
/usr/bin/open -a "$app" --args serve --port {port}
"""


def _ps(text: str) -> str:
    """A PowerShell single-quoted string: nothing inside is interpreted."""
    return "'" + text.replace("'", "''") + "'"


def windows_install_script(*, installer: Path, pid: int, port: int, exit_file: Path,
                           exe: Path, as_app: bool) -> str:
    """Wait for the app to quit, run setup.exe silently as an administrator
    (the UAC prompt), and start the app again from where it is installed, which
    the upgrade keeps. Declining the prompt raises, and is recorded as such.
    `as_app` starts it as the app it was, which quits with its last tab; on
    macOS, opening it with `open` already says so (app_entry.py)."""
    args = ",".join(_ps(a) for a in INNO_SILENT)
    serve = ",".join(_ps(a) for a in ["serve", "--port", str(port)] + (["--as-app"] if as_app else []))
    return f"""# Written by Pickapicka's updater (updater.py), run once, after the app quits.
$ErrorActionPreference = 'Stop'
while (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ Start-Sleep -Milliseconds 200 }}
$code = 0
$message = ''
try {{
  $p = Start-Process -FilePath {_ps(str(installer))} -ArgumentList {args} -Verb RunAs -Wait -PassThru
  $code = $p.ExitCode
}} catch {{
  $code = -1
  $message = $_.Exception.Message
}}
Set-Content -LiteralPath {_ps(str(exit_file))} -Value @("$code", $message) -Encoding UTF8
Start-Process -FilePath {_ps(str(exe))} -ArgumentList {serve}
"""


def read_result(pending_file: Path, exit_file: Path, current: str) -> dict[str, Any] | None:
    """How the last update went, once, as the version after it starts."""
    if not pending_file.is_file():
        return None
    pending = json.loads(fsutil.read_text(pending_file))
    code, message = None, ""
    if exit_file.is_file():
        # PowerShell's UTF8 writes a byte-order mark; the shell's does not.
        lines = exit_file.read_text(encoding="utf-8-sig").splitlines()
        code = int(lines[0]) if lines and re.fullmatch(r"-?\d+", lines[0].strip()) else None
        message = "\n".join(lines[1:]).strip()
        exit_file.unlink()
    pending_file.unlink()
    return {"from": pending["from"], "to": pending["to"], "at": pending["at"],
            "ok": parse_version(current) == parse_version(pending["to"]),
            "code": code, "message": message,
            "url": f"https://github.com/{REPO}/releases/tag/v{pending['to']}"}


# ----- the updater -------------------------------------------------------------

class Updater:
    """What is known about the newest release, and the work of fetching it.

    `state` is what the page is shown (GET /api/update). Checking and
    downloading run on a background thread, one at a time."""

    def __init__(self, *, latest_url: str = LATEST_URL, update_dir: Path = UPDATE_DIR,
                 pending_file: Path = PENDING_FILE, kind: str | None = None,
                 current: str = __version__) -> None:
        self.latest_url = latest_url
        self.update_dir = update_dir
        self.pending_file = pending_file
        self.exit_file = update_dir / EXIT_FILE_NAME
        self.current = current
        self.kind = kind or install_kind()
        # A new id each start, so the page can tell the server that came back
        # after an update from the one that was still shutting down.
        self.boot = uuid.uuid4().hex
        self._lock = threading.Lock()
        self._busy = threading.Lock()
        self._installer: Path | None = None
        self.state: dict[str, Any] = {
            "current": current, "kind": self.kind,
            "can_install": self.kind in ASSET_SUFFIX,
            "command": UPGRADE_COMMAND.get(self.kind),
            "status": "idle",         # idle checking current available downloading ready installing error
            "latest": None,           # {"version", "url", "published"}
            "progress": None,         # {"done", "total"} bytes, while downloading
            "error": None,
            "failures": 0,            # failed checks in a row
            "newer": False,           # `latest` is newer than this version
            "checked_at": None,
            "result": read_result(pending_file, self.exit_file, current),
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self.state))

    def _set(self, **changes: Any) -> None:
        with self._lock:
            self.state.update(changes)

    # -- checking and fetching

    def check(self) -> None:
        """Ask GitHub for the newest release, and fetch its installer if it is
        newer and this install can put it in. A check already under way wins."""
        if not self._busy.acquire(blocking=False):
            return
        try:
            self._check()
        finally:
            self._busy.release()

    def _check(self) -> None:
        if self.state["status"] in ("ready", "installing"):
            return
        self._set(status="checking", error=None)
        try:
            req = urllib.request.Request(self.latest_url, headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": f"Pickapicka/{self.current}"})
            with urllib.request.urlopen(req, timeout=API_TIMEOUT) as r:
                release = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            # Offline is ordinary; the check after RETRY_INTERVAL tries again.
            self._fail(f"Could not reach GitHub: {exc}")
            return
        tag = release["tag_name"]
        latest = {"version": tag.lstrip("v"), "url": release["html_url"],
                  "published": release.get("published_at")}
        newer = parse_version(tag) > parse_version(self.current)
        self._set(latest=latest, newer=newer, checked_at=datetime.now().isoformat(timespec="seconds"))
        if not newer:
            self._set(status="current", failures=0)
            return
        asset = pick_asset(self.kind, release["assets"])
        if asset is None:
            self._set(status="available", failures=0)
            return
        self._download(asset)

    def _fail(self, error: str) -> None:
        with self._lock:
            self.state.update(status="error", progress=None, error=error,
                              failures=self.state["failures"] + 1,
                              checked_at=datetime.now().isoformat(timespec="seconds"))

    def _download(self, asset: dict[str, Any]) -> None:
        digest = asset.get("digest") or ""
        assert digest.startswith("sha256:"), f"{asset['name']} has no sha256 digest to check it against"
        expected = digest.split(":", 1)[1].lower()
        target = self.update_dir / asset["name"]
        try:
            self.update_dir.mkdir(parents=True, exist_ok=True)
            # Fetched on an earlier run and never installed: no need to fetch again.
            if not (target.is_file() and sha256_of(target) == expected):
                if not self._fetch(asset, target, expected):
                    return
            # Only the newest installer is worth its disk space.
            for old in self.update_dir.iterdir():
                if old.is_file() and old != target and old.name != EXIT_FILE_NAME:
                    old.unlink()
        except OSError as exc:
            # On Windows a virus scanner reading the new installer can hold it
            # for longer than fsutil.replace waits. Raised, this ended the
            # updater's thread, and with it every later check, without a word.
            self._fail(f"The installer could not be saved: {exc}")
            return
        self._installer = target
        self._set(status="ready", progress=None, failures=0)

    def _fetch(self, asset: dict[str, Any], target: Path, expected: str) -> bool:
        """Download `asset` to `target` through a .part file, kept only if it
        matches `expected`. False, with the failure set, when it did not."""
        self._set(status="downloading", progress={"done": 0, "total": asset["size"]})
        part = target.with_name(target.name + ".part")
        h = hashlib.sha256()
        done = 0
        try:
            req = urllib.request.Request(asset["browser_download_url"],
                                         headers={"User-Agent": f"Pickapicka/{self.current}"})
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, part.open("wb") as out:
                while chunk := r.read(CHUNK):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    self._set(progress={"done": done, "total": asset["size"]})
        except (urllib.error.URLError, OSError) as exc:
            part.unlink(missing_ok=True)
            self._fail(f"The download stopped: {exc}")
            return False
        if h.hexdigest() != expected:
            part.unlink()
            self._fail("The download did not match the checksum GitHub publishes for it, "
                       "so it was thrown away.")
            return False
        fsutil.replace(part, target)
        return True

    # -- installing

    def start_install(self, *, port: int, as_app: bool) -> None:
        """Hand over to the installer: write the script, start it, and leave
        quitting to the caller. The script waits for this process to go."""
        with self._lock:
            assert self.state["status"] == "ready" and self._installer is not None, \
                f"nothing to install (status {self.state['status']})"
            to = self.state["latest"]["version"]
            self.state["status"] = "installing"
        installer = self._installer
        self.pending_file.parent.mkdir(parents=True, exist_ok=True)
        self.pending_file.write_text(json.dumps({
            "from": self.current, "to": to,
            "at": datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
        self.exit_file.unlink(missing_ok=True)
        pid = os.getpid()
        if self.kind == "macos-app":
            script = self.update_dir / "install.sh"
            script.write_text(mac_install_script(
                pkg=installer, pid=pid, port=port, exit_file=self.exit_file,
                installed_app=MAC_INSTALLED_APP, running_app=running_mac_bundle(),
                prompt=f"Pickapicka is installing version {to}."), encoding="utf-8")
            subprocess.Popen(["/bin/sh", str(script)], start_new_session=True, close_fds=True,
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        elif self.kind == "windows-app":
            script = self.update_dir / "install.ps1"
            # With a byte-order mark: Windows PowerShell 5.1 reads a script
            # without one in the ANSI code page, which has no Korean in it.
            script.write_text(windows_install_script(
                installer=installer, pid=pid, port=port, exit_file=self.exit_file,
                exe=Path(sys.executable).resolve(), as_app=as_app), encoding="utf-8-sig")
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            subprocess.Popen(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy",
                              "Bypass", "-WindowStyle", "Hidden", "-File", str(script)],
                             creationflags=flags, close_fds=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            raise AssertionError(f"{self.kind} installs are updated with their own tool")

    # -- the background loop

    def run(self, enabled) -> None:
        """Check a little after starting and then every CHECK_INTERVAL, or
        RETRY_INTERVAL after a check that failed, while `enabled()` says
        automatic checks are on. Runs on a daemon thread."""
        time.sleep(FIRST_CHECK_DELAY)
        while True:
            if enabled():
                self.check()
            time.sleep(RETRY_INTERVAL if self.snapshot()["status"] == "error" else CHECK_INTERVAL)

    def start_background(self, enabled) -> None:
        threading.Thread(target=self.run, args=(enabled,), daemon=True, name="updater").start()
