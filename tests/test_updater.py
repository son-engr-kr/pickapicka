"""The updater: what counts as newer, which installer is ours, that nothing is
kept that does not match GitHub's checksum, and the scripts that install.

    uv run pytest tests/test_updater.py

A local HTTP server plays GitHub: the release JSON and the installer. The
install scripts are checked for syntax (sh -n, osacompile, and PowerShell's own
parser on Windows) and for carrying Korean paths with spaces and quotes
through intact, but not run: running one installs an app.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pickapicka import updater

PKG_NAME = "Pickapicka-v0.11.0-macos-arm64.pkg"
EXE_NAME = "Pickapicka-0.11.0-windows-x64.exe"
PAYLOAD = b"not really an installer " * 5000


class FakeGitHub:
    """The latest-release endpoint and one asset, counting the asset's fetches."""

    def __init__(self, tag: str = "v0.11.0", payload: bytes = PAYLOAD, digest: str | None = None):
        self.tag, self.payload = tag, payload
        self.digest = digest or "sha256:" + hashlib.sha256(payload).hexdigest()
        self.asset_hits = 0
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if self.path == "/latest":
                    body = json.dumps(fake.release()).encode("utf-8")
                    ctype = "application/json"
                elif self.path == f"/download/{PKG_NAME}":
                    fake.asset_hits += 1
                    body, ctype = fake.payload, "application/octet-stream"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def release(self) -> dict:
        return {
            "tag_name": self.tag,
            "html_url": f"https://github.com/son-engr-kr/pickapicka/releases/tag/{self.tag}",
            "published_at": "2026-10-01T00:00:00Z",
            "assets": [
                {"name": EXE_NAME, "size": 10, "digest": "sha256:" + "0" * 64,
                 "browser_download_url": f"{self.base}/download/{EXE_NAME}"},
                {"name": PKG_NAME, "size": len(self.payload), "digest": self.digest,
                 "browser_download_url": f"{self.base}/download/{PKG_NAME}"},
            ],
        }

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture
def github():
    fakes: list[FakeGitHub] = []

    def make(**kw) -> FakeGitHub:
        fakes.append(FakeGitHub(**kw))
        return fakes[-1]
    yield make
    for f in fakes:
        f.close()


def _updater(tmp_path: Path, gh: FakeGitHub, kind: str = "macos-app", current: str = "0.10.0"):
    return updater.Updater(latest_url=f"{gh.base}/latest", update_dir=tmp_path / "updates",
                           pending_file=tmp_path / "pending.json", kind=kind, current=current)


# ----- versions and assets ----------------------------------------------------

def test_versions_compare_as_numbers_not_text() -> None:
    assert updater.parse_version("v0.10.0") == (0, 10, 0)
    assert updater.parse_version("0.9.1") < updater.parse_version("v0.10.0")
    assert updater.parse_version("1.0") < updater.parse_version("1.0.1")
    with pytest.raises(AssertionError):
        updater.parse_version("v0.11.0-rc1")


def test_the_right_installer_is_picked_for_each_kind_of_install() -> None:
    # The names v0.10.0 really shipped with.
    assets = [{"name": "Pickapicka-0.10.0-windows-x64.exe"}, {"name": "Pickapicka-v0.10.0-macos-arm64.pkg"}]
    assert updater.pick_asset("macos-app", assets)["name"].endswith(".pkg")
    assert updater.pick_asset("windows-app", assets)["name"].endswith(".exe")
    assert updater.pick_asset("uv-tool", assets) is None
    assert updater.pick_asset("homebrew", assets) is None


def test_the_kind_of_install_is_read_from_how_python_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert updater.install_kind() == "macos-app"
    monkeypatch.setattr(sys, "platform", "win32")
    assert updater.install_kind() == "windows-app"
    monkeypatch.delattr(sys, "frozen")
    tool = tmp_path / "uv" / "tools" / "pickapicka"
    tool.mkdir(parents=True)
    (tool / "uv-receipt.toml").write_text("[tool]\n")
    monkeypatch.setattr(sys, "prefix", str(tool))
    assert updater.install_kind() == "uv-tool"
    monkeypatch.setattr(sys, "prefix", "/opt/homebrew/Cellar/pickapicka/0.10.0/libexec/venv")
    assert updater.install_kind() == "homebrew"
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "checkout" / ".venv"))
    assert updater.install_kind() == "other"


# ----- checking and downloading -------------------------------------------------

def test_a_newer_release_is_downloaded_checked_and_kept(tmp_path, github) -> None:
    gh = github()
    u = _updater(tmp_path, gh)
    u.check()
    s = u.snapshot()
    assert s["status"] == "ready", s
    assert s["latest"]["version"] == "0.11.0"
    assert (tmp_path / "updates" / PKG_NAME).read_bytes() == PAYLOAD
    assert not list((tmp_path / "updates").glob("*.part"))
    assert s["progress"] is None


def test_a_download_that_does_not_match_its_checksum_is_thrown_away(tmp_path, github) -> None:
    gh = github(digest="sha256:" + "ab" * 32)
    u = _updater(tmp_path, gh)
    u.check()
    s = u.snapshot()
    assert s["status"] == "error" and "checksum" in s["error"], s
    assert not list((tmp_path / "updates").iterdir())


def test_an_installer_fetched_earlier_is_not_fetched_again(tmp_path, github) -> None:
    gh = github()
    _updater(tmp_path, gh).check()
    again = _updater(tmp_path, gh)
    again.check()
    assert again.snapshot()["status"] == "ready"
    assert gh.asset_hits == 1


def test_an_older_installer_is_cleared_away(tmp_path, github) -> None:
    (tmp_path / "updates").mkdir()
    (tmp_path / "updates" / "Pickapicka-v0.10.5-macos-arm64.pkg").write_bytes(b"old")
    _updater(tmp_path, github()).check()
    assert [p.name for p in (tmp_path / "updates").iterdir()] == [PKG_NAME]


def test_the_same_version_is_up_to_date(tmp_path, github) -> None:
    gh = github(tag="v0.10.0")
    u = _updater(tmp_path, gh)
    u.check()
    assert u.snapshot()["status"] == "current"
    assert gh.asset_hits == 0


def test_a_tool_install_is_told_the_command_and_downloads_nothing(tmp_path, github) -> None:
    gh = github()
    u = _updater(tmp_path, gh, kind="uv-tool")
    u.check()
    s = u.snapshot()
    assert s["status"] == "available" and s["command"] == "uv tool upgrade pickapicka"
    assert not s["can_install"]
    assert gh.asset_hits == 0


def test_being_offline_is_an_error_to_show_not_a_crash(tmp_path) -> None:
    u = updater.Updater(latest_url="http://127.0.0.1:9/latest", update_dir=tmp_path / "u",
                        pending_file=tmp_path / "p.json", kind="macos-app", current="0.10.0")
    u.check()
    s = u.snapshot()
    assert s["status"] == "error" and s["error"].startswith("Could not reach GitHub")


# ----- after the install ---------------------------------------------------------

def _pending(tmp_path: Path, to: str = "0.11.0") -> Path:
    p = tmp_path / "pending.json"
    p.write_text(json.dumps({"from": "0.10.0", "to": to, "at": "2026-10-01T10:00:00"}))
    return p


def test_the_version_that_comes_up_says_whether_the_update_took(tmp_path) -> None:
    exit_file = tmp_path / "exit.txt"
    exit_file.write_text("0\n\n", encoding="utf-8")
    r = updater.read_result(_pending(tmp_path), exit_file, current="0.11.0")
    assert r["ok"] and r["code"] == 0 and r["to"] == "0.11.0"
    assert r["url"].endswith("/releases/tag/v0.11.0")
    assert not (tmp_path / "pending.json").exists() and not exit_file.exists()
    assert updater.read_result(tmp_path / "pending.json", exit_file, current="0.11.0") is None


def test_a_declined_prompt_leaves_the_old_version_and_says_so(tmp_path) -> None:
    exit_file = tmp_path / "exit.txt"
    # As PowerShell writes it: a byte-order mark, then the code and the reason.
    exit_file.write_text("-1\nThis command cannot be run due to the error: The operation was canceled by the user.\n",
                         encoding="utf-8-sig")
    r = updater.read_result(_pending(tmp_path), exit_file, current="0.10.0")
    assert not r["ok"] and r["code"] == -1 and "canceled" in r["message"]


# ----- the install scripts ----------------------------------------------------------

AWKWARD = "사진 앱's \"업데이트\" $HOME `x`"


def test_the_mac_script_carries_awkward_paths_through_intact(tmp_path) -> None:
    pkg = tmp_path / AWKWARD / PKG_NAME
    prompt = f"Pickapicka is installing version 0.11.0 ({AWKWARD})."
    script = updater.mac_install_script(
        pkg=pkg, pid=4242, port=8765, exit_file=tmp_path / AWKWARD / "exit.txt",
        installed_app=Path("/Applications/Pickapicka.app"),
        running_app=Path("/Users/me/다운로드/Pickapicka.app"), prompt=prompt)
    osa_line = next(line for line in script.splitlines() if "osascript" in line)
    args = shlex.split(osa_line.split("$(", 1)[1].rsplit(" 2>&1)", 1)[0])
    assert args[0] == "/usr/bin/osascript"
    assert args[-2:] == [str(pkg), prompt]
    assert [a for a in args[1:-2] if a != "-e"] == list(updater.MAC_APPLESCRIPT)
    assert "kill -0 4242" in script and "--port 8765" in script
    if shutil.which("sh"):
        path = tmp_path / "install.sh"
        path.write_text(script, encoding="utf-8")
        subprocess.run(["sh", "-n", str(path)], check=True)


@pytest.mark.skipif(not shutil.which("osacompile"), reason="osacompile is macOS's")
def test_the_applescript_compiles(tmp_path) -> None:
    cmd = ["osacompile", "-o", str(tmp_path / "x.scpt")]
    for line in updater.MAC_APPLESCRIPT:
        cmd += ["-e", line]
    subprocess.run(cmd, check=True, capture_output=True)


def test_the_windows_script_quotes_for_powershell(tmp_path) -> None:
    installer = tmp_path / AWKWARD / EXE_NAME
    script = updater.windows_install_script(
        installer=installer, pid=4242, port=8765, exit_file=tmp_path / "exit.txt",
        exe=Path(r"C:\Program Files\Pickapicka\pickapicka.exe"))
    assert updater._ps(str(installer)) in script
    assert "'사진 앱''s" in updater._ps(AWKWARD)
    assert "-Verb RunAs -Wait -PassThru" in script and "'/SILENT'" in script
    assert "Get-Process -Id 4242" in script and "'--port','8765'" in script


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell's parser is Windows'")
def test_the_windows_script_parses(tmp_path) -> None:
    path = tmp_path / "install.ps1"
    path.write_text(updater.windows_install_script(
        installer=tmp_path / AWKWARD / EXE_NAME, pid=1, port=8765, exit_file=tmp_path / "exit.txt",
        exe=Path(r"C:\Program Files\Pickapicka\pickapicka.exe")), encoding="utf-8-sig")
    check = ("$e = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
             f"'{path}', [ref]$null, [ref]$e); $e.Count")
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", check],
                         check=True, capture_output=True, text=True).stdout.strip()
    assert out == "0", out


def test_starting_the_install_hands_over_to_a_detached_script(tmp_path, github, monkeypatch) -> None:
    u = _updater(tmp_path, github())
    u.check()
    bundle = tmp_path / "Pickapicka.app"
    (bundle / "Contents" / "MacOS").mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(bundle / "Contents" / "MacOS" / "pickapicka"))
    started = []
    monkeypatch.setattr(updater.subprocess, "Popen", lambda args, **kw: started.append((args, kw)))
    u.start_install(port=8765)
    assert u.snapshot()["status"] == "installing"
    pending = json.loads((tmp_path / "pending.json").read_text())
    assert pending["from"] == "0.10.0" and pending["to"] == "0.11.0"
    (args, kw), = started
    assert args[0] == "/bin/sh" and kw["start_new_session"]
    script = Path(args[1]).read_text(encoding="utf-8")
    assert shlex.quote(str(tmp_path / "updates" / PKG_NAME)) in script
    assert shlex.quote(str(bundle.resolve())) in script
    with pytest.raises(AssertionError):
        u.start_install(port=8765)      # once is enough


@pytest.mark.skipif(os.name != "posix", reason="TIME_WAIT only blocks a bind without SO_REUSEADDR on POSIX")
def test_the_app_comes_back_on_its_own_port_after_quitting() -> None:
    """The page waits for the relaunched app on the port it was using, so the
    connections the old one leaves in TIME_WAIT must not push it elsewhere."""
    from pickapicka.server import _pick_free_port
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    conn, _ = listener.accept()
    conn.close()          # the server side closes first, so it is the one in TIME_WAIT
    client.close()
    listener.close()
    assert _pick_free_port("127.0.0.1", port) == port


# Delegated to pytest, as test_redeye does: these take tmp_path and monkeypatch.
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
