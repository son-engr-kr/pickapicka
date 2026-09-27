"""End-to-end check of a built app: create a project and score it, on paths a
Korean Windows user really has.

    uv run python packaging/smoke_score.py <path to the built executable>

Starting and answering /api/state (the build workflows' first smoke test) says
nothing about opening photos, and 0.7.1 shipped unable to score any folder
whose path held Korean: cv2.imread cannot open such a path on Windows. Here the
home folder, the workspace and the photo folder all have Korean names with
spaces, like "OneDrive\\바탕 화면\\그리스", and HOME / USERPROFILE point the app
there, so the face models (under ~/.insightface) and the default workspace land
on such a path too. The project is then created, scored and read back through
the same endpoints the UI uses.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

PORT = 8798
BASE = f"http://127.0.0.1:{PORT}"
START_TIMEOUT = 180     # seconds for the server to come up
SCORE_TIMEOUT = 1200    # the first scoring downloads the face models


def call(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_bytes(path: str) -> bytes:
    with urllib.request.urlopen(BASE + path, timeout=60) as r:
        assert r.status == 200, f"{path}: HTTP {r.status}"
        return r.read()


def make_photos(root: Path) -> int:
    """Five photos in two scene folders, with capture times, as a camera would."""
    rng = np.random.default_rng(0)
    n = 0
    for scene, count in (("첫째 날", 3), ("둘째 날", 2)):
        (root / scene).mkdir(parents=True)
        for i in range(count):
            h, w = 480, 720
            ramp = np.linspace(40, 220, w, dtype=np.float32)[None, :, None]
            img = np.clip(ramp + rng.normal(0, 18, (h, w, 3)), 0, 255).astype(np.uint8)
            exif = Image.Exif()
            exif.get_ifd(0x8769)[0x9003] = f"2026:05:0{1 + n // 3} 10:{10 + i:02d}:00"
            name = f"사진 {i + 1}.JPG" if i % 2 == 0 else f"DSCF{7454 + i}.jpg"
            Image.fromarray(img).save(root / scene / name, quality=90, exif=exif)
            n += 1
    return n


def main() -> None:
    # A CI log on Windows is not a console, so Python would encode what this
    # prints in the ANSI code page, which has no Korean.
    sys.stdout.reconfigure(encoding="utf-8")
    exe = Path(sys.argv[1]).resolve()
    assert exe.is_file(), f"no executable at {exe}"
    top = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()) / "스모크 테스트"
    if top.exists():
        shutil.rmtree(top)      # this script's own folder, left by a previous run
    home = top / "사용자 홈"
    photos = home / "OneDrive" / "바탕 화면" / "그리스"
    workspace = home / "사진 프로젝트"
    expected = make_photos(photos)
    print(f"photos: {photos} ({expected})")

    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    proc = subprocess.Popen([str(exe), "serve", "--port", str(PORT)], env=env)
    try:
        deadline = time.time() + START_TIMEOUT
        while True:
            assert proc.poll() is None, f"the app exited with code {proc.returncode}"
            try:
                call("GET", "/api/state")
                break
            except OSError:
                assert time.time() < deadline, "the app never answered on /api/state"
                time.sleep(1)

        shots = call("POST", "/api/folder/shots", {"path": str(photos)})
        assert shots["shots"] == expected and shots["untimed"] == 0, shots
        print(f"grouping preview: {shots['folders']}")

        call("POST", "/api/project/create", {
            "name": "그리스", "workspace_dir": str(workspace), "photo_dir": str(photos),
            "scene_grouping_mode": "folder",
        })
        deadline = time.time() + SCORE_TIMEOUT
        last = ""
        while True:
            opening = call("GET", "/api/state").get("opening") or {}
            line = f"{opening.get('phase')} {opening.get('idx')}/{opening.get('total')} {opening.get('message') or ''}"
            if line != last:
                print("  ", line)
                last = line
            if not opening.get("running"):
                break
            assert time.time() < deadline, "scoring did not finish in time"
            time.sleep(2)
        assert not opening.get("error"), f"opening the project failed: {opening['error']}"

        db = call("GET", "/api/db")
        got = db["photos"]
        assert len(got) == expected, f"scored {len(got)} of {expected} photos"
        scenes = {p["scene"] for p in got}
        assert scenes == {"첫째 날", "둘째 날"}, scenes
        enc = lambda rel: "/".join(urllib.parse.quote(s, safe="") for s in rel.split("/"))
        for p in got:
            for kind in ("thumb", "img"):
                data = fetch_bytes(f"/{kind}/{enc(p['rel_path'])}")
                assert len(data) > 1000, f"/{kind}/ of {p['rel_path']} returned {len(data)} bytes"
        print(f"ok: {len(got)} photos scored into {sorted(scenes)}, thumbnails and images served")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
