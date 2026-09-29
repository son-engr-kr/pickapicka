"""Build the website into a folder, ready for GitHub Pages.

    python3 site/build.py OUT_DIR

Standard library only, so the Pages workflow needs nothing installed. The page
is site/index.html with the latest release filled in: its version, its date,
and a direct link and size for each installer. That is why the site is rebuilt
whenever a release finishes building (.github/workflows/pages.yml): the links
are baked in rather than looked up by every visitor's browser.

The images, the icons and the font are the repository's own, copied rather
than kept twice. The promo video is the one the README shows, fetched here so
the site serves it from its own address.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
REPO = "son-engr-kr/pickapicka"
REPO_URL = f"https://github.com/{REPO}"
SITE_URL = "https://son-engr-kr.github.io/pickapicka/"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
# The same installers the app's updater looks for (updater.ASSET_SUFFIX).
MAC_SUFFIX = "-macos-arm64.pkg"
WIN_SUFFIX = "-windows-x64.exe"
PROMO_URL = "https://github.com/user-attachments/assets/0e347ade-1275-46f8-acc5-1e2b6cb20fc5"
GUIDE_YOUTUBE_ID = "y_Lt25yDufM"
COPIES = {
    "style.css": SITE / "style.css",
    "media/promo-poster.jpg": SITE / "media" / "promo-poster.jpg",
    "images/grid.jpg": ROOT / "docs" / "images" / "grid.jpg",
    "images/editor.jpg": ROOT / "docs" / "images" / "editor.jpg",
    "images/guide.jpg": ROOT / "docs" / "images" / "guide.jpg",
    "icons/favicon-32.png": ROOT / "src" / "pickapicka" / "web" / "icons" / "favicon-32.png",
    "icons/apple-touch-icon.png": ROOT / "src" / "pickapicka" / "web" / "icons" / "apple-touch-icon.png",
    "fonts/Poppins-SemiBold.ttf": ROOT / "src" / "pickapicka" / "web" / "fonts" / "Poppins-SemiBold.ttf",
    "fonts/OFL.txt": ROOT / "src" / "pickapicka" / "web" / "fonts" / "OFL.txt",
}


def get(url: str, accept: str) -> urllib.request.addinfourl:
    headers = {"Accept": accept, "User-Agent": "pickapicka-site-build"}
    token = os.environ.get("GITHUB_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)


def one_asset(assets: list[dict], suffix: str) -> dict:
    found = [a for a in assets if a["name"].endswith(suffix)]
    assert len(found) == 1, f"the latest release should have one *{suffix}, has {[a['name'] for a in found]}"
    return found[0]


def megabytes(n: int) -> str:
    return f"{round(n / 1_000_000)} MB"


def main() -> None:
    assert len(sys.argv) == 2, __doc__
    out = Path(sys.argv[1]).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    with get(LATEST_URL, "application/vnd.github+json") as r:
        release = json.loads(r.read().decode("utf-8"))
    mac = one_asset(release["assets"], MAC_SUFFIX)
    win = one_asset(release["assets"], WIN_SUFFIX)
    published = datetime.fromisoformat(release["published_at"].replace("Z", "+00:00"))
    fields = {
        "SITE_URL": SITE_URL,
        "REPO_URL": REPO_URL,
        "VERSION": release["tag_name"].lstrip("v"),
        "RELEASE_URL": release["html_url"],
        "RELEASE_DATE": f"{published.day} {published:%B %Y}",
        "MAC_URL": mac["browser_download_url"],
        "MAC_SIZE": megabytes(mac["size"]),
        "WIN_URL": win["browser_download_url"],
        "WIN_SIZE": megabytes(win["size"]),
        "GUIDE_YOUTUBE_ID": GUIDE_YOUTUBE_ID,
    }
    html = (SITE / "index.html").read_text(encoding="utf-8")
    for key, value in fields.items():
        html = html.replace("{{" + key + "}}", value)
    left = re.findall(r"\{\{[A-Z_]+\}\}", html)
    assert not left, f"placeholders with no value: {sorted(set(left))}"
    (out / "index.html").write_text(html, encoding="utf-8")

    for rel, src in COPIES.items():
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    video = out / "media" / "promo.mp4"
    with get(PROMO_URL, "video/mp4") as r:
        assert r.headers.get_content_type() == "video/mp4", f"the promo came back as {r.headers.get_content_type()}"
        video.write_bytes(r.read())
    assert video.stat().st_size > 1_000_000, f"the promo is only {video.stat().st_size} bytes"

    # Served as files, not as a Jekyll site.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"{out}: version {fields['VERSION']}, macOS {fields['MAC_SIZE']}, Windows {fields['WIN_SIZE']}")


if __name__ == "__main__":
    main()
