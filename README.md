# Picture Classifier

Score photos for blur, exposure, and face presence — then cull them fast in a
local web viewer with face-cluster-aware sorting and pick/review/reject
decisions.

Designed for the post-shoot triage workflow on a personal collection of a few
hundred to a few thousand JPEGs. Runs entirely on your machine; no network.

## Features

- **Per-photo scoring**: Laplacian blur, brightness exposure, optional
  closed-eye detection.
- **Per-scene auto-suggestion**: top 30% pick / middle review / bottom 30%
  reject, normalized within each scene.
- **HDR bracket auto-merge**: detects auto-exposure brackets from EXIF and
  exposure-fuses each into one photo, with a tunable real-estate "look"
  (shadow lift, local contrast, saturation) and an HDR-vs-0 EV compare toggle.
- **Face clustering**: detects faces with `insightface` and clusters them
  per-person via DBSCAN on embeddings.
- **Drag-and-drop people priority**: rank face clusters by importance; photos
  containing higher-priority people sort to the top within each scene.
- **Exclude clusters**: hide irrelevant clusters (background people, false
  positives) from sorting and chips.
- **Scene grouping**: by folder structure, or by EXIF capture-time gaps
  (configurable in minutes). Switch any time without re-scoring.
- **Bulk actions**: reject all undecided in a scene; export all picks to a
  folder (preserving structure or flattened).
- **Keyboard-driven culling**: `R` reject, `V` review, `A`/`P` pick, `U` undo,
  arrow keys to navigate, `Enter` to open the modal viewer, `[`/`]` for pages.
- **Project history**: recent folders are remembered so you can reopen them
  from the landing page.

## Requirements

- Prebuilt installers for macOS (Apple Silicon) and Windows (x64) — see below
- For the source / `uv tool` install: Python 3.12+ and
  [uv](https://github.com/astral-sh/uv). Linux is supported this way.

## Install

Pick whichever installer you prefer. The first run downloads the
`insightface` `buffalo_l` model (~280 MB) into `~/.insightface/`.

### macOS (no terminal needed)

Grab the latest `.pkg` from
[Releases](https://github.com/son-engr-kr/picture-classifier/releases),
double-click it, and follow the installer — it puts **Picture Classifier.app**
into `/Applications`. Launch it and the landing page opens in your browser
automatically.

Apple Silicon (arm64) only. The package is unsigned, so the first launch
needs **Right-click → Open → Open** to clear Gatekeeper.

### Windows (no terminal needed)

Grab the latest `Picture-Classifier-*-windows-x64.exe` from
[Releases](https://github.com/son-engr-kr/picture-classifier/releases),
run the installer, and launch **Picture Classifier** from the Start menu.

x64 only. The installer is unsigned, so SmartScreen warns on first run —
click **More info → Run anyway**.

### `uv tool` (cross-platform — macOS, Linux, Windows)

If you don't have [uv](https://github.com/astral-sh/uv) yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then:

```bash
uv tool install git+https://github.com/son-engr-kr/picture-classifier
pcls serve
```

To upgrade later: `uv tool upgrade picture-classifier`.

### Homebrew (macOS / Linux)

```bash
brew tap son-engr-kr/picture-classifier
brew install picture-classifier
pcls serve
```

### From source

```bash
git clone https://github.com/son-engr-kr/picture-classifier.git
cd picture-classifier
uv sync
uv run pcls serve
```

## Usage

### Open the landing page

```bash
uv run pcls serve
```

This starts the server at <http://127.0.0.1:8765> and opens a landing page
where you can pick a photo folder. Recent projects are listed there too.

### Open an existing project directly

```bash
uv run pcls serve /path/to/picks.json
```

### From the CLI

You can also score and cluster from the terminal:

```bash
uv run pcls score /path/to/photos -o /path/to/picks.json
uv run pcls cluster /path/to/picks.json
uv run pcls report /path/to/picks.json
```

`pcls score --help` for all options.

## Folder layouts

All three of these are supported. Switch scene grouping (by folder vs. by
time gap) inside the app.

```
# Flat
my-photos/
  IMG_0001.JPG
  IMG_0002.JPG
  picks.json   (auto-created)
```

```
# Pre-grouped folders (each subfolder becomes a scene)
wedding-2025/
  Scene_001/
    IMG_0001.JPG
  Scene_002/
    IMG_0010.JPG
  picks.json
```

```
# Project root with a JPEG subfolder (set "JPEG subfolder" under Advanced)
shoot/
  RAW/
  JPEG/
    Scene_001/
      IMG_0001.JPG
  picks.json   (at shoot/)
```

Supported extensions: `.jpg`, `.jpeg`, `.png` (case-insensitive). Other files
(videos, RAW, sidecars) are ignored.

## How decisions are persisted

Everything lives in `<photo_dir>/picks.json` next to your images. Re-scoring
preserves your pick/review/reject decisions by relative path. Re-clustering
resets cluster labels and priorities (face indices change), but per-photo
decisions are kept.

Caches (`picks.json.thumbs/`, `picks.json.faces/`,
`picks.json.embeddings.npy`) are recreated on demand and safe to delete.

## Releasing (maintainer notes)

A release is cut by pushing a `v*` tag. Three GitHub Actions fire on it and
all attach their output to the **same** GitHub Release:

- **Build macOS app** — builds the `.app` and packages a `.pkg`.
- **Build Windows app** — builds the PyInstaller bundle and wraps it in an
  Inno Setup installer (`setup.exe`).
- **Update Homebrew tap** — refreshes the
  [tap](https://github.com/son-engr-kr/homebrew-picture-classifier) formula.

```bash
# 1. bump "version" in pyproject.toml, then:
git commit -am "chore: bump to 0.1.4"
git tag v0.1.4
git push origin main v0.1.4
```

The tag push triggers all three workflows; build artifacts publish to the
GitHub Release automatically — no manual upload. To dry-run a platform build
*without* cutting a release, use **Actions → Build … app → Run workflow**
(workflow_dispatch); it uploads an artifact instead of publishing a release.

The Homebrew workflow needs a `TAP_TOKEN` repository secret — a fine-grained
PAT with `Contents: Write` on `son-engr-kr/homebrew-picture-classifier`.

## License

MIT — see [LICENSE](LICENSE).
