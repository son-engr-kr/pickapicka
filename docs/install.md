# Install and run

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

This starts the server at <http://127.0.0.1:8765> and opens a landing page.
Pick a **workspace** folder (or use the default; the first launch walks
through this), then create a project inside it by name and point it at your
photos. The landing lists every project in the
current workspace; you can also open a project folder directly, and recent
projects are listed under that section.

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

For a subject-driven shoot (a car meet, a dog session), add `--subjects`. It
takes a preset — `vehicle`, `person`, `pet`, `bike` — or any comma-separated
list of COCO class names, and `pcls cluster --subjects` then groups the
look-alikes:

```bash
uv run pcls score /path/to/photos -o picks.json --subjects vehicle
uv run pcls score /path/to/photos -o picks.json --subjects car,truck
uv run pcls cluster picks.json --subjects
```

A plain re-score keeps whatever the project was last scored with, so you only
pass `--subjects` when you want to change it.

`pcls score --help` for all options.
