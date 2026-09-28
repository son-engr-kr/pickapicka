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
[Releases](https://github.com/son-engr-kr/pickapicka/releases),
double-click it, and follow the installer — it puts **Pickapicka.app**
into `/Applications`. Launch it and the landing page opens in your browser
automatically.

Apple Silicon (arm64) only. The package is unsigned, so the first launch
needs **Right-click → Open → Open** to clear Gatekeeper.

### Windows (no terminal needed)

Grab the latest `Pickapicka-*-windows-x64.exe` from
[Releases](https://github.com/son-engr-kr/pickapicka/releases),
run the installer, and launch **Pickapicka** from the Start menu.

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
uv tool install git+https://github.com/son-engr-kr/pickapicka
pickapicka serve
```

To upgrade later: `uv tool upgrade pickapicka`.

### Homebrew (macOS / Linux)

```bash
brew tap son-engr-kr/pickapicka
brew install pickapicka
pickapicka serve
```

### From source

```bash
git clone https://github.com/son-engr-kr/pickapicka.git
cd pickapicka
uv sync
uv run pickapicka serve
```

### Upgrading from Picture Classifier

Up to 0.9.0 the app was called Picture Classifier. Your projects, workspaces,
presets and remembered views all carry over: the first launch moves the app's
own files to their new place (see [Projects and files](projects.md)), and your
workspace folder stays where it is, even if it is named
`PictureClassifier-Projects`. The command is now `pickapicka` rather than `pcls`.

- **macOS**: install the new `.pkg`. It adds **Pickapicka.app** next to the old
  app rather than over it, so drag **Picture Classifier.app** to the Trash.
- **Windows**: run the new installer. It upgrades the old install in place, and
  the Start menu entry becomes **Pickapicka**.
- **`uv tool`**: the package has a new name, so reinstall rather than upgrade:

  ```bash
  uv tool uninstall picture-classifier
  uv tool install git+https://github.com/son-engr-kr/pickapicka
  ```

- **Homebrew**: the tap has a new name too:

  ```bash
  brew uninstall picture-classifier
  brew untap son-engr-kr/picture-classifier
  brew tap son-engr-kr/pickapicka
  brew install pickapicka
  ```

## Usage

### Open the landing page

```bash
uv run pickapicka serve
```

This starts the server at <http://127.0.0.1:8765> and opens a landing page.
Pick a **workspace** folder (or use the default; the first launch walks
through this), then create a project inside it by name and point it at your
photos. The landing lists every project in the
current workspace; you can also open a project folder directly, and recent
projects are listed under that section.

### Open an existing project directly

```bash
uv run pickapicka serve /path/to/picks.json
```

### From the CLI

You can also score and cluster from the terminal:

```bash
uv run pickapicka score /path/to/photos -o /path/to/picks.json
uv run pickapicka cluster /path/to/picks.json
uv run pickapicka report /path/to/picks.json
```

For a subject-driven shoot (a car meet, a dog session), add `--subjects`. It
takes a preset — `vehicle`, `person`, `pet`, `bike` — or any comma-separated
list of COCO class names, and `pickapicka cluster --subjects` then groups the
look-alikes:

```bash
uv run pickapicka score /path/to/photos -o picks.json --subjects vehicle
uv run pickapicka score /path/to/photos -o picks.json --subjects car,truck
uv run pickapicka cluster picks.json --subjects
```

A plain re-score keeps whatever the project was last scored with, so you only
pass `--subjects` when you want to change it.

`pickapicka score --help` for all options.
