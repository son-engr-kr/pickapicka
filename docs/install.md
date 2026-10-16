# Install and run

## Requirements

- Prebuilt installers for macOS (Apple Silicon) and Windows (x64) — see below
- For the source / `uv tool` install: Python 3.12+ and
  [uv](https://github.com/astral-sh/uv). Linux is supported this way.

## Install

Pick whichever installer you prefer. The first run downloads the
`insightface` `buffalo_l` model (~280 MB) into `~/.insightface/`.

### macOS

Apple Silicon (arm64), macOS 12 or later. The easiest way is one line in
Terminal, which installs without macOS stopping it:

1. **Open Terminal**: press ⌘ Space, type `Terminal`, press Return.
2. **Paste this line and press Return:**

   ```bash
   curl -fsSL https://son-engr-kr.github.io/pickapicka/install.sh | bash
   ```

3. **Type your Mac's login password** when it asks, and press Return.
   Nothing appears as you type; that is normal. The first time, macOS shows
   a WARNING about sudo before it asks, which is its usual note.

It puts **Pickapicka.app** into `/Applications` and opens it, and the landing
page opens in your browser. After that the app updates itself.

What the line runs is [`site/install.sh`](../site/install.sh), with each
release's link and checksum filled in by `site/build.py`. It downloads the
same `.pkg` as the Releases page, checks it against the SHA-256 GitHub
publishes for it, and installs it with macOS's own `installer`. Why this gets
past the warning: a browser marks what it downloads as quarantined, and
Gatekeeper checks only marked files; `curl` adds no mark. The in-app updater
works the same way.

**Or download the `.pkg` yourself** from
[Releases](https://github.com/son-engr-kr/pickapicka/releases) and
double-click it. The package is not signed by Apple, so the first time it is
opened macOS stops it ("Apple could not verify … is free of malware"):

- **macOS 15 (Sequoia) and later**: click **Done**, open **System Settings →
  Privacy & Security**, scroll down to *Security*, and click **Open Anyway**
  beside the package's name. It stays there for about an hour after the
  attempt. Confirm with your password, then open the package again and choose
  **Open Anyway**.
- **macOS 12 to 14**: **right-click → Open → Open**.
- Or, in Terminal, take the download's quarantine mark off and open it as
  usual: `xattr -d com.apple.quarantine ~/Downloads/Pickapicka-*.pkg`.

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

## Updating

The installed app keeps itself current. A little after it opens, and every few
hours while it runs, it asks GitHub for the newest release, downloads the
installer in the background, and checks it against the SHA-256 checksum GitHub
publishes for it; a download that does not match is thrown away. When one is
ready, **Update to …** appears in the top bar. Clicking it restarts the app into
the installer, which asks for the system's consent once: your password on
macOS, the "allow this app to make changes" prompt on Windows. The app then
opens again by itself, back in the project you were in, and says whether the
update took. Say no to the prompt and the version you had opens again.

Because the app downloads the installer itself rather than through a browser,
the unsigned-app warnings (Gatekeeper's Open Anyway, SmartScreen's Run
anyway) do not come back for updates.

The **ⓘ** button in the top bar shows the version you have and the newest
release, with its date and what's new in it, and checks again as it opens, as
Check for Updates does in other apps. Its **Download page** goes to the
website's downloads, for installing by hand; with a new version downloaded it
also has **Restart and update**.

The check sends a request to GitHub and nothing else; nothing about your photos
or projects leaves the machine. **Preferences → Updates** turns automatic checks
off, and **Check now** and the ⓘ button work either way.

A `uv tool` or Homebrew install is told when a new version is out, with the
command that updates it (`uv tool upgrade pickapicka`, `brew upgrade
pickapicka`).

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
