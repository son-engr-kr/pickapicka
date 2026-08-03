# Picture Classifier

Score photos for blur, exposure, and face presence — then cull *and edit* them
fast in a local web viewer with face-cluster-aware sorting and pick/review/reject
decisions.

Designed for the post-shoot triage workflow on a personal collection of a few
hundred to a few thousand JPEGs (and RAWs). Runs entirely on your machine; no
network.

## Features

- **In-app photo editing**: adjust any photo right from the picking screen —
  Exposure, Contrast, Highlights/Shadows/Whites/Blacks, Temperature/Tint,
  Vibrance/Saturation, Clarity/Sharpen/Vignette, and a draggable **tone curve**,
  with a live before/after preview and an **Auto** button. Fully
  non-destructive — originals are never modified; adjustments bake in only on
  export. Save **global presets** and **apply an edit/preset to many photos at
  once** (a whole scene, all picks, or a hand-picked selection).
- **Crop & straighten**: drag a box on the photo, or lock it to Free / Original /
  1:1 / 4:5 / 5:4 / 2:3 / 3:2 / 16:9 — corner and edge grips, a rule-of-thirds
  grid, and the output size in pixels as you go. **Straighten** turns the frame
  and cuts back to the largest rectangle of the same aspect that still fits, so
  levelling a horizon never leaves blank corners; positive levels one drooping
  to the right. Both are non-destructive, and you can crop at any point in the
  process: the grade — masks included — is measured against the **original**
  frame and the crop is taken out of the result, so cropping never drags a mask
  off the thing you drew it on or changes its size. The watermark is the one
  thing that follows the crop, since a signature belongs on the picture you end
  up with. Detection boxes are stored against the original frame but drawn
  through the crop, so they stay on the car they were found on; one the crop cuts
  out stops being drawn.
- **Local adjustments (masks)**: grade just part of a photo. Draw a **radial**
  ellipse (drag to place, handles to resize and rotate), a **gradient** for
  skies and foregrounds, or **brush** an area freehand (Alt to erase). Each mask
  carries its own full slider set plus Feather, Amount and an inside/outside
  toggle; stack up to 16 per photo. Press `\` to tint the affected area.
- **Creative effects**, global or masked: **defocus** (disc-shaped lens blur, so
  highlights bloom into circles — put it on an inverted radial for fake shallow
  depth of field), **motion** blur with an angle (mask the car out of it and you
  have a panned shot), **glow** for headlights and low sun, and **mosaic** for a
  number plate. All sized relative to the frame, so the preview is the export.
- **1:1 editing**: the editor previews fitted by default, but fit/50/100/200%
  renders the visible window from the **full-resolution original**, so you can
  judge real sharpness while grading. Scroll to zoom, drag to pan (space or
  middle-drag to pan past a mask).
- **Presets**, including a built-in car set: glossy paint, studio white, night
  neon, golden hour, plus mask-carrying ones for a speed pan, background bokeh
  and plate blurring. Built-ins are read-only; tweak and save your own.
  Presets load **additively** by default, so a local preset drops its mask onto
  whatever you already have instead of wiping it — stack "background bokeh" and
  "blur a plate" on top of your own grade, and a slider the preset leaves
  neutral keeps your value. Switch to *replace* for a clean slate. Bulk apply
  has the same choice, so you can add one mask to every pick without touching
  their individual grades.
- **Watermarks**: stamp a signature and the shooting info in one of five styles
  (minimal, gradient bar, plate, corner rule, filmstrip caption), anywhere in
  the frame. Lines are templates — `{name}`, `{camera}`, `{lens}`, `{focal}`,
  `{aperture}`, `{shutter}`, `{iso}`, `{date}`, `{file}` — and a token that has
  no value takes its separator with it, so a lens-less frame prints
  "α7C II", never "α7C II · ". Sized against the frame, so the preview is the
  export; skipped on grid thumbnails where it would just be noise.
- **Camera names, spelled properly**: EXIF stores codenames, so `ILCE-7CM2`
  becomes **α7C II** and `NIKON Z 6_2` becomes **Z 6II**. ~100 bodies across
  Sony, Canon, Nikon, Fujifilm, Panasonic, OM System, Leica and Ricoh, picked
  from a searchable list or overridden by hand; unknown bodies still read
  sensibly instead of vanishing.
- **Shooting info**: camera, lens, focal length, aperture, shutter and ISO are
  read once at scoring time and shown in the viewer — and available to the
  watermark. RAW files get theirs from the embedded preview.
- **Focus peaking** (`K`): tints what is actually in focus, on the grid tiles
  and in the viewer. The badness score says which frame is sharpest overall;
  this says *what* is sharp, which is the real question when one frame nailed
  the headlight and the next nailed the badge. Sharpness is measured as how
  much an edge collapses under a small extra blur, which is independent of how
  much contrast it has — so a hard-lit but defocused boundary is not mistaken
  for a sharp one, and a frame with nothing in focus lights up nothing.
  Sensitivity is tight / normal / loose.
- **Loupe** (`L`): in the single-photo viewer, hovering shows a live magnified
  panel of whatever is under the cursor, parked in the letterbox margin so it
  stays off the photo (and dodging to the other side when it can't). 1:1 / 2:1 /
  4:1 against the source pixels, nearest-neighbour — the point is to see the
  softness, not to smooth it away.
- **RAW support**: RAW files are first-class in the grid. When a shot exists as
  both a RAW and a JPEG, the RAW is preferred; edit it and export a JPEG.
- **Workspaces & projects**: pick a workspace folder, then create projects
  inside it *by name* (DaVinci-Resolve style). The landing page lists the
  projects in each workspace. Photos are referenced by path; if a photo folder
  moves, the app offers to **re-link** it (matching by folder structure, then by
  file name). **Deleting a project never touches your photos**: it renames the
  project folder to `<name>.deleted-<timestamp>` and hides it from the list —
  decisions and edits stay inside, so renaming the folder back restores it.
- **Per-photo scoring**: Laplacian blur, brightness exposure, optional
  closed-eye detection.
- **Per-scene auto-suggestion**: top 30% pick / middle review / bottom 30%
  reject, normalized within each scene.
- **HDR bracket auto-merge**: detects auto-exposure brackets from EXIF and
  exposure-fuses each into one photo, with a tunable real-estate "look"
  (shadow lift, local contrast, saturation) and an HDR-vs-0 EV compare toggle.
- **Subject detection (cars, pets, anything COCO)**: tell a project what it is
  a shoot *of* — at creation, or any time afterwards: *↻ rescore* asks, since
  changing it is what a re-score is for — and object detection joins the scoring — frames missing the
  subject sort down, prominence counts, and **sharpness is measured on the
  subject instead of the whole frame** (so a panned or bokeh'd shot of a tack-
  sharp car stops reading as blurry). Boxes overlay the grid and viewer with
  `B`, and you can filter by class. Uses YOLOX-tiny (Apache-2.0) on the
  `onnxruntime` already in the stack — no PyTorch; ~20 MB downloaded on first
  use.
- **Subject grouping**: look-alike subjects are grouped so you can filter to
  "just this car". Unlike the person clusters this is *appearance*-based
  (colour and how it sits on the shape), because no reliable consumer vehicle
  re-identification model exists — two same-colour, same-shape cars will land in
  one group, and groups are renameable/hideable for exactly that reason. It can
  be switched off, and switching it off clears what it produced.
- **Grouping settings** (⚙ next to *↻ group*): both passes are configurable per
  project and remembered — how close two faces must be to count as one person
  and the fewest faces that make one, and for subjects the same plus a minimum
  size and confidence so background traffic is ignored. Loosening the subject
  distance from 0.15 to 0.45 on the same shoot goes from 1 group to 3, so it is
  worth a try when the grouping looks wrong. Settings survive a re-score.
- **One button for scoring and grouping**: *↻ rescore* opens a dialog asking
  what to detect, then scores and groups in one run. Scoring recomputes face
  embeddings, which discards every group, so grouping follows automatically
  rather than leaving a project with faces and nobody grouped into them. *↻
  group* on its own re-groups without re-scoring, which is what you want after
  changing a grouping setting. The Subjects ⚙ is for renaming and hiding groups
  and never re-scores — doing so would throw away the names you just typed.
- **Face clustering**: detects faces with `insightface` and clusters them
  per-person via DBSCAN on embeddings.
- **Drag-and-drop people priority**: rank face clusters by importance; photos
  containing higher-priority people sort to the top within each scene.
- **Exclude clusters**: hide irrelevant clusters (background people, false
  positives) from sorting and chips.
- **Scene grouping**: by folder structure, or by EXIF capture-time gaps
  (configurable in minutes). Switch any time without re-scoring.
- **Filter by decision or by edited**: alongside all/undecided/pick/review/reject
  there is `edited`, which shows only the photos you have actually graded —
  useful for a last pass over your own work, or for finding what still needs it.
- **Reopens where you left it**: the filter, grid layout, page and scene are
  remembered per project, so a project opens on the shot you were looking at
  rather than on page 1 of everything. Stored per project in
  `~/.picture-classifier/state.json`; a scene that has since been regrouped away
  falls back to the first one.
- **Bulk actions**: reject all undecided in a scene; export all picks to a
  folder (preserving structure or flattened).
- **Download the selection** (`D`): saves the selected photos straight to your
  Downloads folder — no dialog. Edits are baked and RAW is rendered, so what
  lands there is the photo as you graded it. One photo goes in loose, several
  go into a dated subfolder; nothing already there is ever overwritten.
- **Keyboard-driven culling**: `R` reject, `V` review, `A`/`P` pick, `U` undo,
  `E` edit, `X` toggle-select, arrow keys to navigate, `Enter` to open the modal
  viewer, `[`/`]` for pages, `B` subject boxes, `K` focus peaking, `L` loupe.
  In the editor: `R`/`G`/`B` add a radial/gradient/brush mask, `\` shows the
  mask, `Del` removes it, `C` holds the original, `F` toggles fit ↔ 100%.
  Hovering any of these buttons shows the action and its shortcut.
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

This starts the server at <http://127.0.0.1:8765> and opens a landing page.
Pick a **workspace** folder (or use the default), then create a project inside
it by name and point it at your photos. The landing lists every project in the
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

Supported extensions: `.jpg`, `.jpeg`, `.png` plus common RAW formats
(`.cr2`/`.cr3`, `.nef`, `.arw`, `.raf`, `.rw2`, `.orf`, `.dng`, …),
case-insensitive. When a shot has both a RAW and a same-named JPEG, the RAW is
used. If your RAWs live in a separate `RAW/` tree, set the RAW subfolder under
**Advanced** in the new-project wizard. Videos and sidecars are ignored.

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
