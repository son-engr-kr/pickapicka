# Projects and files

- **Workspaces & projects**: pick a workspace folder, then create projects
  inside it *by name* (DaVinci-Resolve style). The landing page lists the
  projects in each workspace. On first launch it explains the workspace and
  offers `PictureClassifier-Projects` in your home folder; choosing a folder
  that already holds photos gets a warning, since the workspace should be
  separate from them. The new-project wizard counts the photos in the folder
  you pick and stops you at a folder with none. Photos are referenced by path; if a photo folder
  moves, the app offers to **re-link** it (matching by folder structure, then by
  file name). **Deleting a project never touches your photos**: it renames the
  project folder to `<name>.deleted-<timestamp>` and hides it from the list —
  decisions and edits stay inside, so renaming the folder back restores it.
- **See how it works**: an animated walk-through of the above, from the photo
  folder to the workspace, a project pointing at the photos, the work filling
  the project, and what deleting and exporting do. It plays on the first-launch
  screen, and **See how it works** on the landing page or **What is a project?**
  in the new-project wizard replays it.

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

## Where the app keeps its own files

Nothing about your photos lives here — that is all in `picks.json`. This is the
app's own state: the workspace list, recent projects, the remembered per-project
view, and saved edit presets.

| | macOS | Windows | Linux |
| --- | --- | --- | --- |
| `state.json` | `~/Library/Application Support/picture-classifier/` | `%LOCALAPPDATA%\picture-classifier\` | `~/.local/share/picture-classifier/` |
| model weights | `~/Library/Caches/picture-classifier/models/` | `%LOCALAPPDATA%\picture-classifier\Cache\models\` | `~/.cache/picture-classifier/models/` |
| `app.log` (Windows) | | `%LOCALAPPDATA%\picture-classifier\app.log` | |

Linux honours `$XDG_DATA_HOME` and `$XDG_CACHE_HOME` if they are set.

The Windows app runs with no console, so what it would have printed goes to
`app.log` instead. If it misbehaves, that file says why.

The split is by what it costs to lose. `state.json` cannot be regenerated, so it
sits in the data directory and gets backed up. The YOLOX weights (~20 MB)
re-download on demand, so they sit in the cache directory and stay out of every
Time Machine snapshot — deleting them only costs one download.

Versions up to 0.5.0 kept both in `~/.picture-classifier/`. The first run of any
`pcls` command moves an old install across and prints what it moved. It will not
overwrite anything already at the destination, and it leaves the old directory in
place if there is anything in it that it did not move.

InsightFace's face models are the one exception: they stay in `~/.insightface/`,
which is that library's own location, shared with any other tool on the machine
that uses it. Redirecting them would force a several-hundred-MB re-download and
duplicate a cache that is meant to be shared.
