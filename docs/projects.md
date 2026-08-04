# Projects and files

- **Workspaces & projects**: pick a workspace folder, then create projects
  inside it *by name* (DaVinci-Resolve style). The landing page lists the
  projects in each workspace. Photos are referenced by path; if a photo folder
  moves, the app offers to **re-link** it (matching by folder structure, then by
  file name). **Deleting a project never touches your photos**: it renames the
  project folder to `<name>.deleted-<timestamp>` and hides it from the list —
  decisions and edits stay inside, so renaming the folder back restores it.

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
