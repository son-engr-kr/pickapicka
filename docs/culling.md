# Culling

Getting from a card full of frames to the ones worth keeping. The grid is the
whole job: decide, filter, move on.

![The grid](images/grid.jpg)

## The screen

- **The top bar** is on every screen: the project's name (a menu of recent
  projects to switch to), the tabs **Projects**, **Cull** and **Edit**, a note
  of anything running in the background (an export, a re-score, face
  grouping), Export and `?`. The viewer and the editor open under it, so the
  tabs always say where you are; leaving the editor by a tab asks first about
  an unsaved edit.
- **The sidebar** lists the scenes first, each with how much is left and how
  many picks, and below them the grouping, People and Subjects.
- **The filter bar** above the grid shows each filter with how many photos it
  would show in this scene.
- **The grid scrolls.** 1, 2, 4 or 8 per screen sets the tile size; the arrow
  keys scroll a row at a time to keep the cursor in view, `[` `]` move a
  screen, and the header says which photos are showing ("61–68 of 734"). Only
  the rows on screen are built, so a scene of thousands scrolls as smoothly as
  one of twenty. Thumbnails are made in the background as soon as a project
  opens, so a fresh shoot does not show black tiles for long.
- **The info panel** (`I`) beside the grid shows the highlighted photo's
  histogram, the suggestion and why in words, its faces, and the camera and
  exposure, with the decision buttons. It steps aside on a window narrower
  than 1180 px.
- **The viewer** (`Enter`) shows the photo at once from its thumbnail while the
  full image loads, with the decision buttons, the reasons for the suggestion,
  and a **filmstrip** of the scene along the bottom. The editor has the same
  filmstrip; moving to another photo there saves the edit. Preferences can turn
  the filmstrip off, which gives its height to the photo.
- **Zoom in the viewer**: a click goes to 100% where it lands and a second one
  back to fit; scroll or pinch zooms about the pointer (up to 400%), `+` and `-`
  zoom about the middle, `F` or `Z` toggles fit and 100%, and a drag pans.
  Fit / 100% / 200% and the current scale are in the top bar. The zoom and the
  place are kept as you move to the next photo, so a burst is checked frame
  after frame at the same spot. `Tab` hides every bar, leaving only the photo
  (the keys still decide and move), which matters most for a portrait frame on
  a landscape screen.

- **Scoring shows how long is left**: the steps (find photos, score, group
  faces, open), how many are done, the time left, and **Stop**. Stopping a new
  project leaves nothing listed (its RAW previews stay cached for the next
  try); stopping a re-score leaves the project as it was.
- **Preferences** (the gear, or `⌘,`) hold the app's settings (key bar, info
  panel, loupe, what is behind photos, the tours), the workspaces, and a way to
  each of the project's own settings.

## Scoring and suggestions

- **Per-photo scoring**: Laplacian blur, brightness exposure, optional
  closed-eye detection. Face detection can be switched off per project from the
  re-score dialog — on a shoot where every person is a bystander, clustering them
  into People and penalising their closed eyes is just noise.
- **Per-scene auto-suggestion**: top 30% pick / middle review / bottom 30%
  reject, normalized within each scene.

## Scenes

- **Scene grouping**: by folder structure, or by EXIF capture-time gaps
  (configurable in minutes). Switch any time without re-scoring. The
  new-project wizard previews both on the chosen folder before anything is
  scored: the scenes each subfolder would make, or a timeline of the shoot that
  re-splits as you drag the gap slider, with the longest pause marked so you can
  see which gap would separate what.

## Deciding

One key per frame, without leaving the keyboard: `P` pick, `R` reject, `V`
review, `U` clear. Each acts on the highlighted photo and moves on to the next,
in the grid and in the viewer alike; with a filter such as Undecided, where the
decided photo leaves the list, the next one is already in its place. Decisions
show at once and are saved in the background in the order made, so keys can be
pressed as fast as you can judge. The bar along the bottom always shows the
keys for where you are, `?` lists every one (also in the
[README](../README.md#keyboard)), and hovering any button shows its shortcut.

- **Stars and colour labels**, set apart from the decision, with Lightroom's
  keys: `1`–`5` stars and `0` none, `6`–`9` a red, yellow, green or blue label
  (the same key again removes it). They show on the tiles, the filmstrip, the
  viewer and the info panel, where they can be clicked, and the filter bar
  filters on them. Exports carry them as `xmp:Rating` and `xmp:Label`, which
  Lightroom and Bridge read, and the EXIF Rating Windows shows; a copied
  original gets them in its own XMP, its image data untouched. The number keys
  used to be aliases for the decisions: those are `P`, `R` and `V` (and `A`).
- **Compare** (`C`): two to four photos side by side, the selection or the
  highlighted photo and the next. One pane is active (`Tab` or a click); the
  arrows swap its photo for the one before or after, and the decision, star and
  label keys act on it. `Z`, or a click, takes every pane to 100% at the same
  place, and dragging one pans them all, to see which frame of a burst is
  sharpest. `Esc` returns to the grid on the active photo.

The keyboard cursor follows the mouse only when the mouse moves: a pointer
resting on the grid does not pull the cursor when the page turns under it.

- **Bulk actions**: reject all undecided in a scene; export all picks to a
  folder (preserving structure, flattened, or one folder per combination of
  people).
- **Export options**: JPEG at a chosen quality, or uncompressed TIFF for
  print; full size or a long edge (4K, 2048 px, 1080 px or your own); file
  names from a template (`{name}` `{seq}` `{date}` `{time}` `{scene}`
  `{project}`); and metadata kept whole, kept without location, or left out.
  The export runs in the background with a progress bar and can be stopped,
  and the dialog opens on the settings used last.
- **Metadata survives the export.** Camera, lens, exposure, capture time,
  copyright and GPS are carried into every rendered file. Orientation is reset
  because the pixels are already upright, and the size is the exported size.
  The camera's private MakerNote and the embedded thumbnail are left out: the
  first cannot be relocated safely and can push the EXIF past a JPEG's 64 KB
  limit, and the second would show the unedited frame. The colour profile is
  the original's (sRGB for RAW); a file that had none gets none. A photo with
  no edits, exported as a full-size JPEG with all its metadata, is copied byte
  for byte instead.
- **Download the selection** (`D`): saves the selected photos straight to your
  Downloads folder — no dialog. Edits are baked and RAW is rendered, so what
  lands there is the photo as you graded it, as a full-size JPEG with its
  metadata. One photo goes in loose, several go into a dated subfolder; nothing
  already there is ever overwritten.

## Filtering and picking up where you left off

- **Filter by decision or by edited**: alongside all/undecided/pick/review/reject
  there is `edited`, which shows only the photos you have actually graded —
  useful for a last pass over your own work, or for finding what still needs it.
- **Reopens where you left it**: the scene, filter, grid layout and the photo
  you were on are remembered per project, and the full-screen viewer reopens if
  that is where you were. It is the photo that is kept, not just the page, so
  with the Undecided filter (where each decision takes a photo out of the list)
  you land back on the right shot. Saved as you move, and once more on leaving
  the project, quitting or closing the tab. Stored per project in the app's
  `state.json` (see [projects.md](projects.md#where-the-app-keeps-its-own-files));
  a scene that has since been regrouped away falls back to the first one, and a
  photo no longer in the filter falls back to its page.
- **Project history**: recent folders are remembered so you can reopen them
  from the landing page.

## Judging sharpness

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
  softness, not to smooth it away. It is for the fitted photo, and steps aside
  while the viewer is zoomed; the scroll wheel zooms the photo, and the loupe's
  magnification is the 1:1 / 2:1 / 4:1 buttons.

## Subjects

Tell a project what it is a shoot *of* and the scoring changes shape: frames
missing the subject sort down, and sharpness is measured on the subject rather
than on the whole frame.

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

## People

- **Face clustering**: detects faces with `insightface` and clusters them
  per-person via DBSCAN on embeddings.
- **Drag-and-drop people priority**: rank face clusters by importance. The
  grid's **Sort** switch picks the order within each scene: **Time** (the
  default) is shooting order, and **People** puts photos containing
  higher-priority people first. People order pulls a burst apart wherever its
  frames see different people, which is why it is not the default. The choice
  is remembered per project with the filter and layout.
- **Exclude clusters**: hide irrelevant clusters (background people, false
  positives) from sorting and chips.

## Scoring and grouping are one button

- **One button for scoring and grouping**: *↻ rescore* opens a dialog asking
  what to detect, then scores and groups in one run. Scoring recomputes face
  embeddings, which discards every group, so grouping follows automatically
  rather than leaving a project with faces and nobody grouped into them. *↻
  group* on its own re-groups without re-scoring, which is what you want after
  changing a grouping setting. The Subjects ⚙ is for renaming and hiding groups
  and never re-scores — doing so would throw away the names you just typed.

## HDR brackets

- **HDR bracket auto-merge**: detects auto-exposure brackets from EXIF and
  exposure-fuses each into one photo, with a tunable real-estate "look"
  (shadow lift, local contrast, saturation) and an HDR-vs-0 EV compare toggle.
