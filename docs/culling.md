# Culling

Getting from a card full of frames to the ones worth keeping. The grid is the
whole job: decide, filter, move on.

![The grid](images/grid.jpg)

## Scoring and suggestions

- **Per-photo scoring**: Laplacian blur, brightness exposure, optional
  closed-eye detection. Face detection can be switched off per project from the
  re-score dialog — on a shoot where every person is a bystander, clustering them
  into People and penalising their closed eyes is just noise.
- **Per-scene auto-suggestion**: top 30% pick / middle review / bottom 30%
  reject, normalized within each scene.

## Scenes

- **Scene grouping**: by folder structure, or by EXIF capture-time gaps
  (configurable in minutes). Switch any time without re-scoring.

## Deciding

One key per frame, without leaving the keyboard — the full list is in the
[README](../README.md#keyboard). Hovering any button shows what it does and its
shortcut, so the list is not something you have to have memorised.

- **Bulk actions**: reject all undecided in a scene; export all picks to a
  folder (preserving structure or flattened).
- **Download the selection** (`D`): saves the selected photos straight to your
  Downloads folder — no dialog. Edits are baked and RAW is rendered, so what
  lands there is the photo as you graded it. One photo goes in loose, several
  go into a dated subfolder; nothing already there is ever overwritten.

## Filtering and picking up where you left off

- **Filter by decision or by edited**: alongside all/undecided/pick/review/reject
  there is `edited`, which shows only the photos you have actually graded —
  useful for a last pass over your own work, or for finding what still needs it.
- **Reopens where you left it**: the filter, grid layout, page and scene are
  remembered per project, so a project opens on the shot you were looking at
  rather than on page 1 of everything. Stored per project in the app's
  `state.json` (see [projects.md](projects.md#where-the-app-keeps-its-own-files));
  a scene that has since been regrouped away falls back to the first one.
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
  softness, not to smooth it away.

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
- **Drag-and-drop people priority**: rank face clusters by importance; photos
  containing higher-priority people sort to the top within each scene.
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
