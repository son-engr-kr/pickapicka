# Feature parity roadmap

The target is everything Evoto does plus everything Lightroom does. This file is
the ledger: every feature, which batch it lands in, and what it costs. It is
updated as batches land, so `git log docs/roadmap-parity.md` is the record of
how far along the program is.

## The three constraints that decide the order

1. **Local CPU, onnxruntime only.** No torch. `scoring/objects.py` sets the
   pattern for anything model-backed: a small ONNX fetched on first use into the
   platform cache, size- and SHA256-pinned, so a fresh install stays lean.
2. **Parametric and non-destructive.** `editing.py` re-renders from the original
   every time; an edit is a dict of scalars. Anything that *generates* pixels
   (inpainting, expression edits) has no scalar representation, so it needs a
   new concept — a baked raster layer, cached per photo, that the parametric
   grade then runs on top of. That concept is Batch 6 and everything generative
   waits on it.
3. **Redistributable weights.** This project is MIT and ships as an installer.
   Weights under non-commercial terms (CodeFormer's S-Lab licence, LaMa's
   `big-lama`) cannot be bundled. Where no permissive equivalent exists the
   feature is marked **licence-blocked** — that is a legal wall, not an effort
   estimate, and the only ways through are a permissive replacement or an
   opt-in download the user initiates.

## Decisions taken

**Where model files come from.** This repository's own release assets, the same
way `scoring/objects.py` already fetches `yolox_tiny.onnx`: pinned by size and
SHA256, downloaded on first use with a progress callback, temporary file
discarded on a checksum mismatch. Every model-backed batch therefore has a
prerequisite outside the code — the converted `.onnx` has to be uploaded to a
release before its feature can ship. Pointing at an upstream URL was rejected
for the obvious reason: a link that rots takes every fresh install with it.

**What to do about non-commercially-licensed weights.** Both, split by how much
the better model actually buys:

- The **default path is permissively licensed** — `cv2.inpaint` plus OpenCV's
  FSR for small and medium regions. No download, no consent dialog, no licence
  notice. Dust, blemishes, lint and small objects are handled here, and this is
  what the feature does when you click it.
- The **non-commercial model is opt-in and only for large regions**, where the
  quality gap is real (taking a whole person out of a background). One click,
  automatic from there.

The reasoning is that a non-commercial licence does not only block *us* from
redistributing — it follows the *user*. Someone being paid to shoot a wedding
becomes the infringer, and that describes a large share of this app's users. So
the frictionless path has to be the one with no strings, and the better model has
to be a deliberate, informed choice rather than a silent default.

## Status key

| Mark | Meaning |
| --- | --- |
| `done` | shipped |
| `B<n>` | scheduled in batch n |
| `blocked` | needs a licence answer or a decision recorded below |

---

## Batch 1 — parametric sliders, no model (done)

| Feature | Evoto name | Lightroom name | Status |
| --- | --- | --- | --- |
| Dehaze | Dehaze Photo | Dehaze | `done` |
| Texture | — | Texture | `done` |
| Noise reduction (classical) | AI Image Denoiser | Noise Reduction | `done` |

Notes: dehaze uses a dark-channel prior with the atmospheric light fixed at
white rather than estimated per frame. An estimated `A` is a *global* statistic,
so a 1:1 window would estimate a different one than the full-frame preview and
the two would not match — which breaks the pipeline's central promise that the
preview is the export. Fixed `A` keeps the operator local, and padding handles
the rest.

Noise reduction is the one stage that is legitimately resolution-dependent:
noise lives at the pixel level, so a downscaled preview has already averaged it
away. Judge it at 1:1, exactly as in Lightroom.

## Batch 2 — remaining model-free work

Everything here is classical CV expressible as scalars, so it drops into the
existing pipeline and the existing mask system with no architectural change.

- ~~HSL / Color Mixer — hue, saturation, luminance across 8 bands~~ `done`.
  Bands interpolated around the hue circle rather than windowed, so the weights
  are a partition of unity and there is no seam between neighbours; the effect
  is scaled by each pixel's own saturation so near-greys, whose hue is
  numerical noise, are left alone. Global only, like the tone curve.
- ~~Per-channel tone curves — R, G, B in addition to the master~~ `done`. They
  fold into the per-channel lookup `_wb_tone_lut` already builds, so they are
  free at render time; the master applies before them. The canvas keeps the
  inactive channels visible, and Reset clears only the active one.
- ~~Color Grading — shadow/midtone/highlight wheels, balance, blending~~
  `done`. Collapses into one table indexed by luminance, the same move
  `_wb_tone_lut` makes for tone: 239 ms at 24 MP. Tints are zero-luma vectors,
  so strength cannot move brightness by construction rather than by correction.
  Zone weights are a partition of unity at any blending.
- Point colour — sampled-colour selective adjustment
- ~~Sharpening detail controls — radius, detail, masking~~ `done`. 320 ms at
  24 MP, 30 ms at a 2048 px preview edge. Two things to know. It is the
  pipeline's second and last deliberate exception to resolution independence,
  because what sharpening fights is ~1 px of sensor and demosaic softness and
  that does not widen with the file; scaling the radius by `frame_long` would
  make it a third local-contrast operator beside clarity and texture. And it is
  **not a drop-in replacement** for the old one-slider unsharp mask: at the same
  `sharpen` value the result differs by ~6 levels on average and up to 40 on a
  hard edge, and no combination of the other three reproduces the old look,
  because the new operator clamps every pixel into the range of luma around it
  and the old one overshot freely. `sharpen` keeps its name and scale so saved
  edits load unchanged; what they render to improves and does change.
- Lens corrections — chromatic aberration, manual distortion, manual vignetting
- Transform / Upright — `AI Perspective Correction Tool`, auto-level and manual
- ~~Colour range and luminance range masks~~ `done` for luminance, as both a
  `range` mask kind and a refinement every kind can carry (so it composes with
  an automatic mask). Measured on the whole ungraded frame at a fixed grid, for
  the two reasons in `docs/editing.md`. The colour half is implemented and tested
  in `rangemask` but has **no UI yet**: it needs an eyedropper to sample from the
  preview. Half of that now exists — the white-balance picker put a click-to-
  sample tool on the overlay (`editing.neutral_wb`, `POST /api/edit/neutral`),
  so what is left is a second tool sharing the same plumbing and returning the
  sampled colour rather than a slider pair.
- ~~White balance from a clicked neutral~~ `done`. The inverse of `_wb_gains`,
  solved in the encoded 8-bit space the gains are applied in rather than in
  linear light, because the question is which slider pair reads neutral and not
  which illuminant was present. Two constraints, two unknowns, one closed form;
  red and blue land on their harmonic mean. Reports `clamped` when the cast is
  further than 1.25/0.75 per channel, which a deep tungsten frame is, and then
  aims green at the midpoint of the red and blue it could reach so the residual
  is a weaker version of the same cast rather than a green one on top of it.
- ~~AI Color Match, Camera Profile Matching — fit a 3D LUT from a reference
  frame~~ `done`, and
- ~~Profile browser and `.cube` LUT import — Evoto's `AI Color Looks`~~ `done`,
  as the editor's **Look** panel. Applied under every slider, before the grade,
  because a match is fitted on ungraded pixels. Edits store `{key, name,
  amount}` and the tables live in an app-global library (`userstate.LUT_DIR`)
  plus a per-project copy in `picks.json` (`luts`), keyed by `lut.table_key`,
  which now covers the shape and domain as well as the table. The match takes
  the reference as edited and the source as shot, and offers frames from the
  same scene only.
- Dehaze's inverse presets: `Vintage Filter`, `Cinematic Filters`
- ~~Red Eye Remover, pet eye~~ `done`, pipeline and UI (the **Heal & red eye**
  panel: automatic for people, a dragged circle per eye for people and pets). 153 ms at 24 MP. Detection gates on
  geometry before colour, which is what stops it firing on a red jumper.
- Change Color of Image, Image Color Inverter, Double Exposure, Glitch Effect
- ~~Image Resizer~~ `done`, as the export's long-edge setting, with JPEG
  quality, uncompressed TIFF, file-name templates and a metadata choice
  alongside it. Rendered exports used to be written without any EXIF; they now
  carry a fixed list of standard tags rebuilt from the original (see
  `metadata.py` for why not the whole block) and the original's colour
  profile. Export sharpening and colour space selection remain. The second
  needs colour management, which the pipeline does not have: it never
  converts colour, it only labels it.
- ~~Brush healing — the manual half of Batch 6~~ `done`, pipeline and UI (spot,
  heal and clone in the **Heal & red eye** panel). 331 ms at 24 MP for a spot,
  nearly all of which is the two full-frame float conversions rather than the
  heal; converting only the touched bounding boxes would remove most of it and is
  not worth the complexity until it shows up in use. Runs before the grade, and a
  window render is byte-exact against the whole render with its reported padding.
- ~~Snapshots~~ `done`, as six **edit slots** per photo (`server.EDIT_SLOTS`,
  `POST /api/edit/slot`). Lightroom's snapshots are an unbounded named list;
  a fixed rack of six was chosen instead because the thing being solved is
  "try a second version without losing the first", and an unbounded list of
  stashes turns into its own housekeeping job. Stored on the photo in
  `picks.json` and written the moment a slot is filled, so they outlive backing
  out of the editor — which is the point of having them.
- History, virtual copies, before/after, copy-paste settings

## Batch 3 — segmentation (one ONNX model, large payoff)

MediaPipe's multiclass selfie segmentation (Apache-2.0; `mediapipe` is already
an optional dependency) returns background / hair / body-skin / face-skin /
clothes in one pass. That single model unlocks all of the following, and every
one of them arrives as a new *automatic mask kind* inside the existing 16-mask
system — no new architecture.

- ~~Subject, background, skin, face, hair and clothes masks (Lightroom's AI
  masking; Evoto's `AI Masking Editor`)~~ `done`. A new `auto` mask kind in the
  existing 16-mask stack, so it carries the full slider set and composes with
  the drawn kinds. Model hosted at the `models-0.1.0` release tag.

  Three things worth recording. The exported graph stops at the **logits**, not
  probabilities — worth checking rather than assuming, since summing raw logits
  per group would have produced a plausible-looking alpha that was wrong at
  every edge. The field is computed by the *caller* from the whole, ungraded
  frame and passed into `render`, because a segmenter shown only a 1:1 window
  finds a different subject than the fit preview did, and one shown graded
  pixels produces a mask that crawls as you drag a slider. And the alpha is the
  probability itself, never thresholded: at a hair boundary the model is
  genuinely 50/50, and that is the correct alpha for a strand thinner than a
  pixel.
- AI Background Remover, Bulk Background Remover, Cutout Image, PNG Maker,
  Transparent Background Maker, Green Screen Remover
- AI Background Replacer, Background Color Changer, White/Black Background
  Changer
- AI Background Blur, Bokeh Effect with true subject separation
- Floor Reflection Effect — geometric, not generative
- AI Sky Changer — segmentation is fine; convincing relighting of the subject
  is not, so this ships as "replace and match colour", not "relight"
- Clothes Color Changer
- AI Shadow Maker — a geometric drop shadow only; a physically plausible one is
  Batch 7

## Batch 4 — face parsing and landmark warps

`insightface` (106 points) and MediaPipe FaceMesh (478) are already dependencies
and already used by `scoring/faces.py` and `scoring/eyes.py`. Add a face-parsing
model for per-region masks and this whole block opens. Warps are stored as
control-point offsets, so they stay non-destructive — which is strictly better
than Evoto's commit-the-pixels approach.

Skin — the frequency-separation family, all as sliders on a skin mask:
- AI Skin Retouching, AI Blemish Remover, AI Soften Skin
- AI Frequency Separation, AI Dodge and Burn (skin contouring)
- Facial Wrinkle Remover, AI Frown Lines Remover, Remove Marionette Lines,
  Remove Dark Circles, Freckles Filter
- AI Rosy Complexion, AI Skin Tone Changer

Eyes, mouth, brows:
- Eye Color Changer, AI Catchlights, AI Eye Editor, AI Eyebrow Filter
- Teeth whitening (the colour half of `AI Teeth Fixer`)
- AI Makeup Editor — recolour of lips, blush and lids only

Hair (mask-based recolour; dark-to-light does not work and will not pretend to):
- Hair Color Changer, White Hair Blackening, Hair Shine Enhancement

Warps:
- Face Reshaping Tool, Nose Reshaper, AI Face Slimming, Double Chin Remover
- Lightroom's per-person mask parts map onto the same parsing model

## Batch 5 — pose and body

A pose model (RTMPose or MoveNet, 10–30 MB ONNX) plus the Batch 3 person mask.

- AI Body Editor, Flat Stomach Photo Editor — mesh warp driven by the skeleton,
  bounded by the person mask so the background does not follow
- AI Hand Rejuvenation — the frequency-separation work of Batch 4 on hands

## Batch 6 — the baked-layer architecture, then inpainting

This is the fork in the road. A generative edit cannot be a scalar, so it needs:
a per-photo raster layer stored beside `picks.json`, content-addressed by the
mask that produced it, that `render` composites *before* the parametric grade.
Undo stays intact because the layer is a separate object from the edit dict.

Once that exists, inpainting powers all of:

- AI Object Remover, AI Magic Eraser, AI People Remover
- Shadow Remover, AI Tattoo Remover, Glasses Glare Remover
- Clothes Wrinkle Remover, AI Lint Remover, Pet Leash Remover
- AI Stray Hairs Remover, AI Stray Fur Removal, Fill Hair Part
- AI Remove Tan Lines
- Lightroom's content-aware Remove, Heal and Clone

Unblocked, and shipping in two halves per *Decisions taken*: `cv2.inpaint` and
OpenCV FSR are the default and cover everything small, so the common cases work
with no download at all. `big-lama`, whose weights are not redistributable, is
the opt-in path for large regions only.

## Batch 7 — restoration and enhancement models

- Super resolution / `AI Photo Enhancer` / Lightroom Enhance — Real-ESRGAN is
  BSD-3, so this one is clean
- AI Photo Sharpener / Unblur beyond unsharp masking
- **AI Denoise** on raw data, the way Lightroom does it — this needs a model
  trained on Bayer input and is the honest version of Batch 1's slider
- AI B&W Photo Colorization — DDColor or similar, weights need checking
- AI Old Photo Restoration — **licence-blocked** (GFPGAN, CodeFormer)

## Batch 8 — generative, and why it is last

Diffusion on CPU at 24 MP is minutes per image. These can be built; whether
anyone would wait for them is the open question, and that is a product decision
to record before spending the effort.

- AI Background Generator, AI Background Fusion, AI Image Extender
  (outpainting), Lightroom's Generative Remove
- AI Facial Expression Changer, AI Smile Filter, AI Open Eyes, Eye Contact
  Correction, AI Teeth Fixer (alignment), AI Smooth Hair, AI Hair Editor volume
- AI Outfit Extractor, AI Shoes Editor, AI Product Retouching

## Not photo processing

Real work, different discipline, tracked separately so it never blocks the above.

- **Tethered capture** (both Evoto and Lightroom have it) — gphoto2 on
  macOS/Linux plus vendor SDKs on Windows, per-camera testing, SDK licensing
- **Lightroom's other modules** — Print, Slideshow, Web, Book, publish services,
  soft proofing, map/GPS
- **Catalog features** — keywords, colour labels, smart collections, compare and
  survey views, plugin API
- **Preset libraries at Evoto's scale** (800+) — content production, not code

## Already shipped before this roadmap

Culling — blur and exposure scoring, per-scene suggestions, one-key decisions,
focus peaking, loupe, HDR bracket merge, subject and face grouping. Covers
Evoto's `AI Culling` and Lightroom's flags and ratings, in more detail than
either.

Editing — the light and colour sliders, master tone curve, clarity, sharpen,
vignette, crop and straighten, 16 local masks (radial, gradient, brush),
defocus, motion, glow, mosaic, film emulation as a chain, EXIF watermarks,
stacking presets, batch apply, JPEG export, RAW decode for all major
formats. Covers Evoto's `RAW Photo Editor`, `RAW Converter`, `AI Image Cropper`,
`AI Photo Straightener`, `Add Grain to Photo`, `Vignette Effect`,
`AI Glow Effect`, `Photo Filters`, `Fix Overexposed Photos`, `Image Brightener`,
`Darken Image`, `Black and White Filter`, `AI Background Blur` (masked),
`Bokeh Effect` (masked), mosaic, `Presets`, `Batch Editing`.
