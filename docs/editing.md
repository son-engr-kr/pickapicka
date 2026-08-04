# Editing

Non-destructive throughout: the original file is never written to, and
adjustments bake in only on export or download.

![The editor](images/editor.jpg)

## Adjustments

- **In-app photo editing**: adjust any photo right from the picking screen —
  Exposure, Contrast, Highlights/Shadows/Whites/Blacks, Temperature/Tint,
  Vibrance/Saturation, Clarity/Sharpen/Vignette, and a draggable **tone curve**,
  with a live before/after preview and an **Auto** button. Fully
  non-destructive — originals are never modified; adjustments bake in only on
  export. Save **global presets** and **apply an edit/preset to many photos at
  once** (a whole scene, all picks, or a hand-picked selection).

## Crop and straighten

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

## Local adjustments

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

## Film emulation

- **Film emulation** — the chain, not a filter. A "film look" shipped as a colour
  LUT plus white noise reads as a filter because that is what it is. This builds
  the effects in the order the physics happens: a characteristic (Hurter-Driffield)
  response applied to **log exposure** and anchored on mid grey, with independent
  toe and shoulder; a **dye-density crosstalk matrix** in density space, which is
  what produces the shadow/highlight colour crossover a per-channel curve cannot;
  **halation**, light scattering off the film base and re-exposing from behind,
  weighted red-first because that layer sits deepest, so a bright edge bleeds
  warm; and **grain** on a lattice defined against the frame — correlated rather
  than per-pixel, strongest in the mid-densities, the same size whether it is
  rendered as a thumbnail or at 1:1, and the same grain every time so the preview
  is the export. Seven stocks plus a Default button, each one click, then
  twelve parameters underneath. Reading behind it: Newson/Delon/Galerne (CGF 2017) on
  resolution-independent grain, Norkin/Birkbeck (DCC 2018) and AV1 §7.18.3 on
  autoregressive grain synthesis.

## Watermarks and shooting info

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

## Working at full resolution

- **1:1 editing**: the editor previews fitted by default, but fit/50/100/200%
  renders the visible window from the **full-resolution original**, so you can
  judge real sharpness while grading. Scroll to zoom, drag to pan (space or
  middle-drag to pan past a mask).
- **RAW support**: RAW files are first-class in the grid. When a shot exists as
  both a RAW and a JPEG, the RAW is preferred; edit it and export a JPEG.

## Presets

- **Presets**, including a built-in car set: glossy paint, studio white, night
  neon, golden hour, plus mask-carrying ones for a speed pan, background bokeh
  and plate blurring. Built-ins are read-only; tweak and save your own.
  Presets load **additively** by default, so a local preset drops its mask onto
  whatever you already have instead of wiping it — stack "background bokeh" and
  "blur a plate" on top of your own grade, and a slider the preset leaves
  neutral keeps your value. Switch to *replace* for a clean slate. Bulk apply
  has the same choice, so you can add one mask to every pick without touching
  their individual grades.

## Pipeline order

The order the stages run in is not arbitrary, and two parts of it are worth
knowing because they decide what a slider means:

1. **Grade** — white balance, then tone, then colour, then the optical effects
   (defocus, smear, bloom), then sharpen, then the vignette. Masks run after the
   global pass, each blending a locally graded copy through its own alpha.
2. **Film** — the capture medium, applied to the graded frame.
3. **Geometry** — straighten, then crop. This comes *after* the grade, so a mask
   is measured against the original frame and cropping never drags it off the
   thing it was drawn on.
4. **Watermark** — last, and placed against the cropped frame, because a
   signature belongs on the picture you end up with.

Everything that has a size — a blur radius, a vignette, grain, a watermark — is
measured against the whole frame rather than against the array being rendered.
That is what makes the thumbnail, the fit preview, a 1:1 window and the export
agree with each other, and it is why the 1:1 view can be trusted while grading.
