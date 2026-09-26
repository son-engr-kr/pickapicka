# Editing

Non-destructive throughout: the original file is never written to, and
adjustments bake in only on export or download.

![The editor](images/editor.jpg)

## Adjustments

- **In-app photo editing**: adjust any photo right from the picking screen —
  Exposure, Contrast, Highlights/Shadows/Whites/Blacks, Temperature/Tint,
  Vibrance/Saturation, Texture/Clarity/Dehaze/Denoise/Sharpen/Vignette, and a
  draggable **tone curve**,
  with a live before/after preview and an **Auto** button. Fully
  non-destructive — originals are never modified; adjustments bake in only on
  export. Save **global presets** and **apply an edit/preset to many photos at
  once** (a whole scene, all picks, or a hand-picked selection).
- **White balance by click.** Temperature and Tint are channel gains rather
  than a Kelvin conversion, and the eyedropper beside them solves the pair for
  you: click something that ought to be grey — a white wall, a shirt, a grey
  card — and the values that make it grey land on the sliders. Two properties
  are worth knowing. It reads the **ungraded original**, so the answer is
  absolute and replaces whatever is set rather than stacking onto it; and when
  the cast is further than the sliders reach (the gains stop at 1.25/0.75, about
  1.7 stops between red and blue) it goes as far as it can and says so instead
  of quietly pinning at the end. A deep tungsten frame is one of those: take the
  rest with the blue channel curve.

  For a yellow cast on skin specifically, the white balance is only the first
  of three moves — the other two are lifting the exposure end of the tone
  sliders and taking the yellow band down in the colour mixer, because a face
  under a bad lamp keeps some of its cast in the orange and yellow hues after
  the light itself is neutral. The **Portrait** presets are that recipe with
  numbers in it.

## Repairs

Two stages that fix the captured image rather than interpret it, and they run
**before** everything tonal. That order is not cosmetic:

- A red pupil left in place is amplified by every stage that follows — vibrance
  finds it, the curve lifts it, a colour-range mask keys off it.
- A heal applied *after* a grade diffuses from pixels a curve has already
  crushed, so the repair has the grade baked into it and stops matching its
  surroundings the moment you move the curve again.

- **Heal, clone and spot**, in the editor's **Heal & red eye** panel. **Spot**
  is one click on dust or a blemish; **Heal** is painted over what should go
  and fills it from what surrounds it; **Clone** copies from a place you pick
  (Alt-click, or the first click), through a feathered edge, and every stroke
  reads from that same point. **Size** is the brush, `[` and `]` nudge it, and
  each repair in the list can be switched off or removed. Opening the panel
  shows every repair on the photo.
  - What this is good at, and what it is not, stated plainly: dust, sensor
    spots, lint, a blemish, a stray hair, a power line against a sky — excellent,
    indistinguishable in practice. Anything asked to reproduce *texture* —
    fabric, foliage, gravel, hair — over more than a few pixels smears, and no
    parameter fixes that; it is what a diffusion fill is. Measured, the error in
    the filled area is about twenty times larger on a woven texture than on a
    smooth gradient at the same radius.
  - When heal smears, reach for **clone**: it copies real texture from a place
    you choose, which is the honest answer.
  - Nothing here downloads a model or borrows weights, so it carries no licence
    on to you. A heavier model for large regions is a separate, opt-in thing.
- **Red-eye and pet-eye**, which are genuinely different problems and not one
  control with a switch. Red-eye pulls a flash-reddened pupil to a neutral built
  from the channels the flash did *not* contaminate, and **protects the
  catchlight** — killing it is what makes a corrected eye look dead. Pet-eye
  gates on luminance instead, because a tapetum reflection is green or yellow or
  blown white and a red-channel test finds none of it, and it draws a synthetic
  catchlight back in because the real one was destroyed.
  - **Find red eyes** looks for people's eyes automatically. **Fix an eye** and
    **Pet eye** take one eye at a time: drag from the centre of the pupil out
    to the edge of the iris (or of the glow, for a pet). The circle is verified
    and tightened, not taken as drawn, so one much larger than the eye, or over
    anything red that is not an eye, is refused.
  - The automatic detection finds *eyes* geometrically first and only then tests
    for redness. That ordering is the whole defence against the failure this
    feature is infamous for: a red jumper, red lipstick and a brake light are all
    compact saturated red blobs, and none of them is ever looked at, because no
    eye is there.
- Both are stored as normalized coordinates, so a repair means the same thing at
  every render size, and a preset can carry them — sensor dust lands in the same
  place on every frame a body shoots, so "remove the dust spots" is exactly the
  kind of thing to apply across a whole shoot.

## Automatic masks

- **Six one-click masks** — Subject, Background, Skin, Face, Hair, Clothes — from
  one segmentation model, added to the same 16-mask stack as the drawn kinds and
  carrying the same full slider set. So "brighten the subject" and "cool the
  background" are one click and one slider, and they compose with a radial or a
  brush you drew yourself.
- The model (16 MB, Apache-2.0) is fetched the **first time you add an automatic
  mask**, not at startup — a shoot with nobody in it never pays for it.
- **No threshold, so no staircase.** What the mask carries is the model's own
  probability, not a cut at 50%: where hair genuinely half-covers a pixel the
  alpha is genuinely a half, which is the correct answer and the one a hard mask
  throws away. Feather starts at zero for the same reason — the good edge is
  already there — and is available for when you want it softer anyway.
- The mask is computed from the **whole, ungraded frame**, and this matters twice
  over. Shown only the visible window at 1:1, the segmenter would find a
  different subject than the fit view found, and the mask would jump as you
  panned. Shown graded pixels, it would crawl every time you moved a slider.
  Neither happens: one field per photo, cropped to whatever is being rendered.
- Press `\` to see exactly what it selected. The editor draws its own tint for a
  radial or a brush, but a segmentation is not a shape it can redraw, so that one
  comes back from the server as an image.

## Range masks

- **Select by tone, not by shape.** A **Range** mask has no geometry at all: it
  selects every pixel in a band of perceptual lightness, so "every mid-tone" is
  one click.
- Better, a range is a **refinement any mask can carry** — including an automatic
  one. "The subject, but only its highlights" is one mask with one slider set,
  not two masks fighting over the same pixels.
- The selection is measured on the **whole, ungraded frame** at one fixed grid,
  and both halves of that matter. Measured on a 1:1 window it would select a
  different set of tones than the fit preview did, because a window's histogram
  is not the frame's. Measured on graded pixels it would move every time you
  moved a slider — a luminance range would chase the exposure you were setting
  through it.
- The price of the fixed grid, stated plainly: a selection finer than that grid
  is not available. A range mask will not pick single leaves out of a sky or
  catch a one-pixel specular highlight. That is the trade for a slider you can
  tune against the preview and trust on export.
- Press `\` to see the selection. For a shape carrying a refinement the tint is
  the shape *narrowed* to the selection, which is what actually gets graded.

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

## Looks

- **A look sits under every slider**, the way a profile sits under Lightroom's:
  it is applied to the photo as shot, and the sliders then adjust the look.
  **Import a `.cube`** (1D or 3D, the format creative profiles and colourists'
  show LUTs ship in), or **match colours to another photo** of the same scene,
  which fits a table that moves this frame's colours towards that one's. The
  match is for a moment shot on two bodies: same subject, same light. It
  matches distributions, not content, so a grey street matched to a red sunset
  comes out red. **Amount** dials either kind back. Whole photo only.
- **Parsed strictly.** A file that is not a LUT (a wrong entry count, no
  declared size, a video-range flag) is refused with the line that is wrong,
  rather than guessed at.
- **The reference is taken as edited, the source as shot.** Matching to how a
  frame looks is the point, and the look is applied before any grading, so it
  has to be fitted on ungraded pixels.
- **A look library, shared by every project.** Imported and fitted looks are
  kept once each, and an edit stores a reference to one, not its table: a
  33-point cube is 290 KB, and one look across a shoot would otherwise sit in
  `picks.json` once per photo. A project also keeps its own copy of each look
  its edits use, so it still renders after the library is cleared or on
  another machine.

## Tone curves

- **Four point curves**: the master plus one each for **R, G and B**. The master
  runs first — it decides how bright a tone is — and the channel curves then
  decide what colour it takes there, which is the order that makes a split-toned
  shadow behave as you would expect.
- The inactive channels stay drawn on the canvas, thin and faint, because four
  curves that each hid the other three would make a split tone impossible to
  reason about. A tab carrying a change is marked, and **Reset curve** clears
  only the channel you are looking at.
- All four cost **nothing at render time**. White balance, exposure, the tone
  regions, contrast and every curve are pointwise and per-channel, so they
  collapse into the one 256-entry per-channel lookup the pipeline already
  applies with `cv2.LUT` — adding a red curve adds a table to bake, not a pass
  over the frame.

## Colour mixer

- **Colour mixer (HSL)**: eight hue bands — red, orange, yellow, green, aqua,
  blue, purple, magenta — each with Hue, Saturation and Luminance. Two details
  are worth knowing because they decide how it behaves:
  - The bands are **interpolated into each other**, not windowed. A hue sitting
    on a band's centre gets exactly that band's value, and one halfway between
    two centres gets a mix of both — so there is no seam between orange and
    yellow to discover later on a sunset, and no hue is either skipped or
    corrected twice.
  - **Near-greys are left alone.** The hue of an almost-grey pixel is numerical
    noise: a rounding error decides whether a patch of concrete is blue or
    purple. Pushing those pixels would speckle a flat wall with two different
    corrections, so the mixer fades out as saturation approaches zero and only
    moves colours that are actually there.
- Luminance moves a band towards white or towards black and never past either,
  so the brightest members of a band keep their hue instead of clipping and
  flattening into one another.
- The mixer is **global**, with no per-mask version, and deliberately so: it is a
  statement about a colour wherever it appears in the frame, and a mask already
  answers "only here" better than eight bands could.

## Colour grading

- **Three-way wheels** — a hue, a strength and a luminance for shadows,
  midtones and highlights, plus **Blending** (how far the three regions reach
  into each other) and **Balance** (where the shadow/highlight split sits).
- **Adding colour never moves brightness.** Every tint is a vector whose luma is
  zero, so strength changes hue and chroma and leaves the exposure you set
  exactly where you set it — no measuring the luminance and putting it back
  afterwards, and no drift when you stack two zones.
- The three regions are a **partition of the tonal range by construction**, at
  any Blending: no tone is graded twice and none is skipped. A mid grey at the
  default blending is entirely midtone, which is why a flat mid-grey frame will
  not respond to the shadow wheel at all — that is the control working, not
  failing.
- Luminance moves a region towards white or black and never past either, so a
  lifted shadow keeps its colour instead of clipping grey.
- Hue on its own does nothing: it is a direction, and strength is the distance.
  The panel says so rather than showing a number that is having no effect.

## Texture, dehaze and noise

- **Texture** and **Clarity** are the same operation at two scales, and the
  scale is the point. Clarity's wide radius and midtone mask move *regional*
  contrast; texture's small radius moves the finest detail the frame holds. So
  negative texture smooths skin without the flat look a blur gives — the edges
  of a face live in the low frequencies and are left alone — while positive
  texture finds fabric and bark rather than darkening one side of the sky.
- **Dehaze** inverts the haze model, or runs it forwards when negative to put
  atmosphere back into a frame that has none. The atmospheric light is fixed at
  white rather than estimated per frame: an estimate is a global statistic, so a
  1:1 window would arrive at a different one than the fit preview and the two
  renders would disagree. A slightly weaker dehaze that is identical at every
  zoom is worth more here than a better one that drifts.
- **Sharpening** is four sliders, not one: **Amount**, **radius** (the size of
  the edge being enhanced, half a pixel to three), **detail** (how much of the
  finest structure comes up with it) and **masking** (hold it off flat areas, so
  a sky or a cheek keeps its noise unamplified). Detail changes *which*
  frequencies are amplified rather than how much of one fixed mixture is, which
  is what makes it a control rather than a second Amount.
  - Halos are prevented at the source: each pixel is clamped into the range of
    luma already present around it, so on a step edge the transition can be made
    as steep as you like and still cannot overshoot the plateau it is heading
    for. Detail buys a little slack on top of that, which is what turning it up
    actually spends.
  - This replaced a bare unsharp mask, and **it is not the same picture**. The
    `sharpen` slider keeps its name and its scale, so an edit saved earlier
    loads with its number where it was — but at the same number the result now
    differs by around 6 levels on average and up to 40 on a hard edge, because
    the old one was free to overshoot. No setting of the other three brings the
    old look back; refusing to halo is the point of the new one. Every saved edit
    carrying a sharpen value gets better, and it does change.
  - Sharpening is the pipeline's **second and last** deliberate exception to
    resolution independence, for a different reason from denoise: what it fights
    is about a pixel of sensor and demosaic softness, and that does not get wider
    when the file does. Scaling the radius with the frame would not be a bigger
    version of the same look, it would be local contrast — which this pipeline
    already has two of, in Clarity and Texture, both correctly frame-relative.
- **Denoise** splits the job, because luma and chroma noise look nothing alike.
  Chroma noise is coloured blotches several pixels across and the eye carries
  almost no chroma detail, so it is blurred away. Luma noise goes via non-local
  means, which averages a pixel only with the pixels whose neighbourhoods
  resemble its own — so an edge is only ever averaged with other copies of that
  edge, and it survives. See *Pipeline order* for why this one is judged at 1:1.

All three carry onto masks, so denoise can be held back to the shadows where
noise actually lives, and negative texture can be brushed onto skin alone.

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
  Grid, editor and export all show the same decode — camera white balance, no
  auto-brightening — so a tile is a small version of the file you will get, not
  the camera's own rendering of the shot. That decode is cached per RAW inside
  the project on the first scan, which is why a RAW shoot takes a little longer
  to score and nothing after that.

## Slots

Six stashes per photo, listed above the adjustment panel. Each row says what it
holds — `light · colour · 2 masks` — and when it was put there, and carries its
own controls:

| Control | What it does |
| --- | --- |
| the row | Loads that slot. |
| the save button | Writes what is on screen into that slot, empty or not. |
| the × button | Clears it. |

One action per control, on purpose. The first version of this panel was a row of
numbered chips where a click meant *save* on an empty slot and *load* on a full
one, with overwrite hidden behind ⌥ — which of the three you got depended on
state the chip did not show, and overwriting was undiscoverable. A row per slot
costs a little height and removes the mode.

The slot the working edit currently matches is marked **on screen**; once you
move a slider away from it, it says **edited — not saved** instead. Both are
recomputed from the values rather than remembered, so neither can claim
something the pixels do not, and when two slots hold the same thing the one you
last saved into or loaded wins the marker.

**Nothing is written behind your back.** Editing after loading a slot does not
update that slot, and it is the `edited — not saved` line that says so. Auto-
saving would be the one behaviour that defeats the feature: the stash you came
back to would be overwritten by the experiment you were trying to be able to
abandon. Folding a change in is the save button on that row.

Three properties are what make them worth using while experimenting:

- **A slot is not an edit.** Filling one changes nothing about what the photo
  renders as, and does not move `edited_at`. Loading one only changes the
  working edit, which you then still have to **Save** to commit.
- **They are written to `picks.json` immediately**, so they outlive *Cancel*,
  closing the editor, and quitting the app. Backing out of a dead end costs one
  click rather than the whole grade.
- **They carry the whole edit** — masks, curves, film, watermark, crop — because
  they are the same dict the photo stores.

An empty rack is not stored at all, so a photo you have never stashed anything
on stays exactly as small in the database as it was before slots existed.

## Presets

Thirty-nine built-in presets in six groups, each a plain edit dict — so a
preset can carry anything the editor can do, including the colour mixer, the
grading wheels, the channel curves and local masks. They are parametric all the
way down, which is the difference between these and a `.cube` LUT: every number
a preset sets is still a slider you can then move.

| Group | What it is for |
| --- | --- |
| **Portrait** | Skin. Bright skin, airy and fair, clean beauty, warm glow, backlit rescue, the tungsten and fluorescent indoor fixes, and a local one that cools and smooths the skin alone through an automatic mask. |
| **Look** | A colour identity over a correct photo: teal and orange, faded matte, bleach bypass, moody blue, golden hour, cross process, cinematic matte, vintage warm, neon night, pastel. |
| **Mono** | Classic, high key, noir, and a sepia toned by the film stage. |
| **Scene** | Landscape pop, a sky gradient, interior, food, snow and beach. |
| **Fix** | One problem each: underexposed, overexposed, hazy, high ISO, flat JPEG. |
| **Car** | Glossy paint, studio white, night neon, golden hour, plus mask-carrying ones for a speed pan, background bokeh and plate blurring. |

One ordering consequence is worth stating, because it decides what a preset in
this file can be: **saturation runs after the colour mixer and the grading
wheels**, so a toned monochrome cannot be built from `saturation: -100` plus a
grade — the grade is exactly what the saturation then removes. Sepia gets its
colour from the film stage instead, which is the one colour stage running after
saturation.

- Built-ins are read-only; tweak and save your own.
  Presets load **additively** by default, so a local preset drops its mask onto
  whatever you already have instead of wiping it — stack "background bokeh" and
  "blur a plate" on top of your own grade, and a slider the preset leaves
  neutral keeps your value. Switch to *replace* for a clean slate. Bulk apply
  has the same choice, so you can add one mask to every pick without touching
  their individual grades.

## Pipeline order

The order the stages run in is not arbitrary, and two parts of it are worth
knowing because they decide what a slider means:

1. **Grade** — white balance, then tone, then dehaze, denoise and texture, then
   clarity, then the colour mixer, then vibrance and saturation, then the
   optical effects (defocus, smear, bloom), then
   sharpen, then the vignette. Masks run after the global pass, each blending a
   locally graded copy through its own alpha. The middle three are in that order
   for a reason: haze is a property of the light, so it goes before anything
   reads detail out of the frame, and detail is amplified only *after* the noise
   under it has been dealt with. The mixer comes before vibrance and saturation
   for the same kind of reason: it decides what each colour *is*, and those two
   then decide how much of all of them there is.
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

**Denoise** is the one deliberate exception. Noise is a per-pixel quantity, so a
fitted preview has already averaged most of it away and there is nothing left
there to remove; its strength is measured in levels at the resolution being
rendered. Judge it at 1:1, which is what Lightroom asks of you for the same
reason.
