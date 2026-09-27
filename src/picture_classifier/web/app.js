"use strict";

// ---------- layouts ----------
const LAYOUTS = {
  1: { cols: 1, rows: 1 },
  2: { cols: 2, rows: 1 },
  4: { cols: 2, rows: 2 },
  8: { cols: 4, rows: 2 },
};

// ---------- state ----------
const state = {
  photos: [],
  byScene: new Map(),
  sceneOrder: [],
  selectedScene: null,
  filter: "all",
  personFilter: new Set(),  // person ids required (any-match)
  filteredPhotos: [],
  pageSize: 4,
  cursorIdx: 0,             // index into filteredPhotos (focused tile)
  modal: { open: false, idx: 0, fit: true, compare: false },
  people: [],
  peopleById: new Map(),
  subjects: { classes: [], counts: {}, vehicles: [], presets: {}, available: [] },
  subjectClassFilter: new Set(),  // COCO class names required (any-match)
  subjectGroupFilter: new Set(),  // vehicle group ids required (any-match)
  showBoxes: false,
  showPeak: false,                // focus-peaking overlay on tiles + viewer
  peakLevel: "normal",            // how strict the sharpness test is
  brackets: [],
  hdrLook: {},
  presets: [],
  presetMode: "add",        // additive by default: local presets stack
  selection: new Set(),   // rel_paths selected for batch actions
  minRating: 0,           // show photos with at least this many stars
  labelFilter: "",        // show photos with this colour label ("" = any)
};

// Mirror of hdr.DEFAULT_LOOK — the realtor-style starting point.
const LOOK_DEFAULT = { shadows: 0.22, brightness: 1.05, clarity: 1.6, saturation: 1.15, sharpen: 0.45 };

// Editor: the pro adjustment set, grouped. Drives slider render/reset/sync and
// must mirror editing.DEFAULT_EDIT on the server (all sliders neutral at 0).
const EDIT_SCHEMA = {
  light: { title: "Light", fields: [
    { k: "exposure",   label: "Exposure",    min: -2,   max: 2,   step: 0.05, fmt: 2 },
    { k: "contrast",   label: "Contrast",    min: -100, max: 100, step: 1,    fmt: 0 },
    { k: "highlights", label: "Highlights",  min: -100, max: 100, step: 1,    fmt: 0 },
    { k: "shadows",    label: "Shadows",     min: -100, max: 100, step: 1,    fmt: 0 },
    { k: "whites",     label: "Whites",      min: -100, max: 100, step: 1,    fmt: 0 },
    { k: "blacks",     label: "Blacks",      min: -100, max: 100, step: 1,    fmt: 0 },
  ]},
  color: { title: "Color", fields: [
    { k: "temp",       label: "Temperature", min: -100, max: 100, step: 1, fmt: 0 },
    { k: "tint",       label: "Tint",        min: -100, max: 100, step: 1, fmt: 0 },
    { k: "vibrance",   label: "Vibrance",    min: -100, max: 100, step: 1, fmt: 0 },
    { k: "saturation", label: "Saturation",  min: -100, max: 100, step: 1, fmt: 0 },
  ]},
  detail: { title: "Detail & Effects", fields: [
    { k: "texture",    label: "Texture",     min: -100, max: 100, step: 1, fmt: 0,
      hint: "The finest detail only. Negative smooths skin without flattening the face; positive finds fabric and bark." },
    { k: "clarity",    label: "Clarity",     min: -100, max: 100, step: 1, fmt: 0 },
    { k: "dehaze",     label: "Dehaze",      min: -100, max: 100, step: 1, fmt: 0,
      hint: "Cuts through atmosphere, or puts it back when negative." },
    { k: "denoise",    label: "Denoise",     min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Noise lives at the pixel level, so a fit preview has already averaged it away \u2014 judge this at 1:1." },
    { k: "sharpen",    label: "Sharpen",     min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Amount. The other three do nothing until this is above zero." },
    { k: "sharpen_radius",  label: "\u00b7 radius",  min: 0, max: 100, step: 1, fmt: 0,
      hint: "The size of the edge being sharpened, 0.5 to 3 pixels. Neutral in the middle." },
    { k: "sharpen_detail",  label: "\u00b7 detail",  min: 0, max: 100, step: 1, fmt: 0,
      hint: "How much of the finest structure comes up with it. High also brings up noise." },
    { k: "sharpen_masking", label: "\u00b7 masking", min: 0, max: 100, step: 1, fmt: 0,
      hint: "Hold sharpening off flat areas \u2014 skies and skin keep their noise unamplified." },
    { k: "vignette",   label: "Vignette",    min: -100, max: 100, step: 1, fmt: 0 },
  ]},
  creative: { title: "Creative", fields: [
    { k: "blur",         label: "Defocus",   min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Disc-shaped lens blur. Put it on an inverted radial mask for fake shallow depth of field." },
    { k: "motion",       label: "Motion",    min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Directional smear — mask the car out of it to fake a panned shot." },
    { k: "motion_angle", label: "· angle",   min: -180, max: 180, step: 1, fmt: 0,
      hint: "Direction of the motion blur, in degrees. 0 is horizontal." },
    { k: "glow",         label: "Glow",      min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Bloom off the highlights — headlights, neon, low sun." },
    { k: "pixelate",     label: "Mosaic",    min: 0,    max: 100, step: 1, fmt: 0,
      hint: "Censor blocks. Mask it over a number plate or a face." },
  ]},
};
const EDIT_FIELDS = Object.values(EDIT_SCHEMA).flatMap((g) => g.fields);
// Mirrors server.EDIT_SLOTS. Per-photo scratchpad: stash the working edit,
// try something else, bring the first one back.
const EDIT_SLOTS = 6;
const CURVE_IDENTITY = [[0, 0], [1, 1]];
// Mirrors editing.CURVE_KEYS. The master runs first — it says how bright a tone
// is — and the three channel curves then say what colour it takes there.
const CURVE_CHANNELS = [
  { k: "curve",   label: "RGB", tint: null },      // null: use the UI accent
  { k: "curve_r", label: "R",   tint: "#e35d5d" },
  { k: "curve_g", label: "G",   tint: "#5fbf6a" },
  { k: "curve_b", label: "B",   tint: "#5b8ee6" },
];

// Mirrors watermark.DEFAULT_WATERMARK on the server.
const WM_STYLES = ["minimal", "bar", "plate", "corner", "filmstrip"];
const WM_POSITIONS = ["top-left", "top-center", "top-right",
                      "bottom-left", "bottom-center", "bottom-right"];
const WM_TOKENS = ["name", "camera", "lens", "focal", "aperture", "shutter",
                   "iso", "date", "time", "file"];
const WM_DEFAULT = {
  enabled: false, style: "minimal", position: "bottom-right",
  name: "", camera: "",
  line1: "{name}", line2: "{camera} · {lens}",
  line3: "{focal} · {aperture} · {shutter} · {iso}",
  size: 100, opacity: 90, color: "#ffffff", margin: 100,
};
function cloneWatermark(w) { return w ? { ...WM_DEFAULT, ...w } : null; }
function watermarkIsNeutral(w) {
  if (!w || !w.enabled || w.opacity <= 0) return true;
  return !["line1", "line2", "line3"].some((k) => (w[k] || "").trim());
}
// Canonical form for dirty-tracking: the server stores null for anything that
// would not print, so an off watermark must compare equal to no watermark.
function canonWatermark(w) {
  if (watermarkIsNeutral(w)) return "null";
  const full = { ...WM_DEFAULT, ...w };
  return JSON.stringify(Object.keys(WM_DEFAULT).sort().map((k) => full[k]));
}

const EDIT_NEUTRAL = (() => {
  // tilt and crop sit outside EDIT_SCHEMA on purpose: they are frame geometry,
  // not a slider a mask could ever carry, and they are applied before anything
  // tonal. Mirrors editing.DEFAULT_EDIT.
  const e = { masks: [], watermark: null,
              film: null, hsl: null, grading: null, lut: null,
              healing: null, redeye: null, lens: null, transform: null, portrait: null,
              tilt: 0, crop: null };
  for (const c of CURVE_CHANNELS) e[c.k] = CURVE_IDENTITY.map((p) => p.slice());
  // Not every slider is neutral at zero: sharpen_radius sits in the middle,
  // mirroring editing.DEFAULT_EDIT. Anything else here would make a freshly
  // opened photo look dirty.
  const NEUTRAL_OVERRIDES = { sharpen_radius: 50, sharpen_detail: 25 };
  for (const f of EDIT_FIELDS) e[f.k] = NEUTRAL_OVERRIDES[f.k] ?? 0;
  return e;
})();
// Sliders a local mask may carry — mirrors editing.LOCAL_KEYS (no vignette, no
// curve: those stay frame-wide). Rows outside this set hide in mask mode.
const MASK_LOCAL_KEYS = new Set(EDIT_FIELDS.map((f) => f.k).filter((k) => k !== "vignette"));
const MASK_MAX = 16;

function fieldByKey(k) { return EDIT_FIELDS.find((f) => f.k === k); }
function mergeNeutralEdit(edit) {
  const e = { ...EDIT_NEUTRAL };
  for (const c of CURVE_CHANNELS) {
    const src = edit && edit[c.k];
    e[c.k] = (Array.isArray(src) && src.length)
      ? src.map((p) => p.slice()) : CURVE_IDENTITY.map((p) => p.slice());
  }
  e.masks = (edit && Array.isArray(edit.masks)) ? edit.masks.map(cloneMask) : [];
  e.watermark = (edit && edit.watermark) ? cloneWatermark(edit.watermark) : null;
  e.tilt = (edit && Number(edit.tilt)) || 0;
  e.crop = (edit && edit.crop) ? { ...edit.crop } : null;
  e.film = (edit && edit.film) ? { ...edit.film } : null;
  e.hsl = cloneHsl(edit && edit.hsl);
  e.grading = cloneGrading(edit && edit.grading);
  e.lut = (edit && edit.lut) ? { ...edit.lut } : null;
  // The repairs were not copied here at all, so opening a photo that had one
  // and pressing Save wrote the edit back without it.
  e.healing = (edit && edit.healing && (edit.healing.ops || []).length)
    ? { ops: edit.healing.ops.map((o) => JSON.parse(JSON.stringify(o))) } : null;
  e.redeye = (edit && edit.redeye && (edit.redeye.corrections || []).length)
    ? { enabled: true, corrections: edit.redeye.corrections.map((c) => ({ ...c })) } : null;
  e.lens = (edit && edit.lens) ? { ...edit.lens } : null;
  e.portrait = (edit && edit.portrait && PORTRAIT_KEYS.some((k) => edit.portrait[k]))
    ? { ...edit.portrait } : null;
  e.transform = (edit && edit.transform) ? { ...edit.transform } : null;
  if (edit) for (const f of EDIT_FIELDS) if (edit[f.k] != null) e[f.k] = edit[f.k];
  return e;
}
function editsEqual(a, b) {
  for (const f of EDIT_FIELDS) if (Math.abs((a[f.k] || 0) - (b[f.k] || 0)) > 1e-4) return false;
  for (const c of CURVE_CHANNELS) {
    const ca = a[c.k] || CURVE_IDENTITY, cb = b[c.k] || CURVE_IDENTITY;
    if (ca.length !== cb.length) return false;
    for (let i = 0; i < ca.length; i++)
      if (Math.abs(ca[i][0] - cb[i][0]) > 1e-4 || Math.abs(ca[i][1] - cb[i][1]) > 1e-4) return false;
  }
  if (canonWatermark(a.watermark) !== canonWatermark(b.watermark)) return false;
  if (Math.abs((a.tilt || 0) - (b.tilt || 0)) > 1e-4) return false;
  if (canonCrop(a.crop) !== canonCrop(b.crop)) return false;
  if (canonFilm(a.film) !== canonFilm(b.film)) return false;
  if (canonHsl(a.hsl) !== canonHsl(b.hsl)) return false;
  if (canonGrading(a.grading) !== canonGrading(b.grading)) return false;
  if (canonLut(a.lut) !== canonLut(b.lut)) return false;
  if (JSON.stringify(a.healing || null) !== JSON.stringify(b.healing || null)) return false;
  if (JSON.stringify(a.redeye || null) !== JSON.stringify(b.redeye || null)) return false;
  if (canonOptic(a.lens, LENS_DEFAULT) !== canonOptic(b.lens, LENS_DEFAULT)) return false;
  for (const k of PORTRAIT_KEYS) {
    if (((a.portrait && a.portrait[k]) || 0) !== ((b.portrait && b.portrait[k]) || 0)) return false;
  }
  if (canonOptic(a.transform, TRANSFORM_DEFAULT) !== canonOptic(b.transform, TRANSFORM_DEFAULT)) return false;
  return canonMasks(a.masks) === canonMasks(b.masks);
}
// Mirrors lens.normalize and transform.normalize: at its defaults a panel is no
// correction, whatever object is holding the numbers.
function canonOptic(o, defaults) {
  if (!o || opticIsNeutral(o, defaults)) return "";
  return JSON.stringify(Object.keys(defaults).sort().map((k) => o[k] ?? defaults[k]));
}
// Mirrors editing.normalize_lut_ref: a look at zero amount is no look.
function canonLut(l) {
  return l && l.key && l.amount > 0 ? `${l.key}:${l.amount}` : "";
}
function canonFilm(f) {
  if (!f || !f.enabled) return "";
  return JSON.stringify(Object.keys(FILM_DEFAULT).sort()
    .filter((k) => k !== "stock").map((k) => f[k]));
}
// Mirrors editing.normalize_hsl: values at neutral are dropped, an empty band
// is dropped, and an empty mixer is null — so a band the user opened and left
// alone never counts as an edit.
function canonHsl(h) {
  if (!h) return "";
  const out = [];
  for (const b of HSL_BANDS) {
    const src = h[b.k];
    if (!src) continue;
    const vals = HSL_FIELDS.filter((f) => Math.round(src[f.k] || 0) !== 0)
      .map((f) => [f.k, Math.round(src[f.k])]);
    if (vals.length) out.push([b.k, vals]);
  }
  return out.length ? JSON.stringify(out) : "";
}
function cloneHsl(h) {
  if (!h) return null;
  const out = {};
  for (const key of Object.keys(h)) if (h[key]) out[key] = { ...h[key] };
  return Object.keys(out).length ? out : null;
}
// Mirrors grading.normalize: a zone with sat 0 and lum 0 contributes nothing
// whatever its hue, so hue alone must not read as an edit.
function canonGrading(g) {
  if (!g) return "";
  const out = [];
  for (const z of GRADE_ZONES) {
    const src = g[z.k];
    if (!src) continue;
    const sat = Math.round(src.sat || 0), lum = Math.round(src.lum || 0);
    if (!sat && !lum) continue;
    out.push([z.k, Math.round(src.hue || 0), sat, lum]);
  }
  const blending = Math.round(g.blending == null ? 50 : g.blending);
  const balance = Math.round(g.balance || 0);
  if (!out.length) return "";
  return JSON.stringify([out, blending, balance]);
}
function cloneGrading(g) {
  if (!g) return null;
  const out = {};
  for (const k of Object.keys(g)) {
    out[k] = (g[k] && typeof g[k] === "object") ? { ...g[k] } : g[k];
  }
  return out;
}
function canonCrop(c) {
  return c ? JSON.stringify([rnd4(c.x), rnd4(c.y), rnd4(c.w), rnd4(c.h)]) : "";
}
function isNeutralEdit(edit) { return editsEqual(mergeNeutralEdit(edit), EDIT_NEUTRAL); }

// ---------- local-adjustment masks (model) ----------
// Mirrors editing._default_mask / normalize_mask on the server. Geometry is in
// normalized image coordinates (x = fraction of the width, y of the height), so
// the same numbers describe the shape on the preview and on the export.
const MASK_KINDS = {
  radial: { icon: "radial", label: "Radial" },
  linear: { icon: "gradient", label: "Gradient" },
  brush:  { icon: "brush", label: "Brush" },
  auto:   { icon: "wand", label: "Auto" },
  range:  { icon: "palette", label: "Range" },
};

// Mirrors rangemask.LUMA_DEFAULT. A refinement any mask may carry, which is what
// makes "the subject, but only its highlights" one mask instead of two.
const LUMA_RANGE_FIELDS = [
  { k: "lo", label: "From", min: 0, max: 100,
    hint: "The darkest tone selected, in perceptual lightness." },
  { k: "hi", label: "To", min: 0, max: 100,
    hint: "The brightest tone selected." },
  { k: "feather_lo", label: "\u00b7 soften low", min: 0, max: 100,
    hint: "How gradually the selection fades in below From." },
  { k: "feather_hi", label: "\u00b7 soften high", min: 0, max: 100,
    hint: "How gradually it fades out above To." },
];
// A new range mask must select something, or the server normalizes it away the
// moment it is saved. The darker half is a starting point you can see.
const LUMA_RANGE_NEW = { lo: 0, hi: 50, feather_lo: 10, feather_hi: 10 };
function lumaRangeIsAll(r) {
  return !r || (Math.round(r.lo) <= 0 && Math.round(r.hi) >= 100);
}

// Mirrors segment.CLASS_GROUPS. An automatic mask has no shape to drag: what it
// selects is a segmentation of the photo, so the only control is which group.
const AUTO_GROUPS = [
  { k: "subject",    label: "Subject" },
  { k: "background", label: "Background" },
  { k: "skin",       label: "Skin" },
  { k: "face",       label: "Face" },
  { k: "hair",       label: "Hair" },
  { k: "clothes",    label: "Clothes" },
];
function autoGroupLabel(k) {
  const g = AUTO_GROUPS.find((x) => x.k === k);
  return g ? g.label : k;
}

function neutralAdj() {
  const a = {};
  // Mirrors editing._neutral_adj: read the neutral off EDIT_NEUTRAL rather than
  // assuming zero, so a slider whose neutral is its midpoint does not make every
  // new mask look active.
  for (const k of MASK_LOCAL_KEYS) a[k] = EDIT_NEUTRAL[k] ?? 0;
  return a;
}

function newMask(kind, aspect, group) {
  const m = {
    type: kind, name: "", enabled: true, invert: false,
    feather: kind === "linear" ? 100 : 50, amount: 100, adj: neutralAdj(),
    range_luma: null, range_color: null,
  };
  if (kind === "radial") Object.assign(m, { cx: 0.5, cy: 0.5, rx: 0.25, ry: 0.25 * (aspect || 1), angle: 0 });
  else if (kind === "linear") Object.assign(m, { x1: 0.5, y1: 0.15, x2: 0.5, y2: 0.55 });
  else if (kind === "auto") { m.group = group || "subject"; m.feather = 0; }
  else if (kind === "range") { m.range_luma = { ...LUMA_RANGE_NEW }; m.feather = 0; }
  else m.strokes = [];
  return m;
}

function cloneMask(m) {
  const c = { ...m, adj: { ...neutralAdj(), ...(m.adj || {}) } };
  if (m.strokes) c.strokes = m.strokes.map((s) => ({ ...s, points: s.points.map((p) => p.slice()) }));
  c.range_luma = m.range_luma ? { ...m.range_luma } : null;
  c.range_color = m.range_color
    ? { ...m.range_color, samples: (m.range_color.samples || []).map((v) => v.slice()) }
    : null;
  return c;
}

const rnd4 = (v) => Math.round((Number(v) || 0) * 1e4) / 1e4;

// A stable string form used only to answer "is this dirty?" — the server may
// round values on the way back, so compare at the precision both sides keep.
function canonMasks(masks) {
  const list = persistableMasks(masks);
  if (!list.length) return "[]";
  return JSON.stringify(list.map((m) => {
    const o = { t: m.type, e: !!m.enabled, i: !!m.invert, f: Math.round(m.feather),
                a: Math.round(m.amount), n: m.name || "",
                adj: [...MASK_LOCAL_KEYS].sort().map((k) => rnd4(m.adj && m.adj[k])) };
    // Without this, narrowing a mask's tonal range would not register as a
    // change and Save would stay disabled.
    o.rl = m.range_luma
      ? LUMA_RANGE_FIELDS.map((f) => Math.round(m.range_luma[f.k] || 0)) : null;
    o.rc = m.range_color
      ? [(m.range_color.samples || []).map((v) => v.map(Math.round)),
         Math.round(m.range_color.range || 0), Math.round(m.range_color.feather || 0)]
      : null;
    if (m.type === "radial") o.g = [m.cx, m.cy, m.rx, m.ry, m.angle].map(rnd4);
    else if (m.type === "linear") o.g = [m.x1, m.y1, m.x2, m.y2].map(rnd4);
    // The group is the whole of an automatic mask's geometry. Leave it out and
    // switching Subject to Hair would not register as a change.
    else if (m.type === "auto") o.g = m.group;
    else o.g = (m.strokes || []).map((s) => [rnd4(s.radius), !!s.erase,
                                             s.points.map((p) => [rnd4(p[0]), rnd4(p[1])])]);
    return o;
  }));
}

function maskAdjNeutral(m) {
  // Against EDIT_NEUTRAL, not against zero — mirrors editing._adj_is_neutral,
  // and the reason is the same: sharpen_radius is neutral at its midpoint.
  return [...MASK_LOCAL_KEYS].every(
    (k) => Math.abs((Number(m.adj[k]) || 0) - (EDIT_NEUTRAL[k] ?? 0)) < 1e-4);
}

function maskLabel(m, idx) {
  if (m.name) return m.name;
  // "Subject 2" says more than "Auto 2" ever could.
  if (m.type === "auto") return `${autoGroupLabel(m.group)} ${idx + 1}`;
  return `${MASK_KINDS[m.type].label} ${idx + 1}`;
}

// ---------- icons ----------
// One stroked set, drawn in currentColor at a fixed weight, so an icon takes the
// colour and size of the text beside it and looks the same on every platform.
// This replaces the emoji and the assorted arrows and geometric glyphs that used
// to stand in for icons: those rendered differently per OS, could not be sized or
// coloured with the rest of the interface, and several of them were carrying real
// navigational weight.
//
// 24x24 viewBox, 1.75 stroke, round caps. Static markup asks for one with
// `data-icon="name"`; generated markup calls `icon("name")`.
const ICONS = {
  crop: "M6 2v14a2 2 0 0 0 2 2h14M2 6h14a2 2 0 0 1 2 2v14",
  settings: "M12 15.5A3.5 3.5 0 1 0 12 8.5a3.5 3.5 0 0 0 0 7z"
    + "M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06"
    + "a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09"
    + "a1.65 1.65 0 0 0-1.08-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06"
    + "a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3"
    + "a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 8.17a1.65 1.65 0 0 0-.33-1.82l-.06-.06"
    + "a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3"
    + "a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06"
    + "a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V10a1.65 1.65 0 0 0 1.51 1H21"
    + "a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z",
  refresh: "M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6",
  close: "M18 6 6 18M6 6l12 12",
  download: "M12 3v12m0 0 4-4m-4 4-4-4M4 19h16",
  save: "M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zM17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7M7 3v4a1 1 0 0 0 1 1h7",
  pencil: "M12 20h9M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z",
  plus: "M12 5v14M5 12h14",
  undo: "M3 10h11a5 5 0 0 1 0 10H8M3 10l4-4M3 10l4 4",
  power: "M12 2v10M18.36 6.64a9 9 0 1 1-12.73 0",
  trash: "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v5M14 11v5",
  more: "M12 6.5h.01M12 12h.01M12 17.5h.01",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 11v5.5M12 7.5h.01",
  left: "M15 18 9 12l6-6",
  right: "M9 6l6 6-6 6",
  person: "M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2M12 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  open: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2H3zM3 9h18l-2 9H5z",
  focus: "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 2v3m0 14v3M2 12h3m14 0h3",
  boxes: "M4 5h7v6H4zM13 5h7v6h-7zM4 13h7v6H4zM13 13h7v6h-7z",
  hdr: "M3 17h18M6 13a6 6 0 0 1 12 0M12 3v3M5 6l2 2m12-2-2 2",
  wand: "M5 19 17 7M15 3l1 3 3 1-3 1-1 3-1-3-3-1 3-1zM4 13l.7 2 2 .7-2 .7L4 19l-.7-2-2-.7 2-.7z",
  loupe: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM16 16l5 5",
  pipette: "M2 22l1-1h3l9-9M3 21v-3l9-9M15 6l3.4-3.4a2.1 2.1 0 1 1 3 3L18 9l.4.4a2.1 2.1 0 1 1-3 3l-3.8-3.8a2.1 2.1 0 1 1 3-3z",
  palette: "M12 21a9 9 0 1 1 9-9c0 2-1.5 3-3 3h-2a2 2 0 0 0-1 3.7A2 2 0 0 1 12 21zM7.5 11h.01M11 7.5h.01M15.5 9h.01",
  swap: "M4 8h13l-3-3M20 16H7l3 3",
  radial: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z",
  gradient: "M4 4h16v16H4zM4 11h16M4 15h16M4 18h16",
  brush: "M17 3a3 3 0 0 1 4 4l-9 9-4 1 1-4zM7 14c-2 1-3 3-3 6 3 0 5-1 6-3z",
  warning: "M12 3 2 20h20zM12 9v5M12 17.5h.01",
  check: "M20 6 9 17l-5-5",
  eyeOpen: "M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"
    + "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  eyeShut: "M3 3l18 18M10.6 5.2A10.9 10.9 0 0 1 12 5c6.5 0 10 7 10 7a17 17 0 0 1-2.4 3.2M6.4 6.4A17 17 0 0 0 2 12s3.5 7 10 7c1.6 0 3-.4 4.2-1"
};

function icon(name, cls) {
  const d = ICONS[name];
  if (!d) return "";
  return `<svg class="icon${cls ? " " + cls : ""}" viewBox="0 0 24 24" `
    + `fill="none" stroke="currentColor" stroke-width="1.75" `
    + `stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">`
    + `<path d="${d}"/></svg>`;
}

// Static markup carries `data-icon` and gets the svg put in front of its text, so
// index.html stays readable instead of holding thirty path definitions.
// Replace a button's words without disturbing its icon. The first call absorbs
// whatever text the markup started with, so the label is not printed twice.
function setBtnLabel(btn, text) {
  let span = btn.querySelector(".btn-label");
  if (!span) {
    [...btn.childNodes].forEach((n) => {
      if (n.nodeType === Node.TEXT_NODE) n.remove();
    });
    span = document.createElement("span");
    span.className = "btn-label";
    btn.appendChild(span);
  }
  span.textContent = text;
}

function hydrateIcons(root) {
  (root || document).querySelectorAll("[data-icon]").forEach((el) => {
    if (el.dataset.iconDone) return;
    el.dataset.iconDone = "1";
    // The label goes in a span and an icon-only control is marked as such,
    // because CSS cannot tell the two apart on its own: :only-child counts
    // elements and not text, so a button holding an icon and a bare text node
    // matched "icon only" and got squashed to a square with its words cut off.
    const text = el.textContent.trim();
    if (text) {
      el.textContent = "";
      const span = document.createElement("span");
      span.className = "btn-label";
      span.textContent = text;
      el.appendChild(span);
    } else {
      el.classList.add("icon-only");
    }
    el.insertAdjacentHTML("afterbegin", icon(el.dataset.icon));
  });
}

// ---------- tooltips with keyboard shortcuts ----------
// Native `title` is slow to appear and can't show a keycap, and the shortcuts
// were previously only discoverable from the one help strip at the bottom of
// the viewer. One floating element serves every control: no per-button markup,
// and nothing gets clipped by a scrolling panel the way a CSS ::after would.
// Shortcut labels follow the platform: ⌘ on a Mac, Ctrl everywhere else.
const IS_MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MOD_KEY = IS_MAC ? "⌘" : "Ctrl+";

const SHORTCUT_TIPS = {
  // grid + header
  "#prev-page": ["Previous page", "["],
  "#next-page": ["Next page", "]"],
  "#peak-btn": ["Highlight what is actually in focus", "K"],
  "#boxes-btn": ["Show detected subject boxes", "B"],
  "#hdr-btn": ["Review and fix auto-detected HDR brackets", null],
  "#rescore-btn": ["Choose what to detect and re-score. Keeps decisions", null],
  "#people-cluster-btn": ["Group faces into people and subjects into groups", null],
  "#cluster-settings-btn": ["How strictly to group faces and subjects", null],
  "#people-manage-btn": ["Rename people, set priority, exclude clusters", null],
  "#reject-undecided-btn": ["Mark every undecided photo in this scene as reject", null],
  "#export-picks-btn": ["Copy all PICK photos to a folder", null],
  "#selection-apply": ["Apply an edit or preset to the selection", null],
  "#selection-download": ["Save the selected photos to your Downloads folder", "D"],
  // viewer
  "#modal-close": ["Close", "Esc"],
  "#modal-edit": ["Edit this photo", "E"],
  "#modal-compare": ["Toggle HDR merged vs the 0 EV original", "C"],
  "#modal-loupe": ["Magnify under the cursor to check detail", "L"],
  // editor
  "#edit-modal-close": ["Close the editor", "Esc"],
  "#edit-prev": ["Previous photo", "←"],
  "#edit-next": ["Next photo", "→"],
  "#edit-compare": ["Hold to see the original", "C"],
  "#edit-save": ["Save this edit", "Enter"],
  "#edit-auto": ["Auto-tone from the histogram", null],
  "#edit-undo": ["Undo", `${MOD_KEY}Z`],
  "#edit-redo": ["Redo", IS_MAC ? "⇧⌘Z" : "Ctrl+Y"],
  "#edit-apply-more": ["Apply this edit to more photos", null],
  '[data-preset-mode="add"]': ["Lay the preset on top: its masks are added to yours", null],
  '[data-preset-mode="replace"]': ["Discard the current edit and use the preset alone", null],
  "#wb-pick": ["White balance — click something that should be grey", null],
  "#wb-reset": ["Zero Temperature and Tint", null],
  "#crop-tool": ["Crop — drag a box on the photo", null],
  "#crop-reset": ["Back to the whole frame, level", null],
  "#crop-tilt": ["Straighten. Positive levels a horizon drooping to the right", null],
  '[data-add-mask="radial"]': ["Radial mask — drag an ellipse on the photo", "R"],
  '[data-add-mask="linear"]': ["Gradient mask — drag a direction on the photo", "G"],
  '[data-add-mask="brush"]': ["Brush mask — paint on the photo", "B"],
  "#mask-delete": ["Remove this mask", "Del"],
  "#mask-duplicate": ["Copy this mask", null],
  "#mask-brush-undo": ["Remove the last stroke", null],
  '.edit-zoom[data-zoom="0"]': ["Fit the whole photo", "F"],
  '.edit-zoom[data-zoom="1"]': ["Original pixels, 1:1", "F"],
};

function initTooltips() {
  const tip = document.createElement("div");
  tip.id = "tooltip";
  tip.className = "hidden";
  document.body.appendChild(tip);

  for (const [sel, [text, key]] of Object.entries(SHORTCUT_TIPS)) {
    document.querySelectorAll(sel).forEach((el) => {
      el.dataset.tip = text;
      if (key) el.dataset.key = key;
      // Drop the native tooltip so the two don't both appear.
      el.removeAttribute("title");
      el.setAttribute("aria-label", key ? `${text} (${key})` : text);
    });
  }

  const show = (el) => {
    tip.innerHTML = escapeHtml(el.dataset.tip) +
      (el.dataset.key ? ` <kbd>${escapeHtml(el.dataset.key)}</kbd>` : "");
    tip.classList.remove("hidden");
    const r = el.getBoundingClientRect();
    const t = tip.getBoundingClientRect();
    // Below by default, above when that would run off the bottom.
    const below = r.bottom + 6 + t.height < window.innerHeight;
    tip.style.top = `${below ? r.bottom + 6 : r.top - t.height - 6}px`;
    tip.style.left =
      `${Math.max(6, Math.min(window.innerWidth - t.width - 6, r.left + r.width / 2 - t.width / 2))}px`;
  };
  const hide = () => tip.classList.add("hidden");

  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest("[data-tip]");
    if (el) show(el); else hide();
  });
  document.addEventListener("pointerdown", hide);
  window.addEventListener("scroll", hide, true);
}

// ---------- helpers ----------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const enc = (p) => p.split("/").map(encodeURIComponent).join("/");
// Thumb URL with an edit-version cache-buster so edited photos refresh in the grid.
const thumbUrl = (p) => "/thumb/" + enc(p.rel_path) + (p.edited_at ? `?v=${encodeURIComponent(p.edited_at)}` : "");
function matchesDecisionFilter(p) {
  if (state.filter === "all") return true;
  if (state.filter === "undecided") return p.decision == null;
  // Not a decision, but it sits in the same row because it answers the same
  // question: which of these still want my attention. A neutral edit is dropped
  // from the db, so the field being there means the photo really was changed.
  if (state.filter === "edited") return !!p.edit;
  return p.decision === state.filter;
}
function matchesPersonFilter(p) {
  if (state.personFilter.size === 0) return true;
  for (const f of (p.faces || [])) {
    if (f.person_id && state.personFilter.has(f.person_id)) return true;
  }
  return false;
}
function matchesSubjectFilter(p) {
  const byClass = state.subjectClassFilter.size === 0;
  const byGroup = state.subjectGroupFilter.size === 0;
  if (byClass && byGroup) return true;
  let classHit = byClass, groupHit = byGroup;
  for (const o of (p.objects || [])) {
    if (!classHit && state.subjectClassFilter.has(o.cls)) classHit = true;
    if (!groupHit && o.vehicle_id && state.subjectGroupFilter.has(o.vehicle_id)) groupHit = true;
  }
  return classHit && groupHit;
}
function matchesFilter(p) {
  return matchesDecisionFilter(p) && matchesPersonFilter(p) && matchesSubjectFilter(p)
    && (p.rating || 0) >= state.minRating && (!state.labelFilter || p.label === state.labelFilter);
}

// ---------- stars and colour labels ----------
// Set by the photographer, apart from the decision, and written into exports
// (metadata.build_xmp). The keys are Lightroom's: 1-5 stars, 0 none, 6-9 red,
// yellow, green, blue. Neither moves the cursor on, as they do not there.
const LABEL_KEYS = { 6: "red", 7: "yellow", 8: "green", 9: "blue" };
const LABEL_NAMES = { red: "Red", yellow: "Yellow", green: "Green", blue: "Blue", purple: "Purple" };

function starsHtml(rating, cls = "stars") {
  const r = rating || 0;
  return `<span class="${cls}" aria-label="${r ? `${r} star${r > 1 ? "s" : ""}` : "no stars"}">`
    + [1, 2, 3, 4, 5].map((n) => `<i data-star="${n}" class="${n <= r ? "on" : ""}">★</i>`).join("") + "</span>";
}

function applyMarks(photo, marks) {
  if (marks.rating !== undefined) {
    if (marks.rating) photo.rating = marks.rating; else delete photo.rating;
  }
  if (marks.label !== undefined) {
    if (marks.label) photo.label = marks.label; else delete photo.label;
  }
  saveMarks(photo, marks);
}

function markAt(absIdx, marks) {
  const photo = state.filteredPhotos[absIdx];
  if (!photo) return;
  applyMarks(photo, marks);
  recomputeFilter();
  const at = state.filteredPhotos.indexOf(photo);
  // A filter on stars or a label can drop the photo; the next one is then in
  // its place, as with a decision.
  state.cursorIdx = at >= 0 ? at : Math.max(0, Math.min(absIdx, state.filteredPhotos.length - 1));
  showCursor();
  if (state.modal.open) {
    if (!state.filteredPhotos.length) { closeModal(); return; }
    state.modal.idx = state.cursorIdx;
    renderModal();
  }
}

function saveMarks(photo, marks) {
  decisionSaves = decisionSaves.then(async () => {
    const res = await fetch("/api/mark", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rel_paths: [photo.rel_path], ...marks }),
    });
    if (!res.ok) throw new Error(`mark failed: ${res.status}`);
  }).catch((err) => {
    alert(`Saving a star or label failed (${err.message}). The project will reload.`);
    location.reload();
  });
}

// The mark a key stands for, or undefined for any other key.
function markForKey(k, photo) {
  if (/^[0-5]$/.test(k)) return { rating: Number(k) };
  if (LABEL_KEYS[k]) return { label: photo && photo.label === LABEL_KEYS[k] ? "" : LABEL_KEYS[k] };
  return undefined;
}

// Focus peaking rides on the same aspect-ratio trick as the boxes: the overlay
// is generated at thumbnail size, so scaling it to the drawn image rect lines
// it up exactly.
const peakUrl = (p) =>
  `/peak/${enc(p.rel_path)}?level=${state.peakLevel}` +
  (p.edited_at ? `&v=${encodeURIComponent(p.edited_at)}` : "");

function peakLayerHtml(p) {
  if (!state.showPeak || !p.width || !p.height) return "";
  return `<img class="peak-layer" loading="lazy" alt="" src="${peakUrl(p)}" ` +
    `style="aspect-ratio:${p.width}/${p.height}" />`;
}

// Boxes are drawn as percentages of the photo, inside a layer that reproduces
// the object-fit: contain rect via aspect-ratio — no measuring, and it stays
// aligned through every resize.
// A detection box is stored against the original frame, because that is the
// frame the detector saw. Once a photo is cropped or straightened the thumbnail
// is a different frame, so each box is mapped through the photo's geometry
// (supplied by the server as `geom`) before it is drawn. Boxes the crop excludes
// are dropped; the rest land on the car they were found on.
function boxToView(o, p) {
  const [x, y, w, h] = o.bbox_xywh;
  let l = x / p.width, t = y / p.height;
  let r = (x + w) / p.width, b = (y + h) / p.height;
  const g = p.geom;
  if (g) {
    const m = g.xform;
    const at = (u, v) => [m[0] * u + m[1] * v + m[2], m[3] * u + m[4] * v + m[5]];
    // All four corners, because a straighten turns the box: the axis-aligned
    // bounds of the turned box are what can be drawn as a plain rectangle.
    const pts = [at(l, t), at(r, t), at(l, b), at(r, b)];
    l = Math.min(...pts.map((q) => q[0])); r = Math.max(...pts.map((q) => q[0]));
    t = Math.min(...pts.map((q) => q[1])); b = Math.max(...pts.map((q) => q[1]));
    if (r <= 0 || l >= 1 || b <= 0 || t >= 1) return null;   // cropped away
    l = Math.max(0, l); t = Math.max(0, t);
    r = Math.min(1, r); b = Math.min(1, b);
  }
  return { l, t, w: r - l, h: b - t };
}

function boxLayerHtml(p) {
  if (!state.showBoxes || !p.width || !p.height) return "";
  const objs = p.objects || [];
  if (!objs.length) return "";
  const boxes = objs.map((o) => {
    const v = boxToView(o, p);
    if (!v) return "";
    const grouped = o.vehicle_id ? ` grouped` : "";
    const label = o.vehicle_id
      ? (state.subjects.vehicles.find((x) => x.id === o.vehicle_id)?.label || o.cls)
      : o.cls;
    return `<span class="det-box${grouped}" style="left:${v.l * 100}%;` +
      `top:${v.t * 100}%;width:${v.w * 100}%;height:${v.h * 100}%">` +
      `<span class="det-label">${escapeHtml(label)} ${Math.round(o.score * 100)}</span></span>`;
  }).join("");
  if (!boxes) return "";
  // The layer reproduces the object-fit: contain rect via aspect-ratio, so it
  // has to be the aspect of what is *shown* — the cropped frame, when there is
  // one — rather than the original's.
  const ar = p.geom ? `${p.geom.w}/${p.geom.h}` : `${p.width}/${p.height}`;
  return `<span class="box-layer" style="aspect-ratio:${ar}">${boxes}</span>`;
}

function bestPriority(photo) {
  let best = Infinity;
  for (const f of (photo.faces || [])) {
    if (!f.person_id) continue;
    const person = state.peopleById.get(f.person_id);
    if (!person || person.excluded) continue;
    if (person.priority < best) best = person.priority;
  }
  return best;
}

function prunePersonFilter() {
  for (const id of [...state.personFilter]) {
    const person = state.peopleById.get(id);
    if (!person || person.excluded) state.personFilter.delete(id);
  }
}
// The grid scrolls by row, and a "page" is a screenful: `pageSize` photos,
// LAYOUTS' rows of its columns. These read where the grid is scrolled to.
const GRID_PAD = 10, GRID_GAP = 10;

function gridGeom() {
  const L = LAYOUTS[state.pageSize];
  const h = $("#grid").clientHeight;
  const rowH = Math.max(120, (h - 2 * GRID_PAD - GRID_GAP * (L.rows - 1)) / L.rows);
  const totalRows = Math.ceil(state.filteredPhotos.length / L.cols);
  return { cols: L.cols, rows: L.rows, rowH, step: rowH + GRID_GAP, totalRows,
           maxTop: Math.max(0, totalRows - L.rows) };
}

// The row at the top of the screen.
function topRow() {
  const g = gridGeom();
  return Math.min(g.maxTop, Math.max(0, Math.round($("#grid").scrollTop / g.step)));
}

const pageIdx = () => Math.floor(topRow() / LAYOUTS[state.pageSize].rows);
const pageCount = () => Math.max(1, Math.ceil(gridGeom().totalRows / LAYOUTS[state.pageSize].rows));
const visiblePhotos = () => {
  const g = gridGeom(), start = topRow() * g.cols;
  return state.filteredPhotos.slice(start, start + g.rows * g.cols);
};

// Scroll just enough to have photo `i` wholly on screen.
function ensureVisible(i) {
  const grid = $("#grid"), g = gridGeom();
  const row = Math.floor(i / g.cols);
  const y0 = row * g.step;
  if (y0 < grid.scrollTop) grid.scrollTop = y0;
  else if (y0 + g.rowH + 2 * GRID_PAD > grid.scrollTop + grid.clientHeight) {
    grid.scrollTop = (row - g.rows + 1) * g.step;
  }
}

// After the list or the layout changes: lay the grid out, bring the cursor
// into view, and draw.
function showCursor() {
  renderGrid();
  if (state.filteredPhotos.length) ensureVisible(state.cursorIdx);
  renderMain();
}

// ---------- API ----------
async function loadDb() {
  const res = await fetch("/api/db");
  if (!res.ok) throw new Error(`db load failed: ${res.status}`);
  const data = await res.json();
  state.projectDir = data.project_dir;
  $("#project-name").textContent = data.project_name || "";
  $("#project-name").title = data.project_name || "";
  state.photos = data.photos;
  state.byScene = new Map();
  state.sceneOrder = [];
  for (const p of state.photos) {
    if (!state.byScene.has(p.scene)) {
      state.byScene.set(p.scene, []);
      state.sceneOrder.push(p.scene);
    }
    state.byScene.get(p.scene).push(p);
  }
  // In name order, numbers read as numbers, so time-gap scenes run Scene 01,
  // 02, 03 (they came in the order their first photo was filed, which put 02
  // first). Photos in no scene of their own go last.
  const loose = (name) => name.startsWith("(");
  state.sceneOrder.sort((a, b) => (loose(a) - loose(b))
    || a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));
  state.people = data.people || [];
  state.peopleById = new Map(state.people.map((p) => [p.id, p]));
  state.brackets = data.brackets || [];
  state.hdrLook = data.hdr_look || { ...LOOK_DEFAULT };
  state.sceneGrouping = data.scene_grouping || { mode: "folder", gap_minutes: 30 };
  prunePersonFilter();
}

async function postDecision(rel_path, decision) {
  const res = await fetch("/api/decide", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path, decision }),
  });
  if (!res.ok) throw new Error(`decide failed: ${res.status}`);
  return res.json();
}

// ---------- rendering: sidebar ----------
function renderPeopleChips() {
  const wrap = $("#people-chips");
  wrap.innerHTML = "";
  if (!state.people.length) {
    const empty = document.createElement("div");
    empty.id = "people-empty";
    // Saying "click group" to a project that is not looking for faces would send
    // someone round a loop that cannot produce anything.
    empty.textContent = state.subjects.detect_faces === false
      ? "Not looking for faces in this project. Use rescore to change that."
      : "No people yet — press group.";
    wrap.appendChild(empty);
    return;
  }
  const visible = state.people.filter((p) => !p.excluded);
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.id = "people-empty";
    empty.textContent = "All clusters excluded — open the People settings to restore.";
    wrap.appendChild(empty);
    return;
  }
  // sort by priority asc for display
  const sorted = visible.slice().sort((a, b) => a.priority - b.priority);
  for (const person of sorted) {
    const chip = document.createElement("button");
    chip.className = "person-chip" + (state.personFilter.has(person.id) ? " active" : "");
    chip.title = `Toggle filter: only photos containing ${person.label}`;
    chip.innerHTML = `
      <img src="/face/${enc(person.ref.rel_path)}?idx=${person.ref.face_idx}" alt="" />
      <span class="pri">#${person.priority}</span>
      <span class="lbl">${escapeHtml(person.label)}</span>
      <span class="cnt">${person.count}</span>`;
    chip.addEventListener("click", () => {
      if (state.personFilter.has(person.id)) state.personFilter.delete(person.id);
      else state.personFilter.add(person.id);
      renderPeopleChips();
      state.cursorIdx = 0;
      recomputeFilter();
      renderMain();
    });
    wrap.appendChild(chip);
  }
}

// ---------- subjects (detected objects) ----------
async function loadSubjects() {
  try {
    const res = await fetch("/api/subjects", { cache: "no-store" });
    if (res.ok) state.subjects = await res.json();
  } catch { /* keep whatever we have */ }
  renderSubjectPanel();
}

function renderSubjectPanel() {
  const section = $("#subject-section");
  const s = state.subjects;
  const on = (s.classes || []).length > 0;
  // The panel stays put even with detection off, because its cog is the only
  // way to turn detection on: hiding the section hid the switch along with it,
  // so a project created as "just photos" — the wizard's default — had no way
  // to ever start detecting anything. The boxes button does go, having nothing
  // to draw.
  section.classList.remove("hidden");
  $("#boxes-btn").classList.toggle("hidden", !on);
  if (!on) {
    $("#subject-classes").innerHTML =
      `<div class="subject-empty">Off. To detect cars, pets or anything else, ` +
      `use <b>rescore</b>.</div>`;
    $("#subject-groups").innerHTML = "";
    return;
  }

  const classWrap = $("#subject-classes");
  const found = (s.classes || []).filter((c) => (s.counts || {})[c]);
  classWrap.innerHTML = found.length
    ? found.map((c) =>
        `<button class="subject-chip${state.subjectClassFilter.has(c) ? " active" : ""}" ` +
        `data-cls="${escapeHtml(c)}" title="Only photos containing a ${escapeHtml(c)}">` +
        `${escapeHtml(c)}<span class="cnt">${s.counts[c]}</span></button>`).join("")
    : `<div class="subject-empty">None found in this project.</div>`;
  classWrap.querySelectorAll("[data-cls]").forEach((b) =>
    b.addEventListener("click", () => toggleSubjectFilter(state.subjectClassFilter, b.dataset.cls)));

  const groupWrap = $("#subject-groups");
  const groups = (s.vehicles || []).filter((v) => !v.excluded);
  groupWrap.innerHTML = groups.length
    ? groups.map((v) =>
        `<button class="subject-group${state.subjectGroupFilter.has(v.id) ? " active" : ""}" ` +
        `data-group="${v.id}" title="Only photos containing this subject">` +
        `<img loading="lazy" src="/subject/${enc(v.ref.rel_path)}?idx=${v.ref.obj_idx}" alt="" />` +
        `<span class="lbl">${escapeHtml(v.label)}</span>` +
        `<span class="cnt">${v.count}</span></button>`).join("")
    : `<div class="subject-empty">No groups yet — press group.</div>`;
  groupWrap.querySelectorAll("[data-group]").forEach((b) =>
    b.addEventListener("click", () => toggleSubjectFilter(state.subjectGroupFilter, b.dataset.group)));
}

function toggleSubjectFilter(set, key) {
  if (set.has(key)) set.delete(key); else set.add(key);
  renderSubjectPanel();
  state.cursorIdx = 0;
  recomputeFilter();
  renderMain();
}

function toggleBoxes() {
  state.showBoxes = !state.showBoxes;
  $("#boxes-btn").classList.toggle("active", state.showBoxes);
  renderGrid();
  syncModalBoxes();
}

function togglePeak() {
  state.showPeak = !state.showPeak;
  $("#peak-btn").classList.toggle("active", state.showPeak);
  $("#peak-level").classList.toggle("hidden", !state.showPeak);
  renderGrid();
  syncModalPeak();
}

// How strict the "is this actually in focus?" test is. Lenses and subjects
// differ enough that one threshold cannot serve every shoot.
function setPeakLevel(level) {
  state.peakLevel = level;
  $$("#peak-level [data-peak-level]").forEach((b) =>
    b.classList.toggle("active", b.dataset.peakLevel === level));
  if (state.showPeak) { renderGrid(); syncModalPeak(); }
}

// ---------- viewer loupe ----------
// Culling is mostly "is this one actually sharp?", and the viewer shows the
// photo fitted — i.e. downscaled past the point where that is answerable. The
// loupe samples the loaded image at its own resolution, so you get real pixels
// without leaving the frame or waiting for a round trip.
const LOUPE_SIZE = 300;              // css px of the magnified panel
const LOUPE_SCALES = [1, 2, 4];      // device pixels shown per source pixel
const loupe = { on: true, scale: 1, at: null };

function loupeInit() {
  try { loupe.on = localStorage.getItem("pcls.loupe") !== "0"; } catch { /* default */ }
  const wrap = $("#modal-image-wrap");
  wrap.addEventListener("pointermove", (e) => {
    loupe.at = { x: e.clientX, y: e.clientY };
    drawLoupe();
  });
  wrap.addEventListener("pointerleave", () => { loupe.at = null; drawLoupe(); });
  // Magnification lives in the toolbar, not on the panel: the panel dodges the
  // cursor by design, so it could never be clicked.
  $("#loupe-scale").addEventListener("click", (e) => {
    const b = e.target.closest("[data-loupe-scale]");
    if (b) setLoupeScale(parseInt(b.dataset.loupeScale, 10));
  });
  // Wheeling over the photo steps through the magnifications too.
  wrap.addEventListener("wheel", (e) => {
    if (!loupe.on || !state.modal.open) return;
    e.preventDefault();
    const i = LOUPE_SCALES.indexOf(loupe.scale);
    setLoupeScale(LOUPE_SCALES[Math.min(LOUPE_SCALES.length - 1,
                                        Math.max(0, i + (e.deltaY < 0 ? 1 : -1)))]);
  }, { passive: false });
  $("#modal-loupe").addEventListener("click", toggleLoupe);
  syncLoupeButton();
}

function syncLoupeButton() {
  $("#modal-loupe").classList.toggle("active", loupe.on);
  $("#loupe-scale").classList.toggle("hidden", !loupe.on);
  $$("#loupe-scale [data-loupe-scale]").forEach((b) =>
    b.classList.toggle("active", parseInt(b.dataset.loupeScale, 10) === loupe.scale));
}

function setLoupeScale(scale) {
  loupe.scale = scale;
  syncLoupeButton();
  drawLoupe();
}

function toggleLoupe() {
  loupe.on = !loupe.on;
  try { localStorage.setItem("pcls.loupe", loupe.on ? "1" : "0"); } catch { /* private */ }
  syncLoupeButton();
  drawLoupe();
}

function hideLoupe() {
  $("#loupe-box").classList.add("hidden");
  $("#loupe-panel").classList.add("hidden");
}

// Where the panel goes, and how big. A fitted photo usually leaves a letterbox
// band; using it — sized to fit — keeps the loupe entirely off the picture,
// which is the whole point of parking it instead of following the cursor.
// Only when there is no usable band does it overlay a corner, and then it
// dodges to the opposite side rather than sit under the pointer.
const LOUPE_MIN = 150;
const LOUPE_LABEL_H = 20;

function loupeGeometry(wrapRect, imgRect, cursor) {
  const pad = 10;
  const fits = (gap) => gap - pad * 2 - LOUPE_LABEL_H >= LOUPE_MIN;
  const bottomGap = wrapRect.bottom - imgRect.bottom;
  const rightGap = wrapRect.right - imgRect.right;

  if (fits(bottomGap)) {
    const size = Math.min(LOUPE_SIZE, bottomGap - pad * 2 - LOUPE_LABEL_H);
    return { size, left: wrapRect.width - size - pad,
             top: wrapRect.height - size - LOUPE_LABEL_H - pad };
  }
  if (fits(rightGap)) {
    const size = Math.min(LOUPE_SIZE, rightGap - pad * 2);
    return { size, left: wrapRect.width - size - pad,
             top: wrapRect.height - size - LOUPE_LABEL_H - pad };
  }
  const size = LOUPE_SIZE;
  let left = wrapRect.width - size - pad;
  const top = wrapRect.height - size - LOUPE_LABEL_H - pad;
  const cx = cursor.x - wrapRect.left, cy = cursor.y - wrapRect.top;
  if (cx > left - 60 && cy > top - 60) left = pad;   // get out from under it
  return { size, left, top };
}

function drawLoupe() {
  const img = $("#modal-image");
  if (!loupe.on || !state.modal.open || !loupe.at || !img.naturalWidth) {
    hideLoupe();
    return;
  }
  const imgRect = img.getBoundingClientRect();
  const { x, y } = loupe.at;
  if (x < imgRect.left || x > imgRect.right || y < imgRect.top || y > imgRect.bottom) {
    hideLoupe();
    return;
  }
  const wrapRect = $("#modal-image-wrap").getBoundingClientRect();
  const geom = loupeGeometry(wrapRect, imgRect, loupe.at);
  const canvas = $("#loupe-canvas");
  const dpr = window.devicePixelRatio || 1;
  const dev = Math.round(geom.size * dpr);
  if (canvas.width !== dev) {
    canvas.width = canvas.height = dev;
    canvas.style.width = canvas.style.height = `${geom.size}px`;
  }
  // How many source pixels fill the panel at the current magnification.
  const sample = dev / loupe.scale;
  const nx = (x - imgRect.left) / imgRect.width * img.naturalWidth;
  const ny = (y - imgRect.top) / imgRect.height * img.naturalHeight;
  const sx = Math.max(0, Math.min(img.naturalWidth - sample, nx - sample / 2));
  const sy = Math.max(0, Math.min(img.naturalHeight - sample, ny - sample / 2));

  const ctx = canvas.getContext("2d");
  // Nearest-neighbour when magnifying: smoothing would hide the very softness
  // the loupe exists to reveal.
  ctx.imageSmoothingEnabled = loupe.scale < 1;
  ctx.clearRect(0, 0, dev, dev);
  ctx.drawImage(img, sx, sy, sample, sample, 0, 0, dev, dev);

  const panel = $("#loupe-panel");
  panel.style.left = `${Math.max(0, geom.left)}px`;
  panel.style.top = `${Math.max(0, geom.top)}px`;
  $("#loupe-label").textContent = `${loupe.scale}:1 · ${Math.round(sample)}px wide`;
  panel.classList.remove("hidden");

  // The sample box on the photo, so it is obvious what is being magnified.
  const boxW = sample / img.naturalWidth * imgRect.width;
  const boxH = sample / img.naturalHeight * imgRect.height;
  const box = $("#loupe-box");
  box.style.left = `${sx / img.naturalWidth * imgRect.width + imgRect.left - wrapRect.left}px`;
  box.style.top = `${sy / img.naturalHeight * imgRect.height + imgRect.top - wrapRect.top}px`;
  box.style.width = `${boxW}px`;
  box.style.height = `${boxH}px`;
  box.classList.remove("hidden");
}

// The viewer's peaking layer tracks the image element the same way the boxes do.
function syncModalPeak() {
  const layer = $("#modal-peak"), img = $("#modal-image");
  if (!layer) return;
  const photo = state.modal.open ? state.filteredPhotos[state.modal.idx] : null;
  const on = photo && state.showPeak && !state.modal.compare;
  layer.classList.toggle("hidden", !on);
  if (!on) { layer.removeAttribute("src"); return; }
  const url = peakUrl(photo);
  if (layer.getAttribute("src") !== url) layer.setAttribute("src", url);
  layer.style.left = `${img.offsetLeft}px`;
  layer.style.top = `${img.offsetTop}px`;
  layer.style.width = `${img.offsetWidth}px`;
  layer.style.height = `${img.offsetHeight}px`;
}

// ---------- subject settings ----------
const subjectEdit = { classes: new Set(), search: "" };

// The re-score dialog owns *what to detect*, because changing it means
// re-scoring anyway. It used to live in a settings panel that was itself hidden
// on projects which had never detected anything — so a car shoot created as
// "just photos" had no way in at all.
function openRescoreModal() {
  subjectEdit.classes = new Set(state.subjects.classes || []);
  subjectEdit.search = "";
  $("#subject-search").value = "";
  $("#rescore-status").textContent = "";
  $("#subject-model-note").textContent = state.subjects.model_ready
    ? "Detection model ready."
    : "First run downloads a ~20 MB detection model (YOLOX-tiny, Apache-2.0).";
  $("#rescore-go").textContent = `Re-score ${state.photos.length} photos`;
  $("#rescore-faces").checked = state.subjects.detect_faces !== false;
  renderSubjectPresets();
  renderSubjectClassPicker();
  $("#rescore-modal").classList.remove("hidden");
}

function closeRescoreModal() { $("#rescore-modal").classList.add("hidden"); }

// Groups are curation of a finished result, so this never re-scores: doing so
// would discard the groups and undo the renaming along with them.
function openSubjectModal() {
  $("#subject-status").textContent = "";
  renderSubjectGroupEditor();
  $("#subject-modal").classList.remove("hidden");
}

function closeSubjectModal() { $("#subject-modal").classList.add("hidden"); }

function renderSubjectPresets() {
  const wrap = $("#subject-presets");
  const presets = state.subjects.presets || {};
  wrap.innerHTML = Object.entries(presets).map(([name, classes]) => {
    const active = classes.every((c) => subjectEdit.classes.has(c));
    return `<button class="subject-preset${active ? " active" : ""}" data-preset="${name}">` +
      `${escapeHtml(name)}<span class="sub">${classes.join(", ")}</span></button>`;
  }).join("") + `<button class="subject-preset" data-preset="__none__">off<span class="sub">no subject detection</span></button>`;
  wrap.querySelectorAll("[data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.preset === "__none__") subjectEdit.classes = new Set();
      else {
        const classes = presets[b.dataset.preset] || [];
        const allOn = classes.every((c) => subjectEdit.classes.has(c));
        for (const c of classes) {
          if (allOn) subjectEdit.classes.delete(c); else subjectEdit.classes.add(c);
        }
      }
      renderSubjectPresets();
      renderSubjectClassPicker();
    }));
}

function renderSubjectClassPicker() {
  const wrap = $("#subject-class-picker");
  const q = subjectEdit.search.trim().toLowerCase();
  // Selected classes stay pinned at the top so a search never hides them.
  const all = state.subjects.available || [];
  const shown = all.filter((c) => subjectEdit.classes.has(c) || (q && c.includes(q)));
  const list = q ? shown : [...subjectEdit.classes, ...all.filter((c) => !subjectEdit.classes.has(c)).slice(0, 12)];
  wrap.innerHTML = list.map((c) =>
    `<label class="subject-class${subjectEdit.classes.has(c) ? " on" : ""}">` +
    `<input type="checkbox" data-class="${escapeHtml(c)}"${subjectEdit.classes.has(c) ? " checked" : ""} />` +
    `${escapeHtml(c)}</label>`).join("")
    || `<div class="subject-empty">No class matches "${escapeHtml(q)}".</div>`;
  wrap.querySelectorAll("[data-class]").forEach((cb) =>
    cb.addEventListener("change", () => {
      if (cb.checked) subjectEdit.classes.add(cb.dataset.class);
      else subjectEdit.classes.delete(cb.dataset.class);
      renderSubjectPresets();
      renderSubjectClassPicker();
    }));
}

function renderSubjectGroupEditor() {
  const groups = state.subjects.vehicles || [];
  $("#subject-group-editor").classList.toggle("hidden", !groups.length);
  $("#subject-group-rows").innerHTML = groups.map((v) =>
    `<div class="subject-group-row" data-id="${v.id}">` +
    `<img src="/subject/${enc(v.ref.rel_path)}?idx=${v.ref.obj_idx}" alt="" />` +
    `<input type="text" class="subject-group-label" value="${escapeHtml(v.label)}" />` +
    `<span class="cnt">${v.count} photo${v.count === 1 ? "" : "s"}</span>` +
    `<label class="subject-group-hide"><input type="checkbox"${v.excluded ? " checked" : ""} /> hide</label>` +
    `</div>`).join("");
}

async function saveSubjectGroups() {
  const rows = [...$$("#subject-group-rows .subject-group-row")];
  if (!rows.length) return true;
  const groups = rows.map((r) => ({
    id: r.dataset.id,
    label: r.querySelector(".subject-group-label").value.trim() || "Subject",
    excluded: r.querySelector(".subject-group-hide input").checked,
  }));
  const res = await fetch("/api/subjects/groups", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ groups }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#subject-status").textContent = "group save failed: " + (err.detail || res.status);
    return false;
  }
  state.subjects.vehicles = (await res.json()).vehicles || [];
  return true;
}

async function applySubjectSettings() {
  if (!await saveSubjectGroups()) return;
  renderSubjectPanel();
  renderGrid();
  closeSubjectModal();
}

// Always re-scores, even when the classes were left alone: this dialog *is* the
// re-score button, so pressing go has to do what the button says.
async function runRescore() {
  const next = [...subjectEdit.classes];
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      with_faces: false,
      subject_classes: next,
      detect_faces: $("#rescore-faces").checked,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#rescore-status").textContent = "re-score failed: " + (err.detail || res.status);
    return;
  }
  closeRescoreModal();
  startScoreProgress("rescore");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollScoreStatus();
}

function renderSidebar() {
  const list = $("#scene-list");
  list.innerHTML = "";
  let totalPick = 0, totalReview = 0, totalReject = 0, totalUndecided = 0;
  for (const scene of state.sceneOrder) {
    const photos = state.byScene.get(scene);
    const counts = { pick: 0, review: 0, reject: 0, undecided: 0 };
    for (const p of photos) counts[p.decision || "undecided"]++;
    totalPick += counts.pick; totalReview += counts.review;
    totalReject += counts.reject; totalUndecided += counts.undecided;

    const li = document.createElement("li");
    li.className = "scene-item" + (state.selectedScene === scene ? " active" : "");
    const n = photos.length;
    const left = counts.undecided;
    const picks = counts.pick ? ` · ${plural(counts.pick, "pick")}` : "";
    li.innerHTML = `
      <span class="name">${escapeHtml(scene)}</span>
      <span class="stats">${left ? `${left} of ${n} left` : `All ${n} decided`}${picks}</span>
      <span class="bar">
        <span class="b-pick" style="width:${100*counts.pick/n}%"></span>
        <span class="b-review" style="width:${100*counts.review/n}%"></span>
        <span class="b-reject" style="width:${100*counts.reject/n}%"></span>
      </span>`;
    li.addEventListener("click", () => selectScene(scene));
    list.appendChild(li);
  }
  const total = state.photos.length;
  const decided = total - totalUndecided;
  $("#overall-progress").textContent =
    `${decided} of ${plural(total, "photo")} decided (${total ? Math.round(100 * decided / total) : 0}%)`
    + (totalPick ? ` · ${plural(totalPick, "pick")}` : "");
}

// ---------- rendering: main ----------
function selectScene(scene) {
  state.selectedScene = scene;
  state.cursorIdx = 0;
  state.selection.clear();
  lastSelIdx = null;
  recomputeFilter();
  renderSidebar();
  $("#grid").scrollTop = 0;
  showCursor();
}

function recomputeFilter() {
  if (!state.selectedScene) { state.filteredPhotos = []; return; }
  const photos = state.byScene.get(state.selectedScene) || [];
  let arr = photos.filter(matchesFilter);
  if (state.peopleById.size > 0) {
    arr = arr.slice().sort((a, b) => {
      const pa = bestPriority(a), pb = bestPriority(b);
      if (pa !== pb) return pa - pb;
      // stable secondary: by rel_path
      return a.rel_path < b.rel_path ? -1 : a.rel_path > b.rel_path ? 1 : 0;
    });
  }
  state.filteredPhotos = arr;
  if (state.cursorIdx >= state.filteredPhotos.length) {
    state.cursorIdx = Math.max(0, state.filteredPhotos.length - 1);
  }
}

function renderMain() {
  applyLayoutCSS();
  renderHeader();
  renderGrid();
  renderNextPreview();
  renderSelectionBar();
  // Hooked here rather than onto each control, so paging with the keyboard and
  // any future way of moving around are remembered without being wired up
  // individually. Debounced, so a burst of arrow keys still writes once.
  scheduleViewSave();
  syncTopbar();   // Edit needs a photo to open
  renderInspector();
}

function applyLayoutCSS() {
  const L = LAYOUTS[state.pageSize];
  const grid = $("#grid");
  grid.style.setProperty("--cols", L.cols);
  grid.style.setProperty("--rows", L.rows);
  grid.dataset.per = state.pageSize;   // for styles that depend on tile size
}

function renderHeader() {
  const total = state.filteredPhotos.length;
  const scenePhotos = state.selectedScene
    ? (state.byScene.get(state.selectedScene) || []) : [];
  const sceneTotal = scenePhotos.length;
  const sceneUndecided = scenePhotos.reduce((n, p) => n + (p.decision == null ? 1 : 0), 0);
  $("#scene-title").textContent = state.selectedScene || "Select a scene";
  // Progress through the scene first; how much of it the filter shows only
  // when it hides some.
  $("#scene-stats").textContent = !state.selectedScene ? ""
    : `${sceneTotal - sceneUndecided} of ${plural(sceneTotal, "photo")} decided`
      + (total < sceneTotal ? ` · ${total} shown` : "");
  const g = gridGeom(), top = topRow();
  $("#page-indicator").textContent = total
    ? `${top * g.cols + 1}–${Math.min(total, (top + g.rows) * g.cols)} of ${total}`
    : "—";
  // How many each filter would show in this scene, so an empty one is not a
  // click away from finding out.
  const n = { all: sceneTotal, undecided: sceneUndecided, pick: 0, review: 0, reject: 0, edited: 0 };
  for (const p of scenePhotos) {
    if (p.decision) n[p.decision] += 1;
    if (p.edit) n.edited += 1;
  }
  $$("#filter-row .filter").forEach((b) => {
    b.querySelector(".n").textContent = state.selectedScene ? n[b.dataset.filter] : "";
  });
  $("#prev-page").disabled = top === 0 || total === 0;
  $("#next-page").disabled = top >= g.maxTop || total === 0;
  // These three carry both an icon and a changing count. Writing to textContent
  // would delete the injected svg, so the label lives in its own span.
  const rejectBtn = $("#reject-undecided-btn");
  rejectBtn.disabled = sceneUndecided === 0;
  setBtnLabel(rejectBtn, sceneUndecided > 0
    ? `Reject ${sceneUndecided} undecided`
    : "Reject undecided");
  const totalPicks = state.photos.reduce((n, p) => n + (p.decision === "pick" ? 1 : 0), 0);
  const exportBtn = $("#export-picks-btn");
  exportBtn.disabled = totalPicks === 0;
  setBtnLabel(exportBtn, totalPicks > 0
    ? `Export ${totalPicks} pick${totalPicks > 1 ? "s" : ""}`
    : "Export picks");
  setBtnLabel($("#hdr-btn"), state.brackets.length ? `HDR brackets (${state.brackets.length})` : "HDR brackets");
}

// Faces shown on a tile before the rest are counted as "+n".
const TILE_FACES = 4;
const AUTO_LABEL = { pick: "Pick", review: "Review", reject: "Reject" };

const FILTER_EMPTY = {
  undecided: "Every photo in this scene is decided.",
  pick: "No picks in this scene yet.",
  review: "Nothing marked for review in this scene.",
  reject: "Nothing rejected in this scene.",
  edited: "No edited photos in this scene.",
};

// What a tile shows. A tile whose signature is unchanged is kept as it is, so
// moving the cursor or scrolling does not rebuild it or reload its thumbnail.
function tileSig(p, i) {
  const faces = (p.faces || []).map((f) => {
    const person = f.person_id ? state.peopleById.get(f.person_id) : null;
    return person && person.excluded ? "x" : (f.person_id || "-");
  }).join(",");
  return [i, p.rel_path, p.decision, p.rating, p.label, p.edited_at, p.auto_suggestion, p.scores?.badness,
    state.selection.has(p.rel_path), state.showBoxes, state.showPeak, state.peakLevel, faces].join("|");
}

// The rows on screen and one either side are built, the rest of the list is
// rows of empty grid, so a scene of thousands of photos costs what a
// screenful does.
function renderGrid() {
  const grid = $("#grid");
  // A filter that matches nothing used to leave a blank grid, which reads as
  // the app having lost the photos.
  if (!state.filteredPhotos.length) {
    if (!state.selectedScene) { grid.innerHTML = ""; return; }
    const narrowed = state.filter !== "all" || state.personFilter.size
      || state.subjectClassFilter.size || state.subjectGroupFilter.size
      || state.minRating || state.labelFilter;
    grid.innerHTML = `<div id="grid-inner" class="empty"><div class="grid-empty">
      <p>${narrowed ? escapeHtml(FILTER_EMPTY[state.filter] || "No photos match these filters.") : "This scene has no photos."}</p>
      ${narrowed ? `<button type="button" class="primary" id="grid-show-all">Show all photos in the scene</button>` : ""}
    </div></div>`;
    $("#grid-show-all")?.addEventListener("click", () => {
      state.personFilter.clear(); state.subjectClassFilter.clear(); state.subjectGroupFilter.clear();
      state.minRating = 0; state.labelFilter = "";
      $("#rating-filter").value = "0"; $("#label-filter").value = "";
      $$(".filter").forEach((x) => x.classList.toggle("active", x.dataset.filter === "all"));
      state.filter = "all";
      state.cursorIdx = 0;
      recomputeFilter();
      renderSidebar();
      renderPeopleChips();
      renderSubjectPanel();
      renderMain();
    });
    return;
  }
  let inner = $("#grid-inner");
  if (!inner || inner.classList.contains("empty")) {
    grid.innerHTML = `<div id="grid-inner"></div>`;
    inner = $("#grid-inner");
  }
  const list = state.filteredPhotos;
  const g = gridGeom();
  inner.style.gridTemplateColumns = `repeat(${g.cols}, minmax(0, 1fr))`;
  inner.style.gridTemplateRows = `repeat(${Math.max(1, g.totalRows)}, ${g.rowH}px)`;
  const r0 = Math.max(0, Math.floor(grid.scrollTop / g.step) - 1);
  const r1 = Math.min(g.totalRows, Math.ceil((grid.scrollTop + grid.clientHeight) / g.step) + 1);
  const from = r0 * g.cols, to = Math.min(list.length, r1 * g.cols);
  const kept = new Map();
  for (const el of [...inner.children]) {
    const i = Number(el.dataset.idx);
    if (i >= from && i < to && el.dataset.sig === tileSig(list[i], i)) kept.set(i, el);
    else el.remove();
  }
  for (let i = from; i < to; i++) {
    let el = kept.get(i);
    if (!el) {
      el = makeTile(list[i], i);
      el.dataset.sig = tileSig(list[i], i);
      inner.appendChild(el);
    }
    el.style.gridRow = String(Math.floor(i / g.cols) + 1);
    el.style.gridColumn = String((i % g.cols) + 1);
    el.classList.toggle("focused", i === state.cursorIdx);
  }
}

function makeTile(p, absIdx) {
  const tile = document.createElement("div");
  const orientation = (p.width && p.height && p.height > p.width) ? "portrait" : "landscape";
  tile.className = "tile " + orientation
    + (p.decision ? " decision-" + p.decision : "")
    + (absIdx === state.cursorIdx ? " focused" : "")
    + (state.selection.has(p.rel_path) ? " selected" : "");
  const auto = p.auto_suggestion || "";
  const badness = p.scores?.badness != null ? p.scores.badness.toFixed(2) : "—";
  const fname = basename(p.rel_path);
  const visibleFaces = (p.faces || [])
    .map((f, fi) => ({ f, fi, person: f.person_id ? state.peopleById.get(f.person_id) : null }))
    .filter(({ person }) => !(person && person.excluded))
    .sort((a, b) => {
      const pa = a.person ? a.person.priority : Infinity;
      const pb = b.person ? b.person.priority : Infinity;
      return pa - pb;
    });
  const faceCount = visibleFaces.length;
  // Faces sit along the bottom of the photo rather than in a strip of their
  // own, which made tiles with faces shorter than their neighbours and
  // threw every row out of line.
  const shownFaces = visibleFaces.slice(0, TILE_FACES);
  const facesHtml = faceCount
    ? `<span class="tile-faces">${
        shownFaces.map(({ fi }) =>
          `<span class="face-thumb"><img loading="lazy" src="/face/${enc(p.rel_path)}?idx=${fi}" alt="" /></span>`
        ).join("")
      }${faceCount > TILE_FACES ? `<span class="face-more">+${faceCount - TILE_FACES}</span>` : ""}</span>`
    : "";
  // The frame is the photo's own rectangle inside the letterboxed cell (the
  // same aspect-ratio trick the box layer uses), so what is drawn on the photo
  // stays on it whatever its shape.
  const ar = p.geom ? `${p.geom.w}/${p.geom.h}` : (p.width && p.height ? `${p.width}/${p.height}` : "3/2");
  const suggestion = auto ? AUTO_LABEL[auto] || auto : "";
  tile.innerHTML = `
    <div class="tile-content">
      <div class="tile-img" data-action="open">
        <img loading="lazy" src="${thumbUrl(p)}" alt="" />
        ${peakLayerHtml(p)}
        ${boxLayerHtml(p)}
        <span class="tile-frame" style="aspect-ratio:${ar}">
          ${auto ? `<span class="auto-badge ${auto}" title="The app suggests: ${suggestion.toLowerCase()}">${suggestion}</span>` : ""}
          ${p.type === "hdr" ? `<span class="hdr-tile-badge">HDR · ${(p.members || []).length}</span>` : ""}
          ${facesHtml}
        </span>
        <input type="checkbox" class="tile-select"${state.selection.has(p.rel_path) ? " checked" : ""} title="Select (X)" />
        <button class="tile-edit-btn${p.edit ? " edited" : ""}" data-action="edit" title="Edit (E)" aria-label="Edit">${icon("pencil")}</button>
      </div>
    </div>
    <div class="tile-meta">
      ${p.label ? `<span class="label-dot ${p.label}" title="${LABEL_NAMES[p.label]} label"></span>` : ""}
      <span class="tile-name">${escapeHtml(fname)}</span>
      ${p.rating ? starsHtml(p.rating, "stars tile-stars") : ""}
      <span class="badness" title="How bad it looks, from sharpness, exposure and closed eyes: lower is better">badness ${badness}</span>
    </div>
    <div class="tile-action-bar">
      <button class="btn-decision${p.decision === "reject" ? " active" : ""}" data-decision="reject">Reject <span class="kbd">R</span></button>
      <button class="btn-decision${p.decision === "review" ? " active" : ""}" data-decision="review">Review <span class="kbd">V</span></button>
      <button class="btn-decision${p.decision === "pick" ? " active" : ""}" data-decision="pick">Pick <span class="kbd">P</span></button>
    </div>`;
  tile.querySelector(".tile-img").addEventListener("click", () => openModal(absIdx));
  tile.querySelector(".tile-edit-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    openEditModal(absIdx);
  });
  tile.querySelector(".tile-select").addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSelect(absIdx, e.shiftKey);
  });
  tile.querySelectorAll(".btn-decision").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const wantDecision = btn.dataset.decision;
      const newDecision = p.decision === wantDecision ? null : wantDecision;
      decideAt(absIdx, newDecision);
    });
  });
  tile.dataset.idx = absIdx;
  return tile;
}

// Hovering a tile focuses it, but only when the pointer really moves. A page
// turn re-renders the tiles under a resting pointer and the browser reports it
// entering whichever tile landed there, which pulled the cursor off the photo
// the arrow key had just moved to (the 5th photo became the 6th). Closing the
// viewer or compare over a resting pointer does the same, so the position is
// the last one seen anywhere, not only over the grid.
let pointerBefore = null, pointerNow = null;

function bindGridHover() {
  const grid = $("#grid");
  document.addEventListener("pointermove", (e) => {
    pointerBefore = pointerNow;
    pointerNow = { x: e.clientX, y: e.clientY };
  }, true);
  grid.addEventListener("pointermove", (e) => {
    if (pointerBefore && pointerBefore.x === e.clientX && pointerBefore.y === e.clientY) return;
    const tile = e.target.closest(".tile");
    if (tile && tile.dataset.idx !== undefined) focusAt(Number(tile.dataset.idx), false);
  });
  // Scrolling builds the rows coming into view, once a frame.
  let frame = 0;
  grid.addEventListener("scroll", () => {
    if (frame) return;
    frame = requestAnimationFrame(() => { frame = 0; renderGrid(); renderHeader(); });
  });
  // A resized window resizes the rows; the cursor stays in view.
  new ResizeObserver(() => {
    if (!state.filteredPhotos.length) return;
    renderGrid();
    ensureVisible(state.cursorIdx);
    renderGrid();
    renderHeader();
  }).observe(grid);
}

function renderNextPreview() {
  const preview = $("#next-preview");
  if (state.pageSize !== 1 || state.filteredPhotos.length === 0) {
    preview.classList.add("hidden");
    return;
  }
  const next = state.filteredPhotos[state.cursorIdx + 1];
  if (!next) {
    preview.classList.add("hidden");
    return;
  }
  preview.classList.remove("hidden");
  $("#next-preview-img").src = thumbUrl(next);
  $("#next-preview-name").textContent = basename(next.rel_path);
}

// ---------- focus & paging ----------
// `scroll` is false for the mouse: hovering a tile half off the screen should
// not scroll the grid out from under it.
function focusAt(absIdx, scroll = true) {
  if (absIdx < 0 || absIdx >= state.filteredPhotos.length) return;
  state.cursorIdx = absIdx;
  if (scroll) ensureVisible(absIdx);
  renderMain();
}

// ---------- remembered view ----------
// A project reopens on the filter, layout and photo it was left on, in the
// viewer if that is where it was. Saved on the server per project: it is where
// you are in the work, so it should follow the project rather than the browser
// profile that happened to open it. The photo is kept as well as the page: with
// the Undecided filter every decision takes a photo out of the list, so the
// same page number soon points at different photos.
let viewRestored = false;
let viewSaveTimer = null;

function scheduleViewSave() {
  // Nothing is written until the saved view has been read and applied — the
  // renders that happen on the way there would otherwise file the defaults
  // over the very thing they are about to restore.
  if (!viewRestored) return;
  if (viewSaveTimer) clearTimeout(viewSaveTimer);
  viewSaveTimer = setTimeout(saveView, 500);
}

function viewPayload() {
  return {
    filter: state.filter,
    page_size: state.pageSize,
    page: pageIdx(),
    scene: state.selectedScene,
    photo: state.filteredPhotos[state.cursorIdx]?.rel_path ?? null,
    viewer: state.modal.open,
  };
}

async function saveView() {
  viewSaveTimer = null;
  const res = await fetch("/api/view", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(viewPayload()),
  });
  if (!res.ok) throw new Error(`view save failed: ${res.status}`);
}

// The save waits for a pause in the arrow keys, and leaving the project, or
// quitting, straight after the last one used to drop it.
async function flushViewSave() {
  if (!viewSaveTimer) return;
  clearTimeout(viewSaveTimer);
  await saveView();
}

// Closing the tab or the browser gives no time for a fetch to finish; a beacon
// is sent anyway.
window.addEventListener("pagehide", () => {
  if (!viewSaveTimer) return;
  clearTimeout(viewSaveTimer);
  viewSaveTimer = null;
  navigator.sendBeacon("/api/view", new Blob([JSON.stringify(viewPayload())], { type: "application/json" }));
});

// Applies what can be applied before the scene list is built, and hands the
// rest back for the caller to use once it is.
async function restoreView() {
  const res = await fetch("/api/view");
  if (!res.ok) throw new Error(`view load failed: ${res.status}`);
  const v = await res.json();
  if (v.filter) state.filter = v.filter;
  if (v.page_size) state.pageSize = v.page_size;
  return v;
}

function syncViewControls() {
  $$(".filter").forEach((b) =>
    b.classList.toggle("active", b.dataset.filter === state.filter));
  $$("#cols-toggle .cols").forEach((b) =>
    b.classList.toggle("active", parseInt(b.dataset.cols, 10) === state.pageSize));
}

// Land on a page directly, rather than stepping to it as gotoPage does.
function gotoPageIndex(page) {
  renderGrid();   // the rows must exist before they can be scrolled to
  const g = gridGeom();
  const top = Math.min(g.maxTop, Math.max(0, page * g.rows));
  $("#grid").scrollTop = top * g.step;
  state.cursorIdx = Math.min(state.filteredPhotos.length - 1, top * g.cols);
  renderMain();
}

// A screen on or back: the cursor lands on the first photo of the next
// screen, or the last of the previous one.
function gotoPage(delta) {
  const g = gridGeom(), now = topRow();
  const top = Math.min(g.maxTop, Math.max(0, now + delta * g.rows));
  if (top === now) return;
  $("#grid").scrollTop = top * g.step;
  const n = state.filteredPhotos.length;
  state.cursorIdx = delta > 0 ? Math.min(n - 1, top * g.cols) : Math.min(n - 1, (top + g.rows) * g.cols - 1);
  renderMain();
}

function tryAutoAdvance() {
  // Every photo on screen decided by mouse: on to the next screen.
  const visible = visiblePhotos();
  if (!visible.length || !visible.every((p) => p.decision != null)) return;
  if (topRow() < gridGeom().maxTop) gotoPage(+1);
}

// ---------- decisions ----------
// A decision shows at once and is saved behind, in the order made. Waiting
// for each save before moving on let a key pressed during it act on where the
// cursor had been: P, R, V typed quickly in the viewer left one decision, the
// last, on the first photo, with the cursor three photos on.
let decisionSaves = Promise.resolve();

function saveDecision(photo, decision) {
  decisionSaves = decisionSaves.then(async () => {
    const saved = await postDecision(photo.rel_path, decision);
    // A later decision on the same photo may already be showing; only the
    // time stamp of this one is the server's to set.
    if (photo.decision === saved.decision) photo.decided_at = saved.decided_at;
  }).catch((err) => {
    // The screen and the project no longer agree about this photo: say so,
    // and reload the project's own record rather than carry on over it.
    alert(`Saving a decision failed (${err.message}). The project will reload.`);
    location.reload();
  });
  return decisionSaves;
}

// `advance` is for the keyboard: the key acts on the highlighted photo and
// the cursor moves on to the next. A click on a tile's own button decides that
// tile and leaves the cursor alone.
function applyDecision(photo, decision) {
  photo.decision = decision;
  photo.decided_at = decision ? new Date().toISOString() : null;
  saveDecision(photo, decision);
}

function decideAt(absIdx, decision, { advance = false } = {}) {
  const photo = state.filteredPhotos[absIdx];
  if (!photo) return;
  applyDecision(photo, decision);
  recomputeFilter();
  renderSidebar();
  const n = state.filteredPhotos.length;
  const at = state.filteredPhotos.indexOf(photo);
  if (at === -1) {
    // It left the filter (Undecided, or Picks after a reject), so the photo
    // after it has moved up into its place: stay put, and that one is next.
    // Stepping on as well skipped a photo on every decision.
    state.cursorIdx = Math.max(0, Math.min(absIdx, n - 1));
  } else if (advance && decision !== null) {
    state.cursorIdx = Math.min(at + 1, n - 1);
  }
  showCursor();
  if (state.modal.open) {
    if (!n) { closeModal(); return; }
    state.modal.idx = state.cursorIdx;
    state.modal.fit = true;
    state.modal.compare = false;
    renderModal();
    scheduleViewSave();
  } else if (!advance) {
    tryAutoAdvance();
  }
}

// ---------- export picks ----------
// Tokens a file-name template can use, with what each turns into. The server
// checks the template too; this list is only what the chips offer.
const EXPORT_TOKENS = [
  ["name", "original name"], ["seq", "001, 002…"], ["date", "capture date"],
  ["time", "capture time"], ["scene", "scene"], ["project", "project"],
];
const EXPORT_EXT = { jpeg: ".jpg", tiff: ".tif" };
const exportUi = { pollTimer: null, nameTimer: null };

async function openExportModal() {
  const res = await fetch("/api/export/picks/preview");
  if (!res.ok) {
    alert("Could not load export info: " + res.status);
    return;
  }
  const info = await res.json();
  $("#export-count").textContent = `${info.count} photo${info.count === 1 ? "" : "s"}`;
  $("#export-target").value = info.default_target;
  setOptionCardValue("#export-mode-cards", "folder");
  applyExportSettings(info.settings);
  $("#export-status").textContent = "";
  $("#export-progress").classList.add("hidden");
  $("#export-confirm").disabled = info.count === 0;
  setBtnLabel($("#export-cancel"), "Cancel");
  $("#export-modal").classList.remove("hidden");
  // An export started earlier may still be running, if the dialog was closed
  // on it; show that rather than a fresh form.
  const st = await (await fetch("/api/export/status")).json();
  if (st.running) followExport();
}

function closeExportModal() {
  $("#export-modal").classList.add("hidden");
}

function applyExportSettings(st) {
  $$("#export-format button").forEach((b) =>
    b.classList.toggle("active", b.dataset.format === st.format));
  $("#export-quality").value = st.quality;
  const size = st.long_edge == null ? "" : String(st.long_edge);
  const preset = [...$("#export-size").options].some((o) => o.value === size);
  $("#export-size").value = preset ? size : "custom";
  $("#export-size-custom").value = preset ? "" : size;
  $("#export-name").value = st.name_template;
  $("#export-metadata").value = st.metadata;
  syncExportForm();
  previewExportName();
}

function exportFormat() {
  return $("#export-format button.active")?.dataset.format || "jpeg";
}

function syncExportForm() {
  const fmt = exportFormat();
  $("#export-quality-row").classList.toggle("hidden", fmt !== "jpeg");
  $("#export-quality-val").textContent = $("#export-quality").value;
  $("#export-format-note").textContent = fmt === "tiff"
    ? "Lossless and uncompressed, for printing and further editing: about 70 MB for 24 megapixels."
    : "";
  $("#export-size-custom").classList.toggle("hidden", $("#export-size").value !== "custom");
}

// The settings as the server wants them, or a string saying what is wrong.
function readExportSettings() {
  let longEdge = null;
  const size = $("#export-size").value;
  if (size === "custom") {
    longEdge = parseInt($("#export-size-custom").value, 10);
    if (!(longEdge >= 64 && longEdge <= 20000)) return "Enter a long edge between 64 and 20000 px.";
  } else if (size) {
    longEdge = parseInt(size, 10);
  }
  return {
    format: exportFormat(),
    quality: parseInt($("#export-quality").value, 10),
    long_edge: longEdge,
    metadata: $("#export-metadata").value,
    name_template: $("#export-name").value,
  };
}

// Show what the template names the first pick, after a pause in typing.
function previewExportName() {
  clearTimeout(exportUi.nameTimer);
  exportUi.nameTimer = setTimeout(async () => {
    const out = $("#export-name-example");
    const res = await fetch("/api/export/name-preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template: $("#export-name").value }),
    });
    const d = await res.json();
    out.classList.toggle("error", !res.ok);
    if (!res.ok) { out.textContent = d.detail || "Invalid file name."; return; }
    out.textContent = d.example ? `e.g. ${d.example}${EXPORT_EXT[exportFormat()]}` : "";
  }, 250);
}

function insertExportToken(token) {
  const input = $("#export-name");
  const at = input.selectionStart ?? input.value.length;
  const end = input.selectionEnd ?? at;
  const text = `{${token}}`;
  input.value = input.value.slice(0, at) + text + input.value.slice(end);
  input.focus();
  input.setSelectionRange(at + text.length, at + text.length);
  previewExportName();
}

function bindExportForm() {
  $("#export-tokens").innerHTML = EXPORT_TOKENS.map(([t, what]) =>
    `<button type="button" data-token="${t}" title="${escapeHtml(what)}">{${t}}</button>`).join("");
  $$("#export-tokens button").forEach((b) =>
    b.addEventListener("click", () => insertExportToken(b.dataset.token)));
  $$("#export-format button").forEach((b) => b.addEventListener("click", () => {
    $$("#export-format button").forEach((o) => o.classList.toggle("active", o === b));
    syncExportForm();
    previewExportName();
  }));
  $("#export-quality").addEventListener("input", syncExportForm);
  $("#export-size").addEventListener("change", () => {
    syncExportForm();
    if ($("#export-size").value === "custom") $("#export-size-custom").focus();
  });
  $("#export-name").addEventListener("input", previewExportName);
}

async function confirmExport() {
  const target = $("#export-target").value.trim();
  if (!target) { $("#export-status").textContent = "Choose a target folder."; return; }
  const settings = readExportSettings();
  if (typeof settings === "string") { $("#export-status").textContent = settings; return; }
  const mode = getOptionCardValue("#export-mode-cards") || "folder";
  $("#export-confirm").disabled = true;
  const res = await fetch("/api/export/picks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_dir: target, mode, settings }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#export-status").textContent = "Failed: " + (err.detail || res.status);
    $("#export-confirm").disabled = false;
    return;
  }
  followExport();
}

// The Cancel button stops a running export, and closes the dialog otherwise.
async function exportCancelOrClose() {
  const st = await (await fetch("/api/export/status")).json();
  if (!st.running) { closeExportModal(); return; }
  setBtnLabel($("#export-cancel"), "Stopping…");
  await fetch("/api/export/cancel", { method: "POST" });
}

// Poll a running export into the progress bar. Closing the dialog does not
// stop it; the toast says when it is done.
function followExport() {
  $("#export-progress").classList.remove("hidden");
  $("#export-status").textContent = "";
  $("#export-confirm").disabled = true;
  setBtnLabel($("#export-cancel"), "Stop");
  clearInterval(exportUi.pollTimer);
  const tick = async () => {
    const st = await (await fetch("/api/export/status")).json();
    const pct = st.total ? (100 * st.idx) / st.total : 0;
    $("#export-bar-fill").style.width = pct.toFixed(1) + "%";
    $("#export-progress-text").textContent =
      `${st.idx} of ${st.total}` + (st.current ? ` · ${basename(st.current)}` : "");
    if (st.running) return;
    clearInterval(exportUi.pollTimer);
    exportUi.pollTimer = null;
    finishExport(st);
  };
  tick();
  exportUi.pollTimer = setInterval(tick, 400);
}

function finishExport(st) {
  $("#export-progress").classList.add("hidden");
  $("#export-confirm").disabled = false;
  setBtnLabel($("#export-cancel"), "Close");
  const status = $("#export-status");
  if (st.error) {
    status.textContent = "Export failed: " + st.error;
    return;
  }
  const r = st.result;
  const verb = r.cancelled ? "Stopped after" : "Exported";
  let html = `${icon("check")} ${verb} <b>${r.copied}</b> photo${r.copied === 1 ? "" : "s"} to<br>`
    + `<code>${escapeHtml(r.target_dir)}</code>`;
  if (r.skipped) html += `<br>Skipped ${r.skipped} (missing source).`;
  if (r.per_combo && Object.keys(r.per_combo).length) {
    const sorted = Object.entries(r.per_combo).sort((a, b) => b[1] - a[1]);
    html += `<br><span class="combo-summary">${sorted.map(([k, v]) => `${escapeHtml(k)}: ${v}`).join(" · ")}</span>`;
  }
  status.innerHTML = html;
  if ($("#export-modal").classList.contains("hidden")) {
    showDownloadToast(`${verb} ${r.copied} photo${r.copied === 1 ? "" : "s"}`, r.target_dir);
  }
}

// ---------- bulk actions ----------
async function rejectUndecidedInScene() {
  if (!state.selectedScene) return;
  const photos = state.byScene.get(state.selectedScene) || [];
  const targets = photos.filter((p) => p.decision == null);
  if (!targets.length) return;
  const ok = confirm(
    `Reject all ${targets.length} undecided photos in "${state.selectedScene}"?\n` +
    `Use the Undo toast (or ${MOD_KEY}Z) to revert.`
  );
  if (!ok) return;
  const sceneLabel = state.selectedScene;
  const relPaths = targets.map((p) => p.rel_path);
  const res = await fetch("/api/decide/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_paths: relPaths, decision: "reject" }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Bulk reject failed: " + (err.detail || res.status));
    return;
  }
  const now = new Date().toISOString();
  for (const p of targets) {
    p.decision = "reject";
    p.decided_at = now;
  }
  recomputeFilter();
  renderSidebar();
  renderMain();
  showUndoToast(
    `Rejected ${relPaths.length} undecided photos in "${sceneLabel}".`,
    relPaths,
  );
}

// ---------- undo toast ----------
const undoToast = { paths: null, timer: null };

function showUndoToast(msg, relPaths) {
  if (undoToast.timer) clearTimeout(undoToast.timer);
  undoToast.paths = relPaths;
  $("#undo-toast-msg").textContent = msg;
  $("#undo-toast").classList.remove("hidden");
  undoToast.timer = setTimeout(hideUndoToast, 12000);
}

function hideUndoToast() {
  $("#undo-toast").classList.add("hidden");
  undoToast.paths = null;
  if (undoToast.timer) {
    clearTimeout(undoToast.timer);
    undoToast.timer = null;
  }
}

async function performUndo() {
  if (!undoToast.paths || !undoToast.paths.length) return;
  const paths = undoToast.paths;
  hideUndoToast();
  const res = await fetch("/api/decide/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_paths: paths, decision: null }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Undo failed: " + (err.detail || res.status));
    return;
  }
  for (const rp of paths) {
    const photo = state.photos.find((p) => p.rel_path === rp);
    if (photo) {
      photo.decision = null;
      photo.decided_at = null;
    }
  }
  recomputeFilter();
  renderSidebar();
  renderMain();
}

// ---------- HDR brackets modal ----------
const hdrEdit = { groups: [], selected: new Set(), sceneOf: new Map(), origKey: "" };

function hdrGroupsKey(groups) {
  return groups.map((g) => g.slice().sort().join("|")).sort().join("¶");
}

function openHdrModal() {
  // Working copy of the current grouping; edits apply on "Apply".
  hdrEdit.groups = state.brackets.map((b) => b.members.slice());
  hdrEdit.selected = new Set();
  hdrEdit.origKey = hdrGroupsKey(hdrEdit.groups);
  // Scene of every source frame: standalone photos + bracket members.
  hdrEdit.sceneOf = new Map();
  for (const p of state.photos) {
    if (p.type !== "hdr") hdrEdit.sceneOf.set(p.rel_path, p.scene);
  }
  for (const b of state.brackets) {
    for (const m of b.members) hdrEdit.sceneOf.set(m, b.scene);
  }
  $("#hdr-modal").classList.remove("hidden");
  renderHdrModal();
}

function closeHdrModal() {
  $("#hdr-modal").classList.add("hidden");
}

function updateHdrToolbar() {
  const sel = hdrEdit.selected;
  $("#hdr-selcount").textContent = `${sel.size} selected`;
  $("#hdr-group").disabled = sel.size < 3 || sel.size > 9;
  const anyGrouped = [...sel].some(
    (m) => hdrEdit.groups.some((g) => g.includes(m)));
  $("#hdr-ungroup").disabled = !anyGrouped;
  const changed = hdrGroupsKey(hdrEdit.groups) !== hdrEdit.origKey;
  $("#hdr-apply").disabled = !changed;
  $("#hdr-status").textContent = changed ? "Unsaved changes — Apply to re-score" : "";
}

function hdrToggleSelect(relPath) {
  if (hdrEdit.selected.has(relPath)) hdrEdit.selected.delete(relPath);
  else hdrEdit.selected.add(relPath);
  // Light update — re-rendering would collapse expanded scene sections.
  const on = hdrEdit.selected.has(relPath);
  $$(`.hdr-thumb`).forEach((d) => {
    if (d.dataset.rel === relPath) d.classList.toggle("selected", on);
  });
  updateHdrToolbar();
}

function hdrGroupSelected() {
  const picked = [...hdrEdit.selected];
  if (picked.length < 3 || picked.length > 9) return;
  // Pull the picked frames out of any existing group, drop groups now < 3.
  hdrEdit.groups = hdrEdit.groups
    .map((g) => g.filter((m) => !hdrEdit.selected.has(m)))
    .filter((g) => g.length >= 3);
  hdrEdit.groups.push(picked);
  hdrEdit.selected.clear();
  renderHdrModal();
}

function hdrUngroupSelected() {
  if (!hdrEdit.selected.size) return;
  hdrEdit.groups = hdrEdit.groups
    .map((g) => g.filter((m) => !hdrEdit.selected.has(m)))
    .filter((g) => g.length >= 3);
  hdrEdit.selected.clear();
  renderHdrModal();
}

function hdrDissolve(idx) {
  hdrEdit.groups.splice(idx, 1);
  renderHdrModal();
}

function buildHdrThumb(relPath) {
  const div = document.createElement("div");
  div.className = "hdr-thumb" + (hdrEdit.selected.has(relPath) ? " selected" : "");
  div.dataset.rel = relPath;
  div.innerHTML =
    `<img loading="lazy" src="/thumb/${enc(relPath)}" alt="" />` +
    `<span class="hdr-thumb-name">${escapeHtml(basename(relPath))}</span>`;
  div.addEventListener("click", () => hdrToggleSelect(relPath));
  return div;
}

function buildBracketCard(members, idx) {
  const card = document.createElement("div");
  card.className = "bracket-card";

  const head = document.createElement("div");
  head.className = "bracket-head";
  head.innerHTML = `<span class="bracket-title">Bracket · ${members.length} frames</span>`;
  const dissolve = document.createElement("button");
  dissolve.type = "button";
  dissolve.className = "bracket-dissolve";
  dissolve.textContent = "Dissolve";
  dissolve.addEventListener("click", () => hdrDissolve(idx));
  head.appendChild(dissolve);
  card.appendChild(head);

  const row = document.createElement("div");
  row.className = "hdr-thumbs";
  // A group identical to a saved bracket already has a merged result to show.
  const key = members.slice().sort().join("|");
  const saved = state.brackets.find(
    (b) => b.members.slice().sort().join("|") === key);
  const result = document.createElement("div");
  if (saved) {
    result.className = "hdr-thumb merged";
    result.innerHTML =
      `<img loading="lazy" src="/thumb/${enc(saved.merged)}" alt="" />` +
      `<span class="hdr-thumb-name">merged result</span>`;
  } else {
    result.className = "hdr-thumb pending";
    result.innerHTML = `<span>merges<br>on Apply</span>`;
  }
  row.appendChild(result);
  members.forEach((m) => row.appendChild(buildHdrThumb(m)));
  card.appendChild(row);
  return card;
}

function renderHdrModal() {
  const content = $("#hdr-content");
  content.innerHTML = "";

  const grouped = new Set();
  hdrEdit.groups.forEach((g) => g.forEach((m) => grouped.add(m)));

  if (hdrEdit.groups.length) {
    const sec = document.createElement("div");
    sec.className = "hdr-section";
    sec.innerHTML = `<h3>Brackets · ${hdrEdit.groups.length}</h3>`;
    hdrEdit.groups.forEach((g, idx) => sec.appendChild(buildBracketCard(g, idx)));
    content.appendChild(sec);
  }

  const ungrouped = [...hdrEdit.sceneOf.keys()].filter((m) => !grouped.has(m));
  const byScene = new Map();
  for (const m of ungrouped.sort()) {
    const s = hdrEdit.sceneOf.get(m) || "(none)";
    if (!byScene.has(s)) byScene.set(s, []);
    byScene.get(s).push(m);
  }
  const sec = document.createElement("div");
  sec.className = "hdr-section";
  sec.innerHTML = `<h3>Ungrouped photos · ${ungrouped.length}</h3>`;
  if (!ungrouped.length) {
    const empty = document.createElement("div");
    empty.className = "hdr-empty";
    empty.textContent = "Every photo is in a bracket.";
    sec.appendChild(empty);
  }
  for (const [scene, members] of byScene) {
    const det = document.createElement("details");
    det.className = "hdr-scene";
    const sum = document.createElement("summary");
    sum.textContent = `${scene} · ${members.length}`;
    det.appendChild(sum);
    const body = document.createElement("div");
    body.className = "hdr-thumbs";
    det.appendChild(body);
    // Populate lazily on first expand so a big library stays responsive.
    det.addEventListener("toggle", () => {
      if (det.open && !body.childElementCount) {
        members.forEach((m) => body.appendChild(buildHdrThumb(m)));
      }
    });
    sec.appendChild(det);
  }
  content.appendChild(sec);
  updateHdrToolbar();
}

async function applyHdr() {
  const frames = hdrEdit.groups.reduce((n, g) => n + g.length, 0);
  if (!confirm(
      "Apply HDR changes? This re-scores every photo.\n" +
      `${hdrEdit.groups.length} bracket(s) from ${frames} frames.`)) return;
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ with_faces: false, bracket_groups: hdrEdit.groups }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#hdr-status").textContent = "Failed: " + (err.detail || res.status);
    return;
  }
  closeHdrModal();
  $("#score-title").textContent = "Merging & re-scoring…";
  $("#score-progress").classList.remove("hidden");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollScoreStatus();
}

// ---------- HDR look tuner ----------
const lookEdit = { look: {}, bracketIdx: 0, objUrl: null, timer: null };

function openLookModal() {
  if (!state.brackets.length) {
    alert("No HDR brackets yet — run a re-score first so there is something to tune.");
    return;
  }
  lookEdit.look = { ...LOOK_DEFAULT, ...(state.hdrLook || {}) };
  lookEdit.bracketIdx = 0;
  $$("#hdr-look-modal input[type=range]").forEach((sl) => {
    sl.value = lookEdit.look[sl.dataset.look];
  });
  syncLookValues();
  $("#hdr-look-status").textContent = "";
  $("#hdr-look-modal").classList.remove("hidden");
  fetchLookPreview(true);
}

function closeLookModal() {
  $("#hdr-look-modal").classList.add("hidden");
  if (lookEdit.objUrl) { URL.revokeObjectURL(lookEdit.objUrl); lookEdit.objUrl = null; }
  if (lookEdit.timer) { clearTimeout(lookEdit.timer); lookEdit.timer = null; }
}

function syncLookValues() {
  $$("#hdr-look-modal .look-val").forEach((el) => {
    el.textContent = Number(lookEdit.look[el.dataset.val]).toFixed(2);
  });
}

function lookBracketNav(delta) {
  const n = state.brackets.length;
  if (!n) return;
  lookEdit.bracketIdx = (lookEdit.bracketIdx + delta + n) % n;
  fetchLookPreview(true);
}

function fetchLookPreview(immediate) {
  if (lookEdit.timer) { clearTimeout(lookEdit.timer); lookEdit.timer = null; }
  const go = async () => {
    const b = state.brackets[lookEdit.bracketIdx];
    if (!b) return;
    $("#hdr-look-which").textContent =
      `bracket ${lookEdit.bracketIdx + 1} / ${state.brackets.length}`;
    $("#hdr-look-status").textContent = "rendering…";
    try {
      const res = await fetch("/api/hdr/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ members: b.members, look: lookEdit.look }),
      });
      if (!res.ok) { $("#hdr-look-status").textContent = "preview failed"; return; }
      const blob = await res.blob();
      if (lookEdit.objUrl) URL.revokeObjectURL(lookEdit.objUrl);
      lookEdit.objUrl = URL.createObjectURL(blob);
      $("#hdr-look-img").src = lookEdit.objUrl;
      $("#hdr-look-status").textContent = "";
    } catch {
      $("#hdr-look-status").textContent = "preview error";
    }
  };
  if (immediate) go();
  else lookEdit.timer = setTimeout(go, 130);
}

function resetLook() {
  lookEdit.look = { ...LOOK_DEFAULT };
  $$("#hdr-look-modal input[type=range]").forEach((sl) => {
    sl.value = lookEdit.look[sl.dataset.look];
  });
  syncLookValues();
  fetchLookPreview(true);
}

async function applyLook() {
  if (!confirm("Apply this look to every HDR photo? This re-merges and re-scores.")) return;
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ with_faces: false, hdr_look: lookEdit.look }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#hdr-look-status").textContent = "Failed: " + (err.detail || res.status);
    return;
  }
  closeLookModal();
  closeHdrModal();
  $("#score-title").textContent = "Applying look & re-scoring…";
  $("#score-progress").classList.remove("hidden");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollScoreStatus();
}

// ---------- editor ----------
const editSession = {
  idx: 0, relPath: null, edit: null, baseline: null,
  objUrl: null, origUrl: null, timer: null, comparing: false, built: false,
  // Local adjustments: which mask the sliders drive (-1 = global), which create
  // tool is armed, the brush settings, and the in-flight pointer drag.
  activeMask: -1, tool: null, showMask: false, drag: null, hover: null,
  brush: { size: 60, erase: false },
  // Zoom: 0 = fit the whole frame (rendered from the cached preview), anything
  // else = that many CSS pixels per original image pixel, rendered from the
  // full-resolution original through the server's ROI path.
  view: { zoom: 0, cx: 0.5, cy: 0.5 },
  natural: { w: 0, h: 0 },   // the photo's real pixel size, from the db
  // The frame *after* straightening and cropping — what every overlay and the
  // 1:1 view measure against. Starts at the photo's own size (correct whenever
  // the geometry is neutral) and is corrected by each render's X-Frame-Size.
  frame: { w: 0, h: 0 },
  // The straightened frame *before* cropping. A crop box's coordinates are
  // fractions of this, so it stays the reference whether the tool is armed (when
  // it is also what is on screen) or not (when `frame` is the cropped result).
  cropFrame: { w: 0, h: 0 },
  cropAspect: "free",
  // This photo's stashes, as the server stores them: EDIT_SLOTS entries, each
  // either null or {edit, saved_at, name}. `slotEdits` is the same list in
  // working form, and `slotMark` is the one the working edit matches right now.
  slots: [], slotEdits: [], slotMark: -1, slotTouch: -1,
  // Original frame -> displayed frame, as six numbers from the server. Identity
  // until a crop or a straighten makes the two differ.
  maskXform: null,
};

// ---------- tone curve ----------
const CURVE_MAX = 6;         // max control points (keeps it approachable)
const CURVE_PAD = 8;
const curveState = { cssW: 240, cssH: 240, drag: -1, accent: "#4a90e2",
                     channel: "curve" };

function curveChannel() {
  return CURVE_CHANNELS.find((c) => c.k === curveState.channel) || CURVE_CHANNELS[0];
}
function curvePoints() { return editSession.edit[curveState.channel]; }
function curveTint(c) { return c.tint || curveState.accent; }
function curveTouched(key) {
  const pts = editSession.edit[key];
  return !!pts && pts.some(([x, y]) => Math.abs(y - x) > 1e-4);
}
function renderCurveChannels() {
  const wrap = $("#curve-channels");
  if (!wrap || !editSession.edit) return;
  const touched = CURVE_CHANNELS.filter((c) => c.k !== "curve" && curveTouched(c.k)).length;
  $("#curve-summary-state").textContent = touched ? `\u00b7 ${touched} channel${touched > 1 ? "s" : ""}` : "";
  wrap.innerHTML = CURVE_CHANNELS.map((c) =>
    `<button type="button" class="curve-chan${c.k === curveState.channel ? " active" : ""}` +
    `${curveTouched(c.k) ? " touched" : ""}" data-curve-chan="${c.k}" ` +
    `style="--chan: ${curveTint(c)}">${c.label}</button>`).join("");
  $$("#curve-channels .curve-chan").forEach((b) => {
    b.addEventListener("click", () => {
      curveState.channel = b.dataset.curveChan;
      curveState.drag = -1;
      renderCurveChannels();
      drawCurve();
    });
  });
}

function curveInit() {
  const cv = $("#edit-curve");
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth || 240, h = cv.clientHeight || 240;
  curveState.cssW = w; curveState.cssH = h;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  curveState.accent =
    getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#4a90e2";
  cv.addEventListener("pointerdown", curveDown);
  cv.addEventListener("pointermove", curveMove);
  cv.addEventListener("dblclick", curveDoubleClick);
  window.addEventListener("pointerup", curveUp);
  drawCurve();
}

// Monotone-cubic (Fritsch-Carlson) sampler — mirrors editing._pchip so the drawn
// curve matches what the server applies. The server LUT stays the source of truth.
function pchipEval(pts, xs) {
  const n = pts.length;
  const px = pts.map((p) => p[0]), py = pts.map((p) => p[1]);
  if (n === 1) return xs.map(() => py[0]);
  const h = [], d = [];
  for (let i = 0; i < n - 1; i++) { h[i] = px[i + 1] - px[i]; d[i] = (py[i + 1] - py[i]) / h[i]; }
  const m = new Array(n);
  m[0] = d[0]; m[n - 1] = d[n - 2];
  for (let i = 1; i < n - 1; i++) {
    if (d[i - 1] * d[i] <= 0) m[i] = 0;
    else { const w1 = 2 * h[i] + h[i - 1], w2 = h[i] + 2 * h[i - 1]; m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i]); }
  }
  return xs.map((x) => {
    let i = 0; while (i < n - 2 && x > px[i + 1]) i++;
    const t = (x - px[i]) / h[i], t2 = t * t, t3 = t2 * t;
    const a = 2 * t3 - 3 * t2 + 1, b = t3 - 2 * t2 + t, c = -2 * t3 + 3 * t2, e = t3 - t2;
    return Math.min(1, Math.max(0, a * py[i] + b * h[i] * m[i] + c * py[i + 1] + e * h[i] * m[i + 1]));
  });
}

function curveToPx(x, y) {
  const pw = curveState.cssW - 2 * CURVE_PAD, ph = curveState.cssH - 2 * CURVE_PAD;
  return [CURVE_PAD + x * pw, CURVE_PAD + (1 - y) * ph];
}
function curveEventData(e) {
  const cv = $("#edit-curve"), rect = cv.getBoundingClientRect();
  const W = curveState.cssW, H = curveState.cssH, pw = W - 2 * CURVE_PAD, ph = H - 2 * CURVE_PAD;
  const px = (e.clientX - rect.left) / rect.width * W, py = (e.clientY - rect.top) / rect.height * H;
  const x = Math.min(1, Math.max(0, (px - CURVE_PAD) / pw));
  const y = Math.min(1, Math.max(0, 1 - (py - CURVE_PAD) / ph));
  return { px, py, x, y };
}
function curveHit(px, py) {
  const pts = curvePoints();
  for (let i = 0; i < pts.length; i++) {
    const [hx, hy] = curveToPx(pts[i][0], pts[i][1]);
    if (Math.hypot(px - hx, py - hy) <= 10) return i;
  }
  return -1;
}

function drawCurve() {
  const cv = $("#edit-curve");
  if (!cv || !editSession.edit) return;
  const ctx = cv.getContext("2d");
  const W = curveState.cssW, H = curveState.cssH, pw = W - 2 * CURVE_PAD, ph = H - 2 * CURVE_PAD;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const gx = CURVE_PAD + pw * i / 4, gy = CURVE_PAD + ph * i / 4;
    ctx.beginPath(); ctx.moveTo(gx, CURVE_PAD); ctx.lineTo(gx, CURVE_PAD + ph); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CURVE_PAD, gy); ctx.lineTo(CURVE_PAD + pw, gy); ctx.stroke();
  }
  ctx.strokeStyle = "rgba(255,255,255,0.16)";
  ctx.beginPath(); ctx.moveTo(...curveToPx(0, 0)); ctx.lineTo(...curveToPx(1, 1)); ctx.stroke();

  const N = 64, xs = [];
  for (let i = 0; i <= N; i++) xs.push(i / N);
  const trace = (pts, colour, width, alpha) => {
    const ys = pchipEval(pts, xs);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.beginPath();
    xs.forEach((x, i) => { const [qx, qy] = curveToPx(x, ys[i]); i ? ctx.lineTo(qx, qy) : ctx.moveTo(qx, qy); });
    ctx.stroke();
    ctx.restore();
  };
  // The inactive channels stay on screen, thin and faint. Four curves that each
  // hide the other three would make a split tone impossible to reason about.
  for (const c of CURVE_CHANNELS) {
    if (c.k === curveState.channel || !curveTouched(c.k)) continue;
    trace(editSession.edit[c.k], curveTint(c), 1, 0.4);
  }
  const active = curveChannel(), colour = curveTint(active), pts = curvePoints();
  trace(pts, colour, 2, 1);
  ctx.fillStyle = colour;
  for (const [x, y] of pts) { const [qx, qy] = curveToPx(x, y); ctx.beginPath(); ctx.arc(qx, qy, 4, 0, 7); ctx.fill(); }
}

function curveDown(e) {
  if (!editSession.edit) return;
  const { px, py, x, y } = curveEventData(e);
  let i = curveHit(px, py);
  if (i < 0) {
    const pts = curvePoints();
    if (pts.length >= CURVE_MAX || x <= 0 || x >= 1) return;  // only interior points added
    const np = [x, y];
    pts.push(np); pts.sort((a, b) => a[0] - b[0]);
    i = pts.indexOf(np);
  }
  curveState.drag = i;
  $("#edit-curve").setPointerCapture(e.pointerId);
  drawCurve(); setEditDirty(); fetchEditPreview(false);
}
function curveMove(e) {
  if (curveState.drag < 0) return;
  const { x, y } = curveEventData(e);
  const pts = curvePoints(), i = curveState.drag;
  if (i === 0) pts[i] = [0, y];                     // endpoints: x locked, y free
  else if (i === pts.length - 1) pts[i] = [1, y];
  else {
    const lo = pts[i - 1][0] + 0.01, hi = pts[i + 1][0] - 0.01;
    pts[i] = [Math.min(hi, Math.max(lo, x)), y];
  }
  drawCurve(); setEditDirty(); previewDuringDrag();
}
function curveUp() {
  if (curveState.drag < 0) return;
  curveState.drag = -1;
  settleDrag();
}
function curveDoubleClick(e) {
  const { px, py } = curveEventData(e);
  const i = curveHit(px, py), pts = curvePoints();
  if (i > 0 && i < pts.length - 1) {             // can't remove endpoints
    pts.splice(i, 1);
    drawCurve(); setEditDirty(); fetchEditPreview(false);
  }
}
function resetCurve() {
  // The active channel only: wiping a red curve nobody asked about because they
  // clicked reset while looking at RGB would be a surprise.
  editSession.edit[curveState.channel] = CURVE_IDENTITY.map((p) => p.slice());
  renderCurveChannels();
  drawCurve(); setEditDirty(); fetchEditPreview(true);
}

function openEditModal(absIdx) {
  const photo = state.filteredPhotos[absIdx];
  if (!photo) return;
  renderEditControls();
  editSession.idx = absIdx;
  editSession.relPath = photo.rel_path;
  editSession.edit = mergeNeutralEdit(photo.edit);
  editSession.baseline = mergeNeutralEdit(photo.edit);
  editSession.comparing = false;
  if (editSession.objUrl) { URL.revokeObjectURL(editSession.objUrl); editSession.objUrl = null; }
  if (editSession.origUrl) { URL.revokeObjectURL(editSession.origUrl); editSession.origUrl = null; }
  $("#edit-which").textContent =
    `${basename(photo.rel_path)} · ${absIdx + 1}/${state.filteredPhotos.length}`;
  $("#edit-compare").classList.remove("holding");
  $("#edit-status").textContent = "";
  $("#edit-preset-select").value = "";
  // Every photo opens fitted; the zoom is a per-photo inspection, not a mode.
  editSession.view = { zoom: 0, cx: 0.5, cy: 0.5 };
  editSession.natural = { w: photo.width || 0, h: photo.height || 0 };
  editSession.frame = { ...editSession.natural };
  editSession.cropFrame = { ...editSession.natural };
  editSession.cropAspect = "free";
  editSession.maskXform = null;
  editSession.slotTouch = -1;
  setSlots(photo.edit_slots);
  renderSlots();
  $$("#edit-modal .edit-zoom").forEach((b) =>
    b.classList.toggle("active", b.dataset.zoom === "0"));
  $("#edit-zoom-hint").classList.add("hidden");
  layoutPreviewImage();
  renderWatermarkPanel();
  renderFilmPanel();
  renderHslPanel();
  renderGradePanel();
  renderCurveChannels();
  lookState.matching = false;
  repairState.cloneSrc = null;
  portraitState.faces = null;
  portraitState.key = "";
  $("#portrait-faces").textContent = "";
  if ($("#edit-portrait-group").open) setTimeout(loadPortraitFaces, 0);
  renderLookPanel();
  renderRepairPanel();
  renderOpticsPanel();
  renderPortraitPanel();
  loadWatermarkInfo(photo.rel_path);
  selectMask(-1, { silent: true });
  if (editSession.edit.masks.length) $("#edit-mask-group").open = true;
  setEditTool(null);
  syncEditSliders();
  drawCurve();
  resetEditHistory();
  setEditDirty();
  $("#edit-modal").classList.remove("hidden");
  resizeOverlay();
  fetchEditPreview(true);
  fetchOriginalPreview();
  renderFilmstrip($("#edit-filmstrip"), absIdx, (i) => editGoTo(i));
  // After layout, so the controls it points at have their places.
  if (!tourSeen("editor")) setTimeout(() => maybeStartTour("editor"), 400);
}

function closeEditModal() {
  $("#edit-modal").classList.add("hidden");
  if (editSession.timer) { clearTimeout(editSession.timer); editSession.timer = null; }
  if (editSession.objUrl) { URL.revokeObjectURL(editSession.objUrl); editSession.objUrl = null; }
  if (editSession.origUrl) { URL.revokeObjectURL(editSession.origUrl); editSession.origUrl = null; }
  editSession.relPath = null;
  editSession.comparing = false;
  editSession.drag = null;
  setEditTool(null);
}

function renderEditControls() {
  if (editSession.built) return;
  for (const [g, def] of Object.entries(EDIT_SCHEMA)) {
    const body = $(`#edit-modal .edit-group-body[data-group="${g}"]`);
    if (!body) continue;
    body.innerHTML = def.fields.map((f) =>
      `<label class="look-row" data-field="${f.k}"` +
      `${f.hint ? ` title="${escapeHtml(f.hint)}"` : ""}>` +
      `<span class="look-name">${f.label}</span>` +
      `<input type="range" data-edit="${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" />` +
      `<span class="look-val" data-val="${f.k}"></span></label>`
    ).join("");
  }
  editSession.built = true;
}

// The slider panel drives either the global edit or the selected mask's own
// sliders; everything else about it (layout, ranges, formatting) is identical.
function adjTarget() {
  const i = editSession.activeMask;
  return i >= 0 && editSession.edit.masks[i] ? editSession.edit.masks[i].adj : editSession.edit;
}

function bindEditControls() {
  const controls = $(".edit-controls");
  controls.addEventListener("input", (e) => {
    const sl = e.target.closest('input[type=range][data-edit]');
    if (!sl) return;
    const k = sl.dataset.edit;
    const target = adjTarget();
    target[k] = parseFloat(sl.value);
    const f = fieldByKey(k);
    $(`#edit-modal .look-val[data-val="${k}"]`).textContent = Number(target[k]).toFixed(f.fmt);
    if (editSession.activeMask >= 0) renderMaskList();
    setEditDirty();
    previewDuringDrag();     // live drafts while the slider moves
  });
}

function syncEditSliders() {
  const target = adjTarget();
  $$("#edit-modal input[type=range][data-edit]").forEach((sl) => {
    sl.value = target[sl.dataset.edit] || 0;
  });
  syncEditValues();
  // Geometry is global, so it is not part of the per-mask slider set — but it
  // still has to come back looking right when a photo opens with one saved.
  syncCropControls();
}
// A slider fills from its zero: from the middle for one that goes both ways,
// from the left for one that only goes up, and not at all at zero, so a moved
// slider is told from an untouched one at a glance. The fill was the browser's,
// from the left edge on every slider, which made all of them look set.
function paintSlider(sl) {
  const min = Number(sl.min), max = Number(sl.max), v = Number(sl.value);
  if (!(max > min)) return;
  const zero = min < 0 && max > 0 ? 0 : min;
  const at = (x) => ((x - min) / (max - min)) * 100;
  sl.style.setProperty("--fill-a", `${Math.min(at(zero), at(v))}%`);
  sl.style.setProperty("--fill-b", `${Math.max(at(zero), at(v))}%`);
}

function paintSliders() {
  $$("#edit-modal input[type=range]").forEach(paintSlider);
}

function bindSliderLooks() {
  document.addEventListener("input", (e) => {
    if (e.target.matches?.("#edit-modal input[type=range]")) paintSlider(e.target);
  });
  // Double-click puts an adjustment back to zero, as in Lightroom. Only the
  // main adjustments: they are all neutral at 0 (EDIT_SCHEMA), which is not
  // true of every slider in the editor.
  document.addEventListener("dblclick", (e) => {
    const sl = e.target.closest?.("#edit-modal input[type=range][data-edit]");
    if (!sl || Number(sl.value) === 0) return;
    sl.value = 0;
    sl.dispatchEvent(new Event("input", { bubbles: true }));
    sl.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

function syncEditValues() {
  paintSliders();
  const target = adjTarget();
  $$("#edit-modal .look-val[data-val]").forEach((el) => {
    const f = fieldByKey(el.dataset.val);
    el.textContent = Number(target[el.dataset.val] || 0).toFixed(f ? f.fmt : 0);
  });
}

function setEditDirty() {
  // Every change funnels through here, which makes it the one place the slot
  // marker can be kept true without a render per slider frame: markSlot only
  // touches the DOM when the answer actually changes. It is the one place the
  // undo history can see every change for the same reason.
  recordEditHistory();
  markSlot();
  const dirty = !editsEqual(editSession.edit, editSession.baseline);
  $("#edit-save").disabled = !dirty;
  $("#edit-dirty").textContent = dirty ? "unsaved changes" : "";
}

let previewSeq = 0;

// The body for a preview request: whole frame when fitting, the visible window
// at device resolution when zoomed.
// Draft renders trade resolution for latency while something is being dragged.
const DRAFT_EDGE = 1100;
// A settled render is made at the size it is shown at, in device pixels, up to
// the server's preview edge: grading cost follows the pixel count, and on a
// 1x screen the 2048 px render was two to three times what the canvas shows.
// In steps, so resizing the window does not make a new size on every pixel,
// and never below a draft.
const PREVIEW_EDGE = 2048;   // server.EDIT_PREVIEW_EDGE
const FIT_STEP = 256;

function fitEdge() {
  const wrap = $(".edit-canvas-wrap");
  if (!wrap || !wrap.clientWidth || !wrap.clientHeight) return null;   // not laid out yet
  const need = Math.max(wrap.clientWidth, wrap.clientHeight) * (window.devicePixelRatio || 1);
  const steps = Math.max(Math.ceil(DRAFT_EDGE / FIT_STEP), Math.ceil(need / FIT_STEP));
  return Math.min(PREVIEW_EDGE, steps * FIT_STEP);
}

function previewBody(edit, draft) {
  const body = { rel_path: editSession.relPath, edit };
  if (draft) body.max_edge = DRAFT_EDGE;
  else {
    const edge = fitEdge();
    if (edge && edge < PREVIEW_EDGE) body.fit_edge = edge;
  }
  // While the crop tool is armed the frame is shown whole, so the box has
  // something to be dragged over.
  if (editSession.tool === "crop") body.skip_crop = true;
  if (isZoomed()) {
    const roi = viewRoi();
    const dpr = window.devicePixelRatio || 1;
    body.roi = [roi.x0, roi.y0, roi.rw, roi.rh];
    body.out_w = Math.round(viewportSize().w * dpr);
  }
  return body;
}

let previewInFlight = false;
let previewQueued;

function fetchEditPreview(immediate, draft) {
  if (editSession.timer) { clearTimeout(editSession.timer); editSession.timer = null; }
  const go = async () => {
    if (!editSession.relPath) return;
    // One render at a time. Firing a fresh request every 110 ms at a server
    // that needs 300 ms just piles work onto the same CPU and makes every
    // frame slower; the newest request waits, and anything it superseded is
    // simply dropped.
    if (previewInFlight) { previewQueued = { draft }; return; }
    previewInFlight = true;
    // Dragging a mask handle fires these back to back; only the newest response
    // may reach the <img>, otherwise a slow render lands on top of a fresh one.
    const seq = ++previewSeq;
    const rel = editSession.relPath;
    $("#edit-status").textContent = "rendering…";
    try {
      const res = await fetch("/api/edit/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(previewBody(editSession.edit, draft)),
      });
      if (!res.ok) { $("#edit-status").textContent = "preview failed"; return; }
      const ms = res.headers.get("X-Render-Ms");
      // What the frame is after straightening and cropping. The overlay and the
      // 1:1 view measure against it, and the server is the one that knows —
      // better one copy of that arithmetic than a second one here that can drift.
      const frame = res.headers.get("X-Frame-Size");
      const blob = await res.blob();
      // Newest only, and of this photo: stepping to the next photo mid-render
      // queues its request behind this one, so this one is still the newest
      // when it lands, and would show the last photo in the frame of the next.
      if (seq !== previewSeq || rel !== editSession.relPath) return;
      const fullFrame = res.headers.get("X-Frame-Full");
      if (frame) {
        const [fw, fh] = frame.split("x").map(Number);
        if (fw && fh) {
          const moved = fw !== editSession.frame.w || fh !== editSession.frame.h;
          editSession.frame = { w: fw, h: fh };
          if (moved) layoutPreviewImage();
        }
      }
      if (fullFrame) {
        const [cw, ch] = fullFrame.split("x").map(Number);
        if (cw && ch) editSession.cropFrame = { w: cw, h: ch };
      }
      const xf = res.headers.get("X-Mask-Xform");
      if (xf) {
        const v = xf.split(",").map(Number);
        editSession.maskXform = (v.length === 6 && v.every(Number.isFinite)) ? v : null;
      }
      syncCropControls();
      scheduleOverlay();   // guides move with the frame they are drawn on
      if (editSession.objUrl) URL.revokeObjectURL(editSession.objUrl);
      editSession.objUrl = URL.createObjectURL(blob);
      if (!editSession.comparing) $("#edit-img").src = editSession.objUrl;
      // The render time is for whoever is measuring (the tooltip); on screen
      // it read as a number to worry about. The line is for messages.
      $("#edit-status").title = ms ? `last render ${ms} ms${draft ? " (draft)" : ""}` : "";
      // Its own words go once the render is up: "rendering…", or a failure the
      // next render recovered from. Anything else is a message for the user.
      if (["preview error", "rendering…"].includes($("#edit-status").textContent)) $("#edit-status").textContent = "";
      if (!draft) prefetchNeighbours(rel);
    } catch {
      $("#edit-status").textContent = "preview error";
    } finally {
      previewInFlight = false;
      if (previewQueued) {
        const next = previewQueued;
        previewQueued = null;
        fetchEditPreview(true, next.draft);
      }
    }
  };
  if (immediate) go();
  else editSession.timer = setTimeout(go, 130);
}

// Once a photo is on screen, have the photos either side decoded in the
// background, once per photo, so stepping to one does not start with a decode.
let prefetchedFor = null;
function prefetchNeighbours(rel) {
  if (prefetchedFor === rel) return;
  prefetchedFor = rel;
  const rels = [editSession.idx - 1, editSession.idx + 1]
    .map((i) => state.filteredPhotos[i]).filter(Boolean).map((p) => p.rel_path);
  if (!rels.length) return;
  fetch("/api/edit/prefetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_paths: rels }),
  }).then((res) => { if (!res.ok) throw new Error(`prefetch failed: ${res.status}`); });
}

async function fetchOriginalPreview() {
  // The neutral render kept aside for hold-to-compare — of the same window the
  // editor is showing, so comparing works zoomed in too.
  const rel = editSession.relPath;
  try {
    const res = await fetch("/api/edit/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(previewBody(EDIT_NEUTRAL)),
    });
    if (!res.ok || rel !== editSession.relPath) return;
    const blob = await res.blob();
    if (editSession.origUrl) URL.revokeObjectURL(editSession.origUrl);
    editSession.origUrl = URL.createObjectURL(blob);
  } catch { /* compare just won't be available */ }
}

function editCompareOn() {
  if (!editSession.origUrl || editSession.comparing || editSession.drag) return;
  editSession.comparing = true;
  $("#edit-img").src = editSession.origUrl;
  $("#edit-compare").classList.add("holding");
  $("#edit-status").textContent = "showing original";
  drawOverlay();  // mask guides hide while the original is up
}
function editCompareOff() {
  if (!editSession.comparing) return;
  editSession.comparing = false;
  if (editSession.objUrl) $("#edit-img").src = editSession.objUrl;
  $("#edit-compare").classList.remove("holding");
  $("#edit-status").textContent = "";
  drawOverlay();
}

// Reset is context-aware: it clears the selected mask's sliders, or — with no
// mask selected — the whole edit including every mask.
// ---------- edit history (undo / redo) ----------
// Snapshots of the whole edit, as JSON. A change starts an entry and the entry
// is closed once the edit has been still for HISTORY_SETTLE_MS, so one slider
// drag or one mask drag is one step, not sixty. `committed` is the state the
// last closed entry left, which is what the next change is undone back to.
const HISTORY_MAX = 100;
const HISTORY_SETTLE_MS = 400;
const editHistory = { undo: [], redo: [], committed: null, timer: null, restoring: false };

function resetEditHistory() {
  clearTimeout(editHistory.timer);
  editHistory.timer = null;
  editHistory.undo = [];
  editHistory.redo = [];
  editHistory.committed = JSON.stringify(editSession.edit);
  syncHistoryButtons();
}

function recordEditHistory() {
  if (editHistory.restoring || editHistory.committed == null) return;
  if (!editHistory.timer) {
    if (JSON.stringify(editSession.edit) === editHistory.committed) return;
    editHistory.undo.push(editHistory.committed);
    if (editHistory.undo.length > HISTORY_MAX) editHistory.undo.shift();
    editHistory.redo = [];
  }
  clearTimeout(editHistory.timer);
  editHistory.timer = setTimeout(settleEditHistory, HISTORY_SETTLE_MS);
  syncHistoryButtons();
}

function settleEditHistory() {
  clearTimeout(editHistory.timer);
  editHistory.timer = null;
  editHistory.committed = JSON.stringify(editSession.edit);
}

function editUndo() { stepEditHistory(editHistory.undo, editHistory.redo); }
function editRedo() { stepEditHistory(editHistory.redo, editHistory.undo); }

function stepEditHistory(from, to) {
  if (!editSession.relPath) return;
  // A change still settling is closed first, so undo takes back all of it.
  if (editHistory.timer) settleEditHistory();
  if (!from.length) return;
  to.push(editHistory.committed);
  editHistory.committed = from.pop();
  editHistory.restoring = true;
  editSession.edit = JSON.parse(editHistory.committed);
  refreshEditUi();
  editHistory.restoring = false;
  syncHistoryButtons();
}

function syncHistoryButtons() {
  $("#edit-undo").disabled = !editHistory.undo.length && !editHistory.timer;
  $("#edit-redo").disabled = !editHistory.redo.length;
}

// Put every panel back in step with editSession.edit after it was replaced
// wholesale, by an undo, a reset or a loaded slot.
function refreshEditUi() {
  const masks = editSession.edit.masks || [];
  selectMask(editSession.activeMask < masks.length ? editSession.activeMask : -1,
             { silent: true });
  renderWatermarkPanel();
  renderFilmPanel();
  renderHslPanel();
  renderGradePanel();
  renderCurveChannels();
  renderLookPanel();
  renderRepairPanel();
  renderOpticsPanel();
  renderPortraitPanel();
  syncEditSliders();
  drawCurve();
  drawOverlay();
  setEditDirty();
  fetchEditPreview(true);
}

function resetEdit() {
  const m = activeMask();
  if (m) {
    m.adj = neutralAdj();
    renderMaskList();
  } else {
    editSession.edit = mergeNeutralEdit(null);
    selectMask(-1, { silent: true });
    setEditTool(null);
    // Nothing from the preset is left applied, so the picker shouldn't keep
    // claiming one is active.
    $("#edit-preset-select").value = "";
    renderWatermarkPanel();
    renderFilmPanel();
    renderHslPanel();
    renderGradePanel();
    renderCurveChannels();
    renderLookPanel();
    renderRepairPanel();
    renderOpticsPanel();
    renderPortraitPanel();
  }
  syncEditSliders();
  drawCurve();
  drawOverlay();
  setEditDirty();
  fetchEditPreview(true);
}

// The white-balance picker. The pair it returns is absolute — it replaces
// temp/tint rather than nudging them — and it always lands on the *global*
// sliders, mask or no mask: white balance is a statement about the light the
// photo was taken in, and the light does not stop at a mask edge.
async function pickNeutral(f) {
  if (!editSession.relPath) return;
  setEditTool(null);
  $("#edit-status").textContent = "reading white balance\u2026";
  try {
    const res = await fetch("/api/edit/neutral", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rel_path: editSession.relPath, x: f.x, y: f.y, edit: opticsPayload() }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      $("#edit-status").textContent = err.detail || `white balance failed: ${res.status}`;
      return;
    }
    const { wb } = await res.json();
    editSession.edit.temp = wb.temp;
    editSession.edit.tint = wb.tint;
    // Show the sliders it just wrote: with a mask selected the Color panel is
    // the mask's own, and the move would have been invisible.
    selectMask(-1, { silent: true });
    syncEditSliders();
    setEditDirty();
    drawOverlay();
    fetchEditPreview(true);
    // Say when the answer was pinned. The gains stop at 1.25/0.75, so a deep
    // tungsten frame gets as far as the sliders go and keeps a little cast.
    $("#edit-status").textContent = wb.clamped
      ? "as far as the sliders reach \u2014 finish it on the blue curve"
      : "";
  } catch { $("#edit-status").textContent = "white balance error"; }
}

async function autoEdit() {
  if (!editSession.relPath) return;
  $("#edit-status").textContent = "auto…";
  try {
    const res = await fetch("/api/edit/auto", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rel_path: editSession.relPath }),
    });
    if (!res.ok) { $("#edit-status").textContent = "auto failed"; return; }
    const { edit } = await res.json();
    // Auto-tone is a global suggestion; keep the local masks and the watermark.
    const masks = editSession.edit.masks;
    const watermark = editSession.edit.watermark;
    editSession.edit = mergeNeutralEdit(edit);
    editSession.edit.masks = masks;
    editSession.edit.watermark = watermark;
    selectMask(-1, { silent: true });
    renderWatermarkPanel();
    syncEditSliders();
    drawCurve();
    setEditDirty();
    fetchEditPreview(true);
  } catch { $("#edit-status").textContent = "auto error"; }
}

// The edit dict to persist: {} when neutral so the server drops it. Masks the
// server would discard anyway (an unpainted brush) are stripped here.
function editPayload() {
  if (isNeutralEdit(editSession.edit)) return {};
  return { ...editSession.edit, masks: persistableMasks(editSession.edit.masks) };
}

function editIsDirty() {
  return !!editSession.relPath && !editsEqual(editSession.edit, editSession.baseline);
}

// Writes the edit on screen to the project. True when it was saved; on a
// refusal the reason is shown and the edit stays on screen, unsaved.
async function persistEdit() {
  const rel = editSession.relPath;
  const res = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: rel, edit: editPayload() }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#edit-status").textContent = "save failed: " + (err.detail || res.status);
    return false;
  }
  const updated = await res.json();
  const photo = state.photos.find((p) => p.rel_path === rel);
  if (photo) {
    if (updated.edit) { photo.edit = updated.edit; photo.edited_at = updated.edited_at; }
    else { delete photo.edit; delete photo.edited_at; }
  }
  editSession.baseline = mergeNeutralEdit(editSession.edit);
  setEditDirty();
  return true;
}

async function saveEdit() {
  if (!editSession.relPath) return;
  if (!(await persistEdit())) return;
  closeEditModal();
  renderMain();
}

// Moving to another photo keeps the edit, as Lightroom does: it used to be
// dropped without a word, and ← → are what you press to look at the next one.
async function editNav(delta) {
  if (!state.filteredPhotos.length) return;
  await editGoTo(Math.max(0, Math.min(state.filteredPhotos.length - 1, editSession.idx + delta)));
}

async function editGoTo(i) {
  if (i === editSession.idx) return;
  const kept = editIsDirty() ? basename(editSession.relPath) : null;
  if (kept && !(await persistEdit())) return;
  openEditModal(i);
  if (kept) {
    renderMain();
    // Beside Save, where "unsaved changes" shows: the status line under the
    // photo is rewritten by every render.
    $("#edit-dirty").textContent = `Saved your edit to ${kept}`;
  }
}

// Leaving the editor asks first when there is something unsaved. `how` is what
// the button said: Cancel means discard, so only that is confirmed; closing
// offers to save.
async function leaveEditor(how) {
  if (!editIsDirty()) { closeEditModal(); return true; }
  const name = basename(editSession.relPath);
  const answer = how === "cancel"
    ? await askChoice("Discard your changes?", `The edit to ${name} has not been saved.`, [
        { id: "discard", label: "Discard", danger: true }, { id: "keep", label: "Keep editing", primary: true }])
    : await askChoice(`Save your changes to ${name}?`, "They are lost if you close without saving.", [
        { id: "discard", label: "Discard", danger: true }, { id: "keep", label: "Keep editing" },
        { id: "save", label: "Save", primary: true }]);
  if (answer === "save") await saveEdit();
  else if (answer === "discard") closeEditModal();
  return $("#edit-modal").classList.contains("hidden");
}

// ---------- asking with more than OK and Cancel ----------
// Resolves with the id of the button pressed. Esc answers with the choice that
// changes nothing (the non-destructive, non-primary one if there is one) and
// Enter with the primary.
let choiceResolve = null;

function askChoice(title, body, choices) {
  $("#choice-title").textContent = title;
  $("#choice-body").textContent = body;
  const wrap = $("#choice-modal .choice-buttons");
  wrap.innerHTML = "";
  for (const c of choices) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = c.label;
    b.dataset.choice = c.id;
    if (c.primary) b.classList.add("primary");
    if (c.danger) b.classList.add("danger");
    b.addEventListener("click", () => answerChoice(c.id));
    wrap.appendChild(b);
  }
  const safe = choices.find((c) => !c.danger && !c.primary) || choices.find((c) => !c.danger) || choices[0];
  const primary = choices.find((c) => c.primary) || safe;
  $("#choice-modal").dataset.escape = safe.id;
  $("#choice-modal").dataset.enter = primary.id;
  $("#choice-modal").classList.remove("hidden");
  wrap.querySelector(".primary")?.focus();
  return new Promise((resolve) => { choiceResolve = resolve; });
}

function answerChoice(id) {
  $("#choice-modal").classList.add("hidden");
  const resolve = choiceResolve;
  choiceResolve = null;
  resolve(id);
}

// A reload or a closed tab would drop an unsaved edit the same way.
window.addEventListener("beforeunload", (e) => {
  if (editIsDirty()) { e.preventDefault(); e.returnValue = ""; }
});

// ---------- colour grading ----------
// Mirrors grading.ZONES. A wheel is a hue plus a strength; the panel shows them
// as two sliders rather than a dial because a slider can be typed into, nudged
// with an arrow key and read at a glance, and a dial cannot.
const GRADE_ZONES = [
  { k: "shadows",    label: "Shadows" },
  { k: "midtones",   label: "Midtones" },
  { k: "highlights", label: "Highlights" },
];
const GRADE_FIELDS = [
  { k: "hue", label: "Hue", min: 0, max: 359,
    hint: "Which way to push this tonal region. Does nothing on its own \u2014 raise Strength." },
  { k: "sat", label: "Strength", min: 0, max: 100,
    hint: "How far to push it. Brightness is untouched at any setting." },
  { k: "lum", label: "Luminance", min: -100, max: 100,
    hint: "Lift or drop this region, towards white or black and never past either." },
];
const GRADE_GLOBALS = [
  { k: "blending", label: "Blending", min: 0, max: 100, dflt: 50,
    hint: "How far the three regions reach into each other. Low keeps them apart." },
  { k: "balance", label: "Balance", min: -100, max: 100, dflt: 0,
    hint: "Slides the split between shadows and highlights." },
];
const gradeState = { zone: "shadows", built: false };

function currentGrading() { return editSession.edit.grading || {}; }
function gradeZoneVals(zone) { return currentGrading()[zone] || {}; }
function gradeZoneTouched(zone) {
  const v = gradeZoneVals(zone);
  return !!(Math.round(v.sat || 0) || Math.round(v.lum || 0));
}
function gradeGlobal(k, dflt) {
  const v = currentGrading()[k];
  return v == null ? dflt : Math.round(v);
}

function buildGradeFields() {
  if (gradeState.built) return;
  const row = (f, attr) =>
    `<label class="look-row" title="${escapeHtml(f.hint)}">` +
    `<span class="look-label">${f.label}</span>` +
    `<input type="range" data-${attr}="${f.k}" min="${f.min}" max="${f.max}" step="1" />` +
    `<span class="look-val" data-${attr}-val="${f.k}"></span></label>`;
  const zoneWrap = $("#grade-fields");
  if (!zoneWrap) return;
  zoneWrap.innerHTML = GRADE_FIELDS.map((f) => row(f, "grade")).join("");
  zoneWrap.addEventListener("input", (e) => {
    const sl = e.target.closest("input[type=range][data-grade]");
    if (sl) updateGradeZone({ [sl.dataset.grade]: parseInt(sl.value, 10) });
  });
  const globalWrap = $("#grade-globals");
  globalWrap.innerHTML = GRADE_GLOBALS.map((f) => row(f, "gradeg")).join("");
  globalWrap.addEventListener("input", (e) => {
    const sl = e.target.closest("input[type=range][data-gradeg]");
    if (sl) updateGrading({ [sl.dataset.gradeg]: parseInt(sl.value, 10) });
  });
  gradeState.built = true;
}

function renderGradePanel() {
  if (!$("#grade-fields")) return;
  buildGradeFields();
  const touched = GRADE_ZONES.filter((z) => gradeZoneTouched(z.k)).length;
  $("#grade-summary-state").textContent =
    touched ? `\u00b7 ${touched} zone${touched > 1 ? "s" : ""}` : "";
  $("#grade-zones").innerHTML = GRADE_ZONES.map((z) =>
    `<button type="button" class="grade-zone${z.k === gradeState.zone ? " active" : ""}` +
    `${gradeZoneTouched(z.k) ? " touched" : ""}" data-grade-zone="${z.k}">${z.label}</button>`
  ).join("");
  $$("#grade-zones .grade-zone").forEach((b) => {
    b.addEventListener("click", () => {
      gradeState.zone = b.dataset.gradeZone;
      renderGradePanel();
    });
  });
  const vals = gradeZoneVals(gradeState.zone);
  for (const f of GRADE_FIELDS) {
    const v = Math.round(vals[f.k] || 0);
    $(`#grade-fields input[data-grade="${f.k}"]`).value = v;
    const cell = $(`#grade-fields [data-grade-val="${f.k}"]`);
    // A hue with no strength behind it is a setting, not a change — say so
    // rather than showing a number that is doing nothing.
    cell.textContent = (f.k === "hue" && !Math.round(vals.sat || 0)) ? `${v}\u00b0 \u00b7 off` : v;
  }
  for (const f of GRADE_GLOBALS) {
    const v = gradeGlobal(f.k, f.dflt);
    $(`#grade-globals input[data-gradeg="${f.k}"]`).value = v;
    $(`#grade-globals [data-gradeg-val="${f.k}"]`).textContent = v;
  }
  $("#grade-reset").disabled = !gradeZoneTouched(gradeState.zone);
}

function updateGrading(patch) {
  const next = { ...currentGrading(), ...patch };
  editSession.edit.grading = canonGrading(next) ? next : null;
  renderGradePanel();
  setEditDirty();
  previewDuringDrag();
}

function updateGradeZone(patch) {
  const next = cloneGrading(currentGrading()) || {};
  next[gradeState.zone] = { ...gradeZoneVals(gradeState.zone), ...patch };
  editSession.edit.grading = canonGrading(next) ? next : null;
  renderGradePanel();
  setEditDirty();
  previewDuringDrag();
}

// ---------- colour mixer ----------
// Mirrors editing.HSL_BANDS and its centres. The swatch is only a label: the
// server interpolates between the centres, so a hue halfway between two bands
// is moved by a mix of both rather than by whichever button is highlighted.
const HSL_BANDS = [
  { k: "red",     label: "Red",     swatch: "#e04a4a" },
  { k: "orange",  label: "Orange",  swatch: "#e0903a" },
  { k: "yellow",  label: "Yellow",  swatch: "#d4c33a" },
  { k: "green",   label: "Green",   swatch: "#4aba5a" },
  { k: "aqua",    label: "Aqua",    swatch: "#3ec0c0" },
  { k: "blue",    label: "Blue",    swatch: "#4a7ade" },
  { k: "purple",  label: "Purple",  swatch: "#9a5ade" },
  { k: "magenta", label: "Magenta", swatch: "#d94aa8" },
];
const HSL_FIELDS = [
  { k: "hue", label: "Hue",
    hint: "Rotates this band around the colour circle, up to 30 degrees." },
  { k: "sat", label: "Saturation",
    hint: "How much of this colour there is. At -100 the band goes grey." },
  { k: "lum", label: "Luminance",
    hint: "Towards white or towards black, never past either \u2014 so the band keeps its hue." },
];
const hslState = { band: "red", built: false };

function currentHsl() { return editSession.edit.hsl || {}; }
function hslBandVals(band) { return currentHsl()[band] || {}; }
function hslBandTouched(band) {
  const v = hslBandVals(band);
  return HSL_FIELDS.some((f) => Math.round(v[f.k] || 0) !== 0);
}

function buildHslFields() {
  if (hslState.built) return;
  const wrap = $("#hsl-fields");
  if (!wrap) return;
  wrap.innerHTML = HSL_FIELDS.map((f) =>
    `<label class="look-row" title="${escapeHtml(f.hint)}">` +
    `<span class="look-label">${f.label}</span>` +
    `<input type="range" data-hsl="${f.k}" min="-100" max="100" step="1" />` +
    `<span class="look-val" data-hsl-val="${f.k}"></span></label>`).join("");
  wrap.addEventListener("input", (e) => {
    const sl = e.target.closest("input[type=range][data-hsl]");
    if (!sl) return;
    updateHsl({ [sl.dataset.hsl]: parseInt(sl.value, 10) });
  });
  hslState.built = true;
}

function renderHslPanel() {
  if (!$("#hsl-fields")) return;
  buildHslFields();
  const touched = HSL_BANDS.filter((b) => hslBandTouched(b.k)).length;
  $("#hsl-summary-state").textContent = touched ? `\u00b7 ${touched} band${touched > 1 ? "s" : ""}` : "";
  $("#hsl-bands").innerHTML = HSL_BANDS.map((b) =>
    `<button type="button" class="hsl-band${b.k === hslState.band ? " active" : ""}` +
    `${hslBandTouched(b.k) ? " touched" : ""}" data-band="${b.k}" title="${b.label}">` +
    `<span class="hsl-swatch" style="background:${b.swatch}"></span>` +
    `<span class="hsl-band-label">${b.label}</span></button>`).join("");
  $$("#hsl-bands .hsl-band").forEach((btn) => {
    btn.addEventListener("click", () => {
      hslState.band = btn.dataset.band;
      renderHslPanel();
    });
  });
  const vals = hslBandVals(hslState.band);
  for (const f of HSL_FIELDS) {
    const v = Math.round(vals[f.k] || 0);
    $(`#hsl-fields input[data-hsl="${f.k}"]`).value = v;
    $(`#hsl-fields [data-hsl-val="${f.k}"]`).textContent = v;
  }
  $("#hsl-reset").disabled = !hslBandTouched(hslState.band);
}

function updateHsl(patch) {
  const next = cloneHsl(currentHsl()) || {};
  const band = { ...hslBandVals(hslState.band), ...patch };
  // Drop what is back at neutral, so the edit returns to genuinely neutral
  // instead of carrying a band full of zeroes around.
  for (const f of HSL_FIELDS) if (!Math.round(band[f.k] || 0)) delete band[f.k];
  if (Object.keys(band).length) next[hslState.band] = band;
  else delete next[hslState.band];
  editSession.edit.hsl = Object.keys(next).length ? next : null;
  renderHslPanel();
  setEditDirty();
  previewDuringDrag();
}

// ---------- film emulation ----------
// The panel is deliberately a short list: pick a stock, then the four things
// worth pushing per photo. The rest of the chain's parameters live in the stock
// definitions on the server, where they belong.
const FILM_DEFAULT = {
  enabled: false, stock: "", strength: 100,
  contrast: 45, toe: 40, shoulder: 45,
  crosstalk: 25, warmth: 0, split: 0,
  halation: 30, halation_radius: 35,
  grain: 35, grain_size: 40, grain_rough: 55,
};

const FILM_FIELDS = [
  { k: "strength",        label: "Strength",   min: 0,    max: 100, hint: "How far towards the film response to go." },
  { k: "contrast",        label: "Curve",      min: 0,    max: 100, hint: "How much of the characteristic response to apply. At zero the tone is untouched." },
  { k: "toe",             label: "· toe",      min: 0,    max: 100, hint: "How pronounced the shadow roll-off is. Zero leaves the curve straight." },
  { k: "shoulder",        label: "· shoulder", min: 0,    max: 100, hint: "How pronounced the highlight roll-off is — film's highlight retention. Zero leaves the curve straight." },
  { k: "crosstalk",       label: "Crosstalk",  min: 0,    max: 100, hint: "Dye layers absorbing outside their own band, in density space. Half of what people call film colour." },
  { k: "split",           label: "· crossover", min: -100, max: 100, hint: "Which way the shadows and the highlights part company." },
  { k: "warmth",          label: "· warmth",   min: -100, max: 100, hint: "A density offset, so it acts like a filter over the lamp." },
  { k: "halation",        label: "Halation",   min: 0,    max: 100, hint: "Light scattering off the film base and re-exposing from behind. Strongest in red because that layer sits deepest." },
  { k: "halation_radius", label: "· radius",   min: 0,    max: 100, hint: "Relative to the frame, so the preview is the export." },
  { k: "grain",           label: "Grain",      min: 0,    max: 100, hint: "Crystal density. Strongest in the mid-tones, as on real film." },
  { k: "grain_size",      label: "· size",     min: 0,    max: 100, hint: "Crystal size, fixed against the frame — it will not turn to sand in a thumbnail." },
  { k: "grain_rough",     label: "· structure", min: 0,   max: 100, hint: "How much the grain clumps. At zero it is closer to sensor noise." },
];

const filmState = { stocks: [], defaults: null };

async function loadFilmStocks() {
  const res = await fetch("/api/film/stocks", { cache: "no-store" });
  if (!res.ok) return;
  const info = await res.json();
  filmState.stocks = info.stocks || [];
  filmState.defaults = info.defaults || null;
  renderFilmPanel();
}

function currentFilm() {
  // The stock list loads at boot, before any photo is open, so this has to cope
  // with there being no edit yet.
  const e = editSession.edit;
  return { ...FILM_DEFAULT, ...((e && e.film) || {}) };
}

function buildFilmFields() {
  const wrap = $("#film-fields");
  if (wrap.dataset.built) return;
  wrap.dataset.built = "1";
  wrap.innerHTML = FILM_FIELDS.map((f) =>
    `<label class="look-row" data-film-row="${f.k}" title="${escapeHtml(f.hint)}">` +
    `<span class="look-name">${f.label}</span>` +
    `<input type="range" data-film="${f.k}" min="${f.min}" max="${f.max}" step="1" />` +
    `<span class="look-val" data-film-val="${f.k}"></span></label>`).join("");
  wrap.addEventListener("input", (e) => {
    const sl = e.target.closest("input[type=range][data-film]");
    if (!sl) return;
    // Touching a slider means this is no longer that stock, it is yours.
    updateFilm({ [sl.dataset.film]: parseInt(sl.value, 10), stock: "" });
  });
}

// ---------- portrait (skin smoothing) ----------
// The faces come from the server (portrait.analyze on the corrected frame) and
// are only fetched once the panel is opened, so a photo nobody retouches never
// runs the face model.
const portraitState = { faces: null, rel: null, key: "" };
// Mirrors portrait.DEFAULT_PORTRAIT; all three are 0..100 and neutral at 0.
const PORTRAIT_KEYS = ["smooth", "teeth", "eyes"];

async function loadPortraitFaces() {
  const key = `${editSession.relPath}#${JSON.stringify(opticsPayload())}`;
  if (portraitState.key === key && portraitState.faces) return;
  portraitState.key = key;
  portraitState.faces = null;
  $("#portrait-faces").textContent = "finding faces…";
  const res = await fetch("/api/edit/faces", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, edit: opticsPayload() }),
  });
  if (!res.ok) { $("#portrait-faces").textContent = `face search failed: ${res.status}`; return; }
  if (portraitState.key !== key) return;        // moved on to another photo meanwhile
  portraitState.faces = (await res.json()).faces;
  renderPortraitPanel();
  drawOverlay();
}

function portraitSummary() {
  const p = editSession.edit.portrait;
  const on = PORTRAIT_KEYS.filter((k) => p && p[k]);
  return on.length ? `· ${on.join(", ")}` : "";
}

function renderPortraitPanel() {
  if (!$("#portrait-smooth") || !editSession.edit) return;
  for (const k of PORTRAIT_KEYS) {
    const v = (editSession.edit.portrait && editSession.edit.portrait[k]) || 0;
    $(`#portrait-${k}`).value = v;
    $(`#portrait-${k}-val`).textContent = v;
  }
  $("#portrait-summary-state").textContent = portraitSummary();
  const faces = portraitState.faces;
  if (faces) {
    $("#portrait-faces").textContent = faces.length
      ? `${faces.length} face${faces.length === 1 ? "" : "s"} found`
      : "No faces found. Faces have to be at least a few percent of the frame.";
    for (const k of PORTRAIT_KEYS) $(`#portrait-${k}`).disabled = !faces.length;
  }
}

async function removeBlemishes() {
  const status = $("#portrait-faces");
  status.textContent = "looking for blemishes at full resolution…";
  const res = await fetch("/api/edit/blemishes", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, edit: opticsPayload() }),
  });
  if (!res.ok) { status.textContent = `blemish search failed: ${res.status}`; return; }
  const { ops, faces } = await res.json();
  // Skip any already covered by a spot, so pressing it twice does not stack.
  const have = healOps();
  const fresh = ops.filter((o) => !have.some((h) => h.kind === "spot"
    && Math.hypot(h.points[0][0] - o.points[0][0], h.points[0][1] - o.points[0][1]) < Math.max(h.radius, o.radius)));
  if (fresh.length) {
    setHealOps([...have, ...fresh]);
    repairChanged();
    // Open the list they went into, so each one can be seen and taken out.
    $("#edit-repair-group").open = true;
  }
  status.textContent = !faces ? "No faces found."
    : fresh.length === 1 ? "removed 1 blemish; it is a spot in Heal & red eye"
    : fresh.length ? `removed ${fresh.length} blemishes; they are spots in Heal & red eye`
    : ops.length ? "those are already removed" : "No blemishes found.";
}

function drawPortraitFaces(ctx, mr) {
  if (!$("#edit-portrait-group").open || !portraitState.faces) return;
  ctx.save();
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = 1.2;
  for (const f of portraitState.faces) {
    const [x, y, w, h] = f.box;
    ctx.strokeRect(fx2px(mr, x), fy2px(mr, y), w * mr.w, h * mr.h);
  }
  ctx.restore();
}

function bindPortraitPanel() {
  $("#edit-portrait-group").addEventListener("toggle", () => {
    if ($("#edit-portrait-group").open && editSession.relPath) loadPortraitFaces();
    drawOverlay();
  });
  $("#portrait-blemishes").addEventListener("click", removeBlemishes);
  for (const k of PORTRAIT_KEYS) {
    $(`#portrait-${k}`).addEventListener("input", (e) => {
      const v = parseInt(e.target.value, 10);
      const next = { smooth: 0, teeth: 0, eyes: 0, ...(editSession.edit.portrait || {}), [k]: v };
      editSession.edit.portrait = PORTRAIT_KEYS.some((x) => next[x]) ? next : null;
      $(`#portrait-${k}-val`).textContent = v;
      $("#portrait-summary-state").textContent = portraitSummary();
      setEditDirty();
      previewDuringDrag();
    });
  }
}

// ---------- lens & perspective ----------
// Mirrors lens.DEFAULT_LENS and transform.DEFAULT_TRANSFORM. Both are applied
// before anything else, and every position in the edit is a fraction of the
// frame they produce, which is the frame on screen.
const LENS_DEFAULT = { distortion: 0, ca_red_cyan: 0, ca_blue_yellow: 0, ca_auto: false,
                       vignette_amount: 0, vignette_midpoint: 50 };
const TRANSFORM_DEFAULT = { vertical: 0, horizontal: 0, rotate: 0, aspect: 0, scale: 100,
                            offset_x: 0, offset_y: 0, upright: "off" };
const LENS_FIELDS = [
  { k: "distortion", label: "Distortion", min: -100, max: 100, step: 1, fmt: 0,
    hint: "+ straightens barrel distortion, − straightens pincushion" },
  { k: "ca_red_cyan", label: "Red / cyan", min: -100, max: 100, step: 1, fmt: 0,
    hint: "Removes red or cyan fringes along edges towards the corners" },
  { k: "ca_blue_yellow", label: "Blue / yellow", min: -100, max: 100, step: 1, fmt: 0,
    hint: "Removes blue or yellow fringes along edges towards the corners" },
  { k: "vignette_amount", label: "Vignetting", min: -100, max: 100, step: 1, fmt: 0,
    hint: "+ brightens corners the lens left dark" },
  { k: "vignette_midpoint", label: "Midpoint", min: 0, max: 100, step: 1, fmt: 0,
    hint: "How far in from the corners the falloff reaches" },
];
const TRANSFORM_FIELDS = [
  { k: "vertical", label: "Vertical", min: -100, max: 100, step: 0.5, fmt: 1,
    hint: "Keystone: + widens the top, for buildings that lean back" },
  { k: "horizontal", label: "Horizontal", min: -100, max: 100, step: 0.5, fmt: 1,
    hint: "Keystone: + widens the left" },
  { k: "rotate", label: "Rotate", min: -45, max: 45, step: 0.1, fmt: 1,
    hint: "Rotation inside the perspective. To straighten a horizon, use Straighten" },
  { k: "aspect", label: "Aspect", min: -100, max: 100, step: 1, fmt: 0,
    hint: "Undoes the stretch a keystone leaves: + wider, − taller" },
  { k: "scale", label: "Scale", min: 10, max: 200, step: 1, fmt: 0,
    hint: "Zoom, to push the blank edges a perspective leaves out of the frame" },
  { k: "offset_x", label: "Offset X", min: -100, max: 100, step: 1, fmt: 0 },
  { k: "offset_y", label: "Offset Y", min: -100, max: 100, step: 1, fmt: 0 },
];
const opticsState = { built: false };

function opticIsNeutral(o, defaults) {
  return Object.keys(defaults).every((k) => k === "vignette_midpoint" || k === "upright"
    || (o[k] ?? defaults[k]) === defaults[k]) && (!o.upright || o.upright === "off");
}

function opticsPayload() {
  return { lens: editSession.edit.lens, transform: editSession.edit.transform };
}

function buildOpticsFields() {
  if (opticsState.built) return;
  const row = (group, f) => `<label class="look-row"${f.hint ? ` title="${escapeHtml(f.hint)}"` : ""}>`
    + `<span class="look-name">${f.label}</span>`
    + `<input type="range" data-optic="${group}:${f.k}" min="${f.min}" max="${f.max}" step="${f.step}" />`
    + `<span class="look-val" data-optic-val="${group}:${f.k}"></span></label>`;
  $("#optics-lens-fields").innerHTML = LENS_FIELDS.map((f) => row("lens", f)).join("");
  $("#optics-transform-fields").innerHTML = TRANSFORM_FIELDS.map((f) => row("transform", f)).join("");
  opticsState.built = true;
}

function renderOpticsPanel() {
  if (!$("#optics-lens-fields") || !editSession.edit) return;
  buildOpticsFields();
  const lens = { ...LENS_DEFAULT, ...(editSession.edit.lens || {}) };
  const tf = { ...TRANSFORM_DEFAULT, ...(editSession.edit.transform || {}) };
  for (const [group, fields, vals] of [["lens", LENS_FIELDS, lens], ["transform", TRANSFORM_FIELDS, tf]]) {
    for (const f of fields) {
      $(`[data-optic="${group}:${f.k}"]`).value = vals[f.k];
      $(`[data-optic-val="${group}:${f.k}"]`).textContent = Number(vals[f.k]).toFixed(f.fmt);
    }
  }
  $("#lens-ca-auto").checked = !!lens.ca_auto;
  const on = [editSession.edit.lens && "lens", editSession.edit.transform && "perspective"].filter(Boolean);
  $("#optics-summary-state").textContent = on.length ? `· ${on.join(" + ")}` : "";
}

// Write one value, and drop the whole block back to null once it is at its
// defaults, so an untouched panel never makes a photo count as edited.
function setOptic(group, key, value) {
  const defaults = group === "lens" ? LENS_DEFAULT : TRANSFORM_DEFAULT;
  const next = { ...defaults, ...(editSession.edit[group] || {}), [key]: value };
  editSession.edit[group] = opticIsNeutral(next, defaults) ? null : next;
}

async function applyUpright(mode) {
  const status = $("#optics-status");
  status.textContent = "reading the lines in the photo…";
  const res = await fetch("/api/edit/upright", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, mode, edit: opticsPayload() }),
  });
  if (!res.ok) { status.textContent = `upright failed: ${res.status}`; return; }
  const d = await res.json();
  if (!d.found) {
    status.textContent = "No lines strong enough to straighten by. Set it by hand below.";
    return;
  }
  // The modes replace one another, as Lightroom's do. Each one measures the
  // roll afresh from the uncorrected lines, so keeping the last mode's rotation
  // (or a Level's tilt) alongside a new one would turn the photo twice.
  if (d.tilt !== null) {
    // Level is a straighten, which crops to the largest level rectangle
    // instead of needing a zoom (see transform.py).
    editSession.edit.tilt = d.tilt;
    editSession.edit.transform = null;
    status.textContent = `levelled by ${d.tilt.toFixed(1)}° with Straighten`;
  } else {
    editSession.edit.transform = d.transform;
    editSession.edit.tilt = 0;
    status.textContent = "applied; the sliders below show what it did";
  }
  syncCropControls();
  renderOpticsPanel();
  setEditDirty();
  drawOverlay();
  fetchEditPreview(true);
}

function bindOpticsPanel() {
  const body = $("#edit-optics-group");
  body.addEventListener("input", (e) => {
    const sl = e.target.closest("[data-optic]");
    if (!sl) return;
    const [group, key] = sl.dataset.optic.split(":");
    setOptic(group, key, parseFloat(sl.value));
    const f = (group === "lens" ? LENS_FIELDS : TRANSFORM_FIELDS).find((x) => x.k === key);
    $(`[data-optic-val="${group}:${key}"]`).textContent = Number(sl.value).toFixed(f.fmt);
    setEditDirty();
    previewDuringDrag();
  });
  $("#lens-ca-auto").addEventListener("change", (e) => {
    setOptic("lens", "ca_auto", e.target.checked);
    renderOpticsPanel();
    setEditDirty();
    fetchEditPreview(true);
  });
  $$("#optics-upright [data-upright]").forEach((b) =>
    b.addEventListener("click", () => applyUpright(b.dataset.upright)));
  $("#optics-reset").addEventListener("click", () => {
    editSession.edit.lens = null;
    editSession.edit.transform = null;
    $("#optics-status").textContent = "";
    renderOpticsPanel();
    setEditDirty();
    drawOverlay();
    fetchEditPreview(true);
  });
}

// ---------- repairs: spot, heal, clone, red eye ----------
// Stored as the server normalizes them, in original-frame fractions: healing
// is {ops: [...]} and red-eye {enabled, corrections: [...]}, each null when
// empty. A radius is a fraction of the frame WIDTH, as a brush stroke's is.
const REPAIR_TOOLS = ["spot", "heal", "clone", "redeye", "peteye"];
const REPAIR_LABELS = { spot: "Spot", heal: "Heal", clone: "Clone" };
const repairState = { size: 20, cloneSrc: null };   // size in 1/1000 of the width
const EYE_CLICK_RADIUS = 0.02;   // a click with no drag: an eye in a portrait

function isRepairTool(t) { return REPAIR_TOOLS.includes(t); }
function isRepairBrush(t) { return t === "spot" || t === "heal" || t === "clone"; }
function healOps() { return (editSession.edit.healing && editSession.edit.healing.ops) || []; }
function eyeFixes() { return (editSession.edit.redeye && editSession.edit.redeye.corrections) || []; }
function setHealOps(ops) { editSession.edit.healing = ops.length ? { ops } : null; }
function setEyeFixes(list) {
  editSession.edit.redeye = list.length ? { enabled: true, corrections: list } : null;
}
function repairsShown() {
  return isRepairTool(editSession.tool) || $("#edit-repair-group").open;
}

function repairChanged() {
  renderRepairPanel();
  setEditDirty();
  drawOverlay();
  fetchEditPreview(true);
}

function renderRepairPanel() {
  if (!$("#repair-list") || !editSession.edit) return;
  $("#repair-size").value = repairState.size;
  $("#repair-size-val").textContent = repairState.size;
  const ops = healOps(), fixes = eyeFixes();
  $("#repair-list").innerHTML = ops.map((op, i) =>
    `<div class="repair-item"><label><input type="checkbox" data-heal-on="${i}"`
    + `${op.enabled === false ? "" : " checked"} /> ${REPAIR_LABELS[op.kind] || op.kind} ${i + 1}</label>`
    + `<button type="button" class="quiet" data-heal-del="${i}" aria-label="Remove">×</button></div>`).join("");
  $("#redeye-list").innerHTML = fixes.map((c, i) =>
    `<div class="repair-item"><span>${c.kind === "pet" ? "Pet eye" : "Red eye"} ${i + 1}</span>`
    + `<button type="button" class="quiet" data-eye-del="${i}" aria-label="Remove">×</button></div>`).join("");
  const n = ops.length + fixes.length;
  $("#repair-summary-state").textContent = n ? `· ${n}` : "";
}

function repairDown(e, f) {
  const tool = editSession.tool;
  const radius = repairState.size / 1000;
  const base = { radius, feather: 50, opacity: 100, method: "ns", enabled: true };
  e.preventDefault();
  if (tool === "spot") {
    setHealOps([...healOps(), { ...base, kind: "spot", points: [[f.x, f.y]] }]);
    repairChanged();
    return;
  }
  if (tool === "clone" && (e.altKey || !repairState.cloneSrc)) {
    repairState.cloneSrc = { x: f.x, y: f.y };
    setEditTool("clone");            // refreshes the hint for the next step
    drawOverlay();
    return;
  }
  if (tool === "heal" || tool === "clone") {
    const op = { ...base, kind: tool, points: [[f.x, f.y]] };
    // Non-aligned, as Photoshop's clone is by default: every stroke reads from
    // the same source point, however far from the last one it starts.
    if (tool === "clone") {
      op.dx = repairState.cloneSrc.x - f.x;
      op.dy = repairState.cloneSrc.y - f.y;
    }
    setHealOps([...healOps(), op]);
    editSession.drag = { kind: "repair-paint", op };
  } else {
    editSession.drag = { kind: "eye", start: f, r: 0, eyeKind: tool === "peteye" ? "pet" : "red" };
  }
  $("#edit-overlay").setPointerCapture(e.pointerId);
  drawOverlay();
}

// A drag's length as a fraction of the frame width, which is what a radius is.
function widthFraction(a, b) {
  const aspect = (editSession.natural.h || 1) / (editSession.natural.w || 1);
  return Math.hypot(b.x - a.x, (b.y - a.y) * aspect);
}

function repairMove(e, f, d) {
  if (d.kind === "eye") {
    d.r = widthFraction(d.start, f);
    scheduleOverlay();
    return;
  }
  const pts = d.op.points;
  const step = Math.max(0.002, d.op.radius * 0.33);
  for (const q of pointerPath(e, f)) {
    const last = pts[pts.length - 1];
    if (Math.hypot(q.x - last[0], q.y - last[1]) >= step) pts.push([q.x, q.y]);
  }
  scheduleOverlay();
  setEditDirty();
  previewDuringDrag();
}

async function repairUp(d) {
  if (d.kind === "repair-paint") { repairChanged(); return; }
  const status = $("#repair-status");
  const r = d.r < 0.004 ? EYE_CLICK_RADIUS : d.r;
  status.textContent = "checking the eye…";
  const res = await fetch("/api/edit/redeye/region", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, cx: d.start.x, cy: d.start.y, r,
                           kind: d.eyeKind, edit: opticsPayload() }),
  });
  const { correction } = await res.json();
  if (!correction) {
    status.textContent = d.eyeKind === "pet"
      ? "No eye glow found there. Drag from the centre of the eye to the edge of the glow."
      : "No red pupil found there. Drag from the pupil's centre to the edge of the iris; "
        + "a circle much larger than the eye is refused, and so is anything red that is not an eye.";
    drawOverlay();
    return;
  }
  status.textContent = "";
  setEyeFixes([...eyeFixes(), correction]);
  repairChanged();
}

async function findRedEyes() {
  const status = $("#repair-status");
  status.textContent = "looking for eyes…";
  const res = await fetch("/api/edit/redeye/detect", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, edit: opticsPayload() }),
  });
  if (!res.ok) { status.textContent = `red-eye search failed: ${res.status}`; return; }
  const { corrections } = await res.json();
  // Skip what is already fixed, so pressing it twice does not double up.
  const have = eyeFixes();
  const fresh = corrections.filter((c) =>
    !have.some((h) => Math.hypot(h.cx - c.cx, h.cy - c.cy) < Math.max(h.r, c.r)));
  status.textContent = fresh.length
    ? `fixed ${fresh.length} red eye${fresh.length === 1 ? "" : "s"}`
    : corrections.length ? "those eyes are already fixed"
    : "No red eyes found. Faces need to be fairly large and facing the camera; "
      + "you can still fix an eye by hand.";
  if (fresh.length) { setEyeFixes([...have, ...fresh]); repairChanged(); }
}

function drawRepairs(ctx, mr) {
  const px = (x) => fx2px(mr, x), py = (y) => fy2px(mr, y);
  const path = (pts, ox = 0, oy = 0) => {
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(px(x + ox), py(y + oy)) : ctx.moveTo(px(x + ox), py(y + oy))));
    if (pts.length === 1) ctx.lineTo(px(pts[0][0] + ox) + 0.01, py(pts[0][1] + oy));
  };
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const op of healOps()) {
    const on = op.enabled !== false;
    const rad = op.radius * mr.w;
    // A translucent band as wide as the brush, then a thin line down its middle.
    ctx.lineWidth = rad * 2;
    ctx.strokeStyle = on ? "rgba(255,255,255,0.16)" : "rgba(255,255,255,0.06)";
    path(op.points);
    ctx.stroke();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = on ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.35)";
    if (op.kind === "spot") {
      ctx.beginPath();
      ctx.arc(px(op.points[0][0]), py(op.points[0][1]), rad, 0, Math.PI * 2);
    } else {
      path(op.points);
    }
    ctx.stroke();
    if (op.kind === "clone") {
      // Where it copies from: the same path, dashed, and a line joining the two.
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = on ? "rgba(122,167,255,0.9)" : "rgba(122,167,255,0.35)";
      path(op.points, op.dx, op.dy);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px(op.points[0][0] + op.dx), py(op.points[0][1] + op.dy));
      ctx.lineTo(px(op.points[0][0]), py(op.points[0][1]));
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  for (const c of eyeFixes()) {
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = c.kind === "pet" ? "rgba(255,209,102,0.9)" : "rgba(255,120,120,0.9)";
    ctx.beginPath();
    ctx.arc(px(c.cx), py(c.cy), c.r * mr.w, 0, Math.PI * 2);
    ctx.stroke();
  }
  const d = editSession.drag;
  if (d && d.kind === "eye") {
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "#ffffff";
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.arc(px(d.start.x), py(d.start.y), Math.max(d.r, 0.002) * mr.w, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (editSession.tool === "clone" && repairState.cloneSrc) {
    const x = px(repairState.cloneSrc.x), y = py(repairState.cloneSrc.y);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#7aa7ff";
    ctx.beginPath();
    ctx.moveTo(x - 8, y); ctx.lineTo(x + 8, y); ctx.moveTo(x, y - 8); ctx.lineTo(x, y + 8);
    ctx.stroke();
  }
  // The brush ring is the cursor for the three painting tools, as for masks.
  if (isRepairBrush(editSession.tool) && editSession.hover) {
    const x = px(editSession.hover.x), y = py(editSession.hover.y);
    const rad = repairState.size / 1000 * mr.w;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.stroke();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.stroke();
  }
  ctx.restore();
}

function bindRepairPanel() {
  $$("#edit-repair-group [data-repair-tool]").forEach((b) => b.addEventListener("click", () => {
    const tool = b.dataset.repairTool;
    // A mask selected under a repair tool would draw its handles over the work.
    if (editSession.tool !== tool) selectMask(-1, { silent: true });
    setEditTool(editSession.tool === tool ? null : tool);
    drawOverlay();
  }));
  $("#repair-size").addEventListener("input", (e) => {
    repairState.size = parseInt(e.target.value, 10);
    $("#repair-size-val").textContent = repairState.size;
    drawOverlay();
  });
  $("#redeye-auto").addEventListener("click", findRedEyes);
  $("#repair-list").addEventListener("click", (e) => {
    const del = e.target.closest("[data-heal-del]");
    if (del) { setHealOps(healOps().filter((_, i) => i !== Number(del.dataset.healDel))); repairChanged(); }
  });
  $("#repair-list").addEventListener("change", (e) => {
    const on = e.target.closest("[data-heal-on]");
    if (on) { healOps()[Number(on.dataset.healOn)].enabled = on.checked; repairChanged(); }
  });
  $("#redeye-list").addEventListener("click", (e) => {
    const del = e.target.closest("[data-eye-del]");
    if (del) { setEyeFixes(eyeFixes().filter((_, i) => i !== Number(del.dataset.eyeDel))); repairChanged(); }
  });
  // Opening or closing the panel shows or hides the markers on the photo.
  $("#edit-repair-group").addEventListener("toggle", drawOverlay);
}

// ---------- looks (colour LUTs) ----------
// The library is app-global and lists name and key only; the table stays on
// the server. The edit stores {key, name, amount}.
const lookState = { luts: [], matching: false };

async function loadLooks() {
  const res = await fetch("/api/luts", { cache: "no-store" });
  lookState.luts = res.ok ? (await res.json()).luts : [];
  renderLookPanel();
  renderRepairPanel();
}

function renderLookPanel() {
  const sel = $("#look-select");
  if (!sel || !editSession.edit) return;
  const cur = editSession.edit.lut;
  const known = lookState.luts.some((l) => cur && l.key === cur.key);
  // A look the edit uses but the library no longer has (deleted, or another
  // machine) still renders from the project's copy, so it stays selectable.
  const extra = cur && !known ? [{ key: cur.key, name: cur.name || "(this photo's look)" }] : [];
  sel.innerHTML = `<option value="">None</option>` + [...lookState.luts, ...extra]
    .map((l) => `<option value="${l.key}">${escapeHtml(l.name || l.key)}</option>`).join("");
  sel.value = cur ? cur.key : "";
  $("#look-amount").value = cur ? cur.amount : 100;
  $("#look-amount-val").textContent = cur ? cur.amount : "—";
  $("#look-amount").disabled = !cur;
  $("#look-delete").disabled = !(cur && known);
  $("#look-summary-state").textContent = cur ? `· ${cur.name || "on"}` : "";
  $("#look-match-picker").classList.toggle("hidden", !lookState.matching);
}

function setLook(entry, amount) {
  editSession.edit.lut = entry
    ? { key: entry.key, name: entry.name || "", amount: amount ?? (editSession.edit.lut?.amount || 100) }
    : null;
  renderLookPanel();
  setEditDirty();
  fetchEditPreview(true);
}

async function importLook(file) {
  const status = $("#look-status");
  status.textContent = `reading ${file.name}…`;
  const res = await fetch("/api/luts/import", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: file.name, text: await file.text() }),
  });
  const d = await res.json();
  if (!res.ok) { status.textContent = d.detail || `import failed: ${res.status}`; return; }
  status.textContent = `imported ${d.name} (${d.dim === 3 ? `${d.size}³ cube` : `1D, ${d.size} points`})`;
  await loadLooks();
  setLook(d, 100);
}

// The reference comes from the same scene: matching is meant for one moment
// shot on two bodies, and the scene is where those frames sit together.
function renderMatchPicker() {
  const photo = state.filteredPhotos[editSession.idx];
  const scene = photo ? photo.scene : null;
  const pool = (state.byScene.get(scene) || []).filter((p) => p.rel_path !== editSession.relPath);
  $("#look-match-thumbs").innerHTML = pool.length
    ? pool.map((p) => `<button type="button" class="look-match-thumb" data-rel="${escapeAttr(p.rel_path)}" `
        + `title="${escapeHtml(basename(p.rel_path))}"><img loading="lazy" src="${thumbUrl(p)}" alt="" /></button>`).join("")
    : `<p class="hsl-note">No other photos in this scene.</p>`;
  $$("#look-match-thumbs .look-match-thumb").forEach((b) =>
    b.addEventListener("click", () => matchLookTo(b.dataset.rel)));
}

async function matchLookTo(reference) {
  const status = $("#look-status");
  status.textContent = `matching colours to ${basename(reference)}…`;
  const res = await fetch("/api/luts/match", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: editSession.relPath, reference }),
  });
  const d = await res.json();
  if (!res.ok) { status.textContent = d.detail || `match failed: ${res.status}`; return; }
  status.textContent = "";
  lookState.matching = false;
  await loadLooks();
  setLook(d, 100);
}

async function deleteLook() {
  const cur = editSession.edit.lut;
  if (!cur || !confirm(`Remove "${cur.name}" from the look library?\n\n`
      + "Photos already using it keep it; it just stops being offered.")) return;
  await fetch("/api/luts/delete", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key: cur.key }),
  });
  await loadLooks();
}

function bindLookPanel() {
  $("#look-select").addEventListener("change", (e) => {
    const key = e.target.value;
    const entry = lookState.luts.find((l) => l.key === key)
      || (editSession.edit.lut && editSession.edit.lut.key === key ? editSession.edit.lut : null);
    setLook(key ? entry : null);
  });
  $("#look-amount").addEventListener("input", (e) => {
    if (!editSession.edit.lut) return;
    editSession.edit.lut.amount = parseInt(e.target.value, 10);
    $("#look-amount-val").textContent = e.target.value;
    setEditDirty();
    previewDuringDrag();
  });
  $("#look-import").addEventListener("click", () => $("#look-file").click());
  $("#look-file").addEventListener("change", (e) => {
    const f = e.target.files[0];
    e.target.value = "";          // picking the same file again still fires
    if (f) importLook(f);
  });
  $("#look-match").addEventListener("click", () => {
    lookState.matching = !lookState.matching;
    if (lookState.matching) renderMatchPicker();
    renderLookPanel();
  });
  $("#look-delete").addEventListener("click", deleteLook);
}

function renderFilmPanel() {
  if (!$("#film-fields")) return;
  buildFilmFields();
  const f = currentFilm();
  const on = !!f.enabled;
  $("#film-enabled").checked = on;
  $("#film-summary-state").textContent = on ? (f.stock ? `· ${f.stock}` : "· on") : "";
  $("#edit-film-group").classList.toggle("film-on", on);
  // "Default" is a chip like the stocks, because getting back to the starting
  // point is the same kind of action as picking one.
  $("#film-stocks").innerHTML =
    `<button class="film-stock film-default" data-stock="">Default</button>` +
    filmState.stocks.map((st) =>
      `<button class="film-stock${st.name === f.stock ? " active" : ""}" ` +
      `data-stock="${escapeHtml(st.name)}">${escapeHtml(st.name)}</button>`).join("");
  $$("#film-stocks .film-stock").forEach((b) => {
    b.addEventListener("click", () => {
      if (!b.dataset.stock) {
        const d = filmState.defaults || FILM_DEFAULT;
        const { enabled, stock, ...params } = d;
        updateFilm({ ...params, enabled: true, stock: "" });
        return;
      }
      const st = filmState.stocks.find((x) => x.name === b.dataset.stock);
      if (!st) return;
      const { name, ...params } = st;
      updateFilm({ ...params, enabled: true, stock: name });
    });
  });
  for (const fld of FILM_FIELDS) {
    $(`#film-fields input[data-film="${fld.k}"]`).value = f[fld.k];
    $(`#film-fields [data-film-val="${fld.k}"]`).textContent = f[fld.k];
  }
  $$("#film-fields input").forEach((i) => { i.disabled = !on; });
}

function updateFilm(patch) {
  editSession.edit.film = { ...currentFilm(), ...patch };
  renderFilmPanel();
  setEditDirty();
  previewDuringDrag();
}

// ---------- watermark ----------
const wmState = { cameras: [], meta: null, built: false };

async function loadWatermarkInfo(relPath) {
  try {
    const q = relPath ? `?rel_path=${encodeURIComponent(relPath)}` : "";
    const res = await fetch("/api/watermark/info" + q, { cache: "no-store" });
    if (!res.ok) return;
    const info = await res.json();
    wmState.cameras = info.cameras || [];
    wmState.meta = info.meta || null;
  } catch { /* the panel still works, just without the catalogue */ }
  renderWatermarkPanel();
}

function buildWatermarkControls() {
  if (wmState.built) return;
  $("#wm-style").innerHTML = WM_STYLES.map((s) =>
    `<button type="button" data-wm-style="${s}">${s}</button>`).join("");
  $("#wm-position").innerHTML = WM_POSITIONS.map((p) =>
    `<button type="button" data-wm-pos="${p}" title="${p}">` +
    `<span class="wm-dot"></span></button>`).join("");
  $("#wm-tokens").innerHTML =
    `<span class="wm-token-label">insert</span>` +
    WM_TOKENS.map((t) => `<button type="button" data-wm-token="${t}">{${t}}</button>`).join("");
  wmState.built = true;
}

// The watermark only exists on the edit once it is switched on; until then the
// photo carries no watermark at all and stays byte-identical on export.
function ensureWatermark() {
  if (!editSession.edit.watermark) {
    editSession.edit.watermark = { ...WM_DEFAULT, enabled: true, name: lastWatermarkName() };
  }
  return editSession.edit.watermark;
}

// Reuse the last name typed anywhere: nobody wants to retype it per photo.
function lastWatermarkName() {
  try { return localStorage.getItem("pcls.wm.name") || ""; } catch { return ""; }
}
function rememberWatermarkName(name) {
  try { localStorage.setItem("pcls.wm.name", name); } catch { /* private mode */ }
}

function renderWatermarkPanel() {
  buildWatermarkControls();
  const w = editSession.edit ? editSession.edit.watermark : null;
  const on = !!(w && w.enabled);
  const v = w || WM_DEFAULT;
  $("#wm-enabled").checked = on;
  $("#wm-summary-state").textContent = on ? "· on" : "";
  $("#edit-wm-group").classList.toggle("wm-on", on);
  $$("#wm-style [data-wm-style]").forEach((b) =>
    b.classList.toggle("active", b.dataset.wmStyle === v.style));
  $$("#wm-position [data-wm-pos]").forEach((b) =>
    b.classList.toggle("active", b.dataset.wmPos === v.position));
  $("#wm-name").value = v.name;
  $("#wm-camera").value = v.camera;
  for (const k of ["line1", "line2", "line3"]) $(`#wm-${k}`).value = v[k];
  for (const k of ["size", "opacity", "margin"]) {
    $(`#wm-${k}`).value = v[k];
    $(`#wm-${k}-val`).textContent = v[k];
  }
  $("#wm-color").value = v.color;
  $("#wm-camera-list").innerHTML = wmState.cameras
    .map((c) => `<option value="${escapeHtml(c.name)}"></option>`).join("");
  const m = wmState.meta;
  $("#wm-detected").textContent = m && m.camera
    ? `EXIF: ${m.camera}${m.lens ? " · " + m.lens : ""}` +
      `${m.focal ? " · " + [m.focal, m.aperture, m.shutter, m.iso_text].filter(Boolean).join(" · ") : ""}`
    : "No EXIF on this photo — fill the fields in by hand.";
}

// `immediate` renders now; `dragging` is for a slider, which gets the drafts
// a drag does. Neither waits for a pause, which is right for typing.
function updateWatermark(patch, immediate, dragging) {
  const w = ensureWatermark();
  Object.assign(w, patch);
  if (patch.name != null) rememberWatermarkName(patch.name);
  renderWatermarkPanel();
  setEditDirty();
  if (dragging) previewDuringDrag();
  else fetchEditPreview(!!immediate);
}

function bindWatermarkUi() {
  buildWatermarkControls();
  $("#wm-enabled").addEventListener("change", (e) => {
    if (e.target.checked) ensureWatermark().enabled = true;
    else if (editSession.edit.watermark) editSession.edit.watermark.enabled = false;
    renderWatermarkPanel();
    setEditDirty();
    fetchEditPreview(true);
  });
  $("#wm-style").addEventListener("click", (e) => {
    const b = e.target.closest("[data-wm-style]");
    if (b) updateWatermark({ style: b.dataset.wmStyle, enabled: true }, true);
  });
  $("#wm-position").addEventListener("click", (e) => {
    const b = e.target.closest("[data-wm-pos]");
    if (b) updateWatermark({ position: b.dataset.wmPos, enabled: true }, true);
  });
  $("#wm-name").addEventListener("input", (e) => updateWatermark({ name: e.target.value }));
  $("#wm-camera").addEventListener("input", (e) => updateWatermark({ camera: e.target.value }));
  for (const k of ["line1", "line2", "line3"]) {
    $(`#wm-${k}`).addEventListener("input", (e) => updateWatermark({ [k]: e.target.value }));
  }
  for (const k of ["size", "opacity", "margin"]) {
    $(`#wm-${k}`).addEventListener("input", (e) =>
      updateWatermark({ [k]: parseInt(e.target.value, 10) }, false, true));
  }
  $("#wm-color").addEventListener("input", (e) => updateWatermark({ color: e.target.value }, true));
  $("#wm-white").addEventListener("click", () => updateWatermark({ color: "#ffffff" }, true));
  $("#wm-black").addEventListener("click", () => updateWatermark({ color: "#000000" }, true));
  // Token chips type into whichever line was last focused.
  $("#wm-tokens").addEventListener("click", (e) => {
    const b = e.target.closest("[data-wm-token]");
    if (!b) return;
    const target = $(`#wm-${wmState.lastLine || "line1"}`);
    const token = `{${b.dataset.wmToken}}`;
    const at = target.selectionStart ?? target.value.length;
    target.value = target.value.slice(0, at) + token + target.value.slice(at);
    target.focus();
    target.setSelectionRange(at + token.length, at + token.length);
    updateWatermark({ [target.id.replace("wm-", "")]: target.value }, true);
  });
  $$("[data-wm-line]").forEach((el) =>
    el.addEventListener("focus", () => { wmState.lastLine = el.id.replace("wm-", ""); }));
}

// ---------- local adjustments: panel ----------
// A mask that the server would drop on normalize (a brush with nothing painted)
// is invisible to save/dirty-tracking, so it never counts as an unsaved change.
function persistableMasks(masks) {
  return (masks || []).filter((m) => m.type !== "brush" || (m.strokes && m.strokes.length));
}

function activeMask() {
  const i = editSession.activeMask;
  return (editSession.edit && i >= 0) ? (editSession.edit.masks[i] || null) : null;
}

function selectMask(idx, opts) {
  const masks = (editSession.edit && editSession.edit.masks) || [];
  editSession.activeMask = (idx >= 0 && idx < masks.length) ? idx : -1;
  const m = activeMask();
  // Picking a brush mask re-arms the brush, so painting continues immediately.
  if (m && m.type === "brush") setEditTool("brush");
  else if (editSession.tool && (!m || m.type !== editSession.tool)) setEditTool(null);
  renderMaskList();
  renderMaskDetail();
  if (!(opts && opts.silent)) { syncEditSliders(); drawOverlay(); }
}

function setEditTool(tool) {
  const was = editSession.tool;
  editSession.tool = tool;
  // Painting blind is never what you want, so picking up the brush turns the
  // tint on — visibly, by ticking the box, rather than behind its back.
  if (tool === "brush" && !editSession.showMask) {
    editSession.showMask = true;
    $("#mask-show").checked = true;
  }
  $$("#edit-modal .mask-add-btn[data-add-mask]").forEach((b) =>
    b.classList.toggle("armed", b.dataset.addMask === tool));
  $("#wb-pick").classList.toggle("armed", tool === "wb");
  $$("#edit-repair-group [data-repair-tool]").forEach((b) =>
    b.classList.toggle("armed", b.dataset.repairTool === tool));
  const wrap = $(".edit-canvas-wrap");
  wrap.classList.toggle("tool-place", tool === "radial" || tool === "linear"
    || tool === "redeye" || tool === "peteye");
  wrap.classList.toggle("tool-brush", tool === "brush" || isRepairBrush(tool));
  wrap.classList.toggle("tool-crop", tool === "crop");
  wrap.classList.toggle("tool-wb", tool === "wb");
  const text = tool === "radial" ? "Drag on the photo to place the ellipse"
    : tool === "linear" ? "Drag on the photo to set the gradient direction"
    : tool === "brush" ? "Paint over the area · Alt = erase · [ ] = brush size"
    : tool === "crop" ? "Drag the box or its corners · press Crop again when done"
    : tool === "wb" ? "Click something that should be grey — a white wall, a grey card, a white shirt"
    : tool === "spot" ? "Click a dust spot or blemish · [ ] = size"
    : tool === "heal" ? "Paint over what should go; it fills from around it · [ ] = size"
    : tool === "clone" ? (repairState.cloneSrc ? "Paint where the copy goes · Alt-click to pick a new source"
                                               : "Alt-click (or click) where to copy from")
    : tool === "redeye" ? "Drag from the centre of the pupil out to the edge of the iris"
    : tool === "peteye" ? "Drag from the centre of the eye out to the edge of the glow"
    : "";
  const hint = $("#edit-tool-hint");
  hint.textContent = text;
  hint.classList.toggle("hidden", !text);
  // Arming or dropping the crop tool changes whether the frame is shown cropped,
  // so the preview has to be asked for again.
  if (was === "crop" || tool === "crop") {
    syncCropControls();
    fetchEditPreview(true);
  }
  drawOverlay();
}

function addMask(kind, group) {
  $("#edit-mask-group").open = true;
  if (!editSession.relPath) return;
  if (editSession.edit.masks.length >= MASK_MAX) {
    $("#edit-status").textContent = `mask limit reached (${MASK_MAX})`;
    return;
  }
  const r = overlayRect();
  editSession.edit.masks.push(newMask(kind, r.h ? r.w / r.h : 1, group));
  selectMask(editSession.edit.masks.length - 1);
  // Radial/gradient masks land centred so they are visible right away; arming
  // the tool lets the very next drag re-place them where the user wants. An
  // automatic mask has nothing to drag, so no tool is armed for it.
  setEditTool((kind === "auto" || kind === "range") ? null : kind);
  setEditDirty();
  drawOverlay();
  if (kind !== "brush") fetchEditPreview(false);
}

// ---------- range refinement ----------

function buildMaskRangeFields() {
  const wrap = $("#mask-range-fields");
  if (!wrap || wrap.dataset.built) return;
  wrap.innerHTML = LUMA_RANGE_FIELDS.map((f) =>
    `<label class="look-row" title="${escapeHtml(f.hint)}">` +
    `<span class="look-name">${f.label}</span>` +
    `<input type="range" data-luma="${f.k}" min="${f.min}" max="${f.max}" step="1" />` +
    `<span class="look-val" data-luma-val="${f.k}"></span></label>`).join("");
  wrap.addEventListener("input", (e) => {
    const sl = e.target.closest("input[type=range][data-luma]");
    if (!sl) return;
    const m = activeMask();
    if (!m || !m.range_luma) return;
    m.range_luma[sl.dataset.luma] = parseInt(sl.value, 10);
    // Keep the window the right way round rather than letting it invert
    // silently, which would select nothing and look like a broken slider.
    if (m.range_luma.lo > m.range_luma.hi) {
      if (sl.dataset.luma === "lo") m.range_luma.hi = m.range_luma.lo;
      else m.range_luma.lo = m.range_luma.hi;
    }
    maskChanged(false);
  });
  wrap.dataset.built = "1";
}

function renderMaskRange(m) {
  const tools = $("#mask-range-tools");
  if (!tools) return;
  buildMaskRangeFields();
  // A range mask IS its range, so the toggle would be a way to delete the mask.
  const togglable = m.type !== "range";
  tools.classList.remove("hidden");
  $("#mask-range-on").checked = !!m.range_luma;
  $("#mask-range-on").disabled = !togglable;
  $("#mask-range-fields").classList.toggle("hidden", !m.range_luma);
  if (!m.range_luma) return;
  for (const f of LUMA_RANGE_FIELDS) {
    const v = Math.round(m.range_luma[f.k] ?? 0);
    $(`#mask-range-fields input[data-luma="${f.k}"]`).value = v;
    $(`#mask-range-fields [data-luma-val="${f.k}"]`).textContent = v;
  }
  // Say when the range is doing nothing rather than leaving it a mystery.
  $("#mask-range-note").classList.toggle("warn", lumaRangeIsAll(m.range_luma));
}

// ---------- automatic mask previews ----------
// The editor rasterizes a radial or a brush itself, from the same numbers the
// server has. It cannot do that for a segmentation, so the alpha is fetched as
// an image — white with the coverage in its alpha channel, which is exactly what
// the tint pipeline wants — and cached per photo and group.
const autoTints = new Map();

function autoTintImage(relPath, group) {
  const key = `${relPath}|${group}`;
  const hit = autoTints.get(key);
  if (hit !== undefined) return hit.complete && hit.naturalWidth ? hit : null;
  const img = new Image();
  autoTints.set(key, img);
  // Redraw once it lands: the first paint after adding a mask has nothing to
  // show, and without this it would stay blank until the next pointer move.
  img.addEventListener("load", () => drawOverlay());
  img.addEventListener("error", () => autoTints.delete(key));
  img.src = `/api/segment/preview?rel_path=${encodeURIComponent(relPath)}`
    + `&group=${encodeURIComponent(group)}`;
  return null;
}

// A range selection is measured on a fixed grid over the whole ungraded frame,
// so reproducing it here from a displayed preview would give a different answer
// than the one being graded. Fetch it instead, keyed on the parameters that
// decide it — the mask's identity does not, since two masks with the same range
// select the same pixels.
const rangeTints = new Map();

function rangeTintImage(relPath, mask) {
  const key = `${relPath}|${JSON.stringify([mask.range_luma, mask.range_color])}`;
  const hit = rangeTints.get(key);
  if (hit !== undefined) return hit.img.complete && hit.img.naturalWidth ? hit.img : null;
  const img = new Image();
  rangeTints.set(key, { img });
  img.addEventListener("load", () => drawOverlay());
  img.addEventListener("error", () => rangeTints.delete(key));
  // A bare range over the whole frame: the shape, if any, is painted separately
  // and the two are multiplied by the canvas, exactly as the server multiplies
  // them when it grades.
  const probe = { type: "range", enabled: true, invert: false, feather: 0,
                  amount: 100, adj: { exposure: 1 },
                  range_luma: mask.range_luma, range_color: mask.range_color };
  fetch("/api/mask/preview", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: relPath, mask: probe }),
  }).then((r) => (r.ok ? r.blob() : Promise.reject(r.status)))
    .then((b) => { img.src = URL.createObjectURL(b); })
    .catch(() => rangeTints.delete(key));
  return null;
}

async function ensureSegmentModel() {
  const res = await fetch("/api/segment/status", { cache: "no-store" });
  const info = await res.json();
  if (info.model_ready) return true;
  const mb = Math.round((info.size || 0) / 1e6);
  $("#edit-status").textContent = `fetching the segmentation model (${mb} MB)…`;
  const got = await fetch("/api/segment/download", { method: "POST" });
  const done = await got.json();
  $("#edit-status").textContent = done.model_ready ? "" : "model download failed";
  return !!done.model_ready;
}

function deleteMask(idx) {
  if (!editSession.edit.masks[idx]) return;
  editSession.edit.masks.splice(idx, 1);
  selectMask(Math.min(idx, editSession.edit.masks.length - 1));
  setEditDirty();
  fetchEditPreview(true);
}

function duplicateActiveMask() {
  const m = activeMask();
  if (!m || editSession.edit.masks.length >= MASK_MAX) return;
  const copy = cloneMask(m);
  copy.name = "";
  if (copy.type === "radial") { copy.cx += 0.04; copy.cy += 0.04; }
  editSession.edit.masks.splice(editSession.activeMask + 1, 0, copy);
  selectMask(editSession.activeMask + 1);
  setEditDirty();
  fetchEditPreview(true);
}

// The Masks section shows how many masks there are when folded. It opens by
// itself for a photo that has masks and when one is added, and otherwise stays
// as it was left: the sliders act on the whole photo when there are none.
function syncMaskGroup() {
  const n = (editSession.edit?.masks || []).length;
  $("#mask-summary-state").textContent = n ? `· ${n}` : "";
}

// Which editor sections were left open, kept across photos and sessions.
const EDIT_GROUPS_KEY = "pcls.editGroups";

function restoreEditGroups() {
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(EDIT_GROUPS_KEY) || "{}"); } catch { saved = {}; }
  $$("#edit-modal details.edit-group[id]").forEach((d) => {
    if (d.id in saved) d.open = saved[d.id];
    d.addEventListener("toggle", () => {
      let now = {};
      try { now = JSON.parse(localStorage.getItem(EDIT_GROUPS_KEY) || "{}"); } catch { now = {}; }
      now[d.id] = d.open;
      try { localStorage.setItem(EDIT_GROUPS_KEY, JSON.stringify(now)); } catch { /* private */ }
    });
  });
}

function renderMaskList() {
  syncMaskGroup();
  const list = $("#mask-list");
  if (!list) return;
  const masks = (editSession.edit && editSession.edit.masks) || [];
  const rows = [
    `<div class="mask-row${editSession.activeMask < 0 ? " active" : ""}" data-mask="-1">` +
    `<span class="mask-eye-spacer"></span>` +
    `<span class="mask-name">Global — whole photo</span></div>`,
  ];
  masks.forEach((m, i) => {
    const hint = maskAdjNeutral(m) ? "no effect yet"
      : (m.type === "brush" && !m.strokes.length) ? "nothing painted" : "";
    rows.push(
      `<div class="mask-row${i === editSession.activeMask ? " active" : ""}` +
      `${m.enabled ? "" : " off"}" data-mask="${i}">` +
      `<button class="mask-eye" data-mask-toggle="${i}" title="Show / hide this mask">` +
      `${icon(m.enabled ? "eyeOpen" : "eyeShut")}</button>` +
      `<span class="mask-icon">${icon(MASK_KINDS[m.type].icon)}</span>` +
      `<span class="mask-name">${escapeHtml(maskLabel(m, i))}</span>` +
      `<span class="mask-hint">${hint}</span>` +
      `<button class="mask-del" data-mask-del="${i}" title="Delete this mask">×</button></div>`
    );
  });
  list.innerHTML = rows.join("");
}

function renderMaskDetail() {
  const m = activeMask();
  const detail = $("#mask-detail");
  detail.classList.toggle("hidden", !m);
  $("#mask-context").textContent = m
    ? `Local — ${maskLabel(m, editSession.activeMask)}`
    : "Global adjustments";
  $("#mask-context").classList.toggle("local", !!m);
  // Auto-tone and the master curve stay global-only.
  $("#edit-auto").disabled = !!m;
  $("#wb-pick").disabled = !!m;   // it writes the global pair
  $("#edit-curve-group").classList.toggle("hidden", !!m);
  // "Reset mask" read as if it might delete the mask or undo its shape. Say
  // exactly what it zeroes.
  const reset = $("#edit-reset");
  reset.textContent = m ? "Reset sliders" : "Reset all";
  reset.dataset.tip = m
    ? "Zero this mask's adjustments — its shape and position stay"
    : "Clear every adjustment, mask and watermark on this photo";
  $$("#edit-modal .look-row[data-field]").forEach((row) => {
    row.classList.toggle("hidden", !!m && !MASK_LOCAL_KEYS.has(row.dataset.field));
  });
  if (!m) return;
  const nameEl = $("#mask-name");
  // Don't fight the user mid-word if the panel re-renders while typing.
  if (document.activeElement !== nameEl) nameEl.value = m.name || "";
  nameEl.placeholder = maskLabel({ ...m, name: "" }, editSession.activeMask);
  $("#mask-invert").textContent = m.invert ? "Affect: outside" : "Affect: inside";
  $("#mask-invert").classList.toggle("active", m.invert);
  $("#mask-feather").value = m.feather;
  $("#mask-feather-val").textContent = m.feather;
  $("#mask-amount").value = m.amount;
  $("#mask-amount-val").textContent = m.amount;
  renderMaskRange(m);
  const auto = m.type === "auto";
  $("#mask-auto-tools").classList.toggle("hidden", !auto);
  if (auto) {
    $("#mask-auto-groups").innerHTML = AUTO_GROUPS.map((g) =>
      `<button type="button" class="mask-auto-group${g.k === m.group ? " active" : ""}" ` +
      `data-auto-group="${g.k}">${g.label}</button>`).join("");
    $$("#mask-auto-groups .mask-auto-group").forEach((b) => {
      b.addEventListener("click", () => {
        if (b.dataset.autoGroup === m.group) return;
        m.group = b.dataset.autoGroup;
        maskChanged(true);
      });
    });
  }
  const brush = m.type === "brush";
  $("#mask-brush-tools").classList.toggle("hidden", !brush);
  if (brush) {
    $("#mask-brush-size").value = editSession.brush.size;
    $("#mask-brush-size-val").textContent = editSession.brush.size;
    $("#mask-brush-paint").classList.toggle("active", !editSession.brush.erase);
    $("#mask-brush-erase").classList.toggle("active", editSession.brush.erase);
    $("#mask-brush-undo").disabled = !m.strokes.length;
  }
}

// Any change to the selected mask's shape or strength: repaint and re-render.
function maskChanged(immediate) {
  renderMaskDetail();
  renderMaskList();
  drawOverlay();
  setEditDirty();
  fetchEditPreview(!!immediate);
}

// ---------- local adjustments: overlay canvas ----------
const OVERLAY_HIT = 11;              // px grab radius for a handle
const ZOOM_STEPS = [0, 0.5, 1, 2];   // 0 = fit; the rest are CSS px per image px
const MASK_TINT = "#ff4d4f";
const MIN_RADIUS = 0.005;
let tintCanvas = null;

const smoothstep = (t) => t * t * (3 - 2 * t);
const fx2px = (r, x) => r.x + x * r.w;
const fy2px = (r, y) => r.y + y * r.h;

// Masks are positioned against the *original* frame, because that is where they
// are graded — crop later and a mask stays on the thing it was drawn on. The
// overlay, though, sits on the cropped preview. maskRect answers "where would
// the whole original frame be, given that the crop occupies `r`", so every
// existing guide that measures against a rect keeps working unchanged.
//
// The server supplies the transform (see geometry_norm_matrix) rather than the
// arithmetic being repeated here. Its off-diagonal terms are the straighten;
// those are handled by rotating the canvas, since a rect cannot express a turn.
function maskRect(r) {
  const x = editSession.maskXform;
  if (!x) return r;
  return { x: r.x + r.w * x[2], y: r.y + r.h * x[5],
           w: r.w * x[0], h: r.h * x[4] };
}

// Degrees the guides must be turned by to sit on straightened content.
function maskTurn() {
  const x = editSession.maskXform;
  if (!x) return 0;
  return Math.atan2(x[3], x[0]) * 180 / Math.PI;
}

// Original-frame fraction -> the same point as a fraction of what is displayed.
function maskFwd(u, v) {
  const x = editSession.maskXform;
  if (!x) return { x: u, y: v };
  return { x: x[0] * u + x[1] * v + x[2], y: x[3] * u + x[4] * v + x[5] };
}

// ...and back, which is what a pointer position has to go through before it can
// be compared against a mask or used to drag one.
function maskInv(u, v) {
  const x = editSession.maskXform;
  if (!x) return { x: u, y: v };
  const det = x[0] * x[4] - x[1] * x[3];
  if (!det) return { x: u, y: v };
  const p = u - x[2], q = v - x[5];
  return { x: (x[4] * p - x[1] * q) / det, y: (x[0] * q - x[3] * p) / det };
}

// The straighten cannot be folded into a rect, so it goes on the canvas. Written
// relative to maskRect, which already carries the scale, this leaves a matrix
// with a unit diagonal: line widths and handle sizes come out unchanged, which a
// plain scaling transform would have wrecked.
function applyMaskXform(ctx, r, mr) {
  const x = editSession.maskXform;
  const p = 1, s = 1;
  const q = mr.h ? (r.w * x[1]) / mr.h : 0;
  const u = mr.w ? (r.h * x[3]) / mr.w : 0;
  const tx = (r.x + r.w * x[2]) - (p * mr.x + q * mr.y);
  const ty = (r.y + r.h * x[5]) - (u * mr.x + s * mr.y);
  ctx.transform(p, u, q, s, tx, ty);
}

const MASK_XFORM_ID = [1, 0, 0, 0, 1, 0];
function maskXformIsIdentity() {
  const x = editSession.maskXform;
  return !x || x.every((v, i) => Math.abs(v - MASK_XFORM_ID[i]) < 1e-9);
}

const isZoomed = () => editSession.view.zoom > 0;
const clamp01 = (v, span) => Math.min(Math.max(v, 0), Math.max(0, 1 - span));

// The viewport the preview is drawn into. It keeps the size the fit view gave
// it, so switching to 1:1 never makes the dialog jump.
function viewportSize() {
  const wrap = $(".edit-canvas-wrap");
  return { w: wrap.clientWidth || 1, h: wrap.clientHeight || 1 };
}

// Which window of the photo the 1:1 view is showing, in normalized coords.
function viewRoi() {
  const { w: vw, h: vh } = viewportSize();
  const z = editSession.view.zoom;
  const nw = editSession.frame.w || 1, nh = editSession.frame.h || 1;
  const rw = Math.min(1, vw / z / nw), rh = Math.min(1, vh / z / nh);
  return {
    x0: clamp01(editSession.view.cx - rw / 2, rw),
    y0: clamp01(editSession.view.cy - rh / 2, rh),
    rw, rh,
  };
}

// Where the *whole* photo would sit in viewport pixels at the current view.
// Mask geometry is normalized, so every overlay coordinate keeps working when
// zoomed — the parts outside the viewport simply fall off the canvas.
function overlayRect() {
  const img = $("#edit-img");
  if (!isZoomed()) {
    // The overlay canvas covers the whole viewport while the photo is centred
    // inside it, so measure the image against the canvas rather than assuming
    // the two share an origin.
    const ir = img.getBoundingClientRect();
    const wr = $(".edit-canvas-wrap").getBoundingClientRect();
    // Shape the letterbox from the photo's own dimensions, never from whatever
    // bitmap the element is holding. Assigning .src blanks naturalWidth until
    // the new frame decodes, and painting swaps in a draft render several times
    // a second: for those frames this measured 0, fell through to the whole
    // wrap below, and every coordinate mapped through it landed hundreds of
    // pixels sideways. That was the brush jumping mid-stroke.
    const nw = editSession.frame.w || img.naturalWidth;
    const nh = editSession.frame.h || img.naturalHeight;
    if (!nw || !nh || !ir.width || !ir.height) {
      return { x: 0, y: 0, w: wr.width || 1, h: wr.height || 1 };
    }
    const s = Math.min(ir.width / nw, ir.height / nh);
    return {
      x: ir.left - wr.left + (ir.width - nw * s) / 2,
      y: ir.top - wr.top + (ir.height - nh * s) / 2,
      w: nw * s, h: nh * s,
    };
  }
  const { w: vw, h: vh } = viewportSize();
  const z = editSession.view.zoom;
  const nw = editSession.frame.w || 1, nh = editSession.frame.h || 1;
  const roi = viewRoi();
  const fullW = nw * z, fullH = nh * z;
  // Centre the frame in the viewport on any axis it no longer fills.
  const padX = roi.rw >= 1 ? (vw - fullW) / 2 : 0;
  const padY = roi.rh >= 1 ? (vh - fullH) / 2 : 0;
  return { x: padX - roi.x0 * fullW, y: padY - roi.y0 * fullH, w: fullW, h: fullH };
}

// Size the returned crop into place. In fit mode CSS handles it; zoomed, the
// JPEG is exactly the viewport window so it is stretched to the drawn rect.
function layoutPreviewImage() {
  const img = $("#edit-img");
  if (!isZoomed()) {
    img.classList.remove("zoomed");
    img.style.left = img.style.top = img.style.width = img.style.height = "";
    return;
  }
  // No need to freeze the wrap's height any more: it is a flex child that
  // fills the dialog, so it keeps its size when the image goes absolute.
  const { w: vw, h: vh } = viewportSize();
  const roi = viewRoi();
  const r = overlayRect();
  const drawW = Math.min(vw, roi.rw * r.w), drawH = Math.min(vh, roi.rh * r.h);
  img.classList.add("zoomed");
  img.style.left = `${(vw - drawW) / 2}px`;
  img.style.top = `${(vh - drawH) / 2}px`;
  img.style.width = `${drawW}px`;
  img.style.height = `${drawH}px`;
}

function setZoom(z, focus) {
  // Until the matching crop arrives the element still holds the previous
  // frame, and the zoomed layout would stretch it to fill the viewport — a
  // visible squash for as long as the render takes. Keep it letterboxed until
  // the right pixels land.
  $("#edit-img").classList.add("view-pending");
  const prev = editSession.view.zoom;
  // Keep whatever the pointer is over pinned in place while zooming.
  if (focus && prev > 0) { editSession.view.cx = focus.x; editSession.view.cy = focus.y; }
  else if (focus) { editSession.view.cx = focus.x; editSession.view.cy = focus.y; }
  editSession.view.zoom = z;
  $$("#edit-modal .edit-zoom").forEach((b) =>
    b.classList.toggle("active", Math.abs(parseFloat(b.dataset.zoom) - z) < 1e-6));
  $("#edit-zoom-hint").classList.toggle("hidden", !isZoomed());
  layoutPreviewImage();
  resizeOverlay();
  fetchEditPreview(true);
  fetchOriginalPreview();
}

function panBy(dxFrac, dyFrac) {
  const roi = viewRoi();
  editSession.view.cx = clamp01(editSession.view.cx - dxFrac - roi.rw / 2, roi.rw) + roi.rw / 2;
  editSession.view.cy = clamp01(editSession.view.cy - dyFrac - roi.rh / 2, roi.rh) + roi.rh / 2;
  layoutPreviewImage();
  scheduleOverlay();
  previewDuringDrag();
}

function resizeOverlay() {
  const cv = $("#edit-overlay"), img = $("#edit-img");
  if (!cv || !img) return;
  // Cover the whole viewport, not just the image: when zoomed the photo can be
  // letterboxed inside it and the guides still need somewhere to draw.
  const { w, h } = viewportSize();
  if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  cv.style.width = `${w}px`;
  cv.style.height = `${h}px`;
  cv.width = Math.round(w * dpr);
  cv.height = Math.round(h * dpr);
  cv.getContext("2d").setTransform(dpr, 0, 0, dpr, 0, 0);
  drawOverlay();
}

// A pointer position as a fraction of the *displayed* frame. What the crop tool
// works in, since a crop box is measured against what you can see.
function evFracView(e) {
  const b = $("#edit-overlay").getBoundingClientRect(), r = overlayRect();
  return { x: (e.clientX - b.left - r.x) / r.w, y: (e.clientY - b.top - r.y) / r.h };
}

// ...and as a fraction of the original frame, which is the space masks live in.
function evFrac(e) {
  const v = evFracView(e);
  return maskInv(v.x, v.y);
}

// Every position a pointermove stands for, oldest first, as frame fractions.
// Usually that is just where the pointer is now; under load the browser packs
// the positions it skipped into the same event, and a brush has to honour them
// or it cuts the corner. `now` is the caller's already-computed evFrac(e), so
// the common one-position case costs nothing extra.
function pointerPath(e, now) {
  const packed = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
  return packed.length > 1 ? packed.map(evFrac) : [now];
}

// Unit-circle coordinates of a radial mask -> canvas pixels. Mirrors the
// server: rotate in normalized space, then scale to the displayed rect.
function radialPoint(m, r, ux, uy) {
  const a = m.angle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const px = ux * m.rx, py = uy * m.ry;
  return [fx2px(r, m.cx + px * ca - py * sa), fy2px(r, m.cy + px * sa + py * ca)];
}

// Normalized offset from the centre, expressed in the mask's own axes.
function radialLocal(m, f) {
  const a = m.angle * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
  const dx = f.x - m.cx, dy = f.y - m.cy;
  return { u: (dx * ca + dy * sa) / m.rx, v: (dy * ca - dx * sa) / m.ry };
}

// Painting fires pointermove far faster than the overlay can be redrawn, and a
// synchronous redraw per event saturated the main thread: events then arrived
// in bursts and the stroke appeared to leap sideways. Coalesce to one redraw
// per animation frame — the browser never has more work queued than it can do.
let overlayFrame = 0;
function scheduleOverlay() {
  if (overlayFrame) return;
  overlayFrame = requestAnimationFrame(() => {
    overlayFrame = 0;
    drawOverlay();
  });
}

function drawOverlay() {
  const cv = $("#edit-overlay");
  if (!cv || !cv.width) return;
  const ctx = cv.getContext("2d");
  const W = cv.clientWidth, H = cv.clientHeight;
  ctx.clearRect(0, 0, W, H);
  if ($("#edit-modal").classList.contains("hidden") || editSession.comparing) return;
  const r = overlayRect();
  // Cropping shows the frame uncropped, so mask guides drawn against it would
  // sit in the wrong place. One job at a time.
  if (editSession.tool === "crop") { drawCropOverlay(ctx, r, W, H); return; }
  const m = activeMask();

  // "show mask" with no mask selected used to do nothing at all, which read as
  // a broken checkbox. With Global selected it now tints every enabled mask, so
  // you can see the whole local-adjustment layout at a glance. Outlines go on
  // too: a red wash alone is invisible over a red car.
  // Guides are drawn against where the *original* frame would be, because that
  // is the space masks are stored and graded in.
  const mr = maskRect(r);
  const turned = !maskXformIsIdentity();
  if (repairsShown()) {
    if (turned) { ctx.save(); applyMaskXform(ctx, r, mr); }
    drawRepairs(ctx, mr);
    if (turned) ctx.restore();
  }
  if (turned) { ctx.save(); applyMaskXform(ctx, r, mr); }
  drawPortraitFaces(ctx, mr);
  if (turned) ctx.restore();
  if (editSession.showMask && !m) {
    for (const other of (editSession.edit.masks || [])) {
      if (!other.enabled) continue;
      drawMaskTint(ctx, other, r, W, H);
      if (turned) { ctx.save(); applyMaskXform(ctx, r, mr); }
      drawMaskOutline(ctx, other, mr);
      if (turned) ctx.restore();
    }
  }
  if (!m || !m.enabled) return;

  // The checkbox is the only thing that decides this. It used to be overridden
  // while dragging or brushing, which made the control look dead: the red was
  // there whether it was ticked or not. Arming the brush ticks it instead (see
  // setEditTool), so what you see always matches what the box says.
  const painting = m.type === "brush" && editSession.tool === "brush";
  if (editSession.showMask) {
    const stroking = !!editSession.drag && editSession.drag.kind === "paint";
    drawMaskTint(ctx, m, r, W, H, painting ? 0.28 : 0.34, stroking);
  }
  if (turned) { ctx.save(); applyMaskXform(ctx, r, mr); }
  drawMaskHandles(ctx, m, mr);
  // The wrap hides the system cursor for the brush, so this ring *is* the
  // cursor — it has to stay up mid-stroke, which is exactly when you need it.
  if (painting && editSession.hover) drawBrushCursor(ctx, mr);
  if (turned) ctx.restore();
}

// ---------- crop & straighten ----------
// The crop box lives in normalized coordinates of the *straightened* frame,
// which is exactly what the canvas shows while the tool is armed (the preview is
// asked for with skip_crop, so the box can be dragged over everything the tilt
// left available). No conversion, and nudging the tilt afterwards keeps the box
// where it was rather than sliding it around.

const CROP_ASPECTS = {
  "1:1": 1, "4:5": 4 / 5, "5:4": 5 / 4,
  "2:3": 2 / 3, "3:2": 3 / 2, "16:9": 16 / 9,
};
const CROP_MIN = 0.02;          // matches editing.MIN_CROP on the server
const CROP_HANDLE = 11;         // grab radius in CSS px

function cropRect() {
  return editSession.edit.crop || { x: 0, y: 0, w: 1, h: 1 };
}

// Aspect of the straightened, uncropped frame — the space a crop box lives in.
// Deliberately not the displayed frame: with the tool off that is already
// cropped, and every ratio computed from it would be wrong by the crop.
function frameAspect() {
  const f = editSession.cropFrame;
  return (f && f.w && f.h) ? f.w / f.h : 1;
}

// Which aspect the buttons should offer. "orig" follows the photo, so it stays
// right for a portrait and a landscape alike.
function cropTargetAspect() {
  const a = editSession.cropAspect;
  if (a === "free") return null;
  if (a === "orig") return frameAspect();
  return CROP_ASPECTS[a] || null;
}

// Largest box of `ratio` (w/h in *frame* terms) centred on what is there now.
function cropForAspect(ratio) {
  if (ratio === null) return null;
  const fa = frameAspect();
  // A ratio is about the picture; the box is in fractions of the frame, so it
  // has to be divided through by the frame's own aspect.
  let w = 1, h = fa / ratio;
  if (h > 1) { h = 1; w = ratio / fa; }
  const c = cropRect();
  const cx = c.x + c.w / 2, cy = c.y + c.h / 2;
  return {
    x: Math.min(1 - w, Math.max(0, cx - w / 2)),
    y: Math.min(1 - h, Math.max(0, cy - h / 2)),
    w, h,
  };
}

function setCrop(box, opts) {
  const full = !box || (box.w >= 1 - 1e-4 && box.h >= 1 - 1e-4);
  editSession.edit.crop = full ? null : {
    x: +box.x.toFixed(5), y: +box.y.toFixed(5),
    w: +box.w.toFixed(5), h: +box.h.toFixed(5),
  };
  syncCropControls();
  setEditDirty();
  if (!opts || !opts.silent) fetchEditPreview(true, !!(opts && opts.draft));
}

function syncCropControls() {
  const c = editSession.edit.crop;
  $("#crop-tool").classList.toggle("armed", editSession.tool === "crop");
  $("#crop-reset").disabled = !c && Math.abs(editSession.edit.tilt || 0) < 1e-4;
  $("#crop-tilt").value = editSession.edit.tilt || 0;
  $("#crop-tilt-val").textContent = `${(editSession.edit.tilt || 0).toFixed(1)}°`;
  $$("#crop-aspects button").forEach((b) =>
    b.classList.toggle("active", b.dataset.aspect === editSession.cropAspect));
  // Pixel size of what will come out, so a crop can be judged against what it
  // is for rather than as a fraction.
  const f = editSession.cropFrame;
  const out = $("#crop-size");
  if (f && f.w) {
    const box = cropRect();
    const w = Math.round(f.w * box.w), h = Math.round(f.h * box.h);
    out.textContent = c ? `${w} × ${h} px` : `${f.w} × ${f.h} px · full frame`;
  } else {
    out.textContent = "";
  }
}

// Corner and edge grips, in canvas px, keyed by what they move.
function cropHandles(r) {
  const c = cropRect();
  const x0 = fx2px(r, c.x), x1 = fx2px(r, c.x + c.w);
  const y0 = fy2px(r, c.y), y1 = fy2px(r, c.y + c.h);
  const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
  return {
    nw: [x0, y0], ne: [x1, y0], sw: [x0, y1], se: [x1, y1],
    n: [mx, y0], s: [mx, y1], w: [x0, my], e: [x1, my],
  };
}

const CROP_CURSORS = {
  nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize", w: "ew-resize", e: "ew-resize",
  move: "move",
};

function cropHit(r, f) {
  const px = fx2px(r, f.x), py = fy2px(r, f.y);
  const hs = cropHandles(r);
  let best = null, bestD = CROP_HANDLE;
  for (const [k, [hx, hy]] of Object.entries(hs)) {
    const d = Math.hypot(px - hx, py - hy);
    if (d <= bestD) { best = k; bestD = d; }
  }
  if (best) return best;
  const c = cropRect();
  const inside = f.x > c.x && f.x < c.x + c.w && f.y > c.y && f.y < c.y + c.h;
  return inside ? "move" : null;
}

function drawCropOverlay(ctx, r, W, H) {
  const c = cropRect();
  const x0 = fx2px(r, c.x), y0 = fy2px(r, c.y);
  const x1 = fx2px(r, c.x + c.w), y1 = fy2px(r, c.y + c.h);
  ctx.save();
  // Everything outside goes dark, so the eye reads the crop and not the frame.
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.rect(0, 0, W, H);
  ctx.rect(x0, y0, x1 - x0, y1 - y0);
  ctx.fill("evenodd");

  // Thirds, the reason anyone reaches for a crop tool.
  ctx.strokeStyle = "rgba(255,255,255,0.32)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 1; i < 3; i++) {
    const gx = x0 + (x1 - x0) * i / 3, gy = y0 + (y1 - y0) * i / 3;
    ctx.moveTo(gx, y0); ctx.lineTo(gx, y1);
    ctx.moveTo(x0, gy); ctx.lineTo(x1, gy);
  }
  ctx.stroke();

  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);

  // Corner brackets read as grabbable in a way a plain square does not.
  const L = Math.min(26, (x1 - x0) / 3, (y1 - y0) / 3);
  ctx.lineWidth = 3.5;
  ctx.beginPath();
  for (const [cx, cy, sx, sy] of [[x0, y0, 1, 1], [x1, y0, -1, 1],
                                  [x0, y1, 1, -1], [x1, y1, -1, -1]]) {
    ctx.moveTo(cx, cy + sy * L); ctx.lineTo(cx, cy); ctx.lineTo(cx + sx * L, cy);
  }
  ctx.stroke();
  // Edge grips.
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  for (const k of ["n", "s", "w", "e"]) {
    const [hx, hy] = cropHandles(r)[k];
    ctx.beginPath();
    ctx.arc(hx, hy, 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

// Resize from a handle, honouring the locked aspect and never inverting.
function cropResize(kind, orig, f, ratio) {
  let { x, y, w, h } = orig;
  const right = orig.x + orig.w, bottom = orig.y + orig.h;
  if (kind.includes("w")) { x = Math.min(f.x, right - CROP_MIN); w = right - x; }
  if (kind.includes("e")) { w = Math.max(CROP_MIN, Math.min(1, f.x) - x); }
  if (kind.includes("n")) { y = Math.min(f.y, bottom - CROP_MIN); h = bottom - y; }
  if (kind.includes("s")) { h = Math.max(CROP_MIN, Math.min(1, f.y) - y); }
  if (ratio !== null) {
    const fa = frameAspect();
    // Drive the free axis from the one being dragged, so a locked crop follows
    // the pointer on the edge under it instead of fighting it.
    if (kind === "n" || kind === "s") { w = Math.min(1, h * ratio / fa); }
    else if (kind === "w" || kind === "e") { h = Math.min(1, w * fa / ratio); }
    else { h = Math.min(1, w * fa / ratio); }
    if (kind.includes("w")) x = right - w;
    if (kind.includes("n")) y = bottom - h;
  }
  x = Math.min(1 - w, Math.max(0, x));
  y = Math.min(1 - h, Math.max(0, y));
  return { x, y, w: Math.min(w, 1 - x), h: Math.min(h, 1 - y) };
}

function drawBrushCursor(ctx, r) {
  const x = fx2px(r, editSession.hover.x), y = fy2px(r, editSession.hover.y);
  const rad = editSession.brush.size / 1000 * r.w;
  ctx.save();
  // Two rings, dark under light, so the cursor reads on a white car and a
  // black tyre alike.
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.arc(x, y, rad, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1.4;
  ctx.strokeStyle = editSession.brush.erase ? "#ffd166" : "#ffffff";
  ctx.setLineDash(editSession.brush.erase ? [5, 4] : []);
  ctx.beginPath();
  ctx.arc(x, y, rad, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  // A centre dot keeps the aim obvious when the brush is large.
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath();
  ctx.arc(x, y, 1.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// Translucent red wash over the affected area — the same falloff the server
// applies, approximated with canvas gradients.
// A hard edge around the affected area, so the mask is readable whatever the
// photo underneath is doing. For a brush this traces the painted strokes.
function drawMaskOutline(ctx, m, r) {
  // Neither a segmentation nor a tonal selection has an outline to trace.
  if (m.type === "auto" || m.type === "range") return;
  ctx.save();
  ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = "rgba(255,255,255,0.85)";
  ctx.shadowColor = "rgba(0,0,0,0.8)";
  ctx.shadowBlur = 2;
  if (m.type === "radial") {
    ctx.save();
    ctx.translate(fx2px(r, m.cx), fy2px(r, m.cy));
    ctx.scale(r.w, r.h);
    ctx.rotate(m.angle * Math.PI / 180);
    ctx.scale(m.rx, m.ry);
    ctx.beginPath();
    ctx.arc(0, 0, 1, 0, Math.PI * 2);
    ctx.restore();
    ctx.stroke();
  } else if (m.type === "linear") {
    const p1 = [fx2px(r, m.x1), fy2px(r, m.y1)], p2 = [fx2px(r, m.x2), fy2px(r, m.y2)];
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len, span = r.w + r.h;
    for (const p of [p1, p2]) {
      ctx.beginPath();
      ctx.moveTo(p[0] - nx * span, p[1] - ny * span);
      ctx.lineTo(p[0] + nx * span, p[1] + ny * span);
      ctx.stroke();
    }
  }
  // No brush case on purpose. A brush has no boundary to trace cheaply, and
  // re-drawing its strokes here meant the painted area was rendered twice —
  // once as the red tint and once in white on top of it. Worse, that second
  // pass replayed only the paint strokes, so erasing removed the red but left
  // a white ghost behind. The tint already shows the painted area, honouring
  // erase, feather and amount, so it is the single source of truth.
  ctx.setLineDash([]);
  ctx.restore();
}

function drawMaskTint(ctx, m, r, W, H, alpha = 0.34, cheap = false) {
  if (!tintCanvas) tintCanvas = document.createElement("canvas");
  // Built in CSS pixels, not device pixels. On a Retina display that is four
  // times less area to rasterize and — the part that actually hurt — four
  // times less for the feather's blur filter to chew through. It is a soft
  // wash sitting under the photo; nobody can tell it is half resolution.
  tintCanvas.width = Math.max(1, Math.round(W));
  tintCanvas.height = Math.max(1, Math.round(H));
  const o = tintCanvas.getContext("2d");
  o.setTransform(1, 0, 0, 1, 0, 0);
  o.clearRect(0, 0, W, H);
  if (m.invert) {
    o.fillStyle = "#fff";
    // The display rect, not the original frame's: inverted means everything the
    // mask does not cover, and all of that is what you can see.
    o.fillRect(r.x, r.y, r.w, r.h);
    o.globalCompositeOperation = "destination-out";
  }
  const mr = maskRect(r);
  const turned = !maskXformIsIdentity();
  if (turned) { o.save(); applyMaskXform(o, r, mr); }
  paintMaskShape(o, m, mr, m.invert, cheap);
  if (turned) o.restore();
  o.globalCompositeOperation = "source-in";
  o.fillStyle = MASK_TINT;
  o.fillRect(0, 0, W, H);
  ctx.save();
  // Light enough to judge the photo through it — lighter still while painting,
  // where you need to see what you are covering.
  ctx.globalAlpha = alpha * (m.amount / 100);
  ctx.drawImage(tintCanvas, 0, 0, W, H);
  ctx.restore();
}

function paintMaskShape(o, m, r, inverted, cheap = false) {
  // `cheap` drops the feather while a stroke is in progress: blurring every
  // stroke on every frame was the bulk of the redraw, and the soft edge is
  // only an indicator — the rendered photo shows the real falloff, and the
  // feathered tint comes back the moment the pointer lifts.
  const f = cheap ? 0 : Math.max(m.feather / 100, 0.001);
  if (m.type === "radial") {
    o.save();
    o.translate(fx2px(r, m.cx), fy2px(r, m.cy));
    o.scale(r.w, r.h);
    o.rotate(m.angle * Math.PI / 180);
    o.scale(m.rx, m.ry);
    const g = o.createRadialGradient(0, 0, Math.max(0, 1 - f), 0, 0, 1);
    for (let i = 0; i <= 4; i++) {
      const u = i / 4;
      g.addColorStop(u, `rgba(255,255,255,${smoothstep(1 - u).toFixed(3)})`);
    }
    o.fillStyle = g;
    o.beginPath();
    o.arc(0, 0, 1, 0, Math.PI * 2);
    o.fill();
    o.restore();
  } else if (m.type === "linear") {
    const g = o.createLinearGradient(fx2px(r, m.x1), fy2px(r, m.y1),
                                     fx2px(r, m.x2), fy2px(r, m.y2));
    for (let i = 0; i <= 8; i++) {
      const t = i / 8;
      const a = 1 - smoothstep(Math.min(1, Math.max(0, (t - 0.5) / f + 0.5)));
      g.addColorStop(t, `rgba(255,255,255,${a.toFixed(3)})`);
    }
    o.fillStyle = g;
    o.fillRect(r.x, r.y, r.w, r.h);
  } else if (m.type === "range") {
    const img = rangeTintImage(editSession.relPath, m);
    if (img) {
      o.save();
      if (inverted) o.globalCompositeOperation = "destination-out";
      o.drawImage(img, r.x, r.y, r.w, r.h);
      o.restore();
    }
  } else if (m.type === "auto") {
    // The alpha arrives as an image because a segmentation cannot be redrawn
    // from a handful of numbers. Until it lands there is nothing to paint; the
    // load handler calls drawOverlay again.
    const img = autoTintImage(editSession.relPath, m.group);
    if (img) {
      o.save();
      if (inverted) o.globalCompositeOperation = "destination-out";
      o.filter = f > 0 ? `blur(${Math.max(0.5, f * r.w * 0.02).toFixed(2)}px)` : "none";
      o.drawImage(img, r.x, r.y, r.w, r.h);
      o.filter = "none";
      o.restore();
    }
  } else {
    o.save();
    o.lineCap = "round";
    o.lineJoin = "round";
    o.strokeStyle = "#fff";
    o.fillStyle = "#fff";
    for (const s of m.strokes) {
      const rpx = Math.max(1, s.radius * r.w);
      // Inverted masks start from a filled canvas, so paint/erase swap roles.
      o.globalCompositeOperation =
        (!!s.erase !== !!inverted) ? "destination-out" : "source-over";
      o.filter = f > 0 ? `blur(${Math.max(0.5, f * rpx * 0.8).toFixed(2)}px)` : "none";
      o.beginPath();
      if (s.points.length === 1) {
        // One click = one disc of the brush radius (see drawMaskOutline).
        o.arc(fx2px(r, s.points[0][0]), fy2px(r, s.points[0][1]), rpx, 0, Math.PI * 2);
        o.fill();
      } else {
        o.lineWidth = 2 * rpx;
        s.points.forEach((p, i) => {
          const x = fx2px(r, p[0]), y = fy2px(r, p[1]);
          if (i) o.lineTo(x, y); else o.moveTo(x, y);
        });
        o.stroke();
      }
    }
    o.filter = "none";
    o.restore();
  }
  // A drawn shape carrying a range refinement: the tint is the shape narrowed to
  // the selection, which is what the server grades. destination-in multiplies
  // the alphas, so this is the same product, not an approximation of it.
  if (m.type !== "range" && (m.range_luma || m.range_color)) {
    const img = rangeTintImage(editSession.relPath, m);
    if (img) {
      o.save();
      o.globalCompositeOperation = "destination-in";
      o.drawImage(img, r.x, r.y, r.w, r.h);
      o.restore();
    }
  }
}

function strokeHandle(ctx, p, kind) {
  ctx.beginPath();
  ctx.arc(p[0], p[1], kind === "small" ? 3.5 : 5, 0, Math.PI * 2);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(0,0,0,0.6)";
  ctx.stroke();
}

function drawMaskHandles(ctx, m, r) {
  if (m.type === "brush" || m.type === "auto" || m.type === "range") return;
  ctx.save();
  ctx.shadowColor = "rgba(0,0,0,0.75)";
  ctx.shadowBlur = 3;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 1.5;
  if (m.type === "radial") {
    const ellipse = (scale, dash) => {
      ctx.save();
      ctx.translate(fx2px(r, m.cx), fy2px(r, m.cy));
      ctx.scale(r.w, r.h);
      ctx.rotate(m.angle * Math.PI / 180);
      ctx.scale(m.rx, m.ry);
      ctx.beginPath();
      ctx.arc(0, 0, scale, 0, Math.PI * 2);
      ctx.restore();          // path is already in device space; keep 1.5px stroke
      ctx.setLineDash(dash);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    ellipse(1, []);
    const inner = 1 - Math.max(m.feather / 100, 0);
    if (inner > 0.02) ellipse(inner, [4, 4]);
    const rot = radialPoint(m, r, 0, -1.3);
    const top = radialPoint(m, r, 0, -1);
    ctx.beginPath();
    ctx.moveTo(top[0], top[1]);
    ctx.lineTo(rot[0], rot[1]);
    ctx.stroke();
    ctx.shadowBlur = 0;
    for (const [ux, uy] of [[1, 0], [-1, 0], [0, 1], [0, -1]])
      strokeHandle(ctx, radialPoint(m, r, ux, uy), "small");
    strokeHandle(ctx, rot);
    strokeHandle(ctx, radialPoint(m, r, 0, 0));
  } else {
    const p1 = [fx2px(r, m.x1), fy2px(r, m.y1)], p2 = [fx2px(r, m.x2), fy2px(r, m.y2)];
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1];
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len, span = r.w + r.h;
    const edge = (p, dash) => {
      ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(p[0] - nx * span, p[1] - ny * span);
      ctx.lineTo(p[0] + nx * span, p[1] + ny * span);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    edge(p1, []);
    edge(p2, [4, 4]);
    ctx.beginPath();
    ctx.moveTo(p1[0], p1[1]);
    ctx.lineTo(p2[0], p2[1]);
    ctx.stroke();
    ctx.shadowBlur = 0;
    strokeHandle(ctx, p1);
    strokeHandle(ctx, p2);
  }
  ctx.restore();
}

// ---------- local adjustments: pointer interaction ----------
function overlayHit(m, r, f) {
  const px = fx2px(r, f.x), py = fy2px(r, f.y);
  const near = (p) => Math.hypot(px - p[0], py - p[1]) <= OVERLAY_HIT;
  if (m.type === "radial") {
    if (near(radialPoint(m, r, 0, -1.3))) return "rotate";
    if (near(radialPoint(m, r, 1, 0)) || near(radialPoint(m, r, -1, 0))) return "rx";
    if (near(radialPoint(m, r, 0, 1)) || near(radialPoint(m, r, 0, -1))) return "ry";
    const { u, v } = radialLocal(m, f);
    if (Math.hypot(u, v) <= 1) return "move";
  } else if (m.type === "linear") {
    const p1 = [fx2px(r, m.x1), fy2px(r, m.y1)], p2 = [fx2px(r, m.x2), fy2px(r, m.y2)];
    if (near(p1)) return "p1";
    if (near(p2)) return "p2";
    // Anywhere on the axis between the handles drags the whole gradient.
    const dx = p2[0] - p1[0], dy = p2[1] - p1[1], l2 = dx * dx + dy * dy;
    const t = l2 ? Math.min(1, Math.max(0, ((px - p1[0]) * dx + (py - p1[1]) * dy) / l2)) : 0;
    if (Math.hypot(px - (p1[0] + t * dx), py - (p1[1] + t * dy)) <= OVERLAY_HIT) return "line";
  }
  return null;
}

const CURSORS = { move: "move", line: "move", rotate: "grab", rx: "ew-resize",
                  ry: "ns-resize", p1: "grab", p2: "grab" };

function overlayDown(e) {
  if (!editSession.relPath) return;
  const panButton = e.button === 1 || editSession.spaceHeld;   // middle / space-drag
  if (e.button !== 0 && !panButton) return;
  const m = activeMask();
  const r0 = overlayRect(), f0 = evFrac(e);
  if (editSession.tool === "wb" && !panButton) {
    // One click, no drag: sample and disarm. The coordinates are already
    // fractions of the original frame, which is what the server samples.
    pickNeutral(f0);
    e.preventDefault();
    return;
  }
  if (isRepairTool(editSession.tool) && !panButton) {
    repairDown(e, f0);
    return;
  }
  if (editSession.tool === "crop" && !panButton) {
    // A crop box is measured against what is on screen, not against the original
    // frame the masks use, so this branch has its own coordinates.
    const fv = evFracView(e);
    // Outside the box starts a fresh one, which is what dragging on a photo
    // means everywhere else. Inside, the grips and the box itself take over —
    // except before there is a crop at all, when the box *is* the whole frame
    // and "move" would be a no-op drag over the entire photo. Then a drag
    // anywhere draws a new box; the corner grips still resize the frame edges.
    let hit = cropHit(r0, fv);
    if (hit === "move" && !editSession.edit.crop) hit = null;
    editSession.drag = hit
      ? { kind: "crop-" + hit, start: fv, orig: { ...cropRect() } }
      : { kind: "crop-new", start: fv };
    if (!hit) setCrop({ x: fv.x, y: fv.y, w: CROP_MIN, h: CROP_MIN }, { silent: true });
    $("#edit-overlay").setPointerCapture(e.pointerId);
    e.preventDefault();
    drawOverlay();
    return;
  }
  // Zoomed with nothing to grab? Then the drag pans the view. Space or the
  // middle button force a pan even when a mask is sitting under the cursor.
  const wantsMask = !panButton && m && m.enabled &&
    (editSession.tool === m.type || (m.type !== "brush" && overlayHit(m, r0, f0)));
  if (isZoomed() && !wantsMask) {
    editSession.drag = { kind: "pan", last: f0 };
    $("#edit-overlay").setPointerCapture(e.pointerId);
    $("#edit-overlay").style.cursor = "grabbing";
    e.preventDefault();
    return;
  }
  if (!m || !m.enabled) return;
  const r = r0, f = f0;
  const armed = editSession.tool === m.type ? m.type : null;
  let drag = null;
  // An armed tool owns the drag: a brand-new mask sits in the middle of the
  // frame, so its own handles must not steal the drag that places it.
  if (armed === "radial") {
    Object.assign(m, { cx: f.x, cy: f.y, rx: MIN_RADIUS, ry: MIN_RADIUS, angle: 0 });
    drag = { kind: "place-radial", start: f };
  } else if (armed === "linear") {
    Object.assign(m, { x1: f.x, y1: f.y, x2: f.x, y2: f.y });
    drag = { kind: "place-linear", start: f };
  } else if (armed === "brush") {
    const stroke = {
      radius: editSession.brush.size / 1000,
      erase: editSession.brush.erase !== e.altKey,   // Alt inverts the mode
      points: [[f.x, f.y]],
    };
    m.strokes.push(stroke);
    drag = { kind: "paint", stroke };
  } else if (m.type !== "brush") {
    const hit = overlayHit(m, r, f);
    if (hit) drag = { kind: hit, start: f, orig: cloneMask(m) };
  }
  if (!drag) return;
  editSession.drag = drag;
  $("#edit-overlay").setPointerCapture(e.pointerId);
  e.preventDefault();
  drawOverlay();
}

function overlayMove(e) {
  if (!editSession.relPath) return;
  const r = overlayRect(), f = evFrac(e);
  editSession.hover = f;
  const d = editSession.drag, m = activeMask();
  if (editSession.tool === "crop") {
    const cv = $("#edit-overlay");
    const f = evFracView(e);          // view space, as in overlayDown
    if (!d) {
      const hit = cropHit(r, f);
      cv.style.cursor = hit ? CROP_CURSORS[hit] : "crosshair";
      return;
    }
    const ratio = cropTargetAspect();
    if (d.kind === "crop-move") {
      const c = d.orig;
      setCrop({
        x: Math.min(1 - c.w, Math.max(0, c.x + (f.x - d.start.x))),
        y: Math.min(1 - c.h, Math.max(0, c.y + (f.y - d.start.y))),
        w: c.w, h: c.h,
      }, { silent: true, draft: true });
    } else if (d.kind === "crop-new") {
      const x = Math.min(d.start.x, f.x), y = Math.min(d.start.y, f.y);
      let w = Math.max(CROP_MIN, Math.abs(f.x - d.start.x));
      let h = Math.max(CROP_MIN, Math.abs(f.y - d.start.y));
      if (ratio !== null) h = Math.min(1 - y, w * frameAspect() / ratio);
      setCrop({ x, y, w: Math.min(w, 1 - x), h: Math.min(h, 1 - y) },
              { silent: true, draft: true });
    } else {
      setCrop(cropResize(d.kind.slice(5), d.orig, f, ratio),
              { silent: true, draft: true });
    }
    scheduleOverlay();
    return;
  }
  if (!d) {
    const cv = $("#edit-overlay");
    if ((m && m.type === "brush" && editSession.tool === "brush") || isRepairBrush(editSession.tool)) {
      cv.style.cursor = "none";
      scheduleOverlay();           // the brush ring follows the pointer
    } else if (editSession.tool) {
      cv.style.cursor = "crosshair";
    } else {
      const hit = m && m.enabled ? overlayHit(m, r, f) : null;
      cv.style.cursor = hit ? CURSORS[hit] : (isZoomed() ? "grab" : "default");
    }
    return;
  }
  if (d.kind === "pan") {
    panBy(f.x - d.last.x, f.y - d.last.y);
    return;   // `last` stays put: the delta is measured against the grab point
  }
  if (d.kind === "repair-paint" || d.kind === "eye") {
    repairMove(e, f, d);
    return;
  }
  if (!m) return;
  const aspect = r.h ? r.w / r.h : 1;
  if (d.kind === "move") {
    m.cx = d.orig.cx + (f.x - d.start.x);
    m.cy = d.orig.cy + (f.y - d.start.y);
  } else if (d.kind === "rx" || d.kind === "ry") {
    const { u, v } = radialLocal(d.orig, f);
    if (d.kind === "rx") m.rx = Math.max(MIN_RADIUS, Math.abs(u) * d.orig.rx);
    else m.ry = Math.max(MIN_RADIUS, Math.abs(v) * d.orig.ry);
    if (e.shiftKey) {  // keep the on-screen shape circular
      if (d.kind === "rx") m.ry = m.rx * aspect; else m.rx = m.ry / aspect;
    }
  } else if (d.kind === "rotate") {
    let deg = Math.atan2(f.y - m.cy, f.x - m.cx) * 180 / Math.PI + 90;
    if (e.shiftKey) deg = Math.round(deg / 15) * 15;
    m.angle = ((deg + 180) % 360 + 360) % 360 - 180;
  } else if (d.kind === "p1" || d.kind === "p2") {
    const other = d.kind === "p1" ? [m.x2, m.y2] : [m.x1, m.y1];
    let [x, y] = [f.x, f.y];
    if (e.shiftKey) {
      if (Math.abs(x - other[0]) > Math.abs(y - other[1])) y = other[1]; else x = other[0];
    }
    if (d.kind === "p1") { m.x1 = x; m.y1 = y; } else { m.x2 = x; m.y2 = y; }
  } else if (d.kind === "line") {
    const dx = f.x - d.start.x, dy = f.y - d.start.y;
    m.x1 = d.orig.x1 + dx; m.y1 = d.orig.y1 + dy;
    m.x2 = d.orig.x2 + dx; m.y2 = d.orig.y2 + dy;
  } else if (d.kind === "place-radial") {
    m.rx = Math.max(MIN_RADIUS, Math.abs(f.x - d.start.x));
    m.ry = e.shiftKey ? m.rx * aspect : Math.max(MIN_RADIUS, Math.abs(f.y - d.start.y));
  } else if (d.kind === "place-linear") {
    let [x, y] = [f.x, f.y];
    if (e.shiftKey) {
      if (Math.abs(x - m.x1) > Math.abs(y - m.y1)) y = m.y1; else x = m.x1;
    }
    m.x2 = x; m.y2 = y;
  } else if (d.kind === "paint") {
    // A pointermove can stand for several positions: when a frame runs long the
    // browser reports only the newest and folds the rest into it. Painting just
    // that one skips everything the pointer crossed in between, which is why a
    // fast stroke over a slow render used to jump sideways instead of following
    // the hand. getCoalescedEvents() hands back the positions that were folded
    // in, in order, so the stroke is drawn through all of them.
    const pts = d.stroke.points;
    // Decimate: one point per ~a third of the brush radius keeps strokes small.
    const step = Math.max(0.002, d.stroke.radius * 0.33);
    for (const q of pointerPath(e, f)) {
      const last = pts[pts.length - 1];
      if (Math.hypot(q.x - last[0], q.y - last[1]) >= step) pts.push([q.x, q.y]);
    }
  }
  scheduleOverlay();
  setEditDirty();
  previewDuringDrag();
}

// Keeping up with a drag: fire cheap draft renders as it moves, then one
// full-resolution render once it stops. Without the drafts the debounce meant
// nothing repainted at all until the pointer settled.
let lastDraftAt = 0;
let settleTimer = null;
function previewDuringDrag() {
  const now = Date.now();
  if (now - lastDraftAt > 110) {
    lastDraftAt = now;
    fetchEditPreview(true, true);
  }
  if (settleTimer) clearTimeout(settleTimer);
  settleTimer = setTimeout(() => { settleTimer = null; fetchEditPreview(true, false); }, 240);
}

// The drag is over: the full render now, not when the pause timer fires.
function settleDrag() {
  if (settleTimer) { clearTimeout(settleTimer); settleTimer = null; }
  fetchEditPreview(true, false);
}

function overlayUp(e) {
  const d = editSession.drag;
  if (!d) return;
  editSession.drag = null;
  if (d.kind.startsWith("crop-")) {
    if (e && e.pointerId != null) {
      try { $("#edit-overlay").releasePointerCapture(e.pointerId); } catch { /* gone */ }
    }
    // A tap with no drag is a miss, not a request for a 2%-of-the-frame crop.
    if (d.kind === "crop-new" && cropRect().w <= CROP_MIN * 1.5
        && cropRect().h <= CROP_MIN * 1.5) {
      setCrop(null);
    } else {
      syncCropControls();
      fetchEditPreview(true);
    }
    drawOverlay();
    return;
  }
  if (d.kind === "pan") {
    $("#edit-overlay").style.cursor = "grab";
    if (e && e.pointerId != null) {
      try { $("#edit-overlay").releasePointerCapture(e.pointerId); } catch { /* gone */ }
    }
    fetchEditPreview(true);
    fetchOriginalPreview();
    return;
  }
  if (d.kind === "repair-paint" || d.kind === "eye") {
    if (e && e.pointerId != null) {
      try { $("#edit-overlay").releasePointerCapture(e.pointerId); } catch { /* gone */ }
    }
    repairUp(d);
    return;
  }
  const m = activeMask();
  if (!m) return;
  const r = overlayRect(), aspect = r.h ? r.w / r.h : 1;
  // A click without a drag still deserves a usable shape.
  if (d.kind === "place-radial" && m.rx <= 0.02 && m.ry <= 0.02) {
    m.rx = 0.22; m.ry = 0.22 * aspect;
  }
  if (d.kind === "place-linear" && Math.hypot(m.x2 - m.x1, m.y2 - m.y1) < 0.02) {
    m.x2 = m.x1; m.y2 = m.y1 + 0.35;
  }
  if (d.kind === "place-radial" || d.kind === "place-linear") setEditTool(null);
  if (e && e.pointerId != null) {
    try { $("#edit-overlay").releasePointerCapture(e.pointerId); } catch { /* already gone */ }
  }
  maskChanged(true);
}

function undoLastStroke() {
  const m = activeMask();
  if (!m || m.type !== "brush" || !m.strokes.length) return;
  m.strokes.pop();
  maskChanged(true);
}

function bindMaskUi() {
  // Scoped to the buttons that name a kind. An automatic-mask button carries
  // .mask-add-btn too, so a bare class selector matched it here as well and
  // every click on Subject ran both handlers: this one first, with
  // dataset.addMask undefined, pushing a typeless mask that MASK_KINDS has no
  // entry for — which then threw out of renderMaskList and left the real mask
  // missing from the list.
  $$("#edit-modal .mask-add-btn[data-add-mask]").forEach((b) =>
    b.addEventListener("click", () => addMask(b.dataset.addMask)));
  $$("[data-add-auto]").forEach((b) =>
    b.addEventListener("click", async () => {
      // The model is fetched on the first automatic mask, not at startup: a
      // shoot with no people in it should never pay for it.
      if (await ensureSegmentModel()) addMask("auto", b.dataset.addAuto);
    }));

  $("#mask-list").addEventListener("click", (e) => {
    const del = e.target.closest("[data-mask-del]");
    if (del) { deleteMask(Number(del.dataset.maskDel)); return; }
    const eye = e.target.closest("[data-mask-toggle]");
    if (eye) {
      const m = editSession.edit.masks[Number(eye.dataset.maskToggle)];
      m.enabled = !m.enabled;
      maskChanged(true);
      return;
    }
    const row = e.target.closest("[data-mask]");
    if (row) selectMask(Number(row.dataset.mask));
  });
  // Double-clicking a mask's name renames it.
  $("#mask-list").addEventListener("dblclick", (e) => {
    const row = e.target.closest("[data-mask]");
    const idx = row ? Number(row.dataset.mask) : -1;
    const m = editSession.edit.masks[idx];
    if (!m) return;
    const name = prompt("Mask name:", maskLabel(m, idx));
    if (name == null) return;
    m.name = name.trim().slice(0, 40);
    maskChanged(true);
  });

  $("#mask-invert").addEventListener("click", () => {
    const m = activeMask();
    if (!m) return;
    m.invert = !m.invert;
    maskChanged(true);
  });
  $("#mask-name").addEventListener("input", (e) => {
    const m = activeMask();
    if (!m) return;
    m.name = e.target.value.slice(0, 40);
    renderMaskList();          // not renderMaskDetail: that would fight the caret
    setEditDirty();
  });
  $("#mask-delete").addEventListener("click", () => deleteMask(editSession.activeMask));
  $("#mask-duplicate").addEventListener("click", duplicateActiveMask);
  for (const key of ["feather", "amount"]) {
    $(`#mask-${key}`).addEventListener("input", (e) => {
      const m = activeMask();
      if (!m) return;
      m[key] = parseInt(e.target.value, 10);
      $(`#mask-${key}-val`).textContent = m[key];
      drawOverlay();
      setEditDirty();
      previewDuringDrag();
    });
  }
  $("#mask-brush-size").addEventListener("input", (e) => {
    editSession.brush.size = parseInt(e.target.value, 10);
    $("#mask-brush-size-val").textContent = editSession.brush.size;
    drawOverlay();
  });
  $("#mask-brush-paint").addEventListener("click", () => setBrushErase(false));
  $("#mask-brush-erase").addEventListener("click", () => setBrushErase(true));
  $("#mask-brush-undo").addEventListener("click", undoLastStroke);
  // Crop & straighten
  $("#film-enabled").addEventListener("change", (e) =>
    updateFilm({ enabled: e.target.checked }));
  $("#hsl-reset").addEventListener("click", () =>
    updateHsl(Object.fromEntries(HSL_FIELDS.map((f) => [f.k, 0]))));
  $("#grade-reset").addEventListener("click", () =>
    updateGradeZone(Object.fromEntries(GRADE_FIELDS.map((f) => [f.k, 0]))));
  $("#mask-range-on").addEventListener("change", (e) => {
    const m = activeMask();
    if (!m) return;
    m.range_luma = e.target.checked ? { ...LUMA_RANGE_NEW } : null;
    maskChanged(true);
  });
  $("#wb-pick").addEventListener("click", () =>
    setEditTool(editSession.tool === "wb" ? null : "wb"));
  $("#wb-reset").addEventListener("click", () => {
    // Zeroes whatever the Color panel is showing, which is the mask's own pair
    // while one is selected and the global one otherwise.
    const target = adjTarget();
    target.temp = 0;
    target.tint = 0;
    syncEditSliders();
    setEditDirty();
    fetchEditPreview(true);
  });
  $("#crop-tool").addEventListener("click", () =>
    setEditTool(editSession.tool === "crop" ? null : "crop"));
  $("#crop-reset").addEventListener("click", () => {
    editSession.edit.tilt = 0;
    editSession.cropAspect = "free";
    setCrop(null);
  });
  $$("#crop-aspects button").forEach((b) => {
    b.addEventListener("click", () => {
      editSession.cropAspect = b.dataset.aspect;
      // Choosing a shape is a request to see it, so arm the tool if it is not.
      if (editSession.tool !== "crop") setEditTool("crop");
      setCrop(cropForAspect(cropTargetAspect()));
    });
  });
  $("#crop-tilt").addEventListener("input", (e) => {
    editSession.edit.tilt = parseFloat(e.target.value) || 0;
    syncCropControls();
    setEditDirty();
    previewDuringDrag();
  });
  $("#crop-tilt").addEventListener("change", () => fetchEditPreview(true));
  $("#mask-show").addEventListener("change", (e) => {
    editSession.showMask = e.target.checked;
    drawOverlay();
  });

  $$("#edit-modal .edit-zoom").forEach((b) =>
    b.addEventListener("click", () => setZoom(parseFloat(b.dataset.zoom))));

  const cv = $("#edit-overlay");
  cv.addEventListener("pointerdown", overlayDown);
  cv.addEventListener("pointermove", overlayMove);
  cv.addEventListener("pointerup", overlayUp);
  cv.addEventListener("pointercancel", overlayUp);
  cv.addEventListener("wheel", (e) => {
    if (!editSession.relPath) return;
    e.preventDefault();
    const steps = ZOOM_STEPS;
    const cur = steps.indexOf(editSession.view.zoom);
    const at = cur >= 0 ? cur : 0;
    const next = steps[Math.min(steps.length - 1, Math.max(0, at + (e.deltaY < 0 ? 1 : -1)))];
    if (next === editSession.view.zoom) return;
    // Zoom toward the pointer so the detail under it stays under it.
    setZoom(next, next > 0 ? evFrac(e) : null);
  }, { passive: false });
  cv.addEventListener("pointerleave", () => {
    if (!editSession.drag) { editSession.hover = null; drawOverlay(); }
  });
  cv.addEventListener("contextmenu", (e) => e.preventDefault());
  cv.addEventListener("auxclick", (e) => { if (e.button === 1) e.preventDefault(); });
  $("#edit-img").addEventListener("load", () => {
    $("#edit-img").classList.remove("view-pending");
    resizeOverlay();
  });
  window.addEventListener("resize", () => { layoutPreviewImage(); resizeOverlay(); });

  // Space is the usual "grab the canvas" modifier; hold it to pan past a mask.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || $("#edit-modal").classList.contains("hidden")) return;
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    editSession.spaceHeld = true;
    if (isZoomed()) cv.style.cursor = "grab";
    e.preventDefault();
  });
  document.addEventListener("keyup", (e) => {
    if (e.code === "Space") editSession.spaceHeld = false;
  });
}

function setBrushErase(on) {
  editSession.brush.erase = on;
  renderMaskDetail();
}

function nudgeBrushSize(delta) {
  if (isRepairBrush(editSession.tool)) {
    repairState.size = Math.max(1, Math.min(150, repairState.size + Math.sign(delta) * 2));
    renderRepairPanel();
    drawOverlay();
    return;
  }
  const sl = $("#mask-brush-size");
  editSession.brush.size = Math.max(5, Math.min(300, editSession.brush.size + delta));
  sl.value = editSession.brush.size;
  $("#mask-brush-size-val").textContent = editSession.brush.size;
  drawOverlay();
}

// ---------- edit slots (per photo) ----------
//
// Deliberately not presets: a preset is a look you carry between photos, a slot
// is a variant of *this* photo you are not ready to throw away. They are stored
// on the photo and written the moment you fill one, so closing the editor with
// Cancel keeps them and loses only the working edit.
//
// Shaped as a list rather than a row of numbered chips, and that was a
// correction: with chips, one click meant "save" on an empty slot and "load" on
// a full one, and overwriting was a modifier nobody can see. Which of the three
// you were about to get depended on state the chip did not show. A row per slot
// says what it holds, when it was put there, and carries its own save and clear
// buttons — so every action is visible and none of them is a mode.

// What is in a stash, for the row's label. Names the groups that are off
// neutral rather than listing sliders: "light · colour · 2 masks" is what tells
// two stashes apart at a glance.
function slotSummary(e) {
  const parts = [];
  const any = (keys) => keys.some((k) => Math.abs(Number(e[k]) || 0) > 1e-6);
  if (any(["exposure", "contrast", "highlights", "shadows", "whites", "blacks"])) parts.push("light");
  if (any(["temp", "tint", "vibrance", "saturation"])) parts.push("colour");
  if (e.hsl) parts.push("mixer");
  if (e.grading) parts.push("grading");
  if (CURVE_CHANNELS.some((c) => !curveIsIdentity(e[c.k]))) parts.push("curves");
  if (any(["clarity", "texture", "dehaze", "denoise", "sharpen", "vignette",
           "blur", "motion", "glow", "pixelate"])) parts.push("detail");
  if (e.film && e.film.enabled) parts.push("film");
  if (e.watermark) parts.push("watermark");
  if (e.crop || Math.abs(Number(e.tilt) || 0) > 1e-4) parts.push("crop");
  const n = (e.masks || []).length;
  if (n) parts.push(`${n} mask${n === 1 ? "" : "s"}`);
  return parts.length ? parts.join(" · ") : "no adjustments";
}

// "3m ago" beats a timestamp here: what you want to know about a stash is
// whether it is the one you made just now or the one from this morning.
function slotAge(iso) {
  const t = Date.parse(iso || "");
  if (!t) return "";
  const mins = Math.floor((Date.now() - t) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  return new Date(t).toLocaleDateString();
}

// The rack as the server sends it, plus the two derived things the panel needs:
// each slot's edit in working form (so the comparison below allocates nothing)
// and which slot the working edit currently matches.
function setSlots(list) {
  editSession.slots = Array.isArray(list) ? list : [];
  editSession.slotEdits = editSession.slots.map(
    (sl) => (sl ? mergeNeutralEdit(sl.edit) : null));
  editSession.slotMark = -1;
  markSlot();
}

// Which slot the working edit is currently equal to, if any. Derived from the
// values every time rather than remembered from the last load: nudge one slider
// and the marker goes away by itself, which is the honest answer and costs a
// comparison the editor is already doing for the dirty flag.
//
// `slotTouch` breaks the tie when two slots hold the same thing — stash a
// variant twice and both match, and the one you last acted on is the one you
// mean. It is only a preference: the moment it stops matching, the scan decides.
function markSlot() {
  const eq = (i) => {
    const se = editSession.slotEdits[i];
    return !!se && editsEqual(se, editSession.edit);
  };
  let now = -1;
  if (editSession.slotTouch >= 0 && eq(editSession.slotTouch)) {
    now = editSession.slotTouch;
  } else for (let i = 0; i < editSession.slotEdits.length; i++) {
    if (eq(i)) { now = i; break; }
  }
  const changed = now !== editSession.slotMark;
  editSession.slotMark = now;
  if (changed) renderSlots();
}

function renderSlots() {
  const list = $("#edit-slots");
  if (!list) return;
  const used = editSession.slots.filter(Boolean).length;
  $("#slot-summary-state").textContent = used ? `· ${used} of ${EDIT_SLOTS}` : "";
  list.innerHTML = Array.from({ length: EDIT_SLOTS }, (_, i) => {
    const sl = editSession.slots[i];
    const here = i === editSession.slotMark;
    // A slot you loaded or saved and have since edited away from. Nothing is
    // written behind your back — this is the row saying so, and its save button
    // is how you fold the change back in.
    const stale = !here && sl && i === editSession.slotTouch;
    const cls = sl ? `filled${here ? " current" : stale ? " stale" : ""}` : "empty";
    const what = sl ? slotSummary(editSession.slotEdits[i]) : "empty";
    const when = here ? "on screen"
      : stale ? "edited \u2014 not saved" : (sl ? slotAge(sl.saved_at) : "");
    const save = `<button class="slot-btn" data-slot-save="${i}" title="${
      sl ? `Overwrite slot ${i + 1} with what is on screen`
         : `Save what is on screen into slot ${i + 1}`}">${icon("save")}</button>`;
    const del = sl
      ? `<button class="slot-btn" data-slot-del="${i}" title="Clear slot ${i + 1}">×</button>`
      : `<span class="slot-btn">&nbsp;</span>`;
    return `<div class="slot-row ${cls}" data-slot="${i}"${
      sl ? ` title="Click to load slot ${i + 1}"` : ""}>`
      + `<span class="slot-num">${i + 1}</span>`
      + `<span class="slot-what">${escapeHtml(what)}</span>`
      + `<span class="slot-when">${escapeHtml(when)}</span>${save}${del}</div>`;
  }).join("");
}

// Write one slot through to the db. `edit` of null clears it. The server owns
// the rack, so its answer replaces ours rather than being merged into it.
async function slotWrite(i, edit) {
  if (!editSession.relPath) return;
  const rel = editSession.relPath;
  const res = await fetch("/api/edit/slot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: rel, slot: i, edit }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#edit-status").textContent = "slot failed: " + (err.detail || res.status);
    return;
  }
  const { slots } = await res.json();
  editSession.slotTouch = edit ? i : -1;
  setSlots(slots);
  renderSlots();
  // The grid's copy has to agree, or reopening the editor shows the old rack.
  const photo = state.photos.find((p) => p.rel_path === rel);
  if (photo) {
    if (editSession.slots.some(Boolean)) photo.edit_slots = editSession.slots;
    else delete photo.edit_slots;
  }
  $("#edit-status").textContent = edit
    ? `saved into slot ${i + 1}` : `slot ${i + 1} cleared`;
}

// Bring a stash back. Replaces the working edit outright — that is what a slot
// is for — and leaves it unsaved, so Cancel still backs out to the photo as it
// is on disk.
function slotLoad(i) {
  if (!editSession.slotEdits[i]) return;
  editSession.slotTouch = i;
  editSession.edit = mergeNeutralEdit(editSession.slots[i].edit);
  selectMask(-1, { silent: true });
  setEditTool(null);
  renderWatermarkPanel();
  renderFilmPanel();
  renderHslPanel();
  renderGradePanel();
  renderCurveChannels();
  renderLookPanel();
  renderRepairPanel();
  renderOpticsPanel();
  renderPortraitPanel();
  syncEditSliders();
  drawCurve();
  drawOverlay();
  setEditDirty();
  renderSlots();
  fetchEditPreview(true);
  $("#edit-status").textContent = `slot ${i + 1} loaded — press Save to keep it`;
}

function bindSlots() {
  $("#edit-slots").addEventListener("click", (e) => {
    const save = e.target.closest("[data-slot-save]");
    if (save) { slotWrite(Number(save.dataset.slotSave), editPayload()); return; }
    const del = e.target.closest("[data-slot-del]");
    if (del) { slotWrite(Number(del.dataset.slotDel), null); return; }
    // The row itself only ever loads. Saving is the button, always, full slot
    // or empty — one action per control.
    const row = e.target.closest("[data-slot]");
    if (row) slotLoad(Number(row.dataset.slot));
  });
}

// ---------- presets (app-global) ----------
async function loadPresets() {
  try {
    const res = await fetch("/api/presets", { cache: "no-store" });
    if (res.ok) state.presets = (await res.json()).presets || [];
  } catch { /* keep whatever we have */ }
  renderPresetOptions();
}

function renderPresetOptions(selectId) {
  // Built-ins and the user's own presets are separated so a long personal list
  // never buries the shipped starting points.
  const builtin = state.presets.filter((p) => p.builtin);
  const mine = state.presets.filter((p) => !p.builtin);
  const opts = (list) => list.map((p) => {
    // Flag the ones that carry local adjustments — those are the presets the
    // add/replace choice actually matters for.
    const n = ((p.edit && p.edit.masks) || []).length;
    const suffix = n ? `  ·  ${n} mask${n === 1 ? "" : "s"}` : "";
    return `<option value="${p.id}"${p.hint ? ` title="${escapeHtml(p.hint)}"` : ""}>` +
      `${escapeHtml(p.name)}${suffix}</option>`;
  }).join("");
  // Built-ins carry a group — Portrait, Look, Mono and the rest — and the
  // library is long enough that one flat list buried whichever set you were
  // after. Built in the order the server sends so the file stays the running
  // order. The user's own presets stay in one group: those are theirs to name.
  const groups = [];
  for (const p of builtin) {
    const label = p.group || "Built-in";
    const g = groups.find((x) => x.label === label);
    if (g) g.items.push(p); else groups.push({ label, items: [p] });
  }
  for (const sel of [$("#edit-preset-select"), $("#bulk-preset-select")]) {
    if (!sel) continue;
    const keep = selectId != null ? selectId : sel.value;
    const placeholder = sel.id === "bulk-preset-select" ? "Choose a preset…" : "Presets…";
    sel.innerHTML = `<option value="">${placeholder}</option>`
      + groups.map((g) =>
          `<optgroup label="${escapeHtml(g.label)}">${opts(g.items)}</optgroup>`).join("")
      + (mine.length ? `<optgroup label="My presets">${opts(mine)}</optgroup>` : "");
    if (keep && state.presets.some((p) => p.id === keep)) sel.value = keep;
  }
}

const curveIsIdentity = (c) =>
  !c || c.every(([x, y]) => Math.abs(y - x) < 1e-4);

// Mirrors editing.merge_additive: lay a preset over the current edit so local
// presets compose instead of wiping the work already on the photo.
function mergeAdditive(base, overlay) {
  const out = mergeNeutralEdit(base);
  const over = mergeNeutralEdit(overlay);
  for (const f of EDIT_FIELDS) {
    if (Math.abs(over[f.k] || 0) > 1e-4) out[f.k] = over[f.k];
  }
  if (!curveIsIdentity(over.curve)) out.curve = over.curve.map((p) => p.slice());
  if (over.watermark) out.watermark = cloneWatermark(over.watermark);
  // The rest mirrors editing.merge_additive, which the bulk apply uses: this
  // used to stop at the watermark, so a preset added here lost its film, its
  // colour mixer, its grading, its crop and its look, which a bulk apply kept.
  if (over.crop) out.crop = { ...over.crop };
  if (over.film) out.film = { ...over.film };
  if (over.lut) out.lut = { ...over.lut };
  if (over.lens) out.lens = { ...over.lens };
  if (over.portrait) out.portrait = { ...over.portrait };
  if (over.transform) out.transform = { ...over.transform };
  // Appended, as editing.merge_additive does: dust sits in the same place on
  // every frame a body shoots, so a preset of spots is meant to add to a photo.
  if (over.healing) out.healing = { ops: [...(out.healing ? out.healing.ops : []), ...over.healing.ops] };
  if (over.redeye) {
    out.redeye = { enabled: true,
                   corrections: [...(out.redeye ? out.redeye.corrections : []), ...over.redeye.corrections] };
  }
  if (over.hsl) {
    const merged = cloneHsl(out.hsl) || {};
    for (const [band, vals] of Object.entries(over.hsl)) merged[band] = { ...(merged[band] || {}), ...vals };
    out.hsl = merged;
  }
  if (over.grading) {
    const merged = cloneGrading(out.grading) || {};
    for (const [k, v] of Object.entries(over.grading)) {
      merged[k] = (v && typeof v === "object") ? { ...(merged[k] || {}), ...v } : v;
    }
    out.grading = merged;
  }
  out.masks = out.masks.concat(over.masks).slice(0, MASK_MAX);
  return out;
}

async function applyPreset(id) {
  const preset = state.presets.find((p) => p.id === id);
  if (!preset) return;
  // A preset can carry an automatic mask, and the render would otherwise pull
  // the segmentation model down inside the request — a preview that hangs for
  // however long the download takes, with nothing on screen saying why.
  const needsModel = ((preset.edit && preset.edit.masks) || [])
    .some((m) => m.type === "auto");
  if (needsModel && !(await ensureSegmentModel())) return;
  const additive = state.presetMode === "add";
  const before = editSession.edit.masks.length;
  editSession.edit = additive
    ? mergeAdditive(editSession.edit, preset.edit)
    : mergeNeutralEdit(preset.edit);
  const added = editSession.edit.masks.length - before;
  // No status line here: the render that follows overwrites it within a frame.
  // Selecting the new mask below is the feedback, and the mask list shows it.
  // Land on the first mask the preset brought in, so it can be moved at once.
  const focus = additive && added > 0 ? before : -1;
  selectMask(focus, { silent: true });
  setEditTool(null);
  renderWatermarkPanel();
  renderFilmPanel();
  renderHslPanel();
  renderGradePanel();
  renderCurveChannels();
  renderLookPanel();
  renderRepairPanel();
  renderOpticsPanel();
  renderPortraitPanel();
  syncEditSliders();
  drawCurve();
  drawOverlay();
  setEditDirty();
  fetchEditPreview(true);
}

function setPresetMode(mode) {
  state.presetMode = mode;
  try { localStorage.setItem("pcls.presetMode", mode); } catch { /* private */ }
  $$("#preset-mode [data-preset-mode]").forEach((b) =>
    b.classList.toggle("active", b.dataset.presetMode === mode));
}

async function saveCurrentAsPreset() {
  const name = (prompt("Preset name:") || "").trim();
  if (!name) return;
  const res = await fetch("/api/presets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, edit: editSession.edit }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Save preset failed: " + (err.detail || res.status));
    return;
  }
  const result = await res.json();
  state.presets = result.presets || [];
  renderPresetOptions(result.preset ? result.preset.id : null);
}

async function deleteSelectedPreset() {
  const id = $("#edit-preset-select").value;
  if (!id) { alert("Pick a preset to delete first."); return; }
  const preset = state.presets.find((p) => p.id === id);
  if (preset && preset.builtin) {
    alert(`"${preset.name}" is a built-in preset and can't be deleted.\n\n` +
          `Tweak it and use “Save preset…” to keep your own version.`);
    return;
  }
  if (!confirm(`Delete preset "${preset ? preset.name : id}"?`)) return;
  const res = await fetch(`/api/presets/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) { alert("Delete failed"); return; }
  state.presets = (await res.json()).presets || [];
  renderPresetOptions("");
}

// ---------- download the selection ----------
async function downloadSelection() {
  const rels = [...state.selection];
  if (!rels.length) return;
  const btn = $("#selection-download");
  const label = btn.textContent.trim();
  btn.disabled = true;
  // RAW and edited photos are rendered at full size, which is not instant.
  setBtnLabel(btn, `saving ${rels.length}…`);
  try {
    const res = await fetch("/api/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rel_paths: rels }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert("Download failed: " + (err.detail || res.status));
      return;
    }
    const info = await res.json();
    const missing = info.missing.length ? ` · ${info.missing.length} missing` : "";
    showDownloadToast(
      `Saved ${info.saved} photo${info.saved === 1 ? "" : "s"} to ${info.target_dir}${missing}`,
      info.target_dir);
  } finally {
    btn.disabled = false;
    setBtnLabel(btn, label);
  }
}

// Says where the files went and offers to open it — a path in a toast you
// cannot act on is just trivia.
function showDownloadToast(text, dir) {
  const el = $("#download-toast");
  el.innerHTML = `<span></span><button type="button" id="download-reveal">Open folder</button>`;
  el.querySelector("span").textContent = text;
  el.querySelector("#download-reveal").addEventListener("click", () => {
    fetch("/api/reveal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: dir }),
    });
  });
  el.classList.remove("hidden");
  clearTimeout(showDownloadToast.timer);
  showDownloadToast.timer = setTimeout(() => el.classList.add("hidden"), 10000);
}

// ---------- selection + batch apply ----------
let lastSelIdx = null;

function toggleSelect(absIdx, shift) {
  const photo = state.filteredPhotos[absIdx];
  if (!photo) return;
  if (shift && lastSelIdx != null) {
    const a = Math.min(lastSelIdx, absIdx), b = Math.max(lastSelIdx, absIdx);
    for (let i = a; i <= b; i++) {
      const p = state.filteredPhotos[i];
      if (p) state.selection.add(p.rel_path);
    }
  } else if (state.selection.has(photo.rel_path)) {
    state.selection.delete(photo.rel_path);
  } else {
    state.selection.add(photo.rel_path);
  }
  lastSelIdx = absIdx;
  renderGrid();
  renderSelectionBar();
}

function clearSelection() {
  state.selection.clear();
  lastSelIdx = null;
  renderGrid();
  renderSelectionBar();
}

function renderSelectionBar() {
  const bar = $("#selection-bar");
  const n = state.selection.size;
  if (!n) { bar.classList.add("hidden"); return; }
  $("#selection-count").textContent = `${n} selected`;
  bar.classList.remove("hidden");
}

const bulkState = { currentEdit: null };

function openBulkModal(opts) {
  opts = opts || {};
  bulkState.currentEdit = (opts.fromEditor && editSession.edit)
    ? mergeNeutralEdit(editSession.edit) : null;
  $("#bulk-n-selected").textContent = `${state.selection.size} photo(s)`;
  $("#bulk-n-scene").textContent =
    `${(state.byScene.get(state.selectedScene) || []).length} photo(s)`;
  $("#bulk-n-picks").textContent =
    `${state.photos.filter((p) => p.decision === "pick").length} photo(s)`;
  setOptionCardValue("#bulk-scope-cards", opts.scope || (state.selection.size ? "selected" : "scene"));
  const curCard = document.querySelector('#bulk-source-cards .option-card[data-value="current"]');
  curCard.classList.toggle("disabled", !bulkState.currentEdit);
  setOptionCardValue("#bulk-source-cards", bulkState.currentEdit ? "current" : "preset");
  renderPresetOptions();
  $("#bulk-status").textContent = "";
  updateBulkUi();
  $("#bulk-modal").classList.remove("hidden");
}

function closeBulkModal() { $("#bulk-modal").classList.add("hidden"); }

function updateBulkUi() {
  const source = getOptionCardValue("#bulk-source-cards");
  $("#bulk-preset-row").style.display = source === "preset" ? "" : "none";
  // Clearing edits can only mean replace, so the choice is meaningless there.
  $("#bulk-mode-row").style.display = source === "clear" ? "none" : "";
  $("#bulk-mode-note").textContent = bulkMode() === "add"
    ? "keeps each photo's own adjustments; masks are added to them"
    : "discards each photo's current adjustments";
  const n = bulkScopeRelPaths(getOptionCardValue("#bulk-scope-cards")).length;
  $("#bulk-summary").textContent = n ? `${n} photo(s) will change` : "no target photos";
  $("#bulk-confirm").disabled = n === 0;
}

function bulkScopeRelPaths(scope) {
  if (scope === "selected") return [...state.selection];
  if (scope === "scene") return (state.byScene.get(state.selectedScene) || []).map((p) => p.rel_path);
  if (scope === "picks") return state.photos.filter((p) => p.decision === "pick").map((p) => p.rel_path);
  return [];
}

function bulkSourceEdit(source) {
  if (source === "current") return bulkState.currentEdit || {};
  if (source === "clear") return {};
  if (source === "preset") {
    const preset = state.presets.find((p) => p.id === $("#bulk-preset-select").value);
    return preset ? preset.edit : null;
  }
  return null;
}

// "Clear edits" is inherently a replace; anything else respects the toggle.
function bulkMode() {
  if (getOptionCardValue("#bulk-source-cards") === "clear") return "replace";
  const active = $("#bulk-mode .active");
  return active ? active.dataset.bulkMode : "add";
}

async function confirmBulkApply() {
  const scope = getOptionCardValue("#bulk-scope-cards");
  const source = getOptionCardValue("#bulk-source-cards");
  const rels = bulkScopeRelPaths(scope);
  if (!rels.length) return;
  const edit = bulkSourceEdit(source);
  if (edit == null) { $("#bulk-status").textContent = "Pick a preset first."; return; }
  $("#bulk-confirm").disabled = true;
  $("#bulk-status").textContent = "Applying…";
  const res = await fetch("/api/edit/bulk", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_paths: rels, edit, mode: bulkMode() }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#bulk-status").textContent = "Failed: " + (err.detail || res.status);
    $("#bulk-confirm").disabled = false;
    return;
  }
  // Reflect locally (edited_at bumps the thumb cache-buster). In add mode the
  // result differs per photo, so merge against each one the same way the
  // server just did.
  const additive = bulkMode() === "add";
  const now = new Date().toISOString();
  for (const rel of rels) {
    const ph = state.photos.find((p) => p.rel_path === rel);
    if (!ph) continue;
    const merged = additive ? mergeAdditive(ph.edit, edit) : mergeNeutralEdit(edit);
    if (isNeutralEdit(merged)) { delete ph.edit; delete ph.edited_at; }
    else { ph.edit = merged; ph.edited_at = now; }
  }
  closeBulkModal();
  clearSelection();
  renderMain();
}

// ---------- modal ----------
function openModal(absIdx) {
  state.modal.open = true;
  state.modal.idx = absIdx;
  state.modal.fit = true;
  state.modal.compare = false;
  state.cursorIdx = absIdx;
  $("#modal").classList.remove("hidden");
  renderModal();
  scheduleViewSave();
}

function closeModal() {
  state.modal.open = false;
  modalLoadSeq++;
  delete $("#modal-image").dataset.full;
  loupe.at = null;
  hideLoupe();
  $("#modal").classList.add("hidden");
  renderMain();
}

// The viewer image is free to be letterboxed (fit) or larger than its scroll
// container (100%), so the box layer is measured off the <img> rather than
// derived from CSS the way the grid tiles can.
function syncModalBoxes() {
  const layer = $("#modal-boxes"), img = $("#modal-image");
  if (!layer) return;
  const photo = state.modal.open ? state.filteredPhotos[state.modal.idx] : null;
  const objs = (photo && !state.modal.compare && state.showBoxes)
    ? (photo.objects || []) : [];
  layer.classList.toggle("hidden", !objs.length);
  if (!objs.length) { layer.innerHTML = ""; return; }
  layer.style.left = `${img.offsetLeft}px`;
  layer.style.top = `${img.offsetTop}px`;
  layer.style.width = `${img.offsetWidth}px`;
  layer.style.height = `${img.offsetHeight}px`;
  layer.innerHTML = objs.map((o) => {
    // Same mapping the grid uses: the viewer shows the cropped frame too.
    const v = boxToView(o, photo);
    if (!v) return "";
    const label = o.vehicle_id
      ? (state.subjects.vehicles.find((x) => x.id === o.vehicle_id)?.label || o.cls)
      : o.cls;
    return `<span class="det-box${o.vehicle_id ? " grouped" : ""}" style="left:${v.l * 100}%;` +
      `top:${v.t * 100}%;width:${v.w * 100}%;height:${v.h * 100}%">` +
      `<span class="det-label">${escapeHtml(label)} ${Math.round(o.score * 100)}</span></span>`;
  }).join("");
}

function renderModal() {
  const photo = state.filteredPhotos[state.modal.idx];
  if (!photo) { closeModal(); return; }
  const isHdr = photo.type === "hdr" && !!photo.base;
  const comparing = isHdr && state.modal.compare;
  const shownRel = comparing ? photo.base : photo.rel_path;
  const img = $("#modal-image");
  showModalImage(photo, shownRel);
  img.className = state.modal.fit ? "fit" : "actual";
  $("#modal-title").textContent = basename(shownRel);
  $("#modal-title").title = shownRel;
  $("#modal-meta").textContent = `${state.modal.idx + 1} of ${state.filteredPhotos.length} · ${photo.scene}`;
  const cmp = $("#modal-compare");
  cmp.classList.toggle("hidden", !isHdr);
  cmp.classList.toggle("active", comparing);
  cmp.textContent = comparing ? "showing: 0 EV original" : "showing: HDR merged";
  if (isHdr) {
    // Preload the other version so the toggle is instant.
    new Image().src = "/img/" + enc(comparing ? photo.rel_path : photo.base);
  }
  $$("#modal-decide .btn-decision").forEach((b) =>
    b.classList.toggle("active", b.dataset.decision === photo.decision));
  $("#modal-marks").innerHTML = starsHtml(photo.rating, "stars clickable")
    + (photo.label ? `<span class="label-dot ${photo.label}" title="${LABEL_NAMES[photo.label]} label"></span>` : "");
  const ex = photo.exif || {};
  $("#modal-exif").textContent = [
    ex.camera, ex.lens, ex.focal, ex.aperture, ex.shutter, ex.iso_text,
  ].filter(Boolean).join("  ·  ");
  const s = photo.scores || {};
  const eye = s.eye_open != null ? s.eye_open.toFixed(3) : "—";
  const suggestion = photo.auto_suggestion ? `Suggested: ${AUTO_LABEL[photo.auto_suggestion] || photo.auto_suggestion}` : "";
  $("#modal-scores").textContent = [suggestion, ...explainScores(photo)].filter(Boolean).join("  ·  ");
  // The numbers the words come from, for whoever wants them.
  $("#modal-scores").title =
    `blur ${(s.blur ?? 0).toFixed(0)} (rank ${(s.blur_pct ?? 0).toFixed(2)}) · ` +
    `exposure |z| ${(s.exposure_zscore ?? 0).toFixed(2)} · eye ${eye} · ` +
    `badness ${(s.badness ?? 0).toFixed(2)}`;
  img.onload = () => { syncModalBoxes(); syncModalPeak(); drawLoupe(); };
  syncModalBoxes();
  syncModalPeak();
  drawLoupe();
  renderFilmstrip($("#modal-filmstrip"), state.modal.idx, (i) => {
    if (i === state.modal.idx) return;
    state.modal.idx = i;
    state.cursorIdx = i;
    state.modal.fit = true;
    state.modal.compare = false;
    renderModal();
    scheduleViewSave();
  });
}

// What the scores say, in words. The sharpness rank and exposure distance are
// within the photo's scene, which is what they are computed against
// (scorer.compute_scene_badness); the eye cutoff is scorer.EYE_CLOSED_THRESHOLD.
const EYE_CLOSED_THRESHOLD = 0.18;

function explainScores(photo) {
  const s = photo.scores || {};
  const out = [];
  if (s.blur_pct != null) {
    const pct = Math.round(s.blur_pct * 100);
    out.push(pct >= 100 ? "Sharpest in the scene" : pct <= 0 ? "Softest in the scene"
      : `Sharper than ${pct}% of the scene`);
  }
  if (s.exposure_zscore != null) {
    const scene = state.byScene.get(photo.scene) || [];
    const b = scene.map((p) => p.scores?.brightness).filter((v) => v != null);
    const mean = b.length ? b.reduce((x, y) => x + y, 0) / b.length : null;
    const side = mean != null && s.brightness != null && s.brightness < mean ? "darker" : "brighter";
    out.push(s.exposure_zscore < 1 ? "Exposure like the rest"
      : `${s.exposure_zscore < 2 ? "A little" : "Much"} ${side} than the rest`);
  }
  if (s.eye_open != null && s.eye_open < EYE_CLOSED_THRESHOLD) out.push("Eyes look closed");
  if (s.subject_area != null) {
    out.push(s.subject_area <= 0 ? "No subject found" : `Subject fills ${Math.max(1, Math.round(s.subject_area * 100))}%`);
  }
  return out;
}

// The grid's thumbnail is already in the browser, so it goes up at once and
// the full image replaces it when it arrives: a RAW used to show black for
// seconds. A load that fails says so instead of staying black, and only the
// newest request may put its image up.
let modalLoadSeq = 0;

function showModalImage(photo, rel) {
  const img = $("#modal-image");
  const seq = ++modalLoadSeq;
  const full = "/img/" + enc(rel);
  const note = $("#modal-loading");
  if (img.dataset.full === full) return;
  img.dataset.full = full;
  img.src = rel === photo.rel_path ? thumbUrl(photo) : full;
  note.textContent = "Loading full size…";
  note.classList.remove("hidden", "error");
  const loader = new Image();
  loader.onload = () => {
    if (seq !== modalLoadSeq) return;
    img.src = full;
    note.classList.add("hidden");
  };
  loader.onerror = () => {
    if (seq !== modalLoadSeq) return;
    note.textContent = "The full-size photo could not be loaded; this is its thumbnail.";
    note.classList.add("error");
  };
  loader.src = full;
}

function modalNav(delta) {
  if (!state.filteredPhotos.length) return;
  const i = Math.max(0, Math.min(state.filteredPhotos.length - 1, state.modal.idx + delta));
  if (i === state.modal.idx) return;
  state.modal.idx = i;
  state.cursorIdx = i;
  state.modal.fit = true;
  state.modal.compare = false;
  renderModal();
  scheduleViewSave();
}

function toggleCompare() {
  const photo = state.filteredPhotos[state.modal.idx];
  if (!photo || photo.type !== "hdr" || !photo.base) return;
  state.modal.compare = !state.modal.compare;
  renderModal();
}

// ---------- keyboard ----------
// The decision a key stands for: null clears it, undefined is not a decision key.
function decisionForKey(k) {
  if (k === "r" || k === "R") return "reject";
  if (k === "v" || k === "V") return "review";
  if (k === "p" || k === "P" || k === "a" || k === "A") return "pick";
  if (k === "u" || k === "U") return null;
  return undefined;
}

function bindKeys() {
  document.addEventListener("keydown", async (e) => {
    // A tour owns the keyboard while it is up: arrows and Enter walk it, Esc
    // leaves, and nothing reaches the photos underneath.
    if (tourState.name) {
      if (e.key === "Escape") endTour();
      else if (e.key === "ArrowRight" || e.key === "Enter") tourStep(1);
      else if (e.key === "ArrowLeft") tourStep(-1);
      e.preventDefault();
      return;
    }
    // Preferences: ⌘, or Ctrl+, as everywhere; Esc closes; nothing else gets
    // through to the photos behind.
    if ((e.metaKey || e.ctrlKey) && e.key === ",") { e.preventDefault(); openPrefs(); return; }
    if (prefsOpen()) {
      if (e.key === "Escape") { closePrefs(); e.preventDefault(); }
      return;
    }
    // A question on screen takes Esc (the answer that changes nothing) and
    // Enter (the primary one), and nothing else.
    if (!$("#choice-modal").classList.contains("hidden")) {
      if (e.key === "Escape") answerChoice($("#choice-modal").dataset.escape);
      else if (e.key === "Enter") answerChoice($("#choice-modal").dataset.enter);
      e.preventDefault();
      return;
    }
    // The shortcut sheet: ? opens it where you are and ? or Esc closes it;
    // nothing reaches the photos underneath while it is up.
    if (!$("#keys-sheet").classList.contains("hidden")) {
      if (e.key === "Escape" || e.key === "?") closeKeysSheet();
      e.preventDefault();
      return;
    }
    if (e.key === "?" && !(e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName))
        && document.body.classList.contains("landing-mode") === false) {
      e.preventDefault();
      openKeysSheet(keyContext());
      return;
    }
    // In the editor, ⌘Z / Ctrl+Z steps back through the edit and ⇧⌘Z or Ctrl+Y
    // forward. A text field keeps the keys for its own undo.
    if ((e.metaKey || e.ctrlKey) && !e.altKey
        && !$("#edit-modal").classList.contains("hidden")) {
      const t = e.target || {};
      const tag = (t.tagName || "").toLowerCase();
      const typing = tag === "textarea" || t.isContentEditable || (tag === "input"
        && !["checkbox", "radio", "range", "button", "submit", "color"].includes((t.type || "").toLowerCase()));
      const key = (e.key || "").toLowerCase();
      if (!typing && (key === "z" || key === "y")) {
        e.preventDefault();
        if (key === "y" || e.shiftKey) editRedo(); else editUndo();
        return;
      }
    }
    // ⌘Z (or Ctrl+Z) restores the last bulk-decide action.
    if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey
        && (e.key === "z" || e.key === "Z")) {
      if (undoToast.paths && undoToast.paths.length) {
        e.preventDefault();
        performUndo();
        return;
      }
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;

    // A text field owns its letters. Without this a filter box cannot be typed
    // into at all, because "a" is pick and "r" is reject — which is exactly what
    // happened to the class filter in the re-score dialog. Checkboxes, radios and
    // sliders are deliberately not counted: clicking one should not swallow the
    // grid shortcuts afterwards. Escape always gets through, so a dialog can
    // still be dismissed from inside its own input.
    const el = e.target || {};
    const tag = (el.tagName || "").toLowerCase();
    const type = (el.type || "").toLowerCase();
    const textEntry = el.isContentEditable || tag === "textarea"
      || (tag === "input"
          && !["checkbox", "radio", "range", "button", "submit", "color"].includes(type));
    if (textEntry && k !== "Escape") return;
    if (k === "Escape" && closeMenus()) { e.preventDefault(); return; }

    // The projects explainer; Esc only.
    if (!$("#explainer-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeExplainer(); e.preventDefault(); }
      return;
    }

    // Export picks; Esc only, or R/V/A would decide photos behind the dialog.
    if (!$("#export-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeExportModal(); e.preventDefault(); }
      return;
    }

    // The relink dialog (landing) — Esc only.
    if (!$("#relink-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeRelinkModal(); e.preventDefault(); }
      return;
    }

    // The batch-apply dialog sits on top of everything; Esc only.
    if (!$("#bulk-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeBulkModal(); e.preventDefault(); }
      return;
    }

    // Subject groups; Esc only.
    if (!$("#subject-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeSubjectModal(); e.preventDefault(); }
      return;
    }

    // The re-score dialog owns its class filter; Esc only.
    if (!$("#rescore-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeRescoreModal(); e.preventDefault(); }
      return;
    }

    // Clustering settings; Esc only.
    if (!$("#cluster-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeClusterModal(); e.preventDefault(); }
      return;
    }

    // The editor swallows grid/decision shortcuts while open (its sliders and
    // preset input own their own keys); only a few navigation keys pass through.
    if (!$("#edit-modal").classList.contains("hidden")) {
      const tag = (e.target.tagName || "").toLowerCase();
      const typing = tag === "input" || tag === "select" || tag === "textarea";
      if (k === "Escape") {
        // Esc backs out one level: armed tool, then mask selection, then modal.
        if (editSession.tool) setEditTool(null);
        else if (editSession.activeMask >= 0) selectMask(-1);
        else leaveEditor("close");
        e.preventDefault(); return;
      }
      if (typing) return;  // let focused sliders / preset picker keep arrows etc.
      if (k === "ArrowLeft") { editNav(-1); e.preventDefault(); return; }
      if (k === "ArrowRight") { editNav(+1); e.preventDefault(); return; }
      if ((k === "c" || k === "C") && !e.repeat) { editCompareOn(); e.preventDefault(); return; }
      if (k === "Enter") { saveEdit(); e.preventDefault(); return; }
      // Local adjustments
      if (k === "r" || k === "R") { addMask("radial"); e.preventDefault(); return; }
      if (k === "g" || k === "G") { addMask("linear"); e.preventDefault(); return; }
      if (k === "b" || k === "B") { addMask("brush"); e.preventDefault(); return; }
      if (k === "\\") {
        $("#mask-show").checked = editSession.showMask = !editSession.showMask;
        drawOverlay(); e.preventDefault(); return;
      }
      if ((k === "Delete" || k === "Backspace") && editSession.activeMask >= 0) {
        deleteMask(editSession.activeMask); e.preventDefault(); return;
      }
      if (k === "[" || k === "]") { nudgeBrushSize(k === "[" ? -8 : 8); e.preventDefault(); return; }
      if (k === "f" || k === "F") {
        setZoom(isZoomed() ? 0 : 1, isZoomed() ? null : { x: 0.5, y: 0.5 });
        e.preventDefault(); return;
      }
      return;
    }

    // The HDR look tuner swallows shortcuts (its sliders use arrows); Esc only.
    if (!$("#hdr-look-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeLookModal(); e.preventDefault(); }
      return;
    }
    // The HDR modal swallows grid/photo shortcuts; only Esc closes it.
    if (!$("#hdr-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeHdrModal(); e.preventDefault(); }
      return;
    }

    if (compareOpen()) {
      const n = compare.photos.length;
      if (k === "Escape") { closeCompare(); e.preventDefault(); return; }
      if (k === "Tab") { setCompareActive((compare.active + (e.shiftKey ? n - 1 : 1)) % n); e.preventDefault(); return; }
      if (k === "ArrowRight" || k === "ArrowLeft") { compareStep(k === "ArrowRight" ? 1 : -1); e.preventDefault(); return; }
      if (k === "z" || k === "Z" || k === "f" || k === "F") { compareToggleZoom(); e.preventDefault(); return; }
      if (k === "e" || k === "E") {
        const at = state.filteredPhotos.indexOf(compare.photos[compare.active]);
        if (at >= 0) { closeCompare(); openEditModal(at); }
        e.preventDefault(); return;
      }
      const mark = markForKey(k, compare.photos[compare.active]);
      if (mark) { compareMark(mark); e.preventDefault(); return; }
      const decision = decisionForKey(k);
      if (decision !== undefined) { compareDecide(decision); e.preventDefault(); }
      return;
    }

    if (state.modal.open) {
      if (k === "Escape") { closeModal(); e.preventDefault(); return; }
      if (k === "ArrowRight" || k === " ") { modalNav(+1); e.preventDefault(); return; }
      if (k === "ArrowLeft") { modalNav(-1); e.preventDefault(); return; }
      if (k === "f" || k === "F" || k === "z" || k === "Z") {
        state.modal.fit = !state.modal.fit;
        $("#modal-image").className = state.modal.fit ? "fit" : "actual";
        e.preventDefault(); return;
      }
      if (k === "c" || k === "C") { toggleCompare(); e.preventDefault(); return; }
      if (k === "b" || k === "B") { toggleBoxes(); e.preventDefault(); return; }
      if (k === "k" || k === "K") { togglePeak(); e.preventDefault(); return; }
      if (k === "l" || k === "L") { toggleLoupe(); e.preventDefault(); return; }
      if (k === "e" || k === "E") {
        const i = state.modal.idx; closeModal(); openEditModal(i); e.preventDefault(); return;
      }
      const mark = markForKey(k, state.filteredPhotos[state.modal.idx]);
      if (mark) { e.preventDefault(); markAt(state.modal.idx, mark); return; }
      const decision = decisionForKey(k);
      if (decision === undefined) return;
      e.preventDefault();
      decideAt(state.modal.idx, decision, { advance: true });
      return;
    }

    // grid mode
    if (!state.filteredPhotos.length) return;
    const layout = LAYOUTS[state.pageSize];
    const cols = layout.cols;
    const i = state.cursorIdx;

    const total = state.filteredPhotos.length;

    if (k === "ArrowRight") {
      if (i + 1 < total) focusAt(i + 1);
      e.preventDefault(); return;
    }
    if (k === "ArrowLeft") {
      if (i - 1 >= 0) focusAt(i - 1);
      e.preventDefault(); return;
    }
    if (k === "ArrowDown") { focusAt(Math.min(total - 1, i + cols)); e.preventDefault(); return; }
    if (k === "ArrowUp") { focusAt(Math.max(0, i - cols)); e.preventDefault(); return; }
    if (k === "PageDown" || k === "]") { gotoPage(+1); e.preventDefault(); return; }
    if (k === "PageUp" || k === "[") { gotoPage(-1); e.preventDefault(); return; }
    if (k === "Enter") { openModal(i); e.preventDefault(); return; }
    if (k === "e" || k === "E") { openEditModal(state.cursorIdx); e.preventDefault(); return; }
    if (k === "b" || k === "B") { toggleBoxes(); e.preventDefault(); return; }
    if (k === "k" || k === "K") { togglePeak(); e.preventDefault(); return; }
    if (k === "i" || k === "I") { toggleInspector(); e.preventDefault(); return; }
    if (k === "c" || k === "C") { openCompare(); e.preventDefault(); return; }
    if (k === "x" || k === "X") { toggleSelect(state.cursorIdx, e.shiftKey); e.preventDefault(); return; }
    if ((k === "d" || k === "D") && state.selection.size) {
      downloadSelection(); e.preventDefault(); return;
    }
    if (k === "Escape") { if (state.selection.size) { clearSelection(); e.preventDefault(); } return; }

    const mark = markForKey(k, state.filteredPhotos[i]);
    if (mark) { e.preventDefault(); markAt(i, mark); return; }
    const decision = decisionForKey(k);
    if (decision !== undefined) {
      e.preventDefault();
      decideAt(i, decision, { advance: true });
    }
  });

  // Release hold-to-compare in the editor.
  document.addEventListener("keyup", (e) => {
    if ((e.key === "c" || e.key === "C") && !$("#edit-modal").classList.contains("hidden")) {
      editCompareOff();
    }
  });
}

// Back to the projects, or on to another one (`next`, a project folder). The
// page reloads between the two, so nothing of this project is carried into
// the next; the one to open is handed over in sessionStorage.
async function leaveProject(next) {
  if (document.body.classList.contains("landing-mode")) {
    if (next) openProjectByDir(next);
    return;
  }
  if (!$("#edit-modal").classList.contains("hidden") && !(await leaveEditor("close"))) return;
  await decisionSaves;
  await flushViewSave();
  const res = await fetch("/api/close", { method: "POST" });
  if (res.status === 409) {
    const err = await res.json();
    const what = err.detail.charAt(0).toUpperCase() + err.detail.slice(1);
    alert(`${what}. Let it finish, or stop it, before leaving this project.`);
    return;
  }
  if (!res.ok) { alert("Could not close the project: " + res.status); return; }
  if (next) sessionStorage.setItem("pcls.openNext", next);
  location.reload();
}

// ---------- inspector ----------
// The highlighted photo in detail, beside the grid. The histogram is worked
// out here from the thumbnail the grid already loaded: a 256-bin count of the
// three channels, drawn as three overlapping areas.
const histCache = new Map();   // thumb URL -> {r, g, b} bins

function inspectorOn() {
  try { return localStorage.getItem("pcls.inspector") !== "0"; } catch { return true; }
}

function toggleInspector() {
  const on = !inspectorOn();
  try { localStorage.setItem("pcls.inspector", on ? "1" : "0"); } catch { /* private */ }
  document.body.classList.toggle("inspector-off", !on);
  $("#inspector-btn").classList.toggle("active", on);
  renderInspector();
}

function histogramOf(url) {
  if (histCache.has(url)) return Promise.resolve(histCache.get(url));
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const w = 256, h = Math.max(1, Math.round(256 * img.naturalHeight / img.naturalWidth));
      const c = document.createElement("canvas");
      c.width = w; c.height = h;
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(img, 0, 0, w, h);
      const px = g.getImageData(0, 0, w, h).data;
      const bins = { r: new Uint32Array(256), g: new Uint32Array(256), b: new Uint32Array(256) };
      for (let i = 0; i < px.length; i += 4) { bins.r[px[i]]++; bins.g[px[i + 1]]++; bins.b[px[i + 2]]++; }
      if (histCache.size > 400) histCache.clear();
      histCache.set(url, bins);
      resolve(bins);
    };
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function drawHistogram(bins) {
  const cv = $("#insp-histogram"), g = cv.getContext("2d");
  g.clearRect(0, 0, cv.width, cv.height);
  if (!bins) return;
  // Scaled to the tallest bin short of the extremes, so a clipped white sky
  // does not flatten everything else into the floor.
  let top = 1;
  for (const ch of ["r", "g", "b"]) for (let i = 1; i < 255; i++) top = Math.max(top, bins[ch][i]);
  g.globalCompositeOperation = "lighter";
  for (const [ch, col] of [["r", "rgba(230,80,80,0.7)"], ["g", "rgba(80,200,110,0.7)"], ["b", "rgba(90,140,240,0.7)"]]) {
    g.fillStyle = col;
    g.beginPath();
    g.moveTo(0, cv.height);
    for (let i = 0; i < 256; i++) g.lineTo(i, cv.height - Math.min(1, bins[ch][i] / top) * cv.height);
    g.lineTo(255, cv.height);
    g.closePath();
    g.fill();
  }
  g.globalCompositeOperation = "source-over";
}

function renderInspector() {
  if (!inspectorOn() || !$("#inspector").offsetParent) return;
  const p = state.filteredPhotos[state.cursorIdx];
  $("#inspector").classList.toggle("empty", !p);
  if (!p) return;
  const url = thumbUrl(p);
  histogramOf(url).then((bins) => {
    if (state.filteredPhotos[state.cursorIdx] === p) drawHistogram(bins);
  });
  $("#insp-name").textContent = basename(p.rel_path);
  $("#insp-name").title = p.rel_path;
  const ex = p.exif || {};
  $("#insp-when").textContent = [ex.captured_at ? ex.captured_at.replace(/^(\d{4}):(\d\d):(\d\d)/, "$1-$2-$3") : "",
    p.edit ? "Edited" : ""].filter(Boolean).join("  ·  ");
  $("#insp-suggestion").innerHTML = p.auto_suggestion
    ? `<span class="auto-badge ${p.auto_suggestion}">${AUTO_LABEL[p.auto_suggestion]}</span>`
      + `<span class="insp-badness" title="How bad it looks: lower is better">badness ${(p.scores?.badness ?? 0).toFixed(2)}</span>`
    : "—";
  $("#insp-reasons").innerHTML = explainScores(p).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
  $$("#insp-decide .btn-decision").forEach((b) => b.classList.toggle("active", b.dataset.decision === p.decision));
  $("#insp-stars").innerHTML = starsHtml(p.rating, "stars clickable");
  $$("#insp-marks .label-chip").forEach((c) => c.classList.toggle("active", c.dataset.label === p.label));
  const faces = (p.faces || []).map((f, fi) => ({ f, fi, person: f.person_id ? state.peopleById.get(f.person_id) : null }))
    .filter(({ person }) => !(person && person.excluded));
  $("#insp-faces-section").classList.toggle("hidden", !faces.length);
  $("#insp-faces").innerHTML = faces.map(({ fi, person }) =>
    `<figure><img loading="lazy" alt="" src="/face/${enc(p.rel_path)}?idx=${fi}" />`
    + `<figcaption>${person ? escapeHtml(person.label) : ""}</figcaption></figure>`).join("");
  const rows = [["Camera", ex.camera], ["Lens", ex.lens], ["Focal", ex.focal], ["Aperture", ex.aperture],
    ["Shutter", ex.shutter], ["ISO", ex.iso_text], ["Size", p.width && p.height ? `${p.width} × ${p.height}` : ""]];
  $("#insp-exif").innerHTML = rows.filter(([, v]) => v).map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
}

function bindInspector() {
  document.body.classList.toggle("inspector-off", !inspectorOn());
  $("#inspector-btn").classList.toggle("active", inspectorOn());
  $("#inspector-btn").addEventListener("click", toggleInspector);
  $$("#insp-decide .btn-decision").forEach((b) => b.addEventListener("click", () => {
    const p = state.filteredPhotos[state.cursorIdx];
    if (p) decideAt(state.cursorIdx, p.decision === b.dataset.decision ? null : b.dataset.decision);
  }));
  $("#insp-edit").addEventListener("click", () => openEditModal(state.cursorIdx));
  $$("#insp-marks .label-chip").forEach((c) => c.addEventListener("click", () => {
    const p = state.filteredPhotos[state.cursorIdx];
    if (p) markAt(state.cursorIdx, { label: p.label === c.dataset.label ? "" : c.dataset.label });
  }));
  // A star clicked sets that many; the star already set clears them.
  const starClicks = (el, idx) => el.addEventListener("click", (e) => {
    const star = e.target.closest("[data-star]");
    const i = idx();
    const p = state.filteredPhotos[i];
    if (!star || !p) return;
    const n = Number(star.dataset.star);
    markAt(i, { rating: (p.rating || 0) === n ? 0 : n });
  });
  starClicks($("#insp-stars"), () => state.cursorIdx);
  starClicks($("#modal-marks"), () => state.modal.idx);
  // Filters on stars and labels
  for (const [sel, key, parse] of [["#rating-filter", "minRating", Number], ["#label-filter", "labelFilter", String]]) {
    $(sel).addEventListener("change", (e) => {
      state[key] = parse(e.target.value);
      state.cursorIdx = 0;
      recomputeFilter();
      $("#grid").scrollTop = 0;
      showCursor();
      e.target.blur();   // so the next key goes to the photos, not the menu
    });
  }
}

// ---------- compare ----------
// Two to four photos side by side, to choose between frames of a burst: the
// selection if two or more are selected, else the highlighted photo and the
// next. One pane is active; the arrows swap its photo for the one before or
// after it (skipping those already up), and the decision, star and label keys
// act on it. Z zooms every pane to 100% at the same place and panning one
// pans them all, since sharpness is what a burst differs in.
const COMPARE_MAX = 4;
const compare = { photos: [], list: [], active: 0, zoom: false, syncing: false };

function compareOpen() { return !$("#compare").classList.contains("hidden"); }

function openCompare() {
  const list = state.filteredPhotos;
  if (list.length < 2) return;
  const picked = list.filter((p) => state.selection.has(p.rel_path));
  const i = state.cursorIdx;
  compare.photos = picked.length >= 2 ? picked.slice(0, COMPARE_MAX)
    : [list[i], list[i + 1] || list[i - 1]];
  compare.list = list.slice();   // the order to step through, fixed while comparing
  compare.active = 0;
  compare.zoom = false;
  $("#compare-hint").textContent = picked.length > COMPARE_MAX
    ? `The first ${COMPARE_MAX} of the ${picked.length} selected` : "";
  $("#compare").classList.remove("hidden");
  buildComparePanes();
}

function closeCompare() {
  const photo = compare.photos[compare.active];
  $("#compare").classList.add("hidden");
  const at = state.filteredPhotos.indexOf(photo);
  if (at >= 0) state.cursorIdx = at;
  showCursor();
}

function buildComparePanes() {
  const wrap = $("#compare-panes");
  wrap.dataset.n = compare.photos.length;
  wrap.classList.toggle("zoomed", compare.zoom);
  wrap.innerHTML = compare.photos.map((_, i) => `<section class="cmp-pane" data-i="${i}">
      <header><span class="cmp-name"></span><span class="cmp-marks"></span><span class="cmp-decision"></span></header>
      <div class="cmp-view"><img alt="" draggable="false" /></div>
      <footer class="cmp-why"></footer>
    </section>`).join("");
  wrap.querySelectorAll(".cmp-pane").forEach((pane, i) => {
    pane.addEventListener("pointerdown", () => setCompareActive(i));
    const view = pane.querySelector(".cmp-view");
    view.addEventListener("scroll", () => syncCompareScroll(view));
    bindComparePan(view);
    loadComparePhoto(i);
  });
  renderCompare();
}

// The thumbnail at once, the full image when it arrives, as in the viewer.
function loadComparePhoto(i) {
  const pane = $(`#compare-panes .cmp-pane[data-i="${i}"]`);
  const photo = compare.photos[i];
  const img = pane.querySelector("img");
  pane.dataset.rel = photo.rel_path;
  img.src = thumbUrl(photo);
  const full = new Image();
  // Late, and the pane has moved on to another photo: not this one's to show.
  full.onload = () => { if (pane.dataset.rel === photo.rel_path) img.src = full.src; };
  full.src = "/img/" + enc(photo.rel_path);
}

function renderCompare() {
  $$("#compare-panes .cmp-pane").forEach((pane, i) => {
    const p = compare.photos[i];
    pane.classList.toggle("active", i === compare.active);
    pane.querySelector(".cmp-name").textContent = basename(p.rel_path);
    pane.querySelector(".cmp-name").title = p.rel_path;
    pane.querySelector(".cmp-marks").innerHTML = starsHtml(p.rating)
      + (p.label ? `<span class="label-dot ${p.label}"></span>` : "");
    const d = pane.querySelector(".cmp-decision");
    d.className = "cmp-decision" + (p.decision ? ` ${p.decision}` : "");
    d.textContent = p.decision ? AUTO_LABEL[p.decision] : "Undecided";
    const suggestion = p.auto_suggestion ? `Suggested: ${AUTO_LABEL[p.auto_suggestion]}` : "";
    pane.querySelector(".cmp-why").textContent = [suggestion, ...explainScores(p)].filter(Boolean).join("  ·  ");
  });
  $("#compare-title").textContent = `Compare ${compare.photos.length}`;
  setBtnLabel($("#compare-zoom"), compare.zoom ? "Fit" : "100%");
}

function setCompareActive(i) {
  if (i === compare.active) return;
  compare.active = i;
  renderCompare();
}

function compareStep(delta) {
  const list = compare.list, shown = new Set(compare.photos);
  let j = list.indexOf(compare.photos[compare.active]) + delta;
  while (j >= 0 && j < list.length && shown.has(list[j])) j += delta;
  if (j < 0 || j >= list.length) return;
  compare.photos[compare.active] = list[j];
  loadComparePhoto(compare.active);
  renderCompare();
}

// Zoom every pane to 100% around the same point, given as a fraction of the
// photo, or back to fit.
function compareToggleZoom(fx = 0.5, fy = 0.5) {
  compare.zoom = !compare.zoom;
  $("#compare-panes").classList.toggle("zoomed", compare.zoom);
  renderCompare();
  if (!compare.zoom) return;
  requestAnimationFrame(() => {
    compare.syncing = true;
    $$("#compare-panes .cmp-view").forEach((v) => {
      v.scrollLeft = fx * v.scrollWidth - v.clientWidth / 2;
      v.scrollTop = fy * v.scrollHeight - v.clientHeight / 2;
    });
    requestAnimationFrame(() => { compare.syncing = false; });
  });
}

// Panning one pane pans the others to the same place, as a fraction of each
// photo, so frames of different sizes still line up.
function syncCompareScroll(src) {
  if (compare.syncing || !compare.zoom) return;
  compare.syncing = true;
  const fx = (src.scrollLeft + src.clientWidth / 2) / src.scrollWidth;
  const fy = (src.scrollTop + src.clientHeight / 2) / src.scrollHeight;
  $$("#compare-panes .cmp-view").forEach((v) => {
    if (v === src) return;
    v.scrollLeft = fx * v.scrollWidth - v.clientWidth / 2;
    v.scrollTop = fy * v.scrollHeight - v.clientHeight / 2;
  });
  requestAnimationFrame(() => { compare.syncing = false; });
}

// A click on a fitted photo zooms in there; on a zoomed one, a drag pans.
function bindComparePan(view) {
  let drag = null;
  view.addEventListener("pointerdown", (e) => {
    if (!compare.zoom) return;
    drag = { x: e.clientX, y: e.clientY, left: view.scrollLeft, top: view.scrollTop, moved: false };
    view.setPointerCapture(e.pointerId);
  });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    view.scrollLeft = drag.left - dx;
    view.scrollTop = drag.top - dy;
  });
  view.addEventListener("pointerup", (e) => {
    const was = drag;
    drag = null;
    if (was && was.moved) return;
    const img = view.querySelector("img").getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (e.clientX - img.left) / img.width));
    const fy = Math.min(1, Math.max(0, (e.clientY - img.top) / img.height));
    compareToggleZoom(fx, fy);
  });
}

function compareDecide(decision) {
  applyDecision(compare.photos[compare.active], decision);
  afterCompareChange();
}

function compareMark(marks) {
  applyMarks(compare.photos[compare.active], marks);
  afterCompareChange();
}

// The grid behind is kept current, so closing lands on what was decided.
function afterCompareChange() {
  recomputeFilter();
  renderSidebar();
  renderMain();
  renderCompare();
}

function bindCompare() {
  $("#compare-btn").addEventListener("click", openCompare);
  $("#compare-close").addEventListener("click", closeCompare);
  $("#compare-zoom").addEventListener("click", () => compareToggleZoom());
}

// ---------- filmstrip ----------
// The photos of the list you are working through, along the bottom of the
// viewer and the editor. Built once per list and only re-marked as you move,
// so a scene of a thousand photos costs one render, not one per keypress.
function renderFilmstrip(el, current, onPick) {
  const list = state.filteredPhotos;
  const sig = `${state.selectedScene}|${state.filter}|${list.length}|${list[0]?.rel_path || ""}|${list[list.length - 1]?.rel_path || ""}`;
  if (el.dataset.sig !== sig) {
    el.dataset.sig = sig;
    el.innerHTML = list.map((p, i) =>
      `<button type="button" class="film-cell" data-i="${i}" title="${escapeAttr(basename(p.rel_path))}">`
      + `<img loading="lazy" alt="" src="${thumbUrl(p)}" /></button>`).join("");
  }
  el.onclick = (e) => {
    const cell = e.target.closest(".film-cell");
    if (cell) onPick(Number(cell.dataset.i));
  };
  el.querySelectorAll(".film-cell").forEach((cell, i) => {
    const p = list[i];
    cell.className = "film-cell" + (i === current ? " current" : "") + (p.decision ? ` d-${p.decision}` : "")
      + (p.label ? ` l-${p.label}` : "");
    const img = cell.firstElementChild, src = thumbUrl(p);
    if (!img.src.endsWith(src)) img.src = src;   // an edit saved since
  });
  el.querySelector(".film-cell.current")?.scrollIntoView({ block: "nearest", inline: "center" });
}

// ---------- the top bar ----------
// Which tab is lit follows what is on screen, watched rather than set from
// every place that opens or closes a view.
function currentModule() {
  if (document.body.classList.contains("landing-mode")) return "projects";
  return $("#edit-modal").classList.contains("hidden") ? "cull" : "edit";
}

function syncTopbar() {
  const mod = currentModule();
  const open = mod !== "projects";
  $$("#topbar .tb-tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.module === mod);
    b.setAttribute("aria-selected", b.dataset.module === mod ? "true" : "false");
  });
  $("#tab-cull").disabled = !open;
  $("#tab-edit").disabled = !open || !state.filteredPhotos.length;
  $("#project-switcher").classList.toggle("hidden", !open);
  $("#export-picks-btn").classList.toggle("hidden", !open);
  $("#tour-main-btn").classList.toggle("hidden", !open);
}

async function goToModule(mod) {
  const now = currentModule();
  if (mod === now) return;
  if (mod === "projects") { leaveProject(null); return; }
  if (mod === "cull") {
    if (now === "edit") await leaveEditor("close");
    return;
  }
  // Edit: the photo in the viewer or the active compare pane, else the
  // highlighted one in the grid.
  if (compareOpen()) {
    const at = state.filteredPhotos.indexOf(compare.photos[compare.active]);
    if (at < 0) return;
    closeCompare();
    openEditModal(at);
    return;
  }
  const i = state.modal.open ? state.modal.idx : state.cursorIdx;
  if (state.modal.open) closeModal();
  openEditModal(i);
}

async function renderProjectMenu() {
  const menu = $("#project-menu");
  const res = await fetch("/api/recents");
  const recents = res.ok ? (await res.json()).recents || [] : [];
  const here = state.projectDir;
  const others = recents.filter((r) => r.project_dir && r.project_dir !== here).slice(0, 8);
  menu.innerHTML = (others.length ? `<div class="menu-label">Recent projects</div>` : "")
    + others.map((r) => `<button type="button" role="menuitem" data-dir="${escapeAttr(r.project_dir)}" title="${escapeAttr(r.project_dir)}">${icon("folder")}<span>${escapeHtml(r.name || basename(r.project_dir))}</span></button>`).join("")
    + (others.length ? "<hr />" : "")
    + `<button type="button" role="menuitem" data-all="1">${icon("left")}<span>All projects</span></button>`;
  menu.querySelectorAll("button[data-dir]").forEach((b) =>
    b.addEventListener("click", () => leaveProject(b.dataset.dir)));
  menu.querySelector("button[data-all]").addEventListener("click", () => leaveProject(null));
}

// Background work, said in the bar: an export, a re-score or face grouping.
const TASK_WORDS = { export: "Exporting", scoring: "Scoring", grouping: "Grouping faces" };
let activityTimer = null;

async function pollActivity() {
  if (document.body.classList.contains("landing-mode")) { $("#activity").classList.add("hidden"); return; }
  const res = await fetch("/api/state", { cache: "no-store" }).catch(() => null);
  if (!res || !res.ok) return;
  const tasks = (await res.json()).tasks || {};
  const running = Object.entries(tasks).filter(([, t]) => t && t.running);
  $("#activity").classList.toggle("hidden", !running.length);
  $("#activity-text").textContent = running.map(([name, t]) =>
    `${TASK_WORDS[name] || name}${t.total ? ` ${t.idx}/${t.total}` : "…"}`).join(" · ");
}

function bindTopbar() {
  $("#tab-cull").addEventListener("click", () => goToModule("cull"));
  $("#tab-edit").addEventListener("click", () => goToModule("edit"));
  bindMenu("#project-btn", "#project-menu");
  $("#project-btn").addEventListener("click", renderProjectMenu);
  const watch = new MutationObserver(syncTopbar);
  for (const el of [document.body, $("#edit-modal"), $("#modal")]) {
    watch.observe(el, { attributes: true, attributeFilter: ["class"] });
  }
  syncTopbar();
  activityTimer = setInterval(pollActivity, 2000);
}

// ---------- preferences ----------
const PHOTO_BGS = ["black", "dark", "grey"];

function photoBg() {
  try { return localStorage.getItem("pcls.photoBg") || "dark"; } catch { return "dark"; }
}

function applyPhotoBg(bg) {
  document.body.dataset.photoBg = bg;
  $$("#pref-photo-bg button").forEach((b) => b.classList.toggle("active", b.dataset.bg === bg));
}

function prefsOpen() { return !$("#prefs-modal").classList.contains("hidden"); }

async function openPrefs(pane = "general") {
  await loadWorkspaces();
  const open = currentModule() !== "projects";
  $("#pref-project").classList.toggle("hidden", !open);
  $("#pref-project-none").classList.toggle("hidden", open);
  if (open) {
    const sg = state.sceneGrouping || { mode: "folder", gap_minutes: 30 };
    $("#pref-grouping-now").textContent = sg.mode === "time_gap"
      ? `By time gap: a new scene after ${sg.gap_minutes} minutes without a photo.`
      : "By folder: each subfolder is a scene.";
  }
  $("#pref-keybar").checked = !keybarCollapsed();
  $("#pref-inspector").checked = inspectorOn();
  $("#pref-loupe").checked = loupe.on;
  applyPhotoBg(photoBg());
  renderPrefWorkspaces();
  showPrefsPane(pane);
  $("#prefs-modal").classList.remove("hidden");
}

function closePrefs() { $("#prefs-modal").classList.add("hidden"); }

function showPrefsPane(pane) {
  $$("#prefs-modal .prefs-nav button").forEach((b) => b.classList.toggle("active", b.dataset.pane === pane));
  $$("#prefs-modal .prefs-panes > section").forEach((sec) => { sec.hidden = sec.dataset.pane !== pane; });
}

function renderPrefWorkspaces() {
  const ul = $("#pref-workspaces");
  ul.innerHTML = workspaceState.list.map((w) => `<li class="${w === workspaceState.current ? "current" : ""}">
      <span class="pref-ws-name">${icon("folder")}<span title="${escapeAttr(w)}">${escapeHtml(basename(w) || w)}</span></span>
      <span class="pref-ws-path">${escapeHtml(w)}</span>
      ${w === workspaceState.current ? `<span class="pref-ws-now">Current</span>`
        : `<button type="button" data-use="${escapeAttr(w)}">Use</button>`}
      ${workspaceState.list.length > 1 ? `<button type="button" data-forget="${escapeAttr(w)}" title="Remove from the list (deletes nothing)" aria-label="Remove from the list">${icon("close")}</button>` : ""}
    </li>`).join("");
  ul.querySelectorAll("[data-use]").forEach((b) => b.addEventListener("click", async () => {
    await switchWorkspace(b.dataset.use);
    renderWorkspaceSelect();
    renderPrefWorkspaces();
  }));
  ul.querySelectorAll("[data-forget]").forEach((b) => b.addEventListener("click", async () => {
    const res = await fetch("/api/workspaces/forget", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: b.dataset.forget }),
    });
    if (!res.ok) throw new Error(`forget failed: ${res.status}`);
    const d = await res.json();
    workspaceState.list = d.workspaces || [];
    workspaceState.current = d.current;
    renderWorkspaceSelect();
    renderPrefWorkspaces();
    if (currentModule() === "projects") loadWorkspaceProjects();
  }));
}

function bindPrefs() {
  applyPhotoBg(photoBg());
  $("#prefs-btn").addEventListener("click", () => openPrefs());
  $("#prefs-close").addEventListener("click", closePrefs);
  $("#prefs-modal").addEventListener("click", (e) => { if (e.target.id === "prefs-modal") closePrefs(); });
  $$("#prefs-modal .prefs-nav button").forEach((b) => b.addEventListener("click", () => showPrefsPane(b.dataset.pane)));
  $("#pref-keybar").addEventListener("change", (e) => {
    try { localStorage.setItem("pcls.keybar", e.target.checked ? "1" : "0"); } catch { /* private */ }
    renderKeybars();
  });
  $("#pref-inspector").addEventListener("change", (e) => { if (e.target.checked !== inspectorOn()) toggleInspector(); });
  $("#pref-loupe").addEventListener("change", (e) => { if (e.target.checked !== loupe.on) toggleLoupe(); });
  $$("#pref-photo-bg button").forEach((b) => b.addEventListener("click", () => {
    try { localStorage.setItem("pcls.photoBg", b.dataset.bg); } catch { /* private */ }
    applyPhotoBg(b.dataset.bg);
  }));
  $("#pref-tours").addEventListener("click", () => {
    for (const name of Object.keys(TOURS)) {
      try { localStorage.removeItem(`pcls.tour.${name}.seen`); } catch { /* private */ }
    }
    setBtnLabel($("#pref-tours"), "They will show again");
    $("#pref-tours").disabled = true;
  });
  $("#pref-ws-add").addEventListener("click", async () => { await addWorkspace(); renderPrefWorkspaces(); });
  $("#pref-grouping").addEventListener("click", () => {
    closePrefs();
    $("#scene-grouping-section").open = true;
    $("#scene-grouping-section").scrollIntoView({ block: "nearest" });
  });
  $("#pref-rescore").addEventListener("click", () => { closePrefs(); openRescoreModal(); });
  $("#pref-cluster").addEventListener("click", () => { closePrefs(); openClusterModal(); });
  $("#pref-hdr").addEventListener("click", () => { closePrefs(); openLookModal(); });
  $("#pref-keys").addEventListener("click", () => { closePrefs(); openKeysSheet(keyContext()); });
}

// ---------- menus ----------
// A button that drops a list of actions. Opens on click, closes on a choice,
// a click elsewhere or Esc.
function bindMenu(btnSel, menuSel) {
  const btn = $(btnSel), menu = $(menuSel);
  const close = () => { menu.classList.add("hidden"); btn.setAttribute("aria-expanded", "false"); };
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = menu.classList.contains("hidden");
    $$(".menu").forEach((m) => m.classList.add("hidden"));
    menu.classList.toggle("hidden", !open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  menu.addEventListener("click", (e) => { if (e.target.closest("button")) close(); });
  document.addEventListener("click", (e) => { if (!menu.contains(e.target)) close(); });
}

function closeMenus() {
  const open = $$(".menu:not(.hidden)");
  open.forEach((m) => m.classList.add("hidden"));
  return open.length > 0;
}

// ---------- UI bindings ----------
function bindUi() {
  bindPrefs();
  bindScoreProgress();
  bindCompare();
  bindInspector();
  bindTopbar();
  bindMenu("#workspace-btn", "#workspace-menu");
  $("#landing-new-project").addEventListener("click", () => openWizard());
  $("#projects-sort").addEventListener("change", () => {
    try { localStorage.setItem("pcls.projectsSort", $("#projects-sort").value); } catch { /* private */ }
    renderProjectGrid(lastProjects);
  });
  try { $("#projects-sort").value = localStorage.getItem("pcls.projectsSort") || "opened"; } catch { /* private */ }
  restoreEditGroups();
  bindSliderLooks();
  bindMenu("#more-btn", "#more-menu");
  bindGridHover();
  bindKeysSheet();
  $$(".filter").forEach((b) => {
    b.addEventListener("click", () => {
      $$(".filter").forEach((x) => x.classList.toggle("active", x === b));
      state.filter = b.dataset.filter;
      state.cursorIdx = 0;
      recomputeFilter();
      $("#grid").scrollTop = 0;
      showCursor();
    });
  });
  $$("#cols-toggle .cols").forEach((b) => {
    b.addEventListener("click", () => {
      $$("#cols-toggle .cols").forEach((x) => x.classList.toggle("active", x === b));
      state.pageSize = parseInt(b.dataset.cols, 10);
      showCursor();
    });
  });
  $("#prev-page").addEventListener("click", () => gotoPage(-1));
  $("#next-page").addEventListener("click", () => gotoPage(+1));
  $("#modal-close").addEventListener("click", closeModal);
  $("#modal-compare").addEventListener("click", toggleCompare);
  $$("#modal-decide .btn-decision").forEach((b) => b.addEventListener("click", () => {
    const photo = state.filteredPhotos[state.modal.idx];
    if (!photo) return;
    // Pressing the decision it already has clears it, as on a tile.
    decideAt(state.modal.idx, photo.decision === b.dataset.decision ? null : b.dataset.decision, { advance: true });
  }));
  $("#rescore-btn").addEventListener("click", openRescoreModal);
  // Subjects
  $("#subject-settings-btn").addEventListener("click", openSubjectModal);
  $("#rescore-modal-close").addEventListener("click", closeRescoreModal);
  $("#rescore-cancel").addEventListener("click", closeRescoreModal);
  $("#rescore-go").addEventListener("click", runRescore);
  $("#subject-modal-close").addEventListener("click", closeSubjectModal);
  $("#subject-cancel").addEventListener("click", closeSubjectModal);
  $("#subject-apply").addEventListener("click", applySubjectSettings);
  $("#subject-search").addEventListener("input", (e) => {
    subjectEdit.search = e.target.value;
    renderSubjectClassPicker();
  });
  $("#boxes-btn").addEventListener("click", toggleBoxes);
  $("#peak-btn").addEventListener("click", togglePeak);
  $("#peak-level").addEventListener("click", (e) => {
    const b = e.target.closest("[data-peak-level]");
    if (b) setPeakLevel(b.dataset.peakLevel);
  });
  window.addEventListener("resize", () => { syncModalBoxes(); syncModalPeak(); });
  $("#reject-undecided-btn").addEventListener("click", rejectUndecidedInScene);
  $("#export-picks-btn").addEventListener("click", openExportModal);
  $("#export-modal-close").addEventListener("click", closeExportModal);
  $("#export-cancel").addEventListener("click", exportCancelOrClose);
  $("#export-confirm").addEventListener("click", confirmExport);
  bindExportForm();
  $$("#export-mode-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      setOptionCardValue("#export-mode-cards", card.dataset.value);
    });
  });
  $("#hdr-btn").addEventListener("click", openHdrModal);
  $("#hdr-modal-close").addEventListener("click", closeHdrModal);
  $("#hdr-cancel").addEventListener("click", closeHdrModal);
  $("#hdr-apply").addEventListener("click", applyHdr);
  $("#hdr-group").addEventListener("click", hdrGroupSelected);
  $("#hdr-ungroup").addEventListener("click", hdrUngroupSelected);
  $("#hdr-look-btn").addEventListener("click", openLookModal);
  $("#hdr-look-close").addEventListener("click", closeLookModal);
  $("#hdr-look-cancel").addEventListener("click", closeLookModal);
  $("#hdr-look-apply").addEventListener("click", applyLook);
  $("#hdr-look-reset").addEventListener("click", resetLook);
  $("#hdr-look-prev").addEventListener("click", () => lookBracketNav(-1));
  $("#hdr-look-next").addEventListener("click", () => lookBracketNav(+1));
  $$("#hdr-look-modal input[type=range]").forEach((sl) => {
    sl.addEventListener("input", () => {
      const k = sl.dataset.look;
      lookEdit.look[k] = parseFloat(sl.value);
      $(`#hdr-look-modal .look-val[data-val="${k}"]`).textContent =
        parseFloat(sl.value).toFixed(2);
      fetchLookPreview(false);
    });
  });
  // Editor
  renderEditControls();
  bindEditControls();
  bindLookPanel();
  bindRepairPanel();
  bindOpticsPanel();
  bindPortraitPanel();
  bindTour();
  bindExplainer();
  bindMaskUi();
  bindSlots();
  bindWatermarkUi();
  curveInit();
  $("#edit-modal-close").addEventListener("click", () => leaveEditor("close"));
  $("#edit-cancel").addEventListener("click", () => leaveEditor("cancel"));
  $("#edit-save").addEventListener("click", saveEdit);
  $("#edit-auto").addEventListener("click", autoEdit);
  $("#edit-undo").addEventListener("click", editUndo);
  $("#edit-redo").addEventListener("click", editRedo);
  $("#edit-reset").addEventListener("click", resetEdit);
  $("#edit-prev").addEventListener("click", () => editNav(-1));
  $("#edit-next").addEventListener("click", () => editNav(+1));
  $("#edit-curve-reset").addEventListener("click", resetCurve);
  $("#modal-edit").addEventListener("click", () => {
    const i = state.modal.idx;
    closeModal();
    openEditModal(i);
  });
  const editCmp = $("#edit-compare");
  editCmp.addEventListener("pointerdown", (e) => { e.preventDefault(); editCompareOn(); });
  editCmp.addEventListener("pointerup", editCompareOff);
  editCmp.addEventListener("pointerleave", editCompareOff);
  $("#edit-preset-select").addEventListener("change", (e) => {
    if (e.target.value) applyPreset(e.target.value);
  });
  $("#preset-mode").addEventListener("click", (e) => {
    const b = e.target.closest("[data-preset-mode]");
    if (b) setPresetMode(b.dataset.presetMode);
  });
  try { setPresetMode(localStorage.getItem("pcls.presetMode") || "add"); }
  catch { setPresetMode("add"); }
  $("#edit-preset-save").addEventListener("click", saveCurrentAsPreset);
  $("#edit-preset-delete").addEventListener("click", deleteSelectedPreset);
  $("#edit-apply-more").addEventListener("click", () => openBulkModal({ fromEditor: true }));
  // Batch apply
  $("#selection-apply").addEventListener("click", () => openBulkModal({ scope: "selected" }));
  $("#selection-download").addEventListener("click", downloadSelection);
  $("#selection-clear").addEventListener("click", clearSelection);
  $("#bulk-modal-close").addEventListener("click", closeBulkModal);
  $("#bulk-cancel").addEventListener("click", closeBulkModal);
  $("#bulk-confirm").addEventListener("click", confirmBulkApply);
  $("#bulk-mode").addEventListener("click", (e) => {
    const b = e.target.closest("[data-bulk-mode]");
    if (!b) return;
    $$("#bulk-mode [data-bulk-mode]").forEach((x) => x.classList.toggle("active", x === b));
    updateBulkUi();
  });
  $$("#bulk-scope-cards .option-card").forEach((c) =>
    c.addEventListener("click", () => { setOptionCardValue("#bulk-scope-cards", c.dataset.value); updateBulkUi(); }));
  $$("#bulk-source-cards .option-card").forEach((c) =>
    c.addEventListener("click", () => {
      if (c.classList.contains("disabled")) return;
      setOptionCardValue("#bulk-source-cards", c.dataset.value);
      updateBulkUi();
    }));
  $("#bulk-preset-select").addEventListener("change", updateBulkUi);
  $("#people-cluster-btn").addEventListener("click", () => startCluster(null));
  bindClusterModal();
  $("#people-manage-btn").addEventListener("click", openPeopleModal);
  $("#people-modal-close").addEventListener("click", closePeopleModal);
  $("#people-save").addEventListener("click", savePeople);
  // No "are you sure": every decision is saved the moment it is made, so going
  // back to the projects loses nothing. What can stop it is a task still
  // running, and then it says which.
  $("#switch-project-btn").addEventListener("click", () => leaveProject(null));
  $$("#scene-mode-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      setOptionCardValue("#scene-mode-cards", card.dataset.value);
      $("#scene-gap-row").classList.toggle("hidden", card.dataset.value !== "time_gap");
      applySceneGrouping();
    });
  });
  $("#scene-gap").addEventListener("change", () => applySceneGrouping());
  $("#start-open").addEventListener("click", pickAndOpenProject);
  // Workspaces
  $("#workspace-select").addEventListener("change", (e) => switchWorkspace(e.target.value));
  $("#workspace-add").addEventListener("click", addWorkspace);
  $("#workspace-forget").addEventListener("click", forgetWorkspace);
  // Relink
  $("#relink-close").addEventListener("click", closeRelinkModal);
  $("#relink-cancel").addEventListener("click", closeRelinkModal);
  $("#relink-browse").addEventListener("click", () =>
    nativeBrowse($("#relink-new"), $("#relink-status"), "relink"));
  $("#export-browse").addEventListener("click", () =>
    nativeBrowse($("#export-target"), $("#export-status"), "export"));
  // Onboarding
  $("#onboard-use-default").addEventListener("click", () =>
    onboardChoose(workspaceState.defaultDir));
  $("#onboard-choose").addEventListener("click", onboardBrowse);
  $("#onboard-choose-again").addEventListener("click", onboardBrowse);
  $("#onboard-use-anyway").addEventListener("click", () => {
    if (onboardState.pending) commitOnboardWorkspace(onboardState.pending);
  });
  $("#relink-confirm").addEventListener("click", confirmRelink);
  $("#wizard-close").addEventListener("click", () => {
    if (confirm("Cancel project setup?")) closeWizard();
  });
  $("#wizard-back").addEventListener("click", () => {
    if (wizardState.step > 1) showWizardStep(wizardState.step - 1);
  });
  $("#wizard-next").addEventListener("click", async () => {
    if (wizardState.step === 1) {
      // The folder check is shown inline, where the path is, not in an alert.
      clearTimeout(wizardState.inspectTimer);
      if (!$("#wiz-photo-dir").value.trim()) {
        showWizPhotoInfo("error", "Choose the folder with your photos first.");
        return;
      }
      if (!wizardState.photoInfo) await inspectWizardPhotoDir();
      if (!wizardState.photoInfo?.usable) return;
    }
    const ok = validateWizardStep(wizardState.step);
    if (ok !== true) { alert(ok); return; }
    if (wizardState.step < 3) showWizardStep(wizardState.step + 1);
  });
  $("#wizard-create").addEventListener("click", createProject);
  $("#wiz-photo-browse").addEventListener("click", () => {
    $("#wiz-photo-info").className = "folder-info";
    nativeBrowse($("#wiz-photo-dir"), $("#wiz-photo-info"), "photos");
  });
  $("#wiz-photo-dir").addEventListener("input", () => {
    wizardState.photoInfo = null;
    clearTimeout(wizardState.inspectTimer);
    // Typing a path fires this per keystroke; wait for a pause.
    wizardState.inspectTimer = setTimeout(inspectWizardPhotoDir, 450);
  });
  $("#wiz-project-name").addEventListener("input", syncWizardTargetHint);
  // The slider and the box are one setting; either moves the other.
  $("#wiz-gap-range").addEventListener("input", (e) => {
    $("#wiz-gap").value = e.target.value;
    renderScenePreview();
  });
  $("#wiz-gap").addEventListener("input", (e) => {
    const v = parseInt(e.target.value, 10);
    if (v >= 1) $("#wiz-gap-range").value = Math.min(180, v);
    renderScenePreview();
  });
  $$("#wiz-scene-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      wizardState.sceneTouched = true;
      setWizardSceneMode(card.dataset.value);
    });
  });
  $$("#wiz-subject-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      wizardState.subjectPreset = card.dataset.value;
      $$("#wiz-subject-cards .option-card").forEach((c) =>
        c.classList.toggle("active", c === card));
    });
  });
  $("#undo-toast-btn").addEventListener("click", performUndo);
  $("#undo-toast-close").addEventListener("click", hideUndoToast);
}

// ---------- People modal ----------
function openPeopleModal() {
  if (!state.people.length) {
    alert("No people yet. Press group first.");
    return;
  }
  const list = $("#people-list");
  list.innerHTML = "";

  const activeSection = document.createElement("div");
  activeSection.className = "people-section";
  activeSection.innerHTML = `<h3 class="people-section-head">Active <span class="count-pill" id="active-count"></span><span class="hint">drag to reorder · top = highest priority</span></h3>`;
  const activeWrap = document.createElement("div");
  activeWrap.className = "people-active";
  activeSection.appendChild(activeWrap);
  list.appendChild(activeSection);

  const excludedSection = document.createElement("div");
  excludedSection.className = "people-section";
  excludedSection.innerHTML = `<h3 class="people-section-head">Excluded <span class="count-pill" id="excluded-count"></span><span class="hint">not used for sorting or filtering</span></h3>`;
  const excludedWrap = document.createElement("div");
  excludedWrap.className = "people-excluded";
  excludedSection.appendChild(excludedWrap);
  list.appendChild(excludedSection);

  const active = state.people.filter((p) => !p.excluded).sort((a, b) => a.priority - b.priority);
  const excluded = state.people.filter((p) => p.excluded).sort((a, b) => a.priority - b.priority);
  for (const p of active) activeWrap.appendChild(buildPersonCard(p, false));
  for (const p of excluded) excludedWrap.appendChild(buildPersonCard(p, true));

  bindPeopleDrag(activeWrap);
  refreshPeopleSections();
  $("#people-modal").classList.remove("hidden");
}

function buildPersonCard(person, isExcluded) {
  const card = document.createElement("div");
  card.className = "person-card" + (isExcluded ? " excluded" : "");
  card.dataset.id = person.id;
  card.dataset.excluded = isExcluded ? "1" : "0";
  card.draggable = !isExcluded;
  card.innerHTML = `
    <span class="drag-handle" title="Drag to reorder" aria-hidden="true"></span>
    <span class="priority-badge"></span>
    <img src="/face/${enc(person.ref.rel_path)}?idx=${person.ref.face_idx}" alt="" />
    <div class="fields">
      <input type="text" data-field="label" value="${escapeAttr(person.label)}" placeholder="Label" />
      <span class="count">${person.count} faces</span>
    </div>
    <button class="exclude-toggle" type="button">${isExcluded ? "Restore" : "Exclude"}</button>`;
  card.querySelector(".exclude-toggle").addEventListener("click", () => toggleExclude(card));
  return card;
}

function toggleExclude(card) {
  const activeWrap = $(".people-active");
  const excludedWrap = $(".people-excluded");
  const becomingExcluded = card.dataset.excluded === "0";
  card.dataset.excluded = becomingExcluded ? "1" : "0";
  card.classList.toggle("excluded", becomingExcluded);
  card.draggable = !becomingExcluded;
  card.querySelector(".exclude-toggle").textContent = becomingExcluded ? "Restore" : "Exclude";
  if (becomingExcluded) excludedWrap.appendChild(card);
  else activeWrap.appendChild(card);
  refreshPeopleSections();
}

function refreshPeopleSections() {
  const activeWrap = $(".people-active");
  const excludedWrap = $(".people-excluded");
  activeWrap.querySelectorAll(".person-card").forEach((card, i) => {
    card.querySelector(".priority-badge").textContent = `#${i + 1}`;
  });
  excludedWrap.querySelectorAll(".person-card").forEach((card) => {
    card.querySelector(".priority-badge").textContent = "—";
  });
  $("#active-count").textContent = activeWrap.children.length;
  $("#excluded-count").textContent = excludedWrap.children.length;
  // Empty-state placeholders so drop targets remain usable.
  setEmptyPlaceholder(activeWrap, "All clusters are excluded.");
  setEmptyPlaceholder(excludedWrap, "Nothing excluded.");
}

function setEmptyPlaceholder(wrap, text) {
  const hasCards = wrap.querySelector(".person-card");
  let ph = wrap.querySelector(".empty-placeholder");
  if (hasCards) { if (ph) ph.remove(); return; }
  if (!ph) {
    ph = document.createElement("div");
    ph.className = "empty-placeholder";
    wrap.appendChild(ph);
  }
  ph.textContent = text;
}

function bindPeopleDrag(container) {
  let dragSrc = null;
  container.addEventListener("dragstart", (e) => {
    const card = e.target.closest(".person-card");
    if (!card || card.dataset.excluded === "1") return;
    dragSrc = card;
    card.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    // Required by Firefox to actually start the drag.
    e.dataTransfer.setData("text/plain", card.dataset.id);
  });
  container.addEventListener("dragend", () => {
    if (dragSrc) dragSrc.classList.remove("dragging");
    dragSrc = null;
    refreshPeopleSections();
  });
  container.addEventListener("dragover", (e) => {
    if (!dragSrc) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "move";
    const target = e.target.closest(".person-card");
    if (!target || target === dragSrc) {
      // Allow dropping into empty container.
      if (e.target === container && !container.querySelector(".person-card")) {
        container.appendChild(dragSrc);
      }
      return;
    }
    const rect = target.getBoundingClientRect();
    const after = (e.clientY - rect.top) > rect.height / 2;
    if (after) target.after(dragSrc);
    else target.before(dragSrc);
    refreshPeopleSections();
  });
  container.addEventListener("drop", (e) => { e.preventDefault(); });
}

function closePeopleModal() {
  $("#people-modal").classList.add("hidden");
}

function escapeAttr(s) {
  return String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

async function savePeople() {
  const payload = { people: [] };
  const activeCards = $$(".people-active .person-card");
  const excludedCards = $$(".people-excluded .person-card");
  activeCards.forEach((card, i) => {
    const id = card.dataset.id;
    const label = card.querySelector('input[data-field="label"]').value.trim() || "(unnamed)";
    payload.people.push({ id, label, priority: i + 1, excluded: false });
  });
  excludedCards.forEach((card, i) => {
    const id = card.dataset.id;
    const label = card.querySelector('input[data-field="label"]').value.trim() || "(unnamed)";
    payload.people.push({ id, label, priority: 1000 + i, excluded: true });
  });
  const res = await fetch("/api/people", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Save failed: " + (err.detail || res.status));
    return;
  }
  const result = await res.json();
  state.people = result.people;
  state.peopleById = new Map(state.people.map((p) => [p.id, p]));
  prunePersonFilter();
  closePeopleModal();
  renderPeopleChips();
  recomputeFilter();
  renderMain();
}

// ---------- Cluster trigger ----------
// ---------- clustering ----------
// One run, two passes: faces into people, then detected subjects into look-alike
// groups. Both are configured from the same dialog and stored on the project, so
// re-grouping after a tweak does not mean re-scoring every photo.
const clusterEdit = { settings: null, defaults: null, available: false, classes: [] };

const CLUSTER_FIELDS = [
  ["cl-face-eps", "face_eps", 2],
  ["cl-face-min", "face_min_samples", 0],
  ["cl-subject-eps", "subject_eps", 2],
  ["cl-subject-min", "subject_min_samples", 0],
  ["cl-subject-area", "subject_min_area", 3],
  ["cl-subject-score", "subject_min_score", 2],
];

async function openClusterModal() {
  const res = await fetch("/api/cluster/settings", { cache: "no-store" });
  if (!res.ok) { alert("Could not read clustering settings: " + res.status); return; }
  const info = await res.json();
  clusterEdit.settings = { ...info.settings };
  clusterEdit.defaults = info.defaults;
  clusterEdit.available = !!info.subjects_available;
  clusterEdit.classes = info.subject_classes || [];
  renderClusterModal();
  $("#cluster-modal").classList.remove("hidden");
}

function closeClusterModal() { $("#cluster-modal").classList.add("hidden"); }

function renderClusterModal() {
  const v = clusterEdit.settings;
  for (const [id, key, dp] of CLUSTER_FIELDS) {
    $(`#${id}`).value = v[key];
    $(`#${id}-val`).textContent = key === "subject_min_area"
      ? `${(v[key] * 100).toFixed(1)}%`
      : Number(v[key]).toFixed(dp);
  }
  const on = clusterEdit.available && v.group_subjects;
  $("#cl-subject-on").checked = !!v.group_subjects;
  $("#cl-subject-on").disabled = !clusterEdit.available;
  $("#cl-subject-fields").classList.toggle("disabled", !on);
  $$("#cl-subject-fields input").forEach((i) => { i.disabled = !on; });
  $("#cl-subject-note").textContent = clusterEdit.available
    ? `Detected: ${clusterEdit.classes.join(", ")}. Appearance-based, so two
       same-colour, same-shape subjects will land together — rename or hide a
       group from the Subjects panel.`.replace(/\s+/g, " ")
    : "This project has no subject detection turned on, so there is nothing "
      + "detected to group. Turn it on from the Subjects panel (it re-scores).";
}

function bindClusterModal() {
  $("#cluster-settings-btn").addEventListener("click", openClusterModal);
  $("#cluster-modal-close").addEventListener("click", closeClusterModal);
  $("#cluster-cancel").addEventListener("click", closeClusterModal);
  $("#cluster-defaults").addEventListener("click", () => {
    clusterEdit.settings = { ...clusterEdit.defaults };
    renderClusterModal();
  });
  $("#cl-subject-on").addEventListener("change", (e) => {
    clusterEdit.settings.group_subjects = e.target.checked;
    renderClusterModal();
  });
  for (const [id, key] of CLUSTER_FIELDS) {
    $(`#${id}`).addEventListener("input", (e) => {
      clusterEdit.settings[key] = parseFloat(e.target.value);
      renderClusterModal();
    });
  }
  $("#cluster-run").addEventListener("click", () => {
    closeClusterModal();
    startCluster(clusterEdit.settings);
  });
}

async function startCluster(settings) {
  const res = await fetch("/api/cluster", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(settings || {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Grouping failed: " + (err.detail || res.status));
    return;
  }
  $("#score-title").textContent = "Grouping…";
  $("#score-progress").classList.remove("hidden");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollClusterStatus();
}

function pollClusterStatus() {
  if (scorePollTimer) clearInterval(scorePollTimer);
  let finished = false;
  const tick = async () => {
    if (finished) return;
    const res = await fetch("/api/cluster/status", { cache: "no-store" });
    if (!res.ok) { console.warn("cluster status fetch failed", res.status); return; }
    const s = await res.json();
    console.log("cluster status:", s);
    const pct = s.total ? Math.min(100, 100 * s.idx / s.total) : 0;
    $("#score-bar-fill").style.width = pct.toFixed(1) + "%";
    $("#score-progress-text").textContent = (s.phase || "running") +
      (s.total ? ` · ${s.idx}/${s.total}` : "");
    $("#score-current").textContent = "";
    if (!s.running) {
      finished = true;
      clearInterval(scorePollTimer);
      scorePollTimer = null;
      if (s.error) {
        $("#score-progress").classList.add("hidden");
        $("#score-title").textContent = "Scoring photos…";
        alert("Clustering failed: " + s.error);
        return;
      }
      const prevScene = state.selectedScene;
      await loadDb();
      // Refreshed here rather than when the panel is dismissed: the groups are
      // what was just asked for, and waiting for a click (or the 4-second
      // timeout) left the sidebar showing the previous run's answer.
      renderSidebar();
      renderPeopleChips();
      loadSubjects();
      $("#score-title").textContent = `Done · ${state.people.length} groups`;
      $("#score-bar-fill").style.width = "100%";
      $("#score-progress-text").textContent =
        `Total faces grouped: ${state.people.reduce((s, p) => s + p.count, 0)}`;
      $("#score-current").textContent = "Click anywhere to dismiss";
      const dismiss = () => {
        $("#score-progress").classList.add("hidden");
        $("#score-progress").removeEventListener("click", dismiss);
        $("#score-title").textContent = "Scoring photos…";
        renderSidebar();
        renderPeopleChips();
        loadSubjects();
        if (prevScene && state.byScene.has(prevScene)) selectScene(prevScene);
        else if (state.sceneOrder.length) selectScene(state.sceneOrder[0]);
        else renderMain();
      };
      $("#score-progress").addEventListener("click", dismiss);
      setTimeout(dismiss, 4000);
    }
  };
  tick();
  scorePollTimer = setInterval(tick, 200);
}

// ---------- Scoring trigger ----------
let scorePollTimer = null;

// The progress screen for opening a project and for re-scoring one: where in
// the run it is, how far, how long is left, and a way to stop. Other tasks use
// the same screen without steps or a Stop (they cannot be stopped midway);
// `data-kind` says which it is, and goes when the screen is hidden.
// The time left is the pace since this step began, so a slow start (models
// loading) does not stretch the estimate for the rest.
const scoreEta = { step: null, t0: 0, i0: 0 };

function showScoreProgress({ step, title, idx, total, current, message }) {
  // Before the photos are counted the scorer reports what it is doing in the
  // file's place ("Preparing RAW previews… (3/6)"): that is the status line.
  if (!total && current) { message = current; current = null; }
  const steps = ["scanning", "scoring", "clustering", "loading"];
  const at = steps.indexOf(step);
  $$("#score-progress .score-steps li").forEach((li, i) => {
    li.classList.toggle("done", at > i);
    li.classList.toggle("now", at === i);
  });
  $("#score-title").textContent = title;
  const pct = total ? Math.min(100, 100 * idx / total) : 0;
  $("#score-bar-fill").style.width = pct.toFixed(1) + "%";
  $("#score-bar-fill").parentElement.classList.toggle("busy", !total);
  $("#score-progress-text").textContent = total
    ? `${idx.toLocaleString()} of ${total.toLocaleString()}` : (message || "Starting…");
  const now = performance.now();
  if (scoreEta.step !== step || idx < scoreEta.i0) Object.assign(scoreEta, { step, t0: now, i0: idx });
  const done = idx - scoreEta.i0, secs = (now - scoreEta.t0) / 1000;
  let eta = "";
  if (total && done >= 3 && secs > 2) {
    const left = (total - idx) * secs / done;
    eta = left < 60 ? "under a minute left" : `about ${Math.round(left / 60)} min left`;
  }
  $("#score-eta").textContent = eta;
  $("#score-current").textContent = current ? basename(current) : "";
}

function startScoreProgress(kind) {
  $("#score-progress").dataset.kind = kind;
  $("#score-stop").disabled = false;
  setBtnLabel($("#score-stop"), "Stop");
  $("#score-eta").textContent = "";
  scoreEta.step = null;
  $("#score-progress").classList.remove("hidden");
}

function bindScoreProgress() {
  $("#score-stop").addEventListener("click", async () => {
    const url = $("#score-progress").dataset.kind === "rescore" ? "/api/score/cancel" : "/api/open/cancel";
    $("#score-stop").disabled = true;
    setBtnLabel($("#score-stop"), "Stopping…");
    const res = await fetch(url, { method: "POST" });
    if (!res.ok) throw new Error(`stop failed: ${res.status}`);
  });
  new MutationObserver(() => {
    const box = $("#score-progress");
    if (box.classList.contains("hidden")) { delete box.dataset.kind; $("#score-eta").textContent = ""; }
  }).observe($("#score-progress"), { attributes: true, attributeFilter: ["class"] });
}


function pollScoreStatus() {
  if (scorePollTimer) clearInterval(scorePollTimer);
  let finished = false;
  const tick = async () => {
    if (finished) return;
    const res = await fetch("/api/score/status", { cache: "no-store" });
    if (!res.ok) { console.warn("score status fetch failed", res.status); return; }
    const s = await res.json();
    const grouping = s.phase === "grouping";
    showScoreProgress({
      step: grouping ? "clustering" : "scoring",
      title: grouping ? "Grouping faces and subjects…" : `Re-scoring ${plural(state.photos.length, "photo")}…`,
      idx: s.idx, total: s.total, current: grouping ? null : s.current,
      message: grouping ? s.current : null,
    });
    if (!s.running) {
      finished = true;
      clearInterval(scorePollTimer);
      scorePollTimer = null;
      if (s.cancelled) {
        // Stopped while scoring: the project is as it was. While grouping:
        // scored, with the groups still to make (the People panel's group).
        $("#score-progress").classList.add("hidden");
        await loadDb();
        renderSidebar();
        renderPeopleChips();
        renderMain();
        return;
      }
      if (s.error) {
        $("#score-progress").classList.add("hidden");
        alert("Scoring failed: " + s.error);
        return;
      }
      const prevScene = state.selectedScene;
      await loadDb();
      const totalFaces = state.photos.reduce((s, p) => s + (p.faces?.length || 0), 0);
      const groups = state.subjects.vehicles.length;
      $("#score-title").textContent = `Scored ${state.photos.length} photos`;
      $("#score-bar-fill").style.width = "100%";
      // Scoring throws the groups away and rebuilds them in the same run, so the
      // summary reports both rather than telling anyone to press another button.
      $("#score-progress-text").textContent =
        `${totalFaces} faces · ${state.people.length} people`
        + (groups ? ` · ${groups} subject groups` : "");
      $("#score-current").textContent = "Click anywhere to dismiss";
      $("#score-eta").textContent = "";
      delete $("#score-progress").dataset.kind;   // done: no steps, no Stop
      const dismiss = () => {
        $("#score-progress").classList.add("hidden");
        $("#score-progress").removeEventListener("click", dismiss);
        $("#score-title").textContent = "Scoring photos…";
        renderSidebar();
        renderPeopleChips();
        loadSubjects();          // the grouping pass rewrote these
        if (prevScene && state.byScene.has(prevScene)) selectScene(prevScene);
        else if (state.sceneOrder.length) selectScene(state.sceneOrder[0]);
        else renderMain();
      };
      $("#score-progress").addEventListener("click", dismiss);
      setTimeout(dismiss, 4000);
    }
  };
  tick();
  scorePollTimer = setInterval(tick, 250);
}

// ---------- landing / project switching ----------
async function fetchState() {
  const res = await fetch("/api/state", { cache: "no-store" });
  if (!res.ok) throw new Error("state fetch failed: " + res.status);
  return res.json();
}

// ---------- workspaces ----------
// `firstRun` is the server saying no workspace has ever been chosen; the list
// then holds only the default it suggests, which does not exist on disk yet.
const workspaceState = { current: null, list: [], firstRun: false, defaultDir: null, sep: "/" };

async function loadWorkspaces() {
  try {
    const res = await fetch("/api/workspaces", { cache: "no-store" });
    if (res.ok) {
      const d = await res.json();
      workspaceState.list = d.workspaces || [];
      workspaceState.current = d.current || (workspaceState.list[0] || null);
      workspaceState.firstRun = !!d.first_run;
      workspaceState.defaultDir = d.default || null;
      workspaceState.sep = d.sep || "/";
    }
  } catch {}
  renderOnboarding();
  renderWorkspaceSelect();
  loadWorkspaceProjects();
}

async function inspectFolder(path, workspace = null) {
  const res = await fetch("/api/folder/inspect", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, workspace }),
  });
  if (!res.ok) throw new Error("folder inspect failed: " + res.status);
  return res.json();
}

function plural(n, word) {
  return `${n.toLocaleString()} ${word}${n === 1 ? "" : "s"}`;
}

// Why a folder makes a poor workspace, or null if it is fine. The mistake this
// is for is choosing the photo folder itself, which would put every project
// among the photos.
function workspaceProblem(info) {
  if (info.is_project) {
    return "This is a single project's folder. Choose the folder that holds "
      + "your projects instead, such as the one it sits in.";
  }
  const n = info.photos + info.raws;
  if (n > 0 && info.projects === 0) {
    return `This folder already holds ${info.truncated ? "at least " : ""}`
      + `${plural(n, "photo")}. The projects folder should be separate from your `
      + "photos, since the app writes its own files into it. An empty or new "
      + "folder is best.";
  }
  return null;
}

// ---------- how projects work: an animated explainer ----------
// Workspaces and projects are the one idea a new user has to get before the app
// is usable, and a paragraph did not carry it. This plays it out as a small
// animated scene: the photo folder, the workspace beside it, a project pointing
// back at the photos, the work flowing into the project, and what deleting and
// exporting do. Each element carries the scenes it is shown in (data-from,
// data-until); CSS transitions do the moving.
const EXPLAINER_SCENES = [
  { title: "Your photos stay where they are",
    text: "In a folder on your computer, a memory card or an external drive. The app reads them and never moves, renames or changes them." },
  { title: "A workspace is a folder for projects",
    text: "One folder the app writes its own files into, kept apart from your photos. By default it is PictureClassifier-Projects in your home folder." },
  { title: "Each shoot is a project",
    text: "A project points at the folder of one shoot. Nothing is copied: it only remembers where the photos are." },
  { title: "The project keeps your work",
    text: "Your picks, edits, thumbnails and the people it found are saved inside the project, so the photo folder stays exactly as it was." },
  { title: "More shoots, more projects",
    text: "Every project goes into the same workspace, so the start screen lists all of them. Open one and carry on where you left off." },
  { title: "Deleting a project is safe",
    text: "Only the project's folder is set aside (renamed, so it can be brought back). The photos it pointed at are not touched." },
  { title: "When you are done, export",
    text: "Export picks copies the photos you chose, with your edits applied, into a folder of your choice. The originals stay as they were." },
];
const EXPLAINER_STEP_MS = 5200;

const EXPLAINER_HTML = `
<div class="explainer" data-scene="0">
  <div class="ex-stage">
    <div class="ex-label ex-label-drive" data-from="0">On your drive</div>
    <div class="ex-label ex-label-ws" data-from="1">Workspace</div>

    <div class="ex-folder ex-photos ex-photos-a" data-from="0">
      <div class="ex-name">${icon("folder")}<span>2026-05-wedding</span></div>
      <div class="ex-thumbs">${"<i></i>".repeat(6)}</div>
      <div class="ex-badge ex-lock" data-from="3">${icon("check")}never changed</div>
    </div>
    <div class="ex-folder ex-photos ex-photos-b" data-from="4" data-until="5">
      <div class="ex-name">${icon("folder")}<span>2026-06-trip</span></div>
      <div class="ex-thumbs">${"<i></i>".repeat(4)}</div>
      <div class="ex-badge ex-lock" data-from="5" data-until="5">${icon("check")}untouched</div>
    </div>
    <div class="ex-folder ex-exports" data-from="6">
      <div class="ex-name">${icon("download")}<span>Exports</span></div>
      <div class="ex-thumbs">${"<i></i>".repeat(3)}</div>
    </div>

    <div class="ex-ws" data-from="1">
      <div class="ex-name">${icon("folder")}<span>PictureClassifier-Projects</span></div>
      <div class="ex-project ex-project-a" data-from="2">
        <div class="ex-name">${icon("open")}<span>2026-05-wedding</span></div>
        <div class="ex-chips">
          <span data-from="3">${icon("check")}picks</span>
          <span data-from="3">${icon("pencil")}edits</span>
          <span data-from="3">${icon("boxes")}thumbnails</span>
          <span data-from="3">${icon("person")}people</span>
        </div>
      </div>
      <div class="ex-project ex-project-b" data-from="4" data-until="5">
        <div class="ex-name">${icon("open")}<span class="ex-live">2026-06-trip</span><span class="ex-gone">2026-06-trip.deleted</span></div>
      </div>
    </div>

    <svg class="ex-links" viewBox="0 0 600 300" preserveAspectRatio="none" aria-hidden="true">
      <defs><marker id="ex-arrow-@" class="ex-mark" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" /></marker>
        <marker id="ex-arrow-out-@" class="ex-mark-out" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0,0 L8,4 L0,8 z" /></marker></defs>
      <path class="ex-link ex-link-a" data-from="2" d="M352,86 C300,86 290,78 244,78" marker-end="url(#ex-arrow-@)" />
      <path class="ex-link ex-link-b" data-from="4" data-until="4" d="M352,208 C300,208 290,214 244,214" marker-end="url(#ex-arrow-@)" />
      <path class="ex-link ex-link-out" data-from="6" d="M352,112 C300,150 290,200 244,214" marker-end="url(#ex-arrow-out-@)" />
    </svg>
    <div class="ex-tag ex-tag-a" data-from="2" data-until="3">points to</div>
    <div class="ex-tag ex-tag-out" data-from="6">copies picks</div>
  </div>
  <div class="ex-caption">
    <div class="ex-step"></div>
    <h4 class="ex-title"></h4>
    <p class="ex-text"></p>
  </div>
  <div class="ex-controls">
    <button type="button" class="ex-back" aria-label="Previous">${icon("left")}</button>
    <span class="ex-dots"></span>
    <button type="button" class="ex-next" aria-label="Next">${icon("right")}</button>
    <span class="spacer"></span>
    <button type="button" class="ex-play quiet"></button>
  </div>
</div>`;

let explainerCount = 0;

function mountExplainer(host, { autoplay = true } = {}) {
  // Marker ids per instance: the landing and the dialog can both hold one, and
  // a url(#id) that resolves into the hidden one draws no arrowhead at all.
  host.innerHTML = EXPLAINER_HTML.replaceAll("-@", `-${++explainerCount}`);
  const root = host.querySelector(".explainer");
  const items = [...root.querySelectorAll("[data-from]")];
  const last = EXPLAINER_SCENES.length - 1;
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let scene = 0, timer = null;
  root.querySelector(".ex-dots").innerHTML = EXPLAINER_SCENES.map((_, i) =>
    `<button type="button" data-scene="${i}" aria-label="Scene ${i + 1}"></button>`).join("");

  const show = (n) => {
    scene = Math.max(0, Math.min(last, n));
    root.dataset.scene = scene;
    for (const el of items) {
      const from = Number(el.dataset.from), until = el.dataset.until == null ? last : Number(el.dataset.until);
      el.classList.toggle("on", scene >= from && scene <= until);
    }
    root.classList.toggle("deleted", scene === 5);
    const sc = EXPLAINER_SCENES[scene];
    root.querySelector(".ex-step").textContent = `${scene + 1} / ${last + 1}`;
    root.querySelector(".ex-title").textContent = sc.title;
    root.querySelector(".ex-text").textContent = sc.text;
    root.querySelectorAll(".ex-dots button").forEach((b, i) => b.classList.toggle("on", i === scene));
    root.querySelector(".ex-back").disabled = scene === 0;
    root.querySelector(".ex-next").disabled = scene === last;
  };
  const stop = () => {
    clearInterval(timer);
    timer = null;
    root.querySelector(".ex-play").textContent = scene === last ? "Replay" : "Play";
  };
  const play = () => {
    if (scene === last) show(0);
    clearInterval(timer);
    timer = setInterval(() => { if (scene >= last) stop(); else show(scene + 1); }, EXPLAINER_STEP_MS);
    root.querySelector(".ex-play").textContent = "Pause";
  };
  root.querySelector(".ex-back").addEventListener("click", () => { stop(); show(scene - 1); });
  root.querySelector(".ex-next").addEventListener("click", () => { stop(); show(scene + 1); });
  root.querySelector(".ex-dots").addEventListener("click", (e) => {
    const b = e.target.closest("[data-scene]");
    if (b) { stop(); show(Number(b.dataset.scene)); }
  });
  root.querySelector(".ex-play").addEventListener("click", () => (timer ? stop() : play()));
  show(0);
  // The first frame is drawn with nothing on, then the first scene switches
  // on, so even scene 1 animates in rather than appearing.
  items.forEach((el) => el.classList.remove("on"));
  requestAnimationFrame(() => requestAnimationFrame(() => show(0)));
  if (autoplay && !still) play(); else stop();
  return { stop, play, show };
}

let modalExplainer = null;

function openExplainer() {
  $("#explainer-modal").classList.remove("hidden");
  modalExplainer = mountExplainer($("#modal-explainer"));
}

function closeExplainer() {
  if (modalExplainer) modalExplainer.stop();
  modalExplainer = null;
  $("#explainer-modal").classList.add("hidden");
}

function bindExplainer() {
  $$(".explainer-open").forEach((b) => b.addEventListener("click", (e) => {
    e.preventDefault();
    openExplainer();
  }));
  $("#explainer-close").addEventListener("click", closeExplainer);
  $("#explainer-modal").addEventListener("click", (e) => {
    if (e.target.id === "explainer-modal") closeExplainer();
  });
}

// ---------- first-launch onboarding ----------
const onboardState = { pending: null };

let onboardExplainer = null;

function renderOnboarding() {
  const first = workspaceState.firstRun;
  $("#onboard").classList.toggle("hidden", !first);
  $("#workspace-card").classList.toggle("hidden", first);
  $("#projects-card").classList.toggle("hidden", first);
  $("#landing-new-project").classList.toggle("hidden", first);
  if (!first) {
    if (onboardExplainer) { onboardExplainer.stop(); onboardExplainer = null; }
    return;
  }
  if (!onboardExplainer) onboardExplainer = mountExplainer($("#onboard-explainer"));
  $("#onboard-default").textContent = workspaceState.defaultDir || "";
  $("#onboard-warning").classList.add("hidden");
  $("#onboard-choice").classList.remove("hidden");
  $("#onboard-status").textContent = "";
}

async function onboardBrowse() {
  const status = $("#onboard-status");
  status.textContent = BROWSE_WAITING;
  const res = await fetch("/api/browse-folder", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ initial: dirname(workspaceState.defaultDir) || null, purpose: "workspace" }),
  });
  const result = res.ok ? await res.json() : {};
  status.textContent = result.error ? "Browse failed: " + result.error : "";
  if (result.path) await onboardChoose(result.path);
}

async function onboardChoose(dir) {
  if (!dir) return;
  const info = await inspectFolder(dir);
  const problem = workspaceProblem(info);
  if (problem) {
    onboardState.pending = dir;
    $("#onboard-warning-text").textContent = problem;
    $("#onboard-warning-path").textContent = info.path;
    $("#onboard-warning").classList.remove("hidden");
    $("#onboard-choice").classList.add("hidden");
    return;
  }
  await commitOnboardWorkspace(dir, info.projects);
}

async function commitOnboardWorkspace(dir, existingProjects = 0) {
  const status = $("#onboard-status");
  const res = await fetch("/api/workspaces", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dir }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    status.textContent = "Could not use that folder: " + (err.detail || res.status);
    return;
  }
  onboardState.pending = null;
  await loadWorkspaces();
  // A folder that already has projects was chosen to get back to them, so the
  // list is the next step; otherwise it is making the first one.
  if (!existingProjects) openWizard({ first: true });
}

function renderWorkspaceSelect() {
  const sel = $("#workspace-select");
  sel.innerHTML = workspaceState.list
    .map((w) => `<option value="${escapeAttr(w)}">${escapeHtml(basename(w) || w)}</option>`)
    .join("");
  if (workspaceState.current) sel.value = workspaceState.current;
  $("#workspace-path").textContent = workspaceState.current || "";
  setBtnLabel($("#workspace-btn"), workspaceState.current ? basename(workspaceState.current) : "Workspace");
  $("#workspace-btn").title = `Workspace: ${workspaceState.current || "none"}`;
  $("#workspace-forget").style.display = workspaceState.list.length > 1 ? "" : "none";
}

async function switchWorkspace(dir) {
  if (!dir) return;
  await fetch("/api/workspaces/current", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dir }),
  }).catch(() => {});
  workspaceState.current = dir;
  $("#workspace-path").textContent = dir;
  setBtnLabel($("#workspace-btn"), basename(dir));
  $("#workspace-btn").title = `Workspace: ${dir}`;
  loadWorkspaceProjects();
}

async function addWorkspace() {
  const status = $("#landing-status");
  status.textContent = BROWSE_WAITING;
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial: workspaceState.current || null, purpose: "workspace" }),
    });
    const result = res.ok ? await res.json() : {};
    if (!result.path) { status.textContent = ""; return; }
    const problem = workspaceProblem(await inspectFolder(result.path));
    if (problem && !confirm(`${problem}\n\n${result.path}\n\nUse it anyway?`)) {
      status.textContent = "";
      return;
    }
    const add = await fetch("/api/workspaces", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dir: result.path }),
    });
    if (!add.ok) { status.textContent = "Could not add workspace."; return; }
    const d = await add.json();
    workspaceState.list = d.workspaces || [];
    workspaceState.current = d.current;
    status.textContent = "";
    renderWorkspaceSelect();
    loadWorkspaceProjects();
  } catch (e) { status.textContent = "Error: " + e.message; }
}

async function forgetWorkspace() {
  if (!workspaceState.current) return;
  if (!confirm(`Remove this workspace from the list?\n${workspaceState.current}\n\n(The folder and its projects are NOT deleted.)`)) return;
  const res = await fetch("/api/workspaces/forget", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dir: workspaceState.current }),
  });
  if (!res.ok) return;
  const d = await res.json();
  workspaceState.list = d.workspaces || [];
  workspaceState.current = d.current;
  renderWorkspaceSelect();
  loadWorkspaceProjects();
}

async function loadWorkspaceProjects() {
  const wrap = $("#workspace-projects");
  wrap.innerHTML = `<div class="recents-empty">Loading…</div>`;
  let projects = [];
  try {
    const q = workspaceState.current ? `?workspace=${encodeURIComponent(workspaceState.current)}` : "";
    const res = await fetch("/api/workspaces/projects" + q, { cache: "no-store" });
    if (res.ok) projects = (await res.json()).projects || [];
  } catch {}
  renderProjectGrid(projects);
}

// "3 days ago", for when a project was last opened.
function sinceText(iso) {
  if (!iso) return "Not opened yet";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return "Opened just now";
  const m = s / 60, h = m / 60, d = h / 24;
  if (m < 60) return `Opened ${Math.round(m)} min ago`;
  if (h < 24) return `Opened ${plural(Math.round(h), "hour")} ago`;
  if (d < 2) return "Opened yesterday";
  if (d < 30) return `Opened ${Math.round(d)} days ago`;
  return `Opened ${new Date(iso).toLocaleDateString()}`;
}

let lastProjects = [];

function renderProjectGrid(projects) {
  lastProjects = projects;
  const wrap = $("#workspace-projects");
  wrap.innerHTML = "";
  $("#workspace-proj-count").textContent = projects.length || "";
  if (!projects.length) {
    wrap.innerHTML = `<div class="project-empty">
      <p>No projects here yet. Make one per shoot: choose the folder its photos
      are in, and scoring starts straight away.</p>
      <button type="button" class="primary" id="project-empty-new">${icon("plus")} New project</button>
    </div>`;
    $("#project-empty-new").addEventListener("click", () => openWizard());
    return;
  }
  const by = $("#projects-sort").value;
  const sorted = projects.slice().sort((a, b) =>
    by === "name" ? a.name.localeCompare(b.name)
    : by === "photos" ? b.photos - a.photos
    : (b.opened_at || "").localeCompare(a.opened_at || "") || a.name.localeCompare(b.name));

  for (const p of sorted) {
    const card = document.createElement("div");
    card.className = "project-card" + (p.photos_exist ? "" : " missing");
    const pct = p.photos ? (100 * p.decided / p.photos) : 0;
    const counts = p.scored_at
      ? `${plural(p.photos, "photo")}${p.picks ? ` · ${plural(p.picks, "pick")}` : ""}`
      : "Not scored yet";
    const cover = p.cover
      ? `<img loading="lazy" alt="" src="/api/projects/cover?project_dir=${encodeURIComponent(p.project_dir)}&rel=${encodeURIComponent(p.cover)}" />`
      : `<span class="project-cover-empty">${icon("folder")}</span>`;
    card.innerHTML = `
      <span class="project-cover">${cover}</span>
      <span class="project-body">
        <span class="project-name">${escapeHtml(p.name)}</span>
        <span class="project-meta">${counts}</span>
        <span class="project-progress" title="${p.decided} of ${p.photos} decided"><i style="width:${pct.toFixed(1)}%"></i></span>
        <span class="project-foot"><span>${p.decided} of ${p.photos} decided</span><span>${sinceText(p.opened_at)}</span></span>
        <span class="project-path" title="${escapeHtml(p.photo_dir || "")}">${icon("folder")} ${escapeHtml(basename(p.photo_dir || ""))}</span>
        ${p.photos_exist ? "" : `<span class="project-missing">${icon("warning")} Photos not found. Click to re-link.</span>`}
      </span>
      <button class="project-del" type="button" aria-label="Delete this project"
        title="Delete this project (photos are kept)">${icon("trash")}</button>`;
    card.addEventListener("click", () => openProjectByDir(p.project_dir));
    card.querySelector(".project-del").addEventListener("click", (e) => {
      e.stopPropagation();   // don't open the project we're deleting
      deleteProject(p);
    });
    wrap.appendChild(card);
  }
}

async function deleteProject(p) {
  const ok = confirm(
    `Delete the project "${p.name}"?\n\n` +
    `Your photos are NOT touched — nothing under\n${p.photo_dir}\nis modified or removed.\n\n` +
    `The project folder is renamed to "${p.name}.deleted-…" and disappears from ` +
    `this list. Decisions and edits stay inside it, so you can restore the ` +
    `project later by renaming the folder back.`
  );
  if (!ok) return;
  const res = await fetch("/api/projects/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: p.project_dir }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Delete failed: " + (err.detail || res.status));
    return;
  }
  const { name } = await res.json();
  await loadWorkspaceProjects();
  await renderRecents();
  toastProjectDeleted(p.name, name);
}

// A quiet confirmation that says where the folder went — the undo is a rename,
// so the new name is the only thing the user needs.
function toastProjectDeleted(oldName, newName) {
  const el = $("#landing-toast");
  el.textContent = `Deleted "${oldName}" — folder renamed to ${newName}`;
  el.classList.remove("hidden");
  clearTimeout(toastProjectDeleted.timer);
  toastProjectDeleted.timer = setTimeout(() => el.classList.add("hidden"), 8000);
}

// ---------- relink ----------
const relinkState = { projectDir: null, dbPath: null };

function openRelinkModal(info) {
  relinkState.projectDir = info.project_dir || null;
  relinkState.dbPath = info.db_path || null;
  $("#relink-old").textContent =
    (info.photo_dir || "") + (info.jpeg_subdir ? " / " + info.jpeg_subdir : "");
  $("#relink-new").value = "";
  $("#relink-status").textContent = "";
  $("#relink-modal").classList.remove("hidden");
}

function closeRelinkModal() { $("#relink-modal").classList.add("hidden"); }

async function confirmRelink() {
  const newDir = $("#relink-new").value.trim();
  if (!newDir) { $("#relink-status").textContent = "Pick the new photo folder."; return; }
  $("#relink-confirm").disabled = true;
  $("#relink-status").textContent = "Re-linking…";
  const res = await fetch("/api/project/relink", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      project_dir: relinkState.projectDir,
      db_path: relinkState.dbPath,
      new_photo_dir: newDir,
    }),
  });
  $("#relink-confirm").disabled = false;
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#relink-status").textContent = "Failed: " + (err.detail || res.status);
    return;
  }
  const rep = await res.json();
  closeRelinkModal();
  if (relinkState.projectDir) openProjectByDir(relinkState.projectDir);
  else if (relinkState.dbPath) openLegacyDb({ db_path: relinkState.dbPath, photo_dir: newDir, jpeg_subdir: "" });
}

function showLanding() {
  document.body.classList.add("landing-mode");
  $("#landing").classList.remove("hidden");
  loadWorkspaces();
  renderRecents();
}

function hideLanding() {
  document.body.classList.remove("landing-mode");
  $("#landing").classList.add("hidden");
}

async function renderRecents() {
  const wrap = $("#landing-recents");
  wrap.innerHTML = "";
  let recents = [];
  try {
    const res = await fetch("/api/recents", { cache: "no-store" });
    if (res.ok) recents = (await res.json()).recents || [];
  } catch {}
  if (!recents.length) {
    wrap.innerHTML = `<div class="recents-empty">No recent projects yet.</div>`;
    return;
  }
  for (const r of recents) {
    const item = document.createElement("div");
    item.className = "recent-item";
    const opened = r.opened_at ? new Date(r.opened_at).toLocaleString() : "—";
    const isProject = r.kind === "project" && r.project_dir;
    const name = r.name || basename(r.photo_dir || r.project_dir || "");
    const pathLine = isProject
      ? `<div class="recent-path">${icon("open")} ${escapeHtml(r.project_dir)}</div>
         <div class="recent-path dim">photos: ${escapeHtml(r.photo_dir || "")}${r.jpeg_subdir ? ` / ${escapeHtml(r.jpeg_subdir)}` : ""}</div>`
      : `<div class="recent-path">${escapeHtml(r.photo_dir || "")}${r.jpeg_subdir ? ` <span class="dim">/ ${escapeHtml(r.jpeg_subdir)}</span>` : ""}</div>`;
    item.innerHTML = `
      <button class="recent-open" type="button">
        <div class="recent-name">${isProject ? "" : "<span class='legacy-tag'>legacy</span> "}${escapeHtml(name)}</div>
        ${pathLine}
        <div class="recent-meta">${opened}</div>
      </button>
      <button class="recent-forget" type="button" title="Remove from recents">×</button>`;
    item.querySelector(".recent-open").addEventListener("click", () => {
      if (isProject) openProjectByDir(r.project_dir);
      else openLegacyDb(r);
    });
    item.querySelector(".recent-forget").addEventListener("click", async () => {
      await fetch("/api/recents/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_dir: isProject ? r.project_dir : null,
          db_path: r.db_path || null,
        }),
      });
      renderRecents();
    });
    wrap.appendChild(item);
  }
}

// Paths arrive in the server's own form, so on Windows they use backslashes;
// anything that takes one apart has to accept either separator. Splitting on
// "/" alone turned a Windows photo folder into a project named after its whole
// path, "C__Users_…".
function basename(p) {
  if (!p) return "";
  const s = String(p).replace(/[\\/]+$/, "");
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i >= 0 ? s.slice(i + 1) : s;
}
function dirname(p) {
  const s = String(p || "").replace(/[\\/]+$/, "");
  const i = Math.max(s.lastIndexOf("/"), s.lastIndexOf("\\"));
  return i > 0 ? s.slice(0, i) : s;
}
function joinPath(dir, name) {
  const sep = workspaceState.sep || "/";
  return String(dir).replace(/[\\/]+$/, "") + sep + name;
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// The dialog belongs to the server process, not the browser, and on Windows it
// can open behind the browser window. Saying so while it is open is what keeps
// "nothing happened" from being the first impression.
const BROWSE_WAITING = "A folder window is open. If you can't see it, it may be "
  + "behind this browser window; check the taskbar or Dock.";

async function nativeBrowse(targetInput, status, purpose = "photos") {
  const initial = targetInput.value.trim() || null;
  if (status) status.textContent = BROWSE_WAITING;
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial, purpose }),
    });
    if (!res.ok) {
      if (status) status.textContent = `Browse endpoint failed (${res.status}).`;
      return;
    }
    const result = await res.json();
    if (result.path) {
      targetInput.value = result.path;
      targetInput.dispatchEvent(new Event("input", { bubbles: true }));
    }
    if (status) {
      if (result.error) status.textContent = "Browse failed: " + result.error;
      else status.textContent = "";
    }
  } catch (err) {
    if (status) status.textContent = "Browse network error: " + err.message;
  }
}

function showOpenProgress(title) {
  hideLanding();
  closeWizard();
  $("#score-title").textContent = title;
  startScoreProgress("open");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollOpenStatus();
}

async function openProjectByDir(projectDir) {
  $("#landing-status").textContent = "Opening project…";
  const res = await fetch("/api/project/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ project_dir: projectDir }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#landing-status").textContent = "Failed: " + (err.detail || res.status);
    return;
  }
  showOpenProgress("Opening project…");
}

async function openLegacyDb(entry) {
  $("#landing-status").textContent = "Opening project…";
  const res = await fetch("/api/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      photo_dir: entry.photo_dir,
      jpeg_subdir: entry.jpeg_subdir || "",
      db_path: entry.db_path || null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#landing-status").textContent = "Failed: " + (err.detail || res.status);
    return;
  }
  showOpenProgress("Opening project…");
}

// ---------- inline help ----------
const HELP_CONTENT = {
  "scene-grouping": `
    <h3>How scenes are grouped</h3>
    <p><b>By folder</b> — each subdirectory under your photo folder
    becomes one scene. Use this if you organized shots into folders.</p>
    <p><b>By time gap</b> — reads EXIF capture time and starts a new scene
    after a configurable gap. Good for an unsorted dump.</p>
    <p>You can switch any time without re-scoring; only auto-suggestions are
    recomputed within each new scene.</p>`,
  "people": `
    <h3>Drag to set priority, click <i>Exclude</i> to ignore</h3>
    <p>The top of the <b>Active</b> list is the highest priority. Photos
    containing higher-priority people sort to the top within each scene.</p>
    <p>Clicking <b>Exclude</b> on a cluster moves it to the Excluded section.
    Excluded clusters are ignored for sorting and filtering, and their face
    crops are hidden in the grid — but the photos themselves still show.</p>
    <p>Re-grouping resets labels and priorities because
    face indices change. Decisions per photo are kept.</p>`,
  "hdr": `
    <h3>HDR brackets</h3>
    <p>Auto-exposure brackets are detected from EXIF — frames shot close
    together, at different exposures, of the same composition — and merged
    into one photo with exposure fusion.</p>
    <p>A merged result is scored and culled as a single photo; its source
    frames stay on disk but are not shown in the grid.</p>
    <p>If detection got something wrong, select frames and <b>Group</b> or
    <b>Ungroup</b> them, then <b>Apply</b> to re-merge and re-score. A bracket
    holds 3–9 frames.</p>`,
};

function openHelp(key, anchor) {
  const popover = $("#help-popover");
  const body = $("#help-popover-body");
  body.innerHTML = HELP_CONTENT[key] || `<p>No help written for "${key}".</p>`;
  popover.classList.remove("hidden");
  // Position near the anchor; clamp into viewport.
  const rect = anchor.getBoundingClientRect();
  const padding = 12;
  popover.style.left = "auto";
  popover.style.right = "auto";
  popover.style.top = (rect.bottom + 6) + "px";
  popover.style.left = Math.max(padding, rect.left - 80) + "px";
  popover.style.maxWidth = "min(360px, calc(100vw - 24px))";
  // If overflowing on the right, nudge back.
  const popRect = popover.getBoundingClientRect();
  if (popRect.right > window.innerWidth - padding) {
    popover.style.left = (window.innerWidth - popRect.width - padding) + "px";
  }
  // And off the bottom: flip above the anchor, or pin to the top if neither fits.
  if (popRect.bottom > window.innerHeight - padding) {
    const above = rect.top - 6 - popRect.height;
    popover.style.top = Math.max(padding, above) + "px";
  }
}

function closeHelp() {
  $("#help-popover").classList.add("hidden");
}

function bindHelp() {
  document.addEventListener("click", (e) => {
    const btn = e.target.closest(".help-btn");
    if (btn) {
      e.preventDefault();
      e.stopPropagation();
      openHelp(btn.dataset.help, btn);
      return;
    }
    if (!e.target.closest("#help-popover")) closeHelp();
  });
  $("#help-popover-close").addEventListener("click", closeHelp);
}

// ---------- quit ----------
async function quitApp() {
  if (!confirm("Quit Picture Classifier?\n\nYour decisions and saved edits are kept.")) return;
  await decisionSaves;
  await flushViewSave();
  const post = (force) => fetch("/api/quit", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ force }),
  });
  let res = await post(false);
  if (res.status === 409) {
    const err = await res.json();
    const what = err.detail.charAt(0).toUpperCase() + err.detail.slice(1);
    if (!confirm(`${what}. Quit anyway? It stops where it is.`)) return;
    res = await post(true);
  }
  if (!res.ok) { alert("Could not quit: " + res.status); return; }
  $("#quit-screen").classList.remove("hidden");
}


// ---------- keyboard shortcuts ----------
// One list of every key, by where it works. The sheet (?) shows all of it; the
// bar along the bottom of each screen shows the entries marked `bar`, so the
// keys for what you are doing are always in sight.
const MOD = IS_MAC ? "⌘" : "Ctrl";
const KEYMAP = {
  grid: [
    { group: "Decide", note: "Acts on the highlighted photo, then moves to the next.", keys: [
      { k: ["P"], alt: "A", label: "Pick", bar: true },
      { k: ["R"], label: "Reject", bar: true },
      { k: ["V"], label: "Review", bar: true },
      { k: ["U"], label: "Clear the decision", bar: "Clear" },
      { k: [MOD, "Z"], label: "Undo a decision made on many photos at once" },
    ]},
    { group: "Rate and label", note: "Kept apart from the decision, and written into exports.", keys: [
      { k: ["1", "–", "5"], label: "Stars", bar: "Stars" },
      { k: ["0"], label: "No stars" },
      { k: ["6", "7", "8", "9"], label: "Red, yellow, green, blue label (again to remove)" },
    ]},
    { group: "Move", keys: [
      { k: ["←", "→"], label: "Previous / next photo", bar: "Move" },
      { k: ["↑", "↓"], label: "Row up / down" },
      { k: ["[", "]"], alt: "PgUp PgDn", label: "Previous / next page", bar: "Page" },
    ]},
    { group: "Open", keys: [
      { k: ["Enter"], label: "View full screen", bar: "View" },
      { k: ["C"], label: "Compare the selection, or this photo and the next" },
      { k: ["E"], label: "Edit", bar: "Edit" },
    ]},
    { group: "Select", keys: [
      { k: ["X"], label: "Select or unselect" },
      { k: ["⇧", "X"], label: "Select a range" },
      { k: ["D"], label: "Download the selection" },
      { k: ["Esc"], label: "Clear the selection" },
    ]},
    { group: "Show", keys: [
      { k: ["I"], label: "Photo info panel" },
      { k: ["K"], label: "Focus peaking" },
      { k: ["B"], label: "Subject boxes" },
    ]},
  ],
  viewer: [
    { group: "Decide", note: "Acts on the photo shown, then moves to the next.", keys: [
      { k: ["P"], alt: "A", label: "Pick", bar: true },
      { k: ["R"], label: "Reject", bar: true },
      { k: ["V"], label: "Review", bar: true },
      { k: ["U"], label: "Clear the decision", bar: "Clear" },
    ]},
    { group: "Rate and label", note: "Kept apart from the decision, and written into exports.", keys: [
      { k: ["1", "–", "5"], label: "Stars", bar: "Stars" },
      { k: ["0"], label: "No stars" },
      { k: ["6", "7", "8", "9"], label: "Red, yellow, green, blue label (again to remove)" },
    ]},
    { group: "Move", keys: [
      { k: ["←", "→"], alt: "Space", label: "Previous / next photo", bar: "Move" },
    ]},
    { group: "Look closer", keys: [
      { k: ["F"], alt: "Z", label: "Fit or 100%", bar: "100%" },
      { k: ["L"], label: "Loupe", bar: "Loupe" },
      { k: ["K"], label: "Focus peaking" },
      { k: ["B"], label: "Subject boxes" },
      { k: ["C"], label: "HDR merge or its 0 EV frame" },
    ]},
    { group: "Leave", keys: [
      { k: ["E"], label: "Edit this photo", bar: "Edit" },
      { k: ["Esc"], label: "Back to the grid", bar: "Close" },
    ]},
  ],
  compare: [
    { group: "Choose", note: "Acts on the active pane, framed in blue.", keys: [
      { k: ["Tab"], alt: "a click", label: "Next pane", bar: "Pane" },
      { k: ["←", "→"], label: "The photo before or after, in the active pane", bar: "Swap" },
      { k: ["P"], label: "Pick", bar: true },
      { k: ["R"], label: "Reject", bar: true },
      { k: ["V"], label: "Review", bar: true },
      { k: ["U"], label: "Clear the decision" },
      { k: ["1", "–", "5"], label: "Stars", bar: "Stars" },
      { k: ["6", "7", "8", "9"], label: "Colour label" },
    ]},
    { group: "Look closer", keys: [
      { k: ["Z"], alt: "F or a click", label: "Every pane to 100% at the same place, or back to fit", bar: "100%" },
      { k: ["drag"], label: "Pan, all panes together" },
    ]},
    { group: "Leave", keys: [
      { k: ["E"], label: "Edit the active photo" },
      { k: ["Esc"], label: "Back to the grid, on the active photo", bar: "Close" },
    ]},
  ],
  editor: [
    { group: "Photo", keys: [
      { k: ["←", "→"], label: "Previous / next photo", bar: "Photo" },
      { k: ["C"], label: "Hold to see the original", bar: "Original" },
      { k: ["F"], label: "Fit or 100%", bar: "Zoom" },
      { k: ["Enter"], label: "Save", bar: "Save" },
      { k: ["Esc"], label: "Back out: tool, then mask, then the editor", bar: "Close" },
    ]},
    { group: "Undo", keys: [
      { k: [MOD, "Z"], label: "Undo", bar: "Undo" },
      { k: IS_MAC ? ["⇧", "⌘", "Z"] : ["Ctrl", "Y"], label: "Redo" },
    ]},
    { group: "Masks", keys: [
      { k: ["R"], label: "Add a radial mask" },
      { k: ["G"], label: "Add a gradient mask" },
      { k: ["B"], label: "Add a brush mask", bar: "Brush" },
      { k: ["[", "]"], label: "Brush size" },
      { k: ["\\"], label: "Show the mask" },
      { k: ["Delete"], label: "Delete the selected mask" },
    ]},
  ],
};
const KEY_CONTEXT_NAMES = { grid: "Culling", viewer: "Viewer", compare: "Compare", editor: "Editor" };

function keyContext() {
  if (!$("#edit-modal").classList.contains("hidden")) return "editor";
  if (compareOpen()) return "compare";
  return state.modal.open ? "viewer" : "grid";
}

function keyCaps(keys) {
  return keys.map((k) => `<kbd>${escapeHtml(k)}</kbd>`).join("");
}

// The bar's entries run in the sheet's order. Decisions are the core of
// culling, so their labels stay; the rest shorten to fit one line.
function renderKeybars() {
  const collapsed = keybarCollapsed();
  $$(".keybar").forEach((bar) => {
    const ctx = bar.dataset.context;
    const items = KEYMAP[ctx].flatMap((g) => g.keys).filter((e) => e.bar);
    bar.classList.toggle("collapsed", collapsed);
    bar.innerHTML = (collapsed ? "" : items.map((e) =>
      `<span class="kb-item">${keyCaps(e.k)}<span>${escapeHtml(e.bar === true ? e.label : e.bar)}</span></span>`).join(""))
      + `<span class="kb-spacer"></span>`
      + `<button type="button" class="kb-all" title="Every shortcut (?)">${keyCaps(["?"])}<span>All shortcuts</span></button>`
      + `<button type="button" class="kb-toggle" title="${collapsed ? "Show the key bar" : "Hide the key bar"}">${collapsed ? "Show keys" : "Hide"}</button>`;
    bar.querySelector(".kb-all").addEventListener("click", () => openKeysSheet(ctx));
    bar.querySelector(".kb-toggle").addEventListener("click", () => {
      try { localStorage.setItem("pcls.keybar", collapsed ? "1" : "0"); } catch { /* private */ }
      renderKeybars();
    });
  });
}

function keybarCollapsed() {
  try { return localStorage.getItem("pcls.keybar") === "0"; } catch { return false; }
}

function openKeysSheet(ctx) {
  $("#keys-sheet").dataset.ctx = ctx;
  renderKeysSheet(ctx);
  $("#keys-sheet").classList.remove("hidden");
}

function closeKeysSheet() {
  $("#keys-sheet").classList.add("hidden");
}

function renderKeysSheet(ctx) {
  $$("#keys-sheet .keys-tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.ctx === ctx);
    b.setAttribute("aria-selected", b.dataset.ctx === ctx ? "true" : "false");
  });
  $("#keys-body").innerHTML = KEYMAP[ctx].map((g) => `
    <section class="keys-group">
      <h3>${escapeHtml(g.group)}</h3>
      ${g.note ? `<p class="keys-note">${escapeHtml(g.note)}</p>` : ""}
      <dl>${g.keys.map((e) => `
        <dt>${keyCaps(e.k)}${e.alt ? `<span class="keys-alt">or ${escapeHtml(e.alt)}</span>` : ""}</dt>
        <dd>${escapeHtml(e.label)}</dd>`).join("")}
      </dl>
    </section>`).join("");
  setBtnLabel($("#keys-tour"), `Replay the ${ctx === "editor" ? "editor" : "culling"} tour`);
}

function bindKeysSheet() {
  renderKeybars();
  $$("#keys-sheet .keys-tabs button").forEach((b) =>
    b.addEventListener("click", () => { $("#keys-sheet").dataset.ctx = b.dataset.ctx; renderKeysSheet(b.dataset.ctx); }));
  $("#keys-close").addEventListener("click", closeKeysSheet);
  $("#keys-sheet").addEventListener("click", (e) => { if (e.target.id === "keys-sheet") closeKeysSheet(); });
  $("#keys-tour").addEventListener("click", () => {
    const ctx = $("#keys-sheet").dataset.ctx;
    closeKeysSheet();
    startTour(ctx === "editor" ? "editor" : "main");
  });
}

// ---------- guided tour ----------
// A walk-through, game-tutorial style: everything but one control is dimmed,
// and a card beside it says what it is for. It starts by itself the first time
// each screen is seen, and the ? button or the ? key replays it. A step whose
// control is not on screen (no photos yet, a panel not in this build) is
// skipped rather than pointing at nothing.
const TOURS = {
  main: [
    { title: "Welcome to Picture Classifier",
      body: "A one-minute tour of the culling screen. Replay it any time from the <b>?</b> button or the <kbd>?</kbd> key, which also lists every shortcut." },
    { target: "#scene-list", title: "Scenes",
      body: "Your photos, grouped into scenes by folder or by time gaps. Work through them one at a time; the bar under each shows how far you are." },
    { target: () => $(".tile .tile-img"), title: "Every photo has a suggestion",
      body: "The tag on the photo is the app's suggestion, from sharpness, exposure and closed eyes; <b>badness</b> under it is how bad it looks, lower is better. Click a photo, or press <kbd>Enter</kbd>, to see it large." },
    { target: () => $(".tile .tile-action-bar"), title: "Decide",
      body: "<b>Reject</b> <kbd>R</kbd> · <b>Review</b> <kbd>V</kbd> · <b>Pick</b> <kbd>P</kbd>. The keys act on the highlighted photo and move on to the next; the arrow keys move without deciding, and <kbd>U</kbd> clears a decision. The bar along the bottom always shows the keys for where you are." },
    { target: () => $(".tile .tile-edit-btn"), title: "Edit",
      body: "Opens the editor (<kbd>E</kbd>): light and colour, masks, lens and perspective, looks, healing, and portrait retouching." },
    { target: "#filter-row", title: "Filters",
      body: "Show only what is left to decide, or only your picks, to check a scene before you leave it." },
    { target: "#cols-toggle", title: "How many at once",
      body: "One photo to judge focus, eight to compare a burst. <kbd>[</kbd> and <kbd>]</kbd> turn the page." },
    { target: "#people-section", title: "People",
      body: "Faces grouped by person. Click one to see only the photos they are in." },
    { target: "#export-picks-btn", title: "Export your picks",
      body: "Writes every photo marked Pick to a folder, edits applied and camera details kept." },
    { target: "#topbar .tb-tabs", title: "Projects, Cull and Edit",
      body: "<b>Projects</b> goes back to the start screen (everything you decided is already saved), <b>Cull</b> is this screen, and <b>Edit</b> opens the highlighted photo in the editor. The project's name, top left, switches to another project; the power button, top right, quits the app, since closing the browser tab leaves it running." },
    { target: "#tour-main-btn", title: "That's it",
      body: "Press <kbd>?</kbd> for every keyboard shortcut and to replay this tour. The editor has its own tour the first time you open it." },
  ],
  editor: [
    { title: "The editor",
      body: "Everything here is non-destructive: your original file is never changed. A short tour of what is where." },
    { target: ".edit-canvas-wrap", title: "The photo",
      body: "Shown as it will export. Scroll to zoom, drag to pan, hold <kbd>C</kbd> to see the original, and <kbd>F</kbd> switches between fit and 100%." },
    { target: ".edit-actionbar", title: "Auto, undo and presets",
      body: "<b>Auto</b> sets a starting tone. <kbd>⌘Z</kbd> / <kbd>Ctrl+Z</kbd> undoes and <kbd>⇧⌘Z</kbd> / <kbd>Ctrl+Y</kbd> redoes. Presets save a look to reuse." },
    { target: "#edit-mask-group", title: "Adjust the whole photo or part of it",
      body: "The sliders change the whole photo. Open <b>Masks</b> to add a radial, gradient or brush, or an automatic Subject, Background or Skin mask, and the same sliders change only that part." },
    { target: "#edit-optics-group", title: "Lens & perspective",
      body: "Straighten leaning buildings with <b>Upright</b>, and correct distortion, colour fringes and dark corners." },
    { target: "#edit-portrait-group", title: "Portrait",
      body: "Smooth skin, whiten teeth and eyes, and remove blemishes, sized to each face it finds." },
    { target: "#edit-repair-group", title: "Heal & red eye",
      body: "Remove dust and small things by hand, and fix red eyes." },
    { target: "#edit-save", title: "Save",
      body: "Keeps the edit on this photo. <b>Cancel</b> throws the changes away. <b>Apply to more…</b> copies this edit to other photos." },
  ],
};
const tourState = { name: null, steps: [], i: 0 };

function tourSeen(name) {
  try { return localStorage.getItem(`pcls.tour.${name}.seen`) === "1"; } catch { return false; }
}

function maybeStartTour(name) {
  if (!tourSeen(name)) startTour(name);
}

function tourTarget(step) {
  if (!step.target) return null;
  const el = typeof step.target === "function" ? step.target() : $(step.target);
  if (!el) return null;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0 ? el : null;
}

function startTour(name) {
  // Only the steps whose control exists right now.
  tourState.name = name;
  tourState.steps = TOURS[name].filter((st) => !st.target || tourTarget(st));
  tourState.i = 0;
  $("#tour").classList.remove("hidden");
  showTourStep();
}

function endTour() {
  $("#tour").classList.add("hidden");
  try { localStorage.setItem(`pcls.tour.${tourState.name}.seen`, "1"); } catch { /* private */ }
  tourState.name = null;
}

function tourStep(delta) {
  const next = tourState.i + delta;
  if (next < 0) return;
  if (next >= tourState.steps.length) { endTour(); return; }
  tourState.i = next;
  showTourStep();
}

function showTourStep() {
  const step = tourState.steps[tourState.i];
  const n = tourState.steps.length;
  $("#tour-title").textContent = step.title;
  $("#tour-body").innerHTML = step.body;
  $("#tour-count").textContent = `${tourState.i + 1} / ${n}`;
  $("#tour-dots").innerHTML = tourState.steps.map((_, k) =>
    `<span class="${k === tourState.i ? "on" : k < tourState.i ? "done" : ""}"></span>`).join("");
  $("#tour-back").disabled = tourState.i === 0;
  setBtnLabel($("#tour-next"), tourState.i === n - 1 ? "Done" : "Next");
  const el = tourTarget(step);
  // A control inside a scrolling panel (the editor's) is brought into view first.
  if (el) el.scrollIntoView({ block: "nearest", inline: "nearest" });
  placeTour(el);
  $("#tour-next").focus();
}

// The cutout sits over the control and dims everything else with one huge
// shadow; the card goes on whichever side has room, and never off screen.
function placeTour(el) {
  const spot = $("#tour-spot"), card = $("#tour-card");
  const vw = window.innerWidth, vh = window.innerHeight, pad = 6, gap = 14, m = 12;
  if (!el) {
    spot.classList.add("none");
    card.style.left = `${Math.max(m, (vw - card.offsetWidth) / 2)}px`;
    card.style.top = `${Math.max(m, (vh - card.offsetHeight) / 2)}px`;
    return;
  }
  spot.classList.remove("none");
  const r = el.getBoundingClientRect();
  const box = { x: Math.max(2, r.left - pad), y: Math.max(2, r.top - pad) };
  box.w = Math.min(vw - 4, r.right + pad) - box.x;
  box.h = Math.min(vh - 4, r.bottom + pad) - box.y;
  Object.assign(spot.style, { left: `${box.x}px`, top: `${box.y}px`,
                              width: `${box.w}px`, height: `${box.h}px` });
  const cw = card.offsetWidth, ch = card.offsetHeight;
  const room = { right: vw - (box.x + box.w), left: box.x, bottom: vh - (box.y + box.h), top: box.y };
  let x, y;
  if (room.right >= cw + gap + m) { x = box.x + box.w + gap; y = box.y; }
  else if (room.left >= cw + gap + m) { x = box.x - gap - cw; y = box.y; }
  else if (room.bottom >= ch + gap + m) { x = box.x; y = box.y + box.h + gap; }
  else if (room.top >= ch + gap + m) { x = box.x; y = box.y - gap - ch; }
  else { x = (vw - cw) / 2; y = vh - ch - m; }     // no side has room: over the bottom
  card.style.left = `${Math.min(vw - cw - m, Math.max(m, x))}px`;
  card.style.top = `${Math.min(vh - ch - m, Math.max(m, y))}px`;
}

function bindTour() {
  $("#tour-next").addEventListener("click", () => tourStep(1));
  $("#tour-back").addEventListener("click", () => tourStep(-1));
  $("#tour-skip").addEventListener("click", endTour);
  $("#tour-main-btn").addEventListener("click", () => openKeysSheet(keyContext()));
  $("#tour-editor-btn").addEventListener("click", () => openKeysSheet("editor"));
  window.addEventListener("resize", () => {
    if (tourState.name) placeTour(tourTarget(tourState.steps[tourState.i]));
  });
}

// ---------- new-project wizard ----------
// `photoInfo` is the folder check for what is in the photo-folder field now,
// or null while it is stale; `sceneTouched` stops the suggestion from undoing
// a grouping the user picked by hand. `shots` is the capture-time read the
// grouping preview draws from, for the folders named by `shotsKey`.
const wizardState = {
  step: 1, sceneMode: "folder", subjectPreset: "",
  photoInfo: null, inspectTimer: null, sceneTouched: false,
  shots: null, shotsKey: "",
};

function openWizard({ first = false } = {}) {
  if (!workspaceState.current) {
    alert("Add a workspace first (a folder to hold your projects).");
    return;
  }
  wizardState.step = 1;
  wizardState.sceneMode = "folder";
  wizardState.subjectPreset = "";
  wizardState.photoInfo = null;
  wizardState.sceneTouched = false;
  $("#wizard-title").textContent = first ? "Your first project" : "New project";
  $("#wiz-photo-dir").placeholder = examplePhotoDir();
  $("#wiz-photo-info").textContent = "";
  $("#wiz-photo-info").className = "folder-info";
  $("#wiz-scene-note").textContent = "";
  $("#wiz-photo-dir").value = "";
  $("#wiz-jpeg-subdir").value = "";
  $("#wiz-raw-subdir").value = "";
  $("#wiz-project-name").value = "";
  $("#wiz-gap").value = 30;
  $("#wiz-gap-range").value = 30;
  $("#wiz-gap-row").style.display = "none";
  wizardState.shots = null;
  wizardState.shotsKey = "";
  $$("#wiz-scene-cards .option-card").forEach((c) =>
    c.classList.toggle("active", c.dataset.value === "folder"),
  );
  $$("#wiz-subject-cards .option-card").forEach((c) =>
    c.classList.toggle("active", c.dataset.value === ""),
  );
  syncWizardTargetHint();
  showWizardStep(1);
  $("#wizard").classList.remove("hidden");
}

function closeWizard() {
  $("#wizard").classList.add("hidden");
}

function showWizardStep(n) {
  wizardState.step = n;
  $$(".wizard-step").forEach((s) =>
    s.classList.toggle("hidden", parseInt(s.dataset.step, 10) !== n),
  );
  $("#wizard-step-indicator").textContent = `Step ${n} of 3`;
  $("#wizard-back").disabled = n === 1;
  const onLast = n === 3;
  $("#wizard-next").classList.toggle("hidden", onLast);
  $("#wizard-create").classList.toggle("hidden", !onLast);
  if (n === 2 && !$("#wiz-project-name").value.trim()) {
    // Suggest a name from the photo folder.
    const bn = basename($("#wiz-photo-dir").value.trim());
    if (bn) { $("#wiz-project-name").value = bn; }
  }
  syncWizardTargetHint();
  if (n === 2) loadWizardShots();
  if (onLast) renderWizardSummary();
}

function syncWizardTargetHint() {
  const name = $("#wiz-project-name").value.trim() || "<name>";
  const ws = workspaceState.current || "<workspace>";
  $("#wiz-target-hint").textContent = joinPath(ws, name) + (workspaceState.sep || "/");
}

// A believable path for this machine, built from the home folder the default
// workspace sits in, so the placeholder shows which way the slashes go.
function examplePhotoDir() {
  const home = workspaceState.defaultDir ? dirname(workspaceState.defaultDir) : "";
  return home ? `e.g. ${joinPath(joinPath(home, "Pictures"), "2026-05-wedding")}` : "/path/to/photos";
}

function setWizardSceneMode(mode) {
  wizardState.sceneMode = mode;
  setOptionCardValue("#wiz-scene-cards", mode);
  $("#wiz-gap-row").style.display = mode === "time_gap" ? "" : "none";
  renderScenePreview();
}

// ---------- the grouping preview, on the folder's own photos ----------
// The server reads every shot's capture time once (folderinfo.shots, the way
// scoring pairs RAW and JPEG); the split for any gap is then worked out here,
// on every slider move, with the same rule as scenes.group_by_time_gap: sort by
// time, and start a new scene where the pause is longer than the gap.
const SCENE_COLOURS = ["#5b8def", "#46a758", "#d4a02c", "#c56bd6", "#e0735b", "#3fb8c4", "#a3a3b8"];

async function loadWizardShots() {
  const path = $("#wiz-photo-dir").value.trim();
  const sub = $("#wiz-jpeg-subdir").value.trim();
  const rawSub = $("#wiz-raw-subdir").value.trim();
  const key = `${path}|${sub}|${rawSub}`;
  if (wizardState.shotsKey === key && wizardState.shots) { renderScenePreview(); return; }
  wizardState.shotsKey = key;
  wizardState.shots = null;
  renderScenePreview();
  const res = await fetch("/api/folder/shots", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, jpeg_subdir: sub, raw_subdir: rawSub }),
  });
  if (wizardState.shotsKey !== key) return;          // the folder changed meanwhile
  wizardState.shots = res.ok ? await res.json() : { error: `Could not read the folder (${res.status}).` };
  suggestSceneMode(wizardState.shots);
  renderScenePreview();
}

function timeGapScenes(times, gapMin) {
  const out = [];
  let cur = null;
  for (const t of times) {
    if (!cur || t - cur.end > gapMin * 60) { cur = { start: t, end: t, times: [] }; out.push(cur); }
    cur.times.push(t);
    cur.end = t;
  }
  return out;
}

function fmtDuration(sec) {
  const m = Math.round(sec / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60), r = m % 60;
  return r ? `${h} h ${r} min` : `${h} h`;
}

// Capture times come as the camera's wall clock counted as if it were UTC
// (folderinfo._wall_seconds), so they are formatted back in UTC to show it.
function fmtClock(sec) {
  return new Date(sec * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });
}

function renderScenePreview() {
  const box = $("#wiz-scene-preview");
  if (!box) return;
  const shots = wizardState.shots;
  if (!shots) { box.innerHTML = `<p class="sp-wait">Looking at your photos…</p>`; return; }
  if (shots.error) { box.innerHTML = `<p class="sp-wait">${escapeHtml(shots.error)}</p>`; return; }
  box.innerHTML = (wizardState.sceneMode === "folder" ? folderPreview(shots) : timeGapPreview(shots))
    + (shots.truncated ? `<p class="sp-note">Only the first ${shots.shots.toLocaleString()} photos were read.</p>` : "");
}

function folderPreview(shots) {
  const folders = shots.folders;
  if (folders.length === 1 && folders[0].name === "(none)") {
    return `<p class="sp-head">All ${plural(shots.shots, "photo")} are directly in this folder, `
      + `so this would be <b>one scene</b>.</p>`
      + `<p class="sp-note">By time gap would split them where you paused.</p>`;
  }
  const max = Math.max(...folders.map((f) => f.count));
  const rows = folders.slice(0, 10).map((f, i) =>
    `<div class="sp-row"><span class="sp-name">${escapeHtml(f.name === "(none)" ? "(loose photos)" : f.name)}</span>`
    + `<span class="sp-bar"><i style="width:${(100 * f.count / max).toFixed(1)}%;background:${SCENE_COLOURS[i % SCENE_COLOURS.length]}"></i></span>`
    + `<span class="sp-count">${f.count}</span></div>`).join("");
  const more = folders.length > 10 ? `<p class="sp-note">and ${folders.length - 10} more</p>` : "";
  return `<p class="sp-head">Your photos would make <b>${plural(folders.length, "scene")}</b>, one per folder:</p>`
    + rows + more;
}

function timeGapPreview(shots) {
  const gap = parseInt($("#wiz-gap").value, 10) || 30;
  const times = shots.times;
  if (!times.length) {
    return `<p class="sp-head">None of these photos has a capture time, so a time gap cannot split them.</p>`
      + `<p class="sp-note">Group by folder instead.</p>`;
  }
  const groups = timeGapScenes(times, gap);
  const sizes = groups.map((g) => g.times.length);
  const pauses = groups.slice(1).map((g, i) => g.start - groups[i].end);
  // A timeline with the long pauses cut out: each scene is as wide as it was
  // long (with a floor, so a burst is not a sliver), and a fixed break stands
  // for every pause that started a new scene.
  const W = 1000, H = 44, BREAK = groups.length > 1 ? Math.min(14, 300 / (groups.length - 1)) : 0;
  const span = groups.reduce((a, g) => a + (g.end - g.start), 0);
  const floor = Math.max(span / Math.max(groups.length, 1) * 0.25, 1);
  const weights = groups.map((g) => Math.max(g.end - g.start, floor));
  const unit = (W - BREAK * (groups.length - 1)) / weights.reduce((a, b) => a + b, 0);
  let x = 0, svg = "";
  groups.forEach((g, i) => {
    const w = weights[i] * unit, col = SCENE_COLOURS[i % SCENE_COLOURS.length];
    svg += `<rect x="${x.toFixed(1)}" y="8" width="${Math.max(w, 1).toFixed(1)}" height="28" rx="4" fill="${col}" opacity="0.14"/>`;
    const dur = Math.max(g.end - g.start, 1);
    for (const t of g.times) {
      const cx = x + (g.end === g.start ? w / 2 : ((t - g.start) / dur) * w);
      svg += `<circle cx="${cx.toFixed(1)}" cy="22" r="3" fill="${col}"/>`;
    }
    x += w;
    if (i < groups.length - 1) {
      svg += `<line x1="${(x + BREAK / 2).toFixed(1)}" y1="4" x2="${(x + BREAK / 2).toFixed(1)}" y2="40" stroke="#6d6d7c" stroke-dasharray="3 3"/>`;
      x += BREAK;
    }
  });
  const untimed = shots.untimed
    ? `<p class="sp-note">${plural(shots.untimed, "photo")} with no capture time go together into a scene of their own.</p>` : "";
  const detail = groups.length > 1
    ? `Largest ${sizes.length ? Math.max(...sizes) : 0} photos, smallest ${Math.min(...sizes)}. `
      + `The shortest pause that split them was ${fmtDuration(Math.min(...pauses))}.`
    : `Nowhere did you pause for longer than ${fmtDuration(gap * 60)}`
      + (times.length > 1
        ? `; your longest pause was ${fmtDuration(Math.max(...times.slice(1).map((t, i) => t - times[i])))}.`
        : ".");
  return `<p class="sp-head">At a ${gap}-minute gap your photos make <b>${plural(groups.length, "scene")}</b>.</p>`
    + `<svg class="sp-timeline" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true">${svg}</svg>`
    + `<div class="sp-axis"><span>${fmtClock(times[0])}</span>`
    + `<span>each colour is one scene · dashed lines are pauses, cut out</span>`
    + `<span>${fmtClock(times[times.length - 1])}</span></div>`
    + `<p class="sp-note">${detail}</p>${untimed}`;
}

// Check the photo folder as soon as one is chosen, and say what is in it, so
// a wrong folder is obvious here rather than after a failed create.
function showWizPhotoInfo(kind, html) {
  const box = $("#wiz-photo-info");
  box.className = "folder-info " + kind;
  box.innerHTML = (kind === "ok" ? icon("check") : kind ? icon("warning") : "")
    + `<span>${html}</span>`;
}

async function inspectWizardPhotoDir() {
  const path = $("#wiz-photo-dir").value.trim();
  const box = $("#wiz-photo-info");
  const show = showWizPhotoInfo;
  if (!path) { box.textContent = ""; box.className = "folder-info"; wizardState.photoInfo = null; return; }
  show("", "Looking in the folder…");
  const info = await inspectFolder(path, workspaceState.current);
  if ($("#wiz-photo-dir").value.trim() !== path) return;   // typed on since
  const n = info.photos + info.raws;
  let usable = false;
  if (!info.absolute) {
    show("error", "Enter the full path of the folder, for example "
      + `<code>${escapeHtml(examplePhotoDir().replace(/^e\.g\. /, ""))}</code>, or use Browse.`);
  } else if (!info.exists) {
    show("error", "That folder does not exist.");
  } else if (info.is_project) {
    show("error", "That is a project folder, not a photo folder. Choose the folder the photos are in.");
  } else if (info.contains_workspace) {
    show("error", "Your projects folder is inside this folder, so the project would end "
      + "up among your photos. Choose the folder of one shoot instead.");
  } else if (n === 0) {
    show("error", "No photos found in this folder or its subfolders (JPEG, PNG or RAW).");
  } else {
    usable = true;
    const kinds = [info.photos && `${plural(info.photos, "JPEG/PNG file")}`,
                   info.raws && `${plural(info.raws, "RAW file")}`].filter(Boolean).join(" and ");
    const where = info.subfolders
      ? `in ${plural(info.subfolders, "subfolder")}`
        + (info.loose ? `, plus ${info.loose.toLocaleString()} directly in the folder` : "")
      : "directly in this folder";
    if (info.truncated) {
      show("warn", `At least ${kinds} ${where}; counting stopped there. A folder this big `
        + "takes a long time to score. The folder of a single shoot is usually what you want.");
    } else {
      show("ok", `Found <b>${kinds}</b> ${where}.`);
    }
  }
  wizardState.photoInfo = { ...info, usable };
}

// Grouping by folder only makes scenes out of subfolders. With none, every photo
// would land in one scene, so time gaps are the better default there, as long
// as the photos carry capture times to split on.
function suggestSceneMode(shots) {
  const note = $("#wiz-scene-note");
  if (!shots || shots.error || !shots.shots) { note.textContent = ""; return; }
  const loose = shots.folders.length === 1 && shots.folders[0].name === "(none)";
  if (loose && shots.times.length > 1) {
    if (!wizardState.sceneTouched) setWizardSceneMode("time_gap");
    note.textContent = "Suggested for this folder: By time gap. The photos are not in "
      + "subfolders, so By folder would put them all in one scene.";
  } else if (!loose) {
    if (!wizardState.sceneTouched) setWizardSceneMode("folder");
    note.textContent = "Suggested for this folder: By folder. The photos are already "
      + "sorted into subfolders.";
  } else {
    note.textContent = "";
  }
}

function renderWizardSummary() {
  const items = [
    ["Workspace", workspaceState.current || ""],
    ["Project name", $("#wiz-project-name").value],
    ["Photos", $("#wiz-photo-dir").value],
    ["JPEG subfolder", $("#wiz-jpeg-subdir").value || "(none)"],
    ["RAW subfolder", $("#wiz-raw-subdir").value || "(none)"],
    ["Scene grouping",
      wizardState.sceneMode === "time_gap"
        ? `By time gap (${$("#wiz-gap").value || 30} min)`
        : "By folder"],
    ["Subject detection", wizardState.subjectPreset || "off"],
  ];
  $("#wiz-summary").innerHTML = items
    .map(([k, v]) => `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`)
    .join("");
}

function validateWizardStep(n) {
  if (n === 1) {
    return $("#wiz-photo-dir").value.trim() !== "" || "Pick a photo folder.";
  }
  if (n === 2) {
    if (!$("#wiz-project-name").value.trim()) return "Enter a project name.";
    return true;
  }
  return true;
}

async function createProject() {
  const payload = {
    name: $("#wiz-project-name").value.trim(),
    workspace_dir: workspaceState.current,
    photo_dir: $("#wiz-photo-dir").value.trim(),
    jpeg_subdir: $("#wiz-jpeg-subdir").value.trim(),
    raw_subdir: $("#wiz-raw-subdir").value.trim(),
    scene_grouping_mode: wizardState.sceneMode,
    scene_grouping_gap_minutes: parseInt($("#wiz-gap").value, 10) || 30,
    // The server resolves a preset name into its COCO classes.
    subject_classes: wizardState.subjectPreset ? [wizardState.subjectPreset] : [],
  };
  $("#wizard-create").disabled = true;
  $("#wizard-create").textContent = "Creating…";
  const res = await fetch("/api/project/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  $("#wizard-create").disabled = false;
  $("#wizard-create").textContent = "Create project";
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Could not create project: " + (err.detail || res.status));
    return;
  }
  showOpenProgress("Creating project…");
}

async function pickAndOpenProject() {
  const status = $("#landing-status");
  status.textContent = BROWSE_WAITING;
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial: null, purpose: "project" }),
    });
    if (!res.ok) {
      status.textContent = "Browse failed (" + res.status + ").";
      return;
    }
    const result = await res.json();
    if (!result.path) { status.textContent = ""; return; }
    await openProjectByDir(result.path);
  } catch (err) {
    status.textContent = "Browse network error: " + err.message;
  }
}

function pollOpenStatus() {
  if (scorePollTimer) clearInterval(scorePollTimer);
  let finished = false;
  const tick = async () => {
    if (finished) return;
    let s;
    try {
      const stateRes = await fetch("/api/state", { cache: "no-store" });
      if (!stateRes.ok) return;
      s = (await stateRes.json()).opening || {};
    } catch { return; }
    showScoreProgress({
      step: s.phase === "done" ? "loading" : s.phase,
      title: (s.phase === "scoring") ? (s.total ? `Scoring ${plural(s.total, "photo")}…` : "Getting ready to score…")
        : (s.phase === "clustering") ? "Grouping faces…"
        : (s.phase === "scanning") ? "Finding photos…"
        : "Opening the project…",
      idx: s.idx, total: s.total, current: s.current, message: s.message,
    });
    if (!s.running) {
      finished = true;
      clearInterval(scorePollTimer);
      scorePollTimer = null;
      if (s.cancelled) {
        $("#score-progress").classList.add("hidden");
        showLanding();
        return;
      }
      if (s.error) {
        $("#score-progress").classList.add("hidden");
        showLanding();
        if (s.needs_relink) {
          openRelinkModal(s.needs_relink);   // photos moved → offer to re-link
        } else {
          $("#landing-status").textContent = s.error;
          alert("Open failed: " + s.error);
        }
        return;
      }
      $("#score-progress").classList.add("hidden");
      $("#score-title").textContent = "Scoring photos…";
      await bootMain();
    }
  };
  tick();
  scorePollTimer = setInterval(tick, 250);
}

async function bootMain() {
  await loadDb();
  const view = await restoreView();
  loadPresets();
  loadSubjects();
  loadFilmStocks();
  loadLooks();
  syncViewControls();
  renderSidebar();
  renderPeopleChips();
  syncSceneGroupingControls();
  // A remembered scene can be gone — rescored, regrouped, or renamed — so fall
  // back to the first one. The page is only restored within the scene it was
  // counted in; anywhere else the number would point at unrelated photos.
  const scene = state.byScene.has(view.scene) ? view.scene : state.sceneOrder[0];
  if (scene) {
    selectScene(scene);
    const at = scene === view.scene && view.photo
      ? state.filteredPhotos.findIndex((p) => p.rel_path === view.photo) : -1;
    if (at >= 0) {
      state.cursorIdx = at;
      showCursor();
      if (view.viewer) openModal(at);
    } else if (scene === view.scene && view.page) {
      // The photo is gone from this filter (decided, or removed): its page.
      gotoPageIndex(view.page);
    }
  } else {
    renderMain();
  }
  viewRestored = true;
  maybeStartTour("main");
}

function setOptionCardValue(containerSelector, value) {
  $$(`${containerSelector} .option-card`).forEach((c) => {
    c.classList.toggle("active", c.dataset.value === value);
  });
}

function getOptionCardValue(containerSelector) {
  const active = document.querySelector(`${containerSelector} .option-card.active`);
  return active ? active.dataset.value : null;
}

function syncSceneGroupingControls() {
  const sg = state.sceneGrouping || { mode: "folder", gap_minutes: 30 };
  $("#scene-grouping-now").textContent = sg.mode === "time_gap" ? `by ${sg.gap_minutes}-min gaps` : "by folder";
  setOptionCardValue("#scene-mode-cards", sg.mode);
  $("#scene-gap").value = sg.gap_minutes;
  $("#scene-gap-row").classList.toggle("hidden", sg.mode !== "time_gap");
}

async function applySceneGrouping() {
  const mode = getOptionCardValue("#scene-mode-cards") || "folder";
  const gap = parseInt($("#scene-gap").value, 10) || 30;
  $("#score-title").textContent = "Regrouping scenes…";
  $("#score-progress").classList.remove("hidden");
  $("#score-bar-fill").style.width = "60%";
  $("#score-progress-text").textContent = mode === "folder" ? "by folder" : `${gap} min gap`;
  $("#score-current").textContent = "";
  const res = await fetch("/api/scene-grouping", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, gap_minutes: gap }),
  });
  $("#score-progress").classList.add("hidden");
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Scene grouping failed: " + (err.detail || res.status));
    return;
  }
  await bootMain();
}

// ---------- boot ----------
(async () => {
  bindUi();
  bindKeys();
  bindHelp();
  hydrateIcons();
  initTooltips();
  loupeInit();
  $$(".quit-btn").forEach((b) => b.addEventListener("click", quitApp));
  $("#undo-toast-btn .kbd").textContent = `${MOD_KEY}Z`;
  let s;
  try { s = await fetchState(); } catch { showLanding(); return; }
  if (s.opening && s.opening.running) {
    hideLanding();
    $("#score-title").textContent = "Opening project…";
    $("#score-progress").classList.remove("hidden");
    pollOpenStatus();
    return;
  }
  if (!s.ready) {
    showLanding();
    // Switching projects from the top bar closes this one, reloads, and
    // lands here with the next one to open.
    const next = sessionStorage.getItem("pcls.openNext");
    if (next) { sessionStorage.removeItem("pcls.openNext"); openProjectByDir(next); }
    return;
  }
  await bootMain();
  // If a long task is already running on the server, attach to it.
  const sr = await fetch("/api/score/status").catch(() => null);
  if (sr && sr.ok) {
    const sc = await sr.json();
    if (sc.running) {
      $("#score-title").textContent = "Scoring photos…";
      $("#score-progress").classList.remove("hidden");
      pollScoreStatus();
      return;
    }
  }
  const cr = await fetch("/api/cluster/status").catch(() => null);
  if (cr && cr.ok) {
    const c = await cr.json();
    if (c.running) {
      $("#score-title").textContent = "Clustering faces…";
      $("#score-progress").classList.remove("hidden");
      pollClusterStatus();
    }
  }
})();
