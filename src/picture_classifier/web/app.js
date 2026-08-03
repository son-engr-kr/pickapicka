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
    { k: "clarity",    label: "Clarity",     min: -100, max: 100, step: 1, fmt: 0 },
    { k: "sharpen",    label: "Sharpen",     min: 0,    max: 100, step: 1, fmt: 0 },
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
const CURVE_IDENTITY = [[0, 0], [1, 1]];

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
  const e = { curve: CURVE_IDENTITY.map((p) => p.slice()), masks: [], watermark: null,
              tilt: 0, crop: null };
  for (const f of EDIT_FIELDS) e[f.k] = 0;
  return e;
})();
// Sliders a local mask may carry — mirrors editing.LOCAL_KEYS (no vignette, no
// curve: those stay frame-wide). Rows outside this set hide in mask mode.
const MASK_LOCAL_KEYS = new Set(EDIT_FIELDS.map((f) => f.k).filter((k) => k !== "vignette"));
const MASK_MAX = 16;

function fieldByKey(k) { return EDIT_FIELDS.find((f) => f.k === k); }
function mergeNeutralEdit(edit) {
  const e = { ...EDIT_NEUTRAL };
  e.curve = (edit && Array.isArray(edit.curve) && edit.curve.length)
    ? edit.curve.map((p) => p.slice()) : CURVE_IDENTITY.map((p) => p.slice());
  e.masks = (edit && Array.isArray(edit.masks)) ? edit.masks.map(cloneMask) : [];
  e.watermark = (edit && edit.watermark) ? cloneWatermark(edit.watermark) : null;
  e.tilt = (edit && Number(edit.tilt)) || 0;
  e.crop = (edit && edit.crop) ? { ...edit.crop } : null;
  if (edit) for (const f of EDIT_FIELDS) if (edit[f.k] != null) e[f.k] = edit[f.k];
  return e;
}
function editsEqual(a, b) {
  for (const f of EDIT_FIELDS) if (Math.abs((a[f.k] || 0) - (b[f.k] || 0)) > 1e-4) return false;
  const ca = a.curve || CURVE_IDENTITY, cb = b.curve || CURVE_IDENTITY;
  if (ca.length !== cb.length) return false;
  for (let i = 0; i < ca.length; i++)
    if (Math.abs(ca[i][0] - cb[i][0]) > 1e-4 || Math.abs(ca[i][1] - cb[i][1]) > 1e-4) return false;
  if (canonWatermark(a.watermark) !== canonWatermark(b.watermark)) return false;
  if (Math.abs((a.tilt || 0) - (b.tilt || 0)) > 1e-4) return false;
  if (canonCrop(a.crop) !== canonCrop(b.crop)) return false;
  return canonMasks(a.masks) === canonMasks(b.masks);
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
  radial: { icon: "◎", label: "Radial" },
  linear: { icon: "▤", label: "Gradient" },
  brush:  { icon: "🖌", label: "Brush" },
};

function neutralAdj() {
  const a = {};
  for (const k of MASK_LOCAL_KEYS) a[k] = 0;
  return a;
}

function newMask(kind, aspect) {
  const m = {
    type: kind, name: "", enabled: true, invert: false,
    feather: kind === "linear" ? 100 : 50, amount: 100, adj: neutralAdj(),
  };
  if (kind === "radial") Object.assign(m, { cx: 0.5, cy: 0.5, rx: 0.25, ry: 0.25 * (aspect || 1), angle: 0 });
  else if (kind === "linear") Object.assign(m, { x1: 0.5, y1: 0.15, x2: 0.5, y2: 0.55 });
  else m.strokes = [];
  return m;
}

function cloneMask(m) {
  const c = { ...m, adj: { ...neutralAdj(), ...(m.adj || {}) } };
  if (m.strokes) c.strokes = m.strokes.map((s) => ({ ...s, points: s.points.map((p) => p.slice()) }));
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
    if (m.type === "radial") o.g = [m.cx, m.cy, m.rx, m.ry, m.angle].map(rnd4);
    else if (m.type === "linear") o.g = [m.x1, m.y1, m.x2, m.y2].map(rnd4);
    else o.g = (m.strokes || []).map((s) => [rnd4(s.radius), !!s.erase,
                                             s.points.map((p) => [rnd4(p[0]), rnd4(p[1])])]);
    return o;
  }));
}

function maskAdjNeutral(m) {
  return [...MASK_LOCAL_KEYS].every((k) => Math.abs(Number(m.adj[k]) || 0) < 1e-4);
}

function maskLabel(m, idx) {
  return m.name || `${MASK_KINDS[m.type].label} ${idx + 1}`;
}

// ---------- tooltips with keyboard shortcuts ----------
// Native `title` is slow to appear and can't show a keycap, and the shortcuts
// were previously only discoverable from the one help strip at the bottom of
// the viewer. One floating element serves every control: no per-button markup,
// and nothing gets clipped by a scrolling panel the way a CSS ::after would.
const SHORTCUT_TIPS = {
  // grid + header
  "#prev-page": ["Previous page", "["],
  "#next-page": ["Next page", "]"],
  "#peak-btn": ["Highlight what is actually in focus", "K"],
  "#boxes-btn": ["Show detected subject boxes", "B"],
  "#hdr-btn": ["Review and fix auto-detected HDR brackets", null],
  "#rescore-btn": ["Re-score every photo, then re-group. Keeps decisions", null],
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
  "#edit-apply-more": ["Apply this edit to more photos", null],
  '[data-preset-mode="add"]': ["Lay the preset on top: its masks are added to yours", null],
  '[data-preset-mode="replace"]': ["Discard the current edit and use the preset alone", null],
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
  return matchesDecisionFilter(p) && matchesPersonFilter(p) && matchesSubjectFilter(p);
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
const pageIdx = () => Math.floor(state.cursorIdx / state.pageSize);
const pageCount = () => Math.max(1, Math.ceil(state.filteredPhotos.length / state.pageSize));
const visiblePhotos = () => {
  const start = pageIdx() * state.pageSize;
  return state.filteredPhotos.slice(start, start + state.pageSize);
};

// ---------- API ----------
async function loadDb() {
  const res = await fetch("/api/db");
  if (!res.ok) throw new Error(`db load failed: ${res.status}`);
  const data = await res.json();
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
    empty.textContent = "No people yet — click ↻ group.";
    wrap.appendChild(empty);
    return;
  }
  const visible = state.people.filter((p) => !p.excluded);
  if (!visible.length) {
    const empty = document.createElement("div");
    empty.id = "people-empty";
    empty.textContent = "All clusters excluded — open ⚙ to restore.";
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
      <span class="lbl">${person.label}</span>
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
  // The whole panel stays out of the way until a project opts into detection.
  section.classList.toggle("hidden", !on);
  $("#boxes-btn").classList.toggle("hidden", !on);
  if (!on) return;

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
    : `<div class="subject-empty">No groups yet — click ↻ group.</div>`;
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

function openSubjectModal() {
  subjectEdit.classes = new Set(state.subjects.classes || []);
  subjectEdit.search = "";
  $("#subject-search").value = "";
  $("#subject-status").textContent = "";
  $("#subject-model-note").textContent = state.subjects.model_ready
    ? "Detection model ready."
    : "First run downloads a ~20 MB detection model (YOLOX-tiny, Apache-2.0).";
  renderSubjectPresets();
  renderSubjectClassPicker();
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
  const next = [...subjectEdit.classes];
  const prev = state.subjects.classes || [];
  const changed = next.length !== prev.length || next.some((c) => !prev.includes(c));
  if (!changed) {
    renderSubjectPanel();
    renderGrid();
    closeSubjectModal();
    return;
  }
  const ok = confirm(
    next.length
      ? `Re-score all ${state.photos.length} photos detecting: ${next.join(", ")}?\n\n` +
        `Decisions and edits are preserved.`
      : `Turn subject detection off and re-score all ${state.photos.length} photos?\n\n` +
        `Decisions and edits are preserved.`
  );
  if (!ok) return;
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ with_faces: false, subject_classes: next }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#subject-status").textContent = "re-score failed: " + (err.detail || res.status);
    return;
  }
  closeSubjectModal();
  $("#score-progress").classList.remove("hidden");
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
    li.innerHTML = `
      <span class="name">${scene}</span>
      <span class="stats">${n} shots · pick ${counts.pick} · rev ${counts.review} · rej ${counts.reject} · — ${counts.undecided}</span>
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
    `${total} shots · decided ${decided}/${total} (${total ? Math.round(100*decided/total) : 0}%) · ` +
    `pick ${totalPick} rev ${totalReview} rej ${totalReject}`;
}

// ---------- rendering: main ----------
function selectScene(scene) {
  state.selectedScene = scene;
  state.cursorIdx = 0;
  state.selection.clear();
  lastSelIdx = null;
  recomputeFilter();
  renderSidebar();
  renderMain();
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
}

function applyLayoutCSS() {
  const L = LAYOUTS[state.pageSize];
  const grid = $("#grid");
  grid.style.setProperty("--cols", L.cols);
  grid.style.setProperty("--rows", L.rows);
}

function renderHeader() {
  const total = state.filteredPhotos.length;
  const scenePhotos = state.selectedScene
    ? (state.byScene.get(state.selectedScene) || []) : [];
  const sceneTotal = scenePhotos.length;
  const sceneUndecided = scenePhotos.reduce((n, p) => n + (p.decision == null ? 1 : 0), 0);
  $("#scene-title").textContent = state.selectedScene || "— select a scene —";
  $("#scene-stats").textContent = state.selectedScene
    ? `${total} of ${sceneTotal} shown · filter: ${state.filter}`
    : "";
  $("#page-indicator").textContent = total
    ? `page ${pageIdx() + 1}/${pageCount()}`
    : "—";
  $("#prev-page").disabled = pageIdx() === 0 || total === 0;
  $("#next-page").disabled = pageIdx() >= pageCount() - 1 || total === 0;
  const rejectBtn = $("#reject-undecided-btn");
  rejectBtn.disabled = sceneUndecided === 0;
  rejectBtn.textContent = sceneUndecided > 0
    ? `✕ reject ${sceneUndecided} undecided`
    : "✕ reject undecided";
  const totalPicks = state.photos.reduce((n, p) => n + (p.decision === "pick" ? 1 : 0), 0);
  const exportBtn = $("#export-picks-btn");
  exportBtn.disabled = totalPicks === 0;
  exportBtn.textContent = totalPicks > 0
    ? `📁 export ${totalPicks} pick${totalPicks > 1 ? "s" : ""}`
    : "📁 export picks";
  $("#hdr-btn").textContent = `🌅 HDR (${state.brackets.length})`;
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  const start = pageIdx() * state.pageSize;
  const visible = visiblePhotos();
  visible.forEach((p, i) => {
    const absIdx = start + i;
    const tile = document.createElement("div");
    const orientation = (p.width && p.height && p.height > p.width) ? "portrait" : "landscape";
    tile.className = "tile " + orientation
      + (p.decision ? " decision-" + p.decision : "")
      + (absIdx === state.cursorIdx ? " focused" : "")
      + (state.selection.has(p.rel_path) ? " selected" : "");
    const auto = p.auto_suggestion || "";
    const badness = p.scores?.badness != null ? p.scores.badness.toFixed(2) : "—";
    const fname = p.rel_path.split("/").pop();
    const visibleFaces = (p.faces || [])
      .map((f, fi) => ({ f, fi, person: f.person_id ? state.peopleById.get(f.person_id) : null }))
      .filter(({ person }) => !(person && person.excluded))
      .sort((a, b) => {
        const pa = a.person ? a.person.priority : Infinity;
        const pb = b.person ? b.person.priority : Infinity;
        return pa - pb;
      });
    const faceCount = visibleFaces.length;
    const facesHtml = faceCount
      ? `<div class="tile-faces">${
          visibleFaces.map(({ fi }) =>
            `<div class="face-thumb"><img loading="lazy" src="/face/${enc(p.rel_path)}?idx=${fi}" alt="" /></div>`
          ).join("")
        }</div>`
      : "";
    tile.innerHTML = `
      <div class="tile-content">
        ${facesHtml}
        <div class="tile-img" data-action="open">
          <input type="checkbox" class="tile-select"${state.selection.has(p.rel_path) ? " checked" : ""} title="Select (X)" />
          <img loading="lazy" src="${thumbUrl(p)}" alt="" />
          ${peakLayerHtml(p)}
          ${boxLayerHtml(p)}
          ${auto ? `<span class="auto-badge ${auto}">auto: ${auto}</span>` : ""}
          ${p.type === "hdr" ? `<span class="hdr-tile-badge">HDR · ${(p.members || []).length}</span>` : ""}
          <span class="badness">${badness}</span>
          <button class="tile-edit-btn${p.edit ? " edited" : ""}" data-action="edit" title="Edit (E)">✎</button>
        </div>
      </div>
      <div class="tile-name">${fname}${faceCount ? ` · ${faceCount} face${faceCount>1?"s":""}` : ""}</div>
      <div class="tile-action-bar">
        <button class="btn-decision${p.decision === "reject" ? " active" : ""}" data-decision="reject">REJECT <span class="kbd">R</span></button>
        <button class="btn-decision${p.decision === "review" ? " active" : ""}" data-decision="review">REVIEW <span class="kbd">V</span></button>
        <button class="btn-decision${p.decision === "pick" ? " active" : ""}" data-decision="pick">PICK <span class="kbd">A</span></button>
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
    tile.addEventListener("mouseenter", () => focusAt(absIdx, false));
    grid.appendChild(tile);
  });
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
  $("#next-preview-name").textContent = next.rel_path.split("/").pop();
}

// ---------- focus & paging ----------
function focusAt(absIdx, scroll) {
  if (absIdx < 0 || absIdx >= state.filteredPhotos.length) return;
  const oldPage = pageIdx();
  state.cursorIdx = absIdx;
  if (pageIdx() !== oldPage) {
    renderMain();
  } else {
    $$(".tile").forEach((el, i) => el.classList.toggle("focused", i === (absIdx - oldPage * state.pageSize)));
    if (state.pageSize === 1) renderNextPreview();
  }
}

// ---------- remembered view ----------
// A project reopens on the filter, layout and page it was left on. Saved on the
// server per project: it is a preference about how you work, so it should follow
// the project rather than the browser profile that happened to open it.
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

async function saveView() {
  viewSaveTimer = null;
  const res = await fetch("/api/view", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filter: state.filter,
      page_size: state.pageSize,
      page: pageIdx(),
      scene: state.selectedScene,
    }),
  });
  if (!res.ok) throw new Error(`view save failed: ${res.status}`);
}

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
  const target = Math.max(0, Math.min(page, pageCount() - 1));
  state.cursorIdx = target * state.pageSize;
  renderMain();
}

function gotoPage(delta) {
  const newPage = pageIdx() + delta;
  if (newPage < 0 || newPage >= pageCount()) return;
  const newPageStart = newPage * state.pageSize;
  if (delta < 0) {
    // Previous: land on the last tile of the new page.
    const newPageEnd = Math.min(newPageStart + state.pageSize, state.filteredPhotos.length);
    state.cursorIdx = Math.max(newPageStart, newPageEnd - 1);
  } else {
    state.cursorIdx = newPageStart;
  }
  renderMain();
}

function tryAutoAdvance() {
  // If every visible photo has a decision, move to the first photo of the next page.
  const visible = visiblePhotos();
  if (!visible.length) return;
  const allDecided = visible.every((p) => p.decision != null);
  if (!allDecided) return;
  if (pageIdx() < pageCount() - 1) {
    state.cursorIdx = (pageIdx() + 1) * state.pageSize;
    renderMain();
  }
}

// ---------- decisions ----------
async function decideAt(absIdx, decision) {
  const photo = state.filteredPhotos[absIdx];
  if (!photo) return;
  const updated = await postDecision(photo.rel_path, decision);
  Object.assign(photo, updated);
  // Re-filter (a decided photo might leave the current filter view)
  const wasInFilter = matchesFilter(photo);
  recomputeFilter();
  renderSidebar();
  if (!wasInFilter && state.filteredPhotos.indexOf(photo) === -1) {
    // (no-op) photo no longer matches; cursor may have shifted
  }
  // Move cursor: stay at same position (which now points to the next photo
  // if the just-decided one left the filter), else advance by 1 within page.
  if (state.cursorIdx >= state.filteredPhotos.length) {
    state.cursorIdx = Math.max(0, state.filteredPhotos.length - 1);
  }
  renderMain();
  if (state.modal.open) {
    state.modal.idx = state.cursorIdx;
    renderModal();
  } else {
    tryAutoAdvance();
  }
}

// ---------- export picks ----------
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
  $("#export-status").textContent = "";
  $("#export-confirm").disabled = info.count === 0;
  $("#export-modal").classList.remove("hidden");
}

function closeExportModal() {
  $("#export-modal").classList.add("hidden");
}

async function confirmExport() {
  const target = $("#export-target").value.trim();
  if (!target) { alert("Target folder is required."); return; }
  const mode = getOptionCardValue("#export-mode-cards") || "folder";
  $("#export-confirm").disabled = true;
  $("#export-status").textContent = "Copying…";
  const res = await fetch("/api/export/picks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target_dir: target, mode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#export-status").textContent = "Failed: " + (err.detail || res.status);
    $("#export-confirm").disabled = false;
    return;
  }
  const result = await res.json();
  let html = `✓ Copied <b>${result.copied}</b> photos to<br><code>${result.target_dir}</code>`;
  if (result.skipped) html += `<br>Skipped ${result.skipped} (missing source).`;
  if (result.per_combo && Object.keys(result.per_combo).length) {
    const sorted = Object.entries(result.per_combo).sort((a, b) => b[1] - a[1]);
    html += `<br><span class="combo-summary">${sorted.map(([k, v]) => `${escapeHtml(k)}: ${v}`).join(" · ")}</span>`;
  }
  $("#export-status").innerHTML = html;
  $("#export-confirm").disabled = false;
}

// ---------- bulk actions ----------
async function rejectUndecidedInScene() {
  if (!state.selectedScene) return;
  const photos = state.byScene.get(state.selectedScene) || [];
  const targets = photos.filter((p) => p.decision == null);
  if (!targets.length) return;
  const ok = confirm(
    `Reject all ${targets.length} undecided photos in "${state.selectedScene}"?\n` +
    `Use the Undo toast (or ⌘Z) to revert.`
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
    `<span class="hdr-thumb-name">${escapeHtml(relPath.split("/").pop())}</span>`;
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
  // Original frame -> displayed frame, as six numbers from the server. Identity
  // until a crop or a straighten makes the two differ.
  maskXform: null,
};

// ---------- tone curve ----------
const CURVE_MAX = 6;         // max control points (keeps it approachable)
const CURVE_PAD = 8;
const curveState = { cssW: 240, cssH: 240, drag: -1, accent: "#4a90e2" };

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
  const pts = editSession.edit.curve;
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

  const pts = editSession.edit.curve;
  const N = 64, xs = [];
  for (let i = 0; i <= N; i++) xs.push(i / N);
  const ys = pchipEval(pts, xs);
  ctx.strokeStyle = curveState.accent; ctx.lineWidth = 2; ctx.beginPath();
  xs.forEach((x, i) => { const [qx, qy] = curveToPx(x, ys[i]); i ? ctx.lineTo(qx, qy) : ctx.moveTo(qx, qy); });
  ctx.stroke();
  ctx.fillStyle = curveState.accent;
  for (const [x, y] of pts) { const [qx, qy] = curveToPx(x, y); ctx.beginPath(); ctx.arc(qx, qy, 4, 0, 7); ctx.fill(); }
}

function curveDown(e) {
  if (!editSession.edit) return;
  const { px, py, x, y } = curveEventData(e);
  let i = curveHit(px, py);
  if (i < 0) {
    const pts = editSession.edit.curve;
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
  const pts = editSession.edit.curve, i = curveState.drag;
  if (i === 0) pts[i] = [0, y];                     // endpoints: x locked, y free
  else if (i === pts.length - 1) pts[i] = [1, y];
  else {
    const lo = pts[i - 1][0] + 0.01, hi = pts[i + 1][0] - 0.01;
    pts[i] = [Math.min(hi, Math.max(lo, x)), y];
  }
  drawCurve(); setEditDirty(); fetchEditPreview(false);
}
function curveUp() {
  if (curveState.drag < 0) return;
  curveState.drag = -1;
  fetchEditPreview(false);
}
function curveDoubleClick(e) {
  const { px, py } = curveEventData(e);
  const i = curveHit(px, py), pts = editSession.edit.curve;
  if (i > 0 && i < pts.length - 1) {             // can't remove endpoints
    pts.splice(i, 1);
    drawCurve(); setEditDirty(); fetchEditPreview(false);
  }
}
function resetCurve() {
  editSession.edit.curve = CURVE_IDENTITY.map((p) => p.slice());
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
    `${photo.rel_path.split("/").pop()} · ${absIdx + 1}/${state.filteredPhotos.length}`;
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
  $$("#edit-modal .edit-zoom").forEach((b) =>
    b.classList.toggle("active", b.dataset.zoom === "0"));
  $("#edit-zoom-hint").classList.add("hidden");
  layoutPreviewImage();
  renderWatermarkPanel();
  loadWatermarkInfo(photo.rel_path);
  selectMask(-1, { silent: true });
  setEditTool(null);
  syncEditSliders();
  drawCurve();
  setEditDirty();
  $("#edit-modal").classList.remove("hidden");
  resizeOverlay();
  fetchEditPreview(true);
  fetchOriginalPreview();
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
function syncEditValues() {
  const target = adjTarget();
  $$("#edit-modal .look-val[data-val]").forEach((el) => {
    const f = fieldByKey(el.dataset.val);
    el.textContent = Number(target[el.dataset.val] || 0).toFixed(f ? f.fmt : 0);
  });
}

function setEditDirty() {
  const dirty = !editsEqual(editSession.edit, editSession.baseline);
  $("#edit-save").disabled = !dirty;
  $("#edit-dirty").textContent = dirty ? "unsaved changes" : "";
}

let previewSeq = 0;

// The body for a preview request: whole frame when fitting, the visible window
// at device resolution when zoomed.
// Draft renders trade resolution for latency while something is being dragged.
const DRAFT_EDGE = 1100;

function previewBody(edit, draft) {
  const body = { rel_path: editSession.relPath, edit };
  if (draft) body.max_edge = DRAFT_EDGE;
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
      if (seq !== previewSeq) return;
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
      // Showing the render time makes "the editor feels slow" answerable
      // instead of a guess.
      $("#edit-status").textContent = ms ? `${ms} ms${draft ? " · draft" : ""}` : "";
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
  }
  syncEditSliders();
  drawCurve();
  drawOverlay();
  setEditDirty();
  fetchEditPreview(true);
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

async function saveEdit() {
  if (!editSession.relPath) return;
  const rel = editSession.relPath;
  const res = await fetch("/api/edit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rel_path: rel, edit: editPayload() }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    $("#edit-status").textContent = "save failed: " + (err.detail || res.status);
    return;
  }
  const updated = await res.json();
  const photo = state.photos.find((p) => p.rel_path === rel);
  if (photo) {
    if (updated.edit) { photo.edit = updated.edit; photo.edited_at = updated.edited_at; }
    else { delete photo.edit; delete photo.edited_at; }
  }
  editSession.baseline = mergeNeutralEdit(editSession.edit);
  setEditDirty();
  closeEditModal();
  renderMain();
}

function editNav(delta) {
  if (!state.filteredPhotos.length) return;
  const i = Math.max(0, Math.min(state.filteredPhotos.length - 1, editSession.idx + delta));
  if (i === editSession.idx) return;
  openEditModal(i);
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

function updateWatermark(patch, immediate) {
  const w = ensureWatermark();
  Object.assign(w, patch);
  if (patch.name != null) rememberWatermarkName(patch.name);
  renderWatermarkPanel();
  setEditDirty();
  fetchEditPreview(!!immediate);
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
      updateWatermark({ [k]: parseInt(e.target.value, 10) }));
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
  $$("#edit-modal .mask-add-btn").forEach((b) =>
    b.classList.toggle("armed", b.dataset.addMask === tool));
  const wrap = $(".edit-canvas-wrap");
  wrap.classList.toggle("tool-place", tool === "radial" || tool === "linear");
  wrap.classList.toggle("tool-brush", tool === "brush");
  wrap.classList.toggle("tool-crop", tool === "crop");
  const text = tool === "radial" ? "Drag on the photo to place the ellipse"
    : tool === "linear" ? "Drag on the photo to set the gradient direction"
    : tool === "brush" ? "Paint over the area · Alt = erase · [ ] = brush size"
    : tool === "crop" ? "Drag the box or its corners · ✂ again when you are done"
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

function addMask(kind) {
  if (!editSession.relPath) return;
  if (editSession.edit.masks.length >= MASK_MAX) {
    $("#edit-status").textContent = `mask limit reached (${MASK_MAX})`;
    return;
  }
  const r = overlayRect();
  editSession.edit.masks.push(newMask(kind, r.h ? r.w / r.h : 1));
  selectMask(editSession.edit.masks.length - 1);
  // Radial/gradient masks land centred so they are visible right away; arming
  // the tool lets the very next drag re-place them where the user wants.
  setEditTool(kind);
  setEditDirty();
  drawOverlay();
  if (kind !== "brush") fetchEditPreview(false);
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

function renderMaskList() {
  const list = $("#mask-list");
  if (!list) return;
  const masks = (editSession.edit && editSession.edit.masks) || [];
  const rows = [
    `<div class="mask-row${editSession.activeMask < 0 ? " active" : ""}" data-mask="-1">` +
    `<span class="mask-eye-spacer"></span><span class="mask-icon">▢</span>` +
    `<span class="mask-name">Global — whole photo</span></div>`,
  ];
  masks.forEach((m, i) => {
    const hint = maskAdjNeutral(m) ? "no effect yet"
      : (m.type === "brush" && !m.strokes.length) ? "nothing painted" : "";
    rows.push(
      `<div class="mask-row${i === editSession.activeMask ? " active" : ""}` +
      `${m.enabled ? "" : " off"}" data-mask="${i}">` +
      `<button class="mask-eye" data-mask-toggle="${i}" title="Show / hide this mask">` +
      `${m.enabled ? "◉" : "◌"}</button>` +
      `<span class="mask-icon">${MASK_KINDS[m.type].icon}</span>` +
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
  if (m.type === "brush") return;
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
    if (m && m.type === "brush" && editSession.tool === "brush") {
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
  settleTimer = setTimeout(() => fetchEditPreview(true, false), 240);
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
  $$("#edit-modal .mask-add-btn").forEach((b) =>
    b.addEventListener("click", () => addMask(b.dataset.addMask)));

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
      fetchEditPreview(false);
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
  const sl = $("#mask-brush-size");
  editSession.brush.size = Math.max(5, Math.min(300, editSession.brush.size + delta));
  sl.value = editSession.brush.size;
  $("#mask-brush-size-val").textContent = editSession.brush.size;
  drawOverlay();
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
  for (const sel of [$("#edit-preset-select"), $("#bulk-preset-select")]) {
    if (!sel) continue;
    const keep = selectId != null ? selectId : sel.value;
    const placeholder = sel.id === "bulk-preset-select" ? "Choose a preset…" : "Presets…";
    sel.innerHTML = `<option value="">${placeholder}</option>`
      + (builtin.length ? `<optgroup label="Built-in">${opts(builtin)}</optgroup>` : "")
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
  out.masks = out.masks.concat(over.masks).slice(0, MASK_MAX);
  return out;
}

function applyPreset(id) {
  const preset = state.presets.find((p) => p.id === id);
  if (!preset) return;
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
  const label = btn.textContent;
  btn.disabled = true;
  // RAW and edited photos are rendered at full size, which is not instant.
  btn.textContent = `⬇ saving ${rels.length}…`;
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
    btn.textContent = label;
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
}

function closeModal() {
  state.modal.open = false;
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
  img.src = "/img/" + enc(shownRel);
  img.className = state.modal.fit ? "fit" : "actual";
  $("#modal-title").textContent = shownRel;
  $("#modal-meta").textContent =
    `${state.modal.idx + 1}/${state.filteredPhotos.length} in ${photo.scene} · auto: ${photo.auto_suggestion || "—"}`;
  const cmp = $("#modal-compare");
  cmp.classList.toggle("hidden", !isHdr);
  cmp.classList.toggle("active", comparing);
  cmp.textContent = comparing ? "showing: 0 EV original" : "showing: HDR merged";
  if (isHdr) {
    // Preload the other version so the toggle is instant.
    new Image().src = "/img/" + enc(comparing ? photo.rel_path : photo.base);
  }
  const dec = $("#modal-decision");
  dec.className = photo.decision || "none";
  dec.textContent = (photo.decision || "—").toUpperCase();
  const ex = photo.exif || {};
  $("#modal-exif").textContent = [
    ex.camera, ex.lens, ex.focal, ex.aperture, ex.shutter, ex.iso_text,
  ].filter(Boolean).join("  ·  ");
  const s = photo.scores || {};
  const eye = s.eye_open != null ? s.eye_open.toFixed(3) : "—";
  const subject = s.subject_area != null
    ? ` · subject ${(s.subject_area * 100).toFixed(0)}%` +
      (s.subject_blur != null ? ` sharp ${s.subject_blur.toFixed(0)}` : "")
    : "";
  $("#modal-scores").textContent =
    `blur ${(s.blur ?? 0).toFixed(0)} (pct ${(s.blur_pct ?? 0).toFixed(2)}) · ` +
    `exp_z ${(s.exposure_zscore ?? 0).toFixed(2)} · eye ${eye}${subject} · ` +
    `badness ${(s.badness ?? 0).toFixed(2)}`;
  img.onload = () => { syncModalBoxes(); syncModalPeak(); drawLoupe(); };
  syncModalBoxes();
  syncModalPeak();
  drawLoupe();
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
}

function toggleCompare() {
  const photo = state.filteredPhotos[state.modal.idx];
  if (!photo || photo.type !== "hdr" || !photo.base) return;
  state.modal.compare = !state.modal.compare;
  renderModal();
}

// ---------- keyboard ----------
function bindKeys() {
  document.addEventListener("keydown", async (e) => {
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

    // Subject settings owns its text input; Esc only.
    if (!$("#subject-modal").classList.contains("hidden")) {
      if (k === "Escape") { closeSubjectModal(); e.preventDefault(); }
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
        else closeEditModal();
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
      if (k === "k" || k === "K") { togglePeak(); e.preventDefault(); return; }
      if (k === "l" || k === "L") { toggleLoupe(); e.preventDefault(); return; }
      if (k === "e" || k === "E") {
        const i = state.modal.idx; closeModal(); openEditModal(i); e.preventDefault(); return;
      }
      let decision = null, hasDecision = false, advance = true;
      if (k === "1" || k === "r" || k === "R") { decision = "reject"; hasDecision = true; }
      else if (k === "2" || k === "v" || k === "V") { decision = "review"; hasDecision = true; }
      else if (k === "3" || k === "p" || k === "P" || k === "a" || k === "A") { decision = "pick"; hasDecision = true; }
      else if (k === "u" || k === "U") { decision = null; hasDecision = true; advance = false; }
      else return;
      e.preventDefault();
      await decideAt(state.modal.idx, decision);
      if (advance && hasDecision && decision !== null) modalNav(+1);
      return;
    }

    // grid mode
    if (!state.filteredPhotos.length) return;
    const layout = LAYOUTS[state.pageSize];
    const cols = layout.cols;
    const i = state.cursorIdx;

    const total = state.filteredPhotos.length;
    const pageStart = pageIdx() * state.pageSize;
    const pageEnd = Math.min(pageStart + state.pageSize, total);

    if (k === "ArrowRight") {
      if (i + 1 < total) focusAt(i + 1);
      e.preventDefault(); return;
    }
    if (k === "ArrowLeft") {
      if (i - 1 >= 0) focusAt(i - 1);
      e.preventDefault(); return;
    }
    if (k === "ArrowDown") {
      let next = i + cols;
      // If the column-preserving step crosses to the next page, snap to its first tile.
      if (next >= pageEnd && pageEnd < total) next = pageEnd;
      focusAt(Math.min(total - 1, next));
      e.preventDefault(); return;
    }
    if (k === "ArrowUp") {
      let next = i - cols;
      // If the column-preserving step crosses to the previous page, snap to its last tile.
      if (next < pageStart && pageStart > 0) next = pageStart - 1;
      focusAt(Math.max(0, next));
      e.preventDefault(); return;
    }
    if (k === "PageDown" || k === "]") { gotoPage(+1); e.preventDefault(); return; }
    if (k === "PageUp" || k === "[") { gotoPage(-1); e.preventDefault(); return; }
    if (k === "Enter") { openModal(i); e.preventDefault(); return; }
    if (k === "e" || k === "E") { openEditModal(state.cursorIdx); e.preventDefault(); return; }
    if (k === "b" || k === "B") { toggleBoxes(); e.preventDefault(); return; }
    if (k === "x" || k === "X") { toggleSelect(state.cursorIdx, e.shiftKey); e.preventDefault(); return; }
    if ((k === "d" || k === "D") && state.selection.size) {
      downloadSelection(); e.preventDefault(); return;
    }
    if (k === "Escape") { if (state.selection.size) { clearSelection(); e.preventDefault(); } return; }

    let decision = null, hasDecision = false;
    if (k === "1" || k === "r" || k === "R") { decision = "reject"; hasDecision = true; }
    else if (k === "2" || k === "v" || k === "V") { decision = "review"; hasDecision = true; }
    else if (k === "3" || k === "p" || k === "P" || k === "a" || k === "A") { decision = "pick"; hasDecision = true; }
    else if (k === "u" || k === "U") { decision = null; hasDecision = true; }
    if (hasDecision) {
      e.preventDefault();
      await decideAt(i, decision);
    }
  });

  // Release hold-to-compare in the editor.
  document.addEventListener("keyup", (e) => {
    if ((e.key === "c" || e.key === "C") && !$("#edit-modal").classList.contains("hidden")) {
      editCompareOff();
    }
  });
}

// ---------- UI bindings ----------
function bindUi() {
  $$(".filter").forEach((b) => {
    b.addEventListener("click", () => {
      $$(".filter").forEach((x) => x.classList.toggle("active", x === b));
      state.filter = b.dataset.filter;
      state.cursorIdx = 0;
      recomputeFilter();
      renderMain();
    });
  });
  $$("#cols-toggle .cols").forEach((b) => {
    b.addEventListener("click", () => {
      $$("#cols-toggle .cols").forEach((x) => x.classList.toggle("active", x === b));
      state.pageSize = parseInt(b.dataset.cols, 10);
      renderMain();
    });
  });
  $("#prev-page").addEventListener("click", () => gotoPage(-1));
  $("#next-page").addEventListener("click", () => gotoPage(+1));
  $("#modal-close").addEventListener("click", closeModal);
  $("#modal-compare").addEventListener("click", toggleCompare);
  $("#rescore-btn").addEventListener("click", startRescore);
  // Subjects
  $("#subject-settings-btn").addEventListener("click", openSubjectModal);
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
  $("#export-cancel").addEventListener("click", closeExportModal);
  $("#export-confirm").addEventListener("click", confirmExport);
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
  bindMaskUi();
  bindWatermarkUi();
  curveInit();
  $("#edit-modal-close").addEventListener("click", closeEditModal);
  $("#edit-cancel").addEventListener("click", closeEditModal);
  $("#edit-save").addEventListener("click", saveEdit);
  $("#edit-auto").addEventListener("click", autoEdit);
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
  $("#switch-project-btn").addEventListener("click", async () => {
    if (!confirm("Close this project and open a different one?")) return;
    try {
      await fetch("/api/close", { method: "POST" });
    } catch {}
    location.reload();
  });
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
  $("#relink-browse").addEventListener("click", () => nativeBrowse($("#relink-new"), $("#relink-status")));
  $("#relink-confirm").addEventListener("click", confirmRelink);
  $("#wizard-close").addEventListener("click", () => {
    if (confirm("Cancel project setup?")) closeWizard();
  });
  $("#wizard-back").addEventListener("click", () => {
    if (wizardState.step > 1) showWizardStep(wizardState.step - 1);
  });
  $("#wizard-next").addEventListener("click", () => {
    const ok = validateWizardStep(wizardState.step);
    if (ok !== true) { alert(ok); return; }
    if (wizardState.step < 3) showWizardStep(wizardState.step + 1);
  });
  $("#wizard-create").addEventListener("click", createProject);
  $("#wiz-photo-browse").addEventListener("click", () =>
    nativeBrowse($("#wiz-photo-dir"), null));
  $("#wiz-project-name").addEventListener("input", syncWizardTargetHint);
  $$("#wiz-scene-cards .option-card").forEach((card) => {
    card.addEventListener("click", () => {
      wizardState.sceneMode = card.dataset.value;
      $$("#wiz-scene-cards .option-card").forEach((c) =>
        c.classList.toggle("active", c === card));
      $("#wiz-gap-row").style.display =
        card.dataset.value === "time_gap" ? "" : "none";
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
    alert("No people yet. Click ↻ group first.");
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
    <span class="drag-handle" title="Drag to reorder">⋮⋮</span>
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
      $("#score-title").textContent = `✓ Done · ${state.people.length} clusters`;
      $("#score-bar-fill").style.width = "100%";
      $("#score-progress-text").textContent =
        `Total faces grouped: ${state.people.reduce((s, p) => s + p.count, 0)}`;
      $("#score-current").textContent = "Click anywhere to dismiss · ⚙ in sidebar to edit labels";
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

async function startRescore() {
  const total = state.photos.length;
  const ok = confirm(
    `Re-score all ${total} photos, then re-group?\n\n` +
    `Scoring recomputes face embeddings, which discards the existing groups — ` +
    `so grouping runs straight afterwards in the same pass.\n` +
    `Decisions and edits are preserved, but disabled while it runs.`,
  );
  if (!ok) return;
  const res = await fetch("/api/score", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ with_faces: false }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert("Rescore failed: " + (err.detail || res.status));
    return;
  }
  $("#score-progress").classList.remove("hidden");
  $("#score-bar-fill").style.width = "0%";
  $("#score-progress-text").textContent = "starting…";
  $("#score-current").textContent = "";
  pollScoreStatus();
}

function pollScoreStatus() {
  if (scorePollTimer) clearInterval(scorePollTimer);
  let finished = false;
  const tick = async () => {
    if (finished) return;
    const res = await fetch("/api/score/status", { cache: "no-store" });
    if (!res.ok) { console.warn("score status fetch failed", res.status); return; }
    const s = await res.json();
    const pct = s.total ? Math.min(100, 100 * s.idx / s.total) : 0;
    const grouping = s.phase === "grouping";
    $("#score-title").textContent = grouping
      ? "Grouping…" : `Scoring ${state.photos.length} photos…`;
    $("#score-bar-fill").style.width = pct.toFixed(1) + "%";
    $("#score-progress-text").textContent = s.total
      ? `${s.idx}/${s.total} (${pct.toFixed(1)}%)`
      : "starting…";
    $("#score-current").textContent = s.current || (s.running ? "" : "finalizing…");
    if (!s.running) {
      finished = true;
      clearInterval(scorePollTimer);
      scorePollTimer = null;
      if (s.error) {
        $("#score-progress").classList.add("hidden");
        alert("Scoring failed: " + s.error);
        return;
      }
      const prevScene = state.selectedScene;
      await loadDb();
      const totalFaces = state.photos.reduce((s, p) => s + (p.faces?.length || 0), 0);
      const groups = state.subjects.vehicles.length;
      $("#score-title").textContent = `✓ Scored ${state.photos.length} photos`;
      $("#score-bar-fill").style.width = "100%";
      // Scoring throws the groups away and rebuilds them in the same run, so the
      // summary reports both rather than telling anyone to press another button.
      $("#score-progress-text").textContent =
        `${totalFaces} faces · ${state.people.length} people`
        + (groups ? ` · ${groups} subject groups` : "");
      $("#score-current").textContent = "Click anywhere to dismiss";
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
const workspaceState = { current: null, list: [] };

async function loadWorkspaces() {
  try {
    const res = await fetch("/api/workspaces", { cache: "no-store" });
    if (res.ok) {
      const d = await res.json();
      workspaceState.list = d.workspaces || [];
      workspaceState.current = d.current || (workspaceState.list[0] || null);
    }
  } catch {}
  renderWorkspaceSelect();
  loadWorkspaceProjects();
}

function renderWorkspaceSelect() {
  const sel = $("#workspace-select");
  sel.innerHTML = workspaceState.list
    .map((w) => `<option value="${escapeAttr(w)}">${escapeHtml(basename(w) || w)}</option>`)
    .join("");
  if (workspaceState.current) sel.value = workspaceState.current;
  $("#workspace-path").textContent = workspaceState.current || "";
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
  loadWorkspaceProjects();
}

async function addWorkspace() {
  const status = $("#landing-status");
  status.textContent = "Choose a workspace folder…";
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial: workspaceState.current || null }),
    });
    const result = res.ok ? await res.json() : {};
    if (!result.path) { status.textContent = ""; return; }
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

function renderProjectGrid(projects) {
  const wrap = $("#workspace-projects");
  wrap.innerHTML = "";
  $("#workspace-proj-count").textContent = projects.length;

  const add = document.createElement("button");
  add.className = "project-card project-new";
  add.type = "button";
  add.innerHTML = `<span class="project-new-icon">＋</span><span class="project-new-label">New project</span>`;
  add.addEventListener("click", openWizard);
  wrap.appendChild(add);

  for (const p of projects) {
    const card = document.createElement("div");
    card.className = "project-card" + (p.photos_exist ? "" : " missing");
    const decided = p.scored_at ? `${p.decided}/${p.photos} decided` : "not scored";
    card.innerHTML = `
      <span class="project-name">${escapeHtml(p.name)}</span>
      <span class="project-meta">${p.photos} photo${p.photos === 1 ? "" : "s"} · ${decided}</span>
      <span class="project-path">${escapeHtml(p.photo_dir || "")}</span>
      ${p.photos_exist ? "" : `<span class="project-missing">⚠ photos not found — click to re-link</span>`}
      <button class="project-del" type="button" title="Delete this project (photos are kept)">🗑</button>`;
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
      ? `<div class="recent-path">📂 ${escapeHtml(r.project_dir)}</div>
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

function basename(p) {
  if (!p) return "";
  const s = String(p).replace(/\/+$/, "");
  const i = s.lastIndexOf("/");
  return i >= 0 ? s.slice(i + 1) : s;
}
function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

async function nativeBrowse(targetInput, status) {
  const initial = targetInput.value.trim() || null;
  if (status) status.textContent = "Opening Finder dialog…";
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial }),
    });
    if (!res.ok) {
      if (status) status.textContent = `Browse endpoint failed (${res.status}).`;
      return;
    }
    const result = await res.json();
    if (result.path) targetInput.value = result.path;
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
  $("#score-progress").classList.remove("hidden");
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
    <p>Re-clustering (↻ in the sidebar) resets labels and priorities because
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

// ---------- welcome banner ----------
const WELCOME_BANNER_KEY = "pcls.welcomeBannerSeen";

function maybeShowWelcomeBanner() {
  try {
    if (localStorage.getItem(WELCOME_BANNER_KEY) === "1") return;
  } catch {}
  $("#welcome-banner").classList.remove("hidden");
}

function dismissWelcomeBanner() {
  $("#welcome-banner").classList.add("hidden");
  try { localStorage.setItem(WELCOME_BANNER_KEY, "1"); } catch {}
}

// ---------- new-project wizard ----------
const wizardState = { step: 1, sceneMode: "folder", subjectPreset: "" };

function openWizard() {
  if (!workspaceState.current) {
    alert("Add a workspace first (a folder to hold your projects).");
    return;
  }
  wizardState.step = 1;
  wizardState.sceneMode = "folder";
  wizardState.subjectPreset = "";
  $("#wiz-photo-dir").value = "";
  $("#wiz-jpeg-subdir").value = "";
  $("#wiz-raw-subdir").value = "";
  $("#wiz-project-name").value = "";
  $("#wiz-gap").value = 30;
  $("#wiz-gap-row").style.display = "none";
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
  if (onLast) renderWizardSummary();
}

function syncWizardTargetHint() {
  const name = $("#wiz-project-name").value.trim() || "<name>";
  const ws = workspaceState.current || "<workspace>";
  $("#wiz-target-hint").textContent = `${ws}/${name}/`;
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
  status.textContent = "Choose a project folder…";
  try {
    const res = await fetch("/api/browse-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ initial: null }),
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
    const pct = s.total ? Math.min(100, 100 * s.idx / s.total) : 0;
    $("#score-title").textContent = (s.phase === "scoring") ? "Scoring photos…"
      : (s.phase === "clustering") ? "Clustering faces…"
      : (s.phase === "scanning") ? "Scanning…"
      : (s.phase === "loading") ? "Loading project…"
      : "Opening project…";
    $("#score-bar-fill").style.width = pct.toFixed(1) + "%";
    $("#score-progress-text").textContent = s.message || (s.total ? `${s.idx}/${s.total}` : "starting…");
    $("#score-current").textContent = s.current || "";
    if (!s.running) {
      finished = true;
      clearInterval(scorePollTimer);
      scorePollTimer = null;
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
    if (scene === view.scene && view.page) gotoPageIndex(view.page);
  } else {
    renderMain();
  }
  viewRestored = true;
  maybeShowWelcomeBanner();
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
  initTooltips();
  loupeInit();
  $("#welcome-banner-dismiss").addEventListener("click", dismissWelcomeBanner);
  let s;
  try { s = await fetchState(); } catch { showLanding(); return; }
  if (s.opening && s.opening.running) {
    hideLanding();
    $("#score-title").textContent = "Opening project…";
    $("#score-progress").classList.remove("hidden");
    pollOpenStatus();
    return;
  }
  if (!s.ready) { showLanding(); return; }
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
