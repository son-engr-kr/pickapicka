"""FastAPI app: landing page, web viewer, full-size images, on-demand thumbs,
face crops, and persistence of decisions/clusters/scene-grouping."""
from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from . import (
    cameras, db, editing, exifinfo, exporting, film as film_mod, folderinfo, hdr,
    metadata as metadata_mod, presets as presets_mod, raw, relink, scenes, segment as segment_mod,
    userstate, watermark as watermark_mod,
)
from .scoring import objects as objects_mod
from .scorer import (
    SUPPORTED_EXTS,
    _is_supported,
    apply_scene_suggestions,
    excluded_scan_dirs,
    source_rel_paths,
    walk_files,
)

THUMB_LONG_EDGE = 1280
THUMB_QUALITY = 90
# Bump when the *way* a thumb is baked changes rather than its size. The edit
# hash in the filename cannot notice that the same edit now renders differently,
# so without this a cached thumb would outlive the change.
THUMB_VERSION = "v2"
FACE_LONG_EDGE = 360
FACE_QUALITY = 90
FACE_PADDING = 0.85  # crop half-side = max(w, h) * FACE_PADDING (about 70% padding around face)
WEB_DIR = Path(__file__).parent / "web"

EDIT_PREVIEW_EDGE = 2048   # long edge the live editor previews at (cached, re-graded on drag)
EDIT_VIEW_EDGE = 2560      # long edge the full-size viewer renders edited/RAW photos at
EDIT_BASE_CACHE_MAX = 4    # decoded-base LRU size (RAW A/B benefits from >1)
FULL_BASE_CACHE_MAX = 2    # full-res decode LRU: big arrays, so keep very few
EDIT_ZOOM_MAX_OUT = 3200   # cap on the pixels a 1:1 request may return
# Per-photo edit slots: a scratchpad for "try this, keep that". Six fits one
# row of chips in the editor and is more variants than anyone compares at once,
# and the cap is what keeps picks.json sane — every slot is a whole edit dict,
# masks and all.
EDIT_SLOTS = 6

Decision = Literal["pick", "review", "reject"]


# ----- payload models -----------------------------------------------------

class DecidePayload(BaseModel):
    rel_path: str
    decision: Decision | None


class BulkDecidePayload(BaseModel):
    rel_paths: list[str]
    decision: Decision | None


class ScorePayload(BaseModel):
    with_faces: bool = False
    # Explicit HDR bracket grouping (HDR-tab edit); None re-detects from EXIF.
    bracket_groups: list[list[str]] | None = None
    # Real-estate "look" for merged results; None keeps the project's setting.
    hdr_look: dict[str, float] | None = None
    # COCO classes to detect as the subject; None keeps the project's setting,
    # [] turns subject detection off.
    subject_classes: list[str] | None = None
    # Whether to look for faces at all; None keeps the project's setting. A car
    # shoot does not want the bystanders clustered into People.
    detect_faces: bool | None = None


class HdrPreviewPayload(BaseModel):
    members: list[str]
    look: dict[str, float] = {}


class MaskPreviewPayload(BaseModel):
    rel_path: str
    mask: dict[str, Any]


class EditPreviewPayload(BaseModel):
    rel_path: str
    edit: dict[str, Any] = {}
    # 1:1 view: [x0, y0, w, h] normalized window of the photo, and how wide the
    # answer should come back. None means "the whole frame, preview sized".
    roi: list[float] | None = None
    out_w: int | None = None
    # Draft renders during a drag: grade a smaller copy so the frame keeps up,
    # then the client asks again at full size once the pointer settles.
    max_edge: int | None = None
    # The crop tool shows the straightened frame *uncropped*, so the box can be
    # dragged over everything still available. The tilt is honoured either way.
    skip_crop: bool = False


class EditSavePayload(BaseModel):
    rel_path: str
    edit: dict[str, Any] = {}


class EditBulkPayload(BaseModel):
    rel_paths: list[str]
    edit: dict[str, Any] = {}
    # "add" lays the edit over whatever each photo already has (masks append);
    # "replace" overwrites, which is what this endpoint has always done.
    mode: Literal["replace", "add"] = "replace"


class AutoTonePayload(BaseModel):
    rel_path: str


# Where the white-balance picker was clicked, as a fraction of the original
# frame — the space the editor's overlay already reports for masks.
class NeutralPickPayload(BaseModel):
    rel_path: str
    x: float
    y: float


# One of a photo's edit slots. `edit: None` clears the slot; anything else
# replaces what is in it.
class EditSlotPayload(BaseModel):
    rel_path: str
    slot: int
    edit: dict[str, Any] | None = None
    name: str | None = None


# Where the grid was left. `scene` is remembered too because a page number only
# means anything inside the scene it was counted in.
class ViewPayload(BaseModel):
    filter: Literal["all", "undecided", "pick", "review", "reject", "edited"]
    page_size: Literal[1, 2, 4, 8]
    page: int
    scene: str | None = None


class PresetSavePayload(BaseModel):
    name: str
    edit: dict[str, Any] = {}


# Every field optional: what is not sent falls back to the project's saved
# settings, and what has never been set falls back to cluster.DEFAULT_SETTINGS.
class ClusterPayload(BaseModel):
    face_eps: float | None = None
    face_min_samples: int | None = None
    group_subjects: bool | None = None
    subject_eps: float | None = None
    subject_min_samples: int | None = None
    subject_min_area: float | None = None
    subject_min_score: float | None = None


class PersonUpdate(BaseModel):
    id: str
    label: str
    priority: int
    excluded: bool = False


class PeoplePayload(BaseModel):
    people: list[PersonUpdate]


class SubjectGroupUpdate(BaseModel):
    id: str
    label: str
    excluded: bool = False


class SubjectGroupsPayload(BaseModel):
    groups: list[SubjectGroupUpdate]


class ExportSettings(BaseModel):
    """What an export writes. The defaults are what an export always was
    (full-size JPEG at 95), now with its metadata kept."""
    format: exporting.Format = "jpeg"
    quality: int = 95
    long_edge: int | None = None          # None: full size
    metadata: metadata_mod.MetadataMode = "all"
    name_template: str = "{name}"


class ExportPayload(BaseModel):
    target_dir: str | None = None
    mode: Literal["folder", "flat", "by_person"] = "folder"
    settings: ExportSettings = ExportSettings()


class NamePreviewPayload(BaseModel):
    template: str


class QuitPayload(BaseModel):
    force: bool = False   # quit even with a task still running


class OpenPayload(BaseModel):
    photo_dir: str
    jpeg_subdir: str = ""
    db_path: str | None = None


class OpenProjectPayload(BaseModel):
    project_dir: str


class CreateProjectPayload(BaseModel):
    name: str
    workspace_dir: str
    photo_dir: str
    jpeg_subdir: str = ""
    raw_subdir: str = ""
    scene_grouping_mode: Literal["folder", "time_gap"] = "folder"
    scene_grouping_gap_minutes: int = 30
    subject_classes: list[str] = []


class DeleteProjectPayload(BaseModel):
    project_dir: str


class DownloadPayload(BaseModel):
    rel_paths: list[str]


class RevealPayload(BaseModel):
    path: str


class BrowsePayload(BaseModel):
    initial: str | None = None
    # What the folder is for, which picks the dialog's title. A fixed set rather
    # than free text: the title is spliced into an AppleScript on macOS.
    purpose: Literal["photos", "workspace", "project", "relink", "export"] = "photos"


class InspectPayload(BaseModel):
    path: str
    workspace: str | None = None


class ForgetPayload(BaseModel):
    db_path: str | None = None
    project_dir: str | None = None


class SceneGroupingPayload(BaseModel):
    mode: Literal["folder", "time_gap"]
    gap_minutes: int = 30


class WorkspacePayload(BaseModel):
    dir: str


class RelinkPayload(BaseModel):
    new_photo_dir: str
    db_path: str | None = None
    project_dir: str | None = None


# ----- app context --------------------------------------------------------

class AppContext:
    """Holds all per-project mutable state. Swapped on /api/open."""

    def __init__(self) -> None:
        self.db_path: Path | None = None
        self.project_dir: Path | None = None  # set when running in project layout
        self.data: dict[str, Any] = {}
        self.photo_root: Path | None = None
        self.jpeg_root: Path | None = None
        self.thumbs_root: Path | None = None
        self.faces_root: Path | None = None
        self.peaks_root: Path | None = None  # focus-peaking overlays
        self.hdr_root: Path | None = None
        self.raw_root: Path | None = None        # split-layout RAW tree (optional)
        self.raw_cache_root: Path | None = None  # cached RAW preview JPEGs
        self.photo_index: dict[str, dict[str, Any]] = {}
        # Single-entry cache of the last previewed bracket's fused array, so
        # dragging the look sliders re-grades instead of re-fusing.
        self.hdr_fuse_cache: tuple[tuple[str, ...], Any] | None = None
        # LRU of decoded, downscaled *original* RGB arrays keyed by rel_path, so
        # dragging the editor sliders re-grades instead of re-decoding (RAW
        # decode is expensive). Holds originals — survives edit saves.
        self.edit_base_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        # Full-resolution originals for the 1:1 zoom view (see get_full_base).
        self.full_base_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.decode_lock = threading.Lock()

        self.save_lock = threading.Lock()
        self.score_lock = threading.Lock()
        self.scoring_state: dict[str, Any] = self._fresh_scoring_state()
        self.cluster_state: dict[str, Any] = self._fresh_cluster_state()
        self.opening_state: dict[str, Any] = self._fresh_opening_state()
        self.export_state: dict[str, Any] = self._fresh_export_state()

    @staticmethod
    def _fresh_scoring_state() -> dict[str, Any]:
        # `phase` is "scoring" or "grouping": one run does both, because scoring
        # discards the groups and stopping in between leaves the project with
        # faces nobody has been grouped into.
        return {"running": False, "phase": None, "idx": 0, "total": 0,
                "current": None, "started_at": None, "ended_at": None,
                "error": None}

    @staticmethod
    def _fresh_cluster_state() -> dict[str, Any]:
        return {"running": False, "phase": None, "idx": 0, "total": 0,
                "started_at": None, "ended_at": None, "error": None}

    @staticmethod
    def _fresh_export_state() -> dict[str, Any]:
        # `result` is filled when a run finishes, cancelled or not, so the dialog
        # can say what was written either way.
        return {"running": False, "idx": 0, "total": 0, "current": None,
                "cancel": False, "error": None, "result": None,
                "started_at": None, "ended_at": None}

    @staticmethod
    def _fresh_opening_state() -> dict[str, Any]:
        return {"running": False, "phase": None, "message": None,
                "idx": 0, "total": 0, "current": None,
                "started_at": None, "ended_at": None, "error": None,
                "ready": False, "needs_relink": None}

    def is_loaded(self) -> bool:
        return self.db_path is not None

    def close(self) -> None:
        """Drop the current project so the landing page is shown again.
        Caches on disk are kept (cheap to invalidate via mtime checks)."""
        self.db_path = None
        self.project_dir = None
        self.data = {}
        self.photo_root = None
        self.jpeg_root = None
        self.thumbs_root = None
        self.faces_root = None
        self.peaks_root = None
        self.hdr_root = None
        self.raw_root = None
        self.raw_cache_root = None
        self.photo_index = {}
        self.hdr_fuse_cache = None
        self.edit_base_cache.clear()
        self.full_base_cache.clear()
        self.opening_state = self._fresh_opening_state()

    def load_db(self, db_path: Path) -> None:
        """Legacy: adopt a `picks.json` that lives next to the photos."""
        db_path = db_path.resolve()
        data = db.load(db_path)
        photo_root = Path(data["photo_root"])
        jpeg_subdir = data.get("jpeg_subdir", "")
        jpeg_root = photo_root / jpeg_subdir if jpeg_subdir else photo_root

        self.db_path = db_path
        self.project_dir = None
        self.data = data
        self.photo_root = photo_root
        self.jpeg_root = jpeg_root
        self.thumbs_root = db_path.with_suffix(db_path.suffix + ".thumbs")
        self.faces_root = db_path.with_suffix(db_path.suffix + ".faces")
        self.peaks_root = db_path.with_suffix(db_path.suffix + ".peaks")
        self.hdr_root = db_path.with_suffix(db_path.suffix + ".hdr")
        raw_subdir = data.get("raw_subdir", "")
        self.raw_root = photo_root / raw_subdir if raw_subdir else jpeg_root
        self.raw_cache_root = db_path.with_suffix(db_path.suffix + ".rawcache")
        self.thumbs_root.mkdir(exist_ok=True)
        self.faces_root.mkdir(exist_ok=True)
        self.peaks_root.mkdir(exist_ok=True)
        self.hdr_root.mkdir(exist_ok=True)
        self.raw_cache_root.mkdir(exist_ok=True)
        self._rebuild_index()
        self.opening_state["ready"] = True

    def load_project(self, project_dir: Path) -> None:
        """New: adopt a project directory containing picks.json + .cache/."""
        project_dir = project_dir.resolve()
        db_path = project_dir / "picks.json"
        assert db_path.is_file(), f"not a project directory (no picks.json): {project_dir}"
        data = db.load(db_path)
        photo_root = Path(data["photo_root"])
        jpeg_subdir = data.get("jpeg_subdir", "")
        jpeg_root = photo_root / jpeg_subdir if jpeg_subdir else photo_root

        self.db_path = db_path
        self.project_dir = project_dir
        self.data = data
        self.photo_root = photo_root
        self.jpeg_root = jpeg_root
        cache_root = project_dir / ".cache"
        self.thumbs_root = cache_root / "thumbs"
        self.faces_root = cache_root / "faces"
        self.peaks_root = cache_root / "peaks"
        self.hdr_root = project_dir / "hdr"
        raw_subdir = data.get("raw_subdir", "")
        self.raw_root = photo_root / raw_subdir if raw_subdir else jpeg_root
        self.raw_cache_root = cache_root / "raw"
        self.thumbs_root.mkdir(parents=True, exist_ok=True)
        self.faces_root.mkdir(parents=True, exist_ok=True)
        self.peaks_root.mkdir(parents=True, exist_ok=True)
        self.hdr_root.mkdir(parents=True, exist_ok=True)
        self.raw_cache_root.mkdir(parents=True, exist_ok=True)
        self._rebuild_index()
        self.opening_state["ready"] = True

    def reload_data(self) -> None:
        assert self.db_path is not None
        new_data = db.load(self.db_path)
        self.data = new_data
        self._rebuild_index()
        # A re-score may have re-merged HDR results (pixels changed); drop the
        # decoded-base cache so previews decode the fresh files.
        self.edit_base_cache.clear()
        self.full_base_cache.clear()

    def _rebuild_index(self) -> None:
        self.photo_index = {p["rel_path"]: p for p in self.data.get("photos", [])}

    def wipe_face_cache(self) -> None:
        if self.faces_root and self.faces_root.exists():
            shutil.rmtree(self.faces_root, ignore_errors=True)
            self.faces_root.mkdir(exist_ok=True)

    def source_path(self, rel_path: str) -> Path:
        """Absolute path of a photo's image file. A merged HDR result lives in
        the HDR output directory; a RAW under the photo root (its `src`); every
        other photo under the JPEG root."""
        if hdr.is_hdr_rel(rel_path):
            assert self.hdr_root is not None
            return self.hdr_root / rel_path[len(hdr.HDR_PREFIX) + 1:]
        photo = self.photo_index.get(rel_path)
        if photo is not None and photo.get("type") == "raw":
            assert self.photo_root is not None
            return self.photo_root / photo.get("src", rel_path)
        assert self.jpeg_root is not None
        return self.jpeg_root / rel_path

    def _decode_scaled(self, rel_path: str, max_edge: int | None) -> np.ndarray:
        """Oriented RGB uint8 for a photo, optionally downscaled so its long edge
        is <= max_edge. Handles JPEG/PNG and merged HDR results; RAW is decoded
        via rawpy (already oriented by libraw — no exif_transpose)."""
        photo = self.photo_index.get(rel_path)
        if photo is not None and photo.get("type") == "raw":
            from . import raw
            arr = raw.decode_raw(self.source_path(rel_path))
            if max_edge and max(arr.shape[:2]) > max_edge:
                arr = raw.fit_within(arr, max_edge)
            return np.ascontiguousarray(arr)
        src = self.source_path(rel_path)
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if max_edge and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            return np.ascontiguousarray(np.asarray(im))

    def decode_full(self, rel_path: str) -> np.ndarray:
        """Full-resolution oriented RGB uint8 (used for export baking)."""
        return self._decode_scaled(rel_path, None)

    def decode_view(self, rel_path: str) -> np.ndarray:
        """RGB uint8 capped at EDIT_VIEW_EDGE — for rendering the full-size
        viewer image of an edited or RAW photo."""
        return self._decode_scaled(rel_path, EDIT_VIEW_EDGE)

    def get_decoded_base(self, rel_path: str) -> np.ndarray:
        """LRU-cached downscaled (~EDIT_PREVIEW_EDGE) original RGB — the array the
        live editor re-grades on each slider drag (decode happens once)."""
        with self.decode_lock:
            arr = self.edit_base_cache.get(rel_path)
            if arr is not None:
                self.edit_base_cache.move_to_end(rel_path)
                return arr
        arr = self._decode_scaled(rel_path, EDIT_PREVIEW_EDGE)
        with self.decode_lock:
            self.edit_base_cache[rel_path] = arr
            while len(self.edit_base_cache) > EDIT_BASE_CACHE_MAX:
                self.edit_base_cache.popitem(last=False)
        return arr

    def auto_fields(self, rel_path: str,
                    edit: dict[str, Any] | None) -> dict[str, np.ndarray] | None:
        """Segmentation fields for whatever automatic masks `edit` actually uses.

        This exists here rather than in `editing` because of two constraints that
        module cannot satisfy from the inside (see its note on MASK_TYPES). The
        segmenter has to see the whole frame — shown only a 1:1 window it would
        find a different subject than the fit preview found — and it has to see
        the *original* pixels, or the mask would crawl every time a slider moved.
        The cached preview decode is both of those things.

        Returns None when no automatic mask is active, so a photo without one
        never pays for the model, not even to check whether it is downloaded.
        """
        groups = {m["group"] for m in editing.normalize(edit)["masks"]
                  if m["type"] == "auto" and editing.mask_is_active(m)}
        if not groups:
            return None
        base = self.get_decoded_base(rel_path)
        return {g: segment_mod.class_mask(base, g, key=rel_path) for g in groups}

    def range_src(self, rel_path: str,
                  edit: dict[str, Any] | None) -> np.ndarray | None:
        """The whole ungraded frame, for any mask carrying a range refinement.

        `editing.render` needs the SAME array for every render of a photo, or the
        selector's fixed grid would be derived from a different input for an
        export than for a preview and the two would select slightly different
        things. The cached preview decode is that array — deliberately not
        `decode_full`, which would be a different one.

        None when no mask asks for it, so a photo without a range refinement
        never pays for a decode it does not need.
        """
        wanted = any(m["range_luma"] is not None or m["range_color"] is not None
                     for m in editing.normalize(edit)["masks"]
                     if editing.mask_is_active(m))
        return self.get_decoded_base(rel_path) if wanted else None

    def photo_meta(self, rel_path: str) -> dict[str, Any]:
        """Shooting info for the watermark and the info panel. Normally cached
        in the db by the scorer; projects scored before that existed get it read
        (and remembered) on first use, so nothing needs a re-score."""
        photo = self.photo_index.get(rel_path)
        if photo is None:
            return dict(exifinfo.EMPTY)
        info = photo.get("exif")
        if info is None:
            # A RAW is read for its own EXIF, never through _thumb_source: that
            # would decode a preview this does not want in order to reach a file
            # whose metadata is the camera's, not the shot's.
            is_raw = photo.get("type") == "raw"
            src = self.source_path(rel_path) if is_raw else _thumb_source(self, rel_path, photo)
            if not src.is_file():
                # Photo volume detached. Report nothing for now, and do not
                # remember it: caching EMPTY would outlive the drive coming back.
                return {**exifinfo.EMPTY, "file": Path(rel_path).stem}
            info = exifinfo.read_any(src, is_raw=is_raw)
            photo["exif"] = info
        return {**info, "file": Path(rel_path).stem}

    def get_draft_base(self, rel_path: str, max_edge: int) -> np.ndarray:
        """A downscaled copy of the preview base, cached alongside it. Grading
        a quarter of the pixels is what keeps a mask drag interactive when
        several layers are stacked."""
        key = f"{rel_path}@{max_edge}"
        with self.decode_lock:
            arr = self.edit_base_cache.get(key)
            if arr is not None:
                self.edit_base_cache.move_to_end(key)
                return arr
        base = self.get_decoded_base(rel_path)
        h, w = base.shape[:2]
        scale = max_edge / max(h, w)
        arr = base if scale >= 1.0 else cv2.resize(
            base, (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA)
        with self.decode_lock:
            self.edit_base_cache[key] = arr
            while len(self.edit_base_cache) > EDIT_BASE_CACHE_MAX * 2:
                self.edit_base_cache.popitem(last=False)
        return arr

    def get_full_base(self, rel_path: str) -> np.ndarray:
        """LRU-cached *full-resolution* original — what the 1:1 editor view
        grades. Kept to a couple of entries because a 24 MP frame is ~70 MB, but
        one entry is what makes panning around at 100% feel instant."""
        with self.decode_lock:
            arr = self.full_base_cache.get(rel_path)
            if arr is not None:
                self.full_base_cache.move_to_end(rel_path)
                return arr
        arr = self._decode_scaled(rel_path, None)
        with self.decode_lock:
            self.full_base_cache[rel_path] = arr
            while len(self.full_base_cache) > FULL_BASE_CACHE_MAX:
                self.full_base_cache.popitem(last=False)
        return arr


# ----- helpers ------------------------------------------------------------

def _within_roots(ctx: "AppContext", path: Path) -> bool:
    """True when an already-resolved `path` lies inside the JPEG root or the
    HDR output directory — guards the image routes against path traversal."""
    roots = [ctx.jpeg_root.resolve()]
    if ctx.hdr_root is not None:
        roots.append(ctx.hdr_root.resolve())
    if ctx.photo_root is not None:
        roots.append(ctx.photo_root.resolve())  # RAW sources live under the photo root
    if ctx.raw_cache_root is not None:
        roots.append(ctx.raw_cache_root.resolve())  # RAW preview JPEGs (project .cache/raw)
    return any(path == r or r in path.parents for r in roots)


# Focus peaking sensitivity: (sharpness ratio threshold, gradient floor).
# The ratio is scale-free, the floor is on a 0-255 gradient magnitude.
PEAK_LEVELS: dict[str, tuple[float, float]] = {
    "tight": (0.60, 45.0),
    "normal": (0.45, 35.0),
    "loose": (0.30, 25.0),
}
_PEAK_PROBE = 1.0     # sigma of the test blur used to measure edge steepness
_PEAK_K5 = np.ones((5, 5), np.uint8)
_PEAK_K3 = np.ones((3, 3), np.uint8)
# Widening the ridges is what lets the marks survive the browser scaling the
# overlay down to tile size. A plus rather than a full square: it covers a bit
# over half the pixels, which reads as fine dots instead of blobs.
_PEAK_WIDEN = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# Bump when the detector changes: cached overlays are only invalidated by the
# thumbnail's mtime, so without this an upgrade would keep serving old ones.
_PEAK_VERSION = "v3"


def _grad_mag(img: np.ndarray) -> np.ndarray:
    return cv2.magnitude(cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3),
                         cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3))


def _ensure_peak(thumb: Path, peaks_root: Path, rel_path: str, ehash: str,
                 level: str = "normal") -> Path:
    """Build (and cache) the focus-peaking overlay for a thumbnail.

    Sharpness is "how much does a small extra blur change this edge". Blur an
    already-soft edge and little changes; blur a crisp one and its gradient
    collapses. The ratio of the two gradient peaks is therefore a measure of
    *transition steepness* — and, crucially, it is independent of how much
    contrast the edge has, so a hard-lit but out-of-focus boundary no longer
    passes as sharp the way a plain Laplacian threshold let it.

    Two gates keep the result honest:
      - an absolute gradient floor, because a ratio computed on noise or on a
        flat wall is meaningless (and enormous);
      - non-maximum suppression, so only the ridge of an edge lights up. That
        is what makes the overlay read as fine contours rather than a wash.

    Nothing here is relative to the frame's own statistics: a photo where
    nothing is in focus now gets an empty overlay instead of its least-blurry
    region being promoted.
    """
    ratio_min, grad_floor = PEAK_LEVELS.get(level, PEAK_LEVELS["normal"])
    tag = f"{('.' + ehash) if ehash else ''}.{level}.{_PEAK_VERSION}"
    dst = peaks_root / f"{rel_path}{tag}.peak.png"

    def fresh() -> bool:
        return dst.exists() and dst.stat().st_mtime >= thumb.stat().st_mtime

    if fresh():
        return dst
    lock = _cache_lock(dst)
    lock.acquire()
    try:
        if fresh():
            return dst
        return _build_peak(thumb, dst, ratio_min, grad_floor)
    finally:
        lock.release()


def _build_peak(thumb: Path, dst: Path, ratio_min: float, grad_floor: float) -> Path:
    gray = cv2.imread(str(thumb), cv2.IMREAD_GRAYSCALE)
    assert gray is not None, f"failed to read {thumb}"
    # A touch of smoothing first: single-pixel sensor noise would otherwise ace
    # the sharpness test, since blurring wipes it out completely.
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.5)
    raw = _grad_mag(g)
    probed = _grad_mag(cv2.GaussianBlur(g, (0, 0), _PEAK_PROBE))
    # Compare peaks, not averages: an edge's total gradient energy is the same
    # however blurred it is, so averaging first would erase the signal.
    sharpness = cv2.dilate(raw, _PEAK_K5) / (cv2.dilate(probed, _PEAK_K5) + 1e-3) - 1.0
    ridge = raw >= cv2.dilate(raw, _PEAK_K3) - 1e-6

    hit = (sharpness > ratio_min) & (raw > grad_floor) & ridge
    strength = np.clip((sharpness - ratio_min) / max(ratio_min * 0.8, 1e-3), 0.0, 1.0)
    alpha = np.where(hit, 150 + strength * 105, 0).astype(np.uint8)
    # Widen the ridges: a one-pixel contour vanishes when the browser scales the
    # overlay down to grid-tile size, which is where peaking is most useful.
    alpha = cv2.dilate(alpha, _PEAK_WIDEN)

    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255   # a saturated red-orange reads on paint and asphalt
    rgba[..., 1] = 90
    rgba[..., 2] = 20
    rgba[..., 3] = alpha
    with _atomic_write(dst) as tmp:
        cv2.imwrite(str(tmp), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    return dst


def _check_export_settings(settings: ExportSettings) -> None:
    """Refuse settings the writer would otherwise have to guess about."""
    if not 50 <= settings.quality <= 100:
        raise HTTPException(status_code=400, detail="JPEG quality must be 50 to 100")
    if settings.long_edge is not None and not (
            exporting.LONG_EDGE_MIN <= settings.long_edge <= exporting.LONG_EDGE_MAX):
        raise HTTPException(
            status_code=400,
            detail=f"long edge must be {exporting.LONG_EDGE_MIN} to {exporting.LONG_EDGE_MAX} px")
    try:
        exporting.check_template(settings.name_template)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _copies_as_is(ctx: "AppContext", photo: dict[str, Any], rel: str,
                  settings: ExportSettings) -> bool:
    # A merged HDR result is an OpenCV-written JPEG with no metadata of its own,
    # so it is always written fresh, carrying its 0 EV frame's.
    needs = _needs_render(photo) or photo.get("type") == "hdr"
    return exporting.can_copy(ctx.source_path(rel), needs_render=needs, settings=settings)


def _baked_name(ctx: "AppContext", photo: dict[str, Any], rel: str,
                settings: ExportSettings, stem: str | None = None) -> str:
    """Output file name. A copied original keeps its own extension; anything
    written fresh takes the export format's."""
    stem = Path(rel).stem if stem is None else stem
    if _copies_as_is(ctx, photo, rel, settings):
        return stem + Path(rel).suffix
    return stem + exporting.EXTENSIONS[settings.format]


def _templated_stem(ctx: "AppContext", photo: dict[str, Any], rel: str,
                    settings: ExportSettings, seq: int, width: int) -> str:
    project = ctx.project_dir.name if ctx.project_dir else ctx.db_path.stem
    stem = exporting.render_name(
        settings.name_template, name=Path(rel).stem, seq=seq, width=width,
        captured_at=ctx.photo_meta(rel).get("captured_at") or "",
        scene=photo.get("scene") or "", project=project)
    return _sanitize_segment(stem)


def _metadata_source(ctx: "AppContext", photo: dict[str, Any], rel: str) -> tuple[Path, bool]:
    """The file whose EXIF and colour profile an export carries, and whether
    it is a RAW."""
    if photo.get("type") == "hdr" and photo.get("base"):
        assert ctx.jpeg_root is not None
        return ctx.jpeg_root / photo["base"], False
    return ctx.source_path(rel), photo.get("type") == "raw"


def _needs_render(photo: dict[str, Any]) -> bool:
    """RAW isn't shippable as-is, and an edit has to be baked in. Everything
    else is byte-copied — faster and lossless."""
    edit = photo.get("edit")
    return photo.get("type") == "raw" or bool(edit and not editing.is_neutral(edit))


def _unique_name(parent: Path, base: str, seen: set[str]) -> str:
    """A name free in `parent` and not already claimed in this run."""
    stem, suffix = Path(base).stem, Path(base).suffix
    name, n = base, 1
    while name in seen or (parent / name).exists():
        name = f"{stem}_{n}{suffix}"
        n += 1
    seen.add(name)
    return name


def _bake_photo(ctx: "AppContext", photo: dict[str, Any], rel: str, dst: Path,
                settings: ExportSettings) -> None:
    """Write one photo to `dst`: the original copied when it already is the
    export, otherwise rendered, resized and written with its metadata."""
    if _copies_as_is(ctx, photo, rel, settings):
        shutil.copy2(ctx.source_path(rel), dst)
        return
    _bake_rendered(ctx, photo, rel, dst, settings)


def _bake_rendered(ctx: "AppContext", photo: dict[str, Any], rel: str,
                   dst: Path | BinaryIO, settings: ExportSettings) -> None:
    """Render one photo and write it to `dst`, a path or an open buffer."""
    edit = photo.get("edit")
    out = editing.render(ctx.decode_full(rel), edit,
                         meta=ctx.photo_meta(rel),
                         auto=ctx.auto_fields(rel, edit),
                         src=ctx.range_src(rel, edit))
    out = exporting.resize(out, settings.long_edge)
    src_path, is_raw = _metadata_source(ctx, photo, rel)
    src_exif, icc = metadata_mod.read_source(src_path, is_raw)
    exif = metadata_mod.build_exif(
        src_exif, size=(out.shape[1], out.shape[0]),
        captured_at=ctx.photo_meta(rel).get("captured_at") or "",
        mode=settings.metadata, srgb=is_raw)
    exporting.write(out, dst, fmt=settings.format, quality=settings.quality,
                    exif=exif, icc=icc)


def _export_picks_to(ctx: "AppContext", picks: list[dict[str, Any]], target: Path,
                     mode: str, settings: ExportSettings) -> dict[str, Any]:
    """Write `picks` under `target` in one of the three layouts. Runs on the
    export thread; progress and cancellation go through ctx.export_state."""
    state = ctx.export_state
    people = ctx.data.get("people", []) or []
    people_by_id = {p["id"]: p for p in people}
    excluded_ids = {p["id"] for p in people if p.get("excluded")}
    # Per-target-dir filename uniquifier (used in flat & by_person modes).
    used_names: dict[Path, set[str]] = {}

    def unique_in(parent: Path, base: str) -> str:
        return _unique_name(parent, base, used_names.setdefault(parent, set()))

    width = exporting.seq_width(len(picks))
    copied: list[str] = []
    skipped: list[str] = []
    per_combo: dict[str, int] = {}
    for seq, photo in enumerate(picks, start=1):
        if state["cancel"]:
            break
        rel = photo["rel_path"]
        state["idx"], state["current"] = seq - 1, rel
        src = ctx.source_path(rel)
        if not src.is_file():
            skipped.append(rel)
            continue
        stem = _templated_stem(ctx, photo, rel, settings, seq, width)
        out_name = _baked_name(ctx, photo, rel, settings, stem)
        if mode == "by_person":
            relevant_pids: list[str] = []
            seen: set[str] = set()
            for face in photo.get("faces") or []:
                pid = face.get("person_id")
                if not pid or pid in excluded_ids or pid in seen:
                    continue
                if pid not in people_by_id:
                    continue
                seen.add(pid)
                relevant_pids.append(pid)
            if not relevant_pids:
                combo_label = "Others"
            else:
                relevant_pids.sort(key=lambda pid: (
                    people_by_id[pid].get("priority", 999), pid
                ))
                combo_label = " & ".join(
                    _sanitize_segment(people_by_id[pid].get("label", pid))
                    for pid in relevant_pids
                )
            combo_dir = target / _sanitize_segment(combo_label)
            combo_dir.mkdir(parents=True, exist_ok=True)
            dst = combo_dir / unique_in(combo_dir, out_name)
            per_combo[combo_label] = per_combo.get(combo_label, 0) + 1
        elif mode == "flat":
            dst = target / unique_in(target, out_name)
        else:  # "folder"
            dst = (target / rel).parent / out_name
            dst.parent.mkdir(parents=True, exist_ok=True)
        _bake_photo(ctx, photo, rel, dst, settings)
        copied.append(rel)
        state["idx"] = seq
    return {
        "target_dir": str(target),
        "copied": len(copied),
        "skipped": len(skipped),
        "missing": skipped,
        "cancelled": bool(state["cancel"]),
        "mode": mode,
        "per_combo": per_combo,
    }


def _render_roi(ctx: "AppContext", payload: EditPreviewPayload) -> np.ndarray:
    """Grade one window of a photo at full resolution — the editor's 1:1 view.

    The grade happens in the *original* frame's coordinates, because that is
    where masks live: the window is mapped back through the geometry to find
    which original pixels can reach it, that patch is graded, and only then is it
    straightened and cropped into place. Grading the cropped frame instead would
    put every mask somewhere else the moment a crop was set.

    The patch is taken with a margin that is dropped on the way out, because blur
    and bloom pull in neighbouring pixels and the tilt needs somewhere to sample
    from; without it every pan step would show a seam at its edge.
    """
    edit = payload.edit
    if payload.skip_crop:                    # the crop tool shows the frame whole
        edit = {**editing.normalize(edit), "crop": None}

    base = ctx.get_full_base(payload.rel_path)
    fh, fw = base.shape[:2]
    _, (ow, oh) = editing.geometry_matrix(fw, fh, edit)

    x0, y0, rw, rh = (float(v) for v in payload.roi)
    rw = min(max(rw, 1e-3), 1.0)
    rh = min(max(rh, 1e-3), 1.0)
    x0 = min(max(x0, 0.0), 1.0 - rw)
    y0 = min(max(y0, 0.0), 1.0 - rh)
    window = (int(round(x0 * ow)), int(round(y0 * oh)),
              max(1, int(round(rw * ow))), max(1, int(round(rh * oh))))

    # As much surrounding pixel data as this edit reaches for, plus a couple of
    # pixels for the straighten to interpolate against.
    pad = int(round(editing.effect_padding(edit, max(fw, fh)))) + 2
    box = editing.geometry_source_box(fw, fh, edit, window, pad)
    bx, by, bw, bh = box
    graded = editing.render(
        base[by:by + bh, bx:bx + bw], edit,
        roi=(bx / fw, by / fh, bw / fw, bh / fh),
        meta=ctx.photo_meta(payload.rel_path),
        with_watermark=False,     # placed below, against the cropped frame
        geometry=False,           # this is a patch of the original, not a frame
        auto=ctx.auto_fields(payload.rel_path, edit),
        src=ctx.range_src(payload.rel_path, edit),
    )
    out = editing.geometry_window(graded, box, fw, fh, edit, window)

    stamp = editing.normalize(edit)["watermark"]
    if stamp is not None:
        out = watermark_mod.render(
            out, stamp, ctx.photo_meta(payload.rel_path),
            (window[0] / ow, window[1] / oh, window[2] / ow, window[3] / oh))

    # Only ever downscale: upscaling would defeat the point of a 1:1 view.
    target = min(payload.out_w or out.shape[1], EDIT_ZOOM_MAX_OUT, out.shape[1])
    if payload.max_edge:                     # draft frame during a drag
        target = min(target, payload.max_edge)
    if target < out.shape[1]:
        scale = target / out.shape[1]
        out = cv2.resize(out, (target, max(1, int(round(out.shape[0] * scale)))),
                         interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


def _photo_wire(photo: dict[str, Any]) -> dict[str, Any]:
    """A photo as the client needs it, plus `geom` when the frame has been
    straightened or cropped: the output size and where a point of the *original*
    frame lands in it.

    Detection boxes are stored against the original frame, because that is what
    the detector saw. Without this the client could only either draw them in the
    wrong place on a cropped thumbnail or not draw them at all — and a car that
    is plainly in the picture but has no box looks like the detector missed it.
    """
    edit = photo.get("edit")
    if not edit or editing.geometry_is_neutral(edit):
        return photo
    w, h = int(photo.get("width") or 0), int(photo.get("height") or 0)
    if not (w and h):
        return photo
    ow, oh = editing.geometry_size(w, h, edit)
    return {**photo, "geom": {"w": ow, "h": oh,
                              "xform": editing.geometry_norm_matrix(w, h, edit)}}


def _apply_edit_to_photo(photo: dict[str, Any], edit: dict[str, Any] | None) -> None:
    """Store a normalized edit on a photo dict, or drop it when neutral so
    unedited photos stay small in the db and keep their plain thumbnail."""
    norm = editing.normalize(edit)
    if editing.is_neutral(norm):
        photo.pop("edit", None)
        photo.pop("edited_at", None)
    else:
        photo["edit"] = norm
        photo["edited_at"] = datetime.now().isoformat()


def _write_slot(photo: dict[str, Any], slot: int,
                entry: dict[str, Any] | None) -> list[Any]:
    """Set one of a photo's edit slots and return the rack, storing it on the
    photo (or dropping the key when nothing is left in it).

    Two details the endpoint should not have to think about. The rack is padded
    and cut to `EDIT_SLOTS`, so a photo stashed under a different cap loads
    instead of raising and keeps whatever still fits. And an all-empty rack is
    removed rather than written as six nulls, which keeps an untouched photo
    exactly as small in `picks.json` as it was before slots existed.
    """
    assert 0 <= slot < EDIT_SLOTS, f"slot out of range: {slot}"
    rack = (list(photo.get("edit_slots") or []) + [None] * EDIT_SLOTS)[:EDIT_SLOTS]
    rack[slot] = entry
    if any(e is not None for e in rack):
        photo["edit_slots"] = rack
    else:
        photo.pop("edit_slots", None)
    return photo.get("edit_slots") or []


def _thumb_source(ctx: "AppContext", rel_path: str, photo: dict[str, Any] | None) -> Path:
    """The file a thumbnail is generated from. A RAW photo uses its cached
    preview JPEG (PIL can't open a RAW); everything else uses its source file.
    The cache is keyed by rel_path — the exact identity scorer.ensure_cache used.

    Normally the scan has already written it. Building it here as well covers a
    project scored by an older build, whose cached previews are the camera's
    rendering rather than this one's, and means a RAW never falls through to a
    file PIL would refuse to open. The lock is because the grid asks for a
    dozen tiles at once and a RAW decode is not something to do twelve times."""
    if photo is not None and photo.get("type") == "raw" and ctx.raw_cache_root is not None:
        cached = raw.raw_cache_path(ctx.raw_cache_root, rel_path)
        src = ctx.source_path(rel_path)
        # The photo volume being absent is a state this app expects — it is what
        # relinking exists for — and a project on a detached drive stays
        # browsable off its cached previews alone. Only rebuild when the RAW is
        # actually there to be read.
        if not src.is_file() or raw.cache_is_fresh(src, ctx.raw_cache_root, rel_path):
            return cached
        with _cache_lock(cached):
            return raw.ensure_cache(src, ctx.raw_cache_root, rel_path)
    return ctx.source_path(rel_path)


# Derived-image caches (thumbs, face crops, peaking overlays) are written by
# whichever request needs them first, and several requests for the same photo
# arrive together — the grid asks for /thumb and /peak at the same moment, and
# both build the thumbnail. Writing straight to the served path meant a reader
# could be streaming a file that a writer had just truncated, which surfaced as
# "Response content shorter than Content-Length" and a broken tile.
#
# Two guards: build into a temp file and rename it into place (readers only ever
# see a complete file), and serialize builders for the same path so the work is
# not done twice.
_CACHE_LOCKS = [threading.Lock() for _ in range(64)]


def _cache_lock(key: Path) -> threading.Lock:
    return _CACHE_LOCKS[hash(str(key)) % len(_CACHE_LOCKS)]


@contextmanager
def _atomic_write(dst: Path) -> "Iterator[Path]":
    """Yield a temp path next to `dst`; rename it over `dst` on clean exit."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / f".{dst.name}.{threading.get_ident()}.tmp{dst.suffix}"
    try:
        yield tmp
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _ensure_thumb(src: Path, thumbs_root: Path, rel_path: str,
                  edit: dict[str, Any] | None = None,
                  meta: dict[str, Any] | None = None,
                  watermark: bool = True,
                  auto_fields: Callable[[], dict[str, np.ndarray] | None] | None = None,
                  range_src: Callable[[], "np.ndarray | None"] | None = None
                  ) -> Path:
    # A non-neutral edit gets its own cache file (…​.<hash>.<ver>.jpg) so changing
    # an edit invalidates automatically; unedited photos keep the plain name and
    # need no version, having nothing baked into them.
    #
    # `watermark=False` is for focus peaking, which reads a thumb: the text of a
    # signature has edges as crisp as anything in the frame and would be
    # reported as in focus. That variant gets its own cache entry.
    #
    # `auto_fields` is a callable rather than a dict because most calls here
    # return a cached file without rendering anything, and segmenting a photo to
    # build a thumbnail that already exists would be pure waste. It is invoked
    # only on the path that actually grades.
    ehash = editing.edit_hash(edit)
    if ehash:
        tag = f"{ehash}.{THUMB_VERSION}" + ("" if watermark else "-nw")
        dst = thumbs_root / f"{rel_path}.{tag}.jpg"
    else:
        dst = thumbs_root / rel_path

    def fresh() -> bool:
        if not (dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime):
            return False
        # Invalidate cached thumbs whose long edge is smaller than the current
        # target (picks up THUMB_LONG_EDGE bumps without manual cache wipe).
        try:
            with Image.open(dst) as old:
                return max(old.width, old.height) >= THUMB_LONG_EDGE - 4
        except Exception:
            return False

    if fresh():
        return dst
    with _cache_lock(dst):
        if fresh():
            return dst    # another request built it while we waited
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            img.thumbnail((THUMB_LONG_EDGE, THUMB_LONG_EDGE), Image.Resampling.LANCZOS)
            img = img.convert("RGB")
            if ehash:  # grade the downscaled thumb (cheap) so the grid shows the edit
                # Watermark included: it is part of how the photo will look, and
                # the grid is where you decide whether it works. It scales with
                # the frame, so a tile shows it at the proportion it will print.
                img = Image.fromarray(
                    editing.render(np.asarray(img), edit, meta=meta,
                                   with_watermark=watermark,
                                   auto=auto_fields() if auto_fields else None,
                                   src=range_src() if range_src else None))
            with _atomic_write(dst) as tmp:
                img.save(tmp, "JPEG", quality=THUMB_QUALITY, optimize=True)
    return dst


# Saving an edit is followed, almost always, by the grid asking for that tile.
# Building it here means the tile is already on disk under the project's thumbs
# directory by the time the request lands, instead of the old thumbnail sitting
# there through a render. One worker, so a bulk edit over 500 photos queues
# instead of putting 500 decodes in flight; `_ensure_thumb`'s own lock means a
# request that overtakes the prebuild waits for it rather than repeating it.
_PREBUILD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="thumb-prebuild")


def _prebuild_thumbs(ctx: "AppContext", rel_paths: "list[str]") -> None:
    def build() -> None:
        for rel_path in rel_paths:
            photo = ctx.photo_index.get(rel_path)
            if photo is None or ctx.thumbs_root is None:
                continue
            _ensure_thumb(_thumb_source(ctx, rel_path, photo), ctx.thumbs_root,
                          rel_path, photo.get("edit"), ctx.photo_meta(rel_path),
                          auto_fields=lambda r=rel_path, p=photo:
                              ctx.auto_fields(r, p.get("edit")),
                          range_src=lambda r=rel_path, p=photo:
                              ctx.range_src(r, p.get("edit")))

    def report(fut: "Future[None]") -> None:
        # An executor keeps a worker's exception inside the Future, where it
        # would be lost. Nothing here can be recovered from, so print the trace
        # and let the request path hit the same failure out loud.
        exc = fut.exception()
        if exc is not None:
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    _PREBUILD.submit(build).add_done_callback(report)


def _ensure_face_crop(
    src: Path,
    faces_root: Path,
    rel_path: str,
    bbox_xywh: list[int],
    face_idx: int | str,
    db_path: Path,
    padding: float = FACE_PADDING,
    square: bool = True,
) -> Path:
    """Cached crop around one bbox. Faces get a square crop (a head is roughly
    square and the padding reads as breathing room); detected objects keep their
    own aspect, since squaring a wide car box would be mostly asphalt."""
    dst = faces_root / f"{rel_path}.f{face_idx}.jpg"
    db_mtime = db_path.stat().st_mtime if db_path.exists() else 0
    src_mtime = src.stat().st_mtime

    def fresh() -> bool:
        return dst.exists() and dst.stat().st_mtime >= max(src_mtime, db_mtime)

    if fresh():
        return dst
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    if square:
        half_w = half_h = max(w, h) * padding
    else:
        half_w, half_h = w / 2 * (1 + padding), h / 2 * (1 + padding)
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img)
        crop_box = (
            max(0, int(cx - half_w)),
            max(0, int(cy - half_h)),
            min(img.width, int(cx + half_w)),
            min(img.height, int(cy + half_h)),
        )
        crop = img.crop(crop_box)
        crop.thumbnail((FACE_LONG_EDGE, FACE_LONG_EDGE), Image.Resampling.LANCZOS)
        crop = crop.convert("RGB")
    with _cache_lock(dst):
        if fresh():
            return dst
        with _atomic_write(dst) as tmp:
            crop.save(tmp, "JPEG", quality=FACE_QUALITY, optimize=True)
    return dst


def _scan_sources(
    jpeg_root: Path,
    raw_root: Path,
    photo_root: Path,
    exclude_dirs: set[Path] | None = None,
) -> set[str]:
    """Identities of all on-disk sources (for change detection): supported
    images under jpeg_root by their jpeg-relative path, plus RAW files under
    raw_root tagged `raw::<photo-root-relative path>`. Matches
    scorer.source_rel_paths so a re-open only re-scores when files change."""
    excl = exclude_dirs or ()
    found: set[str] = set()
    for img in walk_files(jpeg_root, excl):
        if _is_supported(img):
            found.add(str(img.relative_to(jpeg_root)))
    if raw_root.is_dir():
        for f in walk_files(raw_root, excl):
            if raw.is_raw(f):
                found.add("raw::" + str(f.relative_to(photo_root)))
    return found


def _summarize_other_files(
    jpeg_root: Path,
    exclude_dirs: set[Path] | None = None,
    limit: int = 4,
) -> str:
    """Sample non-image extensions found under jpeg_root for a helpful error msg."""
    counts: dict[str, int] = {}
    for p in walk_files(jpeg_root, exclude_dirs or ()):
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTS or p.name.startswith("._"):
            continue
        counts[ext or "(no-ext)"] = counts.get(ext or "(no-ext)", 0) + 1
    if not counts:
        return "folder is empty or contains only hidden files"
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:limit]
    return "found only " + ", ".join(f"{n} {ext}" for ext, n in top)


_BROWSE_PROMPTS = {
    "photos": "Select the folder with your photos",
    "workspace": "Select a folder to keep your projects in",
    "project": "Select a project folder",
    "relink": "Select the new location of the photos",
    "export": "Select a folder to copy the picks into",
}


def _native_pick_folder(initial: str | None = None, purpose: str = "photos") -> dict[str, Any]:
    """Open a native folder dialog. Returns {path, cancelled, error}.
    macOS uses AppleScript (`osascript`); other platforms fall back to Tk."""
    prompt = _BROWSE_PROMPTS[purpose]
    if platform.system() == "Darwin":
        # `Finder activate` brings the choose-folder dialog to the foreground
        # reliably without requiring Accessibility/Automation permissions.
        if initial and Path(initial).is_dir():
            choose = (
                f'choose folder with prompt "{prompt}" '
                f'default location POSIX file "{initial}"'
            )
        else:
            choose = f'choose folder with prompt "{prompt}"'
        script = (
            'tell application "Finder" to activate\n'
            f'set f to POSIX path of ({choose})\n'
            "return f"
        )
        try:
            r = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=600,
            )
        except FileNotFoundError:
            return {"path": None, "cancelled": False, "error": "osascript not found"}
        except subprocess.TimeoutExpired:
            return {"path": None, "cancelled": True, "error": None}
        if r.returncode != 0:
            stderr = (r.stderr or "").strip()
            cancelled = "User canceled" in stderr or "-128" in stderr
            return {"path": None, "cancelled": cancelled,
                    "error": None if cancelled else stderr or f"exit {r.returncode}"}
        out = r.stdout.strip().rstrip("/")
        return {"path": out or None, "cancelled": not out, "error": None}

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        # Parented to the topmost root, or on Windows the dialog can open
        # behind the browser that asked for it.
        path = filedialog.askdirectory(
            parent=root, title=prompt, initialdir=initial or str(Path.home()),
        )
        root.destroy()
        return {"path": path or None, "cancelled": not path, "error": None}
    except Exception as exc:
        return {"path": None, "cancelled": False, "error": f"{type(exc).__name__}: {exc}"}


_PATH_BAD_CHARS = '/\\:*?"<>|\x00'


def _sanitize_segment(name: str) -> str:
    """Make `name` safe to use as a single path segment. Empty → 'unnamed'."""
    cleaned = "".join("_" if c in _PATH_BAD_CHARS else c for c in name).strip(" .")
    return cleaned or "unnamed"


def _autodetect_jpeg_subdir(photo_dir: Path) -> str:
    """If `photo_dir` has no top-level supported images but a common subfolder
    does, return that subfolder's name. Otherwise return ''."""
    try:
        for entry in photo_dir.iterdir():
            if entry.is_file() and _is_supported(entry):
                return ""  # already has images at the root
    except OSError:
        return ""
    for candidate in ("JPEG", "jpeg", "JPG", "jpg", "JPEGS", "Photos", "photos"):
        if (photo_dir / candidate).is_dir():
            return candidate
    return ""


def _resolve_db_path(payload: OpenPayload) -> Path:
    if payload.db_path:
        return Path(payload.db_path).expanduser().resolve()
    return (Path(payload.photo_dir).expanduser() / "picks.json").resolve()


# ----- app construction ---------------------------------------------------

def create_app(initial_db_path: Path | None = None) -> FastAPI:
    ctx = AppContext()
    if initial_db_path is not None:
        ctx.load_db(initial_db_path)

    app = FastAPI(title="Picture Classifier")

    @app.middleware("http")
    async def _no_cache_for_web_assets(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((WEB_DIR / "index.html").read_text(encoding="utf-8"))

    # ----- project state & open --------------------------------------

    @app.get("/api/state")
    def get_state() -> dict[str, Any]:
        return {
            "ready": ctx.is_loaded() and ctx.opening_state.get("ready", False),
            "db_path": str(ctx.db_path) if ctx.db_path else None,
            "photo_root": str(ctx.photo_root) if ctx.photo_root else None,
            "opening": ctx.opening_state,
        }

    def _view_key() -> str:
        assert ctx.db_path is not None, "a project must be loaded to key its view"
        return userstate.view_key(ctx.db_path, ctx.project_dir)

    @app.get("/api/view")
    def get_view() -> dict[str, Any]:
        """The filter, layout and page this project was last left on. Empty the
        first time it is opened, which the client reads as "use the defaults"."""
        _require_loaded()
        return userstate.get_view(_view_key())

    @app.post("/api/view")
    def set_view(payload: ViewPayload) -> dict[str, Any]:
        _require_loaded()
        if payload.page < 0:
            raise HTTPException(status_code=400, detail="page must not be negative")
        view = {"filter": payload.filter, "page_size": payload.page_size,
                "page": payload.page, "scene": payload.scene}
        userstate.set_view(_view_key(), view)
        return view

    @app.post("/api/quit")
    def quit_app(payload: QuitPayload | None = None) -> dict[str, Any]:
        """Stop the server, which is the whole app: the page is only a view of
        it. Without this, closing the tab left it running in the background,
        and on Windows there was no console to close either."""
        busy = [what for what, st in (
            ("an export", ctx.export_state), ("scoring", ctx.scoring_state),
            ("grouping", ctx.cluster_state), ("opening a project", ctx.opening_state),
        ) if st["running"]]
        if busy and not (payload and payload.force):
            raise HTTPException(status_code=409, detail=" and ".join(busy) + " is still running")
        server = getattr(app.state, "server", None)
        assert server is not None, "quit needs the uvicorn server that serve() started"
        # A beat later, so this response reaches the page before the socket goes.
        threading.Timer(0.3, lambda: setattr(server, "should_exit", True)).start()
        return {"quitting": True}

    @app.post("/api/close")
    def close_project() -> dict[str, Any]:
        if (ctx.scoring_state["running"] or ctx.cluster_state["running"]
                or ctx.opening_state["running"] or ctx.export_state["running"]):
            raise HTTPException(status_code=409, detail="another task is running")
        ctx.close()
        return {"ready": False}

    @app.post("/api/browse-folder")
    def browse_folder(payload: BrowsePayload | None = None) -> dict[str, Any]:
        payload = payload or BrowsePayload()
        return _native_pick_folder(payload.initial, payload.purpose)

    @app.post("/api/folder/inspect")
    def inspect_folder(payload: InspectPayload) -> dict[str, Any]:
        """Describe a folder before it is used as photos or as a workspace."""
        if not payload.path.strip():
            raise HTTPException(status_code=400, detail="path is required")
        workspace = Path(payload.workspace) if payload.workspace else None
        return folderinfo.inspect(Path(payload.path.strip()), workspace=workspace)

    @app.get("/api/recents")
    def get_recents() -> dict[str, Any]:
        return {"recents": userstate.get_recents()}

    @app.post("/api/recents/forget")
    def forget_recent(payload: ForgetPayload) -> dict[str, Any]:
        key = payload.project_dir or payload.db_path
        if key:
            userstate.forget(Path(key))
        return {"recents": userstate.get_recents()}

    def _do_open(
        *,
        photo_dir: Path,
        jpeg_subdir: str,
        db_path: Path,
        project_dir: Path | None,
        allow_autodetect_subdir: bool,
        initial_scene_grouping: dict[str, Any] | None = None,
        raw_subdir_override: str = "",
        subject_classes: list[str] | None = None,
    ) -> None:
        """Shared open worker. If `project_dir` is set we load via project layout
        (caches under project_dir/.cache/), otherwise legacy load_db is used."""
        try:
            existing_data: dict[str, Any] | None = None
            if db_path.exists():
                try:
                    existing_data = db.load(db_path)
                except Exception:
                    existing_data = None

            # Effective jpeg_subdir resolution priority:
            #   caller value > existing db > autodetect (legacy only)
            if not jpeg_subdir and existing_data:
                jpeg_subdir = existing_data.get("jpeg_subdir", "") or ""
            if not jpeg_subdir and allow_autodetect_subdir:
                jpeg_subdir = _autodetect_jpeg_subdir(photo_dir)
            jpeg_root = photo_dir / jpeg_subdir if jpeg_subdir else photo_dir
            raw_subdir = raw_subdir_override or (
                existing_data.get("raw_subdir", "") if existing_data else "")
            raw_root = photo_dir / raw_subdir if raw_subdir else jpeg_root

            ctx.opening_state["phase"] = "scanning"
            ctx.opening_state["message"] = (
                f"Scanning {jpeg_root}"
                + (f" (subfolder: {jpeg_subdir})" if jpeg_subdir else "")
                + "…"
            )
            if not jpeg_root.is_dir():
                # An existing project whose photos moved → let the UI offer relink.
                if existing_data is not None:
                    ctx.opening_state["needs_relink"] = {
                        "project_dir": str(project_dir) if project_dir else None,
                        "db_path": str(db_path),
                        "photo_dir": str(photo_dir),
                        "jpeg_subdir": jpeg_subdir,
                    }
                raise RuntimeError(f"photo folder not found: {jpeg_root}")
            scan_excludes = excluded_scan_dirs(db_path, project_dir)
            current_files = _scan_sources(jpeg_root, raw_root, photo_dir, scan_excludes)
            if not current_files:
                hint = _summarize_other_files(jpeg_root, scan_excludes)
                raise RuntimeError(
                    f"no .jpg/.jpeg/.png or RAW images found under {jpeg_root} ({hint}); "
                    f"pick a different folder or set the JPEG subfolder"
                )

            # Compare the scan against the prior *source* files (standalone
            # photos + bracket members), since merged HDR results carry
            # synthetic rel_paths that never appear in a folder scan.
            existing_files = source_rel_paths(existing_data) if existing_data else set()
            needs_score = (
                existing_data is None
                or existing_data.get("scored_at") is None
                or current_files != existing_files
            )

            if needs_score:
                ctx.opening_state["phase"] = "scoring"
                ctx.opening_state["message"] = "Scoring photos with face detection…"
                from .scorer import run_scoring

                def score_cb(i: int, total: int, current: str | None) -> None:
                    ctx.opening_state["idx"] = i
                    ctx.opening_state["total"] = total
                    ctx.opening_state["current"] = current

                run_scoring(
                    photo_dir, jpeg_subdir, db_path,
                    with_faces=False, progress_cb=score_cb,
                    project_dir=project_dir, raw_subdir=raw_subdir,
                    subject_classes=subject_classes,
                )

                ctx.opening_state["phase"] = "clustering"
                ctx.opening_state["message"] = "Clustering faces…"
                ctx.opening_state["idx"] = 0
                ctx.opening_state["total"] = 0
                from .cluster import run_clustering

                def cluster_cb(phase: str, idx: int, total: int) -> None:
                    ctx.opening_state["message"] = f"Clustering: {phase}"
                    ctx.opening_state["idx"] = idx
                    ctx.opening_state["total"] = total

                run_clustering(db_path, progress_cb=cluster_cb)

                # Apply user-chosen initial scene grouping (wizard).
                if initial_scene_grouping and initial_scene_grouping.get("mode") == "time_gap":
                    fresh = db.load(db_path)
                    scenes.regroup(
                        fresh["photos"],
                        jpeg_root,
                        "time_gap",
                        initial_scene_grouping.get("gap_minutes", 30),
                    )
                    by_scene: dict[str, list[dict[str, Any]]] = {}
                    for p in fresh["photos"]:
                        by_scene.setdefault(p["scene"], []).append(p)
                    for items in by_scene.values():
                        apply_scene_suggestions(items)
                    fresh["scene_grouping"] = {
                        "mode": "time_gap",
                        "gap_minutes": initial_scene_grouping.get("gap_minutes", 30),
                    }
                    db.save(db_path, fresh)

            ctx.opening_state["phase"] = "loading"
            ctx.opening_state["message"] = "Loading project…"
            if project_dir is not None:
                ctx.load_project(project_dir)
                userstate.remember_open(
                    db_path, photo_dir, jpeg_subdir,
                    kind="project", project_dir=project_dir,
                )
            else:
                ctx.load_db(db_path)
                userstate.remember_open(db_path, photo_dir, jpeg_subdir)

            ctx.opening_state["phase"] = "done"
            ctx.opening_state["message"] = None
        except Exception as exc:
            ctx.opening_state["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            ctx.opening_state["running"] = False
            ctx.opening_state["ended_at"] = datetime.now().isoformat()

    def _claim_open_lock() -> None:
        with ctx.score_lock:
            if (ctx.opening_state["running"] or ctx.scoring_state["running"]
                    or ctx.cluster_state["running"] or ctx.export_state["running"]):
                raise HTTPException(status_code=409, detail="another task is running")
            ctx.opening_state.update(
                ctx._fresh_opening_state(),
                running=True,
                started_at=datetime.now().isoformat(),
            )

    @app.post("/api/open")
    def open_legacy(payload: OpenPayload) -> dict[str, Any]:
        """Legacy: picks.json lives next to (or inside) the photo folder."""
        _claim_open_lock()
        photo_dir = Path(payload.photo_dir).expanduser().resolve()
        db_path = _resolve_db_path(payload)
        threading.Thread(
            target=_do_open,
            kwargs=dict(
                photo_dir=photo_dir,
                jpeg_subdir=payload.jpeg_subdir or "",
                db_path=db_path,
                project_dir=None,
                allow_autodetect_subdir=True,
            ),
            daemon=True,
        ).start()
        return {"started": True, "db_path": str(db_path)}

    @app.post("/api/project/open")
    def open_project(payload: OpenProjectPayload) -> dict[str, Any]:
        """Open an existing project directory (created via the wizard)."""
        _claim_open_lock()
        project_dir = Path(payload.project_dir).expanduser().resolve()
        db_path = project_dir / "picks.json"
        if not db_path.is_file():
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = (
                f"not a project directory (no picks.json found): {project_dir}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"not a project directory (no picks.json found): {project_dir}",
            )
        # photo_dir/jpeg_subdir come from the picks.json itself.
        try:
            data = db.load(db_path)
            photo_dir = Path(data["photo_root"]).expanduser().resolve()
            jpeg_subdir = data.get("jpeg_subdir", "") or ""
        except Exception as exc:
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = f"{type(exc).__name__}: {exc}"
            raise HTTPException(status_code=400, detail=str(exc))
        threading.Thread(
            target=_do_open,
            kwargs=dict(
                photo_dir=photo_dir,
                jpeg_subdir=jpeg_subdir,
                db_path=db_path,
                project_dir=project_dir,
                allow_autodetect_subdir=False,
            ),
            daemon=True,
        ).start()
        return {"started": True, "project_dir": str(project_dir)}

    @app.post("/api/project/create")
    def create_project(payload: CreateProjectPayload) -> dict[str, Any]:
        """Create a project *by name* inside the chosen workspace and trigger the
        initial score+cluster. The project folder is <workspace>/<name>/."""
        _claim_open_lock()
        name = _sanitize_segment(payload.name)
        workspace_dir = Path(payload.workspace_dir).expanduser().resolve()
        project_dir = (workspace_dir / name).resolve()
        photo_dir = Path(payload.photo_dir).expanduser().resolve()
        if not payload.name.strip():
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = "project name is required"
            raise HTTPException(status_code=400, detail=ctx.opening_state["error"])
        if not photo_dir.is_dir():
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = f"photo folder does not exist: {photo_dir}"
            raise HTTPException(status_code=400, detail=ctx.opening_state["error"])
        if (project_dir / "picks.json").exists():
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = f"a project named '{name}' already exists in this workspace"
            raise HTTPException(status_code=400, detail=ctx.opening_state["error"])
        jpeg_root = (
            (photo_dir / payload.jpeg_subdir).resolve()
            if payload.jpeg_subdir else photo_dir
        )
        if (
            project_dir == jpeg_root
            or project_dir.is_relative_to(jpeg_root)
            or jpeg_root.is_relative_to(project_dir)
        ):
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = (
                "project folder must not live inside the photo folder (or vice "
                f"versa); cache files would be re-scored as photos. "
                f"project={project_dir}, photos={jpeg_root}"
            )
            raise HTTPException(status_code=400, detail=ctx.opening_state["error"])
        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / ".cache").mkdir(exist_ok=True)
        except OSError as exc:
            ctx.opening_state["running"] = False
            ctx.opening_state["error"] = f"could not create project directory: {exc}"
            raise HTTPException(status_code=400, detail=ctx.opening_state["error"])
        userstate.add_workspace(workspace_dir)  # remember + make current
        threading.Thread(
            target=_do_open,
            kwargs=dict(
                photo_dir=photo_dir,
                jpeg_subdir=payload.jpeg_subdir or "",
                db_path=project_dir / "picks.json",
                project_dir=project_dir,
                allow_autodetect_subdir=False,
                raw_subdir_override=payload.raw_subdir or "",
                initial_scene_grouping={
                    "mode": payload.scene_grouping_mode,
                    "gap_minutes": payload.scene_grouping_gap_minutes,
                },
                subject_classes=objects_mod.resolve_classes(payload.subject_classes),
            ),
            daemon=True,
        ).start()
        return {"started": True, "project_dir": str(project_dir)}

    # ----- workspaces ------------------------------------------------

    @app.get("/api/workspaces")
    def get_workspaces() -> dict[str, Any]:
        ws = userstate.get_workspaces()
        cur = userstate.get_current_workspace()
        # Nothing chosen yet: the landing walks through choosing one rather
        # than quietly using the default.
        first_run = not ws
        if first_run:  # seed a sensible default (not persisted until used)
            cur = str(userstate.DEFAULT_WORKSPACE)
            ws = [cur]
        return {"workspaces": ws, "current": cur, "first_run": first_run,
                "default": str(userstate.DEFAULT_WORKSPACE), "sep": os.sep}

    @app.post("/api/workspaces")
    def add_workspace(payload: WorkspacePayload) -> dict[str, Any]:
        wdir = Path(payload.dir).expanduser().resolve()
        try:
            wdir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"could not create workspace: {exc}")
        userstate.add_workspace(wdir)
        return {"workspaces": userstate.get_workspaces(), "current": str(wdir)}

    @app.post("/api/workspaces/current")
    def set_workspace(payload: WorkspacePayload) -> dict[str, Any]:
        wdir = Path(payload.dir).expanduser().resolve()
        userstate.set_current_workspace(wdir)
        return {"workspaces": userstate.get_workspaces(), "current": str(wdir)}

    @app.post("/api/workspaces/forget")
    def forget_workspace(payload: WorkspacePayload) -> dict[str, Any]:
        userstate.forget_workspace(Path(payload.dir).expanduser().resolve())
        return {"workspaces": userstate.get_workspaces(),
                "current": userstate.get_current_workspace()}

    @app.get("/api/workspaces/projects")
    def list_workspace_projects(workspace: str | None = None) -> dict[str, Any]:
        """List the projects (subfolders with a picks.json) in a workspace."""
        target = workspace or userstate.get_current_workspace() or str(userstate.DEFAULT_WORKSPACE)
        wdir = Path(target).expanduser()
        projects: list[dict[str, Any]] = []
        if wdir.is_dir():
            for child in sorted(wdir.iterdir(), key=lambda c: c.name.lower()):
                if userstate.is_deleted_project(child.name):
                    continue  # renamed away by a delete; still on disk, just hidden
                pj = child / "picks.json"
                if not pj.is_file():
                    continue
                try:
                    data = db.load(pj)
                except Exception:
                    continue
                photo_root = data.get("photo_root", "")
                photos = data.get("photos", [])
                decided = sum(1 for p in photos if p.get("decision"))
                projects.append({
                    "name": child.name,
                    "project_dir": str(child),
                    "photo_dir": photo_root,
                    "jpeg_subdir": data.get("jpeg_subdir", ""),
                    "photos": len(photos),
                    "decided": decided,
                    "scored_at": data.get("scored_at"),
                    "photos_exist": bool(photo_root and Path(photo_root).is_dir()),
                })
        return {"workspace": str(wdir), "projects": projects}

    @app.post("/api/projects/delete")
    def delete_project(payload: DeleteProjectPayload) -> dict[str, Any]:
        """Remove a project from the list without touching a single photo.

        The project folder is renamed to `<name>.deleted-<stamp>` and dropped
        from recents. Everything — picks.json, caches, merged HDR results — stays
        on disk under the new name, so renaming it back restores the project.
        """
        pdir = Path(payload.project_dir).expanduser().resolve()
        if not (pdir / "picks.json").is_file():
            raise HTTPException(
                status_code=400,
                detail=f"not a project directory (no picks.json): {pdir}",
            )
        if ctx.project_dir is not None and ctx.project_dir == pdir:
            raise HTTPException(
                status_code=409,
                detail="this project is open — switch project first, then delete it",
            )
        target = userstate.deleted_project_path(pdir, datetime.now().strftime("%Y%m%d-%H%M"))
        try:
            pdir.rename(target)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"could not rename: {exc}") from exc
        userstate.forget(pdir)
        userstate.forget(pdir / "picks.json")
        return {"deleted": True, "renamed_to": str(target), "name": target.name}

    @app.post("/api/project/relink")
    def relink_project(payload: RelinkPayload) -> dict[str, Any]:
        """Re-resolve a project's photos to a new folder (move/rename recovery)."""
        if payload.project_dir:
            db_path = (Path(payload.project_dir).expanduser() / "picks.json").resolve()
        elif payload.db_path:
            db_path = Path(payload.db_path).expanduser().resolve()
        else:
            raise HTTPException(status_code=400, detail="project_dir or db_path required")
        if not db_path.is_file():
            raise HTTPException(status_code=404, detail=f"project not found: {db_path}")
        new_photo = Path(payload.new_photo_dir).expanduser().resolve()
        if not new_photo.is_dir():
            raise HTTPException(status_code=400, detail=f"folder not found: {new_photo}")
        report = relink.relink_project(db_path, new_photo)
        return report

    # ----- viewer data -----------------------------------------------

    def _require_loaded() -> None:
        if not ctx.is_loaded():
            raise HTTPException(status_code=409, detail="no project loaded")

    @app.get("/api/db")
    def get_db() -> dict[str, Any]:
        _require_loaded()
        return {
            "scored_at": ctx.data["scored_at"],
            "clustered_at": ctx.data.get("clustered_at"),
            "photo_root": ctx.data["photo_root"],
            "jpeg_subdir": ctx.data["jpeg_subdir"],
            "scene_grouping": ctx.data.get("scene_grouping", {"mode": "folder", "gap_minutes": 30}),
            "people": ctx.data.get("people", []),
            "brackets": ctx.data.get("brackets", []),
            "hdr_look": ctx.data.get("hdr_look") or hdr.DEFAULT_LOOK,
            "photos": [_photo_wire(p) for p in ctx.data["photos"]],
        }

    @app.post("/api/decide")
    def decide(payload: DecidePayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"]:
            raise HTTPException(status_code=409, detail="scoring in progress; decisions disabled")
        photo = ctx.photo_index.get(payload.rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        photo["decision"] = payload.decision
        photo["decided_at"] = datetime.now().isoformat() if payload.decision else None
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return photo

    @app.post("/api/decide/bulk")
    def decide_bulk(payload: BulkDecidePayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"]:
            raise HTTPException(status_code=409, detail="scoring in progress; decisions disabled")
        now = datetime.now().isoformat() if payload.decision else None
        for rp in payload.rel_paths:
            photo = ctx.photo_index.get(rp)
            if photo is None:
                raise HTTPException(status_code=404, detail=f"photo not found: {rp}")
            photo["decision"] = payload.decision
            photo["decided_at"] = now
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return {"updated": len(payload.rel_paths)}

    # ----- scoring ---------------------------------------------------

    @app.post("/api/score")
    def start_score(payload: ScorePayload | None = None) -> dict[str, Any]:
        _require_loaded()
        with ctx.score_lock:
            if ctx.scoring_state["running"]:
                raise HTTPException(status_code=409, detail="scoring already running")
            ctx.scoring_state.update(
                running=True, phase="scoring", idx=0, total=0, current=None,
                started_at=datetime.now().isoformat(), ended_at=None, error=None,
            )

        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)

        with_faces = payload.with_faces if payload else False
        bracket_groups = payload.bracket_groups if payload else None
        hdr_look = payload.hdr_look if payload else None
        subject_classes = payload.subject_classes if payload else None
        if subject_classes is not None:
            subject_classes = objects_mod.resolve_classes(subject_classes)

        def progress_cb(i: int, total: int, current: str | None) -> None:
            ctx.scoring_state["idx"] = i
            ctx.scoring_state["total"] = total
            ctx.scoring_state["current"] = current

        def runner() -> None:
            try:
                from .scorer import run_scoring
                run_scoring(
                    ctx.photo_root,
                    ctx.data["jpeg_subdir"],
                    ctx.db_path,
                    with_faces=with_faces,
                    progress_cb=progress_cb,
                    project_dir=ctx.project_dir,
                    bracket_groups=bracket_groups,
                    hdr_look=hdr_look,
                    raw_subdir=ctx.data.get("raw_subdir", ""),
                    subject_classes=subject_classes,
                    detect_faces=payload.detect_faces if payload else None,
                )
                ctx.reload_data()
                ctx.wipe_face_cache()
                # Scoring resets people[] and vehicles[] and puts every
                # person_id back to None, so a run that stopped here would hand
                # back a project whose groups had silently vanished. Grouping is
                # the cheap half; do it now rather than making it a second button
                # someone has to know to press.
                ctx.scoring_state["phase"] = "grouping"
                ctx.scoring_state.update(idx=0, total=0, current=None)

                def cluster_cb(phase: str, idx: int, total: int) -> None:
                    ctx.scoring_state["current"] = phase
                    ctx.scoring_state["idx"] = idx
                    ctx.scoring_state["total"] = total

                _run_clustering(_cluster_settings(), cluster_cb)
                ctx.reload_data()
                ctx.wipe_face_cache()
            except Exception as exc:
                ctx.scoring_state["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                ctx.scoring_state["running"] = False
                ctx.scoring_state["phase"] = None
                ctx.scoring_state["ended_at"] = datetime.now().isoformat()

        threading.Thread(target=runner, daemon=True).start()
        return {"started": True}

    @app.get("/api/score/status")
    def score_status() -> dict[str, Any]:
        return ctx.scoring_state

    @app.post("/api/hdr/preview")
    def hdr_preview(payload: HdrPreviewPayload) -> Response:
        """Fuse a bracket and apply the given look — drives the live look-tuner.
        The fused array is cached (single entry) so dragging the sliders only
        re-grades instead of re-fusing."""
        _require_loaded()
        if len(payload.members) < 2:
            raise HTTPException(status_code=400, detail="need at least 2 frames")
        member_paths = [ctx.jpeg_root / m for m in payload.members]
        for p in member_paths:
            if not p.is_file():
                raise HTTPException(status_code=404, detail=f"missing frame: {p.name}")
        key = tuple(sorted(payload.members))
        if ctx.hdr_fuse_cache is None or ctx.hdr_fuse_cache[0] != key:
            ctx.hdr_fuse_cache = (key, hdr.fuse(member_paths, max_edge=1400))
        graded = hdr.grade(ctx.hdr_fuse_cache[1], payload.look)
        buf = io.BytesIO()
        Image.fromarray(graded).save(buf, "JPEG", quality=86)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    # ----- editing ---------------------------------------------------

    @app.post("/api/edit/preview")
    def edit_preview(payload: EditPreviewPayload) -> Response:
        """Render the given edit onto a cached copy of the photo and return a
        JPEG — drives the live editor. The decoded base is LRU-cached so slider
        drags only re-grade (mirrors /api/hdr/preview).

        With no `roi` this is the fit view: the whole frame at EDIT_PREVIEW_EDGE.
        With one, it is the 1:1 view: the requested window cut from the
        full-resolution original, so zooming in shows real detail instead of an
        upscaled preview.
        """
        _require_loaded()
        if ctx.photo_index.get(payload.rel_path) is None:
            raise HTTPException(status_code=404, detail="photo not found")
        started = time.perf_counter()
        fit_edit = payload.edit
        if payload.skip_crop:   # the crop tool drags its box over the whole frame
            fit_edit = {**editing.normalize(payload.edit), "crop": None}
        if payload.roi is None:
            # Graded whole, then cropped — the same order `render` uses for an
            # export, so the fit view is what the file will be.
            base = ctx.get_decoded_base(payload.rel_path)
            if payload.max_edge and payload.max_edge < max(base.shape[:2]):
                base = ctx.get_draft_base(payload.rel_path, payload.max_edge)
            out = editing.render(base, fit_edit,
                                 meta=ctx.photo_meta(payload.rel_path),
                                 auto=ctx.auto_fields(payload.rel_path, fit_edit),
                                 src=ctx.range_src(payload.rel_path, fit_edit))
        else:
            out = _render_roi(ctx, payload)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, "JPEG", quality=88)
        # What the frame is now, at full resolution. Straightening and cropping
        # change it, and the client lays the overlay and the 1:1 view out against
        # it — reported rather than recomputed in JS so there is one copy of the
        # arithmetic.
        photo = ctx.photo_index[payload.rel_path]
        src_w = int(photo.get("width") or out.shape[1])
        src_h = int(photo.get("height") or out.shape[0])
        uncropped = {**editing.normalize(payload.edit), "crop": None}
        shown = uncropped if payload.skip_crop else payload.edit
        fw, fh = editing.geometry_size(src_w, src_h, shown)
        # How to get from a point on the original frame to the same point on what
        # is displayed. Masks are positioned against the original, so the editor
        # needs this to draw their guides on the right piece of the photo.
        xform = editing.geometry_norm_matrix(src_w, src_h, shown)
        # Two sizes, because they answer different questions: X-Frame-Size is what
        # is on screen right now (the overlay and the 1:1 view measure against it),
        # X-Frame-Full is the straightened frame before cropping, which is what a
        # crop box's own coordinates are fractions of.
        cw, ch = editing.geometry_size(src_w, src_h, uncropped)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store",
                                 "X-Frame-Size": f"{fw}x{fh}",
                                 "X-Frame-Full": f"{cw}x{ch}",
                                 "X-Mask-Xform": ",".join(f"{v:.8f}" for v in xform),
                                 "X-Render-Ms": f"{(time.perf_counter() - started) * 1000:.0f}"})

    @app.post("/api/edit")
    def save_edit(payload: EditSavePayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"]:
            raise HTTPException(status_code=409, detail="scoring in progress; edits disabled")
        photo = ctx.photo_index.get(payload.rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        _apply_edit_to_photo(photo, payload.edit)
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        _prebuild_thumbs(ctx, [payload.rel_path])
        return _photo_wire(photo)

    @app.post("/api/edit/bulk")
    def bulk_edit(payload: EditBulkPayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"]:
            raise HTTPException(status_code=409, detail="scoring in progress; edits disabled")
        for rp in payload.rel_paths:
            photo = ctx.photo_index.get(rp)
            if photo is None:
                raise HTTPException(status_code=404, detail=f"photo not found: {rp}")
        for rp in payload.rel_paths:
            photo = ctx.photo_index[rp]
            edit = payload.edit
            if payload.mode == "add":
                edit = editing.merge_additive(photo.get("edit"), edit)
            _apply_edit_to_photo(photo, edit)
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        _prebuild_thumbs(ctx, list(payload.rel_paths))
        return {"updated": len(payload.rel_paths)}

    @app.post("/api/edit/auto")
    def auto_edit(payload: AutoTonePayload) -> dict[str, Any]:
        """Compute an auto-tone edit from the histogram; returns it (no save) so
        the client can preview and let the user tweak before applying."""
        _require_loaded()
        if ctx.photo_index.get(payload.rel_path) is None:
            raise HTTPException(status_code=404, detail="photo not found")
        base = ctx.get_decoded_base(payload.rel_path)
        return {"edit": editing.auto_tone(base)}

    @app.post("/api/edit/slot")
    def save_edit_slot(payload: EditSlotPayload) -> dict[str, Any]:
        """Stash the working edit in one of the photo's slots, or clear one.

        A slot is a scratchpad, not a second edit: nothing here changes what the
        photo renders as, and `edited_at` does not move. That is exactly what
        makes slots safe to fill while trying things — the cost of a wrong turn
        becomes one click instead of the whole grade. They are written to
        `picks.json` immediately, so backing out of the editor with Cancel
        keeps them.
        """
        _require_loaded()
        if ctx.scoring_state["running"]:
            raise HTTPException(status_code=409, detail="scoring in progress; edits disabled")
        photo = ctx.photo_index.get(payload.rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        if not 0 <= payload.slot < EDIT_SLOTS:
            raise HTTPException(
                status_code=400,
                detail=f"slot must be 0..{EDIT_SLOTS - 1}, got {payload.slot}")
        entry = None if payload.edit is None else {
            "edit": editing.normalize(payload.edit),
            "saved_at": datetime.now().isoformat(),
            "name": (payload.name or "")[:40],
        }
        slots = _write_slot(photo, payload.slot, entry)
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return {"rel_path": payload.rel_path, "slots": slots}

    @app.post("/api/edit/neutral")
    def neutral_pick(payload: NeutralPickPayload) -> dict[str, Any]:
        """The temp/tint that turn the clicked patch grey.

        Sampled from the ungraded base rather than from the preview the user
        clicked, which is the only way the answer can be absolute: white balance
        is the first stage of the grade, so solving on pixels that already carry
        a grade would fold the current sliders into their own replacement.
        """
        _require_loaded()
        if ctx.photo_index.get(payload.rel_path) is None:
            raise HTTPException(status_code=404, detail="photo not found")
        base = ctx.get_decoded_base(payload.rel_path)
        wb = editing.neutral_wb(base, payload.x, payload.y)
        if wb is None:
            raise HTTPException(
                status_code=400,
                detail="that spot is too dark or too bright to read a white "
                       "balance from — pick something mid-grey",
            )
        return {"wb": wb}

    # ----- presets (app-global) --------------------------------------

    @app.get("/api/watermark/info")
    def watermark_info(rel_path: str | None = None) -> dict[str, Any]:
        """Everything the watermark panel needs: the styles and tokens it can
        use, the camera name catalogue, and (optionally) one photo's shooting
        info so the editor can preview real values instead of placeholders."""
        _require_loaded()
        return {
            "styles": list(watermark_mod.STYLES),
            "positions": list(watermark_mod.POSITIONS),
            "tokens": list(watermark_mod.TOKENS),
            "defaults": watermark_mod.DEFAULT_WATERMARK,
            "cameras": cameras.preset_list(),
            "meta": ctx.photo_meta(rel_path) if rel_path else None,
        }

    def _all_presets() -> list[dict[str, Any]]:
        """Built-ins first, then the user's own — one list for the picker."""
        return presets_mod.list_builtins() + userstate.list_presets()

    @app.get("/api/film/stocks")
    def film_stocks() -> dict[str, Any]:
        """The named parameter sets, and the neutral one to reset to."""
        return {"stocks": [{"name": n, **v} for n, v in film_mod.STOCKS.items()],
                "defaults": film_mod.DEFAULT_FILM}

    @app.get("/api/presets")
    def get_presets() -> dict[str, Any]:
        return {"presets": _all_presets()}

    @app.post("/api/presets")
    def save_preset(payload: PresetSavePayload) -> dict[str, Any]:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="preset name is required")
        preset = userstate.save_preset(name, editing.normalize(payload.edit))
        return {"preset": preset, "presets": _all_presets()}

    @app.delete("/api/presets/{preset_id}")
    def delete_preset(preset_id: str) -> dict[str, Any]:
        if presets_mod.is_builtin(preset_id):
            raise HTTPException(
                status_code=400,
                detail="built-in presets can't be deleted — save your own version instead",
            )
        userstate.delete_preset(preset_id)
        return {"presets": _all_presets()}

    # ----- clustering ------------------------------------------------

    def _cluster_settings(patch: ClusterPayload | None = None) -> dict[str, Any]:
        """The project's clustering settings, with anything sent laid over them."""
        # Imported here, not at the top: cluster pulls in sklearn, which is half a
        # second nobody launching the app should wait for.
        from . import cluster as cluster_mod
        stored = cluster_mod.normalize_settings(ctx.data.get("cluster_settings"))
        if patch is None:
            return stored
        sent = {k: v for k, v in patch.model_dump().items() if v is not None}
        return cluster_mod.normalize_settings({**stored, **sent})

    @app.get("/api/cluster/settings")
    def get_cluster_settings() -> dict[str, Any]:
        """What the settings panel shows. `subjects_available` says whether the
        subject pass can run at all: with no target classes there is nothing
        detected to group, and the controls say so rather than doing nothing."""
        _require_loaded()
        from . import cluster as cluster_mod
        return {
            "settings": _cluster_settings(),
            "defaults": cluster_mod.DEFAULT_SETTINGS,
            "subjects_available": bool(ctx.data.get("subject_classes")),
            "subject_classes": ctx.data.get("subject_classes") or [],
        }

    def _run_clustering(settings: dict[str, Any],
                        progress_cb: Callable[[str, int, int], None]) -> None:
        """Both passes, in the order the results depend on. Shared by the cluster
        button and by the tail of a rescore, so the two cannot drift."""
        from functools import partial
        from .cluster import (clear_vehicle_groups, run_clustering,
                              run_vehicle_clustering)
        from .scorer import pixel_path
        run_clustering(ctx.db_path, eps=settings["face_eps"],
                       min_samples=settings["face_min_samples"],
                       progress_cb=progress_cb)
        # Nothing detected means nothing to group, so a project with no target
        # classes never pays for the extra pass.
        if not (settings["group_subjects"] and ctx.data.get("subject_classes")):
            clear_vehicle_groups(ctx.db_path)
        else:
            run_vehicle_clustering(
                ctx.db_path,
                partial(pixel_path, ctx.data, ctx.db_path, ctx.project_dir),
                eps=settings["subject_eps"],
                min_samples=settings["subject_min_samples"],
                min_area=settings["subject_min_area"],
                min_score=settings["subject_min_score"],
                progress_cb=progress_cb,
            )

    @app.post("/api/cluster")
    def start_cluster(payload: ClusterPayload | None = None) -> dict[str, Any]:
        _require_loaded()
        with ctx.score_lock:
            if ctx.scoring_state["running"] or ctx.cluster_state["running"]:
                raise HTTPException(status_code=409, detail="another task is already running")
            ctx.cluster_state.update(
                running=True, phase="starting", idx=0, total=0,
                started_at=datetime.now().isoformat(), ended_at=None, error=None,
            )

        settings = _cluster_settings(payload)
        # Remembered on the project, so the next run and the next session start
        # from what was chosen here.
        ctx.data["cluster_settings"] = settings
        db.save(ctx.db_path, ctx.data)

        def progress_cb(phase: str, idx: int, total: int) -> None:
            ctx.cluster_state["phase"] = phase
            ctx.cluster_state["idx"] = idx
            ctx.cluster_state["total"] = total

        def runner() -> None:
            try:
                _run_clustering(settings, progress_cb)
                ctx.reload_data()
                ctx.wipe_face_cache()
            except Exception as exc:
                ctx.cluster_state["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                ctx.cluster_state["running"] = False
                ctx.cluster_state["ended_at"] = datetime.now().isoformat()

        threading.Thread(target=runner, daemon=True).start()
        return {"started": True}

    @app.get("/api/cluster/status")
    def cluster_status() -> dict[str, Any]:
        return ctx.cluster_state

    # ----- people ----------------------------------------------------

    @app.post("/api/people")
    def update_people(payload: PeoplePayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"] or ctx.cluster_state["running"]:
            raise HTTPException(status_code=409, detail="another task is running")
        existing = {p["id"]: p for p in ctx.data.get("people", [])}
        if not existing:
            raise HTTPException(status_code=400, detail="no clusters yet; run /api/cluster first")
        for upd in payload.people:
            cur = existing.get(upd.id)
            if cur is None:
                raise HTTPException(status_code=400, detail=f"unknown person id: {upd.id}")
            cur["label"] = upd.label
            cur["priority"] = upd.priority
            cur["excluded"] = upd.excluded
        ctx.data["people"] = sorted(existing.values(), key=lambda p: p["priority"])
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return {"people": ctx.data["people"]}

    # ----- subjects (detected objects) -------------------------------

    @app.get("/api/subjects")
    def get_subjects() -> dict[str, Any]:
        """Everything the subject UI needs: which classes this project detects,
        which were actually found, and the appearance groups."""
        _require_loaded()
        counts: dict[str, int] = {}
        for photo in ctx.data.get("photos", []):
            for obj in photo.get("objects", []) or []:
                counts[obj["cls"]] = counts.get(obj["cls"], 0) + 1
        return {
            "classes": ctx.data.get("subject_classes", []),
            "available": objects_mod.COCO_CLASSES,
            "presets": {k: list(v) for k, v in objects_mod.CLASS_PRESETS.items()},
            "counts": counts,
            "vehicles": ctx.data.get("vehicles", []),
            "model_ready": objects_mod.is_model_ready(),
            # The re-score dialog offers this alongside the subject classes.
            "detect_faces": ctx.data.get("detect_faces", True),
        }

    @app.post("/api/subjects/groups")
    def update_subject_groups(payload: SubjectGroupsPayload) -> dict[str, Any]:
        """Rename / hide subject groups. Grouping is appearance-based, so fixing
        up a mis-grouped label by hand is expected rather than exceptional."""
        _require_loaded()
        if ctx.scoring_state["running"] or ctx.cluster_state["running"]:
            raise HTTPException(status_code=409, detail="another task is running")
        existing = {v["id"]: v for v in ctx.data.get("vehicles", [])}
        for upd in payload.groups:
            cur = existing.get(upd.id)
            if cur is None:
                raise HTTPException(status_code=400, detail=f"unknown group id: {upd.id}")
            cur["label"] = upd.label
            cur["excluded"] = upd.excluded
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return {"vehicles": ctx.data.get("vehicles", [])}

    @app.get("/subject/{rel_path:path}")
    def get_subject_crop(rel_path: str, idx: int = 0) -> FileResponse:
        """Cropped thumbnail of one detected object — the subject equivalent of
        /face, reusing the same on-disk crop cache."""
        _require_loaded()
        photo = ctx.photo_index.get(rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        obj_list = photo.get("objects") or []
        if idx < 0 or idx >= len(obj_list):
            raise HTTPException(status_code=404, detail="object not found")
        crop_src = _thumb_source(ctx, rel_path, photo)
        crop_path = _ensure_face_crop(
            crop_src, ctx.faces_root, rel_path, obj_list[idx]["bbox_xywh"],
            f"obj{idx}", ctx.db_path, padding=0.12, square=False)
        return FileResponse(crop_path, media_type="image/jpeg")

    # ----- scene grouping --------------------------------------------

    @app.post("/api/scene-grouping")
    def set_scene_grouping(payload: SceneGroupingPayload) -> dict[str, Any]:
        _require_loaded()
        if ctx.scoring_state["running"] or ctx.cluster_state["running"]:
            raise HTTPException(status_code=409, detail="another task is running")
        photos = ctx.data["photos"]
        scenes.regroup(photos, ctx.jpeg_root, payload.mode, payload.gap_minutes)
        # Recompute per-scene auto-suggestions because the groups changed.
        by_scene: dict[str, list[dict[str, Any]]] = {}
        for p in photos:
            by_scene.setdefault(p["scene"], []).append(p)
        for items in by_scene.values():
            apply_scene_suggestions(items)
        ctx.data["scene_grouping"] = {"mode": payload.mode, "gap_minutes": payload.gap_minutes}
        with ctx.save_lock:
            db.save(ctx.db_path, ctx.data)
        return {"scene_grouping": ctx.data["scene_grouping"], "photos": photos}

    # ----- export ----------------------------------------------------

    @app.get("/api/export/picks/preview")
    def export_picks_preview() -> dict[str, Any]:
        _require_loaded()
        picks = [p for p in ctx.data["photos"] if p.get("decision") == "pick"]
        default_target = ctx.db_path.parent / f"{ctx.db_path.stem}.picks"
        return {"count": len(picks), "default_target": str(default_target),
                "settings": ExportSettings(**userstate.get_export_settings()).model_dump()}

    @app.post("/api/export/name-preview")
    def export_name_preview(payload: NamePreviewPayload) -> dict[str, Any]:
        """What the template names the first pick, so the dialog can show an
        example while it is being typed."""
        _require_loaded()
        settings = ExportSettings(name_template=payload.template)
        try:
            exporting.check_template(settings.name_template)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        picks = [p for p in ctx.data["photos"] if p.get("decision") == "pick"]
        photo = (picks or ctx.data["photos"] or [None])[0]
        if photo is None:
            return {"example": None}
        width = exporting.seq_width(len(picks))
        stem = _templated_stem(ctx, photo, photo["rel_path"], settings, 1, width)
        return {"example": stem}

    @app.post("/api/export/picks")
    def export_picks(payload: ExportPayload | None = None) -> dict[str, Any]:
        """Start writing every PICK to a folder. Rendering a few hundred full-size
        frames takes minutes, so it runs in the background and reports through
        /api/export/status."""
        _require_loaded()
        payload = payload or ExportPayload()
        settings = payload.settings
        _check_export_settings(settings)
        target = (
            Path(payload.target_dir).expanduser()
            if payload.target_dir
            else ctx.db_path.parent / f"{ctx.db_path.stem}.picks"
        )
        if not target.is_absolute():
            raise HTTPException(status_code=400, detail="the target folder must be a full path")
        target = target.resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=400, detail=f"cannot use that folder: {exc}") from exc
        userstate.set_export_settings(settings.model_dump())
        # Nothing between claiming the run and starting its thread may fail, or
        # the run would stay marked as running with nothing behind it.
        with ctx.score_lock:
            if (ctx.export_state["running"] or ctx.scoring_state["running"]
                    or ctx.cluster_state["running"] or ctx.opening_state["running"]):
                raise HTTPException(status_code=409, detail="another task is running")
            picks = [p for p in ctx.data["photos"] if p.get("decision") == "pick"]
            ctx.export_state.update(ctx._fresh_export_state(), running=True,
                                    total=len(picks), started_at=datetime.now().isoformat())

        def runner() -> None:
            try:
                ctx.export_state["result"] = _export_picks_to(
                    ctx, picks, target, payload.mode, settings)
            except Exception as exc:
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                ctx.export_state["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                ctx.export_state["running"] = False
                ctx.export_state["current"] = None
                ctx.export_state["ended_at"] = datetime.now().isoformat()

        threading.Thread(target=runner, daemon=True).start()
        return {"started": True, "total": len(picks), "target_dir": str(target)}

    @app.get("/api/export/status")
    def export_status() -> dict[str, Any]:
        return ctx.export_state

    @app.post("/api/export/cancel")
    def export_cancel() -> dict[str, Any]:
        """Stop after the photo being written now; what is done stays done."""
        ctx.export_state["cancel"] = True
        return ctx.export_state

    @app.post("/api/download")
    def download_selection(payload: DownloadPayload) -> dict[str, Any]:
        """Put the selected photos in the user's Downloads folder.

        The point is that it takes one click: no folder picker, no dialog. Edits
        are baked and RAW is rendered, same as an export, because what lands in
        Downloads should be the photo as you graded it.

        A single photo goes straight into Downloads; several go into a dated
        subfolder, so a batch stays together and cannot quietly overwrite
        anything already there.
        """
        _require_loaded()
        if not payload.rel_paths:
            raise HTTPException(status_code=400, detail="nothing selected")

        downloads = Path.home() / "Downloads"
        try:
            downloads.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise HTTPException(
                status_code=400, detail=f"cannot use the Downloads folder: {exc}") from exc

        if len(payload.rel_paths) == 1:
            target = downloads
        else:
            name = (ctx.project_dir.name if ctx.project_dir else ctx.db_path.stem)
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            target = downloads / f"{_sanitize_segment(name)}_{stamp}"
            n = 2
            while target.exists():
                target = downloads / f"{_sanitize_segment(name)}_{stamp}-{n}"
                n += 1
            target.mkdir(parents=True)

        # One click, so no choices: full size and all metadata, as a JPEG.
        settings = ExportSettings()
        used: set[str] = set()
        saved: list[str] = []
        missing: list[str] = []
        for rel in payload.rel_paths:
            photo = ctx.photo_index.get(rel)
            if photo is None or not ctx.source_path(rel).is_file():
                missing.append(rel)
                continue
            dst = target / _unique_name(target, _baked_name(ctx, photo, rel, settings), used)
            _bake_photo(ctx, photo, rel, dst, settings)
            saved.append(dst.name)

        return {"target_dir": str(target), "saved": len(saved),
                "missing": missing, "names": saved[:12]}

    @app.post("/api/reveal")
    def reveal_path(payload: RevealPayload) -> dict[str, Any]:
        """Show a folder in the OS file manager — the natural follow-up to a
        download, and the app already shells out for the folder picker."""
        path = Path(payload.path).expanduser()
        if not path.exists():
            raise HTTPException(status_code=404, detail="path not found")
        try:
            if platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            elif platform.system() == "Windows":
                subprocess.Popen(["explorer", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"revealed": str(path)}

    @app.get("/api/export/one/{rel_path:path}")
    def export_one(rel_path: str) -> Response:
        """Render one photo (RAW and/or edited) at full resolution and return it
        as a downloadable JPEG — the quick way to get a JPEG out of a RAW."""
        _require_loaded()
        photo = ctx.photo_index.get(rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        settings = ExportSettings()
        buf = io.BytesIO()
        _bake_rendered(ctx, photo, rel_path, buf, settings)
        fname = Path(rel_path).stem + ".jpg"
        return Response(
            content=buf.getvalue(), media_type="image/jpeg",
            headers={"Content-Disposition": f'attachment; filename="{fname}"',
                     "Cache-Control": "no-store"})

    # ----- image serving ---------------------------------------------

    @app.get("/img/{rel_path:path}")
    def get_image(rel_path: str) -> Response:
        _require_loaded()
        path = ctx.source_path(rel_path).resolve()
        if not _within_roots(ctx, path):
            raise HTTPException(status_code=403, detail="forbidden")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        photo = ctx.photo_index.get(rel_path)
        edit = photo.get("edit") if photo else None
        is_raw = bool(photo and photo.get("type") == "raw")
        if is_raw or (edit and not editing.is_neutral(edit)):
            # RAW isn't browser-viewable, and edits must show — render a JPEG.
            out = editing.render(ctx.decode_view(rel_path), edit,
                                 meta=ctx.photo_meta(rel_path),
                                 auto=ctx.auto_fields(rel_path, edit),
                                 src=ctx.range_src(rel_path, edit))
            buf = io.BytesIO()
            Image.fromarray(out).save(buf, "JPEG", quality=90)
            return Response(content=buf.getvalue(), media_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
        media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(path, media_type=media)

    @app.get("/thumb/{rel_path:path}")
    def get_thumb(rel_path: str) -> FileResponse:
        _require_loaded()
        photo = ctx.photo_index.get(rel_path)
        src = _thumb_source(ctx, rel_path, photo).resolve()
        if not _within_roots(ctx, src):
            raise HTTPException(status_code=403, detail="forbidden")
        if not src.is_file():
            raise HTTPException(status_code=404, detail="not found")
        edit = photo.get("edit") if photo else None
        thumb = _ensure_thumb(src, ctx.thumbs_root, rel_path, edit,
                              ctx.photo_meta(rel_path),
                              auto_fields=lambda: ctx.auto_fields(rel_path, edit),
                              range_src=lambda: ctx.range_src(rel_path, edit))
        return FileResponse(thumb, media_type="image/jpeg")

    @app.get("/peak/{rel_path:path}")
    def get_focus_peak(rel_path: str, level: str = "normal") -> Response:
        """Focus-peaking overlay: a transparent PNG with the in-focus edges
        picked out, sized to the thumbnail so it lays straight over one.

        Culling by the badness number tells you which frame is sharpest overall;
        this tells you *what* is sharp, which is the actual question when one
        shot nailed the headlight and the next nailed the badge.
        """
        _require_loaded()
        photo = ctx.photo_index.get(rel_path)
        src = _thumb_source(ctx, rel_path, photo).resolve()
        if not _within_roots(ctx, src):
            raise HTTPException(status_code=403, detail="forbidden")
        if not src.is_file():
            raise HTTPException(status_code=404, detail="not found")
        edit = photo.get("edit") if photo else None
        thumb = _ensure_thumb(src, ctx.thumbs_root, rel_path, edit,
                              ctx.photo_meta(rel_path), watermark=False,
                              auto_fields=lambda: ctx.auto_fields(rel_path, edit),
                              range_src=lambda: ctx.range_src(rel_path, edit))
        if level not in PEAK_LEVELS:
            raise HTTPException(status_code=400, detail=f"unknown level: {level}")
        out = _ensure_peak(thumb, ctx.peaks_root, rel_path,
                           editing.edit_hash(edit), level)
        return FileResponse(out, media_type="image/png")

    # ----- automatic masks -------------------------------------------

    @app.get("/api/segment/status")
    def segment_status() -> dict[str, Any]:
        return {
            "model_ready": segment_mod.is_model_ready(),
            "groups": sorted(segment_mod.CLASS_GROUPS),
            "size": segment_mod.MODEL_SIZE,
        }

    @app.post("/api/segment/download")
    def segment_download() -> dict[str, Any]:
        """Fetch the segmentation model. Blocking on purpose: it is 16 MB, it
        happens once ever, and a progress stream for a download that takes a
        couple of seconds would be more machinery than the problem deserves."""
        segment_mod.ensure_model()
        return {"model_ready": segment_mod.is_model_ready()}

    @app.post("/api/mask/preview")
    def mask_preview(payload: MaskPreviewPayload) -> Response:
        """The alpha of one mask as an RGBA PNG — white, with the coverage in the
        alpha channel so it drops straight into the editor's tint pipeline.

        The editor rasterizes a radial, a gradient and a brush itself, from the
        same numbers the server has. It cannot do that for the two kinds derived
        from the pixels: a segmentation is not a shape, and a range selection is
        measured on a fixed grid over the whole ungraded frame, which is not
        something to reproduce in JS from a displayed preview. So those come back
        from here, and what the user sees is what will be graded.
        """
        _require_loaded()
        if payload.rel_path not in ctx.photo_index:
            raise HTTPException(status_code=404, detail="photo not found")
        mask = editing.normalize_mask(payload.mask)
        if mask is None:
            raise HTTPException(status_code=400, detail="not a usable mask")
        base = ctx.get_decoded_base(payload.rel_path)
        h, w = base.shape[:2]
        one = {"masks": [mask]}
        alpha = editing.mask_alpha(
            mask, h, w,
            auto=ctx.auto_fields(payload.rel_path, one),
            src=ctx.range_src(payload.rel_path, one))
        a8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        white = np.full_like(a8, 255)
        ok, buf = cv2.imencode(".png", cv2.merge([white, white, white, a8]))
        assert ok, "failed to encode the mask preview"
        return Response(content=buf.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/api/segment/preview")
    def segment_preview(rel_path: str, group: str) -> Response:
        """The alpha of one automatic mask as a greyscale PNG.

        The editor draws the tint for a radial or a brush itself, from the same
        numbers the server has. It cannot do that for an automatic mask — the
        selection is a segmentation, not a shape — so the one thing that makes
        the feature usable, seeing what it picked, has to be served from here.
        """
        _require_loaded()
        if group not in segment_mod.CLASS_GROUPS:
            raise HTTPException(status_code=400, detail=f"unknown group: {group}")
        if rel_path not in ctx.photo_index:
            raise HTTPException(status_code=404, detail="photo not found")
        alpha = segment_mod.class_mask(
            ctx.get_decoded_base(rel_path), group, key=rel_path)
        # White with the mask in the alpha channel, rather than a greyscale
        # image. The editor's tint pipeline paints a shape's *coverage* and then
        # fills it through `source-in`, so an image whose alpha is the coverage
        # drops straight into that path and an automatic mask tints exactly like
        # a brush does. A greyscale PNG would arrive fully opaque and the client
        # would have to convert luminance to alpha by hand.
        a8 = np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)
        white = np.full_like(a8, 255)
        ok, buf = cv2.imencode(".png", cv2.merge([white, white, white, a8]))
        assert ok, "failed to encode the mask preview"
        return Response(content=buf.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/face/{rel_path:path}")
    def get_face(rel_path: str, idx: int = 0) -> FileResponse:
        _require_loaded()
        photo = ctx.photo_index.get(rel_path)
        if photo is None:
            raise HTTPException(status_code=404, detail="photo not found")
        face_list = photo.get("faces") or []
        if idx < 0 or idx >= len(face_list):
            raise HTTPException(status_code=404, detail="face not found")
        face = face_list[idx]
        bbox = face["bbox_xywh"]
        # For RAW the bbox is in the cached-preview coordinate space (faces were
        # detected on it), so the crop must come from that same JPEG, not the RAW.
        crop_src = _thumb_source(ctx, rel_path, photo)
        crop_path = _ensure_face_crop(
            crop_src, ctx.faces_root, rel_path, bbox, idx, ctx.db_path)
        return FileResponse(crop_path, media_type="image/jpeg")

    return app


def _is_picture_classifier_running(host: str, port: int) -> bool:
    """Best-effort check: is *our* app already serving on this port?"""
    import json
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(
            f"http://{host}:{port}/api/state", timeout=1
        ) as r:
            payload = json.loads(r.read().decode("utf-8", errors="replace"))
        # `/api/state` is unique to our server's API surface.
        return isinstance(payload, dict) and "ready" in payload and "opening" in payload
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _pick_free_port(host: str, preferred: int, attempts: int = 20) -> int:
    """Try sequential ports starting at `preferred`; fall back to an OS-assigned one."""
    import socket
    for candidate in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, candidate))
                return candidate
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def serve(db_path: Path | None, host: str, port: int, open_browser: bool = False) -> None:
    import uvicorn

    # If our app is already serving on the requested port, just point the
    # browser at it instead of failing with EADDRINUSE.
    if _is_picture_classifier_running(host, port):
        url = f"http://{host}:{port}"
        print(f"\n  Picture Classifier is already running — opening {url}\n")
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        return

    actual_port = _pick_free_port(host, port)
    if actual_port != port:
        print(f"  Port {port} is in use; using {actual_port} instead.")

    app = create_app(db_path)
    url = f"http://{host}:{actual_port}"
    print(f"\n  Picture Classifier — open {url}\n")
    if open_browser:
        import threading
        import time
        import webbrowser

        def _open() -> None:
            time.sleep(1.0)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
    # A Server object rather than uvicorn.run, so /api/quit has something to stop.
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=actual_port, log_level="warning"))
    app.state.server = server
    server.run()
