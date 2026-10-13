"""Scan JPEGs, compute per-photo scores, normalize per scene, write JSON."""
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

from PIL import Image

from . import db, exifinfo, fsutil, hdr, raw, scenes
from .projects import PHOTOS_SUBDIR as PROJECTS_SUBDIR
from .scoring import blur as blur_mod
from .scoring import exposure as exp_mod
from .scoring import faces as faces_mod
from .scoring import objects as objects_mod

EYE_CLOSED_THRESHOLD = 0.18  # EAR below this is treated as "eyes closed"

# Subject (object-detection) weights, applied only when a project has target
# classes set. Tuned so a missing subject dominates, prominence matters clearly,
# and placement is only a tiebreaker — plenty of good shots are off-centre.
SUBJECT_MISSING_PENALTY = 0.8
SUBJECT_SMALL_WEIGHT = 0.4
SUBJECT_OFFCENTER_WEIGHT = 0.2
SUBJECT_FULL_AREA = 0.25  # a subject filling this much of the frame gets full credit


# Concurrency for the RAW preview pass. Capped at 8 rather than left to follow
# the core count: each worker holds a half-size decode in memory, and past this
# the disk is the limit anyway.
RAW_PREVIEW_WORKERS = min(8, (os.cpu_count() or 4))

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}


def _is_supported(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTS and not p.name.startswith("._")


def excluded_scan_dirs(db_path: Path, project_dir: Path | None) -> set[Path]:
    """Resolved paths that scanning must prune. Without this, a project or
    legacy cache folder living inside the photo directory would feed its own
    thumbs/face crops (or merged HDR results) back in as 'photos' on every
    re-score."""
    out = {
        db_path.with_suffix(db_path.suffix + ".thumbs").resolve(),
        db_path.with_suffix(db_path.suffix + ".faces").resolve(),
        # Focus-peaking overlays are .png, i.e. a supported extension — without
        # this they would be picked up as photos on the next re-score.
        db_path.with_suffix(db_path.suffix + ".peaks").resolve(),
        db_path.with_suffix(db_path.suffix + ".hdr").resolve(),
        db_path.with_suffix(db_path.suffix + ".rawcache").resolve(),
    }
    if project_dir is not None:
        out.add(project_dir.resolve())
    return out


def ignored_scan_dirs(jpeg_root: Path, raw_root: Path, ignored: Iterable[str]) -> set[Path]:
    """Resolved paths of the top-level folders a project leaves out (its
    `ignored_dirs`), under both roots: a folder of video stills or exports kept
    inside the shoot is not part of it."""
    out: set[Path] = set()
    for name in ignored:
        assert is_folder_name(name), f"not a top-level folder name: {name!r}"
        out.add((jpeg_root / name).resolve())
        out.add((raw_root / name).resolve())
    return out


def is_folder_name(name: str) -> bool:
    """A single path component: what `ignored_dirs` may hold."""
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


def top_folder(rel: str) -> str | None:
    """The top-level folder of a root-relative path; None for a loose file.
    The same component folder grouping names a scene after."""
    parts = Path(rel).parts
    return parts[0] if len(parts) > 1 else None


def photo_folder(photo: dict[str, Any]) -> str | None:
    """The top-level folder a photo was found in. A RAW's rel_path is relative
    to the RAW root and a JPEG's to the JPEG root, as each was scanned; a merged
    HDR result lives in the project, so its bracket's frames say."""
    if photo.get("type") == "hdr":
        return top_folder(photo["members"][0])
    return top_folder(photo["rel_path"])


def source_folders(data: dict[str, Any]) -> dict[str, str | None]:
    """`source_rel_paths`, each mapped to the top-level folder it is in."""
    out: dict[str, str | None] = {}
    for p in data.get("photos", []):
        t = p.get("type")
        if t == "hdr":
            for m in p.get("members", []):
                out[m] = top_folder(m)
        elif t == "raw":
            out["raw::" + p.get("src", p["rel_path"])] = top_folder(p["rel_path"])
        else:
            out[p["rel_path"]] = top_folder(p["rel_path"])
    for b in data.get("brackets", []):
        for m in b.get("members", []):
            out[m] = top_folder(m)
    return out


def hdr_dir(db_path: Path, project_dir: Path | None) -> Path:
    """Directory holding merged HDR JPEGs. In a project it lives under the
    project folder; in the legacy layout it sits beside the db like the
    thumb/face caches."""
    if project_dir is not None:
        return project_dir / "hdr"
    return db_path.with_suffix(db_path.suffix + ".hdr")


def raw_cache_dir(db_path: Path, project_dir: Path | None) -> Path:
    """Directory holding cached RAW preview JPEGs (project .cache/raw, or beside
    the db in the legacy layout)."""
    if project_dir is not None:
        return project_dir / ".cache" / "raw"
    return db_path.with_suffix(db_path.suffix + ".rawcache")


def source_rel_paths(data: dict[str, Any]) -> set[str]:
    """Identities of the on-disk sources a db was built from, for change
    detection. JPEG sources use their jpeg-root-relative rel_path (and bracket
    members); a RAW is tagged `raw::<photo-root-relative src>` so it matches the
    RAW scan regardless of layout. A merged HDR result is not itself a source."""
    out: set[str] = set()
    for p in data.get("photos", []):
        t = p.get("type")
        if t == "hdr":
            out.update(p.get("members", []))
        elif t == "raw":
            out.add("raw::" + p.get("src", p["rel_path"]))
        else:
            out.add(p["rel_path"])
    for b in data.get("brackets", []):
        out.update(b.get("members", []))
    return out


def walk_files(root: Path, exclude_dirs: Iterable[Path] = ()) -> Iterator[Path]:
    """Yield every file under `root`, pruning the listed directory subtrees.
    Comparison is by resolved path so symlinked aliases are caught too."""
    excluded = {Path(p).resolve() for p in exclude_dirs}
    for dirpath, dirnames, filenames in root.walk():
        # A `.pickapicka/` holds projects kept with their photos (projects.py),
        # whichever project is being scanned: their caches are not photos.
        dirnames[:] = [
            d for d in dirnames
            if d != PROJECTS_SUBDIR and (dirpath / d).resolve() not in excluded
        ]
        for fn in filenames:
            yield dirpath / fn


def pair_group(jpegs: list[Path], raws: list[Path]) -> list[tuple[Path | None, Path | None]]:
    """Shots, as (raw, jpeg), from the files sharing one (scene, stem) key.

    The usual group is one RAW and its JPEG, or either alone, and that is one
    shot. A group can hold more: the same file name in two subfolders of one
    scene, from two camera bodies or a counter that wrapped. Every file then
    stays a photo of its own, and a JPEG is only folded into a RAW it plainly
    belongs to: one in the same folder, else one in a folder of the same name
    (RAW/A beside JPEG/A). This used to keep the last file under the key and
    drop the rest without a word."""
    if len(jpegs) <= 1 and len(raws) <= 1:
        return [(raws[0] if raws else None, jpegs[0] if jpegs else None)]
    jl, rl = sorted(jpegs), sorted(raws)
    shots: list[tuple[Path | None, Path | None]] = []
    for same in (lambda r, j: r.parent == j.parent,
                 lambda r, j: r.parent.name == j.parent.name):
        for j in list(jl):
            match = [r for r in rl if same(r, j)]
            if len(match) == 1:
                shots.append((match[0], j))
                jl.remove(j)
                rl.remove(match[0])
    return shots + [(r, None) for r in rl] + [(None, j) for j in jl]


def _collect_jpegs(
    jpeg_root: Path, exclude_dirs: Iterable[Path] = ()
) -> list[tuple[str, Path]]:
    """Recursively find supported images. Scene name = first subdir component
    of the path relative to `jpeg_root`; loose files at the root → '(none)'."""
    out: list[tuple[str, Path]] = []
    for img in sorted(walk_files(jpeg_root, exclude_dirs)):
        if not _is_supported(img):
            continue
        rel = img.relative_to(jpeg_root)
        scene = rel.parts[0] if len(rel.parts) > 1 else "(none)"
        out.append((scene, img))
    return out


def _collect_raws(
    raw_root: Path, exclude_dirs: Iterable[Path] = ()
) -> list[tuple[str, Path]]:
    """Recursively find RAW files under `raw_root`. Scene = first subdir
    component relative to `raw_root` (mirrors `_collect_jpegs`)."""
    out: list[tuple[str, Path]] = []
    if not raw_root.is_dir():
        return out
    for img in sorted(walk_files(raw_root, exclude_dirs)):
        if not raw.is_raw(img):
            continue
        rel = img.relative_to(raw_root)
        scene = rel.parts[0] if len(rel.parts) > 1 else "(none)"
        out.append((scene, img))
    return out


def _load_existing(db_path: Path) -> dict[str, Any] | None:
    """The previous db contents, or None on first scoring."""
    if not db_path.exists():
        return None
    return db.load(db_path)


# What the photographer set on a photo, as opposed to what scoring measured:
# a re-score keeps all of it.
MARKS = ("decision", "decided_at", "rating", "label")


def _existing_decisions(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map rel_path -> its MARKS so a re-score keeps prior calls, stars and
    colour labels. Bracket ids are content-stable, so merged-result decisions
    survive too."""
    if not data:
        return {}
    return {
        p["rel_path"]: {k: p[k] for k in MARKS if p.get(k) is not None}
        for p in data.get("photos", [])
    }


# A photo's grading: the edit, when it was made, and the saved slots.
EDIT_FIELDS = ("edit", "edited_at", "edit_slots")


def _existing_edits(data: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Map rel_path -> its EDIT_FIELDS so a re-score keeps prior grading. The
    slots are grading too: a re-score used to drop them."""
    if not data:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for p in data.get("photos", []):
        kept = {k: p[k] for k in EDIT_FIELDS if p.get(k)}
        if kept:
            out[p["rel_path"]] = kept
    return out


def _resolve_bracket_groups(
    bracket_groups: list[list[str]] | None,
    src_rels: set[str],
    jpeg_root: Path,
    verbose: bool,
) -> list[list[str]]:
    """Decide the HDR bracket grouping for this scoring run. An explicit
    grouping (from an HDR-tab edit) is used as given; otherwise brackets are
    auto-detected from EXIF. A plain re-score always re-detects — a detection
    improvement then takes effect without recreating the project.

    Groups are validated: members must still exist on disk, no frame may
    appear twice, and the size must stay within 3-9.
    """
    if bracket_groups is not None:
        raw = bracket_groups
    else:
        if verbose:
            click.echo("Detecting HDR brackets from EXIF…")
        raw = hdr.detect_brackets(jpeg_root, sorted(src_rels))

    resolved: list[list[str]] = []
    claimed: set[str] = set()
    for members in raw:
        members = [m for m in members if m in src_rels and m not in claimed]
        if hdr.MIN_BRACKET <= len(members) <= hdr.MAX_BRACKET:
            resolved.append(members)
            claimed.update(members)
    return resolved


def _merged_fresh(merged_path: Path, member_paths: list[Path]) -> bool:
    """True when the merged JPEG already exists and is newer than every source
    frame — lets a re-score skip re-merging brackets that did not change."""
    if not merged_path.exists():
        return False
    merged_mtime = merged_path.stat().st_mtime
    return all(p.exists() and p.stat().st_mtime <= merged_mtime for p in member_paths)


def _prune_hdr_dir(hdr_root: Path, keep: set[str]) -> None:
    """Delete merged JPEGs whose bracket no longer exists (membership changed)."""
    if not hdr_root.exists():
        return
    for f in hdr_root.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink()


def pixel_path(
    data: dict[str, Any], db_path: Path, project_dir: Path | None, photo: dict[str, Any],
) -> Path:
    """Absolute path of the image the scorer actually measured for `photo`: the
    merged result for an HDR bracket, the cached preview for a RAW, otherwise
    the source JPEG. Anything re-reading scored pixels should go through this."""
    rel = photo["rel_path"]
    if photo.get("type") == "hdr" or hdr.is_hdr_rel(rel):
        return hdr_dir(db_path, project_dir) / rel[len(hdr.HDR_PREFIX) + 1:]
    if photo.get("type") == "raw":
        return raw.raw_cache_path(raw_cache_dir(db_path, project_dir), rel)
    photo_root = Path(data["photo_root"])
    jpeg_subdir = data.get("jpeg_subdir", "")
    return (photo_root / jpeg_subdir / rel) if jpeg_subdir else (photo_root / rel)


def apply_scene_suggestions(items: list[dict[str, Any]]) -> None:
    """Mutates items in place to set 'auto_suggestion' based on per-scene normalized scores."""
    n = len(items)
    if n == 0:
        return
    if n == 1:
        items[0]["auto_suggestion"] = "review"
        return

    # Rank sharpness on the subject when one was detected, else on the whole
    # frame. Both are measured at the same working scale so they rank together.
    blurs = np.array([
        p["scores"]["subject_blur"] if p["scores"].get("subject_blur") is not None
        else p["scores"]["blur"]
        for p in items
    ])
    brights = np.array([p["scores"]["brightness"] for p in items])
    eyes = np.array([
        np.nan if p["scores"]["eye_open"] is None else p["scores"]["eye_open"]
        for p in items
    ])

    blur_pct = blurs.argsort().argsort() / max(n - 1, 1)
    b_mean = float(brights.mean())
    b_std = float(brights.std()) + 1e-6
    b_zscore = np.abs((brights - b_mean) / b_std)

    badness = (1.0 - blur_pct) + 0.3 * b_zscore
    eyes_closed_mask = ~np.isnan(eyes) & (eyes < EYE_CLOSED_THRESHOLD)
    badness = badness + eyes_closed_mask.astype(float) * 1.0

    # Subject terms only apply where the project asked for object detection —
    # `subject_area` is None on photos that were never run through a detector.
    if any(p["scores"].get("subject_area") is not None for p in items):
        area = np.array([float(p["scores"].get("subject_area") or 0.0) for p in items])
        off = np.array([float(p["scores"].get("subject_center") or 0.0) for p in items])
        badness += (area <= 0.0).astype(float) * SUBJECT_MISSING_PENALTY
        prominence = np.clip(area / SUBJECT_FULL_AREA, 0.0, 1.0)
        badness += SUBJECT_SMALL_WEIGHT * (1.0 - prominence)
        badness += SUBJECT_OFFCENTER_WEIGHT * off

    for i, p in enumerate(items):
        p["scores"]["blur_pct"] = float(blur_pct[i])
        p["scores"]["exposure_zscore"] = float(b_zscore[i])
        p["scores"]["badness"] = float(badness[i])

    n_pick = max(1, int(round(n * 0.3)))
    n_reject = max(1, int(round(n * 0.3)))
    if n_pick + n_reject > n:
        n_reject = max(0, n - n_pick)

    order = np.argsort(badness)
    for rank, idx in enumerate(order):
        if rank < n_pick:
            items[idx]["auto_suggestion"] = "pick"
        elif rank >= n - n_reject:
            items[idx]["auto_suggestion"] = "reject"
        else:
            items[idx]["auto_suggestion"] = "review"


def drop_folders(data: dict[str, Any], folders: set[str]) -> int:
    """Take every photo found in `folders` out of the project, with the
    brackets made of them (see `_drop`). Returns how many photos left."""
    return _drop(data, lambda p: photo_folder(p) in folders,
                 lambda b: top_folder(b["members"][0]) in folders)


def drop_photos(data: dict[str, Any], rel_paths: set[str]) -> int:
    """Take these photos out of the project (their originals gone from disk),
    a merged HDR result with its bracket (see `_drop`). Returns how many left."""
    return _drop(data, lambda p: p["rel_path"] in rel_paths,
                 lambda b: b["merged"] in rel_paths)


def _drop(data: dict[str, Any], photo_leaves, bracket_leaves) -> int:
    """Take the photos `photo_leaves` picks out of the project, and the
    brackets `bracket_leaves` picks, and repair what pointed at them: a
    person's or a subject group's count and representative, and the group
    itself when none of it is left. Scene suggestions are recomputed for what
    remains, as they are relative to a scene. Returns how many photos left."""
    before = len(data["photos"])
    data["photos"] = [p for p in data["photos"] if not photo_leaves(p)]
    data["brackets"] = [b for b in data.get("brackets", []) if not bracket_leaves(b)]

    def repair(groups: list[dict[str, Any]], field: str, items: str, idx: str) -> list[dict[str, Any]]:
        members: dict[str, list[dict[str, Any]]] = {}
        for p in data["photos"]:
            for i, it in enumerate(p.get(items) or []):
                if it.get(field):
                    members.setdefault(it[field], []).append({"rel_path": p["rel_path"], idx: i})
        kept = []
        for g in groups:
            refs = members.get(g["id"], [])
            if not refs:
                continue
            g["count"] = len(refs)
            if g.get("ref") not in refs:
                g["ref"] = refs[0]
            kept.append(g)
        return kept

    data["people"] = repair(data.get("people", []), "person_id", "faces", "face_idx")
    data["vehicles"] = repair(data.get("vehicles", []), "vehicle_id", "objects", "obj_idx")
    _suggest_by_scene(data["photos"])
    return before - len(data["photos"])


def _subject_scores(
    abs_path: Path, objects: list[dict[str, Any]], width: int, height: int,
) -> dict[str, Any]:
    """Prominence / placement / sharpness of the biggest detected subject.

    "Biggest" rather than "most confident": for a car shoot the hero car is the
    one filling the frame, and a background car detected at 0.9 is not the shot.
    """
    out: dict[str, Any] = {
        "subject_count": len(objects),
        "subject_area": 0.0,
        "subject_center": 0.0,
        "subject_blur": None,
    }
    if not objects or width <= 0 or height <= 0:
        return out
    biggest = max(objects, key=lambda o: o["bbox_xywh"][2] * o["bbox_xywh"][3])
    x, y, bw, bh = biggest["bbox_xywh"]
    out["subject_area"] = round(bw * bh / float(width * height), 5)
    # Distance of the subject's centre from the frame's, as a fraction of the
    # half-diagonal, so 0 is dead centre and 1 is the corner.
    dx = (x + bw / 2) / width - 0.5
    dy = (y + bh / 2) / height - 0.5
    out["subject_center"] = round(float(np.hypot(dx, dy) / 0.7071), 4)
    out["subject_blur"] = blur_mod.region_blur_score(str(abs_path), biggest["bbox_xywh"])
    return out


def _image_dims(path: str) -> tuple[int, int]:
    """Displayed width and height without decoding the pixels — for projects that
    do not detect faces, where nothing else has read the file yet. Orientation is
    applied because the db stores displayed dimensions (which is what the face
    detector's cv2.imread produces)."""
    with Image.open(path) as im:
        w, h = im.size
        orient = im.getexif().get(0x0112, 1)
    return (h, w) if orient in (5, 6, 7, 8) else (w, h)


def _score_one(
    target: dict[str, Any],
    face_detect,
    existing: dict[str, tuple[Any, Any]],
    existing_edits: dict[str, dict[str, Any]],
    embedding_sink: list[np.ndarray],
    object_detect=None,
) -> dict[str, Any]:
    """Score one entity described by `target`: `score_path` is the file actually
    read (a source JPEG, a merged HDR result, or a RAW's cached preview JPEG);
    `rel` is its identity in the db; `members` is set for a merged HDR result;
    `kind` is 'jpeg' | 'raw' | 'hdr'; `src`/`captured_at` are stored when set."""
    scene = target["scene"]
    abs_path = target["score_path"]
    rel_path = target["rel"]
    members = target["members"]
    blur_v = blur_mod.blur_score(str(abs_path))
    bright_v = exp_mod.brightness(str(abs_path))
    if face_detect is None:          # this project does not look for people
        face_list: list[dict[str, Any]] = []
        width, height = _image_dims(str(abs_path))
    else:
        face_list, width, height = face_detect(str(abs_path))
    # Pop embeddings into a separate list; write them to a numpy sidecar later.
    for f in face_list:
        emb = f.pop("embedding", None)
        if emb is None:
            f["embedding_idx"] = None
        else:
            f["embedding_idx"] = len(embedding_sink)
            embedding_sink.append(emb)
        f.setdefault("person_id", None)
    ears = [f["ear"] for f in face_list if f.get("ear") is not None]
    eye_v = min(ears) if ears else None
    objects: list[dict[str, Any]] = []
    subject: dict[str, Any] = {"subject_count": 0, "subject_area": None,
                               "subject_center": None, "subject_blur": None}
    if object_detect is not None:
        objects, _, _ = object_detect(str(abs_path))
        subject = _subject_scores(abs_path, objects, width, height)
    prior = existing.get(rel_path, {})
    # Shooting info is read once here and cached in the db; the watermark and
    # the viewer both read it back rather than re-parsing EXIF per render.
    # A merged HDR result inherits the EXIF of its base frame.
    exif_src = target.get("exif_path") or abs_path
    exif = exifinfo.read_any(Path(exif_src), is_raw=target.get("kind") == "raw")
    photo: dict[str, Any] = {
        "rel_path": rel_path,
        "scene": scene,
        "width": width,
        "height": height,
        "faces": face_list,
        "objects": objects,
        "exif": exif,
        "scores": {
            "blur": blur_v,
            "brightness": bright_v,
            "eye_open": eye_v,
            "blur_pct": None,
            "exposure_zscore": None,
            "badness": None,
            **subject,
        },
        "auto_suggestion": None,
        "decision": prior.get("decision"),
        "decided_at": prior.get("decided_at"),
    }
    for k in ("rating", "label"):
        if k in prior:
            photo[k] = prior[k]
    if target.get("captured_at"):
        photo["captured_at"] = target["captured_at"]
    if members is not None:
        photo["type"] = "hdr"
        photo["members"] = members
    if target.get("kind") == "raw":
        photo["type"] = "raw"
        photo["src"] = target["src"]
        photo["full_size"] = target["full_size"]
    photo.update(existing_edits.get(rel_path, {}))
    return photo


def pixel_size(photo: dict[str, Any]) -> tuple[int, int] | None:
    """(width, height) of a photo as it decodes and exports; None when that is
    not known.

    `width` and `height` are the frame the photo was scored on, which its face
    and subject boxes are measured in. For a JPEG or an HDR result that is the
    file itself. A RAW is scored on its cached preview, so its own size is kept
    apart, as `full_size`.
    """
    if photo.get("type") == "raw":
        size = photo.get("full_size")
        return (int(size[0]), int(size[1])) if size else None
    w, h = photo.get("width"), photo.get("height")
    return (int(w), int(h)) if w and h else None


def fill_full_sizes(data: dict[str, Any], photo_dir: Path, progress_cb=None) -> int:
    """Give each RAW photo scored before `full_size` was kept its size, read
    from the RAW (`raw.output_size`). One whose RAW is not on disk is left for
    a later open. `progress_cb(done, total)` is called as they come in and may
    raise to stop. Returns how many were filled."""
    todo = [p for p in data.get("photos", [])
            if p.get("type") == "raw" and not p.get("full_size")
            and (photo_dir / p.get("src", p["rel_path"])).is_file()]
    if not todo:
        return 0
    pool = ThreadPoolExecutor(max_workers=RAW_PREVIEW_WORKERS)
    try:
        sizes = pool.map(lambda p: raw.output_size(photo_dir / p.get("src", p["rel_path"])), todo)
        for n, (p, size) in enumerate(zip(todo, sizes), 1):
            p["full_size"] = list(size)
            if progress_cb is not None:
                progress_cb(n, len(todo))
    finally:
        # As for the previews: a stop drops what is still queued.
        pool.shutdown(wait=True, cancel_futures=True)
    return len(todo)


def embeddings_path(db_path: Path) -> Path:
    """The face embeddings sidecar: row `embedding_idx` of each face."""
    return db_path.with_suffix(db_path.suffix + ".embeddings.npy")


def _merge_brackets(
    groups: list[list[str]], scene_of: dict[str, str], jpeg_root: Path, hdr_root: Path,
    look: dict[str, float], force: bool, progress_cb=None,
) -> list[dict[str, Any]]:
    """Merge each bracket (unless its merged JPEG is fresh and `force` is not
    set) and describe it as the db keeps it."""
    out: list[dict[str, Any]] = []
    for i, members in enumerate(groups):
        merged_name = hdr.merged_filename(members)
        merged_path = hdr_root / merged_name
        member_paths = [jpeg_root / m for m in members]
        if force or not _merged_fresh(merged_path, member_paths):
            if progress_cb is not None:
                progress_cb(0, 0, f"Merging HDR brackets… ({i + 1}/{len(groups)})")
            elif i == 0:
                click.echo(f"Merging {len(groups)} HDR bracket(s)…")
            hdr.merge_bracket(member_paths, merged_path, look)
        out.append({
            "id": hdr.bracket_id(members),
            "members": members,
            "merged": f"{hdr.HDR_PREFIX}/{merged_name}",
            "scene": scene_of[members[0]],
            "base": hdr.base_frame(members, jpeg_root),
        })
    return out


Shot = tuple[str, "Path | None", "Path | None"]


def _shots(collected: list[tuple[str, Path]], raw_collected: list[tuple[str, Path]],
           jpeg_root: Path, bracketed: set[str]) -> list[Shot]:
    """(scene, raw, jpeg) per shot, paired by `pair_group`; a JPEG merged into
    an HDR bracket is not a shot of its own."""
    groups: dict[tuple[str, str], tuple[list[Path], list[Path]]] = {}
    for scene, abs_raw in raw_collected:
        groups.setdefault((scene, abs_raw.stem.lower()), ([], []))[1].append(abs_raw)
    for scene, jpg in collected:
        if str(jpg.relative_to(jpeg_root)) in bracketed:
            continue  # consumed by an HDR merge
        groups.setdefault((scene, jpg.stem.lower()), ([], []))[0].append(jpg)
    return [(key[0], r, j) for key in sorted(groups) for r, j in pair_group(*groups[key])]


def _shot_targets(shots: list[Shot], photo_dir: Path, jpeg_root: Path, raw_root: Path,
                  raw_cache_root: Path, progress_cb=None) -> list[dict[str, Any]]:
    """A scoring target per shot, the RAW's when it has one."""
    raws_to_prepare = [r for _, r, _ in shots if r is not None]
    # Every RAW gets its preview decoded and cached before anything is scored.
    # This is the one part of a scan that threads cleanly — libraw releases the
    # GIL — and on a RAW shoot it is the part that would otherwise dominate:
    # ~700 ms a frame serially against ~100 ms across eight workers, which is
    # quicker than pulling the camera's embedded JPEG used to be.
    prepared: dict[Path, tuple[Path, datetime | None, tuple[int, int]]] = {}
    if raws_to_prepare:
        def _prepare(abs_raw: Path) -> tuple[Path, datetime | None, tuple[int, int]]:
            rel = str(abs_raw.relative_to(raw_root))
            return (raw.ensure_cache(abs_raw, raw_cache_root, rel),
                    raw.read_capture_time(abs_raw), raw.output_size(abs_raw))

        if progress_cb is None:
            click.echo(f"Preparing {len(raws_to_prepare)} RAW preview(s)…")
        pool = ThreadPoolExecutor(max_workers=RAW_PREVIEW_WORKERS)
        try:
            for n, (abs_raw, done) in enumerate(
                    zip(raws_to_prepare, pool.map(_prepare, raws_to_prepare)), 1):
                prepared[abs_raw] = done
                if progress_cb is not None:
                    progress_cb(0, 0, f"Preparing RAW previews… ({n}/{len(raws_to_prepare)})")
        finally:
            # A progress callback may raise to stop the run; the previews
            # still queued are then dropped, not decoded first. Only those
            # already being decoded are waited for.
            pool.shutdown(wait=True, cancel_futures=True)

    targets: list[dict[str, Any]] = []
    for scene, abs_raw, jpg in shots:
        if abs_raw is not None:  # RAW wins when a shot has both
            rel = str(abs_raw.relative_to(raw_root))
            score_path, captured, size = prepared[abs_raw]
            targets.append({
                "scene": scene, "score_path": score_path, "rel": rel,
                "members": None, "kind": "raw",
                "src": str(abs_raw.relative_to(photo_dir)),
                "exif_path": abs_raw,
                "captured_at": captured.isoformat() if captured else None,
                "full_size": list(size),
            })
        else:
            assert jpg is not None
            rel = str(jpg.relative_to(jpeg_root))
            captured = scenes.read_capture_time(jpg)
            targets.append({
                "scene": scene, "score_path": jpg, "rel": rel,
                "members": None, "kind": "jpeg", "src": None,
                "exif_path": jpg,
                "captured_at": captured.isoformat() if captured else None,
            })
    return targets


def _bracket_targets(brackets_meta: list[dict[str, Any]], jpeg_root: Path,
                     hdr_root: Path) -> list[dict[str, Any]]:
    """A scoring target per merged HDR result."""
    targets = []
    for b in brackets_meta:
        captured = scenes.read_capture_time(jpeg_root / b["base"])
        targets.append({
            "scene": b["scene"],
            "score_path": hdr_root / hdr.merged_filename(b["members"]),
            "rel": b["merged"], "members": b["members"], "kind": "hdr", "src": None,
            "exif_path": jpeg_root / b["base"],
            "captured_at": captured.isoformat() if captured else None,
        })
    return targets


def _detectors(detect_faces: bool, with_faces: bool, subject_classes: list[str] | None,
               progress_cb=None) -> tuple[Any, Any, list[str]]:
    """The face and subject detectors a project scores with, and its subject
    classes. None for a pass the project does not run."""
    verbose = progress_cb is None
    if not detect_faces:
        # A car shoot does not want the bystanders clustered into People, and the
        # eyes-closed penalty is meaningless on them.
        if verbose:
            click.echo("Face detection disabled for this project")
        face_detect = None
    elif with_faces:
        from .scoring import eyes as eyes_mod
        if verbose:
            click.echo("Face mesh + eye-open detection enabled (mediapipe)")
        face_detect = eyes_mod.detect
    else:
        if verbose:
            click.echo("Face detection + embedding enabled (insightface buffalo_l)")
        face_detect = faces_mod.detect

    # Object detection is opt-in per project: with no target classes nothing is
    # loaded and the pipeline costs exactly what it did before.
    object_detect = None
    classes = [c for c in (subject_classes or []) if c in objects_mod.COCO_CLASSES]
    if classes:
        if verbose:
            click.echo(f"Subject detection enabled (YOLOX-tiny): {', '.join(classes)}")
        if not objects_mod.is_model_ready():
            msg = "Downloading the detection model (~20 MB, once)…"
            if verbose:
                click.echo(msg)
            elif progress_cb is not None:
                progress_cb(0, 0, msg)
        objects_mod.ensure_model()

        def object_detect(path: str) -> tuple[list[dict[str, Any]], int, int]:
            return objects_mod.detect(path, classes=classes)
    return face_detect, object_detect, classes


def _score_targets(targets: list[dict[str, Any]], face_detect, existing_dec, existing_edits,
                   object_detect, progress_cb=None) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    """Score every target, and the face embeddings in `embedding_idx` order."""
    scored: list[dict[str, Any]] = []
    embedding_sink: list[np.ndarray] = []
    if progress_cb is None:
        with click.progressbar(targets, label="Scoring", show_pos=True) as bar:
            for t in bar:
                scored.append(_score_one(
                    t, face_detect, existing_dec, existing_edits, embedding_sink,
                    object_detect))
    else:
        for i, t in enumerate(targets):
            progress_cb(i, len(targets), t["rel"])
            scored.append(_score_one(
                t, face_detect, existing_dec, existing_edits, embedding_sink,
                object_detect))
        progress_cb(len(targets), len(targets), None)
    return scored, embedding_sink


def _suggest_by_scene(photos: list[dict[str, Any]]) -> None:
    """Per-scene suggestions for every scene, as they are relative to it."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for p in photos:
        by_scene.setdefault(p["scene"], []).append(p)
    for items in by_scene.values():
        apply_scene_suggestions(items)


def _tag_bases(photos: list[dict[str, Any]], brackets_meta: list[dict[str, Any]]) -> None:
    base_by_merged = {b["merged"]: b["base"] for b in brackets_meta}
    for p in photos:
        if p.get("type") == "hdr" and p["rel_path"] in base_by_merged:
            p["base"] = base_by_merged[p["rel_path"]]


def run_scoring(
    photo_dir: Path,
    jpeg_subdir: str,
    db_path: Path,
    with_faces: bool,
    limit: int | None = None,
    progress_cb=None,
    project_dir: Path | None = None,
    bracket_groups: list[list[str]] | None = None,
    hdr_look: dict[str, float] | None = None,
    raw_subdir: str = "",
    subject_classes: list[str] | None = None,
    detect_faces: bool | None = None,
    scene_grouping: dict[str, Any] | None = None,
) -> None:
    """Scan, merge HDR brackets, then score the standalone frames plus the
    merged results. An HDR bracket is scored once, as its merged output.

    RAW files are first-class: when a shot exists as both a RAW and a JPEG (same
    scene + base name), the RAW is preferred. RAW files are scored/thumbnailed
    via a cached preview JPEG and exported as JPEG.

    `bracket_groups`, when given, is an explicit grouping (from the HDR-tab
    edit); otherwise a prior grouping is preserved across re-scores, and a
    first run auto-detects brackets from EXIF.

    Scenes follow `scene_grouping` when given (the new-project wizard), else
    the project's own, else its folders: a re-score used to drop a time-gap
    grouping and hand back folder scenes. Folders in the project's
    `ignored_dirs` are not scanned.

    If `progress_cb` is given, it's called as `cb(idx, total, current_rel_path)`
    before each photo, and once more with `idx == total` and current=None at the
    end. Otherwise a click progress bar prints to stdout."""
    verbose = progress_cb is None
    jpeg_root = photo_dir / jpeg_subdir if jpeg_subdir else photo_dir
    raw_root = photo_dir / raw_subdir if raw_subdir else jpeg_root
    assert jpeg_root.is_dir(), f"not a directory: {jpeg_root}"

    existing = _load_existing(db_path)
    ignored = sorted((existing or {}).get("ignored_dirs") or [])
    excludes = excluded_scan_dirs(db_path, project_dir) | ignored_scan_dirs(jpeg_root, raw_root, ignored)
    collected = _collect_jpegs(jpeg_root, excludes)
    scene_of = {str(p.relative_to(jpeg_root)): scene for scene, p in collected}
    src_rels = set(scene_of)
    raw_collected = _collect_raws(raw_root, excludes)
    if verbose:
        click.echo(f"Found {len(src_rels)} JPEG/PNG under {jpeg_root}"
                   + (f" and {len(raw_collected)} RAW under {raw_root}" if raw_collected else ""))

    existing_dec = _existing_decisions(existing)
    existing_edits = _existing_edits(existing)
    if verbose and existing_dec:
        click.echo(
            f"Preserving {sum(1 for d in existing_dec.values() if d.get('decision') is not None)} "
            "prior decisions"
        )

    # The merge "look": explicit > prior project setting > default. When it
    # changes, every bracket must be re-merged even if its sources are stale.
    look = hdr_look if hdr_look is not None else (
        (existing or {}).get("hdr_look") or hdr.DEFAULT_LOOK
    )
    look_changed = (existing or {}).get("hdr_look") != look

    # Same precedence for the subject classes: explicit > what the project was
    # last scored with, so a plain re-score keeps detecting what it detected.
    if subject_classes is None:
        subject_classes = (existing or {}).get("subject_classes") or []
    # None means "leave it as the project has it", so a re-score that does not
    # mention faces does not silently change whether they are looked for.
    if detect_faces is None:
        stored = (existing or {}).get("detect_faces")
        detect_faces = True if stored is None else bool(stored)

    # ----- resolve & merge HDR brackets -----
    groups = _resolve_bracket_groups(bracket_groups, src_rels, jpeg_root, verbose)
    hdr_root = hdr_dir(db_path, project_dir)
    brackets_meta = _merge_brackets(groups, scene_of, jpeg_root, hdr_root, look,
                                    look_changed, progress_cb)
    bracketed = {m for b in brackets_meta for m in b["members"]}
    _prune_hdr_dir(hdr_root, {hdr.merged_filename(b["members"]) for b in brackets_meta})

    # ----- pair RAW + JPEG (prefer RAW) and assemble scoring targets -----
    raw_cache_root = raw_cache_dir(db_path, project_dir)
    shots = _shots(collected, raw_collected, jpeg_root, bracketed)
    targets = _shot_targets(shots, photo_dir, jpeg_root, raw_root, raw_cache_root, progress_cb)
    targets += _bracket_targets(brackets_meta, jpeg_root, hdr_root)
    if limit is not None:
        targets = targets[:limit]

    face_detect, object_detect, classes = _detectors(
        detect_faces, with_faces, subject_classes, progress_cb)
    scored, embedding_sink = _score_targets(
        targets, face_detect, existing_dec, existing_edits, object_detect, progress_cb)

    # Scored into folder scenes; regrouped before the per-scene suggestions,
    # which are relative to whatever a scene turns out to hold.
    grouping = scene_grouping if scene_grouping is not None else (existing or {}).get("scene_grouping")
    if grouping and grouping.get("mode") in scenes.TIMED_MODES:
        scenes.regroup(scored, jpeg_root, grouping["mode"], grouping.get("gap_minutes", 30))
    _suggest_by_scene(scored)
    # Tag each merged result with its 0-EV source frame (for the HDR/original
    # compare toggle in the viewer).
    _tag_bases(scored, brackets_meta)

    # Persist embeddings to numpy sidecar; clusters wipe on re-score because
    # face indices may have changed.
    if embedding_sink:
        emb_arr = np.stack(embedding_sink, axis=0).astype(np.float32)
    else:
        emb_arr = np.zeros((0, 512), dtype=np.float32)
    np.save(embeddings_path(db_path), emb_arr)

    data = db.init_db(photo_dir, jpeg_subdir, raw_subdir)
    data["scored_at"] = datetime.now().isoformat()
    data["clustered_at"] = None
    data["people"] = []
    data["brackets"] = brackets_meta
    data["hdr_look"] = look
    data["subject_classes"] = classes
    data["detect_faces"] = bool(detect_faces)
    data["vehicles"] = []
    # Scoring throws the groups away — embeddings are recomputed and every
    # person_id goes back to None — but the *settings* that produced them are a
    # preference, so they are carried across rather than reset with the data.
    if (existing or {}).get("cluster_settings") is not None:
        data["cluster_settings"] = (existing or {})["cluster_settings"]
    # The same goes for how scenes are made and which folders are left out.
    if grouping:
        data["scene_grouping"] = grouping
    if ignored:
        data["ignored_dirs"] = ignored
    # The project's own copies of the looks its edits use: without them an edit
    # whose look has left the app's library, or another machine's, cannot render.
    if (existing or {}).get("luts"):
        data["luts"] = (existing or {})["luts"]
    # Which project this is, wherever its folder goes (projects.ensure_id).
    if (existing or {}).get("project_id"):
        data["project_id"] = (existing or {})["project_id"]
    data["photos"] = scored
    db.save(db_path, data)

    if verbose:
        summary = {"pick": 0, "review": 0, "reject": 0}
        n_with_faces = 0
        total_faces = 0
        for p in scored:
            summary[p["auto_suggestion"]] += 1
            if p["faces"]:
                n_with_faces += 1
                total_faces += len(p["faces"])
        click.echo(f"\nWrote {db_path}")
        if brackets_meta:
            n_frames = sum(len(b["members"]) for b in brackets_meta)
            click.echo(
                f"HDR — merged {len(brackets_meta)} bracket(s) from {n_frames} frames"
            )
        click.echo(
            f"Auto suggestions — pick: {summary['pick']}, "
            f"review: {summary['review']}, reject: {summary['reject']}"
        )
        click.echo(
            f"Faces detected — {n_with_faces}/{len(scored)} photos contain faces "
            f"({total_faces} total)"
        )
        if classes:
            n_with = sum(1 for p in scored if p["objects"])
            n_obj = sum(len(p["objects"]) for p in scored)
            click.echo(
                f"Subjects detected — {n_with}/{len(scored)} photos contain "
                f"{', '.join(classes)} ({n_obj} total)"
            )


# ----- what changed on disk while the project was open ----------------------
# Photos added to the folder are scored and folded in on their own, and photos
# whose originals are gone are found, without re-scoring the rest: a re-score
# costs minutes on a big shoot and regroups faces, which resets people's names.

def _roots(data: dict[str, Any]) -> tuple[Path, Path, Path]:
    photo_dir = Path(data["photo_root"])
    jpeg_subdir, raw_subdir = data.get("jpeg_subdir") or "", data.get("raw_subdir") or ""
    jpeg_root = photo_dir / jpeg_subdir if jpeg_subdir else photo_dir
    return photo_dir, jpeg_root, photo_dir / raw_subdir if raw_subdir else jpeg_root


def source_file(data: dict[str, Any], db_path: Path, project_dir: Path | None,
                photo: dict[str, Any]) -> Path:
    """The file a photo is read from when it is shown or edited: a RAW's
    original, the merged JPEG for an HDR result, else its JPEG."""
    photo_dir, jpeg_root, _ = _roots(data)
    if photo.get("type") == "hdr" or hdr.is_hdr_rel(photo["rel_path"]):
        return hdr_dir(db_path, project_dir) / photo["rel_path"][len(hdr.HDR_PREFIX) + 1:]
    if photo.get("type") == "raw":
        return photo_dir / photo.get("src", photo["rel_path"])
    return jpeg_root / photo["rel_path"]


def find_changes(data: dict[str, Any], db_path: Path, project_dir: Path | None) -> dict[str, Any]:
    """The folder against the project: shots on disk that are not in it, ones
    that pair with a photo already in it (a RAW arriving for a JPEG: only a
    re-score swaps those), and photos whose originals are gone. Scans the
    same files, the same way, as scoring does."""
    photo_dir, jpeg_root, raw_root = _roots(data)
    if not jpeg_root.is_dir():
        # The drive is not plugged in (or the folder moved, which re-linking is
        # for): nothing is reported gone, or the bar would offer to remove all.
        return {"shots": [], "with_existing": 0, "missing": [], "moved": [], "member_moves": {}}
    ignored = data.get("ignored_dirs") or []
    excludes = excluded_scan_dirs(db_path, project_dir) | ignored_scan_dirs(jpeg_root, raw_root, ignored)
    collected = _collect_jpegs(jpeg_root, excludes)
    raw_collected = _collect_raws(raw_root, excludes)

    known: set[Path] = set()
    for p in data["photos"]:
        if p.get("type") == "hdr":
            known.update(jpeg_root / m for m in p.get("members") or [])
        else:
            known.add(source_file(data, db_path, project_dir, p))
    for b in data.get("brackets", []):
        known.update(jpeg_root / m for m in b["members"])

    shots: list[Shot] = []
    with_existing = 0
    for scene, raw_f, jpeg_f in _shots(collected, raw_collected, jpeg_root, set()):
        if raw_f is not None and raw_f in known:
            continue            # in the project; a JPEG of it is folded in, as scoring does
        if raw_f is None and jpeg_f in known:
            continue
        if raw_f is not None and jpeg_f is not None and jpeg_f in known:
            with_existing += 1  # a RAW for a JPEG photo: it would take the photo's place
            continue
        shots.append((scene, raw_f, jpeg_f))
    gone = [p for p in data["photos"] if not source_file(data, db_path, project_dir, p).is_file()]
    moved, member_moves, claimed = _find_moves(data, gone, shots, photo_dir, jpeg_root, raw_root)
    followed = {m["rel"] for m in moved}
    return {"shots": [s for i, s in enumerate(shots) if i not in claimed],
            "with_existing": with_existing,
            "missing": [p["rel_path"] for p in gone if p["rel_path"] not in followed],
            "moved": moved, "member_moves": member_moves}


def shot_files(data: dict[str, Any], db_path: Path, project_dir: Path | None,
               photos: list[dict[str, Any]]) -> dict[str, list[Path]]:
    """Every file on disk that is an original of each photo, by rel_path.

    Paired the way scoring pairs them (`_shots`): a RAW photo is its RAW and
    the JPEG the camera wrote with it, which the project does not record, as
    the RAW won; a JPEG photo is its JPEG and a RAW that arrived for it later.
    An HDR result is its bracket's frames; the merged JPEG is the project's own
    file, not an original. A photo whose file is gone has none."""
    _, jpeg_root, raw_root = _roots(data)
    if not jpeg_root.is_dir():
        return {p["rel_path"]: [] for p in photos}
    ignored = data.get("ignored_dirs") or []
    excludes = excluded_scan_dirs(db_path, project_dir) | ignored_scan_dirs(jpeg_root, raw_root, ignored)
    bracketed = {m for b in data.get("brackets", []) for m in b["members"]}
    shots = _shots(_collect_jpegs(jpeg_root, excludes), _collect_raws(raw_root, excludes),
                   jpeg_root, bracketed)
    by_file: dict[Path, list[Path]] = {}
    for _, r, j in shots:
        files = [f for f in (r, j) if f is not None]
        for f in files:
            by_file[f] = files
    out: dict[str, list[Path]] = {}
    for p in photos:
        if p.get("type") == "hdr":
            out[p["rel_path"]] = [jpeg_root / m for m in p.get("members") or []
                                  if (jpeg_root / m).is_file()]
        else:
            src = source_file(data, db_path, project_dir, p)
            out[p["rel_path"]] = by_file.get(src, [src] if src.is_file() else [])
    return out


def _same_shot(photo: dict[str, Any], f: Path, kind: str) -> bool:
    """Whether file `f` is `photo` moved: the same capture time as it was
    scored with, or for one with none, a JPEG of the same size. A file of the
    same name from another card is then not taken for it."""
    stored = photo.get("captured_at")
    if stored:
        t = raw.read_capture_time(f) if kind == "raw" else scenes.read_capture_time(f)
        return t is not None and t.isoformat() == stored
    if kind == "jpeg" and photo.get("width") and photo.get("height"):
        return _image_dims(str(f)) == (photo["width"], photo["height"])
    return True


def _find_moves(data: dict[str, Any], gone: list[dict[str, Any]], shots: list[Shot],
                photo_dir: Path, jpeg_root: Path, raw_root: Path,
                ) -> tuple[list[dict[str, str]], dict[str, str], set[int]]:
    """Photos whose originals are gone that turn up elsewhere in the folder:
    a subfolder renamed, or files moved between subfolders. A gone photo and a
    new shot of the same file name are one photo moved when `_same_shot` says
    so and no other new shot could be it; two candidates and it is left as it
    is, gone and new. Frames of an HDR bracket are matched by name alone, as
    nothing else about them is kept. Returns the photos moved (rel, new rel,
    new src), the bracket frames moved (old -> new), and which shots were
    claimed."""
    by_name: dict[tuple[str, str], list[int]] = {}
    for i, (_, r, j) in enumerate(shots):
        f = r if r is not None else j
        by_name.setdefault(("raw" if r is not None else "jpeg", f.name.lower()), []).append(i)
    claimed: set[int] = set()
    moved: list[dict[str, str]] = []
    for p in gone:
        if p.get("type") == "hdr":
            continue    # its merged JPEG is in the project; its frames are below
        kind = "raw" if p.get("type") == "raw" else "jpeg"
        name = Path(p.get("src", p["rel_path"]) if kind == "raw" else p["rel_path"]).name.lower()
        cands = [i for i in by_name.get((kind, name), []) if i not in claimed]
        if len(cands) != 1:
            continue
        _, r, j = shots[cands[0]]
        f = r if kind == "raw" else j
        if not _same_shot(p, f, kind):
            continue
        claimed.add(cands[0])
        if kind == "raw":
            moved.append({"rel": p["rel_path"], "new_rel": str(r.relative_to(raw_root)),
                          "src": str(r.relative_to(photo_dir))})
        else:
            moved.append({"rel": p["rel_path"], "new_rel": str(j.relative_to(jpeg_root))})
    member_moves: dict[str, str] = {}
    for b in data.get("brackets", []):
        for m in b["members"]:
            if (jpeg_root / m).is_file():
                continue
            cands = [i for i in by_name.get(("jpeg", Path(m).name.lower()), []) if i not in claimed]
            if len(cands) == 1:
                claimed.add(cands[0])
                member_moves[m] = str(shots[cands[0]][2].relative_to(jpeg_root))
    return moved, member_moves, claimed


def _cache_roots(db_path: Path, project_dir: Path | None) -> list[Path]:
    """The caches whose files are named after a photo's rel_path."""
    if project_dir is not None:
        cache = project_dir / ".cache"
        return [cache / "thumbs", cache / "faces", cache / "peaks"]
    return [db_path.with_suffix(db_path.suffix + ext) for ext in (".thumbs", ".faces", ".peaks")]


def _move_cached(roots: list[Path], raw_cache_root: Path, old: str, new: str) -> None:
    """Move what is cached of a photo to its new rel_path: its thumbnails, face
    crops and peaking overlays (`<rel>`, `<rel>.<tag>.jpg`, …), and a RAW's
    preview, filed under a hash of it. What is not moved is made again.

    Through fsutil.replace, which waits out a reader on Windows: the grid may
    be streaming the very thumbnail, and a bare rename then failed with
    WinError 32 half way through bringing the project in line."""
    for root in roots:
        src_dir = (root / old).parent
        if not src_dir.is_dir():
            continue
        base = Path(old).name
        for f in list(src_dir.glob(glob_escape(base) + "*")):
            rest = f.name[len(base):]
            if rest and rest[0] not in ".-":
                continue    # another photo whose name starts with this one's
            dst = root / (new + rest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            fsutil.replace(f, dst)
    cached = raw.raw_cache_path(raw_cache_root, old)
    if cached.is_file():
        fsutil.replace(cached, raw.raw_cache_path(raw_cache_root, new))


def glob_escape(name: str) -> str:
    return "".join(f"[{c}]" if c in "*?[" else c for c in name)


def follow_moves(data: dict[str, Any], moved: list[dict[str, str]], member_moves: dict[str, str],
                 db_path: Path, project_dir: Path | None) -> int:
    """Point the photos `find_changes` found moved at where they are now,
    keeping all that is theirs: marks, edits, scores, faces. A bracket whose
    frames moved gets the id and merged file name of its new frames. What
    refers to a photo by its path follows it (a group's representative, the
    caches), and scenes are made again, as a folder may be a scene. Returns
    how many photos moved."""
    _, jpeg_root, _ = _roots(data)
    by_rel = {p["rel_path"]: p for p in data["photos"]}
    renames: dict[str, str] = {}
    for mv in moved:
        p = by_rel[mv["rel"]]
        renames[mv["rel"]] = mv["new_rel"]
        p["rel_path"] = mv["new_rel"]
        if "src" in mv:
            p["src"] = mv["src"]

    hdr_root = hdr_dir(db_path, project_dir)
    for b in data.get("brackets", []):
        if not any(m in member_moves for m in b["members"]):
            continue
        members = [member_moves.get(m, m) for m in b["members"]]
        old_merged, new_name = b["merged"], hdr.merged_filename(members)
        old_file = hdr_root / old_merged[len(hdr.HDR_PREFIX) + 1:]
        if old_file.is_file():
            fsutil.replace(old_file, hdr_root / new_name)
        b.update(id=hdr.bracket_id(members), members=members,
                 merged=f"{hdr.HDR_PREFIX}/{new_name}", base=member_moves.get(b["base"], b["base"]))
        hp = by_rel.get(old_merged)
        if hp is not None:
            hp.update(rel_path=b["merged"], members=members, base=b["base"])
            renames[old_merged] = b["merged"]

    for group in data.get("people", []) + data.get("vehicles", []):
        ref = group.get("ref") or {}
        if ref.get("rel_path") in renames:
            ref["rel_path"] = renames[ref["rel_path"]]
    roots = _cache_roots(db_path, project_dir)
    for old, new in renames.items():
        _move_cached(roots, raw_cache_dir(db_path, project_dir), old, new)

    grouping = data.get("scene_grouping") or {}
    scenes.regroup(data["photos"], jpeg_root, grouping.get("mode", "folder"), grouping.get("gap_minutes", 30))
    _suggest_by_scene(data["photos"])
    return len(moved) + sum(1 for b in data.get("brackets", [])
                            if any(m in member_moves.values() for m in b["members"]))


def score_new(data: dict[str, Any], db_path: Path, project_dir: Path | None,
              progress_cb=None) -> dict[str, Any]:
    """Score the shots `find_changes` finds new, as the project would score
    them (its detectors, its HDR look), for `fold_in` to add. HDR brackets are
    looked for among the new frames only."""
    photo_dir, jpeg_root, raw_root = _roots(data)
    changes = find_changes(data, db_path, project_dir)
    shots = changes["shots"]
    hdr_root = hdr_dir(db_path, project_dir)
    new_jpegs = {str(j.relative_to(jpeg_root)): scene for scene, r, j in shots
                 if r is None and j is not None}
    groups = _resolve_bracket_groups(None, set(new_jpegs), jpeg_root, verbose=False)
    look = data.get("hdr_look") or hdr.DEFAULT_LOOK
    brackets_meta = _merge_brackets(groups, new_jpegs, jpeg_root, hdr_root, look, False, progress_cb)
    bracketed = {jpeg_root / m for b in brackets_meta for m in b["members"]}
    shots = [s for s in shots if s[1] is not None or s[2] not in bracketed]

    targets = _shot_targets(shots, photo_dir, jpeg_root, raw_root,
                            raw_cache_dir(db_path, project_dir), progress_cb)
    targets += _bracket_targets(brackets_meta, jpeg_root, hdr_root)
    detect = data.get("detect_faces")
    face_detect, object_detect, _ = _detectors(
        True if detect is None else bool(detect), False, data.get("subject_classes"), progress_cb)
    scored, embeddings = _score_targets(targets, face_detect, {}, {}, object_detect, progress_cb)
    _tag_bases(scored, brackets_meta)
    return {"photos": scored, "brackets": brackets_meta,
            "embeddings": np.stack(embeddings).astype(np.float32) if embeddings
            else np.zeros((0, 512), dtype=np.float32),
            "with_existing": changes["with_existing"]}


def fold_in(data: dict[str, Any], new: dict[str, Any], db_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Add what `score_new` scored to the project: the photos, their brackets,
    their face embeddings (appended to the sidecar, their indices moved past
    the rows already there). Scenes are regrouped by time if the project groups
    that way, and every scene's suggestions recomputed, as they are relative
    to what a scene holds. Returns the new faces and subjects, as (photo_idx,
    face/object idx), for grouping."""
    emb_path = embeddings_path(db_path)
    old = np.load(emb_path) if emb_path.is_file() else np.zeros((0, 512), dtype=np.float32)
    offset = old.shape[0]
    start = len(data["photos"])
    for p in new["photos"]:
        for f in p["faces"]:
            if f.get("embedding_idx") is not None:
                f["embedding_idx"] += offset
    if len(new["embeddings"]):
        np.save(emb_path, np.concatenate([old, new["embeddings"]]).astype(np.float32))
    data["photos"].extend(new["photos"])
    data.setdefault("brackets", []).extend(new["brackets"])

    grouping = data.get("scene_grouping") or {}
    if grouping.get("mode") in scenes.TIMED_MODES:
        scenes.regroup(data["photos"], _roots(data)[1], grouping["mode"], grouping.get("gap_minutes", 30))
    _suggest_by_scene(data["photos"])
    faces = [(pi, fi) for pi in range(start, len(data["photos"]))
             for fi, f in enumerate(data["photos"][pi]["faces"]) if f.get("embedding_idx") is not None]
    objects = [(pi, oi) for pi in range(start, len(data["photos"]))
               for oi in range(len(data["photos"][pi].get("objects") or []))]
    return {"faces": faces, "objects": objects}
