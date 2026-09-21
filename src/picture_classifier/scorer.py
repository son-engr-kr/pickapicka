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

from . import db, exifinfo, hdr, raw, scenes
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
        dirnames[:] = [
            d for d in dirnames if (dirpath / d).resolve() not in excluded
        ]
        for fn in filenames:
            yield dirpath / fn


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


def _existing_decisions(data: dict[str, Any] | None) -> dict[str, tuple[Any, Any]]:
    """Map rel_path -> (decision, decided_at) so a re-score keeps prior calls.
    Bracket ids are content-stable, so merged-result decisions survive too."""
    if not data:
        return {}
    return {
        p["rel_path"]: (p.get("decision"), p.get("decided_at"))
        for p in data.get("photos", [])
    }


def _existing_edits(data: dict[str, Any] | None) -> dict[str, tuple[Any, Any]]:
    """Map rel_path -> (edit, edited_at) so a re-score keeps prior grading."""
    if not data:
        return {}
    return {
        p["rel_path"]: (p.get("edit"), p.get("edited_at"))
        for p in data.get("photos", [])
        if p.get("edit")
    }


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
    existing_edits: dict[str, tuple[Any, Any]],
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
    prior_decision, prior_decided_at = existing.get(rel_path, (None, None))
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
        "decision": prior_decision,
        "decided_at": prior_decided_at,
    }
    if target.get("captured_at"):
        photo["captured_at"] = target["captured_at"]
    if members is not None:
        photo["type"] = "hdr"
        photo["members"] = members
    if target.get("kind") == "raw":
        photo["type"] = "raw"
        photo["src"] = target["src"]
    prior_edit, prior_edited_at = existing_edits.get(rel_path, (None, None))
    if prior_edit:
        photo["edit"] = prior_edit
        if prior_edited_at:
            photo["edited_at"] = prior_edited_at
    return photo


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
) -> None:
    """Scan, merge HDR brackets, then score the standalone frames plus the
    merged results. An HDR bracket is scored once, as its merged output.

    RAW files are first-class: when a shot exists as both a RAW and a JPEG (same
    scene + base name), the RAW is preferred. RAW files are scored/thumbnailed
    via a cached preview JPEG and exported as JPEG.

    `bracket_groups`, when given, is an explicit grouping (from the HDR-tab
    edit); otherwise a prior grouping is preserved across re-scores, and a
    first run auto-detects brackets from EXIF.

    If `progress_cb` is given, it's called as `cb(idx, total, current_rel_path)`
    before each photo, and once more with `idx == total` and current=None at the
    end. Otherwise a click progress bar prints to stdout."""
    verbose = progress_cb is None
    jpeg_root = photo_dir / jpeg_subdir if jpeg_subdir else photo_dir
    raw_root = photo_dir / raw_subdir if raw_subdir else jpeg_root
    assert jpeg_root.is_dir(), f"not a directory: {jpeg_root}"

    excludes = excluded_scan_dirs(db_path, project_dir)
    collected = _collect_jpegs(jpeg_root, excludes)
    scene_of = {str(p.relative_to(jpeg_root)): scene for scene, p in collected}
    src_rels = set(scene_of)
    raw_collected = _collect_raws(raw_root, excludes)
    if verbose:
        click.echo(f"Found {len(src_rels)} JPEG/PNG under {jpeg_root}"
                   + (f" and {len(raw_collected)} RAW under {raw_root}" if raw_collected else ""))

    existing = _load_existing(db_path)
    existing_dec = _existing_decisions(existing)
    existing_edits = _existing_edits(existing)
    if verbose and existing_dec:
        click.echo(
            f"Preserving {sum(1 for d in existing_dec.values() if d[0] is not None)} "
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
    brackets_meta: list[dict[str, Any]] = []
    bracketed: set[str] = set()
    for i, members in enumerate(groups):
        merged_name = hdr.merged_filename(members)
        merged_path = hdr_root / merged_name
        member_paths = [jpeg_root / m for m in members]
        if look_changed or not _merged_fresh(merged_path, member_paths):
            if progress_cb is not None:
                progress_cb(0, 0, f"Merging HDR brackets… ({i + 1}/{len(groups)})")
            elif i == 0:
                click.echo(f"Merging {len(groups)} HDR bracket(s)…")
            hdr.merge_bracket(member_paths, merged_path, look)
        brackets_meta.append({
            "id": hdr.bracket_id(members),
            "members": members,
            "merged": f"{hdr.HDR_PREFIX}/{merged_name}",
            "scene": scene_of[members[0]],
            "base": hdr.base_frame(members, jpeg_root),
        })
        bracketed.update(members)
    _prune_hdr_dir(hdr_root, {hdr.merged_filename(b["members"]) for b in brackets_meta})

    # ----- pair RAW + JPEG (prefer RAW) and assemble scoring targets -----
    raw_cache_root = raw_cache_dir(db_path, project_dir)

    def _key(scene: str, path: Path) -> tuple[str, str]:
        return (scene, path.stem.lower())

    raw_by_key: dict[tuple[str, str], tuple[str, Path]] = {}
    for scene, abs_raw in raw_collected:
        raw_by_key[_key(scene, abs_raw)] = (scene, abs_raw)
    jpeg_by_key: dict[tuple[str, str], tuple[str, Path, str]] = {}
    for scene, jpg in collected:
        rel = str(jpg.relative_to(jpeg_root))
        if rel in bracketed:
            continue  # consumed by an HDR merge
        jpeg_by_key[_key(scene, jpg)] = (scene, jpg, rel)

    # Every RAW gets its preview decoded and cached before anything is scored.
    # This is the one part of a scan that threads cleanly — libraw releases the
    # GIL — and on a RAW shoot it is the part that would otherwise dominate:
    # ~700 ms a frame serially against ~100 ms across eight workers, which is
    # quicker than pulling the camera's embedded JPEG used to be.
    prepared: dict[tuple[str, str], tuple[Path, datetime | None]] = {}
    if raw_by_key:
        def _prepare(key: tuple[str, str]) -> tuple[Path, datetime | None]:
            _scene, abs_raw = raw_by_key[key]
            rel = str(abs_raw.relative_to(raw_root))
            return (raw.ensure_cache(abs_raw, raw_cache_root, rel),
                    raw.read_capture_time(abs_raw))

        raw_keys = sorted(raw_by_key)
        if verbose:
            click.echo(f"Preparing {len(raw_keys)} RAW preview(s)…")
        with ThreadPoolExecutor(max_workers=RAW_PREVIEW_WORKERS) as pool:
            for n, (key, done) in enumerate(zip(raw_keys, pool.map(_prepare, raw_keys)), 1):
                prepared[key] = done
                if progress_cb is not None:
                    progress_cb(0, 0, f"Preparing RAW previews… ({n}/{len(raw_keys)})")

    targets: list[dict[str, Any]] = []
    for key in sorted(set(raw_by_key) | set(jpeg_by_key)):
        if key in raw_by_key:  # RAW wins when a shot has both
            scene, abs_raw = raw_by_key[key]
            rel = str(abs_raw.relative_to(raw_root))
            score_path, captured = prepared[key]
            targets.append({
                "scene": scene, "score_path": score_path, "rel": rel,
                "members": None, "kind": "raw",
                "src": str(abs_raw.relative_to(photo_dir)),
                "exif_path": abs_raw,
                "captured_at": captured.isoformat() if captured else None,
            })
        else:
            scene, jpg, rel = jpeg_by_key[key]
            captured = scenes.read_capture_time(jpg)
            targets.append({
                "scene": scene, "score_path": jpg, "rel": rel,
                "members": None, "kind": "jpeg", "src": None,
                "exif_path": jpg,
                "captured_at": captured.isoformat() if captured else None,
            })
    for b in brackets_meta:
        captured = scenes.read_capture_time(jpeg_root / b["base"])
        targets.append({
            "scene": b["scene"],
            "score_path": hdr_root / hdr.merged_filename(b["members"]),
            "rel": b["merged"], "members": b["members"], "kind": "hdr", "src": None,
            "exif_path": jpeg_root / b["base"],
            "captured_at": captured.isoformat() if captured else None,
        })
    if limit is not None:
        targets = targets[:limit]

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

    scored: list[dict[str, Any]] = []
    embedding_sink: list[np.ndarray] = []
    if verbose:
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

    by_scene: dict[str, list[dict[str, Any]]] = {}
    for p in scored:
        by_scene.setdefault(p["scene"], []).append(p)
    for scene, items in by_scene.items():
        apply_scene_suggestions(items)

    # Tag each merged result with its 0-EV source frame (for the HDR/original
    # compare toggle in the viewer).
    base_by_merged = {b["merged"]: b["base"] for b in brackets_meta}
    for p in scored:
        if p.get("type") == "hdr":
            p["base"] = base_by_merged.get(p["rel_path"])

    # Persist embeddings to numpy sidecar; clusters wipe on re-score because
    # face indices may have changed.
    emb_path = db_path.with_suffix(db_path.suffix + ".embeddings.npy")
    if embedding_sink:
        emb_arr = np.stack(embedding_sink, axis=0).astype(np.float32)
    else:
        emb_arr = np.zeros((0, 512), dtype=np.float32)
    np.save(emb_path, emb_arr)

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
