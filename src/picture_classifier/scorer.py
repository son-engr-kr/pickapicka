"""Scan JPEGs, compute per-photo scores, normalize per scene, write JSON."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np

from . import db, hdr
from .scoring import blur as blur_mod
from .scoring import exposure as exp_mod
from .scoring import faces as faces_mod

EYE_CLOSED_THRESHOLD = 0.18  # EAR below this is treated as "eyes closed"


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
        db_path.with_suffix(db_path.suffix + ".hdr").resolve(),
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


def source_rel_paths(data: dict[str, Any]) -> set[str]:
    """Rel_paths of the on-disk source JPEGs a db was built from: standalone
    photos plus every bracket member. A merged HDR result is not itself a
    source file, so its synthetic rel_path is excluded."""
    out: set[str] = set()
    for p in data.get("photos", []):
        if p.get("type") == "hdr":
            out.update(p.get("members", []))
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


def apply_scene_suggestions(items: list[dict[str, Any]]) -> None:
    """Mutates items in place to set 'auto_suggestion' based on per-scene normalized scores."""
    n = len(items)
    if n == 0:
        return
    if n == 1:
        items[0]["auto_suggestion"] = "review"
        return

    blurs = np.array([p["scores"]["blur"] for p in items])
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


def _score_one(
    scene: str,
    abs_path: Path,
    rel_path: str,
    members: list[str] | None,
    face_detect,
    existing: dict[str, tuple[Any, Any]],
    embedding_sink: list[np.ndarray],
) -> dict[str, Any]:
    """Score one entity. `abs_path` is the file actually read (a source JPEG,
    or a merged HDR result); `rel_path` is its identity in the db. `members`
    is the bracket's source frames when this entity is a merged HDR result."""
    blur_v = blur_mod.blur_score(str(abs_path))
    bright_v = exp_mod.brightness(str(abs_path))
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
    prior_decision, prior_decided_at = existing.get(rel_path, (None, None))
    photo: dict[str, Any] = {
        "rel_path": rel_path,
        "scene": scene,
        "width": width,
        "height": height,
        "faces": face_list,
        "scores": {
            "blur": blur_v,
            "brightness": bright_v,
            "eye_open": eye_v,
            "blur_pct": None,
            "exposure_zscore": None,
            "badness": None,
        },
        "auto_suggestion": None,
        "decision": prior_decision,
        "decided_at": prior_decided_at,
    }
    if members is not None:
        photo["type"] = "hdr"
        photo["members"] = members
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
) -> None:
    """Scan, merge HDR brackets, then score the standalone frames plus the
    merged results. An HDR bracket is scored once, as its merged output.

    `bracket_groups`, when given, is an explicit grouping (from the HDR-tab
    edit); otherwise a prior grouping is preserved across re-scores, and a
    first run auto-detects brackets from EXIF.

    If `progress_cb` is given, it's called as `cb(idx, total, current_rel_path)`
    before each photo, and once more with `idx == total` and current=None at the
    end. Otherwise a click progress bar prints to stdout."""
    verbose = progress_cb is None
    jpeg_root = photo_dir / jpeg_subdir
    assert jpeg_root.is_dir(), f"not a directory: {jpeg_root}"

    collected = _collect_jpegs(jpeg_root, excluded_scan_dirs(db_path, project_dir))
    scene_of = {str(p.relative_to(jpeg_root)): scene for scene, p in collected}
    src_rels = set(scene_of)
    if verbose:
        click.echo(f"Found {len(src_rels)} JPEGs under {jpeg_root}")

    existing = _load_existing(db_path)
    existing_dec = _existing_decisions(existing)
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

    # ----- assemble what gets scored: standalone frames + merged results -----
    targets: list[tuple[str, Path, str, list[str] | None]] = []
    for scene, jpg in collected:
        rel = str(jpg.relative_to(jpeg_root))
        if rel not in bracketed:
            targets.append((scene, jpg, rel, None))
    for b in brackets_meta:
        targets.append((
            b["scene"], hdr_root / hdr.merged_filename(b["members"]),
            b["merged"], b["members"],
        ))
    if limit is not None:
        targets = targets[:limit]

    if with_faces:
        from .scoring import eyes as eyes_mod
        if verbose:
            click.echo("Face mesh + eye-open detection enabled (mediapipe)")
        face_detect = eyes_mod.detect
    else:
        if verbose:
            click.echo("Face detection + embedding enabled (insightface buffalo_l)")
        face_detect = faces_mod.detect

    scored: list[dict[str, Any]] = []
    embedding_sink: list[np.ndarray] = []
    if verbose:
        with click.progressbar(targets, label="Scoring", show_pos=True) as bar:
            for scene, abs_path, rel, members in bar:
                scored.append(_score_one(
                    scene, abs_path, rel, members, face_detect, existing_dec, embedding_sink))
    else:
        for i, (scene, abs_path, rel, members) in enumerate(targets):
            progress_cb(i, len(targets), rel)
            scored.append(_score_one(
                scene, abs_path, rel, members, face_detect, existing_dec, embedding_sink))
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

    data = db.init_db(photo_dir, jpeg_subdir)
    data["scored_at"] = datetime.now().isoformat()
    data["clustered_at"] = None
    data["people"] = []
    data["brackets"] = brackets_meta
    data["hdr_look"] = look
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
