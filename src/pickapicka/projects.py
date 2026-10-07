"""Where a project folder lives, moving it between the two places, and merging.

A project folder (picks.json + .cache/) lives in one of two places:

- a workspace: `<workspace>/<name>/`, apart from the photos. The original layout.
- with its photos: `<photo folder>/.pickapicka/<name>/`. Like `.git`, the work
  travels with the photos: moving or copying the photo folder takes the project
  along, so it never needs re-linking. Scanning skips every `.pickapicka`
  folder, so its thumbnails are never scored as photos.

Which one new projects get is a preference (`userstate.get_project_location`).
Changing it offers to move the existing projects across with `move_project`.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from . import db, hdr, healing, raw

PHOTOS_SUBDIR = ".pickapicka"
# Where a project keeps its AI fill patches (see `aifill`): files the edits
# name, so they move with the folder and are carried when projects merge.
FILLS_DIR = "fills"

LOCATIONS = ("workspace", "photos")


def is_with_photos(project_dir: Path) -> bool:
    """A project kept inside its photo folder's `.pickapicka/`."""
    return project_dir.parent.name == PHOTOS_SUBDIR


def photo_root_of(project_dir: Path, data: dict[str, Any]) -> Path:
    """The photo folder a project describes. One kept with its photos is
    inside that folder, so its own location says where the photos are, and
    the photo_root it stored goes stale the moment the folder moves."""
    if is_with_photos(project_dir):
        return project_dir.parent.parent
    return Path(data["photo_root"]).expanduser()


def heal_photo_root(project_dir: Path) -> bool:
    """Rewrite a with-photos project's stored photo_root to where it is now.
    True when it had moved. A workspace project is left alone: its photos
    moving is what re-linking is for."""
    if not is_with_photos(project_dir):
        return False
    db_path = project_dir / "picks.json"
    data = db.load(db_path)
    now = str(photo_root_of(project_dir, data).resolve())
    if data.get("photo_root") == now:
        return False
    data["photo_root"] = now
    db.save(db_path, data)
    return True


def target_dir(name: str, photo_root: Path, location: str, workspace: Path) -> Path:
    """Where a project named `name` goes for `location`."""
    assert location in LOCATIONS, f"unknown project location: {location!r}"
    if location == "photos":
        return photo_root / PHOTOS_SUBDIR / name
    return workspace / name


def location_of(project_dir: Path) -> str:
    return "photos" if is_with_photos(project_dir) else "workspace"


def move_project(src: Path, dst: Path) -> Path:
    """Move a project folder to `dst`, which must not exist yet. Across drives
    shutil.move copies then deletes, so the source is only gone once the copy
    is complete; a copy that fails part-way is removed again, or it would be
    listed as a second, broken project. Returns `dst`."""
    assert (src / "picks.json").is_file(), f"not a project folder: {src}"
    assert not dst.exists(), f"already exists: {dst}"
    made_parent = not dst.parent.exists()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except OSError:
        if (src / "picks.json").is_file():
            shutil.rmtree(dst)
            if made_parent and not any(dst.parent.iterdir()):
                dst.parent.rmdir()
        raise
    # A `.pickapicka/` left with nothing in it is just clutter in a photo folder.
    if is_with_photos(src) and not any(src.parent.iterdir()):
        src.parent.rmdir()
    if is_with_photos(dst):
        heal_photo_root(dst)
    return dst


# ----- merging ------------------------------------------------------------
# Merged projects become one new project over the folder the sources' photo
# folders have in common. What the photographer set (decisions, stars, labels,
# edits) is carried across, keyed by each photo's path under that folder;
# everything measured (scores, faces, scenes, brackets) is measured again by
# the first scoring, which keeps carried marks the way any re-score does.

# Project settings carried only when every source has the same one.
SHARED_SETTINGS = ("hdr_look", "cluster_settings")
DEFAULT_GAP_MINUTES = 30


class MergeError(ValueError):
    """Projects that cannot be merged, with why, for the person merging."""


def _left_out(data: dict[str, Any], r: Path, root: Path) -> tuple[set[str], set[str]]:
    """A source's left-out folders, split into those the merged project can
    leave out too (a top-level folder of `root`) and those it cannot."""
    sub, raw_sub = data.get("jpeg_subdir") or "", data.get("raw_subdir") or ""
    jroot = r / sub if sub else r
    rroot = r / raw_sub if raw_sub else jroot
    top: set[str] = set()
    deeper: set[str] = set()
    for name in data.get("ignored_dirs") or []:
        for base in {jroot, rroot}:
            rel = (base / name).relative_to(root)
            (top if len(rel.parts) == 1 else deeper).add(str(rel))
    return top, deeper


def _seed_rel(data: dict[str, Any], r: Path, root: Path, p: dict[str, Any]) -> str:
    """Where a source photo is keyed in the merged project: its path under
    `root`, or for a merged HDR result the name its bracket merges to there,
    so its marks carry when the bracket is detected again."""
    sub = data.get("jpeg_subdir") or ""
    jroot = r / sub if sub else r
    t = p.get("type")
    if t == "hdr":
        return hdr.merged_rel_path([str((jroot / m).relative_to(root)) for m in p["members"]])
    if t == "raw":
        return str((r / p.get("src", p["rel_path"])).relative_to(root))
    return str((jroot / p["rel_path"]).relative_to(root))


def _stamp(m: dict[str, Any]) -> str:
    return max(m.get("decided_at") or "", m.get("edited_at") or "")


def merge_plan(project_dirs: list[Path]) -> dict[str, Any]:
    """What merging these projects makes, before anything is written.

    - root: the folder their photo folders have in common, scanned whole.
    - ignored_dirs: top-level folders of root that belong to none of them,
      plus the folders they left out that root can express.
    - marks: what was set on each photo, keyed by its path under root. Two
      projects that marked the same photo differently keep the more recent.
    - settings: per-project settings carried across (see `_settings`).
    - photos: how many photos the merged project will have, against
      source_photos, so a common folder that sweeps in more than the sources
      shows before it is scored.
    """
    from .scorer import EDIT_FIELDS, MARKS, pair_group, walk_files, _is_supported

    if len(set(project_dirs)) < 2:
        raise MergeError("choose at least two projects to merge")
    sources = []
    for d in project_dirs:
        data = db.load(d / "picks.json")
        r = photo_root_of(d, data)
        if not r.is_dir():
            raise MergeError(f"the photos of '{d.name}' are not found at {r}; re-link it first")
        sources.append((d, data, r.resolve()))
    roots = [r for _, _, r in sources]
    root = Path(os.path.commonpath([str(r) for r in roots]))
    if root == Path(root.anchor):
        raise MergeError(
            "these projects' photos have nothing in common below the top of the "
            f"drive ({root}), so a merged project would scan the whole drive")

    # Each source covers the top-level folder under `root` its photos are in;
    # a source whose photos are `root` itself covers all of it.
    covered = {r.relative_to(root).parts[0] for r in roots if r != root}
    ignored: set[str] = set()
    if root not in roots:
        ignored = {c.name for c in root.iterdir()
                   if c.is_dir() and c.name not in covered and c.name != PHOTOS_SUBDIR}
    still_scanned: set[str] = set()
    for _, data, r in sources:
        top, deeper = _left_out(data, r, root)
        ignored |= top
        still_scanned |= deeper
    # A folder one project left out but another is made of stays in.
    ignored -= covered
    still_scanned = {p for p in still_scanned
                     if not any(r.is_relative_to(root / p) for r in roots)}

    marks: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for _, data, r in sources:
        for p in data.get("photos", []):
            kept = {k: p[k] for k in MARKS + EDIT_FIELDS if p.get(k) not in (None, "", [], {})}
            if not kept:
                continue
            rel = _seed_rel(data, r, root, p)
            if rel in marks:
                conflicts += 1
                if _stamp(kept) < _stamp(marks[rel]):
                    continue
            marks[rel] = kept

    # Pair the merged folder's files the way its scoring will. Where a RAW now
    # wins a shot that a source scored as its JPEG (a RAW folder the source
    # did not scan), the JPEG's marks go to the RAW.
    groups: dict[tuple[str, str], tuple[list[Path], list[Path]]] = {}
    for f in walk_files(root, {root / n for n in ignored}):
        kind = 0 if _is_supported(f) else 1 if raw.is_raw(f) else None
        if kind is None:
            continue
        rel = f.relative_to(root)
        scene = rel.parts[0] if len(rel.parts) > 1 else "(none)"
        groups.setdefault((scene, f.stem.lower()), ([], []))[kind].append(f)
    photos = 0
    for jpegs, raws in groups.values():
        for raw_f, jpeg_f in pair_group(jpegs, raws):
            photos += 1
            if raw_f is None or jpeg_f is None:
                continue
            j, rw = str(jpeg_f.relative_to(root)), str(raw_f.relative_to(root))
            if j in marks and rw not in marks:
                marks[rw] = marks[j]

    luts: dict[str, Any] = {}
    for _, data, _ in sources:
        luts.update(data.get("luts") or {})
    settings, notes = _settings([data for _, data, _ in sources], whole=root in roots)
    return {
        "root": root,
        "ignored_dirs": sorted(ignored),
        "still_scanned": sorted(still_scanned),
        "marks": marks,
        "luts": luts,
        "settings": settings,
        "notes": notes,
        "conflicts": conflicts,
        "photos": photos,
        "source_photos": sum(len(data.get("photos", [])) for _, data, _ in sources),
        "names": [d.name for d, _, _ in sources],
    }


def _settings(datas: list[dict[str, Any]], *, whole: bool) -> tuple[dict[str, Any], list[str]]:
    """The merged project's settings, and a note for each that could not
    simply be carried.

    What any source detected is detected (subject classes, faces), so nothing
    a source found goes missing. A setting the sources disagree on falls back
    to its default. Scenes by folder take each scene from a top-level folder,
    which under the merged root is a whole source project, so unless one
    source already covered the merged root, scenes go by time gap instead."""
    out: dict[str, Any] = {}
    notes: list[str] = []
    for key in SHARED_SETTINGS:
        vals = [d.get(key) for d in datas]
        if all(v == vals[0] for v in vals):
            if vals[0] is not None:
                out[key] = vals[0]
        else:
            notes.append(f"{key.replace('_', ' ')} differs between the projects, so the default is used")
    classes = sorted({c for d in datas for c in d.get("subject_classes") or []})
    if classes:
        out["subject_classes"] = classes
    out["detect_faces"] = any(d.get("detect_faces") is not False for d in datas)

    groupings = [d.get("scene_grouping") or {"mode": "folder"} for d in datas]
    gaps = {g.get("gap_minutes", DEFAULT_GAP_MINUTES) for g in groupings if g.get("mode") == "time_gap"}
    gap = gaps.pop() if len(gaps) == 1 else DEFAULT_GAP_MINUTES
    if all(g.get("mode") != "time_gap" for g in groupings) and whole:
        out["scene_grouping"] = {"mode": "folder"}
    else:
        out["scene_grouping"] = {"mode": "time_gap", "gap_minutes": gap}
        if any(g.get("mode") != "time_gap" for g in groupings):
            notes.append(
                f"scenes are grouped by time gap ({gap} min): by folder, each project "
                "would become one scene. Preferences → This project changes it")
    return out, notes


def carry_fills(sources: list[Path], marks: dict[str, dict[str, Any]], dest: Path) -> int:
    """Copy into `dest`'s fills folder every AI fill patch the carried edits
    (and their slots) name, from whichever source project has it. Without
    them a carried AI heal names a file the merged project does not have.
    Returns how many were copied."""
    wanted: set[str] = set()
    for kept in marks.values():
        for edit in [kept.get("edit")] + list(kept.get("edit_slots") or []):
            if isinstance(edit, dict):
                wanted.update(healing.fill_ids(edit.get("healing")))
    if not wanted:
        return 0
    (dest / FILLS_DIR).mkdir(parents=True, exist_ok=True)
    for fid in sorted(wanted):
        src = next((d / FILLS_DIR / f"{fid}.png" for d in sources
                    if (d / FILLS_DIR / f"{fid}.png").is_file()), None)
        assert src is not None, f"AI fill {fid[:12]} is in none of the merged projects"
        shutil.copy2(src, dest / FILLS_DIR / f"{fid}.png")
    return len(wanted)


def merge_seed(plan: dict[str, Any]) -> dict[str, Any]:
    """The picks.json a merged project starts from: never scored, so the first
    open scores it, with its carried settings and each carried photo listed
    under its path relative to the common root, so the scoring's
    keep-what-was-set step finds it. No subfolders are set, since the merged
    root is scanned whole, so a RAW is keyed by its root-relative path like a
    JPEG."""
    data = db.init_db(plan["root"], "", "")
    data["ignored_dirs"] = plan["ignored_dirs"]
    data.update(plan["settings"])
    if plan["luts"]:
        data["luts"] = plan["luts"]
    for rel, kept in sorted(plan["marks"].items()):
        data["photos"].append({"rel_path": rel, **kept})
    return data


# ----- merging workspaces ---------------------------------------------------
# A workspace is a plain folder the person chose. The app keeps nothing in it
# but project folders, so merging one workspace into another is moving those
# folders across, soft-deleted ones too, so restoring them by renaming still
# works. The old folder goes only once nothing else is left in it.

# What the OS leaves in any folder it has shown; not the person's files.
OS_LITTER = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def workspace_merge_plan(source: Path, target: Path, open_dir: Path | None) -> dict[str, Any]:
    """Each project folder in `source`, with where it goes in `target`, and
    the other things in `source` that are not the app's.

    A name already taken in `target` gets the source workspace's name added
    rather than overwriting anything. Names compare without case, since macOS
    and Windows do."""
    from .userstate import is_deleted_project

    assert source != target, "a workspace cannot be merged into itself"
    taken = {c.name.casefold() for c in target.iterdir()} if target.is_dir() else set()
    moves: list[dict[str, Any]] = []
    others: list[str] = []
    for c in sorted(source.iterdir(), key=lambda c: c.name.lower()):
        if not (c / "picks.json").is_file():
            if c.name not in OS_LITTER:
                others.append(c.name)
            continue
        name, n = c.name, 1
        while name.casefold() in taken:
            name = f"{c.name} ({source.name})" if n == 1 else f"{c.name} ({source.name} {n})"
            n += 1
        taken.add(name.casefold())
        moves.append({"src": c, "dst": target / name, "deleted": is_deleted_project(c.name),
                      "open": open_dir is not None and open_dir == c.resolve()})
    return {"moves": moves, "others": others}


def remove_if_empty(folder: Path) -> bool:
    """Delete `folder` when the OS's own litter is all that is in it. True
    when it went."""
    left = list(folder.iterdir())
    if any(c.name not in OS_LITTER or not c.is_file() for c in left):
        return False
    for c in left:
        c.unlink()
    folder.rmdir()
    return True
