"""Cluster face embeddings into person groups using DBSCAN with cosine distance.

Reads embeddings from `<db_path>.embeddings.npy`, runs DBSCAN, and writes
`people: [...]` and `face.person_id` back into the JSON db.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

from . import db
from .scoring import appearance


DEFAULT_EPS = 0.55           # cosine distance threshold for DBSCAN
DEFAULT_MIN_SAMPLES = 3      # smallest valid cluster size


def _representative_face(
    photos: list[dict[str, Any]],
    member_ids: list[tuple[int, int]],
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """Pick the face closest to the cluster centroid as the representative."""
    rows = [embeddings[photos[pi]["faces"][fi]["embedding_idx"]] for pi, fi in member_ids]
    centroid = np.mean(np.stack(rows, axis=0), axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    best_dist = float("inf")
    best = member_ids[0]
    for (pi, fi), emb in zip(member_ids, rows):
        emb_n = emb / (np.linalg.norm(emb) + 1e-9)
        d = float(1.0 - np.dot(emb_n, centroid))
        if d < best_dist:
            best_dist = d
            best = (pi, fi)
    pi, fi = best
    return {"rel_path": photos[pi]["rel_path"], "face_idx": fi}


def run_clustering(
    db_path: Path,
    eps: float = DEFAULT_EPS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    progress_cb=None,
) -> None:
    """Run DBSCAN over saved embeddings and persist `people[]` + `face.person_id`."""
    data = db.load(db_path)
    photos = data["photos"]

    emb_path = db_path.with_suffix(db_path.suffix + ".embeddings.npy")
    assert emb_path.is_file(), (
        f"embeddings sidecar not found at {emb_path}; run `pickapicka score` first"
    )
    embeddings = np.load(emb_path)
    n = embeddings.shape[0]

    # Each face with an embedding, as (photo_idx, face_idx), and its row. The
    # rows are looked up, not assumed to run 0, 1, 2… in photo order: photos
    # leave a project (a folder left out, an original gone) and take their
    # rows' place in that order with them.
    row_to_face: list[tuple[int, int]] = []
    rows: list[int] = []
    for pi, photo in enumerate(photos):
        for fi, face in enumerate(photo.get("faces", [])):
            face["person_id"] = None  # reset
            ei = face.get("embedding_idx")
            if ei is None or ei >= n:
                continue
            row_to_face.append((pi, fi))
            rows.append(ei)
    if not rows:
        # No faces to group (a project that does not detect them, or none
        # left), and DBSCAN will not take an empty array. Leave it with no
        # people rather than stale ones from before.
        data["people"] = []
        data["clustered_at"] = datetime.now().isoformat()
        db.save(db_path, data)
        return

    if progress_cb:
        progress_cb("normalizing", 0, len(rows))
    picked = embeddings[rows]
    normalized = picked / (np.linalg.norm(picked, axis=1, keepdims=True) + 1e-9)

    if progress_cb:
        progress_cb("clustering", 0, len(rows))
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(normalized)

    # Group rows by cluster label (ignore -1 noise).
    clusters: dict[int, list[tuple[int, int]]] = {}
    for row, label in enumerate(labels):
        if label < 0:
            continue
        clusters.setdefault(int(label), []).append(row_to_face[row])

    # Sort by size desc — biggest cluster becomes Person 1.
    sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))

    if progress_cb:
        progress_cb("building people", 0, len(sorted_clusters))

    people: list[dict[str, Any]] = []
    for new_idx, (_orig_label, members) in enumerate(sorted_clusters):
        pid = f"p{new_idx}"
        for pi, fi in members:
            photos[pi]["faces"][fi]["person_id"] = pid
        ref = _representative_face(photos, members, embeddings)
        people.append({
            "id": pid,
            "label": f"Person {new_idx + 1}",
            "priority": new_idx + 1,
            "excluded": False,
            "count": len(members),
            "ref": ref,
        })
        if progress_cb:
            progress_cb("building people", new_idx + 1, len(sorted_clusters))

    data["clustered_at"] = datetime.now().isoformat()
    data["people"] = people
    db.save(db_path, data)


# ----- subject (vehicle) grouping -----------------------------------------
#
# The same DBSCAN, but over appearance descriptors instead of face embeddings.
# Read scoring/appearance.py before touching the thresholds: these groups mean
# "looks like the same vehicle", which is a weaker claim than the person groups.

# Both passes in one place, because they are configured together and the
# defaults are the only thing most projects ever need. Stored per project in the
# db so a rescore does not reset them.
DEFAULT_SETTINGS: dict[str, Any] = {
    "face_eps": DEFAULT_EPS,
    "face_min_samples": DEFAULT_MIN_SAMPLES,
    # Subject grouping is appearance-based, so it is a weaker claim than the
    # person groups and worth being able to switch off.
    "group_subjects": True,
    "subject_eps": 0.28,
    "subject_min_samples": 2,
    "subject_min_area": 0.02,
    "subject_min_score": 0.5,
}

_SETTING_RANGES: dict[str, tuple[float, float]] = {
    "face_eps": (0.20, 1.20),
    "face_min_samples": (1, 20),
    "subject_eps": (0.05, 1.00),
    "subject_min_samples": (1, 20),
    "subject_min_area": (0.0, 0.50),
    "subject_min_score": (0.0, 1.0),
}
_INT_SETTINGS = frozenset({"face_min_samples", "subject_min_samples"})


def normalize_settings(raw: Any) -> dict[str, Any]:
    """Merge over the defaults and clamp. Loosening eps too far collapses
    everyone into one group and tightening it splits one person into five, so the
    range is bounded rather than left to whatever arrives."""
    out = dict(DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return out
    out["group_subjects"] = bool(raw.get("group_subjects", out["group_subjects"]))
    for key, (lo, hi) in _SETTING_RANGES.items():
        if raw.get(key) is None:
            continue
        try:
            val = float(raw[key])
        except (TypeError, ValueError):
            continue
        val = min(hi, max(lo, val))
        out[key] = int(round(val)) if key in _INT_SETTINGS else val
    return out


VEHICLE_EPS = 0.28          # cosine distance; appearance space is tighter than ArcFace
VEHICLE_MIN_SAMPLES = 2     # a car that shows up in two frames is worth grouping
VEHICLE_MIN_AREA = 0.02     # ignore background traffic: <2% of the frame
VEHICLE_MIN_SCORE = 0.5


def clear_vehicle_groups(db_path: Path) -> None:
    """Drop every subject group and the ids pointing at them.

    Switching the subject pass off has to remove what it produced: the face pass
    does not touch `vehicles`, so without this the sidebar would keep offering
    groups that nothing is going to refresh.
    """
    data = db.load(db_path)
    for photo in data["photos"]:
        for obj in photo.get("objects", []):
            obj["vehicle_id"] = None
    data["vehicles"] = []
    db.save(db_path, data)


def run_vehicle_clustering(
    db_path: Path,
    resolve_path: Callable[[dict[str, Any]], Path],
    eps: float = VEHICLE_EPS,
    min_samples: int = VEHICLE_MIN_SAMPLES,
    min_area: float = VEHICLE_MIN_AREA,
    min_score: float = VEHICLE_MIN_SCORE,
    progress_cb=None,
) -> int:
    """Group detected subjects that look like the same object. Writes
    `vehicles[]` plus `object.vehicle_id` and returns the group count.

    `resolve_path` maps a photo dict to the image file to read — pass
    `scorer.pixel_path`-style resolution so RAW and merged HDR photos land on
    the same pixels the scorer measured."""
    data = db.load(db_path)
    photos = data["photos"]

    rows: list[np.ndarray] = []
    row_to_obj: list[tuple[int, int]] = []
    candidates: list[tuple[int, int, str, list[int]]] = []
    for pi, photo in enumerate(photos):
        for oi, obj in enumerate(photo.get("objects", [])):
            obj["vehicle_id"] = None  # reset every run
            x, y, bw, bh = obj["bbox_xywh"]
            area = bw * bh / float(max(1, photo["width"] * photo["height"]))
            if area < min_area or obj["score"] < min_score:
                continue
            candidates.append((pi, oi, photo["rel_path"], obj["bbox_xywh"]))

    total = len(candidates)
    for k, (pi, oi, _rel, bbox) in enumerate(candidates):
        if progress_cb:
            progress_cb("describing subjects", k, total)
        desc = appearance.describe_box(str(resolve_path(photos[pi])), bbox)
        if desc is None:
            continue
        rows.append(desc)
        row_to_obj.append((pi, oi))

    vehicles: list[dict[str, Any]] = []
    if len(rows) >= min_samples:
        if progress_cb:
            progress_cb("grouping subjects", 0, len(rows))
        feats = np.stack(rows, axis=0)
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(feats)

        clusters: dict[int, list[int]] = {}
        for row, label in enumerate(labels):
            if label >= 0:
                clusters.setdefault(int(label), []).append(row)

        for new_idx, (_label, member_rows) in enumerate(
            sorted(clusters.items(), key=lambda kv: -len(kv[1]))
        ):
            vid = f"v{new_idx}"
            for row in member_rows:
                pi, oi = row_to_obj[row]
                photos[pi]["objects"][oi]["vehicle_id"] = vid
            # Representative: the member closest to the group centroid.
            centroid = feats[member_rows].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            best = min(member_rows, key=lambda r: 1.0 - float(
                np.dot(feats[r] / (np.linalg.norm(feats[r]) + 1e-9), centroid)))
            pi, oi = row_to_obj[best]
            vehicles.append({
                "id": vid,
                "label": f"Subject {new_idx + 1}",
                "priority": new_idx + 1,
                "excluded": False,
                "count": len(member_rows),
                "cls": photos[pi]["objects"][oi]["cls"],
                "ref": {"rel_path": photos[pi]["rel_path"], "obj_idx": oi},
            })

    data["vehicles"] = vehicles
    data["vehicles_clustered_at"] = datetime.now().isoformat()
    db.save(db_path, data)
    return len(vehicles)


# ----- adding photos without regrouping ---------------------------------------
# Photos added to an open project join the groups already there, so the names,
# order and exclusions someone gave them stay. A full grouping (the People
# panel's group) is still there to start over.

def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / (np.linalg.norm(rows, axis=-1, keepdims=True) + 1e-9)


def _next_group(groups: list[dict[str, Any]]) -> tuple[int, int]:
    """The number for the next group's id, and its priority, after those there."""
    ids = [int(g["id"][1:]) for g in groups if g["id"][1:].isdigit()]
    return (max(ids) + 1 if ids else 0), max((g.get("priority", 0) for g in groups), default=0) + 1


def add_faces(data: dict[str, Any], embeddings: np.ndarray, new: list[tuple[int, int]],
              eps: float = DEFAULT_EPS, min_samples: int = DEFAULT_MIN_SAMPLES) -> int:
    """Put new faces (photo_idx, face_idx) into people. A face within `eps` of
    a person's centroid joins the nearest; the rest are grouped with the faces
    no person took before, as DBSCAN would, and a group holding a new face
    becomes a new person after the others. Returns how many faces found a
    person. A project never grouped is left for its first grouping."""
    if data.get("clustered_at") is None or not new:
        return 0
    photos, people = data["photos"], data.setdefault("people", [])
    is_new = set(new)
    members: dict[str, list[int]] = {}
    loose: list[tuple[int, int]] = []
    for pi, p in enumerate(photos):
        for fi, f in enumerate(p.get("faces") or []):
            if f.get("embedding_idx") is None or (pi, fi) in is_new:
                continue
            if f.get("person_id"):
                members.setdefault(f["person_id"], []).append(f["embedding_idx"])
            else:
                loose.append((pi, fi))
    ids = [g["id"] for g in people if g["id"] in members]
    centroids = _unit(np.stack([_unit(embeddings[members[i]]).mean(axis=0) for i in ids])) if ids else None

    joined = 0
    left: list[tuple[int, int]] = []
    for pi, fi in new:
        face = photos[pi]["faces"][fi]
        if centroids is not None:
            d = 1.0 - centroids @ _unit(embeddings[face["embedding_idx"]])
            k = int(np.argmin(d))
            if d[k] <= eps:
                face["person_id"] = ids[k]
                joined += 1
                continue
        face["person_id"] = None
        left.append((pi, fi))

    pool = left + loose
    if len(pool) >= min_samples:
        rows = _unit(embeddings[[photos[pi]["faces"][fi]["embedding_idx"] for pi, fi in pool]])
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(rows)
        fresh = set(left)
        groups: dict[int, list[tuple[int, int]]] = {}
        for ref, label in zip(pool, labels):
            if label >= 0:
                groups.setdefault(int(label), []).append(ref)
        num, priority = _next_group(people)
        for refs in sorted(groups.values(), key=len, reverse=True):
            if not fresh.intersection(refs):
                continue        # only ever loose before: not this add's to change
            pid = f"p{num}"
            for pi, fi in refs:
                photos[pi]["faces"][fi]["person_id"] = pid
            people.append({"id": pid, "label": f"Person {priority}", "priority": priority,
                           "excluded": False, "count": 0,
                           "ref": _representative_face(photos, refs, embeddings)})
            joined += len(fresh.intersection(refs))
            num, priority = num + 1, priority + 1
    _recount(people, photos, "faces", "person_id")
    return joined


def _recount(groups: list[dict[str, Any]], photos: list[dict[str, Any]], items: str, field: str) -> None:
    counts: dict[str, int] = {}
    for p in photos:
        for it in p.get(items) or []:
            if it.get(field):
                counts[it[field]] = counts.get(it[field], 0) + 1
    for g in groups:
        g["count"] = counts.get(g["id"], 0)


# A group's look is averaged from at most this many of its members, each read
# from its photo: enough to place a new subject, without re-reading a shoot.
SUBJECT_SAMPLE = 8


def add_subjects(data: dict[str, Any], resolve_path: Callable[[dict[str, Any]], Path],
                 new: list[tuple[int, int]], settings: dict[str, Any]) -> int:
    """Put new subjects (photo_idx, object_idx) into subject groups: the
    nearest within `subject_eps` of a group's look, else grouped among the
    new ones, as `run_vehicle_clustering` would, into new groups after the
    others. Returns how many found a group."""
    if not settings["group_subjects"] or data.get("vehicles_clustered_at") is None or not new:
        return 0
    photos, groups = data["photos"], data.setdefault("vehicles", [])
    eps = settings["subject_eps"]

    def candidate(pi: int, oi: int) -> bool:
        p, o = photos[pi], photos[pi]["objects"][oi]
        x, y, bw, bh = o["bbox_xywh"]
        return (bw * bh / float(max(1, p["width"] * p["height"])) >= settings["subject_min_area"]
                and o["score"] >= settings["subject_min_score"])

    def describe(pi: int, oi: int) -> np.ndarray | None:
        return appearance.describe_box(str(resolve_path(photos[pi])), photos[pi]["objects"][oi]["bbox_xywh"])

    is_new = set(new)
    by_group: dict[str, list[tuple[int, int]]] = {}
    for pi, p in enumerate(photos):
        for oi, o in enumerate(p.get("objects") or []):
            if o.get("vehicle_id") and (pi, oi) not in is_new:
                by_group.setdefault(o["vehicle_id"], []).append((pi, oi))
    ids, looks = [], []
    for g in groups:
        refs = by_group.get(g["id"], [])
        step = max(1, len(refs) // SUBJECT_SAMPLE)
        descs = [d for d in (describe(*r) for r in refs[::step][:SUBJECT_SAMPLE]) if d is not None]
        if descs:
            ids.append(g["id"])
            looks.append(_unit(np.stack(descs)).mean(axis=0))
    centroids = _unit(np.stack(looks)) if looks else None

    joined = 0
    left: list[tuple[tuple[int, int], np.ndarray]] = []
    for pi, oi in new:
        photos[pi]["objects"][oi]["vehicle_id"] = None
        if not candidate(pi, oi):
            continue
        desc = describe(pi, oi)
        if desc is None:
            continue
        if centroids is not None:
            d = 1.0 - centroids @ _unit(desc)
            k = int(np.argmin(d))
            if d[k] <= eps:
                photos[pi]["objects"][oi]["vehicle_id"] = ids[k]
                joined += 1
                continue
        left.append(((pi, oi), desc))

    if len(left) >= settings["subject_min_samples"]:
        labels = DBSCAN(eps=eps, min_samples=settings["subject_min_samples"], metric="cosine").fit_predict(
            np.stack([d for _, d in left]))
        clusters: dict[int, list[int]] = {}
        for row, label in enumerate(labels):
            if label >= 0:
                clusters.setdefault(int(label), []).append(row)
        num, priority = _next_group(groups)
        for rows in sorted(clusters.values(), key=len, reverse=True):
            vid = f"v{num}"
            for r in rows:
                pi, oi = left[r][0]
                photos[pi]["objects"][oi]["vehicle_id"] = vid
            pi, oi = left[rows[0]][0]
            groups.append({"id": vid, "label": f"Subject {priority}", "priority": priority,
                           "excluded": False, "count": 0, "cls": photos[pi]["objects"][oi]["cls"],
                           "ref": {"rel_path": photos[pi]["rel_path"], "obj_idx": oi}})
            joined += len(rows)
            num, priority = num + 1, priority + 1
    _recount(groups, photos, "objects", "vehicle_id")
    return joined
