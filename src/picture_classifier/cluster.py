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
        f"embeddings sidecar not found at {emb_path}; run `pcls score` first"
    )
    embeddings = np.load(emb_path)
    n = embeddings.shape[0]

    # Build a parallel list of (photo_idx, face_idx) rows so we can map cluster
    # labels back to faces in the db.
    row_to_face: list[tuple[int, int]] = []
    for pi, photo in enumerate(photos):
        for fi, face in enumerate(photo.get("faces", [])):
            ei = face.get("embedding_idx")
            if ei is None or ei >= n:
                continue
            assert ei == len(row_to_face), (
                f"embedding_idx mismatch at photo {pi} face {fi}: "
                f"got {ei}, expected {len(row_to_face)}"
            )
            row_to_face.append((pi, fi))
            face["person_id"] = None  # reset

    if progress_cb:
        progress_cb("normalizing", 0, n)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)

    if progress_cb:
        progress_cb("clustering", 0, n)
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

VEHICLE_EPS = 0.28          # cosine distance; appearance space is tighter than ArcFace
VEHICLE_MIN_SAMPLES = 2     # a car that shows up in two frames is worth grouping
VEHICLE_MIN_AREA = 0.02     # ignore background traffic: <2% of the frame
VEHICLE_MIN_SCORE = 0.5


def run_vehicle_clustering(
    db_path: Path,
    resolve_path: Callable[[dict[str, Any]], Path],
    eps: float = VEHICLE_EPS,
    min_samples: int = VEHICLE_MIN_SAMPLES,
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
            if area < VEHICLE_MIN_AREA or obj["score"] < VEHICLE_MIN_SCORE:
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
