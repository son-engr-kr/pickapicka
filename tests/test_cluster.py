"""Checks for clustering settings and the subject pass.

    uv run python tests/test_cluster.py

Grouping is two passes over the same db — faces into people, subjects into
look-alike groups — configured together and stored on the project. These pin the
schema and the parts that are easy to get wrong without noticing: a setting that
is read but never used, and a switched-off pass that leaves its results behind.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pickapicka import cluster, db


# ----- settings -----------------------------------------------------------

def test_defaults_cover_both_passes() -> None:
    d = cluster.DEFAULT_SETTINGS
    assert set(d) == {"face_eps", "face_min_samples", "group_subjects",
                      "subject_eps", "subject_min_samples", "subject_min_area",
                      "subject_min_score"}
    # The face defaults must stay the module's own, or a project that never
    # opened the panel would silently cluster differently.
    assert d["face_eps"] == cluster.DEFAULT_EPS
    assert d["face_min_samples"] == cluster.DEFAULT_MIN_SAMPLES
    assert d["subject_eps"] == cluster.VEHICLE_EPS
    assert d["subject_min_samples"] == cluster.VEHICLE_MIN_SAMPLES
    assert d["subject_min_area"] == cluster.VEHICLE_MIN_AREA
    assert d["subject_min_score"] == cluster.VEHICLE_MIN_SCORE


def test_nothing_gives_the_defaults() -> None:
    for raw in (None, {}, "nonsense", 42, []):
        assert cluster.normalize_settings(raw) == cluster.DEFAULT_SETTINGS, raw


def test_values_are_clamped_not_trusted() -> None:
    """A distance of 9 would put every face in one person and a distance of 0
    would put each in its own, so the range is bounded."""
    s = cluster.normalize_settings({"face_eps": 99, "subject_eps": -5,
                                    "face_min_samples": 0,
                                    "subject_min_area": 5.0,
                                    "subject_min_score": 12})
    assert s["face_eps"] == 1.20
    assert s["subject_eps"] == 0.05
    assert s["face_min_samples"] == 1
    assert s["subject_min_area"] == 0.50
    assert s["subject_min_score"] == 1.0


def test_sample_counts_stay_whole() -> None:
    s = cluster.normalize_settings({"face_min_samples": 3.7,
                                    "subject_min_samples": 2.2})
    assert s["face_min_samples"] == 4 and isinstance(s["face_min_samples"], int)
    assert s["subject_min_samples"] == 2 and isinstance(s["subject_min_samples"], int)


def test_garbage_values_fall_back_per_field() -> None:
    s = cluster.normalize_settings({"face_eps": "abc", "subject_eps": 0.4})
    assert s["face_eps"] == cluster.DEFAULT_SETTINGS["face_eps"]
    assert s["subject_eps"] == 0.4, "one bad field must not discard a good one"


def test_the_subject_pass_can_be_switched_off() -> None:
    assert cluster.normalize_settings({"group_subjects": False})["group_subjects"] is False
    assert cluster.normalize_settings({"group_subjects": 0})["group_subjects"] is False
    assert cluster.normalize_settings({"group_subjects": 1})["group_subjects"] is True


def test_a_partial_patch_keeps_the_rest() -> None:
    """The panel sends only what it knows about; anything else must survive."""
    stored = cluster.normalize_settings({"face_eps": 0.7, "subject_eps": 0.4})
    patched = cluster.normalize_settings({**stored, "subject_eps": 0.2})
    assert patched["face_eps"] == 0.7 and patched["subject_eps"] == 0.2


# ----- clearing the subject groups ----------------------------------------

def _db(tmp_path: Path, vehicles: int = 2) -> Path:
    """A minimal db with subject groups already assigned."""
    path = tmp_path / "picks.json"
    photos = []
    for i in range(3):
        photos.append({
            "rel_path": f"a_{i}.jpg", "scene": "s", "width": 400, "height": 300,
            "faces": [],
            "objects": [{"cls": "car", "score": 0.9, "bbox_xywh": [10, 10, 200, 150],
                         "vehicle_id": f"v{i % max(1, vehicles)}"}],
        })
    path.write_text(json.dumps({
        "version": db.DB_VERSION,
        "photo_root": str(tmp_path), "jpeg_subdir": "", "raw_subdir": "",
        "scored_at": "2026-08-03T00:00:00", "clustered_at": None,
        "people": [], "brackets": [], "subject_classes": ["car"],
        "vehicles": [{"id": f"v{i}", "label": f"Subject {i + 1}", "count": 1}
                     for i in range(vehicles)],
        "photos": photos,
    }), encoding="utf-8")
    return path


def test_clearing_removes_the_groups_and_the_ids(tmp_path) -> None:
    """Switching the subject pass off has to take its results with it — the face
    pass never touches `vehicles`, so nothing else would ever clean them up."""
    path = _db(tmp_path, vehicles=2)
    before = json.loads(path.read_text(encoding="utf-8"))
    assert before["vehicles"] and any(
        o["vehicle_id"] for p in before["photos"] for o in p["objects"])

    cluster.clear_vehicle_groups(path)

    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["vehicles"] == []
    assert all(o["vehicle_id"] is None
               for p in after["photos"] for o in p["objects"])


def test_clearing_leaves_everything_else_alone(tmp_path) -> None:
    path = _db(tmp_path)
    before = json.loads(path.read_text(encoding="utf-8"))
    cluster.clear_vehicle_groups(path)
    after = json.loads(path.read_text(encoding="utf-8"))
    for key in ("photo_root", "jpeg_subdir", "scored_at", "subject_classes"):
        assert after[key] == before[key], key
    assert len(after["photos"]) == len(before["photos"])
    # The detections themselves are scoring output and must survive.
    assert after["photos"][0]["objects"][0]["cls"] == "car"
    assert after["photos"][0]["objects"][0]["score"] == 0.9


def test_clearing_is_safe_with_nothing_to_clear(tmp_path) -> None:
    path = _db(tmp_path, vehicles=1)
    cluster.clear_vehicle_groups(path)
    cluster.clear_vehicle_groups(path)      # again, on already-empty groups
    after = json.loads(path.read_text(encoding="utf-8"))
    assert after["vehicles"] == []


# ----- the subject pass honours its settings ------------------------------

def test_min_area_and_score_are_used_not_just_accepted(tmp_path) -> None:
    """These used to be module constants read inside the loop. If a caller's
    values were ignored, the panel's controls would do nothing at all."""
    path = _db(tmp_path, vehicles=1)
    data = json.loads(path.read_text(encoding="utf-8"))
    # One big confident car, one tiny one, one low-confidence one.
    data["photos"][0]["objects"] = [
        {"cls": "car", "score": 0.95, "bbox_xywh": [0, 0, 300, 200], "vehicle_id": None}]
    data["photos"][1]["objects"] = [
        {"cls": "car", "score": 0.95, "bbox_xywh": [0, 0, 10, 8], "vehicle_id": None}]
    data["photos"][2]["objects"] = [
        {"cls": "car", "score": 0.10, "bbox_xywh": [0, 0, 300, 200], "vehicle_id": None}]
    path.write_text(json.dumps(data), encoding="utf-8")
    # Real files, since a photo that cannot be read is an error, not a skip.
    rng = np.random.default_rng(0)
    for photo in data["photos"]:
        Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8)).save(
            tmp_path / photo["rel_path"])

    seen: list[tuple[int, int]] = []

    def resolve(photo):
        return tmp_path / photo["rel_path"]

    def progress(phase, idx, total):
        if phase == "describing subjects":
            seen.append((idx, total))

    # The *count* of candidates is the filter's decision, which is what is
    # under test; whether any groups come out of three noise images is not.
    cluster.run_vehicle_clustering(path, resolve, min_area=0.02, min_score=0.5,
                                   progress_cb=progress)
    strict = seen[-1][1] if seen else 0
    seen.clear()
    cluster.run_vehicle_clustering(path, resolve, min_area=0.0, min_score=0.0,
                                   progress_cb=progress)
    loose = seen[-1][1] if seen else 0
    assert strict == 1, f"expected only the big confident car, got {strict}"
    assert loose == 3, f"expected all three once the filters are open, got {loose}"


def _main() -> None:
    import inspect
    import tempfile
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        if "tmp_path" in inspect.signature(fn).parameters:
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
        else:
            fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} checks passed.")


if __name__ == "__main__":
    _main()
