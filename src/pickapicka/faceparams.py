"""Which portrait settings a face gets: the panel's own, or the face's.

The portrait panel holds one set of sliders for every face found, and may
hold, under "faces", settings of a face's own: each entry is where that face
was ("at", the centre of its box, as fractions of the frame) and its slider
values. A face with an entry takes the entry's values instead of the panel's,
all of them, so a face can be left out of a slimming the others get by its
own zero, and the panel can be changed afterwards without moving it.

Faces are found again on every analysis (a new lens correction moves them,
another detector run can list them in a different order), so an entry is
tied to a face by place, not by number: the face whose centre lies within
half its own width of the entry's, the nearest if two do. An entry that no
face is near any more is kept and does nothing, so it comes back if the face
does.
"""
from __future__ import annotations

import math
from typing import Any

FACES_MAX = 32             # entries per photo
MATCH = 0.5                # how far a face may have moved, in its own widths


def _centre(face: dict[str, Any]) -> tuple[float, float]:
    x, y, w, h = face["box"]
    return x + w / 2.0, y + h / 2.0


def entry_for(params: dict[str, Any] | None, face: dict[str, Any],
              frame_w: float, frame_h: float) -> dict[str, Any] | None:
    """The entry of `params["faces"]` that belongs to `face`, or None."""
    entries = (params or {}).get("faces") or []
    if not entries:
        return None
    cx, cy = _centre(face)
    reach = MATCH * face["box"][2] * frame_w
    best, best_d = None, math.inf
    for e in entries:
        d = math.hypot((e["at"][0] - cx) * frame_w, (e["at"][1] - cy) * frame_h)
        if d <= reach and d < best_d:
            best, best_d = e, d
    return best


def for_face(params: dict[str, Any] | None, face: dict[str, Any],
             frame_w: float, frame_h: float) -> dict[str, Any]:
    """The slider values `face` gets: its own entry's, or the panel's."""
    own = entry_for(params, face, frame_w, frame_h)
    src = own if own is not None else (params or {})
    return {k: v for k, v in src.items() if k not in ("faces", "at")}


def every_set(params: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The panel's values and every face's own, for a question about any of
    them (is anything set, how far could anything reach)."""
    if not params:
        return []
    base = {k: v for k, v in params.items() if k != "faces"}
    return [base] + [{k: v for k, v in e.items() if k != "at"} for e in params.get("faces") or []]


def anything(params: dict[str, Any] | None, keys) -> bool:
    return any(s.get(k) for s in every_set(params) for k in keys)
