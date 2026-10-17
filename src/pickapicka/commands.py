"""The commands an assistant works a shoot with, from a terminal.

Every one prints a single JSON object on stdout (indented when a person is at
the terminal), and a failure prints one on stderr, `{"error", "status"}`, and
exits 1. They reach the project through `link`: the open app when it has the
project open, so the window shows each change as it is made, else an app built
in this process. Each change is recorded, so `pickapicka undo` can put back
what the last one replaced.
"""
from __future__ import annotations

import functools
import io
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, get_args
from urllib.parse import quote

import click

from . import editing, exporting, fsutil, metadata, userstate
from .link import Link, LinkError, connect, find_running

DECISIONS = ("pick", "review", "reject")
LABELS = ("red", "yellow", "green", "blue", "purple")
UNDO_FILE = "cli-undo.json"
UNDO_KEEP = 20            # changes remembered per project
VIEW_EDGE = 1600          # what `view` sends back: enough to judge, small to read
# A contact sheet as wide as a vision model takes an image in without scaling
# it down: Claude's is 1568 px on the long edge. Four across gives each photo
# about 390 px, enough to tell a closed eye or a wrong framing apart.
SHEET_WIDTH = 1568
SHEET_LABEL_H = 30
DECISION_COLOURS = {"pick": (64, 192, 96), "review": (224, 176, 48), "reject": (224, 72, 64)}


# ----- output -------------------------------------------------------------

def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2 if sys.stdout.isatty() else None)


def emit(obj: Any) -> None:
    click.echo(_dump(obj))


def agent_command(fn: Callable[..., Any]) -> Callable[..., Any]:
    """A route's refusal is the command's answer, not a crash: printed as JSON
    for whoever ran it to read. Anything else still raises with its trace."""
    @functools.wraps(fn)
    def run(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except LinkError as exc:
            click.echo(json.dumps({"error": exc.detail, "status": exc.status},
                                  ensure_ascii=False), err=True)
            sys.exit(1)
    return run


project_option = click.option(
    "--project", "-p", default=None,
    help="A project's name (as `pickapicka projects` lists it), its folder, or the photo "
         "folder it is kept in. Default: the project open in the app.")


def _rels(rel_paths: tuple[str, ...]) -> list[str]:
    """The photos named, with `-` read as one per line from stdin."""
    out: list[str] = []
    for r in rel_paths:
        if r == "-":
            out += [line.strip() for line in sys.stdin if line.strip()]
        else:
            out.append(r)
    if not out:
        raise click.UsageError("name at least one photo (its rel_path), or - to read them from stdin")
    return out


def _json_arg(text: str) -> Any:
    """JSON given inline, or @path for a file of it."""
    raw = fsutil.read_text(Path(text[1:]).expanduser()) if text.startswith("@") else text
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"not JSON: {exc}") from exc


def _out_dir(out: Path | None) -> Path:
    d = out if out is not None else Path(tempfile.mkdtemp(prefix="pickapicka-"))
    d.mkdir(parents=True, exist_ok=True)
    return d.resolve()


def _file_name(rel: str, suffix: str = ".jpg") -> str:
    return rel.replace("/", "__").replace("\\", "__").rsplit(".", 1)[0] + suffix


# ----- photos ---------------------------------------------------------------

def _db(link: Link) -> dict[str, Any]:
    return link.get("/api/db")


def _scene_order(names: list[str]) -> list[str]:
    """As the sidebar lists them: by name, numbers as numbers, and the photos
    in no scene of their own, "(none)", last."""
    import re

    def key(name: str) -> tuple[bool, list[Any]]:
        parts = re.split(r"(\d+)", name.casefold())
        return name.startswith("("), [int(p) if p.isdigit() else p for p in parts]
    return sorted(names, key=key)


def _shot(exif: dict[str, Any]) -> str | None:
    words = [exif.get(k) for k in ("focal", "aperture", "shutter", "iso_text")]
    return " ".join(w for w in words if w) or None


def summary(p: dict[str, Any]) -> dict[str, Any]:
    """A photo in the few fields that decide a cull. The whole record is
    `pickapicka photo REL`."""
    s = p.get("scores") or {}
    faces = p.get("faces") or []
    slots = [e for e in p.get("edit_slots") or [] if e is not None]

    def r3(v: Any) -> Any:
        return None if v is None else round(float(v), 3)
    return {
        "rel_path": p["rel_path"],
        "scene": p.get("scene"),
        "captured_at": p.get("captured_at"),
        "type": p.get("type"),
        "decision": p.get("decision"),
        "suggested": p.get("auto_suggestion"),
        "rating": p.get("rating", 0),
        "label": p.get("label"),
        "edited": bool(p.get("edit")),
        "versions": len(slots),
        "scores": {"badness": r3(s.get("badness")), "blur_pct": r3(s.get("blur_pct")),
                   "exposure_zscore": r3(s.get("exposure_zscore")),
                   "eye_open": r3(s.get("eye_open"))},
        "faces": len(faces),
        "people": sorted({f["person_id"] for f in faces if f.get("person_id")}),
        "subjects": sorted({o["cls"] for o in p.get("objects") or []}),
        "shot": _shot(p.get("exif") or {}),
    }


def _capture_key(p: dict[str, Any]) -> tuple[bool, str, str]:
    return p.get("captured_at") is None, p.get("captured_at") or "", p["rel_path"]


def photo_filters(fn: Callable[..., Any]) -> Callable[..., Any]:
    """The options that choose photos, shared by `photos` and `sheet`."""
    opts = [
        click.option("--scene", "scenes", multiple=True,
                     help="Only this scene (repeatable). `pickapicka scenes` lists them."),
        click.option("--decision", "decisions", multiple=True,
                     type=click.Choice(DECISIONS + ("none",)),
                     help="Only photos with this decision (repeatable); none is undecided."),
        click.option("--suggested", type=click.Choice(DECISIONS),
                     help="Only photos the scorer suggested this for."),
        click.option("--edited/--unedited", default=None, help="Only edited, or only unedited, photos."),
        click.option("--rating", "min_rating", type=click.IntRange(0, 5), default=None,
                     help="Only photos with at least this many stars."),
        click.option("--label", type=click.Choice(LABELS), help="Only photos with this colour label."),
        click.option("--person", help="Only photos with this person in them (an id from `pickapicka people`)."),
        click.option("--sort", type=click.Choice(["time", "badness"]), default="time", show_default=True,
                     help="time: as shot. badness: best first, as the scorer ranks them."),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def select(photos: list[dict[str, Any]], *, scenes: tuple[str, ...], decisions: tuple[str, ...],
           suggested: str | None, edited: bool | None, min_rating: int | None,
           label: str | None, person: str | None, sort: str) -> list[dict[str, Any]]:
    known = {p.get("scene") for p in photos}
    for s in scenes:
        if s not in known:
            raise LinkError(404, {"message": f"no scene '{s}'", "scenes": _scene_order(
                [n for n in known if n is not None])})
    wanted = {None if d == "none" else d for d in decisions}
    out = [p for p in photos
           if (not scenes or p.get("scene") in scenes)
           and (not decisions or p.get("decision") in wanted)
           and (suggested is None or p.get("auto_suggestion") == suggested)
           and (edited is None or bool(p.get("edit")) == edited)
           and (min_rating is None or p.get("rating", 0) >= min_rating)
           and (label is None or p.get("label") == label)
           and (person is None or any(f.get("person_id") == person for f in p.get("faces") or []))]
    if sort == "badness":
        return sorted(out, key=lambda p: ((p.get("scores") or {}).get("badness") is None,
                                          (p.get("scores") or {}).get("badness") or 0.0))
    return sorted(out, key=_capture_key)


def _known(db: dict[str, Any], rels: list[str]) -> list[dict[str, Any]]:
    by_rel = {p["rel_path"]: p for p in db["photos"]}
    missing = [r for r in rels if r not in by_rel]
    if missing:
        raise LinkError(404, {"message": "no such photo in this project", "rel_paths": missing})
    return [by_rel[r] for r in rels]


# ----- undo -----------------------------------------------------------------

def _undo_path() -> Path:
    return userstate.CONFIG_DIR / UNDO_FILE


def _undo_load() -> dict[str, list[dict[str, Any]]]:
    path = _undo_path()
    return json.loads(fsutil.read_text(path)) if path.is_file() else {}


def _undo_save(stacks: dict[str, list[dict[str, Any]]]) -> None:
    userstate.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fsutil.write_text(_undo_path(), json.dumps(stacks, ensure_ascii=False))


def remember(link: Link, command: str, kind: str, photos: list[dict[str, Any]]) -> None:
    """What a change is about to replace, for `undo`. `kind` says which route
    puts it back: decisions, edits or marks."""
    assert link.db_path is not None
    stacks = _undo_load()
    stack = stacks.setdefault(str(link.db_path), [])
    stack.append({"command": command, "at": datetime.now().isoformat(timespec="seconds"),
                  "kind": kind, "photos": photos})
    del stack[:-UNDO_KEEP]
    _undo_save(stacks)


def _where(link: Link) -> dict[str, Any]:
    """Said with every change: whether the window is showing it."""
    return {"project": str(link.db_path.parent) if link.db_path else None,
            "in_app": link.live}


# ----- reading ----------------------------------------------------------------

@click.command()
@agent_command
def status() -> None:
    """What is open: the app, its project, and what the person is looking at.

    `screen` is where the window was left: the scene, the photo the cursor is
    on (`photo`, which is what "this photo" means), and whether that photo is
    open full-screen (`viewer`). It is only given when the app is running."""
    running = find_running()
    out: dict[str, Any] = {"app": {"running": bool(running)}}
    if running is None:
        emit(out)
        return
    state = running["state"]
    out["app"].update(url=running["url"], version=running["version"],
                      tasks={k: v for k, v in state["tasks"].items() if v.get("running")})
    if not state["db_path"]:
        out["project"] = None
        emit(out)
        return
    with connect(None) as link:
        db = _db(link)
        photos = db["photos"]
        count = Counter(p.get("decision") or "none" for p in photos)
        out["project"] = {
            "name": db["project_name"], "dir": db["project_dir"], "photo_root": db["photo_root"],
            "photos": len(photos), "scenes": len({p.get("scene") for p in photos}),
            "pick": count["pick"], "review": count["review"], "reject": count["reject"],
            "undecided": count["none"], "edited": sum(1 for p in photos if p.get("edit")),
        }
        view = link.get("/api/view")
        out["screen"] = {k: view.get(k) for k in ("scene", "photo", "viewer", "filter", "sort")} \
            if view else None
    emit(out)


@click.command()
@agent_command
def projects() -> None:
    """Every project the app knows of, with its photo and decision counts."""
    with connect(None, need_project=False) as link:
        found = link.get("/api/projects/all")
    emit({"projects": [{k: v for k, v in c.items() if k != "cover"} | {"open": c["project_dir"] == found["open"]}
                       for c in found["projects"]]})


@click.command()
@project_option
@agent_command
def scenes(project: str | None) -> None:
    """The project's scenes in sidebar order, with how far each is culled."""
    with connect(project) as link:
        photos = _db(link)["photos"]
    by: dict[str, list[dict[str, Any]]] = {}
    for p in photos:
        by.setdefault(p.get("scene"), []).append(p)
    out = []
    for name in _scene_order(list(by)):
        ps = sorted(by[name], key=_capture_key)
        count = Counter(p.get("decision") or "none" for p in ps)
        out.append({"name": name, "photos": len(ps), "pick": count["pick"],
                    "review": count["review"], "reject": count["reject"],
                    "undecided": count["none"], "edited": sum(1 for p in ps if p.get("edit")),
                    "first": ps[0].get("captured_at"), "last": ps[-1].get("captured_at")})
    emit({"scenes": out})


@click.command()
@project_option
@photo_filters
@click.option("--limit", type=int, default=200, show_default=True,
              help="At most this many; 0 for all. `count` is how many matched.")
@agent_command
def photos(project: str | None, limit: int, **filters: Any) -> None:
    """The project's photos, chosen by the options, each in a short summary.

    \b
    decision   what the person (or an assistant) decided: pick, review, reject, null
    suggested  the scorer's suggestion, ranked within the scene: its best 30%
               pick, its worst 30% reject, the rest review
    scores     badness: lower is better; what `suggested` ranks by. Sharpness
               (on the subject when one was detected), brightness against the
               scene, eyes closed and subject size all add to it.
               blur_pct: sharpness rank in the scene, 0 the softest, 1 the sharpest.
               exposure_zscore: how far its brightness is from the scene's, in
               standard deviations (unsigned).
               eye_open: eye openness, below 0.18 is closed; null when not measured.
    people     person ids from `pickapicka people`; subjects: detected objects."""
    with connect(project) as link:
        chosen = select(_db(link)["photos"], **filters)
    shown = chosen if limit <= 0 else chosen[:limit]
    emit({"count": len(chosen), "shown": len(shown), "photos": [summary(p) for p in shown]})


@click.command()
@click.argument("rel_path")
@project_option
@agent_command
def photo(rel_path: str, project: str | None) -> None:
    """One photo's whole record: scores, faces, EXIF, its edit and versions."""
    with connect(project) as link:
        (p,) = _known(_db(link), [rel_path])
    emit(p)


@click.command()
@project_option
@agent_command
def people(project: str | None) -> None:
    """The people found in the project, by how many photos each is in."""
    with connect(project) as link:
        db = _db(link)
    emit({"people": [{k: v for k, v in p.items() if k != "ref"} for p in db["people"]]})


# ----- seeing -------------------------------------------------------------------

@click.command()
@click.argument("rel_paths", nargs=-1)
@project_option
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              help="Folder to write them to. Default: a new temporary folder.")
@agent_command
def thumb(rel_paths: tuple[str, ...], project: str | None, out: Path | None) -> None:
    """Each photo's thumbnail as the grid shows it (1280 px, its edit applied),
    written to a JPEG file."""
    rels = _rels(rel_paths)
    d = _out_dir(out)
    files = []
    with connect(project) as link:
        _known(_db(link), rels)
        for rel in rels:
            f = d / _file_name(rel)
            f.write_bytes(link.image("/thumb/" + quote(rel)))
            files.append({"rel_path": rel, "file": str(f)})
    emit({"files": files})


def _fit(data: bytes, edge: int) -> tuple[bytes, tuple[int, int]]:
    """A JPEG no longer than `edge` on its long side."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if max(im.size) <= edge:
        return data, im.size
    im = im.convert("RGB")
    im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return buf.getvalue(), im.size


@click.command()
@click.argument("rel_path")
@project_option
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path),
              help="File to write. Default: in a new temporary folder.")
@click.option("--edge", type=int, default=VIEW_EDGE, show_default=True,
              help="Long edge in pixels.")
@click.option("--original", is_flag=True, help="Without its edit: as it was shot.")
@click.option("--crop", metavar="X,Y,W,H",
              help="A region at full resolution, to check focus or eyes: fractions of the "
                   "frame, e.g. 0.4,0.2,0.2,0.2. Up to 3200 px wide; --edge is not applied.")
@click.option("--edit", "edit_json", metavar="JSON|@FILE",
              help="Render this edit instead of the saved one, without saving it.")
@click.option("--preset", help="Render the photo with this preset or look laid on its edit, "
                               "without saving it. An id or a name from `pickapicka presets`.")
@click.option("--amount", type=click.FloatRange(0, 200), default=100, show_default=True,
              help="With --preset: its strength in percent.")
@agent_command
def view(rel_path: str, project: str | None, out: Path | None, edge: int, original: bool,
         crop: str | None, edit_json: str | None, preset: str | None, amount: float) -> None:
    """One photo as an image file, to look at: as edited, as shot, a region at
    full resolution, or with an edit or a preset tried on it first."""
    if original and (edit_json or preset):
        raise click.UsageError("--original shows the photo without any edit; drop --edit and --preset")
    if edit_json and preset:
        raise click.UsageError("--edit or --preset, not both")
    f = out.resolve() if out else _out_dir(None) / _file_name(rel_path)
    with connect(project) as link:
        (p,) = _known(_db(link), [rel_path])
        edit = None if original else p.get("edit")
        shown = "original" if original or not edit else "saved edit"
        if edit_json:
            edit, shown = _json_arg(edit_json), "given edit"
        elif preset:
            found = _preset(link, preset)
            edit = link.post("/api/presets/apply", {"rel_path": rel_path, "edit": p.get("edit"),
                                                    "preset_id": found["id"], "amount": amount})["edit"]
            shown = f"preset {found['name']} at {amount:g}%"
        if crop:
            roi = [float(v) for v in crop.split(",")]
            if len(roi) != 4:
                raise click.BadParameter("--crop takes four fractions: X,Y,W,H")
            data = link.image("/api/edit/preview", {"rel_path": rel_path, "edit": edit or {}, "roi": roi})
        elif edit_json or preset:
            data = link.image("/api/edit/preview", {"rel_path": rel_path, "edit": edit, "fit_edge": edge})
        else:
            data = link.image("/img/" + quote(rel_path), edge=edge, original=original)
    if not crop:
        data, _size = _fit(data, edge)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data)
    from PIL import Image
    size = Image.open(f).size
    emit({"rel_path": rel_path, "file": str(f), "size": list(size), "shows": shown})


def _sheet_font(size: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def _draw_sheet(tiles: list[tuple[int, dict[str, Any], bytes]], cols: int, rows: int,
                width: int) -> bytes:
    """Thumbnails in a grid, each with its number under it and the decision
    and stars it has, on the dark ground the app shows photos on.

    The cells are as tall as the tallest photo on the sheet needs, so a sheet
    of landscapes is not half empty, and each photo stands on the line its
    number is written on."""
    from PIL import Image, ImageDraw
    pad = 6
    cell = width // cols
    ims = [Image.open(io.BytesIO(data)).convert("RGB") for _n, _p, data in tiles]
    tallest = max(im.height / im.width for im in ims)
    cell_h = round((cell - 2 * pad) * min(tallest, 1.5)) + 2 * pad
    n_rows = min(rows, -(-len(tiles) // cols))
    sheet = Image.new("RGB", (cell * cols, n_rows * (cell_h + SHEET_LABEL_H)), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    font = _sheet_font(20)
    for i, ((n, p, _data), im) in enumerate(zip(tiles, ims)):
        x0, y0 = (i % cols) * cell, (i // cols) * (cell_h + SHEET_LABEL_H)
        im.thumbnail((cell - 2 * pad, cell_h - 2 * pad), Image.Resampling.LANCZOS)
        sheet.paste(im, (x0 + (cell - im.width) // 2, y0 + cell_h - pad - im.height))
        ty = y0 + cell_h
        text = f"#{n}"
        if p.get("rating"):
            text += "  " + "*" * p["rating"]
        draw.text((x0 + pad, ty), text, fill=(240, 240, 240), font=font)
        if p.get("decision"):
            word = p["decision"]
            w = draw.textlength(word, font=font)
            draw.text((x0 + cell - pad - w, ty), word, fill=DECISION_COLOURS[word], font=font)
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=88)
    return buf.getvalue()


@click.command()
@click.argument("rel_paths", nargs=-1)
@project_option
@photo_filters
@click.option("--out", type=click.Path(file_okay=False, path_type=Path),
              help="Folder to write the sheets to. Default: a new temporary folder.")
@click.option("--cols", type=click.IntRange(1, 8), default=4, show_default=True)
@click.option("--rows", type=click.IntRange(1, 8), default=3, show_default=True,
              help="Rows per sheet; more photos go on further sheets.")
@click.option("--width", type=click.IntRange(400, 4000), default=SHEET_WIDTH, show_default=True,
              help="Sheet width in pixels.")
@agent_command
def sheet(rel_paths: tuple[str, ...], project: str | None, out: Path | None, cols: int,
          rows: int, width: int, **filters: Any) -> None:
    """Contact sheets: many photos in one image, numbered, to compare a scene
    at a glance. Name photos, or choose them with the options (a --scene is
    the usual). Each photo shows its number, its stars and its decision; the
    JSON says which number is which photo."""
    d = _out_dir(out)
    with connect(project) as link:
        db = _db(link)
        chosen = _known(db, _rels(rel_paths)) if rel_paths else select(db["photos"], **filters)
        if not chosen:
            raise LinkError(404, "no photos match")
        per = cols * rows
        numbered = list(enumerate(chosen, start=1))
        sheets = []
        for k in range(0, len(numbered), per):
            batch = [(n, p, link.image("/thumb/" + quote(p["rel_path"]))) for n, p in numbered[k:k + per]]
            f = d / f"sheet-{k // per + 1:02d}.jpg"
            f.write_bytes(_draw_sheet(batch, cols, rows, width))
            sheets.append({"file": str(f), "from": batch[0][0], "to": batch[-1][0]})
    emit({"sheets": sheets,
          "photos": [{"n": n, "rel_path": p["rel_path"], "decision": p.get("decision"),
                      "suggested": p.get("auto_suggestion")} for n, p in numbered]})


# ----- culling ------------------------------------------------------------------

@click.command()
@click.argument("decision", type=click.Choice(DECISIONS + ("none",)))
@click.argument("rel_paths", nargs=-1)
@project_option
@agent_command
def decide(decision: str, rel_paths: tuple[str, ...], project: str | None) -> None:
    """Pick, review or reject photos; none clears the decision. `-` reads the
    photos from stdin, one per line."""
    rels = _rels(rel_paths)
    with connect(project) as link:
        before = _known(_db(link), rels)
        remember(link, f"decide {decision}", "decisions",
                 [{"rel_path": p["rel_path"], "decision": p.get("decision"),
                   "decided_at": p.get("decided_at")} for p in before])
        r = link.post("/api/decide/bulk", {"rel_paths": rels,
                                           "decision": None if decision == "none" else decision},
                      change=True)
        emit({"updated": r["updated"], "decision": decision, **_where(link)})


@click.command()
@click.argument("rel_paths", nargs=-1)
@project_option
@click.option("--rating", type=click.IntRange(0, 5), help="Stars, 0 to clear.")
@click.option("--label", type=click.Choice(LABELS + ("none",)), help="Colour label; none clears it.")
@agent_command
def mark(rel_paths: tuple[str, ...], project: str | None, rating: int | None,
         label: str | None) -> None:
    """Give photos stars and/or a colour label, apart from their decision."""
    if rating is None and label is None:
        raise click.UsageError("give --rating, --label or both")
    rels = _rels(rel_paths)
    with connect(project) as link:
        before = _known(_db(link), rels)
        remember(link, "mark", "marks", [{"rel_path": p["rel_path"], "rating": p.get("rating", 0),
                                          "label": p.get("label") or ""} for p in before])
        r = link.post("/api/mark", {"rel_paths": rels, "rating": rating,
                                    "label": None if label is None else ("" if label == "none" else label)},
                      change=True)
        emit({"updated": len(r["photos"]), **_where(link)})


# ----- editing ------------------------------------------------------------------

def _sliders(pairs: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for pair in pairs:
        for item in pair.split(","):
            key, sep, val = item.partition("=")
            if not sep:
                raise click.BadParameter(f"'{item}' is not KEY=VALUE")
            out[key.strip()] = float(val)
    return out


def _photos_back(r: dict[str, Any]) -> list[dict[str, Any]]:
    """What the edit routes hand back, cut to what an assistant reads: the
    edit itself is `pickapicka photo REL`."""
    return [{"rel_path": rel, "edited": bool(p.get("edit")),
             "versions": len([e for e in p.get("edit_slots") or [] if e is not None])}
            for rel, p in r["photos"].items()]


@click.command()
@click.argument("rel_paths", nargs=-1)
@project_option
@click.option("--by", "by", multiple=True, metavar="KEY=VALUE",
              help="Move a slider by this much on each photo, from wherever it is "
                   "(exposure=0.33 is a third of a stop brighter). Repeatable, or comma-separated.")
@click.option("--to", "to", multiple=True, metavar="KEY=VALUE",
              help="Set a slider outright on each photo.")
@click.option("--auto", is_flag=True, help="Each photo's own auto tone: exposure, contrast, whites, blacks.")
@click.option("--level", is_flag=True, help="Straighten each photo's horizon from its own lines.")
@agent_command
def quick(rel_paths: tuple[str, ...], project: str | None, by: tuple[str, ...],
          to: tuple[str, ...], auto: bool, level: bool) -> None:
    """Quick edit, as the culling screen's: sliders moved or set on many photos
    at once, each keeping the rest of its edit. `pickapicka sliders` lists the
    sliders and their ranges; values past a range are held at its end."""
    rels = _rels(rel_paths)
    with connect(project) as link:
        _known(_db(link), rels)
        r = link.post("/api/edit/quick", {"rel_paths": rels, "by": _sliders(by), "to": _sliders(to),
                                          "auto": auto, "level": level}, change=True)
        remember(link, "quick", "edits", r["before"])
        emit({"photos": _photos_back(r), "levelled": r["levelled"], **_where(link)})


@click.command()
@click.argument("rel_paths", nargs=-1)
@project_option
@click.option("--set", "set_json", metavar="JSON|@FILE",
              help="Replace each photo's edit with this one ({} clears it). The photo's own "
                   "AI fills and per-face settings are kept.")
@click.option("--add", "add_json", metavar="JSON|@FILE",
              help="Lay this edit over each photo's own: what it sets is replaced, masks are added.")
@click.option("--slot", type=click.IntRange(0, 5),
              help="Save the result as a version in this slot (0-5) instead; the photo's own "
                   "edit is left as it is.")
@click.option("--name", default="", help="With --slot: the version's name.")
@agent_command
def edit(rel_paths: tuple[str, ...], project: str | None, set_json: str | None,
         add_json: str | None, slot: int | None, name: str) -> None:
    """Write an edit (the editor's whole settings, as `pickapicka photo REL`
    shows one under "edit") onto photos. Try it first with `pickapicka view REL
    --edit JSON`."""
    if (set_json is None) == (add_json is None):
        raise click.UsageError("give --set or --add")
    rels = _rels(rel_paths)
    body = {"rel_paths": rels, "edit": _json_arg(set_json or add_json),
            "mode": "replace" if set_json is not None else "add", "slot": slot, "slot_name": name}
    with connect(project) as link:
        _known(_db(link), rels)
        r = link.post("/api/edit/bulk", body, change=True)
        remember(link, "edit", "edits", r["before"])
        emit({"photos": _photos_back(r), "skipped": r["skipped"], **_where(link)})


def _all_presets(link: Link) -> list[dict[str, Any]]:
    return link.get("/api/presets")["presets"]


def _preset(link: Link, spec: str) -> dict[str, Any]:
    found = _all_presets(link)
    hit = [p for p in found if p["id"] == spec] or \
        [p for p in found if p["name"].casefold() == spec.casefold()]
    if len(hit) != 1:
        raise LinkError(404, {"message": f"no single preset '{spec}'",
                              "presets": [{"id": p["id"], "name": p["name"]} for p in found]})
    return hit[0]


@click.command()
@agent_command
def presets() -> None:
    """The person's presets, then the looks that come with the app. `parts`
    says what each sets; the rest of a photo's edit is left as it is. Every
    look keeps the photo's brightness, so none moves its exposure."""
    with connect(None, need_project=False) as link:
        found = _all_presets(link)
    emit({"presets": [{"id": p["id"], "name": p["name"], "group": p.get("group"),
                       "hint": p.get("hint"), "builtin": bool(p.get("builtin")),
                       "parts": p.get("parts")} for p in found]})


@click.command()
@click.argument("preset_spec", metavar="PRESET")
@click.argument("rel_paths", nargs=-1)
@project_option
@click.option("--amount", type=click.FloatRange(0, 200), default=100, show_default=True,
              help="Strength in percent.")
@click.option("--slot", type=click.IntRange(0, 5),
              help="Save the result as a version in this slot (0-5) instead.")
@click.option("--name", default="", help="With --slot: the version's name. Default: the preset's.")
@agent_command
def apply(preset_spec: str, rel_paths: tuple[str, ...], project: str | None, amount: float,
          slot: int | None, name: str) -> None:
    """Lay a preset or look (an id or a name from `pickapicka presets`) on
    photos. A layer it brings that a photo has nothing for (a sky on a photo
    with none) is left out of that photo, and said under `skipped`."""
    rels = _rels(rel_paths)
    with connect(project) as link:
        _known(_db(link), rels)
        found = _preset(link, preset_spec)
        r = link.post("/api/edit/bulk", {"rel_paths": rels, "preset_id": found["id"], "amount": amount,
                                         "slot": slot, "slot_name": name or found["name"]}, change=True)
        remember(link, f"apply {found['name']}", "edits", r["before"])
        emit({"preset": found["name"], "photos": _photos_back(r), "skipped": r["skipped"],
              **_where(link)})


@click.command()
def sliders() -> None:
    """The sliders `quick --by/--to` move, with their ranges. Exposure is in
    stops; temp and tint are relative to the photo's own white balance."""
    keys = sorted(set(editing._RANGES) - editing._MODIFIER_KEYS)
    emit({"sliders": {k: list(editing._RANGES[k]) for k in keys}})


# ----- export and undo ----------------------------------------------------------

@click.command()
@project_option
@click.option("--which", type=click.Choice(["picks", "kept", "all"]), default="picks", show_default=True,
              help="picks; kept is everything not rejected; or all.")
@click.option("--scene", "scenes", multiple=True, help="Only this scene (repeatable).")
@click.option("--to", "target", type=click.Path(file_okay=False, path_type=Path),
              help="Folder to write to. Default: the app's, a dated folder in Downloads "
                   "or in the photo folder's Exports.")
@click.option("--mode", type=click.Choice(["folder", "flat", "by_person"]), default="folder",
              show_default=True, help="folder: a folder per scene. flat: all in one. "
                                      "by_person: a folder per person.")
@click.option("--format", "fmt", type=click.Choice(get_args(exporting.Format)))
@click.option("--quality", type=click.IntRange(1, 100))
@click.option("--max-mb", type=float, help="Largest a JPEG may be; 0 for no limit.")
@click.option("--long-edge", type=int, help="Long edge in pixels; 0 for full size.")
@click.option("--metadata", "meta", type=click.Choice(get_args(metadata.MetadataMode)))
@click.option("--name", "template", help="File name template, e.g. '{name}'.")
@click.option("--versions/--no-versions", default=None, help="Also each photo's saved versions.")
@agent_command
def export(project: str | None, which: str, scenes: tuple[str, ...], target: Path | None,
           mode: str, fmt: str | None, quality: int | None, max_mb: float | None,
           long_edge: int | None, meta: str | None, template: str | None,
           versions: bool | None) -> None:
    """Write photos out, edits applied and RAW rendered, and wait until done.
    What is not given is as the person last exported."""
    with connect(project) as link:
        params: dict[str, Any] = {"which": which}
        if scenes:
            params["scenes"] = list(scenes)
        preview = link.get("/api/export/picks/preview", **params)
        if not preview["count"]:
            raise LinkError(404, f"nothing to export: no {which} in what was chosen")
        settings = dict(preview["settings"])
        given = {"format": fmt, "quality": quality, "metadata": meta,
                 "name_template": template, "versions": versions}
        settings.update({k: v for k, v in given.items() if v is not None})
        if max_mb is not None:
            settings["max_mb"] = max_mb or None
        if long_edge is not None:
            settings["long_edge"] = long_edge or None
        r = link.post("/api/export/picks", {
            "target_dir": str(target.expanduser().resolve()) if target else None, "mode": mode,
            "settings": settings, "scenes": list(scenes) or None, "which": which,
            "remember": False})
        said = 0.0
        while True:
            st = link.get("/api/export/status")
            if not st["running"]:
                break
            if time.monotonic() - said > 5:
                click.echo(json.dumps({"exported": st["idx"], "of": st["total"]}), err=True)
                said = time.monotonic()
            time.sleep(0.5)
        if st["error"]:
            raise LinkError(500, st["error"])
    emit({**st["result"], "target_dir": r["target_dir"], "settings": settings})


@click.command()
@project_option
@agent_command
def undo(project: str | None) -> None:
    """Put back what the last change made from the command line replaced, in
    this project. Twenty are remembered. It puts back those photos as they were
    then, so a change made to the same photos in the window since is undone too."""
    with connect(project) as link:
        stacks = _undo_load()
        stack = stacks.get(str(link.db_path)) or []
        if not stack:
            raise LinkError(404, "nothing to undo in this project")
        entry = stack[-1]
        if entry["kind"] == "decisions":
            link.post("/api/decide/restore", {"decisions": entry["photos"]}, change=True)
        elif entry["kind"] == "edits":
            link.post("/api/edit/restore", {"photos": entry["photos"]}, change=True)
        else:
            assert entry["kind"] == "marks", entry["kind"]
            groups: dict[tuple[int, str], list[str]] = {}
            for p in entry["photos"]:
                groups.setdefault((p["rating"], p["label"]), []).append(p["rel_path"])
            for (rating, label), rels in groups.items():
                link.post("/api/mark", {"rel_paths": rels, "rating": rating, "label": label}, change=True)
        stack.pop()
        _undo_save(stacks)
        emit({"undone": entry["command"], "made_at": entry["at"], "photos": len(entry["photos"]),
              "left": len(stack), **_where(link)})


ALL = [status, projects, scenes, photos, photo, people, thumb, view, sheet,
       decide, mark, quick, edit, presets, apply, sliders, export, undo]
