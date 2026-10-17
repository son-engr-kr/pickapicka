"""How the command line reaches a project: through the app when the app has
that project open, else through an app built in this process.

The server keeps a project's picks.json in memory and writes the whole of it
back on every change, so a command that wrote the file itself while the window
had the project open would be undone by the window's next save. A command goes
where the project is open instead. Either way it calls the same routes with the
same bodies, so what the command line asks for is checked exactly as a click in
the window is.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

from . import __version__, fsutil, projects as projects_mod, userstate

RUNNING_FILE = "running.json"
# Sent with every request that changes the project. The server counts them in
# /api/state, which the window asks for every two seconds, and the window
# reloads its photos when the count moves: what an assistant decides from a
# terminal shows on the person's screen without them doing anything.
CHANGE_HEADER = "X-Pickapicka-Change"
# A render of a large RAW, or an export's wait, is far longer than httpx's
# default of five seconds. Nothing here is worth giving up on part way.
TIMEOUT = httpx.Timeout(None, connect=5.0)


class LinkError(Exception):
    """A route said no: its status and what it said, as the window shows it."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


# ----- the running app ---------------------------------------------------

def _running_path() -> Path:
    return userstate.CONFIG_DIR / RUNNING_FILE


def note_running(url: str) -> None:
    """Called by serve() once it has its port."""
    userstate.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    fsutil.write_text(_running_path(), json.dumps(
        {"url": url, "pid": os.getpid(), "version": __version__}))


def forget_running() -> None:
    """Called by serve() on its way out. A second app started on another port
    since then wrote its own, which is left for it."""
    path = _running_path()
    if path.is_file() and json.loads(fsutil.read_text(path)).get("pid") == os.getpid():
        path.unlink()


def find_running() -> dict[str, Any] | None:
    """The app this user has open: its address and its /api/state. None when
    none is, which includes one that stopped without clearing its note (a
    crash, a forced quit): nothing answers there, or what does is not it."""
    path = _running_path()
    if not path.is_file():
        return None
    note = json.loads(fsutil.read_text(path))
    try:
        r = httpx.get(note["url"] + "/api/state", timeout=2.0)
    except httpx.TransportError:
        return None
    if r.status_code != 200:
        return None
    state = r.json()
    if not isinstance(state, dict) or "outside_rev" not in state:
        return None
    return {"url": note["url"], "version": note.get("version"), "state": state}


# ----- which project ------------------------------------------------------

def _kept_in(photo_dir: Path) -> list[Path]:
    kept = photo_dir / projects_mod.PHOTOS_SUBDIR
    if not kept.is_dir():
        return []
    return [c for c in sorted(kept.iterdir()) if (c / "picks.json").is_file()
            and not userstate.is_deleted_project(c.name)]


def resolve_project(spec: str, known: list[dict[str, Any]]) -> Path:
    """The picks.json that `spec` names: a project folder, a photo folder with
    one project kept in it, a legacy picks.json, or a project's name as the
    project list has it (`known`, from /api/projects/all). A name matches
    whole first, ignoring case, then as part of one name."""
    path = Path(spec).expanduser()
    if path.is_file():
        if path.name != "picks.json" and path.suffix != ".json":
            raise LinkError(400, f"not a project: {path}")
        return path.resolve()
    if path.is_dir():
        if (path / "picks.json").is_file():
            return (path / "picks.json").resolve()
        inside = _kept_in(path)
        if len(inside) == 1:
            return (inside[0] / "picks.json").resolve()
        if inside:
            raise LinkError(400, {"message": f"{path} keeps several projects; name one",
                                  "projects": [str(c) for c in inside]})
        raise LinkError(400, f"no project in {path}: it has no picks.json and keeps none")
    want = spec.casefold()
    live = [c for c in known if not c.get("project_missing")]
    found = [c for c in live if c["name"].casefold() == want] \
        or [c for c in live if want in c["name"].casefold()]
    if len(found) == 1:
        return (Path(found[0]["project_dir"]) / "picks.json").resolve()
    if found:
        raise LinkError(400, {"message": f"several projects match '{spec}'; name one whole",
                              "projects": [c["project_dir"] for c in found]})
    raise LinkError(404, {"message": f"no project named '{spec}'",
                          "projects": [c["name"] for c in live]})


# ----- the link -----------------------------------------------------------

class Link:
    """One way to the app's routes. `live` when they are the open app's, whose
    window then shows every change."""

    def __init__(self, client: httpx.Client, *, live: bool, url: str | None,
                 db_path: Path | None) -> None:
        self.client = client
        self.live = live
        self.url = url
        self.db_path = db_path

    @staticmethod
    def _check(r: httpx.Response) -> httpx.Response:
        if r.status_code < 400:
            return r
        try:
            detail = r.json().get("detail", r.json())
        except ValueError:
            detail = r.text
        raise LinkError(r.status_code, detail)

    def get(self, path: str, **params: Any) -> Any:
        return self._check(self.client.get(path, params=params or None)).json()

    def post(self, path: str, body: Any = None, *, change: bool = False) -> Any:
        """`change` for a route that changes the project, which the window is
        then told of. A route that only works something out (a preview, a
        check) is not a change, and would only make the window reload."""
        headers = {CHANGE_HEADER: "1"} if change else None
        return self._check(self.client.post(path, json=body, headers=headers)).json()

    def image(self, path: str, body: Any = None, **params: Any) -> bytes:
        """A route that answers with a picture: GET, or POST with a body."""
        if body is None:
            r = self.client.get(path, params=params or None)
        else:
            r = self.client.post(path, json=body, params=params or None)
        return self._check(r).content

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "Link":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def _local(db_path: Path | None) -> Link:
    from fastapi.testclient import TestClient

    from . import server
    app = server.create_app()
    ctx = app.state.ctx
    # No grid here to make thumbnails ahead of: the window makes any it finds
    # missing, and waiting for them on the way out would hold the prompt.
    ctx.ahead = False
    if db_path is not None:
        # A project folder has its caches in .cache/; a picks.json beside the
        # photos (`pickapicka score`) keeps them beside itself instead.
        if db_path.name == "picks.json" and (db_path.parent / ".cache").is_dir():
            projects_mod.heal_photo_root(db_path.parent)
            ctx.load_project(db_path.parent)
        else:
            ctx.load_db(db_path)
    return Link(TestClient(app), live=False, url=None, db_path=db_path)


def connect(project: str | None, *, need_project: bool = True) -> Link:
    """A link to `project` (see resolve_project), or with none named, to the
    project open in the app. `need_project` False for what needs no project
    (the project list, the presets)."""
    running = find_running()
    open_db = running["state"]["db_path"] if running else None
    if project is None:
        if running and (open_db or not need_project):
            return Link(httpx.Client(base_url=running["url"], timeout=TIMEOUT), live=True,
                        url=running["url"], db_path=Path(open_db) if open_db else None)
        if need_project:
            raise LinkError(400, "no project is open in the app; name one with --project "
                                 "(`pickapicka projects` lists them)")
        return _local(None)
    if running:
        with httpx.Client(base_url=running["url"], timeout=TIMEOUT) as c:
            known = Link._check(c.get("/api/projects/all")).json()["projects"]
    else:
        with _local(None) as lk:
            known = lk.get("/api/projects/all")["projects"]
    db_path = resolve_project(project, known)
    if running and open_db and Path(open_db).resolve() == db_path:
        return Link(httpx.Client(base_url=running["url"], timeout=TIMEOUT), live=True,
                    url=running["url"], db_path=db_path)
    return _local(db_path)
