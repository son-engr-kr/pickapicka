"""Checks for the optional models: downloading, resuming, refusing a bad file,
cancelling, deleting, switching off, and the licence a download needs.

    uv run pytest tests/test_modelstore.py

No network: the downloads read from an in-memory server that honours range
requests the way GitHub's and Hugging Face's do.
"""
from __future__ import annotations

import hashlib
import io
import threading
import time
import urllib.error

import pytest

from pickapicka import modelstore, server, userstate

A = b"first file " * 50_000
B = bytes(range(256)) * 9_000


def _file(name: str, data: bytes) -> modelstore.ModelFile:
    return modelstore.ModelFile(name, f"https://example.invalid/{name}", hashlib.sha256(data).hexdigest(), len(data))


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int) -> None:
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeServer:
    """Serves `files` by URL; `broken` names files it truncates mid-way once,
    `gate` holds every read until it is set, and `ranges` records what was asked."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.broken: set[str] = set()
        self.ranges: list[str | None] = []
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, req, timeout=None):
        name = req.full_url.rsplit("/", 1)[1]
        if name not in self.files:
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
        data = self.files[name]
        rng = req.get_header("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = data[start:]
        if name in self.broken:
            self.broken.discard(name)
            server = self

            class Cut(_Resp):
                def read(self, n=-1):
                    server.gate.wait()
                    if self.tell() >= len(body) // 2:
                        raise urllib.error.URLError("connection reset")
                    return super().read(min(n, len(body) // 2 - self.tell()))
            return Cut(body, 206 if rng else 200)
        gate = self.gate

        class Slow(_Resp):
            def read(self, n=-1):
                gate.wait()
                return super().read(n)
        return Slow(body, 206 if rng else 200)


@pytest.fixture
def pack(tmp_path, monkeypatch):
    monkeypatch.setattr(userstate, "CONFIG_DIR", tmp_path / "app-data")
    monkeypatch.setattr(userstate, "STATE_FILE", tmp_path / "app-data" / "state.json")
    p = modelstore.Pack(id="test", title="Test", detail="d", licence="MIT", folder=tmp_path / "models" / "test",
                        files=(_file("a.bin", A), _file("b.bin", B)), default_on=False,
                        terms=("licences/CreativeML-OpenRAIL-M.txt",))
    monkeypatch.setitem(modelstore.PACKS, "test", p)
    return p


def test_a_download_lands_checked_and_in_place(pack) -> None:
    dl = modelstore.Downloads(FakeServer({"a.bin": A, "b.bin": B}))
    assert not modelstore.is_ready("test")
    dl.run_now("test")
    assert modelstore.is_ready("test")
    assert (pack.folder / "a.bin").read_bytes() == A and (pack.folder / "b.bin").read_bytes() == B
    assert not any(p.suffix == ".part" for p in pack.folder.iterdir())
    st = dl.status("test")
    assert st["ready"] and st["state"] == "idle" and st["done"] == st["total"] == len(A) + len(B)


def test_a_broken_download_resumes_where_it_stopped(pack) -> None:
    srv = FakeServer({"a.bin": A, "b.bin": B})
    srv.broken.add("b.bin")
    dl = modelstore.Downloads(srv)
    with pytest.raises(urllib.error.URLError):
        dl.run_now("test")
    st = dl.status("test")
    assert st["state"] == "error" and "reach the server" in st["error"]
    part = pack.folder / "b.bin.part"
    assert part.is_file() and 0 < part.stat().st_size < len(B)
    have = part.stat().st_size
    dl.run_now("test")
    assert modelstore.is_ready("test") and (pack.folder / "b.bin").read_bytes() == B
    assert srv.ranges[-1] == f"bytes={have}-"          # only the rest was asked for


def test_a_file_that_does_not_match_is_thrown_away(pack) -> None:
    dl = modelstore.Downloads(FakeServer({"a.bin": A, "b.bin": B[:-1] + b"\x00"}))
    with pytest.raises(RuntimeError, match="checksum"):
        dl.run_now("test")
    assert (pack.folder / "a.bin").is_file()
    assert not (pack.folder / "b.bin").exists() and not (pack.folder / "b.bin.part").exists()
    assert "checksum" in dl.status("test")["error"]


def test_a_missing_file_is_an_error_with_the_servers_answer(pack) -> None:
    dl = modelstore.Downloads(FakeServer({"a.bin": A}))
    with pytest.raises(urllib.error.HTTPError):
        dl.run_now("test")
    assert "404" in dl.status("test")["error"]


def test_cancelling_stops_it_and_keeps_what_came(pack) -> None:
    srv = FakeServer({"a.bin": A, "b.bin": B})
    srv.gate.clear()
    dl = modelstore.Downloads(srv)
    dl.start("test")
    time.sleep(0.05)
    assert dl.status("test")["state"] == "downloading"
    dl.cancel("test")
    srv.gate.set()
    for _ in range(100):
        if dl.status("test")["state"] != "downloading":
            break
        time.sleep(0.02)
    assert dl.status("test")["state"] == "error" and dl.status("test")["error"] == "cancelled"
    assert not modelstore.is_ready("test")


def test_delete_takes_finished_and_partial_files_and_the_folder(pack) -> None:
    modelstore.Downloads(FakeServer({"a.bin": A, "b.bin": B})).run_now("test")
    (pack.folder / "b.bin.part").write_bytes(b"x")
    modelstore.delete("test")
    assert not pack.folder.exists()
    assert not modelstore.is_ready("test")


def test_switching_off_and_on_is_remembered(pack) -> None:
    assert not modelstore.is_on("test")                 # default_on=False
    userstate.set_model_on("test", True)
    assert modelstore.is_on("test")
    modelstore.Downloads(FakeServer({"a.bin": A, "b.bin": B})).run_now("test")
    assert modelstore.usable("test")
    userstate.set_model_on("test", False)
    assert modelstore.is_ready("test") and not modelstore.usable("test")


def _route(app, path: str, method: str):
    for r in app.routes:
        if getattr(r, "path", None) == path and method in r.methods:
            return r.endpoint
    raise KeyError(path)


def test_a_licence_with_use_restrictions_is_accepted_before_the_download(pack) -> None:
    app = server.create_app()
    app.state.downloads._open = FakeServer({"a.bin": A, "b.bin": B})
    download = _route(app, "/api/models/{pack_id}/download", "POST")
    with pytest.raises(server.HTTPException) as refused:
        download("test", server.ModelDownloadPayload(accept_terms=False))
    assert refused.value.status_code == 400 and userstate.get_model_terms("test") is None
    download("test", server.ModelDownloadPayload(accept_terms=True))
    assert userstate.get_model_terms("test") is not None
    assert modelstore.is_on("test")                     # asking for it switches it on
    for _ in range(200):
        if modelstore.is_ready("test"):
            break
        time.sleep(0.02)
    assert modelstore.is_ready("test")


def test_a_fill_with_a_model_that_is_off_is_refused(pack, tmp_path) -> None:
    app = server.create_app()
    fill = _route(app, "/api/edit/fill", "POST")
    userstate.set_model_on("genfill", False)
    with pytest.raises(server.HTTPException):
        # No project is open either; whichever it trips on first, nothing runs.
        fill(server.FillPayload(rel_path="x.jpg", ops=[{"kind": "spot", "points": [[0.5, 0.5]]}],
                                model="genfill"))


def test_every_pack_has_pinned_files() -> None:
    for p in modelstore.PACKS.values():
        assert p.files and all(len(f.sha256) == 64 and f.size > 0 and f.url.startswith("https://") for f in p.files)
        for t in p.terms:
            assert (modelstore.paths.Path(server.__file__).parent / "web" / t).is_file(), t
