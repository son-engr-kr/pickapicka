"""The optional models: what each one is, downloading it, deleting it, and
whether the feature it powers is switched on.

A feature that needs a model shows in the editor whether or not the model is
there. Using it before the model is downloaded, or while it is switched off
in Preferences, opens the download instead, which states the size and the
licence and, for a model whose licence carries use restrictions, asks for
them to be accepted first. Preferences lists every model with its size and
state, and can download, delete or switch each one off.

Downloads run in the background, one file after another, each into a .part
file that a later attempt resumes with an HTTP range request, and each file
is checked against its pinned SHA-256 before it is moved into place, so an
interrupted, truncated or tampered download never leaves a file the app would
trust. A failure is kept as a message on the pack, for the page to show next
to a retry.
"""
from __future__ import annotations

import hashlib
import shutil
import threading
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import paths, userstate

RELEASE = "https://github.com/son-engr-kr/pickapicka/releases/download/models-0.2.0/"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class Pack:
    id: str
    title: str
    detail: str
    licence: str
    folder: Path
    files: tuple[ModelFile, ...]
    # Licence texts the person accepts before downloading, as paths under the
    # app's web folder; empty for a licence that asks nothing of its users.
    terms: tuple[str, ...] = ()
    # Whether the feature can be switched off in Preferences, and its state
    # when nothing has been said.
    switchable: bool = True
    default_on: bool = True
    # Resident memory while the model works, measured; 0 for the small ones.
    memory: int = 0

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)


_SD = "sd15-inpaint-lcm"
PACKS: dict[str, Pack] = {p.id: p for p in (
    Pack(
        id="aifill", title="AI fill",
        detail="Draws what belongs in a heal: skin with its pores, an edge carried through. "
               "MI-GAN (ICCV 2023).",
        licence="MIT (code and weights)", folder=paths.MODEL_DIR,
        files=(ModelFile("migan_pipeline_v2.onnx",
                         "https://huggingface.co/andraniksargsyan/migan/resolve/main/migan_pipeline_v2.onnx",
                         "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b", 28079181),)),
    Pack(
        id="genfill", title="Generative fill",
        detail="A diffusion model for large regions, where AI fill runs out: an eye bag, crow's feet, "
               "glasses or a strand of hair across skin. Stable Diffusion 1.5 inpainting with LCM-LoRA, "
               "four steps; about 20 seconds a fill on a laptop's CPU.",
        licence="CreativeML OpenRAIL-M and OpenRAIL++-M: use restrictions apply", folder=paths.MODEL_DIR / _SD,
        files=tuple(ModelFile(n, RELEASE + n, h, s) for n, s, h in (
            ("unet.onnx", 532093, "aa18b01246f9f1ba3311e61d78a6aa71d7a92d9e9c709eedaf1038e494cb0e5f"),
            ("unet.onnx.data", 1719446400, "4a3792122198dc04dead85e7e4bfd60d2cddf6fe29d8b548a1b78945fd640483"),
            ("vae_encoder.onnx", 122884, "5998e90bf01a113bf8ee65d27d18239a6e7e4720995db3aade28469f58dfb5ff"),
            ("vae_encoder.onnx.data", 68375296, "0941bad47aa4f5819076306be15aa8558bde8d50b0c4ff95bd4252cac1d2c000"),
            ("vae_decoder.onnx", 132127, "26d86e74cfdedf4d74a21e6010c442020545cdc7c66bc59df9ffcf73045a0381"),
            ("vae_decoder.onnx.data", 99046144, "726bc0bc3e4d8adc5300c81149821562f9c3bebcc5c6a3261965a4e3d9f2e430"),
            ("empty_prompt.npy", 236672, "d2d2f2583ceded75236c4190bd1df1086d7ff7e608ca402210f21231c193b28b"),
            ("LICENSE-CreativeML-OpenRAIL-M.txt", 14385, "be351ebe7ac01bcdbb018639aadcfd38f136b7dc3f2a3d4d3a24db51d1b210ef"),
            ("LICENSE-CreativeML-OpenRAIL++-M.txt", 14206, "e514164be7f2ca2c0fb39e7283b01ab4a06d802837eaf171aa4645cfd7ee1157"),
            ("NOTICE.txt", 1597, "9e0b27c1092045c21889722776ba2eeba7c37297e1aa7a7e7719c9845c5deb81"),
        )),
        terms=("licences/CreativeML-OpenRAIL-M.txt", "licences/CreativeML-OpenRAIL++-M.txt"),
        default_on=False, memory=12 * 10 ** 9),
    Pack(
        id="segment", title="Automatic masks",
        detail="Subject, background, skin, face, hair and clothes masks, and the skin the Portrait "
               "panel works on. MediaPipe multiclass selfie segmentation.",
        licence="Apache 2.0", folder=paths.MODEL_DIR, switchable=False,
        files=(ModelFile("selfie_multiclass_256x256.onnx",
                         "https://github.com/son-engr-kr/pickapicka/releases/download/models-0.1.0/"
                         "selfie_multiclass_256x256.onnx",
                         "46532ed4d2f36a60038b729042d8f01acf4f6e7a6a0b016ddc0e6782c217ce66", 16454469),)),
    Pack(
        id="objects", title="Subject detection",
        detail="Finds cars, people, pets and the rest when scoring, so sharpness is measured on the "
               "subject. YOLOX-Tiny.",
        licence="Apache 2.0", folder=paths.MODEL_DIR, switchable=False,
        files=(ModelFile("yolox_tiny.onnx",
                         "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx",
                         "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7", 20219662),)),
)}

_CHUNK = 1 << 20


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def file_ready(pack: Pack, f: ModelFile) -> bool:
    p = pack.folder / f.name
    return p.is_file() and p.stat().st_size == f.size


def is_ready(pack_id: str) -> bool:
    pack = PACKS[pack_id]
    return all(file_ready(pack, f) for f in pack.files)


def is_on(pack_id: str) -> bool:
    pack = PACKS[pack_id]
    return pack.default_on if not pack.switchable else userstate.get_model_on(pack_id, pack.default_on)


def usable(pack_id: str) -> bool:
    return is_ready(pack_id) and is_on(pack_id)


# ----- downloading -----------------------------------------------------------

@dataclass
class _Job:
    state: str = "idle"          # idle | downloading | error
    done: int = 0
    total: int = 0
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)


class Downloads:
    """One download at a time per pack, on its own thread."""

    def __init__(self, opener: Callable[..., Any] = urllib.request.urlopen) -> None:
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._open = opener

    def status(self, pack_id: str) -> dict[str, Any]:
        pack = PACKS[pack_id]
        with self._lock:
            job = self._jobs.get(pack_id) or _Job()
            if job.state == "error" and is_ready(pack_id):
                job = _Job()        # a failure the pack has since got past
            return {"id": pack.id, "title": pack.title, "detail": pack.detail, "licence": pack.licence,
                    "size": pack.size, "ready": is_ready(pack_id), "on": is_on(pack_id),
                    "switchable": pack.switchable, "terms": list(pack.terms),
                    "accepted": userstate.get_model_terms(pack_id),
                    "memory": pack.memory, "ram": total_memory(), "memory_mode": memory_mode(pack_id),
                    "keep_loaded": keep_loaded(pack_id),
                    "tight": bool(pack.memory) and total_memory() < WARN_FACTOR * pack.memory,
                    "state": job.state, "done": job.done, "total": job.total or pack.size,
                    "error": job.error}

    def start(self, pack_id: str) -> None:
        with self._lock:
            job = self._jobs.get(pack_id)
            if job is not None and job.state == "downloading":
                return
            job = self._jobs[pack_id] = _Job(state="downloading", total=PACKS[pack_id].size)
        threading.Thread(target=self._run, args=(pack_id, job), daemon=True,
                         name=f"model-{pack_id}").start()

    def cancel(self, pack_id: str) -> None:
        with self._lock:
            job = self._jobs.get(pack_id)
        if job is not None:
            job.cancel.set()

    def run_now(self, pack_id: str) -> None:
        """The same download on the calling thread, raising on failure."""
        job = _Job(state="downloading", total=PACKS[pack_id].size)
        with self._lock:
            self._jobs[pack_id] = job
        self._run(pack_id, job, reraise=True)

    def _run(self, pack_id: str, job: _Job, reraise: bool = False) -> None:
        pack = PACKS[pack_id]
        try:
            pack.folder.mkdir(parents=True, exist_ok=True)
            job.done = sum(f.size for f in pack.files if file_ready(pack, f))
            for f in pack.files:
                if file_ready(pack, f):
                    continue
                self._fetch(pack, f, job)
            with self._lock:
                job.state = "idle"
        except Exception as exc:     # shown to the person, next to a retry
            traceback.print_exc()
            with self._lock:
                job.state = "error"
                job.error = "cancelled" if job.cancel.is_set() else _describe(exc)
            if reraise:
                raise

    def _fetch(self, pack: Pack, f: ModelFile, job: _Job) -> None:
        dest = pack.folder / f.name
        part = dest.with_name(dest.name + ".part")
        have = part.stat().st_size if part.is_file() else 0
        if have > f.size:
            part.unlink()
            have = 0
        req = urllib.request.Request(f.url, headers={"Range": f"bytes={have}-"} if have else {})
        base = job.done
        with self._open(req, timeout=60) as resp:
            # A server that ignores the range sends the whole file again.
            if have and getattr(resp, "status", 200) != 206:
                have = 0
            with part.open("ab" if have else "wb") as out:
                got = have
                job.done = base + got
                while chunk := resp.read(_CHUNK):
                    if job.cancel.is_set():
                        raise RuntimeError("cancelled")
                    out.write(chunk)
                    got += len(chunk)
                    job.done = base + got
        if part.stat().st_size != f.size or _digest(part) != f.sha256:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"{f.name} did not match its checksum; it was thrown away")
        shutil.move(str(part), str(dest))


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"the server answered {exc.code} ({exc.reason})"
    if isinstance(exc, urllib.error.URLError):
        return f"could not reach the server ({exc.reason})"
    return str(exc) or type(exc).__name__


# ----- memory -----------------------------------------------------------------
# Generative fill holds 10 to 12.5 GB while it works (peak resident memory on
# macOS over repeated runs). Kept loaded between fills it is ready at once;
# freed after each it costs about five seconds more a fill (loading the graphs
# and their first runs) and gives that memory back. "auto" keeps it only where
# that leaves plenty over: on a machine with at least KEEP_FACTOR times what it
# needs, so a 32 GB machine keeps it and a 16 GB one does not.

MEMORY_MODES = ("auto", "keep", "release")
KEEP_FACTOR = 2.5
WARN_FACTOR = 1.33         # below this much memory per what it needs, the dialog warns


def total_memory() -> int:
    """This machine's physical memory in bytes."""
    import os
    import sys
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MemoryStatus()
        st.dwLength = ctypes.sizeof(MemoryStatus)
        assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)), "GlobalMemoryStatusEx failed"
        return int(st.ullTotalPhys)
    return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))


def memory_mode(pack_id: str) -> str:
    mode = userstate.get_model_memory(pack_id)
    return mode if mode in MEMORY_MODES else "auto"


def keep_loaded(pack_id: str, total: int | None = None) -> bool:
    """Whether a pack stays loaded between uses, by its setting, or for "auto"
    by this machine's memory."""
    mode = memory_mode(pack_id)
    if mode != "auto":
        return mode == "keep"
    need = PACKS[pack_id].memory
    return need == 0 or (total if total is not None else total_memory()) >= KEEP_FACTOR * need


def delete(pack_id: str) -> None:
    """Remove a pack's files, finished or partial. A pack in its own folder
    takes the folder; one sharing the models folder only its own files."""
    pack = PACKS[pack_id]
    for f in pack.files:
        for p in (pack.folder / f.name, pack.folder / (f.name + ".part")):
            p.unlink(missing_ok=True)
    if pack.folder != paths.MODEL_DIR and pack.folder.is_dir() and not any(pack.folder.iterdir()):
        pack.folder.rmdir()


def accept_terms(pack_id: str) -> None:
    userstate.set_model_terms(pack_id, datetime.now().isoformat(timespec="seconds"))
