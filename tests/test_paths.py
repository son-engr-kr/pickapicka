"""Checks for the app-data directories and the one-shot legacy migration.

    uv run python tests/test_paths.py

The migration moves real user state, so what matters here is that it never
overwrites and never deletes anything it did not move. Module-level constants are
read at import time, so they are repointed inside a throwaway directory.
"""
from __future__ import annotations

import importlib
from pathlib import Path


def _fresh(home: Path):
    """A `paths` module whose three directories live under `home`."""
    from picture_classifier import paths
    importlib.reload(paths)
    paths.DATA_DIR = home / "data"
    paths.CACHE_DIR = home / "cache"
    paths.MODEL_DIR = paths.CACHE_DIR / "models"
    paths.LEGACY_DIR = home / ".picture-classifier"
    paths._LEGACY_MOVES = (
        ("state.json", paths.DATA_DIR),
        ("models", paths.CACHE_DIR),
    )
    return paths


def _legacy_install(p, *, state: str = '{"recents": []}', model: bool = True) -> None:
    p.LEGACY_DIR.mkdir(parents=True, exist_ok=True)
    (p.LEGACY_DIR / "state.json").write_text(state, encoding="utf-8")
    if model:
        (p.LEGACY_DIR / "models").mkdir()
        (p.LEGACY_DIR / "models" / "yolox_tiny.onnx").write_bytes(b"weights")


# ----- directory shape ----------------------------------------------------

def test_directories_are_platform_specific_not_dotfiles() -> None:
    """The whole point of the change: nothing lands in a home dotfile."""
    from picture_classifier import paths
    importlib.reload(paths)
    for d in (paths.DATA_DIR, paths.CACHE_DIR):
        assert not d.name.startswith("."), d
        assert paths.APP_NAME in str(d), d
    # State and re-downloadable weights are kept apart.
    assert paths.MODEL_DIR.is_relative_to(paths.CACHE_DIR)
    assert not paths.MODEL_DIR.is_relative_to(paths.DATA_DIR)


# ----- migration ----------------------------------------------------------

def test_nothing_to_do_without_a_legacy_install(tmp_path) -> None:
    p = _fresh(tmp_path)
    assert p.migrate_legacy() == []


def test_state_and_models_move_to_their_own_directories(tmp_path) -> None:
    p = _fresh(tmp_path)
    _legacy_install(p)

    moved = p.migrate_legacy()

    assert len(moved) == 2, moved
    assert (p.DATA_DIR / "state.json").read_text(encoding="utf-8") == '{"recents": []}'
    assert (p.MODEL_DIR / "yolox_tiny.onnx").read_bytes() == b"weights"
    # The old directory is gone once it has been emptied.
    assert not p.LEGACY_DIR.exists()


def test_migration_is_idempotent(tmp_path) -> None:
    p = _fresh(tmp_path)
    _legacy_install(p)
    p.migrate_legacy()
    assert p.migrate_legacy() == [], "a second run must be a no-op"
    assert (p.DATA_DIR / "state.json").is_file()


def test_an_existing_destination_is_never_overwritten(tmp_path) -> None:
    """Newer state at the destination wins, and the old copy is left to look at
    rather than silently destroyed."""
    p = _fresh(tmp_path)
    _legacy_install(p, state='{"recents": ["old"]}')
    p.DATA_DIR.mkdir(parents=True)
    (p.DATA_DIR / "state.json").write_text('{"recents": ["new"]}', encoding="utf-8")

    moved = p.migrate_legacy()

    assert (p.DATA_DIR / "state.json").read_text(encoding="utf-8") == '{"recents": ["new"]}'
    assert (p.LEGACY_DIR / "state.json").is_file(), "the unmoved copy must survive"
    assert p.LEGACY_DIR.is_dir(), "a directory with something left in it stays"
    assert len(moved) == 1, "models still moved even though state did not"


def test_unrecognised_files_are_not_deleted(tmp_path) -> None:
    p = _fresh(tmp_path)
    _legacy_install(p)
    (p.LEGACY_DIR / "state.json.tmp").write_text("half a write", encoding="utf-8")

    p.migrate_legacy()

    assert p.LEGACY_DIR.is_dir(), "removing a dir with unknown contents is the one bad move"
    assert (p.LEGACY_DIR / "state.json.tmp").is_file()


def test_a_partial_legacy_install_migrates_what_is_there(tmp_path) -> None:
    p = _fresh(tmp_path)
    _legacy_install(p, model=False)

    moved = p.migrate_legacy()

    assert len(moved) == 1
    assert (p.DATA_DIR / "state.json").is_file()
    assert not p.LEGACY_DIR.exists()


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
