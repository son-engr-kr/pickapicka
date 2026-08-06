# Development

## Running the tests

Each suite runs standalone or under pytest, and takes no fixtures beyond a
temporary directory:

```bash
uv run python tests/test_editing.py     # the grade pipeline and masks
uv run python tests/test_geometry.py    # crop and straighten
uv run python tests/test_film.py        # the film chain
uv run python tests/test_watermark.py   # camera names, EXIF, stamping
uv run python tests/test_objects.py     # subject detection
uv run python tests/test_cluster.py     # grouping settings
uv run python tests/test_userstate.py   # remembered view, presets
uv run python tests/test_paths.py       # app-data dirs, legacy migration
```

They lean on properties rather than golden images: that a window of a render
matches the same part of the whole render, that a neutral setting is a genuine
no-op, that grain does not change size with the render resolution. Those are the
things that broke in practice.

## Cutting a release

A release is cut by pushing a `v*` tag. Three GitHub Actions fire on it and
all attach their output to the **same** GitHub Release:

- **Build macOS app** — builds the `.app` and packages a `.pkg`.
- **Build Windows app** — builds the PyInstaller bundle and wraps it in an
  Inno Setup installer (`setup.exe`).
- **Update Homebrew tap** — refreshes the
  [tap](https://github.com/son-engr-kr/homebrew-picture-classifier) formula.

```bash
# 1. bump "version" in pyproject.toml, then:
git commit -am "chore: bump to 0.1.4"
git tag v0.1.4
git push origin main v0.1.4
```

The tag push triggers all three workflows; build artifacts publish to the
GitHub Release automatically — no manual upload. To dry-run a platform build
*without* cutting a release, use **Actions → Build … app → Run workflow**
(workflow_dispatch); it uploads an artifact instead of publishing a release.

The Homebrew workflow needs a `TAP_TOKEN` repository secret — a fine-grained
PAT with `Contents: Write` on `son-engr-kr/homebrew-picture-classifier`.
