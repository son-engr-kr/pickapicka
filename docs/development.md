# Development

## Running the tests

```bash
uv run pytest tests/          # all of it
```

pytest is in the `dev` dependency group, which uv installs by default, so that
line needs no flags. Every suite also runs on its own, and reports the same
count either way:

```bash
uv run python tests/test_editing.py      # the grade pipeline and masks
uv run python tests/test_grading.py      # colour wheels, HSL, split tone
uv run python tests/test_healing.py      # spot, heal, clone
uv run python tests/test_redeye.py       # red-eye and pet-eye
uv run python tests/test_segment.py      # automatic-mask segmentation
uv run python tests/test_rangemask.py    # selection by tone and by colour
uv run python tests/test_sharpening.py   # texture, dehaze, denoise, sharpen
uv run python tests/test_lens.py         # distortion, CA, vignetting
uv run python tests/test_lut.py          # .cube parsing and application
uv run python tests/test_transform.py    # crop, straighten, perspective
uv run python tests/test_geometry.py     # crop and straighten
uv run python tests/test_film.py         # the film chain
uv run python tests/test_watermark.py    # camera names, EXIF, stamping
uv run python tests/test_objects.py      # subject detection
uv run python tests/test_cluster.py      # grouping settings
uv run python tests/test_userstate.py    # remembered view, presets
uv run python tests/test_paths.py        # app-data dirs, legacy migration
uv run python tests/test_folderinfo.py   # what the setup screens say about a folder
uv run python tests/test_metadata.py     # EXIF and colour profile on export
uv run python tests/test_exporting.py    # export names, sizes, formats, copying
uv run python tests/test_portrait.py     # skin smoothing
```

Most suites collect their own `test_*` functions out of `globals()` and call
them. Two things follow, and both have bitten:

- **The runner has to be the last thing in the file.** A test defined below it
  does not exist when it collects, so it is skipped without a word.
  `test_editing.py` ran 37 of 97 that way while printing a pass;
  it now asserts that what it collected matches what the file declares.
- A test that takes a pytest fixture cannot be called by such a loop.
  `test_redeye.py` (monkeypatch) and `test_segment.py` (skipif on the
  downloaded model) hand `__main__` to `pytest.main` instead of skipping.

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
