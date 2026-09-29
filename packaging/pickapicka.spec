# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Pickapicka — cross-platform.

macOS wraps the bundle in a .app; on Windows the COLLECT tree
(dist/pickapicka/) is the distributable, wrapped by Inno Setup.

Run from the project root:
    uvx --with-requirements pyproject.toml pyinstaller packaging/pickapicka.spec
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
WEB_DIR = ROOT / "src" / "pickapicka" / "web"
ENTRY = ROOT / "src" / "pickapicka" / "app_entry.py"
# Drawn by packaging/make_icon.py. Without them PyInstaller puts its own icon on
# the app, which is what the Dock and the taskbar showed up to 0.10.0.
ICON_DIR = ROOT / "packaging" / "icons"
APP_VERSION = os.environ.get("APP_VERSION", "0.0.0-dev")

block_cipher = None

a = Analysis(
    [str(ENTRY)],
    pathex=[str(ROOT / "src")],
    # rawpy bundles libraw as a shared lib inside its wheel; pull it in.
    binaries=collect_dynamic_libs("rawpy"),
    # The two Haar cascades redeye.py finds faces and eyes with. PyInstaller's
    # cv2 hook collects cv2's config files but not cv2/data/, so without these
    # "Find red eyes" fails in the bundle with the cascade missing.
    datas=[(str(WEB_DIR), "pickapicka/web")] + collect_data_files(
        "cv2", includes=["data/haarcascade_frontalface_default.xml",
                         "data/haarcascade_eye.xml"]),
    hiddenimports=[
        "rawpy",
        "rawpy._rawpy",
        # scoring.objects imports it lazily inside the session factory, so the
        # static analysis only reaches it via insightface — be explicit.
        "onnxruntime",
        # uvicorn pulls these dynamically; PyInstaller's static analysis misses them.
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # insightface uses dynamic plugin discovery for face models.
        "insightface",
        "insightface.app",
        "insightface.app.face_analysis",
        "insightface.data",
        "insightface.model_zoo",
        "insightface.model_zoo.arcface_onnx",
        "insightface.model_zoo.attribute",
        "insightface.model_zoo.landmark",
        "insightface.model_zoo.model_zoo",
        "insightface.model_zoo.retinaface",
        "insightface.model_zoo.scrfd",
        "insightface.utils",
        "insightface.utils.face_align",
        # sklearn submodules used at runtime via DBSCAN(metric="cosine")
        "sklearn.cluster",
        "sklearn.metrics.pairwise",
        "sklearn.utils._typedefs",
        "sklearn.utils._heap",
        "sklearn.utils._sorting",
        "sklearn.utils._vector_sentinel",
        "sklearn.neighbors._partition_nodes",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="pickapicka",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # The exe's own icon is what Windows shows for it everywhere; on macOS the
    # bundle's icon below is the one that counts.
    icon=str(ICON_DIR / "pickapicka.ico") if sys.platform == "win32" else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="pickapicka",
)

# macOS wraps the COLLECT tree in a .app bundle; on Windows/Linux the COLLECT
# directory (dist/pickapicka/) is itself the distributable.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Pickapicka.app",
        icon=str(ICON_DIR / "pickapicka.icns"),
        bundle_identifier="kr.son-engr.pickapicka",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": "Pickapicka",
            "CFBundleDisplayName": "Pickapicka",
            "CFBundleIdentifier": "kr.son-engr.pickapicka",
            "CFBundleVersion": APP_VERSION,
            "CFBundleShortVersionString": APP_VERSION,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "LSUIElement": False,
        },
    )
