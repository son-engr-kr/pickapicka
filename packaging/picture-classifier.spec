# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Picture Classifier — cross-platform.

macOS wraps the bundle in a .app; on Windows the COLLECT tree
(dist/picture-classifier/) is the distributable, wrapped by Inno Setup.

Run from the project root:
    uvx --with-requirements pyproject.toml pyinstaller packaging/picture-classifier.spec
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
WEB_DIR = ROOT / "src" / "picture_classifier" / "web"
ENTRY = ROOT / "src" / "picture_classifier" / "app_entry.py"
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
    datas=[(str(WEB_DIR), "picture_classifier/web")] + collect_data_files(
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
    name="picture-classifier",
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="picture-classifier",
)

# macOS wraps the COLLECT tree in a .app bundle; on Windows/Linux the COLLECT
# directory (dist/picture-classifier/) is itself the distributable.
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Picture Classifier.app",
        icon=None,
        bundle_identifier="kr.son-engr.picture-classifier",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": "Picture Classifier",
            "CFBundleDisplayName": "Picture Classifier",
            "CFBundleIdentifier": "kr.son-engr.picture-classifier",
            "CFBundleVersion": APP_VERSION,
            "CFBundleShortVersionString": APP_VERSION,
            "LSMinimumSystemVersion": "12.0",
            "NSHighResolutionCapable": True,
            "LSUIElement": False,
        },
    )
