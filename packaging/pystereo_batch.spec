# -*- mode: python ; coding: utf-8 -*-
#
# Build a standalone app for pystereo_core (Qt GUI + CLI).
#
# From the repository root (with .venv activated and deps installed):
#   pip install pyinstaller
#   pyinstaller packaging/pystereo_batch.spec
#
# Output: dist/PyStereo/  (Windows: binary + _internal; macOS: PyStereo.app via BUNDLE).
#
# Expect a large bundle (PyTorch + Qt). Model weights still download on first
# inference unless you ship them separately.
#
import pathlib
import sys

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None

REPO = pathlib.Path(SPECPATH).resolve().parent
_PACKAGING = pathlib.Path(SPECPATH).resolve()

datas = [
    (str(REPO / "pystereo_core" / "_version.py"), "pystereo_core"),
]

hiddenimports = (
    collect_submodules("pystereo_core")
    + collect_submodules("pystereo_core.stereo")
    + collect_submodules("pystereo_core.stereo.methods")
    + [
        "PIL",
        "PIL.Image",
        "kornia",
        "timm",
        "transformers",
        "huggingface_hub",
    ]
)

a = Analysis(
    [str(REPO / "packaging" / "entry_batch.py")],
    pathex=[str(REPO)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="PyStereo",
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
    name="PyStereo",
)

if sys.platform == "darwin":
    _version_file = REPO / "pystereo_core" / "_version.py"
    _ns: dict = {}
    exec(_version_file.read_text(encoding="utf-8"), _ns)
    _app_version = _ns["__version__"]

    app = BUNDLE(
        coll,
        name="PyStereo.app",
        bundle_identifier="io.pystereo.batch",
        version=_app_version,
        info_plist={
            "CFBundleDisplayName": "PyStereo",
            "CFBundleName": "PyStereo",
            "CFBundleShortVersionString": _app_version,
            "CFBundleVersion": _app_version,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
