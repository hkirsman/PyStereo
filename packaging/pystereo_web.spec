# -*- mode: python ; coding: utf-8 -*-
#
# Build a standalone Flask web UI (app.py - browser at http://127.0.0.1:8766).
#
# From the repository root (with .venv and deps installed):
#   pip install pyinstaller
#   pyinstaller packaging/pystereo_web.spec
#
# Output: dist/PyStereoWeb/  (Windows: binary + _internal; macOS: PyStereoWeb.app via BUNDLE).
#
import pathlib
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

REPO = pathlib.Path(SPECPATH).resolve().parent
_PACKAGING = pathlib.Path(SPECPATH).resolve()

datas = [
    (str(REPO / "pystereo_core" / "_version.py"), "pystereo_core"),
]
if (REPO / "static").is_dir():
    datas.append((str(REPO / "static"), "static"))

for pkg in ("flask", "werkzeug", "jinja2"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("pystereo_core")
    + collect_submodules("pystereo_core.stereo")
    + collect_submodules("pystereo_core.stereo.methods")
    + [
        "app",
        "PIL",
        "PIL.Image",
        "kornia",
        "timm",
        "transformers",
        "huggingface_hub",
        "flask",
        "werkzeug",
        "jinja2",
    ]
)

a = Analysis(
    [str(REPO / "packaging" / "entry_web.py")],
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
    name="PyStereoWeb",
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
    name="PyStereoWeb",
)

if sys.platform == "darwin":
    _version_file = REPO / "pystereo_core" / "_version.py"
    _ns: dict = {}
    exec(_version_file.read_text(encoding="utf-8"), _ns)
    _app_version = _ns["__version__"]

    app = BUNDLE(
        coll,
        name="PyStereoWeb.app",
        bundle_identifier="io.pystereo.web",
        version=_app_version,
        info_plist={
            "CFBundleDisplayName": "PyStereo Web",
            "CFBundleName": "PyStereoWeb",
            "CFBundleShortVersionString": _app_version,
            "CFBundleVersion": _app_version,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
        },
    )
