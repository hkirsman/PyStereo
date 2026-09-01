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

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

block_cipher = None

REPO = pathlib.Path(SPECPATH).resolve().parent
_PACKAGING = pathlib.Path(SPECPATH).resolve()

if sys.platform == "win32":
    ICON = (_PACKAGING / "pystereo.ico").resolve()
elif sys.platform == "darwin":
    ICON = (_PACKAGING / "pystereo.icns").resolve()
else:
    ICON = None

if ICON is not None and not ICON.is_file():
    raise SystemExit(f"Missing {ICON} - run: python packaging/brand_icon.py")

datas = [
    (str(REPO / "version.txt"), "."),
]
if (REPO / "ml-sharp" / "src").is_dir():
    datas.append((str(REPO / "ml-sharp" / "src"), "ml-sharp/src"))
else:
    print("WARNING: ml-sharp/src missing - SHARP methods will not work in the bundle.")
    print("Run: git submodule update --init ml-sharp")
if (REPO / "static").is_dir():
    datas.append((str(REPO / "static"), "static"))

try:
    datas += copy_metadata("imageio")
except Exception:
    pass

for pkg in ("flask", "werkzeug", "jinja2"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

hiddenimports = (
    collect_submodules("pystereo_core")
    + collect_submodules("pystereo_core.stereo")
    + collect_submodules("pystereo_core.stereo.methods")
    + (collect_submodules("sharp") if (REPO / "ml-sharp" / "src").is_dir() else [])
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
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "pystereo_core.web_launch_dialog",
        "pystereo_core.logging_config",
        "pystereo_core.sharp_imports",
        "imageio",
        "imageio.v2",
        "imageio.core",
        "imageio.plugins",
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
    icon=str(ICON) if ICON is not None else None,
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
    _app_version = (REPO / "version.txt").read_text(encoding="utf-8").strip()

    app = BUNDLE(
        coll,
        name="PyStereoWeb.app",
        icon=str(ICON),
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
