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

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# Spec files are not on sys.path; load the helper next to this file.
sys.path.insert(0, str(pathlib.Path(SPECPATH).resolve()))
from torchvision_binaries import collect_torchvision_binaries, merge_binaries  # noqa: E402

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


def _tree_datas(src: pathlib.Path, dest: str) -> list:
    """Collect a directory as datas, skipping __pycache__ and .pyc.

    Copying a tree wholesale also ships whatever stale bytecode the build
    machine happened to have. Every .pyc bakes the build-time absolute path
    into co_filename, so a traceback in the shipped app prints the builder's
    home directory. Ship source only - the app compiles what it needs.
    """
    out = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".pyo"):
            continue
        rel = path.relative_to(src).parent
        out.append((str(path), str(pathlib.PurePosixPath(dest) / rel)))
    return out


datas = [
    (str(REPO / "version.txt"), "."),
]
if (REPO / "ml-sharp" / "src").is_dir():
    datas += _tree_datas(REPO / "ml-sharp" / "src", "ml-sharp/src")
else:
    print("WARNING: ml-sharp/src missing - SHARP methods will not work in the bundle.")
    print("Run: git submodule update --init ml-sharp")
if (REPO / "static").is_dir():
    datas.append((str(REPO / "static"), "static"))

_taichi_kernels = REPO / "pystereo_core" / "stereo" / "_taichi_kernels.py"
if _taichi_kernels.is_file():
    datas.append((str(_taichi_kernels), "pystereo_core/stereo"))
_taichi_full_kernels = REPO / "pystereo_core" / "stereo" / "_taichi_full_kernels.py"
if _taichi_full_kernels.is_file():
    datas.append((str(_taichi_full_kernels), "pystereo_core/stereo"))

try:
    datas += copy_metadata("imageio")
except Exception:
    pass

_taichi_hidden: list[str] = []
_taichi_binaries: list = []
try:
    _taichi_hidden = collect_submodules("taichi")
    datas += collect_data_files("taichi")
    _taichi_binaries = collect_dynamic_libs("taichi")
except Exception:
    print("WARNING: taichi not collected - SHARP taichi methods will use torch.")

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
    + _taichi_hidden
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
        # Belt-and-braces: collect_submodules("pystereo_core.stereo")
        # above already yields these four. Listing them is safe even
        # without taichi installed - PyInstaller resolves hiddenimports
        # statically, so the kernel modules' top-level "import taichi"
        # only logs a missing-module warning, and taichi_render /
        # taichi_full import taichi lazily anyway.
        "pystereo_core.stereo._taichi_kernels",
        "pystereo_core.stereo._taichi_full_kernels",
        "pystereo_core.stereo.taichi_render",
        "pystereo_core.stereo.taichi_full",
        "imageio",
        "imageio.v2",
        "imageio.core",
        "imageio.plugins",
    ]
)

# Torchvision 0.29+ ops live in _C_stable / image_stable extensions loaded via
# torch.ops.load_library - PyInstaller does not discover them as imports.
_tv_binaries = collect_torchvision_binaries()
if not _tv_binaries:
    print("WARNING: no torchvision native libs found - stereo methods may fail at runtime.")

a = Analysis(
    [str(REPO / "packaging" / "entry_web.py")],
    pathex=[str(REPO)],
    binaries=merge_binaries(_taichi_binaries, _tv_binaries),
    datas=datas,
    hiddenimports=hiddenimports
    + [
        "torchvision",
        "torchvision.transforms",
        "torchvision.extension",
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
