"""Locate and import Apple ml-sharp in dev and PyInstaller bundles."""

from __future__ import annotations

import sys
from pathlib import Path


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent


ML_SHARP_SRC = _bundle_root() / "ml-sharp" / "src"


def is_sharp_code_available() -> bool:
    """True when the ml-sharp source tree is present on disk."""
    return ML_SHARP_SRC.is_dir() and (ML_SHARP_SRC / "sharp").is_dir()


def ensure_sharp_imports() -> None:
    """Add ml-sharp to ``sys.path`` and verify ``import sharp`` works."""
    if not is_sharp_code_available():
        raise RuntimeError(
            f"ml-sharp is not available at {ML_SHARP_SRC}. "
            "From the repo root run: git submodule update --init ml-sharp"
        )
    src = str(ML_SHARP_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    import sharp  # noqa: F401
