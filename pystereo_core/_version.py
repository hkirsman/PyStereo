import sys
from pathlib import Path

# Single source of truth for PyStereo (web + batch) - edit version.txt at
# the repo root to bump the version everywhere at once.


def _version_file() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        base = Path(meipass) if meipass else Path(sys.executable).resolve().parent / "_internal"
        return base / "version.txt"
    return Path(__file__).resolve().parent.parent / "version.txt"


__version__: str = _version_file().read_text(encoding="utf-8").strip()
