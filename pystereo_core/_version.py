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


def _read_version() -> str:
    path = _version_file()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        # Both specs bundle version.txt, but a partial build should not stop
        # the app: the version only labels windows, logs and the About line.
        print(f"PyStereo: no version file at {path}", file=sys.stderr)
        return "0.0.0+unknown"


__version__: str = _read_version()
