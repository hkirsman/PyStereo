from pathlib import Path

__version__ = (Path(__file__).resolve().parent.parent / "version.txt").read_text().strip()
