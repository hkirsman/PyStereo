"""One-shot stderr (+ optional file) logging for CLI/GUI/web."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def ensure_stdio() -> None:
    """Restore stdout/stderr when a frozen ``console=False`` build left them as None.

    Libraries such as ``torch.hub.load_state_dict_from_url(..., progress=True)``
    write to ``sys.stdout`` and crash otherwise.
    """
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8", errors="replace")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8", errors="replace")


def pystereo_data_dir() -> Path:
    """Per-user data root (logs, outputs, settings) for PyStereo apps."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "PyStereo"
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        if local:
            return Path(local) / "PyStereo"
        return Path.home() / "AppData" / "Local" / "PyStereo"
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "pystereo"
    return Path.home() / ".local" / "share" / "pystereo"


def batch_log_path() -> Path:
    return pystereo_data_dir() / "logs" / "pystereo-batch.log"


def web_log_path() -> Path:
    return pystereo_data_dir() / "logs" / "pystereo-web.log"


def attach_file_handler(log: logging.Logger, log_file_name: str) -> Path | None:
    """Append a FileHandler to ``log`` if one for this path is not already present."""
    try:
        log_dir = pystereo_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file_name
    except OSError:
        return None

    resolved = str(log_path.resolve())
    for existing in log.handlers:
        if isinstance(existing, logging.FileHandler):
            try:
                if str(Path(existing.baseFilename).resolve()) == resolved:
                    return log_path
            except OSError:
                continue

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s:%(name)s:%(message)s")
    )
    log.addHandler(handler)
    log.info("Logging to %s", log_path)
    return log_path


def ensure_stderr_info_logging(*, log_file_name: str | None = None) -> Path | None:
    """Attach INFO handlers if the root logger is bare.

    When ``log_file_name`` is set (or the process is frozen), also append to a
    file under Application Support / LocalAppData so double-clicked apps leave
    a readable trail. Returns the log file path when file logging is enabled.
    """
    ensure_stdio()
    frozen = bool(getattr(sys, "frozen", False))
    name = log_file_name or ("pystereo-batch.log" if frozen else None)
    log_path: Path | None = None
    if name:
        try:
            log_dir = pystereo_data_dir() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / name
        except OSError:
            log_path = None

    root = logging.getLogger()
    if root.handlers:
        return log_path

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        handlers=handlers,
    )
    if log_path is not None:
        logging.getLogger(__name__).info("Logging to %s", log_path)
    return log_path
