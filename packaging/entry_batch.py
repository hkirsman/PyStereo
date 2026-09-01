"""PyInstaller entrypoint: run ``python -m pystereo_core`` (GUI or --cli)."""

from __future__ import annotations

import multiprocessing


def main() -> None:
    import logging

    from pystereo_core._version import __version__
    from pystereo_core.logging_config import ensure_stderr_info_logging, ensure_stdio

    ensure_stdio()
    ensure_stderr_info_logging(log_file_name="pystereo-batch.log")
    logging.getLogger("PyStereo").info("PyStereo batch %s", __version__)

    from pystereo_core.__main__ import main as batch_main

    batch_main()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
