"""PyInstaller entrypoint: run the Flask web UI (``app.py``)."""

from __future__ import annotations

import logging
import multiprocessing


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    from pystereo_core._version import __version__
    from app import app

    parser = argparse.ArgumentParser(description="PyStereo web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No launch dialog or browser; log the URL and run the server only",
    )
    args = parser.parse_args()

    logging.getLogger("PyStereo").info(
        "PyStereo web %s - http://%s:%d", __version__, args.host, args.port
    )

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
