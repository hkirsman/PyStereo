"""HuggingFace Hub environment defaults, applied before the Hub is imported.

``huggingface_hub`` reads its settings into module constants at import time,
so every default here must be in ``os.environ`` before the first
``import huggingface_hub``. ``pystereo_core/__init__.py`` calls
:func:`configure_hf_env` as its very first statement for that reason.
"""

from __future__ import annotations

import os
import sys


def configure_hf_env() -> None:
    """Apply Hub defaults that PyStereo needs, without overriding the user."""
    if sys.platform == "win32":
        # The Hub cache stores one blob per file and points snapshots at it
        # with a symlink. Creating a symlink on Windows needs Developer Mode
        # or an elevated process, so an ordinary user account fails the whole
        # download with "[WinError 1314] A required privilege is not held by
        # the client". The Hub probes for symlink support itself, but the
        # probe races when files download in parallel and can report support
        # the individual downloads then do not have. Turning symlinks off up
        # front is deterministic: fresh blobs are moved into the snapshot
        # instead of linked, so a first download costs no extra disk.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
