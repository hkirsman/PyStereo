"""Collect torchvision native extensions that PyInstaller misses.

Torchvision 0.29+ ships ops as ``_C_stable`` / ``image_stable`` extension
modules loaded via ``torch.ops.load_library``, not as regular Python imports.
PyInstaller's analysis therefore ships the pure-Python package without those
``.pyd`` / ``.so`` files (or their sibling DLLs). Importing torchvision then
fails with ``operator torchvision::nms does not exist`` because the C++ ops
never register.

Both ``pystereo_web.spec`` and ``pystereo_batch.spec`` call
:func:`collect_torchvision_binaries` and pass the result to ``Analysis``.
"""

from __future__ import annotations

import pathlib
from typing import Any


def collect_torchvision_binaries() -> list[tuple[str, str]]:
    """Return ``(src, dest_dir)`` pairs for torchvision native libs."""
    try:
        import torchvision
    except ImportError:
        return []

    root = pathlib.Path(torchvision.__file__).resolve().parent
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in ("*.pyd", "*.dll", "*.so", "*.dylib"):
        for path in root.glob(pattern):
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((str(path), "torchvision"))
    return out


def merge_binaries(
    existing: list[Any] | None,
    extra: list[tuple[str, str]],
) -> list[Any]:
    """Append *extra* binaries, skipping duplicates by destination name."""
    merged: list[Any] = list(existing or [])
    have = {pathlib.Path(str(item[0])).name.lower() for item in merged}
    for src, dest in extra:
        name = pathlib.Path(src).name.lower()
        if name in have:
            continue
        have.add(name)
        merged.append((src, dest))
    return merged
