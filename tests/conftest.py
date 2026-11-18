"""Shared fixtures.

Nothing here may read or write the real ``settings.json``, ``outputs/``,
``.sharp_cache/`` or log files: every path the code under test would use is
pointed at a temporary folder before the code runs.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # ``app.py`` sits at the repo root and is not an installed package.
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def app_module(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    """``app.py``, imported once with its log files in a temporary folder.

    Importing it attaches a log file under the per-user data folder, so that
    folder is swapped before the import.
    """
    from pystereo_core import logging_config

    data_dir = tmp_path_factory.mktemp("pystereo-data")
    patcher = pytest.MonkeyPatch()
    patcher.setattr(logging_config, "pystereo_data_dir", lambda: data_dir)
    import app

    yield app
    patcher.undo()


@pytest.fixture
def web(
    app_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    """``app.py`` with its settings, results and caches in ``tmp_path``.

    The integration service starts switched off, as on a fresh install.
    """
    from pystereo_core.stereo import sharp_predict

    outputs = tmp_path / "outputs"
    outputs.mkdir()
    sharp_cache = tmp_path / "sharp-cache"
    sharp_cache.mkdir()
    monkeypatch.setattr(app_module, "OUTPUTS_DIR", outputs)
    monkeypatch.setattr(app_module, "_settings_path", lambda: tmp_path / "settings.json")
    # Saving a cache limit prunes the SHARP cache at once.
    monkeypatch.setattr(sharp_predict, "_DEFAULT_CACHE", sharp_cache)
    monkeypatch.setattr(sharp_predict, "_cache_max_bytes", sharp_predict._cache_max_bytes)
    monkeypatch.setattr(sharp_predict, "IDLE_UNLOAD_S", sharp_predict.IDLE_UNLOAD_S)
    monkeypatch.setattr(app_module, "_cache_settings_applied", False)
    # No pipeline: a saved setting must not reach one left by another test.
    monkeypatch.setattr(app_module, "_pipeline_singleton", None)
    monkeypatch.setattr(app_module, "_service_forced", False)
    monkeypatch.delenv("PYSTEREO_SERVICE", raising=False)
    return app_module


@pytest.fixture
def client(web: ModuleType) -> Any:
    return web.app.test_client()


@pytest.fixture
def service_on(web: ModuleType, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The integration service switched on, as ``PYSTEREO_SERVICE=1`` does."""
    monkeypatch.setenv("PYSTEREO_SERVICE", "1")
    return web
