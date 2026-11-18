"""Size control of the on-disk SHARP prediction cache, and timing notes.

Only the bookkeeping: no prediction is made and no checkpoint is needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pystereo_core.stereo import sharp_predict
from pystereo_core.stereo.timing import record_step

MB = 1024 * 1024


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty cache folder that is also the module's default one."""
    folder = tmp_path / "sharp-cache"
    folder.mkdir()
    monkeypatch.setattr(sharp_predict, "_DEFAULT_CACHE", folder)
    monkeypatch.setattr(sharp_predict, "_cache_max_bytes", 3 * MB)
    return folder


def _entry(cache: Path, name: str, megabytes: int, age_s: int) -> Path:
    """A cache file of ``megabytes``, last used ``age_s`` seconds ago."""
    path = cache / f"sharp_{name}.npz"
    path.write_bytes(b"\0" * (megabytes * MB))
    used = path.stat().st_mtime - age_s
    os.utime(path, (used, used))
    return path


def test_stats_count_only_cache_files(cache: Path) -> None:
    _entry(cache, "a", 1, age_s=10)
    _entry(cache, "b", 2, age_s=5)
    (cache / "notes.txt").write_text("not a prediction")
    stats = sharp_predict.cache_stats()
    assert stats["files"] == 2
    assert stats["bytes"] == 3 * MB
    assert stats["path"] == str(cache)
    assert stats["max_mb"] == 3


def test_a_cache_inside_its_limit_is_left_alone(cache: Path) -> None:
    _entry(cache, "a", 1, age_s=10)
    _entry(cache, "b", 2, age_s=5)
    assert sharp_predict.prune_cache() == 0
    assert len(list(cache.iterdir())) == 2


def test_the_least_recently_used_entries_go_first(cache: Path) -> None:
    oldest = _entry(cache, "oldest", 2, age_s=300)
    middle = _entry(cache, "middle", 2, age_s=200)
    newest = _entry(cache, "newest", 2, age_s=100)
    assert sharp_predict.prune_cache() == 2
    assert not oldest.exists()
    assert not middle.exists()
    assert newest.exists()


def test_the_entry_in_use_is_never_evicted(cache: Path) -> None:
    oldest = _entry(cache, "oldest", 2, age_s=300)
    newest = _entry(cache, "newest", 2, age_s=100)
    assert sharp_predict.prune_cache(keep=oldest) == 1
    assert oldest.exists()
    assert not newest.exists()


def test_a_limit_of_zero_switches_eviction_off(
    cache: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sharp_predict, "_cache_max_bytes", 0)
    _entry(cache, "a", 4, age_s=10)
    assert sharp_predict.prune_cache() == 0
    assert sharp_predict.get_cache_max_mb() == 0


def test_setting_the_limit_prunes_at_once(cache: Path) -> None:
    old = _entry(cache, "old", 1, age_s=200)
    new = _entry(cache, "new", 1, age_s=100)
    sharp_predict.set_cache_max_mb(1)
    assert sharp_predict.get_cache_max_mb() == 1
    assert not old.exists()
    assert new.exists()


def test_clearing_removes_predictions_and_nothing_else(cache: Path) -> None:
    _entry(cache, "a", 1, age_s=10)
    _entry(cache, "b", 1, age_s=5)
    other = cache / "notes.txt"
    other.write_text("not a prediction")
    assert sharp_predict.clear_cache() == {"deleted_bytes": 2 * MB, "deleted_files": 2}
    assert [p.name for p in cache.iterdir()] == ["notes.txt"]


def test_a_missing_cache_folder_is_an_empty_cache(tmp_path: Path) -> None:
    missing = tmp_path / "never-made"
    assert sharp_predict.cache_stats(missing)["files"] == 0
    assert sharp_predict.clear_cache(missing) == {"deleted_bytes": 0, "deleted_files": 0}


@pytest.mark.parametrize(
    ("intermediates", "note"),
    [
        (None, ""),
        ({}, ""),
        ({"sharp_cache": "miss"}, ""),
        ({"sharp_cache": "hit"}, " (cached)"),
        ({"sharp_cache": "off"}, " (cache off)"),
        ({"sharp_cache": "miss", "sharp_model_loaded": True}, " (model load)"),
        ({"sharp_cache": "off", "sharp_model_loaded": True}, " (cache off, model load)"),
    ],
)
def test_the_timing_label_says_where_the_prediction_came_from(
    intermediates: dict[str, object] | None, note: str,
) -> None:
    assert sharp_predict.cache_note(intermediates) == note


def test_step_timings_are_kept_in_order_and_rounded() -> None:
    intermediates: dict[str, object] = {}
    record_step(intermediates, "Depth estimation", 1.2345)
    record_step(intermediates, "Warp", 0.004)
    assert intermediates["timings"] == [("Depth estimation", 1.23), ("Warp", 0.0)]
    # Nothing to record into: not an error.
    record_step(None, "Warp", 1.0)
