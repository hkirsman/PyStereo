"""``StereoSettings``: environment overrides and per-method defaults."""

from __future__ import annotations

import os
from dataclasses import fields

import pytest

from pystereo_core.stereo.config import DEFAULT_METHOD, StereoSettings
from pystereo_core.stereo.methods import available_methods


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``PYSTEREO_*`` variable from the shell reaches these tests."""
    for name in list(os.environ):
        if name.startswith("PYSTEREO_"):
            monkeypatch.delenv(name)


def test_with_nothing_set_the_default_method_and_its_defaults_apply() -> None:
    settings = StereoSettings.from_env()
    assert settings.stereo_method == DEFAULT_METHOD
    assert settings == StereoSettings().with_method_defaults()


def test_the_method_comes_from_the_argument_before_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSTEREO_METHOD", "sharp_splat")
    assert StereoSettings.from_env().stereo_method == "sharp_splat"
    assert StereoSettings.from_env(method="clean_fill").stereo_method == "clean_fill"


def test_an_unknown_method_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSTEREO_METHOD", "no_such_method")
    assert StereoSettings.from_env().stereo_method == DEFAULT_METHOD


@pytest.mark.parametrize("name", sorted(available_methods()))
def test_each_method_gets_its_own_setting_overrides(name: str) -> None:
    settings = StereoSettings.from_env(method=name)  # type: ignore[arg-type]
    valid = {f.name for f in fields(StereoSettings)}
    for key, value in available_methods()[name].SETTING_OVERRIDES.items():
        if key in valid:
            assert getattr(settings, key) == value, key


@pytest.mark.parametrize("name", sorted(available_methods()))
def test_method_overrides_name_only_real_settings(name: str) -> None:
    valid = {f.name for f in fields(StereoSettings)}
    unknown = set(available_methods()[name].SETTING_OVERRIDES) - valid
    assert not unknown, f"{name} overrides settings that do not exist: {unknown}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4096", 4096), ("100", 512), ("not-a-number", 2048)],
)
def test_max_dim_has_a_floor_and_survives_nonsense(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: int,
) -> None:
    monkeypatch.setenv("PYSTEREO_MAX_DIM", raw)
    assert StereoSettings.from_env().max_processing_dim == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("4", 0.04), ("0.04", 0.04), ("nonsense", 0.030)],
)
def test_divergence_is_read_as_a_percentage_or_a_ratio(
    monkeypatch: pytest.MonkeyPatch, raw: str, expected: float,
) -> None:
    monkeypatch.setenv("PYSTEREO_DIVERGENCE", raw)
    assert StereoSettings.from_env().divergence_ratio == pytest.approx(expected)


@pytest.mark.parametrize("off", ["0", "false", "no", "off", "OFF"])
def test_healing_and_the_sharp_cache_can_be_switched_off(
    monkeypatch: pytest.MonkeyPatch, off: str,
) -> None:
    monkeypatch.setenv("PYSTEREO_HEAL", off)
    monkeypatch.setenv("PYSTEREO_SHARP_CACHE", off)
    settings = StereoSettings.from_env()
    assert settings.depth_healing is False
    assert settings.sharp_disk_cache is False


def test_an_unknown_inpaint_backend_falls_back_to_lama(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSTEREO_INPAINT", "opencv")
    assert StereoSettings.from_env().inpaint_backend == "opencv"
    monkeypatch.setenv("PYSTEREO_INPAINT", "photoshop")
    assert StereoSettings.from_env().inpaint_backend == "lama"


def test_tuning_variables_win_over_the_method_and_keep_their_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYSTEREO_DEPTH_GAMMA", "0.01")
    monkeypatch.setenv("PYSTEREO_NARROW_PX", "0")
    monkeypatch.setenv("PYSTEREO_GUIDED_EPS", "0.5")
    settings = StereoSettings.from_env()
    assert settings.depth_gamma == pytest.approx(0.1)
    assert settings.narrow_strip_max_px == 1
    assert settings.guided_filter_eps == pytest.approx(0.5)
