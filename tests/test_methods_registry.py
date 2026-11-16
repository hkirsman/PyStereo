"""The stereo method registry and the order the web UI lists it in."""

from __future__ import annotations

from typing import get_args

import pytest

from pystereo_core.stereo.config import DEFAULT_METHOD, StereoMethodName
from pystereo_core.stereo.methods import (
    METHOD_UI_ORDER,
    available_methods,
    get_method,
    list_methods_for_ui,
)
from pystereo_core.stereo.methods.base import BaseStereoMethod


def test_the_ui_order_lists_every_method_exactly_once() -> None:
    assert len(METHOD_UI_ORDER) == len(set(METHOD_UI_ORDER))
    assert set(METHOD_UI_ORDER) == set(available_methods())


def test_the_type_hint_names_the_same_methods_as_the_registry() -> None:
    assert set(get_args(StereoMethodName)) == set(available_methods())


def test_the_default_method_exists_and_is_not_deprecated() -> None:
    assert available_methods()[DEFAULT_METHOD].deprecated is False


@pytest.mark.parametrize("name", sorted(available_methods()))
def test_a_method_is_registered_under_its_own_name(name: str) -> None:
    cls = available_methods()[name]
    assert cls.name == name
    assert cls.label.strip()
    assert cls.description.strip()
    assert isinstance(get_method(name), BaseStereoMethod)


@pytest.mark.parametrize("name", sorted(available_methods()))
def test_a_method_without_a_depth_map_synthesizes_by_itself(name: str) -> None:
    cls = available_methods()[name]
    if cls.needs_depth:
        return
    assert cls.synthesize is not BaseStereoMethod.synthesize


def test_deprecated_methods_come_last_in_the_ui() -> None:
    flags = [cls.deprecated for _, cls in list_methods_for_ui()]
    assert flags == sorted(flags)
    assert len(flags) == len(available_methods())


def test_the_ui_keeps_the_declared_order_within_each_group() -> None:
    for deprecated in (False, True):
        shown = [n for n, cls in list_methods_for_ui() if cls.deprecated is deprecated]
        declared = [n for n in METHOD_UI_ORDER if n in shown]
        assert shown == declared


def test_an_unknown_method_is_refused_with_the_names_that_exist() -> None:
    with pytest.raises(ValueError, match="per_eye_inpaint"):
        get_method("no_such_method")


def test_the_registry_handed_out_is_a_copy() -> None:
    available_methods().clear()
    assert available_methods()
