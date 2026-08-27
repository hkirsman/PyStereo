"""Pluggable stereo synthesis methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pystereo_core.stereo.methods.base import BaseStereoMethod

_REGISTRY: dict[str, type[BaseStereoMethod]] = {}


def _populate() -> None:
    if _REGISTRY:
        return
    from pystereo_core.stereo.methods.per_eye_inpaint import PerEyeInpaintMethod
    from pystereo_core.stereo.methods.bg_plate_fill import BgPlateFillMethod
    from pystereo_core.stereo.methods.routed_fill import RoutedFillMethod
    from pystereo_core.stereo.methods.direct_fill import DirectFillMethod
    from pystereo_core.stereo.methods.clean_fill import CleanFillMethod
    from pystereo_core.stereo.methods.combo_fill import ComboFillMethod
    from pystereo_core.stereo.methods.ldi_inpaint import LdiInpaintMethod
    from pystereo_core.stereo.methods.iterative_fill import IterativeFillMethod
    from pystereo_core.stereo.methods.fullres_warp import FullResWarpMethod
    from pystereo_core.stereo.methods.sharp_splat import (
        SharpDepthMethod,
        SharpDetailMethod,
        SharpHiresMethod,
        SharpAlphaMethod,
        SharpMeshMethod,
        SharpSplatMethod,
        SharpTaichiMethod,
    )

    _REGISTRY["per_eye_inpaint"] = PerEyeInpaintMethod
    _REGISTRY["fullres_warp"] = FullResWarpMethod
    _REGISTRY["bg_plate_fill"] = BgPlateFillMethod
    _REGISTRY["routed_fill"] = RoutedFillMethod
    _REGISTRY["direct_fill"] = DirectFillMethod
    _REGISTRY["clean_fill"] = CleanFillMethod
    _REGISTRY["combo_fill"] = ComboFillMethod
    _REGISTRY["ldi_inpaint"] = LdiInpaintMethod
    _REGISTRY["iterative_fill"] = IterativeFillMethod
    _REGISTRY["sharp_splat"] = SharpSplatMethod
    _REGISTRY["sharp_detail"] = SharpDetailMethod
    _REGISTRY["sharp_hires"] = SharpHiresMethod
    _REGISTRY["sharp_alpha"] = SharpAlphaMethod
    _REGISTRY["sharp_depth"] = SharpDepthMethod
    _REGISTRY["sharp_mesh"] = SharpMeshMethod
    _REGISTRY["sharp_taichi"] = SharpTaichiMethod


def get_method(name: str) -> BaseStereoMethod:
    """Instantiate a stereo method by name."""
    _populate()
    cls = _REGISTRY.get(name)
    if cls is None:
        avail = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown stereo method {name!r}; available: {avail}")
    return cls()


def available_methods() -> dict[str, type[BaseStereoMethod]]:
    """Return the registry of all available methods."""
    _populate()
    return dict(_REGISTRY)
