"""Configuration for AI stereo synthesis (2D → SBS)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

InpaintBackendName = Literal["lama", "opencv", "none", "flux", "aotgan"]
StereoMethodName = Literal["per_eye_inpaint", "fullres_warp", "bg_plate_fill", "routed_fill", "direct_fill", "clean_fill", "combo_fill", "ldi_inpaint", "iterative_fill", "sharp_splat", "sharp_detail", "sharp_hires", "sharp_alpha", "sharp_alpha_taichi", "sharp_depth", "sharp_mesh", "sharp_taichi"]
DepthModelSize = Literal["small", "base", "large"]

DEFAULT_METHOD: StereoMethodName = "per_eye_inpaint"
DEFAULT_DEPTH_MODEL: DepthModelSize = "small"


@dataclass(frozen=True)
class StereoSettings:
    """Tunable stereo synthesis parameters.

    Defaults here are a neutral baseline.  Each stereo method declares
    ``SETTING_OVERRIDES`` which are applied on top — see
    :func:`from_env` and :meth:`with_method_defaults`.
    """

    stereo_method: StereoMethodName = DEFAULT_METHOD

    divergence_ratio: float = 0.030
    min_divergence_ratio: float = 0.015
    max_divergence_ratio: float = 0.060
    adaptive_depth: bool = True
    max_processing_dim: int = 2048
    inpaint_backend: InpaintBackendName = "lama"
    inpaint_mask_dilate_px: int = 3
    guided_filter_radius: int = 8
    guided_filter_eps: float = 1e-4
    depth_gamma: float = 1.2
    depth_healing: bool = True
    depth_healing_bg_threshold: float = 0.75
    depth_healing_edge_blur_sigma: float = 12.0
    depth_healing_mask_dilate_px: int = 25
    segmenter_padding: int = 200
    narrow_strip_max_px: int = 12
    bg_plate_tight_dilate_px: int = 10
    #: Read SHARP predictions back from ``.sharp_cache``. ``False`` re-runs
    #: the network every time (benchmarking); the resident predictor and the
    #: model weights are unaffected.
    sharp_disk_cache: bool = True

    def with_method_defaults(self) -> StereoSettings:
        """Return a copy with method-specific overrides applied.

        Method overrides (``SETTING_OVERRIDES``) are applied unconditionally.
        """
        from pystereo_core.stereo.methods import available_methods

        registry = available_methods()
        cls = registry.get(self.stereo_method)
        if cls is None:
            return self
        overrides = cls.SETTING_OVERRIDES
        if not overrides:
            return self
        valid = {f.name for f in fields(self)}
        filtered = {k: v for k, v in overrides.items() if k in valid}
        if not filtered:
            return self
        return replace(self, **filtered)

    @classmethod
    def from_env(cls, *, method: StereoMethodName | None = None) -> StereoSettings:
        """Build settings from environment variables.

        Apply primary env vars, then method ``SETTING_OVERRIDES``, then
        optional tuning env overrides.
        """
        method_raw = method or os.environ.get("PYSTEREO_METHOD", "").strip().lower()
        stereo_method: StereoMethodName = DEFAULT_METHOD
        if method_raw in ("per_eye_inpaint", "fullres_warp", "bg_plate_fill", "routed_fill", "direct_fill", "clean_fill", "combo_fill", "ldi_inpaint", "iterative_fill", "sharp_splat", "sharp_detail", "sharp_hires", "sharp_alpha", "sharp_alpha_taichi", "sharp_depth", "sharp_mesh", "sharp_taichi"):
            stereo_method = method_raw  # type: ignore[assignment]

        backend_raw = os.environ.get("PYSTEREO_INPAINT", "lama").strip().lower()
        backend: InpaintBackendName = "lama"
        if backend_raw in ("lama", "opencv", "none", "flux", "aotgan"):
            backend = backend_raw  # type: ignore[assignment]

        divergence_pct = os.environ.get("PYSTEREO_DIVERGENCE")
        divergence_ratio = 0.030
        if divergence_pct:
            try:
                val = float(divergence_pct)
                divergence_ratio = val / 100.0 if val > 1.0 else val
            except ValueError:
                pass

        max_dim_raw = os.environ.get("PYSTEREO_MAX_DIM", "2048")
        try:
            max_processing_dim = max(512, int(max_dim_raw))
        except ValueError:
            max_processing_dim = 2048

        heal_raw = os.environ.get("PYSTEREO_HEAL", "1").strip().lower()
        depth_healing = heal_raw not in ("0", "false", "no", "off")

        cache_raw = os.environ.get("PYSTEREO_SHARP_CACHE", "1").strip().lower()
        sharp_disk_cache = cache_raw not in ("0", "false", "no", "off")

        # Build with neutral defaults, then apply method-specific overrides,
        # then apply any explicit env-var overrides on top.
        base = cls(
            stereo_method=stereo_method,
            divergence_ratio=divergence_ratio,
            inpaint_backend=backend,
            max_processing_dim=max_processing_dim,
            depth_healing=depth_healing,
            sharp_disk_cache=sharp_disk_cache,
        )
        base = base.with_method_defaults()

        # Env-var overrides (only applied when explicitly set)
        overrides: dict[str, Any] = {}

        guided_eps_raw = os.environ.get("PYSTEREO_GUIDED_EPS")
        if guided_eps_raw:
            try:
                overrides["guided_filter_eps"] = max(1e-6, float(guided_eps_raw))
            except ValueError:
                pass

        depth_gamma_raw = os.environ.get("PYSTEREO_DEPTH_GAMMA")
        if depth_gamma_raw:
            try:
                overrides["depth_gamma"] = max(0.1, float(depth_gamma_raw))
            except ValueError:
                pass

        narrow_px_raw = os.environ.get("PYSTEREO_NARROW_PX")
        if narrow_px_raw:
            try:
                overrides["narrow_strip_max_px"] = max(1, int(narrow_px_raw))
            except ValueError:
                pass

        if overrides:
            base = replace(base, **overrides)

        return base
