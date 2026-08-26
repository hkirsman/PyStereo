"""Configuration for AI stereo synthesis (2D → SBS)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

InpaintBackendName = Literal["lama", "opencv", "none", "flux", "aotgan"]
StereoMethodName = Literal[
    "per_eye_inpaint", "fullres_warp", "anchored_left", "anchored_right",
    "cutout_layers", "bg_plate_fill", "routed_fill", "direct_fill",
    "clean_fill", "combo_fill", "ldi_inpaint", "iterative_fill",
]
DepthModelSize = Literal["small", "base", "large"]

DEFAULT_METHOD: StereoMethodName = "per_eye_inpaint"
DEFAULT_DEPTH_MODEL: DepthModelSize = "small"

METHOD_NAMES: tuple[str, ...] = (
    "per_eye_inpaint", "fullres_warp", "anchored_left", "anchored_right",
    "cutout_layers", "bg_plate_fill", "routed_fill", "direct_fill",
    "clean_fill", "combo_fill", "ldi_inpaint", "iterative_fill",
)
INPAINT_BACKEND_NAMES: tuple[str, ...] = (
    "lama", "opencv", "none", "flux", "aotgan",
)


def env_tuning_overrides() -> dict[str, Any]:
    """Return tuning fields that were explicitly set via environment vars.

    Kept separate from :meth:`StereoSettings.from_env` so the same values
    can be re-applied on top of a method's ``SETTING_OVERRIDES`` whenever
    the method changes — see :meth:`StereoSettings.resolved_for`.  An
    explicit env var always wins over a method default.
    """
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

    residual_raw = os.environ.get("PYSTEREO_TELEA_RESIDUAL")
    if residual_raw:
        overrides["fill_telea_residual"] = (
            residual_raw.strip().lower() not in ("0", "false", "no", "off")
        )

    crack_raw = os.environ.get("PYSTEREO_CRACK_FILL")
    if crack_raw:
        try:
            overrides["warp_crack_fill_px"] = max(0, int(crack_raw))
        except ValueError:
            pass

    uni_raw = os.environ.get("PYSTEREO_UNILATERAL_DILATE")
    if uni_raw:
        overrides["unilateral_mask_dilate"] = (
            uni_raw.strip().lower() not in ("0", "false", "no", "off")
        )

    plate_dilate_raw = os.environ.get("PYSTEREO_PLATE_DILATE")
    if plate_dilate_raw:
        try:
            overrides["bg_plate_dilate_max_px"] = max(0, int(plate_dilate_raw))
        except ValueError:
            pass

    tight_dilate_raw = os.environ.get("PYSTEREO_TIGHT_DILATE")
    if tight_dilate_raw:
        try:
            overrides["bg_plate_tight_dilate_px"] = max(0, int(tight_dilate_raw))
        except ValueError:
            pass

    return overrides


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
    # The background plate erases the subject plus a margin so the inpainter
    # has room to work.  The margin is derived from the disparity
    # (``max_disp * 0.6 + 3``), which at large disparities grows far wider
    # than the disocclusion it needs to cover and destroys real background
    # near the subject — thin gaps between limb and torso, ground lines
    # passing behind them.  This caps it.  0 disables the cap.
    bg_plate_dilate_max_px: int = 24
    # Widest horizontal gap in the forward-splat z-buffer still treated as
    # a sampling crack rather than a disocclusion.  0 disables.
    warp_crack_fill_px: int = 2
    # Dilate hole masks toward the background side only, so the margin
    # given to the inpainter is not taken out of the foreground.
    unilateral_mask_dilate: bool = True
    # After compositing the background plate, Telea-sweep the pixels the
    # plate warp itself could not fill.  That mask reaches ~96% of the
    # disocclusion, so the sweep discards the plate's inpainted texture
    # and replaces it with diffusion smear.  Off: sweep only the seam.
    fill_telea_residual: bool = False

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

    def resolved_for(self, method: StereoMethodName | None) -> StereoSettings:
        """Return settings with *method* active and its overrides applied.

        A per-call method override must bring its own ``SETTING_OVERRIDES``
        with it; otherwise the method's warp/fill code runs against whatever
        method was active when the pipeline was constructed.  Explicit env
        tuning is re-applied last so it still wins.
        """
        if method is None or method == self.stereo_method:
            return self
        base = replace(self, stereo_method=method).with_method_defaults()
        env = env_tuning_overrides()
        return replace(base, **env) if env else base

    @classmethod
    def from_env(cls, *, method: StereoMethodName | None = None) -> StereoSettings:
        """Build settings from environment variables.

        Apply primary env vars, then method ``SETTING_OVERRIDES``, then
        optional tuning env overrides.
        """
        method_raw = method or os.environ.get("PYSTEREO_METHOD", "").strip().lower()
        stereo_method: StereoMethodName = DEFAULT_METHOD
        if method_raw in METHOD_NAMES:
            stereo_method = method_raw  # type: ignore[assignment]

        backend_raw = os.environ.get("PYSTEREO_INPAINT", "lama").strip().lower()
        backend: InpaintBackendName = "lama"
        if backend_raw in INPAINT_BACKEND_NAMES:
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

        # Build with neutral defaults, then apply method-specific overrides,
        # then apply any explicit env-var overrides on top.
        base = cls(
            stereo_method=stereo_method,
            divergence_ratio=divergence_ratio,
            inpaint_backend=backend,
            max_processing_dim=max_processing_dim,
            depth_healing=depth_healing,
        )
        base = base.with_method_defaults()

        overrides = env_tuning_overrides()
        if overrides:
            base = replace(base, **overrides)

        return base
