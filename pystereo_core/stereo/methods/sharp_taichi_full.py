"""SHARP splat methods rendered entirely by Taichi kernels.

Unlike ``sharp_taichi`` / ``sharp_alpha_taichi`` (torch projection + taichi
compositing), these methods' whole render path - projection, tile binning,
compositing - runs outside torch (``stereo/taichi_full.py``). SHARP
prediction itself stays PyTorch. Two variants matching the two compositing
looks:

- ``sharp_alpha_full``: depth-sorted alpha compositing (the SHARP Alpha
  look) at standard 1536 resolution, pure splat colour.
- ``sharp_splat_full``: 2-pass z-buffer EWA splatting (the SHARP Splat
  look).

Both fall back to the equivalent torch-projection render path when taichi
is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.methods.sharp_splat import BASELINE_M, _SharpBase
from pystereo_core.stereo.splat_render import RenderMode

logger = logging.getLogger(__name__)


class _FullTaichiBase(_SharpBase):
    """Shared synthesize() for the full-taichi render variants."""

    #: Compositing rule for taichi_full.render_stereo_taichi.
    _taichi_mode: ClassVar[str] = "alpha"

    uses_taichi: ClassVar[bool] = True
    _detail_transfer: ClassVar[bool] = False
    _internal: ClassVar[int] = 1536

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pystereo_core.stereo.taichi_full import (
            is_full_taichi_available,
            render_stereo_taichi,
        )

        if not is_full_taichi_available():
            logger.info(
                "%s: taichi unavailable, falling back to %s render",
                self.name, self._render_mode,
            )
            return super().synthesize(image, fg_mask, settings, intermediates)

        from pystereo_core.stereo.sharp_predict import predict_gaussians

        npz_path = predict_gaussians(image, internal=self._internal)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        result = render_stereo_taichi(
            str(npz_path),
            baseline_m=BASELINE_M,
            converge_m=None,
            subject_mask=subject_mask,
            mode=self._taichi_mode,
        )

        if intermediates is not None:
            intermediates["splat_rgb"] = result["center_rgb"]
            intermediates["depth01"] = result["depth01"]

        notes = result.get("notes", {})
        logger.info(
            "SHARP full-taichi (%s): converge=%.2fm, disp=[%.1f, %.1f]px, "
            "holes L/R=%.3f/%.3f%%",
            self._taichi_mode,
            notes.get("converge_m", 0),
            notes.get("disp_px_far", 0),
            notes.get("disp_px_near", 0),
            notes.get("hole_pct_left", 0),
            notes.get("hole_pct_right", 0),
        )

        return result["left"], result["right"]


class SharpAlphaFullMethod(_FullTaichiBase):
    name: ClassVar[str] = "sharp_alpha_full"
    label: ClassVar[str] = "SHARP Alpha (full taichi)"
    description: ClassVar[str] = (
        "Depth-sorted alpha compositing (the SHARP Alpha look) with the "
        "entire render path - projection and compositing - as Taichi "
        "kernels on Metal/GPU, no torch in the renderer. Standard 1536 "
        "resolution, pure splat colour (no detail transfer). Both eyes "
        "render in about a second; SHARP prediction (~1 min) is the only "
        "remaining cost. Falls back to the alpha_taichi/torch path when "
        "taichi is unavailable. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Experimental all-Taichi renderer: same clean alpha-composited look "
        "as SHARP Alpha, at standard 1536 resolution and with the fastest "
        "render step. In the packaged Mac app this usually falls back to "
        "the torch renderer."
    )
    _taichi_mode: ClassVar[str] = "alpha"
    _render_mode: ClassVar[RenderMode] = "alpha_taichi"  # fallback path only


class SharpSplatFullMethod(_FullTaichiBase):
    name: ClassVar[str] = "sharp_splat_full"
    label: ClassVar[str] = "SHARP Splat (full taichi)"
    description: ClassVar[str] = (
        "Same z-buffer EWA splatting look as sharp_splat, with the entire "
        "render path - projection and compositing - as Taichi kernels on "
        "Metal/GPU, no torch in the renderer. Both eyes render in about a "
        "second; SHARP prediction (~1 min) is the only remaining cost. "
        "Falls back to the torch renderer when taichi is unavailable. "
        "Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Experimental all-Taichi renderer: same look as SHARP Splat with "
        "the fastest render step. In the packaged Mac app this usually "
        "falls back to the torch renderer."
    )
    _taichi_mode: ClassVar[str] = "zbuf"
    _render_mode: ClassVar[RenderMode] = "zbuf"  # fallback path only
