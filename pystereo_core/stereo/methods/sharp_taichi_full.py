"""SHARP splat method rendered entirely by Taichi kernels.

Unlike ``sharp_taichi`` / ``sharp_alpha_taichi`` (torch projection + taichi
compositing), this method's whole render path - projection, tile binning,
alpha compositing - runs outside torch (``stereo/taichi_full.py``). SHARP
prediction itself stays PyTorch. Falls back to the alpha_taichi render path
(and from there to torch) when taichi is unavailable.
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


class SharpTaichiFullMethod(_SharpBase):
    name: ClassVar[str] = "sharp_taichi_full"
    label: ClassVar[str] = "SHARP Splat (full taichi)"
    description: ClassVar[str] = (
        "Same SHARP Gaussian scene as sharp_splat, but the entire render "
        "path - projection and depth-sorted alpha compositing - runs as "
        "Taichi kernels on Metal/GPU, with no torch in the renderer. Both "
        "eyes render in about a second; SHARP prediction (~1 min) is the "
        "only remaining cost. Falls back to the alpha_taichi/torch path "
        "when taichi is unavailable. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Experimental all-Taichi renderer: same clean alpha-composited look "
        "as SHARP Alpha, at standard 1536 resolution and with the fastest "
        "render step. In the packaged Mac app this usually falls back to "
        "the torch renderer."
    )
    uses_taichi: ClassVar[bool] = True
    _detail_transfer: ClassVar[bool] = False
    _internal: ClassVar[int] = 1536
    _render_mode: ClassVar[RenderMode] = "alpha_taichi"  # fallback path only

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
                "sharp_taichi_full: taichi unavailable, falling back to %s render",
                self._render_mode,
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
        )

        if intermediates is not None:
            intermediates["splat_rgb"] = result["center_rgb"]
            intermediates["depth01"] = result["depth01"]

        notes = result.get("notes", {})
        logger.info(
            "SHARP full-taichi: converge=%.2fm, disp=[%.1f, %.1f]px, holes L/R=%.3f/%.3f%%",
            notes.get("converge_m", 0),
            notes.get("disp_px_far", 0),
            notes.get("disp_px_near", 0),
            notes.get("hole_pct_left", 0),
            notes.get("hole_pct_right", 0),
        )

        return result["left"], result["right"]
