"""Method 5: Clean Fill — AOT-GAN per-eye inpainting (no LaMa workarounds).

Uses AOT-GAN (Apache 2.0) instead of LaMa for inpainting.  AOT-GAN uses
dilated convolutions (not FFCs) and sees the original image in masked
regions (not zero-filled), eliminating the two core LaMa issues:

- **No dark bias:** input is ``[image, mask]``, not ``image * (1 - mask)``
- **No global texture cloning:** dilated convolutions have a large but
  local receptive field — they can't reach distant textures

Because the model doesn't need workarounds, the pipeline is simpler:
hybrid warp → unilateral dilation → AOT-GAN inpaint per eye → Poisson
seam blend → Telea sweep.  No width routing, no banding, no bg plate.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import AotGanInpaintBackend, InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.routed_fill import _poisson_blend_wide, _unilateral_dilate
from pystereo_core.stereo.warp import hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)


class CleanFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "clean_fill"
    label: ClassVar[str] = "Clean Fill / AOT-GAN"
    deprecated: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Hybrid z-buffer warp with per-eye AOT-GAN inpainting.  "
        "AOT-GAN uses dilated convolutions (no FFC dark bias, no "
        "global texture cloning) so no width routing, banding, or "
        "bg-plate workarounds are needed.  Unilateral mask dilation "
        "toward foreground only + Poisson seam blending."
    )
    ui_info: ClassVar[str] = (
        "Deprecated. Per-eye AOT-GAN fill without LaMa dark-bias "
        "workarounds. Experimental alternate inpainter."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "inpaint_backend": "aotgan",
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
        "inpaint_mask_dilate_px": 4,
    }

    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(inpainter, AotGanInpaintBackend):
            logger.warning(
                "clean_fill is designed for AOT-GAN but got %s; "
                "proceeding anyway",
                type(inpainter).__name__,
            )

        left, left_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "left",
        )
        right, right_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "right",
        )

        dilate_px = settings.inpaint_mask_dilate_px
        left_mask = _unilateral_dilate(left_mask, "left", dilate_px)
        right_mask = _unilateral_dilate(right_mask, "right", dilate_px)

        total_px = int(np.count_nonzero(left_mask) + np.count_nonzero(right_mask))
        logger.info("Clean fill (AOT-GAN): %d total hole px", total_px)

        for eye_img, mask, eye_name in (
            (left, left_mask, "left"),
            (right, right_mask, "right"),
        ):
            if not np.any(mask):
                continue

            eye_before = eye_img.copy()
            filled = inpainter.inpaint(eye_img, mask)
            eye_img[:] = filled
            _poisson_blend_wide(eye_img, eye_before, mask)

        return left, right
