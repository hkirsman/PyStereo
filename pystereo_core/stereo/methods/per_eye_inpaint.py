"""Method 1: Per-Eye Inpaint — inverse remap warp + independent LaMa fill.

Each eye is warped via pure inverse ``cv2.remap`` with stretch-based
occlusion detection, then inpainted independently.  Simple and fast,
but the two eyes may receive different hallucinated content in the
filled regions.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.warp import dilate_occlusion_mask, inverse_remap_warp_eye


class PerEyeInpaintMethod(BaseStereoMethod):
    name: ClassVar[str] = "per_eye_inpaint"
    label: ClassVar[str] = "Per-Eye Inpaint"
    description: ClassVar[str] = (
        "Inverse cv2.remap warp with stretch-based occlusion detection. "
        "Each eye is inpainted independently - fast but fills may differ "
        "between eyes."
    )
    ui_info: ClassVar[str] = (
        "About 15 seconds on a MacBook M4. Works well with empty rooms, "
        "landscapes, and tunnels. Can struggle with busy foregrounds and "
        "people - warping artifacts often show at cutoffs."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 1e-4,
        "depth_gamma": 1.2,
        "depth_healing_edge_blur_sigma": 12.0,
        "depth_healing_mask_dilate_px": 25,
    }

    def _warp_eyes(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Warp both eyes and dilate occlusion masks."""
        left, left_mask = inverse_remap_warp_eye(
            rgb_arr, depth_f32, max_disp, "left"
        )
        right, right_mask = inverse_remap_warp_eye(
            rgb_arr, depth_f32, max_disp, "right"
        )

        if settings.inpaint_mask_dilate_px > 0:
            left_mask = dilate_occlusion_mask(
                left_mask, settings.inpaint_mask_dilate_px
            )
            right_mask = dilate_occlusion_mask(
                right_mask, settings.inpaint_mask_dilate_px
            )

        return left, right, left_mask, right_mask

    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        left, right, left_mask, right_mask = self._warp_eyes(
            rgb_arr, depth_f32, max_disp, settings,
        )

        left = inpainter.inpaint(left, left_mask)
        right = inpainter.inpaint(right, right_mask)

        return left, right

    def warp_preview(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        left, right, left_mask, right_mask = self._warp_eyes(
            rgb_arr, depth_f32, max_disp, settings,
        )

        # Zero out masked pixels so disocclusion holes are visible as black
        # (cv2.remap fills them with stretched interpolation otherwise).
        left[left_mask > 0] = 0
        right[right_mask > 0] = 0

        return left, right, left_mask, right_mask
