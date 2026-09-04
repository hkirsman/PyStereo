"""Method 4: Direct Fill - per-strip anisotropic crop inpainting (no bg plate).

Routes disocclusion holes by width after unilateral mask dilation:

- **Narrow** (bounding-box width ≤ ``narrow_strip_max_px``): filled by
  mirroring adjacent background pixels (same as routed_fill).
- **Wide** (> threshold): each connected component is inpainted directly
  in the eye view via a tight anisotropic crop - wide horizontal context,
  minimal vertical padding.  LaMa sees only local texture and cannot
  clone distant materials.

Architecturally distinct from bg-plate methods (2 and 3): no background
plate is constructed or warped.  This eliminates plate-warp artifacts and
cannot fail for high-foreground images (>70 %).  Trade-off: each eye is
filled independently (potential stereo inconsistency), but tight local
crops limit LaMa's ability to hallucinate different content per eye.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.bg_plate_fill import _match_inpaint_color
from pystereo_core.stereo.methods.routed_fill import (
    _classify_holes,
    _fill_narrow_strips,
    _poisson_blend_wide,
    _unilateral_dilate,
)
from pystereo_core.stereo.warp import hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)

_H_PAD = 250
_V_PAD = 40


# ------------------------------------------------------------------
# Per-strip anisotropic crop inpainting
# ------------------------------------------------------------------

def _fill_wide_direct(
    eye_img: np.ndarray,
    wide_mask: np.ndarray,
    inpainter: InpaintBackend,
) -> None:
    """In-place per-component anisotropic crop inpainting for wide holes.

    Each connected component gets its own tight crop (wide horizontal
    context, minimal vertical padding).  LaMa's FFCs can only see
    textures within the crop - preventing cross-material cloning.
    Per-component colour matching is applied before paste-back.
    """
    if not np.any(wide_mask):
        return

    h, w = eye_img.shape[:2]
    binary = (wide_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    if num_labels <= 1:
        return

    src_img = eye_img.copy()
    n_filled = 0

    for i in range(1, num_labels):
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        crop_x0 = max(0, cx - _H_PAD)
        crop_y0 = max(0, cy - _V_PAD)
        crop_x1 = min(w, cx + cw + _H_PAD)
        crop_y1 = min(h, cy + ch + _V_PAD)

        crop_h = crop_y1 - crop_y0
        crop_w = crop_x1 - crop_x0

        crop_img = src_img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

        component_local = labels[crop_y0:crop_y1, crop_x0:crop_x1] == i
        crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        crop_mask[component_local] = 255

        if not np.any(crop_mask):
            continue

        crop_result = inpainter.inpaint(crop_img, crop_mask)

        if crop_result.shape[:2] != (crop_h, crop_w):
            rh, rw = crop_result.shape[:2]
            if rh >= crop_h and rw >= crop_w:
                crop_result = crop_result[:crop_h, :crop_w]
            else:
                crop_result = cv2.resize(
                    crop_result, (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR,
                )

        crop_result = _match_inpaint_color(crop_img, crop_result, crop_mask)

        eye_img[crop_y0:crop_y1, crop_x0:crop_x1][component_local] = \
            crop_result[component_local]
        n_filled += 1

    logger.debug(
        "Direct fill: inpainted %d wide components (crops %dx%d avg)",
        n_filled,
        int(np.mean([
            min(w, stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + _H_PAD)
            - max(0, stats[i, cv2.CC_STAT_LEFT] - _H_PAD)
            for i in range(1, num_labels)
        ])) if num_labels > 1 else 0,
        int(np.mean([
            min(h, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + _V_PAD)
            - max(0, stats[i, cv2.CC_STAT_TOP] - _V_PAD)
            for i in range(1, num_labels)
        ])) if num_labels > 1 else 0,
    )


# ------------------------------------------------------------------
# Method class
# ------------------------------------------------------------------

class DirectFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "direct_fill"
    label: ClassVar[str] = "Direct Anisotropic Fill"
    deprecated: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Hybrid z-buffer warp with width-routed disocclusion fill.  "
        "Narrow strips (≤ threshold) are mirror-filled from adjacent "
        "background pixels (CPU, zero dark bias).  Wide regions are "
        "filled directly in each eye view via per-strip anisotropic "
        "crop inpainting (no background plate).  Each strip gets a "
        "tight horizontal crop - LaMa sees only local texture.  "
        "Poisson seam blending + unilateral mask dilation."
    )
    ui_info: ClassVar[str] = (
        "Deprecated. Mirror-fills narrow gaps; inpaints wide ones "
        "per-eye with tight crops. No shared background plate."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
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
        h, w = rgb_arr.shape[:2]
        threshold = settings.narrow_strip_max_px

        left, left_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "left",
        )
        right, right_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "right",
        )

        dilate_px = settings.inpaint_mask_dilate_px
        left_mask = _unilateral_dilate(left_mask, "left", dilate_px)
        right_mask = _unilateral_dilate(right_mask, "right", dilate_px)

        left_narrow, left_wide = _classify_holes(left_mask, threshold)
        right_narrow, right_wide = _classify_holes(right_mask, threshold)

        narrow_total = int(
            np.count_nonzero(left_narrow) + np.count_nonzero(right_narrow)
        )
        wide_total = int(
            np.count_nonzero(left_wide) + np.count_nonzero(right_wide)
        )
        logger.info(
            "Direct fill: %d narrow px, %d wide px (threshold=%d)",
            narrow_total, wide_total, threshold,
        )

        left_unfilled = _fill_narrow_strips(left, left_narrow, "left")
        right_unfilled = _fill_narrow_strips(right, right_narrow, "right")

        if wide_total > 0:
            for eye_img, wide_mask, eye_name in (
                (left, left_wide, "left"),
                (right, right_wide, "right"),
            ):
                if not np.any(wide_mask):
                    continue
                eye_before = eye_img.copy()
                _fill_wide_direct(eye_img, wide_mask, inpainter)
                _poisson_blend_wide(eye_img, eye_before, wide_mask)

        for eye_img, rem in (
            (left, left_unfilled),
            (right, right_unfilled),
        ):
            if np.any(rem):
                bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
                eye_img[:] = cv2.cvtColor(
                    cv2.inpaint(bgr, rem, 15, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )

        return left, right
