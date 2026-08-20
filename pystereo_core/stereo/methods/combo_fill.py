"""Method 6: Combo Fill — per-strip AOT-GAN (best of Methods 4 + 5).

Combines the two strongest ideas:

- **From Method 4 (direct_fill):** per-strip anisotropic crop inpainting —
  each wide disocclusion strip gets a tight crop so the model only sees
  immediately adjacent background texture.
- **From Method 5 (clean_fill):** AOT-GAN instead of LaMa — no dark bias
  (model sees original pixels, not zero-filled) and no global texture
  cloning (dilated convolutions, not FFCs).

Narrow strips are still filled by CPU mirror propagation (exact
photographic texture, no model involved).  AOT-GAN on small crops
(~400×300 px) runs much faster than on full eye views (~5s vs ~50s).
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
_BORDER_PX = 4


def _fill_wide_direct(
    eye_img: np.ndarray,
    wide_mask: np.ndarray,
    inpainter: InpaintBackend,
) -> np.ndarray:
    """Per-component anisotropic crop inpainting for wide holes.

    Returns a mask of border-edge components that were skipped (left for
    Telea) because the crop would have no context on the image boundary.
    """
    h, w = eye_img.shape[:2]
    skipped = np.zeros((h, w), dtype=np.uint8)

    if not np.any(wide_mask):
        return skipped

    binary = (wide_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )
    if num_labels <= 1:
        return skipped

    src_img = eye_img.copy()
    n_filled = 0
    n_skipped = 0

    for i in range(1, num_labels):
        cx = stats[i, cv2.CC_STAT_LEFT]
        cy = stats[i, cv2.CC_STAT_TOP]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]

        touches_left = cx <= _BORDER_PX
        touches_right = (cx + cw) >= (w - _BORDER_PX)
        if touches_left or touches_right:
            component_pixels = labels == i
            skipped[component_pixels] = 255
            n_skipped += 1
            continue

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
        "Combo fill: inpainted %d wide components, skipped %d border components",
        n_filled, n_skipped,
    )
    return skipped


class ComboFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "combo_fill"
    label: ClassVar[str] = "Combo Fill / AOT-GAN (Deprecated)"
    description: ClassVar[str] = (
        "Best of Methods 4 + 5: per-strip anisotropic crop inpainting "
        "with AOT-GAN.  Narrow strips filled by CPU mirror (exact "
        "texture).  Wide strips get tight per-component crops fed to "
        "AOT-GAN (no dark bias, no FFC global cloning, fast on small "
        "crops).  Poisson seam blending + unilateral mask dilation."
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
            "Combo fill: %d narrow px, %d wide px (threshold=%d)",
            narrow_total, wide_total, threshold,
        )

        left_unfilled = _fill_narrow_strips(left, left_narrow, "left")
        right_unfilled = _fill_narrow_strips(right, right_narrow, "right")

        left_border = np.zeros_like(left_mask)
        right_border = np.zeros_like(right_mask)

        if wide_total > 0:
            for eye_img, wide_mask, border_out in (
                (left, left_wide, left_border),
                (right, right_wide, right_border),
            ):
                if not np.any(wide_mask):
                    continue
                eye_before = eye_img.copy()
                skipped = _fill_wide_direct(eye_img, wide_mask, inpainter)
                border_out[:] = skipped
                filled_wide = wide_mask.copy()
                filled_wide[skipped > 0] = 0
                if np.any(filled_wide):
                    _poisson_blend_wide(eye_img, eye_before, filled_wide)

        for eye_img, rem, border in (
            (left, left_unfilled, left_border),
            (right, right_unfilled, right_border),
        ):
            combined = cv2.bitwise_or(rem, border)
            if np.any(combined):
                bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
                eye_img[:] = cv2.cvtColor(
                    cv2.inpaint(bgr, combined, 15, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )

        return left, right
