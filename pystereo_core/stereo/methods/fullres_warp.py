"""Method: Per-Eye Inpaint (Full-Res) — sharp warped pixels with downscaled inpaint fills.

Full-resolution variant of :class:`~pystereo_core.stereo.methods.per_eye_inpaint.PerEyeInpaintMethod`.
Warps at full source resolution (cheap numpy/cv2 math), then only
downscales the warped image + occlusion mask to processing resolution
for LaMa inpainting.  The inpainted fills are upscaled back and
composited into the full-res warped image through a feathered occlusion
mask, giving a smooth sharpness transition at fill boundaries.

Uses the background-plate strategy for stereo-consistent fills (same as
:class:`~pystereo_core.stereo.methods.bg_plate_fill.BgPlateFillMethod`).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.bg_plate_fill import (
    BgPlateFillMethod,
    _match_inpaint_color,
    _sharpen_inpainted_region,
)
from pystereo_core.stereo.warp import dilate_occlusion_mask, hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)

_EDGE_TELEA_PX = 2
_AA_SIGMA = 0.8


def _inward_alpha(mask_u8: np.ndarray) -> np.ndarray:
    """Build a [0, 1] float32 alpha that anti-aliases the hole boundary inward.

    A tiny gaussian blur softens jagged mask edges, then the result is
    clamped to the original mask so alpha is strictly 0.0 outside the
    hole.  This prevents any blurry fill from bleeding into the sharp
    warped region.
    """
    mask_f = mask_u8.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(mask_f, (0, 0), _AA_SIGMA)
    np.minimum(blurred, mask_f, out=blurred)
    return blurred


def _downscale_for_inpaint(
    img: np.ndarray,
    mask_u8: np.ndarray,
    max_dim: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Downscale image + mask to fit within *max_dim*, return scale factor."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, mask_u8, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    mask_small = cv2.resize(mask_u8, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_small, mask_small, scale


def _upscale_to(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upscale image to target dimensions with Lanczos."""
    if img.shape[:2] == (target_h, target_w):
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


class FullResWarpMethod(BaseStereoMethod):
    name: ClassVar[str] = "fullres_warp"
    label: ClassVar[str] = "Per-Eye Inpaint (Full-Res)"
    description: ClassVar[str] = (
        "Same warp strategy as Per-Eye Inpaint but at full source "
        "resolution. Downscales only for inpainting, then composites "
        "fills back via a feathered occlusion mask. Preserves maximum "
        "sharpness for the ~97% of pixels that are directly warped."
    )

    wants_full_res: ClassVar[bool] = True

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
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
        max_dim = settings.max_processing_dim

        # --- Full-res warp for both eyes ---
        left, left_mask = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "left")
        right, right_mask = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "right")

        if settings.inpaint_mask_dilate_px > 0:
            left_mask = dilate_occlusion_mask(left_mask, settings.inpaint_mask_dilate_px)
            right_mask = dilate_occlusion_mask(right_mask, settings.inpaint_mask_dilate_px)

        # --- Build background plate at processing resolution ---
        tight_px = settings.bg_plate_tight_dilate_px
        tight_mask_u8 = (
            BgPlateFillMethod._build_bg_inpaint_mask(
                depth_f32, fg_mask, max_disp, dilate_r_override=tight_px,
            )
            if tight_px > 0
            else None
        )
        full_mask_u8 = BgPlateFillMethod._build_bg_inpaint_mask(
            depth_f32, fg_mask, max_disp,
        )

        if full_mask_u8 is not None:
            # Downscale source for inpainting
            rgb_small, _, inp_scale = _downscale_for_inpaint(rgb_arr, full_mask_u8, max_dim)

            if inp_scale < 1.0:
                small_h, small_w = rgb_small.shape[:2]
                lama_mask_small = cv2.resize(
                    tight_mask_u8 if tight_mask_u8 is not None else full_mask_u8,
                    (small_w, small_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                full_mask_small = cv2.resize(
                    full_mask_u8, (small_w, small_h), interpolation=cv2.INTER_NEAREST,
                )
            else:
                lama_mask_small = tight_mask_u8 if tight_mask_u8 is not None else full_mask_u8
                full_mask_small = full_mask_u8

            bg_plate_small = inpainter.inpaint(rgb_small, lama_mask_small)
            bg_plate_small = _match_inpaint_color(rgb_small, bg_plate_small, lama_mask_small)

            if tight_mask_u8 is not None:
                outer_ring = (full_mask_small > 0) & (lama_mask_small == 0)
                if np.any(outer_ring):
                    ring_u8 = outer_ring.astype(np.uint8) * 255
                    bgr = cv2.cvtColor(bg_plate_small, cv2.COLOR_RGB2BGR)
                    bg_plate_small = cv2.cvtColor(
                        cv2.inpaint(bgr, ring_u8, 15, cv2.INPAINT_TELEA),
                        cv2.COLOR_BGR2RGB,
                    )

            _sharpen_inpainted_region(bg_plate_small, full_mask_small)

            # Upscale plate back to full-res
            bg_plate = _upscale_to(bg_plate_small, h, w)

            bg_depth = BgPlateFillMethod._build_bg_depth(depth_f32, full_mask_u8)

            # Warp background plate at full-res
            bg_left, bg_left_mask = hybrid_zbuf_remap_eye(bg_plate, bg_depth, max_disp, "left")
            bg_right, bg_right_mask = hybrid_zbuf_remap_eye(bg_plate, bg_depth, max_disp, "right")

            # --- Composite: sharp warped + tone-matched upscaled fills ---
            for eye_img, eye_mask, bg_img, bg_eye_mask in (
                (left, left_mask, bg_left, bg_left_mask),
                (right, right_mask, bg_right, bg_right_mask),
            ):
                holes = eye_mask > 0
                if not np.any(holes):
                    continue

                alpha = _inward_alpha(eye_mask)
                alpha_3 = alpha[:, :, np.newaxis]
                blended = (
                    eye_img.astype(np.float32) * (1.0 - alpha_3)
                    + bg_img.astype(np.float32) * alpha_3
                )
                eye_img[:] = np.clip(blended, 0, 255).astype(np.uint8)

                # Telea for any remaining holes in the bg plate warp itself
                remaining = holes & (bg_eye_mask > 0)
                dist = cv2.distanceTransform(holes.astype(np.uint8), cv2.DIST_L2, 3)
                border = holes & (dist <= _EDGE_TELEA_PX)
                telea_mask = (border | remaining).astype(np.uint8) * 255
                if np.any(telea_mask):
                    bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
                    eye_img[:] = cv2.cvtColor(
                        cv2.inpaint(bgr, telea_mask, _EDGE_TELEA_PX + 2, cv2.INPAINT_TELEA),
                        cv2.COLOR_BGR2RGB,
                    )
        else:
            # Fallback: per-eye inpainting at processing resolution
            for eye_img, eye_mask in ((left, left_mask), (right, right_mask)):
                img_small, mask_small, inp_scale = _downscale_for_inpaint(
                    eye_img, eye_mask, max_dim,
                )
                orig_small = img_small.copy()
                filled_small = inpainter.inpaint(img_small, mask_small)
                filled_small = _match_inpaint_color(orig_small, filled_small, mask_small)
                filled = _upscale_to(filled_small, h, w)

                alpha = _inward_alpha(eye_mask)
                alpha_3 = alpha[:, :, np.newaxis]
                blended = (
                    eye_img.astype(np.float32) * (1.0 - alpha_3)
                    + filled.astype(np.float32) * alpha_3
                )
                eye_img[:] = np.clip(blended, 0, 255).astype(np.uint8)

        return left, right
