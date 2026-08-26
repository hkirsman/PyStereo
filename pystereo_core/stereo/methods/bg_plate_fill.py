"""Method 2: Background-Plate Fill — hybrid z-buffer warp + stereo-consistent fill.

Builds a single inpainted "background plate" (foreground removed from the
source image), warps it with background-depth parallax into both eyes,
and composites over the disocclusion holes.  Both eyes see the same
fill content → stereo-consistent.

Two-pass inpainting:
1. Tight LaMa mask (10 px dilation) for maximum quality with most context.
2. Telea to extend fill to full disocclusion width.

Set ``bg_plate_tight_dilate_px`` to 0 (``PYSTEREO_TIGHT_DILATE=0``) to
inpaint the full-width mask in a single LaMa pass instead.

Includes spatially-varying colour correction and UnsharpMask on fills.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.warp import dilate_hole_mask, hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)


def _match_inpaint_color(
    original: np.ndarray,
    inpainted: np.ndarray,
    mask_u8: np.ndarray,
    ring_px: int = 25,
    n_bands: int = 8,
) -> np.ndarray:
    """Spatially-varying per-channel colour correction for inpainted pixels.

    LaMa zeros masked pixels before inference, biasing output dark.
    This splits the image into horizontal bands and computes a per-channel
    mean offset within each band, interpolating for a smooth gradient.
    """
    hole = mask_u8 > 0
    if not np.any(hole):
        return inpainted

    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1)
    )
    ring = (cv2.dilate(mask_u8, kern) > 0) & ~hole

    if np.count_nonzero(ring) < 64:
        return inpainted

    h, w = mask_u8.shape[:2]
    result = inpainted.copy()
    band_h = h / n_bands
    window_half = int(band_h)

    for c in range(3):
        orig_c = original[:, :, c].astype(np.float32)
        inp_c = inpainted[:, :, c].astype(np.float32)

        centres: list[float] = []
        offsets: list[float] = []

        for b in range(n_bands):
            yc = (b + 0.5) * band_h
            y0 = max(0, int(yc) - window_half)
            y1 = min(h, int(yc) + window_half)

            band_ring = ring[y0:y1]
            band_hole = hole[y0:y1]

            if np.count_nonzero(band_ring) < 16 or np.count_nonzero(band_hole) < 4:
                continue

            ref_mean = float(orig_c[y0:y1][band_ring].mean())
            inp_mean = float(inp_c[y0:y1][band_hole].mean())
            centres.append(yc)
            offsets.append(ref_mean - inp_mean)

        if not centres:
            ref_global = float(orig_c[ring].mean())
            inp_global = float(inp_c[hole].mean())
            corrected = inp_c + (ref_global - inp_global)
            result[hole, c] = np.clip(corrected[hole], 0, 255).astype(np.uint8)
            continue

        offset_col = np.interp(
            np.arange(h, dtype=np.float32),
            np.array(centres, dtype=np.float32),
            np.array(offsets, dtype=np.float32),
        )

        corrected = inp_c + offset_col[:, np.newaxis]
        result[hole, c] = np.clip(corrected[hole], 0, 255).astype(np.uint8)

    return result


def _sharpen_inpainted_region(
    bg_plate: np.ndarray,
    mask_u8: np.ndarray,
    amount: float = 0.5,
    sigma: float = 1.5,
) -> None:
    """In-place UnsharpMask on inpainted pixels to restore texture detail."""
    blurred = cv2.GaussianBlur(bg_plate, (0, 0), sigma)
    sharpened = cv2.addWeighted(bg_plate, 1.0 + amount, blurred, -amount, 0)
    fg = mask_u8 > 0
    bg_plate[fg] = sharpened[fg]


class BgPlateFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "bg_plate_fill"
    label: ClassVar[str] = "Background Plate (Deprecated)"
    description: ClassVar[str] = (
        "Hybrid z-buffer + cv2.remap warp with stereo-consistent "
        "background-plate fill.  Inpaints foreground out of the source "
        "once, warps the clean plate into both eyes.  Two-pass LaMa + "
        "Telea with colour correction and sharpening."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
    }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_bg_inpaint_mask(
        depth_f32: np.ndarray,
        fg_mask: np.ndarray | None,
        max_disp: float,
        *,
        dilate_r_override: int | None = None,
        dilate_cap_px: int = 0,
    ) -> np.ndarray | None:
        """Build a uint8 mask of foreground regions to inpaint away.

        The dilation margin is derived from the disparity, which at large
        disparities overshoots the disocclusion it needs to cover and eats
        real background next to the subject.  *dilate_cap_px* bounds it;
        0 leaves the derived value alone.  Ignored when *dilate_r_override*
        is given, since that is already an explicit radius.

        Returns ``None`` when the mask covers >70 % of the image —
        the caller falls back to per-eye inpainting.
        """
        if fg_mask is not None:
            base = fg_mask.copy()
        else:
            grad_x = np.abs(cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=5))
            grad_y = np.abs(cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=5))
            grad = grad_x + grad_y
            near_thresh = float(np.percentile(depth_f32, 60))
            base = ((grad > 0.02) & (depth_f32 > near_thresh)).astype(np.float32)

        if dilate_r_override is not None:
            dilate_r = dilate_r_override
        else:
            dilate_r = max(1, int(max_disp * 0.6) + 3)
            if dilate_cap_px > 0 and dilate_r > dilate_cap_px:
                logger.info(
                    "Background plate dilation capped: %d px → %d px "
                    "(max_disp %.0f px)",
                    dilate_r, dilate_cap_px, max_disp,
                )
                dilate_r = dilate_cap_px
        d = dilate_r * 2 + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        dilated = cv2.dilate(base, kern)
        mask = (dilated > 0.3).astype(np.uint8) * 255

        pct = float(np.count_nonzero(mask)) / float(mask.size)
        if pct > 0.70:
            logger.info(
                "Background inpaint mask too large (%.0f%%); "
                "falling back to per-eye inpainting",
                pct * 100,
            )
            return None
        return mask

    @staticmethod
    def _build_bg_depth(
        depth_f32: np.ndarray,
        bg_mask_u8: np.ndarray,
    ) -> np.ndarray:
        """Replace foreground depth with median background depth."""
        bg_depth = depth_f32.copy()
        fg_region = bg_mask_u8 > 0
        bg_region = ~fg_region
        median_bg = float(np.median(depth_f32[bg_region])) if np.any(bg_region) else 0.0
        bg_depth[fg_region] = median_bg
        return bg_depth

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def inpaint_preview(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> np.ndarray | None:
        full_mask = self._build_bg_inpaint_mask(
            depth_f32, fg_mask, max_disp,
            dilate_cap_px=settings.bg_plate_dilate_max_px,
        )
        if full_mask is None:
            return None
        tight = settings.bg_plate_tight_dilate_px
        lama_mask = full_mask
        if tight > 0:
            tight_mask = self._build_bg_inpaint_mask(
                depth_f32, fg_mask, max_disp, dilate_r_override=tight,
            )
            if tight_mask is not None:
                lama_mask = tight_mask
        plate = inpainter.inpaint(rgb_arr, lama_mask)
        return _match_inpaint_color(rgb_arr, plate, lama_mask)

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

        crack = settings.warp_crack_fill_px
        left, left_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "left", crack_fill_px=crack,
        )
        right, right_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "right", crack_fill_px=crack,
        )

        if settings.inpaint_mask_dilate_px > 0:
            uni = settings.unilateral_mask_dilate
            left_mask = dilate_hole_mask(
                left_mask, "left", settings.inpaint_mask_dilate_px, unilateral=uni,
            )
            right_mask = dilate_hole_mask(
                right_mask, "right", settings.inpaint_mask_dilate_px, unilateral=uni,
            )

        tight_px = settings.bg_plate_tight_dilate_px
        tight_mask_u8 = (
            self._build_bg_inpaint_mask(
                depth_f32, fg_mask, max_disp, dilate_r_override=tight_px
            )
            if tight_px > 0
            else None
        )
        full_mask_u8 = self._build_bg_inpaint_mask(
            depth_f32, fg_mask, max_disp,
            dilate_cap_px=settings.bg_plate_dilate_max_px,
        )

        if full_mask_u8 is not None:
            lama_mask = tight_mask_u8 if tight_mask_u8 is not None else full_mask_u8
            bg_plate = inpainter.inpaint(rgb_arr, lama_mask)
            bg_plate = _match_inpaint_color(rgb_arr, bg_plate, lama_mask)

            if tight_mask_u8 is not None:
                outer_ring = (full_mask_u8 > 0) & (tight_mask_u8 == 0)
                if np.any(outer_ring):
                    ring_u8 = outer_ring.astype(np.uint8) * 255
                    bgr = cv2.cvtColor(bg_plate, cv2.COLOR_RGB2BGR)
                    bg_plate = cv2.cvtColor(
                        cv2.inpaint(bgr, ring_u8, 15, cv2.INPAINT_TELEA),
                        cv2.COLOR_BGR2RGB,
                    )

            _sharpen_inpainted_region(bg_plate, full_mask_u8)

            bg_depth = self._build_bg_depth(depth_f32, full_mask_u8)

            if bg_plate.shape[:2] != (h, w):
                bh, bw = bg_plate.shape[:2]
                if bh >= h and bw >= w:
                    bg_plate = bg_plate[:h, :w]
                else:
                    bg_plate = cv2.resize(
                        bg_plate, (w, h), interpolation=cv2.INTER_LINEAR
                    )

            bg_left, bg_left_mask = hybrid_zbuf_remap_eye(
                bg_plate, bg_depth, max_disp, "left"
            )
            bg_right, bg_right_mask = hybrid_zbuf_remap_eye(
                bg_plate, bg_depth, max_disp, "right"
            )

            _EDGE_TELEA_PX = 2

            for eye_img, eye_mask, bg_img, bg_eye_mask in (
                (left, left_mask, bg_left, bg_left_mask),
                (right, right_mask, bg_right, bg_right_mask),
            ):
                holes = eye_mask > 0
                if not np.any(holes):
                    continue
                eye_img[holes] = bg_img[holes]

                dist = cv2.distanceTransform(
                    holes.astype(np.uint8), cv2.DIST_L2, 3
                )
                telea_mask = holes & (dist <= _EDGE_TELEA_PX)
                if settings.fill_telea_residual:
                    telea_mask = telea_mask | (holes & (bg_eye_mask > 0))
                telea_mask = telea_mask.astype(np.uint8) * 255
                if np.any(telea_mask):
                    bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
                    eye_img[:] = cv2.cvtColor(
                        cv2.inpaint(
                            bgr, telea_mask, _EDGE_TELEA_PX + 2, cv2.INPAINT_TELEA
                        ),
                        cv2.COLOR_BGR2RGB,
                    )
        else:
            left_orig = left.copy()
            right_orig = right.copy()
            left = inpainter.inpaint(left, left_mask)
            right = inpainter.inpaint(right, right_mask)
            left = _match_inpaint_color(left_orig, left, left_mask)
            right = _match_inpaint_color(right_orig, right, right_mask)
            if left.shape[:2] != (h, w):
                lh, lw = left.shape[:2]
                left = (
                    left[:h, :w]
                    if (lh >= h and lw >= w)
                    else cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
                )
            if right.shape[:2] != (h, w):
                rh, rw = right.shape[:2]
                right = (
                    right[:h, :w]
                    if (rh >= h and rw >= w)
                    else cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)
                )

        return left, right
