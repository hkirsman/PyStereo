"""Method 3: Width-Routed Fill - narrow-strip CPU propagation + bg-plate LaMa.

Routes disocclusion holes by width after unilateral mask dilation:

- **Narrow** (bounding-box width ≤ ``narrow_strip_max_px``): filled by
  mirroring adjacent background pixels from the geometrically correct side.
  No neural network, no dark bias, exact photographic texture.
- **Wide** (> threshold): filled via stereo-consistent bg-plate with
  **anisotropic banded inpainting** - the source image is split into
  independent horizontal crops so LaMa's FFCs only see local vertical
  context.  Per-band colour matching + sharpen + Poisson seam blending.

Uses directional (foreground-side only) mask dilation instead of the
isotropic dilation of other methods.
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
from pystereo_core.stereo.warp import EyeSide, hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Unilateral (directional) mask dilation
# ------------------------------------------------------------------

def _unilateral_dilate(
    mask: np.ndarray,
    eye: EyeSide,
    pixels: int = 4,
) -> np.ndarray:
    """Dilate occlusion mask toward the foreground side only.

    DIBR geometry guarantees a fixed relationship between eye side and
    foreground/background placement around each hole:

      left eye  → holes appear left of foreground → fg RIGHT → extend right
      right eye → holes appear right of foreground → fg LEFT  → extend left
    """
    if pixels <= 0:
        return mask.copy()

    kw = pixels + 1
    kern = np.ones((1, kw), dtype=np.uint8)

    if eye == "left":
        return cv2.dilate(mask, kern, anchor=(kw - 1, 0), iterations=1)
    return cv2.dilate(mask, kern, anchor=(0, 0), iterations=1)


# ------------------------------------------------------------------
# Connected-component width classification
# ------------------------------------------------------------------

def _classify_holes(
    mask: np.ndarray,
    threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an occlusion mask into narrow and wide components.

    Uses the bounding-box width of each 8-connected component.
    Returns ``(narrow_mask, wide_mask)`` both uint8, 255 = hole.
    """
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8,
    )

    narrow = np.zeros_like(mask)
    wide = np.zeros_like(mask)

    for i in range(1, num_labels):
        component = labels == i
        if stats[i, cv2.CC_STAT_WIDTH] <= threshold:
            narrow[component] = 255
        else:
            wide[component] = 255

    return narrow, wide


# ------------------------------------------------------------------
# Narrow-strip background-pixel propagation
# ------------------------------------------------------------------

def _fill_narrow_strips(
    eye_img: np.ndarray,
    narrow_mask: np.ndarray,
    eye: EyeSide,
) -> np.ndarray:
    """In-place mirror-fill narrow holes from adjacent background pixels.

    Left eye  → background is LEFT of each hole → mirror from left.
    Right eye → background is RIGHT             → mirror from right.

    Returns a uint8 mask (255 = unfilled) for runs that could not be
    filled because they abut the image border on the background side.
    """
    h, w = eye_img.shape[:2]
    unfilled = np.zeros((h, w), dtype=np.uint8)

    if not np.any(narrow_mask):
        return unfilled

    src_img = eye_img.copy()

    for y in range(h):
        row = narrow_mask[y]
        if not np.any(row):
            continue

        binary_row = (row > 0).astype(np.int8)
        padded = np.empty(w + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = binary_row
        padded[-1] = 0
        changes = np.diff(padded)
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        for x0, x1_exc in zip(starts, ends):
            run_w = x1_exc - x0

            if eye == "left":
                if x0 == 0:
                    unfilled[y, x0:x1_exc] = 255
                    continue
                src_idx = np.clip(x0 - 1 - np.arange(run_w), 0, w - 1)
            else:
                if x1_exc >= w:
                    unfilled[y, x0:x1_exc] = 255
                    continue
                src_idx = np.clip(
                    x1_exc + (run_w - 1) - np.arange(run_w), 0, w - 1,
                )

            eye_img[y, x0:x1_exc] = src_img[y, src_idx]

    return unfilled


# ------------------------------------------------------------------
# Poisson seam blend for wide fills
# ------------------------------------------------------------------

def _poisson_blend_wide(
    eye_img: np.ndarray,
    eye_before: np.ndarray,
    wide_mask: np.ndarray,
) -> None:
    """In-place Poisson (gradient-domain) blend of wide filled regions.

    ``eye_img`` already has bg-plate content pasted at wide holes.
    ``eye_before`` has correct original content at the mask boundary.
    The solver takes texture gradients from ``eye_img`` and adjusts
    overall luminance to match ``eye_before``'s boundary values.
    """
    if not np.any(wide_mask):
        return

    h, w = eye_img.shape[:2]

    safe = wide_mask.copy()
    safe[0, :] = 0
    safe[-1, :] = 0
    safe[:, 0] = 0
    safe[:, -1] = 0
    if not np.any(safe):
        return

    src_bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
    dst_bgr = cv2.cvtColor(eye_before, cv2.COLOR_RGB2BGR)
    center = (w // 2, h // 2)

    try:
        blended = cv2.seamlessClone(
            src_bgr, dst_bgr, safe, center, cv2.NORMAL_CLONE,
        )
        result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        holes = wide_mask > 0
        eye_img[holes] = result_rgb[holes]
    except cv2.error:
        logger.debug("seamlessClone failed; keeping direct paste")


# ------------------------------------------------------------------
# Anisotropic banded inpainting
# ------------------------------------------------------------------

_ANISO_BANDS = 5
_ANISO_OVERLAP_PX = 60


def _inpaint_banded(
    rgb_arr: np.ndarray,
    mask_u8: np.ndarray,
    inpainter: InpaintBackend,
) -> np.ndarray:
    """Inpaint via independent horizontal crops to limit LaMa's vertical context.

    Each band is a physically separate inference - FFCs can only see
    textures within the band height (~300-400 px).  Bands are
    feather-blended in overlap zones.  Colour matching is applied
    per-band with local context before stitching.

    Key differences from the reverted sequential-strip approach
    (see STEREO_STATUS.md "Attempted and reverted"):
    - Bands are *independent* - no sequential tonal drift.
    - Input is *physically cropped* - FFCs cannot reach distant textures
      even though they have a global receptive field within each crop.
    """
    h, w = rgb_arr.shape[:2]

    if _ANISO_BANDS <= 1 or h <= 400:
        result = inpainter.inpaint(rgb_arr, mask_u8)
        return _match_inpaint_color(rgb_arr, result, mask_u8)

    n = _ANISO_BANDS
    overlap = _ANISO_OVERLAP_PX
    band_step = max(1, (h - overlap) // n)
    fade = max(1, overlap // 2)

    result_f = np.zeros((h, w, 3), dtype=np.float32)
    weight_sum = np.zeros((h, 1, 1), dtype=np.float32)
    bands_inpainted = 0

    for b in range(n):
        y0 = max(0, b * band_step)
        y1 = min(h, y0 + band_step + overlap)
        if b == n - 1:
            y1 = h

        band_h = y1 - y0
        band_mask = mask_u8[y0:y1]
        band_img = rgb_arr[y0:y1].copy()

        if np.any(band_mask):
            band_result = inpainter.inpaint(band_img, band_mask)
            if band_result.shape[:2] != (band_h, w):
                rh, rw = band_result.shape[:2]
                if rh >= band_h and rw >= w:
                    band_result = band_result[:band_h, :w]
                else:
                    band_result = cv2.resize(
                        band_result, (w, band_h),
                        interpolation=cv2.INTER_LINEAR,
                    )
            band_result = _match_inpaint_color(
                rgb_arr[y0:y1], band_result, band_mask,
            )
            bands_inpainted += 1
        else:
            band_result = band_img

        w_col = np.ones(band_h, dtype=np.float32)
        f = min(fade, band_h // 4)
        if f > 0:
            if b > 0:
                w_col[:f] = np.linspace(0, 1, f)
            if b < n - 1:
                w_col[-f:] = np.linspace(1, 0, f)

        w_3d = w_col[:, np.newaxis, np.newaxis]
        result_f[y0:y1] += band_result.astype(np.float32) * w_3d
        weight_sum[y0:y1] += w_3d

    weight_sum = np.maximum(weight_sum, 1e-6)
    logger.debug(
        "Banded inpaint: %d/%d bands had foreground to remove", bands_inpainted, n,
    )
    return np.clip(result_f / weight_sum, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Method class
# ------------------------------------------------------------------

class RoutedFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "routed_fill"
    label: ClassVar[str] = "Width-Routed Fill"
    deprecated: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Hybrid z-buffer warp with width-routed disocclusion fill.  "
        "Narrow strips (≤ threshold) are mirror-filled from adjacent "
        "background pixels (CPU, zero dark bias).  Wide regions use "
        "stereo-consistent bg-plate with anisotropic banded LaMa "
        "(independent horizontal crops limit FFC vertical context) "
        "+ Poisson seam blending.  "
        "Unilateral mask dilation toward foreground only."
    )
    ui_info: ClassVar[str] = (
        "Deprecated. Narrow holes get a fast mirror fill; wide holes use "
        "the background-plate path. Aimed at cleaner edges than plain "
        "per-eye inpaint."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
        "inpaint_mask_dilate_px": 4,
    }

    # ------------------------------------------------------------------
    # Wide-strip bg-plate fill (reuses BgPlateFillMethod helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_wide_bg_plate(
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        inpainter: InpaintBackend,
        left: np.ndarray,
        right: np.ndarray,
        left_wide: np.ndarray,
        right_wide: np.ndarray,
    ) -> bool:
        """Build a bg plate via banded inpainting, warp, paste, Poisson blend.

        Uses ``_inpaint_banded`` to split the source image into
        independent horizontal crops - each LaMa inference only sees
        local vertical context, preventing cross-material texture cloning.

        Returns ``True`` on success, ``False`` if the bg plate could not
        be built (foreground >70 %).
        """
        h, w = rgb_arr.shape[:2]

        _TIGHT_DILATE = 10
        tight_mask = BgPlateFillMethod._build_bg_inpaint_mask(
            depth_f32, fg_mask, max_disp, dilate_r_override=_TIGHT_DILATE,
        )
        full_mask = BgPlateFillMethod._build_bg_inpaint_mask(
            depth_f32, fg_mask, max_disp,
        )
        if full_mask is None:
            return False

        lama_mask = tight_mask if tight_mask is not None else full_mask
        bg_plate = _inpaint_banded(rgb_arr, lama_mask, inpainter)

        if tight_mask is not None:
            outer_ring = (full_mask > 0) & (tight_mask == 0)
            if np.any(outer_ring):
                ring_u8 = outer_ring.astype(np.uint8) * 255
                bgr = cv2.cvtColor(bg_plate, cv2.COLOR_RGB2BGR)
                bg_plate = cv2.cvtColor(
                    cv2.inpaint(bgr, ring_u8, 15, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )

        _sharpen_inpainted_region(bg_plate, full_mask)

        if bg_plate.shape[:2] != (h, w):
            bh, bw = bg_plate.shape[:2]
            if bh >= h and bw >= w:
                bg_plate = bg_plate[:h, :w]
            else:
                bg_plate = cv2.resize(
                    bg_plate, (w, h), interpolation=cv2.INTER_LINEAR,
                )

        bg_depth = BgPlateFillMethod._build_bg_depth(depth_f32, full_mask)
        bg_left, _ = hybrid_zbuf_remap_eye(bg_plate, bg_depth, max_disp, "left")
        bg_right, _ = hybrid_zbuf_remap_eye(bg_plate, bg_depth, max_disp, "right")

        for eye_img, bg_img, wide_mask in (
            (left, bg_left, left_wide),
            (right, bg_right, right_wide),
        ):
            wide_holes = wide_mask > 0
            if not np.any(wide_holes):
                continue

            eye_before = eye_img.copy()
            eye_img[wide_holes] = bg_img[wide_holes]
            _poisson_blend_wide(eye_img, eye_before, wide_mask)

        return True

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

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

        # 1. Warp both eyes (hybrid z-buf + sub-pixel remap)
        left, left_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "left",
        )
        right, right_mask = hybrid_zbuf_remap_eye(
            rgb_arr, depth_f32, max_disp, "right",
        )

        # 2. Unilateral mask dilation (toward foreground only)
        dilate_px = settings.inpaint_mask_dilate_px
        left_mask = _unilateral_dilate(left_mask, "left", dilate_px)
        right_mask = _unilateral_dilate(right_mask, "right", dilate_px)

        # 3. Classify connected components by bounding-box width
        left_narrow, left_wide = _classify_holes(left_mask, threshold)
        right_narrow, right_wide = _classify_holes(right_mask, threshold)

        narrow_total = int(
            np.count_nonzero(left_narrow) + np.count_nonzero(right_narrow)
        )
        wide_total = int(
            np.count_nonzero(left_wide) + np.count_nonzero(right_wide)
        )
        logger.info(
            "Routed fill: %d narrow px, %d wide px (threshold=%d)",
            narrow_total, wide_total, threshold,
        )

        # 4. Narrow strips → background-pixel mirror propagation (CPU)
        left_unfilled = _fill_narrow_strips(left, left_narrow, "left")
        right_unfilled = _fill_narrow_strips(right, right_narrow, "right")

        # 5. Wide strips → bg-plate LaMa + Poisson seam blend
        remaining_left = left_unfilled.copy()
        remaining_right = right_unfilled.copy()

        if wide_total > 0:
            ok = self._fill_wide_bg_plate(
                rgb_arr, depth_f32, max_disp, fg_mask, inpainter,
                left, right, left_wide, right_wide,
            )
            if not ok:
                remaining_left = np.maximum(remaining_left, left_wide)
                remaining_right = np.maximum(remaining_right, right_wide)

        # 6. Telea sweep for any pixels that couldn't be filled above
        #    (border-edge narrow runs, failed bg-plate fallback)
        for eye_img, rem in ((left, remaining_left), (right, remaining_right)):
            if np.any(rem):
                bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
                eye_img[:] = cv2.cvtColor(
                    cv2.inpaint(bgr, rem, 15, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )

        return left, right
