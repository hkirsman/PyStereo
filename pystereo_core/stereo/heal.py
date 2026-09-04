"""Composite depth healing: fix broken foreground edges using a semantic mask.

Depth Anything V2 produces excellent global depth for backgrounds and scenery
but creates hard 90-degree cutoffs on low-contrast hair/fur boundaries.
BiRefNet captures those fine edges accurately.  This module merges both
signals using a two-tier approach:

- **Core foreground** (mask > 0.7): reliable anchor for the "true" foreground
  depth.  Used to compute the reference median and seed the fill.
- **Extended foreground** (dilated mask > 0.05): captures the soft hair/fur
  fringe *plus* a morphological dilation band beyond BiRefNet's own edge.

The final depth is alpha-blended using a dilated + blurred version of the
mask so the transition from subject to background is wide and smooth.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def heal_depth_with_mask(
    depth_f32: np.ndarray,
    fg_mask: np.ndarray,
    *,
    bg_threshold_ratio: float = 0.6,
    edge_blur_sigma: float = 8.0,
    core_threshold: float = 0.7,
    extended_threshold: float = 0.05,
    mask_dilate_px: int = 15,
) -> np.ndarray:
    """Heal broken foreground depth using a semantic foreground mask.

    Parameters
    ----------
    depth_f32:
        ``(H, W)`` float32 depth in ``[0, 1]``.  1.0 = closest.
    fg_mask:
        ``(H, W)`` float32 in ``[0, 1]``.  BiRefNet foreground alpha.
    bg_threshold_ratio:
        A foreground pixel whose depth is below
        ``core_median * bg_threshold_ratio`` is considered "broken".
    edge_blur_sigma:
        Gaussian sigma for the final alpha-blend ramp.
    core_threshold:
        Mask confidence above which a pixel is "core" foreground.
    extended_threshold:
        Mask confidence (after dilation) above which a pixel is eligible
        for healing.
    mask_dilate_px:
        Morphological dilation radius applied to the continuous mask
        before the extended-foreground test.  Pushes the healing zone
        outward beyond BiRefNet's own edge so the depth ramp starts
        further from the subject.
    """
    core_fg = fg_mask > core_threshold

    if not np.any(core_fg):
        logger.debug("No core foreground detected - skipping depth healing")
        return depth_f32.copy()

    # -- Dilate the continuous mask outward ---------------------------------
    # This extends the foreground alpha beyond BiRefNet's tight edge so we
    # heal (and blend) over a wider band.  cv2.dilate on the float mask
    # propagates the local maximum.
    if mask_dilate_px > 0:
        d = mask_dilate_px * 2 + 1
        dilate_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
        dilated_mask = cv2.dilate(fg_mask, dilate_kern)
    else:
        dilated_mask = fg_mask.copy()

    extended_fg = dilated_mask > extended_threshold

    # -- Step 1: reference depth from core foreground -----------------------
    core_median = float(np.median(depth_f32[core_fg]))
    threshold = core_median * bg_threshold_ratio

    missing = extended_fg & (depth_f32 < threshold)

    n_missing = int(np.sum(missing))
    if n_missing == 0:
        logger.debug("No broken foreground pixels - skipping depth healing")
        return depth_f32.copy()

    n_ext = int(np.sum(extended_fg))
    logger.info(
        "Depth healing: %d broken pixels (%.1f%% of %d extended-fg, "
        "core_median=%.3f, threshold=%.3f)",
        n_missing,
        100.0 * n_missing / max(1, n_ext),
        n_ext,
        core_median,
        threshold,
    )

    healed = depth_f32.copy()

    # -- Step 2: propagate valid foreground depth into broken pixels ---------
    valid = extended_fg & ~missing
    fill = np.where(valid, healed, 0.0).astype(np.float32)
    validity = valid.astype(np.float32)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    remaining = missing.copy()
    for _ in range(150):
        if not np.any(remaining):
            break
        expanded_sum = cv2.dilate(fill, kern)
        expanded_cnt = cv2.dilate(validity, kern)
        reachable = remaining & (expanded_cnt > 0)
        if not np.any(reachable):
            break
        avg = np.divide(
            expanded_sum,
            expanded_cnt,
            out=np.zeros_like(expanded_sum),
            where=expanded_cnt > 0,
        )
        fill[reachable] = avg[reachable]
        validity[reachable] = 1.0
        remaining = remaining & ~reachable

    fill[remaining] = core_median
    healed[missing] = fill[missing]

    # -- Step 3: alpha-blend using the dilated + blurred mask ---------------
    # A wide Gaussian blur on the dilated mask creates a gentle ramp that
    # starts well outside the subject silhouette, preventing a visible
    # depth cliff.
    alpha = cv2.GaussianBlur(dilated_mask, (0, 0), edge_blur_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    result = alpha * healed + (1.0 - alpha) * depth_f32

    return np.clip(result, 0.0, 1.0)
