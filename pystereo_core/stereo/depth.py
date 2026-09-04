"""Depth estimation wrapper with continuous float32 output and guided filtering.

Provides:
- Raw float32 depth normalized [0.0, 1.0] (1.0 = closest)
- Edge-aware guided filter refinement using RGB as guide
- Optional depth gamma to redistribute disparity budget
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

from pystereo_core.depth import DepthEstimator

logger = logging.getLogger(__name__)


def estimate_depth_f32(
    image: Image.Image,
    estimator: DepthEstimator | None = None,
) -> np.ndarray:
    """Return a continuous float32 depth map ``(H, W)`` in [0.0, 1.0].

    1.0 = closest, 0.0 = farthest.  Uses ``process_raw`` when available
    to bypass 8-bit quantization; falls back to the legacy path otherwise.
    """
    if estimator is None:
        raise ValueError("depth estimator instance is required")
    if hasattr(estimator, "process_raw"):
        return estimator.process_raw(image.convert("RGB"))
    depth_pil = estimator.process(image.convert("RGB")).convert("L")
    depth = np.array(depth_pil, dtype=np.float32) / 255.0
    lo, hi = float(depth.min()), float(depth.max())
    if hi <= lo:
        return depth
    depth = (depth - lo) / (hi - lo)
    return depth


def guided_filter_depth(
    depth_f32: np.ndarray,
    guide_rgb: np.ndarray,
    *,
    radius: int = 8,
    eps: float = 1e-4,
) -> np.ndarray:
    """Refine depth edges using the RGB image as a structural guide.

    Uses ``cv2.ximgproc.createGuidedFilter`` when available (opencv-contrib),
    otherwise falls back to a bilateral filter approximation.

    Parameters
    ----------
    depth_f32:
        ``(H, W)`` float32 depth in [0, 1].
    guide_rgb:
        ``(H, W, 3)`` uint8 RGB image (same resolution as depth).
    radius:
        Filter kernel radius.
    eps:
        Regularization.  Smaller = sharper edge-snapping;
        larger = allows more variation within uniform colours.
    """
    try:
        gf = cv2.ximgproc.createGuidedFilter(guide_rgb, radius, eps)
        refined = gf.filter(depth_f32)
    except AttributeError:
        logger.warning(
            "cv2.ximgproc unavailable - falling back to bilateral filter. "
            "Install opencv-contrib-python for best quality."
        )
        refined = _fallback_bilateral(depth_f32, guide_rgb, radius)
    return np.clip(refined, 0.0, 1.0)


def _fallback_bilateral(
    depth_f32: np.ndarray,
    guide_rgb: np.ndarray,
    radius: int,
) -> np.ndarray:
    """Pure OpenCV bilateral filter fallback (no ximgproc)."""
    d = radius * 2 + 1
    return cv2.bilateralFilter(depth_f32, d, sigmaColor=0.1, sigmaSpace=float(radius))


def apply_depth_gamma(depth_f32: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply ``depth^(1/gamma)`` to redistribute disparity budget.

    gamma > 1.0 spreads out far (low) depth values, giving backgrounds
    more distinct shift levels.  1.0 = linear (no change).
    """
    if gamma == 1.0:
        return depth_f32
    return np.power(np.clip(depth_f32, 0.0, 1.0), 1.0 / gamma)


# --- Legacy compatibility --------------------------------------------------

def estimate_depth(
    image: Image.Image,
    estimator: DepthEstimator | None = None,
) -> Image.Image:
    """Return a normalized grayscale depth map (255 = closest, 0 = farthest)."""
    depth_f32 = estimate_depth_f32(image, estimator)
    return Image.fromarray((depth_f32 * 255.0).astype(np.uint8), mode="L")


def depth_to_u8_array(depth: Image.Image) -> np.ndarray:
    """Convert a depth PIL image to ``(H, W)`` uint8 numpy."""
    return np.array(depth.convert("L"), dtype=np.uint8)
