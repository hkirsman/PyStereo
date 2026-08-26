"""Stereo warping primitives — all strategies live here.

Three warp implementations are available:

1. ``inverse_remap_warp_eye``  — Pure inverse ``cv2.remap`` with continuous
   flow maps and stretch-based occlusion detection.  Used by
   :class:`~pystereo_core.stereo.methods.per_eye_inpaint.PerEyeInpaintMethod`.

2. ``hybrid_zbuf_remap_eye``  — Vectorised forward z-buffer (clean occlusion
   boundaries) + backward ``cv2.remap`` (sub-pixel sampling).  Used by
   :class:`~pystereo_core.stereo.methods.bg_plate_fill.BgPlateFillMethod`.

3. ``forward_splat_eye``  — Legacy row-loop forward splatting.  Kept for
   reference / debugging only.
"""

from __future__ import annotations

import logging
from typing import Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

EyeSide = Literal["left", "right"]


# =====================================================================
# Shared utilities
# =====================================================================

def adaptive_max_disparity(
    depth_f32: np.ndarray,
    width: int,
    *,
    base_ratio: float,
    min_ratio: float,
    max_ratio: float,
    adaptive: bool = True,
) -> float:
    """Return max horizontal separation (px) at nearest depth."""
    if depth_f32.dtype == np.uint8:
        depth_norm = depth_f32.astype(np.float32) / 255.0
    else:
        depth_norm = depth_f32

    if adaptive:
        q25 = float(np.percentile(depth_norm, 25))
        q75 = float(np.percentile(depth_norm, 75))
        iqr = q75 - q25
        scale = max(0.5, min(1.5, iqr / 0.4))
        ratio = base_ratio * scale
    else:
        ratio = base_ratio
    ratio = max(min_ratio, min(max_ratio, ratio))
    return width * ratio


def dilate_occlusion_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expand hole mask so inpainting covers edge halos.

    Symmetric, so it also grows the hole *into* the foreground that borders
    it.  Prefer :func:`unilateral_dilate_occlusion_mask` when the eye side
    is known.
    """
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1)
    )
    return cv2.dilate(mask, kernel, iterations=1)


def unilateral_dilate_occlusion_mask(
    mask: np.ndarray,
    eye: EyeSide,
    pixels: int,
) -> np.ndarray:
    """Dilate a hole mask toward the background side only.

    DIBR geometry fixes which side of a hole the occluder sits on:

      left eye  → holes open left of the foreground → extend right
      right eye → holes open right of the foreground → extend left

    Extending only that way gives the inpainter its margin without eating
    into the foreground silhouette bordering the hole.
    """
    if pixels <= 0:
        return mask.copy()

    kw = pixels + 1
    kernel = np.ones((1, kw), dtype=np.uint8)
    anchor = (kw - 1, 0) if eye == "left" else (0, 0)
    return cv2.dilate(mask, kernel, anchor=anchor, iterations=1)


def dilate_hole_mask(
    mask: np.ndarray,
    eye: EyeSide,
    pixels: int,
    *,
    unilateral: bool = True,
) -> np.ndarray:
    """Grow a hole mask to give the inpainter margin.

    Unilateral by default — see
    :func:`unilateral_dilate_occlusion_mask` for why.
    """
    if unilateral:
        return unilateral_dilate_occlusion_mask(mask, eye, pixels)
    return dilate_occlusion_mask(mask, pixels)


def _fill_zbuffer_cracks(z_buf: np.ndarray, max_crack_px: int) -> np.ndarray:
    """Close narrow horizontal gaps left by the ±1 px splat footprint.

    The forward splat writes each source pixel to ``floor(dst)`` and
    ``floor(dst) + 1`` only.  Wherever the warp magnifies a surface,
    consecutive source pixels land more than two columns apart and leave a
    1-2 px gap.  Those gaps are sampling artefacts on a continuous surface,
    not disocclusions — the depth either side is essentially equal — but an
    unfilled z-buffer entry makes them read as holes, and dilating them
    afterwards inflates each speck into a blob.

    Gaps wider than *max_crack_px* are left alone: those are real
    disocclusions, bounded by a genuine depth step.
    """
    if max_crack_px <= 0:
        return z_buf

    valid = (z_buf >= 0.0).astype(np.uint8)
    kernel = np.ones((1, max_crack_px * 2 + 1), dtype=np.uint8)
    closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
    cracks = (closed > 0) & (valid == 0)
    if not np.any(cracks):
        return z_buf

    # Propagate depth inward from each crack's horizontal neighbours.  The
    # invalid sentinel is -1, so a max filter always loses to a real depth.
    filled = z_buf.copy()
    step = np.ones((1, 3), dtype=np.uint8)
    for _ in range(max_crack_px):
        spread = cv2.dilate(filled, step)
        filled = np.where(filled < 0.0, spread, filled)

    return np.where(cracks, filled, z_buf)


def zbuf_remap_maps(
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
    *,
    crack_fill_px: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward z-buffer, then the backward sampling maps it implies.

    Returns ``(src_x, src_y, occlusion_mask)``.  Split out so a layer's
    colour and its alpha matte can be resampled through the *same* maps —
    otherwise the two drift apart and the matte stops matching its own
    silhouette.
    """
    h, w = depth_f32.shape[:2]
    half_sep = max_disparity * 0.5
    sign = 1.0 if eye == "left" else -1.0

    shifts = depth_f32 * (half_sep * sign)
    src_x_f = np.arange(w, dtype=np.float32)[None, :]
    dst_f = src_x_f + shifts

    dst_lo = np.floor(dst_f).astype(np.int32)
    dst_hi = dst_lo + 1
    y_full = np.broadcast_to(np.arange(h, dtype=np.int32)[:, None], (h, w))

    z_buf = np.full((h, w), -1.0, dtype=np.float32)
    valid_lo = (dst_lo >= 0) & (dst_lo < w)
    np.maximum.at(z_buf, (y_full[valid_lo], dst_lo[valid_lo]), depth_f32[valid_lo])
    valid_hi = (dst_hi >= 0) & (dst_hi < w)
    np.maximum.at(z_buf, (y_full[valid_hi], dst_hi[valid_hi]), depth_f32[valid_hi])

    z_buf = _fill_zbuffer_cracks(z_buf, crack_fill_px)
    occ_mask = np.where(z_buf < 0.0, np.uint8(255), np.uint8(0))

    dst_x = np.arange(w, dtype=np.float32)[None, :]
    vis_depth = np.where(z_buf > 0, z_buf, np.float32(0))
    src_x = np.ascontiguousarray(
        (dst_x - vis_depth * (half_sep * sign)).astype(np.float32)
    )
    src_y = np.broadcast_to(
        np.arange(h, dtype=np.float32)[:, None], (h, w)
    ).copy()
    return src_x, src_y, occ_mask


def warp_layer_eye(
    image: np.ndarray,
    alpha: np.ndarray,
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
    *,
    crack_fill_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one RGB layer together with its alpha matte.

    Both are resampled through the same maps, so the matte keeps matching
    the colour it belongs to.  Disoccluded pixels get alpha 0 — whatever is
    behind this layer should show through there.

    Returns ``(warped_rgb, warped_alpha)`` with alpha float32 in [0, 1].
    """
    src_x, src_y, occ = zbuf_remap_maps(
        depth_f32, max_disparity, eye, crack_fill_px=crack_fill_px,
    )
    warped = cv2.remap(
        image, src_x, src_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    warped_a = cv2.remap(
        alpha.astype(np.float32), src_x, src_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_a[occ > 0] = 0.0
    return warped, np.clip(warped_a, 0.0, 1.0)


# =====================================================================
# Strategy 1: Inverse remap (pure cv2.remap, stretch-based occlusion)
# =====================================================================

def _build_inverse_remap_flow(
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
) -> tuple[np.ndarray, np.ndarray]:
    """Build continuous X/Y remap flow maps for inverse warping."""
    h, w = depth_f32.shape[:2]
    half_sep = max_disparity * 0.5

    sign = -1.0 if eye == "left" else 1.0
    shift = depth_f32 * half_sep * sign

    map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = map_x + shift

    return map_x, map_y


def _detect_occlusion_inverse(
    map_x: np.ndarray,
    depth_f32: np.ndarray,
    width: int,
    eye: EyeSide,
    *,
    stretch_threshold: float = 2.5,
) -> np.ndarray:
    """Detect occlusion holes from the inverse remap flow.

    Marks pixels as occluded (255) when:
    1. The source coordinate is out of bounds.
    2. Adjacent pixels sample from very different source locations (stretch).
    """
    h, w = map_x.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    oob = (map_x < 0.0) | (map_x >= float(width - 1))
    mask[oob] = 255

    dx = np.abs(np.diff(map_x, axis=1, prepend=map_x[:, :1]))
    stretched = dx > stretch_threshold
    mask[stretched] = 255

    depth_dx = np.abs(np.diff(depth_f32, axis=1, prepend=depth_f32[:, :1]))
    depth_edges = depth_dx > 0.05
    roll_dir = -1 if eye == "left" else 1
    neighbor = np.roll(depth_f32, roll_dir, axis=1)
    mask[depth_edges & (depth_f32 < neighbor)] = 255

    return mask


def inverse_remap_warp_eye(
    image: np.ndarray,
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
    *,
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp via pure inverse ``cv2.remap`` with stretch-based occlusion.

    Returns ``(warped, occlusion_mask)`` — 255 = hole.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB image, got shape {image.shape}")
    if depth_f32.shape[:2] != image.shape[:2]:
        raise ValueError("Depth map must match image height and width")

    h, w = image.shape[:2]

    map_x, map_y = _build_inverse_remap_flow(depth_f32, max_disparity, eye)
    occlusion_mask = _detect_occlusion_inverse(map_x, depth_f32, w, eye)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    hole_pct = 100.0 * float(np.count_nonzero(occlusion_mask)) / float(h * w)
    logger.debug("Inverse-remap %s eye: %.1f%% occlusion holes", eye, hole_pct)
    return warped, occlusion_mask


# =====================================================================
# Strategy 2: Hybrid z-buffer + sub-pixel remap
# =====================================================================

def hybrid_zbuf_remap_eye(
    image: np.ndarray,
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
    *,
    crack_fill_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Hybrid warp: vectorised forward z-buffer + sub-pixel bilinear remap.

    1. Forward z-buffer (vectorised): project source pixels to destination,
       nearest depth wins.  Unprojected destinations are true occlusion holes.
    2. Close sampling cracks up to *crack_fill_px* wide (see
       :func:`_fill_zbuffer_cracks`); 0 disables.
    3. Backward remap: use z-buffer depth to compute exact sub-pixel source
       column, sample with bilinear interpolation.

    Returns ``(warped, occlusion_mask)`` — 255 = hole.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB image, got shape {image.shape}")
    if depth_f32.shape[:2] != image.shape[:2]:
        raise ValueError("Depth map must match image height and width")

    src_x, src_y, occ_mask = zbuf_remap_maps(
        depth_f32, max_disparity, eye, crack_fill_px=crack_fill_px,
    )
    h, w = image.shape[:2]

    warped = cv2.remap(
        image, src_x, src_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    hole_pct = 100.0 * float(np.count_nonzero(occ_mask)) / float(h * w)
    logger.debug("Z-buf+remap %s eye: %.1f%% occlusion holes", eye, hole_pct)
    return warped, occ_mask


# =====================================================================
# Legacy: row-loop forward splatting (reference only)
# =====================================================================

def forward_splat_eye(
    image: np.ndarray,
    depth_f32: np.ndarray,
    max_disparity: float,
    eye: EyeSide,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-splat with per-row z-buffer.  Kept for reference."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 RGB image, got shape {image.shape}")
    if depth_f32.shape[:2] != image.shape[:2]:
        raise ValueError("Depth map must match image height and width")

    h, w = image.shape[:2]
    half_sep = max_disparity * 0.5
    sign = 1.0 if eye == "left" else -1.0

    out = np.zeros((h, w, 3), dtype=np.uint8)
    z_buf = np.full((h, w), -1.0, dtype=np.float32)

    src_indices = np.arange(w, dtype=np.int32)
    src_x_f = np.arange(w, dtype=np.float32)

    for y in range(h):
        row_d = depth_f32[y]
        row_rgb = image[y]
        shifts_row = row_d * (half_sep * sign)
        dst_f = src_x_f + shifts_row

        dst_lo = np.floor(dst_f).astype(np.int32)
        dst_hi = dst_lo + 1

        all_src = np.concatenate([src_indices, src_indices])
        all_dst = np.concatenate([dst_lo, dst_hi])
        all_dep = np.concatenate([row_d, row_d])

        valid = (all_dst >= 0) & (all_dst < w)
        all_src = all_src[valid]
        all_dst = all_dst[valid]
        all_dep = all_dep[valid]

        order = np.argsort(all_dep, kind="stable")
        all_src = all_src[order]
        all_dst = all_dst[order]

        out[y, all_dst] = row_rgb[all_src]
        z_buf[y, all_dst] = all_dep[order]

    occ_mask = np.where(z_buf < 0.0, np.uint8(255), np.uint8(0))

    hole_pct = 100.0 * float(np.count_nonzero(occ_mask)) / float(h * w)
    logger.debug("Forward-splat %s eye: %.1f%% occlusion holes", eye, hole_pct)
    return out, occ_mask
