"""Forward mesh stereo rendering from a SHARP depth map.

Renders the original photo from virtual stereo cameras using depth-based
forward splatting with a z-buffer painter's algorithm. Small gaps between
pixels are filled by morphological dilation; large disoccluded regions
remain stretched (no AI inpainting).

Visual character: sharp edges at depth boundaries, slight stretching in
disoccluded bands, no inpainting artifacts.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _forward_render(
    rgb: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    eye_x: float,
    cx_shift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-warp *rgb* through *depth* from a shifted camera.

    Uses a painter's algorithm (far-to-near scatter) for z-buffering and
    morphological dilation to fill small gaps.

    Returns (rendered uint8 HxWx3, hole mask bool HxW).
    """
    H, W = depth.shape
    cx0 = (W - 1) / 2.0
    cy = (H - 1) / 2.0
    cx = cx0 + cx_shift

    z = np.nan_to_num(depth, nan=float(np.nanmax(depth))).astype(np.float32)
    valid = z > 0.05

    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )

    x3d = (u_grid - cx0) / f_px * z
    y3d = (v_grid - cy) / f_px * z
    x_eye = x3d - eye_x

    u_proj = f_px * x_eye / z + cx
    v_proj = f_px * y3d / z + cy

    flat_z = z.ravel()
    flat_u = np.round(u_proj.ravel()).astype(np.intp)
    flat_v = np.round(v_proj.ravel()).astype(np.intp)
    flat_rgb = rgb.reshape(-1, 3)
    flat_valid = valid.ravel()

    in_bounds = flat_valid & (flat_u >= 0) & (flat_u < W) & (flat_v >= 0) & (flat_v < H)
    indices = np.where(in_bounds)[0]
    indices = indices[np.argsort(flat_z[indices])[::-1]]

    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[flat_v[indices], flat_u[indices]] = flat_rgb[indices]

    filled = np.zeros((H, W), dtype=bool)
    filled[flat_v[indices], flat_u[indices]] = True

    holes = ~filled
    if holes.any():
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(6):
            dilated = cv2.dilate(out, kernel, iterations=1)
            out[holes] = dilated[holes]
            filled = out.sum(axis=-1) > 0
            holes = ~filled
            if not holes.any():
                break

    return out, ~filled


def render_mesh_stereo(
    rgb: np.ndarray,
    depth: np.ndarray,
    f_px: float,
    baseline_m: float,
    converge_m: float,
) -> dict[str, np.ndarray]:
    """Render a stereo pair via forward mesh splatting.

    Parameters
    ----------
    rgb:
        Original photo ``(H, W, 3)`` uint8 sRGB.
    depth:
        Metric depth ``(H, W)`` float32 in metres (from SHARP centre render).
    f_px:
        Focal length in pixels.
    baseline_m:
        Inter-ocular distance in metres.
    converge_m:
        Convergence distance in metres (screen plane).

    Returns
    -------
    Dict with ``left``, ``right`` (uint8 RGB arrays), ``holes_left``,
    ``holes_right`` (bool masks).
    """
    shift = f_px * baseline_m / (2 * converge_m)
    left, holes_l = _forward_render(rgb, depth, f_px, -baseline_m / 2, -shift)
    right, holes_r = _forward_render(rgb, depth, f_px, +baseline_m / 2, +shift)

    logger.info(
        "Mesh stereo: %dx%d, baseline=%.3fm, converge=%.2fm, "
        "holes_left=%.2f%%, holes_right=%.2f%%",
        rgb.shape[1], rgb.shape[0], baseline_m, converge_m,
        holes_l.mean() * 100, holes_r.mean() * 100,
    )

    return {
        "left": left,
        "right": right,
        "holes_left": holes_l,
        "holes_right": holes_r,
    }
