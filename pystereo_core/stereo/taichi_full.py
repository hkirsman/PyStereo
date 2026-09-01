"""Fully-Taichi SHARP splat renderer - no torch anywhere in the render path.

The regular renderer (``splat_render.SharpScene``) projects Gaussians with
torch and, at best, composites with Taichi (``taichi_render``). This module
moves the projection into a Taichi kernel too (``_taichi_full_kernels.
project_pass``), so rendering a cached SHARP ``.npz`` needs only numpy,
taichi, and cv2. SHARP *prediction* (photo -> Gaussians) remains PyTorch -
it is a neural network, out of scope for a compute-kernel language.

Promoted from ``experiments/taichi_full_render.py``. Compositing reuses the
proven ``_taichi_kernels.alpha_raster_pass`` tile rasteriser, so the output
matches ``splat_render.render_stereo(mode="alpha_taichi")`` up to float
ordering (measured: mean abs pixel diff ~0.4/255 on a 1.18 M Gaussian
scene).

This module deliberately does not import ``splat_render`` (which imports
torch at module level); the render constants below mirror the values there.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Mirrors splat_render.py - keep in sync.
R_MAX = 7
SOFT_T = 0.04
ALPHA_MAX = 0.99
MEDIAN_ALPHA = 0.5

_available: bool | None = None


def is_full_taichi_available() -> bool:
    """True when taichi is initialised and the projection kernel compiles."""
    global _available
    if _available is not None:
        return _available
    _available = False
    try:
        from pystereo_core.stereo.taichi_render import is_taichi_available

        if is_taichi_available():
            from pystereo_core.stereo import _taichi_full_kernels

            _taichi_full_kernels.probe_kernels()
            _available = True
    except Exception as exc:
        logger.info("Full-taichi renderer unavailable (%s)", exc)
    return _available


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _depth_to_01(depth: np.ndarray) -> np.ndarray:
    """Normalise a metric depth buffer to [0, 1] via inverse depth."""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros_like(depth, dtype=np.float32)
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / depth[valid]
    far = float(inv[valid].min())
    inv[~valid] = far
    lo, hi = far, float(inv[valid].max())
    return ((inv - lo) / max(hi - lo, 1e-6)).astype(np.float32)


class TaichiScene:
    """A SHARP ``.npz`` scene rendered entirely with Taichi kernels."""

    def __init__(self, npz_path: str) -> None:
        d = np.load(npz_path)
        self.means = np.ascontiguousarray(d["means"].astype(np.float32))
        self.scales = np.ascontiguousarray(d["scales"].astype(np.float32))
        self.quats = np.ascontiguousarray(d["quats"].astype(np.float32))
        self.colors = np.ascontiguousarray(d["colors"].astype(np.float32))
        self.opac = np.ascontiguousarray(d["opacities"].astype(np.float32))
        self.f_px = float(d["f_px"])
        self.width = int(d["width"])
        self.height = int(d["height"])
        self.n = len(self.opac)

    def render(self, eye_x: float, cx_shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render one eye. Returns (rgb linear HxWx3, depth HxW metres, holes uint8)."""
        from pystereo_core.stereo import _taichi_full_kernels, _taichi_kernels
        from pystereo_core.stereo.taichi_render import TILE, _bin_tiles

        n, W, H = self.n, self.width, self.height
        u = np.empty(n, np.float32)
        v = np.empty(n, np.float32)
        z = np.empty(n, np.float32)
        inv = np.empty((n, 4), np.float32)
        radius = np.empty(n, np.float32)
        _taichi_full_kernels.project_pass(
            self.means, self.scales, self.quats, self.opac,
            u, v, z, inv, radius,
            eye_x, cx_shift, self.f_px, W, H, R_MAX, SOFT_T,
        )
        # radius == 0 marks culled Gaussians; their u/v/z/inv are uninitialised.
        keep = radius > 0
        u, v, z = u[keep], v[keep], z[keep]
        inv, radius = np.ascontiguousarray(inv[keep]), radius[keep]
        col = np.ascontiguousarray(self.colors[keep])
        op = np.ascontiguousarray(self.opac[keep])

        glist, starts, ends, tw = _bin_tiles(u, v, z, radius, W, H)
        rgb = np.zeros((H * W, 3), np.float32)
        depth = np.full(H * W, np.nan, np.float32)
        asum = np.zeros(H * W, np.float32)
        _taichi_kernels.alpha_raster_pass(
            u, v, z, inv, radius, col, op, glist, starts, ends,
            rgb, depth, asum, W, H, tw, TILE, SOFT_T, ALPHA_MAX, MEDIAN_ALPHA,
        )
        holes = (asum < 0.01).reshape(H, W).astype(np.uint8) * 255
        return rgb.reshape(H, W, 3), depth.reshape(H, W), holes


def render_stereo_taichi(
    npz_path: str,
    baseline_m: float,
    converge_m: float | None,
    subject_mask: np.ndarray | None,
) -> dict[str, Any]:
    """Render a stereo pair from a SHARP scene, projection + compositing in Taichi.

    Same contract as ``splat_render.render_stereo`` with ``photo=None``:
    returns ``left`` / ``right`` (uint8 RGB, holes inpainted), ``center_rgb``,
    ``depth01``, ``holes``, and ``notes``. Callers must check
    :func:`is_full_taichi_available` first.
    """
    import cv2

    scene = TaichiScene(npz_path)
    f = scene.f_px

    c_rgb, d0, _ = scene.render(0.0, 0.0)
    if converge_m is None:
        if subject_mask is not None and subject_mask.any():
            converge_m = float(np.nanmedian(d0[subject_mask]))
        else:
            converge_m = float(np.nanpercentile(d0, 10))

    shift = f * baseline_m / (2 * converge_m)
    l_rgb, l_d, l_h = scene.render(-baseline_m / 2, -shift)
    r_rgb, r_d, r_h = scene.render(+baseline_m / 2, +shift)

    def finish(rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
        img = (_linear_to_srgb(rgb) * 255).astype(np.uint8)
        if holes.max():
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(
                cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA),
                cv2.COLOR_BGR2RGB,
            )
        return img

    depth01 = _depth_to_01(l_d)
    zn = float(np.nanpercentile(l_d, 1))
    zf = float(np.nanpercentile(l_d, 99))
    return {
        "left": finish(l_rgb, l_h),
        "right": finish(r_rgb, r_h),
        "center_rgb": (_linear_to_srgb(c_rgb) * 255).astype(np.uint8),
        "depth01": depth01.astype(np.float32),
        "holes": l_h | r_h,
        "notes": {
            "f_px": round(f, 1),
            "baseline_m": baseline_m,
            "converge_m": round(converge_m, 2),
            "disp_px_near": round(f * baseline_m * (1 / zn - 1 / converge_m), 1),
            "disp_px_far": round(f * baseline_m * (1 / zf - 1 / converge_m), 1),
            "hole_pct_left": round(float((l_h > 0).mean() * 100), 3),
            "hole_pct_right": round(float((r_h > 0).mean() * 100), 3),
        },
    }
