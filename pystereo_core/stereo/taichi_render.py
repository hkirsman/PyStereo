"""Taichi-accelerated Gaussian splatting (optional, falls back to torch).

Two kernels:

- :func:`composite_ewa_taichi` - same 2-pass z-buffer algorithm as
  :func:`splat_render.composite_ewa_torch`, per-Gaussian scatter on
  Metal (macOS) or CUDA, typically 5-10x faster.
- :func:`composite_alpha_taichi` - depth-sorted front-to-back alpha
  compositing, same result as :func:`splat_render.composite_alpha_torch`.
  Standard 3DGS tile rasteriser: Gaussians are binned into 16x16 pixel
  tiles and sorted by depth per tile (numpy), then one thread per pixel
  walks its tile's list with early termination. ~0.2 s instead of ~50 s
  per eye for 3.5 M Gaussians.

Install with ``pip install taichi`` (~50 MB). Taichi 1.7 ships wheels for
Python 3.9-3.13; on newer interpreters the torch fallback is used
transparently.

PyInstaller / frozen builds: Taichi needs readable kernel source. Kernels
live in ``_taichi_kernels.py`` at module scope; if compilation still
fails, :func:`is_taichi_available` returns False and callers use torch.
"""

# No ``from __future__ import annotations`` here: taichi reads kernel argument
# annotations at runtime and cannot parse them as strings.
import logging
import sys
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from pystereo_core.stereo.splat_render import _Projected

logger = logging.getLogger(__name__)

_ti = None
_ti_checked = False
_kernels = None


def _load_kernels():
    global _kernels
    if _kernels is None:
        from pystereo_core.stereo import _taichi_kernels

        _kernels = _taichi_kernels
    return _kernels


def is_taichi_available() -> bool:
    global _ti, _ti_checked
    if _ti_checked:
        return _ti is not None
    _ti_checked = True
    try:
        import taichi as ti
        import torch

        if torch.cuda.is_available():
            arch = ti.cuda
        elif torch.backends.mps.is_available():
            arch = ti.metal
        else:
            arch = ti.cpu
        ti.init(arch=arch, log_level=ti.WARN)
        _load_kernels().probe_kernels()
        _ti = ti
        logger.info("Taichi initialised (arch=%s)", arch)
        return True
    except Exception as exc:
        if getattr(sys, "frozen", False):
            logger.info(
                "Taichi kernels unavailable in packaged app (%s); using torch fallback",
                exc,
            )
        else:
            logger.info("Taichi not available (%s); will use torch fallback", exc)
        _ti = None
        return False


def composite_ewa_taichi(
    p: "_Projected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EWA splatting via taichi - same result as the torch path, faster on GPU."""
    from pystereo_core.stereo.splat_render import DEPTH_TOL, HARD_T, R_MAX, SOFT_T

    if _ti is None:
        raise RuntimeError("Taichi not initialised")

    kernels = _load_kernels()
    H, W = p.H, p.W
    N = len(p.z)

    u_np = p.u.cpu().numpy().astype(np.float32)
    v_np = p.v.cpu().numpy().astype(np.float32)
    z_np = p.z.cpu().numpy().astype(np.float32)
    inv_np = p.inv.cpu().numpy().astype(np.float32).reshape(-1, 4)
    radius_np = p.radius.cpu().numpy().astype(np.float32)
    col_np = p.col.cpu().numpy().astype(np.float32)
    op_np = p.op.cpu().numpy().astype(np.float32)

    zbuf = np.full(H * W, np.inf, dtype=np.float32)
    acc = np.zeros((H * W, 3), dtype=np.float32)
    wsum = np.zeros(H * W, dtype=np.float32)

    kernels.ewa_zbuffer_pass(
        u_np, v_np, z_np, inv_np, radius_np, op_np, zbuf,
        W, H, N, R_MAX, HARD_T,
    )
    kernels.ewa_composite_pass(
        u_np, v_np, z_np, inv_np, radius_np, col_np, op_np,
        zbuf, acc, wsum, W, H, N, SOFT_T, DEPTH_TOL,
    )

    holes = wsum < 1e-4
    safe_w = np.maximum(wsum, 1e-4)
    rgb = (acc / safe_w[:, None]).reshape(H, W, 3)
    depth_out = zbuf.reshape(H, W)
    depth_out[~np.isfinite(depth_out)] = np.nan
    return rgb, depth_out, holes.reshape(H, W).astype(np.uint8) * 255


TILE = 16


def _bin_tiles(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, radius: np.ndarray, W: int, H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sort Gaussians into TILE x TILE screen tiles, depth-ordered within a tile.

    Returns (gaussian index list, per-tile start, per-tile end, tiles per row).
    A splat radius is capped at R_MAX < TILE, so a Gaussian overlaps at most
    2x2 tiles.
    """
    tw, th = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    pu, pv = np.round(u), np.round(v)
    tx0 = np.clip((pu - radius) // TILE, 0, tw - 1).astype(np.int64)
    tx1 = np.clip((pu + radius) // TILE, 0, tw - 1).astype(np.int64)
    ty0 = np.clip((pv - radius) // TILE, 0, th - 1).astype(np.int64)
    ty1 = np.clip((pv + radius) // TILE, 0, th - 1).astype(np.int64)
    valid = (pu + radius >= 0) & (pu - radius < W) & (pv + radius >= 0) & (pv - radius < H)
    gids: list[np.ndarray] = []
    tids: list[np.ndarray] = []
    for dy in range(2):
        for dx in range(2):
            ty, tx = ty0 + dy, tx0 + dx
            g = np.flatnonzero(valid & (ty <= ty1) & (tx <= tx1))
            gids.append(g)
            tids.append(ty[g] * tw + tx[g])
    g = np.concatenate(gids)
    t = np.concatenate(tids)
    order = np.lexsort((z[g], t))
    g, t = g[order].astype(np.int32), t[order]
    tiles = np.arange(tw * th)
    starts = np.searchsorted(t, tiles).astype(np.int32)
    ends = np.searchsorted(t, tiles, side="right").astype(np.int32)
    return g, starts, ends, tw


def composite_alpha_taichi(
    p: "_Projected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Depth-sorted alpha compositing via a taichi tile rasteriser.

    Same rule as :func:`splat_render.composite_alpha_torch` (front-to-back
    ``C = sum T_i a_i c_i``, median depth at accumulated alpha 0.5, nearest
    contribution if never reached, hole where accumulated alpha < 0.01).
    """
    from pystereo_core.stereo.splat_render import ALPHA_MAX, MEDIAN_ALPHA, SOFT_T

    if _ti is None:
        raise RuntimeError("Taichi not initialised")

    kernels = _load_kernels()
    H, W = p.H, p.W
    u_np = p.u.cpu().numpy().astype(np.float32)
    v_np = p.v.cpu().numpy().astype(np.float32)
    z_np = p.z.cpu().numpy().astype(np.float32)
    inv_np = p.inv.cpu().numpy().astype(np.float32).reshape(-1, 4)
    radius_np = p.radius.cpu().numpy().astype(np.float32)
    col_np = p.col.cpu().numpy().astype(np.float32)
    op_np = p.op.cpu().numpy().astype(np.float32)

    glist, starts, ends, tw = _bin_tiles(u_np, v_np, z_np, radius_np, W, H)

    rgb = np.zeros((H * W, 3), dtype=np.float32)
    depth = np.full(H * W, np.nan, dtype=np.float32)
    asum_out = np.zeros(H * W, dtype=np.float32)

    kernels.alpha_raster_pass(
        u_np, v_np, z_np, inv_np, radius_np, col_np, op_np, glist, starts, ends,
        rgb, depth, asum_out, W, H, tw, TILE, SOFT_T, ALPHA_MAX, MEDIAN_ALPHA,
    )

    holes = (asum_out < 0.01).reshape(H, W).astype(np.uint8) * 255
    return rgb.reshape(H, W, 3), depth.reshape(H, W), holes


def select_ewa_compositor() -> Callable:
    """Return the best available EWA compositor (taichi or torch)."""
    from pystereo_core.stereo.splat_render import composite_ewa_torch

    if is_taichi_available():
        return composite_ewa_taichi
    return composite_ewa_torch
