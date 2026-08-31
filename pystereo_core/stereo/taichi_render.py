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
"""

# No ``from __future__ import annotations`` here: taichi reads kernel argument
# annotations at runtime and cannot parse them as strings.
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pystereo_core.stereo.splat_render import _Projected

logger = logging.getLogger(__name__)

_ti = None
_ti_checked = False


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
        _ti = ti
        logger.info("Taichi initialised (arch=%s)", arch)
        return True
    except Exception as exc:
        logger.info("Taichi not available (%s); will use torch fallback", exc)
        return False


def composite_ewa_taichi(
    p: "_Projected",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EWA splatting via taichi - same result as the torch path, faster on GPU."""
    import torch

    from pystereo_core.stereo.splat_render import DEPTH_TOL, HARD_T, R_MAX, SOFT_T

    ti = _ti
    if ti is None:
        raise RuntimeError("Taichi not initialised")

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

    @ti.kernel
    def zbuffer_pass(
        u: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inv_flat: ti.types.ndarray(dtype=ti.f32, ndim=2),
        radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
        op: ti.types.ndarray(dtype=ti.f32, ndim=1),
        zbuf_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        W_: ti.i32, H_: ti.i32, N_: ti.i32,
        r_max: ti.i32, hard_t: ti.f32,
    ):
        for idx in range(N_):
            mu_x = u[idx]
            mu_y = v[idx]
            rad = ti.cast(radius[idx], ti.i32)
            x0 = ti.max(0, ti.cast(ti.round(mu_x) - rad, ti.i32))
            x1 = ti.min(W_, ti.cast(ti.round(mu_x) + rad + 1, ti.i32))
            y0 = ti.max(0, ti.cast(ti.round(mu_y) - rad, ti.i32))
            y1 = ti.min(H_, ti.cast(ti.round(mu_y) + rad + 1, ti.i32))
            a = inv_flat[idx, 0]
            b = inv_flat[idx, 1]
            c = inv_flat[idx, 3]
            o = op[idx]
            d = z[idx]
            for py in range(y0, y1):
                for px in range(x0, x1):
                    dx = ti.cast(px, ti.f32) - mu_x
                    dy = ti.cast(py, ti.f32) - mu_y
                    q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
                    w = o * ti.exp(-0.5 * q)
                    if w > hard_t:
                        flat = py * W_ + px
                        ti.atomic_min(zbuf_out[flat], d)

    @ti.kernel
    def composite_pass(
        u: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inv_flat: ti.types.ndarray(dtype=ti.f32, ndim=2),
        radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
        col: ti.types.ndarray(dtype=ti.f32, ndim=2),
        op: ti.types.ndarray(dtype=ti.f32, ndim=1),
        zbuf_in: ti.types.ndarray(dtype=ti.f32, ndim=1),
        acc_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
        wsum_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        W_: ti.i32, H_: ti.i32, N_: ti.i32,
        soft_t: ti.f32, depth_tol: ti.f32,
    ):
        for idx in range(N_):
            mu_x = u[idx]
            mu_y = v[idx]
            rad = ti.cast(radius[idx], ti.i32)
            x0 = ti.max(0, ti.cast(ti.round(mu_x) - rad, ti.i32))
            x1 = ti.min(W_, ti.cast(ti.round(mu_x) + rad + 1, ti.i32))
            y0 = ti.max(0, ti.cast(ti.round(mu_y) - rad, ti.i32))
            y1 = ti.min(H_, ti.cast(ti.round(mu_y) + rad + 1, ti.i32))
            a = inv_flat[idx, 0]
            b = inv_flat[idx, 1]
            c = inv_flat[idx, 3]
            o = op[idx]
            d = z[idx]
            r = col[idx, 0]
            g = col[idx, 1]
            bl = col[idx, 2]
            for py in range(y0, y1):
                for px in range(x0, x1):
                    dx = ti.cast(px, ti.f32) - mu_x
                    dy = ti.cast(py, ti.f32) - mu_y
                    q = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy
                    w = o * ti.exp(-0.5 * q)
                    if w > soft_t:
                        flat = py * W_ + px
                        if ti.abs(d - zbuf_in[flat]) <= zbuf_in[flat] * depth_tol:
                            ti.atomic_add(wsum_out[flat], w)
                            ti.atomic_add(acc_out[flat, 0], w * r)
                            ti.atomic_add(acc_out[flat, 1], w * g)
                            ti.atomic_add(acc_out[flat, 2], w * bl)

    zbuffer_pass(u_np, v_np, z_np, inv_np, radius_np, op_np, zbuf,
                 W, H, N, R_MAX, HARD_T)
    composite_pass(u_np, v_np, z_np, inv_np, radius_np, col_np, op_np,
                   zbuf, acc, wsum, W, H, N, SOFT_T, DEPTH_TOL)

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

    ti = _ti
    if ti is None:
        raise RuntimeError("Taichi not initialised")

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

    @ti.kernel
    def raster(
        u: ti.types.ndarray(dtype=ti.f32, ndim=1),
        v: ti.types.ndarray(dtype=ti.f32, ndim=1),
        z: ti.types.ndarray(dtype=ti.f32, ndim=1),
        inv_flat: ti.types.ndarray(dtype=ti.f32, ndim=2),
        radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
        col: ti.types.ndarray(dtype=ti.f32, ndim=2),
        op: ti.types.ndarray(dtype=ti.f32, ndim=1),
        gl: ti.types.ndarray(dtype=ti.i32, ndim=1),
        st: ti.types.ndarray(dtype=ti.i32, ndim=1),
        en: ti.types.ndarray(dtype=ti.i32, ndim=1),
        rgb_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
        depth_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
        asum_o: ti.types.ndarray(dtype=ti.f32, ndim=1),
        W_: ti.i32, H_: ti.i32, tw_: ti.i32, tile: ti.i32,
        soft_t: ti.f32, a_max: ti.f32, med_alpha: ti.f32,
    ):
        for pix in range(W_ * H_):
            px = pix % W_
            py = pix // W_
            tid = (py // tile) * tw_ + px // tile
            fx = ti.cast(px, ti.f32)
            fy = ti.cast(py, ti.f32)
            T = 1.0
            ar = 0.0
            ag = 0.0
            ab = 0.0
            asum = 0.0
            med = ti.f32(1e30)
            near = ti.f32(1e30)
            for k in range(st[tid], en[tid]):
                if T < 1e-4:
                    break
                gi = gl[k]
                if ti.abs(fx - ti.round(u[gi])) > radius[gi] or ti.abs(fy - ti.round(v[gi])) > radius[gi]:
                    continue
                dx = fx - u[gi]
                dy = fy - v[gi]
                q = inv_flat[gi, 0] * dx * dx + 2.0 * inv_flat[gi, 1] * dx * dy + inv_flat[gi, 3] * dy * dy
                a = ti.min(op[gi] * ti.exp(-0.5 * q), a_max)
                if a <= soft_t:
                    continue
                if near > 1e29:
                    near = z[gi]
                w = T * a
                ar += w * col[gi, 0]
                ag += w * col[gi, 1]
                ab += w * col[gi, 2]
                asum += w
                if med > 1e29 and asum >= med_alpha:
                    med = z[gi]
                T *= 1.0 - a
            s = ti.max(asum, 1e-4)
            rgb_out[pix, 0] = ar / s
            rgb_out[pix, 1] = ag / s
            rgb_out[pix, 2] = ab / s
            asum_o[pix] = asum
            if med < 1e29:
                depth_out[pix] = med
            elif near < 1e29:
                depth_out[pix] = near

    raster(u_np, v_np, z_np, inv_np, radius_np, col_np, op_np, glist, starts, ends,
           rgb, depth, asum_out, W, H, tw, TILE, SOFT_T, ALPHA_MAX, MEDIAN_ALPHA)

    holes = (asum_out < 0.01).reshape(H, W).astype(np.uint8) * 255
    return rgb.reshape(H, W, 3), depth.reshape(H, W), holes
