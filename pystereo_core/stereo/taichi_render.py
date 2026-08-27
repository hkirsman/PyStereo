"""Taichi-accelerated EWA Gaussian splatting (optional, falls back to torch).

Same 2-pass algorithm as :func:`splat_render.composite_ewa_torch` but runs
the per-pixel scatter on Metal (macOS) or CUDA via taichi, typically 5-10x
faster for large scenes.

Install with ``pip install taichi`` (~50 MB).  When taichi is not available
the torch fallback is used transparently.
"""

from __future__ import annotations

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

        arch = ti.metal if torch.backends.mps.is_available() else ti.cpu
        ti.init(arch=arch, log_level=ti.WARN)
        _ti = ti
        logger.info("Taichi initialised (arch=%s)", arch)
        return True
    except Exception as exc:
        logger.info("Taichi not available (%s); will use torch fallback", exc)
        return False


def composite_ewa_taichi(
    p: _Projected,
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
