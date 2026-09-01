"""Module-level Taichi kernels for Gaussian splat compositing.

Taichi compiles kernels by reading Python source via ``inspect``. Nested
``@ti.kernel`` functions (inside ``composite_ewa_taichi`` etc.) break in
PyInstaller bundles and some frozen environments. Keep all kernels here at
module scope and import this module only after ``ti.init()`` in
``taichi_render.is_taichi_available()``.
"""

import taichi as ti


@ti.kernel
def ewa_zbuffer_pass(
    u: ti.types.ndarray(dtype=ti.f32, ndim=1),
    v: ti.types.ndarray(dtype=ti.f32, ndim=1),
    z: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inv_flat: ti.types.ndarray(dtype=ti.f32, ndim=2),
    radius: ti.types.ndarray(dtype=ti.f32, ndim=1),
    op: ti.types.ndarray(dtype=ti.f32, ndim=1),
    zbuf_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    W_: ti.i32,
    H_: ti.i32,
    N_: ti.i32,
    r_max: ti.i32,
    hard_t: ti.f32,
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
def ewa_composite_pass(
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
    W_: ti.i32,
    H_: ti.i32,
    N_: ti.i32,
    soft_t: ti.f32,
    depth_tol: ti.f32,
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


@ti.kernel
def alpha_raster_pass(
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
    W_: ti.i32,
    H_: ti.i32,
    tw_: ti.i32,
    tile: ti.i32,
    soft_t: ti.f32,
    a_max: ti.f32,
    med_alpha: ti.f32,
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


def probe_kernels() -> None:
    """Compile kernels with minimal inputs (raises if source is unavailable)."""
    import numpy as np

    u = np.zeros(0, dtype=np.float32)
    v = np.zeros(0, dtype=np.float32)
    z = np.zeros(0, dtype=np.float32)
    inv = np.zeros((0, 4), dtype=np.float32)
    radius = np.zeros(0, dtype=np.float32)
    op = np.zeros(0, dtype=np.float32)
    col = np.zeros((0, 3), dtype=np.float32)
    zbuf = np.full(1, np.inf, dtype=np.float32)
    acc = np.zeros((1, 3), dtype=np.float32)
    wsum = np.zeros(1, dtype=np.float32)
    ewa_zbuffer_pass(u, v, z, inv, radius, op, zbuf, 1, 1, 0, 1, 0.35)
    ewa_composite_pass(u, v, z, inv, radius, col, op, zbuf, acc, wsum, 1, 1, 0, 0.04, 0.03)
    glist = np.zeros(0, dtype=np.int32)
    starts = np.zeros(1, dtype=np.int32)
    ends = np.zeros(1, dtype=np.int32)
    rgb = np.zeros((1, 3), dtype=np.float32)
    depth = np.full(1, np.nan, dtype=np.float32)
    asum_out = np.zeros(1, dtype=np.float32)
    alpha_raster_pass(
        u, v, z, inv, radius, col, op, glist, starts, ends,
        rgb, depth, asum_out, 1, 1, 1, 16, 0.04, 0.99, 0.5,
    )
