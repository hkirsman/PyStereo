"""Module-level Taichi kernel for Gaussian projection (full-taichi renderer).

Companion to ``_taichi_kernels.py``: same frozen-build rule applies - Taichi
compiles kernels by reading Python source via ``inspect``, so the kernel
lives at module scope and this module is imported only after ``ti.init()``
(see ``taichi_render.is_taichi_available``).

The compositing pass is reused from ``_taichi_kernels.alpha_raster_pass``;
this module adds only the 3D -> 2D projection that ``splat_render.
SharpScene.project`` otherwise does in torch: quaternion -> rotation ->
3D covariance -> EWA 2D covariance -> screen position, inverse covariance
and pixel radius. ``radius_out = 0`` marks a culled Gaussian (behind the
camera, transparent, degenerate covariance, or off-screen); u/v/z/inv of
culled entries are left untouched and must be filtered by the caller.
"""

import taichi as ti


@ti.kernel
def project_pass(
    means: ti.types.ndarray(dtype=ti.f32, ndim=2),
    scales: ti.types.ndarray(dtype=ti.f32, ndim=2),
    quats: ti.types.ndarray(dtype=ti.f32, ndim=2),
    opac: ti.types.ndarray(dtype=ti.f32, ndim=1),
    u_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    v_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    z_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    inv_out: ti.types.ndarray(dtype=ti.f32, ndim=2),
    radius_out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    eye_x: ti.f32,
    cx_shift: ti.f32,
    f: ti.f32,
    W_: ti.i32,
    H_: ti.i32,
    r_max: ti.i32,
    soft_t: ti.f32,
):
    cx = (ti.cast(W_, ti.f32) - 1.0) / 2.0 + cx_shift
    cy = (ti.cast(H_, ti.f32) - 1.0) / 2.0
    for i in range(means.shape[0]):
        radius_out[i] = 0.0
        mx = means[i, 0] - eye_x
        my = means[i, 1]
        mz = means[i, 2]
        if mz <= 0.05 or opac[i] <= soft_t:
            continue
        # Quaternion (w,x,y,z) -> rotation matrix.
        qw = quats[i, 0]
        qx = quats[i, 1]
        qy = quats[i, 2]
        qz = quats[i, 3]
        r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        r01 = 2.0 * (qx * qy - qw * qz)
        r02 = 2.0 * (qx * qz + qw * qy)
        r10 = 2.0 * (qx * qy + qw * qz)
        r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        r12 = 2.0 * (qy * qz - qw * qx)
        r20 = 2.0 * (qx * qz - qw * qy)
        r21 = 2.0 * (qy * qz + qw * qx)
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        s0 = scales[i, 0] * scales[i, 0]
        s1 = scales[i, 1] * scales[i, 1]
        s2 = scales[i, 2] * scales[i, 2]
        # cov3 = R * diag(s^2) * R^T (symmetric, 6 unique terms).
        c00 = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2
        c01 = r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2
        c02 = r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2
        c11 = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2
        c12 = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2
        c22 = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2
        # EWA: cov2 = J cov3 J^T, J = [[f/z, 0, -f x/z^2], [0, f/z, -f y/z^2]],
        # plus the same 0.3 px^2 low-pass as splat_render.SharpScene.project.
        j00 = f / mz
        j02 = -f * mx / (mz * mz)
        j11 = f / mz
        j12 = -f * my / (mz * mz)
        a = j00 * (j00 * c00 + j02 * c02) + j02 * (j00 * c02 + j02 * c22) + 0.3
        b = j00 * (j11 * c01 + j12 * c02) + j02 * (j11 * c12 + j12 * c22)
        c = j11 * (j11 * c11 + j12 * c12) + j12 * (j11 * c12 + j12 * c22) + 0.3
        det = a * c - b * b
        if det <= 0.0:
            continue
        eig_max = 0.5 * (a + c) + ti.sqrt(ti.max(0.25 * (a - c) * (a - c) + b * b, 0.0))
        rad = ti.min(ti.max(ti.ceil(3.0 * ti.sqrt(eig_max)), 1.0), ti.cast(r_max, ti.f32))
        u = f * mx / mz + cx
        v = f * my / mz + cy
        fr = ti.cast(r_max, ti.f32)
        if u <= -fr or u >= ti.cast(W_, ti.f32) + fr or v <= -fr or v >= ti.cast(H_, ti.f32) + fr:
            continue
        u_out[i] = u
        v_out[i] = v
        z_out[i] = mz
        inv_out[i, 0] = c / det
        inv_out[i, 1] = -b / det
        inv_out[i, 2] = -b / det
        inv_out[i, 3] = a / det
        radius_out[i] = rad


def probe_kernels() -> None:
    """Compile the projection kernel with minimal inputs (raises on failure)."""
    import numpy as np

    empty1 = np.zeros(0, dtype=np.float32)
    project_pass(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0, 4), dtype=np.float32),
        empty1,
        empty1.copy(), empty1.copy(), empty1.copy(),
        np.zeros((0, 4), dtype=np.float32),
        empty1.copy(),
        0.0, 0.0, 1.0, 1, 1, 7, 0.04,
    )
