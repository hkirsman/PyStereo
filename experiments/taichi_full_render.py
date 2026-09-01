"""Experiment: fully Taichi SHARP splat renderer (zero torch).

Proof-of-concept for rendering a SHARP Gaussian scene end-to-end without
PyTorch. The production path (splat_render.py + taichi_render.py) does the
3D -> 2D projection in torch and only the compositing in Taichi; here the
projection (quaternion -> 3D covariance -> EWA 2D covariance -> screen
position/radius) is a Taichi kernel too, so the only imports are numpy,
taichi, and Pillow/cv2 for the output image.

What this does NOT cover: the SHARP prediction itself (photo -> Gaussians).
That is a ~2.8 GB ViT-based network; Taichi is a compute-kernel language,
not a neural-net inference runtime, so prediction stays in PyTorch (or a
future CoreML/ONNX export). This script starts from a cached .npz produced
by pystereo_core.stereo.sharp_predict.

Usage:
    .venv/bin/python3 experiments/taichi_full_render.py .sharp_cache/sharp_XXXX.npz
    .venv/bin/python3 experiments/taichi_full_render.py            # newest cache entry

Output: experiments/out/<name>_taichi_sbs.jpg plus timing breakdown.

Standalone by design - duplicates constants and the tile rasteriser from
pystereo_core rather than importing them, so it can never drag torch in.
"""

# No ``from __future__ import annotations``: taichi reads kernel argument
# annotations at runtime and cannot parse them as strings.
import sys
import time
from pathlib import Path

import numpy as np

# Same rendering constants as pystereo_core.stereo.splat_render.
R_MAX = 7
SOFT_T = 0.04
ALPHA_MAX = 0.99
MEDIAN_ALPHA = 0.5
TILE = 16

BASELINE_M = 0.063  # human IPD

import taichi as ti

ti.init(arch=ti.metal if sys.platform == "darwin" else ti.gpu, log_level=ti.WARN)


# --------------------------------------------------------------------------
# Kernel 1: projection. One thread per Gaussian: build the 3D covariance
# from quaternion + scales, project through a pinhole camera at (eye_x,0,0)
# with principal point shifted by cx_shift, EWA-project the covariance,
# invert it, and derive the pixel radius. radius_out = 0 marks a culled
# Gaussian (behind camera, transparent, or off-screen).
# --------------------------------------------------------------------------
@ti.kernel
def project_kernel(
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
        # cov3 = R * diag(s^2) * R^T (symmetric, keep 6 unique terms).
        c00 = r00 * r00 * s0 + r01 * r01 * s1 + r02 * r02 * s2
        c01 = r00 * r10 * s0 + r01 * r11 * s1 + r02 * r12 * s2
        c02 = r00 * r20 * s0 + r01 * r21 * s1 + r02 * r22 * s2
        c11 = r10 * r10 * s0 + r11 * r11 * s1 + r12 * r12 * s2
        c12 = r10 * r20 * s0 + r11 * r21 * s1 + r12 * r22 * s2
        c22 = r20 * r20 * s0 + r21 * r21 * s1 + r22 * r22 * s2
        # EWA: cov2 = J cov3 J^T with J = [[f/z, 0, -f x/z^2], [0, f/z, -f y/z^2]].
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


# --------------------------------------------------------------------------
# Kernel 2: tile rasteriser - same algorithm as
# pystereo_core.stereo._taichi_kernels.alpha_raster_pass (front-to-back
# alpha compositing, median depth), duplicated to stay standalone.
# --------------------------------------------------------------------------
@ti.kernel
def alpha_raster(
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
):
    for pix in range(W_ * H_):
        px = pix % W_
        py = pix // W_
        tid = (py // TILE) * tw_ + px // TILE
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
            a = ti.min(op[gi] * ti.exp(-0.5 * q), ALPHA_MAX)
            if a <= SOFT_T:
                continue
            if near > 1e29:
                near = z[gi]
            w = T * a
            ar += w * col[gi, 0]
            ag += w * col[gi, 1]
            ab += w * col[gi, 2]
            asum += w
            if med > 1e29 and asum >= MEDIAN_ALPHA:
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


def bin_tiles(
    u: np.ndarray, v: np.ndarray, z: np.ndarray, radius: np.ndarray, W: int, H: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Depth-ordered per-tile Gaussian lists (numpy port of _bin_tiles)."""
    tw, th = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    valid = radius > 0
    pu, pv = np.round(u), np.round(v)
    tx0 = np.clip((pu - radius) // TILE, 0, tw - 1).astype(np.int64)
    tx1 = np.clip((pu + radius) // TILE, 0, tw - 1).astype(np.int64)
    ty0 = np.clip((pv - radius) // TILE, 0, th - 1).astype(np.int64)
    ty1 = np.clip((pv + radius) // TILE, 0, th - 1).astype(np.int64)
    on = valid & (pu + radius >= 0) & (pu - radius < W) & (pv + radius >= 0) & (pv - radius < H)
    gids: list[np.ndarray] = []
    tids: list[np.ndarray] = []
    for dy in range(2):
        for dx in range(2):
            ty, tx = ty0 + dy, tx0 + dx
            g = np.flatnonzero(on & (ty <= ty1) & (tx <= tx1))
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


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class TaichiScene:
    """A SHARP .npz scene rendered entirely with Taichi kernels."""

    def __init__(self, npz_path: Path) -> None:
        d = np.load(npz_path)
        self.means = np.ascontiguousarray(d["means"].astype(np.float32))
        self.scales = np.ascontiguousarray(d["scales"].astype(np.float32))
        self.quats = np.ascontiguousarray(d["quats"].astype(np.float32))
        self.colors = np.ascontiguousarray(d["colors"].astype(np.float32))
        self.opac = np.ascontiguousarray(d["opacities"].astype(np.float32))
        self.f_px = float(d["f_px"])
        self.W = int(d["width"])
        self.H = int(d["height"])
        self.n = len(self.opac)

    def render(self, eye_x: float, cx_shift: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (rgb linear float HxWx3, depth HxW metres, holes uint8)."""
        n, W, H = self.n, self.W, self.H
        u = np.empty(n, np.float32)
        v = np.empty(n, np.float32)
        z = np.empty(n, np.float32)
        inv = np.empty((n, 4), np.float32)
        radius = np.empty(n, np.float32)
        project_kernel(
            self.means, self.scales, self.quats, self.opac,
            u, v, z, inv, radius,
            eye_x, cx_shift, self.f_px, W, H, R_MAX, SOFT_T,
        )
        glist, starts, ends, tw = bin_tiles(u, v, z, radius, W, H)
        rgb = np.zeros((H * W, 3), np.float32)
        depth = np.full(H * W, np.nan, np.float32)
        asum = np.zeros(H * W, np.float32)
        alpha_raster(
            u, v, z, inv, radius, self.colors, self.opac,
            glist, starts, ends, rgb, depth, asum, W, H, tw,
        )
        holes = (asum < 0.01).reshape(H, W).astype(np.uint8) * 255
        return rgb.reshape(H, W, 3), depth.reshape(H, W), holes


def render_sbs(npz_path: Path, out_path: Path, baseline_m: float = BASELINE_M) -> dict:
    import cv2

    t0 = time.time()
    scene = TaichiScene(npz_path)
    t_load = time.time() - t0

    t0 = time.time()
    _, d0, _ = scene.render(0.0, 0.0)  # centre pass: convergence + kernel warmup
    finite = d0[np.isfinite(d0) & (d0 > 0)]
    converge_m = float(np.percentile(finite, 10)) if finite.size else 2.0
    shift = scene.f_px * baseline_m / (2 * converge_m)
    t_centre = time.time() - t0

    t0 = time.time()
    l_rgb, _, l_h = scene.render(-baseline_m / 2, -shift)
    r_rgb, _, r_h = scene.render(+baseline_m / 2, +shift)
    t_eyes = time.time() - t0

    def finish(rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
        img = (linear_to_srgb(rgb) * 255).astype(np.uint8)
        if holes.max():
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
        return img

    sbs = np.concatenate([finish(l_rgb, l_h), finish(r_rgb, r_h)], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    return {
        "gaussians": scene.n,
        "size": f"{scene.W}x{scene.H}",
        "converge_m": round(converge_m, 2),
        "t_load_s": round(t_load, 2),
        "t_centre_s": round(t_centre, 2),
        "t_both_eyes_s": round(t_eyes, 2),
        "hole_pct_left": round(float((l_h > 0).mean() * 100), 3),
        "hole_pct_right": round(float((r_h > 0).mean() * 100), 3),
        "out": str(out_path),
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        npz = Path(sys.argv[1])
    else:
        cache = sorted((root / ".sharp_cache").glob("sharp_*.npz"), key=lambda p: p.stat().st_mtime)
        if not cache:
            sys.exit("No cached scenes in .sharp_cache/ - run a SHARP method once first.")
        npz = cache[-1]
    out = root / "experiments" / "out" / f"{npz.stem}_taichi_sbs.jpg"
    print(f"Scene: {npz.name}")
    stats = render_sbs(npz, out)
    for k, val in stats.items():
        print(f"  {k}: {val}")


if __name__ == "__main__":
    main()
