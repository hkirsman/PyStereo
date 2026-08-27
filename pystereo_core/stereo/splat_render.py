"""Stereo pair from Apple SHARP Gaussians without gsplat/CUDA.

Virtual stereo rig: two pinhole cameras offset +-baseline/2 along x, parallel
axes, converged on the subject by shifting the principal point (no keystone).
Renderer: project every Gaussian (EWA 2D covariance), z-buffer the front
surface per pixel, then blend the contributions within a small depth
tolerance of that surface for anti-aliased edges. Pure torch (MPS or CPU).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

R_MAX = 7
HARD_T = 0.35
SOFT_T = 0.04
DEPTH_TOL = 0.03


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class _Projected:
    """2D projections of Gaussians for a given camera position."""

    __slots__ = ("u", "v", "z", "inv", "radius", "col", "op", "pu", "pv", "W", "H", "dev")

    def __init__(
        self, u: torch.Tensor, v: torch.Tensor, z: torch.Tensor,
        inv: torch.Tensor, radius: torch.Tensor, col: torch.Tensor,
        op: torch.Tensor, pu: torch.Tensor, pv: torch.Tensor,
        W: int, H: int, dev: torch.device,
    ) -> None:
        self.u, self.v, self.z = u, v, z
        self.inv, self.radius = inv, radius
        self.col, self.op = col, op
        self.pu, self.pv = pu, pv
        self.W, self.H, self.dev = W, H, dev


class SharpScene:
    """Loaded SHARP Gaussian scene, ready for stereo rendering."""

    def __init__(self, npz_path: str) -> None:
        d = np.load(npz_path)
        dev = _device()
        f = lambda k: torch.from_numpy(d[k].astype(np.float32)).to(dev)
        self.means = f("means")
        self.scales = f("scales")
        self.colors = f("colors")
        self.opac = f("opacities")
        self.rot = _quat_to_rot(f("quats"))
        self.f_px = float(d["f_px"])
        self.width = int(d["width"])
        self.height = int(d["height"])
        self.dev = dev
        s2 = torch.diag_embed(self.scales ** 2)
        self.cov3 = self.rot @ s2 @ self.rot.transpose(1, 2)

    @torch.no_grad()
    def project(self, eye_x: float, cx_shift: float) -> _Projected:
        """Project 3D Gaussians to 2D for a given camera position."""
        W, H, f, dev = self.width, self.height, self.f_px, self.dev
        cx, cy = (W - 1) / 2 + cx_shift, (H - 1) / 2
        m = self.means.clone()
        m[:, 0] -= eye_x
        z = m[:, 2]
        keep = (z > 0.05) & (self.opac > SOFT_T)
        m, z = m[keep], z[keep]
        cov3, col, op = self.cov3[keep], self.colors[keep], self.opac[keep]
        u = f * m[:, 0] / z + cx
        v = f * m[:, 1] / z + cy

        J = torch.zeros((len(z), 2, 3), device=dev)
        J[:, 0, 0] = f / z
        J[:, 1, 1] = f / z
        J[:, 0, 2] = -f * m[:, 0] / z ** 2
        J[:, 1, 2] = -f * m[:, 1] / z ** 2
        cov2 = J @ cov3 @ J.transpose(1, 2)
        cov2[:, 0, 0] += 0.3
        cov2[:, 1, 1] += 0.3
        det = cov2[:, 0, 0] * cov2[:, 1, 1] - cov2[:, 0, 1] ** 2
        inv = torch.stack([
            cov2[:, 1, 1], -cov2[:, 0, 1], -cov2[:, 0, 1], cov2[:, 0, 0],
        ], -1).reshape(-1, 2, 2) / det[:, None, None]
        eig_max = 0.5 * (cov2[:, 0, 0] + cov2[:, 1, 1]) + torch.sqrt(
            torch.clamp(
                0.25 * (cov2[:, 0, 0] - cov2[:, 1, 1]) ** 2
                + cov2[:, 0, 1] ** 2,
                min=0,
            )
        )
        radius = torch.clamp(torch.ceil(3 * torch.sqrt(eig_max)), 1, R_MAX)
        inside = (u > -R_MAX) & (u < W + R_MAX) & (v > -R_MAX) & (v < H + R_MAX)
        u, v, z = u[inside], v[inside], z[inside]
        inv, radius = inv[inside], radius[inside]
        col, op = col[inside], op[inside]
        pu, pv = torch.round(u), torch.round(v)
        return _Projected(u, v, z, inv, radius, col, op, pu, pv, W, H, dev)

    @torch.no_grad()
    def render(
        self, eye_x: float, cx_shift: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render from a camera at (eye_x, 0, 0) with principal point shifted.

        Returns (rgb float linear HxWx3, depth HxW metres, hole mask uint8).
        """
        p = self.project(eye_x, cx_shift)
        return composite_ewa_torch(p)


def composite_ewa_torch(p: _Projected) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EWA splatting compositing pass (pure torch, 2-pass z-buffer + blend)."""
    W, H, dev = p.W, p.H, p.dev
    zbuf = torch.full((H * W,), float("inf"), device=dev)
    offsets = [
        (dx, dy)
        for dy in range(-R_MAX, R_MAX + 1)
        for dx in range(-R_MAX, R_MAX + 1)
    ]

    def contributions(dx: int, dy: int):  # noqa: ANN202
        px, py = p.pu + dx, p.pv + dy
        ok = (
            (px >= 0) & (px < W) & (py >= 0) & (py < H)
            & (abs(dx) <= p.radius) & (abs(dy) <= p.radius)
        )
        d0, d1 = px - p.u, py - p.v
        q = p.inv[:, 0, 0] * d0 * d0 + 2 * p.inv[:, 0, 1] * d0 * d1 + p.inv[:, 1, 1] * d1 * d1
        w = p.op * torch.exp(-0.5 * q)
        idx = (py * W + px).long().clamp(0, H * W - 1)
        return ok, w, idx

    for dx, dy in offsets:
        ok, w, idx = contributions(dx, dy)
        sel = ok & (w > HARD_T)
        zbuf.scatter_reduce_(0, idx[sel], p.z[sel], reduce="amin")

    acc = torch.zeros((H * W, 3), device=dev)
    wsum = torch.zeros((H * W,), device=dev)
    for dx, dy in offsets:
        ok, w, idx = contributions(dx, dy)
        sel = ok & (w > SOFT_T)
        idx_s, w_s = idx[sel], w[sel]
        # Two-sided: Gaussians in front of the z-buffered surface (weak tails
        # of a foreground object) must not bleed into background pixels.
        front = (p.z[sel] - zbuf[idx_s]).abs() <= zbuf[idx_s] * DEPTH_TOL
        idx_s, w_s = idx_s[front], w_s[front]
        acc.index_add_(0, idx_s, p.col[sel][front] * w_s[:, None])
        wsum.index_add_(0, idx_s, w_s)

    holes = (wsum < 1e-4)
    rgb = (acc / torch.clamp(wsum, min=1e-4)[:, None]).reshape(H, W, 3).cpu().numpy()
    depth = zbuf.reshape(H, W).cpu().numpy()
    depth[~np.isfinite(depth)] = np.nan
    return rgb, depth, holes.reshape(H, W).cpu().numpy().astype(np.uint8) * 255


def detail_transfer(
    photo: np.ndarray,
    eye_rgb: np.ndarray,
    eye_depth: np.ndarray,
    centre_depth: np.ndarray,
    f: float,
    eye_x: float,
    cx_shift: float,
    tol: float = 0.03,
    edge_px: int = 3,
    edge_jump: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-sample the original photo through the eye view's rendered depth.

    Each eye pixel + depth is a 3D point; project it into the original camera
    and take the photo's colour there, unless the photo pixel there is not the
    surface this eye sees: (a) it is hidden behind something in the original
    (real disocclusion, ``tol``), or (b) a surface at least ``edge_jump``
    nearer lies within ``edge_px`` of it. SHARP's depth silhouette sits 1-2 px
    inside the photo's real silhouette, so without (b) those photo pixels
    carry the subject's anti-aliased edge and draw a dark outline into the
    background. In both cases keep the splat render's colour.
    Returns (rgb uint8, mask of splat-filled pixels).
    """
    H, W = eye_depth.shape
    cx0, cy = (W - 1) / 2, (H - 1) / 2
    cx = cx0 + cx_shift
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.nan_to_num(eye_depth, nan=np.nanmax(eye_depth)).astype(np.float32)
    x = (uu - cx) / f * z + eye_x
    y = (vv - cy) / f * z
    u0 = (f * x / z + cx0).astype(np.float32)
    v0 = (f * y / z + cy).astype(np.float32)
    sampled = cv2.remap(photo, u0, v0, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    d0n = np.nan_to_num(centre_depth, nan=1e9).astype(np.float32)

    def _sample(a: np.ndarray) -> np.ndarray:
        return cv2.remap(
            a, u0, v0, cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1e9,
        )

    # 3x3 max: the eye and centre depth edges differ by ~1 px, which would
    # otherwise flag a ring around every silhouette.
    d0max = _sample(cv2.dilate(d0n, np.ones((3, 3), np.uint8)))
    k = np.ones((2 * edge_px + 1,) * 2, np.uint8)
    d0min = _sample(cv2.erode(d0n, k))
    occluded = (z > d0max * (1 + tol)) | (z > d0min * (1 + edge_jump))
    outside = (u0 < 0) | (u0 > W - 1) | (v0 < 0) | (v0 > H - 1)
    use_splat = occluded | outside
    use_splat = cv2.morphologyEx(
        use_splat.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8),
    ).astype(bool)
    a = cv2.GaussianBlur(use_splat.astype(np.float32), (0, 0), 1.0)[..., None]
    splat = linear_to_srgb(eye_rgb) * 255
    out = sampled.astype(np.float32) * (1 - a) + splat * a
    return np.clip(out, 0, 255).astype(np.uint8), use_splat.astype(np.uint8) * 255


def render_stereo(
    npz_path: str,
    baseline_m: float,
    converge_m: float | None,
    subject_mask: np.ndarray | None,
    photo: np.ndarray | None = None,
) -> dict[str, Any]:
    """Render a stereo pair from a SHARP Gaussian scene.

    Parameters
    ----------
    npz_path:
        Path to the ``.npz`` from :func:`sharp_predict.predict_gaussians`.
    baseline_m:
        Inter-ocular distance in metres (0.063 = human IPD).
    converge_m:
        Fixed convergence distance, or ``None`` for auto (median depth
        inside *subject_mask*, falling back to 10th percentile).
    subject_mask:
        ``(H, W)`` bool mask of the salient subject for convergence.
    photo:
        Original photo as ``(H, W, 3)`` uint8 sRGB for detail transfer.
        If ``None``, both eyes are pure splat renders.

    Returns
    -------
    Dict with keys ``left``, ``right`` (uint8 RGB), ``depth01`` (float32),
    ``holes`` (uint8), ``notes`` (dict of camera parameters).
    """
    scene = SharpScene(npz_path)
    f = scene.f_px

    _, d0, _ = scene.render(0.0, 0.0)
    if converge_m is None:
        if subject_mask is not None and subject_mask.any():
            converge_m = float(np.nanmedian(d0[subject_mask]))
        else:
            converge_m = float(np.nanpercentile(d0, 10))

    shift = f * baseline_m / (2 * converge_m)
    l_rgb, l_d, l_h = scene.render(-baseline_m / 2, -shift)
    r_rgb, r_d, r_h = scene.render(+baseline_m / 2, +shift)

    def finish(rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
        img = (linear_to_srgb(rgb) * 255).astype(np.uint8)
        if holes.max():
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(
                cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA),
                cv2.COLOR_BGR2RGB,
            )
        return img

    if photo is not None:
        left, l_h = detail_transfer(
            photo, l_rgb, l_d, d0, f, -baseline_m / 2, -shift,
        )
        right, r_h = detail_transfer(
            photo, r_rgb, r_d, d0, f, +baseline_m / 2, +shift,
        )
        inv = 1.0 / np.nan_to_num(l_d, nan=np.nanmax(l_d))
        depth01 = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
        zn = float(np.nanpercentile(l_d, 1))
        zf = float(np.nanpercentile(l_d, 99))
        return {
            "left": left,
            "right": right,
            "depth01": depth01.astype(np.float32),
            "holes": l_h | r_h,
            "notes": {
                "f_px": round(f, 1),
                "baseline_m": baseline_m,
                "converge_m": round(converge_m, 2),
                "disp_px_near": round(f * baseline_m * (1 / zn - 1 / converge_m), 1),
                "disp_px_far": round(f * baseline_m * (1 / zf - 1 / converge_m), 1),
                "splat_filled_pct_left": round(float((l_h > 0).mean() * 100), 2),
                "splat_filled_pct_right": round(float((r_h > 0).mean() * 100), 2),
            },
        }

    inv = 1.0 / np.nan_to_num(l_d, nan=np.nanmax(l_d))
    depth01 = (inv - inv.min()) / max(inv.max() - inv.min(), 1e-6)
    zn = float(np.nanpercentile(l_d, 1))
    zf = float(np.nanpercentile(l_d, 99))
    return {
        "left": finish(l_rgb, l_h),
        "right": finish(r_rgb, r_h),
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
