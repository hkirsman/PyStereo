"""Shared helpers for the ``notes`` diagnostics both splat renderers return.

Torch-free on purpose: ``taichi_full`` keeps torch out of its render path.
"""

from __future__ import annotations

import math

import numpy as np

#: Convergence to fall back on when the depth buffer offers nothing usable.
DEFAULT_CONVERGE_M = 2.0


def disparity_px(f_px: float, baseline_m: float, z_m: float, converge_m: float) -> float:
    """Horizontal disparity in pixels at depth ``z_m``, 0 when undefined.

    ``z_m`` is a percentile of the rendered depth buffer, so it is 0 or NaN
    whenever that slice of the frame never received a splat. The notes are
    diagnostics - a missing depth must not sink a render that already
    succeeded, which a bare ``1 / z_m`` did with ZeroDivisionError.
    """
    if not math.isfinite(z_m) or z_m <= 0:
        return 0.0
    if not math.isfinite(converge_m) or converge_m <= 0:
        return 0.0
    return round(f_px * baseline_m * (1 / z_m - 1 / converge_m), 1)


def depth_to_01(depth: np.ndarray) -> np.ndarray:
    """Normalise a metric depth buffer to [0, 1] via inverse depth.

    NaN (holes) and zero or negative depth (invalid) map to the farthest
    valid distance, and a buffer with nothing valid at all comes back as
    zeros - taking min/max over the empty selection would raise instead.
    """
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros_like(depth, dtype=np.float32)
    inv = np.zeros_like(depth)
    inv[valid] = 1.0 / depth[valid]
    far = float(inv[valid].min())
    inv[~valid] = far
    lo, hi = far, float(inv[valid].max())
    return ((inv - lo) / max(hi - lo, 1e-6)).astype(np.float32)


def converge_distance(
    depth: np.ndarray,
    subject_mask: np.ndarray | None = None,
    *,
    default_m: float = DEFAULT_CONVERGE_M,
) -> float:
    """Distance the two virtual cameras converge on, in metres.

    Median depth across *subject_mask*, or the 10th percentile of the frame
    without one. Both renderers leave uncovered pixels as NaN, so a frame -
    or a subject - that never received a splat would otherwise give a NaN
    convergence, and the shift derived from it would carry NaN into the
    render. Fall back to *default_m* instead.
    """
    usable = np.isfinite(depth) & (depth > 0)
    if subject_mask is not None and (subject_mask & usable).any():
        z = float(np.median(depth[subject_mask & usable]))
    elif usable.any():
        z = float(np.percentile(depth[usable], 10))
    else:
        z = default_m
    return z if math.isfinite(z) and z > 0 else default_m
