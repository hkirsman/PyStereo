"""Shared helpers for the ``notes`` diagnostics both splat renderers return.

Torch-free on purpose: ``taichi_full`` keeps torch out of its render path.
"""

from __future__ import annotations

import math


def disparity_px(f_px: float, baseline_m: float, z_m: float, converge_m: float) -> float:
    """Horizontal disparity in pixels at depth ``z_m``, 0 when undefined.

    ``z_m`` is a percentile of the rendered depth buffer, so it is 0 or NaN
    whenever that slice of the frame never received a splat. The notes are
    diagnostics - a missing depth must not sink a render that already
    succeeded, which a bare ``1 / z_m`` did with ZeroDivisionError.
    """
    if not math.isfinite(z_m) or z_m <= 0 or not converge_m:
        return 0.0
    return round(f_px * baseline_m * (1 / z_m - 1 / converge_m), 1)
