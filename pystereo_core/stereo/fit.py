"""Fit a rendered SBS image to a caller's output pixel budget.

Synthesis deliberately runs at the source photo's resolution: SHARP's Gaussian
canvas is the input size (see :func:`sharp_predict.predict_gaussians`), and
rendering above the delivered size supersamples splat footprints and keeps the
original photo detail intact through the warp for detail-transfer methods.

So the downscale belongs at the very end, right before encoding - not on the
input. Doing it here also means a caller with a 14 MP budget never has to
decode a 50 MP JPEG just to throw most of it away.
"""

from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

MAX_TEXTURE_EDGE = 16384
"""GPU max texture edge. A safety rail for absurd panoramas, not the control."""


def fit_to_pixel_budget(
    img: Image.Image,
    max_pixels: int,
    *,
    max_edge: int = MAX_TEXTURE_EDGE,
    even_width: bool = False,
) -> Image.Image:
    """Downscale *img* to fit *max_pixels* of total area. Never upscales.

    Total area is the primary control, so one budget means the same thing for
    any aspect ratio; *max_edge* only rejects impossible one-axis sizes.

    Set *even_width* for SBS output, where an odd width would split the two
    eyes into unequal halves.
    """
    if max_pixels <= 0:
        return img

    w, h = img.size
    if w > max_edge or h > max_edge:
        img = img.copy()
        img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        w, h = img.size

    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w = max(1, int(w * scale))
        h = max(1, int(h * scale))

    if even_width and w % 2 and w > 1:
        w -= 1

    if (w, h) != img.size:
        img = img.resize((w, h), Image.Resampling.LANCZOS)
    return img
