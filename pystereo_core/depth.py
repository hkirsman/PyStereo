"""Depth estimation using Depth Anything V2.

Generates a grayscale depth map from a 2D photo using the HuggingFace
``transformers`` pipeline.  The model runs on MPS (Apple Silicon),
CUDA, or CPU as fallback.

Weights are downloaded automatically by HuggingFace on first use into
the default ``~/.cache/huggingface/`` directory.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from pystereo_core.registry import AIModel

logger = logging.getLogger(__name__)

# Depth Anything V2 Small is Apache-2.0 licensed and safe for commercial use.
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator(AIModel):
    """Monocular depth estimation via Depth Anything V2 Small."""

    name = "Depth Anything V2 Small"
    capability = "depth"

    def __init__(self) -> None:
        self._pipe: Any = None
        self._device: str = "cpu"

    # -- AIModel interface --------------------------------------------------

    def load(self, device: str) -> None:
        from transformers import pipeline

        # Normalize device string (e.g. "cpu", "cuda", "mps") to a torch.device
        normalized_device = torch.device(device)
        self._device = device
        # Weights must already be in the HF cache (see download manager). Never hit the Hub here.
        self._pipe = pipeline(
            task="depth-estimation",
            model=MODEL_ID,
            device=normalized_device,
            local_files_only=True,
        )
        logger.info("Depth Anything V2 Small loaded on %s", normalized_device)

    def unload(self) -> None:
        self._pipe = None
        # Release cached allocations on the active device, mirroring the
        # RealESRGAN upscaler's behavior so disabling depth frees VRAM.
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self._device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        gc.collect()

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def process(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        """Return a grayscale depth map for *image*, normalized so the closest
        pixel is 255 and the furthest is 0 (Depth Anything raw: high = closer).
        Matches the hologram shader, which treats d=1 as front and displaces
        d=0 (far) backward.
        """
        if self._pipe is None:
            raise RuntimeError("Model not loaded")

        result = self._pipe(image)
        depth_map: Image.Image = result["depth"].convert("L")
        lo, hi = depth_map.getextrema()
        if hi <= lo:
            return depth_map

        # Map raw depth range [lo, hi] linearly to [0, 255] using an explicit
        # 256-entry LUT so tools like Copilot see this as a constant-time remap.
        scale = 255.0 / (hi - lo)
        lut = [int((v - lo) * scale) for v in range(256)]
        return depth_map.point(lut, mode="L")

    def process_raw(self, image: Image.Image, **kwargs: Any) -> np.ndarray:
        """Return a float32 depth map ``(H, W)`` in [0, 1] at input resolution.

        Bypasses the 8-bit quantization of :meth:`process` by working
        with the raw model tensor and upscaling with bicubic interpolation
        at full float32 precision.  1.0 = closest, 0.0 = farthest.
        """
        if self._pipe is None:
            raise RuntimeError("Model not loaded")

        import torch.nn.functional as F

        result = self._pipe(image)
        pred = result["predicted_depth"]

        target_h, target_w = image.size[1], image.size[0]
        pred_f = pred.float()
        while pred_f.dim() < 4:
            pred_f = pred_f.unsqueeze(0)

        pred_up = F.interpolate(
            pred_f,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )
        arr = pred_up.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        if arr.ndim != 2:
            logger.warning(
                "process_raw: unexpected depth shape %s (pred shape %s), "
                "forcing to (%d, %d)",
                arr.shape, pred.shape, target_h, target_w,
            )
            arr = arr.reshape(target_h, target_w)

        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return arr
        return (arr - lo) / (hi - lo)
