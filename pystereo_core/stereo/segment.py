"""Semantic foreground segmentation using BiRefNet (Apache 2.0).

Generates a high-resolution float32 alpha mask that captures fine details
like hair, fur, and translucent edges - areas where monocular depth models
(e.g. Depth Anything V2) often produce hard 90-degree cutoffs.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

BIREFNET_MODEL_ID = "ZhengPeng7/BiRefNet"


class ForegroundSegmenter:
    """Lazy-loaded BiRefNet segmenter producing float32 foreground masks."""

    def __init__(self) -> None:
        self._model: Any = None
        self._device: torch.device = torch.device("cpu")
        self._transform: transforms.Compose | None = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Load BiRefNet once, even if several threads ask at the same time."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForImageSegmentation

        from pystereo_core.registry import detect_device

        self._device = torch.device(detect_device())

        logger.info("Loading BiRefNet on %s…", self._device)
        # Weights must already be in the HF cache (see download manager). Never hit the Hub here.
        self._model = AutoModelForImageSegmentation.from_pretrained(
            BIREFNET_MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        self._model.to(self._device)
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def segment(self, image: Image.Image, *, padding: int = 100) -> np.ndarray:
        """Return a float32 foreground mask ``(H, W)`` in ``[0.0, 1.0]``.

        Adds *padding* px around the image before inference and crops back
        to prevent BiRefNet from cutting off subjects touching the border.

        Parameters
        ----------
        image:
            Source RGB photo.
        padding:
            Pixels of black border added on every side before inference.

        Returns
        -------
        mask:
            ``(H, W)`` float32 array. 1.0 = foreground, 0.0 = background.
        """
        self._ensure_loaded()
        assert self._transform is not None

        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        padded_w, padded_h = orig_w + 2 * padding, orig_h + 2 * padding
        padded = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
        padded.paste(rgb, (padding, padding))

        input_tensor: torch.Tensor = self._transform(padded).unsqueeze(0).to(self._device)

        with torch.no_grad():
            preds = self._model(input_tensor)[-1]

        mask_t = torch.sigmoid(preds).squeeze()
        if mask_t.ndim == 3:
            mask_t = mask_t[0]

        mask_t = F.interpolate(
            mask_t.unsqueeze(0).unsqueeze(0),
            size=(padded_h, padded_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        mask_np: np.ndarray = mask_t.cpu().numpy().astype(np.float32)
        mask_np = mask_np[padding : padding + orig_h, padding : padding + orig_w]

        fg_pct = 100.0 * float(np.mean(mask_np > 0.5))
        logger.info("BiRefNet segmentation: %.1f%% foreground", fg_pct)

        return np.clip(mask_np, 0.0, 1.0)

    def unload(self) -> None:
        """Release model weights and GPU memory."""
        self._model = None
        self._transform = None
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self._device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        gc.collect()
