"""Occlusion inpainting backends (commercial-safe weights only)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from pystereo_core.stereo.config import InpaintBackendName

logger = logging.getLogger(__name__)


def _release_torch_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


class InpaintBackend(ABC):
    """Fill occlusion holes in a warped eye view."""

    @abstractmethod
    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        """Return inpainted ``(H, W, 3)`` uint8 RGB.

        ``occlusion_mask``: 255 = hole to fill, 0 = keep original pixel.
        """

    def unload(self) -> None:
        """Release model weights, if any are resident. Reloads lazily on next use."""

    def is_loaded(self) -> bool:
        return False


class NoInpaintBackend(InpaintBackend):
    """Leave holes as black (debug / fast path)."""

    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        out = image_rgb.copy()
        holes = occlusion_mask > 0
        out[holes] = 0
        return out


class OpenCVInpaintBackend(InpaintBackend):
    """Fast Telea inpaint (BSD). Lower quality than LaMa but zero extra deps."""

    def __init__(self, radius: int = 5) -> None:
        self._radius = radius

    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        if not np.any(occlusion_mask):
            return image_rgb.copy()
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        filled = cv2.inpaint(bgr, occlusion_mask, self._radius, cv2.INPAINT_TELEA)
        return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


class LamaInpaintBackend(InpaintBackend):
    """LaMa large-mask inpainting (Apache 2.0 via ``simple-lama-inpainting``)."""

    def __init__(self) -> None:
        self._model: object | None = None

    def _ensure_loaded(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from simple_lama_inpainting import SimpleLama
            from simple_lama_inpainting.models.model import download_model, LAMA_MODEL_URL
        except ImportError as exc:
            raise RuntimeError(
                "simple-lama-inpainting is not installed. "
                "Run: pip install simple-lama-inpainting"
            ) from exc

        import os
        import torch

        from pystereo_core.registry import detect_device

        device = torch.device(detect_device())

        model_path = os.environ.get("LAMA_MODEL") or download_model(LAMA_MODEL_URL)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LaMa model not found: {model_path}")

        logger.info("Loading LaMa inpainting model on %s…", device)
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        model.to(device)

        lama = SimpleLama.__new__(SimpleLama)
        lama.model = model
        lama.device = device
        self._model = lama
        return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        _release_torch_memory()
        logger.info("LaMa inpainting model unloaded")

    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        if not np.any(occlusion_mask):
            return image_rgb.copy()
        lama = self._ensure_loaded()
        orig_h, orig_w = image_rgb.shape[:2]
        image_pil = Image.fromarray(image_rgb)
        mask_pil = Image.fromarray(occlusion_mask)
        result = lama(image_pil, mask_pil)
        out = np.array(result.convert("RGB"))
        out_h, out_w = out.shape[:2]
        if (out_h, out_w) != (orig_h, orig_w):
            if out_h >= orig_h and out_w >= orig_w:
                out = out[:orig_h, :orig_w]
            else:
                logger.warning(
                    "LaMa returned %dx%d for %dx%d input; resizing",
                    out_w, out_h, orig_w, orig_h,
                )
                out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return out


class AotGanInpaintBackend(InpaintBackend):
    """AOT-GAN inpainting (Apache 2.0).

    Uses dilated convolutions (not FFCs) - no global receptive field, no
    zero-initialization dark bias.  Weights must already be present in the
    local HuggingFace cache (downloaded by pystereo_core.download); this
    backend never hits the Hub at runtime.
    """

    _HF_REPO = "NimaBoscarino/aot-gan-places2"

    def __init__(self) -> None:
        self._model: object | None = None
        self._device: object | None = None

    def _ensure_loaded(self) -> tuple[object, object]:
        if self._model is not None:
            return self._model, self._device

        import torch
        from huggingface_hub import hf_hub_download

        from pystereo_core.registry import detect_device
        from pystereo_core.stereo.aotgan_arch import InpaintGenerator

        device = torch.device(detect_device())

        logger.info("Loading AOT-GAN inpainting model on %s…", device)

        # Weights must already be in the HF cache (see download manager). Never hit the Hub here.
        weights_path = hf_hub_download(
            repo_id=self._HF_REPO,
            filename="pytorch_model.bin",
            local_files_only=True,
        )

        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        model = InpaintGenerator()
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        self._model = model
        self._device = device
        return model, device

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        self._device = None
        _release_torch_memory()
        logger.info("AOT-GAN inpainting model unloaded")

    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        if not np.any(occlusion_mask):
            return image_rgb.copy()

        import torch

        model, device = self._ensure_loaded()
        h, w = image_rgb.shape[:2]

        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4

        img_f = image_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0
        mask_f = (occlusion_mask > 0).astype(np.float32)

        if pad_h > 0 or pad_w > 0:
            img_f = np.pad(
                img_f, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect",
            )
            mask_f = np.pad(mask_f, ((0, pad_h), (0, pad_w)), mode="constant")

        img_t = (
            torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        mask_t = (
            torch.from_numpy(mask_f).unsqueeze(0).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            out_t = model(img_t, mask_t)

        out_np = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_np = np.clip((out_np + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)

        if pad_h > 0 or pad_w > 0:
            out_np = out_np[:h, :w]

        result = image_rgb.copy()
        holes = occlusion_mask > 0
        result[holes] = out_np[holes]

        return result


class FluxInpaintBackend(InpaintBackend):
    """Experimental FLUX Fill inpainting - **not Apache 2.0** (BFL non-commercial license)."""

    def inpaint(self, image_rgb: np.ndarray, occlusion_mask: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "FLUX inpainting is disabled for commercial-safe builds. "
            "Use --inpaint lama (Apache 2.0) or --inpaint opencv (BSD)."
        )


def create_inpaint_backend(name: InpaintBackendName) -> InpaintBackend:
    """Factory for inpainting backends."""
    if name == "none":
        return NoInpaintBackend()
    if name == "opencv":
        return OpenCVInpaintBackend()
    if name == "lama":
        return LamaInpaintBackend()
    if name == "aotgan":
        return AotGanInpaintBackend()
    if name == "flux":
        return FluxInpaintBackend()
    raise ValueError(f"Unknown inpaint backend: {name!r}")
