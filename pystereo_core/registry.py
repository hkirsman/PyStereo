"""Generic AI model registry.

Models register themselves by capability (e.g. ``"upscale"``, ``"depth"``).
The registry lazily loads models on first use and caches them in memory.
GPU capability is auto-detected at startup; callers can query readiness
before invoking a model.

Hub weight downloads are owned by :mod:`pystereo_core.download` — ``get()`` only loads
from the local cache and returns ``None`` while downloads are in progress.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class every AI model must implement
# ---------------------------------------------------------------------------

class AIModel(ABC):
    """Base class for pluggable AI models."""

    name: str = "unnamed"
    capability: str = "unknown"  # e.g. "upscale", "depth"
    model_enabled: bool = True  # per-model toggle

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model onto *device* from the local weight cache."""

    @abstractmethod
    def unload(self) -> None:
        """Free GPU/CPU memory."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return ``True`` if the model is ready to run inference."""

    @abstractmethod
    def process(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        """Run inference on a PIL Image and return the result."""


# ---------------------------------------------------------------------------
# GPU / device helpers
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Return the best available PyTorch device string.

    Checks in order: MPS (Apple Silicon) → CUDA → CPU.
    Returns ``"none"`` if PyTorch is not installed at all.
    """
    try:
        import torch  # noqa: F811
    except ImportError:
        return "none"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Registry singleton
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe registry of AI models keyed by capability."""

    def __init__(self) -> None:
        self._models: Dict[str, AIModel] = {}
        self._enabled: bool = True
        self._device: str = "none"
        self._lock = threading.Lock()
        self._detected = False

    # -- discovery ----------------------------------------------------------

    def detect_gpu(self) -> str:
        """Auto-detect GPU and cache the result. Returns device string."""
        if not self._detected:
            self._device = detect_device()
            self._detected = True
            logger.info("AI device detected: %s", self._device)
        return self._device

    @property
    def device(self) -> str:
        return self._device

    @property
    def available(self) -> bool:
        """True if PyTorch is installed and at least CPU is usable."""
        self.detect_gpu()
        return self._device != "none"

    @property
    def enabled(self) -> bool:
        return self._enabled and self.available

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if not value:
            self._unload_all()

    # -- model management ---------------------------------------------------

    def register(self, model: AIModel) -> None:
        """Register a model for its declared capability."""
        self._models[model.capability] = model
        logger.info("Registered AI model: %s (capability=%s)", model.name, model.capability)

    def has_capability(self, capability: str) -> bool:
        """Return True if a model is registered for *capability*."""
        return capability in self._models

    def get(self, capability: str) -> Optional[AIModel]:
        """Return the model for *capability*, loading from local cache if needed.

        Returns ``None`` when AI is disabled, the capability is off, weights
        are still downloading, or local load fails. Never hits the Hub.
        """
        with self._lock:
            model = self._models.get(capability)
            if model is None:
                return None
            if not self.enabled:
                return None
            if not model.model_enabled:
                return None
            if model.is_loaded():
                return model

        # Depth / stereo pack must be fully cached before first load.
        if capability == "depth":
            from pystereo_core.download import get_download_manager

            mgr = get_download_manager()
            if mgr.is_downloading() or not mgr.is_pack_ready():
                return None

        with self._lock:
            model = self._models.get(capability)
            if model is None or not self.enabled or not model.model_enabled:
                return None
            if model.is_loaded():
                return model
            self.detect_gpu()
            logger.info("Loading model %s on %s …", model.name, self._device)
            try:
                model.load(self._device)
            except Exception:
                logger.exception("Failed to load model %s from local cache", model.name)
                return None
            logger.info("Model %s ready.", model.name)
            return model

    def _unload_all(self) -> None:
        with self._lock:
            for model in self._models.values():
                if model.is_loaded():
                    logger.info("Unloading model %s", model.name)
                    model.unload()

    def unload_all(self) -> None:
        """Free all loaded model weights from memory."""
        self._unload_all()

    def toggle_capability(self, capability: str, enable: bool) -> bool:
        """Enable or disable a specific model. Returns True if found."""
        with self._lock:
            model = self._models.get(capability)
            if model is None:
                return False
            model.model_enabled = enable
            if not enable and model.is_loaded():
                logger.info("Unloading model %s (disabled)", model.name)
                model.unload()
            return True

    def status(self) -> Dict[str, Any]:
        """Return a JSON-friendly status dict for the /api/ai/status endpoint."""
        from pystereo_core.download import get_download_manager

        self.detect_gpu()
        dl = get_download_manager().status_dict()
        dl_models = dl.get("models", {})

        models_info: Dict[str, Any] = {}
        for cap, model in self._models.items():
            entry: Dict[str, Any] = {
                "name": model.name,
                "loaded": model.is_loaded(),
                "enabled": model.model_enabled,
            }
            # Merge download artifact fields for registered capabilities.
            if cap in dl_models:
                art = dl_models[cap]
                entry.update(
                    {
                        "repo_id": art.get("repo_id"),
                        "local": art.get("local", False),
                        "download_state": art.get("download_state", "absent"),
                        "percent": art.get("percent", 0),
                        "bytes_downloaded": art.get("bytes_downloaded", 0),
                        "bytes_total": art.get("bytes_total", 0),
                        "size_exact": art.get("size_exact", False),
                    }
                )
                if art.get("cache_path"):
                    entry["cache_path"] = art["cache_path"]
            models_info[cap] = entry

        # Include non-registry pack artifacts (segment, inpaint) for the UI.
        for art_id, art in dl_models.items():
            if art_id not in models_info:
                models_info[art_id] = {
                    "name": art.get("name", art_id),
                    "loaded": False,
                    "enabled": self._enabled,
                    "repo_id": art.get("repo_id"),
                    "url": art.get("url"),
                    "local": art.get("local", False),
                    "download_state": art.get("download_state", "absent"),
                    "percent": art.get("percent", 0),
                    "bytes_downloaded": art.get("bytes_downloaded", 0),
                    "bytes_total": art.get("bytes_total", 0),
                    "size_exact": art.get("size_exact", False),
                }
                if art.get("cache_path"):
                    models_info[art_id]["cache_path"] = art["cache_path"]

        return {
            "available": self.available,
            "enabled": self.enabled,
            "device": self._device,
            "download": {
                "state": dl.get("state", "idle"),
                "percent": dl.get("percent", 0),
                "bytes_downloaded": dl.get("bytes_downloaded", 0),
                "bytes_total": dl.get("bytes_total", 0),
                "message": dl.get("message", ""),
                "error": dl.get("error"),
                "deletable_paths": dl.get("deletable_paths", []),
            },
            "models": models_info,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: Optional[ModelRegistry] = None

def get_registry() -> ModelRegistry:
    """Return (or create) the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
