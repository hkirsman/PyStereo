"""PyStereo - standalone AI stereo synthesis (2D to SBS)."""

from pystereo_core.hf_env import configure_hf_env

# Must run before anything imports huggingface_hub, which freezes its
# environment into module constants at import time.
configure_hf_env()

from pystereo_core.download import apply_ai_stereo_enabled, get_download_manager  # noqa: E402
from pystereo_core.registry import ModelRegistry, get_registry  # noqa: E402

__all__ = [
    "ModelRegistry",
    "get_registry",
    "get_download_manager",
    "apply_ai_stereo_enabled",
]
