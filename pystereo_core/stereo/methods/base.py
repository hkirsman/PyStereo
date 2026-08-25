"""Abstract base class for pluggable stereo synthesis methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend


class BaseStereoMethod(ABC):
    """A stereo warp-and-fill strategy.

    Subclasses implement the core warp + disocclusion-fill logic.
    Shared preprocessing (resize, depth healing, guided filter, gamma)
    is handled by :class:`~pystereo_core.stereo.pipeline.StereoPipeline` before
    calling :meth:`warp_and_fill`.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]

    wants_full_res: ClassVar[bool] = False

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp + fill disocclusion holes, returning ``(left, right)``.

        Parameters
        ----------
        rgb_arr:
            ``(H, W, 3)`` uint8 RGB source image (processing resolution).
        depth_f32:
            ``(H, W)`` float32 depth in ``[0, 1]`` — already healed,
            guided-filtered, and gamma-corrected.
        max_disp:
            Maximum horizontal separation at nearest depth (pixels).
        fg_mask:
            ``(H, W)`` float32 in ``[0, 1]`` from BiRefNet, or ``None``
            if depth healing is disabled.
        settings:
            Current stereo settings.
        inpainter:
            Inpainting backend instance (LaMa, OpenCV, etc.).

        Returns
        -------
        left:
            ``(H, W, 3)`` uint8 RGB — left eye view, holes filled.
        right:
            ``(H, W, 3)`` uint8 RGB — right eye view, holes filled.
        """

    def warp_preview(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Return warped eyes and occlusion masks *before* inpainting.

        Subclasses override to expose pre-inpaint intermediates.
        Returns ``None`` when the method does not support preview.

        Returns
        -------
        left:
            ``(H, W, 3)`` uint8 RGB — left eye, holes are black.
        right:
            ``(H, W, 3)`` uint8 RGB — right eye, holes are black.
        left_mask:
            ``(H, W)`` uint8 — 255 = hole, 0 = valid.
        right_mask:
            ``(H, W)`` uint8 — 255 = hole, 0 = valid.
        """
        return None
