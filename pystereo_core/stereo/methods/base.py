"""Abstract base class for pluggable stereo synthesis methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from PIL import Image

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend

if TYPE_CHECKING:
    from pystereo_core.stereo.pipeline import StereoPipeline


class BaseStereoMethod(ABC):
    """A stereo warp-and-fill strategy.

    Subclasses implement the core warp + disocclusion-fill logic.
    Shared preprocessing (resize, depth healing, guided filter, gamma)
    is handled by :class:`~pystereo_core.stereo.pipeline.StereoPipeline` before
    calling :meth:`warp_and_fill`.

    Methods that set ``needs_depth = False`` bypass all depth-based
    preprocessing and use :meth:`synthesize` instead of :meth:`warp_and_fill`.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    description: ClassVar[str]
    #: Short user-facing tip shown under the method picker in the web UI.
    ui_info: ClassVar[str] = ""
    deprecated: ClassVar[bool] = False

    wants_full_res: ClassVar[bool] = False
    needs_depth: ClassVar[bool] = True
    #: True when the method can use Taichi for the splat render step.
    uses_taichi: ClassVar[bool] = False

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {}

    #: Pipeline that built this instance, set by ``StereoPipeline._get_method``.
    #: ``None`` when a method is used standalone (experiments, tests).
    _owner: StereoPipeline | None = None

    def nested_pipeline(self, settings: StereoSettings) -> StereoPipeline:
        """Pipeline for a nested synthesis pass, reusing loaded models.

        A method that needs a full depth-based pass (SHARP Depth, for one)
        must go through this rather than construct a fresh
        :class:`~pystereo_core.stereo.pipeline.StereoPipeline` - a new
        pipeline gets its own BiRefNet and inpainter, reloading both on
        every photo.

        *settings* wins over the owner's own settings, so a method reached
        through :meth:`StereoPipeline.derive` still runs at the derived
        settings even though ``_owner`` points at the original pipeline.
        """
        from pystereo_core.stereo.pipeline import StereoPipeline

        if self._owner is None:
            return StereoPipeline(settings=settings)
        return self._owner.with_settings(settings)

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
            ``(H, W)`` float32 depth in ``[0, 1]`` - already healed,
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
            ``(H, W, 3)`` uint8 RGB - left eye view, holes filled.
        right:
            ``(H, W, 3)`` uint8 RGB - right eye view, holes filled.
        """

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Produce stereo eyes without external depth estimation.

        Called instead of :meth:`warp_and_fill` when ``needs_depth`` is
        ``False``.  The method handles its own 3D prediction and rendering.

        Parameters
        ----------
        image:
            Source RGB photo (original resolution, PIL).
        fg_mask:
            ``(H, W)`` float32 in ``[0, 1]`` from BiRefNet, or ``None``.
        settings:
            Current stereo settings.
        intermediates:
            If not ``None``, the method may populate this dict with
            intermediate artifacts (e.g. ``splat_rgb``, ``depth01``).

        Returns
        -------
        left:
            ``(H, W, 3)`` uint8 RGB.
        right:
            ``(H, W, 3)`` uint8 RGB.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets needs_depth=False but does not "
            "implement synthesize()"
        )

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
            ``(H, W, 3)`` uint8 RGB - left eye, holes are black.
        right:
            ``(H, W, 3)`` uint8 RGB - right eye, holes are black.
        left_mask:
            ``(H, W)`` uint8 - 255 = hole, 0 = valid.
        right_mask:
            ``(H, W)`` uint8 - 255 = hole, 0 = valid.
        """
        return None
