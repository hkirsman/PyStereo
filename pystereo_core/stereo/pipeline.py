"""End-to-end 2D → SBS stereo pipeline with pluggable warp-and-fill methods.

Shared preprocessing:
  1. Parse + resize to processing resolution.
  2. Normalise depth to [0, 1].
  3. Semantic mask (BiRefNet) + depth healing.
  4. Guided-filter refinement + depth gamma.
  5. Compute adaptive max disparity.

Then delegates to the selected :class:`BaseStereoMethod` for warp + fill,
and composes the final SBS image.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import replace as dc_replace
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from pystereo_core.stereo.config import StereoMethodName, StereoSettings
from pystereo_core.stereo.depth import apply_depth_gamma, guided_filter_depth
from pystereo_core.stereo.heal import heal_depth_with_mask
from pystereo_core.stereo.inpaint import InpaintBackend, create_inpaint_backend
from pystereo_core.stereo.methods import available_methods, get_method
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.segment import ForegroundSegmenter
from pystereo_core.stereo.timing import record_step
from pystereo_core.stereo.warp import adaptive_max_disparity

logger = logging.getLogger(__name__)


def _compose_sbs(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1)


class _PreparedInput(NamedTuple):
    """Preprocessed arrays ready for a stereo method."""
    rgb_arr: np.ndarray
    depth_f32: np.ndarray
    max_disp: float
    fg_mask: np.ndarray | None
    method_name: str
    stereo_method: BaseStereoMethod


class WarpPreviewResult(NamedTuple):
    """Pre-inpaint warp preview images."""
    warp_sbs: Image.Image
    mask_sbs: Image.Image


class _SharedModels:
    """Lazily loaded model instances shared by a pipeline and its derivatives.

    Kept in one holder (instead of plain attributes) so :meth:`StereoPipeline.derive`
    can hand out a pipeline with different settings that still reuses the
    resident BiRefNet and inpainter - loading either costs several seconds
    and was happening on every web request.
    """

    def __init__(self, inpainter: InpaintBackend) -> None:
        self.inpainter = inpainter
        self.segmenter: ForegroundSegmenter | None = None
        self.methods: dict[str, BaseStereoMethod] = {}


class StereoPipeline:
    """Production 2D → SBS stereo converter.

    Shared preprocessing → method-specific warp+fill → SBS composition.

    The active method is determined by ``settings.stereo_method`` and can
    be overridden per-call via the ``method`` parameter on :meth:`synthesize`.
    Per-request setting tweaks should go through :meth:`derive` rather than
    a fresh pipeline, so loaded models are reused.
    """

    def __init__(
        self,
        settings: StereoSettings | None = None,
        inpainter: InpaintBackend | None = None,
    ) -> None:
        self.settings = settings or StereoSettings.from_env()
        self._shared = _SharedModels(
            inpainter or create_inpaint_backend(self.settings.inpaint_backend),
        )

    @property
    def _inpainter(self) -> InpaintBackend:
        return self._shared.inpainter

    @property
    def _segmenter(self) -> ForegroundSegmenter | None:
        return self._shared.segmenter

    @_segmenter.setter
    def _segmenter(self, value: ForegroundSegmenter | None) -> None:
        self._shared.segmenter = value

    @property
    def _methods(self) -> dict[str, BaseStereoMethod]:
        return self._shared.methods

    def derive(self, **overrides: Any) -> StereoPipeline:
        """Return a pipeline with ``settings`` fields replaced by *overrides*.

        The result shares this pipeline's loaded models (segmenter,
        inpainter, method instances), so it is cheap to create per request.
        ``inpaint_backend`` cannot be overridden here - the inpainter is
        shared, not rebuilt.
        """
        if "inpaint_backend" in overrides:
            raise ValueError("derive() cannot change inpaint_backend; build a new StereoPipeline")
        clone = copy.copy(self)
        clone.settings = dc_replace(self.settings, **overrides)
        return clone

    def with_settings(self, settings: StereoSettings) -> StereoPipeline:
        """Like :meth:`derive`, but takes a whole ``StereoSettings``.

        For callers that already hold a settings object instead of a set of
        overrides - a stereo method running a nested pass, say. Falls back
        to a fresh pipeline when *settings* asks for a different inpaint
        backend, since the inpainter is shared rather than rebuilt.
        """
        if settings.inpaint_backend != self.settings.inpaint_backend:
            return StereoPipeline(settings=settings)
        clone = copy.copy(self)
        clone.settings = settings
        return clone

    def _ensure_segmenter(self) -> ForegroundSegmenter:
        if self._segmenter is None:
            self._segmenter = ForegroundSegmenter()
        return self._segmenter

    def segmenter_loaded(self) -> bool:
        """True while BiRefNet weights are resident (for UI load notes)."""
        return getattr(self._segmenter, "_model", None) is not None

    @staticmethod
    def _load_note(loaded_before: bool, loaded_after: bool) -> str:
        return " (model load)" if loaded_after and not loaded_before else ""

    def unload_models(self) -> list[str]:
        """Drop the segmenter and inpainter weights held by this pipeline.

        Both reload lazily on the next use. Returns the names of what was
        actually resident, for UI feedback.
        """
        released: list[str] = []
        if self._segmenter is not None:
            was_loaded = self.segmenter_loaded()
            self._segmenter.unload()
            self._segmenter = None
            if was_loaded:
                released.append("segmenter")
        if self._inpainter.is_loaded():
            self._inpainter.unload()
            released.append("inpainter")
        return released

    def _get_method(self, name: str) -> BaseStereoMethod:
        if name not in self._methods:
            method = get_method(name)
            # So a method needing a nested pass can reuse our loaded models.
            method._owner = self
            self._methods[name] = method
        return self._methods[name]

    def _preprocess(
        self,
        image: Image.Image,
        depth_map: Image.Image | np.ndarray,
        *,
        divergence_ratio: float | None = None,
        method: StereoMethodName | None = None,
    ) -> _PreparedInput:
        """Shared preprocessing: resize, depth healing, filter, gamma, disparity."""
        method_name = method or self.settings.stereo_method
        stereo_method = self._get_method(method_name)

        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        if isinstance(depth_map, np.ndarray):
            depth_f32 = depth_map.astype(np.float32)
        else:
            depth_f32 = np.array(depth_map.convert("L"), dtype=np.float32) / 255.0

        if stereo_method.wants_full_res:
            logger.info(
                "Stereo [%s]: full-res bypass (%dx%d), method handles its own scaling",
                method_name, orig_w, orig_h,
            )
        elif max(orig_w, orig_h) > self.settings.max_processing_dim:
            scale = self.settings.max_processing_dim / float(max(orig_w, orig_h))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            rgb = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
            depth_f32 = cv2.resize(
                depth_f32, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
            logger.info(
                "Stereo processing scale %.3f (%dx%d → %dx%d)",
                scale, orig_w, orig_h, new_w, new_h,
            )

        rgb_arr = np.array(rgb)

        lo, hi = float(depth_f32.min()), float(depth_f32.max())
        if hi > lo:
            depth_f32 = (depth_f32 - lo) / (hi - lo)

        fg_mask: np.ndarray | None = None
        if self.settings.depth_healing:
            try:
                segmenter = self._ensure_segmenter()
                fg_mask = segmenter.segment(
                    rgb, padding=self.settings.segmenter_padding
                )
                depth_f32 = heal_depth_with_mask(
                    depth_f32,
                    fg_mask,
                    bg_threshold_ratio=self.settings.depth_healing_bg_threshold,
                    edge_blur_sigma=self.settings.depth_healing_edge_blur_sigma,
                    mask_dilate_px=self.settings.depth_healing_mask_dilate_px,
                )
            except Exception as exc:
                logger.warning(
                    "Depth healing unavailable (%s); continuing without it",
                    exc,
                )
                fg_mask = None

        depth_f32 = guided_filter_depth(
            depth_f32,
            rgb_arr,
            radius=self.settings.guided_filter_radius,
            eps=self.settings.guided_filter_eps,
        )
        depth_f32 = apply_depth_gamma(depth_f32, self.settings.depth_gamma)

        ratio = (
            divergence_ratio
            if divergence_ratio is not None
            else self.settings.divergence_ratio
        )
        max_disp = adaptive_max_disparity(
            depth_f32,
            rgb_arr.shape[1],
            base_ratio=ratio,
            min_ratio=self.settings.min_divergence_ratio,
            max_ratio=self.settings.max_divergence_ratio,
            adaptive=self.settings.adaptive_depth,
        )
        logger.info(
            "Stereo [%s]: %.1f px disp (%.2f%% of width), inpaint=%s, heal=%s",
            method_name,
            max_disp,
            100.0 * max_disp / rgb_arr.shape[1],
            self.settings.inpaint_backend,
            self.settings.depth_healing,
        )

        return _PreparedInput(rgb_arr, depth_f32, max_disp, fg_mask,
                              method_name, stereo_method)

    @staticmethod
    def _release_gpu_cache() -> None:
        try:
            import torch
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _synthesize_no_depth(
        self,
        image: Image.Image,
        stereo_method: BaseStereoMethod,
        method_name: str,
        intermediates: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Bypass depth preprocessing for methods that predict their own 3D."""
        rgb = image.convert("RGB")
        fg_mask: np.ndarray | None = None
        t0 = time.perf_counter()
        seg_loaded = self.segmenter_loaded()
        try:
            segmenter = self._ensure_segmenter()
            fg_mask = segmenter.segment(
                rgb, padding=self.settings.segmenter_padding
            )
        except Exception as exc:
            logger.warning(
                "Segmenter unavailable (%s); SHARP convergence will use "
                "depth percentile fallback",
                exc,
            )
        record_step(
            intermediates,
            "Segmentation" + self._load_note(seg_loaded, self.segmenter_loaded()),
            time.perf_counter() - t0,
        )

        logger.info("Stereo [%s]: needs_depth=False, skipping depth pipeline", method_name)
        left, right = stereo_method.synthesize(rgb, fg_mask, self.settings, intermediates)
        self._release_gpu_cache()
        return Image.fromarray(_compose_sbs(left, right))

    def synthesize(
        self,
        image: Image.Image,
        depth_map: Image.Image | np.ndarray,
        *,
        divergence_ratio: float | None = None,
        method: StereoMethodName | None = None,
        intermediates: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Build a horizontal SBS stereo image from a 2D photo + depth map.

        Parameters
        ----------
        image:
            Source RGB photo.
        depth_map:
            Grayscale depth - PIL ``L`` (uint8, 255 = closest) or
            float32 ndarray ``(H, W)`` in [0, 1] (1.0 = closest).
        divergence_ratio:
            Optional override for max separation as fraction of width.
        method:
            Override the stereo method for this call (ignores settings).
        intermediates:
            If not ``None``, populated with intermediate artifacts.
        """
        method_name = method or self.settings.stereo_method
        stereo_method = self._get_method(method_name)

        if not stereo_method.needs_depth:
            return self._synthesize_no_depth(image, stereo_method, method_name, intermediates)

        t0 = time.perf_counter()
        seg_loaded = self.segmenter_loaded()
        p = self._preprocess(
            image, depth_map,
            divergence_ratio=divergence_ratio, method=method,
        )
        record_step(
            intermediates,
            "Preprocess (heal + filter)" + self._load_note(seg_loaded, self.segmenter_loaded()),
            time.perf_counter() - t0,
        )

        t0 = time.perf_counter()
        inpaint_loaded = self._inpainter.is_loaded()
        left, right = p.stereo_method.warp_and_fill(
            p.rgb_arr, p.depth_f32, p.max_disp, p.fg_mask,
            self.settings, self._inpainter,
        )
        record_step(
            intermediates,
            "Warp + inpaint" + self._load_note(inpaint_loaded, self._inpainter.is_loaded()),
            time.perf_counter() - t0,
        )

        self._release_gpu_cache()

        sbs_arr = _compose_sbs(left, right)
        return Image.fromarray(sbs_arr)

    def warp_preview(
        self,
        image: Image.Image,
        depth_map: Image.Image | np.ndarray,
        *,
        method: StereoMethodName | None = None,
    ) -> WarpPreviewResult | None:
        """Return pre-inpaint warp SBS + mask SBS, or None if unsupported."""
        p = self._preprocess(image, depth_map, method=method)

        result = p.stereo_method.warp_preview(
            p.rgb_arr, p.depth_f32, p.max_disp, p.fg_mask, self.settings,
        )
        if result is None:
            return None

        left, right, left_mask, right_mask = result
        warp_sbs = Image.fromarray(_compose_sbs(left, right))
        mask_sbs = Image.fromarray(_compose_sbs(left_mask, right_mask))
        return WarpPreviewResult(warp_sbs, mask_sbs)

    def synthesize_with_depth_estimator(
        self,
        image: Image.Image,
        depth_estimator: Any,
        *,
        method: StereoMethodName | None = None,
        intermediates: dict[str, Any] | None = None,
    ) -> Image.Image:
        """Run depth estimation then synthesize SBS."""
        method_name = method or self.settings.stereo_method
        stereo_method = self._get_method(method_name)

        if not stereo_method.needs_depth:
            return self._synthesize_no_depth(image, stereo_method, method_name, intermediates)

        if hasattr(depth_estimator, "process_raw"):
            depth_f32 = depth_estimator.process_raw(image.convert("RGB"))
            return self.synthesize(image, depth_f32, method=method, intermediates=intermediates)
        depth = depth_estimator.process(image.convert("RGB"))
        return self.synthesize(image, depth.convert("L"), method=method, intermediates=intermediates)


def synthesize_sbs(
    original: Image.Image,
    depth_map: Image.Image,
    *,
    settings: StereoSettings | None = None,
    divergence_ratio: float | None = None,
    method: StereoMethodName | None = None,
) -> Image.Image:
    """Functional API used by ``api.py`` and legacy imports."""
    pipeline = StereoPipeline(settings=settings)
    return pipeline.synthesize(
        original, depth_map, divergence_ratio=divergence_ratio, method=method,
    )
