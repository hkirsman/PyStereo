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

import logging
from typing import Any, NamedTuple

import cv2
import numpy as np
from PIL import Image

from pystereo_core.stereo.config import (
    InpaintBackendName,
    StereoMethodName,
    StereoSettings,
)
from pystereo_core.stereo.depth import apply_depth_gamma, guided_filter_depth
from pystereo_core.stereo.heal import heal_depth_with_mask
from pystereo_core.stereo.inpaint import InpaintBackend, create_inpaint_backend
from pystereo_core.stereo.methods import available_methods, get_method
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.segment import ForegroundSegmenter
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
    method_name: StereoMethodName
    stereo_method: BaseStereoMethod
    settings: StereoSettings


class WarpPreviewResult(NamedTuple):
    """Pre-inpaint warp preview images."""
    warp_sbs: Image.Image
    mask_sbs: Image.Image


class StereoPipeline:
    """Production 2D → SBS stereo converter.

    Shared preprocessing → method-specific warp+fill → SBS composition.

    The active method is determined by ``settings.stereo_method`` and can
    be overridden per-call via the ``method`` parameter on :meth:`synthesize`.
    """

    def __init__(
        self,
        settings: StereoSettings | None = None,
        inpainter: InpaintBackend | None = None,
    ) -> None:
        self.settings = settings or StereoSettings.from_env()
        self._explicit_inpainter = inpainter
        self._inpainters: dict[str, InpaintBackend] = {}
        if inpainter is None:
            backend = self.settings.inpaint_backend
            self._inpainters[backend] = create_inpaint_backend(backend)
        self._segmenter: ForegroundSegmenter | None = None
        self._methods: dict[str, BaseStereoMethod] = {}

    def _ensure_segmenter(self) -> ForegroundSegmenter:
        if self._segmenter is None:
            self._segmenter = ForegroundSegmenter()
        return self._segmenter

    def _get_method(self, name: str) -> BaseStereoMethod:
        if name not in self._methods:
            self._methods[name] = get_method(name)
        return self._methods[name]

    def _get_inpainter(self, backend: InpaintBackendName) -> InpaintBackend:
        """Return the backend for *backend*, loading it on first use.

        A per-call method override can select a different backend (e.g.
        AOT-GAN), so the one built at construction is not always the right
        one.  An explicitly injected inpainter always wins.
        """
        if self._explicit_inpainter is not None:
            return self._explicit_inpainter
        if backend not in self._inpainters:
            logger.info("Loading %s inpaint backend on demand", backend)
            self._inpainters[backend] = create_inpaint_backend(backend)
        return self._inpainters[backend]

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
        settings = self.settings.resolved_for(method_name)

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
        elif max(orig_w, orig_h) > settings.max_processing_dim:
            scale = settings.max_processing_dim / float(max(orig_w, orig_h))
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
        if settings.depth_healing:
            try:
                segmenter = self._ensure_segmenter()
                fg_mask = segmenter.segment(
                    rgb, padding=settings.segmenter_padding
                )
                depth_f32 = heal_depth_with_mask(
                    depth_f32,
                    fg_mask,
                    bg_threshold_ratio=settings.depth_healing_bg_threshold,
                    edge_blur_sigma=settings.depth_healing_edge_blur_sigma,
                    mask_dilate_px=settings.depth_healing_mask_dilate_px,
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
            radius=settings.guided_filter_radius,
            eps=settings.guided_filter_eps,
        )
        depth_f32 = apply_depth_gamma(depth_f32, settings.depth_gamma)

        ratio = (
            divergence_ratio
            if divergence_ratio is not None
            else settings.divergence_ratio
        )
        max_disp = adaptive_max_disparity(
            depth_f32,
            rgb_arr.shape[1],
            base_ratio=ratio,
            min_ratio=settings.min_divergence_ratio,
            max_ratio=settings.max_divergence_ratio,
            adaptive=settings.adaptive_depth,
        )
        logger.info(
            "Stereo [%s]: %.1f px disp (%.2f%% of width), inpaint=%s, heal=%s",
            method_name,
            max_disp,
            100.0 * max_disp / rgb_arr.shape[1],
            settings.inpaint_backend,
            settings.depth_healing,
        )

        return _PreparedInput(rgb_arr, depth_f32, max_disp, fg_mask,
                              method_name, stereo_method, settings)

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

    def synthesize(
        self,
        image: Image.Image,
        depth_map: Image.Image | np.ndarray,
        *,
        divergence_ratio: float | None = None,
        method: StereoMethodName | None = None,
    ) -> Image.Image:
        """Build a horizontal SBS stereo image from a 2D photo + depth map.

        Parameters
        ----------
        image:
            Source RGB photo.
        depth_map:
            Grayscale depth — PIL ``L`` (uint8, 255 = closest) or
            float32 ndarray ``(H, W)`` in [0, 1] (1.0 = closest).
        divergence_ratio:
            Optional override for max separation as fraction of width.
        method:
            Override the stereo method for this call (ignores settings).
        """
        p = self._preprocess(
            image, depth_map,
            divergence_ratio=divergence_ratio, method=method,
        )

        left, right = p.stereo_method.warp_and_fill(
            p.rgb_arr, p.depth_f32, p.max_disp, p.fg_mask,
            p.settings, self._get_inpainter(p.settings.inpaint_backend),
        )

        self._release_gpu_cache()

        sbs_arr = _compose_sbs(left, right)
        return Image.fromarray(sbs_arr)

    def inpaint_preview(
        self,
        image: Image.Image,
        depth_map: Image.Image | np.ndarray,
        *,
        method: StereoMethodName | None = None,
    ) -> Image.Image | None:
        """Return the method's inpainted background plate, or None."""
        p = self._preprocess(image, depth_map, method=method)
        plate = p.stereo_method.inpaint_preview(
            p.rgb_arr, p.depth_f32, p.max_disp, p.fg_mask, p.settings,
            self._get_inpainter(p.settings.inpaint_backend),
        )
        if plate is None:
            return None
        return Image.fromarray(plate)

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
            p.rgb_arr, p.depth_f32, p.max_disp, p.fg_mask, p.settings,
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
    ) -> Image.Image:
        """Run depth estimation then synthesize SBS."""
        if hasattr(depth_estimator, "process_raw"):
            depth_f32 = depth_estimator.process_raw(image.convert("RGB"))
            return self.synthesize(image, depth_f32, method=method)
        depth = depth_estimator.process(image.convert("RGB"))
        return self.synthesize(image, depth.convert("L"), method=method)


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
