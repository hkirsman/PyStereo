"""SHARP Gaussian-splat stereo methods.

Two variants that bypass the depth-map pipeline entirely:

- ``sharp_splat``: both eyes are pure splat renders from Apple SHARP's
  3D Gaussian prediction. Correct parallax everywhere, slightly soft
  (SHARP works at 1536^2).
- ``sharp_detail``: same geometry, but colour is re-sampled from the
  original photo wherever the original camera could see that surface
  (~99% of pixels); splat colour only in the disoccluded band.

SHARP weights are research-only (Apple ML Research license). These
methods are opt-in and clearly labelled non-commercial.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod

logger = logging.getLogger(__name__)

BASELINE_M = 0.063


class _SharpBase(BaseStereoMethod):
    """Shared logic for both SHARP splat variants."""

    needs_depth: ClassVar[bool] = False
    _detail_transfer: ClassVar[bool] = False

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {}

    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(
            f"{type(self).__name__} uses synthesize(), not warp_and_fill()"
        )

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pystereo_core.stereo.sharp_predict import predict_gaussians
        from pystereo_core.stereo.splat_render import render_stereo

        npz_path = predict_gaussians(image)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        photo: np.ndarray | None = None
        if self._detail_transfer:
            photo = np.asarray(image.convert("RGB"))

        result = render_stereo(
            str(npz_path),
            baseline_m=BASELINE_M,
            converge_m=None,
            subject_mask=subject_mask,
            photo=photo,
        )

        notes = result.get("notes", {})
        logger.info(
            "SHARP %s: converge=%.2fm, disp=[%.1f, %.1f]px",
            self.name,
            notes.get("converge_m", 0),
            notes.get("disp_px_far", 0),
            notes.get("disp_px_near", 0),
        )

        return result["left"], result["right"]


class SharpSplatMethod(_SharpBase):
    name: ClassVar[str] = "sharp_splat"
    label: ClassVar[str] = "SHARP Splat"
    description: ClassVar[str] = (
        "3D Gaussian splat via Apple SHARP, rendered from two virtual "
        "cameras 63 mm apart. True parallax for every object, no "
        "inpainting step. Slightly soft (SHARP works at 1536^2). "
        "Research-only license."
    )
    _detail_transfer: ClassVar[bool] = False


class SharpDetailMethod(_SharpBase):
    name: ClassVar[str] = "sharp_detail"
    label: ClassVar[str] = "SHARP Detail"
    description: ClassVar[str] = (
        "Same SHARP splat geometry as sharp_splat, but colour is "
        "re-sampled from the original photo wherever the original camera "
        "could see that surface (~99%% of pixels). Full photo sharpness, "
        "same 3D geometry. Research-only license."
    )
    _detail_transfer: ClassVar[bool] = True


class SharpDepthMethod(_SharpBase):
    """Use SHARP's rendered depth with the existing warp+inpaint pipeline."""

    name: ClassVar[str] = "sharp_depth"
    label: ClassVar[str] = "SHARP Depth"
    description: ClassVar[str] = (
        "Renders SHARP's 3D Gaussian scene to extract a high-quality "
        "metric depth map, then feeds it through the proven warp+inpaint "
        "pipeline (per_eye_inpaint). Better depth than Depth Anything, "
        "familiar warp look. Research-only license."
    )
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from pystereo_core.stereo.sharp_predict import predict_gaussians
        from pystereo_core.stereo.splat_render import SharpScene, linear_to_srgb

        npz_path = predict_gaussians(image)
        scene = SharpScene(str(npz_path))
        _, center_depth, _ = scene.render(0.0, 0.0)

        valid = np.isfinite(center_depth) & (center_depth > 0)
        inv_d = np.zeros_like(center_depth)
        inv_d[valid] = 1.0 / center_depth[valid]
        lo, hi = float(inv_d[valid].min()), float(inv_d[valid].max())
        depth_01 = (inv_d - lo) / max(hi - lo, 1e-6)
        depth_01[~valid] = 0.0
        depth_01 = depth_01.astype(np.float32)

        w_img, h_img = image.size
        if depth_01.shape != (h_img, w_img):
            depth_01 = cv2.resize(depth_01, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

        from pystereo_core.stereo.pipeline import StereoPipeline

        pipeline = StereoPipeline(settings=settings)
        sbs = pipeline.synthesize(image, depth_01, method="per_eye_inpaint")

        w = sbs.width // 2
        left = np.array(sbs.crop((0, 0, w, sbs.height)))
        right = np.array(sbs.crop((w, 0, sbs.width, sbs.height)))
        return left, right


class SharpMeshMethod(_SharpBase):
    """Forward mesh rendering from SHARP depth - no AI inpainting."""

    name: ClassVar[str] = "sharp_mesh"
    label: ClassVar[str] = "SHARP Mesh"
    description: ClassVar[str] = (
        "Renders SHARP's depth as a forward-splatted mesh. Triangles "
        "fill small gaps naturally; large disocclusions get stretched "
        "from neighbours (no AI inpainting). Sharp edges at depth "
        "boundaries. Research-only license."
    )
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pystereo_core.stereo.mesh_render import render_mesh_stereo
        from pystereo_core.stereo.sharp_predict import predict_gaussians
        from pystereo_core.stereo.splat_render import SharpScene

        npz_path = predict_gaussians(image)
        scene = SharpScene(str(npz_path))

        _, center_depth, _ = scene.render(0.0, 0.0)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        if subject_mask is not None and subject_mask.any():
            converge_m = float(np.nanmedian(center_depth[subject_mask]))
        else:
            converge_m = float(np.nanpercentile(center_depth, 10))

        rgb = np.asarray(image.convert("RGB"))
        result = render_mesh_stereo(
            rgb, center_depth, scene.f_px,
            baseline_m=BASELINE_M,
            converge_m=converge_m,
        )

        logger.info(
            "SHARP mesh: converge=%.2fm",
            converge_m,
        )

        return result["left"], result["right"]


class SharpTaichiMethod(_SharpBase):
    """Taichi-accelerated Gaussian splatting (same look, faster render)."""

    name: ClassVar[str] = "sharp_taichi"
    label: ClassVar[str] = "SHARP Taichi"
    description: ClassVar[str] = (
        "Same EWA Gaussian splatting as sharp_splat but compositing runs "
        "on Metal/GPU via taichi (5-10x faster). Falls back to the torch "
        "renderer if taichi is not installed. Research-only license."
    )
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from pystereo_core.stereo.sharp_predict import predict_gaussians
        from pystereo_core.stereo.splat_render import (
            SharpScene,
            composite_ewa_torch,
            linear_to_srgb,
        )
        from pystereo_core.stereo.taichi_render import (
            composite_ewa_taichi,
            is_taichi_available,
        )

        npz_path = predict_gaussians(image)
        scene = SharpScene(str(npz_path))

        composite = composite_ewa_taichi if is_taichi_available() else composite_ewa_torch
        backend = "taichi" if is_taichi_available() else "torch"
        logger.info("SHARP taichi: using %s compositing backend", backend)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        _, d0, _ = scene.render(0.0, 0.0)
        if subject_mask is not None and subject_mask.any():
            converge_m = float(np.nanmedian(d0[subject_mask]))
        else:
            converge_m = float(np.nanpercentile(d0, 10))

        f = scene.f_px
        shift = f * BASELINE_M / (2 * converge_m)

        l_proj = scene.project(-BASELINE_M / 2, -shift)
        r_proj = scene.project(+BASELINE_M / 2, +shift)

        l_rgb, l_d, l_h = composite(l_proj)
        r_rgb, r_d, r_h = composite(r_proj)

        def finish(rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
            img = (linear_to_srgb(rgb) * 255).astype(np.uint8)
            if holes.max():
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(
                    cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )
            return img

        logger.info(
            "SHARP taichi: converge=%.2fm, backend=%s",
            converge_m, backend,
        )

        return finish(l_rgb, l_h), finish(r_rgb, r_h)
