"""SHARP Gaussian-splat stereo methods.

Two variants that bypass the depth-map pipeline entirely:

- ``sharp_splat``: both eyes are pure splat renders from Apple SHARP's
  3D Gaussian prediction. Correct parallax everywhere, slightly soft
  (SHARP works at 1536^2).
- ``sharp_detail``: same geometry, but colour is re-sampled from the
  original photo wherever the original camera could see that surface
  (~95% of pixels); splat colour only in the disoccluded band.
- ``sharp_hires``: sharp_detail with SHARP run at 2688^2 (1344^2 grid).
- ``sharp_alpha``: sharp_hires with depth-sorted alpha compositing.
- ``sharp_alpha_taichi``: sharp_alpha via the taichi tile rasteriser.

SHARP weights are research-only (Apple ML Research license). These
methods are opt-in and clearly labelled non-commercial.
"""

from __future__ import annotations

import logging
import time
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.splat_render import RenderMode
from pystereo_core.stereo.timing import record_step

logger = logging.getLogger(__name__)

BASELINE_M = 0.063


class _SharpBase(BaseStereoMethod):
    """Shared logic for both SHARP splat variants."""

    needs_depth: ClassVar[bool] = False
    _detail_transfer: ClassVar[bool] = False
    _internal: ClassVar[int] = 1536  # SHARP internal size, see sharp_predict.predict_gaussians
    _render_mode: ClassVar[RenderMode] = "zbuf"

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
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pystereo_core.stereo.sharp_predict import cache_note, predict_gaussians
        from pystereo_core.stereo.splat_render import render_stereo

        t0 = time.perf_counter()
        npz_path = predict_gaussians(
            image, internal=self._internal,
            use_cache=settings.sharp_disk_cache, intermediates=intermediates,
        )
        record_step(
            intermediates, "SHARP prediction" + cache_note(intermediates),
            time.perf_counter() - t0,
        )

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        photo: np.ndarray | None = None
        if self._detail_transfer:
            photo = np.asarray(image.convert("RGB"))

        t0 = time.perf_counter()
        result = render_stereo(
            str(npz_path),
            baseline_m=BASELINE_M,
            converge_m=None,
            subject_mask=subject_mask,
            photo=photo,
            mode=self._render_mode,
        )
        record_step(intermediates, "Splat render", time.perf_counter() - t0)
        if intermediates is not None:
            backend = "torch"
            if self._render_mode == "alpha_taichi":
                from pystereo_core.stereo.taichi_render import is_taichi_available

                backend = "taichi" if is_taichi_available() else "torch"
            intermediates["render_backend"] = backend

        if intermediates is not None:
            if "center_rgb" in result:
                intermediates["splat_rgb"] = result["center_rgb"]
            if "depth01" in result:
                intermediates["depth01"] = result["depth01"]

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
    ui_info: ClassVar[str] = (
        "About 1 minute on a MacBook M4. Selfies can feel uncomfortable "
        "to look at - suits mid-to-far shots better. True 3D parallax, "
        "but the image is slightly soft and depth edges can look harsh."
    )
    _detail_transfer: ClassVar[bool] = False


class SharpDetailMethod(_SharpBase):
    # TODO: Investigate why the lower part of the output looks broken while
    # the upper part looks pretty good (detail-transfer / reprojection bug?).
    name: ClassVar[str] = "sharp_detail"
    label: ClassVar[str] = "SHARP Detail"
    description: ClassVar[str] = (
        "Same SHARP splat geometry as sharp_splat, but colour is "
        "re-sampled from the original photo wherever the original camera "
        "could see that surface (~99%% of pixels). Full photo sharpness, "
        "same 3D geometry. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "About 50 seconds on a MacBook M4. Keeps original photo sharpness "
        "on most pixels. Known bug: the lower part of the image often "
        "looks broken while the upper part looks fine."
    )
    _detail_transfer: ClassVar[bool] = True


class SharpHiresMethod(_SharpBase):
    # TODO: Same lower-half breakage as sharp_detail (detail_transfer). Also
    # remeasure peak memory on M4 - suspected higher than sharp_detail.
    name: ClassVar[str] = "sharp_hires"
    label: ClassVar[str] = "SHARP Hi-res Detail"
    description: ClassVar[str] = (
        "sharp_detail with SHARP run at 2688^2 instead of 1536^2 (1344^2 "
        "Gaussian grid, 3.6 M Gaussians): tighter silhouettes and a visibly "
        "sharper disocclusion band. ~5x slower prediction (about 95 s on an "
        "M-series Mac), 3x memory. Experimental - outside the model's "
        "training resolution. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "About 1 minute 30 seconds on a MacBook M4. Same known bug as "
        "SHARP Detail: the lower half of the image often looks broken. "
        "May use more memory too (needs a fresh measurement)."
    )
    _detail_transfer: ClassVar[bool] = True
    _internal: ClassVar[int] = 2688


class SharpAlphaMethod(_SharpBase):
    name: ClassVar[str] = "sharp_alpha"
    label: ClassVar[str] = "SHARP Alpha"
    description: ClassVar[str] = (
        "sharp_hires rendered with proper 3DGS compositing: Gaussians "
        "depth-sorted per pixel and alpha-blended front to back, median "
        "depth. Cleanest silhouettes and sharpest disocclusion band of the "
        "SHARP methods. Slow: about 2 min per photo on an M-series Mac "
        "(the per-pixel sort runs in torch). Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Around 2+ minutes on a MacBook M4. Cleanest SHARP silhouettes. "
        "Prefer SHARP Alpha (taichi) if available - same look, much faster "
        "render step."
    )
    _detail_transfer: ClassVar[bool] = True
    _internal: ClassVar[int] = 2688
    _render_mode: ClassVar[RenderMode] = "alpha"


class SharpAlphaTaichiMethod(_SharpBase):
    name: ClassVar[str] = "sharp_alpha_taichi"
    label: ClassVar[str] = "SHARP Alpha (taichi)"
    description: ClassVar[str] = (
        "Same output as sharp_alpha, rendered by a taichi tile rasteriser "
        "on Metal/GPU: the render step drops from about 2 min to under a "
        "second, leaving SHARP prediction (~90 s at 2688^2) as the only "
        "cost. Needs taichi (pip install taichi, Python <= 3.13); falls "
        "back to the torch renderer otherwise. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Same quality as SHARP Alpha, with a faster splat render when Taichi "
        "is available. The result line shows whether Taichi or the torch "
        "fallback rendered it."
    )
    uses_taichi: ClassVar[bool] = True
    _detail_transfer: ClassVar[bool] = True
    _internal: ClassVar[int] = 2688
    _render_mode: ClassVar[RenderMode] = "alpha_taichi"


class SharpDepthMethod(_SharpBase):
    """Use SHARP's rendered depth with the existing warp+inpaint pipeline."""

    # TODO: Compare vs Per-Eye Inpaint - results look similar; decide if we
    # need both or can drop one.
    name: ClassVar[str] = "sharp_depth"
    label: ClassVar[str] = "SHARP Depth"
    description: ClassVar[str] = (
        "Renders SHARP's 3D Gaussian scene to extract a high-quality "
        "metric depth map, then feeds it through the proven warp+inpaint "
        "pipeline (per_eye_inpaint). Better depth than Depth Anything, "
        "familiar warp look. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Uses SHARP for depth, then the same warp + inpaint path as "
        "Per-Eye Inpaint - results look similar. Needs the SHARP "
        "checkpoint (slower than plain Per-Eye Inpaint)."
    )
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from pystereo_core.stereo.sharp_predict import cache_note, predict_gaussians
        from pystereo_core.stereo.splat_render import SharpScene

        t0 = time.perf_counter()
        npz_path = predict_gaussians(
            image,
            use_cache=settings.sharp_disk_cache, intermediates=intermediates,
        )
        record_step(
            intermediates, "SHARP prediction" + cache_note(intermediates),
            time.perf_counter() - t0,
        )
        t0 = time.perf_counter()
        scene = SharpScene(str(npz_path))
        _, center_depth, _ = scene.render(0.0, 0.0)
        record_step(intermediates, "Depth render", time.perf_counter() - t0)

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

        if intermediates is not None:
            intermediates["depth01"] = depth_01

        pipeline = self.nested_pipeline(settings)
        sbs = pipeline.synthesize(
            image, depth_01, method="per_eye_inpaint", intermediates=intermediates,
        )

        w = sbs.width // 2
        left = np.array(sbs.crop((0, 0, w, sbs.height)))
        right = np.array(sbs.crop((w, 0, sbs.width, sbs.height)))
        return left, right


class SharpMeshMethod(_SharpBase):
    """Forward mesh rendering from SHARP depth - no AI inpainting."""

    name: ClassVar[str] = "sharp_mesh"
    label: ClassVar[str] = "SHARP Mesh"
    deprecated: ClassVar[bool] = True
    description: ClassVar[str] = (
        "Renders SHARP's depth as a forward-splatted mesh. Triangles "
        "fill small gaps naturally; large disocclusions get stretched "
        "from neighbours (no AI inpainting). Sharp edges at depth "
        "boundaries. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Deprecated. Mesh fill from SHARP depth - no AI inpainting. "
        "Sharp edges at depth boundaries; large gaps can stretch."
    )
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pystereo_core.stereo.mesh_render import render_mesh_stereo
        from pystereo_core.stereo.sharp_predict import cache_note, predict_gaussians
        from pystereo_core.stereo.splat_render import SharpScene

        t0 = time.perf_counter()
        npz_path = predict_gaussians(
            image,
            use_cache=settings.sharp_disk_cache, intermediates=intermediates,
        )
        record_step(
            intermediates, "SHARP prediction" + cache_note(intermediates),
            time.perf_counter() - t0,
        )
        t0 = time.perf_counter()
        scene = SharpScene(str(npz_path))

        _, center_depth, _ = scene.render(0.0, 0.0)
        record_step(intermediates, "Depth render", time.perf_counter() - t0)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        if subject_mask is not None and subject_mask.any():
            converge_m = float(np.nanmedian(center_depth[subject_mask]))
        else:
            converge_m = float(np.nanpercentile(center_depth, 10))

        rgb = np.asarray(image.convert("RGB"))
        t0 = time.perf_counter()
        result = render_mesh_stereo(
            rgb, center_depth, scene.f_px,
            baseline_m=BASELINE_M,
            converge_m=converge_m,
        )
        record_step(intermediates, "Mesh render", time.perf_counter() - t0)

        if intermediates is not None:
            valid = np.isfinite(center_depth) & (center_depth > 0)
            inv_d = np.zeros_like(center_depth)
            inv_d[valid] = 1.0 / center_depth[valid]
            lo, hi = float(inv_d[valid].min()), float(inv_d[valid].max())
            depth_01 = (inv_d - lo) / max(hi - lo, 1e-6)
            depth_01[~valid] = 0.0
            intermediates["depth01"] = depth_01.astype(np.float32)

        logger.info(
            "SHARP mesh: converge=%.2fm",
            converge_m,
        )

        return result["left"], result["right"]


class SharpTaichiMethod(_SharpBase):
    """Taichi-accelerated Gaussian splatting (same look, faster render)."""

    name: ClassVar[str] = "sharp_taichi"
    label: ClassVar[str] = "SHARP Splat (taichi)"
    description: ClassVar[str] = (
        "Same EWA Gaussian splatting as sharp_splat but compositing runs "
        "on Metal/GPU via taichi (5-10x faster). Falls back to the torch "
        "renderer if taichi is not installed. Research-only license."
    )
    ui_info: ClassVar[str] = (
        "Same stereo as SHARP Splat, with a faster splat render when Taichi "
        "is available (~20 s on a MacBook M4). The result line shows whether "
        "Taichi or the torch fallback rendered it."
    )
    uses_taichi: ClassVar[bool] = True
    _detail_transfer: ClassVar[bool] = False

    def synthesize(
        self,
        image: Image.Image,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        intermediates: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        from pystereo_core.stereo.sharp_predict import cache_note, predict_gaussians
        from pystereo_core.stereo.splat_render import (
            SharpScene,
            linear_to_srgb,
        )
        from pystereo_core.stereo.taichi_render import select_ewa_compositor

        t0 = time.perf_counter()
        npz_path = predict_gaussians(
            image,
            use_cache=settings.sharp_disk_cache, intermediates=intermediates,
        )
        record_step(
            intermediates, "SHARP prediction" + cache_note(intermediates),
            time.perf_counter() - t0,
        )
        t_render = time.perf_counter()
        scene = SharpScene(str(npz_path))

        composite = select_ewa_compositor()
        backend = "taichi" if composite.__name__ == "composite_ewa_taichi" else "torch"
        logger.info("SHARP taichi: using %s compositing backend", backend)

        subject_mask: np.ndarray | None = None
        if fg_mask is not None:
            subject_mask = fg_mask > 0.5

        center_rgb, d0, _ = scene.render(0.0, 0.0)
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
        record_step(intermediates, "Splat render", time.perf_counter() - t_render)

        def finish(rgb: np.ndarray, holes: np.ndarray) -> np.ndarray:
            img = (linear_to_srgb(rgb) * 255).astype(np.uint8)
            if holes.max():
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.cvtColor(
                    cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA),
                    cv2.COLOR_BGR2RGB,
                )
            return img

        if intermediates is not None:
            intermediates["render_backend"] = backend
            intermediates["splat_rgb"] = (
                linear_to_srgb(center_rgb) * 255
            ).astype(np.uint8)
            valid = np.isfinite(d0) & (d0 > 0)
            inv_d = np.zeros_like(d0)
            inv_d[valid] = 1.0 / d0[valid]
            lo, hi = float(inv_d[valid].min()), float(inv_d[valid].max())
            depth_01 = (inv_d - lo) / max(hi - lo, 1e-6)
            depth_01[~valid] = 0.0
            intermediates["depth01"] = depth_01.astype(np.float32)

        logger.info(
            "SHARP taichi: converge=%.2fm, backend=%s",
            converge_m, backend,
        )

        return finish(l_rgb, l_h), finish(r_rgb, r_h)
