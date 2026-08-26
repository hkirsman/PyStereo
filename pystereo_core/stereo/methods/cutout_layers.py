"""Method: Cutout Layers — alpha matte over a fully inpainted background.

Every other method warps one flat image and then patches the holes the
warp tears open.  That is the hard version of the problem: the fill region
is a thin, ragged band whose shape is dictated by the depth edge, and any
error in it lands directly on the subject's silhouette.

This method splits the photo into two layers first:

1. **Background layer** — the subject is inpainted away *entirely*, giving
   a complete background plate with no holes anywhere.  One large coherent
   region is what LaMa is good at, and it is filled with full surrounding
   context rather than from the two sides of a narrow slit.
2. **Foreground layer** — the subject, carried by BiRefNet's soft alpha
   matte so its edge stays anti-aliased.

Each layer is warped independently and the foreground is composited back
over the background.  Nothing is patched along the subject's outline: its
edge quality is set by the matte, not by a hole boundary.

Removing the subject does not make the background flat, though — the rest
of the scene keeps its own depth steps (a building edge, the line where a
path meets grass) and those still disocclude when the plate is warped.
Measured on a portrait-in-a-courtyard photo, none of that residual lands
near the subject, but it is visible, so the warped plate is patched for it
before compositing.

This is the layered-depth-image approach used by "3D Photography using
Context-aware Layered Depth Inpainting" (Shih et al., CVPR 2020) and,
broadly, by phone Portrait/Spatial photo pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.depth import guided_filter_depth
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.bg_plate_fill import _match_inpaint_color
from pystereo_core.stereo.methods.fullres_warp import (
    FullResWarpMethod,
    _downscale_for_inpaint,
    _upscale_to,
)
from pystereo_core.stereo.warp import (
    dilate_hole_mask,
    hybrid_zbuf_remap_eye,
    warp_layer_eye,
)

logger = logging.getLogger(__name__)


def _refine_matte(
    alpha: np.ndarray,
    rgb: np.ndarray,
    *,
    radius: int = 4,
    eps: float = 1e-4,
) -> np.ndarray:
    """Snap the matte to colour edges, keeping it soft.

    BiRefNet's alpha is already soft but can sit a pixel or two off the
    true edge on low-contrast boundaries.  A guided filter against the
    photo pulls it onto the edge without hardening it — the partial
    coverage at the boundary is what anti-aliases the composite.
    """
    refined = guided_filter_depth(alpha, rgb, radius=radius, eps=eps)
    return np.clip(refined, 0.0, 1.0)


def _extrapolate_background_depth(
    depth_f32: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    """Continue the background's depth across the subject's footprint.

    The background layer has to be warped by the depth of whatever is
    *behind* the subject.  Substituting a single flat value there (the
    older background-plate code uses the median) invents a depth cliff at
    the subject's outline, which tears fresh holes in the plate warp.
    Diffusing the surrounding depth inward keeps the ground plane and any
    gradient continuous, so the plate warps cleanly.
    """
    depth_u8 = np.clip(depth_f32 * 255.0, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(depth_u8, subject_mask, 9, cv2.INPAINT_TELEA)
    bg_depth = filled.astype(np.float32) / 255.0
    return np.where(subject_mask > 0, bg_depth, depth_f32).astype(np.float32)


class CutoutLayersMethod(BaseStereoMethod):
    name: ClassVar[str] = "cutout_layers"
    label: ClassVar[str] = "Cutout Layers (Matte + Full BG Inpaint)"
    description: ClassVar[str] = (
        "Splits the photo into a soft-matted foreground and a fully "
        "inpainted background, warps each layer separately, then "
        "composites. The background plate is complete, so there is no "
        "disocclusion mask to patch and the subject's edge is set by the "
        "alpha matte rather than by a hole boundary. Needs a good "
        "foreground segmentation; falls back to Full-Res Warp without one."
    )

    wants_full_res: ClassVar[bool] = True

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
    }

    #: Margin (px) added around the subject before inpainting it away.  Only
    #: needs to cover matte slop, not the disparity — the plate is warped,
    #: so the revealed strip comes from real plate pixels either way.
    subject_margin_px: ClassVar[int] = 12

    def _build_layers(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        fg_mask: np.ndarray,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(background_layer, background_depth, refined_alpha)``."""
        h, w = rgb_arr.shape[:2]

        alpha = _refine_matte(fg_mask, rgb_arr)

        margin = self.subject_margin_px
        kern = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        subject = cv2.dilate((alpha > 0.05).astype(np.uint8), kern) * 255

        coverage = float(np.count_nonzero(subject)) / float(subject.size)
        logger.info(
            "Cutout layers: subject covers %.1f%% of frame, inpainting it away",
            100.0 * coverage,
        )

        small, mask_small, scale = _downscale_for_inpaint(
            rgb_arr, subject, settings.max_processing_dim
        )
        plate_small = inpainter.inpaint(small, mask_small)
        plate_small = _match_inpaint_color(small, plate_small, mask_small)
        # Same trap as above: keep the backend's output only where it filled.
        # (_match_inpaint_color needs the full reconstruction first, since it
        # reads the surrounding ring to measure the colour offset.)
        outside = mask_small == 0
        plate_small[outside] = small[outside]
        bg_layer = _upscale_to(plate_small, h, w) if scale < 1.0 else plate_small

        # Outside the subject the plate must stay the real photo — upscaling
        # or the inpainter's own resampling can otherwise soften the whole
        # frame, not just the part that was invented.
        keep = subject == 0
        bg_layer[keep] = rgb_arr[keep]

        bg_depth = _extrapolate_background_depth(depth_f32, subject)
        return bg_layer, bg_depth, alpha

    def inpaint_preview(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> np.ndarray | None:
        if fg_mask is None or not np.any(fg_mask > 0.5):
            return None
        bg_layer, _, _ = self._build_layers(
            rgb_arr, depth_f32, fg_mask, settings, inpainter,
        )
        return bg_layer

    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        if fg_mask is None or not np.any(fg_mask > 0.5):
            logger.info(
                "Cutout layers: no foreground matte available; "
                "falling back to full-res warp"
            )
            return FullResWarpMethod().warp_and_fill(
                rgb_arr, depth_f32, max_disp, fg_mask, settings, inpainter,
            )

        crack = settings.warp_crack_fill_px
        bg_layer, bg_depth, alpha = self._build_layers(
            rgb_arr, depth_f32, fg_mask, settings, inpainter,
        )

        eyes: list[np.ndarray] = []
        for eye in ("left", "right"):
            bg_warped, bg_holes = hybrid_zbuf_remap_eye(
                bg_layer, bg_depth, max_disp, eye, crack_fill_px=crack,  # type: ignore[arg-type]
            )

            # Removing the subject does not make the background flat: the
            # rest of the scene still has depth steps (building edges, a
            # path boundary) that disocclude when the plate is warped.
            # Those are away from the subject, but they are visible, so
            # they still need filling.
            if np.any(bg_holes):
                hole_pct = 100.0 * float(np.count_nonzero(bg_holes)) / bg_holes.size
                logger.info(
                    "Cutout layers %s eye: %.2f%% of the background plate "
                    "disoccludes at scene depth edges; inpainting",
                    eye, hole_pct,
                )
                bg_holes = dilate_hole_mask(
                    bg_holes, eye, settings.inpaint_mask_dilate_px,  # type: ignore[arg-type]
                    unilateral=settings.unilateral_mask_dilate,
                )
                # Take only the filled pixels: the backends return a full
                # reconstruction, and its unmasked pixels come back softer
                # than the photo they replace.
                patched = inpainter.inpaint(bg_warped, bg_holes)
                filled = bg_holes > 0
                bg_warped[filled] = patched[filled]

            fg_warped, fg_alpha = warp_layer_eye(
                rgb_arr, alpha, depth_f32, max_disp, eye,  # type: ignore[arg-type]
                crack_fill_px=crack,
            )

            a = fg_alpha[:, :, np.newaxis]
            composed = bg_warped.astype(np.float32) * (1.0 - a) + \
                fg_warped.astype(np.float32) * a
            eyes.append(np.clip(composed, 0, 255).astype(np.uint8))

        return eyes[0], eyes[1]

    def warp_preview(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        _unused: Any = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Show each eye's warped foreground matte as the "hole" preview."""
        if fg_mask is None:
            return None
        crack = settings.warp_crack_fill_px
        alpha = _refine_matte(fg_mask, rgb_arr)
        out: list[np.ndarray] = []
        for eye in ("left", "right"):
            warped, a = warp_layer_eye(
                rgb_arr, alpha, depth_f32, max_disp, eye,  # type: ignore[arg-type]
                crack_fill_px=crack,
            )
            out.append(warped)
            out.append((a * 255.0).astype(np.uint8))
        return out[0], out[2], out[1], out[3]
