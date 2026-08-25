"""Method: Iterative Local Patch Fill.

Applies the key insight from "3D Photography using Context-aware Layered
Depth Inpainting" (CVPR 2020): instead of inpainting all disocclusion holes
in one pass, break them into connected components, sort by depth
(back-to-front), and inpaint each patch locally.  Each filled region becomes
context for subsequent patches, giving the inpainter small well-constrained
holes with rich surroundings every time.

Pipeline:
1. Warp to left / right eyes via hybrid z-buffer remap.
2. For each eye, find connected components of the occlusion mask.
3. Merge tiny components, sort by median depth (farthest first).
4. For each component: extract a local crop with context padding,
   inpaint just that component with LaMa, paste back.
5. Spatially-varying colour correction + sharpening.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import cv2
import numpy as np

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.bg_plate_fill import _match_inpaint_color, _sharpen_inpainted_region
from pystereo_core.stereo.warp import dilate_occlusion_mask, hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)

_CONTEXT_PAD = 64
_MIN_COMPONENT_PX = 32
_TELEA_THRESHOLD_PX = 80
_LAMA_MIN_CROP = 128


def _iterative_patch_fill(
    eye_img: np.ndarray,
    occ_mask: np.ndarray,
    depth_f32: np.ndarray,
    inpainter: InpaintBackend,
) -> None:
    """In-place iterative local-patch inpainting, back-to-front by depth.

    Modifies *eye_img* directly — each filled region becomes visible
    context for subsequent patches.
    """
    h, w = occ_mask.shape[:2]
    mask_bin = (occ_mask > 0).astype(np.uint8)
    total_hole = int(mask_bin.sum())
    if total_hole == 0:
        return

    n_labels, labels = cv2.connectedComponents(mask_bin, connectivity=8)
    if n_labels <= 1:
        return

    components: list[tuple[float, int, np.ndarray]] = []
    telea_combined = np.zeros_like(mask_bin)

    for label_id in range(1, n_labels):
        comp_mask = (labels == label_id)
        n_px = int(comp_mask.sum())

        if n_px < _MIN_COMPONENT_PX:
            telea_combined[comp_mask] = 1
            continue

        if n_px < _TELEA_THRESHOLD_PX:
            telea_combined[comp_mask] = 1
            continue

        ring = cv2.dilate(
            comp_mask.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )
        neighbor = (ring > 0) & ~comp_mask & (mask_bin == 0)
        if np.any(neighbor):
            median_depth = float(np.median(depth_f32[neighbor]))
        else:
            median_depth = 0.5

        components.append((median_depth, label_id, comp_mask))

    components.sort(key=lambda t: t[0])

    logger.info(
        "Iterative fill: %d LaMa patches + %d px Telea (total %d hole px)",
        len(components),
        int(telea_combined.sum()),
        total_hole,
    )

    for i, (depth_val, _label_id, comp_mask) in enumerate(components):
        ys, xs = np.where(comp_mask)
        y0 = max(0, int(ys.min()) - _CONTEXT_PAD)
        y1 = min(h, int(ys.max()) + 1 + _CONTEXT_PAD)
        x0 = max(0, int(xs.min()) - _CONTEXT_PAD)
        x1 = min(w, int(xs.max()) + 1 + _CONTEXT_PAD)

        crop_h, crop_w = y1 - y0, x1 - x0
        if crop_h < _LAMA_MIN_CROP:
            pad_y = (_LAMA_MIN_CROP - crop_h) // 2 + 1
            y0, y1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
        if crop_w < _LAMA_MIN_CROP:
            pad_x = (_LAMA_MIN_CROP - crop_w) // 2 + 1
            x0, x1 = max(0, x0 - pad_x), min(w, x1 + pad_x)

        crop_rgb = eye_img[y0:y1, x0:x1].copy()
        crop_mask = comp_mask[y0:y1, x0:x1].astype(np.uint8) * 255

        filled = inpainter.inpaint(crop_rgb, crop_mask)

        if filled.shape[:2] != crop_rgb.shape[:2]:
            fh, fw = filled.shape[:2]
            ch, cw = crop_rgb.shape[:2]
            if fh >= ch and fw >= cw:
                filled = filled[:ch, :cw]
            else:
                filled = cv2.resize(filled, (cw, ch), interpolation=cv2.INTER_LINEAR)

        hole_local = comp_mask[y0:y1, x0:x1]
        eye_img[y0:y1, x0:x1][hole_local] = filled[hole_local]

    if np.any(telea_combined):
        telea_mask_u8 = telea_combined * 255
        bgr = cv2.cvtColor(eye_img, cv2.COLOR_RGB2BGR)
        eye_img[:] = cv2.cvtColor(
            cv2.inpaint(bgr, telea_mask_u8, 5, cv2.INPAINT_TELEA),
            cv2.COLOR_BGR2RGB,
        )


class IterativeFillMethod(BaseStereoMethod):
    name: ClassVar[str] = "iterative_fill"
    label: ClassVar[str] = "Iterative Local Patch (Deprecated)"
    description: ClassVar[str] = (
        "Per-eye disocclusion fill using depth-sorted iterative local "
        "patching.  Breaks holes into connected components, sorts "
        "back-to-front by depth, and inpaints each patch locally with "
        "LaMa so each fill becomes context for the next."
    )

    SETTING_OVERRIDES: ClassVar[dict[str, Any]] = {
        "guided_filter_eps": 5e-3,
        "depth_gamma": 1.5,
        "depth_healing_edge_blur_sigma": 4.0,
        "depth_healing_mask_dilate_px": 8,
    }

    def warp_and_fill(
        self,
        rgb_arr: np.ndarray,
        depth_f32: np.ndarray,
        max_disp: float,
        fg_mask: np.ndarray | None,
        settings: StereoSettings,
        inpainter: InpaintBackend,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = rgb_arr.shape[:2]

        left, left_mask = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "left")
        right, right_mask = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "right")

        if settings.inpaint_mask_dilate_px > 0:
            left_mask = dilate_occlusion_mask(left_mask, settings.inpaint_mask_dilate_px)
            right_mask = dilate_occlusion_mask(right_mask, settings.inpaint_mask_dilate_px)

        for eye_name, eye_img, occ_mask in (
            ("left", left, left_mask),
            ("right", right, right_mask),
        ):
            if not np.any(occ_mask):
                continue

            eye_before = eye_img.copy()
            _iterative_patch_fill(eye_img, occ_mask, depth_f32, inpainter)
            eye_img[:] = _match_inpaint_color(eye_before, eye_img, occ_mask)
            _sharpen_inpainted_region(eye_img, occ_mask)

        return left, right
