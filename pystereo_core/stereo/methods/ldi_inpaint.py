"""Method 7: LDI Inpaint — Context-aware neural inpainting (CVPR 2020).

Uses the three specialised partial-convolution networks from "3D Photography
using Context-aware Layered Depth Inpainting" (Shih et al.) to fill
disocclusion holes with structure-aware content:

1. **Edge network** hallucinate depth-edge continuations into masked regions.
2. **Depth network** inpaint depth guided by those edges.
3. **Color network** inpaint RGB guided by edges.

Pipeline: build foreground inpaint mask → detect depth edges → run
Edge → Depth → Color networks to synthesise a background plate → warp
the plate and the original image to left/right eyes → composite.

Checkpoints are lazily loaded from ``PYSTEREO_LDI_CHECKPOINT_DIR`` (default:
``experiments/ldi_inpainting/repo/checkpoints/`` relative to the repo root).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import torch

from pystereo_core.stereo.config import StereoSettings
from pystereo_core.stereo.inpaint import InpaintBackend
from pystereo_core.stereo.methods.base import BaseStereoMethod
from pystereo_core.stereo.methods.bg_plate_fill import (
    BgPlateFillMethod,
    _match_inpaint_color,
    _sharpen_inpainted_region,
)
from pystereo_core.stereo.warp import dilate_occlusion_mask, hybrid_zbuf_remap_eye

logger = logging.getLogger(__name__)

_DEFAULT_CKPT_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "ldi_inpainting"
    / "repo"
    / "checkpoints"
)

_MODELS: dict[str, Any] = {}
_DEVICE: torch.device | None = None


def _get_device() -> torch.device:
    global _DEVICE
    if _DEVICE is None:
        from pystereo_core.registry import detect_device

        _DEVICE = torch.device(detect_device())
        logger.info("LDI inpaint device: %s", _DEVICE)
    return _DEVICE


def _ensure_models() -> dict[str, Any]:
    """Lazily load checkpoints on first use (thread-unsafe but fine for us)."""
    if _MODELS:
        return _MODELS

    from pystereo_core.stereo.ldi_nets import InpaintColorNet, InpaintDepthNet, InpaintEdgeNet

    ckpt_dir = Path(os.environ.get("PYSTEREO_LDI_CHECKPOINT_DIR", str(_DEFAULT_CKPT_DIR)))
    device = _get_device()

    edge_path = ckpt_dir / "edge-model.pth"
    depth_path = ckpt_dir / "depth-model.pth"
    color_path = ckpt_dir / "color-model.pth"

    for p in (edge_path, depth_path, color_path):
        if not p.exists():
            raise FileNotFoundError(
                f"LDI checkpoint not found: {p}\n"
                f"Download from the 3D-Photo-Inpainting repo or set PYSTEREO_LDI_CHECKPOINT_DIR."
            )

    logger.info("Loading LDI edge model from %s …", edge_path)
    edge_net = InpaintEdgeNet()
    edge_state = torch.load(str(edge_path), map_location=device, weights_only=False)
    edge_net.load_state_dict(edge_state)
    edge_net.to(device).eval()

    logger.info("Loading LDI depth model from %s …", depth_path)
    depth_net = InpaintDepthNet()
    depth_state = torch.load(str(depth_path), map_location=device, weights_only=False)
    depth_net.load_state_dict(depth_state)
    depth_net.to(device).eval()

    logger.info("Loading LDI color model from %s …", color_path)
    color_net = InpaintColorNet()
    color_state = torch.load(str(color_path), map_location=device, weights_only=False)
    color_net.load_state_dict(color_state)
    color_net.to(device).eval()

    _MODELS["edge"] = edge_net
    _MODELS["depth"] = depth_net
    _MODELS["color"] = color_net
    logger.info("LDI models loaded successfully (%s)", device)
    return _MODELS


# ── Depth-edge detection (adapted from bilateral_filtering.py) ────────


def _depth_discontinuity_map(
    depth_f32: np.ndarray,
    threshold: float = 0.04,
) -> np.ndarray:
    """Binary map of depth discontinuities via disparity gradient thresholding.

    Operates on inverse depth (disparity) as in the original paper:
    large disparity jumps indicate foreground/background boundaries.
    """
    disp = 1.0 / np.maximum(depth_f32, 1e-6)

    u_diff = np.abs(disp[1:, :] - disp[:-1, :])
    l_diff = np.abs(disp[:, 1:] - disp[:, :-1])

    u_over = np.pad((u_diff > threshold).astype(np.float32), ((0, 1), (0, 0)))
    b_over = np.pad((u_diff > threshold).astype(np.float32), ((1, 0), (0, 0)))
    l_over = np.pad((l_diff > threshold).astype(np.float32), ((0, 0), (0, 1)))
    r_over = np.pad((l_diff > threshold).astype(np.float32), ((0, 0), (1, 0)))

    return np.clip(u_over + b_over + l_over + r_over, 0.0, 1.0)


def _canny_edges(depth_f32: np.ndarray) -> np.ndarray:
    """Canny edge detector on the depth map for structural guidance."""
    depth_u8 = (np.clip(depth_f32, 0, 1) * 255).astype(np.uint8)
    edges = cv2.Canny(depth_u8, 30, 100)
    return (edges > 0).astype(np.float32)


# ── Neural inpainting of a masked region ──────────────────────────────


def _run_ldi_inpaint(
    rgb_f32: np.ndarray,
    depth_f32: np.ndarray,
    mask_u8: np.ndarray,
    models: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the 3-stage LDI inpainting pipeline on a masked region.

    Parameters
    ----------
    rgb_f32 : (H,W,3) float32 in [0,1]
    depth_f32 : (H,W) float32 in [0,1], 1.0 = closest (disparity convention)
    mask_u8 : (H,W) uint8, 255 = hole to fill

    Returns
    -------
    inpainted_rgb : (H,W,3) float32 in [0,1]
    inpainted_depth : (H,W) float32 in [0,1]
    """
    h, w = mask_u8.shape[:2]
    hole = mask_u8 > 0

    if not np.any(hole):
        return rgb_f32.copy(), depth_f32.copy()

    mask_bin = hole.astype(np.float32)
    context_bin = 1.0 - mask_bin

    # Depth edges: combine disparity-gradient discontinuities + Canny
    disc_map = _depth_discontinuity_map(depth_f32)
    canny_map = _canny_edges(depth_f32)
    edge_map = np.clip(disc_map + canny_map, 0, 1)
    # Edges are kept in both context and mask regions (original mesh.py l.1700)
    edge_combined = edge_map * (context_bin + mask_bin)

    # depth_f32 is already in disparity convention (1=close, 0=far),
    # same as MiDaS output the networks were trained on.
    disp_norm = depth_f32 / (depth_f32.max() + 1e-8)

    # The original pipeline zeroes input channels in the hole region:
    # the networks were trained to see black pixels there and paint over
    # them.  Passing foreground RGB/depth into the hole causes white output.
    rgb_for_net = rgb_f32 * context_bin[:, :, np.newaxis]
    depth_for_net = depth_f32 * context_bin
    disp_for_net = disp_norm * context_bin

    # Crop to bounding box of the hole with generous padding so the
    # networks get plenty of surrounding context.
    ys, xs = np.where(hole)
    if len(ys) == 0:
        return rgb_f32.copy(), depth_f32.copy()

    pad = max(64, int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.3))
    y0 = max(0, ys.min() - pad)
    y1 = min(h, ys.max() + 1 + pad)
    x0 = max(0, xs.min() - pad)
    x1 = min(w, xs.max() + 1 + pad)

    c_rgb = rgb_for_net[y0:y1, x0:x1]
    c_depth = depth_for_net[y0:y1, x0:x1]
    c_disp = disp_for_net[y0:y1, x0:x1]
    c_mask = mask_bin[y0:y1, x0:x1]
    c_context = context_bin[y0:y1, x0:x1]
    c_edge = edge_combined[y0:y1, x0:x1]

    def _to_t(arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2:
            return torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

    t_rgb = _to_t(c_rgb)
    t_depth = _to_t(c_depth)
    t_disp = _to_t(c_disp)
    t_mask = _to_t(c_mask)
    t_context = _to_t(c_context)
    t_edge = _to_t(c_edge)

    # Stage 1: Edge inpainting — hallucinate depth edges in the hole
    edge_out = models["edge"].forward_3P(
        t_mask, t_context, t_rgb, t_disp, t_edge, device=device
    )
    merged_edge = t_edge * (1 - t_mask) + edge_out * t_mask
    merged_edge = (merged_edge > 0.5).float()

    # Stage 2: Depth inpainting — fill depth guided by edges
    depth_out = models["depth"].forward_3P(
        t_mask, t_context, t_depth, merged_edge, device=device
    )

    # Stage 3: Color inpainting — fill RGB guided by edges
    color_out = models["color"].forward_3P(
        t_mask, t_context, t_rgb, merged_edge, device=device
    )

    color_np = color_out[0].cpu().numpy().transpose(1, 2, 0)
    depth_np = depth_out[0, 0].cpu().numpy()

    out_rgb = rgb_f32.copy()
    out_depth = depth_f32.copy()
    c_hole = c_mask > 0
    out_rgb[y0:y1, x0:x1][c_hole] = np.clip(color_np[c_hole], 0, 1)
    out_depth[y0:y1, x0:x1][c_hole] = np.clip(depth_np[c_hole], 0, 1)

    return out_rgb, out_depth


# ── Stereo method class ──────────────────────────────────────────────


class LdiInpaintMethod(BaseStereoMethod):
    name: ClassVar[str] = "ldi_inpaint"
    label: ClassVar[str] = "LDI Context-Aware Inpaint (Deprecated)"
    description: ClassVar[str] = (
        "Context-aware layered depth inpainting (Shih et al., CVPR 2020). "
        "Uses three specialised partial-convolution networks to hallucinate "
        "depth edges, inpaint depth, and inpaint colour behind foreground "
        "objects.  Produces a neural background plate that is warped into "
        "both eyes for stereo-consistent fill."
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
        """Hybrid pipeline: bg_plate_fill for bulk, LDI refinement on strips.

        1. Delegate to BgPlateFillMethod for the initial warp + fill
           (LaMa handles large background-plate fills well).
        2. Refine each eye's disocclusion strips with the LDI partial-
           convolution networks — these small, edge-adjacent fills are
           exactly what the networks were trained for.
        """
        # Stage 1: standard bg_plate_fill for the bulk warp + fill
        left, right = BgPlateFillMethod().warp_and_fill(
            rgb_arr, depth_f32, max_disp, fg_mask, settings, inpainter,
        )

        # Stage 2: LDI neural refinement on per-eye disocclusion strips
        try:
            models = _ensure_models()
            device = _get_device()
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning(
                "LDI models unavailable (%s); returning bg_plate_fill result",
                exc,
            )
            return left, right

        h, w = rgb_arr.shape[:2]

        # Re-warp to get the original occlusion masks (before bg fill)
        _, left_occ = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "left")
        _, right_occ = hybrid_zbuf_remap_eye(rgb_arr, depth_f32, max_disp, "right")
        if settings.inpaint_mask_dilate_px > 0:
            left_occ = dilate_occlusion_mask(left_occ, settings.inpaint_mask_dilate_px)
            right_occ = dilate_occlusion_mask(right_occ, settings.inpaint_mask_dilate_px)

        for eye_name, eye_img, occ_mask in (
            ("left", left, left_occ),
            ("right", right, right_occ),
        ):
            if not np.any(occ_mask):
                continue

            hole_px = int(np.count_nonzero(occ_mask))
            hole_pct = 100.0 * hole_px / occ_mask.size
            if hole_pct > 15.0:
                logger.info(
                    "LDI %s eye: strip too large (%.1f%%); keeping bg_plate fill",
                    eye_name, hole_pct,
                )
                continue

            logger.info(
                "LDI %s eye: refining %d strip pixels (%.1f%%)",
                eye_name, hole_px, hole_pct,
            )

            eye_f32 = eye_img.astype(np.float32) / 255.0
            eye_before = eye_img.copy()

            strip_rgb, _ = _run_ldi_inpaint(
                eye_f32, depth_f32, occ_mask, models, device,
            )

            strip_u8 = np.clip(strip_rgb * 255, 0, 255).astype(np.uint8)
            strip_u8 = _match_inpaint_color(eye_before, strip_u8, occ_mask)

            hole = occ_mask > 0
            eye_img[hole] = strip_u8[hole]

        return left, right
