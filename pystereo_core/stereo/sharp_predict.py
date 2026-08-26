"""Run Apple SHARP on a photo and cache the resulting 3D Gaussians.

The SHARP model predicts a full 3D Gaussian splat from a single image,
including a hallucinated second layer behind occluders. The Gaussians
are saved as a compressed .npz for the splat renderer.

Requires ``ml-sharp`` (git submodule) and the SHARP checkpoint at
``~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt`` (2.8 GB).
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

SHARP_CKPT_FILENAME = "sharp_2572gikvuh.pt"
SHARP_CKPT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/"
    + SHARP_CKPT_FILENAME
)

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / ".sharp_cache"


def _ckpt_path() -> Path:
    return (
        Path(torch.hub.get_dir()) / "checkpoints" / SHARP_CKPT_FILENAME
    )


def is_sharp_available() -> bool:
    return _ckpt_path().is_file()


def _image_hash(img: np.ndarray) -> str:
    return hashlib.sha1(img.tobytes()).hexdigest()[:12]


def _focal_length_px(width: int, height: int, f_35mm: float) -> float:
    return f_35mm * np.sqrt(width**2.0 + height**2.0) / np.sqrt(36**2 + 24**2)


def _extract_focal_35mm(pil_image: Image.Image) -> float:
    try:
        exif = pil_image.getexif()
        ifd = exif.get_ifd(0x8769)
        f35 = ifd.get(41989)  # FocalLengthIn35mmFilm
        if f35 and f35 >= 10:
            return float(f35)
        fl = ifd.get(37386)  # FocalLength
        if fl and fl >= 1:
            return float(fl) * 8.4 if fl < 10 else float(fl)
    except Exception:
        pass
    return 30.0


@torch.no_grad()
def _predict_image(
    predictor: torch.nn.Module,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
) -> object:
    """Run SHARP inference - inlined from sharp.cli.predict to avoid gsplat import."""
    import torch.nn.functional as F

    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (1536, 1536)
    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)
    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )
    gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    intrinsics = torch.tensor(
        [
            [f_px, 0, width / 2, 0],
            [0, f_px, height / 2, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=device,
        dtype=torch.float32,
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height
    return unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )


def predict_gaussians(
    pil_image: Image.Image,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Convert a photo to 3D Gaussians via Apple SHARP.

    Returns the path to the cached ``.npz`` file containing means, scales,
    quaternions, colours, opacities, focal length, and image dimensions.
    Subsequent calls with the same image return instantly from cache.
    """
    from PIL import ImageOps

    from sharp.models import PredictorParams, create_predictor

    cache = cache_dir or _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    pil_image = ImageOps.exif_transpose(pil_image)
    img = np.asarray(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    out = cache / f"sharp_{_image_hash(img)}.npz"
    if out.exists():
        logger.info("SHARP cache hit: %s", out.name)
        return out

    ckpt = _ckpt_path()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"SHARP checkpoint not found at {ckpt}. "
            "Download it via the pystereo UI or run: "
            f"curl -L -o {ckpt} {SHARP_CKPT_URL}"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    f_35mm = _extract_focal_35mm(pil_image)
    f_px = _focal_length_px(w, h, f_35mm)

    t0 = time.time()
    logger.info("SHARP predict: %dx%d, f_px=%.1f, device=%s", w, h, f_px, device)

    predictor = create_predictor(PredictorParams())
    predictor.load_state_dict(torch.load(ckpt, weights_only=True))
    predictor.eval().to(device)

    g = _predict_image(predictor, img, f_px, torch.device(device))

    np.savez_compressed(
        out,
        means=g.mean_vectors[0].cpu().numpy(),
        scales=g.singular_values[0].cpu().numpy().astype(np.float16),
        quats=g.quaternions[0].cpu().numpy().astype(np.float16),
        colors=g.colors[0].cpu().numpy().astype(np.float16),
        opacities=g.opacities[0].cpu().numpy().astype(np.float16),
        f_px=np.float32(f_px),
        width=w,
        height=h,
        color_space="linearRGB",
    )

    del predictor
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    logger.info(
        "SHARP predict done: %d gaussians in %.1fs -> %s",
        g.mean_vectors.shape[1],
        time.time() - t0,
        out.name,
    )
    return out
