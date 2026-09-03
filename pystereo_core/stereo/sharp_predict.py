"""Run Apple SHARP on a photo and cache the resulting 3D Gaussians.

The SHARP model predicts a full 3D Gaussian splat from a single image,
including a hallucinated second layer behind occluders. The Gaussians
are saved as a compressed .npz for the splat renderer.

Requires ``ml-sharp`` (git submodule) and the SHARP checkpoint at
``~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt`` (2.8 GB).
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

#: Seconds the resident predictor may sit unused before it is unloaded to
#: free ~3 GB of (unified) memory. Loading it back costs ~6 s. Zero or a
#: negative value keeps the predictor loaded until :func:`unload_predictor`.
#: Runtime changes go through :func:`set_idle_unload_s` (web UI setting).
IDLE_UNLOAD_S = float(os.environ.get("PYSTEREO_SHARP_IDLE_S", "60"))

#: Upper bound for the on-disk Gaussian cache. Oldest entries (by last use)
#: are evicted after each new prediction. Zero disables eviction.
DEFAULT_CACHE_MAX_MB = 2048
_cache_max_bytes: int = int(
    float(os.environ.get("PYSTEREO_SHARP_CACHE_MB", str(DEFAULT_CACHE_MAX_MB))) * 1024 * 1024
)

SHARP_CKPT_FILENAME = "sharp_2572gikvuh.pt"
SHARP_CKPT_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/"
    + SHARP_CKPT_FILENAME
)

_DEFAULT_CACHE = Path(__file__).resolve().parent.parent.parent / ".sharp_cache"
_CACHE_GLOB = "sharp_*.npz"

DEFAULT_INTERNAL = 1536
VIT_TILE = 384          # SHARP's ViT tile size; the sliding split needs 384 + 288 k


def _spn_forward_any_size(self: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
    """``SlidingPyramidNetwork.forward`` for internal sizes other than 1536.

    SHARP's encoder is Depth Pro's sliding pyramid: 384^2 ViT tiles with 25 %
    / 50 % overlap at 1x and 0.5x, plus one global 384^2 ViT at 0.25x. The tile
    split/merge is generic in the number of tiles, but the global ViT has a
    fixed 24x24 token grid, so that branch is kept at 384^2 and its feature
    maps are bilinearly resized to the larger grid. Out of the model's training
    distribution - visibly sharper on test photos but treat as experimental.
    """
    import torch.nn.functional as F

    from sharp.models.encoders import spn_encoder
    from sharp.utils.training import checkpoint_wrapper

    batch_size = x.shape[0]
    size = x.shape[-1]
    x0, x1, x2 = self._create_pyramid(x)
    x2_small = F.interpolate(x2, size=(VIT_TILE, VIT_TILE), mode="bilinear", align_corners=False)
    overlap0, overlap1, padding = (0.25, 0.5, 3) if self.use_patch_overlap else (0.0, 0.0, 0)
    x0_patches = spn_encoder.split(x0, overlap_ratio=overlap0, patch_size=self.patch_size)
    x1_patches = spn_encoder.split(x1, overlap_ratio=overlap1, patch_size=self.patch_size)
    x0_tile_size = x0_patches.shape[0]
    x_pyramid_patches = torch.cat((x0_patches, x1_patches, x2_small), dim=0)
    x_pyramid_encodings, inter = self.patch_encoder(x_pyramid_patches)
    ids = self.patch_intermediate_features_ids
    lat0 = self.patch_encoder.reshape_feature(inter[ids[0]])
    lat1 = self.patch_encoder.reshape_feature(inter[ids[1]])
    x_latent0 = spn_encoder.merge(lat0[: batch_size * x0_tile_size], batch_size=batch_size, padding=padding)
    x_latent1 = spn_encoder.merge(lat1[: batch_size * x0_tile_size], batch_size=batch_size, padding=padding)
    x0_enc, x1_enc, x2_enc = torch.split(
        x_pyramid_encodings, [len(x0_patches), len(x1_patches), len(x2_small)], dim=0,
    )
    x0_features = spn_encoder.merge(x0_enc, batch_size=batch_size, padding=padding)
    x1_features = spn_encoder.merge(x1_enc, batch_size=batch_size, padding=2 * padding)
    x_lowres, _ = self.image_encoder(x2_small)
    grid = size // 64  # 24 at 1536, 42 at 2688
    if grid != x2_enc.shape[-1]:
        x2_enc = F.interpolate(x2_enc, size=(grid, grid), mode="bilinear", align_corners=False)
        x_lowres = F.interpolate(x_lowres, size=(grid, grid), mode="bilinear", align_corners=False)
    cw = checkpoint_wrapper
    x_latent0 = cw(self, self.upsample_latent0, x_latent0)
    x_latent1 = cw(self, self.upsample_latent1, x_latent1)
    x0_features = cw(self, self.upsample0, x0_features)
    x1_features = cw(self, self.upsample1, x1_features)
    x2_features = cw(self, self.upsample2, x2_enc)
    x_lowres = cw(self, self.upsample_lowres, x_lowres)
    x_lowres = cw(self, self.fuse_lowres, torch.cat((x2_features, x_lowres), dim=1))
    return [x_latent0, x_latent1, x0_features, x1_features, x_lowres]


_SPN_PATCH_LOCK = threading.Lock()
_spn_patched = False


def _ensure_spn_any_size_patch() -> None:
    """Install the size-dispatching ``SlidingPyramidNetwork.forward`` once.

    Patching only for a hi-res run and undoing it afterwards would race with a
    concurrent prediction - the predictor is shared and only its *load* is
    locked, inference is not - so the wrapper goes in once and stays: stock
    1536 keeps upstream's forward, any other size takes ours.
    """
    global _spn_patched

    from sharp.models.encoders import spn_encoder

    with _SPN_PATCH_LOCK:
        if _spn_patched:
            return
        original = spn_encoder.SlidingPyramidNetwork.forward

        @functools.wraps(original)
        def forward(self: torch.nn.Module, x: torch.Tensor) -> list[torch.Tensor]:
            if x.shape[-1] == DEFAULT_INTERNAL:
                return original(self, x)
            return _spn_forward_any_size(self, x)

        spn_encoder.SlidingPyramidNetwork.forward = forward  # type: ignore[method-assign]
        _spn_patched = True


# Resident SHARP predictor. The checkpoint takes ~6 s to load and the model
# holds ~3 GB, so it is kept between photos (a batch pays the load once, like
# sharp-local does) and dropped after IDLE_UNLOAD_S without use.
_PREDICTOR_LOCK = threading.Lock()
_predictor: torch.nn.Module | None = None
_predictor_device: str | None = None
_idle_timer: threading.Timer | None = None
_last_used: float = 0.0


def _get_predictor() -> tuple[torch.nn.Module, str, bool]:
    """Return the resident SHARP predictor, loading it on first use.

    The third value is ``True`` when this call paid the checkpoint load.
    """
    from sharp.models import PredictorParams, create_predictor

    global _predictor, _predictor_device, _idle_timer, _last_used
    loaded_now = False
    with _PREDICTOR_LOCK:
        _last_used = time.time()
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if _predictor is None:
            loaded_now = True
            ckpt = _ckpt_path()
            if not ckpt.is_file():
                raise FileNotFoundError(
                    f"SHARP checkpoint not found at {ckpt}. "
                    "Download it via the pystereo UI or run: "
                    f"curl -L -o {ckpt} {SHARP_CKPT_URL}"
                )
            from pystereo_core.registry import detect_device

            device = detect_device()
            t0 = time.time()
            predictor = create_predictor(PredictorParams())
            predictor.load_state_dict(torch.load(ckpt, weights_only=True))
            predictor.eval().to(device)
            _predictor = predictor
            _predictor_device = device
            logger.info(
                "SHARP predictor loaded in %.1fs (device=%s)", time.time() - t0, device,
            )
        assert _predictor_device is not None
        return _predictor, _predictor_device, loaded_now


def _release_device_memory(device: str | None) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def _unload_predictor_if_idle() -> None:
    global _predictor, _predictor_device, _idle_timer
    with _PREDICTOR_LOCK:
        _idle_timer = None
        if _predictor is None or IDLE_UNLOAD_S <= 0:
            return
        # A prediction may have started (and cancelled too late) or finished
        # after this timer was armed - reschedule instead of unloading.
        remaining = _last_used + IDLE_UNLOAD_S - time.time()
        if remaining > 0.5:
            _idle_timer = threading.Timer(remaining, _unload_predictor_if_idle)
            _idle_timer.daemon = True
            _idle_timer.start()
            return
        device = _predictor_device
        _predictor = None
        _predictor_device = None
        logger.info("SHARP predictor idle for %.0fs, unloading", IDLE_UNLOAD_S)
    _release_device_memory(device)


def _schedule_idle_unload() -> None:
    global _idle_timer, _last_used
    with _PREDICTOR_LOCK:
        _last_used = time.time()
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if IDLE_UNLOAD_S <= 0:
            return
        _idle_timer = threading.Timer(IDLE_UNLOAD_S, _unload_predictor_if_idle)
        _idle_timer.daemon = True
        _idle_timer.start()


def get_idle_unload_s() -> float:
    return IDLE_UNLOAD_S


def set_idle_unload_s(seconds: float) -> None:
    """Change the idle timeout at runtime; ``<= 0`` keeps the predictor loaded.

    A loaded predictor gets its timer re-armed from its last use so the new
    value takes effect immediately rather than after the next prediction.
    """
    global IDLE_UNLOAD_S, _idle_timer
    seconds = float(seconds)
    with _PREDICTOR_LOCK:
        if seconds == IDLE_UNLOAD_S:
            return
        IDLE_UNLOAD_S = seconds
        logger.info("SHARP predictor idle timeout set to %.0fs", seconds)
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if _predictor is None or seconds <= 0:
            return
        remaining = max(0.5, _last_used + seconds - time.time())
        _idle_timer = threading.Timer(remaining, _unload_predictor_if_idle)
        _idle_timer.daemon = True
        _idle_timer.start()


def is_predictor_loaded() -> bool:
    with _PREDICTOR_LOCK:
        return _predictor is not None


def unload_predictor() -> bool:
    """Drop the resident predictor now. Returns ``True`` if one was loaded."""
    global _predictor, _predictor_device, _idle_timer
    with _PREDICTOR_LOCK:
        if _idle_timer is not None:
            _idle_timer.cancel()
            _idle_timer = None
        if _predictor is None:
            return False
        device = _predictor_device
        _predictor = None
        _predictor_device = None
        logger.info("SHARP predictor unloaded on request")
    _release_device_memory(device)
    return True


# ---------------------------------------------------------------------------
# On-disk Gaussian cache management
# ---------------------------------------------------------------------------


def cache_dir() -> Path:
    return _DEFAULT_CACHE


def get_cache_max_mb() -> int:
    return _cache_max_bytes // (1024 * 1024)


def set_cache_max_mb(megabytes: int) -> None:
    """Set the eviction bound; ``0`` disables eviction. Prunes immediately."""
    global _cache_max_bytes
    _cache_max_bytes = max(0, int(megabytes)) * 1024 * 1024
    prune_cache()


def _cache_entries(cache: Path) -> list[tuple[Path, int, float]]:
    """``(path, bytes, last_use)`` for every complete cache file."""
    entries: list[tuple[Path, int, float]] = []
    if not cache.is_dir():
        return entries
    for path in cache.glob(_CACHE_GLOB):
        try:
            st = path.stat()
        except OSError:
            continue
        entries.append((path, st.st_size, st.st_mtime))
    return entries


def cache_stats(cache: Path | None = None) -> dict[str, object]:
    entries = _cache_entries(cache or _DEFAULT_CACHE)
    return {
        "path": str(cache or _DEFAULT_CACHE),
        "bytes": sum(size for _, size, _ in entries),
        "files": len(entries),
        "max_mb": get_cache_max_mb(),
    }


def clear_cache(cache: Path | None = None) -> dict[str, int]:
    """Delete every cached prediction. Returns ``{deleted_bytes, deleted_files}``."""
    deleted_bytes = 0
    deleted_files = 0
    for path, size, _ in _cache_entries(cache or _DEFAULT_CACHE):
        try:
            path.unlink()
        except OSError:
            continue
        deleted_bytes += size
        deleted_files += 1
    logger.info("SHARP cache cleared: %d files, %d bytes", deleted_files, deleted_bytes)
    return {"deleted_bytes": deleted_bytes, "deleted_files": deleted_files}


def prune_cache(cache: Path | None = None, keep: Path | None = None) -> int:
    """Evict least recently used entries until the cache fits the size bound.

    ``keep`` is never evicted (the entry the current render still needs).
    Returns the number of files removed.
    """
    if _cache_max_bytes <= 0:
        return 0
    entries = _cache_entries(cache or _DEFAULT_CACHE)
    total = sum(size for _, size, _ in entries)
    if total <= _cache_max_bytes:
        return 0
    removed = 0
    for path, size, _ in sorted(entries, key=lambda e: e[2]):
        if total <= _cache_max_bytes:
            break
        if keep is not None and path == keep:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed += 1
    if removed:
        logger.info(
            "SHARP cache pruned: %d files evicted, %.0f MB kept",
            removed, total / (1024 * 1024),
        )
    return removed


def cache_note(intermediates: dict[str, object] | None) -> str:
    """Suffix for the ``SHARP prediction`` timing label, e.g. `` (cached)``."""
    if not intermediates:
        return ""
    notes: list[str] = []
    state = intermediates.get("sharp_cache")
    if state == "hit":
        notes.append("cached")
    elif state == "off":
        notes.append("cache off")
    if intermediates.get("sharp_model_loaded"):
        notes.append("model load")
    return f" ({', '.join(notes)})" if notes else ""


_REQUIRED_KEYS = frozenset({"means", "scales", "quats", "colors", "opacities", "f_px", "width", "height"})


def _cache_is_complete(path: Path) -> bool:
    try:
        with np.load(path) as d:
            return _REQUIRED_KEYS <= set(d.files)
    except Exception:
        return False


def validate_internal(internal: int) -> int:
    if internal < VIT_TILE or (internal - VIT_TILE) % (VIT_TILE * 3 // 4):
        raise ValueError(f"SHARP internal size must be 384 + 288k (1536, 1824, 2112, 2400, 2688, ...), got {internal}")
    return internal


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
    internal: int = DEFAULT_INTERNAL,
) -> object:
    """Run SHARP inference - inlined from sharp.cli.predict to avoid gsplat import."""
    import torch.nn.functional as F

    from sharp.utils.gaussians import unproject_gaussians

    internal_shape = (internal, internal)
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
    internal: int = DEFAULT_INTERNAL,
    use_cache: bool = True,
    intermediates: dict[str, object] | None = None,
) -> Path:
    """Convert a photo to 3D Gaussians via Apple SHARP.

    ``internal`` is the square size the photo is resized to before the
    network (Gaussian grid = internal / 2 per layer). 1536 is the stock size;
    2688 gives a 1344^2 grid at ~5x the prediction time and 3x the Gaussians.

    Returns the path to the cached ``.npz`` file containing means, scales,
    quaternions, colours, opacities, focal length, and image dimensions.
    Subsequent calls with the same image return instantly from cache unless
    ``use_cache`` is ``False``, which ignores any existing entry and runs the
    network again (the fresh result still overwrites the file, since the
    renderer reads from it). The resident predictor is unaffected either way.

    When ``intermediates`` is given, ``sharp_cache`` is set to ``"hit"``,
    ``"miss"`` or ``"off"`` and ``sharp_model_loaded`` to ``True`` when this
    call paid the checkpoint load - see :func:`cache_note`.
    """
    from PIL import ImageOps

    from pystereo_core.sharp_imports import ensure_sharp_imports

    ensure_sharp_imports()

    validate_internal(internal)

    cache = cache_dir or _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    pil_image = ImageOps.exif_transpose(pil_image)
    img = np.asarray(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    tag = "" if internal == DEFAULT_INTERNAL else f"_{internal}"
    out = cache / f"sharp_{_image_hash(img)}{tag}.npz"
    if intermediates is not None:
        intermediates["sharp_cache"] = "miss" if use_cache else "off"
    if out.exists():
        if not use_cache:
            logger.info("SHARP cache bypassed: %s", out.name)
        elif _cache_is_complete(out):
            logger.info("SHARP cache hit: %s", out.name)
            if intermediates is not None:
                intermediates["sharp_cache"] = "hit"
            try:
                os.utime(out)   # LRU: mark as recently used
            except OSError:
                pass
            return out
        else:
            logger.warning("SHARP cache file %s is incomplete, regenerating", out.name)
            out.unlink()

    torch.manual_seed(0)

    f_35mm = _extract_focal_35mm(pil_image)
    f_px = _focal_length_px(w, h, f_35mm)

    t0 = time.time()
    predictor, device, loaded_now = _get_predictor()
    if loaded_now and intermediates is not None:
        intermediates["sharp_model_loaded"] = True
    logger.info("SHARP predict: %dx%d, f_px=%.1f, internal=%d, device=%s", w, h, f_px, internal, device)

    if internal != DEFAULT_INTERNAL:
        _ensure_spn_any_size_patch()
    g = _predict_image(predictor, img, f_px, torch.device(device), internal=internal)

    # Write to a temp file and rename so an interrupted run (killed request,
    # app restart) never leaves a truncated archive behind as a cache hit.
    tmp = out.with_name(out.stem + ".tmp.npz")   # savez appends .npz unless the name already ends with it
    np.savez_compressed(
        tmp,
        means=g.mean_vectors[0].cpu().numpy(),
        scales=g.singular_values[0].cpu().numpy().astype(np.float16),
        quats=g.quaternions[0].cpu().numpy().astype(np.float16),
        colors=g.colors[0].cpu().numpy().astype(np.float16),
        opacities=g.opacities[0].cpu().numpy().astype(np.float16),
        f_px=np.float32(f_px),
        width=w,
        height=h,
        color_space="linearRGB",
        internal=internal,
    )
    os.replace(tmp, out)
    prune_cache(cache, keep=out)

    _schedule_idle_unload()

    logger.info(
        "SHARP predict done: %d gaussians in %.1fs -> %s",
        g.mean_vectors.shape[1],
        time.time() - t0,
        out.name,
    )
    return out
