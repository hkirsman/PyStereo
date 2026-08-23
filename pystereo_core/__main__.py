"""Run ``python -m pystereo_core`` (GUI default) or ``--cli`` for headless."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _format_model_bytes(n: int) -> str:
    if n <= 0:
        return "-"
    mb = n / (1024 * 1024)
    if mb >= 1024:
        gb = mb / 1024
        return f"{round(gb) if gb >= 10 else f'{gb:.1f}'} GB"
    if mb < 100:
        return f"{mb:.1f} MB"
    return f"{round(mb)} MB"


def _ensure_model_ready_for_cli() -> int:
    """Download the stereo model pack synchronously if missing. Returns exit code."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    if mgr.is_pack_ready():
        status = mgr.status_dict()
        total = int(status.get("bytes_total") or 0)
        print(
            f"Stereo model pack ready ({_format_model_bytes(total)}).",
            flush=True,
        )
        return 0

    print("Stereo model pack missing - downloading...", flush=True)
    last_pct = -1

    def _progress(done: int, total: int, percent: int) -> None:
        nonlocal last_pct
        if percent == last_pct and percent not in (0, 100):
            return
        last_pct = percent
        if total > 0:
            print(
                f"  {percent}% - {_format_model_bytes(done)} / {_format_model_bytes(total)}",
                flush=True,
            )
        else:
            print(f"  {_format_model_bytes(done)}...", flush=True)

    try:
        mgr.ensure_stereo_pack_async()
        while not mgr.is_pack_ready():
            status = mgr.status_dict()
            if status.get("state") == "error":
                print(f"Download failed: {status.get('error')}", file=sys.stderr)
                return 1
            done = int(status.get("bytes_downloaded") or 0)
            total = int(status.get("bytes_total") or 0)
            pct = int(status.get("percent") or 0)
            _progress(done, total, pct)
            time.sleep(1.0)
    except Exception as exc:
        print(f"Stereo model download failed: {exc}", file=sys.stderr)
        return 1
    status = mgr.status_dict()
    total = int(status.get("bytes_total") or 0)
    print(f"Stereo model pack ready ({_format_model_bytes(total)}).", flush=True)
    return 0


def _download_model_main() -> int:
    return _ensure_model_ready_for_cli()


def _remove_model_main() -> int:
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    try:
        result = mgr.delete_stereo_pack()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    freed = int(result.get("deleted_bytes") or 0)
    if freed > 0:
        print(f"Removed stereo model pack ({_format_model_bytes(freed)}).")
    else:
        print("No stereo model pack found to remove.")
    return 0


def _cli_main() -> int:
    from pystereo_core._version import __version__

    p = argparse.ArgumentParser(
        description="PyStereo batch: generate SBS stereo images from 2D photos.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    p.add_argument(
        "--download-model",
        action="store_true",
        help="Download the stereo model pack and exit",
    )
    p.add_argument(
        "--remove-model",
        action="store_true",
        help="Delete the cached stereo model pack and exit",
    )
    p.add_argument(
        "--folder",
        type=Path,
        required=False,
        help="Root path to scan (a directory of images)",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Include subfolders",
    )
    p.add_argument(
        "--method",
        type=str,
        default=None,
        help="Stereo method to use (default: per_eye_inpaint)",
    )
    p.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Max processing dimension in pixels (default: 2048)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for SBS images (default: next to each image)",
    )
    p.add_argument(
        "--inpaint",
        type=str,
        choices=("lama", "opencv", "none", "aotgan", "flux"),
        default=None,
        help="Occlusion inpainting backend (default: lama)",
    )
    p.add_argument(
        "--depth-model",
        type=str,
        default="small",
        choices=["small", "base", "large"],
        help="Depth Anything V2 model size (default: small)",
    )
    args = p.parse_args()

    if args.download_model and args.remove_model:
        print(
            "Use only one of --download-model or --remove-model.",
            file=sys.stderr,
        )
        return 1
    if args.download_model:
        return _download_model_main()
    if args.remove_model:
        return _remove_model_main()

    if args.folder is None:
        p.error("--folder is required unless --download-model or --remove-model")

    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 1

    model_rc = _ensure_model_ready_for_cli()
    if model_rc != 0:
        return model_rc

    from pystereo_core.depth import DepthEstimator
    from pystereo_core.registry import get_registry
    from pystereo_core.stereo.config import StereoSettings
    from pystereo_core.stereo.pipeline import StereoPipeline

    registry = get_registry()
    registry.detect_gpu()
    registry.register(DepthEstimator(model_size=args.depth_model))
    registry.enabled = True

    from dataclasses import replace as dc_replace

    settings = StereoSettings.from_env(method=args.method)
    overrides: dict = {"max_processing_dim": args.max_dim}
    if args.inpaint is not None:
        overrides["inpaint_backend"] = args.inpaint
    settings = dc_replace(settings, **overrides)
    pipeline = StereoPipeline(settings=settings)

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}
    images = []
    if args.recursive:
        for ext in IMAGE_EXTS:
            images.extend(root.rglob(f"*{ext}"))
            images.extend(root.rglob(f"*{ext.upper()}"))
    else:
        for ext in IMAGE_EXTS:
            images.extend(root.glob(f"*{ext}"))
            images.extend(root.glob(f"*{ext.upper()}"))

    images = sorted(set(images))
    if not images:
        print("No supported images found.", file=sys.stderr)
        return 0

    output_dir = args.output_dir
    if output_dir:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image
    import numpy as np

    depth_model = registry.get("depth")
    if depth_model is None:
        print("Depth model not available.", file=sys.stderr)
        return 1

    n = len(images)
    exit_code = 0
    batch_t0 = time.perf_counter()
    for i, img_path in enumerate(images, start=1):
        print(f"[{i}/{n}] {img_path}", flush=True)
        try:
            rgb = Image.open(img_path).convert("RGB")
            if hasattr(depth_model, "process_raw"):
                depth_f32 = depth_model.process_raw(rgb)
            else:
                depth_pil = depth_model.process(rgb)
                depth_f32 = np.array(depth_pil.convert("L"), dtype=np.float32) / 255.0

            sbs_img = pipeline.synthesize(rgb, depth_f32, method=args.method)

            if output_dir:
                out_path = output_dir / f"{img_path.stem}_sbs.jpg"
            else:
                out_path = img_path.parent / f"{img_path.stem}_sbs.jpg"

            sbs_img.save(str(out_path), format="JPEG", quality=95, subsampling=0)
            elapsed = time.perf_counter() - batch_t0
            print(f"    ok: {out_path.name} ({elapsed:.1f}s total)", flush=True)
        except Exception as exc:
            print(f"    err: {exc}", flush=True)
            exit_code = 1

    batch_elapsed = time.perf_counter() - batch_t0
    print(f"Batch finished: {n} image(s) in {batch_elapsed:.1f}s.", flush=True)
    return exit_code


_GUI_HELP = """\
No batch GUI available.

Fix (pick one):
  - pip install PySide6-Essentials
    Then: python -m pystereo_core
    (Qt UI; smaller than full PySide6; no system Tcl/Tk.)
  - Headless: python -m pystereo_core --cli --folder /path/to/photos --recursive
"""


def _is_expected_missing_qt(exc: ImportError) -> bool:
    """True only when PySide6 (or a PySide6.* submodule) is absent."""
    if isinstance(exc, ModuleNotFoundError):
        name = (exc.name or "").lower()
        if name == "pyside6" or name.startswith("pyside6."):
            return True
    return "pyside6" in str(exc).lower()


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--version", "-V"):
        from pystereo_core._version import __version__

        print(f"pystereo {__version__}")
        raise SystemExit(0)
    if len(sys.argv) >= 2 and sys.argv[1] in ("--download-model", "--remove-model"):
        if sys.argv[1] == "--download-model":
            raise SystemExit(_download_model_main())
        raise SystemExit(_remove_model_main())
    if len(sys.argv) >= 2 and sys.argv[1] == "--cli":
        sys.argv.pop(1)
        raise SystemExit(_cli_main())
    try:
        from pystereo_core.gui_qt import main as gui_main

        gui_main()
        return
    except ImportError as e:
        if not _is_expected_missing_qt(e):
            raise
    print(_GUI_HELP, file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
