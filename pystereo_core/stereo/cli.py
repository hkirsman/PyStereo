"""CLI for offline 2D → SBS stereo conversion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image

from pystereo_core.stereo.config import InpaintBackendName, StereoSettings
from pystereo_core.stereo.methods import available_methods
from pystereo_core.stereo.pipeline import StereoPipeline

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a 2D image to a side-by-side (SBS) stereo image.",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=Path,
        help="Input RGB image path",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output SBS path (default: <input>_sbs.png)",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="Optional precomputed depth PNG (255=close, 0=far). Skips depth model.",
    )
    parser.add_argument(
        "--divergence",
        type=float,
        default=3.0,
        help="Max binocular separation as %% of image width (default: 3.0)",
    )
    parser.add_argument(
        "--inpaint",
        choices=("lama", "opencv", "none", "aotgan", "flux"),
        default="lama",
        help="Occlusion inpainting backend (default: lama)",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=2048,
        help="Longest side for warp/inpaint processing (default: 2048)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=sorted(available_methods()),
        help="Stereo method (default: per_eye_inpaint)",
    )
    parser.add_argument(
        "--no-heal",
        action="store_true",
        default=False,
        help="Disable BiRefNet depth healing (skip composite depth recovery)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for depth model: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _resolve_device(name: str) -> str:
    if name != "auto":
        return name
    from pystereo_core.registry import detect_device

    device = detect_device()
    return "cpu" if device == "none" else device


def _run_synthesis(
    pipeline: StereoPipeline,
    rgb: Image.Image,
    args: argparse.Namespace,
) -> Image.Image:
    method = getattr(args, "method", None)

    if method:
        # Through the pipeline, so a method running a nested pass (sharp_depth)
        # reuses the models this pipeline already holds. argparse validated the
        # name, so a failure here is a real error - wrapping it in RuntimeError
        # sent it down main()'s LaMa retry path.
        stereo_method = pipeline._get_method(method)
        if not stereo_method.needs_depth:
            return pipeline._synthesize_no_depth(rgb, stereo_method, method)

    if args.depth is not None:
        depth_path = args.depth.expanduser().resolve()
        if not depth_path.is_file():
            raise FileNotFoundError(f"Depth map not found: {depth_path}")
        with Image.open(depth_path) as dimg:
            depth = dimg.convert("L")
        logger.info("Using depth map %s", depth_path)
        return pipeline.synthesize(rgb, depth, method=method)

    from pystereo_core.depth import DepthEstimator

    device = _resolve_device(args.device)
    estimator = DepthEstimator()
    logger.info("Loading depth model on %s...", device)
    estimator.load(device)
    try:
        return pipeline.synthesize_with_depth_estimator(rgb, estimator, method=method)
    finally:
        estimator.unload()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    input_path: Path = args.input.expanduser().resolve()
    if not input_path.is_file():
        logger.error("Input not found: %s", input_path)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_sbs.png")
    else:
        output_path = output_path.expanduser().resolve()

    settings = StereoSettings(
        divergence_ratio=args.divergence / 100.0,
        max_processing_dim=max(512, args.max_dim),
        inpaint_backend=args.inpaint,  # type: ignore[arg-type]
        depth_healing=not args.no_heal,
    )

    with Image.open(input_path) as img:
        rgb = img.convert("RGB")

    pipeline = StereoPipeline(settings=settings)
    try:
        sbs = _run_synthesis(pipeline, rgb, args)
    except RuntimeError as exc:
        if settings.inpaint_backend != "lama":
            logger.error("Stereo synthesis failed: %s", exc)
            return 1
        from dataclasses import replace

        logger.warning("LaMa unavailable (%s); retrying with opencv inpaint", exc)
        pipeline = StereoPipeline(settings=replace(settings, inpaint_backend="opencv"))
        try:
            sbs = _run_synthesis(pipeline, rgb, args)
        except (OSError, RuntimeError, FileNotFoundError) as retry_exc:
            logger.error("Stereo synthesis failed: %s", retry_exc)
            return 1
    except (OSError, FileNotFoundError) as exc:
        logger.error("Stereo synthesis failed: %s", exc)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sbs.save(output_path, format="PNG")
    logger.info("Wrote %s (%dx%d)", output_path, sbs.width, sbs.height)
    return 0


if __name__ == "__main__":
    sys.exit(main())
