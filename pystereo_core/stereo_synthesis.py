"""AI stereo synthesis — re-exports the modular stereo pipeline."""

from pystereo_core.stereo.pipeline import StereoPipeline, synthesize_sbs
from pystereo_core.stereo.config import StereoSettings

__all__ = ["StereoPipeline", "StereoSettings", "synthesize_sbs"]
