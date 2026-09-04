"""Per-step timing collection for the stereo pipeline.

Steps are recorded into the ``intermediates`` dict that already flows
through the pipeline and stereo methods, under the ``"timings"`` key as an
ordered list of ``(label, seconds)`` tuples. UIs (web stages panel, batch
log) read them to show how long each stage of a generation took.
"""

from __future__ import annotations

from typing import Any


def record_step(
    intermediates: dict[str, Any] | None, label: str, seconds: float,
) -> None:
    """Append one step timing; no-op when no intermediates dict is passed."""
    if intermediates is None:
        return
    intermediates.setdefault("timings", []).append((label, round(seconds, 2)))
