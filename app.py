"""PyStereo web UI: browser + Flask for 2D-to-SBS stereo conversion.

Inference runs locally via PyTorch (CPU / MPS / CUDA).

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python app.py
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _dev_root() -> Path:
    return Path(__file__).resolve().parent


def _bundle_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return _dev_root()


def _outputs_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        if sys.platform == "darwin":
            return (
                Path.home()
                / "Library"
                / "Application Support"
                / "PyStereo"
                / "outputs"
            )
        if sys.platform == "win32":
            local = os.environ.get("LOCALAPPDATA")
            if local:
                return Path(local) / "PyStereo" / "outputs"
            return Path.home() / "AppData" / "Local" / "PyStereo" / "outputs"
        xdg = os.environ.get("XDG_DATA_HOME")
        if xdg:
            return Path(xdg) / "pystereo" / "outputs"
        return Path.home() / ".local" / "share" / "pystereo" / "outputs"
    return _dev_root() / "outputs"


STATIC_DIR = _bundle_root() / "static"
OUTPUTS_DIR = _outputs_dir()
PREDICT_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _configure_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.propagate = False
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        log.addHandler(handler)
    return log


LOGGER = _configure_logger("pystereo-web")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

_gui_log_sink: Callable[[str], None] | None = None


def set_gui_log_sink(sink: Callable[[str], None] | None) -> None:
    """When set, gui_log lines appear in a batch-app log window."""
    global _gui_log_sink
    _gui_log_sink = sink


def gui_log(message: str) -> None:
    """User-facing line for the batch GUI log (not stderr / werkzeug)."""
    if _gui_log_sink is not None:
        try:
            _gui_log_sink(message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# AI model setup (lazy)
# ---------------------------------------------------------------------------

_pipeline_singleton: Any = None
_pipeline_lock = threading.Lock()


def _get_depth_estimator() -> Any:
    """Return the depth estimator from the registry, or None."""
    from pystereo_core.registry import get_registry

    registry = get_registry()
    return registry.get("depth")


def _ensure_registry() -> None:
    """Register the depth estimator if not already done."""
    from pystereo_core.depth import DepthEstimator
    from pystereo_core.registry import get_registry

    registry = get_registry()
    if not registry.has_capability("depth"):
        registry.detect_gpu()
        registry.register(DepthEstimator())


def _get_pipeline() -> Any:
    """Return (or create) the stereo pipeline singleton."""
    global _pipeline_singleton
    with _pipeline_lock:
        if _pipeline_singleton is None:
            from pystereo_core.stereo.config import StereoSettings
            from pystereo_core.stereo.pipeline import StereoPipeline

            settings = StereoSettings.from_env()
            _pipeline_singleton = StereoPipeline(settings=settings)
        return _pipeline_singleton


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------


def _suppress_flask_startup_noise() -> None:
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    try:
        import flask.cli

        flask.cli.show_server_banner = lambda *args, **kwargs: None  # type: ignore[method-assign]
    except Exception:
        pass


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024


@app.errorhandler(413)
def request_entity_too_large(_e: Exception) -> Any:
    return jsonify({"error": "File too large"}), 413


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_id_ok(result_id: str) -> bool:
    try:
        uuid.UUID(result_id)
        return True
    except ValueError:
        return False


def _model_status_payload() -> dict[str, Any]:
    from pystereo_core.download import get_download_manager

    return get_download_manager().status_dict()


def _model_not_ready_response() -> tuple[Any, int]:
    status = _model_status_payload()
    code = (
        "model_downloading"
        if status.get("state") == "downloading"
        else "model_not_ready"
    )
    return jsonify({"error": code, "model": status}), 503


def _require_model_ready() -> Optional[tuple[Any, int]]:
    from pystereo_core.download import get_download_manager

    if get_download_manager().is_pack_ready():
        return None
    return _model_not_ready_response()


# ---------------------------------------------------------------------------
# Routes - static
# ---------------------------------------------------------------------------


@app.route("/favicon.ico")
def favicon() -> Any:
    path = STATIC_DIR / "favicon.ico"
    if path.is_file():
        return send_file(path, mimetype="image/x-icon")
    return "", 204


@app.route("/")
def index() -> Any:
    return send_from_directory(str(STATIC_DIR), "index.html")


# ---------------------------------------------------------------------------
# Routes - health
# ---------------------------------------------------------------------------


@app.route("/health")
def health() -> Any:
    """Media service readiness probe."""
    from pystereo_core._version import __version__ as VERSION

    status = _model_status_payload()
    return jsonify({
        "ok": True,
        "kind": "stereo",
        "name": "pystereo",
        "version": VERSION,
        "model_ready": status.get("state") == "ready",
        "model_state": status.get("state"),
    })


@app.route("/api/health")
def api_health() -> Any:
    from pystereo_core._version import __version__ as VERSION
    from pystereo_core.registry import get_registry

    registry = get_registry()
    status = _model_status_payload()
    return jsonify({
        "ok": True,
        "app": "pystereo-web",
        "version": VERSION,
        "model_ready": status.get("state") == "ready",
        "model_state": status.get("state"),
        "device": registry.device,
        "model": status,
    })


# ---------------------------------------------------------------------------
# Routes - model management
# ---------------------------------------------------------------------------


@app.route("/api/model/status", methods=["GET"])
def api_model_status() -> Any:
    return jsonify(_model_status_payload())


@app.route("/api/model/download", methods=["POST"])
def api_model_download() -> Any:
    """Start a background download of the stereo model pack."""
    from pystereo_core.download import get_download_manager

    _ensure_registry()
    mgr = get_download_manager()
    mgr.ensure_stereo_pack_async()
    return jsonify(mgr.status_dict())


@app.route("/api/model/cancel", methods=["POST"])
def api_model_cancel() -> Any:
    """Request cancellation of an in-progress download.

    The download thread runs to completion per-artifact, so this sets the
    pack state to idle and the UI will stop showing progress.  A full
    interrupt would require cooperative cancellation in the download thread.
    """
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    mgr.refresh_local_state()
    return jsonify(mgr.status_dict())


@app.route("/api/model/delete", methods=["POST"])
def api_model_delete() -> Any:
    """Delete cached model weights to free disk space."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    try:
        result = mgr.delete_stereo_pack()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    status = mgr.status_dict()
    status["deleted_bytes"] = result.get("deleted_bytes", 0)
    status["deleted_paths"] = result.get("deleted_paths", [])
    return jsonify(status)


# ---------------------------------------------------------------------------
# Routes - stereo methods
# ---------------------------------------------------------------------------


@app.route("/api/stereo-methods", methods=["GET"])
def api_stereo_methods() -> Any:
    from pystereo_core.stereo.methods import available_methods

    methods = sorted(available_methods().keys())
    return jsonify(methods)


# ---------------------------------------------------------------------------
# Routes - transform (media service endpoint)
# ---------------------------------------------------------------------------


@app.route("/transform", methods=["POST"])
def transform() -> Any:
    """Media service endpoint: JPEG in, SBS JPEG bytes out."""
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    name = Path(upload.filename).name
    blocked = _require_model_ready()
    if blocked is not None:
        gui_log("  Failed - stereo model not ready (download required)")
        return blocked

    method_override = (request.form.get("method") or "").strip().lower() or None
    max_dim_raw = (request.form.get("max_dim") or "").strip()

    _ensure_registry()
    gui_log(f"Generating stereo SBS for {name}...")

    t0 = time.perf_counter()
    try:
        with PREDICT_LOCK:
            rgb_pil = Image.open(upload.stream).convert("RGB")
            w, h = rgb_pil.size
            gui_log(f"  Running depth + stereo ({w}x{h})...")

            depth_estimator = _get_depth_estimator()
            if depth_estimator is None:
                gui_log("  Failed - depth model not loaded")
                return jsonify({"error": "Depth model not loaded"}), 503

            pipeline = _get_pipeline()

            if max_dim_raw:
                try:
                    max_dim = max(512, int(max_dim_raw))
                    from dataclasses import replace as dc_replace

                    from pystereo_core.stereo.config import StereoSettings
                    from pystereo_core.stereo.pipeline import StereoPipeline

                    settings = dc_replace(pipeline.settings, max_processing_dim=max_dim)
                    local_pipeline = StereoPipeline(settings=settings)
                    sbs_img = local_pipeline.synthesize_with_depth_estimator(
                        rgb_pil, depth_estimator, method=method_override,
                    )
                except ValueError:
                    sbs_img = pipeline.synthesize_with_depth_estimator(
                        rgb_pil, depth_estimator, method=method_override,
                    )
            else:
                sbs_img = pipeline.synthesize_with_depth_estimator(
                    rgb_pil, depth_estimator, method=method_override,
                )
    except Exception:
        LOGGER.exception("Inference failed for %s", name)
        gui_log(f"  Failed - {name} (see terminal for details)")
        return jsonify({"error": "Inference failed"}), 500

    elapsed = time.perf_counter() - t0
    gui_log(f"  Done - {name} ({elapsed:.1f}s)")

    buf = io.BytesIO()
    sbs_img.save(buf, format="JPEG", quality=95)
    payload = buf.getvalue()

    return Response(
        payload,
        mimetype="image/jpeg",
        headers={
            "Content-Disposition": 'attachment; filename="stereo.jpg"',
            "Content-Length": str(len(payload)),
        },
    )


# ---------------------------------------------------------------------------
# Routes - generate (web UI endpoint)
# ---------------------------------------------------------------------------


@app.route("/api/generate", methods=["POST"])
def api_generate() -> Any:
    """Web UI endpoint: generate SBS + depth, return JSON with result URLs."""
    if "file" not in request.files:
        return jsonify({"error": "Missing file field"}), 400
    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "Empty filename"}), 400

    blocked = _require_model_ready()
    if blocked is not None:
        return blocked

    method_override = (request.form.get("method") or "").strip().lower() or None
    max_dim_raw = (request.form.get("max_dim") or "").strip()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())
    result_dir = OUTPUTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(upload.filename).suffix.lower() or ".jpg"
    input_path = result_dir / f"input{ext}"
    upload.save(str(input_path))

    _ensure_registry()

    t0 = time.perf_counter()
    try:
        with PREDICT_LOCK:
            rgb_pil = Image.open(input_path).convert("RGB")
            w, h = rgb_pil.size

            depth_estimator = _get_depth_estimator()
            if depth_estimator is None:
                shutil.rmtree(result_dir, ignore_errors=True)
                return jsonify({"error": "Depth model not loaded"}), 503

            # Generate depth map
            if hasattr(depth_estimator, "process_raw"):
                import numpy as np

                depth_f32 = depth_estimator.process_raw(rgb_pil)
                depth_u8 = (depth_f32 * 255).clip(0, 255).astype(np.uint8)
                depth_pil = Image.fromarray(depth_u8, mode="L")
            else:
                depth_pil = depth_estimator.process(rgb_pil)

            depth_pil.save(str(result_dir / "depth.png"))

            # Generate SBS
            pipeline = _get_pipeline()
            if max_dim_raw:
                try:
                    max_dim = max(512, int(max_dim_raw))
                    from dataclasses import replace as dc_replace

                    from pystereo_core.stereo.config import StereoSettings
                    from pystereo_core.stereo.pipeline import StereoPipeline

                    settings = dc_replace(pipeline.settings, max_processing_dim=max_dim)
                    local_pipeline = StereoPipeline(settings=settings)
                    sbs_img = local_pipeline.synthesize_with_depth_estimator(
                        rgb_pil, depth_estimator, method=method_override,
                    )
                except ValueError:
                    sbs_img = pipeline.synthesize_with_depth_estimator(
                        rgb_pil, depth_estimator, method=method_override,
                    )
            else:
                sbs_img = pipeline.synthesize_with_depth_estimator(
                    rgb_pil, depth_estimator, method=method_override,
                )

            sbs_img.save(str(result_dir / "sbs.jpg"), quality=95)

    except Exception:
        LOGGER.exception("Inference failed for %s", upload.filename)
        shutil.rmtree(result_dir, ignore_errors=True)
        return jsonify({"error": "Inference failed; check server logs."}), 500

    elapsed = round(time.perf_counter() - t0, 3)

    meta: dict[str, Any] = {
        "id": result_id,
        "original_name": upload.filename,
        "created": datetime.now(timezone.utc).isoformat(),
        "width": w,
        "height": h,
        "method": method_override or pipeline.settings.stereo_method,
        "elapsed_seconds": elapsed,
    }
    (result_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return jsonify({
        "id": result_id,
        "sbs_url": f"/api/results/{result_id}/sbs.jpg",
        "depth_url": f"/api/results/{result_id}/depth.png",
        "input_url": f"/api/results/{result_id}/input{ext}",
        "method": meta["method"],
        "width": w,
        "height": h,
        "elapsed_seconds": elapsed,
    })


# ---------------------------------------------------------------------------
# Routes - serve results
# ---------------------------------------------------------------------------

_RESULT_MIMES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@app.route("/api/results/<result_id>/<filename>", methods=["GET"])
def get_result_file(result_id: str, filename: str) -> Any:
    if not _result_id_ok(result_id):
        return jsonify({"error": "Invalid result id"}), 400
    safe_name = Path(filename).name
    if safe_name != filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400
    file_path = OUTPUTS_DIR / result_id / safe_name
    if not file_path.is_file():
        return jsonify({"error": "Not found"}), 404
    ext = Path(safe_name).suffix.lower()
    mime = _RESULT_MIMES.get(ext, "application/octet-stream")
    return send_file(file_path, mimetype=mime)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyStereo web server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Port (default: 8766)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_registry()

    LOGGER.info("PyStereo web - http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
