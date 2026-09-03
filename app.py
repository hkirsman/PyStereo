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

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from PIL import Image

from pystereo_core.logging_config import ensure_stderr_info_logging

ensure_stderr_info_logging(log_file_name="pystereo-web.log")

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


def _settings_path() -> Path:
    """Persistent user settings file, next to outputs."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return OUTPUTS_DIR.parent / "settings.json"
    return _dev_root() / "settings.json"


_USER_SETTINGS_KEYS = frozenset({
    "depth_model", "max_dim", "method",
    # Cache & memory (see the "Cache & memory" panel in the web UI)
    "disable_cache", "sharp_cache_max_mb", "outputs_keep", "sharp_idle_s",
})

#: Result folders kept under ``outputs/`` (oldest are pruned after each run).
#: Only the run on screen is ever read back; older folders are just history
#: on disk. TODO: add a way to browse past results in the web UI, then a
#: larger default would earn its space.
DEFAULT_OUTPUTS_KEEP = 10


def _setting_int(saved: dict[str, Any], key: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(saved.get(key, default)))
    except (TypeError, ValueError):
        return default


def _setting_bool(saved: dict[str, Any], key: str, default: bool = False) -> bool:
    val = saved.get(key, default)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _form_flag(name: str) -> bool:
    return (request.form.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


def _load_user_settings() -> dict[str, Any]:
    p = _settings_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if k in _USER_SETTINGS_KEYS}
    except Exception:
        return {}


def _save_user_settings(data: dict[str, Any]) -> None:
    filtered = {k: v for k, v in data.items() if k in _USER_SETTINGS_KEYS}
    merged = _load_user_settings()
    merged.update(filtered)
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")


_cache_settings_applied = False


def _apply_cache_settings(saved: dict[str, Any]) -> None:
    """Push persisted cache / memory settings into the SHARP module.

    Safe to call before ml-sharp is importable: the module only needs torch.
    """
    global _cache_settings_applied
    try:
        from pystereo_core.stereo import sharp_predict
    except Exception:
        return
    _cache_settings_applied = True
    if "sharp_idle_s" in saved:
        try:
            sharp_predict.set_idle_unload_s(float(saved["sharp_idle_s"]))
        except (TypeError, ValueError):
            pass
    if "sharp_cache_max_mb" in saved:
        sharp_predict.set_cache_max_mb(
            _setting_int(saved, "sharp_cache_max_mb", sharp_predict.DEFAULT_CACHE_MAX_MB),
        )

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
WEB_LOG_PATH: Path | None = None
try:
    from pystereo_core.logging_config import attach_file_handler, web_log_path

    WEB_LOG_PATH = attach_file_handler(LOGGER, "pystereo-web.log") or web_log_path()
except Exception:
    WEB_LOG_PATH = None
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


def _inference_failed_payload() -> dict[str, str]:
    """JSON body for failed inference; points at the on-disk log when available."""
    if WEB_LOG_PATH is not None:
        return {
            "error": f"Inference failed. Check logs at {WEB_LOG_PATH}",
            "log_path": str(WEB_LOG_PATH),
            "log_url": "/api/logs",
        }
    return {"error": "Inference failed; check server logs."}


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


def _ensure_registry(model_size: str = "small") -> None:
    """Register the depth estimator if not already done, or switch model size."""
    from pystereo_core.depth import DepthEstimator
    from pystereo_core.registry import get_registry

    registry = get_registry()
    registry.detect_gpu()
    if not registry.has_capability("depth"):
        registry.register(DepthEstimator(model_size=model_size))
        return

    existing = registry._models.get("depth")
    if existing is not None and hasattr(existing, "model_size") and existing.model_size == model_size:
        return

    if existing is not None and existing.is_loaded():
        existing.unload()
    registry.register(DepthEstimator(model_size=model_size))


def _get_pipeline() -> Any:
    """Return (or create) the stereo pipeline singleton."""
    global _pipeline_singleton
    with _pipeline_lock:
        if _pipeline_singleton is None:
            from dataclasses import replace as dc_replace

            from pystereo_core.stereo.config import StereoSettings
            from pystereo_core.stereo.pipeline import StereoPipeline

            saved = _load_user_settings()
            saved_method = (saved.get("method") or "").strip().lower() or None
            settings = StereoSettings.from_env(method=saved_method)
            overrides: dict[str, Any] = {}
            if "max_dim" in saved:
                try:
                    overrides["max_processing_dim"] = max(512, int(saved["max_dim"]))
                except (ValueError, TypeError):
                    pass
            if _setting_bool(saved, "disable_cache"):
                overrides["sharp_disk_cache"] = False
            if overrides:
                settings = dc_replace(settings, **overrides)
            _pipeline_singleton = StereoPipeline(settings=settings)
            _apply_cache_settings(saved)
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


@app.errorhandler(Exception)
def _log_unhandled_exception(exc: Exception) -> tuple[Any, int]:
    if isinstance(exc, HTTPException):
        return exc
    LOGGER.exception("Unhandled request error")
    return jsonify({"error": str(exc)}), 500


@app.after_request
def _no_cache_dynamic(response: Response) -> Response:
    if (
        request.path == "/"
        or request.path.startswith("/api/")
        or request.path.endswith((".js", ".css", ".html"))
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


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


def _attach_sharp_status(payload: dict[str, Any]) -> dict[str, Any]:
    """Add SHARP checkpoint fields expected by the web UI."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    ready = mgr.is_sharp_model_local()
    payload["sharp_ready"] = ready
    sharp_art = mgr._artifacts.get("sharp")
    downloading = (
        not ready
        and sharp_art is not None
        and sharp_art.state in ("queued", "downloading")
    )
    payload["sharp_downloading"] = downloading
    payload["sharp_queued"] = downloading and sharp_art.state == "queued"
    if downloading and sharp_art is not None:
        payload["sharp_percent"] = int(sharp_art.percent or 0)
        payload["sharp_bytes_downloaded"] = int(sharp_art.bytes_downloaded or 0)
        payload["sharp_bytes_total"] = int(sharp_art.bytes_total or 0)
    else:
        payload["sharp_percent"] = 100 if ready else 0
        payload["sharp_bytes_downloaded"] = 0
        payload["sharp_bytes_total"] = 0
    return payload


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
    from pystereo_core._version import __version__ as VERSION

    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    html = html.replace("{{version}}", VERSION)
    return Response(html, mimetype="text/html")


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


@app.route("/api/logs", methods=["GET"])
def download_logs() -> Any:
    """Download the PyStereo web log file (for packaged apps with no console)."""
    path = WEB_LOG_PATH
    if path is None or not path.is_file():
        return jsonify({"error": "Log file not available"}), 404
    for handler in LOGGER.handlers:
        try:
            handler.flush()
        except Exception:
            pass
    return send_file(
        path,
        mimetype="text/plain; charset=utf-8",
        as_attachment=True,
        download_name="pystereo-web.log",
    )


@app.route("/api/model/status", methods=["GET"])
def api_model_status() -> Any:
    return jsonify(_attach_sharp_status(_model_status_payload()))


@app.route("/api/model/download-sharp", methods=["POST"])
def api_model_download_sharp() -> Any:
    """Start a background download of the SHARP checkpoint."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    mgr.ensure_sharp_model_async()
    return jsonify(_attach_sharp_status(mgr.status_dict()))


@app.route("/api/model/cancel-sharp", methods=["POST"])
def api_model_cancel_sharp() -> Any:
    """Cancel an in-progress SHARP checkpoint download."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    result = mgr.cancel_sharp_download()
    payload = _attach_sharp_status(mgr.status_dict())
    payload["cancelled"] = result.get("cancelled", False)
    return jsonify(payload)


@app.route("/api/model/delete-sharp", methods=["POST"])
def api_model_delete_sharp() -> Any:
    """Delete the cached SHARP checkpoint to free disk space."""
    from pystereo_core.download import get_download_manager

    mgr = get_download_manager()
    try:
        result = mgr.delete_sharp_model()
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 409
    payload = _attach_sharp_status(mgr.status_dict())
    payload["deleted_bytes"] = result.get("deleted_bytes", 0)
    payload["path"] = result.get("path")
    return jsonify(payload)


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
# Routes - depth models
# ---------------------------------------------------------------------------


@app.route("/api/depth-models", methods=["GET"])
def api_depth_models() -> Any:
    """Return available depth models and their download status."""
    try:
        from pystereo_core.depth import DEPTH_MODELS
        from pystereo_core.download import get_download_manager

        mgr = get_download_manager()
        result = []
        for size, info in DEPTH_MODELS.items():
            result.append({
                "size": size,
                "name": info["name"],
                "license": info["license"],
                "size_mb": info["size_mb"],
                "downloaded": mgr.is_depth_model_local(size),
            })
        return jsonify(result)
    except Exception as exc:
        LOGGER.exception("Failed to list depth models")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/depth-models/download", methods=["POST"])
def api_depth_model_download() -> Any:
    """Download a specific depth model (small/base/large)."""
    from pystereo_core.download import get_download_manager

    model_size = (request.json or {}).get("model", "small") if request.is_json else request.form.get("model", "small")
    model_size = model_size.strip().lower()
    if model_size not in ("small", "base", "large"):
        return jsonify({"error": f"Invalid model: {model_size}"}), 400

    mgr = get_download_manager()
    if mgr.is_depth_model_local(model_size):
        return jsonify({"status": "ready", "model": model_size})

    started = mgr.ensure_depth_model_async(model_size)
    if not started:
        return jsonify({"error": "Failed to start download"}), 500

    return jsonify({"status": "downloading", "model": model_size})


# ---------------------------------------------------------------------------
# Routes - persistent settings
# ---------------------------------------------------------------------------


@app.route("/api/settings", methods=["GET"])
def api_settings_get() -> Any:
    return jsonify(_load_user_settings())


@app.route("/api/settings", methods=["PUT"])
def api_settings_put() -> Any:
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "JSON body required"}), 400

    _save_user_settings(data)

    saved = _load_user_settings()

    global _pipeline_singleton
    with _pipeline_lock:
        if _pipeline_singleton is not None:
            from dataclasses import replace as dc_replace

            overrides: dict[str, Any] = {}
            if "method" in saved:
                method_raw = (str(saved["method"]) or "").strip().lower()
                from pystereo_core.stereo.methods import available_methods
                if method_raw in available_methods():
                    overrides["stereo_method"] = method_raw
            if "max_dim" in saved:
                try:
                    overrides["max_processing_dim"] = max(512, int(saved["max_dim"]))
                except (ValueError, TypeError):
                    pass
            if "disable_cache" in saved:
                overrides["sharp_disk_cache"] = not _setting_bool(saved, "disable_cache")
            if overrides:
                new_settings = dc_replace(
                    _pipeline_singleton.settings, **overrides,
                )
                if "stereo_method" in overrides:
                    new_settings = new_settings.with_method_defaults()
                _pipeline_singleton.settings = new_settings

    _apply_cache_settings(saved)

    if "depth_model" in saved:
        _ensure_registry(model_size=str(saved["depth_model"]))

    return jsonify(saved)


# ---------------------------------------------------------------------------
# Routes - cache & memory
# ---------------------------------------------------------------------------


def _result_dirs() -> list[tuple[Path, int, float]]:
    """``(dir, bytes, mtime)`` for each result folder under ``outputs/``."""
    entries: list[tuple[Path, int, float]] = []
    if not OUTPUTS_DIR.is_dir():
        return entries
    for child in OUTPUTS_DIR.iterdir():
        if not child.is_dir() or not _result_id_ok(child.name):
            continue
        size = 0
        try:
            mtime = child.stat().st_mtime
            for f in child.iterdir():
                if f.is_file():
                    size += f.stat().st_size
        except OSError:
            continue
        entries.append((child, size, mtime))
    return entries


def _outputs_stats() -> dict[str, Any]:
    entries = _result_dirs()
    return {
        "path": str(OUTPUTS_DIR),
        "bytes": sum(size for _, size, _ in entries),
        "files": len(entries),
    }


def _prune_outputs(keep: int, protect: str | None = None) -> int:
    """Delete the oldest result folders beyond ``keep``. ``0`` keeps all."""
    if keep <= 0:
        return 0
    entries = sorted(_result_dirs(), key=lambda e: e[2], reverse=True)
    removed = 0
    for path, _, _ in entries[keep:]:
        if protect is not None and path.name == protect:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    if removed:
        LOGGER.info("Pruned %d old result folder(s), keeping %d", removed, keep)
    return removed


def _clear_outputs() -> dict[str, int]:
    deleted_bytes = 0
    deleted_files = 0
    for path, size, _ in _result_dirs():
        shutil.rmtree(path, ignore_errors=True)
        deleted_bytes += size
        deleted_files += 1
    LOGGER.info("Results cleared: %d folder(s), %d bytes", deleted_files, deleted_bytes)
    return {"deleted_bytes": deleted_bytes, "deleted_files": deleted_files}


def _loaded_models() -> list[str]:
    """Names of model weights currently resident in memory."""
    names: list[str] = []
    try:
        from pystereo_core.stereo import sharp_predict

        if sharp_predict.is_predictor_loaded():
            names.append("SHARP predictor")
    except Exception:
        pass
    try:
        from pystereo_core.registry import get_registry

        registry = get_registry()
        for capability, model in registry._models.items():
            if model.is_loaded():
                names.append(f"{capability} ({model.name})")
    except Exception:
        pass
    with _pipeline_lock:
        pipe = _pipeline_singleton
    if pipe is not None:
        if getattr(getattr(pipe, "_segmenter", None), "_model", None) is not None:
            names.append("segmenter (BiRefNet)")
        try:
            if pipe._inpainter.is_loaded():
                names.append(f"inpainter ({pipe.settings.inpaint_backend})")
        except Exception:
            pass
    return names


def _cache_payload() -> dict[str, Any]:
    sharp: dict[str, Any]
    idle_s: float | None = None
    if not _cache_settings_applied:
        # First look at the panel before any generation: make the reported
        # idle timeout / cache limit match settings.json, not the env defaults.
        _apply_cache_settings(_load_user_settings())
    try:
        from pystereo_core.stereo import sharp_predict

        sharp = sharp_predict.cache_stats()
        idle_s = sharp_predict.get_idle_unload_s()
    except Exception as exc:
        sharp = {"path": None, "bytes": 0, "files": 0, "error": str(exc)}
    saved = _load_user_settings()
    return {
        "sharp": sharp,
        "outputs": _outputs_stats(),
        "outputs_keep": _setting_int(saved, "outputs_keep", DEFAULT_OUTPUTS_KEEP),
        "sharp_idle_s": idle_s,
        "loaded_models": _loaded_models(),
    }


@app.route("/api/cache", methods=["GET"])
def api_cache_get() -> Any:
    return jsonify(_cache_payload())


@app.route("/api/cache/clear", methods=["POST"])
def api_cache_clear() -> Any:
    """Delete on-disk caches. JSON body: ``{"target": "sharp" | "outputs" | "all"}``."""
    body = request.get_json(silent=True) or {}
    target = str(body.get("target") or "all").strip().lower()
    if target not in ("sharp", "outputs", "all"):
        return jsonify({"error": f"Unknown target: {target}"}), 400
    if not PREDICT_LOCK.acquire(blocking=False):
        return jsonify({"error": "A generation is running; try again when it finishes."}), 409
    try:
        deleted: dict[str, Any] = {}
        if target in ("sharp", "all"):
            from pystereo_core.stereo import sharp_predict

            deleted["sharp"] = sharp_predict.clear_cache()
        if target in ("outputs", "all"):
            deleted["outputs"] = _clear_outputs()
    finally:
        PREDICT_LOCK.release()
    payload = _cache_payload()
    payload["deleted"] = deleted
    return jsonify(payload)


@app.route("/api/models/unload", methods=["POST"])
def api_models_unload() -> Any:
    """Free every resident model (SHARP, depth, segmenter, inpainter) now."""
    if not PREDICT_LOCK.acquire(blocking=False):
        return jsonify({"error": "A generation is running; try again when it finishes."}), 409
    try:
        released: list[str] = []
        try:
            from pystereo_core.stereo import sharp_predict

            if sharp_predict.unload_predictor():
                released.append("SHARP predictor")
        except Exception:
            pass
        try:
            from pystereo_core.registry import get_registry

            registry = get_registry()
            loaded = [m.name for m in registry._models.values() if m.is_loaded()]
            registry.unload_all()
            released.extend(loaded)
        except Exception:
            pass
        with _pipeline_lock:
            pipe = _pipeline_singleton
        if pipe is not None:
            released.extend(pipe.unload_models())
    finally:
        PREDICT_LOCK.release()
    payload = _cache_payload()
    payload["released"] = released
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Routes - stereo methods
# ---------------------------------------------------------------------------


@app.route("/api/stereo-methods", methods=["GET"])
def api_stereo_methods() -> Any:
    try:
        from pystereo_core.sharp_imports import is_sharp_code_available
        from pystereo_core.stereo.config import DEFAULT_METHOD
        from pystereo_core.stereo.methods import list_methods_for_ui

        sharp_code = is_sharp_code_available()
        taichi_render_available = False
        try:
            from pystereo_core.stereo.taichi_render import is_taichi_available

            taichi_render_available = is_taichi_available()
        except Exception:
            pass
        result = []
        for name, cls in list_methods_for_ui():
            if not cls.needs_depth and not sharp_code:
                continue
            entry: dict[str, Any] = {
                "name": name,
                "label": cls.label,
                "needs_depth": cls.needs_depth,
                "deprecated": cls.deprecated,
                "default": name == DEFAULT_METHOD,
                "ui_info": cls.ui_info,
                "uses_taichi": bool(getattr(cls, "uses_taichi", False)),
            }
            result.append(entry)
        return jsonify({
            "methods": result,
            "taichi_render_available": taichi_render_available,
        })
    except Exception as exc:
        LOGGER.exception("Failed to list stereo methods")
        return jsonify({"error": str(exc)}), 500


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

    method_override = (request.form.get("method") or "").strip().lower() or None
    max_dim_raw = (request.form.get("max_dim") or "").strip()
    depth_model = (request.form.get("depth_model") or "small").strip().lower()

    # Output budget, applied after synthesis. Unlike max_dim (a processing
    # dimension) this caps the delivered SBS, so the render keeps its
    # supersampling and the caller does not receive a 50 MP JPEG.
    max_pixels = 0
    max_pixels_raw = (request.form.get("max_pixels") or "").strip()
    if max_pixels_raw:
        try:
            max_pixels = max(0, int(max_pixels_raw))
        except ValueError:
            LOGGER.warning("Ignoring invalid max_pixels %r", max_pixels_raw)

    effective_method = method_override or _get_pipeline().settings.stereo_method
    method_needs_depth = True
    try:
        from pystereo_core.stereo.methods import get_method
        method_needs_depth = get_method(effective_method).needs_depth
    except ValueError:
        pass

    if method_needs_depth:
        blocked = _require_model_ready()
        if blocked is not None:
            gui_log("  Failed - stereo model not ready (download required)")
            return blocked
        _ensure_registry(model_size=depth_model)
    gui_log(f"Generating stereo SBS for {name}...")

    t0 = time.perf_counter()
    try:
        with PREDICT_LOCK:
            rgb_pil = Image.open(upload.stream).convert("RGB")
            w, h = rgb_pil.size
            gui_log(f"  Running depth + stereo ({w}x{h})...")

            depth_estimator = _get_depth_estimator()
            if depth_estimator is None and method_needs_depth:
                gui_log("  Failed - depth model not loaded")
                return jsonify({"error": "Depth model not loaded"}), 503

            pipeline = _get_pipeline()

            overrides: dict[str, Any] = {}
            if max_dim_raw:
                try:
                    overrides["max_processing_dim"] = max(512, int(max_dim_raw))
                except ValueError:
                    pass
            if _form_flag("no_cache"):
                overrides["sharp_disk_cache"] = False

            if overrides:
                local_pipeline = pipeline.derive(**overrides)
                sbs_img = local_pipeline.synthesize_with_depth_estimator(
                    rgb_pil, depth_estimator, method=method_override,
                )
            else:
                sbs_img = pipeline.synthesize_with_depth_estimator(
                    rgb_pil, depth_estimator, method=method_override,
                )
    except Exception:
        LOGGER.exception("Inference failed for %s", name)
        gui_log(f"  Failed - {name} (see terminal for details)")
        return jsonify(_inference_failed_payload()), 500

    if max_pixels:
        from pystereo_core.stereo.fit import fit_to_pixel_budget

        before_w, before_h = sbs_img.size
        sbs_img = fit_to_pixel_budget(sbs_img, max_pixels, even_width=True)
        if sbs_img.size != (before_w, before_h):
            msg = (
                f"Fitted SBS {before_w}x{before_h} ({before_w * before_h / 1e6:.1f} MP)"
                f" -> {sbs_img.size[0]}x{sbs_img.size[1]}"
                f" ({sbs_img.size[0] * sbs_img.size[1] / 1e6:.1f} MP,"
                f" budget {max_pixels / 1e6:.1f} MP)"
            )
            LOGGER.info(msg)
            gui_log(f"  {msg}")

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

    method_override = (request.form.get("method") or "").strip().lower() or None
    max_dim_raw = (request.form.get("max_dim") or "").strip()
    depth_model = (request.form.get("depth_model") or "small").strip().lower()
    disable_cache = _form_flag("disable_cache")

    effective_method = method_override or _get_pipeline().settings.stereo_method
    gen_needs_depth = True
    try:
        from pystereo_core.stereo.methods import get_method as _get_stereo_method

        method_obj = _get_stereo_method(effective_method)
        gen_needs_depth = method_obj.needs_depth
        if not gen_needs_depth:
            from pystereo_core.sharp_imports import is_sharp_code_available

            if not is_sharp_code_available():
                return jsonify({
                    "error": (
                        "This build does not include ml-sharp. "
                        "Choose a depth-based method or rebuild with "
                        "git submodule update --init ml-sharp."
                    ),
                }), 503
    except ValueError:
        pass

    if gen_needs_depth:
        blocked = _require_model_ready()
        if blocked is not None:
            return blocked
        _ensure_registry(model_size=depth_model)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())
    result_dir = OUTPUTS_DIR / result_id
    result_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(upload.filename).suffix.lower() or ".jpg"
    input_path = result_dir / f"input{ext}"
    upload.save(str(input_path))

    t0 = time.perf_counter()
    try:
        with PREDICT_LOCK:
            rgb_pil = Image.open(input_path).convert("RGB")
            w, h = rgb_pil.size

            # Resolve pipeline with per-request overrides
            pipeline = _get_pipeline()
            overrides: dict[str, Any] = {}
            if max_dim_raw:
                try:
                    overrides["max_processing_dim"] = max(512, int(max_dim_raw))
                except ValueError:
                    pass
            # The checkbox is the source of truth for this request; the
            # persisted setting only sets the pipeline default for /transform.
            if pipeline.settings.sharp_disk_cache == disable_cache:
                overrides["sharp_disk_cache"] = not disable_cache
            if overrides:
                active_pipeline = pipeline.derive(**overrides)
            else:
                active_pipeline = pipeline

            from pystereo_core.stereo.timing import record_step

            warp_result = None
            sharp_intermediates: dict[str, Any] = {}
            if not gen_needs_depth:
                sbs_img = active_pipeline.synthesize_with_depth_estimator(
                    rgb_pil, None, method=method_override,
                    intermediates=sharp_intermediates,
                )
            else:
                depth_estimator = _get_depth_estimator()
                if depth_estimator is None:
                    shutil.rmtree(result_dir, ignore_errors=True)
                    return jsonify({"error": "Depth model not loaded"}), 503

                # Generate depth map
                import numpy as np

                t_step = time.perf_counter()
                if hasattr(depth_estimator, "process_raw"):
                    depth_f32 = depth_estimator.process_raw(rgb_pil)
                    depth_u8 = (depth_f32 * 255).clip(0, 255).astype(np.uint8)
                    depth_pil = Image.fromarray(depth_u8, mode="L")
                else:
                    depth_pil = depth_estimator.process(rgb_pil)
                    depth_f32 = np.array(
                        depth_pil.convert("L"), dtype=np.float32,
                    ) / 255.0
                record_step(
                    sharp_intermediates, "Depth estimation",
                    time.perf_counter() - t_step,
                )

                depth_pil.save(str(result_dir / "depth.png"))

                # Warp preview (pre-inpaint intermediates)
                t_step = time.perf_counter()
                seg_loaded = active_pipeline.segmenter_loaded()
                warp_result = active_pipeline.warp_preview(
                    rgb_pil, depth_f32, method=method_override,
                )
                if warp_result is not None:
                    warp_result.warp_sbs.save(
                        str(result_dir / "warp.jpg"), quality=95,
                    )
                    warp_result.mask_sbs.save(str(result_dir / "mask.png"))
                    load_note = (
                        " (model load)"
                        if active_pipeline.segmenter_loaded() and not seg_loaded
                        else ""
                    )
                    record_step(
                        sharp_intermediates, "Warp preview" + load_note,
                        time.perf_counter() - t_step,
                    )

                # Generate SBS
                sbs_img = active_pipeline.synthesize(
                    rgb_pil, depth_f32, method=method_override,
                    intermediates=sharp_intermediates,
                )
            sbs_img.save(str(result_dir / "sbs.jpg"), quality=95)

            if sharp_intermediates.get("splat_rgb") is not None:
                Image.fromarray(sharp_intermediates["splat_rgb"]).save(
                    str(result_dir / "splat.jpg"), quality=95,
                )
            if sharp_intermediates.get("depth01") is not None:
                import numpy as np

                d01 = sharp_intermediates["depth01"]
                d_u8 = (d01 * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(d_u8, mode="L").save(str(result_dir / "depth.png"))

    except FileNotFoundError as exc:
        LOGGER.exception("Model not found for %s", upload.filename)
        shutil.rmtree(result_dir, ignore_errors=True)
        return jsonify({"error": str(exc)}), 503
    except Exception:
        LOGGER.exception("Inference failed for %s", upload.filename)
        shutil.rmtree(result_dir, ignore_errors=True)
        return jsonify(_inference_failed_payload()), 500

    elapsed = round(time.perf_counter() - t0, 3)
    step_timings = [
        {"label": label, "seconds": seconds}
        for label, seconds in sharp_intermediates.get("timings", [])
    ]
    render_backend = sharp_intermediates.get("render_backend")
    sharp_cache = sharp_intermediates.get("sharp_cache")

    meta: dict[str, Any] = {
        "id": result_id,
        "original_name": upload.filename,
        "created": datetime.now(timezone.utc).isoformat(),
        "width": w,
        "height": h,
        "method": effective_method,
        "elapsed_seconds": elapsed,
        "timings": step_timings,
        "render_backend": render_backend,
        "sharp_cache": sharp_cache,
        "disable_cache": disable_cache,
    }
    (result_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _prune_outputs(
        _setting_int(_load_user_settings(), "outputs_keep", DEFAULT_OUTPUTS_KEEP),
        protect=result_id,
    )

    resp: dict[str, Any] = {
        "id": result_id,
        "sbs_url": f"/api/results/{result_id}/sbs.jpg",
        "input_url": f"/api/results/{result_id}/input{ext}",
        "method": meta["method"],
        "width": w,
        "height": h,
        "elapsed_seconds": elapsed,
        "timings": step_timings,
        "render_backend": render_backend,
        "sharp_cache": sharp_cache,
        "disable_cache": disable_cache,
    }
    if gen_needs_depth:
        resp["depth_url"] = f"/api/results/{result_id}/depth.png"
        if warp_result is not None:
            resp["warp_url"] = f"/api/results/{result_id}/warp.jpg"
            resp["mask_url"] = f"/api/results/{result_id}/mask.png"
    else:
        if sharp_intermediates.get("splat_rgb") is not None:
            resp["splat_url"] = f"/api/results/{result_id}/splat.jpg"
        if sharp_intermediates.get("depth01") is not None:
            resp["depth_url"] = f"/api/results/{result_id}/depth.png"
    return jsonify(resp)


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
