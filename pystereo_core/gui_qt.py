"""Qt (PySide6) UI for PyStereo - stereo batch + server mode."""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}

_SETTINGS_KEYS = frozenset({
    "depth_model", "max_dim", "method",
})


def _settings_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        if sys.platform == "darwin":
            return Path.home() / "Library" / "Application Support" / "PyStereo" / "settings.json"
        local = os.environ.get("LOCALAPPDATA") if sys.platform == "win32" else None
        if local:
            return Path(local) / "PyStereo" / "settings.json"
        return root / "settings.json"
    return root / "settings.json"


def _load_settings() -> dict[str, Any]:
    p = _settings_path()
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if k in _SETTINGS_KEYS}
    except Exception:
        return {}


def _save_settings(data: dict[str, Any]) -> None:
    merged = _load_settings()
    merged.update({k: v for k, v in data.items() if k in _SETTINGS_KEYS})
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(merged, indent=2), encoding="utf-8")


class _Bridge(QObject):
    """Cross-thread signals (Qt delivers slots on the GUI thread)."""

    job_finished = Signal(object)
    server_log = Signal(str)
    model_status = Signal(object)


def _scan_images(root: Path, recursive: bool) -> list[Path]:
    images: list[Path] = []
    for ext in IMAGE_EXTS:
        if recursive:
            images.extend(root.rglob(f"*{ext}"))
            images.extend(root.rglob(f"*{ext.upper()}"))
        else:
            images.extend(root.glob(f"*{ext}"))
            images.extend(root.glob(f"*{ext.upper()}"))
    return sorted(set(images))


class PyStereoQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        from pystereo_core._version import __version__

        self.setWindowTitle(f"PyStereo {__version__}")
        self.resize(720, 580)

        self._job_q: queue.Queue[Path] = queue.Queue()
        self._quit_app = threading.Event()
        self._batch_total = 0
        self._batch_done = 0
        self._batch_running = False
        self._model_state = "idle"

        self._bridge = _Bridge()
        self._bridge.job_finished.connect(self._on_job_done, Qt.QueuedConnection)
        self._bridge.server_log.connect(self._on_server_log, Qt.QueuedConnection)
        self._bridge.model_status.connect(self._on_model_status, Qt.QueuedConnection)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Model section
        model_box = QGroupBox("Stereo models")
        model_layout = QVBoxLayout(model_box)
        model_layout.setContentsMargins(8, 10, 8, 8)
        self._model_status_label = QLabel("Checking stereo models...")
        self._model_status_label.setWordWrap(True)
        model_layout.addWidget(self._model_status_label)
        self._model_progress = QProgressBar()
        self._model_progress.setRange(0, 100)
        self._model_progress.setValue(0)
        self._model_progress.setVisible(False)
        model_layout.addWidget(self._model_progress)
        model_btns = QHBoxLayout()
        self._model_download_btn = QPushButton("Download")
        self._model_download_btn.setToolTip(
            "Download stereo model pack (Depth Anything V2, BiRefNet, LaMa, AOT-GAN)."
        )
        self._model_download_btn.clicked.connect(self._on_model_download)
        model_btns.addWidget(self._model_download_btn)
        self._model_cancel_btn = QPushButton("Cancel")
        self._model_cancel_btn.setVisible(False)
        model_btns.addWidget(self._model_cancel_btn)
        self._model_remove_btn = QPushButton("Remove")
        self._model_remove_btn.clicked.connect(self._on_model_remove)
        model_btns.addWidget(self._model_remove_btn)
        model_btns.addStretch()
        model_layout.addLayout(model_btns)
        layout.addWidget(model_box)

        # Folder row
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Folder"))
        self._folder_edit = QLineEdit()
        row1.addWidget(self._folder_edit, stretch=1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row1.addWidget(browse)
        layout.addLayout(row1)

        # Options row
        row2 = QHBoxLayout()
        self._recursive_chk = QCheckBox("Include subfolders")
        self._recursive_chk.setChecked(True)
        row2.addWidget(self._recursive_chk)
        row2.addWidget(QLabel("Depth model"))
        self._depth_model_combo = QComboBox()
        self._depth_model_combo.setMinimumWidth(140)
        self._depth_model_combo.addItem("Small (~95 MB)", "small")
        self._depth_model_combo.addItem("Base (~400 MB)", "base")
        self._depth_model_combo.addItem("Large (~1.3 GB)", "large")
        row2.addWidget(self._depth_model_combo)
        row2.addWidget(QLabel("Method"))
        self._method_combo = QComboBox()
        self._method_combo.setMinimumWidth(160)
        row2.addWidget(self._method_combo)
        row2.addWidget(QLabel("Max dim"))
        self._max_dim_spin = QSpinBox()
        self._max_dim_spin.setRange(512, 3072)
        self._max_dim_spin.setSingleStep(128)
        self._max_dim_spin.setValue(2048)
        row2.addWidget(self._max_dim_spin)
        row2.addStretch()
        layout.addLayout(row2)

        # Output row
        row3 = QHBoxLayout()
        self._output_chk = QCheckBox("Output to folder")
        self._output_chk.toggled.connect(self._sync_output_widgets)
        row3.addWidget(self._output_chk)
        self._output_edit = QLineEdit()
        self._output_edit.setEnabled(False)
        row3.addWidget(self._output_edit, stretch=1)
        self._output_browse_btn = QPushButton("Browse...")
        self._output_browse_btn.clicked.connect(self._browse_output)
        self._output_browse_btn.setEnabled(False)
        row3.addWidget(self._output_browse_btn)
        layout.addLayout(row3)

        # Action row
        row4 = QHBoxLayout()
        self._scan_btn = QPushButton("Start batch")
        self._scan_btn.clicked.connect(self._on_scan)
        row4.addWidget(self._scan_btn)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self._on_stop)
        row4.addWidget(stop_btn)
        layout.addLayout(row4)

        # Server row
        srv_row = QHBoxLayout()
        self._srv_btn = QPushButton("Start server")
        self._srv_btn.setToolTip(
            "Start a local HTTP server. Clients can POST images to /transform "
            "and receive SBS stereo JPEGs."
        )
        self._srv_btn.clicked.connect(self._on_toggle_server)
        srv_row.addWidget(self._srv_btn)
        self._srv_label = QLabel("Server stopped")
        self._srv_label.setStyleSheet("color: #666;")
        srv_row.addWidget(self._srv_label, stretch=1)
        layout.addLayout(srv_row)
        self._srv_thread: threading.Thread | None = None
        self._srv_running = False

        # Progress
        layout.addWidget(QLabel("Progress"))
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        # Log
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(140)
        layout.addWidget(self._log, stretch=1)

        self._populate_methods()
        self._load_saved_settings()

        self._depth_model_combo.currentIndexChanged.connect(self._on_setting_changed)
        self._method_combo.currentIndexChanged.connect(self._on_setting_changed)
        self._max_dim_spin.valueChanged.connect(self._on_setting_changed)
        self._model_timer = QTimer(self)
        self._model_timer.timeout.connect(self._poll_model)
        self._model_timer.start(1500)
        self._poll_model()

    def _load_saved_settings(self) -> None:
        s = _load_settings()
        if "depth_model" in s:
            idx = self._depth_model_combo.findData(s["depth_model"])
            if idx >= 0:
                self._depth_model_combo.setCurrentIndex(idx)
        if "max_dim" in s:
            try:
                self._max_dim_spin.setValue(int(s["max_dim"]))
            except (ValueError, TypeError):
                pass
        if "method" in s:
            idx = self._method_combo.findData(s["method"])
            if idx >= 0:
                self._method_combo.setCurrentIndex(idx)

    def _on_setting_changed(self) -> None:
        _save_settings({
            "depth_model": self._depth_model_combo.currentData() or "small",
            "max_dim": self._max_dim_spin.value(),
            "method": self._method_combo.currentData() or "",
        })

    def _populate_methods(self) -> None:
        try:
            from pystereo_core.stereo.methods import available_methods

            methods = available_methods()
            self._method_combo.clear()
            for name in sorted(methods.keys()):
                cls = methods[name]
                label = getattr(cls, "label", name)
                self._method_combo.addItem(label, name)
            default_idx = self._method_combo.findData("per_eye_inpaint")
            if default_idx >= 0:
                self._method_combo.setCurrentIndex(default_idx)
        except Exception:
            self._method_combo.addItem("per_eye_inpaint", "per_eye_inpaint")

    def _log_line(self, msg: str) -> None:
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        if sb:
            sb.setValue(sb.maximum())

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder")
        if path:
            self._folder_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if path:
            self._output_edit.setText(path)

    def _sync_output_widgets(self, checked: bool = False) -> None:
        self._output_edit.setEnabled(self._output_chk.isChecked())
        self._output_browse_btn.setEnabled(self._output_chk.isChecked())

    def _poll_model(self) -> None:
        try:
            from pystereo_core.download import get_download_manager

            mgr = get_download_manager()
            status = mgr.status_dict()
            self._bridge.model_status.emit(status)
        except Exception:
            pass

    def _on_model_status(self, status: dict) -> None:
        state = status.get("state", "idle")
        self._model_state = state
        pct = int(status.get("percent") or 0)
        msg = str(status.get("message") or "")

        if state == "ready":
            self._model_status_label.setText("Stereo model pack ready.")
            self._model_progress.setVisible(False)
            self._model_download_btn.setEnabled(False)
            self._model_download_btn.setText("Downloaded")
            self._model_remove_btn.setEnabled(True)
            self._model_cancel_btn.setVisible(False)
            self._scan_btn.setEnabled(True)
        elif state == "downloading":
            self._model_status_label.setText(msg or "Downloading...")
            self._model_progress.setVisible(True)
            self._model_progress.setValue(pct)
            self._model_download_btn.setEnabled(False)
            self._model_cancel_btn.setVisible(False)
            self._model_remove_btn.setEnabled(False)
            self._scan_btn.setEnabled(False)
        elif state == "error":
            err = status.get("error") or "Unknown error"
            self._model_status_label.setText(f"Download failed: {err}")
            self._model_progress.setVisible(False)
            self._model_download_btn.setEnabled(True)
            self._model_download_btn.setText("Retry")
            self._model_cancel_btn.setVisible(False)
            self._model_remove_btn.setEnabled(True)
        else:
            self._model_status_label.setText("Stereo models not downloaded.")
            self._model_progress.setVisible(False)
            self._model_download_btn.setEnabled(True)
            self._model_download_btn.setText("Download")
            self._model_cancel_btn.setVisible(False)
            self._model_remove_btn.setEnabled(False)
            self._scan_btn.setEnabled(False)

    def _on_model_download(self) -> None:
        from pystereo_core.download import get_download_manager

        mgr = get_download_manager()
        mgr.ensure_stereo_pack_async()
        self._log_line("Starting model download...")

    def _on_model_remove(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove models",
            "Delete all cached stereo model weights?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        from pystereo_core.download import get_download_manager

        mgr = get_download_manager()
        try:
            result = mgr.delete_stereo_pack()
            freed = int(result.get("deleted_bytes") or 0)
            self._log_line(f"Removed stereo model pack ({freed / 1024 / 1024:.0f} MB).")
        except RuntimeError as exc:
            self._log_line(f"Cannot remove: {exc}")

    def _on_scan(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder:
            QMessageBox.warning(self, "No folder", "Please select a folder first.")
            return
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            QMessageBox.warning(self, "Invalid folder", f"Not a directory: {root}")
            return

        recursive = self._recursive_chk.isChecked()
        images = _scan_images(root, recursive)
        if not images:
            self._log_line("No supported images found.")
            return

        self._batch_total = len(images)
        self._batch_done = 0
        self._batch_running = True
        self._progress.setValue(0)
        self._log_line(f"Starting batch: {self._batch_total} image(s)...")

        method = self._method_combo.currentData() or "per_eye_inpaint"
        max_dim = self._max_dim_spin.value()
        depth_model_size = self._depth_model_combo.currentData() or "small"
        output_dir = None
        if self._output_chk.isChecked():
            od = self._output_edit.text().strip()
            if od:
                output_dir = Path(od).expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)

        t = threading.Thread(
            target=self._batch_worker,
            args=(images, method, max_dim, depth_model_size, output_dir),
            daemon=True,
        )
        t.start()

    def _batch_worker(
        self,
        images: list[Path],
        method: str,
        max_dim: int,
        depth_model_size: str,
        output_dir: Path | None,
    ) -> None:
        import numpy as np
        from PIL import Image

        from pystereo_core.depth import DepthEstimator
        from pystereo_core.registry import get_registry
        from pystereo_core.stereo.config import StereoSettings
        from pystereo_core.stereo.pipeline import StereoPipeline

        registry = get_registry()
        registry.detect_gpu()
        registry.replace(DepthEstimator(model_size=depth_model_size))
        registry.enabled = True

        from dataclasses import replace as dc_replace

        settings = StereoSettings.from_env(method=method)
        settings = dc_replace(
            settings,
            max_processing_dim=max_dim,
        )
        pipeline = StereoPipeline(settings=settings)

        depth_model = registry.get("depth")
        if depth_model is None:
            self._bridge.server_log.emit("Depth model not available.")
            return

        for img_path in images:
            if not self._batch_running:
                self._bridge.server_log.emit("Batch stopped.")
                break
            try:
                t0 = time.perf_counter()
                rgb = Image.open(img_path).convert("RGB")
                if hasattr(depth_model, "process_raw"):
                    depth_f32 = depth_model.process_raw(rgb)
                else:
                    depth_pil = depth_model.process(rgb)
                    depth_f32 = np.array(depth_pil.convert("L"), dtype=np.float32) / 255.0

                sbs_img = pipeline.synthesize(rgb, depth_f32, method=method)

                if output_dir:
                    out_path = output_dir / f"{img_path.stem}_sbs.jpg"
                else:
                    out_path = img_path.parent / f"{img_path.stem}_sbs.jpg"
                sbs_img.save(str(out_path), format="JPEG", quality=95, subsampling=0)
                elapsed = time.perf_counter() - t0
                self._bridge.job_finished.emit(
                    {"ok": True, "path": str(img_path), "elapsed": elapsed}
                )
            except Exception as exc:
                self._bridge.job_finished.emit(
                    {"ok": False, "path": str(img_path), "error": str(exc)}
                )

    def _on_job_done(self, result: dict) -> None:
        self._batch_done += 1
        pct = int(100 * self._batch_done / max(1, self._batch_total))
        self._progress.setValue(pct)
        path = Path(result.get("path", "")).name
        if result.get("ok"):
            elapsed = result.get("elapsed", 0)
            self._log_line(f"  ok: {path} ({elapsed:.1f}s)")
        else:
            err = result.get("error", "unknown")
            self._log_line(f"  err: {path} - {err}")
        if self._batch_done >= self._batch_total:
            self._log_line(f"Batch complete: {self._batch_done}/{self._batch_total}")
            self._batch_running = False

    def _on_stop(self) -> None:
        self._batch_running = False

    def _on_toggle_server(self) -> None:
        if self._srv_running:
            self._srv_label.setText("Server stopped (restart app to fully stop)")
            self._srv_label.setStyleSheet("color: #666;")
            self._srv_running = False
            self._srv_btn.setText("Start server")
            return

        self._srv_running = True
        self._srv_btn.setText("Stop server")
        self._srv_label.setText("Server running at http://127.0.0.1:8766")
        self._srv_label.setStyleSheet("color: #4a4;")
        self._log_line("Starting server on http://127.0.0.1:8766 ...")

        def _run_server() -> None:
            from app import app, force_service_enabled, set_gui_log_sink

            set_gui_log_sink(lambda msg: self._bridge.server_log.emit(msg))
            # The user pressed "Start server": that is the opt-in.
            force_service_enabled(True)

            import logging as _logging

            _logging.getLogger("werkzeug").setLevel(_logging.ERROR)
            try:
                import flask.cli

                flask.cli.show_server_banner = lambda *a, **kw: None  # type: ignore[method-assign]
            except Exception:
                pass
            app.run(host="127.0.0.1", port=8766, debug=False, threaded=True)

        self._srv_thread = threading.Thread(target=_run_server, daemon=True)
        self._srv_thread.start()

    def _on_server_log(self, msg: str) -> None:
        self._log_line(msg)

    def closeEvent(self, event: object) -> None:
        self._quit_app.set()
        self._batch_running = False
        super().closeEvent(event)  # type: ignore[arg-type]


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PyStereo")
    window = PyStereoQtWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
