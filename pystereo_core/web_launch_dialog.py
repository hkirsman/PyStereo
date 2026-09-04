"""Small launch window for PyStereo web: show URL and open the browser."""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import urllib.error
import urllib.request
import webbrowser
from typing import Callable

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


def browser_url(host: str, port: int) -> str:
    """URL for a local browser (map bind-all hosts to loopback)."""
    if host in ("0.0.0.0", "::", "[::]"):
        return f"http://127.0.0.1:{port}"
    return f"http://{host}:{port}"


def _set_probe_reuse(sock: socket.socket) -> None:
    """Make a probe bind answer "is anyone on this port?" on this platform.

    POSIX: SO_REUSEADDR, so a port left in TIME_WAIT by a just-quit server
    reads as free - werkzeug sets the same option and would bind it fine.
    Windows: SO_REUSEADDR means the opposite there - it lets the probe bind
    on top of a live listener and report a busy port free - so ask for
    SO_EXCLUSIVEADDRUSE instead, which fails if anything else owns the
    address.
    """
    if sys.platform == "win32":
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        return
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)


def port_is_free(host: str, port: int) -> bool:
    """Return True when *host*/*port* can be bound for the Flask server.

    Probes the address the server will actually bind, picked the way
    werkzeug picks it: IPv6 when *host* looks like an IPv6 address, then
    the first ``getaddrinfo`` result for that family. A bind-all host stays
    bind-all here - probing loopback instead would miss a conflict on
    another interface, and would fail outright on an IPv6-only host.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        infos = socket.getaddrinfo(
            host.strip("[]") or None,
            port,
            family,
            socket.SOCK_STREAM,
            flags=socket.AI_PASSIVE,
        )
    except socket.gaierror:
        return True  # unresolvable host - let the server report the real error
    family, socktype, proto, _canonname, addr = infos[0]
    with socket.socket(family, socktype, proto) as sock:
        _set_probe_reuse(sock)
        try:
            sock.bind(addr)
        except OSError:
            return False
    return True


def _port_conflict_help(port: int) -> tuple[str, str]:
    """Where to find the other instance, and how to kill it, on this platform."""
    if sys.platform == "win32":
        return (
            "Task Manager",
            "From Command Prompt:\n"
            f"  netstat -ano | findstr :{port}\n"
            "  taskkill /PID <pid> /F",
        )
    if sys.platform == "darwin":
        return (
            "Dock icon or Activity Monitor",
            f"From Terminal:\n  lsof -ti :{port} | xargs kill",
        )
    return (
        "your system monitor",
        f"From a terminal:\n  lsof -ti :{port} | xargs kill",
    )


def ensure_port_available(host: str, port: int, *, gui: bool = True) -> None:
    """Exit with a clear message when another instance already owns the port."""
    if port_is_free(host, port):
        return
    where, how = _port_conflict_help(port)
    message = (
        f"Port {port} is already in use.\n\n"
        "Quit the other PyStereo Web instance first "
        f"({where}), then try again.\n\n"
        f"{how}"
    )
    logger.error("Cannot start PyStereo Web: port %d already in use", port)
    if gui:
        qt_app = QApplication.instance() or QApplication([])
        QMessageBox.critical(None, "PyStereo Web", message)
    raise SystemExit(message)


class WebLaunchWindow(QWidget):
    def __init__(
        self,
        url: str,
        version: str,
        on_quit: Callable[[], None],
        *,
        log_url: str | None = None,
        server_error: Callable[[], str | None] | None = None,
    ) -> None:
        super().__init__()
        self._url = url
        self._log_url = log_url
        self._on_quit = on_quit
        self._server_error = server_error

        self.setWindowTitle(f"PyStereo web {version}")
        self.setMinimumWidth(420)

        self._title = QLabel("Server is running")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setWeight(QFont.Weight.DemiBold)
        self._title.setFont(title_font)

        self._hint = QLabel("Open this address in your browser:")
        self._hint.setStyleSheet("color: #666;")

        self._link = QLabel(f'<a href="{url}">{url}</a>')
        self._link.setTextFormat(Qt.TextFormat.RichText)
        self._link.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
            | Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._link.setOpenExternalLinks(True)
        link_font = QFont()
        link_font.setPointSize(14)
        self._link.setFont(link_font)

        open_btn = QPushButton("Open in Browser")
        open_btn.setDefault(True)
        open_btn.clicked.connect(self._open_browser)

        copy_btn = QPushButton("Copy Link")
        copy_btn.clicked.connect(self._copy_link)

        logs_btn = QPushButton("Download logs")
        logs_btn.setEnabled(bool(log_url))
        logs_btn.clicked.connect(self._download_logs)

        quit_btn = QPushButton("Quit")
        quit_btn.clicked.connect(self._quit)

        row = QHBoxLayout()
        row.addWidget(open_btn)
        row.addWidget(copy_btn)
        row.addWidget(logs_btn)
        row.addStretch(1)
        row.addWidget(quit_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addWidget(self._title)
        layout.addWidget(self._hint)
        layout.addWidget(self._link)
        layout.addSpacing(8)
        layout.addLayout(row)

        self._health_timer = QTimer(self)
        self._health_timer.setInterval(1000)
        self._health_timer.timeout.connect(self._check_server_health)
        self._health_timer.start()
        QTimer.singleShot(400, self._check_server_health)

    def _check_server_health(self) -> None:
        if self._server_error is not None:
            err = self._server_error()
            if err:
                self._show_server_error(err)
                return
        try:
            # /api/health, not /health: the latter is the integration service
            # probe and answers 503 while the service endpoint is off (default).
            with urllib.request.urlopen(f"{self._url}/api/health", timeout=1.5) as resp:
                if resp.status == 200:
                    self._show_server_ok()
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        self._show_server_error(
            "Server is still starting, or another app is already using this port."
        )

    def _show_server_ok(self) -> None:
        self._title.setText("Server is running")
        self._hint.setText("Open this address in your browser:")
        self._hint.setStyleSheet("color: #666;")
        self._link.setVisible(True)

    def _show_server_error(self, message: str) -> None:
        self._title.setText("Server did not start")
        self._hint.setText(message)
        self._hint.setStyleSheet("color: #b42318;")
        self._link.setVisible(False)

    def _open_browser(self) -> None:
        QDesktopServices.openUrl(QUrl(self._url))

    def _copy_link(self) -> None:
        QGuiApplication.clipboard().setText(self._url)

    def _download_logs(self) -> None:
        if self._log_url:
            QDesktopServices.openUrl(QUrl(self._log_url))

    def _quit(self) -> None:
        self._on_quit()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._on_quit()
        event.accept()


def run_server_with_launch_dialog(
    *,
    host: str,
    port: int,
    version: str,
    start_server: Callable[[], None],
    open_browser: bool = True,
) -> None:
    """Start Flask in a background thread and show the launch dialog on the main thread."""
    ensure_port_available(host, port, gui=True)
    url = browser_url(host, port)
    log_url = f"{url}/api/logs"
    server_error: list[str] = []

    def _thread_target() -> None:
        try:
            start_server()
        except Exception as exc:
            logger.exception("Flask server failed")
            server_error.append(str(exc))

    server_thread = threading.Thread(
        target=_thread_target,
        name="pystereo-web-flask",
        daemon=True,
    )
    server_thread.start()

    qt_app = QApplication.instance() or QApplication([])
    window = WebLaunchWindow(
        url,
        version,
        on_quit=lambda: _force_exit(qt_app),
        log_url=log_url,
        server_error=lambda: server_error[0] if server_error else None,
    )
    window.show()
    window.raise_()
    window.activateWindow()

    if open_browser:
        # Delay slightly so the first paint / listener are more likely ready.
        def _open() -> None:
            if server_error:
                return
            try:
                with urllib.request.urlopen(f"{url}/api/health", timeout=2.0) as resp:
                    if resp.status != 200:
                        return
            except (urllib.error.URLError, TimeoutError, OSError):
                return
            webbrowser.open(url)

        threading.Timer(0.6, _open).start()

    qt_app.exec()


def _force_exit(qt_app: QApplication) -> None:
    qt_app.quit()
    # Flask's threaded server does not expose a clean shutdown from here.
    os._exit(0)
