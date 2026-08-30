"""Background download manager for AI stereo model weights.

Downloads HuggingFace / URL artifacts when the user enables AI stereo,
exposing thread-safe progress for the dashboard and Quest clients.
Inference paths must never trigger Hub downloads — only this manager does.
"""

from __future__ import annotations

import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEPTH_REPO_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_BASE_REPO_ID = "depth-anything/Depth-Anything-V2-Base-hf"
DEPTH_LARGE_REPO_ID = "depth-anything/Depth-Anything-V2-Large-hf"
SEGMENT_REPO_ID = "ZhengPeng7/BiRefNet"
AOTGAN_REPO_ID = "NimaBoscarino/aot-gan-places2"
LAMA_MODEL_URL = (
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/"
    "download/v0.1.0/big-lama.pt"
)
SHARP_MODEL_URL = (
    "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
)
SHARP_CKPT_FILENAME = "sharp_2572gikvuh.pt"

# Last-resort size when neither local cache nor remote metadata is available.
_FALLBACK_BYTES: Dict[str, int] = {
    "depth": 95 * 1024 * 1024,
    "depth_base": 400 * 1024 * 1024,
    "depth_large": 1300 * 1024 * 1024,
    "segment": 424 * 1024 * 1024,
    "inpaint": 196 * 1024 * 1024,
    "aotgan": 58 * 1024 * 1024,
    "sharp": 2800 * 1024 * 1024,
}

DEPTH_MODEL_SPECS: Dict[str, "ArtifactSpec"] = {}  # populated after ArtifactSpec


@dataclass(frozen=True)
class ArtifactSpec:
    """One downloadable weight package in the AI stereo pack."""

    id: str
    name: str
    kind: str  # "hf" | "url"
    repo_id: Optional[str] = None
    url: Optional[str] = None


STEREO_PACK: List[ArtifactSpec] = [
    ArtifactSpec(
        id="depth",
        name="Depth Anything V2 Small",
        kind="hf",
        repo_id=DEPTH_REPO_ID,
    ),
    ArtifactSpec(
        id="segment",
        name="BiRefNet",
        kind="hf",
        repo_id=SEGMENT_REPO_ID,
    ),
    ArtifactSpec(
        id="inpaint",
        name="LaMa Inpainting",
        kind="url",
        url=LAMA_MODEL_URL,
    ),
    ArtifactSpec(
        id="aotgan",
        name="AOT-GAN Inpainting",
        kind="hf",
        repo_id=AOTGAN_REPO_ID,
    ),
]

DEPTH_MODEL_SPECS.update({
    "small": ArtifactSpec(id="depth", name="Depth Anything V2 Small", kind="hf", repo_id=DEPTH_REPO_ID),
    "base": ArtifactSpec(id="depth_base", name="Depth Anything V2 Base", kind="hf", repo_id=DEPTH_BASE_REPO_ID),
    "large": ArtifactSpec(id="depth_large", name="Depth Anything V2 Large", kind="hf", repo_id=DEPTH_LARGE_REPO_ID),
})

SHARP_SPEC = ArtifactSpec(
    id="sharp",
    name="Apple SHARP (research-only)",
    kind="url",
    url=SHARP_MODEL_URL,
)


def _sharp_cache_path() -> Path:
    from torch.hub import get_dir

    return Path(get_dir()) / "checkpoints" / SHARP_CKPT_FILENAME


def _lama_cache_path(url: str) -> Path:
    """Mirror ``simple_lama_inpainting`` cache layout under torch hub checkpoints."""
    from urllib.parse import urlparse

    from torch.hub import get_dir

    filename = os.path.basename(urlparse(url).path)
    return Path(get_dir()) / "checkpoints" / filename


def _hf_repo_is_local(repo_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_files_only=True)
        return True
    except Exception:
        return False


def _artifact_is_local(spec: ArtifactSpec) -> bool:
    if spec.kind == "hf" and spec.repo_id:
        return _hf_repo_is_local(spec.repo_id)
    if spec.kind == "url" and spec.url:
        if spec.id == "inpaint":
            env_path = os.environ.get("LAMA_MODEL")
            if env_path and Path(env_path).is_file():
                return True
        return _lama_cache_path(spec.url).is_file()
    return False


def _dir_size_bytes(root: Path) -> int:
    total = 0
    try:
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def _measure_local_bytes(spec: ArtifactSpec) -> int:
    """Exact on-disk size for a cached artifact (0 if not present)."""
    if spec.kind == "url" and spec.url:
        if spec.id == "inpaint":
            env_path = os.environ.get("LAMA_MODEL")
            if env_path and Path(env_path).is_file():
                return Path(env_path).stat().st_size
        path = _lama_cache_path(spec.url)
        return path.stat().st_size if path.is_file() else 0
    if spec.kind == "hf" and spec.repo_id:
        try:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(repo_id=spec.repo_id, local_files_only=True)
            )
            return _dir_size_bytes(root)
        except Exception:
            return 0
    return 0


def _fetch_remote_bytes(spec: ArtifactSpec) -> int:
    """Exact remote size from Hub / HTTP headers (0 if unavailable)."""
    if spec.kind == "hf" and spec.repo_id:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(spec.repo_id, files_metadata=True)
            return sum(
                (s.size or 0)
                for s in (info.siblings or [])
                if getattr(s, "size", None)
            )
        except Exception:
            return 0
    if spec.kind == "url" and spec.url:
        try:
            req = urllib.request.Request(
                spec.url,
                method="HEAD",
                headers={"User-Agent": "PyStereo/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                length = resp.headers.get("Content-Length")
                if length:
                    return int(length)
        except Exception:
            return 0
    return 0


def _hf_repo_folder_name(repo_id: str) -> str:
    """HF hub on-disk folder name for a model repo (``models--org--name``)."""
    return "models--" + repo_id.replace("/", "--")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _hf_hub_root() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE).expanduser().resolve()
    except Exception:
        return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def _hf_repo_cache_dir(repo_id: str) -> Path | None:
    """Return the HF hub cache folder for *repo_id*, if present and trusted."""
    if not repo_id or "/" not in repo_id or repo_id.strip() != repo_id:
        return None
    expected_name = _hf_repo_folder_name(repo_id)
    hub_root = _hf_hub_root()
    try:
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_id != repo_id:
                continue
            path = Path(repo.repo_path).resolve()
            # Hard allowlist: must be hub_root/models--org--name exactly.
            if path.name != expected_name:
                logger.warning(
                    "Refusing HF delete: unexpected folder name %s (want %s)",
                    path.name,
                    expected_name,
                )
                return None
            if path.parent != hub_root:
                logger.warning(
                    "Refusing HF delete: %s is not directly under %s",
                    path,
                    hub_root,
                )
                return None
            if not path.is_dir():
                return None
            return path
    except Exception:
        return None
    return None


_ALLOWED_URL_CHECKPOINTS = {"big-lama.pt", SHARP_CKPT_FILENAME}


def _safe_url_cache_file(url: str) -> Path | None:
    """Return the torch-hub checkpoint path for *url* if its filename is allowlisted."""
    if not url:
        return None
    from urllib.parse import urlparse

    from torch.hub import get_dir

    filename = os.path.basename(urlparse(url).path)
    if filename not in _ALLOWED_URL_CHECKPOINTS:
        logger.warning("Refusing delete: unexpected checkpoint filename %r", filename)
        return None
    checkpoints = (Path(get_dir()).expanduser().resolve() / "checkpoints")
    path = (checkpoints / filename).resolve()
    if path.parent != checkpoints or path.name != filename:
        logger.warning("Refusing delete: path escaped checkpoints dir (%s)", path)
        return None
    return path


def _artifact_deletable_path(spec: ArtifactSpec) -> Path | None:
    """Allowlisted on-disk path for *spec* if it exists and may be deleted."""
    if spec.kind == "hf" and spec.repo_id:
        root = _hf_repo_cache_dir(spec.repo_id)
        return root if root is not None and root.is_dir() else None
    if spec.kind == "url" and spec.url:
        if spec.id != "sharp" and os.environ.get("LAMA_MODEL"):
            return None
        path = _safe_url_cache_file(spec.url)
        return path if path is not None and path.is_file() else None
    return None


def _delete_artifact(spec: ArtifactSpec) -> tuple[int, str | None]:
    """Delete one pack artifact from disk.

    Returns ``(bytes_removed, path_str_or_none)``. Empty / unexpected names
    abort instead of falling through to a broad ``rmtree``.
    """
    target = _artifact_deletable_path(spec)
    if target is None:
        # Still log skip reasons for HF/url policy (override, missing, etc.).
        if spec.kind == "url" and spec.url and os.environ.get("LAMA_MODEL"):
            logger.info(
                "Skipping delete of %s; LAMA_MODEL override is set",
                spec.id,
            )
        return 0, None

    if spec.kind == "hf" and spec.repo_id:
        removed = _dir_size_bytes(target)
        import shutil

        shutil.rmtree(target)
        hub_root = _hf_hub_root()
        locks = (hub_root / ".locks" / target.name).resolve()
        if (
            locks.is_dir()
            and locks.name == target.name
            and locks.parent == (hub_root / ".locks").resolve()
            and _is_relative_to(locks, hub_root / ".locks")
        ):
            shutil.rmtree(locks, ignore_errors=True)
        logger.info("Deleted HF cache for %s (%s bytes)", spec.repo_id, removed)
        return removed, str(target)

    if spec.kind == "url" and spec.url:
        removed = target.stat().st_size
        target.unlink()
        logger.info("Deleted %s (%s bytes)", target, removed)
        return removed, str(target)

    return 0, None


def _fallback_bytes(spec: ArtifactSpec) -> int:
    return _FALLBACK_BYTES.get(spec.id, 0)


def _make_progress_tqdm(
    reporter: "ModelDownloadManager",
    artifact_id: str,
) -> type:
    """Return a tqdm-compatible class for ``snapshot_download(tqdm_class=…)``.

    Must be a class (HF's ``thread_map`` calls ``.get_lock()``), and must accept
    HF-specific kwargs like ``name=`` that stock tqdm rejects. HF also mutates
    ``.total`` on the live instance as file sizes are discovered.
    """
    import threading as _threading

    class _ProgressTqdm:
        # tqdm.contrib.concurrent.ensure_lock may ``del cls._lock`` — use getattr.
        _lock: Optional[_threading.RLock] = None

        @classmethod
        def get_lock(cls) -> _threading.RLock:
            lock = getattr(cls, "_lock", None)
            if lock is None:
                lock = _threading.RLock()
                cls._lock = lock
            return lock

        @classmethod
        def set_lock(cls, lock: Optional[_threading.RLock]) -> None:
            cls._lock = lock

        def __init__(self, iterable: Any = None, *args: Any, **kwargs: Any) -> None:
            # HF uses this class two ways:
            # 1) bytes bar: kwargs only (unit="B", total grows dynamically)
            # 2) thread_map file-count bar: first arg is an iterable
            self.iterable = iterable
            self.n = float(kwargs.get("initial") or 0)
            self.total = float(kwargs.get("total") or 0)
            self.desc = str(kwargs.get("desc") or "")
            self.disable = bool(kwargs.get("disable", True))
            self.format_dict: dict[str, Any] = {}
            self._closed = False
            # Only the shared bytes bar should drive dashboard progress.
            self._report = (
                kwargs.get("unit") == "B"
                or bool(kwargs.get("unit_scale"))
                or kwargs.get("name") == "huggingface_hub.snapshot_download"
            )
            if self._report and self.total > 0:
                reporter._on_file_start(artifact_id, self.total, self.desc)

        def __iter__(self) -> Any:
            if self.iterable is None:
                return
            for item in self.iterable:
                yield item
                # File-count bar: advance without treating counts as bytes.
                self.n += 1.0

        def __enter__(self) -> "_ProgressTqdm":
            return self

        def __exit__(self, *args: Any) -> None:
            self.close()

        def update(self, n: Optional[float] = 1) -> None:
            if n is None:
                n = 1
            self.n += float(n)
            if self._report:
                reporter._on_file_progress(
                    artifact_id, self.n, float(self.total or 0), self.desc
                )

        def refresh(self) -> None:
            # HF bumps ``.total`` then calls refresh when a new file size is known.
            if self._report and self.total > 0:
                reporter._on_file_progress(
                    artifact_id, self.n, float(self.total), self.desc
                )

        def set_description(self, desc: Optional[str] = None, refresh: bool = True) -> None:
            if desc is not None:
                self.desc = str(desc)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            if self._closed:
                return
            self._closed = True
            if self._report and self.total > 0:
                reporter._on_file_done(artifact_id, float(self.total))

        def __del__(self) -> None:
            try:
                self.close()
            except Exception:
                pass

    return _ProgressTqdm


@dataclass
class _ArtifactRuntime:
    # absent | queued | downloading | ready | error
    state: str = "absent"
    percent: int = 0
    bytes_downloaded: int = 0
    bytes_total: int = 0
    error: Optional[str] = None


@dataclass
class ModelDownloadManager:
    """Thread-safe download coordinator for the AI stereo weight pack."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _pack_state: str = field(default="idle", init=False)  # idle|downloading|ready|error
    _message: str = field(default="", init=False)
    _error: Optional[str] = field(default=None, init=False)
    _artifacts: Dict[str, _ArtifactRuntime] = field(default_factory=dict, init=False)
    # Cached exact sizes (local measurement or remote metadata).
    _known_bytes: Dict[str, int] = field(default_factory=dict, init=False)
    _remote_probed: bool = field(default=False, init=False)
    # Aggregated byte accounting across the active download job.
    _job_bytes_done: int = field(default=0, init=False)
    _job_bytes_total: int = field(default=0, init=False)
    _file_reported: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for spec in STEREO_PACK:
            self._artifacts[spec.id] = _ArtifactRuntime()
        self.refresh_local_state()

    def _exact_bytes(self, spec: ArtifactSpec) -> int:
        """Best-known exact size: measured local > cached > fallback."""
        known = self._known_bytes.get(spec.id, 0)
        if known > 0:
            return known
        return _fallback_bytes(spec)

    def _remember_bytes(self, artifact_id: str, size: int) -> None:
        if size > 0:
            self._known_bytes[artifact_id] = size

    def refresh_local_state(self) -> None:
        """Re-scan disk cache and update per-artifact / pack readiness."""
        with self._lock:
            if self._pack_state == "downloading":
                return
            all_ready = True
            for spec in STEREO_PACK:
                local = _artifact_is_local(spec)
                art = self._artifacts[spec.id]
                if local:
                    measured = _measure_local_bytes(spec)
                    if measured > 0:
                        self._remember_bytes(spec.id, measured)
                    size = self._exact_bytes(spec)
                    art.state = "ready"
                    art.percent = 100
                    art.error = None
                    art.bytes_total = size
                    art.bytes_downloaded = size
                else:
                    all_ready = False
                    if art.state != "error":
                        art.state = "absent"
                        art.percent = 0
                        art.bytes_downloaded = 0
                        art.bytes_total = self._exact_bytes(spec)
            if all_ready:
                self._pack_state = "ready"
                self._message = "AI models ready"
                self._error = None
                self._job_bytes_done = sum(
                    self._artifacts[s.id].bytes_total for s in STEREO_PACK
                )
                self._job_bytes_total = self._job_bytes_done
            elif self._pack_state != "error":
                self._pack_state = "idle"
                self._message = ""

    def is_pack_ready(self) -> bool:
        self.refresh_local_state()
        with self._lock:
            return self._pack_state == "ready"

    def is_downloading(self) -> bool:
        with self._lock:
            return self._pack_state == "downloading"

    def delete_stereo_pack(self) -> Dict[str, Any]:
        """Remove cached stereo-pack weights from disk and reset pack state.

        Refuses while a download is in progress. Does not change the
        ``enable_ai_stereo`` config flag — callers disable that separately.
        """
        with self._lock:
            if self._pack_state == "downloading":
                raise RuntimeError("Cannot delete AI models while a download is in progress")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Cannot delete AI models while a download is in progress")

        deleted: Dict[str, int] = {}
        deleted_paths: List[str] = []
        total = 0
        for spec in STEREO_PACK:
            bytes_removed, path = _delete_artifact(spec)
            deleted[spec.id] = bytes_removed
            total += bytes_removed
            if path:
                deleted_paths.append(path)

        with self._lock:
            self._pack_state = "idle"
            self._message = ""
            self._error = None
            self._job_bytes_done = 0
            self._job_bytes_total = 0
            self._remote_probed = False
            for spec in STEREO_PACK:
                art = self._artifacts[spec.id]
                art.state = "absent"
                art.percent = 0
                art.bytes_downloaded = 0
                art.error = None
                # Keep known remote sizes so the UI can still show expected MB.

        self.refresh_local_state()
        logger.info(
            "AI stereo model pack deleted (%s bytes across %s artifacts)",
            total,
            len(deleted),
        )
        return {
            "deleted_bytes": total,
            "artifacts": deleted,
            "deleted_paths": deleted_paths,
        }

    def is_depth_model_local(self, model_size: str) -> bool:
        """Check if a specific depth model's weights are cached locally."""
        spec = DEPTH_MODEL_SPECS.get(model_size)
        if spec is None:
            return False
        return _artifact_is_local(spec)

    def ensure_depth_model_async(self, model_size: str) -> bool:
        """Start downloading a specific depth model in the background.

        Returns True if a download was started (or already running),
        False if the model size is invalid.
        """
        spec = DEPTH_MODEL_SPECS.get(model_size)
        if spec is None:
            return False
        if _artifact_is_local(spec):
            return True
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            art_id = spec.id
            if art_id not in self._artifacts:
                self._artifacts[art_id] = _ArtifactRuntime()
            self._error = None
            self._message = f"Downloading {spec.name}…"
            size = _FALLBACK_BYTES.get(art_id, 0)
            self._file_reported = 0
            art = self._artifacts[art_id]
            art.state = "queued"
            art.percent = 0
            art.error = None
            art.bytes_total = size
            art.bytes_downloaded = 0
            self._thread = threading.Thread(
                target=self._run_single_download,
                args=(spec,),
                name=f"pystereo-dl-{model_size}",
                daemon=True,
            )
            self._thread.start()
        return True

    def is_sharp_model_local(self) -> bool:
        """Check if the SHARP checkpoint is cached locally."""
        return _artifact_is_local(SHARP_SPEC)

    def ensure_sharp_model_async(self) -> bool:
        """Start downloading the SHARP checkpoint in the background.

        Returns True if a download was started (or the model is already local).
        """
        if _artifact_is_local(SHARP_SPEC):
            return True
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            art_id = SHARP_SPEC.id
            if art_id not in self._artifacts:
                self._artifacts[art_id] = _ArtifactRuntime()
            self._error = None
            self._message = f"Downloading {SHARP_SPEC.name}..."
            size = _FALLBACK_BYTES.get(art_id, 0)
            self._file_reported = 0
            art = self._artifacts[art_id]
            art.state = "queued"
            art.percent = 0
            art.error = None
            art.bytes_total = size
            art.bytes_downloaded = 0
            self._thread = threading.Thread(
                target=self._run_single_download,
                args=(SHARP_SPEC,),
                name="pystereo-dl-sharp",
                daemon=True,
            )
            self._thread.start()
        return True

    def delete_sharp_model(self) -> dict[str, Any]:
        """Remove the SHARP checkpoint from disk."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("Cannot delete while a download is in progress")
        bytes_removed, path = _delete_artifact(SHARP_SPEC)
        with self._lock:
            art = self._artifacts.get(SHARP_SPEC.id)
            if art:
                art.state = "absent"
                art.percent = 0
                art.bytes_downloaded = 0
                art.error = None
        return {"deleted_bytes": bytes_removed, "path": path}

    def _run_single_download(self, spec: ArtifactSpec) -> None:
        """Download a single artifact (used for optional models)."""
        try:
            if _artifact_is_local(spec):
                with self._lock:
                    art = self._artifacts.get(spec.id)
                    if art:
                        art.state = "ready"
                        art.percent = 100
            else:
                with self._lock:
                    self._message = f"Downloading {spec.name}..."
                    art = self._artifacts.get(spec.id)
                    if art:
                        art.state = "downloading"
                        art.percent = 0
                    self._file_reported = 0
                if spec.kind == "hf" and spec.repo_id:
                    self._download_hf(spec)
                elif spec.kind == "url" and spec.url:
                    self._download_url(spec)
                measured = _measure_local_bytes(spec)
                with self._lock:
                    if measured > 0:
                        self._remember_bytes(spec.id, measured)
                    art = self._artifacts.get(spec.id)
                    if art:
                        art.state = "ready"
                        art.percent = 100
                        art.bytes_total = measured or art.bytes_total
                        art.bytes_downloaded = art.bytes_total
            with self._lock:
                self._message = f"{spec.name} ready"
                self._error = None
            self.refresh_local_state()
            logger.info("Downloaded %s", spec.name)
        except Exception as exc:
            logger.exception("Download of %s failed", spec.name)
            with self._lock:
                self._error = str(exc)
                self._message = f"Download failed: {exc}"
                art = self._artifacts.get(spec.id)
                if art and art.state == "downloading":
                    art.state = "error"
                    art.error = str(exc)
            self.refresh_local_state()

    def ensure_stereo_pack_async(self) -> None:
        """Start a background download if the pack is not already ready."""
        self.refresh_local_state()
        with self._lock:
            if self._pack_state == "ready":
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._pack_state = "downloading"
            self._error = None
            self._message = "Starting AI model download…"
            # Count already-local models toward overall progress immediately.
            self._job_bytes_done = 0
            self._job_bytes_total = sum(self._exact_bytes(s) for s in STEREO_PACK)
            self._file_reported = 0
            for spec in STEREO_PACK:
                art = self._artifacts[spec.id]
                if art.state == "ready":
                    self._job_bytes_done += art.bytes_total or self._exact_bytes(spec)
                    continue
                # Waiting in line — not actively downloading yet.
                art.state = "queued"
                art.percent = 0
                art.error = None
                art.bytes_total = self._exact_bytes(spec)
                art.bytes_downloaded = 0
            self._thread = threading.Thread(
                target=self._run_download,
                name="pystereo-ai-download",
                daemon=True,
            )
            self._thread.start()

    def status_dict(self) -> Dict[str, Any]:
        self.refresh_local_state()
        self._ensure_remote_sizes()
        self.refresh_local_state()
        # Resolve allowlisted paths outside the lock (disk I/O).
        path_by_id: Dict[str, str] = {}
        deletable_paths: List[str] = []
        for spec in STEREO_PACK:
            path = _artifact_deletable_path(spec)
            if path is not None:
                path_str = str(path)
                path_by_id[spec.id] = path_str
                deletable_paths.append(path_str)
        with self._lock:
            pack_total = sum(self._exact_bytes(s) for s in STEREO_PACK)
            if self._pack_state == "downloading" and self._job_bytes_total > 0:
                percent = min(
                    100,
                    int(100 * self._job_bytes_done / self._job_bytes_total),
                )
                pack_total = self._job_bytes_total
                pack_done = self._job_bytes_done
            elif self._pack_state == "ready":
                percent = 100
                pack_done = pack_total
            else:
                percent = 0
                pack_done = sum(
                    self._artifacts[s.id].bytes_downloaded for s in STEREO_PACK
                )
            models: Dict[str, Any] = {}
            for spec in STEREO_PACK:
                art = self._artifacts[spec.id]
                size = art.bytes_total or self._exact_bytes(spec)
                entry: Dict[str, Any] = {
                    "name": spec.name,
                    "local": art.state == "ready",
                    "download_state": art.state,
                    "percent": art.percent,
                    "bytes_downloaded": art.bytes_downloaded,
                    "bytes_total": size,
                    # True when size came from disk measurement or remote metadata.
                    "size_exact": self._known_bytes.get(spec.id, 0) > 0,
                }
                if spec.repo_id:
                    entry["repo_id"] = spec.repo_id
                if spec.url:
                    entry["url"] = spec.url
                if art.error:
                    entry["error"] = art.error
                cache_path = path_by_id.get(spec.id)
                if cache_path:
                    entry["cache_path"] = cache_path
                models[spec.id] = entry
            return {
                "state": self._pack_state,
                "percent": percent,
                "bytes_downloaded": pack_done,
                "bytes_total": pack_total,
                "message": self._message,
                "error": self._error,
                "models": models,
                "deletable_paths": deletable_paths,
            }

    def _ensure_remote_sizes(self) -> None:
        """One-shot Hub/HTTP size probe for artifacts not yet measured locally."""
        with self._lock:
            if self._remote_probed or self._pack_state == "downloading":
                return
            missing = [
                s
                for s in STEREO_PACK
                if self._known_bytes.get(s.id, 0) <= 0
                and self._artifacts[s.id].state != "ready"
            ]
        if not missing:
            with self._lock:
                self._remote_probed = True
            return
        probed: Dict[str, int] = {}
        for spec in missing:
            remote = _fetch_remote_bytes(spec)
            if remote > 0:
                probed[spec.id] = remote
        with self._lock:
            self._remote_probed = True
            for art_id, size in probed.items():
                self._remember_bytes(art_id, size)
                art = self._artifacts.get(art_id)
                if art is not None and art.state != "ready":
                    art.bytes_total = size

    # -- progress callbacks (called from download thread / tqdm) -------------

    def _on_file_start(
        self, artifact_id: Optional[str], total: float, desc: str
    ) -> None:
        with self._lock:
            self._file_reported = 0
            if artifact_id and artifact_id in self._artifacts:
                art = self._artifacts[artifact_id]
                art.state = "downloading"
                if total > 0:
                    art.bytes_total = int(total)
                self._message = f"Downloading {self._name_for(artifact_id)}…"
            elif desc:
                self._message = f"Downloading {desc}…"

    def _on_file_progress(
        self,
        artifact_id: Optional[str],
        n: float,
        total: float,
        desc: str,
    ) -> None:
        with self._lock:
            delta = int(n) - self._file_reported
            if delta > 0:
                self._job_bytes_done += delta
                self._file_reported = int(n)
            if artifact_id and artifact_id in self._artifacts:
                art = self._artifacts[artifact_id]
                art.bytes_downloaded = int(n)
                if total > 0:
                    art.bytes_total = int(total)
                    art.percent = min(100, int(100 * n / total))
                self._message = f"Downloading {self._name_for(artifact_id)}…"

    def _on_file_done(self, artifact_id: Optional[str], total: float) -> None:
        with self._lock:
            # Ensure final bytes are counted once.
            remaining = int(total) - self._file_reported
            if remaining > 0:
                self._job_bytes_done += remaining
            self._file_reported = 0

    def _name_for(self, artifact_id: str) -> str:
        for spec in STEREO_PACK:
            if spec.id == artifact_id:
                return spec.name
        return artifact_id

    def _run_download(self) -> None:
        try:
            # Resolve remote sizes up front so progress totals are exact.
            for spec in STEREO_PACK:
                with self._lock:
                    known = self._known_bytes.get(spec.id, 0)
                if known <= 0 and not _artifact_is_local(spec):
                    remote = _fetch_remote_bytes(spec)
                    if remote > 0:
                        with self._lock:
                            prev = self._exact_bytes(spec)
                            self._remember_bytes(spec.id, remote)
                            self._artifacts[spec.id].bytes_total = remote
                            self._job_bytes_total = max(
                                0, self._job_bytes_total - prev
                            ) + remote

            for spec in STEREO_PACK:
                if _artifact_is_local(spec):
                    measured = _measure_local_bytes(spec)
                    with self._lock:
                        if measured > 0:
                            prev = self._exact_bytes(spec)
                            self._remember_bytes(spec.id, measured)
                            self._job_bytes_total = max(
                                0, self._job_bytes_total - prev
                            ) + measured
                        size = self._exact_bytes(spec)
                        art = self._artifacts[spec.id]
                        already_counted = art.state == "ready"
                        art.state = "ready"
                        art.percent = 100
                        art.bytes_total = size
                        art.bytes_downloaded = size
                        # ensure_stereo_pack_async already counted ready models.
                        if not already_counted:
                            self._job_bytes_done += size
                    continue
                with self._lock:
                    self._message = f"Downloading {spec.name}…"
                    art = self._artifacts[spec.id]
                    art.state = "downloading"
                    art.percent = 0
                    self._file_reported = 0
                if spec.kind == "hf" and spec.repo_id:
                    self._download_hf(spec)
                elif spec.kind == "url" and spec.url:
                    self._download_url(spec)
                measured = _measure_local_bytes(spec)
                with self._lock:
                    if measured > 0:
                        self._remember_bytes(spec.id, measured)
                    art = self._artifacts[spec.id]
                    art.state = "ready"
                    art.percent = 100
                    size = measured or art.bytes_total or self._exact_bytes(spec)
                    art.bytes_total = size
                    art.bytes_downloaded = size
            with self._lock:
                self._pack_state = "ready"
                self._message = "AI models ready"
                self._error = None
                self._job_bytes_total = sum(
                    self._artifacts[s.id].bytes_total for s in STEREO_PACK
                )
                self._job_bytes_done = self._job_bytes_total
            logger.info("AI stereo model pack download complete")
        except Exception as exc:
            logger.exception("AI stereo model pack download failed")
            with self._lock:
                self._pack_state = "error"
                self._error = str(exc)
                self._message = f"Download failed: {exc}"
                for art in self._artifacts.values():
                    if art.state == "downloading":
                        art.state = "error"
                        art.error = str(exc)

    def _download_hf(self, spec: ArtifactSpec) -> None:
        from huggingface_hub import snapshot_download

        with self._lock:
            total = self._known_bytes.get(spec.id, 0) or self._exact_bytes(spec)
            if total > 0:
                self._artifacts[spec.id].bytes_total = total

        snapshot_download(
            repo_id=spec.repo_id,
            tqdm_class=_make_progress_tqdm(self, spec.id),
        )

    def _download_url(self, spec: ArtifactSpec) -> None:
        assert spec.url is not None
        dest = _lama_cache_path(spec.url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".partial")

        req = urllib.request.Request(spec.url, headers={"User-Agent": "PyStereo/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with self._lock:
                if total <= 0:
                    total = self._exact_bytes(spec)
                else:
                    prev = self._exact_bytes(spec)
                    self._remember_bytes(spec.id, total)
                    self._job_bytes_total = max(0, self._job_bytes_total - prev) + total
                self._artifacts[spec.id].bytes_total = total
            self._on_file_start(spec.id, float(total), spec.name)
            downloaded = 0
            with open(tmp, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    self._on_file_progress(
                        spec.id, float(downloaded), float(total), spec.name
                    )
        tmp.replace(dest)
        self._on_file_done(spec.id, float(total))

_manager: Optional[ModelDownloadManager] = None
_manager_lock = threading.Lock()


def get_download_manager() -> ModelDownloadManager:
    """Return (or create) the process-wide download manager singleton."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ModelDownloadManager()
        return _manager


def apply_ai_stereo_enabled(enabled: bool) -> None:
    """Apply persisted AI-stereo enable flag to registry + download manager."""
    from pystereo_core.registry import get_registry

    registry = get_registry()
    registry.enabled = enabled
    if registry.has_capability("depth"):
        registry.toggle_capability("depth", enabled)
    if enabled and registry.available:
        get_download_manager().ensure_stereo_pack_async()
    else:
        get_download_manager().refresh_local_state()
