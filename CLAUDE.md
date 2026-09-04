# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

PyStereo - standalone AI stereo synthesis that converts 2D photos into side-by-side (SBS) stereo pairs. Depth estimation, perspective warping, and inpainting run locally via PyTorch (CPU / MPS / CUDA). Weights download on first use. Use the web UI, Qt desktop GUI, CLI, or the HTTP `/transform` service.

## Setup & Run

```bash
./bootstrap.sh                          # create .venv, install deps
source .venv/bin/activate

# ml-sharp submodule (Apple SHARP methods, research-only license)
git submodule update --init
pip install -e ./ml-sharp --no-deps

# Download model weights (~1-2 GB)
python -m pystereo_core --download-model
```

Always use `.venv/bin/python3` - system python does not have deps installed.

### Three interfaces

```bash
python app.py                                        # Web UI at http://127.0.0.1:8766
python -m pystereo_core                              # Qt desktop GUI (needs PySide6-Essentials)
python -m pystereo_core --cli --folder /path --recursive  # CLI batch mode
```

### HTTP service

Both the web UI and desktop GUI expose:
- `GET /health` - `{"ok": true, "kind": "stereo"}`
- `POST /transform` - accepts JPEG/PNG upload, returns SBS JPEG. Optional form fields: `method`, `depth_model`, `max_dim`, `max_pixels`

Off by default (`/health` 503, `/transform` 403). Enabled by the web UI
setting `service_enabled`, `PYSTEREO_SERVICE=1`, or the desktop GUI's
"Start server" button (`app.force_service_enabled`).

Headless: `python app.py --host 127.0.0.1 --port 8766`

## Testing

No project test suite yet. Verify changes by running the web UI (`python app.py`) and processing a photo through `/transform`.

## Architecture

### Core pipeline (depth-map methods)

1. **Depth estimation** - Depth Anything V2 (`pystereo_core/depth.py`)
2. **Foreground segmentation** - BiRefNet for depth healing (`stereo/segment.py`, `stereo/heal.py`)
3. **Perspective warp** - sub-pixel forward splatting with z-buffer (`stereo/warp.py`)
4. **Inpainting** - LaMa (default) or AOT-GAN for disocclusion fill (`stereo/inpaint.py`)
5. **SBS assembly** - left + right eye side-by-side JPEG (`stereo/pipeline.py`)

SHARP methods skip depth estimation entirely - Apple SHARP predicts a 3D Gaussian splat and renders from two virtual cameras (`stereo/sharp_predict.py`, `stereo/splat_render.py`, `stereo/taichi_render.py`). The predictor stays resident between photos and is never unloaded on a timer by default (`PYSTEREO_SHARP_IDLE_S`, 0 = keep loaded). The `*_full` methods render entirely in Taichi - projection and compositing, no torch in the render path (`stereo/taichi_full.py`, `stereo/_taichi_full_kernels.py`).

### Key modules

| Module | Role |
|--------|------|
| `app.py` | Flask web UI + HTTP service (single file) |
| `pystereo_core/__main__.py` | CLI entry point + Qt GUI launcher |
| `pystereo_core/registry.py` | Thread-safe AI model registry (singleton, keyed by capability) |
| `pystereo_core/download.py` | Background weight download manager (HuggingFace Hub + direct URLs) |
| `pystereo_core/stereo/pipeline.py` | Orchestrates preprocessing + method dispatch + SBS composition; `derive(**overrides)` gives per-request settings without reloading BiRefNet / LaMa |
| `pystereo_core/stereo/config.py` | `StereoSettings` dataclass, env-var config, method defaults |
| `pystereo_core/stereo/methods/` | Pluggable stereo methods (one file per method, auto-registered) |
| `pystereo_core/stereo/methods/base.py` | `BaseStereoMethod` ABC - `warp_and_fill()` or `synthesize()` |
| `pystereo_core/stereo/taichi_full.py` | Torch-free SHARP splat renderer (Taichi projection + compositing) |
| `pystereo_core/stereo/timing.py` | Per-step timing collection (`record_step` into `intermediates["timings"]`) |
| `static/` | Web UI frontend (HTML/JS/CSS) |
| `packaging/` | PyInstaller specs + entry points |
| `ml-sharp/` | Git submodule - Apple SHARP (research-only license, never bundled) |
| `experiments/` | Standalone proof-of-concept code, never imported by `pystereo_core` |

### Method registry

Methods live in `pystereo_core/stereo/methods/`, one file per method. Each subclasses `BaseStereoMethod` and declares class variables: `name`, `label`, `description`, `deprecated`, `needs_depth`, `SETTING_OVERRIDES`. Registration is in `methods/__init__.py`. UI order is the `METHOD_UI_ORDER` tuple.

Depth-map methods implement `warp_and_fill()`. SHARP methods set `needs_depth = False` and implement `synthesize()` instead.

Methods report per-step timings via `stereo/timing.py:record_step` into the `intermediates` dict (shown in the web UI stages panel); SHARP methods also set `intermediates["render_backend"]` to `"taichi"` or `"torch"` so the UI can say which renderer actually ran.

### Settings

User preferences persist in `settings.json` (gitignored). `StereoSettings` is a frozen dataclass with env-var overrides (`PYSTEREO_METHOD`, `PYSTEREO_INPAINT`, `PYSTEREO_MAX_DIM`, etc.) and per-method `SETTING_OVERRIDES`. `PYSTEREO_SHARP_IDLE_S` (default 0 = never unload) sets how long the resident SHARP predictor survives idle before unloading; the web UI setting `sharp_idle_s` overrides it at runtime.

### Caches

`.sharp_cache/` holds one `.npz` per SHARP prediction, keyed by pixel hash and internal size; `outputs/` holds web UI results. `StereoSettings.sharp_disk_cache = False` (web UI "Disable cache", `/transform` `no_cache=1`, `PYSTEREO_SHARP_CACHE=0`) skips reading `.sharp_cache/` but still writes it - the renderer reads the file - and never touches the resident predictor. `predict_gaussians` reports `sharp_cache` (`hit` / `miss` / `off`) and `sharp_model_loaded` into `intermediates`; `sharp_predict.cache_note` turns them into the "(cached)" / "(cache off, model load)" suffix on the "SHARP prediction" timing label. Size control lives in `sharp_predict` (`set_cache_max_mb`, LRU by mtime, `prune_cache` after each write) and `app.py` (`_prune_outputs`, `outputs_keep`). `GET /api/cache`, `POST /api/cache/clear`, `POST /api/models/unload` back the "Cache & memory" panel.

### Model weights

Weights download via `pystereo_core/download.py` into the HuggingFace cache or `.sharp_cache/`. Inference code loads with `local_files_only=True` and never hits the Hub. The download manager exposes thread-safe progress for UIs.

## Key Conventions

### PyTorch device selection

Prefer `mps` on Apple Silicon, then `cuda`, then `cpu`. Never hard-code CUDA only. See `registry.py:detect_device()`.

### Deprecated methods stay

Deprecated stereo methods remain in the codebase (marked `deprecated: ClassVar[bool] = True`). They document dead ends and serve as regression baselines. See `docs/DECISIONS.md`.

### Punctuation

Use ASCII hyphens (`-`) everywhere - no em dashes or en dashes in UI copy, comments, or commit messages.

### Git commits

Chris Beams style. Imperative mood, 50-char subject, 72-char body wrap. Ticket prefix from branch name (e.g. `GH-4: Add max_pixels output budget`).

### Licensing

Depth Anything V2 Small is Apache-2.0 (default). Base/Large are CC-BY-NC-4.0. All SHARP methods are Apple ML Research license (research-only) - opt-in, never bundled.
