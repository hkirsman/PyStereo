# PyStereo

Standalone AI stereo synthesis - convert 2D photos into side-by-side (SBS)
stereo pairs. Depth estimation, perspective warping, and inpainting run locally
via PyTorch (CPU / MPS / CUDA). Weights download on first use.

PyStereo is a standalone stereo synthesis service designed to be used on its own
or integrated into any application that needs 2D-to-SBS conversion over HTTP.

## Setup (macOS / Linux)

```bash
./bootstrap.sh
```

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## Run

Download model weights before first use (~1-2 GB depending on depth model):

```bash
source .venv/bin/activate
python -m pystereo_core --download-model
# Free disk later:
# python -m pystereo_core --remove-model
```

### Web UI (browser)

Upload an image and watch each pipeline stage (depth, warp, inpaint) in the
browser:

```bash
source .venv/bin/activate
python app.py
```

Open **http://127.0.0.1:8766**

### Desktop GUI (`pystereo_core`)

PySide6 window with batch processing, model management, and an embedded server
toggle for HTTP service integration:

```bash
source .venv/bin/activate
python -m pystereo_core
```

### CLI batch mode

```bash
source .venv/bin/activate
python -m pystereo_core --cli --folder /path/to/photos --recursive
```

### HTTP service

The web UI and the desktop GUI both expose HTTP endpoints for integration:

- `GET /health` - returns `{"status": "ok", "kind": "stereo"}`
- `POST /transform` - accepts a JPEG/PNG upload, returns an SBS JPEG

Any application can point its stereo service URL to `http://127.0.0.1:8766`
to use PyStereo for AI stereo generation.

Headless equivalent: `python app.py --host 127.0.0.1 --port 8766`

## Standalone builds (PyInstaller)

With the repo venv active, install PyInstaller (`pip install pyinstaller`), then
from the repo root:

- **Batch tool:** `pyinstaller packaging/pystereo_batch.spec` -
  `dist/PyStereo/PyStereo.app` (large: PyTorch + Qt). Run
  `./dist/PyStereo/PyStereo.app` (add `--cli ...` for headless).
- **Web UI (Flask):** `pyinstaller packaging/pystereo_web.spec` -
  `dist/PyStereoWeb/PyStereoWeb.app`. Run `./dist/PyStereoWeb/PyStereoWeb.app`,
  then open **http://127.0.0.1:8766**.

Or use the convenience scripts:

```bash
# macOS
./compile-binaries-mac.sh

# Windows
compile-binaries-win.bat
```

First launch still needs an explicit model download (web **Download**, batch
**Models** group, or `--download-model`) unless you ship weights separately.
Distribute either bundle by zipping the whole output folder, including
`_internal/`.

## Pipeline stages

1. **Depth estimation** - Depth Anything V2 (or configurable model)
2. **Foreground segmentation** - BiRefNet for depth healing
3. **Perspective warp** - sub-pixel forward splatting with z-buffer
4. **Inpainting** - LaMa (default) or AOTGAN for disocclusion fill
5. **SBS assembly** - left + right eye side-by-side JPEG output

Multiple stereo methods are available (bg_plate_fill, clean_fill, combo_fill,
direct_fill, fullres_warp, iterative_fill, ldi_inpaint, per_eye_inpaint,
routed_fill) - configurable via the web UI, a `method` form field, or a
`?method=name` query parameter.
