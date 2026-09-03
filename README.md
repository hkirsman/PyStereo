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

- `GET /health` - returns `{"ok": true, "kind": "stereo"}`
- `POST /transform` - accepts a JPEG/PNG upload, returns an SBS JPEG

The service is off by default: `/health` answers 503 and `/transform` 403
until it is enabled. Turn it on under "Integration service" in the web UI
(`service_enabled` in `settings.json`), with `PYSTEREO_SERVICE=1` for
headless runs, or with the desktop GUI's "Start server" button. The web UI
itself never uses these endpoints.

Optional `/transform` form fields: `method`, `depth_model`, `max_dim`
(processing resolution), `max_pixels` and `no_cache`. `max_pixels` caps the
**output** SBS area in pixels - synthesis still runs at full resolution and
the result is downscaled just before encoding, so callers with a fixed
display budget (a headset, say) do not receive a 50 MP JPEG they will only
shrink. `no_cache=1` ignores SHARP predictions cached on disk and runs the
network again (for timing comparisons); loaded models stay resident.

### Caches

Two things grow on disk: SHARP predictions in `.sharp_cache/` (one `.npz`
per photo, 20-60 MB each) and web UI results in `outputs/`. The "Cache &
memory" panel in the web UI shows both sizes, clears either one, and sets
the limits (`sharp_cache_max_mb`, least recently used entries are evicted
after each prediction, default 2048; `outputs_keep`, latest N results,
default 10; 0 disables either). The same panel sets how long the resident
SHARP predictor survives idle (`sharp_idle_s`, default 0 keeps it loaded)
and can unload every model right away. The "Disable cache" checkbox next to
Generate skips reading `.sharp_cache/` for that run - the stages panel then
labels the prediction step "cache off" instead of "cached".

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

## Stereo methods

Pick a method in the web UI, with `?method=name` on `/transform`, or
`--method name` on the CLI. The saved choice in `settings.json` is what
external HTTP callers get. Two families:

**Depth-map methods** run the pipeline above: monocular depth, BiRefNet
healing, forward warp, inpaint. Fast (a few seconds), commercial-safe
weights (Apache 2.0).

| Method | Label | Input | What it does |
|--------|-------|-------|--------------|
| `per_eye_inpaint` | Per-Eye Inpaint | depth map | Inverse cv2.remap warp with stretch-based occlusion detection. Each eye is inpainted independently - fast but fills may differ between eyes. |
| `fullres_warp` | Per-Eye Inpaint (Full-Res) (Deprecated) | depth map | Same warp strategy as Per-Eye Inpaint but at full source resolution. Downscales only for inpainting, then composites fills back via a feathered occlusion mask. Preserves maximum sharpness for the ~97% of pixels that are directly warped. |
| `bg_plate_fill` | Background Plate (Deprecated) | depth map | Hybrid z-buffer + cv2.remap warp with stereo-consistent background-plate fill. Inpaints foreground out of the source once, warps the clean plate into both eyes. Two-pass LaMa + Telea with colour correction and sharpening. |
| `routed_fill` | Width-Routed Fill (Deprecated) | depth map | Hybrid z-buffer warp with width-routed disocclusion fill. Narrow strips (≤ threshold) are mirror-filled from adjacent background pixels (CPU, zero dark bias). Wide regions use stereo-consistent bg-plate with anisotropic banded LaMa (independent horizontal crops limit FFC vertical context) + Poisson seam blending. Unilateral mask dilation toward foreground only. |
| `direct_fill` | Direct Anisotropic Fill (Deprecated) | depth map | Hybrid z-buffer warp with width-routed disocclusion fill. Narrow strips (≤ threshold) are mirror-filled from adjacent background pixels (CPU, zero dark bias). Wide regions are filled directly in each eye view via per-strip anisotropic crop inpainting (no background plate). Each strip gets a tight horizontal crop - LaMa sees only local texture. Poisson seam blending + unilateral mask dilation. |
| `clean_fill` | Clean Fill / AOT-GAN (Deprecated) | depth map | Hybrid z-buffer warp with per-eye AOT-GAN inpainting. AOT-GAN uses dilated convolutions (no FFC dark bias, no global texture cloning) so no width routing, banding, or bg-plate workarounds are needed. Unilateral mask dilation toward foreground only + Poisson seam blending. |
| `combo_fill` | Combo Fill / AOT-GAN (Deprecated) | depth map | Best of Methods 4 + 5: per-strip anisotropic crop inpainting with AOT-GAN. Narrow strips filled by CPU mirror (exact texture). Wide strips get tight per-component crops fed to AOT-GAN (no dark bias, no FFC global cloning, fast on small crops). Poisson seam blending + unilateral mask dilation. |
| `ldi_inpaint` | LDI Context-Aware Inpaint (Deprecated) | depth map | Context-aware layered depth inpainting (Shih et al., CVPR 2020). Uses three specialised partial-convolution networks to hallucinate depth edges, inpaint depth, and inpaint colour behind foreground objects. Produces a neural background plate that is warped into both eyes for stereo-consistent fill. |
| `iterative_fill` | Iterative Local Patch (Deprecated) | depth map | Per-eye disocclusion fill using depth-sorted iterative local patching. Breaks holes into connected components, sorts back-to-front by depth, and inpaints each patch locally with LaMa so each fill becomes context for the next. |

**SHARP methods** skip the depth map: Apple SHARP predicts a 3D Gaussian
splat of the photo (including a hallucinated layer behind occluders) and
both eyes are rendered from two virtual cameras 63 mm apart. The SHARP
weights are research-only (Apple ML Research license) - opt-in, never
bundled, download shows the license. `sharp_detail` and later keep the
photo's own pixels wherever the original camera saw the surface, so only
the disocclusion band beside depth edges is generated.

| Method | Label | Input | What it does |
|--------|-------|-------|--------------|
| `sharp_depth` | SHARP Depth | SHARP splat | Renders SHARP's 3D Gaussian scene to extract a high-quality metric depth map, then feeds it through the proven warp+inpaint pipeline (per_eye_inpaint). Better depth than Depth Anything, familiar warp look. Research-only license. |
| `sharp_mesh` | SHARP Mesh | SHARP splat | Renders SHARP's depth as a forward-splatted mesh. Triangles fill small gaps naturally; large disocclusions get stretched from neighbours (no AI inpainting). Sharp edges at depth boundaries. Research-only license. |
| `sharp_splat` | SHARP Splat | SHARP splat | 3D Gaussian splat via Apple SHARP, rendered from two virtual cameras 63 mm apart. True parallax for every object, no inpainting step. Slightly soft (SHARP works at 1536^2). Research-only license. |
| `sharp_taichi` | SHARP Taichi | SHARP splat | Same EWA Gaussian splatting as sharp_splat but compositing runs on Metal/GPU via taichi (5-10x faster). Falls back to the torch renderer if taichi is not installed. Research-only license. |
| `sharp_splat_full` | SHARP Splat (full taichi) | SHARP splat | Same z-buffer EWA look as sharp_splat, but the entire render path - projection and compositing - runs as Taichi kernels, no torch in the renderer. Both eyes render in about a second. Falls back to torch when taichi is unavailable. Research-only license. |
| `sharp_alpha_full` | SHARP Alpha (full taichi) | SHARP splat | Depth-sorted alpha compositing (the SHARP Alpha look) at standard 1536 resolution, pure splat colour, with the whole render path in Taichi. Fastest render step of the SHARP methods. Falls back to torch when taichi is unavailable. Research-only license. |
| `sharp_detail` | SHARP Detail | SHARP splat | Same SHARP splat geometry as sharp_splat, but colour is re-sampled from the original photo wherever the original camera could see that surface (most pixels; splat colour only in the disoccluded band). Full photo sharpness, same 3D geometry. Research-only license. |
| `sharp_hires` | SHARP Hi-res Detail | SHARP splat | sharp_detail with SHARP run at 2688^2 instead of 1536^2 (1344^2 Gaussian grid, 3.6 M Gaussians): tighter silhouettes and a visibly sharper disocclusion band. ~5x slower prediction (about 95 s on an M-series Mac), 3x memory. Experimental - outside the model's training resolution. Research-only license. |
| `sharp_alpha` | SHARP Alpha | SHARP splat | sharp_hires rendered with proper 3DGS compositing: Gaussians depth-sorted per pixel and alpha-blended front to back, median depth. Cleanest silhouettes and sharpest disocclusion band of the SHARP methods. Slow: about 2 min per photo on an M-series Mac (the per-pixel sort runs in torch). Research-only license. |
| `sharp_alpha_taichi` | SHARP Alpha (taichi) | SHARP splat | Same output as sharp_alpha, rendered by a taichi tile rasteriser on Metal/GPU: the render step drops from about 2 min to under a second, leaving SHARP prediction (~90 s at 2688^2) as the only cost. Needs taichi (pip install taichi, Python <= 3.13); falls back to the torch renderer otherwise. Research-only license. |

Recommended: `sharp_alpha_taichi` when taichi is installed (SHARP
prediction ~90 s at 2688^2, render under a second), else `sharp_hires`
(~2 min total). `sharp_alpha` gives the same picture as `sharp_alpha_taichi`
with a pure-torch renderer (~2 min extra). `sharp_alpha*` use proper 3DGS
compositing (Gaussians depth-sorted per pixel, alpha-blended front to
back, median depth); the other SHARP methods use a faster z-buffer
approximation that is softer at silhouettes. Taichi 1.7 has wheels for
Python 3.9-3.13 only; on a 3.14 venv the taichi methods use the torch
renderer instead - the result line in the web UI shows which renderer
actually ran. The SHARP predictor stays loaded between photos (about
8 s saved per photo after the first) and unloads after 60 s idle.

The prototypes these came from, with side-by-side outputs, are in
`../stereo-experiments/README.md` (exps 13-21).
