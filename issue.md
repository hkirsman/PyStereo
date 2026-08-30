## SHARP Gaussian splat stereo synthesis

Add stereo synthesis via Apple SHARP - turn a single photo into a 3D Gaussian splat, then render that splat from two virtual cameras 63 mm apart. No depth map, no inpainting step - the second eye sees whatever the splat contains behind the subject (SHARP predicts a hidden second layer itself).

### Methods

| Method | Description |
|--------|-------------|
| `sharp_splat` | Pure splat renders from both cameras. Correct parallax everywhere, slightly soft (SHARP works at 1536^2). |
| `sharp_detail` | Same geometry, but colour is re-sampled from the original photo wherever the original camera could see that surface (~95% of pixels). Full photo sharpness, splat colour only in the disoccluded band. |
| `sharp_hires` | `sharp_detail` with SHARP run at 2688^2 (1344^2 Gaussian grid, 3.6 M Gaussians). Tighter silhouettes. ~5x slower prediction. Experimental - outside SHARP's training resolution. |
| `sharp_alpha` | `sharp_hires` with proper 3DGS alpha compositing (depth-sorted per pixel, front-to-back blend). Cleanest silhouettes. ~2 min per photo on M-series (per-pixel sort in torch). |
| `sharp_alpha_taichi` | Same output as `sharp_alpha`, rendered by a taichi tile rasteriser on Metal/GPU. Render step drops from ~2 min to under a second. Needs `pip install taichi` (Python <= 3.13); falls back to torch otherwise. |
| `sharp_depth` | Renders SHARP's scene to extract a metric depth map, then feeds it through the existing warp+inpaint pipeline (`per_eye_inpaint`). Better depth than Depth Anything, familiar warp look. |

`sharp_mesh` and `sharp_taichi` are also included but marked deprecated (forward mesh rendering and zbuf-mode taichi, respectively - superseded by the alpha methods above).

All SHARP methods are `needs_depth=False` (except `sharp_depth` which uses depth as an intermediate) - they bypass the depth model download entirely.

### How it works

1. **Predict**: load the SHARP checkpoint, run `predict_image` (EXIF focal length gives metric f_px), save means (float32) and scales/quaternions/colours/opacities (float16) to a cached `.npz`.
2. **Render**: a custom EWA Gaussian rasteriser in torch (MPS/CUDA/CPU) since ml-sharp's gsplat renderer is CUDA-only. Two pinhole cameras at +-baseline/2, parallel axes, converged on the subject by principal-point shift. Convergence distance = median rendered depth inside the BiRefNet subject mask (10th percentile fallback without a subject).
3. **Detail transfer** (sharp_detail/hires/alpha variants): each eye pixel is reprojected into the original camera and bicubic-sampled from the photo, except where that point is occluded in the original view - those pixels keep the splat colour.

### Requirements

- `ml-sharp` git submodule (Apple ML Research, research-only license)
- SHARP checkpoint at `~/.cache/torch/hub/checkpoints/sharp_2572gikvuh.pt` (2.8 GB, downloaded via the UI)
- torch with MPS (Apple Silicon), CUDA, or CPU
- Optional: `pip install taichi` for GPU-accelerated rendering in alpha modes

### Known limitations

- The disoccluded band beside the subject comes from SHARP's hallucinated second layer (768^2) - visibly softer than the photo around it. Narrower baseline hides it; no way to get real texture there from a single photo.
- SHARP's metric scale depends on the EXIF focal length; disparity numbers are only as accurate as that guess.
- ~20 s prediction + ~25 s render per photo on M-series Mac (alpha modes ~2 min without taichi). A gsplat/CUDA or Metal renderer would bring the render to well under a second.
- TODO (`sharp_detail` / `sharp_hires`): lower half of the image often looks broken while the upper part looks pretty good - dig into detail-transfer / reprojection. Also remeasure peak memory for `sharp_hires` on M4.
- TODO (`sharp_depth`): results look similar to Per-Eye Inpaint - compare which is actually better and whether we need both.
