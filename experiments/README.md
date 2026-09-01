# Experiments

Standalone proof-of-concept code, not wired into the app. Nothing here is
imported by `pystereo_core` - safe to break, safe to delete.

## taichi_full_render.py

Status: promoted into the app as the `sharp_taichi_full` method
(`pystereo_core/stereo/taichi_full.py` + `_taichi_full_kernels.py` +
`methods/sharp_taichi_full.py`). This file stays as the standalone,
torch-free reference version.

Renders a SHARP Gaussian scene (`.sharp_cache/*.npz`) to an SBS stereo JPEG
entirely with Taichi kernels - no PyTorch import at all. The production path
in `pystereo_core/stereo/` still projects Gaussians with torch and only
composites in Taichi; this moves the projection (quaternion -> 3D covariance
-> EWA 2D covariance -> screen position) into a Taichi kernel too.

```bash
.venv/bin/python3 experiments/taichi_full_render.py                # newest cached scene
.venv/bin/python3 experiments/taichi_full_render.py path/to/x.npz
```

Verified 2026-09-01 against the production renderer (`mode="alpha_taichi"`)
on a 1.18 M Gaussian scene: mean abs pixel diff 0.4/255, ~0.3 % of pixels
differ by more than 5/255 (fp ordering + JPEG round-trip), identical
convergence and hole stats. Both eyes render in ~1.2 s on M-series Metal.

Scope note: the SHARP *prediction* (photo -> Gaussians) cannot move to
Taichi - it is a ~2.8 GB ViT network and Taichi is a compute-kernel
language, not an inference runtime. Prediction stays in PyTorch; a
torch-free predictor would need a CoreML or ONNX export of the model
instead. What this experiment enables later: a render path (and possibly a
packaged viewer) that needs only numpy + taichi + cv2, no torch.
