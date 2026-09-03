# Architectural Decisions

## Depth estimation: trust the model, don't post-process (2026-08-23)

**Decision:** Use Depth Anything V2 geometry as the primary signal - don't try to invent depth detail via post-processing. Switch model size (Small/Base/Large) for better quality instead.

**Context:** DA V2 Small produces flat foreground depth on some subjects (the "cardboard cutout" effect - a person pops out as a uniform slab). We tried amplifying internal depth variation using the BiRefNet foreground mask (stretch depth range around midpoint within masked region). The stretch had no visible effect because Small's foreground depth is genuinely flat - there's nothing to amplify. The guided filter then smoothed what little signal remained.

**Alternatives rejected:**
- Per-subject FG depth stretch (implemented, tested, removed) - can't amplify variation that doesn't exist
- Synthetic depth from mask distance transform - invents geometry rather than estimating it, would look wrong on non-convex subjects

**Trade-off:** Base/Large produce noticeably better foreground depth but are CC-BY-NC-4.0 (non-commercial) and larger downloads (400 MB / 1.3 GB vs 95 MB). Small (Apache-2.0) is the default for size and license reasons. Users pick their own trade-off via the depth model selector.

## Deprecated stereo methods stay in the codebase (2026-08-30)

**Decision:** Keep deprecated methods (marked `deprecated: ClassVar[bool] = True`) visible in the UI picker rather than deleting them. They appear at the bottom of the method list with a "(deprecated)" label.

**Context:** Stereo synthesis is an exploratory problem - there is no single correct approach, and the best method depends on the input photo, the viewer's tolerance for artifacts, and the rendering budget. Over the course of development we built and evaluated many strategies for filling the disoccluded band (the pixels revealed beside the subject when shifting to a second viewpoint): direct anisotropic fill, iterative local patch, width-routed fill, background plate compositing, AOT-GAN variants, LDI context-aware inpainting, full-resolution warp, forward mesh rendering, and multiple Gaussian splat rendering modes. Each one solved a specific class of artifact but introduced others. Keeping them around serves two purposes:

1. **Prevent re-discovery of dead ends.** Each deprecated method encodes a hypothesis that was tested and found inferior to the current default (`per_eye_inpaint` for depth-based, `sharp_alpha_taichi` for SHARP). Without the code, a future contributor seeing the same artifact would likely try the same approach again. The method's existence (and its deprecated flag) is the record that it was tried.
2. **Regression comparison.** When evaluating a new method or tuning the default, it is useful to re-run deprecated methods on the same photo to confirm the new approach is actually better. Deleting them removes the baseline.

**Why not just document them?** A prose description of "we tried anisotropic fill and it produced streaks at depth edges" is less useful than being able to select the method in the UI, see the streaks yourself, and understand why the current approach exists. The code is the documentation.

**Cost:** Each method is a single file (50-300 lines), auto-discovered by the method registry. They add no runtime overhead when not selected and no maintenance burden since they have no reason to change.

## SHARP predictor stays resident between photos (2026-09-02)

**Decision:** Cache the SHARP predictor in a module global after the first prediction and keep it there (`PYSTEREO_SHARP_IDLE_S` sets an idle unload timer; 0, the default, disables it).

**Context:** `predict_gaussians` used to load the 2.8 GB checkpoint and delete the model on every photo, adding ~6-8 s per prediction. sharp-local (the sibling repo wrapping the same model) loads it once per process, which made it look faster for consecutive photos even though raw inference speed is identical (~11 s at 1536 on an M4).

**Trade-off:** A resident predictor holds ~3 GB of unified memory. A 60 s idle timer was the original default so a machine left alone got the memory back, but it made the common case the slow one: photos sent from a headset arrive minutes apart, so nearly every one paid the reload. Measured on cup.JPEG (2048x1536, sharp_splat_full): 24.8 s with the predictor resident against 49.3 s after an unload, the reload alone accounting for 28 s. The timer now defaults to off; a memory-constrained machine can set `sharp_idle_s` in the web UI. A timer that fires while a prediction is running reschedules itself instead of unloading mid-inference.

## "Disable cache" only bypasses the disk cache (2026-09-02)

**Decision:** The web UI "Disable cache" checkbox (and `/transform` `no_cache=1`) makes `predict_gaussians` ignore an existing `.sharp_cache/` entry and run the network again. It does not unload the resident predictor, and the fresh prediction still overwrites the cache file.

**Context:** Timing comparisons between methods were confusing because a second run of the same photo skipped the 10-40 s SHARP prediction without saying so. The natural "disable cache" from browser devtools bypasses stored responses but does not restart the browser; the equivalent here is to skip the disk read while keeping model weights warm. Cold-start cost (checkpoint load, Taichi kernel compile) is a separate axis - benchmark after a warm-up run - and is surfaced as a "model load" note on the prediction step instead of being controlled by the checkbox.

**Why still write the file:** The splat renderers read the `.npz` from disk, so a bypassed prediction needs a path anyway. Overwriting the existing entry keeps a single code path and means the eviction bound (`sharp_cache_max_mb`, least recently used by mtime) is the only thing that decides what stays on disk.

## Full-taichi render path duplicates constants deliberately (2026-09-02)

**Decision:** `stereo/taichi_full.py` (the torch-free renderer behind the `*_full` methods) re-declares the render constants from `splat_render.py` instead of importing them.

**Context:** `splat_render.py` imports torch at module level. The point of the full-taichi path is a render path with no torch in it - usable later in a torch-free packaged viewer. Importing the constants would silently drag torch back in. The mirror is marked "keep in sync" in both files; the compositing kernels themselves are shared (`_taichi_kernels.py`), so outputs stay pixel-identical to the torch-projection paths.

**Reminder for frozen builds:** Taichi compiles kernels by reading Python source, so every kernel module must be at module scope and listed in the PyInstaller spec `datas` (`_taichi_kernels.py`, `_taichi_full_kernels.py`). A new kernels file that is not added to both specs will silently fall back to torch in the packaged app.

## Torchvision native libs must be forced into PyInstaller bundles (2026-09-03)

**Decision:** Both PyInstaller specs call `packaging/torchvision_binaries.py:collect_torchvision_binaries` and pass the results as `Analysis(binaries=...)`.

**Context:** Torchvision 0.29+ registers C++ ops (`torchvision::nms` and friends) from `_C_stable` / `image_stable` extension modules loaded with `torch.ops.load_library`, not via a normal `import torchvision._C`. PyInstaller's module graph therefore ships the pure-Python package and silently omits those `.pyd` / `.so` files (and their sibling DLLs on Windows). The packaged web UI then returns 500 on `/api/stereo-methods` with `operator torchvision::nms does not exist` as soon as anything imports `pystereo_core.stereo` (which pulls in `segment.py` → `torchvision.transforms`).

**Why not only lazy-import torchvision?** Listing methods would survive, but any depth method that runs BiRefNet would still crash. Bundling the native libs is the real fix; the helper is shared so batch and web stay in sync.
