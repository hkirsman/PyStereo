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
