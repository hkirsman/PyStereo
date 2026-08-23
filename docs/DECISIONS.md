# Architectural Decisions

## Depth estimation: trust the model, don't post-process

**Decision:** Use Depth Anything V2 output as-is. Switch model size (Small/Base/Large) to get better quality rather than trying to fix depth maps after the fact.

**Context:** DA V2 Small produces flat foreground depth on some subjects (the "cardboard cutout" effect - a person pops out as a uniform slab). We tried amplifying internal depth variation using the BiRefNet foreground mask (stretch depth range around midpoint within masked region). The stretch had no visible effect because Small's foreground depth is genuinely flat - there's nothing to amplify. The guided filter then smoothed what little signal remained.

**Alternatives rejected:**
- Per-subject FG depth stretch (implemented, tested, removed) - can't amplify variation that doesn't exist
- Synthetic depth from mask distance transform - invents geometry rather than estimating it, would look wrong on non-convex subjects

**Trade-off:** Base/Large produce noticeably better foreground depth but are CC-BY-NC-4.0 (non-commercial) and larger downloads (400 MB / 1.3 GB vs 95 MB). Small (Apache-2.0) is the default for size and license reasons. Users pick their own trade-off via the depth model selector.
