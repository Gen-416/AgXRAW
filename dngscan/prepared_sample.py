# SPDX-License-Identifier: GPL-3.0-or-later
"""Job-local source samples shared by plan compilation and exposure probes."""
from dataclasses import dataclass, replace
from typing import Any

from ._deps import np
from .sampling import sample_indices


@dataclass(frozen=True)
class PreparedSceneSample:
    """One bounded population, in storage units and original source order.

    The plan compiler already shares its transformed rows internally. Exposure
    probes must still apply their gain before lens/scene transforms at every EV;
    caching a transformed zero-EV image would move float32 rounding points.
    This object lives only through one AutoEV call, never in a global cache.
    """
    rgb: Any
    masks: Any | None
    source_indices: Any | None

    @classmethod
    def from_bundle(cls, bundle):
        stored = getattr(bundle, "_tone_plan_sample", None)
        if stored is not None:
            # GUI proxies already carry the full source population and matching
            # masks. Resampling the proxy would erase small source highlights.
            return cls(np.asarray(stored),
                       getattr(bundle, "_tone_plan_sample_masks", None), None)
        scene = bundle.scene_rec2020_render
        flat = scene.reshape(-1, scene.shape[-1])
        indices = sample_indices(flat.shape[0])
        rgb = flat[indices, :3]
        masks = None
        if getattr(bundle, "clip_masks", None) is not None:
            from .retreat import clip_masks_for_render

            masks = clip_masks_for_render(bundle, scene.shape[:2])[indices]
            masks.flags.writeable = False
        rgb.flags.writeable = False
        indices.flags.writeable = False
        return cls(rgb, masks, indices)

    def bind(self, bundle):
        """Borrow the source population on a private bundle, not the capture."""
        return replace(bundle, _tone_plan_sample=self.rgb,
                       _tone_plan_sample_masks=self.masks)
