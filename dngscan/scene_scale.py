# SPDX-License-Identifier: GPL-3.0-or-later
"""Compatibility helpers for :class:`~dngscan.models.SceneScaleContract`.

Name the multiplies that already exist without
reordering them. Pixel maths must stay identical to
``stored / bundle.scene_scale * bundle.exposure_gain``.
"""
from __future__ import annotations

from dataclasses import replace

from .models import RawBundle, SceneScaleContract
from .tone import compute_exposure_gain, exposure_mode_for_tone_core


def calibration_confidence_for_mode(decoder: str, scale_mode: str | None) -> str:
    """Label how trustworthy the decoder comparison scale is for HDR claims."""
    decoder = str(decoder or "libraw").lower()
    mode = str(scale_mode or "").lower()
    if decoder == "libraw":
        # LibRaw black/white + BaselineExposure divisor is a fixed file recipe.
        return "calibrated"
    if mode == "aligned":
        # Per-file green-median comparison: preserves ratios, not absolute radiometry.
        return "relative"
    if mode == "unity":
        return "decoder-native"
    if mode == "measured":
        # Legacy fixed Sigma-fp fit — camera-specific, not per-image AE.
        return "relative"
    return "relative"


def scene_scale_contract_from_bundle(
    bundle: RawBundle,
    *,
    user_ev: float | None = None,
    exposure_gain: float | None = None,
    tone_core: str = "agx",
) -> SceneScaleContract:
    """Compatibility constructor: express the current scale×gain product.

    Passing ``user_ev`` rebuilds the mid-gray × EV split. Passing only
    ``exposure_gain`` (or neither, reading ``bundle.exposure_gain``) splits that
    combined gain against the EV0 mid-gray anchor so the product is unchanged.
    """
    mid = float(compute_exposure_gain(exposure_mode_for_tone_core(tone_core), 0.0))
    if user_ev is not None:
        user = float(2.0 ** float(user_ev))
        fixed = mid
    else:
        combined = float(
            bundle.exposure_gain if exposure_gain is None else exposure_gain
        )
        fixed = mid
        user = combined / mid if mid != 0.0 else combined

    decoder = str(getattr(bundle, "scene_decoder", "libraw") or "libraw")
    scale_mode = str(
        getattr(bundle, "scene_scale_mode", None)
        or ("libraw" if decoder == "libraw" else "aligned")
    )
    # Both normal paths fold BaselineExposure into scene_scale. The explicit flag only
    # becomes true when an older Core Image API cannot clear it inside CIRAWFilter.
    baseline_baked = bool(getattr(bundle, "baseline_exposure_baked_in", False))
    return SceneScaleContract(
        storage_scale=float(bundle.scene_scale),
        decoder_calibration_gain=1.0,
        baseline_render_gain=1.0,
        fixed_midgray_gain=fixed,
        user_ev_gain=user,
        baseline_baked_in=baseline_baked,
        scale_mode=scale_mode,
        calibration_confidence=calibration_confidence_for_mode(decoder, scale_mode),
    )


def with_intent_exposure(
    bundle: RawBundle,
    *,
    user_ev: float,
    tone_core: str = "agx",
) -> RawBundle:
    """Return a copy carrying intent exposure without mutating the caller's bundle.

    Preview/export must not share mutable ``exposure_gain`` across concurrent work.
    """
    contract = scene_scale_contract_from_bundle(
        bundle, user_ev=user_ev, tone_core=tone_core
    )
    return replace(bundle, exposure_gain=contract.legacy_exposure_gain)
