# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-linear pre-AgX colour transforms.

This layer sits after camera colour interpretation and before AgX.  It is not a
display look: the operator keeps scene-linear values scene-linear, leaves the
neutral axis unchanged, and only blends a constrained 3x3 matrix inside soft
chromaticity windows.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._deps import np

EPS = 1e-8
SCENE_TRANSFORM_REGION_PARALLEL_MIN_PIXELS = 64 * 1024
_REGION_WORKERS = min(8, max(2, (os.cpu_count() or 4) - 2))
_REGION_POOL = ThreadPoolExecutor(
    max_workers=_REGION_WORKERS, thread_name_prefix="dngscan-scene-region"
)
SCENE_TRANSFORM_PRESETS_JSON = Path(__file__).with_name("scene_transform_presets.json")


@dataclass(frozen=True)
class SceneTransformComponent:
    """One Gaussian of a (possibly multi-modal) chromaticity window."""

    mu_rg_bg: tuple[float, float]
    cov_rg_bg: tuple[tuple[float, float], tuple[float, float]]
    weight: float = 1.0


@dataclass(frozen=True)
class SceneTransformRegion:
    name: str
    matrix: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    mu_rg_bg: tuple[float, float]
    cov_rg_bg: tuple[tuple[float, float], tuple[float, float]]
    scale: float = 2.5
    strength: float = 1.0
    # Calibration confidence in [0,1]: how much of the fp->target divergence the fitted
    # matrix actually explains on its material class (from the calibrator's error report).
    # Folded into the effective region weight so poorly-constrained fits act gently.
    confidence: float = 1.0
    # Mixture window: within-scene illumination (sun/shade/tungsten) moves a material's
    # chromaticity further than camera differences do, so one Gaussian cannot cover a
    # material across lighting. When components are present they replace the scalar
    # mu/cov; the region weight is the MAX over components (never the sum, so overlap
    # cannot double-count). Empty tuple = legacy single-Gaussian behavior.
    components: tuple[SceneTransformComponent, ...] = ()


@dataclass(frozen=True)
class SceneTransformPreset:
    name: str
    label: str
    illuminant: str
    working_space: str
    regions: tuple[SceneTransformRegion, ...]
    note: str = ""


def _component_from_dict(raw: dict[str, Any]) -> SceneTransformComponent:
    return SceneTransformComponent(
        mu_rg_bg=tuple(float(v) for v in raw["mu_rg_bg"]),  # type: ignore[arg-type]
        cov_rg_bg=tuple(tuple(float(v) for v in row) for row in raw["cov_rg_bg"]),  # type: ignore[arg-type]
        weight=float(raw.get("weight", 1.0)),
    )


def _region_from_dict(name: str, raw: dict[str, Any]) -> SceneTransformRegion:
    components: tuple[SceneTransformComponent, ...] = ()
    raw_components = raw.get("components")
    if isinstance(raw_components, list):
        parsed = []
        for item in raw_components:
            if isinstance(item, dict):
                try:
                    parsed.append(_component_from_dict(item))
                except (KeyError, TypeError, ValueError):
                    continue
        components = tuple(parsed)
    return SceneTransformRegion(
        name=str(raw.get("name", name)),
        matrix=tuple(tuple(float(v) for v in row) for row in raw["matrix"]),  # type: ignore[arg-type]
        mu_rg_bg=tuple(float(v) for v in raw["mu_rg_bg"]),  # type: ignore[arg-type]
        cov_rg_bg=tuple(tuple(float(v) for v in row) for row in raw["cov_rg_bg"]),  # type: ignore[arg-type]
        scale=float(raw.get("scale", 2.5)),
        strength=float(raw.get("strength", 1.0)),
        confidence=float(raw.get("confidence", 1.0)),
        components=components,
    )


def _preset_from_dict(name: str, raw: dict[str, Any]) -> SceneTransformPreset:
    regions_raw = raw.get("regions", [])
    regions = tuple(_region_from_dict(str(i), r) for i, r in enumerate(regions_raw) if isinstance(r, dict))
    return SceneTransformPreset(
        name=str(raw.get("name", name)),
        label=str(raw.get("label", name)),
        illuminant=str(raw.get("illuminant", "")),
        working_space=str(raw.get("working_space", "Rec2020")),
        regions=regions,
        note=str(raw.get("note", "")),
    )


def _load_presets() -> dict[str, SceneTransformPreset]:
    presets: dict[str, SceneTransformPreset] = {}
    try:
        raw = json.loads(SCENE_TRANSFORM_PRESETS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    transforms = raw.get("transforms", raw) if isinstance(raw, dict) else {}
    if isinstance(transforms, dict):
        for name, item in transforms.items():
            if not isinstance(name, str) or not isinstance(item, dict):
                continue
            try:
                preset = _preset_from_dict(name, item)
            except (KeyError, TypeError, ValueError):
                continue
            if preset.regions:
                presets[name] = preset
    return presets


SCENE_TRANSFORMS: dict[str, SceneTransformPreset] = _load_presets()

DECODER_ANCHOR_TRANSPORT_JSON = Path(__file__).with_name("decoder_anchor_transport.json")


def _load_decoder_transport() -> dict:
    try:
        return json.loads(DECODER_ANCHOR_TRANSPORT_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_DECODER_TRANSPORT = _load_decoder_transport()


def window_transport_tag(bundle: Any) -> str:
    """Opaque decoder+camera token for the window transport ('libraw' = identity)."""
    decoder = str(getattr(bundle, "scene_decoder", "libraw") or "libraw")
    if decoder == "libraw":
        return "libraw"
    model = str(getattr(bundle, "shot_model", "") or "").strip()
    return f"{decoder}|{model}" if model else decoder


def decoder_window_ratios(scene_decoder: str, region_name: str) -> tuple[float, float] | None:
    """Measured chromaticity transport moving a calibration window into the given
    decoder's reference frame (see tools/calibrate_raw9_anchors.py). None = identity.

    The prefeed windows are calibrated against LibRaw-decoded responses; RAW 9
    realises the same declared balance through Apple's own calibration and interprets
    hue regions differently (measured global B/G x0.82 on the fp corpus, skin
    shifting hardest). Windows follow the pixels; matrices and pixels are untouched.
    """
    token = str(scene_decoder)
    decoder, _, model = token.partition("|")
    scopes = _DECODER_TRANSPORT.get(decoder)
    if not isinstance(scopes, dict):
        return None
    # v2 layout: decoder -> camera scope -> transport; per-camera wins, else default.
    if "global_ratio_rg_bg" in scopes:
        entry = scopes  # v1 flat layout
    else:
        entry = scopes.get(model) if model else None
        if not isinstance(entry, dict):
            entry = scopes.get("default")
    if not isinstance(entry, dict):
        return None
    per_class = entry.get("per_class", {})
    ratio = None
    if isinstance(per_class, dict):
        cls = per_class.get(str(region_name))
        if isinstance(cls, dict):
            ratio = cls.get("ratio_rg_bg")
    if ratio is None:
        ratio = entry.get("global_ratio_rg_bg")
    if not ratio:
        return None
    r = (float(ratio[0]), float(ratio[1]))
    return None if abs(r[0] - 1.0) < 1e-4 and abs(r[1] - 1.0) < 1e-4 else r
SCENE_TRANSFORM_CHOICES = ("none",) + tuple(SCENE_TRANSFORMS)


def scene_transform_label(name: str) -> str:
    if name == "none":
        return "无"
    preset = SCENE_TRANSFORMS.get(name)
    return preset.label if preset is not None else name.replace("_", " ")


def validate_scene_transform(name: str) -> str:
    if name == "none" or name in SCENE_TRANSFORMS:
        return name
    raise ValueError(f"未知 scene transform：{name}")


def wb_adaptation_ratios(
    wb_mode: str,
    applied_wb: list[float] | None,
    daylight_wb: list[float] | None,
    scene_decoder: str = "libraw",
) -> tuple | None:
    """(R/G, B/G) chromaticity transport from the calibration balance to the applied one.

    Region anchors are calibrated under a daylight-balanced render (the preset's D55 is
    approximated by libraw's daylight multipliers). When the export uses a different
    balance (AsShot), a surface that sat at chromaticity (rg, bg) in the calibration
    render sits at ~(rg*rR, bg*rB) now, with r = G-normalized(applied/daylight) — a
    von Kries transport of the anchor. Returns None (identity) for the daylight balance
    or when either multiplier set is unusable."""
    if wb_mode == "daylight":
        return (1.0, 1.0, scene_decoder) if scene_decoder != "libraw" else None
    if not applied_wb or not daylight_wb or len(applied_wb) < 3 or len(daylight_wb) < 3:
        return (1.0, 1.0, scene_decoder) if scene_decoder != "libraw" else None
    ar, ag, ab = (float(v) for v in applied_wb[:3])
    dr, dg, db = (float(v) for v in daylight_wb[:3])
    if min(ar, ag, ab, dr, dg, db) <= 0.0:
        return None
    r_r = min(5.0, max(0.2, (ar / ag) / (dr / dg)))
    r_b = min(5.0, max(0.2, (ab / ag) / (db / dg)))
    if abs(r_r - 1.0) < 1e-3 and abs(r_b - 1.0) < 1e-3:
        return (1.0, 1.0, scene_decoder) if scene_decoder != "libraw" else None
    if scene_decoder != "libraw":
        return (r_r, r_b, scene_decoder)
    return (r_r, r_b)


def _apply_matrix(rgb: Any, matrix: Any) -> Any:
    out = np.empty_like(rgb, dtype=np.float32)
    out[:, 0] = matrix[0, 0] * rgb[:, 0] + matrix[0, 1] * rgb[:, 1] + matrix[0, 2] * rgb[:, 2]
    out[:, 1] = matrix[1, 0] * rgb[:, 0] + matrix[1, 1] * rgb[:, 1] + matrix[1, 2] * rgb[:, 2]
    out[:, 2] = matrix[2, 0] * rgb[:, 0] + matrix[2, 1] * rgb[:, 1] + matrix[2, 2] * rgb[:, 2]
    return out


@lru_cache(maxsize=256)
def _compiled_gaussian_parameters(
    mu_rg_bg: tuple[float, float],
    cov_rg_bg: tuple[tuple[float, float], tuple[float, float]],
    scale: float,
    wb_adapt: tuple[float, float] | None,
) -> tuple[Any, Any]:
    """Compile immutable Gaussian constants with the reference operation order."""
    mu = np.asarray(mu_rg_bg, dtype=np.float32)
    cov = np.asarray(cov_rg_bg, dtype=np.float32) * np.float32(max(scale, EPS) ** 2)
    if wb_adapt is not None:
        scale_vec = np.asarray(wb_adapt, dtype=np.float32)
        mu = mu * scale_vec
        cov = cov * np.outer(scale_vec, scale_vec).astype(np.float32)
    try:
        inv_cov = np.linalg.inv(cov).astype(np.float32, copy=False)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov).astype(np.float32, copy=False)
    mu.setflags(write=False)
    inv_cov.setflags(write=False)
    return mu, inv_cov


@lru_cache(maxsize=128)
def _compiled_region_matrix(
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
) -> Any:
    result = np.asarray(matrix, dtype=np.float32)
    result.setflags(write=False)
    return result


def _gaussian_weight(
    chroma: Any,
    mu_rg_bg: tuple[float, float],
    cov_rg_bg: tuple[tuple[float, float], tuple[float, float]],
    scale: float,
    wb_adapt: tuple[float, float] | None,
) -> Any:
    transport = (
        None
        if wb_adapt is None
        else (float(wb_adapt[0]), float(wb_adapt[1]))
    )
    mu, inv_cov = _compiled_gaussian_parameters(
        mu_rg_bg, cov_rg_bg, float(scale), transport
    )
    d = chroma - mu[None, :]
    mahal = d[:, 0] * (inv_cov[0, 0] * d[:, 0] + inv_cov[0, 1] * d[:, 1])
    mahal += d[:, 1] * (inv_cov[1, 0] * d[:, 0] + inv_cov[1, 1] * d[:, 1])
    return np.exp(np.clip(-0.5 * mahal, -80.0, 0.0)).astype(np.float32, copy=False)


def _compose_transport(
    wb_adapt: tuple | None, region_name: str
) -> tuple[float, float] | None:
    """Combine the WB window transport with the measured decoder transport."""
    decoder = None
    base = wb_adapt
    if wb_adapt is not None and len(wb_adapt) == 3:
        base = (float(wb_adapt[0]), float(wb_adapt[1]))
        decoder = str(wb_adapt[2])
    dec = decoder_window_ratios(decoder, region_name) if decoder else None
    if dec is None:
        if base is not None and abs(base[0] - 1.0) < 1e-6 and abs(base[1] - 1.0) < 1e-6:
            return None
        return base
    if base is None:
        return dec
    return (base[0] * dec[0], base[1] * dec[1])


def _scene_chroma_and_signal(rgb: Any) -> tuple[Any, Any]:
    denom = np.maximum(rgb[:, 1], np.float32(EPS))
    chroma = np.empty((rgb.shape[0], 2), dtype=np.float32)
    chroma[:, 0] = rgb[:, 0] / denom
    chroma[:, 1] = rgb[:, 2] / denom
    return chroma, np.max(rgb, axis=1)


def _region_weight_from_chroma(
    chroma: Any,
    signal: Any,
    region: SceneTransformRegion,
    wb_adapt: tuple | None = None,
) -> Any:
    transport = _compose_transport(wb_adapt, region.name)
    if region.components:
        # Mixture window: MAX over components, not sum — overlapping lobes must not
        # double-count. Each component transports through von Kries individually.
        weight = np.zeros((chroma.shape[0],), dtype=np.float32)
        for comp in region.components:
            comp_w = _gaussian_weight(chroma, comp.mu_rg_bg, comp.cov_rg_bg, region.scale, transport)
            comp_w *= np.float32(min(1.0, max(0.0, comp.weight)))
            np.maximum(weight, comp_w, out=weight)
    else:
        weight = _gaussian_weight(chroma, region.mu_rg_bg, region.cov_rg_bg, region.scale, transport)
    return np.where(signal > np.float32(EPS), weight, np.float32(0.0))


def _region_weight(rgb: Any, region: SceneTransformRegion, wb_adapt: tuple | None = None) -> Any:
    chroma, signal = _scene_chroma_and_signal(rgb)
    return _region_weight_from_chroma(chroma, signal, region, wb_adapt)


def apply_scene_transform_rec2020(
    rgb: Any,
    transform: str = "none",
    strength: float = 1.0,
    wb_adapt: tuple[float, float] | None = None,
) -> Any:
    return _apply_scene_transform_rec2020(
        rgb, transform, strength, wb_adapt, parallel=True
    )


def _apply_scene_transform_rec2020_reference(
    rgb: Any,
    transform: str = "none",
    strength: float = 1.0,
    wb_adapt: tuple[float, float] | None = None,
) -> Any:
    """Serial bit-exact oracle retained for optimized-path validation."""
    return _apply_scene_transform_rec2020(
        rgb, transform, strength, wb_adapt, parallel=False
    )


def _apply_scene_transform_rec2020(
    rgb: Any,
    transform: str,
    strength: float,
    wb_adapt: tuple[float, float] | None,
    *,
    parallel: bool,
) -> Any:
    """Apply a soft chromaticity-windowed 3x3 scene transform in linear Rec.2020.

    `strength=0` is exact identity.  Multiple regions blend by normalizing only
    when their raw weights sum above one, so a single region keeps its full mask
    while overlap cannot double-apply competing matrices.  `wb_adapt` transports the
    calibrated chromaticity windows to the applied white balance (see
    wb_adaptation_ratios); None keeps the calibration-balance windows.
    """
    if transform == "none" or strength <= 0.0:
        return rgb
    preset = SCENE_TRANSFORMS.get(transform)
    if preset is None or not preset.regions:
        return rgb

    rgb32 = np.nan_to_num(rgb.astype(np.float32, copy=False), nan=0.0, posinf=1e6, neginf=0.0)
    shared_chroma = shared_signal = None
    if parallel:
        # Every region consumes the same scene chromaticity and signal mask. Building
        # those arrays once removes repeated full-frame divides/max reductions without
        # changing any region's Gaussian or blend arithmetic.
        shared_chroma, shared_signal = _scene_chroma_and_signal(rgb32)

    def region_weight(region: SceneTransformRegion) -> Any:
        eff = max(0.0, region.strength) * min(1.0, max(0.0, region.confidence))
        base = (
            _region_weight_from_chroma(
                shared_chroma, shared_signal, region, wb_adapt
            )
            if shared_chroma is not None
            else _region_weight(rgb32, region, wb_adapt)
        )
        return base * np.float32(eff)

    if (
        parallel
        and rgb32.shape[0] >= SCENE_TRANSFORM_REGION_PARALLEL_MIN_PIXELS
        and len(preset.regions) > 1
    ):
        futures = [_REGION_POOL.submit(region_weight, region) for region in preset.regions]
        # Collect in declaration order; the accumulation below therefore retains the
        # exact float32 order of the serial oracle.
        weights = [future.result() for future in futures]
    else:
        weights = [region_weight(region) for region in preset.regions]
    total = np.zeros((rgb32.shape[0],), dtype=np.float32)
    for w in weights:
        total += w
    norm = np.maximum(total, np.float32(1.0))

    out = rgb32.copy()
    global_strength = np.float32(max(0.0, float(strength)))
    for region, weight in zip(preset.regions, weights):
        w = (weight / norm * global_strength).astype(np.float32, copy=False)
        if not bool(np.any(w > 1e-6)):
            continue
        matrix = _compiled_region_matrix(region.matrix)
        mapped = _apply_matrix(rgb32, matrix)
        out += w[:, None] * (mapped - rgb32)
    return np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
