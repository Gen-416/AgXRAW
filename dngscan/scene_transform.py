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
import hashlib
import math
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
DECODER_ANCHOR_SCHEMA = "decoder-anchor-transport-3"
RAW9_SCOPE_SCHEMA = "raw9-anchor-scope-1"
RAW9_TAG_SCHEMA = "raw9-window-tag-1"
ANCHOR_RATIO_BOUNDS = (.65, 1.30)
ANCHOR_MAX_HOLDOUT_MEDIAN_ERROR = .03
ANCHOR_MAX_HOLDOUT_P95_ERROR = .06


def decoder_anchor_fingerprint() -> str:
    """Content identity of the currently installed anchor asset, including absence."""
    try:
        payload = DECODER_ANCHOR_TRANSPORT_JSON.read_bytes()
    except OSError:
        payload = b""
    return hashlib.sha256(payload).hexdigest()


def _decoder_transport_snapshot() -> tuple[str, dict]:
    try:
        payload = DECODER_ANCHOR_TRANSPORT_JSON.read_bytes()
    except OSError:
        payload = b""
    fingerprint = hashlib.sha256(payload).hexdigest()
    try:
        asset = json.loads(payload)
    except (ValueError, UnicodeError):
        asset = {}
    return fingerprint, asset if isinstance(asset, dict) else {}


def _validated_anchor_scope(raw: Any) -> dict | None:
    if not isinstance(raw, dict) or raw.get("schema") != RAW9_SCOPE_SCHEMA:
        return None
    keys = ("make", "model", "actual_decoder_token", "runtime", "configuration_id")
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in keys):
        return None
    if raw["actual_decoder_token"].split(".", 1)[0] != "9":
        return None
    identity = raw["configuration_id"]
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        return None
    return {"schema": RAW9_SCOPE_SCHEMA, **{key: raw[key] for key in keys}}


def decoder_anchor_scope(bundle: Any) -> dict | None:
    """Exact camera and actual RAW9 processing identity; missing proof is unmeasured."""
    if getattr(bundle, "scene_decoder", None) != "coreimage" or getattr(bundle, "scene_decoder_fallback", None):
        return None
    recipe = getattr(bundle, "scene_decoder_recipe", None)
    if not isinstance(recipe, dict) or recipe.get("schema") != "coreimage-decode-recipe-1":
        return None
    access = recipe.get("linear_access")
    fallback = recipe.get("fallback")
    if (not isinstance(access, dict) or access.get("status") != "parameter-state-verified"
            or not isinstance(fallback, dict) or fallback.get("used") is not False
            or getattr(bundle, "scene_decoder_version", None) != recipe.get("actual_decoder_token")
            or getattr(bundle, "scene_decoder_runtime", None) != recipe.get("runtime")):
        return None
    return _validated_anchor_scope({"schema": RAW9_SCOPE_SCHEMA,
        "make": getattr(bundle, "shot_make", None), "model": getattr(bundle, "shot_model", None),
        **{key: recipe.get(key) for key in ("actual_decoder_token", "runtime", "configuration_id")}})


def _qualified_anchor_entry(entry: Any) -> bool:
    if (not isinstance(entry, dict) or entry.get("status") != "qualified"
            or not isinstance(entry.get("id"), str) or not entry["id"]):
        return False
    validation = entry.get("validation") or {}
    source = entry.get("source") or {}
    if (not isinstance(validation, dict) or validation.get("status") != "passed"
            or validation.get("independent_holdout") is not True
            or validation.get("spatial_support") not in ("flat-target-rois", "validated-registration")):
        return False
    inputs = source.get("inputs") if isinstance(source, dict) else None
    if not isinstance(inputs, list) or len(inputs) < 4:
        return False
    hashes_valid = all(isinstance(row, dict) and isinstance(row.get("sha256"), str)
               and len(row["sha256"]) == 64
               and all(c in "0123456789abcdef" for c in row["sha256"]) for row in inputs)
    if not hashes_valid or len({row["sha256"] for row in inputs}) != len(inputs):
        return False
    if any(sum(row.get("split") == split for row in inputs) < 2 for split in ("train", "holdout")):
        return False
    transport = entry.get("transport")
    if not isinstance(transport, dict):
        return False
    classes = transport.get("per_class") or {}
    candidates = [transport.get("global"), *(classes.values() if isinstance(classes, dict) else ())]
    return any(_anchor_ratio(candidate) is not None for candidate in candidates)


def _anchor_ratio(record: Any) -> tuple[float, float] | None:
    if not isinstance(record, dict) or record.get("status") != "qualified":
        return None
    ratio = record.get("ratio_rg_bg")
    if not isinstance(ratio, (list, tuple)) or len(ratio) != 2:
        return None
    counts = record.get("independent_frames")
    if (not isinstance(counts, dict) or any(type(counts.get(split)) is not int or counts[split] < 2
                                          for split in ("train", "holdout"))):
        return None
    try:
        values = tuple(float(value) for value in ratio)
        for key, maximum in (("holdout_median_relative_error", ANCHOR_MAX_HOLDOUT_MEDIAN_ERROR),
                             ("holdout_p95_relative_error", ANCHOR_MAX_HOLDOUT_P95_ERROR)):
            errors = record.get(key)
            if not isinstance(errors, (list, tuple)) or len(errors) != 2:
                return None
            if any(not math.isfinite(float(error)) or not 0 <= float(error) <= maximum for error in errors):
                return None
    except (TypeError, ValueError, OverflowError):
        return None
    return values if all(math.isfinite(value) and ANCHOR_RATIO_BOUNDS[0] <= value <= ANCHOR_RATIO_BOUNDS[1]
                         for value in values) else None


def _select_decoder_anchor(scope: dict | None, asset: dict) -> tuple[dict | None, str]:
    if scope is None:
        return None, "camera-or-actual-raw9-recipe-unmeasured"
    if asset.get("schema") != DECODER_ANCHOR_SCHEMA or asset.get("version") != 3:
        return None, "legacy-or-invalid-anchor-asset"
    profiles = asset.get("profiles")
    if not isinstance(profiles, list):
        return None, "invalid-anchor-profiles"
    matched = [entry for entry in profiles if isinstance(entry, dict)
               and _validated_anchor_scope(entry.get("scope")) == scope and _qualified_anchor_entry(entry)]
    if len(matched) != 1:
        return None, "ambiguous-anchor-scope" if len(matched) > 1 else "exact-anchor-scope-unmeasured"
    return matched[0], "exact-qualified-anchor-scope"


def decoder_window_calibration(bundle: Any) -> dict:
    """Describe anchor qualification without changing pixels or blocking decoding."""
    fingerprint, asset = _decoder_transport_snapshot()
    scope = decoder_anchor_scope(bundle)
    entry, reason = _select_decoder_anchor(scope, asset)
    reference = getattr(bundle, "scene_decoder", None) == "libraw"
    profile_hash = (hashlib.sha256(json.dumps(entry, sort_keys=True, separators=(",", ":"),
                                            ensure_ascii=False).encode()).hexdigest() if entry else None)
    return {"status": "identity" if reference else "qualified" if entry else "unmeasured",
            "reason": "libraw-reference-windows" if reference else reason,
            "profile_id": entry.get("id") if entry else None,
            "profile_sha256": profile_hash, "asset_sha256": fingerprint, "scope": scope}


def window_transport_tag(bundle: Any) -> str:
    """Bind window transport to exact decoder/camera scope and asset contents."""
    decoder = str(getattr(bundle, "scene_decoder", "libraw") or "libraw")
    if decoder == "libraw":
        return "libraw"
    scope = decoder_anchor_scope(bundle)
    return json.dumps({"schema": RAW9_TAG_SCHEMA, "scope": scope,
                       "asset_sha256": decoder_anchor_fingerprint()}, sort_keys=True, separators=(",", ":"))


def decoder_window_ratios(scene_decoder: str, region_name: str) -> tuple[float, float] | None:
    """Qualified exact-scope window transport. Legacy/default tokens are identity."""
    try:
        tag = json.loads(scene_decoder)
    except (TypeError, ValueError):
        return None
    if not isinstance(tag, dict) or tag.get("schema") != RAW9_TAG_SCHEMA:
        return None
    fingerprint, asset = _decoder_transport_snapshot()
    if tag.get("asset_sha256") != fingerprint:
        return None
    entry, _ = _select_decoder_anchor(_validated_anchor_scope(tag.get("scope")), asset)
    if entry is None:
        return None
    transport = entry.get("transport") or {}
    if not isinstance(transport, dict):
        return None
    per_class = transport.get("per_class") or {}
    record = per_class.get(str(region_name)) if isinstance(per_class, dict) else None
    r = _anchor_ratio(record)
    if r is None:
        record = transport.get("global")
        r = _anchor_ratio(record)
    if r is None:
        return None
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
    von Kries transport of the anchor. Identity cases (daylight balance, missing or
    near-unity multipliers) return None on the libraw path; a non-libraw decoder gets
    an identity (1.0, 1.0, decoder) triple instead, so the decoder tag survives for
    decoder_window_ratios. Non-positive multipliers return None for every decoder."""
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


def window_transport(bundle: Any) -> tuple | None:
    """The transport the prefeed windows must follow for this bundle's pixels.

    Preferred form carries the full hot-WB window matrix (see
    raw_io.wb_window_transport_matrix_rec2020) plus the decoder tag; when the
    calibration matrices are unavailable the diagonal von Kries ratios remain
    as the declared approximation fallback.
    """
    from . import raw_io

    tag = window_transport_tag(bundle)
    matrix = raw_io.wb_window_transport_matrix_rec2020(bundle)
    if matrix is not None:
        m9 = tuple(float(v) for v in np.asarray(matrix, dtype=np.float64).reshape(-1))
        return ("matrix", m9, tag)
    return wb_adaptation_ratios(
        getattr(bundle, "wb_mode", "camera"),
        getattr(bundle, "applied_wb", None) or getattr(bundle, "camera_wb", None),
        getattr(bundle, "daylight_wb", None),
        tag,
    )


def _projective_window_transport(
    mu: Any, cov: Any, m9: tuple[float, ...]
) -> tuple[Any, Any]:
    """Move a chroma window (mu, cov) through the full 3x3 balance transform.

    Chroma lives at v = [r/g, 1, b/g]; the pixels move by v' = M v, so the
    window center is the projective image mu' = (v'_R/v'_G, v'_B/v'_G) and the
    covariance follows the Jacobian of that map at mu: cov' = J cov J^T. The
    old two-ratio shortcut is exact only for a channel-diagonal M; the hot-WB
    matrix mixes channels.
    """
    m = np.asarray(m9, dtype=np.float64).reshape(3, 3)
    x, z = float(mu[0]), float(mu[1])
    v = np.array([x, 1.0, z], dtype=np.float64)
    vp = m @ v
    g = vp[1]
    if not np.isfinite(g) or abs(g) <= 1e-9:
        return mu, cov
    mu_p = np.array([vp[0] / g, vp[2] / g], dtype=np.float64)
    j = np.empty((2, 2), dtype=np.float64)
    j[0, 0] = (m[0, 0] * g - vp[0] * m[1, 0]) / (g * g)
    j[0, 1] = (m[0, 2] * g - vp[0] * m[1, 2]) / (g * g)
    j[1, 0] = (m[2, 0] * g - vp[2] * m[1, 0]) / (g * g)
    j[1, 1] = (m[2, 2] * g - vp[2] * m[1, 2]) / (g * g)
    cov_p = j @ np.asarray(cov, dtype=np.float64) @ j.T
    return (
        mu_p.astype(np.float32),
        cov_p.astype(np.float32),
    )


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
    wb_adapt: tuple | None,
) -> tuple[Any, Any]:
    """Compile immutable Gaussian constants with the reference operation order.

    ``wb_adapt`` is either the legacy diagonal ratio pair, or the full-matrix
    form ("matrix", m9, dec_ratios_or_None): projective center transport plus a
    Jacobian covariance push, with the measured decoder chroma ratios applied
    afterwards as their own approximation layer.
    """
    mu = np.asarray(mu_rg_bg, dtype=np.float32)
    cov = np.asarray(cov_rg_bg, dtype=np.float32) * np.float32(max(scale, EPS) ** 2)
    if wb_adapt is not None and len(wb_adapt) == 3 and wb_adapt[0] == "matrix":
        mu, cov = _projective_window_transport(mu, cov, wb_adapt[1])
        dec = wb_adapt[2]
        if dec is not None:
            scale_vec = np.asarray(dec, dtype=np.float32)
            mu = mu * scale_vec
            cov = cov * np.outer(scale_vec, scale_vec).astype(np.float32)
        mu = mu.astype(np.float32)
        cov = cov.astype(np.float32)
    elif wb_adapt is not None:
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
    if wb_adapt is None:
        transport = None
    elif len(wb_adapt) == 3 and wb_adapt[0] == "matrix":
        transport = wb_adapt
    else:
        transport = (float(wb_adapt[0]), float(wb_adapt[1]))
    mu, inv_cov = _compiled_gaussian_parameters(
        mu_rg_bg, cov_rg_bg, float(scale), transport
    )
    d = chroma - mu[None, :]
    mahal = d[:, 0] * (inv_cov[0, 0] * d[:, 0] + inv_cov[0, 1] * d[:, 1])
    mahal += d[:, 1] * (inv_cov[1, 0] * d[:, 0] + inv_cov[1, 1] * d[:, 1])
    return np.exp(np.clip(-0.5 * mahal, -80.0, 0.0)).astype(np.float32, copy=False)


def _compose_transport(
    wb_adapt: tuple | None, region_name: str
) -> tuple | None:
    """Combine the WB window transport with the measured decoder transport."""
    if wb_adapt is not None and len(wb_adapt) == 3 and wb_adapt[0] == "matrix":
        decoder = str(wb_adapt[2]) if wb_adapt[2] else None
        dec = decoder_window_ratios(decoder, region_name) if decoder else None
        return ("matrix", wb_adapt[1], dec)
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
        # double-count. Each component transports individually through the composed
        # window transport (full-matrix projective when the hot-WB matrix is
        # available, diagonal von Kries ratios as the fallback).
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
        # S3: inside an outer chunk worker the fair share is 1 — run the
        # serial oracle instead of stacking the region pool on the render pool
        and __import__("dngscan.cpu_budget", fromlist=["current_inner"]).current_inner() > 1
    ):
        from .cpu_budget import ordered_budget_map
        # The shared executor's width is not this render's allowance. Bound its
        # submitted work to the actual share and retain declaration-order sums.
        weights = ordered_budget_map(
            _REGION_POOL, region_weight, preset.regions, max_workers=_REGION_WORKERS)
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
