# SPDX-License-Identifier: GPL-3.0-or-later
"""Body-specific dark/flat products. Explicit signed experiments, never auto ISP.

Observed means, qualified phase references and stable pixel residuals are
different products. Held-out measurements are required; a CSV dark mean alone
does not qualify any of them. No hot-pixel interpolation is authorized here.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
from .sensor_research import (PHASES, PersonalBlackCalibration, dark_stack_products,
                              signed_black_correct)
from .calibration_registry import digest

SCHEMA = "dngscan-personal-pixel-calibration-1"
MAX_BYTES = 512 * 1024 * 1024
INPUT_STATE = "metadata-black-subtracted-signed-dn"


def metadata_black_identity(metadata_black, shape):
    """Hash the complete broadcast metadata black field, independent of storage.

    A scalar and an equivalent spatial array have the same identity. Geometry
    origin/CFA/decoder are matched separately; this binds a residual correction
    to the exact phase + spatial baseline which was removed from its input.
    Row-wise hashing avoids a second full-resolution broadcast allocation.
    """
    if (len(shape) != 2 or any(type(x) is not int or x < 2 for x in shape)):
        raise ValueError("metadata black identity requires a Bayer grid")
    field = np.broadcast_to(_array(metadata_black, "metadata black", ndim=np.ndim(metadata_black)), shape)
    result = hashlib.sha256()
    result.update(json.dumps({"algorithm": "metadata-black-field-1", "shape": list(shape),
                              "dtype": "little-endian-float64"}, sort_keys=True).encode())
    for row in field:
        result.update(np.asarray(row, dtype="<f8").tobytes())
    return result.hexdigest()


def _bind_metadata_black(conditions, metadata_black, shape):
    value = copy.deepcopy(conditions)
    identity = metadata_black_identity(metadata_black, shape)
    if value.get("metadata_black_identity") not in (None, identity):
        raise ValueError("declared metadata black identity differs from the complete field")
    value["metadata_black_identity"] = identity
    return value


def _array(value, name, *, ndim):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not result.size or not np.isfinite(result).all():
        raise ValueError(name + " must be a finite array of the declared dimension")
    return result


def _conditions(value, shape):
    result = copy.deepcopy(value)
    for key in ("make", "model", "body_serial", "readout_identity", "decoder_identity", "linearisation_identity", "metadata_black_identity"):
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError("personal product requires " + key)
    identity = result["metadata_black_identity"]
    if len(identity) != 64 or any(character not in "0123456789abcdef" for character in identity):
        raise ValueError("metadata black identity must be a complete SHA256")
    if result.get("domain") != "linearized-raw-dn-before-stage2":
        raise ValueError("pixel calibration domain/order is unsupported")
    if result.get("cfa_pattern") not in ("RGGB", "GRBG", "GBRG", "BGGR"):
        raise ValueError("personal pixel product currently requires explicit Bayer CFA")
    if result.get("origin") is None or len(result["origin"]) != 2 or any(type(x) is not int or x < 0 for x in result["origin"]):
        raise ValueError("native sensor origin is required")
    if result.get("shape") != list(shape): raise ValueError("declared native grid and array differ")
    # Reuse the existing expiry, exposure, ISO, temperature and exact body rules.
    black = PersonalBlackCalibration(**{k: result[k] for k in
        ("make", "model", "body_serial", "readout_identity", "iso", "exposure_seconds")},
        phase_dn=dict.fromkeys(PHASES, 0.), phase_uncertainty_dn=dict.fromkeys(PHASES, 0.),
        source_hash="condition-validation", temperature_c=result.get("temperature_c"),
        firmware=result.get("firmware"), valid_until_utc=result.get("valid_until_utc"))
    black.validate()
    return result


def _sources(training, heldout, purpose):
    if not isinstance(purpose, dict) or purpose.get("training") != training or purpose.get("heldout") != heldout:
        raise ValueError("training/heldout frame purposes must be explicitly declared")
    train, test = purpose.get("training_hashes"), purpose.get("heldout_hashes")
    if (not isinstance(train, list) or not isinstance(test, list) or not train or not test
            or any(not isinstance(x, str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in train+test)
            or len(set(train+test)) != len(train+test)):
        raise ValueError("independent training/heldout source SHA256 identities are required")
    return copy.deepcopy(purpose)


def _finish(kind, conditions, source, arrays, diagnostics, qualification):
    maps = {key: np.ascontiguousarray(value, dtype=np.float32 if np.asarray(value).dtype != bool else bool)
            for key, value in arrays.items()}
    manifest = {"schema": SCHEMA, "kind": kind, "conditions": conditions, "source": source,
        "diagnostics": diagnostics, "qualification": qualification,
        "automatic_application": False, "experimental_signed_only": True,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype),
            "sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in maps.items()}}
    manifest["identity"] = digest(manifest)
    return {"manifest": manifest, "arrays": maps}


def build_dark_product(training, heldout, *, conditions, source, metadata_black,
                       black_reference=None, mode="residual", raw_floor_dn=0.):
    train = _array(training, "training dark", ndim=3)
    test = _array(heldout, "heldout dark", ndim=3)
    if train.shape[1:] != test.shape[1:] or min(train.shape[0], test.shape[0]) < 4:
        raise ValueError("at least four independent matched frames per split are required")
    conditions = _conditions(_bind_metadata_black(conditions, metadata_black, train.shape[1:]), train.shape[1:])
    if mode not in ("absolute", "residual"):
        raise ValueError("black mode must be absolute or residual")
    if isinstance(raw_floor_dn, (bool, np.bool_)) or not np.isscalar(raw_floor_dn) or not np.isfinite(raw_floor_dn):
        raise ValueError("RAW lower coding bound must be a finite scalar")
    source = _sources("dark", "dark", source)
    if len(source["training_hashes"]) != len(train) or len(source["heldout_hashes"]) != len(test):
        raise ValueError("source hash count must equal frame count")
    meta = np.broadcast_to(_array(metadata_black, "metadata black", ndim=np.ndim(metadata_black)), train.shape[1:])
    stats = dark_stack_products(train, calibration_identity=digest({"conditions": conditions, "source": source}))
    observed = {phase: float(train[:, dy::2, dx::2].mean())
                for phase, (dy, dx) in zip(PHASES, ((0,0), (0,1), (1,0), (1,1)))}
    clipped = float((np.count_nonzero(train <= raw_floor_dn) + np.count_nonzero(test <= raw_floor_dn)) / (train.size + test.size))
    residual = stats["mean_bias_map_dn"] - meta
    before = test.mean(axis=0)-meta
    after = before-residual
    diagnostics = {"observed_dark_mean_dn": observed, "qualified_black_reference_dn": None,
        "lower_clipped_fraction": clipped, "raw_floor_dn": float(raw_floor_dn),
        "heldout_fixed_rmse_before_dn": float(np.sqrt(np.mean(before**2))),
        "heldout_fixed_rmse_after_dn": float(np.sqrt(np.mean(after**2))),
        "temporal_variance_dn2": float(np.mean(stats["temporal_variance_dn2"])),
        "calibration_mean_variance_dn2": float(np.mean(stats["mean_uncertainty_dn"]**2)),
        "hot_candidates_only": int(stats["hot_pixel_candidates"].sum()),
        "compatibility_integer_path": "not applied; unsigned writeback clips negative tails"}
    qualification = "observed-only"
    if black_reference is not None:
        if black_reference.get("mode") not in (None, mode):
            raise ValueError("black reference mode conflicts with requested correction mode")
        if (mode not in ("absolute", "residual") or set(black_reference.get("phase_dn", {})) != set(PHASES)
                or set(black_reference.get("phase_uncertainty_dn", {})) != set(PHASES)
                or not black_reference.get("method") or black_reference.get("independent_zero_evidence") is not True):
            raise ValueError("qualified black reference needs method, independent zero evidence and all phase uncertainties")
        values = {phase: float(black_reference["phase_dn"][phase]) for phase in PHASES}
        uncertainty = {phase: float(black_reference["phase_uncertainty_dn"][phase]) for phase in PHASES}
        if not all(np.isfinite(x) for x in [*values.values(), *uncertainty.values()]) or min(uncertainty.values()) < 0:
            raise ValueError("black reference and uncertainties must be finite")
        diagnostics["qualified_black_reference_dn"] = {**black_reference, "mode": mode}
        # Stable DSNU is relative structure. Its phase offset comes from the
        # independently qualified reference, not the observed dark mean.
        for phase, (dy, dx) in zip(PHASES, ((0,0), (0,1), (1,0), (1,1))):
            part = residual[dy::2, dx::2]
            reference_offset = values[phase] if mode == "residual" else values[phase]-float(meta[dy::2, dx::2].mean())
            residual[dy::2, dx::2] = part - part.mean() + reference_offset
            stats["mean_uncertainty_dn"][dy::2, dx::2] = np.sqrt(stats["mean_uncertainty_dn"][dy::2, dx::2]**2 + uncertainty[phase]**2)
        diagnostics["heldout_fixed_rmse_after_dn"] = float(np.sqrt(np.mean((before-residual)**2)))
        diagnostics["calibration_mean_variance_dn2"] = float(np.mean(stats["mean_uncertainty_dn"]**2))
        if clipped == 0 and diagnostics["heldout_fixed_rmse_after_dn"] < diagnostics["heldout_fixed_rmse_before_dn"]:
            qualification = "heldout-supported-experiment"
    return _finish("dark", conditions, source,
        {"residual_dn": residual, "mean_uncertainty_dn": stats["mean_uncertainty_dn"],
         "temporal_variance_dn2": stats["temporal_variance_dn2"], "hot_candidates": stats["hot_pixel_candidates"]},
        diagnostics, qualification)


def build_flat_product(training, heldout, *, conditions, source, dark_product,
                       metadata_black=None, input_state=None, illumination_verified=False):
    if input_state != INPUT_STATE or metadata_black is None:
        raise ValueError("flat construction requires explicit metadata-black-subtracted-signed-dn input state and complete metadata black field")
    validate_product(dark_product)
    if dark_product["manifest"]["qualification"] != "heldout-supported-experiment":
        raise ValueError("flat requires independently supported additive dark")
    reference_record = dark_product["manifest"]["diagnostics"].get("qualified_black_reference_dn")
    if not isinstance(reference_record, dict) or reference_record.get("mode") != "residual":
        raise ValueError("flat requires a residual dark product supported by the signed application path")
    train = _array(training, "training flat", ndim=3)
    test = _array(heldout, "heldout flat", ndim=3)
    if train.shape[1:] != test.shape[1:] or min(train.shape[0], test.shape[0]) < 4:
        raise ValueError("four independent flat frames per split required")
    conditions = _conditions(_bind_metadata_black(conditions, metadata_black, train.shape[1:]), train.shape[1:])
    for key in ("lens_identity", "optical_stack", "aperture", "focal_length_mm", "focus_distance", "illumination_identity"):
        if conditions.get(key) is None: raise ValueError("spatial flat requires " + key)
    source = _sources("flat", "flat", source)
    if len(source["training_hashes"]) != len(train) or len(source["heldout_hashes"]) != len(test):
        raise ValueError("source hash count must equal frame count")
    manifest = dark_product["manifest"]
    for key in manifest["conditions"]:
        if conditions.get(key) != manifest["conditions"][key]: raise ValueError("dark/flat condition mismatch: " + key)
    # Flat buffers must already have the declared metadata black removed.
    corrected = train - dark_product["arrays"]["residual_dn"]
    reference = corrected.mean(axis=0)
    if np.any(reference <= 0): raise ValueError("flat has nonpositive signal")
    gain = np.empty(reference.shape)
    before, after = [], []
    for dy, dx in ((0,0), (0,1), (1,0), (1,1)):
        phase = reference[dy::2, dx::2]
        gain[dy::2, dx::2] = np.median(phase)/phase
        validation = (test-dark_product["arrays"]["residual_dn"])[:, dy::2, dx::2]
        normalizer = np.median(validation, axis=(1,2))[:, None, None]
        if np.any(normalizer <= 0): raise ValueError("heldout flat has nonpositive median")
        before.append(float(np.mean((validation/normalizer-1)**2)))
        after.append(float(np.mean((validation*gain[dy::2, dx::2]/normalizer-1)**2)))
    temporal_variance = corrected.var(axis=0, ddof=1)
    # The same dark map is reused across every flat frame. Its uncertainty is a
    # shared calibration error, so do not divide it by the flat frame count.
    mean_uncertainty = np.sqrt(temporal_variance / len(train)
                              + dark_product["arrays"]["mean_uncertainty_dn"].astype(np.float64)**2)
    diagnostics = {"dark_identity": manifest["identity"], "illumination_verified": illumination_verified is True,
        "input_state": input_state,
        "uncertainty_budget": {"status": "incomplete", "automatic_qualification": False,
            "included": ["flat temporal mean variance", "shared dark-map marginal uncertainty"],
            "missing": ["per-phase median normalization uncertainty", "normalizer/pixel covariance",
                        "spatial covariance of reused dark calibration", "illumination systematic uncertainty"],
            "interpretation": "reference mean marginal estimate only; no complete spatial-gain uncertainty or SNR claim"},
        "phase_relative_mse_before": before, "phase_relative_mse_after": after,
        "quantity": "spatial-relative-gain; per-phase median normalized; no spectral correction",
        "gain_variance_rule": "conditional on a fixed gain map: output temporal variance = input temporal variance * gain**2; calibration uncertainty is separate",
        "calibration_mean_variance_dn2": float(np.mean(mean_uncertainty**2)),
        "dark_subtracted_input_required": True}
    qualified = illumination_verified is True and all(a < b for a, b in zip(after, before))
    return _finish("flat", conditions, source,
        {"spatial_gain": gain, "reference_mean_uncertainty_dn": mean_uncertainty,
         "reference_temporal_variance_dn2": temporal_variance}, diagnostics,
                   "heldout-supported-experiment" if qualified else "observed-only")


def match_product(product, capture, *, applied_corrections=()):
    validate_product(product)
    manifest = product["manifest"]; required = manifest["conditions"]
    if manifest["qualification"] != "heldout-supported-experiment": return False, "unqualified-measurement"
    for key, expected in required.items():
        if key in ("exposure_seconds", "temperature_c", "valid_until_utc"): continue
        if capture.get(key) != expected: return False, key + "-mismatch-or-unknown"
    # Reuse reference scope matcher rather than inventing another expiry policy.
    probe = PersonalBlackCalibration(**{k: required[k] for k in ("make", "model", "body_serial", "readout_identity", "iso", "exposure_seconds")},
        phase_dn=dict.fromkeys(PHASES, 0.), phase_uncertainty_dn=dict.fromkeys(PHASES, 0.), source_hash=manifest["identity"],
        temperature_c=required.get("temperature_c"), firmware=required.get("firmware"), valid_until_utc=required.get("valid_until_utc"))
    matched, reason = probe.match(capture)
    if not matched: return False, reason
    if manifest["kind"] == "flat" and any(x in applied_corrections for x in
        ("DNG-GainMap", "lensfun-vignette", "embedded-lens-shading", manifest["identity"])):
        return False, "spatial-shading-already-applied"
    if (manifest["kind"] == "flat"
            and manifest["diagnostics"]["dark_identity"] not in applied_corrections):
        return False, "matched-dark-correction-not-recorded"
    if manifest["identity"] in applied_corrections: return False, "product-already-applied"
    return True, "matched-experimental-personal-product"


def apply_signed_product(raw, product, capture, *, experimental=False, applied_corrections=(),
                         input_state=None, metadata_black=None):
    if experimental is not True: raise ValueError("personal pixel application requires explicit experimental=True")
    if input_state != INPUT_STATE:
        raise ValueError("application requires explicit metadata-black-subtracted-signed-dn input state")
    if metadata_black is None:
        raise ValueError("application requires the complete metadata black field removed from the input")
    input_shape = np.shape(raw)
    actual_identity = metadata_black_identity(metadata_black, input_shape)
    if capture.get("metadata_black_identity") != actual_identity:
        raise ValueError("capture metadata black identity differs from the supplied field")
    matched, reason = match_product(product, capture, applied_corrections=applied_corrections)
    if not matched: raise ValueError(reason)
    result = _array(raw, "signed working buffer", ndim=2).copy()
    if list(result.shape) != product["manifest"]["conditions"]["shape"]: raise ValueError("working grid mismatch")
    if product["manifest"]["kind"] == "dark":
        # This API consumes metadata-black-subtracted signed work, so only the
        # measured residual is subtracted. Absolute phase replacement is exposed
        # separately by existing signed_black_correct(), never stacked with this.
        if product["manifest"]["diagnostics"]["qualified_black_reference_dn"]["mode"] != "residual":
            raise ValueError("absolute phase replacement must use signed_black_correct, not residual map subtraction")
        result -= product["arrays"]["residual_dn"]
    else:
        result *= product["arrays"]["spatial_gain"]
    return result, {"identity": product["manifest"]["identity"], "status": "applied-experimental-signed",
        "negative_fraction": float(np.mean(result < 0)), "hot_pixel_interpolation": False,
        "automatic_application": False,
        "uncertainty_budget_status": product["manifest"]["diagnostics"].get("uncertainty_budget", {}).get("status", "marginal-dark-estimate")}


def save_product(path, product):
    validate_product(product)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as target:
            temporary = Path(target.name)
            np.savez_compressed(target, manifest=json.dumps(product["manifest"], allow_nan=False), **product["arrays"])
        os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def load_product(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES: raise ValueError("pixel product file too large")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 8 or len({i.filename for i in infos}) != len(infos) or sum(i.file_size for i in infos) > MAX_BYTES:
            raise ValueError("pixel product inflated size/member limit")
    with np.load(path, allow_pickle=False) as value:
        product = {"manifest": json.loads(str(value["manifest"].item())),
                   "arrays": {key: value[key] for key in value.files if key != "manifest"}}
    validate_product(product)
    return product


def validate_product(product):
    manifest = product["manifest"]
    if manifest.get("schema") != SCHEMA or manifest.get("kind") not in ("dark", "flat"):
        raise ValueError("unsupported personal product schema")
    if manifest.get("automatic_application") is not False or manifest.get("experimental_signed_only") is not True:
        raise ValueError("personal product cannot authorize automatic production correction")
    if manifest.get("identity") != digest({k:v for k,v in manifest.items() if k != "identity"}):
        raise ValueError("personal product manifest identity mismatch")
    if set(product["arrays"]) != set(manifest["arrays"]): raise ValueError("personal product array table mismatch")
    expected = ({"residual_dn", "mean_uncertainty_dn", "temporal_variance_dn2", "hot_candidates"}
                if manifest["kind"] == "dark" else {"spatial_gain", "reference_mean_uncertainty_dn", "reference_temporal_variance_dn2"})
    if set(product["arrays"]) != expected: raise ValueError("unsupported personal product map roles")
    _conditions(manifest["conditions"], tuple(manifest["conditions"]["shape"]))
    if manifest["kind"] == "flat":
        budget = manifest.get("diagnostics", {}).get("uncertainty_budget", {})
        if budget.get("status") != "incomplete" or budget.get("automatic_qualification") is not False:
            raise ValueError("flat product lacks an honest incomplete uncertainty budget")
    for key, data in product["arrays"].items():
        record = manifest["arrays"][key]
        if (list(data.shape) != record["shape"] or list(data.shape) != manifest["conditions"]["shape"]
                or str(data.dtype) != record["dtype"] or data.dtype not in (np.dtype("float32"), np.dtype("bool")) or not np.isfinite(data).all()
                or hashlib.sha256(data.tobytes()).hexdigest() != record["sha256"]):
            raise ValueError("personal product array checksum/shape mismatch")
        if key in ("mean_uncertainty_dn", "temporal_variance_dn2", "reference_mean_uncertainty_dn", "reference_temporal_variance_dn2") and np.any(data < 0):
            raise ValueError("personal uncertainty/variance must be nonnegative")
        if key == "spatial_gain" and np.any(data <= 0): raise ValueError("spatial gain must be positive")
    return product
