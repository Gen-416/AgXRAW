# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent-repeat statistics in a fixed decoder's RGB output domain.

This is an offline experiment, never a RAW noise prior or production NR hook.
Capture IDs and hashes enforce a declared contract; they cannot establish that
an operator actually captured independent frames under controlled conditions.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

SCHEMA = "agxraw-decoder-residual-contract-1"
DOMAIN = "scene-linear-rec2020-output"
LUMA = np.array([.2627, .6780, .0593], dtype=np.float64)
CHROMA_BASIS = np.linalg.svd(LUMA[None, :], full_matrices=True)[2][1:].T


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _frames(value):
    original = np.asarray(value)
    if original.dtype.kind != "f":
        raise ValueError("residual measurements require signed floating decoder outputs, not integer images")
    array = np.asarray(original, dtype=np.float64)
    if (array.ndim != 4 or array.shape[-1] != 3 or min(array.shape[1:3]) < 8
            or array.shape[0] < 4 or array.shape[0] % 2 or not np.isfinite(array).all()):
        raise ValueError("need an even number (at least four) of finite independent HxWx3 frames")
    return array


def output_transform_identity(recipe):
    """Include per-file WB/BE/alignment facts omitted from configuration ID.

    Source hashes and fallback history are provenance, not an RGB mapping.
    In contrast effective as-shot neutral values and project alignment factors
    can change residual units even when the Apple configuration ID is stable.
    """
    if not isinstance(recipe, dict): raise ValueError("complete per-frame decoder recipe required")
    return _identity({key: recipe.get(key) for key in
        ("configuration_id", "actual_decoder_token", "runtime", "geometry", "output", "controls", "project_handoff")})


def _contract(contract, records, count, shape):
    if not isinstance(contract, dict) or contract.get("schema") != SCHEMA:
        raise ValueError("unsupported decoder residual contract")
    if contract.get("domain") != DOMAIN or contract.get("frame_independence") != "separate-captures":
        raise ValueError("explicit output domain and separate-captures contract required")
    if contract.get("acquisition_kind") not in ("dark", "flat"):
        raise ValueError("ordinary photographs are not dark/flat residual measurements")
    recipe = contract.get("decoder_recipe")
    if (not isinstance(recipe, dict) or not recipe.get("configuration_id") or not recipe.get("actual_decoder_token")
            or any(key not in recipe for key in ("runtime", "geometry", "output", "controls"))):
        raise ValueError("a complete fixed decoder recipe is required")
    declared = recipe["geometry"].get("output_shape") if isinstance(recipe["geometry"], dict) else None
    if declared not in (list(shape[:2]), list(shape)) or shape[-1] != 3:
        raise ValueError("decoder recipe output_shape differs from the complete HxWx3 frame grid")
    conditions = contract.get("capture_conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("explicit capture conditions are required")
    for field in ("make", "model", "body_serial", "readout_id"):
        if not isinstance(conditions.get(field), str) or not conditions[field].strip():
            raise ValueError("capture conditions require " + field)
    for field in ("iso", "exposure_seconds"):
        value = conditions.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
            raise ValueError("capture conditions require positive finite " + field)
    transform_id = output_transform_identity(recipe)
    if not isinstance(records, (list, tuple)) or len(records) != count:
        raise ValueError("one capture record per independent frame is required")
    hashes, captures = [], []
    for row in records:
        if not isinstance(row, dict): raise ValueError("invalid capture record")
        source = row.get("source_sha256")
        capture = row.get("capture_id")
        if (not isinstance(source, str) or len(source) != 64
                or any(c not in "0123456789abcdef" for c in source)
                or not isinstance(capture, str) or not capture.strip()):
            raise ValueError("capture IDs and complete source SHA256 hashes are required")
        if row.get("decoder_configuration_id") != recipe["configuration_id"]:
            raise ValueError("decoder configuration differs between independent frames")
        if output_transform_identity(row.get("decoder_recipe")) != transform_id:
            raise ValueError("per-frame output geometry, white balance, baseline or scale differs")
        if row.get("capture_conditions") != conditions:
            raise ValueError("capture conditions differ between independent frames")
        hashes.append(source); captures.append(capture)
    if len(set(hashes)) != count or len(set(captures)) != count:
        raise ValueError("rerendering the same RAW is not an independent noise capture")
    _identity(contract)  # reject NaN/Infinity and unserializable declarations
    return {"source_sha256": hashes, "capture_ids": captures, "contract_sha256": _identity(contract),
            "output_transform_id": transform_id}


def _regions(regions, shape):
    height, width = shape
    result = {"full": (0, 0, height, width)} if regions is None else dict(regions)
    if not result: raise ValueError("at least one region is required")
    for name, box in result.items():
        if (not isinstance(name, str) or not name or len(box) != 4
                or any(type(v) is not int for v in box)):
            raise ValueError("regions require names and integer y0,x0,y1,x1 bounds")
        y0, x0, y1, x1 = box
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width and min(y1-y0, x1-x0) >= 8):
            raise ValueError("region exceeds the output grid or is smaller than 8 pixels")
    return result


def _reference(value, shape):
    if value is None: return None
    ref = np.asarray(value, dtype=np.float64)
    if ref.shape not in ((3,), shape) or not np.isfinite(ref).all():
        raise ValueError("reference must be finite RGB or the exact independently registered output grid")
    return np.broadcast_to(ref, shape)


def _reference_contract(contract):
    evidence = contract.get("reference_provenance")
    if (not isinstance(evidence, dict) or not evidence.get("source_id")
            or evidence.get("independent_of_captures") is not True
            or evidence.get("domain") != DOMAIN):
        raise ValueError("reference provenance and declared capture independence are required")


def _region_statistics(frames, reference):
    count, height, width, _ = frames.shape
    pixels = height * width
    cov, second = np.zeros((3, 3)), np.zeros((3, 3))
    means, spectra = [], np.zeros((height, width, 2, 2))
    for first, second_frame in zip(frames[::2], frames[1::2]):
        diff = (first-second_frame).reshape(-1, 3)
        mean = diff.mean(0)
        centered = diff-mean
        cov += centered.T @ centered / (2*(pixels-1))
        second += diff.T @ diff / (2*pixels)
        means.append(mean.tolist())
        coordinates = (centered @ CHROMA_BASIS).reshape(height, width, 2)
        fft = np.fft.fft2(coordinates, axes=(0, 1))
        spectra += np.real(fft[..., :, None] * fft[..., None, :].conj()) / (2*pixels*pixels)
    pairs = count//2
    cov /= pairs; second /= pairs; spectra /= pairs
    radial = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :])
    edges = (0., 1/128., 1/64., 1/32., 1/16., 1/8., math.inf)
    bands = [{"minimum_cycles_per_output_pixel": low,
              "maximum_cycles_per_output_pixel": high if math.isfinite(high) else None,
              "zero_y_covariance_power_2x2": spectra[(radial >= low)&(radial < high)].sum(0).tolist()}
             for low, high in zip(edges[:-1], edges[1:])]
    result = {"shape": [height, width, 3], "independent_pairs": pairs,
        "pair_difference_mean_rgb": means,
        "within_region_temporal_covariance_rgb_3x3": cov.tolist(),
        "pair_difference_raw_second_moment_rgb_3x3": second.tolist(),
        "zero_y_temporal_covariance_2x2": (CHROMA_BASIS.T @ cov @ CHROMA_BASIS).tolist(),
        "luminance_temporal_variance": float(LUMA @ cov @ LUMA),
        "frequency_unit": "cycles/output-pixel", "bands": bands,
        "ensemble_mean_rgb": frames.mean((0, 1, 2)).tolist()}
    if reference is None:
        result.update(mean_bias_status="unavailable-without-independent-reference",
                      fixed_pattern_status="unavailable-scene-structure-is-not-FPN")
    else:
        residual = (frames.mean(0)-reference).reshape(-1, 3)
        mean = residual.mean(0)
        centered = residual-mean
        spatial = centered.T @ centered / (pixels-1)
        debiased = spatial-cov/count
        result.update(mean_bias_status="reference-relative-experiment", mean_bias_rgb=mean.tolist(),
            ensemble_mean_residual_mse_rgb=np.mean(residual**2, axis=0).tolist(),
            fixed_pattern_status="candidate-reference-error-and-finite-repeat-noise-remain",
            ensemble_mean_spatial_covariance_rgb_3x3=spatial.tolist(),
            mean_temporal_uncertainty_covariance_rgb_3x3=(cov/count).tolist(),
            candidate_debiased_spatial_covariance_rgb_3x3=debiased.tolist(),
            candidate_spatial_covariance_status=("not-resolved" if np.linalg.eigvalsh(debiased).min() < 0
                                                  else "nonnegative-candidate-not-qualification"))
    return result


def repeat_residual_statistics(frames, frame_records, *, contract, regions=None, reference_mean=None):
    """Measure independent A−B covariance/2, separately from mean bias/FPN.

    Region demeaning removes each pair's common RGB offset; its offset and raw
    second moment are retained separately. PSD bands use output pixels, never
    sensor-pixel units. An optional independent reference permits bias/FPN
    candidates, not proof of a physical sensor correction.
    """
    data = _frames(frames)
    provenance = _contract(contract, frame_records, len(data), data.shape[1:])
    reference = _reference(reference_mean, data.shape[1:])
    if reference is not None: _reference_contract(contract)
    boxes = _regions(regions, data.shape[1:3])
    report = {"schema": "agxraw-decoder-residual-statistics-1", "domain": DOMAIN,
        "decoder_recipe": contract["decoder_recipe"], "capture_conditions": contract["capture_conditions"],
        "acquisition_kind": contract["acquisition_kind"], "provenance": provenance,
        "frame_count": len(data), "zero_y_basis_rgb_columns": CHROMA_BASIS.tolist(),
        "luminance_weights_rgb": LUMA.tolist(), "regions": {},
        "production_applicable": False, "cfa_noise_gate_qualified": False,
        "limitations": ["declared independent captures require acquisition audit",
            "output-domain residuals do not identify predecoder RAW noise",
            "region demeaning excludes common-mode drift from within-region covariance",
            "spatial FPN candidates retain reference uncertainty and finite-repeat uncertainty",
            "no adaptive-decoder covariance transfer or downstream NR permission"]}
    for name, (y0,x0,y1,x1) in boxes.items():
        report["regions"][name] = _region_statistics(data[:, y0:y1, x0:x1],
            None if reference is None else reference[y0:y1, x0:x1])
        report["regions"][name]["bounds_yxxy"] = [y0,x0,y1,x1]
    report["identity"] = _identity(report)
    return report


def build_output_bias_candidate(training, training_records, heldout, heldout_records, *, contract,
                                reference_mean):
    """Return a signed output-offset experiment and independent holdout metrics.

    Subtracting this map after a nonlinear/adaptive decoder is not equivalent
    to sensor black/DSNU correction before decoding. Nothing is installed.
    """
    train, test = _frames(training), _frames(heldout)
    if train.shape[1:] != test.shape[1:]: raise ValueError("training and heldout output grids differ")
    if not isinstance(contract, dict) or contract.get("acquisition_kind") != "dark":
        raise ValueError("an additive output bias candidate requires independent dark captures")
    train_sources = _contract(contract, training_records, len(train), train.shape[1:])
    test_sources = _contract(contract, heldout_records, len(test), test.shape[1:])
    if (set(train_sources["source_sha256"]) & set(test_sources["source_sha256"])
            or set(train_sources["capture_ids"]) & set(test_sources["capture_ids"])):
        raise ValueError("training and heldout captures must be independent")
    ref = _reference(reference_mean, train.shape[1:])
    if ref is None: raise ValueError("an independent output reference is required")
    _reference_contract(contract)
    bias = np.ascontiguousarray(train.mean(0)-ref, dtype=np.float32)
    before = repeat_residual_statistics(test, heldout_records, contract=contract, reference_mean=ref)
    after = repeat_residual_statistics(test-bias, heldout_records, contract=contract, reference_mean=ref)
    manifest = {"schema": "agxraw-output-bias-candidate-1", "domain": DOMAIN,
        "decoder_recipe": contract["decoder_recipe"], "contract_sha256": _identity(contract),
        "training": train_sources, "heldout": test_sources,
        "semantics": "subtract signed RGB output offset only in the identical decoder configuration",
        "automatic_application": False, "production_applicable": False,
        "predecoder_raw_correction_equivalent": False,
        "qualification": "experimental-output-candidate; real-camera-acceptance-required",
        "bias_shape": list(bias.shape), "bias_dtype": "float32",
        "heldout_before": before, "heldout_after": after,
        "limitations": ["does not recover clipped decoder values or suppressed signal",
            "not a sensor black level, gain, read-noise or full-well calibration",
            "nonlinear decoder processing can make output bias signal dependent",
            "no production hook, downstream NR or CFA-gate permission"]}
    array = bias
    manifest["bias_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
    manifest["identity"] = _identity(manifest)
    return {"manifest": manifest, "output_bias_rgb": array}
