#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproducible offline lens/SSF experiment; no profile is installed or applied.

Without --input, use measured Contax G35 relative optics with deliberately
synthetic camera/illuminants/materials. This tests contracts, not real-camera
color quality. --input accepts the versioned experiment dataset described in
LENS_SPECTRAL_INTEGRATION_2026-10-09.zh-CN.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dngscan import lens_transmittance as LT
from dngscan.spectral_color_research import (
    CameraSpectralResponse, Spectrum, camera_responses, color_error_report,
    experiment_identity, fit_white_constrained_matrix, neutral_white_balance_suggestion,
    propagate_color_covariance,
)


def demo_dataset() -> dict:
    wl = np.arange(380., 781., 5.)
    response = np.column_stack([np.exp(-((wl-610)/55)**2), np.exp(-((wl-540)/45)**2),
                                np.exp(-((wl-450)/35)**2)])
    camera = {"camera_id": "synthetic-three-gaussian-camera-not-Sigma-fp", "wavelengths_nm": wl.tolist(),
              "values_rgb": response.tolist(), "response_convention": "energy", "channel_normalization": "common",
              "reference_optical_stack": "synthetic-unit-reference", "includes_reference_optics": False,
              "source_id": "analytic-synthetic-v1"}
    def spectrum(identity, values, quantity):
        return {"measurement_id": identity, "wavelengths_nm": wl.tolist(), "values": np.asarray(values).tolist(), "quantity": quantity}
    train_lights = [spectrum("synthetic-broad-warm", np.exp((wl-550)/360), "energy_spd"),
                    spectrum("synthetic-broad-cool", np.exp(-(wl-550)/540), "energy_spd")]
    held_lights = [spectrum("held-synthetic-led", .1 + np.exp(-((wl-450)/12)**2) + 1.3*np.exp(-((wl-550)/35)**2)
                            + .7*np.exp(-((wl-620)/20)**2), "energy_spd"),
                   spectrum("held-synthetic-broad-curved", .4 + np.exp(-((wl-550)/140)**2), "energy_spd")]
    rng = np.random.default_rng(912)
    def materials(prefix, count):
        rows = []
        for index in range(count):
            centers = rng.uniform(400, 750, 4); widths = rng.uniform(12, 90, 4)
            values = np.sum(rng.uniform(.05, 1, 4)[:, None] * np.exp(-((wl[None, :]-centers[:, None])/widths[:, None])**2), axis=0)
            values = .02 + .94 * values / max(values.max(), 1)
            rows.append(spectrum(f"{prefix}-{index:02d}", values, "reflectance"))
        if prefix == "held":
            rows.append(spectrum(f"{prefix}-neutral", np.ones(len(wl))*.18, "reflectance"))
        return rows
    optical_wl, optical_values = LT.get_curve("0612-contax-g35")
    target = {"measurement_id": "0612-contax-g35", "wavelengths_nm": optical_wl,
              "values": optical_values, "quantity": "relative_spectral_transmittance"}
    return {"schema": "agxraw-spectral-color-experiment-1", "camera": camera,
            "target_optics": target, "reference_optics": spectrum("synthetic-unit-reference", np.ones(len(wl)), "relative_spectral_transmittance"),
            "training_illuminants": train_lights, "validation_illuminants": held_lights,
            "training_reflectances": materials("train", 32), "validation_reflectances": materials("held", 24),
            "reference_rgb_to_xyz": [[.636958, .144617, .168881], [.2627, .678, .0593], [0, .028073, 1.060985]],
            "reference_matrix_source": "synthetic-Rec2020-shaped-coordinate-map-not-camera-characterization",
            "wavelength_range_nm": [380, 755], "white_balance_policy": "per-illuminant-target-neutral",
            "evidence_scope": "measured-relative-lens-shape-plus-synthetic-camera-light-materials",
            "measured_optics_metadata": LT.get_metadata("0612-contax-g35")}


def run_experiment(dataset: dict) -> dict:
    if dataset.get("schema") != "agxraw-spectral-color-experiment-1":
        raise ValueError("unsupported spectral experiment schema")
    if dataset.get("white_balance_policy") != "per-illuminant-target-neutral":
        raise ValueError("experiment requires an explicit neutral white-balance basis")
    camera = CameraSpectralResponse(**dataset["camera"])
    reference, target = Spectrum(**dataset["reference_optics"]), Spectrum(**dataset["target_optics"])
    train = [Spectrum(**x) for x in dataset["training_reflectances"]]
    held = [Spectrum(**x) for x in dataset["validation_reflectances"]]
    training_lights = [Spectrum(**x) for x in dataset["training_illuminants"]]
    validation_lights = [Spectrum(**x) for x in dataset["validation_illuminants"]]
    for group in (train, held, training_lights, validation_lights):
        ids = [x.measurement_id for x in group]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("empty or duplicate experiment identities")
    if ({x.measurement_id for x in train} & {x.measurement_id for x in held}
            or {x.measurement_id for x in training_lights} & {x.measurement_id for x in validation_lights}):
        raise ValueError("training and validation spectra/illuminants must be independent")
    # IDs alone cannot establish a holdout when the exact sampled spectrum is reused.
    def signature(spec):
        wl, values = spec.arrays()
        return hashlib.sha256(wl.tobytes() + values.tobytes()).hexdigest()
    if ({signature(x) for x in train} & {signature(x) for x in held}
            or {signature(x) for x in training_lights} & {signature(x) for x in validation_lights}):
        raise ValueError("held-out data reuse exact training spectral samples")
    rgb_to_xyz = np.asarray(dataset["reference_rgb_to_xyz"], float)
    if rgb_to_xyz.shape != (3, 3) or not np.isfinite(rgb_to_xyz).all() or not dataset.get("reference_matrix_source"):
        raise ValueError("reference XYZ map and its source must be explicit")
    camera_wl, _ = camera.arrays()
    neutral = Spectrum("integration-white", camera_wl, np.ones(len(camera_wl)), "reflectance")
    restriction = dataset.get("wavelength_range_nm")
    kwargs = {"reference_optics": reference, "wavelength_range_nm": restriction}
    def formed(light, materials):
        target_rgb, evidence = camera_responses(camera, light, materials+[neutral], target, **kwargs)
        reference_rgb, _ = camera_responses(camera, light, materials+[neutral], reference, **kwargs)
        if np.any(target_rgb[-1] <= 0) or np.any(reference_rgb[-1] <= 0):
            raise ValueError("white point has unobservable channels")
        x = target_rgb[:-1] / target_rgb[-1]
        y = (reference_rgb[:-1] / reference_rgb[-1]) @ rgb_to_xyz.T
        return x, y, evidence
    xs, ys = [], []
    for light in training_lights:
        x, y, _ = formed(light, train); xs.append(x); ys.append(y)
    white = rgb_to_xyz @ np.ones(3)
    matrix = fit_white_constrained_matrix(np.concatenate(xs), np.concatenate(ys), source_white=np.ones(3), target_white=white)
    validation = []
    for light in validation_lights:
        x, y, evidence = formed(light, held)
        ids = [x.measurement_id for x in held]
        validation.append({"evidence": evidence,
                           "baseline": color_error_report(x @ rgb_to_xyz.T, y, white_xyz=white, sample_ids=ids, illuminant_id=light.measurement_id),
                           "candidate": color_error_report(x @ matrix.T, y, white_xyz=white, sample_ids=ids, illuminant_id=light.measurement_id)})
    covariance = np.array([[.04, .012, -.003], [.012, .03, .005], [-.003, .005, .02]])
    predicted = propagate_color_covariance(matrix, covariance)
    rng = np.random.default_rng(913)
    observed = np.cov(rng.multivariate_normal(np.zeros(3), covariance, size=250000) @ matrix.T, rowvar=False)
    max_relative = float(np.max(np.abs(observed-predicted)) / np.max(np.abs(predicted)))
    suggestions = [neutral_white_balance_suggestion(camera, light, target, reference,
                                                    existing_wb_state="reference-optics", wavelength_range_nm=restriction)
                   for light in validation_lights]
    return {"schema": "agxraw-spectral-color-report-1", "algorithm": "agxraw-spectral-color-research-1",
            "experiment_identity": experiment_identity(dataset), "input_scope": dataset.get("evidence_scope", "caller-supplied"),
            "camera_id": camera.camera_id, "matrix": matrix.tolist(), "white_constraint_error": float(np.max(np.abs(matrix @ np.ones(3)-white))),
            "training_reflectance_ids": [x.measurement_id for x in train],
            "training_illuminant_ids": [x.measurement_id for x in training_lights],
            "validation": validation, "neutral_suggestions": suggestions,
            "covariance_monte_carlo": {"draws": 250000, "seed": 913, "input_covariance": covariance.tolist(),
                                       "predicted_covariance": predicted.tolist(), "observed_covariance": observed.tolist(),
                                       "max_error_relative_to_max_variance": max_relative},
            "production_eligible": False, "pixel_correction_applied": False,
            "reason": "offline-experiment; real-camera independent validation and optical identity qualification required"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="versioned spectral experiment JSON; omit for explicit synthetic demonstration")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = json.loads(args.input.read_text()) if args.input else demo_dataset()
    result = run_experiment(dataset)
    result["repository_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[1], text=True).strip()
    result["source_files"] = {str(Path(__file__).relative_to(Path(__file__).parents[1])): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              "dngscan/spectral_color_research.py": hashlib.sha256((Path(__file__).parents[1]/"dngscan/spectral_color_research.py").read_bytes()).hexdigest(),
                              "dngscan/data/lens_transmittance.json": hashlib.sha256((Path(__file__).parents[1]/"dngscan/data/lens_transmittance.json").read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"experiment_identity": result["experiment_identity"], "production_eligible": False,
                      "held_out": [{"light": row["candidate"]["illuminant_id"],
                                    "baseline_p95": row["baseline"]["p95_delta_e76"],
                                    "candidate_p95": row["candidate"]["p95_delta_e76"],
                                    "candidate_max": row["candidate"]["max_delta_e76"]} for row in result["validation"]],
                      "covariance_max_relative_error": result["covariance_monte_carlo"]["max_error_relative_to_max_variance"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
