#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent-repeat colour bias/noise tradeoffs, not real-camera validation.

Joint candidate and baseline use identical B3 bands, scenes and random draws.
Joint strengths are selected on a separate flat-noise ensemble to match the
baseline's noise reduction where its bounded shrinkage can reach that target;
unreachable points are marked and cannot support an equal-benefit comparison.
Reported texture/noise results use held-out draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dngscan.chroma_nr import LUMA_W, chroma_correction_map
from dngscan.chroma_research import joint_multiscale_correction
from dngscan.chroma_policy import CONSERVATIVE_AMOUNT, POLICY_VERSION


def projection():
    w = LUMA_W.astype(float)
    return np.eye(3) - np.ones((3, 1)) * w[None, :] / w.sum()


def run_ensemble(clean, noise, amount, operator, covariance):
    outputs, max_y = [], 0.
    for draw in noise:
        noisy = (clean + draw).astype(np.float32)
        correction = (np.zeros_like(noisy) if amount == 0 else
                      operator(noisy, amount, 8., noise_covariance=covariance))
        max_y = max(max_y, float(np.max(np.abs(correction @ LUMA_W))))
        outputs.append(noisy + correction)
    return np.asarray(outputs), max_y


def metrics(outputs, clean, *, signal=None):
    # The common noise-mean bias of a finite repeat set is not temporal noise.
    # Variance about the ensemble mean and bias about the known clean image
    # are recorded separately. No result-dependent registration is performed.
    chroma = outputs @ projection().T
    centered = chroma - chroma.mean(0)
    result = {"temporal_chroma_variance": float(np.mean(centered ** 2)),
              "mean_chroma_bias_mse": float(np.mean(((outputs.mean(0) - clean) @ projection().T) ** 2)),
              "mean_scene_rgb": outputs.mean((0, 1, 2)).astype(float).tolist()}
    basis = np.linalg.svd(LUMA_W.astype(float)[None, :], full_matrices=True)[2][1:].T
    coordinates = centered @ basis
    result["temporal_2d_chroma_covariance"] = (np.einsum("na,nb->ab", coordinates.reshape(-1, 2),
                                                       coordinates.reshape(-1, 2)) / np.prod(coordinates.shape[:-1])).tolist()
    # Report the residual temporal noise spectrum separately from texture
    # bias. Annuli are diagnostic frequency bins, not hard B3 cutoffs.
    height, width = centered.shape[1:3]
    spectra = np.mean(np.sum(np.abs(np.fft.fft2(centered, axes=(1, 2))) ** 2, axis=-1), axis=0) / (height * width) ** 2
    radial = np.hypot(np.fft.fftfreq(height)[:, None], np.fft.fftfreq(width)[None, :]) / 8.
    edges = (0., 1 / 128., 1 / 64., 1 / 32., 1 / 16., float("inf"))
    result["temporal_chroma_noise_power_bands"] = [
        {"frequency_min_cycles_per_sensor_pixel": low,
         "frequency_max_cycles_per_sensor_pixel": None if not np.isfinite(high) else high,
         "summed_rgb_chroma_power": float(spectra[(radial >= low) & (radial < high)].sum())}
        for low, high in zip(edges[:-1], edges[1:])]
    if signal is not None:
        denominator = float(np.sum(signal ** 2))
        amplitudes = np.sum((outputs - .2) * signal, axis=(1, 2, 3)) / denominator
        mean_field = outputs.mean(0) - .2
        gain = float(amplitudes.mean())
        orthogonal = mean_field - gain * signal
        strongest = signal.reshape(-1, 3)[np.argmax(np.sum(signal.reshape(-1, 3) ** 2, axis=1))]
        direction = strongest / np.linalg.norm(strongest)
        wave = signal @ direction
        fitted_vector = np.sum(mean_field * wave[..., None], axis=(0, 1)) / np.sum(wave ** 2)
        cosine = float(np.dot(fitted_vector, direction) / max(np.linalg.norm(fitted_vector), 1e-30))
        result.update(amplitude_transfer=gain,
                      amplitude_transfer_standard_error=float(amplitudes.std(ddof=1) / np.sqrt(len(amplitudes))),
                      signal_relative_orthogonal_error=float(np.linalg.norm(orthogonal) / np.linalg.norm(signal)),
                      fitted_colour_vector_angle_error_degrees=float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))))
    return result


def build_report(size=96, repeats=24):
    sigma = .01
    # Deliberately correlated RGB covariance tests the joint candidate rather
    # than granting it only an isotropic model favourable to independent axes.
    covariance = sigma ** 2 * np.asarray([[1., .25, -.10], [.25, .8, .15], [-.10, .15, 1.2]])
    transform = np.linalg.cholesky(covariance)
    draws = np.random.default_rng(71).normal(size=(repeats, size, size, 3)) @ transform.T
    train = np.random.default_rng(9871).normal(size=(12, size, size, 3)) @ transform.T
    flat = np.full((size, size, 3), .2, np.float32)
    strengths = (.0, .05, .10, .15, .25, .5)
    algorithms = {"legacy-coarse": chroma_correction_map, "joint-candidate": joint_multiscale_correction}
    train_reference = metrics(run_ensemble(flat, train, 0, chroma_correction_map, covariance)[0], flat)
    held_reference = metrics(run_ensemble(flat, draws, 0, chroma_correction_map, covariance)[0], flat)
    flat_curves, matched = [], []
    for amount in strengths:
        baseline_train, _ = run_ensemble(flat, train, amount, chroma_correction_map, covariance)
        target = metrics(baseline_train, flat)["temporal_chroma_variance"]
        low, high = 0., 1.
        if amount:
            for _ in range(11):
                mid = (low + high) / 2
                out, _ = run_ensemble(flat, train, mid, joint_multiscale_correction, covariance)
                if metrics(out, flat)["temporal_chroma_variance"] > target:
                    low = mid
                else:
                    high = mid
        joint_amount = (low + high) / 2 if amount else 0.
        joint_train, _ = run_ensemble(flat, train, joint_amount, joint_multiscale_correction, covariance)
        actual = metrics(joint_train, flat)["temporal_chroma_variance"]
        relative_gap = actual / target - 1.
        match_status = "matched" if abs(relative_gap) <= .005 else "unreachable-under-bounded-candidate"
        matched.append({"baseline_amount": amount, "joint_amount": joint_amount,
                        "noise_benefit_match_status": match_status,
                        "training_relative_noise_variance_gap": relative_gap,
                        "training_target_chroma_variance_ratio": target / train_reference["temporal_chroma_variance"]})
        for name, strength in (("legacy-coarse", amount), ("joint-candidate", joint_amount)):
            outputs, yerr = run_ensemble(flat, draws, strength, algorithms[name], covariance)
            result = metrics(outputs, flat)
            flat_curves.append({"algorithm": name, "baseline_amount": amount, "operator_amount": strength,
                                "noise_benefit_match_status": match_status,
                                "heldout_chroma_variance_ratio": result["temporal_chroma_variance"] / held_reference["temporal_chroma_variance"],
                                "max_abs_delta_scene_Y": yerr, **result})
    basis = np.linalg.svd(LUMA_W.astype(float)[None, :], full_matrices=True)[2][1:].T
    yy, xx = np.indices((size, size))
    waves = {"sinusoid-64-sensor-px": np.sin(2 * np.pi * xx / 8.),
             "equal-Y-weave": .5 * (np.sin(2 * np.pi * xx / 8.) + np.sin(2 * np.pi * yy / 8.)),
             "equal-Y-thin-lines": np.where(xx % 8 < 1, 1., -1. / 7.),
             "equal-Y-step-edge": np.where(xx < size // 2, -1., 1.)}
    texture_curves = []
    for pattern, wave in waves.items():
        for angle in (0., 45., 90.):
            direction = basis @ np.asarray([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            for ratio in (.25, 1., 4.):
                signal = (ratio * sigma * wave[..., None] * direction).astype(np.float32)
                clean = flat + signal
                # Full sweep for the diagnostic sinusoid; edge/weave/lines
                # test selected conservative and stronger custom settings.
                use = matched if pattern.startswith("sinusoid") else [r for r in matched if r["baseline_amount"] in (0, CONSERVATIVE_AMOUNT, .5)]
                for record in use:
                    for name, strength in (("legacy-coarse", record["baseline_amount"]), ("joint-candidate", record["joint_amount"])):
                        outputs, yerr = run_ensemble(clean, draws, strength, algorithms[name], covariance)
                        result = metrics(outputs, clean, signal=signal)
                        texture_curves.append({"pattern": pattern, "chroma_angle_degrees": angle,
                                               "signal_amplitude_over_reference_rgb_sigma": ratio,
                                               "algorithm": name, "baseline_amount": record["baseline_amount"],
                                               "noise_benefit_match_status": record["noise_benefit_match_status"],
                                               "operator_amount": strength, "max_abs_delta_scene_Y": yerr, **result})
    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "dngscan/chroma_nr.py", ROOT / "dngscan/chroma_policy.py",
                                 ROOT / "dngscan/chroma_research.py", Path(__file__))}
    selected = [r for r in texture_curves if r["algorithm"] == "legacy-coarse" and r["baseline_amount"] == CONSERVATIVE_AMOUNT]
    return {"schema": "agxraw-chroma-tradeoff/1", "policy_version": POLICY_VERSION,
            "agxraw_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_changes": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
            "source_sha256": fingerprints, "python": platform.python_version(), "numpy": np.__version__,
            "seed": 71, "independent_noise_matching_seed": 9871, "grid_size": size, "sensor_pixels_per_grid_cell": 8,
            "repeats": repeats, "reference_rgb_sigma": sigma, "rgb_covariance": covariance.tolist(),
            "training_noise_matching": matched, "heldout_flat_curves": flat_curves, "texture_curves": texture_curves,
            "selection": {"coarse_opt_in_amount": CONSERVATIVE_AMOUNT,
                          "engineering_selection_target": "heldout flat chroma variance reduction >=5%; strong-signal amplitude >=0.98; no claim for weak texture or real RAW",
                          "minimum_observed_amplitude_transfer": min(r["amplitude_transfer"] for r in selected),
                          "minimum_high_signal_amplitude_transfer": min(r["amplitude_transfer"] for r in selected if r["signal_amplitude_over_reference_rgb_sigma"] == 4),
                          "joint_default_enabled": False, "fine_chroma_enabled": False,
                          "real_camera_image_quality_validated": False},
            "limits": ["synthetic fixed independent scene-grid noise, not RAW decoder calibration",
                       "matched flat-noise benefit is not matched benefit in every scene",
                       "independent repeat errors quantify shrinkage bias; zero Y does not certify colour detail",
                       "radial spectrum bins are diagnostics on an 8-sensor-pixel grid, not fine-chroma coverage or hard B3 bands",
                       "no real personal dark/flat or low-noise texture reference was supplied"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args()
    if args.size < 64 or args.size % 8 or args.repeats < 4:
        parser.error("size must be a multiple of 8 >=64 and repeats >=4")
    report = build_report(args.size, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["selection"], ensure_ascii=False))


if __name__ == "__main__":
    main()
