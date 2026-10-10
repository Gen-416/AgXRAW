# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline, explicit-contract spectral color experiments; never a render hook.

Relative optical spectra predict normalized color only. These routines do not
recover photons, T-stop, object spectra from RGB or exact Sigma fp sensitivities.
The implementation is original NumPy code; no upstream simulator code is copied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Spectrum:
    measurement_id: str
    wavelengths_nm: Any
    values: Any
    quantity: str

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        wl = np.asarray(self.wavelengths_nm, dtype=np.float64)
        val = np.asarray(self.values, dtype=np.float64)
        if (wl.ndim != 1 or val.shape != wl.shape or len(wl) < 2
                or not np.isfinite(wl).all() or not np.isfinite(val).all()
                or np.any(np.diff(wl) <= 0) or np.any(wl <= 0) or np.any(val < 0)):
            raise ValueError(f"invalid spectrum: {self.measurement_id}")
        if not self.measurement_id or (self.quantity != "reflectance" and not np.any(val > 0)):
            raise ValueError("spectrum needs identity and nonzero data")
        if self.quantity == "reflectance" and np.any(val > 1):
            raise ValueError("non-fluorescent reflectance must lie in [0,1]")
        return wl, val


@dataclass(frozen=True)
class CameraSpectralResponse:
    camera_id: str
    wavelengths_nm: Any
    values_rgb: Any
    response_convention: str  # energy or photon
    channel_normalization: str  # common, known-prescale, unknown
    reference_optical_stack: str | None
    includes_reference_optics: bool | None
    source_id: str
    channel_prescale: tuple[float, float, float] | None = None

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        wl = np.asarray(self.wavelengths_nm, dtype=np.float64)
        rgb = np.asarray(self.values_rgb, dtype=np.float64)
        if (not self.camera_id or not self.source_id or wl.ndim != 1 or len(wl) < 2
                or rgb.shape != (len(wl), 3) or not np.isfinite(wl).all()
                or not np.isfinite(rgb).all() or np.any(np.diff(wl) <= 0)
                or np.any(wl <= 0) or np.any(rgb < 0) or np.any(rgb.max(axis=0) <= 0)):
            raise ValueError("invalid camera response")
        if self.response_convention not in ("energy", "photon"):
            raise ValueError("camera energy/photon response convention is unknown")
        if self.channel_normalization == "known-prescale":
            factors = np.asarray(self.channel_prescale, dtype=np.float64)
            if factors.shape != (3,) or not np.isfinite(factors).all() or np.any(factors <= 0):
                raise ValueError("known-prescale response needs positive channel factors")
            rgb = rgb / factors  # supplied curves already include these factors
        elif self.channel_normalization != "common" or self.channel_prescale is not None:
            raise ValueError("unknown or inconsistent relative channel normalization")
        if not isinstance(self.includes_reference_optics, bool) or not self.reference_optical_stack:
            raise ValueError("camera response reference optical path is unknown")
        return wl, rgb


def match_camera_response(camera_id: str, candidates: list[CameraSpectralResponse]) -> dict:
    """Exact camera identity only; sharing a sensor does not qualify a response.

    Qualification checks the supplied spectral contract, not its empirical
    accuracy. A match therefore remains offline research, never production.
    """
    matched = [entry for entry in candidates if entry.camera_id == camera_id]
    if not matched:
        return {"status": "unmeasured", "reason": "no-exact-camera-response", "production_eligible": False}
    if len(matched) > 1:
        return {"status": "ambiguous", "reason": "multiple-exact-camera-responses", "production_eligible": False}
    try:
        matched[0].arrays()
    except ValueError as error:
        return {"status": "unqualified", "reason": str(error), "production_eligible": False}
    return {"status": "matched-research", "camera_id": camera_id,
            "source_id": matched[0].source_id, "production_eligible": False}


def _sample(spectrum: Spectrum, grid: np.ndarray, *, quantity: tuple[str, ...]) -> np.ndarray:
    if spectrum.quantity not in quantity:
        raise ValueError(f"{spectrum.measurement_id}: unexpected quantity {spectrum.quantity}")
    wl, values = spectrum.arrays()
    if grid[0] < wl[0] or grid[-1] > wl[-1]:
        raise ValueError(f"{spectrum.measurement_id}: wavelength coverage is incomplete; no extrapolation")
    sampled = np.interp(grid, wl, values)
    if spectrum.quantity == "relative_spectral_transmittance":
        sampled = sampled / np.max(values)  # arbitrary relative global scale has no exposure meaning
    return sampled


def camera_responses(camera: CameraSpectralResponse, illuminant: Spectrum,
                     reflectances: list[Spectrum], target_optics: Spectrum, *,
                     reference_optics: Spectrum | None = None,
                     wavelength_range_nm: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
    """Integrate known spectra, with no silent wavelength extension or optical duplication.

    For a full-camera SSF, replace its reference optics using target/reference.
    For a sensor-only SSF, multiply target once. A reduced wavelength range is
    allowed only explicitly and is recorded as a limited-range experiment.
    """
    grid, response = camera.arrays()
    original_range = (float(grid[0]), float(grid[-1]))
    if wavelength_range_nm is not None:
        lo, hi = map(float, wavelength_range_nm)
        if not np.isfinite([lo, hi]).all() or not original_range[0] <= lo < hi <= original_range[1]:
            raise ValueError("invalid explicit wavelength range")
        selected = (grid >= lo) & (grid <= hi)
        grid, response = grid[selected], response[selected]
        if len(grid) < 2 or grid[0] != lo or grid[-1] != hi:
            raise ValueError("explicit range endpoints must be present on the camera grid")
    light = _sample(illuminant, grid, quantity=("energy_spd", "photon_spd"))
    if illuminant.quantity == "energy_spd" and camera.response_convention == "photon":
        light = light * grid / 555.0
    elif illuminant.quantity == "photon_spd" and camera.response_convention == "energy":
        light = light * 555.0 / grid
    # Normalize illumination for color-only integration, not an electron count.
    light = light / np.max(light)
    target = _sample(target_optics, grid, quantity=("relative_spectral_transmittance",))
    if camera.includes_reference_optics:
        if reference_optics is None or reference_optics.measurement_id != camera.reference_optical_stack:
            raise ValueError("full-camera SSF requires the exact reference optical spectrum")
        reference = _sample(reference_optics, grid, quantity=("relative_spectral_transmittance",))
        if np.any(reference <= 0):
            raise ValueError("reference optical spectrum has zero transmission in evaluated range")
        effective = response * (target / reference)[:, None]
    else:
        effective = response * target[:, None]
    if not reflectances:
        raise ValueError("no reflectance spectra")
    rows = []
    for reflectance in reflectances:
        material = _sample(reflectance, grid, quantity=("reflectance",))
        rows.append(np.trapezoid(material[:, None] * light[:, None] * effective, x=grid, axis=0))
    result = np.asarray(rows)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite spectral integration")
    evidence = {
        "status": "offline-research", "pixel_correction_applied": False,
        "absolute_throughput_available": False,
        "camera_id": camera.camera_id, "camera_source_id": camera.source_id,
        "response_convention": camera.response_convention,
        "channel_normalization": camera.channel_normalization,
        "reference_optical_stack": camera.reference_optical_stack,
        "includes_reference_optics": camera.includes_reference_optics,
        "target_optical_stack": target_optics.measurement_id,
        "illuminant_id": illuminant.measurement_id,
        "reflectance_ids": [x.measurement_id for x in reflectances],
        "wavelength_range_nm": [float(grid[0]), float(grid[-1])],
        "full_camera_wavelength_range_nm": list(original_range),
        "range_was_explicitly_restricted": wavelength_range_nm is not None,
        "interpolation": "linear-within-measured-range; trapezoid-integration",
    }
    return result, evidence


def neutral_white_balance_suggestion(camera: CameraSpectralResponse, illuminant: Spectrum,
                                     target_optics: Spectrum, reference_optics: Spectrum, *,
                                     existing_wb_state: str,
                                     wavelength_range_nm: tuple[float, float] | None = None) -> dict:
    """Suggest a relative WB delta only when existing WB describes the reference.

    AsShot/gray-card WB for the current target optics already includes their
    neutral effect. Unknown provenance cannot justify a second compensation.
    """
    if existing_wb_state in ("as-shot-target", "gray-card-target", "profile-includes-target"):
        return {"status": "already-compensated", "relative_rgb_gains": [1.0, 1.0, 1.0],
                "reason": "existing-white-balance-includes-target-optics", "pixel_correction_applied": False}
    if existing_wb_state != "reference-optics":
        return {"status": "unavailable", "relative_rgb_gains": None,
                "reason": "white-balance-reference-unknown", "pixel_correction_applied": False}
    grid, _ = camera.arrays()
    neutral = Spectrum("neutral-unit-reflectance", grid, np.ones(len(grid)), "reflectance")
    target, evidence = camera_responses(camera, illuminant, [neutral], target_optics,
                                       reference_optics=reference_optics,
                                       wavelength_range_nm=wavelength_range_nm)
    reference, _ = camera_responses(camera, illuminant, [neutral], reference_optics,
                                    reference_optics=reference_optics,
                                    wavelength_range_nm=wavelength_range_nm)
    if np.any(target <= 0) or np.any(reference <= 0):
        raise ValueError("neutral response has an unobservable channel")
    gains = (reference / target)[0]
    gains = gains / gains[1]
    return {**evidence, "status": "model-suggestion", "relative_rgb_gains": gains.tolist(),
            "white_balance_basis": "reference-optics", "automatic_application": False}


def fit_white_constrained_matrix(source_rgb: Any, target_rgb: Any, *,
                                 source_white: Any, target_white: Any) -> np.ndarray:
    """Fit M for y=Mx with exact white constraint; no claim of universal RGB inversion."""
    x, y = np.asarray(source_rgb, dtype=np.float64), np.asarray(target_rgb, dtype=np.float64)
    ws, wt = np.asarray(source_white, dtype=np.float64), np.asarray(target_white, dtype=np.float64)
    if (x.ndim != 2 or x.shape != y.shape or x.shape[1] != 3 or len(x) < 4
            or ws.shape != (3,) or wt.shape != (3,) or not np.isfinite(x).all()
            or not np.isfinite(y).all() or not np.isfinite(ws).all()
            or not np.isfinite(wt).all() or np.linalg.norm(ws) <= 0):
        raise ValueError("invalid matrix fitting inputs")
    normal = x.T @ x
    if np.linalg.matrix_rank(x) != 3 or np.linalg.cond(normal) > 1e10:
        raise ValueError("matrix training set is rank deficient or ill conditioned")
    kkt = np.block([[normal, ws[:, None]], [ws[None, :], np.zeros((1, 1))]])
    solved = np.linalg.solve(kkt, np.vstack([x.T @ y, wt]))
    return solved[:3].T


def propagate_color_covariance(matrix: Any, covariance: Any) -> np.ndarray:
    """Exact linear covariance update M Sigma M^T; rejects invalid covariance."""
    m, c = np.asarray(matrix, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    if (m.shape != (3, 3) or c.shape != (3, 3) or not np.isfinite(m).all()
            or not np.isfinite(c).all() or not np.allclose(c, c.T, rtol=1e-12, atol=1e-15)
            or np.linalg.eigvalsh(c).min() < -1e-14):
        raise ValueError("invalid color matrix or covariance")
    return m @ c @ m.T


def color_error_report(predicted_xyz: Any, reference_xyz: Any, *, white_xyz: Any,
                       sample_ids: list[str], illuminant_id: str) -> dict:
    """Report individual held-out DeltaE76, neutral/hue/chroma, with tails.

    XYZ values and reference white must share a relative scale. This implements
    CIELAB explicitly for an offline experiment; a small average is not a gate.
    """
    p, r = np.asarray(predicted_xyz, float), np.asarray(reference_xyz, float)
    w = np.asarray(white_xyz, float)
    if (p.shape != r.shape or p.ndim != 2 or p.shape[1] != 3 or len(p) != len(sample_ids)
            or not len(p) or w.shape != (3,) or np.any(w <= 0)
            or not np.isfinite(p).all() or not np.isfinite(r).all() or not np.isfinite(w).all()):
        raise ValueError("invalid held-out XYZ color report")
    def lab(v):
        t = v / w
        delta = 6 / 29
        f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4 / 29)
        return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                                200 * (f[:, 1] - f[:, 2])])
    pl, rl = lab(p), lab(r)
    error = np.linalg.norm(pl - rl, axis=1)
    chroma_p, chroma_r = np.linalg.norm(pl[:, 1:], axis=1), np.linalg.norm(rl[:, 1:], axis=1)
    hue_p, hue_r = np.arctan2(pl[:, 2], pl[:, 1]), np.arctan2(rl[:, 2], rl[:, 1])
    hue = np.degrees(np.abs(np.arctan2(np.sin(hue_p-hue_r), np.cos(hue_p-hue_r))))
    chromatic = (chroma_p > 1e-6) & (chroma_r > 1e-6)
    return {"illuminant_id": illuminant_id, "sample_ids": sample_ids,
            "delta_e76": error.tolist(), "mean_delta_e76": float(error.mean()),
            "p95_delta_e76": float(np.percentile(error, 95)), "max_delta_e76": float(error.max()),
            "chroma_error": (chroma_p-chroma_r).tolist(),
            "hue_error_degrees": [float(v) if ok else None for v, ok in zip(hue, chromatic)],
            "lightness_error": (pl[:, 0]-rl[:, 0]).tolist(),
            "neutral_reference_chroma": chroma_r.tolist()}


def experiment_identity(payload: dict) -> str:
    """Content identity for offline experiment inputs and algorithm contract."""
    encoded = json.dumps({"algorithm": "agxraw-spectral-color-research-1", "inputs": payload},
                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
