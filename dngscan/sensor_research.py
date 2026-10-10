# SPDX-License-Identifier: GPL-3.0-or-later
"""Explicit, opt-in sensor precision experiments; never a default RAW path.

All signal/noise inputs come from independent calibration, not scene texture.
These bounded prototypes deliberately do not claim LibRaw-quality demosaicing,
arbitrary opcode compatibility, or a calibrated adaptive-decoder covariance.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
from numbers import Real
from typing import Mapping

from ._deps import np

LUMA = np.asarray([.2627, .6780, .0593], dtype=np.float64)
PHASES = ("C00", "C01", "C10", "C11")


def _finite(value, name):
    try:
        value = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric scalar")
    return value


@dataclass(frozen=True)
class PersonalBlackCalibration:
    """One measured ISO/readout/body; values in linearized RAW DN.

    ``absolute`` replaces only the per-phase metadata baseline, retaining
    the metadata's spatial residual. ``residual`` adds a measured residual
    correction after the complete metadata black field. Unknown capture
    conditions cannot satisfy declared bounds. No interpolation is implied.
    Standard production metadata remains authoritative outside this prototype.
    """
    make: str
    model: str
    body_serial: str
    readout_identity: str
    iso: float
    exposure_seconds: tuple[float, float]
    phase_dn: Mapping[str, float]
    phase_uncertainty_dn: Mapping[str, float]
    source_hash: str
    mode: str = "absolute"
    domain: str = "linearized-raw-dn-before-stage2"
    temperature_c: tuple[float, float] | None = None
    firmware: str | None = None
    valid_until_utc: str | None = None

    def identity(self):
        """Include every applicability/value field in the cache identity."""
        self.validate()
        value = {field.name: getattr(self,field.name) for field in fields(self)}
        value.update(iso=float(self.iso), exposure_seconds=list(map(float,self.exposure_seconds)),
                     phase_dn={key:float(v) for key,v in self.phase_dn.items()},
                     phase_uncertainty_dn={key:float(v) for key,v in self.phase_uncertainty_dn.items()})
        if self.temperature_c is not None:
            value["temperature_c"] = list(map(float,self.temperature_c))
        return hashlib.sha256(json.dumps(value, sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def validate(self):
        if self.mode not in ("absolute", "residual"):
            raise ValueError("black calibration mode must be absolute or residual")
        if self.domain != "linearized-raw-dn-before-stage2":
            raise ValueError("black calibration domain/order is unsupported")
        if any(not isinstance(x, str) or not x.strip() for x in
               (self.make, self.model, self.body_serial, self.readout_identity, self.source_hash)):
            raise ValueError("personal black calibration needs body/readout/source identity")
        if _scalar(self.iso, "calibration ISO") <= 0:
            raise ValueError("invalid calibration ISO")
        if self.exposure_seconds is None:
            raise ValueError("personal black calibration needs measured exposure bounds")
        for key, bounds in (("exposure", self.exposure_seconds), ("temperature", self.temperature_c)):
            if bounds is not None:
                values = _finite(bounds, key)
                if values.shape != (2,) or values[0] > values[1] or (key == "exposure" and values[0] <= 0):
                    raise ValueError(f"invalid {key} bounds")
                for value in bounds:
                    _scalar(value, key)
        if not isinstance(self.phase_dn, Mapping) or not isinstance(self.phase_uncertainty_dn, Mapping):
            raise ValueError("phase values and uncertainties must be mappings")
        if set(self.phase_dn) != set(PHASES) or set(self.phase_uncertainty_dn) != set(PHASES):
            raise ValueError("prototype requires four explicitly measured Bayer phases")
        for value in self.phase_dn.values():
            _scalar(value, "phase black")
        if any(_scalar(value, "black uncertainty") < 0 for value in self.phase_uncertainty_dn.values()):
            raise ValueError("black uncertainty must be nonnegative")
        if self.firmware is not None and (not isinstance(self.firmware,str) or not self.firmware.strip()):
            raise ValueError("calibration firmware must be nonempty text when declared")
        if self.valid_until_utc is not None:
            from datetime import datetime
            try:
                end = datetime.fromisoformat(self.valid_until_utc.replace("Z", "+00:00"))
                if end.tzinfo is None:
                    raise ValueError("expiration requires a timezone")
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("calibration expiration must be a timezone-aware ISO timestamp") from exc

    def match(self, capture):
        from datetime import datetime
        self.validate()
        if not isinstance(capture, Mapping):
            raise ValueError("capture conditions must be a mapping")
        for key in ("make", "model", "body_serial", "readout_identity", "firmware"):
            required = getattr(self, key)
            if required is not None and capture.get(key) != required:
                return False, f"{key}-mismatch-or-unknown"
        try:
            iso = _scalar(capture.get("iso"), "capture ISO")
        except ValueError:
            return False, "iso-mismatch-or-unknown"
        if iso != self.iso:
            return False, "iso-mismatch-or-unknown"
        for key, bounds in (("exposure_seconds", self.exposure_seconds), ("temperature_c", self.temperature_c)):
            if bounds is not None:
                try:
                    value = _scalar(capture.get(key), key)
                except ValueError:
                    return False, f"{key}-outside-measured-domain"
                if not bounds[0] <= value <= bounds[1]:
                    return False, f"{key}-outside-measured-domain"
        if self.valid_until_utc is not None:
            now = capture.get("evaluation_utc")
            try:
                end = datetime.fromisoformat(self.valid_until_utc.replace("Z", "+00:00"))
                current = datetime.fromisoformat(now.replace("Z", "+00:00"))
                if end.tzinfo is None or current.tzinfo is None or current > end:
                    return False, "calibration-expired-or-time-unknown"
            except (ValueError, AttributeError, TypeError):
                return False, "calibration-expired-or-time-unknown"
        return True, "matched-personal-measurement"


def signed_black_correct(raw, metadata_black, calibration, capture, *, metadata_phase_black=None):
    """Return a new signed float64 buffer and provenance, or reject mismatch.

    Caller supplies the post-LUT, pre-stage2 work buffer and complete metadata
    black field. Original RAW/evidence is never mutated. A measured systematic
    uncertainty remains reported; it is not random pixel noise or exact zero.
    """
    matched, reason = calibration.match(capture)
    if not matched:
        raise ValueError(reason)
    image = _finite(raw, "RAW")
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("signed black prototype requires a Bayer raster")
    black = np.broadcast_to(_finite(metadata_black, "metadata black"), image.shape)
    if calibration.mode == "absolute" and (not isinstance(metadata_phase_black,Mapping) or set(metadata_phase_black) != set(PHASES)):
        raise ValueError("absolute replacement requires metadata phase baselines")
    if calibration.mode == "absolute":
        for value in metadata_phase_black.values():
            _scalar(value, "metadata phase black")
    corrected = image - black
    for phase in PHASES:
        dy, dx = int(phase[1]), int(phase[2])
        offset = float(calibration.phase_dn[phase])
        if calibration.mode == "absolute":
            offset -= float(_finite(metadata_phase_black[phase], "metadata phase black"))
        corrected[dy::2, dx::2] -= offset
    return corrected, {"status": "experimental-personal-black", "identity": calibration.identity(),
                       "domain": calibration.domain, "mode": calibration.mode,
                       "source_hash": calibration.source_hash,
                       "phase_uncertainty_dn": dict(calibration.phase_uncertainty_dn),
                       "low_tail_fraction": float(np.mean(corrected < 0)),
                       "low_tail_is_not_a_bad_pixel_mask": True}


def _tent_filter(image):
    value = np.asarray(image, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * value.ndim
        pad[axis] = (1, 1)
        padded = np.pad(value, pad, mode="reflect")
        slices = []
        for offset in range(3):
            sl = [slice(None)] * value.ndim
            sl[axis] = slice(offset, offset + value.shape[axis])
            slices.append(padded[tuple(sl)])
        value = (slices[0] + 2 * slices[1] + slices[2]) * .25
    return value


def signed_bilinear_bayer(raw, phase_colours, *, phase_gains=None):
    """Simple linear, sign-preserving reference; no highlight reconstruction.

    Colours are spatial C00,C01,C10,C11, not LibRaw colour-index order. Phase
    WB/gains apply before normalized linear interpolation. This is an audited
    precision reference, not a proposed replacement for adaptive demosaicing.
    """
    raw = _finite(raw, "signed CFA")
    if raw.ndim != 2 or min(raw.shape) < 4 or sorted(phase_colours) != ["B", "G", "G", "R"]:
        raise ValueError("requires a Bayer raster and explicit RGB phase colours")
    gains = _finite(phase_gains if phase_gains is not None else [1.] * 4, "phase gains")
    if gains.shape != (4,) or np.any(gains <= 0):
        raise ValueError("invalid phase gains")
    result = np.empty((*raw.shape, 3), dtype=np.float64)
    for ci, colour in enumerate("RGB"):
        samples, weights = np.zeros_like(raw), np.zeros_like(raw)
        for i, label in enumerate(phase_colours):
            if label == colour:
                dy, dx = divmod(i, 2)
                samples[dy::2, dx::2] = raw[dy::2, dx::2] * gains[i]
                weights[dy::2, dx::2] = 1.
        result[..., ci] = _tent_filter(samples) / _tent_filter(weights)
    return result


def dark_stack_products(frames, *, calibration_identity):
    """Separate repeatable mean structure from same-pixel temporal variance.

    Units are supplied linearized RAW DN. Input frames must share ISO,
    exposure, temperature/readout and have no changing scene; identity carries
    that independently checked contract. A mean map contains finite-stack
    uncertainty, so it is not automatically authorized for pixel subtraction.
    """
    samples = _finite(frames, "dark stack")
    if (samples.ndim != 3 or samples.shape[0] < 4 or min(samples.shape[1:]) < 1
            or not isinstance(calibration_identity,str) or not calibration_identity.strip()):
        raise ValueError("need at least four matched dark frames and an identity")
    mean_map = samples.mean(axis=0)
    temporal = samples.var(axis=0, ddof=1)
    difference = samples[1::2] - samples[:samples.shape[0] // 2 * 2:2]
    pair_variance = np.mean(np.square(difference), axis=0) * .5
    mean_uncertainty = np.sqrt(temporal / samples.shape[0])
    residual = mean_map - np.median(mean_map)
    # A detection proposal, never interpolation or generic denoising.
    hot_candidate = residual > np.maximum(8 * mean_uncertainty, 8.)
    return {"identity": calibration_identity, "domain": "linearized-raw-dn",
            "frame_count": samples.shape[0], "mean_bias_map_dn": mean_map,
            "temporal_variance_dn2": temporal, "pair_variance_dn2": pair_variance,
            "mean_uncertainty_dn": mean_uncertainty,
            "row_mean_bias_dn": residual.mean(axis=1),
            "column_mean_bias_dn": residual.mean(axis=0),
            "hot_pixel_candidates": hot_candidate,
            "correction_authorized": False}


def joint_chroma_shrink(detail, covariance, *, strength=1., max_condition=1e6):
    """Full 2D zero-luminance covariance, whitened vector soft shrink.

    Only a known linear detail operator with independently propagated 3x3
    covariance is admitted. Rank-deficient/ill-conditioned 2D models are
    rejected, never repaired by arbitrary diagonal noise. A zero strength
    preserves all input detail; output includes untouched luminance detail.
    Shrinkage is biased for real low-SNR chroma; this prototype does not infer
    which weak vector is real structure and remains off the default pipeline.
    """
    detail = _finite(detail, "detail")
    covariance = _finite(covariance, "detail covariance")
    if detail.shape[-1:] != (3,) or covariance.shape[-2:] != (3, 3):
        raise ValueError("RGB detail and 3x3 covariance required")
    if not math.isfinite(strength) or strength < 0 or not math.isfinite(max_condition) or max_condition < 1:
        raise ValueError("invalid shrinkage contract")
    if not np.allclose(covariance, np.swapaxes(covariance, -1, -2), rtol=1e-10, atol=1e-16):
        raise ValueError("covariance must be symmetric")
    eigen = np.linalg.eigvalsh(covariance)
    scale = np.max(np.abs(eigen), axis=-1)
    if np.any(eigen[..., 0] < -1e-12 * np.maximum(scale, np.finfo(float).tiny)):
        raise ValueError("covariance must be positive semidefinite")
    # The SVD nullspace is an orthonormal basis of {v: LUMA @ v = 0}.
    basis = np.linalg.svd(LUMA[None, :], full_matrices=True)[2][1:].T
    projection = np.eye(3) - np.ones((3, 1)) * LUMA[None, :]
    chroma = detail @ projection.T
    c2 = np.einsum("ia,ij,...jk,lk,lb->...ab", basis, projection, covariance, projection, basis)
    vals, vecs = np.linalg.eigh(c2)
    if np.any(vals[..., 0] <= 0) or np.any(vals[..., 1] / vals[..., 0] > max_condition):
        raise ValueError("singular or ill-conditioned chroma covariance")
    coords = chroma @ basis
    eig_coords = np.einsum("...ia,...i->...a", vecs, coords)
    white = eig_coords / np.sqrt(vals)
    norm = np.linalg.norm(white, axis=-1, keepdims=True)
    factor = np.maximum(0., 1. - strength * np.sqrt(2.) / np.maximum(norm, np.finfo(float).tiny))
    recovered = np.einsum("...ia,...a->...i", vecs, white * factor * np.sqrt(vals)) @ basis.T
    output = recovered + (detail @ LUMA)[..., None]
    return output, {"model": "experimental-full-2d-chroma", "max_condition": float(np.max(vals[..., 1] / vals[..., 0])),
                    "low_snr_structure_is_not_identifiable": True}


def quantized_gain_chain(raw, gains, *, white, policy):
    """Audit repeated GainMap boundaries in original stored DN units.

    ``truncate`` mirrors uint16 handoffs; ``nearest`` isolates their rounding
    choice; ``deferred`` retains fractional values until one final boundary.
    The upper clip stays at each operation, matching a declared clip policy.
    """
    result = _finite(raw, "RAW").copy()
    if policy not in ("truncate", "nearest", "deferred") or not math.isfinite(white) or white <= 0:
        raise ValueError("invalid gain quantization policy")
    for gain in gains:
        gain = _finite(gain, "gain")
        if np.any(gain <= 0):
            raise ValueError("gain must be positive")
        result = np.clip(result * gain, 0, white)
        if policy == "truncate":
            result = np.floor(result)
        elif policy == "nearest":
            result = np.rint(result)
    return np.rint(result) if policy == "deferred" else result
