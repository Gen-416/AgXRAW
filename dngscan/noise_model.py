# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene-independent noise evidence in black-subtracted, normalized RAW units.

Local image variation is deliberately not an input to the model. A model
describes conditional variance, not which individual pixels are noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import struct
from typing import Any

from ._deps import np


@dataclass(frozen=True)
class NoiseModel:
    status: str = "unavailable"
    source: str = "none"
    reason: str = "no-matched-calibration"
    channel_variance: dict[str, tuple[float, float]] = field(default_factory=dict)
    domain: str = "normalized-raw"
    approximation: str | None = None
    correlation: str = "unknown"
    spectral_ratios: dict[str, float] = field(default_factory=dict)
    noise_reduction_status: str = "absent"
    fallback_source: str | None = None
    fallback_reason: str | None = None
    phase_variance: dict[str, tuple[float, float]] = field(default_factory=dict)
    phase_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def coefficients(self, label: str) -> tuple[float, float] | None:
        if self.status != "valid" or self.domain != "normalized-raw":
            return None
        value = self.channel_variance.get(label)
        if value is None:
            value = self.channel_variance.get(label[:1])
        return value


def _labels(bundle, ids):
    from .analysis import channel_labels
    return channel_labels(bundle.color_desc, ids)


def _phase_coefficients(bundle, prior, iso, scales, fallback):
    """Retain independently measured phase variance without inventing gain.

    Collect's stored variance includes the declared ADC contribution. A
    phase lacking an independently fitted gain keeps the scalar shot-noise
    ruler, explicitly labelled; its independently measured temporal variance
    can still replace the scalar constant term in the same reference DN.
    """
    from .calibration import curve_value, read_noise_issue
    from . import priors
    from .raw_units import normalized_raw_span

    products = prior.get("phase_calibration") or {}
    if not products:
        return {}, {}, fallback
    pattern = np.asarray(getattr(bundle, "raw_pattern", ()))
    if pattern.shape != (2, 2):
        return {}, {"scope": {"status": "unsupported-cfa"}}, fallback
    labels = _labels(bundle, sorted(set(map(int, pattern.flat))))
    phases, metadata, per_label = {}, {}, {}
    shared_gain = priors.gain_e_per_dn(prior, iso)
    for index, cid in enumerate(pattern.flat):
        key = f"C{index // 2}{index % 2}"
        cid = int(cid); label = labels[cid]
        product = products.get(key)
        if product is None:
            matches = [p for p in products.values() if p.get("channel") is None and p.get("color_index") == cid]
            product = matches[0] if len(matches) == 1 else None
        status = {"color_index": cid, "label": label, "status": "scalar-fallback"}
        status["normalized_raw_span_dn"] = normalized_raw_span(bundle, cid)
        metadata[key] = status
        if not product:
            status["reason"] = "phase-measurement-unavailable"
            continue
        if product.get("color_index") != cid or (product.get("color_desc") is not None
                                                  and product["color_desc"] != bundle.color_desc):
            status["reason"] = "phase-identity-mismatch"
            continue
        if product.get("fit_quality") not in (None, "ok"):
            status["reason"] = "phase-fit-quality-" + product["fit_quality"]
            continue
        phase_prior = dict(product, gain_jump_isos=prior.get("gain_jump_isos", []))
        issue = read_noise_issue(phase_prior, iso, "read_noise_dn_log2iso")
        # Stored total variance may still have a measured point where the
        # physical read component was unresolved. Do not use that point as
        # an interpolation endpoint while losing its explicit failure state.
        variance_curve = phase_prior.get("stored_dark_variance_dn2_log2iso") or []
        x = math.log2(iso)
        if issue is None and not any(abs(x-px)<1e-7 for px,_ in variance_curve):
            if any(x0 < x < x1 and any(x0 <= math.log2(failed) <= x1
                    for failed in phase_prior.get("read_noise_unresolved_isos", []))
                    for (x0,_),(x1,_) in zip(variance_curve,variance_curve[1:])):
                issue = "read-noise-unresolved-interval"
        if issue:
            status["reason"] = issue
            continue
        stored = curve_value(phase_prior, "stored_dark_variance_dn2_log2iso", iso)
        if stored is None or not math.isfinite(stored) or stored < 0:
            status["reason"] = "phase-stored-variance-unavailable"
            continue
        scale = scales[label]
        a = fallback[label][0]
        measured_gain = shared_gain
        gain = curve_value(phase_prior, "gain_log2iso_log2epd", iso)
        if gain is not None:
            measured_gain = 2. ** gain
            reference = product.get("reference_dn_range")
            span = normalized_raw_span(bundle, cid)
            if reference is None or not math.isfinite(reference) or reference <= 0:
                status["reason"] = "phase-gain-dn-reference-unavailable"
                continue
            ratio = span / reference
            power = round(math.log2(ratio))
            if abs(math.log2(ratio) - power) > math.log2(1.05):
                status["reason"] = "phase-gain-dn-scale-mismatch"
                continue
            scale = measured_gain * span / (2. ** power)
            a = 1. / scale
            status["gain_source"] = "independent-phase-fit"
        else:
            status["gain_source"] = "shared-scalar-approximation"
        b = stored * (measured_gain / scale) ** 2
        if not all(math.isfinite(x) and x >= 0 for x in (a, b)) or a == 0:
            status["reason"] = "nonfinite-phase-model"
            continue
        phases[key] = (a, b)
        status.update(status="measured-phase-temporal-variance", variance_domain="stored-reference-dn2",
                      stored_variance_dn2=stored,
                      uncertainty=product.get("uncertainty") or "not-quantified")
        per_label.setdefault(label, []).append((a, b))
    channels = dict(fallback)
    # Multiple phases sharing a label describe the average physical sensel
    # distribution here; the coarse renderer below uses separate squared
    # averaging weights, rather than averaging their standard deviations.
    for label, values in per_label.items():
        if len(values) == sum(labels[int(cid)] == label for cid in pattern.flat):
            channels[label] = tuple(map(float, np.mean(values, axis=0)))
    return phases, metadata, channels


def _prior_signal_scales(bundle, fullwell, prior, iso):
    """Validate the file's DN/readout scale independently of read-noise fit."""
    from . import priors
    from .raw_units import normalized_raw_span

    ids = sorted(fullwell)
    labels = _labels(bundle, ids)
    scales = {}
    for cid in ids:
        span = normalized_raw_span(bundle, cid)
        if not math.isfinite(span) or span <= 0:
            return None
        gain = priors.gain_for_file(prior, iso, span)
        if gain is None or not math.isfinite(gain) or gain <= 0:
            return None
        scale = gain * span
        if not math.isfinite(scale) or scale <= 0:
            return None
        scales[labels[cid]] = scale
    return scales


def _prior_spectrum(prior, iso):
    """Independent measured constraints; read-noise resolution is irrelevant."""
    from .calibration import curve_value
    from .noise_spectrum import validate_spectrum

    spectral_ratios = {}
    measured = prior.get("noise_spectrum")
    qualified_axes = set()
    if measured is not None:
        measured = validate_spectrum(measured)
        mapping = measured.get("mapping", {})
        if mapping.get("status") in ("bayer", "single-colour"):
            for axis, record in measured.get("axes", {}).items():
                for phase, curve in record.get("ratios_log2iso", {}).items():
                    colour = mapping.get("phases", {}).get(phase, {}).get("color")
                    if colour not in ("R", "G", "B") and mapping["status"] != "single-colour":
                        continue
                    ratio = curve_value({"curve": curve,
                                         "gain_jump_isos": prior.get("gain_jump_isos", [])}, "curve", iso)
                    if ratio is not None:
                        spectral_ratios[f"{axis}:{phase}"] = float(ratio)
                        qualified_axes.add(axis)
    for axis in ("h", "v"):
        if axis in qualified_axes:
            continue
        key = f"noise_whiteness_{axis}_log2iso"
        ratio = curve_value(prior, key, iso)
        if ratio is None:
            # Older Collect JSON rounded this diagnostic's log2 ISO to four
            # decimals. Recover its nominal measured point only; do not relax
            # the gain/read-noise domain or extrapolate a spectral curve.
            x = math.log2(iso)
            ratio = next((float(y) for px, y in prior.get(key) or ()
                          if abs(x - px) <= 5.1e-5), None)
        if ratio is not None and math.isfinite(ratio) and ratio >= 0:
            spectral_ratios[axis] = float(ratio)
    # This is only a conservative incompatibility guard for the independent
    # coarse approximation. A high/mid PSD ratio near one does not prove white
    # noise or exclude narrow peaks. Never derive this flag from scene texture.
    correlation = ("measured-spectral-imbalance"
                   if any(r < .5 or r > 2 for r in spectral_ratios.values())
                   else "measured-spectrum-summary" if spectral_ratios else "unknown")
    return correlation, spectral_ratios


def model_from_prior(bundle, fullwell: dict[int, float], prior) -> NoiseModel:
    from . import priors

    if prior is None:
        return NoiseModel()
    prior = priors.with_readout(prior, getattr(bundle, "capture_readout", None))
    usable, reason = priors.prior_usability(prior)
    source = str(prior.get("source") or prior.get("id") or "sensor-prior")
    if not usable:
        return NoiseModel(status="rejected", source=source, reason=reason)
    iso = getattr(bundle, "shot_iso", None)
    if not iso or iso <= 0:
        return NoiseModel(source=source, reason="iso-unavailable")
    if prior.get("suspect_iso_min") and iso >= prior["suspect_iso_min"]:
        return NoiseModel(status="rejected", source=source, reason="suspect-iso")
    # A failed read-noise fit does not invalidate a separate spectrum
    # measurement. Its camera/mode/ISO and DN-scale checks still apply.
    scales = _prior_signal_scales(bundle, fullwell, prior, iso)
    try:
        correlation, spectral_ratios = _prior_spectrum(prior, iso) if scales else ("unknown", {})
    except (ValueError, TypeError, KeyError, OverflowError):
        return NoiseModel(status="rejected", source=source, reason="invalid-spectral-evidence")
    model = NoiseModel(source=source, correlation=correlation, spectral_ratios=spectral_ratios)
    from .calibration import read_noise_issue
    issue = read_noise_issue(prior, iso)
    if issue:
        return replace(model, status="unresolved", reason=issue)
    read = priors.read_noise_e(prior, iso)
    if read is None or not math.isfinite(read) or read < 0:
        return replace(model, reason="read-noise-unavailable")
    if scales is None:
        return replace(model, status="rejected", reason="unmatched-dn-scale")
    if not scales:
        return replace(model, reason="raw-channels-unavailable")
    from .calibration import stored_dark_variance
    stored_variance, variance_source = stored_dark_variance(prior, iso)
    if variance_source in ("stored-dark-variance-below-physical-read-variance",
                           "stored-dark-variance-nonfinite"):
        # Contradictory total-variance evidence is unresolved, not missing.
        # Keep its independent spectrum and permit a valid DNG profile to
        # supply alternative coefficients through the existing resolver.
        return replace(model, status="unresolved", reason=variance_source,
                       approximation=variance_source)
    measured_gain = None
    if stored_variance is not None:
        # Stored variance is in the measurement's reference DN, while scales
        # are electrons per normalized file sample. Use measured e-/DN here;
        # gain_for_file already checked any power-of-two DN storage transport.
        measured_gain = priors.gain_e_per_dn(prior, iso)
        if measured_gain is None or not math.isfinite(measured_gain) or measured_gain <= 0:
            return replace(model, status="unresolved", reason="stored-dark-variance-gain-unavailable")
    approximation = "shared-gain-read-noise-across-colour-planes"
    approximation += "; " + str(prior.get("readout_match_reason") or "sub-readout-not-declared")
    if prior.get("source_format") == "dngscan-jptc-collect-1":
        approximation += "; " + variance_source
    try:
        coefficients = {label: (1.0 / scale, (read / scale) ** 2 if stored_variance is None
                                 else stored_variance * (measured_gain / scale) ** 2)
                        for label, scale in scales.items()}
    except (OverflowError, ZeroDivisionError):
        coefficients = {}
    if not coefficients or any(not math.isfinite(a) or a <= 0 or not math.isfinite(b) or b < 0
                               for a, b in coefficients.values()):
        return replace(model, status="unresolved", reason="nonfinite-noise-model-coefficients",
                       approximation=approximation)
    try:
        phases, phase_metadata, coefficients = _phase_coefficients(bundle, prior, iso, scales, coefficients)
    except (ValueError, TypeError, KeyError, OverflowError):
        return replace(model, status="unresolved", reason="invalid-phase-calibration")
    if phases:
        approximation += "; measured-phase-temporal-variance; phase-gain-source-declared-separately"
    return replace(model, status="valid", reason="matched-shot-read-model",
                   channel_variance=coefficients, approximation=approximation,
                   phase_variance=phases, phase_metadata=phase_metadata)


def _file_model(bundle) -> NoiseModel:
    """Read only a CFA Raw IFD profile; never borrow an enhanced RGB profile."""
    from .spatial_black import sensor_tags
    from .metadata import UndefinedRational

    try:
        tags = sensor_tags(bundle.path, {51041, 50935, 50710})
    except (OSError, ValueError, OverflowError, TypeError, IndexError, struct.error):
        return NoiseModel(reason="file-noise-metadata-unreadable", noise_reduction_status="unreadable")
    reduction = tags.get(50935)
    declaration = "absent"
    if reduction is not None:
        try:
            if len(reduction) != 1:
                raise ValueError
            rational = reduction[0]
            if isinstance(rational, UndefinedRational) and rational.state == "unknown":
                declaration, value = "unknown", None
            else:
                value = float(rational)
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError
                declaration = "none" if value == 0 else "applied"
        except (ValueError, TypeError, IndexError, OverflowError):
            return NoiseModel(status="rejected", source="DNG NoiseReductionApplied",
                              reason="invalid-noise-reduction-declaration", noise_reduction_status="invalid")
        if value is not None and value > 0:
            return NoiseModel(status="rejected", source="DNG NoiseReductionApplied",
                              reason="raw-noise-reduction-declared", noise_reduction_status=declaration)
    def result(**kwargs):
        return NoiseModel(noise_reduction_status=declaration, **kwargs)
    if 51041 not in tags:
        return result()
    try:
        if tags.get(262, [None])[0] != 32803:
            return result(source="DNG NoiseProfile", reason="profile-is-not-cfa-raw")
        values = [float(x) for x in tags[51041]]
        ids = sorted(int(x) for x in np.unique(bundle.raw_colors))
        labels = _labels(bundle, ids)
        planes = list(tags.get(50710, [0, 1, 2]))
        if (not planes or len(planes) > 4 or len(set(planes)) != len(planes)
                or any(p not in (0, 1, 2) for p in planes)):
            raise ValueError("invalid-profile-colour-planes")
        if len(values) not in (2, 2 * len(planes)):
            raise ValueError("invalid-profile-count")
        if not all(math.isfinite(x) for x in values):
            raise ValueError("nonfinite-profile")
        pairs = list(zip(values[::2], values[1::2]))
        if any(a <= 0 or b < 0 for a, b in pairs):
            raise ValueError("invalid-profile-variance")
    except (ValueError, TypeError, IndexError, OverflowError) as exc:
        return result(status="rejected", source="DNG NoiseProfile",
                      reason=str(exc) or "malformed-file-noise-profile")
    coefficients = {}
    for cid in ids:
        label = labels[cid]
        code = {"R": 0, "G": 1, "B": 2}.get(label[:1])
        if code is None or code not in planes:
            return result(status="rejected", source="DNG NoiseProfile",
                          reason="unsupported-colour-plane")
        coefficients[label] = pairs[0] if len(pairs) == 1 else pairs[planes.index(code)]
    return result(status="valid", source="DNG NoiseProfile", reason="file-declared-model",
                  channel_variance=coefficients,
                  approximation="manufacturer-declared-white-noise-model")


def resolve_noise_model(bundle, fullwell: dict[int, float], prior=None) -> NoiseModel:
    raw = getattr(bundle, "raw_image", None)
    if raw is None:
        return NoiseModel(reason="sensor-evidence-unavailable")
    if raw.ndim != 2:
        return NoiseModel(reason="not-independent-cfa-samples")
    if prior is None:
        from .priors import find_priors
        prior = find_priors(bundle.shot_make, bundle.shot_model,
                            shutter=getattr(bundle, "shot_shutter", None),
                            iso=getattr(bundle, "shot_iso", None),
                            readout=getattr(bundle, "capture_readout", None))
    model = model_from_prior(bundle, fullwell, prior)
    file_model = _file_model(bundle)
    # A profile's variance values are an alternative source. The RAW's
    # declaration that processing was applied is a compatibility constraint
    # on every white-noise source, including a matched external calibration.
    if file_model.reason in ("raw-noise-reduction-declared", "invalid-noise-reduction-declaration"):
        return file_model
    if model.status == "valid":
        return replace(model, noise_reduction_status=file_model.noise_reduction_status)
    # File coefficients replace missing variance evidence, not independent
    # spectral constraints from the applicable external calibration.
    if file_model.status == "valid" and model.status in ("unresolved", "unavailable"):
        if model.status == "unresolved" or model.spectral_ratios:
            return replace(file_model,
                           reason=("file-declared-model-after-unresolved-prior"
                                   if model.status == "unresolved" else file_model.reason),
                           fallback_source=model.source, fallback_reason=model.reason,
                           correlation=model.correlation,
                           spectral_ratios=dict(model.spectral_ratios))
    if model.spectral_ratios and file_model.status == "unavailable":
        # An unusable file profile cannot erase independent negative evidence
        # either (for example, a profile attached to an enhanced RGB IFD).
        return replace(model, noise_reduction_status=file_model.noise_reduction_status)
    if model.status == "unresolved":
        if file_model.status == "unavailable":
            return replace(model, noise_reduction_status=file_model.noise_reduction_status)
        return replace(file_model, fallback_source=model.source, fallback_reason=model.reason)
    if model.status == "rejected" and file_model.status == "unavailable":
        return replace(model, noise_reduction_status=file_model.noise_reduction_status)
    if model.status == "rejected" and file_model.status == "valid":
        return replace(file_model, fallback_source=model.source, fallback_reason=model.reason)
    return (file_model if file_model.source != "none" else
            replace(model, noise_reduction_status=file_model.noise_reduction_status))


def model_snr_curves(model: NoiseModel, channel_ids, labels):
    """Evaluate a calibrated curve; scene texture never sets its denominator."""
    from .constants import EV_REPORT_FLOOR
    from .analysis import interpolate_zero_db_stop, rgb_channel_groups

    bins = np.linspace(EV_REPORT_FLOOR, 0.0, 85)
    centers = (bins[:-1] + bins[1:]) * 0.5
    signal = np.exp2(centers)
    curves, dr, zero = {}, {}, {}
    for group, ids in rgb_channel_groups(channel_ids, labels):
        candidates = []
        for cid in ids:
            coeff = model.coefficients(labels[cid])
            if coeff is not None:
                a, b = coeff
                candidates.append(20 * np.log10(signal / np.sqrt(a * signal + b)))
        snr = (np.minimum.reduce(candidates).astype(np.float32) if candidates
               else np.full(centers.shape, np.nan, np.float32))
        crossing = interpolate_zero_db_stop(centers, snr)
        curves[group] = {"stops": centers.copy(), "snr_db": snr,
                         "count": np.zeros(centers.shape, np.int32), "ids": ids,
                         "source": model.source, "kind": "model" if candidates else "unavailable"}
        zero[group] = crossing
        dr[group] = -crossing if math.isfinite(crossing) else float("nan")
    return curves, dr, zero


def model_read_floor(model: NoiseModel) -> float:
    """Conservative normalized read-noise floor, excluding scene structure."""
    if model.status != "valid" or not model.channel_variance:
        return float("nan")
    return max(math.sqrt(b) for _, b in model.channel_variance.values())
