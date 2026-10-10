# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured CFA spectral evidence, with positions kept distinct from colours."""
from __future__ import annotations

import math
import re

SCHEMA = "dngscan-noise-spectrum-1"
PHASES = ("C00", "C01", "C10", "C11")


def dark_phase_mapping(header, rows):
    """Resolve collector Cxx positions from measured Channel + ColorIndex.

    Upstream's CfaPattern header is LibRaw's colour *description*, not an
    RGGB layout. Missing metadata cannot establish a Bayer phase mapping.
    """
    description = str(header.get("CfaPattern", "")).strip().upper()
    mapping = {}
    if not description or len(description) > 4 or not description.isalpha():
        return {"status": "unavailable", "reason": "colour-description-unavailable", "phases": {}}
    for row in rows:
        phase = str(row.get("Channel", "")).strip()
        if not phase:
            continue
        if phase not in PHASES:
            raise ValueError("dark Channel must name a 2x2 CFA position C00/C01/C10/C11")
        try:
            index = int(row["ColorIndex"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dark CFA mapping requires a valid ColorIndex") from exc
        if not 0 <= index < len(description):
            raise ValueError("dark ColorIndex is outside the declared colour description")
        current = {"color_index": index, "color": description[index]}
        if phase in mapping and mapping[phase] != current:
            raise ValueError("dark CFA position-to-colour mapping changes across ISO or frames")
        mapping[phase] = current
    if not mapping:
        return {"status": "unavailable", "reason": "channel-position-mapping-unavailable",
                "color_description": description, "phases": {}}
    colours = [mapping[p]["color"] for p in PHASES if p in mapping]
    if len(mapping) != 4:
        status, reason = "partial", "incomplete-cfa-position-mapping"
    elif sorted(colours) == ["B", "G", "G", "R"]:
        status, reason = "bayer", "measured-position-to-colour-mapping"
    elif len(set(colours)) == 1:
        status, reason = "single-colour", "measured-single-colour-position-mapping"
    else:
        status, reason = "unsupported", "not-a-supported-2x2-colour-arrangement"
    return {"status": status, "reason": reason, "color_description": description,
            "phases": mapping}


def validate_spectrum(value):
    """Validate retained per-phase summaries before granting model authority."""
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported noise_spectrum schema")
    mapping = value.get("mapping") or {}
    if not isinstance(mapping, dict) or not isinstance(mapping.get("phases", {}), dict):
        raise ValueError("noise_spectrum mapping must be an object")
    phases = mapping.get("phases", {})
    description = mapping.get("color_description", "")
    if not isinstance(description, str):
        raise ValueError("noise_spectrum colour description must be text")
    for phase, record in phases.items():
        if (phase not in PHASES or not isinstance(record, dict) or
                type(record.get("color_index")) is not int or
                not 0 <= record["color_index"] < len(description) or
                record.get("color") != description[record["color_index"]]):
            raise ValueError("invalid noise_spectrum CFA mapping")
    if mapping.get("status") == "bayer":
        if set(phases) != set(PHASES) or sorted(r["color"] for r in phases.values()) != ["B", "G", "G", "R"]:
            raise ValueError("noise_spectrum Bayer mapping is incomplete")
    if mapping.get("status") == "single-colour":
        if set(phases) != set(PHASES) or len({r["color"] for r in phases.values()}) != 1:
            raise ValueError("noise_spectrum single-colour mapping is incomplete")
    axes = value.get("axes", {})
    if not isinstance(axes, dict) or any(axis not in ("h", "v") for axis in axes):
        raise ValueError("noise_spectrum axes must be h/v")
    for name, axis in axes.items():
        if not isinstance(axis, dict) or not isinstance(axis.get("ratios_log2iso", {}), dict):
            raise ValueError("noise_spectrum phase curves must be objects")
        for phase, curve in axis.get("ratios_log2iso", {}).items():
            if phase not in PHASES or not isinstance(curve, list):
                raise ValueError("invalid noise_spectrum phase curve")
            previous = -math.inf
            for point in curve:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("invalid noise_spectrum ISO ratio point")
                x, ratio = map(float, point)
                if not math.isfinite(x) or not math.isfinite(ratio) or x <= previous or ratio < 0:
                    raise ValueError("noise_spectrum ISO ratios must be finite and ordered")
                previous = x
        _validate_complete_axis(axis)
        if (axis.get("source_headers", {}).get("Axis") and
                axis["source_headers"]["Axis"] != {"h": "horizontal", "v": "vertical"}[name]):
            raise ValueError("complete spectrum axis disagrees with its source header")
    return value


def spectrum_curves(rows, freqs, lo=(.05, .20), hi=(.35, .499)):
    """Keep every measured position; a missing band has no ratio evidence."""
    import numpy as np

    curves = {}
    if not rows:
        return curves
    low = (freqs >= lo[0]) & (freqs <= lo[1])
    high = (freqs >= hi[0]) & (freqs <= hi[1])
    for column in rows[0]:
        match = re.fullmatch(r"iso([1-9][0-9]*)_(C[01][01])_diff", column)
        if not match:
            continue
        iso, phase = int(match[1]), match[2]
        power = np.asarray([float(row[column]) for row in rows])
        if not np.all(np.isfinite(power)) or np.any(power < 0):
            raise ValueError("spectrum power must be finite and nonnegative")
        if not low.any() or not high.any() or float(power[low].mean()) <= 0:
            continue
        ratio = float(power[high].mean() / power[low].mean())
        curves.setdefault(phase, []).append([math.log2(iso), ratio])
    return {phase: sorted(points) for phase, points in curves.items()}


def conservative_ratio(values):
    """A compatibility scalar which cannot average opposing anomalies away."""
    return max(values, key=lambda r: abs(math.log(max(float(r), 1e-300))))


def _normalization_contract(header, n):
    normalization = header.get("Normalisation", "")
    norm_declared = "".join(normalization.split(",", 1)[0].split()) == "|Y(k)|^2/(N*sum(w^2))"
    undoubled = "NOT doubled" in header.get("OneSided", "")
    try:
        diff_factor = float(header.get("DiffPowerFactor", "").split()[0])
    except (ValueError, IndexError):
        diff_factor = None
    if diff_factor is not None and not math.isfinite(diff_factor):
        diff_factor = None
    return n is not None and norm_declared and undoubled and diff_factor == 2., undoubled, diff_factor


def _integration_status(contract, window, comparisons):
    if not contract:
        return "incomplete-contract"
    if not comparisons:
        return "normalized-no-reference"
    if window in ("none", "rectangular"):
        return "verified-rectangular" if all(comparisons) else "reference-mismatch"
    if window == "hann":
        return "windowed-reference-comparison"
    return "window-unavailable"


def complete_axis_spectrum(header, rows, axis, *, scalar_rows=()):
    """Retain source powers and the exact contract needed to interpret them.

    A summary-only legacy CSV remains readable, but cannot silently acquire a
    transform length, window, normalization or Parseval verification.
    """
    import numpy as np
    from .spectral_variance import one_sided_power_mass, sensor_frequencies

    if axis not in ("h", "v"):
        raise ValueError("spectrum axis must be h/v")
    declared_axis = header.get("Axis")
    if declared_axis and declared_axis != {"h": "horizontal", "v": "vertical"}[axis]:
        raise ValueError("spectrum axis declaration disagrees with its file")
    frequencies = np.asarray([float(row["freq"]) for row in rows], dtype=np.float64)
    if (not np.isfinite(frequencies).all() or np.any(frequencies < 0) or
            np.any(frequencies > .5) or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("spectrum frequencies must be finite, increasing and within [0,.5]")
    n = None
    if header.get("TransformLength"):
        try:
            n = int(header["TransformLength"])
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid spectrum TransformLength") from exc
        if not 2 <= n <= 1_048_576 or len(frequencies) != n // 2 + 1:
            raise ValueError("spectrum bins disagree with TransformLength")
        if not np.allclose(frequencies, np.arange(len(frequencies)) / n, rtol=6e-6, atol=1e-9):
            raise ValueError("spectrum frequency bins disagree with TransformLength")
        if any("bin" in row and int(row["bin"]) != i for i, row in enumerate(rows)):
            raise ValueError("spectrum bin indexes are not consecutive")
    contract, undoubled, diff_factor = _normalization_contract(header, n)
    records = {}
    count = 0
    for column in rows[0] if rows else ():
        match = re.fullmatch(r"iso([1-9][0-9]*)_(C[01][01])_(single|diff)", column)
        if not match:
            continue
        iso, phase, source = int(match[1]), match[2], match[3]
        power = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
        count += power.size
        if count > 4_000_000:
            raise ValueError("complete spectrum exceeds the bounded calibration size")
        if not np.isfinite(power).all() or np.any(power < 0):
            raise ValueError("spectrum power must be finite and nonnegative")
        item = records.setdefault(iso, {}).setdefault(phase, {})
        item[source + "_power"] = power.tolist()
        if contract:
            variance = float(one_sided_power_mass(power, n, difference=source == "diff").sum())
            item["temporal_variance_dn2" if source == "diff" else "single_variance_dn2"] = variance
    ref_axis = "Row" if axis == "h" else "Col"
    refs = {}
    for row in scalar_rows:
        if not row.get("Channel") or not row.get("ISO"):
            continue
        key = (int(row["ISO"]), row["Channel"])
        current = {}
        for source in ("Single", "Diff"):
            raw = row.get(f"Within{ref_axis}Var{source}")
            if raw not in (None, ""):
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("within-line spectrum reference variance is invalid")
                current[source.lower()] = value
        if key in refs and refs[key] != current:
            raise ValueError("repeated ISO/phase scalar references cannot identify a spectrum pair")
        refs[key] = current
    for iso, phases in records.items():
        for phase, item in phases.items():
            comparisons = []
            for source, reference in refs.get((iso, phase), {}).items():
                item[f"reference_{source}_within_line_variance_dn2"] = reference
                integral = item.get("temporal_variance_dn2" if source == "diff" else "single_variance_dn2")
                if integral is not None and reference > 0:
                    ratio = integral * (2 if source == "diff" else 1) / reference
                    item[f"parseval_ratio_{source}"] = ratio
                    comparisons.append(abs(ratio - 1) <= 2e-5)
            item["integration_status"] = _integration_status(contract, header.get("Window"), comparisons)
    result = {
        "transform_length": n,
        "window": header.get("Window"),
        "normalization_status": "declared" if contract else "incomplete-contract",
        "source_headers": dict(header),
        "power_domain": "linearized-raw-dn2-per-bin",
        "sample_spacing_sensor_pixels": 2,
        "frequencies_channel_plane": frequencies.tolist(),
        "frequencies_sensor": sensor_frequencies(frequencies).tolist(),
        "one_sided_storage": "interior-bins-undoubled" if undoubled else "undeclared",
        "diff_power_factor": diff_factor,
        "mean_removal": "per-transformed-line; line-to-line power is not measured",
        "measurements": [{"iso": iso, "phases": phases} for iso, phases in sorted(records.items())],
    }
    _validate_complete_axis(result)
    return result


def _validate_complete_axis(value):
    if "measurements" not in value:
        return
    import numpy as np
    from .spectral_variance import one_sided_power_mass

    frequencies = np.asarray(value.get("frequencies_channel_plane"), dtype=np.float64)
    if (frequencies.ndim != 1 or not 2 <= frequencies.size <= 524_289 or
            not np.isfinite(frequencies).all() or np.any(frequencies < 0) or
            np.any(frequencies > .5) or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("invalid complete spectrum frequencies")
    n = value.get("transform_length")
    if n is not None and (type(n) is not int or not 2 <= n <= 1_048_576 or frequencies.size != n // 2 + 1):
        raise ValueError("invalid complete spectrum transform length")
    if n is not None and not np.allclose(frequencies, np.arange(frequencies.size) / n, rtol=6e-6, atol=1e-9):
        raise ValueError("complete spectrum frequencies disagree with transform length")
    if value.get("sample_spacing_sensor_pixels") != 2:
        raise ValueError("collector spectrum must retain its two-sensor-pixel phase spacing")
    sensor = np.asarray(value.get("frequencies_sensor"), dtype=np.float64)
    if sensor.shape != frequencies.shape or not np.allclose(sensor, frequencies / 2., rtol=1e-12, atol=1e-15):
        raise ValueError("complete spectrum sensor frequencies use the wrong sampling units")
    headers = value.get("source_headers")
    if not isinstance(headers, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in headers.items()):
        raise ValueError("complete spectrum source headers must be text")
    if n is not None:
        try:
            declared_length = int(headers.get("TransformLength", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("complete spectrum source transform length is unavailable") from exc
        if declared_length != n:
            raise ValueError("complete spectrum transform length disagrees with source header")
    contract, undoubled, diff_factor = _normalization_contract(headers, n)
    if (value.get("normalization_status") != ("declared" if contract else "incomplete-contract") or
            value.get("window") != headers.get("Window") or
            value.get("one_sided_storage") != ("interior-bins-undoubled" if undoubled else "undeclared") or
            value.get("diff_power_factor") != diff_factor or
            value.get("power_domain") != "linearized-raw-dn2-per-bin"):
        raise ValueError("complete spectrum interpretation disagrees with its source headers")
    records = value["measurements"]
    if not isinstance(records, list):
        raise ValueError("complete spectrum measurements must be a list")
    last = 0
    count = 0
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("complete spectrum measurement must be an object")
        iso = record.get("iso")
        if type(iso) not in (int, float) or not math.isfinite(iso) or iso <= last:
            raise ValueError("complete spectrum ISOs must be positive and ordered")
        last = iso
        if not isinstance(record.get("phases"), dict):
            raise ValueError("complete spectrum phases must be an object")
        for phase, sample in record["phases"].items():
            if phase not in PHASES or not isinstance(sample, dict):
                raise ValueError("invalid complete spectrum phase")
            for key, number in sample.items():
                if "variance_dn2" in key or key.startswith("parseval_ratio_"):
                    if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                        raise ValueError("complete spectrum variance/integration diagnostic is invalid")
            comparisons = []
            for source in ("single", "diff"):
                if source + "_power" in sample:
                    power = np.asarray(sample[source + "_power"], dtype=np.float64)
                    count += power.size
                    if (power.shape != frequencies.shape or not np.isfinite(power).all() or
                            np.any(power < 0) or count > 4_000_000):
                        raise ValueError("invalid or excessive complete spectrum power")
                    if contract:
                        integral = float(one_sided_power_mass(power, n, difference=source == "diff").sum())
                        key = "temporal_variance_dn2" if source == "diff" else "single_variance_dn2"
                        if key not in sample or not math.isclose(sample[key], integral, rel_tol=1e-10, abs_tol=1e-14):
                            raise ValueError("complete spectrum variance disagrees with retained power")
                        reference = sample.get(f"reference_{source}_within_line_variance_dn2")
                        ratio_key = f"parseval_ratio_{source}"
                        if reference is not None and reference > 0:
                            ratio = integral * (2 if source == "diff" else 1) / reference
                            if ratio_key not in sample or not math.isclose(sample[ratio_key], ratio, rel_tol=1e-10, abs_tol=1e-14):
                                raise ValueError("complete spectrum Parseval diagnostic disagrees with retained evidence")
                            comparisons.append(abs(ratio - 1) <= 2e-5)
            expected = _integration_status(contract, value.get("window"), comparisons)
            if sample.get("integration_status") != expected:
                raise ValueError("complete spectrum qualification disagrees with retained evidence")
