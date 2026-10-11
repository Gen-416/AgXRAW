# SPDX-License-Identifier: GPL-3.0-or-later
"""Units and unpack-domain qualification for Collect sensor measurements.

Authority: JiangtherapeeTesterView 57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c,
native/src/binding.cc, src/analysis/{darkPair,ptcPair}.mjs and the CSV writers.
The four supported formats measure LibRaw unpacked uint16 codes, before black
subtraction, WB or demosaic. StdDiff is A-B; spectral powers are code squared.
Format defaults describe units, never supply another CSV's capture conditions.
No conversion from electrons, normalized samples or scene RGB is implemented.
"""
from __future__ import annotations

import math
import re

from ._deps import np

SCHEMA = "dngscan-sample-domain-1"
DN = "libraw-unpacked-dn"
DN2 = "libraw-unpacked-dn2"
FORMAT_AUTHORITY = "JiangtherapeeTesterView@57567edfa0ec16ee0b00c6b4a1a325c8da40bf1c"


def canonical_unit(value, *, variance=False):
    text = str(value).strip().casefold().replace("²", "2").replace("^", "")
    text = re.sub(r"[ _]+", "-", text)
    aliases = ({"dn2", "raw-dn2", "linearized-raw-dn2", "linearised-raw-dn2", DN2}
               if variance else {"dn", "raw-dn", "linearized-raw-dn", "linearised-raw-dn", DN})
    return (DN2 if variance else DN) if text in aliases else None


def decoder_family(value):
    # The official writer reports LibRaw::version(). Versions are provenance;
    # equality alone neither proves nor disproves identical unpack semantics.
    text = str(value).strip()
    return "libraw" if re.fullmatch(r"libraw(?:\s+\d+\.\d+(?:\.\d+)?(?:[- .+][\w.+-]+)?)?", text, re.I) else None


def declaration_issues(contract, *, check_adc=None):
    """Reject explicit unsupported semantics, including legacy JSON contracts."""
    issues = []
    for field, variance in (("dn_domain", False), ("variance_domain", True)):
        if contract.get(field) is not None and canonical_unit(contract[field], variance=variance) is None:
            issues.append("measurement-" + field.replace("_", "-") + "-unsupported")
    if contract.get("decoder_identity") is not None and decoder_family(contract["decoder_identity"]) is None:
        issues.append("measurement-decoder-domain-unverified")
    if check_adc is None:
        check_adc = any(name in contract for name in ("dn_domain", "variance_domain", "decoder_identity"))
    if check_adc and contract.get("adc_step") is not None:
        try:
            step = float(contract["adc_step"])
            if not math.isfinite(step) or step <= 0:
                issues.append("measurement-adc-step-invalid")
            elif step > 65535:
                issues.append("measurement-adc-step-unrepresentable")
            elif step != int(step):
                issues.append("measurement-adc-step-noninteger")
        except (ValueError, TypeError, OverflowError):
            issues.append("measurement-adc-step-invalid")
    return issues


def format_domains(measurements):
    domains, issues = {}, []
    for identity, record in measurements.items():
        declared = record["declared"]
        issues.extend(declaration_issues(declared, check_adc=True))
        domains[identity] = {
            "signal_unit": canonical_unit(declared.get("dn_domain", DN)),
            "variance_unit": canonical_unit(declared.get("variance_domain", DN2), variance=True),
            "decoder_family": decoder_family(declared["decoder_identity"]) if "decoder_identity" in declared else "libraw",
            "authority": FORMAT_AUTHORITY,
            "unit_source": {name: "explicit" if name in declared else "supported-format"
                            for name in ("dn_domain", "variance_domain", "decoder_identity")},
            "mean_zero": "unsubtracted-unpacked-code",
            "difference": "A-B; no sqrt(2) division or Sheppard correction",
        }
    return domains, sorted(set(issues))


def capture_domain(raw, image, pattern, color_desc, whites, blacks, provider_version):
    """Small immutable-evidence descriptor; no scene pixels or inferred noise.

    rawpy's pinned tone_curve property exposes rawdata.color.curve, the same
    unpack LUT inspected by Collect. A sampled code GCD is only a necessary
    lattice check, never an ADC measurement or proof of decoder equivalence.
    """
    linearisation = "unavailable"
    try:
        curve = np.asarray(raw.tone_curve)
        nonzero = np.flatnonzero(curve)
        end = int(nonzero[-1]) + 1 if len(nonzero) else 0
        linearisation = "identity" if not end or np.array_equal(curve[:end], np.arange(end)) else "companded"
    except (AttributeError, ValueError, TypeError, RuntimeError):
        pass
    values = np.asarray(image)
    def grid_step(codes, limit):
        if codes.dtype.kind not in "ui" or not codes.size:
            return None
        # A bounded deterministic sample, including all CFA positions.
        stride = max(1, codes.size // limit)
        if stride % 2 == 0:
            stride += 1
        sampled = np.unique(codes.flat[::stride].astype(np.int64))
        if len(sampled) > 1:
            return int(np.gcd.reduce(np.diff(sampled)))
        return None
    step = grid_step(values, 65536)
    phase_steps = {}
    if values.ndim == 2 and pattern and all(len(row) == len(pattern[0]) for row in pattern):
        ph, pw = len(pattern), len(pattern[0])
        if ph * pw <= 256:
            for y in range(ph):
                for x in range(pw):
                    phase_steps[f"C{y}{x}"] = grid_step(values[y::ph, x::pw], max(256, 65536 // (ph*pw)))
    # WhiteLevel may be scalar while the file black model is per channel.
    # Match raw_units.normalized_raw_span's clamped indexing for both lists.
    spans = [float(whites[min(i, len(whites)-1)]) - float(blacks[min(i, len(blacks)-1)])
             for i in range(max(len(whites), len(blacks)))] if whites and blacks else []
    return {"schema": SCHEMA, "provider": "libraw", "provider_version": provider_version,
            "stage": "unpack-before-project-corrections", "signal_unit": DN, "variance_unit": DN2,
            "sample_kind": "cfa" if values.ndim == 2 else "linear-camera-rgb", "dtype": str(values.dtype),
            "channel_description": color_desc, "phase_pattern": pattern,
            "linearisation": linearisation, "coding_spans_dn": spans,
            "zero_reference": "file-black-model; raw codes remain unsubtracted",
            "sampled_code_difference_gcd": step,
            "sampled_phase_code_difference_gcd": phase_steps,
            "adc_evidence": "necessary-code-lattice-check-only; not an ADC measurement"}


def match(entry, capture):
    """Bind measured units to actual evidence; scene-decoder choice is irrelevant."""
    contract = entry.get("acquisition_contract") or {}
    active = (entry.get("measurement_acquisition") is not None or
              any(name in contract for name in ("dn_domain", "variance_domain", "decoder_identity")))
    result = {"status": "not-declared", "reason": "measurement-domain-not-declared"}
    if not active:
        return result
    def fail(status, reason):
        return {**result, "status": status, "reason": reason}
    issues = declaration_issues(contract)
    if issues:
        return fail("unverified", issues[0])
    actual = (capture or {}).get("sample_domain")
    if not isinstance(actual, dict) or actual.get("schema") != SCHEMA:
        return fail("unverified", "file-sample-domain-unavailable")
    result.update(signal_unit=actual.get("signal_unit"), variance_unit=actual.get("variance_unit"),
                  measurement_decoder=contract.get("decoder_identity"),
                  evidence_decoder=actual.get("provider_version"),
                  compatibility_basis="LibRaw unpacked integer DN; identity LUT; declared readout and coding-span transport")
    if (actual.get("provider") != "libraw" or actual.get("stage") != "unpack-before-project-corrections"
            or actual.get("signal_unit") != DN or actual.get("variance_unit") != DN2):
        return fail("unverified", "file-unpack-domain-unsupported")
    if actual.get("sample_kind") != "cfa" or actual.get("dtype") != "uint16":
        return fail("unverified", "file-independent-integer-cfa-domain-unavailable")
    if actual.get("linearisation") != "identity":
        return fail("unverified", "file-linearisation-domain-" + str(actual.get("linearisation") or "unavailable"))
    if str(contract.get("linearisation_curve") or "").casefold() != "identity":
        return fail("unverified", "measurement-linearisation-domain-unverified")
    description = contract.get("cfa_pattern")
    if description is not None and str(description).casefold() != str(actual.get("channel_description")).casefold():
        return fail("mismatch", "file-channel-description-mismatch")
    pattern = actual.get("phase_pattern") or []
    phase_products = [entry.get("phase_calibration") or {},
                      ((entry.get("noise_spectrum") or {}).get("mapping") or {}).get("phases", {})]
    for name, phase in [item for products in phase_products for item in products.items()]:
        cid = phase.get("color_index")
        if cid is None:
            continue
        # Legacy rows identified a colour index without a spatial phase.
        # Keep that scalar scope; never manufacture a C00/C01 mapping.
        if name.startswith("index:") and phase.get("channel") is None:
            if cid not in [value for row in pattern for value in row]:
                return fail("mismatch", "file-measurement-channel-index-mismatch")
            continue
        if re.fullmatch(r"C[01][01]", name) is None or len(pattern) != 2 or any(len(row) != 2 for row in pattern):
            return fail("unverified", "file-measurement-cfa-phase-unavailable")
        if pattern[int(name[1])][int(name[2])] != cid:
            return fail("mismatch", "file-measurement-cfa-phase-mismatch")
    from .priors import reference_dn_range
    reference = reference_dn_range(entry)
    spans = actual.get("coding_spans_dn") or []
    try:
        if not reference or not spans or not math.isfinite(float(reference)) or reference <= 0:
            return fail("unverified", "measurement-dn-reference-unavailable")
        transports = []
        for span in spans:
            if not math.isfinite(span) or span <= 0:
                return fail("unverified", "file-coding-span-unavailable")
            ratio = span / reference
            power = round(math.log2(ratio))
            if abs(math.log2(ratio) - power) > math.log2(1.05):
                return fail("mismatch", "file-measurement-dn-scale-mismatch")
            transports.append(power)
        step = float(contract["adc_step"])
        if not math.isfinite(step) or step <= 0:
            return fail("unverified", "measurement-adc-step-unresolved")
        # Phase offsets are not quantisation steps. Never use a combined CFA
        # GCD to reject separate phase grids with different constant offsets.
        phase_steps = actual.get("sampled_phase_code_difference_gcd")
        if not isinstance(phase_steps, dict) or not phase_steps:
            return fail("unverified", "file-measurement-adc-grid-unavailable")
        observed_steps = phase_steps.values()
        for power in transports:
            expected = step * 2.**power
            if expected < 1 or expected > 65535 or expected != int(expected):
                return fail("unverified", "file-measurement-adc-grid-unrepresentable")
            if any(observed is not None and observed % int(expected) for observed in observed_steps):
                return fail("mismatch", "file-measurement-adc-grid-mismatch")
    except (KeyError, TypeError, ValueError, OverflowError):
        return fail("unverified", "measurement-dn-scale-unverified")
    return {**result, "status": "matched", "reason": "qualified-unpack-domain-and-scale",
            "coding_scale_log2": transports, "adc_evidence": actual.get("adc_evidence")}
