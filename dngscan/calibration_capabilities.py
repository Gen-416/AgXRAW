# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured products keep their own dependencies, permissions and use stages.

Fits are retained for inspection. This module grants no permission from a fit's
mere existence, and never fills a missing CSV condition from an unrelated CSV.
"""
from __future__ import annotations

import math

PRODUCTS = ("ptc_gain", "relative_gain", "absolute_gain", "read_noise_dn",
            "read_noise_e", "stored_dark_variance", "spectrum_h", "spectrum_v")
ALIASES = {"read_noise": "read_noise_dn"}
FIELDS = {"linearisation_curve": "linearisation", "adc_step": "adc_step",
          "cfa_pattern": "cfa", "decoder_identity": "decoder_identity",
          "dn_domain": "dn_domain", "variance_domain": "variance_domain"}


def qualification(entry, capability):
    name = ALIASES.get(capability, capability)
    supplied = (entry.get("composed_capabilities") or {}).get(name)
    if supplied is not None:
        return supplied["qualification"]
    acquisition = entry.get("measurement_acquisition")
    if acquisition is None:
        return None  # Explicit legacy contract; not newly measured permission.
    return acquisition.get("capability_qualification", {}).get(name,
        {"status": "unmeasured", "source_ids": [], "reasons": ["measurement-not-supplied"]})


def usable(entry, capability):
    item = qualification(entry, capability)
    if item is None:
        return True
    name = ALIASES.get(capability, capability)
    match = entry.get("capability_matches", {}).get(name, {})
    return item.get("status") == "qualified" and match.get("status") not in ("unverified", "mismatch")


def capability_prior(entry, capability):
    """A dependency-specific view for the existing unit/readout matcher.

    Only source declarations in this product enter its runtime domain. The full
    acquisition object remains on the parent for diagnosis and other products.
    """
    name = ALIASES.get(capability, capability)
    supplied = (entry.get("composed_capabilities") or {}).get(name)
    if supplied is not None:
        result = dict(entry)
        for key in ("acquisition_contract", "readout_contract", "readout_declaration_issues", "shutter", "measurement_acquisition"):
            result[key] = supplied["supplier"].get(key)
        result.pop("composed_capabilities", None)
        return result
    item = qualification(entry, name)
    if item is None:
        return dict(entry)
    from .collect_manifest import acquisition_qualification
    from .measurement_domain import DN, DN2
    result = dict(entry)
    acquisition = entry["measurement_acquisition"]
    selected = {key: acquisition["measurements"][key] for key in item["source_ids"]}
    subset = acquisition_qualification(selected) if selected else None
    common = subset["common_contract"] if subset else {}
    contract = dict(entry.get("acquisition_contract") or {})
    for field, source in FIELDS.items():
        contract.pop(field, None)
        if source in common:
            value = common[source]
            contract[field] = value.lower() if isinstance(value, str) else value
    contract.setdefault("dn_domain", DN)
    contract.setdefault("variance_domain", DN2)
    contract.setdefault("decoder_identity", "LibRaw")
    # Conditions supplied explicitly by the user/profile (for example an
    # exposure range or exact lens identity) remain restrictions. Only CSV
    # fields are rebuilt from this product's own dependencies.
    readout = dict(entry.get("readout_contract") or {"version": 1})
    for field in ("libraw_raw_geometry", "body_serial", "firmware", "readout_id", "capture_kind", "storage_lossless"):
        readout.pop(field, None)
    for source, target in (("raw_geometry", "libraw_raw_geometry"), ("body", "body_serial"),
                           ("firmware", "firmware"), ("readout", "readout_id"), ("capture_kind", "capture_kind")):
        if source in common:
            readout[target] = common[source]
    if common.get("compression") in ("LOSSLESS", "LOSSLESS COMPRESSION", "UNCOMPRESSED", "NONE", "无损压缩", "无压缩"):
        readout["storage_lossless"] = True
    result.update(acquisition_contract=contract, readout_contract=readout,
                  shutter=common.get("shutter"), measurement_acquisition=subset,
                  readout_declaration_issues=list(item.get("reasons", [])))
    # Retain failures not derived from the profile-wide measurement union.
    result["readout_declaration_issues"].extend(entry.get("capability_contract_issues", []))
    result["readout_declaration_issues"].extend(current_contract_conflicts(entry))
    result["readout_declaration_issues"] = sorted(set(result["readout_declaration_issues"]))
    if not any(record["kind"] == "dark" for record in selected.values()):
        # A PTC's own pedestal is sufficient for its scalar gain fit. Phase
        # identities inherited from an unrelated dark cannot authorize it or
        # disqualify it; they are checked for the products consuming that dark.
        result["phase_calibration"] = {}
        result["noise_spectrum"] = None
    elif name.startswith("spectrum_") and result.get("noise_spectrum"):
        result["noise_spectrum"] = dict(result["noise_spectrum"], axes={
            axis: record for axis, record in result["noise_spectrum"]["axes"].items() if axis == name[-1]})
    else:
        result["noise_spectrum"] = None
    return result


def dependency_views(entry, capability):
    """All already qualified suppliers must still match the current file."""
    name = ALIASES.get(capability, capability)
    supplied = (entry.get("composed_capabilities") or {}).get(name)
    return ([capability_prior(entry, name), *supplied.get("additional_dependencies", [])]
            if supplied else [capability_prior(entry, name)])


def primary_product(entry):
    for name, found in (("absolute_gain", bool(entry.get("gain_log2iso_log2epd"))),
                        ("ptc_gain", bool(entry.get("ptc_gain_log2iso_log2epd"))),
                        ("read_noise_dn", bool(entry.get("read_noise_dn_log2iso"))),
                        ("stored_dark_variance", bool(entry.get("stored_dark_variance_dn2_log2iso"))),
                        ("spectrum_h", bool((entry.get("noise_spectrum") or {}).get("axes", {}).get("h"))),
                        ("spectrum_v", bool((entry.get("noise_spectrum") or {}).get("axes", {}).get("v")))):
        if found and usable(entry, name):
            return name
    return "absolute_gain" if entry.get("gain_log2iso_log2epd") else "read_noise_dn"


def bind_primary(entry):
    if entry.get("measurement_acquisition") is None:
        return entry
    view = capability_prior(entry, primary_product(entry))
    for field in ("acquisition_contract", "readout_contract", "readout_declaration_issues", "shutter"):
        entry[field] = view[field]
    entry["capability_bound_contract"] = dict(entry["acquisition_contract"])
    return entry


def current_contract_conflicts(entry):
    acquisition = entry.get("measurement_acquisition")
    if acquisition is None:
        return []
    contract = entry.get("acquisition_contract") or {}
    bound = entry.get("capability_bound_contract")
    # The union contract may originally have come from an optional CSV. A
    # normalized capability view is permitted to replace it. Later caller
    # edits must still be checked, even with an old runtime match attached.
    changed = ({field: value for field, value in contract.items() if bound.get(field) != value}
               if bound is not None else contract)
    from .measurement_domain import declaration_issues
    return contract_conflicts(changed, acquisition) + (declaration_issues(changed) if bound is not None else [])


def contract_conflicts(contract, acquisition):
    """A manually edited scalar contract cannot contradict its own sources."""
    from .measurement_domain import canonical_unit, decoder_family
    result = []
    associated = acquisition["association_contract"]
    for field, source in FIELDS.items():
        if contract.get(field) is None or source not in associated:
            continue
        left, right = contract[field], associated[source]
        if field in ("dn_domain", "variance_domain"):
            left = canonical_unit(left, variance=field == "variance_domain")
            right = canonical_unit(right, variance=field == "variance_domain")
        elif field == "decoder_identity":
            left, right = decoder_family(left), decoder_family(right)
        elif field == "adc_step":
            try:
                left, right = float(left), float(right)
            except (ValueError, TypeError):
                pass
        else:
            left, right = str(left).casefold(), str(right).casefold()
        if left != right:
            result.append("measurement-contract-" + field.replace("_", "-") + "-conflict")
    return result


def stages(entry, *, iso=None, applied=(), consumers=None):
    """JSON-stable states; fit existence, current use and pixels differ.

    Numeric validity uses the production accessors, including failed samples
    and disconnected PTC anchors. A retained curve alone grants no current-ISO
    permission. Consumers describe the actual profile's source contribution,
    rather than claiming that each individual fitted product was consumed.
    """
    from .calibration import curve_value, stored_dark_variance
    from .priors import gain_e_per_dn, read_noise_e

    def valid_at_iso(name):
        if not iso or iso <= 0 or not math.isfinite(float(iso)) or not usable(entry, name):
            return False
        if name == "ptc_gain":
            # A PTC supplies an absolute anchor; only a qualified relative
            # ladder can authorize values between independent anchors.
            return any(abs(float(x) - math.log2(iso)) < 1e-7 and math.isfinite(float(y))
                       for x, y in entry.get("ptc_gain_log2iso_log2epd") or [])
        if name == "absolute_gain":
            return gain_e_per_dn(entry, iso) is not None
        if name in ("read_noise_dn", "read_noise_e"):
            if entry.get("read_noise_composition_iso") not in (None, iso):
                return False
            return (curve_value(entry, "read_noise_dn_log2iso", iso) is not None
                    if name == "read_noise_dn" else read_noise_e(entry, iso) is not None)
        if name == "stored_dark_variance":
            return stored_dark_variance(entry, iso)[0] is not None
        if name == "relative_gain":
            return any(iso in component.get("isos", []) for component in
                       (entry.get("gain_ladder_diagnostics") or {}).get("components", []))
        # Use the same CFA mapping, jump barriers and rounded legacy-ISO
        # handling as production. Independent suppliers are separate products
        # and must not turn this profile's missing axis into a valid one.
        from .noise_model import _prior_spectrum
        _, ratios = _prior_spectrum({key: value for key, value in entry.items()
                                     if key != "independent_constraint_priors"}, iso)
        axis = name[-1]
        return any(key == axis or key.startswith(axis + ":") for key in ratios)

    applied = set(applied)
    consumers = consumers or {}
    curve_keys = {"ptc_gain": "ptc_gain_log2iso_log2epd", "absolute_gain": "gain_log2iso_log2epd",
                  "read_noise_dn": "read_noise_dn_log2iso", "read_noise_e": "read_noise_log2iso_log2e",
                  "stored_dark_variance": "stored_dark_variance_dn2_log2iso"}
    result = {}
    for name in PRODUCTS:
        q = qualification(entry, name)
        if q is None:
            continue
        match = entry.get("capability_matches", {}).get(name)
        if name in curve_keys:
            curve = entry.get(curve_keys[name]) or []
            fit_exists = bool(curve)
        elif name == "relative_gain":
            fit_exists = bool((entry.get("gain_ladder_diagnostics") or {}).get("components"))
        else:
            axis = name[-1]
            curves = ((entry.get("noise_spectrum") or {}).get("axes", {}).get(axis) or {}).get("ratios_log2iso", {})
            fit_exists = bool(curves or entry.get("noise_whiteness_" + axis + "_log2iso"))
        try:
            valid = valid_at_iso(name) if iso is not None else None
        except (ValueError, TypeError, KeyError, OverflowError):
            valid = False
        matched = None if match is None else match.get("status") in ("matched", "not-declared")
        applicable = None if iso is None or match is None else bool(matched and valid and usable(entry, name))
        result[name] = {"found": q.get("status") != "unmeasured", "qualified": q.get("status") == "qualified",
                        "matched": matched, "fit_exists": fit_exists, "valid": valid, "applicable": applicable,
                        "applied": name in applied, "source_ids": q.get("source_ids", []),
                        "consumers": list(consumers.get(name, [])) if applicable else [],
                        "consumer_semantics": "profile-level source contribution; individual product use not resolved",
                        "reasons": list(q.get("reasons", [])) + ([match["reason"]] if match and not matched else []),
                        "application_semantics": "sensor diagnostic/model permission; not evidence of pixel correction"}
    return result
