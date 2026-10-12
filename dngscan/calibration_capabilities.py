# SPDX-License-Identifier: GPL-3.0-or-later
"""Measured products keep their own dependencies, permissions and use stages.

Fits are retained for inspection. This module grants no permission from a fit's
mere existence, and never fills a missing CSV condition from an unrelated CSV.
"""
from __future__ import annotations

import copy
import math

PERMISSION_SEMANTICS_VERSION = 3
READOUT_SCOPE_VERSION = 1
PRODUCTS = ("ptc_gain", "relative_gain", "absolute_gain", "read_noise_dn",
            "read_noise_e", "stored_dark_variance", "spectrum_h", "spectrum_v")
ALIASES = {"read_noise": "read_noise_dn"}
FIELDS = {"linearisation_curve": "linearisation", "adc_step": "adc_step",
          "cfa_pattern": "cfa", "decoder_identity": "decoder_identity",
          "dn_domain": "dn_domain", "variance_domain": "variance_domain"}
_READOUT_FIELDS = {"raw_geometry": "libraw_raw_geometry", "body": "body_serial",
                   "firmware": "firmware", "readout": "readout_id", "capture_kind": "capture_kind"}
_LOSSLESS = {"LOSSLESS", "LOSSLESS COMPRESSION", "UNCOMPRESSED", "NONE", "无损压缩", "无压缩"}


def _same_constraint(left, right):
    if isinstance(left, str) and isinstance(right, str):
        return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()
    return left == right


def _automatic_readout(acquisition, *, common=False):
    """CSV scope and its declaring files; association is never permission."""
    declared = acquisition.get("common_contract" if common else "association_contract", {})
    result, sources = {"version": 1}, {}
    for source, target in _READOUT_FIELDS.items():
        if source in declared:
            result[target] = copy.deepcopy(declared[source])
            sources[target] = sorted(identity for identity, record in acquisition.get("measurements", {}).items()
                                     if source in record["declared"])
    if declared.get("compression") in _LOSSLESS:
        result["storage_lossless"] = True
        sources["storage_lossless"] = sorted(identity for identity, record in acquisition.get("measurements", {}).items()
                                           if "compression" in record["declared"])
    return result, sources


def new_readout_scope(acquisition):
    """A newly built CSV package has no implicit user-wide restrictions."""
    automatic, sources = _automatic_readout(acquisition)
    return {"version": READOUT_SCOPE_VERSION, "automatic": automatic,
            "automatic_sources": sources, "explicit": {"version": 1},
            "explicit_shutter": None, "compatibility": "layered", "legacy_ambiguous_fields": []}


def public_readout_scope(scope):
    """Report scope origins while retaining exact serials only in the profile."""
    if scope is None:
        return None
    from .capture_identity import public_capture
    return {**copy.deepcopy(scope), "automatic": public_capture(scope.get("automatic")),
            "explicit": public_capture(scope.get("explicit"))}


def readout_scope_fields(item, acquisition):
    """Retain JSON edits separately from the recomputable CSV union.

    Unlayered historical JSON cannot distinguish an equal user restriction
    from a CSV summary. Such fields remain ambiguous; only a capability that
    already has the same condition in its own dependencies can use them.
    """
    from .readout import measurement_fields

    incoming = measurement_fields(item)["readout_contract"]
    automatic, sources = _automatic_readout(acquisition)
    supplied = item.get("readout_scope")
    if supplied is None:
        result = new_readout_scope(acquisition)
        result["compatibility"] = "legacy-unlayered"
        result["legacy_ambiguous_fields"] = sorted(key for key, value in incoming.items()
            if key != "version" and key in automatic and _same_constraint(value, automatic[key]))
    else:
        allowed = {"version", "automatic", "automatic_sources", "explicit", "explicit_shutter", "compatibility", "legacy_ambiguous_fields"}
        if (not isinstance(supplied, dict) or set(supplied) - allowed
                or type(supplied.get("version")) is not int or supplied["version"] != READOUT_SCOPE_VERSION):
            raise ValueError("readout_scope requires version 1 and supported fields")
        snapshot = measurement_fields({"readout_contract": supplied.get("automatic", automatic)})["readout_contract"]
        if set(snapshot) != set(automatic) or any(not _same_constraint(value, automatic[key]) for key, value in snapshot.items()):
            raise ValueError("readout_scope automatic conditions disagree with measurement CSV declarations")
        if supplied.get("automatic_sources", sources) != sources:
            raise ValueError("readout_scope automatic sources disagree with measurement CSV declarations")
        explicit = measurement_fields({"readout_contract": supplied.get("explicit")})["readout_contract"]
        compatibility = supplied.get("compatibility", "layered")
        ambiguous = supplied.get("legacy_ambiguous_fields", [])
        if (compatibility not in ("layered", "legacy-unlayered") or not isinstance(ambiguous, list)
                or any(not isinstance(key, str) or key not in automatic or key == "version" for key in ambiguous)
                or compatibility == "layered" and ambiguous):
            raise ValueError("invalid readout_scope compatibility state")
        result = {"version": READOUT_SCOPE_VERSION, "automatic": automatic, "automatic_sources": sources,
                  "explicit": explicit, "explicit_shutter": supplied.get("explicit_shutter"), "compatibility": compatibility,
                  "legacy_ambiguous_fields": sorted(set(ambiguous))}
    from .calibration import _shutter
    mode = (item.get("user_applicability") or {}).get("shutter_override", item.get("shutter"))
    mode = _shutter(mode)
    if mode is not None and mode != acquisition.get("association_contract", {}).get("shutter"):
        result["explicit_shutter"] = mode
    if result["explicit_shutter"] not in (None, "mechanical", "electronic", "efcs", "any"):
        raise ValueError("readout_scope explicit_shutter is unsupported")
    # Existing callers may continue editing readout_contract. A changed or
    # newly added value is an explicit restriction, never a replacement for
    # the measurement's own condition.
    for key, value in incoming.items():
        if key != "version" and (key not in automatic or not _same_constraint(value, automatic[key])):
            result["explicit"][key] = copy.deepcopy(value)
    result["legacy_ambiguous_fields"] = [key for key in result["legacy_ambiguous_fields"]
                                         if key not in result["explicit"]]
    return result


def _current_scope(entry):
    scope = copy.deepcopy(entry.get("readout_scope"))
    if scope is None:
        scope = readout_scope_fields(entry, entry["measurement_acquisition"])
    from .readout import measurement_fields
    scope["explicit"] = measurement_fields({"readout_contract": scope["explicit"]})["readout_contract"]
    bound = entry.get("capability_bound_readout_contract")
    if bound is not None:
        current = measurement_fields({"readout_contract": entry.get("readout_contract")})["readout_contract"]
        for key, value in current.items():
            if key != "version" and (key not in bound or not _same_constraint(value, bound[key])):
                scope["explicit"][key] = copy.deepcopy(value)
    if "capability_bound_shutter" in entry and entry.get("shutter") is not None:
        if entry["shutter"] != entry["capability_bound_shutter"]:
            scope["explicit_shutter"] = entry["shutter"]
    scope["legacy_ambiguous_fields"] = [key for key in scope["legacy_ambiguous_fields"]
                                        if key not in scope["explicit"]]
    return scope


def _scoped_readout(entry, common):
    scope = _current_scope(entry)
    automatic, _ = _automatic_readout({"common_contract": common}, common=True)
    result, issues = dict(automatic), []
    for key, value in scope["explicit"].items():
        if key != "version":
            if key in automatic and not _same_constraint(value, automatic[key]):
                issues.append("measurement-readout-" + key.replace("_", "-") + "-conflict")
            result[key] = copy.deepcopy(value)
    for key in scope["legacy_ambiguous_fields"]:
        if key not in automatic or not _same_constraint(automatic[key], scope["automatic"][key]):
            issues.append("readout-scope-reimport-or-explicit-scope-required:" + key.replace("_", "-"))
    mode = scope.get("explicit_shutter")
    if mode not in (None, "any") and common.get("shutter") not in (None, "any", mode):
        issues.append("measurement-readout-shutter-conflict")
    return result, scope, sorted(set(issues))


def _scope_issues(entry, product):
    acquisition = entry.get("measurement_acquisition") or {}
    measurements = acquisition.get("measurements")
    if not measurements or not product.get("source_ids"):
        return []
    selected = [measurements[identity]["declared"] for identity in product["source_ids"]]
    common = {key: value for key, value in selected[0].items()
              if all(key in record and _same_constraint(record[key], value) for record in selected[1:])}
    return _scoped_readout(entry, common)[2]


def qualification(entry, capability):
    name = ALIASES.get(capability, capability)
    supplied = (entry.get("composed_capabilities") or {}).get(name)
    if supplied is not None:
        return supplied["qualification"]
    acquisition = entry.get("measurement_acquisition")
    if acquisition is None:
        return None  # Explicit legacy contract; not newly measured permission.
    item = acquisition.get("capability_qualification", {}).get(name,
        {"status": "unmeasured", "source_ids": [], "reasons": ["measurement-not-supplied"]})
    issues = _scope_issues(entry, item)
    return ({**item, "status": "unverified", "reasons": sorted(set(item.get("reasons", []) + issues))}
            if issues else item)


def usable(entry, capability):
    name = ALIASES.get(capability, capability)
    match = entry.get("capability_matches", {}).get(name, {})
    if match.get("status") in ("unverified", "mismatch"):
        return False
    item = qualification(entry, capability)
    if item is None:
        return True
    return item.get("status") == "qualified"


def capability_prior(entry, capability):
    """A dependency-specific view for the existing unit/readout matcher.

    Only source declarations in this product enter its runtime domain. The full
    acquisition object remains on the parent for diagnosis and other products.
    """
    name = ALIASES.get(capability, capability)
    supplied = (entry.get("composed_capabilities") or {}).get(name)
    if supplied is not None:
        result = dict(entry)
        for key in ("acquisition_contract", "readout_contract", "readout_declaration_issues", "shutter", "measurement_acquisition",
                    "readout_scope", "capability_bound_readout_contract", "capability_bound_shutter"):
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
    readout, scope, scope_issues = _scoped_readout(entry, common)
    # User scope may narrow a measured wildcard, never supply a missing CSV
    # condition or widen a specific measurement. Conflicts remain unqualified.
    shutter = common.get("shutter")
    if shutter == "any" and scope.get("explicit_shutter") not in (None, "any"):
        shutter = scope["explicit_shutter"]
    result.update(acquisition_contract=contract, readout_contract=readout,
                  shutter=shutter, measurement_acquisition=subset,
                  readout_scope=scope, readout_declaration_issues=list(item.get("reasons", [])) + scope_issues)
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
    entry["readout_scope"] = view["readout_scope"]
    entry["capability_bound_readout_contract"] = copy.deepcopy(entry["readout_contract"])
    entry["capability_bound_shutter"] = entry.get("shutter")
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
