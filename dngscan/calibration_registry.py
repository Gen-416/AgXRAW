# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline, versioned public-data catalogue and conservative capability composition.

Upstream comparison derivations are recorded, not promoted to pixel calibration.
Library installation accepts a maintainer-built JSON release; rendering never
fetches a website. User measurements remain in their separate existing store.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from functools import lru_cache

SCHEMA = "dngscan-public-calibration-library-1"
SOURCE_COMMIT = "504e119bcfb9ff2b58fc9e47c204ddb77eee5879"
PACKAGED = Path(__file__).parent / "data/priors/collect_registry.json"
MAX_RELEASE_BYTES = 128 * 1024 * 1024
# This identifies the shipped numerical asset, not every future independently
# qualified measurement bearing the same registry set ID. Selecting an older
# or partial catalogue cannot rehabilitate its retired dark statistics.
RETIRED_PACKAGED_COLLECT_SETS = frozenset({"s1r2-mech-20260812"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def compile_catalog(raw: dict, *, source_commit: str, raw_sha256: str) -> dict:
    """Compile provenance/dependencies, not reconstructed measurement facts.

    This catalogue compiler does not refit CSVs or execute borrowed-gain,
    nominal-ISO or manual PTC recipes. Only separately qualified release
    profile records can add numerical models to the runtime library.
    """
    if not isinstance(raw, dict):
        raise ValueError("registry must be an object")
    if not isinstance(source_commit, str) or not isinstance(raw_sha256, str):
        raise ValueError("registry requires fixed source commit and SHA256")
    if not re.fullmatch(r"[a-f0-9]{40}", source_commit) or not re.fullmatch(r"[a-f0-9]{64}", raw_sha256):
        raise ValueError("registry requires fixed source commit and SHA256")
    sets = raw.get("sets")
    if not isinstance(sets, list) or len(sets) > 1024:
        raise ValueError("registry sets must be a bounded list")
    entries = {}
    for value in sets:
        if not isinstance(value, dict):
            raise ValueError("registry set records must be objects")
        item = copy.deepcopy(value)
        identity = item.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-z0-9-]{1,120}", identity) or identity in entries:
            raise ValueError("registry has invalid or duplicate set ID")
        camera = item.get("camera", {})
        if not isinstance(camera, dict):
            raise ValueError("registry camera declaration must be an object")
        for field in ("brand", "model"):
            if field in camera and (not isinstance(camera[field], str) or not camera[field].strip()):
                raise ValueError("registry camera identity must be nonempty text")
        for field in ("calibrationOnly", "nominalGain"):
            if field in item and not isinstance(item[field], bool):
                raise ValueError(f"registry {field} must be a boolean")
        if "ptcFit" in item and not isinstance(item["ptcFit"], dict):
            raise ValueError("registry manual PTC recommendation must be an object")
        for field in ("gainFromSet", "ptcFromSet"):
            if field in item and (not isinstance(item[field], str) or not re.fullmatch(r"[a-z0-9-]{1,120}", item[field])):
                raise ValueError("registry dependency must name one measurement set")
        files = item.get("files") or {}
        if not isinstance(files, dict) or set(files)-{"dark", "gain", "gainLinks", "ptc", "spectrumH", "spectrumV"}:
            raise ValueError("unsupported registry file role")
        def paths(value):
            if isinstance(value, dict):
                for child in value.values(): yield from paths(child)
            elif isinstance(value, list):
                for child in value: yield from paths(child)
            elif isinstance(value, str): yield value
            else: raise ValueError("invalid registry file path")
        for name in paths(files):
            if (not name or "\\" in name or PurePosixPath(name).is_absolute()
                    or any(x in ("", ".", "..") for x in name.split("/"))):
                raise ValueError("unsafe registry file path")
        scale = item.get("ptcGainScale", 1.)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("PTC gain scale must be positive and explicit")
        deps = {role: item[key] for role, key in (("relative_gain", "gainFromSet"), ("absolute_gain", "ptcFromSet")) if item.get(key)}
        derived = bool(deps or item.get("nominalGain") or item.get("ptcFit"))
        retired = item.get("calibrationOnly") is True
        monochrome = "MONOCHROM" in str(item.get("camera", {}).get("model", "")).upper()
        item.update(dependencies=deps, derivation={"ptc_gain_scale": float(scale),
            "gain_method": "nominal-ISO" if item.get("nominalGain") else "borrowed-measurement" if "relative_gain" in deps else "measured-ladder" if files.get("gain") else "unmeasured",
            "ptc_method": "manual-recommendation" if item.get("ptcFit") else "borrowed-measurement" if "absolute_gain" in deps else "measured-PTC" if files.get("ptc") else "unmeasured",
            "scale_authority": "registry-explicit" if "ptcGainScale" in item else "no-scale-declared; never inferred from bit depth"},
            channel_semantics="monochrome-position-slots" if monochrome else "source-CFA-contract-required",
            capability_status={"read_noise": "retired" if retired else "measured-unqualified" if files.get("dark") else "unmeasured",
                "gain": "research-derived" if derived else "measured-unqualified" if files.get("gain") or files.get("ptc") else "unmeasured",
                "pixel_maps": "unmeasured", "automatic_pixel_correction": False},
            runtime_qualification="retired-noise-source" if retired else "requires-compiled-profile-and-mode-validation",
            audit_notes=item.get("notes", []))
        entries[identity] = item
    ordered, visiting = [], set()
    def visit(identity):
        if identity not in entries: raise ValueError("registry dependency does not exist: " + identity)
        if identity in visiting: raise ValueError("registry dependency cycle")
        if identity in ordered: return
        visiting.add(identity)
        for parent in entries[identity]["dependencies"].values(): visit(parent)
        visiting.remove(identity); ordered.append(identity)
    for identity in entries: visit(identity)
    result = {"schema": SCHEMA, "source": {"project": "y-g-jiang/y-g-jiang.github.io",
        "commit": source_commit, "raw_sha256": raw_sha256, "path": "data/collect/sets.json"},
        "dependency_order": ordered, "sets": list(entries.values()), "profiles": [],
        "status": "catalogue-only; coefficients remain separately qualified"}
    result["version"] = digest(result)[:24]
    return result


def library_root():
    return Path(os.environ.get("DNGSCAN_CALIBRATION_LIBRARY_DIR",
        str(Path.home() / "Library/Application Support/AgXRAW/calibration-library")))


def _read(path):
    stat = path.stat()
    if stat.st_size > MAX_RELEASE_BYTES: raise ValueError("calibration library exceeds size limit")
    return _read_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _read_cached(path, mtime, ctime, size):
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_release(value)
    return value


def validate_release(value):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError("unsupported calibration library schema")
    unsigned = {k: v for k, v in value.items() if k != "version"}
    if value.get("version") != digest(unsigned)[:24]: raise ValueError("calibration library content identity mismatch")
    if (not isinstance(value.get("sets"), list) or not isinstance(value.get("profiles"), list)
            or len(value["profiles"]) > 4096):
        raise ValueError("invalid or oversized calibration library tables")
    source = value.get("source")
    if not isinstance(source, dict):
        raise ValueError("calibration library source must be an object")
    rebuilt = compile_catalog({"sets": value["sets"]}, source_commit=source.get("commit"), raw_sha256=source.get("raw_sha256"))
    if rebuilt["dependency_order"] != value.get("dependency_order"): raise ValueError("invalid dependency order")
    ids = {item["id"]: item for item in rebuilt["sets"]}
    from .calibration import _validated_prior
    for record in value["profiles"]:
        if not isinstance(record, dict):
            raise ValueError("release profile records must be objects")
        if not isinstance(record.get("set_id"), str) or record["set_id"] not in ids: raise ValueError("profile references unknown measurement set")
        if record.get("qualification") != "mode-validated-statistical-model":
            raise ValueError("release profile requires explicit statistical mode qualification")
        if ids[record["set_id"]].get("calibrationOnly"):
            raise ValueError("retired calibration-only dark cannot be a runtime noise profile")
        payload = record.get("payload")
        if record.get("sha256") != digest(payload): raise ValueError("release profile checksum mismatch")
        prior = _validated_prior(payload)
        contract = prior.get("readout_contract") or {}
        evidence = record.get("source_evidence")
        if (contract.get("storage_lossless") is not True
                or not any(key in contract for key in ("raw_geometry", "libraw_raw_geometry"))
                or prior.get("shutter") not in ("mechanical", "electronic", "efcs")
                or not prior.get("reference_dn_range")
                or not isinstance(evidence, dict)
                or not re.fullmatch(r"[a-f0-9]{40}", str(evidence.get("commit", "")))
                or not isinstance(evidence.get("input_sha256"), list)
                or not evidence["input_sha256"]
                or any(not re.fullmatch(r"[a-f0-9]{64}", str(h)) for h in evidence["input_sha256"])
                or not evidence.get("decoder_identity") or not evidence.get("units")):
            raise ValueError("release profile lacks qualified readout, DN scale or source evidence")
    return value


def current_library():
    pointer = library_root() / "current.json"
    if pointer.exists():
        selected = json.loads(pointer.read_text())
        if not isinstance(selected, dict):
            raise ValueError("calibration library selection must be an object")
        identity = selected.get("version")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{24}", identity):
            raise ValueError("invalid calibration library selection")
        return _read(library_root() / (identity + ".json"))
    return _read(PACKAGED)


def select_version(identity):
    root = library_root()
    if identity == "packaged":
        (root / "current.json").unlink(missing_ok=True)
    else:
        if not re.fullmatch(r"[a-f0-9]{24}", str(identity)): raise ValueError("invalid library version")
        _read(root / (identity + ".json"))
        from .calibration import _atomic_write
        _atomic_write(root / "current.json", {"version": identity})
    from . import priors
    priors._JPTC_CACHE = None
    return library_status()


def install_library(path):
    value = _read(Path(path))
    from .calibration import _atomic_write
    _atomic_write(library_root() / (value["version"] + ".json"), value)
    return select_version(value["version"])


def library_status():
    versions = ["packaged", *sorted(p.stem for p in library_root().glob("*.json")
                                   if re.fullmatch(r"[a-f0-9]{24}", p.stem))]
    try:
        value = current_library()
        return {"version": value["version"], "source": value["source"], "sets": value["sets"],
            "profiles": len(value["profiles"]), "offline": True, "versions": versions,
            "selection": value["version"] if (library_root()/"current.json").exists() else "packaged"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        return {"version": None, "source": None, "sets": [], "profiles": 0,
            "offline": True, "versions": versions, "selection": None,
            "error": str(error), "runtime_status": "public-library-disabled; basic-rendering-available"}


def library_fingerprint():
    # Failure still invalidates caches; it must not silently select another release.
    pointer = library_root() / "current.json"
    try:
        value = current_library(); return digest(value)
    except (OSError, ValueError, KeyError, TypeError):
        return "invalid-library-" + hashlib.sha256(pointer.read_bytes() if pointer.exists() else b"missing").hexdigest()


def compose_capabilities(base, candidates, iso):
    """Fill measured dark evidence only across an exact, explicit physical domain.

    Missing DN/linearisation/mode proof is an unavailable capability, never a
    reason to infer compatibility from same camera model or nominal bits.
    """
    from .calibration import curve_value, read_noise_issue, stored_dark_variance
    from .calibration_capabilities import usable, capability_prior, qualification
    from .priors import gain_e_per_dn, read_noise_e
    result = base
    sources = {"gain": base.get("source"), "read_noise": base.get("source")}
    required = {"libraw_raw_geometry", "storage_lossless"}
    gain_product = "absolute_gain" if usable(base, "absolute_gain") else "ptc_gain"
    gain_view = capability_prior(base, gain_product)
    contract = gain_view.get("readout_contract") or {}
    reference = base.get("reference_dn_range")
    domain = gain_view.get("acquisition_contract") or {}
    # File lists, coverage and fitting metadata are provenance, not the
    # physical DN domain. Actual CSV packages necessarily have different
    # source files; comparing their whole acquisition dictionaries prevented
    # any separately measured capabilities from composing.
    domain_fields = ("linearisation_curve", "adc_step", "cfa_pattern", "dn_domain", "variance_domain", "decoder_identity")
    def physical_domain(entry):
        from .measurement_domain import canonical_unit, decoder_family
        values = entry.get("acquisition_contract") or {}
        result = {key: values[key] for key in domain_fields if key in values}
        for key in ("dn_domain", "variance_domain"):
            if key in result:
                result[key] = canonical_unit(result[key], variance=key == "variance_domain")
        if "decoder_identity" in result:
            result["decoder_identity"] = decoder_family(result["decoder_identity"])
        return result
    def qualified(entry, capability):
        from .measurement_domain import declaration_issues
        view = capability_prior(entry, capability)
        if declaration_issues(view.get("acquisition_contract") or {}) or view.get("readout_declaration_issues"):
            return False
        return usable(entry, capability)
    if (base.get("readout_match_status") != "matched" or not required <= contract.keys()
            or domain.get("linearisation_curve") != "identity"
            or not qualified(base, gain_product) or not reference or not iso): return result
    gain_value = gain_e_per_dn(base, iso)
    if gain_value is None: return result
    gain = math.log2(gain_value)
    def compatible_product(candidate, product):
        view = capability_prior(candidate, product)
        return (qualified(candidate, product)
            and view.get("readout_contract") == contract
            and view.get("shutter") == gain_view.get("shutter")
            and candidate.get("reference_dn_range") == reference
            and candidate.get("make_equals") == base.get("make_equals")
            and set(candidate.get("model_equals", ())) == set(base.get("model_equals", ()))
            and physical_domain(view) == physical_domain(gain_view))
    compatible = []
    for candidate in candidates:
        if base.get("calibration_id") and candidate.get("calibration_id") == base["calibration_id"]:
            continue  # The same measurement is not an independent supplier.
        if candidate.get("readout_match_status") != "matched":
            continue
        permissions = {name: compatible_product(candidate, name)
                       for name in ("read_noise_dn", "stored_dark_variance", "spectrum_h", "spectrum_v")}
        if any(permissions.values()):
            candidate = copy.deepcopy(candidate)
            matches = candidate.setdefault("capability_matches", {})
            for name, permitted in permissions.items():
                if not permitted:
                    matches[name] = {"status": "mismatch", "reason": "cross-source-physical-domain-mismatch"}
            compatible.append(candidate)
    if compatible:
        result = copy.deepcopy(base)
        result["independent_constraint_priors"] = compatible
    if read_noise_issue(base, iso): return result  # Keep independent constraints even when RN is unresolved.
    for candidate in compatible:
        if read_noise_e(result, iso) is not None: break
        if not compatible_product(candidate, "read_noise_dn"): continue
        if read_noise_issue(candidate, iso, "read_noise_dn_log2iso"): continue
        read = curve_value(candidate, "read_noise_dn_log2iso", iso)
        if read is None: continue
        stored, stored_status = stored_dark_variance(candidate, iso)
        # Explicit contradictory/unresolved stored-code evidence cannot become
        # a physical-RN-only approximation merely because capabilities join.
        candidate_stored = candidate.get("stored_dark_variance_dn2_log2iso") or []
        if candidate_stored and stored is None:
            continue
        # One explicitly supported ISO, no new interpolated/extrapolated range.
        result["read_noise_log2iso_log2e"] = [[math.log2(iso), math.log2(read) + gain]]
        result["read_noise_dn_log2iso"] = [[math.log2(iso), read]]
        result["read_noise_composition_iso"] = iso
        result["phase_calibration"] = {}  # No invented cross-source phase gain.
        stored_supplier = None
        if result.get("stored_dark_variance_dn2_log2iso") and stored_dark_variance(result, iso)[0] is not None:
            stored_supplier = base
            sources["stored_dark_variance"] = base.get("source")
        if stored is not None and not result.get("stored_dark_variance_dn2_log2iso"):
            # Keep the original measured curve and its measured/interpolated
            # status. The composed RN curve authorizes only this selected ISO;
            # retaining source samples does not create a new runtime ISO span.
            result["source_format"] = candidate["source_format"]
            # These describe the dark estimator, not the shared physical DN
            # domain. Keep the supplier's contract with its copied variance;
            # a gain-only profile cannot authorize or relabel that estimator.
            statistics = ("stored_dark_variance_measurement", "stored_dark_variance_domain",
                          "sigma_clip_correction", "clip_variance_factor", "sheppard_assumption")
            result["acquisition_contract"] = dict(result.get("acquisition_contract") or {})
            for key in statistics:
                result["acquisition_contract"].pop(key, None)
                if key in (candidate.get("acquisition_contract") or {}):
                    result["acquisition_contract"][key] = copy.deepcopy(candidate["acquisition_contract"][key])
            if candidate_stored:
                result["stored_dark_variance_dn2_log2iso"] = copy.deepcopy(candidate_stored)
            else:
                # Legacy recovery remains explicitly an approximation; never
                # relabel a reconstructed variance as a direct measurement.
                result["noise_aperture"] = candidate.get("noise_aperture")
            sources["stored_dark_variance"] = candidate.get("source")
            stored_supplier = candidate
            result["stored_dark_variance_composition"] = {"source": candidate.get("source"),
                "iso": iso, "status": stored_status, "variance_dn2": stored}
        sources["read_noise"] = candidate.get("source")
        result["source"] = "Compatible capability composition"
        # Every composition retains its suppliers, including a legacy base.
        # None means an explicit legacy contract, not a qualified CSV record.
        # Electron RN depends on both suppliers: one legacy dependency keeps
        # that distinction even when the other has a measured qualification.
        composed = result.setdefault("composed_capabilities", {})
        dark_view = capability_prior(candidate, "read_noise_dn")
        dark_q = qualification(candidate, "read_noise_dn")
        gain_q = qualification(base, gain_product)
        for name in ("read_noise_dn", "read_noise_e"):
            q = copy.deepcopy(dark_q)
            if name == "read_noise_e":
                if q is not None and gain_q is not None:
                    q["source_ids"] = sorted(set(q["source_ids"]) | set(gain_q.get("source_ids", [])))
                else:
                    q = None
            composed[name] = {"qualification": q, "supplier": copy.deepcopy(dark_view),
                "qualification_semantics": "legacy-explicit-contract" if q is None else "measured-dependency-qualification",
                "additional_dependencies": [copy.deepcopy(gain_view)] if name == "read_noise_e" else [],
                "iso": iso, "source": candidate.get("source")}
            result.setdefault("capability_matches", {})[name] = {"status": "matched", "reason": "compatible-capability-suppliers-matched"}
        if stored_supplier is not None:
            stored_q = qualification(stored_supplier, "stored_dark_variance")
            composed["stored_dark_variance"] = {
                "qualification": copy.deepcopy(stored_q),
                "qualification_semantics": "legacy-explicit-contract" if stored_q is None else "measured-dependency-qualification",
                "supplier": copy.deepcopy(capability_prior(stored_supplier, "stored_dark_variance")),
                "iso": iso, "source": stored_supplier.get("source")}
            result["capability_matches"]["stored_dark_variance"] = {"status": "matched", "reason": "compatible-capability-suppliers-matched"}
        result["capability_sources"] = sources
        result["capability_measurements"] = {
            "gain": copy.deepcopy(base.get("measurement_acquisition")),
            "read_noise": copy.deepcopy(candidate.get("measurement_acquisition")),
            "stored_dark_variance": copy.deepcopy(stored_supplier.get("measurement_acquisition")) if stored_supplier is not None else None}
        break
    return result
