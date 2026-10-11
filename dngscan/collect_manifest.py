# SPDX-License-Identifier: GPL-3.0-or-later
"""Content-based, bounded JPTC Collect ingestion shared by CLI and GUI.

File names are provenance, never measurements. Format/version, columns and
acquisition declarations determine meaning. This module does not extract ZIP
members onto disk and never combines incompatible acquisition contracts.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile
import zlib

SCHEMA = "dngscan-collect-manifest-1"
MAX_FILES = 2048
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
FORMATS = {"JPTC-DARK/1": "dark", "JPTC-ISOGAIN/1": "gain",
           "JPTC-SPECTRUM/1": "spectrum", "JPTC/2": "ptc"}
ACQUISITION_SCHEMA = "dngscan-collect-acquisition-1"
JSON_FORMATS = {"dngscan-jptc-collect-1", "dngscan-jptc-prior-1", "dngscan-user-calibration-1"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def _safe_name(name):
    if not isinstance(name, str) or not name or "\x00" in name:
        raise ValueError("invalid input member name")
    # Backslashes and drive/UNC paths are rejected on every host, not only Windows.
    if "\\" in name or ntpath.splitdrive(name)[0] or name.startswith("/"):
        raise ValueError(f"unsafe input member path: {name}")
    parts = PurePosixPath(name).parts
    if (any(part in ("", "..", ".") for part in name.split("/")) or not parts
            or len(parts) > 16 or len(name) > 1024):
        raise ValueError(f"unsafe input member path: {name}")
    return str(PurePosixPath(name))


def _bounded_read(stream, limit):
    chunks, size = [], 0
    while True:
        chunk = stream.read(min(1024 * 1024, limit - size + 1))
        if not chunk:
            return b"".join(chunks)
        size += len(chunk)
        if size > limit:
            raise ValueError("measurement item exceeds decompressed size limit")
        chunks.append(chunk)


def _expand_entries(entries):
    """Validate containers before semantic parsing; all byte limits are inflated."""
    expanded, total, count = [], 0, 0
    for entry in entries:
        name = _safe_name(entry["name"])
        data = entry["content"]
        if not isinstance(data, bytes):
            raise ValueError("entry content must be bytes")
        if len(data) > MAX_TOTAL_BYTES:
            raise ValueError("input container exceeds size limit")
        if zipfile.is_zipfile(io.BytesIO(data)):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    infos = archive.infolist()
                    if len(infos) + count > MAX_FILES:
                        raise ValueError("input exceeds member count limit")
                    declared = sum(info.file_size for info in infos)
                    if declared + total > MAX_TOTAL_BYTES:
                        raise ValueError("ZIP exceeds decompressed total size limit")
                    names = {}
                    for info in infos:
                        member = _safe_name(info.filename.rstrip("/"))
                        mode = info.external_attr >> 16
                        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                            raise ValueError(f"ZIP contains unsupported special member: {info.filename}")
                        if info.flag_bits & 1:
                            raise ValueError("encrypted ZIP members are unsupported")
                        count += 1
                        if info.is_dir():
                            continue
                        if info.file_size > MAX_FILE_BYTES:
                            raise ValueError(f"ZIP member exceeds decompressed size limit: {member}")
                        with archive.open(info) as stream:
                            content = _bounded_read(stream, min(MAX_FILE_BYTES, MAX_TOTAL_BYTES-total))
                        total += len(content)
                        if member in names and names[member] != content:
                            raise ValueError(f"ZIP has conflicting duplicate member path: {member}")
                        names[member] = content
                        expanded.append({"name": member, "container": name, "content": content})
            except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as exc:
                raise ValueError(f"invalid ZIP input: {name}: {exc}") from exc
        else:
            count += 1
            total += len(data)
            if count > MAX_FILES or total > MAX_TOTAL_BYTES or len(data) > MAX_FILE_BYTES:
                raise ValueError("input exceeds measurement count or decompressed size limit")
            expanded.append({"name": name, "content": data})
    return expanded


def _source_entries(sources):
    paths = [sources] if isinstance(sources, (str, os.PathLike)) else list(sources)
    if not paths:
        raise ValueError("at least one calibration source is required")
    result, total = [], 0
    for source in paths:
        path = Path(source).expanduser()
        if path.is_symlink():
            raise ValueError(f"symbolic-link calibration input is unsupported: {path.name}")
        if path.is_dir():
            for root, dirs, files in os.walk(path, followlinks=False):
                for child in [*dirs, *files]:
                    candidate = Path(root)/child
                    if candidate.is_symlink():
                        raise ValueError(f"symbolic-link calibration input is unsupported: {candidate.name}")
                for child in sorted(files):
                    candidate = Path(root)/child
                    size = candidate.stat().st_size
                    if not candidate.is_file():
                        raise ValueError("special calibration input is unsupported")
                    total += size
                    if len(result) >= MAX_FILES or total > MAX_TOTAL_BYTES or size > MAX_TOTAL_BYTES:
                        raise ValueError("input exceeds measurement count or size limit")
                    with candidate.open("rb") as stream:
                        content = _bounded_read(stream, MAX_TOTAL_BYTES)
                    result.append({"name": candidate.relative_to(path).as_posix(), "content": content})
        else:
            if not path.is_file():
                raise ValueError(f"calibration source is not a file or directory: {path}")
            size = path.stat().st_size
            total += size
            if len(result) >= MAX_FILES or total > MAX_TOTAL_BYTES or size > MAX_TOTAL_BYTES:
                raise ValueError("input exceeds measurement count or size limit")
            with path.open("rb") as stream:
                result.append({"name": path.name, "content": _bounded_read(stream, MAX_TOTAL_BYTES)})
    return result


def parse_table(content):
    """Strict CSV structure; comments are metadata and conflicting headers fail."""
    text = content.decode("utf-8-sig")
    headers, body = {}, []
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            key, separator, value = line.lstrip()[1:].partition(":")
            key, value = key.strip(), value.strip().rstrip(",")
            if separator and key and " " not in key:
                if key in headers and headers[key] != value:
                    raise ValueError(f"conflicting #{key} header declarations")
                headers[key] = value
        else:
            body.append(line)
    reader = csv.reader(body, strict=True)
    columns = next(reader, [])
    columns = [column.strip() for column in columns]
    if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("missing or duplicate CSV columns")
    rows = []
    for number, values in enumerate(reader, 2):
        if len(values) != len(columns):
            raise ValueError(f"CSV row {number}: column count disagrees with header")
        rows.append(dict(zip(columns, (value.strip() for value in values))))
    if not rows:
        raise ValueError("measurement table has no rows")
    return headers, columns, rows


def _iso_number(value):
    result = float(value)
    if not math.isfinite(result) or not 1 <= result <= 2**24 or not result.is_integer():
        raise ValueError("measurement ISO must be a finite positive integer")
    return int(result)


def resolve_ptc_iso(header, rows, name, *, allow_filename=True):
    """Authoritative header/row ISO wins; legacy fallback is explicitly labelled."""
    declared = _iso_number(header["ISO"]) if header.get("ISO") else None
    values = {_iso_number(row["ISO"]) for row in rows if row.get("ISO")}
    if any("ISO" in row and not row["ISO"] for row in rows):
        raise ValueError("PTC ISO column contains missing values")
    if len(values) > 1 or declared is not None and values and values != {declared}:
        raise ValueError("PTC declared ISO conflicts with row ISO; separate ISO curves are required")
    if declared is not None or values:
        return declared if declared is not None else next(iter(values)), "header-and-rows" if declared and values else "header" if declared else "rows"
    if allow_filename:
        match = re.search(r"ptc[-_]iso([1-9][0-9]*)", Path(name).stem, re.I)
        if match:
            return _iso_number(match[1]), "legacy-filename-fallback"
    raise ValueError("PTC has no declared or row ISO; filename-independent interpretation is unavailable")


def _contract(header):
    from .calibration import _shutter
    result = {}
    fields = {"camera": ("Camera",), "body": ("BodySerialNumber", "CameraSerialNumber", "SerialNumber"),
        "firmware": ("Firmware",), "shutter": ("ShutterType",),
        "readout": ("ReadoutMode", "Readout"), "capture_kind": ("DriveMode",), "compression": ("Compression",),
        "raw_geometry": ("RawSize",), "cfa": ("CfaPattern",),
        "linearisation": ("LinearisationCurve",), "adc_step": ("AdcStep",),
        "dn_maximum": ("SensorMaximum",), "dn_domain": ("DNDomain",),
        "variance_domain": ("VarianceDomain",),
        "decoder_identity": ("Decoder",)}
    for key, aliases in fields.items():
        values = {" ".join(header[a].split()).upper() for a in aliases if header.get(a)}
        if len(values) > 1:
            raise ValueError(f"conflicting acquisition declarations: {key}")
        if not values:
            continue
        value = next(iter(values))
        if key == "shutter":
            value = _shutter(value)
        elif key == "raw_geometry":
            match = re.fullmatch(r"\s*([1-9][0-9]*)\s*[xX×]\s*([1-9][0-9]*)\s*", value)
            if not match:
                raise ValueError("invalid RawSize declaration")
            value = [int(match[1]), int(match[2])]
        elif key in ("adc_step", "dn_maximum"):
            value = float(value)
            if not math.isfinite(value) or value < 0 or key == "dn_maximum" and value == 0:
                raise ValueError(f"invalid acquisition {key}")
        else:
            value = value.upper()
            if key in ("dn_domain", "variance_domain"):
                from .measurement_domain import canonical_unit
                value = (canonical_unit(value, variance=key == "variance_domain") or value).upper()
        result[key] = value
    # PTC can use its own explicitly supplied four-slot pedestal. Otherwise
    # read_anchors() consumes the paired dark's phase reference.
    if header.get("Format") == "JPTC/2" and header.get("BlackLevel"):
        levels = [float(v) for v in header["BlackLevel"].split(",")]
        if len(levels) >= 4 and all(math.isfinite(v) for v in levels):
            result["black_reference"] = "ptc-header"
    return result


def _compatible(a, b):
    return all(a[key] == b[key] for key in a.keys() & b.keys())


def acquisition_qualification(measurements):
    """Separate a unique association from proof of a shared acquisition domain.

    Every value remains attached to the CSV that declared it. No other file's
    shutter, LUT or ADC declaration fills a measurement's missing metadata.
    In particular, a single compatible group is not a shared batch statement.
    """
    if not isinstance(measurements, dict) or not measurements:
        raise ValueError("measurement acquisition requires explicit source records")
    valid_kinds = {"dark", "gain", "ptc", "spectrum_h", "spectrum_v"}
    allowed = {"camera", "body", "firmware", "shutter", "readout", "capture_kind", "compression",
               "raw_geometry", "cfa", "linearisation", "adc_step", "dn_maximum", "dn_domain",
               "variance_domain", "decoder_identity", "black_reference"}
    fields = set()
    for identity, record in measurements.items():
        if (not isinstance(identity, str) or re.fullmatch(r"[a-f0-9]{64}", identity) is None
                or not isinstance(record, dict) or set(record) != {"kind", "declared"}
                or record["kind"] not in valid_kinds or not isinstance(record["declared"], dict)):
            raise ValueError("invalid measurement acquisition source record")
        if set(record["declared"]) - allowed:
            raise ValueError("unsupported measurement acquisition field")
        for field, value in record["declared"].items():
            if field == "raw_geometry":
                if (not isinstance(value, list) or len(value) != 2
                        or any(type(v) is not int or v <= 0 for v in value)):
                    raise ValueError("invalid measurement acquisition geometry")
            elif field in ("adc_step", "dn_maximum"):
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        or value < 0 or field == "dn_maximum" and value == 0):
                    raise ValueError("invalid measurement acquisition DN scale")
            elif not isinstance(value, str) or not value.strip():
                raise ValueError("invalid measurement acquisition text")
        fields.update(record["declared"])
    coverage, common, associated = {}, {}, {}
    for field in sorted(fields):
        supplied = {identity: record["declared"][field] for identity, record in measurements.items()
                    if field in record["declared"]}
        values = list(supplied.values())
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"conflicting acquisition declarations: {field}")
        missing = sorted(set(measurements) - supplied.keys())
        coverage[field] = {"status": "incomplete" if missing else "complete",
                           "declared_by": supplied, "missing_sources": missing}
        associated[field] = values[0]
        if not missing:
            common[field] = values[0]
    # DN maximum is a separate endpoint measurement: dark exports do not
    # necessarily declare one. It is never inferred from ADC/storage bits.
    physical = {"camera", "shutter", "compression", "raw_geometry", "cfa",
                "linearisation", "adc_step"}
    conditional = {"body", "firmware", "readout", "capture_kind", "dn_domain", "variance_domain", "decoder_identity"} & fields
    required = physical | conditional
    issues = ["measurement-" + field.replace("_", "-") + "-incomplete"
              for field in sorted(required) if field not in common]
    if common.get("linearisation") not in (None, "IDENTITY"):
        issues.append("measurement-linearisation-unsupported")
    if common.get("adc_step") == 0:
        issues.append("measurement-adc-step-unresolved")
    if common.get("compression") not in (None, "LOSSLESS", "LOSSLESS COMPRESSION", "UNCOMPRESSED", "NONE", "无损压缩", "无压缩"):
        issues.append("measurement-compression-unverified")
    from .measurement_domain import format_domains
    numeric_domains, domain_issues = format_domains(measurements)
    issues.extend(domain_issues)
    by_kind = {kind: {i for i, r in measurements.items() if r["kind"] == kind}
               for kind in valid_kinds}
    dark, gain, ptc = by_kind["dark"], by_kind["gain"], by_kind["ptc"]
    ptc_needs_dark = any(measurements[i]["declared"].get("black_reference") != "ptc-header" for i in ptc)
    ptc_sources = ptc | (dark if ptc_needs_dark else set())
    # These mirror read_anchors(), read_isogain(), read_dark(), and
    # read_whiteness(): a product sees only the CSVs its arithmetic consumes.
    capability_files = {
        "ptc_gain": (ptc_sources, bool(ptc), {"dark"} if ptc_needs_dark else set()),
        "relative_gain": (dark | gain, bool(gain), {"dark"}),
        "absolute_gain": (ptc_sources | (dark | gain if gain else set()), bool(ptc),
                          {"dark"} if ptc_needs_dark or gain else set()),
        "read_noise_dn": (dark, bool(dark), set()),
        # Electron RN can use an independently qualified PTC anchor at that
        # ISO. A relative-gain ladder is an optional range extension.
        "read_noise_e": (ptc_sources | dark, bool(ptc and dark), set()),
        "stored_dark_variance": (dark, bool(dark), set()),
        "spectrum_h": (dark | by_kind["spectrum_h"], bool(by_kind["spectrum_h"]), {"dark"}),
        "spectrum_v": (dark | by_kind["spectrum_v"], bool(by_kind["spectrum_v"]), {"dark"}),
    }
    products = {}
    for name, (sources, present, needed) in capability_files.items():
        local_fields = set().union(*(measurements[i]["declared"] for i in sources)) if sources else set()
        local_required = physical | ({"body", "firmware", "readout", "capture_kind", "dn_domain",
                                     "variance_domain", "decoder_identity"} & local_fields)
        missing = {field: sorted(i for i in sources if field not in measurements[i]["declared"])
                   for field in sorted(local_required)} if present else {}
        missing = {field: ids for field, ids in missing.items() if ids}
        reasons = ["measurement-" + field.replace("_", "-") + "-incomplete" for field in missing]
        reasons += ["measurement-" + kind + "-dependency-unavailable" for kind in sorted(needed) if not by_kind[kind]]
        if present:
            selected = {i: measurements[i] for i in sources}
            _, domain_errors = format_domains(selected)
            reasons.extend(domain_errors)
            for record in selected.values():
                declared = record["declared"]
                if declared.get("linearisation") not in (None, "IDENTITY"):
                    reasons.append("measurement-linearisation-unsupported")
                if declared.get("adc_step") == 0:
                    reasons.append("measurement-adc-step-unresolved")
                if declared.get("compression") not in (None, "LOSSLESS", "LOSSLESS COMPRESSION", "UNCOMPRESSED", "NONE", "无损压缩", "无压缩"):
                    reasons.append("measurement-compression-unverified")
        products[name] = {"status": "unmeasured" if not present else "unverified" if reasons else "qualified",
                          "source_ids": sorted(sources), "missing_fields": missing,
                          "reasons": sorted(set(reasons)) if present else ["measurement-not-supplied"],
                          "dependency_policy": "actual fitting inputs; no cross-CSV condition inheritance"}
    # Existing API uses read_noise for the independently measured DN product.
    products["read_noise"] = dict(products["read_noise_dn"])
    return {"schema": ACQUISITION_SCHEMA, "measurements": measurements,
            "association_contract": associated, "common_contract": common,
            "field_coverage": coverage, "capability_qualification": products,
            "numeric_domains": numeric_domains,
            "issues": issues, "inheritance": "none; each measurement declares its own physical conditions"}


def _classify(entry):
    digest = hashlib.sha256(entry["content"]).hexdigest()
    record = {"id": digest, "sha256": digest, "size": len(entry["content"]),
        "name": entry["name"], "occurrences": [{k:v for k,v in entry.items() if k != "content"}],
        "status": "unused", "diagnostics": []}
    try:
        if entry["content"].lstrip().startswith(b"{"):
            payload = json.loads(entry["content"])
            if not isinstance(payload, dict) or payload.get("format") not in JSON_FORMATS:
                raise ValueError("unsupported calibration JSON format")
            if payload["format"] == "dngscan-user-calibration-1":
                payload = payload.get("payload")
            from .calibration import _validated_prior
            prior = _validated_prior(payload)
            _digest(payload)  # Reject non-finite values even in otherwise unused provenance.
            record.update(kind="json", format=payload["format"], iso_points=[2**p[0] for p in prior.get("gain_log2iso_log2epd", [])],
                contract={"camera": " ".join([prior["make_contains"], *prior["model_labels"][:1]]), "shutter": prior.get("shutter")},
                status="recognized")
            return record, {"payload": payload}
        header, columns, rows = parse_table(entry["content"])
        fmt = header.get("Format")
        if fmt not in FORMATS:
            raise ValueError("unsupported or missing #Format version" + (f": {fmt}" if fmt else ""))
        kind = FORMATS[fmt]
        required = {"dark": {"ISO", "ColorIndex", "BlackA", "BlackB", "StdDiffClipped"},
            "gain": {"ISO", "ColorIndex", "ClipFrac", "Mean", "ShutterSec"},
            "ptc": {"G1_Mean", "G1_Std"}, "spectrum": {"freq"}}[kind]
        if missing := required-set(columns):
            raise ValueError(f"{fmt}: missing required columns: {', '.join(sorted(missing))}")
        isos = []
        if kind == "ptc":
            iso, authority = resolve_ptc_iso(header, rows, entry["name"])
            isos = [iso]
            record["iso_authority"] = authority
            if authority == "legacy-filename-fallback":
                record["diagnostics"].append("legacy PTC ISO inferred from filename; authoritative ISO is absent")
        elif kind == "spectrum":
            axis = {"horizontal": "h", "vertical": "v", "h": "h", "v": "v"}.get(header.get("Axis", ""))
            if axis is None:
                # Older tables omitted #Axis; unlike modern metadata this is explicitly degraded.
                match = re.search(r"spectrum[-_]([hv])(?:\.|$)", entry["name"], re.I)
                if match:
                    axis = match[1].lower()
                    record["diagnostics"].append("legacy spectrum axis inferred from filename")
                else:
                    raise ValueError("spectrum has no declared Axis; cannot establish horizontal/vertical direction")
            kind = "spectrum_" + axis
            powers = [c for c in columns if re.fullmatch(r"iso[1-9][0-9]*_C[01][01]_(single|diff)", c)]
            if not powers:
                raise ValueError("spectrum has no supported ISO/CFA power columns")
            isos = sorted({int(re.match(r"iso([0-9]+)_", c)[1]) for c in powers})
            if header.get("ISO") and set(isos) != {_iso_number(header["ISO"])}:
                raise ValueError("spectrum declared ISO conflicts with power-column ISOs")
            unit = header.get("FreqUnit", "")
            if not unit.startswith("cycles per channel-plane pixel"):
                raise ValueError("spectrum frequency unit is missing or unsupported; a sensor/channel-plane scale cannot be inferred")
        else:
            isos = sorted({_iso_number(row["ISO"]) for row in rows})
            if header.get("ISO") and set(isos) != {_iso_number(header["ISO"])}:
                raise ValueError("declared ISO conflicts with row ISO")
            if kind == "dark":
                identities = [(row["ISO"], row.get("Channel") or row["ColorIndex"]) for row in rows]
                if len(set(identities)) != len(identities):
                    raise ValueError("repeated dark ISO/CFA measurement rows require explicit selection; averaging is not permitted")
        contract = _contract(header)
        record.update(kind=kind, format=fmt, iso_points=isos, contract=contract,
            columns=columns, row_count=len(rows), status="recognized")
        return record, {"header": header, "rows": rows, "content": entry["content"]}
    except (ValueError, TypeError, KeyError, UnicodeError, csv.Error, OverflowError) as exc:
        record["reason"] = str(exc)
        record["diagnostics"].append(str(exc))
        return record, {}


def _empty_capabilities():
    from .calibration_capabilities import PRODUCTS
    return {**dict.fromkeys(PRODUCTS, False), "read_noise": False, "spectrum_2d": False}


def _build_group(group, records, contents):
    """Deterministic temporary names let legacy numeric parsers consume explicit selections."""
    from .calibration import build, sanitize_json, _validated_prior
    selected = [records[i] for kind, ids in group["files"].items() for i in ids]
    if any(record["kind"] == "json" for record in selected):
        return contents[selected[0]["id"]]["payload"]
    with tempfile.TemporaryDirectory(prefix="dngscan-collect-") as temporary:
        directory = Path(temporary)
        paths = {}
        for record in selected:
            kind, identity = record["kind"], record["id"]
            filename = {"dark": "dark-scalars.csv", "gain": "gain-levels.csv",
                "spectrum_h": "spectrum-h.csv", "spectrum_v": "spectrum-v.csv"}.get(kind)
            if filename is None:
                filename = f"ptc-iso{record['iso_points'][0]}-{identity[:16]}.csv"
            path = directory/filename
            path.write_bytes(contents[identity]["content"])
            paths.setdefault(kind, []).append(path)
        payload = sanitize_json(build(directory, None, files=paths))
    payload["id"] = f"{payload['camera']} (user JPTC Collect)"
    payload["source"].update(kind="user-imported JPTC Collect measurements",
        set="local-" + group["content_id"][:12], inputs={r["id"]: r["sha256"] for r in selected})
    for key in ("url_base", "url_note"):
        payload["source"].pop(key, None)
    payload["collect_content_id"] = group["content_id"]
    _validated_prior(payload)
    return payload


def _prepare(entries):
    expanded = _expand_entries(entries)
    records, contents = {}, {}
    for entry in expanded:
        record, parsed = _classify(entry)
        identity = record["id"]
        if identity in records:
            records[identity]["occurrences"].extend(record["occurrences"])
            records[identity]["diagnostics"].append("identical content occurs more than once; not averaged")
        else:
            records[identity], contents[identity] = record, parsed
    recognized = [r for r in records.values() if r["status"] == "recognized"]
    groups = {}
    for record in recognized:
        if record["kind"] == "json":
            key = "json-"+record["id"]
            combined = record["contract"]
        else:
            compatible = [r for r in recognized if r["kind"] != "json" and _compatible(record["contract"], r["contract"])]
            combined, ambiguity = dict(record["contract"]), []
            for other in compatible:
                for field, value in other["contract"].items():
                    if field in combined and combined[field] != value:
                        ambiguity.append(field)
                    else:
                        combined[field] = value
            if ambiguity:
                record.update(status="ambiguous", reason="incomplete acquisition metadata matches multiple groups: " + ", ".join(sorted(set(ambiguity))))
                record["diagnostics"].append(record["reason"])
                continue
            key = _digest(combined)
        group = groups.setdefault(key, {"contract": combined, "files": {}, "diagnostics": [],
            "capabilities": _empty_capabilities(), "status": "partial", "import_status": "not-imported",
            "match_status": "not-evaluated", "apply_status": "not-applied"})
        group["files"].setdefault(record["kind"], []).append(record["id"])
        record["status"] = "assigned"
    payloads = {}
    for group in groups.values():
        for ids in group["files"].values():
            ids.sort()
        members = [records[i] for ids in group["files"].values() for i in ids]
        group["content_id"] = _digest(sorted(r["id"] for r in members))
        group["id"] = group["content_id"][:24]
        contract = group["contract"]
        camera = contract.get("camera", "")
        make = camera.split()[0] if camera else ""
        group.update(camera=camera, make=make, models=[camera[len(make):].strip()] if camera else [],
            shutter=contract.get("shutter"), compression=contract.get("compression"),
            raw_geometry=contract.get("raw_geometry"), cfa=contract.get("cfa"),
            iso_points=sorted({i for record in members for i in record.get("iso_points", [])}))
        duplicates = [kind for kind,ids in group["files"].items() if kind != "ptc" and len(ids)>1]
        ptc_isos = [records[i]["iso_points"][0] for i in group["files"].get("ptc", [])]
        if len(set(ptc_isos)) != len(ptc_isos):
            duplicates.append("same-ISO PTC")
        if duplicates:
            group["status"] = "ambiguous"
            group["diagnostics"].append("independent repeated measurement tables require explicit selection; not averaged: " + ", ".join(duplicates))
            for record in members:
                record["status"] = "ambiguous"
                record["reason"] = group["diagnostics"][-1]
            continue
        if not camera:
            group["status"] = "rejected"
            group["diagnostics"].append("camera make/model are undeclared")
            for record in members:
                record.update(status="rejected", reason=group["diagnostics"][-1], group_id=group["id"])
            continue
        for record in members:
            missing = sorted(set(contract)-set(record["contract"]))
            if missing:
                record["diagnostics"].append("uniquely associated despite undeclared acquisition fields: " + ", ".join(missing))
        try:
            payload = _build_group(group, records, contents)
            from .calibration import _validated_prior
            prior = _validated_prior(payload)
            acquisition = prior.get("measurement_acquisition") or {}
            from .calibration_capabilities import qualification as product_qualification
            qualification = {name: product_qualification(prior, name)
                             for name in acquisition.get("capability_qualification", {})}
            group["measurement_acquisition"] = acquisition
            group["capability_qualification"] = qualification
            group["runtime_contract"] = {"shutter": prior.get("shutter"),
                "compression": payload.get("compression"),
                "readout_contract": prior.get("readout_contract"),
                "issues": prior.get("readout_declaration_issues", [])}
            group.update(shutter=prior.get("shutter"), compression=payload.get("compression"))
            cap = group["capabilities"]
            cap.update(absolute_gain=bool(prior.get("gain_log2iso_log2epd")),
                read_noise=bool(prior.get("read_noise_dn_log2iso") or prior.get("read_noise_log2iso_log2e")),
                ptc_gain=bool(prior.get("ptc_gain_log2iso_log2epd")),
                relative_gain=bool((prior.get("gain_ladder_diagnostics") or {}).get("components")),
                read_noise_dn=bool(prior.get("read_noise_dn_log2iso")),
                read_noise_e=bool(prior.get("read_noise_log2iso_log2e")),
                stored_dark_variance=bool(prior.get("stored_dark_variance_dn2_log2iso")))
            for axis in ("h", "v"):
                cap["spectrum_"+axis] = bool(prior.get("noise_spectrum", {}).get("axes", {}).get(axis))
            for capability, item in qualification.items():
                if item["status"] != "qualified":
                    cap[capability] = False
            from .calibration_capabilities import stages
            group["capability_states"] = stages(prior)
            group["diagnostics"].extend(acquisition.get("issues", []))
            group["status"] = "ready" if cap["absolute_gain"] and cap["read_noise"] else "partial"
            if not cap["absolute_gain"]:
                group["diagnostics"].append("no qualified PTC anchor: absolute electron gain is unavailable")
            if not cap["read_noise"]:
                group["diagnostics"].append("no usable dark measurement: independently measured read noise is unavailable")
            if cap["spectrum_h"] != cap["spectrum_v"]:
                group["diagnostics"].append("one measured spectrum axis cannot establish a 2D spectrum")
            payloads[group["id"]] = payload
            for anchor in payload.get("ptc_anchors", []):
                if anchor.get("status") != "usable":
                    identity = next((r["id"] for r in members if r["kind"] == "ptc" and r["id"].startswith(
                        anchor["file"].rsplit("-",1)[-1].removesuffix(".csv"))), None)
                    if identity is not None:
                        records[identity].update(status="rejected", reason="PTC fit rejected: " + anchor.get("reason", "unusable"))
        except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError, IndexError, SystemExit) as exc:
            group["status"] = "partial"
            group["diagnostics"].append("no usable runtime profile: " + str(exc))
            for record in members:
                record.update(status="rejected", reason=group["diagnostics"][-1])
        for record in members:
            record["group_id"] = group["id"]
    files = sorted(records.values(), key=lambda r:r["id"])
    content_id = _digest(sorted(records))
    result = {"schema": SCHEMA, "id": content_id[:24], "content_id": content_id,
        "groups": sorted(groups.values(), key=lambda g:g["id"]), "files": files,
        "unused_files": [r["id"] for r in files if r["status"] in ("unused", "rejected")],
        "ambiguous_files": [r["id"] for r in files if r["status"] == "ambiguous"],
        "diagnostics": [], "import_status": "not-imported", "match_status": "not-evaluated", "apply_status": "not-applied"}
    if result["unused_files"]:
        result["diagnostics"].append("some input files are unsupported or unusable; inspect each file reason")
    if result["ambiguous_files"]:
        result["diagnostics"].append("some measurements require explicit selection before import")
    if not groups:
        result["diagnostics"].append("no unambiguous measurement groups were found")
    return result, payloads


def prepare_sources(sources):
    """Internal adapter: validated public manifest and runtime payloads."""
    return _prepare(_source_entries(sources))


def inspect_sources(sources):
    """Inspect one directory/file or a sequence of archives without installing."""
    return prepare_sources(sources)[0]


def inspect_entries(entries):
    """Browser adapter; the same safety, classification and numerical checks."""
    return _prepare(entries)[0]
