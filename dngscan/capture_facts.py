# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable, file-authored capture facts; never calibration-supplied defaults.

ExposureTime, FNumber and FocalLength retain their original TIFF rationals.
LensMake/LensModel form a *name* identity in an explicit EXIF namespace; they
are not a MakerNote hardware ID. Missing hard IDs and unmeasured conditions
remain unavailable. This object is independent of a decoder's pixel recipe.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct

from .capture_identity import _Tiff, read as read_identity
from .source_metadata import cached_source_metadata, source_metadata_session, current_source_session

SCHEMA = "agxraw-capture-facts-1"
ADAPTER_VERSION = 1
_TEXT_LIMIT = 160
_FIELDS = ("exposure_seconds", "aperture_fnumber", "focal_length_mm",
           "lens_make", "lens_model", "lens_identity", "lens_hardware_id",
           "body_serial", "lens_serial", "firmware", "temperature_c",
           "focus_distance_m", "optical_stack", "adapter_identity")
_NUMERIC = {33434: ("exposure_seconds", "s"), 33437: ("aperture_fnumber", "f-number"),
            37386: ("focal_length_mm", "mm")}
_TEXT = {42035: "lens_make", 42036: "lens_model", 42037: "lens_serial"}
_PRIVATE = {"body_serial", "lens_serial"}
_STATES = {"verified", "missing", "invalid", "conflict"}


@dataclass(frozen=True)
class LensIdentity:
    namespace: str
    kind: str
    model: str
    make: str | None = None

    def to_dict(self):
        return {"namespace": self.namespace, "kind": self.kind,
                "make": self.make, "model": self.model}


@dataclass(frozen=True)
class Observation:
    container: str
    ifd: str
    tag: int | None
    status: str
    value: str | float | None = None
    rational: tuple[int, int] | None = None
    reason: str | None = None
    adapter: str = "agxraw-exif-capture-facts-1"

    def to_dict(self):
        return {"source": {"container": self.container, "ifd": self.ifd,
                           "tag": self.tag, "adapter": self.adapter,
                           "adapter_version": ADAPTER_VERSION},
                "status": self.status, "value": self.value,
                "rational": list(self.rational) if self.rational is not None else None,
                "reason": self.reason}


@dataclass(frozen=True)
class Fact:
    status: str = "missing"
    value: str | float | LensIdentity | None = None
    unit: str | None = None
    observations: tuple[Observation, ...] = ()
    reason: str | None = None

    def to_dict(self):
        return {"status": self.status,
                "value": self.value.to_dict() if isinstance(self.value, LensIdentity) else self.value,
                "unit": self.unit, "reason": self.reason,
                "observations": [row.to_dict() for row in self.observations]}


@dataclass(frozen=True)
class CaptureFacts:
    fields: tuple[tuple[str, Fact], ...]
    container: str = "unknown"
    metadata_status: str = "unavailable"
    metadata_reason: str | None = None
    source_stat: tuple[int, int, int, int, int] | None = None
    source_header_sha256: str | None = None
    source_verification: str = "unavailable"

    def __deepcopy__(self, memo):
        # All members are frozen scalars/tuples. Metadata-session reuse keeps
        # the exact same immutable snapshot instead of a second mutable dict.
        return self

    def get(self, name):
        return dict(self.fields).get(name, Fact())

    def value(self, name):
        field = self.get(name)
        return field.value if field.status == "verified" else None

    def to_dict(self):
        return {"schema": SCHEMA, "adapter_version": ADAPTER_VERSION,
                "container": self.container, "metadata_status": self.metadata_status,
                "metadata_reason": self.metadata_reason,
                "source_identity": {"verification": self.source_verification,
                                    "stat": list(self.source_stat) if self.source_stat is not None else None,
                                    "header_sha256": self.source_header_sha256},
                "fields": {name: value.to_dict() for name, value in self.fields}}

    def matcher_values(self):
        values = {name: self.value(name) for name in _FIELDS}
        if isinstance(values["lens_identity"], LensIdentity):
            values["lens_identity"] = values["lens_identity"].to_dict()
        return {**values, "capture_facts": self.to_dict()}


def _qualified(observations, unit=None):
    observations = tuple(observations)
    valid = [row.value for row in observations if row.status == "verified"]
    distinct = set(valid)
    status = ("conflict" if len(distinct) > 1 else "invalid" if any(
        row.status == "invalid" for row in observations) else "verified" if valid else "missing")
    return Fact(status, valid[0] if status == "verified" else None, unit, observations,
                "disagreeing-file-observations" if status == "conflict" else
                "invalid-file-observation" if status == "invalid" else
                "file-field-not-present" if status == "missing" else None)


def _observe(tiff, entries, ifd, container, observations):
    for tag, rows in entries.items():
        if tag not in _NUMERIC and tag not in _TEXT:
            continue
        field = _NUMERIC[tag][0] if tag in _NUMERIC else _TEXT[tag]
        for kind, count, inline in rows:
            value = rational = None
            try:
                if tag in _NUMERIC:
                    if count != 1 or kind not in (5, 10):
                        raise ValueError("expected-one-unsigned-rational")
                    offset, = struct.unpack(tiff.endian + "L", inline)
                    rational = struct.unpack(tiff.endian + ("LL" if kind == 5 else "ll"), tiff.take(offset, 8))
                    if kind != 5:
                        raise ValueError("expected-unsigned-rational")
                    if rational[1] == 0:
                        raise ValueError("zero-denominator")
                    value = rational[0] / rational[1]
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError("nonpositive-capture-value")
                else:
                    if kind != 2 or not 0 < count <= _TEXT_LIMIT + 1:
                        raise ValueError("expected-bounded-ascii")
                    blob = inline[:count] if count <= 4 else tiff.take(
                        struct.unpack(tiff.endian + "L", inline)[0], count)
                    value = blob.rstrip(b"\0").decode("ascii", "strict").strip()
                    if not value or len(value) > _TEXT_LIMIT or any(ord(c) < 32 or ord(c) == 127 for c in value):
                        raise ValueError("invalid-capture-text")
                observation = Observation(container, ifd, tag, "verified", value, rational)
            except (ValueError, UnicodeError, struct.error) as error:
                observation = Observation(container, ifd, tag, "invalid", None, rational, str(error))
            observations[field].append(observation)


def _tiff_stream(path):
    """Bounded native TIFF or RAF embedded EXIF, never a full RAW read."""
    from . import metadata as md
    stream = Path(path).open("rb")
    try:
        head = stream.read(100)
        stream.seek(0)
        if head.startswith(md._RAF_MAGIC):
            if len(head) < 92:
                raise ValueError("truncated-raf-header")
            offset, count = struct.unpack_from(">LL", head, 84)
            stream.seek(0, 2)
            size = stream.tell()
            if count < 4 or offset > size or count > size - offset:
                raise ValueError("raf-exif-outside-container")
            stream.seek(offset)
            tiff = md._exif_tiff_from_jpeg(stream.read(min(count, 256 * 1024)))
            stream.close()
            if tiff is None:
                raise ValueError("raf-exif-unavailable")
            return io.BytesIO(tiff), "raf-embedded-exif"
        return stream, "tiff"
    except Exception:
        stream.close()
        raise


def read(path: Path) -> CaptureFacts:
    """Read and authorize one snapshot within the existing or a new source scope."""
    with source_metadata_session(path) as session:
        session.assert_consistent(phase="capture-facts-start")
        result = _read(path)
        session.assert_consistent(phase="capture-facts-commit")
        return result


@cached_source_metadata
def _read(path: Path) -> CaptureFacts:
    observations = {name: [] for name in _FIELDS}
    container, metadata_status, reason = "unknown", "unavailable", None
    try:
        stream, container = _tiff_stream(path)
        with stream:
            tiff = _Tiff(stream)
            root = tiff.ifd(tiff.first)
            if container == "tiff":
                container = "dng" if 50706 in root else "native-tiff"
            _observe(tiff, root, "IFD0", container, observations)
            try:
                pointer = tiff.pointer(root, 34665)
                if pointer is not None:
                    _observe(tiff, tiff.ifd(pointer), "IFD0.ExifIFD", container, observations)
            except (ValueError, struct.error) as error:
                for name in (*[value[0] for value in _NUMERIC.values()], *_TEXT.values()):
                    observations[name].append(Observation(container, "IFD0.ExifIFD", 34665,
                                                          "invalid", reason=str(error)))
            metadata_status = "verified"
    except (OSError, ValueError, UnicodeError, struct.error) as error:
        reason = type(error).__name__ + ": capture-metadata-unavailable"
    fields = {name: _qualified(observations[name], next(
        (unit for field, unit in _NUMERIC.values() if field == name), None)) for name in _FIELDS}
    for name in ("lens_hardware_id", "temperature_c", "focus_distance_m", "optical_stack", "adapter_identity"):
        fields[name] = Fact(reason="not-measured-by-this-adapter")
    # Reuse the independently scoped native body/firmware adapter. Its MakerNote
    # version must still never become firmware, including converted DNG inputs.
    identity = read_identity(path)
    for name in ("body_serial", "firmware"):
        diagnostic = (identity.get("identity_fields") or {}).get(name) or {}
        status = diagnostic.get("status", "missing")
        source_tags = {"DNG:IFD0.CameraSerialNumber": 50735, "EXIF:BodySerialNumber": 42033,
                       "Nikon:type2.SerialNumber": 29, "Nikon:native-NEF.Software.Ver": 305,
                       "Sony:native-ARW.Software.model-version": 305}
        rows = tuple(Observation(container, source, source_tags.get(source), status,
                                 identity.get(name) if status == "verified" else None,
                                 adapter="agxraw-capture-identity-1")
                     for source in diagnostic.get("sources", []))
        rows += tuple(Observation(container, row.get("source", "unknown"), None, "invalid",
                                  reason=row.get("reason"), adapter="agxraw-capture-identity-1")
                      for row in diagnostic.get("invalid_observations", []))
        fields[name] = Fact(status, identity.get(name) if status == "verified" else None,
                            observations=rows, reason=None if status == "verified" else
                            "identity-" + status)
    model, make = fields["lens_model"], fields["lens_make"]
    if model.status == "verified" and make.status in ("verified", "missing"):
        fields["lens_identity"] = Fact("verified", LensIdentity(
            "exif:LensMake+LensModel" if make.value is not None else "exif:LensModel",
            "metadata-name", model.value, make.value), observations=model.observations + make.observations)
    elif model.status in ("conflict", "invalid") or make.status in ("conflict", "invalid"):
        fields["lens_identity"] = Fact("conflict" if "conflict" in (model.status, make.status)
                                      else "invalid", reason="lens-name-evidence-not-qualified")
    session = current_source_session(path)
    identity = session.identity if session is not None else None
    return CaptureFacts(tuple((name, fields[name]) for name in _FIELDS), container, metadata_status, reason,
                        identity.stat if identity is not None else None,
                        identity.header_sha256 if identity is not None else None,
                        session.status if session is not None else "unavailable")


def from_dict(value) -> CaptureFacts | None:
    """Restore only this version's complete JSON-stable snapshot."""
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("adapter_version") != ADAPTER_VERSION
            or not isinstance(value.get("fields"), dict)
            or set(value["fields"]) != set(_FIELDS)
            or not isinstance(value.get("container"), str)
            or value.get("metadata_status") not in ("verified", "unavailable")
            or (value.get("metadata_reason") is not None and not isinstance(value["metadata_reason"], str))):
        raise ValueError("unsupported-capture-facts-snapshot")
    fields = []
    for name in _FIELDS:
        item = value["fields"][name]
        if not isinstance(item, dict) or item.get("status") not in _STATES:
            raise ValueError("invalid-capture-fact-status")
        if any(item.get(key) is not None and not isinstance(item[key], str) for key in ("unit", "reason")):
            raise ValueError("invalid-capture-fact-description")
        actual = item.get("value")
        if name == "lens_identity" and actual is not None:
            if (not isinstance(actual, dict) or set(actual) != {"namespace", "kind", "model", "make"}):
                raise ValueError("invalid-capture-lens-identity")
            actual = LensIdentity(**actual)
            if (actual.kind != "metadata-name" or not isinstance(actual.model, str) or not actual.model
                    or actual.namespace != ("exif:LensMake+LensModel" if actual.make is not None else "exif:LensModel")
                    or (actual.make is not None and (not isinstance(actual.make, str) or not actual.make))):
                raise ValueError("invalid-capture-lens-identity")
        elif actual is not None and (not isinstance(actual, (str, float, int)) or isinstance(actual, bool)):
            raise ValueError("invalid-capture-fact-value")
        if name in {value[0] for value in _NUMERIC.values()} and actual is not None and (
                not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual <= 0):
            raise ValueError("invalid-capture-number")
        if name in (*_TEXT.values(), "body_serial", "firmware") and actual is not None and (
                not isinstance(actual, str) or not actual or len(actual) > _TEXT_LIMIT):
            raise ValueError("invalid-capture-text")
        if item["status"] != "verified" and actual is not None:
            raise ValueError("unqualified-capture-fact-value")
        if item["status"] == "verified" and actual is None:
            raise ValueError("verified-capture-fact-without-value")
        rows = []
        if not isinstance(item.get("observations", []), list):
            raise ValueError("invalid-capture-observations")
        for row in item.get("observations", []):
            if not isinstance(row, dict):
                raise ValueError("invalid-capture-observation")
            source = row.get("source")
            if (row.get("status") not in _STATES or not isinstance(source, dict)
                    or any(not isinstance(source.get(key), str) for key in ("container", "ifd", "adapter"))
                    or source.get("adapter_version") != ADAPTER_VERSION
                    or (source.get("tag") is not None and type(source["tag"]) is not int)
                    or (row.get("reason") is not None and not isinstance(row["reason"], str))
                    or (row.get("value") is not None and (not isinstance(row["value"], (str, float, int))
                        or isinstance(row["value"], bool)))
                    or (isinstance(row.get("value"), (int, float)) and not math.isfinite(row["value"]))):
                raise ValueError("invalid-capture-observation")
            rational = row.get("rational")
            if rational is not None and (not isinstance(rational, list) or len(rational) != 2
                                        or any(type(number) is not int for number in rational)):
                raise ValueError("invalid-capture-rational")
            rows.append(Observation(source["container"], source["ifd"], source.get("tag"),
                row["status"], row.get("value"), tuple(rational) if rational is not None else None,
                row.get("reason"), source["adapter"]))
        fields.append((name, Fact(item["status"], actual, item.get("unit"), tuple(rows), item.get("reason"))))
    source = value.get("source_identity") or {}
    if (not isinstance(source, dict) or source.get("verification", "unavailable") not in
            ("verified", "unavailable", "changed", "unverifiable")
            or (source.get("header_sha256") is not None and (not isinstance(source["header_sha256"], str)
                or len(source["header_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in source["header_sha256"])))):
        raise ValueError("invalid-capture-source-identity")
    stat = source.get("stat")
    if (stat is not None and (not isinstance(stat, list) or len(stat) != 5
                             or any(type(number) is not int for number in stat))):
        raise ValueError("invalid-capture-source-identity")
    return CaptureFacts(tuple(fields), value["container"], value["metadata_status"], value.get("metadata_reason"),
                        tuple(stat) if stat is not None else None, source.get("header_sha256"),
                        source.get("verification", "unavailable"))


def public_facts(value) -> dict | None:
    """Publish values and provenance with both body and lens serials redacted."""
    if value is None:
        return None
    import copy
    result = value.to_dict() if isinstance(value, CaptureFacts) else copy.deepcopy(value)
    for name in _PRIVATE:
        item = (result.get("fields") or {}).get(name)
        if not isinstance(item, dict):
            continue
        serial = item.pop("value", None)
        if serial is not None:
            item["value_sha256"] = hashlib.sha256(str(serial).encode()).hexdigest()
        item["redacted"] = True
        for observation in item.get("observations", []):
            observed = observation.pop("value", None)
            if observed is not None:
                observation["value_sha256"] = hashlib.sha256(str(observed).encode()).hexdigest()
    return result


def fingerprint(value) -> str:
    payload = value.to_dict() if isinstance(value, CaptureFacts) else value
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()
