# SPDX-License-Identifier: GPL-3.0-or-later
"""Immutable sensor statistics, memoized only on truly immutable RAW evidence.

The existing analysis helpers remain the algorithm and patch seams. A summary
contains scalars only; its private per-evidence memo never owns a sensor array.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import weakref

from ._deps import np
from .models import RawEvidence


_SENSOR_SUMMARY_SCHEMA_VERSION = 1
_CEILING_POLICY_NAMES = (
    "CEILING_MIN_PILE_FRACTION", "CEILING_MIN_PILE_PIXELS",
    "CEILING_NEAR_WINDOW_SCALE", "CEILING_PLAUSIBLE_FRACTION",
)


@dataclass(frozen=True)
class SensorSummary:
    channel_ids: tuple[int, ...]
    ceilings: tuple[tuple[int, int], ...]
    exact_counts: tuple[tuple[int, int], ...]
    near_counts: tuple[tuple[int, int], ...]
    spike_ok: tuple[tuple[int, bool], ...]
    saturation_levels: tuple[tuple[int, int], ...]
    fullwell: int
    fullwell_channel_ids: tuple[int, ...]
    fullwell_note: str
    channel_fullwell: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _CacheSignature:
    key: tuple
    # Arrays AND algorithm callables are weakly bound. A replaced object cannot
    # accidentally match an old entry merely because Python reuses its address.
    references: tuple[weakref.ReferenceType, ...]
    metadata: tuple

    def matches(self, other: _CacheSignature) -> bool:
        return (self.key == other.key and len(self.references) == len(other.references)
                and all(a() is not None and a() is b()
                        for a, b in zip(self.references, other.references)))


@dataclass(frozen=True)
class _CacheEntry:
    signature: _CacheSignature
    summary: SensorSummary


def _number_snapshot(value):
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in (int, float, bool) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("not a finite metadata scalar")
    return value


def _metadata_snapshot(white_level, camera_white_levels):
    return (_number_snapshot(white_level), tuple(_number_snapshot(v) for v in camera_white_levels))


def _immutable_array_signature(array):
    """Describe ndarray metadata while proving the final buffer is immutable.

    Merely clearing WRITEABLE on owned memory, or on a view of mutable memory,
    is reversible. Only an ndarray base chain ending in bytes qualifies.
    """
    chain, references, seen = [], [], set()
    node = array
    while type(node) is np.ndarray:
        if id(node) in seen or node.flags.writeable or node.dtype.hasobject:
            return None
        seen.add(id(node))
        references.append(weakref.ref(node))
        chain.append((id(node), tuple(node.shape), node.dtype.str, tuple(node.strides),
                      int(node.__array_interface__["data"][0]),
                      bool(node.flags.writeable), bool(node.flags.owndata),
                      bool(node.flags.aligned), bool(node.flags.c_contiguous),
                      bool(node.flags.f_contiguous)))
        node = node.base
    if not chain or type(node) is not bytes:
        return None
    # Store only the buffer's scalar identity, never the full bytes owner.
    return (tuple(chain), id(node), len(node)), tuple(references)


def _cache_signature(raw_image, raw_colors, white_level, camera_white_levels, *, evidence):
    """Private benchmark seam: returning None disables only summary reuse."""
    from . import analysis

    if (not isinstance(evidence, RawEvidence)
            or raw_image is not evidence.raw_image or raw_colors is not evidence.raw_colors
            or evidence.sample_kind not in ("cfa", "linear-camera-rgb")):
        return None
    image = _immutable_array_signature(raw_image)
    colors = _immutable_array_signature(raw_colors)
    if image is None or colors is None:
        return None
    try:
        metadata = _metadata_snapshot(white_level, camera_white_levels)
        if metadata != _metadata_snapshot(evidence.white_level, evidence.camera_white_levels):
            return None
        provenance = (evidence.provider, evidence.provider_version, evidence.sample_kind)
        if any(value is not None and type(value) is not str for value in provenance):
            return None
        policy = tuple(_number_snapshot(getattr(analysis, name)) for name in _CEILING_POLICY_NAMES)
        algorithms = (analysis.channel_saturation_levels, analysis.detect_ceilings,
                      analysis.resolve_fullwell)
        algorithm_refs = tuple(weakref.ref(function) for function in algorithms)
    except (TypeError, ValueError, AttributeError):
        # Unusual/manual inputs keep the historical algorithm and error behavior;
        # they do not acquire authority to reuse an older summary.
        return None
    return _CacheSignature(
        (_SENSOR_SUMMARY_SCHEMA_VERSION, image[0], colors[0], metadata, provenance,
         policy, tuple(id(function) for function in algorithms)),
        image[1] + colors[1] + algorithm_refs,
        metadata,
    )


def summarize_sensor(raw_image: Any, raw_colors: Any, white_level: int,
                     camera_white_levels: list[float], *, evidence=None) -> SensorSummary:
    """Run the existing sensor reductions or reuse this evidence's exact summary.

    The cache excludes margin, labels, noise, SNR, scene metrics and clip-mask
    refresh: those operations still run in their original callers every time.
    """
    from . import analysis

    signature = _cache_signature(raw_image, raw_colors, white_level, camera_white_levels,
                                 evidence=evidence)
    entry = getattr(evidence, "_sensor_summary_cache", None) if signature is not None else None
    if isinstance(entry, _CacheEntry) and entry.signature.matches(signature):
        return entry.summary

    channel_ids = [int(value) for value in sorted(np.unique(raw_colors).tolist())]
    # The signature's metadata is also the computation's input, not only a
    # lookup key. A mutable white-level list changing and changing back during
    # the scan cannot attach a different calculation to the original key.
    if signature is None:
        calculation_white, calculation_camera = white_level, camera_white_levels
    else:
        calculation_white, camera_snapshot = signature.metadata
        calculation_camera = list(camera_snapshot)
    sat = analysis.channel_saturation_levels(channel_ids, calculation_camera, calculation_white)
    ceilings, exact, near, spike = analysis.detect_ceilings(raw_image, raw_colors, channel_ids, sat)
    fullwell, fullwell_ids, note, channel_fullwell = analysis.resolve_fullwell(
        channel_ids, ceilings, spike, sat)
    # Explicit scalar conversion also prevents a custom helper returning NumPy
    # scalar subclasses or mutable mappings from leaking into the cached value.
    scalar_map = lambda mapping: tuple((int(cid), int(value)) for cid, value in mapping.items())
    summary = SensorSummary(
        channel_ids=tuple(channel_ids), ceilings=scalar_map(ceilings),
        exact_counts=scalar_map(exact), near_counts=scalar_map(near),
        spike_ok=tuple((int(cid), bool(value)) for cid, value in spike.items()),
        saturation_levels=scalar_map(sat), fullwell=int(fullwell),
        fullwell_channel_ids=tuple(int(cid) for cid in fullwell_ids), fullwell_note=str(note),
        channel_fullwell=scalar_map(channel_fullwell),
    )
    if signature is not None:
        # Metadata/layout may have been replaced while the reductions ran. A
        # concurrent same-key calculation is harmless; an obsolete key is not.
        current = _cache_signature(raw_image, raw_colors, white_level, camera_white_levels,
                                   evidence=evidence)
        if current is not None and signature.matches(current):
            object.__setattr__(evidence, "_sensor_summary_cache", _CacheEntry(signature, summary))
    return summary
