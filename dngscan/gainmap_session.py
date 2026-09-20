# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, search-local HEVC donors and verified SDR scalar measurements.

The caller owns the temporary root and one independent, read-only master. No
decoded frame is retained. The winner is pinned until select() replaces it;
ordinary candidate churn can only evict the other entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from pathlib import Path
import tempfile


_METRIC_NAMES = frozenset((
    "base_mean_code_error", "base_p99_code_error", "base_max_code_error",
    "base_channel_bias_code_error", "base_block_p99_code_error",
    "coding_luma_rmse", "coding_chroma_rmse", "coding_local_luma_p99",
))


def _primary_identity(path: Path) -> bytes | None:
    """Identify a standalone primary, excluding item IDs and unrelated payloads.

    Preserve property order and essential flags, even where a decoder might ignore
    them. Unsupported/derived layouts miss the cache instead of bypassing readback.
    """
    from .heif_gainmap import _boxes, _parse

    try:
        top, children, primary, infos, refs, props, assocs, payloads = _parse(path.read_bytes())
        if any(kind not in (b"hdlr", b"pitm", b"iinf", b"iref", b"iprp", b"iloc", b"idat",
                            b"dinf", b"grpl")
               for kind, _ in children):
            return None
        tone_maps = {item for item, value in infos.items() if value[8:12] == b"tmap"}
        for kind, value in children:
            if kind == b"dinf":
                # ImageIO's self-contained data reference; external/unknown forms miss.
                if list(_boxes(value)) != [(b"dref", b"\0\0\0\0\0\0\0\1"
                                            b"\0\0\0\x0curl \0\0\0\1")]:
                    return None
            if kind == b"grpl":
                for group_kind, group in _boxes(value):
                    if group_kind != b"altr" or len(group) != 20 or group[:4] != bytes(4):
                        return None
                    count = int.from_bytes(group[8:12], "big")
                    targets = {int.from_bytes(group[i:i + 4], "big") for i in (12, 16)}
                    if count != 2 or primary not in targets or len(targets & tone_maps) != 1:
                        return None
        info = infos[primary]
        if info[8:12] != b"hvc1" or any(source == primary for _, source, _ in refs):
            return None
        associations = assocs.get(primary, [])
        kinds = [props[index - 1][0] for _, index in associations]
        if not payloads[primary] or kinds.count(b"hvcC") != 1 or kinds.count(b"ispe") != 1:
            return None
        auxiliary_roots, metadata = [], set()
        for kind, source, targets in refs:
            if primary not in targets:
                continue
            source_type = infos[source][8:12]
            if kind == b"cdsc" and source_type in (b"Exif", b"mime", b"uri "):
                metadata.add(source)
                continue
            if kind == b"dimg" and source_type == b"tmap":
                if (len(targets) != 2 or targets[0] != primary or targets[1] == primary
                        or infos[targets[1]][8:12] not in (b"hvc1", b"grid")):
                    return None
                continue
            if kind == b"auxl" and source_type in (b"hvc1", b"grid"):
                if not set(targets).issubset(tone_maps | {primary}):
                    return None
                auxiliary_roots.append(source)
                continue
            return None

        item_digests = {}

        def item_digest(item):
            if item in item_digests:
                return item_digests[item]
            digest = sha256()

            def add(data):
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)

            add(infos[item][:4] + infos[item][6:])  # Exclude only the relocatable item ID.
            add(payloads[item])
            for essential, index in assocs.get(item, []):
                prop, value = props[index - 1]
                add(bytes((bool(essential),)) + prop)
                add(value)
            item_digests[item] = digest.digest()
            return item_digests[item]

        dependencies = set()

        def node_digest(item, ancestors):
            item_info = infos[item]
            if item in ancestors or item_info[8:12] not in (b"hvc1", b"grid"):
                raise ValueError("unsupported primary dependency")
            dependencies.add(item)
            digest = sha256(item_digest(item))
            descendants = []
            for relation, source, targets in refs:
                if source != item:
                    continue
                if relation == b"auxl" and item in auxiliary_roots:
                    if primary not in targets or not set(targets).issubset(tone_maps | {primary}):
                        raise ValueError("unknown auxiliary relationship")
                    continue
                if relation != b"dimg" or item_info[8:12] != b"grid":
                    raise ValueError("unknown primary dependency reference")
                descendants.append(targets)
            if len(descendants) != (1 if item_info[8:12] == b"grid" else 0):
                raise ValueError("unknown primary dependency grid")
            for targets in descendants:
                for target in targets:
                    digest.update(node_digest(target, ancestors | {item}))
            return digest.digest()

        digest = sha256(node_digest(primary, set()))
        # Alpha/depth/legacy auxiliary images may affect ordinary SDR decoding.
        # Include their complete dependency trees, not only their auxC label.
        for auxiliary in auxiliary_roots:
            digest.update(node_digest(auxiliary, set()))
        for kind, source, targets in refs:
            if not (set(targets) & (dependencies - {primary})):
                continue
            if kind == b"dimg" and source in dependencies and infos[source][8:12] == b"grid":
                continue
            if kind == b"cdsc" and infos[source][8:12] in (b"Exif", b"mime", b"uri "):
                metadata.add(source)
                continue
            return None
        for item in sorted(metadata):
            digest.update(item_digest(item))
            for kind, source, targets in refs:
                if source != item:
                    continue
                if kind != b"cdsc" or not targets or not set(targets).issubset(dependencies | tone_maps):
                    return None
                digest.update(kind)
                digest.update(len(targets).to_bytes(8, "big"))
                for target in targets:
                    digest.update(item_digest(target))
        # ISO parameters are fixed during auxiliary-quality search. Retain their
        # own item semantics while excluding the separately compressed gain image.
        for item in sorted(tone_maps):
            digest.update(item_digest(item))
        # These searches occur before capture metadata is carried. Changes to
        # file-level interpretation or metadata are therefore conservative misses.
        for kind, value in top + children:
            if kind in (b"ftyp", b"hdlr", b"grpl"):
                digest.update(kind)
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)
        return digest.digest()
    except (OSError, ValueError, KeyError, IndexError, TypeError, RecursionError):
        return None


@dataclass
class _Entry:
    path: Path
    info: dict
    identity: bytes | None = None
    metrics: dict | None = None


class PrimarySearchSession:
    """One Display P3/x265 search; at most a winner and a current donor on disk.

    Call primary() before metrics()/remember_metrics() for each candidate. Call
    select() only when the search changes its winner. The supplied root must be a
    search-owned temporary directory and is cleaned up by its caller.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._base = None
        self._entries = {}
        self._current = None
        self._winner = None

    @staticmethod
    def _key(profile):
        return (int(profile.quality), str(profile.chroma), int(profile.heif_bit_depth),
                str(profile.heif_preset), str(profile.heif_tune))

    def _prune(self):
        keep = {self._winner, self._current}
        for key in list(self._entries):
            if key not in keep:
                self._entries[key].path.unlink(missing_ok=True)
                del self._entries[key]

    def primary(self, base, profile) -> tuple[Path, dict]:
        """Encode or reuse the current immutable master's requested HEVC primary."""
        from . import heif_encoder

        if base.flags.writeable:
            raise ValueError("primary search requires a read-only master")
        if self._base is not None and base is not self._base:
            raise ValueError("primary search cannot change its master")
        if profile.container != "heic" or profile.heif_encoder == "apple":
            raise ValueError("primary search requires a libheif/x265 HEIC profile")
        self._base = base
        key = self._key(profile)
        self._current = key
        self._prune()
        if key in self._entries:
            entry = self._entries[key]
            return entry.path, dict(entry.info)
        # Delete the old unselected candidate before allocating another donor.
        # A failed encode never changes or overwrites the pinned winner.
        with tempfile.NamedTemporaryFile(prefix="primary-", suffix=".heic", dir=self.root,
                                         delete=False) as file:
            path = Path(file.name)
        try:
            info = heif_encoder.encode(
                base, path, profile.quality, profile.chroma,
                bit_depth=profile.heif_bit_depth, preset=profile.heif_preset,
                tune=profile.heif_tune, output_gamut="p3",
            )
            self._entries[key] = _Entry(path, dict(info))
        except BaseException:
            path.unlink(missing_ok=True)
            self._current = None
            raise
        return path, dict(info)

    def select(self, info) -> None:
        """Pin a known winner; incomplete/foreign selection metadata is a no-op."""
        try:
            key = (int(info["delivery_quality"]), str(info["delivery_chroma_requested"]),
                   int(info["bit_depth"]), str(info["preset"]), str(info["tune"]))
        except (KeyError, TypeError, ValueError):
            return
        if key not in self._entries:
            return
        self._winner = key
        self._prune()

    def metrics(self, path: Path) -> dict | None:
        """Return a copy only when this final file has the measured primary."""
        entry = self._entries.get(self._current)
        if entry is None or entry.metrics is None or entry.identity is None:
            return None
        if _primary_identity(Path(path)) != entry.identity:
            return None
        return dict(entry.metrics)

    def remember_metrics(self, path: Path, metrics) -> None:
        """Remember SDR scalars for the current donor, never frames or HDR gates."""
        entry = self._entries.get(self._current)
        if entry is None:
            return
        values = {}
        for name, value in metrics.items():
            if name not in _METRIC_NAMES or not isinstance(value, Real):
                raise ValueError("primary metrics must contain only SDR numeric scalars")
            values[name] = float(value)
        identity = _primary_identity(Path(path))
        if identity is None:
            return
        entry.identity, entry.metrics = identity, values
