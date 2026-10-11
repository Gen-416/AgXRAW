# SPDX-License-Identifier: GPL-3.0-or-later
"""Protect known RAW inputs at output preflight and final publication.

Array-only writers cannot discover a RAW's provenance. Their callers must pass
``protected_sources`` when the buffers came from files that must be preserved.
This contract protects declared inputs, not arbitrary files selected as outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import errno
import os
from pathlib import Path


class SourceOutputConflictError(RuntimeError):
    """An output aliases a protected input, or its safety cannot be verified."""


def _resolved(path):
    try:
        return Path(path).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise SourceOutputConflictError(f"无法验证输出与原片的路径关系：{path}") from exc


def _file_identity(path):
    try:
        info = Path(path).stat()
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOTDIR):
            return None
        raise SourceOutputConflictError(f"无法验证输出与原片的文件身份：{path}") from exc
    return info.st_dev, info.st_ino


@dataclass(frozen=True)
class _Source:
    path: Path
    resolved: Path
    identity: tuple[int, int] | None


class ProtectedSources:
    """Retain original path/inode identities, including across later link changes."""

    def __init__(self, *sources):
        records = []

        def add(source):
            if source is None:
                return
            if isinstance(source, ProtectedSources):
                records.extend(source._sources)
            elif isinstance(source, (str, os.PathLike)):
                path = Path(source).absolute()
                records.append(_Source(path, _resolved(path), _file_identity(path)))
            else:
                for item in source:
                    add(item)

        for source in sources:
            add(source)
        self._sources = tuple(dict.fromkeys(records))

    def __bool__(self):
        return bool(self._sources)

    def check(self, *destinations, phase="output-preflight"):
        if not self._sources:
            return self
        for destination in destinations:
            if destination is None:
                continue
            target = Path(destination)
            resolved = _resolved(target)
            identity = _file_identity(target)
            for source in self._sources:
                current = _resolved(source.path)
                current_identity = _file_identity(source.path)
                if (resolved in (source.resolved, current)
                        or (identity is not None and identity in
                            (source.identity, current_identity))):
                    raise SourceOutputConflictError(
                        f"输出目标与输入原片相同或互为别名，禁止覆盖：{target}（{phase}）")
        return self

    def replace(self, candidate, destination):
        self.check(destination, phase="output-commit")
        os.replace(candidate, destination)


def raw_protection(path=None, bundle=None, protected_sources=None):
    """Protect explicit metadata source and actual decoded bundle/evidence sources."""
    evidence = getattr(bundle, "evidence", None)
    return ProtectedSources(protected_sources, path, getattr(bundle, "path", None),
                            getattr(evidence, "path", None))


def final_image_destination(out_path, output_format):
    """Share the established container/suffix policy with preflight callers."""
    path = Path(out_path)
    if output_format in ("sdr-heic", "ultrahdr-heic") and path.suffix.lower() not in (".heic", ".heif"):
        return path.with_suffix(".heic")
    if output_format == "ultrahdr" and path.suffix.lower() in (".heic", ".heif"):
        return path.with_suffix(".jpg")
    return path
