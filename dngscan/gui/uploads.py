# SPDX-License-Identifier: GPL-3.0-or-later
"""Process-local staging for RAW files selected by the browser GUI."""
from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import AsyncIterable
from pathlib import Path
from uuid import uuid4

from starlette.concurrency import run_in_threadpool

from .constants import RAW_EXTS


UPLOAD_MAX_BYTES = 512 * 1024 * 1024
UPLOAD_TOTAL_MAX_BYTES = 8 * 1024 * 1024 * 1024
UPLOAD_MAX_CONCURRENT = 3


def _upload_root_usage(
    upload_root: Path,
    excluded_slots: set[Path] | None = None,
) -> int:
    total = 0
    excluded_slots = excluded_slots or set()
    for candidate in upload_root.rglob("*"):
        try:
            if (
                candidate.parent not in excluded_slots
                and candidate.is_file()
                and candidate.suffix != ".uploading"
            ):
                total += candidate.stat().st_size
        except OSError:
            continue
    return total


def _safe_upload_name(filename: str) -> tuple[str, str]:
    clean_name = Path(filename.replace("\\", "/")).name
    suffix = Path(clean_name).suffix.lower()
    if suffix not in RAW_EXTS:
        allowed = "、".join(sorted(RAW_EXTS))
        raise ValueError(f"不支持的 RAW 文件类型；请选择：{allowed}")

    stem = Path(clean_name).stem
    safe_stem = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in stem
    )
    return safe_stem.strip("._")[:64] or "photo", suffix


class UploadStore:
    """Own the browser GUI's temporary RAW copies for one app instance."""

    def __init__(self, upload_root: Path | None = None) -> None:
        self._temporary_directory = (
            tempfile.TemporaryDirectory(prefix="dngscan-gui-uploads-")
            if upload_root is None
            else None
        )
        self.root = Path(
            self._temporary_directory.name
            if self._temporary_directory is not None
            else upload_root
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._slots = threading.BoundedSemaphore(UPLOAD_MAX_CONCURRENT)
        self._quota_lock = threading.Lock()
        self._active_slots: set[Path] = set()
        self._committed_bytes = _upload_root_usage(self.root)
        self._reserved_bytes = 0

    def _reserve(self, content_length: int) -> None:
        with self._quota_lock:
            # Preserve the old server's recoverable quota behavior: if a user
            # removes a staged copy during the session, the next upload sees
            # the reclaimed space. In-flight ``.uploading`` files are covered
            # by ``_reserved_bytes`` instead of being counted twice.
            self._committed_bytes = _upload_root_usage(
                self.root, self._active_slots
            )
            projected = (
                self._committed_bytes + self._reserved_bytes + content_length
            )
            if projected > UPLOAD_TOTAL_MAX_BYTES:
                raise ValueError(
                    "上传临时目录已达总容量上限；请清理或重启 GUI"
                )
            self._reserved_bytes += content_length

    def _release_reservation(
        self,
        content_length: int,
        *,
        committed: bool,
        slot: Path | None,
    ) -> None:
        with self._quota_lock:
            self._reserved_bytes -= content_length
            if committed:
                self._committed_bytes += content_length
            if slot is not None:
                self._active_slots.discard(slot)

    async def store_stream(
        self,
        filename: str,
        source: AsyncIterable[bytes],
        content_length: int,
    ) -> Path:
        """Stream one request body to a private temporary RAW path."""
        if not self._slots.acquire(blocking=False):
            raise ValueError("并发上传数已达上限，请稍候")

        slot: Path | None = None
        partial: Path | None = None
        reserved = False
        committed = False
        try:
            safe_stem, suffix = _safe_upload_name(filename)
            if content_length <= 0:
                raise ValueError("选择的 RAW 文件为空")
            if content_length > UPLOAD_MAX_BYTES:
                raise ValueError(
                    f"RAW 文件超过单文件上限 "
                    f"{UPLOAD_MAX_BYTES // (1024 * 1024)} MB"
                )
            self._reserve(content_length)
            reserved = True

            slot = self.root / uuid4().hex
            slot.mkdir(parents=True)
            with self._quota_lock:
                self._active_slots.add(slot)
            target = slot / f"{safe_stem}{suffix}"
            partial = slot / f".{safe_stem}.uploading"

            received = 0
            with partial.open("wb") as handle:
                async for chunk in source:
                    if not chunk:
                        continue
                    received += len(chunk)
                    if received > content_length:
                        raise ValueError("RAW 文件传输长度与声明不符")
                    await run_in_threadpool(handle.write, chunk)
            if received != content_length:
                raise ValueError("RAW 文件传输未完成")

            await run_in_threadpool(os.replace, partial, target)
            committed = True
            return target
        except BaseException:
            if partial is not None:
                partial.unlink(missing_ok=True)
            if slot is not None:
                try:
                    slot.rmdir()
                except OSError:
                    pass
            raise
        finally:
            if reserved:
                self._release_reservation(
                    content_length,
                    committed=committed,
                    slot=slot,
                )
            self._slots.release()

    def close(self) -> None:
        """Remove app-owned temporary files at server shutdown."""
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
