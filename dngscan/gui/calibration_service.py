# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, picker-selected calibration text imports for the localhost GUI."""
from __future__ import annotations

import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .preview_cache import PREVIEW_STORE
from .preview_scheduler import PREVIEW_COORDINATOR


MAX_CALIBRATION_REQUEST_BYTES = 16 * 1024 * 1024
MAX_CALIBRATION_TEXT_BYTES = 8 * 1024 * 1024
MAX_CALIBRATION_FILES = 128
CALIBRATION_SUFFIXES = {".json", ".csv", ".txt"}


def _invalidate_preview() -> None:
    # Entries also carry the persisted calibration fingerprint in their
    # identity: another process importing a profile cannot reuse old analysis.
    # Clearing generations prevents a render already in flight from publishing
    # the old profile after this request has completed.
    PREVIEW_COORDINATOR.clear()
    PREVIEW_STORE.clear_memory()


def list_calibrations(_params: dict[str, Any]) -> dict[str, Any]:
    from dngscan import calibration

    return {"ok": True, "calibrations": calibration.list_calibrations()}


def import_calibration(params: dict[str, Any]) -> dict[str, Any]:
    """Import only files explicitly selected by the browser's file picker.

    The browser cannot submit an arbitrary server-side path. Reconstituting a
    small UTF-8 text package also keeps RAWs and unrelated binary data out of
    the persistent calibration store.
    """
    from dngscan import calibration

    files = params.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_CALIBRATION_FILES:
        raise ValueError(f"请选择 1–{MAX_CALIBRATION_FILES} 个标定 JSON/CSV 文件")
    selected: list[tuple[PurePosixPath, str]] = []
    names: set[str] = set()
    text_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("标定文件条目必须是对象")
        name, content = item.get("name"), item.get("text")
        if not isinstance(name, str) or not isinstance(content, str):
            raise ValueError("标定文件必须带有文件名和 UTF-8 文本")
        relative = PurePosixPath(name)
        if (not name or "\\" in name or "\x00" in name or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or len(relative.parts) > 8 or len(name) > 512
                or relative.suffix.lower() not in CALIBRATION_SUFFIXES):
            raise ValueError("标定文件名无效；仅接受所选目录内的 JSON/CSV/TXT")
        if str(relative) in names:
            raise ValueError("标定包包含重复文件名")
        names.add(str(relative))
        text_bytes += len(content.encode("utf-8"))
        if text_bytes > MAX_CALIBRATION_TEXT_BYTES:
            raise ValueError("标定文本超过 8 MiB，请选择单个标定集")
        selected.append((relative, content))
    # webkitRelativePath includes the selected directory's own basename. The
    # importer wants that directory, rather than the temporary parent around it.
    common_root = selected[0][0].parts[0]
    strip_root = all(len(path.parts) > 1 and path.parts[0] == common_root
                     for path, _ in selected)
    with tempfile.TemporaryDirectory(prefix="dngscan-calibration-") as temporary:
        root = Path(temporary)
        written: list[Path] = []
        for relative, content in selected:
            parts = relative.parts[1:] if strip_root else relative.parts
            target = root.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            written.append(target)
        source = written[0] if len(written) == 1 and written[0].suffix.lower() == ".json" else root
        shutter = params.get("shutterMode")
        if shutter not in (None, "", "any", "electronic", "mechanical", "efcs"):
            raise ValueError("读出模式声明无效")
        summary = calibration.import_calibration(source, shutter_override=shutter or None)
    _invalidate_preview()
    return {"ok": True, "calibration": summary,
            "calibrations": calibration.list_calibrations()}


def remove_calibration(params: dict[str, Any]) -> dict[str, Any]:
    from dngscan import calibration

    profile_id = params.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("缺少标定 ID")
    result = calibration.remove_calibration(profile_id)
    _invalidate_preview()
    return {"ok": True, "result": result,
            "calibrations": calibration.list_calibrations()}


def set_calibration_active(params: dict[str, Any]) -> dict[str, Any]:
    from dngscan import calibration

    profile_id, active = params.get("id"), params.get("active")
    if not isinstance(profile_id, str) or not profile_id or not isinstance(active, bool):
        raise ValueError("标定 ID 或启用状态无效")
    result = calibration.set_calibration_active(profile_id, active)
    _invalidate_preview()
    return {"ok": True, "result": result,
            "calibrations": calibration.list_calibrations()}
