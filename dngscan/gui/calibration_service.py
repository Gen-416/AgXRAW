# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded, picker-selected measurement packages for the localhost GUI."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .preview_cache import PREVIEW_STORE
from .preview_scheduler import PREVIEW_COORDINATOR


MAX_CALIBRATION_REQUEST_BYTES = 96 * 1024 * 1024
MAX_CALIBRATION_TEXT_BYTES = 64 * 1024 * 1024
MAX_CALIBRATION_FILES = 512
CALIBRATION_SUFFIXES = {".json", ".csv", ".txt", ".zip"}


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


@contextmanager
def _selected_sources(params: dict[str, Any]):
    """Materialize only selected bytes; archive expansion is bounded by manifest."""
    files = params.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= MAX_CALIBRATION_FILES:
        raise ValueError(f"请选择 1–{MAX_CALIBRATION_FILES} 个标定 ZIP/JSON/CSV 文件")
    selected: list[tuple[PurePosixPath, bytes]] = []
    names: set[str] = set()
    text_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("标定文件条目必须是对象")
        name = item.get("name")
        if not isinstance(name, str):
            raise ValueError("标定文件必须带有文件名")
        relative = PurePosixPath(name)
        if (not name or "\\" in name or "\x00" in name or relative.is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or len(relative.parts) > 8 or len(name) > 512
                or relative.suffix.lower() not in CALIBRATION_SUFFIXES):
            raise ValueError("标定文件名无效；仅接受所选目录内的 ZIP/JSON/CSV/TXT")
        if str(relative) in names:
            raise ValueError("标定包包含重复文件名")
        names.add(str(relative))
        if relative.suffix.lower() == ".zip":
            if not isinstance(item.get("base64"), str) or "text" in item:
                raise ValueError("ZIP 测量包必须使用 base64 二进制传输")
            try:
                content = base64.b64decode(item["base64"], validate=True)
            except (ValueError, base64.binascii.Error) as error:
                raise ValueError("ZIP 测量包编码损坏") from error
        else:
            if not isinstance(item.get("text"), str) or "base64" in item:
                raise ValueError("CSV/JSON 必须使用 UTF-8 文本传输")
            content = item["text"].encode("utf-8")
        text_bytes += len(content)
        if text_bytes > MAX_CALIBRATION_TEXT_BYTES:
            raise ValueError("所选测量包超过 64 MiB，请分批导入；不会截断频谱")
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
            target.write_bytes(content)
            written.append(target)
        # Use the directory for text sets, individual paths for archives. The
        # shared importer owns grouping, so a batch never picks the first dark.
        source = (written if any(p.suffix.lower() == ".zip" for p in written)
                  else written[0] if len(written) == 1 and written[0].suffix.lower() == ".json" else root)
        yield source


def inspect_calibration(params: dict[str, Any]) -> dict[str, Any]:
    from dngscan.collect_manifest import inspect_sources

    with _selected_sources(params) as source:
        manifest = inspect_sources(source)
    return {"ok": True, "manifest": manifest}


def import_calibration(params: dict[str, Any]) -> dict[str, Any]:
    """Reparse inspected bytes; never trust a client-provided capability report."""
    from dngscan import calibration

    with _selected_sources(params) as source:
        shutter = params.get("shutterMode")
        if shutter not in (None, "", "any", "electronic", "mechanical", "efcs"):
            raise ValueError("读出模式声明无效")
        groups = params.get("groupIds")
        if groups is not None and (not isinstance(groups, list)
                                  or not all(isinstance(value, str) for value in groups)):
            raise ValueError("测量分组选择无效")
        result = calibration.import_calibrations(source, shutter_override=shutter or None,
                                                group_ids=groups)
    _invalidate_preview()
    profiles = result["profiles"]
    return {"ok": True, "calibration": profiles[0] if len(profiles) == 1 else None,
            "imported": profiles, "manifest": result["manifest"],
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


def public_library_status(_params: dict[str, Any]) -> dict[str, Any]:
    from dngscan.calibration_registry import library_status
    return {"ok": True, "library": library_status()}


def install_public_library(params: dict[str, Any]) -> dict[str, Any]:
    from dngscan.calibration_registry import install_library
    files = params.get("files")
    if (not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict)
            or not str(files[0].get("name", "")).lower().endswith(".json")):
        raise ValueError("请选择一个版本化公共标定库 JSON")
    with _selected_sources(params) as source:
        result = install_library(source)
    _invalidate_preview()
    return {"ok": True, "library": result}


def select_public_library(params: dict[str, Any]) -> dict[str, Any]:
    from dngscan.calibration_registry import select_version
    result = select_version(params.get("version"))
    _invalidate_preview()
    return {"ok": True, "library": result}
