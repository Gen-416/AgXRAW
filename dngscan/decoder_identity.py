# SPDX-License-Identifier: GPL-3.0-or-later
"""JSON-stable identities for an actual scene decoder and output calibration."""
from __future__ import annotations

import hashlib
import json
import uuid


def mapping_identity(value) -> str:
    """Malformed provenance disables identity reuse instead of stopping rendering."""
    try:
        encoded = json.dumps(value or {}, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        # A shared sentinel would make unrelated malformed recipes cache-equal.
        return "unverified-provenance-" + uuid.uuid4().hex
    return hashlib.sha256(encoded).hexdigest()


def decoder_output_identity(bundle) -> tuple[str, ...]:
    """Bind in-memory plans/pixels to the recipe and current window asset."""
    from .scene_transform import decoder_anchor_fingerprint

    return (str(getattr(bundle, "scene_decoder", "libraw")),
            str(getattr(bundle, "scene_decoder_version", None) or ""),
            str(getattr(bundle, "scene_decoder_runtime", None) or ""),
            mapping_identity(getattr(bundle, "scene_decoder_recipe", {})),
            mapping_identity(getattr(bundle, "scene_window_calibration", {})),
            decoder_anchor_fingerprint())


def bind_project_handoff(recipe: dict, handoff: dict) -> dict:
    """Include project-owned scale/WB/reference strategy in output profile scope.

    Image-dependent exposure, alignment gain and WB numbers remain provenance,
    rather than pretending each image is a reusable decoder configuration.
    """
    import copy
    result = copy.deepcopy(recipe)
    result["project_handoff"] = copy.deepcopy(handoff)
    core_id = result.get("coreimage_configuration_id", result.get("configuration_id"))
    if (not isinstance(core_id, str) or len(core_id) != 64
            or any(c not in "0123456789abcdef" for c in core_id)):
        result["configuration_id"] = None
        return result
    reference = handoff.get("alignment_reference") or {}
    configuration = {
        "schema": "project-coreimage-handoff-1", "coreimage_configuration_id": core_id,
        "scale_policy": handoff.get("scale_policy"),
        "hot_wb_implementation": handoff.get("hot_wb_implementation"),
        "decode_wb": handoff.get("decode_wb"),
        "reference_strategy": "corrected-libraw-green-median-v1",
        "reference_decoder": reference.get("decoder"), "reference_version": reference.get("version"),
        "reference_highlight_mode": reference.get("highlight_mode"),
        "reference_status": reference.get("status"),
    }
    result["coreimage_configuration_id"] = core_id
    result["project_configuration"] = configuration
    result["configuration_id"] = mapping_identity(configuration)
    return result
