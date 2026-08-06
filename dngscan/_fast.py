# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional C++ AgX core: import, dispatch policy, and fallback."""

from __future__ import annotations

import logging
import os
from typing import Any

from ._deps import np
from .fast_plan import NATIVE_ABI_VERSION
from .models import ToneCompressionPlan

_LOG = logging.getLogger(__name__)

_extension: Any | None = None
_extension_error: str | None = None


class NativeKernelError(RuntimeError):
    """Raised when DNGSCAN_FAST=1 and the native kernel fails."""


def _load_extension() -> Any | None:
    global _extension, _extension_error
    if _extension is not None:
        return _extension
    if _extension_error is not None:
        return None
    try:
        from . import _dngscan_fast as ext
    except ImportError as exc:
        _extension_error = str(exc)
        return None
    if int(ext.native_abi_version()) != int(NATIVE_ABI_VERSION):
        _extension_error = "native ABI mismatch"
        return None
    if not bool(ext.self_test()):
        _extension_error = "native self_test failed"
        return None
    _extension = ext
    return _extension


def _require_extension() -> Any:
    ext = _load_extension()
    if ext is None:
        raise NativeKernelError(_extension_error or "native extension unavailable")
    return ext


def _fast_mode() -> str:
    raw = os.environ.get("DNGSCAN_FAST", "auto").strip().lower()
    if raw in {"0", "false", "off", "numpy"}:
        return "off"
    if raw in {"1", "true", "on", "cpp", "native"}:
        return "strict"
    return "auto"


def strict_requested() -> bool:
    return _fast_mode() == "strict"


def available() -> bool:
    if _fast_mode() == "off":
        return False
    return _load_extension() is not None


def backend_name() -> str:
    return "cpp" if available() else "numpy"


def supports_agx(plan: ToneCompressionPlan) -> bool:
    if str(getattr(plan, "tone_core", "agx")) != "agx":
        return False
    if not bool(getattr(plan, "use_c1_endpoints", False)):
        return False
    if str(getattr(plan, "film_mode", "observe")) == "full" and str(
        getattr(plan, "curve_preset", "none")
    ) != "none":
        # Film takeover renders through film_develop, not the AgX kernel. In the
        # default "observe" mode a film preset is just curve parameters — the
        # native kernel handles it at full speed.
        return False
    if float(getattr(plan, "color_head_y", 0.0)) > 0.0 or float(
        getattr(plan, "color_head_m", 0.0)
    ) > 0.0:
        # Enlarger colour head: an EV-dependent LMS gain field the kernel
        # does not model. Zero dials keep the native path — and the byte-exact
        # status quo.
        return False
    if _fast_mode() == "off":
        return False
    if _load_extension() is None:
        return False
    try:
        from .fast_plan import compile_agx_plan

        compile_agx_plan(plan)
    except Exception:
        return False
    return True


def can_use_agx(rgb: Any, plan: ToneCompressionPlan) -> bool:
    if not supports_agx(plan):
        return False
    arr = np.asarray(rgb)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return False
    if arr.dtype != np.float32:
        return False
    if not arr.flags["C_CONTIGUOUS"]:
        return False
    if not np.isfinite(arr).all():
        return False
    return True


def compile_agx_plan(plan: ToneCompressionPlan) -> Any:
    from .fast_plan import compile_agx_plan as _compile

    return _compile(plan)


def apply_agx_core_f32(rgb: np.ndarray, plan: Any) -> np.ndarray:
    ext = _require_extension()
    arr = np.ascontiguousarray(rgb, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("rgb must be (N, 3) float32")
    return ext.apply_agx_core_f32(arr, plan)


def supports_hdr_formation(formation_plan: Any) -> bool:
    """Whether the native HDR formation kernel may be dispatched for this plan.

    The kernel covers the full _form_hdr_chunk chain; film_mode="full" with an
    active curve preset swaps in the takeover LUT — a per-pixel transfer the
    kernel does not model — so those plans keep the NumPy path (same exclusion
    precedent as supports_agx).
    """
    if _fast_mode() == "off":
        return False
    if _load_extension() is None:
        return False
    if str(getattr(formation_plan, "film_mode", "observe")) == "full" and str(
        getattr(formation_plan, "curve_preset", "none")
    ) != "none":
        return False
    if float(getattr(formation_plan, "color_head_y", 0.0)) > 0.0 or float(
        getattr(formation_plan, "color_head_m", 0.0)
    ) > 0.0:
        # The colour-head gain field is per-pixel work the HDR kernel does not
        # model either; those plans keep the NumPy path.
        return False
    return True


def compile_hdr_plan(
    hdr_plan: Any,
    formation_plan: Any,
    inset_matrix: Any,
    outset_matrix: Any,
    formation_luma: Any,
    curve_tables: tuple[Any, Any],
    peak: float,
    output_gamut: str,
) -> Any:
    from .fast_plan import compile_hdr_formation_plan as _compile

    return _compile(
        hdr_plan,
        formation_plan,
        inset_matrix,
        outset_matrix,
        formation_luma,
        curve_tables,
        peak,
        output_gamut,
    )


def apply_hdr_formation_f32(rgb: Any, clip_masks: Any | None, plan: Any) -> np.ndarray:
    ext = _require_extension()
    arr = _output_array(rgb, "rgb")
    if clip_masks is None:
        return ext.apply_hdr_formation_f32(arr, None, plan)
    masks = _output_array(clip_masks, "clip_masks")
    if masks.shape != arr.shape:
        raise ValueError("clip_masks must match rgb shape")
    return ext.apply_hdr_formation_f32(arr, masks, plan)


def supports_output_finalizer() -> bool:
    """Whether the independent SDR output kernel may be dispatched."""
    if _fast_mode() == "off":
        return False
    return _load_extension() is not None


def compile_output_plan(output_gamut: str, alpha: float = 0.05) -> Any:
    from .fast_plan import compile_output_plan as _compile

    return _compile(output_gamut, alpha)


def _output_array(value: Any, name: str) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3) float32")
    return arr


def fit_output_gamut_f32(rgb: Any, plan: Any) -> np.ndarray:
    ext = _require_extension()
    return ext.fit_output_gamut_f32(_output_array(rgb, "rgb"), plan)


def _finalize_output_u8_f32(
    function_name: str,
    rgb: Any,
    noise_a: Any,
    noise_b: Any,
    plan: Any,
) -> np.ndarray:
    ext = _require_extension()
    arr = _output_array(rgb, "rgb")
    first = _output_array(noise_a, "noise_a")
    second = _output_array(noise_b, "noise_b")
    if first.shape != arr.shape or second.shape != arr.shape:
        raise ValueError("dither noise must match rgb shape")
    return getattr(ext, function_name)(arr, first, second, plan)


def finalize_rec2020_u8_f32(
    rgb: Any, noise_a: Any, noise_b: Any, plan: Any
) -> np.ndarray:
    return _finalize_output_u8_f32(
        "finalize_rec2020_u8_f32", rgb, noise_a, noise_b, plan
    )


def finalize_output_u8_f32(
    rgb: Any, noise_a: Any, noise_b: Any, plan: Any
) -> np.ndarray:
    return _finalize_output_u8_f32(
        "finalize_output_u8_f32", rgb, noise_a, noise_b, plan
    )


def _finalize_output_u8_noise_f32(
    function_name: str, rgb: Any, noise: Any, plan: Any
) -> np.ndarray:
    ext = _require_extension()
    arr = _output_array(rgb, "rgb")
    combined = _output_array(noise, "noise")
    if combined.shape != arr.shape:
        raise ValueError("dither noise must match rgb shape")
    return getattr(ext, function_name)(arr, combined, plan)


def finalize_rec2020_u8_noise_f32(
    rgb: Any, noise: Any, plan: Any
) -> np.ndarray:
    return _finalize_output_u8_noise_f32(
        "finalize_rec2020_u8_noise_f32", rgb, noise, plan
    )


def finalize_output_u8_noise_f32(
    rgb: Any, noise: Any, plan: Any
) -> np.ndarray:
    return _finalize_output_u8_noise_f32(
        "finalize_output_u8_noise_f32", rgb, noise, plan
    )
