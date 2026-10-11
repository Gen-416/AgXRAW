# SPDX-License-Identifier: GPL-3.0-or-later
"""Core Image RAW scene-linear Rec.2020 decoder with an auditable decode recipe.

Exact pixel correspondence with the corrected LibRaw output has not been established,
even when both interpret the same file opcodes. Sensor-grid masks are therefore not
transported across this handoff. Aggregate RAW measurements may still characterize the
original sensor evidence; they do not measure Core Image's processed output noise.

The decode follows Apple's recipe for reaching linear scene-referred data (WWDC21
"Capture and process ProRAW images"): baselineExposure, shadowBias, boostAmount,
localToneMapAmount and the RAW exposure control at 0, EDR and gamut mapping off, rendered
into extendedLinearITUR_2020. Adjustable look controls are neutralised where the filter
exposes a meaningful neutral value; moire is left at Apple's default and is documented as
unsupported in RAW 9. Zero legacy parameter values do not disable RAW 9's integrated
demosaic/denoise model. Highlight recovery and lens correction are requested
explicitly because they belong to Apple's camera interpretation, not to dngscan's
downstream display transform.

The file-authored baselineExposure is recorded before it is cleared. ``raw_io`` restores
that scalar exactly once through ``scene_scale`` after the decoder handoff, matching the
LibRaw path without baking rendering intent into the decoder pixels. This keeps the handoff
strictly scene-linear while preserving the file's intended relative exposure downstream.

The handoff is signed float16. Extended-linear Rec.2020 legitimately contains negative
components and values above 1.0; quantising it through an unsigned sensor-white buffer
would destroy both before AgX sees them.

Importing this module must not raise when Quartz/PyObjC is absent.
"""
from __future__ import annotations

import struct
import threading
import platform
import hashlib
import json
import os
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from ._deps import np
from .source_metadata import (SourceChangedError, SourceReadTransactionError,
                              SourceVerificationError, cached_source_metadata)
from .constants import (
    COREIMAGE_SCALE_DEFAULT_MODE,
    COREIMAGE_SCALE_MEASURED_RATIO,
    COREIMAGE_VERSION_CHOICES,
)

# Fixed decode-time multipliers only. Per-file ``aligned`` scaling belongs to raw_io,
# where both decoded buffers are available; it intentionally has a unity multiplier here.
# The old empirical 1/1.0293 fit remains only to reproduce earlier A/B renders.
COREIMAGE_SCALE_MODES = {
    "aligned": 1.0,
    "measured": 1.0 / COREIMAGE_SCALE_MEASURED_RATIO,
    "unity": 1.0,
}
COREIMAGE_SCALE_COMPENSATION = COREIMAGE_SCALE_MODES[COREIMAGE_SCALE_DEFAULT_MODE]
COREIMAGE_PREVIEW_LONG_EDGE = 1280
COREIMAGE_EXPORT_MEMORY_LIMIT_MB = 1024
COREIMAGE_RUNTIME_FAILURE_TTL_SECONDS = 60.
COREIMAGE_RECIPE_VERSION = "coreimage-decode-recipe-2"
SCENE_INPUT_POLICY = "required-parameter-readback-v1"
PROJECT_HOT_WB_IMPLEMENTATION = "rec2020-conjugate-camera-matrix-approx-v1"

_CONTEXTS: dict[bool, Any] = {}
_CONTEXT_RECIPES: dict[bool, dict[str, Any]] = {}
_CONTEXT_LOCK = threading.Lock()


class SceneInputContractError(RuntimeError):
    """The filter can be diagnosed but cannot supply the requested scene input."""

    def __init__(self, cfg: dict[str, Any]):
        self.qualification = _scene_input_qualification(cfg)
        self.controls = cfg.get("controls", {})
        self.actual_decoder_token = cfg.get("actual_decoder_token")
        controls = ", ".join(self.qualification["unverified_controls"]) or "unrecorded required state"
        super().__init__(f"CIRAWFilter scene input contract failed for RAW "
                         f"{self.actual_decoder_token!r}: {controls}")


def _scene_input_qualification(cfg: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on unknown required controls; optional RAW9 legacy NR stays inert.

    This is an API-state contract, not a claim of empirical linearity or absence
    of Apple's internal processing. A failed write is safe only when readback
    independently confirms the required target state.
    """
    access = cfg.get("linear_access", {})
    required = list(access.get("required_controls", ()))
    controls = cfg.get("controls", {})
    unverified = []
    for name in required:
        item = controls.get(name, {})
        state = item.get("effective", {}).get("status")
        support = item.get("support", {})
        inert = (state == "not-applicable" and
                 (item.get("applicability") == "documented-inert-raw9" or
                  (support.get("status") == "read" and support.get("value") is False)))
        if state != "parameter-confirmed" and not inert:
            unverified.append(name)
    qualified = (bool(required) and not unverified and
                 access.get("status") == "parameter-state-verified" and
                 isinstance(cfg.get("actual_decoder_token"), str) and
                 bool(cfg["actual_decoder_token"]))
    return {"policy": SCENE_INPUT_POLICY, "status": "qualified" if qualified else "rejected",
            "required_controls": required, "unverified_controls": unverified,
            "unknown_required_control_policy": "reject-candidate",
            "scope": "required-API-state-not-empirical-linearity-or-noise-qualification"}


def decoder_runtime_id() -> str:
    """Return an OS/build fingerprint because Apple can revise RAW models by OS build."""
    try:
        from Foundation import NSProcessInfo  # type: ignore

        value = str(NSProcessInfo.processInfo().operatingSystemVersionString()).strip()
        if value:
            return value
    except Exception:
        pass
    version = platform.mac_ver()[0]
    return f"macOS {version}" if version else platform.platform()


@lru_cache(maxsize=1)
def decoder_hardware_id() -> dict[str, Any]:
    """Non-unique hardware facts; CIRAWFilter does not expose actual ANE scheduling."""
    facts: dict[str, Any] = {"architecture": platform.machine(), "processor": platform.processor(),
                             "execution_device": "not-observable-via-CIRAWFilter"}
    if platform.system() == "Darwin":
        for field, key in (("model", "hw.model"), ("cpu_brand", "machdep.cpu.brand_string")):
            try:
                result = subprocess.run(["/usr/sbin/sysctl", "-n", key], capture_output=True,
                                        text=True, timeout=2, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    facts[field] = result.stdout.strip()
            except (OSError, subprocess.SubprocessError):
                pass
    return facts


def describe_scale_compensation(gain: float) -> str:
    """Describe the gain that was actually applied.

    A fixed note would keep advertising the fitted ratio even under --coreimage-scale
    unity, where nothing was applied; the whole point of offering both is that a reader
    can tell which one produced a given buffer.
    """
    if abs(float(gain) - 1.0) <= 1e-9:
        return "no fixed decode-time correction (native or per-file aligned handoff)"
    return (
        f"legacy empirical fixed multiplier; inverse ratio {1.0 / float(gain):.4f}; "
        "not a per-camera or per-file calibration"
    )


def scale_compensation_for_mode(mode: str) -> float:
    """Resolve a --coreimage-scale mode name to its gain."""
    try:
        return COREIMAGE_SCALE_MODES[mode]
    except KeyError:
        raise ValueError(
            f"unknown coreimage scale mode: {mode}; "
            f"expected one of {tuple(COREIMAGE_SCALE_MODES)}"
        ) from None

COREIMAGE_DECODER_VERSIONS = COREIMAGE_VERSION_CHOICES

# DNG opcode IDs (DNG 1.7 spec, Chapter 6) that make Apple's decoded frame geometrically
# or radiometrically incomparable to LibRaw's. WarpRectilinear is a per-plane radial
# polynomial; a crop alone does not prove that sensor-grid masks align with its output.
# lensCorrectionEnabled requests Apple's camera interpretation. Its getter is not proof
# of an exact opcode execution history or pixel correspondence with another decoder.
DNG_OPCODE_WARP_RECTILINEAR = 1
DNG_OPCODE_WARP_FISHEYE = 2
DNG_OPCODE_FIX_VIGNETTE_RADIAL = 3
DNG_OPCODE_GAIN_MAP = 9
DNG_GEOMETRY_OPCODES = (DNG_OPCODE_WARP_RECTILINEAR, DNG_OPCODE_WARP_FISHEYE)
_DNG_OPCODE_NAMES = {
    DNG_OPCODE_WARP_RECTILINEAR: "WarpRectilinear",
    DNG_OPCODE_WARP_FISHEYE: "WarpFisheye",
    DNG_OPCODE_FIX_VIGNETTE_RADIAL: "FixVignetteRadial",
    DNG_OPCODE_GAIN_MAP: "GainMap",
}


def available() -> bool:
    """True when Quartz exposes CIRAWFilter (macOS + PyObjC). Never raises."""
    try:
        import Quartz  # type: ignore

        return hasattr(Quartz, "CIRAWFilter") and callable(
            getattr(Quartz.CIRAWFilter, "alloc", None)
        )
    except Exception:
        return False


_RUNTIME_AVAILABLE: dict[bool, bool] = {}
_RUNTIME_FAILED_AT: dict[bool, float] = {}


def runtime_available(*, interactive: bool = False, retry: bool = False) -> bool:
    """True when THIS workload's render context can be built and can
    render one pixel, cached per (process, workload).

    A8's first cut probed a plain ``CIContext.context()`` — a second false
    positive (A9 item 3): the actual decode path builds its context with
    ``contextWithOptions_`` via _render_context, and hosts exist where the
    plain constructor succeeds while the optioned one returns None. The
    probe therefore exercises the REAL path — _render_context plus a
    minimal 1x1 bitmap render — and preview (interactive) and export
    capabilities are cached separately because their option sets differ.
    Failures expire after 60 seconds; ``retry=True`` explicitly probes a fresh
    context immediately. Successful probes may be reused for the process.
    Never raises."""
    key = bool(interactive)
    if key in _RUNTIME_AVAILABLE and not retry:
        if _RUNTIME_AVAILABLE[key]:
            return True
        if time.monotonic() - _RUNTIME_FAILED_AT.get(key, float("-inf")) < COREIMAGE_RUNTIME_FAILURE_TTL_SECONDS:
            return False
    if retry or key in _RUNTIME_FAILED_AT:
        with _CONTEXT_LOCK:
            _CONTEXTS.pop(key, None)
            _CONTEXT_RECIPES.pop(key, None)
    ok = False
    if available():
        try:
            import Quartz  # type: ignore
            from Foundation import NSData  # type: ignore

            ctx = _render_context(Quartz, interactive=key)
            # A10 item 2: probe the REAL workload, not a toy one — the
            # decode renders RGBAh into extended linear Rec.2020, and hosts
            # exist where a plain RGBA8/no-colourspace render succeeds
            # while this combination fails (or vice versa).
            color_space = Quartz.CGColorSpaceCreateWithName(
                Quartz.kCGColorSpaceExtendedLinearITUR_2020
            )
            if color_space is None:
                raise RuntimeError("extended linear Rec.2020 unavailable")
            img = Quartz.CIImage.imageWithColor_(
                Quartz.CIColor.colorWithRed_green_blue_(0.5, 0.5, 0.5)
            ).imageByCroppingToRect_(Quartz.CGRectMake(0, 0, 1, 1))
            buf = bytearray(8)   # RGBAh: 4 x float16
            ctx.render_toBitmap_rowBytes_bounds_format_colorSpace_(
                img, buf, 8, Quartz.CGRectMake(0, 0, 1, 1),
                Quartz.kCIFormatRGBAh, color_space,
            )
            ok = True
        except Exception:
            ok = False
    _RUNTIME_AVAILABLE[key] = ok
    if ok:
        _RUNTIME_FAILED_AT.pop(key, None)
    else:
        _RUNTIME_FAILED_AT[key] = time.monotonic()
    return ok


def _require_quartz() -> Any:
    if not available():
        raise RuntimeError(
            "Core Image RAW decoder unavailable: need macOS with PyObjC Quartz "
            "(pyobjc-framework-Quartz) and CIRAWFilter"
        )
    import Quartz  # type: ignore

    return Quartz


def _image_url(path: Path) -> Any:
    from Foundation import NSURL  # type: ignore

    return NSURL.fileURLWithPath_(str(path.resolve()))


def _open_filter(path: Path) -> Any:
    Quartz = _require_quartz()
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    filt = Quartz.CIRAWFilter.alloc().initWithImageURL_(_image_url(path))
    if filt is None:
        raise RuntimeError(f"CIRAWFilter could not open: {path}")
    return filt


def supported_versions(path: Path) -> tuple[str, ...]:
    """Decoder version strings offered for this file (e.g. '9', '8', '7')."""
    filt = _open_filter(path)
    raw = filt.supportedDecoderVersions()
    if not raw:
        return ()
    out: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def probe_raw9_support(path: Path) -> dict[str, Any]:
    """Probe RAW 9 support for one file without rendering any pixels.

    Core Image's per-file ``supportedDecoderVersions`` is the authority here. A static
    camera list would become stale with macOS updates and cannot account for containers
    that expose different decoder versions for the same camera family.
    """
    result: dict[str, Any] = {
        # A12 item 1: this probe answers "which decoder VERSIONS does the
        # file offer", an API-surface question — pinning it to the export
        # context blocked preview-only hosts. Runtime capability belongs to
        # load_raw's per-workload precheck.
        "coreimage_available": available(),
        "raw9_supported": False,
        "versions_offered": (),
        "fallback_version": None,
        "error": None,
    }
    if not result["coreimage_available"]:
        result["error"] = "Core Image / CIRAWFilter is unavailable"
        return result
    try:
        offered = supported_versions(Path(path))
    except Exception as exc:  # a probe must report an unsupported file, not crash the UI
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    majors = {_normalize_version_token(item) for item in offered}
    result["versions_offered"] = offered
    result["raw9_supported"] = "9" in majors
    if not result["raw9_supported"]:
        for major in ("8", "7", "6"):
            if major in majors:
                result["fallback_version"] = major
                break
    return result


def _normalize_version_token(token: str) -> str:
    text = str(token).strip().lower()
    if text.endswith(".dng"):
        text = text[: -len(".dng")]
    return text


def resolve_decoder_version(requested: str, offered: tuple[str, ...]) -> str:
    """Pick a concrete version string from the filter's offered list.

    ``auto`` selects the highest of 9/8/7/6 present (preferring bare tokens over ``*.dng``).
    An explicit unsupported version raises rather than silently downgrading.
    """
    if not offered:
        raise RuntimeError("CIRAWFilter reports no supported decoder versions for this file")
    requested = str(requested).strip().lower()
    # Internal preview/export pinning carries the exact token. Bare public
    # versions retain their preferred-token behavior, but 9.dng must not become
    # 9 after a failed bare-9 render selected this offered alternative.
    if requested.endswith(".dng") and _normalize_version_token(requested) in COREIMAGE_DECODER_VERSIONS:
        exact = next((item for item in offered if item.lower() == requested), None)
        if exact is None:
            raise RuntimeError(f"decoder version {requested!r} is not offered for this file")
        return exact
    if requested not in COREIMAGE_DECODER_VERSIONS:
        raise ValueError(
            f"unknown coreimage version {requested!r}; "
            f"expected one of {COREIMAGE_DECODER_VERSIONS}"
        )

    by_major: dict[str, list[str]] = {}
    for item in offered:
        major = _normalize_version_token(item)
        by_major.setdefault(major, []).append(item)

    def prefer(major: str) -> str | None:
        options = by_major.get(major)
        if not options:
            return None
        bare = [o for o in options if _normalize_version_token(o) == o.lower() or o == major]
        # Prefer '9' over '9.dng' when both exist.
        for cand in (major, f"{major}.dng"):
            for opt in options:
                if opt.lower() == cand:
                    return opt
        return options[0]

    if requested == "auto":
        for major in ("9", "8", "7", "6"):
            chosen = prefer(major)
            if chosen is not None:
                return chosen
        return offered[-1]

    chosen = prefer(requested)
    if chosen is None:
        raise RuntimeError(
            f"decoder version {requested!r} is not supported for this file; "
            f"offered: {', '.join(offered)}"
        )
    return chosen


def _observe_control(filt: Any, selector: str, kind: str = "float") -> dict[str, Any]:
    fn = getattr(filt, selector, None)
    result: dict[str, Any] = {"selector": selector, "status": "absent", "value": None}
    if not callable(fn):
        return result
    try:
        raw = fn()
        if raw is None:
            result["status"] = "no-value"
            return result
        value = bool(raw) if kind == "bool" else str(raw) if kind == "str" else float(raw)
        if kind == "float" and not np.isfinite(value):
            result["status"] = "nonfinite"
            return result
        result.update(status="read", value=value)
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return result


def _configure_control(filt: Any, *, setter: str | None, getter: str, requested: Any,
                       predicate: str | None = None, kind: str = "float",
                       semantics: str, documented_inert: bool = False,
                       force_legacy_parameter: bool = False) -> dict[str, Any]:
    """Keep API support, writes, readback, and actual processing semantics distinct."""
    before = _observe_control(filt, getter, kind)
    support = (_observe_control(filt, predicate, "bool") if predicate else
               {"selector": None, "status": "no-predicate", "value": None})
    supported = support["value"] if support["status"] == "read" else None
    fn = getattr(filt, setter, None) if setter else None
    written: dict[str, Any] = {"selector": setter, "available": callable(fn),
                               "attempted": False, "status": "preserved-default"}
    if requested is not None:
        if not callable(fn):
            written["status"] = "absent"
        elif supported is False and not force_legacy_parameter:
            written["status"] = "skipped-unsupported"
        else:
            written["attempted"] = True
            try:
                fn(requested)
                written["status"] = "applied"
            except Exception as exc:
                written.update(status="error", error=f"{type(exc).__name__}: {exc}")
    after = _observe_control(filt, getter, kind)
    match = (after["status"] == "read" and requested is not None and
             (after["value"] == requested if kind != "float" else
              abs(after["value"] - float(requested)) <= 1e-6 * max(1., abs(float(requested)))))
    if documented_inert:
        applicability, effective = "documented-inert-raw9", "not-applicable"
    elif supported is False:
        applicability, effective = "unsupported-by-instance", "not-applicable"
    else:
        applicability = "supported-by-instance" if supported is True else "not-declared"
        effective = ("preserved-default" if requested is None else "parameter-confirmed" if match
                     else "parameter-mismatch" if after["status"] == "read" else "unconfirmed")
    return {"requested": requested, "support": support, "applicability": applicability,
            "setter": written, "getter": {"selector": getter, "before": before, "after": after},
            "effective": {"status": effective, "value": after["value"], "semantics": semantics,
                          "processing_effect_verified": False}}


def _read_float(filt: Any, getter: str) -> float | None:
    """Read an optional CIRAW amount without making capability probes fatal."""
    fn = getattr(filt, getter, None)
    if not callable(fn):
        return None
    try:
        value = float(fn())
    except (TypeError, ValueError, OverflowError):
        return None
    return value if np.isfinite(value) else None


def preview_scale_factor(
    filt: Any, *, long_edge: int = COREIMAGE_PREVIEW_LONG_EDGE
) -> float:
    """Choose a decode-time scale that is no larger than the displayed proxy."""
    native_size = getattr(filt, "nativeSize", None)
    if not callable(native_size):
        return 0.5
    try:
        size = native_size()
        width = float(size.width)
        height = float(size.height)
        longest = max(width, height)
        if longest <= 0.0:
            return 0.5
        return float(min(1.0, float(max(1, int(long_edge))) / longest))
    except (AttributeError, TypeError, ValueError):
        return 0.5


def configure_linear_filter(
    filt: Any,
    *,
    version: str,
    scale_factor: float,
    exposure: float = 0.0,
    neutral_cct: float | None = None,
) -> dict[str, Any]:
    """Request Apple's linear handoff and audit each control independently.

    Getter readback confirms a parameter, not the physical absence of an internal
    algorithm. RAW 9's joint demosaic/denoise remains active at zero legacy NR values.
    Missing or unconfirmed required controls are reported, not declared verified.
    """
    controls: dict[str, Any] = {}

    def control(name: str, setter: str | None, getter: str, requested: Any, *,
                predicate: str | None = None, kind: str = "float", semantics: str,
                documented_inert: bool = False, force_legacy_parameter: bool = False):
        result = _configure_control(filt, setter=setter, getter=getter, requested=requested,
                                    predicate=predicate, kind=kind, semantics=semantics,
                                    documented_inert=documented_inert,
                                    force_legacy_parameter=force_legacy_parameter)
        controls[name] = result
        return result

    selected = control("decoderVersion", "setDecoderVersion_", "decoderVersion", str(version),
                       kind="str", semantics="exact-offered-token-selection")
    if selected["setter"]["status"] != "applied":
        raise RuntimeError("CIRAWFilter decoderVersion could not be set: " +
                           str(selected["setter"].get("error", selected["setter"]["status"])))
    actual_token = selected["effective"]["value"]
    if actual_token is not None and str(actual_token) != str(version):
        raise RuntimeError(f"CIRAWFilter selected {actual_token!r} instead of {version!r}")
    raw9 = _normalize_version_token(str(actual_token or version)) == "9"

    # WWDC21's code explicitly clears these five display-look controls. Read the
    # decoder-specific authored baseline before clearing; raw_io restores it once.
    control("baselineExposure", "setBaselineExposure_", "baselineExposure", 0.,
            semantics="file-baseline-externalized-after-successful-clear")
    control("boostAmount", "setBoostAmount_", "boostAmount", 0., semantics="disable-default-look-boost")
    control("boostShadowAmount", "setBoostShadowAmount_", "boostShadowAmount", 0.,
            semantics="inactive-when-boost-is-zero")
    control("shadowBias", "setShadowBias_", "shadowBias", 0., semantics="disable-shadow-bias")
    control("exposure", "setExposure_", "exposure", float(exposure), semantics="decoder-exposure-EV")
    control("neutralTemperature", "setNeutralTemperature_", "neutralTemperature", neutral_cct,
            semantics="as-shot-preserved" if neutral_cct is None else "native-fixed-Kelvin-neutral")
    control("neutralTint", "setNeutralTint_", "neutralTint", None if neutral_cct is None else 0.,
            semantics="as-shot-preserved" if neutral_cct is None else "native-fixed-Kelvin-zero-tint")
    control("extendedDynamicRangeAmount", "setExtendedDynamicRangeAmount_", "extendedDynamicRangeAmount", 0.,
            semantics="disable-Apple-display-EDR-transform")
    control("luminanceNoiseReductionAmount", "setLuminanceNoiseReductionAmount_", "luminanceNoiseReductionAmount", 0.,
            predicate="isLuminanceNoiseReductionSupported", semantics="minimum-exposed-luma-NR-parameter-not-internal-NR-off")
    # Keep the historical compatibility write, but RAW 9 documents this property
    # as inert. A zero getter does not mean its CoreML color denoise is disabled.
    control("colorNoiseReductionAmount", "setColorNoiseReductionAmount_", "colorNoiseReductionAmount", 0.,
            predicate="isColorNoiseReductionSupported", documented_inert=raw9, force_legacy_parameter=True,
            semantics="legacy-parameter-zero-not-joint-demosaic-denoise-off" if raw9 else "legacy-color-NR-parameter-zero")
    control("detailAmount", "setDetailAmount_", "detailAmount", 0., predicate="isDetailSupported",
            documented_inert=raw9, semantics="unsupported-in-RAW9" if raw9 else "minimum-detail-parameter")
    control("contrastAmount", "setContrastAmount_", "contrastAmount", 0.,
            predicate="isContrastSupported", semantics="disable-additional-contrast-parameter")
    # Spatial sharpening need not be nonlinear. It is disabled to keep this
    # decoder's additional sharpening policy separate from the project formation.
    control("sharpnessAmount", "setSharpnessAmount_", "sharpnessAmount", 0.,
            predicate="isSharpnessSupported", force_legacy_parameter=True,
            semantics="additional-sharpening-parameter-zero")
    control("moireReductionAmount", None, "moireReductionAmount", None,
            predicate="isMoireReductionSupported", documented_inert=raw9,
            semantics="unsupported-in-RAW9-default-preserved" if raw9 else "decoder-default-preserved")
    control("localToneMapAmount", "setLocalToneMapAmount_", "localToneMapAmount", 0.,
            predicate="isLocalToneMapSupported", semantics="disable-local-display-tone-map")
    control("gamutMappingEnabled", "setGamutMappingEnabled_", "isGamutMappingEnabled", False,
            kind="bool", semantics="disable-output-gamut-map")
    control("highlightRecoveryEnabled", "setHighlightRecoveryEnabled_", "isHighlightRecoveryEnabled", True,
            predicate="isHighlightRecoverySupported", kind="bool", semantics="request-Apple-highlight-reconstruction")
    control("lensCorrectionEnabled", "setLensCorrectionEnabled_", "isLensCorrectionEnabled", True,
            predicate="isLensCorrectionSupported", kind="bool", semantics="request-Apple-lens-interpretation-not-opcode-proof")
    control("scaleFactor", "setScaleFactor_", "scaleFactor", float(scale_factor), semantics="native-decode-scale")
    control("draftModeEnabled", None, "isDraftModeEnabled", None, kind="bool", semantics="decoder-draft-default-preserved")

    required = ("decoderVersion", "baselineExposure", "boostAmount", "shadowBias", "exposure",
                "localToneMapAmount", "gamutMappingEnabled", "extendedDynamicRangeAmount", "scaleFactor")
    if neutral_cct is not None:
        required += ("neutralTemperature", "neutralTint")
    unverified = [name for name in required if controls[name]["effective"]["status"]
                  not in ("parameter-confirmed", "not-applicable")]
    access = {"status": "parameter-state-unverified" if unverified else "parameter-state-verified",
              "required_controls": list(required), "unverified_controls": unverified,
              "scope": "API-parameter-state-only-not-empirical-linearity-or-noise-qualification"}

    def current(name):
        return controls[name]["effective"]["value"]

    baseline_control = controls["baselineExposure"]
    # A setter may change the value and then report an error. The observed zero
    # is authoritative for externalizing the authored baseline exactly once.
    baseline_cleared = baseline_control["effective"]["status"] == "parameter-confirmed"
    color_nr = current("colorNoiseReductionAmount")
    return {
        "version": str(actual_token or version), "actual_decoder_token": actual_token,
        "scale_factor": current("scaleFactor") if current("scaleFactor") is not None else float(scale_factor),
        "color_noise_reduction_amount": color_nr,
        "color_noise_cleared": color_nr is not None and abs(float(color_nr)) <= 1e-6,
        "color_noise_cleared_semantics": "legacy-parameter-zero-only-not-internal-NR-off",
        "sharpness_amount": current("sharpnessAmount"),
        "highlight_recovery": current("highlightRecoveryEnabled"),
        "lens_correction": current("lensCorrectionEnabled"), "shadow_bias": current("shadowBias"),
        "baseline_exposure_authored": baseline_control["getter"]["before"]["value"],
        "baseline_exposure_applied": current("baselineExposure"),
        "baseline_exposure_cleared": bool(baseline_cleared),
        "extended_dynamic_range_amount": current("extendedDynamicRangeAmount"),
        "neutral_temperature": current("neutralTemperature"), "neutral_tint": current("neutralTint"),
        "controls": controls, "linear_access": access,
        "white_balance_implementation": "native-as-shot-preserved" if neutral_cct is None else "native-neutral-temperature-zero-tint",
        "integrated_demosaic_denoise": "RAW9-joint-model-not-disabled" if raw9 else "decoder-internal-processing-not-qualified",
    }


def _render_context(Quartz: Any, *, interactive: bool) -> Any:
    """Reuse one CIContext per workload, following Apple's RAW 9 guidance."""
    with _CONTEXT_LOCK:
        cached = _CONTEXTS.get(bool(interactive))
        if cached is not None:
            return cached
        cache_key = getattr(Quartz, "kCIContextCacheIntermediates", None)
        memory_key = getattr(Quartz, "kCIContextMemoryLimit", None)
        options: dict[Any, Any] = {}
        if cache_key is not None:
            options[cache_key] = bool(interactive)
        if not interactive and memory_key is not None:
            options[memory_key] = COREIMAGE_EXPORT_MEMORY_LIMIT_MB
        ctx = Quartz.CIContext.contextWithOptions_(options)
        # Older Core Image builds can expose a newly added option constant before the
        # selected renderer accepts it. Keep the intended cache policy and retry without
        # the export-only memory hint rather than failing a valid RAW decode.
        memory_retried = ctx is None and memory_key is not None and memory_key in options
        if memory_retried:
            options.pop(memory_key, None)
            ctx = Quartz.CIContext.contextWithOptions_(options)
        if ctx is None:
            raise RuntimeError("CIContext.contextWithOptions_ returned None")
        _CONTEXTS[bool(interactive)] = ctx
        _CONTEXT_RECIPES[bool(interactive)] = {
            "cache_intermediates": bool(interactive) if cache_key in options else None,
            "memory_limit_mb": options.get(memory_key),
            "memory_hint_rejected_then_retried": bool(memory_retried),
            "scope": "CIContext-options-not-proof-of-execution-device",
        }
        return ctx


def _render_linear_rec2020(filt: Any, *, interactive: bool,
                          metadata: dict[str, Any] | None = None) -> np.ndarray:
    """Render signed float16 HxWx3 extended-linear Rec.2020 (direct rect, no flip)."""
    Quartz = _require_quartz()
    image = filt.outputImage()
    if image is None:
        raise RuntimeError("CIRAWFilter.outputImage returned None")
    extent = image.extent()
    width = int(round(float(extent.size.width)))
    height = int(round(float(extent.size.height)))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"CIRAWFilter produced empty extent: {extent}")

    origin_x = float(extent.origin.x)
    origin_y = float(extent.origin.y)
    if metadata is not None:
        # Record the actual CIImage used below. A later outputImage() can return
        # a different graph/extent, so it cannot testify to this bitmap render.
        metadata["native_extent"] = {"status": "read", "xywh": [origin_x, origin_y,
            float(extent.size.width), float(extent.size.height)]}
    ctx = _render_context(Quartz, interactive=interactive)
    color_space = Quartz.CGColorSpaceCreateWithName(Quartz.kCGColorSpaceExtendedLinearITUR_2020)
    if color_space is None:
        raise RuntimeError("kCGColorSpaceExtendedLinearITUR_2020 is unavailable")

    row_bytes = width * 8  # RGBA float16
    buf = bytearray(height * row_bytes)
    bounds = Quartz.CGRectMake(origin_x, origin_y, width, height)
    # Render the decoder's native rectangle without an extra flip. This alone does
    # not establish correspondence with another decoder's sensor/output grid.
    ctx.render_toBitmap_rowBytes_bounds_format_colorSpace_(
        image,
        buf,
        row_bytes,
        bounds,
        Quartz.kCIFormatRGBAh,
        color_space,
    )
    rgba = np.frombuffer(memoryview(buf), dtype=np.float16).reshape(height, width, 4)
    return np.ascontiguousarray(rgba[:, :, :3])


def _resize_rgb(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_h, target_w = shape
    if image.shape[:2] == (target_h, target_w):
        return image
    from PIL import Image

    out = np.empty((target_h, target_w, image.shape[2]), dtype=np.float16)
    for idx in range(image.shape[2]):
        im = Image.fromarray(image[:, :, idx].astype(np.float32, copy=False), mode="F")
        im = im.resize((target_w, target_h), Image.Resampling.BILINEAR)
        out[:, :, idx] = np.asarray(im, dtype=np.float32).astype(np.float16)
    return out


def luma_rec2020(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    return (
        np.float32(0.2627) * arr[:, :, 0]
        + np.float32(0.6780) * arr[:, :, 1]
        + np.float32(0.0593) * arr[:, :, 2]
    )


def _box_downsample(plane: np.ndarray, factor: int = 8) -> np.ndarray:
    h, w = plane.shape
    h2 = max(1, h // factor)
    w2 = max(1, w // factor)
    cropped = plane[: h2 * factor, : w2 * factor]
    if cropped.size == 0:
        return plane[::factor, ::factor].astype(np.float64, copy=False)
    return (
        cropped.reshape(h2, factor, w2, factor)
        .mean(axis=(1, 3))
        .astype(np.float64, copy=False)
    )


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n < 16:
        return 0.0
    x = x[:n]
    y = y[:n]
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-18:
        return 0.0
    return float(x @ y) / denom


def geometry_correlation(coreimage_rgb: np.ndarray, libraw_rgb: np.ndarray) -> float:
    """Downsampled luma correlation after top-left fractional mapping (no flip).

    Extreme-DR frames disagree in the highlight tail (LibRaw u16 clips at 1.0 while
    CIRAW keeps headroom). Correlate inside the shared robust mid-range so the score
    reflects spatial alignment rather than demosaic/highlight policy.
    """
    ci = np.asarray(coreimage_rgb, dtype=np.float32)
    lr_raw = np.asarray(libraw_rgb)
    if lr_raw.ndim == 3 and lr_raw.shape[2] >= 3:
        if np.issubdtype(lr_raw.dtype, np.integer):
            lr = lr_raw.astype(np.float32) / float(np.iinfo(lr_raw.dtype).max)
        else:
            lr = lr_raw.astype(np.float32, copy=False)
        lr_y = luma_rec2020(lr)
    else:
        lr_y = np.asarray(lr_raw, dtype=np.float32)
    ci_y = luma_rec2020(ci) if ci.ndim == 3 else np.asarray(ci, dtype=np.float32)
    from PIL import Image

    mapped = np.asarray(
        Image.fromarray(lr_y.astype(np.float32, copy=False), mode="F").resize(
            (ci_y.shape[1], ci_y.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )
    lo = float(max(np.percentile(ci_y, 5), np.percentile(mapped, 5), 0.0))
    hi = float(min(np.percentile(ci_y, 95), np.percentile(mapped, 95)))
    if hi <= lo + 1e-9:
        hi = lo + 1e-6
    return pearson_corr(
        _box_downsample(np.clip(ci_y, lo, hi)),
        _box_downsample(np.clip(mapped, lo, hi)),
    )


def _opcode_summary_ids(source) -> tuple[int, ...] | None:
    """Read only TIFF entries and opcode headers, preserving the Apple policy.

    This is intentionally independent of the stricter LibRaw opcode executor:
    diagnostics follow all previously visited IFDs and retain unknown opcode IDs.
    An IFD table is at most 65535 * 12 bytes; opcode payloads are skipped entirely.
    """
    source.seek(0, 2)
    length = source.tell()

    def read_at(offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset > length - size:
            raise ValueError("opcode summary range is outside the file")
        source.seek(offset)
        data = source.read(size)
        if len(data) != size:
            raise ValueError("opcode summary source changed or is truncated")
        return data

    if length < 8:
        return None
    head = read_at(0, 8)
    if head[:2] not in (b"II", b"MM"):
        return None
    # Preserve the existing best-effort policy, which does not reject a TIFF
    # header based on magic or opcode flags/version before reporting its IDs.
    order = "<" if head[:2] == b"II" else ">"
    found: list[int] = []
    seen_ifds: set[int] = set()

    def walk(offset: int, depth: int = 0) -> None:
        if depth > 3 or offset in seen_ifds or offset <= 0 or offset + 2 > length:
            return
        seen_ifds.add(offset)
        count = struct.unpack(order + "H", read_at(offset, 2))[0]
        if count > (length - offset - 2) // 12:
            return
        entries = read_at(offset + 2, count * 12)
        for i in range(count):
            entry = offset + 2 + i * 12
            tag, typ, cnt, value = struct.unpack_from(order + "HHII", entries, i * 12)
            if tag in (0xC740, 0xC741, 0xC74E):
                # At most four BYTE/ASCII/UNDEFINED values live in the entry.
                value_off = entry + 8 if cnt <= 4 and typ in (1, 2, 7) else value
                if value_off + 4 > length:
                    continue
                # Keep the old diagnostic bounds: the file, not a required
                # execution recipe or a possibly incomplete tag byte count.
                n_ops = struct.unpack(">I", read_at(value_off, 4))[0]
                cursor = value_off + 4
                for _ in range(min(n_ops, 16)):
                    if cursor + 16 > length:
                        break
                    op_id, _, _, size = struct.unpack(">IIII", read_at(cursor, 16))
                    found.append(int(op_id))
                    cursor += 16 + size
            elif tag == 0x014A and typ in (4, 13):
                if cnt == 1:
                    walk(value, depth + 1)
                else:
                    for k in range(min(cnt, 8)):
                        pos = value + k * 4
                        if pos + 4 <= length:
                            walk(struct.unpack(order + "I", read_at(pos, 4))[0], depth + 1)
        next_pos = offset + 2 + count * 12
        if next_pos + 4 <= length:
            next_ifd = struct.unpack(order + "I", read_at(next_pos, 4))[0]
            if next_ifd:
                walk(next_ifd, depth + 1)

    walk(struct.unpack(order + "I", head[4:8])[0])
    return tuple(sorted(set(found)))


@cached_source_metadata(cacheable=lambda value: value["parsed"])
def read_dng_opcodes(path: Path) -> dict[str, Any]:
    """Report DNG OpcodeList IDs without retaining the complete RAW file.

    Core Image executes these opcodes during decode. This diagnostic parser
    remains best-effort and independent of LibRaw's required-opcode policy.
    """
    result: dict[str, Any] = {"ids": (), "names": (), "geometry": False, "parsed": False}
    try:
        with path.open("rb") as source:
            ids = _opcode_summary_ids(source)
        if ids is not None:
            result.update(
                ids=ids,
                names=tuple(_DNG_OPCODE_NAMES.get(i, f"opcode{i}") for i in ids),
                geometry=any(i in DNG_GEOMETRY_OPCODES for i in ids),
                parsed=True,
            )
    except Exception:  # diagnostics must never break a render
        return result
    return result


def _file_stat_identity(stat: Any) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@cached_source_metadata(cacheable=lambda value: value["status"] == "hashed")
def _recipe_source_identity(path: Path) -> dict[str, Any]:
    """Hash the source bytes; never substitute a path/header digest for source identity."""
    before = None
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            before = _file_stat_identity(os.fstat(source.fileno()))
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
            after = _file_stat_identity(os.fstat(source.fileno()))
        if before != after or after != _file_stat_identity(path.stat()):
            raise SourceChangedError(path, phase="hash", reason="source changed while computing SHA-256")
        return {"status": "hashed", "sha256": digest.hexdigest(), "size_bytes": after[2], "_stat": after}
    except (OSError, ValueError) as exc:
        if before is not None:
            raise SourceVerificationError(path, phase="hash", reason=f"{type(exc).__name__}: {exc}") from exc
        return {"status": "unavailable", "sha256": None, "error": f"{type(exc).__name__}: {exc}"}


def _assert_recipe_source_current(path: Path, source: dict[str, Any], *, phase: str) -> None:
    """Unknown initial identity is distinct from a proven change or lost verification."""
    if source.get("status") in ("changed-during-hash", "changed-during-decode"):
        raise SourceChangedError(path, phase=phase, reason=str(source["status"]))
    recorded = source.get("_stat")
    if recorded is None:
        return
    try:
        current = _file_stat_identity(path.stat())
    except OSError as exc:
        raise SourceVerificationError(path, phase=phase, reason=f"{type(exc).__name__}: {exc}") from exc
    if current != tuple(recorded):
        raise SourceChangedError(path, phase=phase, reason="source changed since its SHA-256 was recorded")


def _build_decode_recipe(*, cfg: dict[str, Any], requested: str, offered: tuple[str, ...],
                         failures: list[str], half_size: bool, scale_compensation: float,
                         native_extent: dict[str, Any], rendered_shape: tuple[int, int],
                         output_shape: tuple[int, int], target_shape: tuple[int, int] | None,
                         source: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    controls = cfg.get("controls", {})
    runtime, hardware = decoder_runtime_id(), dict(decoder_hardware_id())
    white_balance = cfg.get("white_balance_implementation", "unrecorded")
    actual_token = cfg.get("actual_decoder_token")
    configuration_controls = {}
    for name, item in controls.items():
        effective = item["effective"]
        value = effective["value"]
        # As-shot neutral is file evidence. Its implementation/state belongs to
        # configuration identity, its continuously varying per-file value does not.
        if ((name in ("neutralTemperature", "neutralTint") and item["requested"] is None)
                or item["applicability"] == "documented-inert-raw9"):
            value = None
        configuration_controls[name] = {
            "requested": item["requested"], "support_status": item["support"]["status"],
            "supported": item["support"]["value"], "applicability": item["applicability"],
            "setter_status": item["setter"]["status"], "getter_status": item["getter"]["after"]["status"],
            "effective_status": effective["status"], "effective_value": value,
            "semantics": effective["semantics"],
        }
    configuration = {
        "schema": COREIMAGE_RECIPE_VERSION, "actual_decoder_token": actual_token,
        "scene_input_policy": SCENE_INPUT_POLICY,
        "runtime": runtime, "hardware": hardware, "controls": configuration_controls,
        "white_balance_implementation": white_balance,
        "project_hot_wb_implementation": PROJECT_HOT_WB_IMPLEMENTATION,
        "scale_factor": cfg.get("scale_factor"), "scale_compensation": float(scale_compensation),
        "context": {key: context.get(key) for key in ("cache_intermediates", "memory_limit_mb", "scope")},
        "output_dtype": "float16", "render_format": "RGBAh",
        "color_space": "extendedLinearITUR_2020",
        "post_resize": {"kernel": "bilinear" if target_shape is not None else "none",
                        "target_shape": list(target_shape) if target_shape is not None else None},
    }
    # Request selection/fallback history, image content, source hash, native extent,
    # authored BaselineExposure, and as-shot neutral numbers are deliberately absent.
    identity = hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                                         ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    return {
        "schema": COREIMAGE_RECIPE_VERSION, "configuration_id": identity,
        "configuration": configuration, "requested_decoder_token": str(requested),
        "actual_decoder_token": actual_token, "offered_tokens": list(offered),
        "fallback": {"used": bool(failures), "failed_attempts": list(failures)},
        "runtime": runtime, "hardware": hardware, "source": source,
        "controls": controls, "linear_access": cfg.get("linear_access", {"status": "unrecorded"}),
        "scene_input_qualification": _scene_input_qualification(cfg),
        "white_balance_implementation": white_balance,
        "project_hot_wb_implementation": PROJECT_HOT_WB_IMPLEMENTATION,
        "integrated_demosaic_denoise": cfg.get("integrated_demosaic_denoise", "unrecorded"),
        "geometry": {"native_extent": native_extent, "rendered_shape": list(rendered_shape),
                     "output_shape": list(output_shape), "interactive": bool(half_size),
                     "scale_factor": cfg.get("scale_factor")},
        "output": {"dtype": "float16", "render_format": "RGBAh", "color_space": "extendedLinearITUR_2020",
                   "signed_extended_range": True, "fixed_multiplier": float(scale_compensation),
                   "post_resize": configuration["post_resize"]},
        "context": context,
        "camera_interpretation": {
            "highlight_recovery": cfg.get("highlight_recovery"), "lens_correction": cfg.get("lens_correction"),
            "scope": "API-state-not-per-opcode-execution-or-sensor-mask-correspondence-proof"},
        "baseline_exposure": {"authored": cfg.get("baseline_exposure_authored"),
                              "applied": cfg.get("baseline_exposure_applied"),
                              "cleared": cfg.get("baseline_exposure_cleared")},
    }


def decode_scene_rec2020(
    path: Path,
    *,
    half_size: bool,
    version: str = "auto",
    target_shape: tuple[int, int] | None = None,
    exposure: float = 0.0,
    scale_compensation: float = COREIMAGE_SCALE_COMPENSATION,
    neutral_cct: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode to signed float16 HxWx3 linear Rec.2020.

    Subjective CIRAW controls are configured for the documented linear handoff (not
    blindly zeroed). ``scale_compensation`` is only the optional legacy fixed multiplier;
    load_raw performs the separate per-file alignment policy.
    Optional ``target_shape`` resamples after render. ``neutral_cct`` declares a fixed
    white reference in Kelvin through CIRAWFilter's native neutralTemperature/neutralTint
    interface (tint forced to 0 so the neutral sits on the locus); None keeps as-shot.
    """
    path = Path(path)
    offered = supported_versions(path)
    source = dict(_recipe_source_identity(path))
    _assert_recipe_source_current(path, source, phase="decode-start")
    resolved = resolve_decoder_version(version, offered)
    candidates = [resolved]
    if version == "auto":
        for major in (9, 8, 7, 6):
            matches = sorted(v for v in offered if str(v).split(".")[0] == str(major))
            for candidate in matches:
                if candidate not in candidates:
                    candidates.append(candidate)
    failures = []
    for candidate in candidates:
        try:
            _assert_recipe_source_current(path, source, phase="candidate-start")
            # A failed render can leave a filter in an unusable state. Retry each
            # advertised decoder with a fresh object, including all linear controls.
            filt = _open_filter(path)
            scale_factor = preview_scale_factor(filt) if half_size else 1.0
            cfg = configure_linear_filter(
                filt, version=candidate, scale_factor=scale_factor,
                exposure=float(exposure), neutral_cct=neutral_cct,
            )
            _assert_recipe_source_current(path, source, phase="filter-configuration")
            if _scene_input_qualification(cfg)["status"] != "qualified":
                raise SceneInputContractError(cfg)
            render_metadata: dict[str, Any] = {}
            rgb = _render_linear_rec2020(filt, interactive=bool(half_size), metadata=render_metadata)
            _assert_recipe_source_current(path, source, phase="render")
            if (rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size
                    or not np.isfinite(rgb).all()):
                raise RuntimeError("decoder returned an empty, non-finite or invalid RGB buffer")
            break
        except SourceReadTransactionError:
            # A different decoder must never consume evidence from another
            # generation of the source. The whole outer load must fail/restart.
            raise
        except Exception as exc:
            _assert_recipe_source_current(path, source, phase="candidate-failure")
            failures.append(f"RAW {candidate}: {type(exc).__name__}: {exc}")
            if version != "auto":
                raise
    else:
        raise RuntimeError("All offered Apple RAW decoders failed: " + "; ".join(failures))
    if abs(float(scale_compensation) - 1.0) > 1e-12:
        rgb = (rgb.astype(np.float32) * np.float32(scale_compensation)).astype(np.float16)
    extent = (int(rgb.shape[0]), int(rgb.shape[1]))
    if target_shape is not None and tuple(target_shape) != extent:
        rgb = _resize_rgb(rgb, (int(target_shape[0]), int(target_shape[1])))
    _assert_recipe_source_current(path, source, phase="decode-commit")
    if source.pop("_stat", None) is not None:
        source["status"] = "verified-through-decode"
    recipe = _build_decode_recipe(
        cfg=cfg, requested=version, offered=offered, failures=failures, half_size=half_size,
        scale_compensation=scale_compensation,
        native_extent=render_metadata.get("native_extent", {"status": "unavailable", "xywh": None}),
        rendered_shape=extent, output_shape=tuple(rgb.shape[:2]), target_shape=target_shape,
        source=source, context=dict(_CONTEXT_RECIPES.get(bool(half_size), {"status": "unrecorded"})),
    )
    info = {
        "decoder": "coreimage",
        "version_requested": version,
        "version": cfg["version"],
        "versions_offered": offered,
        "fallback_errors": failures,
        "extent": extent,
        "shape": (int(rgb.shape[0]), int(rgb.shape[1])),
        "half_size": bool(half_size),
        "scale_factor": cfg["scale_factor"],
        "scale_compensation": float(scale_compensation),
        "scale_compensation_note": describe_scale_compensation(scale_compensation),
        "color_noise_reduction_amount": cfg["color_noise_reduction_amount"],
        "color_noise_cleared": cfg["color_noise_cleared"],
        "color_noise_cleared_semantics": "legacy-parameter-zero-only-not-internal-NR-off",
        "sharpness_amount": cfg.get("sharpness_amount"),
        "highlight_recovery": cfg.get("highlight_recovery"),
        "lens_correction": cfg.get("lens_correction"),
        "shadow_bias": cfg.get("shadow_bias"),
        "baseline_exposure_authored": cfg.get("baseline_exposure_authored"),
        "baseline_exposure_applied": cfg.get("baseline_exposure_applied"),
        "baseline_exposure_cleared": cfg.get("baseline_exposure_cleared"),
        "extended_dynamic_range_amount": cfg.get(
            "extended_dynamic_range_amount"
        ),
        "exposure": float(exposure),
        "neutral_temperature": cfg.get("neutral_temperature"),
        "neutral_tint": cfg.get("neutral_tint"),
        "decoder_runtime_id": decoder_runtime_id(),
        "recipe": recipe,
    }
    return rgb.astype(np.float16, copy=False), info


def scene_headroom(rgb: np.ndarray, *, percentile: float = 99.995) -> float:
    """Diagnostic scene headroom above diffuse white; does not rescale the handoff."""
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.size == 0:
        return 1.0
    top = float(np.percentile(arr, percentile))
    return float(max(1.0, top))


def scene_float_to_half(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    """Preserve Core Image's signed extended-linear values in a compact handoff."""
    limit = float(np.finfo(np.float16).max)
    source = np.asarray(rgb)
    if source.dtype == np.dtype(np.float16):
        # The decoder already owns a half raster. Every finite half is within
        # this range, so another full RGB float32 conversion and clip are inert.
        # Keep an independent writable owner even for finite/read-only inputs.
        linear = source.copy()
        return np.nan_to_num(linear, copy=False, nan=0.0, posinf=limit, neginf=-limit), 1.0
    linear = np.asarray(rgb, dtype=np.float32)
    finite = np.nan_to_num(linear, nan=0.0, posinf=limit, neginf=-limit)
    return np.clip(finite, -limit, limit).astype(np.float16), 1.0
