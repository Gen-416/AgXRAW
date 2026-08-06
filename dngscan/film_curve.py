# SPDX-License-Identifier: GPL-3.0-or-later
"""Film curve presets: named coordinates in AgX's parameter space.

A preset pins the complete curve (endpoints included) to values solved offline from a
stock's published characteristic curves (tools/fit_film_curve.py). Scene-adaptive tone
compilation is deliberately bypassed while a preset is active: film's response is fixed
— the same scene always receives the same curve — and that whole-roll consistency is
exactly what the user selected. The EV0 -> 0.18 anchor survives by construction (the
fit target is balanced to 18% at mid-scale and AgX's pivot is immovable), and the
paper-Dmax shadow floor rides in through target_black_linear as a declared, measured
part of the look. User tone adjustments still apply on top, reported as departures
from the named coordinate.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ._deps import np

FILM_CURVE_PRESETS_JSON = Path(__file__).with_name("film_curve_presets.json")


_SCHEMA_ERROR: str | None = None


def _load() -> dict[str, dict[str, Any]]:
    global _SCHEMA_ERROR
    try:
        raw = json.loads(FILM_CURVE_PRESETS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    version = int(raw.get("version", 0) or 0)
    if version != 4:
        # v1 presets were calibrated in linear sRGB; v2 presets baked the
        # display room's viewing flare (IEC 1% for papers, 0.5% projection for
        # slides) into the medium's fitted black — a viewing-environment
        # property masquerading as film character. Schema v3 is medium-native:
        # black is the medium's own Dmax and calibration flare is zero by
        # contract. The refusal is LAZY: importing dngscan (including the fit
        # tool that regenerates this very file) must not die on stale data;
        # any actual film request does, loudly.
        _SCHEMA_ERROR = (
            f"film_curve_presets.json schema v{version} is not supported; "
            "regenerate with tools/fit_film_curve.py (schema v4: honestly named "
            "black policy, dual floors, content-pinned colour heads)"
        )
        return {}
    presets = raw.get("presets", {})
    if not isinstance(presets, dict):
        return {}
    for name, preset in presets.items():
        # Hard v3 contract per preset — a mixed or hand-edited file fails
        # closed instead of silently mixing black policies.
        try:
            policy = str(preset.get("black_policy"))
            exponent = float(preset["surround_translation_exponent"])
            native = float(preset["medium_floor_native_linear"])
            translated = float(preset["medium_floor_linear"])
            head = preset.get("color_head")
            ok = (
                float(preset.get("calibration_viewing_flare", -1.0)) == 0.0
                and abs(
                    float(preset["params"]["target_black_linear"]) - translated
                ) <= 1e-6
                and 0.0 < native <= translated + 1e-9
                and (
                    (policy == "medium-native" and abs(exponent - 1.0) < 1e-9)
                    or (policy == "medium-translated" and 0.0 < exponent < 1.0)
                )
                and (
                    head is None
                    or (
                        str(head.get("format")) == "joint-lms-npz-v4"
                        and len(str(head.get("sha256", ""))) == 64
                    )
                )
            )
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            _SCHEMA_ERROR = (
                f"film preset '{name}' violates the v4 contract (black policy/"
                "translation exponent/dual floors/zero calibration flare/"
                "content-pinned colour head); regenerate with tools/fit_film_curve.py"
            )
            return {}
    return presets


FILM_CURVE_PRESETS: dict[str, dict[str, Any]] = _load()
FILM_CURVE_CHOICES = ("none",) + tuple(FILM_CURVE_PRESETS)


def film_curve_label(name: str) -> str:
    if name == "none":
        return "无"
    preset = FILM_CURVE_PRESETS.get(name)
    return str(preset.get("label", name)) if preset else name


def validate_film_curve(name: str) -> str:
    if name != "none" and _SCHEMA_ERROR:
        raise RuntimeError(_SCHEMA_ERROR)
    if name == "none" or name in FILM_CURVE_PRESETS:
        return name
    raise ValueError(
        f"未知胶片曲线预设：{name}（可选：{'/'.join(FILM_CURVE_CHOICES)}）"
    )


# Editorial style pairings for the observe mode — the declared "look layer" on the
# validated base (the FilmLight-shaped architecture: a stable house DRT plus a
# separate look). Each entry pairs a prefeed separation over-drive with one of
# AgX's own validated primaries geometries. These are EDITORIAL DECLARATIONS by
# stock reputation, not measurements — first-drafted 2026-07-30, deliberately
# conservative, and any explicitly given layer value overrides the pairing (the
# combo rule: nothing is baked). Keys fall back to _DEFAULT_STYLE.
FILM_STYLE_PAIRINGS: dict[str, tuple[float, str]] = {
    # Kodak negatives: gentle portrait separation; Ektar is the vivid outlier.
    "portra160": (1.3, "base"),
    "portra400": (1.3, "base"),
    "portra800": (1.3, "base"),
    "portra800push1": (1.35, "base"),
    "portra800push2": (1.4, "base"),
    "ektar100": (1.4, "punchy"),
    "gold200": (1.4, "base"),
    "ultramax400": (1.4, "base"),
    # Fuji negatives: consumer crispness; Pro 400H's airy softness.
    "superia400": (1.5, "base"),
    "c200": (1.4, "base"),
    "pro400h": (1.3, "muted"),
    # Reversals: Velvia is THE saturated slide; Kodachrome's dense restraint.
    "provia100f": (1.4, "base"),
    "velvia100": (1.6, "punchy"),
    "ektachrome100": (1.4, "base"),
    "kodachrome64": (1.4, "muted"),
    # Cine negatives: flat wide-latitude scan look; theatrical quotes get punch.
    "vision350d": (1.2, "muted"),
    "vision3250d": (1.2, "muted"),
    "vision3200t": (1.2, "muted"),
    "vision3500t": (1.2, "muted"),
    "verita200d": (1.2, "muted"),
    "vision350d_theatrical": (1.4, "punchy"),
    "vision3250d_theatrical": (1.4, "punchy"),
    "vision3200t_theatrical": (1.4, "punchy"),
    "vision3500t_theatrical": (1.4, "punchy"),
    "verita200d_theatrical": (1.4, "punchy"),
}
_DEFAULT_STYLE = (1.3, "base")


def film_style_pairing(name: str) -> tuple[float, str]:
    """(separation strength, agx primaries) declared for a film preset."""
    return FILM_STYLE_PAIRINGS.get(str(name), _DEFAULT_STYLE)


def film_process(name: str) -> str | None:
    """Physical process class of a preset: "negative" (print-through C-41/ECN-2),
    "reversal" (E-6/K-14 slide — its own display medium), or None when unknown.

    Read from the preset's declared classification (tools/fit_film_curve.py derives
    it from whether the stock's profile carries a target print); consumers must not
    guess from the name.
    """
    preset = FILM_CURVE_PRESETS.get(str(name))
    if not isinstance(preset, dict):
        return None
    process = preset.get("process")
    return str(process) if process in ("negative", "reversal") else None


# --- Enlarger colour head -----------------------------------------------------
# Real darkroom units: CC filter density steps. 30 CC = 0.30 optical density on
# the named paper layer's printing exposure (~one stop). Steps of 5 mirror the
# detented dial of a real colour head.
COLOR_HEAD_CC_MAX = 200.0
COLOR_HEAD_CC_STEP = 5.0


def color_head_supported(name: str) -> bool:
    """Whether a preset carries a colour-head field (negative presets only)."""
    preset = FILM_CURVE_PRESETS.get(str(name))
    return isinstance(preset, dict) and isinstance(preset.get("color_head"), dict)


def validate_color_head_cc(value: Any, label: str) -> float:
    """One colour-head dial: 0-200 CC in detents of 5, like the physical dial."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字") from exc
    if not np.isfinite(v) or not 0.0 <= v <= COLOR_HEAD_CC_MAX:
        raise ValueError(f"{label} 需在 0-{COLOR_HEAD_CC_MAX:g} CC 之间")
    if abs(v / COLOR_HEAD_CC_STEP - round(v / COLOR_HEAD_CC_STEP)) > 1e-6:
        raise ValueError(
            f"{label} 按真实色头档位取值：{COLOR_HEAD_CC_STEP:g} CC 步进（收到 {v:g}）"
        )
    return v


_COLOR_HEAD_FIELD_CACHE: dict[str, tuple[Any, Any, dict[str, Any]] | None] = {}


DATA_DIR = Path(__file__).with_name("data")


def color_head_joint_field(name: str):
    """(ev[N], cc_grid[41], gains_lms[41,41,N,3] float32) of a preset, or None.

    Stage-3 joint Y x M paper-exposure field: every real detent solved together
    through the printing chain with one re-time; gains diagonal in Bradford LMS
    (see tools/fit_film_curve.build_joint_color_head_field). Loaded once per
    preset from the packaged npz; float16 storage upcasts to float32 here.
    """
    key = str(name)
    if key in _COLOR_HEAD_FIELD_CACHE:
        return _COLOR_HEAD_FIELD_CACHE[key]
    preset = FILM_CURVE_PRESETS.get(key)
    pointer = preset.get("color_head") if isinstance(preset, dict) else None
    field = None
    if isinstance(pointer, dict) and pointer.get("file"):
        path = DATA_DIR / str(pointer["file"])
        try:
            with np.load(path, allow_pickle=False) as payload:
                ev = np.asarray(payload["ev"], dtype=np.float32)
                cc = np.asarray(payload["cc_grid"], dtype=np.float64)
                gains = np.asarray(payload["gains_lms"], dtype=np.float32)
                # Hard loading contract (review batch 7): schema, basis and
                # value sanity fail CLOSED — a wrong-basis or corrupted field
                # must never be silently applied to pixels.
                schema = int(payload["schema"])
                basis = str(np.asarray(payload["basis"]))
                audit = float(payload["audit_max_stop"])
            if schema != 4:
                raise ValueError(f"colour-head schema {schema}, expected 4")
            expected_sha = str(pointer.get("sha256", ""))
            import hashlib as _hashlib

            actual_sha = _hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                # Content pinning: a v4 preset names its exact asset bytes; a
                # stale or foreign npz must fail here, not misindex at apply.
                raise ValueError(
                    f"colour-head content hash mismatch for {path.name}: "
                    f"preset pins {expected_sha[:12]}..., file is {actual_sha[:12]}..."
                )
            if basis != "bradford-lms":
                raise ValueError(f"colour-head basis {basis!r}, expected bradford-lms")
            if not (0.0 <= audit <= 0.02):
                raise ValueError(f"colour-head audit_max_stop {audit} out of gate")
            if not bool(np.isfinite(gains).all()):
                raise ValueError("colour-head gains contain non-finite values")
            if float(gains.min()) <= 1e-4 or float(gains.max()) >= 1e4:
                raise ValueError("colour-head gains outside sane multiplier range")
            # The runtime indexes detents DIRECTLY as round(cc / step): that
            # assumption (zero-based, uniform, hardware 5 CC step) is validated
            # here, hard — an offset or non-uniform grid would silently index
            # the wrong detent at apply time (review batch 8).
            if cc.ndim != 1 or cc.size < 2:
                raise ValueError("colour-head cc_grid is mis-shaped")
            step = float(cc[1] - cc[0])
            if abs(float(cc[0])) > 1e-6 or abs(step - 5.0) > 1e-6 or \
                    not bool(np.allclose(np.diff(cc), step, atol=1e-6)):
                raise ValueError(
                    "colour-head cc_grid violates the zero-based uniform 5 CC contract"
                )
            if (
                ev.ndim == 1 and ev.size >= 2 and bool(np.all(np.diff(ev) > 0))
                and gains.shape == (cc.size, cc.size, ev.size, 3)
            ):
                ev.setflags(write=False)
                gains.setflags(write=False)
                field = (ev, cc, gains)
        except (OSError, KeyError, ValueError):
            field = None
        if field is None:
            raise RuntimeError(
                f"colour-head field for '{key}' is declared but unreadable at "
                f"{path}; regenerate with tools/fit_film_curve.py"
            )
    _COLOR_HEAD_FIELD_CACHE[key] = field
    return field


def color_head_gain_lms(
    name: str, y_cc: float, m_cc: float
) -> tuple[Any, Any] | None:
    """Joint-field gains for a dialled detent pair: (ev_grid, gains_lms[N,3]).

    Detents index the field DIRECTLY — validate_color_head_cc restricts both
    dials to the 5 CC hardware步进, so there is no CC interpolation and no
    separable composition anywhere; only EV is interpolated at apply time.
    Returns None when the preset has no field or both dials are at zero.
    """
    y_cc, m_cc = float(y_cc), float(m_cc)
    if y_cc <= 0.0 and m_cc <= 0.0:
        return None
    field = color_head_joint_field(str(name))
    if field is None:
        return None
    ev, cc_grid, gains = field
    yi = int(round(y_cc / float(cc_grid[1] - cc_grid[0])))
    mi = int(round(m_cc / float(cc_grid[1] - cc_grid[0])))
    yi = max(0, min(yi, cc_grid.size - 1))
    mi = max(0, min(mi, cc_grid.size - 1))
    return ev, gains[yi, mi]

def apply_film_curve_preset(tone_plan: Any, name: str) -> Any:
    """Replace the scene-compiled curve with the preset's fixed coordinate.

    Curve-shape fields move, and the two PER-IMAGE automatic look inputs are
    neutralized (review batch 9): auto punch_strength and view_brightness are
    scene-adaptive editorial compensations, not film calibration data — left
    live they made the same preset change purity and tone photo by photo
    (measured auto punch 0 to 0.803 across samples). Everything else on the
    tone plan (scene metrics, colour policy inputs, tone_core) is untouched,
    so HDR budgeting still reads the real scene.
    """
    if name == "none":
        return tone_plan
    preset = FILM_CURVE_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"未知胶片曲线预设：{name}")
    p = preset["params"]
    return replace(
        tone_plan,
        black_ev=float(p["black_ev"]),
        white_ev=float(p["white_ev"]),
        dynamic_range_ev=float(p["white_ev"]) - float(p["black_ev"]),
        contrast=float(p["contrast"]),
        toe_power=float(p["toe_power"]),
        shoulder_power=float(p["shoulder_power"]),
        latitude_lo_ev=float(p["latitude_lo_ev"]),
        latitude_hi_ev=float(p["latitude_hi_ev"]),
        toe_start_ev=-float(p["latitude_lo_ev"]),
        shoulder_start_ev=float(p["latitude_hi_ev"]),
        target_black_linear=float(p.get("target_black_linear", 0.0)),
        curve_preset=str(name),
        punch_strength=0.0,
        view_brightness=1.0,
    )
