# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 plan objects (FILM_PRINT_RENDERING_PLAN §4, §7.2).

Four immutable plans carried on RenderPlan.film by the v2 pipeline. Every
field defaults to identity: compiling the defaults describes exactly the
v1 behaviour (fixed timing at q(0), measured development, no analog
finish). Field provenance is a declared table — the JSON report and debug
output carry it; the GUI does not repeat it per widget.

Validation implements the per-medium field-validity contract (plan §7.2):
`reversal_direct` admits only fixed timing, zero printer filtration and zero
print exposure; violations FAIL CLOSED — never reset-and-continue, never
silently ignored, never downgraded to fixed.
"""
from __future__ import annotations

import math

from dataclasses import dataclass

# Declared source class per field (plan §4): measured / modelled / editorial.
FIELD_PROVENANCE: dict[str, str] = {
    "FilmExposurePlan.stock_id": "measured",
    "FilmExposurePlan.exposure_ev": "modelled",
    "FilmExposurePlan.reference_cct": "measured",
    "FilmExposurePlan.observer_asset_id": "modelled",
    "FilmDevelopmentPlan.recipe_id": "measured",
    "FilmDevelopmentPlan.contrast_delta": "editorial",
    "FilmDevelopmentPlan.fog_delta": "editorial",
    "FilmDevelopmentPlan.color_density": "editorial",
    "FilmDevelopmentPlan.interimage_mode": "modelled",
    "FilmDevelopmentPlan.interimage_beta": "modelled",
    "FilmPrintPlan.medium_id": "measured",
    "FilmPrintPlan.timing_policy": "modelled",
    "FilmPrintPlan.neutralization_policy": "modelled",
    "FilmPrintPlan.printer_y_cc": "modelled",
    "FilmPrintPlan.printer_m_cc": "modelled",
    "FilmPrintPlan.print_exposure_ev": "modelled",
    "AnalogFinishPlan.compression": "editorial",
    "AnalogFinishPlan.compression_knee_ev": "editorial",
    "AnalogFinishPlan.highlight_color_density": "editorial",
    "AnalogFinishPlan.grain_profile": "editorial",
    "AnalogFinishPlan.grain_amount": "editorial",
    "AnalogFinishPlan.halation_profile": "editorial",
    "AnalogFinishPlan.halation_amount": "editorial",
    "AnalogFinishPlan.bloom_amount": "editorial",
}

# First-release public film-exposure domain (plan §5.3). Assets may narrow or
# (per stock, with measurement) widen it; callers obey the asset's range.
FILM_EXPOSURE_EV_MIN = -2.0
FILM_EXPOSURE_EV_MAX = 2.0

TIMING_POLICIES = ("fixed", "retimed", "custom")
# Canonical names since appearance P3 (plan §8). "bounded"/"datasheet"
# remain accepted as deprecated aliases for hand-built plans; the compiler
# only ever writes canonical values.
NEUTRALIZATION_POLICIES = (
    "technical-neutral", "print-balanced", "native",
    "bounded", "datasheet",
)
DEVELOPMENT_RECIPES = ("measured_default", "editorial_custom")


@dataclass(frozen=True)
class FilmExposurePlan:
    stock_id: str = "none"
    exposure_ev: float = 0.0          # relative to the stock's nominal EI
    reference_cct: float = 5500.0
    observer_asset_id: str = ""


@dataclass(frozen=True)
class FilmDevelopmentPlan:
    recipe_id: str = "measured_default"
    contrast_delta: float = 0.0
    fog_delta: float = 0.0
    color_density: float = 0.0
    provenance: str = "measured"
    # Mainline A2/A3: the compiled inter-image state. `interimage_mode` is
    # the switch ("declared" applies the stock's modelled beta, "off" is the
    # pure spectral base the oracles certify); `interimage_beta` is the
    # EFFECTIVE value resolved at compile time so the immutable plan is
    # auditable. The DATACLASS default is "off"/0.0 — the only pair that is
    # self-consistent without knowing a stock — and the compiler always
    # writes both fields explicitly; validation then holds declared plans to
    # the stock's declared value exactly.
    interimage_mode: str = "off"
    interimage_beta: float = 0.0


@dataclass(frozen=True)
class FilmPrintPlan:
    medium_id: str = "none"           # Endura / 2383 / reversal_direct / none
    timing_policy: str = "fixed"
    neutralization_policy: str = "technical-neutral"
    printer_y_cc: float = 0.0
    printer_m_cc: float = 0.0
    print_exposure_ev: float = 0.0
    native_range: bool = False


@dataclass(frozen=True)
class AnalogFinishPlan:
    compression: float = 0.0
    compression_knee_ev: float = 0.0
    highlight_color_density: float = 0.0
    grain_profile: str = "off"
    grain_amount: float = 0.0
    halation_profile: str = "off"
    halation_amount: float = 0.0
    bloom_amount: float = 0.0
    seed: int = 0


def is_identity_finish(finish: AnalogFinishPlan) -> bool:
    """The strict fast-path predicate (plan §13): everything off is identity."""
    return (
        finish.compression == 0.0
        and finish.highlight_color_density == 0.0
        and finish.grain_profile == "off"
        and finish.grain_amount == 0.0
        and finish.halation_profile == "off"
        and finish.halation_amount == 0.0
        and finish.bloom_amount == 0.0
    )


def validate_film_plans(
    exposure: FilmExposurePlan,
    development: FilmDevelopmentPlan,
    print_plan: FilmPrintPlan,
    finish: AnalogFinishPlan,
    *,
    exposure_ev_min: float = FILM_EXPOSURE_EV_MIN,
    exposure_ev_max: float = FILM_EXPOSURE_EV_MAX,
    film_mode: str = "full",
) -> None:
    """Fail-closed contract from plan §5.3 / §7.2. Raises ValueError."""
    # A8 item 4: every numeric field must be FINITE before any range check
    # runs — NaN passes every comparison-based bound ("nan < 0" and
    # "nan > 1" are both False), so grain_amount=nan and
    # print_exposure_ev=nan sailed through. One validator serves the CLI,
    # the GUI service and the Python API alike.
    _numeric = (
        ("film_exposure_ev", exposure.exposure_ev),
        ("reference_cct", exposure.reference_cct),
        ("dev_contrast", development.contrast_delta),
        ("dev_fog", development.fog_delta),
        ("dev_density", development.color_density),
        ("printer_y_cc", print_plan.printer_y_cc),
        ("printer_m_cc", print_plan.printer_m_cc),
        ("print_exposure_ev", print_plan.print_exposure_ev),
        ("grain_amount", finish.grain_amount),
        ("halation_amount", finish.halation_amount),
        ("bloom_amount", finish.bloom_amount),
        ("compression", finish.compression),
        ("compression_knee_ev", finish.compression_knee_ev),
        ("highlight_color_density", finish.highlight_color_density),
    )
    for _name, _val in _numeric:
        if not (isinstance(_val, (int, float)) and math.isfinite(float(_val))):
            raise ValueError(f"{_name}={_val!r} 非法（必须是有限数值）")
    # A9 item 5 established the 5CC detent contract; A10 item 5 routes it
    # through the ONE public validator (film_curve.validate_color_head_cc)
    # so a float-noise value like 10.000000000000002 gets the same verdict
    # from every entry point instead of two.
    from .film_curve import validate_color_head_cc

    for _name, _cc in (("printer_y_cc", print_plan.printer_y_cc),
                       ("printer_m_cc", print_plan.printer_m_cc)):
        validate_color_head_cc(_cc, _name)
    if not 2000.0 <= float(exposure.reference_cct) <= 20000.0:
        raise ValueError(
            f"reference_cct={exposure.reference_cct} 域为 [2000, 20000] K"
        )
    if not isinstance(finish.seed, int) or isinstance(finish.seed, bool):
        raise ValueError(f"seed={finish.seed!r} 必须是整数")
    for _pname, _pval in (("grain_profile", finish.grain_profile),
                          ("halation_profile", finish.halation_profile)):
        if _pval not in ("off", "modelled_default"):
            # A9 item 5: an unknown profile with amount=0 previously slid
            # through because only the amount>0 branch checked it.
            raise ValueError(f"{_pname}={_pval!r} 未知(可选 off/modelled_default)")
    if not -8.0 <= float(print_plan.print_exposure_ev) <= 8.0:
        raise ValueError(
            f"print_exposure_ev={print_plan.print_exposure_ev} 域为 [-8, 8]"
        )
    if development.interimage_mode not in ("declared", "off", "custom"):
        raise ValueError(
            f"film_interimage={development.interimage_mode!r} 未知"
            "（可选 declared/off/custom）"
        )
    beta = development.interimage_beta
    if not (isinstance(beta, (int, float)) and math.isfinite(beta)) or beta < 0.0:
        raise ValueError(f"interimage_beta={beta!r} 非法（需有限且 >= 0）")
    if development.interimage_mode == "off" and beta != 0.0:
        raise ValueError("film_interimage=off 时 interimage_beta 必须为 0")
    if development.interimage_mode == "declared":
        from .film_develop import interimage_beta as _declared_beta

        expected = _declared_beta(exposure.stock_id)
        if beta != expected:
            raise ValueError(
                f"interimage_beta={beta} 与 stock '{exposure.stock_id}' 声明值 "
                f"{expected} 不一致"
            )
    # Taste-to-dial (2026-08-14): custom carries the USER's beta — bounded by
    # the compiler's [0, 1.5] domain, checked again here so a hand-built plan
    # cannot smuggle a wider value past the audit surface.
    if development.interimage_mode == "custom" and not 0.0 <= beta <= 1.5:
        raise ValueError(f"interimage_beta={beta} 域为 [0, 1.5]（custom 档）")
    if not (exposure_ev_min <= float(exposure.exposure_ev) <= exposure_ev_max):
        raise ValueError(
            f"film_exposure_ev={exposure.exposure_ev} 超出资产声明域 "
            f"[{exposure_ev_min}, {exposure_ev_max}]；超域值硬拒绝，不静默钳制"
        )
    if development.recipe_id not in DEVELOPMENT_RECIPES:
        raise ValueError(f"未知显影配方：{development.recipe_id}")
    if development.recipe_id == "measured_default" and (
        development.contrast_delta != 0.0
        or development.fog_delta != 0.0
        or development.color_density != 0.0
    ):
        raise ValueError(
            "measured_default 显影配方的参数全部锁定；要调整显影请显式声明 "
            "editorial_custom（报告将如实标注编辑显影配方）"
        )
    if print_plan.timing_policy not in TIMING_POLICIES:
        raise ValueError(f"未知印相 timing：{print_plan.timing_policy}")
    if print_plan.neutralization_policy not in NEUTRALIZATION_POLICIES:
        raise ValueError(f"未知灰阶中性化：{print_plan.neutralization_policy}")
    if print_plan.medium_id == "reversal_direct":
        # §7.2 per-medium validity: no printing stage exists. Explicit failure,
        # never a silent downgrade.
        if print_plan.timing_policy != "fixed":
            raise ValueError(
                "reversal_direct 无印相环节：timing 只能是 fixed"
                f"（收到 {print_plan.timing_policy}）"
            )
        if print_plan.printer_y_cc != 0.0 or print_plan.printer_m_cc != 0.0:
            raise ValueError("reversal_direct 无放大机色头：Y/M CC 必须为 0")
        if print_plan.print_exposure_ev != 0.0:
            raise ValueError("reversal_direct 无印相曝光：print_exposure_ev 必须为 0")
    else:
        # Joint-solve exclusivity is a FULL-chain statement: fixed/retimed
        # print states are solved, so manual CC/exposure cannot ride them.
        # In observe mode the print stage these dials touch is the declared
        # AgX-side colour head (film_curve.py), timing is inert, and the
        # audit tuple merely RECORDS the dials — rejecting them here broke
        # the documented observe colour head outright (showcase regen R5).
        if film_mode == "full" and print_plan.timing_policy in ("fixed", "retimed") and (
            print_plan.printer_y_cc != 0.0
            or print_plan.printer_m_cc != 0.0
            or print_plan.print_exposure_ev != 0.0
        ):
            raise ValueError(
                "fixed/retimed timing 下印相由联合求解决定：Y/M CC 与 "
                "print_exposure_ev 必须为 0；要手动印相请声明 custom"
            )
    if finish.grain_amount < 0.0 or finish.halation_amount < 0.0 or finish.bloom_amount < 0.0:
        raise ValueError("模拟光学强度不能为负")
    if finish.grain_amount > 1.0 or finish.halation_amount > 1.0 or finish.bloom_amount > 1.0:
        raise ValueError("模拟光学强度域为 [0,1]")
    if finish.grain_amount > 0.0 and finish.grain_profile == "off":
        raise ValueError("grain_amount > 0 需要一个 grain profile")
    if finish.halation_amount > 0.0 and finish.halation_profile == "off":
        raise ValueError("halation_amount > 0 需要一个 halation profile")
    if not 0.0 <= float(finish.compression) <= 1.0:
        raise ValueError("film compression 强度域为 [0,1]")
    if finish.compression > 0.0 and not 0.0 <= float(finish.compression_knee_ev) <= 6.0:
        raise ValueError("film compression knee 域为 [0,6] EV(中灰之上)")
    if not 0.0 <= float(finish.highlight_color_density) <= 2.0:
        raise ValueError("高光色密度 rho 域为 [0,2]")
    if development.recipe_id == "editorial_custom":
        # §6 bounded perturbation: declared editorial domains, hard-rejected
        # beyond (the anchor-preserving construction only holds inside them,
        # and the baked shaper domains cover exactly this envelope). The
        # numeric bounds are shared with the runtime guard and the asset
        # builder via film_v2_math — one declaration, three enforcers.
        from .film_v2_math import (
            EDITORIAL_CONTRAST_LIMIT,
            EDITORIAL_DENSITY_LIMIT,
            EDITORIAL_FOG_MAX,
        )

        _c = EDITORIAL_CONTRAST_LIMIT
        _d = EDITORIAL_DENSITY_LIMIT
        if not -_c <= float(development.contrast_delta) <= _c:
            raise ValueError(f"显影对比扰动域为 [-{_c}, {_c}]")
        if not 0.0 <= float(development.fog_delta) <= EDITORIAL_FOG_MAX:
            raise ValueError(
                f"显影 fog 域为 [0, {EDITORIAL_FOG_MAX}](fog 只会增加密度)"
            )
        if not -_d <= float(development.color_density) <= _d:
            raise ValueError(f"显影色密度扰动域为 [-{_d}, {_d}]")
        # The retimed tau table and the bounded neutralization casts are both
        # SOLVED against the measured development; an editorial recipe moves
        # the negative's densities out from under them. Fail closed instead of
        # serving stale solutions as if they still applied.
        if print_plan.timing_policy == "retimed":
            raise ValueError(
                "editorial_custom 显影与 retimed timing 互斥:retimed τ 表按 "
                "measured 显影求解;编辑显影请配 fixed 或 custom timing"
            )
        if print_plan.neutralization_policy in (
            "bounded", "technical-neutral", "print-balanced",
        ):
            raise ValueError(
                "editorial_custom 显影与有界灰阶中性化互斥:cast 曲线按 "
                "measured 显影求解;请配 --film-neutralization native"
            )
