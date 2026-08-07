# SPDX-License-Identifier: GPL-3.0-or-later
"""film v2 plan objects (FILM_PRINT_RENDERING_PLAN §4, §7.2).

Four immutable plans that will be referenced from RenderPlan as the v2
pipeline lands. In P1 every new field defaults to identity: compiling the
defaults describes exactly the v1 behaviour (fixed timing at q(0), measured
development, no analog finish). Field provenance is a declared table — the
JSON report and debug output carry it; the GUI does not repeat it per widget.

Validation implements the per-medium field-validity contract (plan §7.2):
`reversal_direct` admits only fixed timing, zero printer filtration and zero
print exposure; violations FAIL CLOSED — never reset-and-continue, never
silently ignored, never downgraded to fixed.
"""
from __future__ import annotations

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
NEUTRALIZATION_POLICIES = ("bounded", "datasheet")
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


@dataclass(frozen=True)
class FilmPrintPlan:
    medium_id: str = "none"           # Endura / 2383 / reversal_direct / none
    timing_policy: str = "fixed"
    neutralization_policy: str = "bounded"
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
) -> None:
    """Fail-closed contract from plan §5.3 / §7.2. Raises ValueError."""
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
        if print_plan.timing_policy in ("fixed", "retimed") and (
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
    if not 0.0 <= float(finish.compression) <= 1.0:
        raise ValueError("film compression 强度域为 [0,1]")
    if finish.compression > 0.0 and not 0.0 <= float(finish.compression_knee_ev) <= 6.0:
        raise ValueError("film compression knee 域为 [0,6] EV(中灰之上)")
    if not 0.0 <= float(finish.highlight_color_density) <= 2.0:
        raise ValueError("高光色密度 rho 域为 [0,2]")
    if development.recipe_id == "editorial_custom":
        # §6 bounded perturbation: declared editorial domains, hard-rejected
        # beyond (the anchor-preserving construction only holds inside them).
        if not -0.5 <= float(development.contrast_delta) <= 0.5:
            raise ValueError("显影对比扰动域为 [-0.5, 0.5]")
        if not 0.0 <= float(development.fog_delta) <= 0.3:
            raise ValueError("显影 fog 域为 [0, 0.3](fog 只会增加密度)")
        if not -0.5 <= float(development.color_density) <= 0.5:
            raise ValueError("显影色密度扰动域为 [-0.5, 0.5]")
        # The retimed tau table and the bounded neutralization casts are both
        # SOLVED against the measured development; an editorial recipe moves
        # the negative's densities out from under them. Fail closed instead of
        # serving stale solutions as if they still applied.
        if print_plan.timing_policy == "retimed":
            raise ValueError(
                "editorial_custom 显影与 retimed timing 互斥:retimed τ 表按 "
                "measured 显影求解;编辑显影请配 fixed 或 custom timing"
            )
        if print_plan.neutralization_policy == "bounded":
            raise ValueError(
                "editorial_custom 显影与有界灰阶中性化互斥:cast 曲线按 "
                "measured 显影求解;请配 --film-neutralization datasheet"
            )
