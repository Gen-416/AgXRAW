# SPDX-License-Identifier: GPL-3.0-or-later
"""Unified grade picker: chromatic look OR display filter, never both."""
from __future__ import annotations

from .display_filter import DISPLAY_FILTERS, FILTER_CHOICES, filter_available
from .look import LOOK_CHOICES, LOOK_FIELDS

# Legacy pipeline selector kept at its single supported value. It picks the exposure
# anchor family and the Rec.2020 working space, NOT the tone core — the actual core
# (agx / gated / lum / neutral) is chosen per render via --tone-core and lives on the plan.
# Do not read this as "output is always AgX".
RENDER_MODE = "agx"

LOOK_GRADE_PREFIX = "look:"
FILTER_GRADE_PREFIX = "filter:"

_LOOK_LABELS = {
    "none": "无",
    "optic_warm_cyan": "暖肤冷调",
}


def grade_id_for_look(name: str) -> str:
    return f"{LOOK_GRADE_PREFIX}{name}"


def grade_id_for_filter(name: str) -> str:
    return f"{FILTER_GRADE_PREFIX}{name}"


def parse_grade_id(grade: str) -> tuple[str, str]:
    """Return (kind, bare_name) where kind is none | look | filter."""
    if grade == "none":
        return "none", "none"
    if grade.startswith(LOOK_GRADE_PREFIX):
        return "look", grade[len(LOOK_GRADE_PREFIX) :]
    if grade.startswith(FILTER_GRADE_PREFIX):
        return "filter", grade[len(FILTER_GRADE_PREFIX) :]
    in_look = grade in LOOK_FIELDS
    in_filter = grade in DISPLAY_FILTERS
    if in_look and in_filter:
        raise ValueError(
            f"成片风格 ID 冲突：{grade!r} 同时存在色度 Look 与输出滤镜，"
            f"请使用 {grade_id_for_look(grade)!r} 或 {grade_id_for_filter(grade)!r}"
        )
    if in_filter:
        return "filter", grade
    if in_look:
        return "look", grade
    raise ValueError(f"未知成片风格：{grade}")


def grade_label(grade_id: str) -> str:
    if grade_id == "none":
        return "无"
    kind, bare = parse_grade_id(grade_id)
    if kind == "filter":
        return DISPLAY_FILTERS[bare].label
    return _LOOK_LABELS.get(bare, bare.replace("fuji_", "Fujifilm ").replace("_", " "))


def grade_choices() -> tuple[str, ...]:
    looks = tuple(grade_id_for_look(n) for n in LOOK_CHOICES if n != "none")
    filters = tuple(
        grade_id_for_filter(n) for n in FILTER_CHOICES if n != "none" and filter_available(n)
    )
    return ("none",) + looks + filters



def resolve_grade(name: str, strength: float) -> tuple[str, float, str, float]:
    """Map a grade id to (look, look_strength, filter, filter_strength)."""
    s = max(0.0, min(1.5, float(strength)))
    kind, bare = parse_grade_id(name)
    if kind == "none":
        return "none", 0.0, "none", 0.0
    if kind == "filter":
        return "none", 0.0, bare, s
    return bare, s, "none", 0.0


def resolve_grade_id(params: dict) -> tuple[str, float]:
    """The canonical grade id and strength a payload RENDERS with — unified
    ``grade`` or the legacy look/filter pair. Review batch 24: export naming
    and the fingerprint read the raw ``grade`` key, so a legacy payload
    rendered a look under a name and fingerprint that claimed grade=none."""
    grade = params.get("grade")
    if grade is not None:
        gid = str(grade)
        if gid == "none":
            return "none", 1.0
        raw = float(params.get("gradeStrength", params.get("grade_strength", 1.0)))
        # the render clamps (resolve_grade): name what renders, so 2.0 and
        # 1.5 (byte-identical) share a name and -1 is not named as the grade
        return gid, max(0.0, min(1.5, raw))
    look, look_strength, filt, filter_strength = resolve_grade_params(params)
    if filt != "none":
        return grade_id_for_filter(filt), float(filter_strength)
    if look != "none":
        return grade_id_for_look(look), float(look_strength)
    return "none", 1.0


def resolve_grade_params(params: dict) -> tuple[str, float, str, float]:
    """Accept unified `grade` or legacy separate look/filter (mutually exclusive)."""
    grade = params.get("grade")
    if grade is not None:
        strength = float(params.get("gradeStrength", params.get("grade_strength", 1.0)))
        return resolve_grade(str(grade), strength)

    look = str(params.get("look", "none"))
    filt = str(params.get("filter", "none"))
    if look != "none" and filt != "none":
        raise ValueError("色度 Look 与输出滤镜不能同时选用，请只选一种成片风格")
    look_strength = float(params.get("lookStrength", 1.0))
    filter_strength = float(params.get("filterStrength", 1.0))
    if filt != "none":
        return resolve_grade(grade_id_for_filter(filt), filter_strength)
    if look != "none":
        return resolve_grade(grade_id_for_look(look), look_strength)
    return "none", 0.0, "none", 0.0
