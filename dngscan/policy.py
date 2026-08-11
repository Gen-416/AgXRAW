# SPDX-License-Identifier: GPL-3.0-or-later
"""The empirical-policy register (review A8 advisory).

Several numbers in this project are STRATEGY, not standards: they were
chosen from measurements on the frames at hand plus editorial judgement,
and nothing in a datasheet or an ISO document pins them. Scattered across
modules they read like physical constants; collected here each one
declares what it is, why its value was chosen, and what evidence would
revise it.

This register does not MOVE any value — the consuming modules keep their
constants (zero pixel change by construction) and a self-check test pins
the register to the live values, so silent drift on either side fails the
suite. Bump POLICY_VERSION whenever an entry's value or meaning changes,
and record the reason in the entry's ``history``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as _c

POLICY_VERSION = 1


@dataclass(frozen=True)
class PolicyEntry:
    """One empirical policy number and its provenance."""

    name: str                 # the constant's name at its consuming site
    value: float
    unit: str
    rationale: str            # why THIS value
    constrained_by: str       # what corpus/measurement would revise it
    history: tuple[str, ...] = field(default_factory=tuple)


ENTRIES: tuple[PolicyEntry, ...] = (
    PolicyEntry(
        name="MIDGRAY_HEADROOM_STOPS",
        value=float(_c.MIDGRAY_HEADROOM_STOPS),
        unit="stops",
        rationale=(
            "auto-EV places mid grey this many stops under the usable "
            "ceiling; 3.0 keeps typical highlight rolloff inside the tone "
            "shoulder without starving the mids"
        ),
        constrained_by=(
            "a corpus of camera frames spanning DR classes, scored for "
            "highlight retention vs mid exposure at each candidate value"
        ),
    ),
    PolicyEntry(
        name="CEILING_MIN_PILE_PIXELS",
        value=float(_c.CEILING_MIN_PILE_PIXELS),
        unit="pixels",
        rationale=(
            "fewer equal-valued pixels than this is noise or a specular "
            "point, not evidence of sensor saturation"
        ),
        constrained_by=(
            "per-camera dark/flat frames: the smallest genuine clip "
            "extent observed across bodies and ISOs"
        ),
    ),
    PolicyEntry(
        name="CEILING_MIN_PILE_FRACTION",
        value=float(_c.CEILING_MIN_PILE_FRACTION),
        unit="fraction of channel pixels",
        rationale="the pixel-count floor scaled for very large sensors",
        constrained_by="same corpus as CEILING_MIN_PILE_PIXELS",
    ),
    PolicyEntry(
        name="CEILING_PLAUSIBLE_FRACTION",
        value=float(_c.CEILING_PLAUSIBLE_FRACTION),
        unit="fraction of metadata white",
        rationale=(
            "a pile below ~0.42 stop under the declared white level is an "
            "ordinary highlight plateau, not the full well (A8 item 1)"
        ),
        constrained_by=(
            "per-camera measured clip points vs metadata WhiteLevel: the "
            "largest legitimate shortfall observed"
        ),
        history=("v1: introduced by review A8 (was: any pile overrode)",),
    ),
    PolicyEntry(
        name="CEILING_NEAR_WINDOW_SCALE",
        value=8192.0,
        unit="DN per window step (level/scale, floor 2 DN)",
        rationale=(
            "the near-pile window tracks bit depth (~2 DN at 14 bits) "
            "instead of a fixed DN count that means different things at "
            "12 and 16 bits (A8 item 1)"
        ),
        constrained_by="clip-edge histograms across bit depths",
    ),
    PolicyEntry(
        name="CLIP_MARGIN_DN",
        value=4.0,
        unit="DN under resolved full well",
        rationale=(
            "the clip statistic counts pixels within this margin of the "
            "full well; callers pass 4 (the CLI default) so PRNU and "
            "quantisation right at the rail still count as clipped"
        ),
        constrained_by=(
            "rail-noise width measured on saturated flats per camera"
        ),
    ),
    PolicyEntry(
        name="SNR_TILE",
        value=float(_c.SNR_TILE),
        unit="pixels (tile edge)",
        rationale="local-mean tile for the SNR curve estimator",
        constrained_by="estimator bias/variance sweep on synthetic ramps",
    ),
    PolicyEntry(
        name="SNR_LOW_PERCENTILE",
        value=float(_c.SNR_LOW_PERCENTILE),
        unit="percentile",
        rationale=(
            "the shadow anchor of the SNR curve reads this percentile "
            "rather than the minimum to stay off dead pixels"
        ),
        constrained_by="dead-pixel census across bodies",
    ),
    PolicyEntry(
        name="SNR_BRIGHT_UNRELIABLE_STOP",
        value=float(_c.SNR_BRIGHT_UNRELIABLE_STOP),
        unit="stops vs mid grey",
        rationale=(
            "above this the tile statistics mix highlight rolloff into "
            "the noise estimate and the curve is flagged unreliable"
        ),
        constrained_by="tile-variance decomposition on graded exposures",
    ),
    PolicyEntry(
        name="DEFAULT_HDR_HEADROOM_EV",
        value=float(_c.DEFAULT_HDR_HEADROOM_EV),
        unit="EV over reference white",
        rationale=(
            "project default authoring headroom (800 nit at the 100 nit "
            "convention); a delivery choice, not an ISO 21496-1 constant"
        ),
        constrained_by="owner review across display classes",
    ),
    PolicyEntry(
        name="MAX_HDR_PEAK_NITS",
        value=float(_c.MAX_HDR_PEAK_NITS),
        unit="nit",
        rationale="project ceiling for authored peaks, not a format limit",
        constrained_by="owner review; display availability",
    ),
)

_BY_NAME = {e.name: e for e in ENTRIES}


def entry(name: str) -> PolicyEntry:
    return _BY_NAME[name]


def policy_line() -> str:
    """One report line: the policy version and entry count, so a reader
    knows which strategy register produced the numbers around it."""
    return f"经验策略常数 v{POLICY_VERSION}（{len(ENTRIES)} 项,dngscan/policy.py）"
