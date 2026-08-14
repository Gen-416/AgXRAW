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
and record the reason in the entry's ``history``. A9 item 6: each
version's value set is FINGERPRINTED (POLICY_FINGERPRINTS) — editing a
value and the register together without bumping the version fails the
suite, because the stored fingerprint no longer matches.

Known candidates NOT yet registered (they first need names at their
consuming sites): the view-brightness gate literals (tone.py — 0.30 gain,
5.5/8.5 EV smoothstep), the punch gate family, and the sparse-emitter
detection thresholds in analysis. The register is honest about being a
growing inventory, not a completed one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as _c

POLICY_VERSION = 3


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
            "a legal DNG WhiteLevel is AUTHORITATIVE; a single-frame pile "
            "may only override within this narrow tolerance (~0.074 stop) "
            "— anything further under is a scene plateau, not the full "
            "well. Per-camera saturation calibration is the sanctioned "
            "wider override path (A9 item 2)"
        ),
        constrained_by=(
            "per-camera measured clip points vs metadata WhiteLevel: the "
            "largest legitimate shortfall observed"
        ),
        history=(
            "v1: 0.75, introduced by review A8 (was: any pile overrode)",
            "v2: 0.95 — A9 showed a 13000/16383 plateau still passing 0.75",
        ),
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
        name="TAIL_SNR_WINDOW_EV",
        value=float(_c.TAIL_SNR_WINDOW_EV),
        unit="EV",
        rationale=(
            "width of the brightest still-reliable SNR-curve window (just "
            "below SNR_BRIGHT_UNRELIABLE_STOP) read as the tail SNR for "
            "HDR channel-separation confidence (review R2 item 1)"
        ),
        constrained_by="EDR corpus calibration (pending)",
    ),
    PolicyEntry(
        name="TAIL_SNR_ZERO_DB",
        value=float(_c.TAIL_SNR_ZERO_DB),
        unit="dB",
        rationale=(
            "tail SNR at/below which per-channel highlight freedom is "
            "fully withdrawn (2:1 amplitude — tail chroma is noise)"
        ),
        constrained_by="EDR corpus calibration (pending)",
    ),
    PolicyEntry(
        name="TAIL_SNR_FULL_DB",
        value=float(_c.TAIL_SNR_FULL_DB),
        unit="dB",
        rationale=(
            "tail SNR at/above which the SNR factor withdraws nothing "
            "(10:1 amplitude); linear confidence between the two anchors"
        ),
        constrained_by="EDR corpus calibration (pending)",
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
        name="RHO_BASE",
        value=0.5,
        unit="fraction (channel-separation freedom)",
        rationale="the starting HDR rho before evidence-based withdrawal",
        constrained_by="EDR corpus scored for highlight hue fidelity",
    ),
    PolicyEntry(
        name="MULTICHANNEL_CLIP_ZERO_CONFIDENCE_PCT",
        value=10.0,
        unit="% of frame",
        rationale=(
            "multi-channel CFA clipping at this share is treated as total "
            "loss of hue confidence — deliberately strict first cut"
        ),
        constrained_by="EDR corpus with known clipped-hue ground truth",
    ),
    PolicyEntry(
        name="P3_PRESSURE_ZERO_CONFIDENCE_PCT",
        value=20.0,
        unit="% out-of-P3 among bright pixels",
        rationale="heavy gamut pressure withdraws per-channel freedom",
        constrained_by="same EDR corpus, projector-pullback measurements",
    ),
    PolicyEntry(
        name="UNALIGNED_DECODER_RHO_CAP",
        value=0.25,
        unit="fraction",
        rationale=(
            "non-libraw decoders lack the aligned RAW tail evidence, so "
            "rho is capped rather than trusted"
        ),
        constrained_by="cross-decoder alignment corpus",
    ),
    PolicyEntry(
        name="NORMAL_WHITE_MARGIN_EV",
        value=0.30,
        unit="EV",
        rationale="latitude kept under the chosen HDR white (normal scenes)",
        constrained_by="owner review across display classes",
    ),
    PolicyEntry(
        name="SPARSE_EMITTER_WHITE_MARGIN_EV",
        value=0.50,
        unit="EV",
        rationale="wider margin when highlights ARE the subject",
        constrained_by="owner review on emitter-dominant scenes",
    ),
    PolicyEntry(
        name="NORMAL_MINIMUM_WHITE_EV",
        value=3.00,
        unit="EV over mid grey",
        rationale="the HDR white never drops below this for normal scenes",
        constrained_by="owner review across display classes",
    ),
    PolicyEntry(
        name="SPARSE_EMITTER_MINIMUM_WHITE_EV",
        value=3.50,
        unit="EV over mid grey",
        rationale="sparse-emitter floor sits higher for the same reason",
        constrained_by="owner review on emitter-dominant scenes",
    ),
    PolicyEntry(
        name="MAXIMUM_WHITE_EV",
        value=8.50,
        unit="EV over mid grey",
        rationale="authoring ceiling for the HDR white",
        constrained_by="owner review; display availability",
    ),
    PolicyEntry(
        name="NORMAL_SHOULDER_START_EV",
        value=0.20,
        unit="scene EV over mid grey",
        rationale=(
            "the HDR shoulder leaves the darktable body slightly above the "
            "pivot so ordinary bright subjects keep the body's contrast"
        ),
        constrained_by="corpus latitude calibration",
    ),
    PolicyEntry(
        name="SPARSE_EMITTER_SHOULDER_START_EV",
        value=0.00,
        unit="scene EV over mid grey",
        rationale="emitter highlights are the subject; shoulder starts at the pivot",
        constrained_by="corpus latitude calibration",
    ),
    PolicyEntry(
        name="MAX_HDR_PEAK_NITS",
        value=float(_c.MAX_HDR_PEAK_NITS),
        unit="nit",
        rationale="project ceiling for authored peaks, not a format limit",
        constrained_by="owner review; display availability",
    ),
)

def _fingerprint(entries: tuple[PolicyEntry, ...]) -> str:
    import hashlib

    # A10 item 3 widened the hash to the meaning fields; A11 item 3 makes
    # the encoding COLLISION-FREE: the naive ~/;-joined string let
    # rationale="a~b",constrained_by="c" collide with
    # rationale="a",constrained_by="b~c". Canonical JSON with explicit
    # keys, sorted entries and compact separators cannot smear content
    # across field boundaries.
    import json

    payload = json.dumps(
        [
            {
                "name": e.name,
                "value": repr(e.value),
                "unit": e.unit,
                "rationale": e.rationale,
                "constrained_by": e.constrained_by,
                "history": list(e.history),
            }
            for e in sorted(entries, key=lambda e: e.name)
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# One fingerprint per shipped POLICY_VERSION. Changing any value (or the
# entry set) without bumping the version breaks the match; bumping demands
# a new pinned line here — a conscious, reviewable act.
POLICY_FINGERPRINTS = {
    2: "de4a3ff468320ff60e213ca4895fcc7c2e3f0e657c56d34b857f4f01ea85c418",
    # v3 (R2 item 1): TAIL_SNR_WINDOW_EV / TAIL_SNR_ZERO_DB / TAIL_SNR_FULL_DB
    # registered — the HDR channel-separation tail-SNR factor goes live.
    3: "96253a7904f41ff8da20d937b6066544c60cc0cc7f32756db5a85a154d6f71b8",
}


_BY_NAME = {e.name: e for e in ENTRIES}


def entry(name: str) -> PolicyEntry:
    return _BY_NAME[name]


def policy_line() -> str:
    """One report line: the policy version and entry count, so a reader
    knows which strategy register produced the numbers around it."""
    return f"经验策略常数 v{POLICY_VERSION}（{len(ENTRIES)} 项,dngscan/policy.py）"
