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

R2 item 15 closed the known backlog: the view-brightness gate, the punch
gate family and the sparse-emitter detection thresholds are named at
their consuming sites (tone.py) and registered below. The register stays
a growing inventory — new strategy numbers must land here with names.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import constants as _c

POLICY_VERSION = 5


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
    # R2 item 15: the previously-unregistered backlog, named in tone.py.
    PolicyEntry(
        name="VIEW_BRIGHTNESS_MAX_GAIN",
        value=0.30,
        unit="fraction (display-referred gain)",
        rationale=(
            "maximum dark-scene view-brightness lift; applied only as "
            "dark_body*shadow_quality scales it, never an exposure gain"
        ),
        constrained_by="night-frame corpus scored for subject legibility",
    ),
    PolicyEntry(
        name="VIEW_BRIGHTNESS_DR_LO_EV",
        value=5.5,
        unit="EV plan dynamic range",
        rationale="below this the shadows are too noisy to lift at all",
        constrained_by="night-frame corpus; noise-floor measurements",
    ),
    PolicyEntry(
        name="VIEW_BRIGHTNESS_DR_HI_EV",
        value=8.5,
        unit="EV plan dynamic range",
        rationale="above this the shadow quality supports the full lift",
        constrained_by="night-frame corpus; noise-floor measurements",
    ),
    PolicyEntry(
        name="PUNCH_BODY_LO_EV",
        value=-3.0,
        unit="EV scene body median",
        rationale="below this the scene is dark and auto punch stays off",
        constrained_by="corpus scored for night-scene chroma restraint",
    ),
    PolicyEntry(
        name="PUNCH_BODY_HI_EV",
        value=-1.2,
        unit="EV scene body median",
        rationale="above this the bright-body punch gate is fully open",
        constrained_by="corpus scored for night-scene chroma restraint",
    ),
    PolicyEntry(
        name="PUNCH_QUALITY_DR_LO_EV",
        value=7.5,
        unit="EV plan dynamic range",
        rationale="below this shadow quality withdraws auto punch",
        constrained_by="corpus scored for shadow noise amplification",
    ),
    PolicyEntry(
        name="PUNCH_QUALITY_DR_HI_EV",
        value=9.5,
        unit="EV plan dynamic range",
        rationale="above this the quality punch gate is fully open",
        constrained_by="corpus scored for shadow noise amplification",
    ),
    PolicyEntry(
        name="PUNCH_DR_LO_EV",
        value=6.5,
        unit="EV plan dynamic range",
        rationale="start of the DR bonus window on punch strength",
        constrained_by="corpus A/B on flat vs deep scenes",
    ),
    PolicyEntry(
        name="PUNCH_DR_HI_EV",
        value=8.0,
        unit="EV plan dynamic range",
        rationale="end of the DR bonus window (full bonus)",
        constrained_by="corpus A/B on flat vs deep scenes",
    ),
    PolicyEntry(
        name="BLACK_BELOW_NOISE_FLOOR_EV",
        value=1.5,
        unit="scene EV",
        rationale=(
            "adaptive black endpoint may sit this far below the declared "
            "scene-EV noise floor (noise_floor_ev_estimate); the pre-v5 code "
            "used an analysis-domain DR here and sat 4.5 EV too deep"
        ),
        constrained_by="self-review 2026-08-27; margin inherited from the old -1.5",
    ),
    PolicyEntry(
        name="GATED_BELOW_NOISE_FLOOR_EV",
        value=1.0,
        unit="scene EV",
        rationale=(
            "gated colour path opens only above the noise floor less this "
            "margin; pre-v5 used -DR-1.0 in the wrong domain"
        ),
        constrained_by="self-review 2026-08-27; margin inherited from the old -1.0",
    ),
    PolicyEntry(
        name="PUNCH_BASE_STRENGTH",
        value=0.55,
        unit="fraction of full punch",
        rationale=(
            "strength floor once the gates open; the remaining "
            "1-base rides the DR bonus window"
        ),
        constrained_by="corpus A/B on flat vs deep scenes",
    ),
    PolicyEntry(
        name="SPARSE_EMITTER_TAIL_MAX_PCT",
        value=3.0,
        unit="% of frame above EV 0",
        rationale=(
            "a bright area larger than this reads as a broad region, "
            "not point emitters"
        ),
        constrained_by="night corpus with street lamps vs lit interiors",
    ),
    PolicyEntry(
        name="SPARSE_EMITTER_EXTREMITY_MIN",
        value=0.12,
        unit="fraction (share of tail above +2 EV)",
        rationale=(
            "the small tail must also be extreme before the sparse-"
            "emitter policies (white margin, shoulder start) engage"
        ),
        constrained_by="night corpus with street lamps vs lit interiors",
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
    # v4 (R2 item 15): the register's known backlog closed — view-brightness
    # gate, punch gate family and sparse-emitter detection registered with
    # names at their tone.py consuming sites. No value moved.
    4: "10d56455c4d098e4b53f5a318c23addb9c2e5bfc4701fc68b275afb201014863",
    # v5 (self-review 2026-08-27, P2): the noise-floor domain fix registers
    # BLACK_BELOW_NOISE_FLOOR_EV / GATED_BELOW_NOISE_FLOOR_EV — the black
    # endpoint and the gated floor now derive from the declared scene-EV
    # noise floor instead of a plan-DR or midgray-anchored proxy.
    5: "9f993b07cecf39dc778f2c0da64025db3b02dae4f3aa09a72af2bf55df15f3eb",
}


_BY_NAME = {e.name: e for e in ENTRIES}


def entry(name: str) -> PolicyEntry:
    return _BY_NAME[name]


def policy_line() -> str:
    """One report line: the policy version and entry count, so a reader
    knows which strategy register produced the numbers around it."""
    return f"经验策略常数 v{POLICY_VERSION}（{len(ENTRIES)} 项,dngscan/policy.py）"
