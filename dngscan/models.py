# SPDX-License-Identifier: GPL-3.0-or-later
"""Core datatypes passed through the dngscan pipeline."""
from __future__ import annotations

import math

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DEFAULT_HDR_PEAK_NITS, HDR_REFERENCE_WHITE_NITS


@dataclass(frozen=True)
class RawEvidence:
    """Decoder-independent sensor evidence acquired through LibRaw.

    Scene decoders consume this contract but never choose or mutate its provider.
    Keeping the evidence handle explicit prevents an Apple/LibRaw scene switch from
    changing the mosaic, levels, CFA identity, or provider provenance used by analysis.
    """

    path: Path
    raw_image: Any
    raw_colors: Any
    white_level: int
    black_levels: list[float]
    camera_wb: list[float]
    daylight_wb: list[float] | None
    color_desc: str
    raw_pattern: list[list[int]]
    camera_white_levels: list[float]
    orientation_flip: int
    xyz_to_cam: Any | None
    provider: str = "libraw"
    provider_version: str | None = None
    # LibRaw's decode matrix ``rgb_cam`` (camera -> linear sRGB, 3x4 with the second
    # green in the fourth column).  Captured beside ``xyz_to_cam`` because some bodies
    # (e.g. Sigma fp DNGs) ship an all-zero Adobe-table ``rgb_xyz_matrix`` while LibRaw
    # still decodes through a fully valid ``rgb_cam`` built from the DNG ColorMatrix
    # tags.  This is what the fixed reconstruction actually applied, so the hot-WB
    # stage can fall back to it without guessing.
    color_matrix: Any | None = None


@dataclass
class RawBundle:
    path: Path
    raw_image: Any
    raw_colors: Any
    xyz_render: Any
    render_scale: float
    scene_rec2020_render: Any
    scene_scale: float
    white_level: int
    black_levels: list[float]
    camera_wb: list[float]
    color_desc: str
    raw_pattern: list[list[int]]
    camera_white_levels: list[float]
    scene_highlight_mode: str = "clip"
    orientation_flip: int = 0
    # Migration field: intent exposure as fixed_midgray × user_ev. Prefer
    # SceneScaleContract / with_intent_exposure(); do not mutate in place across
    # concurrent preview/export work.
    exposure_gain: float = 1.0
    wb_mode: str = "camera"
    daylight_wb: list[float] | None = None
    shot_make: str | None = None
    shot_model: str | None = None
    shot_iso: int | None = None
    # DNG BaselineExposure as written by the camera, or None when the file omits it. This
    # is file-authored baseline rendering compensation, not shutter/aperture/ISO or an
    # auto-gray target. Both decoders honour it before dngscan's explicit EV adjustment.
    baseline_exposure: float | None = None
    # Whether BaselineExposure remains multiplied into decoder pixels. New Core Image
    # handoffs clear it inside CIRAWFilter and restore it through scene_scale, so this is
    # normally False for both decoders. Kept explicit to detect fallback API behaviour.
    baseline_exposure_baked_in: bool = False
    # Declared lens conversion filter (Wratten), applied to the scene-linear render at
    # float conversion. Capture optics, not a look: the reliable tail, HDR budget and
    # every downstream stage see the world through the glass, exactly as film would.
    lens_filter: str = "none"
    # Which DNG dark-field correction the LibRaw decode applied (gainmap/vignette/None).
    lens_shading: str | None = None
    # The WB multipliers actually applied to this decode: camera metadata for as-shot,
    # daylight metadata for the daylight anchor, or the solved fixed-Kelvin multipliers.
    # Prefeed window transport reads this so calibrated chromaticity anchors follow the
    # balance the pixels really received. None means "fall back to camera_wb".
    applied_wb: list[float] | None = None
    # Non-None when a declared WB could not be fully realised for this body (missing
    # colour calibration): the render stays usable, the report must carry this note.
    wb_degradation: str | None = None
    # Consolidated per-body data-support marker (raw_io.camera_data_support_note):
    # None = fully supported; otherwise a truthful label that rendering proceeds but
    # this model lacks the data to guarantee accuracy. Never a gate.
    camera_data_support: str | None = None
    # Half-resolution, orientation-correct RGB soft clip masks in raw/CFA space.
    # Shape is (H, W, 3), aligned to scene_rec2020_render when scene_half_size=True.
    # Full-resolution renders resize this mask to the render buffer on demand.
    clip_masks: Any | None = None
    # Per-channel endpoints used to build clip_masks. None means the initial metadata
    # white levels; analysis records observed full wells here when it rebuilds the mask.
    _clip_mask_fullwell: dict[int, int] | None = None
    # Lazily filled by retreat.clip_masks_for_shape when a render resizes masks.
    _clip_masks_cache_shape: tuple[int, int] | None = None
    _clip_masks_resized: Any | None = None
    # Optional RAW-gated guidance maps (headroom, clip class, SNR confidence).
    raw_guidance: Any | None = None
    # Lazily resized RAW guidance for the current render geometry.
    _raw_guidance_cache_shape: tuple[int, int] | None = None
    _raw_guidance_resized: Any | None = None
    _raw_guidance_has_sensor_snr: bool = False
    # R4 F5: declared as a field (like the sensor-SNR flag) so
    # dataclasses.replace copies preserve it — an ad-hoc attribute was lost on
    # every bundle stage transition, re-running full-frame guidance builds.
    _raw_guidance_has_resolved_fullwell: bool = False
    # R2 item 20: FULL-RESOLUTION tone-plan sample rows (storage RGB units,
    # the exact `flat[::step]` stride the exporter takes) and the identically
    # strided clip-mask rows, attached to cache-proxy bundles at entry build.
    # Tone-plan compilation prefers these when present, so a preview plan's
    # endpoints are compiled from the SAME statistics the export sees instead
    # of the proxy's downsampled pixels. None on full bundles (which sample
    # their own scene) and on pre-v14 cache entries.
    _tone_plan_sample: Any | None = None
    _tone_plan_sample_masks: Any | None = None
    # Scene-linear RGB producer. Mosaic and levels are always LibRaw-derived; per-pixel
    # clip_masks are LibRaw-derived on the libraw path and absent (None) on the coreimage
    # one, whose frame geometry cannot carry them.
    scene_decoder: str = "libraw"
    scene_decoder_version: str | None = None
    # OS/build identity matters for system-distributed RAW decoders even when the public
    # decoder version token is unchanged.
    scene_decoder_runtime: str | None = None
    # Which Core Image scale policy produced the buffer: aligned (per-file decoded-green
    # comparison → calibration_confidence=relative), unity (Apple-native /
    # decoder-native), or measured (legacy fixed Sigma-fp fit, also relative).
    scene_scale_mode: str | None = None
    # Per-file scalar used only by the aligned policy. It compares two decoded green
    # medians; it is not an absolute sensor calibration or content-adaptive auto exposure.
    scene_align_factor: float = 1.0
    # Why aligned mode fell back to identity. None is expected for unity/measured modes.
    scene_align_error: str | None = None
    # DNG opcodes the decoder executed (Core Image path only). Reported, not acted on:
    # their presence is why that path cannot share LibRaw's per-pixel CFA evidence.
    scene_opcode_names: tuple[str, ...] = ()
    # Shape of clip_masks / LibRaw scene frame when scene_decoder != "libraw".
    evidence_shape: tuple[int, int] | None = None
    # Crop of the evidence frame covering the current scene, in evidence pixel coords
    # (y0, x0, y1, x1). None means the full evidence frame maps onto the scene (pure scale).
    scene_geometry_crop: tuple[float, float, float, float] | None = None
    scene_geometry_corr: float | None = None
    # Explicit evidence-layer contract. The flattened fields above remain as a
    # compatibility facade for analysis/render callers while they migrate.
    evidence: RawEvidence | None = None
    evidence_provider: str = "libraw"
    evidence_provider_version: str | None = None
    # Camera ColorMatrix (XYZ -> camera channels) used by the project-owned hot-WB
    # stage.  The decoder always reconstructs with the fixed as-shot preconditioner;
    # this matrix lets later WB choices recover/reapply camera-channel gains without
    # reopening the RAW or asking the decoder to demosaic again.  Kept separately from
    # ``evidence`` because compact preview-cache entries deliberately discard the large
    # RawEvidence payload while retaining this tiny calibration fact.
    wb_xyz_to_cam: Any | None = None
    # Multipliers baked into the one fixed reconstruction.  User WB is expressed as a
    # relative camera-channel transform from this immutable base.
    decode_wb: list[float] | None = None
    # LibRaw's decode matrix ``rgb_cam`` (RawEvidence.color_matrix), retained beside
    # ``wb_xyz_to_cam`` for the hot-WB C0 ladder: when the Adobe/DNG evidence matrix is
    # missing or all-zero, this is the matrix the fixed decode really used.  Kept on the
    # bundle (like ``wb_xyz_to_cam``) because compact preview-cache entries discard the
    # large RawEvidence payload but must still rebalance.
    wb_color_matrix: Any | None = None


@dataclass
class RawGuidanceMaps:
    """Per-pixel RAW permission rasters aligned to clip_masks resolution."""

    headroom: Any
    clip_class: Any
    snr_confidence: Any
    # WB/EV-independent RAW loss permission compiled once from headroom + clip class.
    # Float32 is intentional: reducing this to float16 changes the gated blend.
    raw_permission: Any | None = None


@dataclass
class Analysis:
    channel_ids: list[int]
    labels: dict[int, str]
    ceilings: dict[int, int]
    ceil_spike_counts: dict[int, int]
    ceil_near_counts: dict[int, int]
    ceil_spike_ok: dict[int, bool]
    fullwell_channel_ids: list[int]
    fullwell_note: str
    saturation_levels: dict[int, int]
    channel_fullwell: dict[int, int]
    channel_thresholds: dict[int, int]
    fullwell: int
    threshold: int
    clip_pct: dict[int, float]
    cfa_cell_supported: bool
    cell_union_pct: float
    cell_ge2_of_clipped_pct: float
    cell_k_of_clipped_pct: dict[int, float]
    cell_k_of_all_pct: dict[int, float]
    ev_p1: float
    ev_raw_p1: float
    ev_median: float
    ev_p99: float
    ev_p999: float
    ev_dr_p1_p999: float
    ev_floor_hit_pct: float
    median_vs_gray_ev: float
    median_y: float
    noise_floor: float
    usable_dr_ev: float
    snr_curves: dict[str, dict[str, Any]]
    snr1_dr: dict[str, float]
    snr1_stop: dict[str, float]
    gamut_out_pct: dict[str, float]
    bright_pixel_pct: float
    survivor_channel: str
    container_bits_est: int
    prior_id: str | None = None
    gain_e_per_dn: float | None = None
    noise_floor_e: float | None = None
    prior_read_noise_e: float | None = None
    prior_pdr_ev: float | None = None
    usable_dr_eff_ev: float = float("nan")
    health_lag1_corr: float = float("nan")
    health_hist_empty_pct: float = float("nan")


@dataclass
class ToneCompressionPlan:
    target_gamut: str
    luma_p1: float
    luma_p50: float
    luma_p99: float
    luma_p999: float
    black_ev: float
    white_ev: float
    dynamic_range_ev: float
    contrast: float
    toe_power: float
    shoulder_power: float
    chroma_p95: float
    negative_rgb_pct: float
    over_rgb_pct: float
    # Linear latitude around the pivot (EV): shoulder starts latitude_hi_ev above mid
    # gray instead of at it, keeping bright subject colors out of the channel-converging
    # shoulder; a small lower run keeps upper shadows off the toe. Zero = pure sigmoid.
    latitude_lo_ev: float = 0.0
    latitude_hi_ev: float = 0.0
    # Scene-driven purity compensation applied after the AgX curve (see dngscan/punch.py).
    # 0 = identity (night/high-ISO scenes gate to exactly zero).
    punch_strength: float = 0.0
    # Tone core selector: agx (full geometry), gated (RAW-permission mix), lum
    # (luminance-ratio C1), or neutral (fixed diagnostic Y-ratio curve).
    tone_core: str = "agx"
    # Norm for the luminance core: "y", "power", or "max".
    lum_norm: str = "y"
    # Optional manual pivot offset. The automatic compiler keeps this at zero until a
    # constrained C1 solver can move local contrast without moving the EV=0 anchor.
    pivot_ev_offset: float = 0.0
    # Fraction of the pre-curve hue restored after the curve. darktable semantics:
    # 0 keeps the per-channel curve shift; 1 fully restores the recorded input hue.
    hue_restore: float = 0.6
    # Linear output floor of the curve; >0 lifts blacks for faded film looks.
    target_black_linear: float = 0.0
    # Curve endpoint in display-linear units (darktable target_white): <1 fades an SDR
    # white, 1 is reference white, and >1 requests extended white.
    target_white_linear: float = 1.0
    # Internal AgX curve encoding. SDR keeps darktable's historical 2.2, and so does
    # the v2 HDR compiler: it pins this gamma and reaches extended white through its
    # own shoulder solve above the knee (hdr_agx_math), never by raising the encoding.
    curve_gamma: float = 2.2
    # AgX primaries preset (base/punchy/muted/smooth); pinned darktable uses base.
    agx_primaries: str = "base"
    # The endpoint-normalized C1 DRT keeps the calibrated scene EV=0 pivot fixed while
    # re-scaling only its black/white bounds. These values share that scene-relative EV
    # domain; `shoulder_start_ev` is the requested linear latitude above the pivot.
    toe_start_ev: float = -4.0
    shoulder_start_ev: float = 1.0
    use_c1_endpoints: bool = True
    # Named film curve coordinate this plan's curve fields were pinned to, or "none"
    # for scene-adaptive compilation. Informational: consumers must read the curve
    # fields themselves, never re-derive behaviour from the name.
    curve_preset: str = "none"
    # Mainline A2: whether the stock's declared modelled inter-image beta
    # applies in full mode. "declared" (default) or "off" (the pure spectral
    # base — what the oracle gates certify). Compiled into
    # FilmDevelopmentPlan.interimage_beta; fail-closed on unknown values.
    # Appearance layer (FILM_APPEARANCE_RECIPE_PLAN P1): "technical" is the
    # strict fast path (no asset touched, frozen bytes unchanged);
    # "reference" resolves the stock x medium recipe at compile, fail-closed.
    film_appearance: str = "technical"
    film_appearance_strength: float = 1.0
    # E2: which recipe interpretation (reference/extended) was selected.
    film_appearance_variant: str = "reference"
    # The COMPILED FilmAppearancePlan (A3 doctrine: the runtime consumes the
    # compiled object, never re-resolves from disk or a registry). None on
    # hand-built plans; a hand-built plan claiming "reference" without it
    # fails closed at runtime.
    film_appearance_compiled: object = None
    # P6 custom controls, meaningful only with film_appearance="custom":
    # bounded multiplicative modifiers about the recipe's own values (plan
    # §13: 颜色丰度/色密度/灰阶偏色). 0/0/1 keeps custom == reference exactly.
    film_richness: float = 0.0
    film_color_density: float = 0.0
    film_neutral_bias: float = 1.0
    film_interimage: str = "declared"
    # The EFFECTIVE beta, resolved by the compiler from the declared table.
    # None means "not compiled" (hand-built test plans): the runtime falls
    # back to the table for those, but a COMPILED plan always carries the
    # value, which is what makes it immutable — A3 measured a 0.0726 max
    # pixel difference from mutating the module table after compile when the
    # runtime still consulted it.
    film_interimage_beta: float | None = None
    # Two-mode film contract: "observe" (default) = the film declares what the
    # observer saw (WB/separation/tone signature), AgX develops — colour stays with
    # the pipeline's validated rendering. "full" = the film v2 factorized chain
    # takes over (film_develop core: Stage A analytic front -> B1/tau/paper/B2,
    # EXPERIMENTAL: colour side has no external oracle); AgX keeps only
    # delivery-side gamut safety, and Ultra HDR serves full mode as the
    # "film print + scene HDR extension" pair. Meaningful only while a
    # curve_preset is active.
    film_mode: str = "observe"
    # Enlarger colour head (negative film presets only): Y/M subtractive filtration
    # in real darkroom CC units — NN CC = 0.NN optical density on the paper layer
    # the filter's complementary band exposes (Y attenuates the blue-sensitive
    # layer, M the green-sensitive), ~one stop of that separation's printing
    # exposure per 30 CC. 0 = the preset's neutral printing decision (byte-exact
    # status quo). Consumed through the preset's spectrally derived colour-head
    # field (film_curve.color_head_gain_lms, stage-3 joint field); meaningless without a negative
    # curve_preset, and physically absent for reversal film (no printing stage).
    color_head_y: float = 0.0
    color_head_m: float = 0.0
    # Declared crossover switch for the film chain's neutral-axis serving.
    # Meaningful only with film_mode="full"; otherwise inert. "off"
    # (default) is the DIGITAL NEUTRALIZED variant: the developed output is
    # divided per pixel by the shipped bounded neutral-cast curve at the
    # pixel's luminance exposure (grays neutral wherever the medium's own gray
    # is within two stops of neutral per channel). "datasheet" serves the
    # spectral chain verbatim — mid-grey anchored by the printer-light solve,
    # the rest of the neutral axis drifting per the inter-layer data (the
    # photographic meaning of crossover: e.g. Velvia's mildly cool shadows).
    film_crossover: str = "off"
    # film v2 (FILM_PRINT_RENDERING_PLAN §5.1): the emulsion's exposure state
    # relative to nominal EI — NOT an output exposure. 0.0 is the exact v1
    # identity; the public domain is declared by the stock's asset and the
    # plan compiler fail-closes out-of-domain values (§5.3).
    film_exposure_ev: float = 0.0
    # film v2 (§7.2): print timing policy. "fixed" = the EV0 joint solve's
    # q(0) regardless of exposure state (same enlarger setting); "retimed" =
    # q(E) re-solved per exposure via the factorized Stage B (negatives with
    # retimed assets only; fail-closed elsewhere).
    film_print_timing: str = "fixed"
    # film v2 P3 (§7.1/§7.2): the selected print medium id ("" = the stock's
    # default pairing) and the custom-timing manual print exposure in EV
    # (log2; only meaningful under timing="custom", identity 0.0).
    film_print_medium: str = ""
    film_print_exposure_ev: float = 0.0
    # film v2 P4 (§6): editorial developer recipe — a bounded analytic
    # perturbation of the three characteristic curves (contrast scales the
    # logE axis about the mid-grey anchor, colour density scales amounts about
    # the same anchor, fog adds uniform density and deliberately moves mid).
    # "measured_default" locks all three deltas at 0.
    film_development: str = "measured_default"
    film_dev_contrast: float = 0.0
    film_dev_fog: float = 0.0
    film_dev_density: float = 0.0
    # film v2 P4 (§8): optional editorial Film Compression — a C1 saturating
    # map on scene luminance EV above the knee, applied BEFORE the emulsion
    # (impact 0 = strict identity fast path), plus highlight colour density
    # rho driving C' = C*exp(-rho*d) toward the luminance-preserved neutral.
    film_compression: float = 0.0
    film_compression_knee: float = 2.0
    film_highlight_density: float = 0.0
    # film v2 P5 (§9): analog optics amounts (modelled_default profile, first
    # version is profile+amount only). Grain modulates density on a
    # deterministic film-space field; halation reinjects red-heavy backscatter
    # into layer exposure; bloom is the positive medium's intrinsic scatter.
    # All zero = strict identity fast path (chunk streaming preserved).
    film_grain: float = 0.0
    film_halation: float = 0.0
    film_bloom: float = 0.0
    film_optics_seed: int = 0
    # Review R1 item 4: the media scatter (emulsion §5.1 / formation §6.2)
    # is a property of the declared media, not a look amount, so it gets its
    # own enablement instead of riding whichever optics slider first engages
    # the spatial context. "declared" applies the compiled profile's scatter;
    # "off" renders without it (also the operator-isolation setting for
    # measurement tooling).
    film_media_scatter: str = "declared"
    # Display-referred dark-scene lift, implemented like darktable's look brightness:
    # it leaves encoded black/white fixed and is never an exposure gain.
    view_brightness: float = 1.0
    # Which endpoint policy compiled black_ev / white_ev. "adaptive" (default) follows
    # the scene body/tail percentiles exactly as before; "evidence" pins the black
    # endpoint to the measured sensor noise floor (prior read-noise when the sensor
    # prior is available, single-frame estimate otherwise) and the white endpoint to
    # the reliable RAW tail. Informational for reports; consumers must read the
    # endpoint fields themselves, never re-derive behaviour from this name.
    endpoint_mode: str = "adaptive"
    # Truthful note about how the evidence endpoints were sourced or degraded
    # (e.g. "黑端点=单帧噪声底估计（无传感器先验）"). None under adaptive compilation.
    endpoint_note: str | None = None


@dataclass(frozen=True)
class SceneToneMetrics:
    """Scene-referred luminance facts used only to compile the tone plan.

    On LibRaw, the reliable distribution excludes CFA sites with exhausted headroom. On
    Core Image, opcode geometry prevents that mapping, so an equal aggregate RAW-clipped
    fraction is removed from the top luminance rank instead. The latter is a conservative
    comparison heuristic, not pixel-level evidence. Neither path mixes creative or
    output-gamut decisions into these metrics.
    """

    reliable_sample_pct: float
    body_ev_p1: float
    body_ev_p5: float
    body_ev_p50: float
    body_ev_p95: float
    body_ev_p99: float
    body_ev_p999: float
    tail_ev_p9999: float
    tail_area_ev0_pct: float
    tail_area_ev2_pct: float
    tail_extremity: float
    sparse_emitter_tail: bool
    raw_clip_union_pct: float
    # Same percentile as tail_ev_p9999, excluding RAW sites with exhausted CFA headroom.
    # This is the only tail statistic allowed to set a global white endpoint or grant
    # HDR display budget. It is NaN when too little trustworthy evidence remains.
    reliable_tail_ev_p9999: float = float("nan")


@dataclass(frozen=True)
class ColorGeometryPlan:
    """Colour-only decisions for one output gamut.

    `raw_clip_retreat_strength` is applied only when a CFA-derived per-pixel mask exists;
    it is therefore inactive on Core Image buffers. Output-gamut pressure controls the
    final hue-preserving fit, never the tone endpoints.
    """

    target_gamut: str
    raw_clip_retreat_strength: float
    output_gamut_pressure_pct: float
    gamut_fit_alpha: float = 0.05
    # A restrained display-side safety valve for the luminance core. AgX already has
    # its own inset/outset path toward white, so this is zero for the AgX core.
    display_highlight_chroma_retreat: float = 0.0
    display_highlight_chroma_start: float = 0.75
    display_highlight_chroma_end: float = 0.98
    # RAW-gated DRT (tone_core=gated): master scale on color-path blend weight.
    color_path_master: float = 1.0
    gated_midtone_protect: float = 0.92
    color_path_highlight_ev_lo: float = 0.25
    color_path_highlight_ev_hi: float = 2.75
    # Scene EV below which SNR is too low to open the color path on scene evidence alone.
    gated_noise_ev_floor: float = -12.0


@dataclass(frozen=True)
class RenderPlan:
    """Immutable contract between analysis and the renderer."""

    tone: ToneCompressionPlan
    color: ColorGeometryPlan
    scene: SceneToneMetrics
    # film v2 plan objects (FILM_PRINT_RENDERING_PLAN §4): populated (with
    # identity defaults) whenever a film curve preset is active; None outside
    # the film domain. Validated fail-closed at compile time.
    film: tuple | None = None


@dataclass(frozen=True)
class RenderAdjustments:
    """Bounded user biases applied after the automatic render plan is compiled.

    Zero is an exact identity. These controls deliberately do not expose or replace the
    automatically derived pivot, tone endpoints, or RAW evidence decisions.
    """

    midtone_brightness: float = 0.0
    midtone_contrast: float = 0.0
    shadow_transition: float = 0.0
    highlight_transition: float = 0.0
    highlight_fade: float = 0.0
    # Toe-end offset in scene EV, range [-3.0, +0.5]. Moves the compiled curve's
    # toe-end point (the scene EV where display output falls to the near-black
    # reference level) by re-solving toe_power; the black endpoint itself and the
    # shoulder/white side do not move. Negative keeps deeper shadows readable
    # longer before the final dive to black; positive tightens them earlier.
    toe_end_offset: float = 0.0
    # Shoulder-white offset in scene EV, range [-2.0, +3.0]; values outside the
    # range are clamped at apply time (the RenderAdjustments-internal clamp
    # pattern shared by every field here), matching the declared range instead of
    # silently accepting wilder requests. Moves the compiled curve's shoulder
    # white point (the scene EV where display output reaches the near-white
    # reference, 90% of the floor-to-white span) by re-solving shoulder_power;
    # the white endpoint itself, the shoulder start and the toe/black side do
    # not move. Positive postpones the approach to white — highlight gradations
    # merge later, softer roll-off; negative reaches white earlier — a harder
    # shoulder. Replaces the former shoulder_start_offset, whose "move the
    # start" semantics had no geometric freedom: with contrast 3 the display
    # range above the pivot is spent within ~1 EV, so the C1 legality clamps
    # absorbed any start move before it could render.
    shoulder_white_offset: float = 0.0

    def is_identity(self) -> bool:
        return all(
            abs(float(value)) <= 1e-12
            for value in (
                self.midtone_brightness,
                self.midtone_contrast,
                self.shadow_transition,
                self.highlight_transition,
                self.highlight_fade,
                self.toe_end_offset,
                self.shoulder_white_offset,
            )
        )


@dataclass
class AutoEvResult:
    ev: float
    ev_median_target: float
    ev_boost: float
    highlight_limited: bool
    highlight_cap_ev: float
    anchored_median_ev: float


@dataclass(frozen=True)
class SceneScaleContract:
    """Immutable factors that turn stored decoder RGB into intent-scene RGB.

    Phase 1 expresses the existing product
    ``stored / storage_scale * total_render_gain`` without reordering multiplies.
    See ``dngscan.scene_scale`` for the compatibility constructor.
    """

    storage_scale: float
    decoder_calibration_gain: float = 1.0
    baseline_render_gain: float = 1.0
    fixed_midgray_gain: float = 1.0
    user_ev_gain: float = 1.0
    baseline_baked_in: bool = False
    scale_mode: str = "libraw"
    # calibrated: fixed file recipe; relative: per-file decoder comparison;
    # decoder-native: Apple/unity handoff without LibRaw alignment.
    calibration_confidence: str = "relative"

    @property
    def total_render_gain(self) -> float:
        return float(
            self.decoder_calibration_gain
            * self.baseline_render_gain
            * self.fixed_midgray_gain
            * self.user_ev_gain
        )

    @property
    def legacy_exposure_gain(self) -> float:
        """Product historically stored on ``RawBundle.exposure_gain``."""
        return float(self.fixed_midgray_gain * self.user_ev_gain)


@dataclass(frozen=True)
class HdrDisplayTarget:
    """What the display can show, independent of any photograph.

    The 100 nit default is dngscan's authoring normalization, not an Apple requirement and
    not ITU's 203 nit broadcast HDR reference white. Changing it changes the conversion
    from relative headroom to nominal nits, so it stays explicit.
    """

    reference_white_nits: float = HDR_REFERENCE_WHITE_NITS
    peak_nits: float = DEFAULT_HDR_PEAK_NITS
    limiting_gamut: str = "p3"

    @property
    def display_headroom_ev(self) -> float:
        return math.log2(self.peak_nits / self.reference_white_nits)


@dataclass(frozen=True)
class HdrShoulderSegment:
    """One monotone cubic Hermite piece of the HDR shoulder, in (scene EV -> stops)."""

    e0: float
    e1: float
    z0: float
    z1: float
    m0: float
    m1: float

    @property
    def alpha(self) -> float:
        """Normalized start tangent.

        For a SINGLE segment with a flat white end (m1 = 0), alpha <= 3 is the
        exact monotonicity bound. Interior segments of a subdivided chain have
        m1 > 0 and are validated by the exact quadratic-derivative check
        instead — their alpha may legitimately exceed 3 (see the HDR plan
        §7.4; alpha > 3 on the requested single segment is what TRIGGERS
        subdivision, not an illegal state)."""
        span_e = self.e1 - self.e0
        span_z = self.z1 - self.z0
        if span_e <= 0.0 or span_z <= 1e-12:
            return math.inf
        return self.m0 * span_e / span_z

    @property
    def beta(self) -> float:
        """Normalized end tangent; zero for the final segment by construction."""
        span_e = self.e1 - self.e0
        span_z = self.z1 - self.z0
        if span_e <= 0.0 or span_z <= 1e-12:
            return math.inf
        return self.m1 * span_e / span_z


@dataclass(frozen=True)
class HdrToneCurve:
    """The scene-authorized native HDR AgX curve.

    Three headrooms stay separate: display is capacity, requested is what the reliable RAW
    tail earns, and rendered is the endpoint the shoulder actually carries. Actual is
    measured from rendered pixels afterwards. Collapsing them is how HDR implementations
    end up normalising every frame to peak white.

    The body fields describe everything below `shoulder_start_ev`, and no field here that
    depends on headroom may influence them. That separation is the whole design: v1 raised
    a global gamma to buy peak, which silently darkened the shadows by up to three stops.
    """

    black_ev: float
    shoulder_start_ev: float
    white_ev: float
    body_gamma: float
    body_contrast: float
    toe_power: float

    reference_white_stops: float
    display_headroom_ev: float
    requested_headroom_ev: float
    rendered_headroom_ev: float
    peak_linear: float
    reliable_tail_ev: float
    white_margin_ev: float

    shoulder_segments: tuple[HdrShoulderSegment, ...] = ()
    # Diagnostic only: the request's normalized start tangent before any subdivision.
    # One segment when it is <= 3; above that the stored chain is the subdivided monotone
    # compile of the same request. The compiler must never re-read this to choose a shape.
    shoulder_alpha: float = float("nan")


@dataclass(frozen=True)
class HdrColorGeometry:
    """How much of the extra range each channel may use independently."""

    channel_separation: float
    raw_clip_retreat: float
    snr_gate: float
    hue_restore: float
    primaries_preset: str
    # R2 item 10: gamut_fit_margin was removed — it compiled to 0.0
    # unconditionally, which made it a dead knob wearing a live name. A
    # future volume-fit margin must arrive with a calibration and a policy
    # register entry, not as a field waiting to be believed.


@dataclass(frozen=True)
class HdrAgxPlan:
    """Immutable HDR DRT plan compiled from the shared scene analysis.

    ``formation`` belongs to the HDR branch.  It may initially inherit numerical
    parameters from the scene's SDR plan, but the HDR renderer never treats SDR pixels as
    its baseline and is free to evolve its own curve and colour geometry.
    """

    formation: ToneCompressionPlan
    display: HdrDisplayTarget
    tone: HdrToneCurve
    color: HdrColorGeometry
