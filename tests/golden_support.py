# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic golden-render scenes, plans, and case enumeration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from dngscan._deps import np
from dngscan.color import apply_rgb_matrix3
from dngscan.constants import OKLAB_M1, OKLAB_M1_INV, OKLAB_M2, OKLAB_M2_INV, XYZ_TO_RGB
from dngscan.models import Analysis, RawBundle, RenderPlan, SceneToneMetrics, ToneCompressionPlan
from dngscan.raw_io import build_clip_masks
from dngscan.tone import build_color_geometry_plan, build_render_plan

SCENE_SCALE = 65535.0
GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
CORES = ("agx", "gated", "lum", "neutral")
AGX_PRIMARIES = ("smooth", "base", "punchy", "muted")
PLAN_KINDS = ("fixed", "compiled")

# Do NOT add Core Image / CIRAWFilter cases to the golden matrix. Apple changes that
# decoder between OS releases; pinned bytes would rot. The optional decoder is covered
# by tests/test_coreimage_decode.py (skipped when Quartz is unavailable).


@dataclass(frozen=True)
class GoldenScene:
    scene_id: str
    bundle: RawBundle
    analysis: Analysis
    rois: dict[str, np.ndarray]


@dataclass(frozen=True)
class GoldenCase:
    scene_id: str
    plan_kind: str
    tone_core: str
    agx_primaries: str

    @property
    def fixture_name(self) -> str:
        return f"{self.scene_id}__{self.plan_kind}__{self.tone_core}__{self.agx_primaries}.npz"

    @property
    def fixture_path(self) -> Path:
        return GOLDEN_DIR / self.fixture_name


def _lin_u16(linear: np.ndarray) -> np.ndarray:
    return np.clip(linear * SCENE_SCALE, 0.0, SCENE_SCALE).astype(np.uint16)


def _bundle_from_scene(
    scene_u16: np.ndarray,
    *,
    clip_masks: np.ndarray | None = None,
    raw_image: np.ndarray | None = None,
    raw_colors: np.ndarray | None = None,
) -> RawBundle:
    h, w = scene_u16.shape[:2]
    raw = raw_image if raw_image is not None else np.zeros((max(2, h // 2), max(2, w // 2)), dtype=np.uint16)
    colors = raw_colors if raw_colors is not None else np.asarray([[0, 1], [3, 2]], dtype=np.uint8)
    return RawBundle(
        path=Path("golden.dng"),
        raw_image=raw,
        raw_colors=colors,
        xyz_render=np.zeros_like(scene_u16),
        render_scale=SCENE_SCALE,
        scene_rec2020_render=scene_u16,
        scene_scale=SCENE_SCALE,
        white_level=65535,
        black_levels=[0.0, 0.0, 0.0, 0.0],
        camera_wb=[1.0, 1.0, 1.0, 1.0],
        color_desc="RGBG",
        raw_pattern=[[0, 1], [3, 2]],
        camera_white_levels=[65535.0] * 4,
        clip_masks=clip_masks,
    )


def _analysis_for(
    *,
    median_vs_gray_ev: float,
    ev_median: float,
    ev_p99: float,
    ev_p999: float,
    usable_dr_ev: float,
    bright_pixel_pct: float = 0.0,
    gamut_out: dict[str, float] | None = None,
) -> Analysis:
    return Analysis(
        channel_ids=[0, 1, 2, 3],
        labels={0: "R", 1: "G1", 2: "B", 3: "G2"},
        ceilings={0: 65535, 1: 65535, 2: 65535, 3: 65535},
        ceil_spike_counts={0: 0, 1: 0, 2: 0, 3: 0},
        ceil_near_counts={0: 0, 1: 0, 2: 0, 3: 0},
        ceil_spike_ok={0: False, 1: False, 2: False, 3: False},
        fullwell_channel_ids=[0, 1, 3],
        fullwell_note="golden",
        saturation_levels={0: 65535, 1: 65535, 2: 65535, 3: 65535},
        channel_fullwell={0: 65535, 1: 65535, 2: 65535, 3: 65535},
        channel_thresholds={0: 65531, 1: 65531, 2: 65531, 3: 65531},
        fullwell=65535,
        threshold=65531,
        clip_pct={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        cfa_cell_supported=True,
        cell_union_pct=0.0,
        cell_ge2_of_clipped_pct=0.0,
        cell_k_of_clipped_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        cell_k_of_all_pct={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        ev_p1=-10.0,
        ev_raw_p1=-10.0,
        ev_median=ev_median,
        ev_p99=ev_p99,
        ev_p999=ev_p999,
        ev_dr_p1_p999=usable_dr_ev,
        ev_floor_hit_pct=0.0,
        median_vs_gray_ev=median_vs_gray_ev,
        median_y=0.18 * (2.0 ** median_vs_gray_ev),
        noise_floor=0.002,
        usable_dr_ev=usable_dr_ev,
        snr_curves={},
        snr1_dr={},
        snr1_stop={},
        gamut_out_pct=gamut_out or {"sRGB": 0.0, "P3": 0.0, "Rec2020": 0.0},
        bright_pixel_pct=bright_pixel_pct,
        survivor_channel="G1",
        container_bits_est=14,
        prior_id="golden",
        gain_e_per_dn=0.8,
        noise_floor_e=2.0,
        prior_read_noise_e=3.0,
        prior_pdr_ev=usable_dr_ev,
        usable_dr_eff_ev=usable_dr_ev,
        health_lag1_corr=0.0,
        health_hist_empty_pct=0.0,
    )


def _scene_metrics(
    body_ev_p50: float,
    *,
    sparse_emitter_tail: bool = False,
    tail_ev_p9999: float = 2.0,
) -> SceneToneMetrics:
    return SceneToneMetrics(
        reliable_sample_pct=99.0,
        body_ev_p1=body_ev_p50 - 4.0,
        body_ev_p5=body_ev_p50 - 2.0,
        body_ev_p50=body_ev_p50,
        body_ev_p95=body_ev_p50 + 2.0,
        body_ev_p99=body_ev_p50 + 3.0,
        body_ev_p999=body_ev_p50 + 4.0,
        tail_ev_p9999=tail_ev_p9999,
        tail_area_ev0_pct=0.5,
        tail_area_ev2_pct=0.1,
        tail_extremity=0.2,
        sparse_emitter_tail=sparse_emitter_tail,
        raw_clip_union_pct=0.0,
        reliable_tail_ev_p9999=tail_ev_p9999,
    )


def _rgb_from_oklab(l_: float, a_: float, b_: float) -> np.ndarray:
    lab = np.asarray([[l_, a_, b_]], dtype=np.float32)
    lms_ = apply_rgb_matrix3(lab, OKLAB_M2_INV)
    xyz = apply_rgb_matrix3(lms_**3, OKLAB_M1_INV)
    return apply_rgb_matrix3(xyz, XYZ_TO_RGB["Rec2020"])[0]


def build_daylight_wide_dr(seed: int = 11) -> GoldenScene:
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.full((h, w, 3), 0.02, dtype=np.float32)
    rois = {"saturated": np.zeros((h, w), dtype=bool)}
    colors = (
        (1.0, 0.05, 0.05),
        (0.05, 0.85, 0.15),
        (0.08, 0.15, 0.95),
        (0.05, 0.75, 0.85),
        (0.95, 0.85, 0.05),
        (0.85, 0.08, 0.75),
    )
    for idx, rgb in enumerate(colors):
        cy = 18 + (idx // 3) * 24
        cx = 18 + (idx % 3) * 24
        mask = (yy >= cy) & (yy < cy + 16) & (xx >= cx) & (xx < cx + 16)
        base[mask] = np.asarray(rgb, dtype=np.float32) * 0.35
        rois["saturated"][mask] = True
    base[yy < 10, :] = np.stack(
        [0.75 + 0.02 * xx[yy < 10] / w, 0.75 + 0.02 * xx[yy < 10] / w, 0.75 + 0.02 * xx[yy < 10] / w],
        axis=-1,
    )
    scene = _lin_u16(base)
    analysis = _analysis_for(
        median_vs_gray_ev=-0.5,
        ev_median=-0.5,
        ev_p99=2.5,
        ev_p999=3.5,
        usable_dr_ev=11.0,
        bright_pixel_pct=2.0,
    )
    return GoldenScene("daylight_wide_dr", _bundle_from_scene(scene), analysis, rois)


def build_night_sparse_lamps(seed: int = 17) -> GoldenScene:
    h, w = 96, 96
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0008, 0.012, size=(h, w, 3)).astype(np.float32)
    rois = {"lamps": np.zeros((h, w), dtype=bool), "body": np.ones((h, w), dtype=bool)}
    for _ in range(8):
        cy, cx = int(rng.integers(8, h - 8)), int(rng.integers(8, w - 8))
        yy, xx = np.mgrid[0:h, 0:w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= 16
        base[mask] = rng.uniform(0.75, 1.2)
        rois["lamps"][mask] = True
        rois["body"][mask] = False
    scene = _lin_u16(base)
    analysis = _analysis_for(
        median_vs_gray_ev=-3.2,
        ev_median=-3.2,
        ev_p99=1.0,
        ev_p999=2.2,
        usable_dr_ev=9.0,
        bright_pixel_pct=0.2,
    )
    return GoldenScene("night_sparse_lamps", _bundle_from_scene(scene), analysis, rois)


def build_high_key(seed: int = 23) -> GoldenScene:
    h, w = 64, 64
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.42, 0.82, size=(h, w, 3)).astype(np.float32)
    rois = {"high_key": np.ones((h, w), dtype=bool)}
    scene = _lin_u16(base)
    analysis = _analysis_for(
        median_vs_gray_ev=1.4,
        ev_median=1.4,
        ev_p99=2.8,
        ev_p999=3.2,
        usable_dr_ev=8.0,
        bright_pixel_pct=35.0,
    )
    return GoldenScene("high_key", _bundle_from_scene(scene), analysis, rois)


def build_skin_grid(seed: int = 29) -> GoldenScene:
    h, w = 96, 96
    base = np.zeros((h, w, 3), dtype=np.float32)
    rois = {"skin": np.zeros((h, w), dtype=bool)}
    hues = (25.0, 40.0, 55.0)
    levels = (0.35, 0.55, 0.72)
    chroma = 0.08
    for row, l_ in enumerate(levels):
        for col, hue_deg in enumerate(hues):
            rad = math.radians(hue_deg)
            rgb = _rgb_from_oklab(l_, chroma * math.cos(rad), chroma * math.sin(rad))
            y0, x0 = 8 + row * 28, 8 + col * 28
            base[y0 : y0 + 20, x0 : x0 + 20] = rgb
            rois["skin"][y0 : y0 + 20, x0 : x0 + 20] = True
    scene = _lin_u16(base)
    analysis = _analysis_for(
        median_vs_gray_ev=-0.2,
        ev_median=-0.2,
        ev_p99=1.0,
        ev_p999=1.5,
        usable_dr_ev=9.5,
    )
    return GoldenScene("skin_grid", _bundle_from_scene(scene), analysis, rois)


def build_neutral_hue_wheel(seed: int = 31) -> GoldenScene:
    h, w = 96, 96
    base = np.zeros((h, w, 3), dtype=np.float32)
    rois = {"gray_axis": np.zeros((h, w), dtype=bool)}
    for row in range(3):
        for x in range(w // 2):
            level = 0.08 + 0.75 * x / max(1, w // 2 - 1)
            base[row * 28 : row * 28 + 20, x] = level
            rois["gray_axis"][row * 28 : row * 28 + 20, x] = True
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy, radius = w * 3 // 4, h // 2, min(h, w) // 3
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    wheel = dist <= radius
    hue = (np.arctan2(yy - cy, xx - cx) / (2.0 * math.pi)) % 1.0
    for level_i, level in enumerate((0.25, 0.5, 0.75)):
        band = wheel & (dist >= radius * (level_i / 3.0)) & (dist < radius * ((level_i + 1) / 3.0))
        rgb = np.stack(
            [
                level * (0.5 + 0.5 * np.cos(2.0 * math.pi * hue)),
                level * (0.5 + 0.5 * np.cos(2.0 * math.pi * (hue - 1.0 / 3.0))),
                level * (0.5 + 0.5 * np.cos(2.0 * math.pi * (hue - 2.0 / 3.0))),
            ],
            axis=-1,
        )
        base[band] = np.clip(rgb[band], 0.0, None)
    scene = _lin_u16(base)
    analysis = _analysis_for(
        median_vs_gray_ev=0.0,
        ev_median=0.0,
        ev_p99=1.2,
        ev_p999=1.8,
        usable_dr_ev=10.0,
    )
    return GoldenScene("neutral_hue_wheel", _bundle_from_scene(scene), analysis, rois)


def build_staggered_clip(seed: int = 37) -> GoldenScene:
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    grad = (xx / max(1, w - 1)).astype(np.float32)
    base = np.stack(
        [
            0.05 + 1.4 * grad,
            0.05 + 1.1 * grad**1.2,
            0.05 + 0.9 * grad**1.4,
        ],
        axis=-1,
    )
    scene = _lin_u16(base)
    raw_h, raw_w = max(4, h // 2), max(4, w // 2)
    raw = np.full((raw_h, raw_w), 2000, dtype=np.uint16)
    colors = np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (raw_h // 2, raw_w // 2))
    yy_raw, _ = np.mgrid[0:raw_h, 0:raw_w]
    raw[yy_raw < raw_h // 3] = 64000
    raw[(yy_raw >= raw_h // 3) & (yy_raw < 2 * raw_h // 3)] = 62000
    raw[yy_raw >= 2 * raw_h // 3] = 60000
    clip_masks = build_clip_masks(
        raw,
        colors,
        "RGBG",
        white_level=65535,
        black_levels=[0.0, 0.0, 0.0, 0.0],
        camera_white_levels=[65535.0] * 4,
        orientation_flip=0,
        scene_shape=(h, w),
        raw_pattern=[[0, 1], [3, 2]],
    ).astype(np.float32)
    rois = {
        "highlight": grad > 0.7,
        "clip_band": (yy < h // 3) | ((yy >= h // 3) & (yy < 2 * h // 3)) | (yy >= 2 * h // 3),
    }
    bundle = _bundle_from_scene(scene, clip_masks=clip_masks, raw_image=raw, raw_colors=colors)
    analysis = _analysis_for(
        median_vs_gray_ev=0.2,
        ev_median=0.2,
        ev_p99=3.0,
        ev_p999=3.8,
        usable_dr_ev=12.0,
        bright_pixel_pct=8.0,
    )
    return GoldenScene("staggered_clip", bundle, analysis, rois)


def _build_real_crop(crop_path: Path) -> GoldenScene:
    """Real-scene excerpt exported by tools/regen_golden.py --from-dng.

    The stored buffer is a scene-linear Rec.2020 uint16 crop of an actual capture, so
    these cases exercise real material/illumination statistics rather than synthetic
    manifolds. Pickle stays disabled: every stored field is a plain array."""
    with np.load(crop_path, allow_pickle=False) as payload:
        scene = np.asarray(payload["scene"], dtype=np.uint16)
        median_ev = float(payload["analysis_median_ev"])
    rois = {"full": np.ones(scene.shape[:2], dtype=bool)}
    analysis = _analysis_for(
        median_vs_gray_ev=median_ev,
        ev_median=median_ev,
        ev_p99=min(3.0, median_ev + 4.0),
        ev_p999=min(3.5, median_ev + 5.0),
        usable_dr_ev=10.0,
    )
    return GoldenScene(crop_path.stem, _bundle_from_scene(scene), analysis, rois)


SCENE_BUILDERS = {
    "daylight_wide_dr": build_daylight_wide_dr,
    "night_sparse_lamps": build_night_sparse_lamps,
    "high_key": build_high_key,
    "skin_grid": build_skin_grid,
    "neutral_hue_wheel": build_neutral_hue_wheel,
    "staggered_clip": build_staggered_clip,
}

# Real-scene crops are committed fixtures (crop__<name>.npz): register whichever are
# present so they run through the same case matrix as the synthetic scenes. Their OWN
# rendered fixtures also start with "crop__" (crop__<name>__<plan>__<core>__...), so
# filter to source crops only: exactly one "__" separator in the stem.
for _crop_path in sorted(GOLDEN_DIR.glob("crop__*.npz")):
    if _crop_path.stem.count("__") == 1:
        SCENE_BUILDERS[_crop_path.stem] = (lambda p=_crop_path: _build_real_crop(p))


def all_scenes() -> dict[str, GoldenScene]:
    return {scene_id: builder() for scene_id, builder in SCENE_BUILDERS.items()}


def fixed_tone_plan(scene: GoldenScene, tone_core: str, agx_primaries: str) -> ToneCompressionPlan:
    body_ev = float(scene.analysis.ev_median)
    common = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        chroma_p95=0.5,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core=tone_core,
        use_c1_endpoints=True,
    )
    if scene.scene_id == "night_sparse_lamps":
        return ToneCompressionPlan(
            **common,
            black_ev=-9.0,
            white_ev=3.0,
            dynamic_range_ev=12.0,
            contrast=3.0,
            toe_power=1.5,
            shoulder_power=3.3,
            latitude_hi_ev=1.5,
            punch_strength=0.0,
            view_brightness=1.12,
            agx_primaries=agx_primaries if tone_core == "agx" else "base",
        )
    if scene.scene_id == "high_key":
        return ToneCompressionPlan(
            **common,
            black_ev=-6.0,
            white_ev=4.5,
            dynamic_range_ev=10.5,
            contrast=2.8,
            toe_power=1.4,
            shoulder_power=3.0,
            punch_strength=0.15,
            agx_primaries=agx_primaries if tone_core == "agx" else "base",
        )
    if scene.scene_id == "staggered_clip":
        return ToneCompressionPlan(
            **common,
            black_ev=-7.0,
            white_ev=5.5,
            dynamic_range_ev=12.5,
            contrast=3.1,
            toe_power=1.5,
            shoulder_power=3.2,
            latitude_hi_ev=2.0,
            punch_strength=0.1,
            agx_primaries=agx_primaries if tone_core == "agx" else "base",
        )
    return ToneCompressionPlan(
        **common,
        black_ev=-8.0,
        white_ev=5.0,
        dynamic_range_ev=13.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=2.9,
        latitude_hi_ev=1.0 if scene.scene_id == "daylight_wide_dr" else 0.5,
        punch_strength=0.2 if scene.scene_id == "daylight_wide_dr" else 0.0,
        hue_restore=0.6,
        agx_primaries=agx_primaries if tone_core == "agx" else "base",
    )


def render_plan_for_case(scene: GoldenScene, case: GoldenCase) -> RenderPlan | ToneCompressionPlan:
    if case.plan_kind == "compiled":
        return build_render_plan(
            scene.bundle,
            scene.analysis,
            "agx",
            "srgb",
            tone_core=case.tone_core,
            agx_primaries=case.agx_primaries if case.tone_core == "agx" else "base",
        )
    tone = fixed_tone_plan(scene, case.tone_core, case.agx_primaries)
    if scene.bundle.clip_masks is not None or case.tone_core == "gated":
        return RenderPlan(
            tone=tone,
            color=build_color_geometry_plan(scene.analysis, "srgb", case.tone_core),
            scene=_scene_metrics(float(scene.analysis.ev_median), sparse_emitter_tail=scene.scene_id == "night_sparse_lamps"),
        )
    return tone


def iter_cases() -> Iterator[GoldenCase]:
    for scene_id in SCENE_BUILDERS:
        for plan_kind in PLAN_KINDS:
            for tone_core in CORES:
                primaries_list = AGX_PRIMARIES if tone_core == "agx" else ("smooth",)
                for agx_primaries in primaries_list:
                    yield GoldenCase(scene_id, plan_kind, tone_core, agx_primaries)


def oklab_stats(u8: np.ndarray, roi: np.ndarray | None = None) -> dict[str, float]:
    rgb = u8.astype(np.float32) / 255.0
    if roi is not None:
        rgb = rgb[roi]
    flat = rgb.reshape(-1, 3)
    xyz = apply_rgb_matrix3(flat, np.linalg.inv(XYZ_TO_RGB["Rec2020"]).astype(np.float32))
    lab = apply_rgb_matrix3(np.cbrt(np.maximum(apply_rgb_matrix3(xyz, OKLAB_M1), 0.0)), OKLAB_M2)
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    return {
        "luma_p10": float(np.percentile(lab[:, 0], 10)),
        "luma_p50": float(np.percentile(lab[:, 0], 50)),
        "luma_p99": float(np.percentile(lab[:, 0], 99)),
        "chroma_mean": float(np.mean(chroma)),
        "chroma_p90": float(np.percentile(chroma, 90)),
    }
