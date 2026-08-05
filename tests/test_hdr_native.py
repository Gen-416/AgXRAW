# SPDX-License-Identifier: GPL-3.0-or-later
"""Parity tests for the native HDR formation kernel (cpp/src/hdr_core.cpp).

The NumPy body of hdr_agx._form_hdr_chunk is the reference implementation; the
native kernel must reproduce it per pixel on real compiled plans, including the
RAW-gated chroma blend, the aliased-table identity, and the NaN/Inf contract.
"""

from __future__ import annotations

import dataclasses
import os
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import _fast as fast_backend
from dngscan import agx as agx_engine
from dngscan import hdr_agx
from dngscan.drt import curve_params_from_plan
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.hdr_color import formation_luma_weights
from dngscan.hdr_curve import compile_hdr_curve_table_pair
from dngscan.models import (
    ColorGeometryPlan,
    RenderPlan,
    SceneToneMetrics,
    ToneCompressionPlan,
)

# Thresholds for |native - numpy| on display-linear output up to peak (~3-8 here).
# Measured worst case across the variant sweep is 8.5e-5 max / 5.5e-6 p99; the gates
# leave ~2x headroom while staying far below one 8-bit output step.
MAX_ABS_TOL = 2e-4
P99_ABS_TOL = 2e-5


def _scene_plan(**tone_overrides) -> RenderPlan:
    tone = dict(
        target_gamut="Rec2020",
        luma_p1=0.01,
        luma_p50=0.18,
        luma_p99=1.0,
        luma_p999=2.0,
        black_ev=-8.0,
        white_ev=5.0,
        dynamic_range_ev=13.0,
        contrast=3.0,
        toe_power=1.5,
        shoulder_power=2.9,
        chroma_p95=0.5,
        negative_rgb_pct=0.0,
        over_rgb_pct=0.0,
        tone_core="agx",
        use_c1_endpoints=True,
        punch_strength=0.3,
        hue_restore=0.6,
        agx_primaries="base",
    )
    sparse = bool(tone_overrides.pop("sparse_emitter_tail", False))
    tail = float(tone_overrides.pop("reliable_tail_ev_p9999", 4.2))
    tone.update(tone_overrides)
    color = ColorGeometryPlan(
        target_gamut="Rec2020",
        raw_clip_retreat_strength=0.4,
        output_gamut_pressure_pct=1.0,
    )
    scene = SceneToneMetrics(
        reliable_sample_pct=99.0,
        body_ev_p1=-7.0,
        body_ev_p5=-5.0,
        body_ev_p50=-0.5,
        body_ev_p95=2.5,
        body_ev_p99=3.5,
        body_ev_p999=4.5,
        tail_ev_p9999=5.0,
        tail_area_ev0_pct=8.0,
        tail_area_ev2_pct=1.0,
        tail_extremity=0.4,
        sparse_emitter_tail=sparse,
        raw_clip_union_pct=0.5,
        reliable_tail_ev_p9999=tail,
    )
    return RenderPlan(ToneCompressionPlan(**tone), color, scene)


def _formation_setup(channel_separation: float | None = None, **tone_overrides):
    """Compile a real HdrAgxPlan and everything _form_hdr_chunk consumes."""
    plan = _scene_plan(**tone_overrides)
    # SimpleNamespace analysis: no clipping evidence recorded, so
    # compile_channel_separation grants the full RHO_BASE (0.5).
    hdr_plan = compile_hdr_agx_plan(plan, analysis=SimpleNamespace())
    if channel_separation is not None:
        hdr_plan = dataclasses.replace(
            hdr_plan,
            color=dataclasses.replace(
                hdr_plan.color, channel_separation=float(channel_separation)
            ),
        )
    hdr_tone_plan = hdr_agx._hdr_tone_plan(hdr_plan)
    inset, outset = agx_engine.formation_matrices(hdr_tone_plan)
    formation_y = formation_luma_weights(outset)
    body_params = curve_params_from_plan(hdr_tone_plan)
    curve_tables = compile_hdr_curve_table_pair(
        hdr_plan.tone,
        hdr_tone_plan,
        need_reference=hdr_agx._hdr_reference_needed(hdr_plan),
        body_params=body_params,
    )
    peak = hdr_agx._pack_peak(hdr_plan)
    return hdr_plan, hdr_tone_plan, inset, outset, formation_y, curve_tables, peak


def _sample_pixels(count: int = 60_000, with_edges: bool = True) -> np.ndarray:
    """Scene-linear samples spanning deep shadow to far past the white endpoint."""
    rng = np.random.default_rng(20260801)
    rgb = np.exp2(rng.uniform(-14.0, 7.0, size=(count, 3))).astype(np.float32) * 0.18
    rgb[::17] *= -1.0  # negative scene values exercise the gamut guard rail
    if not with_edges:
        return np.ascontiguousarray(rgb)
    edges = np.asarray(
        [
            [np.nan, 0.3, 0.4],
            [0.3, np.nan, 0.4],
            [0.3, 0.4, np.nan],
            [np.inf, 0.3, 0.4],
            [0.3, np.inf, 0.4],
            [0.3, 0.4, np.inf],
            [-np.inf, 0.3, 0.4],
            [0.3, -np.inf, 0.4],
            [0.3, 0.4, -np.inf],
            [np.nan, np.inf, -np.inf],
            [0.0, 0.0, 0.0],
            [1e8, 1e8, 1e8],
            [-5.0, -5.0, -5.0],
            [0.18, 0.18, 0.18],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(np.concatenate((edges, rgb), axis=0))


def _sample_masks(shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(41)
    masks = np.clip(rng.uniform(-0.2, 1.2, size=shape), 0.0, 1.0).astype(np.float32)
    masks[::3] = 0.0  # unclipped majority
    masks[1::97] = 1.0  # fully clipped sites withdraw all separation
    return np.ascontiguousarray(masks)


def _reference_chunk(setup, rgb, masks, output_gamut):
    hdr_plan, hdr_tone_plan, inset, outset, formation_y, curve_tables, peak = setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return hdr_agx._form_hdr_chunk(
            rgb,
            hdr_plan,
            hdr_tone_plan,
            inset,
            outset,
            formation_y,
            curve_tables,
            masks,
            peak,
            output_gamut,
        )


def _native_plan(setup, output_gamut):
    hdr_plan, hdr_tone_plan, inset, outset, formation_y, curve_tables, peak = setup
    return fast_backend.compile_hdr_plan(
        hdr_plan,
        hdr_tone_plan,
        inset,
        outset,
        formation_y,
        curve_tables,
        peak,
        output_gamut,
    )


@unittest.skipUnless(fast_backend.available(), "native extension not built")
class NativeHdrFormationParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "1"

    def tearDown(self) -> None:
        if self._env is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = self._env

    def _assert_parity(self, setup, rgb, masks, output_gamut) -> None:
        ref = _reference_chunk(setup, rgb, masks, output_gamut)
        out = fast_backend.apply_hdr_formation_f32(rgb, masks, _native_plan(setup, output_gamut))
        self.assertTrue(np.isfinite(out).all())
        delta = np.abs(out - ref)
        self.assertLessEqual(float(delta.max()), MAX_ABS_TOL)
        self.assertLessEqual(float(np.percentile(delta, 99)), P99_ABS_TOL)

    def test_reference_blend_without_masks(self) -> None:
        setup = _formation_setup()
        self.assertIsNot(setup[5][0], setup[5][1])  # distinct reference table
        rgb = _sample_pixels()
        for gamut in ("p3", "srgb"):
            self._assert_parity(setup, rgb, None, gamut)

    def test_mask_gated_blend(self) -> None:
        setup = _formation_setup()
        rgb = _sample_pixels()
        masks = _sample_masks(rgb.shape)
        for gamut in ("p3", "srgb"):
            self._assert_parity(setup, rgb, masks, gamut)

    def test_masks_change_the_result(self) -> None:
        """The gated branch must actually be exercised, not silently ignored."""
        setup = _formation_setup()
        rgb = _sample_pixels(count=20_000, with_edges=False)
        plan = _native_plan(setup, "p3")
        without = fast_backend.apply_hdr_formation_f32(rgb, None, plan)
        with_masks = fast_backend.apply_hdr_formation_f32(
            rgb, np.ones_like(rgb), plan
        )
        self.assertGreater(float(np.abs(with_masks - without).max()), 1e-4)

    def test_zero_separation_aliases_reference(self) -> None:
        setup = _formation_setup(channel_separation=0.0)
        self.assertIs(setup[5][0], setup[5][1])  # aliased tables
        rgb = _sample_pixels()
        masks = _sample_masks(rgb.shape)
        for m in (None, masks):
            self._assert_parity(setup, rgb, m, "p3")

    def test_plan_variants(self) -> None:
        rgb = _sample_pixels(count=30_000)
        variants = (
            dict(hue_restore=0.0, punch_strength=0.0),
            dict(hue_restore=1.0, agx_primaries="punchy"),
            dict(agx_primaries="muted", punch_strength=0.8),
            dict(sparse_emitter_tail=True, reliable_tail_ev_p9999=7.5),
        )
        for overrides in variants:
            setup = _formation_setup(**overrides)
            self._assert_parity(setup, rgb, None, "p3")

    def test_dispatch_uses_native_plan(self) -> None:
        """_form_hdr_chunk with a compiled plan routes through the kernel."""
        setup = _formation_setup()
        hdr_plan, hdr_tone_plan, inset, outset, formation_y, curve_tables, peak = setup
        native_plan = hdr_agx._compile_native_hdr_plan(
            hdr_plan, hdr_tone_plan, inset, outset, formation_y,
            curve_tables, peak, "p3",
        )
        self.assertIsNotNone(native_plan)
        rgb = _sample_pixels(count=4_096, with_edges=False)
        dispatched = hdr_agx._form_hdr_chunk(
            rgb, hdr_plan, hdr_tone_plan, inset, outset, formation_y,
            curve_tables, None, peak, "p3", native_plan=native_plan,
        )
        direct = fast_backend.apply_hdr_formation_f32(rgb, None, native_plan)
        np.testing.assert_array_equal(dispatched, direct)

    def test_native_does_not_mutate_input(self) -> None:
        setup = _formation_setup()
        rgb = _sample_pixels(count=1_024, with_edges=False)
        masks = _sample_masks(rgb.shape)
        rgb_before = rgb.copy()
        masks_before = masks.copy()
        fast_backend.apply_hdr_formation_f32(rgb, masks, _native_plan(setup, "p3"))
        np.testing.assert_array_equal(rgb, rgb_before)
        np.testing.assert_array_equal(masks, masks_before)


class NativeHdrDispatchTests(unittest.TestCase):
    def test_film_takeover_is_excluded(self) -> None:
        setup = _formation_setup(film_mode="full", curve_preset="portra400")
        hdr_tone_plan = setup[1]
        self.assertFalse(fast_backend.supports_hdr_formation(hdr_tone_plan))
        self.assertIsNone(
            hdr_agx._compile_native_hdr_plan(
                setup[0], hdr_tone_plan, setup[2], setup[3], setup[4],
                setup[5], setup[6], "p3",
            )
        )

    def test_strict_mode_raises_when_extension_unavailable(self) -> None:
        setup = _formation_setup()
        rgb = _sample_pixels(count=16, with_edges=False)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
            with mock.patch.object(fast_backend, "_load_extension", return_value=None):
                with self.assertRaises(fast_backend.NativeKernelError):
                    fast_backend.apply_hdr_formation_f32(rgb, None, object())

    @unittest.skipUnless(fast_backend.available(), "native extension not built")
    def test_dispatch_falls_back_in_auto_and_raises_in_strict(self) -> None:
        setup = _formation_setup()
        hdr_plan, hdr_tone_plan, inset, outset, formation_y, curve_tables, peak = setup
        native_plan = _native_plan(setup, "p3")
        rgb = _sample_pixels(count=512, with_edges=False)

        def _chunk():
            return hdr_agx._form_hdr_chunk(
                rgb, hdr_plan, hdr_tone_plan, inset, outset, formation_y,
                curve_tables, None, peak, "p3", native_plan=native_plan,
            )

        reference = hdr_agx._form_hdr_chunk(
            rgb, hdr_plan, hdr_tone_plan, inset, outset, formation_y,
            curve_tables, None, peak, "p3", native_plan=None,
        )
        boom = mock.patch.object(
            fast_backend, "apply_hdr_formation_f32",
            side_effect=RuntimeError("injected"),
        )
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "auto"}):
            with boom:
                fallback = _chunk()
        np.testing.assert_array_equal(fallback, reference)
        with mock.patch.dict(os.environ, {"DNGSCAN_FAST": "1"}):
            with boom:
                with self.assertRaises(fast_backend.NativeKernelError):
                    _chunk()


if __name__ == "__main__":
    unittest.main()
