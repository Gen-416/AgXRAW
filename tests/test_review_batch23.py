# SPDX-License-Identifier: GPL-3.0-or-later
"""External review batch 23 (2026-09-02 handoff): chroma NR shippable.

1. R-P1-3 / R-P3-1: the GUI had no chroma-NR path at all — no control, no
   service forwarding, no cache-key or fingerprint entry (an API payload's
   ``chromaNr`` was silently ignored). Now a page control (SDR-only, greyed
   and snapped to 0 under an HDR container), a parser that refuses a
   nonzero HDR payload, and the dial threaded through every plan compile,
   the auto-EV probes, the preview pixel key, the export suffix and the
   fingerprint.
2. R-P1-1: the chroma-NR working set was outside the §9.3 optics budget.
   The à-trous cascade now runs one channel at a time (byte-identical,
   ~1/3 the transient) and the resident map is charged to the band-row
   budget; the batch-13 RSS gate runs a chroma_nr case.
3. R-P1-4: the band is declared in SENSOR pixels but was measured against
   the render grid, so a preview proxy's band sat at ~6x the export's on a
   61 MP frame. Proxy bundles now record ``proxy_scale`` and the map builder
   folds it into the decimation factor.
4. R-P2-4: the any-overlap level rule let the top level reach 2x the declared
   upper band edge; the geometric-centre rule bounds the realized band to
   within √2 of the declaration, documented as such.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVICE_SRC = (ROOT / "dngscan" / "gui" / "service.py").read_text(encoding="utf-8")
PAGE = (ROOT / "dngscan" / "gui" / "page.py").read_text(encoding="utf-8")


def _calls(source: str, callee: str):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == callee:
                yield {kw.arg for kw in node.keywords if kw.arg is not None}


class ChromaNrParser(unittest.TestCase):
    def test_default_and_range(self) -> None:
        from dngscan.gui.service import parse_chroma_nr

        self.assertEqual(parse_chroma_nr({}, "sdr"), 0.0)
        self.assertEqual(parse_chroma_nr({"chromaNr": ""}, "sdr"), 0.0)
        self.assertAlmostEqual(parse_chroma_nr({"chromaNr": "0.3"}, "sdr"), 0.3)
        self.assertAlmostEqual(parse_chroma_nr({"chroma_nr": 1.0}, "sdr"), 1.0)
        with self.assertRaises(ValueError):
            parse_chroma_nr({"chromaNr": 1.5}, "sdr")
        with self.assertRaises(ValueError):
            parse_chroma_nr({"chromaNr": "nan"}, "sdr")

    def test_hdr_container_refuses_a_nonzero_dial(self) -> None:
        from dngscan.gui.service import parse_chroma_nr

        for fmt in ("ultrahdr", "ultrahdr-heic"):
            self.assertEqual(parse_chroma_nr({"chromaNr": 0}, fmt), 0.0)
            with self.assertRaises(ValueError):
                parse_chroma_nr({"chromaNr": 0.2}, fmt)


class ServiceThreadsTheDial(unittest.TestCase):
    def test_every_plan_and_probe_call_carries_chroma_nr(self) -> None:
        for callee in ("build_render_plan", "_cached_render_plan", "compute_auto_ev",
                       "max_safe_ev", "export_preview_jpeg",
                       "_preview_pixel_key", "export_suffix_parts",
                       "export_plan_fingerprint", "estimate_ev_headroom"):
            calls = [c for c in _calls(SERVICE_SRC, callee) if c]  # keyword calls only
            self.assertTrue(calls, callee)
            for kwargs in calls:
                with self.subTest(callee=callee):
                    self.assertIn("chroma_nr", kwargs)

    def test_parsed_in_every_entry_point(self) -> None:
        for fn in ("run_preview", "prepare_preview", "run_export"):
            body = SERVICE_SRC[SERVICE_SRC.index(f"def {fn}("):]
            body = body[: body.index("\ndef ", 1)]
            self.assertIn("chroma_nr = parse_chroma_nr(params, output_format)", body, fn)

    def test_preview_pixel_key_and_plan_key_include_it(self) -> None:
        from dngscan.gui import service

        sig = inspect.signature(service._preview_pixel_key)
        self.assertIn("chroma_nr", sig.parameters)
        self.assertIn("chroma_nr", inspect.signature(service._cached_render_plan).parameters)
        bundle = SimpleNamespace(scene_scale=1.0, scene_decoder_runtime="", lens_filter="none")
        base = dict(gamut="p3", ev=0.0, look="none", look_strength=1.0, display_filter="none",
                    filter_strength=1.0, scene_transform="none", scene_transform_strength=1.0,
                    punch_scale=1.0, tone_core="agx", lum_norm="y", agx_primaries="base",
                    lens_filter="none", film_curve="none", adjustments=None)
        a = service._preview_pixel_key(bundle, **base, chroma_nr=0.0)
        b = service._preview_pixel_key(bundle, **base, chroma_nr=0.3)
        self.assertNotEqual(a, b)
        plan_key = SERVICE_SRC[SERVICE_SRC.index("def _cached_render_plan("):]
        plan_key = plan_key[: plan_key.index("base = cached.get_or_build_plan(")]
        self.assertIn("_cache_float(chroma_nr),", plan_key)

    def test_suffix_names_a_repaired_render(self) -> None:
        from dngscan.gui.service import export_suffix_parts

        clean = export_suffix_parts("clip", "srgb", "sdr")
        repaired = export_suffix_parts("clip", "srgb", "sdr", chroma_nr=0.3)
        self.assertNotIn("cnr", clean)
        self.assertIn("cnr0_3", repaired)


class PageControl(unittest.TestCase):
    def test_control_persists_and_rides_the_payload_sdr_only(self) -> None:
        self.assertIn('id="chromaNr"', PAGE)
        self.assertIn('id="chromaNrVal"', PAGE)
        payload = PAGE[PAGE.index("function payload("):]
        payload = payload[: payload.index("return p;")]
        self.assertIn('chromaNr:["ultrahdr","ultrahdr-heic"].includes($("#format").value)?0:+$("#chromaNr").value', payload)
        self.assertIn('"clipMargin","chromaNr"]', PAGE)  # restoreSettings list
        self.assertIn('chromaNr:$("#chromaNr").value', PAGE)  # saveSettings
        fmt = PAGE[PAGE.index("function updateFormatUi("):]
        fmt = fmt[: fmt.index("\n}")]
        self.assertIn('$("#chromaNr").disabled=hdr;', fmt)
        self.assertIn('$("#chromaNr").value=0;setChromaNrLabel();', fmt)
        reset = PAGE[PAGE.index("function resetToDefaults(){"):]
        reset = reset[: reset.index("\n}")]
        self.assertIn("setChromaNrLabel", reset)


class BandRule(unittest.TestCase):
    def test_realized_band_is_within_root2_of_the_declaration(self) -> None:
        from dngscan.chroma_nr import BAND_HI_PX, BAND_LO_PX, atrous_levels_for

        r2 = 2.0 ** 0.5
        for factor in (1.0, 1.5, 2.0, 3.0, 4.26, 6.8, 8.0, 12.0, 20.0):
            levels = atrous_levels_for(factor)
            with self.subTest(factor=factor):
                self.assertTrue(levels)
                lo = factor * 2.0 ** levels[0]
                hi = factor * 2.0 ** (levels[-1] + 1)
                self.assertGreaterEqual(lo, BAND_LO_PX / r2 - 1e-9)
                self.assertLessEqual(hi, BAND_HI_PX * r2 + 1e-9)
                # coverage: the grid itself bounds the finest realizable scale
                self.assertLessEqual(lo, max(BAND_LO_PX * r2, factor) + 1e-9)
                self.assertGreaterEqual(hi, BAND_HI_PX / r2 - 1e-9)
                self.assertEqual(levels, tuple(range(levels[0], levels[-1] + 1)))
        self.assertEqual(atrous_levels_for(1.0), (3, 4, 5, 6))
        self.assertEqual(atrous_levels_for(6.8)[0], 0)
        # the case that broke the declaration: level 4 at 6.8x spans 109-218 px
        self.assertNotIn(4, atrous_levels_for(6.8))


def _reference_three_channel_map(scene_dec, amount, levels):
    """The pre-batch-23 cascade, three channels at once (for byte identity)."""
    from dngscan.chroma_nr import LUMA_W, _MAD_TO_SIGMA, _THRESHOLD_K, _atrous_smooth

    dec = np.asarray(scene_dec, dtype=np.float32)
    y = dec @ LUMA_W
    chroma = dec - y[..., None]
    smooth = chroma
    total_removed = np.zeros_like(chroma)
    max_step = max((min(chroma.shape[:2]) - 1) // 2, 1)
    included = set(levels)
    top = max(included) if included else -1
    for level in range(top + 1):
        if (1 << level) > max_step:
            break
        coarser = _atrous_smooth(smooth, level)
        if level in included:
            detail = smooth - coarser
            mad = np.median(np.abs(detail.reshape(-1, 3)), axis=0)
            threshold = (np.float32(float(amount) * _THRESHOLD_K * _MAD_TO_SIGMA) * mad).astype(np.float32)
            t2 = np.square(threshold)[None, None, :]
            total_removed += detail * (t2 / (t2 + np.square(detail) + np.float32(1e-30)))
        smooth = coarser
    correction = -total_removed
    correction -= (correction @ LUMA_W)[..., None]
    return correction.astype(np.float32, copy=False)


class PerChannelCascade(unittest.TestCase):
    def test_byte_identical_to_the_three_channel_form(self) -> None:
        from dngscan import chroma_nr

        rng = np.random.default_rng(23)
        dec = (rng.uniform(0.01, 0.6, (96, 128, 3)) + rng.normal(0.0, 0.02, (96, 128, 3))).astype(np.float32)
        for factor in (1.0, 4.26):
            levels = chroma_nr.atrous_levels_for(factor)
            got = chroma_nr.chroma_correction_map(dec, 0.5, decimation_factor=factor)
            ref = _reference_three_channel_map(dec, 0.5, levels)
            self.assertTrue(np.array_equal(got, ref), f"factor {factor}")
            self.assertLess(float(np.abs(got @ chroma_nr.LUMA_W).max()), 1e-6)


class SensorScaleAndBudget(unittest.TestCase):
    def test_proxy_scale_reaches_the_decimation_factor(self) -> None:
        from dngscan import render

        self.assertEqual(render.sensor_px_per_render_px(SimpleNamespace(proxy_scale=5.94), 10, 10), 5.94)
        self.assertEqual(render.sensor_px_per_render_px(SimpleNamespace(), 10, 10), 1.0)
        src = inspect.getsource(render._prepare_chroma_nr_map)
        self.assertIn("sensor_px_per_render_px(bundle, h, w)", src)

    def test_proxy_entries_record_and_round_trip_proxy_scale(self) -> None:
        from dngscan import models
        from dngscan.gui import preview_cache

        self.assertIn("proxy_scale", {f.name for f in models.RawBundle.__dataclass_fields__.values()})
        self.assertEqual(models.RawBundle.__dataclass_fields__["proxy_scale"].default, 1.0)
        src = inspect.getsource(preview_cache.build_proxy_entry)
        self.assertIn('meta["proxy_scale"]', src)
        self.assertIn('"proxy_scale"', inspect.getsource(preview_cache._bundle_metadata))
        self.assertIn('proxy_scale=float(metadata.get("proxy_scale", 1.0))', inspect.getsource(preview_cache._bundle_from_cache))
        # pre-v15 entries have no proxy_scale: the version bump invalidates them
        self.assertGreaterEqual(preview_cache.PREVIEW_CACHE_VERSION, 15)

    def test_resident_map_is_charged_to_the_band_budget(self) -> None:
        from dngscan import render

        plain = render._optics_band_rows(6000)
        charged = render._optics_band_rows(6000, reserved_mib=200.0)
        self.assertLess(charged, plain)
        src = inspect.getsource(render._prepare_spatial_pass1)
        self.assertIn("_optics_band_rows(w, _retained_map_mib(chroma_map))", src)
        self.assertEqual(render._retained_map_mib(None), 0.0)
        self.assertAlmostEqual(render._retained_map_mib(np.zeros((1024, 256, 1), np.float32)), 1.0)


if __name__ == "__main__":
    unittest.main()
