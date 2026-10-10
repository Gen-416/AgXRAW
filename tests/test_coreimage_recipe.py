# SPDX-License-Identifier: GPL-3.0-or-later
"""Decode recipe evidence, stable configuration scope, and retryable runtime probes."""
import hashlib
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from dngscan import coreimage_decode as ci


class Filter:
    def __init__(self, *, baseline=1.25, neutral=5500.):
        self.values = {"decoderVersion": "8", "scaleFactor": 1., "baselineExposure": baseline,
                       "boostAmount": 1., "boostShadowAmount": 1., "shadowBias": 5., "exposure": 0.,
                       "neutralTemperature": neutral, "neutralTint": 10., "extendedDynamicRangeAmount": 1.,
                       "luminanceNoiseReductionAmount": .5, "colorNoiseReductionAmount": .5,
                       "detailAmount": .5, "contrastAmount": .5, "sharpnessAmount": .5,
                       "moireReductionAmount": .5, "localToneMapAmount": .5,
                       "gamutMappingEnabled": True, "highlightRecoveryEnabled": False,
                       "lensCorrectionEnabled": False, "draftModeEnabled": False}
        self.missing, self.ignored, self.broken = set(), set(), set()
        self.support = {name: True for name in ("isLuminanceNoiseReductionSupported",
                        "isColorNoiseReductionSupported", "isDetailSupported", "isContrastSupported",
                        "isSharpnessSupported", "isMoireReductionSupported", "isLocalToneMapSupported",
                        "isHighlightRecoverySupported", "isLensCorrectionSupported")}

    def __getattr__(self, name):
        if name in self.missing:
            raise AttributeError(name)
        if name in self.support:
            return lambda: self.support[name]
        if name.startswith("set") and name.endswith("_"):
            key = name[3].lower() + name[4:-1]
            if key in self.values:
                def setter(value):
                    if name in self.broken:
                        raise RuntimeError("injected write failure")
                    if name not in self.ignored:
                        self.values[key] = value
                return setter
        key = name[2].lower() + name[3:] if name.startswith("is") else name
        if key in self.values:
            return lambda: self.values[key]
        raise AttributeError(name)

    def outputImage(self):
        extent = SimpleNamespace(origin=SimpleNamespace(x=11., y=13.),
                                 size=SimpleNamespace(width=32., height=24.))
        return SimpleNamespace(extent=lambda: extent)


def configure(filt=None, **kwargs):
    return ci.configure_linear_filter(filt or Filter(), version=kwargs.pop("version", "9"),
                                     scale_factor=kwargs.pop("scale_factor", .25), **kwargs)


def recipe(cfg, **overrides):
    args = dict(cfg=cfg, requested="auto", offered=("9", "9.dng", "8"), failures=[], half_size=True,
                scale_compensation=1., native_extent={"status": "read", "xywh": [11., 13., 32., 24.]},
                rendered_shape=(24, 32), output_shape=(24, 32), target_shape=None,
                source={"status": "verified-through-decode", "sha256": "a" * 64},
                context={"cache_intermediates": True, "memory_limit_mb": None,
                         "scope": "CIContext-options-not-proof-of-execution-device"})
    args.update(overrides)
    with patch.object(ci, "decoder_runtime_id", return_value="macOS test-build"), \
         patch.object(ci, "decoder_hardware_id", return_value={"architecture": "arm64", "model": "fixture"}):
        return ci._build_decode_recipe(**args)


class ControlEvidenceTests(unittest.TestCase):
    def test_zero_legacy_getter_does_not_claim_RAW9_internal_NR_is_disabled(self):
        cfg = configure()
        self.assertTrue(cfg["color_noise_cleared"])
        self.assertIn("parameter-zero-only", cfg["color_noise_cleared_semantics"])
        self.assertEqual(cfg["integrated_demosaic_denoise"], "RAW9-joint-model-not-disabled")
        color = cfg["controls"]["colorNoiseReductionAmount"]
        self.assertEqual(color["setter"]["status"], "applied")
        self.assertEqual(color["getter"]["after"]["value"], 0.)
        self.assertEqual(color["effective"]["status"], "not-applicable")
        self.assertFalse(color["effective"]["processing_effect_verified"])

    def test_required_control_missing_or_ignored_is_not_verified(self):
        for missing in (True, False):
            filt = Filter()
            if missing:
                filt.missing.update(("setBoostAmount_", "boostAmount"))
            else:
                filt.ignored.add("setBoostAmount_")
            cfg = configure(filt)
            self.assertEqual(cfg["linear_access"]["status"], "parameter-state-unverified")
            self.assertIn("boostAmount", cfg["linear_access"]["unverified_controls"])
            item = cfg["controls"]["boostAmount"]
            self.assertEqual(item["effective"]["status"], "unconfirmed" if missing else "parameter-mismatch")

    def test_failed_write_and_unsupported_are_distinct(self):
        filt = Filter()
        filt.broken.add("setBaselineExposure_")
        filt.support["isLocalToneMapSupported"] = False
        cfg = configure(filt)
        self.assertFalse(cfg["baseline_exposure_cleared"])
        self.assertEqual(cfg["controls"]["baselineExposure"]["setter"]["status"], "error")
        self.assertEqual(cfg["controls"]["localToneMapAmount"]["setter"]["status"], "skipped-unsupported")
        self.assertEqual(cfg["controls"]["localToneMapAmount"]["applicability"], "unsupported-by-instance")
        self.assertNotIn("localToneMapAmount", cfg["linear_access"]["unverified_controls"])

    def test_actual_token_mismatch_is_rejected_and_missing_getter_remains_unknown(self):
        filt = Filter()
        filt.ignored.add("setDecoderVersion_")
        with self.assertRaisesRegex(RuntimeError, "instead of"):
            configure(filt)
        filt = Filter()
        filt.missing.add("decoderVersion")
        cfg = configure(filt)
        self.assertIsNone(cfg["actual_decoder_token"])
        self.assertIn("decoderVersion", cfg["linear_access"]["unverified_controls"])


class ConfigurationIdentityTests(unittest.TestCase):
    def test_identity_ignores_request_history_content_extent_and_file_neutral_baseline(self):
        first = recipe(configure(Filter(baseline=1.25, neutral=5500.)))
        second = recipe(configure(Filter(baseline=-.5, neutral=4200.)), requested="9", offered=("8", "9"),
                        failures=["RAW 9.dng: failed attempt"],
                        source={"status": "verified-through-decode", "sha256": "b" * 64},
                        native_extent={"status": "read", "xywh": [0., 0., 1024., 768.]},
                        rendered_shape=(768, 1024), output_shape=(768, 1024))
        self.assertEqual(first["configuration_id"], second["configuration_id"])
        self.assertNotEqual(first["baseline_exposure"]["authored"], second["baseline_exposure"]["authored"])
        self.assertNotEqual(first["controls"]["neutralTemperature"]["effective"]["value"],
                            second["controls"]["neutralTemperature"]["effective"]["value"])
        self.assertEqual(first["schema"], ci.COREIMAGE_RECIPE_VERSION)
        self.assertEqual(first["configuration"]["project_hot_wb_implementation"], ci.PROJECT_HOT_WB_IMPLEMENTATION)

    def test_effective_token_scale_WB_and_control_state_change_identity(self):
        base = recipe(configure())["configuration_id"]
        variants = [recipe(configure(version="9.dng")), recipe(configure(scale_factor=.5)),
                    recipe(configure(neutral_cct=5500.)), recipe(configure(), scale_compensation=.97)]
        filt = Filter()
        filt.support["isLensCorrectionSupported"] = False
        variants.append(recipe(configure(filt)))
        for variant in variants:
            self.assertNotEqual(base, variant["configuration_id"])

    def test_runtime_and_hardware_are_in_configuration_scope(self):
        cfg = configure()
        first = recipe(cfg)
        with patch.object(ci, "decoder_runtime_id", return_value="different-build"), \
             patch.object(ci, "decoder_hardware_id", return_value={"architecture": "arm64"}):
            args = dict(cfg=cfg, requested="auto", offered=("9",), failures=[], half_size=True,
                        scale_compensation=1., native_extent={}, rendered_shape=(24, 32),
                        output_shape=(24, 32), target_shape=None, source={}, context={})
            second = ci._build_decode_recipe(**args)
        self.assertNotEqual(first["configuration_id"], second["configuration_id"])

    def test_rendered_recipe_hashes_real_source_and_keeps_native_extent(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path = Path(tmp) / "fixture.raw"
            path.write_bytes(b"first captured source")
            stack.enter_context(patch.object(ci, "supported_versions", return_value=("9",)))
            stack.enter_context(patch.object(ci, "_open_filter", side_effect=lambda p: Filter()))
            stack.enter_context(patch.object(ci, "preview_scale_factor", return_value=.25))
            stack.enter_context(patch.object(ci, "_render_linear_rec2020", return_value=np.ones((24, 32, 3), np.float16)))
            first_pixels, first = ci.decode_scene_rec2020(path, half_size=True, version="auto")
            path.write_bytes(b"another captured source")
            _, second = ci.decode_scene_rec2020(path, half_size=True, version="9")
        self.assertEqual(first_pixels.dtype, np.float16)
        self.assertEqual(first["recipe"]["source"]["sha256"], hashlib.sha256(b"first captured source").hexdigest())
        self.assertNotEqual(first["recipe"]["source"]["sha256"], second["recipe"]["source"]["sha256"])
        self.assertEqual(first["recipe"]["configuration_id"], second["recipe"]["configuration_id"])
        self.assertEqual(first["recipe"]["geometry"]["native_extent"]["status"], "unavailable")

    def test_extent_is_recorded_from_the_image_actually_rendered_without_second_getter(self):
        filt = Filter()
        filt.outputImage = Mock(side_effect=[Filter().outputImage(), None])
        quartz = SimpleNamespace(kCGColorSpaceExtendedLinearITUR_2020="ext2020", kCIFormatRGBAh="RGBAh",
                                 CGColorSpaceCreateWithName=lambda name: object(),
                                 CGRectMake=lambda *args: args)
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            path = Path(tmp) / "fixture.raw"
            path.write_bytes(b"captured source")
            stack.enter_context(patch.object(ci, "supported_versions", return_value=("9",)))
            stack.enter_context(patch.object(ci, "_open_filter", return_value=filt))
            stack.enter_context(patch.object(ci, "preview_scale_factor", return_value=.25))
            stack.enter_context(patch.object(ci, "_require_quartz", return_value=quartz))
            stack.enter_context(patch.object(ci, "_render_context", return_value=MagicMock()))
            pixels, info = ci.decode_scene_rec2020(path, half_size=True)
        self.assertEqual(filt.outputImage.call_count, 1)
        self.assertEqual(info["recipe"]["geometry"]["native_extent"]["xywh"], [11., 13., 32., 24.])
        self.assertEqual(pixels.shape, (24, 32, 3))


class ContextAndRuntimeTests(unittest.TestCase):
    def test_context_without_new_memory_constant_reports_intended_error(self):
        quartz = SimpleNamespace(kCIContextCacheIntermediates="cache",
                                 CIContext=SimpleNamespace(contextWithOptions_=lambda opts: None))
        with patch.dict(ci._CONTEXTS, {}, clear=True), patch.dict(ci._CONTEXT_RECIPES, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "returned None"):
                ci._render_context(quartz, interactive=False)

    def test_memory_hint_rejection_retries_without_unguarded_constant(self):
        context = object()
        captured = []
        factory = Mock(side_effect=lambda options: captured.append(dict(options)) or
                       (None if len(captured) == 1 else context))
        quartz = SimpleNamespace(kCIContextCacheIntermediates="cache", kCIContextMemoryLimit="memory",
                                 CIContext=SimpleNamespace(contextWithOptions_=factory))
        with patch.dict(ci._CONTEXTS, {}, clear=True), patch.dict(ci._CONTEXT_RECIPES, {}, clear=True):
            self.assertIs(ci._render_context(quartz, interactive=False), context)
            self.assertTrue(ci._CONTEXT_RECIPES[False]["memory_hint_rejected_then_retried"])
        self.assertEqual(captured[0], {"cache": False, "memory": 1024})
        self.assertEqual(captured[1], {"cache": False})

    def test_runtime_failure_expires_and_success_is_cached_without_sleep(self):
        quartz = MagicMock()
        quartz.CGColorSpaceCreateWithName.return_value = object()
        context = MagicMock()
        clock = [100.]
        with patch.dict(sys.modules, {"Quartz": quartz, "Foundation": MagicMock()}), \
             patch.dict(ci._RUNTIME_AVAILABLE, {}, clear=True), patch.dict(ci._RUNTIME_FAILED_AT, {}, clear=True), \
             patch.dict(ci._CONTEXTS, {False: object()}, clear=True), patch.dict(ci._CONTEXT_RECIPES, {}, clear=True), \
             patch.object(ci.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(ci, "available", return_value=False) as available, \
             patch.object(ci, "_render_context", return_value=context) as build:
            self.assertFalse(ci.runtime_available())
            available.return_value = True
            clock[0] = 159.999
            self.assertFalse(ci.runtime_available())
            self.assertEqual(available.call_count, 1)
            clock[0] = 160.
            self.assertTrue(ci.runtime_available())
            self.assertNotIn(False, ci._CONTEXTS)
            clock[0] = 1000.
            self.assertTrue(ci.runtime_available())
            self.assertEqual(build.call_count, 1)
            available.return_value = False
            self.assertFalse(ci.runtime_available(retry=True))
            self.assertEqual(available.call_count, 3)


if __name__ == "__main__":
    unittest.main()
