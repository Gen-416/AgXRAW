# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the optional Core Image (CIRAWFilter) scene decoder."""
from __future__ import annotations

import unittest
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dngscan import coreimage_decode
from dngscan.models import RawBundle
from dngscan.retreat import clip_masks_for_shape, resize_clip_masks

PICTURES = Path.home() / "Pictures"
SIGMA_DNG = PICTURES / "AgXRAW样张" / "_SDI0150.DNG"
IPHONE_DNG = PICTURES / "Original RAW 26-07-12 182506394.dng"
SIGMA_VERTICAL_DNG = PICTURES / "AgXRAW样张" / "_SDI0165.DNG"
FUJI_RAF = PICTURES / "DSCF0614.RAF"


def _skip_unless_available() -> None:
    if not coreimage_decode.available():
        raise unittest.SkipTest("Core Image / CIRAWFilter unavailable")


class CoreImageDecodeImportTests(unittest.TestCase):
    def test_available_never_raises(self) -> None:
        # Must be False-safe on any platform.
        value = coreimage_decode.available()
        self.assertIsInstance(value, bool)

    def test_import_is_side_effect_free(self) -> None:
        # Re-importing must not raise even when Quartz is absent.
        import importlib

        importlib.reload(coreimage_decode)
        self.assertIsInstance(coreimage_decode.available(), bool)

    def test_signed_half_handoff_preserves_extended_range(self) -> None:
        source = np.asarray(
            [[[-0.25, 0.18, 2.5], [np.nan, np.inf, -np.inf]]], dtype=np.float32
        )
        scene, scale = coreimage_decode.scene_float_to_half(source)
        self.assertEqual(scene.dtype, np.float16)
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(float(scene[0, 0, 0]), -0.25, places=3)
        self.assertAlmostEqual(float(scene[0, 0, 2]), 2.5, places=3)
        self.assertEqual(float(scene[0, 1, 0]), 0.0)
        self.assertGreater(float(scene[0, 1, 1]), 1.0)
        self.assertLess(float(scene[0, 1, 2]), 0.0)

    def test_preview_scale_targets_1280_long_edge(self) -> None:
        class Size:
            width = 6000
            height = 4000

        class Filter:
            @staticmethod
            def nativeSize():
                return Size()

        self.assertAlmostEqual(
            coreimage_decode.preview_scale_factor(Filter()), 1280.0 / 6000.0
        )

    def test_linear_configuration_externalizes_baseline_and_disables_edr(self) -> None:
        class Filter:
            def __init__(self) -> None:
                self.version = "8"
                self.scale = 1.0
                self.baseline = 1.25
                self.edr = 1.0
                self.shadow = 5.0

            def setDecoderVersion_(self, value):
                self.version = value

            def decoderVersion(self):
                return self.version

            def setScaleFactor_(self, value):
                self.scale = value

            def scaleFactor(self):
                return self.scale

            def baselineExposure(self):
                return self.baseline

            def setBaselineExposure_(self, value):
                self.baseline = value

            def extendedDynamicRangeAmount(self):
                return self.edr

            def setExtendedDynamicRangeAmount_(self, value):
                self.edr = value

            def shadowBias(self):
                return self.shadow

            def setShadowBias_(self, value):
                self.shadow = value

        filt = Filter()
        cfg = coreimage_decode.configure_linear_filter(
            filt, version="9", scale_factor=0.25
        )
        self.assertEqual(cfg["baseline_exposure_authored"], 1.25)
        self.assertEqual(cfg["baseline_exposure_applied"], 0.0)
        self.assertTrue(cfg["baseline_exposure_cleared"])
        self.assertEqual(cfg["extended_dynamic_range_amount"], 0.0)
        self.assertEqual(cfg["shadow_bias"], 0.0)


class CoreImageVersionTests(unittest.TestCase):
    def test_auto_prefers_nine(self) -> None:
        chosen = coreimage_decode.resolve_decoder_version(
            "auto", ("6.dng", "7", "7.dng", "8", "8.dng", "9", "9.dng")
        )
        self.assertEqual(chosen, "9")

    def test_auto_falls_back(self) -> None:
        chosen = coreimage_decode.resolve_decoder_version("auto", ("7", "8"))
        self.assertEqual(chosen, "8")

    def test_explicit_unsupported_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            coreimage_decode.resolve_decoder_version("9", ("7", "8"))

    def test_raw9_probe_accepts_dng_version_token(self) -> None:
        with (
            patch.object(coreimage_decode, "available", return_value=True),
            patch.object(
                coreimage_decode,
                "supported_versions",
                return_value=("8", "8.dng", "9.dng"),
            ),
        ):
            result = coreimage_decode.probe_raw9_support(Path("camera.dng"))
        self.assertTrue(result["raw9_supported"])
        self.assertIsNone(result["fallback_version"])

    def test_raw9_probe_reports_explicit_legacy_fallback(self) -> None:
        with (
            patch.object(coreimage_decode, "available", return_value=True),
            patch.object(
                coreimage_decode,
                "supported_versions",
                return_value=("7.dng", "8.dng"),
            ),
        ):
            result = coreimage_decode.probe_raw9_support(Path("older.dng"))
        self.assertFalse(result["raw9_supported"])
        self.assertEqual(result["fallback_version"], "8")
        self.assertEqual(result["versions_offered"], ("7.dng", "8.dng"))

    def test_raw9_probe_contains_open_error(self) -> None:
        with (
            patch.object(coreimage_decode, "available", return_value=True),
            patch.object(
                coreimage_decode,
                "supported_versions",
                side_effect=RuntimeError("unsupported container"),
            ),
        ):
            result = coreimage_decode.probe_raw9_support(Path("not-raw.bin"))
        self.assertFalse(result["raw9_supported"])
        self.assertIn("unsupported container", str(result["error"]))

    def test_opcode_reader_follows_inline_single_subifd_offset(self) -> None:
        # Little-endian classic TIFF: IFD0 contains one inline SubIFD offset; that SubIFD
        # contains an external OpcodeList2 with one WarpRectilinear record.
        ifd0_offset = 8
        subifd_offset = 26
        payload_offset = 44
        payload = struct.pack(">IIIII", 1, 1, 1, 0, 0)
        data = bytearray(payload_offset + len(payload))
        data[:8] = b"II" + struct.pack("<H", 42) + struct.pack("<I", ifd0_offset)
        data[8:10] = struct.pack("<H", 1)
        data[10:22] = struct.pack("<HHII", 0x014A, 4, 1, subifd_offset)
        data[22:26] = struct.pack("<I", 0)
        data[26:28] = struct.pack("<H", 1)
        data[28:40] = struct.pack("<HHII", 0xC741, 7, len(payload), payload_offset)
        data[40:44] = struct.pack("<I", 0)
        data[payload_offset:] = payload
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subifd.dng"
            path.write_bytes(data)
            result = coreimage_decode.read_dng_opcodes(path)
        self.assertTrue(result["parsed"])
        self.assertEqual(result["ids"], (1,))
        self.assertTrue(result["geometry"])


@unittest.skipUnless(coreimage_decode.available(), "Core Image unavailable")
class CoreImageLiveTests(unittest.TestCase):
    def test_color_noise_default_cleared(self) -> None:
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        from Foundation import NSURL
        import Quartz

        filt = Quartz.CIRAWFilter.alloc().initWithImageURL_(
            NSURL.fileURLWithPath_(str(SIGMA_DNG))
        )
        self.assertGreater(float(filt.colorNoiseReductionAmount()), 0.0)
        coreimage_decode.configure_linear_filter(filt, version="9", scale_factor=0.1)
        self.assertAlmostEqual(float(filt.colorNoiseReductionAmount()), 0.0, places=6)

    def test_linearity_contract(self) -> None:
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        a, _ = coreimage_decode.decode_scene_rec2020(
            SIGMA_DNG, half_size=True, version="auto", exposure=0.0, scale_compensation=1.0
        )
        b, _ = coreimage_decode.decode_scene_rec2020(
            SIGMA_DNG, half_size=True, version="auto", exposure=1.0, scale_compensation=1.0
        )
        mask = a.mean(axis=2) > 0.01
        if not np.any(mask):
            raise unittest.SkipTest("no midtone pixels for linearity check")
        ratio = float(np.median(b[mask] / np.maximum(a[mask], 1e-8)))
        self.assertAlmostEqual(ratio, 2.0, delta=0.02)

    def test_externalized_baseline_matches_decoder_applied_gain(self) -> None:
        """Moving BaselineExposure out of CIRAWFilter must preserve scene intent."""
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")

        linear_filter = coreimage_decode._open_filter(SIGMA_DNG)
        linear_cfg = coreimage_decode.configure_linear_filter(
            linear_filter, version="9", scale_factor=0.2
        )
        baseline = linear_cfg["baseline_exposure_authored"]
        if baseline is None or abs(float(baseline)) <= 1e-6:
            raise unittest.SkipTest("sample has no non-zero BaselineExposure")
        linear = coreimage_decode._render_linear_rec2020(
            linear_filter, interactive=True
        ).astype(np.float32)

        baked_filter = coreimage_decode._open_filter(SIGMA_DNG)
        coreimage_decode.configure_linear_filter(
            baked_filter, version="9", scale_factor=0.2
        )
        baked_filter.setBaselineExposure_(float(baseline))
        baked = coreimage_decode._render_linear_rec2020(
            baked_filter, interactive=True
        ).astype(np.float32)

        restored = linear * np.float32(2.0 ** float(baseline))
        mask = np.all(np.isfinite(restored), axis=2) & (np.max(np.abs(baked), axis=2) > 0.01)
        if not np.any(mask):
            raise unittest.SkipTest("no valid scene samples for baseline comparison")
        relative = np.abs(restored[mask] - baked[mask]) / np.maximum(
            np.abs(baked[mask]), 0.01
        )
        self.assertLess(float(np.median(relative)), 0.001)
        self.assertLess(float(np.percentile(relative, 99.0)), 0.01)

    def test_preview_decode_is_signed_half_at_proxy_resolution(self) -> None:
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        rgb, info = coreimage_decode.decode_scene_rec2020(
            SIGMA_DNG, half_size=True, version="auto", scale_compensation=1.0
        )
        self.assertEqual(rgb.dtype, np.float16)
        self.assertLessEqual(max(rgb.shape[:2]), coreimage_decode.COREIMAGE_PREVIEW_LONG_EDGE + 1)
        self.assertGreater(float(np.max(rgb)), 1.0)
        self.assertTrue(bool(info["highlight_recovery"]))
        self.assertTrue(bool(info["lens_correction"]))
        self.assertTrue(bool(info["baseline_exposure_cleared"]))
        self.assertAlmostEqual(float(info["baseline_exposure_applied"]), 0.0, places=6)
        self.assertAlmostEqual(
            float(info["extended_dynamic_range_amount"]), 0.0, places=6
        )

    def test_vertical_raw9_proxy_is_already_upright(self) -> None:
        _skip_unless_available()
        if not SIGMA_VERTICAL_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_VERTICAL_DNG}")
        from dngscan.raw_io import load_raw

        bundle = load_raw(SIGMA_VERTICAL_DNG, scene_half_size=True, decoder="coreimage")
        height, width = bundle.scene_rec2020_render.shape[:2]
        self.assertGreater(height, width)
        self.assertIn(bundle.orientation_flip, (5, 6, 7, 8))

    def test_separate_pipeline_drops_cfa_masks_and_scales(self) -> None:
        """Strict Core Image pipeline: no per-pixel CFA evidence, correct scale.

        Core Image executes the file's DNG opcodes (WarpRectilinear here), so its frame
        is a nonlinear warp of LibRaw's — corners measured ~70 px away on this capture.
        Reusing LibRaw masks would put clip retreat on the wrong pixels, so the bundle
        must carry none, while the aggregate LibRaw facts stay available.
        """
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        from dngscan.raw_io import baseline_exposure_gain, load_raw

        libraw = load_raw(SIGMA_DNG, scene_half_size=True, decoder="libraw")
        ci_bundle = load_raw(SIGMA_DNG, scene_half_size=True, decoder="coreimage")
        self.assertIsNotNone(libraw.clip_masks)
        self.assertIsNone(ci_bundle.clip_masks)
        self.assertEqual(ci_bundle.scene_decoder, "coreimage")
        self.assertEqual(ci_bundle.scene_highlight_mode, "reconstruct")
        self.assertEqual(ci_bundle.scene_rec2020_render.dtype, np.float16)
        self.assertIn("WarpRectilinear", ci_bundle.scene_opcode_names)
        # Aggregate (geometry-free) RAW facts survive: same mosaic, same levels.
        self.assertEqual(int(ci_bundle.white_level), int(libraw.white_level))
        self.assertEqual(list(ci_bundle.black_levels), list(libraw.black_levels))
        # The float16 buffer stays directly scene-linear. scene_scale carries both the
        # externalized file BaselineExposure and the default aligned-mode scalar.
        self.assertEqual(ci_bundle.scene_scale_mode, "aligned")
        self.assertIsNone(ci_bundle.scene_align_error)
        self.assertNotAlmostEqual(ci_bundle.scene_align_factor, 1.0, places=3)
        self.assertFalse(ci_bundle.baseline_exposure_baked_in)
        baseline_gain = baseline_exposure_gain(ci_bundle.baseline_exposure)
        self.assertAlmostEqual(
            ci_bundle.scene_scale,
            1.0 / (baseline_gain * ci_bundle.scene_align_factor),
            places=6,
        )

    def test_alignment_puts_both_decoders_on_one_exposure_scale(self) -> None:
        """The whole point of the measured alignment: a body median that agrees.

        Before it, the two decoders' body medians sat 0.12 EV apart on this Sigma fp
        frame and up to 0.85 EV apart on iPhone, because LibRaw's 1.0 is sensor
        saturation while Apple's is an estimate of *this frame's* diffuse white. The
        bound below is well inside the visible threshold and roughly 4x the residual
        measured at full resolution, leaving room for the half-size proxy.
        """
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        from dngscan.raw_io import load_raw

        libraw = load_raw(SIGMA_DNG, scene_half_size=True, decoder="libraw")
        ci_bundle = load_raw(SIGMA_DNG, scene_half_size=True, decoder="coreimage")

        def body_median(bundle) -> float:
            arr = np.asarray(bundle.scene_rec2020_render, dtype=np.float32) / float(
                bundle.scene_scale
            )
            y = 0.2627 * arr[:, :, 0] + 0.6780 * arr[:, :, 1] + 0.0593 * arr[:, :, 2]
            return float(np.median(y[(y > 0.01) & (y < 0.5)]))

        delta_ev = float(np.log2(body_median(ci_bundle) / body_median(libraw)))
        self.assertLess(abs(delta_ev), 0.15)

    def test_alignment_reference_carries_the_dng_dark_field(self) -> None:
        """Batch-8 P1: the aligned-mode LibRaw reference skipped the DNG's
        dark-field opcodes (pre-demosaic GainMap, post-render
        FixVignetteRadial), so lens shading read as a decoder exposure
        difference — measured up to +0.88 EV on iPhone Standard RAW, whose
        files carry FixVignetteRadial. The reference now reuses the production
        decode context; the decoder pair must agree on this iPhone frame.
        The bound is looser than the Sigma one: Apple's iPhone rendition
        differs more from LibRaw's than its Sigma rendition does (post-fix
        residual measured 0.152 EV on this frame at the half-size proxy,
        down from 0.62+ before the reference carried the dark field)."""
        _skip_unless_available()
        if not IPHONE_DNG.is_file():
            raise unittest.SkipTest(f"missing {IPHONE_DNG}")
        from dngscan.raw_io import load_raw

        libraw = load_raw(IPHONE_DNG, scene_half_size=True, decoder="libraw")
        ci_bundle = load_raw(IPHONE_DNG, scene_half_size=True, decoder="coreimage")
        self.assertEqual(ci_bundle.scene_scale_mode, "aligned")
        self.assertIsNone(ci_bundle.scene_align_error)

        def body_median(bundle) -> float:
            arr = np.asarray(bundle.scene_rec2020_render, dtype=np.float32) / float(
                bundle.scene_scale
            )
            y = 0.2627 * arr[:, :, 0] + 0.6780 * arr[:, :, 1] + 0.0593 * arr[:, :, 2]
            return float(np.median(y[(y > 0.01) & (y < 0.5)]))

        delta_ev = float(np.log2(body_median(ci_bundle) / body_median(libraw)))
        self.assertLess(abs(delta_ev), 0.2)

    def test_full_resolution_production_path_renders(self) -> None:
        """Exercise the resolution the exporter actually uses.

        The earlier gate passed at half size and rejected every full-size export; any
        future check must be verified where production runs.
        """
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        import numpy as np

        from dngscan.analysis import analyze
        from dngscan.raw_io import load_raw
        from dngscan.render import render_output_u8
        from dngscan.tone import build_render_plan

        bundle = load_raw(SIGMA_DNG, scene_half_size=False, decoder="coreimage")
        analysis, _, _ = analyze(bundle, 4, diagnostics=False, gamut_names=("P3",))
        plan = build_render_plan(bundle, analysis, "agx", "p3")
        rgb = render_output_u8(bundle, analysis, "p3", plan)
        self.assertEqual(rgb.shape[:2], bundle.scene_rec2020_render.shape[:2])
        self.assertGreater(int(np.asarray(rgb).max()), 32)


    def test_fuji_resolves_without_claiming_v9(self) -> None:
        _skip_unless_available()
        if not FUJI_RAF.is_file():
            raise unittest.SkipTest(f"missing {FUJI_RAF}")
        offered = coreimage_decode.supported_versions(FUJI_RAF)
        self.assertNotIn("9", {_normalize(v) for v in offered})
        with self.assertRaises(RuntimeError):
            coreimage_decode.resolve_decoder_version("9", offered)
        _, info = coreimage_decode.decode_scene_rec2020(
            FUJI_RAF, half_size=True, version="auto"
        )
        self.assertEqual(coreimage_decode._normalize_version_token(info["version"]), "8")


def _normalize(token: str) -> str:
    return coreimage_decode._normalize_version_token(token)


class LoadRawDecoderGuardTests(unittest.TestCase):
    def test_daylight_uses_project_hot_wb_after_fixed_coreimage_decode(self) -> None:
        from dngscan.raw_io import load_raw

        if not coreimage_decode.available():
            raise unittest.SkipTest("Core Image unavailable")
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        bundle = load_raw(
            SIGMA_DNG,
            scene_half_size=True,
            decoder="coreimage",
            wb_mode="daylight",
        )
        self.assertEqual(bundle.wb_mode, "daylight")
        self.assertEqual(bundle.decode_wb, bundle.camera_wb)
        self.assertEqual(bundle.applied_wb, bundle.daylight_wb)


if __name__ == "__main__":
    unittest.main()


class DecoderGuardTests(unittest.TestCase):
    def test_gated_core_rejected_with_coreimage(self) -> None:
        """gated means "RAW evidence gates the colour path"; that evidence does not
        exist on the Core Image pipeline, so the combination must be refused rather
        than silently degraded."""
        from dngscan.cli import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["photo.dng", "--jpeg", "out.jpg",
                        "--decoder", "coreimage", "--tone-core", "gated"])

    def test_scale_mode_rejected_on_libraw(self) -> None:
        """The flag only means something on the Core Image path; accepting it silently
        elsewhere would imply an alignment that never happened."""
        from dngscan.cli import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["photo.dng", "--jpeg", "out.jpg", "--coreimage-scale", "unity"])

    def test_scale_modes_are_distinct_and_default_is_aligned(self) -> None:
        """Per-file alignment, Apple-native units, and the legacy fixed fit are separate
        policies; a fixed multiplier must not masquerade as per-file alignment."""
        from dngscan.cli import parse_args

        measured = coreimage_decode.scale_compensation_for_mode("measured")
        unity = coreimage_decode.scale_compensation_for_mode("unity")
        aligned = coreimage_decode.scale_compensation_for_mode("aligned")
        self.assertEqual(unity, 1.0)
        self.assertEqual(aligned, 1.0)
        self.assertNotAlmostEqual(measured, unity, places=3)
        self.assertAlmostEqual(
            measured, 1.0 / coreimage_decode.COREIMAGE_SCALE_MEASURED_RATIO, places=9
        )
        with self.assertRaises(ValueError):
            coreimage_decode.scale_compensation_for_mode("nope")
        args = parse_args(["photo.dng", "--jpeg", "out.jpg", "--decoder", "coreimage"])
        self.assertEqual(args.coreimage_scale, "aligned")

    def test_raw9_normalizes_libraw_only_controls(self) -> None:
        from dngscan.cli import parse_args

        args = parse_args(
            [
                "photo.dng",
                "--jpeg",
                "out.jpg",
                "--decoder",
                "coreimage",
                "--highlight-mode",
                "blend",
                "--demosaic",
                "dcb",
            ]
        )
        self.assertEqual(args.highlight_mode, "reconstruct")
        self.assertEqual(args.demosaic, "auto")

    def test_opcode_reader_is_best_effort(self) -> None:
        """Never fatal: a non-TIFF container just reports nothing."""
        result = coreimage_decode.read_dng_opcodes(Path("/nonexistent/x.raf"))
        self.assertFalse(result["geometry"])
        self.assertEqual(result["names"], ())

    def test_opcode_reader_finds_dng_geometry_opcodes(self) -> None:
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        result = coreimage_decode.read_dng_opcodes(SIGMA_DNG)
        self.assertTrue(result["parsed"])
        self.assertIn("WarpRectilinear", result["names"])
        self.assertTrue(result["geometry"])


class SubjectiveControlTests(unittest.TestCase):
    """Where the line falls between a look control and reconstruction.

    Spatial and subjective controls must be off for a scene-linear decode; the controls
    that reconstruct clipped data must not be, which is why this class asserts both
    directions rather than just "everything is zero".
    """

    def test_sharpening_is_cleared(self) -> None:
        """sharpnessAmount defaults to 0.485 and is a spatial operator. It was inert on
        decoder version 8 but is live on version 9, so it silently started altering the
        buffer once this decoder began requesting RAW 9 — a fresh filter reports 8, so
        version 9 is only ever reached by asking for it."""
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        _, info = coreimage_decode.decode_scene_rec2020(
            SIGMA_DNG, half_size=True, version="auto"
        )
        self.assertIsNotNone(info["sharpness_amount"])
        self.assertAlmostEqual(float(info["sharpness_amount"]), 0.0, places=6)
        self.assertTrue(info["color_noise_cleared"])

    def test_moire_reduction_is_left_at_apples_default(self) -> None:
        """The one look-adjacent control deliberately not cleared.

        moireReductionAmount reads like something to zero. On version 9,
        isMoireReductionSupported is False, so a gated setter would skip — and that
        accidental skip is what kept detail. Forcing 0 is actively harmful: measured at
        full resolution it costs 59.8 % of high-frequency energy, because zero is the
        control's smoothest end rather than "off", while 0.5 and 1.0 render identically
        on the plateau its default sits on. configure_linear_filter must not touch it
        at all, even if a future SDK reports the control as supported.
        """
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        from Foundation import NSURL
        import Quartz

        filt = Quartz.CIRAWFilter.alloc().initWithImageURL_(
            NSURL.fileURLWithPath_(str(SIGMA_DNG))
        )
        # Defaults are DECODER-SPECIFIC (the module's own rule: read after
        # selecting the version). The first cut read the fresh-filter
        # default, which broke when a RawCamera update changed the file's
        # default decoder — decoder 9's own moire default is 0.0 and the
        # policy under test is "configure must not TOUCH the control",
        # not "the default is nonzero".
        filt.setDecoderVersion_("9")
        default = float(filt.moireReductionAmount())
        coreimage_decode.configure_linear_filter(filt, version="9", scale_factor=0.1)
        self.assertAlmostEqual(float(filt.moireReductionAmount()), default, places=6)
        # Policy pin: the module must not expose a path that zeros this control.
        source = Path(coreimage_decode.__file__).read_text(encoding="utf-8")
        self.assertNotIn("setMoireReductionAmount_", source)

    def test_highlight_recovery_stays_on_and_keeps_clipped_highlights_neutral(
        self,
    ) -> None:
        """Highlight recovery is reconstruction, not taste, so it is the one control the
        scene-linear decode leaves enabled. Without it Apple returns clipped highlights
        with green pinned below red and blue, which renders as magenta highlight cores."""
        _skip_unless_available()
        if not SIGMA_DNG.is_file():
            raise unittest.SkipTest(f"missing {SIGMA_DNG}")
        rgb, info = coreimage_decode.decode_scene_rec2020(
            SIGMA_DNG, half_size=True, version="auto"
        )
        self.assertTrue(info["highlight_recovery"])
        headroom = coreimage_decode.scene_headroom(rgb)
        near_top = rgb.max(axis=2) > 0.8 * headroom
        if int(np.count_nonzero(near_top)) < 500:
            raise unittest.SkipTest("frame carries too few clipped highlights to judge")
        mean = rgb[near_top].reshape(-1, 3).mean(axis=0)
        magenta_bias = float(mean[0] + mean[2] - 2.0 * mean[1])
        self.assertLess(abs(magenta_bias), 0.25 * float(mean[1]))
