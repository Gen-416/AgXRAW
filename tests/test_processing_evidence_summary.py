# SPDX-License-Identifier: GPL-3.0-or-later
"""Generic rendering remains distinct from physical-noise qualification."""
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan.analysis import analyze
from dngscan.auto_ev import compute_auto_ev
from dngscan.hdr_agx_plan import scene_headroom_ev
from dngscan.noise_model import NoiseModel
from dngscan.raw_io import load_raw
from dngscan.render import render_output_u8
from dngscan.report import csv_row, print_report, processing_evidence_summary, summary_lines
from dngscan.tone import build_render_plan
from tests.test_pipeline_corrections import write_sensor_dng


def bundle(**changes):
    values = dict(raw_image=np.zeros((2, 2), np.uint16), evidence_provider="libraw",
                  scene_reliability_source="sensor-spatial", chroma_nr_status="disabled",
                  chroma_nr_reason=None)
    values.update(changes)
    return SimpleNamespace(**values)


def analysis(model=None, *, channel_ids=(0, 1, 2, 3)):
    return SimpleNamespace(noise_model=model, noise_evidence_status="unavailable",
                           channel_ids=list(channel_ids))


class ProcessingEvidenceSummaryTests(unittest.TestCase):
    def test_missing_model_keeps_sensor_hdr_authority(self):
        source = bundle()
        evidence = analysis(NoiseModel())
        scene = SimpleNamespace(reliability_source="sensor-spatial", reliable_tail_ev_p9999=6.)
        earned = scene_headroom_ev(scene)
        result = processing_evidence_summary(source, evidence, scene)
        self.assertEqual(result["mode"], "general-image")
        self.assertEqual(result["label"], "通用成像（噪声未标定）")
        self.assertEqual(result["sensor_evidence"], "available")
        self.assertEqual(result["hdr_reliability_source"], "sensor-spatial")
        self.assertGreater(earned, 1.)
        self.assertEqual(scene_headroom_ev(scene), earned)

    def test_released_mosaic_does_not_revoke_persisted_sensor_evidence(self):
        model = NoiseModel(status="valid", source="DNG NoiseProfile")
        evidence = analysis(model)
        resident = processing_evidence_summary(bundle(), evidence)
        released = processing_evidence_summary(bundle(raw_image=None), evidence)
        self.assertEqual(released, resident)
        self.assertEqual(released["sensor_evidence"], "available")
        self.assertEqual(released["label"], "文件噪声声明辅助成像")

    def test_rejected_and_unresolved_reasons_remain_distinct(self):
        for status, reason in (("rejected", "file-libraw-raw-geometry-mismatch"),
                               ("unresolved", "read-noise-unresolved")):
            with self.subTest(status=status):
                model = NoiseModel(status=status, source="user calibration", reason=reason)
                result = processing_evidence_summary(bundle(), analysis(model))
                self.assertEqual(result["mode"], "general-image")
                self.assertEqual(result["label"], "通用成像（噪声模型受限）")
                self.assertEqual((result["noise_status"], result["noise_reason"]), (status, reason))
                self.assertEqual(result["noise_source"], model.source)

    def test_file_and_matched_models_have_explicit_sources(self):
        for source, label in (("DNG NoiseProfile", "文件噪声声明辅助成像"),
                              ("User JPTC", "匹配噪声模型辅助成像")):
            with self.subTest(source=source):
                model = NoiseModel(status="valid", source=source, reason="file-declared-model")
                result = processing_evidence_summary(bundle(), analysis(model))
                self.assertEqual(result["mode"], "noise-model-assisted")
                self.assertEqual(result["label"], label)
                self.assertEqual(result["noise_source"], source)

    def test_valid_model_does_not_claim_optional_denoising_was_applied(self):
        model = NoiseModel(status="valid", source="DNG NoiseProfile")
        source = bundle(chroma_nr_status="skipped", chroma_nr_reason="Apple decoder covariance is not calibrated")
        result = processing_evidence_summary(source, analysis(model))
        self.assertEqual(result["mode"], "noise-model-assisted")
        self.assertEqual(result["chroma_nr_status"], "skipped")
        self.assertEqual(result["chroma_nr_reason"], source.chroma_nr_reason)
        self.assertIn("仍取决于解码传递", result["detail"])

    def test_spectral_constraint_and_fallback_provenance_are_preserved(self):
        model = NoiseModel(status="valid", source="DNG NoiseProfile",
                           correlation="measured-spectral-imbalance", spectral_ratios={"h": .1},
                           fallback_source="User JPTC", fallback_reason="read-noise-unresolved")
        result = processing_evidence_summary(bundle(), analysis(model))
        self.assertEqual(result["mode"], "noise-model-assisted")
        self.assertIn("保留 HDR 噪声限制", result["detail"])
        self.assertEqual(result["noise_correlation"], model.correlation)
        self.assertEqual(result["noise_fallback_reason"], model.fallback_reason)
        self.assertEqual(result["noise_fallback_source"], model.fallback_source)
        self.assertEqual(result["spectral_ratios"], {"h": .1})
        result["spectral_ratios"]["h"] = 1.
        self.assertEqual(model.spectral_ratios, {"h": .1})

    def test_image_only_fallback_uses_compiled_reliability_source(self):
        source = bundle(raw_image=None, evidence_provider="unavailable")
        scene = SimpleNamespace(reliability_source="decoded-image-estimate")
        result = processing_evidence_summary(source, analysis(channel_ids=()), scene)
        self.assertEqual(result["mode"], "general-image")
        self.assertEqual(result["label"], "图像统计回退")
        self.assertEqual(result["noise_reason"], "sensor-evidence-unavailable")
        self.assertEqual(result["hdr_reliability_source"], "decoded-image-estimate")
        self.assertEqual(result["sensor_evidence"], "unavailable")
        self.assertIn("不声明传感器剪切", result["detail"])

    def test_unknown_camera_real_dng_reports_and_renders_without_a_noise_model(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "generic.dng"
            write_sensor_dng(path, signal=1000)
            source = load_raw(path, scene_half_size=True)
            evidence, _, _ = analyze(source, 4, diagnostics=False)
            self.assertEqual((source.shot_make, source.shot_model), ("Review", "Synthetic"))
            self.assertIsNone(evidence.prior_id)
            self.assertEqual(evidence.noise_model.status, "unavailable")
            plan = build_render_plan(source, evidence, "agx", "srgb")
            auto_ev = compute_auto_ev(source, evidence)
            self.assertTrue(np.isfinite(auto_ev.ev))
            before = render_output_u8(source, evidence, tone_plan=plan)
            source_state, analysis_state = dict(vars(source)), dict(vars(evidence))
            result = processing_evidence_summary(source, evidence, plan.scene)
            self.assertEqual(result["mode"], "general-image")
            for key, value in source_state.items():
                self.assertIs(vars(source)[key], value, key)
            for key, value in analysis_state.items():
                self.assertIs(vars(evidence)[key], value, key)
            report = "\n".join(summary_lines(source, evidence))
            self.assertIn("通用成像（噪声未标定）", report)
            self.assertIn("曝光、色调与导出继续", report)
            row = csv_row(source, evidence, None, tone_plan=plan.tone, scene=plan.scene)
            self.assertEqual(row["processing_evidence_mode"], "general-image")
            self.assertEqual(row["processing_hdr_reliability_source"], plan.scene.reliability_source)
            self.assertEqual(row["noise_model_status"], "unavailable")
            after = render_output_u8(source, evidence, tone_plan=plan)
            np.testing.assert_array_equal(after, before)
            self.assertEqual(before.dtype, np.uint8)
            self.assertTrue(np.isfinite(before).all())
            restricted = replace(evidence, noise_model=NoiseModel(
                status="rejected", reason="raw-noise-reduction-declared",
                noise_reduction_status="applied", correlation="measured-spectral-imbalance",
                spectral_ratios={"v": .1}), noise_evidence_status="model-rejected")
            restricted_row = csv_row(source, restricted, None)
            self.assertEqual(restricted_row["noise_model_status"], "rejected")
            self.assertEqual(restricted_row["noise_model_reason"], "raw-noise-reduction-declared")
            self.assertEqual(restricted_row["noise_model_correlation"], "measured-spectral-imbalance")
            self.assertEqual(restricted_row["noise_model_spectral_ratios"], '{"v": 0.1}')


class DeliveredPrecisionReportTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "generic.dng"
        write_sensor_dng(self.path, signal=1000)
        self.bundle = load_raw(self.path, scene_half_size=True)
        self.analysis, _, _ = analyze(self.bundle, 4, diagnostics=False)
        self.plan = build_render_plan(self.bundle, self.analysis, "agx", "srgb").tone

    def test_report_and_csv_use_final_compiled_scene_qualification(self):
        # A compiled qualification can supersede the decode-time source.
        # Reports must consume it separately from the tone compression plan.
        scene = SimpleNamespace(reliability_source="decoder-support-uncertified")
        self.bundle.scene_reliability_source = "sensor-spatial"
        text = io.StringIO()
        with redirect_stdout(text):
            print_report(self.bundle, self.analysis, None, None, None, 95, "agx", False,
                         tone_plan=self.plan, scene=scene)
        self.assertIn("HDR 证据来源: decoder-support-uncertified", text.getvalue())
        row = csv_row(self.bundle, self.analysis, None, tone_plan=self.plan, scene=scene)
        self.assertEqual(row["processing_hdr_reliability_source"], scene.reliability_source)
        self.assertEqual(row["scene_reliability_source"], scene.reliability_source)

    def report(self, output, info=None):
        text = io.StringIO()
        with redirect_stdout(text):
            print_report(self.bundle, self.analysis, None, None, output, 95, "agx", False,
                         tone_plan=self.plan, export_info=info)
        row = csv_row(self.bundle, self.analysis, None, output, 95, "agx", False,
                      tone_plan=self.plan, export_info=info)
        return text.getvalue(), row

    def test_10bit_sdr_export_facts_override_jpeg_defaults(self):
        from dngscan.heif_encoder import _quantized_band
        ramp = np.repeat(np.linspace(0., 1., 1024, dtype=np.float32)[None, :, None], 3, axis=2)
        self.assertEqual(np.unique(_quantized_band(ramp, 10)).size, 1024)
        actual_path = self.path.with_suffix(".heic")
        info = {"output_path": str(actual_path), "delivery_container": "heic", "bit_depth": 10,
                "chroma_subsampling": "4:2:0", "profile": "Display P3", "delivery_quality": 98,
                "icc_embedded": True, "quantization_dither": "TPDF-10bit-seed0",
                "readback_precision": "float32"}
        text, row = self.report(self.path.with_suffix(".jpg"), info)
        self.assertIn(f"HEIF 图像: {actual_path}", text)
        self.assertIn("HEIF 设置:", text)
        self.assertIn("10-bit Display P3", text)
        self.assertIn("浮点 SDR 母版，编码出口一次 10-bit 量化，TPDF 抖动", text)
        self.assertIn("色度采样=4:2:0", text)
        self.assertIn("质量=98；ICC=已嵌入", text)
        self.assertNotIn("8-bit", text)
        self.assertNotIn("JPEG 设置:", text)
        self.assertEqual(row["output_path"], str(actual_path))
        self.assertEqual((row["output_container"], row["output_bit_depth"],
                          row["output_chroma_subsampling"], row["output_profile"]),
                         ("heic", 10, "4:2:0", "Display P3"))
        # Legacy requested-setting columns remain stable for older CSV readers.
        self.assertEqual(row["jpeg_quality"], 95)

    def test_10bit_hdr_base_reports_its_float_master_without_inventing_dither(self):
        info = {"delivery_container": "heic", "bit_depth": 10, "has_iso_gainmap": True,
                "sdr_master_precision": "float32", "chroma_subsampling": "4:4:4",
                "profile": "Display P3", "delivery_quality": 100}
        text, row = self.report(self.path.with_suffix(".heic"), info)
        self.assertIn("SDR base=10-bit Display P3", text)
        self.assertIn("浮点 SDR 母版，编码出口一次 10-bit 量化", text)
        self.assertNotIn("TPDF", text)
        self.assertEqual(row["output_bit_depth"], 10)

    def test_heif_suffix_without_verified_depth_does_not_claim_eight_bits(self):
        text, row = self.report(self.path.with_suffix(".heif"))
        self.assertIn("HEIF 设置:", text)
        self.assertIn("位深未报告", text)
        self.assertNotIn("8-bit", text)
        self.assertNotIn("TPDF", text)
        self.assertEqual(row["output_container"], "heic")
        self.assertEqual(row["output_bit_depth"], "")
        self.assertEqual(row["output_chroma_subsampling"], "")

    def test_legacy_jpeg_call_preserves_its_actual_eight_bit_contract(self):
        text, row = self.report(self.path.with_suffix(".jpg"))
        self.assertIn("JPEG 设置:", text)
        self.assertIn("8-bit", text)
        self.assertIn("TPDF 抖动", text)
        self.assertEqual(row["output_container"], "jpeg")
        self.assertEqual(row["output_bit_depth"], 8)
        self.assertEqual(row["output_chroma_subsampling"], "4:4:4")


if __name__ == "__main__":
    unittest.main()
