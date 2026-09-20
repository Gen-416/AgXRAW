# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise staged writer wiring with real metrics and temporary file ownership.

Only the platform/codec/container I/O is stubbed. The writer, gate order, pixel
reductions, failure handling and destination publication all run normally.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dngscan import auto_encode, gainmap
from dngscan.delivery import resolve_delivery_profile


class _SessionSeam:
    """Cache seam for writer tests; real HEIF identity is tested by its owner."""

    def __init__(self, directory):
        self.directory = directory
        self.values = {}
        self.lookups = []
        self.remembered = []

    def primary(self, base, profile):
        donor = self.directory / "session-primary.heic"
        donor.write_text(json.dumps({"primary": f"q{profile.quality}-{profile.chroma}"}))
        return donor, {"encoder": "x265"}

    def metrics(self, path):
        key = json.loads(path.read_text())["primary"]
        self.lookups.append(key)
        metrics = self.values.get(key)
        return None if metrics is None else dict(metrics)

    def remember_metrics(self, path, metrics):
        key = json.loads(path.read_text())["primary"]
        self.remembered.append(key)
        self.values[key] = dict(metrics)


class SearchSdrMetricsTests(unittest.TestCase):
    def test_one_readback_supplies_both_original_measurements(self):
        intended = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
        decoded = intended.copy()
        decoded[::3, 2::3, 0] = np.minimum(decoded[::3, 2::3, 0], 200)
        expected = {
            **gainmap._base_roundtrip_error_arrays(decoded, intended),
            **auto_encode.coding_metrics(decoded, intended),
        }
        with patch.object(gainmap, "read_primary_rgb_u8", return_value=decoded) as read, \
             patch.object(gainmap, "_base_roundtrip_error_arrays",
                          wraps=gainmap._base_roundtrip_error_arrays) as absolute, \
             patch.object(auto_encode, "coding_metrics", wraps=auto_encode.coding_metrics) as coding:
            result = gainmap._search_sdr_metrics(Path("candidate.heic"), intended)
        read.assert_called_once_with(Path("candidate.heic"))
        self.assertIs(absolute.call_args.args[0], decoded)
        self.assertIs(coding.call_args.args[0], decoded)
        self.assertIs(absolute.call_args.args[1], intended)
        self.assertIs(coding.call_args.args[1], intended)
        self.assertEqual(result, expected)

    def test_cache_hit_skips_pixels_and_both_reductions_but_miss_measures(self):
        base = np.full((8, 8, 3), 180, np.uint8)
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            session = _SessionSeam(directory)
            first, auxiliary_variant, different_primary = (
                directory / name for name in ("first.heic", "aux.heic", "other.heic")
            )
            first.write_text(json.dumps({"primary": "same", "aux": 100}))
            auxiliary_variant.write_text(json.dumps({"primary": "same", "aux": 90}))
            different_primary.write_text(json.dumps({"primary": "different", "aux": 90}))
            with patch.object(gainmap, "read_primary_rgb_u8", return_value=base) as read, \
                 patch.object(gainmap, "_base_roundtrip_error_arrays",
                              wraps=gainmap._base_roundtrip_error_arrays) as absolute, \
                 patch.object(auto_encode, "coding_metrics", wraps=auto_encode.coding_metrics) as coding:
                original = gainmap._search_sdr_metrics(first, base, session)
                cached = gainmap._search_sdr_metrics(auxiliary_variant, base, session)
                measured = gainmap._search_sdr_metrics(different_primary, base, session)
            self.assertEqual(original, cached)
            self.assertEqual(cached, measured)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(absolute.call_count, 2)
            self.assertEqual(coding.call_count, 2)
            self.assertEqual(session.lookups, ["same", "same", "different"])
            self.assertEqual(session.remembered, ["same", "different"])


class GainmapStagedWriterTests(unittest.TestCase):
    @contextmanager
    def writer_fixture(self, directory, *, decoded_value=180):
        base = np.full((16, 24, 3), 180, np.uint8)
        hdr = np.full((16, 24, 4), 2., np.float16)
        hdr[..., 3] = 1.
        profile = replace(
            resolve_delivery_profile("share", quality=95, chroma="444", container="heic"),
            heif_encoder="x265",
        )
        template = directory / "template.heic"
        template.write_text(json.dumps({"aux": 100}))
        image = SimpleNamespace()
        image.imageBySettingContentHeadroom_ = lambda headroom: image
        # An existing template exercises the real copy/replace/validation route.
        # Accidentally rebuilding it is a failure rather than an invisible stub.
        context = SimpleNamespace(
            writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_=Mock(
                side_effect=AssertionError("template unexpectedly rebuilt")),
        )
        quartz = SimpleNamespace(
            CGColorSpaceCreateWithName=lambda name: name,
            CIContext=SimpleNamespace(contextWithOptions_=lambda options: context),
            kCGColorSpaceDisplayP3="p3", kCGColorSpaceExtendedLinearDisplayP3="linear-p3",
            kCIFormatRGBA8="rgba8", kCIFormatRGBAh="rgbah", kCIContextCacheIntermediates="cache",
            kCGImageDestinationEncodeBaseIsSDR="base-sdr",
            kCGImageDestinationEncodeGainMapSubsampleFactor="gainmap-subsample",
            kCGImageDestinationEncodeBasePixelFormatRequest="base-format",
            kCGImageDestinationLossyCompressionQuality="quality",
            kCIImageRepresentationHDRImage="hdr-image",
            kCIImageRepresentationHDRGainMapAsRGB="rgb-gainmap",
            kCGImageDestinationEncodeRequest="request",
            kCGImageDestinationEncodeToISOGainmap="iso-gainmap",
            kCGImageDestinationEncodeRequestOptions="request-options",
        )
        foundation = SimpleNamespace(
            NSNumber=SimpleNamespace(numberWithInt_=int, numberWithUnsignedInt_=int),
            NSURL=SimpleNamespace(fileURLWithPath_=str),
        )
        inspection_overrides = {}

        def inspect(path):
            payload = json.loads(path.read_text())
            self.assertIn("primary", payload)
            return {
                "width": 24, "height": 16, "bit_depth": profile.heif_bit_depth,
                "has_iso_gainmap": True, "profile": "Display P3",
                "chroma_subsampling": "4:4:4", "gainmap_pixel_format": "444f",
                "headroom": 2., **inspection_overrides,
            }

        def encode(source, donor, quality, chroma, **options):
            donor.write_text(json.dumps({"primary": f"q{quality}-{chroma}"}))
            return {"encoder": "x265"}

        def replace_primary(target, donor):
            payload = json.loads(target.read_text())
            payload.update(json.loads(donor.read_text()))
            target.write_text(json.dumps(payload))

        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"Quartz": quartz, "Foundation": foundation}))
            stack.enter_context(patch.object(gainmap, "_apple_gainmap_api_status", return_value=(True, "fixture")))
            stack.enter_context(patch.object(gainmap, "_ciimage_from_rgba", return_value=(image, b"owner")))
            stack.enter_context(patch.object(gainmap, "_nsnumber_bool", side_effect=bool))
            stack.enter_context(patch("dngscan.heif_encoder.encode", side_effect=encode))
            stack.enter_context(patch("dngscan.heif_gainmap.replace_primary", side_effect=replace_primary))
            inspector = stack.enter_context(patch.object(gainmap, "inspect_gainmap_file", side_effect=inspect))
            read = stack.enter_context(patch.object(gainmap, "read_primary_rgb_u8",
                                                    return_value=np.full_like(base, decoded_value)))
            hdr_read = stack.enter_context(patch.object(gainmap, "_read_expanded_hdr_rgba_half", return_value=hdr))
            absolute_gate = stack.enter_context(patch.object(
                gainmap, "_base_roundtrip_is_acceptable", wraps=gainmap._base_roundtrip_is_acceptable))
            yield SimpleNamespace(
                base=base, hdr=hdr, profile=profile, template=template, inspector=inspector,
                read=read, hdr_read=hdr_read, overrides=inspection_overrides,
                absolute_gate=absolute_gate,
            )

    def write(self, fixture, out, *, session=None, precheck=None):
        return gainmap.write_apple_gainmap_file(
            fixture.base, fixture.hdr, out, 1., delivery=fixture.profile,
            _verify_roundtrip_capability=False, _template_path=fixture.template,
            _gainmap_quality=100, _primary_session=session,
            _collect_coding_metrics=True, _sdr_precheck=precheck,
        )

    def test_precheck_failure_never_reads_hdr_or_publishes_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "existing.heic"
            out.write_bytes(b"previous good delivery")
            with self.writer_fixture(directory) as fixture:
                def reject(metrics, *, encoded_bytes):
                    self.assertIn("base_mean_code_error", metrics)
                    self.assertIn("coding_local_luma_p99", metrics)
                    self.assertNotIn("chroma_error", metrics)
                    self.assertGreater(encoded_bytes, 0)
                    self.assertEqual(out.read_bytes(), b"previous good delivery")
                    raise auto_encode.EncodingStageRejected(
                        "coding budget exceeded", metrics=metrics,
                        rejected_at="sdr_additional", file_size_bytes=encoded_bytes,
                    )

                with self.assertRaisesRegex(auto_encode.EncodingStageRejected, "coding budget"):
                    self.write(fixture, out, precheck=reject)
                fixture.inspector.assert_called_once()
                fixture.read.assert_called_once()
                fixture.absolute_gate.assert_called_once()
                fixture.hdr_read.assert_not_called()
            self.assertEqual(out.read_bytes(), b"previous good delivery")
            self.assertEqual(set(directory.iterdir()), {out, directory / "template.heic"})

    def test_session_hit_reuses_sdr_metrics_but_revalidates_container_and_hdr(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            session = _SessionSeam(directory)
            with self.writer_fixture(directory) as fixture:
                precheck = Mock()
                first = self.write(fixture, directory / "first.heic", session=session, precheck=precheck)
                fixture.template.write_text(json.dumps({"aux": 90}))
                second = self.write(fixture, directory / "second.heic", session=session, precheck=precheck)
                self.assertEqual(fixture.inspector.call_count, 2)
                self.assertEqual(fixture.absolute_gate.call_count, 2)
                self.assertEqual(fixture.hdr_read.call_count, 2)
                fixture.read.assert_called_once()
                self.assertEqual(precheck.call_count, 2)
                self.assertEqual(session.remembered, ["q95-444"])
                self.assertEqual(session.lookups, ["q95-444", "q95-444"])
                for key in ("base_mean_code_error", "coding_luma_rmse", "coding_chroma_rmse"):
                    self.assertEqual(first[key], second[key])

    def test_cached_primary_cannot_bypass_any_container_gate(self):
        invalid = (
            {"width": 25}, {"bit_depth": 8}, {"has_iso_gainmap": False},
            {"profile": "sRGB"}, {"chroma_subsampling": "4:2:0"},
            {"gainmap_pixel_format": "L008"}, {"headroom": 1.}, {"headroom": 4.},
        )
        for override in invalid:
            with self.subTest(override=override), tempfile.TemporaryDirectory() as td:
                directory = Path(td)
                out = directory / "existing.heic"
                out.write_bytes(b"previous good delivery")
                session = _SessionSeam(directory)
                with self.writer_fixture(directory) as fixture:
                    self.write(fixture, directory / "first.heic", session=session)
                    fixture.overrides.update(override)
                    with self.assertRaises(RuntimeError):
                        self.write(fixture, out, session=session)
                    self.assertEqual(fixture.inspector.call_count, 2)
                    fixture.read.assert_called_once()
                    fixture.hdr_read.assert_called_once()
                    fixture.absolute_gate.assert_called_once()
                    self.assertEqual(session.lookups, ["q95-444"])
                self.assertEqual(out.read_bytes(), b"previous good delivery")
                self.assertFalse(any(".tmp" in path.name for path in directory.iterdir()))

    def test_absolute_sdr_rejection_reports_measurements_before_precheck_or_hdr(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td)
            out = directory / "existing.heic"
            out.write_bytes(b"previous good delivery")
            session = _SessionSeam(directory)
            with self.writer_fixture(directory, decoded_value=220) as fixture:
                precheck = Mock()
                with self.assertRaises(auto_encode.EncodingStageRejected) as caught:
                    self.write(fixture, out, session=session, precheck=precheck)
                self.assertEqual(caught.exception.rejected_at, "sdr_base")
                self.assertIn("base_mean_code_error", caught.exception.metrics)
                self.assertIn("coding_luma_rmse", caught.exception.metrics)
                self.assertNotIn("chroma_error", caught.exception.metrics)
                self.assertGreater(caught.exception.file_size_bytes, 0)
                precheck.assert_not_called()
                fixture.hdr_read.assert_not_called()
                fixture.read.assert_called_once()
            self.assertEqual(out.read_bytes(), b"previous good delivery")
            self.assertFalse(any(".tmp" in path.name for path in directory.iterdir()))


if __name__ == "__main__":
    unittest.main()
