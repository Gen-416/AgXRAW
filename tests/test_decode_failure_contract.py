# SPDX-License-Identifier: GPL-3.0-or-later
"""Support probing and sensor acquisition share truthful failure categories."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import rawpy

from dngscan import decode_support
from dngscan.evidence import EvidenceAcquisitionError, acquire_raw_evidence
from tests.test_pipeline_corrections import write_sensor_dng
from tests import test_sensor_support as support_fixtures


class DecodeFailureContractTests(unittest.TestCase):
    def test_exception_categories_agree_and_nef_suffix_is_not_codec_evidence(self):
        cases = (
            (rawpy.LibRawFileUnsupportedError("unsupported"), "unsupported_format", "unsupported-codec"),
            (rawpy.LibRawDataError("corrupt payload"), "corrupt_data", "corrupt-data"),
            (rawpy.LibRawIOError("LibRaw read failure"), "io_error", "io"),
            (OSError(5, "read failure"), "io_error", "io"),
            (RuntimeError("unsupported file format text is not an exception type"), "runtime_failure", "runtime"),
        )
        for error, status, kind in cases:
            with self.subTest(error=type(error).__name__):
                with patch("rawpy.imread", side_effect=error):
                    probe = decode_support._libraw_tier(Path("sample.nef"), "NIKON", "unknown")
                    with self.assertRaises(EvidenceAcquisitionError) as caught:
                        acquire_raw_evidence(Path("sample.nef"))
                self.assertEqual(probe["status"], status)
                self.assertEqual(probe["failure_kind"], kind)
                self.assertEqual(caught.exception.failure, probe)
                self.assertEqual(caught.exception.unsupported_format, status == "unsupported_format")
                self.assertNotIn("授权", probe["detail"])
                self.assertNotIn("HE", probe["detail"])

    def test_actual_truncated_input_is_reported_as_io_not_format_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.nef"
            write_sensor_dng(path, signal=1000)
            path.write_bytes(path.read_bytes()[:4])
            report = decode_support._libraw_tier(path, "NIKON", "test")
            self.assertEqual(report["status"], "io_error")
            with self.assertRaises(EvidenceAcquisitionError) as caught:
                acquire_raw_evidence(path)
            self.assertEqual(caught.exception.failure_kind, "io")

    def test_all_libraw_failures_keep_apple_scene_preflight_independent(self):
        helper = support_fixtures.DecodeSupportProbeTests()
        for status in ("unsupported_format", "corrupt_data", "io_error", "runtime_failure"):
            with self.subTest(status=status):
                report = helper._probe({"status": status, "detail": "injected"}, {
                    "coreimage_available": True, "raw9_supported": True,
                    "versions_offered": ("9",), "fallback_version": None,
                })
                self.assertEqual(report["coreimage"]["status"], "raw9")
                self.assertFalse(report["coreimage"]["blocked_by_libraw"])
                self.assertFalse(report["coreimage"]["sensor_evidence_preflight"])
                self.assertIn("传感器证据不可用", "\n".join(report["lines"]))


if __name__ == "__main__":
    unittest.main()
