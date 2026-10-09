# SPDX-License-Identifier: GPL-3.0-or-later
"""Unknown rational declarations must survive the actual TIFF reader."""
from __future__ import annotations

import io
import struct
import tempfile
import unittest
from pathlib import Path

from dngscan import metadata
from dngscan.analysis import analyze
from dngscan.hdr_agx_plan import compile_tail_snr_gate
from dngscan.noise_model import resolve_noise_model
from dngscan.spatial_black import sensor_tags
from tests.test_noise_model import frame


def noise_metadata_dng(reduction=(0, 0), *, endian="<", typ=5, profile=True, extra=()):
    """A real TIFF RAW IFD with the optional noise tags, requiring no decoder."""
    pack = lambda fmt, *values: struct.pack(endian + fmt, *values)
    entries = [
        (254, 4, 1, pack("L", 0)),
        (256, 4, 1, pack("L", 512)),
        (257, 4, 1, pack("L", 512)),
        (262, 3, 1, pack("H", 32803)),
        (50706, 1, 4, bytes([1, 4, 0, 0])),
        (50710, 1, 3, bytes([0, 1, 2])),
    ]
    if reduction is not None:
        entries.append((50935, typ, 1, pack("ll" if typ == 10 else "LL", *reduction)))
    if profile:
        entries.append((51041, 12, 2, pack("dd", 1e-4, 1e-8)))
    entries.extend(extra)
    entries.sort(key=lambda e: e[0])
    payload_offset = 8 + 2 + len(entries)*12 + 4
    payload, table = bytearray(), bytearray()
    for tag, entry_type, count, data in entries:
        field = data.ljust(4, b"\0") if len(data) <= 4 else pack("L", payload_offset+len(payload))
        table.extend(pack("HHL", tag, entry_type, count) + field)
        if len(data) > 4:
            payload.extend(data)
    return (b"II" if endian == "<" else b"MM") + pack("HL", 42, 8) + pack("H", len(entries)) + table + pack("L", 0) + payload


class RationalStateTests(unittest.TestCase):
    def test_common_parser_preserves_zero_denominators_and_normal_numeric_values(self):
        for endian in ("<", ">"):
            for typ in (5, 10):
                fmt = "ll" if typ == 10 else "LL"
                for numerator, denominator, state in ((0, 0, "unknown"), (1, 0, "invalid"),
                                                       (0, 1, "finite"), (1, 2, "finite")):
                    with self.subTest(endian=endian, typ=typ, rational=(numerator, denominator)):
                        stream = io.BytesIO(struct.pack(endian+fmt, numerator, denominator))
                        value, = metadata._entry_values(stream, typ, 1, struct.pack(endian+"L", 0), endian)
                        if state == "finite":
                            self.assertIsInstance(value, float)
                            self.assertEqual(value, numerator/denominator)
                        else:
                            self.assertIsInstance(value, metadata.UndefinedRational)
                            self.assertEqual((value.numerator, value.denominator, value.state), (numerator, denominator, state))
                            with self.assertRaises(ValueError):
                                float(value)

    def test_actual_sensor_tags_distinguish_unknown_none_invalid_and_applied(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for endian in ("<", ">"):
                for typ in (5, 10):
                    for rational, declaration, model_status in (
                        ((0, 0), "unknown", "valid"), ((0, 1), "none", "valid"),
                        ((1, 0), "invalid", "rejected"), ((1, 2), "applied", "rejected"),
                        (None, "absent", "valid"),
                    ):
                        with self.subTest(endian=endian, typ=typ, rational=rational):
                            path = root/"noise.dng"
                            path.write_bytes(noise_metadata_dng(rational, endian=endian, typ=typ))
                            tags = sensor_tags(path, {50935, 51041})
                            if rational is not None and rational[1] == 0:
                                self.assertEqual(tags[50935][0].state, declaration)
                            bundle = frame(known=False)
                            bundle.path = path
                            model = resolve_noise_model(bundle, {i: 16383 for i in range(4)})
                            self.assertEqual(model.noise_reduction_status, declaration)
                            self.assertEqual(model.status, model_status)
                            if model_status == "valid":
                                self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))

    def test_unknown_declaration_does_not_reject_a_matched_external_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"noise.dng"
            for rational, expected in (((0, 0), "unknown"), ((0, 1), "none")):
                path.write_bytes(noise_metadata_dng(rational, profile=False))
                bundle = frame()
                bundle.path = path
                model = resolve_noise_model(bundle, {i: 16383 for i in range(4)})
                self.assertEqual(model.status, "valid")
                self.assertEqual(model.reason, "matched-shot-read-model")
                self.assertEqual(model.noise_reduction_status, expected)

    def test_declaration_survives_without_profile_or_matched_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"no-profile.dng"
            for endian in ("<", ">"):
                for rational, declaration in (((0, 0), "unknown"), ((0, 1), "none"), (None, "absent")):
                    with self.subTest(endian=endian, rational=rational):
                        path.write_bytes(noise_metadata_dng(rational, endian=endian, profile=False))
                        bundle = frame(known=False)
                        bundle.path = path
                        result, _, _ = analyze(bundle, 4)
                        self.assertEqual(result.noise_model.status, "unavailable")
                        self.assertEqual(result.noise_model.reason, "no-matched-calibration")
                        self.assertEqual(result.noise_model.noise_reduction_status, declaration)
                        self.assertEqual(result.noise_evidence_status, "unavailable")
                        # Keeping the declaration does not promote unknown
                        # processing into rejected evidence or invent SNR.
                        self.assertEqual(compile_tail_snr_gate(result), 1.)

    def test_declaration_survives_rejected_prior_without_file_profile(self):
        prior = {"source": "rejected calibration", "quality": {"status": "high-residual"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"no-profile.dng"
            for endian in ("<", ">"):
                for rational, declaration in (((0, 0), "unknown"), ((0, 1), "none"), (None, "absent")):
                    with self.subTest(endian=endian, rational=rational):
                        path.write_bytes(noise_metadata_dng(rational, endian=endian, profile=False))
                        bundle = frame(known=False)
                        bundle.path = path
                        model = resolve_noise_model(bundle, {i: 16383 for i in range(4)}, prior)
                        self.assertEqual(model.status, "rejected")
                        self.assertEqual(model.source, "rejected calibration")
                        self.assertEqual(model.reason, "quality-high-residual")
                        self.assertEqual(model.noise_reduction_status, declaration)

    def test_bad_other_numeric_tags_remain_best_effort_metadata(self):
        for endian in ("<", ">"):
            pack = lambda fmt, *values: struct.pack(endian+fmt, *values)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/"malformed.dng"
                path.write_bytes(noise_metadata_dng(endian=endian, extra=[
                    (271, 2, 6, b"SIGMA\0"),
                    (272, 2, 3, b"fp\0"),
                    (50730, 10, 1, pack("ll", 1, 0)),
                ]))
                shot = metadata.read_dng_shot_info(path)
                self.assertEqual(shot.make, "SIGMA")
                self.assertEqual(shot.model, "fp")
                self.assertIsNone(shot.baseline_exposure)
                path.write_bytes(noise_metadata_dng(endian=endian, extra=[
                    (50721, 10, 9, pack("ll", 1, 0) + pack("ll", 0, 1)*8),
                    (50778, 3, 1, pack("H", 21)),
                ]))
                self.assertIsNone(metadata.read_dng_color_calibration(path))


if __name__ == "__main__":
    unittest.main()
