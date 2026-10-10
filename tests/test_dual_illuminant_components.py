# SPDX-License-Identifier: GPL-3.0-or-later
"""Independent DNG calibration interpolation reaches actual LibRaw/hot-WB pixels."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from dngscan import metadata, raw_io, wb
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_review_r6 import _write_tiff, _SRATIONAL, _RATIONAL, _SHORT


CM1 = np.array([[.72, -.16, -.06], [-.31, 1.18, .18], [-.04, .12, .66]])
CM2 = np.array([[.66, -.12, -.04], [-.25, 1.10, .12], [-.02, .08, .72]])
CC1 = np.array([[1.05, .02, 0.], [.01, 1., 0.], [0., .01, .95]])
CC2 = np.array([[.95, -.01, .02], [0., 1.03, .01], [.01, 0., 1.05]])
AB = np.array([1.1, 1., .9])


def _replace_tags(path, additions):
    """Append a new real TIFF directory, leaving the original image strips intact."""
    data = bytearray(path.read_bytes())
    ifd, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, ifd)
    entries = {}
    for index in range(count):
        entry = bytes(data[ifd + 2 + 12 * index:ifd + 14 + 12 * index])
        tag, = struct.unpack_from("<H", entry)
        entries[tag] = entry
    directory = len(data)
    payload_offset = directory + 2 + 12 * len(set(entries) | set(additions)) + 4
    payload = bytearray()
    for tag, (kind, values) in additions.items():
        if kind in (5, 10):
            integer = "l" if kind == 10 else "L"
            value = b"".join(struct.pack("<" + 2 * integer, round(float(v) * 1000000),
                                         1000000) for v in np.asarray(values).ravel())
            size = np.asarray(values).size
        elif kind == 2:
            value = values.encode("ascii") + b"\0"
            size = len(value)
        elif kind == 3:
            value = struct.pack("<" + "H" * len(values), *values)
            size = len(values)
        else:
            raise AssertionError(kind)
        field = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack(
            "<L", payload_offset + len(payload))
        entries[tag] = struct.pack("<HHL", tag, kind, size) + field
        if len(value) > 4:
            payload.extend(value)
    struct.pack_into("<L", data, 4, directory)
    data.extend(struct.pack("<H", len(entries)))
    data.extend(b"".join(entries[tag] for tag in sorted(entries)))
    data.extend(bytes(4))
    data.extend(payload)
    path.write_bytes(data)


def _write_calibrated_dng(path, *, cc2=CC2, signatures=("body", "body"),
                          illuminants=(17, 21), third=False, cc1=CC1):
    y, x = np.indices((128, 128))
    write_sensor_dng(path, signal=(700 + x + y).astype(np.uint16), neutral=(.5, 1., .75))
    additions = {
        50721: (10, CM1), 50722: (10, CM2),
        50778: (3, [illuminants[0]]), 50779: (3, [illuminants[1]]),
        50727: (5, AB), 50931: (2, signatures[0]), 50932: (2, signatures[1]),
    }
    if cc1 is not None:
        additions[50723] = (10, cc1)
    if cc2 is not None:
        additions[50724] = (10, cc2)
    if third:
        additions.update({52529: (3, [22]), 52530: (10, CC1), 52531: (10, CM1)})
    _replace_tags(path, additions)


def _expected(cct, *, cc1=CC1, cc2=CC2, signatures_match=True,
              temperatures=(2856., 6504.)):
    # Independent reciprocal-temperature reference. Keep AB outside both
    # interpolations; matrix multiplication of the endpoints is not equivalent.
    t1, t2 = temperatures
    weight = np.clip((1. / cct - 1. / t2) / (1. / t1 - 1. / t2), 0., 1.)
    cm = weight * CM1 + (1. - weight) * CM2
    a = np.eye(3) if cc1 is None or not signatures_match else cc1
    b = np.eye(3) if cc2 is None or not signatures_match else cc2
    cc = weight * a + (1. - weight) * b
    return AB[:, None] * (cc @ cm)


class DualIlluminantComponentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "calibration.dng"

    def test_parser_preserves_components_and_signature_validity(self):
        for signatures, applicable in ((('body', 'body'), True),
                                       (('body', 'profile'), False), (('', ''), True)):
            with self.subTest(signatures=signatures):
                _write_calibrated_dng(self.path, signatures=signatures)
                calibration = metadata.read_dng_color_calibration(self.path)
                self.assertEqual(calibration.calibration_signatures_match, applicable)
                np.testing.assert_array_equal(calibration.color_matrix1, CM1)
                np.testing.assert_array_equal(calibration.color_matrix2, CM2)
                np.testing.assert_array_equal(calibration.camera_calibration1, CC1)
                np.testing.assert_array_equal(calibration.camera_calibration2, CC2)
                np.testing.assert_array_equal(calibration.analog_balance, AB)
                # Existing readers can still inspect composed endpoint matrices.
                np.testing.assert_allclose(calibration.matrix1,
                                           _expected(2856., signatures_match=applicable),
                                           rtol=0., atol=2e-16)

    def test_varying_cc_and_cm_are_independently_interpolated(self):
        _write_calibrated_dng(self.path)
        calibration = metadata.read_dng_color_calibration(self.path)
        for cct in (3400., 5500.):
            with self.subTest(cct=cct):
                actual = wb.interpolated_color_matrix(calibration, cct)
                np.testing.assert_allclose(actual, _expected(cct), rtol=2e-15, atol=1e-16)
                old_calibration = SimpleNamespace(
                    matrix1=calibration.matrix1, cct1=calibration.cct1,
                    matrix2=calibration.matrix2, cct2=calibration.cct2)
                old = wb.interpolated_color_matrix(old_calibration, cct)
                self.assertGreater(float(np.max(np.abs(actual - old))), .0001)

    def test_controls_keep_single_cc_signature_and_endpoint_behavior(self):
        cases = [
            (dict(cc2=CC1), dict(cc2=CC1)),
            (dict(signatures=("body", "profile")), dict(signatures_match=False)),
            (dict(cc1=None, cc2=None), dict(cc1=None, cc2=None)),
            (dict(cc2=None), dict(cc2=None)),
        ]
        for settings, reference in cases:
            with self.subTest(settings=settings):
                _write_calibrated_dng(self.path, **settings)
                calibration = metadata.read_dng_color_calibration(self.path)
                for cct in (2000., 3400., 5500., 9300.):
                    np.testing.assert_allclose(wb.interpolated_color_matrix(calibration, cct),
                                               _expected(cct, **reference),
                                               rtol=2e-15, atol=2e-16)

    def test_third_illuminant_uses_components_from_the_active_bracket(self):
        _write_calibrated_dng(self.path, third=True)
        calibration = metadata.read_dng_color_calibration(self.path)
        np.testing.assert_array_equal(calibration.color_matrix3, CM1)
        np.testing.assert_array_equal(calibration.camera_calibration3, CC1)
        cct = 7000.
        weight = (1. / cct - 1. / 7504.) / (1. / 6504. - 1. / 7504.)
        expected = AB[:, None] * ((weight * CC2 + (1. - weight) * CC1)
                                  @ (weight * CM2 + (1. - weight) * CM1))
        np.testing.assert_allclose(wb.interpolated_color_matrix(calibration, cct),
                                   expected, rtol=2e-15, atol=2e-16)

    def test_color_matrix2_only_keeps_its_matching_calibration_components(self):
        _write_tiff(self.path, [
            (50722, _SRATIONAL, CM2.ravel()), (50779, _SHORT, [21]),
            (50724, _SRATIONAL, CC2.ravel()), (50727, _RATIONAL, AB),
        ])
        calibration = metadata.read_dng_color_calibration(self.path)
        self.assertEqual(calibration.cct1, 6504.)
        self.assertIsNone(calibration.matrix2)
        np.testing.assert_array_equal(calibration.color_matrix1, CM2)
        np.testing.assert_array_equal(calibration.camera_calibration1, CC2)
        for cct in (3400., 5500., 9300.):
            np.testing.assert_allclose(wb.interpolated_color_matrix(calibration, cct),
                                       AB[:, None] * (CC2 @ CM2), rtol=2e-15)

    def test_real_dng_kelvin_load_uses_correct_multipliers_and_scene_transform(self):
        for settings, reference, mode in (
            ({}, {}, "3400k"), ({}, {}, "5500k"),
            (dict(cc2=CC1), dict(cc2=CC1), "5500k"),
            (dict(signatures=("body", "profile")), dict(signatures_match=False), "5500k"),
            ({}, {}, "9300k"),
            (dict(illuminants=(23, 21)), dict(temperatures=(5003., 6504.)), "3200k"),
        ):
            with self.subTest(settings=settings, mode=mode):
                _write_calibrated_dng(self.path, **settings)
                camera = raw_io.load_raw(self.path, wb_mode="camera")
                actual = raw_io.load_raw(self.path, wb_mode=mode)
                cct = wb.kelvin_mode_cct(mode)
                expected_matrix = _expected(cct, **reference)
                expected_wb = wb.kelvin_camera_multipliers(cct, expected_matrix)
                self.assertEqual(actual.wb_mode, mode)
                self.assertIsNone(actual.wb_degradation)
                np.testing.assert_allclose(actual.applied_wb, expected_wb, rtol=2e-15)
                decode, _, source = raw_io.resolve_hot_wb_c0(camera, cct)
                target = raw_io.d65_row_normalize(expected_matrix) if source == "evidence+cct" else decode
                transform = raw_io.hot_wb_matrix_rec2020(decode, camera.decode_wb,
                                                         expected_wb, target)
                expected_scene = raw_io.apply_hot_wb_rec2020(camera.scene_rec2020_render, transform)
                np.testing.assert_array_equal(actual.scene_rec2020_render, expected_scene)


if __name__ == "__main__":
    unittest.main()
