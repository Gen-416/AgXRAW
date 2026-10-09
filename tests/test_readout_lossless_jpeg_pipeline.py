# SPDX-License-Identifier: GPL-3.0-or-later
"""Real lossless-JPEG DNG decoding distinguishes a zero point transform."""
from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from dngscan._deps import np, rawpy
from dngscan.evidence import acquire_raw_evidence
from dngscan import readout
from tests.test_pipeline_corrections import write_sensor_dng
from tests.test_readout_contract import edit_ifd


def _constant_lossless_jpeg(value: int, point_transform: int) -> bytes:
    """Encode a 128x128 one-component SOF3 frame, predictor 1, no mocks.

    Two one-bit Huffman codes encode category zero and the initial predictor
    residual. Subsequent constant samples all have zero residual. The point
    transform is applied before predictive coding, as specified by T.81.
    """
    def segment(marker, payload):
        return b"\xff" + bytes([marker]) + struct.pack(">H", len(payload) + 2) + payload

    precision = 14
    transformed = value >> point_transform
    residual = transformed - (1 << (precision - point_transform - 1))
    category = abs(residual).bit_length()
    amplitude = residual + (1 << category) - 1 if residual < 0 else residual
    bits = "1" + format(amplitude, f"0{category}b") + "0" * (128 * 128 - 1)
    bits += "1" * (-len(bits) % 8)
    entropy = bytes(int(bits[index:index + 8], 2) for index in range(0, len(bits), 8))
    entropy = entropy.replace(b"\xff", b"\xff\x00")
    huffman = b"\x00\x02" + bytes(15) + bytes([0, category])
    frame = struct.pack(">BHHB", precision, 128, 128, 1) + b"\x01\x11\x00"
    scan = b"\x01\x01\x00\x01\x00" + bytes([point_transform])
    return (b"\xff\xd8" + segment(0xc4, huffman) + segment(0xc3, frame)
            + segment(0xda, scan) + entropy + b"\xff\xd9")


def _write_jpeg_dng(path: Path, *, value: int, point_transform: int):
    write_sensor_dng(path, signal=value)
    data = bytearray(path.read_bytes())
    offset = len(data)
    encoded = _constant_lossless_jpeg(value, point_transform)
    data.extend(encoded)
    path.write_bytes(data)
    edit_ifd(path, {
        259: (3, 1, struct.pack("<H", 7)),
        273: (4, 1, struct.pack("<L", offset)),
        279: (4, 1, struct.pack("<L", len(encoded))),
    })


@unittest.skipIf(rawpy is None, "rawpy unavailable")
class LosslessJpegReadoutPipelineTests(unittest.TestCase):
    def check_point_transform(self, point_transform, expected_lossless):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "real-sof3.dng"
            value = 2401
            _write_jpeg_dng(path, value=value, point_transform=point_transform)
            evidence = acquire_raw_evidence(path)
            np.testing.assert_array_equal(
                evidence.raw_image, np.full((128, 128), value >> point_transform, np.uint16))
            capture = readout.read(path)
            self.assertEqual(capture["compression_code"], 7)
            self.assertIs(capture["storage_lossless"], expected_lossless)
            self.assertEqual(capture["compression_process"],
                             "lossless-jpeg-sof3-pt0" if expected_lossless else "lossy-jpeg-process")
            self.assertIs(evidence.capture_readout["storage_lossless"], expected_lossless)
            # SOF precision is not used to guess the sensor ADC precision, or
            # overwrite the TIFF stored-sample declaration.
            self.assertEqual(capture["sample_bits"], 16)
            self.assertIsNone(capture["sensor_bits"])

    def test_zero_point_transform_preserves_actual_libraw_samples(self):
        self.check_point_transform(0, True)

    def test_nonzero_point_transform_changes_samples_and_rejects_lossless_claim(self):
        self.check_point_transform(1, False)


if __name__ == "__main__":
    unittest.main()
