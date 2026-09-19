# SPDX-License-Identifier: GPL-3.0-or-later
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from dngscan.delivery_integrity import encoded_content_signature
from dngscan.export import carry_capture_metadata, carry_capture_metadata_hdr
from tests.test_codec_repack import heif_fixture, jpeg_template


class DeliveryIntegrityTests(unittest.TestCase):
    def test_jpeg_metadata_may_move_but_coding_and_gainmap_may_not_change(self):
        rgb = np.random.default_rng(21).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        original, _ = jpeg_template(rgb)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.jpg"
            path.write_bytes(original)
            expected = encoded_content_signature(path, "jpeg")
            exif = b"Exif\0\0test-metadata"
            path.write_bytes(original[:2]+b"\xff\xe1"+struct.pack(">H", len(exif)+2)+exif+original[2:])
            self.assertEqual(encoded_content_signature(path, "jpeg"), expected)
            path.write_bytes(original.replace(b"opaque-gainmap", b"altered-gainmap"))
            self.assertNotEqual(encoded_content_signature(path, "jpeg"), expected)
            path.write_bytes(original)
            from dngscan.jpeg_gainmap import replace_primary
            replace_primary(path, rgb, 80, "420")
            self.assertNotEqual(encoded_content_signature(path, "jpeg"), expected)

    def test_heif_signature_detects_auxiliary_and_colour_property_changes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "image.heic"
            original = heif_fixture()
            path.write_bytes(original)
            expected = encoded_content_signature(path, "heic")
            for old, new in ((b"gainmap-payload", b"damaged-payload"),
                             (b"codec-config", b"other-config")):
                path.write_bytes(original.replace(old, new))
                self.assertNotEqual(encoded_content_signature(path, "heic"), expected)

    def test_metadata_reencode_leaves_verified_jpeg_untouched(self):
        rgb = np.random.default_rng(82).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "photo.jpg"
            Image.fromarray(rgb).save(path, quality=98)
            original = path.read_bytes()
            def rewrite(src, dst, *args, **kwargs):
                Image.fromarray(rgb).save(dst, format="JPEG", quality=40)
                return True
            with patch("dngscan.export._scrubbed_capture_metadata", return_value=object()), \
                 patch("dngscan.export._rewrite_with_metadata", side_effect=rewrite):
                self.assertFalse(carry_capture_metadata(Path("capture.dng"), path))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(td).iterdir()), [path])

    def test_hdr_metadata_carry_rejects_changed_auxiliary_despite_same_descriptors(self):
        info = dict(has_iso_gainmap=True, headroom=4., profile="Display P3",
                    chroma_subsampling="4:4:4", gainmap_pixel_format="420f",
                    width=16, height=12, bit_depth=10, gainmap_width=16, gainmap_height=12)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "photo.heic"
            original = heif_fixture()
            path.write_bytes(original)
            def rewrite(src, dst, *args, **kwargs):
                dst.write_bytes(original.replace(b"gainmap-payload", b"damaged-payload"))
                return True
            with patch("dngscan.export._scrubbed_capture_metadata", return_value=object()), \
                 patch("dngscan.gainmap.inspect_gainmap_file", return_value=info), \
                 patch("dngscan.export._rewrite_with_metadata", side_effect=rewrite):
                self.assertFalse(carry_capture_metadata_hdr(Path("capture.dng"), path, "heic"))
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
