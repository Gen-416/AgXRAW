# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

from dngscan import coreimage_decode, raw_io
from dngscan.evidence import acquire_raw_evidence
from dngscan.models import RawEvidence


class _ContextRaw:
    def __init__(self) -> None:
        self.raw_image_visible = np.arange(16, dtype=np.uint16).reshape(4, 4)
        self.raw_colors_visible = np.tile(
            np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (2, 2)
        )
        self.white_level = 4095
        self.black_level_per_channel = [512, 512, 512, 512]
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.daylight_whitebalance = [2.2, 1.0, 1.4, 1.0]
        self.camera_white_level_per_channel = [4095, 4095, 4095, 4095]
        self.color_desc = b"RGBG\x00"
        self.raw_pattern = np.asarray([[0, 1], [3, 2]], dtype=np.uint8)
        self.rgb_xyz_matrix = np.eye(4, 3, dtype=np.float32)
        self.sizes = SimpleNamespace(flip=5)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class EvidenceContractTests(unittest.TestCase):
    def test_acquisition_copies_complete_libraw_evidence(self) -> None:
        source = _ContextRaw()
        with patch("dngscan.evidence.rawpy.imread", return_value=source):
            evidence = acquire_raw_evidence(Path("camera.arw"))

        self.assertEqual(evidence.provider, "libraw")
        self.assertIn("LibRaw", evidence.provider_version or "")
        self.assertEqual(evidence.white_level, 4095)
        self.assertEqual(evidence.orientation_flip, 5)
        self.assertEqual(evidence.color_desc, "RGBG")
        self.assertEqual(evidence.raw_pattern, [[0, 1], [3, 2]])
        source.raw_image_visible[0, 0] = 999
        self.assertNotEqual(int(evidence.raw_image[0, 0]), 999)

    def test_scene_decoder_switch_reuses_identical_evidence_contract(self) -> None:
        evidence = RawEvidence(
            path=Path("camera.dng"),
            raw_image=np.arange(16, dtype=np.uint16).reshape(4, 4),
            raw_colors=np.tile(
                np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (2, 2)
            ),
            white_level=4095,
            black_levels=[512.0] * 4,
            camera_wb=[2.0, 1.0, 1.5, 1.0],
            daylight_wb=[2.2, 1.0, 1.4, 1.0],
            color_desc="RGBG",
            raw_pattern=[[0, 1], [3, 2]],
            camera_white_levels=[4095.0] * 4,
            orientation_flip=0,
            xyz_to_cam=np.eye(4, 3, dtype=np.float32),
            provider_version="rawpy 0.27.0/LibRaw 0.22.0",
        )
        shot = SimpleNamespace(
            make="SONY", model="ILCE-7M5", iso=100, baseline_exposure=None
        )
        scene_u16 = np.full((2, 2, 3), 1024, dtype=np.uint16)
        scene_float = np.full((2, 2, 3), 0.25, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".dng") as source_file:
            path = Path(source_file.name)
            with (
                patch.object(raw_io, "acquire_raw_evidence", return_value=evidence) as acquire,
                patch.object(raw_io.dng_metadata, "read_dng_shot_info", return_value=shot),
                patch.object(raw_io.dng_metadata, "read_dng_color_calibration", return_value=None),
                patch.object(
                    raw_io.dng_metadata,
                    "read_dng_shading_ops",
                    return_value={"gain_maps": [], "vignette": None},
                ),
                patch("dngscan.camera_matrices.fallback_xyz_to_cam", return_value=None),
                patch("dngscan.priors.find_priors", return_value=None),
                patch.object(raw_io, "solve_wb_for_mode", return_value=(None, None)),
                patch.object(raw_io.rawpy, "imread", return_value=_ContextRaw()),
                patch.object(raw_io, "resolve_demosaic_algorithm", return_value=None),
                patch.object(raw_io, "render_to_scene_rec2020", return_value=scene_u16),
                patch.object(
                    raw_io,
                    "build_clip_masks",
                    return_value=np.zeros((2, 2, 3), dtype=np.float16),
                ),
                patch.object(coreimage_decode, "available", return_value=True),
                patch.object(
                    coreimage_decode,
                    "decode_scene_rec2020",
                    return_value=(
                        scene_float,
                        {
                            "version": "9",
                            "decoder_runtime_id": "test-runtime",
                            "baseline_exposure_cleared": True,
                        },
                    ),
                ),
                patch.object(
                    coreimage_decode,
                    "read_dng_opcodes",
                    return_value={"names": ()},
                ),
            ):
                libraw_bundle = raw_io.load_raw(path, decoder="libraw")
                apple_bundle = raw_io.load_raw(
                    path, decoder="coreimage", coreimage_scale="unity"
                )

        self.assertEqual(acquire.call_args_list, [call(path), call(path)])
        self.assertIs(libraw_bundle.evidence, evidence)
        self.assertIs(apple_bundle.evidence, evidence)
        self.assertEqual(libraw_bundle.evidence_provider, "libraw")
        self.assertEqual(apple_bundle.evidence_provider, "libraw")
        self.assertEqual(
            libraw_bundle.evidence_provider_version,
            apple_bundle.evidence_provider_version,
        )
        np.testing.assert_array_equal(
            libraw_bundle.raw_image, apple_bundle.raw_image
        )
        self.assertEqual(libraw_bundle.scene_decoder, "libraw")
        self.assertEqual(apple_bundle.scene_decoder, "coreimage")


if __name__ == "__main__":
    unittest.main()
