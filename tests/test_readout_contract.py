# SPDX-License-Identifier: GPL-3.0-or-later
"""Per-file capture constraints, with real DNG acquisition and analysis."""
from __future__ import annotations

from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan import calibration, priors, readout
from dngscan.analysis import analyze, sensor_prior_evidence
from dngscan.raw_io import load_raw
from tests.test_spectral_fallback_pipeline import write_noise_dng
from tests.test_user_calibration import collect_profile


def edit_ifd(path, replacements):
    data = bytearray(path.read_bytes())
    start, = struct.unpack_from("<L", data, 4)
    count, = struct.unpack_from("<H", data, start)
    entries = {}
    for index in range(count):
        entry = bytes(data[start + 2 + 12*index:start + 14 + 12*index])
        tag, = struct.unpack_from("<H", entry)
        entries[tag] = entry
    for tag, replacement in replacements.items():
        if replacement is None:
            entries.pop(tag, None)
    offset = len(data)
    payload_offset = offset + 2 + 12*len(set(entries) | {tag for tag, value in replacements.items() if value is not None}) + 4
    payload = bytearray()
    for tag, replacement in replacements.items():
        if replacement is None:
            continue
        kind, size, value = replacement
        field = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack("<L", payload_offset + len(payload))
        entries[tag] = struct.pack("<HHL", tag, kind, size) + field
        if len(value) > 4:
            payload.extend(value)
    data[4:8] = struct.pack("<L", offset)
    data.extend(struct.pack("<H", len(entries)))
    data.extend(b"".join(entries[tag] for tag in sorted(entries)))
    data.extend(struct.pack("<L", 0))
    data.extend(payload)
    path.write_bytes(data)


def jpeg_header(*, point_transform=0, sof=0xc3):
    # One component, 14-bit, 128x128; enough header for process inspection.
    frame = struct.pack(">BHHB", 14, 128, 128, 1) + b"\x01\x11\x00"
    scan = b"\x01\x01\x00\x01\x00" + bytes([point_transform])
    return (b"\xff\xd8\xff" + bytes([sof]) + struct.pack(">H", len(frame)+2) + frame +
            b"\xff\xda" + struct.pack(">H", len(scan)+2) + scan)


def append_linear_ifd(path, *, kind=16, baseline=1., extra=None):
    """A real second LinearRAW frame with deliberately conflicting metadata."""
    data = bytearray(path.read_bytes())
    offset = len(data)
    fields = {254: (4, 1, struct.pack("<L", kind)), 256: (4, 1, struct.pack("<L", 256)),
              257: (4, 1, struct.pack("<L", 256)), 258: (3, 1, struct.pack("<H", 12)),
              259: (3, 1, struct.pack("<H", 1)), 262: (3, 1, struct.pack("<H", 34892)),
              277: (3, 1, struct.pack("<H", 3)), 278: (4, 1, struct.pack("<L", 256)),
              279: (4, 1, struct.pack("<L", 256*256*3*2)),
              273: (4, 1, bytes(4)), 50717: (4, 1, struct.pack("<L", 4095)),
              50720: (4, 2, struct.pack("<LL", 240, 240)),
              51041: (12, 2, struct.pack("<dd", .01, .001))}
    if baseline is not None:
        fields[50730] = (10, 1, struct.pack("<ll", int(baseline*1000), 1000))
    fields.update(extra or {})
    payload_offset = offset + 2 + 12*len(fields) + 4
    entries, payload = [], bytearray()
    for tag, (typ, count, value) in sorted(fields.items()):
        field = value.ljust(4, b"\0") if len(value) <= 4 else struct.pack("<L", payload_offset+len(payload))
        entries.append((tag, struct.pack("<HHL", tag, typ, count)+field))
        if len(value) > 4:
            payload.extend(value)
    pixels = struct.pack("<L", payload_offset+len(payload))
    entries = [(tag, entry[:8]+pixels if tag == 273 else entry) for tag, entry in entries]
    data.extend(struct.pack("<H", len(fields)))
    data.extend(b"".join(entry for _, entry in entries))
    data.extend(bytes(4))
    data.extend(payload)
    data.extend(bytes(256*256*3*2))
    path.write_bytes(data)
    return offset


def wrap_frames(path, frames, *, baseline=-1.):
    """Put global DNG metadata in IFD0, with the RAW frames in SubIFDs."""
    raw = bytearray(path.read_bytes())
    wrapper = len(raw)
    payload = wrapper + 2 + 12*3 + 4
    raw[4:8] = struct.pack("<L", wrapper)
    raw.extend(struct.pack("<H", 3))
    raw.extend(struct.pack("<HHL L", 330, 4, len(frames), payload))
    raw.extend(struct.pack("<HHL4B", 50706, 1, 4, 1, 4, 0, 0))
    raw.extend(struct.pack("<HHL L", 50730, 10, 1, payload+4*len(frames)))
    raw.extend(bytes(4))
    raw.extend(struct.pack("<"+"L"*len(frames), *frames))
    raw.extend(struct.pack("<ll", int(baseline*1000), 1000))
    path.write_bytes(raw)


class CaptureReadoutTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        environment = patch.dict(os.environ, DNGSCAN_CALIBRATION_DIR=str(self.root / "store"))
        environment.start()
        self.addCleanup(environment.stop)

    def dng(self, *, file_profile=True, replacements=None):
        path = self.root / "capture.dng"
        write_noise_dng(path, file_profile=file_profile)
        if replacements:
            edit_ifd(path, replacements)
        return path

    def install(self, profile):
        path = self.root / "profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return calibration.import_calibration(path)

    def profile(self, **constraints):
        item = collect_profile()
        item["readout_contract"] = {"version": 1, **constraints}
        return item

    def test_fp_shutter_acquired_without_lens_and_adc_bits_remain_unknown(self):
        path = self.dng()
        with patch("dngscan.embedded_lens.read", side_effect=AssertionError("lens must not supply capture metadata")):
            from dngscan.evidence import acquire_raw_evidence
            evidence = acquire_raw_evidence(path)
        self.assertEqual(evidence.shot_shutter, "electronic")
        capture = evidence.capture_readout
        self.assertEqual(capture["shutter_source"], "manufacturer-capability:SIGMA-fp")
        self.assertIn("sigma-global.com", capture["shutter_source_url"])
        self.assertEqual(capture["sample_bits"], 16)
        self.assertIsNone(capture["sensor_bits"])
        self.assertIsNone(capture["sensor_binning"])
        self.assertIsNone(capture["capture_kind"])
        self.assertEqual(capture["libraw_raw_geometry"], [128, 128])

    def test_fp_l_and_unknown_models_do_not_borrow_fp_capability(self):
        for name in (b"fp L\0", b"unknown\0"):
            with self.subTest(model=name):
                path = self.dng(replacements={272: (2, len(name), name)})
                self.assertIsNone(readout.read(path)["shutter"])

    def test_primary_thumbnail_fields_never_replace_main_raw_ifd(self):
        path = self.dng()
        data = bytearray(path.read_bytes())
        raw_ifd, = struct.unpack_from("<L", data, 4)
        root = len(data)
        fields = {254: (4, 1, struct.pack("<L", 1)), 256: (4, 1, struct.pack("<L", 160)),
                  257: (4, 1, struct.pack("<L", 120)), 258: (3, 1, struct.pack("<H", 8)),
                  259: (3, 1, struct.pack("<H", 7)), 262: (3, 1, struct.pack("<H", 2)),
                  330: (4, 1, struct.pack("<L", raw_ifd)), 50706: (1, 4, bytes([1, 4, 0, 0]))}
        data[4:8] = struct.pack("<L", root)
        data.extend(struct.pack("<H", len(fields)))
        data.extend(b"".join(struct.pack("<HHL", tag, kind, count) + value.ljust(4, b"\0")
                             for tag, (kind, count, value) in sorted(fields.items())))
        data.extend(struct.pack("<L", 0))
        path.write_bytes(data)
        capture = readout.read(path)
        self.assertEqual(capture["raw_geometry"], [128, 128])
        self.assertEqual(capture["sample_bits"], 16)
        self.assertTrue(capture["storage_lossless"])
        self.assertEqual(capture["compression_code"], 1)

    def test_fractional_or_missing_default_crop_does_not_suppress_other_fields(self):
        for crop in (None, (5, 2, struct.pack("<LLLL", 255, 2, 253, 2))):
            with self.subTest(crop=crop):
                path = self.dng(replacements={50720: crop, 50829: (4, 4, struct.pack("<LLLL", 0, 0, 128, 128))})
                capture = readout.read(path)
                self.assertEqual(capture["default_crop"], None if crop is None else [127.5, 126.5])
                self.assertEqual(capture["active_geometry"], [128, 128])
                self.assertTrue(capture["storage_lossless"])
                self.assertEqual(capture["shutter"], "electronic")

    def test_enhanced_ifd_cannot_change_primary_recipe_noise_or_exposure(self):
        from dngscan.dng_opcodes import read_plan
        from dngscan.noise_model import _file_model
        path = self.dng(replacements={50730: None})
        before = load_raw(path, scene_half_size=True)
        optional = struct.pack(">L4L", 1, 999, 0x01060000, 1, 0)
        enhanced = append_linear_ifd(path, extra={51022: (7, len(optional), optional)})
        edit_ifd(path, {330: (4, 1, struct.pack("<L", enhanced))})
        after = load_raw(path, scene_half_size=True)
        self.assertEqual(after.raw_image.shape, (128, 128))
        self.assertEqual(after.capture_readout["raw_geometry"], [128, 128])
        self.assertEqual(after.capture_readout["sample_bits"], 16)
        plan = read_plan(path)
        self.assertEqual(plan.white_levels, (16383.,))
        self.assertEqual(plan.crop, (0., 0., 128., 128.))
        self.assertEqual((plan.names, plan.skipped), ([], []))
        self.assertEqual(_file_model(after).coefficients("G1"), (1e-4, 1e-8))
        self.assertIsNone(after.baseline_exposure)
        self.assertEqual(after.scene_scale, before.scene_scale)
        np.testing.assert_array_equal(after.scene_rec2020_render, before.scene_rec2020_render)

    def test_first_main_frame_and_raw_baseline_override_match_libraw(self):
        from dngscan import metadata as md
        from dngscan.dng_opcodes import read_plan
        from dngscan.spatial_black import sensor_tags
        for frame_baseline, expected in ((1., 1.), (None, -1.)):
            with self.subTest(frame_baseline=frame_baseline):
                path = self.dng(replacements={50730: None} if frame_baseline is None else None)
                main, = struct.unpack_from("<L", path.read_bytes(), 4)
                # A second main frame is larger, but default LibRaw decodes
                # the first main frame; its explicit BE wins over IFD0.
                later = append_linear_ifd(path, kind=0, baseline=2.)
                wrap_frames(path, [main, later])
                bundle = load_raw(path, scene_half_size=True)
                self.assertEqual(bundle.raw_image.shape, (128, 128))
                self.assertEqual(sensor_tags(path, {258})[258], [16])
                self.assertEqual(read_plan(path).white_levels, (16383.,))
                self.assertEqual(md.read_dng_shot_info(path).baseline_exposure, expected)
                self.assertEqual(bundle.baseline_exposure, expected)
                # Missing RAW BE falls back to global IFD0 -1 EV, never to
                # the later RAW frame's +2 EV. Pixels and scale use one frame.
                self.assertEqual(bundle.scene_scale, 65535./(2.**expected))

    def test_jpeg7_requires_all_headers_sof3_and_zero_point_transform(self):
        path = self.root / "tiles.bin"
        first = jpeg_header()
        for second, expected in ((jpeg_header(), True), (jpeg_header(point_transform=1), False),
                                 (jpeg_header(sof=0xc0), False), (b"invalid", None)):
            with self.subTest(expected=expected, second=second[:4]):
                path.write_bytes(first + second)
                result = readout._compression(path, {259: [7], 324: [0, len(first)],
                                                     325: [len(first), len(second)]})
                self.assertEqual(result[1], expected)
        self.assertIsNone(readout._jpeg_process(io.BytesIO(b"\xff\xd8\xff\xc3"), 0, 4))
        self.assertIsNone(readout._compression(path, {259: [7], 324: [0]*4097})[1])

    def test_jpeg7_header_cannot_escape_declared_blocks(self):
        path = self.root / "bounded.bin"
        header = jpeg_header()
        path.write_bytes(header)
        for tags in ({324: [0], 325: [2]}, {324: [0]}, {324: [0], 325: [len(header)+1]},
                     {324: [0, 2], 325: [len(header)]}, {324: [0], 279: [len(header)]},
                     {324: [True], 325: [len(header)]}, {324: [0.], 325: [len(header)]},
                     {324: [0], 325: [-1]}, {324: [0], 325: [len(header)], 273: [0], 279: [len(header)]}):
            with self.subTest(tags=tags):
                self.assertIsNone(readout._compression(path, {259: [7], **tags})[1])
        self.assertTrue(readout._compression(path, {259: [7], 273: [0], 279: [len(header)]})[1])

    def test_real_dng_matched_user_calibration_reaches_model_and_reports(self):
        item = self.profile(sample_bits=16, raw_geometry=[128, 128], libraw_raw_geometry=[128, 128],
                            storage_lossless=True)
        self.install(item)
        bundle = load_raw(self.dng(file_profile=False), scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.prior_id, item["id"])
        self.assertEqual(result.noise_model.status, "valid")
        self.assertIn("declared-readout-constraints-matched", result.noise_model.approximation)
        self.assertIsNotNone(result.gain_e_per_dn)
        diagnostic = calibration.calibration_diagnostics("SIGMA", "fp", bundle.shot_shutter,
                                                       bundle.shot_iso, bundle.capture_readout)[0]
        self.assertEqual(diagnostic["readout_match_status"], "matched")

    def test_readout_mismatch_does_not_silently_fall_back_to_curated(self):
        item = self.profile(sample_bits=14)
        self.install(item)
        bundle = load_raw(self.dng(file_profile=False), scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.prior_id, item["id"])
        self.assertEqual(result.noise_model.status, "rejected")
        self.assertEqual(result.noise_model.reason, "file-sample-bits-mismatch")
        self.assertEqual(result.prior_quality_status, "file-sample-bits-mismatch")
        self.assertIsNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)
        self.assertIsNone(result.prior_pdr_ev)

    def test_readout_failure_uses_only_independent_dng_alternative(self):
        item = self.profile(sensor_binning=[1, 1])
        self.install(item)
        bundle = load_raw(self.dng(), scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        model = result.noise_model
        self.assertEqual(result.prior_id, item["id"])
        self.assertEqual(model.source, "DNG NoiseProfile")
        self.assertEqual(model.coefficients("G1"), (1e-4, 1e-8))
        self.assertEqual(model.fallback_reason, "file-sensor-binning-unavailable")
        self.assertEqual(result.prior_quality_status, "file-sensor-binning-unavailable")
        self.assertIsNone(result.gain_e_per_dn)
        self.assertIsNone(result.prior_read_noise_e)

    def test_two_same_iso_submodes_select_the_actual_match_before_recency(self):
        first = self.profile(sample_bits=14)
        first["id"] = "fp 14-bit"
        self.install(first)
        second = self.profile(sample_bits=12)
        second["id"] = "fp 12-bit"
        self.install(second)
        for bits, expected in ((14, first["id"]), (12, second["id"]), (14, first["id"])):
            with self.subTest(bits=bits):
                selected = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=200,
                                              readout={"sample_bits": bits})
                self.assertEqual(selected["id"], expected)
                self.assertTrue(priors.prior_usability(selected)[0])
        unknown = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=200, readout={})
        self.assertFalse(priors.prior_usability(unknown)[0])
        self.assertIn(unknown["id"], (first["id"], second["id"]))

    def test_crop_only_change_does_not_change_declared_sensor_raster_match(self):
        self.install(self.profile(raw_geometry=[128, 128], storage_lossless=True))
        outputs = []
        for size in (128, 120):
            path = self.dng(file_profile=False, replacements={50720: (4, 2, struct.pack("<LL", size, size))})
            bundle = load_raw(path, scene_half_size=True)
            result, _, _ = analyze(bundle, 4)
            self.assertEqual(result.noise_model.status, "valid")
            self.assertEqual(bundle.capture_readout["raw_geometry"], [128, 128])
            self.assertEqual(bundle.capture_readout["default_crop"], [float(size)]*2)
            outputs.append(result.noise_model.coefficients("G1"))
        self.assertEqual(outputs[0], outputs[1])

    def test_legacy_geometry_and_overloaded_compression_stay_unverified(self):
        for field in ({"geometry": [6000, 4000]}, {"compression": "14bit"}, {"compression": "RAW HQ"}):
            with self.subTest(field=field):
                prior = calibration._validated_prior({**collect_profile(), **field})
                bound = priors.with_readout(prior, {"sample_bits": 14, "raw_geometry": [6000, 4000]})
                self.assertFalse(priors.prior_usability(bound)[0])
                self.assertEqual(bound["readout_match_status"], "unverified")
        for compression in ("lossless", "无损压缩", "uncompressed", "无压缩"):
            prior = calibration._validated_prior({**collect_profile(), "compression": compression})
            self.assertEqual(priors.with_readout(prior, {"storage_lossless": True})["readout_match_status"], "matched")

    def test_invalid_typed_constraints_are_rejected_at_import(self):
        for constraints in ([], False, 0, {}, {"version": True}, {"version": 1.},
                            {"version": 2}, {"version": 1, "sample_bits": 0},
                            {"version": 1, "sensor_bits": True}, {"version": 1, "sensor_binning": [1, 0]},
                            {"version": 1, "storage_lossless": 1}, {"version": 1, "made_up_mode": "14bit"}):
            with self.subTest(constraints=constraints), self.assertRaises(ValueError):
                calibration._validated_prior({**collect_profile(), "readout_contract": constraints})

    def csv_directory(self, *, dark_size="128x128", ptc_size="128x128"):
        directory = self.root / "collect"
        directory.mkdir(exist_ok=True)
        raw_header = f"#RawSize: {dark_size}\n" if dark_size is not None else ""
        (directory / "dark-scalars.csv").write_text(
            "#Format: JPTC-DARK/1\n#Camera: SIGMA fp\n#ShutterType: 电子快门\n"
            "#Compression: lossless\n#ImageWidth: 64\n#ImageHeight: 64\n" + raw_header +
            "#AdcStep: 1\n#LinearisationCurve: identity\n#ClipVarianceFactor: 1\n"
            "ISO,ColorIndex,BlackA,BlackB,StdDiffClipped\n" +
            "".join(f"100,{ch},1024,1024,{math.sqrt(2./3.):.17g}\n" for ch in (1, 3)))
        rows = []
        for signal in np.geomspace(2, 15359*1.15, 80):
            mean = min(1024+signal, 16383)
            std = math.sqrt(signal/4 + 1./3.) if mean < 16383 else 0.
            rows.append(f"{mean},{std}")
        raw_header = f"#RawSize: {ptc_size}\n" if ptc_size is not None else ""
        (directory / "ptc-iso100.csv").write_text(
            "#Format: JPTC/2\n#BlackLevel: 1024,1024,1024,1024\n" + raw_header +
            "G1_Mean,G1_Std\n" + "\n".join(rows))
        return directory

    def test_collect_raw_size_reaches_real_dng_and_jpeg_size_is_informational(self):
        summary = calibration.import_calibration(self.csv_directory())
        path = self.dng(file_profile=False, replacements={34855: (3, 1, struct.pack("<H", 100))})
        bundle = load_raw(path, scene_half_size=True)
        result, _, _ = analyze(bundle, 4)
        self.assertEqual(result.noise_model.status, "valid")
        prior = priors.find_priors("SIGMA", "fp", shutter=bundle.shot_shutter, iso=100,
                                  readout=bundle.capture_readout)
        self.assertEqual(prior["calibration_id"], summary["id"])
        self.assertEqual(prior["readout_contract"]["libraw_raw_geometry"], [128, 128])
        self.assertEqual(prior["readout_informational"]["camera_jpeg_geometry"], [64, 64])
        self.assertEqual(prior["readout_match_status"], "matched")
        self.assertAlmostEqual(result.noise_model.coefficients("G1")[1], (1./3.) / 15359.**2)

    def test_collect_mosaic_conflict_rejected_and_partial_sizes_not_inferred(self):
        with self.assertRaisesRegex(ValueError, "RawSize disagrees"):
            calibration.import_calibration(self.csv_directory(ptc_size="256x128"))
        for missing in ("dark_size", "ptc_size"):
            with self.subTest(missing=missing):
                directory = self.csv_directory(**{missing: None})
                payload = calibration.build(directory, None)
                self.assertFalse(payload["acquisition_contract"]["raw_geometry_complete"])
                self.assertIn(None, payload["acquisition_contract"]["raw_geometry_measurements"].values())
                summary = calibration.import_calibration(directory)
                self.assertIn("sub-readout-mode-not-verified", summary["warnings"])
                prior = priors.find_priors("SIGMA", "fp", shutter="electronic", iso=100,
                                          readout={"libraw_raw_geometry": [128, 128], "storage_lossless": True})
                self.assertEqual(prior["readout_match_status"], "unverified")
                self.assertEqual(priors.prior_usability(prior)[1], "measurement-raw-geometry-incomplete")

    def test_capture_contract_round_trips_and_binds_full_source_cache(self):
        from dngscan.gui import preview_cache as pc
        bundle = load_raw(self.dng(), scene_half_size=True)
        metadata = pc._bundle_metadata(bundle)
        rebuilt = pc._bundle_from_cache(bundle.path, json.loads(json.dumps(metadata)),
                                        bundle.scene_rec2020_render, None, None)
        self.assertEqual(rebuilt.capture_readout, bundle.capture_readout)
        changed = replace(bundle, capture_readout={**bundle.capture_readout, "sample_bits": 12})
        self.assertNotEqual(pc._bundle_metadata(changed), metadata)
        rebuilt.capture_readout["raw_geometry"][0] = 42
        self.assertEqual(bundle.capture_readout["raw_geometry"], [128, 128])
        self.assertEqual(pc.PREVIEW_CACHE_VERSION, 29)


if __name__ == "__main__":
    unittest.main()
