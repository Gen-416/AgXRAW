# SPDX-License-Identifier: GPL-3.0-or-later
"""Primary codestream reuse must never replace package verification on retries."""
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from dngscan import gainmap, heif_encoder, jpeg_gainmap
from dngscan.gainmap_session import PrimarySearchSession
from tests import test_gainmap_staged_delivery as staged


class ManualPrimaryReuseTests(unittest.TestCase):
    @contextmanager
    def fixture(self, directory, container, *, accepted):
        # Stub container I/O, retaining the real session, retry ladder and pixel gates.
        with staged.GainmapStagedWriterTests().writer_fixture(directory) as fixture:
            out = directory / ("finished.heic" if container == "heic" else "finished.jpg")
            out.write_bytes(b"previous")
            profile = replace(fixture.profile, container=container)
            options_seen, coded_seen, donors = [], [], []

            def write(path, options):
                self.assertEqual(out.read_bytes(), b"previous")
                auxiliary = round(options[fixture.quartz.kCGImageDestinationLossyCompressionQuality] * 100)
                options_seen.append(auxiliary)
                Path(path).write_text(json.dumps({"aux": auxiliary, "primary": "apple"}))
                return True, None

            fixture.context.writeHEIFRepresentationOfImage_toURL_format_colorSpace_options_error_.side_effect = (
                lambda image, path, fmt, space, options, error: write(path, options)
            )
            fixture.context.writeJPEGRepresentationOfImage_toURL_colorSpace_options_error_ = Mock(
                side_effect=lambda image, path, space, options, error: write(path, options)
            )

            def encode(base, donor, quality, chroma, **kwargs):
                self.assertEqual(out.read_bytes(), b"previous")
                self.assertFalse(base.flags.writeable)
                donors.append(donor)
                donor.write_text(json.dumps({"primary": f"q{quality}-{chroma}"}))
                return {"encoder": "x265"}

            def replace_jpeg(path, coded):
                coded_seen.append(coded)
                payload = json.loads(path.read_text())
                payload["primary"] = sha256(coded).hexdigest()
                path.write_text(json.dumps(payload))

            def hdr_gate(metrics, tolerances):
                self.assertEqual(out.read_bytes(), b"previous")
                # Exercise real pixel tolerances before the simulated auxiliary loss.
                self.assertTrue(original_hdr_gate(metrics, tolerances))
                return accepted[len(options_seen) - 1]

            original_hdr_gate = gainmap._hdr_roundtrip_is_acceptable
            with patch.object(heif_encoder, "encode", side_effect=encode) as heif_encode, \
                 patch.object(jpeg_gainmap, "encode_primary_codestream",
                              wraps=jpeg_gainmap.encode_primary_codestream) as jpeg_encode, \
                 patch.object(jpeg_gainmap, "replace_primary_codestream", side_effect=replace_jpeg), \
                 patch.object(jpeg_gainmap, "replace_primary", side_effect=lambda path, base, quality, chroma:
                              replace_jpeg(path, jpeg_gainmap.encode_primary_codestream(base, quality, chroma))), \
                 patch.object(gainmap, "_hdr_roundtrip_is_acceptable", side_effect=hdr_gate) as gate, \
                 patch.object(gainmap, "_new_hdr_metrics_workspace", return_value=None):
                fixture.out, fixture.profile = out, profile
                fixture.heif_encode, fixture.jpeg_encode, fixture.gate = heif_encode, jpeg_encode, gate
                fixture.options_seen, fixture.coded_seen, fixture.donors = options_seen, coded_seen, donors
                yield fixture

    def assert_full_verification(self, fixture, rounds):
        self.assertEqual(fixture.inspector.call_count, rounds)
        self.assertEqual(fixture.read.call_count, rounds)
        self.assertEqual(fixture.absolute_gate.call_count, rounds)
        self.assertEqual(fixture.hdr_read.call_count, rounds)
        self.assertEqual(fixture.gate.call_count, rounds)

    def assert_primary_encoded_once(self, fixture, container, rounds):
        if container == "heic":
            fixture.heif_encode.assert_called_once()
            fixture.jpeg_encode.assert_not_called()
            self.assertEqual(len(fixture.donors), 1)
            self.assertFalse(fixture.donors[0].exists())
        else:
            fixture.heif_encode.assert_not_called()
            fixture.jpeg_encode.assert_called_once()
            self.assertEqual(len(fixture.coded_seen), rounds)
            self.assertTrue(all(coded is fixture.coded_seen[0] for coded in fixture.coded_seen))

    def test_manual_retries_reuse_primary_but_verify_each_new_auxiliary(self):
        for container, qualities in (("heic", [95, 97, 98]), ("jpeg", [95, 100])):
            with self.subTest(container=container), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td), container, accepted=[False] * (len(qualities) - 1) + [True]) as fixture:
                info = gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                    delivery=fixture.profile, _verify_roundtrip_capability=False)
                self.assert_primary_encoded_once(fixture, container, len(qualities))
                self.assert_full_verification(fixture, len(qualities))
                self.assertEqual(fixture.options_seen, qualities)
                self.assertEqual(info["gainmap_encoding_quality"], qualities[-1])
                self.assertEqual(json.loads(fixture.out.read_text())["aux"], qualities[-1])
                self.assertEqual(set(Path(td).iterdir()), {fixture.template, fixture.out})

    def test_share_hq_retry_keeps_q97_420_and_original_dimensions(self):
        from PIL import Image, JpegImagePlugin
        import io

        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td), "jpeg", accepted=[False, True]) as fixture:
            profile = replace(fixture.profile, name="share-hq", quality=97, chroma="420")
            fixture.overrides["chroma_subsampling"] = "4:2:0"
            info = gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                delivery=profile, _verify_roundtrip_capability=False)
            self.assert_primary_encoded_once(fixture, "jpeg", 2)
            self.assert_full_verification(fixture, 2)
            self.assertEqual(fixture.jpeg_encode.call_args.args[1:], (97, "420"))
            self.assertEqual((info["delivery_quality"], info["delivery_chroma_requested"]), (97, "420"))
            with Image.open(io.BytesIO(fixture.coded_seen[0])) as image:
                self.assertEqual(image.size, (24, 16))
                self.assertEqual(JpegImagePlugin.get_sampling(image), 2)

    def test_absolute_sdr_failure_does_not_enter_auxiliary_ladder(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td), "heic", accepted=[]) as fixture:
            fixture.read.return_value = fixture.base * 0
            with self.assertRaisesRegex(RuntimeError, "SDR 底图"):
                gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                    delivery=fixture.profile, _verify_roundtrip_capability=False)
            self.assertEqual(fixture.options_seen, [95])
            fixture.heif_encode.assert_called_once()
            fixture.absolute_gate.assert_called_once()
            fixture.hdr_read.assert_not_called()
            fixture.gate.assert_not_called()
            self.assertFalse(fixture.donors[0].exists())
            self.assertEqual(fixture.out.read_bytes(), b"previous")
            self.assertEqual(set(Path(td).iterdir()), {fixture.template, fixture.out})

    def test_exhausted_auxiliary_ladder_keeps_previous_output_and_cleans_resources(self):
        for container, qualities in (("heic", [95, 97, 98, 99, 100]), ("jpeg", [95, 100])):
            with self.subTest(container=container), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td), container, accepted=[False] * len(qualities)) as fixture:
                with self.assertRaises(gainmap.HdrRoundtripError):
                    gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                        delivery=fixture.profile, _verify_roundtrip_capability=False)
                self.assert_primary_encoded_once(fixture, container, len(qualities))
                self.assert_full_verification(fixture, len(qualities))
                self.assertEqual(fixture.options_seen, qualities)
                self.assertEqual(fixture.out.read_bytes(), b"previous")
                self.assertEqual(set(Path(td).iterdir()), {fixture.template, fixture.out})

    def test_non_hdr_failure_does_not_retry_or_publish(self):
        with tempfile.TemporaryDirectory() as td, self.fixture(Path(td), "heic", accepted=[]) as fixture:
            fixture.overrides["width"] = 99
            with self.assertRaisesRegex(RuntimeError, "尺寸"):
                gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                    delivery=fixture.profile, _verify_roundtrip_capability=False)
            self.assertEqual(fixture.options_seen, [95])
            fixture.heif_encode.assert_called_once()
            fixture.gate.assert_not_called()
            self.assertFalse(fixture.donors[0].exists())
            self.assertEqual(fixture.out.read_bytes(), b"previous")
            self.assertEqual(set(Path(td).iterdir()), {fixture.template, fixture.out})

    def test_archive_and_explicit_auxiliary_precision_do_not_retry(self):
        for archive in (False, True):
            with self.subTest(archive=archive), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td), "heic", accepted=[False]) as fixture:
                profile = replace(fixture.profile, name="archive") if archive else fixture.profile
                kwargs = {} if archive else {"_gainmap_quality": 97}
                with self.assertRaises(gainmap.HdrRoundtripError):
                    gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                        delivery=profile, _verify_roundtrip_capability=False, **kwargs)
                self.assertEqual(fixture.options_seen, [100 if archive else 97])
                self.assert_primary_encoded_once(fixture, "heic", 1)
                self.assert_full_verification(fixture, 1)
                self.assertEqual(fixture.out.read_bytes(), b"previous")

    def test_apple_heif_does_not_create_an_x265_session_or_retry(self):
        for encoder in ("apple", "auto"):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td), "heic", accepted=[False]) as fixture, \
                 patch.object(heif_encoder, "available", return_value=False), \
                 patch("dngscan.gainmap_session.PrimarySearchSession", side_effect=AssertionError("x265 session")):
                profile = replace(fixture.profile, heif_encoder=encoder)
                with self.assertRaises(gainmap.HdrRoundtripError):
                    gainmap.write_apple_gainmap_file(fixture.base, fixture.hdr, fixture.out, 1.,
                        delivery=profile, _verify_roundtrip_capability=False)
                fixture.heif_encode.assert_not_called()
                self.assert_full_verification(fixture, 1)
                self.assertEqual(fixture.out.read_bytes(), b"previous")

    def test_supplied_primary_sessions_are_preserved_across_retries(self):
        for container, rounds in (("heic", 3), ("jpeg", 2)):
            with self.subTest(container=container), tempfile.TemporaryDirectory() as td, \
                 self.fixture(Path(td), container, accepted=[False] * (rounds - 1) + [True]) as fixture:
                prepared = gainmap._PreparedGainmapMaster(fixture.base, fixture.hdr, 1., verify_capability=False)
                try:
                    if container == "heic":
                        external_root = Path(td) / "external"
                        session = PrimarySearchSession(external_root)
                        kwargs = {"_primary_session": session}
                        constructor = "dngscan.gainmap_session.PrimarySearchSession"
                    else:
                        session = jpeg_gainmap.PrimaryCodestreamSession(prepared.base)
                        kwargs = {"_jpeg_primary_session": session}
                        constructor = "dngscan.jpeg_gainmap.PrimaryCodestreamSession"
                    precheck = Mock()
                    with patch(constructor, side_effect=AssertionError("supplied session replaced")):
                        info = gainmap._write_gainmap_prepared(prepared, fixture.out,
                            delivery=fixture.profile, use_heif=container == "heic",
                            _sdr_precheck=precheck, **kwargs)
                    self.assert_full_verification(fixture, rounds)
                    self.assertEqual(precheck.call_count, rounds)
                    if container == "heic":
                        fixture.heif_encode.assert_called_once()
                        self.assertTrue(fixture.donors[0].exists())
                        self.assertEqual(fixture.donors[0].parent, external_root)
                    else:
                        fixture.jpeg_encode.assert_called_once()
                    self.assertGreater(info["gainmap_encoding_quality"], 95)
                finally:
                    prepared.close()


if __name__ == "__main__":
    unittest.main()
