# SPDX-License-Identifier: GPL-3.0-or-later
"""Bounded donor reuse and conservative primary identity using real BMFF tables."""
from dataclasses import replace
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dngscan.delivery import resolve_delivery_profile
from dngscan.gainmap_session import PrimarySearchSession, _primary_identity
from dngscan.heif_gainmap import _box, _parse, _write_items


def write_fixture(path, *, primary=1, payload=b"primary", gain=b"gain", legacy=b"alpha",
                  transform=b"\0", essential=True, reverse=False, name=b"image\0",
                  icc=b"P3", outgoing=False, unknown_incoming=False, grid=False,
                  apple_meta=False, exif=b"metadata", unknown_tile_ref=False,
                  tone_payload=b"tone", exif_on_tile=False):
    gain_id, tone, auxiliary, tile = (primary + i for i in range(1, 5))

    def info(item, kind, item_name=b"image\0"):
        return b"\2\0\0\0" + struct.pack(">HH", item, 0) + kind + item_name

    infos = {primary: info(primary, b"grid" if grid else b"hvc1", name),
             gain_id: info(gain_id, b"hvc1"), tone: info(tone, b"tmap"),
             auxiliary: info(auxiliary, b"grid"), tile: info(tile, b"hvc1")}
    payloads = {primary: payload, gain_id: gain, tone: tone_payload, auxiliary: b"grid",
                tile: legacy}
    props = [(b"hvcC", b"codec"), (b"ispe", bytes(4) + struct.pack(">LL", 8, 8)),
             (b"colr", b"prof" + icc), (b"irot", transform),
             (b"auxC", b"\0\0\0\0urn:mpeg:hevc:2015:auxid:1\0")]
    main_assocs = [(True, 1), (True, 2), (essential, 3), (False, 4)]
    if reverse:
        main_assocs.reverse()
    assocs = {primary: main_assocs, gain_id: [(True, 1), (True, 2)],
              auxiliary: [(True, 2), (True, 5)], tile: [(True, 1), (True, 2)]}
    refs = [(b"dimg", tone, [primary, gain_id]), (b"auxl", auxiliary, [primary, tone]),
            (b"dimg", auxiliary, [tile])]
    metadata = primary + 5
    infos[metadata] = info(metadata, b"Exif")
    payloads[metadata] = exif
    refs.append((b"cdsc", metadata, [tile] if exif_on_tile else [primary, tone]))
    if unknown_tile_ref:
        refs.append((b"xxxx", gain_id, [tile]))
    if outgoing:
        refs.append((b"dimg", primary, [gain_id]))
    if unknown_incoming:
        refs.append((b"xxxx", gain_id, [primary]))
    children = [(b"pitm", bytes(4) + struct.pack(">H", primary)), (b"iinf", b""),
                (b"iref", b""), (b"iprp", b""), (b"iloc", b"")]
    if apple_meta:
        children.extend([
            (b"dinf", _box(b"dref", bytes(4) + struct.pack(">L", 1)
                           + _box(b"url ", b"\0\0\0\1"))),
            (b"grpl", _box(b"altr", bytes(4) + struct.pack(">LLLL", 100, 2, tone, primary))),
        ])
    _write_items(path, children, infos, refs, props, assocs, payloads,
                 b"heic\0\0\0\0heicmif1tmap")


class PrimarySearchSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.session = PrimarySearchSession(self.folder / "donors")
        self.base = np.zeros((8, 8, 3), np.uint8)
        self.base.flags.writeable = False
        self.profile = replace(resolve_delivery_profile("share", quality=90, chroma="444",
                                                       container="heic"), heif_encoder="x265")
        self.calls = []

        def encode(base, path, quality, chroma, **kwargs):
            self.assertIs(base, self.base)
            self.assertEqual(kwargs["output_gamut"], "p3")
            self.assertLessEqual(len(list(self.session.root.iterdir())), 2)
            self.calls.append((quality, chroma, kwargs))
            write_fixture(path)
            return {"delivery_quality": quality, "delivery_chroma_requested": chroma,
                    "bit_depth": kwargs["bit_depth"], "preset": kwargs["preset"],
                    "tune": kwargs["tune"], "encoder": "fake x265"}

        self.encoder = self.enterContext(patch("dngscan.heif_encoder.encode", side_effect=encode))

    def test_selection_pins_winner_across_candidate_churn_with_two_donor_bound(self):
        winner, info = self.session.primary(self.base, self.profile)
        self.session.select(info)
        previous = None
        for quality in (85, 80, 75, 70):
            candidate, _ = self.session.primary(self.base, replace(self.profile, quality=quality))
            self.assertTrue(winner.exists())
            if previous is not None:
                self.assertFalse(previous.exists())
            self.assertEqual(set(self.session.root.iterdir()), {winner, candidate})
            previous = candidate
        self.session.select({"delivery_quality": 12})
        reused, repeated = self.session.primary(self.base, self.profile)
        self.assertEqual(reused, winner)
        self.assertEqual(repeated, info)
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(list(self.session.root.iterdir()), [winner])

    def test_new_selection_replaces_pin_and_copies_encoder_metadata(self):
        old, old_info = self.session.primary(self.base, self.profile)
        self.session.select(old_info)
        new, info = self.session.primary(self.base, replace(self.profile, quality=80))
        self.session.select(info)
        self.assertFalse(old.exists())
        info["preset"] = "mutated"
        _, again = self.session.primary(self.base, replace(self.profile, quality=80))
        self.assertEqual(again["preset"], "slow")
        self.session.primary(self.base, replace(self.profile, quality=70))
        self.assertTrue(new.exists())

    def test_quality_chroma_depth_preset_and_tune_are_all_in_the_cache_key(self):
        for changes in ({}, {"quality": 85}, {"chroma": "422"}, {"heif_bit_depth": 8},
                        {"heif_preset": "fast"}, {"heif_tune": "psnr"}):
            profile = replace(self.profile, **changes)
            first, _ = self.session.primary(self.base, profile)
            second, _ = self.session.primary(self.base, profile)
            self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(len(list(self.session.root.iterdir())), 1)

    def test_failed_encode_cleans_partial_donor_and_preserves_winner(self):
        winner, info = self.session.primary(self.base, self.profile)
        self.session.select(info)
        self.session.primary(self.base, replace(self.profile, quality=85))

        def fail(base, path, *args, **kwargs):
            path.write_bytes(b"partial")
            raise RuntimeError("encoder failed")

        self.encoder.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.session.primary(self.base, replace(self.profile, quality=80))
        self.assertEqual(list(self.session.root.iterdir()), [winner])
        self.session.select({**info, "delivery_quality": 80})
        self.assertEqual(self.session.primary(self.base, self.profile)[0], winner)

    def test_changed_or_writeable_master_is_rejected_without_disturbing_winner(self):
        winner, info = self.session.primary(self.base, self.profile)
        self.session.select(info)
        changed = self.base.copy()
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.session.primary(changed, self.profile)
        changed.flags.writeable = False
        with self.assertRaisesRegex(ValueError, "change its master"):
            self.session.primary(changed, self.profile)
        self.assertEqual(list(self.session.root.iterdir()), [winner])

    def test_auxiliary_variants_reuse_only_scalar_sdr_metrics_for_current_key(self):
        self.session.primary(self.base, self.profile)
        path = self.folder / "candidate.heic"
        write_fixture(path)
        metrics = {"base_mean_code_error": .25, "coding_luma_rmse": np.float32(.5)}
        self.session.remember_metrics(path, metrics)
        metrics["base_mean_code_error"] = 99
        write_fixture(path, gain=b"changed ISO gain")
        cached = self.session.metrics(path)
        self.assertEqual(cached, {"base_mean_code_error": .25, "coding_luma_rmse": .5})
        cached["coding_luma_rmse"] = 999
        self.assertEqual(self.session.metrics(path)["coding_luma_rmse"], .5)
        self.session.primary(self.base, replace(self.profile, quality=80))
        self.assertIsNone(self.session.metrics(path))
        self.assertEqual(len(list(self.session.root.iterdir())), 1)

    def test_pixels_properties_and_legacy_alpha_dependency_invalidate_metrics(self):
        self.session.primary(self.base, self.profile)
        path = self.folder / "candidate.heic"
        write_fixture(path)
        self.session.remember_metrics(path, {"coding_luma_rmse": .5})
        for change in ({"payload": b"changed"}, {"icc": b"sRGB"}, {"transform": b"\1"},
                       {"essential": False}, {"reverse": True}, {"name": b"changed\0"},
                       {"legacy": b"changed alpha tile"}, {"exif": b"changed metadata"},
                       {"tone_payload": b"changed ISO parameters"}, {"exif_on_tile": True}):
            with self.subTest(change=change):
                write_fixture(path, **change)
                self.assertIsNone(self.session.metrics(path))
        write_fixture(path)
        self.assertEqual(self.session.metrics(path), {"coding_luma_rmse": .5})

    def test_unknown_layout_never_hits_or_overwrites_a_known_measurement(self):
        self.session.primary(self.base, self.profile)
        path = self.folder / "candidate.heic"
        write_fixture(path)
        self.session.remember_metrics(path, {"coding_luma_rmse": .5})
        for change in ({"outgoing": True}, {"unknown_incoming": True}, {"grid": True},
                       {"unknown_tile_ref": True}):
            write_fixture(path, **change)
            self.assertIsNone(self.session.metrics(path))
            self.session.remember_metrics(path, {"coding_luma_rmse": 99.})
        path.write_bytes(b"invalid HEIF")
        self.assertIsNone(self.session.metrics(path))
        write_fixture(path)
        self.assertEqual(self.session.metrics(path), {"coding_luma_rmse": .5})

    def test_metrics_reject_frames_and_hdr_statistics_without_cache_pollution(self):
        self.session.primary(self.base, self.profile)
        path = self.folder / "candidate.heic"
        write_fixture(path)
        self.session.remember_metrics(path, {"coding_luma_rmse": .5})
        for bad in ({"coding_luma_rmse": self.base}, {"chroma_error": .1}):
            with self.assertRaisesRegex(ValueError, "SDR numeric scalars"):
                self.session.remember_metrics(path, bad)
        self.assertEqual(self.session.metrics(path), {"coding_luma_rmse": .5})

    def test_identity_ignores_item_ids_and_accepts_known_apple_container_structure(self):
        a, b = self.folder / "a.heic", self.folder / "b.heic"
        for apple_meta in (False, True):
            write_fixture(a, apple_meta=apple_meta)
            write_fixture(b, primary=10, apple_meta=apple_meta)
            identity = _primary_identity(a)
            self.assertIsNotNone(identity)
            self.assertIsNotNone(_primary_identity(b))
            if not apple_meta:
                self.assertEqual(identity, _primary_identity(b))


class PrimaryIdentityLiveTests(unittest.TestCase):
    def test_actual_libheif_primary_is_cacheable(self):
        from dngscan import heif_encoder
        if not heif_encoder.available():
            self.skipTest("libheif/x265 required")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "primary.heic"
            heif_encoder.encode(np.zeros((16, 16, 3), np.uint8), path, 90, "444", preset="fast")
            self.assertEqual(_parse(path.read_bytes())[3][1][8:12], b"hvc1")
            self.assertIsNotNone(_primary_identity(path))


if __name__ == "__main__":
    unittest.main()
