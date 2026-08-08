# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from dngscan._deps import np
from dngscan.gui.preview_cache import (
    MAX_FRAME_CACHE_ITEMS,
    MAX_MEMORY_PROXY_ITEMS,
    MAX_PIXEL_CACHE_ITEMS,
    PreviewCache,
    PreviewEntry,
    _cache_identity,
    _evidence_cache_identity,
    _read_disk_entry,
    _write_disk_entry,
    build_proxy_entry,
    downsample_mean,
    proxy_target_size,
)
from dngscan.models import Analysis, RawBundle, RawGuidanceMaps


def _analysis() -> Analysis:
    channels = [0, 1, 2, 3]
    labels = {0: "R", 1: "G", 2: "B", 3: "G"}
    return Analysis(
        channel_ids=channels,
        labels=labels,
        ceilings={key: 1000 for key in channels},
        ceil_spike_counts={key: 0 for key in channels},
        ceil_near_counts={key: 0 for key in channels},
        ceil_spike_ok={key: False for key in channels},
        fullwell_channel_ids=channels,
        fullwell_note="test",
        saturation_levels={key: 1000 for key in channels},
        channel_fullwell={key: 1000 for key in channels},
        channel_thresholds={key: 996 for key in channels},
        fullwell=1000,
        threshold=996,
        clip_pct={key: 0.0 for key in channels},
        cfa_cell_supported=True,
        cell_union_pct=0.0,
        cell_ge2_of_clipped_pct=0.0,
        cell_k_of_clipped_pct={key: 0.0 for key in range(1, 5)},
        cell_k_of_all_pct={key: 0.0 for key in range(1, 5)},
        ev_p1=-6.0,
        ev_raw_p1=-6.0,
        ev_median=-1.0,
        ev_p99=2.0,
        ev_p999=3.0,
        ev_dr_p1_p999=9.0,
        ev_floor_hit_pct=0.0,
        median_vs_gray_ev=-1.0,
        median_y=0.1,
        noise_floor=0.001,
        usable_dr_ev=9.0,
        snr_curves={},
        snr1_dr={},
        snr1_stop={},
        gamut_out_pct={"sRGB": 0.1, "P3": 0.0, "Rec2020": 0.0},
        bright_pixel_pct=50.0,
        survivor_channel="G",
        container_bits_est=10,
    )


def _bundle() -> RawBundle:
    raw = np.arange(64, dtype=np.uint16).reshape(8, 8)
    colors = np.tile(np.asarray([[0, 1], [3, 2]], dtype=np.uint8), (4, 4))
    scene = np.arange(8 * 8 * 3, dtype=np.uint16).reshape(8, 8, 3)
    return RawBundle(
        path=Path("synthetic.dng"),
        raw_image=raw,
        raw_colors=colors,
        xyz_render=scene.copy(),
        render_scale=65535.0,
        scene_rec2020_render=scene,
        scene_scale=65535.0,
        white_level=1000,
        black_levels=[0.0] * 4,
        camera_wb=[2.0, 1.0, 1.5, 1.0],
        color_desc="RGBG",
        raw_pattern=[[0, 1], [3, 2]],
        camera_white_levels=[1000.0] * 4,
        daylight_wb=[1.6, 1.0, 2.1, 1.0],
        applied_wb=[2.0, 1.0, 1.5, 1.0],
        decode_wb=[2.0, 1.0, 1.5, 1.0],
        wb_xyz_to_cam=np.eye(3, dtype=np.float64),
        wb_color_matrix=np.hstack(
            [np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)]
        ),
        clip_masks=np.linspace(0.0, 1.0, 8 * 8 * 3, dtype=np.float16).reshape(8, 8, 3),
        scene_scale_mode="measured",
        baseline_exposure=0.75,
        scene_decoder_runtime="Version 27.0 (Build TEST)",
        scene_align_factor=0.875,
        scene_opcode_names=("WarpRectilinear", "GainMap"),
        evidence_provider_version="rawpy 0.27.0/LibRaw 0.22.0",
    )


class PreviewCacheTest(unittest.TestCase):
    def test_render_plan_transforms_its_sample_once(self) -> None:
        import dngscan.tone as tone

        original = tone.scene_transform_engine.apply_scene_transform_rec2020
        with patch.object(
            tone.scene_transform_engine,
            "apply_scene_transform_rec2020",
            wraps=original,
        ) as transform:
            tone.build_render_plan(
                _bundle(),
                _analysis(),
                "agx",
                "srgb",
                "portra400_d55",
                1.0,
            )
        self.assertEqual(transform.call_count, 1)

    def test_parallel_balance_analysis_is_bit_exact(self) -> None:
        from dataclasses import replace

        from dngscan.analysis import (
            compute_ev_metrics,
            compute_gamut_metrics,
            luminance_from_xyz_render,
            reanalyze_balanced_scene,
        )

        bundle = _bundle()
        capture = _analysis()
        y = luminance_from_xyz_render(bundle.xyz_render, bundle.render_scale)
        (_ev, raw_p1, p1, p50, p99, p999, dr, floor, gray) = compute_ev_metrics(y)
        gamut, bright = compute_gamut_metrics(
            bundle.xyz_render, bundle.render_scale, y
        )
        expected = replace(
            capture,
            ev_p1=p1,
            ev_raw_p1=raw_p1,
            ev_median=p50,
            ev_p99=p99,
            ev_p999=p999,
            ev_dr_p1_p999=dr,
            ev_floor_hit_pct=floor,
            median_vs_gray_ev=gray,
            median_y=float(np.median(y)),
            gamut_out_pct=gamut,
            bright_pixel_pct=bright,
        )
        actual = reanalyze_balanced_scene(capture, bundle)
        for name in (
            "ev_p1",
            "ev_raw_p1",
            "ev_median",
            "ev_p99",
            "ev_p999",
            "ev_dr_p1_p999",
            "ev_floor_hit_pct",
            "median_vs_gray_ev",
            "median_y",
            "gamut_out_pct",
            "bright_pixel_pct",
        ):
            self.assertEqual(getattr(actual, name), getattr(expected, name), name)

    def test_realtime_proxy_has_one_fixed_1920px_long_edge(self) -> None:
        image = np.zeros((600, 2400, 3), dtype=np.float32)
        proxy = downsample_mean(image)
        self.assertEqual(proxy.shape, (480, 1920, 3))

    def test_target_size_preserves_decoded_source_aspect_ratio(self) -> None:
        self.assertEqual(proxy_target_size(6000, 4000, 1920), (1920, 1280))
        self.assertEqual(proxy_target_size(7008, 4672, 1920), (1920, 1280))
        self.assertEqual(proxy_target_size(4000, 6000, 1920), (1280, 1920))

    def test_runtime_plan_and_frame_caches_are_bounded_lrus(self) -> None:
        entry = PreviewEntry(bundle=_bundle(), analysis=_analysis())
        builds = 0

        def build() -> object:
            nonlocal builds
            builds += 1
            return object()

        first = entry.get_or_build_plan(("agx", "srgb"), build)
        self.assertIs(entry.get_or_build_plan(("agx", "srgb"), build), first)
        self.assertEqual(builds, 1)

        for index in range(MAX_FRAME_CACHE_ITEMS + 2):
            entry.put_frame((index,), {"preview": str(index), "metrics": {"v": index}})
        self.assertIsNone(entry.get_frame((0,)))
        latest = entry.get_frame((MAX_FRAME_CACHE_ITEMS + 1,))
        self.assertIsNotNone(latest)
        assert latest is not None
        latest["metrics"]["v"] = -1
        self.assertEqual(
            entry.get_frame((MAX_FRAME_CACHE_ITEMS + 1,))["metrics"]["v"],
            MAX_FRAME_CACHE_ITEMS + 1,
        )

        for index in range(MAX_PIXEL_CACHE_ITEMS + 1):
            source = np.full((2, 3, 3), index, dtype=np.uint8)
            stored = entry.put_pixels((index,), source)
            source.fill(255)
            self.assertFalse(stored.flags.writeable)
        self.assertIsNone(entry.get_pixels((0,)))
        np.testing.assert_array_equal(
            entry.get_pixels((MAX_PIXEL_CACHE_ITEMS,)),
            np.full((2, 3, 3), MAX_PIXEL_CACHE_ITEMS, dtype=np.uint8),
        )

        first_noise = entry.get_or_build_dither_noise()
        second_noise = entry.get_or_build_dither_noise()
        self.assertIs(first_noise, second_noise)
        self.assertEqual(len(first_noise), 2)
        for plane in first_noise:
            self.assertFalse(plane.flags.writeable)
            self.assertEqual(plane.shape, entry.bundle.scene_rec2020_render.shape)

    def test_evidence_identity_is_scene_decoder_independent(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            path = Path(source.name)
            evidence_key = _evidence_cache_identity(path)
            libraw_key, _ = _cache_identity(path, "clip", "camera", "libraw", "auto")
            apple_key, _ = _cache_identity(
                path, "reconstruct", "camera", "coreimage", "9"
            )
        self.assertEqual(libraw_key[:6], evidence_key)  # 6-tuple since batch 17 (inode+header hash)
        self.assertEqual(apple_key[:6], evidence_key)
        self.assertNotEqual(libraw_key, apple_key)

    def test_cache_identity_includes_demosaic_recipe(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            path = Path(source.name)
            auto_key, auto_digest = _cache_identity(
                path, "clip", "camera", "libraw", "auto", "auto"
            )
            dht_key, dht_digest = _cache_identity(
                path, "clip", "camera", "libraw", "auto", "dht"
            )
        self.assertNotEqual(auto_key, dht_key)
        self.assertNotEqual(auto_digest, dht_digest)

    def test_cache_identity_excludes_user_white_balance(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            path = Path(source.name)
            camera_key, camera_digest = _cache_identity(
                path, "clip", "camera", "libraw", "auto", "dht"
            )
            daylight_key, daylight_digest = _cache_identity(
                path, "clip", "daylight", "libraw", "auto", "dht"
            )
        self.assertEqual(camera_key, daylight_key)
        self.assertEqual(camera_digest, daylight_digest)

    def test_switching_white_balance_reuses_fixed_decode_context(self) -> None:
        cache = PreviewCache()
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, patch(
            "dngscan.gui.preview_cache._read_disk_entry", return_value=None
        ), patch(
            "dngscan.gui.preview_cache._write_disk_entry"
        ), patch(
            "dngscan.gui.preview_cache.dg.load_raw", return_value=_bundle()
        ) as load, patch(
            "dngscan.gui.preview_cache.dg.analyze",
            return_value=(_analysis(), None, None),
        ) as analyze:
            path = Path(source.name)
            camera = cache.get(path, "clip", "camera", demosaic="dht")
            daylight = cache.get(path, "clip", "daylight", demosaic="dht")
            daylight_again = cache.get(path, "clip", "daylight", demosaic="dht")

        self.assertEqual(load.call_count, 1)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(load.call_args.kwargs["wb_mode"], "camera")
        self.assertIs(daylight_again, daylight)
        self.assertIsNot(camera, daylight)
        self.assertEqual(camera.bundle.wb_mode, "camera")
        self.assertEqual(daylight.bundle.wb_mode, "daylight")
        self.assertEqual(len(cache.entries), 1)

    def test_cold_proxy_uses_full_resolution_selected_demosaic(self) -> None:
        cache = PreviewCache()
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, patch(
            "dngscan.gui.preview_cache._read_disk_entry", return_value=None
        ), patch(
            "dngscan.gui.preview_cache._write_disk_entry"
        ), patch(
            "dngscan.gui.preview_cache.dg.load_raw", return_value=_bundle()
        ) as load, patch(
            "dngscan.gui.preview_cache.dg.analyze",
            return_value=(_analysis(), None, None),
        ):
            cache.get(Path(source.name), "clip", "camera", demosaic="dht")
        self.assertFalse(load.call_args.kwargs["scene_half_size"])
        self.assertEqual(load.call_args.kwargs["demosaic"], "dht")

    def test_memory_proxy_cache_is_a_bounded_lru(self) -> None:
        cache = PreviewCache()
        entries = [PreviewEntry(_bundle(), _analysis()) for _ in range(3)]
        with tempfile.NamedTemporaryFile(suffix=".dng") as source, patch(
            "dngscan.gui.preview_cache._read_disk_entry", side_effect=entries
        ) as read:
            path = Path(source.name)
            cache.get(path, "clip", "camera")
            cache.get(path, "blend", "camera")
            cache.get(path, "reconstruct", "camera")
            cache.get(path, "blend", "camera")
        self.assertEqual(len(cache.entries), MAX_MEMORY_PROXY_ITEMS)
        self.assertEqual(read.call_count, 3)

    def test_round_trip_keeps_compact_proxy_and_guidance(self) -> None:
        entry = build_proxy_entry(_bundle(), _analysis())
        entry.bundle.raw_guidance = RawGuidanceMaps(
            headroom=np.full((8, 8, 3), 0.8, dtype=np.float16),
            clip_class=np.full((8, 8), 3, dtype=np.uint8),
            snr_confidence=np.full((8, 8), 0.7, dtype=np.float16),
        )
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "preview.npz"
            _write_disk_entry(cache_path, entry)
            restored = _read_disk_entry(cache_path, Path("synthetic.dng"), require_guidance=True)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertIsNone(restored.bundle.raw_image)
        self.assertIsNone(restored.bundle.raw_colors)
        np.testing.assert_array_equal(
            restored.bundle.scene_rec2020_render,
            entry.bundle.scene_rec2020_render,
        )
        np.testing.assert_array_equal(restored.bundle.clip_masks, entry.bundle.clip_masks)
        self.assertEqual(restored.analysis.labels, entry.analysis.labels)
        self.assertEqual(restored.analysis.channel_thresholds, entry.analysis.channel_thresholds)
        self.assertEqual(restored.bundle.scene_scale_mode, "measured")
        self.assertEqual(restored.bundle.baseline_exposure, 0.75)
        self.assertEqual(
            restored.bundle.scene_decoder_runtime, "Version 27.0 (Build TEST)"
        )
        self.assertEqual(restored.bundle.scene_align_factor, 0.875)
        self.assertEqual(
            restored.bundle.scene_opcode_names, ("WarpRectilinear", "GainMap")
        )
        self.assertEqual(restored.bundle.evidence_provider, "libraw")
        self.assertEqual(
            restored.bundle.evidence_provider_version,
            "rawpy 0.27.0/LibRaw 0.22.0",
        )
        self.assertEqual(restored.bundle.decode_wb, [2.0, 1.0, 1.5, 1.0])
        self.assertEqual(restored.bundle.applied_wb, [2.0, 1.0, 1.5, 1.0])
        np.testing.assert_array_equal(restored.bundle.wb_xyz_to_cam, np.eye(3))
        np.testing.assert_array_equal(
            restored.bundle.wb_color_matrix,
            np.hstack([np.eye(3), np.zeros((3, 1))]),
        )
        assert restored.bundle.raw_guidance is not None
        np.testing.assert_array_equal(
            restored.bundle.raw_guidance.clip_class,
            entry.bundle.raw_guidance.clip_class,
        )

    def test_raw9_signed_half_proxy_round_trip(self) -> None:
        bundle = _bundle()
        bundle.scene_decoder = "coreimage"
        bundle.scene_scale = 1.0
        bundle.render_scale = 1.0
        bundle.clip_masks = None
        bundle.scene_rec2020_render = np.linspace(
            -0.25, 2.5, 8 * 8 * 3, dtype=np.float16
        ).reshape(8, 8, 3)
        entry = build_proxy_entry(bundle, _analysis())

        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "raw9.npz"
            _write_disk_entry(cache_path, entry)
            restored = _read_disk_entry(
                cache_path, Path("synthetic.dng"), require_guidance=False
            )

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored.bundle.scene_rec2020_render.dtype, np.float16)
        self.assertEqual(restored.bundle.scene_scale, 1.0)
        self.assertLess(float(restored.bundle.scene_rec2020_render.min()), 0.0)
        self.assertGreater(float(restored.bundle.scene_rec2020_render.max()), 1.0)


if __name__ == "__main__":
    unittest.main()


class SceneDecoderRuntimeBindingTest(unittest.TestCase):
    """Core Image pixels are only valid for the OS build that decoded them."""

    def test_coreimage_identity_binds_the_system_decoder_build(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".dng") as source:
            path = Path(source.name)
            with patch(
                "dngscan.coreimage_decode.decoder_runtime_id", return_value="macOS 26.0"
            ):
                key_a, digest_a = _cache_identity(
                    path, "reconstruct", "camera", "coreimage", "9", "auto"
                )
                libraw_a = _cache_identity(path, "clip", "camera", "libraw", "auto", "auto")
            with patch(
                "dngscan.coreimage_decode.decoder_runtime_id", return_value="macOS 26.1"
            ):
                key_b, digest_b = _cache_identity(
                    path, "reconstruct", "camera", "coreimage", "9", "auto"
                )
                libraw_b = _cache_identity(path, "clip", "camera", "libraw", "auto", "auto")
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(digest_a, digest_b)
        # A LibRaw decode does not depend on the Apple decoder build.
        self.assertEqual(libraw_a, libraw_b)

    def test_read_rejects_an_entry_from_a_foreign_decoder_build(self) -> None:
        import json as _json
        from dngscan.gui.preview_cache import PREVIEW_CACHE_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "shot.dng"
            raw.write_bytes(b"II*\x00 fake raw payload")
            entry = tmp_path / "entry.npz"
            metadata = {
                "version": PREVIEW_CACHE_VERSION,
                "has_guidance": False,
                "bundle": {"scene_decoder_runtime": "macOS 26.0"},
            }
            with open(entry, "wb") as handle:
                np.savez(handle, metadata=np.asarray(_json.dumps(metadata)))
            # Foreign build: rejected before any payload is touched; the file survives.
            self.assertIsNone(
                _read_disk_entry(entry, raw, False, expected_runtime="macOS 26.1")
            )
            self.assertTrue(entry.exists())
            # Matching build: the runtime gate passes and the (deliberately truncated)
            # payload is then discarded by the integrity path, which unlinks the file.
            self.assertIsNone(
                _read_disk_entry(entry, raw, False, expected_runtime="macOS 26.0")
            )
            self.assertFalse(entry.exists())


class ExportAnalysisReuseTest(unittest.TestCase):
    """run_export must reuse the persisted full-res Analysis only on exact identity."""

    def _write_cache_npz(self, cache_dir: Path, digest: str, version: int) -> Path:
        import json as _json
        from dataclasses import asdict

        payload_path = cache_dir / f"{digest}.npz"
        metadata = {"version": version, "analysis": asdict(_analysis())}
        with open(payload_path, "wb") as handle:
            np.savez(handle, metadata=np.asarray(_json.dumps(metadata)))
        return payload_path

    def test_hit_returns_the_stored_analysis_and_stale_entries_miss(self) -> None:
        import os
        from dngscan.gui.preview_cache import PREVIEW_CACHE_VERSION
        from dngscan.gui.service import _cached_full_analysis

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "shot.dng"
            raw.write_bytes(b"II*\x00 fake raw payload")
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()
            with patch.dict(os.environ, {"DNGSCAN_PREVIEW_CACHE_DIR": str(cache_dir)}):
                _, digest = _cache_identity(raw, "clip", "camera", "libraw", "auto", "auto")
                self._write_cache_npz(cache_dir, digest, PREVIEW_CACHE_VERSION)

                hit = _cached_full_analysis(raw, "clip", "camera", "libraw", "auto", "auto")
                self.assertIsNotNone(hit)
                # repr-compare: dataclass == is False whenever any field is NaN.
                self.assertEqual(repr(hit), repr(_analysis()))

                # Different decode parameters -> different digest -> miss.
                self.assertIsNone(
                    _cached_full_analysis(raw, "blend", "camera", "libraw", "auto", "auto")
                )

                # File modified after the cache was written -> identity changes -> miss.
                raw.write_bytes(b"II*\x00 fake raw payload, edited")
                os.utime(raw, ns=(1, 1))
                self.assertIsNone(
                    _cached_full_analysis(raw, "clip", "camera", "libraw", "auto", "auto")
                )

    def test_schema_version_bump_invalidates_the_entry(self) -> None:
        import os
        from dngscan.gui.preview_cache import PREVIEW_CACHE_VERSION
        from dngscan.gui.service import _cached_full_analysis

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "shot.dng"
            raw.write_bytes(b"II*\x00 fake raw payload")
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir()
            with patch.dict(os.environ, {"DNGSCAN_PREVIEW_CACHE_DIR": str(cache_dir)}):
                _, digest = _cache_identity(raw, "clip", "camera", "libraw", "auto", "auto")
                self._write_cache_npz(cache_dir, digest, PREVIEW_CACHE_VERSION - 1)
                self.assertIsNone(
                    _cached_full_analysis(raw, "clip", "camera", "libraw", "auto", "auto")
                )


class DegradedBalanceOnProxyTests(unittest.TestCase):
    """A degraded rebalance must never require the proxy's absent xyz_render.

    GUI BalanceContexts feed proxy DecodeContexts whose xyz_render is deliberately
    None. When the requested balance degrades back to camera (missing multipliers or
    calibration), the balanced entry must still be renderable and must carry the
    degradation note to the UI — a scene-only reanalysis over None is the crash this
    pins down (previously a TypeError -> /preview 500).
    """

    def _proxy_entry(self, **bundle_overrides) -> PreviewEntry:
        from dataclasses import replace

        bundle = replace(_bundle(), xyz_render=None, **bundle_overrides)
        return PreviewEntry(bundle=bundle, analysis=_analysis())

    def test_degraded_balance_keeps_camera_pixels_and_the_note(self) -> None:
        entry = self._proxy_entry(daylight_wb=None)
        balanced = PreviewCache._build_balance(entry, "daylight")
        self.assertEqual(balanced.bundle.wb_mode, "camera")
        self.assertIsNotNone(balanced.bundle.wb_degradation)
        self.assertIn("daylight", balanced.bundle.wb_degradation)
        # Scene pixels are exactly the base proxy's, so the persisted camera
        # analysis is already the truth for them: no reanalysis, no crash.
        self.assertIs(balanced.analysis, entry.analysis)
        self.assertIs(
            balanced.bundle.scene_rec2020_render, entry.bundle.scene_rec2020_render
        )

    def test_successful_balance_still_reanalyzes_the_new_scene(self) -> None:
        entry = PreviewEntry(bundle=_bundle(), analysis=_analysis())
        balanced = PreviewCache._build_balance(entry, "daylight")
        self.assertEqual(balanced.bundle.wb_mode, "daylight")
        self.assertIsNone(balanced.bundle.wb_degradation)
        self.assertIsNot(balanced.analysis, entry.analysis)
