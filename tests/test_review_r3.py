# SPDX-License-Identifier: GPL-3.0-or-later
"""Gates for review R3: non-Bayer NaN containment, DNG GainMap plane
semantics, default media scatter, guidance fullwell, the auto-EV brightness
reference on emitter scenes, and SDR JPEG metadata carry."""
from __future__ import annotations

import math
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SAMPLE_DNG = Path.home() / "Pictures" / "AgXRAW样张" / "_SDI0150.DNG"


class NonBayerCellMetricTests(unittest.TestCase):
    """R3 item 1: "not applicable" must never be spelled NaN downstream."""

    def test_channel_separation_is_zero_not_nan_without_topology(self) -> None:
        from dngscan.hdr_agx_plan import compile_channel_separation

        nan = float("nan")
        analysis = SimpleNamespace(
            cell_k_of_all_pct={k: nan for k in range(1, 5)},
            cell_union_pct=nan,
            cell_ge2_of_clipped_pct=nan,
            gamut_out_pct={"Display P3": 0.0},
        )
        rho = compile_channel_separation(analysis, "libraw")
        self.assertTrue(math.isfinite(rho))
        self.assertEqual(rho, 0.0)

    def test_rank_trim_treats_non_finite_as_stated_no_trim(self) -> None:
        from dngscan.tone import rank_trim_reconstructed_highlights

        ev = np.linspace(-5.0, 5.0, 1000).astype(np.float32)
        valid = np.ones(1000, dtype=bool)
        out = rank_trim_reconstructed_highlights(ev, valid, float("nan"))
        self.assertTrue(bool(np.array_equal(out, valid)))
        # And a finite rate still trims from the top.
        trimmed = rank_trim_reconstructed_highlights(ev, valid, 10.0)
        self.assertLess(int(trimmed.sum()), 1000)
        self.assertFalse(bool(trimmed[-1]))

    def test_generic_cell_union_measures_any_cfa_period(self) -> None:
        from dngscan.analysis import compute_generic_cell_union

        # X-Trans-like 6x6 period, 12x12 sensor -> 4 cells; one clipped
        # photosite -> union 25%.
        pattern = [[(r + c) % 3 for c in range(6)] for r in range(6)]
        raw = np.zeros((12, 12), dtype=np.uint16)
        raw[1, 1] = 5000
        colors = np.tile(np.asarray(pattern, dtype=np.uint8), (2, 2))
        union = compute_generic_cell_union(
            raw, colors, {0: 4000, 1: 4000, 2: 4000}, pattern
        )
        self.assertAlmostEqual(union, 25.0, places=6)


class GainMapPlaneSemanticsTests(unittest.TestCase):
    """R3 item 2: DNG SDK plane semantics, not an average over map planes."""

    @staticmethod
    def _payload(plane: int, planes: int, map_planes: int, values: list[float]) -> bytes:
        head = struct.pack(
            ">4L2L2L2L", 0, 0, 4, 4, plane, planes, 1, 1, 1, 1
        )
        tail = struct.pack(">4d", 1.0, 1.0, 0.0, 0.0) + struct.pack(">L", map_planes)
        gains = b"".join(struct.pack(">f", v) for v in values)
        return head + tail + gains

    def test_parser_keeps_plane_and_planes(self) -> None:
        from dngscan.metadata import _parse_gain_map_payload

        m = _parse_gain_map_payload(self._payload(0, 1, 2, [2.0, 9.0]))
        self.assertIsNotNone(m)
        self.assertEqual((m.plane, m.planes, m.map_planes), (0, 1, 2))

    def _apply(self, m) -> np.ndarray:
        from dngscan.raw_io import _apply_gain_maps_mosaic

        img = np.full((4, 4), 1000, dtype=np.uint16)
        raw = SimpleNamespace(
            raw_image_visible=img,
            raw_colors_visible=np.zeros((4, 4), dtype=np.uint8),
        )
        _apply_gain_maps_mosaic(raw, [m], [100.0], 4000, [4000.0])
        return img

    def test_mosaic_reads_map_plane_zero_not_the_average(self) -> None:
        from dngscan.metadata import _parse_gain_map_payload

        m = _parse_gain_map_payload(self._payload(0, 1, 2, [2.0, 9.0]))
        img = self._apply(m)
        # 100 + (1000-100)*2.0 = 1900 (plane 0), not the 5.5x average.
        self.assertEqual(int(img[0, 0]), 1900)

    def test_opcode_for_a_missing_image_plane_does_not_apply(self) -> None:
        from dngscan.metadata import _parse_gain_map_payload

        m = _parse_gain_map_payload(self._payload(1, 1, 1, [3.0]))
        img = self._apply(m)
        self.assertEqual(int(img[0, 0]), 1000)

    def test_result_caps_at_the_channel_white_level(self) -> None:
        from dngscan.metadata import _parse_gain_map_payload
        from dngscan.raw_io import _apply_gain_maps_mosaic

        m = _parse_gain_map_payload(self._payload(0, 1, 1, [4.0]))
        img = np.full((4, 4), 1000, dtype=np.uint16)
        raw = SimpleNamespace(
            raw_image_visible=img,
            raw_colors_visible=np.zeros((4, 4), dtype=np.uint8),
        )
        # channel white 3000 < global 4000: the sensel's own channel caps.
        _apply_gain_maps_mosaic(raw, [m], [100.0], 4000, [3000.0])
        self.assertEqual(int(img[0, 0]), 3000)


class DefaultMediaScatterTests(unittest.TestCase):
    """R3 item 3: declared media scatter engages without any look amount."""

    @staticmethod
    def _plan(**kw) -> SimpleNamespace:
        base = dict(
            curve_preset="portra400", film_mode="full", film_crossover="datasheet",
            film_exposure_ev=0.0, film_print_timing="fixed", film_print_medium="",
            film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default", film_dev_contrast=0.0,
            film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
            film_compression_knee=2.0, film_highlight_density=0.0,
            film_grain=0.0, film_halation=0.0, film_bloom=0.0,
            film_optics_seed=0, film_media_scatter="declared",
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_declared_engages_scatter_only_context_at_fine_pitch(self) -> None:
        from dngscan.film_develop import prepare_film_spatial

        ctx = prepare_film_spatial(self._plan(), 512, 4096)  # 8.8 um/px
        self.assertIsNotNone(ctx)
        self.assertGreater(ctx.scatter_halo_rows(), 0)
        self.assertEqual(ctx.grain, 0.0)
        self.assertEqual(ctx.halation, 0.0)
        self.assertEqual(ctx.bloom, 0.0)

    def test_off_and_observe_stay_identity(self) -> None:
        from dngscan.film_develop import prepare_film_spatial
        from dngscan.film_optics_assets import compile_film_optics_plan

        self.assertIsNone(
            prepare_film_spatial(self._plan(film_media_scatter="off"), 512, 4096)
        )
        self.assertIsNone(
            compile_film_optics_plan(self._plan(film_mode="observe"))
        )

    def test_default_full_render_carries_the_declared_scatter(self) -> None:
        from dngscan.film_develop import apply_film_core

        rng = np.random.default_rng(7)
        h, w = 96, 4096  # 8.8 um/px: both scatter stages resolve
        scene = (0.18 * np.exp2(rng.uniform(-3, 3, size=(h * w, 3)))).astype(np.float32)
        declared = apply_film_core(scene, self._plan(), spatial_shape=(h, w))
        off = apply_film_core(
            scene, self._plan(film_media_scatter="off"), spatial_shape=(h, w)
        )
        delta = float(np.max(np.abs(np.asarray(declared) - np.asarray(off))))
        self.assertGreater(delta, 1e-4, "declared media scatter must act by default")


class GuidanceFullwellTests(unittest.TestCase):
    """R3 item 4: headroom follows the analysis-resolved fullwell."""

    def test_resolved_fullwell_shrinks_headroom(self) -> None:
        from dngscan.guidance import _raw_headroom_rgb

        raw = np.full((4, 4), 3500, dtype=np.uint16)
        bundle = SimpleNamespace(
            raw_image=raw,
            raw_colors=np.zeros((4, 4), dtype=np.uint8),
            color_desc="RGBG",
            black_levels=[100.0],
            white_level=4000,
            camera_white_levels=[4000.0],
            orientation_flip=0,
        )
        meta = _raw_headroom_rgb(bundle, (2, 2), None)
        resolved = _raw_headroom_rgb(
            bundle, (2, 2), SimpleNamespace(channel_fullwell={0: 3600})
        )
        # 3500/4000-window headroom ~0.128; against the observed 3600 pile
        # the same sensel has only ~0.029 left.
        self.assertGreater(float(meta[0, 0, 0]), float(resolved[0, 0, 0]))
        self.assertAlmostEqual(float(resolved[0, 0, 0]), (3600 - 3500) / 3500.0, places=3)


class BrightnessReferenceEmitterTests(unittest.TestCase):
    """R3 item 5: pre-existing emitters must not veto the whole search."""

    def test_baseline_emitters_do_not_fail_the_percentile_gates(self) -> None:
        from dngscan.auto_ev import (
            NEAR_WHITE_LINEAR,
            output_highlight_margin,
            output_highlight_stats,
        )

        rng = np.random.default_rng(3)
        rgb = rng.uniform(0.05, 0.4, size=(100_000, 3)).astype(np.float32)
        rgb[:300] = 0.995  # lamps: already near-white at the baseline
        mask = np.max(rgb, axis=1) < np.float32(NEAR_WHITE_LINEAR)
        # Unmasked, the emitters own p99.9 and the absolute gate fails at
        # the STARTING EV — the recorded pathology.
        stats_full = output_highlight_stats(rgb, "p3")
        self.assertLessEqual(output_highlight_margin(rgb, "p3", stats_full), 0.0)
        # Masked to the reliable body, the baseline passes and the search
        # can run; the area budgets still police new clipping.
        stats_body = output_highlight_stats(rgb, "p3", mask)
        self.assertGreater(
            output_highlight_margin(rgb, "p3", stats_body, mask), 0.0
        )

    def test_area_budgets_still_police_new_clipping(self) -> None:
        from dngscan.auto_ev import (
            NEAR_WHITE_LINEAR,
            output_highlight_margin,
            output_highlight_stats,
        )

        rng = np.random.default_rng(4)
        base = rng.uniform(0.05, 0.4, size=(100_000, 3)).astype(np.float32)
        base[:300] = 0.995
        mask = np.max(base, axis=1) < np.float32(NEAR_WHITE_LINEAR)
        stats_body = output_highlight_stats(base, "p3", mask)
        boosted = np.clip(base * 4.0, 0.0, 1.0)
        self.assertLess(
            output_highlight_margin(boosted, "p3", stats_body, mask), 0.0
        )


@unittest.skipUnless(SAMPLE_DNG.is_file(), "sample frames unavailable")
class MetadataCarryTests(unittest.TestCase):
    """R3 item 6: the SDR JPEG carries the capture's EXIF, losslessly."""

    def test_exif_carried_pixels_untouched(self) -> None:
        import tempfile

        from PIL import Image
        from PIL.ExifTags import TAGS

        from dngscan.export import carry_capture_metadata, save_jpeg_array

        rgb = np.random.default_rng(0).integers(
            0, 255, size=(48, 64, 3), dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "meta.jpg"
            save_jpeg_array(rgb, out, 92, "p3", 0)
            before = np.asarray(Image.open(out))
            self.assertTrue(carry_capture_metadata(SAMPLE_DNG, out))
            img = Image.open(out)
            after = np.asarray(img)
            self.assertTrue(bool(np.array_equal(before, after)))
            self.assertIsNotNone(img.info.get("icc_profile"))
            named = {TAGS.get(k, k): v for k, v in img.getexif().items()}
            self.assertEqual(named.get("Make"), "SIGMA")
            self.assertEqual(int(named.get("Orientation", 0)), 1)
            # Mosaic-format descriptors must not follow the metadata over.
            self.assertNotIn("PhotometricInterpretation", named)


if __name__ == "__main__":
    unittest.main()
