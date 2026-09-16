# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage 1 of the Rust migration (2026-09-15): decode-side evidence and
analysis/delivery metrics. The NumPy bodies stay the reference; each kernel
is pinned against them on random inputs — bit-identical where the NumPy
computation is elementwise/order-defined (feathering, GainMap opcodes, gamut
counts, HDR round-trip metrics), and within float64 last-bits for the two
sum-based base round-trip statistics (declared: sequential vs pairwise sum).

The NumPy semantics the Rust side replicates were pinned by experiment on
NumPy 2.5: median of float32 = float32 (a+b)/2 of the middle pair;
percentile 'linear' = _lerp with gamma cast to float32 (reversed form for
gamma >= 0.5); reshape(bh,8,bw,8,c).mean(axis=(1,3)) = sequential float32
sum over the 64 samples in (i, j) order, divided by 64.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from dngscan import _fast


def _native_available() -> bool:
    return _fast._load_extension() is not None and hasattr(_fast._load_extension(), "feather_masks_f16")


class _NoNative:
    """Force the NumPy reference body inside a dispatching function."""

    def __enter__(self):
        self._p = mock.patch.object(_fast, "kernel", lambda name: None)
        self._p.start()
        return self

    def __exit__(self, *exc):
        self._p.stop()
        return False


@unittest.skipUnless(_native_available(), "native extension not built")
class FeatherMasks(unittest.TestCase):
    def test_bit_identical(self) -> None:
        from dngscan import raw_io

        rng = np.random.default_rng(1)
        for (h, w, c) in ((1, 1, 3), (7, 5, 3), (64, 97, 3), (300, 1201, 3), (5, 3, 1)):
            mask = rng.uniform(-0.2, 1.3, (h, w, c)).astype(np.float32)
            mask[rng.uniform(size=(h, w, c)) < 0.3] = 0.0
            with _NoNative():
                ref = raw_io._feather_masks_f16(mask)
            got = raw_io._feather_masks_f16(mask)
            self.assertEqual(got.dtype, np.float16)
            self.assertEqual(got.shape, ref.shape)
            self.assertTrue(np.array_equal(got, ref), (h, w, c))


@unittest.skipUnless(_native_available(), "native extension not built")
class GainMapOpcodes(unittest.TestCase):
    def _case(self, rng, h, w, pitch, pv, ph, planes, scalar_tables):
        img = rng.integers(0, 16000, (h, w), dtype=np.uint16)
        colors = rng.integers(0, 4, (h, w)).astype(np.uint8)
        raw = SimpleNamespace(raw_image_visible=img, raw_colors_visible=colors)
        gains = rng.uniform(0.8, 1.4, (pv, ph, planes))
        op = SimpleNamespace(
            top=1, left=2, bottom=h - 1, right=w - 1, row_pitch=pitch, col_pitch=pitch,
            origin_v=0.0, origin_h=0.0, spacing_v=1.0 / max(pv - 1, 1), spacing_h=1.0 / max(ph - 1, 1),
            points_v=pv, points_h=ph, gains=gains, plane=0, planes=1,
        )
        blacks = [512.0] if scalar_tables else [500.0, 512.0, 520.0, 530.0]
        whites = [15000.0] if scalar_tables else [15000.0, 15500.0, 14000.0, 15200.0]
        return raw, op, blacks, whites

    def test_bit_identical(self) -> None:
        from dngscan import raw_io

        rng = np.random.default_rng(2)
        for (h, w, pitch, pv, ph, planes, scalar) in (
            (16, 24, 1, 5, 7, 1, True), (33, 41, 2, 9, 3, 3, False),
            (40, 40, 1, 1, 1, 1, False), (25, 61, 3, 2, 8, 1, True),
        ):
            raw, op, blacks, whites = self._case(rng, h, w, pitch, pv, ph, planes, scalar)
            ref_img = raw.raw_image_visible.copy()
            got_img = raw.raw_image_visible.copy()
            with _NoNative():
                raw_io._apply_gain_maps_mosaic(
                    SimpleNamespace(raw_image_visible=ref_img, raw_colors_visible=raw.raw_colors_visible),
                    [op], blacks, 15000, whites,
                )
            raw_io._apply_gain_maps_mosaic(
                SimpleNamespace(raw_image_visible=got_img, raw_colors_visible=raw.raw_colors_visible),
                [op], blacks, 15000, whites,
            )
            self.assertFalse(np.array_equal(ref_img, raw.raw_image_visible), "gain map must change something")
            self.assertTrue(np.array_equal(got_img, ref_img), (h, w, pitch, pv, ph))

    def test_strided_visible_window(self) -> None:
        """rawpy's visible view is a strided window of the full frame."""
        from dngscan import raw_io

        rng = np.random.default_rng(3)
        full = rng.integers(0, 16000, (50, 70), dtype=np.uint16)
        colors_full = rng.integers(0, 4, (50, 70)).astype(np.uint8)
        gains = rng.uniform(0.9, 1.3, (4, 6, 1))
        op = SimpleNamespace(top=0, left=0, bottom=44, right=60, row_pitch=1, col_pitch=1,
                             origin_v=0.0, origin_h=0.0, spacing_v=1 / 3, spacing_h=1 / 5,
                             points_v=4, points_h=6, gains=gains, plane=0, planes=1)
        a = full.copy(); b = full.copy()
        with _NoNative():
            raw_io._apply_gain_maps_mosaic(SimpleNamespace(raw_image_visible=a[3:47, 5:65], raw_colors_visible=colors_full[3:47, 5:65]), [op], [400.0], 15000, [15000.0])
        raw_io._apply_gain_maps_mosaic(SimpleNamespace(raw_image_visible=b[3:47, 5:65], raw_colors_visible=colors_full[3:47, 5:65]), [op], [400.0], 15000, [15000.0])
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a[:3], full[:3]))  # outside the window untouched


@unittest.skipUnless(_native_available(), "native extension not built")
class GamutCounts(unittest.TestCase):
    def test_bit_identical(self) -> None:
        from dngscan import analysis

        rng = np.random.default_rng(4)
        for (h, w, ch) in ((1, 1, 3), (40, 50, 3), (120, 130, 4), (2, 3, 3)):
            scene = (np.exp2(rng.uniform(-8, 4, (h, w, ch))) * 0.18).astype(np.float32)
            scene[..., :3] *= rng.uniform(0.2, 3.0, (h, w, 3)).astype(np.float32)  # push some out of gamut
            scene[rng.uniform(size=(h, w)) < 0.05] = 0.0
            if h * w > 4:
                scene[0, 0, :3] = [np.inf, 0.1, 0.2]
                scene[0, 1, :3] = [np.nan, 0.1, 0.2]
            y = np.clip(rng.uniform(0, 2, (h, w)).astype(np.float32), 0, None)
            for names in (None, ("sRGB",), ("Rec2020", "DisplayP3")):
                with _NoNative():
                    ref = analysis.compute_gamut_metrics(scene, 1.7, y, names)
                got = analysis.compute_gamut_metrics(scene, 1.7, y, names)
                self.assertEqual(got, ref, (h, w, ch, names))


@unittest.skipUnless(_native_available(), "native extension not built")
class RoundTripMetrics(unittest.TestCase):
    def test_hdr_bit_identical(self) -> None:
        from dngscan import gainmap

        rng = np.random.default_rng(5)
        for (h, w) in ((8, 8), (17, 23), (64, 96), (130, 257)):
            intended = (np.exp2(rng.uniform(-6, 2, (h, w, 3)))).astype(np.float16)
            noise = rng.normal(0, 0.02, (h, w, 3)).astype(np.float32)
            expanded = (intended.astype(np.float32) * (1 + noise)).astype(np.float16)
            expanded[rng.uniform(size=(h, w)) < 0.02] = 0
            with _NoNative():
                ref = gainmap._roundtrip_error_arrays(expanded, intended)
            got = gainmap._roundtrip_error_arrays(expanded, intended)
            self.assertEqual(set(got), set(ref))
            for k in ref:
                self.assertEqual(got[k], ref[k], (k, h, w))

    def test_base_exact_except_declared_sums(self) -> None:
        from dngscan import gainmap

        rng = np.random.default_rng(6)
        for (h, w) in ((8, 8), (19, 21), (64, 100), (128, 300)):
            intended = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            decoded = np.clip(intended.astype(np.int16) + rng.integers(-6, 7, (h, w, 3)), 0, 255).astype(np.uint8)
            with _NoNative():
                ref = gainmap._base_roundtrip_error_arrays(decoded, intended)
            got = gainmap._base_roundtrip_error_arrays(decoded, intended)
            for k in ("base_p99_code_error", "base_max_code_error", "base_block_p99_code_error"):
                self.assertEqual(got[k], ref[k], k)
            for k in ("base_mean_code_error", "base_channel_bias_code_error"):
                self.assertAlmostEqual(got[k], ref[k], delta=1e-9 * max(1.0, abs(ref[k])), msg=k)


class NumpySemanticsPinned(unittest.TestCase):
    """The experiments the Rust replicas rest on, kept as tests."""

    def test_median_percentile_and_block_mean(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.uniform(0, 3, 10_000).astype(np.float32)
        s = np.sort(a)
        self.assertEqual(np.median(a), np.float32((s[4999] + s[5000]) / np.float32(2)))
        for q in (95.0, 99.0, 99.9):
            pos = (a.size - 1) * (q / 100.0)
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            g = np.float32(pos - lo)
            d = np.float32(s[hi] - s[lo])
            lerp = s[hi] - d * (np.float32(1) - g) if g >= 0.5 else s[lo] + d * g
            self.assertEqual(np.percentile(a, q), lerp)
        x = rng.uniform(0, 4, (2, 8, 3, 8, 3)).astype(np.float32)
        acc = np.zeros((2, 3, 3), np.float32)
        for i in range(8):
            for j in range(8):
                acc = acc + x[:, i, :, j]
        self.assertTrue(np.array_equal(x.mean(axis=(1, 3)), acc / np.float32(64)))


if __name__ == "__main__":
    unittest.main()
