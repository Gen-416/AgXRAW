# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for scene-linear pre-AgX transforms."""

from __future__ import annotations

import unittest

import numpy as np

from dngscan import scene_transform

import dngscan as dg


class SceneTransformTests(unittest.TestCase):
    def test_parallel_regions_are_bit_exact_to_serial_oracle(self) -> None:
        from dngscan.scene_transform import (
            _apply_scene_transform_rec2020_reference,
            apply_scene_transform_rec2020,
        )

        rng = dg.np.random.default_rng(20260804)
        rgb = rng.lognormal(0.0, 1.2, size=(80_000, 3)).astype(dg.np.float32)
        rgb[:3] = dg.np.asarray(
            [[dg.np.nan, 1.0, 0.0], [dg.np.inf, 0.2, 1.0], [-1.0, 0.0, 2.0]],
            dtype=dg.np.float32,
        )
        for transform in ("arri_skin_d55", "portra400_d55"):
            for adapt in (None, (0.82, 1.17), (0.91, 1.08, "coreimage")):
                expected = _apply_scene_transform_rec2020_reference(
                    rgb, transform, 1.3, adapt
                )
                actual = apply_scene_transform_rec2020(rgb, transform, 1.3, adapt)
                dg.np.testing.assert_array_equal(actual, expected)

    def test_strength_zero_is_identity(self) -> None:
        rgb = dg.np.asarray([[1.4, 1.0, 0.25], [0.2, 0.2, 0.2]], dtype=dg.np.float32)
        out = dg.apply_scene_transform_rec2020(rgb, "arri_skin_d55", 0.0)
        self.assertTrue(dg.np.allclose(out, rgb))

    def test_neutral_axis_is_preserved(self) -> None:
        rgb = dg.np.asarray([[0.18, 0.18, 0.18], [2.0, 2.0, 2.0]], dtype=dg.np.float32)
        out = dg.apply_scene_transform_rec2020(rgb, "arri_skin_d55", 1.0)
        self.assertTrue(dg.np.allclose(out, rgb, atol=1e-6))

    def test_skin_region_changes_colour(self) -> None:
        preset = dg.SCENE_TRANSFORMS["arri_skin_d55"]
        mu = preset.regions[0].mu_rg_bg
        rgb = dg.np.asarray([[mu[0], 1.0, mu[1]]], dtype=dg.np.float32)
        out = dg.apply_scene_transform_rec2020(rgb, "arri_skin_d55", 1.0)
        self.assertGreater(float(dg.np.max(dg.np.abs(out - rgb))), 1e-4)


    def test_wb_adaptation_identity_for_daylight(self) -> None:
        from dngscan.scene_transform import wb_adaptation_ratios

        self.assertIsNone(wb_adaptation_ratios("daylight", [1.5, 1.0, 2.3], [2.6, 1.3, 2.3]))
        self.assertIsNone(wb_adaptation_ratios("camera", None, [2.6, 1.3, 2.3]))
        # applied == daylight -> identity
        self.assertIsNone(wb_adaptation_ratios("camera", [2.6, 1.3, 2.3], [2.6, 1.3, 2.3]))

    def test_wb_adaptation_transports_anchor(self) -> None:
        from dngscan.scene_transform import SCENE_TRANSFORMS, _region_weight, wb_adaptation_ratios

        region = SCENE_TRANSFORMS["arri_skin_d55"].regions[0]
        ratios = wb_adaptation_ratios("camera", [1.484, 1.0, 2.328], [2.617, 1.312, 2.284])
        self.assertIsNotNone(ratios)
        r_r, r_b = ratios
        mu = region.mu_rg_bg
        # a pixel AT the transported anchor gets full weight under adaptation
        moved = dg.np.asarray([[mu[0] * r_r, 1.0, mu[1] * r_b]], dtype=dg.np.float32)
        w_adapt = float(_region_weight(moved, region, ratios)[0])
        self.assertGreater(w_adapt, 0.95)
        # while the ORIGINAL calibration anchor no longer peaks under adaptation
        original = dg.np.asarray([[mu[0], 1.0, mu[1]]], dtype=dg.np.float32)
        w_orig = float(_region_weight(original, region, ratios)[0])
        self.assertLess(w_orig, w_adapt)


class FullMatrixWindowTransportTests(unittest.TestCase):
    """The windows must follow the full hot-WB matrix, not a diagonal shortcut.

    The pixels receive Ctarget Gtarget (C0 G0)^-1 — a 3x3 with channel mixing.
    Measured on _SDI0150 + Portra 400 before this fix, the von Kries two-ratio
    transport left the foliage window at 6.9e-16 and neutral at 1.8e-35 where
    the true transformed center should weigh 1.
    """

    # A deliberately channel-mixing balance transform (row-normalized-ish).
    _M9 = (0.74, 0.02, 0.07, 0.0, 0.99, -0.02, -0.05, 0.10, 1.35)

    @staticmethod
    def _true_center(mu, m9):
        m = np.asarray(m9, dtype=np.float64).reshape(3, 3)
        v = m @ np.array([mu[0], 1.0, mu[1]], dtype=np.float64)
        return np.array([v[0] / v[1], v[2] / v[1]], dtype=np.float32)

    def test_center_weighs_one_at_the_projected_location(self) -> None:
        mu = (0.62, 0.85)
        cov = ((0.004, 0.0), (0.0, 0.006))
        center = self._true_center(mu, self._M9)[None, :]
        weight = scene_transform._gaussian_weight(
            center, mu, cov, 1.0, ("matrix", self._M9, None)
        )
        self.assertGreater(float(weight[0]), 0.999999)

    def test_diagonal_shortcut_is_not_equivalent_under_mixing(self) -> None:
        mu = (0.62, 0.85)
        cov = ((0.004, 0.0), (0.0, 0.006))
        center = self._true_center(mu, self._M9)[None, :]
        # The best diagonal approximation of the same transform.
        m = np.asarray(self._M9, dtype=np.float64).reshape(3, 3)
        ratios = (float(m[0, 0] / m[1, 1]), float(m[2, 2] / m[1, 1]))
        weight = scene_transform._gaussian_weight(center, mu, cov, 1.0, ratios)
        self.assertLess(float(weight[0]), 0.5)

    def test_covariance_follows_the_jacobian(self) -> None:
        mu = (0.62, 0.85)
        cov = ((0.004, 0.001), (0.001, 0.006))
        m = np.asarray(self._M9, dtype=np.float64).reshape(3, 3)
        # A one-sigma point pushed through the exact projective map should keep
        # its Mahalanobis distance (weight exp(-0.5)) under the transported window.
        vals, vecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
        sigma_pt = np.asarray(mu, dtype=np.float64) + vecs[:, 1] * np.sqrt(vals[1])
        v = m @ np.array([sigma_pt[0], 1.0, sigma_pt[1]])
        pushed = np.array([[v[0] / v[1], v[2] / v[1]]], dtype=np.float32)
        weight = scene_transform._gaussian_weight(
            pushed, mu, cov, 1.0, ("matrix", self._M9, None)
        )
        self.assertAlmostEqual(float(weight[0]), float(np.exp(-0.5)), delta=0.02)

    def test_decoder_ratios_still_apply_after_the_matrix(self) -> None:
        mu = (0.62, 0.85)
        cov = ((0.004, 0.0), (0.0, 0.006))
        center = self._true_center(mu, self._M9)[None, :]
        dec = (0.9, 1.1)
        moved = center * np.asarray(dec, dtype=np.float32)[None, :]
        weight = scene_transform._gaussian_weight(
            moved, mu, cov, 1.0, ("matrix", self._M9, dec)
        )
        self.assertGreater(float(weight[0]), 0.999999)


if __name__ == "__main__":
    unittest.main()

    def test_region_confidence_scales_effect(self) -> None:
        from dataclasses import replace as dc_replace

        from dngscan.scene_transform import SCENE_TRANSFORMS

        preset = SCENE_TRANSFORMS["alev_material_d55"]
        names = [r.name for r in preset.regions]
        self.assertEqual(set(names), {"skin", "foliage", "cyan", "neutral", "magenta"})
        for region in preset.regions:
            self.assertGreaterEqual(region.confidence, 0.0)
            self.assertLessEqual(region.confidence, 1.0)
        # zero-confidence copy must be inert even inside its own window
        skin = next(r for r in preset.regions if r.name == "skin")
        mu = skin.mu_rg_bg
        rgb = dg.np.asarray([[mu[0], 1.0, mu[1]]], dtype=dg.np.float32)
        import dngscan.scene_transform as st

        w_full = st._region_weight(rgb, skin, None) * skin.strength * skin.confidence
        zero = dc_replace(skin, confidence=0.0)
        w_zero = st._region_weight(rgb, zero, None) * zero.strength * zero.confidence
        self.assertGreater(float(w_full[0]), 0.0)
        self.assertEqual(float(w_zero[0]), 0.0)


class MixtureWindowTests(unittest.TestCase):
    """Multi-modal windows: one material under several illuminants."""

    @staticmethod
    def _region(**kw):
        from dngscan.scene_transform import SceneTransformRegion

        base = dict(
            name="skin",
            matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            mu_rg_bg=(1.90, 0.42),
            cov_rg_bg=((0.03, 0.0), (0.0, 0.004)),
            scale=1.5,
            strength=1.0,
        )
        base.update(kw)
        return SceneTransformRegion(**base)

    def test_absent_components_keep_legacy_behaviour(self) -> None:
        from dngscan.scene_transform import _region_weight

        rgb = dg.np.asarray([[1.90, 1.0, 0.42], [1.30, 1.0, 0.95]], dtype=dg.np.float32)
        legacy = _region_weight(rgb, self._region())
        explicit_empty = _region_weight(rgb, self._region(components=()))
        self.assertTrue(dg.np.allclose(legacy, explicit_empty))
        # the second illumination cluster is out of reach for a single Gaussian
        self.assertLess(float(legacy[1]), 0.01)

    def test_components_cover_second_illumination_cluster(self) -> None:
        from dngscan.scene_transform import SceneTransformComponent, _region_weight

        region = self._region(
            components=(
                SceneTransformComponent((1.90, 0.42), ((0.03, 0.0), (0.0, 0.004))),
                SceneTransformComponent((1.30, 0.95), ((0.03, 0.0), (0.0, 0.004))),
            )
        )
        rgb = dg.np.asarray([[1.90, 1.0, 0.42], [1.30, 1.0, 0.95]], dtype=dg.np.float32)
        w = _region_weight(rgb, region)
        self.assertGreater(float(w[0]), 0.95)
        self.assertGreater(float(w[1]), 0.95)

    def test_overlapping_components_do_not_exceed_one(self) -> None:
        # MAX, not sum: two lobes on the same spot must not double-count into >1.
        from dngscan.scene_transform import SceneTransformComponent, _region_weight

        region = self._region(
            components=(
                SceneTransformComponent((1.90, 0.42), ((0.03, 0.0), (0.0, 0.004))),
                SceneTransformComponent((1.90, 0.42), ((0.03, 0.0), (0.0, 0.004))),
            )
        )
        rgb = dg.np.asarray([[1.90, 1.0, 0.42]], dtype=dg.np.float32)
        self.assertLessEqual(float(_region_weight(rgb, region)[0]), 1.0 + 1e-6)

    def test_json_components_round_trip(self) -> None:
        from dngscan.scene_transform import _region_from_dict

        region = _region_from_dict("skin", {
            "name": "skin",
            "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "mu_rg_bg": [1.9, 0.42],
            "cov_rg_bg": [[0.03, 0.0], [0.0, 0.004]],
            "components": [
                {"mu_rg_bg": [1.9, 0.42], "cov_rg_bg": [[0.03, 0.0], [0.0, 0.004]], "weight": 0.7},
                {"mu_rg_bg": [1.3, 0.95], "cov_rg_bg": [[0.02, 0.0], [0.0, 0.005]], "weight": 0.3},
                {"broken": True},
            ],
        })
        self.assertEqual(len(region.components), 2)  # malformed entry skipped, not fatal
        self.assertAlmostEqual(region.components[0].weight, 0.7)


class DecoderTransportTests(unittest.TestCase):
    """Windows follow the pixels into the RAW 9 reference frame; pixels never move."""

    def test_libraw_is_identity(self) -> None:
        from dngscan.scene_transform import decoder_window_ratios

        self.assertIsNone(decoder_window_ratios("libraw", "skin"))

    def test_coreimage_transport_is_measured_and_bounded(self) -> None:
        from dngscan.scene_transform import decoder_window_ratios

        for name in ("skin", "foliage", "neutral", "unknown"):
            ratios = decoder_window_ratios("coreimage", name)
            self.assertIsNotNone(ratios)
            for r in ratios:
                self.assertGreater(r, 0.6)
                self.assertLess(r, 1.4)

    def test_composition_multiplies_wb_and_decoder(self) -> None:
        from dngscan.scene_transform import _compose_transport, decoder_window_ratios

        dec = decoder_window_ratios("coreimage", "skin")
        combined = _compose_transport((1.1, 0.9, "coreimage"), "skin")
        self.assertAlmostEqual(combined[0], 1.1 * dec[0], places=9)
        self.assertAlmostEqual(combined[1], 0.9 * dec[1], places=9)
        self.assertEqual(_compose_transport((1.1, 0.9), "skin"), (1.1, 0.9))
        self.assertIsNone(_compose_transport(None, "skin"))

    def test_wb_ratios_tag_nonlibraw_decoders(self) -> None:
        from dngscan.scene_transform import wb_adaptation_ratios

        tagged = wb_adaptation_ratios("5500k", [2.0, 1.0, 1.8], [2.6, 1.3, 2.3], "coreimage")
        self.assertEqual(len(tagged), 3)
        self.assertEqual(tagged[2], "coreimage")
        legacy = wb_adaptation_ratios("camera", [2.0, 1.0, 1.8], [2.6, 1.3, 2.3])
        self.assertEqual(len(legacy), 2)


class PerCameraTransportTests(unittest.TestCase):
    """Composite decoder|model tokens resolve per-camera scopes with honest fallback."""

    def test_transport_tag_composition(self) -> None:
        from types import SimpleNamespace
        from dngscan.scene_transform import window_transport_tag

        self.assertEqual(
            window_transport_tag(SimpleNamespace(scene_decoder="libraw", shot_model="fp")),
            "libraw",
        )
        self.assertEqual(
            window_transport_tag(
                SimpleNamespace(scene_decoder="coreimage", shot_model="Apple iPhone 16 Pro")
            ),
            "coreimage|Apple iPhone 16 Pro",
        )
        self.assertEqual(
            window_transport_tag(SimpleNamespace(scene_decoder="coreimage", shot_model="")),
            "coreimage",
        )

    def test_per_camera_scope_wins_and_unknown_falls_back(self) -> None:
        from dngscan.scene_transform import decoder_window_ratios

        iphone = decoder_window_ratios("coreimage|Apple iPhone 16 Pro", "skin")
        default = decoder_window_ratios("coreimage", "skin")
        unknown = decoder_window_ratios("coreimage|Some Future Camera", "skin")
        self.assertIsNotNone(iphone)
        self.assertIsNotNone(default)
        self.assertNotEqual(iphone, default)
        self.assertEqual(unknown, default)

    def test_measured_scopes_are_bounded_and_distinct(self) -> None:
        from dngscan.scene_transform import decoder_window_ratios

        for token in ("coreimage", "coreimage|Apple iPhone 16 Pro"):
            for cls in ("skin", "foliage", "magenta", "unknown"):
                ratios = decoder_window_ratios(token, cls)
                self.assertIsNotNone(ratios)
                for r in ratios:
                    self.assertGreater(r, 0.6)
                    self.assertLess(r, 1.35)
