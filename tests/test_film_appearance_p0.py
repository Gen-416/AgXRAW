# SPDX-License-Identifier: GPL-3.0-or-later
"""FILM_APPEARANCE_RECIPE phase P0 gates.

Same order of business as the optics P0: prove the instrument before trusting
what it says, then pin what it said.

The colour maths is checked against the CIE's own CIEDE2000 test pairs rather
than against itself, because every acceptance threshold in the appearance plan
(§15.2) is quoted in dE00 and a private dE00 would make those numbers
incomparable to anybody else's.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

from dngscan import film_palette_diag as pal

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "tests" / "appearance_freeze"

# Sharma, Wu & Dalal (2005), the published CIEDE2000 verification set. Eight
# rows spanning the blue-hue rotation term, the neutral region and the low-L
# corner where the implementations usually diverge.
SHARMA_PAIRS = (
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
)


class ColourMathTests(unittest.TestCase):
    def test_delta_e00_matches_the_published_verification_set(self) -> None:
        for lab1, lab2, expect in SHARMA_PAIRS:
            got = float(pal.delta_e00(np.array(lab1), np.array(lab2)))
            self.assertAlmostEqual(got, expect, places=3, msg=f"{lab1} vs {lab2}")

    def test_oklab_round_trips(self) -> None:
        rng = np.random.default_rng(4)
        rgb = rng.random((512, 3)) * 4.0
        back = pal.oklab_to_rec2020(pal.rec2020_to_oklab(rgb))
        np.testing.assert_allclose(back, rgb, rtol=1e-6, atol=1e-9)

    def test_rec2020_white_is_lab_100_and_neutral(self) -> None:
        lab = pal.rec2020_to_lab(np.ones((1, 3)))
        self.assertAlmostEqual(float(lab[0, 0]), 100.0, delta=1e-6)
        self.assertAlmostEqual(float(lab[0, 1]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(lab[0, 2]), 0.0, delta=1e-6)

    def test_compare_separates_hue_rotation_from_chroma_gain(self) -> None:
        """The decomposition exists so that "stronger colour" can be told from
        "different colour"; if a pure rotation leaked into the chroma term the
        plan's own failure criterion (§17 risk 2) would be unmeasurable."""
        lab = np.array([[0.7, 0.10, 0.02]])
        base = pal.oklab_to_rec2020(lab)
        angle = np.radians(12.0)
        rot = np.array([[
            0.7,
            0.10 * np.cos(angle) - 0.02 * np.sin(angle),
            0.10 * np.sin(angle) + 0.02 * np.cos(angle),
        ]])
        d = pal.compare(base, pal.oklab_to_rec2020(rot))
        self.assertAlmostEqual(float(d["d_hue_deg"][0]), 12.0, delta=0.05)
        self.assertAlmostEqual(float(d["log2_chroma_ratio"][0]), 0.0, delta=1e-6)

        scaled = pal.oklab_to_rec2020(np.array([[0.7, 0.20, 0.04]]))
        d2 = pal.compare(base, scaled)
        self.assertAlmostEqual(float(d2["log2_chroma_ratio"][0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(d2["d_hue_deg"][0]), 0.0, delta=0.02)

    def test_compare_of_identical_inputs_is_zero(self) -> None:
        volume, _ = pal.palette_volume()
        d = pal.compare(volume, volume)
        self.assertAlmostEqual(float(np.nanmax(np.abs(d["d_hue_deg"]))), 0.0, delta=1e-9)
        self.assertAlmostEqual(float(np.max(d["delta_e00"])), 0.0, delta=1e-9)


class ProbeVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.volume, cls.index = pal.palette_volume()
        cls.dec = pal.decompose(cls.volume)

    def test_every_sample_lands_on_its_declared_exposure(self) -> None:
        """Colour first, exposure second: the EV axis has to be exact or the
        recipe's rows are indexed by something other than what they claim."""
        err = np.abs(self.dec["ev"] - self.index.ev)
        self.assertLess(float(err.max()), 1e-5)

    def test_wheel_samples_land_on_their_declared_hue(self) -> None:
        w = self.index.kind == "wheel"
        dh = (self.dec["h_deg"][w] - self.index.hue_deg[w] + 180.0) % 360.0 - 180.0
        self.assertLess(float(np.abs(dh).max()), 0.01)

    def test_chroma_fraction_is_monotone_in_measured_chroma(self) -> None:
        w = self.index.kind == "wheel"
        means = [
            float(self.dec["C"][w & (self.index.chroma_frac == c)].mean())
            for c in pal.CHROMA_LEVELS
        ]
        self.assertTrue(all(b > a for a, b in zip(means, means[1:])), means)

    def test_neutral_ramp_carries_no_chroma(self) -> None:
        n = self.index.kind == "neutral"
        self.assertLess(float(self.dec["C"][n].max()), 1e-3)

    def test_volume_is_deterministic(self) -> None:
        again, _ = pal.palette_volume()
        np.testing.assert_array_equal(self.volume, again)


class AppearanceFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (FREEZE_DIR / "MANIFEST.json").is_file():
            raise unittest.SkipTest(
                "missing tests/appearance_freeze; run tools/regen_appearance_freeze.py"
            )
        cls._fast = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"
        cls.manifest = json.loads((FREEZE_DIR / "MANIFEST.json").read_text("utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fast is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = cls._fast

    def test_manifest_hashes_match(self) -> None:
        for name, digest in self.manifest["fixture_sha256"].items():
            path = FREEZE_DIR / name
            self.assertTrue(path.is_file(), f"missing {name}")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest,
                f"{name} drifted from the manifest",
            )

    def test_golden_tree_is_pinned(self) -> None:
        """`technical` must stay byte-identical through the appearance work,
        and the golden tree is where the wider pipeline's bytes live."""
        from tools.regen_appearance_freeze import golden_tree_digest

        self.assertEqual(golden_tree_digest(), self.manifest["golden_tree_sha256"])

    # One float32 ULP at unity. The probe is pinned to this rather than to
    # exact equality because the observe path is matrix-heavy and its last bit
    # is not portable: CI reproduced 23 of 2229 elements differing by at most
    # 2.4e-07 (exactly 2^-22) against this machine, on both Python 3.11 and
    # 3.12, while the full path matched exactly. The bound is four orders of
    # magnitude below any colour difference the appearance layer could make,
    # so it still fails on a real change — see the dE00 assertion below, which
    # states the same thing in units a reader can judge.
    PROBE_ATOL = 1e-6

    def test_probe_output_is_stable(self) -> None:
        from tools.film_palette_probe import render_probe
        from tools.regen_appearance_freeze import probe_path

        volume, _ = pal.palette_volume()
        for stock in self.manifest["probe_stocks"]:
            for mode in ("observe", "full"):
                with self.subTest(stock=stock, mode=mode):
                    stored = np.load(probe_path(stock, mode), allow_pickle=False)["mapped"]
                    live = render_probe(volume, stock, mode).astype(np.float32)
                    delta = float(np.max(np.abs(live - stored)))
                    self.assertLessEqual(
                        delta, self.PROBE_ATOL,
                        f"{stock}/{mode}: probe drifted (max_abs={delta:.3g})",
                    )
                    worst = float(
                        pal.delta_e00(
                            pal.rec2020_to_lab(stored.astype(np.float64)),
                            pal.rec2020_to_lab(live.astype(np.float64)),
                        ).max()
                    )
                    self.assertLess(
                        worst, 1e-3,
                        f"{stock}/{mode}: probe moved by {worst:.4f} dE00",
                    )

    def test_technical_render_is_byte_identical(self) -> None:
        from tools.regen_appearance_freeze import render_path, render_technical

        for scene_id in self.manifest["render_scenes"]:
            with self.subTest(scene=scene_id):
                stored = np.load(render_path(scene_id), allow_pickle=False)
                linear, u8 = render_technical(scene_id)
                np.testing.assert_array_equal(stored["u8"], u8)
                delta = float(
                    np.max(np.abs(linear - stored["linear"].astype(np.float32)))
                )
                self.assertLessEqual(delta, float(np.finfo(np.float16).eps) * 4)


class MeasuredWeaknessTests(unittest.TestCase):
    """Where the P0 exit gate lands: "full looks weak" as coordinates.

    These assertions describe the chain as it is today. They are meant to be
    inverted as the recipes land, not deleted — a plan that cannot say what it
    is fixing cannot say when it is done.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = FREEZE_DIR / "BASELINE.json"
        if not path.is_file():
            raise unittest.SkipTest("missing BASELINE.json")
        cls.base = json.loads(path.read_text("utf-8"))

    def test_baseline_recomputes(self) -> None:
        """Recompute the whole report rather than only reading it.

        A stored baseline nobody recomputes rots quietly: the numbers below
        would keep passing long after the chain they describe had moved. The
        tolerance is 1% relative because these are medians and percentiles
        over 743 samples sitting on top of the float32 noise measured above,
        not because the numbers are vague.
        """
        from tools.film_palette_probe import build_report

        live = build_report(tuple(self.base["stocks"]))

        def walk(stored, got, path=""):
            if isinstance(stored, dict):
                self.assertEqual(set(stored), set(got), f"key set changed at {path}")
                for k in stored:
                    walk(stored[k], got[k], f"{path}/{k}")
            elif isinstance(stored, list):
                self.assertEqual(len(stored), len(got), f"length changed at {path}")
                for i, (x, y) in enumerate(zip(stored, got)):
                    walk(x, y, f"{path}[{i}]")
            elif isinstance(stored, (int, float)) and not isinstance(stored, bool):
                if not np.isfinite(float(stored)):
                    self.assertFalse(np.isfinite(float(got)), f"NaN-ness at {path}")
                    return
                self.assertAlmostEqual(
                    float(got), float(stored),
                    delta=max(abs(float(stored)) * 1e-2, 1e-6),
                    msg=f"baseline drifted at {path}",
                )
            else:
                self.assertEqual(stored, got, f"baseline drifted at {path}")

        walk(self.base, live)

    def test_within_family_stocks_are_not_separated(self) -> None:
        """Portra 400 and Ektar 100 are two C-41 negatives with famously
        different palettes. Today the full chain puts them 0.7 dE00 apart —
        below the plan's own 2.0 floor for stock identity, and effectively a
        small hue rotation with no chroma or lightness difference at all."""
        pe = self.base["stock_identity"]["portra400__vs__ektar100"]["full"]
        self.assertLess(pe["delta_e00"]["overall"]["median"], 1.5)
        self.assertLess(abs(pe["log2_chroma_ratio"]["overall"]["median"]), 0.1)
        for region, stats in pe["by_region"].items():
            with self.subTest(region=region):
                self.assertLess(stats["delta_e00"]["median"], 1.5)

    def test_cross_family_separation_already_exists(self) -> None:
        """The complaint is not that the chain cannot separate anything: a
        negative against a reversal is already well past the 2.0 floor. The
        appearance layer's job is identity WITHIN a family, not more contrast
        between families."""
        for key in ("portra400__vs__velvia100", "velvia100__vs__vision3250d"):
            with self.subTest(pair=key):
                med = self.base["stock_identity"][key]["full"]["delta_e00"]["overall"]["median"]
                self.assertGreater(med, 3.0)

    def test_the_c41_negatives_lose_chroma_against_observe(self) -> None:
        """The numeric form of "full looks weak": on the C-41 negatives the
        full chain delivers roughly two thirds of observe's Oklab chroma. It
        is NOT true of the reversal or the cine negative, so a global
        saturation lift would be the wrong fix."""
        for stock in ("portra400", "ektar100"):
            with self.subTest(stock=stock):
                med = self.base["observe_vs_technical"][stock][
                    "log2_chroma_ratio"]["overall"]["median"]
                self.assertLess(med, -0.3)
        self.assertGreater(
            self.base["observe_vs_technical"]["velvia100"][
                "log2_chroma_ratio"]["overall"]["median"],
            0.0,
        )

    def test_full_uses_less_of_the_gamut_than_observe(self) -> None:
        for stock in ("portra400", "ektar100", "vision3250d"):
            with self.subTest(stock=stock):
                g = self.base["gamut_pressure"][stock]
                self.assertLess(
                    g["full"]["outside_fraction"], g["observe"]["outside_fraction"]
                )

    def test_the_two_modes_agree_in_the_highlights_and_differ_in_the_midtones(self) -> None:
        """Useful for aiming a recipe: whatever the appearance layer does, the
        place it has room to act is the midtone, not the shoulder."""
        for stock in ("portra400", "ektar100"):
            with self.subTest(stock=stock):
                by_ev = self.base["observe_vs_technical"][stock]["delta_e00"]["by_ev"]
                self.assertGreater(by_ev["+0"]["median"], 5.0)
                self.assertLess(by_ev["+6"]["median"], 1.5)


if __name__ == "__main__":
    unittest.main()
