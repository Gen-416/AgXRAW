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
        self.assertAlmostEqual(float(d["log2_saturation_ratio"][0]), 0.0, delta=1e-6)

        scaled = pal.oklab_to_rec2020(np.array([[0.7, 0.20, 0.04]]))
        d2 = pal.compare(base, scaled)
        self.assertAlmostEqual(float(d2["log2_saturation_ratio"][0]), 1.0, delta=1e-6)
        self.assertAlmostEqual(float(d2["d_hue_deg"][0]), 0.0, delta=0.02)

    def test_saturation_is_invariant_to_linear_exposure_scale(self) -> None:
        base = pal.oklab_to_rec2020(np.array([[0.65, 0.12, -0.04]]))
        d = pal.compare(base, base * 4.0)
        # Oklab is homogeneous of degree 1/3: +2 linear EV is +2/3 stop
        # absolute colourfulness, while C/L (purity) does not move.
        self.assertAlmostEqual(
            float(d["log2_colorfulness_ratio"][0]), 2.0 / 3.0, delta=1e-6
        )
        self.assertAlmostEqual(float(d["log2_saturation_ratio"][0]), 0.0, delta=1e-6)
        self.assertAlmostEqual(float(d["d_output_ev"][0]), 2.0, delta=1e-6)

    def test_de00_is_not_claimed_for_negative_xyz(self) -> None:
        valid = np.array([[0.18, 0.18, 0.18]])
        invalid = np.array([[-0.1, 0.2, 0.2]])
        d = pal.compare(valid, invalid)
        self.assertTrue(np.isnan(d["delta_e00"][0]))
        self.assertTrue(np.isfinite(d["delta_e_ok"][0]))

    def test_gamut_pressure_accepts_every_declared_gamut(self) -> None:
        volume, _ = pal.palette_volume()
        for gamut in ("srgb", "p3", "rec2020"):
            with self.subTest(gamut=gamut):
                got = pal.gamut_pressure(volume, gamut)
                self.assertGreaterEqual(got["outside_fraction"], 0.0)
                self.assertLessEqual(got["outside_fraction"], 1.0)

    def test_compare_of_identical_inputs_is_zero(self) -> None:
        volume, _ = pal.palette_volume()
        d = pal.compare(volume, volume)
        self.assertAlmostEqual(float(np.nanmax(np.abs(d["d_hue_deg"]))), 0.0, delta=1e-9)
        self.assertAlmostEqual(float(np.nanmax(d["delta_e00"])), 0.0, delta=1e-9)

    def test_report_freeze_distinguishes_nonfinite_values(self) -> None:
        from tools.regen_appearance_freeze import _reports_close

        self.assertTrue(_reports_close(float("nan"), float("nan")))
        self.assertTrue(_reports_close(float("inf"), float("inf")))
        self.assertTrue(_reports_close(float("-inf"), float("-inf")))
        self.assertFalse(_reports_close(float("nan"), float("inf")))
        self.assertFalse(_reports_close(float("inf"), float("-inf")))


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

    def test_gamut_ray_fraction_is_monotone_in_measured_saturation(self) -> None:
        w = self.index.kind == "wheel"
        means = [
            float(self.dec["S"][w & (self.index.chroma_frac == c)].mean())
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
            raise AssertionError(
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
        from tools.regen_appearance_freeze import expected_fixture_paths

        self.assertEqual(
            set(self.manifest["fixture_sha256"]),
            {path.name for path in expected_fixture_paths()},
        )
        for name, digest in self.manifest["fixture_sha256"].items():
            path = FREEZE_DIR / name
            self.assertTrue(path.is_file(), f"missing {name}")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest,
                f"{name} drifted from the manifest",
            )

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
                    # Respect the metric contract here too: pre-gamut negative
                    # XYZ has no defined CIEDE2000 meaning.
                    worst = float(np.nanmax(
                        pal.compare(stored.astype(np.float64), live.astype(np.float64))[
                            "delta_e00"
                        ]
                    ))
                    self.assertLess(
                        worst, 1e-3,
                        f"{stock}/{mode}: probe moved by {worst:.4f} dE00",
                    )

    def test_technical_render_is_byte_identical(self) -> None:
        from tools.regen_appearance_freeze import render_path, render_technical

        for scene_id in self.manifest["render_scenes"]:
            with self.subTest(scene=scene_id):
                stored = np.load(render_path(scene_id), allow_pickle=False)
                meta = json.loads(str(stored["meta"]))
                self.assertEqual(meta["neutralization"], "bounded")
                linear, u8 = render_technical(scene_id)
                np.testing.assert_array_equal(stored["u8"], u8)
                np.testing.assert_allclose(
                    linear, stored["linear"].astype(np.float32), rtol=0.0, atol=1e-6
                )

    def test_frozen_technical_is_the_user_visible_default(self) -> None:
        from tools.film_palette_probe import reference_plan

        plan, _bundle, _transform, _strength = reference_plan("portra400", "full")
        self.assertEqual(
            self.manifest["technical_definition"]["neutralization"], "bounded"
        )
        self.assertEqual(plan.tone.film_crossover, "off")
        self.assertIsNotNone(plan.film)
        # The plan records the canonical name since appearance P3; the
        # manifest keeps the historical word for the frozen bytes it pins.
        self.assertEqual(
            plan.film[2].neutralization_policy, "technical-neutral"
        )

    def test_regenerator_refuses_a_missing_source_scene(self) -> None:
        import contextlib
        import io
        from unittest import mock

        from tools import regen_appearance_freeze as regen

        with mock.patch.object(regen, "all_scenes", return_value={}):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(regen.regen(check=True), 1)


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
            raise AssertionError("missing tracked appearance BASELINE.json")
        cls.base = json.loads(path.read_text("utf-8"))

    def test_baseline_declares_the_measured_branch(self) -> None:
        self.assertEqual(
            self.base["technical_definition"],
            {
                "tone_core": "agx",
                "film_mode": "full",
                "neutralization": "bounded",
                "legacy_film_crossover": "off",
                "plan_output_gamut": "srgb",
                "measurement_space": "pre-gamut-fit-linear-rec2020",
            },
        )

    def test_baseline_recomputes(self) -> None:
        """Recompute the whole report rather than only reading it.

        A stored baseline nobody recomputes rots quietly: the numbers below
        would keep passing long after the chain they describe had moved. The
        tolerance follows the measured float32/BLAS portability bound. A 1%
        allowance hid changes of several tenths of dE00 in the largest tails,
        which is too loose for an instrument baseline.
        """
        from tools.film_palette_probe import build_report
        from tools.regen_appearance_freeze import REPORT_ATOL, REPORT_RTOL

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
                    delta=max(abs(float(stored)) * REPORT_RTOL, REPORT_ATOL),
                    msg=f"baseline drifted at {path}",
                )
            else:
                self.assertEqual(stored, got, f"baseline drifted at {path}")

        walk(self.base, live)

    def test_within_family_separation_arrived_with_mainline_a(self) -> None:
        """INVERTED 2026-08-10. P0 recorded Portra vs Ektar at 0.46 dE00 —
        a hue whisper with no chroma or lightness difference. The mainline A
        inter-image spread (Portra beta 0.50, Ektar 0.80) is the lever that
        bought the separation, so this baseline now records its presence."""
        pe = self.base["stock_identity"]["portra400__vs__ektar100"]["full"]
        self.assertGreater(pe["delta_e00"]["overall"]["median"], 1.2)
        self.assertGreater(
            pe["log2_saturation_ratio"]["overall"]["median"], 0.1,
            "Ektar must read more saturated than Portra",
        )

    def test_cross_family_separation_survives_mainline_a(self) -> None:
        """Cross-family distance narrowed when the C-41 stocks recovered
        their saturation (Portra moved toward Velvia: 3.54 -> 2.23), which is
        the correct direction — but it must stay above the plan's 2.0 floor
        or the recovery has erased a real identity instead of adding one."""
        for key in ("portra400__vs__velvia100", "velvia100__vs__vision3250d"):
            with self.subTest(pair=key):
                med = self.base["stock_identity"][key]["full"]["delta_e00"]["overall"]["median"]
                self.assertGreater(med, 2.0)

    def test_the_c41_saturation_loss_is_recovered(self) -> None:
        """INVERTED 2026-08-10. P0's headline: the C-41 print-through chain
        delivered 0.53-0.58x of observe's saturation (log2 -0.78/-0.92).
        After the inter-image term the gap closes to within a quarter stop —
        and it must not overshoot into cartoon either."""
        for stock, lo, hi in (
            ("portra400", -0.35, 0.1),
            ("ektar100", -0.2, 0.25),
        ):
            with self.subTest(stock=stock):
                med = self.base["observe_vs_technical"][stock][
                    "log2_saturation_ratio"]["overall"]["median"]
                self.assertGreater(med, lo)
                self.assertLess(med, hi)
        self.assertAlmostEqual(
            self.base["observe_vs_technical"]["velvia100"][
                "log2_saturation_ratio"]["overall"]["median"],
            -0.05, delta=0.1,
            msg="the reversal declares beta 0 and must not have moved",
        )

    def test_gamut_pressure_stays_inside_the_observe_envelope_for_c41(self) -> None:
        """P0 recorded full at a third of observe's out-of-gamut share; the
        recovery spends some of that headroom, which is fine — what it must
        not do is exceed the envelope observe already renders for the same
        stock. vision3250d is the declared exception: its observe pairing is
        deliberately muted (x1.2), so its full chain now carries more chroma
        than observe by design and is bounded absolutely instead."""
        for stock in ("portra400", "ektar100"):
            with self.subTest(stock=stock):
                g = self.base["gamut_pressure"][stock]["srgb"]
                self.assertLess(
                    g["full"]["outside_fraction"], g["observe"]["outside_fraction"]
                )
        self.assertLess(
            self.base["gamut_pressure"]["vision3250d"]["srgb"]["full"]["outside_fraction"],
            0.35,
        )

    def test_highlight_median_converges_but_saturated_tail_does_not(self) -> None:
        """Most +6 EV patches converge toward white, but the high-chroma tail
        remains far apart. A recipe may focus on the midtone; it must not treat
        the entire shoulder as already equivalent."""
        for stock in ("portra400", "ektar100"):
            with self.subTest(stock=stock):
                by_ev = self.base["observe_vs_technical"][stock]["delta_e00"]["by_ev"]
                # Floor re-pinned 5.0 -> 3.0 (route C, 2026-08-26): the
                # Stage A field moved full-mode colour closer to the
                # observe-side look at EV0 (portra 3.77 / ektar 4.18,
                # was >5 under the 3x3). The claim survives — midtone
                # observe and technical are still far from equivalent.
                self.assertGreater(by_ev["+0"]["median"], 3.0)
                self.assertLess(by_ev["+6"]["median"], 1.5)
                self.assertGreater(by_ev["+6"]["p95"], 10.0)


if __name__ == "__main__":
    unittest.main()
