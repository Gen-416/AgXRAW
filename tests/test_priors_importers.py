# SPDX-License-Identifier: GPL-3.0-or-later
"""Priors importer gates: JPTC PTC fit, P2P bulk conversion, fallback chain.

Three tiers feed dngscan.priors.find_priors: curated entries (hand-checked),
JPTC first-party PTC fits (dngscan/data/priors/jptc/), and the P2P bulk
table (dngscan/data/priors/p2p_bulk.json). These tests pin the conversion
math against hand-computed values and gate the ordering so a bulk entry can
never shadow a curated one.
"""
from __future__ import annotations

import math
import subprocess
import sys
import unittest
from pathlib import Path

from dngscan import priors

REPO = Path(__file__).parents[1]


class TestJptcImporter(unittest.TestCase):
    def test_self_test_synthetic_sensor(self):
        """The importer must recover a known synthetic sensor's parameters."""
        proc = subprocess.run(
            [sys.executable, str(REPO / "tools" / "import_jptc.py"), "--self-test"],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("self-test: PASS", proc.stdout)

    def test_a7m5_entries_load_as_tier2_priors(self):
        entries = priors._jptc_entries()
        hits = [e for e in entries if "ILCE-7M5" in e["model_equals"]]
        # Mechanical + electronic measurements coexist; the ELECTRONIC one is
        # preferred because the mechanical floor (~0.4 DN) is below the PTC's
        # resolution, so its read-noise curve is empty.
        self.assertEqual(len(hits), 2)
        e = hits[0]
        self.assertEqual(e["shutter"], "electronic")
        # PRNU-corrected primary at its converged fixed point (review R9:
        # iteration cap raised 3->16; A7M5 converges in 5 rounds at 4.64148);
        # fwc_e is the ADC code-saturation capacity (white-black)*gain
        self.assertAlmostEqual(e["unity_gain_ev"], 8.8588, places=3)
        self.assertAlmostEqual(e["fwc_e"], 73663, delta=10)
        self.assertGreater(e["fwc_e_uncertainty"], 0)
        self.assertEqual(e["quality"]["status"], "ok")
        self.assertEqual(e["quality"]["prnu_status"], "corrected")
        # Single-ISO entry: read noise extrapolates flat, PDR degrades to None.
        self.assertAlmostEqual(priors.read_noise_e(e, 100), 8.803, places=2)
        self.assertAlmostEqual(priors.read_noise_e(e, 6400), 8.803, places=2)
        self.assertIsNone(priors.pdr_ev(e, 100))
        # The mechanical entry's unresolved read noise degrades to None.
        self.assertIsNone(priors.read_noise_e(hits[1], 100))
        self.assertEqual(hits[1]["shutter"], "mechanical")

    def test_multi_measurement_preference(self):
        """A7RM6 has three single-ISO PTC entries and three collect sets in
        the JPTC tier: the collect entry with a full gain curve wins, and
        among those the mechanical-shutter set. Queried on the tier directly
        — in find_priors the curated A7RM6 entry shadows all of these."""
        hits = [e for e in priors._jptc_entries() if "ILCE-7RM6" in e["model_equals"]]
        self.assertEqual(len(hits), 6)
        e = hits[0]
        self.assertTrue(e["gain_log2iso_log2epd"])
        self.assertEqual(e["shutter"], "mechanical")
        self.assertIsNotNone(priors.read_noise_e(e, 100))

    def test_collect_gain_curve_cross_instrument(self):
        """The collect gain curve at ISO 640 must agree with the completely
        independent iso640 PTC ramp (two instruments, same sensor): the
        strongest internal-consistency pin the tier has."""
        hits = [e for e in priors._jptc_entries()
                if "ILCE-7RM6" in e["model_equals"] and e.get("gain_log2iso_log2epd")
                and e["shutter"] == "mechanical"]
        curve_gain = priors.gain_e_per_dn(hits[0], 640)
        ptc640 = [e for e in priors._jptc_entries()
                  if "ILCE-7RM6" in e["model_equals"] and e.get("measured_iso") == 640]
        ptc_gain = priors.gain_e_per_dn(ptc640[0], 640)
        self.assertLess(abs(curve_gain - ptc_gain) / ptc_gain, 0.06)

    def test_collect_entries_resolve_new_makes(self):
        for mk, md in [("Canon", "Canon EOS R5 Mark II"),
                       ("Canon", "Canon EOS R6 Mark II"),
                       ("Panasonic", "DC-S1RM2"),
                       ("Leica", "M11 Monochrom")]:
            e = priors.find_priors(mk, md)
            self.assertIsNotNone(e, md)
            self.assertIn("collect", e["source"])
            self.assertGreater(priors.gain_e_per_dn(e, 100), 0)
            self.assertIsNotNone(priors.read_noise_e(e, 100))

    def test_collect_whiteness_evidence_present(self):
        e = priors.find_priors("Canon", "Canon EOS R6 Mark II")
        w = dict((round(2 ** x), v) for x, v in e["noise_whiteness_h_log2iso"])
        # R6 II's base-ISO RAW carries visible spatial filtering (whiteness
        # well below 1) that fades by high ISO — the oracle this field exists
        # for; if this pin breaks, the derivation changed, not the camera.
        self.assertLess(w[100], 0.75)
        self.assertGreater(max(w.values()), 0.9)

    def test_gain_none_without_anchor(self):
        """Dark-only collect sets have no absolute gain: consumer -> None."""
        darkonly = [e for e in priors._jptc_entries()
                    if "s1r2-elec" in e["source"]]
        self.assertTrue(darkonly)
        self.assertIsNone(priors.gain_e_per_dn(darkonly[0], 100))

    def test_new_makes_resolve(self):
        for make, model in [
            ("NIKON CORPORATION", "NIKON Z 7"),
            ("Panasonic", "DC-G9M2"),
            ("Panasonic", "DC-S1M2"),
        ]:
            e = priors.find_priors(make, model)
            self.assertIsNotNone(e, model)
            self.assertIn("JPTC", e["id"])
            self.assertGreater(priors.gain_e_per_dn(e, 100), 0)


class TestP2pBulk(unittest.TestCase):
    def test_table_loads(self):
        entries = priors._bulk_entries()
        self.assertGreaterEqual(len(entries), 130)
        for e in entries:
            self.assertTrue(e["pdr_log2iso_ev"])
            self.assertTrue(e["read_noise_log2iso_log2e"])

    def test_a77v_pinned_conversion(self):
        """Hand-validated cross-check camera (vs the public P2P PDR chart)."""
        e = priors.find_priors("SONY", "SLT-A77V")
        self.assertIsNotNone(e)
        self.assertIn("p2p_bulk", e["source"])
        self.assertAlmostEqual(priors.pdr_ev(e, 100), 9.91, delta=0.02)
        self.assertAlmostEqual(priors.read_noise_e(e, 100), 4.04, delta=0.05)
        # High-ISO read noise converges to the table's published 2.7 e-.
        self.assertAlmostEqual(priors.read_noise_e(e, 6400), 2.70, delta=0.05)
        self.assertAlmostEqual(priors.gain_e_per_dn(e, 100), 1.192, delta=0.01)

    def test_sanity_ranges(self):
        for e in priors._bulk_entries():
            for _, y in e["read_noise_log2iso_log2e"]:
                self.assertTrue(0.05 <= 2.0 ** y <= 200.0, e["id"])
            for _, y in e["pdr_log2iso_ev"]:
                self.assertTrue(1.0 <= y <= 17.0, e["id"])


class TestFallbackOrdering(unittest.TestCase):
    def test_curated_shadows_bulk(self):
        """Sigma fp exists in both tiers; the curated entry must win."""
        e = priors.find_priors("SIGMA", "SIGMA FP")
        self.assertIn("dcg_switch_iso", e)  # curated-only annotation

    def test_curated_shadows_jptc(self):
        """A7M5 exists curated + JPTC; curated (with PDR curve) must win."""
        e = priors.find_priors("SONY", "ILCE-7M5")
        self.assertEqual(e["id"], "Sony ILCE-7M5 (A7 V)")
        self.assertIsNotNone(priors.pdr_ev(e, 100))

    def test_unknown_camera_none(self):
        self.assertIsNone(priors.find_priors("ACME", "FOOCAM 9000"))

    def test_a7m5_internal_consistency(self):
        """Regression pin for the corrected unity_gain_ev: the gain implied
        by unity gain at ISO 100 must reproduce fwc_e from the 14-bit range
        (the defect fixed 2026-08-24 was a 3x contradiction here)."""
        e = priors.find_priors("SONY", "ILCE-7M5")
        implied_fwc = priors.gain_e_per_dn(e, 100) * (16383 - 512)
        self.assertLess(abs(implied_fwc - e["fwc_e"]) / e["fwc_e"], 0.10)

    def test_a7rm6_internal_consistency(self):
        """Same defect class, fixed 2026-08-24: the curated unity_gain_ev
        inherited the chart-axis decoding error (P2P's x-axis is
        ISO = 3.125*2^x, read as 2^x) and contradicted fwc_e by ~3x; the
        corrected value must reproduce fwc_e from the 14-bit range and
        agree with the first-party JPTC ramps."""
        e = priors.find_priors("SONY", "ILCE-7RM6")
        self.assertNotIn("measured_iso", e)  # curated, not the JPTC tier
        implied_fwc = priors.gain_e_per_dn(e, 100) * (16383 - 512)
        self.assertLess(abs(implied_fwc - e["fwc_e"]) / e["fwc_e"], 0.10)
        jptc = [x for x in priors._jptc_entries() if "ILCE-7RM6" in x["model_equals"]]
        self.assertAlmostEqual(
            e["unity_gain_ev"], jptc[0]["unity_gain_ev"], delta=0.05
        )


class TestCuratedAxisAudit(unittest.TestCase):
    """Pins for the 2026-08-24 chart-axis audit.

    Root cause: the curated extraction decoded the P2P PDR/RN_e x-axis as
    log2(ISO) when the axis is actually ISO = 3.125 * 2^x (verified against
    the chart's rendered tick labels — position 20 is labelled 3276800 —
    and eight cameras' native ISO ranges lining up exactly once decoded).
    Every curve x and every chart-anchored unity_gain_ev sat
    log2(3.125) = 1.6439 EV low. Each pin below reproduces fwc_e from the
    unity gain at the camera's NATIVE base ISO over its 14-bit DN range —
    the invariant the old values violated by ~3x.
    """

    # id -> (make, model, native base ISO, black level)
    BASES = {
        "Sigma fp": ("SIGMA", "SIGMA FP", 100, 1024),
        "Sony ILCE-7M5 (A7 V)": ("SONY", "ILCE-7M5", 100, 512),
        "Sony ILCE-7SM3 (A7S III)": ("SONY", "ILCE-7SM3", 80, 512),
        "Sony ILCE-7RM6 (A7R VI)": ("SONY", "ILCE-7RM6", 100, 512),
        "Ricoh GR IV": ("RICOH IMAGING COMPANY, LTD.", "RICOH GR IV", 100, 0),
        "Nikon Z f": ("NIKON CORPORATION", "Z f", 100, 1008),
        "Fujifilm X100VI": ("FUJIFILM", "X100VI", 125, 1023),
        "Fujifilm X-E5": ("FUJIFILM", "X-E5", 125, 1023),
    }

    def _entry(self, prior_id):
        make, model, base, black = self.BASES[prior_id]
        e = priors.find_priors(make, model)
        self.assertIsNotNone(e, prior_id)
        self.assertEqual(e["id"], prior_id)
        return e, base, black

    def _assert_fwc_consistent(self, prior_id, tol=0.10):
        e, base, black = self._entry(prior_id)
        implied_fwc = priors.gain_e_per_dn(e, base) * (16383 - black)
        self.assertLess(abs(implied_fwc - e["fwc_e"]) / e["fwc_e"], tol, prior_id)

    def test_sigma_fp_internal_consistency(self):
        """7.29 -> 8.93. Anchors: fwc_e/14-bit range at base 100 (black 1024
        measured on owner DNGs); first-party pair-difference PTC on three
        owner ISO-100 frames (4.70-4.77 e-/DN vs 4.88 implied); pixel
        density 2.1 ke-/um^2. The DCG switch decodes to the known IMX410
        ISO 640 point."""
        self._assert_fwc_consistent("Sigma fp")
        e, _, _ = self._entry("Sigma fp")
        self.assertEqual(e["dcg_switch_iso"], 640)

    def test_a7sm3_internal_consistency(self):
        """8.51 -> 10.15 = log2(1139.3), P2P's DxOMark-derived unity ISO for
        this body — fully independent of the chart gain model — and
        consistent with fwc_e at base 80 within ~1%."""
        self._assert_fwc_consistent("Sony ILCE-7SM3 (A7S III)")
        e, _, _ = self._entry("Sony ILCE-7SM3 (A7S III)")
        self.assertAlmostEqual(e["unity_gain_ev"], math.log2(1139.3), delta=0.05)

    def test_gr_iv_internal_consistency(self):
        """5.73 -> 7.37 (fwc anchor at base 100; no DxO/JPTC data exists for
        this body; density 1.9 ke-/um^2 physical, old value implied 0.6)."""
        self._assert_fwc_consistent("Ricoh GR IV")

    def test_zf_internal_consistency(self):
        """7.42 -> 9.06 (fwc anchor at base 100, Nikon black 1008);
        cross-checked vs the same-sensor Nikon Z 6II DxO-derived unity ISO
        508.3 (ug_ev 8.99, delta 0.07)."""
        self._assert_fwc_consistent("Nikon Z f")
        e, _, _ = self._entry("Nikon Z f")
        self.assertAlmostEqual(e["unity_gain_ev"], math.log2(508.3), delta=0.12)

    def test_x100vi_internal_consistency(self):
        """5.98 -> 7.61 (fwc anchor at base 125, RAF black 1023 verified on a
        first-party file; owner ISO-250 RAF refutes the old value: its
        shadow noise sits 4-5x below the old shot-noise floor)."""
        self._assert_fwc_consistent("Fujifilm X100VI")

    def test_xe5_internal_consistency(self):
        """5.91 -> 7.54 (fwc anchor at base 125; same 40MP X-Trans HR
        platform as the X100VI)."""
        self._assert_fwc_consistent("Fujifilm X-E5")

    def test_curated_curves_start_at_native_base_iso(self):
        """Structural pin for the axis decode: after re-referencing, every
        curated curve's first point is the camera's native base ISO (the
        chart's solid-marker onset). A regression to raw chart x would miss
        by 1.64EV; a future mis-extraction that includes extended-low
        points would miss by the extension span."""
        for prior_id, (_, _, base, _) in self.BASES.items():
            with self.subTest(camera=prior_id):
                e, _, _ = self._entry(prior_id)
                x0 = e["pdr_log2iso_ev"][0][0]
                self.assertLess(abs(x0 - math.log2(base)), 0.02, prior_id)
                rn_x0 = e["read_noise_log2iso_log2e"][0][0]
                self.assertLess(abs(rn_x0 - math.log2(base)), 0.02, prior_id)

    def test_fuji_suspect_iso_decodes_to_extended_setting(self):
        """suspect_iso_min was stored as 2^chart_x; decoded it must land on
        the real hollow-marker onset — for both 40MP X-Trans bodies that is
        exactly the extended ISO 25600 setting."""
        for prior_id in ("Fujifilm X100VI", "Fujifilm X-E5"):
            e, _, _ = self._entry(prior_id)
            self.assertEqual(e["suspect_iso_min"], 25600, prior_id)


if __name__ == "__main__":
    unittest.main()


class TestReviewR9Contracts(unittest.TestCase):
    def test_strict_json_everywhere(self):
        """RFC 8259 has no NaN/Infinity literal; every priors data file must
        parse under a strict reader (review P2-2)."""
        import json as _json

        def _reject(tok):
            raise ValueError(f"non-RFC constant {tok}")

        for f in (REPO / "dngscan" / "data" / "priors").rglob("*.json"):
            with self.subTest(file=f.name):
                _json.loads(f.read_text(), parse_constant=_reject)

    def test_guidance_gates_low_confidence_priors(self):
        """high-residual or wide estimator spread must NOT build the
        electron-domain SNR confidence (review P1-1)."""
        from types import SimpleNamespace
        from dngscan.guidance import _has_sensor_snr_prior

        base = dict(gain_e_per_dn=2.0, prior_read_noise_e=3.0,
                    prior_quality_status=None, prior_model_spread=None)
        self.assertTrue(_has_sensor_snr_prior(SimpleNamespace(**base)))
        self.assertFalse(_has_sensor_snr_prior(
            SimpleNamespace(**{**base, "prior_quality_status": "high-residual"})))
        self.assertFalse(_has_sensor_snr_prior(
            SimpleNamespace(**{**base, "prior_model_spread": 0.154})))
        self.assertTrue(_has_sensor_snr_prior(
            SimpleNamespace(**{**base, "prior_model_spread": 0.04})))

    def test_r6ii_prior_is_gated_end_to_end(self):
        """The R6 II entry (rms 13.3%, spread 15.4%) must resolve with the
        quality evidence that makes the guidance gate reject it."""
        e = priors.find_priors("Canon", "Canon EOS R6 Mark II")
        self.assertEqual(e["quality"]["status"], "high-residual")
        self.assertGreater(e["quality"]["model_sensitivity"], 0.10)

    def test_bulk_entries_carry_mode_match(self):
        e = priors.find_priors("SONY", "SLT-A77V")
        self.assertEqual(e["mode_match"], "bulk-model-only")

    def test_collect_anchor_declares_effective_model(self):
        """Collect ramps lack top-of-ramp samples: the prnu correction is
        UNRESOLVED and the asset must say the effective path is plain linear
        (review P2-1), with the full estimator evidence present."""
        import json as _json
        c = _json.loads((REPO / "dngscan" / "data" / "priors" / "jptc_collect"
                         / "a7rm6-mech-20260818.json").read_text())
        a = c["ptc_anchor"]
        self.assertEqual(a["prnu_status"], "unresolved")
        self.assertIn("plain linear", a["fit_model_effective"])
        for k in ("gain_alternatives", "fit_relative_rms_alternatives",
                  "gain_estimator_spread_rel", "last_unsaturated_signal_e"):
            self.assertIn(k, a)

    def test_undeclared_clip_factor_is_unresolved(self):
        import json as _json
        g = _json.loads((REPO / "dngscan" / "data" / "priors" / "jptc_collect"
                         / "gfx100-mech-20260816.json").read_text())
        self.assertEqual(g["acquisition_contract"]["sigma_clip_correction"],
                         "unresolved")
        self.assertNotIn("undone via the declared", g["noise_aperture"])
