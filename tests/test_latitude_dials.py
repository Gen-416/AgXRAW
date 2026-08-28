# SPDX-License-Identifier: GPL-3.0-or-later
"""Latitude dials (owner decision 2026-08-14, taste-to-dial): the HDR
rho/white-margin/shoulder-start trio and the film inter-image beta.

Contract under test: None/auto is byte-identical to the registered policy
defaults; an explicit dial moves exactly its own quantity; out-of-domain and
mis-coupled payloads fail closed; the evidence gates stay un-bypassable."""
from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from dngscan.analysis import analyze
from dngscan.grade import RENDER_MODE
from dngscan.hdr_agx_plan import compile_hdr_agx_plan
from dngscan.raw_io import load_raw
from dngscan.tone import build_render_plan

PICTURES = Path.home() / "Pictures"
FRAME = PICTURES / "AgXRAW样张" / "_SDI0150.DNG"


@unittest.skipUnless(FRAME.is_file(), "sample frame unavailable")
class HdrDialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_raw(FRAME, scene_half_size=True)
        cls.analysis, _, _ = analyze(cls.bundle, margin=4, diagnostics=False)
        cls.plan = build_render_plan(cls.bundle, cls.analysis, RENDER_MODE, "p3")

    def _compile(self, **kw):
        return compile_hdr_agx_plan(self.plan, analysis=self.analysis, **kw)

    def test_auto_is_identical_to_policy(self) -> None:
        base = self._compile()
        dialed = self._compile(
            rho_base=None, white_margin_ev=None, shoulder_start_ev=None
        )
        self.assertEqual(base.tone, dialed.tone)
        self.assertEqual(base.color, dialed.color)

    def test_each_dial_moves_exactly_its_quantity(self) -> None:
        base = self._compile()
        rho0 = self._compile(rho_base=0.0)
        self.assertEqual(rho0.color.channel_separation, 0.0)
        self.assertEqual(rho0.tone, base.tone)  # rho never touches tone

        margin = self._compile(white_margin_ev=1.0)
        self.assertGreater(margin.tone.white_ev, base.tone.white_ev)
        self.assertEqual(margin.tone.white_margin_ev, 1.0)
        self.assertEqual(
            margin.color.channel_separation, base.color.channel_separation
        )

        knee = self._compile(shoulder_start_ev=1.0)
        self.assertEqual(knee.tone.shoulder_start_ev, 1.0)

    def test_rho_dial_cannot_bypass_evidence_gates(self) -> None:
        """The dial scales the BASE; the measured confidences still multiply,
        so a dialed rho can never exceed dial * (what evidence permits / 0.5)
        — in particular it stays capped at the dial value itself."""
        base = self._compile()
        dialed = self._compile(rho_base=1.0)
        self.assertLessEqual(
            dialed.color.channel_separation, 1.0 + 1e-9
        )
        # doubling the base at fixed evidence exactly doubles rho (or hits
        # the unaligned cap) — proportionality proves the gates still apply
        self.assertAlmostEqual(
            dialed.color.channel_separation,
            min(2.0 * base.color.channel_separation, 1.0),
            places=6,
        )

    def test_out_of_domain_dials_fail_closed(self) -> None:
        for kw in (
            {"rho_base": 1.2},
            {"rho_base": -0.1},
            {"white_margin_ev": 2.5},
            {"shoulder_start_ev": 4.0},
            {"rho_base": float("nan")},
        ):
            with self.subTest(kw=kw):
                with self.assertRaises(ValueError):
                    self._compile(**kw)


@unittest.skipUnless(FRAME.is_file(), "sample frame unavailable")
class InterimageBetaDialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_raw(FRAME, scene_half_size=True)
        cls.analysis, _, _ = analyze(cls.bundle, margin=4, diagnostics=False)

    def _plan(self, **kw):
        return build_render_plan(
            self.bundle, self.analysis, RENDER_MODE, "p3",
            film_curve="portra400", film_mode="full", **kw,
        )

    def test_custom_beta_lands_on_the_plan(self) -> None:
        declared = self._plan()
        self.assertAlmostEqual(declared.tone.film_interimage_beta, 0.62, places=6)
        dialed = self._plan(
            film_interimage="custom", film_interimage_beta_dial=0.9
        )
        self.assertEqual(dialed.tone.film_interimage, "custom")
        self.assertAlmostEqual(dialed.tone.film_interimage_beta, 0.9, places=6)

    def test_coupling_fails_closed(self) -> None:
        with self.assertRaises(ValueError):  # custom without the dial
            self._plan(film_interimage="custom")
        with self.assertRaises(ValueError):  # dial without custom
            self._plan(film_interimage_beta_dial=0.5)
        with self.assertRaises(ValueError):  # out of domain
            self._plan(film_interimage="custom", film_interimage_beta_dial=2.0)
        with self.assertRaises(ValueError):  # unknown mode
            self._plan(film_interimage="banana")


class ServiceDialParsingTests(unittest.TestCase):
    def test_hdr_dials_parse_and_guard_format(self) -> None:
        from dngscan.gui.service import parse_hdr_dials

        self.assertEqual(
            parse_hdr_dials({}, "ultrahdr"), (None, None, None)
        )
        self.assertEqual(
            parse_hdr_dials(
                {"hdrRho": 0.3, "hdrWhiteMargin": "0.5", "hdrShoulderStart": 0},
                "ultrahdr",
            ),
            (0.3, 0.5, 0.0),
        )
        # auto under SDR is fine; explicit values are refused
        self.assertEqual(parse_hdr_dials({"hdrRho": "auto"}, "sdr"), (None, None, None))
        with self.assertRaises(ValueError):
            parse_hdr_dials({"hdrRho": 0.3}, "sdr")
        with self.assertRaises(ValueError):
            parse_hdr_dials({"hdrRho": 1.5}, "ultrahdr")

    def test_film_interimage_beta_parse(self) -> None:
        from dngscan.gui.service import parse_film_params

        base = {"filmCurve": "portra400", "filmMode": "full"}
        out = parse_film_params(
            {**base, "filmInterimage": "custom", "filmInterimageBeta": 0.8}
        )
        self.assertEqual(out[21], 0.8)  # film_interimage_beta's position
        self.assertIsNone(parse_film_params(base)[21])
        with self.assertRaises(ValueError):  # beta without custom
            parse_film_params({**base, "filmInterimageBeta": 0.8})
        with self.assertRaises(ValueError):  # custom without beta
            parse_film_params({**base, "filmInterimage": "custom"})
        with self.assertRaises(ValueError):  # out of domain
            parse_film_params(
                {**base, "filmInterimage": "custom", "filmInterimageBeta": 1.6}
            )


class CliDialTests(unittest.TestCase):
    def test_sdr_rejects_explicit_hdr_dials(self) -> None:
        from dngscan.cli import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["x.dng", "--hdr-rho", "0.3"])
        args = parse_args(["x.dng", "--output-format", "ultrahdr", "--hdr-rho", "0.3"])
        self.assertEqual(args.hdr_rho, 0.3)
        args = parse_args(["x.dng"])
        self.assertIsNone(args.hdr_rho)
        self.assertIsNone(args.hdr_white_margin)
        self.assertIsNone(args.hdr_shoulder_start)
        with self.assertRaises(SystemExit):
            parse_args(["x.dng", "--output-format", "ultrahdr", "--hdr-rho", "1.5"])


if __name__ == "__main__":
    unittest.main()
