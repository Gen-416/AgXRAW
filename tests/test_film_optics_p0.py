# SPDX-License-Identifier: GPL-3.0-or-later
"""FILM_OPTICS_V2 phase P0 gates: the diagnosis has to be trustworthy first.

Two layers of gate live here, and the order matters.

The first layer checks the MEASUREMENTS against closed-form answers — a
white-noise field really does obey Selwyn's law, a Gaussian-blurred edge
really does have a known MTF50. Without these, every later "the halo is too
wide" is an opinion with a decimal point. One of them exists because the
measurement was wrong first time: a whole-row centroid edge fit is dragged off
the edge by a wide shallow halo, and reported a 12x resolution loss that was
not there.

The second layer pins the frozen legacy behaviour: the rendered bytes P1 must
not move, and the measured baseline P2-P4 are judged against.
"""
from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path

import numpy as np

from dngscan import film_optics_charts as charts
from dngscan import film_optics_diag as diag

ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = ROOT / "tests" / "optics_freeze"


def _gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian with an explicit kernel — no scipy dependency in
    the test path, and the taps are visible so the analytic comparison below
    is against something the reader can check.

    Reflect padding, not zero padding: a `mode="same"` convolution darkens a
    border band the width of the kernel, and those rows carry an edge of the
    right width but the wrong amplitude and asymmetric shape. Averaged into
    the ESF they shifted the measured MTF50 by 18% — a stimulus artefact that
    would have been read as a property of the operator under test.
    """
    radius = int(np.ceil(4.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    out = np.asarray(img, dtype=np.float64)
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect")
        out = np.apply_along_axis(
            lambda m: np.convolve(m, k, mode="valid"), axis, padded
        )
    return out


class DiagnosticPrimitiveTests(unittest.TestCase):
    def test_selwyn_slope_of_white_noise_is_minus_one(self) -> None:
        """Selwyn's law: sigma * sqrt(aperture area) is aperture-independent,
        so log sigma falls with slope -1. This is the yardstick that tells a
        granularity process from a blotch field."""
        rng = np.random.default_rng(20260809)
        field = rng.standard_normal((512, 512))
        self.assertAlmostEqual(diag.selwyn_slope(field), -1.0, delta=0.05)

    def test_selwyn_slope_rises_when_the_field_is_correlated(self) -> None:
        """A field correlated over several cells cannot average down like
        grain does; the slope must move decisively away from -1, which is how
        a too-coarse field is detected regardless of its amplitude."""
        rng = np.random.default_rng(7)
        blurred = _gaussian_blur(rng.standard_normal((512, 512)), 2.0)
        slope = diag.selwyn_slope(blurred)
        self.assertGreater(slope, -0.7)
        self.assertLess(slope, -0.1)

    def test_radial_psd_is_a_variance_density(self) -> None:
        rng = np.random.default_rng(3)
        field = rng.standard_normal((256, 256)) * 2.0
        _, psd = diag.radial_psd(field, bins=64)
        self.assertAlmostEqual(float(np.mean(psd)), float(field.var()), delta=0.15)

    def test_correlation_length_separates_white_from_blurred(self) -> None:
        rng = np.random.default_rng(11)
        white = rng.standard_normal((256, 256))
        self.assertLess(diag.correlation_length_cells(white), 1.0)
        blurred = _gaussian_blur(rng.standard_normal((256, 256)), 3.0)
        self.assertGreater(diag.correlation_length_cells(blurred), 2.0)

    def test_half_energy_radius_matches_the_analytic_gaussian(self) -> None:
        """For a 2-D Gaussian the radius holding half the r-weighted energy is
        sigma * sqrt(2 ln 2)."""
        h = w = 401
        yy, xx = np.mgrid[0:h, 0:w]
        for sigma in (4.0, 8.0, 16.0):
            spot = np.exp(-((yy - 200.0) ** 2 + (xx - 200.0) ** 2) / (2 * sigma ** 2))
            radii, prof, _ = diag.radial_profile(spot[:, :, None], (200.0, 200.0))
            got = diag.half_energy_radius(radii, prof[:, 0], baseline=0.0)
            self.assertAlmostEqual(
                got / (sigma * np.sqrt(2 * np.log(2))), 1.0, delta=0.03
            )

    def test_slanted_edge_mtf_matches_the_analytic_gaussian(self) -> None:
        """MTF of a Gaussian blur is exp(-2 pi^2 sigma^2 f^2), so
        MTF50 = sqrt(2 ln 2) / (2 pi sigma)."""
        for sigma in (2.0, 3.0):
            edge = _gaussian_blur(
                charts.edge_chart(400, 400, tilt_deg=5.0)[:, :, 0].astype(np.float64),
                sigma,
            )
            freq, mtf = diag.slanted_edge_mtf(edge, half_window_px=24.0)
            expect = np.sqrt(2 * np.log(2)) / (2 * np.pi * sigma)
            self.assertAlmostEqual(diag.mtf50(freq, mtf) / expect, 1.0, delta=0.06)

    def test_mtf_edge_fit_survives_a_wide_shallow_halo(self) -> None:
        """The regression this file exists for.

        Adding a broad, low-amplitude halo around the edge must not change the
        measured MTF50: the halo is not resolution loss. A whole-row gradient
        centroid fails this — hundreds of columns of tiny gradient outweigh the
        edge spike, the fitted edge line tilts, and the smeared ESF reports a
        12x loss that never happened.
        """
        edge = _gaussian_blur(
            charts.edge_chart(400, 400, tilt_deg=5.0)[:, :, 0].astype(np.float64), 2.0
        )
        clean = diag.mtf50(*diag.slanted_edge_mtf(edge, half_window_px=24.0))
        veiled = edge + 0.02 * _gaussian_blur(edge, 40.0)
        got = diag.mtf50(*diag.slanted_edge_mtf(veiled, half_window_px=24.0))
        self.assertAlmostEqual(got / clean, 1.0, delta=0.05)

    def test_chroma_luma_ratio_separates_luma_from_colour_speckle(self) -> None:
        rng = np.random.default_rng(5)
        n = rng.standard_normal((128, 128))
        luma_only = np.repeat(n[:, :, None], 3, axis=2)
        self.assertLess(diag.chroma_luma_ratio(luma_only), 1e-6)
        chroma_only = np.stack([n, -n, np.zeros_like(n)], axis=2)
        self.assertGreater(diag.chroma_luma_ratio(chroma_only), 2.0)

    def test_energy_ratio_reads_additive_and_conservative_apart(self) -> None:
        base = np.full((64, 64, 3), 0.5)
        additive = np.full((64, 64, 3), 0.01)
        np.testing.assert_allclose(
            diag.energy_ratio(additive, base), 0.02, rtol=1e-9
        )
        conservative = np.zeros((64, 64, 3))
        conservative[:32] = 0.01
        conservative[32:] = -0.01
        np.testing.assert_allclose(
            diag.energy_ratio(conservative, base), 0.0, atol=1e-12
        )


class ChartContractTests(unittest.TestCase):
    def test_emitter_geometry_is_declared_in_film_millimetres(self) -> None:
        """Same physical source, two output resolutions: the centre must land
        at the same MILLIMETRE, which is what makes a 35 mm / 16 mm comparison
        meaningful rather than a pixel coincidence."""
        for width, height in ((600, 400), (1200, 800)):
            _, (cy, cx) = charts.single_emitter(height, width, diameter_mm=1.0)
            mm_x = cx / charts.px_per_mm(width)
            self.assertAlmostEqual(mm_x, charts.GATE_35MM_W_MM / 2.0, delta=0.02)
            self.assertAlmostEqual(cy / height, 0.5, delta=1e-9)

    def test_emitter_energy_scales_with_area(self) -> None:
        """Supersampled coverage, not a binary mask: doubling the diameter
        must quadruple the emitted energy even at sub-pixel sizes."""
        bg = charts.ev(-6.0)
        energies = []
        for d in (0.05, 0.10, 0.20):
            img, _ = charts.single_emitter(
                256, 256, diameter_mm=d, exposure_ev=2.0, background_ev=-6.0
            )
            energies.append(float((img[:, :, 0] - bg).sum()))
        self.assertAlmostEqual(energies[1] / energies[0], 4.0, delta=0.25)
        self.assertAlmostEqual(energies[2] / energies[1], 4.0, delta=0.15)

    def test_density_wedge_patches_carry_the_declared_exposures(self) -> None:
        img, evs = charts.density_wedge(64, 210, steps=21)
        self.assertEqual(len(evs), 21)
        for i, e in enumerate(evs):
            patch = img[:, i * 10 + 4, 0]
            np.testing.assert_allclose(patch, charts.ev(float(e)), rtol=1e-5)

    def test_edge_chart_is_slanted_enough_to_measure(self) -> None:
        """A vertical edge is sampled at one sub-pixel phase per row and cannot
        be reconstructed above Nyquist; the tilt is a measurement requirement,
        not a decoration."""
        img = charts.edge_chart(256, 256, tilt_deg=5.0)[:, :, 0]
        peaks = np.argmax(np.abs(np.diff(img, axis=1)), axis=1)
        self.assertGreater(int(peaks.max() - peaks.min()), 8)


class LegacyFreezeGateTests(unittest.TestCase):
    """The frozen legacy behaviour. P1 rewrites structure, not maths."""

    @classmethod
    def setUpClass(cls) -> None:
        if not (FREEZE_DIR / "MANIFEST.json").is_file():
            raise unittest.SkipTest(
                "missing tests/optics_freeze; run tools/regen_optics_freeze.py"
            )
        cls._fast = os.environ.get("DNGSCAN_FAST")
        os.environ["DNGSCAN_FAST"] = "0"

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._fast is None:
            os.environ.pop("DNGSCAN_FAST", None)
        else:
            os.environ["DNGSCAN_FAST"] = cls._fast

    def test_manifest_hashes_match_the_stored_fixtures(self) -> None:
        manifest = json.loads((FREEZE_DIR / "MANIFEST.json").read_text("utf-8"))
        for stem, digest in manifest["fixture_sha256"].items():
            path = FREEZE_DIR / f"{stem}.npz"
            self.assertTrue(path.is_file(), f"missing fixture {stem}")
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), digest,
                f"{stem} bytes drifted from the manifest",
            )

    def test_legacy_optics_render_is_byte_identical(self) -> None:
        from tools.regen_optics_freeze import iter_cases, render_case

        cases = iter_cases()
        self.assertGreaterEqual(len(cases), 6, "freeze matrix collapsed")
        for case in cases:
            with self.subTest(case=case.stem):
                stored = np.load(case.path, allow_pickle=False)
                linear, u8 = render_case(case)
                np.testing.assert_array_equal(
                    stored["u8"], u8,
                    err_msg=f"{case.stem}: legacy optics output moved",
                )
                # u8 above is the exact gate. The linear plane is stored as
                # float16 to keep the fixtures small, so it is checked at that
                # storage precision — tightening it further would only pin
                # the rounding of the fixture, not the renderer.
                expected = np.asarray(stored["linear"], dtype=np.float32)
                delta = float(np.max(np.abs(linear - expected)))
                self.assertLessEqual(
                    delta, float(np.finfo(np.float16).eps) * 4,
                    f"{case.stem}: legacy optics linear drifted (max_abs={delta:.6g})",
                )


class MeasuredBaselineTests(unittest.TestCase):
    """The numbers P2-P4 are judged against.

    Pinned with a relative tolerance rather than exactly: these come out of
    FFTs and percentile reductions, and a byte-exact pin across platforms
    would only teach the next person to regenerate it.
    """

    @classmethod
    def setUpClass(cls) -> None:
        path = FREEZE_DIR / "BASELINE.json"
        if not path.is_file():
            raise unittest.SkipTest(
                "missing BASELINE.json; run tools/regen_optics_freeze.py"
            )
        cls.stored = json.loads(path.read_text("utf-8"))
        from tools.film_optics_report import build_report

        cls.live = build_report(cls.stored["stock"])

    def _walk(self, stored, live, path=""):
        if isinstance(stored, dict):
            self.assertEqual(set(stored), set(live), f"key set changed at {path}")
            for k in stored:
                self._walk(stored[k], live[k], f"{path}/{k}")
        elif isinstance(stored, list):
            self.assertEqual(len(stored), len(live), f"length changed at {path}")
            for i, (a, b) in enumerate(zip(stored, live)):
                self._walk(a, b, f"{path}[{i}]")
        elif isinstance(stored, (int, float)) and not isinstance(stored, bool):
            self.assertAlmostEqual(
                float(live), float(stored),
                delta=max(abs(float(stored)) * 5e-3, 1e-9),
                msg=f"baseline drifted at {path}",
            )
        else:
            self.assertEqual(stored, live, f"baseline drifted at {path}")

    def test_baseline_reproduces(self) -> None:
        self._walk(self.stored, self.live)

    def test_baseline_records_the_defects_it_was_written_for(self) -> None:
        """The P0 exit gate in one place: each complaint has a number.

        These assertions describe the CURRENT implementation. They are meant
        to be inverted, not deleted, as each phase lands — a plan that cannot
        say what it is fixing cannot say when it is done.
        """
        grain = self.live["grain"]["standard"]
        # INVERTED by P4 multi-band: the field used to be a single blotch
        # (FWHM > 40 um, Selwyn slope shallower than -0.7). The particle-
        # oracle-fitted band mixture brings the correlation length down to
        # the render pitch and the slope into the granularity regime the
        # dye-cloud Boolean model predicts for a 12 um representation.
        self.assertLess(grain["field_blob_fwhm_um"], 20.0)
        self.assertLess(grain["field_selwyn_slope"], -0.8)
        self.assertGreater(grain["field_selwyn_slope"], -1.05)
        # INVERTED by P4 (grain V2): the 48 um RMS used to sit an order of
        # magnitude above any datasheet stock (>40 x1000); the measured
        # sigma(D) kernel now lands the as-rendered figures inside the
        # 5207 chart's own window (~4-17 x1000 across channels).
        quotes = grain["rms_granularity_48um_at_span2"]
        self.assertLess(max(quotes), 20.0)
        self.assertGreater(min(quotes), 3.0)

        # Halation was fixed in P2, so its assertions are INVERTED here rather
        # than deleted: the baseline is where a phase says what it changed.
        halation = self.live["halation"]["standard"]
        self.assertLess(
            max(halation["half_energy_radius_mm"]), 0.20,
            "P2 brought the halo back to a physical radius",
        )
        white = halation["halo_inner_ratio"]
        blue = self.live["halation"]["blue_source"]["halo_inner_ratio"]
        self.assertGreater(
            abs(blue[1] / blue[0] - white[1] / white[0]), 0.1,
            "P2 gates per layer, so a blue source no longer returns the "
            "white source's halo",
        )
        self.assertGreater(
            white[1] / white[0],
            halation["halo_outer_ratio"][1] / halation["halo_outer_ratio"][0] + 0.1,
            "P2's component set makes the inner ring warmer than the outer",
        )

        # RESOLVED by deletion (P5e): the P0 numbers here recorded the
        # legacy operator's domain error (a display-threshold source gate
        # seeing <1 EV of overrange where the scene offered 6). P3 replaced
        # the operator with the scene-linear capture bloom and P5e deleted
        # the legacy code and asset; the baseline keeps a tombstone so the
        # section cannot silently vanish.
        self.assertEqual(
            self.live["bloom"]["source_gate"],
            "deleted_with_legacy_print_scatter",
        )

        # §11.1 CLOSED (ledger batch, 2026-08-14): the runtime honours the
        # bilinear/area resample contract, and the old "still-open 4.73" was
        # the METRIC aliasing the edge chart's single transition spike
        # against the 8-px modulus on a chart too small for the resample to
        # even run (ratio grew with ANY probe modulus; autocorrelation
        # showed no periodicity). The measurement now engages decimation for
        # real, excludes the edge's own neighbourhood, and reads both the
        # legacy moduli and the true bilinear knot pitch. Contract: no
        # modulus reads block-like, knots included (measured 0.78-1.03).
        block = self.live["bloom"]["pyramid_blockiness"]
        for key in ("step_2px_ratio", "step_4px_ratio", "step_8px_ratio",
                    "step_16px_ratio", "knot_aligned_ratio"):
            self.assertLess(block[key], 1.30, f"{key} reads block-like")
            self.assertGreater(block[key], 0.60, f"{key} implausibly low")


if __name__ == "__main__":
    unittest.main()
