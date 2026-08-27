# SPDX-License-Identifier: GPL-3.0-or-later
"""Route-C phase-2 runtime contracts: the chromaticity-field Stage A.

The dispatcher (film_v2_math.stage_a_log_exposure) routes CV-adopted stocks
through E = Y * 2^L(x,y) and everything else through the 3x3 observer. The
contracts pinned here are the ones the phase-2 design document promised:
exposure homogeneity survives the LUT, grey ramps stay on the observer's
logE axis (white re-anchor), degenerate inputs fall back to the exact 3x3,
and the non-adopted stock is bit-identical to the pure observer path.
"""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.film_develop import _load_v2
from dngscan.film_v2_math import (
    chroma_field_log_exposure,
    layer_log_exposure,
    stage_a_log_exposure,
)

ADOPTED = "portra400"


def _stock(name: str) -> dict:
    return _load_v2(name)[0]


def _record():
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parents[1] / "docs" / "chroma_field_cv.json").read_text()
    )["stocks"]


class DispatchTests(unittest.TestCase):
    def test_adoption_matches_the_cv_selection(self) -> None:
        """Every shipped stock dispatches exactly as the record's frequency
        rule decided (external review 2026-08-27, F3)."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
        try:
            import fit_chroma_field as fcf
        finally:
            sys.path.pop(0)
        record = _record()
        for name, entry in record.items():
            with self.subTest(stock=name):
                has_field = _stock(name)["chroma_delta_lut"] is not None
                self.assertEqual(has_field, fcf.adopts(entry))

    def test_a_3x3_stock_is_bit_identical_to_the_observer_path(self) -> None:
        """The dispatcher's 3x3 branch is the plain signed observer product.
        No stock currently retains the 3x3, so the branch is exercised on a
        stock dict with the field withdrawn — the same shape the loader
        builds for a retained stock."""
        stock = dict(_stock(ADOPTED))
        stock["chroma_delta_lut"] = None
        rng = np.random.default_rng(20260826)
        rgb = 0.18 * np.exp2(rng.uniform(-9.0, 5.0, (256, 3)))
        np.testing.assert_array_equal(
            stage_a_log_exposure(rgb, stock),
            layer_log_exposure(rgb, stock["observer"]),
        )

    def test_adopted_stock_actually_diverges_from_the_observer(self) -> None:
        stock = _stock(ADOPTED)
        rng = np.random.default_rng(20260826)
        rgb = 0.18 * np.exp2(rng.uniform(-4.0, 3.0, (256, 3)))
        field = stage_a_log_exposure(rgb, stock)
        obs = layer_log_exposure(rgb, stock["observer"])
        self.assertGreater(float(np.abs(field - obs).max()), 1e-3)


class FieldContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stock = _stock(ADOPTED)

    def test_exposure_homogeneity_through_the_lut(self) -> None:
        rng = np.random.default_rng(7)
        rgb = 0.18 * np.exp2(rng.uniform(-6.0, 4.0, (128, 3)))
        base = stage_a_log_exposure(rgb, self.stock)
        for k in (0.125, 2.0, 11.3):
            scaled = stage_a_log_exposure(rgb * k, self.stock)
            np.testing.assert_allclose(
                scaled - base,
                np.full_like(base, np.log10(k)),
                atol=1e-9,
                err_msg=f"E(k*x) != k*E(x) at k={k}",
            )

    def test_grey_ramp_stays_on_the_observer_logE_axis(self) -> None:
        """The white re-anchor promise: neutral inputs must land where the
        3x3 lands them (the characteristic tables' axis), to bilinear
        exactness — otherwise every neutral gate downstream would drift."""
        g = 0.18 * np.exp2(np.linspace(-8.0, 4.0, 49))
        grey = np.repeat(g[:, None], 3, axis=1)
        field = stage_a_log_exposure(grey, self.stock)
        obs = layer_log_exposure(grey, self.stock["observer"])
        np.testing.assert_allclose(field, obs, atol=5e-7)

    def test_negative_channels_take_the_signed_observer_exactly(self) -> None:
        """F4 (review 2026-08-27): a negative Rec.2020 component is a legal
        out-of-gamut coordinate, not negative light. Such pixels leave the
        field and evaluate the SIGNED 3x3 on their original values — no
        channel clamp anywhere; non-finite rows do the same."""
        a = self.stock["observer"]
        # Out-of-gamut rows built so every layer's SIGNED product stays
        # positive: for the negative channel c, |delta| is half the smallest
        # ratio (other weights)/(weight on c) over the layers that weight c.
        rows = []
        for c in range(3):
            others = np.ones(3); others[c] = 0.0
            ratios = [
                float(a[l] @ others) / float(a[l, c]) for l in range(3) if a[l, c] > 0
            ]
            if not ratios or min(ratios) <= 0.0:
                continue  # a pure-channel layer: no negative value keeps it positive
            row = np.ones(3); row[c] = -0.5 * min(ratios)
            rows.append(row)
        self.assertGreaterEqual(len(rows), 1, "observer has no feasible out-of-gamut row")
        n_neg = len(rows)
        rgb = np.asarray(
            rows + [[0.0, 0.0, 0.0], [np.nan, 0.2, 0.3]], dtype=np.float64
        )
        out = chroma_field_log_exposure(
            rgb,
            self.stock["chroma_delta_lut"],
            self.stock["chroma_domain"],
            self.stock["chroma_xyz_from_rec2020"],
            self.stock["observer"],
        )
        np.testing.assert_array_equal(out, layer_log_exposure(rgb, self.stock["observer"]))
        # ... and the signed product really is positive for the first two rows,
        # i.e. the old clamp would have answered for a different colour.
        e = rgb[:n_neg] @ self.stock["observer"].T
        self.assertTrue(bool(np.all(e > 0.0)), e)
        clamped = np.maximum(rgb[:n_neg], 1e-9)
        self.assertGreater(
            float(np.abs(layer_log_exposure(clamped, self.stock["observer"]) - out[:n_neg]).max()),
            1e-3,
        )

    def test_hand_over_at_the_gamut_edge_is_continuous(self) -> None:
        """Crossing a channel through zero must not step in LINEAR layer
        exposure — for EVERY shipped stock, over random boundary points on
        all three edges (third review 2026-08-27, F1: the previous probe was
        one stock and two channel ratios and missed 1.6-3.2%-of-mid-grey
        steps elsewhere). In the correction form the operator at the edge
        IS the analytic signed 3x3 on both sides, so the residual is the
        bilinear interpolation of a delta that is 0 at the edge."""
        rng = np.random.default_rng(20260827)
        record = _record()
        worst = 0.0
        for name in record:
            stock = _stock(name)
            if stock["chroma_delta_lut"] is None:
                continue
            mid = stock["observer"] @ np.full(3, 0.18)
            for _ in range(24):
                ch = int(rng.integers(0, 3))
                others = rng.uniform(0.02, 2.0, 3)
                inside = others.copy(); inside[ch] = 1e-6
                outside = others.copy(); outside[ch] = -1e-6
                a = 10.0 ** stage_a_log_exposure(inside[None, :], stock)[0] * mid
                b = 10.0 ** stage_a_log_exposure(outside[None, :], stock)[0] * mid
                step = float(np.abs(a - b).max() / mid.max())
                worst = max(worst, step)
                self.assertLess(
                    step, 1e-4,
                    f"{name}: linear step at channel {ch} zero crossing: {a} vs {b}",
                )

    def test_correction_is_zero_at_the_white_chromaticity(self) -> None:
        """delta(white) == 0 by construction: the anchored field and the
        anchored 3x3 agree there, so a near-neutral pixel is the 3x3 up to
        bilinear interpolation of a vanishing correction."""
        g = 0.18 * np.exp2(np.linspace(-6.0, 4.0, 11))
        tint = np.stack([g * 1.0000001, g, g * 0.9999999], axis=1)
        field = stage_a_log_exposure(tint, self.stock)
        obs = layer_log_exposure(tint, self.stock["observer"])
        # Fourth review (F3): the bake removes the bilinear residual at the
        # white chromaticity along the weight field. What remains at a 1e-7
        # tint is the correction's gradient times the tint plus float32 LUT
        # storage: measured 7.9e-9 log10 (2.6e-8 stop; was 3.5e-5 stop).
        np.testing.assert_allclose(field, obs, atol=5e-8)
        # ... and the stored LUT's bilinear sample AT the white chromaticity
        # is zero to float32 node precision.
        m = self.stock["chroma_xyz_from_rec2020"]
        lut = self.stock["chroma_delta_lut"]
        x0, x1, y0, y1 = self.stock["chroma_domain"]
        xyz = np.ones(3) @ m.T
        wx, wy = xyz[0] / xyz.sum(), xyz[1] / xyz.sum()
        n = lut.shape[0]
        fx = (wx - x0) / (x1 - x0) * (n - 1)
        fy = (wy - y0) / (y1 - y0) * (n - 1)
        ix, iy = min(int(fx), n - 2), min(int(fy), n - 2)
        ax, ay = fx - ix, fy - iy
        at_white = (
            lut[ix, iy] * (1 - ax) * (1 - ay) + lut[ix + 1, iy] * ax * (1 - ay)
            + lut[ix, iy + 1] * (1 - ax) * ay + lut[ix + 1, iy + 1] * ax * ay
        )
        self.assertLess(float(np.abs(at_white).max()), 1e-8)

    def test_runtime_chromaticities_cannot_leave_the_lut_domain(self) -> None:
        """Channel-clamped positive Rec.2020 keeps chromaticity inside the
        primary triangle, whose bounding box (plus pad) is the LUT domain —
        so saturated primaries still ride the LUT, not the fallback."""
        prim = np.eye(3) * 3.7 + 1e-9
        out = stage_a_log_exposure(prim, self.stock)
        self.assertTrue(bool(np.isfinite(out).all()))
        x0, x1, y0, y1 = self.stock["chroma_domain"]
        m = self.stock["chroma_xyz_from_rec2020"]
        xyz = np.maximum(prim, 1e-9) @ m.T
        s = xyz.sum(axis=1)
        self.assertTrue(bool(np.all((xyz[:, 0] / s >= x0) & (xyz[:, 0] / s <= x1))))
        self.assertTrue(bool(np.all((xyz[:, 1] / s >= y0) & (xyz[:, 1] / s <= y1))))


if __name__ == "__main__":
    unittest.main()
