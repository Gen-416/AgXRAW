# SPDX-License-Identifier: GPL-3.0-or-later
"""Optics V2 P5: scatter mixes (§5.1 emulsion / §6.2 formation) and the
halo row-band protocol.

Gate 13 finally has content: the operator's measured MTF must equal the
analytic transfer of the fitted kernel mix. Plus the §6.2 invariance
(uniform patches unchanged), the sub-pixel identity that keeps previews
untouched, seam-free banded rendering at a pitch where the kernels
resolve, and the no-silent-seams enforcement."""
from __future__ import annotations

import unittest

import numpy as np

from dngscan.film_optics import (
    apply_scatter_mix,
    scatter_halo_px,
)
from dngscan.film_optics_assets import (
    DEFAULT_PRINT_OPTICS,
    DEFAULT_STOCK_OPTICS,
    ScatterKernelAsset,
    load_print_optics,
    load_stock_optics,
)


def _kernels():
    stock = load_stock_optics(DEFAULT_STOCK_OPTICS).emulsion_scatter
    form = load_print_optics(DEFAULT_PRINT_OPTICS).formation_scatter
    return stock, form


class ScatterMixTests(unittest.TestCase):
    def test_assets_declare_measured_scatter(self) -> None:
        stock, form = _kernels()
        self.assertIsNotNone(stock)
        self.assertIsNotNone(form)
        # R1 item 7: a kernel FITTED from a measured chart is derived
        self.assertEqual(stock.provenance, "derived")
        self.assertEqual(stock.model, "bi_gaussian_v1")
        self.assertEqual(form.model, "gaussian_v1")

    def test_gate13_operator_mtf_matches_the_analytic_mix(self) -> None:
        stock, _ = _kernels()
        mm_per_px = 0.006  # 6 um pixels: the kernels resolve
        h, w = 64, 4096
        for ch in range(3):
            for cycles_per_mm in (20.0, 50.0):
                fx = cycles_per_mm * mm_per_px  # cycles per pixel
                x = np.arange(w, dtype=np.float64)
                sin = np.sin(2.0 * np.pi * fx * x)
                img = np.zeros((h, w, 3), dtype=np.float64)
                img[..., ch] = 1.0 + 0.5 * sin[None, :]
                out = apply_scatter_mix(img, mm_per_px, stock)
                # measured modulation via projection onto the input sinusoid
                got = out[h // 2, :, ch] - np.mean(out[h // 2, :, ch])
                mod = 2.0 * float(np.mean(got * sin)) / 0.5
                s = stock.s[ch]
                wgt = stock.w[ch]
                sg = stock.sigma_um[ch] * 1e-3
                ts = stock.tail_sigma_um[ch] * 1e-3
                g = np.exp(-2.0 * np.pi ** 2 * sg ** 2 * cycles_per_mm ** 2)
                # bi_gaussian_v1: this analytic form IS the executed one
                e = np.exp(-2.0 * np.pi ** 2 * ts ** 2 * cycles_per_mm ** 2)
                want = (1.0 - s) + s * ((1.0 - wgt) * g + wgt * e)
                self.assertLess(
                    abs(mod - want), 0.03,
                    f"ch{ch} f={cycles_per_mm}: got {mod:.4f} want {want:.4f}",
                )

    def test_uniform_patch_is_invariant(self) -> None:
        stock, form = _kernels()
        img = np.full((48, 200, 3), 0.63, dtype=np.float64)
        for kernel in (stock, form):
            out = apply_scatter_mix(img, 0.006, kernel)
            self.assertLess(
                float(np.max(np.abs(out - 0.63))) / 0.63, 1e-5,
                "normalized kernels must leave a uniform patch unchanged",
            )

    def test_inert_pitch_is_exact_identity(self) -> None:
        stock, form = _kernels()
        rng = np.random.default_rng(3)
        img = rng.uniform(0.1, 2.0, (32, 64, 3))
        # Review R1 item 2 changed the skip rule from kernel SCALE to mix
        # DEPTH (s*w*(1-MTF@Nyquist) < 1%). At 180px over the gate
        # (200 um pixels) every component's removable depth is
        # sub-percent, so the mix is exact identity and pays no halo; at
        # working pitches the components stay live even when sub-pixel —
        # dropping the R tail at 6 um/px cost 5.6 pp at 50 c/mm.
        out = apply_scatter_mix(img, 36.0 / 180.0, stock)
        np.testing.assert_allclose(out, img.astype(np.float32), rtol=2e-6)
        out = apply_scatter_mix(img, 36.0 / 180.0, form)
        np.testing.assert_allclose(out, img.astype(np.float32), rtol=2e-6)
        self.assertEqual(scatter_halo_px((stock, form), 36.0 / 180.0), 0)

    def test_tail_depth_is_present_at_working_pitch(self) -> None:
        # the R channel's exponential tail must NOT collapse to identity
        # at 6 um/px: its removal shifts 50 c/mm response by ~5.6 pp
        stock, _ = _kernels()
        from dngscan.film_optics import _scatter_components

        comps = _scatter_components(stock, 0, 0.006)  # R channel
        self.assertEqual(len(comps), 2, "core AND tail must both be live")

    def test_halo_rows_cover_the_largest_kernel(self) -> None:
        stock, form = _kernels()
        mm_per_px = 0.006
        r = scatter_halo_px((stock, form), mm_per_px)
        sigmas = [v * 1e-3 / mm_per_px
                  for v in stock.sigma_um + form.sigma_um]
        self.assertGreaterEqual(r, int(np.ceil(3 * max(sigmas))) - 1)
        self.assertLess(r, 40)

    def test_gaussian_v1_rejects_tail_fields(self) -> None:
        """Audit R11: the old payload smuggled lambda_um and actually
        tripped a DIFFERENT rule (w>0 with zero tail scale) — and a payload
        with ONLY the legacy lambda_um key loaded silently with the tail
        dropped. Two explicit contracts now: gaussian_v1 refuses tail
        fields, and unknown channel keys refuse loudly."""
        with self.assertRaisesRegex(Exception, "gaussian_v1 carries no tail"):
            ScatterKernelAsset.from_json({
                "provenance": "measured", "model": "gaussian_v1",
                "channels": {n: {"s": 0.3, "sigma_um": 8.0, "w": 0.2,
                                 "tail_sigma_um": 16.0}
                             for n in ("R", "G", "B")},
            }, "t")

    def test_unknown_channel_fields_refuse_loudly(self) -> None:
        with self.assertRaisesRegex(Exception, "unknown scatter fields"):
            ScatterKernelAsset.from_json({
                "provenance": "measured", "model": "gaussian_v1",
                "channels": {n: {"s": 0.3, "sigma_um": 8.0, "lambda_um": 2.0}
                             for n in ("R", "G", "B")},
            }, "t")


class HaloBandProtocolTests(unittest.TestCase):
    """Banded rendering with halo rows equals the full-frame oracle at a
    pitch where the kernels resolve; partial bands without their declared
    halo rows fail loudly instead of seaming silently."""

    H, W = 64, 4096  # 8.8 um/px over the 36 mm gate

    @staticmethod
    def _plan(**kw):
        from types import SimpleNamespace

        base = dict(
            curve_preset="portra400", film_mode="full",
            film_crossover="datasheet", film_exposure_ev=0.0,
            film_print_timing="fixed", film_print_medium="",
            film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default", film_dev_contrast=0.0,
            film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
            film_compression_knee=2.0, film_highlight_density=0.0,
            film_grain=0.4, film_halation=0.0, film_bloom=0.0,
            film_optics_seed=0,
        )
        base.update(kw)
        return SimpleNamespace(**base)

    def test_banded_with_halo_matches_full_frame(self) -> None:
        from dngscan.film_develop import apply_film_core, prepare_film_spatial

        h, w = self.H, self.W
        rng = np.random.default_rng(11)
        img = rng.uniform(0.05, 0.6, (h, w, 3)).astype(np.float32)
        img[20:28, 1000:1200] = 4.0
        flat = img.reshape(-1, 3)
        plan = self._plan()
        full = apply_film_core(flat, plan, spatial_shape=(h, w))
        ctx = prepare_film_spatial(plan, h, w)
        self.assertGreater(ctx.scatter_halo_rows(), 0,
                           "at 8.8 um/px the kernels must resolve")
        halo = ctx.scatter_halo_rows()
        out = np.empty_like(np.asarray(full))
        for band in (16, 32):
            for y0 in range(0, h, band):
                y1 = min(y0 + band, h)
                y0e, y1e = max(0, y0 - halo), min(h, y1 + halo)
                got = apply_film_core(
                    flat[y0e * w:y1e * w], plan,
                    spatial=(ctx, y0, y1, y0e, y1e),
                )
                out[y0 * w:y1 * w] = got
            np.testing.assert_allclose(
                out, np.asarray(full), rtol=2e-5, atol=2e-6,
                err_msg=f"band={band} must match the full-frame oracle",
            )

    def test_partial_band_without_halo_fails_loudly(self) -> None:
        from dngscan.film_develop import apply_film_core, prepare_film_spatial

        h, w = self.H, self.W
        flat = np.full((h * w, 3), 0.2, dtype=np.float32)
        plan = self._plan()
        ctx = prepare_film_spatial(plan, h, w)
        self.assertGreater(ctx.scatter_halo_rows(), 0)
        with self.assertRaises(ValueError):
            apply_film_core(flat[:16 * w], plan, spatial=(ctx, 0, 16))

    def test_media_scatter_off_disables_both_stages(self) -> None:
        """Review R1 item 4: film_media_scatter="off" must silence the halo
        demand and both develop stages — the media optics have their own
        enablement instead of riding whichever look slider engages the
        spatial context."""
        from dngscan.film_develop import apply_film_core, prepare_film_spatial

        h, w = self.H, self.W
        plan_on = self._plan()
        plan_off = self._plan()
        plan_off.film_media_scatter = "off"
        ctx_off = prepare_film_spatial(plan_off, h, w)
        self.assertEqual(ctx_off.scatter_halo_rows(), 0)
        # A halation render with scatter off differs from the same render
        # with scatter on wherever the kernels resolve — and NOT through
        # the halation operator itself (same seed, same maps).
        rng = np.random.default_rng(7)
        flat = rng.uniform(0.05, 0.6, size=(h * w, 3)).astype(np.float32)
        ctx_on = prepare_film_spatial(plan_on, h, w)
        self.assertGreater(ctx_on.scatter_halo_rows(), 0)
        on = apply_film_core(flat, plan_on, spatial_shape=(h, w))
        off = apply_film_core(flat, plan_off, spatial_shape=(h, w))
        self.assertGreater(float(np.max(np.abs(on - off))), 1e-7)


class MtfResidualBudgetTests(unittest.TestCase):
    """Review R1 item 2: the transfer budget MTF_measured / MTF_explicit is
    computed from the SHIPPED assets and bounded. The bound is the declared
    chart read-off uncertainty (±5%, ±8% low-frequency) plus the documented
    adjacency bump (a chemical edge effect a passive scatter mix cannot
    reproduce, ~3-8%): |log residual| <= log(1.15). A regression past that
    means the shipped kernel no longer explains the measurement it cites."""

    def test_residual_stays_inside_declared_budget(self) -> None:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        try:
            import film_optics_report as report_tool
        finally:
            sys.path.pop(0)
        res = report_tool.measure_mtf_residual()
        for name in ("5207_emulsion", "2383_formation"):
            with self.subTest(asset=name):
                self.assertIsNotNone(res[name])
                self.assertLess(
                    res[name]["max_abs_log_residual"], float(np.log(1.15))
                )


if __name__ == "__main__":
    unittest.main()
