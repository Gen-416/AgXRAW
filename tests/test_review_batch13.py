# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 13 regression gates.

1. Portrait / non-3:2 gate geometry: every pixel maps to real emulsion (no
   zero rows), the fitted region stays inside the physical gate, and rotated
   sampling shares the one field.
2. Analog-optics memory: an independent-process RSS gate pins the measured
   extra peak of a spatial render inside the budget tier.
3. Bloom pyramid: odd edges keep their bloom, tiny inputs never crash.
4. Editorial developer envelope: the baked shaper domains cover the maximum
   recipe perturbation for EVERY stock — zero silently clamped curve nodes.
5. Published assets are pinned by MANIFEST.json; the loader refuses
   non-finite tables and identity mismatches.
"""
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

# P1 §7.1: the operators take the specific asset they implement, so the tests
# pull the same declared assets the renderer compiles rather than a shared
# profile struct that no longer exists.
from dngscan.film_optics_assets import (  # noqa: E402
    DEFAULT_PRINT_OPTICS,
    DEFAULT_STOCK_OPTICS,
    load_print_optics,
    load_stock_optics,
)

_GRAIN = load_stock_optics(DEFAULT_STOCK_OPTICS).grain
_HALATION = load_stock_optics(DEFAULT_STOCK_OPTICS).halation
_SCATTER = load_print_optics(DEFAULT_PRINT_OPTICS).print_scatter

from tests.test_film_v2_assets import _stock_files

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "dngscan" / "data" / "film_v2"


class GateGeometryTests(unittest.TestCase):
    def test_fit_stays_inside_the_gate_in_both_orientations(self) -> None:
        from dngscan.film_optics import GATE_H_MM, GATE_W_MM, FilmGeometry

        for h, w in ((400, 600), (600, 400), (300, 400), (400, 300), (500, 500)):
            g = FilmGeometry.fit(h, w)
            x0, y0, w_mm, h_mm = g.region()
            gate_w, gate_h = (
                (GATE_H_MM, GATE_W_MM) if h > w else (GATE_W_MM, GATE_H_MM)
            )
            self.assertLessEqual(x0 + w_mm, gate_w + 1e-9, (h, w))
            self.assertLessEqual(y0 + h_mm, gate_h + 1e-9, (h, w))
            self.assertGreaterEqual(min(x0, y0), -1e-9, (h, w))
            # centered letterbox
            self.assertAlmostEqual(x0, gate_w - (x0 + w_mm), places=9)
            self.assertAlmostEqual(y0, gate_h - (y0 + h_mm), places=9)

    def test_no_zero_grain_rows_for_portrait_and_4x3(self) -> None:
        from dngscan.film_optics import (
            FilmGeometry,
            grain_field_for,
            integral_from_field,
            sample_field,
        )

        # sample_field takes an INTEGRAL image by contract (batch 19): the
        # caller declares which one it holds instead of it being sniffed.
        field = integral_from_field(grain_field_for(_GRAIN, 0))
        for h, w in ((600, 400), (300, 400), (400, 300)):
            got = sample_field(field, FilmGeometry.fit(h, w))
            row_rms = np.sqrt(np.mean(np.square(got, dtype=np.float64), axis=(1, 2)))
            self.assertGreater(
                float(row_rms.min()), 0.05,
                f"{h}x{w}: rows sampled a dead field (the review's measured "
                "failure: 333/600 portrait rows were zero)",
            )

    def test_rotated_band_split_matches_full_frame(self) -> None:
        from dngscan.film_develop import apply_film_core, prepare_film_spatial
        from types import SimpleNamespace

        stock = next(
            s for s in _stock_files()
            if s.startswith(("portra", "pro400h", "c200", "gold"))
        )
        plan = SimpleNamespace(
            curve_preset=stock, film_mode="full", film_crossover="datasheet",
            film_exposure_ev=0.0, film_print_timing="fixed",
            film_print_medium="", film_print_exposure_ev=0.0,
            color_head_y=0.0, color_head_m=0.0,
            film_development="measured_default",
            film_dev_contrast=0.0, film_dev_fog=0.0, film_dev_density=0.0,
            film_compression=0.0, film_compression_knee=2.0,
            film_highlight_density=0.0,
            film_grain=0.6, film_halation=0.5, film_bloom=0.4,
            film_optics_seed=0,
        )
        h, w = 72, 48  # portrait
        rng = np.random.default_rng(5)
        img = rng.uniform(0.02, 0.5, (h, w, 3)).astype(np.float32)
        img[30:36, 20:26] = 6.0
        flat = img.reshape(-1, 3)
        full = apply_film_core(flat, plan, spatial_shape=(h, w))
        from dngscan.film_optics import area_decimate, spread_grid_shape

        ctx = prepare_film_spatial(plan, h, w)
        dh, dw = spread_grid_shape(h, w)
        scene_dec = area_decimate(img, dh, dw)
        # P3 lifecycle: one scene pass, bloom before halation.
        if ctx.bloom > 0.0:
            ctx.begin_bloom_source()
            ctx.accumulate_bloom_source(flat, 0, h)
            ctx.finish_bloom_map(scene_dec)
        if ctx.halation > 0.0:
            ctx.finish_maps(scene_dec, plan, stock)
        out = np.empty_like(full)
        for y0 in range(0, h, 13):
            y1 = min(y0 + 13, h)
            out[y0 * w:y1 * w] = apply_film_core(
                flat[y0 * w:y1 * w], plan, spatial=(ctx, y0, y1)
            )
        np.testing.assert_allclose(out, full, atol=2e-6)


class BloomPyramidEdgeTests(unittest.TestCase):
    def test_odd_edges_keep_their_bloom_and_tiny_inputs_survive(self) -> None:
        from dngscan.film_optics import bloom_delta_map

        # 33x35: highlights at the four corners must all participate — the
        # conservative delta is NEGATIVE at each impulse core (it loses
        # energy) and positive in each neighbourhood; truncation used to
        # zero out the bottom-right corner entirely.
        img = np.zeros((33, 35, 3), dtype=np.float32)
        for y, x in ((0, 0), (0, 34), (32, 0), (32, 34)):
            img[y, x] = 1.0
        delta = bloom_delta_map(img, _SCATTER)
        cores = [float(delta[y, x].mean()) for y, x in
                 ((0, 0), (0, 34), (32, 0), (32, 34))]
        self.assertLess(max(cores), 0.0, "every corner core must shed energy")
        self.assertLess(
            min(cores) / min(max(cores, key=abs), -1e-9), 4.0,
            "corner participation must be comparable, not lost to truncation",
        )
        for y, x in ((2, 2), (2, 32), (30, 2), (30, 32)):
            self.assertGreater(
                float(delta[y, x].mean()), 0.0, "neighbourhood must gain"
            )
        # conservation on the map itself
        self.assertLess(
            float(np.abs(delta.sum(axis=(0, 1), dtype=np.float64)).max()), 1e-4
        )
        # tiny inputs: no crash, finite output
        for shape in ((5, 5), (1, 7), (3, 2)):
            tiny = np.ones(shape + (3,), dtype=np.float32)
            out = bloom_delta_map(tiny, _SCATTER)
            self.assertTrue(np.isfinite(out).all(), shape)


class EditorialDomainTests(unittest.TestCase):
    def test_every_stock_domain_covers_the_editorial_envelope(self) -> None:
        """The B1 / reversal-B2 shaper domains must contain the maximum
        declared recipe perturbation with ZERO clamped curve nodes (review
        batch 13 measured 35-47% silently clamped before the domain bake)."""
        from dngscan.film_develop import _load_v2
        from dngscan.film_v2_math import developer_perturbation

        for stock_name in _stock_files():
            stock, media = _load_v2(stock_name)
            for deltas in (
                dict(contrast_delta=0.5, fog_delta=0.3, color_density=0.5),
                dict(contrast_delta=-0.5, fog_delta=0.0, color_density=-0.5),
                dict(fog_delta=0.3, color_density=0.5),
            ):
                perturbed = developer_perturbation(
                    stock["char_le"], stock["char_amounts"], **deltas
                )
                if stock["reversal"]:
                    _, b2 = next(iter(media.values()))
                    lo, hi = b2["dye_lo"], b2["dye_hi"]
                else:
                    lo, hi = stock["lo"], stock["hi"]
                out = np.mean(
                    (perturbed < lo[None, :] - 1e-9)
                    | (perturbed > hi[None, :] + 1e-9)
                )
                self.assertEqual(
                    float(out), 0.0,
                    f"{stock_name} {deltas}: {out*100:.1f}% of perturbed "
                    "curve nodes fall outside the baked shaper domain",
                )


class AssetManifestTests(unittest.TestCase):
    def test_shipped_assets_match_the_pinned_manifest(self) -> None:
        manifest = json.loads((ASSET_DIR / "MANIFEST.json").read_text())
        files = sorted(p.name for p in ASSET_DIR.glob("*.npz"))
        self.assertEqual(files, sorted(manifest["files"]))
        self.assertEqual(len(files), int(manifest["count"]))
        for name, want in manifest["files"].items():
            got = hashlib.sha256((ASSET_DIR / name).read_bytes()).hexdigest()
            self.assertEqual(
                got, want,
                f"{name}: shipped bytes differ from the pinned manifest — "
                "rebakes must regenerate tools/gen_film_v2_manifest.py "
                "explicitly, never drift silently",
            )

    def test_loader_refuses_nan_and_identity_mismatch(self) -> None:
        import tempfile
        import unittest.mock as mock

        from dngscan import film_develop

        stock = next(s for s in _stock_files() if not s.startswith(
            ("velvia", "provia", "ektachrome", "kodachrome", "astia")
        ))
        with tempfile.TemporaryDirectory() as td:
            bad_dir = Path(td)
            for src in ASSET_DIR.glob("*.npz"):
                (bad_dir / src.name).write_bytes(src.read_bytes())
            # NaN into the print state's tau
            ps_path = next(bad_dir.glob(f"print__{stock}__*.npz"))
            with np.load(ps_path, allow_pickle=False) as z:
                payload = {k: np.asarray(z[k]) for k in z.files}
            payload["tau"] = payload["tau"].copy()
            payload["tau"][0, 0] = np.nan
            np.savez_compressed(ps_path.with_suffix(""), **payload)
            with mock.patch.object(film_develop, "_V2_DIR", bad_dir), \
                    mock.patch.object(film_develop, "_V2_CACHE", {}):
                with self.assertRaises(RuntimeError):
                    film_develop._load_v2(stock)
        with tempfile.TemporaryDirectory() as td:
            bad_dir = Path(td)
            for src in ASSET_DIR.glob("*.npz"):
                (bad_dir / src.name).write_bytes(src.read_bytes())
            # identity swap: another stock's asset under this stock's name
            others = [s for s in _stock_files() if s != stock]
            (bad_dir / f"{stock}.npz").write_bytes(
                (ASSET_DIR / f"{others[0]}.npz").read_bytes()
            )
            with mock.patch.object(film_develop, "_V2_DIR", bad_dir), \
                    mock.patch.object(film_develop, "_V2_CACHE", {}):
                with self.assertRaises(RuntimeError):
                    film_develop._load_v2(stock)


class OpticsMemoryTests(unittest.TestCase):
    def test_independent_process_rss_stays_inside_the_tier(self) -> None:
        """§9.3 budget as measured reality, not solver arithmetic: the
        RENDERER'S banded spatial pipeline in a fresh process may exceed the
        same banded colorimetric loop by at most the 512 MiB default tier
        plus allocator slack. (The full-frame oracle API deliberately
        materializes whole-image temporaries and is NOT the budgeted path.)"""
        import subprocess
        import sys

        script = r"""
import resource, sys
import numpy as np
from types import SimpleNamespace
from dngscan.film_develop import apply_film_core, prepare_film_spatial
from dngscan.film_optics import area_decimate_rows, spread_grid_shape
from dngscan.render import _optics_band_rows
plan = SimpleNamespace(
    curve_preset="portra400", film_mode="full", film_crossover="datasheet",
    film_exposure_ev=0.0, film_print_timing="fixed", film_print_medium="",
    film_print_exposure_ev=0.0, color_head_y=0.0, color_head_m=0.0,
    film_development="measured_default", film_dev_contrast=0.0,
    film_dev_fog=0.0, film_dev_density=0.0, film_compression=0.0,
    film_compression_knee=2.0, film_highlight_density=0.0,
    film_grain=GRAIN, film_halation=HAL, film_bloom=BLOOM,
    film_optics_seed=0,
)
h, w = 2000, 3000
rng = np.random.default_rng(0)
img = rng.uniform(0.02, 0.6, (h, w, 3)).astype(np.float32)
flat = img.reshape(-1, 3)
ctx = prepare_film_spatial(plan, h, w)
band_rows = _optics_band_rows(w)
if ctx is not None:
    dh, dw = spread_grid_shape(h, w)
    acc = np.zeros((dh, dw, 3), dtype=np.float64)
    # P3: ONE scene pass feeds both operators — the capture bloom's finest
    # rung at full resolution, the halation source from the decimated result.
    ctx.begin_bloom_source()
    for y0 in range(0, h, band_rows):
        y1 = min(y0 + band_rows, h)
        rows = flat[y0*w:y1*w]
        ctx.accumulate_bloom_source(rows, y0, y1)
        area_decimate_rows(rows.reshape(-1, w, 3), y0, h, w, dh, dw, acc)
    scene_dec = acc.astype(np.float32)
    del acc
    if ctx.bloom > 0.0:
        ctx.finish_bloom_map(scene_dec)
    if ctx.halation > 0.0:
        ctx.finish_maps(scene_dec, plan, "portra400")
    del scene_dec
halo = ctx.scatter_halo_rows() if ctx is not None else 0
for y0 in range(0, h, band_rows):
    y1 = min(y0 + band_rows, h)
    if ctx is not None:
        y0e, y1e = max(0, y0 - halo), min(h, y1 + halo)
        out = apply_film_core(
            flat[y0e*w:y1e*w], plan, spatial=(ctx, y0, y1, y0e, y1e),
        )
    else:
        out = apply_film_core(flat[y0*w:y1*w], plan, spatial=None)
    assert np.isfinite(out).all()
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
"""

        def run(grain, hal, bloom, tier=512):
            import os

            body = (script.replace("GRAIN", str(grain))
                    .replace("HAL", str(hal)).replace("BLOOM", str(bloom)))
            env = dict(os.environ, DNGSCAN_OPTICS_BUDGET_MIB=str(tier))
            proc = subprocess.run(
                [sys.executable, "-c", body], capture_output=True,
                text=True, cwd=str(ROOT), timeout=600, env=env,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            return int(proc.stdout.strip().splitlines()[-1])

        # EVERY advertised tier gets its own measured gate, with the
        # BASELINE AND SPATIAL RUNS SAMPLED AT THE SAME TIER (review batch
        # 19: comparing a 512-tier baseline against a 256-tier spatial run
        # made the gate meaningless — the tier also sizes the colorimetric
        # bands, so the baseline moves with it).
        from dngscan.render import OPTICS_BUDGET_TIERS_MIB

        for tier in OPTICS_BUDGET_TIERS_MIB:
            baseline = run(0.0, 0.0, 0.0, tier=tier)
            spatial = run(0.7, 0.5, 0.4, tier=tier)
            extra_mib = (spatial - baseline) / (1 << 20)
            self.assertLess(
                extra_mib, tier + 96,
                f"tier {tier}: spatial extra peak {extra_mib:.0f} MiB "
                "exceeds the advertised budget (+96 MiB allocator slack)",
            )


if __name__ == "__main__":
    unittest.main()
