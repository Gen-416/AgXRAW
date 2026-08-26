# SPDX-License-Identifier: GPL-3.0-or-later
"""Review batch 17 (full-code review) regression gates for the immediate
fixes: session token + Origin gate, upload quotas, finite-value validation,
cache identity hardening, export deadline, fused film HDR pair, tile-local
noise floor."""
from __future__ import annotations

import unittest

import numpy as np


class ServerSecurityTests(unittest.TestCase):
    def test_page_carries_token_and_api_fetch_wrapper(self) -> None:
        from dngscan.gui.page import render_page

        html = render_page("/tmp", session_token="tok-abc").decode()
        self.assertIn("tok-abc", html)
        self.assertIn("X-DngScan-Token", html)
        for needle in ('apiFetch("/list', 'apiFetch("/upload', "apiFetch(path"):
            self.assertIn(needle, html)


class FiniteValidationTests(unittest.TestCase):
    def test_nan_and_infinity_are_rejected(self) -> None:
        from dngscan.gui.service import _finite_number

        self.assertEqual(_finite_number("1.5", "x", -8, 8), 1.5)
        for bad in (float("nan"), float("inf"), "-Infinity", "NaN"):
            with self.assertRaises(ValueError):
                _finite_number(bad, "x", -8, 8)
        with self.assertRaises(ValueError):
            _finite_number(9.0, "x", -8, 8)

    def test_parse_job_uses_finite_validation_for_ev(self) -> None:
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service)
        self.assertIn('_finite_number(params.get("ev"', src)


class CacheIdentityTests(unittest.TestCase):
    def test_identity_carries_inode_and_header_hash(self) -> None:
        import tempfile
        from pathlib import Path

        from dngscan.gui.preview_cache import _evidence_cache_identity

        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "x.dng"
            f.write_bytes(b"A" * 70000)
            ident1 = _evidence_cache_identity(f)
            stat = f.stat()
            # replace IN PLACE with same size, restore mtime
            f.write_bytes(b"B" * 70000)
            import os

            os.utime(f, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            ident2 = _evidence_cache_identity(f)
            self.assertNotEqual(
                ident1, ident2,
                "same path/size/mtime with different bytes must change identity",
            )


class ExportDeadlineTests(unittest.TestCase):
    def test_timeout_source_pins_deadline_terminate_and_messages(self) -> None:
        """Source pin only: real timeout termination is not exercised here
        (it needs a hung child process); this keeps the deadline/terminate
        path and its user-facing messages from being deleted silently."""
        import inspect

        from dngscan.gui import service

        src = inspect.getsource(service.run_export_isolated)
        for needle in ("deadline", "terminate", "导出超时", "崩溃"):
            self.assertIn(needle, src)


class FusedFilmPairTests(unittest.TestCase):
    def test_pair_runs_the_film_chain_once(self) -> None:
        from unittest import mock

        from tests.golden_support import build_daylight_wide_dr
        from tests.test_film_v2_assets import _stock_files
        from dngscan import film_develop
        from dngscan.export import DEFAULT_HDR_HEADROOM_EV
        from dngscan.hdr_agx import render_ultrahdr_film_pair
        from dngscan.hdr_agx_plan import compile_hdr_agx_plan
        from dngscan.models import HdrDisplayTarget
        from dngscan.render import render_output_u8
        from dngscan.tone import build_render_plan

        stock = next(
            s for s in _stock_files()
            if s.startswith(("portra", "pro400h", "c200", "gold"))
        )
        scene = build_daylight_wide_dr()
        plan = build_render_plan(
            scene.bundle, scene.analysis, "agx", "p3",
            film_curve=stock, film_mode="full",
            film_crossover="datasheet", film_exposure_ev=1.5,
        )
        target = HdrDisplayTarget(
            peak_nits=100.0 * float(2.0 ** float(DEFAULT_HDR_HEADROOM_EV))
        )
        hdr_plan = compile_hdr_agx_plan(
            plan, target, analysis=scene.analysis,
            scene_decoder=str(scene.bundle.scene_decoder),
        )
        pixel_counts = []
        real = film_develop._apply_film_core_v2

        def spy(rgb, p, preset, spatial=None):
            pixel_counts.append(int(np.asarray(rgb).shape[0]))
            return real(rgb, p, preset, spatial)

        with mock.patch.object(film_develop, "_apply_film_core_v2", spy):
            base_u8, hdr_linear = render_ultrahdr_film_pair(
                scene.bundle, scene.analysis, plan, hdr_plan, "p3"
            )
        h, w = scene.bundle.scene_rec2020_render.shape[:2]
        total_px = sum(c for c in pixel_counts if c > 200)  # ignore tiny probes
        self.assertLessEqual(
            total_px, h * w + 1000,
            f"film chain processed {total_px} px for a {h*w}-px frame — "
            "the pair must walk the chain ONCE (review batch 17)",
        )
        # and the base still equals the standalone SDR export byte for byte
        standalone = render_output_u8(scene.bundle, scene.analysis, "p3", plan)
        np.testing.assert_array_equal(base_u8, standalone)


class NoiseFloorLocalityTests(unittest.TestCase):
    def test_tile_local_matches_full_frame_reference(self) -> None:
        from types import SimpleNamespace

        from dngscan.analysis import (
            estimate_noise_floor,
            estimate_raw_noise_floor,
            normalized_raw_signal,
        )

        rng = np.random.default_rng(1)
        h, w = 320, 480
        raw = (rng.normal(600, 30, (h, w))
               + rng.uniform(0, 12000, (h, w))).astype(np.uint16)
        colors = np.indices((h, w)).sum(axis=0) % 4
        b = SimpleNamespace(
            raw_image=raw, raw_colors=colors,
            black_levels=[512.0, 514.0, 512.0, 513.0],
        )
        fw = {0: 16000, 1: 15800, 2: 16000, 3: 15900}
        got = estimate_raw_noise_floor(b, fw)
        want = estimate_noise_floor(
            normalized_raw_signal(raw, colors, b.black_levels, fw)
        )
        self.assertAlmostEqual(got, want, places=12)


if __name__ == "__main__":
    unittest.main()
